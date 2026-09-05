"""
Train Predicted-Attribute HyperAdapt on PAPILA（实验 P，age_group 由 g 预测）
===========================================================================
把 `train_papila_hyperadapt.py` 的条件输入从**真值 age_group**（≤60/>60 = 0/1）换成**子群分类器
g 的 softmax 概率**（训练与测试两阶段都用）。g + HyperAdapt 视为**单一方法**；p̂ 由
`scripts/build_attr_predictions.py` 逐 fold 在 train 上拟合、in-sample 产出。

方案：`docs/predicted_attribute_hyperadapt_plan.md`。四臂：soft(主线)/hard(消融)/perm(路由对照)/
const(容量对照)。评估侧不变：分组键恒用**真值** sex/age_group。

PAPILA 特点（与 HAM 的差异）：
  · age_group 二值且全部合法，**无需过滤 -1**；
  · **CV 产物不放 cv5 子目录**（`output_dir` 不随 cv 变，与 train_papila_* 的既有规约一致），
    故 attr_pred 落在 `outputs/papila/attr_pred/fold{k}`；
  · n 极小（420 眼 / 210 患者，每折 test ≈84），读数功效很低，结论只作参考。

运行方式：经 sbatch 提交（见 slurm/p_pred_attr.sh）。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.papila_dataset import PAPILADataset
from src.datasets.soft_attr_wrapper import (
    ATTR_MODES, SoftAttrDataset, align_probs_by_key, materialize_probs,
)
from src.models.resnet18_hyperadapt import ResNet18HyperAdaptSoftAge
from src.training.attr_predictor import load_fold_predictions
from src.training.harness.hparam_grid import get_hparam_config, grid_size
from src.training.harness.run import seed_everything
from src.training.harness.train_loop import forward_soft_age, run_training, unpack_sex_age_soft
# transform 复用 GT 臂实现（单一事实来源，保证「仅属性来源」的单变量差异）
from src.training.train_papila_hyperadapt import build_transforms
from src.utils.papila_fairness import print_papila_fairness_report

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "papila_baseline.yaml"
OUTPUT_DIR = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/papila")

DATASET = "papila"
AXIS = "age"
KEY_COL = "image_path"
NUM_AGE = 2

METHOD_BY_MODE: dict[str, str] = {
    "soft": "hyperadapt_pred",
    "hard": "hyperadapt_predhard",
    "perm": "hyperadapt_predperm",
    "const": "hyperadapt_predconst",
}

UNPACK_FN = unpack_sex_age_soft
FORWARD_FN = forward_soft_age


def load_config(path: Path) -> dict:
    """读取 YAML 训练配置。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def MODEL_FACTORY(cfg: dict):                       # noqa: N802（与 smoke 约定的名字对齐）
    """按 config 造 pred-attr 模型（单 age 轴，2 类）。"""
    return ResNet18HyperAdaptSoftAge(
        num_classes=1, num_age=NUM_AGE, pretrained=cfg["model"]["pretrained"])


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Train Predicted-Attribute HyperAdapt on PAPILA（实验 P）")
    parser.add_argument("--attr_mode", choices=ATTR_MODES, default="soft",
                        help="条件输入模式：soft(主线)/hard(消融)/perm(路由对照)/const(容量对照)。")
    parser.add_argument("--config_index", type=int, default=5,
                        help=f"超参网格序号 0–{grid_size() - 1}；缺省 5=lr3e-04_wd1e-03，"
                             "即 PAPILA GT-HyperAdapt 的 Pareto 选定配置。")
    parser.add_argument("--seed", type=int, default=None, help="随机种子；CV 时由 slurm 传 42+fold。")
    parser.add_argument("--batch_size", type=int, default=None, help="覆盖 config 的 batch_size。")
    parser.add_argument("--cv", type=int, default=5, help="K 折患者级 GroupKFold（实验 P 用 5）。")
    parser.add_argument("--fold", type=int, default=None, help="--cv>0 时的折号 k ∈ [0,K)。")
    parser.add_argument("--swad", action="store_true", help="额外派生 SWAD 权重平均模型。")
    return parser.parse_args()


def resolve_paths(cfg: dict, cv: int, fold: int | None) -> tuple[Path, Path, Path]:
    """
    解析 (split_dir, output_dir, attr_pred_dir)。

    PAPILA 的 output_dir **不随 cv 变**（沿用 train_papila_* 的既有规约），
    因此 attr_pred 也落在 `outputs/papila/attr_pred/fold{k}`。

    Args:
        cfg : 数据集 yaml。
        cv  : 折数。
        fold: 折号。

    Returns:
        (split_dir, output_dir, attr_pred_dir)。

    Raises:
        ValueError: cv>=2 但 fold 非法。
    """
    base = REPO_ROOT / cfg["data"]["split_dir"]
    if cv and cv >= 2:
        if fold is None or not (0 <= fold < cv):
            raise ValueError(f"--cv={cv} 需配合合法 --fold ∈ [0,{cv})")
        return base / f"cv{cv}" / f"fold{fold}", OUTPUT_DIR, OUTPUT_DIR / "attr_pred" / f"fold{fold}"
    return base, OUTPUT_DIR, OUTPUT_DIR / "attr_pred"


def build_soft_dataset(
    split_dir: Path, attr_pred_dir: Path, split_name: str, transform,
    attr_mode: str, prior: np.ndarray | None, seed: int,
) -> tuple[SoftAttrDataset, np.ndarray]:
    """
    构建某 split 的 SoftAttrDataset：base dataset + 按 image_path join 的 p̂。

    Args:
        split_dir / attr_pred_dir / split_name / transform: 见同名参数。
        attr_mode: 条件输入模式。
        prior    : const 模式用的先验向量。
        seed     : perm 模式的置换种子。

    Returns:
        (dataset, 校准后的 soft 概率矩阵)。

    Raises:
        ValueError: 预测文件的真值 age_group 与 split 不一致。
    """
    csv_path = split_dir / f"{split_name}.csv"
    base = PAPILADataset(csv_path, transform=transform)
    keys = pd.read_csv(csv_path)[KEY_COL].astype(str).to_numpy()

    pred_keys, pred_probs, pred_gt = load_fold_predictions(
        attr_pred_dir / f"{split_name}.npz", axis=AXIS)
    probs = align_probs_by_key(keys, pred_keys, pred_probs)
    gt_aligned = align_probs_by_key(keys, pred_keys, pred_gt.reshape(-1, 1)).ravel().astype(np.int64)
    if not np.array_equal(gt_aligned, np.asarray(base.age_groups, dtype=np.int64)):
        raise ValueError(f"{split_name}: 属性预测文件的真值 age_group 与 split 不一致")

    perm_seed = seed * 100 + {"train": 0, "val": 1, "test": 2}[split_name]
    materialized = materialize_probs(probs, attr_mode, prior=prior, seed=perm_seed)
    return SoftAttrDataset(base, materialized), probs


def get_dataloaders(
    cfg: dict, split_dir: Path, attr_pred_dir: Path,
    train_transform, eval_transform, attr_mode: str, seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建三个 split 的 DataLoader（条件输入按 attr_mode 物化）。"""
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["dataloader"]["num_workers"]
    pin_memory = cfg["dataloader"]["pin_memory"]

    _, train_probs = build_soft_dataset(
        split_dir, attr_pred_dir, "train", eval_transform, "soft", None, seed)
    prior = train_probs.mean(axis=0)

    sets = {}
    for name, transform in (("train", train_transform), ("val", eval_transform), ("test", eval_transform)):
        sets[name], probs = build_soft_dataset(
            split_dir, attr_pred_dir, name, transform, attr_mode, prior, seed)
        entropy = float((-probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1).mean())
        agreement = float((probs.argmax(axis=1) == np.asarray(sets[name].base.age_groups)).mean())
        print(f"  {name:5s}: n={len(sets[name]):>5,}  p̂ 平均熵={entropy:.3f}  "
              f"argmax 与真值一致率={agreement:.3f}")
    print(f"  条件输入模式={attr_mode}"
          + (f"  prior={np.round(prior, 3).tolist()}" if attr_mode == "const" else ""))

    return (
        DataLoader(sets["train"], batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(sets["val"], batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(sets["test"], batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=pin_memory),
    )


def _fairness_report(y_true, y_score, attrs) -> None:
    """PAPILA 公平性报告回调：分组键恒用**真值** sex / age_group。"""
    print_papila_fairness_report(
        y_true=y_true, y_score=y_score, sex=attrs["sex"], age_group=attrs["age"], threshold=0.0,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(CONFIG_PATH)
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    seed = args.seed if args.seed is not None else cfg["training"]["seeds"][0]
    hparam = get_hparam_config(args.config_index)
    split_dir, output_dir, attr_pred_dir = resolve_paths(cfg, args.cv, args.fold)
    method = METHOD_BY_MODE[args.attr_mode]

    seed_everything(seed)
    print(f"Loading PAPILA (pred-attr HyperAdapt, mode={args.attr_mode}, method={method})  "
          f"split_dir={split_dir.parent.name}/{split_dir.name}  attr_pred={attr_pred_dir.name}...")
    train_transform, eval_transform = build_transforms(cfg)
    train_loader, val_loader, test_loader = get_dataloaders(
        cfg, split_dir, attr_pred_dir, train_transform, eval_transform, args.attr_mode, seed)

    run_training(
        dataset=DATASET, method=method, output_dir=output_dir, cfg=cfg, hparam=hparam, seed=seed,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        build_model=lambda: MODEL_FACTORY(cfg),
        unpack_fn=UNPACK_FN, forward_fn=FORWARD_FN,
        fairness_report_fn=_fairness_report,
        swad=args.swad,
    )


if __name__ == "__main__":
    main()
