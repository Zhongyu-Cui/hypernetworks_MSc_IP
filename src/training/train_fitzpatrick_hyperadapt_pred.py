"""
Train Predicted-Attribute HyperAdapt on Fitzpatrick17k（实验 P，skin 由 g 预测）
==============================================================================
把 `train_fitzpatrick_hyperadapt.py` 的条件输入从**真值 skin 索引**换成**子群分类器 g 的
softmax 概率 p̂**（训练与测试两阶段都用 p̂）。g + HyperAdapt 被定义为**单一方法**：g 已由
`scripts/build_attr_predictions.py` 在该 fold 的 train 上拟合并对三个 split 出好 p̂
（in-sample，无交叉拟合），本脚本只负责按样本唯一键 md5hash 把 p̂ join 进 dataset。

方案：`docs/predicted_attribute_hyperadapt_plan.md`。四种条件输入模式（--attr_mode）对应四个臂：

  soft  → method=hyperadapt_pred       主线：g 的校准概率
  hard  → method=hyperadapt_predhard   消融：argmax one-hot
  perm  → method=hyperadapt_predperm   对照：p̂ 行内置换（保边际、破对应）⇒ 分离路由 vs 容量
  const → method=hyperadapt_predconst  对照：train 集先验向量（纯容量上界）

评估侧完全不变：worst-group / 公平性分组键**恒用真值 skin**，p̂ 只作模型输入。

⚠️ 与 GT 臂等 batch（协议：重型 HN 显存对照必须等 batch），超参默认复用 GT-HyperAdapt 选定配置。
运行方式：重型任务，经 sbatch 提交（见 slurm/p_pred_attr.sh）。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.fitzpatrick_dataset import FitzpatrickDataset
from src.datasets.soft_attr_wrapper import (
    ATTR_MODES, SoftAttrDataset, align_probs_by_key, materialize_probs,
)
from src.models.resnet18_hyperadapt import ResNet18HyperAdaptSoftSkin
from src.training.attr_predictor import load_fold_predictions
from src.training.harness.hparam_grid import get_hparam_config, grid_size
from src.training.harness.run import seed_everything
from src.training.harness.train_loop import forward_soft_skin, run_training, unpack_skin_soft
from src.utils.fitzpatrick_fairness import print_fitzpatrick_fairness_report
from src.paths import OUTPUTS_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "fitzpatrick_baseline.yaml"
OUTPUT_DIR = OUTPUTS_DIR / "fitzpatrick"

DATASET = "fitzpatrick"
AXIS = "skin"
KEY_COL = "md5hash"
NUM_SKIN = 6

# 条件输入模式 -> method 名（与 GT 臂 hyperadapt 严格隔离，互不覆盖）
METHOD_BY_MODE: dict[str, str] = {
    "soft": "hyperadapt_pred",
    "hard": "hyperadapt_predhard",
    "perm": "hyperadapt_predperm",
    "const": "hyperadapt_predconst",
}


def load_config(path: Path) -> dict:
    """读取 YAML 训练配置。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Train Predicted-Attribute HyperAdapt on Fitzpatrick17k（实验 P）")
    parser.add_argument("--attr_mode", choices=ATTR_MODES, default="soft",
                        help="条件输入模式：soft(主线)/hard(消融)/perm(路由对照)/const(容量对照)。")
    parser.add_argument("--config_index", type=int, default=5,
                        help=f"超参网格序号 0–{grid_size() - 1}；缺省 5=lr3e-04_wd1e-03，"
                             "即 Fitzpatrick GT-HyperAdapt 的 Pareto 选定配置（单变量对照）。")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子；CV 时由 slurm 脚本传 42+fold。")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="覆盖 config 的 batch_size（须与 GT 臂等 batch）。")
    parser.add_argument("--cv", type=int, default=5,
                        help="K 折 CV（实验 P 默认 5，与 GT 臂 cv5 产物同址）。")
    parser.add_argument("--fold", type=int, default=None, help="--cv>0 时的折号 k ∈ [0,K)。")
    parser.add_argument("--swad", action="store_true", help="额外派生 SWAD 权重平均模型。")
    return parser.parse_args()


def resolve_paths(cfg: dict, cv: int, fold: int | None) -> tuple[Path, Path, Path]:
    """
    解析 (split_dir, output_dir, attr_pred_dir)。

    Args:
        cfg : 数据集 yaml。
        cv  : 折数（0 表示单-split，实验 P 不用）。
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
        return (base / f"cv{cv}" / f"fold{fold}",
                OUTPUT_DIR / f"cv{cv}",
                OUTPUT_DIR / f"cv{cv}" / "attr_pred" / f"fold{fold}")
    return base, OUTPUT_DIR, OUTPUT_DIR / "attr_pred"


def build_transforms(cfg: dict) -> tuple[transforms.Compose, transforms.Compose]:
    """构建训练 / 评估 transform（与 GT 臂逐项一致：224 原生，翻转+旋转，无 Resize）。"""
    t_cfg = cfg["transforms"]
    norm_mean, norm_std = t_cfg["normalize"]["mean"], t_cfg["normalize"]["std"]

    train_ops: list = []
    if t_cfg["train"].get("horizontal_flip", False):
        train_ops.append(transforms.RandomHorizontalFlip())
    train_ops.append(transforms.RandomRotation(degrees=t_cfg["train"]["rotation_degrees"]))
    train_ops += [transforms.ToTensor(), transforms.Normalize(mean=norm_mean, std=norm_std)]

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])
    return transforms.Compose(train_ops), eval_transform


def build_soft_dataset(
    split_dir: Path, attr_pred_dir: Path, split_name: str, transform,
    attr_mode: str, prior: np.ndarray | None, seed: int,
) -> tuple[SoftAttrDataset, np.ndarray]:
    """
    构建某个 split 的 SoftAttrDataset：base dataset + 按 md5hash join 的 p̂。

    Args:
        split_dir    : cv{K}/fold{k} 的 split 目录。
        attr_pred_dir: 该 fold 的 attr_pred 目录。
        split_name   : "train"/"val"/"test"。
        transform    : 该 split 的 transform。
        attr_mode    : 条件输入模式。
        prior        : const 模式用的先验向量（由 train 的 p̂ 均值给出）。
        seed         : perm 模式的置换种子。

    Returns:
        (dataset, 校准后的原始 soft 概率矩阵)——后者用于导出 prior 与打印诊断。

    Raises:
        ValueError: 预测文件与 split 的真值属性不一致（fold/split 配对错误的兜底断言）。
    """
    csv_path = split_dir / f"{split_name}.csv"
    base = FitzpatrickDataset(csv_path, transform=transform)
    keys = pd.read_csv(csv_path)[KEY_COL].astype(str).to_numpy()

    pred_keys, pred_probs, pred_gt = load_fold_predictions(
        attr_pred_dir / f"{split_name}.npz", axis=AXIS)
    probs = align_probs_by_key(keys, pred_keys, pred_probs)

    # 兜底断言：预测文件里记录的真值属性必须与 split CSV 一致（防 fold 配错、口径漂移）
    gt_aligned = align_probs_by_key(keys, pred_keys, pred_gt.reshape(-1, 1)).ravel().astype(np.int64)
    if not np.array_equal(gt_aligned, np.asarray(base.skins, dtype=np.int64)):
        raise ValueError(f"{split_name}: 属性预测文件的真值 {AXIS} 与 split CSV 不一致")

    # perm 的种子按 (run seed, split) 区分，避免三个 split 用同一置换、也保证可复现
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

    # const 模式的先验向量取自 **train** 的 p̂ 均值（与生成脚本 metrics.json 记录的一致）
    _, train_probs = build_soft_dataset(
        split_dir, attr_pred_dir, "train", eval_transform, "soft", None, seed)
    prior = train_probs.mean(axis=0)

    sets = {}
    for name, transform in (("train", train_transform), ("val", eval_transform), ("test", eval_transform)):
        sets[name], probs = build_soft_dataset(
            split_dir, attr_pred_dir, name, transform, attr_mode, prior, seed)
        entropy = float((-probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1).mean())
        agreement = float((probs.argmax(axis=1) == np.asarray(sets[name].base.skins)).mean())
        print(f"  {name:5s}: n={len(sets[name]):>6,}  p̂ 平均熵={entropy:.3f}  "
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
    """Fitzpatrick 公平性报告回调：分组键恒用**真值** skin（p̂ 只作模型输入）。"""
    print_fitzpatrick_fairness_report(
        y_true=y_true, y_score=y_score, skin=attrs["skin"], threshold=0.0,
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
    print(f"Loading Fitzpatrick17k (pred-attr HyperAdapt, mode={args.attr_mode}, method={method})  "
          f"split_dir={split_dir.parent.name}/{split_dir.name}  attr_pred={attr_pred_dir.name}...")
    train_transform, eval_transform = build_transforms(cfg)
    train_loader, val_loader, test_loader = get_dataloaders(
        cfg, split_dir, attr_pred_dir, train_transform, eval_transform, args.attr_mode, seed)

    run_training(
        dataset=DATASET, method=method, output_dir=output_dir, cfg=cfg, hparam=hparam, seed=seed,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        build_model=lambda: ResNet18HyperAdaptSoftSkin(
            num_classes=1, num_skin=NUM_SKIN, pretrained=cfg["model"]["pretrained"]),
        unpack_fn=unpack_skin_soft, forward_fn=forward_soft_skin,
        fairness_report_fn=_fairness_report,
        swad=args.swad,
    )


if __name__ == "__main__":
    main()
