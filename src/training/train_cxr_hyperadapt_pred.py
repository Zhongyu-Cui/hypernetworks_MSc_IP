"""
Train Predicted-Attribute HyperAdapt on MIMIC-CXR / CheXpert（实验 P，三轴由 g 预测）
====================================================================================
把 `train_mimic_hyperadapt.py` / `train_chexpert_hyperadapt.py` 的条件输入从**真值
(sex, race, age_group) 索引**换成**子群分类器 g 的三份 softmax 概率**（训练与测试两阶段都用）。
g + HyperAdapt 视为**单一方法**；p̂ 由 `scripts/build_attr_predictions.py` 逐 fold 在 train 上
拟合、in-sample 产出。

两个 CXR 库的差异只有 config / split / 输出目录 / 选定超参，故合并为一个脚本，
用 `--dataset {mimic,chexpert}` 切换（`DATASET_SPECS` 是唯一差异来源）。

方案：`docs/predicted_attribute_hyperadapt_plan.md`；Fitzpatrick 首轮结果见
`docs/predicted_attribute_hyperadapt_results.md`。四臂：soft(主线)/hard(消融)/perm(路由对照)/
const(容量对照)。评估侧不变：分组键恒用**真值** sex/race/age。

⚠️ 三份概率的顺序固定为 (sex, race, age)，与 `SoftPatientEmbedding.forward` 及
`unpack_sex_race_age_soft` 的解包顺序严格一致。
⚠️ 与 GT 臂等 batch（128）、同超参（--config_index 缺省按库取 GT-HyperAdapt 的 Pareto 选定值）。

运行方式：重型任务，经 sbatch 提交（见 slurm/p_pred_attr.sh，需额外传 DS_ARG）。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.mimic_cxr_dataset import MIMICCXRDataset
from src.datasets.soft_attr_wrapper import (
    ATTR_MODES, SoftAttrDataset, align_probs_by_key, materialize_probs,
)
from src.models.resnet18_hyperadapt import ResNet18HyperAdaptSoftPatient
from src.training.attr_predictor import load_fold_predictions
from src.training.harness.hparam_grid import get_hparam_config, grid_size
from src.training.harness.run import seed_everything
from src.training.harness.train_loop import (
    forward_soft_patient, run_training, unpack_sex_race_age_soft,
)
from src.utils.mimic_fairness import print_mimic_fairness_report
from src.paths import OUTPUTS_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = OUTPUTS_DIR

# 三个敏感轴的固定顺序（= SoftPatientEmbedding 的输入顺序）
AXES = ("sex", "race", "age")
KEY_COL = "image_path"      # CXR split CSV 无单图 id；image_path 唯一（已校验）

# 每库差异：config / 输出目录 / GT-HyperAdapt 的 Pareto 选定超参序号
DATASET_SPECS: dict[str, dict] = {
    "mimic": {
        "dataset": "mimic",
        "config": REPO_ROOT / "configs" / "mimic_cxr_baseline.yaml",
        "output_dir": OUTPUTS_ROOT / "mimic_cxr",
        # MIMIC GT-HyperAdapt 选定 lr1e-04_wd1e-03 → 网格序号 3
        "config_index": 3,
    },
    "chexpert": {
        "dataset": "chexpert",
        "config": REPO_ROOT / "configs" / "chexpert_baseline.yaml",
        "output_dir": OUTPUTS_ROOT / "chexpert_cxr",
        # CheXpert GT-HyperAdapt 选定 lr3e-04_wd1e-04 → 网格序号 4
        "config_index": 4,
    },
}

METHOD_BY_MODE: dict[str, str] = {
    "soft": "hyperadapt_pred",
    "hard": "hyperadapt_predhard",
    "perm": "hyperadapt_predperm",
    "const": "hyperadapt_predconst",
}

UNPACK_FN = unpack_sex_race_age_soft
FORWARD_FN = forward_soft_patient


def load_config(path: Path) -> dict:
    """读取 YAML 训练配置。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_model(cfg: dict, spec: dict):
    """按 config 造 pred-attr 模型（三轴软条件通路，各轴 2 类）。"""
    return ResNet18HyperAdaptSoftPatient(
        num_classes=1, num_sex=2, num_race=2, num_age=2,
        pretrained=cfg["model"]["pretrained"])


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Train Predicted-Attribute HyperAdapt on MIMIC-CXR / CheXpert（实验 P）")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_SPECS))
    parser.add_argument("--attr_mode", choices=ATTR_MODES, default="soft",
                        help="条件输入模式：soft(主线)/hard(消融)/perm(路由对照)/const(容量对照)。")
    parser.add_argument("--config_index", type=int, default=None,
                        help=f"超参网格序号 0–{grid_size() - 1}；缺省按库取 GT-HyperAdapt 的选定配置。")
    parser.add_argument("--seed", type=int, default=None, help="随机种子；CV 时由 slurm 传 42+fold。")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="覆盖 config 的 batch_size（须与 GT 臂等 batch，CXR=128）。")
    parser.add_argument("--cv", type=int, default=5, help="K 折患者级划分（实验 P 用 5）。")
    parser.add_argument("--fold", type=int, default=None, help="--cv>0 时的折号 k ∈ [0,K)。")
    parser.add_argument("--swad", action="store_true", help="额外派生 SWAD 权重平均模型。")
    return parser.parse_args()


def resolve_paths(spec: dict, cfg: dict, cv: int, fold: int | None) -> tuple[Path, Path, Path]:
    """
    解析 (split_dir, output_dir, attr_pred_dir)。

    Args:
        spec: 数据集配置。
        cfg : 该库 yaml。
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
        return (base / f"cv{cv}" / f"fold{fold}",
                spec["output_dir"] / f"cv{cv}",
                spec["output_dir"] / f"cv{cv}" / "attr_pred" / f"fold{fold}")
    return base, spec["output_dir"], spec["output_dir"] / "attr_pred"


def build_transforms(cfg: dict) -> tuple[transforms.Compose, transforms.Compose]:
    """构建训练 / 评估 transform（与 CXR baseline 一致：Grayscale(3)、不水平翻转）。"""
    t_cfg = cfg["transforms"]
    norm_mean, norm_std = t_cfg["normalize"]["mean"], t_cfg["normalize"]["std"]
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(t_cfg["train"]["resize"]),
        transforms.RandomRotation(degrees=t_cfg["train"]["rotation_degrees"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])
    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(t_cfg["eval"]["resize"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])
    return train_transform, eval_transform


def build_soft_dataset(
    cfg: dict, split_dir: Path, attr_pred_dir: Path, split_name: str, transform,
    attr_mode: str, priors: dict[str, np.ndarray] | None, seed: int,
) -> tuple[SoftAttrDataset, dict[str, np.ndarray]]:
    """
    构建某 split 的 SoftAttrDataset：base dataset + 按 image_path join 的三轴 p̂。

    Args:
        cfg          : 该库 yaml（提供 image_size / age_threshold）。
        split_dir    : cv{K}/fold{k} 的 split 目录。
        attr_pred_dir: 该 fold 的 attr_pred 目录。
        split_name   : "train"/"val"/"test"。
        transform    : 该 split 的 transform。
        attr_mode    : 条件输入模式。
        priors       : const 模式用的逐轴先验向量。
        seed         : perm 模式的置换种子。

    Returns:
        (dataset, {轴 -> 校准后的 soft 概率矩阵})。

    Raises:
        ValueError: 某轴的预测真值与 split 的真值不一致。
    """
    csv_path = split_dir / f"{split_name}.csv"
    base = MIMICCXRDataset(
        csv_path, transform=transform, image_size=cfg["data"]["image_size"],
        age_threshold=cfg["attributes"]["age"]["age_threshold"])
    keys = pd.read_csv(csv_path)[KEY_COL].astype(str).to_numpy()

    gt_by_axis = {"sex": np.asarray(base.sexes, dtype=np.int64),
                  "race": np.asarray(base.races, dtype=np.int64),
                  "age": np.asarray(base.age_groups, dtype=np.int64)}

    probs_by_axis: dict[str, np.ndarray] = {}
    materialized: list[np.ndarray] = []
    perm_seed = seed * 100 + {"train": 0, "val": 1, "test": 2}[split_name]
    for axis_index, axis in enumerate(AXES):
        pred_keys, pred_probs, pred_gt = load_fold_predictions(
            attr_pred_dir / f"{split_name}.npz", axis=axis)
        probs = align_probs_by_key(keys, pred_keys, pred_probs)
        gt_aligned = align_probs_by_key(
            keys, pred_keys, pred_gt.reshape(-1, 1)).ravel().astype(np.int64)
        if not np.array_equal(gt_aligned, gt_by_axis[axis]):
            raise ValueError(f"{split_name}/{axis}: 属性预测文件的真值与 split 不一致")
        probs_by_axis[axis] = probs
        # 各轴用不同的置换种子，避免三轴同步置换（那会保留轴间相关、削弱对照的破坏力）
        materialized.append(materialize_probs(
            probs, attr_mode,
            prior=None if priors is None else priors[axis],
            seed=perm_seed + 10 * (axis_index + 1)))
    return SoftAttrDataset(base, materialized), probs_by_axis


def get_dataloaders(
    spec: dict, cfg: dict, split_dir: Path, attr_pred_dir: Path,
    train_transform, eval_transform, attr_mode: str, seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建三个 split 的 DataLoader（三轴条件输入按 attr_mode 物化）。"""
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["dataloader"]["num_workers"]
    pin_memory = cfg["dataloader"]["pin_memory"]

    # const 的逐轴先验取自 train 的 p̂ 均值
    _, train_probs = build_soft_dataset(
        cfg, split_dir, attr_pred_dir, "train", eval_transform, "soft", None, seed)
    priors = {axis: probs.mean(axis=0) for axis, probs in train_probs.items()}

    sets = {}
    for name, transform in (("train", train_transform), ("val", eval_transform), ("test", eval_transform)):
        sets[name], probs_by_axis = build_soft_dataset(
            cfg, split_dir, attr_pred_dir, name, transform, attr_mode, priors, seed)
        summary = "  ".join(
            f"{axis}: 熵={float((-p * np.log(np.clip(p, 1e-12, None))).sum(axis=1).mean()):.3f}"
            f"/一致率={float((p.argmax(axis=1) == getattr(sets[name].base, {'sex': 'sexes', 'race': 'races', 'age': 'age_groups'}[axis])).mean()):.3f}"
            for axis, p in probs_by_axis.items())
        print(f"  {name:5s}: n={len(sets[name]):>7,}  {summary}")
    print(f"  条件输入模式={attr_mode}"
          + (f"  priors={ {a: np.round(v, 3).tolist() for a, v in priors.items()} }"
             if attr_mode == "const" else ""))

    return (
        DataLoader(sets["train"], batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(sets["val"], batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(sets["test"], batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=pin_memory),
    )


def _fairness_report(y_true, y_score, attrs) -> None:
    """CXR 公平性报告回调：分组键恒用**真值** sex/race/age（p̂ 只作模型输入）。"""
    print_mimic_fairness_report(
        y_true=y_true, y_score=y_score,
        sex=attrs["sex"], race=attrs["race"], age=attrs["age"], threshold=0.0,
    )


def main() -> None:
    args = parse_args()
    spec = DATASET_SPECS[args.dataset]
    cfg = load_config(spec["config"])
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    seed = args.seed if args.seed is not None else cfg["training"]["seeds"][0]
    config_index = args.config_index if args.config_index is not None else spec["config_index"]
    hparam = get_hparam_config(config_index)
    split_dir, output_dir, attr_pred_dir = resolve_paths(spec, cfg, args.cv, args.fold)
    method = METHOD_BY_MODE[args.attr_mode]

    seed_everything(seed)
    print(f"Loading {args.dataset} (pred-attr HyperAdapt, mode={args.attr_mode}, method={method})  "
          f"split_dir={split_dir.parent.name}/{split_dir.name}  attr_pred={attr_pred_dir.name}...")
    train_transform, eval_transform = build_transforms(cfg)
    train_loader, val_loader, test_loader = get_dataloaders(
        spec, cfg, split_dir, attr_pred_dir, train_transform, eval_transform, args.attr_mode, seed)

    run_training(
        dataset=spec["dataset"], method=method, output_dir=output_dir, cfg=cfg,
        hparam=hparam, seed=seed,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        build_model=lambda: make_model(cfg, spec),
        unpack_fn=UNPACK_FN, forward_fn=FORWARD_FN,
        fairness_report_fn=_fairness_report,
        swad=args.swad,
    )


if __name__ == "__main__":
    main()
