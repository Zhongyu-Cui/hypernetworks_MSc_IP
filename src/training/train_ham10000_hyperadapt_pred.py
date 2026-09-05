"""
Train Predicted-Attribute HyperAdapt on HAM10000（实验 P，敏感属性由 g 预测）
=============================================================================
把 `train_ham10000_hyperadapt.py` 的条件输入从**真值属性索引**换成**子群分类器 g 的
softmax 概率 p̂**（训练与测试两阶段都用 p̂）。g + HyperAdapt 视为**单一方法**：g 已由
`scripts/build_attr_predictions.py` 在该 fold 的 train 上拟合并对三个 split 出好 p̂（in-sample）。

方案：`docs/predicted_attribute_hyperadapt_plan.md`；五库结果见
`docs/predicted_attribute_hyperadapt_results.md`。四种条件输入模式（--attr_mode）对应四臂：
soft(主线) / hard(消融) / perm(路由对照) / const(容量对照)。

**双属性对照臂（--cond sex_age）**：`--cond age`（缺省）保持原「只条件化 age」行为逐位不变；
`--cond sex_age` 把条件通路换成 sex+age 两份 p̂（模型类 *SoftSexAge，forward_soft_sex_age），
method 记为 `<method>_sexage`，产物与 age-only 臂分开存放。动机与 GT 臂的
`train_ham10000_hyperadapt.py --cond sex_age` 完全一致：worst-group 评估口径是 Sex / Age /
Sex×Age 三套分组，而 age-only 臂从未拿到 sex；本臂把条件输入补齐为评估分组变量全集，
与 age-only pred 臂构成**唯一差异=条件输入**的单变量对照（样本集、过滤、超参、split/seed 全同）。

⚠️ 两个 HAM 特有约束：
  1. **age_group==-1（0-20 排除组）必须过滤**（embedding 不吃 -1，且与 GT 臂口径一致）。过滤是
     *就地*进行、会改变行序，因此 p̂ 的 join **必须按 image_id 唯一键**、并用与过滤同一个 mask
     导出键序（`build_soft_dataset`）。
  2. 与 GT 臂**等 batch**（128）与同超参（--config_index 0 = lr3e-05_wd1e-04 = HAM GT-HyperAdapt
     的 Pareto 选定配置），否则重蹈「重型 HN 显存对照必须等 batch」的坑。sex+age 臂沿用同一
     config_index，使其与 age-only pred 臂天然 config-matched。

评估侧不变：worst-group / 公平性分组键恒用**真值** sex 与 age_group。
运行方式：重型任务，经 sbatch 提交（见 slurm/p_pred_attr.sh，双属性臂传 COND=sex_age）。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.ham10000_dataset import HAM10000Dataset
from src.datasets.soft_attr_wrapper import (
    ATTR_MODES, SoftAttrDataset, align_probs_by_key, materialize_probs,
)
from src.models.resnet18_hyperadapt import (
    ResNet18HyperAdaptSoftAge, ResNet18HyperAdaptSoftSexAge,
)
from src.training.attr_predictor import load_fold_predictions
from src.training.harness.hparam_grid import get_hparam_config, grid_size
# transform 直接复用 GT 臂的实现（单一事实来源，保证「仅属性来源」的单变量差异）
from src.training.train_ham10000_hyperadapt import build_transforms
from src.training.harness.run import seed_everything
from src.training.harness.train_loop import (
    forward_soft_age, forward_soft_sex_age, run_training,
    unpack_sex_age_soft, unpack_sex_age_soft_pair,
)
from src.utils.ham10000_fairness import print_ham10000_fairness_report

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "ham10000_baseline.yaml"
OUTPUT_DIR = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/ham10000")

DATASET = "ham10000"
KEY_COL = "image_id"
NUM_AGE = 4
NUM_SEX = 2   # HAM sex: 0=Male / 1=Female（双属性臂用）

# 条件通路 -> (属性轴顺序, attr_pred 目录名)。轴顺序即 SoftAttrDataset 追加概率矩阵的顺序，
# 须与 unpack 回调的解包顺序一致。两个 cond 的 p̂ 来自**不同的 attr_pred 目录**：
# age-only 沿用既有 `attr_pred/`（内含 prob_age）；sex+age 用 `attr_pred_sexage/`
# （内含 prob_sex + prob_age，由 `build_attr_predictions.py --dataset ham10000_sexage` 生成，
# 与既有产物完全隔离，避免改写支撑已冻结结果的文件）。
COND_AXES: dict[str, tuple[str, ...]] = {"age": ("age",), "sex_age": ("sex", "age")}
COND_ATTR_DIR: dict[str, str] = {"age": "attr_pred", "sex_age": "attr_pred_sexage"}
# base dataset 上各轴真值的属性名（用于校验 p̂ 文件与 split 逐样本对齐）
GT_ATTR_OF_AXIS: dict[str, str] = {"sex": "sexes", "age": "age_groups"}

METHOD_BY_MODE: dict[str, str] = {
    "soft": "hyperadapt_pred",
    "hard": "hyperadapt_predhard",
    "perm": "hyperadapt_predperm",
    "const": "hyperadapt_predconst",
}

# 供 smoke test / p_reeval_from_checkpoints 引用的回调与模型工厂。默认取 age-only 臂（历史行为）；
# 双属性臂用下面的 *_of_cond 版本按 cond 取。
UNPACK_FN = unpack_sex_age_soft
FORWARD_FN = forward_soft_age


def unpack_fn_of_cond(cond: str):
    """按条件通路取 batch 解包回调（age: 1 份 p̂；sex_age: 2 份 p̂）。"""
    return unpack_sex_age_soft if cond == "age" else unpack_sex_age_soft_pair


def forward_fn_of_cond(cond: str):
    """按条件通路取 forward 回调。"""
    return forward_soft_age if cond == "age" else forward_soft_sex_age


def method_of(attr_mode: str, cond: str) -> str:
    """method 名：age-only 臂沿用历史名，双属性臂加 `_sexage` 后缀（产物分开存放）。"""
    base = METHOD_BY_MODE[attr_mode]
    return base if cond == "age" else f"{base}_sexage"


def build_model_of_cond(cfg: dict, cond: str):
    """按 cond 造 pred-attr 模型：age-only 用 SoftAge，双属性用 SoftSexAge。"""
    if cond == "age":
        return ResNet18HyperAdaptSoftAge(
            num_classes=1, num_age=NUM_AGE, pretrained=cfg["model"]["pretrained"])
    return ResNet18HyperAdaptSoftSexAge(
        num_classes=1, num_sex=NUM_SEX, num_age=NUM_AGE,
        pretrained=cfg["model"]["pretrained"])


def MODEL_FACTORY(cfg: dict, cond: str = "age"):    # noqa: N802 (与 smoke / p_reeval 约定的名字对齐)
    """按 config 造 pred-attr 模型（与 main() 中 build_model 一致）。"""
    return build_model_of_cond(cfg, cond)


def load_config(path: Path) -> dict:
    """读取 YAML 训练配置。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Train Predicted-Attribute HyperAdapt on HAM10000（实验 P）")
    parser.add_argument("--attr_mode", choices=ATTR_MODES, default="soft",
                        help="条件输入模式：soft(主线)/hard(消融)/perm(路由对照)/const(容量对照)。")
    parser.add_argument("--cond", type=str, default="age", choices=tuple(COND_AXES),
                        help="条件通路属性集：age=只用 age 的 p̂（默认，历史臂）；"
                             "sex_age=同时用 sex 与 age 的 p̂（双属性对照臂，method 记为 "
                             "<method>_sexage，p̂ 取自 attr_pred_sexage/）。")
    parser.add_argument("--config_index", type=int, default=0,
                        help=f"超参网格序号 0–{grid_size() - 1}；缺省 0=lr3e-05_wd1e-04，"
                             "即 HAM GT-HyperAdapt 的 Pareto 选定配置（单变量对照）。")
    parser.add_argument("--seed", type=int, default=None, help="随机种子；CV 时由 slurm 传 42+fold。")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="覆盖 config 的 batch_size（须与 GT 臂等 batch，HAM=128）。")
    parser.add_argument("--cv", type=int, default=5, help="K 折 lesion 级 GroupKFold（实验 P 用 5）。")
    parser.add_argument("--fold", type=int, default=None, help="--cv>0 时的折号 k ∈ [0,K)。")
    parser.add_argument("--swad", action="store_true", help="额外派生 SWAD 权重平均模型。")
    return parser.parse_args()


def resolve_paths(cfg: dict, cv: int, fold: int | None,
                  cond: str = "age") -> tuple[Path, Path, Path]:
    """
    解析 (split_dir, output_dir, attr_pred_dir)。

    Args:
        cfg : 数据集 yaml。
        cv  : 折数。
        fold: 折号。
        cond: 条件通路（决定取哪个 attr_pred 目录）。

    Returns:
        (split_dir, output_dir, attr_pred_dir)。

    Raises:
        ValueError: cv>=2 但 fold 非法。
    """
    base = REPO_ROOT / cfg["data"]["split_dir"]
    attr_sub = COND_ATTR_DIR[cond]
    if cv and cv >= 2:
        if fold is None or not (0 <= fold < cv):
            raise ValueError(f"--cv={cv} 需配合合法 --fold ∈ [0,{cv})")
        return (base / f"cv{cv}" / f"fold{fold}",
                OUTPUT_DIR / f"cv{cv}",
                OUTPUT_DIR / f"cv{cv}" / attr_sub / f"fold{fold}")
    return base, OUTPUT_DIR, OUTPUT_DIR / attr_sub


def filter_age_valid(dataset: HAM10000Dataset) -> np.ndarray:
    """
    就地过滤 age_group==-1（0-20 排除组），只保留 age_group∈{0..3}。

    与 `train_ham10000_hyperadapt.py::_filter_age_valid` 口径完全一致（embedding 不吃 -1）。

    Args:
        dataset: 待过滤的 HAM10000Dataset（就地修改）。

    Returns:
        保留行的布尔 mask（供调用方用同一 mask 导出样本键，保证 join 行序一致）。
    """
    mask = np.asarray(dataset.age_groups).astype(int) >= 0
    dataset.image_paths = dataset.image_paths[mask]
    dataset.labels = dataset.labels[mask]
    dataset.sexes = dataset.sexes[mask]
    dataset.age_groups = dataset.age_groups[mask]
    return mask


def build_soft_dataset(
    split_dir: Path, attr_pred_dir: Path, split_name: str, transform,
    attr_mode: str, priors: dict[str, np.ndarray] | None, seed: int, cond: str = "age",
) -> tuple[SoftAttrDataset, dict[str, np.ndarray]]:
    """
    构建某 split 的 SoftAttrDataset：过滤 age_group>=0 的 base dataset + 按 image_id join 的逐轴 p̂。

    Args:
        split_dir    : cv{K}/fold{k} 的 split 目录。
        attr_pred_dir: 该 fold 的 attr_pred 目录（随 cond 变）。
        split_name   : "train"/"val"/"test"。
        transform    : 该 split 的 transform。
        attr_mode    : 条件输入模式。
        priors       : const 模式用的逐轴先验向量。
        seed         : perm 模式的置换种子。
        cond         : 条件通路（决定用哪些属性轴）。

    Returns:
        (dataset, {轴 -> 校准后的原始 soft 概率矩阵})。

    Raises:
        ValueError: 预测文件的真值属性与过滤后的 split 不一致。
    """
    csv_path = split_dir / f"{split_name}.csv"
    base = HAM10000Dataset(csv_path, transform=transform)
    mask = filter_age_valid(base)
    # 用与过滤同一个 mask 导出键序：**禁止按行号对齐**（过滤已改变行序）
    keys = pd.read_csv(csv_path)[KEY_COL].astype(str).to_numpy()[mask]

    axes = COND_AXES[cond]
    perm_seed = seed * 100 + {"train": 0, "val": 1, "test": 2}[split_name]
    probs_by_axis: dict[str, np.ndarray] = {}
    materialized: list[np.ndarray] = []
    for axis_index, axis in enumerate(axes):
        pred_keys, pred_probs, pred_gt = load_fold_predictions(
            attr_pred_dir / f"{split_name}.npz", axis=axis)
        probs = align_probs_by_key(keys, pred_keys, pred_probs)
        gt_aligned = align_probs_by_key(
            keys, pred_keys, pred_gt.reshape(-1, 1)).ravel().astype(np.int64)
        gt_true = np.asarray(getattr(base, GT_ATTR_OF_AXIS[axis]), dtype=np.int64)
        if not np.array_equal(gt_aligned, gt_true):
            raise ValueError(f"{split_name}/{axis}: 属性预测文件的真值与过滤后的 split 不一致")
        probs_by_axis[axis] = probs
        # 单轴保持历史行为（不加偏移，与既有 age-only 产物逐位可复现）；多轴时各轴用不同置换
        # 种子，避免同步置换保留轴间相关、削弱 perm 对照的破坏力（与 CXR 三轴脚本同口径）。
        axis_seed = perm_seed if len(axes) == 1 else perm_seed + 10 * (axis_index + 1)
        materialized.append(materialize_probs(
            probs, attr_mode,
            prior=None if priors is None else priors[axis], seed=axis_seed))
    return SoftAttrDataset(base, materialized), probs_by_axis


def get_dataloaders(
    cfg: dict, split_dir: Path, attr_pred_dir: Path,
    train_transform, eval_transform, attr_mode: str, seed: int, cond: str = "age",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建三个 split 的 DataLoader（均已过滤 age_group>=0，条件输入按 attr_mode 物化）。"""
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["dataloader"]["num_workers"]
    pin_memory = cfg["dataloader"]["pin_memory"]

    # const 的逐轴先验取自 train 的 p̂ 均值
    _, train_probs = build_soft_dataset(
        split_dir, attr_pred_dir, "train", eval_transform, "soft", None, seed, cond)
    priors = {axis: probs.mean(axis=0) for axis, probs in train_probs.items()}

    sets = {}
    for name, transform in (("train", train_transform), ("val", eval_transform), ("test", eval_transform)):
        sets[name], probs_by_axis = build_soft_dataset(
            split_dir, attr_pred_dir, name, transform, attr_mode, priors, seed, cond)
        summary = "  ".join(
            f"{axis}: 熵={float((-p * np.log(np.clip(p, 1e-12, None))).sum(axis=1).mean()):.3f}"
            f"/一致率={float((p.argmax(axis=1) == np.asarray(getattr(sets[name].base, GT_ATTR_OF_AXIS[axis]))).mean()):.3f}"
            for axis, p in probs_by_axis.items())
        print(f"  {name:5s}: n={len(sets[name]):>6,}（已排除 age_group=-1）  {summary}")
    print(f"  条件输入模式={attr_mode}  条件通路={cond}"
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
    """HAM 公平性报告回调：分组键恒用**真值** sex / age_group（p̂ 只作模型输入）。"""
    print_ham10000_fairness_report(
        y_true=y_true, y_score=y_score, sex=attrs["sex"], age_group=attrs["age"], threshold=0.0,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(CONFIG_PATH)
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    seed = args.seed if args.seed is not None else cfg["training"]["seeds"][0]
    hparam = get_hparam_config(args.config_index)
    split_dir, output_dir, attr_pred_dir = resolve_paths(cfg, args.cv, args.fold, args.cond)
    method = method_of(args.attr_mode, args.cond)

    seed_everything(seed)
    print(f"Loading HAM10000 (pred-attr HyperAdapt, mode={args.attr_mode}, cond={args.cond}, "
          f"method={method})  split_dir={split_dir.parent.name}/{split_dir.name}  "
          f"attr_pred={attr_pred_dir.parent.name}/{attr_pred_dir.name}...")
    train_transform, eval_transform = build_transforms(cfg)
    train_loader, val_loader, test_loader = get_dataloaders(
        cfg, split_dir, attr_pred_dir, train_transform, eval_transform,
        args.attr_mode, seed, args.cond)

    run_training(
        dataset=DATASET, method=method, output_dir=output_dir, cfg=cfg, hparam=hparam, seed=seed,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        build_model=lambda: build_model_of_cond(cfg, args.cond),
        unpack_fn=unpack_fn_of_cond(args.cond), forward_fn=forward_fn_of_cond(args.cond),
        fairness_report_fn=_fairness_report,
        swad=args.swad,
    )


if __name__ == "__main__":
    main()
