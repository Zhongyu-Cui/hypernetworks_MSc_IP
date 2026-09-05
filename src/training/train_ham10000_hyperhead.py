"""
Train ResNet18-HyperHead on HAM10000 (malignant, age-conditioned)
=================================================================
在 image-only baseline 基础上，仅让**分类头**由**单一敏感属性 age_group**（0–3）经超网络逐样本
生成（ResNet18HyperHeadAge，末层近零初始、起步等价统一头），backbone 与 baseline 共享。为什么
只条件化 age：HAM 诊断 I(Y;age|X)>0、sex≈0。HyperHead 是「最浅」HN（只改 fc），与 HyperFusion
（单点深层）、HyperAdapt（每层低秩）构成注入深度对照。

**比较协议改造（A0.4）**：训练循环统一走 harness.run_training；本脚本保留 HAM-HN 特有部分
（过滤 age_group>=0；unpack_sex_age + forward_age + HAM 公平性报告）。超参由 --config_index 取网格。

**双属性对照臂（--cond sex_age）**：`--cond age`（缺省）保持原「只条件化 age」行为不变；
`--cond sex_age` 把条件通路换成 sex+age（模型类 *SexAge，forward_sex_age），method 记为
`<method>_sexage`，产物与 age-only 臂分开存放。动机：worst-group 评估口径是 Sex / Age /
Sex×Age 三套分组，而 age-only 臂从未拿到 sex，条件化口径与评估口径不对称；本臂把条件输入
补齐为评估分组变量全集，与 age-only 臂构成**唯一差异=条件输入**的单变量对照（样本集、
age_group>=0 过滤、超参网格、split/seed 规程全部一致）。

运行方式：重型任务，经 sbatch 提交（见 slurm/train_ham10000_hyperhead.sh）。
"""

import argparse
from pathlib import Path

import numpy as np
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.ham10000_dataset import HAM10000Dataset
from src.models.resnet18_hyperhead import ResNet18HyperHeadAge, ResNet18HyperHeadSexAge
from src.training.harness.hparam_grid import get_hparam_config, grid_size
from src.training.harness.run import seed_everything
from src.training.harness.train_loop import (
    forward_age, forward_sex_age, run_training, unpack_sex_age,
)
from src.utils.ham10000_fairness import print_ham10000_fairness_report
from src.paths import OUTPUTS_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "ham10000_baseline.yaml"
OUTPUT_DIR = OUTPUTS_DIR / "ham10000"

DATASET = "ham10000"
METHOD = "hyperhead"
NUM_AGE = 4
NUM_SEX = 2   # HAM sex: 0=Male / 1=Female（双属性对照臂用）


def load_config(path: Path) -> dict:
    """读取 YAML 训练配置。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """解析命令行参数（--config_index 超参网格；--seed；--batch_size 可选覆盖）。"""
    parser = argparse.ArgumentParser(description="Train ResNet18-HyperHead on HAM10000 (age-conditioned)")
    parser.add_argument("--config_index", type=int, default=2,
                        help=f"超参网格配置序号 0–{grid_size() - 1}；缺省 2=中心点 (lr=1e-4, wd=1e-4)。")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子；缺省取 ham10000_baseline.yaml 的 training.seeds[0]")
    parser.add_argument("--batch_size", type=int, default=None, help="覆盖 config 的 batch_size（通常无需）。")
    parser.add_argument("--swad", action="store_true",
                        help="开启 SWAD 权重平均派生（**实验 G**：HN×SWAD 融合，方案见 "
                             "docs/hyperadapt_swad_fusion_plan.md）：额外产出 method=hyperhead_swad 的 "
                             "checkpoint / val 日志 / test 预测，与本 run 同 config/seed，"
                             "与 ERM 派生的 SWAD 基线（method=swad）隔离。")
    parser.add_argument("--cv", type=int, default=0,
                        help="K 折 lesion 级 GroupKFold（路线 A 用 5）；0=用单次 80/10/10 划分（默认）。")
    parser.add_argument("--fold", type=int, default=None,
                        help="--cv>0 时指定折号 k ∈ [0,K)。")
    parser.add_argument("--cond", type=str, default="age", choices=("age", "sex_age"),
                        help="条件通路属性集：age=只条件化 age_group（默认，历史臂）；"
                             "sex_age=同时条件化 sex+age_group（对称性对照臂，method 记为 "
                             "<method>_sexage，与 age-only 臂产物分开存放）。两者唯一差异就是"
                             "条件输入，样本集/过滤/评估口径完全一致。")
    return parser.parse_args()


def resolve_paths(cfg: dict, cv: int, fold: int | None) -> tuple[Path, Path]:
    """
    按 --cv/--fold 解析 (split_dir, output_dir)。CV 时 split 取 cv{K}/fold{k}，output_dir 重定向到
    OUTPUT_DIR/cv{K}，与单次划分完全隔离，不覆盖已冻结的单-split 结果。
    """
    base = REPO_ROOT / cfg["data"]["split_dir"]
    if cv and cv >= 2:
        if fold is None or not (0 <= fold < cv):
            raise ValueError(f"--cv={cv} 需配合合法 --fold ∈ [0,{cv})")
        return base / f"cv{cv}" / f"fold{fold}", OUTPUT_DIR / f"cv{cv}"
    return base, OUTPUT_DIR


def build_transforms(cfg: dict) -> tuple[transforms.Compose, transforms.Compose]:
    """构建训练 / 评估 transform（与 HAM baseline 严格一致）。"""
    t_cfg = cfg["transforms"]
    norm_mean, norm_std = t_cfg["normalize"]["mean"], t_cfg["normalize"]["std"]
    train_c, eval_c = t_cfg["train"], t_cfg["eval"]

    train_ops: list = [
        transforms.Resize(tuple(train_c["resize"])),
        transforms.RandomCrop(train_c["random_crop"]),
    ]
    if train_c.get("horizontal_flip", False):
        train_ops.append(transforms.RandomHorizontalFlip())
    train_ops.append(transforms.RandomRotation(degrees=train_c["rotation_degrees"]))
    train_ops += [transforms.ToTensor(), transforms.Normalize(mean=norm_mean, std=norm_std)]
    train_transform = transforms.Compose(train_ops)

    eval_transform = transforms.Compose([
        transforms.Resize(tuple(eval_c["resize"])),
        transforms.CenterCrop(eval_c["center_crop"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])
    return train_transform, eval_transform


def _filter_age_valid(dataset: HAM10000Dataset) -> int:
    """就地过滤 age_group==-1（0-20），只保留 age_group∈{0..3}（embedding 不能吃 -1）。返回剩余数。"""
    mask = np.asarray(dataset.age_groups).astype(int) >= 0
    dataset.image_paths = dataset.image_paths[mask]
    dataset.labels = dataset.labels[mask]
    dataset.sexes = dataset.sexes[mask]
    dataset.age_groups = dataset.age_groups[mask]
    return int(mask.sum())


def get_dataloaders(
    cfg: dict, split_dir: Path,
    train_transform: transforms.Compose, eval_transform: transforms.Compose,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建 train/val/test DataLoader，三 split 均过滤 age_group>=0。"""
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["dataloader"]["num_workers"]
    pin_memory = cfg["dataloader"]["pin_memory"]

    train_set = HAM10000Dataset(split_dir / "train.csv", transform=train_transform)
    val_set = HAM10000Dataset(split_dir / "val.csv", transform=eval_transform)
    test_set = HAM10000Dataset(split_dir / "test.csv", transform=eval_transform)

    n_tr, n_va, n_te = _filter_age_valid(train_set), _filter_age_valid(val_set), _filter_age_valid(test_set)
    print(f"  过滤 age_group>=0 后: train {n_tr:,} / val {n_va:,} / test {n_te:,}（已排除 0-20）")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


def _fairness_report(y_true, y_score, attrs) -> None:
    """HAM 公平性报告回调：从 attrs 取 sex / age(=age_group) 分组键。"""
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
    split_dir, output_dir = resolve_paths(cfg, args.cv, args.fold)

    # 条件通路选择：age-only（历史臂）或 sex+age（对称性对照臂），二者仅差条件输入
    sex_age = args.cond == "sex_age"
    method = f"{METHOD}_sexage" if sex_age else METHOD
    build_model = (
        (lambda: ResNet18HyperHeadSexAge(num_classes=1, num_sex=NUM_SEX, num_age=NUM_AGE))
        if sex_age else
        (lambda: ResNet18HyperHeadAge(num_classes=1, num_age=NUM_AGE))
    )
    forward_fn = forward_sex_age if sex_age else forward_age

    seed_everything(seed)
    print(f"Loading HAM10000 (malignant, {args.cond}-conditioned HyperHead)  "
          f"split_dir={split_dir.name}  output_dir={output_dir.name}...")
    train_transform, eval_transform = build_transforms(cfg)
    train_loader, val_loader, test_loader = get_dataloaders(cfg, split_dir, train_transform, eval_transform)

    run_training(
        dataset=DATASET, method=method, output_dir=output_dir, cfg=cfg, hparam=hparam, seed=seed,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        build_model=build_model,
        unpack_fn=unpack_sex_age, forward_fn=forward_fn,
        fairness_report_fn=_fairness_report,
        swad=args.swad,
    )


if __name__ == "__main__":
    main()
