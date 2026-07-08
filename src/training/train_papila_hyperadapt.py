"""
Train ResNet18-HyperAdapt on PAPILA (glaucoma, age-conditioned)
===============================================================
把 backbone **每个卷积层**（通道级乘性调制）与分类头 fc（加性低秩更新）都接上由**单一敏感属性
age_group**（≤60/>60 = 0/1）驱动的超网络（ResNet18HyperAdaptAge，num_age=2，初始 Δθ≈0 起步等价
baseline）。HyperAdapt 是「介入最遍布」的深层 HN（除 stem 外每层），与 HyperFusion（单点深层）、
HyperHead（最浅）构成注入深度对照。为什么条件化 age：PAPILA age 为主敏感轴（60 分箱，age 强轴）。
PAPILA age_group 二值、全部合法，**无需过滤 -1**（区别于 HAM）。PAPILA 极小，支持 --cv/--fold
GroupKFold 保功效。

**比较协议改造（A0.4）**：训练循环统一走 harness.run_training；本脚本保留 PAPILA 特有的 CV split
解析 / transforms / dataloader（cache），注入 unpack_sex_age + forward_age + print_papila_fairness_report。
超参 lr/wd 由 --config_index 从网格取（缺省 index=2=中心点）。⚠️ 逐样本卷积核显存大，`--batch_size`
覆盖 config（协议要求重型 HN 显存对照等 batch）。

运行方式：重型任务，经 sbatch 提交（见 slurm/train_papila_hyperadapt.sh）。
"""

import argparse
from pathlib import Path

import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.papila_dataset import PAPILADataset
from src.models.resnet18_hyperadapt import ResNet18HyperAdaptAge
from src.training.harness.hparam_grid import get_hparam_config, grid_size
from src.training.harness.run import seed_everything
from src.training.harness.train_loop import forward_age, run_training, unpack_sex_age
from src.utils.papila_fairness import print_papila_fairness_report

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "papila_baseline.yaml"
OUTPUT_DIR = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/papila")

DATASET = "papila"
METHOD = "hyperadapt"
NUM_AGE = 2   # PAPILA age_group: ≤60=0 / >60=1（二值，无排除组）


def load_config(path: Path) -> dict:
    """读取 YAML 训练配置。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """解析命令行参数（--config_index 超参网格；--seed；--cv/--fold GroupKFold；--batch_size 覆盖）。"""
    parser = argparse.ArgumentParser(description="Train ResNet18-HyperAdapt on PAPILA (age-conditioned)")
    parser.add_argument("--config_index", type=int, default=2,
                        help=f"超参网格配置序号 0–{grid_size() - 1}；缺省 2=中心点 (lr=1e-4, wd=1e-4)。")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子；缺省取 papila_baseline.yaml 的 training.seeds[0]")
    parser.add_argument("--cv", type=int, default=0, help="K 折 GroupKFold；0=用单次 70/10/20 划分（默认）")
    parser.add_argument("--fold", type=int, default=None, help="--cv>0 时指定折号 k ∈ [0,K)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="覆盖 config 的 batch_size；HyperAdapt 逐样本卷积核显存大，协议要求等 batch 对照。")
    return parser.parse_args()


def resolve_split_dir(cfg: dict, cv: int, fold: int | None) -> Path:
    """根据 --cv/--fold 解析 split 目录（cv>=2 时取 cv{K}/fold{k}，否则用单次划分）。"""
    base = REPO_ROOT / cfg["data"]["split_dir"]
    if cv and cv >= 2:
        if fold is None or not (0 <= fold < cv):
            raise ValueError(f"--cv={cv} 需配合合法 --fold ∈ [0,{cv})")
        return base / f"cv{cv}" / f"fold{fold}"
    return base


def build_transforms(cfg: dict) -> tuple[transforms.Compose, transforms.Compose]:
    """构建训练 / 评估 transform（与 PAPILA baseline 一致；源图已缓存 256，Resize 幂等）。"""
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


def get_dataloaders(
    cfg: dict, split_dir: Path,
    train_transform: transforms.Compose, eval_transform: transforms.Compose,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建 train / val / test DataLoader（PAPILA 极小，dataset 开 cache=True 整库常驻内存）。"""
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["dataloader"]["num_workers"]
    pin_memory = cfg["dataloader"]["pin_memory"]
    cache = cfg["data"].get("cache", True)

    train_set = PAPILADataset(split_dir / "train.csv", transform=train_transform, cache=cache)
    val_set = PAPILADataset(split_dir / "val.csv", transform=eval_transform, cache=cache)
    test_set = PAPILADataset(split_dir / "test.csv", transform=eval_transform, cache=cache)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


def _fairness_report(y_true, y_score, attrs) -> None:
    """PAPILA 公平性报告回调：从 attrs 取 sex / age(=age_group) 分组键。"""
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
    split_dir = resolve_split_dir(cfg, args.cv, args.fold)

    seed_everything(seed)
    print(f"Loading PAPILA (glaucoma, age-conditioned HyperAdapt)  split_dir={split_dir.name}...")
    train_transform, eval_transform = build_transforms(cfg)
    train_loader, val_loader, test_loader = get_dataloaders(
        cfg, split_dir, train_transform, eval_transform,
    )
    print(f"  Train: {len(train_loader.dataset):,} | Val: {len(val_loader.dataset):,} | "
          f"Test: {len(test_loader.dataset):,}")

    run_training(
        dataset=DATASET, method=METHOD, output_dir=OUTPUT_DIR, cfg=cfg, hparam=hparam, seed=seed,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        build_model=lambda: ResNet18HyperAdaptAge(
            num_classes=1, num_age=NUM_AGE, pretrained=cfg["model"]["pretrained"]),
        unpack_fn=unpack_sex_age, forward_fn=forward_age,
        fairness_report_fn=_fairness_report,
    )


if __name__ == "__main__":
    main()
