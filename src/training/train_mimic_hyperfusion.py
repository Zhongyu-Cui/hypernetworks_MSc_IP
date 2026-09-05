"""
Train ResNet18-HyperFusion on MIMIC-CXR (No Finding, sex/race/age-conditioned)
==============================================================================
仅在 backbone 的 layer4[0].downsample 1x1 卷积处接由 sex/race/age 驱动的 HyperFusion 超网络
（ResNet18HyperFusion，MIP additive + E_L2 投影，初始 Δθ≈0 起步等价 baseline）。HyperFusion 是
「单点、深层」注入（对齐 Duenias et al. 2025, Medical Image Analysis 102:103503），与 HyperHead（只改 fc）、HyperAdapt（每层低秩）
构成注入深度对照；单点注入显存接近 baseline。

**比较协议改造（A0.4）**：训练循环统一走 harness.run_training；本脚本保留 MIMIC 特有的 CXR
transforms / dataloader，注入 unpack_sex_race_age + forward_sex_race_age + print_mimic_fairness_report。
超参 lr/wd 由 --config_index 从网格取（缺省 index=2=中心点）。

运行方式：重型任务，经 sbatch 提交（见 slurm/train_mimic_hyperfusion.sh）。
"""

import argparse
from pathlib import Path

import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.mimic_cxr_dataset import MIMICCXRDataset
from src.models.resnet18_hyperfusion import ResNet18HyperFusion
from src.training.harness.hparam_grid import get_hparam_config, grid_size
from src.training.harness.run import seed_everything
from src.training.harness.train_loop import (
    forward_sex_race_age, run_training, unpack_sex_race_age,
)
from src.utils.mimic_fairness import print_mimic_fairness_report
from src.paths import OUTPUTS_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "mimic_cxr_baseline.yaml"
OUTPUT_DIR = OUTPUTS_DIR / "mimic_cxr"

DATASET = "mimic"
METHOD = "hyperfusion"


def load_config(path: Path) -> dict:
    """读取 YAML 训练配置。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """解析命令行参数（--config_index 超参网格；--seed；--batch_size 可选覆盖，通常无需）。"""
    parser = argparse.ArgumentParser(description="Train ResNet18-HyperFusion on MIMIC-CXR (sex/race/age)")
    parser.add_argument("--config_index", type=int, default=2,
                        help=f"超参网格配置序号 0–{grid_size() - 1}；缺省 2=中心点 (lr=1e-4, wd=1e-4)。")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子；缺省取 mimic_cxr_baseline.yaml 的 training.seeds[0]")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="覆盖 config 的 batch_size；HyperFusion 单点注入显存接近 baseline，通常无需。")
    parser.add_argument("--cv", type=int, default=0,
                        help="K 折患者级 GroupKFold（CV-OOF 用 5）；0=用 master 自带单-split（默认）。")
    parser.add_argument("--fold", type=int, default=None,
                        help="--cv>0 时指定折号 k ∈ [0,K)。")
    return parser.parse_args()


def resolve_paths(cfg: dict, cv: int, fold: int | None) -> tuple[Path, Path]:
    """按 --cv/--fold 解析 (split_dir, output_dir)；CV 时重定向到 cv{K}/fold{k} 与 OUTPUT_DIR/cv{K}，
    与单-split 结果隔离，绝不覆盖。"""
    base = REPO_ROOT / cfg["data"]["split_dir"]
    if cv and cv >= 2:
        if fold is None or not (0 <= fold < cv):
            raise ValueError(f"--cv={cv} 需配合合法 --fold ∈ [0,{cv})")
        return base / f"cv{cv}" / f"fold{fold}", OUTPUT_DIR / f"cv{cv}"
    return base, OUTPUT_DIR


def build_transforms(cfg: dict) -> tuple[transforms.Compose, transforms.Compose]:
    """构建训练 / 评估 transform（与 MIMIC baseline 一致；Grayscale(3)、不水平翻转）。"""
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


def get_dataloaders(
    cfg: dict, split_dir: Path, train_transform: transforms.Compose, eval_transform: transforms.Compose,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建 train / val / test DataLoader（split 已患者级划分；val/test 保持真实分布，HN 不重采样）。"""
    image_size = cfg["data"]["image_size"]
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["dataloader"]["num_workers"]
    pin_memory = cfg["dataloader"]["pin_memory"]
    age_threshold = cfg["attributes"]["age"]["age_threshold"]

    train_set = MIMICCXRDataset(split_dir / "train.csv", transform=train_transform,
                                image_size=image_size, age_threshold=age_threshold)
    val_set = MIMICCXRDataset(split_dir / "val.csv", transform=eval_transform,
                              image_size=image_size, age_threshold=age_threshold)
    test_set = MIMICCXRDataset(split_dir / "test.csv", transform=eval_transform,
                               image_size=image_size, age_threshold=age_threshold)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


def _fairness_report(y_true, y_score, attrs) -> None:
    """MIMIC 公平性报告回调：从 attrs 取 sex / race / age 分组键。"""
    print_mimic_fairness_report(
        y_true=y_true, y_score=y_score,
        sex=attrs["sex"], race=attrs["race"], age=attrs["age"], threshold=0.0,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(CONFIG_PATH)
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    seed = args.seed if args.seed is not None else cfg["training"]["seeds"][0]
    hparam = get_hparam_config(args.config_index)

    split_dir, output_dir = resolve_paths(cfg, args.cv, args.fold)

    seed_everything(seed)
    print(f"Loading MIMIC-CXR (No Finding, sex/race/age-conditioned HyperFusion)  "
          f"split_dir={split_dir.name}  output_dir={output_dir.name}...")
    train_transform, eval_transform = build_transforms(cfg)
    train_loader, val_loader, test_loader = get_dataloaders(cfg, split_dir, train_transform, eval_transform)

    run_training(
        dataset=DATASET, method=METHOD, output_dir=output_dir, cfg=cfg, hparam=hparam, seed=seed,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        build_model=lambda: ResNet18HyperFusion(num_classes=1, pretrained=cfg["model"]["pretrained"]),
        unpack_fn=unpack_sex_race_age, forward_fn=forward_sex_race_age,
        fairness_report_fn=_fairness_report,
    )


if __name__ == "__main__":
    main()
