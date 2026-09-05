"""
Train ResNet18 on MIMIC-CXR for No Finding Binary Classification (Image-only, ERM)
=================================================================================
仅以胸片图像作为输入（不引入 sex/race/age），预测 No Finding（0=有病灶, 1=无病灶/健康）。
Backbone 为 ImageNet 预训练 ResNet-18。⚠️ CXR7-1M PNG 为 16-bit，Dataset 内部已做 min-max
归一化（禁用 convert("L")，见仓库 CLAUDE.md）。

**比较协议改造（A0.4）**：训练循环（早停 / 双 checkpoint / 逐 epoch 子群 AUC 向量日志 /
测试集公平性报告）统一走 harness.run_training；本脚本保留 MIMIC 特有的 transforms / dataloader
（含可选重采样），注入回调：unpack_sex_race_age + forward_image_only + print_mimic_fairness_report。
超参 lr/wd 由 --config_index 从网格取（协议 §3 的 6 配置）；缺省 index=2=中心点 (lr=1e-4, wd=1e-4)。

运行方式：重型任务，经 sbatch 提交（见 slurm/train_mimic_resnet18.sh）。
"""

import argparse
from pathlib import Path

import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.mimic_cxr_dataset import MIMICCXRDataset
from src.models.resnet18_pretrained import ResNet18Pretrained
from src.training.harness.hparam_grid import get_hparam_config, grid_size
from src.training.harness.run import seed_everything
from src.training.harness.train_loop import (
    forward_image_only, run_training, unpack_sex_race_age,
)
from src.utils.mimic_fairness import print_mimic_fairness_report
from src.utils.resampling import build_group_label_sampler
from src.paths import OUTPUTS_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "mimic_cxr_baseline.yaml"
OUTPUT_DIR = OUTPUTS_DIR / "mimic_cxr"

DATASET = "mimic"
METHOD = "erm"


def load_config(path: Path) -> dict:
    """读取 YAML 训练配置。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """解析命令行参数（--config_index 超参网格；--seed；--resample_alpha 敏感组×标签平衡重采样）。"""
    parser = argparse.ArgumentParser(description="Train ResNet18 on MIMIC-CXR (No Finding, image-only, ERM)")
    parser.add_argument("--config_index", type=int, default=2,
                        help=f"超参网格配置序号 0–{grid_size() - 1}；缺省 2=中心点 (lr=1e-4, wd=1e-4)。")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子；缺省取 mimic_cxr_baseline.yaml 的 training.seeds[0]")
    parser.add_argument("--resample_alpha", type=float, default=None,
                        help="敏感组×标签平衡重采样强度 ∈[0,1]（0=自然分布，1=完全平衡）；缺省不重采样。")
    parser.add_argument("--swad", action="store_true",
                        help="开启 SWAD 权重平均派生（协议 C*.7）：逐 epoch 缓存 CPU 权重，训练末在 loss "
                             "谷区间平均，额外产出 method=swad 的 checkpoint / val 日志，与 ERM 同 config/seed。")
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
    """
    构建训练 / 评估 transform。CXR7-1M 已预处理至 224×224，直接旋转增强，不水平翻转
    （胸片有固定解剖左右关系）。Grayscale(3) 复制单通道为 3 通道以兼容 ImageNet backbone。
    """
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
    cfg: dict,
    split_dir: Path,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
    resample_alpha: float | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    构建 train / val / test DataLoader（split 已由 build_mimic_splits_nofinding.py 患者级划分）。
    val/test 保持人群真实分布。resample_alpha 给定时对 train 用 WeightedRandomSampler 平衡（仅 train）。
    """
    image_size = cfg["data"]["image_size"]
    train_csv_name = cfg["data"].get("train_csv", "train.csv")
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["dataloader"]["num_workers"]
    pin_memory = cfg["dataloader"]["pin_memory"]
    age_threshold = cfg["attributes"]["age"]["age_threshold"]

    train_set = MIMICCXRDataset(split_dir / train_csv_name, transform=train_transform,
                                image_size=image_size, age_threshold=age_threshold)
    val_set = MIMICCXRDataset(split_dir / "val.csv", transform=eval_transform,
                              image_size=image_size, age_threshold=age_threshold)
    test_set = MIMICCXRDataset(split_dir / "test.csv", transform=eval_transform,
                               image_size=image_size, age_threshold=age_threshold)

    if resample_alpha is None:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory)
    else:
        sampler = build_group_label_sampler(
            train_set, dims=cfg["resampling"]["dims"], alpha=resample_alpha, verbose=True,
        )
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,
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
    seed = args.seed if args.seed is not None else cfg["training"]["seeds"][0]
    hparam = get_hparam_config(args.config_index)

    resample_alpha = args.resample_alpha
    if resample_alpha is None and cfg.get("resampling", {}).get("enabled", False):
        resample_alpha = cfg["resampling"]["alpha"]

    split_dir, output_dir = resolve_paths(cfg, args.cv, args.fold)

    seed_everything(seed)
    print(f"Loading MIMIC-CXR (No Finding, image-only ERM)  split_dir={split_dir.name}  "
          f"output_dir={output_dir.name}  resample={'OFF' if resample_alpha is None else resample_alpha}...")
    train_transform, eval_transform = build_transforms(cfg)
    train_loader, val_loader, test_loader = get_dataloaders(
        cfg, split_dir, train_transform, eval_transform, resample_alpha=resample_alpha,
    )
    print(f"  Train: {len(train_loader.dataset):,} | Val: {len(val_loader.dataset):,} | "
          f"Test: {len(test_loader.dataset):,}")

    run_training(
        dataset=DATASET, method=METHOD, output_dir=output_dir, cfg=cfg, hparam=hparam, seed=seed,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        build_model=lambda: ResNet18Pretrained(num_classes=1),
        unpack_fn=unpack_sex_race_age, forward_fn=forward_image_only,
        fairness_report_fn=_fairness_report,
        swad=args.swad,
    )


if __name__ == "__main__":
    main()
