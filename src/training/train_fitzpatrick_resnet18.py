"""
Train ResNet18 on Fitzpatrick17k for Malignant Binary Classification (Image-only, ERM)
======================================================================================
仅以皮肤镜 RGB 图像作为输入（不引入肤色），预测 malignant（0=benign/non-neoplastic, 1=malignant；
恶性占比约 13.5%）。敏感属性为单一肤色轴 skin（Fitzpatrick I–VI → 0–5）。preproc_224x224 已是
自然 RGB 且尺寸即 224，无需 Grayscale/Resize，训练增强保留水平翻转（皮损无左右约束）。

**比较协议改造（A0.4）**：训练循环统一走 harness.run_training；本脚本保留 Fitzpatrick 特有的
transforms / dataloader（含可选重采样），注入 unpack_skin + forward_image_only +
print_fitzpatrick_fairness_report。超参 lr/wd 由 --config_index 从网格取（缺省 index=2=中心点）。

运行方式：重型任务，经 sbatch 提交（见 slurm/train_fitzpatrick_resnet18.sh）。
"""

import argparse
from pathlib import Path

import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.fitzpatrick_dataset import FitzpatrickDataset
from src.models.resnet18_pretrained import ResNet18Pretrained
from src.training.harness.hparam_grid import get_hparam_config, grid_size
from src.training.harness.run import seed_everything
from src.training.harness.train_loop import forward_image_only, run_training, unpack_skin
from src.utils.fitzpatrick_fairness import print_fitzpatrick_fairness_report
from src.utils.resampling import build_group_label_sampler

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "fitzpatrick_baseline.yaml"
OUTPUT_DIR = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/fitzpatrick")

DATASET = "fitzpatrick"
METHOD = "erm"


def load_config(path: Path) -> dict:
    """读取 YAML 训练配置。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """解析命令行参数（--config_index 超参网格；--seed；--resample_alpha 肤色×标签平衡重采样）。"""
    parser = argparse.ArgumentParser(description="Train ResNet18 on Fitzpatrick17k (malignant, image-only, ERM)")
    parser.add_argument("--config_index", type=int, default=2,
                        help=f"超参网格配置序号 0–{grid_size() - 1}；缺省 2=中心点 (lr=1e-4, wd=1e-4)。")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子；缺省取 fitzpatrick_baseline.yaml 的 training.seeds[0]")
    parser.add_argument("--resample_alpha", type=float, default=None,
                        help="skin×label 平衡重采样强度 ∈[0,1]（0=自然分布，1=完全平衡）；缺省不重采样。")
    parser.add_argument("--swad", action="store_true",
                        help="开启 SWAD 权重平均派生（协议 C*.7）：逐 epoch 缓存 CPU 权重，训练末在 loss "
                             "谷区间平均，额外产出 method=swad 的 checkpoint / val 日志，与 ERM 同 config/seed。")
    parser.add_argument("--cv", type=int, default=0,
                        help="K 折 StratifiedKFold（实验 F CV-OOF 用 5）；0=用单次 80/10/10 划分（默认）。")
    parser.add_argument("--fold", type=int, default=None,
                        help="--cv>0 时指定折号 k ∈ [0,K)。")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="冻结 regime：冻结 ImageNet 预训练 backbone 卷积/BN 仿射参数，只训任务头 fc "
                             "(linear probe)，作为冻结 HN 的 ERM 对照。method 记为 erm_frozen，与全微调分开存放。")
    return parser.parse_args()


def resolve_paths(cfg: dict, cv: int, fold: int | None) -> tuple[Path, Path]:
    """
    按 --cv/--fold 解析 (split_dir, output_dir)。CV 时 split 取 cv{K}/fold{k}，output_dir
    重定向到 OUTPUT_DIR/cv{K}——与单次划分的预测/权重完全隔离，绝不覆盖已冻结的单-split 结果。
    """
    base = REPO_ROOT / cfg["data"]["split_dir"]
    if cv and cv >= 2:
        if fold is None or not (0 <= fold < cv):
            raise ValueError(f"--cv={cv} 需配合合法 --fold ∈ [0,{cv})")
        return base / f"cv{cv}" / f"fold{fold}", OUTPUT_DIR / f"cv{cv}"
    return base, OUTPUT_DIR


def build_transforms(cfg: dict) -> tuple[transforms.Compose, transforms.Compose]:
    """
    构建训练 / 评估 transform。Fitzpatrick preproc_224x224 已是 224 原生 RGB，无需 Grayscale/Resize；
    训练保留水平翻转（皮损无左右约束）+ 旋转；评估不做几何增强。
    """
    t_cfg = cfg["transforms"]
    norm_mean, norm_std = t_cfg["normalize"]["mean"], t_cfg["normalize"]["std"]

    train_ops: list = []
    if t_cfg["train"].get("horizontal_flip", False):
        train_ops.append(transforms.RandomHorizontalFlip())
    train_ops.append(transforms.RandomRotation(degrees=t_cfg["train"]["rotation_degrees"]))
    train_ops += [transforms.ToTensor(), transforms.Normalize(mean=norm_mean, std=norm_std)]
    train_transform = transforms.Compose(train_ops)

    eval_transform = transforms.Compose([
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
    """构建 train / val / test DataLoader（split 已图像级划分；val/test 保持真实分布）。"""
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["dataloader"]["num_workers"]
    pin_memory = cfg["dataloader"]["pin_memory"]

    train_set = FitzpatrickDataset(split_dir / "train.csv", transform=train_transform)
    val_set = FitzpatrickDataset(split_dir / "val.csv", transform=eval_transform)
    test_set = FitzpatrickDataset(split_dir / "test.csv", transform=eval_transform)

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
    """Fitzpatrick 公平性报告回调：从 attrs 取 skin 分组键（6 肤色子群，不报 EqOdd）。"""
    print_fitzpatrick_fairness_report(
        y_true=y_true, y_score=y_score, skin=attrs["skin"], threshold=0.0,
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
    method = "erm_frozen" if args.freeze_backbone else METHOD

    seed_everything(seed)
    print(f"Loading Fitzpatrick17k (malignant, image-only ERM)  split_dir={split_dir.name}  "
          f"output_dir={output_dir.name}  resample={'OFF' if resample_alpha is None else resample_alpha}...")
    train_transform, eval_transform = build_transforms(cfg)
    train_loader, val_loader, test_loader = get_dataloaders(
        cfg, split_dir, train_transform, eval_transform, resample_alpha=resample_alpha,
    )
    print(f"  Train: {len(train_loader.dataset):,} | Val: {len(val_loader.dataset):,} | "
          f"Test: {len(test_loader.dataset):,}")

    run_training(
        dataset=DATASET, method=method, output_dir=output_dir, cfg=cfg, hparam=hparam, seed=seed,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        build_model=lambda: ResNet18Pretrained(num_classes=1, freeze_backbone=args.freeze_backbone),
        unpack_fn=unpack_skin, forward_fn=forward_image_only,
        fairness_report_fn=_fairness_report,
        swad=args.swad,
    )


if __name__ == "__main__":
    main()
