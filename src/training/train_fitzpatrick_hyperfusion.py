"""
Train ResNet18-HyperFusion on Fitzpatrick17k (malignant, skin-conditioned)
==========================================================================
仅在 backbone 的 layer4[0].downsample 处接由**单一敏感属性 skin**（Fitzpatrick I–VI → 0–5）驱动的
HyperFusion 超网络（ResNet18HyperFusionSkin，num_skin=6，MIP additive + E_L2 投影，初始 Δθ≈0）。
HyperFusion 是「中层单点」HN（仅末 block downsample），介入深度介于 HyperHead（最浅）与 HyperAdapt
（最遍布）之间。注：Fitzpatrick 条件互信息 I(Y;skin|X)≈0（作业 67913），据核心论点 HN 在此注定无
收益，本脚本产出「信号缺失处的 null result」对照。

**比较协议改造（A0.4）**：训练循环统一走 harness.run_training；本脚本保留 Fitzpatrick 特有的
transforms / dataloader，注入 unpack_skin + forward_skin + print_fitzpatrick_fairness_report。
超参 lr/wd 由 --config_index 从网格取（缺省 index=2=中心点）。HyperFusion 单点注入显存接近 baseline，
用原 batch_size=128。

运行方式：重型任务，经 sbatch 提交（见 slurm/train_fitzpatrick_hyperfusion.sh）。
"""

import argparse
from pathlib import Path

import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.fitzpatrick_dataset import FitzpatrickDataset
from src.models.resnet18_hyperfusion import ResNet18HyperFusionSkin
from src.training.harness.hparam_grid import get_hparam_config, grid_size
from src.training.harness.run import seed_everything
from src.training.harness.train_loop import forward_skin, run_training, unpack_skin
from src.utils.fitzpatrick_fairness import print_fitzpatrick_fairness_report

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "fitzpatrick_baseline.yaml"
OUTPUT_DIR = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/fitzpatrick")

DATASET = "fitzpatrick"
METHOD = "hyperfusion"
NUM_SKIN = 6   # Fitzpatrick 肤色 I–VI → 0..5


def load_config(path: Path) -> dict:
    """读取 YAML 训练配置。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """解析命令行参数（--config_index 超参网格；--seed；--batch_size 可选覆盖）。"""
    parser = argparse.ArgumentParser(description="Train ResNet18-HyperFusion on Fitzpatrick17k (skin-conditioned)")
    parser.add_argument("--config_index", type=int, default=2,
                        help=f"超参网格配置序号 0–{grid_size() - 1}；缺省 2=中心点 (lr=1e-4, wd=1e-4)。")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子；缺省取 fitzpatrick_baseline.yaml 的 training.seeds[0]")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="覆盖 config 的 batch_size；HyperFusion 单点注入显存近 baseline，通常无需。")
    return parser.parse_args()


def build_transforms(cfg: dict) -> tuple[transforms.Compose, transforms.Compose]:
    """构建训练 / 评估 transform（与 Fitzpatrick baseline 一致；224 原生，翻转+旋转，无 Resize）。"""
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
    cfg: dict, train_transform: transforms.Compose, eval_transform: transforms.Compose,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建 train / val / test DataLoader（split 已图像级划分；val/test 保持真实分布，HN 不重采样）。"""
    split_dir = REPO_ROOT / cfg["data"]["split_dir"]
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["dataloader"]["num_workers"]
    pin_memory = cfg["dataloader"]["pin_memory"]

    train_set = FitzpatrickDataset(split_dir / "train.csv", transform=train_transform)
    val_set = FitzpatrickDataset(split_dir / "val.csv", transform=eval_transform)
    test_set = FitzpatrickDataset(split_dir / "test.csv", transform=eval_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
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
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    seed = args.seed if args.seed is not None else cfg["training"]["seeds"][0]
    hparam = get_hparam_config(args.config_index)

    seed_everything(seed)
    print("Loading Fitzpatrick17k (malignant, skin-conditioned HyperFusion)...")
    train_transform, eval_transform = build_transforms(cfg)
    train_loader, val_loader, test_loader = get_dataloaders(cfg, train_transform, eval_transform)

    run_training(
        dataset=DATASET, method=METHOD, output_dir=OUTPUT_DIR, cfg=cfg, hparam=hparam, seed=seed,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        build_model=lambda: ResNet18HyperFusionSkin(
            num_classes=1, num_skin=NUM_SKIN, pretrained=cfg["model"]["pretrained"]),
        unpack_fn=unpack_skin, forward_fn=forward_skin,
        fairness_report_fn=_fairness_report,
    )


if __name__ == "__main__":
    main()
