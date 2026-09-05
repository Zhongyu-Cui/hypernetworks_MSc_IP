"""
Train ResNet18-HyperHead on Fitzpatrick17k (malignant, skin-conditioned)
========================================================================
在 image-only baseline 基础上，仅让**分类头**由**单一敏感属性 skin**（Fitzpatrick I–VI → 0–5）
经超网络逐样本生成（ResNet18HyperHeadSkin，末层近零初始、起步等价统一头），backbone 与 baseline
共享。HyperHead 是「最浅」HN（只改 fc），与 HyperFusion（单点深层）、HyperAdapt（每层低秩）构成
注入深度对照。注：Fitzpatrick 条件互信息 I(Y;skin|X)≈0（作业 67913），据核心论点 HN 在此注定无
收益，本脚本产出「信号缺失处的 null result」对照。

**比较协议改造（A0.4）**：训练循环统一走 harness.run_training；本脚本保留 Fitzpatrick 特有的
transforms / dataloader，注入 unpack_skin + forward_skin + print_fitzpatrick_fairness_report。
超参 lr/wd 由 --config_index 从网格取（缺省 index=2=中心点）。HyperHead 只改 fc、显存与 baseline
一致，用原 batch_size=128。

运行方式：重型任务，经 sbatch 提交（见 slurm/train_fitzpatrick_hyperhead.sh）。
"""

import argparse
from pathlib import Path

import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.fitzpatrick_dataset import FitzpatrickDataset
from src.models.resnet18_hyperhead import ResNet18HyperHeadSkin
from src.training.harness.hparam_grid import get_hparam_config, grid_size
from src.training.harness.run import seed_everything
from src.training.harness.train_loop import forward_skin, run_training, unpack_skin
from src.utils.fitzpatrick_fairness import print_fitzpatrick_fairness_report
from src.paths import OUTPUTS_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "fitzpatrick_baseline.yaml"
OUTPUT_DIR = OUTPUTS_DIR / "fitzpatrick"

DATASET = "fitzpatrick"
METHOD = "hyperhead"
NUM_SKIN = 6   # Fitzpatrick 肤色 I–VI → 0..5


def load_config(path: Path) -> dict:
    """读取 YAML 训练配置。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """解析命令行参数（--config_index 超参网格；--seed；--batch_size 可选覆盖）。"""
    parser = argparse.ArgumentParser(description="Train ResNet18-HyperHead on Fitzpatrick17k (skin-conditioned)")
    parser.add_argument("--config_index", type=int, default=2,
                        help=f"超参网格配置序号 0–{grid_size() - 1}；缺省 2=中心点 (lr=1e-4, wd=1e-4)。")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子；缺省取 fitzpatrick_baseline.yaml 的 training.seeds[0]")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="覆盖 config 的 batch_size；HyperHead 只改 fc、显存同 baseline，通常无需。")
    parser.add_argument("--swad", action="store_true",
                        help="开启 SWAD 权重平均派生（**实验 G**：HN×SWAD 融合，方案见 "
                             "docs/hyperadapt_swad_fusion_plan.md）：额外产出 method=hyperhead_swad 的 "
                             "checkpoint / val 日志 / test 预测，与本 run 同 config/seed，"
                             "与 ERM 派生的 SWAD 基线（method=swad）隔离。")
    parser.add_argument("--cv", type=int, default=0,
                        help="K 折 StratifiedKFold（CV-OOF 用 5）；0=用单次 80/10/10 划分（默认）。")
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
    cfg: dict, split_dir: Path, train_transform: transforms.Compose, eval_transform: transforms.Compose,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建 train / val / test DataLoader（split 已图像级划分；val/test 保持真实分布，HN 不重采样）。"""
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

    split_dir, output_dir = resolve_paths(cfg, args.cv, args.fold)

    seed_everything(seed)
    print(f"Loading Fitzpatrick17k (malignant, skin-conditioned HyperHead)  "
          f"split_dir={split_dir.name}  output_dir={output_dir.name}...")
    train_transform, eval_transform = build_transforms(cfg)
    train_loader, val_loader, test_loader = get_dataloaders(cfg, split_dir, train_transform, eval_transform)

    run_training(
        dataset=DATASET, method=METHOD, output_dir=output_dir, cfg=cfg, hparam=hparam, seed=seed,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        build_model=lambda: ResNet18HyperHeadSkin(num_classes=1, num_skin=NUM_SKIN),
        unpack_fn=unpack_skin, forward_fn=forward_skin,
        fairness_report_fn=_fairness_report,
        swad=args.swad,
    )


if __name__ == "__main__":
    main()
