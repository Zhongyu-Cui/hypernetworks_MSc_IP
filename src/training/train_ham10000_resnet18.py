"""
Train ResNet18 on HAM10000 for Malignant Binary Classification (Image-only, ERM)
================================================================================
仅以皮肤镜 RGB 图像作为输入（不引入 sex / age 属性），预测 malignant（0=benign, 1=malignant；
{akiec, mel}=恶性，恶性占比约 14.5%）。Backbone 为 ImageNet 预训练 ResNet-18。

**比较协议改造（A0.4）**：训练循环（早停 / 双 checkpoint / 逐 epoch 子群 AUC 向量日志 /
测试集公平性报告）统一走 src/training/harness/train_loop.run_training；本脚本只保留 HAM 特有的
transforms / dataloader（含可选重采样）构建，并注入回调：
  - unpack = unpack_sex_age（loader 返回 (image,label,sex,age_group)）
  - forward = forward_image_only（ERM 忽略属性，仅吃图像）
  - fairness = print_ham10000_fairness_report（Sex/Age/Sex×Age）

超参 lr/wd 由 `--config_index`（0–5）从超参网格取（协议 §3 的 6 配置搜索）；缺省 index=2 即
网格中心点 (lr=1e-4, wd=1e-4)，与改造前默认一致。checkpoint / val 日志按 run_id 命名，
超参搜索与多 seed 确认互不覆盖。

运行方式：重型任务，必须经 sbatch 提交到 SLURM 集群（见 slurm/train_ham10000_resnet18.sh）。
"""

import argparse
from pathlib import Path

import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.ham10000_dataset import HAM10000Dataset
from src.models.resnet18_pretrained import ResNet18Pretrained
from src.training.harness.hparam_grid import get_hparam_config, grid_size
from src.training.harness.run import seed_everything
from src.training.harness.train_loop import (
    forward_image_only, run_training, unpack_sex_age,
)
from src.utils.ham10000_fairness import print_ham10000_fairness_report
from src.utils.resampling import build_group_label_sampler

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "ham10000_baseline.yaml"
# 模型权重属于「由集群生成的大文件」，按项目规范存放在仓库外的根目录 outputs/，不进仓库
OUTPUT_DIR = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/ham10000")

DATASET = "ham10000"
METHOD = "erm"


def load_config(path: Path) -> dict:
    """读取 YAML 训练配置。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """解析命令行参数（--config_index 超参网格；--seed 多 seed；--resample_alpha 平衡重采样）。"""
    parser = argparse.ArgumentParser(description="Train ResNet18 on HAM10000 (malignant, image-only, ERM)")
    parser.add_argument("--config_index", type=int, default=2,
                        help=f"超参网格配置序号 0–{grid_size() - 1}（协议 §3 的 6 配置）；"
                             f"缺省 2=网格中心点 (lr=1e-4, wd=1e-4)。")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子；缺省时取 configs/ham10000_baseline.yaml 中 training.seeds[0]")
    parser.add_argument("--resample_alpha", type=float, default=None,
                        help="sex×age×label 平衡重采样强度 ∈[0,1]（0=自然分布，1=完全平衡）。"
                             "缺省不重采样。⚠️ age_group==-1（0-20）会破坏混合进制 cell 编码，"
                             "如按 age 维重采样须先在数据侧过滤 age_group>=0（见仓库 CLAUDE.md）。")
    parser.add_argument("--swad", action="store_true",
                        help="开启 SWAD 权重平均派生（协议 C1.7）：逐 epoch 缓存 CPU 权重，训练末在 loss "
                             "谷区间平均，额外产出 method=swad 的 checkpoint / val 日志，与 ERM 同 config/seed。")
    parser.add_argument("--cv", type=int, default=0,
                        help="K 折 lesion 级 GroupKFold（路线 A 用 5）；0=用单次 80/10/10 划分（默认）。")
    parser.add_argument("--fold", type=int, default=None,
                        help="--cv>0 时指定折号 k ∈ [0,K)。")
    return parser.parse_args()


def resolve_paths(cfg: dict, cv: int, fold: int | None) -> tuple[Path, Path]:
    """
    按 --cv/--fold 解析 (split_dir, output_dir)。CV 时 split 取 cv{K}/fold{k}，且 output_dir
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
    构建训练 / 评估 transform。HAM images/ 为自然 RGB（600×450, uint8），先 resize 到 256×256
    再裁剪到 224（对齐 MEDFAIR）；训练用 RandomCrop + 水平翻转（皮损无左右约束）+ 旋转增强。
    """
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
    cfg: dict,
    split_dir: Path,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
    resample_alpha: float | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    构建 train / val / test DataLoader（split 已由 build_ham10000_splits.py 按 lesion 级划分）。
    val/test 保持人群真实分布。resample_alpha 给定时对 train 用 WeightedRandomSampler 平衡（仅 train）。
    """
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["dataloader"]["num_workers"]
    pin_memory = cfg["dataloader"]["pin_memory"]

    train_set = HAM10000Dataset(split_dir / "train.csv", transform=train_transform)
    val_set = HAM10000Dataset(split_dir / "val.csv", transform=eval_transform)
    test_set = HAM10000Dataset(split_dir / "test.csv", transform=eval_transform)

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
    """HAM 公平性报告回调：从 attrs 取 sex / age(=age_group) 分组键。"""
    print_ham10000_fairness_report(
        y_true=y_true, y_score=y_score, sex=attrs["sex"], age_group=attrs["age"], threshold=0.0,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(CONFIG_PATH)
    seed = args.seed if args.seed is not None else cfg["training"]["seeds"][0]
    hparam = get_hparam_config(args.config_index)
    split_dir, output_dir = resolve_paths(cfg, args.cv, args.fold)

    # 重采样：CLI 优先；缺省回落 config（仅当 resampling.enabled）
    resample_alpha = args.resample_alpha
    if resample_alpha is None and cfg.get("resampling", {}).get("enabled", False):
        resample_alpha = cfg["resampling"]["alpha"]

    # 播种须在构建 loader（shuffle/sampler 依赖 RNG）之前
    seed_everything(seed)
    print(f"Loading HAM10000 (malignant, image-only ERM)  split_dir={split_dir.name}  "
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
        unpack_fn=unpack_sex_age, forward_fn=forward_image_only,
        fairness_report_fn=_fairness_report,
        swad=args.swad,
    )


if __name__ == "__main__":
    main()
