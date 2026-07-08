"""
合成信号剂量-反应训练：MIMIC-CXR + 注入侧信道属性 A_syn（评审补强 R1.3）
=====================================================================
在 MIMIC-CXR No Finding 二分类上，以**注入的合成属性 A_syn = Y ⊕ Bernoulli(η)** 作为敏感属性，
统一 regime 训练 ERM + 3 HN，检验 HN 相对 attribute-blind ERM 的增益是否随信号强度
`I(Y;A_syn|X)`（由 η 经 R1.2 标定）单调上升——R1 的充分性因果检验。

**方法 × A_syn 用法**：
  - erm        : image-only，**忽略 A_syn**（attribute-blind 参照零点）。ERM 训练与 η 无关
    （不吃 A_syn），故 4 档共用同一 ERM——通常只在负控制档 η=0.5 训一次、R1.4 叠加各档 A_syn 复用。
  - hyperhead  : 仅分类头由 A_syn 经超网络生成（最浅 HN）。
  - hyperfusion: 仅 layer4[0].downsample 注入 A_syn（中层单点 HN）。
  - hyperadapt : 除 stem 外各层低秩残差由 A_syn 调制（最深 HN）。
  三 HN 均复用单一二值属性通路 `*Age(num_age=2)`，A_syn 走 age 位置。

**统一 regime**：复用 A0 harness run_training（AdamW + BCE + grad_clip + 双 checkpoint +
子群 AUC 向量日志 + 落盘预测）。超参 lr/wd 由 --config_index 取网格；子群 = A_syn 两组
（dataset="mimic_synth"）。每档一个 output 子目录 outputs/mimic_synth/eta<η>，使不同档位的
checkpoint/预测互不覆盖（run_id 不含 η）。

运行：重型任务，经 sbatch 提交（slurm/r1_synth_search.sh / r1_synth_confirm.sh）。
"""

import argparse
from pathlib import Path

import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.mimic_synth_dataset import MIMICSynthDataset
from src.models.resnet18_pretrained import ResNet18Pretrained
from src.models.resnet18_hyperhead import ResNet18HyperHeadAge
from src.models.resnet18_hyperfusion import ResNet18HyperFusionAge
from src.models.resnet18_hyperadapt import ResNet18HyperAdaptAge
from src.training.harness.hparam_grid import get_hparam_config, grid_size
from src.training.harness.run import seed_everything
from src.training.harness.train_loop import (
    forward_asyn, forward_image_only, run_training, unpack_asyn,
)
from src.utils.synthetic_fairness import print_synthetic_fairness_report

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "mimic_cxr_baseline.yaml"
OUTPUT_ROOT = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/mimic_synth")

DATASET = "mimic_synth"
NUM_ASYN = 2  # A_syn 二值

# 方法 → (模型工厂, forward 回调)。ERM 忽略 A_syn；HN 复用 num_age=2 单属性通路。
METHOD_REGISTRY = {
    "erm":         (lambda: ResNet18Pretrained(num_classes=1), forward_image_only),
    "hyperhead":   (lambda: ResNet18HyperHeadAge(num_classes=1, num_age=NUM_ASYN), forward_asyn),
    "hyperfusion": (lambda: ResNet18HyperFusionAge(num_classes=1, num_age=NUM_ASYN), forward_asyn),
    "hyperadapt":  (lambda: ResNet18HyperAdaptAge(num_classes=1, num_age=NUM_ASYN), forward_asyn),
}


def load_config(path: Path) -> dict:
    """读取 YAML 训练配置（复用 MIMIC baseline 的 transforms / training / early_stopping）。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """解析命令行参数（--method 方法；--eta 信号档翻转率；--config_index；--seed；--batch_size；--swad）。"""
    parser = argparse.ArgumentParser(description="Train ERM/HN on MIMIC-CXR with synthetic side-channel A_syn (R1)")
    parser.add_argument("--method", type=str, required=True, choices=list(METHOD_REGISTRY),
                        help="erm / hyperhead / hyperfusion / hyperadapt")
    parser.add_argument("--eta", type=float, required=True,
                        help="A_syn 翻转率 η（信号档，见 R1.2 标定表：0.5/0.3621/0.2942/0.2096）")
    parser.add_argument("--config_index", type=int, default=2,
                        help=f"超参网格配置序号 0–{grid_size() - 1}；缺省 2=中心点 (lr=1e-4, wd=1e-4)。")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子；缺省取 mimic_cxr_baseline.yaml 的 training.seeds[0]")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="覆盖 config 的 batch_size（HyperAdapt 等 batch 对照可用）。")
    parser.add_argument("--swad", action="store_true",
                        help="额外派生 SWAD 权重平均（仅 ERM 有意义）。")
    return parser.parse_args()


def build_transforms(cfg: dict) -> tuple[transforms.Compose, transforms.Compose]:
    """构建训练 / 评估 transform（与 MIMIC baseline 一致：Grayscale(3)、不水平翻转）。"""
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
    cfg: dict, eta: float, train_transform: transforms.Compose, eval_transform: transforms.Compose,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建 train/val/test DataLoader，各样本附加确定性 A_syn(η)（val/test 保持真实分布，不重采样）。"""
    split_dir = REPO_ROOT / cfg["data"]["split_dir"]
    image_size = cfg["data"]["image_size"]
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["dataloader"]["num_workers"]
    pin_memory = cfg["dataloader"]["pin_memory"]

    def _ds(split: str, tf: transforms.Compose) -> MIMICSynthDataset:
        return MIMICSynthDataset(split_dir / f"{split}.csv", eta=eta, split=split,
                                 transform=tf, image_size=image_size)

    train_set, val_set, test_set = _ds("train", train_transform), _ds("val", eval_transform), _ds("test", eval_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


def eta_tag(eta: float) -> str:
    """信号档目录标签（4 位小数，稳定命名）：eta0.5000 / eta0.3621 / ...。"""
    return f"eta{eta:.4f}"


def _fairness_report(y_true, y_score, attrs) -> None:
    """合成信号公平性报告回调：子群 = A_syn。"""
    print_synthetic_fairness_report(y_true=y_true, y_score=y_score, a_syn=attrs["a_syn"], threshold=0.0)


def main() -> None:
    args = parse_args()
    cfg = load_config(CONFIG_PATH)
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    seed = args.seed if args.seed is not None else cfg["training"]["seeds"][0]
    hparam = get_hparam_config(args.config_index)
    build_model, forward_fn = METHOD_REGISTRY[args.method]
    output_dir = OUTPUT_ROOT / eta_tag(args.eta)

    seed_everything(seed)
    print(f"Loading MIMIC-synth (method={args.method}, η={args.eta}, A_syn side-channel)...")
    train_transform, eval_transform = build_transforms(cfg)
    train_loader, val_loader, test_loader = get_dataloaders(cfg, args.eta, train_transform, eval_transform)
    # 报告经验翻转率，确认 A_syn 注入强度符合该档
    emp = float((train_loader.dataset.a_syn != train_loader.dataset.labels).mean())
    print(f"  train A_syn 经验翻转率 = {emp:.4f}（目标 η={args.eta}）；输出目录 {output_dir}")

    run_training(
        dataset=DATASET, method=args.method, output_dir=output_dir, cfg=cfg, hparam=hparam, seed=seed,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        build_model=build_model, unpack_fn=unpack_asyn, forward_fn=forward_fn,
        fairness_report_fn=_fairness_report, swad=args.swad,
    )


if __name__ == "__main__":
    main()
