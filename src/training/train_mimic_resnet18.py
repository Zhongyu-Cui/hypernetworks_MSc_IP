"""
Train ResNet18 on MIMIC-CXR for No Finding Binary Classification (Image-only)
==============================================================================
端到端 baseline 训练流程: 仅以胸部 X 光图像作为输入 (不引入 sex / race / age)，
预测 No Finding (U-Zeros 标签策略：0=有病灶，1=无病灶/健康)。

Backbone 为 ImageNet 预训练 ResNet-18 (src/models/resnet18_pretrained.py)。
训练过程中同时按两种 model selection 策略保存 checkpoint：
  - best_overall  : 验证集 Overall AUC 最高 (用于早停判据)
  - best_worstcase: 验证集 Worst-case AUC (Sex/Race/Age/交叉组 minimax) 最高
两者训练结束后均在测试集上输出整体指标与分组公平性报告，用于对比
model selection 策略对公平性的影响。

超参数、transform 均从 configs/mimic_cxr_baseline.yaml 读取（AdamW + 较小学习率
+ weight decay + 梯度裁剪）；早停监控验证集 Overall AUC（对类别不平衡鲁棒，与公平性
报告分组 AUC 口径一致）。

运行方式:
    重型任务，必须通过 sbatch 提交到 SLURM 集群 (见 slurm/train_mimic_resnet18.sh)。
    通过 --seed 指定随机种子（用于多 seed 重复实验，默认取
    configs/mimic_cxr_baseline.yaml 中 training.seeds 的第一个值）。
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.mimic_cxr_dataset import MIMICCXRDataset
from src.models.resnet18_pretrained import ResNet18Pretrained
from src.utils.mimic_fairness import print_mimic_fairness_report, worst_case_auc
from src.utils.resampling import build_group_label_sampler

# ============================================================
# 路径
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "mimic_cxr_baseline.yaml"

# 模型权重属于「由集群生成的大文件」，按项目规范存放在仓库外的根目录 outputs/，不进仓库
OUTPUT_DIR = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/mimic_cxr")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 配置加载
# ============================================================
def load_config(path: Path) -> dict:
    """读取 YAML 训练配置。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """解析命令行参数（目前仅 --seed，用于多 seed 重复实验）。"""
    parser = argparse.ArgumentParser(description="Train ResNet18 on MIMIC-CXR (No Finding, image-only)")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子；缺省时取 configs/mimic_cxr_baseline.yaml 中 training.seeds[0]")
    parser.add_argument("--resample_alpha", type=float, default=None,
                        help="敏感组×标签平衡重采样强度 ∈[0,1]（0=自然分布，1=完全平衡）。"
                             "缺省时不重采样，train 仍用 shuffle=True，行为与原 baseline 一致；"
                             "给定时按该 alpha 启用 WeightedRandomSampler，平衡维度取 config 的 resampling.dims。")
    return parser.parse_args()


def build_transforms(cfg: dict) -> tuple[transforms.Compose, transforms.Compose]:
    """
    根据配置构建训练 / 评估 transform。

    CXR7-1M 已预处理至 224×224，直接在此尺寸上做旋转增强。
    不使用水平翻转（胸片有固定解剖左右关系）。
    """
    t_cfg     = cfg["transforms"]
    norm_mean = t_cfg["normalize"]["mean"]
    norm_std  = t_cfg["normalize"]["std"]

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
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
    resample_alpha: float | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    构建 train / val / test DataLoader。

    三个 split 已由 scripts/build_mimic_splits_nofinding.py 按患者级划分写入独立 CSV
    (data/splits/mimic_cxr_nofinding/{train,val,test}.csv)，此处直接加载，无需再次切分。
    val / test 保持人群真实分布以确保评估指标可信。

    Args:
        resample_alpha: 若为 None（默认），train 用 shuffle=True，与原 baseline 一致；
            若给定，则用 WeightedRandomSampler 按 config 的 resampling.dims 做敏感组×标签
            平衡（强度 alpha），仅作用于 train（val/test 不变）。
    """
    split_dir = REPO_ROOT / cfg["data"]["split_dir"]
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

    # 重采样仅改变 train 的取样方式：传 sampler 时必须 shuffle=False（二者互斥）
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


# ============================================================
# 训练 / 评估
# ============================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    grad_clip_norm: float,
    epoch: int,
) -> tuple[float, float]:
    """跑一个训练 epoch，返回 (平均 loss, accuracy %)。"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels, _sexes, _races, _ages) in enumerate(loader):
        # sex / race / age_group 不进入 image-only 模型，仍从 loader 解包以保持接口一致
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)                      # [B, 1]
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        preds = (logits > 0).float()
        correct += preds.eq(labels).sum().item()
        total += batch_size

        if (i + 1) % 200 == 0:
            print(f"  [Epoch {epoch} | Batch {i + 1:5d}/{len(loader)}] "
                  f"loss={total_loss / total:.4f}  acc={100. * correct / total:.2f}%")

    return total_loss / total, 100. * correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float, float, float]:
    """
    在给定 loader 上评估，返回 (平均 loss, accuracy %, overall AUC, worst-case AUC)。

    Overall AUC 用于「Overall AUC 选择」early stopping 判据；worst-case AUC
    （Sex/Race/Age/交叉组 minimax）用于「Worst-case AUC 选择」判据，
    两者口径与下方分组公平性报告一致。
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_logits, all_labels, all_sexes, all_races, all_ages = [], [], [], [], []

    for images, labels, sexes, races, ages in loader:
        images = images.to(DEVICE, non_blocking=True)
        label_col = labels.float().unsqueeze(1).to(DEVICE)

        logits = model(images)
        loss = criterion(logits, label_col)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        preds = (logits > 0).float()
        correct += preds.eq(label_col).sum().item()
        total += batch_size

        all_logits.append(logits.squeeze(1).cpu().numpy())
        all_labels.append(labels.numpy())
        all_sexes.append(sexes.numpy())
        all_races.append(races.numpy())
        all_ages.append(ages.numpy())

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    all_sexes = np.concatenate(all_sexes)
    all_races = np.concatenate(all_races)
    all_ages = np.concatenate(all_ages)

    auc = roc_auc_score(all_labels, all_logits)
    wc_auc = worst_case_auc(all_labels, all_logits, all_sexes, all_races, all_ages)

    return total_loss / total, 100. * correct / total, auc, wc_auc


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    在整个 loader 上收集 logits 与 (label, sex, race, age_group)，供公平性评估使用。
    sex / race / age_group 不进入模型，仅作为分组键，用来观察 image-only 模型在
    不同敏感属性子群上的表现差异 (隐式偏差)。
    """
    model.eval()
    all_logits, all_labels, all_sexes, all_races, all_ages = [], [], [], [], []

    for images, labels, sexes, races, ages in loader:
        images = images.to(DEVICE, non_blocking=True)
        logits = model(images).squeeze(1)            # [B]
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())
        all_sexes.append(sexes.numpy())
        all_races.append(races.numpy())
        all_ages.append(ages.numpy())

    return (
        np.concatenate(all_logits),
        np.concatenate(all_labels),
        np.concatenate(all_sexes),
        np.concatenate(all_races),
        np.concatenate(all_ages),
    )


@torch.no_grad()
def evaluate_checkpoint(
    model: nn.Module,
    ckpt_path: Path,
    test_loader: DataLoader,
    criterion: nn.Module,
    label: str,
) -> None:
    """加载指定 checkpoint，在测试集上评估并打印分组公平性报告。"""
    print(f"\n{'#' * 70}\n# Evaluating checkpoint: {label} ({ckpt_path.name})\n{'#' * 70}")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    test_loss, test_acc, test_auc, test_wc_auc = evaluate(model, test_loader, criterion)
    print(f"Test loss={test_loss:.4f}  accuracy={test_acc:.2f}%  AUC={test_auc:.4f}  "
          f"Worst-case AUC={test_wc_auc:.4f}")

    logits, labels, sexes, races, ages = collect_predictions(model, test_loader)
    print_mimic_fairness_report(
        y_true=labels, y_score=logits,
        sex=sexes, race=races, age=ages,
        threshold=0.0,   # logits > 0 判正类
    )


# ============================================================
# 主函数
# ============================================================
def main() -> None:
    args = parse_args()
    cfg = load_config(CONFIG_PATH)
    train_cfg = cfg["training"]
    es_cfg = cfg["early_stopping"]

    seed = args.seed if args.seed is not None else train_cfg["seeds"][0]
    torch.manual_seed(seed)

    # 重采样：CLI --resample_alpha 优先；缺省时回落到 config（仅当 resampling.enabled 时启用）
    resample_alpha = args.resample_alpha
    if resample_alpha is None and cfg.get("resampling", {}).get("enabled", False):
        resample_alpha = cfg["resampling"]["alpha"]
    # checkpoint 命名加 resample tag，避免覆盖原 baseline (resnet18_no_finding_seed*) 权重
    tag = f"_resample_a{resample_alpha}" if resample_alpha is not None else ""

    if resample_alpha is None:
        resample_status = "OFF (shuffle=True)"
    else:
        resample_status = f"ON  alpha={resample_alpha}  dims={cfg['resampling']['dims']}"

    print(f"Device: {DEVICE}")
    print(f"Seed  : {seed}")
    print(f"Resampling: {resample_status}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_overall_path = OUTPUT_DIR / f"resnet18_no_finding{tag}_seed{seed}_best_overall.pth"
    ckpt_worstcase_path = OUTPUT_DIR / f"resnet18_no_finding{tag}_seed{seed}_best_worstcase.pth"

    print("\nLoading MIMIC-CXR (No Finding, U-Zeros)...")
    train_transform, eval_transform = build_transforms(cfg)
    train_loader, val_loader, test_loader = get_dataloaders(
        cfg, train_transform, eval_transform, resample_alpha=resample_alpha,
    )
    print(f"  Train: {len(train_loader.dataset):,} images")
    print(f"  Val  : {len(val_loader.dataset):,}")
    print(f"  Test : {len(test_loader.dataset):,}")

    model = ResNet18Pretrained(num_classes=1).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel : ResNet18 (image-only, ImageNet 预训练)")
    print(f"Params: {n_params:,}")
    print(f"Optim : AdamW(lr={train_cfg['learning_rate']}, weight_decay={train_cfg['weight_decay']})")
    print(f"Misc  : grad_clip_norm={train_cfg['grad_clip_norm']}, batch_size={train_cfg['batch_size']}")
    print(f"Early stopping: monitor={es_cfg['monitor']} (mode={es_cfg['mode']}), "
          f"patience={es_cfg['patience']}, min_delta={es_cfg['min_delta']}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])

    best_val_auc = -np.inf
    best_val_wc_auc = -np.inf
    epochs_no_improve = 0
    max_epochs = train_cfg["num_epochs"]
    patience = es_cfg["patience"]
    min_delta = es_cfg["min_delta"]

    print(f"\nTraining for up to {max_epochs} epochs (early stopping enabled)...\n" + "=" * 70)
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, train_cfg["grad_clip_norm"], epoch,
        )
        val_loss, val_acc, val_auc, val_wc_auc = evaluate(model, val_loader, criterion)
        dt = time.time() - t0

        print(f"Epoch {epoch:2d}/{max_epochs} [{dt:.1f}s] | "
              f"train loss={train_loss:.4f} acc={train_acc:.2f}% | "
              f"val loss={val_loss:.4f} acc={val_acc:.2f}% AUC={val_auc:.4f} "
              f"worst-case AUC={val_wc_auc:.4f}")

        # --- model selection: best worst-case AUC checkpoint ---
        if val_wc_auc > best_val_wc_auc + min_delta:
            best_val_wc_auc = val_wc_auc
            torch.save(model.state_dict(), ckpt_worstcase_path)
            print(f"  -> New best val worst-case AUC ({best_val_wc_auc:.4f}); "
                  f"checkpoint saved to {ckpt_worstcase_path}")

        # --- model selection + early stopping: best overall AUC checkpoint ---
        if val_auc > best_val_auc + min_delta:
            best_val_auc = val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_overall_path)
            print(f"  -> New best val overall AUC ({best_val_auc:.4f}); "
                  f"checkpoint saved to {ckpt_overall_path}")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement for {epochs_no_improve}/{patience} epoch(s) "
                  f"(best overall so far: {best_val_auc:.4f})")
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch} "
                      f"(best val overall AUC = {best_val_auc:.4f}, "
                      f"best val worst-case AUC = {best_val_wc_auc:.4f}).")
                break
        print("-" * 70)

    # --- 用两种 model selection 策略的最优 checkpoint 在测试集上评估 ---
    evaluate_checkpoint(model, ckpt_overall_path, test_loader, criterion, label="Overall AUC selection")
    evaluate_checkpoint(model, ckpt_worstcase_path, test_loader, criterion, label="Worst-case AUC selection")


if __name__ == "__main__":
    main()
