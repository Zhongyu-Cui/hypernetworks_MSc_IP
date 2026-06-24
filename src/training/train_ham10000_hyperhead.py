"""
Train ResNet18-HyperHead on HAM10000 for Malignant Binary Classification (age-conditioned)
===========================================================================================
属性融合变体训练流程：在 image-only baseline (src/training/train_ham10000_resnet18.py) 的
基础上，把最后的全连接分类头替换为由**单一敏感属性 age_group** (20-40/40-60/60-80/80+ → 0–3)
驱动的 HyperHead 超网络 (src/models/resnet18_hyperhead.py 的 ResNet18HyperHeadAge)：backbone
与 baseline 完全一致 (ImageNet 预训练 torchvision ResNet-18, 共享 φ(X))，仅分类头逐样本由 age
生成。末层近零初始化、起步等价于与属性无关的统一头，可与 baseline 在相同预处理/早停下直接对比。

为什么 HyperHead + 只条件化 age：HAM 条件互信息诊断 (作业 68297) 显示 sex 轴 I(Y;sex|X)≈0、
age 轴读出可复现 I(Y;age|X)>0。而 HyperHead「在共享 φ(X) 上按 age 生成逐样本线性头」恰好等价
于诊断 E1 的 int 模型 (逐组独立线性头)——是与该正向信号最对口的架构，故先用 HyperHead 测 age。

与 HyperAdapt (train_ham10000_hyperadapt.py) 的差异：
  - 模型换成 ResNet18HyperHeadAge（只改 fc，backbone 共享；HyperAdapt 改每个 conv+fc）
  - 显存与 baseline 一致（逐样本只在 fc，非逐样本卷积核），故用原 batch_size=128（无需调小）

与 baseline (image-only) 训练脚本的区别：
  - 模型换成 ResNet18HyperHeadAge，训练/评估把 age 喂进模型 model(images, age)
  - ⚠️ **过滤 age_group>=0**：age embedding 是 nn.Embedding(4)，age_group==-1（0-20）非法索引，
    故 train/val/test 三 split 都过滤掉 -1（test 995→969）。与 baseline 的 age/worst-case 评估
    内部排除 -1 一致；唯一影响是 Overall AUC 改在 age_group>=0 子集上算，横比时在同一子集对账。
其余训练策略 (AdamW lr=1e-4/wd=1e-4/grad_clip、两种 model selection、早停监控 val Overall AUC、
双 checkpoint、Sex/Age/Sex×Age worst-case AUC 公平性报告、超参读 ham10000_baseline.yaml) 与
baseline 严格一致。

运行方式:
    重型任务，必须通过 sbatch 提交到 SLURM 集群 (见 slurm/train_ham10000_hyperhead.sh)。
    通过 --seed 指定随机种子（默认取 configs/ham10000_baseline.yaml 中 training.seeds[0]）。
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

from src.datasets.ham10000_dataset import HAM10000Dataset
from src.models.resnet18_hyperhead import ResNet18HyperHeadAge
from src.utils.ham10000_fairness import print_ham10000_fairness_report, worst_case_auc

# ============================================================
# 路径
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "ham10000_baseline.yaml"

# 模型权重属于「由集群生成的大文件」，按项目规范存放在仓库外的根目录 outputs/，不进仓库
OUTPUT_DIR = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/ham10000")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 配置加载
# ============================================================
def load_config(path: Path) -> dict:
    """读取 YAML 训练配置。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """解析命令行参数（--seed 多 seed 重复实验；--batch_size 可选覆盖，HyperHead 通常无需）。"""
    parser = argparse.ArgumentParser(
        description="Train ResNet18-HyperHead on HAM10000 (malignant, age-conditioned)"
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子；缺省时取 configs/ham10000_baseline.yaml 中 training.seeds[0]")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="覆盖 config 的 training.batch_size；HyperHead 只改 fc、显存与 baseline 一致，"
                             "通常无需覆盖（默认用 config 的 128，与 baseline 对齐）。")
    return parser.parse_args()


def build_transforms(cfg: dict) -> tuple[transforms.Compose, transforms.Compose]:
    """
    根据配置构建训练 / 评估 transform（与 HAM baseline 严格一致）。

    HAM10000 images/ 为自然 RGB（600×450, uint8），需先 resize 到 256×256 再裁剪到 224
    （对齐 MEDFAIR）。训练用 RandomCrop + 水平翻转 + 旋转增强；评估用 CenterCrop、无几何增强。
    """
    t_cfg = cfg["transforms"]
    norm_mean = t_cfg["normalize"]["mean"]
    norm_std = t_cfg["normalize"]["std"]
    train_c = t_cfg["train"]
    eval_c = t_cfg["eval"]

    train_ops: list = [
        transforms.Resize(tuple(train_c["resize"])),
        transforms.RandomCrop(train_c["random_crop"]),
    ]
    # 水平翻转：皮损无左右解剖约束（与 CXR 流程「不翻转」相反，与 MEDFAIR 一致）
    if train_c.get("horizontal_flip", False):
        train_ops.append(transforms.RandomHorizontalFlip())
    train_ops.append(transforms.RandomRotation(degrees=train_c["rotation_degrees"]))
    train_ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ]
    train_transform = transforms.Compose(train_ops)

    eval_transform = transforms.Compose([
        transforms.Resize(tuple(eval_c["resize"])),
        transforms.CenterCrop(eval_c["center_crop"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    return train_transform, eval_transform


def _filter_age_valid(dataset: HAM10000Dataset) -> int:
    """
    就地过滤掉 age_group==-1（0-20 排除组）的行，使数据集只剩 age_group∈{0..3}。

    age 条件化的 age embedding 是 nn.Embedding(4)，-1 不是合法索引；且 MEDFAIR 的 age 任务本就
    排除 0-20。直接对 HAM10000Dataset 内部并行数组应用布尔掩码——比改共享 Dataset 类更局部。

    Args:
        dataset: 待过滤的 HAM10000Dataset（就地修改）。

    Returns:
        过滤后剩余样本数。
    """
    mask = np.asarray(dataset.age_groups).astype(int) >= 0
    dataset.image_paths = dataset.image_paths[mask]
    dataset.labels = dataset.labels[mask]
    dataset.sexes = dataset.sexes[mask]
    dataset.age_groups = dataset.age_groups[mask]
    return int(mask.sum())


def get_dataloaders(
    cfg: dict,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    构建 train / val / test DataLoader（与 baseline 一致，额外过滤 age_group>=0）。

    三个 split 已由 scripts/build_ham10000_splits.py 按 lesion 级 80/10/10 划分写入独立 CSV，
    此处直接加载后过滤掉 age_group==-1（age 条件化模型不能吃 -1）。val / test 保持人群真实分布
    （除排除 -1 外不重采样）以确保评估指标可信。本实验不启用重采样（与「先做干净对照」一致）。
    """
    split_dir = REPO_ROOT / cfg["data"]["split_dir"]
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["dataloader"]["num_workers"]
    pin_memory = cfg["dataloader"]["pin_memory"]

    train_set = HAM10000Dataset(split_dir / "train.csv", transform=train_transform)
    val_set = HAM10000Dataset(split_dir / "val.csv", transform=eval_transform)
    test_set = HAM10000Dataset(split_dir / "test.csv", transform=eval_transform)

    # age 条件化：三个 split 都过滤 age_group>=0（embedding 不能吃 -1，且对齐 MEDFAIR age 任务）
    n_tr = _filter_age_valid(train_set)
    n_va = _filter_age_valid(val_set)
    n_te = _filter_age_valid(test_set)
    print(f"  过滤 age_group>=0 后: train {n_tr:,} / val {n_va:,} / test {n_te:,}（已排除 0-20）")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
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

    for i, (images, labels, _sex, ages) in enumerate(loader):
        # 与 baseline 不同：age 进入模型，作为 HyperHead 超网络的条件（sex 仍不进模型）
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(DEVICE)
        ages = ages.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images, ages)                 # [B, 1]
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        preds = (logits > 0).float()
        correct += preds.eq(labels).sum().item()
        total += batch_size

        if (i + 1) % 20 == 0:
            print(f"  [Epoch {epoch} | Batch {i + 1:4d}/{len(loader)}] "
                  f"loss={total_loss / total:.4f}  acc={100. * correct / total:.2f}%")

    return total_loss / total, 100. * correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float, float, float]:
    """
    在给定 loader 上评估，返回 (平均 loss, accuracy %, overall AUC, worst-case AUC)。
    age 既作为模型输入 (超网络条件)，又与 sex 一起作公平性分组键。

    worst_case_auc（Sex / Age / Sex×Age 子群 minimax）在无有效子群时返回 None，回落为 0.0。
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_logits, all_labels, all_sex, all_age = [], [], [], []

    for images, labels, sex, ages in loader:
        images = images.to(DEVICE, non_blocking=True)
        label_col = labels.float().unsqueeze(1).to(DEVICE)
        ages_d = ages.to(DEVICE, non_blocking=True)

        logits = model(images, ages_d)
        loss = criterion(logits, label_col)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        preds = (logits > 0).float()
        correct += preds.eq(label_col).sum().item()
        total += batch_size

        all_logits.append(logits.squeeze(1).cpu().numpy())
        all_labels.append(labels.numpy())
        all_sex.append(sex.numpy())
        all_age.append(ages.numpy())

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    all_sex = np.concatenate(all_sex)
    all_age = np.concatenate(all_age)

    auc = roc_auc_score(all_labels, all_logits)
    wc = worst_case_auc(all_labels, all_logits, all_sex, all_age)
    wc_auc = wc if wc is not None else 0.0

    return total_loss / total, 100. * correct / total, auc, wc_auc


@torch.no_grad()
def collect_predictions(
    model: nn.Module, loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    在整个 loader 上收集 logits 与 (label, sex, age_group)，供公平性评估使用。
    age 既是超网络条件、又是分组键；sex 仅作分组键（不进模型）。
    """
    model.eval()
    all_logits, all_labels, all_sex, all_age = [], [], [], []

    for images, labels, sex, ages in loader:
        images = images.to(DEVICE, non_blocking=True)
        ages_d = ages.to(DEVICE, non_blocking=True)
        logits = model(images, ages_d).squeeze(1)    # [B]
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())
        all_sex.append(sex.numpy())
        all_age.append(ages.numpy())

    return (
        np.concatenate(all_logits),
        np.concatenate(all_labels),
        np.concatenate(all_sex),
        np.concatenate(all_age),
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

    logits, labels, sex, age = collect_predictions(model, test_loader)
    print_ham10000_fairness_report(
        y_true=labels, y_score=logits, sex=sex, age_group=age,
        threshold=0.0,   # logits > 0 判正类（malignant）
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

    # CLI 可选覆盖 batch_size（HyperHead 通常无需；保留以便统一接口）
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size

    print(f"Device: {DEVICE}")
    print(f"Seed  : {seed}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # checkpoint 命名带 age 标识，区别于 baseline / hyperadapt
    ckpt_overall_path = OUTPUT_DIR / f"hyperhead_age_malignant_seed{seed}_best_overall.pth"
    ckpt_worstcase_path = OUTPUT_DIR / f"hyperhead_age_malignant_seed{seed}_best_worstcase.pth"

    print("\nLoading HAM10000 (malignant, age-conditioned HyperHead)...")
    train_transform, eval_transform = build_transforms(cfg)
    train_loader, val_loader, test_loader = get_dataloaders(cfg, train_transform, eval_transform)
    print(f"  Train: {len(train_loader.dataset):,} images")
    print(f"  Val  : {len(val_loader.dataset):,}")
    print(f"  Test : {len(test_loader.dataset):,}")

    # backbone 与 baseline 一致 (ImageNet 预训练)；HyperHead 末层近零、起步等价于统一头
    model = ResNet18HyperHeadAge(num_classes=1, num_age=4).to(DEVICE)
    n_bb = sum(p.numel() for p in model.backbone.parameters())
    n_hy = sum(p.numel() for p in model.hyper.parameters())
    print(f"\nModel : ResNet18-HyperHead (backbone=ImageNet 预训练共享, 分类头由 age 逐样本生成)")
    print(f"Params: {n_bb + n_hy:,} (backbone {n_bb:,} + hyper {n_hy:,})")
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
