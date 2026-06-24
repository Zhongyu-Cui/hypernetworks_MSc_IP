"""
Train ResNet18-HyperAdapt on HAM10000 for Malignant Binary Classification (age-conditioned)
============================================================================================
属性融合变体训练流程：在 image-only baseline (src/training/train_ham10000_resnet18.py) 的
基础上，把 backbone 的每个卷积层 (通道级乘性调制) 与分类头 fc (加性低秩更新) 都接上由**单一
敏感属性 age_group** (20-40/40-60/60-80/80+ → 0–3) 驱动的 HyperAdapt 超网络
(src/models/resnet18_hyperadapt.py 的 ResNet18HyperAdaptAge)。backbone 与 baseline 完全一致
(ImageNet 预训练 torchvision ResNet-18)，初始化保证 Δθ≈0、起始等价于纯预训练 backbone，因此
可与 baseline 在相同预处理、相同早停设置下直接对比公平性。

为什么只条件化 age：HAM 条件互信息诊断 (作业 68297, outputs/ham10000/1st_conditional_mi.md)
显示 sex 轴 I(Y;sex|X)≈0 (弱/null)、而 age 轴读出小但 3-seed 可复现的 I(Y;age|X)>0
(E1 ΔNLL 3 seed CI 下界全>0；E2 模型无关 χ² 3 seed 全显著)。故先只接 age——sex 轴无信号、不接。

与 baseline (image-only) 训练脚本的区别：
  - 模型换成 ResNet18HyperAdaptAge（注入到每个 conv + fc，由 age 逐样本调制）
  - 训练 / 评估循环把 age 喂进模型 model(images, age)（baseline 中 age 仅作分组键）
  - ⚠️ **过滤 age_group>=0**：age embedding 是 nn.Embedding(4)，age_group==-1（0-20 排除组）
    不是合法索引，故 train/val/test 三个 split 都先过滤掉 -1 行 (_filter_age_valid)。
    这与 baseline 的「age/worst-case 评估内部排除 -1」一致；唯一影响是 Overall AUC 改在
    age_group>=0 子集上算 (test 995→~969)，与 baseline 横比时应在同一过滤子集上对账。
其余训练策略 (AdamW + lr=1e-4 + weight_decay + 梯度裁剪、两种 model selection、早停监控
val Overall AUC、双 checkpoint best_overall / best_worstcase、Sex/Age/Sex×Age 子群 worst-case
AUC 公平性报告、超参均读 configs/ham10000_baseline.yaml) 与 baseline 严格保持一致。

⚠️ 显存提醒：HyperAdapt 用 grouped-conv 展开「逐样本卷积核」(把权重复制 B 份)，显存占用显著
高于 baseline。为放进 gpus24 (24GiB)，slurm 脚本用 --batch_size 32 覆盖 config 的 128
（仅改 batch_size，lr 等训练策略不变）；若想用原 bs=128 则改投 gpus48。

运行方式:
    重型任务，必须通过 sbatch 提交到 SLURM 集群 (见 slurm/train_ham10000_hyperadapt.sh)。
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
from src.models.resnet18_hyperadapt import ResNet18HyperAdaptAge
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
    """解析命令行参数（--seed 多 seed 重复实验；--batch_size 覆盖显存）。"""
    parser = argparse.ArgumentParser(
        description="Train ResNet18-HyperAdapt on HAM10000 (malignant, age-conditioned)"
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子；缺省时取 configs/ham10000_baseline.yaml 中 training.seeds[0]")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="覆盖 config 的 training.batch_size；HyperAdapt 逐样本卷积核显存大，"
                             "在 gpus24 (24GiB) 上需调小（如 32）以免 OOM。缺省时用 config 值（128）。"
                             "仅改 batch_size，lr 等训练策略不变（与 baseline 对齐）。")
    parser.add_argument("--ckpt_suffix", type=str, default="",
                        help="checkpoint 文件名追加标识（如 '_b128'），避免不同 batch_size 重跑互相覆盖。"
                             "缺省为空，保持原命名 hyperadapt_age_malignant_seed*。")
    return parser.parse_args()


def build_transforms(cfg: dict) -> tuple[transforms.Compose, transforms.Compose]:
    """
    根据配置构建训练 / 评估 transform（与 HAM baseline 严格一致）。

    HAM10000 images/ 为自然 RGB（600×450, uint8），需先 resize 到 256×256 再裁剪到 224
    （对齐 MEDFAIR；区别于 Fitzpatrick 源图已 224 的「224 原生」）。训练用 RandomCrop +
    水平翻转 + 旋转增强；评估用 CenterCrop、无几何增强。
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

    age 条件化 HyperAdapt 的 AgeEmbedding 是 nn.Embedding(4)，-1 不是合法索引；且 MEDFAIR
    的 age 任务本就排除 0-20。直接对 HAM10000Dataset 内部并行数组 (image_paths/labels/
    sexes/age_groups) 应用布尔掩码——比改共享 Dataset 类更局部、只作用于本实验。

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
    此处直接加载后过滤掉 age_group==-1（age 条件化模型不能吃 -1）。val / test 保持人群真实
    分布（除排除 -1 外不重采样）以确保评估指标可信。本实验不启用重采样（与「先做干净对照」一致）。
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
        # 与 baseline 不同：age 进入模型，作为 HyperAdapt 超网络的条件（sex 仍不进模型）
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

    # CLI 覆盖 batch_size（不动共享 baseline yaml）：gpus24 上 HyperAdapt 需调小以免 OOM
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size

    print(f"Device: {DEVICE}")
    print(f"Seed  : {seed}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # checkpoint 命名带 age 标识，区别于 baseline (resnet18_*) 与未来 sex/joint 变体；
    # --ckpt_suffix 用于区分不同 batch_size 重跑（如 _b128），避免覆盖
    sfx = args.ckpt_suffix
    ckpt_overall_path = OUTPUT_DIR / f"hyperadapt_age_malignant{sfx}_seed{seed}_best_overall.pth"
    ckpt_worstcase_path = OUTPUT_DIR / f"hyperadapt_age_malignant{sfx}_seed{seed}_best_worstcase.pth"

    print("\nLoading HAM10000 (malignant, age-conditioned HyperAdapt)...")
    train_transform, eval_transform = build_transforms(cfg)
    train_loader, val_loader, test_loader = get_dataloaders(cfg, train_transform, eval_transform)
    print(f"  Train: {len(train_loader.dataset):,} images")
    print(f"  Val  : {len(val_loader.dataset):,}")
    print(f"  Test : {len(test_loader.dataset):,}")

    # pretrained=True：backbone 与 baseline 一致 (ImageNet 预训练)，adapter 初始 Δθ≈0
    model = ResNet18HyperAdaptAge(num_classes=1, num_age=4,
                                  pretrained=cfg["model"]["pretrained"]).to(DEVICE)
    n_total, n_adapter, n_backbone = model.param_breakdown()
    print(f"\nModel : ResNet18-HyperAdapt (backbone=ImageNet 预训练, 每个 conv + fc 由 age 调制)")
    print(f"Params: {n_total:,} (backbone {n_backbone:,} + adapter {n_adapter:,})")
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
