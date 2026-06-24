"""
Train ResNet18-HyperAdapt on Fitzpatrick17k for Malignant Binary Classification
===============================================================================
属性融合变体训练流程：在 image-only baseline (src/training/train_fitzpatrick_resnet18.py)
的基础上，把 backbone 的每个卷积层 (通道级乘性调制) 与分类头 fc (加性低秩更新) 都接上
由唯一敏感属性「肤色 skin」(Fitzpatrick I–VI → 0–5) 驱动的 HyperAdapt 超网络
(src/models/resnet18_hyperadapt.py 的 ResNet18HyperAdaptSkin)。backbone 与 baseline
完全一致 (ImageNet 预训练 torchvision ResNet-18)，初始化保证 Δθ≈0、起始等价于纯预训练
backbone，因此可与 baseline 在相同预处理、相同早停设置下直接对比公平性。

与 baseline (image-only) 训练脚本的唯一区别：
  - 模型换成 ResNet18HyperAdaptSkin（注入到每个 conv + fc，由 skin 逐样本调制）
  - 训练 / 评估循环把 skin 喂进模型 model(images, skins)（baseline 中 skin 仅作分组键）
其余训练策略 (AdamW + lr=1e-4 + weight_decay + 梯度裁剪、两种 model selection、早停监控
val Overall AUC、双 checkpoint best_overall / best_worstcase、6 肤色子群 worst-case AUC
公平性报告、超参均读 configs/fitzpatrick_baseline.yaml) 与 baseline 严格保持一致。

与 MIMIC HyperAdapt (train_mimic_hyperadapt.py) 的差异：单一 6 类 skin 属性（非
sex/race/age 三属性），公平性口径走 src/utils/fitzpatrick_fairness（6 肤色子群）。

⚠️ 显存提醒：HyperAdapt 用 grouped-conv 展开「逐样本卷积核」(把权重复制 B 份)，显存
占用显著高于 baseline。若 batch_size=128 在 gpus24 上 OOM，可在 configs 里调小
batch_size（训练策略其余部分不变），或改投 gpus48（slurm 脚本默认已选 gpus48）。

运行方式:
    重型任务，必须通过 sbatch 提交到 SLURM 集群 (见 slurm/train_fitzpatrick_hyperadapt.sh)。
    通过 --seed 指定随机种子（默认取 configs/fitzpatrick_baseline.yaml 中 training.seeds[0]）。
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

from src.datasets.fitzpatrick_dataset import FitzpatrickDataset
from src.models.resnet18_hyperadapt import ResNet18HyperAdaptSkin
from src.utils.fitzpatrick_fairness import print_fitzpatrick_fairness_report, worst_case_auc
from src.utils.resampling import build_group_label_sampler

# ============================================================
# 路径
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "fitzpatrick_baseline.yaml"

# 模型权重属于「由集群生成的大文件」，按项目规范存放在仓库外的根目录 outputs/，不进仓库
OUTPUT_DIR = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/fitzpatrick")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 配置加载
# ============================================================
def load_config(path: Path) -> dict:
    """读取 YAML 训练配置。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """解析命令行参数（--seed 多 seed 重复实验；--resample_alpha 肤色×标签平衡重采样）。"""
    parser = argparse.ArgumentParser(
        description="Train ResNet18-HyperAdapt on Fitzpatrick17k (malignant, skin-conditioned)"
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子；缺省时取 configs/fitzpatrick_baseline.yaml 中 training.seeds[0]")
    parser.add_argument("--resample_alpha", type=float, default=None,
                        help="肤色×标签平衡重采样强度 ∈[0,1]（0=自然分布，1=完全平衡）。"
                             "缺省时不重采样，train 仍用 shuffle=True，行为与 baseline 一致；"
                             "给定时按该 alpha 启用 WeightedRandomSampler，平衡维度取 config 的 resampling.dims。")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="覆盖 config 的 training.batch_size；HyperAdapt 逐样本卷积核显存大，"
                             "在 gpus24 (24GiB) 上需调小（如 32）以免 OOM。缺省时用 config 值（128）。"
                             "仅改 batch_size，lr 等训练策略不变（与 baseline 对齐）。")
    parser.add_argument("--ckpt_suffix", type=str, default="",
                        help="追加到 checkpoint 文件名末尾的标识（如 '_b128'），用于区分不同 batch_size "
                             "等同一配置下的并行实验、避免互相覆盖。缺省为空，命名与原行为完全一致。")
    return parser.parse_args()


def build_transforms(cfg: dict) -> tuple[transforms.Compose, transforms.Compose]:
    """
    根据配置构建训练 / 评估 transform（与 baseline 一致）。

    Fitzpatrick17k preproc_224x224 已是自然 RGB（uint8, 0~255）且尺寸即 224，
    因此无需 Grayscale / Resize；训练增强保留水平翻转（皮损无解剖学左右约束）。
    """
    t_cfg     = cfg["transforms"]
    norm_mean = t_cfg["normalize"]["mean"]
    norm_std  = t_cfg["normalize"]["std"]

    train_ops: list = []
    # 水平翻转：皮损无左右解剖约束（与本项目 CXR 流程「不翻转」相反，与 MEDFAIR 一致）
    if t_cfg["train"].get("horizontal_flip", False):
        train_ops.append(transforms.RandomHorizontalFlip())
    train_ops.append(transforms.RandomRotation(degrees=t_cfg["train"]["rotation_degrees"]))
    train_ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ]
    train_transform = transforms.Compose(train_ops)

    # 验证/测试不做几何增强（config 中 eval: {}）
    eval_transform = transforms.Compose([
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
    构建 train / val / test DataLoader（与 baseline 一致）。

    三个 split 已由 scripts/build_fitzpatrick_splits.py 按图像级 80/10/10 划分写入独立 CSV
    (data/splits/fitzpatrick17k/{train,val,test}.csv)，此处直接加载。
    val / test 保持人群真实分布以确保评估指标可信。

    Args:
        resample_alpha: 若为 None（默认），train 用 shuffle=True，与 baseline 一致；
            若给定，则用 WeightedRandomSampler 按 config 的 resampling.dims 做肤色×标签
            平衡（强度 alpha），仅作用于 train（val/test 不变）。
    """
    split_dir = REPO_ROOT / cfg["data"]["split_dir"]
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["dataloader"]["num_workers"]
    pin_memory = cfg["dataloader"]["pin_memory"]

    train_set = FitzpatrickDataset(split_dir / "train.csv", transform=train_transform)
    val_set = FitzpatrickDataset(split_dir / "val.csv", transform=eval_transform)
    test_set = FitzpatrickDataset(split_dir / "test.csv", transform=eval_transform)

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

    for i, (images, labels, skins) in enumerate(loader):
        # 与 baseline 不同：skin 进入模型，作为 HyperAdapt 超网络的条件
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(DEVICE)
        skins = skins.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images, skins)                # [B, 1]
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        preds = (logits > 0).float()
        correct += preds.eq(labels).sum().item()
        total += batch_size

        if (i + 1) % 50 == 0:
            print(f"  [Epoch {epoch} | Batch {i + 1:5d}/{len(loader)}] "
                  f"loss={total_loss / total:.4f}  acc={100. * correct / total:.2f}%")

    return total_loss / total, 100. * correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float, float, float]:
    """
    在给定 loader 上评估，返回 (平均 loss, accuracy %, overall AUC, worst-case AUC)。
    skin 既作为模型输入 (超网络条件)，又作为公平性分组键。

    worst_case_auc 在无有效子群时返回 None，此处回落为 0.0（仅用于排序）。
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_logits, all_labels, all_skins = [], [], []

    for images, labels, skins in loader:
        images = images.to(DEVICE, non_blocking=True)
        label_col = labels.float().unsqueeze(1).to(DEVICE)
        skins_d = skins.to(DEVICE, non_blocking=True)

        logits = model(images, skins_d)
        loss = criterion(logits, label_col)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        preds = (logits > 0).float()
        correct += preds.eq(label_col).sum().item()
        total += batch_size

        all_logits.append(logits.squeeze(1).cpu().numpy())
        all_labels.append(labels.numpy())
        all_skins.append(skins.numpy())

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    all_skins = np.concatenate(all_skins)

    auc = roc_auc_score(all_labels, all_logits)
    wc = worst_case_auc(all_labels, all_logits, all_skins)
    wc_auc = wc if wc is not None else 0.0

    return total_loss / total, 100. * correct / total, auc, wc_auc


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    在整个 loader 上收集 logits 与 (label, skin)，供公平性评估使用。
    skin 既是超网络条件，又是分组键。
    """
    model.eval()
    all_logits, all_labels, all_skins = [], [], []

    for images, labels, skins in loader:
        images = images.to(DEVICE, non_blocking=True)
        skins_d = skins.to(DEVICE, non_blocking=True)
        logits = model(images, skins_d).squeeze(1)       # [B]
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())
        all_skins.append(skins.numpy())

    return (
        np.concatenate(all_logits),
        np.concatenate(all_labels),
        np.concatenate(all_skins),
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

    logits, labels, skins = collect_predictions(model, test_loader)
    print_fitzpatrick_fairness_report(
        y_true=labels, y_score=logits, skin=skins,
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

    # 重采样：CLI --resample_alpha 优先；缺省时回落到 config（仅当 resampling.enabled 时启用）
    resample_alpha = args.resample_alpha
    if resample_alpha is None and cfg.get("resampling", {}).get("enabled", False):
        resample_alpha = cfg["resampling"]["alpha"]
    # checkpoint 命名加 resample tag，避免覆盖未重采样的权重；再叠加 --ckpt_suffix（如 _b128）区分等 batch 对照实验
    tag = f"_resample_a{resample_alpha}" if resample_alpha is not None else ""
    tag = f"{tag}{args.ckpt_suffix}"

    if resample_alpha is None:
        resample_status = "OFF (shuffle=True)"
    else:
        resample_status = f"ON  alpha={resample_alpha}  dims={cfg['resampling']['dims']}"

    print(f"Device: {DEVICE}")
    print(f"Seed  : {seed}")
    print(f"Resampling: {resample_status}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_overall_path = OUTPUT_DIR / f"hyperadapt_malignant{tag}_seed{seed}_best_overall.pth"
    ckpt_worstcase_path = OUTPUT_DIR / f"hyperadapt_malignant{tag}_seed{seed}_best_worstcase.pth"

    print("\nLoading Fitzpatrick17k (malignant, skin-conditioned HyperAdapt)...")
    train_transform, eval_transform = build_transforms(cfg)
    train_loader, val_loader, test_loader = get_dataloaders(
        cfg, train_transform, eval_transform, resample_alpha=resample_alpha,
    )
    print(f"  Train: {len(train_loader.dataset):,} images")
    print(f"  Val  : {len(val_loader.dataset):,}")
    print(f"  Test : {len(test_loader.dataset):,}")

    # pretrained=True：backbone 与 baseline 一致 (ImageNet 预训练)，adapter 初始 Δθ≈0
    model = ResNet18HyperAdaptSkin(num_classes=1, num_skin=6,
                                   pretrained=cfg["model"]["pretrained"]).to(DEVICE)
    n_total, n_adapter, n_backbone = model.param_breakdown()
    print(f"\nModel : ResNet18-HyperAdapt (backbone=ImageNet 预训练, 每个 conv + fc 由 skin 调制)")
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
