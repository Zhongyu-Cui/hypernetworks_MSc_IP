"""
Diagnose the HyperHead attribute pathway: positive control + capacity/scale sweep
=================================================================================
回答「超网络在 MIMIC No Finding 上不提升公平性，是工程问题（初始化过小 / 学习率不足 /
容量不够导致属性通路被人为压制）还是信号本身弱？」这一问题，提供两类诊断实验：

1. **正控制 (--synthetic_xor {sex,race,age})**
   把训练/评估标签替换为 `y_syn = label XOR attr`。此时图像信号本身无法独立解出标签
   （在某属性组内目标被翻转），**唯有用到该属性才能恢复高 AUC**。
   - `--model hyperhead`：若 AUC 恢复到 ≈baseline → 超网络机器健康、能在信号存在时利用属性；
   - `--model imageonly`：看不到属性，AUC 应坍到 ≈0.5（反向对照，锚定下界）。
   两者对比即可判定「机器是否坏」：HyperHead 远高于 image-only ⇒ 工程实现没问题。

2. **容量/尺度扫描 (--hyper_lr_mult / --hyper_init_std / --hyper_hidden / --hyper_embed)**
   在**真实标签**下放大超网络通路（独立更高学习率、更大初始化、更大 embedding/hidden），
   看 worst-case AUC 是否随之改善：
   - 若属性效应变大但 worst-case AUC 不动 → 坐实「信号受限」，工程已排除；
   - 若 worst-case AUC 跟着改善 → 之前确被过小尺度钳制，值得继续放大。
   （属性翻转敏感度可在训练后用 scripts/diagnose_attr_pathway.py 指向本脚本存的 checkpoint 复测。）

backbone / transform / 早停 / 优化器主体与 train_mimic_hyperhead.py 完全一致，仅多出上述
诊断开关；不传任何开关时退化为标准 HyperHead 训练。checkpoint 带 --tag 区分，绝不覆盖
既有 65887 等正式权重。

运行方式:
    重型任务，必须通过 sbatch 提交到 SLURM 集群 (见 slurm/diagnose_hyperhead_pathway.sh)。
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
from src.models.resnet18_hyperhead import ResNet18HyperHead
from src.models.resnet18_pretrained import ResNet18Pretrained
from src.utils.mimic_fairness import print_mimic_fairness_report, worst_case_auc

# ============================================================
# 路径
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "mimic_cxr_baseline.yaml"

# 模型权重属于「由集群生成的大文件」，按项目规范存放在仓库外的根目录 outputs/，不进仓库
OUTPUT_DIR = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/mimic_cxr")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

XOR_ATTR_INDEX = {"sex": 0, "race": 1, "age": 2}  # 合成标签可异或的属性

# 干净正控制用的随机属性种子（固定常量，与训练 seed 解耦，
# 保证 hyperhead 与 image-only 两臂看到完全相同的 r 分配，可比）
RANDOM_ATTR_SEED = 12345


# ============================================================
# 干净正控制数据集：注入与图像无关的随机二值属性 r
# ============================================================
class RandomAttrDataset(MIMICCXRDataset):
    """
    在标准 MIMICCXRDataset 基础上注入一个**与图像内容无关**的随机二值属性 r，
    并把标签替换为 `label XOR r`，r 通过 sex 槽位喂给模型（race/age 仍为真实值，
    仅用于公平性分组、不参与构造标签）。

    动机：XOR sex 的正控制被"胸片可解码 sex"污染（image-only 也能解出 → AUC 不坍）。
    改用随机 r 后，image-only 无从得知 r → 在 `y XOR r` 上 AUC 必坍到 ≈0.5；而 HyperHead
    把 r 作为显式输入 → 可逐样本翻转决策、恢复 ≈baseline。两臂之差即为「超网络机器能否
    利用属性」的干净证据。

    r 在 __init__ 时按 attr_seed 一次性确定（每个 split 各自固定），保证跨 epoch、跨
    两臂一致、评估可复现。返回签名与父类一致 (image, label, sex, race, age_group)，
    因此训练/评估循环无需任何改动。
    """

    def __init__(self, *args, attr_seed: int = RANDOM_ATTR_SEED, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        rng = np.random.default_rng(attr_seed)
        self.rand_attr = rng.integers(0, 2, size=len(self.labels)).astype(np.int64)

    def __getitem__(self, idx: int):
        image, label, _sex, race, age_group = super().__getitem__(idx)
        r = int(self.rand_attr[idx])
        # 标签异或随机属性；r 占用 sex 槽位作为模型条件，race/age 保留真实值供分组
        return image, label ^ r, r, race, age_group


# ============================================================
# 配置加载与参数
# ============================================================
def load_config(path: Path) -> dict:
    """读取 YAML 训练配置。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """解析诊断实验的命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Diagnose HyperHead attribute pathway (positive control + scale sweep)"
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子；缺省取 config 的 training.seeds[0]。")
    parser.add_argument("--model", choices=["hyperhead", "imageonly"], default="hyperhead",
                        help="hyperhead=属性条件分类头；imageonly=纯图像 baseline（正控制的反向对照）。")
    parser.add_argument("--synthetic_xor", choices=["none", "sex", "race", "age"], default="none",
                        help="正控制：标签改为 label XOR <真实属性>。none 时用真实标签。"
                             "注意 sex/race/age 可从胸片部分解码，该正控制会被泄漏污染——"
                             "干净版用 --random_attr。")
    parser.add_argument("--random_attr", action="store_true",
                        help="干净正控制：注入与图像无关的随机二值属性 r，标签=label XOR r，"
                             "r 经 sex 槽位喂给模型（见 RandomAttrDataset）。开启时忽略 --synthetic_xor。")
    parser.add_argument("--hyper_lr_mult", type=float, default=1.0,
                        help="超网络参数相对 backbone 的学习率倍数（仅 hyperhead 生效）。")
    parser.add_argument("--hyper_init_std", type=float, default=1e-3,
                        help="超网络末层权重初始化 std（默认 1e-3 近零起步；调大以检验是否被压制）。")
    parser.add_argument("--hyper_hidden", type=int, default=64,
                        help="超网络 MLP 隐藏层维度（仅 hyperhead 生效）。")
    parser.add_argument("--hyper_embed", type=int, default=4,
                        help="每个敏感属性 embedding 维度（仅 hyperhead 生效）。")
    parser.add_argument("--tag", type=str, default="",
                        help="checkpoint / 输出文件名后缀，区分不同诊断配置，避免覆盖正式权重。")
    return parser.parse_args()


# ============================================================
# 合成标签（正控制）
# ============================================================
def apply_synthetic_xor(
    labels: torch.Tensor,
    sexes: torch.Tensor,
    races: torch.Tensor,
    ages: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """
    正控制标签变换：返回 `label XOR attr`（mode="none" 时原样返回）。

    在 attr=1 的子群内目标被翻转，图像通路无法独立解出 → 唯有用到该属性才能恢复高 AUC，
    据此判定属性通路是否真的可用。所有输入均为 0/1 的 LongTensor。

    Args:
        labels: [B] 原始二分类标签。
        sexes / races / ages: [B] 三个二值敏感属性。
        mode: "none" / "sex" / "race" / "age"。

    Returns:
        [B] 变换后的标签（LongTensor）。
    """
    if mode == "none":
        return labels
    attr = {"sex": sexes, "race": races, "age": ages}[mode]
    # 整数张量按位异或；两者均取值 {0,1}，结果仍为 {0,1}
    return torch.bitwise_xor(labels, attr)


def build_transforms(cfg: dict) -> tuple[transforms.Compose, transforms.Compose]:
    """构建训练 / 评估 transform（与 baseline / hyperhead 完全一致）。"""
    t_cfg = cfg["transforms"]
    norm_mean = t_cfg["normalize"]["mean"]
    norm_std = t_cfg["normalize"]["std"]

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
    random_attr: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    构建 train / val / test DataLoader（患者级划分的独立 CSV，与 baseline 一致）。

    Args:
        random_attr: True 时改用 RandomAttrDataset（干净正控制：标签 XOR 随机属性 r）。
    """
    split_dir = REPO_ROOT / cfg["data"]["split_dir"]
    image_size = cfg["data"]["image_size"]
    train_csv_name = cfg["data"].get("train_csv", "train.csv")
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["dataloader"]["num_workers"]
    pin_memory = cfg["dataloader"]["pin_memory"]
    age_threshold = cfg["attributes"]["age"]["age_threshold"]

    ds_cls = RandomAttrDataset if random_attr else MIMICCXRDataset
    train_set = ds_cls(split_dir / train_csv_name, transform=train_transform,
                       image_size=image_size, age_threshold=age_threshold)
    val_set = ds_cls(split_dir / "val.csv", transform=eval_transform,
                     image_size=image_size, age_threshold=age_threshold)
    test_set = ds_cls(split_dir / "test.csv", transform=eval_transform,
                      image_size=image_size, age_threshold=age_threshold)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


# ============================================================
# 前向：兼容 hyperhead（带属性）与 imageonly（仅图像）
# ============================================================
def model_forward(
    model: nn.Module, is_hyper: bool,
    images: torch.Tensor, sexes: torch.Tensor, races: torch.Tensor, ages: torch.Tensor,
) -> torch.Tensor:
    """根据模型类型选择 forward 签名：hyperhead 吃属性，imageonly 仅吃图像。"""
    if is_hyper:
        return model(images, sexes, races, ages)
    return model(images)


# ============================================================
# 训练 / 评估
# ============================================================
def train_one_epoch(
    model: nn.Module, is_hyper: bool, xor_mode: str,
    loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer,
    grad_clip_norm: float, epoch: int,
) -> tuple[float, float]:
    """跑一个训练 epoch，返回 (平均 loss, accuracy %)。"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels, sexes, races, ages) in enumerate(loader):
        labels = apply_synthetic_xor(labels, sexes, races, ages, xor_mode)  # 正控制（none 时无变化）
        images = images.to(DEVICE, non_blocking=True)
        label_col = labels.float().unsqueeze(1).to(DEVICE)
        sexes_d = sexes.to(DEVICE, non_blocking=True)
        races_d = races.to(DEVICE, non_blocking=True)
        ages_d = ages.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        logits = model_forward(model, is_hyper, images, sexes_d, races_d, ages_d)  # [B, 1]
        loss = criterion(logits, label_col)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        preds = (logits > 0).float()
        correct += preds.eq(label_col).sum().item()
        total += batch_size

        if (i + 1) % 200 == 0:
            print(f"  [Epoch {epoch} | Batch {i + 1:5d}/{len(loader)}] "
                  f"loss={total_loss / total:.4f}  acc={100. * correct / total:.2f}%")

    return total_loss / total, 100. * correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module, is_hyper: bool, xor_mode: str,
    loader: DataLoader, criterion: nn.Module,
) -> tuple[float, float, float, float]:
    """评估，返回 (平均 loss, accuracy %, overall AUC, worst-case AUC)。"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_logits, all_labels, all_sexes, all_races, all_ages = [], [], [], [], []

    for images, labels, sexes, races, ages in loader:
        labels = apply_synthetic_xor(labels, sexes, races, ages, xor_mode)
        images = images.to(DEVICE, non_blocking=True)
        label_col = labels.float().unsqueeze(1).to(DEVICE)
        sexes_d = sexes.to(DEVICE, non_blocking=True)
        races_d = races.to(DEVICE, non_blocking=True)
        ages_d = ages.to(DEVICE, non_blocking=True)

        logits = model_forward(model, is_hyper, images, sexes_d, races_d, ages_d)
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
def collect_predictions(
    model: nn.Module, is_hyper: bool, xor_mode: str, loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """收集 logits 与 (label, sex, race, age_group)，供公平性评估。"""
    model.eval()
    all_logits, all_labels, all_sexes, all_races, all_ages = [], [], [], [], []

    for images, labels, sexes, races, ages in loader:
        labels = apply_synthetic_xor(labels, sexes, races, ages, xor_mode)
        images = images.to(DEVICE, non_blocking=True)
        sexes_d = sexes.to(DEVICE, non_blocking=True)
        races_d = races.to(DEVICE, non_blocking=True)
        ages_d = ages.to(DEVICE, non_blocking=True)
        logits = model_forward(model, is_hyper, images, sexes_d, races_d, ages_d).squeeze(1)
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
    model: nn.Module, is_hyper: bool, xor_mode: str,
    ckpt_path: Path, test_loader: DataLoader, criterion: nn.Module, label: str,
) -> None:
    """加载 checkpoint，在测试集上评估并打印分组公平性报告。"""
    print(f"\n{'#' * 70}\n# Evaluating checkpoint: {label} ({ckpt_path.name})\n{'#' * 70}")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    test_loss, test_acc, test_auc, test_wc_auc = evaluate(model, is_hyper, xor_mode, test_loader, criterion)
    print(f"Test loss={test_loss:.4f}  accuracy={test_acc:.2f}%  AUC={test_auc:.4f}  "
          f"Worst-case AUC={test_wc_auc:.4f}")

    logits, labels, sexes, races, ages = collect_predictions(model, is_hyper, xor_mode, test_loader)
    print_mimic_fairness_report(
        y_true=labels, y_score=logits, sex=sexes, race=races, age=ages, threshold=0.0,
    )


# ============================================================
# 优化器：超网络参数可用独立（更高）学习率
# ============================================================
def build_optimizer(
    model: nn.Module, is_hyper: bool, base_lr: float, weight_decay: float, hyper_lr_mult: float,
) -> optim.Optimizer:
    """
    构建 AdamW。hyperhead 且 hyper_lr_mult≠1 时把超网络参数单独分组、用 base_lr*mult，
    以检验「超网络与 backbone 共享 lr 是否使其学得不足」。其余情形与 baseline 一致（单组）。
    """
    if not is_hyper or hyper_lr_mult == 1.0:
        return optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    hyper_params = list(model.hyper.parameters())
    hyper_ids = {id(p) for p in hyper_params}
    backbone_params = [p for p in model.parameters() if id(p) not in hyper_ids]
    return optim.AdamW(
        [
            {"params": backbone_params, "lr": base_lr},
            {"params": hyper_params, "lr": base_lr * hyper_lr_mult},
        ],
        lr=base_lr, weight_decay=weight_decay,
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

    is_hyper = args.model == "hyperhead"
    # --random_attr 已在 RandomAttrDataset 内部完成 label XOR r，循环里不能再异或一次
    xor_mode = "none" if args.random_attr else args.synthetic_xor
    tag = f"_{args.tag}" if args.tag else ""

    print(f"Device: {DEVICE}")
    print(f"Seed  : {seed}")
    print(f"Model : {args.model}")
    print(f"Random-attr clean control: {args.random_attr}")
    print(f"Synthetic XOR label : {xor_mode}  (none=真实标签)")
    if is_hyper:
        print(f"HyperNet knobs: lr_mult={args.hyper_lr_mult}  init_std={args.hyper_init_std}  "
              f"hidden={args.hyper_hidden}  embed={args.hyper_embed}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_overall_path = OUTPUT_DIR / f"diag_{args.model}{tag}_seed{seed}_best_overall.pth"
    ckpt_worstcase_path = OUTPUT_DIR / f"diag_{args.model}{tag}_seed{seed}_best_worstcase.pth"

    print("\nLoading MIMIC-CXR (No Finding, U-Zeros)...")
    train_transform, eval_transform = build_transforms(cfg)
    train_loader, val_loader, test_loader = get_dataloaders(
        cfg, train_transform, eval_transform, random_attr=args.random_attr,
    )
    print(f"  Train: {len(train_loader.dataset):,} images")
    print(f"  Val  : {len(val_loader.dataset):,}")
    print(f"  Test : {len(test_loader.dataset):,}")

    if is_hyper:
        model = ResNet18HyperHead(
            num_classes=1,
            sex_embed=args.hyper_embed, race_embed=args.hyper_embed, age_embed=args.hyper_embed,
            hyper_hidden=args.hyper_hidden, init_std=args.hyper_init_std,
        ).to(DEVICE)
    else:
        model = ResNet18Pretrained(num_classes=1).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel : {args.model}  (backbone=ImageNet 预训练 ResNet-18)")
    print(f"Params: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(
        model, is_hyper, train_cfg["learning_rate"], train_cfg["weight_decay"], args.hyper_lr_mult,
    )

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
            model, is_hyper, xor_mode, train_loader, criterion, optimizer,
            train_cfg["grad_clip_norm"], epoch,
        )
        val_loss, val_acc, val_auc, val_wc_auc = evaluate(model, is_hyper, xor_mode, val_loader, criterion)
        dt = time.time() - t0

        print(f"Epoch {epoch:2d}/{max_epochs} [{dt:.1f}s] | "
              f"train loss={train_loss:.4f} acc={train_acc:.2f}% | "
              f"val loss={val_loss:.4f} acc={val_acc:.2f}% AUC={val_auc:.4f} "
              f"worst-case AUC={val_wc_auc:.4f}")

        if val_wc_auc > best_val_wc_auc + min_delta:
            best_val_wc_auc = val_wc_auc
            torch.save(model.state_dict(), ckpt_worstcase_path)
            print(f"  -> New best val worst-case AUC ({best_val_wc_auc:.4f}); saved {ckpt_worstcase_path.name}")

        if val_auc > best_val_auc + min_delta:
            best_val_auc = val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_overall_path)
            print(f"  -> New best val overall AUC ({best_val_auc:.4f}); saved {ckpt_overall_path.name}")
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

    evaluate_checkpoint(model, is_hyper, xor_mode, ckpt_overall_path, test_loader, criterion,
                        label="Overall AUC selection")
    evaluate_checkpoint(model, is_hyper, xor_mode, ckpt_worstcase_path, test_loader, criterion,
                        label="Worst-case AUC selection")


if __name__ == "__main__":
    main()
