"""
Train PreactivResNet18 on UTKFace for Binary Age Classification (Image-only)
=============================================================================
任务: 年龄二分类, [0, 35) vs [35, +inf)
      仅以图像作为输入, 不引入 sex / race 等属性信息。

更新 (vs. 上一版):
    - LEARNING_RATE: 1e-3 -> 1e-4   (避免初期跨入 trivial 解)
    - Optimizer    : Adam -> AdamW + weight_decay=1e-4
                     (HyperFusion 论文 Table 1 即用 weight decay)
    - 加梯度裁剪    : max_norm=1.0   (双保险, 防止极端首步)
    - 训练循环输出 batch logit 的 mean/std (便于直接看初始化是否健康)

故障诊断笔记 (上一版):
    上一版 fairness 报告里 TPR=FPR=0、Overall AUC=0.40 < 0.5,
    说明模型崩溃到全预测多数类. 根因是 preactiv_resnet18.py 的
    _init_weights 把 fc2 (out=1) 用 fan_out 模式 Kaiming-normal 初始化,
    std ≈ sqrt(2/1) ≈ 1.41, 导致初始 logit ~10+ 量级, 第一个 batch
    BCE loss 飙到 6.889. 现已在模型文件里把 Linear 改回 PyTorch 默认.
    本训练脚本的 lr 降低 + weight decay + grad clip 是配套的稳定性手段.
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from PreactivResNet18 import PreactivResNet18_2D
from utkface_dataset import UTKFaceCSVDataset
from fairness_metrics import print_fairness_report


# ============================================================
# 超参数
# ============================================================
CSV_PATH = "./data/age_gender.csv"
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4    # 上一版 1e-3 + 错误初始化 -> 第一步就跨进 trivial 解
WEIGHT_DECAY = 1e-4     # 与 HyperFusion 论文 Table 1 一致
GRAD_CLIP_NORM = 1.0    # 防止极端首步
NUM_WORKERS = 0         # Windows 下保持 0
NUM_CLASSES = 1         # BCE 二分类: 单 logit 输出
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "preactiv_resnet18_utkface_best.pth"


# ============================================================
# 数据预处理 (与 attrconcat 版完全一致, 保证可比性)
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ============================================================
# 数据加载
# ============================================================
def get_dataloaders():
    full_train = UTKFaceCSVDataset(csv_path=CSV_PATH, transform=train_transform)
    full_eval  = UTKFaceCSVDataset(csv_path=CSV_PATH, transform=eval_transform)

    n = len(full_train)
    n_test = int(n * TEST_RATIO)
    n_val  = int(n * VAL_RATIO)
    n_train = n - n_val - n_test

    gen = torch.Generator().manual_seed(SEED)
    train_set, val_set, test_set = random_split(
        full_train, [n_train, n_val, n_test], generator=gen,
    )
    val_set.dataset = full_eval
    test_set.dataset = full_eval

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, test_loader


# ============================================================
# 训练 / 评估
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    first_batch_logit_stats = None    # 用于诊断初始化是否健康

    for i, (images, labels, sexes, races) in enumerate(loader):
        # sexes / races 不进入模型 (image-only); 仍从 loader 解包以保持接口
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)                 # [B, 1]
        loss = criterion(logits, labels)
        loss.backward()
        # 梯度裁剪: 上一版 epoch 1 loss 飙到 6.889, 这里做双保险
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()

        if epoch == 1 and i == 0:
            first_batch_logit_stats = (
                logits.detach().mean().item(),
                logits.detach().std().item(),
            )

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        preds = (logits > 0).float()
        correct += preds.eq(labels).sum().item()
        total += batch_size

        if (i + 1) % 50 == 0:
            print(f"  [Epoch {epoch} | Batch {i+1:4d}/{len(loader)}] "
                  f"loss={total_loss/total:.3f}  acc={100.*correct/total:.2f}%")

    if first_batch_logit_stats is not None:
        m, s = first_batch_logit_stats
        print(f"  [Init check] epoch1 batch0 logit  mean={m:+.3f}  std={s:.3f}  "
              f"(健康范围: |mean|<2 且 std<5)")

    return total_loss / total, 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels, sexes, races in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        preds = (logits > 0).float()
        correct += preds.eq(labels).sum().item()
        total += batch_size

    return total_loss / total, 100. * correct / total


@torch.no_grad()
def collect_predictions(model, loader):
    """
    在整个 loader 上收集 logits 和所有属性, 供公平性评估使用。
    属性 sex / race 不进入模型, 仅作为分组键 — 用来观察 image-only 模型
    在不同子群上的表现差异 (隐式偏差)。
    """
    model.eval()
    all_logits, all_labels, all_sexes, all_races = [], [], [], []

    for images, labels, sexes, races in loader:
        images = images.to(DEVICE, non_blocking=True)
        logits = model(images).squeeze(1)      # [B]
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())
        all_sexes.append(sexes.numpy())
        all_races.append(races.numpy())

    return (
        np.concatenate(all_logits),
        np.concatenate(all_labels),
        np.concatenate(all_sexes),
        np.concatenate(all_races),
    )


# ============================================================
# 主函数
# ============================================================
def main():
    print(f"Device: {DEVICE}")

    print("\nLoading UTKFace...")
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"  Train: {len(train_loader.dataset)} images")
    print(f"  Val  : {len(val_loader.dataset)}")
    print(f"  Test : {len(test_loader.dataset)}")

    model = PreactivResNet18_2D(
        in_channels=3, n_outputs=NUM_CLASSES, init_features=16,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel : PreactivResNet18_2D  (image-only, no attribute concat)")
    print(f"Params: {n_params:,}")
    print(f"Optim : AdamW(lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
    print(f"Misc  : grad_clip_norm={GRAD_CLIP_NORM}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_acc = 0.0
    print(f"\nTraining for {NUM_EPOCHS} epochs...\n" + "=" * 60)
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        dt = time.time() - t0

        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} [{dt:.1f}s] | "
              f"train loss={train_loss:.3f} acc={train_acc:.2f}% | "
              f"val loss={val_loss:.3f} acc={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"  ↳ New best val acc, saved.")
        print("-" * 60)

    # 最终用 best checkpoint 在 test set 上跑
    print("\nLoading best checkpoint and evaluating on test set...")
    model.load_state_dict(torch.load(CKPT_PATH))
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"Test loss={test_loss:.3f}  accuracy={test_acc:.2f}%")

    # --- 公平性评估 ---
    logits, labels, sexes, races = collect_predictions(model, test_loader)
    print_fairness_report(
        y_true=labels, y_score=logits,
        sex=sexes, race=races,
        threshold=0.0,   # logits > 0 判正类
    )


if __name__ == "__main__":
    main()