"""
Train ResNet18HyperHead on UTKFace for Binary Age Classification
================================================================
任务: 年龄二分类, [0, 35) vs [35, +inf)
      使用 BCEWithLogitsLoss (单 logit 输出)。

使用方法:
    1. 从 Kaggle 下载 CSV 版 UTKFace 数据集:
       https://www.kaggle.com/datasets/nipunarora8/age-gender-and-ethnicity-face-data-csv
    2. 解压得到 age_gender.csv, 放到 ./data/ 目录下
    3. python train_utkface.py
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from ResNet18_hyperhead import ResNet18HyperHead
from utkface_dataset import UTKFaceCSVDataset
from fairness_metrics import print_fairness_report


# ============================================================
# 超参数
# ============================================================
CSV_PATH = "./data/age_gender.csv"
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
NUM_WORKERS = 0         # Windows 下保持 0
NUM_CLASSES = 1         # BCE 二分类: 单 logit 输出
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 数据预处理
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # 48x48 灰度 -> 3 通道伪 RGB
    transforms.RandomHorizontalFlip(),    # 人脸左右翻转不改年龄
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
    # 先整体读一次 CSV (内部会把所有像素缓存成 numpy 数组)
    full_train = UTKFaceCSVDataset(csv_path=CSV_PATH, transform=train_transform)
    full_eval  = UTKFaceCSVDataset(csv_path=CSV_PATH, transform=eval_transform)

    # 用相同的索引划分, 保证训练/验证/测试不重叠
    n = len(full_train)
    n_test = int(n * TEST_RATIO)
    n_val  = int(n * VAL_RATIO)
    n_train = n - n_val - n_test

    gen = torch.Generator().manual_seed(SEED)
    train_set, val_set, test_set = random_split(
        full_train, [n_train, n_val, n_test], generator=gen,
    )
    # 验证/测试集用无增强的 transform
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

    for i, (images, labels, sexes, races) in enumerate(loader):
        images = images.to(DEVICE, non_blocking=True)
        # BCEWithLogitsLoss 要求 float 标签, 形状与 logits 匹配 [B, 1]
        labels = labels.float().unsqueeze(1).to(DEVICE)
        sexes = sexes.long().to(DEVICE)
        races = races.long().to(DEVICE)

        optimizer.zero_grad()
        logits = model(images, sexes, races)         # [B, 1]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        # 阈值 0.5 判正类: sigmoid(logit) > 0.5  <=>  logit > 0
        preds = (logits > 0).float()
        correct += preds.eq(labels).sum().item()
        total += batch_size

        if (i + 1) % 50 == 0:
            print(f"  [Epoch {epoch} | Batch {i+1:4d}/{len(loader)}] "
                  f"loss={total_loss/total:.3f}  acc={100.*correct/total:.2f}%")

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
        sexes = sexes.long().to(DEVICE)
        races = races.long().to(DEVICE)

        logits = model(images, sexes, races)
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
    返回 numpy 数组: logits, labels, sexes, races
    """
    model.eval()
    all_logits, all_labels, all_sexes, all_races = [], [], [], []

    for images, labels, sexes, races in loader:
        images = images.to(DEVICE, non_blocking=True)
        sexes_gpu = sexes.long().to(DEVICE)
        races_gpu = races.long().to(DEVICE)

        logits = model(images, sexes_gpu, races_gpu).squeeze(1)  # [B]
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

    model = ResNet18HyperHead(
        out_dim=NUM_CLASSES, num_sex=2, num_race=5,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel params: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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
            torch.save(model.state_dict(), "resnet18_hyper_utkface_best.pth")
            print(f"  ↳ New best val acc, saved.")
        print("-" * 60)

    # 最终用 best checkpoint 在 test set 上跑
    print("\nLoading best checkpoint and evaluating on test set...")
    model.load_state_dict(torch.load("resnet18_hyper_utkface_best.pth"))
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