"""
Fashion-MNIST Training Script for ResNet-18
===========================================
用 Fashion-MNIST (10 类) 做 sanity check, 测试 resnet18_binary.py 的正确性。

使用方法:
    python train_fmnist.py

数据会自动下载到 ./data 目录。
"""

import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ResNet18 import ResNet18


# ============================================================
# 超参数
# ============================================================
DATA_DIR = "./data"
BATCH_SIZE = 128
NUM_EPOCHS = 5          # sanity check 够用; 追求精度可调到 20+
LEARNING_RATE = 1e-3
NUM_WORKERS = 0         # Windows 下建议保持 0; Linux/Mac 可以设为 2 或 4 加速
NUM_CLASSES = 10        # Fashion-MNIST 是 10 类
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FMNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


# ============================================================
# 数据预处理
# ============================================================
# Fashion-MNIST 是 1 通道 28x28 灰度图, 我们的 ResNet-18 要 3 通道 224x224
# 所以需要: Resize -> Grayscale(3) 把 1 通道复制成 3 通道 -> ToTensor -> Normalize
#
# 注意: 不要用 transforms.Lambda(lambda x: x.repeat(3,1,1)),
# 因为 Windows 下 DataLoader 多进程时 lambda 无法被 pickle, 会报错。
# 用 Grayscale(num_output_channels=3) 等价但可以被 pickle。
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),  # [1, H, W] -> [3, H, W]
    transforms.ToTensor(),                        # PIL -> Tensor, 值域 [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# 测试集用同样的 transform (没有数据增强)
test_transform = train_transform


# ============================================================
# 数据加载
# ============================================================
def get_dataloaders():
    train_set = datasets.FashionMNIST(
        root=DATA_DIR, train=True,  download=True, transform=train_transform,
    )
    test_set = datasets.FashionMNIST(
        root=DATA_DIR, train=False, download=True, transform=test_transform,
    )

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    return train_loader, test_loader


# ============================================================
# 训练 / 评估
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(loader):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = logits.max(dim=1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        # 每 100 个 batch 打印一次进度
        if (i + 1) % 100 == 0:
            print(f"  [Epoch {epoch} | Batch {i+1:4d}/{len(loader)}] "
                  f"loss={running_loss/total:.4f}  acc={100.*correct/total:.2f}%")

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = logits.max(dim=1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100. * correct / total


# ============================================================
# 主函数
# ============================================================
def main():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cpu":
        print("⚠️  No GPU detected. Training will be slow. "
              "Consider reducing NUM_EPOCHS or image size.")

    # 数据
    print("\nLoading Fashion-MNIST...")
    train_loader, test_loader = get_dataloaders()
    print(f"  Train batches: {len(train_loader)} ({len(train_loader.dataset)} images)")
    print(f"  Test  batches: {len(test_loader)} ({len(test_loader.dataset)} images)")

    # 模型
    model = ResNet18(num_classes=NUM_CLASSES).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: ResNet-18  ({n_params:,} params)")

    # 损失 / 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练循环
    print(f"\nTraining for {NUM_EPOCHS} epochs...\n" + "=" * 60)
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch,
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        dt = time.time() - t0
        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} [{dt:.1f}s] | "
              f"train loss={train_loss:.4f} acc={train_acc:.2f}% | "
              f" test loss={test_loss:.4f} acc={test_acc:.2f}%")
        print("-" * 60)

    # 保存
    torch.save(model.state_dict(), "resnet18_fmnist.pth")
    print("\n✅ Training done. Model saved to resnet18_fmnist.pth")


if __name__ == "__main__":
    main()