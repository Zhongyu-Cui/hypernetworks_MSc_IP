"""
Train HyperFusion_PreactivResNet18 on UTKFace for Binary Age Classification
============================================================================
任务: 年龄二分类, [0, 35) vs [35, +inf)
      backbone 处理图像, 在 layer4 第一个 ResBlock 的 downsample 上由
      (sex, race) 通过超网络生成卷积权重 (HyperFusion 风格 + MIP 初始化).

与 train_UTKface_preactive.py (image-only) 的差异:
    模型     : PreactivResNet18_2D    ->  HyperFusionPreactivResNet18
    forward  : model(image)           ->  model(image, sex, race)
    属性用途 : 仅用于公平性评估       ->  作为超网络条件输入 + 公平性评估

与 train_UTKface_attrconcat.py (拼接基线) 的差异:
    融合方式 : 把属性 broadcast 成空间通道与图像 concat
              ->  通过超网络生成 layer4 downsample 的权重
    主网络   : ResNet18AttrConcat
              ->  HyperFusionPreactivResNet18 (pre-activation + MIP)

稳定性手段 (沿用 image-only 版的修复):
    - LEARNING_RATE = 1e-4
    - AdamW(weight_decay=1e-4)
    - 梯度裁剪 max_norm=1.0
    - 训练首 batch 输出 logit 的 mean/std 用于诊断初始化健康
    - 额外报告超网络 / theta_0 的初始梯度量级, 验证双路径都通

使用方法:
    1. 从 Kaggle 下载 CSV 版 UTKFace 数据集:
       https://www.kaggle.com/datasets/nipunarora8/age-gender-and-ethnicity-face-data-csv
    2. 解压得到 age_gender.csv, 放到 ./data/ 目录下
    3. python train_UTKface_hyperfusion.py
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from PreactivResNet18_hyperfusion_MIP import HyperFusionPreactivResNet18
from utkface_dataset import UTKFaceCSVDataset
from fairness_metrics import print_fairness_report


# ============================================================
# 超参数
# ============================================================
CSV_PATH = "./data/age_gender.csv"
BATCH_SIZE = 64
NUM_EPOCHS = 40
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 0          # Windows 下保持 0
NUM_CLASSES = 1          # BCE 二分类: 单 logit 输出
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "hyperfusion_preactiv_resnet18_utkface_best.pth"

# HyperFusion / MIP 相关
NUM_SEX = 2              # UTKFace: 0=Male, 1=Female
NUM_RACE = 5             # UTKFace: 0=White, 1=Black, 2=Asian, 3=Indian, 4=Others
SEX_EMBED = 4
RACE_EMBED = 8
HYPER_HIDDEN = 64
MIP_SCALE = 0.01         # 输出层小尺度初始化, 让初始 delta ≈ 0
USE_L2_NORM = True       # E_L2 投影, 切断 magnitude proportionality


# ============================================================
# 数据预处理 (与前两版完全一致, 保证可比性)
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
    first_batch_diagnostics = None    # 诊断信息: logit + hyper / theta_0 梯度

    for i, (images, labels, sexes, races) in enumerate(loader):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(DEVICE)
        sexes = sexes.long().to(DEVICE)
        races = races.long().to(DEVICE)

        optimizer.zero_grad()
        logits = model(images, sexes, races)            # [B, 1]
        loss = criterion(logits, labels)
        loss.backward()

        # 在 step 之前抓一次诊断信息 (epoch1 的第一个 batch)
        if epoch == 1 and i == 0:
            with torch.no_grad():
                hyper_grad = sum(
                    p.grad.abs().sum().item()
                    for n, p in model.named_parameters()
                    if n.startswith("hyper_net") and p.grad is not None
                )
                theta0_grad = model.layer4_block0.ds_conv.weight.grad.abs().sum().item()
            first_batch_diagnostics = {
                "logit_mean": logits.detach().mean().item(),
                "logit_std":  logits.detach().std().item(),
                "hyper_grad": hyper_grad,
                "theta0_grad": theta0_grad,
            }

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        preds = (logits > 0).float()
        correct += preds.eq(labels).sum().item()
        total += batch_size

        if (i + 1) % 50 == 0:
            print(f"  [Epoch {epoch} | Batch {i+1:4d}/{len(loader)}] "
                  f"loss={total_loss/total:.3f}  acc={100.*correct/total:.2f}%")

    if first_batch_diagnostics is not None:
        d = first_batch_diagnostics
        print(f"  [Init check] epoch1 batch0  logit mean={d['logit_mean']:+.3f}  "
              f"std={d['logit_std']:.3f}  (健康: |mean|<2 且 std<5)")
        print(f"  [Init check] hyper_net grad sum = {d['hyper_grad']:.4e}  "
              f"theta_0 grad sum = {d['theta0_grad']:.4e}")
        print(f"               (两者都应 > 0; MIP 下 hyper_grad 通常远小于 theta0_grad)")

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
    注意: 这里 sex / race 既参与模型 (作为超网络条件), 也作为分组键。
    与 image-only 版的对照实验逻辑稍有不同 — 但 fairness gap 仍然有意义,
    它衡量的是"显式给属性 + hypernetwork 融合"是否减少了子群差异。
    """
    model.eval()
    all_logits, all_labels, all_sexes, all_races = [], [], [], []

    for images, labels, sexes, races in loader:
        images = images.to(DEVICE, non_blocking=True)
        sexes_gpu = sexes.long().to(DEVICE)
        races_gpu = races.long().to(DEVICE)

        logits = model(images, sexes_gpu, races_gpu).squeeze(1)   # [B]
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

    model = HyperFusionPreactivResNet18(
        in_channels=3, n_outputs=NUM_CLASSES, init_features=16,
        num_sex=NUM_SEX, num_race=NUM_RACE,
        sex_embed=SEX_EMBED, race_embed=RACE_EMBED,
        hyper_hidden=HYPER_HIDDEN,
        mip_scale=MIP_SCALE, use_l2_norm=USE_L2_NORM,
    ).to(DEVICE)

    main_params = sum(p.numel() for n, p in model.named_parameters()
                      if not n.startswith("hyper_net"))
    hyper_params = sum(p.numel() for n, p in model.named_parameters()
                       if n.startswith("hyper_net"))

    print(f"\nModel : HyperFusionPreactivResNet18  (hyper layer4[0].downsample)")
    print(f"        main params  = {main_params:,}")
    print(f"        hyper params = {hyper_params:,}")
    print(f"        total params = {main_params + hyper_params:,}")
    print(f"Optim : AdamW(lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
    print(f"Misc  : grad_clip_norm={GRAD_CLIP_NORM}  "
           f"mip_scale={MIP_SCALE}  use_l2_norm={USE_L2_NORM}")

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
    # 对照实验视角:
    #   - image-only      : 模型不见属性, 评估子群差异 (隐式偏差)
    #   - attrconcat      : 属性以空间通道拼接进入模型
    #   - hyperfusion(本) : 属性通过超网络生成 layer4 downsample 权重
    # 三者对比可以看出 hypernetwork fusion 是减少还是放大了 fairness gap.
    logits, labels, sexes, races = collect_predictions(model, test_loader)
    print_fairness_report(
        y_true=labels, y_score=logits,
        sex=sexes, race=races,
        threshold=0.0,
    )


if __name__ == "__main__":
    main()