"""
Train ResNet18-HyperAdapt (Frozen Backbone) on MIMIC-CXR for No Finding
=======================================================================
本脚本是 train_mimic_hyperadapt.py 的「冻结 backbone」变体：模型、数据、优化器、
早停、两种 model selection 等全部设置与原 HyperAdapt 实验严格一致，**唯一区别**是
构造模型时置 `freeze_backbone=True`——冻结特征提取 backbone (stem conv1/bn1 + 各
stage 内每个 block 的 backbone conv1/bn1/conv2/bn2/downsample) 的可学习参数，使梯度
只能流入超网络生成器与任务头 fc。

动机：原 HyperAdapt (backbone 联合微调) 整体 AUC 与 baseline 持平、公平性未稳健提升，
且 Δθ 近乎停留在 0 (两种 selection 退化为同一 epoch、AUC 与 baseline 逐位重合)。推测
联合微调时强力 backbone 自己吸走全部梯度、缺乏「必须使用条件」的激励，conditioning
通路退化为冗余分支。论文里 backbone 是冻结的 (只训 adapter ϕ)。本实验复刻该设定，检验
冻结 backbone 是否把梯度逼入 adapter、从而真正激活条件注入并改善子群可靠性。

注: 与原脚本一致地用两种 model selection (best_overall / best_worstcase) 在测试集评估,
但 checkpoint / 输出命名加 `_frozen` 后缀, 与原 HyperAdapt 结果分开存放, 互不覆盖。
优化器仅接收 requires_grad=True 的参数 (冻结参数本就不会被更新, 这里显式过滤以免被
AdamW weight decay 误伤, 并使日志中的可训练参数量清晰)。

运行方式:
    重型任务，必须通过 sbatch 提交到 SLURM 集群 (见 slurm/train_mimic_hyperadapt_frozen.sh)。
    通过 --seed 指定随机种子（默认取 configs/mimic_cxr_baseline.yaml 中 training.seeds[0]）。
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from src.models.resnet18_hyperadapt import ResNet18HyperAdapt

# 训练 / 评估循环与原 HyperAdapt 脚本完全一致，直接复用以确保严格可比
from src.training.train_mimic_hyperadapt import (
    build_transforms,
    evaluate,
    evaluate_checkpoint,
    get_dataloaders,
    load_config,
    train_one_epoch,
)

# ============================================================
# 路径
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "mimic_cxr_baseline.yaml"

# 模型权重属于「由集群生成的大文件」，按项目规范存放在仓库外的根目录 outputs/，不进仓库
OUTPUT_DIR = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/mimic_cxr")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    """解析命令行参数（目前仅 --seed，用于多 seed 重复实验）。"""
    parser = argparse.ArgumentParser(
        description="Train ResNet18-HyperAdapt (frozen backbone) on MIMIC-CXR (No Finding)"
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子；缺省时取 configs/mimic_cxr_baseline.yaml 中 training.seeds[0]")
    return parser.parse_args()


# ============================================================
# 主函数（与 train_mimic_hyperadapt.main 一致，仅冻结 backbone + 改命名）
# ============================================================
def main() -> None:
    args = parse_args()
    cfg = load_config(CONFIG_PATH)
    train_cfg = cfg["training"]
    es_cfg = cfg["early_stopping"]

    seed = args.seed if args.seed is not None else train_cfg["seeds"][0]
    torch.manual_seed(seed)

    print(f"Device: {DEVICE}")
    print(f"Seed  : {seed}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 与原 HyperAdapt 结果分开：checkpoint 命名加 _frozen 后缀
    ckpt_overall_path = OUTPUT_DIR / f"hyperadapt_frozen_no_finding_seed{seed}_best_overall.pth"
    ckpt_worstcase_path = OUTPUT_DIR / f"hyperadapt_frozen_no_finding_seed{seed}_best_worstcase.pth"

    print("\nLoading MIMIC-CXR (No Finding, U-Zeros)...")
    train_transform, eval_transform = build_transforms(cfg)
    train_loader, val_loader, test_loader = get_dataloaders(cfg, train_transform, eval_transform)
    print(f"  Train: {len(train_loader.dataset):,} images")
    print(f"  Val  : {len(val_loader.dataset):,}")
    print(f"  Test : {len(test_loader.dataset):,}")

    # 唯一区别：freeze_backbone=True —— 冻结特征提取 backbone，梯度只流入 adapter + fc
    model = ResNet18HyperAdapt(
        num_classes=1,
        pretrained=cfg["model"]["pretrained"],
        freeze_backbone=True,
    ).to(DEVICE)

    n_total, n_adapter, n_backbone = model.param_breakdown()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = n_total - n_trainable
    print(f"\nModel : ResNet18-HyperAdapt (FROZEN backbone; 每个 conv + fc 由 sex/race/age 调制)")
    print(f"Params: {n_total:,} (backbone {n_backbone:,} + adapter {n_adapter:,})")
    print(f"        trainable {n_trainable:,} / frozen {n_frozen:,} "
          f"(backbone 卷积/BN 仿射已冻结; fc 与超网络生成器可训练)")
    print(f"Optim : AdamW(lr={train_cfg['learning_rate']}, weight_decay={train_cfg['weight_decay']}) "
          f"[仅可训练参数]")
    print(f"Misc  : grad_clip_norm={train_cfg['grad_clip_norm']}, batch_size={train_cfg['batch_size']}")
    print(f"Early stopping: monitor={es_cfg['monitor']} (mode={es_cfg['mode']}), "
          f"patience={es_cfg['patience']}, min_delta={es_cfg['min_delta']}")

    criterion = nn.BCEWithLogitsLoss()
    # 仅把 requires_grad=True 的参数交给优化器（冻结参数不参与更新 / weight decay）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params,
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
