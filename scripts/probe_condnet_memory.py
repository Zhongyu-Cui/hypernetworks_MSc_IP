"""
E1 前置探针：四个 cell 在锁定 batch=128 下的显存与单步耗时
============================================================
plan §1 锁死 `bs=128`，且项目纪律要求「重型 HN 显存对照必须等 batch」（曾被 batch=32 的 OOM
假象坑过、得出错误的「HyperAdapt 负结果」——见记忆 ham10000-conditional-mi-age-signal）。
C-Full 的逐样本 grouped-conv 会把卷积权重复制 batch 份，显存随 batch 线性增长，故在提交
~120 个训练 run 之前，必须先确认**最重的 C-Full 在 bs=128 下放得进目标分区**，并据此定分区。

本探针跑真实的前向+反向+optimizer step（含 AdamW 状态），报告各 cell 的峰值显存与单步耗时，
用随机张量（不读数据）以隔离出纯模型侧开销。

运行（GPU 节点，经 sbatch）：
    python scripts/probe_condnet_memory.py --batch_size 128
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from src.models.resnet18_condnet import ResNet18CondNetAge, build_base_fc_seed

LOCATIONS = ("none", "head", "deep", "full")


def probe(location: str, batch_size: int, steps: int = 5) -> tuple[float, float]:
    """
    跑若干训练步，返回 (峰值显存 GiB, 稳态单步秒数)。

    Args:
        location  : cell 开关。
        batch_size: 批大小（锁定 128）。
        steps     : 计时步数（首步含 cudnn 预热，单独排除）。

    Returns:
        (peak_mem_gib, sec_per_step)。
    """
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    seed = build_base_fc_seed("ham10000", fold=0, seed=42)
    model = ResNet18CondNetAge(location=location, num_age=4, pretrained=False,
                               base_fc_seed=seed).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    crit = nn.BCEWithLogitsLoss()
    model.train()

    image = torch.randn(batch_size, 3, 224, 224, device=device)
    age = torch.randint(0, 4, (batch_size,), device=device)
    label = torch.randint(0, 2, (batch_size, 1), device=device).float()

    for i in range(steps + 1):
        if i == 1:                                   # 首步含 cudnn 算法搜索/预热，不计时
            torch.cuda.synchronize()
            t0 = time.time()
        opt.zero_grad()
        logits = model(image, age) if model.needs_embedding else model(image)
        loss = crit(logits, label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    del model, opt, image, age, label
    torch.cuda.empty_cache()
    return peak, dt


def main() -> None:
    p = argparse.ArgumentParser(description="ResNet18CondNet 四 cell 显存/速度探针")
    p.add_argument("--batch_size", type=int, default=128, help="批大小（plan 锁定 128）。")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("需 GPU 运行（经 sbatch 提交）。")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"GPU: {name}  总显存 {total:.1f} GiB  |  batch_size={args.batch_size}\n")
    print(f"| {'cell':<8s} | {'峰值显存(GiB)':>13s} | {'占比':>6s} | {'秒/步':>7s} | {'相对ERM':>7s} |")

    base_dt = None
    for loc in LOCATIONS:
        try:
            mem, dt = probe(loc, args.batch_size)
        except torch.cuda.OutOfMemoryError:
            print(f"| {loc:<8s} | {'OOM':>13s} | {'-':>6s} | {'-':>7s} | {'-':>7s} |")
            torch.cuda.empty_cache()
            continue
        base_dt = base_dt or dt
        print(f"| {loc:<8s} | {mem:>13.2f} | {mem / total:>5.0%} | {dt:>7.3f} | {dt / base_dt:>6.2f}x |")

    print("\n判读：峰值占比 <~70% 方可安心（真实训练另有 DataLoader/碎片开销）；"
          "若 C-Full 在 24GiB 卡超标 ⇒ 全部 cell 改投 gpus48（**不得只给 C-Full 降 batch**，"
          "否则破坏等-batch 对照）。")


if __name__ == "__main__":
    main()
