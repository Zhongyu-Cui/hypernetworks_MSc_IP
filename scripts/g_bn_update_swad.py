"""
实验 G · BN 统计与官方 SWAD 对齐（`update_bn` 重估）+ 重推理
=============================================================
**动机**：本项目的 SWAD 权重平均（`harness/swad.py::average_state_dicts`）对 BatchNorm 的
`running_mean` / `running_var` 也做**算术平均**，而官方 SWAD（khanrc/swad，`domainbed/trainer.py`）
在拿到平均模型后调用 `swa_utils.update_bn(train_minibatches_iterator, swad_algorithm, n_steps=500)`
——重置 running stats 后用**训练数据前向 500 步重估**。这是 SWA 原论文（Izmailov et al., 2018）的
标准要求：平均权重产生的激活分布不等于任一单点的，旧统计必然失配；`running_var` 尤甚
（方差不线性可加，平均各点的 var ≠ 平均权重的 var）。

**待检验的假说**：`docs/hyperadapt_swad_fusion_full_report.md` §5.3.4② 实测 SWAD 系统性压低
calibration slope（M→C 1.094→0.843、C→M 0.718→0.594，四个「加 SWAD」对比 4/4 方向一致、
2/4 超过 σ_train），即**判别不变而输出刻度被压缩**——BN 统计失配正是这一模式的典型来源。
本脚本把 BN 处理与官方对齐后重推理，检验该读数是否消失。

**不需要重训**：`update_bn` 是无梯度前向，消费既有的 `*_averaged.pth`。

**严格的产物隔离**（既有冻结产物一律不覆盖）：
  · 新 checkpoint：`{run_id}_averaged_bnupd.pth`（与 `_averaged.pth` 并存于原 cv5 目录）。
  · ID 预测：`outputs/<ds>/cv5_bnupd/predictions/`（平行目录，非 `cv5/predictions/`）。
  · OOD 预测：`outputs/ood_cxr/<src>2<tgt>/cv5_full_target_bnupd/predictions/`。
  · method 名**不变**（`swad` / `hyperadapt_swad`），故分析脚本只需换根目录即可复用。

**每臂用自己的训练 loader**：BN 重估必须复现该模型**自己训练时**的数据分布，故
`swad` 走 `train_<ds>_resnet18.py`、`hyperadapt_swad` 走 `train_<ds>_hyperadapt.py`
（HAM 两者训练集不同：HyperAdapt 侧过滤 `age_group>=0`）。

运行（重型 GPU，经 sbatch 提交；见 slurm/g_bn_update.sh）：
    python scripts/g_bn_update_swad.py --dataset ham10000 --stage id
    python scripts/g_bn_update_swad.py --dataset mimic --stage ood --trials 0,1,2
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
from torch.utils.data import DataLoader

# 与 run_ood_cxr_full_target 同样的坑：多 worker × 多并发任务会耗尽 fd 上限
torch.multiprocessing.set_sharing_strategy("file_system")

from src.models.resnet18_hyperadapt import (
    ResNet18HyperAdapt, ResNet18HyperAdaptAge, ResNet18HyperAdaptSkin,
)
from src.models.resnet18_pretrained import ResNet18Pretrained
from src.training.harness.predictions import save_predictions
from src.training.harness.run import seed_everything
from src.training.harness.train_loop import (
    DEVICE, evaluate, forward_age, forward_image_only, forward_skin, forward_sex_race_age,
    unpack_sex_age, unpack_sex_race_age, unpack_skin,
)

OUTPUTS = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs")
ARMS = ("swad", "hyperadapt_swad")          # 仅权重平均派生的两臂需要 BN 重估
N_BN_STEPS = 500                            # 官方 trainer.py: n_steps = 500
CV = 5

# 数据集 → (harness dataset key, 输出根, 各臂的训练模块 / 模型工厂 / 回调)
DATASETS: dict[str, dict] = {
    "ham10000": {
        "ds_key": "ham10000", "out": OUTPUTS / "ham10000",
        "unpack": unpack_sex_age,
        "arms": {
            "swad": ("src.training.train_ham10000_resnet18",
                     lambda: ResNet18Pretrained(num_classes=1), forward_image_only),
            "hyperadapt_swad": ("src.training.train_ham10000_hyperadapt",
                                lambda: ResNet18HyperAdaptAge(num_classes=1, num_age=4,
                                                              pretrained=False), forward_age),
        },
    },
    "fitzpatrick": {
        "ds_key": "fitzpatrick", "out": OUTPUTS / "fitzpatrick",
        "unpack": unpack_skin,
        "arms": {
            "swad": ("src.training.train_fitzpatrick_resnet18",
                     lambda: ResNet18Pretrained(num_classes=1), forward_image_only),
            "hyperadapt_swad": ("src.training.train_fitzpatrick_hyperadapt",
                                lambda: ResNet18HyperAdaptSkin(num_classes=1, num_skin=6,
                                                               pretrained=False), forward_skin),
        },
    },
    "mimic": {
        "ds_key": "mimic", "out": OUTPUTS / "mimic_cxr",
        "unpack": unpack_sex_race_age,
        "arms": {
            "swad": ("src.training.train_mimic_resnet18",
                     lambda: ResNet18Pretrained(num_classes=1), forward_image_only),
            "hyperadapt_swad": ("src.training.train_mimic_hyperadapt",
                                lambda: ResNet18HyperAdapt(num_classes=1, pretrained=False),
                                forward_sex_race_age),
        },
    },
    "chexpert": {
        "ds_key": "chexpert", "out": OUTPUTS / "chexpert_cxr",
        "arms": {
            "swad": ("src.training.train_chexpert_resnet18",
                     lambda: ResNet18Pretrained(num_classes=1), forward_image_only),
            "hyperadapt_swad": ("src.training.train_chexpert_hyperadapt",
                                lambda: ResNet18HyperAdapt(num_classes=1, pretrained=False),
                                forward_sex_race_age),
        },
        "unpack": unpack_sex_race_age,
    },
}
# OOD 方向：source → (target, ood 子目录)
OOD_DIRECTION = {"mimic": ("chexpert", "mimic2chexpert"),
                 "chexpert": ("mimic", "chexpert2mimic")}


def seed_of(fold: int, trial: int) -> int:
    """OOD v2 预注册 §2.1 的 seed 编码：42 + fold + 10 × trial（trial 0 即 ID 的 42–46）。"""
    return 42 + fold + 10 * trial


@torch.no_grad()
def update_bn_stats(
    model: nn.Module, loader: DataLoader, unpack_fn: Callable, forward_fn: Callable,
    n_steps: int = N_BN_STEPS,
) -> int:
    """
    对齐官方 `swa_utils.update_bn`：重置全部 BN 的 running stats，用**训练 loader**（带增广）
    前向 `n_steps` 步以累积平均方式重估。

    与 PyTorch `torch.optim.swa_utils.update_bn` 的唯一差别是 forward 经回调转发——本项目的 HN
    需要 `model(image, *attrs)`，无法用 `model(input)` 的固定签名。

    `momentum=None` 使 BN 走**累积移动平均**（cumulative moving average），即 n_steps 个 batch 的
    无偏统计，而非指数滑动——这是官方/PyTorch 的做法，结果不依赖 batch 顺序。

    Args:
        model: 已载入平均权重的模型（会被就地修改）。
        loader: 训练 loader（shuffle + 增广，与该模型训练时一致）。
        unpack_fn / forward_fn: 该数据集/方法的 batch 解包与前向回调。
        n_steps: 前向步数（官方 500）。

    Returns:
        实际前向的 batch 数。
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            momenta[module] = module.momentum
            module.momentum = None          # 累积移动平均
    if not momenta:
        return 0

    was_training = model.training
    model.train()                            # BN 只在 train 模式更新 running stats
    steps = 0
    while steps < n_steps:
        for batch in loader:
            if steps >= n_steps:
                break
            images, _, attrs = unpack_fn(batch)
            forward_fn(model, images.to(DEVICE, non_blocking=True), attrs)
            steps += 1

    for module, momentum in momenta.items():
        module.momentum = momentum
    model.train(was_training)
    return steps


def build_loaders(module_name: str, cv: int, fold: int, batch_size: int) -> tuple:
    """
    按训练模块重建该折的 (train, val, test) loader —— 与原训练逐位同源（同 config / 同 transform /
    同 split_dir），仅覆盖 batch_size（cv5 训练统一 b128）。

    Args:
        module_name: `src.training.train_<ds>_<method>` 模块名。
        cv / fold: 折配置。
        batch_size: 覆盖 config 的 batch_size。

    Returns:
        (train_loader, val_loader, test_loader)。
    """
    mod = importlib.import_module(module_name)
    cfg = mod.load_config(mod.CONFIG_PATH)
    cfg["training"]["batch_size"] = batch_size
    split_dir, _ = mod.resolve_paths(cfg, cv, fold)
    train_tf, eval_tf = mod.build_transforms(cfg)
    return mod.get_dataloaders(cfg, split_dir, train_tf, eval_tf)


def bnupd_checkpoint(out_dir: Path, method: str, tag: str, seed: int) -> Path:
    """BN 重估后的 checkpoint 路径（与既有 `_averaged.pth` 并存，绝不覆盖）。"""
    return out_dir / "cv5" / f"{method}_{tag}_seed{seed}_averaged_bnupd.pth"


def make_bnupd_checkpoints(
    dataset: str, folds: list[int], trials: tuple[int, ...], batch_size: int,
    n_steps: int, force: bool,
) -> dict[tuple[str, int], Path]:
    """
    对指定 (臂 × fold × trial) 的平均权重做 BN 重估并落盘。

    Returns:
        {(method, seed): bnupd_ckpt_path}。
    """
    spec = DATASETS[dataset]
    cfg_tags = json.loads((spec["out"] / "cv5" / "selected_configs.json").read_text())["config"]
    made: dict[tuple[str, int], Path] = {}

    for method in ARMS:
        tag = cfg_tags[method]
        module_name, model_fn, forward_fn = spec["arms"][method]
        for trial in trials:
            for fold in folds:
                seed = seed_of(fold, trial)
                src_ckpt = spec["out"] / "cv5" / f"{method}_{tag}_seed{seed}_averaged.pth"
                dst_ckpt = bnupd_checkpoint(spec["out"], method, tag, seed)
                if not src_ckpt.exists():
                    raise FileNotFoundError(f"缺少平均权重：{src_ckpt}")
                made[(method, seed)] = dst_ckpt
                if dst_ckpt.exists() and not force:
                    print(f"  [BN] {method:<16} fold{fold} trial{trial} seed{seed}: 已存在，跳过")
                    continue

                # 逐 (fold, seed) 固定随机性：BN 重估要过带增广/shuffle 的训练流
                seed_everything(seed)
                train_loader, _, _ = build_loaders(module_name, CV, fold, batch_size)
                model = model_fn().to(DEVICE)
                model.load_state_dict(torch.load(src_ckpt, map_location=DEVICE))
                t0 = time.time()
                steps = update_bn_stats(model, train_loader, spec["unpack"], forward_fn, n_steps)
                torch.save(model.state_dict(), dst_ckpt)
                print(f"  [BN] {method:<16} fold{fold} trial{trial} seed{seed}: "
                      f"{steps} steps × b{batch_size} = {steps * batch_size:,} 张  "
                      f"[{time.time() - t0:.1f}s] → {dst_ckpt.name}")
    return made


def run_id_inference(dataset: str, folds: list[int], batch_size: int) -> None:
    """ID：BN 重估后的模型在**本折 test** 上重推理，落盘到 `cv5_bnupd/predictions/`。"""
    spec = DATASETS[dataset]
    cfg_tags = json.loads((spec["out"] / "cv5" / "selected_configs.json").read_text())["config"]
    pred_dir = spec["out"] / "cv5_bnupd" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.BCEWithLogitsLoss()

    for method in ARMS:
        tag = cfg_tags[method]
        module_name, model_fn, forward_fn = spec["arms"][method]
        for fold in folds:
            seed = seed_of(fold, 0)
            ckpt = bnupd_checkpoint(spec["out"], method, tag, seed)
            _, _, test_loader = build_loaders(module_name, CV, fold, batch_size)
            model = model_fn().to(DEVICE)
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            er = evaluate(model, test_loader, criterion, spec["ds_key"],
                          spec["unpack"], forward_fn)
            npz = pred_dir / f"{method}_{tag}_seed{seed}_overall.npz"
            save_predictions(npz, er.labels, er.logits, er.attrs)
            wc = f"{er.wc_auc:.4f}" if er.wc_auc is not None else "N/A"
            print(f"  [ID ] {method:<16} fold{fold}: Overall={er.overall_auc:.4f} worst={wc} "
                  f"n={len(er.labels):,} → {npz.name}")


def run_ood_inference(source: str, folds: list[int], trials: tuple[int, ...]) -> None:
    """OOD：BN 重估后的 source 模型评**完整 target**，落盘到 `cv5_full_target_bnupd/`。"""
    # 延迟 import：这两个模块只在 CXR 方向需要，且会构建 full-target CSV
    from src.training.eval_ood_cxr import evaluate_ood
    from src.training.run_ood_cxr_full_target import build_full_target_loader

    spec = DATASETS[source]
    target, sub = OOD_DIRECTION[source]
    cfg_tags = json.loads((spec["out"] / "cv5" / "selected_configs.json").read_text())["config"]
    pred_dir = OUTPUTS / "ood_cxr" / sub / "cv5_full_target_bnupd" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    loader, df = build_full_target_loader(target=target, num_workers=8)
    patient_id = df["patient_id"].values
    print(f"  完整 target({target}): n={len(df):,}  患者={df['patient_id'].nunique():,}")

    for method in ARMS:
        tag = cfg_tags[method]
        for trial in trials:
            for fold in folds:
                seed = seed_of(fold, trial)
                npz = pred_dir / f"{method}_{tag}_seed{seed}_overall.npz"
                if npz.exists():
                    print(f"  [OOD] {method:<16} fold{fold} trial{trial}: 已存在，跳过")
                    continue
                ckpt = bnupd_checkpoint(spec["out"], method, tag, seed)
                result = evaluate_ood(method, ckpt, source=source, target=target,
                                      test_loader=loader, report=False)
                er = result.eval_result
                assert len(er.labels) == len(patient_id), "预测数与 CSV 行数不一致"
                save_predictions(npz, er.labels, er.logits, er.attrs,
                                 extra={"patient_id": patient_id})
                wc = f"{er.wc_auc:.4f}" if er.wc_auc is not None else "N/A"
                print(f"  [OOD] {method:<16} fold{fold} trial{trial} seed{seed}: "
                      f"Overall={er.overall_auc:.4f} worst={wc} → {npz.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="实验 G：SWAD 的 BN 统计与官方对齐（update_bn）+ 重推理")
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--stage", required=True, choices=("id", "ood", "bn-only"),
                    help="id=BN重估+本折test推理；ood=BN重估+full-target推理；bn-only=只做BN重估")
    ap.add_argument("--trials", default="0", help="逗号分隔 trial 编号（OOD 用 0,1,2）")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--batch-size", type=int, default=128, help="cv5 训练统一 b128，勿改")
    ap.add_argument("--n-steps", type=int, default=N_BN_STEPS, help="BN 重估前向步数（官方 500）")
    ap.add_argument("--force", action="store_true", help="已存在的 bnupd checkpoint 也重算")
    args = ap.parse_args()

    # 坏 GPU 节点上 CUDA 初始化会静默失败并退化到 CPU（v2 记录的 mira05）；宁可立刻报错换节点
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用：本作业为重型 GPU 推理，拒绝静默退化到 CPU，请换节点重投。")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    folds = [int(x) for x in args.folds.split(",")]
    trials = tuple(int(t) for t in args.trials.split(","))
    if args.stage == "ood" and args.dataset not in OOD_DIRECTION:
        raise ValueError(f"--stage ood 仅支持 CXR 两库，收到 {args.dataset!r}")

    print(f"\n{'=' * 78}\nBN 对齐（update_bn，官方 n_steps={args.n_steps}）  "
          f"dataset={args.dataset}  stage={args.stage}  folds={folds}  trials={trials}\n{'=' * 78}")

    make_bnupd_checkpoints(args.dataset, folds, trials, args.batch_size, args.n_steps, args.force)
    if args.stage == "id":
        run_id_inference(args.dataset, folds, args.batch_size)
    elif args.stage == "ood":
        run_ood_inference(args.dataset, folds, trials)
    print("\n完成。分析：scripts/g_bn_update_analysis.py")


if __name__ == "__main__":
    main()
