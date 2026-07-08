"""
共享训练循环 harness（比较协议 A0.4）
====================================
把 5 数据集 × {ERM, HyperHead, HyperFusion, HyperAdapt} 十余个训练脚本中**结构 95% 相同**的
训练循环（train_one_epoch / evaluate / 早停 / 双 checkpoint / 逐 epoch val 日志 / 测试集公平性报告）
抽成单一 `run_training`，各脚本仅保留数据集特有的部分（transforms / dataloader 构建 / 属性过滤 /
重采样）并注入三个回调：

  - unpack_fn(batch)  -> (images, labels, attrs)  : **依数据集**（loader 元组数不同）。
        attrs 是 {子集 of "sex"/"race"/"age"/"skin" -> CPU Tensor}，键名与 subgroup_auc /
        val_log 的 kwargs 完全一致，可直接 splat；即便 image-only 模型不吃这些属性，也照样返回，
        供公平性分组与子群 AUC 向量使用。
  - forward_fn(model, images, attrs) -> logits     : **依方法**（模型吃哪些属性不同）。
        ERM 忽略 attrs；HN 从 attrs 取所需属性并 .to(device) 后喂入。
  - fairness_report_fn(y_true, y_score, attrs)      : **依数据集**（公平性口径不同）。
        末尾对两个 checkpoint 在测试集打印分组公平性报告。

harness 与 subgroup_auc.py（子群向量）/ val_log.py（JSONL 日志）/ run.py（run_id + checkpoint 命名
+ seed）协同：超参 lr/wd 取自 hparam（HParamConfig，覆盖 yaml），checkpoint 与 val 日志按
run_id 唯一命名，故超参搜索（6 config）与多 seed 确认互不覆盖。

模块级提供常见 unpack / forward 回调，各脚本直接复用，无需重写：
  unpack: unpack_skin(3 元组) / unpack_sex_age(4 元组) / unpack_sex_race_age(5 元组)
  forward: forward_image_only / forward_skin / forward_age / forward_sex_race_age
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.training.harness.hparam_grid import HParamConfig
from src.training.harness.predictions import default_prediction_path, save_predictions, val_prediction_path
from src.training.harness.run import checkpoint_path, run_id, seed_everything  # noqa: F401 (seed_everything re-export)
from src.training.harness.subgroup_auc import subgroup_auc_vector, vector_worst_case
from src.training.harness.swad import (
    DEFAULT_N_CONVERGE, DEFAULT_N_TOLERANCE, DEFAULT_TOLERANCE_RATIO,
    swad_average_over_interval,
)
from src.training.harness.val_log import (
    append_val_record, default_val_log_path, record_from_vector,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 回调类型别名
Batch = tuple
Attrs = dict[str, torch.Tensor]
UnpackFn = Callable[[Batch], tuple[torch.Tensor, torch.Tensor, Attrs]]
ForwardFn = Callable[[nn.Module, torch.Tensor, Attrs], torch.Tensor]
FairnessReportFn = Callable[[np.ndarray, np.ndarray, dict[str, np.ndarray]], None]


# ============================================================
# 通用 unpack 回调（依数据集：loader 元组数）
# ============================================================
def unpack_skin(batch: Batch) -> tuple[torch.Tensor, torch.Tensor, Attrs]:
    """Fitzpatrick17k：loader 返回 (image, label, skin) → attrs={skin}。"""
    images, labels, skin = batch
    return images, labels, {"skin": skin}


def unpack_sex_age(batch: Batch) -> tuple[torch.Tensor, torch.Tensor, Attrs]:
    """HAM10000 / PAPILA：loader 返回 (image, label, sex, age_group) → attrs={sex, age}。"""
    images, labels, sex, age = batch
    return images, labels, {"sex": sex, "age": age}


def unpack_sex_race_age(batch: Batch) -> tuple[torch.Tensor, torch.Tensor, Attrs]:
    """MIMIC / CheXpert：loader 返回 (image, label, sex, race, age) → attrs={sex, race, age}。"""
    images, labels, sex, race, age = batch
    return images, labels, {"sex": sex, "race": race, "age": age}


def unpack_asyn(batch: Batch) -> tuple[torch.Tensor, torch.Tensor, Attrs]:
    """MIMIC-synth（R1）：loader 返回 (image, label, a_syn) → attrs={a_syn}。"""
    images, labels, a_syn = batch
    return images, labels, {"a_syn": a_syn}


# ============================================================
# 通用 forward 回调（依方法：模型吃哪些属性）
# ============================================================
def forward_image_only(model: nn.Module, images: torch.Tensor, attrs: Attrs) -> torch.Tensor:
    """ERM / image-only baseline：忽略属性，仅吃图像。"""
    return model(images)


def forward_skin(model: nn.Module, images: torch.Tensor, attrs: Attrs) -> torch.Tensor:
    """Fitzpatrick HN：model(image, skin)。"""
    return model(images, attrs["skin"].to(images.device, non_blocking=True))


def forward_age(model: nn.Module, images: torch.Tensor, attrs: Attrs) -> torch.Tensor:
    """HAM / PAPILA HN：model(image, age)（单一 age 通路）。"""
    return model(images, attrs["age"].to(images.device, non_blocking=True))


def forward_sex_race_age(model: nn.Module, images: torch.Tensor, attrs: Attrs) -> torch.Tensor:
    """MIMIC / CheXpert HN：model(image, sex, race, age)。"""
    d = images.device
    return model(
        images,
        attrs["sex"].to(d, non_blocking=True),
        attrs["race"].to(d, non_blocking=True),
        attrs["age"].to(d, non_blocking=True),
    )


def forward_asyn(model: nn.Module, images: torch.Tensor, attrs: Attrs) -> torch.Tensor:
    """MIMIC-synth HN（R1）：model(image, a_syn)（单一二值合成属性通路，复用 *Age(num_age=2)）。"""
    return model(images, attrs["a_syn"].to(images.device, non_blocking=True))


# ============================================================
# 推理 / 评估
# ============================================================
@dataclass
class EvalResult:
    """一次验证/测试集评估的完整结果。"""

    loss: float
    acc: float
    overall_auc: float
    wc_auc: float | None
    vector: "dict"                       # subgroup_auc_vector 的返回（OrderedDict）
    logits: np.ndarray
    labels: np.ndarray
    attrs: dict[str, np.ndarray]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    grad_clip_norm: float,
    epoch: int,
    unpack_fn: UnpackFn,
    forward_fn: ForwardFn,
) -> tuple[float, float]:
    """跑一个训练 epoch，返回 (平均 loss, accuracy %)。逻辑与各脚本原实现一致，仅经回调泛化。"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for i, batch in enumerate(loader):
        images, labels, attrs = unpack_fn(batch)
        images = images.to(DEVICE, non_blocking=True)
        label_col = labels.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        logits = forward_fn(model, images, attrs)         # [B, 1]
        loss = criterion(logits, label_col)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        correct += (logits > 0).float().eq(label_col).sum().item()
        total += bs

        if (i + 1) % 20 == 0:
            print(f"  [Epoch {epoch} | Batch {i + 1:4d}/{len(loader)}] "
                  f"loss={total_loss / total:.4f}  acc={100. * correct / total:.2f}%")

    return total_loss / total, 100. * correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    dataset: str,
    unpack_fn: UnpackFn,
    forward_fn: ForwardFn,
) -> EvalResult:
    """
    在给定 loader 上评估，一次性算出 loss/acc/Overall AUC/worst-case + 完整子群 AUC 向量。

    worst-case 与向量统一由 subgroup_auc 计算（与 fairness 模块口径一致，A0.2 已断言）；
    向量随 EvalResult 返回，供 val 日志复用、避免二次计算。
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    attrs_all: dict[str, list[np.ndarray]] = {}

    for batch in loader:
        images, labels, attrs = unpack_fn(batch)
        images = images.to(DEVICE, non_blocking=True)
        label_col = labels.float().unsqueeze(1).to(DEVICE)

        logits = forward_fn(model, images, attrs)
        total_loss += criterion(logits, label_col).item() * images.size(0)
        correct += (logits > 0).float().eq(label_col).sum().item()
        total += images.size(0)

        logits_all.append(logits.squeeze(1).cpu().numpy())
        labels_all.append(labels.numpy())
        for k, v in attrs.items():
            attrs_all.setdefault(k, []).append(v.numpy())

    logits_np = np.concatenate(logits_all)
    labels_np = np.concatenate(labels_all)
    attrs_np = {k: np.concatenate(v) for k, v in attrs_all.items()}

    overall = float(roc_auc_score(labels_np, logits_np))
    vec = subgroup_auc_vector(dataset, labels_np, logits_np, **attrs_np)
    wc = vector_worst_case(vec)

    return EvalResult(
        loss=total_loss / total, acc=100. * correct / total,
        overall_auc=overall, wc_auc=wc, vector=vec,
        logits=logits_np, labels=labels_np, attrs=attrs_np,
    )


@torch.no_grad()
def evaluate_checkpoint(
    model: nn.Module,
    ckpt_path: Path,
    test_loader: DataLoader,
    criterion: nn.Module,
    dataset: str,
    unpack_fn: UnpackFn,
    forward_fn: ForwardFn,
    fairness_report_fn: FairnessReportFn,
    label: str,
) -> EvalResult:
    """加载 checkpoint，在测试集评估并打印整体指标 + 数据集特有的分组公平性报告；返回 EvalResult（供落盘预测）。"""
    print(f"\n{'#' * 70}\n# Evaluating checkpoint: {label} ({ckpt_path.name})\n{'#' * 70}")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    er = evaluate(model, test_loader, criterion, dataset, unpack_fn, forward_fn)
    wc_str = f"{er.wc_auc:.4f}" if er.wc_auc is not None else "N/A"
    print(f"Test loss={er.loss:.4f}  accuracy={er.acc:.2f}%  AUC={er.overall_auc:.4f}  "
          f"Worst-case AUC={wc_str}")
    # 阈值 0：logits>0 判正类（与各脚本一致，CheXpert 等极不均衡数据 acc 无意义、AUC 为主指标）
    fairness_report_fn(er.labels, er.logits, er.attrs)
    return er


# ============================================================
# SWAD 收尾（比较协议 A3.2）：从 ERM 训练派生权重平均模型并评估
# ============================================================
@torch.no_grad()
def _finalize_swad(
    *,
    model: nn.Module,
    val_losses: list[float],
    state_dicts: list[dict[str, torch.Tensor]],
    output_dir: Path,
    config_tag: str,
    seed: int,
    dataset: str,
    cfg: dict,
    criterion: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    unpack_fn: UnpackFn,
    forward_fn: ForwardFn,
    fairness_report_fn: FairnessReportFn,
    n_converge: int,
    n_tolerance: int,
    tolerance_ratio: float,
) -> dict:
    """
    A3.2：把逐 epoch 收集的 (val_loss, state_dict) 做 SWAD loss-valley 权重平均，落盘并在
    val/test 评估。SWAD 是 ERM 的**派生**基线（method="swad"，同 config/seed），产出单一平均模型。

    命名：checkpoint 与 val 日志以 run_id("swad", config_tag, seed) 为基名，与 ERM 区分开，
    供 A1/A2 / D 阶段按 method="swad" 寻址。写一条 val 记录（epoch = 平均区间末 epoch 的 1-based 值）
    承载平均模型的验证集完整子群向量。

    Returns:
        {swad_ckpt, swad_val_log, swad_interval:(t_s,t_e,n_averaged), swad_val_overall_auc, swad_val_wc_auc}。
    """
    print(f"\n{'#' * 70}\n# SWAD 权重平均（派生自 ERM，method=swad）\n{'#' * 70}")
    averaged_sd, interval = swad_average_over_interval(
        val_losses, state_dicts,
        n_converge=n_converge, n_tolerance=n_tolerance, tolerance_ratio=tolerance_ratio,
    )
    # 区间 0-based → 报告用 1-based epoch（与训练打印一致）
    print(f"loss valley 区间: epoch {interval.t_s + 1}..{interval.t_e + 1} "
          f"(共 {interval.n_averaged} 个，L_min={interval.l_min:.4f}, thr={interval.threshold:.4f})")

    swad_rid = run_id("swad", config_tag, seed)
    swad_ckpt = output_dir / f"{swad_rid}_averaged.pth"
    torch.save(averaged_sd, swad_ckpt)
    print(f"已保存 SWAD 平均权重: {swad_ckpt.name}")

    # 把平均权重载入模型副本评估（不污染主 model 的 device 状态）
    model.load_state_dict(averaged_sd)
    # --- 验证集：写一条 val 记录（承载平均模型的子群向量）---
    er_val = evaluate(model, val_loader, criterion, dataset, unpack_fn, forward_fn)
    swad_log = default_val_log_path(output_dir, "swad", config_tag, seed)
    if swad_log.exists():
        swad_log.unlink()
    rec = record_from_vector(
        er_val.vector, dataset=dataset, method="swad", config_tag=config_tag,
        lr=cfg["training"].get("learning_rate", 0.0), wd=cfg["training"].get("weight_decay", 0.0),
        seed=seed, epoch=interval.t_e + 1, overall_auc=er_val.overall_auc,
    )
    append_val_record(swad_log, rec)
    wc_str = f"{er_val.wc_auc:.4f}" if er_val.wc_auc is not None else "N/A"
    print(f"SWAD val: AUC={er_val.overall_auc:.4f} worst-case AUC={wc_str}")

    # --- 测试集：完整指标 + 分组公平性报告 + 落盘逐样本预测（method="swad"）---
    er_test = evaluate(model, test_loader, criterion, dataset, unpack_fn, forward_fn)
    wc_test = f"{er_test.wc_auc:.4f}" if er_test.wc_auc is not None else "N/A"
    print(f"\n{'#' * 70}\n# Evaluating SWAD averaged model (test)\n{'#' * 70}")
    print(f"Test loss={er_test.loss:.4f}  accuracy={er_test.acc:.2f}%  AUC={er_test.overall_auc:.4f}  "
          f"Worst-case AUC={wc_test}")
    fairness_report_fn(er_test.labels, er_test.logits, er_test.attrs)
    # SWAD 是单一平均模型（无 overall/worstcase 之分）：预测存到 "overall" 槽（唯一 selection）
    swad_pred = default_prediction_path(output_dir, "swad", config_tag, seed, "overall")
    save_predictions(swad_pred, er_test.labels, er_test.logits, er_test.attrs)
    print(f"已落盘 SWAD test 预测: {swad_pred.name}")

    return {
        "swad_ckpt": str(swad_ckpt),
        "swad_val_log": str(swad_log),
        "swad_pred": str(swad_pred),
        "swad_interval": (interval.t_s + 1, interval.t_e + 1, interval.n_averaged),
        "swad_val_overall_auc": float(er_val.overall_auc),
        "swad_val_wc_auc": (None if er_val.wc_auc is None else float(er_val.wc_auc)),
    }


# ============================================================
# 主编排：run_training
# ============================================================
def run_training(
    *,
    dataset: str,
    method: str,
    output_dir: Path | str,
    cfg: dict,
    hparam: HParamConfig,
    seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    build_model: Callable[[], nn.Module],
    unpack_fn: UnpackFn,
    forward_fn: ForwardFn,
    fairness_report_fn: FairnessReportFn,
    swad: bool = False,
    swad_n_converge: int = DEFAULT_N_CONVERGE,
    swad_n_tolerance: int = DEFAULT_N_TOLERANCE,
    swad_tolerance_ratio: float = DEFAULT_TOLERANCE_RATIO,
) -> dict:
    """
    统一训练编排：AdamW(lr,wd 来自 hparam) + BCEWithLogitsLoss + grad_clip，逐 epoch 评估、
    写 val 日志、早停、按 Overall / Worst-case 两策略各存一个 checkpoint，末尾在测试集评估两者。

    **超参来源**：lr/wd 取自 hparam（覆盖 cfg 的 learning_rate/weight_decay，构成超参网格搜索）；
    batch_size / num_epochs / grad_clip / early_stopping 仍取自 cfg（且 loader 已按 cfg.batch_size
    在调用方构建）。

    **播种约定**：调用方须在**构建 loader 之前**调用 run.seed_everything(seed)（loader shuffle /
    sampler 依赖 RNG）；本函数只在此之后构建模型，不再重复播种。

    **日志/权重命名**：均按 run.run_id(method, hparam.tag, seed)，超参搜索与多 seed 互不覆盖；
    val 日志每次运行前清空重写（避免 append 到旧 run 记录）。

    Args:
        dataset: {"mimic","chexpert","ham10000","fitzpatrick","papila"}。
        method : {"erm","hyperhead","hyperfusion","hyperadapt"}。
        output_dir: 该数据集 outputs 根（如 outputs/ham10000）。
        cfg: 该数据集 yaml 配置（training / early_stopping 段）。
        hparam: 超参配置（提供 lr/wd/config_tag）。
        seed: 随机种子（仅用于命名与日志记录；实际播种由调用方在建 loader 前完成）。
        train/val/test_loader: 已构建好的 DataLoader（含 transforms / 过滤 / 重采样）。
        build_model: 无参工厂，返回未搬到 device 的模型（本函数负责 .to(DEVICE)）。
        unpack_fn / forward_fn / fairness_report_fn: 见模块 docstring。
        swad: 是否额外派生 SWAD 权重平均模型（比较协议 A3.2）。为 True 时逐 epoch 缓存 CPU 权重，
            训练后做 loss-valley 平均并在 val/test 评估。**仅对 ERM 有意义**（协议：SWAD 套在 ERM 上、
            不与 HN 组合），method != "erm" 时会打印警告但仍执行。默认 False（HN / 搜索阶段不开销）。
        swad_n_converge / swad_n_tolerance / swad_tolerance_ratio: SWAD 超参（见 harness/swad.py）。

    Returns:
        summary dict：{best_val_overall_auc, best_val_wc_auc, best_overall_epoch,
                       best_worstcase_epoch, val_log_path, ckpt_overall, ckpt_worstcase}；
                      swad=True 时额外含 {swad_ckpt, swad_val_log, swad_interval,
                       swad_val_overall_auc, swad_val_wc_auc}。
    """
    train_cfg = cfg["training"]
    es_cfg = cfg["early_stopping"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_overall = checkpoint_path(output_dir, method, hparam.tag, seed, "overall")
    ckpt_worstcase = checkpoint_path(output_dir, method, hparam.tag, seed, "worstcase")
    log_path = default_val_log_path(output_dir, method, hparam.tag, seed)
    # 本 run 的 val 日志清空重写（防止重跑时 append 到旧记录，破坏 A1 的 argmax 选择）
    if log_path.exists():
        log_path.unlink()

    model = build_model().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {DEVICE}")
    print(f"Run   : dataset={dataset} method={method} config={hparam.tag} seed={seed}")
    print(f"Model : {type(model).__name__}  params={n_params:,}")
    # HN 模型若提供 param_breakdown，额外打印 backbone/hyper 拆分（与原脚本一致）
    if hasattr(model, "param_breakdown"):
        n_total, n_hyper, n_backbone = model.param_breakdown()
        print(f"        (backbone {n_backbone:,} + hyper {n_hyper:,})")
    print(f"Optim : AdamW(lr={hparam.lr:.0e}, weight_decay={hparam.wd:.0e})  "
          f"[来自超参网格 config={hparam.tag}]")
    print(f"Misc  : grad_clip_norm={train_cfg['grad_clip_norm']}, batch_size={train_cfg['batch_size']}")
    print(f"Early stopping: monitor={es_cfg['monitor']} (mode={es_cfg['mode']}), "
          f"patience={es_cfg['patience']}, min_delta={es_cfg['min_delta']}")
    print(f"Val log: {log_path}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=hparam.lr, weight_decay=hparam.wd)

    best_val_auc = -np.inf
    best_val_wc_auc = -np.inf
    best_overall_epoch = 0
    best_worstcase_epoch = 0
    epochs_no_improve = 0
    max_epochs = train_cfg["num_epochs"]
    patience = es_cfg["patience"]
    min_delta = es_cfg["min_delta"]

    # SWAD（A3.2）：逐 epoch 缓存 (val_loss, CPU state_dict) 供训练后 loss-valley 权重平均
    if swad and method != "erm":
        print(f"⚠️  swad=True 但 method={method!r}（非 erm）：SWAD 协议上仅套在 ERM 上，仍照常执行。")
    swad_val_losses: list[float] = []
    swad_state_dicts: list[dict[str, torch.Tensor]] = []

    print(f"\nTraining for up to {max_epochs} epochs (early stopping enabled)...\n" + "=" * 70)
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer,
            train_cfg["grad_clip_norm"], epoch, unpack_fn, forward_fn,
        )
        er = evaluate(model, val_loader, criterion, dataset, unpack_fn, forward_fn)
        dt = time.time() - t0

        # 逐 epoch 写完整子群 AUC 向量到 val 日志（复用 evaluate 已算的 vector，不二次计算）
        rec = record_from_vector(
            er.vector, dataset=dataset, method=method, config_tag=hparam.tag,
            lr=hparam.lr, wd=hparam.wd, seed=seed, epoch=epoch, overall_auc=er.overall_auc,
        )
        append_val_record(log_path, rec)

        # SWAD：缓存本 epoch 的 val loss 与 CPU 权重快照（放 CPU 避免占 GPU 显存）
        if swad:
            swad_val_losses.append(er.loss)
            swad_state_dicts.append(
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            )

        wc_val = er.wc_auc if er.wc_auc is not None else -np.inf
        wc_str = f"{er.wc_auc:.4f}" if er.wc_auc is not None else "N/A"
        print(f"Epoch {epoch:2d}/{max_epochs} [{dt:.1f}s] | "
              f"train loss={train_loss:.4f} acc={train_acc:.2f}% | "
              f"val loss={er.loss:.4f} acc={er.acc:.2f}% AUC={er.overall_auc:.4f} "
              f"worst-case AUC={wc_str}")

        # --- model selection: best worst-case AUC checkpoint ---
        if wc_val > best_val_wc_auc + min_delta:
            best_val_wc_auc = wc_val
            best_worstcase_epoch = epoch
            torch.save(model.state_dict(), ckpt_worstcase)
            print(f"  -> New best val worst-case AUC ({er.wc_auc:.4f}); saved {ckpt_worstcase.name}")

        # --- model selection + early stopping: best overall AUC checkpoint ---
        if er.overall_auc > best_val_auc + min_delta:
            best_val_auc = er.overall_auc
            best_overall_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_overall)
            print(f"  -> New best val overall AUC ({best_val_auc:.4f}); saved {ckpt_overall.name}")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement for {epochs_no_improve}/{patience} epoch(s) "
                  f"(best overall so far: {best_val_auc:.4f})")
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best val overall AUC={best_val_auc:.4f}, "
                      f"best val worst-case AUC={best_val_wc_auc:.4f}).")
                break
        print("-" * 70)

    # --- 两种 model selection 策略的最优 checkpoint 在测试集上评估 + 落盘逐样本预测（供 A2/A4/A5/D）---
    er_overall = evaluate_checkpoint(model, ckpt_overall, test_loader, criterion, dataset,
                                     unpack_fn, forward_fn, fairness_report_fn, label="Overall AUC selection")
    er_worstcase = evaluate_checkpoint(model, ckpt_worstcase, test_loader, criterion, dataset,
                                       unpack_fn, forward_fn, fairness_report_fn, label="Worst-case AUC selection")
    pred_overall = default_prediction_path(output_dir, method, hparam.tag, seed, "overall")
    pred_worstcase = default_prediction_path(output_dir, method, hparam.tag, seed, "worstcase")
    save_predictions(pred_overall, er_overall.labels, er_overall.logits, er_overall.attrs)
    save_predictions(pred_worstcase, er_worstcase.labels, er_worstcase.logits, er_worstcase.attrs)
    print(f"已落盘 test 预测: {pred_overall.name} / {pred_worstcase.name}")

    # ROC 后处理（A4 / C*.8）需 val 预测定 deprived 子群与选 margin θ：用 overall-selection
    # checkpoint 在 val 集落盘一份 val 预测（selection="val_overall"），使 ROC 派生保持数据集无关。
    er_val_overall = evaluate_checkpoint(model, ckpt_overall, val_loader, criterion, dataset,
                                         unpack_fn, forward_fn, fairness_report_fn,
                                         label="[val] Overall AUC selection")
    pred_val_overall = val_prediction_path(output_dir, method, hparam.tag, seed)
    save_predictions(pred_val_overall, er_val_overall.labels, er_val_overall.logits, er_val_overall.attrs)
    print(f"已落盘 val 预测（供 ROC）: {pred_val_overall.name}")

    summary = {
        "best_val_overall_auc": float(best_val_auc),
        "best_val_wc_auc": (None if best_val_wc_auc == -np.inf else float(best_val_wc_auc)),
        "best_overall_epoch": best_overall_epoch,
        "best_worstcase_epoch": best_worstcase_epoch,
        "val_log_path": str(log_path),
        "ckpt_overall": str(ckpt_overall),
        "ckpt_worstcase": str(ckpt_worstcase),
        "pred_overall": str(pred_overall),
        "pred_worstcase": str(pred_worstcase),
        "pred_val_overall": str(pred_val_overall),
    }

    # --- SWAD 派生（A3.2）：loss-valley 权重平均 + val/test 评估 ---
    if swad:
        summary.update(_finalize_swad(
            model=model, val_losses=swad_val_losses, state_dicts=swad_state_dicts,
            output_dir=output_dir, config_tag=hparam.tag, seed=seed, dataset=dataset, cfg=cfg,
            criterion=criterion, val_loader=val_loader, test_loader=test_loader,
            unpack_fn=unpack_fn, forward_fn=forward_fn, fairness_report_fn=fairness_report_fn,
            n_converge=swad_n_converge, n_tolerance=swad_n_tolerance,
            tolerance_ratio=swad_tolerance_ratio,
        ))

    return summary
