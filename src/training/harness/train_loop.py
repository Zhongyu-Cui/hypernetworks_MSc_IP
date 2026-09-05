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
from src.training.harness.run import (  # noqa: F401 (seed_everything / swad_method_name re-export)
    checkpoint_path, run_id, seed_everything, swad_method_name,
)
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
# **分组属性白名单**：attrs 字典里只有这些键是 subgroup_auc_vector / fairness 模块的合法分组 kwarg。
# 实验 P（Predicted-Attribute HyperAdapt）会在 attrs 里额外放 `soft_<axis>` 的 [B, C] 概率矩阵作为
# 超网络的条件输入；它**不是分组属性**（分组永远用真值属性），故 splat 前必须过滤，落盘时改走
# save_predictions 的 `extra` 通道。见 docs/predicted_attribute_hyperadapt_plan.md §4/§5。
GROUP_ATTR_KEYS: frozenset[str] = frozenset({"sex", "race", "age", "skin", "a_syn"})

# 可选的**训练目标**替换（默认 None = 普通 BCE 均值）。签名 (logits, labels, attrs) -> 标量 loss，
# 使需要子群标签的目标（GroupDRO）能拿到 attrs。只作用于训练；val/test 恒用普通 BCE，保证早停、
# SWAD loss 谷、跨方法 loss 可比性不被目标函数差异污染。
ObjectiveFn = Callable[[torch.Tensor, torch.Tensor, Attrs], torch.Tensor]


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


def forward_sex_age(model: nn.Module, images: torch.Tensor, attrs: Attrs) -> torch.Tensor:
    """HAM 双属性 HN（对称性对照臂）：model(image, sex, age)——条件输入补齐为 worst-group
    评估分组变量全集（Sex / Age / Sex×Age），与 forward_age 的单一 age 通路构成单变量对照。"""
    d = images.device
    return model(
        images,
        attrs["sex"].to(d, non_blocking=True),
        attrs["age"].to(d, non_blocking=True),
    )


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


# ---- 实验 P（Predicted-Attribute）：条件输入是 g 的属性概率，分组属性仍为真值 ----
def unpack_skin_soft(batch: Batch) -> tuple[torch.Tensor, torch.Tensor, Attrs]:
    """Fitzpatrick + SoftAttrDataset：loader 返回 (image, label, skin, skin_prob)
    → attrs={skin(真值,分组用), soft_skin([B,6] 条件输入)}。"""
    images, labels, skin, skin_prob = batch
    return images, labels, {"skin": skin, "soft_skin": skin_prob}


def unpack_sex_age_soft(batch: Batch) -> tuple[torch.Tensor, torch.Tensor, Attrs]:
    """HAM / PAPILA + SoftAttrDataset：loader 返回 (image, label, sex, age, age_prob)
    → attrs={sex, age(真值,分组用), soft_age([B,C] 条件输入)}。"""
    images, labels, sex, age, age_prob = batch
    return images, labels, {"sex": sex, "age": age, "soft_age": age_prob}


def forward_soft_skin(model: nn.Module, images: torch.Tensor, attrs: Attrs) -> torch.Tensor:
    """Fitzpatrick pred-attr HN：model(image, skin_prob)。"""
    return model(images, attrs["soft_skin"].to(images.device, non_blocking=True))


def forward_soft_age(model: nn.Module, images: torch.Tensor, attrs: Attrs) -> torch.Tensor:
    """HAM / PAPILA pred-attr HN：model(image, age_prob)。"""
    return model(images, attrs["soft_age"].to(images.device, non_blocking=True))


def unpack_sex_age_soft_pair(batch: Batch) -> tuple[torch.Tensor, torch.Tensor, Attrs]:
    """HAM 双属性 pred-attr 臂 + SoftAttrDataset：loader 返回
    (image, label, sex, age, sex_prob, age_prob)
    → attrs={sex, age（真值,分组用）, soft_sex, soft_age（条件输入）}。
    两份概率的顺序须与 SoftAttrDataset 构造时给的矩阵顺序一致（sex, age）。"""
    images, labels, sex, age, sex_prob, age_prob = batch
    return images, labels, {
        "sex": sex, "age": age, "soft_sex": sex_prob, "soft_age": age_prob,
    }


def forward_soft_sex_age(model: nn.Module, images: torch.Tensor, attrs: Attrs) -> torch.Tensor:
    """HAM 双属性 pred-attr HN：model(image, sex_prob, age_prob)——与 forward_soft_age 的单一
    age 通路构成「只差是否含 sex 概率」的单变量对照。"""
    d = images.device
    return model(
        images,
        attrs["soft_sex"].to(d, non_blocking=True),
        attrs["soft_age"].to(d, non_blocking=True),
    )


def unpack_sex_race_age_soft(batch: Batch) -> tuple[torch.Tensor, torch.Tensor, Attrs]:
    """MIMIC / CheXpert + SoftAttrDataset：loader 返回
    (image, label, sex, race, age, sex_prob, race_prob, age_prob)
    → attrs={sex, race, age（真值,分组用）, soft_sex, soft_race, soft_age（条件输入）}。
    三份概率的顺序须与 SoftAttrDataset 构造时给的矩阵顺序一致（sex, race, age）。"""
    images, labels, sex, race, age, sex_prob, race_prob, age_prob = batch
    return images, labels, {
        "sex": sex, "race": race, "age": age,
        "soft_sex": sex_prob, "soft_race": race_prob, "soft_age": age_prob,
    }


def forward_soft_patient(model: nn.Module, images: torch.Tensor, attrs: Attrs) -> torch.Tensor:
    """MIMIC / CheXpert pred-attr HN：model(image, sex_prob, race_prob, age_prob)。"""
    d = images.device
    return model(
        images,
        attrs["soft_sex"].to(d, non_blocking=True),
        attrs["soft_race"].to(d, non_blocking=True),
        attrs["soft_age"].to(d, non_blocking=True),
    )


# ============================================================
# 推理 / 评估
# ============================================================
def _group_attrs(attrs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """取 attrs 中的真值分组属性（白名单内的键），供 subgroup_auc / fairness 模块 splat。"""
    return {k: v for k, v in attrs.items() if k in GROUP_ATTR_KEYS}


def _extra_attrs(attrs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """取 attrs 中的非分组附加列（如实验 P 的 soft_* 条件输入），落盘时走 `extra` 通道。"""
    return {k: v for k, v in attrs.items() if k not in GROUP_ATTR_KEYS}


def _save_eval_predictions(path, er: "EvalResult") -> None:
    """把一次评估的预测落盘：分组属性进 attrs，其余（soft_* 条件输入）进 extra。"""
    save_predictions(path, er.labels, er.logits, _group_attrs(er.attrs),
                     extra=_extra_attrs(er.attrs) or None)


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
    on_after_backward: Callable[[nn.Module], None] | None = None,
    objective_fn: ObjectiveFn | None = None,
) -> tuple[float, float]:
    """
    跑一个训练 epoch，返回 (平均 loss, accuracy %)。逻辑与各脚本原实现一致，仅经回调泛化。

    on_after_backward：可选探针，在 `loss.backward()` 之后、**梯度裁剪之前**逐 batch 调用一次
    （故看到的是**未裁剪的真实梯度**，供 E1③ 累积 conv adapter 梯度范数）。默认 None = 无开销。

    objective_fn：可选训练目标替换（如 GroupDRO 的 robust loss）。给定时用
    `objective_fn(logits, labels, attrs)` 取代 `criterion(logits, labels)`——注意此时返回的
    「平均 loss」是该目标的值（GroupDRO 下为 Σ q_g L_g），与 val loss（恒为普通 BCE）**不同尺度**。
    """
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
        # objective_fn 给定时用它替换普通 BCE（GroupDRO 需要 attrs 取组序号）
        loss = (criterion(logits, label_col) if objective_fn is None
                else objective_fn(logits, label_col, attrs))
        loss.backward()
        if on_after_backward is not None:                 # 裁剪前捕获真实梯度（探针）
            on_after_backward(model)
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
    # 只把白名单内的**真值分组属性**splat 进子群 AUC；条件输入用的 soft_* 概率矩阵在此被排除
    vec = subgroup_auc_vector(dataset, labels_np, logits_np, **_group_attrs(attrs_np))
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
    method: str,
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
    val/test 评估。SWAD 是**派生**产物（同 config/seed，单一平均模型），派生自 ERM 时即协议 §1
    的 SWAD 基线，派生自 HN 时即实验 G 的融合臂。

    命名：checkpoint / 预测 / val 日志均以 run_id(swad_method_name(method), config_tag, seed) 为
    基名——ERM 派生仍为 `"swad"`（向后兼容既有产物），HN 派生为 `"<method>_swad"`（见
    `swad_method_name` 的撞车说明）。写一条 val 记录（epoch = 平均区间末 epoch 的 1-based 值）
    承载平均模型的验证集完整子群向量。

    Args:
        method: 派生来源的方法名（用于推导 SWAD 变体名；不是 SWAD 自身的 method 名）。

    Returns:
        {swad_method, swad_ckpt, swad_val_log, swad_pred, swad_interval:(t_s,t_e,n_averaged),
         swad_val_overall_auc, swad_val_wc_auc}。
    """
    swad_method = swad_method_name(method)
    print(f"\n{'#' * 70}\n# SWAD 权重平均（派生自 {method}，method={swad_method}）\n{'#' * 70}")
    averaged_sd, interval = swad_average_over_interval(
        val_losses, state_dicts,
        n_converge=n_converge, n_tolerance=n_tolerance, tolerance_ratio=tolerance_ratio,
    )
    # 区间 0-based → 报告用 1-based epoch（与训练打印一致）
    print(f"loss valley 区间: epoch {interval.t_s + 1}..{interval.t_e + 1} "
          f"(共 {interval.n_averaged} 个，L_min={interval.l_min:.4f}, thr={interval.threshold:.4f})")

    swad_rid = run_id(swad_method, config_tag, seed)
    swad_ckpt = output_dir / f"{swad_rid}_averaged.pth"
    torch.save(averaged_sd, swad_ckpt)
    print(f"已保存 SWAD 平均权重: {swad_ckpt.name}")

    # 把平均权重载入模型副本评估（不污染主 model 的 device 状态）
    model.load_state_dict(averaged_sd)
    # --- 验证集：写一条 val 记录（承载平均模型的子群向量）---
    er_val = evaluate(model, val_loader, criterion, dataset, unpack_fn, forward_fn)
    swad_log = default_val_log_path(output_dir, swad_method, config_tag, seed)
    if swad_log.exists():
        swad_log.unlink()
    rec = record_from_vector(
        er_val.vector, dataset=dataset, method=swad_method, config_tag=config_tag,
        lr=cfg["training"].get("learning_rate", 0.0), wd=cfg["training"].get("weight_decay", 0.0),
        seed=seed, epoch=interval.t_e + 1, overall_auc=er_val.overall_auc,
    )
    append_val_record(swad_log, rec)
    wc_str = f"{er_val.wc_auc:.4f}" if er_val.wc_auc is not None else "N/A"
    print(f"SWAD val: AUC={er_val.overall_auc:.4f} worst-case AUC={wc_str}")

    # --- 测试集：完整指标 + 分组公平性报告 + 落盘逐样本预测（method=swad_method）---
    er_test = evaluate(model, test_loader, criterion, dataset, unpack_fn, forward_fn)
    wc_test = f"{er_test.wc_auc:.4f}" if er_test.wc_auc is not None else "N/A"
    print(f"\n{'#' * 70}\n# Evaluating SWAD averaged model (test)\n{'#' * 70}")
    print(f"Test loss={er_test.loss:.4f}  accuracy={er_test.acc:.2f}%  AUC={er_test.overall_auc:.4f}  "
          f"Worst-case AUC={wc_test}")
    fairness_report_fn(er_test.labels, er_test.logits, er_test.attrs)
    # SWAD 是单一平均模型（无 overall/worstcase 之分）：预测存到 "overall" 槽（唯一 selection）
    swad_pred = default_prediction_path(output_dir, swad_method, config_tag, seed, "overall")
    _save_eval_predictions(swad_pred, er_test)
    print(f"已落盘 SWAD test 预测: {swad_pred.name}")

    return {
        "swad_method": swad_method,
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
    on_after_backward: Callable[[nn.Module], None] | None = None,
    epoch_diag_fn: Callable[[int, "EvalResult"], None] | None = None,
    objective: ObjectiveFn | None = None,
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
            训练后做 loss-valley 平均并在 val/test 评估。派生模型的 method 名由
            `swad_method_name(method)` 决定：ERM → "swad"（协议 §1 基线），HN → "<method>_swad"
            （**实验 G** 的 HN×SWAD 融合臂，见 docs/hyperadapt_swad_fusion_plan.md）。
            默认 False（搜索阶段不开销）。
        swad_n_converge / swad_n_tolerance / swad_tolerance_ratio: SWAD 超参（见 harness/swad.py）。
        objective: 可选的**训练目标**替换（默认 None = 普通 BCE 均值）。用于 **GroupDRO**
            （`harness.groupdro.GroupDROObjective`）这类需要子群标签的目标：训练时用
            `objective(logits, labels, attrs)` 代替 criterion，而 **val/test 评估、早停、SWAD
            loss 谷仍一律用普通 BCE**——保证模型选择口径与其它方法完全一致（协议决策①）。
            若对象提供 `reset_epoch_stats()` / `epoch_summary()` / `describe()`，harness 会在
            每个 epoch 前后调用以打印诊断（如各组损失与对抗权重 q），无则静默跳过。

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
    # 冻结 regime：build_model 可返回带 requires_grad=False 的 backbone（如 freeze_backbone=True）。
    # 仅把可训练参数交给优化器——冻结参数本就 grad=None 不更新，这里显式过滤以免 AdamW 解耦
    # weight decay 误伤，并让日志中的可训练参数量清晰（若无冻结则与 model.parameters() 等价）。
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    if n_trainable != n_params:
        print(f"        FROZEN regime: trainable {n_trainable:,} / frozen {n_params - n_trainable:,} "
              f"(仅可训练参数进优化器)")
    print(f"Optim : AdamW(lr={hparam.lr:.0e}, weight_decay={hparam.wd:.0e})  "
          f"[来自超参网格 config={hparam.tag}]")
    print(f"Misc  : grad_clip_norm={train_cfg['grad_clip_norm']}, batch_size={train_cfg['batch_size']}")
    print(f"Early stopping: monitor={es_cfg['monitor']} (mode={es_cfg['mode']}), "
          f"patience={es_cfg['patience']}, min_delta={es_cfg['min_delta']}")
    print(f"Val log: {log_path}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(trainable_params, lr=hparam.lr, weight_decay=hparam.wd)

    # 训练目标替换（GroupDRO 等）：只影响训练 loss，val/test 仍用上面的 criterion
    if objective is not None:
        print(f"Objective: 训练目标已替换为 {type(objective).__name__}"
              f"（val/test 评估与早停仍用普通 BCE）")
        if hasattr(objective, "describe"):
            print(objective.describe())

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
        print(f"ℹ️  swad=True 且 method={method!r}（非 erm）：这是**实验 G**（HN×SWAD 融合）的路径，"
              f"派生模型将记为 method={swad_method_name(method)!r}，与协议 §1 的 ERM-SWAD 基线"
              f"（method='swad'）分开存放，互不覆盖。")
    swad_val_losses: list[float] = []
    swad_state_dicts: list[dict[str, torch.Tensor]] = []

    print(f"\nTraining for up to {max_epochs} epochs (early stopping enabled)...\n" + "=" * 70)
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        if objective is not None and hasattr(objective, "reset_epoch_stats"):
            objective.reset_epoch_stats()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer,
            train_cfg["grad_clip_norm"], epoch, unpack_fn, forward_fn,
            on_after_backward=on_after_backward, objective_fn=objective,
        )
        er = evaluate(model, val_loader, criterion, dataset, unpack_fn, forward_fn)
        dt = time.time() - t0
        if objective is not None and hasattr(objective, "epoch_summary"):
            print(f"  {objective.epoch_summary()}")

        # 逐 epoch 写完整子群 AUC 向量到 val 日志（复用 evaluate 已算的 vector，不二次计算）
        rec = record_from_vector(
            er.vector, dataset=dataset, method=method, config_tag=hparam.tag,
            lr=hparam.lr, wd=hparam.wd, seed=seed, epoch=epoch, overall_auc=er.overall_auc,
        )
        append_val_record(log_path, rec)

        # 可选逐 epoch 诊断探针（如 E1③ 的 conv ρ / 梯度范数 / 权重范数）：调用方自行落盘，
        # harness 保持通用（不改 val 日志 schema）。放在 checkpoint/早停判定之前，确保每 epoch 都记。
        if epoch_diag_fn is not None:
            epoch_diag_fn(epoch, er)

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
    _save_eval_predictions(pred_overall, er_overall)
    _save_eval_predictions(pred_worstcase, er_worstcase)
    print(f"已落盘 test 预测: {pred_overall.name} / {pred_worstcase.name}")

    # ROC 后处理（A4 / C*.8）需 val 预测定 deprived 子群与选 margin θ：用 overall-selection
    # checkpoint 在 val 集落盘一份 val 预测（selection="val_overall"），使 ROC 派生保持数据集无关。
    er_val_overall = evaluate_checkpoint(model, ckpt_overall, val_loader, criterion, dataset,
                                         unpack_fn, forward_fn, fairness_report_fn,
                                         label="[val] Overall AUC selection")
    pred_val_overall = val_prediction_path(output_dir, method, hparam.tag, seed)
    _save_eval_predictions(pred_val_overall, er_val_overall)
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
            model=model, method=method, val_losses=swad_val_losses, state_dicts=swad_state_dicts,
            output_dir=output_dir, config_tag=hparam.tag, seed=seed, dataset=dataset, cfg=cfg,
            criterion=criterion, val_loader=val_loader, test_loader=test_loader,
            unpack_fn=unpack_fn, forward_fn=forward_fn, fairness_report_fn=fairness_report_fn,
            n_converge=swad_n_converge, n_tolerance=swad_n_tolerance,
            tolerance_ratio=swad_tolerance_ratio,
        ))

    return summary


# ============================================================
# self-test：SWAD 派生的命名隔离与落盘链路（合成数据、秒级，无需 GPU/真实数据集）
# ============================================================
def _make_synthetic_loader(
    n: int, batch_size: int, *, shuffle: bool, seed: int, noise: float, img: int,
) -> DataLoader:
    """
    造一批 schema 对齐 HAM10000 loader 的合成样本：(image, label, sex, age_group)。

    标签由图像信号 + 噪声决定，保证各子群 AUC 有定义（否则 subgroup_auc 返回 NaN、worst-case 为空）。

    Args:
        n         : 样本数。
        batch_size: 批大小。
        shuffle   : 是否打乱（train=True）。
        seed      : 该 split 的随机种子（train/val/test 须互不相同）。
        noise     : 标签噪声强度（越大越易过拟合 ⇒ val loss 先降后升，形成 loss 谷）。
        img       : 图像边长。

    Returns:
        DataLoader，逐 batch 产出 4 元组。
    """
    from torch.utils.data import TensorDataset

    g = np.random.default_rng(seed)
    sex = g.integers(0, 2, size=n)
    age = g.integers(0, 4, size=n)                     # HAM 的 4 个有效年龄组
    signal = g.normal(size=n)
    label = (signal + noise * g.normal(size=n) > 0).astype(np.int64)
    images = (signal[:, None, None, None] * np.ones((1, 1, img, img))
              + 0.5 * g.normal(size=(n, 1, img, img))).astype(np.float32)
    ds = TensorDataset(torch.from_numpy(images), torch.from_numpy(label),
                       torch.from_numpy(sex), torch.from_numpy(age))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


class _SyntheticNet(nn.Module):
    """自测用极小模型：图像头 + 可选 age 条件偏置（复刻 HN 的 `forward(image, age)` 签名）。"""

    def __init__(self, conditioned: bool, img: int, hidden: int = 0) -> None:
        """
        Args:
            conditioned: True = 吃 age（模拟 HN）；False = image-only（模拟 ERM）。
            img        : 图像边长。
            hidden     : >0 时改用该宽度的 3 层 MLP（高容量，用于制造过拟合与 loss 谷）。
        """
        super().__init__()
        self.fc = (nn.Linear(img * img, 1) if hidden == 0 else
                   nn.Sequential(nn.Linear(img * img, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)))
        self.age_bias = nn.Embedding(4, 1) if conditioned else None
        if self.age_bias is not None:
            nn.init.zeros_(self.age_bias.weight)

    def forward(self, image: torch.Tensor, age: torch.Tensor | None = None) -> torch.Tensor:
        """Args: image [B,1,H,W]；age [B]（conditioned 时必给）。Returns: [B,1] logits。"""
        out = self.fc(image.flatten(1))
        if self.age_bias is not None and age is not None:
            out = out + self.age_bias(age.long())
        return out


def _selftest() -> None:
    """
    验证 `run_training(swad=True)` 的派生命名与落盘链路（实验 G 的关键回归测试）。

    断言三件事：
      1. HN（method="hyperadapt"）的派生产物全部落在 `hyperadapt_swad_*` 命名空间；
      2. **不产生任何 `swad_*` 文件**，且与 ERM 派生的 SWAD 在**同 config_tag 同 seed** 下互不覆盖
         —— 这正是 HAM cv5 的真实撞车场景（HyperAdapt 选定 lr3e-05_wd1e-04，而 ERM 搜索带 SWAD
         跑满 6 配置、`swad_lr3e-05_wd1e-04` 已存在）；
      3. ERM 派生仍为 `swad_*`（向后兼容未破坏），且 val 日志内 `method` 字段与文件名一致
         （下游 A1/A2 按该字段筛选，仅文件名对不够）。

    另含一个**过拟合用例**：前两个用例 val loss 单调下降 ⇒ 谷底=末 epoch、区间退化为单点，
    「平均多个 epoch 快照」这条路径不会被走到；故用小样本 + 高容量 + 强标签噪声制造真实 loss 谷，
    断言 n_averaged>1 且平均权重 ≠ 端点权重。
    """
    import json
    import shutil
    import tempfile

    from src.training.harness.hparam_grid import get_hparam_config

    img, seed = 8, 42
    out = Path(tempfile.mkdtemp(prefix="train_loop_selftest_"))

    def run(method: str, conditioned: bool, *, config_index: int = 0, num_epochs: int = 5,
            n_train: int = 256, hidden: int = 0, noise: float = 0.3,
            out_dir: Path | None = None, swad: bool = True,
            objective: "ObjectiveFn | None" = None) -> dict:
        """跑一次训练（默认 swad=True）；config_index=0 → lr3e-05_wd1e-04（真实撞车 tag）。"""
        cfg = {
            "training": {"batch_size": 64, "num_epochs": num_epochs, "grad_clip_norm": 1.0,
                         "learning_rate": 3e-5, "weight_decay": 1e-4},
            # patience=num_epochs：关早停，确保跑满、谷后回升能被观察到
            "early_stopping": {"monitor": "val_auc", "mode": "max",
                               "patience": num_epochs, "min_delta": 0.0},
        }
        seed_everything(seed)                          # 规程：建 loader 前播种
        mk = lambda n, sh, sd: _make_synthetic_loader(  # noqa: E731
            n, 64, shuffle=sh, seed=sd, noise=noise, img=img)
        return run_training(
            dataset="ham10000", method=method, output_dir=(out_dir or out), cfg=cfg,
            hparam=get_hparam_config(config_index), seed=seed,
            train_loader=mk(n_train, True, 1), val_loader=mk(128, False, 2),
            test_loader=mk(128, False, 3),
            build_model=lambda: _SyntheticNet(conditioned, img, hidden),
            unpack_fn=unpack_sex_age,
            forward_fn=forward_age if conditioned else forward_image_only,
            fairness_report_fn=lambda y, s, a: None,   # 公平性报告非本测试目标
            swad=swad, objective=objective,
        )

    try:
        tag = get_hparam_config(0).tag
        s_hn = run("hyperadapt", True)
        s_erm = run("erm", False)                      # 同 tag 同 seed = 撞车场景

        assert s_hn["swad_method"] == "hyperadapt_swad", s_hn["swad_method"]
        assert s_erm["swad_method"] == "swad", s_erm["swad_method"]
        for rel in (f"hyperadapt_swad_{tag}_seed{seed}_averaged.pth",
                    f"val_logs/hyperadapt_swad_{tag}_seed{seed}.jsonl",
                    f"predictions/hyperadapt_swad_{tag}_seed{seed}_overall.npz"):
            assert (out / rel).exists(), f"缺 HN 派生产物：{rel}"
        for rel in (f"swad_{tag}_seed{seed}_averaged.pth",
                    f"val_logs/swad_{tag}_seed{seed}.jsonl",
                    f"predictions/swad_{tag}_seed{seed}_overall.npz"):
            assert (out / rel).exists(), f"缺 ERM-SWAD 产物：{rel}"
        names = [p.name for p in out.rglob("*") if p.is_file()]
        assert len([n for n in names if n.startswith("hyperadapt_swad_")]) == 3
        assert len([n for n in names if n.startswith(f"swad_{tag}")]) == 3   # 未被 HN 覆盖
        rec = json.loads((out / f"val_logs/hyperadapt_swad_{tag}_seed{seed}.jsonl")
                         .read_text().strip())
        assert rec["method"] == "hyperadapt_swad", rec["method"]

        # 过拟合用例：把谷区间撑开到 >1，覆盖多-epoch 平均路径
        out2 = out / "valley"
        s_of = run("hyperadapt", True, config_index=5, num_epochs=25,
                   n_train=64, hidden=256, noise=1.2, out_dir=out2)
        _, _, n_avg = s_of["swad_interval"]
        assert n_avg > 1, f"未制造出宽度>1 的谷区间（n_averaged={n_avg}），多-epoch 平均未覆盖"
        tag5 = get_hparam_config(5).tag
        avg_sd = torch.load(out2 / f"hyperadapt_swad_{tag5}_seed{seed}_averaged.pth",
                            map_location="cpu", weights_only=True)
        best_sd = torch.load(out2 / f"hyperadapt_{tag5}_seed{seed}_best_overall.pth",
                             map_location="cpu", weights_only=True)
        assert any(not torch.allclose(avg_sd[k], best_sd[k])
                   for k in avg_sd if torch.is_floating_point(avg_sd[k])), \
            "平均权重与 best_overall 逐张量相同 ⇒ 平均未生效"

        # GroupDRO 目标接线：训练走 robust loss、评估仍普通 BCE，产物命名与 ERM 隔离，q 被真实更新
        from src.training.harness.groupdro import GroupDROObjective
        out3 = out / "groupdro"
        gdro = GroupDROObjective("ham10000", step_size=1.0)   # η 放大以在 5 epoch 内看出偏移
        s_gd = run("groupdro", False, out_dir=out3, swad=False, objective=gdro)
        tag0 = get_hparam_config(0).tag
        assert (out3 / f"groupdro_{tag0}_seed{seed}_best_overall.pth").exists()
        assert (out3 / f"predictions/groupdro_{tag0}_seed{seed}_overall.npz").exists()
        assert "swad_method" not in s_gd, "GroupDRO 不派生 SWAD"
        q = gdro.q.detach().cpu().numpy()
        assert abs(q.max() - 1.0 / gdro.n_groups) > 1e-4, f"q 未被更新（仍均匀）：{q}"
        assert abs(q.sum() - 1.0) < 1e-6, q.sum()

        print(f"\ntrain_loop self-test 全部通过 ✓（HN→hyperadapt_swad_* / ERM→swad_*，"
              f"同 tag 同 seed 无覆盖，val 日志 method 字段一致，"
              f"多-epoch 谷平均生效 n_averaged={n_avg}，GroupDRO 目标接线 q 已更新）")
    finally:
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    _selftest()
