"""
合成信号实验的分组公平性报告（评审补强 R1）
===========================================
R1 剂量-反应实验的敏感属性是**注入的侧信道 A_syn**（见 src/datasets/synthetic_attribute.py），
故公平性子群 = A_syn 的两组（Asyn=0 / Asyn=1），与 Fitzpatrick 单一肤色轴同构（单轴、无交叉），
只是组数为 2。口径与其它数据集的 fairness 模块一致：Overall AUC、各子群 AUC、Worst-case
(minimax) AUC、AUC gap（max−min）。

**为什么子群是 A_syn**：R1 把 A_syn 当作「信号可控的敏感属性」，检验 HN 用上 A_syn 后能否
在给定信号强度下超过 attribute-blind 的 ERM。这与真实数据集（sex/race/age/skin 的 I≈0、HN 打平）
构成对照——此处 I(Y;A_syn|X)>0 且可调，看增益是否随信号出现（R1 的充分性因果检验）。
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from src.datasets.synthetic_attribute import SYNTH_ATTR_NAMES


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """单一子群内的 AUC；子群为空或只含一种类别时返回 None（与各 fairness 模块口径一致）。"""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def auc_by_asyn(
    y_true: np.ndarray, y_score: np.ndarray, a_syn: np.ndarray,
) -> "dict[int, float | None]":
    """返回每个 A_syn 组（0/1）的 AUC（子群单一类别时为 None）。"""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    a_syn = np.asarray(a_syn).astype(int)
    return {g: _safe_auc(y_true[a_syn == g], y_score[a_syn == g]) for g in SYNTH_ATTR_NAMES}


def worst_case_auc(
    y_true: np.ndarray, y_score: np.ndarray, a_syn: np.ndarray,
) -> float | None:
    """
    worst-case (minimax) AUC = A_syn 两组中最低的子群 AUC，用于 worst-case model selection。

    Args:
        y_true : [N] 0/1 标签。
        y_score: [N] 模型输出（logit/概率）。
        a_syn  : [N] 合成属性（0/1）。

    Returns:
        最小非 None 子群 AUC；无有效子群时 None。
    """
    aucs = [a for a in auc_by_asyn(y_true, y_score, a_syn).values() if a is not None]
    return min(aucs) if aucs else None


def auc_gap(y_true: np.ndarray, y_score: np.ndarray, a_syn: np.ndarray) -> float | None:
    """A_syn 两组 AUC 的 max−min（group 公平性 gap）；有效子群不足 2 时 None。"""
    aucs = [a for a in auc_by_asyn(y_true, y_score, a_syn).values() if a is not None]
    return (max(aucs) - min(aucs)) if len(aucs) >= 2 else None


def print_synthetic_fairness_report(
    y_true: np.ndarray, y_score: np.ndarray, a_syn: np.ndarray, threshold: float = 0.0,
) -> None:
    """打印 Overall / 各 A_syn 子群 / Worst-case / gap 的公平性小结（threshold 仅为签名对齐，AUC 不用）。"""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    overall = _safe_auc(y_true, y_score)
    per = auc_by_asyn(y_true, y_score, a_syn)
    wc = worst_case_auc(y_true, y_score, a_syn)
    gap = auc_gap(y_true, y_score, a_syn)
    print("\n--- Synthetic-signal fairness report (subgroup = A_syn) ---")
    print(f"Overall AUC : {overall:.4f}" if overall is not None else "Overall AUC : N/A")
    for g, name in SYNTH_ATTR_NAMES.items():
        n = int(np.sum(np.asarray(a_syn).astype(int) == g))
        auc = per[g]
        print(f"  {name:10s} (n={n:6d}) AUC = " + (f"{auc:.4f}" if auc is not None else "N/A"))
    print(f"Worst-case AUC: {wc:.4f}" if wc is not None else "Worst-case AUC: N/A")
    print(f"AUC gap       : {gap:.4f}" if gap is not None else "AUC gap       : N/A")
