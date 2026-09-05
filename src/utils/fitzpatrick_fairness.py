"""
Fitzpatrick17k malignant 分组公平性报告（MEDFAIR 对齐）
======================================================
复用 drafts/fairness_metrics.py 中与数据集无关的 auc_by_group，但敏感属性是
单一的 6 类肤色（Fitzpatrick I–VI），而非 MIMIC 的 sex/race/age 三个二值轴。

MEDFAIR 对 Fitzpatrick17k 的公平性口径（论文 Table A9 / A12）：
  - 6 个肤色子群（I–VI）的分组 AUC；
  - Worst-case AUC = 6 组最小（max-min / minimax 公平性）；
  - AUC Gap        = 6 组 max - min（group 公平性）；
  - **不报 Equalized Odds**：MEDFAIR 仅对二值敏感属性报 EqOdd，
    Fitzpatrick 的 6 类肤色不报（与论文 Table A12 中 Fitzpatrick 行无 EqOdd 一致）。

属性编码：
  skin : 0–5 对应 Fitzpatrick 肤色 I–VI
  label: 0=benign/non-neoplastic, 1=malignant
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from src.utils.fairness_metrics import auc_by_group

FITZ_SKIN_NAMES = {0: "I", 1: "II", 2: "III", 3: "IV", 4: "V", 5: "VI"}


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """单一子群内的 AUC；若该子群只含一种类别则返回 None。"""
    if len(np.unique(y_true)) < 2:
        return None
    return roc_auc_score(y_true, y_score)


def worst_case_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    skin: np.ndarray,
) -> float | None:
    """
    计算 worst-case (minimax) AUC：6 个肤色子群中 AUC 最低者。

    用于训练循环中按 epoch 监控，对比 "Overall AUC 选择" 与
    "Worst-case AUC 选择" 两种 model selection 策略对公平性的影响。

    Returns:
        所有有效子群（同时含正负样本）AUC 的最小值；若无有效子群返回 None。
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    skin = np.asarray(skin).astype(int)

    aucs: list[float] = []
    for auc, _n in auc_by_group(y_true, y_score, skin, FITZ_SKIN_NAMES).values():
        if auc is not None:
            aucs.append(auc)
    return min(aucs) if aucs else None


def auc_gap(
    y_true: np.ndarray,
    y_score: np.ndarray,
    skin: np.ndarray,
) -> float | None:
    """
    计算 AUC Gap = 6 个肤色子群 AUC 的 max - min（group 公平性指标）。

    Returns:
        有效子群 AUC 的极差；有效子群少于 2 个时返回 None。
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    skin = np.asarray(skin).astype(int)

    aucs = [
        auc for auc, _n in auc_by_group(y_true, y_score, skin, FITZ_SKIN_NAMES).values()
        if auc is not None
    ]
    return (max(aucs) - min(aucs)) if len(aucs) >= 2 else None


def print_fitzpatrick_fairness_report(
    y_true: np.ndarray,
    y_score: np.ndarray,
    skin: np.ndarray,
    threshold: float = 0.0,
) -> None:
    """
    打印 Fitzpatrick17k malignant 模型的分组公平性报告
    （Overall AUC、6 个肤色子群 AUC、Worst-case AUC、AUC Gap）。

    Args:
        y_true   : [N] 0/1，malignant 真实标签（1=恶性）
        y_score  : [N] 模型输出（logits 或概率均可；AUC 对单调变换不敏感）
        skin     : [N] 0–5，Fitzpatrick 肤色 I–VI
        threshold: 判正类阈值（logits 用 0，probabilities 用 0.5）
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    skin = np.asarray(skin).astype(int)
    y_pred = (y_score > threshold).astype(int)

    print("\n" + "=" * 60)
    print("Fitzpatrick17k malignant — Fairness Report")
    print("=" * 60)

    overall_auc = _safe_auc(y_true, y_score)
    print(f"\n[Overall]")
    print(f"  N        = {len(y_true)}")
    print(f"  AUC      = {overall_auc:.4f}" if overall_auc is not None else "  AUC      = N/A")
    print(f"  Accuracy = {(y_pred == y_true).mean():.4f}")

    print(f"\n[AUC by Skin type (Fitzpatrick I–VI)]")
    for name, (auc, n) in auc_by_group(y_true, y_score, skin, FITZ_SKIN_NAMES).items():
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"  型{name:<4s} n={n:6d}   AUC={auc_str}")

    wc = worst_case_auc(y_true, y_score, skin)
    gap = auc_gap(y_true, y_score, skin)
    print(f"\n[Worst-case AUC]  min over 6 skin groups = "
          f"{wc:.4f}" if wc is not None else "\n[Worst-case AUC]  N/A")
    print(f"[AUC Gap]         max - min over 6 skin groups = "
          f"{gap:.4f}" if gap is not None else "[AUC Gap]         N/A")

    print("=" * 60)


if __name__ == "__main__":
    # 生成假数据自检
    rng = np.random.default_rng(0)
    N = 2000
    y_true = rng.integers(0, 2, N)
    y_score = rng.standard_normal(N) + 0.5 * y_true
    skin = rng.integers(0, 6, N)
    print_fitzpatrick_fairness_report(y_true, y_score, skin, threshold=0.0)
