"""
HAM10000 malignant 分组公平性报告（MEDFAIR 对齐）
=================================================
复用 drafts/fairness_metrics.py 中与数据集无关的 auc_by_group / eqodd_by_group。
HAM10000 的敏感属性是 sex（二值）与 age_group（4 有效组），无肤色/种族——
介于 Fitzpatrick（单一肤色轴）与 MIMIC（sex/race/age 三轴）之间。

MEDFAIR 对 HAM10000 的公平性口径（论文 Table A9 / A11 / A17）：
  - Sex（2 组）、Age（4 组）、Sex×Age（8 交叉子群）各自的分组 AUC；
  - Worst-case AUC = 上述所有子群最小（max-min / minimax）；
  - AUC Gap        = 各轴内 max - min（group 公平性）；
  - Equalized Odds 仅对二值属性 Sex 报（与 MEDFAIR 仅对二值属性报 EqOdd 一致；
    age_group 为多类，不报 EqOdd，与 Fitzpatrick 多类肤色不报 EqOdd 同理）。

属性编码：
  sex       : 0=Male, 1=Female
  age_group : 0=20-40, 1=40-60, 2=60-80, 3=80+；**-1=0-20（排除组）**
  label     : 0=benign, 1=malignant

⚠️ age_group==-1（0-20，约 2.4%）在所有「按年龄」与「Sex×Age 交叉」的统计中被排除，
   对齐 MEDFAIR Table A16「excluding age 0-20 as it has too few samples」；
   但 Overall 与「按 Sex」统计仍使用全部样本（sex 任务用全量 9948）。
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from src.utils.fairness_metrics import auc_by_group, eqodd_by_group

HAM_SEX_NAMES = {0: "Male", 1: "Female"}
HAM_AGE_NAMES = {0: "20-40", 1: "40-60", 2: "60-80", 3: "80+"}  # 不含 -1（排除组）


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """单一子群内的 AUC；若该子群只含一种类别则返回 None。"""
    if len(np.unique(y_true)) < 2:
        return None
    return roc_auc_score(y_true, y_score)


def _age_valid_mask(age_group: np.ndarray) -> np.ndarray:
    """返回 age_group>=0 的布尔掩码（排除 -1=0-20 组，对齐 MEDFAIR）。"""
    return np.asarray(age_group).astype(int) >= 0


def worst_case_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sex: np.ndarray,
    age_group: np.ndarray,
) -> float | None:
    """
    计算 worst-case (minimax) AUC：在 Sex / Age 各自分组以及 Sex×Age 交叉分组中，
    取 AUC 最低的子群作为模型选择判据。age_group==-1（0-20）被排除。

    用于训练循环中按 epoch 监控，对比 "Overall AUC 选择" 与 "Worst-case AUC 选择"
    两种 model selection 策略对公平性的影响。

    Returns:
        所有有效子群（同时含正负样本）AUC 的最小值；若无有效子群返回 None。
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    sex = np.asarray(sex).astype(int)
    age_group = np.asarray(age_group).astype(int)

    aucs: list[float] = []

    # Sex：使用全部样本
    for auc, _n in auc_by_group(y_true, y_score, sex, HAM_SEX_NAMES).values():
        if auc is not None:
            aucs.append(auc)

    # Age：排除 -1
    av = _age_valid_mask(age_group)
    for auc, _n in auc_by_group(
        y_true[av], y_score[av], age_group[av], HAM_AGE_NAMES
    ).values():
        if auc is not None:
            aucs.append(auc)

    # Sex×Age 交叉（8 子群，排除 -1）
    for s in np.unique(sex):
        for a in sorted(HAM_AGE_NAMES):
            mask = (sex == s) & (age_group == a)
            if mask.sum() == 0:
                continue
            auc = _safe_auc(y_true[mask], y_score[mask])
            if auc is not None:
                aucs.append(auc)

    return min(aucs) if aucs else None


def auc_gap(
    y_true: np.ndarray,
    y_score: np.ndarray,
    group_ids: np.ndarray,
    group_names: dict[int, str],
) -> float | None:
    """计算某一分组轴内 AUC 的 max - min（group 公平性）；有效子群<2 时返回 None。"""
    aucs = [
        auc for auc, _n in auc_by_group(
            np.asarray(y_true).astype(int),
            np.asarray(y_score).astype(float),
            np.asarray(group_ids).astype(int),
            group_names,
        ).values()
        if auc is not None
    ]
    return (max(aucs) - min(aucs)) if len(aucs) >= 2 else None


def print_ham10000_fairness_report(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sex: np.ndarray,
    age_group: np.ndarray,
    threshold: float = 0.0,
) -> None:
    """
    打印 HAM10000 malignant 模型的分组公平性报告（Overall、Sex / Age / Sex×Age 分组 AUC、
    各轴 AUC Gap、Sex 的 Equalized Odds、Worst-case AUC）。

    Args:
        y_true   : [N] 0/1，malignant 真实标签（1=恶性）
        y_score  : [N] 模型输出（logits 或概率均可；AUC 对单调变换不敏感）
        sex      : [N] 0=Male, 1=Female
        age_group: [N] 0=20-40/1=40-60/2=60-80/3=80+；-1=0-20（排除组）
        threshold: 判正类阈值（logits 用 0，probabilities 用 0.5）
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    sex = np.asarray(sex).astype(int)
    age_group = np.asarray(age_group).astype(int)
    y_pred = (y_score > threshold).astype(int)
    av = _age_valid_mask(age_group)

    print("\n" + "=" * 60)
    print("HAM10000 malignant — Fairness Report")
    print("=" * 60)

    overall_auc = _safe_auc(y_true, y_score)
    print(f"\n[Overall]")
    print(f"  N        = {len(y_true)}")
    print(f"  AUC      = {overall_auc:.4f}" if overall_auc is not None else "  AUC      = N/A")
    print(f"  Accuracy = {(y_pred == y_true).mean():.4f}")

    print(f"\n[AUC by Sex]")
    for name, (auc, n) in auc_by_group(y_true, y_score, sex, HAM_SEX_NAMES).items():
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"  {name:<8s} n={n:6d}   AUC={auc_str}")
    gap = auc_gap(y_true, y_score, sex, HAM_SEX_NAMES)
    print(f"  AUC Gap (max-min) = {gap:.4f}" if gap is not None else "  AUC Gap = N/A")

    print(f"\n[AUC by Age group]  (excl. 0-20)")
    for name, (auc, n) in auc_by_group(
        y_true[av], y_score[av], age_group[av], HAM_AGE_NAMES
    ).items():
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"  {name:<8s} n={n:6d}   AUC={auc_str}")
    gap = auc_gap(y_true[av], y_score[av], age_group[av], HAM_AGE_NAMES)
    print(f"  AUC Gap (max-min) = {gap:.4f}" if gap is not None else "  AUC Gap = N/A")

    print(f"\n[AUC by Sex x Age group]  (8 subgroups, excl. 0-20)")
    inter_aucs: list[float] = []
    for s in np.unique(sex):
        for a in sorted(HAM_AGE_NAMES):
            mask = (sex == s) & (age_group == a)
            if mask.sum() == 0:
                continue
            name = f"{HAM_SEX_NAMES[int(s)]} & {HAM_AGE_NAMES[int(a)]}"
            auc = _safe_auc(y_true[mask], y_score[mask])
            auc_str = f"{auc:.4f}" if auc is not None else "N/A"
            print(f"  {name:<18s} n={int(mask.sum()):6d}   AUC={auc_str}")
            if auc is not None:
                inter_aucs.append(auc)
    if len(inter_aucs) >= 2:
        print(f"  AUC Gap (max-min) = {max(inter_aucs) - min(inter_aucs):.4f}")

    print(f"\n[Equalized Odds by Sex]")
    per_group, gap = eqodd_by_group(y_true, y_pred, sex, HAM_SEX_NAMES)
    for name, stats in per_group.items():
        tpr = f"{stats['tpr']:.4f}" if stats["tpr"] is not None else "N/A"
        fpr = f"{stats['fpr']:.4f}" if stats["fpr"] is not None else "N/A"
        print(f"  {name:<8s} n={stats['n']:6d}   TPR={tpr}   FPR={fpr}")
    print(f"  EqOdd gap = {gap:.4f}" if gap is not None else "  EqOdd gap = N/A")

    wc_auc = worst_case_auc(y_true, y_score, sex, age_group)
    print(f"\n[Worst-case AUC]")
    print(f"  min over Sex / Age / Sex x Age groups = "
          f"{wc_auc:.4f}" if wc_auc is not None else "  N/A")

    print("=" * 60)


if __name__ == "__main__":
    # 生成假数据自检（含 -1 排除组）
    rng = np.random.default_rng(0)
    N = 2000
    y_true = rng.integers(0, 2, N)
    y_score = rng.standard_normal(N) + 0.5 * y_true
    sex = rng.integers(0, 2, N)
    age_group = rng.choice([-1, 0, 1, 2, 3], size=N, p=[0.024, 0.16, 0.45, 0.29, 0.076])
    print_ham10000_fairness_report(y_true, y_score, sex, age_group, threshold=0.0)
