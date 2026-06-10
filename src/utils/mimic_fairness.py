"""
MIMIC-CXR No Finding 分组公平性报告
====================================
复用 drafts/fairness_metrics.py 中与数据集无关的分组统计逻辑
(auc_by_group / eqodd_by_group)，但不直接调用其 print_fairness_report——
该函数的 RACE_NAMES 按 UTKFace 编码，与 MIMIC-CXR 的 race 编码不同。

MIMIC-CXR（MEDFAIR 对齐）属性编码：
  sex : 0=Male, 1=Female
  race: 0=White, 1=Non-White（预处理阶段已排除 Unknown，split CSV 中无 race=2+）
  age : 0=<60,   1=>=60（split CSV 原始 age 按 age_threshold 在线二分箱）
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from drafts.fairness_metrics import auc_by_group, eqodd_by_group

MIMIC_SEX_NAMES  = {0: "Male",      1: "Female"}
MIMIC_RACE_NAMES = {0: "White",     1: "Non-White"}
MIMIC_AGE_NAMES  = {0: "<60",       1: ">=60"}


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """单一子群内的 AUC；若该子群只含一种类别则返回 None。"""
    if len(np.unique(y_true)) < 2:
        return None
    return roc_auc_score(y_true, y_score)


def worst_case_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sex: np.ndarray,
    race: np.ndarray,
    age: np.ndarray,
) -> float | None:
    """
    计算 worst-case (minimax) AUC：在 Sex / Race / Age 各自分组以及
    Sex x Race x Age 交叉分组中，取 AUC 最低的子群作为模型选择判据。

    用于训练循环中按 epoch 监控，对比 "Overall AUC 选择" 与
    "Worst-case AUC 选择" 两种 model selection 策略对公平性的影响。

    Returns:
        所有有效子群（同时含正负样本）AUC 的最小值；若没有任何有效子群
        （理论上不应发生），返回 None。
    """
    y_true  = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    sex     = np.asarray(sex).astype(int)
    race    = np.asarray(race).astype(int)
    age     = np.asarray(age).astype(int)

    aucs: list[float] = []

    for group_ids, names in (
        (sex, MIMIC_SEX_NAMES),
        (race, MIMIC_RACE_NAMES),
        (age, MIMIC_AGE_NAMES),
    ):
        for auc, _n in auc_by_group(y_true, y_score, group_ids, names).values():
            if auc is not None:
                aucs.append(auc)

    for s in np.unique(sex):
        for r in np.unique(race):
            for a in np.unique(age):
                mask = (sex == s) & (race == r) & (age == a)
                if mask.sum() == 0:
                    continue
                auc = _safe_auc(y_true[mask], y_score[mask])
                if auc is not None:
                    aucs.append(auc)

    return min(aucs) if aucs else None


def print_mimic_fairness_report(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sex: np.ndarray,
    race: np.ndarray,
    age: np.ndarray,
    threshold: float = 0.0,
) -> None:
    """
    打印 MIMIC-CXR No Finding 模型的分组公平性报告（Overall AUC、
    分组 AUC、Equalized Odds、Worst-case AUC），指标定义与
    drafts/fairness_metrics.py 一致。

    Args:
        y_true   : [N] 0/1，No Finding 真实标签（U-Zeros：0=有病灶，1=无病灶/健康）
        y_score  : [N] 模型输出（logits 或概率均可）
        sex      : [N] 0=Male, 1=Female
        race     : [N] 0=White, 1=Non-White
        age      : [N] 0=<60, 1=>=60
        threshold: 判正类阈值（logits 用 0，probabilities 用 0.5）
    """
    y_true  = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    sex     = np.asarray(sex).astype(int)
    race    = np.asarray(race).astype(int)
    age     = np.asarray(age).astype(int)
    y_pred  = (y_score > threshold).astype(int)

    print("\n" + "=" * 60)
    print("MIMIC-CXR No Finding — Fairness Report")
    print("=" * 60)

    overall_auc = _safe_auc(y_true, y_score)
    print(f"\n[Overall]")
    print(f"  N        = {len(y_true)}")
    print(f"  AUC      = {overall_auc:.4f}" if overall_auc is not None else "  AUC      = N/A")
    print(f"  Accuracy = {(y_pred == y_true).mean():.4f}")

    print(f"\n[AUC by Sex]")
    for name, (auc, n) in auc_by_group(y_true, y_score, sex, MIMIC_SEX_NAMES).items():
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"  {name:<12s} n={n:6d}   AUC={auc_str}")

    print(f"\n[AUC by Race]")
    for name, (auc, n) in auc_by_group(y_true, y_score, race, MIMIC_RACE_NAMES).items():
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"  {name:<12s} n={n:6d}   AUC={auc_str}")

    print(f"\n[AUC by Age]")
    for name, (auc, n) in auc_by_group(y_true, y_score, age, MIMIC_AGE_NAMES).items():
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"  {name:<12s} n={n:6d}   AUC={auc_str}")

    print(f"\n[AUC by Sex x Race]")
    for s in np.unique(sex):
        for r in np.unique(race):
            mask = (sex == s) & (race == r)
            if mask.sum() == 0:
                continue
            name = f"{MIMIC_SEX_NAMES[int(s)]} & {MIMIC_RACE_NAMES[int(r)]}"
            auc = _safe_auc(y_true[mask], y_score[mask])
            auc_str = f"{auc:.4f}" if auc is not None else "N/A"
            print(f"  {name:<24s} n={int(mask.sum()):6d}   AUC={auc_str}")

    print(f"\n[AUC by Sex x Race x Age]")
    for s in np.unique(sex):
        for r in np.unique(race):
            for a in np.unique(age):
                mask = (sex == s) & (race == r) & (age == a)
                if mask.sum() == 0:
                    continue
                name = f"{MIMIC_SEX_NAMES[int(s)]} & {MIMIC_RACE_NAMES[int(r)]} & {MIMIC_AGE_NAMES[int(a)]}"
                auc = _safe_auc(y_true[mask], y_score[mask])
                auc_str = f"{auc:.4f}" if auc is not None else "N/A"
                print(f"  {name:<32s} n={int(mask.sum()):6d}   AUC={auc_str}")

    print(f"\n[Equalized Odds by Sex]")
    per_group, gap = eqodd_by_group(y_true, y_pred, sex, MIMIC_SEX_NAMES)
    for name, stats in per_group.items():
        tpr = f"{stats['tpr']:.4f}" if stats["tpr"] is not None else "N/A"
        fpr = f"{stats['fpr']:.4f}" if stats["fpr"] is not None else "N/A"
        print(f"  {name:<12s} n={stats['n']:6d}   TPR={tpr}   FPR={fpr}")
    print(f"  EqOdd gap = {gap:.4f}" if gap is not None else "  EqOdd gap = N/A")

    print(f"\n[Equalized Odds by Race]")
    per_group, gap = eqodd_by_group(y_true, y_pred, race, MIMIC_RACE_NAMES)
    for name, stats in per_group.items():
        tpr = f"{stats['tpr']:.4f}" if stats["tpr"] is not None else "N/A"
        fpr = f"{stats['fpr']:.4f}" if stats["fpr"] is not None else "N/A"
        print(f"  {name:<12s} n={stats['n']:6d}   TPR={tpr}   FPR={fpr}")
    print(f"  EqOdd gap = {gap:.4f}" if gap is not None else "  EqOdd gap = N/A")

    print(f"\n[Equalized Odds by Age]")
    per_group, gap = eqodd_by_group(y_true, y_pred, age, MIMIC_AGE_NAMES)
    for name, stats in per_group.items():
        tpr = f"{stats['tpr']:.4f}" if stats["tpr"] is not None else "N/A"
        fpr = f"{stats['fpr']:.4f}" if stats["fpr"] is not None else "N/A"
        print(f"  {name:<12s} n={stats['n']:6d}   TPR={tpr}   FPR={fpr}")
    print(f"  EqOdd gap = {gap:.4f}" if gap is not None else "  EqOdd gap = N/A")

    wc_auc = worst_case_auc(y_true, y_score, sex, race, age)
    print(f"\n[Worst-case AUC]")
    print(f"  min over Sex / Race / Age / Sex x Race x Age groups = "
          f"{wc_auc:.4f}" if wc_auc is not None else "  N/A")

    print("=" * 60)
