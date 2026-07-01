"""
PAPILA 青光眼分组公平性报告（MEDFAIR 对齐）
=============================================
复用 drafts/fairness_metrics.py 中与数据集无关的 auc_by_group / eqodd_by_group。
PAPILA 的敏感属性是 sex（二值）与 age_group（**二值**，60 岁分箱）——比 HAM（age 4 类）更简单，
无肤色/种族。

MEDFAIR 对 PAPILA 的公平性口径（论文 Table A9 / A13）：
  - Sex（2 组）、Age（2 组）、Sex×Age（4 交叉子群）各自的分组 AUC；
  - Worst-case AUC = 上述所有子群最小（max-min / minimax）；
  - AUC Gap        = 各轴内 max - min（group 公平性）；
  - Equalized Odds 对**两个**二值属性 Sex 与 Age 均报（区别于 HAM 仅 sex——HAM age 为多类）。

属性编码：
  sex       : 0=Male, 1=Female
  age_group : 0=<=60, 1=>60（无排除组，区别于 HAM 的 -1）
  label     : 0=非青光眼, 1=青光眼

⚠️ 小样本警告：PAPILA test≈84 眼-图（单次划分仅 ~9 阳性）。子群 AUC、尤其 Sex×Age 4 子群
   方差极大，建议在 5 折 GroupKFold 上合并解读（见 docs/papila_preprocessing_plan.md）。
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from drafts.fairness_metrics import auc_by_group, eqodd_by_group

PAPILA_SEX_NAMES = {0: "Male", 1: "Female"}
PAPILA_AGE_NAMES = {0: "<=60", 1: ">60"}


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """单一子群内的 AUC；若该子群只含一种类别则返回 None。"""
    if len(np.unique(y_true)) < 2:
        return None
    return roc_auc_score(y_true, y_score)


def worst_case_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sex: np.ndarray,
    age_group: np.ndarray,
) -> float | None:
    """
    计算 worst-case (minimax) AUC：在 Sex / Age 各自分组以及 Sex×Age 交叉分组中，
    取 AUC 最低的子群作为模型选择判据。

    Returns:
        所有有效子群（同时含正负样本）AUC 的最小值；若无有效子群返回 None。
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    sex = np.asarray(sex).astype(int)
    age_group = np.asarray(age_group).astype(int)

    aucs: list[float] = []
    for auc, _n in auc_by_group(y_true, y_score, sex, PAPILA_SEX_NAMES).values():
        if auc is not None:
            aucs.append(auc)
    for auc, _n in auc_by_group(y_true, y_score, age_group, PAPILA_AGE_NAMES).values():
        if auc is not None:
            aucs.append(auc)
    for s in (0, 1):
        for a in (0, 1):
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


def _print_eqodd(y_true, y_pred, group_ids, group_names, axis_name: str) -> None:
    """打印某二值属性轴的 Equalized Odds。"""
    print(f"\n[Equalized Odds by {axis_name}]")
    per_group, gap = eqodd_by_group(y_true, y_pred, group_ids, group_names)
    for name, stats in per_group.items():
        tpr = f"{stats['tpr']:.4f}" if stats["tpr"] is not None else "N/A"
        fpr = f"{stats['fpr']:.4f}" if stats["fpr"] is not None else "N/A"
        print(f"  {name:<8s} n={stats['n']:6d}   TPR={tpr}   FPR={fpr}")
    print(f"  EqOdd gap = {gap:.4f}" if gap is not None else "  EqOdd gap = N/A")


def print_papila_fairness_report(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sex: np.ndarray,
    age_group: np.ndarray,
    threshold: float = 0.0,
) -> None:
    """
    打印 PAPILA 青光眼模型的分组公平性报告（Overall、Sex / Age / Sex×Age 分组 AUC、
    各轴 AUC Gap、Sex 与 Age 的 Equalized Odds、Worst-case AUC）。

    Args:
        y_true   : [N] 0/1，青光眼真实标签（1=青光眼）
        y_score  : [N] 模型输出（logits 或概率均可；AUC 对单调变换不敏感）
        sex      : [N] 0=Male, 1=Female
        age_group: [N] 0=<=60, 1=>60
        threshold: 判正类阈值（logits 用 0，probabilities 用 0.5）
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    sex = np.asarray(sex).astype(int)
    age_group = np.asarray(age_group).astype(int)
    y_pred = (y_score > threshold).astype(int)

    print("\n" + "=" * 60)
    print("PAPILA glaucoma — Fairness Report")
    print("=" * 60)

    overall_auc = _safe_auc(y_true, y_score)
    print(f"\n[Overall]")
    print(f"  N        = {len(y_true)}")
    print(f"  AUC      = {overall_auc:.4f}" if overall_auc is not None else "  AUC      = N/A")
    print(f"  Accuracy = {(y_pred == y_true).mean():.4f}")

    for axis_ids, axis_names, axis_label in [
        (sex, PAPILA_SEX_NAMES, "Sex"),
        (age_group, PAPILA_AGE_NAMES, "Age group"),
    ]:
        print(f"\n[AUC by {axis_label}]")
        for name, (auc, n) in auc_by_group(y_true, y_score, axis_ids, axis_names).items():
            auc_str = f"{auc:.4f}" if auc is not None else "N/A"
            print(f"  {name:<8s} n={n:6d}   AUC={auc_str}")
        gap = auc_gap(y_true, y_score, axis_ids, axis_names)
        print(f"  AUC Gap (max-min) = {gap:.4f}" if gap is not None else "  AUC Gap = N/A")

    print(f"\n[AUC by Sex x Age group]  (4 subgroups)")
    inter_aucs: list[float] = []
    for s in (0, 1):
        for a in (0, 1):
            mask = (sex == s) & (age_group == a)
            if mask.sum() == 0:
                continue
            name = f"{PAPILA_SEX_NAMES[s]} & {PAPILA_AGE_NAMES[a]}"
            auc = _safe_auc(y_true[mask], y_score[mask])
            auc_str = f"{auc:.4f}" if auc is not None else "N/A"
            print(f"  {name:<18s} n={int(mask.sum()):6d}   AUC={auc_str}")
            if auc is not None:
                inter_aucs.append(auc)
    if len(inter_aucs) >= 2:
        print(f"  AUC Gap (max-min) = {max(inter_aucs) - min(inter_aucs):.4f}")

    _print_eqodd(y_true, y_pred, sex, PAPILA_SEX_NAMES, "Sex")
    _print_eqodd(y_true, y_pred, age_group, PAPILA_AGE_NAMES, "Age group")

    wc_auc = worst_case_auc(y_true, y_score, sex, age_group)
    print(f"\n[Worst-case AUC]")
    print(f"  min over Sex / Age / Sex x Age groups = "
          f"{wc_auc:.4f}" if wc_auc is not None else "  N/A")
    print("=" * 60)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 420
    y_true = rng.integers(0, 2, N)
    y_score = rng.standard_normal(N) + 0.5 * y_true
    sex = rng.integers(0, 2, N)
    age_group = rng.integers(0, 2, N)
    print_papila_fairness_report(y_true, y_score, sex, age_group, threshold=0.0)
