"""
Fairness Metrics for Binary Classification
==========================================
- Overall AUC
- Per-group AUC (by sex, by race, by sex*race)
- Equalized Odds (gap of TPR and FPR across groups)

所有函数都期望 numpy 数组作为输入:
    y_true : [N], 0/1 整数
    y_score: [N], 实数 (未经 sigmoid 的 logit, 或经过 sigmoid 的概率都行,
             因为 AUC 对单调变换不敏感, TPR/FPR 用的是阈值判定)
    y_pred : [N], 0/1 整数 (阈值判定后的结果)
    sex    : [N], 0/1
    race   : [N], 0~4
"""

import numpy as np
from sklearn.metrics import roc_auc_score


SEX_NAMES = {0: "Male", 1: "Female"}
RACE_NAMES = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}


# ============================================================
# 单个子群的 AUC (安全版本)
# ============================================================
def _safe_auc(y_true, y_score):
    """单一群内的 AUC; 如果某群只有一个类别则返回 None."""
    if len(np.unique(y_true)) < 2:
        return None
    return roc_auc_score(y_true, y_score)


# ============================================================
# 分组 AUC
# ============================================================
def auc_by_group(y_true, y_score, group_ids, group_names):
    """
    返回 dict: {group_name: (auc, n)}
    auc 为 None 表示该组缺少正/负样本无法计算。
    """
    out = {}
    for gid in np.unique(group_ids):
        mask = group_ids == gid
        name = group_names.get(int(gid), f"group_{gid}")
        out[name] = (_safe_auc(y_true[mask], y_score[mask]), int(mask.sum()))
    return out


def auc_by_sex_race(y_true, y_score, sex, race):
    """按 sex*race 联合分组的 AUC."""
    out = {}
    for s in np.unique(sex):
        for r in np.unique(race):
            mask = (sex == s) & (race == r)
            if mask.sum() == 0:
                continue
            name = f"{SEX_NAMES[int(s)]} & {RACE_NAMES[int(r)]}"
            out[name] = (_safe_auc(y_true[mask], y_score[mask]), int(mask.sum()))
    return out


# ============================================================
# Equalized Odds
# ============================================================
def _tpr_fpr(y_true, y_pred):
    """计算单个群的 TPR 和 FPR. 若某项分母为 0 返回 None."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pos = y_true == 1
    neg = y_true == 0
    tpr = y_pred[pos].mean() if pos.sum() > 0 else None  # P(Ŷ=1 | Y=1)
    fpr = y_pred[neg].mean() if neg.sum() > 0 else None  # P(Ŷ=1 | Y=0)
    return tpr, fpr


def eqodd_by_group(y_true, y_pred, group_ids, group_names):
    """
    返回:
        per_group: {name: {"tpr": ..., "fpr": ..., "n": ...}}
        gap      : max(TPR) - min(TPR) 和 max(FPR) - min(FPR) 的较大者
                   (只对同时有 TPR 和 FPR 的群做聚合)
    """
    per_group = {}
    tprs, fprs = [], []
    for gid in np.unique(group_ids):
        mask = group_ids == gid
        name = group_names.get(int(gid), f"group_{gid}")
        tpr, fpr = _tpr_fpr(y_true[mask], y_pred[mask])
        per_group[name] = {"tpr": tpr, "fpr": fpr, "n": int(mask.sum())}
        if tpr is not None: tprs.append(tpr)
        if fpr is not None: fprs.append(fpr)

    if len(tprs) >= 2 and len(fprs) >= 2:
        gap = max(max(tprs) - min(tprs), max(fprs) - min(fprs))
    else:
        gap = None

    return per_group, gap


# ============================================================
# 一站式: 打印完整报告
# ============================================================
def print_fairness_report(y_true, y_score, sex, race, threshold=0.0):
    """
    y_true : [N] 0/1
    y_score: [N] logits (或 probabilities, 都可以)
    sex    : [N] 0/1
    race   : [N] 0~4
    threshold: 判正类的阈值 (logits 用 0, probs 用 0.5)
    """
    y_true  = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    sex     = np.asarray(sex).astype(int)
    race    = np.asarray(race).astype(int)
    y_pred  = (y_score > threshold).astype(int)

    print("\n" + "=" * 60)
    print("Fairness Report")
    print("=" * 60)

    # --- Overall AUC ---
    overall_auc = _safe_auc(y_true, y_score)
    print(f"\n[Overall]")
    print(f"  N       = {len(y_true)}")
    print(f"  AUC     = {overall_auc:.4f}")
    print(f"  Accuracy= {(y_pred == y_true).mean():.4f}")

    # --- AUC by sex ---
    print(f"\n[AUC by Sex]")
    for name, (auc, n) in auc_by_group(y_true, y_score, sex, SEX_NAMES).items():
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"  {name:<10s} n={n:5d}   AUC={auc_str}")

    # --- AUC by race ---
    print(f"\n[AUC by Race]")
    for name, (auc, n) in auc_by_group(y_true, y_score, race, RACE_NAMES).items():
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"  {name:<10s} n={n:5d}   AUC={auc_str}")

    # --- AUC by sex x race ---
    print(f"\n[AUC by Sex x Race]")
    for name, (auc, n) in auc_by_sex_race(y_true, y_score, sex, race).items():
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"  {name:<22s} n={n:5d}   AUC={auc_str}")

    # --- Equalized Odds ---
    print(f"\n[Equalized Odds by Sex]")
    per_g, gap = eqodd_by_group(y_true, y_pred, sex, SEX_NAMES)
    for name, s in per_g.items():
        tpr = f"{s['tpr']:.4f}" if s['tpr'] is not None else "N/A"
        fpr = f"{s['fpr']:.4f}" if s['fpr'] is not None else "N/A"
        print(f"  {name:<10s} n={s['n']:5d}   TPR={tpr}   FPR={fpr}")
    print(f"  EqOdd gap = {gap:.4f}" if gap is not None else "  EqOdd gap = N/A")

    print(f"\n[Equalized Odds by Race]")
    per_g, gap = eqodd_by_group(y_true, y_pred, race, RACE_NAMES)
    for name, s in per_g.items():
        tpr = f"{s['tpr']:.4f}" if s['tpr'] is not None else "N/A"
        fpr = f"{s['fpr']:.4f}" if s['fpr'] is not None else "N/A"
        print(f"  {name:<10s} n={s['n']:5d}   TPR={tpr}   FPR={fpr}")
    print(f"  EqOdd gap = {gap:.4f}" if gap is not None else "  EqOdd gap = N/A")

    print("=" * 60)


if __name__ == "__main__":
    # 生成一些假数据做自检
    rng = np.random.default_rng(0)
    N = 1000
    y_true = rng.integers(0, 2, N)
    y_score = rng.standard_normal(N) + 0.5 * y_true  # 正类分数偏高
    sex = rng.integers(0, 2, N)
    race = rng.integers(0, 5, N)

    print_fairness_report(y_true, y_score, sex, race, threshold=0.0)
