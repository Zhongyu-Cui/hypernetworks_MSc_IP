"""
操作点（阈值相关）公平度量（评审补强 R2.1）
==========================================
比较协议主报告的公平只挂 **AUC / worst-group AUC / AUC gap**——三者皆**阈值无关**。这带来两个缺口：
  1. **公平切面过窄**：AUC 只测「排序」是否跨子群一致，不测「在实际部署阈值上，各子群的
     真阳率(TPR)/假阳率(FPR)是否平等」，而后者才是 Equalized Odds 关心的操作层公平。
  2. **让 Reject-Option（ROC）在结构上失效**：ROC 是**阈值上的**公平干预（把决策边界附近的
     deprived 样本翻正、privileged 翻负），只在阈值处起作用。用阈值无关的 worst-group AUC 选
     其 margin θ 时，θ=0 恒不劣（AUC 对 ROC 的非单调翻转无动于衷），故 ROC 恒退化为 ERM，
     其真实目的（子群 TPR/FPR 奇偶）**从未被测量**。

本模块提供**阈值相关**的操作点公平度量，全部在已落盘预测上重算（零重训）：
  - `subgroup_tpr_fpr`      —— 各子群在给定全局阈值下的 TPR / FPR（复用 subgroup_auc 的 schema）。
  - `equalized_odds_gap`    —— Equalized Odds gap = max(TPR 组间极差, FPR 组间极差)。
  - `worst_group_tpr`       —— 给定阈值下的最小子群 TPR。
  - `threshold_at_overall_fpr` —— 选使**整体 FPR ≈ 目标**（如 0.2）的单一全局阈值（类别不均衡稳健）。
  - `operating_point_report`   —— 一站式：在①原生决策阈值(prob=0.5，ROC 锚点) 与
                                  ②固定整体 FPR=0.2 两个操作点上，给 EqOdds gap / worst-group TPR。

**分数尺度约定（关键）**：ERM/各 HN 落盘的是 **logits**（阈值 0 ⇔ prob 0.5）；ROC 落盘的是
**已调整概率**（阈值 0.5）。本模块统一在**概率**上工作：`score_is_prob=False` 时先 sigmoid，
`True` 时直接用。原生操作点恒取 prob=0.5，对两种尺度语义一致。

**子群集合**：复用 `subgroup_auc.subgroup_masks`（子群 schema 单一事实来源）。headline 度量默认
只用**边缘子群**（键不含 `|`，即各敏感轴自身的组）——Equalized Odds 经典定义即在受保护属性的
边缘组上，且边缘组样本量足够；数据集的 joint 小格（HAM sex×age ~26 眼）在**阈值**度量下会退化成
0/1 噪声，故不进 headline（全子群明细仍可选返回，供透明核查）。

**cross-validation**：self-test 用 sklearn（`confusion_matrix` / `recall_score` / `roc_curve`）
独立复算 TPR/FPR 与固定-FPR 阈值，逐一断言与本模块一致（R2.1 完成判据）。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_curve

from src.training.harness.subgroup_auc import subgroup_masks


def to_prob(score: np.ndarray, score_is_prob: bool) -> np.ndarray:
    """
    把模型分数统一成概率 ∈[0,1]。

    Args:
        score        : [N] 分数（logits 或已是概率）。
        score_is_prob: True=已是概率（如 ROC 调整后分数）；False=logits，需 sigmoid。

    Returns:
        [N] 概率数组（float）。
    """
    s = np.asarray(score, dtype=float)
    return s if score_is_prob else expit(s)


def _tpr_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float | None, float | None, int, int]:
    """
    单组的 (TPR, FPR, n_pos, n_neg)。TPR=P(Ŷ=1|Y=1)，FPR=P(Ŷ=1|Y=0)；无正/负样本对应项为 None。

    Args:
        y_true: [n] 0/1 真实标签（组内）。
        y_pred: [n] 0/1 预测（组内，已按阈值判定）。

    Returns:
        (tpr 或 None, fpr 或 None, 正样本数, 负样本数)。
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    pos = y_true == 1
    neg = y_true == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    tpr = float(y_pred[pos].mean()) if n_pos > 0 else None
    fpr = float(y_pred[neg].mean()) if n_neg > 0 else None
    return tpr, fpr, n_pos, n_neg


@dataclass(frozen=True)
class SubgroupRate:
    """单子群在某阈值下的操作点率。tpr/fpr 为 None 表示该组缺正/负样本。"""

    tpr: float | None
    fpr: float | None
    n: int
    n_pos: int
    n_neg: int


def subgroup_tpr_fpr(
    dataset: str,
    y_true: np.ndarray,
    prob: np.ndarray,
    attrs: dict[str, np.ndarray],
    threshold: float,
    *,
    marginal_only: bool = True,
) -> "OrderedDict[str, SubgroupRate]":
    """
    各子群在**单一全局阈值**下的 TPR / FPR（子群集合来自 `subgroup_masks`）。

    用同一全局阈值判定所有样本（部署单一操作点 → 暴露该操作点上的子群差异），再按子群切分算率。

    Args:
        dataset      : 数据集名（见 subgroup_masks）。
        y_true       : [N] 0/1 标签。
        prob         : [N] 概率（∈[0,1]，来自 to_prob）。
        attrs        : 敏感属性字典（按数据集所需键，直接 splat 进 subgroup_masks）。
        threshold    : 全局判正阈值；y_pred = (prob >= threshold)。
        marginal_only: True 仅返回边缘子群（键不含 `|`）；False 返回全子群（含 joint 交叉）。

    Returns:
        OrderedDict[子群键 -> SubgroupRate]。
    """
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    y_pred = (prob >= threshold).astype(int)
    masks = subgroup_masks(dataset, **{k: np.asarray(v) for k, v in attrs.items()})
    out: "OrderedDict[str, SubgroupRate]" = OrderedDict()
    for key, mask in masks.items():
        if marginal_only and "|" in key:
            continue
        tpr, fpr, n_pos, n_neg = _tpr_fpr(y_true[mask], y_pred[mask])
        out[key] = SubgroupRate(tpr=tpr, fpr=fpr, n=int(mask.sum()), n_pos=n_pos, n_neg=n_neg)
    return out


def equalized_odds_gap(
    rates: "OrderedDict[str, SubgroupRate]",
    *,
    min_class_n: int = 10,
) -> tuple[float | None, float | None, float | None]:
    """
    从子群率字典算 Equalized Odds gap。

    EqOdds 要求各子群 TPR 相等且 FPR 相等 → gap = max(TPR 组间极差, FPR 组间极差)，越小越公平。
    仅纳入**该类样本数 ≥ min_class_n** 的子群（TPR 需 ≥min_class_n 个正样本、FPR 需 ≥min_class_n
    个负样本），避免 n 极小子群的 0/1 退化率污染极差。

    Args:
        rates      : subgroup_tpr_fpr 的输出。
        min_class_n: 纳入某率所需的该类最小样本数。

    Returns:
        (tpr_gap, fpr_gap, eo_gap)；某项可比子群 < 2 时该项为 None，eo_gap 取两者中非 None 的较大者
        （皆 None 时为 None）。
    """
    tprs = [r.tpr for r in rates.values() if r.tpr is not None and r.n_pos >= min_class_n]
    fprs = [r.fpr for r in rates.values() if r.fpr is not None and r.n_neg >= min_class_n]
    tpr_gap = (max(tprs) - min(tprs)) if len(tprs) >= 2 else None
    fpr_gap = (max(fprs) - min(fprs)) if len(fprs) >= 2 else None
    candidates = [g for g in (tpr_gap, fpr_gap) if g is not None]
    eo_gap = max(candidates) if candidates else None
    return tpr_gap, fpr_gap, eo_gap


def worst_group_tpr(
    rates: "OrderedDict[str, SubgroupRate]",
    *,
    min_class_n: int = 10,
) -> float | None:
    """
    给定阈值下的**最小子群 TPR**（worst-group recall）；仅纳入正样本数 ≥ min_class_n 的子群。

    Args:
        rates      : subgroup_tpr_fpr 的输出。
        min_class_n: 纳入所需的最小正样本数。

    Returns:
        最小 TPR；无合格子群时 None。
    """
    tprs = [r.tpr for r in rates.values() if r.tpr is not None and r.n_pos >= min_class_n]
    return min(tprs) if tprs else None


def threshold_at_overall_fpr(
    y_true: np.ndarray, prob: np.ndarray, target_fpr: float,
) -> float:
    """
    选使**整体 FPR ≈ target**（不超过）的单一全局阈值——类别不均衡下比固定 0.5 更有意义的操作点。

    用 ROC 曲线取「FPR ≤ target 的最大 FPR」对应阈值（最保守地满足 FPR 上限）。

    Args:
        y_true    : [N] 0/1 标签。
        prob      : [N] 概率。
        target_fpr: 目标整体 FPR（如 0.2），∈(0,1)。

    Returns:
        全局阈值 t；判正 y_pred = (prob >= t)。

    Raises:
        ValueError: target_fpr 不在 (0,1)，或 y_true 无负样本（FPR 无定义）。
    """
    if not 0.0 < target_fpr < 1.0:
        raise ValueError(f"target_fpr 须在 (0,1)，收到 {target_fpr}")
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    if (y_true == 0).sum() == 0:
        raise ValueError("y_true 无负样本，整体 FPR 无定义。")
    fpr, _tpr, thr = roc_curve(y_true, prob)
    ok = np.nonzero(fpr <= target_fpr + 1e-12)[0]
    idx = int(ok[-1])  # fpr 单调升 → 最后一个满足者即 ≤target 的最大 FPR 点
    # roc_curve thr[0] 为 +inf（全判负）；退回一个有限阈值使操作点非平凡
    t = thr[idx]
    if not np.isfinite(t):
        t = float(prob.max()) + 1.0  # 全判负（FPR=0）
    return float(t)


@dataclass(frozen=True)
class OperatingPointReport:
    """
    某 run 的操作点公平汇总（两个操作点）。

    Attributes:
        native_threshold : 原生决策阈值（prob=0.5，ROC 锚点）。
        native_eo_gap    : 原生阈值上的 EqOdds gap。
        native_tpr_gap / native_fpr_gap: 原生阈值上的 TPR / FPR 组间极差。
        native_worst_tpr : 原生阈值上的最小子群 TPR。
        fpr_target       : 固定-FPR 操作点的目标整体 FPR（如 0.2）。
        fpr_threshold    : 达成该整体 FPR 的全局阈值。
        fpr_eo_gap       : 该操作点上的 EqOdds gap。
        fpr_worst_tpr    : 该操作点上的最小子群 TPR（= worst-group TPR @ 固定 FPR）。
        native_rates / fpr_rates: 两操作点的逐子群率明细（边缘子群）。
    """

    native_threshold: float
    native_eo_gap: float | None
    native_tpr_gap: float | None
    native_fpr_gap: float | None
    native_worst_tpr: float | None
    fpr_target: float
    fpr_threshold: float
    fpr_eo_gap: float | None
    fpr_worst_tpr: float | None
    native_rates: "OrderedDict[str, SubgroupRate]"
    fpr_rates: "OrderedDict[str, SubgroupRate]"


def operating_point_report(
    dataset: str,
    y_true: np.ndarray,
    score: np.ndarray,
    attrs: dict[str, np.ndarray],
    *,
    score_is_prob: bool,
    target_fpr: float = 0.2,
    min_class_n: int = 10,
    marginal_only: bool = True,
) -> OperatingPointReport:
    """
    一站式操作点公平报告：在原生阈值(prob=0.5) 与 固定整体 FPR=target 两个操作点上汇总。

    Args:
        dataset      : 数据集名。
        y_true       : [N] 0/1 标签。
        score        : [N] 模型分数（logits 或概率，由 score_is_prob 指定）。
        attrs        : 敏感属性字典。
        score_is_prob: 见 to_prob（ERM/HN=False；ROC=True）。
        target_fpr   : 固定-FPR 操作点目标（默认 0.2）。
        min_class_n  : headline 度量纳入子群的该类最小样本数。
        marginal_only: headline 是否仅用边缘子群（默认 True）。

    Returns:
        OperatingPointReport。
    """
    y_true = np.asarray(y_true).astype(int)
    prob = to_prob(score, score_is_prob)

    # ① 原生决策阈值（prob=0.5，logits 的 0，ROC 的锚点）
    native_rates = subgroup_tpr_fpr(dataset, y_true, prob, attrs, 0.5, marginal_only=marginal_only)
    n_tpr_gap, n_fpr_gap, n_eo = equalized_odds_gap(native_rates, min_class_n=min_class_n)
    n_worst = worst_group_tpr(native_rates, min_class_n=min_class_n)

    # ② 固定整体 FPR=target 操作点（不均衡稳健）
    t_fpr = threshold_at_overall_fpr(y_true, prob, target_fpr)
    fpr_rates = subgroup_tpr_fpr(dataset, y_true, prob, attrs, t_fpr, marginal_only=marginal_only)
    _f_tpr_gap, _f_fpr_gap, f_eo = equalized_odds_gap(fpr_rates, min_class_n=min_class_n)
    f_worst = worst_group_tpr(fpr_rates, min_class_n=min_class_n)

    return OperatingPointReport(
        native_threshold=0.5,
        native_eo_gap=n_eo, native_tpr_gap=n_tpr_gap, native_fpr_gap=n_fpr_gap,
        native_worst_tpr=n_worst,
        fpr_target=target_fpr, fpr_threshold=t_fpr, fpr_eo_gap=f_eo, fpr_worst_tpr=f_worst,
        native_rates=native_rates, fpr_rates=fpr_rates,
    )


# ============================================================
# self-test：sklearn 交叉验证 TPR/FPR、固定-FPR 阈值、EqOdds gap、worst TPR
# ============================================================
def _selftest() -> None:
    """用 sklearn 独立复算，断言本模块 TPR/FPR/固定-FPR 阈值/EqOdds gap 一致。"""
    from sklearn.metrics import confusion_matrix, recall_score

    rng = np.random.default_rng(0)

    # --- 1) _tpr_fpr 与 sklearn confusion_matrix 交叉验证 ---
    y = rng.integers(0, 2, 500)
    pred = rng.integers(0, 2, 500)
    tpr, fpr, npos, nneg = _tpr_fpr(y, pred)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    assert abs(tpr - tp / (tp + fn)) < 1e-12, (tpr, tp / (tp + fn))
    assert abs(fpr - fp / (fp + tn)) < 1e-12, (fpr, fp / (fp + tn))
    assert npos == (tp + fn) and nneg == (tn + fp)
    # recall_score 复核 TPR
    assert abs(tpr - recall_score(y, pred, pos_label=1)) < 1e-12

    # --- 2) to_prob：logits 阈值0 ⇔ prob 阈值0.5 ---
    logits = rng.standard_normal(300)
    p = to_prob(logits, score_is_prob=False)
    assert np.array_equal((logits >= 0), (p >= 0.5))
    assert np.array_equal(to_prob(p, score_is_prob=True), p)

    # --- 3) threshold_at_overall_fpr：达成的整体 FPR ≤ target 且尽量贴近；与 roc_curve 复核 ---
    y = rng.integers(0, 2, 4000)
    prob = np.clip(0.15 * (2 * y - 1) + rng.uniform(0, 1, 4000), 0, 1)
    for target in (0.1, 0.2, 0.35):
        t = threshold_at_overall_fpr(y, prob, target)
        pred = (prob >= t).astype(int)
        _, achieved_fpr, _, _ = _tpr_fpr(y, pred)
        assert achieved_fpr <= target + 1e-9, (target, achieved_fpr)
        # 贴近性：ROC 网格上再放宽一档就会超 target（说明取的是最大不超者）
        fpr_grid, _, thr = roc_curve(y, prob)
        below = fpr_grid[fpr_grid <= target + 1e-12]
        assert abs(achieved_fpr - below.max()) < 1e-9

    # --- 4) 端到端 report + EqOdds gap / worst TPR 手工复核（mimic schema）---
    n = 6000
    y = rng.integers(0, 2, n)
    sex = rng.integers(0, 2, n); race = rng.integers(0, 2, n); age = rng.integers(0, 2, n)
    # 让 age==1 组分数系统偏低 → 该组 TPR 更低（制造可检的 EqOdds 差异）
    logits = 0.8 * (2 * y - 1) - 0.6 * age + rng.standard_normal(n)
    attrs = {"sex": sex, "race": race, "age": age}
    rep = operating_point_report("mimic", y, logits, attrs, score_is_prob=False, target_fpr=0.2)
    # 原生阈值：手工复核 age 边缘组 TPR
    prob = to_prob(logits, False)
    pred = (prob >= 0.5).astype(int)
    tpr_age0 = recall_score(y[age == 0], pred[age == 0], pos_label=1)
    tpr_age1 = recall_score(y[age == 1], pred[age == 1], pos_label=1)
    assert abs(rep.native_rates["age:<60"].tpr - tpr_age0) < 1e-12
    assert abs(rep.native_rates["age:>=60"].tpr - tpr_age1) < 1e-12
    # age==1 分数偏低 → 其 TPR 更小 → 应等于 worst-group TPR（边缘组里最差之一）
    assert rep.native_worst_tpr <= tpr_age1 + 1e-12
    # EqOdds gap 手工复核 = 边缘组 TPR/FPR 极差的较大者
    tprs = [r.tpr for r in rep.native_rates.values() if r.tpr is not None and r.n_pos >= 10]
    fprs = [r.fpr for r in rep.native_rates.values() if r.fpr is not None and r.n_neg >= 10]
    assert abs(rep.native_eo_gap - max(max(tprs) - min(tprs), max(fprs) - min(fprs))) < 1e-12
    # 固定-FPR 操作点：整体 FPR ≤ 0.2
    predf = (prob >= rep.fpr_threshold).astype(int)
    _, ov_fpr, _, _ = _tpr_fpr(y, predf)
    assert ov_fpr <= 0.2 + 1e-9

    # --- 5) 单轴数据集（fitzpatrick skin）无 joint 键，marginal_only 不删任何组 ---
    n = 3000
    y = rng.integers(0, 2, n); skin = rng.integers(0, 6, n)
    logits = 0.7 * (2 * y - 1) + rng.standard_normal(n)
    rep2 = operating_point_report("fitzpatrick", y, logits, {"skin": skin}, score_is_prob=False)
    assert len(rep2.native_rates) == 6

    # --- 6) 校验：坏 target_fpr / 无负样本 ---
    for bad in (lambda: threshold_at_overall_fpr(y, prob[:n], 0.0),
                lambda: threshold_at_overall_fpr(np.ones(10, int), np.zeros(10), 0.2)):
        try:
            bad(); raise AssertionError("应报错")
        except ValueError:
            pass

    print("operating_point_fairness self-test 全部通过 ✓ "
          "(sklearn 交叉验证 TPR/FPR/recall、固定-FPR 阈值、EqOdds gap、worst TPR、端到端 report、校验)")


if __name__ == "__main__":
    _selftest()
