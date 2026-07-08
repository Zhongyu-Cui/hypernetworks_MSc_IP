"""
Reject-Option Classification 后处理（比较协议 A4.1）
==================================================
比较协议 §1 把 **ROC**（Reject-Option Classification, Kamiran et al., ICDM 2012）列为「属性感知、
最浅介入」的基线：**复用 ERM 已训练模型的分数做后处理、零训练**。本模块实现 A4.1「reject-option
后处理（近边界预测调整）」的核心机制，A4.2 负责接入各数据集评估/输出管线。

**方法（对齐 Kamiran 原义，分数版以便 AUC 评估）**：
  - 临界区（critical region）：决策边界 p=0.5 附近的不确定样本，`|p − 0.5| ≤ margin θ`。
  - 区内调整（favor deprived / disfavor privileged）：
      · **deprived**（弱势子群）→ 授予 favorable（正类）：`p → max(p, 1−p) ≥ 0.5`；
      · **privileged**（优势子群）→ 授予 unfavorable（负类）：`p → min(p, 1−p) ≤ 0.5`。
    即把区内「本会判负的 deprived」翻到正侧、「本会判正的 privileged」翻到负侧，区外分数不动。
  - 该变换只动临界区样本，故会改变**子群内排序** → 子群 AUC 随之变化（普通 per-group 单调平移
    不改子群 AUC，故必须用这种非单调的区内翻转才可能撬动 worst-group AUC）。

**逐轴二分 privileged/deprived（本项目多子群设定，按你的决定：按表现二分）**：
  经典 ROC 需二值受保护属性；本项目敏感轴多为多水平（skin 6 级、sex×age 交叉）。据「按表现二分」：
  在**验证集**上算该轴各子群 AUC，低于中位数者判 deprived、其余 privileged（子群 AUC 不可算的
  小子群并入 deprived——它们正是弱势侧）。见 `binarize_axis_by_performance`。

**margin 选择（零训练、仅后处理）**：在 val 上对 θ 网格（含 0）取使目标（worst-group AUC）最优者。
  θ=0 时临界区为空、退化为 ERM，故 ROC 在 val 目标上**恒不劣于 ERM**（网格含 0 保证）。见 `select_margin`。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.training.harness.subgroup_auc import subgroup_auc_vector, vector_worst_case
from src.training.harness.significance import worst_and_gap

# 默认 margin 网格：0（=ERM）到 0.5（全样本入临界区），步长 0.05
DEFAULT_MARGIN_GRID: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(11))  # 0.0..0.5

# 各数据集 ROC 默认二分轴（可覆盖）：age 为多数数据集主公平轴，fitzpatrick 用 skin
DATASET_PRIMARY_AXIS: dict[str, str] = {
    "mimic": "age", "chexpert": "age", "ham10000": "age",
    "fitzpatrick": "skin", "papila": "age",
}


def logits_to_prob(logits: np.ndarray) -> np.ndarray:
    """把模型 logits 经 sigmoid 转为概率（用 scipy expit，数值稳定、不溢出）。"""
    from scipy.special import expit
    return expit(np.asarray(logits, dtype=float))


def reject_option_adjust(
    prob: np.ndarray, deprived_mask: np.ndarray, margin: float,
) -> np.ndarray:
    """
    A4.1 核心：对临界区样本做 reject-option 分数调整（favor deprived / disfavor privileged）。

    Args:
        prob         : [N] 概率（∈[0,1]，通常来自 logits_to_prob）。
        deprived_mask: [N] 布尔，True=deprived（弱势），False=privileged。
        margin       : 临界区半宽 θ∈[0,0.5]；`|p−0.5|≤θ` 为区内。θ=0 → 退化为原分数。

    Returns:
        [N] 调整后的分数（区外 = 原 prob；区内 deprived=max(p,1−p)，privileged=min(p,1−p)）。

    Raises:
        ValueError: margin 不在 [0,0.5]，或形状不一致。
    """
    prob = np.asarray(prob, dtype=float)
    deprived_mask = np.asarray(deprived_mask, dtype=bool)
    if prob.shape != deprived_mask.shape:
        raise ValueError(f"prob {prob.shape} 与 deprived_mask {deprived_mask.shape} 形状不一致。")
    if not 0.0 <= margin <= 0.5:
        raise ValueError(f"margin 须在 [0,0.5]，收到 {margin}")

    adjusted = prob.copy()
    in_critical = np.abs(prob - 0.5) <= margin
    fav = np.maximum(prob, 1.0 - prob)   # favorable：推向正侧（≥0.5）
    unfav = np.minimum(prob, 1.0 - prob)  # unfavorable：推向负侧（≤0.5）
    adjusted[in_critical & deprived_mask] = fav[in_critical & deprived_mask]
    adjusted[in_critical & ~deprived_mask] = unfav[in_critical & ~deprived_mask]
    return adjusted


def binarize_axis_by_performance(
    group_ids: np.ndarray, per_group_score: dict[int, float | None],
) -> np.ndarray:
    """
    按验证集表现把某敏感轴的子群二分为 deprived / privileged（A4.1 逐轴二分）。

    子群得分（通常 val 子群 AUC）低于**中位数**者判 deprived；得分为 None（子群太小/单类，AUC 不可算）
    的子群并入 deprived（弱势侧）。全部子群同分时按「严格低于中位数」判定 → 无 deprived（此时 ROC
    对该轴无可调，等价 ERM，合理）。

    Args:
        group_ids      : [N] 每个样本在该轴的子群 id（整数）。
        per_group_score: {子群 id -> 得分或 None}（如 val 子群 AUC），须覆盖 group_ids 中出现的所有 id。

    Returns:
        [N] 布尔 deprived_mask。

    Raises:
        ValueError: group_ids 中某 id 不在 per_group_score 键内。
    """
    group_ids = np.asarray(group_ids)
    valid_scores = [s for s in per_group_score.values() if s is not None]
    # 中位数只在可算子群上取；无可算子群 → 阈值设 +inf（全判 deprived，最保守）
    thr = float(np.median(valid_scores)) if valid_scores else float("inf")

    deprived_ids = {
        gid for gid, s in per_group_score.items()
        if s is None or s < thr   # None 或低于中位数 → deprived
    }
    present = set(np.unique(group_ids).tolist())
    missing = present - set(per_group_score.keys())
    if missing:
        raise ValueError(f"group_ids 含未在 per_group_score 中登记的子群：{sorted(missing)}")
    return np.isin(group_ids, list(deprived_ids))


@dataclass(frozen=True)
class MarginSelection:
    """
    margin 选择结果。

    Attributes:
        best_margin: 使 val 目标最优的 θ。
        best_value : 该 θ 下的 val 目标值。
        erm_value  : θ=0（=ERM）的 val 目标值（对照，best_value 恒 ≥ 此）。
        grid       : 扫描的 margin 网格。
        curve      : 各 margin 对应的目标值（与 grid 对齐；None=该 θ 下目标不可算）。
    """

    best_margin: float
    best_value: float
    erm_value: float | None
    grid: tuple[float, ...]
    curve: tuple[float | None, ...]


def select_margin(
    prob: np.ndarray,
    deprived_mask: np.ndarray,
    objective_fn,
    *,
    margins: "tuple[float, ...]" = DEFAULT_MARGIN_GRID,
) -> MarginSelection:
    """
    A4.1：在 val 上扫描 margin 网格，选使目标（越大越好，如 worst-group AUC）最优的 θ。

    网格含 0.0（=ERM），故所选 θ 在 val 目标上恒不劣于 ERM（零训练、纯后处理的安全性）。目标不可算
    （返回 None）的 margin 被跳过；平手取更小的 θ（更保守、更接近 ERM）。

    Args:
        prob         : [N] val 概率。
        deprived_mask: [N] val deprived 掩码（来自 binarize_axis_by_performance）。
        objective_fn : 可调用，`objective_fn(adjusted_prob)->float|None`；越大越好（如 val
                       worst-group AUC，内部按子群向量计算）。
        margins      : 待扫描的 θ 网格（须含 0.0 以保证不劣于 ERM）。

    Returns:
        MarginSelection。

    Raises:
        ValueError: 所有 margin 目标均不可算，或 margins 为空。
    """
    if not margins:
        raise ValueError("select_margin：margins 网格为空。")
    curve: list[float | None] = []
    for m in margins:
        adj = reject_option_adjust(prob, deprived_mask, m)
        val = objective_fn(adj)
        curve.append(None if val is None else float(val))

    erm_value = curve[margins.index(0.0)] if 0.0 in margins else None
    valid = [(m, v) for m, v in zip(margins, curve) if v is not None]
    if not valid:
        raise ValueError("select_margin：所有 margin 下目标均不可算。")
    # 目标最大；平手取更小 θ（更接近 ERM）
    best_margin, best_value = max(valid, key=lambda mv: (mv[1], -mv[0]))
    return MarginSelection(
        best_margin=best_margin, best_value=best_value, erm_value=erm_value,
        grid=tuple(margins), curve=tuple(curve),
    )


# ============================================================
# A4.2：接入评估/输出管线（数据集无关）
# ============================================================
def _per_group_auc(
    y_true: np.ndarray, y_score: np.ndarray, group_ids: np.ndarray,
) -> dict[int, float | None]:
    """该轴每个子群 id 的 AUC（子群空/单类 → None），供 binarize_axis_by_performance 判 deprived。"""
    from sklearn.metrics import roc_auc_score
    out: dict[int, float | None] = {}
    for gid in np.unique(group_ids):
        mask = group_ids == gid
        yt = y_true[mask]
        out[int(gid)] = (float(roc_auc_score(yt, y_score[mask]))
                         if len(np.unique(yt)) >= 2 else None)
    return out


@dataclass
class ROCResult:
    """
    ROC 后处理在某数据集/某二分轴上的结果（method="roc"，派生自 ERM 预测）。

    Attributes:
        dataset            : 数据集名。
        axis               : 二分所用敏感轴（"sex"/"race"/"age"/"skin"）。
        best_margin        : val 选出的临界区半宽 θ。
        deprived_group_ids : 该轴被判 deprived 的子群 id 列表（由 val 子群 AUC 定）。
        per_group_val_auc  : 该轴各子群的 val AUC（None=不可算）。
        val_worst_before/after   : val 全子群 worst-group AUC（ERM θ=0 vs ROC θ*）。
        test_overall_before/after: test Overall AUC（ERM vs ROC）。
        test_worst_before/after  : test 全子群 worst-group AUC（ERM vs ROC）。
        test_gap_before/after    : test 全子群 AUC gap（ERM vs ROC）。
        adjusted_test_scores     : [N] ROC 调整后的 test 分数（供 A2 显著性/下游使用）。
    """

    dataset: str
    axis: str
    best_margin: float
    deprived_group_ids: list[int]
    per_group_val_auc: dict[int, float | None]
    val_worst_before: float | None
    val_worst_after: float | None
    test_overall_before: float
    test_overall_after: float
    test_worst_before: float | None
    test_worst_after: float | None
    test_gap_before: float | None
    test_gap_after: float | None
    adjusted_test_scores: np.ndarray


def apply_roc(
    dataset: str,
    axis: str,
    *,
    y_val_true: np.ndarray,
    y_val_logits: np.ndarray,
    val_attrs: dict[str, np.ndarray],
    y_test_true: np.ndarray,
    y_test_logits: np.ndarray,
    test_attrs: dict[str, np.ndarray],
    margins: "tuple[float, ...]" = DEFAULT_MARGIN_GRID,
) -> ROCResult:
    """
    A4.2：在指定二分轴上对 ERM 预测做 ROC 后处理，并评估 test 全子群公平度量（数据集无关）。

    流程（零训练）：val 子群 AUC 定该轴 deprived/privileged → val 上按**全子群 worst-group AUC**
    选 θ（含 θ=0 保证不劣 ERM）→ 用同一 deprived 映射与 θ 调整 test 分数 → 在 test 上算全子群
    worst-group / gap / Overall AUC。worst-group / gap 恒对**全子群向量**（含所有轴与交叉），仅
    deprived 二分用单轴 `axis`。

    Args:
        dataset      : {"mimic","chexpert","ham10000","fitzpatrick","papila"}。
        axis         : 二分轴，须是 val_attrs/test_attrs 的键（如 "age"）。
        y_val_true / y_val_logits / val_attrs   : ERM 在 val 上的标签/logits/属性字典。
        y_test_true / y_test_logits / test_attrs: ERM 在 test 上的对应量。
        margins      : θ 网格（须含 0.0）。

    Returns:
        ROCResult。

    Raises:
        ValueError: axis 不在属性字典中。
    """
    if axis not in val_attrs or axis not in test_attrs:
        raise ValueError(f"axis={axis!r} 不在属性字典中（可用：{sorted(val_attrs)}）。")
    from sklearn.metrics import roc_auc_score

    y_val_true = np.asarray(y_val_true).astype(int)
    y_test_true = np.asarray(y_test_true).astype(int)
    val_prob = logits_to_prob(y_val_logits)
    test_prob = logits_to_prob(y_test_logits)
    val_gids = np.asarray(val_attrs[axis]).astype(int)
    test_gids = np.asarray(test_attrs[axis]).astype(int)

    # 1) val 子群 AUC 定该轴 deprived（同一 per-group 得分用于 val 与 test 掩码）
    per_group_val_auc = _per_group_auc(y_val_true, val_prob, val_gids)
    val_deprived = binarize_axis_by_performance(val_gids, per_group_val_auc)
    # test 掩码：test 可能出现 val 未见的子群 id → 补记为 None（并入 deprived，弱势侧）
    per_group_for_test = dict(per_group_val_auc)
    for gid in np.unique(test_gids):
        per_group_for_test.setdefault(int(gid), None)
    test_deprived = binarize_axis_by_performance(test_gids, per_group_for_test)
    sorted_gids = np.array(sorted(per_group_val_auc))
    deprived_flags = binarize_axis_by_performance(sorted_gids, per_group_val_auc)
    deprived_ids = [int(g) for g, dep in zip(sorted_gids, deprived_flags) if dep]

    # 2) val 目标 = 全子群 worst-group AUC；选 θ（含 0 → 不劣 ERM）
    def val_objective(adj: np.ndarray) -> float | None:
        return vector_worst_case(
            subgroup_auc_vector(dataset, y_val_true, adj, **_as_int_attrs(val_attrs))
        )
    margin_sel = select_margin(val_prob, val_deprived, val_objective, margins=margins)
    best_margin = margin_sel.best_margin

    # 3) 用同一映射 + θ 调整 test 分数
    adjusted_test = reject_option_adjust(test_prob, test_deprived, best_margin)

    # 4) test 度量：ERM（θ=0）vs ROC（θ*）
    vec_before = subgroup_auc_vector(dataset, y_test_true, test_prob, **_as_int_attrs(test_attrs))
    vec_after = subgroup_auc_vector(dataset, y_test_true, adjusted_test, **_as_int_attrs(test_attrs))
    worst_b, gap_b = worst_and_gap(vec_before)
    worst_a, gap_a = worst_and_gap(vec_after)

    return ROCResult(
        dataset=dataset, axis=axis, best_margin=best_margin,
        deprived_group_ids=deprived_ids, per_group_val_auc=per_group_val_auc,
        val_worst_before=margin_sel.erm_value, val_worst_after=margin_sel.best_value,
        test_overall_before=float(roc_auc_score(y_test_true, test_prob)),
        test_overall_after=float(roc_auc_score(y_test_true, adjusted_test)),
        test_worst_before=worst_b, test_worst_after=worst_a,
        test_gap_before=gap_b, test_gap_after=gap_a,
        adjusted_test_scores=adjusted_test,
    )


def _as_int_attrs(attrs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """把属性字典各数组转 int（subgroup_auc_vector 内部按 int 比较；age 为 age_group 已是 int 编码）。"""
    return {k: np.asarray(v).astype(int) for k, v in attrs.items()}


def apply_roc_auto(
    dataset: str,
    *,
    y_val_true: np.ndarray,
    y_val_logits: np.ndarray,
    val_attrs: dict[str, np.ndarray],
    y_test_true: np.ndarray,
    y_test_logits: np.ndarray,
    test_attrs: dict[str, np.ndarray],
    margins: "tuple[float, ...]" = DEFAULT_MARGIN_GRID,
) -> ROCResult:
    """
    自动选轴的 ROC：对属性字典中每个敏感轴各跑一次 `apply_roc`，选 **val worst-group AUC 最优**者。

    数据集无关：轴集合 = val_attrs 的键（mimic/chexpert={sex,race,age}、ham/papila={sex,age}、
    fitzpatrick={skin}）。平手取 `DATASET_PRIMARY_AXIS` 的默认轴（更稳定、可解释）。

    Returns:
        选中轴的 ROCResult。
    """
    primary = DATASET_PRIMARY_AXIS.get(dataset)
    results: list[ROCResult] = []
    for axis in val_attrs:
        results.append(apply_roc(
            dataset, axis,
            y_val_true=y_val_true, y_val_logits=y_val_logits, val_attrs=val_attrs,
            y_test_true=y_test_true, y_test_logits=y_test_logits, test_attrs=test_attrs,
            margins=margins,
        ))
    # 选 val worst-group after 最优；平手偏好主轴，再偏好更小 margin
    def key(r: ROCResult) -> tuple:
        v = -np.inf if r.val_worst_after is None else r.val_worst_after
        return (v, r.axis == primary, -r.best_margin)
    return max(results, key=key)


def roc_fairness_report(result: ROCResult) -> None:
    """打印 ROC 后处理的 before/after 对照（输出管线用）。"""
    def f(v: float | None) -> str:
        return "N/A" if v is None else f"{v:.4f}"
    print(f"\n{'#' * 70}\n# ROC 后处理 (method=roc, 二分轴={result.axis}, θ*={result.best_margin})\n{'#' * 70}")
    print(f"deprived 子群 id: {result.deprived_group_ids}  "
          f"(该轴 val 子群 AUC: {{{', '.join(f'{k}:{f(v)}' for k, v in result.per_group_val_auc.items())}}})")
    print(f"val  worst-group AUC : ERM {f(result.val_worst_before)} → ROC {f(result.val_worst_after)}")
    print(f"test Overall  AUC    : ERM {f(result.test_overall_before)} → ROC {f(result.test_overall_after)}")
    print(f"test worst-group AUC : ERM {f(result.test_worst_before)} → ROC {f(result.test_worst_after)}")
    print(f"test AUC gap         : ERM {f(result.test_gap_before)} → ROC {f(result.test_gap_after)}")


# ============================================================
# 操作点版 ROC（评审补强 R2.3）：θ 选择用**操作点** EqOdds gap（阈值相关），
# 而非阈值无关的 worst-group AUC——后者使 reject-option 恒 θ*=0、ROC≡ERM，
# 其子群奇偶目的从未被测。本节让 ROC 在其真正设计目标上被选与被测（零重训、纯后处理）。
# ============================================================
def _per_group_tpr_at_half(
    y_true: np.ndarray, prob: np.ndarray, group_ids: np.ndarray,
) -> dict[int, float | None]:
    """该轴各子群在原生阈值 0.5 上的 TPR（无正样本 → None），供按操作点表现二分 deprived。"""
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(prob, dtype=float) >= 0.5).astype(int)
    out: dict[int, float | None] = {}
    for gid in np.unique(group_ids):
        m = group_ids == gid
        pos = y_true[m] == 1
        out[int(gid)] = float(pred[m][pos].mean()) if pos.sum() > 0 else None
    return out


@dataclass
class ROCOperatingPointResult:
    """
    操作点版 ROC 结果（method="roc"，θ 由 val EqOdds gap 选定；派生自 ERM 预测）。

    Attributes:
        dataset / axis     : 数据集名 / 二分所用敏感轴。
        best_margin        : val EqOdds gap 最小的 θ*（θ=0 即 ERM）。
        deprived_group_ids : 该轴 val TPR 低于中位（弱势）的子群 id。
        val_eo_before/after: val EqOdds gap（ERM θ=0 vs ROC θ*，越小越公平）。
        adjusted_test_scores: [N] ROC 调整后的 test **概率**（∈[0,1]，供操作点度量/落盘）。
    """

    dataset: str
    axis: str
    best_margin: float
    deprived_group_ids: list[int]
    val_eo_before: float | None
    val_eo_after: float | None
    adjusted_test_scores: np.ndarray


def apply_roc_operating_point(
    dataset: str,
    axis: str,
    *,
    y_val_true: np.ndarray,
    y_val_logits: np.ndarray,
    val_attrs: dict[str, np.ndarray],
    y_test_logits: np.ndarray,
    test_attrs: dict[str, np.ndarray],
    margins: "tuple[float, ...]" = DEFAULT_MARGIN_GRID,
    min_class_n: int = 10,
) -> ROCOperatingPointResult:
    """
    R2.3：在指定轴上做 ROC 后处理，但 **θ 选择的目标 = 最小化 val EqOdds gap（原生阈值 0.5）**。

    与 `apply_roc`（目标=val worst-group AUC）的唯一区别就是**选择目标换成操作点度量**，从而
    reject-option 的阈值干预能真正被选中（θ*>0）并在其设计目标上被测。deprived 亦按**操作点表现**
    （val 子群 0.5-TPR 低于中位者）二分，与目标同口径。网格含 0 → val EqOdds gap 恒不劣于 ERM。

    Args:
        dataset      : 数据集名。
        axis         : 二分轴（须在 val/test 属性字典中）。
        y_val_true / y_val_logits / val_attrs: ERM 在 val 上的标签/logits/属性。
        y_test_logits / test_attrs           : ERM 在 test 上的 logits/属性（test 真值在评估侧另取）。
        margins      : θ 网格（须含 0.0）。
        min_class_n  : EqOdds gap 纳入子群的该类最小样本数（同 operating_point_fairness）。

    Returns:
        ROCOperatingPointResult（含调整后的 test 概率）。

    Raises:
        ValueError: axis 不在属性字典中。
    """
    # 延迟导入避免与 operating_point_fairness 的模块级依赖环（后者 import subgroup_auc）
    from src.utils.operating_point_fairness import (
        subgroup_tpr_fpr, equalized_odds_gap,
    )

    if axis not in val_attrs or axis not in test_attrs:
        raise ValueError(f"axis={axis!r} 不在属性字典中（可用：{sorted(val_attrs)}）。")

    y_val_true = np.asarray(y_val_true).astype(int)
    val_prob = logits_to_prob(y_val_logits)
    test_prob = logits_to_prob(y_test_logits)
    val_gids = np.asarray(val_attrs[axis]).astype(int)
    test_gids = np.asarray(test_attrs[axis]).astype(int)

    # deprived：按 val 子群 0.5-TPR 低于中位二分（与 EqOdds 目标同口径）
    per_group_tpr = _per_group_tpr_at_half(y_val_true, val_prob, val_gids)
    val_deprived = binarize_axis_by_performance(val_gids, per_group_tpr)
    per_group_for_test = dict(per_group_tpr)
    for gid in np.unique(test_gids):
        per_group_for_test.setdefault(int(gid), None)
    test_deprived = binarize_axis_by_performance(test_gids, per_group_for_test)
    sorted_gids = np.array(sorted(per_group_tpr))
    dep_flags = binarize_axis_by_performance(sorted_gids, per_group_tpr)
    deprived_ids = [int(g) for g, d in zip(sorted_gids, dep_flags) if d]

    # 目标：最小化 val EqOdds gap（select_margin 取最大 → 传负 gap）
    val_int_attrs = _as_int_attrs(val_attrs)

    def val_objective(adj_prob: np.ndarray) -> float | None:
        rates = subgroup_tpr_fpr(dataset, y_val_true, adj_prob, val_int_attrs, 0.5)
        _tg, _fg, eo = equalized_odds_gap(rates, min_class_n=min_class_n)
        return None if eo is None else -eo

    sel = select_margin(val_prob, val_deprived, val_objective, margins=margins)
    adjusted_test = reject_option_adjust(test_prob, test_deprived, sel.best_margin)

    return ROCOperatingPointResult(
        dataset=dataset, axis=axis, best_margin=sel.best_margin,
        deprived_group_ids=deprived_ids,
        val_eo_before=(None if sel.erm_value is None else -sel.erm_value),
        val_eo_after=-sel.best_value,
        adjusted_test_scores=adjusted_test,
    )


def apply_roc_operating_point_auto(
    dataset: str,
    *,
    y_val_true: np.ndarray,
    y_val_logits: np.ndarray,
    val_attrs: dict[str, np.ndarray],
    y_test_logits: np.ndarray,
    test_attrs: dict[str, np.ndarray],
    margins: "tuple[float, ...]" = DEFAULT_MARGIN_GRID,
    min_class_n: int = 10,
) -> ROCOperatingPointResult:
    """
    自动选轴的操作点版 ROC：每个敏感轴各跑一次，选 **val EqOdds gap 最小（after）** 者。

    平手偏好 `DATASET_PRIMARY_AXIS` 主轴，再偏好更小 θ（更接近 ERM）。
    """
    primary = DATASET_PRIMARY_AXIS.get(dataset)
    results = [
        apply_roc_operating_point(
            dataset, axis,
            y_val_true=y_val_true, y_val_logits=y_val_logits, val_attrs=val_attrs,
            y_test_logits=y_test_logits, test_attrs=test_attrs,
            margins=margins, min_class_n=min_class_n,
        )
        for axis in val_attrs
    ]

    def key(r: ROCOperatingPointResult) -> tuple:
        eo = np.inf if r.val_eo_after is None else r.val_eo_after
        return (-eo, r.axis == primary, -r.best_margin)   # 越小 eo 越好 → 取最大 -eo

    return max(results, key=key)


# ============================================================
# self-test：临界区变换 + 逐轴二分 + margin 选择（含不劣于 ERM）
# ============================================================
def _selftest() -> None:
    """验证 reject-option 变换语义、逐轴二分、margin 选择恒不劣于 ERM、退化与校验。"""
    # --- reject_option_adjust：区内翻转、区外不动、margin=0 不变 ---
    prob = np.array([0.10, 0.45, 0.48, 0.52, 0.55, 0.90])
    deprived = np.array([True, True, True, False, False, False])
    adj = reject_option_adjust(prob, deprived, margin=0.1)
    # 区内 = |p-0.5|<=0.1 → 索引 1,2,3,4；区外 0(0.10),5(0.90) 不动
    assert adj[0] == 0.10 and adj[5] == 0.90
    # deprived 区内(1:0.45,2:0.48) → max(p,1-p) ≥0.5
    assert abs(adj[1] - 0.55) < 1e-12 and abs(adj[2] - 0.52) < 1e-12
    # privileged 区内(3:0.52,4:0.55) → min(p,1-p) ≤0.5
    assert abs(adj[3] - 0.48) < 1e-12 and abs(adj[4] - 0.45) < 1e-12
    # margin=0 → 完全不变
    assert np.allclose(reject_option_adjust(prob, deprived, 0.0), prob)
    # margin=0.5 → 全部入区
    adj_full = reject_option_adjust(prob, deprived, 0.5)
    assert np.all(adj_full[deprived] >= 0.5 - 1e-12) and np.all(adj_full[~deprived] <= 0.5 + 1e-12)
    # 校验
    for bad in [lambda: reject_option_adjust(prob, deprived, 0.7),
                lambda: reject_option_adjust(prob, deprived[:-1], 0.1)]:
        try:
            bad(); raise AssertionError("应报错")
        except ValueError:
            pass

    # --- binarize_axis_by_performance：低于中位 AUC = deprived，None 并入 deprived ---
    gids = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    scores = {0: 0.90, 1: 0.70, 2: 0.60, 3: None}   # 中位(0.9,0.7,0.6)=0.7 → <0.7 的 2 及 None 的 3 = deprived
    mask = binarize_axis_by_performance(gids, scores)
    assert mask.tolist() == [False, False, True, True, False, False, True, True], mask.tolist()
    # 二值轴：较差者 deprived
    m2 = binarize_axis_by_performance(np.array([0, 1]), {0: 0.8, 1: 0.6})
    assert m2.tolist() == [False, True]   # 中位=0.7，0.6<0.7 → id1 deprived；0.8 不<0.7
    # 未登记子群报错
    try:
        binarize_axis_by_performance(np.array([0, 5]), {0: 0.8}); raise AssertionError("应报错")
    except ValueError:
        pass

    # --- select_margin：构造「某 θ 提升目标」场景，验证选中且不劣于 ERM ---
    rng = np.random.default_rng(0)
    n = 400
    p = rng.uniform(0, 1, n)
    dep = rng.random(n) < 0.5
    # 目标：deprived 样本推向正侧的平均分（θ 越大越高）→ 应选 best_margin=0.5
    def obj_increasing(a):
        return float(a[dep].mean())
    sel = select_margin(p, dep, obj_increasing)
    assert sel.best_margin == 0.5, sel.best_margin
    assert sel.best_value >= (sel.erm_value or -np.inf)   # 不劣于 ERM

    # 目标与 θ 无关（常数）→ 平手取最小 θ=0（最保守）
    sel_flat = select_margin(p, dep, lambda a: 1.0)
    assert sel_flat.best_margin == 0.0, sel_flat.best_margin

    # 目标恒 None → 报错
    try:
        select_margin(p, dep, lambda a: None); raise AssertionError("全 None 应报错")
    except ValueError:
        pass

    # --- logits_to_prob：sigmoid 正确、数值稳定 ---
    assert abs(logits_to_prob(np.array([0.0]))[0] - 0.5) < 1e-12
    assert logits_to_prob(np.array([1e3]))[0] == 1.0 and logits_to_prob(np.array([-1e3]))[0] == 0.0

    print("roc A4.1 self-test 全部通过 ✓（区内翻转/区外不动/margin极值 + 逐轴二分/None并入 + margin选择不劣ERM + sigmoid稳定）")


def _selftest_a42() -> None:
    """验证 A4.2：apply_roc 端到端（val 定 deprived/选 θ、test 评估、不劣 ERM）、自动选轴、报告。"""
    rng = np.random.default_rng(21)

    def make(n, seed):
        g = np.random.default_rng(seed)
        y = g.integers(0, 2, n)
        sex = g.integers(0, 2, n)
        age = g.integers(0, 2, n)          # papila：age_group 二值
        # 让 age==1 子群信噪比更低（表现更差 → 应被判 deprived）
        snr = np.where(age == 1, 0.3, 1.0)
        logits = snr * (2 * y - 1) + g.standard_normal(n)
        return y, logits, {"sex": sex, "age": age}

    yv, lv, av = make(1200, 1)
    yt, lt, at = make(1200, 2)

    res = apply_roc("papila", "age",
                    y_val_true=yv, y_val_logits=lv, val_attrs=av,
                    y_test_true=yt, y_test_logits=lt, test_attrs=at)
    # 结构完整
    assert res.axis == "age" and 0.0 <= res.best_margin <= 0.5
    assert isinstance(res.adjusted_test_scores, np.ndarray) and len(res.adjusted_test_scores) == 1200
    # 表现差的 age==1 应入 deprived
    assert 1 in res.deprived_group_ids, (res.deprived_group_ids, res.per_group_val_auc)
    # val worst-group：ROC 不劣于 ERM（margin 网格含 0 保证）
    assert res.val_worst_after is not None and res.val_worst_before is not None
    assert res.val_worst_after >= res.val_worst_before - 1e-9, (res.val_worst_before, res.val_worst_after)
    # margin=0 时 test 度量应与 ERM 完全一致（退化健全性）
    res0 = apply_roc("papila", "age",
                     y_val_true=yv, y_val_logits=lv, val_attrs=av,
                     y_test_true=yt, y_test_logits=lt, test_attrs=at, margins=(0.0,))
    assert abs(res0.test_overall_after - res0.test_overall_before) < 1e-12
    assert abs((res0.test_worst_after or 0) - (res0.test_worst_before or 0)) < 1e-12

    # 自动选轴：在 {sex, age} 上各跑，返回一个合法轴
    auto = apply_roc_auto("papila",
                          y_val_true=yv, y_val_logits=lv, val_attrs=av,
                          y_test_true=yt, y_test_logits=lt, test_attrs=at)
    assert auto.axis in ("sex", "age")
    assert auto.val_worst_after is not None

    # 报告不抛异常
    roc_fairness_report(res)

    # 校验：axis 不在属性字典
    try:
        apply_roc("papila", "race",
                  y_val_true=yv, y_val_logits=lv, val_attrs=av,
                  y_test_true=yt, y_test_logits=lt, test_attrs=at)
        raise AssertionError("未知 axis 应报错")
    except ValueError:
        pass

    print("roc A4.2 self-test 全部通过 ✓（端到端不劣ERM / deprived定位 / margin=0退化 / 自动选轴 / 报告 / 校验）")


if __name__ == "__main__":
    _selftest()
    _selftest_a42()
