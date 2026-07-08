"""
数据集内显著性检验（比较协议 A2）
================================
比较协议 §5「统计检验（仅数据集内）」：≥5 seed 的 worst-group AUC / AUC gap 报**均值 ± 95%CI**；
方法 vs 基线做**配对 bootstrap CI / DeLong**，证明提升非种子噪声。本模块提供三块互相独立、
可单测的统计原语，供 D 阶段汇总消费：

  A2.1 `mean_ci`               —— 多-seed 标量指标的均值 ± 95%CI（已实现）。
  A2.2 `paired_bootstrap_diff` —— 方法 vs 基线在同一 test 集上的配对 bootstrap 差值 CI（已实现）。
  A2.3 `delong_test`           —— 两条相关 ROC 的 **Overall AUC** 差异 DeLong 检验（已实现）。

**A2.1 设计**：确认阶段每个方法有 5 个 seed，每 seed 在 test 集上得到一个 worst-group AUC 与一个
AUC gap（由该 seed 的完整子群向量派生，见 `worst_and_gap`）。把这 5 个标量视作 i.i.d. 样本，
用 **Student-t 分布**（df=n-1）算双侧 95% CI——n=5 时 t 分布比正态更诚实地反映小样本不确定度
（t_{.975,4}=2.776 vs z=1.96），避免低估 CI 宽度。这是「数据集内、跨 seed」的不确定度，
与 A2.2/A2.3「数据集内、跨 test 样本」的不确定度正交，二者在报告中并列。

**A2.2 设计**：配对（paired）指同一次 bootstrap 迭代对 test 样本重采样出的**同一组索引**同时施加
于方法与基线的预测——消除 test 集本身的样本构成波动，只留下「两方法在同一批样本上的差异」这一项，
比非配对更紧、更对，正是「提升是否非噪声」要问的。原语 `paired_bootstrap_diff` 与具体度量**解耦**：
它只认「吃索引、吐标量」的两个闭包 `metric(idx)->float|None`，故 worst-group AUC / AUC gap 这类
**重采样后子群构成会变**（小子群可能整组消失或单类化）的度量也能正确处理——由 `make_subgroup_metric`
把 (dataset, y_true, y_score, attrs) 包成这样的闭包。要证「提升非**种子**噪声」，用多-seed 变体
`make_subgroup_metric_multiseed`：每次迭代在重采样索引之外再随机抽一个 seed 的预测，从而**样本 +
种子两层不确定度联合**进入 bootstrap 分布。

**A2.3 设计**：DeLong 检验是**两条相关 ROC 曲线**（同一 test 集、样本配对）AUC 差异的**解析**检验，
无需重采样。用 Sun & Xu (2014) 的 **fast DeLong**（O(N log N)，midrank 实现）算两 AUC 的结构分量
与协方差矩阵，得差值的标准误 → z 统计量 → 双侧正态 p。它专治 **Overall AUC 差异**（worst-group AUC
是「多子群 AUC 取 min」的非光滑泛函，无 DeLong 解析方差，仍归 A2.2 bootstrap）。与 A2.2 互补：
DeLong 快且对 AUC 差异是渐近精确的解析解，bootstrap 通用于任意度量（worst-group / gap）。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from src.training.harness.subgroup_auc import subgroup_auc_vector, vector_worst_case

# 吃「重采样索引」、吐「该子集上的标量度量或 None」的闭包类型（A2.2 与度量解耦的核心）
MetricFn = Callable[[np.ndarray], "float | None"]


@dataclass(frozen=True)
class MeanCI:
    """
    一个标量指标在多-seed 上的均值与置信区间。

    Attributes:
        mean      : 样本均值。
        ci_low    : 置信区间下界。
        ci_high   : 置信区间上界。
        half_width: 半宽（= t * sem = (ci_high - ci_low)/2），报告 `mean ± half_width` 用。
        std       : 样本标准差（ddof=1，无偏）。
        sem       : 均值标准误（std / sqrt(n)）。
        n         : 样本量（有效 seed 数）。
        confidence: 置信水平（如 0.95）。
    """

    mean: float
    ci_low: float
    ci_high: float
    half_width: float
    std: float
    sem: float
    n: int
    confidence: float

    def format(self, digits: int = 4) -> str:
        """人类可读 `mean ± half_width`（保留 digits 位）。"""
        return f"{self.mean:.{digits}f} ± {self.half_width:.{digits}f}"


def mean_ci(values: "list[float] | np.ndarray", confidence: float = 0.95) -> MeanCI:
    """
    A2.1：多-seed 标量的均值 ± 95%CI（Student-t，df=n-1，双侧）。

    用 t 分布而非正态：确认阶段 n=5 属小样本，t 分布对小样本不确定度更诚实（不低估 CI 宽度）。
    n==1 时无法估计离散度，CI 退化为点（half_width=0）并给出 std=nan 的诚实标注。

    Args:
        values    : 各 seed 的标量指标值（如 5 个 worst-group AUC）。须至少 1 个有限值。
        confidence: 置信水平，默认 0.95。

    Returns:
        MeanCI。

    Raises:
        ValueError: values 为空、含非有限值、或 confidence 不在 (0,1)。
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence 须在 (0,1)，收到 {confidence}")
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("mean_ci 收到空 values。")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"mean_ci 的 values 含非有限值：{arr!r}（请先剔除 None/NaN seed）。")

    n = int(arr.size)
    mean = float(arr.mean())
    if n == 1:
        # 单样本无法估离散度：CI 退化为点，std/sem 记 nan（诚实标注，不假装有区间）
        return MeanCI(mean=mean, ci_low=mean, ci_high=mean, half_width=0.0,
                      std=float("nan"), sem=float("nan"), n=1, confidence=confidence)

    std = float(arr.std(ddof=1))               # 无偏样本标准差
    sem = std / np.sqrt(n)
    from scipy.stats import t as student_t
    t_crit = float(student_t.ppf(0.5 + confidence / 2.0, df=n - 1))
    half = t_crit * sem
    return MeanCI(mean=mean, ci_low=mean - half, ci_high=mean + half, half_width=half,
                  std=std, sem=sem, n=n, confidence=confidence)


def worst_and_gap(
    vec: "OrderedDict[str, tuple[float | None, int]]",
) -> tuple[float | None, float | None]:
    """
    从一个 seed 的完整子群 AUC 向量派生 (worst-group AUC, AUC gap)。

    worst-group = min 非 None AUC（复用 subgroup_auc.vector_worst_case，口径一致）；
    gap = max − min（有效子群 < 2 时为 None，与各 fairness 模块 auc_gap 一致）。

    Args:
        vec: subgroup_auc_vector 的返回（OrderedDict[键 -> (AUC|None, n)]）。

    Returns:
        (worst_group_auc 或 None, auc_gap 或 None)。
    """
    aucs = [auc for auc, _n in vec.values() if auc is not None]
    if not aucs:
        return None, None
    worst = vector_worst_case(vec)
    gap = (max(aucs) - min(aucs)) if len(aucs) >= 2 else None
    return worst, gap


def aggregate_seed_scalars(
    per_seed: "list[float | None]", confidence: float = 0.95,
) -> MeanCI:
    """
    把「各 seed 一个标量」（worst-group AUC 或 AUC gap）聚合为均值 ± 95%CI。

    None（该 seed 无有效子群 / gap 不可算）会被剔除后再聚合；全为 None 则报错（无可聚合样本）。

    Args:
        per_seed  : 各 seed 的标量（可含 None）。
        confidence: 置信水平。

    Returns:
        MeanCI（基于非 None 的 seed）。

    Raises:
        ValueError: 全部为 None。
    """
    vals = [v for v in per_seed if v is not None]
    if not vals:
        raise ValueError("aggregate_seed_scalars：所有 seed 的指标均为 None，无法聚合。")
    return mean_ci(vals, confidence=confidence)


# ============================================================
# A2.2：配对 bootstrap 差值 CI（方法 vs 基线）
# ============================================================
@dataclass(frozen=True)
class BootstrapDiff:
    """
    配对 bootstrap 的方法−基线差值分布摘要。

    Attributes:
        point       : 全样本点估计（metric_a − metric_b，未重采样）。
        ci_low      : 差值分布的下分位（percentile CI 下界）。
        ci_high     : 差值分布的上分位（percentile CI 上界）。
        p_value     : 双侧 bootstrap p 值（差值分布越过 0 的比例，见 `paired_bootstrap_diff`）。
        mean_diff   : bootstrap 差值均值（诊断用，理想接近 point）。
        n_boot      : 请求的 bootstrap 次数。
        n_valid     : 两方法度量都非 None 的有效迭代数（无效迭代已丢弃）。
        confidence  : 置信水平（如 0.95）。
        significant : CI 是否不含 0（在该置信水平下差异显著）。
    """

    point: float
    ci_low: float
    ci_high: float
    p_value: float
    mean_diff: float
    n_boot: int
    n_valid: int
    confidence: float
    significant: bool

    def format(self, digits: int = 4) -> str:
        """人类可读：`Δ=point [ci_low, ci_high] p=... (显著?)`。"""
        mark = "显著" if self.significant else "n.s."
        return (f"Δ={self.point:.{digits}f} "
                f"[{self.ci_low:.{digits}f}, {self.ci_high:.{digits}f}] "
                f"p={self.p_value:.4f} ({mark})")


def paired_bootstrap_diff(
    n: int,
    metric_a: MetricFn,
    metric_b: MetricFn,
    *,
    n_boot: int = 2000,
    confidence: float = 0.95,
    rng: "np.random.Generator | int | None" = None,
    min_valid_frac: float = 0.5,
) -> BootstrapDiff:
    """
    A2.2：对 n 个 test 样本做配对 case-resampling bootstrap，估计 metric_a − metric_b 的差值 CI。

    每次迭代抽一组**同一**放回索引 idx（长度 n），同时喂给两方法的度量闭包（配对，消掉样本构成
    波动）；差值 = metric_a(idx) − metric_b(idx)。两度量与本原语解耦——只要它们「吃 idx、吐标量或
    None」（子群重采样后为空/单类时吐 None）。点估计用全样本 idx=arange(n)。

    p 值：双侧 bootstrap 估计 `2·min(P(Δ≤0), P(Δ≥0))`（截断到 [0,1]），衡量差值分布落在 0 另一侧
    的证据强度。CI 用 percentile 法（分布的 (1±conf)/2 分位）。

    Args:
        n             : test 样本数。
        metric_a/b    : 度量闭包（method A / baseline B）；`metric(idx)->float|None`。
        n_boot        : bootstrap 迭代次数（默认 2000）。
        confidence    : 置信水平。
        rng           : np.random.Generator / 整数种子 / None（可复现）。
        min_valid_frac: 有效迭代占比下限；低于此则报错（度量太常 None，CI 不可信）。

    Returns:
        BootstrapDiff。

    Raises:
        ValueError: n<2、点估计不可算（全样本任一度量 None）、或有效迭代不足。
    """
    if n < 2:
        raise ValueError(f"paired_bootstrap_diff 需 n≥2，收到 n={n}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence 须在 (0,1)，收到 {confidence}")
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    full = np.arange(n)
    pa, pb = metric_a(full), metric_b(full)
    if pa is None or pb is None:
        raise ValueError("全样本点估计不可算（metric_a 或 metric_b 在全样本上为 None）。")
    point = float(pa) - float(pb)

    diffs: list[float] = []
    for _ in range(n_boot):
        idx = generator.integers(0, n, size=n)
        va, vb = metric_a(idx), metric_b(idx)
        if va is None or vb is None:
            continue
        diffs.append(float(va) - float(vb))

    n_valid = len(diffs)
    if n_valid < max(2, int(min_valid_frac * n_boot)):
        raise ValueError(
            f"有效 bootstrap 迭代仅 {n_valid}/{n_boot}（< {min_valid_frac:.0%}）："
            "度量在重采样下过于频繁地 None，CI 不可信。可能子群过小。"
        )

    darr = np.asarray(diffs)
    lo_q = (1.0 - confidence) / 2.0
    ci_low = float(np.quantile(darr, lo_q))
    ci_high = float(np.quantile(darr, 1.0 - lo_q))
    # 双侧 bootstrap p：差值分布越过 0 的比例（两侧取小者 ×2）
    frac_le = float(np.mean(darr <= 0.0))
    frac_ge = float(np.mean(darr >= 0.0))
    p_value = float(min(1.0, 2.0 * min(frac_le, frac_ge)))
    significant = not (ci_low <= 0.0 <= ci_high)

    return BootstrapDiff(
        point=point, ci_low=ci_low, ci_high=ci_high, p_value=p_value,
        mean_diff=float(darr.mean()), n_boot=n_boot, n_valid=n_valid,
        confidence=confidence, significant=significant,
    )


def make_subgroup_metric(
    dataset: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    attrs: dict[str, np.ndarray],
    which: Literal["worst", "gap"],
) -> MetricFn:
    """
    把单 seed 的 test 预测包成「吃索引、吐 worst-group AUC 或 AUC gap」的度量闭包，供 A2.2。

    闭包对传入的 idx 同时切片 y_true / y_score / 各属性，重算子群向量并派生标量——因此 bootstrap
    重采样时子群构成随之变化（正确反映不确定度）。

    Args:
        dataset : {"mimic","chexpert","ham10000","fitzpatrick","papila"}。
        y_true  : [N] 0/1 标签。
        y_score : [N] 该 seed 的模型输出。
        attrs   : 该数据集所需属性 {名 -> [N] 数组}（键须匹配 subgroup_auc_vector：sex/race/age/skin）。
        which   : "worst"（worst-group AUC）或 "gap"（AUC gap）。

    Returns:
        MetricFn：`metric(idx) -> float | None`。
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    attrs = {k: np.asarray(v) for k, v in attrs.items()}
    sel = 0 if which == "worst" else 1

    def metric(idx: np.ndarray) -> float | None:
        vec = subgroup_auc_vector(
            dataset, y_true[idx], y_score[idx],
            **{k: v[idx] for k, v in attrs.items()},
        )
        return worst_and_gap(vec)[sel]

    return metric


def make_subgroup_metric_multiseed(
    dataset: str,
    y_true: np.ndarray,
    y_scores: "list[np.ndarray]",
    attrs: dict[str, np.ndarray],
    which: Literal["worst", "gap"],
    rng: "np.random.Generator",
) -> MetricFn:
    """
    多-seed 变体：闭包每次调用先用 rng 随机抽一个 seed 的预测，再按 idx 切片算度量。

    与 `paired_bootstrap_diff` 组合后，每次迭代 = 重采样样本 × 随机抽 seed，使**样本 + 种子两层
    不确定度联合**进入 bootstrap 分布——直接回答「提升是否非**种子**噪声」。两方法各自传入自己的
    多-seed 预测列表与**同一** rng（seed 抽取相互独立，符合两方法是独立训练运行的事实）。

    Args:
        dataset : 数据集名。
        y_true  : [N] 标签（各 seed 共享同一 test 集，故标签唯一）。
        y_scores: 各 seed 的 [N] 预测列表（长度 = seed 数，须 ≥1）。
        attrs   : 属性字典（同 `make_subgroup_metric`）。
        which   : "worst" / "gap"。
        rng     : np.random.Generator（供 seed 抽取；与 bootstrap 的 rng 可同一个）。

    Returns:
        MetricFn。

    Raises:
        ValueError: y_scores 为空。
    """
    if not y_scores:
        raise ValueError("make_subgroup_metric_multiseed：y_scores 不能为空。")
    y_true = np.asarray(y_true)
    scores = [np.asarray(s) for s in y_scores]
    attrs = {k: np.asarray(v) for k, v in attrs.items()}
    sel = 0 if which == "worst" else 1
    n_seed = len(scores)

    def metric(idx: np.ndarray) -> float | None:
        s = scores[int(rng.integers(0, n_seed))]
        vec = subgroup_auc_vector(
            dataset, y_true[idx], s[idx],
            **{k: v[idx] for k, v in attrs.items()},
        )
        return worst_and_gap(vec)[sel]

    return metric


# ============================================================
# A2.3：DeLong 检验（两条相关 ROC 的 Overall AUC 差异）
# ============================================================
@dataclass(frozen=True)
class DeLongResult:
    """
    DeLong 检验结果（method A vs baseline B 的 Overall AUC 差异）。

    Attributes:
        auc_a, auc_b: 两方法的 Overall AUC（DeLong 内部算，与 sklearn 数值一致）。
        diff        : auc_a − auc_b。
        se_diff     : 差值标准误（sqrt(Var_a + Var_b − 2Cov)）。
        z           : z 统计量（diff / se_diff）。
        p_value     : 双侧正态 p 值。
        n_pos, n_neg: 正/负样本数。
        significant : 在 alpha 下是否显著（|z| 超临界值 ⟺ p<alpha）。
        alpha       : 显著性水平（默认 0.05）。
    """

    auc_a: float
    auc_b: float
    diff: float
    se_diff: float
    z: float
    p_value: float
    n_pos: int
    n_neg: int
    significant: bool
    alpha: float

    def format(self, digits: int = 4) -> str:
        """人类可读：`AUC A/B, Δ, z, p (显著?)`。"""
        mark = "显著" if self.significant else "n.s."
        return (f"AUC {self.auc_a:.{digits}f} vs {self.auc_b:.{digits}f}  "
                f"Δ={self.diff:.{digits}f}  z={self.z:.3f}  p={self.p_value:.4g} ({mark})")


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """
    计算 midrank（并列取平均秩，1-based），fast DeLong 的核心子程序。

    Args:
        x: [N] 一维分数数组。

    Returns:
        [N] midrank 数组（与 x 同序）。
    """
    order = np.argsort(x)
    x_sorted = x[order]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1.0  # 并列区间 [i,j) 的平均秩（1-based）
        i = j
    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def _fast_delong(
    predictions_sorted: np.ndarray, n_pos: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sun & Xu (2014) fast DeLong：给定「正样本在前」排列的 k 个预测，算各 AUC 与协方差矩阵。

    Args:
        predictions_sorted: [k, N] 数组，列已按标签重排（前 n_pos 列为正样本）。
        n_pos             : 正样本数 m。

    Returns:
        (aucs[k], delong_cov[k,k])：各预测的 AUC 与 DeLong 协方差矩阵。
    """
    m = n_pos
    n = predictions_sorted.shape[1] - m
    k = predictions_sorted.shape[0]
    positive = predictions_sorted[:, :m]
    negative = predictions_sorted[:, m:]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n          # [k, m] 正样本结构分量
    v10 = 1.0 - (tz[:, m:] - ty) / m    # [k, n] 负样本结构分量
    # np.cov 需 k≥2 行；k=2（method vs baseline）恒满足。ddof 默认=1（无偏）。
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, np.atleast_2d(delong_cov)


def delong_test(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    *,
    alpha: float = 0.05,
) -> DeLongResult:
    """
    A2.3：DeLong 检验，比较同一 test 集上两方法的 **Overall AUC** 差异（method A − baseline B）。

    解析法（无重采样）：fast DeLong 算两 AUC 的结构协方差 → 差值标准误 → z → 双侧正态 p。
    要求正负样本各 ≥1。两分数完全相同（或差值方差数值为 0）时：diff=0、z=0、p=1（无差异证据）。

    Args:
        y_true : [N] 0/1 标签。
        score_a: [N] method A 输出。
        score_b: [N] baseline B 输出（与 score_a 样本一一对应，配对）。
        alpha  : 显著性水平（默认 0.05）。

    Returns:
        DeLongResult。

    Raises:
        ValueError: 长度不一致、标签非二类、或缺正/负样本。
    """
    y_true = np.asarray(y_true).astype(int)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    if not (len(y_true) == len(score_a) == len(score_b)):
        raise ValueError("y_true / score_a / score_b 长度须一致。")
    uniq = np.unique(y_true)
    if not np.array_equal(uniq, [0, 1]):
        raise ValueError(f"y_true 须恰含 {{0,1}} 两类，实得 {uniq}。")

    # 正样本在前重排（DeLong 约定）
    order = np.argsort(-y_true, kind="mergesort")  # 稳定排序：label=1 在前
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    predictions = np.vstack((score_a, score_b))[:, order]

    aucs, cov = _fast_delong(predictions, n_pos)
    auc_a, auc_b = float(aucs[0]), float(aucs[1])
    diff = auc_a - auc_b
    # Var(diff) = Var_a + Var_b − 2Cov（对比向量 [1,-1]）
    var_diff = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])

    if var_diff <= 0.0:
        # 数值退化（如两分数同秩结构）：无差异证据
        z, p_value = 0.0, 1.0
        se_diff = 0.0
    else:
        se_diff = float(np.sqrt(var_diff))
        z = diff / se_diff
        from scipy.stats import norm
        p_value = float(2.0 * norm.sf(abs(z)))

    return DeLongResult(
        auc_a=auc_a, auc_b=auc_b, diff=diff, se_diff=se_diff, z=z, p_value=p_value,
        n_pos=n_pos, n_neg=n_neg, significant=(p_value < alpha), alpha=alpha,
    )


# ============================================================
# self-test：A2.1 均值/CI 正确性 + 与手算/scipy 交叉验证
# ============================================================
def _selftest() -> None:
    """验证 mean_ci 的 t-CI 数值、退化情形、worst_and_gap 派生、聚合剔 None。"""
    from scipy.stats import t as student_t

    # --- 已知数值：5 个值，与手算 t-CI 对齐 ---
    vals = [0.80, 0.82, 0.78, 0.81, 0.79]
    m = mean_ci(vals, 0.95)
    arr = np.asarray(vals)
    exp_mean = arr.mean()
    exp_std = arr.std(ddof=1)
    exp_sem = exp_std / np.sqrt(5)
    exp_half = student_t.ppf(0.975, df=4) * exp_sem
    assert abs(m.mean - exp_mean) < 1e-12, m.mean
    assert abs(m.std - exp_std) < 1e-12, m.std
    assert abs(m.half_width - exp_half) < 1e-12, m.half_width
    assert abs(m.ci_low - (exp_mean - exp_half)) < 1e-12
    assert abs(m.ci_high - (exp_mean + exp_half)) < 1e-12
    assert m.n == 5 and m.confidence == 0.95
    # t_{.975,4}=2.776：half ≈ 2.776 * sem
    assert abs(student_t.ppf(0.975, df=4) - 2.7764451) < 1e-5

    # --- 零方差：CI 收缩为点 ---
    z = mean_ci([0.9, 0.9, 0.9], 0.95)
    assert z.half_width == 0.0 and z.ci_low == z.ci_high == 0.9

    # --- n==1：退化为点、std/sem = nan ---
    one = mean_ci([0.75])
    assert one.mean == 0.75 and one.half_width == 0.0
    assert np.isnan(one.std) and np.isnan(one.sem) and one.n == 1

    # --- 置信水平更高 → CI 更宽 ---
    m90 = mean_ci(vals, 0.90)
    m99 = mean_ci(vals, 0.99)
    assert m90.half_width < m.half_width < m99.half_width

    # --- 输入校验 ---
    for bad, err in [([], ValueError), ([0.1, float("nan")], ValueError)]:
        try:
            mean_ci(bad); raise AssertionError("应报错")
        except ValueError:
            pass
    try:
        mean_ci(vals, 1.5); raise AssertionError("confidence 越界应报错")
    except ValueError:
        pass

    # --- worst_and_gap：从子群向量派生 ---
    vec: "OrderedDict[str, tuple[float | None, int]]" = OrderedDict([
        ("g1", (0.90, 100)), ("g2", (0.70, 50)), ("g3", (0.85, 80)),
    ])
    worst, gap = worst_and_gap(vec)
    assert abs(worst - 0.70) < 1e-12 and abs(gap - 0.20) < 1e-12
    # 含 None 子群：忽略 None
    vec2: "OrderedDict[str, tuple[float | None, int]]" = OrderedDict([
        ("g1", (0.90, 100)), ("g2", (None, 0)), ("g3", (0.88, 80)),
    ])
    w2, gp2 = worst_and_gap(vec2)
    assert abs(w2 - 0.88) < 1e-12 and abs(gp2 - 0.02) < 1e-12
    # 单一有效子群：gap=None，worst 仍有值
    vec3: "OrderedDict[str, tuple[float | None, int]]" = OrderedDict([
        ("g1", (0.90, 100)), ("g2", (None, 0)),
    ])
    w3, gp3 = worst_and_gap(vec3)
    assert abs(w3 - 0.90) < 1e-12 and gp3 is None
    # 全 None
    vec4: "OrderedDict[str, tuple[float | None, int]]" = OrderedDict([("g1", (None, 0))])
    assert worst_and_gap(vec4) == (None, None)

    # --- aggregate_seed_scalars：剔 None 后聚合 ---
    agg = aggregate_seed_scalars([0.80, None, 0.82, 0.78, None])
    assert agg.n == 3 and abs(agg.mean - np.mean([0.80, 0.82, 0.78])) < 1e-12
    try:
        aggregate_seed_scalars([None, None]); raise AssertionError("全 None 应报错")
    except ValueError:
        pass

    print("significance A2.1 self-test 全部通过 ✓（t-CI 数值/退化/校验 + worst_and_gap + 聚合剔 None）")


def _selftest_a22() -> None:
    """验证 A2.2 配对 bootstrap：真差异 CI 排除 0、同预测 CI 含 0、配对更紧、多-seed、子群度量、校验。"""
    rng = np.random.default_rng(7)
    n = 2000
    y = rng.integers(0, 2, n)
    # A 比 B 信噪比高 → A 的 AUC 系统性更高
    score_a = 1.2 * y + rng.standard_normal(n)
    score_b = 0.4 * y + rng.standard_normal(n)

    from sklearn.metrics import roc_auc_score

    def auc_metric(scores: np.ndarray) -> MetricFn:
        def m(idx: np.ndarray) -> float | None:
            yt = y[idx]
            if len(np.unique(yt)) < 2:
                return None
            return float(roc_auc_score(yt, scores[idx]))
        return m

    # --- 真差异：A−B 的 Δ 显著 > 0，CI 排除 0，p 小 ---
    res = paired_bootstrap_diff(n, auc_metric(score_a), auc_metric(score_b),
                                n_boot=1000, rng=42)
    assert res.point > 0.1, res.point
    assert res.ci_low > 0.0 and res.significant, res.format()
    assert res.p_value < 0.05, res.p_value

    # --- 零差异：同一预测两侧，Δ≈0，CI 含 0，不显著 ---
    res0 = paired_bootstrap_diff(n, auc_metric(score_a), auc_metric(score_a),
                                 n_boot=1000, rng=42)
    assert abs(res0.point) < 1e-9 and not res0.significant
    assert res0.ci_low <= 0.0 <= res0.ci_high

    # --- 配对 vs 非配对：配对 CI 更紧（消掉共同样本波动）---
    # 构造强相关两方法（B = A 加小噪声），配对差值方差应远小于独立各自方差之和
    score_b2 = score_a + 0.05 * rng.standard_normal(n)
    paired = paired_bootstrap_diff(n, auc_metric(score_a), auc_metric(score_b2),
                                   n_boot=1000, rng=1)
    paired_width = paired.ci_high - paired.ci_low
    # 非配对：各自独立重采样索引（打断配对）
    g = np.random.default_rng(1)
    ma, mb = auc_metric(score_a), auc_metric(score_b2)
    unp_diffs = []
    for _ in range(1000):
        ia = g.integers(0, n, n); ib = g.integers(0, n, n)
        va, vb = ma(ia), mb(ib)
        if va is not None and vb is not None:
            unp_diffs.append(va - vb)
    unp_width = np.quantile(unp_diffs, 0.975) - np.quantile(unp_diffs, 0.025)
    assert paired_width < unp_width, (paired_width, unp_width)

    # --- 子群度量闭包（worst-group / gap）在 papila schema 下可跑 ---
    sex = rng.integers(0, 2, n); age = rng.integers(0, 2, n)
    attrs = {"sex": sex, "age": age}
    m_worst_a = make_subgroup_metric("papila", y, score_a, attrs, "worst")
    m_worst_b = make_subgroup_metric("papila", y, score_b, attrs, "worst")
    rw = paired_bootstrap_diff(n, m_worst_a, m_worst_b, n_boot=500, rng=3)
    assert rw.n_valid > 0 and rw.point > 0.0  # A 更强 → worst-group 更高
    m_gap_a = make_subgroup_metric("papila", y, score_a, attrs, "gap")
    _ = paired_bootstrap_diff(n, m_gap_a, m_gap_a, n_boot=300, rng=3)  # 同度量 Δ=0

    # --- 多-seed 变体：抽 seed 不报错，Δ 仍 > 0 ---
    seeds_a = [score_a + 0.1 * rng.standard_normal(n) for _ in range(5)]
    seeds_b = [score_b + 0.1 * rng.standard_normal(n) for _ in range(5)]
    ms_rng = np.random.default_rng(11)
    ma_ms = make_subgroup_metric_multiseed("papila", y, seeds_a, attrs, "worst", ms_rng)
    mb_ms = make_subgroup_metric_multiseed("papila", y, seeds_b, attrs, "worst", ms_rng)
    rms = paired_bootstrap_diff(n, ma_ms, mb_ms, n_boot=500, rng=ms_rng)
    assert rms.n_valid > 0 and rms.point is not None

    # --- 输入校验 ---
    for bad_n in (1, 0):
        try:
            paired_bootstrap_diff(bad_n, auc_metric(score_a), auc_metric(score_b))
            raise AssertionError("n<2 应报错")
        except ValueError:
            pass
    # 度量恒 None → 有效迭代不足报错
    try:
        paired_bootstrap_diff(n, lambda idx: None, lambda idx: None, n_boot=100)
        raise AssertionError("全 None 应报错")
    except ValueError:
        pass

    print("significance A2.2 self-test 全部通过 ✓（真差异CI排0 / 零差异CI含0 / 配对更紧 / 子群度量 / 多-seed / 校验）")


def _selftest_a23() -> None:
    """验证 A2.3 DeLong：AUC 与 sklearn 一致、真差异显著、同分数 p=1、SE 与 bootstrap 交叉验证、校验。"""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(13)
    n = 3000
    y = rng.integers(0, 2, n)
    score_a = 1.1 * y + rng.standard_normal(n)   # 强
    score_b = 0.4 * y + rng.standard_normal(n)   # 弱

    res = delong_test(y, score_a, score_b)
    # AUC 与 sklearn 逐位一致（DeLong 的 AUC 公式正确性）
    assert abs(res.auc_a - roc_auc_score(y, score_a)) < 1e-9, (res.auc_a, roc_auc_score(y, score_a))
    assert abs(res.auc_b - roc_auc_score(y, score_b)) < 1e-9
    assert abs(res.diff - (res.auc_a - res.auc_b)) < 1e-12
    # 真差异：显著、p 极小、z>0
    assert res.significant and res.p_value < 1e-6 and res.z > 0, res.format()
    assert res.n_pos + res.n_neg == n

    # 同一分数两侧：diff=0、p=1、不显著
    same = delong_test(y, score_a, score_a)
    assert abs(same.diff) < 1e-12 and same.p_value == 1.0 and not same.significant

    # 对称性：交换 A/B → diff 反号、z 反号、p 与 AUC 不变
    swap = delong_test(y, score_b, score_a)
    assert abs(swap.diff + res.diff) < 1e-12 and abs(swap.z + res.z) < 1e-9
    assert abs(swap.p_value - res.p_value) < 1e-12

    # DeLong 单-AUC 方差 与 bootstrap 方差交叉验证（弱一致，容差宽松）
    # 用 [score_a, score_a+微噪] 的差不合适；改直接验证单 AUC 的 DeLong 方差：
    aucs, cov = _fast_delong(np.vstack((score_a, score_b))[:, np.argsort(-y, kind="mergesort")],
                             int((y == 1).sum()))
    var_a_delong = float(cov[0, 0])
    # bootstrap 单 AUC 方差
    boot_aucs = []
    g = np.random.default_rng(5)
    for _ in range(800):
        idx = g.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y[idx], score_a[idx]))
    var_a_boot = float(np.var(boot_aucs, ddof=1))
    # 同数量级即可（DeLong 渐近、bootstrap 有限样本，容差 ×/÷ 2.5）
    assert 0.4 < var_a_delong / var_a_boot < 2.5, (var_a_delong, var_a_boot)

    # 独立无差异两预测：多数情况下 p 不显著（此处只验证能跑且 p∈[0,1]）
    ind_a = 0.5 * y + rng.standard_normal(n)
    ind_b = 0.5 * y + rng.standard_normal(n)
    ri = delong_test(y, ind_a, ind_b)
    assert 0.0 <= ri.p_value <= 1.0

    # 输入校验
    for bad in [
        lambda: delong_test(y, score_a, score_b[:-1]),      # 长度不一致
        lambda: delong_test(np.ones(n, dtype=int), score_a, score_b),  # 单类
    ]:
        try:
            bad(); raise AssertionError("应报错")
        except ValueError:
            pass

    print("significance A2.3 self-test 全部通过 ✓（AUC==sklearn / 真差异显著 / 同分数p=1 / 对称性 / SE~bootstrap / 校验）")


if __name__ == "__main__":
    _selftest()
    _selftest_a22()
    _selftest_a23()
