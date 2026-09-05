"""
逐子群水平 + 组间 gap + 阈值上的逐组 accuracy（averaging 口径，零重训）
========================================================================
主报告目前只摆 Overall AUC 与 worst-group AUC。导师要求补两样：**组间差异（gap）**，以及
**先在全体上定一个阈值、再逐组报 accuracy**。本脚本在已落盘预测上重算这两类读数，
口径与 `build_oof_results_averaging.py` 完全一致（逐折算 → 折间平均 → 再跨组取 min/max）。

产出（`outputs/conditioning_ablation/group_levels_averaging.json`）：
  1. **逐组 AUC 水平**：每个子群的折间平均 AUC（marginal 与 canonical 全给），补上主表只给
     min 而看不到分布的缺口。
  2. **两套 gap**：`marginal_gap` 为新增——既有 JSON 的 `gap` 是 canonical 分区（含 joint 交叉格）
     的极差，而主终点 worst-group 用的是 marginal 分区，两者分区不一致。
  3. **阈值上的逐组 accuracy**：三条阈值规则各一套（见下），逐组给
     accuracy / balanced accuracy / sensitivity / specificity，并给 worst 与 gap。

**阈值口径（三条规则，主口径 youden_val）**
  - `youden_val` : 每折在该折**验证集**上取 Youden J 最大的阈值。**主口径**——组无关（一个阈值
                   施加到所有组）、不碰评估数据、患病率稳健（CheXpert 患病率 8.2%，
                   最大化 accuracy 的阈值会退化成近乎全判负）。
  - `fpr20_val`  : 每折在验证集上取使整体 FPR≈0.2 的阈值（衔接既有操作点公平口径）。
  - `half`       : 恒取 prob=0.5（即 logit 0）的原生阈值。
  三条并列是**敏感性检查**，不是可挑选的口径：结论若随规则而变，须在文档里如实说明。

**为什么阈值必须逐折选、逐折施加、绝不池化**：把 5 折预测拼起来再定一个阈值，会把逐折的
logit 校准漂移当成信号（pooled AUC 的负偏已有前例，见 `docs/oof_regime_results.md`）；
阈值类指标对这种漂移比 AUC 敏感得多。故每折用自己的阈值，指标折间平均。

**bootstrap 中阈值固定不重估**：阈值来自与评估集互斥的验证集，按 plug-in 操作点惯例视为已知常量。
区间因此不含「阈值估计本身」的不确定性；同理，进入 worst/gap 的**合格子群集合**由观测计数一次定死
（门限 min_class_n=10），不随 replicate 变化——否则被最小化的对象会逐 replicate 漂移，
估计的就不再是同一个量（与 AUC 侧「先折间平均再取 min」同一条理由）。

**回归自检**：本脚本重算的 Overall / canonical worst / marginal worst / canonical gap
必须与 `oof_results_averaging.json` 既有值逐位一致（默认开启，`--no-regression` 可关）。
它保证新管线寻址到的是同一批预测、同一份选定配置、同一套过滤规则。

运行（轻量 CPU；MIMIC/CheXpert 因样本量大，1000 次 bootstrap 需数十分钟）：
    python scripts/build_group_levels_averaging.py                    # 全部 regime
    python scripts/build_group_levels_averaging.py --dataset HAM10000 --n-boot 200
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.build_oof_results_averaging import (
    FOLD_SEEDS, OUTPUTS, SPECS, Spec, Views, build_masks, clusters, load_folds,
)
from scripts.eaudit_m2c_full_target import weighted_auc
from src.training.harness.predictions import load_predictions
from src.utils.operating_point_fairness import (
    rates_from_counts, threshold_at_overall_fpr, threshold_youden, to_prob,
)

# 报告里的六个臂（ROC 与 PAPILA 已全文删除，不出现）。HAM 上三个 HN 走双属性 `_sexage` 键。
REPORT_ARMS = ("erm", "swad", "groupdro", "hyperhead", "hyperfusion", "hyperadapt")
HAM_SEXAGE_ARMS = ("hyperhead", "hyperfusion", "hyperadapt")
# 配对 bootstrap 的参照基线（同既有主脚本）
BASELINES = ("erm", "swad", "groupdro")
# 三条阈值规则；元组序即报表列序，第一个是主口径
THRESHOLD_RULES = ("youden_val", "fpr20_val", "half")
FPR_TARGET = 0.2
MIN_CLASS_N = 10                 # 进入 worst/gap 所需的对应分母计数（同 operating_point_fairness）
N_BOOT = 1000
# 逐组报的四个阈值指标（accuracy 必须与 balanced accuracy 成对读，见 operating_point_fairness）
CLF_REPORT = ("accuracy", "balanced_accuracy", "sensitivity", "specificity")
# AUC 侧的标量终点（顺序即 bootstrap 向量的前五位）
AUC_KEYS = ("overall", "canonical_worst", "marginal_worst", "gap", "marginal_gap")
# 既有 JSON 里对应的键名（`marginal_gap` 是本脚本新增，故不参与回归比对）
REGRESSION_KEYS = ("overall", "canonical_worst", "marginal_worst", "gap")

# 各 regime 的**验证集**预测目录（阈值来源）。ID = 自身；OOD = **源域**的 val
# ——目标域没有验证集，部署时能用的只有源域训练出来的那个操作点。
VAL_DIR_REL = {
    "HAM10000": "ham10000/cv5/predictions",
    "Fitzpatrick": "fitzpatrick/cv5/predictions",
    "MIMIC": "mimic_cxr/cv5/predictions",
    "CheXpert": "chexpert_cxr/cv5/predictions",
    "M2C_OOD": "mimic_cxr/cv5/predictions",
    "C2M_OOD": "chexpert_cxr/cv5/predictions",
}
# PAPILA 已从报告删除，不进本脚本
SKIP_REGIMES = ("PAPILA",)


def arm_key(spec: Spec, arm: str) -> str:
    """把报告里的臂名映射为该 regime 的预测键（HAM 的三个 HN 用 `_sexage` 双属性版）。"""
    if spec.name == "HAM10000" and arm in HAM_SEXAGE_ARMS:
        return f"{arm}_sexage"
    return arm


def load_val_folds(spec: Spec, method: str, tag: str) -> list[dict] | None:
    """
    载入某臂五折的**验证集**预测（只取标签与分数，阈值选择不需要属性）。

    Args:
        spec  : regime 规格（提供 HAM 的 age 过滤规则）。
        method: 预测键（如 "hyperadapt_sexage"）。
        tag   : 选定配置 tag。

    Returns:
        [{y, prob}] 逐折列表；缺任一折返回 None（该臂在该规则下记为不可用）。
    """
    val_dir = OUTPUTS / VAL_DIR_REL[spec.name]
    out: list[dict] = []
    for seed in FOLD_SEEDS:
        p = val_dir / f"{method}_{tag}_seed{seed}_val_overall.npz"
        if not p.exists():
            return None
        y, s, a = load_predictions(p)
        y = np.asarray(y).astype(int)
        s = np.asarray(s, dtype=np.float64)
        if spec.filter_age_neg1 and "age" in a:      # HAM：与 test 侧同一条过滤规则
            keep = np.asarray(a["age"]).astype(int) >= 0
            y, s = y[keep], s[keep]
        out.append({"y": y, "prob": to_prob(s, score_is_prob=False)})
    return out


def fold_thresholds(rule: str, val_folds: list[dict] | None, n_folds: int) -> list[float] | None:
    """
    按规则给出逐折阈值（概率尺度）。

    Args:
        rule     : THRESHOLD_RULES 之一。
        val_folds: 验证集预测（`half` 规则下可为 None）。
        n_folds  : 折数。

    Returns:
        长度 n_folds 的阈值列表；规则需要 val 而 val 缺失时返回 None。
    """
    if rule == "half":
        return [0.5] * n_folds
    if val_folds is None:
        return None
    if rule == "youden_val":
        return [threshold_youden(f["y"], f["prob"]) for f in val_folds]
    if rule == "fpr20_val":
        return [threshold_at_overall_fpr(f["y"], f["prob"], FPR_TARGET) for f in val_folds]
    raise ValueError(f"未知阈值规则 {rule!r}，合法：{THRESHOLD_RULES}")


class AucViews(Views):
    """在主脚本的 `Views` 上补两样：逐组 AUC 水平，以及 marginal 分区的 gap。"""

    def compute_full(self, offs: np.ndarray, w: np.ndarray) -> tuple[dict[str, float], list[float]]:
        """
        逐折算 → 折间平均 → 再跨组归约。

        Args:
            offs: 各折在拼接向量中的起止偏移。
            w   : [N] 逐样本权重（bootstrap 的抽中次数；点估计时全 1）。

        Returns:
            (逐组折间平均 AUC 字典, 五个标量终点 [overall, canon_worst, marg_worst,
             canon_gap, marg_gap])。某组在所有折都无定义时不进字典。
        """
        ov: list[float] = []
        per: dict[str, list[float]] = {g: [] for g in self.groups}
        for k in range(self.n_folds):
            wk = w[offs[k]:offs[k + 1]]
            i, ys = self.v[(k, "__all__")]
            a = weighted_auc_safe(ys, wk[i])
            if a is not None:
                ov.append(a)
            for g in self.groups:
                ig, yg = self.v[(k, g)]
                ag = weighted_auc_safe(yg, wk[ig])
                if ag is not None:
                    per[g].append(ag)
        means = {g: float(np.mean(v)) for g, v in per.items() if v}
        marg = [v for g, v in means.items() if "|" not in g]
        canon = list(means.values())
        scalars = [float(np.mean(ov)), min(canon), min(marg),
                   max(canon) - min(canon), max(marg) - min(marg)]
        return means, scalars


def weighted_auc_safe(y_sorted: np.ndarray, w_sorted: np.ndarray) -> float | None:
    """加权 AUC，无定义（该 replicate 该组只剩一类）时返回 None 而非 nan。"""
    a = weighted_auc(y_sorted, w_sorted)
    return None if np.isnan(a) else float(a)


class ClfViews:
    """
    阈值类指标的预计算视图：把每折每样本归入混淆矩阵四格之一，其后一切读数都是加权计数之比。

    阈值一旦固定，「某样本是 TP 还是 FN」就是**常量**，于是 bootstrap 的每个 replicate 只需在
    权重上做四次加权求和（`np.bincount`），不必重跑分类——这让 1000 次重抽样几乎不花时间。
    """

    CELL_TP, CELL_FN, CELL_TN, CELL_FP = 0, 1, 2, 3

    def __init__(self, spec: Spec, folds: list[dict], thresholds: list[float]) -> None:
        """
        Args:
            spec      : regime 规格（决定子群 schema）。
            folds     : 逐折 test 预测（含 y / s / attrs）。
            thresholds: 逐折阈值（概率尺度）。
        """
        self.n_folds = len(folds)
        # 只用边缘子群：阈值类指标在 joint 小格上会退化成 0/1 噪声（同 operating_point_fairness）
        self.groups: list[str] = [g for g in build_masks(spec, folds[0]["attrs"]) if "|" not in g]
        # 逐 (折, 组) 预存样本索引与**已切好**的四格编码——bootstrap 每 replicate 只剩
        # 一次 gather + 一次 bincount，不必重跑分类，也不必重复切片编码数组
        self.idx: dict[tuple[int, str], np.ndarray | None] = {}
        self.cell: dict[tuple[int, str], np.ndarray] = {}
        for k, f in enumerate(folds):
            prob = to_prob(np.asarray(f["s"], dtype=float), score_is_prob=False)
            y = np.asarray(f["y"]).astype(int)
            pred = (prob >= thresholds[k]).astype(int)
            code = np.where(y == 1,
                            np.where(pred == 1, self.CELL_TP, self.CELL_FN),
                            np.where(pred == 1, self.CELL_FP, self.CELL_TN)).astype(np.int8)
            self.idx[(k, "__all__")] = None                    # None = 整折，省一次全量 gather
            self.cell[(k, "__all__")] = code
            for g, m in build_masks(spec, f["attrs"]).items():
                if "|" in g:
                    continue
                gi = np.flatnonzero(m)
                self.idx[(k, g)] = gi
                self.cell[(k, g)] = code[gi]
        # 合格子群集合按**观测**计数一次定死，不随 bootstrap replicate 变化（见模块 docstring）
        self.eligible: dict[str, set[str]] = self._eligible_groups()

    def _counts(self, k: int, group: str, w: np.ndarray | None,
                ) -> tuple[float, float, float, float]:
        """
        某折某组的加权四计数 (tp, fn, tn, fp)。

        Args:
            k    : 折号。
            group: 子群键，或 "__all__" 表示整折。
            w    : 该折的逐样本权重；None = 不加权（观测计数）。

        Returns:
            (tp, fn, tn, fp) 四个加权计数。
        """
        idx = self.idx[(k, group)]
        weights = None if w is None else (w if idx is None else w[idx])
        c = np.bincount(self.cell[(k, group)], weights=weights, minlength=4)
        return float(c[0]), float(c[1]), float(c[2]), float(c[3])

    def _eligible_groups(self) -> dict[str, set[str]]:
        """
        每个指标的合格子群集合：该指标对应的分母（跨折累加的观测计数）≥ MIN_CLASS_N。

        Returns:
            {指标名 -> 合格子群键集合}。
        """
        denom = {"accuracy": lambda c: c[0] + c[1] + c[2] + c[3],
                 "balanced_accuracy": lambda c: min(c[0] + c[1], c[2] + c[3]),
                 "sensitivity": lambda c: c[0] + c[1],
                 "specificity": lambda c: c[2] + c[3]}
        out: dict[str, set[str]] = {m: set() for m in CLF_REPORT}
        for g in self.groups:
            total = np.zeros(4)
            for k in range(self.n_folds):
                total += np.array(self._counts(k, g, None))
            for m in CLF_REPORT:
                if denom[m](total) >= MIN_CLASS_N:
                    out[m].add(g)
        return out

    def compute_full(self, offs: np.ndarray, w: np.ndarray,
                     ) -> tuple[dict[str, dict[str, float]], list[float]]:
        """
        逐折算 → 折间平均 → 再跨组归约（与 AUC 侧同一顺序）。

        Args:
            offs: 各折在拼接向量中的起止偏移。
            w   : [N] 逐样本权重。

        Returns:
            (逐组指标字典 {组 -> {指标 -> 值}}, 标量终点列表)。标量顺序 =
            [overall_accuracy, overall_balanced_accuracy, 然后每个 CLF_REPORT 指标的 (worst, gap)]。
        """
        per: dict[str, dict[str, list[float]]] = {g: {m: [] for m in CLF_REPORT}
                                                  for g in self.groups}
        overall: dict[str, list[float]] = {m: [] for m in CLF_REPORT}
        for k in range(self.n_folds):
            wk = w[offs[k]:offs[k + 1]]
            rates_all = rates_from_counts(*self._counts(k, "__all__", wk))
            for m in CLF_REPORT:
                if rates_all[m] is not None:
                    overall[m].append(float(rates_all[m]))
            for g in self.groups:
                r = rates_from_counts(*self._counts(k, g, wk))
                for m in CLF_REPORT:
                    if r[m] is not None:
                        per[g][m].append(float(r[m]))
        means = {g: {m: float(np.mean(v)) for m, v in d.items() if v} for g, d in per.items()}
        scalars = [_mean_or_nan(overall["accuracy"]), _mean_or_nan(overall["balanced_accuracy"])]
        for m in CLF_REPORT:
            vals = [means[g][m] for g in self.groups
                    if g in self.eligible[m] and m in means.get(g, {})]
            scalars += ([min(vals), max(vals) - min(vals)] if vals else [np.nan, np.nan])
        return means, scalars


def _mean_or_nan(values: list[float]) -> float:
    """列表均值；空列表返回 nan（bootstrap 里权重全落空的极端 replicate）。"""
    return float(np.mean(values)) if values else float("nan")


def clf_scalar_keys() -> list[str]:
    """阈值类标量终点的键名，顺序与 `ClfViews.compute_full` 的返回一致。"""
    keys = ["overall_accuracy", "overall_balanced_accuracy"]
    for m in CLF_REPORT:
        keys += [f"worst_{m}", f"gap_{m}"]
    return keys


@dataclass
class ArmData:
    """一个臂在一个 regime 下的全部预计算视图。"""

    tag: str
    auc: AucViews
    clf: dict[str, ClfViews]          # 阈值规则 -> 视图（规则不可用时缺键）
    thresholds: dict[str, list[float]]


def build_arm(spec: Spec, arm: str, method: str, tag: str) -> ArmData | None:
    """
    载入一个臂的预测并构建 AUC / 各阈值规则的视图。

    Args:
        spec  : regime 规格。
        arm   : 报告里的臂名。
        method: 该 regime 的预测键。
        tag   : 选定配置 tag。

    Returns:
        ArmData；预测缺失时返回 None。
    """
    folds = load_folds(spec, method, tag)
    if folds is None:
        return None
    val_folds = load_val_folds(spec, method, tag)
    if val_folds is None:
        print(f"    [注意] {arm}: 缺 val 预测，仅 `half` 规则可用")
    clf: dict[str, ClfViews] = {}
    thr: dict[str, list[float]] = {}
    for rule in THRESHOLD_RULES:
        t = fold_thresholds(rule, val_folds, len(folds))
        if t is None:
            continue
        thr[rule] = t
        clf[rule] = ClfViews(spec, folds, t)
    return ArmData(tag=tag, auc=AucViews(spec, folds), clf=clf, thresholds=thr)


def metric_vector(arm: ArmData, offs: np.ndarray, w: np.ndarray,
                  ) -> tuple[np.ndarray, dict, dict]:
    """
    一个臂在一次（重）抽样下的全部标量终点，拼成定长向量供配对 bootstrap 相减。

    Args:
        arm : 该臂的视图。
        offs: 折偏移。
        w   : 逐样本权重。

    Returns:
        (标量向量, 逐组 AUC 字典, {阈值规则 -> 逐组指标字典})。
        向量布局 = AUC 五项 + 每条可用阈值规则的 clf 标量（规则缺失时填 nan）。
    """
    per_auc, vec = arm.auc.compute_full(offs, w)
    per_clf: dict[str, dict] = {}
    n_clf = len(clf_scalar_keys())
    for rule in THRESHOLD_RULES:
        if rule not in arm.clf:
            vec += [np.nan] * n_clf
            continue
        means, scal = arm.clf[rule].compute_full(offs, w)
        per_clf[rule] = means
        vec += scal
    return np.asarray(vec, dtype=float), per_auc, per_clf


def vector_keys() -> list[str]:
    """标量向量的键名（与 `metric_vector` 的布局一致）。"""
    keys = list(AUC_KEYS)
    for rule in THRESHOLD_RULES:
        keys += [f"{rule}:{k}" for k in clf_scalar_keys()]
    return keys


def run_spec(spec: Spec, n_boot: int, regression: bool) -> dict | None:
    """
    跑一个 regime：点估计（含逐组水平）+ 配对 cluster bootstrap（对 BASELINES）。

    Args:
        spec      : regime 规格。
        n_boot    : bootstrap 次数。
        regression: 是否与既有 oof_results_averaging.json 做逐位回归比对。

    Returns:
        结果字典；预测或配置缺失时 None。
    """
    if not spec.pred_dir.exists() or not spec.cfg.exists():
        print(f"[跳过] {spec.name}: 缺预测或 config")
        return None
    cfg = json.loads(spec.cfg.read_text())["config"]
    print(f"\n{'=' * 78}\n{spec.name}\n{'=' * 78}")

    arms: dict[str, ArmData] = {}
    for arm in REPORT_ARMS:
        method = arm_key(spec, arm)
        if method not in cfg:
            print(f"    [跳过] {arm}: selected_configs 无 {method}")
            continue
        data = build_arm(spec, arm, method, cfg[method])
        if data is None:
            print(f"    [跳过] {arm}: 缺预测（{method}_{cfg[method]}）")
            continue
        arms[arm] = data
    if "erm" not in arms:
        print(f"[跳过] {spec.name}: 无 ERM")
        return None

    folds_erm = load_folds(spec, arm_key(spec, "erm"), arms["erm"].tag)
    cidx, offs = clusters(spec, folds_erm)
    n_cl = int(cidx.max() + 1)
    ones = np.ones(offs[-1])

    point: dict[str, dict] = {}
    obs: dict[str, np.ndarray] = {}
    for arm, data in arms.items():
        vec, per_auc, per_clf = metric_vector(data, offs, ones)
        obs[arm] = vec
        point[arm] = {
            "config": data.tag,
            "scalars": {k: _jsonable(v) for k, v in zip(vector_keys(), vec)},
            "per_group_auc": per_auc,
            "thresholds": data.thresholds,
            "per_group_clf": per_clf,
            "eligible": {r: {m: sorted(g) for m, g in v.eligible.items()}
                         for r, v in data.clf.items()},
        }

    if regression:
        check_regression(spec, point)

    print(f"  bootstrap: {n_boot} 次 × {len(arms)} 臂（cluster n={n_cl}）...")
    rng = np.random.default_rng(42)
    boot = {arm: np.empty((n_boot, len(vector_keys()))) for arm in arms}
    for b in range(n_boot):
        draw = rng.integers(0, n_cl, size=n_cl)
        w = np.bincount(draw, minlength=n_cl).astype(float)[cidx]
        for arm, data in arms.items():
            boot[arm][b], _, _ = metric_vector(data, offs, w)

    vs: dict[str, dict] = {}
    for base in BASELINES:
        if base not in arms:
            continue
        for arm in arms:
            if arm == base:
                continue
            entry = {}
            for j, key in enumerate(vector_keys()):
                d = boot[arm][:, j] - boot[base][:, j]
                d = d[np.isfinite(d)]
                delta = float(obs[arm][j] - obs[base][j])
                if len(d) < 2 or not np.isfinite(delta):
                    entry[key] = {"delta": _jsonable(delta), "ci": None, "sig": None, "p": None}
                    continue
                lo, hi = np.percentile(d, [2.5, 97.5])
                # 两侧 percentile bootstrap p：自举分布中心化到观测 Δ 得零分布，读两尾质量
                # （第 4 章 §4.3.3 的定义；(1+cnt)/(B+1) 避免 p=0）
                null = d - delta
                cnt = int((np.abs(null) >= abs(delta)).sum())
                entry[key] = {"delta": delta, "ci": [float(lo), float(hi)],
                              "sig": bool(lo > 0 or hi < 0),
                              "p": float((1 + cnt) / (len(null) + 1))}
            vs[f"{arm}_vs_{base}"] = entry

    return {"n": int(offs[-1]), "n_clusters": n_cl, "n_boot": n_boot,
            "arms": list(arms), "point": point, "vs": vs}


def _jsonable(v: float) -> float | None:
    """nan → None（JSON 无 nan 字面量）。"""
    return None if not np.isfinite(v) else float(v)


def check_regression(spec: Spec, point: dict) -> None:
    """
    与既有 `oof_results_averaging.json` 逐位比对四个既有终点，不一致即抛错。

    这条断言保证新管线寻址到的是同一批预测、同一份选定配置、同一套过滤规则；
    `marginal_gap` 与阈值类指标是新增量，无可比对象。

    Args:
        spec : regime 规格。
        point: 本脚本的点估计。

    Raises:
        AssertionError: 任一既有终点与既有 JSON 不一致（容差 1e-12）。
    """
    ref_path = OUTPUTS / "conditioning_ablation" / "oof_results_averaging.json"
    if not ref_path.exists():
        print("  [回归自检跳过] 无 oof_results_averaging.json")
        return
    ref = json.loads(ref_path.read_text()).get(spec.name)
    if ref is None:
        print(f"  [回归自检跳过] 既有 JSON 无 {spec.name}")
        return
    checked = 0
    for arm, data in point.items():
        method = arm_key(spec, arm)
        if method not in ref["point"]:
            continue
        for key in REGRESSION_KEYS:
            new = data["scalars"][key]
            old = ref["point"][method][key]
            assert abs(new - old) < 1e-12, (
                f"{spec.name}/{arm}/{key}: 新 {new!r} vs 既有 {old!r}——"
                f"寻址或口径与 build_oof_results_averaging.py 不一致")
            checked += 1
    print(f"  ✓ 回归自检: {checked} 个既有终点与 oof_results_averaging.json 逐位一致")


def print_summary(name: str, res: dict) -> None:
    """打印一个 regime 的速读表（gap 与主口径 accuracy）。"""
    print(f"\n  [{name}] n={res['n']:,}  clusters={res['n_clusters']:,}")
    hdr = (f"  {'臂':<12s} {'Ov AUC':>8s} {'WG AUC':>8s} {'marg gap':>9s} "
           f"{'canon gap':>10s} | {'Ov acc':>7s} {'WG acc':>7s} {'acc gap':>8s} {'WG bacc':>8s}")
    print(hdr)
    for arm, d in res["point"].items():
        s = d["scalars"]
        r = "youden_val:"
        def g(k: str) -> str:
            v = s.get(k)
            return "   n/a" if v is None else f"{v:.4f}"
        print(f"  {arm:<12s} {g('overall'):>8s} {g('marginal_worst'):>8s} "
              f"{g('marginal_gap'):>9s} {g('gap'):>10s} | "
              f"{g(r + 'overall_accuracy'):>7s} {g(r + 'worst_accuracy'):>7s} "
              f"{g(r + 'gap_accuracy'):>8s} {g(r + 'worst_balanced_accuracy'):>8s}")


def main() -> None:
    """遍历 regime，落盘 JSON 并打印速读表。"""
    ap = argparse.ArgumentParser(description="逐组水平 / 组间 gap / 阈值上的逐组 accuracy")
    ap.add_argument("--dataset", default=None, help="只跑某 regime（缺省=全部）")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--no-regression", action="store_true", help="关掉与既有 JSON 的回归自检")
    ap.add_argument("--out", default=None, help="输出 JSON 路径（缺省 outputs/conditioning_ablation/）")
    args = ap.parse_args()

    out_path = (OUTPUTS / "conditioning_ablation" / "group_levels_averaging.json"
                if args.out is None else Path(args.out))
    results: dict[str, dict] = {}
    if out_path.exists():                       # 单 regime 重跑时保留其余 regime 的既有结果
        results = json.loads(out_path.read_text())
    for spec in SPECS:
        if spec.name in SKIP_REGIMES:
            continue
        if args.dataset and spec.name != args.dataset:
            continue
        res = run_spec(spec, args.n_boot, not args.no_regression)
        if res is not None:
            results[spec.name] = res
            print_summary(spec.name, res)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"\n已写出 {out_path}")


if __name__ == "__main__":
    main()
