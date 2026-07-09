"""
逐敏感组分类四联指标（sensitivity/specificity/PPV/NPV）+ 跨方法排名
====================================================================
在**已落盘预测 npz** 上零重训重算**阈值相关**的临床四联指标，补主报告（AUC / worst-group AUC /
AUC gap）与 D8（EqOdds gap / worst-TPR）之外的切面：给每个敏感子群的
  - sensitivity（=TPR=召回，越大越好）
  - specificity（=TNR=1−FPR，越大越好）
  - PPV（阳性预测值=precision，依子群患病率，越大越好）
  - NPV（阴性预测值，越大越好）
并据此对 6 方法在每个数据集上按各指标排名（含已算的 AUC 类指标一并入排名）。

**口径（与 D3/D5/D8 完全一致，单一事实来源 = build_results_tables.DATASETS）**：
  - 同 Pareto 选定 config、同 5 seed、HAM 过滤 age≥0、PAPILA 5 折 OOF 池化。
  - ERM/SWAD/3 HN：落盘 logits → sigmoid 成概率；**ROC**：走 R2.3 操作点重选 θ 后的调整概率。
  - **操作点 = 固定整体 FPR=0.2**（类别不均衡稳健；原生 0.5 在低患病率 CXR 上退化，故不作主口径）。
    该操作点令各方法**整体 FPR 相同（整体特异度=0.8）**，故跨方法差异纯来自四联指标在子群间的分布。
  - 子群 = 边缘敏感组（键不含 `|`）；worst/gap 纳入门限 min_class_n=10（对应分母计数）。

排名标量（每方法每指标一个）：
  已算类：Overall AUC↑、Worst-group AUC↑、AUC gap↓（来自 build_results_tables._metrics_one）。
  新增类（@FPR0.2）：Worst-group {sensitivity, specificity, PPV, NPV}↑。
  worst-group 敏感度在此操作点 == D8 的 worst-TPR@FPR0.2（互为一致性校验）。

用法：python scripts/build_subgroup_clf_metrics.py [--target-fpr 0.2] [--out docs/subgroup_clf_metrics.md]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.training.harness.significance import mean_ci
from src.utils.operating_point_fairness import (
    to_prob, threshold_at_overall_fpr, subgroup_classification_metrics,
    worst_group_clf, clf_gap, clf_metric_values, _clf_metrics, CLF_METRIC_COUNT,
)

# 复用 D3/D5/D8 的数据集/方法/seed/选定配置 + AUC 类指标（单一事实来源）
from scripts.build_results_tables import (
    DATASETS, SEEDS, METHOD_ORDER, METHOD_LABEL, _load, _metrics_one,
)
from scripts.build_operating_point_tables import _roc_op_test_scores

# 四联指标展示名 + 排名方向（True=越大越好）
CLF_METRICS = ("sensitivity", "specificity", "ppv", "npv")
CLF_LABEL = {"sensitivity": "Sensitivity", "specificity": "Specificity",
             "ppv": "PPV", "npv": "NPV"}
# 排名标量表：指标键 → (展示名, 越大越好?)
RANK_METRICS: list[tuple[str, str, bool]] = [
    ("overall_auc", "Overall AUC", True),
    ("worst_auc", "Worst-group AUC", True),
    ("auc_gap", "AUC gap", False),
    ("worst_sensitivity", "Worst-group Sensitivity", True),
    ("worst_specificity", "Worst-group Specificity", True),
    ("worst_ppv", "Worst-group PPV", True),
    ("worst_npv", "Worst-group NPV", True),
]


def _get_prob(spec: dict, method: str, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """单 run → (y_true, 概率, attrs)。ROC 用 R2.3 重选 θ 后的调整概率，余 logits→sigmoid。"""
    if method == "roc":
        prob, y, attrs, _theta, _axis = _roc_op_test_scores(spec, seed)
        return (np.asarray(y), np.asarray(prob, dtype=float),
                {k: np.asarray(v) for k, v in attrs.items()})
    tag = spec["cfg"][method]
    y, s, attrs = _load(spec["pdir"], method, tag, seed, spec["ham_filter"])
    return y, to_prob(s, score_is_prob=False), attrs


def _run_clf(spec: dict, method: str, seed: int, target_fpr: float):
    """单 run @FPR 操作点：返回 (逐组四联指标 rates, 整体四联指标, 各指标 worst, 各指标 gap)。"""
    y, prob, attrs = _get_prob(spec, method, seed)
    t = threshold_at_overall_fpr(y, prob, target_fpr)
    rates = subgroup_classification_metrics(spec["key"], y, prob, attrs, t)
    overall = _clf_metrics(y, (prob >= t).astype(int))
    worst = {m: worst_group_clf(rates, m) for m in CLF_METRICS}
    gap = {m: clf_gap(rates, m) for m in CLF_METRICS}
    return rates, overall, worst, gap


def _pooled_clf(spec: dict, method: str, target_fpr: float):
    """PAPILA OOF：池化 5 折 disjoint test（ROC 逐折重选 θ 后池化概率），一次算四联指标。"""
    ys, ps, pool = [], [], {k: [] for k in spec["attrs"]}
    for sd in SEEDS:
        y, prob, attrs = _get_prob(spec, method, sd)
        ys.append(y); ps.append(prob)
        for k in spec["attrs"]:
            pool[k].append(attrs[k])
    y = np.concatenate(ys); prob = np.concatenate(ps)
    A = {k: np.concatenate(v) for k, v in pool.items()}
    t = threshold_at_overall_fpr(y, prob, target_fpr)
    rates = subgroup_classification_metrics(spec["key"], y, prob, A, t)
    overall = _clf_metrics(y, (prob >= t).astype(int))
    worst = {m: worst_group_clf(rates, m) for m in CLF_METRICS}
    gap = {m: clf_gap(rates, m) for m in CLF_METRICS}
    return rates, overall, worst, gap


def _mean(vals: list[float | None]) -> float | None:
    """跳过 None 的均值；全 None 时 None。"""
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def _fmt(x: float | None) -> str:
    return "N/A" if x is None else f"{x:.3f}"


def _fmt_ci(vals: list[float | None]) -> str:
    """多-seed → 'mean±half'；<2 有效值降级点估计 / N/A。"""
    v = [x for x in vals if x is not None]
    if not v:
        return "N/A"
    if len(v) == 1:
        return f"{v[0]:.3f}"
    mc = mean_ci(v)
    return f"{mc.mean:.3f}±{mc.half_width:.3f}"


def _aggregate(spec: dict, method: str, target_fpr: float) -> dict:
    """
    聚合单方法在某数据集上的四联指标（非-OOF 走 5-seed，OOF 走池化点估计）。

    Returns dict:
      group_keys: 边缘子群键顺序
      cells[metric][gkey]  : seed 均值点估计（逐组表格用）
      overall[metric]      : 整体指标（seed 均值 / OOF 点估计）
      worst_ci[metric]     : worst-group 的 'mean±half' 文本
      worst_pt[metric]     : worst-group 点估计（排名用标量）
      gap_ci[metric]       : gap 的 'mean±half' 文本
    """
    if spec["oof"]:
        rates, overall, worst, gap = _pooled_clf(spec, method, target_fpr)
        gkeys = list(rates.keys())
        cells = {m: {g: getattr(rates[g], m) for g in gkeys} for m in CLF_METRICS}
        return dict(
            group_keys=gkeys,
            cells=cells,
            overall={m: getattr(overall, m) for m in CLF_METRICS},
            worst_ci={m: _fmt(worst[m]) for m in CLF_METRICS},
            worst_pt={m: worst[m] for m in CLF_METRICS},
            gap_ci={m: _fmt(gap[m]) for m in CLF_METRICS},
        )
    # 非-OOF：逐 seed 收集后聚合
    per_seed = [_run_clf(spec, method, sd, target_fpr) for sd in SEEDS]
    gkeys = list(per_seed[0][0].keys())
    cells = {m: {g: _mean([getattr(rates[g], m) for rates, *_ in per_seed])
                 for g in gkeys} for m in CLF_METRICS}
    overall = {m: _mean([getattr(s[1], m) for s in per_seed]) for m in CLF_METRICS}
    worst_seed = {m: [w[m] for *_, w, _g in per_seed] for m in CLF_METRICS}
    gap_seed = {m: [g[m] for *_, g in per_seed] for m in CLF_METRICS}
    return dict(
        group_keys=gkeys,
        cells=cells,
        overall=overall,
        worst_ci={m: _fmt_ci(worst_seed[m]) for m in CLF_METRICS},
        worst_pt={m: _mean(worst_seed[m]) for m in CLF_METRICS},
        gap_ci={m: _fmt_ci(gap_seed[m]) for m in CLF_METRICS},
    )


def _auc_scalars(spec: dict, method: str) -> tuple[float, float | None, float | None]:
    """AUC 类排名标量（seed 均值 / OOF 点估计）：(Overall AUC, Worst-group AUC, AUC gap)。"""
    if spec["oof"]:
        ys, ss, pool = [], [], {k: [] for k in spec["attrs"]}
        for sd in SEEDS:
            tag = spec["cfg"][method]
            y, s, a = _load(spec["pdir"], method, tag, sd, spec["ham_filter"])
            ys.append(y); ss.append(s)
            for k in spec["attrs"]:
                pool[k].append(a[k])
        y = np.concatenate(ys); s = np.concatenate(ss)
        A = {k: np.concatenate(v) for k, v in pool.items()}
        return _metrics_one(spec["key"], y, s, A)
    O, W, G = [], [], []
    for sd in SEEDS:
        tag = spec["cfg"][method]
        y, s, a = _load(spec["pdir"], method, tag, sd, spec["ham_filter"])
        ov, w, g = _metrics_one(spec["key"], y, s, a)
        O.append(ov); W.append(w); G.append(g)
    return _mean(O), _mean(W), _mean(G)


def _ranks(scalars: dict[str, float | None], higher_better: bool) -> dict[str, float]:
    """
    把 {方法: 标量} 转成 {方法: 排名}（1=最好），并列取平均名次；None 排末位。

    Args:
        scalars      : 方法 → 标量（可 None）。
        higher_better: True 时值越大名次越前。
    """
    valid = {m: v for m, v in scalars.items() if v is not None}
    order = sorted(valid, key=lambda m: valid[m], reverse=higher_better)
    ranks: dict[str, float] = {}
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs(valid[order[j + 1]] - valid[order[i]]) < 1e-9:
            j += 1
        avg = (i + 1 + j + 1) / 2.0  # 并列取平均名次
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    for m, v in scalars.items():
        if v is None:
            ranks[m] = float(len(scalars))  # None → 末位
    return ranks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target-fpr", type=float, default=0.2)
    p.add_argument("--out", type=str, default=None, help="额外写出的 markdown 路径（默认仅 stdout）。")
    args = p.parse_args()
    tf = args.target_fpr

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit(f"# 逐敏感组分类四联指标 + 跨方法排名（操作点：固定整体 FPR={tf:g}）\n")
    emit("> 零重训，在 D3/D5 同款落盘预测上重算。**Sensitivity/Specificity/PPV/NPV** 逐边缘敏感组；"
         "worst/gap 门限 min_class_n=10。ROC 走 R2.3 操作点重选 θ 的调整概率。此操作点下各方法整体 "
         f"FPR={tf:g}（整体特异度={1 - tf:g}），跨方法差异纯来自指标在子群间的分布。\n")

    # 逐数据集：聚合 → 逐组四联表 + 排名标量 + 排名
    all_ranks: dict[str, dict[str, dict[str, float]]] = {}   # ds -> metric_key -> method -> rank
    for ds, spec in DATASETS.items():
        agg = {m: _aggregate(spec, m, tf) for m in METHOD_ORDER}
        auc = {m: _auc_scalars(spec, m) for m in METHOD_ORDER}
        gkeys = agg[METHOD_ORDER[0]]["group_keys"]
        oof_note = "; PAPILA 5 折 OOF 池化 n=420（点估计）" if spec["oof"] else "; 5-seed 均值"
        emit(f"\n## {ds}  (敏感轴: {', '.join(spec['attrs'])}{oof_note})")

        # --- 逐组四联指标明细（每指标一表，行=方法，列=各敏感组 + Overall + Worst±CI + Gap）---
        short = [g.split(":", 1)[1] if ":" in g else g for g in gkeys]
        for metric in CLF_METRICS:
            emit(f"\n### {ds} · {CLF_LABEL[metric]}（逐敏感组；越大越好，Gap 越小越公平）")
            emit("| 方法 | " + " | ".join(short) + " | Overall | Worst | Gap |")
            emit("|" + "------|" * (len(short) + 4))
            for m in METHOD_ORDER:
                a = agg[m]
                row = [_fmt(a["cells"][metric][g]) for g in gkeys]
                emit(f"| {METHOD_LABEL[m]} | " + " | ".join(row) +
                     f" | {_fmt(a['overall'][metric])} | {a['worst_ci'][metric]} | {a['gap_ci'][metric]} |")

        # --- 该数据集排名标量收集 ---
        scalars: dict[str, dict[str, float | None]] = {}
        for mk, _label, _hi in RANK_METRICS:
            scalars[mk] = {}
        for m in METHOD_ORDER:
            ov_auc, w_auc, g_auc = auc[m]
            scalars["overall_auc"][m] = ov_auc
            scalars["worst_auc"][m] = w_auc
            scalars["auc_gap"][m] = g_auc
            for metric in CLF_METRICS:
                scalars[f"worst_{metric}"][m] = agg[m]["worst_pt"][metric]

        # --- 排名表（行=指标，列=方法，cell=名次；↑/↓ 标方向）---
        ds_ranks: dict[str, dict[str, float]] = {}
        emit(f"\n### {ds} · 方法排名（1=最好；括注标量点估计）")
        emit("| 指标 (方向) | " + " | ".join(METHOD_LABEL[m] for m in METHOD_ORDER) + " |")
        emit("|" + "------|" * (len(METHOD_ORDER) + 1))
        for mk, label, hi in RANK_METRICS:
            rk = _ranks(scalars[mk], hi)
            ds_ranks[mk] = rk
            arrow = "↑" if hi else "↓"
            cells = []
            for m in METHOD_ORDER:
                v = scalars[mk][m]
                r = rk[m]
                rtxt = f"{r:.1f}" if r != int(r) else f"{int(r)}"
                cells.append(f"**{rtxt}** ({_fmt(v)})")
            emit(f"| {label} ({arrow}) | " + " | ".join(cells) + " |")
        all_ranks[ds] = ds_ranks

    # --- 跨数据集平均名次汇总（每方法每指标，对 5 数据集取平均名次）---
    emit("\n## 跨数据集平均名次汇总（每指标对 5 数据集取平均名次，越小越好）")
    emit("| 指标 (方向) | " + " | ".join(METHOD_LABEL[m] for m in METHOD_ORDER) + " |")
    emit("|" + "------|" * (len(METHOD_ORDER) + 1))
    for mk, label, hi in RANK_METRICS:
        arrow = "↑" if hi else "↓"
        cells = []
        for m in METHOD_ORDER:
            rs = [all_ranks[ds][mk][m] for ds in DATASETS]
            cells.append(f"{np.mean(rs):.2f}")
        emit(f"| {label} ({arrow}) | " + " | ".join(cells) + " |")

    if args.out:
        outp = Path(args.out)
        if not outp.is_absolute():
            outp = Path(__file__).resolve().parents[1] / outp
        outp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n[written] {outp}", file=sys.stderr)


if __name__ == "__main__":
    main()
