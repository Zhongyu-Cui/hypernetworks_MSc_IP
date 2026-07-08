"""
操作点公平表（评审补强 R2.2 + R2.3）
====================================
在**已落盘预测 npz** 上重算**阈值相关**的操作点公平度量（零重训），补主报告只挂 AUC 的窄切面，
并让 Reject-Option（ROC）在其真正设计目标（子群 TPR/FPR 奇偶）上被测。

两个操作点（见 `src.utils.operating_point_fairness`）：
  ① 原生决策阈值 prob=0.5（logits 的 0，ROC 锚点）：EqOdds gap、worst-group TPR。
  ② 固定整体 FPR=0.2（类别不均衡稳健）：EqOdds gap、worst-group TPR。

**方法口径**（与 D3/D5 完全一致：同 Pareto 选定 config、同 5 seed、HAM 过滤 age≥0、PAPILA OOF 池化）：
  - ERM / SWAD / 3 HN：落盘 **logits**，`score_is_prob=False`。
  - **ROC（R2.3 关键）**：不复用「AUC 选 θ」的旧 npz（那使 θ*=0、ROC≡ERM），而是从 ERM 的 val/test
    预测**重选 θ**，目标 = **最小化 val EqOdds gap（操作点度量）**，使 reject-option 阈值干预真正被
    选中（θ*>0）并被测；产出调整后 test **概率**，`score_is_prob=True`。纯后处理、零重训。

用法：python scripts/build_operating_point_tables.py [--target-fpr 0.2]
"""

import argparse

import numpy as np

from src.training.harness.predictions import load_predictions
from src.training.harness.roc import apply_roc_operating_point_auto
from src.training.harness.significance import mean_ci
from src.utils.operating_point_fairness import operating_point_report, to_prob

# 复用 D3/D5 的数据集/方法/seed/选定配置（单一事实来源）
from scripts.build_results_tables import (
    DATASETS, SEEDS, METHOD_ORDER, METHOD_LABEL, _load, _sel_for,
)


def _load_val(pdir, tag: str, seed: int, ham_filter: bool):
    """加载 ERM 的 **val** 预测（供 ROC 重选 θ）；HAM 同 test 过滤 age≥0，口径与评估一致。"""
    y, s, a = load_predictions(pdir / f"erm_{tag}_seed{seed}_val_overall.npz")
    if ham_filter:
        m = a["age"] >= 0
        y, s = y[m], s[m]
        a = {k: v[m] for k, v in a.items()}
    return np.asarray(y), np.asarray(s), {k: np.asarray(v) for k, v in a.items()}


def _roc_op_test_scores(spec: dict, seed: int):
    """
    R2.3：对某 seed，从 ERM val/test 预测重选操作点 θ，返回 (调整后 test 概率, y_test, test_attrs, θ*, axis)。

    test 真值/属性取 ERM test（ROC 是 ERM 分数的后处理，标签/属性与 ERM 同）。
    """
    tag = spec["cfg"]["erm"]          # roc 与 erm 同 config（派生自 ERM）
    pdir, hf = spec["pdir"], spec["ham_filter"]
    vy, vl, va = _load_val(pdir, tag, seed, hf)
    ty, tl, ta = _load(pdir, "erm", tag, seed, hf)   # ERM test（logits）
    res = apply_roc_operating_point_auto(
        spec["key"], y_val_true=vy, y_val_logits=vl, val_attrs=va,
        y_test_logits=tl, test_attrs=ta,
    )
    return res.adjusted_test_scores, ty, ta, res.best_margin, res.axis


def _run_metrics(spec: dict, method: str, seed: int, target_fpr: float):
    """单 run 的操作点报告；ROC 走重选 θ（概率），余走 logits。返回 (report, θ*|None, axis|None)。"""
    if method == "roc":
        prob, y, attrs, theta, axis = _roc_op_test_scores(spec, seed)
        rep = operating_point_report(spec["key"], y, prob, attrs,
                                     score_is_prob=True, target_fpr=target_fpr)
        return rep, theta, axis
    tag = spec["cfg"][method]
    y, s, attrs = _load(spec["pdir"], method, tag, seed, spec["ham_filter"])
    rep = operating_point_report(spec["key"], y, s, attrs,
                                 score_is_prob=False, target_fpr=target_fpr)
    return rep, None, None


def _fmt_ci(vals: list[float | None]) -> str:
    """多-seed 标量 → 'mean±half'；None 过滤后 <2 个有效值时降级为点估计或 N/A。"""
    v = [x for x in vals if x is not None]
    if not v:
        return "N/A"
    if len(v) == 1:
        return f"{v[0]:.4f}"
    mc = mean_ci(v)
    return f"{mc.mean:.4f}±{mc.half_width:.4f}"


def _pooled_report(spec: dict, method: str, target_fpr: float):
    """PAPILA OOF：池化 5 折 disjoint test（ROC 逐折重选 θ 后池化调整概率），一次算操作点报告。"""
    ys, scores, pool_attrs, is_prob = [], [], {k: [] for k in spec["attrs"]}, (method == "roc")
    thetas = []
    for sd in SEEDS:
        if method == "roc":
            prob, y, attrs, theta, _axis = _roc_op_test_scores(spec, sd)
            s = prob
            thetas.append(theta)
        else:
            tag = spec["cfg"][method]
            y, s, attrs = _load(spec["pdir"], method, tag, sd, spec["ham_filter"])
        ys.append(y); scores.append(s)
        for k in spec["attrs"]:
            pool_attrs[k].append(attrs[k])
    y = np.concatenate(ys); s = np.concatenate(scores)
    A = {k: np.concatenate(v) for k, v in pool_attrs.items()}
    rep = operating_point_report(spec["key"], y, s, A, score_is_prob=is_prob, target_fpr=target_fpr)
    return rep, (float(np.mean(thetas)) if thetas else None)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target-fpr", type=float, default=0.2)
    args = p.parse_args()
    tf = args.target_fpr

    print(f"# 操作点公平表（R2.2/R2.3）  target_fpr={tf}")
    print("> EqOdds gap = max(TPR 组间极差, FPR 组间极差)，**越小越公平**；worst-TPR = 最小子群 TPR，"
          "**越大越好**。边缘子群、min_class_n=10。ROC 为**操作点重选 θ**（R2.3）。\n")

    roc_engage = []   # 收集 ROC 参与情况用于末尾小结
    for ds, spec in DATASETS.items():
        oof = spec["oof"]
        print(f"\n### {ds}  (轴: {', '.join(spec['attrs'])}"
              f"{'; PAPILA OOF 池化 n=420' if oof else '; 5-seed'})")
        print("| 方法 | EqOdds gap @0.5 | worst-TPR @0.5 | EqOdds gap @FPR"
              f"{tf:g} | worst-TPR @FPR{tf:g} |")
        print("|------|-----------------|----------------|------------------|-------------------|")
        for method in METHOD_ORDER:
            if oof:
                rep, theta = _pooled_report(spec, method, tf)
                cells = [rep.native_eo_gap, rep.native_worst_tpr, rep.fpr_eo_gap, rep.fpr_worst_tpr]
                row = " | ".join("N/A" if c is None else f"{c:.4f}" for c in cells)
                print(f"| {METHOD_LABEL[method]} | {row} |")
                if method == "roc":
                    roc_engage.append((ds, "OOF", theta, rep.native_eo_gap))
            else:
                n_eo, n_wt, f_eo, f_wt, thetas, axes = [], [], [], [], [], []
                for sd in SEEDS:
                    rep, theta, axis = _run_metrics(spec, method, sd, tf)
                    n_eo.append(rep.native_eo_gap); n_wt.append(rep.native_worst_tpr)
                    f_eo.append(rep.fpr_eo_gap); f_wt.append(rep.fpr_worst_tpr)
                    if theta is not None:
                        thetas.append(theta); axes.append(axis)
                print(f"| {METHOD_LABEL[method]} | {_fmt_ci(n_eo)} | {_fmt_ci(n_wt)} | "
                      f"{_fmt_ci(f_eo)} | {_fmt_ci(f_wt)} |")
                if method == "roc":
                    ax = max(set(axes), key=axes.count) if axes else "?"
                    roc_engage.append((ds, ax, float(np.mean(thetas)) if thetas else None,
                                       _fmt_ci(n_eo)))

    # ROC 参与小结（R2.3 核心：证 ROC 不再恒等 ERM）
    print("\n### R2.3 ROC 操作点重选小结（θ* 由 val EqOdds gap 选定，含 0 → 不劣 ERM）")
    print("| 数据集 | 二分轴 | 平均 θ* | ROC test EqOdds gap @0.5 |")
    print("|--------|--------|---------|--------------------------|")
    for ds, ax, theta, eo in roc_engage:
        ts = "N/A" if theta is None else f"{theta:.3f}"
        print(f"| {ds} | {ax} | {ts} | {eo} |")


if __name__ == "__main__":
    main()
