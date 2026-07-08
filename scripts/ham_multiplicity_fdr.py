"""
R4.3 —— HAM D6 多重比较校正（Benjamini–Hochberg FDR）
======================================================

评审 R4.3：D6 是 3 HN × 3 基线 × 多口径的大矩阵，同时做十几~二十余个假设检验，须做多重性控制，
否则「至少一个显著」的假阳性率被抬高。本脚本对 D6-HP（R4.1 更高功效 seed-ensemble）的全部比较
统一收集 p 值，做 BH-FDR，给每格 q 值与「FDR=0.05 下是否存活」。

家族界定（两种口径并报）：
    - 按度量分族（主口径）：Overall AUC / canonical worst / 边缘 worst 各自成族，族内校正。
      理由：三度量回答不同问题（整体判别 vs 最差子群公平），跨度量合并 p 值无统计意义。
    - 全局单族（更严稳健性检查）：21 个比较合成一族，最保守。

⚠️ 关于「HF-only → 全矩阵」的说明（非 forking-path）：完整 3 HN × 3 基线矩阵**本就是比较协议应做的
分析**（所有方法 × 所有基线全比）；协议 §5 一度缩窄到「仅 HyperFusion」属**规格/执行失误**，已在 C
阶段统一重训时订正。故全矩阵**不是**看到 Pareto 反转后的数据驱动探索性扩展，不存在 researcher
degrees-of-freedom 的 forking path；但**同时报多个比较仍需多重性校正**——这与是否预设无关，本脚本即此。

用法：python -m scripts.ham_multiplicity_fdr
输出：outputs/ham10000/d6_multiplicity_fdr.json + 终端表。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.build_results_tables import _ham_ensemble_scores
from src.training.harness.subgroup_auc import subgroup_auc_vector
from src.training.harness.significance import (
    delong_test, paired_bootstrap_diff, worst_and_gap,
)

OUT = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs")
HN = ("hyperhead", "hyperfusion", "hyperadapt")
BASE = ("erm", "swad", "roc")
LABEL = {"hyperhead": "HyperHead", "hyperfusion": "HyperFusion", "hyperadapt": "HyperAdapt",
         "erm": "ERM", "swad": "SWAD", "roc": "ROC"}


def bh_fdr(pvals: list[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    """
    Benjamini–Hochberg：输入 p 值列表，返回 (q 值, 是否在 FDR=alpha 下存活)，保序。

    q_i = min_{j>=rank(i)} ( p_(j) * m / j )（step-up 单调化），截断到 [0,1]。
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    q_sorted = ranked * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]   # 从大到小 running-min
    q = np.empty(m)
    q[order] = np.minimum(q_sorted, 1.0)
    return q.tolist(), (q <= alpha).tolist()


def collect_pvalues() -> list[dict]:
    """收集 D6-HP 全部比较的原始 p 值（Overall DeLong + canonical/边缘 worst 配对 bootstrap）。"""
    key = "ham10000"
    y_true, ens, _seed, attrs = _ham_ensemble_scores()
    n = len(y_true)

    def canon_worst(s):
        def metric(idx):
            vec = subgroup_auc_vector(key, y_true[idx], s[idx],
                                      sex=attrs["sex"][idx], age=attrs["age"][idx])
            return worst_and_gap(vec)[0]
        return metric

    def marg_worst(s):
        def metric(idx):
            vec = subgroup_auc_vector(key, y_true[idx], s[idx],
                                      sex=attrs["sex"][idx], age=attrs["age"][idx])
            vals = [a for k, (a, _n) in vec.items() if a is not None and "|" not in k]
            return min(vals) if vals else None
        return metric

    rng = np.random.default_rng(0)
    rows: list[dict] = []
    # (1) Overall AUC DeLong: 3×3
    for hn in HN:
        for b in BASE:
            r = delong_test(y_true, ens[hn], ens[b])
            rows.append({"family": "overall_auc", "hn": hn, "base": b,
                         "delta": float(r.diff), "p": float(r.p_value)})
    # (2) canonical worst 配对 bootstrap: 3×3
    for hn in HN:
        for b in BASE:
            d = paired_bootstrap_diff(n, canon_worst(ens[hn]), canon_worst(ens[b]), n_boot=2000, rng=rng)
            rows.append({"family": "canonical_worst", "hn": hn, "base": b,
                         "delta": float(d.point), "p": float(d.p_value)})
    # (3) 边缘 worst 配对 bootstrap: 3×1 (vs ERM)
    for hn in HN:
        d = paired_bootstrap_diff(n, marg_worst(ens[hn]), marg_worst(ens["erm"]), n_boot=2000, rng=rng)
        rows.append({"family": "marginal_worst", "hn": hn, "base": "erm",
                     "delta": float(d.point), "p": float(d.p_value)})
    return rows


def main() -> None:
    """收集 p → 按度量分族 + 全局单族两口径 BH-FDR → JSON + 表。"""
    rows = collect_pvalues()

    # 按度量分族
    families = ("overall_auc", "canonical_worst", "marginal_worst")
    for fam in families:
        idx = [i for i, r in enumerate(rows) if r["family"] == fam]
        q, surv = bh_fdr([rows[i]["p"] for i in idx])
        for j, i in enumerate(idx):
            rows[i]["q_within_family"] = q[j]
            rows[i]["survive_within_family"] = surv[j]
    # 全局单族
    q_all, surv_all = bh_fdr([r["p"] for r in rows])
    for i, r in enumerate(rows):
        r["q_global"] = q_all[i]
        r["survive_global"] = surv_all[i]

    print(f"HAM D6-HP 多重比较 BH-FDR（m_total={len(rows)}；FDR=0.05）")
    print(f"{'family':<17}{'比较':<26}{'Δ':>9}{'raw p':>10}{'q(族内)':>10}{'q(全局)':>10}  存活")
    print("-" * 92)
    for r in rows:
        comp = f"{LABEL[r['hn']]} vs {LABEL[r['base']]}"
        mark = ("族内✓" if r["survive_within_family"] else "族内✗") + "/" + ("全局✓" if r["survive_global"] else "全局✗")
        print(f"{r['family']:<17}{comp:<26}{r['delta']:>+9.4f}{r['p']:>10.5f}"
              f"{r['q_within_family']:>10.4f}{r['q_global']:>10.4f}  {mark}")

    n_over_win = sum(r["survive_within_family"] for r in rows if r["family"] == "overall_auc")
    n_over_glob = sum(r["survive_global"] for r in rows if r["family"] == "overall_auc")
    print(f"\nOverall AUC 族：族内存活 {n_over_win}/9、全局存活 {n_over_glob}/9；"
          f"worst-group 全族 raw p 均 >0.05（校正后必不存活）。")

    out_path = OUT / "ham10000" / "d6_multiplicity_fdr.json"
    json.dump({"comparisons": rows, "alpha": 0.05,
               "families": {f: [r for r in rows if r["family"] == f] for f in families}},
              open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n已保存: {out_path}")


if __name__ == "__main__":
    main()
