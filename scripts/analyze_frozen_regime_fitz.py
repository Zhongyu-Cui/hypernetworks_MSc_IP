"""实验 F(Fitzpatrick17k, skin)分析：冻结 vs 全微调，单-split 5-seed 确认。
判决 H1(方差)/H2(worst-group AUC)/H3(操作点 F1·recall·Eopp)/H4(Overall)。
Fitzpatrick 单一肤色轴 6 子群，无 age 过滤；full-finetune 参照沿用 Pareto idx3。"""
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

from src.training.harness.predictions import load_predictions
from src.training.harness.subgroup_auc import subgroup_auc_vector
from src.training.harness.significance import (
    mean_ci, worst_and_gap, paired_bootstrap_diff, delong_test, make_subgroup_metric,
)
from src.utils.operating_point_fairness import (
    to_prob, threshold_at_overall_fpr, subgroup_tpr_fpr, worst_group_tpr, equalized_odds_gap,
)

PRED = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/fitzpatrick/predictions")
SEEDS = (42, 43, 44, 45, 46)
SEL = "overall"
ARMS = {
    "ERM full":          ("erm",               "lr1e-04_wd1e-03"),  # 全微调 Pareto idx3
    "HyperAdapt full":   ("hyperadapt",        "lr1e-04_wd1e-03"),  # 全微调 Pareto idx3
    "ERM frozen":        ("erm_frozen",        "lr3e-04_wd1e-03"),  # 冻结 Pareto idx5
    "HyperAdapt frozen": ("hyperadapt_frozen", "lr1e-04_wd1e-04"),  # 冻结 Pareto idx2
}


def load_seed(prefix, tag, seed):
    y, s, attrs = load_predictions(PRED / f"{prefix}_{tag}_seed{seed}_{SEL}.npz")
    return y, s, attrs["skin"]


def per_seed(prefix, tag):
    r = {"overall": [], "worst": [], "gap": [], "wtpr": [], "eopp": []}
    for seed in SEEDS:
        y, s, skin = load_seed(prefix, tag, seed)
        vec = subgroup_auc_vector("fitzpatrick", y, s, skin=skin)
        worst, gap = worst_and_gap(vec)
        r["overall"].append(roc_auc_score(y, s)); r["worst"].append(worst); r["gap"].append(gap)
        prob = to_prob(s, score_is_prob=False)
        thr = threshold_at_overall_fpr(y, prob, 0.2)
        rates = subgroup_tpr_fpr("fitzpatrick", y, prob, {"skin": skin}, thr, marginal_only=False)
        r["wtpr"].append(worst_group_tpr(rates, min_class_n=10))
        r["eopp"].append(equalized_odds_gap(rates, min_class_n=10)[2])
    return r


print(f"{'='*78}\n实验 F·Fitzpatrick 判决：冻结 vs 全微调（skin，5-seed 单-split，sel={SEL}）\n{'='*78}")
for name, (prefix, tag) in ARMS.items():
    r = per_seed(prefix, tag)
    ov, wo, gp = mean_ci(r["overall"]), mean_ci(r["worst"]), mean_ci(r["gap"])
    wt = mean_ci([x for x in r["wtpr"] if x is not None])
    eo = mean_ci([x for x in r["eopp"] if x is not None])
    print(f"\n### {name}  ({prefix}_{tag})")
    print(f"  Overall AUC : {ov.mean:.4f} ±{ov.half_width:.4f}  (seeds {[f'{v:.3f}' for v in r['overall']]})")
    print(f"  worst-group : {wo.mean:.4f} ±{wo.half_width:.4f}  (seeds {[f'{v:.3f}' for v in r['worst']]})")
    print(f"  H1 worst std: {np.std(r['worst'], ddof=1):.4f}   gap {gp.mean:.4f}")
    print(f"  wTPR@.2 {wt.mean:.3f}±{wt.half_width:.3f}   Eopp {eo.mean:.3f}±{eo.half_width:.3f}")


def paired(hn, erm):
    hp, ht = ARMS[hn]; ep, et = ARMS[erm]
    yh, sh, skinh = load_seed(hp, ht, 42)
    ye, se, _ = load_seed(ep, et, 42)
    assert np.array_equal(yh, ye), "配对未对齐"
    attrs = {"skin": skinh}
    m_hn = make_subgroup_metric("fitzpatrick", yh, sh, attrs, "worst")
    m_er = make_subgroup_metric("fitzpatrick", ye, se, attrs, "worst")
    bd = paired_bootstrap_diff(len(yh), m_hn, m_er, n_boot=2000, rng=0)
    dl = delong_test(yh, sh, se)
    print(f"\n[{hn} − {erm}] (seed42)")
    print(f"  worst Δ={bd.point:+.4f} CI[{bd.ci_low:+.4f},{bd.ci_high:+.4f}] {'✓' if (bd.ci_low>0 or bd.ci_high<0) else '✗跨0'}")
    print(f"  Overall DeLong Δ={dl.diff:+.4f} p={dl.p_value:.3g} {'✓' if dl.p_value<0.05 else '✗n.s.'}")


print(f"\n{'='*78}\n配对（HN vs ERM）\n{'='*78}")
paired("HyperAdapt frozen", "ERM frozen")
paired("HyperAdapt full", "ERM full")
print("\nDONE")
