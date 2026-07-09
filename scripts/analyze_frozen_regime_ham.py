"""实验 F 分析：冻结 regime vs 全微调（HAM10000, age 条件）5-seed 单-split 确认。
判决 H1(方差)/H2(worst-group AUC)/H3(操作点 F1·recall·Eopp)/H4(Overall 不掉)。
全部对齐到 age_group>=0 的公共 969 子集，保证配对检验同样本。"""
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

from src.training.harness.predictions import load_predictions
from src.training.harness.subgroup_auc import subgroup_auc_vector
from src.training.harness.significance import (
    mean_ci, worst_and_gap, paired_bootstrap_diff, delong_test,
    make_subgroup_metric,
)
from src.utils.operating_point_fairness import (
    to_prob, threshold_at_overall_fpr, subgroup_tpr_fpr,
    worst_group_tpr, equalized_odds_gap,
)

PRED = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/ham10000/predictions")
SEEDS = (42, 43, 44, 45, 46)
SEL = "overall"  # best_overall selection（协议头条口径）

# (标签, run 前缀, config_tag)
ARMS = {
    "ERM full":        ("erm",             "lr1e-04_wd1e-04"),  # 全微调 Pareto idx2
    "HyperAdapt full": ("hyperadapt",      "lr3e-04_wd1e-04"),  # 全微调 Pareto idx4
    "ERM frozen":      ("erm_frozen",      "lr3e-04_wd1e-04"),  # 冻结 Pareto idx4
    "HyperAdapt frozen": ("hyperadapt_frozen", "lr3e-04_wd1e-03"),  # 冻结 Pareto idx5
}


def load_seed(prefix: str, tag: str, seed: int):
    """载入单 seed test 预测，对齐到 age_group>=0 的 969 公共子集。"""
    y, s, attrs = load_predictions(PRED / f"{prefix}_{tag}_seed{seed}_{SEL}.npz")
    sex, age = attrs["sex"], attrs["age"]
    m = age >= 0
    return y[m], s[m], sex[m], age[m]


def per_seed_metrics(prefix, tag):
    """每 seed：Overall AUC / worst-group AUC / gap / worst-group TPR / Eopp。"""
    rows = {"overall": [], "worst": [], "gap": [], "wtpr": [], "eopp": []}
    for seed in SEEDS:
        y, s, sex, age = load_seed(prefix, tag, seed)
        vec = subgroup_auc_vector("ham10000", y, s, sex=sex, age=age)
        worst, gap = worst_and_gap(vec)
        rows["overall"].append(roc_auc_score(y, s))
        rows["worst"].append(worst)
        rows["gap"].append(gap)
        # 操作点：整体 FPR≈0.2 的单一阈值（比 0.5 对 14.5% 不均衡更有意义）
        prob = to_prob(s, score_is_prob=False)
        thr = threshold_at_overall_fpr(y, prob, 0.2)
        rates = subgroup_tpr_fpr("ham10000", y, prob, {"sex": sex, "age": age},
                                 thr, marginal_only=False)
        rows["wtpr"].append(worst_group_tpr(rates, min_class_n=5))
        rows["eopp"].append(equalized_odds_gap(rates, min_class_n=5)[2])  # eo_gap
    return rows


print(f"{'='*78}\n实验 F 判决：冻结 vs 全微调（HAM age，5-seed，对齐 969，selection={SEL}）\n{'='*78}")
stats = {}
for name, (prefix, tag) in ARMS.items():
    r = per_seed_metrics(prefix, tag)
    stats[name] = r
    ov, wo, gp = mean_ci(r["overall"]), mean_ci(r["worst"]), mean_ci(r["gap"])
    wt = mean_ci([x for x in r["wtpr"] if x is not None]) if any(x is not None for x in r["wtpr"]) else None
    eo = mean_ci([x for x in r["eopp"] if x is not None]) if any(x is not None for x in r["eopp"]) else None
    print(f"\n### {name}  ({prefix}_{tag})")
    print(f"  Overall AUC : {ov.mean:.4f} ±{ov.half_width:.4f}   (seeds: {[f'{v:.3f}' for v in r['overall']]})")
    print(f"  worst-group : {wo.mean:.4f} ±{wo.half_width:.4f}   (seeds: {[f'{v:.3f}' for v in r['worst']]})")
    print(f"  AUC gap     : {gp.mean:.4f} ±{gp.half_width:.4f}")
    print(f"  H1 方差(worst std) : {np.std(r['worst'], ddof=1):.4f}")
    if wt: print(f"  worst-grp TPR@0 : {wt.mean:.4f} ±{wt.half_width:.4f}")
    if eo: print(f"  Eopp gap@0      : {eo.mean:.4f} ±{eo.half_width:.4f}")


# ---- 配对显著性：HN vs ERM，冻结组 与 全微调组 各一 ----
def paired(name_hn, name_erm):
    hn_p, hn_t = ARMS[name_hn]; er_p, er_t = ARMS[name_erm]
    # 用 seed42 做配对 bootstrap（同 test 样本）；DeLong 同理
    yh, sh, sexh, ageh = load_seed(hn_p, hn_t, 42)
    ye, se, sexe, agee = load_seed(er_p, er_t, 42)
    assert len(yh) == len(ye) and np.array_equal(yh, ye), "配对样本未对齐"
    attrs = {"sex": sexh, "age": ageh}
    m_hn = make_subgroup_metric("ham10000", yh, sh, attrs, "worst")
    m_er = make_subgroup_metric("ham10000", ye, se, attrs, "worst")
    bd = paired_bootstrap_diff(len(yh), m_hn, m_er, n_boot=2000, rng=0)
    dl = delong_test(yh, sh, se)
    print(f"\n[{name_hn} − {name_erm}] (seed42, 配对)")
    print(f"  worst-group Δ = {bd.point:+.4f}  95%CI [{bd.ci_low:+.4f}, {bd.ci_high:+.4f}]  "
          f"{'✓显著' if (bd.ci_low>0 or bd.ci_high<0) else '✗跨0'}")
    print(f"  Overall DeLong Δ = {dl.diff:+.4f}  p={dl.p_value:.3f}  "
          f"{'✓显著' if dl.p_value<0.05 else '✗n.s.'}")


print(f"\n{'='*78}\n配对检验（HN vs ERM）\n{'='*78}")
paired("HyperAdapt frozen", "ERM frozen")
paired("HyperAdapt full", "ERM full")
print("\nDONE")
