"""实验 F 终判：CV-OOF 池化（HAM age，5 折 disjoint test → 全 9707 OOF，消除评估-n 地板）。
对比 冻结 vs 全微调 的 Overall / worst-group AUC / 操作点，配对 bootstrap + DeLong 出显著性。"""
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

from src.training.harness.predictions import load_predictions
from src.training.harness.subgroup_auc import subgroup_auc_vector
from src.training.harness.significance import (
    worst_and_gap, paired_bootstrap_diff, delong_test, make_subgroup_metric,
)
from src.utils.operating_point_fairness import (
    to_prob, threshold_at_overall_fpr, subgroup_tpr_fpr,
    worst_group_tpr, equalized_odds_gap,
)

CVP = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/ham10000/cv5/predictions")
FOLD_SEEDS = (42, 43, 44, 45, 46)  # fold k ↔ seed 42+k
CFG = {
    "ERM full":          ("erm",               "lr1e-04_wd1e-04"),
    "HyperAdapt full":   ("hyperadapt",        "lr3e-04_wd1e-04"),
    "ERM frozen":        ("erm_frozen",        "lr3e-04_wd1e-04"),
    "HyperAdapt frozen": ("hyperadapt_frozen", "lr3e-04_wd1e-03"),
}


def pool_oof(prefix, tag):
    """逐折读 seed=42+k 的 test 预测，过滤 age>=0，池化成全 OOF。"""
    ys, ss, sexs, ages = [], [], [], []
    for k, seed in enumerate(FOLD_SEEDS):
        y, s, at = load_predictions(CVP / f"{prefix}_{tag}_seed{seed}_overall.npz")
        m = at["age"] >= 0
        ys.append(y[m]); ss.append(s[m]); sexs.append(at["sex"][m]); ages.append(at["age"][m])
    return (np.concatenate(ys), np.concatenate(ss),
            np.concatenate(sexs), np.concatenate(ages))


pooled = {name: pool_oof(*cfg) for name, cfg in CFG.items()}
# 对齐检查：4 臂池化样本须同序（同折结构+同 age 过滤）
ref_y = pooled["ERM full"][0]
for name, (y, s, sex, age) in pooled.items():
    assert len(y) == len(ref_y) and np.array_equal(y, ref_y), f"{name} 池化未对齐"
N = len(ref_y)
print(f"{'='*80}\n实验 F 终判：CV-OOF 池化（N={N} OOF，age>=0，evaluation-n 地板已消除）\n{'='*80}")
print(f"\n{'Arm':<20}{'Overall':>9}{'worst-grp':>11}{'gap':>8}{'wTPR@.2':>9}{'Eopp':>7}")
for name, (y, s, sex, age) in pooled.items():
    ov = roc_auc_score(y, s)
    vec = subgroup_auc_vector("ham10000", y, s, sex=sex, age=age)
    worst, gap = worst_and_gap(vec)
    prob = to_prob(s, score_is_prob=False)
    thr = threshold_at_overall_fpr(y, prob, 0.2)
    rates = subgroup_tpr_fpr("ham10000", y, prob, {"sex": sex, "age": age}, thr, marginal_only=False)
    wt = worst_group_tpr(rates, min_class_n=10)
    eo = equalized_odds_gap(rates, min_class_n=10)[2]
    print(f"{name:<20}{ov:>9.4f}{worst:>11.4f}{gap:>8.4f}"
          f"{(wt if wt else float('nan')):>9.3f}{(eo if eo else float('nan')):>7.3f}")


def paired(name_hn, name_erm):
    yh, sh, sexh, ageh = pooled[name_hn]
    _, se, _, _ = pooled[name_erm]
    attrs = {"sex": sexh, "age": ageh}
    m_hn = make_subgroup_metric("ham10000", yh, sh, attrs, "worst")
    m_er = make_subgroup_metric("ham10000", yh, se, attrs, "worst")
    bd = paired_bootstrap_diff(len(yh), m_hn, m_er, n_boot=3000, rng=0)
    dl = delong_test(yh, sh, se)
    sig_w = "✓显著" if (bd.ci_low > 0 or bd.ci_high < 0) else "✗跨0"
    sig_o = "✓显著" if dl.p_value < 0.05 else "✗n.s."
    print(f"\n[{name_hn} − {name_erm}]  (池化 OOF 配对)")
    print(f"  worst-group Δ = {bd.point:+.4f}  95%CI [{bd.ci_low:+.4f}, {bd.ci_high:+.4f}]  p={bd.p_value:.3f}  {sig_w}")
    print(f"  Overall DeLong Δ = {dl.diff:+.4f}  p={dl.p_value:.3g}  {sig_o}")


print(f"\n{'='*80}\n配对显著性（HN vs 同 regime ERM，池化 OOF）\n{'='*80}")
paired("HyperAdapt frozen", "ERM frozen")
paired("HyperAdapt full", "ERM full")
# 冻结 vs 全微调 的直接对比（同 HN）
print(f"\n{'='*80}\n冻结 vs 全微调（同方法，Overall DeLong）\n{'='*80}")
for hn in ("HyperAdapt", "ERM"):
    yf, sf, _, _ = pooled[f"{hn} frozen"]; _, su, _, _ = pooled[f"{hn} full"]
    dl = delong_test(yf, sf, su)
    print(f"  {hn} frozen − full: Overall Δ={dl.diff:+.4f} p={dl.p_value:.3g}")
print("\nDONE")
