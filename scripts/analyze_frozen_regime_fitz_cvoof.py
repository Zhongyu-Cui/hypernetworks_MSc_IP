"""实验 F·Fitzpatrick 终判：CV-OOF 池化（5 折 disjoint → 全 16012 OOF，肤色 VI 55→635）。
冻结 HA vs 冻结 ERM = 论文「Ours vs Vanilla」在 Fitzpatrick 的直接对应（论文亦冻结 backbone）。
逐肤色 AUC + worst-group + 操作点(recall/Eopp) + 配对 bootstrap/DeLong，对标论文 Fig3/Table。"""
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

from src.training.harness.predictions import load_predictions
from src.training.harness.subgroup_auc import subgroup_auc_vector, vector_worst_case
from src.training.harness.significance import (
    worst_and_gap, paired_bootstrap_diff, delong_test, make_subgroup_metric,
)
from src.utils.operating_point_fairness import (
    to_prob, threshold_at_overall_fpr, subgroup_tpr_fpr, worst_group_tpr, equalized_odds_gap,
)

CVP = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/fitzpatrick/cv5/predictions")
FOLD_SEEDS = (42, 43, 44, 45, 46)
# 论文分析用冻结 regime（论文 backbone 亦冻结）；full-finetune 无 cv5 preds，仅单-split 有
ARMS = {
    "ERM frozen":        ("erm_frozen",        "lr3e-04_wd1e-03"),
    "HyperAdapt frozen": ("hyperadapt_frozen", "lr1e-04_wd1e-04"),
}
SKIN_NAMES = ["I", "II", "III", "IV", "V", "VI"]


def pool(prefix, tag):
    ys, ss, sk = [], [], []
    for k, seed in enumerate(FOLD_SEEDS):
        y, s, at = load_predictions(CVP / f"{prefix}_{tag}_seed{seed}_overall.npz")
        ys.append(y); ss.append(s); sk.append(at["skin"])
    return np.concatenate(ys), np.concatenate(ss), np.concatenate(sk)


pooled = {name: pool(*cfg) for name, cfg in ARMS.items()}
ref_y = pooled["ERM frozen"][0]
for name, (y, s, sk) in pooled.items():
    assert len(y) == len(ref_y) and np.array_equal(y, ref_y), f"{name} 未对齐"
N = len(ref_y)
print(f"{'='*82}\n实验 F·Fitzpatrick 终判：CV-OOF 池化（N={N}，肤色 VI 评估 n≈635，地板已消除）\n"
      f"冻结 HA vs 冻结 ERM = 论文 Ours vs Vanilla 直接对应\n{'='*82}")

print(f"\n{'Arm':<20}{'Overall':>9}{'worst':>8}{'gap':>7}{'wTPR@.2':>9}{'Eopp':>7}  逐肤色 AUC(I..VI)")
for name, (y, s, sk) in pooled.items():
    ov = roc_auc_score(y, s)
    vec = subgroup_auc_vector("fitzpatrick", y, s, skin=sk)
    worst, gap = worst_and_gap(vec)
    prob = to_prob(s, score_is_prob=False)
    thr = threshold_at_overall_fpr(y, prob, 0.2)
    rates = subgroup_tpr_fpr("fitzpatrick", y, prob, {"skin": sk}, thr, marginal_only=False)
    wt = worst_group_tpr(rates, min_class_n=10)
    eo = equalized_odds_gap(rates, min_class_n=10)[2]
    aucs = [v[0] for v in vec.values()]  # OrderedDict[skin -> (auc, n)]，取 auc 按 I..VI 序
    per = "  ".join(f"{SKIN_NAMES[i]}:{aucs[i]:.3f}" if aucs[i] is not None else f"{SKIN_NAMES[i]}:--"
                    for i in range(6))
    print(f"{name:<20}{ov:>9.4f}{worst:>8.4f}{gap:>7.4f}{wt:>9.3f}{eo:>7.3f}  {per}")

# 配对：冻结 HA vs 冻结 ERM（论文对应）
yh, sh, skh = pooled["HyperAdapt frozen"]
_, se, _ = pooled["ERM frozen"]
attrs = {"skin": skh}
mh = make_subgroup_metric("fitzpatrick", yh, sh, attrs, "worst")
me = make_subgroup_metric("fitzpatrick", yh, se, attrs, "worst")
bd = paired_bootstrap_diff(len(yh), mh, me, n_boot=3000, rng=0)
dl = delong_test(yh, sh, se)
print(f"\n{'='*82}\n配对显著性（冻结 HyperAdapt − 冻结 ERM，池化 OOF）\n{'='*82}")
print(f"  worst-group Δ = {bd.point:+.4f}  95%CI [{bd.ci_low:+.4f}, {bd.ci_high:+.4f}]  p={bd.p_value:.3f}  "
      f"{'✓显著' if (bd.ci_low>0 or bd.ci_high<0) else '✗跨0'}")
print(f"  Overall DeLong Δ = {dl.diff:+.4f}  p={dl.p_value:.3g}  {'✓显著' if dl.p_value<0.05 else '✗n.s.'}")
print("\nDONE")
