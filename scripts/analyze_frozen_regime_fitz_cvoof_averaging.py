"""实验 F·Fitzpatrick 终判（**averaging 口径**）：CV-OOF averaging（skin，5 折 disjoint）。

把 `analyze_frozen_regime_fitz_cvoof.py` 从 **pooled**（跨折池化 raw logit + DeLong）改为 **averaging**
（逐折算 → 折间平均；worst-group AUC = `min_g[mean_f AUC(g,f)]`）。**零重训**。冻结 HA vs 冻结 ERM =
论文「Ours vs Vanilla」在 Fitzpatrick 的直接对应（论文亦冻结 backbone）。改口径原因见
[[pooled-oof-calibration-drift-artifact]]。

- **AUC 显著性**：配对**样本级 bootstrap**（Fitz 每 md5hash 唯一、无聚类结构，B=3000），重采样→重算
  averaging 估计量→Δ 的 95%CI（口径与 `build_oof_results_averaging.py` 一致；DeLong 弃用）。
- **worst-group** = skin 单轴（Fitz 无 joint），canonical≡marginal；逐肤色 AUC 亦折均。
- **wTPR@.2 / Eopp**：逐折自校准 FPR=0.2 → worst-group TPR / Eopp gap → 折间平均（点估计）。
"""
from __future__ import annotations

import numpy as np

from scripts.build_oof_results_averaging import Spec, load_folds, clusters, Views, build_masks
from src.utils.operating_point_fairness import (
    to_prob, threshold_at_overall_fpr, subgroup_tpr_fpr, worst_group_tpr, equalized_odds_gap,
)

SPEC = Spec("Fitzpatrick", "fitzpatrick/cv5/predictions", "fitzpatrick", "fitzpatrick/cv5",
            None, None, False)
ARMS = {
    "ERM frozen":        ("erm_frozen",        "lr3e-04_wd1e-03"),   # 论文 Vanilla
    "HyperAdapt frozen": ("hyperadapt_frozen", "lr1e-04_wd1e-04"),   # 论文 Ours
}
SKIN_NAMES = ["skin:I", "skin:II", "skin:III", "skin:IV", "skin:V", "skin:VI"]
N_BOOT = 3000
IDX = {"Overall": 0, "worst": 1, "gap": 3}   # Views.compute: (Overall, canon_worst, marg_worst, gap, undef)


def per_skin_auc_averaging(folds: list[dict]) -> dict[str, float]:
    """各肤色 AUC 逐折算 → 折间平均（复用 Views 的 group 键，取 marginal skin:*）。"""
    from scripts.eaudit_m2c_full_target import sorted_view, weighted_auc
    per: dict[str, list[float]] = {}
    for f in folds:
        masks = build_masks(SPEC, f["attrs"])
        for g, m in masks.items():
            i, ys = sorted_view(f["y"], f["s"], m)
            a = weighted_auc(ys, np.ones(m.sum()))
            if not np.isnan(a):
                per.setdefault(g, []).append(a)
    return {g: float(np.mean(v)) for g, v in per.items() if v}


def op_point_averaging(folds: list[dict]) -> tuple[float, float]:
    """逐折自校准 FPR=0.2 → worst-group TPR / Eopp gap → 折间平均。"""
    wts, eos = [], []
    for f in folds:
        prob = to_prob(f["s"], score_is_prob=False)
        thr = threshold_at_overall_fpr(f["y"], prob, 0.2)
        rates = subgroup_tpr_fpr("fitzpatrick", f["y"], prob, {"skin": f["attrs"]["skin"]},
                                 thr, marginal_only=False)
        wt = worst_group_tpr(rates, min_class_n=10)
        eo = equalized_odds_gap(rates, min_class_n=10)[2]
        if wt is not None:
            wts.append(wt)
        if eo is not None:
            eos.append(eo)
    return (float(np.mean(wts)) if wts else float("nan"),
            float(np.mean(eos)) if eos else float("nan"))


def main() -> None:
    folds_by = {name: load_folds(SPEC, m, t) for name, (m, t) in ARMS.items()}
    for name, fl in folds_by.items():
        assert fl is not None, f"{name} 预测缺失"
    views = {name: Views(SPEC, fl) for name, fl in folds_by.items()}
    cidx, offs = clusters(SPEC, folds_by["ERM frozen"])
    n_cl = int(cidx.max() + 1)
    ones = np.ones(offs[-1])

    obs = {name: views[name].compute(offs, ones) for name in views}
    op = {name: op_point_averaging(fl) for name, fl in folds_by.items()}
    skin = {name: per_skin_auc_averaging(fl) for name, fl in folds_by.items()}
    N = int(offs[-1])

    print(f"{'=' * 96}\n实验 F·Fitzpatrick 终判（averaging 口径）：CV-OOF averaging（N={N}，肤色 VI 折均评估）\n"
          f"冻结 HA vs 冻结 ERM = 论文 Ours vs Vanilla 直接对应\n{'=' * 96}")
    print(f"\n{'Arm':<20}{'Overall':>9}{'worst':>8}{'gap':>7}{'wTPR@.2':>9}{'Eopp':>7}  逐肤色 AUC(I..VI)")
    for name in views:
        ov, cw, _mw, gp, _undef = obs[name]
        wt, eo = op[name]
        per = "  ".join(f"{SKIN_NAMES[i].split(':')[1]}:{skin[name].get(SKIN_NAMES[i], float('nan')):.3f}"
                        for i in range(6))
        print(f"{name:<20}{ov:>9.4f}{cw:>8.4f}{gp:>7.4f}{wt:>9.3f}{eo:>7.3f}  {per}")

    # 配对样本级 bootstrap（冻结 HA − 冻结 ERM）
    rng = np.random.default_rng(0)
    boot = {name: [] for name in views}
    for _ in range(N_BOOT):
        draw = rng.integers(0, n_cl, size=n_cl)
        w = np.bincount(draw, minlength=n_cl).astype(float)[cidx]
        for name in views:
            boot[name].append(views[name].compute(offs, w)[:4])
    B = {name: np.asarray(boot[name]) for name in views}

    print(f"\n{'=' * 96}\n配对显著性（冻结 HyperAdapt − 冻结 ERM，averaging + 样本级 bootstrap B={N_BOOT}）\n{'=' * 96}")
    for key in ("worst", "Overall"):
        j = IDX[key]
        d = B["HyperAdapt frozen"][:, j] - B["ERM frozen"][:, j]
        lo, hi = np.percentile(d, [2.5, 97.5])
        delta = obs["HyperAdapt frozen"][j] - obs["ERM frozen"][j]
        p = 2.0 * min((d < 0).mean(), (d > 0).mean())
        sig = "✓显著" if (lo > 0 or hi < 0) else "✗n.s."
        label = "worst-group AUC" if key == "worst" else "Overall AUC"
        print(f"  {label:<16} Δ={delta:+.4f}  95%CI [{lo:+.4f}, {hi:+.4f}]  p={p:.3f}  {sig}")
    # gap 点估计变化
    ge, gh = obs["ERM frozen"][IDX["gap"]], obs["HyperAdapt frozen"][IDX["gap"]]
    print(f"  AUC gap          {ge:.4f} → {gh:.4f}（{'变宽' if gh > ge else '收窄'}）")
    print("\nDONE")


if __name__ == "__main__":
    main()
