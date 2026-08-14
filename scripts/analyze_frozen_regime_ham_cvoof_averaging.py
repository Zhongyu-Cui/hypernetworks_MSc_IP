"""实验 F 终判（**averaging 口径**）：CV-OOF averaging（HAM age，5 折 disjoint test）。

把 `analyze_frozen_regime_ham_cvoof.py` 从 **pooled**（跨折池化 raw logit + DeLong）改为 **averaging**
（逐折算指标 → 折间平均；worst-group AUC = `min_g[mean_f AUC(g,f)]`）。**零重训**——同一批逐折预测，
仅改聚合。改口径原因见 [[pooled-oof-calibration-drift-artifact]]：各折不同模型、早停轮数异质使 logit 尺度
漂移，池化对排序指标注入系统性负偏（HN 受害重于 ERM），旧口径曾把 HN 的 worst-group 压出假性劣势。

- **AUC 显著性**：配对 **lesion 级 cluster bootstrap**（B=3000），逐次重采样 lesion→重算 averaging 估计量
  → 取 Δ 的 95%CI（口径与 `build_oof_results_averaging.py` 一致；DeLong 是池化 AUC 检验、averaging 下弃用）。
- **worst-group** 主口径 = **canonical**（含 sex×age joint，对齐原脚本 `worst_and_gap`）；附 **marginal**（更稳）。
- **wTPR@.2 / Eopp**：操作点指标，逐折各自校准 FPR=0.2 → 算 worst-group TPR / Eopp gap → 折间平均（点估计，
  原脚本亦未对其做配对显著性）。
"""
from __future__ import annotations

import numpy as np

from scripts.build_oof_results_averaging import Spec, load_folds, clusters, Views
from src.utils.operating_point_fairness import (
    to_prob, threshold_at_overall_fpr, subgroup_tpr_fpr, worst_group_tpr, equalized_odds_gap,
)

SPEC = Spec("HAM10000", "ham10000/cv5/predictions", "ham10000", "ham10000/cv5",
            "ham10000", "lesion_id", True)
ARMS = {
    "ERM full":          ("erm",               "lr1e-04_wd1e-04"),
    "HyperAdapt full":   ("hyperadapt",        "lr3e-04_wd1e-04"),
    "ERM frozen":        ("erm_frozen",        "lr3e-04_wd1e-04"),
    "HyperAdapt frozen": ("hyperadapt_frozen", "lr3e-04_wd1e-03"),
}
N_BOOT = 3000
# Views.compute 返回 (Overall, canonical_worst, marginal_worst, gap_canonical, undefined_fold_cells)
IDX = {"Overall": 0, "canonical worst": 1, "marginal worst": 2, "gap": 3}


def op_point_averaging(folds: list[dict]) -> tuple[float, float]:
    """逐折自校准 FPR=0.2 → worst-group TPR / Eopp gap（含 joint）→ 折间平均。"""
    wts, eos = [], []
    for f in folds:
        prob = to_prob(f["s"], score_is_prob=False)
        thr = threshold_at_overall_fpr(f["y"], prob, 0.2)
        rates = subgroup_tpr_fpr("ham10000", f["y"], prob,
                                 {"sex": f["attrs"]["sex"], "age": f["attrs"]["age"]},
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
    cidx, offs = clusters(SPEC, folds_by["ERM full"])
    n_cl = int(cidx.max() + 1)
    ones = np.ones(offs[-1])

    obs = {name: views[name].compute(offs, ones) for name in views}
    op = {name: op_point_averaging(fl) for name, fl in folds_by.items()}
    N = int(offs[-1])

    print(f"{'=' * 92}\n实验 F 终判（averaging 口径）：CV-OOF averaging（N={N} OOF，age>=0，lesion 聚类={n_cl}）\n{'=' * 92}")
    print(f"\n{'Arm':<20}{'Overall':>9}{'canon worst':>12}{'marg worst':>11}{'gap':>8}{'wTPR@.2':>9}{'Eopp':>7}")
    for name in views:
        ov, cw, mw, gp, _undef = obs[name]
        wt, eo = op[name]
        print(f"{name:<20}{ov:>9.4f}{cw:>12.4f}{mw:>11.4f}{gp:>8.4f}{wt:>9.3f}{eo:>7.3f}")

    # 配对 cluster bootstrap（lesion 级）
    rng = np.random.default_rng(0)
    boot = {name: [] for name in views}
    for _ in range(N_BOOT):
        draw = rng.integers(0, n_cl, size=n_cl)
        w = np.bincount(draw, minlength=n_cl).astype(float)[cidx]
        for name in views:
            boot[name].append(views[name].compute(offs, w)[:4])
    B = {name: np.asarray(boot[name]) for name in views}

    def paired(hn: str, erm: str) -> None:
        print(f"\n[{hn} − {erm}]  (averaging + lesion cluster bootstrap B={N_BOOT})")
        for key in ("canonical worst", "marginal worst", "Overall"):
            j = IDX[key]
            d = B[hn][:, j] - B[erm][:, j]
            lo, hi = np.percentile(d, [2.5, 97.5])
            delta = obs[hn][j] - obs[erm][j]
            p = 2.0 * min((d < 0).mean(), (d > 0).mean())
            sig = "✓显著" if (lo > 0 or hi < 0) else "✗n.s."
            print(f"  {key:<16} Δ={delta:+.4f}  95%CI [{lo:+.4f}, {hi:+.4f}]  p={p:.3f}  {sig}")

    print(f"\n{'=' * 92}\n配对显著性（HN vs 同 regime ERM）\n{'=' * 92}")
    paired("HyperAdapt frozen", "ERM frozen")
    paired("HyperAdapt full", "ERM full")

    print(f"\n{'=' * 92}\n冻结 vs 全微调（同方法，Overall averaging Δ + cluster bootstrap）\n{'=' * 92}")
    for hn in ("HyperAdapt", "ERM"):
        j = IDX["Overall"]
        d = B[f"{hn} frozen"][:, j] - B[f"{hn} full"][:, j]
        lo, hi = np.percentile(d, [2.5, 97.5])
        delta = obs[f"{hn} frozen"][j] - obs[f"{hn} full"][j]
        sig = "✓显著" if (lo > 0 or hi < 0) else "✗n.s."
        print(f"  {hn} frozen − full: Overall Δ={delta:+.4f}  95%CI [{lo:+.4f}, {hi:+.4f}]  {sig}")
    print("\nDONE")


if __name__ == "__main__":
    main()
