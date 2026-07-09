"""
路线 A·HAM 5 折 GroupKFold OOF 池化的 worst-group 显著性
========================================================
评审 R4.1（D6-HP）用 **seed-ensemble** 把 Overall 的 seed 方差降 ~√5，把 Overall AUC 增益证到显著；
但 worst-group 仍 n.s.，R4.2 诊断出瓶颈是**评估端子群小样本**（HAM 单-split test 仅 995，joint 小格
Female|80+ n_pos=4 进噪声地板），seed-ensemble 救不了——因为它压模型方差、不加评估样本。

本脚本是该问题的**评估侧对应解**：把 HAM 从「单 split 5-seed」改成「5 折 lesion 级 GroupKFold」重训后
（slurm/c1_ham_cv_oof.sh，产物在 outputs/ham10000/cv5/），**把 5 折 disjoint test 池化成全数据集 OOF**
（每样本恰好被其留出折预测一次，n≈9707 age 有效），少数子群评估 n 放大约 5–10×（Female|80+ n_pos 4→~40）。
在池化 OOF 上做**样本级配对 bootstrap / DeLong**——这是 seed-ensemble（降模型方差）的公平侧对应
（增评估样本），检验 worst-group 增益能否在抬起评估-n 地板后达显著。

⚠️ 与 D6/D6-HP 的机制区别：CV-OOF **无 seed-ensemble**（每样本仅 1 个 OOF 预测），显著性来自
「池化大 n + 少数格放大」。与单-split 结果完全隔离（读 cv5/predictions，不碰已冻结的 D3/D6）。

前置：slurm/c1_ham_cv_oof.sh 的 4 作业（ERM+SWAD / HyperHead / HyperFusion / HyperAdapt）5 折全跑完，
且 ROC 已派生（`python -m src.training.derive_roc --dataset ham10000 --config-tag lr1e-04_wd1e-04
--seeds 42 43 44 45 46 --output-dir outputs/ham10000/cv5`）。

用法：python scripts/ham_cv_oof_significance.py [--section perf|diag|sig|all]
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.training.harness.predictions import load_predictions
from src.training.harness.subgroup_auc import subgroup_auc_vector, subgroup_masks
from src.training.harness.significance import (
    worst_and_gap, paired_bootstrap_diff, delong_test,
)
from src.utils.operating_point_fairness import (
    to_prob, _clf_metrics, subgroup_classification_metrics,
    worst_group_clf, clf_gap, threshold_at_overall_fpr,
)

KEY = "ham10000"
CV_PDIR = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/ham10000/cv5/predictions")
FOLD_SEEDS = (42, 43, 44, 45, 46)   # fold k ↔ seed 42+k（slurm/c1_ham_cv_oof.sh 约定）

# C1.5 Pareto 选定配置（与 build_results_tables.DATASETS["ham10000"]["cfg"] 一致；CV 复用同配置）
CFG: dict[str, str] = {
    "erm": "lr1e-04_wd1e-04", "swad": "lr1e-04_wd1e-04", "roc": "lr1e-04_wd1e-04",
    "hyperhead": "lr1e-04_wd1e-03", "hyperfusion": "lr3e-04_wd1e-03", "hyperadapt": "lr3e-04_wd1e-04",
}
METHOD_ORDER = ("erm", "swad", "roc", "hyperhead", "hyperfusion", "hyperadapt")
METHOD_LABEL = {"erm": "ERM", "swad": "SWAD", "roc": "ROC",
                "hyperhead": "HyperHead", "hyperfusion": "HyperFusion", "hyperadapt": "HyperAdapt"}
HN = ("hyperhead", "hyperfusion", "hyperadapt")
BASE = ("erm", "swad", "roc")
# 分数尺度：ERM/HN/SWAD 落 logits，ROC 落已调整概率（决定 to_prob 是否先 sigmoid）
SCORE_IS_PROB = {m: (m == "roc") for m in METHOD_ORDER}


def _sel(method: str) -> str:
    """ROC/SWAD 仅落 overall 预测（派生自 ERM overall），恒 overall；HN/ERM 亦取 overall selection。"""
    return "overall"


def pool_oof(method: str) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    池化某方法 5 折 disjoint test 预测为全数据集 OOF（每样本恰好 1 个预测，来自其留出折）。

    逐折读 `<method>_<cfg>_seed{42+k}_overall.npz`，过滤 age_group>=0（与 HN 训练/单-split 口径一致，
    使基线与 HN 同 test 集、配对 bootstrap 合法），按折序拼接。ROC 存概率、余存 logits；worst/DeLong
    对单调变换不敏感，故直接拼原分数即可（不做 sigmoid，与 D6-a/c「on logits」同口径）。

    Args:
        method: erm/swad/roc/hyperhead/hyperfusion/hyperadapt。

    Returns:
        (y_true_pool [N], score_pool [N], attrs_pool {sex,age -> [N]})；N≈9707。
    """
    tag = CFG[method]
    ys, ss = [], []
    pool_attrs: dict[str, list[np.ndarray]] = {"sex": [], "age": []}
    for seed in FOLD_SEEDS:
        path = CV_PDIR / f"{method}_{tag}_seed{seed}_{_sel(method)}.npz"
        y, s, a = load_predictions(path)
        m = np.asarray(a["age"]) >= 0                 # 过滤 0-20 排除组，全方法同 test 口径
        ys.append(np.asarray(y)[m]); ss.append(np.asarray(s)[m])
        pool_attrs["sex"].append(np.asarray(a["sex"])[m])
        pool_attrs["age"].append(np.asarray(a["age"])[m])
    y = np.concatenate(ys); s = np.concatenate(ss)
    A = {k: np.concatenate(v) for k, v in pool_attrs.items()}
    return y, s, A


def load_all_pooled() -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    载入全部方法的池化 OOF，并**硬断言样本对齐**（配对 bootstrap/DeLong 的正确性前提）。

    各方法的池化按同一折序、同一 age 过滤，故 y_true 应逐元素相等；不等即样本错位，立即中止。

    Returns:
        (y_true [N], scores {method -> [N]}, attrs {sex,age -> [N]})，均以 ERM 的样本序为准。
    """
    y_ref, _, attrs = pool_oof("erm")
    scores: dict[str, np.ndarray] = {}
    for method in METHOD_ORDER:
        y_m, s_m, a_m = pool_oof(method)
        assert y_m.shape == y_ref.shape and np.array_equal(y_m, y_ref), (
            f"{method} 池化 OOF 与 ERM 样本不对齐（配对检验非法）：n={y_m.shape} vs {y_ref.shape}")
        assert np.array_equal(a_m["sex"], attrs["sex"]) and np.array_equal(a_m["age"], attrs["age"]), (
            f"{method} 池化 OOF 属性与 ERM 不对齐")
        scores[method] = s_m
    return y_ref, scores, attrs


def _canon_worst(y: np.ndarray, s: np.ndarray, attrs: dict[str, np.ndarray]):
    """canonical worst-group（含 joint 14 格）的 metric 闭包，供 paired_bootstrap_diff。"""
    def metric(idx: np.ndarray):
        vec = subgroup_auc_vector(KEY, y[idx], s[idx], sex=attrs["sex"][idx], age=attrs["age"][idx])
        return worst_and_gap(vec)[0]
    return metric


def _marg_worst(y: np.ndarray, s: np.ndarray, attrs: dict[str, np.ndarray]):
    """边缘 worst-group（仅 sex/age 6 组，排 joint）的 metric 闭包，隔离小格噪声。"""
    def metric(idx: np.ndarray):
        vec = subgroup_auc_vector(KEY, y[idx], s[idx], sex=attrs["sex"][idx], age=attrs["age"][idx])
        vals = [a for k, (a, _n) in vec.items() if a is not None and "|" not in k]
        return min(vals) if vals else None
    return metric


def perf_table(y: np.ndarray, scores: dict[str, np.ndarray], attrs: dict[str, np.ndarray]) -> None:
    """D-A1：池化 OOF 上 6 方法的 Overall / canonical worst / 边缘 worst / gap（单点估计，n≈全数据集）。"""
    print(f"\n### D-A1 HAM CV-OOF 性能与公平（池化全 {len(y):,} 样本，单点估计）")
    print("| 方法 | Overall AUC | canonical worst | 边缘 worst | AUC gap |")
    print("|------|-------------|-----------------|-----------|---------|")
    for method in METHOD_ORDER:
        s = scores[method]
        ov = roc_auc_score(y, s)
        vec = subgroup_auc_vector(KEY, y, s, sex=attrs["sex"], age=attrs["age"])
        cw, gap = worst_and_gap(vec)
        mvals = [a for k, (a, _n) in vec.items() if a is not None and "|" not in k]
        mw = min(mvals) if mvals else float("nan")
        print(f"| {METHOD_LABEL[method]} | {ov:.4f} | {cw:.4f} | {mw:.4f} | {gap:.4f} |")


def diag_table(y: np.ndarray, scores: dict[str, np.ndarray], attrs: dict[str, np.ndarray]) -> None:
    """
    D-A2：逐子群 n / n_pos / n_neg / AUC / bootstrap-SE 诊断（ERM 代表分数）——坐实评估-n 地板已抬起。

    与单-split R4.2（D6-diag）对照：单-split 下 3/14 joint 格进噪声地板（n_pos=3/8/4）；CV-OOF 池化后
    这些格 n 放大约 K 倍，应基本脱离地板（阈：SE≥0.10 或 min(n_pos,n_neg)<10）。
    """
    s = scores["erm"]
    masks = subgroup_masks(KEY, sex=attrs["sex"], age=attrs["age"])
    rng = np.random.default_rng(0)
    print(f"\n### D-A2 HAM CV-OOF 逐子群样本量诊断（ERM 代表分数，池化全 {len(y):,}）")
    print("| 子群 | n | n_pos | n_neg | AUC | boot-SE | 地板 |")
    print("|------|---|-------|-------|-----|---------|------|")
    for name, m in masks.items():
        yi, si = y[m], s[m]
        n, n_pos = int(m.sum()), int(yi.sum())
        n_neg = n - n_pos
        if n_pos == 0 or n_neg == 0:
            print(f"| {name} | {n} | {n_pos} | {n_neg} | N/A | N/A | ⚠️退化 |")
            continue
        auc = roc_auc_score(yi, si)
        # 格内 bootstrap SE（500 次；小格用于判是否仍在噪声地板）
        boots = []
        idx = np.arange(n)
        for _ in range(500):
            b = rng.choice(idx, size=n, replace=True)
            if 0 < yi[b].sum() < n:
                boots.append(roc_auc_score(yi[b], si[b]))
        se = float(np.std(boots)) if boots else float("nan")
        floor = "⚠️噪声地板" if (se >= 0.10 or min(n_pos, n_neg) < 10) else ""
        print(f"| {name} | {n} | {n_pos} | {n_neg} | {auc:.3f} | {se:.3f} | {floor} |")


def sig_tables(y: np.ndarray, scores: dict[str, np.ndarray], attrs: dict[str, np.ndarray]) -> None:
    """D-A3/D-A4：池化 OOF 上 worst-group 样本级配对 bootstrap + Overall DeLong（HN vs 3 基线）。"""
    n = len(y)
    rng = np.random.default_rng(0)

    # (a) canonical worst-group（含 joint 小格）配对 bootstrap
    print(f"\n### D-A3a HAM CV-OOF canonical worst-group 样本级配对 bootstrap（HN − 基线，n={n:,}，2000 次）")
    print("| HN \\ 基线 | vs ERM | vs SWAD | vs ROC |")
    print("|-----------|--------|---------|--------|")
    for hn in HN:
        cells = []
        for base in BASE:
            d = paired_bootstrap_diff(n, _canon_worst(y, scores[hn], attrs),
                                      _canon_worst(y, scores[base], attrs), n_boot=2000, rng=rng)
            cells.append(f"{d.point:+.3f} [{d.ci_low:+.3f},{d.ci_high:+.3f}] "
                         f"{'**显著**' if d.significant else 'n.s.'}")
        print(f"| {METHOD_LABEL[hn]} | {cells[0]} | {cells[1]} | {cells[2]} |")

    # (b) 边缘 worst-group（隔离 joint 噪声）配对 bootstrap vs 3 基线
    print(f"\n### D-A3b HAM CV-OOF 边缘 worst-group 样本级配对 bootstrap（HN − 基线，仅 sex/age 6 组，n={n:,}）")
    print("| HN \\ 基线 | vs ERM | vs SWAD | vs ROC |")
    print("|-----------|--------|---------|--------|")
    for hn in HN:
        cells = []
        for base in BASE:
            d = paired_bootstrap_diff(n, _marg_worst(y, scores[hn], attrs),
                                      _marg_worst(y, scores[base], attrs), n_boot=2000, rng=rng)
            cells.append(f"{d.point:+.4f} [{d.ci_low:+.4f},{d.ci_high:+.4f}] "
                         f"{'**显著**' if d.significant else 'n.s.'}")
        print(f"| {METHOD_LABEL[hn]} | {cells[0]} | {cells[1]} | {cells[2]} |")

    # (c) Overall AUC DeLong（池化 OOF 单次，非逐 seed 平均）
    print(f"\n### D-A4 HAM CV-OOF Overall AUC DeLong（HN vs 基线，池化 OOF n={n:,}）")
    print("| HN \\ 基线 | vs ERM | vs SWAD | vs ROC |")
    print("|-----------|--------|---------|--------|")
    for hn in HN:
        cells = []
        for base in BASE:
            r = delong_test(y, scores[hn], scores[base])
            cells.append(f"Δ={r.diff:+.4f} (p={r.p_value:.4f}{', **显著**' if r.significant else ''})")
        print(f"| {METHOD_LABEL[hn]} | {cells[0]} | {cells[1]} | {cells[2]} |")


def _fmt(v: float | None) -> str:
    """None（该组缺对应样本/退化）显示 N/A，否则四位小数。"""
    return "N/A" if v is None else f"{v:.4f}"


def clinical_tables(y: np.ndarray, scores: dict[str, np.ndarray], attrs: dict[str, np.ndarray]) -> None:
    """
    D-A5/D-A6：临床四联指标 sensitivity/specificity/PPV/NPV（阈值相关，复用 operating_point_fairness）。

    两个操作点（对齐 R2/D8 口径）：① 原生阈值 prob=0.5（logits 的 0，ROC 的锚点）；② 固定整体
    FPR=0.2（specificity≈0.8，跨方法可比——不均衡任务下 0.5 阈值几乎全判负、sensitivity 失真）。
    正类=malignant（阳性率 0.148）。Overall 一行 + worst-group（边缘 6 组，min_class_n=10）公平视图。
    """
    for target_desc, get_thr in (
        ("原生阈值 prob=0.5", None),
        ("固定整体 FPR=0.2（specificity≈0.8，跨方法可比）", 0.2),
    ):
        # ---- Overall 四联 ----
        print(f"\n### D-A5 HAM CV-OOF 整体临床四联指标（{target_desc}，正类=malignant，n={len(y):,}）")
        print("| 方法 | Sensitivity | Specificity | PPV | NPV | 用阈值 |")
        print("|------|-------------|-------------|-----|-----|--------|")
        for method in METHOD_ORDER:
            prob = to_prob(scores[method], SCORE_IS_PROB[method])
            thr = 0.5 if get_thr is None else threshold_at_overall_fpr(y, prob, get_thr)
            m = _clf_metrics(y, (prob >= thr).astype(int))
            thr_show = "0.5(prob)" if get_thr is None else f"p={thr:.3f}"
            print(f"| {METHOD_LABEL[method]} | {_fmt(m.sensitivity)} | {_fmt(m.specificity)} | "
                  f"{_fmt(m.ppv)} | {_fmt(m.npv)} | {thr_show} |")

        # ---- worst-group 公平视图（边缘 6 组：sex/age）----
        print(f"\n### D-A6 HAM CV-OOF worst-group 临床指标（{target_desc}，边缘 6 组，min_class_n=10）")
        print("| 方法 | worst Sens | Sens gap | worst Spec | Spec gap | worst PPV | worst NPV |")
        print("|------|-----------|----------|-----------|----------|-----------|-----------|")
        for method in METHOD_ORDER:
            prob = to_prob(scores[method], SCORE_IS_PROB[method])
            thr = 0.5 if get_thr is None else threshold_at_overall_fpr(y, prob, get_thr)
            rates = subgroup_classification_metrics(KEY, y, prob, attrs, thr, marginal_only=True)
            print(f"| {METHOD_LABEL[method]} | {_fmt(worst_group_clf(rates, 'sensitivity'))} | "
                  f"{_fmt(clf_gap(rates, 'sensitivity'))} | {_fmt(worst_group_clf(rates, 'specificity'))} | "
                  f"{_fmt(clf_gap(rates, 'specificity'))} | {_fmt(worst_group_clf(rates, 'ppv'))} | "
                  f"{_fmt(worst_group_clf(rates, 'npv'))} |")


def main() -> None:
    parser = argparse.ArgumentParser(description="HAM 5 折 CV-OOF worst-group 显著性（路线 A）")
    parser.add_argument("--section", choices=("perf", "diag", "sig", "clinical", "all"), default="all")
    args = parser.parse_args()

    y, scores, attrs = load_all_pooled()
    print(f"[CV-OOF 池化] n={len(y):,}  阳性率={y.mean():.3f}  "
          f"（{len(FOLD_SEEDS)} 折 disjoint test，全方法样本对齐已断言）")
    if args.section in ("perf", "all"):
        perf_table(y, scores, attrs)
    if args.section in ("diag", "all"):
        diag_table(y, scores, attrs)
    if args.section in ("sig", "all"):
        sig_tables(y, scores, attrs)
    if args.section in ("clinical", "all"):
        clinical_tables(y, scores, attrs)


if __name__ == "__main__":
    main()
