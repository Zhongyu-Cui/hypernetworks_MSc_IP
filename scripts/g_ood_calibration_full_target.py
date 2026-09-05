"""
实验 G · 未决项 1：OOD 校准分析扩到 full-target × 15 replicate（补 σ_train 检验）
================================================================================
`docs/hyperadapt_swad_fusion_full_report.md` §5.3 记录了本实验**唯一方向一致的正面读数**——
融合臂（HyperAdapt+SWAD）修复了 v2 记录的 HyperAdapt OOD 校准劣化，且 SWAD 的 ECE 改善向 HN
的传递在两方向上量级一致（差同为 +0.0012）。但该读数当时**不构成结论**，原因有二：

  1. 取数为 `cv5/predictions`（**配折**产物）且只有 trial 0，即 **5 replicate**；
     而 v2 的 OOD 主口径是 **full-target 估计量 × 5 fold × 3 trial = 15 replicate**。
  2. **未做 σ_train 检验**。v2 已证 cluster bootstrap 的 CI 只含**样本**不确定性、不含**训练**
     随机性；在 |Δ| ~ 1e-3 ~ 1e-2 的尺度上训练随机性才是主导项，故窄 CI 不可作判据。

本脚本用**与 AUC 主分析逐位相同**的口径重做校准：full-target 取数、15 replicate、
逐 (fold, trial) 配对差的 fold×trial 双因子方差分解 → σ_train，判据 |d̄| > σ_train。
**无需再训练**——所需预测已由作业 75216/75226/75240 全部落盘。

**与 `ood_calibration.py` 的关系**：该脚本（v2 D2 / H3）**不改动**，其 cv5 口径结果继续有效；
本脚本是估计量升级后的**平行产物**，共用其指标实现（`calibration_metrics` / `cross_fit_isotonic`）
与配色/标签，共用 `ood_variance_decomposition.py` 的 `anova_two_way` / `seed_of` / `holm`。

**指标与判读方向**（全部「越小越好」，故 Δ<0 = 校准更好）：
  · `abs_citl` = |CITL| = |mean(p̂) − mean(y)|：CITL 是**有符号**量，其 Δ 的符号不代表优劣，
    校准好坏须看是否更接近 0，故检验对象取绝对值。
  · `ece`（**主要终点**，等频 10 箱）、`brier`、`abs_slope_err` = |slope − 1|。

**两条不确定性**（与 AUC 主分析同构）：
  · **H1 样本侧**：患者级配对 cluster bootstrap，全方法共用同一批重采样索引；
    分箱边界由**原始**分数预先定死，使每次重采样落在同一组箱内（否则 ECE 不可比）。
  · **H2 训练侧**：对配对差矩阵 d_{f,t}（5×3）做 ANOVA，σ_train = √(σ²_trial + σ²_e)。
    **这是判据**；H1 仅作参照。

**六个对比**（每方向）：三个 vs ERM + 融合臂 vs SWAD / vs HyperAdapt +
**交互项** Δ_int = (HA+SWAD − HA) − (SWAD − ERM)，即「SWAD 的校准收益在 HN 基座上是否更大」，
与 §2.2 的 AUC 交互项定义代数等价。

**Holm 校正**：family = 4 个对比（3 个 vs ERM + Δ_int）× 2 方向 = 8，仅对**主要终点 ECE** 施加；
Holm 判的是 H1（样本侧），**不替代** σ_train。

运行（轻量 CPU，无需 GPU；full-target n≈14–20 万 × 60 cell，约 10–20 分钟）：
    python scripts/g_ood_calibration_full_target.py
    python scripts/g_ood_calibration_full_target.py --direction M2C --n-boot 200
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from scripts.eaudit_m2c_full_target import sorted_view, weighted_auc
# 复用 v2 校准脚本的指标实现与配色（import 时其已完成中文字体注册，此处不重复设置）
from scripts.ood_calibration import (
    COLORS, DIRECTIONS, INK, LABELS, METHODS, MUTED, N_BINS,
    calibration_metrics, cross_fit_isotonic, sigmoid,
)
from scripts.ood_variance_decomposition import N_FOLDS, TRIALS, anova_two_way, holm, seed_of
from src.training.harness.subgroup_auc import subgroup_masks

OUTPUTS = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs")
OUT_DIR = OUTPUTS / "ood_cxr" / "calibration_g_full_target"
# --bnupd：把两个**权重平均派生**臂的取数换到 BN 与官方 SWAD 对齐（update_bn 重估）后的预测。
# ERM / HyperAdapt 不是平均权重、不受 BN 重估影响，故仍读原目录 ⇒ 对照臂逐位不变。
BNUPD_METHODS = ("swad", "hyperadapt_swad")
BNUPD_OUT_DIR = OUTPUTS / "ood_cxr" / "calibration_g_full_target_bnupd"

# 检验对象（全部越小越好）；ece 为主要终点
CAL_KEYS = ("abs_citl", "ece", "brier", "abs_slope_err")
PRIMARY_KEY = "ece"
# bootstrap 只覆盖能被样本权重快速重算的三项（slope 需每次重拟合加权 logistic，代价过高）
BOOT_KEYS = ("abs_citl", "ece", "brier")

# 对比：(名称, 正项, 负项) —— Δ_int 另行处理（四臂线性组合）
CONTRASTS = (
    ("swad_vs_erm", "swad", "erm"),
    ("hyperadapt_vs_erm", "hyperadapt", "erm"),
    ("hyperadapt_swad_vs_erm", "hyperadapt_swad", "erm"),
    ("hyperadapt_swad_vs_swad", "hyperadapt_swad", "swad"),
    ("hyperadapt_swad_vs_hyperadapt", "hyperadapt_swad", "hyperadapt"),
    # OOD v2 偏离 #3 的 GroupDRO 第四臂：校准**同口径纳入但只作描述性报告**，
    # 故进 CONTRASTS 而**不进 HOLM_KEYS**——family 保持实验 G 的 8 个检验不变，
    # 既有 G 结论因此逐位不受影响。
    ("groupdro_vs_erm", "groupdro", "erm"),
    # OOD v2 偏离 #4 的 HyperHead / HyperFusion 两臂：与 GroupDRO 同一处置——校准同口径纳入
    # 但**只作描述性报告**，进 CONTRASTS 而不进 HOLM_KEYS，故实验 G 的 8 检验 family 逐位不变。
    ("hyperhead_vs_erm", "hyperhead", "erm"),
    ("hyperfusion_vs_erm", "hyperfusion", "erm"),
)
INTERACTION = "interaction_dint"
# Holm family 只含「3 个 vs ERM + 交互项」（与 AUC 主分析的 family 构造精神一致）
HOLM_KEYS = ("swad_vs_erm", "hyperadapt_vs_erm", "hyperadapt_swad_vs_erm", INTERACTION)


def cell_metrics(y: np.ndarray, score: np.ndarray, groups: dict[str, np.ndarray],
                 recal_seed: int) -> dict:
    """
    单个 (方法, fold, trial) 在**完整 target** 上的校准量与 bootstrap 预计算视图。

    Args:
        y: [N] 0/1 标签（全 cell 共享，调用方已断言一致）。
        score: [N] 该 cell 的 logits。
        groups: {marginal 子群名: [N] bool 掩码}，用于 worst-group AUC 的重校准不变性核查。
        recal_seed: 交叉拟合 isotonic 的 KFold 种子。

    Returns:
        {"raw", "recalibrated": 指标字典；"p", "bin_idx", "sq_err": bootstrap 预计算数组；
         "marginal_worst_auc_raw" / "_recal"}。
    """
    p = sigmoid(score)
    p_cal = cross_fit_isotonic(y.astype(np.float64), p, seed=recal_seed)

    raw = calibration_metrics(y, p)
    recal = calibration_metrics(y, p_cal)
    for d in (raw, recal):
        d["abs_citl"] = abs(d["citl"])
        # slope 退化时（见 ood_calibration.SLOPE_SANE_MAX）保持 nan，不填充假值
        d["abs_slope_err"] = abs(d["slope"] - 1.0) if np.isfinite(d["slope"]) else float("nan")

    # 分箱边界由**原始**分数预先定死：bootstrap 各 replicate 落在同一组箱内，ECE 才可比
    edges = np.unique(np.quantile(p, np.linspace(0, 1, N_BINS + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    bin_idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)

    # isotonic 是单调映射 ⇒ 理论上不改判别；此处实测以核查（并列分数会带来极小偏差）
    ones = np.ones(len(y))
    worst_raw, worst_cal = [], []
    for m in groups.values():
        for src, sink in ((score, worst_raw), (p_cal, worst_cal)):
            i, ys = sorted_view(y, np.asarray(src, dtype=np.float64), m)
            v = weighted_auc(ys, ones[i])
            if not np.isnan(v):
                sink.append(v)

    return {
        "raw": raw, "recalibrated": recal,
        "p": p, "bin_idx": bin_idx.astype(np.int16), "sq_err": (p - y) ** 2,
        "n_bins_used": int(len(edges) - 1),
        "marginal_worst_auc_raw": float(min(worst_raw)) if worst_raw else float("nan"),
        "marginal_worst_auc_recal": float(min(worst_cal)) if worst_cal else float("nan"),
    }


def weighted_cell(cell: dict, w: np.ndarray, wy_total: float, wy_bin: np.ndarray,
                  w_sum: float, w_bin: np.ndarray) -> tuple[float, float, float]:
    """
    给定样本权重下该 cell 的 (|CITL|, ECE, Brier)。

    y 与 patient 聚类全 cell 共享，故与方法无关的加权量（`wy_total` / `w_sum`）由调用方算一次；
    但分箱是**逐 cell** 的（边界取自该 cell 自己的分数分位），故 `wy_bin` / `w_bin` 需按 cell 索引传入。

    Args:
        cell: `cell_metrics` 的返回值。
        w: [N] 样本权重（= 所属 patient 被抽中的次数）。
        wy_total: Σ w·y。
        wy_bin: [B] 该 cell 分箱下的 Σ w·y。
        w_sum: Σ w。
        w_bin: [B] 该 cell 分箱下的 Σ w。

    Returns:
        (|CITL|, ECE, Brier)。
    """
    p, idx = cell["p"], cell["bin_idx"]
    nb = len(w_bin)
    wp_total = float(w @ p)
    citl = wp_total / w_sum - wy_total / w_sum
    brier = float(w @ cell["sq_err"]) / w_sum
    wp_bin = np.bincount(idx, weights=w * p, minlength=nb)
    nz = w_bin > 0
    ece = float((w_bin[nz] / w_sum * np.abs(wp_bin[nz] / w_bin[nz] - wy_bin[nz] / w_bin[nz])).sum())
    return abs(citl), ece, brier


def run_direction(direction: str, n_boot: int, bnupd: bool = False) -> dict | None:
    """跑一个方向的 full-target 校准分析（点估计 + H1 bootstrap + H2 σ_train）。"""
    d = DIRECTIONS[direction]
    cfg = json.loads((OUTPUTS / d["cfg_dir"] / "cv5" /
                      "selected_configs.json").read_text())["config"]
    base_dir = OUTPUTS / "ood_cxr" / d["ood_sub"]

    def pred_dir_of(method: str) -> Path:
        """该方法的预测目录：--bnupd 下只有权重平均派生臂换到 bnupd 目录。"""
        sub = "cv5_full_target_bnupd" if (bnupd and method in BNUPD_METHODS) else "cv5_full_target"
        return base_dir / sub / "predictions"

    # --- 共享的 y / patient / 子群掩码（full-target ⇒ 全 cell 同一批样本、同一行序）---
    ref_path = pred_dir_of("erm") / f"erm_{cfg['erm']}_seed{seed_of(0, 0)}_overall.npz"
    if not ref_path.exists():
        print(f"[跳过] {direction}: 缺 {ref_path.name}")
        return None
    with np.load(ref_path, allow_pickle=True) as ref:
        y = ref["y_true"].astype(int)
        patient_id = ref["extra_patient_id"]
        attrs = {k: ref[f"attr_{k}"] for k in (str(x) for x in ref["attr_keys"])}
    kw = {k: np.asarray(v).astype(int) for k, v in attrs.items() if k in ("sex", "race", "age")}
    groups = {g: m for g, m in subgroup_masks(d["tgt_key"], **kw).items() if "|" not in g}

    # --- 逐 cell 载入与预计算 ---
    cells: dict[str, dict[tuple[int, int], dict]] = {}
    for m in METHODS:
        cells[m] = {}
        for trial in TRIALS:
            for fold in range(N_FOLDS):
                seed = seed_of(fold, trial)
                path = pred_dir_of(m) / f"{m}_{cfg[m]}_seed{seed}_overall.npz"
                if not path.exists():
                    print(f"[跳过] {direction}: 缺 {path.name}")
                    return None
                with np.load(path, allow_pickle=True) as z:
                    if not np.array_equal(z["y_true"].astype(int), y):
                        raise ValueError(f"{path.name}: y_true 与参照 cell 不一致，full-target 对齐被破坏")
                    score = z["y_score"].astype(np.float64)
                cells[m][(fold, trial)] = cell_metrics(y, score, groups, recal_seed=42)
        print(f"  [{direction}] {LABELS[m]:<16s} 15 cell 载入完成  config={cfg[m]}")

    order = [(f, t) for f in range(N_FOLDS) for t in TRIALS]

    def mat(method: str, key: str, block: str = "raw") -> np.ndarray:
        """该方法某指标的 [F, T] 矩阵。"""
        return np.array([[cells[method][(f, t)][block][key] for t in TRIALS]
                         for f in range(N_FOLDS)])

    point = {m: {b: {k: float(np.nanmean(mat(m, k, b))) for k in CAL_KEYS + ("citl", "slope")}
                 for b in ("raw", "recalibrated")} for m in METHODS}
    for m in METHODS:
        point[m]["marginal_worst_auc_raw"] = float(
            np.nanmean([cells[m][c]["marginal_worst_auc_raw"] for c in order]))
        point[m]["marginal_worst_auc_recal"] = float(
            np.nanmean([cells[m][c]["marginal_worst_auc_recal"] for c in order]))

    # --- H2：逐 (fold, trial) 配对差 → 双因子方差分解 → σ_train ---
    def diff_mat(name: str, key: str) -> np.ndarray:
        if name == INTERACTION:
            # Δ_int = (HA+SWAD − HA) − (SWAD − ERM)，与 §2.2 的 AUC 交互项代数等价
            return ((mat("hyperadapt_swad", key) - mat("hyperadapt", key))
                    - (mat("swad", key) - mat("erm", key)))
        pos, neg = next((p, n) for c, p, n in CONTRASTS if c == name)
        return mat(pos, key) - mat(neg, key)

    names = [c[0] for c in CONTRASTS] + [INTERACTION]
    h2 = {name: {k: anova_two_way(diff_mat(name, k)) for k in CAL_KEYS
                 if np.isfinite(diff_mat(name, k)).all()} for name in names}

    # --- H1：患者级配对 cluster bootstrap（全方法共用同一批重采样索引）---
    _, cidx = np.unique(patient_id, return_inverse=True)
    n_cl = int(cidx.max() + 1)
    yf = y.astype(np.float64)
    rng = np.random.default_rng(42)
    boot: dict[str, list[np.ndarray]] = {m: [] for m in METHODS}
    for _ in range(n_boot):
        draw = rng.integers(0, n_cl, size=n_cl)
        w = np.bincount(draw, minlength=n_cl).astype(float)[cidx]
        w_sum, wy_total, wy = float(w.sum()), float(w @ yf), w * yf
        # 与方法无关的分箱量仍需按 cell 的箱定义聚合，故按 (方法, cell) 内联计算
        for m in METHODS:
            rows = []
            for c in order:
                cell = cells[m][c]
                nb = cell["n_bins_used"]
                w_bin = np.bincount(cell["bin_idx"], weights=w, minlength=nb)
                wy_bin = np.bincount(cell["bin_idx"], weights=wy, minlength=nb)
                rows.append(weighted_cell(cell, w, wy_total, wy_bin, w_sum, w_bin))
            boot[m].append(np.mean(rows, axis=0))          # 15 cell 等权平均
    B = {m: np.asarray(boot[m]) for m in METHODS}          # [n_boot, 3]

    obs_boot = {m: np.array([point[m]["raw"][k] for k in BOOT_KEYS]) for m in METHODS}
    h1 = {}
    for name in names:
        if name == INTERACTION:
            dd = ((B["hyperadapt_swad"] - B["hyperadapt"]) - (B["swad"] - B["erm"]))
            obs = ((obs_boot["hyperadapt_swad"] - obs_boot["hyperadapt"])
                   - (obs_boot["swad"] - obs_boot["erm"]))
        else:
            pos, neg = next((p, n) for c, p, n in CONTRASTS if c == name)
            dd, obs = B[pos] - B[neg], obs_boot[pos] - obs_boot[neg]
        e = {}
        for j, k in enumerate(BOOT_KEYS):
            lo, hi = np.percentile(dd[:, j], [2.5, 97.5])
            se = (hi - lo) / (2 * 1.96)
            pval = float(2 * (1 - norm.cdf(abs(obs[j]) / se))) if se > 0 else 1.0
            e[k] = {"delta": float(obs[j]), "ci": [float(lo), float(hi)],
                    "sig": bool(lo > 0 or hi < 0), "p": pval}
        h1[name] = e

    # --- 描述性：逐臂 ECE 在 15 个 (fold, trial) 上的抖动（level 侧，非配对差）---
    # 配对差的 σ_train 判的是「效应是否可信」；这里判的是「哪一臂的校准对训练随机性更稳」。
    arm_var = {m: {"sd_across_cells": float(np.std(mat(m, PRIMARY_KEY), ddof=1)),
                   **anova_two_way(mat(m, PRIMARY_KEY))} for m in METHODS}

    res = {"direction": direction, "n": int(len(y)), "n_clusters": n_cl, "n_boot": n_boot,
           "n_replicates": len(order), "config": {m: cfg[m] for m in METHODS},
           "bnupd": bool(bnupd),
           "pred_dirs": {m: str(pred_dir_of(m)) for m in METHODS},
           "point": point, "arm_ece_variance": arm_var,
           "per_cell": {m: {k: mat(m, k).tolist() for k in CAL_KEYS} for m in METHODS},
           "h1_bootstrap": h1, "h2_variance": h2}
    del cells
    gc.collect()
    return res


def print_direction(direction: str, r: dict) -> None:
    """打印一个方向的点估计表与两条不确定性判据。"""
    d = DIRECTIONS[direction]
    print(f"\n{'=' * 118}\n{d['title']}   full-target n={r['n']:,}（患者 {r['n_clusters']:,}）  "
          f"{r['n_replicates']} replicate = {N_FOLDS} fold × {len(TRIALS)} trial\n{'=' * 118}")
    print(f"{'方法':<16s} {'CITL':>9s} {'|CITL|':>8s} {'slope':>8s} {'ECE':>8s} {'Brier':>8s}"
          f" | {'ECE(重校准)':>12s} {'Brier(重校准)':>14s} | {'margWorst':>10s} {'重校准后':>10s}")
    for m in METHODS:
        p = r["point"][m]
        print(f"{LABELS[m]:<16s} {p['raw']['citl']:>+9.4f} {p['raw']['abs_citl']:>8.4f} "
              f"{p['raw']['slope']:>8.3f} {p['raw']['ece']:>8.4f} {p['raw']['brier']:>8.4f} | "
              f"{p['recalibrated']['ece']:>12.4f} {p['recalibrated']['brier']:>14.4f} | "
              f"{p['marginal_worst_auc_raw']:>10.4f} {p['marginal_worst_auc_recal']:>10.4f}")

    print(f"\n  逐臂 ECE 的训练间抖动（描述性；level 侧 15 个 (fold,trial) cell）")
    print(f"    {'方法':<16s} {'ECE 均值':>9s} {'SD(15 cell)':>12s} {'σ_train':>9s} {'σ_fold':>8s}")
    for m in METHODS:
        v = r["arm_ece_variance"][m]
        print(f"    {LABELS[m]:<16s} {v['mean']:>9.4f} {v['sd_across_cells']:>12.4f} "
              f"{v['sigma_train']:>9.4f} {v['sigma_fold']:>8.4f}")

    print(f"\n  判读：所有检验量**越小越好** ⇒ Δ<0 = 校准更好；"
          f"**σ_train 为判据**，H1 的 CI 只含样本不确定性、不含训练随机性")
    for name in r["h2_variance"]:
        print(f"\n  [{name.replace('_vs_', ' − ')}]")
        print(f"    {'指标':<14s} {'Δ':>9s} {'95% CI (H1)':>21s} {'H1':>8s} | "
              f"{'|d̄|':>8s} {'σ_train':>9s} {'σ_fold':>8s} {'t':>6s} {'H2（超训练噪声？）':>18s}")
        for k in CAL_KEYS:
            v = r["h2_variance"][name].get(k)
            if v is None:
                continue
            e = r["h1_bootstrap"][name].get(k)
            if e is None:
                ci_txt, h1_txt = f"{'—':>21s}", f"{'—':>8s}"
            else:
                ci_txt = f"[{e['ci'][0]:>+8.4f},{e['ci'][1]:>+8.4f}]"
                h1_txt = f"{'**显著**' if e['sig'] else 'n.s.':>8s}"
            mark = "✅超过噪声" if v["exceeds_train_noise"] else "❌未超过"
            print(f"    {k:<14s} {v['mean']:>+9.4f} {ci_txt} {h1_txt} | "
                  f"{abs(v['mean']):>8.4f} {v['sigma_train']:>9.4f} {v['sigma_fold']:>8.4f} "
                  f"{v['t_ratio']:>6.2f} {mark:>18s}")


def plot_all(payload: dict, path: Path) -> None:
    """每方向一行 × 三列：逐 cell ECE / H1 的 ΔECE CI / H2 的 |d̄| vs σ_train。"""
    dirs = [k for k in DIRECTIONS if k in payload]
    fig, axes = plt.subplots(len(dirs), 3, figsize=(16.0, 4.5 * len(dirs)), squeeze=False)
    for row, dk in enumerate(dirs):
        r = payload[dk]

        # 列 1：逐 (fold, trial) ECE 散点
        ax = axes[row][0]
        for m in METHODS:
            arr = np.asarray(r["per_cell"][m][PRIMARY_KEY])
            xs = np.repeat(np.arange(N_FOLDS), len(TRIALS)) + \
                (np.tile(np.arange(len(TRIALS)), N_FOLDS) - 1) * 0.18
            ax.scatter(xs, arr.ravel(), s=34, color=COLORS[m], label=LABELS[m],
                       zorder=3, edgecolors="white", linewidths=0.9)
            ax.hlines(arr.mean(), -0.5, N_FOLDS - 0.5, color=COLORS[m],
                      linewidth=1.2, linestyle="--", alpha=0.8)
        ax.set_xticks(range(N_FOLDS), [f"fold{f}" for f in range(N_FOLDS)], fontsize=8.5)
        ax.set_ylabel("ECE（越低越好）", fontsize=9, color=MUTED)
        ax.set_title(f"{DIRECTIONS[dk]['title']}：逐 (fold, trial) ECE（虚线=均值）",
                     fontsize=10, color=INK)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
        ax.set_axisbelow(True)

        # 列 2：H1 的 ΔECE 配对 CI
        ax = axes[row][1]
        keys = list(r["h1_bootstrap"])
        for i, k in enumerate(keys):
            e = r["h1_bootstrap"][k][PRIMARY_KEY]
            lo, hi = e["ci"]
            c = "#1baf7a" if (e["sig"] and e["delta"] < 0) else ("#e34948" if e["sig"] else MUTED)
            ax.plot([lo, hi], [i, i], color=c, linewidth=2.2, solid_capstyle="round")
            ax.scatter([e["delta"]], [i], s=62, color=c, zorder=3,
                       edgecolors="white", linewidths=1.2)
            ax.annotate(f"{e['delta']:+.4f}", (max(hi, e["delta"]), i),
                        textcoords="offset points", xytext=(8, -3), fontsize=8, color=INK)
        ax.axvline(0, color=INK, linewidth=1.0)
        ax.set_yticks(range(len(keys)), [k.replace("_vs_", " − ") for k in keys], fontsize=8)
        ax.set_ylim(-0.6, len(keys) - 0.3)
        ax.set_xlabel("ΔECE（患者级配对 bootstrap；<0 = 更好）", fontsize=9, color=MUTED)
        ax.set_title("H1：样本不确定性（不含训练随机性）", fontsize=10, color=INK)
        ax.margins(x=0.35)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.grid(axis="x", color="#e6e5e1", linewidth=0.8)
        ax.set_axisbelow(True)

        # 列 3：H2 的 |d̄| vs σ_train
        ax = axes[row][2]
        keys = [k for k in r["h2_variance"] if PRIMARY_KEY in r["h2_variance"][k]]
        ys = np.arange(len(keys))
        eff = [abs(r["h2_variance"][k][PRIMARY_KEY]["mean"]) for k in keys]
        noise = [r["h2_variance"][k][PRIMARY_KEY]["sigma_train"] for k in keys]
        ax.barh(ys - 0.18, eff, height=0.32, color="#1baf7a", label="|效应量 d̄|",
                edgecolor="white", linewidth=0.8)
        ax.barh(ys + 0.18, noise, height=0.32, color="#9c9b95", label="σ_train（训练噪声）",
                edgecolor="white", linewidth=0.8)
        for i, k in enumerate(keys):
            v = r["h2_variance"][k][PRIMARY_KEY]
            # 用 √/× 而非 ✅/❌——CJK 字体缺后者的字形，会渲染成方框
            ax.annotate("√ 超过噪声" if v["exceeds_train_noise"] else "× 未超过噪声",
                        (max(eff[i], noise[i]), i), textcoords="offset points",
                        xytext=(6, -3), fontsize=8.5,
                        color="#1baf7a" if v["exceeds_train_noise"] else "#e34948")
        ax.set_yticks(ys, [k.replace("_vs_", " − ") for k in keys], fontsize=8)
        ax.set_xlabel("ECE 尺度", fontsize=9, color=MUTED)
        ax.set_title("H2：效应量 vs 训练噪声（**判据**）", fontsize=10, color=INK)
        # 留足右侧余量，使图例不与最下一行的「超过/未超过」标注重叠
        ax.margins(x=0.62)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
        ax.grid(axis="x", color="#e6e5e1", linewidth=0.8)
        ax.set_axisbelow(True)

    for r_ in axes:
        for a in r_:
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
            a.tick_params(colors=MUTED, labelsize=8.5)
    fig.suptitle("OOD 校准 · full-target 估计量（5 fold × 3 trial = 15 replicate）· "
                 "主要终点 ECE", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="实验 G 未决项 1：OOD 校准 full-target × 15 replicate + σ_train 检验")
    ap.add_argument("--direction", choices=("M2C", "C2M"), default=None)
    ap.add_argument("--n-boot", type=int, default=1000, help="患者级配对 cluster bootstrap 次数")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="缺省按 --bnupd 自动选 calibration_g_full_target[_bnupd]")
    ap.add_argument("--bnupd", action="store_true",
                    help="两个权重平均派生臂改读 BN 与官方对齐（update_bn）后的预测；"
                         "ERM/HyperAdapt 仍读原目录，作逐位不变的对照")
    ap.add_argument("--plot-only", action="store_true",
                    help="不重算，直接从既有 JSON 重绘（调图样式时用，省去数十分钟 bootstrap）")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = BNUPD_OUT_DIR if args.bnupd else OUT_DIR
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        saved = json.loads((args.out_dir / "calibration_full_target.json").read_text())
        for dk in [k for k in DIRECTIONS if k in saved]:
            print_direction(dk, saved[dk])
        plot_all({k: v for k, v in saved.items() if not k.startswith("_")},
                 args.out_dir / "calibration_full_target_ece.png")
        print(f"\n（--plot-only）图已重绘 → {args.out_dir / 'calibration_full_target_ece.png'}")
        return

    payload = {}
    for dk in DIRECTIONS:
        if args.direction and dk != args.direction:
            continue
        r = run_direction(dk, args.n_boot, bnupd=args.bnupd)
        if r is None:
            continue
        payload[dk] = r
        print_direction(dk, r)

    # Holm：family = 4 个对比 × 方向数，仅对主要终点 ECE（判 H1；不替代 σ_train）
    fam = [(f"{dk}:{k}", payload[dk]["h1_bootstrap"][k][PRIMARY_KEY]["p"])
           for dk in payload for k in HOLM_KEYS]
    if fam:
        hres = holm(fam)
        print(f"\n{'=' * 118}\nHolm 校正（family = {len(fam)} 个检验，α=0.05，主要终点 ECE）"
              f"\n{'=' * 118}")
        for k, p in sorted(fam, key=lambda kv: kv[1]):
            print(f"  {k:<44s} p={p:.4g}  →  {'**保持显著**' if hres[k] else '不显著'}")
        payload["_holm_ece"] = {"family": dict(fam), "reject": hres}

    plot_all({k: v for k, v in payload.items() if not k.startswith("_")},
             args.out_dir / "calibration_full_target_ece.png")
    out = args.out_dir / "calibration_full_target.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n图 → {args.out_dir / 'calibration_full_target_ece.png'}")
    print(f"已落盘 → {out}")


if __name__ == "__main__":
    main()
