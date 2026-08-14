"""
Effective robustness（MIMIC ↔ CheXpert）—— v2 实验 D3 的第二部分 / 假设 H4
==========================================================================
`docs/ood_experiment_v2_preregistration.md` §4 · H4。

**要回答的问题**：某方法 OOD 表现更好，是因为它**真的更抗分布偏移**，还是仅仅因为它**ID 本来就更强**、
OOD 只是顺带更好？Taori et al. (2020, NeurIPS) 与 Miller et al. (2021, ICML "Accuracy on the Line")
给出的标准答案是：把所有模型画在 (ID 指标, OOD 指标) 平面上，**同一分布偏移下这些点通常落在一条线上**；
真正的鲁棒性 = **位于该线之上的垂直距离**（effective robustness），而非 OOD 绝对值高。

**probit 尺度**：按 Miller 的做法，两轴均取 `Φ⁻¹(·)` 后再做线性拟合——原始尺度下关系是弯的，
probit 后近似线性，拟合残差才有意义。

**拟合线用尽可能多的点，判定只对预注册的 3 方法**：拟合线是"这一对数据集在此偏移下的基准趋势"，
点越多越稳，故**纳入全部可用模型**（含 HyperHead / HyperFusion / 超参对照臂 ERM@matched）；
但**残差的假设判定只对 ERM / SWAD / HyperAdapt**（预注册 §1 范围）。这不违反预注册——预注册限定的是
方法比较范围，不是拟合线的数据来源。

**一个点 = 一个 (方法, config, fold)**：ID 指标取该 fold 的 source-域 test，OOD 指标取该 fold 模型
评的 target。逐折成点（不跨折平均），使拟合有足够样本量。

运行（轻量 CPU）：
    python scripts/ood_effective_robustness.py
    python scripts/ood_effective_robustness.py --metric marginal_worst
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy.stats import norm

from src.training.harness.predictions import load_predictions
from src.training.harness.subgroup_auc import subgroup_masks
from scripts.eaudit_m2c_full_target import sorted_view, weighted_auc

for _p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"):
    if Path(_p).exists():
        font_manager.fontManager.addfont(_p)
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUTS = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs")
OUT_DIR = OUTPUTS / "ood_cxr" / "effective_robustness"
FOLD_SEEDS = (42, 43, 44, 45, 46)
N_BOOT = 2000

# ---- 拟合可用性守卫 ----------------------------------------------------------------
# effective robustness 的前提是「这些模型在 ID 上有足够的质量跨度，足以定出一条趋势线」。
# Taori/Miller 的原始研究用的是 accuracy 从 ~0.2 到 ~0.9 的上百个模型；若所有点挤在一个
# 极窄区间内，点的散布几乎全是评估噪声，拟合出的斜率与残差**没有实质含义**，据此判定
# 「某方法位于线之上」会得到看似显著、实则虚假的结论。故设双重守卫，不满足则拒绝判定 H4。
MIN_ID_SPAN = 0.05      # ID 指标 max−min 的最小跨度（原始 AUC 尺度）
MIN_R2 = 0.25           # probit 线性拟合的最小 R²

# 方向 → (source 目录, target 子群键, OOD 预测目录名)
DIRECTIONS = {
    "M2C": {"src_dir": "mimic_cxr", "src_key": "mimic", "tgt_key": "chexpert",
            "ood_sub": "mimic2chexpert", "title": "MIMIC → CheXpert"},
    "C2M": {"src_dir": "chexpert_cxr", "src_key": "chexpert", "tgt_key": "mimic",
            "ood_sub": "chexpert2mimic", "title": "CheXpert → MIMIC"},
}
# 预注册范围内的方法（只有这三个参与 H4 判定）
PREREG_METHODS = ("erm", "swad", "hyperadapt")
# dataviz 分类槽位：前三槽 all-pairs 安全，其余为辅助点（灰）
COLORS = {"erm": "#2a78d6", "hyperadapt": "#eb6834", "swad": "#1baf7a"}
AUX_COLOR = "#9c9b95"
LABELS = {"erm": "ERM", "swad": "SWAD", "hyperadapt": "HyperAdapt",
          "hyperhead": "HyperHead（仅拟合）", "hyperfusion": "HyperFusion（仅拟合）"}
INK, MUTED = "#0b0b0b", "#52514e"


def metric_from_npz(path: Path, ds_key: str, metric: str) -> float | None:
    """
    从一份预测 npz 算单个指标。

    Args:
        path: 预测文件路径。
        ds_key: subgroup_masks 的数据集键（决定子群 schema）。
        metric: "overall" 或 "marginal_worst"。

    Returns:
        指标值；文件不存在或指标无定义时返回 None。
    """
    if not path.exists():
        return None
    y, s, a = load_predictions(path)
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    # sorted_view 返回原始索引 + 排序后标签；无 bootstrap 时权重恒为全 1（长度 = 该子群大小）
    if metric == "overall":
        _, ys = sorted_view(y, s, np.ones(len(y), bool))
        v = weighted_auc(ys, np.ones(len(ys)))
        return None if np.isnan(v) else float(v)
    kw = {k: np.asarray(v).astype(int) for k, v in a.items()
          if k in ("sex", "race", "age", "skin")}
    vals = []
    for g, m in subgroup_masks(ds_key, **kw).items():
        if "|" in g:                                   # marginal only
            continue
        _, ys = sorted_view(y, s, m)
        v = weighted_auc(ys, np.ones(len(ys)))
        if not np.isnan(v):
            vals.append(v)
    return float(min(vals)) if vals else None


def collect_points(direction: str, metric: str) -> list[dict]:
    """收集某方向下全部 (方法, config, fold) 的 (ID, OOD) 配对点。"""
    cfg_dir = OUTPUTS / DIRECTIONS[direction]["src_dir"] / "cv5"
    sel = json.loads((cfg_dir / "selected_configs.json").read_text())["config"]
    ood_root = OUTPUTS / "ood_cxr" / DIRECTIONS[direction]["ood_sub"]
    src_key, tgt_key = DIRECTIONS[direction]["src_key"], DIRECTIONS[direction]["tgt_key"]

    # (方法, config, OOD 预测目录)：主目录的选中 config + 隔离目录的超参对照臂 ERM
    entries = [(m, tag, ood_root / "cv5" / "predictions") for m, tag in sel.items()
               if m in LABELS]
    hp = ood_root / "cv5_hpmatch" / "predictions"
    if hp.exists():
        tags = {p.name.split("_seed")[0].replace("erm_", "") for p in hp.glob("erm_*_overall.npz")}
        entries += [("erm", t, hp) for t in sorted(tags)]

    points = []
    for method, tag, ood_pred in entries:
        for fold, seed in enumerate(FOLD_SEEDS):
            id_v = metric_from_npz(
                cfg_dir / "predictions" / f"{method}_{tag}_seed{seed}_overall.npz", src_key, metric)
            ood_v = metric_from_npz(
                ood_pred / f"{method}_{tag}_seed{seed}_overall.npz", tgt_key, metric)
            if id_v is None or ood_v is None:
                continue
            points.append({"method": method, "config": tag, "fold": fold,
                           "id": id_v, "ood": ood_v})
    return points


def fit_and_residuals(points: list[dict]) -> dict:
    """
    probit 尺度线性拟合 + 逐方法残差（effective robustness）及其 bootstrap CI。

    Args:
        points: `collect_points` 的输出。

    Returns:
        含 slope/intercept、逐方法残差均值与 95% CI 的字典。
    """
    x = norm.ppf(np.clip([p["id"] for p in points], 1e-6, 1 - 1e-6))
    y = norm.ppf(np.clip([p["ood"] for p in points], 1e-6, 1 - 1e-6))
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    methods = np.array([p["method"] for p in points])

    rng = np.random.default_rng(42)
    id_span = float(max(p["id"] for p in points) - min(p["id"] for p in points))
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    usable = bool(id_span >= MIN_ID_SPAN and r2 >= MIN_R2)
    reasons = []
    if id_span < MIN_ID_SPAN:
        reasons.append(f"ID 质量跨度仅 {id_span:.4f} < {MIN_ID_SPAN}（点全挤在一处，散布≈评估噪声）")
    if r2 < MIN_R2:
        reasons.append(f"probit 拟合 R²={r2:.3f} < {MIN_R2}（ID 与 OOD 近乎不相关，趋势线不成立）")

    out = {"slope": float(slope), "intercept": float(intercept), "r2": r2,
           "id_span": id_span, "n_points": len(points),
           "fit_usable": usable, "unusable_reasons": reasons, "residuals": {}}
    for m in PREREG_METHODS:
        sel = methods == m
        if not sel.any():
            continue
        obs = float(resid[sel].mean())
        # 重抽全部点 → 重拟合 → 重算该方法残差均值（把拟合线的不确定性一并计入）
        boot = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(x), size=len(x))
            if len(np.unique(x[idx])) < 2:
                continue
            s_b, i_b = np.polyfit(x[idx], y[idx], 1)
            r_b = y[sel] - (s_b * x[sel] + i_b)
            boot.append(r_b.mean())
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out["residuals"][m] = {
            "mean": obs, "ci": [float(lo), float(hi)], "n": int(sel.sum()),
            # 拟合不可用时一律不判定，避免基于坏趋势线给出虚假的「线之上」
            "above_line": bool(lo > 0) if usable else None,
        }
    return out


def plot_direction(direction: str, points: list[dict], fit: dict, metric: str, path: Path) -> None:
    """散点 + probit 拟合线（左：原始尺度标注；右：逐方法 effective robustness 残差）。"""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    x = norm.ppf(np.clip([p["id"] for p in points], 1e-6, 1 - 1e-6))
    y = norm.ppf(np.clip([p["ood"] for p in points], 1e-6, 1 - 1e-6))
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, fit["slope"] * xs + fit["intercept"], color=MUTED, linewidth=1.6,
            label=f"probit 线性拟合 (R²={fit['r2']:.2f})", zorder=2)
    for m in sorted({p["method"] for p in points}):
        sel = [i for i, p in enumerate(points) if p["method"] == m]
        prereg = m in PREREG_METHODS
        ax.scatter(x[sel], y[sel], s=46 if prereg else 30,
                   color=COLORS.get(m, AUX_COLOR), label=LABELS.get(m, m), zorder=3,
                   edgecolors="white", linewidths=1.0, alpha=1.0 if prereg else 0.65)
    # 副轴刻度换回原始 AUC，便于阅读
    ticks = np.array([0.78, 0.80, 0.82, 0.84, 0.86, 0.88])
    ax.set_xticks(norm.ppf(ticks), [f"{t:.2f}" for t in ticks], fontsize=8.5)
    yt = np.array([0.76, 0.78, 0.80, 0.82, 0.84, 0.86])
    ax.set_yticks(norm.ppf(yt), [f"{t:.2f}" for t in yt], fontsize=8.5)
    ax.set_xlabel(f"ID（source 域）{metric} AUC", fontsize=9, color=MUTED)
    ax.set_ylabel(f"OOD（target 域）{metric} AUC", fontsize=9, color=MUTED)
    ax.set_title(f"{DIRECTIONS[direction]['title']}：ID vs OOD（probit 尺度，n={fit['n_points']} 点）",
                 fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)

    ax = axes[1]
    ms = [m for m in PREREG_METHODS if m in fit["residuals"]]
    usable = fit["fit_usable"]
    for i, m in enumerate(ms):
        r = fit["residuals"][m]
        lo, hi = r["ci"]
        col = COLORS[m] if r["above_line"] else MUTED     # 不可用时 above_line=None ⇒ 灰
        ax.plot([lo, hi], [i, i], color=col, linewidth=2.2, solid_capstyle="round",
                alpha=1.0 if usable else 0.45)
        ax.scatter([r["mean"]], [i], s=64, color=col, zorder=3,
                   edgecolors="white", linewidths=1.2, alpha=1.0 if usable else 0.45)
        ax.annotate(f"{r['mean']:+.4f} [{lo:+.4f},{hi:+.4f}]"
                    f"{'  线之上' if r['above_line'] else ''}",
                    (max(hi, r["mean"]), i), textcoords="offset points", xytext=(8, -3),
                    fontsize=8.5, color=INK if usable else MUTED)
    ax.axvline(0, color=INK, linewidth=1.0)
    ax.set_yticks(range(len(ms)), [LABELS[m] for m in ms], fontsize=9)
    ax.set_ylim(-0.6, len(ms) - 0.3)
    ax.set_xlabel("effective robustness（probit 残差；>0 = 优于基准趋势）", fontsize=9, color=MUTED)
    if usable:
        ax.set_title("H4：是否位于拟合线之上", fontsize=10, color=INK)
    else:
        ax.set_title("H4 不可评估 —— 拟合线不成立", fontsize=10, color="#e34948")
        ax.text(0.5, -0.42, "；".join(fit["unusable_reasons"]), transform=ax.transAxes,
                ha="center", va="top", fontsize=8, color="#e34948", wrap=True)
    ax.margins(x=0.5)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)

    for a in axes:
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.tick_params(colors=MUTED, labelsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Effective robustness（Taori 2020 / Miller 2021）")
    ap.add_argument("--metric", choices=("overall", "marginal_worst"), default="overall")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    payload = {}
    for d in DIRECTIONS:
        points = collect_points(d, args.metric)
        if len(points) < 8:
            print(f"[跳过] {d}: 可用点仅 {len(points)} 个，不足以拟合")
            continue
        fit = fit_and_residuals(points)
        payload[d] = {"fit": fit, "points": points, "metric": args.metric}
        print(f"\n{'=' * 88}\n{DIRECTIONS[d]['title']}  metric={args.metric}  "
              f"n_points={fit['n_points']}  slope={fit['slope']:.3f}  R²={fit['r2']:.3f}  "
              f"ID跨度={fit['id_span']:.4f}\n{'=' * 88}")
        if not fit["fit_usable"]:
            print("  🛑 拟合线不成立 ⇒ **H4 不可评估**（残差仅供记录，不得作结论）：")
            for r in fit["unusable_reasons"]:
                print(f"     · {r}")
        for m, r in fit["residuals"].items():
            lo, hi = r["ci"]
            verdict = ("**线之上**" if r["above_line"] else "n.s.") if fit["fit_usable"] else "不可评估"
            print(f"  {LABELS[m]:<12s} n={r['n']:<3d} 残差={r['mean']:+.4f} "
                  f"[{lo:+.4f},{hi:+.4f}]  {verdict}")
        p = args.out_dir / f"{d.lower()}_effective_robustness_{args.metric}.png"
        plot_direction(d, points, fit, args.metric, p)
        print(f"  图 → {p}")

    out = args.out_dir / f"effective_robustness_{args.metric}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已落盘 → {out}")


if __name__ == "__main__":
    main()
