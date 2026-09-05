"""
OOD v2 主分析：H1（复现）与 H2（效应是否在训练噪声之上）
==========================================================
预注册：`docs/ood_experiment_v2_preregistration.md` §3–§5。

**估计量 = full-target**：每个 source `(fold, trial)` 模型评**完整** target，逐 (f,t) 算指标再等权
平均。所有模型在**同一批完整 target 样本**上被比较 ⇒ 逐 (f,t) 天然配对，且从不跨折池化。

**两条正交的不确定性**，分别对应两个假设：

  · **H1（样本不确定性）** —— 患者级配对 cluster bootstrap：重采样 target 患者，全方法共用同一批
    重采样索引。回答「换一批病人，结论还成立吗」。
  · **H2（训练不确定性）** —— 对**配对差** `d_{f,t} = 指标^HA_{f,t} − 指标^ERM_{f,t}` 做
    fold × trial 双因子方差分解。回答「换一次训练随机种子，结论还成立吗」。
    这是 v2 引入 trial 的唯一目的：现行设计 `seed = 42 + fold` 把两者绑死，**无法分离**。

**方差分解（每格 1 观测的双因子随机效应）**：
    d_{f,t} = μ + α_f + β_t + ε_{f,t}
  用 ANOVA 均方估计分量：σ²_fold = (MS_f − MS_e)/T，σ²_trial = (MS_t − MS_e)/F，σ²_e = MS_e。
  每格仅 1 观测 ⇒ **交互项与残差混淆**，故 σ²_e 含「fold×trial 交互 + 纯训练噪声」；
  据此把**训练侧噪声**定义为 σ_train = √(σ²_trial + σ²_e)（保守，宁可高估噪声）。
  负的方差分量按惯例截断为 0（Searle et al., *Variance Components*）。

**H2 判据**（预注册 §4）：效应量 |d̄| > σ_train ⇒ 效应超过训练噪声尺度。
另报 d̄ 的标准误 SE(d̄) 与 t 比值，供参照。

运行（轻量 CPU）：
    python scripts/ood_variance_decomposition.py
    python scripts/ood_variance_decomposition.py --metric overall
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

from scripts.eaudit_m2c_full_target import sorted_view, weighted_auc
from src.training.harness.subgroup_auc import subgroup_masks

for _p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"):
    if Path(_p).exists():
        font_manager.fontManager.addfont(_p)
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUTS = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs")
OUT_DIR = OUTPUTS / "ood_cxr" / "variance_decomposition"
N_FOLDS, TRIALS = 5, (0, 1, 2)
N_BOOT = 1000
BOOT_SEED = 42

DIRECTIONS = {
    "M2C": {"cfg_dir": "mimic_cxr", "sub": "mimic2chexpert", "tgt_key": "chexpert",
            "title": "MIMIC → CheXpert"},
    "C2M": {"cfg_dir": "chexpert_cxr", "sub": "chexpert2mimic", "tgt_key": "mimic",
            "title": "CheXpert → MIMIC"},
}
# 追加实验 G 的融合臂（HyperAdapt+SWAD）与 GroupDRO 基线——追加不插入，既有方法顺序/配色不变。
# OOD v2 偏离 #4：再追加 HyperHead / HyperFusion 两臂（恢复条件化深度轴）——同样是**追加不插入**，
# 既有方法顺序/配色逐位不变。
METHODS = ("erm", "swad", "hyperadapt", "hyperadapt_swad", "groupdro",
           "hyperhead", "hyperfusion")
BASE = "erm"
COLORS = {"erm": "#2a78d6", "hyperadapt": "#eb6834", "swad": "#1baf7a",
          "hyperadapt_swad": "#8b53c9", "groupdro": "#c2185b",
          "hyperhead": "#d4a017", "hyperfusion": "#00838f"}
LABELS = {"erm": "ERM", "swad": "SWAD", "hyperadapt": "HyperAdapt",
          "hyperadapt_swad": "HyperAdapt+SWAD", "groupdro": "GroupDRO",
          "hyperhead": "HyperHead", "hyperfusion": "HyperFusion"}
INK, MUTED = "#0b0b0b", "#52514e"


def seed_of(fold: int, trial: int) -> int:
    """预注册 §2.1 的 seed 编码：42 + fold + 10 × trial。"""
    return 42 + fold + 10 * trial


class TargetViews:
    """
    单个 (方法, fold, trial) 在**完整 target** 上的预排序视图。

    full-target 下所有模型共享同一批样本与行序，故 patient cluster 索引全局只需算一次。
    """

    def __init__(self, path: Path, ds_key: str) -> None:
        with np.load(path, allow_pickle=True) as d:
            y = d["y_true"].astype(int)
            s = d["y_score"].astype(np.float64)
            attrs = {k: d[f"attr_{k}"] for k in (str(x) for x in d["attr_keys"])}
            self.patient_id = d["extra_patient_id"]
        kw = {k: np.asarray(v).astype(int) for k, v in attrs.items()
              if k in ("sex", "race", "age")}
        self.groups = [g for g in subgroup_masks(ds_key, **kw) if "|" not in g]
        self.v = {"__all__": sorted_view(y, s, np.ones(len(y), bool))}
        for g, m in subgroup_masks(ds_key, **kw).items():
            if "|" not in g:
                self.v[g] = sorted_view(y, s, m)
        self.n = len(y)

    def metrics(self, w: np.ndarray) -> tuple[float, float]:
        """给定样本权重下的 (Overall AUC, marginal worst AUC)。"""
        i, ys = self.v["__all__"]
        overall = weighted_auc(ys, w[i])
        vals = []
        for g in self.groups:
            ig, yg = self.v[g]
            a = weighted_auc(yg, w[ig])
            if not np.isnan(a):
                vals.append(a)
        return float(overall), float(min(vals)) if vals else float("nan")


def anova_two_way(d: np.ndarray) -> dict[str, float]:
    """
    每格 1 观测的双因子随机效应方差分解。

    Args:
        d: [F, T] 配对差矩阵（行 = fold，列 = trial）。

    Returns:
        方差分量与派生量；负分量截断为 0。
    """
    f, t = d.shape
    grand = d.mean()
    row, col = d.mean(axis=1), d.mean(axis=0)
    ss_f = t * ((row - grand) ** 2).sum()
    ss_t = f * ((col - grand) ** 2).sum()
    ss_e = ((d - row[:, None] - col[None, :] + grand) ** 2).sum()
    ms_f = ss_f / (f - 1)
    ms_t = ss_t / (t - 1)
    ms_e = ss_e / ((f - 1) * (t - 1))
    var_f = max((ms_f - ms_e) / t, 0.0)
    var_t = max((ms_t - ms_e) / f, 0.0)
    var_e = max(ms_e, 0.0)
    # 训练侧噪声：trial 主效应 + 残差（每格 1 观测 ⇒ 残差含 fold×trial 交互，保守计入）
    sigma_train = float(np.sqrt(var_t + var_e))
    se_mean = float(np.sqrt((var_f / f) + (var_t / t) + (var_e / (f * t))))
    return {"mean": float(grand), "var_fold": float(var_f), "var_trial": float(var_t),
            "var_resid": float(var_e), "sigma_fold": float(np.sqrt(var_f)),
            "sigma_train": sigma_train, "se_mean": se_mean,
            "t_ratio": float(grand / se_mean) if se_mean > 0 else float("nan"),
            "exceeds_train_noise": bool(abs(grand) > sigma_train)}


def run_direction(direction: str, metric: str, n_boot: int) -> dict | None:
    """跑一个方向的 H1 + H2 分析。"""
    d = DIRECTIONS[direction]
    cfg = json.loads((OUTPUTS / d["cfg_dir"] / "cv5" / "selected_configs.json").read_text())["config"]
    pred_dir = OUTPUTS / "ood_cxr" / d["sub"] / "cv5_full_target" / "predictions"

    views: dict[str, dict[tuple[int, int], TargetViews]] = {}
    for m in METHODS:
        views[m] = {}
        for trial in TRIALS:
            for fold in range(N_FOLDS):
                p = pred_dir / f"{m}_{cfg[m]}_seed{seed_of(fold, trial)}_overall.npz"
                if not p.exists():
                    print(f"[跳过] {direction}: 缺 {p.name}")
                    return None
                views[m][(fold, trial)] = TargetViews(p, d["tgt_key"])

    # full-target ⇒ 所有 (f,t) 共享同一批样本，cluster 索引只需算一次
    ref = views[BASE][(0, 0)]
    _, cidx = np.unique(ref.patient_id, return_inverse=True)
    n_cl = int(cidx.max() + 1)
    col = 0 if metric == "overall" else 1
    ones = np.ones(ref.n)

    def matrix(m: str, w: np.ndarray) -> np.ndarray:
        """该方法的 [F, T] 指标矩阵。"""
        return np.array([[views[m][(f, t)].metrics(w)[col] for t in TRIALS]
                         for f in range(N_FOLDS)])

    obs = {m: matrix(m, ones) for m in METHODS}

    # --- H1：患者级配对 cluster bootstrap（全方法共用重采样索引）---
    rng = np.random.default_rng(BOOT_SEED)
    boot: dict[str, list[float]] = {m: [] for m in METHODS}
    for _ in range(n_boot):
        draw = rng.integers(0, n_cl, size=n_cl)
        w = np.bincount(draw, minlength=n_cl).astype(float)[cidx]
        for m in METHODS:
            boot[m].append(float(matrix(m, w).mean()))
    B = {m: np.asarray(boot[m]) for m in METHODS}

    res = {"direction": direction, "metric": metric, "n": ref.n, "n_clusters": n_cl,
           "n_boot": n_boot, "config": {m: cfg[m] for m in METHODS},
           "point": {m: float(obs[m].mean()) for m in METHODS},
           "per_cell": {m: obs[m].tolist() for m in METHODS},
           # 逐方法的 bootstrap 分布（全方法共用同一批重采样索引）⇒ **任意一对**方法的配对差
           # 都是 B[m1] − B[m2]，无须重跑 bootstrap。供第 5 章报告预注册的
           # 「3 HN × 3 baseline = 每方向 9 对比」确认性家族。
           "boot": {m: B[m].tolist() for m in METHODS},
           "h1_bootstrap": {}, "h2_variance": {}}
    for m in METHODS:
        if m == BASE:
            continue
        diff = B[m] - B[BASE]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        res["h1_bootstrap"][f"{m}_vs_{BASE}"] = {
            "delta": float(obs[m].mean() - obs[BASE].mean()),
            "ci": [float(lo), float(hi)], "sig": bool(lo > 0 or hi < 0)}
        res["h2_variance"][f"{m}_vs_{BASE}"] = anova_two_way(obs[m] - obs[BASE])
    return res


def holm(pairs: list[tuple[str, float]]) -> dict[str, bool]:
    """
    Holm 校正（预注册 §5：family = 2 方法 × 2 方向 = 4 个检验）。

    Args:
        pairs: [(键, 双侧 p 值)]。

    Returns:
        {键: 校正后是否显著}（α=0.05）。
    """
    order = sorted(pairs, key=lambda kv: kv[1])
    n = len(order)
    out, prev_reject = {}, True
    for i, (k, p) in enumerate(order):
        thresh = 0.05 / (n - i)
        rej = prev_reject and p <= thresh
        out[k] = rej
        prev_reject = rej
    return out


def plot_all(payload: dict, metric: str, path: Path) -> None:
    """双方向 × (逐 cell 指标 / H1 CI / H2 方差分量) 三列。"""
    dirs = [k for k in DIRECTIONS if k in payload]
    fig, axes = plt.subplots(len(dirs), 3, figsize=(15.5, 4.3 * len(dirs)), squeeze=False)
    for r, dk in enumerate(dirs):
        res = payload[dk]
        # 列 1：逐 (fold, trial) 指标散点
        ax = axes[r][0]
        for m in METHODS:
            arr = np.asarray(res["per_cell"][m])
            xs = np.repeat(np.arange(N_FOLDS), len(TRIALS)) + \
                (np.tile(np.arange(len(TRIALS)), N_FOLDS) - 1) * 0.18
            ax.scatter(xs, arr.ravel(), s=34, color=COLORS[m], label=LABELS[m],
                       zorder=3, edgecolors="white", linewidths=0.9)
            ax.hlines(arr.mean(), -0.5, N_FOLDS - 0.5, color=COLORS[m],
                      linewidth=1.2, linestyle="--", alpha=0.8)
        ax.set_xticks(range(N_FOLDS), [f"fold{f}" for f in range(N_FOLDS)], fontsize=8.5)
        ax.set_ylabel(f"{metric} AUC", fontsize=9, color=MUTED)
        ax.set_title(f"{DIRECTIONS[dk]['title']}：逐 (fold, trial)（虚线=均值）",
                     fontsize=10, color=INK)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
        ax.set_axisbelow(True)

        # 列 2：H1 bootstrap CI
        ax = axes[r][1]
        keys = list(res["h1_bootstrap"])
        for i, k in enumerate(keys):
            e = res["h1_bootstrap"][k]
            lo, hi = e["ci"]
            c = "#2a78d6" if (e["sig"] and e["delta"] > 0) else ("#e34948" if e["sig"] else MUTED)
            ax.plot([lo, hi], [i, i], color=c, linewidth=2.2, solid_capstyle="round")
            ax.scatter([e["delta"]], [i], s=62, color=c, zorder=3,
                       edgecolors="white", linewidths=1.2)
            ax.annotate(f"{e['delta']:+.4f} [{lo:+.4f},{hi:+.4f}]",
                        (max(hi, e["delta"]), i), textcoords="offset points",
                        xytext=(8, -3), fontsize=8, color=INK)
        ax.axvline(0, color=INK, linewidth=1.0)
        ax.set_yticks(range(len(keys)), [k.replace("_vs_", " − ") for k in keys], fontsize=8.5)
        ax.set_ylim(-0.6, len(keys) - 0.3)
        ax.set_xlabel("Δ（患者级配对 bootstrap）", fontsize=9, color=MUTED)
        ax.set_title("H1：样本不确定性", fontsize=10, color=INK)
        ax.margins(x=0.5)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.grid(axis="x", color="#e6e5e1", linewidth=0.8)
        ax.set_axisbelow(True)

        # 列 3：H2 效应量 vs 训练噪声
        ax = axes[r][2]
        keys = list(res["h2_variance"])
        ys = np.arange(len(keys))
        eff = [abs(res["h2_variance"][k]["mean"]) for k in keys]
        noise = [res["h2_variance"][k]["sigma_train"] for k in keys]
        ax.barh(ys - 0.18, eff, height=0.32, color="#1baf7a", label="|效应量 d̄|",
                edgecolor="white", linewidth=0.8)
        ax.barh(ys + 0.18, noise, height=0.32, color="#9c9b95", label="σ_train（训练噪声）",
                edgecolor="white", linewidth=0.8)
        for i, k in enumerate(keys):
            v = res["h2_variance"][k]
            ax.annotate("✅ 超过噪声" if v["exceeds_train_noise"] else "❌ 未超过噪声",
                        (max(eff[i], noise[i]), i), textcoords="offset points",
                        xytext=(6, -3), fontsize=8.5,
                        color="#1baf7a" if v["exceeds_train_noise"] else "#e34948")
        ax.set_yticks(ys, [k.replace("_vs_", " − ") for k in keys], fontsize=8.5)
        ax.set_xlabel("AUC 尺度", fontsize=9, color=MUTED)
        ax.set_title("H2：效应量 vs 训练噪声", fontsize=10, color=INK)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
        ax.margins(x=0.35)
        ax.grid(axis="x", color="#e6e5e1", linewidth=0.8)
        ax.set_axisbelow(True)

    for row in axes:
        for a in row:
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
            a.tick_params(colors=MUTED, labelsize=8.5)
    fig.suptitle(f"OOD v2 主分析（full-target，5 fold × 3 trial = 15 replicate）· {metric}",
                 fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)


def main() -> None:
    global METHODS                      # --methods 可限定方法集（须在本函数任何 METHODS 使用之前声明）
    ap = argparse.ArgumentParser(description="OOD v2 主分析：H1 复现 + H2 训练噪声")
    ap.add_argument("--metric", choices=("marginal_worst", "overall"), default="marginal_worst")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    # Holm family = (方法数 − 1) × 2 方向，故**方法集直接决定 family 大小**（预注册 §5 / 偏离 #3）。
    # 缺省跑全部注册方法；给 --methods 可限定为某个 family（如确认性四臂
    # erm,swad,hyperadapt,groupdro = family 6，把探索性的 hyperadapt_swad 留给实验 G 自己的文档）。
    ap.add_argument("--methods", default=None,
                    help=f"逗号分隔的方法子集（须含基线 {BASE}）；缺省 = {','.join(METHODS)}")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.methods:
        sel = tuple(m.strip() for m in args.methods.split(","))
        unknown = [m for m in sel if m not in LABELS]
        if unknown or BASE not in sel:
            raise SystemExit(f"--methods 非法：未知 {unknown}，或缺基线 {BASE}")
        METHODS = sel
    print(f"方法集 = {METHODS}  ⇒  Holm family = {(len(METHODS) - 1) * len(DIRECTIONS)} 个检验")

    payload = {}
    for dk in DIRECTIONS:
        r = run_direction(dk, args.metric, args.n_boot)
        if r is None:
            continue
        payload[dk] = r
        print(f"\n{'=' * 96}\n{DIRECTIONS[dk]['title']}  metric={args.metric}  "
              f"n={r['n']:,}  患者={r['n_clusters']:,}\n{'=' * 96}")
        for m in METHODS:
            print(f"  {LABELS[m]:<12s} {r['point'][m]:.4f}   config={r['config'][m]}")
        print(f"\n  {'比较':<22s} {'Δ':>9s} {'95% CI':>21s} {'H1':>8s} | "
              f"{'|d̄|':>8s} {'σ_train':>9s} {'σ_fold':>8s} {'t':>6s} {'H2':>12s}")
        for k in r["h1_bootstrap"]:
            e, v = r["h1_bootstrap"][k], r["h2_variance"][k]
            lo, hi = e["ci"]
            print(f"  {k.replace('_vs_', ' − '):<22s} {e['delta']:>+9.4f} "
                  f"[{lo:>+8.4f},{hi:>+8.4f}] {'**显著**' if e['sig'] else 'n.s.':>8s} | "
                  f"{abs(v['mean']):>8.4f} {v['sigma_train']:>9.4f} {v['sigma_fold']:>8.4f} "
                  f"{v['t_ratio']:>6.2f} {'✅超过噪声' if v['exceeds_train_noise'] else '❌未超过':>12s}")

    # Holm 校正（family = 2 方法 × 2 方向），用 bootstrap 双侧 p 的正态近似
    fam = []
    for dk, r in payload.items():
        for k, e in r["h1_bootstrap"].items():
            lo, hi = e["ci"]
            se = (hi - lo) / (2 * 1.96)
            from scipy.stats import norm as _n
            p = 2 * (1 - _n.cdf(abs(e["delta"]) / se)) if se > 0 else 1.0
            fam.append((f"{dk}:{k}", float(p)))
    hres = holm(fam)
    print(f"\n{'=' * 96}\nHolm 校正（family = {len(fam)} 个检验，α=0.05）\n{'=' * 96}")
    for k, p in sorted(fam, key=lambda kv: kv[1]):
        print(f"  {k:<34s} p={p:.4g}  →  {'**保持显著**' if hres[k] else '不显著'}")
    payload["_holm"] = {"family": dict(fam), "reject": hres}

    plot_all({k: v for k, v in payload.items() if not k.startswith("_")},
             args.metric, args.out_dir / f"variance_decomposition_{args.metric}.png")
    out = args.out_dir / f"variance_decomposition_{args.metric}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n图 → {args.out_dir / f'variance_decomposition_{args.metric}.png'}")
    print(f"已落盘 → {out}")


if __name__ == "__main__":
    main()
