"""
OOD 属性 knockout 分析：判别机制 A（属性迁移）vs 机制 B（条件化的正则副作用）
==============================================================================
配套推理：`src/training/run_ood_attr_knockout.py`。设计与判读见
`docs/ood_experiment_v2_results.md` §5.1。

**三个条件**（同一份 HyperAdapt 权重，只改「模型看到什么属性」）：
  · `real`     ：真实属性（复用 `cv5_full_target/` 既有预测）
  · `permute`  ：全局置换（保持属性边缘分布与属性间联合结构，只打破 属性↔图像 配对）
  · `marginal` ：**输出边缘化**（主判据）`p(y|x)=Σ_a p(a)·p(y|x,a)`，对 8 种组合按 target
    经验先验加权平均概率——这是「移除属性信息」的正确定义

> `constant`（全置 majority cell）曾作为第三种 knockout 跑过，但它只是 `marginal` 把先验塌缩成
> 点质量的**退化特例**，且该点的 age 恰好等于主指标 worst 组的取值（碰巧对齐）；`marginal` 就位后
> 它不再提供独立信息，故已从分析中移除（预测产物保留在 `cv5_knockout_constant/` 供追溯）。

**判读**（Δ = HyperAdapt − ERM，marginal worst AUC；ERM 是 attribute-blind，各条件下恒定）：

    real 下 Δ>0 而 marginal 下 Δ→0   ⇒ 增益**依赖属性** ⇒ 机制 A 存活
    两条件下 Δ 都差不多               ⇒ 增益**与属性无关** ⇒ 机制 B（正则/容量）

**关键的第二个量**：`real − marginal` 这个**配对差**才是「属性信息的净贡献」。它比「Δ 是否>0」更
直接——同一份权重、同一批样本，架构/容量/超参全被消掉，唯一变化是模型是否知道真实属性。

口径与主分析一致：full-target、5 fold × 3 trial = 15 replicate、患者级配对 cluster bootstrap、
逐 (f,t) 算指标再等权平均。

运行（轻量 CPU）：
    python scripts/ood_knockout_analysis.py
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

from scripts.ood_variance_decomposition import (
    DIRECTIONS, N_FOLDS, TRIALS, TargetViews, anova_two_way, seed_of,
)

for _p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"):
    if Path(_p).exists():
        font_manager.fontManager.addfont(_p)
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUTS = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs")
OUT_DIR = OUTPUTS / "ood_cxr" / "knockout"
N_BOOT = 1000
BOOT_SEED = 42
CONDITIONS = ("real", "permute", "marginal")
COND_LABELS = {"real": "真实属性", "permute": "置换属性", "marginal": "边缘化(严格)"}
# dataviz 分类槽位 2/4/1（前三槽 all-pairs 安全；ERM 用中性灰作参照线）
COND_COLORS = {"real": "#eb6834", "permute": "#eda100", "marginal": "#1baf7a"}
INK, MUTED = "#0b0b0b", "#52514e"


def cond_dir(direction: str, cond: str) -> Path:
    """某条件的预测目录；real 复用主 full-target 产物。"""
    root = OUTPUTS / "ood_cxr" / DIRECTIONS[direction]["sub"]
    sub = "cv5_full_target" if cond == "real" else f"cv5_knockout_{cond}"
    return root / sub / "predictions"


def run_direction(direction: str, metric: str, n_boot: int) -> dict | None:
    """跑一个方向的 knockout 判别分析。"""
    d = DIRECTIONS[direction]
    cfg = json.loads((OUTPUTS / d["cfg_dir"] / "cv5" / "selected_configs.json").read_text())["config"]
    ha_tag, erm_tag = cfg["hyperadapt"], cfg["erm"]

    views: dict[str, dict[tuple[int, int], TargetViews]] = {}
    for cond in CONDITIONS:
        views[cond] = {}
        pdir = cond_dir(direction, cond)
        for trial in TRIALS:
            for fold in range(N_FOLDS):
                p = pdir / f"hyperadapt_{ha_tag}_seed{seed_of(fold, trial)}_overall.npz"
                if not p.exists():
                    print(f"[跳过] {direction}: 缺 {cond} 的 {p.name}")
                    return None
                views[cond][(fold, trial)] = TargetViews(p, d["tgt_key"])
    # ERM 参照（attribute-blind，三条件下恒定）
    erm: dict[tuple[int, int], TargetViews] = {}
    for trial in TRIALS:
        for fold in range(N_FOLDS):
            p = cond_dir(direction, "real") / f"erm_{erm_tag}_seed{seed_of(fold, trial)}_overall.npz"
            erm[(fold, trial)] = TargetViews(p, d["tgt_key"])

    ref = erm[(0, 0)]
    _, cidx = np.unique(ref.patient_id, return_inverse=True)
    n_cl = int(cidx.max() + 1)
    col = 0 if metric == "overall" else 1
    ones = np.ones(ref.n)

    def mat(vs: dict, w: np.ndarray) -> np.ndarray:
        return np.array([[vs[(f, t)].metrics(w)[col] for t in TRIALS] for f in range(N_FOLDS)])

    obs = {c: mat(views[c], ones) for c in CONDITIONS}
    obs_erm = mat(erm, ones)

    rng = np.random.default_rng(BOOT_SEED)
    boot: dict[str, list[float]] = {c: [] for c in CONDITIONS}
    boot["erm"] = []
    for _ in range(n_boot):
        draw = rng.integers(0, n_cl, size=n_cl)
        w = np.bincount(draw, minlength=n_cl).astype(float)[cidx]
        for c in CONDITIONS:
            boot[c].append(float(mat(views[c], w).mean()))
        boot["erm"].append(float(mat(erm, w).mean()))
    B = {k: np.asarray(v) for k, v in boot.items()}

    res = {"direction": direction, "metric": metric, "n": ref.n, "n_clusters": n_cl,
           "n_boot": n_boot, "config": {"hyperadapt": ha_tag, "erm": erm_tag},
           "point": {c: float(obs[c].mean()) for c in CONDITIONS} | {"erm": float(obs_erm.mean())},
           "vs_erm": {}, "vs_real": {}}

    # ① 各条件相对 ERM 的 Δ —— 判「增益是否还在」
    for c in CONDITIONS:
        diff = B[c] - B["erm"]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        v = anova_two_way(obs[c] - obs_erm)
        res["vs_erm"][c] = {"delta": float(obs[c].mean() - obs_erm.mean()),
                            "ci": [float(lo), float(hi)], "sig": bool(lo > 0 or hi < 0),
                            "sigma_train": v["sigma_train"], "t_ratio": v["t_ratio"],
                            "exceeds_train_noise": v["exceeds_train_noise"]}

    # ② real − knockout 的配对差 —— **属性信息的净贡献**（架构/容量/超参全被消掉）
    for c in ("permute", "marginal"):
        diff = B["real"] - B[c]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        v = anova_two_way(obs["real"] - obs[c])
        res["vs_real"][f"real_minus_{c}"] = {
            "delta": float(obs["real"].mean() - obs[c].mean()),
            "ci": [float(lo), float(hi)], "sig": bool(lo > 0 or hi < 0),
            "sigma_train": v["sigma_train"], "t_ratio": v["t_ratio"],
            "exceeds_train_noise": v["exceeds_train_noise"]}
    return res


def verdict(res: dict) -> str:
    """
    按 §5.1 的判读规则给出机制归属。

    **以 marginal 为主判据，permute 为辅**——两种 knockout 的反事实含义不同：

      · `marginal`：对属性先验积分 ⇒ **移除**属性信息的正确定义。`real − marginal` 即
        「属性**信息**的净贡献」，正对应机制 A 要问的量。
      · `permute` ：给出与图像**不匹配**的属性 ⇒ 模型施加**错误方向**的调制。
        `real − permute` 混入「错配惩罚」⇒ 系统性**高估**净贡献。

    故以 marginal 为准；若只有 permute 显著，说明证据主要来自「给错属性会变差」，
    而非「属性信息本身有用」——这两件事不等价。
    """
    key_main = "real_minus_marginal"      # 输出边缘化 = 「移除属性信息」的正确定义
    net_c = res["vs_real"][key_main]
    net_p = res["vs_real"]["real_minus_permute"]
    gain = res["vs_erm"]["real"]
    pos_c = net_c["sig"] and net_c["delta"] > 0
    pos_p = net_p["sig"] and net_p["delta"] > 0

    if pos_c and net_c["exceeds_train_noise"]:
        return f"机制 A：属性信息的净贡献在严格口径（{key_main}）下显著且超训练噪声"
    if pos_p and not pos_c:
        return ("机制 A 证据**弱**：仅 permute 口径显著（{:+.4f}），而严格口径 {} 不显著"
                "（{:+.4f}，CI 含 0）⇒ 证据主要来自「给错属性会变差」，"
                "而非「属性信息本身有用」".format(net_p["delta"], key_main, net_c["delta"]))
    if pos_c or pos_p:
        return "机制 A 方向，但净贡献未超过训练噪声 ⇒ 证据弱"
    if gain["sig"] and gain["delta"] > 0:
        return "🛑 机制 B：相对 ERM 有增益，但 knockout 后不变 ⇒ 与属性信息无关"
    return "无可归因增益（real 相对 ERM 本身即不显著为正）"


def plot_all(payload: dict, metric: str, path: Path) -> None:
    """左：三条件相对 ERM 的 Δ；右：real − knockout 的属性净贡献。"""
    import textwrap
    dirs = [k for k in DIRECTIONS if k in payload]
    verdicts: list[tuple[str, str]] = []
    fig, axes = plt.subplots(len(dirs), 2, figsize=(13.5, 4.1 * len(dirs)), squeeze=False)
    for r, dk in enumerate(dirs):
        res = payload[dk]
        ax = axes[r][0]
        for i, c in enumerate(CONDITIONS):
            e = res["vs_erm"][c]
            lo, hi = e["ci"]
            col = COND_COLORS[c] if e["sig"] else MUTED
            ax.plot([lo, hi], [i, i], color=col, linewidth=2.2, solid_capstyle="round")
            ax.scatter([e["delta"]], [i], s=64, color=col, zorder=3,
                       edgecolors="white", linewidths=1.2)
            ax.annotate(f"{e['delta']:+.4f} [{lo:+.4f},{hi:+.4f}]"
                        f"{'' if e['sig'] else '  n.s.'}",
                        (max(hi, e["delta"]), i), textcoords="offset points",
                        xytext=(8, -3), fontsize=8, color=INK)
        ax.axvline(0, color=INK, linewidth=1.0)
        ax.set_yticks(range(len(CONDITIONS)), [COND_LABELS[c] for c in CONDITIONS], fontsize=9)
        ax.set_ylim(-0.6, len(CONDITIONS) - 0.3)
        ax.set_xlabel(f"Δ {metric} AUC（HyperAdapt − ERM）", fontsize=9, color=MUTED)
        ax.set_title(f"{DIRECTIONS[dk]['title']}：各属性条件下相对 ERM 的增益",
                     fontsize=10, color=INK)
        ax.margins(x=0.5)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.grid(axis="x", color="#e6e5e1", linewidth=0.8)
        ax.set_axisbelow(True)

        ax = axes[r][1]
        keys = list(res["vs_real"])
        for i, k in enumerate(keys):
            e = res["vs_real"][k]
            lo, hi = e["ci"]
            col = "#1baf7a" if (e["sig"] and e["delta"] > 0) else MUTED
            ax.plot([lo, hi], [i, i], color=col, linewidth=2.2, solid_capstyle="round")
            ax.scatter([e["delta"]], [i], s=64, color=col, zorder=3,
                       edgecolors="white", linewidths=1.2)
            ax.annotate(f"{e['delta']:+.4f} [{lo:+.4f},{hi:+.4f}]  "
                        f"σ_train={e['sigma_train']:.4f}",
                        (max(hi, e["delta"]), i), textcoords="offset points",
                        xytext=(8, -3), fontsize=8, color=INK)
        ax.axvline(0, color=INK, linewidth=1.0)
        # 标签由 key 派生，避免条件数增减时与硬编码列表错位
        ax.set_yticks(range(len(keys)),
                      [f"真实 − {COND_LABELS[k.replace('real_minus_', '')]}" for k in keys],
                      fontsize=9)
        ax.set_ylim(-0.6, len(keys) - 0.3)
        ax.set_xlabel("属性信息的净贡献", fontsize=9, color=MUTED)
        ax.set_title("同权重、同样本，唯一变化=属性是否正确配对", fontsize=10, color=INK)
        ax.margins(x=0.6)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.grid(axis="x", color="#e6e5e1", linewidth=0.8)
        ax.set_axisbelow(True)
        verdicts.append((dk, verdict(res)))

    for row in axes:
        for a in row:
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
            a.tick_params(colors=MUTED, labelsize=8.5)
    fig.suptitle(f"OOD 属性 knockout：机制 A（属性迁移） vs 机制 B（正则副作用）· {metric}",
                 fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0.19 * len(dirs), 1, 0.94))   # 底部留白给判决句
    # 判决挂 figure 坐标系，避免与 axes 的 xlabel 重叠
    for i, (dk, txt) in enumerate(verdicts):
        fig.text(0.5, 0.155 - i * 0.09, "\n".join(textwrap.wrap(f"[{dk}] {txt}", width=64)),
                 ha="center", va="top", fontsize=8.5, color=INK, linespacing=1.5)
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="OOD 属性 knockout：机制 A/B 判别")
    ap.add_argument("--metric", choices=("marginal_worst", "overall"), default="marginal_worst")
    ap.add_argument("--direction", default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    payload = {}
    for dk in DIRECTIONS:
        if args.direction and dk != args.direction:
            continue
        r = run_direction(dk, args.metric, args.n_boot)
        if r is None:
            continue
        payload[dk] = r
        print(f"\n{'=' * 100}\n{DIRECTIONS[dk]['title']}  metric={args.metric}  "
              f"n={r['n']:,}  患者={r['n_clusters']:,}\n{'=' * 100}")
        print(f"  ERM 参照 = {r['point']['erm']:.4f}")
        print(f"\n  {'条件':<10s} {'HyperAdapt':>11s} {'Δ vs ERM':>10s} {'95% CI':>21s} "
              f"{'σ_train':>9s} {'t':>6s}")
        for c in CONDITIONS:
            e = r["vs_erm"][c]
            lo, hi = e["ci"]
            print(f"  {COND_LABELS[c]:<10s} {r['point'][c]:>11.4f} {e['delta']:>+10.4f} "
                  f"[{lo:>+8.4f},{hi:>+8.4f}] {e['sigma_train']:>9.4f} {e['t_ratio']:>6.2f}"
                  f"{'' if e['sig'] else '  n.s.'}")
        print(f"\n  属性信息的净贡献（real − knockout）：")
        for k, e in r["vs_real"].items():
            lo, hi = e["ci"]
            print(f"    {k:<20s} {e['delta']:>+9.4f} [{lo:>+8.4f},{hi:>+8.4f}]  "
                  f"σ_train={e['sigma_train']:.4f}  t={e['t_ratio']:+.2f}  "
                  f"{'**显著**' if e['sig'] else 'n.s.'}"
                  f"{'  ✅超噪声' if e['exceeds_train_noise'] else '  ❌未超噪声'}")
        print(f"\n  ⇒ 判决：{verdict(r)}")

    if payload:
        plot_all(payload, args.metric, args.out_dir / f"knockout_{args.metric}.png")
        out = args.out_dir / f"knockout_{args.metric}.json"
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n图 → {args.out_dir / f'knockout_{args.metric}.png'}")
        print(f"已落盘 → {out}")


if __name__ == "__main__":
    main()
