"""
OOD worst-group 的 **argmin 归属诊断**（M→C / C→M 双向）
=========================================================
回答一个前置问题：`docs/oof_regime_results.md` §1.6/§1.7 里 HyperAdapt 对 ERM 的
**+0.0045 / +0.0017（marginal worst AUC）**，究竟是「**同一个子群内真的更好**」，还是
「**worst 的 argmin 换了组**」造成的定义伪影？

文档 §4.5 已记录：worst-group 的动态 argmin 在小数据集上频繁翻转。OOD 上 n 很大（138k/199k），
argmin 理应更稳，但**必须验证而非假定**——因为 Δ 只有 0.002~0.005，一次 argmin 切换就足以制造它。

**三种 Δ 口径**（全部走 averaging：逐折算 → 折间平均 → 再取组）：
  1. `dynamic`      : `min_g mean_f AUC_HA(g) − min_g mean_f AUC_ERM(g)` —— **现行口径**，
                      两方法各自取各自的 argmin，若 argmin 不同则**比较的不是同一个组**。
  2. `fixed@erm`    : `mean_f AUC_HA(g*) − mean_f AUC_ERM(g*)`，`g*` = ERM 的 worst 组 ——
                      **同组比较**，免疫 argmin 切换。
  3. `min_delta`    : `min_g [mean_f AUC_HA(g) − mean_f AUC_ERM(g)]` —— 最保守：
                      要求「**没有任何一组变差**」，是 worst-group 精神的严格版本。
若 (1) 显著为正而 (2)(3) 不是，则 §1.6/§1.7 的胜利**主要来自 argmin 归属切换**。

**argmin 稳定性**用两种方式量化：
  · 逐折 argmin 计数（5 折里各组当选 worst 的次数）——反映折间噪声；
  · **cluster bootstrap 下 averaging-worst 的归属分布**——正规做法，直接给出「worst 组是谁」的
    不确定性。若某方法的归属分布弥散（无组占比 >50%），其 worst-group 数值不可作点比较。

聚类 bootstrap 与 `build_oof_results_averaging.py` 完全同源（同 `clusters()`、同 seed=42、
同 1000 次重采样、全方法共用同一批重采样索引 ⇒ 配对）。

运行（轻量 CPU，无需 GPU）：
    python scripts/ood_argmin_attribution.py                 # 双向
    python scripts/ood_argmin_attribution.py --regime M2C_OOD
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                                  # 无头环境
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# 系统 CJK 字体注册进 matplotlib，否则中文标签渲染成方框。
# 注意：NotoSansCJK 是 .ttc 合集，matplotlib 只注册其首个 face（family 名 "Noto Sans CJK JP"，
# 汉字字形集与 SC 变体共享），故不能按 "Noto Sans CJK SC" 寻址。
_CJK_FONTS = (
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "Noto Sans CJK JP"),
    ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", "Droid Sans Fallback"),
)
_families = []
for _path, _name in _CJK_FONTS:
    if Path(_path).exists():
        font_manager.fontManager.addfont(_path)
        _families.append(_name)
if _families:
    plt.rcParams["font.family"] = _families + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False             # 负号用 ASCII，避免缺字

from scripts.build_oof_results_averaging import (
    SPECS, Spec, build_masks, clusters, load_folds,
)
from scripts.eaudit_m2c_full_target import sorted_view, weighted_auc

OUT_DIR = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/ood_cxr/argmin_attribution")
N_BOOT = 1000
BOOT_SEED = 42                                          # 与 build_oof_results_averaging 一致
PAIR = ("erm", "hyperadapt")                            # 本诊断只关心这一对
# dataviz 分类槽位 1/2（validated，all-pairs 安全）
COLORS = {"erm": "#2a78d6", "hyperadapt": "#eb6834"}
LABELS = {"erm": "ERM", "hyperadapt": "HyperAdapt"}
INK, MUTED = "#0b0b0b", "#52514e"


class GroupViews:
    """逐折 × 逐组的预排序视图；`per_group` 在给定 bootstrap 权重下返回**折均**逐组 AUC。"""

    def __init__(self, spec: Spec, folds: list[dict]) -> None:
        self.groups: list[str] = list(build_masks(spec, folds[0]["attrs"]).keys())
        self.marginal: list[str] = [g for g in self.groups if "|" not in g]
        self.n_folds = len(folds)
        self.v: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
        for k, f in enumerate(folds):
            for g, m in build_masks(spec, f["attrs"]).items():
                self.v[(k, g)] = sorted_view(f["y"], f["s"], m)

    def per_fold(self, offs: np.ndarray, w: np.ndarray) -> dict[str, list[float]]:
        """逐组的**逐折 AUC 列表**（无定义折已剔除）。"""
        out: dict[str, list[float]] = {g: [] for g in self.groups}
        for k in range(self.n_folds):
            wk = w[offs[k]:offs[k + 1]]
            for g in self.groups:
                ig, yg = self.v[(k, g)]
                a = weighted_auc(yg, wk[ig])
                if not np.isnan(a):
                    out[g].append(a)
        return out

    def per_group(self, offs: np.ndarray, w: np.ndarray) -> dict[str, float]:
        """averaging 口径的逐组**折均** AUC（先平均、后取组，与主结果同源）。"""
        return {g: float(np.mean(v)) for g, v in self.per_fold(offs, w).items() if v}


def three_deltas(mu_a: dict[str, float], mu_b: dict[str, float],
                 groups: list[str]) -> tuple[float, float, float]:
    """
    在给定的一组逐组折均 AUC 上算三种 Δ 口径（`a` 为处理组 HyperAdapt，`b` 为参照 ERM）。

    Args:
        mu_a: 处理组 {组名: 折均 AUC}。
        mu_b: 参照组 {组名: 折均 AUC}。
        groups: 参与取 min 的组集合（marginal 或 canonical）。

    Returns:
        (dynamic, fixed@erm, min_delta) 三个 Δ 标量。
    """
    gs = [g for g in groups if g in mu_a and g in mu_b]
    dynamic = min(mu_a[g] for g in gs) - min(mu_b[g] for g in gs)
    g_star = min(gs, key=lambda g: mu_b[g])             # 参照组自己的 worst 组
    fixed = mu_a[g_star] - mu_b[g_star]
    min_delta = min(mu_a[g] - mu_b[g] for g in gs)
    return dynamic, fixed, min_delta


def run_regime(spec: Spec, n_boot: int) -> dict | None:
    """跑一个 OOD regime 的完整 argmin 归属诊断，返回结构化结果。"""
    cfg = json.loads(spec.cfg.read_text())["config"]
    folds_by, views = {}, {}
    for me in PAIR:
        fl = load_folds(spec, me, cfg[me])
        if fl is None:
            print(f"[跳过] {spec.name}: 缺 {me} 预测")
            return None
        folds_by[me], views[me] = fl, GroupViews(spec, fl)

    cidx, offs = clusters(spec, folds_by["erm"])
    n_cl = int(cidx.max() + 1)
    ones = np.ones(offs[-1])
    groups = views["erm"].groups
    marginal = views["erm"].marginal

    # ---- 点估计：逐组折均 AUC + 逐折 argmin 计数 ----
    mu = {m: views[m].per_group(offs, ones) for m in PAIR}
    per_fold = {m: views[m].per_fold(offs, ones) for m in PAIR}
    fold_argmin = {m: {"marginal": [], "canonical": []} for m in PAIR}
    for m in PAIR:
        for k in range(views[m].n_folds):
            # 该折各组 AUC（只取该折有定义的组）
            f_auc = {g: per_fold[m][g][k] for g in groups if len(per_fold[m][g]) > k}
            fm = {g: v for g, v in f_auc.items() if "|" not in g}
            fold_argmin[m]["marginal"].append(min(fm, key=fm.get))
            fold_argmin[m]["canonical"].append(min(f_auc, key=f_auc.get))

    obs = {scope: three_deltas(mu["hyperadapt"], mu["erm"],
                               marginal if scope == "marginal" else groups)
           for scope in ("marginal", "canonical")}

    # ---- cluster bootstrap：配对 Δ 的 CI + argmin 归属分布 ----
    rng = np.random.default_rng(BOOT_SEED)
    boot_delta = {s: {k: [] for k in ("dynamic", "fixed", "min_delta")}
                  for s in ("marginal", "canonical")}
    boot_argmin = {m: {"marginal": [], "canonical": []} for m in PAIR}
    for _ in range(n_boot):
        draw = rng.integers(0, n_cl, size=n_cl)
        w = np.bincount(draw, minlength=n_cl).astype(float)[cidx]
        mub = {m: views[m].per_group(offs, w) for m in PAIR}
        for m in PAIR:
            boot_argmin[m]["marginal"].append(min(marginal, key=lambda g: mub[m][g]))
            boot_argmin[m]["canonical"].append(min(groups, key=lambda g: mub[m][g]))
        for scope, gs in (("marginal", marginal), ("canonical", groups)):
            d, f, md = three_deltas(mub["hyperadapt"], mub["erm"], gs)
            boot_delta[scope]["dynamic"].append(d)
            boot_delta[scope]["fixed"].append(f)
            boot_delta[scope]["min_delta"].append(md)

    res = {
        "regime": spec.name, "n": int(offs[-1]), "n_clusters": n_cl, "n_boot": n_boot,
        "config": {m: cfg[m] for m in PAIR},
        "group_mean_auc": {m: mu[m] for m in PAIR},
        "worst_group": {m: {"marginal": min(marginal, key=lambda g: mu[m][g]),
                            "canonical": min(groups, key=lambda g: mu[m][g])} for m in PAIR},
        "fold_argmin": fold_argmin,
        "boot_argmin_share": {
            m: {sc: {g: c / n_boot for g, c in
                     zip(*np.unique(boot_argmin[m][sc], return_counts=True))}
                for sc in ("marginal", "canonical")} for m in PAIR},
        "delta": {},
    }
    for scope in ("marginal", "canonical"):
        res["delta"][scope] = {}
        for j, key in enumerate(("dynamic", "fixed", "min_delta")):
            arr = np.asarray(boot_delta[scope][key])
            lo, hi = np.percentile(arr, [2.5, 97.5])
            res["delta"][scope][key] = {"delta": float(obs[scope][j]),
                                        "ci": [float(lo), float(hi)],
                                        "sig": bool(lo > 0 or hi < 0)}
    return res


def plot_regime(res: dict, path: Path) -> None:
    """出三面板诊断图：逐组折均 AUC / bootstrap argmin 归属 / 三口径 Δ。"""
    marg = [g for g in res["group_mean_auc"]["erm"] if "|" not in g]
    marg = sorted(marg, key=lambda g: res["group_mean_auc"]["erm"][g])   # 按 ERM 由弱到强
    y = np.arange(len(marg))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 0.42 * len(marg) + 3.4),
                             gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})

    # --- 面板 1：逐组折均 AUC（点 = 折均，横线 = 折间 min–max） ---
    ax = axes[0]
    for m in PAIR:
        xs = [res["group_mean_auc"][m][g] for g in marg]
        fo = [res["fold_argmin"][m]["marginal"]]                        # 仅用于下方标注
        ax.scatter(xs, y + (0.16 if m == "hyperadapt" else -0.16), s=42, zorder=3,
                   color=COLORS[m], label=LABELS[m], edgecolors="white", linewidths=1.2)
        gw = res["worst_group"][m]["marginal"]                          # 各自的 worst 组
        ax.scatter([res["group_mean_auc"][m][gw]],
                   [marg.index(gw) + (0.16 if m == "hyperadapt" else -0.16)],
                   s=170, facecolors="none", edgecolors=COLORS[m], linewidths=2.0, zorder=4)
        del fo
    ax.set_yticks(y, marg, fontsize=8.5)
    ax.set_xlabel("折均 AUC（averaging 口径）", fontsize=9, color=MUTED)
    ax.set_title("逐 marginal 子群 AUC（空心圈 = 该方法的 worst）", fontsize=10, color=INK)
    ax.legend(fontsize=8.5, loc="lower right", frameon=False)
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)

    # --- 面板 2：bootstrap 下 worst 归属份额 ---
    ax = axes[1]
    h = 0.34
    for i, m in enumerate(PAIR):
        share = [res["boot_argmin_share"][m]["marginal"].get(g, 0.0) for g in marg]
        ax.barh(y + (h / 2 if i else -h / 2), share, height=h, color=COLORS[m],
                label=LABELS[m], edgecolor="white", linewidth=0.8)
    ax.set_yticks(y, [""] * len(marg))
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"{res['n_boot']} 次 cluster bootstrap 中当选 worst 的比例",
                  fontsize=9, color=MUTED)
    ax.set_title("worst 归属的不确定性", fontsize=10, color=INK)
    ax.axvline(0.5, color=MUTED, linestyle="--", linewidth=1.0)
    # worst 归属常 100% 集中在最底一行 ⇒ 图例放上方，避免压住数据条
    ax.legend(fontsize=8.5, loc="upper right", frameon=False)
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)

    # --- 面板 3：三种 Δ 口径 + 95% CI ---
    ax = axes[2]
    keys = ("dynamic", "fixed", "min_delta")
    names = ("dynamic argmin\n（现行口径）", "fixed @ ERM worst\n（同组比较）",
             "min over groups\n（最保守）")
    for i, k in enumerate(keys):
        d = res["delta"]["marginal"][k]
        lo, hi = d["ci"]
        col = "#2a78d6" if d["sig"] and d["delta"] > 0 else ("#e34948" if d["sig"] else MUTED)
        ax.plot([lo, hi], [i, i], color=col, linewidth=2.0, solid_capstyle="round")
        ax.scatter([d["delta"]], [i], s=64, color=col, zorder=3,
                   edgecolors="white", linewidths=1.2)
        ax.annotate(f"{d['delta']:+.4f}  [{lo:+.4f},{hi:+.4f}]  "
                    f"{'显著' if d['sig'] else 'n.s.'}",
                    (max(hi, d["delta"]), i), textcoords="offset points", xytext=(8, -3),
                    fontsize=8.5, color=INK)
    ax.axvline(0, color=INK, linewidth=1.0)
    ax.set_yticks(range(len(keys)), names, fontsize=8.5)
    ax.set_ylim(-0.6, len(keys) - 0.15)
    ax.set_xlabel("Δ marginal-worst AUC（HyperAdapt − ERM）", fontsize=9, color=MUTED)
    ax.set_title("三种 worst-group Δ 口径", fontsize=10, color=INK)
    ax.margins(x=0.55)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))     # Δ 量级极小，密刻度会重叠
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)

    for a in axes:
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.tick_params(colors=MUTED, labelsize=8.5)
    fig.suptitle(f"{res['regime']}  worst-group argmin 归属诊断"
                 f"（n={res['n']:,}，患者 cluster={res['n_clusters']:,}）",
                 fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)


def report(res: dict) -> None:
    """把关键判据打到 stdout。"""
    print(f"\n{'=' * 96}\n{res['regime']}  n={res['n']:,}  clusters={res['n_clusters']:,}  "
          f"config={res['config']}\n{'=' * 96}")
    for scope in ("marginal", "canonical"):
        we, wh = (res["worst_group"]["erm"][scope], res["worst_group"]["hyperadapt"][scope])
        same = "同组 ✅" if we == wh else "**不同组 ⚠️**"
        print(f"\n[{scope}]  ERM worst = {we}   HyperAdapt worst = {wh}   → {same}")
        for m in PAIR:
            fa = res["fold_argmin"][m][scope]
            uniq, cnt = np.unique(fa, return_counts=True)
            top = uniq[cnt.argmax()]
            sh = res["boot_argmin_share"][m][scope]
            best = max(sh, key=sh.get)
            print(f"   {LABELS[m]:<11s} 逐折 argmin: {dict(zip(uniq, cnt.tolist()))}  "
                  f"（主导 {top} {cnt.max()}/5）  | bootstrap 主导 {best} {sh[best]:.0%}")
        print(f"   {'口径':<22s} {'Δ':>10s} {'95% CI':>22s}   判决")
        for k, nm in (("dynamic", "dynamic argmin(现行)"), ("fixed", "fixed @ ERM worst"),
                      ("min_delta", "min over groups")):
            d = res["delta"][scope][k]
            lo, hi = d["ci"]
            print(f"   {nm:<22s} {d['delta']:>+10.4f} [{lo:>+8.4f},{hi:>+8.4f}]   "
                  f"{'**显著**' if d['sig'] else 'n.s.'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="OOD worst-group argmin 归属诊断")
    ap.add_argument("--regime", default=None, help="M2C_OOD / C2M_OOD；缺省=两者")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {}
    for spec in SPECS:
        if not spec.name.endswith("_OOD"):
            continue
        if args.regime and spec.name != args.regime:
            continue
        r = run_regime(spec, args.n_boot)
        if r is None:
            continue
        out[spec.name] = r
        report(r)
        p = OUT_DIR / f"{spec.name.lower()}_argmin_attribution.png"
        plot_regime(r, p)
        print(f"\n   图 → {p}")
    (OUT_DIR / "argmin_attribution.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已落盘 → {OUT_DIR / 'argmin_attribution.json'}")


if __name__ == "__main__":
    main()
