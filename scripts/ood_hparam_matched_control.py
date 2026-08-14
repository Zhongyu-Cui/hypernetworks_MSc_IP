"""
OOD 增益的**超参匹配对照**（M→C / C→M）
=========================================
`docs/oof_regime_results.md` §1.6/§1.7：HyperAdapt 在两个 OOD 方向都显著胜 ERM
（marginal worst AUC +0.0045 / +0.0017）。但两者**选中的超参不同**：

    M→C：ERM = lr1e-04_wd1e-04   vs   HyperAdapt = lr1e-04_wd1e-03  （wd 大 10×）
    C→M：ERM = lr1e-04_wd1e-04   vs   HyperAdapt = lr3e-04_wd1e-04  （lr 大 3×）

两个方向 HyperAdapt 都落在**更强正则 / 更大学习率**一侧——而这正是 OOD 泛化文献里不靠任何
架构就能拿到的增益。所以「HN 赢」的最简解释可能是 **Minimax-Pareto 选择器（S1，按 source 域
折均 marginal-worst 选 config）替 HN 选到了更 OOD-友好的超参**，与条件化无关。这与
`docs/oof_regime_results.md` §3.3「增益可能来自范围以外的轴（容量/参数化/初始化）」是同一件事，
但更具体、也更容易证伪。

**对照设计**：补训**与各方向 HyperAdapt 同超参的 ERM**（作业 hpmatch_mimic_erm / hpmatch_chex_erm），
跑同一套 OOD 评估（`slurm/hpmatch_ood_cv.sh` → `cv5_hpmatch/predictions/`，与主结果隔离），
形成三臂：

    A) ERM@selected   —— 主结果里的 ERM（lr1e-04_wd1e-04）
    B) ERM@matched    —— 超参对齐 HyperAdapt 的 ERM（**新增对照臂**）
    C) HyperAdapt     —— 主结果里的 HyperAdapt

**判据**：
  · `C − A` 复现主结果的显著正（+0.0045 / +0.0017）——健全性检查。
  · **`C − B` 若退化为 n.s.** ⇒ OOD 增益**归因于超参**，不是条件化 ⇒ §1.6/§1.7 的
    「HyperAdapt 是唯一一致显著为正的 HN」失去架构层面的解释。
  · `B − A` 度量**纯超参效应**的大小；若它本身就吃掉大部分 `C − A`，结论同上。

口径与 `build_oof_results_averaging.py` 完全同源：averaging（逐折算 → 折间平均 → 再取组）、
患者级配对 cluster bootstrap（同 seed=42、同重采样索引 ⇒ 三臂配对）。

运行（轻量 CPU；需先跑完 `slurm/hpmatch_ood_cv.sh`）：
    python scripts/ood_hparam_matched_control.py
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from scripts.build_oof_results_averaging import SPECS, Spec, clusters, load_folds
from scripts.eaudit_m2c_full_target import sorted_view, weighted_auc
from scripts.ood_argmin_attribution import GroupViews

# 中文字体（同 ood_argmin_attribution：ttc 只注册首个 face，family 名为 "Noto Sans CJK JP"）
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
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/ood_cxr/hparam_matched")
N_BOOT = 1000
BOOT_SEED = 42
HPMATCH_SUBDIR = "cv5_hpmatch"

# 各 regime 的 HyperAdapt 选中配置 = 对照臂 ERM 应当匹配的超参
# （OOD 两向与 slurm/hpmatch_ood_cv.sh 一致；两个 ID regime 的对照臂 ERM 由同一批 hpmatch 训练
#  作业顺带产出——MIMIC 的 lr1e-04_wd1e-03 与 CheXpert 的 lr3e-04_wd1e-04 正是各自 HN 的选中配置，
#  故 ID 侧**零额外计算**即可做同一对照。）
MATCHED_CONFIG = {"M2C_OOD": "lr1e-04_wd1e-03", "C2M_OOD": "lr3e-04_wd1e-04",
                  "MIMIC": "lr1e-04_wd1e-03", "CheXpert": "lr3e-04_wd1e-04"}
# OOD 的对照臂预测在隔离子目录；ID 的对照臂与主结果同目录，仅 config tag 不同
ISOLATED_REGIMES = ("M2C_OOD", "C2M_OOD")

ARMS = ("erm_selected", "erm_matched", "hyperadapt")
LABELS = {"erm_selected": "ERM@selected", "erm_matched": "ERM@matched（超参对照）",
          "hyperadapt": "HyperAdapt"}
# dataviz 分类槽位 1/3/2（前三槽 all-pairs 安全）
COLORS = {"erm_selected": "#2a78d6", "erm_matched": "#1baf7a", "hyperadapt": "#eb6834"}
INK, MUTED = "#0b0b0b", "#52514e"
METRICS = ("overall", "marginal_worst", "canonical_worst", "gap")
METRIC_NAMES = {"overall": "Overall AUC", "marginal_worst": "marginal worst AUC",
                "canonical_worst": "canonical worst AUC", "gap": "canonical gap"}


def metrics_from_groups(views: GroupViews, offs: np.ndarray, w: np.ndarray,
                        overall_view: dict[int, tuple[np.ndarray, np.ndarray]]) -> tuple:
    """
    在给定 bootstrap 权重下算一臂的四个指标（averaging 口径：逐折算 → 折间平均 → 再 min/max）。

    Args:
        views: 该臂的逐折逐组预排序视图。
        offs: 折边界偏移（长度 n_folds+1）。
        w: 样本级 bootstrap 权重。
        overall_view: 逐折的「全体样本」预排序视图。

    Returns:
        (Overall, canonical worst, marginal worst, gap) 四元组。
    """
    ov = []
    for k in range(views.n_folds):
        i, ys = overall_view[k]
        a = weighted_auc(ys, w[offs[k]:offs[k + 1]][i])
        if not np.isnan(a):
            ov.append(a)
    mu = views.per_group(offs, w)
    marg = {g: v for g, v in mu.items() if "|" not in g}
    canon_w, canon_b = min(mu.values()), max(mu.values())
    return float(np.mean(ov)), canon_w, min(marg.values()), canon_b - canon_w


def run_regime(spec: Spec, n_boot: int) -> dict | None:
    """跑一个 OOD regime 的三臂超参匹配对照。缺对照臂预测时返回 None。"""
    cfg = json.loads(spec.cfg.read_text())["config"]
    matched_tag = MATCHED_CONFIG[spec.name]
    if matched_tag != cfg["hyperadapt"]:
        raise ValueError(f"{spec.name}: 对照配置 {matched_tag} 与 HyperAdapt 选中配置 "
                         f"{cfg['hyperadapt']} 不一致——脚本常量已过期，请核对 selected_configs.json")

    # OOD：对照臂预测在隔离目录 cv5_hpmatch/predictions/（其余寻址/口径与主 spec 相同）；
    # ID：对照臂与主结果同目录，仅 config tag 不同，无需隔离
    spec_match = copy.copy(spec)
    if spec.name in ISOLATED_REGIMES:
        spec_match.pred_dir = spec.pred_dir.parent.parent / HPMATCH_SUBDIR / "predictions"

    sources = {"erm_selected": (spec, "erm", cfg["erm"]),
               "erm_matched": (spec_match, "erm", matched_tag),
               "hyperadapt": (spec, "hyperadapt", cfg["hyperadapt"])}
    folds_by, views, ov_views, tags = {}, {}, {}, {}
    for arm, (sp, method, tag) in sources.items():
        fl = load_folds(sp, method, tag)
        if fl is None:
            print(f"[跳过] {spec.name}: 缺 {arm} 预测（{sp.pred_dir}/{method}_{tag}_seed*.npz）")
            return None
        folds_by[arm], views[arm], tags[arm] = fl, GroupViews(sp, fl), tag
        ov_views[arm] = {k: sorted_view(f["y"], f["s"], np.ones(len(f["y"]), bool))
                         for k, f in enumerate(fl)}

    cidx, offs = clusters(spec, folds_by["erm_selected"])
    n_cl = int(cidx.max() + 1)
    ones = np.ones(offs[-1])
    obs = {a: metrics_from_groups(views[a], offs, ones, ov_views[a]) for a in ARMS}

    rng = np.random.default_rng(BOOT_SEED)
    boot: dict[str, list[tuple]] = {a: [] for a in ARMS}
    for _ in range(n_boot):
        draw = rng.integers(0, n_cl, size=n_cl)
        w = np.bincount(draw, minlength=n_cl).astype(float)[cidx]
        for a in ARMS:
            boot[a].append(metrics_from_groups(views[a], offs, w, ov_views[a]))
    B = {a: np.asarray(boot[a]) for a in ARMS}
    # metrics_from_groups 的返回序 = (overall, canonical, marginal, gap)
    col = {"overall": 0, "canonical_worst": 1, "marginal_worst": 2, "gap": 3}

    res = {"regime": spec.name, "n": int(offs[-1]), "n_clusters": n_cl, "n_boot": n_boot,
           "config": tags,
           "point": {a: {m: float(obs[a][col[m]]) for m in METRICS} for a in ARMS},
           "vs": {}}
    for treat, base in (("hyperadapt", "erm_selected"), ("hyperadapt", "erm_matched"),
                        ("erm_matched", "erm_selected")):
        e = {}
        for m in METRICS:
            d = B[treat][:, col[m]] - B[base][:, col[m]]
            lo, hi = np.percentile(d, [2.5, 97.5])
            e[m] = {"delta": float(obs[treat][col[m]] - obs[base][col[m]]),
                    "ci": [float(lo), float(hi)], "sig": bool(lo > 0 or hi < 0)}
        res["vs"][f"{treat}_vs_{base}"] = e
    return res


def plot_regime(res: dict, path: Path) -> None:
    """出两面板图：三臂点估计 / 三组配对 Δ（marginal worst 与 Overall）。"""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4),
                             gridspec_kw={"width_ratios": [1.0, 1.35]})

    # --- 面板 1：三臂在四个指标上的点估计 ---
    ax = axes[0]
    xs = np.arange(len(METRICS))
    width = 0.26
    for i, a in enumerate(ARMS):
        vals = [res["point"][a][m] for m in METRICS]
        ax.bar(xs + (i - 1) * width, vals, width * 0.92, color=COLORS[a],
               label=LABELS[a], edgecolor="white", linewidth=0.8)
    ax.set_xticks(xs, [METRIC_NAMES[m] for m in METRICS], fontsize=8, rotation=12)
    ax.set_ylabel("AUC", fontsize=9, color=MUTED)
    ax.set_title("三臂点估计（averaging 口径）", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)

    # --- 面板 2：三组配对 Δ + 95% CI（关键判据面板） ---
    ax = axes[1]
    rows = [("hyperadapt_vs_erm_selected", "marginal_worst", "HA − ERM@selected\n（主结果口径）"),
            ("hyperadapt_vs_erm_matched", "marginal_worst", "HA − ERM@matched\n（**超参匹配后**）"),
            ("erm_matched_vs_erm_selected", "marginal_worst", "ERM@matched − ERM@selected\n（纯超参效应）"),
            ("hyperadapt_vs_erm_selected", "overall", "HA − ERM@selected（Overall）"),
            ("hyperadapt_vs_erm_matched", "overall", "HA − ERM@matched（Overall）")]
    for i, (key, metric, name) in enumerate(rows):
        d = res["vs"][key][metric]
        lo, hi = d["ci"]
        col = "#2a78d6" if (d["sig"] and d["delta"] > 0) else ("#e34948" if d["sig"] else MUTED)
        ax.plot([lo, hi], [i, i], color=col, linewidth=2.0, solid_capstyle="round")
        ax.scatter([d["delta"]], [i], s=60, color=col, zorder=3,
                   edgecolors="white", linewidths=1.2)
        ax.annotate(f"{d['delta']:+.4f} [{lo:+.4f},{hi:+.4f}] {'显著' if d['sig'] else 'n.s.'}",
                    (max(hi, d["delta"]), i), textcoords="offset points", xytext=(8, -3),
                    fontsize=8, color=INK)
    ax.axhline(2.5, color="#e6e5e1", linewidth=1.2)          # 分隔 worst / Overall 两块
    ax.axvline(0, color=INK, linewidth=1.0)
    ax.set_yticks(range(len(rows)), [r[2] for r in rows], fontsize=8)
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.set_xlabel("Δ AUC", fontsize=9, color=MUTED)
    ax.set_title("配对 Δ 与 95% CI（患者级 cluster bootstrap）", fontsize=10, color=INK)
    ax.margins(x=0.55)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)

    for a in axes:
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.tick_params(colors=MUTED, labelsize=8)
    fig.suptitle(f"{res['regime']}  超参匹配对照（n={res['n']:,}，患者 cluster={res['n_clusters']:,}；"
                 f"ERM@matched = {res['config']['erm_matched']}）", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)


def report(res: dict) -> None:
    """把三臂点估计与关键判据打到 stdout。"""
    print(f"\n{'=' * 100}\n{res['regime']}  n={res['n']:,}  clusters={res['n_clusters']:,}"
          f"\n{'=' * 100}")
    print(f"{'臂':<26s} {'config':<18s} " + " ".join(f"{METRIC_NAMES[m]:>20s}" for m in METRICS))
    for a in ARMS:
        print(f"{LABELS[a]:<26s} {res['config'][a]:<18s} "
              + " ".join(f"{res['point'][a][m]:>20.4f}" for m in METRICS))
    print()
    for key in res["vs"]:
        print(f"  [{key}]")
        for m in METRICS:
            d = res["vs"][key][m]
            lo, hi = d["ci"]
            print(f"    {METRIC_NAMES[m]:<22s} {d['delta']:>+9.4f} "
                  f"[{lo:>+8.4f},{hi:>+8.4f}]  {'**显著**' if d['sig'] else 'n.s.'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="OOD 增益的超参匹配对照（三臂）")
    ap.add_argument("--regime", default=None, help="M2C_OOD / C2M_OOD；缺省=两者")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {}
    for spec in SPECS:
        if spec.name not in MATCHED_CONFIG:
            continue
        if args.regime and spec.name != args.regime:
            continue
        r = run_regime(spec, args.n_boot)
        if r is None:
            continue
        out[spec.name] = r
        report(r)
        p = OUT_DIR / f"{spec.name.lower()}_hparam_matched.png"
        plot_regime(r, p)
        print(f"\n   图 → {p}")
    if out:
        (OUT_DIR / "hparam_matched.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n已落盘 → {OUT_DIR / 'hparam_matched.json'}")


if __name__ == "__main__":
    main()
