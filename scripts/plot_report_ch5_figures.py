"""
第 5 章结果图（写入 `report/figures/`，供 LaTeX `\includegraphics` 使用）。

三张图：
  1. `ch5_id_worst.pdf`   —— ID 框架，各臂相对 ERM 的 marginal worst-group AUC 增量，
                             四库分面，配对 cluster bootstrap 95% CI。
  2. `ch5_ood_noise.pdf`  —— OOD 框架，效应量 |d̄| 与训练噪声地板 sigma_train 并排，双方向。
  3. `ch5_rho.pdf`        —— 消融各 cell 的逐层 rho / rho_between（HAM 与 MIMIC）。

运行（轻量 CPU）：
    PYTHONPATH=. python scripts/plot_report_ch5_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from src.paths import OUTPUTS_DIR

OUTPUTS = OUTPUTS_DIR
FIGDIR = Path(__file__).resolve().parents[1] / "report" / "figures"

# 报告正文字体为 XCharter（Charter 家族），图内文字取 serif 以求一致。
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8.5,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
})

INK, MUTED, GRID = "#22211f", "#6d6b66", "#e3e2de"
POS, NEG, NS = "#2a6fb5", "#c0453f", "#a8a6a1"

# HAM 的三个超网络臂一律取 sex+age 双轴 run（age-only 一族已作废）。
ID_ARMS = [
    ("swad", "SWAD"),
    ("groupdro", "GroupDRO"),
    ("hyperhead", "HyperHead"),
    ("hyperfusion", "HyperFusion"),
    ("hyperadapt", "HyperAdapt"),
]
SEXAGE = {"hyperhead", "hyperfusion", "hyperadapt"}
DATASETS = [("HAM10000", "HAM10000"), ("Fitzpatrick", "Fitzpatrick17k"),
            ("MIMIC", "MIMIC-CXR"), ("CheXpert", "CheXpert")]


def _style(ax: plt.Axes) -> None:
    """统一去掉上/右边框并设置刻度颜色。"""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=2.5, width=0.6)


def plot_id_worst(path: Path) -> None:
    """
    ID 框架的两个主终点相对 ERM 的增量，上行 Overall AUC、下行 marginal worst-group AUC。

    两行都画是必要的，因为正文对 Overall 与 worst-group 各自都做了显著性陈述，
    只画一行会让另一半读数在正文里无处可查。
    """
    res = json.loads((OUTPUTS / "conditioning_ablation" /
                      "oof_results_averaging.json").read_text())
    HOLM = json.loads((OUTPUTS / "analysis" / "id_holm_verdicts.json").read_text())
    TITLE_OF = dict(DATASETS)
    metrics = [("overall", "$\\Delta$ Overall AUC"),
               ("marginal_worst", "$\\Delta$ marginal worst-group AUC")]
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 4.3))
    for r, (metric, ylab) in enumerate(metrics):
        for c, (key, title) in enumerate(DATASETS):
            ax = axes[r][c]
            vs = res[key]["vs"]
            deltas, los, his, labels, colors = [], [], [], [], []
            for arm, label in ID_ARMS:
                name = f"{arm}_sexage" if (key == "HAM10000" and arm in SEXAGE) else arm
                e = vs[f"{name}_vs_erm"][metric]
                deltas.append(e["delta"])
                los.append(e["delta"] - e["ci"][0])
                his.append(e["ci"][1] - e["delta"])
                labels.append(label)
                # 主终点按 Holm 校正后的判决着色（family = 本框架每库实际报告的 11 个比较，
                # 见 scripts/id_holm_correction.py）；Overall 是次要终点、不并入该 family，
                # 故仍按未校正区间着色。两者的区别在图题里写明。
                if metric == "marginal_worst":
                    sig = HOLM[TITLE_OF[key]][f"{label} - ERM"]
                else:
                    sig = e["sig"]
                colors.append((POS if e["delta"] > 0 else NEG) if sig else NS)
            y = np.arange(len(labels))[::-1]
            ax.barh(y, deltas, height=0.62, color=colors, edgecolor="white",
                    linewidth=0.5)
            ax.errorbar(deltas, y, xerr=[los, his], fmt="none", ecolor=INK,
                        elinewidth=0.7, capsize=1.8, capthick=0.7)
            ax.axvline(0, color=INK, linewidth=0.7)
            ax.set_yticks(y, labels if c == 0 else [""] * len(labels), fontsize=8)
            # 两个大数据集的效应小一个量级，共用一把尺子会贴在零线上读不出，故右侧两栏放大 5 倍
            wide = key in ("HAM10000", "Fitzpatrick")
            if r == 0:
                ax.set_title(title if wide else f"{title} ($5\\times$ scale)",
                             fontsize=8.5, color=INK, pad=4)
            if wide:
                ax.set_xlim(-0.115, 0.115)
                ax.set_xticks([-0.1, 0, 0.1])
                ax.set_xticklabels(["$-$.10", "0", ".10"], fontsize=7.5)
            else:
                ax.set_xlim(-0.023, 0.023)
                ax.set_xticks([-0.02, 0, 0.02])
                ax.set_xticklabels(["$-$.02", "0", ".02"], fontsize=7.5)
            ax.grid(axis="x", color=GRID, linewidth=0.5)
            ax.set_axisbelow(True)
            _style(ax)
        axes[r][0].set_ylabel(ylab, fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_ood_ratio(path: Path) -> None:
    """
    OOD 确认性家族（3 HN x 3 baseline x 2 方向）的 H2 判决图。

    画比值 r = |d_bar| / sigma_train 而非效应与噪声两根柱：一条 r=1 的参考线就把判决讲完，
    且一张图容得下全部 18 个对比（双柱版只装得下 vs-ERM 的 5 个）。
    """
    src = OUTPUTS / "ood_cxr" / "variance_decomposition_family" / "mw"
    d = json.loads((src / "variance_decomposition_marginal_worst.json").read_text())
    hn = [("hyperhead", "HyperHead"), ("hyperfusion", "HyperFusion"),
          ("hyperadapt", "HyperAdapt")]
    base = [("erm", "ERM"), ("swad", "SWAD"), ("groupdro", "GroupDRO")]
    titles = {"M2C": "MIMIC-CXR $\\rightarrow$ CheXpert",
              "C2M": "CheXpert $\\rightarrow$ MIMIC-CXR"}
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharex=True)
    for ax, dk in zip(axes, ("M2C", "C2M")):
        e = d[dk]
        cell = {m: np.asarray(v) for m, v in e["per_cell"].items()}
        labels, ratios, colours = [], [], []
        for h, hlab in hn:
            for b, blab in base:
                diff = cell[h] - cell[b]
                mean = float(diff.mean())
                f, t = diff.shape
                row, col, grand = diff.mean(1), diff.mean(0), diff.mean()
                ms_t = f * ((col - grand) ** 2).sum() / (t - 1)
                ms_e = ((diff - row[:, None] - col[None, :] + grand) ** 2).sum() / ((f - 1) * (t - 1))
                sigma = float(np.sqrt(max((ms_t - ms_e) / f, 0.0) + max(ms_e, 0.0)))
                labels.append(f"{hlab} $-$ {blab}")
                ratios.append(abs(mean) / sigma)
                colours.append(POS if mean > 0 else NEG)
        y = np.arange(len(labels))[::-1]
        ax.barh(y, ratios, height=0.66, color=colours, edgecolor="white", linewidth=0.5)
        # 正文按两位小数引用这些比值，故在柱端标出数值，避免只能从刻度上估读
        for yi, r in zip(y, ratios):
            ax.annotate(f"{r:.2f}", (r, yi), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=6.2, color=MUTED)
        ax.axvline(1.0, color=INK, linewidth=0.9, linestyle="--")
        ax.set_yticks(y, labels if ax is axes[0] else [""] * len(labels), fontsize=7.5)
        # 分隔三个超网络的行组，便于按行块阅读
        for cut in (2.5, 5.5):
            ax.axhline(cut, color=GRID, linewidth=0.7)
        ax.set_title(titles[dk], fontsize=8.5, color=INK, pad=4)
        ax.set_xlim(0, 1.65)
        ax.set_xticks([0, 0.5, 1.0, 1.5])
        ax.tick_params(labelsize=7.5)
        ax.grid(axis="x", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        _style(ax)
    axes[0].annotate("clears $\\sigma_{\\mathrm{train}}$", xy=(1.02, 8.4), fontsize=7,
                     color=MUTED, style="italic", ha="left")
    fig.supxlabel("$|\\bar{d}| \\,/\\, \\sigma_{\\mathrm{train}}$", fontsize=8.5,
                  color=MUTED, y=0.01)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _stage_of(layer: str) -> str | None:
    """把层名归并到 stage（layer1--layer4 或 fc），非条件化层返回 None。"""
    if layer == "fc":
        return "fc"
    for k in ("layer1", "layer2", "layer3", "layer4"):
        if layer.startswith(k):
            return k
    return None


def plot_rho(path: Path) -> None:
    """
    左两格：消融各 cell 的 stage 级 rho_between（对数纵轴，fc 与 conv 差一到两个数量级）。
    右一格：C-Deep / C-Full 的 conv rho_between 在三种口径下——两种 checkpoint 选择规则，
            以及关掉 head 通路重训之后——用以显示该通路的弱不随这些设定改变。
    """
    srcs = [("HAM10000", OUTPUTS / "conditioning_ablation" / "ham10000" / "cv5" /
             "e1_rho_between.json"),
            ("MIMIC-CXR", OUTPUTS / "conditioning_ablation" / "mimic_cxr" / "cv5" /
             "e2_rho_between.json")]
    cells = [("condnet_head", "C-Head", "#9db9d6"), ("condnet_deep", "C-Deep", "#4c86bd"),
             ("condnet_full", "C-Full", "#1f4e79")]
    stages = ["layer1", "layer2", "layer3", "layer4", "fc"]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5),
                             gridspec_kw={"width_ratios": [1, 1, 0.62]})
    # 左两格必须共享对数刻度：正文要拿 HAM 与 MIMIC 的量级作比较，
    # 各自独立缩放会让「MIMIC 的 fc 比 HAM 弱约二十倍」在图上完全看不出来。
    axes[1].sharey(axes[0])
    for ax, (title, f) in zip(axes[:2], srcs):
        payload = json.loads(f.read_text())
        agg: dict[tuple[str, str], list[float]] = {}
        for key, v in payload.items():
            cell, layer = key.split("|")
            st = _stage_of(layer)
            if st is not None:
                agg.setdefault((cell, st), []).append(v["rho_between"])
        x = np.arange(len(stages), dtype=float)
        for i, (cell, label, colour) in enumerate(cells):
            vals = [float(np.mean(agg[(cell, st)])) if (cell, st) in agg else np.nan
                    for st in stages]
            ax.bar(x + (i - 1) * 0.26, vals, width=0.24, color=colour,
                   edgecolor="white", linewidth=0.5, label=label)
        ax.set_yscale("log")
        ax.set_xticks(x, stages, fontsize=7.5)
        ax.set_title(title, fontsize=8.5, color=INK, pad=4)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        _style(ax)
    plt.setp(axes[1].get_yticklabels(), visible=False)
    axes[0].set_ylabel("$\\rho^{\\mathrm{between}}$", fontsize=8.5, color=MUTED)
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")

    # 右格：三种口径下的 conv rho_between（HAM10000）
    base = OUTPUTS / "conditioning_ablation" / "ham10000" / "cv5"
    r_over = json.loads((base / "e1_rho_between.json").read_text())
    r_wc = json.loads((base / "e1_rho_between_worstcase.json").read_text())
    e3 = json.loads((base / "e3_nofc_results.json").read_text())

    def conv_mean(payload: dict, cell: str) -> float:
        return float(np.mean([v["rho_between"] for k, v in payload.items()
                              if k.startswith(cell + "|") and not k.endswith("|fc")]))

    ax = axes[2]
    conds = ["best\noverall", "best\nWG", "head\noff"]
    x = np.arange(len(conds), dtype=float)
    for i, (cell, label, colour) in enumerate(cells[1:]):
        vals = [conv_mean(r_over, cell), conv_mean(r_wc, cell),
                e3["final_rho"][cell + "_nofc"]["conv_rho_between_mean"]]
        ax.bar(x + (i - 0.5) * 0.34, vals, width=0.32, color=colour,
               edgecolor="white", linewidth=0.5, label=label)
    ax.set_xticks(x, conds, fontsize=7)
    ax.set_title("HAM10000, convolution only", fontsize=8.5, color=INK, pad=4)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 0.026)
    ax.tick_params(labelsize=7.5)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    plot_id_worst(FIGDIR / "ch5_id_worst.pdf")
    plot_ood_ratio(FIGDIR / "ch5_ood_ratio.pdf")
    plot_rho(FIGDIR / "ch5_rho.pdf")
    plot_hyperplane(FIGDIR / "ch5_hyperplane.pdf")
    print("done ->", FIGDIR)




def plot_hyperplane(path: Path) -> None:
    """
    四库合成的逐子群判别面图（报告 §5.5）。

    坐标即 \\cref{sec:hyperplane-def} 的判别基：横轴 e1 = 平均超平面法向（诊断方向），
    纵轴 e2 = 扣掉 e1 后组间差异最大的方向。

    ⚠️ **每个子群既有自己的决策线，也有自己的特征云**——HyperAdapt 条件化了卷积层，
    故 f(x,a) 随假设的子群而变。早先只画一个云配全部线的画法默认了表征不变，
    在 HAM 上尤其错（两云质心相距 1.17 个参照云标准差）。且云与线一起移动时存在规范自由度，
    只有「云相对于它自己那条线的位置」有物理含义。故此处**成对**呈现两个子群，
    各自的云与线同色，读者可直接比较两组的相对位置是否一致。

    选哪一对：取**换序最多**的一对（行为上最不同），与标注里的换序比例是同一对。
    """
    from scipy.stats import kendalltau

    from src.utils.hyperplane_viz import line_from_coeffs

    specs = [
        ("HAM10000", "ham10000", "ham_sexage_fold0_seed42", "sex $\\times$ age"),
        ("Fitzpatrick17k", "fitzpatrick", "fitz_fold0_seed42", "skin type"),
        ("MIMIC-CXR", "mimic_cxr", "mimic_fold0_seed42", "sex $\\times$ race $\\times$ age"),
        ("CheXpert", "chexpert_cxr", "chexpert_fold0_seed42", "sex $\\times$ race $\\times$ age"),
    ]
    pair_colours = ("#1f4e79", "#c0703a")          # 两个子群，云与线同色
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.45))
    for ax, (title, sub, tag, axis_lab) in zip(axes, specs):
        base = OUTPUTS / sub / "hyperplane_viz"
        arr = np.load(base / f"hyperplane_arrays_{tag}.npz")
        met = json.loads((base / f"hyperplane_metrics_{tag}.json").read_text())
        U, w_eff, mu = arr["U_discriminative"], arr["w_eff"], arr["mu"]
        bias, labels, P, logits = float(arr["bias"]), arr["labels"], arr["P_discriminative"], arr["logits"]
        n_groups = w_eff.shape[0]

        # 换序最多的一对 = (1 - Kendall tau)/2 最大者
        i, j, reorder = max(
            ((i, j, (1 - kendalltau(logits[i], logits[j]).statistic) / 2)
             for i in range(n_groups) for j in range(i + 1, n_groups)),
            key=lambda t: t[2])
        cos_ij = float(np.asarray(met["pairwise_cosine_w_eff"])[i, j])
        angle = float(np.degrees(np.arccos(np.clip(cos_ij, -1, 1))))
        shift = float(np.linalg.norm((U[j].mean(0) - U[i].mean(0)) / U[i].std(axis=0)))

        # 点按**类别**着色（保留「两类沿 e1 分开」这层信息），子群则用椭圆与线表示：
        # 两个云直接叠散点会糊成一团，1 个 SD 的协方差椭圆能紧凑地给出各自的位置与形状。
        rng = np.random.default_rng(0)
        idx = rng.choice(U.shape[1], size=min(300, U.shape[1]), replace=False)
        for g in (i, j):
            for cls, colour, mk, sz in ((0, "#9fb6cc", "o", 2.4), (1, "#d99a95", "^", 3.4)):
                sel = idx[labels[idx] == cls]
                ax.scatter(U[g][sel, 0], U[g][sel, 1], s=sz, c=colour, marker=mk,
                           linewidths=0, alpha=0.5, zorder=1)
        for g, colour in zip((i, j), pair_colours):
            pts = U[g]
            cen = pts.mean(axis=0)
            vals, vecs = np.linalg.eigh(np.cov(pts.T))
            ang = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
            ax.add_patch(Ellipse(cen, 2 * np.sqrt(vals[-1]), 2 * np.sqrt(vals[0]),
                                 angle=ang, facecolor="none", edgecolor=colour,
                                 linewidth=1.2, zorder=4))
            ax.scatter([cen[0]], [cen[1]], s=11, color=colour, zorder=5,
                       edgecolors="white", linewidths=0.6)
        both = np.concatenate([U[i][idx], U[j][idx]])
        xl = (float(both[:, 0].min()), float(both[:, 0].max()))
        yl = (float(both[:, 1].min()), float(both[:, 1].max()))
        pad = 0.07
        xl = (xl[0] - pad * (xl[1] - xl[0]), xl[1] + pad * (xl[1] - xl[0]))
        # 顶部多留白，让标注框落在点云之上而不是压在椭圆上
        yl = (yl[0] - pad * (yl[1] - yl[0]), yl[1] + 0.62 * (yl[1] - yl[0]))
        for g, colour in zip((i, j), pair_colours):
            seg = line_from_coeffs(P.T @ w_eff[g], float(w_eff[g] @ mu + bias), xl, yl)
            if seg is not None:
                ax.plot(seg[0], seg[1], color=colour, linewidth=1.5, zorder=3)
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)

        per = met.get("per_group") or met.get("per_age") or met.get("per_skin")
        aucs = [v["counterfactual_auc"] for v in per.values()]
        ax.set_title(title, fontsize=8.5, color=INK, pad=4)
        ax.text(0.035, 0.955,
                f"{n_groups} groups, {axis_lab}\n"
                f"pair shown {angle:.1f}$^\\circ$ apart\n"
                f"clouds {shift:.2f} SD apart\n"
                f"{reorder * 100:.0f}% of pairs reordered\n"
                f"AUC span {max(aucs) - min(aucs):.3f}",
                transform=ax.transAxes, fontsize=6.2, color=INK, va="top", ha="left",
                linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.26", facecolor="white", alpha=0.86,
                          edgecolor=GRID, linewidth=0.5))
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_color(GRID)
    axes[0].set_xlabel("$e_1$", fontsize=8.5, color=MUTED)
    axes[0].set_ylabel("$e_2$", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)




if __name__ == "__main__":
    main()
