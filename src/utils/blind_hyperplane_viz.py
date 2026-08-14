"""
attribute-blind 模型（ERM / SWAD）的单-panel 分类超平面可视化
==============================================================
与 `src/utils/hyperplane_viz.py`（HyperAdapt 逐子群版）互为对照，**方法学上必须区分开**：

| | HyperAdapt 版 | 本模块（ERM / SWAD） |
|---|---|---|
| 属性是否进模型 | 是（conv 乘性调制 + fc 低秩） | **否**（image-only，forward 不吃 sex/race/age） |
| 超平面条数 | 每子群一条 `w_eff(a)` | **全局唯一一条 `w`** |
| 组间差异来源 | HN 的条件化（反事实扫掠固定病人 ⇒ 100% 归因于模型） | **数据本身**（不同人群的影像分布不同） |
| 纵轴 e2 | 组间 `w_g - w_bar` 的主方向 | **组间特征均值 `mu_g - mu` 的主方向** |

⚠️ **反事实扫掠在这里不存在也不需要**：属性不进模型 ⇒ 同一张图只有一个特征、一个 logit。
因此本模块回答的问题不是「HN 动了什么」，而是「**在一条对所有人一视同仁的超平面下，各子群
天然落在哪、相对这条线偏多少**」——正是 HN 那套图的**零对照**：HN 图里的组间位移来自条件化，
这里的组间位移全部来自数据侧。两者不可混为一谈，也**不能**据此说 ERM/SWAD「调整了超平面」
（它根本没有可调的自由度）。

规范自由度（gauge freedom）在这里**不适用**：只有一条线、一套特征，点相对线的位置就是全部信息。
但「投影损失」仍适用 ⇒ in-plane 能量必报。

配色（依 dataviz skill，已用 `validate_palette.js` 验证）
--------------------------------------------------------
散点密集重叠属 `--pairs all` 口径，8 个分类色在该口径下无法通过 CVD/normal-vision 地板，
故**不用分类色板**，改用**单 hue ordinal ramp**（HAM 4 级 / Fitzpatrick 6 级 / CXR 4 级，
均 `--ordinal` 全项 PASS）：

  · HAM `age_group`、Fitzpatrick `skin` 本就是**有序**属性 ⇒ sequential ramp 是正解而非将就；
  · CXR 的 `race×age` 四组合**并非有序量**，ramp 仅用于稳定区分，图注已声明，且每子群另有
    **直接文字标签** ⇒ identity 不由颜色单独承载（满足可达性要求）。

标签 y **不逐点着色**（颜色通道已给子群），改由每子群的 **y=0 / y=1 质心对**表达：
空心标记 = y0 质心，实心 = y1 质心，两者沿判别轴的间距即该子群的类间分离度。
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# CLASS_COLORS 与 HyperAdapt 版共用同一套类别配色（y=0 蓝 / y=1 红），两套图才能直接对照
from src.utils.hyperplane_viz import CLASS_COLORS, line_from_coeffs

# ---- 单 hue ordinal ramp（validate_palette.js --ordinal 全 PASS，light surface #fcfcfb）----
# 4 级：HAM 的 4 个 age 组 / CXR 的 4 个 race×age 组合
RAMP_4 = ("#86b6ef", "#5598e7", "#2a78d6", "#184f95")
# 6 级：Fitzpatrick 的 6 个肤色型（最深两级为本项目扩展步，已随整条 ramp 一并验证）
RAMP_6 = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0f3f7c", "#061f3d")

CENTROID_EDGE = "#0b0b0b"                                   # 质心标记描边（text-primary）
LINE_COLOR = "#0b0b0b"                                      # 决策线
GRID_COLOR = "0.85"


def ordinal_ramp(n_levels: int) -> tuple[str, ...]:
    """
    取长度为 `n_levels` 的单 hue ordinal ramp。

    Args:
        n_levels: 需要的色阶数（4 或 6；其它值从 6 级 ramp 上等距取样）。

    Returns:
        长度 n_levels 的 hex 颜色元组，由浅到深。

    Raises:
        ValueError: n_levels < 2 或 > 6（超过 6 级无法同时满足 ΔL>=0.06 与浅端 2:1 对比度）。
    """
    if not 2 <= n_levels <= 6:
        raise ValueError(f"ordinal ramp 只支持 2..6 级（收到 {n_levels}）；"
                         f"更多子群请改用分面而非加色阶。")
    if n_levels == 4:
        return RAMP_4
    if n_levels == 6:
        return RAMP_6
    idx = np.linspace(0, len(RAMP_6) - 1, n_levels).round().astype(int)
    return tuple(RAMP_6[i] for i in idx)


def build_blind_basis(
    w: np.ndarray, group_means: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    attribute-blind 模型的 2D 投影基：判别轴 × **组间特征均值**分离轴。

      e1 = w/‖w‖                                              （唯一超平面的法向量 = 诊断方向）
      e2 = 在 e1 正交补内对 {mu_g − mu} 做 PCA 的第一主成分       （组间分离方向）

    与 `hyperplane_viz.build_discriminative_basis` 的唯一区别是 e2 的原料：那里是各组
    **打分表** `w_g` 的差异（模型条件化的产物），这里模型只有一个 `w`、无差异可言，故改用各组
    **特征均值**的差异（数据侧的产物）。同样先投影到正交补再取主成分（deflation），精确求解
        max_{e ⊥ e1, ‖e‖=1}  Σ_g <mu_g − mu, e>²
    而非「全空间 PCA1 事后 Gram-Schmidt」（后者在组间差异主要沿 e1 时方向由噪声决定）。

    Args:
        w: [512] 唯一的超平面法向量（fc.weight 单行）。
        group_means: [G, 512] 各子群的特征均值（同一批样本按真实属性分组求均值）。

    Returns:
        (P [512,2] 列正交单位基, 诊断量字典)：
          · between_var_along_e1  : 组间均值差异沿判别轴的份额（= 子群平均得分差异 / 先验偏移）；
          · between_var_in_perp   : 其补，落在正交补、可被 e2 展示的份额；
          · e2_perp_var_ratio     : e2 占**正交补内**组间方差的比例；
          · e2_singular_value_ratio: 补空间内 s1/s2，接近 1 时 e2 方向近乎简并、纵轴不可作方向性解读。

    Raises:
        RuntimeError: 组间均值差异几乎全部沿 e1（正交补内无方差），e2 退化。
    """
    e1 = w / (np.linalg.norm(w) + 1e-12)

    mu_bar = group_means.mean(axis=0)
    dev = group_means - mu_bar                              # 各组均值相对总均值的偏离 [G, 512]
    along_e1 = dev @ e1
    dev_perp = dev - np.outer(along_e1, e1)

    total_var = float((dev ** 2).sum())
    var_along = float((along_e1 ** 2).sum())
    var_perp = total_var - var_along
    if var_perp <= 1e-12 * max(total_var, 1e-12):
        raise RuntimeError("组间特征均值差异几乎全部沿判别轴，正交补内无方差；e2 退化。")

    _u, s_vals, vh = np.linalg.svd(dev_perp, full_matrices=False)
    e2 = vh[0]
    e2 = e2 - (e2 @ e1) * e1                                # 消除浮点残留，确保严格正交
    e2 = e2 / (np.linalg.norm(e2) + 1e-12)

    s_sq = s_vals ** 2
    return np.stack([e1, e2], axis=1), {
        "e2_perp_var_ratio": float(s_sq[0] / (s_sq.sum() + 1e-12)),
        "between_var_along_e1": var_along / (total_var + 1e-12),
        "between_var_in_perp": var_perp / (total_var + 1e-12),
        "e2_singular_value_ratio": float(s_vals[0] / (s_vals[1] + 1e-12))
        if len(s_vals) > 1 else float("inf"),
    }


def permutation_test_between_group(
    feats: np.ndarray, group_ids: np.ndarray, e1: np.ndarray, n_groups: int,
    n_perm: int = 2000, seed: int = 0,
) -> dict[str, float]:
    """
    组间特征均值差异的**置换检验**：分解为「沿判别轴」与「正交补内」两部分分别检验。

    统计量为经典的 between-group sum of squares（按组规模加权）：
        S_total = Σ_g n_g ‖mu_g − mu‖²
        S_along = Σ_g n_g <mu_g − mu, e1>²        （沿判别轴 ⇒ 纯得分平移，会动操作点/校准）
        S_perp  = S_total − S_along               （正交于判别轴 ⇒ 完全不影响该模型的预测）
    零假设：子群标签与特征无关。置换 `group_ids` 后重算，p = (1 + #{perm ≥ obs}) / (n_perm + 1)。

    ⚠️ **显著 ≠ 需要条件化**。本检验测的是 `I(A;X) > 0`（属性能否从图像/特征读出），而
    HN 有收益的必要条件是 `I(Y;A|X) > 0`（给定图像后属性对标签还有用），两者是不同的量：
    Fitzpatrick 的肤色 I(A;X)≥0.37 bits 却 I(Y;skin|X)≈0（见 docs/conditional_v_information_gate.md）。
    `S_perp` 占比高正是「组间分得开、但分开的方向与判别无关」的直接量化。
    ⚠️ n 大时极小差异也会显著，故必须同时看效应量（见 subgroup_offsets 的标准化位移）。

    Args:
        feats: [N, 512] 特征；group_ids: [N] 子群下标；e1: [512] 判别方向单位向量。
        n_groups: 子群数；n_perm: 置换次数；seed: 置换随机种子。

    Returns:
        观测统计量、三个 p 值，以及 S_perp 占 S_total 的比例。
    """
    rng = np.random.default_rng(seed)
    mu = feats.mean(axis=0)
    dev_all = feats - mu

    def stats(gids: np.ndarray) -> tuple[float, float, float]:
        s_total = s_along = 0.0
        for g in range(n_groups):
            sel = gids == g
            n_g = int(sel.sum())
            if n_g == 0:
                continue
            d = dev_all[sel].mean(axis=0)
            s_total += n_g * float(d @ d)
            s_along += n_g * float((d @ e1) ** 2)
        return s_total, s_along, s_total - s_along

    obs_total, obs_along, obs_perp = stats(group_ids)
    cnt_total = cnt_along = cnt_perp = 0
    permuted = group_ids.copy()
    for _ in range(n_perm):
        rng.shuffle(permuted)
        p_total, p_along, p_perp = stats(permuted)
        cnt_total += p_total >= obs_total
        cnt_along += p_along >= obs_along
        cnt_perp += p_perp >= obs_perp

    return {
        "s_total": obs_total, "s_along_e1": obs_along, "s_perp": obs_perp,
        "perp_share": obs_perp / (obs_total + 1e-12),
        "p_total": (1 + cnt_total) / (n_perm + 1),
        "p_along_e1": (1 + cnt_along) / (n_perm + 1),
        "p_perp": (1 + cnt_perp) / (n_perm + 1),
        "n_perm": float(n_perm),
    }


def subgroup_offsets(U: np.ndarray, group_ids: np.ndarray, n_groups: int) -> dict[int, dict[str, float]]:
    """
    逐子群在 2D 投影平面内的**标准化质心位移**（效应量，与图上看到的位移一一对应）。

    以全体样本在各轴上的 SD 为尺度，故读数是「该组质心偏离总体多少个标准差」：
      · d_along : 沿 e1（判别轴）的位移 ⇒ 纯得分/操作点偏移，**不改组内排序**；
      · d_perp  : 沿 e2（正交补）的位移 ⇒ 与该模型的判别完全无关的方向。

    Args:
        U: [N, 2] 投影坐标；group_ids: [N] 子群下标；n_groups: 子群数。

    Returns:
        {子群下标: {"d_along": ..., "d_perp": ...}}。
    """
    sd = U.std(axis=0) + 1e-12
    centre = U.mean(axis=0)
    out: dict[int, dict[str, float]] = {}
    for g in range(n_groups):
        sel = group_ids == g
        if not sel.any():
            out[g] = {"d_along": float("nan"), "d_perp": float("nan")}
            continue
        d = U[sel].mean(axis=0) - centre
        out[g] = {"d_along": float(d[0] / sd[0]), "d_perp": float(d[1] / sd[1])}
    return out


def render_blind_subgroup_panels(
    U: np.ndarray, P: np.ndarray, w: np.ndarray, bias: float, mu: np.ndarray,
    labels: np.ndarray, group_ids: np.ndarray, logits: np.ndarray,
    group_labels: dict[int, str], per_group_stats: dict[int, dict[str, float]],
    offsets: dict[int, dict[str, float]], perm: dict[str, float], diag: dict[str, float],
    plot_idx: np.ndarray, class_names: tuple[str, str], ncols: int, header: str,
    out_path: Path,
) -> None:
    """
    渲染 attribute-blind 模型的**逐子群 panel** 图（与 HyperAdapt 版布局对齐，便于并排对照）。

    与 HyperAdapt 版的关键区别（读图前必看）：
      · 每 panel 只画该子群的**真实成员**（blind 模型没有反事实，不存在「假设他属于 X 组」）；
      · G 个 panel 的决策线**完全相同**——这不是 bug，模型只有一条超平面；
      · 因此本图**不是**在看「超平面怎么变」，而是在看「**同一条线下各子群的点云落在哪、
        相对总体偏多少**」，即组间聚类差异；
      · 灰色空心/实心大标记 = **全体样本**的 y0/y1 质心（每 panel 相同的固定参照），
        彩色标记 = 该子群自己的 y0/y1 质心 ⇒ 两者之差即该组的组间位移。

    ⚠️ 判读红线：组间点云分得开只证明 `I(A;X) > 0`（属性可从图像读出），**不能**推出
    「需要属性条件化」——后者要 `I(Y;A|X) > 0`。panel 标题里的 `d∥`（沿判别轴）与
    `d⊥`（正交补）就是为分辨这两者：位移几乎全在 `d⊥` 时，组间差异与该模型的判别无关。

    Args:
        U          : [N, 2] 投影坐标；P: [512,2] 基；w: [512] 唯一法向量；bias: fc.bias；mu: 中心。
        labels     : [N] 标签；group_ids: [N] 真实子群；logits: [N] logit。
        group_labels: 子群下标 -> 显示名；per_group_stats: 逐组 n/auc/mean_logit/pos_rate。
        offsets    : subgroup_offsets 的标准化位移；perm: permutation_test_between_group 的结果。
        diag       : build_blind_basis 的诊断量。
        plot_idx   : 参与绘制的样本下标（仅控制散点密度）。
        class_names: (y=0 名, y=1 名)；ncols: panel 列数；header: suptitle 首行。
        out_path   : 输出 PNG 路径。
    """
    n_groups = len(group_labels)
    nrows = math.ceil(n_groups / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 5.3 * nrows), squeeze=False)

    # 全 panel 共用视窗：panel 间坐标可比是「组间位移可读」的前提
    pad = 0.06 * (U.max(axis=0) - U.min(axis=0) + 1e-9)
    x_lim = (float(U[:, 0].min() - pad[0]), float(U[:, 0].max() + pad[0]))
    y_lim = (float(U[:, 1].min() - pad[1]), float(U[:, 1].max() + pad[1]))
    seg = line_from_coeffs(P.T @ w, float(w @ mu + bias), x_lim, y_lim)

    # 全体 y0/y1 质心：每 panel 相同的固定参照
    pooled = {y: U[labels == y].mean(axis=0) for y in (0, 1) if (labels == y).any()}

    for g in range(nrows * ncols):
        ax = axes[g // ncols][g % ncols]
        if g >= n_groups:
            ax.axis("off")
            continue

        in_g = group_ids == g
        sel = plot_idx[in_g[plot_idx]]
        # 点按标签着色（与 HyperAdapt 版同一套 CLASS_COLORS，两图可直接对照）
        ax.scatter(U[sel, 0], U[sel, 1], c=CLASS_COLORS[labels[sel]], s=17, alpha=0.55,
                   edgecolors="none", zorder=3)

        # 固定参照：全体质心（灰）
        for y_val, c_pool in pooled.items():
            ax.scatter([c_pool[0]], [c_pool[1]], marker="o", s=150,
                       facecolors="0.55" if y_val == 1 else "none", edgecolors="0.35",
                       linewidths=1.8, zorder=5)
        # 该子群自己的质心（类别色）
        for y_val in (0, 1):
            sub = in_g & (labels == y_val)
            if not sub.any():
                continue
            c = U[sub].mean(axis=0)
            ax.scatter([c[0]], [c[1]], marker="o", s=185,
                       facecolors=CLASS_COLORS[y_val] if y_val == 1 else "none",
                       edgecolors=CENTROID_EDGE if y_val == 1 else CLASS_COLORS[y_val],
                       linewidths=2.0, zorder=6)

        if seg is not None:
            ax.plot(seg[0], seg[1], color=LINE_COLOR, lw=2.4, zorder=7)

        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.grid(True, color=GRID_COLOR, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xticks([])
        ax.set_yticks([])
        st, off = per_group_stats[g], offsets[g]
        ax.set_title(f"{group_labels[g]}   n={st['n']:.0f}\n"
                     f"AUC={st['auc']:.4f}   mean logit={st['mean_logit']:+.2f}   "
                     f"{class_names[1]} rate={st['pos_rate']:.3f}\n"
                     f"d$\\parallel$={off['d_along']:+.2f} SD    "
                     f"d$\\perp$={off['d_perp']:+.2f} SD", fontsize=9.5)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLORS[0], markersize=9,
               label=f"{class_names[0]} (y=0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLORS[1], markersize=9,
               label=f"{class_names[1]} (y=1)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor=CLASS_COLORS[0], markeredgewidth=2.0, markersize=12,
               label="this subgroup's y=0 centroid"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLORS[1],
               markeredgecolor=CENTROID_EDGE, markersize=12,
               label="this subgroup's y=1 centroid"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="0.35",
               markeredgewidth=1.8, markersize=12, label="POOLED y=0 centroid (same in every panel)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.55", markeredgecolor="0.35",
               markersize=12, label="POOLED y=1 centroid (same in every panel)"),
        Line2D([0], [0], color=LINE_COLOR, lw=2.4,
               label="the single decision line (IDENTICAL in every panel)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.012))

    fig.suptitle(
        f"{header}\n"
        "Attribute-blind model: the decision line is THE SAME in every panel — what varies is "
        "where each subgroup's point cloud sits relative to it\n"
        f"Between-group feature-mean differences (permutation test, B={perm['n_perm']:.0f}): "
        f"total p={perm['p_total']:.4f}, along-e1 p={perm['p_along_e1']:.4f}, "
        f"perp p={perm['p_perp']:.4f}; {perm['perp_share']:.1%} of the between-group variance is "
        f"ORTHOGONAL to w\n"
        "Centroid offset vs the POOLED centroid:  d|| = shift along e1 (pure score / "
        "operating-point shift, does NOT change within-group ranking);\n"
        "d-perp = shift inside e1's orthogonal complement (invisible to w). "
        "Large d-perp with small d|| = separable but irrelevant to this classifier.\n"
        "!! Separable subgroups show only I(A;X)>0. Conditioning pays off only if I(Y;A|X)>0 "
        "— a different quantity (docs/conditional_v_information_gate.md).",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.99 - 0.05 * nrows / max(1, nrows)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def render_blind_figure(
    U: np.ndarray, P: np.ndarray, w: np.ndarray, bias: float, mu: np.ndarray,
    labels: np.ndarray, group_ids: np.ndarray, logits: np.ndarray,
    group_labels: dict[int, str], group_short: dict[int, str], group_colors: list[str],
    group_markers: list[str], marker_legend: list[tuple[str, str]] | None,
    per_group_stats: dict[int, dict[str, float]], energy: float, diag: dict[str, float],
    plot_idx: np.ndarray, class_names: tuple[str, str], header: str,
    colour_note: str, out_path: Path,
) -> None:
    """
    渲染 attribute-blind 模型的**单 panel** 图：全部样本一张散点 + 唯一一条决策线 +
    逐子群 y0/y1 质心对 + 右侧逐子群统计表。

    读法：横轴 = 诊断方向（点的横坐标 ≈ 其 logit）；纵轴 = 组间特征均值分离方向。
    子群之间的横向错位 = 该子群平均得分的系统性偏移（数据侧先验差异，**非**模型条件化）；
    每个子群 y0→y1 两质心的横向间距 = 该子群的类间分离度，间距越大该组越可分。

    Args:
        U          : [N, 2] 已投影坐标（(f - mu) @ P）。
        P          : [512, 2] 投影基；w: [512] 唯一超平面法向量；bias: fc.bias；mu: [512] 中心。
        labels     : [N] 真实标签；group_ids: [N] 每个样本的真实子群下标；logits: [N] 模型 logit。
        group_labels: 子群下标 -> 图例全名；group_short: 子群下标 -> 质心旁的短标签。
        group_colors : 每子群的 hex 颜色（单 hue ordinal ramp）。
        group_markers: 每子群散点/质心的 marker（CXR 用形状额外编码 sex，其余库统一 'o'）。
        marker_legend: [(marker, 说明)] 形状通道的图例项；无额外形状编码时传 None。
        per_group_stats: 子群下标 -> {n, auc, mean_logit, pos_rate}，渲染右侧统计表。
        energy     : in-plane 能量 ‖Pᵀw‖/‖w‖（唯一 w ⇒ 单值）。
        diag       : build_blind_basis 的诊断量。
        plot_idx   : 参与绘制的样本下标（仅控制散点密度，统计仍用全部样本）。
        class_names: (y=0 名, y=1 名)。
        header     : suptitle 首行（数据集/方法/checkpoint）。
        colour_note: 配色说明（有序属性 vs 仅作区分），写进 suptitle。
        out_path   : 输出 PNG 路径。
    """
    n_groups = len(group_labels)
    fig, (ax, ax_tab) = plt.subplots(
        1, 2, figsize=(15.5, 8.2), gridspec_kw={"width_ratios": [2.7, 1.0]})

    pad = 0.06 * (U.max(axis=0) - U.min(axis=0) + 1e-9)
    x_lim = (float(U[:, 0].min() - pad[0]), float(U[:, 0].max() + pad[0]))
    y_lim = (float(U[:, 1].min() - pad[1]), float(U[:, 1].max() + pad[1]))

    # ---- 散点：颜色 = 子群，形状 = marker 通道；label 不在此编码（见质心对）----
    for g in range(n_groups):
        sel = plot_idx[group_ids[plot_idx] == g]
        if sel.size:
            ax.scatter(U[sel, 0], U[sel, 1], c=group_colors[g], marker=group_markers[g],
                       s=19, alpha=0.45, edgecolors="none", zorder=2)

    # ---- 逐子群 y0 / y1 质心对：空心 = y0，实心 = y1；连线示意类间分离 ----
    for g in range(n_groups):
        in_g = group_ids == g
        for y_val in (0, 1):
            sel = in_g & (labels == y_val)
            if not sel.any():
                continue
            cx, cy = U[sel, 0].mean(), U[sel, 1].mean()
            ax.scatter([cx], [cy], marker=group_markers[g], s=190,
                       facecolors=group_colors[g] if y_val == 1 else "none",
                       edgecolors=CENTROID_EDGE if y_val == 1 else group_colors[g],
                       linewidths=1.8 if y_val == 1 else 2.2, zorder=6)
        # 同组两质心连线：长度 = 类间分离度，方向多半沿判别轴
        if (in_g & (labels == 0)).any() and (in_g & (labels == 1)).any():
            c0 = U[in_g & (labels == 0)].mean(axis=0)
            c1 = U[in_g & (labels == 1)].mean(axis=0)
            ax.plot([c0[0], c1[0]], [c0[1], c1[1]], color=group_colors[g], lw=1.6,
                    alpha=0.9, zorder=5)
            # 标签上下交错：质心常沿判别轴挤成一串，同侧标注会互相压字
            dy = 10 if g % 2 == 0 else -14
            ax.annotate(group_short[g], xy=(c1[0], c1[1]), xytext=(9, dy),
                        textcoords="offset points", fontsize=8.5, color="#3f3e3b",
                        zorder=7,
                        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none",
                              "boxstyle": "round,pad=0.15"})

    # ---- 唯一决策线：<Pᵀw, u> + (<w, mu> + b) = 0 ----
    seg = line_from_coeffs(P.T @ w, float(w @ mu + bias), x_lim, y_lim)
    if seg is not None:
        ax.plot(seg[0], seg[1], color=LINE_COLOR, lw=2.4, zorder=8)

    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_xlabel("e1 = w/||w||  (diagnosis direction; x-position ≈ logit, so the decision line "
                  "is exactly vertical by construction)", fontsize=10)
    ax.set_ylabel("e2 = between-group feature-mean direction\n(inside e1's orthogonal complement)",
                  fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, color=GRID_COLOR, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.text(0.015, 0.985,
            f"in-plane energy = {energy:.3f} (=1 by construction: e1 || w,\n"
            f"  so NOT a diagnostic here — unlike the HyperAdapt figures)\n"
            f"between-group feature-mean var:\n"
            f"  {diag['between_var_along_e1']:.1%} along e1 (pure score offset)\n"
            f"  {diag['between_var_in_perp']:.1%} in e1-perp\n"
            f"e2 captures {diag['e2_perp_var_ratio']:.1%} of perp\n"
            f"s1/s2 = {diag['e2_singular_value_ratio']:.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.7",
                  "boxstyle": "round,pad=0.35"}, zorder=9)

    # ---- 图例：子群色 + 质心形状 + (可选) marker 通道 + 决策线 ----
    # 图例的 marker 必须与该子群散点实际使用的形状一致——CXR 把 sex 编码在形状上，
    # 若图例统一用方块，Male/Female 两组会显示成同色同形，等于抹掉一个编码通道。
    handles = [Line2D([0], [0], marker=group_markers[g], color="w",
                      markerfacecolor=group_colors[g], markersize=9, label=group_labels[g])
               for g in range(n_groups)]
    handles += [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="#2a78d6", markeredgewidth=2.0, markersize=11,
               label=f"subgroup centroid, {class_names[0]} (y=0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2a78d6",
               markeredgecolor=CENTROID_EDGE, markersize=11,
               label=f"subgroup centroid, {class_names[1]} (y=1)"),
    ]
    if marker_legend:
        handles += [Line2D([0], [0], marker=m, color="w", markerfacecolor="#52514e",
                           markersize=9, label=lab) for m, lab in marker_legend]
    handles.append(Line2D([0], [0], color=LINE_COLOR, lw=2.4,
                          label="the single decision line (shared by every subgroup)"))
    # 图例放在右栏（表格下方的空白），不压住点云——质心常落在图区边角，图内图例会遮住它们
    ax_tab.legend(handles=handles, fontsize=8.5, loc="lower left", frameon=True,
                  framealpha=0.95, labelspacing=0.4, borderaxespad=0.2)

    # ---- 右侧逐子群统计表（数值以表格给出，图内不叠数字）----
    ax_tab.axis("off")
    rows = [[group_labels[g], f"{per_group_stats[g]['n']:.0f}",
             f"{per_group_stats[g]['auc']:.4f}", f"{per_group_stats[g]['mean_logit']:+.2f}",
             f"{per_group_stats[g]['pos_rate']:.3f}"] for g in range(n_groups)]
    table = ax_tab.table(cellText=rows,
                         colLabels=["subgroup", "n", "AUC", "mean\nlogit",
                                    f"{class_names[1]}\nrate"],
                         colWidths=[0.46, 0.09, 0.14, 0.14, 0.19],
                         cellLoc="center", colLoc="center", loc="upper center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.7)
    for g in range(n_groups):                               # 子群名左对齐，避免长名被挤成截断样
        table[(g + 1, 0)].set_text_props(ha="left")
        table[(g + 1, 0)].PAD = 0.04
    for g in range(n_groups):                               # 首列用该子群色的小色块提示
        table[(g + 1, 0)].set_facecolor(group_colors[g])
        table[(g + 1, 0)].set_alpha(0.22)
    for c in range(5):
        table[(0, c)].set_facecolor("#f0efec")
        table[(0, c)].set_text_props(fontweight="bold")
    ax_tab.set_title("Per-subgroup readout (all samples, not the plotted subset)",
                     fontsize=10, pad=14)

    fig.suptitle(
        f"{header}\n"
        "Attribute-blind model: ONE hyperplane for everyone — subgroup offsets below come from "
        "the DATA, not from any conditioning\n"
        f"{colour_note}  |  label y is not colour-coded: it is carried by the open (y=0) / "
        "filled (y=1) centroid pair of each subgroup",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")
