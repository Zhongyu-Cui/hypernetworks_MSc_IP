"""
HyperAdapt 分类超平面逐子群可视化的共享工具
============================================
被 scripts/viz_hyperadapt_hyperplane_ham.py（HAM，4 个 age 组）与
scripts/viz_hyperadapt_hyperplane_mimic.py（MIMIC，8 个 sex×race×age 子群）共用。

三块内容：
  1. 两套 2D 投影基的构造 + 投影诚实性指标（in-plane 能量）；
  2. Δlogit 的 fc / conv 通路分解（代数恒等式）；
  3. 通用渲染：单套基一张图（G 个子群 panel），以及通路分解图。

方法学要点（两个脚本共同遵守，详见各脚本 docstring）：
  · HyperAdapt 的条件化有两条通路——conv 层的 channel-wise 乘性调制让特征 f(x,a) 随属性变，
    fc 的加性低秩更新让 w_eff(a) = W + ΔW(a) 随属性变；fc.bias 不随属性变。
    只比较 w_eff(a) 的几何会完全漏掉 conv 通路，故必须用反事实扫掠把两者一起纳入。
  · 特征云与超平面同时移动 ⇒ 存在规范自由度（gauge freedom），单看决策线的位移无意义，
    因此渲染强制把点云与该组决策线画在同一 panel 内。
  · 投影必须是线性的：f = mu + P u ⇒ logit(u) = <P^T w, u> + (<w, mu> + b) 在 2D 上仍是
    直线，可解析绘制（这是选 PCA 类线性基、而非 t-SNE/UMAP 的决定性理由）。
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

# 图内文字统一英文（matplotlib 默认字体无中文字形，且与仓库既有图风格一致）
BASIS_KEYS = ("discriminative", "pca")
BASIS_TITLES = {
    "discriminative": "Discriminative axis x between-group adjustment axis",
    "pca": "Default PCA (first two principal components)",
}
BASIS_EXPLAIN = {
    "discriminative": ("x = mean hyperplane normal (diagnosis direction);  "
                       "y = principal direction of between-group w differences computed "
                       "INSIDE the orthogonal complement of x (deflation, not post-hoc Gram-Schmidt)"),
    "pca": ("x, y = the two directions of largest feature variance. These are variance "
            "directions, not discriminative ones, so much of w can fall outside the plane "
            "(see in-plane energy) and between-group adjustment gets flattened."),
}
# 类别配色（与仓库 deepview 脚本一致）：y=0 蓝，y=1 红
CLASS_COLORS = np.array([[0.13, 0.40, 0.67], [0.70, 0.09, 0.17]])


# ============================================================
# 1. 两套投影基
# ============================================================
def build_discriminative_basis(w_eff: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """
    判别轴 x 组间调整轴。
      e1 = w_bar/||w_bar||                                    （判别轴 / 诊断方向）
      e2 = 在 **e1 的正交补内** 对 {w_g - w_bar} 做 PCA 的第一主成分   （组间调整轴）

    含义明确：横轴 = 诊断方向（沿之 logit 变化最快），纵轴 = 扣掉判别方向后组间差异最大的方向，
    因此「组间调整」这一被研究量在图内的可见比例最大化。

    ⚠️ **先降维到正交补、再做 PCA（deflation），而不是「全空间 PCA1 再 Gram-Schmidt」**。
    两者不等价：本函数精确求解
            max_{e ⊥ e1, ||e||=1}  sum_g <w_g - w_bar, e>^2
    而「先取全空间 pc1 再投影」只是把一个未受约束的解事后压进正交补——当组间偏离的主方差
    大部分沿 e1 时，pc1≈e1，其垂直残量方向由噪声决定，可任意偏离补空间内的真正主方向。
    实测（HAM/MIMIC/CheXpert fold0）两法夹角仅 1.1–1.6°、捕获方差差 <0.05pp，故历史结果不受影响；
    但本式无额外成本且有最优性保证，故为准。

    Args:
        w_eff: [G, 512] 各组有效超平面法向量。

    Returns:
        (P [512,2] 列正交单位基, 诊断量字典)：
          · e2_perp_var_ratio      : e2 占**正交补内**组间方差的比例（纵轴的诚实读数）。
          · between_var_along_e1   : 组间方差中沿 e1 的份额 = 纯尺度变化（图上表现为决策线平移
                                     而非转向）；1 - 该值即落在正交补、可被 e2 展示的份额。
          · between_var_in_perp    : 上者的补，便于直接引用。
          · e2_singular_value_ratio: 补空间内 s1/s2，越大说明 e2 越是唯一主方向（接近 1 时
                                     e2 的选择近乎简并、方向不稳，读图须谨慎）。

    Raises:
        RuntimeError: 组间偏离几乎全部沿 e1（模型只做纯缩放、无转向），补空间内无方差、e2 退化。
    """
    w_bar = w_eff.mean(axis=0)
    e1 = w_bar / (np.linalg.norm(w_bar) + 1e-12)

    dev = w_eff - w_bar                                     # 各组相对平均打分表的偏离 [G, 512]
    along_e1 = dev @ e1                                     # 每组偏离沿判别轴的分量 [G]
    dev_perp = dev - np.outer(along_e1, e1)                 # 投影到 e1 的正交补

    total_var = float((dev ** 2).sum())
    var_along = float((along_e1 ** 2).sum())
    var_perp = total_var - var_along
    if var_perp <= 1e-12 * max(total_var, 1e-12):
        raise RuntimeError("组间 w 偏离几乎全部沿判别轴，正交补内无方差；该模型只做了纯缩放调整。")

    # 补空间内的 PCA1（G 个向量，直接 SVD 即可）
    _u, s_vals, vh = np.linalg.svd(dev_perp, full_matrices=False)
    e2 = vh[0]
    # 数值保险：SVD 结果理论上已在补空间内，此处再正交化一次消除浮点残留
    e2 = e2 - (e2 @ e1) * e1
    e2 = e2 / (np.linalg.norm(e2) + 1e-12)

    s_sq = s_vals ** 2
    return np.stack([e1, e2], axis=1), {
        "e2_perp_var_ratio": float(s_sq[0] / (s_sq.sum() + 1e-12)),
        "between_var_along_e1": var_along / (total_var + 1e-12),
        "between_var_in_perp": var_perp / (total_var + 1e-12),
        "e2_singular_value_ratio": float(s_vals[0] / (s_vals[1] + 1e-12))
        if len(s_vals) > 1 else float("inf"),
    }


def build_pca_basis(feats_flat: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """
    默认 PCA 基（对照组）：对所有组堆叠的特征做 PCA，取前 2 主成分。

    方差最大方向常与 w 大幅错位，故预期 in-plane 能量低、组间差异被压扁——与判别基
    并列即可量化「基的选择」对结论的影响。

    Args:
        feats_flat: [G*N, 512] 所有组特征堆叠（PCA 内部会中心化）。

    Returns:
        (P [512,2] 列正交单位基, 诊断量：两个主成分的解释方差比)。
    """
    pca = PCA(n_components=2, svd_solver="randomized", random_state=0).fit(feats_flat)
    return pca.components_.T, {
        "pc1_explained_var_ratio": float(pca.explained_variance_ratio_[0]),
        "pc2_explained_var_ratio": float(pca.explained_variance_ratio_[1]),
    }


def in_plane_energy(P: np.ndarray, w: np.ndarray) -> float:
    """投影诚实性：||P^T w|| / ||w||，打分表投影后剩下的比例（1=无损，0=全丢）。"""
    return float(np.linalg.norm(P.T @ w) / (np.linalg.norm(w) + 1e-12))


def format_basis_diag(basis: str, diag: dict[str, float]) -> str:
    """把投影基诊断量排成 panel 内可读的短多行文本。"""
    if basis == "discriminative":
        return (f"between-group var:\n"
                f"  {diag['between_var_along_e1']:.1%} along e1 (pure scaling)\n"
                f"  {diag['between_var_in_perp']:.1%} in e1-perp\n"
                f"e2 captures {diag['e2_perp_var_ratio']:.1%} of perp\n"
                f"s1/s2 = {diag['e2_singular_value_ratio']:.2f}")
    return (f"PC1 var {diag['pc1_explained_var_ratio']:.1%}\n"
            f"PC2 var {diag['pc2_explained_var_ratio']:.1%}")


# ============================================================
# 2. Δlogit 通路分解
# ============================================================
def decompose_delta_logit(
    feats: np.ndarray, w_eff: np.ndarray, logits: np.ndarray, ref_idx: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    把 logit 随属性的变化拆成 fc 头贡献与 conv 特征贡献（代数恒等式，非近似）：
        Δlogit(a_ref -> g) = <w_g - w_ref, f(x, a_ref)>  +  <w_g, f(x, g) - f(x, a_ref)>
    bias 不随属性变，恒等式中自动抵消。

    Args:
        feats  : [G, N, 512] 反事实特征。
        w_eff  : [G, 512]    各组有效超平面。
        logits : [G, N]      各组反事实 logit（用于校验恒等式）。
        ref_idx: 参照组下标。

    Returns:
        (fc_term [G, N], conv_term [G, N], 最大恒等式残差 max|fc+conv-Δlogit|)。
    """
    n_groups = w_eff.shape[0]
    f_ref, w_ref = feats[ref_idx], w_eff[ref_idx]
    fc_term = np.stack([(w_eff[g] - w_ref) @ f_ref.T for g in range(n_groups)])
    conv_term = np.stack([(feats[g] - f_ref) @ w_eff[g] for g in range(n_groups)])
    residual = float(np.abs(fc_term + conv_term - (logits - logits[ref_idx][None, :])).max())
    return fc_term, conv_term, residual


def audit_decomposition_orders(
    feats: np.ndarray, w_eff: np.ndarray, logits: np.ndarray, ref_idx: int,
) -> dict[str, np.ndarray]:
    """
    分解**顺序敏感性**审计。decompose_delta_logit 用的是「先换 w 再换 f」这一种顺序：
        order-1: Δlogit = <Δw, f_ref>        + <w_g, Δf>
    反序同样是精确恒等式，但归因不同：
        order-2: Δlogit = <Δw, f_g>          + <w_ref, Δf>
    两者之差正是交互项 I = <Δw, Δf>（order-1 把 I 全给 conv，order-2 全给 fc）。
    因此单一顺序的「fc 占 X%」是**路径依赖**的读数，必须一并报告对称（Shapley）版——它把 I 均分，
    是两个顺序的平均，也仍是精确分解：
        shapley: fc = <Δw, (f_ref+f_g)/2>,   conv = <(w_ref+w_g)/2, Δf>

    Args:
        feats  : [G, N, 512] 反事实特征；w_eff: [G, 512]；logits: [G, N]；ref_idx: 参照组下标。

    Returns:
        各口径的 [G, N] 逐样本贡献（fc/conv × order1/order2/shapley）与交互项 interaction，
        外加三口径各自的最大恒等式残差（key 以 residual_ 开头，标量数组）。
    """
    n_groups = w_eff.shape[0]
    f_ref, w_ref = feats[ref_idx], w_eff[ref_idx]
    delta_true = logits - logits[ref_idx][None, :]

    fc_o1 = np.stack([(w_eff[g] - w_ref) @ f_ref.T for g in range(n_groups)])
    conv_o1 = np.stack([(feats[g] - f_ref) @ w_eff[g] for g in range(n_groups)])
    fc_o2 = np.stack([(w_eff[g] - w_ref) @ feats[g].T for g in range(n_groups)])
    conv_o2 = np.stack([(feats[g] - f_ref) @ w_ref for g in range(n_groups)])
    interaction = fc_o2 - fc_o1                             # = <Δw, Δf>，也 = conv_o1 - conv_o2
    fc_sh, conv_sh = 0.5 * (fc_o1 + fc_o2), 0.5 * (conv_o1 + conv_o2)

    return {
        "fc_order1": fc_o1, "conv_order1": conv_o1,
        "fc_order2": fc_o2, "conv_order2": conv_o2,
        "fc_shapley": fc_sh, "conv_shapley": conv_sh,
        "interaction": interaction,
        "residual_order1": np.array(np.abs(fc_o1 + conv_o1 - delta_true).max()),
        "residual_order2": np.array(np.abs(fc_o2 + conv_o2 - delta_true).max()),
        "residual_shapley": np.array(np.abs(fc_sh + conv_sh - delta_true).max()),
    }


def audit_pathway_auc_attribution(
    fc_term: np.ndarray, conv_term: np.ndarray, logits: np.ndarray, labels: np.ndarray,
    ref_idx: int,
) -> dict[int, dict[str, float]]:
    """
    把通路归因换到 **AUC（排序）口径**，与条件化消融的 knockout 实验同构、可直接对账。

    logit 幅度口径（mean |贡献|）回答「哪条通路把 logit 推得更多」，但 logit 平移**不改排序**、
    对 AUC 无影响。消融实验（docs/conditioning_ablation_e1_pathway_knockout.md）用的是 AUC 口径：
    置零某通路生成器后看真实-置换 age 的性能差是否塌掉。这里用分解项做等价的加法版归因——
    在参照组 logit 上**只叠加一条通路的贡献**，看 AUC 动多少：

        auc_fc_only   = AUC(logit_ref + fc_term_g)      仅 fc 通路生效
        auc_conv_only = AUC(logit_ref + conv_term_g)    仅 conv 通路生效
        auc_full      = AUC(logit_g)                    两条都生效（= 真实反事实 logit）

    同时报告每条通路贡献自身的 mean / std 与「单独作为分数的 AUC」：
      · std≈0 且 auc_of_term≈0.5 ⇒ 该通路只是**对所有样本近乎一致的平移**，不可能改排序；
      · auc_of_term 明显偏离 0.5 ⇒ 该通路的贡献与标签相关，真的在改排序。

    Args:
        fc_term / conv_term: [G, N] 某一口径下的逐样本贡献；logits: [G, N]；labels: [N]；
        ref_idx: 参照组下标。

    Returns:
        {组下标: {各项 AUC 与统计量}}，参照组自身跳过。
    """
    from sklearn.metrics import roc_auc_score

    auc_ref = float(roc_auc_score(labels, logits[ref_idx]))
    out: dict[int, dict[str, float]] = {}
    for g in range(logits.shape[0]):
        if g == ref_idx:
            continue
        out[g] = {
            "auc_ref": auc_ref,
            "auc_full": float(roc_auc_score(labels, logits[g])),
            "auc_fc_only": float(roc_auc_score(labels, logits[ref_idx] + fc_term[g])),
            "auc_conv_only": float(roc_auc_score(labels, logits[ref_idx] + conv_term[g])),
            "fc_mean": float(fc_term[g].mean()), "fc_std": float(fc_term[g].std()),
            "conv_mean": float(conv_term[g].mean()), "conv_std": float(conv_term[g].std()),
            # 该通路贡献单独作为分数的 AUC：≈0.5 表示与标签无关（纯平移或纯噪声）
            "auc_of_fc_term": float(roc_auc_score(labels, fc_term[g])),
            "auc_of_conv_term": float(roc_auc_score(labels, conv_term[g])),
        }
    return out


# ============================================================
# 3. 渲染
# ============================================================
def line_from_coeffs(
    coef: np.ndarray, const: float, x_lim: tuple[float, float], y_lim: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    把 2D 决策线 coef·u + const = 0 解析地裁剪到视窗内。

    Args:
        coef : [2] = P^T w（2D 系数）。
        const: <w, mu> + b。
        x_lim / y_lim: 视窗范围。

    Returns:
        (xs, ys) 线段两端点；线与视窗不相交时返回 None。
    """
    c1, c2 = float(coef[0]), float(coef[1])
    pts: list[tuple[float, float]] = []
    if abs(c2) > 1e-12:                                     # 与左右边界求交
        for x in x_lim:
            pts.append((x, -(c1 * x + const) / c2))
    if abs(c1) > 1e-12:                                     # 与上下边界求交（近竖直线走这支）
        for y in y_lim:
            pts.append((-(c2 * y + const) / c1, y))
    tol_x = 1e-9 * max(1.0, abs(x_lim[1] - x_lim[0]))
    tol_y = 1e-9 * max(1.0, abs(y_lim[1] - y_lim[0]))
    inside = [(x, y) for x, y in pts
              if x_lim[0] - tol_x <= x <= x_lim[1] + tol_x
              and y_lim[0] - tol_y <= y <= y_lim[1] + tol_y]
    if len(inside) < 2:
        return None
    # 取距离最远的两点为端点（避免角点处重复交点）
    best = max(((a, b) for i, a in enumerate(inside) for b in inside[i + 1:]),
               key=lambda ab: (ab[0][0] - ab[1][0]) ** 2 + (ab[0][1] - ab[1][1]) ** 2)
    return np.array([best[0][0], best[1][0]]), np.array([best[0][1], best[1][1]])


def render_basis_figure(
    basis: str, U: np.ndarray, P: np.ndarray, w_eff: np.ndarray, bias: float, mu: np.ndarray,
    labels: np.ndarray, group_ids: np.ndarray, aucs: np.ndarray, logits: np.ndarray,
    energies: list[float], diag: dict[str, float], plot_idx: np.ndarray,
    group_labels: dict[int, str], class_names: tuple[str, str], ncols: int,
    header: str, out_path: Path,
) -> None:
    """
    渲染**单套投影基**的一张图：G 个子群 panel，每格 = 该组反事实点云 + 该组精确决策线
    + 全组平均超平面（虚线固定参照）。所有 panel 共用视窗（同一 mu、同一 P）故坐标可比。

    Args:
        basis      : 'discriminative' | 'pca'，决定标题与诊断文本格式。
        U          : [G, N, 2] 已投影坐标。
        P          : [512, 2] 投影基。
        w_eff      : [G, 512] 各组有效超平面；bias: 共享 fc.bias；mu: [512] 共同中心。
        labels     : [N] 真实标签；group_ids: [N] 每个样本真实所属子群下标。
        aucs       : [G] 各反事实条件下对全体样本的 AUC；logits: [G, N] 反事实 logit。
        energies   : [G] 各组 in-plane 能量；diag: 该基的诊断量。
        plot_idx   : 参与绘制的样本下标（仅控制散点密度，统计仍用全部样本）。
        group_labels: 子群下标 -> 显示名；class_names: (y=0 名, y=1 名)。
        ncols      : panel 列数；header: suptitle 首行（数据集/模型/checkpoint 信息）。
        out_path   : 输出 PNG 路径。
    """
    n_groups = w_eff.shape[0]
    nrows = math.ceil(n_groups / ncols)
    w_bar = w_eff.mean(axis=0)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.9 * nrows), squeeze=False)

    # 全 panel 共用视窗：坐标可比是「组间位移可读」的前提
    all_u = U.reshape(-1, 2)
    pad = 0.06 * (all_u.max(axis=0) - all_u.min(axis=0) + 1e-9)
    x_lim = (float(all_u[:, 0].min() - pad[0]), float(all_u[:, 0].max() + pad[0]))
    y_lim = (float(all_u[:, 1].min() - pad[1]), float(all_u[:, 1].max() + pad[1]))

    for g in range(nrows * ncols):
        ax = axes[g // ncols][g % ncols]
        if g >= n_groups:
            ax.axis("off")
            continue

        u_g, in_group = U[g], group_ids == g
        sel_other = plot_idx[~in_group[plot_idx]]
        sel_in = plot_idx[in_group[plot_idx]]
        # 其他组真实样本：淡（纯反事实）；真实属于本组的：黑边高亮（条件 == 真实属性）
        ax.scatter(u_g[sel_other, 0], u_g[sel_other, 1], c=CLASS_COLORS[labels[sel_other]],
                   s=14, alpha=0.26, edgecolors="none", zorder=2)
        ax.scatter(u_g[sel_in, 0], u_g[sel_in, 1], c=CLASS_COLORS[labels[sel_in]],
                   s=40, alpha=0.95, edgecolors="black", linewidths=0.7, zorder=3)

        # 该组精确决策线：<P^T w_g, u> + (<w_g, mu> + b) = 0
        seg = line_from_coeffs(P.T @ w_eff[g], float(w_eff[g] @ mu + bias), x_lim, y_lim)
        if seg is not None:
            ax.plot(seg[0], seg[1], color="black", lw=2.4, zorder=5)
        # 全组平均超平面（固定参照，各 panel 相同）
        seg_bar = line_from_coeffs(P.T @ w_bar, float(w_bar @ mu + bias), x_lim, y_lim)
        if seg_bar is not None:
            ax.plot(seg_bar[0], seg_bar[1], color="0.45", lw=1.6, ls="--", zorder=4)

        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_title(f"{group_labels[g]}   n_true={int(in_group.sum())}\n"
                     f"counterfactual AUC={aucs[g]:.4f}   mean logit={logits[g].mean():+.2f}\n"
                     f"||w||={np.linalg.norm(w_eff[g]):.2f}   "
                     f"in-plane energy={energies[g]:.3f}", fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
        if g == 0:
            ax.text(0.025, 0.975, format_basis_diag(basis, diag), transform=ax.transAxes,
                    va="top", ha="left", fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7",
                          "boxstyle": "round,pad=0.35"}, zorder=6)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLORS[0], markersize=9,
               label=f"{class_names[0]} (y=0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLORS[1], markersize=9,
               label=f"{class_names[1]} (y=1)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.5", markeredgecolor="black",
               markersize=9, label="true member of this subgroup"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.6", alpha=0.35, markersize=8,
               label="other subgroups (pure counterfactual)"),
        Line2D([0], [0], color="black", lw=2.4, label="exact decision line of this subgroup"),
        Line2D([0], [0], color="0.45", lw=1.6, ls="--", label="mean hyperplane (fixed reference)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3 if ncols <= 2 else 6, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.008))
    fig.suptitle(
        f"{header}\n"
        f"Projection basis: {BASIS_TITLES[basis]}\n"
        f"{BASIS_EXPLAIN[basis]}\n"
        "Points and line shown together: features and hyperplane move jointly (gauge freedom), "
        "so only the position of points RELATIVE to the line is meaningful",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.99 - 0.055 * nrows / max(1, nrows)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def render_pathway_figure(
    fc_term: np.ndarray, conv_term: np.ndarray, ref_idx: int, residual: float,
    group_labels: dict[int, str], header: str, out_path: Path,
) -> None:
    """
    Δlogit 通路分解图：左 = 逐组 fc/conv 贡献 boxplot；右 = 各组两通路平均绝对贡献占比。

    Args:
        fc_term / conv_term: [G, N] 两通路的逐样本贡献。
        ref_idx  : 参照组下标（该组自身恒为 0，不绘制）。
        residual : 恒等式最大残差（数值校验，应在 float32 误差量级）。
        group_labels: 子群下标 -> 显示名；header: suptitle。
        out_path : 输出 PNG 路径。
    """
    n_groups = fc_term.shape[0]
    groups = [g for g in range(n_groups) if g != ref_idx]
    width = max(13.0, 1.9 * len(groups) + 6.0)
    fig, (ax_box, ax_bar) = plt.subplots(1, 2, figsize=(width, 5.8),
                                        gridspec_kw={"width_ratios": [1.6, 1.0]})

    positions, box_data, colors = [], [], []
    for i, g in enumerate(groups):
        positions.extend([i - 0.17, i + 0.17])
        box_data.extend([fc_term[g], conv_term[g]])
        colors.extend(["#2b6cb0", "#c05621"])
    bp = ax_box.boxplot(box_data, positions=positions, widths=0.3, patch_artist=True,
                        showfliers=False, medianprops={"color": "black", "lw": 1.4})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax_box.axhline(0.0, color="0.4", lw=1.0, ls=":")
    ax_box.set_xticks(range(len(groups)))
    ax_box.set_xticklabels([group_labels[g] for g in groups], fontsize=8.5, rotation=20,
                           ha="right")
    ax_box.set_ylabel(f"delta-logit contribution\n(vs reference {group_labels[ref_idx]})")
    ax_box.set_title("Per-sample delta-logit decomposition (exact identity, not an approximation)\n"
                     f"fc head <dw, f_ref>  vs  conv features <w_g, df>   "
                     f"max residual={residual:.2e}", fontsize=11)
    ax_box.legend(handles=[
        Line2D([0], [0], color="#2b6cb0", lw=8, alpha=0.75,
               label="fc head: the scoring table changed"),
        Line2D([0], [0], color="#c05621", lw=8, alpha=0.75,
               label="conv features: the representation changed"),
    ], fontsize=9, loc="best")

    fc_abs = np.array([np.abs(fc_term[g]).mean() for g in groups])
    conv_abs = np.array([np.abs(conv_term[g]).mean() for g in groups])
    share = fc_abs / (fc_abs + conv_abs + 1e-12)
    x = np.arange(len(groups))
    ax_bar.bar(x, share, color="#2b6cb0", alpha=0.8, label="fc head share")
    ax_bar.bar(x, 1.0 - share, bottom=share, color="#c05621", alpha=0.8, label="conv feature share")
    for xi, s, fa, ca in zip(x, share, fc_abs, conv_abs):
        ax_bar.text(xi, 0.5, f"fc {s:.0%}\nconv {1 - s:.0%}", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
        ax_bar.text(xi, 1.02, f"|fc|={fa:.2f}\n|conv|={ca:.2f}", ha="center", fontsize=7.5)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([group_labels[g] for g in groups], fontsize=8.5, rotation=20,
                           ha="right")
    ax_bar.set_ylim(0, 1.22)
    ax_bar.set_ylabel("share of mean |contribution|")
    ax_bar.set_title("Relative weight of the two pathways (mean |contribution|)", fontsize=11)
    ax_bar.legend(fontsize=9, loc="lower right")

    fig.suptitle(f"{header}\nWhich pathway carries the attribute conditioning — "
                 "fc classifier head vs conv feature reorganisation", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")
