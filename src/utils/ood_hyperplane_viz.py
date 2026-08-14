"""
跨库迁移（OOD）下的超平面可视化：ID↔OOD **同帧**叠加
=====================================================
`docs/hyperplane_subgroup_visualisation.md` 的方法在 OOD 设定下的扩展，服务
`docs/ood_experiment_v2_results.md`（MIMIC↔CheXpert，ERM / SWAD / HyperAdapt）。

为什么 OOD 需要一张**额外**的图
--------------------------------
把既有那套图原样搬到 target 上（只换评估数据）只能回答「模型在 target 上长什么样」，
**看不到迁移本身**。而 OOD 的关键事实恰恰是一个几何恒等式：

  · **超平面完全没动**。ERM/SWAD 的 `w`、HyperAdapt 的 `w_eff(a)` 与 `fc.bias` 全部只由
    checkpoint 决定，与评估数据无关 ⇒ ID 与 OOD 的决策线**逐位相同**。
  · 变的只有**特征云** `f(x)`：target 的图像落到了特征空间的另一处。

⇒ 「迁移出了什么问题」在这套坐标里必然表现为**点云相对一条不动的线发生了平移**，
   而不是「边界变了」。本模块就是把这件事画出来并量化。

同帧的实现（精确，无需重跑 ID）
--------------------------------
投影基取 **source-ID 那次运行的 `P` 与 `mu`**（已落盘于 `outputs/<source>/hyperplane_viz/*.npz`），
OOD 特征用同一 `P`、同一 `mu` 投影，两块点云因此在**同一坐标系**内可直接比较：

    U_id  = (f_id  − mu_id) @ P_id          （ID 那次运行已存盘，直接读）
    U_ood = (f_ood − mu_id) @ P_id          （本模块现算）

对 HyperAdapt 还有一个更强的性质：`P_discriminative` 只由 `w_eff` 构造、`w_eff` 只由 checkpoint
决定 ⇒ **OOD 运行自己算出的 P 与 ID 存盘的 P 必然逐位相同**，可作 checkpoint 一致性断言
（`assert_same_frame`）。blind 模型的 `e1 = w/‖w‖` 同理相同，但 `e2` 取自各组特征均值、
是数据侧产物，故 blind 只断言 `e1`。

判读红线（继承自 hyperplane_subgroup_visualisation.md §7，并新增两条）
---------------------------------------------------------------------
· **不得**说「OOD 下超平面变了/HN 调整了边界」——线是模型的属性，与评估数据无关，逐位未动。
· `d∥`（沿判别轴的平移）**不改任何组内排序**，因此**不能**用它解释 AUC 差异；它对应的是
  **校准**（CITL/ECE），正是 `ood_experiment_v2_results.md` §5.2 报的那件事。
· `d⊥` 对该分类器**完全不可见**，更不能用来推性能。
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.utils.hyperplane_viz import CLASS_COLORS, line_from_coeffs

GRID_COLOR = "0.85"
LINE_COLOR = "#0b0b0b"
DOMAIN_EDGE = {"id": "#2f7d63", "ood": "#a3541c"}            # ID = 绿, OOD = 橙（质心/箭头描边）


def assert_same_frame(p_ood: np.ndarray, p_id: np.ndarray, axes: int, atol: float = 1e-5) -> None:
    """
    断言 OOD 运行与 ID 运行处在同一投影帧（= 用的是同一份 checkpoint）。

    基向量由 SVD/PCA 得到，符号自由 ⇒ 只比较 |cos|，不比较原始分量。

    Args:
        p_ood: [512, 2] OOD 运行自算的基。
        p_id : [512, 2] ID 运行存盘的基。
        axes : 要比较的轴数——HyperAdapt 判别基两轴均由 w_eff 决定故传 2；
               blind 基只有 e1 = w/‖w‖ 由模型决定（e2 取自数据）故传 1。
        atol : |cos| 与 1 的允许偏差。

    Raises:
        AssertionError: 任一轴不平行 ⇒ checkpoint 不一致或 ID npz 张冠李戴。
    """
    for k in range(axes):
        cos = abs(float(p_ood[:, k] @ p_id[:, k]))
        assert abs(cos - 1.0) <= atol, (
            f"投影基第 {k + 1} 轴与 ID 运行不平行（|cos|={cos:.6f}）：ID/OOD 用的不是同一 checkpoint"
        )


def domain_shift_metrics(
    u_id: np.ndarray, u_ood: np.ndarray, labels_id: np.ndarray, labels_ood: np.ndarray,
    logits_id: np.ndarray, logits_ood: np.ndarray,
) -> dict[str, float]:
    """
    量化 ID→OOD 的点云位移（同一帧内），拆成判别轴与其正交补两部分。

    尺度统一取 **ID 云在各轴上的 SD**（而非 pooled SD）：读数因此是「target 云的质心比
    source 云偏了多少个 source 标准差」，与 `blind_hyperplane_viz.subgroup_offsets` 的
    `d∥/d⊥` 同量纲、可互相引用。

    另报**逐类**位移：prior shift（P(Y) 变了）会让整体质心动，但只有当 y=0 与 y=1 两类
    **同向同幅**移动时才是纯粹的「云整体平移」；两类位移不同 ⇒ 类间分离度本身也变了，
    那才可能动 AUC。

    Args:
        u_id / u_ood      : [N, 2] 同一帧内的 ID / OOD 投影坐标。
        labels_id / labels_ood: [N] 各自的真实标签。
        logits_id / logits_ood: [N] 各自的 logit（用于 mean-logit 平移的原始尺度读数）。

    Returns:
        位移与分离度读数字典（SD 单位的 d_along/d_perp、逐类版本、mean logit、类间分离度）。
    """
    sd = u_id.std(axis=0) + 1e-12                           # 以 source 云为尺子
    d_all = u_ood.mean(axis=0) - u_id.mean(axis=0)

    def class_sep(u: np.ndarray, y: np.ndarray) -> float:
        """该域内 y=1 与 y=0 质心沿判别轴的间距（SD 单位）= 类间分离度。"""
        if not (y == 0).any() or not (y == 1).any():
            return float("nan")
        return float((u[y == 1, 0].mean() - u[y == 0, 0].mean()) / sd[0])

    out = {
        "d_along": float(d_all[0] / sd[0]), "d_perp": float(d_all[1] / sd[1]),
        "mean_logit_id": float(logits_id.mean()), "mean_logit_ood": float(logits_ood.mean()),
        "mean_logit_shift": float(logits_ood.mean() - logits_id.mean()),
        "pos_rate_id": float(labels_id.mean()), "pos_rate_ood": float(labels_ood.mean()),
        "class_separation_id": class_sep(u_id, labels_id),
        "class_separation_ood": class_sep(u_ood, labels_ood),
    }
    for y_val in (0, 1):
        if (labels_id == y_val).any() and (labels_ood == y_val).any():
            d_y = u_ood[labels_ood == y_val].mean(axis=0) - u_id[labels_id == y_val].mean(axis=0)
            out[f"d_along_y{y_val}"] = float(d_y[0] / sd[0])
            out[f"d_perp_y{y_val}"] = float(d_y[1] / sd[1])
    return out


def render_domain_shift_figure(
    u_id: np.ndarray, u_ood: np.ndarray, labels_id: np.ndarray, labels_ood: np.ndarray,
    lines: list[tuple[np.ndarray, float]], mu: np.ndarray, shift: dict[str, float],
    auc_id: float, auc_ood: float, domain_names: tuple[str, str], plot_n: int,
    header: str, footer: str, out_path: Path, seed: int = 0,
) -> None:
    """
    渲染 ID | OOD 两 panel 的**同帧**对照图：一条（或一组）不动的决策线 + 两块点云。

    读法：两 panel 的坐标系、视窗、决策线**完全相同**——唯一变的是点。因此
    「点云整体横移了多少」就是分布偏移打在这个模型判别轴上的投影，直接对应校准漂移；
    「y0/y1 两质心的横向间距变了多少」才是判别能力（AUC）的变化。

    Args:
        u_id / u_ood      : [N, 2] 同帧投影坐标；labels_id / labels_ood: [N] 标签。
        lines : [(coef2d, const)] 要画的决策线系数——blind 传 1 条，HyperAdapt 传 8 条
                （逐子群 `w_eff`；CXR 上它们几乎重合，重合本身即读数）。
        mu    : [512] 共同中心（仅供标题记录，绘制用不到）。
        shift : domain_shift_metrics 的返回值。
        auc_id / auc_ood: 两域的 Overall AUC（全部样本，非绘制子集）。
        domain_names    : (ID panel 名, OOD panel 名)，如 ("MIMIC-CXR (source, ID)", ...)。
        plot_n: 每 panel 最多绘制的散点数（仅控制密度，统计仍用全部样本）。
        header / footer : suptitle 首行 / 末段说明。
        out_path: 输出 PNG 路径；seed: 散点抽样种子。
    """
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 7.0), sharex=True, sharey=True)

    # 共用视窗：两域坐标可比是本图的全部意义所在
    both = np.vstack([u_id, u_ood])
    pad = 0.06 * (both.max(axis=0) - both.min(axis=0) + 1e-9)
    x_lim = (float(both[:, 0].min() - pad[0]), float(both[:, 0].max() + pad[0]))
    y_lim = (float(both[:, 1].min() - pad[1]), float(both[:, 1].max() + pad[1]))
    segs = [line_from_coeffs(coef, const, x_lim, y_lim) for coef, const in lines]

    panels = ((u_id, labels_id, domain_names[0], auc_id, "id"),
              (u_ood, labels_ood, domain_names[1], auc_ood, "ood"))
    for ax, (u, y, name, auc, key) in zip(axes, panels):
        sel = np.sort(rng.choice(len(y), size=min(plot_n, len(y)), replace=False))
        ax.scatter(u[sel, 0], u[sel, 1], c=CLASS_COLORS[y[sel]], s=17, alpha=0.5,
                   edgecolors="none", zorder=3)
        # 该域的 y0/y1 质心（空心 = y0，实心 = y1），描边用域色以便跨 panel 追踪
        for y_val in (0, 1):
            if not (y == y_val).any():
                continue
            c = u[y == y_val].mean(axis=0)
            ax.scatter([c[0]], [c[1]], marker="o", s=210,
                       facecolors=CLASS_COLORS[y_val] if y_val == 1 else "none",
                       edgecolors=DOMAIN_EDGE[key] if y_val == 1 else CLASS_COLORS[y_val],
                       linewidths=2.4, zorder=6)
        # 另一域的整体质心（灰十字）+ 位移箭头：把「云动了」画成一个可量的量
        c_from, c_to = u_id.mean(axis=0), u_ood.mean(axis=0)
        ax.scatter([c_from[0]], [c_from[1]], marker="P", s=150, c=DOMAIN_EDGE["id"],
                   edgecolors="white", linewidths=1.0, zorder=7)
        ax.scatter([c_to[0]], [c_to[1]], marker="X", s=150, c=DOMAIN_EDGE["ood"],
                   edgecolors="white", linewidths=1.0, zorder=7)
        if not np.allclose(c_from, c_to):
            ax.annotate("", xy=tuple(c_to), xytext=tuple(c_from), zorder=8,
                        arrowprops={"arrowstyle": "-|>", "lw": 2.0, "color": "#52514e",
                                    "shrinkA": 6, "shrinkB": 6})
        for seg in segs:
            if seg is not None:
                lw = 2.6 if len(segs) == 1 else 1.5
                ax.plot(seg[0], seg[1], color=LINE_COLOR, lw=lw, alpha=0.95, zorder=9)

        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.grid(True, color=GRID_COLOR, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{name}\nn={len(y)}   overall AUC={auc:.4f}   "
                     f"y=1 rate={y.mean():.3f}   mean logit="
                     f"{shift['mean_logit_id' if key == 'id' else 'mean_logit_ood']:+.2f}",
                     fontsize=11)
    axes[0].set_xlabel("e1 = discriminative direction (x-position is monotone in the logit)",
                       fontsize=10)
    axes[1].set_xlabel("e1 = discriminative direction (x-position is monotone in the logit)",
                       fontsize=10)
    axes[0].set_ylabel("e2 = between-group direction inside e1's orthogonal complement",
                       fontsize=10)

    sep_delta = shift["class_separation_ood"] - shift["class_separation_id"]
    axes[0].text(
        0.015, 0.985,
        f"cloud shift ID -> OOD (in SD of the ID cloud)\n"
        f"  d|| = {shift['d_along']:+.2f} SD   (score / calibration axis)\n"
        f"  d-perp = {shift['d_perp']:+.2f} SD   (invisible to w)\n"
        f"  per class: d||(y=0)={shift.get('d_along_y0', float('nan')):+.2f}, "
        f"d||(y=1)={shift.get('d_along_y1', float('nan')):+.2f}\n"
        f"mean logit {shift['mean_logit_id']:+.2f} -> {shift['mean_logit_ood']:+.2f} "
        f"({shift['mean_logit_shift']:+.2f})\n"
        f"class separation (y1-y0 along e1)\n"
        f"  {shift['class_separation_id']:.2f} -> {shift['class_separation_ood']:.2f} SD "
        f"({sep_delta:+.2f})",
        transform=axes[0].transAxes, va="top", ha="left", fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.7",
              "boxstyle": "round,pad=0.35"}, zorder=10)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLORS[0], markersize=9,
               label="y=0 (finding)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLORS[1], markersize=9,
               label="y=1 (no finding)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="0.35",
               markeredgewidth=2.2, markersize=12, label="y=0 centroid of this panel's domain"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.55", markeredgecolor="0.2",
               markersize=12, label="y=1 centroid of this panel's domain"),
        Line2D([0], [0], marker="P", color="w", markerfacecolor=DOMAIN_EDGE["id"], markersize=11,
               label="overall centroid, ID (same in both panels)"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor=DOMAIN_EDGE["ood"], markersize=11,
               label="overall centroid, OOD (same in both panels)"),
        Line2D([0], [0], color=LINE_COLOR, lw=2.6,
               label=("the decision line (IDENTICAL in both panels)" if len(segs) == 1 else
                      "the 8 per-subgroup decision lines (IDENTICAL in both panels)")),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{header}\n{footer}", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def render_subgroup_shift_bars(
    per_group: dict[int, dict[str, float]], group_labels: dict[int, str],
    domain_names: tuple[str, str], header: str, out_path: Path,
) -> None:
    """
    逐子群的三栏对照条形图：`d∥`（校准侧）| 跨域 AUC 差（混杂）| **迁移落差**（受控）。

    三栏各回答一个不同的问题，混谈会得出错误结论：

    1. **`d∥`**：同一权重下 target 组质心相对 source 组质心沿判别轴的平移
       ⇒ 纯操作点/校准偏移，**不改组内排序**（`hyperplane_subgroup_visualisation.md` §4.3 口径 B）。
    2. **AUC(OOD) − AUC(src-ID)**：同一权重、**不同数据集**上的组内 AUC 差。
       ⚠️ 它把「target 本身更难」与「迁移掉了多少」**混在一起**，单看它会把库间难度差误判成迁移损失。
    3. **AUC(OOD) − AUC(tgt-ID)**：**同一批 target 图像**上，source 模型 vs target 自训模型
       ⇒ 数据侧完全受控，这才是**迁移落差**的干净读数（口径 C，唯一能支持性能论断的一栏）。

    Args:
        per_group   : {组下标: {"d_along", "auc_id", "auc_ood", "auc_tgt_id"}}。
        group_labels: 组下标 -> 显示名。
        domain_names: (source 库名, target 库名)，写进各栏轴标签以免读者搞混比的是谁。
        header      : suptitle；out_path: 输出 PNG。
    """
    groups = sorted(per_group)
    names = [group_labels[g] for g in groups]
    x = np.arange(len(groups))
    d_along = np.array([per_group[g]["d_along"] for g in groups])
    d_auc = np.array([per_group[g]["auc_ood"] - per_group[g]["auc_id"] for g in groups])
    d_gap = np.array([per_group[g]["auc_ood"] - per_group[g]["auc_tgt_id"] for g in groups])
    src_name, tgt_name = domain_names

    fig, (ax_l, ax_m, ax_r) = plt.subplots(
        1, 3, figsize=(max(16.0, 1.35 * len(groups) + 8.0), 6.4))
    for ax, vals, title, ylab, color in (
        (ax_l, d_along, "1. Score-axis translation  (calibration side)",
         f"d|| : {src_name} -> {tgt_name} centroid shift along e1 (SD)", "#2b6cb0"),
        (ax_m, d_auc, "2. Cross-dataset AUC difference  (CONFOUNDED)",
         f"AUC on {tgt_name} - AUC on {src_name}\n(same weights, DIFFERENT data)", "#c05621"),
        (ax_r, d_gap, "3. Transfer gap  (controlled: same images)",
         f"AUC of {src_name}-trained - AUC of {tgt_name}-trained\n(both on {tgt_name})", "#6b46a3"),
    ):
        ax.bar(x, vals, color=color, alpha=0.85)
        ax.axhline(0.0, color="0.4", lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8.5, rotation=20, ha="right")
        ax.set_ylabel(ylab, fontsize=9.5)
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="y", color=GRID_COLOR, lw=0.6)
        ax.set_axisbelow(True)
        span = float(np.nanmax(np.abs(vals))) if len(vals) else 1.0
        for xi, v in zip(x, vals):
            if math.isnan(v):
                continue
            ax.text(xi, v + math.copysign(0.04 * max(span, 1e-9), v), f"{v:+.3f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=8)

    fig.suptitle(f"{header}\n"
                 "Panel 1 moves the operating point, panels 2-3 move the ranking. A large d|| "
                 "with flat AUC bars = calibration drift WITHOUT loss of discrimination.\n"
                 "Only panel 3 is a valid performance statement: panel 2 confounds transfer "
                 "loss with how hard the two datasets are.", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")
