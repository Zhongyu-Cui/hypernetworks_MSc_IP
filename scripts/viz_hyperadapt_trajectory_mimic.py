"""
HyperAdapt 属性条件化可视化（MIMIC 三属性版）：sex/race/age 子群 → Δθ 权重轨迹
=============================================================================
HAM 版（scripts/viz_hyperadapt_weight_trajectory.py）沿单一 age 序数轴扫掠；MIMIC 的
HyperAdapt 是三属性条件通路 (sex, race, age_group)，每属性二值 → 8 个子群，没有单一
连续轴。故这里改为：

  1. 枚举 8 个离散子群 (sex×race×age) 作为**锚点**，看它们在 Δθ 权重空间的分布
     （挤成一簇=近坍缩/几乎不条件化；散开=条件化了，不论是否帮到公平）。
  2. 三条**轴向探针**：对每个属性把它的 16 维 cat-embedding 从 level0 线性插值到 level1
     （另两属性固定在 level0），得到 3 段光滑探针曲线——分别度量 sex/race/age 单独把
     权重推动了多远、往哪个方向。
  3. **逐属性边际位移** Δ_attr = 对另两属性的 4 种取值求平均的 ‖Δθ(attr=1)-Δθ(attr=0)‖，
     是「该属性驱动权重变化」的标量，可比出哪个属性主导（若三者都≈0=HN 基本没条件化）。

复用 HAM 脚本的 delta_theta_from_profile / pca_2d（Δθ 只依赖属性、与图像无关，纯 CPU）。
主要动机：MIMIC 的 I(Y;A|X)≈0 是判决性 null，预期轨迹应比 HAM **明显更坍缩**，作为
「HAM 强条件化」的反面对照，展示该诊断的判别力。

产物写入 outputs/analysis/hyperadapt_trajectory/<tag>/：
    1. trajectory_pca.png       8 子群锚点 + 3 轴探针的 PCA-2D 权重轨迹
    2. per_subgroup_modulation.png 每层 ‖M‖ 随 8 子群
    3. scalars.json             逐属性边际位移 / 逐层 / embedding 距离 / Δθ 相对 backbone
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无头后端
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path("/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# 复用 HAM 脚本里与属性数无关的核心工具
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from src.models.resnet18_hyperadapt import ResNet18HyperAdapt  # noqa: E402
from viz_hyperadapt_weight_trajectory import delta_theta_from_profile, pca_2d  # noqa: E402

# MIMIC 三属性二值编码（对齐 mimic_cxr_dataset：sex Male=0/Female=1；race White=0/Non-White=1；
# age_group 0=<thr / 1=>=thr）。子群标签用 3 位缩写便于图上标注。
ATTR_NAMES = ["sex", "race", "age"]
ATTR_LEVEL_LABELS = {
    "sex": ["M", "F"],
    "race": ["W", "N"],   # White / Non-White
    "age": ["y", "o"],    # young(<thr) / old(>=thr)
}
AXIS_COLORS = {"sex": "#1f77b4", "race": "#d62728", "age": "#2ca02c"}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="MIMIC HyperAdapt (sex/race/age) 的 Δθ 权重轨迹可视化（无需图像）"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default=str(
            "/vol/biomedic2/bglocker_studproj/zc125/outputs/mimic_cxr/cv5/"
            "hyperadapt_lr1e-04_wd1e-03_seed42_best_overall.pth"
        ),
        help="MIMIC HyperAdapt checkpoint（默认 oof-regime fold0=seed42 best_overall）",
    )
    parser.add_argument(
        "--out_dir", type=str,
        default=str(
            "/vol/biomedic2/bglocker_studproj/zc125/outputs/analysis/"
            "hyperadapt_trajectory/mimic_fold0"
        ),
        help="产物输出目录",
    )
    parser.add_argument(
        "--steps_per_axis", type=int, default=20,
        help="每条属性轴探针的插值步数",
    )
    return parser.parse_args()


def subgroup_label(combo: tuple[int, int, int]) -> str:
    """(sex, race, age) 三位二值 → 缩写标签，如 (0,1,0)->'M-N-y'。"""
    return "-".join(ATTR_LEVEL_LABELS[ATTR_NAMES[i]][combo[i]] for i in range(3))


@torch.no_grad()
def fused_from_cat(
    model: ResNet18HyperAdapt,
    s_vec: torch.Tensor, r_vec: torch.Tensor, a_vec: torch.Tensor,
) -> torch.Tensor:
    """
    由三个 16 维 cat-embedding 直接拼接并过 fuse MLP，得到 128 维 profile。
    绕开 nn.Embedding 的整数索引，从而能对某属性做连续插值。

    Args:
        model: MIMIC HyperAdapt（PatientEmbedding 三属性通路）。
        s_vec, r_vec, a_vec: 各 [16] 的 sex/race/age cat-embedding。

    Returns:
        [128] fused profile 向量。
    """
    cat = torch.cat([s_vec, r_vec, a_vec], dim=-1).unsqueeze(0)  # [1, 48]
    return model.patient_embed.fuse(cat).squeeze(0)             # [128]


@torch.no_grad()
def build_anchors_and_probes(
    model: ResNet18HyperAdapt, steps_per_axis: int
) -> tuple[list[torch.Tensor], list[tuple[int, int, int]],
           dict[str, tuple[np.ndarray, list[torch.Tensor]]]]:
    """
    构造 8 个子群锚点 profile，以及三条属性轴探针（每属性在 level0→level1 插值，
    另两属性固定 level0）。

    Returns:
        anchor_profiles: 长度 8 的 profile 向量列表。
        anchor_combos  : 对应的 (sex,race,age) 组合列表。
        probes         : {attr: (ts[N], profile 列表)}，ts∈[0,1] 为该轴插值坐标。
    """
    sex_tab = model.patient_embed.sex_embed.weight.detach()   # [2, 16]
    race_tab = model.patient_embed.race_embed.weight.detach()
    age_tab = model.patient_embed.age_embed.weight.detach()
    tabs = {"sex": sex_tab, "race": race_tab, "age": age_tab}

    # 8 子群锚点
    anchor_profiles: list[torch.Tensor] = []
    anchor_combos: list[tuple[int, int, int]] = []
    for combo in itertools.product((0, 1), repeat=3):
        s, r, a = combo
        prof = fused_from_cat(model, sex_tab[s], race_tab[r], age_tab[a])
        anchor_profiles.append(prof)
        anchor_combos.append(combo)

    # 三条轴向探针（另两属性固定 level0）
    probes: dict[str, tuple[np.ndarray, list[torch.Tensor]]] = {}
    ref = {"sex": sex_tab[0], "race": race_tab[0], "age": age_tab[0]}
    for attr in ATTR_NAMES:
        ts: list[float] = []
        profs: list[torch.Tensor] = []
        for j in range(steps_per_axis + 1):
            frac = j / steps_per_axis
            v = (1.0 - frac) * tabs[attr][0] + frac * tabs[attr][1]
            vecs = dict(ref)
            vecs[attr] = v
            profs.append(fused_from_cat(model, vecs["sex"], vecs["race"], vecs["age"]))
            ts.append(frac)
        probes[attr] = (np.asarray(ts), profs)
    return anchor_profiles, anchor_combos, probes


def plot_trajectory(
    proj_anchor: np.ndarray, anchor_combos: list[tuple[int, int, int]],
    proj_probes: dict[str, np.ndarray], evr: np.ndarray, out_path: Path,
) -> None:
    """画 PCA-2D：8 子群锚点（标签）+ 3 条属性轴探针曲线。"""
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    # 三条轴探针
    for attr, pp in proj_probes.items():
        ax.plot(pp[:, 0], pp[:, 1], "-", color=AXIS_COLORS[attr], lw=1.8,
                alpha=0.8, label=f"{attr} probe (0->1)", zorder=1)
    # 8 锚点
    ax.scatter(proj_anchor[:, 0], proj_anchor[:, 1], s=120, marker="o",
               edgecolor="k", facecolor="none", linewidths=1.4, zorder=3)
    for i, combo in enumerate(anchor_combos):
        ax.annotate(f" {subgroup_label(combo)}", (proj_anchor[i, 0], proj_anchor[i, 1]),
                    fontsize=9, fontweight="bold", zorder=4)
    ax.set_xlabel(f"PC1 (explained var {evr[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 (explained var {evr[1] * 100:.1f}%)")
    ax.set_title("MIMIC HyperAdapt: weight-space trajectory of dtheta(sex,race,age)\n"
                 "(circles = 8 subgroups S-R-A; lines = per-attribute 0->1 probes)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_layer(
    layer_norms_per_sub: list[dict[str, float]],
    anchor_combos: list[tuple[int, int, int]], out_path: Path,
) -> list[str]:
    """画每层 ‖M‖（fc 为相对 ‖ΔW‖/‖W_fc‖）随 8 个子群的折线，按 backbone 深度排序。"""
    layer_names = list(layer_norms_per_sub[0].keys())

    def sort_key(name: str) -> tuple[int, int, int]:
        if name == "fc":
            return (5, 0, 0)
        stage = int(name[5])
        blk = int(name.split(".")[1])
        conv_order = {"conv1": 0, "conv2": 1, "downsample": 2}[name.split(".")[2]]
        return (stage, blk, conv_order)

    ordered = sorted(layer_names, key=sort_key)
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(ordered))
    cmap = plt.get_cmap("tab10")
    for si in range(len(anchor_combos)):
        ys = [layer_norms_per_sub[si][name] for name in ordered]
        ax.plot(x, ys, "-o", ms=3, color=cmap(si % 10),
                label=subgroup_label(anchor_combos[si]))
    ax.set_xticks(x)
    ax.set_xticklabels(ordered, rotation=90, fontsize=7)
    ax.set_ylabel("relative weight change ||dtheta||/||theta|| (conv: ||W*M||/||W||; fc: ||dW||/||W_fc||)")
    ax.set_title("MIMIC HyperAdapt per-layer RELATIVE weight change across 8 subgroups\n"
                 "(same scale for conv & fc; fc dominates -- consistent with conditioning ablation)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return ordered


def compute_marginal_displacement(
    vec_by_combo: dict[tuple[int, int, int], np.ndarray],
) -> dict[str, float]:
    """
    逐属性边际位移：Δ_attr = 对另两属性 4 种取值求平均的 ‖Δθ(attr=1,·)-Δθ(attr=0,·)‖。

    Args:
        vec_by_combo: {(sex,race,age): 全 Δθ 向量}。

    Returns:
        {attr: 平均全 Δθ 位移范数}。三者都≈0 ⇒ HN 基本没条件化。
    """
    out: dict[str, float] = {}
    for ai, attr in enumerate(ATTR_NAMES):
        dists = []
        others = [i for i in range(3) if i != ai]
        for ov in itertools.product((0, 1), repeat=2):
            c0 = [0, 0, 0]
            c0[others[0]], c0[others[1]] = ov
            c1 = list(c0)
            c0[ai], c1[ai] = 0, 1
            dists.append(float(np.linalg.norm(
                vec_by_combo[tuple(c1)] - vec_by_combo[tuple(c0)]
            )))
        out[attr] = float(np.mean(dists))
    return out


def main() -> None:
    """主流程：载入 checkpoint → 8 子群+3 轴探针 → 轨迹/逐层图 + 标量 JSON。"""
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 载入三属性 HyperAdapt（pretrained=False；权重由 checkpoint 覆盖）
    model = ResNet18HyperAdapt(num_classes=1, pretrained=False)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] missing={list(missing)[:4]}... unexpected={list(unexpected)[:4]}...")
    model.eval()

    # 1. 8 锚点 + 3 轴探针 profile
    anchor_profiles, anchor_combos, probes = build_anchors_and_probes(
        model, args.steps_per_axis
    )

    # 2. 每个 profile → 权重向量 + 逐层 norm
    anchor_vecs: list[np.ndarray] = []
    layer_norms_per_sub: list[dict[str, float]] = []
    for prof in anchor_profiles:
        vec, lnorm = delta_theta_from_profile(model, prof)
        anchor_vecs.append(vec)
        layer_norms_per_sub.append(lnorm)
    anchor_arr = np.stack(anchor_vecs, axis=0)  # [8, D]
    vec_by_combo = {anchor_combos[i]: anchor_vecs[i] for i in range(len(anchor_combos))}

    probe_vecs: dict[str, np.ndarray] = {}
    probe_ts: dict[str, np.ndarray] = {}
    for attr, (ts, profs) in probes.items():
        vs = np.stack([delta_theta_from_profile(model, p)[0] for p in profs], axis=0)
        probe_vecs[attr] = vs
        probe_ts[attr] = ts

    # 3. PCA（在 8 锚点 + 全部探针点上联合拟合，保证同一坐标系）
    all_pts = np.concatenate([anchor_arr] + [probe_vecs[a] for a in ATTR_NAMES], axis=0)
    proj_all, evr = pca_2d(all_pts)
    n_anchor = anchor_arr.shape[0]
    proj_anchor = proj_all[:n_anchor]
    proj_probes: dict[str, np.ndarray] = {}
    cur = n_anchor
    for attr in ATTR_NAMES:
        m = probe_vecs[attr].shape[0]
        proj_probes[attr] = proj_all[cur:cur + m]
        cur += m

    plot_trajectory(proj_anchor, anchor_combos, proj_probes, evr,
                    out_dir / "trajectory_pca.png")
    ordered = plot_per_layer(layer_norms_per_sub, anchor_combos,
                             out_dir / "per_subgroup_modulation.png")

    # 4. 标量
    marginal = compute_marginal_displacement(vec_by_combo)
    # 逐层 mean_abs / spread（8 子群上）
    per_layer = {}
    for name in ordered:
        vals = np.array([layer_norms_per_sub[si][name] for si in range(n_anchor)])
        per_layer[name] = {
            "mean_abs": float(vals.mean()),
            "spread_max_minus_min": float(vals.max() - vals.min()),
            "rel_spread": float((vals.max() - vals.min()) / (vals.mean() + 1e-12)),
            "per_subgroup": {subgroup_label(anchor_combos[si]): float(vals[si])
                             for si in range(n_anchor)},
        }
    # 8 锚点两两全 Δθ 距离
    dtheta_dist = np.linalg.norm(
        anchor_arr[:, None, :] - anchor_arr[None, :, :], axis=-1
    )
    # Δθ 相对 backbone 卷积权重总范数
    backbone_sq = 0.0
    for pname, p in model.named_parameters():
        if pname.startswith(("conv1", "bn1", "layer")) and (
            ".conv" in pname or pname.startswith("conv1") or ".downsample.0" in pname
        ):
            backbone_sq += float((p.detach() ** 2).sum())
    backbone_norm = float(backbone_sq ** 0.5)
    mean_dtheta_norm = float(np.linalg.norm(anchor_arr, axis=1).mean())
    # embedding 距离（8 fused profile）
    prof_arr = torch.stack(anchor_profiles).numpy()
    prof_dist = np.linalg.norm(prof_arr[:, None, :] - prof_arr[None, :, :], axis=-1)
    # 逐属性 cat-embedding 两 level 距离
    raw_attr_sep = {
        attr: float(torch.linalg.norm(
            getattr(model.patient_embed, f"{attr}_embed").weight[1]
            - getattr(model.patient_embed, f"{attr}_embed").weight[0]
        ).item())
        for attr in ATTR_NAMES
    }

    scalars = {
        "checkpoint": args.checkpoint,
        "subgroup_order": [subgroup_label(c) for c in anchor_combos],
        "marginal_displacement_full_dtheta": marginal,
        "mean_dtheta_norm_over_subgroups": mean_dtheta_norm,
        "backbone_conv_weight_norm": backbone_norm,
        "mean_dtheta_over_backbone_ratio": mean_dtheta_norm / (backbone_norm + 1e-12),
        "dtheta_pairwise_distance_matrix": dtheta_dist.tolist(),
        "per_layer": per_layer,
        "profile_embedding_pairwise_distance": prof_dist.tolist(),
        "raw_cat_embedding_level_separation": raw_attr_sep,
        "pca_explained_variance_ratio": evr.tolist(),
        "dtheta_vector_dim": int(anchor_arr.shape[1]),
    }
    with open(out_dir / "scalars.json", "w") as f:
        json.dump(scalars, f, indent=2, ensure_ascii=False)

    # 5. 终端摘要
    print("\n=== MIMIC HyperAdapt 三属性条件化轨迹诊断 ===")
    print(f"checkpoint : {Path(args.checkpoint).name}")
    print(f"PCA 方差解释: PC1 {evr[0]*100:.1f}% / PC2 {evr[1]*100:.1f}%")
    print(f"Δθ/backbone : {scalars['mean_dtheta_over_backbone_ratio']:.3e}")
    print("逐属性边际位移 (全 Δθ 空间; 三者都≈0 ⇒ 基本没条件化):")
    for attr in ATTR_NAMES:
        print(f"  Δ_{attr:4s} = {marginal[attr]:.4g}   "
              f"(cat-embed level 间距 {raw_attr_sep[attr]:.3f})")
    print("\n逐层调制 (mean_abs=是否启用; rel_spread=是否随子群变化):")
    for name in ordered:
        pl = scalars["per_layer"][name]
        print(f"  {name:22s} mean|M|={pl['mean_abs']:.4e}  "
              f"spread={pl['spread_max_minus_min']:.4e}  rel_spread={pl['rel_spread']:.3f}")
    print("\n8 子群 fused profile 距离矩阵 (坍缩=全≈0):")
    print("  order:", scalars["subgroup_order"])
    for row in prof_dist:
        print("  " + "  ".join(f"{v:5.2f}" for v in row))
    print(f"\n产物 → {out_dir}")


if __name__ == "__main__":
    main()
