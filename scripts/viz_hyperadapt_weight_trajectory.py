"""
HyperAdapt 属性条件化可视化：age embedding 连续扫掠 → Δθ 权重轨迹
=================================================================
理念对齐 Li et al. 2018 "Visualizing the Loss Landscape of Neural Nets"
(arXiv:1712.09913) 的低维路径插值 + 投影思想：这里**不插值 loss**，而是沿
age 轴连续插值超网络的条件输入 embedding，观察超网络输出的调制量 Δθ(age) 在
权重空间里画出的移动轨迹，从而直接回答「HN 是否真的条件化了敏感属性」。

关键性质（使本诊断极其廉价、无需任何图像）：
    HyperAdapt 的 Δθ 只依赖敏感属性 —— 卷积调制 M = A(emb)·B(emb)、
    fc 加性更新 ΔW = A_fc(emb)·B_fc(emb) 都是 embedding 的函数，与图像 x 无关。
    因此同一 age 组的所有样本共享同一组 Δθ，整套条件化几何完全由 checkpoint +
    age 输入决定 ⇒ 纯 CPU 前向、秒级完成。

诊断逻辑（把「null 结果」拆成两种机制）：
    · 逐层 ‖M_ℓ(a)‖ ≈ 0（所有 age）      → adapter 未被启用，退化成纯 baseline。
    · ‖M_ℓ(a)‖ > 0 但随 a 几乎不变        → 学到了 age 无关的固定偏移，非条件化。
    · ‖M_ℓ(a)‖ > 0 且随 a 明显移动        → 确实在按 age 条件化（不论最终是否帮到公平）。

对 age（序数属性 20-40<40-60<60-80<80+）沿 0→1→2→3 连续插值 16 维 age embedding，
经 fuse MLP → 128 维 profile → 各生成器，得到权重空间里的一条曲线；4 个整数点为
真实 age 组锚点，锚点之间为探针（无物理含义，只探测映射的光滑变化幅度与方向）。

产物写入 outputs/analysis/hyperadapt_trajectory/<tag>/：
    1. trajectory_pca.png       全 Δθ 向量 PCA-2D 轨迹（4 锚点高亮）
    2. per_layer_modulation.png 每层 ‖M‖（相对权重变化）随 4 个 age 组
    3. scalars.json             弧长 / 端点位移 / 逐层「绝对 vs 随属性变化」/ fc / embedding 距离
"""

from __future__ import annotations

import argparse
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

from src.models.resnet18_hyperadapt import ResNet18HyperAdaptAge  # noqa: E402

# HAM age 组的人类可读标签（对齐 build_ham10000_splits.py 的 20 岁分箱 0..3）
AGE_LABELS = ["20-40", "40-60", "60-80", "80+"]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="HyperAdapt age 条件化的 Δθ 权重轨迹可视化（无需图像）"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default=str(
            "/vol/biomedic2/bglocker_studproj/zc125/outputs/ham10000/cv5/"
            "hyperadapt_lr3e-05_wd1e-04_seed42_best_overall.pth"
        ),
        help="HyperAdapt(age) checkpoint（默认 HAM oof-regime fold0=seed42 best_overall）",
    )
    parser.add_argument(
        "--out_dir", type=str,
        default=str(
            "/vol/biomedic2/bglocker_studproj/zc125/outputs/analysis/"
            "hyperadapt_trajectory/ham_fold0"
        ),
        help="产物输出目录",
    )
    parser.add_argument(
        "--num_age", type=int, default=4, help="age 组数（HAM=4）",
    )
    parser.add_argument(
        "--steps_per_segment", type=int, default=20,
        help="相邻 age 组之间的插值步数（越大轨迹越平滑）",
    )
    return parser.parse_args()


@torch.no_grad()
def build_profile_sweep(
    model: ResNet18HyperAdaptAge, num_age: int, steps_per_segment: int
) -> tuple[torch.Tensor, np.ndarray, list[int]]:
    """
    沿 age 轴 0→1→...→(num_age-1) 连续插值 16 维 age embedding，经 fuse MLP 得到
    128 维 profile 向量序列。

    Args:
        model            : 已载入权重的 HyperAdapt(age) 模型（eval）。
        num_age          : age 组数。
        steps_per_segment: 每段（相邻两组之间）插值步数。

    Returns:
        profiles : [N, 128] 每个扫掠点的 fused profile 向量。
        ts       : [N] 每点在 age 轴上的连续坐标（0..num_age-1）。
        anchor_idx: 长度 num_age 的列表，profiles 中对应真实 age 组的行下标。
    """
    age_vecs = model.patient_embed.age_embed.weight.detach()  # [num_age, 16]
    profiles: list[torch.Tensor] = []
    ts: list[float] = []
    anchor_idx: list[int] = []

    for seg in range(num_age - 1):
        # 每段采样 steps_per_segment 个点；除最后一段外不含右端点，避免锚点重复计数
        n = steps_per_segment + (1 if seg == num_age - 2 else 0)
        for j in range(n):
            frac = j / steps_per_segment
            t = seg + frac
            v16 = (1.0 - frac) * age_vecs[seg] + frac * age_vecs[seg + 1]  # [16]
            emb = model.patient_embed.fuse(v16.unsqueeze(0))  # [1, 128]
            if abs(t - round(t)) < 1e-9:  # 命中整数 = 真实 age 组锚点
                anchor_idx.append(len(profiles))
            profiles.append(emb.squeeze(0))
            ts.append(t)
    return torch.stack(profiles, dim=0), np.asarray(ts), anchor_idx


@torch.no_grad()
def delta_theta_from_profile(
    model: ResNet18HyperAdaptAge, emb: torch.Tensor
) -> tuple[np.ndarray, dict[str, float]]:
    """
    对单个 profile 向量，复算 HyperAdapt 前向里所有卷积层的**实际权重偏移** Δθ 与 fc 的
    加性更新 ΔW，拼成一个「权重空间向量」，并返回逐层**相对权重变化** ‖Δθ‖/‖θ‖。

    ⚠️ 归一化口径（2026-07-26 修正）：conv 的乘性调制 Θ'=Θ·(1+M)，实际权重偏移是 Δθ=Θ⊙M，
    而非 M 本身。早期版本用原始 ‖M‖_F 作逐层强度，它在 C_out×C_in 上求和、被层尺寸严重放大
    （layer4 达 26 万项），与 fc 的相对量 ‖ΔW‖/‖W_fc‖ **不可比**，会假性显示「conv 主导」。
    现统一用**相对权重变化** conv=‖Θ⊙M‖_F/‖Θ‖_F、fc=‖ΔW‖/‖W_fc‖（同一把尺子），修正后
    fc 远主导（与 ResNet18CondNet 条件消融的 fc/conv≈48–63× 一致）。

    PCA 轨迹坐标仍用轻量的调制 M（conv）+ ΔW（fc）拼接：实际偏移 Θ⊙M 含 K×K 维度会使向量
    暴涨到 ~11M 而在轻量机上 OOM；调制空间坐标足以刻画子群间轨迹几何（本函数只用它做 PCA
    的相对位置，不用它比 conv/fc 强度——强度由 layer_norms 给）。

    Args:
        model: HyperAdapt(age) 模型。
        emb  : [128] 单个 profile 向量。

    Returns:
        vec        : 1-D numpy，拼接所有层调制 M（flatten）+ fc ΔW（flatten），作 PCA 坐标。
        layer_norms: {层名: 相对权重变化 ‖Δθ‖/‖θ‖}，conv/fc 同一量纲，供逐层曲线。
    """
    emb = emb.unsqueeze(0)  # [1, 128]
    batch = 1
    rank = model.rank
    parts: list[np.ndarray] = []
    layer_norms: dict[str, float] = {}

    # 预生成各 stage 的 shared-A（与 forward 一致）
    a_shared = {
        c: model.A_gen[f"c{c}"](emb).view(batch, c, rank)
        for c in model.STAGE_CHANNELS
    }
    stage_channels = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}

    for lname in ("layer1", "layer2", "layer3", "layer4"):
        c_out = stage_channels[lname]
        for bi, blk in enumerate(getattr(model, lname)):
            a_c = a_shared[c_out]
            # conv1: 输入通道 = blk.in_channels；conv2: 输入通道 = blk.out_channels
            m1 = torch.bmm(a_c, blk.B_gen_conv1(emb).view(batch, rank, blk.in_channels))
            m2 = torch.bmm(a_c, blk.B_gen_conv2(emb).view(batch, rank, blk.out_channels))
            convs = [(f"{lname}.{bi}.conv1", m1, blk.conv1.weight),
                     (f"{lname}.{bi}.conv2", m2, blk.conv2.weight)]
            if blk.B_gen_down is not None:  # downsample 也带 adapter
                md = torch.bmm(a_c, blk.B_gen_down(emb).view(batch, rank, blk.in_channels))
                convs.append((f"{lname}.{bi}.downsample", md, blk.downsample[0].weight))
            for tag, m, w in convs:
                m2d = m.squeeze(0)  # [C_out, C_in]
                # PCA 轨迹坐标：用轻量的调制 M（否则 Θ⊙M 含 K×K 维度会使向量暴涨到 ~11M 而 OOM）。
                parts.append(m2d.flatten().numpy())
                # 逐层强度：用**相对权重变化** ‖Δθ‖/‖Θ‖，Δθ=Θ⊙M（M 广播到 K×K），与 fc 同量纲。
                # 因 M 对同一 (out,in) 通道对在 K×K 上是常数：‖Θ⊙M‖² = Σ_{o,i} M² · ‖Θ_{o,i}‖²
                w_chan_sq = (w ** 2).sum(dim=(2, 3))  # [C_out, C_in] 每通道对的核范数²
                d_theta_norm = torch.sqrt((m2d ** 2 * w_chan_sq).sum())
                layer_norms[tag] = float(d_theta_norm / (torch.linalg.norm(w) + 1e-12))

    # fc 加性低秩更新 ΔW = A_fc · B_fc，相对量 ‖ΔW‖/‖W_fc‖
    a_fc = model.fc_A_gen(emb).view(batch, model.out_dim, rank)
    b_fc = model.fc_B_gen(emb).view(batch, rank, 512)
    delta_w = torch.bmm(a_fc, b_fc).squeeze(0)  # [out_dim, 512]
    parts.append(delta_w.flatten().numpy())
    fc_w_norm = float(torch.linalg.norm(model.fc.weight).item())
    layer_norms["fc"] = float(torch.linalg.norm(delta_w).item()) / (fc_w_norm + 1e-12)

    return np.concatenate(parts), layer_norms


def pca_2d(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    对 [N, D] 扫掠向量做 2D PCA（居中 + SVD），返回 2D 投影与前两主成分方差解释率。

    Args:
        vectors: [N, D] 每行一个扫掠点的权重空间向量。

    Returns:
        proj  : [N, 2] 前两主成分坐标。
        evr   : [2] 前两主成分的方差解释率。
    """
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    # 经济 SVD；N 远小于 D，直接对 centered 分解
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    proj = u[:, :2] * s[:2]
    total_var = float((s ** 2).sum()) + 1e-12
    evr = (s[:2] ** 2) / total_var
    return proj, evr


def plot_trajectory(
    proj: np.ndarray, evr: np.ndarray, ts: np.ndarray, anchor_idx: list[int],
    out_path: Path,
) -> None:
    """画 PCA-2D 权重轨迹：连续曲线 + 4 个真实 age 组锚点高亮。"""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(proj[:, 0], proj[:, 1], "-", color="#888", lw=1.2, zorder=1,
            label="interpolation probe")
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=ts, cmap="viridis", s=18, zorder=2)
    for k, idx in enumerate(anchor_idx):
        ax.scatter(proj[idx, 0], proj[idx, 1], s=180, marker="*",
                   edgecolor="k", facecolor="crimson", zorder=3)
        ax.annotate(f"  age={AGE_LABELS[k]}", (proj[idx, 0], proj[idx, 1]),
                    fontsize=11, fontweight="bold")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("age-axis continuous coord t (0->3)")
    ax.set_xlabel(f"PC1 (explained var {evr[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 (explained var {evr[1] * 100:.1f}%)")
    ax.set_title("HyperAdapt: weight-space trajectory of dtheta(age)\n"
                 "(star = real age group; curve = embedding interpolation)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_layer(
    layer_norms_per_age: list[dict[str, float]], out_path: Path,
) -> list[str]:
    """
    画每层 ‖M‖（fc 为相对 ‖ΔW‖/‖W_fc‖）随 4 个真实 age 组的变化折线，按 backbone
    深度排序，直观看哪些层被启用、是否随 age 移动。

    Args:
        layer_norms_per_age: 长度=age 组数的列表，每项是 {层名: norm}。
        out_path           : 图片保存路径。

    Returns:
        ordered_layers: 排序后的层名列表（供 scalars 复用同一顺序）。
    """
    layer_names = list(layer_norms_per_age[0].keys())

    def sort_key(name: str) -> tuple[int, int, int]:
        if name == "fc":
            return (5, 0, 0)
        stage = int(name[5])  # layer{X}
        blk = int(name.split(".")[1])
        conv_order = {"conv1": 0, "conv2": 1, "downsample": 2}[name.split(".")[2]]
        return (stage, blk, conv_order)

    ordered = sorted(layer_names, key=sort_key)
    n_age = len(layer_norms_per_age)

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(ordered))
    cmap = plt.get_cmap("plasma")
    for a in range(n_age):
        ys = [layer_norms_per_age[a][name] for name in ordered]
        ax.plot(x, ys, "-o", ms=4, color=cmap(a / max(1, n_age - 1)),
                label=f"age {AGE_LABELS[a]}")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered, rotation=90, fontsize=7)
    ax.set_ylabel("relative weight change ||dtheta||/||theta|| (conv: ||W*M||/||W||; fc: ||dW||/||W_fc||)")
    ax.set_title("HyperAdapt per-layer RELATIVE weight change across age groups\n"
                 "(same scale for conv & fc; fc dominates -- consistent with conditioning ablation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return ordered


def compute_scalars(
    vectors: np.ndarray, anchor_idx: list[int],
    layer_norms_per_age: list[dict[str, float]], ordered_layers: list[str],
    profiles: torch.Tensor, model: ResNet18HyperAdaptAge, num_age: int,
) -> dict:
    """
    汇总量化标量：轨迹弧长、端点位移、逐层「绝对调制 vs 随 age 变化」、embedding 距离。

    Args:
        vectors            : [N, D] 全部扫掠点的权重空间向量。
        anchor_idx         : 真实 age 组在 vectors 中的行下标。
        layer_norms_per_age: 各 age 组的逐层 norm。
        ordered_layers     : 排序层名。
        profiles           : [N, 128] fused profile（供 embedding 距离；仅用锚点行）。
        model              : 模型（取 age_embed 原始 16 维向量）。
        num_age            : age 组数。

    Returns:
        标量字典（可 JSON 序列化）。
    """
    # 1. 轨迹弧长（全 Δθ 空间）与端点位移
    seg = np.linalg.norm(np.diff(vectors, axis=0), axis=1)
    arc_len = float(seg.sum())
    endpoint_disp = float(np.linalg.norm(vectors[anchor_idx[-1]] - vectors[anchor_idx[0]]))
    # 锚点两两之间的全 Δθ 距离矩阵
    anchor_vecs = vectors[anchor_idx]
    dtheta_dist = np.linalg.norm(
        anchor_vecs[:, None, :] - anchor_vecs[None, :, :], axis=-1
    )

    # 2. 逐层：绝对调制（是否启用）vs 随 age 变化（是否条件化）
    per_layer = {}
    for name in ordered_layers:
        vals = np.array([layer_norms_per_age[a][name] for a in range(num_age)])
        per_layer[name] = {
            "mean_abs": float(vals.mean()),          # 平均调制幅度（≈0 ⇒ 未启用）
            "spread_max_minus_min": float(vals.max() - vals.min()),  # 随 age 波动
            "rel_spread": float((vals.max() - vals.min()) / (vals.mean() + 1e-12)),
            "per_age": {AGE_LABELS[a]: float(vals[a]) for a in range(num_age)},
        }

    # 3. embedding 距离（检查是否坍缩）：128 维 fused profile & 16 维原始 age embedding
    prof_anchor = profiles[anchor_idx].numpy()
    prof_dist = np.linalg.norm(
        prof_anchor[:, None, :] - prof_anchor[None, :, :], axis=-1
    )
    raw_age = model.patient_embed.age_embed.weight.detach().numpy()  # [num_age, 16]
    raw_dist = np.linalg.norm(raw_age[:, None, :] - raw_age[None, :, :], axis=-1)

    # 4. 全 Δθ 相对 backbone 权重总范数（整体适配是否微小）
    backbone_sq = 0.0
    for pname, p in model.named_parameters():
        if pname.startswith(("conv1", "bn1", "layer")) and (
            ".conv" in pname or pname.startswith("conv1") or ".downsample.0" in pname
        ):
            backbone_sq += float((p.detach() ** 2).sum())
    backbone_norm = float(backbone_sq ** 0.5)
    mean_dtheta_norm = float(np.linalg.norm(anchor_vecs, axis=1).mean())

    return {
        "trajectory_arc_length_full_dtheta": arc_len,
        "endpoint_displacement_full_dtheta": endpoint_disp,
        "dtheta_pairwise_distance_matrix": dtheta_dist.tolist(),
        "mean_dtheta_norm_over_ages": mean_dtheta_norm,
        "backbone_conv_weight_norm": backbone_norm,
        "mean_dtheta_over_backbone_ratio": mean_dtheta_norm / (backbone_norm + 1e-12),
        "per_layer": per_layer,
        "profile_embedding_pairwise_distance": prof_dist.tolist(),
        "raw_age_embedding_pairwise_distance": raw_dist.tolist(),
        "age_labels": AGE_LABELS[:num_age],
    }


def main() -> None:
    """主流程：载入 checkpoint → age 扫掠 → 轨迹/逐层图 + 标量 JSON。"""
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 载入模型（pretrained=False 避免联网；权重全部由 checkpoint 覆盖）
    model = ResNet18HyperAdaptAge(num_classes=1, num_age=args.num_age, pretrained=False)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] missing={list(missing)[:4]}... unexpected={list(unexpected)[:4]}...")
    model.eval()

    # 1. age 轴连续扫掠 → profile 序列
    profiles, ts, anchor_idx = build_profile_sweep(
        model, args.num_age, args.steps_per_segment
    )
    assert len(anchor_idx) == args.num_age, f"锚点数 {len(anchor_idx)} != {args.num_age}"

    # 2. 每个扫掠点 → 权重空间向量 + 逐层 norm
    vectors: list[np.ndarray] = []
    layer_norms_all: list[dict[str, float]] = []
    for i in range(profiles.shape[0]):
        vec, lnorm = delta_theta_from_profile(model, profiles[i])
        vectors.append(vec)
        layer_norms_all.append(lnorm)
    vectors_arr = np.stack(vectors, axis=0)  # [N, D]
    layer_norms_per_age = [layer_norms_all[idx] for idx in anchor_idx]

    # 3. PCA-2D 轨迹图
    proj, evr = pca_2d(vectors_arr)
    plot_trajectory(proj, evr, ts, anchor_idx, out_dir / "trajectory_pca.png")

    # 4. 逐层调制强度图
    ordered = plot_per_layer(layer_norms_per_age, out_dir / "per_layer_modulation.png")

    # 5. 量化标量
    scalars = compute_scalars(
        vectors_arr, anchor_idx, layer_norms_per_age, ordered,
        profiles, model, args.num_age,
    )
    scalars["checkpoint"] = args.checkpoint
    scalars["pca_explained_variance_ratio"] = evr.tolist()
    scalars["n_sweep_points"] = int(vectors_arr.shape[0])
    scalars["dtheta_vector_dim"] = int(vectors_arr.shape[1])
    with open(out_dir / "scalars.json", "w") as f:
        json.dump(scalars, f, indent=2, ensure_ascii=False)

    # 6. 终端摘要
    print(f"\n=== HyperAdapt age 条件化轨迹诊断 ===")
    print(f"checkpoint : {Path(args.checkpoint).name}")
    print(f"扫掠点数    : {scalars['n_sweep_points']}  |  Δθ 维度 {scalars['dtheta_vector_dim']:,}")
    print(f"PCA 方差解释: PC1 {evr[0]*100:.1f}% / PC2 {evr[1]*100:.1f}%")
    print(f"轨迹弧长    : {scalars['trajectory_arc_length_full_dtheta']:.4g}")
    print(f"端点位移    : {scalars['endpoint_displacement_full_dtheta']:.4g}")
    print(f"Δθ/backbone : {scalars['mean_dtheta_over_backbone_ratio']:.3e} "
          f"(整体适配相对 backbone 权重的量级)")
    print("\n逐层调制 (mean_abs = 是否启用; rel_spread = 是否随 age 变化):")
    for name in ordered:
        pl = scalars["per_layer"][name]
        print(f"  {name:22s} mean|M|={pl['mean_abs']:.4e}  "
              f"spread={pl['spread_max_minus_min']:.4e}  rel_spread={pl['rel_spread']:.3f}")
    print("\nfused profile 锚点距离矩阵 (坍缩=全≈0):")
    for row in scalars["profile_embedding_pairwise_distance"]:
        print("  " + "  ".join(f"{v:6.3f}" for v in row))
    print(f"\n产物 → {out_dir}")


if __name__ == "__main__":
    main()
