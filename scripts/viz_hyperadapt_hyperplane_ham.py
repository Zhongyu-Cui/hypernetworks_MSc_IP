"""
HyperAdapt(age) 分类超平面的逐子群可视化（HAM10000）
=====================================================
目标：看清 HyperAdapt 在不同 age 子群条件下，如何调整 512 维特征空间里的**分类超平面**。

为什么不能只画 w_eff(a)
-----------------------
HyperAdapt 的条件化有两条通路（见 src/models/resnet18_hyperadapt.py:568-621）：
  · conv 通路：patient embedding 逐层对卷积核做 channel-wise 乘性调制 ⇒ **特征 f(x,a) 本身随 a 变**；
  · fc  通路：w_eff(a) = W + ΔW(a)，ΔW = fc_A_gen(emb) @ fc_B_gen(emb)（加性低秩）。
  · 注意 fc.bias **不随 a 变** ⇒ 任何「阈值平移」只能由 w 与特征分布的交互产生，不可能来自 bias。
只比较各组 w_eff(a) 的夹角/范数会完全漏掉 conv 通路（MIMIC 上 conv 造成的特征位移虽小，
但 Δlogit 幅度口径下反而占 74%，见 docs/hyperplane_subgroup_visualisation.md §4），
故本脚本把两条通路一起纳入，并且**点云与决策线必须成对呈现**：特征云与超平面同时移动，存在
规范自由度（gauge freedom），单看线的位移没有意义，唯一有物理含义的量是「点云相对于线的位置」。

方法（三步）
------------
1. **反事实属性扫掠**：固定同一批图像 X（fold 的 held-out test），对每个 age g∈{0..3} 各前向一遍，
   得到 F_g = f(X, g) ∈ R^{N×512} 与 w_g = w_eff(g) ∈ R^512。病人固定 ⇒ 组间差异 100% 归因于
   HN 的调制，不混杂「老人和年轻人的痣本来就不同」。
2. **两套 2D 投影基**（同一中心 mu = 所有组特征的总均值，保证 panel 间坐标可比）：
   · discriminative（推荐）：e1 = w_bar/||w_bar||（判别轴）；e2 = 把 {w_g - w_bar} 先投影到 e1 的
     正交补、再取 PCA1（deflation，非「全空间 PCA1 事后 Gram-Schmidt」，见 hyperplane_viz.py
     build_discriminative_basis）（组间调整轴）。含义明确：横轴=诊断方向，纵轴=组间调整方向。
   · pca（默认 PCA 对照）：对全部特征做 PCA 取前 2 主成分。方差最大方向常与 w 近正交，
     故预期 in-plane 能量低、组间差异被压扁——两套并排即可看出基的选择有多关键。
   PCA 是**线性**投影 ⇒ 超平面在 2D 上仍是可解析绘制的直线（无需 DeepView 的逆映射合成网格）：
       f = mu + P u  ⇒  logit(u) = <P^T w, u> + (<w, mu> + b)
3. **Δlogit 通路分解**（代数恒等式，非近似）：
       logit(x,a') - logit(x,a) = <w(a')-w(a), f(x,a)>  +  <w(a'), f(x,a')-f(x,a)>
                                  \_____ fc 头贡献 _____/    \____ conv 特征贡献 ____/
   定量回答「调整走哪条通路」，与已有权重轨迹结论（HAM fc 主导）互为独立验证。

诚实性指标（必报，随图打印并写入 JSON）
  · in-plane 能量 ||P^T w_g|| / ||w_g||：投影后还剩多少打分表信息。低则「两组线重合」不可解读为相同。
  · e2 解释的组间 w 方差比例；PCA 基的 explained_variance_ratio。

运行（medimg env，实验室机器即可，纯推理）：
    python scripts/viz_hyperadapt_hyperplane_ham.py \
        --ckpt /vol/.../outputs/ham10000/cv5/hyperadapt_lr3e-05_wd1e-04_seed42_best_overall.pth \
        --split_dir data/splits/ham10000/cv5/fold0

输出：outputs/ham10000/hyperplane_viz/ 下 PNG（主图 + Δlogit 分解）+ .npz + metrics JSON。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.datasets.ham10000_dataset import HAM10000Dataset
from src.models.resnet18_hyperadapt import ResNet18HyperAdaptAge
# 投影基 / 通路分解 / 决策线裁剪 / 通路图与 MIMIC 版共用（src/utils/hyperplane_viz.py）；
# 本脚本只保留 HAM 专用的「两套基并成一张 2x4 图」的渲染。
from src.utils.pathway_audit_report import run_pathway_audit
from src.utils.hyperplane_viz import (
    BASIS_KEYS,
    CLASS_COLORS,
    build_discriminative_basis,
    build_pca_basis,
    decompose_delta_logit,
    format_basis_diag,
    in_plane_energy,
    line_from_coeffs,
    render_pathway_figure,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "ham10000_baseline.yaml"
OUTPUTS_ROOT = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs")
DEFAULT_CKPT = (
    OUTPUTS_ROOT / "ham10000" / "cv5" / "hyperadapt_lr3e-05_wd1e-04_seed42_best_overall.pth"
)
DEFAULT_OUT_DIR = OUTPUTS_ROOT / "ham10000" / "hyperplane_viz"

NUM_AGE = 4
AGE_LABELS = {0: "20-40", 1: "40-60", 2: "60-80", 3: "80+"}
# 行标签用短版（合成图里作 ylabel，共享模块的长版塞不进侧边）；图内文字统一英文
ROW_TITLES = {
    "discriminative": "Discriminative x between-group axis\n(recommended)",
    "pca": "Default PCA\n(first two components)",
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser(
        description="HyperAdapt(age) 分类超平面逐子群可视化（反事实扫掠 + 双投影基 + 通路分解）"
    )
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                   help="HyperAdapt(age) checkpoint（plain state_dict）。默认 cv5 选中配置 seed42。")
    p.add_argument("--split_dir", type=str, default="data/splits/ham10000/cv5/fold0",
                   help="仓库相对 split 目录。默认 cv5/fold0（与 seed42 checkpoint 对应的 held-out，"
                        "避免数据泄漏）。")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--ref_age", type=int, default=1, choices=list(range(NUM_AGE)),
                   help="Δlogit 分解的参照组 a（默认 1=40-60，HAM 最大组）。")
    p.add_argument("--batch_size", type=int, default=64,
                   help="前向 batch（HyperAdapt 逐样本卷积核显存重，CPU 上可调小）。")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_points", type=int, default=900,
                   help="散点图最多绘制的样本数（仅影响绘图密度，所有统计仍用全部样本）。")
    p.add_argument("--seed", type=int, default=42, help="绘图抽样种子。")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--tag", type=str, default="ham_fold0_seed42",
                   help="输出文件名后缀，便于区分 fold/seed。")
    return p.parse_args()


# ============================================================
# 模型 / 数据
# ============================================================
def load_model(ckpt_path: Path, device: torch.device) -> ResNet18HyperAdaptAge:
    """构建 ResNet18HyperAdaptAge 并严格载入 plain state_dict，切 eval。"""
    model = ResNet18HyperAdaptAge(
        num_classes=1, num_age=NUM_AGE, rank=4, patient_embed_dim=128,
        pretrained=False, freeze_backbone=False,
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def build_eval_loader(
    split_dir: str, split: str, batch_size: int, num_workers: int,
) -> tuple[DataLoader, np.ndarray, np.ndarray]:
    """
    构建 HAM eval DataLoader（Resize(256,256)->CenterCrop(224)->ToTensor->ImageNet Normalize）。

    仅保留 age_group>=0：AgeEmbedding 不接受 -1（0-20 排除组），与训练脚本 _filter_age_valid 一致。

    Returns:
        (loader, labels, true_ages)：labels/true_ages 为过滤后按 CSV 顺序的 numpy 数组。
    """
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ev, norm = cfg["transforms"]["eval"], cfg["transforms"]["normalize"]
    transform = transforms.Compose([
        transforms.Resize(tuple(ev["resize"])),
        transforms.CenterCrop(ev["center_crop"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm["mean"], std=norm["std"]),
    ])

    csv_path = REPO_ROOT / split_dir / f"{split}.csv"
    dataset = HAM10000Dataset(csv_path=csv_path, transform=transform)
    # 用 Subset 在内存里过滤（不落临时 CSV、不改动 Dataset 内部状态）；shuffle=False 保证
    # 返回的 labels/true_ages 与前向顺序逐位对应
    keep = np.flatnonzero(dataset.age_groups >= 0)
    subset = Subset(dataset, keep.tolist())
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return loader, dataset.labels[keep].astype(int), dataset.age_groups[keep].astype(int)


# ============================================================
# 反事实扫掠：逐组抽取 512 维特征与有效超平面
# ============================================================
@torch.no_grad()
def sweep_features_and_hyperplanes(
    model: ResNet18HyperAdaptAge, loader: DataLoader, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    对同一批图像做属性反事实扫掠：每个 age g 前向一遍，取 penultimate 特征与有效超平面。

    特征通过 forward hook 抓 avgpool 输出（等价于 forward 里 torch.flatten(out, 1) 的值），
    不改动模型代码。w_eff(g) 由 fc.weight + fc_A_gen(emb) @ fc_B_gen(emb) 解析算出——同一组内
    整批 embedding 相同，故每组只需取第一行。

    Returns:
        feats  : [G, N, 512] 各组反事实特征。
        w_eff  : [G, 512]    各组有效超平面法向量（num_classes=1 ⇒ 单行）。
        bias   : float       fc.bias（HyperAdapt 不条件化 bias，全组共用）。
        logits : [G, N]      各组反事实 logit（用于校验分解恒等式与算 AUC）。
    """
    captured: list[torch.Tensor] = []

    def hook(_module, _inp, out: torch.Tensor) -> None:
        # avgpool 输出 [B, 512, 1, 1] -> flatten 成 [B, 512]，与 fc 的输入完全一致
        captured.append(torch.flatten(out, 1).detach().float().cpu())

    handle = model.avgpool.register_forward_hook(hook)

    feats_per_age, logits_per_age, w_per_age = [], [], []
    try:
        for g in range(NUM_AGE):
            captured.clear()
            logit_chunks = []
            for image, _label, _sex, _age in loader:
                image = image.to(device, non_blocking=True)
                age_t = torch.full((image.shape[0],), g, dtype=torch.long, device=device)
                logit_chunks.append(model(image, age_t).squeeze(1).detach().float().cpu())
            feats_per_age.append(torch.cat(captured).numpy())
            logits_per_age.append(torch.cat(logit_chunks).numpy())

            # 该组的有效超平面：W + ΔW(emb_g)
            age_one = torch.tensor([g], dtype=torch.long, device=device)
            emb = model.patient_embed(age_one)                              # [1, 128]
            a_fc = model.fc_A_gen(emb).view(1, model.out_dim, model.rank)   # [1, 1, k]
            b_fc = model.fc_B_gen(emb).view(1, model.rank, 512)             # [1, k, 512]
            delta_w = torch.bmm(a_fc, b_fc)[0]                              # [1, 512]
            w_per_age.append((model.fc.weight + delta_w)[0].detach().float().cpu().numpy())
    finally:
        handle.remove()

    bias = float(model.fc.bias.detach().cpu().numpy()[0])
    return (np.stack(feats_per_age), np.stack(w_per_age), bias, np.stack(logits_per_age))


def render_main_figure(
    proj: dict[str, np.ndarray], w_eff: np.ndarray, bias: float, mu: np.ndarray,
    labels: np.ndarray, true_ages: np.ndarray, aucs: np.ndarray, logits: np.ndarray,
    energies: dict[str, list[float]], basis_diag: dict[str, dict[str, float]],
    plot_idx: np.ndarray, ckpt_name: str, out_path: Path,
) -> None:
    """
    主图：2 行（两套投影基）× 4 列（age 子群）。每格 = 该组反事实点云 + 该组精确决策线
    + 全组平均超平面（虚线参照）。点云与线必须同格呈现（规范自由度）。
    """
    w_bar = w_eff.mean(axis=0)
    fig, axes = plt.subplots(len(BASIS_KEYS), NUM_AGE, figsize=(5.0 * NUM_AGE, 5.0 * len(BASIS_KEYS)))

    for row, basis in enumerate(BASIS_KEYS):
        U = proj[basis]                                     # [G, N, 2] 已投影坐标
        # 该行共用视窗（panel 间坐标可比：同一 mu、同一基）
        all_u = U.reshape(-1, 2)
        pad = 0.06 * (all_u.max(axis=0) - all_u.min(axis=0) + 1e-9)
        x_lim = (float(all_u[:, 0].min() - pad[0]), float(all_u[:, 0].max() + pad[0]))
        y_lim = (float(all_u[:, 1].min() - pad[1]), float(all_u[:, 1].max() + pad[1]))
        P = proj[f"{basis}_P"]

        for g in range(NUM_AGE):
            ax = axes[row, g]
            u_g = U[g]
            in_group = true_ages == g

            # 其他组真实样本：淡；真实属于本组的样本：黑边高亮（反事实条件 == 真实属性）
            sel_other = plot_idx[~in_group[plot_idx]]
            sel_in = plot_idx[in_group[plot_idx]]
            ax.scatter(u_g[sel_other, 0], u_g[sel_other, 1], c=CLASS_COLORS[labels[sel_other]],
                       s=16, alpha=0.30, edgecolors="none", zorder=2)
            ax.scatter(u_g[sel_in, 0], u_g[sel_in, 1], c=CLASS_COLORS[labels[sel_in]],
                       s=42, alpha=0.95, edgecolors="black", linewidths=0.7, zorder=3)

            # 该组精确决策线：<P^T w_g, u> + (<w_g, mu> + b) = 0
            seg = line_from_coeffs(P.T @ w_eff[g], float(w_eff[g] @ mu + bias), x_lim, y_lim)
            if seg is not None:
                ax.plot(seg[0], seg[1], color="black", lw=2.4, zorder=5,
                        label=f"decision line (age={g})")
            # 全组平均超平面（固定参照，四格相同）
            seg_bar = line_from_coeffs(P.T @ w_bar, float(w_bar @ mu + bias), x_lim, y_lim)
            if seg_bar is not None:
                ax.plot(seg_bar[0], seg_bar[1], color="0.45", lw=1.6, ls="--", zorder=4,
                        label="mean hyperplane")

            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
            ax.set_title(f"age_group = {g} ({AGE_LABELS[g]})   n_true={int(in_group.sum())}\n"
                         f"counterfactual AUC={aucs[g]:.4f}   mean logit={logits[g].mean():+.2f}   "
                         f"||w||={np.linalg.norm(w_eff[g]):.2f}\n"
                         f"in-plane energy={energies[basis][g]:.3f}",
                         fontsize=10)
            if g == 0:
                # 基名称留在 ylabel，诊断量放 panel 内文本框（竖排 ylabel 塞不下会重叠）
                ax.set_ylabel(ROW_TITLES[basis], fontsize=10)
                ax.text(0.025, 0.975, format_basis_diag(basis, basis_diag[basis]),
                        transform=ax.transAxes, va="top", ha="left", fontsize=8,
                        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7",
                              "boxstyle": "round,pad=0.35"}, zorder=6)
            ax.set_xticks([])
            ax.set_yticks([])

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLORS[0], markersize=9,
               label="benign (y=0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLORS[1], markersize=9,
               label="malignant (y=1)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.5", markeredgecolor="black",
               markersize=9, label="true member of this group"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.6", alpha=0.35, markersize=8,
               label="other groups (counterfactual)"),
        Line2D([0], [0], color="black", lw=2.4, label="exact decision line of this group"),
        Line2D([0], [0], color="0.45", lw=1.6, ls="--", label="mean hyperplane (fixed reference)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.012))
    fig.suptitle(
        "HAM10000 | HyperAdapt(age): how the decision hyperplane is adjusted per age subgroup "
        "(counterfactual attribute sweep)\n"
        "Same held-out images forwarded once per assumed age_group; shared centre mu = grand mean of all "
        f"conditions | ckpt={ckpt_name}\n"
        "Top row = discriminative axis x between-group adjustment axis;  bottom row = default PCA "
        "(first two components, control)\n"
        "Points and line are shown together: features and hyperplane move jointly (gauge freedom), so only "
        "the position of points RELATIVE to the line is meaningful",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.90))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


# ============================================================
# main
# ============================================================
def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  ckpt={args.ckpt.name}  split={args.split_dir}/{args.split}")

    model = load_model(args.ckpt, device)
    loader, labels, true_ages = build_eval_loader(
        args.split_dir, args.split, args.batch_size, args.num_workers)
    n_samples = len(labels)
    print(f"样本 n={n_samples}（已过滤 age_group>=0）  malignant={labels.mean():.3f}  "
          f"真实 age 分布={np.bincount(true_ages, minlength=NUM_AGE).tolist()}")

    # ---- 1. 反事实属性扫掠 ----
    print(f"反事实扫掠：{NUM_AGE} 组 × {n_samples} 张前向 ...")
    feats, w_eff, bias, logits = sweep_features_and_hyperplanes(model, loader, device)
    aucs = np.array([roc_auc_score(labels, logits[g]) for g in range(NUM_AGE)])
    print(f"fc.bias={bias:+.4f}（HyperAdapt 不条件化 bias）")
    for g in range(NUM_AGE):
        print(f"  age={g} ({AGE_LABELS[g]:>5s})  ||w_eff||={np.linalg.norm(w_eff[g]):.4f}  "
              f"反事实全体 AUC={aucs[g]:.4f}  logit 均值={logits[g].mean():+.4f}")

    # 组间几何（补充读数，不作主图）：与平均打分表的夹角
    w_bar = w_eff.mean(axis=0)
    cos_to_bar = [float(w_eff[g] @ w_bar / (np.linalg.norm(w_eff[g]) * np.linalg.norm(w_bar) + 1e-12))
                  for g in range(NUM_AGE)]
    cos_pairwise = np.array([[float(w_eff[i] @ w_eff[j] /
                                    (np.linalg.norm(w_eff[i]) * np.linalg.norm(w_eff[j]) + 1e-12))
                              for j in range(NUM_AGE)] for i in range(NUM_AGE)])
    print(f"与平均超平面的余弦: {[f'{c:.5f}' for c in cos_to_bar]}")
    print(f"组间最小余弦（最大夹角）: {cos_pairwise[np.triu_indices(NUM_AGE, 1)].min():.5f}")
    # 特征侧的条件化强度：反事实特征相对参照组的相对位移
    feat_shift = [float(np.linalg.norm(feats[g] - feats[args.ref_age], axis=1).mean() /
                        (np.linalg.norm(feats[args.ref_age], axis=1).mean() + 1e-12))
                  for g in range(NUM_AGE)]
    print(f"特征相对位移 ||Δf||/||f_ref||（vs age={args.ref_age}）: "
          f"{[f'{s:.4f}' for s in feat_shift]}")

    # ---- 2. 两套投影基 ----
    mu = feats.reshape(-1, 512).mean(axis=0)                # 共同中心（跨组总均值）
    proj: dict[str, np.ndarray] = {}
    basis_diag: dict[str, dict[str, float]] = {}
    energies: dict[str, list[float]] = {}
    for basis in BASIS_KEYS:
        if basis == "discriminative":
            P, diag = build_discriminative_basis(w_eff)
        else:
            P, diag = build_pca_basis(feats.reshape(-1, 512))
        proj[f"{basis}_P"] = P
        proj[basis] = np.stack([(feats[g] - mu) @ P for g in range(NUM_AGE)])   # [G, N, 2]
        basis_diag[basis] = diag
        energies[basis] = [in_plane_energy(P, w_eff[g]) for g in range(NUM_AGE)]
        print(f"[{basis}] 诊断={ {k: round(v, 4) for k, v in diag.items()} }  "
              f"in-plane 能量={[round(e, 4) for e in energies[basis]]}")

    # ---- 3. Δlogit 通路分解 ----
    fc_term, conv_term, residual = decompose_delta_logit(feats, w_eff, logits, args.ref_age)
    print(f"Δlogit 分解恒等式最大残差={residual:.3e}（应 ~1e-4 量级，float32 数值误差）")
    pathway = {}
    for g in range(NUM_AGE):
        if g == args.ref_age:
            continue
        fa, ca = float(np.abs(fc_term[g]).mean()), float(np.abs(conv_term[g]).mean())
        pathway[f"age{g}"] = {
            "mean_abs_fc": fa, "mean_abs_conv": ca,
            "fc_share": fa / (fa + ca + 1e-12),
            "mean_delta_logit": float((fc_term[g] + conv_term[g]).mean()),
        }
        print(f"  age {args.ref_age}->{g}: |fc|={fa:.4f}  |conv|={ca:.4f}  "
              f"fc 占比={pathway[f'age{g}']['fc_share']:.1%}  "
              f"平均 Δlogit={pathway[f'age{g}']['mean_delta_logit']:+.4f}")

    # ---- 4. 出图 + 落盘 ----
    plot_idx = np.sort(rng.choice(n_samples, size=min(args.max_points, n_samples), replace=False))
    main_png = args.out_dir / f"hyperplane_by_age_{args.tag}.png"
    path_png = args.out_dir / f"delta_logit_pathway_{args.tag}.png"
    render_main_figure(proj, w_eff, bias, mu, labels, true_ages, aucs, logits, energies,
                       basis_diag, plot_idx, args.ckpt.name, main_png)
    render_pathway_figure(
        fc_term, conv_term, args.ref_age, residual,
        group_labels={g: f"age {g} ({AGE_LABELS[g]})" for g in range(NUM_AGE)},
        header=f"HAM10000 | HyperAdapt(age) | ckpt={args.ckpt.name}", out_path=path_png,
    )

    # ---- 4b. 通路归因审计（顺序敏感性 + AUC 口径 + 平移/重排），与条件化消融对账 ----
    audit = run_pathway_audit(
        feats, w_eff, logits, labels, args.ref_age,
        group_labels={g: f"age {g} ({AGE_LABELS[g]})" for g in range(NUM_AGE)},
    )

    metrics = {
        "ckpt": str(args.ckpt), "split": f"{args.split_dir}/{args.split}", "n_samples": n_samples,
        "malignant_rate": float(labels.mean()), "ref_age": args.ref_age,
        "fc_bias_shared": bias,
        "per_age": {f"age{g}": {"n_true": int((true_ages == g).sum()),
                                "counterfactual_auc": float(aucs[g]),
                                "w_eff_norm": float(np.linalg.norm(w_eff[g])),
                                "cos_to_mean_hyperplane": cos_to_bar[g],
                                "mean_logit": float(logits[g].mean()),
                                "feat_rel_shift_vs_ref": feat_shift[g],
                                "in_plane_energy_discriminative": energies["discriminative"][g],
                                "in_plane_energy_pca": energies["pca"][g]}
                    for g in range(NUM_AGE)},
        "basis_diagnostics": basis_diag,
        "pairwise_cosine_w_eff": cos_pairwise.tolist(),
        "delta_logit_pathway": pathway,
        "identity_max_residual": residual,
        "pathway_audit": audit,
    }
    metrics_path = args.out_dir / f"hyperplane_metrics_{args.tag}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[saved] {metrics_path}")

    npz_path = args.out_dir / f"hyperplane_arrays_{args.tag}.npz"
    np.savez_compressed(
        npz_path, w_eff=w_eff, bias=bias, mu=mu, labels=labels, true_ages=true_ages,
        logits=logits, fc_term=fc_term, conv_term=conv_term,
        P_discriminative=proj["discriminative_P"], P_pca=proj["pca_P"],
        U_discriminative=proj["discriminative"], U_pca=proj["pca"],
    )
    print(f"[saved] {npz_path}")


if __name__ == "__main__":
    main()
