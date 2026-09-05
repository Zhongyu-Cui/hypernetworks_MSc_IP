"""
HyperAdapt(skin) 分类超平面的逐子群可视化（Fitzpatrick17k，6 个肤色组）
=======================================================================
与 scripts/viz_hyperadapt_hyperplane_{ham,mimic,chexpert}.py 同方法（数学与渲染共用
src/utils/hyperplane_viz.py，通路审计共用 src/utils/pathway_audit_report.py），换成
Fitzpatrick17k malignant 二分类、**单一肤色属性**条件通路（SkinEmbedding: skin ∈ {0..5}
= Fitzpatrick I–VI，共 6 个子群）。方法学规格见 docs/hyperplane_subgroup_visualisation.md。

方法（三步）
------------
1. **反事实属性扫掠**：固定同一批 held-out 图像，对 6 个肤色各前向一遍，得到 F_g = f(X, g)
   与 w_g = W + ΔW(g)。病人固定 ⇒ 组间差异 100% 归因于 HN 的调制，不混杂「不同肤色人群的
   皮损本来就不同」（这一点对 Fitzpatrick 尤其关键：肤色本身在图像里高度可解码）。
2. **两套 2D 投影基**（同一中心 mu = 全部组特征总均值，保证 panel 间坐标可比），分两张图保存：
     · hyperplane_discriminative_<tag>.png —— 判别轴 × 组间调整轴（主口径）
     · hyperplane_pca_<tag>.png            —— 默认 PCA 前两主成分（对照）
   6 个子群按 2×3 排布（ncols=3）。
3. **Δlogit 通路分解**（代数恒等式）：fc 头贡献 <Δw, f_ref> vs conv 特征贡献 <w_g, Δf>，
   参照组默认取样本量最大的肤色组（split 内 argmax，通常为型 II）。

Fitzpatrick 特有注意
--------------------
· 标签语义 1 = malignant（与 HAM 同向，与 CXR 的 no-finding 相反），恶性率约 13.5%。
· `preproc_224x224` 为自然 RGB JPEG（uint8），**无** CXR 那套 16-bit 处理；eval transform
  仅 ToTensor + ImageNet Normalize（源图已 224 原生，无 Resize/Crop），与训练脚本一致。
· split 构建阶段已排除 `fitzpatrick_scale == -1`，故**无需**像 HAM 那样在内存里过滤属性。
· fold0 test 仅 3,203 张 ⇒ **全量扫掠，不抽样**（不存在 CXR 那种分层抽样的口径偏差）。
· 深肤色组绝对样本少（型 VI 全 split 仅 126 张、恶性 12 例），逐组 AUC 的噪声本身就大。

运行（medimg env，实验室机器即可，纯推理）：
    PYTHONPATH=. python scripts/viz_hyperadapt_hyperplane_fitzpatrick.py \
        --ckpt "$HN_OUTPUTS"/fitzpatrick/cv5/hyperadapt_lr3e-04_wd1e-03_seed42_best_overall.pth \
        --split_dir data/splits/fitzpatrick17k/cv5/fold0

输出：outputs/fitzpatrick/hyperplane_viz/ 下 3 张 PNG + metrics JSON + arrays npz。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.fitzpatrick_dataset import FitzpatrickDataset
from src.models.resnet18_hyperadapt import ResNet18HyperAdaptSkin
from src.utils.hyperplane_viz import (
    BASIS_KEYS,
    build_discriminative_basis,
    build_pca_basis,
    decompose_delta_logit,
    in_plane_energy,
    render_basis_figure,
    render_pathway_figure,
)
from src.utils.pathway_audit_report import run_pathway_audit
from src.paths import OUTPUTS_DIR

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "fitzpatrick_baseline.yaml"
OUTPUTS_ROOT = OUTPUTS_DIR
# cv5 选中配置（cv5/selected_configs.json: hyperadapt = lr3e-04_wd1e-03），seed42 ↔ fold0
DEFAULT_CKPT = (
    OUTPUTS_ROOT / "fitzpatrick" / "cv5" / "hyperadapt_lr3e-04_wd1e-03_seed42_best_overall.pth"
)
DEFAULT_OUT_DIR = OUTPUTS_ROOT / "fitzpatrick" / "hyperplane_viz"

NUM_SKIN = 6                                                # Fitzpatrick I–VI
FEAT_DIM = 512                                              # ResNet-18 penultimate 维度
SKIN_ROMAN = {0: "I", 1: "II", 2: "III", 3: "IV", 4: "V", 5: "VI"}
GROUP_LABELS = {g: f"Fitzpatrick {SKIN_ROMAN[g]} (skin={g})" for g in range(NUM_SKIN)}
CLASS_NAMES = ("benign", "malignant")                       # label 0 / 1


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser(
        description="HyperAdapt(skin) 分类超平面逐子群可视化（Fitzpatrick17k 6 肤色组）"
    )
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                   help="HyperAdapt(skin) checkpoint（plain state_dict）。默认 cv5 选中配置 seed42。")
    p.add_argument("--split_dir", type=str, default="data/splits/fitzpatrick17k/cv5/fold0",
                   help="仓库相对 split 目录。默认 cv5/fold0（与 seed42 checkpoint 对应的 "
                        "held-out，避免数据泄漏）。")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--ref_skin", type=int, default=-1,
                   help="Δlogit 分解的参照肤色组 0..5；-1（默认）= 自动取 split 内样本量最大的组。")
    p.add_argument("--batch_size", type=int, default=64,
                   help="前向 batch（HyperAdapt 逐样本卷积核显存重）。")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_points", type=int, default=900,
                   help="散点图最多绘制的样本数（仅影响绘图密度，统计仍用全部样本）。")
    p.add_argument("--seed", type=int, default=42, help="绘图抽样种子。")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--tag", type=str, default="fitz_fold0_seed42",
                   help="输出文件名后缀，便于区分 fold/seed。")
    return p.parse_args()


# ============================================================
# 模型 / 数据
# ============================================================
def load_model(ckpt_path: Path, device: torch.device) -> ResNet18HyperAdaptSkin:
    """构建单属性（skin）ResNet18HyperAdaptSkin 并严格载入 plain state_dict，切 eval。"""
    model = ResNet18HyperAdaptSkin(
        num_classes=1, num_skin=NUM_SKIN, patient_embed_dim=128, rank=4,
        pretrained=False, freeze_backbone=False,
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def build_eval_loader(
    split_dir: str, split: str, batch_size: int, num_workers: int,
) -> tuple[DataLoader, np.ndarray, np.ndarray]:
    """
    构建 Fitzpatrick eval DataLoader（ToTensor -> ImageNet Normalize，224 原生无 Resize/Crop，
    与 src/training/train_fitzpatrick_hyperadapt.py 的 eval_transform 一致）。

    fold0 test 仅 3,203 张，6 组全量扫掠可承受 ⇒ 不抽样，逐组 AUC 即 split 全量指标。

    Args:
        split_dir: 仓库相对 split 目录。
        split: train / val / test。
        batch_size: 前向 batch。
        num_workers: DataLoader worker 数。

    Returns:
        (loader, labels, true_skins)：后两者按 CSV 顺序（shuffle=False）与前向顺序逐位对应。
    """
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    norm = cfg["transforms"]["normalize"]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=norm["mean"], std=norm["std"]),
    ])

    dataset = FitzpatrickDataset(csv_path=REPO_ROOT / split_dir / f"{split}.csv",
                                 transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return loader, dataset.labels.astype(int), dataset.skins.astype(int)


# ============================================================
# 反事实扫掠
# ============================================================
@torch.no_grad()
def sweep_features_and_hyperplanes(
    model: ResNet18HyperAdaptSkin, loader: DataLoader, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    对同一批图像做 6 肤色组反事实扫掠：每个 skin 前向一遍，取 penultimate 特征与有效超平面。

    特征通过 forward hook 抓 avgpool 输出（等价 forward 里 torch.flatten(out, 1) 的值），
    不改动模型代码。w_eff(g) = fc.weight + fc_A_gen(emb) @ fc_B_gen(emb)，同组内整批
    embedding 相同故只需一行。

    Args:
        model: 已载入权重并切 eval 的 ResNet18HyperAdaptSkin。
        loader: shuffle=False 的 eval DataLoader。
        device: 前向设备。

    Returns:
        feats  : [6, N, 512] 各肤色组反事实特征。
        w_eff  : [6, 512]    各肤色组有效超平面法向量。
        bias   : float       fc.bias（HyperAdapt 不条件化 bias，全组共用）。
        logits : [6, N]      各肤色组反事实 logit。
    """
    captured: list[torch.Tensor] = []

    def hook(_module, _inp, out: torch.Tensor) -> None:
        # avgpool 输出 [B, 512, 1, 1] -> flatten 成 [B, 512]，与 fc 输入完全一致
        captured.append(torch.flatten(out, 1).detach().float().cpu())

    handle = model.avgpool.register_forward_hook(hook)

    feats_per_group, logits_per_group, w_per_group = [], [], []
    try:
        for g in range(NUM_SKIN):
            captured.clear()
            logit_chunks = []
            for image, _label, _skin in loader:
                image = image.to(device, non_blocking=True)
                skin_t = torch.full((image.shape[0],), g, dtype=torch.long, device=device)
                logit_chunks.append(model(image, skin_t).squeeze(1).detach().float().cpu())
            feats_per_group.append(torch.cat(captured).numpy())
            logits_per_group.append(torch.cat(logit_chunks).numpy())

            skin_one = torch.tensor([g], dtype=torch.long, device=device)
            emb = model.patient_embed(skin_one)                              # [1, 128]
            a_fc = model.fc_A_gen(emb).view(1, model.out_dim, model.rank)    # [1, 1, k]
            b_fc = model.fc_B_gen(emb).view(1, model.rank, FEAT_DIM)         # [1, k, 512]
            delta_w = torch.bmm(a_fc, b_fc)[0]                               # [1, 512]
            w_per_group.append((model.fc.weight + delta_w)[0].detach().float().cpu().numpy())
    finally:
        handle.remove()

    bias = float(model.fc.bias.detach().cpu().numpy()[0])
    return (np.stack(feats_per_group), np.stack(w_per_group), bias, np.stack(logits_per_group))


# ============================================================
# main
# ============================================================
def main() -> None:
    """执行完整流程：全量扫掠 -> 两套投影基作图 -> Δlogit 分解 + 三口径审计 -> 落盘。"""
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  ckpt={args.ckpt.name}  split={args.split_dir}/{args.split}")

    model = load_model(args.ckpt, device)
    loader, labels, true_skins = build_eval_loader(
        args.split_dir, args.split, args.batch_size, args.num_workers)
    n_samples = len(labels)
    counts = np.bincount(true_skins, minlength=NUM_SKIN)
    print(f"全量 n={n_samples}（不抽样）  malignant 率={labels.mean():.3f}")
    for g in range(NUM_SKIN):
        in_g = true_skins == g
        print(f"  [{g}] {GROUP_LABELS[g]:<26s} n={counts[g]:>4d}  "
              f"malignant 率={labels[in_g].mean() if in_g.any() else float('nan'):.3f}")

    ref_idx = int(np.argmax(counts)) if args.ref_skin < 0 else args.ref_skin
    print(f"参照子群 = [{ref_idx}] {GROUP_LABELS[ref_idx]}（n={counts[ref_idx]}）")

    # ---- 1. 反事实扫掠 ----
    print(f"反事实扫掠：{NUM_SKIN} 组 × {n_samples} 张前向 ...")
    feats, w_eff, bias, logits = sweep_features_and_hyperplanes(model, loader, device)
    aucs = np.array([roc_auc_score(labels, logits[g]) for g in range(NUM_SKIN)])
    print(f"fc.bias={bias:+.4f}（HyperAdapt 不条件化 bias）")
    for g in range(NUM_SKIN):
        print(f"  [{g}] {GROUP_LABELS[g]:<26s} ||w_eff||={np.linalg.norm(w_eff[g]):.4f}  "
              f"反事实全体 AUC={aucs[g]:.4f}  logit 均值={logits[g].mean():+.4f}")

    # 组间几何（补充读数）：与平均打分表的夹角、两两余弦
    w_bar = w_eff.mean(axis=0)
    cos_to_bar = [float(w_eff[g] @ w_bar / (np.linalg.norm(w_eff[g]) * np.linalg.norm(w_bar) + 1e-12))
                  for g in range(NUM_SKIN)]
    cos_pairwise = np.array([[float(w_eff[i] @ w_eff[j] /
                                    (np.linalg.norm(w_eff[i]) * np.linalg.norm(w_eff[j]) + 1e-12))
                              for j in range(NUM_SKIN)] for i in range(NUM_SKIN)])
    print(f"与平均超平面的余弦: {[f'{c:.5f}' for c in cos_to_bar]}")
    print(f"组间最小余弦（最大夹角）: {cos_pairwise[np.triu_indices(NUM_SKIN, 1)].min():.5f}")
    feat_shift = [float(np.linalg.norm(feats[g] - feats[ref_idx], axis=1).mean() /
                        (np.linalg.norm(feats[ref_idx], axis=1).mean() + 1e-12))
                  for g in range(NUM_SKIN)]
    print(f"特征相对位移 ||Δf||/||f_ref||: {[f'{s:.4f}' for s in feat_shift]}")

    # ---- 2. 两套投影基 → 分两张图保存 ----
    mu = feats.reshape(-1, FEAT_DIM).mean(axis=0)           # 共同中心（跨组总均值）
    plot_idx = np.sort(rng.choice(n_samples, size=min(args.max_points, n_samples), replace=False))
    header = (f"Fitzpatrick17k | HyperAdapt(skin): decision-hyperplane adjustment across the 6 "
              f"Fitzpatrick skin-type subgroups (counterfactual attribute sweep)\n"
              f"Same held-out images forwarded once per assumed skin type; shared centre mu = grand "
              f"mean over all conditions | n={n_samples} (full test) | ckpt={args.ckpt.name}")

    proj: dict[str, np.ndarray] = {}
    basis_diag: dict[str, dict[str, float]] = {}
    energies: dict[str, list[float]] = {}
    for basis in BASIS_KEYS:
        if basis == "discriminative":
            P, diag = build_discriminative_basis(w_eff)
        else:
            P, diag = build_pca_basis(feats.reshape(-1, FEAT_DIM))
        U = np.stack([(feats[g] - mu) @ P for g in range(NUM_SKIN)])         # [6, N, 2]
        proj[f"{basis}_P"], proj[basis] = P, U
        basis_diag[basis] = diag
        energies[basis] = [in_plane_energy(P, w_eff[g]) for g in range(NUM_SKIN)]
        print(f"[{basis}] 诊断={ {k: round(v, 4) for k, v in diag.items()} }  "
              f"in-plane 能量={[round(e, 4) for e in energies[basis]]}")

        render_basis_figure(
            basis=basis, U=U, P=P, w_eff=w_eff, bias=bias, mu=mu, labels=labels,
            group_ids=true_skins, aucs=aucs, logits=logits, energies=energies[basis],
            diag=diag, plot_idx=plot_idx, group_labels=GROUP_LABELS,
            class_names=CLASS_NAMES, ncols=3, header=header,
            out_path=args.out_dir / f"hyperplane_{basis}_{args.tag}.png",
        )

    # ---- 3. Δlogit 通路分解 ----
    fc_term, conv_term, residual = decompose_delta_logit(feats, w_eff, logits, ref_idx)
    print(f"Δlogit 分解恒等式最大残差={residual:.3e}（float32 数值误差量级）")
    pathway = {}
    for g in range(NUM_SKIN):
        if g == ref_idx:
            continue
        fa, ca = float(np.abs(fc_term[g]).mean()), float(np.abs(conv_term[g]).mean())
        pathway[GROUP_LABELS[g]] = {
            "mean_abs_fc": fa, "mean_abs_conv": ca, "fc_share": fa / (fa + ca + 1e-12),
            "mean_delta_logit": float((fc_term[g] + conv_term[g]).mean()),
        }
        print(f"  [{ref_idx}]->[{g}] {GROUP_LABELS[g]:<26s} |fc|={fa:.4f}  |conv|={ca:.4f}  "
              f"fc 占比={pathway[GROUP_LABELS[g]]['fc_share']:.1%}  "
              f"平均 Δlogit={pathway[GROUP_LABELS[g]]['mean_delta_logit']:+.4f}")

    render_pathway_figure(
        fc_term, conv_term, ref_idx, residual, GROUP_LABELS,
        header=f"Fitzpatrick17k | HyperAdapt(skin) | ckpt={args.ckpt.name}",
        out_path=args.out_dir / f"delta_logit_pathway_{args.tag}.png",
    )

    # ---- 3b. 通路归因审计（顺序敏感性 + AUC 口径 + 平移/重排），与条件化消融对账 ----
    audit = run_pathway_audit(feats, w_eff, logits, labels, ref_idx, GROUP_LABELS)

    # ---- 4. 落盘 ----
    metrics = {
        "dataset": "fitzpatrick17k", "ckpt": str(args.ckpt),
        "split": f"{args.split_dir}/{args.split}", "n_samples": n_samples,
        "malignant_rate": float(labels.mean()), "ref_group": ref_idx,
        "ref_group_label": GROUP_LABELS[ref_idx], "fc_bias_shared": bias,
        "per_group": {GROUP_LABELS[g]: {"index": g, "n_true": int(counts[g]),
                                        "counterfactual_auc": float(aucs[g]),
                                        "w_eff_norm": float(np.linalg.norm(w_eff[g])),
                                        "cos_to_mean_hyperplane": cos_to_bar[g],
                                        "mean_logit": float(logits[g].mean()),
                                        "feat_rel_shift_vs_ref": feat_shift[g],
                                        "in_plane_energy_discriminative":
                                            energies["discriminative"][g],
                                        "in_plane_energy_pca": energies["pca"][g]}
                      for g in range(NUM_SKIN)},
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
        npz_path, w_eff=w_eff, bias=bias, mu=mu, labels=labels, true_skins=true_skins,
        logits=logits, fc_term=fc_term, conv_term=conv_term,
        P_discriminative=proj["discriminative_P"], P_pca=proj["pca_P"],
        U_discriminative=proj["discriminative"], U_pca=proj["pca"],
    )
    print(f"[saved] {npz_path}")


if __name__ == "__main__":
    main()
