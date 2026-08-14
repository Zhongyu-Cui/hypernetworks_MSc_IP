"""
胸片（CXR）HyperAdapt 分类超平面可视化的共享流程
================================================
被 scripts/viz_hyperadapt_hyperplane_mimic.py（MIMIC-CXR）与
scripts/viz_hyperadapt_hyperplane_chexpert.py（CheXpert）共用。

两个数据集在本可视化上**完全同构**：同一份 Dataset 类（MIMICCXRDataset，schema 相同）、
同一个模型类（ResNet18HyperAdapt，sex×race×age = 8 子群条件通路）、同一套 224×224 eval
transform，仅 config / split / checkpoint / 输出路径不同。差异全部收敛到 `CXRVizSpec`。

方法（三步，数学细节见 src/utils/hyperplane_viz.py 与
docs/hyperplane_subgroup_visualisation.md）：
  1. **反事实属性扫掠**：固定同一批 held-out 图像，对 8 个 (sex,race,age) 组合各前向一遍，
     得到 F_g = f(X, a_g) 与 w_g = W + ΔW(a_g)。病人固定 ⇒ 组间差异 100% 归因于 HN 的调制，
     不混杂「不同人群的胸片本来就不同」。
  2. **两套 2D 投影基**（同一中心 mu = 全部组特征总均值，保证 panel 间坐标可比），分两张图保存：
       · hyperplane_discriminative_<tag>.png —— 判别轴 × 组间调整轴（主口径）
       · hyperplane_pca_<tag>.png            —— 默认 PCA 前两主成分（对照）
  3. **Δlogit 通路分解**（代数恒等式）：fc 头贡献 <Δw, f_ref> vs conv 特征贡献 <w_g, Δf>，
     参照组默认取**抽样前** split 内真实样本量最大的子群。

CXR 共有注意事项
----------------
· 标签语义 0 = 有病灶，1 = 无病灶/健康（与 HAM 的 benign/malignant 相反方向，图例已相应命名）。
· 16-bit PNG 坑由 MIMICCXRDataset 内部处理（min-max 归一化后转 8-bit），此处不重复实现；
  **禁止** convert("L")（见仓库 CLAUDE.md「图像加载」节）。
· fold0 test 有数万张，全量扫掠 8 遍代价过高 ⇒ 按 (label × sex × race × age) 16 格分层抽样。
· age_group 由 age >= age_threshold(60) 在线二分箱，与训练时一致。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.datasets.mimic_cxr_dataset import MIMICCXRDataset
from src.models.resnet18_hyperadapt import ResNet18HyperAdapt
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

REPO_ROOT = Path(__file__).resolve().parents[2]

NUM_SEX, NUM_RACE, NUM_AGE = 2, 2, 2
NUM_GROUPS = NUM_SEX * NUM_RACE * NUM_AGE                   # 8 个 sex×race×age 子群
SEX_NAMES = {0: "Male", 1: "Female"}
RACE_NAMES = {0: "White", 1: "Non-White"}
AGE_NAMES = {0: "<60", 1: ">=60"}
CLASS_NAMES = ("finding", "no finding")                     # label 0 / 1（MEDFAIR 口径）
FEAT_DIM = 512                                              # ResNet-18 penultimate 维度


def group_index(sex: int, race: int, age: int) -> int:
    """(sex, race, age) -> 子群下标 0..7（混合进制编码，sex 最高位）。"""
    return sex * (NUM_RACE * NUM_AGE) + race * NUM_AGE + age


def group_triplet(idx: int) -> tuple[int, int, int]:
    """子群下标 0..7 -> (sex, race, age)，group_index 的逆。"""
    return idx // (NUM_RACE * NUM_AGE), (idx // NUM_AGE) % NUM_RACE, idx % NUM_AGE


GROUP_LABELS = {
    g: f"{SEX_NAMES[s]} / {RACE_NAMES[r]} / {AGE_NAMES[a]}"
    for g, (s, r, a) in ((g, group_triplet(g)) for g in range(NUM_GROUPS))
}


@dataclass(frozen=True)
class CXRVizSpec:
    """
    单个 CXR 数据集在本可视化中的全部差异项。

    Attributes:
        dataset_key: 落盘 metrics 的 dataset 字段与 outputs 子目录名（如 "chexpert_cxr"）。
        display_name: 图标题中的数据集名（如 "CheXpert"）。
        config_path: 训练配置 YAML（提供 eval transform / image_size / age_threshold）。
        default_ckpt: 默认 HyperAdapt checkpoint（cv5 选中配置 seed42）。
        default_split_dir: 默认仓库相对 split 目录（与 seed42 对应的 fold0，避免泄漏）。
        default_out_dir: 默认输出目录。
        default_tag: 默认输出文件名后缀。
        model_note: 追加到图标题的权重来源说明。ID 场景留空；**OOD 场景必填**
            （如 "weights trained on MIMIC-CXR"），否则读者会把图误读成 target 自训模型。
    """

    dataset_key: str
    display_name: str
    config_path: Path
    default_ckpt: Path
    default_split_dir: str
    default_out_dir: Path
    default_tag: str
    model_note: str = ""


def parse_args(spec: CXRVizSpec) -> argparse.Namespace:
    """按数据集 spec 构造并解析命令行参数。"""
    p = argparse.ArgumentParser(
        description=f"HyperAdapt(sex,race,age) 分类超平面逐子群可视化（{spec.display_name} 8 子群）"
    )
    p.add_argument("--ckpt", type=Path, default=spec.default_ckpt,
                   help="HyperAdapt checkpoint（plain state_dict）。默认 cv5 选中配置 seed42。")
    p.add_argument("--split_dir", type=str, default=spec.default_split_dir,
                   help="仓库相对 split 目录。默认 cv5/fold0（与 seed42 checkpoint 对应的 "
                        "held-out，避免数据泄漏）。")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--n_samples", type=int, default=4000,
                   help="扫掠样本数，按 label×sex×race×age 共 16 格分层抽样（fold0 test "
                        "数万张，全量 × 8 组代价过高）。")
    p.add_argument("--ref_group", type=int, default=-1,
                   help="Δlogit 分解的参照子群下标 0..7；-1（默认）= 自动取 split 内真实规模最大的"
                        "子群（不是抽样后规模——分层抽样后各组等量）。")
    p.add_argument("--batch_size", type=int, default=32,
                   help="前向 batch（HyperAdapt 逐样本卷积核显存重）。")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_points", type=int, default=900,
                   help="散点图最多绘制的样本数（仅影响绘图密度，统计仍用全部抽样样本）。")
    p.add_argument("--seed", type=int, default=42, help="分层抽样 + 绘图抽样种子。")
    p.add_argument("--out_dir", type=Path, default=spec.default_out_dir)
    p.add_argument("--tag", type=str, default=spec.default_tag,
                   help="输出文件名后缀，便于区分数据集/fold/seed。")
    return p.parse_args()


# ============================================================
# 模型 / 数据
# ============================================================
def load_model(ckpt_path: Path, device: torch.device) -> ResNet18HyperAdapt:
    """构建三属性 ResNet18HyperAdapt 并严格载入 plain state_dict，切 eval。"""
    model = ResNet18HyperAdapt(
        num_classes=1, patient_embed_dim=128, rank=4, pretrained=False, freeze_backbone=False,
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def build_stratified_loader(
    config_path: Path, split_dir: str, split: str, n_samples: int, batch_size: int,
    num_workers: int, rng: np.random.Generator,
) -> tuple[DataLoader, np.ndarray, np.ndarray, np.ndarray]:
    """
    构建 CXR eval DataLoader（Grayscale(3)->Resize(224)->ToTensor->ImageNet Normalize），
    并按 label × sex × race × age 共 16 格分层抽样到约 n_samples 张。

    分层的理由：CXR 子群规模极不均衡（White & Male 占多数，且健康率仅 8%~30%），随机抽样会让
    Non-White & >=60 等小子群、以及少数类样本过少、逐组 AUC 不可读；分层保证每格都有支撑。

    Args:
        config_path: 数据集训练配置 YAML（取 eval transform / image_size / age_threshold）。
        split_dir: 仓库相对 split 目录。
        split: train / val / test。
        n_samples: 目标抽样总数（16 格均分配额，格内不足则全取，故实际可少于此数）。
        batch_size: 前向 batch。
        num_workers: DataLoader worker 数。
        rng: 抽样随机数发生器。

    Returns:
        (loader, labels, group_ids, full_counts)：前三者与前向顺序（shuffle=False）逐位对应；
        full_counts 为**抽样前**整个 split 的各子群规模（抽样后各组近似等量，选参照组须用抽样前的
        真实规模，否则「最大子群」会退化成下标 0）。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ev, norm = cfg["transforms"]["eval"], cfg["transforms"]["normalize"]
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(ev["resize"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm["mean"], std=norm["std"]),
    ])

    dataset = MIMICCXRDataset(
        csv_path=REPO_ROOT / split_dir / f"{split}.csv",
        transform=transform,
        image_size=cfg["data"]["image_size"],
        age_threshold=cfg["attributes"]["age"]["age_threshold"],
    )
    group_all = np.array([group_index(s, r, a) for s, r, a
                          in zip(dataset.sexes, dataset.races, dataset.age_groups)])

    # 16 格（8 子群 × 2 类）分层抽样：每格配额相同，格内不足则全取
    per_cell = max(1, n_samples // (NUM_GROUPS * 2))
    picks: list[int] = []
    for g in range(NUM_GROUPS):
        for y in (0, 1):
            cell = np.flatnonzero((group_all == g) & (dataset.labels == y))
            if cell.size:
                picks.extend(rng.choice(cell, size=min(cell.size, per_cell),
                                        replace=False).tolist())
    keep = np.sort(np.array(picks, dtype=int))

    subset = Subset(dataset, keep.tolist())
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    full_counts = np.bincount(group_all, minlength=NUM_GROUPS)
    return loader, dataset.labels[keep].astype(int), group_all[keep], full_counts


# ============================================================
# 反事实扫掠
# ============================================================
@torch.no_grad()
def sweep_features_and_hyperplanes(
    model: ResNet18HyperAdapt, loader: DataLoader, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    对同一批图像做 8 子群反事实扫掠：每个 (sex,race,age) 组合前向一遍，取 penultimate
    特征与有效超平面。

    特征通过 forward hook 抓 avgpool 输出（等价 forward 里 torch.flatten(out, 1) 的值），
    不改动模型代码。w_eff(g) = fc.weight + fc_A_gen(emb) @ fc_B_gen(emb)，同组内整批
    embedding 相同故只需一行。

    Args:
        model: 已载入权重并切 eval 的 ResNet18HyperAdapt。
        loader: shuffle=False 的 eval DataLoader。
        device: 前向设备。

    Returns:
        feats  : [8, N, 512] 各子群反事实特征。
        w_eff  : [8, 512]    各子群有效超平面法向量。
        bias   : float       fc.bias（HyperAdapt 不条件化 bias，全组共用）。
        logits : [8, N]      各子群反事实 logit。
    """
    captured: list[torch.Tensor] = []

    def hook(_module, _inp, out: torch.Tensor) -> None:
        # avgpool 输出 [B, 512, 1, 1] -> flatten 成 [B, 512]，与 fc 输入完全一致
        captured.append(torch.flatten(out, 1).detach().float().cpu())

    handle = model.avgpool.register_forward_hook(hook)

    feats_per_group, logits_per_group, w_per_group = [], [], []
    try:
        for g in range(NUM_GROUPS):
            sex_v, race_v, age_v = group_triplet(g)
            captured.clear()
            logit_chunks = []
            for image, _label, _sex, _race, _age in loader:
                image = image.to(device, non_blocking=True)
                n = image.shape[0]
                attrs = [torch.full((n,), v, dtype=torch.long, device=device)
                         for v in (sex_v, race_v, age_v)]
                logit_chunks.append(model(image, *attrs).squeeze(1).detach().float().cpu())
            feats_per_group.append(torch.cat(captured).numpy())
            logits_per_group.append(torch.cat(logit_chunks).numpy())

            one = [torch.tensor([v], dtype=torch.long, device=device)
                   for v in (sex_v, race_v, age_v)]
            emb = model.patient_embed(*one)                                  # [1, 128]
            a_fc = model.fc_A_gen(emb).view(1, model.out_dim, model.rank)    # [1, 1, k]
            b_fc = model.fc_B_gen(emb).view(1, model.rank, FEAT_DIM)         # [1, k, 512]
            delta_w = torch.bmm(a_fc, b_fc)[0]                               # [1, 512]
            w_per_group.append((model.fc.weight + delta_w)[0].detach().float().cpu().numpy())
    finally:
        handle.remove()

    bias = float(model.fc.bias.detach().cpu().numpy()[0])
    return (np.stack(feats_per_group), np.stack(w_per_group), bias, np.stack(logits_per_group))


# ============================================================
# 主流程
# ============================================================
def run_hyperplane_viz(spec: CXRVizSpec, args: argparse.Namespace) -> dict[str, np.ndarray]:
    """
    执行完整流程：分层抽样 -> 反事实扫掠 -> 两套投影基作图 -> Δlogit 通路分解 + 三口径审计 -> 落盘。

    Args:
        spec: 数据集差异项（config / 默认路径 / 图标题名）。
        args: parse_args(spec) 的解析结果（CLI 可覆盖 spec 的默认路径）。

    Returns:
        本次扫掠的原始数组（feats/w_eff/bias/logits/labels/group_ids/mu/P_discriminative/ref_idx）。
        独立作图时可忽略；OOD 驱动脚本用它做 ID↔OOD 同帧叠加，避免再前向一遍。
        产物同时写入 args.out_dir：2 张基图 + 1 张通路图 + metrics JSON + arrays npz。
    """
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  ckpt={args.ckpt.name}  split={args.split_dir}/{args.split}")

    model = load_model(args.ckpt, device)
    loader, labels, group_ids, full_counts = build_stratified_loader(
        spec.config_path, args.split_dir, args.split, args.n_samples,
        args.batch_size, args.num_workers, rng)
    n_samples = len(labels)
    counts = np.bincount(group_ids, minlength=NUM_GROUPS)
    print(f"分层抽样 n={n_samples}  no-finding 率={labels.mean():.3f}")
    for g in range(NUM_GROUPS):
        in_g = group_ids == g
        print(f"  [{g}] {GROUP_LABELS[g]:<26s} n={counts[g]:>4d}（split 内 {full_counts[g]:>6d}）  "
              f"no-finding 率={labels[in_g].mean() if in_g.any() else float('nan'):.3f}")

    # 参照组按**抽样前**的真实规模选（抽样后各组近似等量，argmax 会退化成下标 0）
    ref_idx = int(np.argmax(full_counts)) if args.ref_group < 0 else args.ref_group
    print(f"参照子群 = [{ref_idx}] {GROUP_LABELS[ref_idx]}"
          f"（split 内 n={full_counts[ref_idx]}，抽样 n={counts[ref_idx]}）")

    # ---- 1. 反事实扫掠 ----
    print(f"反事实扫掠：{NUM_GROUPS} 子群 × {n_samples} 张前向 ...")
    feats, w_eff, bias, logits = sweep_features_and_hyperplanes(model, loader, device)
    aucs = np.array([roc_auc_score(labels, logits[g]) for g in range(NUM_GROUPS)])
    print(f"fc.bias={bias:+.4f}（HyperAdapt 不条件化 bias）")
    for g in range(NUM_GROUPS):
        print(f"  [{g}] {GROUP_LABELS[g]:<26s} ||w_eff||={np.linalg.norm(w_eff[g]):.4f}  "
              f"反事实全体 AUC={aucs[g]:.4f}  logit 均值={logits[g].mean():+.4f}")

    # 组间几何（补充读数）：与平均打分表的夹角、两两余弦
    w_bar = w_eff.mean(axis=0)
    cos_to_bar = [float(w_eff[g] @ w_bar / (np.linalg.norm(w_eff[g]) * np.linalg.norm(w_bar) + 1e-12))
                  for g in range(NUM_GROUPS)]
    cos_pairwise = np.array([[float(w_eff[i] @ w_eff[j] /
                                    (np.linalg.norm(w_eff[i]) * np.linalg.norm(w_eff[j]) + 1e-12))
                              for j in range(NUM_GROUPS)] for i in range(NUM_GROUPS)])
    print(f"与平均超平面的余弦: {[f'{c:.5f}' for c in cos_to_bar]}")
    print(f"组间最小余弦（最大夹角）: {cos_pairwise[np.triu_indices(NUM_GROUPS, 1)].min():.5f}")
    feat_shift = [float(np.linalg.norm(feats[g] - feats[ref_idx], axis=1).mean() /
                        (np.linalg.norm(feats[ref_idx], axis=1).mean() + 1e-12))
                  for g in range(NUM_GROUPS)]
    print(f"特征相对位移 ||Δf||/||f_ref||: {[f'{s:.4f}' for s in feat_shift]}")

    # ---- 2. 两套投影基 → **分两张图保存** ----
    mu = feats.reshape(-1, FEAT_DIM).mean(axis=0)           # 共同中心（跨组总均值）
    plot_idx = np.sort(rng.choice(n_samples, size=min(args.max_points, n_samples), replace=False))
    note = f" | {spec.model_note}" if spec.model_note else ""
    header = (f"{spec.display_name} | HyperAdapt(sex,race,age): decision-hyperplane adjustment "
              f"across the 8 sex x race x age subgroups (counterfactual attribute sweep)"
              f"{note}\n"
              f"Same held-out images forwarded once per assumed subgroup; shared centre mu = grand "
              f"mean over all conditions | n={n_samples} (stratified) | ckpt={args.ckpt.name}")

    proj: dict[str, np.ndarray] = {}
    basis_diag: dict[str, dict[str, float]] = {}
    energies: dict[str, list[float]] = {}
    for basis in BASIS_KEYS:
        if basis == "discriminative":
            P, diag = build_discriminative_basis(w_eff)
        else:
            P, diag = build_pca_basis(feats.reshape(-1, FEAT_DIM))
        U = np.stack([(feats[g] - mu) @ P for g in range(NUM_GROUPS)])       # [8, N, 2]
        proj[f"{basis}_P"], proj[basis] = P, U
        basis_diag[basis] = diag
        energies[basis] = [in_plane_energy(P, w_eff[g]) for g in range(NUM_GROUPS)]
        print(f"[{basis}] 诊断={ {k: round(v, 4) for k, v in diag.items()} }  "
              f"in-plane 能量={[round(e, 4) for e in energies[basis]]}")

        render_basis_figure(
            basis=basis, U=U, P=P, w_eff=w_eff, bias=bias, mu=mu, labels=labels,
            group_ids=group_ids, aucs=aucs, logits=logits, energies=energies[basis],
            diag=diag, plot_idx=plot_idx, group_labels=GROUP_LABELS,
            class_names=CLASS_NAMES, ncols=4, header=header,
            out_path=args.out_dir / f"hyperplane_{basis}_{args.tag}.png",
        )

    # ---- 3. Δlogit 通路分解 ----
    fc_term, conv_term, residual = decompose_delta_logit(feats, w_eff, logits, ref_idx)
    print(f"Δlogit 分解恒等式最大残差={residual:.3e}（float32 数值误差量级）")
    pathway = {}
    for g in range(NUM_GROUPS):
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
        header=f"{spec.display_name} | HyperAdapt(sex,race,age){note} | ckpt={args.ckpt.name}",
        out_path=args.out_dir / f"delta_logit_pathway_{args.tag}.png",
    )

    # ---- 3b. 通路归因审计（顺序敏感性 + AUC 口径 + 平移/重排），与条件化消融对账 ----
    audit = run_pathway_audit(feats, w_eff, logits, labels, ref_idx, GROUP_LABELS)

    # ---- 4. 落盘 ----
    metrics = {
        "dataset": spec.dataset_key, "ckpt": str(args.ckpt),
        "split": f"{args.split_dir}/{args.split}", "n_samples": n_samples,
        "no_finding_rate": float(labels.mean()), "ref_group": ref_idx,
        "ref_group_label": GROUP_LABELS[ref_idx], "fc_bias_shared": bias,
        "per_group": {GROUP_LABELS[g]: {"index": g, "n_sampled": int(counts[g]),
                                        "n_in_split": int(full_counts[g]),
                                        "counterfactual_auc": float(aucs[g]),
                                        "w_eff_norm": float(np.linalg.norm(w_eff[g])),
                                        "cos_to_mean_hyperplane": cos_to_bar[g],
                                        "mean_logit": float(logits[g].mean()),
                                        "feat_rel_shift_vs_ref": feat_shift[g],
                                        "in_plane_energy_discriminative":
                                            energies["discriminative"][g],
                                        "in_plane_energy_pca": energies["pca"][g]}
                      for g in range(NUM_GROUPS)},
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
        npz_path, w_eff=w_eff, bias=bias, mu=mu, labels=labels, group_ids=group_ids,
        logits=logits, fc_term=fc_term, conv_term=conv_term,
        P_discriminative=proj["discriminative_P"], P_pca=proj["pca_P"],
        U_discriminative=proj["discriminative"], U_pca=proj["pca"],
    )
    print(f"[saved] {npz_path}")

    return {
        "feats": feats, "w_eff": w_eff, "bias": np.array(bias), "logits": logits,
        "labels": labels, "group_ids": group_ids, "mu": mu,
        "P_discriminative": proj["discriminative_P"], "ref_idx": np.array(ref_idx),
    }
