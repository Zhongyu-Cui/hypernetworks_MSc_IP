"""
HAM10000 · HyperAdapt(sex+age) 的反事实超平面扫掠（arrays + metrics，不出图）
=============================================================================
`viz_hyperadapt_hyperplane_ham.py` 硬编码 `ResNet18HyperAdaptAge`（4 个 age 组），而报告第 5 章
的 HAM 三个超网络臂已统一为 **sex+age 双轴**（`*_sexage`）。本脚本按双轴口径重跑同一套扫掠，
产出与 age-only 版**同 schema** 的 `.npz` 与 metrics JSON，供报告合成图取用；
**不改动原脚本**，age-only 的历史产物仍可复现。

方法与 `docs/hyperplane_subgroup_visualisation.md` §2 完全一致：
固定同一批 held-out 图像，对 8 个 (sex, age) 组合各前向一次，取 penultimate 特征 f(x,g) 与
有效超平面 w_eff(g) = W + A_fc(emb_g)·B_fc(emb_g)，再投影到 §3.5.2 的判别基 e1/e2。

运行（medimg env，纯推理，GPU 可选）：
    PYTHONPATH=. python scripts/viz_hyperplane_ham_sexage.py
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.datasets.ham10000_dataset import HAM10000Dataset
from src.models.resnet18_hyperadapt import ResNet18HyperAdaptSexAge
from src.utils.hyperplane_viz import (
    build_discriminative_basis, build_pca_basis, in_plane_energy,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "ham10000_baseline.yaml"
OUTPUTS_ROOT = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs")
DEFAULT_CKPT = (OUTPUTS_ROOT / "ham10000" / "cv5" /
                "hyperadapt_sexage_lr3e-05_wd1e-03_seed42_best_overall.pth")
OUT_DIR = OUTPUTS_ROOT / "ham10000" / "hyperplane_viz"

NUM_SEX, NUM_AGE = 2, 4
GROUPS = list(itertools.product(range(NUM_SEX), range(NUM_AGE)))     # 8 个 (sex, age)
AGE_NAMES = {0: "20-40", 1: "40-60", 2: "60-80", 3: "80+"}
SEX_NAMES = {0: "M", 1: "F"}
GROUP_LABELS = [f"{SEX_NAMES[s]} {AGE_NAMES[a]}" for s, a in GROUPS]
REF_GROUP = 1          # 参照组 = (Male, 40-60)，HAM 最大格，与 age-only 版取 40-60 一致


def build_loader(split_dir: str, split: str, batch_size: int, num_workers: int):
    """HAM eval 变换下的 DataLoader，仅保留 age_group>=0（embedding 不接受 -1）。"""
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    ev, norm = cfg["transforms"]["eval"], cfg["transforms"]["normalize"]
    transform = transforms.Compose([
        transforms.Resize(tuple(ev["resize"])),
        transforms.CenterCrop(ev["center_crop"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm["mean"], std=norm["std"]),
    ])
    dataset = HAM10000Dataset(csv_path=REPO_ROOT / split_dir / f"{split}.csv",
                              transform=transform)
    keep = np.flatnonzero(dataset.age_groups >= 0)
    loader = DataLoader(Subset(dataset, keep.tolist()), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers,
                        pin_memory=torch.cuda.is_available())
    return (loader, dataset.labels[keep].astype(int),
            dataset.sexes[keep].astype(int), dataset.age_groups[keep].astype(int))


@torch.no_grad()
def sweep(model: ResNet18HyperAdaptSexAge, loader: DataLoader,
          device: torch.device) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """对 8 个 (sex, age) 组合各前向一次，返回 (feats, w_eff, bias, logits)。"""
    captured: list[torch.Tensor] = []

    def hook(_m, _i, out: torch.Tensor) -> None:
        captured.append(torch.flatten(out, 1).detach().float().cpu())

    handle = model.avgpool.register_forward_hook(hook)
    feats, logits, w_eff = [], [], []
    try:
        for sex_v, age_v in GROUPS:
            captured.clear()
            chunks = []
            for image, _label, _sex, _age in loader:
                image = image.to(device, non_blocking=True)
                n = image.shape[0]
                s_t = torch.full((n,), sex_v, dtype=torch.long, device=device)
                a_t = torch.full((n,), age_v, dtype=torch.long, device=device)
                chunks.append(model(image, s_t, a_t).squeeze(1).detach().float().cpu())
            feats.append(torch.cat(captured).numpy())
            logits.append(torch.cat(chunks).numpy())
            emb = model.patient_embed(
                torch.tensor([sex_v], device=device), torch.tensor([age_v], device=device))
            a_fc = model.fc_A_gen(emb).view(1, model.out_dim, model.rank)
            b_fc = model.fc_B_gen(emb).view(1, model.rank, 512)
            delta = torch.bmm(a_fc, b_fc)[0]
            w_eff.append((model.fc.weight + delta)[0].detach().float().cpu().numpy())
    finally:
        handle.remove()
    return (np.stack(feats), np.stack(w_eff),
            float(model.fc.bias.detach().cpu().numpy()[0]), np.stack(logits))


def main() -> None:
    ap = argparse.ArgumentParser(description="HAM10000 HyperAdapt(sex+age) 超平面扫掠")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--split_dir", type=str, default="data/splits/ham10000/cv5/fold0")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--tag", type=str, default="ham_sexage_fold0_seed42")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  ckpt={args.ckpt.name}")
    model = ResNet18HyperAdaptSexAge(
        num_classes=1, num_sex=NUM_SEX, num_age=NUM_AGE, rank=4,
        patient_embed_dim=128, pretrained=False, freeze_backbone=False)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=False),
                          strict=True)
    model = model.to(device).eval()

    loader, labels, true_sex, true_age = build_loader(
        args.split_dir, args.split, args.batch_size, args.num_workers)
    group_ids = true_sex * NUM_AGE + true_age
    feats, w_eff, bias, logits = sweep(model, loader, device)
    print(f"n={feats.shape[1]}  groups={len(GROUPS)}")

    mu = feats.reshape(-1, feats.shape[-1]).mean(axis=0)
    P_disc, diag_disc = build_discriminative_basis(w_eff)
    P_pca, diag_pca = build_pca_basis(feats.reshape(-1, feats.shape[-1]) - mu)
    U_disc = np.stack([(feats[g] - mu) @ P_disc for g in range(len(GROUPS))])
    U_pca = np.stack([(feats[g] - mu) @ P_pca for g in range(len(GROUPS))])

    aucs = np.array([roc_auc_score(labels, logits[g]) for g in range(len(GROUPS))])
    w_bar = w_eff.mean(axis=0)
    cos_to_bar = [float(w_eff[g] @ w_bar / (np.linalg.norm(w_eff[g]) * np.linalg.norm(w_bar)))
                  for g in range(len(GROUPS))]
    wn = w_eff / np.linalg.norm(w_eff, axis=1, keepdims=True)
    cos_pairwise = wn @ wn.T
    ref = feats[REF_GROUP]
    feat_shift = [float(np.linalg.norm(feats[g] - ref) / np.linalg.norm(ref))
                  for g in range(len(GROUPS))]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / f"hyperplane_arrays_{args.tag}.npz",
        w_eff=w_eff, bias=np.array(bias), mu=mu, labels=labels, group_ids=group_ids,
        logits=logits, P_discriminative=P_disc, P_pca=P_pca,
        U_discriminative=U_disc, U_pca=U_pca)
    metrics = {
        "ckpt": str(args.ckpt), "split": f"{args.split_dir}/{args.split}",
        "n_samples": int(feats.shape[1]), "malignant_rate": float(labels.mean()),
        "ref_group": REF_GROUP, "group_labels": GROUP_LABELS, "fc_bias_shared": bias,
        "per_group": {GROUP_LABELS[g]: {
            "n_true": int((group_ids == g).sum()),
            "counterfactual_auc": float(aucs[g]),
            "w_eff_norm": float(np.linalg.norm(w_eff[g])),
            "cos_to_mean_hyperplane": cos_to_bar[g],
            "mean_logit": float(logits[g].mean()),
            "feat_rel_shift_vs_ref": feat_shift[g],
            "in_plane_energy_discriminative": in_plane_energy(P_disc, w_eff[g]),
            "in_plane_energy_pca": in_plane_energy(P_pca, w_eff[g]),
        } for g in range(len(GROUPS))},
        "basis_diagnostics": {"discriminative": diag_disc, "pca": diag_pca},
        "pairwise_cosine_w_eff": cos_pairwise.tolist(),
    }
    (OUT_DIR / f"hyperplane_metrics_{args.tag}.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    off = cos_pairwise[~np.eye(len(GROUPS), dtype=bool)]
    print(f"min cosine={off.min():.3f}  cf-AUC range={aucs.min():.4f}-{aucs.max():.4f} "
          f"(span {aucs.max()-aucs.min():.4f})")
    print("已落盘 ->", OUT_DIR / f"hyperplane_metrics_{args.tag}.json")


if __name__ == "__main__":
    main()
