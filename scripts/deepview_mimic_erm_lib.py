"""
DeepView (official library) for the ERM image-only baseline: MIMIC (ID) vs CheXpert (OOD)
=========================================================================================
Companion to scripts/deepview_mimic_hyperadapt_lib.py. The ERM model (ResNet18Pretrained, image-only)
has NO attribute conditioning, so its decision boundary does not change across subgroups — there is a
single boundary per dataset. Hence exactly two panels: the in-distribution MIMIC-CXR test and the
MIMIC->CheXpert OOD test, side by side.

Engine: official DeepView library (github.com/LucaHermes/DeepView) — Fisher discriminative distance,
UMAP projection, learned InvMapper — one independent DeepView per dataset (different data => different
projection; ERM has no subgroup conditioning so nothing is shared across panels).

Task: No Finding binary (label 1 = No Finding / healthy). MIMIC prevalence ~30.5%, CheXpert ~8.6%.
Threshold: tau = 0.305 (MIMIC operating point) for BOTH panels, so the two are directly comparable and
consistent with the HyperAdapt figures. (CheXpert's own 8.6% prevalence would flood the OOD background
to all No-Finding; the ERM model's outputs are calibrated to the ~30% training prior.)

16-bit CXR handled via MIMICCXRDataset (per-image min-max to 8-bit; NEVER convert("L")).
Inference only; ~40 min for n=110 x 2 datasets due to DeepView's numpy per-row Fisher (O(N^2)).

Run (medimg env, GPU):
    python scripts/deepview_mimic_erm_lib.py \
        --ckpt outputs/mimic_cxr/erm_lr3e-04_wd1e-03_seed42_best_overall.pth
Output: outputs/mimic_cxr/deepview/deepview_lib_erm_id_vs_ood.png (+ .npz).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from deepview import DeepView
from matplotlib.lines import Line2D
from torchvision import transforms

from src.datasets.mimic_cxr_dataset import MIMICCXRDataset
from src.models.resnet18_pretrained import ResNet18Pretrained

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "mimic_cxr_baseline.yaml"
DEFAULT_CKPT = (
    REPO_ROOT.parent.parent
    / "outputs" / "mimic_cxr" / "erm_lr3e-04_wd1e-03_seed42_best_overall.pth"
)
OUTPUT_DIR = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/mimic_cxr/deepview")

DATA_SHAPE = (3, 224, 224)
# label 0 = finding/disease -> red, label 1 = No Finding/healthy -> blue
CLASS_COLORS = np.array([[0.70, 0.09, 0.17], [0.13, 0.40, 0.67]])
# the two datasets: (name, repo-relative split_dir)
DATASETS = [
    ("MIMIC-CXR (in-distribution)", "data/splits/mimic_cxr_nofinding"),
    ("CheXpert (OOD)", "data/splits/chexpert_nofinding"),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Official-DeepView ERM baseline: MIMIC ID vs CheXpert OOD")
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                   help="ERM (image-only) checkpoint (plain state_dict). Default: selected config seed42.")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--id_split_dir", type=str, default="data/splits/mimic_cxr_nofinding",
                   help="ID split dir. For the OOF cv5 fold model use data/splits/mimic_cxr_nofinding/cv5/fold0.")
    p.add_argument("--ood_split_dir", type=str, default="data/splits/chexpert_nofinding",
                   help="OOD split dir. For OOF cv5 use data/splits/chexpert_nofinding/cv5/fold0 (external => no leakage).")
    p.add_argument("--n_samples", type=int, default=110,
                   help="Points per dataset (Fisher distance is O(N^2)); stratified by label.")
    p.add_argument("--fisher_n", type=int, default=5, help="DeepView 'n': #interpolations for Fisher distance.")
    p.add_argument("--lam", type=float, default=0.65, help="DeepView lambda: eucl*lam + discr*(1-lam).")
    p.add_argument("--resolution", type=int, default=90, help="DeepView grid resolution (background).")
    p.add_argument("--micro_batch", type=int, default=128, help="Forward micro-batch (ERM is light).")
    p.add_argument("--seed", type=int, default=42, help="Seed (sampling + UMAP random_state).")
    p.add_argument("--tau", type=float, default=0.305,
                   help="Display threshold for BOTH panels (default 0.305 = MIMIC operating point; "
                        "keeps ID and OOD comparable and matches the HyperAdapt figures).")
    p.add_argument("--out", type=Path, default=OUTPUT_DIR / "deepview_lib_erm_id_vs_ood.png")
    return p.parse_args()


def load_model(ckpt_path: Path, device: torch.device) -> ResNet18Pretrained:
    """Build the image-only ERM ResNet18Pretrained and load a plain-state_dict checkpoint (strict)."""
    model = ResNet18Pretrained(num_classes=1)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def build_eval_transform() -> transforms.Compose:
    """MIMIC/CheXpert eval transform: Grayscale(3) -> Resize(224) -> ToTensor -> ImageNet Normalize."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    size, norm = cfg["transforms"]["eval"]["resize"], cfg["transforms"]["normalize"]
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm["mean"], std=norm["std"]),
    ])


def load_samples(
    split_dir: str, split: str, n_samples: int, transform: transforms.Compose,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Class-stratified sample from a CXR split via MIMICCXRDataset. Returns (images[N,3,224,224], labels[N])."""
    ds = MIMICCXRDataset(REPO_ROOT / split_dir / f"{split}.csv",
                         transform=transform, image_size="224x224", age_threshold=60)
    labels_all = ds.labels
    picks, per = [], n_samples // 2
    for cls in (0, 1):
        idx = np.where(labels_all == cls)[0]
        picks.extend(rng.choice(idx, size=min(len(idx), per), replace=False).tolist())
    if len(picks) < n_samples:
        remain = np.setdiff1d(np.arange(len(labels_all)), np.array(picks, dtype=int))
        picks.extend(rng.choice(remain, size=min(len(remain), n_samples - len(picks)),
                                 replace=False).tolist())
    picks = picks[:n_samples]
    imgs = np.stack([ds[i][0].numpy() for i in picks])
    return imgs, labels_all[picks]


def make_pred_fn(model: ResNet18Pretrained, device: torch.device, micro_batch: int):
    """DeepView-compatible pred fn for the image-only ERM model. numpy [n,3,224,224] -> [n,2]."""
    @torch.no_grad()
    def pred_fn(x: np.ndarray) -> np.ndarray:
        xt = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        p1_chunks = []
        for i in range(0, xt.shape[0], micro_batch):
            chunk = xt[i:i + micro_batch].to(device, non_blocking=True)
            p1_chunks.append(torch.sigmoid(model(chunk).squeeze(1)).cpu())
        p1 = torch.cat(p1_chunks).numpy()
        return np.stack([1.0 - p1, p1], axis=1)

    return pred_fn


def run_dataset(
    model: ResNet18Pretrained, split_dir: str, split: str, n_samples: int, fisher_n: int,
    lam: float, resolution: int, micro_batch: int, seed: int, transform, rng, device,
) -> dict:
    """Fit one DeepView on a dataset and return its embedding, labels, grid probabilities and AUC."""
    from sklearn.metrics import roc_auc_score

    imgs, labels = load_samples(split_dir, split, n_samples, transform, rng)
    pred_fn = make_pred_fn(model, device, micro_batch)
    dv = DeepView(pred_fn=pred_fn, classes=["finding", "No Finding"], max_samples=n_samples + 5,
                  batch_size=micro_batch, data_shape=DATA_SHAPE, n=fisher_n, lam=lam,
                  resolution=resolution, cmap="RdBu_r", interactive=False,
                  title="DeepView ERM", random_state=seed, n_neighbors=min(15, n_samples - 1))
    dv.add_samples(imgs, labels)
    emb = dv.embedded.copy()

    x_min, y_min, x_max, y_max = dv._get_plot_measures()
    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    grid = np.swapaxes(np.array(np.meshgrid(xs, ys)).reshape(2, -1), 0, 1)
    grid_p1 = pred_fn(dv.inverse(grid))[:, 1].reshape(resolution, resolution)

    p_pts = pred_fn(imgs)[:, 1]
    auc = float(roc_auc_score(labels, p_pts)) if len(set(labels)) == 2 else float("nan")
    return {"emb": emb, "labels": labels, "xs": xs, "ys": ys, "grid": grid_p1, "auc": auc}


def render(results: list[dict], datasets: list[tuple[str, str]], tau: float, out_path: Path) -> None:
    """Two-panel render (ID | OOD): each its own DeepView; background split at the shared tau."""
    all_grids = np.concatenate([r["grid"].ravel() for r in results])
    cert_scale = float(np.percentile(np.abs(all_grids - tau), 95)) + 1e-8
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5))

    for ax, (name, _), r in zip(axes, datasets, results):
        p1, xs, ys = r["grid"], r["xs"], r["ys"]
        extent = (xs.min(), xs.max(), ys.min(), ys.max())
        certainty = np.clip(np.abs(p1 - tau) / cert_scale, 0.0, 1.0)
        pred_cls = (p1 > tau).astype(int)
        rgba = np.zeros((*p1.shape, 4))
        rgba[..., :3] = CLASS_COLORS[pred_cls]
        rgba[..., 3] = 0.25 + 0.50 * certainty
        ax.imshow(rgba, extent=extent, origin="lower", aspect="auto", interpolation="bilinear")
        xx, yy = np.meshgrid(xs, ys)
        ax.contour(xx, yy, p1, levels=[tau], colors="black", linewidths=1.4, linestyles="--")
        emb, labels = r["emb"], r["labels"]
        ax.scatter(emb[:, 0], emb[:, 1], c=CLASS_COLORS[labels], marker="o", s=55,
                   edgecolors="white", linewidths=0.8, zorder=3)
        ax.set_title(f"{name}\nsubset AUC={r['auc']:.3f}   grid No-Finding (p>tau): "
                     f"{float((pred_cls == 1).mean()):.0%}", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])

    cls_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLORS[c],
                          markersize=11, label=lab) for c, lab in enumerate(["finding (0)", "No Finding (1)"])]
    fig.legend(handles=cls_handles, loc="lower center", ncol=2, fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Official DeepView | ERM image-only baseline (no attribute conditioning) - single boundary per dataset\n"
        f"Independent DeepView per dataset (ERM has no subgroups to compare); shared display threshold "
        f"tau={tau:.3f} (MIMIC operating point). Background = inverse-mapped grid re-scored (synthetic, "
        "qualitative); point colour != background = misclassified",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.90))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_path}")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  ckpt={args.ckpt.name}  model=ERM(image-only)  tau={args.tau}")

    model = load_model(args.ckpt, device)
    transform = build_eval_transform()

    # (name, split_dir) for the two panels; defaults are standard splits, cv5/fold* for the OOF regime
    datasets = [
        ("MIMIC-CXR (in-distribution)", args.id_split_dir),
        ("CheXpert (OOD)", args.ood_split_dir),
    ]
    results = []
    for name, split_dir in datasets:
        print(f"--- {name} ({split_dir}) ---")
        r = run_dataset(model, split_dir, args.split, args.n_samples, args.fisher_n, args.lam,
                        args.resolution, args.micro_batch, args.seed, transform, rng, device)
        print(f"  subset AUC={r['auc']:.3f}  grid No-Finding(p>tau={args.tau})="
              f"{float((r['grid'] > args.tau).mean()):.0%}")
        results.append(r)

    render(results, datasets, args.tau, args.out)
    np.savez_compressed(
        args.out.with_suffix(".npz"), tau=args.tau,
        **{f"emb_{i}": r["emb"] for i, r in enumerate(results)},
        **{f"labels_{i}": r["labels"] for i, r in enumerate(results)},
        **{f"grid_{i}": r["grid"] for i, r in enumerate(results)},
        **{f"auc_{i}": r["auc"] for i, r in enumerate(results)},
    )
    print(f"[saved] {args.out.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
