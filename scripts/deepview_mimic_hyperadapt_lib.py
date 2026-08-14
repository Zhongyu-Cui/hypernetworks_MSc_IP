"""
DeepView (official library) decision-boundary viz for MIMIC-CXR age/sex/race-conditioned HyperAdapt
===================================================================================================
Same approach as scripts/deepview_ham_hyperadapt_lib.py, ported to MIMIC-CXR. The trained model is the
3-attribute `ResNet18HyperAdapt`, whose whole backbone is conditioned on (sex, race, age_group), each
binary -> 8 subgroups. Task: No Finding binary classification (label 1 = No Finding / healthy, ~30.5%).

Engine: official DeepView library (github.com/LucaHermes/DeepView, `deepview` 0.1.0) — Fisher
discriminative distance, UMAP projection, learned InvMapper — exactly as in the HAM script.

Goal: show how the decision boundary moves across the 8 (sex x race x age) subgroups.

Design (identical rationale to the HAM version):
  * SHARED coordinate system. HyperAdapt modulates the whole backbone, so each subgroup is a different
    network. We fit ONE DeepView with an anchor subgroup (--anchor_idx); Fisher distances, the UMAP
    embedding and the inverse mapper are fit once. All 8 panels reuse the same 2D point coordinates and
    the same inverse mapper; only the subgroup used to score the back-projected grid (`dv.model`) changes.
    => the sole variable across panels is the conditioning subgroup.
  * Threshold. No Finding prevalence ~30.5%. We render with a shared prevalence threshold tau = pi
    (option D, simplest robust form) instead of DeepView's built-in argmax (0.5). On this skewed
    synthetic-grid a "match predicted-positive rate to pi" quantile would degenerate to tau~=0, so we
    threshold at the prior directly. --threshold youden kept for reference (unstable under imbalance).
  * O(N^2) Fisher -> --n_samples default 120, stratified by (label x subgroup). Micro-batched no_grad.
  * Synthetic inverse-mapped inputs are blurry -> background qualitative only.
  * 2D projection lossy / seed-dependent -> UMAP random_state fixed via --seed.
  * MIMIC CXR PNGs are 16-bit (PIL 'I;16'); loaded via MIMICCXRDataset which does the mandatory
    per-image min-max to 8-bit (NEVER convert("L")). age_group = (age >= 60).

NOTE on expectation: MIMIC's conditional signal I(Y;A|X) is ~0 (see ledger / project memory), so the
HN gives no fairness benefit and the boundary is expected to be nearly subgroup-invariant. A flat
across-panel comparison here is the honest, predicted outcome (contrast with HAM-age's modest movement).

Run (medimg env, GPU machine; inference only; ~30 min for n=120 due to DeepView's numpy Fisher):
    python scripts/deepview_mimic_hyperadapt_lib.py \
        --ckpt outputs/mimic_cxr/hyperadapt_lr3e-04_wd1e-04_seed42_best_overall.pth
Output: outputs/mimic_cxr/deepview/deepview_lib_hyperadapt_subgroups.png (+ .npz).
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
from src.models.resnet18_hyperadapt import ResNet18HyperAdapt

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "mimic_cxr_baseline.yaml"
DEFAULT_CKPT = (
    REPO_ROOT.parent.parent
    / "outputs" / "mimic_cxr" / "hyperadapt_lr3e-04_wd1e-04_seed42_best_overall.pth"
)
OUTPUT_DIR = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/mimic_cxr/deepview")

DATA_SHAPE = (3, 224, 224)
# 8 subgroups (sex, race, age_group); sex Male=0/Female=1, race White=0/Non-White=1, age <60=0/>=60=1
SUBGROUPS = [(s, r, a) for s in (0, 1) for r in (0, 1) for a in (0, 1)]
def sg_name(sr_a: tuple[int, int, int]) -> str:
    s, r, a = sr_a
    return f"{'M' if s == 0 else 'F'} / {'White' if r == 0 else 'Non-W'} / {'<60' if a == 0 else '>=60'}"
# class colours: label 0 = finding/disease -> red, label 1 = No Finding/healthy -> blue
CLASS_COLORS = np.array([[0.70, 0.09, 0.17], [0.13, 0.40, 0.67]])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Official-DeepView boundary viz (MIMIC HyperAdapt subgroups)")
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                   help="MIMIC HyperAdapt checkpoint (plain state_dict). Default: selected config seed42.")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--split_dir", type=str, default="data/splits/mimic_cxr_nofinding",
                   help="Repo-relative split dir to sample the visualisation points from. Point at "
                        "data/splits/chexpert_nofinding for the MIMIC->CheXpert OOD test (model stays "
                        "MIMIC-trained; only the evaluated images change).")
    p.add_argument("--dataset_name", type=str, default="MIMIC-CXR (in-distribution)",
                   help="Label for titles/legend, e.g. 'CheXpert (OOD)'.")
    p.add_argument("--n_samples", type=int, default=120,
                   help="Points to visualise (Fisher distance is O(N^2)); stratified by label x subgroup.")
    p.add_argument("--anchor_idx", type=int, default=1, choices=list(range(8)),
                   help="Index into SUBGROUPS of the anchor used to fit the shared projection/inverse "
                        "(default 1 = M/White/>=60, the largest test subgroup).")
    p.add_argument("--fisher_n", type=int, default=5, help="DeepView 'n': #interpolations for Fisher distance.")
    p.add_argument("--lam", type=float, default=0.65, help="DeepView lambda: eucl*lam + discr*(1-lam).")
    p.add_argument("--resolution", type=int, default=90, help="DeepView grid resolution (background).")
    p.add_argument("--micro_batch", type=int, default=64, help="Forward micro-batch (HyperAdapt is memory-heavy).")
    p.add_argument("--seed", type=int, default=42, help="Seed (sampling + UMAP random_state).")
    p.add_argument("--threshold", type=str, default="prevalence", choices=["prevalence", "youden"],
                   help="Display threshold. 'prevalence' (default): tau = pi (No Finding prior); stable. "
                        "'youden': max(TPR-FPR); unstable under imbalance.")
    p.add_argument("--prevalence", type=float, default=None,
                   help="No Finding prevalence pi. Default: computed from the split labels (~0.305).")
    p.add_argument("--independent", action="store_true",
                   help="Fit a SEPARATE DeepView per subgroup (own Fisher distance + UMAP + inverse + "
                        "background, all with that subgroup's conditioning), instead of the shared-anchor "
                        "mode. Faithfully shows each subgroup-network's full decision function incl. its "
                        "representation reorganisation (points move across panels), at the cost of "
                        "non-shared coordinates (compare structure, not boundary displacement). ~8x cost.")
    p.add_argument("--out", type=Path, default=OUTPUT_DIR / "deepview_lib_hyperadapt_subgroups.png")
    return p.parse_args()


def load_model(ckpt_path: Path, device: torch.device) -> ResNet18HyperAdapt:
    """Build the 3-attribute ResNet18HyperAdapt and load a plain-state_dict checkpoint (strict)."""
    model = ResNet18HyperAdapt(
        num_classes=1, num_sex=2, num_race=2, num_age=2, rank=4, patient_embed_dim=128,
        pretrained=False, freeze_backbone=False,
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def build_eval_transform() -> transforms.Compose:
    """MIMIC eval transform: Grayscale(3) -> Resize(224) -> ToTensor -> ImageNet Normalize."""
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
    split: str, split_dir: str, n_samples: int, transform: transforms.Compose,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified (label x subgroup) sample from a CXR split via MIMICCXRDataset (handles 16-bit; CheXpert
    shares the schema and the 512x512->224x224 path replacement). Returns
    (images[N,3,224,224], labels[N], subgroup_idx[N] in 0..7).
    """
    ds = MIMICCXRDataset(REPO_ROOT / split_dir / f"{split}.csv",
                         transform=transform, image_size="224x224", age_threshold=60)
    sg_idx_all = np.array([SUBGROUPS.index((int(s), int(r), int(a)))
                           for s, r, a in zip(ds.sexes, ds.races, ds.age_groups)])
    labels_all = ds.labels

    # stratify by (label, subgroup) so both classes and all 8 subgroups are covered
    strata: dict[tuple[int, int], list[int]] = {}
    for i, (lab, sg) in enumerate(zip(labels_all, sg_idx_all)):
        strata.setdefault((int(lab), int(sg)), []).append(i)
    per, picks = max(1, n_samples // max(1, len(strata))), []
    for idx in strata.values():
        picks.extend(rng.choice(idx, size=min(len(idx), per), replace=False).tolist())
    if len(picks) < n_samples:
        remain = np.setdiff1d(np.arange(len(labels_all)), np.array(picks, dtype=int))
        picks.extend(rng.choice(remain, size=min(len(remain), n_samples - len(picks)),
                                 replace=False).tolist())
    picks = picks[:n_samples]

    imgs = np.stack([ds[i][0].numpy() for i in picks])
    return imgs, labels_all[picks], sg_idx_all[picks]


def make_pred_fn(model: ResNet18HyperAdapt, subgroup: tuple[int, int, int],
                 device: torch.device, micro_batch: int):
    """
    DeepView-compatible prediction function for a fixed (sex, race, age) subgroup.
    Takes numpy [n,3,224,224], returns [n,2] = [P(finding), P(No Finding)]. Micro-batched, no_grad.
    """
    s, r, a = subgroup

    @torch.no_grad()
    def pred_fn(x: np.ndarray) -> np.ndarray:
        xt = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        p1_chunks = []
        for i in range(0, xt.shape[0], micro_batch):
            chunk = xt[i:i + micro_batch].to(device, non_blocking=True)
            n = chunk.shape[0]
            sex = torch.full((n,), s, dtype=torch.long, device=device)
            race = torch.full((n,), r, dtype=torch.long, device=device)
            age = torch.full((n,), a, dtype=torch.long, device=device)
            p1_chunks.append(torch.sigmoid(model(chunk, sex, race, age).squeeze(1)).cpu())
        p1 = torch.cat(p1_chunks).numpy()  # P(No Finding)
        return np.stack([1.0 - p1, p1], axis=1)

    return pred_fn


def compute_threshold(p_anchor: np.ndarray, labels: np.ndarray, mode: str,
                      prevalence: float) -> tuple[float, str]:
    """Display threshold: prevalence -> tau = pi (prior); youden -> max(TPR-FPR). See HAM script."""
    from sklearn.metrics import roc_curve

    if mode == "youden":
        fpr, tpr, thr = roc_curve(labels, p_anchor)
        return float(thr[np.argmax(tpr - fpr)]), "Youden J"
    return float(np.clip(prevalence, 1e-4, 1 - 1e-4)), f"threshold = prevalence pi={prevalence:.3f}"


def render(
    emb: np.ndarray, labels: np.ndarray, sg_idx: np.ndarray,
    grid_probs: dict[int, np.ndarray], xs: np.ndarray, ys: np.ndarray,
    anchor_idx: int, auc_anchor: float, tau: float, thr_desc: str,
    dataset_name: str, out_path: Path,
) -> None:
    """
    2x4 render: shared coordinates/points, only the background conditioning subgroup changes.
    In each panel the points belonging to that panel's subgroup are highlighted (bold, opaque);
    the rest are faded — so each panel reads as "the personalised boundary for this subgroup + its
    own patients". Background split at tau; p1 = P(No Finding).
    """
    extent = (xs.min(), xs.max(), ys.min(), ys.max())
    cert_scale = float(np.percentile(np.abs(np.stack(list(grid_probs.values())) - tau), 95)) + 1e-8
    xx, yy = np.meshgrid(xs, ys)
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))

    for ax, g in zip(axes.ravel(), range(8)):
        p1 = grid_probs[g]
        certainty = np.clip(np.abs(p1 - tau) / cert_scale, 0.0, 1.0)
        pred_cls = (p1 > tau).astype(int)  # 1 = predicted No Finding (blue)
        rgba = np.zeros((*p1.shape, 4))
        rgba[..., :3] = CLASS_COLORS[pred_cls]
        rgba[..., 3] = 0.25 + 0.50 * certainty
        ax.imshow(rgba, extent=extent, origin="lower", aspect="auto", interpolation="bilinear")
        ax.contour(xx, yy, p1, levels=[tau], colors="black", linewidths=1.3, linestyles="--")

        in_sg = sg_idx == g
        # faded: samples from other subgroups
        if (~in_sg).any():
            ax.scatter(emb[~in_sg, 0], emb[~in_sg, 1], c=CLASS_COLORS[labels[~in_sg]],
                       marker="o", s=22, alpha=0.22, edgecolors="none", zorder=2)
        # highlighted: samples that actually belong to this panel's subgroup
        if in_sg.any():
            ax.scatter(emb[in_sg, 0], emb[in_sg, 1], c=CLASS_COLORS[labels[in_sg]],
                       marker="o", s=70, edgecolors="black", linewidths=1.0, zorder=4)

        tag = "  [anchor]" if g == anchor_idx else ""
        ax.set_title(f"{sg_name(SUBGROUPS[g])}{tag}   (n={int(in_sg.sum())})\n"
                     f"grid predicted No-Finding (p>tau): {float((pred_cls == 1).mean()):.0%}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    cls_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLORS[c],
                          markersize=10, label=lab) for c, lab in enumerate(["finding (0)", "No Finding (1)"])]
    hl_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.4", markeredgecolor="black",
               markersize=11, label="in this subgroup"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.6", alpha=0.4, markersize=8,
               label="other subgroups"),
    ]
    fig.legend(handles=cls_handles + hl_handles, loc="lower center", ncol=4, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"Official DeepView | MIMIC-trained HyperAdapt(sex,race,age) evaluated on {dataset_name} "
        "- decision boundary across 8 subgroups\n"
        f"Shared UMAP coords + inverse mapper fit once on anchor {sg_name(SUBGROUPS[anchor_idx])}; "
        f"only the background's conditioning subgroup varies | anchor subset AUC={auc_anchor:.3f}, "
        f"display threshold tau={tau:.3f} ({thr_desc})\n"
        "Background = DeepView inverse-mapped grid re-scored per subgroup (synthetic inputs, qualitative); "
        "point colour != background colour = misclassified. Expected near-invariant across subgroups (I(Y;A|X)~0)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_path}")


def fit_one_subgroup(
    model: ResNet18HyperAdapt, sg: tuple[int, int, int], imgs: np.ndarray, labels: np.ndarray,
    n_samples: int, fisher_n: int, lam: float, resolution: int, micro_batch: int, seed: int,
    device: torch.device,
) -> dict:
    """
    Fit an INDEPENDENT DeepView for one subgroup: its own Fisher distances / UMAP / inverse mapper /
    background, all computed with that subgroup's conditioning f(., sg). Returns its 2D embedding of the
    (shared) sample set, its grid probabilities, and subset AUC. Coordinates are subgroup-specific.
    """
    from sklearn.metrics import roc_auc_score

    pred_fn = make_pred_fn(model, sg, device, micro_batch)
    dv = DeepView(pred_fn=pred_fn, classes=["finding", "No Finding"], max_samples=n_samples + 5,
                  batch_size=micro_batch, data_shape=DATA_SHAPE, n=fisher_n, lam=lam,
                  resolution=resolution, cmap="RdBu_r", interactive=False,
                  title=f"DeepView {sg_name(sg)}", random_state=seed, n_neighbors=min(15, n_samples - 1))
    dv.add_samples(imgs, labels)
    emb = dv.embedded.copy()
    x_min, y_min, x_max, y_max = dv._get_plot_measures()
    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    grid = np.swapaxes(np.array(np.meshgrid(xs, ys)).reshape(2, -1), 0, 1)
    grid_p1 = pred_fn(dv.inverse(grid))[:, 1].reshape(resolution, resolution)
    p_pts = pred_fn(imgs)[:, 1]
    auc = float(roc_auc_score(labels, p_pts)) if len(set(labels)) == 2 else float("nan")
    return {"emb": emb, "xs": xs, "ys": ys, "grid": grid_p1, "auc": auc}


def render_independent(
    results: list[dict], labels: np.ndarray, sg_idx: np.ndarray, tau: float, thr_desc: str,
    dataset_name: str, out_path: Path,
) -> None:
    """
    2x4 render, INDEPENDENT mode: each panel is that subgroup's own DeepView (own coords + background).
    Points move across panels (representation reorganisation is now visible); coordinates are NOT shared,
    so compare qualitative structure (class separation, boundary shape), NOT boundary displacement.
    """
    cert_scale = float(np.percentile(
        np.abs(np.concatenate([r["grid"].ravel() for r in results]) - tau), 95)) + 1e-8
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))

    for ax, g, r in zip(axes.ravel(), range(8), results):
        p1, xs, ys = r["grid"], r["xs"], r["ys"]
        extent = (xs.min(), xs.max(), ys.min(), ys.max())
        certainty = np.clip(np.abs(p1 - tau) / cert_scale, 0.0, 1.0)
        pred_cls = (p1 > tau).astype(int)
        rgba = np.zeros((*p1.shape, 4))
        rgba[..., :3] = CLASS_COLORS[pred_cls]
        rgba[..., 3] = 0.25 + 0.50 * certainty
        ax.imshow(rgba, extent=extent, origin="lower", aspect="auto", interpolation="bilinear")
        xx, yy = np.meshgrid(xs, ys)
        ax.contour(xx, yy, p1, levels=[tau], colors="black", linewidths=1.3, linestyles="--")
        emb, in_sg = r["emb"], sg_idx == g
        if (~in_sg).any():
            ax.scatter(emb[~in_sg, 0], emb[~in_sg, 1], c=CLASS_COLORS[labels[~in_sg]],
                       marker="o", s=22, alpha=0.22, edgecolors="none", zorder=2)
        if in_sg.any():
            ax.scatter(emb[in_sg, 0], emb[in_sg, 1], c=CLASS_COLORS[labels[in_sg]],
                       marker="o", s=70, edgecolors="black", linewidths=1.0, zorder=4)
        ax.set_title(f"{sg_name(SUBGROUPS[g])}   (n={int(in_sg.sum())})   own DeepView\n"
                     f"subset AUC={r['auc']:.3f}   grid No-Finding (p>tau): "
                     f"{float((pred_cls == 1).mean()):.0%}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    cls_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLORS[c],
                          markersize=10, label=lab) for c, lab in enumerate(["finding (0)", "No Finding (1)"])]
    hl_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.4", markeredgecolor="black",
               markersize=11, label="in this subgroup"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.6", alpha=0.4, markersize=8,
               label="other subgroups"),
    ]
    fig.legend(handles=cls_handles + hl_handles, loc="lower center", ncol=4, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"Official DeepView (INDEPENDENT per subgroup) | MIMIC-trained HyperAdapt evaluated on {dataset_name}\n"
        "Each panel is that subgroup-network's OWN DeepView (own Fisher/UMAP/inverse/background) on the same "
        f"sample set | display threshold tau={tau:.3f} ({thr_desc})\n"
        "Coordinates are NOT shared across panels: compare qualitative structure (class separation, boundary "
        "shape), NOT boundary position. Points move across panels = representation reorganisation is visible.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_path}")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    anchor_sg = SUBGROUPS[args.anchor_idx]
    print(f"device={device}  ckpt={args.ckpt.name}  anchor={sg_name(anchor_sg)}  lib=DeepView")

    model = load_model(args.ckpt, device)
    transform = build_eval_transform()
    imgs, labels, sg_idx = load_samples(args.split, args.split_dir, args.n_samples, transform, rng)
    print(f"sampled {imgs.shape[0]} ({args.dataset_name} / {args.split})  labels={np.bincount(labels)}  "
          f"subgroup counts={np.bincount(sg_idx, minlength=8).tolist()}")

    # prevalence pi (both modes): true rate from the sampled split unless overridden
    prevalence = args.prevalence
    if prevalence is None:
        import pandas as pd
        df_full = pd.read_csv(REPO_ROOT / args.split_dir / f"{args.split}.csv")
        prevalence = float(df_full["label"].astype(int).mean())

    # ============ INDEPENDENT mode: a separate DeepView per subgroup ============
    if args.independent:
        tau, thr_desc = compute_threshold(np.zeros(len(labels)), labels, args.threshold, prevalence)
        print(f"INDEPENDENT mode ({len(SUBGROUPS)} DeepViews)  tau={tau:.3f}  ({thr_desc})")
        results = []
        for g, sg in enumerate(SUBGROUPS):
            print(f"--- [{g + 1}/{len(SUBGROUPS)}] independent DeepView: {sg_name(sg)} ---")
            r = fit_one_subgroup(model, sg, imgs, labels, args.n_samples, args.fisher_n, args.lam,
                                 args.resolution, args.micro_batch, args.seed, device)
            print(f"    subset AUC={r['auc']:.3f}  grid No-Finding(p>tau)={float((r['grid'] > tau).mean()):.0%}")
            results.append(r)
        render_independent(results, labels, sg_idx, tau, thr_desc, args.dataset_name, args.out)
        np.savez_compressed(
            args.out.with_suffix(".npz"), labels=labels, sg_idx=sg_idx, tau=tau,
            **{f"emb_sg{g}": r["emb"] for g, r in enumerate(results)},
            **{f"grid_sg{g}": r["grid"] for g, r in enumerate(results)},
            **{f"gx_sg{g}": r["xs"] for g, r in enumerate(results)},
            **{f"gy_sg{g}": r["ys"] for g, r in enumerate(results)},
        )
        print(f"[saved] {args.out.with_suffix('.npz')}")
        return

    # ---- Build DeepView on the anchor subgroup; add_samples runs Fisher + UMAP + inverse fit ----
    dv = DeepView(
        pred_fn=make_pred_fn(model, anchor_sg, device, args.micro_batch),
        classes=["finding", "No Finding"], max_samples=args.n_samples + 5,
        batch_size=args.micro_batch, data_shape=DATA_SHAPE, n=args.fisher_n,
        lam=args.lam, resolution=args.resolution, cmap="RdBu_r", interactive=False,
        title="DeepView MIMIC HyperAdapt", random_state=args.seed,
        n_neighbors=min(15, args.n_samples - 1),
    )
    print("DeepView.add_samples (Fisher distance + UMAP + inverse mapper) ...")
    dv.add_samples(imgs, labels)
    emb = dv.embedded.copy()

    # ---- threshold tau from the anchor predictions ----
    from sklearn.metrics import roc_auc_score
    p_anchor = make_pred_fn(model, anchor_sg, device, args.micro_batch)(imgs)[:, 1]
    auc_anchor = float(roc_auc_score(labels, p_anchor)) if len(set(labels)) == 2 else float("nan")
    prevalence = args.prevalence
    if prevalence is None:
        import pandas as pd
        df_full = pd.read_csv(REPO_ROOT / args.split_dir / f"{args.split}.csv")
        prevalence = float(df_full["label"].astype(int).mean())
    tau, thr_desc = compute_threshold(p_anchor, labels, args.threshold, prevalence)
    print(f"anchor subset AUC={auc_anchor:.3f}  tau={tau:.3f}  ({thr_desc})")

    # ---- shared grid + DeepView inverse mapper; re-score the SAME grid per subgroup ----
    x_min, y_min, x_max, y_max = dv._get_plot_measures()
    xs = np.linspace(x_min, x_max, args.resolution)
    ys = np.linspace(y_min, y_max, args.resolution)
    grid = np.swapaxes(np.array(np.meshgrid(xs, ys)).reshape(2, -1), 0, 1)
    grid_samples = dv.inverse(grid)
    print(f"inverse-mapped grid -> {grid_samples.shape}; re-scoring per subgroup ...")

    grid_probs: dict[int, np.ndarray] = {}
    for g, sg in enumerate(SUBGROUPS):
        p1 = make_pred_fn(model, sg, device, args.micro_batch)(grid_samples)[:, 1]
        grid_probs[g] = p1.reshape(args.resolution, args.resolution)
        print(f"  {sg_name(sg)}: grid No-Finding(p>tau)={float((grid_probs[g] > tau).mean()):.0%}")

    render(emb, labels, sg_idx, grid_probs, xs, ys, args.anchor_idx, auc_anchor, tau, thr_desc,
           args.dataset_name, args.out)
    np.savez_compressed(
        args.out.with_suffix(".npz"), emb=emb, labels=labels, sg_idx=sg_idx, p_anchor=p_anchor,
        tau=tau, grid_x=xs, grid_y=ys,
        **{f"grid_prob_sg{g}": grid_probs[g] for g in range(8)},
    )
    print(f"[saved] {args.out.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
