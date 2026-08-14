"""
DeepView (official library) decision-boundary viz for HAM10000 age-conditioned HyperAdapt
=========================================================================================
Uses the *official* DeepView library (Schulz et al., IJCAI 2020, arXiv:1909.09154;
github.com/LucaHermes/DeepView, installed as `deepview` 0.1.0) as the core engine —
Fisher discriminative distance (`calculate_fisher`), UMAP projection, and the learned
inverse mapper (`InvMapper`). The trained model is `ResNet18HyperAdaptAge`, which
conditions the whole backbone on age_group in {0..3} for HAM malignant classification.

Goal: show how the decision boundary moves across the four age groups.

Design choices (addressing the DeepView pitfalls discussed earlier):
  * SHARED coordinate system (the key one). HyperAdapt modulates the entire backbone, so
    each age is effectively a different network with a different representation. To keep
    the four panels comparable we build ONE DeepView with an anchor age (--anchor_age,
    default 1 = 40-60, the largest group): this fits the Fisher distances, the UMAP
    embedding, and the inverse mapper once. All four panels then reuse the *same* 2D point
    coordinates and the *same* inverse mapper; the only thing swapped between panels is the
    age condition used to classify the back-projected grid (`dv.model` is re-pointed at the
    age-g prediction function and the grid is re-scored). => the sole variable across panels
    is the conditioning age.
  * Threshold. Malignant prevalence is ~14%, so the sigmoid rarely exceeds 0.5 and DeepView's
    built-in argmax (0.5) background collapses to all-benign. We render the background with a
    shared Youden-J operating threshold tau (computed once on the anchor's predictions), and
    draw the p=tau iso-contour as the decision boundary.
  * O(N^2). Fisher distance is quadratic in samples; --n_samples default 120, stratified by
    (label, age_group). Prediction is micro-batched (no_grad) to bound GPU memory.
  * Synthetic inverse-mapped inputs are blurry -> background is qualitative only.
  * 2D projection is lossy / seed-dependent -> UMAP random_state fixed via --seed.
  * HAM is natural RGB (no CXR 16-bit / convert("L")); age_group==-1 filtered out.

Run (medimg env, GPU machine; inference only):
    python scripts/deepview_ham_hyperadapt_lib.py \
        --ckpt outputs/ham10000/hyperadapt_lr3e-04_wd1e-04_seed42_best_overall.pth

Output: outputs/ham10000/deepview/deepview_lib_hyperadapt_age.png (+ .npz).
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
from PIL import Image
from torchvision import transforms

from src.models.resnet18_hyperadapt import ResNet18HyperAdaptAge

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "ham10000_baseline.yaml"
DEFAULT_CKPT = (
    REPO_ROOT.parent.parent
    / "outputs" / "ham10000" / "hyperadapt_lr3e-04_wd1e-04_seed42_best_overall.pth"
)
OUTPUT_DIR = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/ham10000/deepview")

NUM_AGE = 4
AGE_LABELS = {0: "20-40", 1: "40-60", 2: "60-80", 3: "80+"}
DATA_SHAPE = (3, 224, 224)
# class colours (benign=blue, malignant=red); background & points share the semantics,
# so a point whose colour differs from the background under it is a misclassification
CLASS_COLORS = np.array([[0.13, 0.40, 0.67], [0.70, 0.09, 0.17]])
AGE_MARKERS = {0: "o", 1: "s", 2: "^", 3: "D"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Official-DeepView boundary viz (HAM HyperAdapt age)")
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                   help="HyperAdapt(age) checkpoint (plain state_dict). Default: selected config seed42.")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--split_dir", type=str, default="data/splits/ham10000",
                   help="Repo-relative split dir to sample from. For the OOF-regime cv5 fold model, point "
                        "at data/splits/ham10000/cv5/fold0 (the fold's held-out test, no leakage) and pass "
                        "the matching cv5 checkpoint via --ckpt.")
    p.add_argument("--n_samples", type=int, default=120,
                   help="Points to visualise (Fisher distance is O(N^2)); stratified by label x age.")
    p.add_argument("--anchor_age", type=int, default=1, choices=list(range(NUM_AGE)),
                   help="Age condition used to fit the shared projection/inverse (default 1=40-60).")
    p.add_argument("--fisher_n", type=int, default=5, help="DeepView 'n': #interpolations for Fisher distance.")
    p.add_argument("--lam", type=float, default=0.65,
                   help="DeepView lambda: eucl*lam + discr*(1-lam). Lower => more classification-driven.")
    p.add_argument("--resolution", type=int, default=100, help="DeepView grid resolution (background).")
    p.add_argument("--micro_batch", type=int, default=64, help="Forward micro-batch (HyperAdapt is memory-heavy).")
    p.add_argument("--seed", type=int, default=42, help="Seed (sampling + UMAP random_state).")
    p.add_argument("--threshold", type=str, default="prevalence", choices=["prevalence", "youden"],
                   help="Display threshold rule. 'prevalence' (default): tau = (1-pi) quantile of anchor "
                        "predictions -> predicted-positive rate matches malignant prevalence pi; stable & "
                        "comparable across seeds. 'youden': max(TPR-FPR); unstable under class imbalance.")
    p.add_argument("--prevalence", type=float, default=None,
                   help="Malignant prevalence pi for --threshold prevalence. Default: computed from the "
                        "age-filtered split labels (~0.155 on HAM test).")
    p.add_argument("--independent", action="store_true",
                   help="Fit a SEPARATE DeepView per age group (own Fisher/UMAP/inverse/background with "
                        "that group's conditioning) instead of shared-anchor. Shows each group-network's "
                        "full decision function incl. representation reorganisation (points move across "
                        "panels); coordinates NOT shared (compare structure, not boundary displacement).")
    p.add_argument("--out", type=Path, default=OUTPUT_DIR / "deepview_lib_hyperadapt_age.png")
    return p.parse_args()


def compute_threshold(
    p_anchor: np.ndarray, labels: np.ndarray, mode: str, prevalence: float | None,
) -> tuple[float, str]:
    """
    Pick the display threshold tau that binarises the probability background.

    prevalence (option D, simplest form): tau = pi, i.e. threshold the probability at the class prior.
        Constant, prevalence-aware, identical across seeds/panels -> the red area is not pinned but the
        cut is a fixed, interpretable value, so cross-panel movement is signal not threshold noise.
        (We deliberately do NOT use the "match predicted-positive rate to pi" quantile variant here:
        DeepView's InvMapper synthetic grid is extremely skewed toward p~=0, so that variant degenerates
        to tau~=0 and becomes hypersensitive. See also cost/Bayes threshold + F1 + DCA for deployment;
        Youden is prevalence-agnostic and unstable under imbalance.)
    youden: max(TPR - FPR) operating point (kept for reference; unstable here).
    """
    from sklearn.metrics import roc_curve

    if mode == "youden":
        fpr, tpr, thr = roc_curve(labels, p_anchor)
        return float(thr[np.argmax(tpr - fpr)]), "Youden J"
    return float(np.clip(prevalence, 1e-4, 1 - 1e-4)), f"threshold = prevalence pi={prevalence:.3f}"


def load_model(ckpt_path: Path, device: torch.device) -> ResNet18HyperAdaptAge:
    """Build ResNet18HyperAdaptAge and load a plain-state_dict checkpoint (strict)."""
    model = ResNet18HyperAdaptAge(
        num_classes=1, num_age=NUM_AGE, rank=4, patient_embed_dim=128,
        pretrained=False, freeze_backbone=False,
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def build_eval_transform() -> transforms.Compose:
    """HAM eval transform (Resize(256,256)->CenterCrop(224)->ToTensor->ImageNet Normalize)."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ev, norm = cfg["transforms"]["eval"], cfg["transforms"]["normalize"]
    return transforms.Compose([
        transforms.Resize(tuple(ev["resize"])),
        transforms.CenterCrop(ev["center_crop"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm["mean"], std=norm["std"]),
    ])


def load_samples(
    split: str, split_dir: str, n_samples: int, transform: transforms.Compose,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified (label x age_group) sample from the HAM split CSV (age_group>=0 only)."""
    import pandas as pd

    df = pd.read_csv(REPO_ROOT / split_dir / f"{split}.csv")
    df = df[df["age_group"].astype(int) >= 0].reset_index(drop=True)

    strata, picks = df.groupby(["label", "age_group"]).indices, []
    per = max(1, n_samples // max(1, len(strata)))
    for idx in strata.values():
        picks.extend(rng.choice(idx, size=min(len(idx), per), replace=False).tolist())
    if len(picks) < n_samples:
        remain = np.setdiff1d(np.arange(len(df)), np.array(picks, dtype=int))
        picks.extend(rng.choice(remain, size=min(len(remain), n_samples - len(picks)),
                                 replace=False).tolist())
    picks = picks[:n_samples]

    sub = df.iloc[picks].reset_index(drop=True)
    imgs = np.stack([
        transform(Image.open(pth).convert("RGB")).numpy() for pth in sub["image_path"].astype(str)
    ])
    return imgs, sub["label"].astype(int).to_numpy(), sub["age_group"].astype(int).to_numpy()


def make_pred_fn(model: ResNet18HyperAdaptAge, age: int, device: torch.device, micro_batch: int):
    """
    Build a DeepView-compatible prediction function for a fixed age condition.
    Takes numpy [n,3,224,224], returns 2-class probabilities [n,2]=[P(benign),P(malignant)].
    Micro-batched under no_grad so DeepView's grid/interpolation calls stay within GPU memory.
    """
    @torch.no_grad()
    def pred_fn(x: np.ndarray) -> np.ndarray:
        xt = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        p1_chunks = []
        for i in range(0, xt.shape[0], micro_batch):
            chunk = xt[i:i + micro_batch].to(device, non_blocking=True)
            age_t = torch.full((chunk.shape[0],), age, dtype=torch.long, device=device)
            p1_chunks.append(torch.sigmoid(model(chunk, age_t).squeeze(1)).cpu())
        p1 = torch.cat(p1_chunks).numpy()
        return np.stack([1.0 - p1, p1], axis=1)  # [n,2]

    return pred_fn


def render(
    emb: np.ndarray, labels: np.ndarray, ages: np.ndarray,
    grid_probs: dict[int, np.ndarray], xs: np.ndarray, ys: np.ndarray,
    anchor_age: int, auc_anchor: float, tau: float, thr_desc: str, out_path: Path,
) -> None:
    """2x2 render: shared coordinates/points, only the background age condition changes."""
    extent = (xs.min(), xs.max(), ys.min(), ys.max())
    cert_scale = float(np.percentile(
        np.abs(np.stack(list(grid_probs.values())) - tau), 95)) + 1e-8
    xx, yy = np.meshgrid(xs, ys)
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))

    for ax, g in zip(axes.ravel(), range(NUM_AGE)):
        p1 = grid_probs[g]
        certainty = np.clip(np.abs(p1 - tau) / cert_scale, 0.0, 1.0)
        pred_cls = (p1 > tau).astype(int)
        rgba = np.zeros((*p1.shape, 4))
        rgba[..., :3] = CLASS_COLORS[pred_cls]
        rgba[..., 3] = 0.25 + 0.50 * certainty
        ax.imshow(rgba, extent=extent, origin="lower", aspect="auto", interpolation="bilinear")
        ax.contour(xx, yy, p1, levels=[tau], colors="black", linewidths=1.4, linestyles="--")

        for a in range(NUM_AGE):
            m = ages == a
            if m.any():
                ax.scatter(emb[m, 0], emb[m, 1], c=CLASS_COLORS[labels[m]],
                           marker=AGE_MARKERS[a], s=48, edgecolors="white", linewidths=0.8, zorder=3)

        tag = "  (anchor: fits this coord system)" if g == anchor_age else ""
        ax.set_title(f"background age_group = {g} ({AGE_LABELS[g]}){tag}\n"
                     f"grid predicted malignant (p>tau): {float((pred_cls == 1).mean()):.0%}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    cls_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLORS[c],
                          markersize=10, label=lab) for c, lab in enumerate(["benign (0)", "malignant (1)"])]
    age_handles = [Line2D([0], [0], marker=AGE_MARKERS[a], color="w", markerfacecolor="0.4",
                          markeredgecolor="white", markersize=10, label=f"true age {AGE_LABELS[a]}")
                   for a in range(NUM_AGE)]
    fig.legend(handles=cls_handles + age_handles, loc="lower center", ncol=6, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Official DeepView | HAM10000 - HyperAdapt(age) - decision boundary across age groups\n"
        f"Shared UMAP coords + inverse mapper fit once on anchor age={anchor_age} "
        f"({AGE_LABELS[anchor_age]}); only the background's conditioning age varies | "
        f"anchor subset AUC={auc_anchor:.3f}, display threshold tau={tau:.3f} ({thr_desc})\n"
        "Background = DeepView inverse-mapped grid re-scored per age (synthetic inputs, qualitative); "
        "a point whose colour differs from its background = misclassified",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_path}")


def fit_one_age(
    model: ResNet18HyperAdaptAge, age: int, imgs: np.ndarray, labels: np.ndarray,
    n_samples: int, fisher_n: int, lam: float, resolution: int, micro_batch: int, seed: int,
    device: torch.device,
) -> dict:
    """Fit an INDEPENDENT DeepView for one age group (own Fisher/UMAP/inverse/background with f(., age))."""
    from sklearn.metrics import roc_auc_score

    pred_fn = make_pred_fn(model, age, device, micro_batch)
    dv = DeepView(pred_fn=pred_fn, classes=["benign", "malignant"], max_samples=n_samples + 5,
                  batch_size=micro_batch, data_shape=DATA_SHAPE, n=fisher_n, lam=lam,
                  resolution=resolution, cmap="RdBu_r", interactive=False,
                  title=f"DeepView age={age}", random_state=seed, n_neighbors=min(15, n_samples - 1))
    dv.add_samples(imgs, labels)
    emb = dv.embedded.copy()
    x_min, y_min, x_max, y_max = dv._get_plot_measures()
    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    grid = np.swapaxes(np.array(np.meshgrid(xs, ys)).reshape(2, -1), 0, 1)
    grid_p1 = pred_fn(dv.inverse(grid))[:, 1].reshape(resolution, resolution)
    auc = float(roc_auc_score(labels, pred_fn(imgs)[:, 1])) if len(set(labels)) == 2 else float("nan")
    return {"emb": emb, "xs": xs, "ys": ys, "grid": grid_p1, "auc": auc}


def render_independent(
    results: list[dict], labels: np.ndarray, ages: np.ndarray, tau: float, thr_desc: str,
    out_path: Path,
) -> None:
    """2x2 render, INDEPENDENT: each panel is that age group's own DeepView (own coords + background)."""
    cert_scale = float(np.percentile(
        np.abs(np.concatenate([r["grid"].ravel() for r in results]) - tau), 95)) + 1e-8
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))

    for ax, g, r in zip(axes.ravel(), range(NUM_AGE), results):
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
        emb, in_g = r["emb"], ages == g
        if (~in_g).any():
            ax.scatter(emb[~in_g, 0], emb[~in_g, 1], c=CLASS_COLORS[labels[~in_g]],
                       marker="o", s=24, alpha=0.22, edgecolors="none", zorder=2)
        if in_g.any():
            ax.scatter(emb[in_g, 0], emb[in_g, 1], c=CLASS_COLORS[labels[in_g]],
                       marker="o", s=70, edgecolors="black", linewidths=1.0, zorder=4)
        ax.set_title(f"age_group = {g} ({AGE_LABELS[g]})   own DeepView   (n={int(in_g.sum())})\n"
                     f"subset AUC={r['auc']:.3f}   grid malignant (p>tau): "
                     f"{float((pred_cls == 1).mean()):.0%}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    cls_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLORS[c],
                          markersize=10, label=lab) for c, lab in enumerate(["benign (0)", "malignant (1)"])]
    hl_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.4", markeredgecolor="black",
               markersize=11, label="in this age group"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.6", alpha=0.4, markersize=8,
               label="other age groups"),
    ]
    fig.legend(handles=cls_handles + hl_handles, loc="lower center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Official DeepView (INDEPENDENT per age group) | HAM10000 - HyperAdapt(age)\n"
        "Each panel is that age-network's OWN DeepView (own Fisher/UMAP/inverse/background) on the same "
        f"sample set | display threshold tau={tau:.3f} ({thr_desc})\n"
        "Coordinates NOT shared: compare qualitative structure (class separation, boundary shape), NOT "
        "boundary position. Points move across panels = representation reorganisation is visible.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_path}")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  ckpt={args.ckpt.name}  anchor_age={args.anchor_age}  lib=DeepView")

    model = load_model(args.ckpt, device)
    transform = build_eval_transform()
    imgs, labels, ages = load_samples(args.split, args.split_dir, args.n_samples, transform, rng)
    print(f"sampled {imgs.shape[0]} ({args.split})  labels={np.bincount(labels)}  "
          f"ages={np.bincount(ages, minlength=NUM_AGE)}")

    # prevalence pi (both modes): age-filtered split malignant rate unless overridden
    prevalence = args.prevalence
    if prevalence is None:
        import pandas as pd
        df_full = pd.read_csv(REPO_ROOT / args.split_dir / f"{args.split}.csv")
        df_full = df_full[df_full["age_group"].astype(int) >= 0]
        prevalence = float(df_full["label"].astype(int).mean())

    # ============ INDEPENDENT mode: a separate DeepView per age group ============
    if args.independent:
        tau, thr_desc = compute_threshold(np.zeros(len(labels)), labels, args.threshold, prevalence)
        print(f"INDEPENDENT mode ({NUM_AGE} DeepViews)  tau={tau:.3f}  ({thr_desc})")
        results = []
        for g in range(NUM_AGE):
            print(f"--- [{g + 1}/{NUM_AGE}] independent DeepView: age={g} ({AGE_LABELS[g]}) ---")
            r = fit_one_age(model, g, imgs, labels, args.n_samples, args.fisher_n, args.lam,
                            args.resolution, args.micro_batch, args.seed, device)
            print(f"    subset AUC={r['auc']:.3f}  grid malignant(p>tau)={float((r['grid'] > tau).mean()):.0%}")
            results.append(r)
        render_independent(results, labels, ages, tau, thr_desc, args.out)
        np.savez_compressed(
            args.out.with_suffix(".npz"), labels=labels, ages=ages, tau=tau,
            **{f"emb_age{g}": r["emb"] for g, r in enumerate(results)},
            **{f"grid_age{g}": r["grid"] for g, r in enumerate(results)},
            **{f"gx_age{g}": r["xs"] for g, r in enumerate(results)},
            **{f"gy_age{g}": r["ys"] for g, r in enumerate(results)},
        )
        print(f"[saved] {args.out.with_suffix('.npz')}")
        return

    # ---- Build DeepView with the anchor-age predictor; add_samples runs Fisher + UMAP + inverse fit ----
    dv = DeepView(
        pred_fn=make_pred_fn(model, args.anchor_age, device, args.micro_batch),
        classes=["benign", "malignant"], max_samples=args.n_samples + 5,
        batch_size=args.micro_batch, data_shape=DATA_SHAPE, n=args.fisher_n,
        lam=args.lam, resolution=args.resolution, cmap="RdBu_r",
        interactive=False, title="DeepView HAM HyperAdapt(age)",
        # UMAP config via kwargs -> deepview.embeddings.init_umap (fixed seed for reproducibility)
        random_state=args.seed, n_neighbors=min(15, args.n_samples - 1),
    )
    print("DeepView.add_samples (Fisher distance + UMAP + inverse mapper) ...")
    dv.add_samples(imgs, labels)
    emb = dv.embedded.copy()  # shared 2D coordinates across all panels

    # ---- display threshold tau from the anchor's predictions on the sampled points ----
    from sklearn.metrics import roc_auc_score
    p_anchor = make_pred_fn(model, args.anchor_age, device, args.micro_batch)(imgs)[:, 1]
    auc_anchor = float(roc_auc_score(labels, p_anchor)) if len(set(labels)) == 2 else float("nan")
    # prevalence pi: use the age-filtered split's true malignant rate unless overridden
    prevalence = args.prevalence
    if prevalence is None:
        import pandas as pd
        df_full = pd.read_csv(REPO_ROOT / args.split_dir / f"{args.split}.csv")
        df_full = df_full[df_full["age_group"].astype(int) >= 0]
        prevalence = float(df_full["label"].astype(int).mean())
    tau, thr_desc = compute_threshold(p_anchor, labels, args.threshold, prevalence)
    print(f"anchor subset AUC={auc_anchor:.3f}  tau={tau:.3f}  ({thr_desc})")

    # ---- Shared grid + DeepView's inverse mapper; re-score the SAME grid per age condition ----
    x_min, y_min, x_max, y_max = dv._get_plot_measures()
    xs = np.linspace(x_min, x_max, args.resolution)
    ys = np.linspace(y_min, y_max, args.resolution)
    grid = np.swapaxes(np.array(np.meshgrid(xs, ys)).reshape(2, -1), 0, 1)  # [res^2, 2]
    grid_samples = dv.inverse(grid)  # DeepView learned inverse: 2D -> (synthetic) image space
    print(f"inverse-mapped grid -> {grid_samples.shape}; re-scoring per age ...")

    grid_probs: dict[int, np.ndarray] = {}
    for g in range(NUM_AGE):
        p1 = make_pred_fn(model, g, device, args.micro_batch)(grid_samples)[:, 1]
        grid_probs[g] = p1.reshape(args.resolution, args.resolution)
        auc_g = roc_auc_score(labels, make_pred_fn(model, g, device, args.micro_batch)(imgs)[:, 1]) \
            if len(set(labels)) == 2 else float("nan")
        print(f"  age={g} subset AUC={auc_g:.3f}  grid malignant(p>tau)={float((grid_probs[g] > tau).mean()):.0%}")

    render(emb, labels, ages, grid_probs, xs, ys, args.anchor_age, auc_anchor, tau, thr_desc, args.out)
    np.savez_compressed(
        args.out.with_suffix(".npz"), emb=emb, labels=labels, ages=ages, p_anchor=p_anchor,
        tau=tau, grid_x=xs, grid_y=ys,
        **{f"grid_prob_age{g}": grid_probs[g] for g in range(NUM_AGE)},
    )
    print(f"[saved] {args.out.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
