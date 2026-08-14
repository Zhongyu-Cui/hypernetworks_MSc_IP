"""
attribute-blind 基线（ERM / SWAD）的分类超平面可视化 —— 四数据集统一入口
========================================================================
与 `scripts/viz_hyperadapt_hyperplane_*.py`（HyperAdapt 逐子群版）配套的**零对照**。

为什么不能照搬 HyperAdapt 那套图
--------------------------------
ERM / SWAD 都是 image-only（SWAD = ERM 训练 + loss-valley 区间权重平均，见
`src/training/harness/swad.py`），forward **不吃** sex/race/age。因此：
  · **没有反事实属性扫掠**——同一张图只有一个特征、一个 logit，无「假设他属于 X 组」可言；
  · **只有一条超平面** `w = backbone.fc.weight`，不存在 `w_eff(a)`；
  · Δlogit 的 fc / conv 通路分解**恒为 0**（两项都需要属性引起的差分），故本脚本不做通路图；
  · `build_discriminative_basis` 会因 `w_g − w_bar ≡ 0` 直接抛错，纵轴必须换定义。

本脚本改画：**唯一超平面 + 全部样本单 panel 散点 + 逐子群 y0/y1 质心对**，纵轴取
**组间特征均值差异**的主方向（`src/utils/blind_hyperplane_viz.build_blind_basis`）。
它回答的是「一条一视同仁的超平面下，各子群天然落在哪」——HN 图里的组间位移来自条件化，
这里的组间位移**全部来自数据**。两者的位移不可混谈。

与 HyperAdapt 图逐点可比
------------------------
样本选取与 HyperAdapt 版**完全一致**（同 split、同 fold、同 seed、CXR 同 16 格分层抽样配额），
故两套图画的是同一批图像，可直接并排对照。

运行（medimg env，实验室机器即可，纯推理；四库 × 两方法共 8 次）：
    PYTHONPATH=. python scripts/viz_blind_hyperplane.py --dataset ham  --method swad
    PYTHONPATH=. python scripts/viz_blind_hyperplane.py --dataset fitz --method erm
    ...
输出：outputs/<dataset>/hyperplane_viz/blind_hyperplane_<method>_<tag>.png + metrics JSON + npz。
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

from src.datasets.fitzpatrick_dataset import FitzpatrickDataset
from src.datasets.ham10000_dataset import HAM10000Dataset
from src.models.resnet18_pretrained import ResNet18Pretrained
from src.utils.blind_hyperplane_viz import (
    build_blind_basis,
    ordinal_ramp,
    permutation_test_between_group,
    render_blind_figure,
    render_blind_subgroup_panels,
    subgroup_offsets,
)
from src.utils.hyperplane_cxr_runner import (
    GROUP_LABELS as CXR_GROUP_LABELS,
    NUM_GROUPS as CXR_NUM_GROUPS,
    build_stratified_loader,
    group_triplet,
)
from src.utils.hyperplane_viz import in_plane_energy

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs")
FEAT_DIM = 512

# CXR 8 子群：颜色只编码 race×age（4 级 ramp），sex 交给 marker 形状——
# 8 个分类色在散点（--pairs all）口径下无法通过 CVD 地板，故拆成两个通道。
CXR_SEX_MARKERS = {0: "o", 1: "^"}                          # 0=Male, 1=Female
CXR_SEX_LEGEND = [("o", "Male (marker shape)"), ("^", "Female (marker shape)")]

# 逐子群 panel 图的列数（与 HyperAdapt 版各库的排布一致，便于并排对照）
PANEL_NCOLS = {"ham": 2, "fitz": 3, "mimic": 4, "chexpert": 4}


@dataclass(frozen=True)
class BlindVizSpec:
    """
    单个数据集在本可视化中的全部差异项。

    Attributes:
        outputs_key : outputs 下的子目录名。
        display_name: 图标题中的数据集名。
        config_path : 训练配置 YAML（提供 eval transform / image_size / age_threshold）。
        split_dir   : 仓库相对 split 目录（cv5/fold0，与 seed42 checkpoint 对应的 held-out）。
        attr_name   : 分组属性在图上的名字。
        class_names : (y=0 名, y=1 名)。
        group_labels: 子群下标 -> 图例全名。
        group_short : 子群下标 -> 质心旁短标签。
        colour_note : 配色说明（该属性是否有序），写进 suptitle。
        tag         : 输出文件名后缀。
    """

    outputs_key: str
    display_name: str
    config_path: Path
    split_dir: str
    attr_name: str
    class_names: tuple[str, str]
    group_labels: dict[int, str]
    group_short: dict[int, str]
    colour_note: str
    tag: str


HAM_AGE_LABELS = {0: "20-40", 1: "40-60", 2: "60-80", 3: "80+"}
SKIN_ROMAN = {0: "I", 1: "II", 2: "III", 3: "IV", 4: "V", 5: "VI"}
# CXR 质心旁的紧凑标签：8 个质心在图上很密，用全名会互相压字
CXR_SHORT = {
    g: f"{'MF'[s]}/{'W' if r == 0 else 'NW'}/{'<60' if a == 0 else '>=60'}"
    for g, (s, r, a) in ((g, group_triplet(g)) for g in range(CXR_NUM_GROUPS))
}

SPECS: dict[str, BlindVizSpec] = {
    "ham": BlindVizSpec(
        outputs_key="ham10000", display_name="HAM10000",
        config_path=REPO_ROOT / "configs" / "ham10000_baseline.yaml",
        split_dir="data/splits/ham10000/cv5/fold0", attr_name="age_group",
        class_names=("benign", "malignant"),
        group_labels={g: f"age_group = {g} ({HAM_AGE_LABELS[g]})" for g in range(4)},
        group_short={g: HAM_AGE_LABELS[g] for g in range(4)},
        colour_note="colour = age group on a single-hue ordinal ramp (age IS an ordered "
                    "attribute, so light→dark carries real meaning)",
        tag="ham_fold0_seed42",
    ),
    "fitz": BlindVizSpec(
        outputs_key="fitzpatrick", display_name="Fitzpatrick17k",
        config_path=REPO_ROOT / "configs" / "fitzpatrick_baseline.yaml",
        split_dir="data/splits/fitzpatrick17k/cv5/fold0", attr_name="skin",
        class_names=("benign", "malignant"),
        group_labels={g: f"Fitzpatrick {SKIN_ROMAN[g]} (skin={g})" for g in range(6)},
        group_short={g: SKIN_ROMAN[g] for g in range(6)},
        colour_note="colour = Fitzpatrick skin type on a single-hue ordinal ramp (skin type IS "
                    "an ordered attribute, so light→dark carries real meaning)",
        tag="fitz_fold0_seed42",
    ),
    "mimic": BlindVizSpec(
        outputs_key="mimic_cxr", display_name="MIMIC-CXR",
        config_path=REPO_ROOT / "configs" / "mimic_cxr_baseline.yaml",
        split_dir="data/splits/mimic_cxr_nofinding/cv5/fold0", attr_name="sex x race x age",
        class_names=("finding", "no finding"),
        group_labels=dict(CXR_GROUP_LABELS), group_short=dict(CXR_SHORT),
        colour_note="colour = race x age (4-step ramp), marker shape = sex\n"
                    "(race x age is NOT an ordered quantity — the ramp only provides stable "
                    "separation; every subgroup is also directly labelled)",
        tag="mimic_fold0_seed42",
    ),
    "chexpert": BlindVizSpec(
        outputs_key="chexpert_cxr", display_name="CheXpert",
        config_path=REPO_ROOT / "configs" / "chexpert_baseline.yaml",
        split_dir="data/splits/chexpert_nofinding/cv5/fold0", attr_name="sex x race x age",
        class_names=("finding", "no finding"),
        group_labels=dict(CXR_GROUP_LABELS), group_short=dict(CXR_SHORT),
        colour_note="colour = race x age (4-step ramp), marker shape = sex\n"
                    "(race x age is NOT an ordered quantity — the ramp only provides stable "
                    "separation; every subgroup is also directly labelled)",
        tag="chexpert_fold0_seed42",
    ),
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser(
        description="attribute-blind 基线（ERM / SWAD）分类超平面的单-panel 可视化")
    p.add_argument("--dataset", type=str, required=True, choices=sorted(SPECS))
    p.add_argument("--method", type=str, required=True, choices=["swad", "erm"],
                   help="swad = loss-valley 区间平均权重；erm = best_overall checkpoint。")
    p.add_argument("--ckpt", type=Path, default=None,
                   help="显式指定 checkpoint；默认按 cv5/selected_configs.json 的选中配置解析。")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--n_samples", type=int, default=4000,
                   help="仅 CXR：按 label×sex×race×age 共 16 格分层抽样的目标总数"
                        "（与 HyperAdapt 版默认一致，保证两套图画同一批图像）。")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_points", type=int, default=1400,
                   help="散点最多绘制的样本数（仅影响绘图密度，统计仍用全部样本）。")
    p.add_argument("--n_perm", type=int, default=2000,
                   help="组间特征均值差异置换检验的置换次数（纯 numpy，秒级）。")
    p.add_argument("--seed", type=int, default=42, help="分层抽样 + 绘图抽样 + 置换种子。")
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--tag", type=str, default=None,
                   help="输出文件名后缀；缺省用 spec 自带的 tag。OOD 场景须显式区分。")
    p.add_argument("--model_note", type=str, default="",
                   help="追加到图标题的权重来源说明（OOD 场景必填，如 "
                        "'weights trained on MIMIC-CXR'）。")
    return p.parse_args()


def resolve_ckpt(spec: BlindVizSpec, method: str) -> Path:
    """
    按 `cv5/selected_configs.json` 的选中配置解析 seed42（↔ fold0）的 checkpoint 路径。

    SWAD 落地的是区间平均权重（`_averaged.pth`），ERM 用 `best_overall`——与
    `docs/oof_regime_results.md` 的评估口径一致。

    Args:
        spec: 数据集 spec。
        method: "swad" 或 "erm"。

    Returns:
        checkpoint 绝对路径。

    Raises:
        FileNotFoundError: 选中配置对应的 checkpoint 不存在。
    """
    cv5_dir = OUTPUTS_ROOT / spec.outputs_key / "cv5"
    with open(cv5_dir / "selected_configs.json", "r", encoding="utf-8") as f:
        cfg_name = json.load(f)["config"][method]
    suffix = "_averaged" if method == "swad" else "_best_overall"
    ckpt = cv5_dir / f"{method}_{cfg_name}_seed42{suffix}.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"选中配置 {method}={cfg_name} 的 checkpoint 不存在: {ckpt}")
    return ckpt


def load_blind_model(ckpt_path: Path, device: torch.device) -> ResNet18Pretrained:
    """构建 image-only ResNet18Pretrained 并严格载入 plain state_dict，切 eval。"""
    model = ResNet18Pretrained(num_classes=1, pretrained=False, freeze_backbone=False)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def build_skin_loader(
    spec: BlindVizSpec, split: str, batch_size: int, num_workers: int,
) -> tuple[DataLoader, np.ndarray, np.ndarray]:
    """
    构建皮肤镜数据集（HAM10000 / Fitzpatrick17k）的 eval DataLoader。

    两库的 eval transform 与各自训练脚本严格一致：HAM 需 Resize(256,256)+CenterCrop(224)
    （源图 600×450），Fitzpatrick 为 224 原生故只 ToTensor+Normalize。HAM 另需在内存里过滤
    `age_group >= 0`（0-20 排除组），与 HyperAdapt 版的样本集合逐位一致，保证两套图可比。

    Args:
        spec: 数据集 spec（提供 config / split_dir）。
        split: train / val / test。
        batch_size: 前向 batch。
        num_workers: DataLoader worker 数。

    Returns:
        (loader, labels, group_ids)：后两者按前向顺序（shuffle=False）逐位对应。
    """
    with open(spec.config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ev, norm = cfg["transforms"]["eval"], cfg["transforms"]["normalize"]
    csv_path = REPO_ROOT / spec.split_dir / f"{split}.csv"

    if spec.outputs_key == "ham10000":
        transform = transforms.Compose([
            transforms.Resize(tuple(ev["resize"])),
            transforms.CenterCrop(ev["center_crop"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm["mean"], std=norm["std"]),
        ])
        dataset = HAM10000Dataset(csv_path=csv_path, transform=transform)
        # AgeEmbedding 不接受 -1，HyperAdapt 版也过滤了；此处为保持样本集合一致同样过滤
        keep = np.flatnonzero(dataset.age_groups >= 0)
        loader = DataLoader(Subset(dataset, keep.tolist()), batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=torch.cuda.is_available())
        return loader, dataset.labels[keep].astype(int), dataset.age_groups[keep].astype(int)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=norm["mean"], std=norm["std"]),
    ])
    dataset = FitzpatrickDataset(csv_path=csv_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return loader, dataset.labels.astype(int), dataset.skins.astype(int)


@torch.no_grad()
def extract_features_and_logits(
    model: ResNet18Pretrained, loader: DataLoader, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    单遍前向抽取 penultimate 特征与 logit（**无属性扫掠**：属性不进模型，每张图只有一个表示）。

    特征通过 backbone.avgpool 的 forward hook 抓取，等价于 torchvision ResNet forward 里
    `torch.flatten(x, 1)` 送进 fc 的值，不改动模型代码。

    Args:
        model: 已载入权重并切 eval 的 image-only 模型。
        loader: shuffle=False 的 eval DataLoader（batch 首元素为 image）。
        device: 前向设备。

    Returns:
        (feats [N, 512], logits [N])。
    """
    captured: list[torch.Tensor] = []

    def hook(_module, _inp, out: torch.Tensor) -> None:
        captured.append(torch.flatten(out, 1).detach().float().cpu())

    handle = model.backbone.avgpool.register_forward_hook(hook)
    logit_chunks: list[torch.Tensor] = []
    try:
        for batch in loader:
            image = batch[0].to(device, non_blocking=True)
            logit_chunks.append(model(image).squeeze(1).detach().float().cpu())
    finally:
        handle.remove()
    return torch.cat(captured).numpy(), torch.cat(logit_chunks).numpy()


def build_group_style(
    dataset_key: str, n_groups: int,
) -> tuple[list[str], list[str], list[tuple[str, str]] | None]:
    """
    构造逐子群的颜色与 marker 通道。

    皮肤镜两库的分组属性（age_group / skin）是**有序**的 ⇒ 直接用单 hue ordinal ramp。
    CXR 的 8 个子群是 sex×race×age 的笛卡尔积、并非有序量：8 个分类色在散点口径下无法通过
    CVD 地板，故把 sex 拆到 marker 形状，颜色只承担 race×age 四组合。

    Args:
        dataset_key: SPECS 的键。
        n_groups: 子群数。

    Returns:
        (每子群颜色, 每子群 marker, marker 通道图例项或 None)。
    """
    if dataset_key in ("mimic", "chexpert"):
        ramp = ordinal_ramp(4)
        colors, markers = [], []
        for g in range(n_groups):
            sex, race, age = group_triplet(g)
            colors.append(ramp[race * 2 + age])             # race×age -> 4 级
            markers.append(CXR_SEX_MARKERS[sex])            # sex -> 形状
        return colors, markers, CXR_SEX_LEGEND
    return list(ordinal_ramp(n_groups)), ["o"] * n_groups, None


def run_blind_viz(args: argparse.Namespace) -> dict[str, np.ndarray]:
    """
    执行完整流程：单遍前向 -> 组间均值基 -> 单 panel + 逐子群 panel 作图 -> 落盘。

    Args:
        args: parse_args() 的解析结果，或由调用方（如 OOD 驱动脚本）构造的等价 Namespace。

    Returns:
        本次前向的原始数组（feats/logits/labels/group_ids/w/bias/P/mu）。独立作图时可忽略；
        OOD 驱动脚本用它做 ID↔OOD 同帧叠加，避免再前向一遍。
    """
    spec = SPECS[args.dataset]
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = args.ckpt if args.ckpt is not None else resolve_ckpt(spec, args.method)
    out_dir = args.out_dir or (OUTPUTS_ROOT / spec.outputs_key / "hyperplane_viz")
    tag = args.tag or spec.tag
    print(f"device={device}  dataset={spec.display_name}  method={args.method}\n"
          f"ckpt={ckpt.name}  split={spec.split_dir}/{args.split}")

    model = load_blind_model(ckpt, device)

    # ---- 数据（样本选取与 HyperAdapt 版逐位一致，两套图可直接对照）----
    if args.dataset in ("mimic", "chexpert"):
        loader, labels, group_ids, full_counts = build_stratified_loader(
            spec.config_path, spec.split_dir, args.split, args.n_samples,
            args.batch_size, args.num_workers, rng)
        n_groups = CXR_NUM_GROUPS
        print(f"16 格分层抽样：请求 n={args.n_samples}，实得 n={len(labels)}；"
              f"抽样前 split 各子群规模={full_counts.tolist()}")
    else:
        loader, labels, group_ids = build_skin_loader(
            spec, args.split, args.batch_size, args.num_workers)
        n_groups = len(spec.group_labels)
        print(f"全量 n={len(labels)}（不抽样）")

    counts = np.bincount(group_ids, minlength=n_groups)
    print(f"y=1 率={labels.mean():.3f}  逐子群 n={counts.tolist()}")

    # ---- 单遍前向：唯一特征、唯一 logit、唯一超平面 ----
    feats, logits = extract_features_and_logits(model, loader, device)
    w = model.backbone.fc.weight.detach().cpu().numpy()[0]
    bias = float(model.backbone.fc.bias.detach().cpu().numpy()[0])
    overall_auc = float(roc_auc_score(labels, logits))
    print(f"||w||={np.linalg.norm(w):.4f}  fc.bias={bias:+.4f}  overall AUC={overall_auc:.4f}")

    per_group: dict[int, dict[str, float]] = {}
    for g in range(n_groups):
        in_g = group_ids == g
        y_g = labels[in_g]
        # 单类子群无法算 AUC（CheXpert 最小格可能全 0），置 nan 并照常出图
        auc_g = (float(roc_auc_score(y_g, logits[in_g]))
                 if in_g.any() and 0 < y_g.mean() < 1 else float("nan"))
        per_group[g] = {"n": float(in_g.sum()), "auc": auc_g,
                        "mean_logit": float(logits[in_g].mean()) if in_g.any() else float("nan"),
                        "pos_rate": float(y_g.mean()) if in_g.any() else float("nan")}
        print(f"  [{g}] {spec.group_labels[g]:<26s} n={counts[g]:>5d}  AUC={auc_g:.4f}  "
              f"mean logit={per_group[g]['mean_logit']:+.3f}  "
              f"y=1 率={per_group[g]['pos_rate']:.3f}")

    # ---- 投影基：判别轴 × 组间特征均值分离轴 ----
    mu = feats.mean(axis=0)
    group_means = np.stack([feats[group_ids == g].mean(axis=0) if (group_ids == g).any()
                            else mu for g in range(n_groups)])
    P, diag = build_blind_basis(w, group_means)
    energy = in_plane_energy(P, w)
    print(f"投影基诊断={ {k: round(v, 4) for k, v in diag.items()} }  "
          f"in-plane 能量={energy:.4f}")

    U = (feats - mu) @ P
    plot_idx = np.sort(rng.choice(len(labels), size=min(args.max_points, len(labels)),
                                  replace=False))
    colors, markers, marker_legend = build_group_style(args.dataset, n_groups)

    method_name = "SWAD (ERM + loss-valley weight averaging)" if args.method == "swad" \
        else "ERM (image-only baseline)"
    sampling = ("16-cell stratified subsample, same quotas/seed as the HyperAdapt figure"
                if args.dataset in ("mimic", "chexpert") else "full split, no subsampling")
    note = f" | {args.model_note}" if args.model_note else ""
    header = (f"{spec.display_name} | {method_name}: where each {spec.attr_name} subgroup falls "
              f"relative to the ONE shared decision boundary{note}\n"
              f"n={len(labels)} ({sampling}) | overall AUC={overall_auc:.4f} | ckpt={ckpt.name}")

    render_blind_figure(
        U=U, P=P, w=w, bias=bias, mu=mu, labels=labels, group_ids=group_ids, logits=logits,
        group_labels=spec.group_labels, group_short=spec.group_short, group_colors=colors,
        group_markers=markers, marker_legend=marker_legend, per_group_stats=per_group,
        energy=energy, diag=diag, plot_idx=plot_idx, class_names=spec.class_names,
        header=header, colour_note=spec.colour_note,
        out_path=out_dir / f"blind_hyperplane_{args.method}_{tag}.png",
    )

    # ---- 逐子群 panel 版（布局对齐 HyperAdapt 图，回答「组间聚类差异是否显著」）----
    perm = permutation_test_between_group(feats, group_ids, P[:, 0], n_groups,
                                          n_perm=args.n_perm, seed=args.seed)
    offsets = subgroup_offsets(U, group_ids, n_groups)
    print(f"组间均值置换检验（B={args.n_perm}）: p_total={perm['p_total']:.4f}  "
          f"p_along_e1={perm['p_along_e1']:.4f}  p_perp={perm['p_perp']:.4f}  "
          f"正交补占组间方差 {perm['perp_share']:.1%}")
    for g in range(n_groups):
        print(f"  [{g}] {spec.group_labels[g]:<26s} d_along={offsets[g]['d_along']:+.3f} SD  "
              f"d_perp={offsets[g]['d_perp']:+.3f} SD")

    render_blind_subgroup_panels(
        U=U, P=P, w=w, bias=bias, mu=mu, labels=labels, group_ids=group_ids, logits=logits,
        group_labels=spec.group_labels, per_group_stats=per_group, offsets=offsets, perm=perm,
        diag=diag, plot_idx=plot_idx, class_names=spec.class_names,
        ncols=PANEL_NCOLS[args.dataset], header=header,
        out_path=out_dir / f"blind_subgroup_panels_{args.method}_{tag}.png",
    )

    # ---- 落盘 ----
    metrics = {
        "dataset": spec.outputs_key, "method": args.method, "ckpt": str(ckpt),
        "split": f"{spec.split_dir}/{args.split}", "n_samples": int(len(labels)),
        "attribute": spec.attr_name, "pos_rate": float(labels.mean()),
        "overall_auc": overall_auc, "w_norm": float(np.linalg.norm(w)), "fc_bias": bias,
        "in_plane_energy": energy, "basis_diagnostics": diag,
        "between_group_permutation_test": perm,
        "per_group": {spec.group_labels[g]: {"index": g, **per_group[g], **offsets[g]}
                      for g in range(n_groups)},
        "note": ("attribute-blind model: a single hyperplane shared by all subgroups; "
                 "between-group offsets are data-side, NOT model conditioning. "
                 "No counterfactual sweep and no fc/conv pathway decomposition exist here."),
    }
    metrics_path = out_dir / f"blind_hyperplane_metrics_{args.method}_{tag}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[saved] {metrics_path}")

    npz_path = out_dir / f"blind_hyperplane_arrays_{args.method}_{tag}.npz"
    np.savez_compressed(npz_path, w=w, bias=bias, mu=mu, P=P, U=U, labels=labels,
                        group_ids=group_ids, logits=logits, group_means=group_means)
    print(f"[saved] {npz_path}")

    return {"feats": feats, "logits": logits, "labels": labels, "group_ids": group_ids,
            "w": w, "bias": np.array(bias), "P": P, "mu": mu}


def main() -> None:
    """CLI 入口：解析参数后跑一遍 blind 超平面可视化。"""
    run_blind_viz(parse_args())


if __name__ == "__main__":
    main()
