"""
OOD 实验 v2（MIMIC↔CheXpert）的超平面可视化 —— 统一驱动
========================================================
把 `docs/hyperplane_subgroup_visualisation.md` 的方法搬到 `docs/ood_experiment_v2_results.md`
的设定上：**source 训练的权重，评 target 的数据**，覆盖 ERM / SWAD / HyperAdapt × M→C / C→M。

每个 (direction, method) 产出三块
---------------------------------
1. **target 上的标准图**（既有渲染原样复用，只是模型来自另一个库）：
   · HyperAdapt → `hyperplane_{discriminative,pca}_*` + `delta_logit_pathway_*`（8 子群反事实扫掠）；
   · ERM / SWAD → `blind_hyperplane_*` + `blind_subgroup_panels_*`（单一超平面，无扫掠）。
2. **ID↔OOD 同帧叠加**（`ood_domain_shift_*`）——本脚本的核心新内容。超平面只由 checkpoint 决定，
   与评估数据无关 ⇒ **ID 与 OOD 的决策线逐位相同**；变的只有特征云。该图把「迁移」画成
   「点云相对一条不动的线平移了多少」，并量化 `d∥`（判别轴 = 校准侧）与 `d⊥`（不可见）。
3. **逐子群 ID→OOD 变化条形图**（`ood_subgroup_shift_*`）：左栏 `d∥`（操作点）、右栏 ΔAUC（排序）。
   两栏并列是为强制分辨 `hyperplane_subgroup_visualisation.md` §4.3 的口径 B 与口径 C。

三个参照系（读结论前必须分清）
------------------------------
| 记号 | 权重 | 数据 | 用途 |
|---|---|---|---|
| `src-ID` | source | source fold0 test | 同帧叠加的左 panel（模型不变，数据换库） |
| `OOD`    | **source** | **target fold0 test** | 本脚本现算 |
| `tgt-ID` | target | target fold0 test | 迁移落差参照（数据不变，模型换库）——从既有 npz 直接读 |

`OOD` 与 `tgt-ID` 用的是**同一批 target 图像**（同 fold、同 16 格分层抽样配额与 seed），
故两者可逐点配对比较；`src-ID` 与 `OOD` 用的是**同一份权重**。两条对照互补。

评估口径的边界（**必读**）
--------------------------
`ood_experiment_v2_results.md` 的主估计量是 **full-target × 5 fold × 3 trial = 15 replicate**；
本脚本是 **fold0 / trial 0 / seed42 的单点**，且在 target fold0 test 上做 16 格**分层**抽样
（为与既有 ID 图逐点可比）。⇒ 本脚本的任何数字**不得**用来复核 H1–H4，也不等于台账指标；
它回答的是**机制/几何**问题（迁移在这个模型的判别坐标里长什么样），不是性能问题。

运行（medimg env，纯推理，本地 GTX 1050 Ti 即可；HyperAdapt 每方向 ~15 min，blind ~2 min）：
    PYTHONPATH=. python scripts/viz_ood_hyperplane_cxr.py --direction m2c --method hyperadapt
    PYTHONPATH=. python scripts/viz_ood_hyperplane_cxr.py            # 6 个组合全跑
输出：outputs/ood_cxr/hyperplane_viz/{m2c,c2m}/
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.viz_blind_hyperplane import SPECS as BLIND_SPECS, run_blind_viz
from src.utils.hyperplane_cxr_runner import (
    CXRVizSpec, GROUP_LABELS, NUM_GROUPS, run_hyperplane_viz,
)
from src.utils.ood_hyperplane_viz import (
    assert_same_frame, domain_shift_metrics, render_domain_shift_figure,
    render_subgroup_shift_bars,
)
from src.paths import OUTPUTS_DIR

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = OUTPUTS_DIR
OOD_VIZ_ROOT = OUTPUTS_ROOT / "ood_cxr" / "hyperplane_viz"
METHODS = ("erm", "swad", "hyperadapt")


@dataclass(frozen=True)
class DomainSpec:
    """
    单个 CXR 库作为 source 或 target 时的全部定位信息。

    Attributes:
        key         : 本脚本内部键（"mimic" / "chexpert"），也是 blind SPECS 的键。
        outputs_key : outputs 下的子目录名（"mimic_cxr" / "chexpert_cxr"）。
        display_name: 图标题中的库名。
        config_path : 训练配置 YAML（提供 eval transform / image_size / age_threshold）。
        split_dir   : 仓库相对 split 目录（cv5/fold0，与 seed42 对应的 held-out）。
        id_tag      : 该库自训模型 ID 图的文件名后缀（既有产物）。
    """

    key: str
    outputs_key: str
    display_name: str
    config_path: Path
    split_dir: str
    id_tag: str


DOMAINS = {
    "mimic": DomainSpec(
        key="mimic", outputs_key="mimic_cxr", display_name="MIMIC-CXR",
        config_path=REPO_ROOT / "configs" / "mimic_cxr_baseline.yaml",
        split_dir="data/splits/mimic_cxr_nofinding/cv5/fold0", id_tag="mimic_fold0_seed42"),
    "chexpert": DomainSpec(
        key="chexpert", outputs_key="chexpert_cxr", display_name="CheXpert",
        config_path=REPO_ROOT / "configs" / "chexpert_baseline.yaml",
        split_dir="data/splits/chexpert_nofinding/cv5/fold0", id_tag="chexpert_fold0_seed42"),
}
DIRECTIONS = {"m2c": ("mimic", "chexpert"), "c2m": ("chexpert", "mimic")}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser(description="OOD 实验 v2（MIMIC↔CheXpert）超平面可视化")
    p.add_argument("--direction", type=str, default=None, choices=sorted(DIRECTIONS),
                   help="迁移方向；缺省 = 两个方向都跑。")
    p.add_argument("--method", type=str, default=None, choices=METHODS,
                   help="方法；缺省 = 三个方法都跑。")
    p.add_argument("--n_samples", type=int, default=4000,
                   help="target 上按 label×sex×race×age 共 16 格分层抽样的目标总数"
                        "（默认与既有 ID 图一致，保证同一批图像）。")
    p.add_argument("--batch_size", type=int, default=32,
                   help="HyperAdapt 逐样本卷积核显存重，默认 32；blind 内部放宽到 2 倍。")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_points", type=int, default=1400, help="散点最多绘制样本数。")
    p.add_argument("--seed", type=int, default=42,
                   help="分层抽样 / 绘图 / 置换种子。**必须与既有 ID 图一致（42）**，"
                        "否则 target 上抽到的不是同一批图像，OOD 与 tgt-ID 不再可配对。")
    p.add_argument("--rerender_bars", action="store_true",
                   help="不跑推理，只用既有 metrics JSON 重画逐子群条形图（改渲染后省一遍前向）。")
    return p.parse_args()


def resolve_ckpt(domain: DomainSpec, method: str) -> Path:
    """
    取 source 库 cv5 的 S1 选中配置在 seed42（↔ fold0）的 checkpoint。

    与 `src/training/run_ood_cxr_full_target.py::checkpoint_path` 同一寻址规则
    （SWAD = `_averaged`，其余 = `_best_overall`），保证本图用的权重与 OOD 台账一致。

    Args:
        domain: source 库 spec；method: erm / swad / hyperadapt。

    Returns:
        checkpoint 绝对路径。

    Raises:
        FileNotFoundError: 该配置的 checkpoint 不存在。
    """
    cv5 = OUTPUTS_ROOT / domain.outputs_key / "cv5"
    cfg_name = json.loads((cv5 / "selected_configs.json").read_text())["config"][method]
    suffix = "averaged" if method == "swad" else "best_overall"
    ckpt = cv5 / f"{method}_{cfg_name}_seed42_{suffix}.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"{domain.key} 的 {method}={cfg_name} checkpoint 不存在: {ckpt}")
    return ckpt


def load_reference_run(domain: DomainSpec, method: str) -> dict[str, np.ndarray]:
    """
    读某库**自训模型**在自身 fold0 test 上的既有 npz，并还原为「事实」口径的统一结构。

    HyperAdapt 的 npz 存的是 8 组**反事实**扫掠 [8, N, ·]；部署时每个样本只走**自己真实属性**
    那一条，故取 `logits[group_ids[i], i]` 得事实 logit / 事实投影坐标。blind 模型本就只有一条，
    直接用。两者因此可走同一套下游代码。

    Args:
        domain: 该 npz 所属的库；method: erm / swad / hyperadapt。

    Returns:
        {"U": [N,2] 事实投影坐标, "logits": [N] 事实 logit, "labels", "group_ids",
         "P": [512,2] 该次运行的投影基, "mu": [512] 该次运行的中心}。

    Raises:
        FileNotFoundError: 既有 ID 产物缺失（需先跑 viz_hyperadapt_hyperplane_* / viz_blind_hyperplane）。
    """
    viz_dir = OUTPUTS_ROOT / domain.outputs_key / "hyperplane_viz"
    name = (f"hyperplane_arrays_{domain.id_tag}.npz" if method == "hyperadapt"
            else f"blind_hyperplane_arrays_{method}_{domain.id_tag}.npz")
    path = viz_dir / name
    if not path.exists():
        raise FileNotFoundError(f"缺既有 ID 产物 {path}；请先跑对应的 ID 可视化脚本。")
    z = np.load(path)
    labels, group_ids = z["labels"].astype(int), z["group_ids"].astype(int)
    idx = np.arange(len(labels))
    if method == "hyperadapt":
        u_fact = z["U_discriminative"][group_ids, idx]
        logits_fact = z["logits"][group_ids, idx]
        basis = z["P_discriminative"]
    else:
        u_fact, logits_fact, basis = z["U"], z["logits"], z["P"]
    return {"U": u_fact, "logits": logits_fact, "labels": labels, "group_ids": group_ids,
            "P": basis, "mu": z["mu"]}


def natural_prevalence_readout(direction: str, method: str) -> dict[str, dict[str, float]]:
    """
    **零 GPU** 的自然患病率对照读数，直接读既有预测落盘。

    为什么必须有这一节：本脚本的几何图跑在 **16 格分层抽样**的子样本上（为保证 8 个子群都有支撑），
    而分层**按构造抹掉了 P(Y)** —— MIMIC/CheXpert 的抽样后 y=1 率被拉到 0.44–0.50，
    `ood_experiment_v2_results.md` §5.1 的「P(Y) 30.2% → 8.2%」在图里**根本看不到**。
    因此 `d∥` 在图上的读数不能直接当成部署时的操作点漂移。本函数用**未经抽样**的既有预测
    （source fold0 test vs 完整 target）给出自然患病率下的同向对照：若两者方向一致，
    图上的平移就不是分层伪影。

    Args:
        direction: "m2c" / "c2m"；method: erm / swad / hyperadapt。

    Returns:
        {"src_id": {...}, "ood": {...}, "shift": {...}}，各含 n / 患病率 / mean logit /
        平均预测概率 / 平均预测概率 − 实际患病率（患病率失配，正 = 系统性高估）/ AUC。
    """
    src, tgt = (DOMAINS[k] for k in DIRECTIONS[direction])
    cfg = json.loads((OUTPUTS_ROOT / src.outputs_key / "cv5" / "selected_configs.json")
                     .read_text())["config"][method]
    paths = {
        "src_id": (OUTPUTS_ROOT / src.outputs_key / "cv5" / "predictions"
                   / f"{method}_{cfg}_seed42_overall.npz"),
        "ood": (OUTPUTS_ROOT / "ood_cxr" / f"{src.key}2{tgt.key}" / "cv5_full_target"
                / "predictions" / f"{method}_{cfg}_seed42_overall.npz"),
    }
    out: dict[str, dict[str, float]] = {}
    for name, path in paths.items():
        z = np.load(path, allow_pickle=True)
        y, s = z["y_true"].astype(int), z["y_score"].astype(np.float64)
        prob = 1.0 / (1.0 + np.exp(-s))
        out[name] = {"n": float(len(y)), "prevalence": float(y.mean()),
                     "mean_logit": float(s.mean()), "mean_prob": float(prob.mean()),
                     "prob_minus_prevalence": float(prob.mean() - y.mean()),
                     "auc": float(roc_auc_score(y, s))}
    out["shift"] = {k: out["ood"][k] - out["src_id"][k]
                    for k in ("prevalence", "mean_logit", "mean_prob", "auc")}
    return out


def subgroup_auc(labels: np.ndarray, logits: np.ndarray, group_ids: np.ndarray) -> dict[int, float]:
    """
    逐子群的**事实** AUC（组内排序质量）。单类子群无法定义 AUC，置 nan。

    ⚠️ 这是在 16 格分层抽样的子样本上算的，**不等于** OOD 台账的 full-target 指标。
    """
    out: dict[int, float] = {}
    for g in range(NUM_GROUPS):
        sel = group_ids == g
        y = labels[sel]
        out[g] = (float(roc_auc_score(y, logits[sel]))
                  if sel.any() and 0 < y.mean() < 1 else float("nan"))
    return out


def run_ood_pass(direction: str, method: str, args: argparse.Namespace) -> dict[str, np.ndarray]:
    """
    在 target 上跑一遍 source 模型，产出 target 侧的标准图，并返回事实口径的原始数组。

    完全复用既有渲染流程（HyperAdapt 走 `run_hyperplane_viz`，ERM/SWAD 走 `run_blind_viz`），
    只把 checkpoint 换成 source 的、并在图标题里标注权重来源。

    Args:
        direction: "m2c" / "c2m"；method: erm / swad / hyperadapt；args: 全局 CLI 参数。

    Returns:
        {"feats" [N,512] 事实特征, "logits" [N] 事实 logit, "labels", "group_ids",
         "w" [G,512] 或 [1,512] 各条超平面, "bias"}。
    """
    src, tgt = (DOMAINS[k] for k in DIRECTIONS[direction])
    ckpt = resolve_ckpt(src, method)
    out_dir = OOD_VIZ_ROOT / direction
    tag = f"{direction}_{method}_fold0_seed42"
    note = f"weights trained on {src.display_name} (OOD transfer {src.key}->{tgt.key})"
    print(f"\n{'=' * 78}\n[{direction} | {method}] source={src.display_name} -> "
          f"target={tgt.display_name}\nckpt={ckpt.name}\n{'=' * 78}")

    if method == "hyperadapt":
        spec = CXRVizSpec(
            dataset_key=f"ood_{direction}", display_name=tgt.display_name,
            config_path=tgt.config_path, default_ckpt=ckpt, default_split_dir=tgt.split_dir,
            default_out_dir=out_dir, default_tag=tag, model_note=note)
        res = run_hyperplane_viz(spec, Namespace(
            ckpt=ckpt, split_dir=tgt.split_dir, split="test", n_samples=args.n_samples,
            ref_group=-1, batch_size=args.batch_size, num_workers=args.num_workers,
            max_points=min(args.max_points, 900), seed=args.seed, out_dir=out_dir, tag=tag))
        group_ids, idx = res["group_ids"], np.arange(len(res["labels"]))
        return {"feats": res["feats"][group_ids, idx], "logits": res["logits"][group_ids, idx],
                "labels": res["labels"], "group_ids": group_ids,
                "w": res["w_eff"], "bias": res["bias"]}

    # blind 渲染的文件名本就带 method，故这里的 tag 不再重复带（否则出 ..._erm_m2c_erm_...）
    res = run_blind_viz(Namespace(
        dataset=tgt.key, method=method, ckpt=ckpt, split="test", n_samples=args.n_samples,
        batch_size=2 * args.batch_size, num_workers=args.num_workers,
        max_points=args.max_points, n_perm=2000, seed=args.seed, out_dir=out_dir,
        tag=f"{direction}_fold0_seed42", model_note=note))
    return {"feats": res["feats"], "logits": res["logits"], "labels": res["labels"],
            "group_ids": res["group_ids"], "w": res["w"][None, :], "bias": res["bias"]}


def run_cell(direction: str, method: str, args: argparse.Namespace) -> dict:
    """
    跑完一个 (direction, method) 组合：target 标准图 + ID↔OOD 同帧叠加 + 逐子群条形图 + metrics。

    Args:
        direction: "m2c" / "c2m"；method: erm / swad / hyperadapt；args: 全局 CLI 参数。

    Returns:
        可写入 metrics JSON 的字典（域级位移、逐子群位移与 AUC 三参照系对照）。
    """
    src, tgt = (DOMAINS[k] for k in DIRECTIONS[direction])
    out_dir = OOD_VIZ_ROOT / direction
    tag = f"{direction}_{method}_fold0_seed42"

    ood = run_ood_pass(direction, method, args)
    src_id = load_reference_run(src, method)                # 同一权重、source 数据
    tgt_id = load_reference_run(tgt, method)                # 同一数据、target 自训权重

    # ---- 同帧：沿用 src-ID 那次运行的 (P, mu)。模型侧的轴（HyperAdapt 两轴 / blind 的 e1）
    #      只由 checkpoint 决定 ⇒ 断言与本次运行自算的基平行，即 checkpoint 一致性检查 ----
    basis, mu = src_id["P"], src_id["mu"]
    w_mean = ood["w"].mean(axis=0)
    e1_ood = w_mean / np.linalg.norm(w_mean)
    assert_same_frame(e1_ood[:, None], basis[:, :1], axes=1)

    u_ood = (ood["feats"] - mu) @ basis
    shift = domain_shift_metrics(src_id["U"], u_ood, src_id["labels"], ood["labels"],
                                 src_id["logits"], ood["logits"])
    auc_src_id = float(roc_auc_score(src_id["labels"], src_id["logits"]))
    auc_ood = float(roc_auc_score(ood["labels"], ood["logits"]))
    auc_tgt_id = float(roc_auc_score(tgt_id["labels"], tgt_id["logits"]))
    print(f"Overall AUC  src-ID={auc_src_id:.4f}  OOD={auc_ood:.4f}  tgt-ID={auc_tgt_id:.4f}")
    print(f"云位移 d∥={shift['d_along']:+.3f} SD  d⊥={shift['d_perp']:+.3f} SD  "
          f"mean logit {shift['mean_logit_id']:+.2f} -> {shift['mean_logit_ood']:+.2f}")

    method_name = {"erm": "ERM (image-only)", "swad": "SWAD (loss-valley weight averaging)",
                   "hyperadapt": "HyperAdapt (sex x race x age conditioning)"}[method]
    lines = [(basis.T @ ood["w"][g], float(ood["w"][g] @ mu + float(ood["bias"])))
             for g in range(ood["w"].shape[0])]
    render_domain_shift_figure(
        u_id=src_id["U"], u_ood=u_ood, labels_id=src_id["labels"], labels_ood=ood["labels"],
        lines=lines, mu=mu, shift=shift, auc_id=auc_src_id, auc_ood=auc_ood,
        domain_names=(f"{src.display_name} — source, ID", f"{tgt.display_name} — target, OOD"),
        plot_n=args.max_points,
        header=(f"{src.display_name} -> {tgt.display_name} | {method_name} | ONE fixed model, "
                f"two domains, one shared projection frame"),
        footer=("The hyperplane depends only on the checkpoint, NOT on the evaluation data — "
                "the line(s) are bit-identical in both panels. Only the feature cloud moves.\n"
                "d|| is a pure operating-point / calibration shift (it cannot change within-group "
                "ranking); d-perp is invisible to w. Neither explains an AUC difference.\n"
                f"fold0 / seed42 single point on a 16-cell stratified subsample — NOT the "
                f"full-target 15-replicate estimator of docs/ood_experiment_v2_results.md"),
        out_path=out_dir / f"ood_domain_shift_{tag}.png", seed=args.seed)

    # ---- 逐子群：d∥（同帧内 target 组质心 − source 组质心）与组内 AUC 变化 ----
    sd_along = float(src_id["U"][:, 0].std()) + 1e-12
    auc_by_group = {"src_id": subgroup_auc(src_id["labels"], src_id["logits"], src_id["group_ids"]),
                    "ood": subgroup_auc(ood["labels"], ood["logits"], ood["group_ids"]),
                    "tgt_id": subgroup_auc(tgt_id["labels"], tgt_id["logits"], tgt_id["group_ids"])}
    per_group: dict[int, dict[str, float]] = {}
    for g in range(NUM_GROUPS):
        in_src, in_ood = src_id["group_ids"] == g, ood["group_ids"] == g
        d_along = (float(u_ood[in_ood, 0].mean() - src_id["U"][in_src, 0].mean()) / sd_along
                   if in_src.any() and in_ood.any() else float("nan"))
        per_group[g] = {
            "d_along": d_along,
            "auc_id": auc_by_group["src_id"][g], "auc_ood": auc_by_group["ood"][g],
            "auc_tgt_id": auc_by_group["tgt_id"][g],
            "n_src_id": float(in_src.sum()), "n_ood": float(in_ood.sum()),
            "mean_logit_ood": float(ood["logits"][in_ood].mean()) if in_ood.any() else float("nan"),
        }
    render_subgroup_shift_bars(
        per_group, GROUP_LABELS, domain_names=(src.display_name, tgt.display_name),
        header=(f"{src.display_name} -> {tgt.display_name} | {method_name} | per-subgroup "
                f"ID -> OOD change"),
        out_path=out_dir / f"ood_subgroup_shift_{tag}.png")

    natural = natural_prevalence_readout(direction, method)
    print(f"自然患病率对照（未抽样）: 患病率 {natural['src_id']['prevalence']:.3f} -> "
          f"{natural['ood']['prevalence']:.3f}  mean logit {natural['src_id']['mean_logit']:+.2f} "
          f"-> {natural['ood']['mean_logit']:+.2f}  平均预测概率−患病率 "
          f"{natural['src_id']['prob_minus_prevalence']:+.3f} -> "
          f"{natural['ood']['prob_minus_prevalence']:+.3f}")

    metrics = {
        "direction": direction, "method": method,
        "source": src.outputs_key, "target": tgt.outputs_key,
        "ckpt": str(resolve_ckpt(src, method)),
        "target_split": f"{tgt.split_dir}/test", "n_ood": int(len(ood["labels"])),
        "n_src_id": int(len(src_id["labels"])),
        "overall_auc": {"src_id": auc_src_id, "ood": auc_ood, "tgt_id": auc_tgt_id},
        "domain_shift": shift,
        "natural_prevalence_control": natural,
        "per_group": {GROUP_LABELS[g]: {"index": g, **per_group[g]} for g in range(NUM_GROUPS)},
        "note": ("Figures and `domain_shift` are a fold0 / trial 0 / seed42 single point on a "
                 "16-cell stratified subsample of the target fold0 test — stratification "
                 "REMOVES the P(Y) shift, so read `natural_prevalence_control` (unsampled "
                 "source fold0 test vs full target) for the deployment operating point. "
                 "Mechanism/geometry only — NOT comparable to the full-target 15-replicate "
                 "numbers in docs/ood_experiment_v2_results.md. The hyperplane is a property "
                 "of the checkpoint alone and is identical between ID and OOD; only the "
                 "feature cloud moves."),
    }
    path = out_dir / f"ood_hyperplane_metrics_{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[saved] {path}")
    return metrics


def rerender_bars(direction: str, method: str) -> dict:
    """
    只用既有 metrics JSON 重画逐子群条形图（渲染改动后无需重跑 GPU 推理）。

    Args:
        direction: "m2c" / "c2m"；method: erm / swad / hyperadapt。

    Returns:
        读到的 metrics 字典（供 main 汇总）。
    """
    src, tgt = (DOMAINS[k] for k in DIRECTIONS[direction])
    tag = f"{direction}_{method}_fold0_seed42"
    out_dir = OOD_VIZ_ROOT / direction
    metrics = json.loads((out_dir / f"ood_hyperplane_metrics_{tag}.json").read_text())
    per_group = {v["index"]: v for v in metrics["per_group"].values()}
    method_name = {"erm": "ERM (image-only)", "swad": "SWAD (loss-valley weight averaging)",
                   "hyperadapt": "HyperAdapt (sex x race x age conditioning)"}[method]
    render_subgroup_shift_bars(
        per_group, GROUP_LABELS, domain_names=(src.display_name, tgt.display_name),
        header=(f"{src.display_name} -> {tgt.display_name} | {method_name} | per-subgroup "
                f"ID -> OOD change"),
        out_path=out_dir / f"ood_subgroup_shift_{tag}.png")
    return metrics


def main() -> None:
    """按 CLI 选择跑 1 个或全部 6 个 (direction, method) 组合，并汇总一份总表。"""
    args = parse_args()
    directions = (args.direction,) if args.direction else tuple(sorted(DIRECTIONS))
    methods = (args.method,) if args.method else METHODS

    summary = []
    for direction in directions:
        for method in methods:
            summary.append(rerender_bars(direction, method) if args.rerender_bars
                           else run_cell(direction, method, args))

    if len(summary) > 1:
        path = OOD_VIZ_ROOT / "ood_hyperplane_summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
