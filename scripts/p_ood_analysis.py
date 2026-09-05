"""
实验 P·OOD 分析：pred-attr HyperAdapt 的跨库读数（OOD v2 口径）
==============================================================
消费 `src/training/run_ood_cxr_pred.py`（pred 臂）与 `run_ood_cxr_full_target.py`（参照臂）
落盘的 **full-target** 预测，按 `docs/ood_experiment_v2_preregistration.md` 的口径出数：

  · **估计量**：每个 source `(fold, trial)` 模型评完整 target，逐 (f,t) 等权平均（**从不跨折池化**）；
  · **主指标**：marginal worst-group AUC；次要：Overall；
  · **推断**：**患者级配对 cluster bootstrap**——只重采样 patient，所有方法共用同一批重采样索引，
    每个 replicate 内对 15 个 (f,t) 等权平均后再配对相减；
  · **H2（训练噪声）**：对配对差矩阵 `D[fold, trial]` 做双因子分解，
    σ_train = √(σ²_trial + σ²_resid)，判据 |mean D| > σ_train；
  · **多重校正**：Holm，family = {pred 臂} × {方向} 的主指标检验。

**两套 seed 编码并存且无害**：参照臂用 OOD v2 的 `42+fold+10×trial`，pred 臂用实验 P 的
`42+fold+100×trial`。配对发生在 **patient 重采样层面**（各方法各自对自己的 15 个模型平均后相减），
不按 seed 逐一配对，故编码差异不影响任何数字。

性能：用加权 Mann–Whitney AUC（预排序 + 前缀和），与「重采样出带重复的数据集再算 AUC」数学等价。

运行（轻量 CPU）：
    python scripts/p_ood_analysis.py --direction m2c --n-boot 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.eaudit_m2c_full_target import sorted_view, weighted_auc
from src.training.harness.predictions import load_extra, load_predictions
from src.training.harness.subgroup_auc import subgroup_masks

OUTPUTS_ROOT = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs")
DIRECTIONS = {"m2c": ("mimic", "chexpert"), "c2m": ("chexpert", "mimic")}
CV = 5
TRIALS = (0, 1, 2)
# 参照臂（OOD v2 既有产物）用 stride 10；实验 P 的 pred 臂用 stride 100（见模块 docstring）
STRIDE_REF, STRIDE_PRED = 10, 100
REF_METHODS = ("erm", "swad", "hyperadapt", "groupdro")
PRED_METHODS = ("hyperadapt_pred", "hyperadapt_predconst")
# 预注册的主要对比（Holm family 逐方向计）
CONTRASTS = (("hyperadapt_pred", "hyperadapt"), ("hyperadapt_pred", "erm"),
             ("hyperadapt_pred", "hyperadapt_predconst"), ("hyperadapt_pred", "swad"))


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    ap = argparse.ArgumentParser(description="实验 P·OOD 分析（full-target，OOD v2 口径）")
    ap.add_argument("--direction", choices=sorted(DIRECTIONS), default="m2c")
    ap.add_argument("--n-boot", type=int, default=1000)
    return ap.parse_args()


def seeds_for(method: str) -> list[tuple[int, int, int]]:
    """该方法的 (fold, trial, seed) 列表（两套 seed 编码见模块 docstring）。"""
    stride = STRIDE_PRED if method.startswith("hyperadapt_pred") else STRIDE_REF
    return [(f, t, 42 + f + stride * t) for t in TRIALS for f in range(CV)]


def load_method(pred_dir: Path, method: str, tag: str) -> tuple[list[tuple[int, int]], np.ndarray] | None:
    """
    载入某方法的全部 (fold,trial) full-target 预测分数。

    Args:
        pred_dir: full-target 预测目录。
        method  : 方法名。
        tag     : config_tag。

    Returns:
        (ft_keys, scores[n_model, N])；任一 replicate 缺失则跳过该 (f,t) 并在 ft_keys 中略去。
        全缺时返回 None。
    """
    ft_keys, scores = [], []
    for fold, trial, seed in seeds_for(method):
        path = pred_dir / f"{method}_{tag}_seed{seed}_overall.npz"
        if not path.exists():
            continue
        _, s, _ = load_predictions(path)
        ft_keys.append((fold, trial))
        scores.append(np.asarray(s, dtype=np.float64))
    if not scores:
        return None
    return ft_keys, np.stack(scores)


def two_way_sigma_train(values: np.ndarray, ft_keys: list[tuple[int, int]]) -> float:
    """
    对 (fold, trial) 结构的指标做双因子分解，返回训练侧噪声 σ_train = √(σ²_trial + σ²_resid)。

    与实验 G / OOD v2 的定义一致：把 fold（数据划分）方差剔除后，剩下的都算训练侧。
    数据不满足完整矩形（有缺失 replicate）时退化为「按 fold 去中心化后的残差 SD」。

    Args:
        values : [n_model] 每个 (fold,trial) 的指标值。
        ft_keys: 与 values 同序的 (fold, trial) 键。

    Returns:
        σ_train（≥0）。
    """
    folds = np.array([f for f, _ in ft_keys])
    centred = values.astype(float).copy()
    for fold in np.unique(folds):                 # 去掉 fold 主效应（数据划分方差）
        sel = folds == fold
        centred[sel] -= centred[sel].mean()
    dof = max(len(values) - len(np.unique(folds)), 1)
    return float(np.sqrt((centred ** 2).sum() / dof))


def main() -> None:
    args = parse_args()
    source, target = DIRECTIONS[args.direction]
    pred_dir = OUTPUTS_ROOT / "ood_cxr" / f"{source}2{target}" / "cv5_full_target" / "predictions"
    cfg = json.loads((OUTPUTS_ROOT / f"{source}_cxr" / "cv5" / "selected_configs.json").read_text())["config"]
    # pred 臂复用 GT-HyperAdapt 的选定 config
    tags = {m: cfg[m] for m in REF_METHODS} | {m: cfg["hyperadapt"] for m in PRED_METHODS}

    # 标签 / 分组属性 / patient_id 对所有方法相同（同一份 full-target CSV，行序一致）
    probe_path = pred_dir / f"erm_{tags['erm']}_seed42_overall.npz"
    y_true, _, attrs = load_predictions(probe_path)
    y_true = np.asarray(y_true).astype(int)
    patient_id = load_extra(probe_path)["patient_id"]
    _, cluster_idx = np.unique(patient_id, return_inverse=True)
    n_cluster = int(cluster_idx.max() + 1)

    masks = subgroup_masks("mimic", **{k: np.asarray(v).astype(int)
                                       for k, v in attrs.items() if k in ("sex", "race", "age")})
    marginal = {g: m for g, m in masks.items() if "|" not in g}
    print(f"\n{'=' * 92}\n实验 P·OOD full-target: {source} → {target}"
          f"  n={len(y_true):,}  患者={n_cluster:,}  marginal 组={len(marginal)}\n{'=' * 92}")

    loaded: dict[str, tuple[list[tuple[int, int]], np.ndarray]] = {}
    for method, tag in tags.items():
        got = load_method(pred_dir, method, tag)
        if got is None:
            print(f"  [跳过] {method}（{tag}）：无 full-target 预测")
            continue
        loaded[method] = got
        print(f"  {method:<22} replicate={len(got[0]):>2}  config={tag}")

    # 预排序视图：每个方法每个模型、每个子群各一份（bootstrap 内只做前缀和）
    views: dict[str, list[dict[str, tuple]]] = {}
    for method, (_, scores) in loaded.items():
        per_model = []
        for row in scores:
            v = {"__all__": sorted_view(y_true, row, np.ones(len(y_true), bool))}
            for g, m in marginal.items():
                v[g] = sorted_view(y_true, row, m)
            per_model.append(v)
        views[method] = per_model

    def metrics(method: str, weights: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
        """给定样本权重，算该方法的 (Overall, marginal worst) 及其逐模型向量。"""
        ov, wc = [], []
        for v in views[method]:
            idx, ys = v["__all__"]
            ov.append(weighted_auc(ys, weights[idx]))
            per_group = [weighted_auc(v[g][1], weights[v[g][0]]) for g in marginal]
            per_group = [a for a in per_group if not np.isnan(a)]
            wc.append(min(per_group) if per_group else np.nan)
        ov_arr, wc_arr = np.array(ov), np.array(wc)
        return float(np.nanmean(ov_arr)), float(np.nanmean(wc_arr)), ov_arr, wc_arr

    ones = np.ones(len(y_true))
    obs = {m: metrics(m, ones) for m in loaded}
    print(f"\n{'方法':<24}{'Overall':>12}{'marginal worst':>18}")
    for m in list(REF_METHODS) + list(PRED_METHODS):
        if m in obs:
            print(f"{m:<24}{obs[m][0]:>12.4f}{obs[m][1]:>18.4f}")

    # ---- 患者级配对 cluster bootstrap（全方法共用重采样索引）----
    rng = np.random.default_rng(42)
    boot: dict[str, list[tuple[float, float]]] = {m: [] for m in loaded}
    for _ in range(args.n_boot):
        draw = rng.integers(0, n_cluster, size=n_cluster)
        w = np.bincount(draw, minlength=n_cluster).astype(float)[cluster_idx]
        for m in loaded:
            ov, wc, _, _ = metrics(m, w)
            boot[m].append((ov, wc))
    boot_arr = {m: np.array(v) for m, v in boot.items()}

    print(f"\n{'-' * 92}\n配对对比（Δ = 前者 − 后者；CI 来自患者级配对 cluster bootstrap；"
          f"σ_train 来自 (fold,trial) 分解）\n{'-' * 92}")
    results: dict[str, dict] = {}
    p_primary: dict[str, float] = {}
    for arm, ref in CONTRASTS:
        if arm not in loaded or ref not in loaded:
            continue
        entry: dict[str, dict] = {}
        print(f"\n{arm}  vs  {ref}")
        for j, key in enumerate(("overall", "marginal_worst")):
            diff = boot_arr[arm][:, j] - boot_arr[ref][:, j]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            delta = obs[arm][j] - obs[ref][j]
            p = 2.0 * min(float((diff <= 0).mean()), float((diff >= 0).mean()))
            p = float(min(1.0, max(1.0 / args.n_boot, p)))
            # σ_train：逐 (f,t) 的配对差（两臂 replicate 数可能不同，取共有的 (fold,trial)）
            keys_a, keys_r = loaded[arm][0], loaded[ref][0]
            shared = [k for k in keys_a if k in keys_r]
            va = obs[arm][2 + j][[keys_a.index(k) for k in shared]]
            vr = obs[ref][2 + j][[keys_r.index(k) for k in shared]]
            sigma = two_way_sigma_train(va - vr, shared)
            ratio = abs(delta) / sigma if sigma > 0 else float("inf")
            entry[key] = {"delta": float(delta), "ci": [float(lo), float(hi)], "p_raw": p,
                          "sigma_train": sigma, "ratio_vs_sigma": ratio,
                          "n_paired_ft": len(shared)}
            print(f"  {key:<16} Δ={delta:+.4f}  CI=[{lo:+.4f},{hi:+.4f}]  p={p:.3f}  "
                  f"σ_train={sigma:.4f}  |Δ|/σ={ratio:.2f}  "
                  f"{'超噪声' if ratio > 1 else '不超噪声'}")
            if key == "marginal_worst":
                p_primary[f"{arm}_vs_{ref}"] = p
        results[f"{arm}_vs_{ref}"] = entry

    # ---- Holm（主指标，family = 本方向的预注册对比）----
    if p_primary:
        items = sorted(p_primary.items(), key=lambda kv: kv[1])
        running, holm = 0.0, {}
        for rank, (name, p) in enumerate(items):
            running = max(running, (len(items) - rank) * p)
            holm[name] = float(min(1.0, running))
        print(f"\n{'-' * 92}\n主指标（marginal worst）Holm 校正  family={len(p_primary)}\n{'-' * 92}")
        for name, p in p_primary.items():
            print(f"  {name:<52} p_raw={p:.3f}  p_holm={holm[name]:.3f}  "
                  f"{'显著' if holm[name] < 0.05 else 'n.s.'}")
            results[name]["marginal_worst"]["p_holm"] = holm[name]

    out = OUTPUTS_ROOT / "ood_cxr" / f"{source}2{target}" / "p_ood_results.json"
    out.write_text(json.dumps({
        "direction": args.direction, "n": int(len(y_true)), "n_clusters": n_cluster,
        "n_boot": args.n_boot, "config": tags,
        "replicates": {m: len(loaded[m][0]) for m in loaded},
        "point": {m: {"overall": obs[m][0], "marginal_worst": obs[m][1]} for m in obs},
        "contrasts": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main()
