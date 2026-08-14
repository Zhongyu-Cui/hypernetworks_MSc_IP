"""
E-audit.2 + E-audit.3：M→C full-target 效应复核（go/no-go）+ 动态-min 校准自检
==============================================================================
消费 `src/training/run_ood_cxr_full_target.py`（作业 72185）落盘的 full-target 预测，完成
docs/conditioning_ablation_plan.md 的 E-audit 后两项：

**E-audit.2（go/no-go，plan §3.1）**：历史 M→C 胜利（HyperFusion/HyperAdapt marginal worst 显著超
ERM 与 SWAD）来自 legacy **k→k** 口径，须在冻结的 **full-target** 主口径下复核：
  · **估计量**：每个 source `fold×seed` 模型评完整 target，`(f,s)` **等权平均**得配对效应。
  · **推断**：**患者级配对 cluster bootstrap**——只重采样 patient（cluster）、不重采样 seed；
    所有方法共用**同一批重采样 cluster 索引**（配对）；replicate 内对 5 个 `(f,s)` 等权平均。
  · **worst-group**：**动态 argmin**——每 replicate 在同一批 cluster 上分别重算各模型 `min_g AUC_g`
    再配对相减（候选组由 E-audit.1 冻结的 manifest 提供，全模型共用）。
  · **go/no-go 判据（预先冻结，属算力决策非结果判决）**：HyperFusion 或 HyperAdapt 中**至少一个**，
    其 full-target marginal worst 相对 ERM **与** SWAD **均为正**，**且至少一个比较的 95% CI 不含 0**
    ⇒ go（启动 E2 范围 OOD 训练）；否则 no-go（M→C 降为探索性机制跟进，不新训范围网格）。

**E-audit.3（动态-min 校准自检，plan §3.3/§4）**：**只估覆盖率/偏差，不切换主终点**。
min 算子对含噪的组 AUC 是**向下有偏**的（取最小值会系统性挑中负向噪声）。本脚本报：
  · **argmin 稳定性**：各候选组在 replicate 中当选 argmin 的频率——若某组稳定占优，动态 min ≈ 固定 min、
    偏差在**配对差值**中大部分抵消；若频繁翻转，则 min 由噪声主导、须谨慎解读。
  · **min 算子偏差**：level 侧 bootstrap 偏差 `mean_b[min*_b] − min_obs`，以及**配对差值**侧偏差
    `mean_b[Δ*_b] − Δ_obs`（后者才是确认性估计量，是解读重点）。

**性能**：cluster bootstrap 需 1000 replicate × 25 模型 × 7 指标 = 17.5 万次 AUC。逐次 `roc_auc_score`
（每次 O(n log n)、n≈138k）不可行。故用**加权 Mann–Whitney AUC**：cluster 重采样等价于给每个原样本一个
整数权重（= 其 patient 被抽中的次数），在**预先排好序**的分数上做一次前缀和即得 AUC，单次 O(n_group)、
与「真的重采样出带重复的数据集再算 AUC」**数学等价**（无并列分数时严格相等，并列率见运行时打印）。

运行（轻量 CPU，实验室机器可跑）：
    python scripts/eaudit_m2c_full_target.py --n-boot 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.training.harness.predictions import load_extra, load_predictions

OUTPUTS_ROOT = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs")
ABLATION_DIR = OUTPUTS_ROOT / "conditioning_ablation"
PRED_DIR = OUTPUTS_ROOT / "ood_cxr" / "mimic2chexpert" / "cv5_full_target" / "predictions"
MANIFEST = ABLATION_DIR / "eaudit_candidate_groups.json"

FOLD_SEEDS = (42, 43, 44, 45, 46)
METHODS = ("erm", "swad", "hyperhead", "hyperfusion", "hyperadapt")
HN_METHODS = ("hyperhead", "hyperfusion", "hyperadapt")
BASELINES = ("erm", "swad")
# go/no-go 只看深层 HN（plan §3.1 明确「HyperFusion 或 HyperAdapt」）；HyperHead 仅作展示对照
GO_NOGO_HN = ("hyperfusion", "hyperadapt")

DELTA_NI = -0.005          # Overall 非劣界值（plan §3.3）
SELECTED_CONFIG = {        # MIMIC cv5 S1 选定 config（与 run_ood_cxr_full_target 一致）
    "erm": "lr1e-04_wd1e-04", "swad": "lr1e-04_wd1e-04", "hyperhead": "lr1e-04_wd1e-04",
    "hyperfusion": "lr1e-04_wd1e-04", "hyperadapt": "lr1e-04_wd1e-03",
}


# ============================================================
# 加权 AUC（cluster bootstrap 的核心加速）
# ============================================================
def sorted_view(y: np.ndarray, score: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    预计算某子群按分数升序的视图，供加权 AUC 复用。

    Args:
        y    : [N] 0/1 标签。
        score: [N] 模型 logits。
        mask : [N] bool，该子群掩码。

    Returns:
        (idx_sorted, y_sorted)：idx_sorted 为**原始索引**（用于取每 replicate 的权重），
        y_sorted 为对应的 0/1 标签（升序分数下）。
    """
    idx = np.flatnonzero(mask)
    order = np.argsort(score[idx], kind="stable")
    idx_sorted = idx[order]
    return idx_sorted, y[idx_sorted].astype(np.float64)


def weighted_auc(y_sorted: np.ndarray, w_sorted: np.ndarray) -> float:
    """
    加权 Mann–Whitney AUC：`AUC = Σ_{i∈pos} w_i · W_neg(<i) / (W_pos · W_neg)`。

    `w` 为整数权重时，等价于把每个样本复制 w 次后算 AUC（无并列分数时严格相等）——这正是
    cluster bootstrap 的重采样语义（权重 = 该样本所属 patient 被抽中的次数）。

    Args:
        y_sorted: [n] 按分数**升序**排列的 0/1 标签。
        w_sorted: [n] 同序权重。

    Returns:
        AUC；某类权重和为 0（该 replicate 里子群只剩一类）时返回 np.nan。
    """
    w_pos = w_sorted * y_sorted
    w_neg = w_sorted - w_pos                       # y∈{0,1} ⇒ w_neg = w·(1−y)
    total_pos, total_neg = w_pos.sum(), w_neg.sum()
    if total_pos <= 0 or total_neg <= 0:
        return np.nan
    # 每个位置之前（分数更低）的累计负类权重
    cum_neg_below = np.cumsum(w_neg) - w_neg
    return float((w_pos * cum_neg_below).sum() / (total_pos * total_neg))


# ============================================================
# 载入 full-target 预测并对齐
# ============================================================
def load_all_predictions() -> tuple[np.ndarray, np.ndarray, dict, dict[tuple[str, int], np.ndarray]]:
    """
    载入 5 方法 × 5 折的 full-target 预测，并断言逐样本对齐（同一批 target、同一行序）。

    Returns:
        (y_true, patient_id, attrs, scores)：scores 键为 (method, fold)，值为 [N] logits。
    """
    y_ref: np.ndarray | None = None
    pid_ref: np.ndarray | None = None
    attrs_ref: dict | None = None
    scores: dict[tuple[str, int], np.ndarray] = {}

    for method in METHODS:
        tag = SELECTED_CONFIG[method]
        for fold, seed in enumerate(FOLD_SEEDS):
            path = PRED_DIR / f"{method}_{tag}_seed{seed}_overall.npz"
            if not path.exists():
                raise FileNotFoundError(f"缺少 full-target 预测：{path}（作业 72185 是否完成？）")
            y, s, attrs = load_predictions(path)
            pid = load_extra(path)["patient_id"]
            if y_ref is None:
                y_ref, pid_ref, attrs_ref = y, pid, attrs
            else:
                # 配对推断的前提：所有模型评的是同一批样本、同一行序
                assert np.array_equal(y, y_ref), f"{path.name} 的 y_true 与基准不一致"
                assert np.array_equal(pid, pid_ref), f"{path.name} 的 patient_id 与基准不一致"
            scores[(method, fold)] = s
    return y_ref, pid_ref, attrs_ref, scores


def build_candidate_masks(attrs: dict, manifest: dict) -> dict[str, np.ndarray]:
    """
    按 E-audit.1 冻结的 manifest 构建候选组掩码（只取通过门槛的组，全模型共用）。

    Args:
        attrs   : 预测里的属性字典（sex/race/age，age 已由 Dataset 按阈值二值化）。
        manifest: eaudit_candidate_groups.json 内容。

    Returns:
        {组名 -> [N] bool 掩码}，组名与 manifest / subgroup_auc 键一致。
    """
    names = {"sex": {0: "Male", 1: "Female"}, "race": {0: "White", 1: "Non-White"},
             "age": {0: "<60", 1: ">=60"}}
    frozen = manifest["regimes"]["M2C-OOD"]["groups"]
    masks: dict[str, np.ndarray] = {}
    for axis, mapping in names.items():
        col = np.asarray(attrs[axis]).astype(int)
        for v, nm in mapping.items():
            key = f"{axis}:{nm}"
            if not frozen[key]["passes"]:            # 未过门槛的组不进候选（本 regime 实际全过）
                continue
            masks[key] = col == v
    # 冻结的候选组必须与实际构建的一一对应，防 manifest 与代码漂移
    assert set(masks) == {k for k, v in frozen.items() if v["passes"]}, "候选组与冻结 manifest 不一致"
    return masks


# ============================================================
# 指标计算（给定权重）
# ============================================================
class MetricViews:
    """预排序视图容器：为每个 (method, fold) × {overall, 各候选组} 预存排序索引与标签。"""

    def __init__(self, y: np.ndarray, scores: dict, masks: dict[str, np.ndarray]) -> None:
        self.group_names = list(masks)
        full = np.ones(len(y), dtype=bool)
        self.views: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        for (method, fold), s in scores.items():
            self.views[(method, fold, "__overall__")] = sorted_view(y, s, full)
            for gname, m in masks.items():
                self.views[(method, fold, gname)] = sorted_view(y, s, m)

    def metrics(self, method: str, fold: int, w: np.ndarray) -> tuple[float, float, float, str]:
        """
        给定样本权重，返回该 (method, fold) 的 (Overall AUC, marginal worst AUC, gap, argmin 组名)。
        worst/best 用**动态 argmin/argmax**（在当前权重下重算各组 AUC 再取极值）；一并返回 argmin
        组名供稳定性诊断，避免二次重算各组 AUC。
        """
        idx, ys = self.views[(method, fold, "__overall__")]
        overall = weighted_auc(ys, w[idx])
        aucs: list[float] = []
        names: list[str] = []
        for gname in self.group_names:
            idx_g, y_g = self.views[(method, fold, gname)]
            a = weighted_auc(y_g, w[idx_g])
            if not np.isnan(a):
                aucs.append(a)
                names.append(gname)
        if not aucs:
            return np.nan, np.nan, np.nan, ""
        arr = np.asarray(aucs)
        i_min = int(arr.argmin())
        return overall, float(arr[i_min]), float(arr.max() - arr[i_min]), names[i_min]


def method_metrics(
    views: MetricViews, method: str, w: np.ndarray,
) -> tuple[float, float, float, list[str]]:
    """
    对 5 个 (f,s) 等权平均，得该方法的 (Overall, marginal worst, gap)——full-target 估计量；
    另返回各 fold 的 argmin 组名列表（动态-min 稳定性诊断，覆盖全部 fold 而非仅 fold0）。
    """
    rows = [views.metrics(method, f, w) for f in range(len(FOLD_SEEDS))]
    vals = np.array([r[:3] for r in rows], dtype=float)
    mean = np.nanmean(vals, axis=0)
    return float(mean[0]), float(mean[1]), float(mean[2]), [r[3] for r in rows]


# ============================================================
# 主流程
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E-audit.2/.3：M→C full-target 复核 + 动态-min 校准")
    p.add_argument("--n-boot", type=int, default=1000, help="cluster bootstrap 重复次数（默认 1000）。")
    p.add_argument("--seed", type=int, default=42, help="bootstrap 随机种子。")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(MANIFEST.read_text())
    print("E-audit.2/.3  M→C full-target 复核（go/no-go）+ 动态-min 校准自检")
    print(f"预测目录: {PRED_DIR}")

    y, pid, attrs, scores = load_all_predictions()
    masks = build_candidate_masks(attrs, manifest)
    print(f"\n完整 target: n={len(y):,}  唯一患者={len(np.unique(pid)):,}  正例率={y.mean():.4f}")
    print(f"冻结候选组（E-audit.1）: {list(masks)}")

    # 并列分数比例（加权 AUC 在无并列时与重采样 AUC 严格等价，故显式报告）
    s0 = scores[("erm", 0)]
    tie_frac = 1.0 - len(np.unique(s0)) / len(s0)
    print(f"分数并列比例（erm fold0）: {tie_frac:.2e}（≈0 ⇒ 加权 AUC 与重采样 AUC 等价）")

    views = MetricViews(y, scores, masks)
    ones = np.ones(len(y), dtype=np.float64)

    # ---------- 点估计（full-target，(f,s) 等权平均）----------
    print(f"\n{'=' * 78}\nE-audit.2  full-target 点估计（每模型评完整 target，5 个 (f,s) 等权平均）\n{'=' * 78}")
    print(f"| {'方法':<12s} | {'Overall AUC':>11s} | {'marginal worst':>14s} | {'AUC gap':>8s} | argmin 组（各 fold）")
    obs: dict[str, tuple[float, float, float]] = {}
    obs_argmin: dict[str, list[str]] = {}
    for m in METHODS:
        ovr, worst, gap, argmins = method_metrics(views, m, ones)
        obs[m] = (ovr, worst, gap)
        obs_argmin[m] = argmins
        uniq = sorted(set(argmins))
        print(f"| {m:<12s} | {ovr:>11.4f} | {worst:>14.4f} | {gap:>8.4f} | "
              f"{'/'.join(u.split(':')[-1] for u in uniq)}")

    # ---------- 患者级配对 cluster bootstrap ----------
    print(f"\n患者级配对 cluster bootstrap（B={args.n_boot}，只重采样 patient、不重采样 seed，"
          f"全方法共用同一批 cluster 索引）...")
    uniq_pid, pid_idx = np.unique(pid, return_inverse=True)
    n_clusters = len(uniq_pid)
    rng = np.random.default_rng(args.seed)

    boot: dict[str, np.ndarray] = {m: np.empty((args.n_boot, 3)) for m in METHODS}
    # argmin 稳定性：统计**全部 fold**（非仅 fold0）在各 replicate 中当选 worst 的候选组
    argmin_counts: dict[str, dict[str, int]] = {m: {g: 0 for g in masks} for m in METHODS}
    for b in range(args.n_boot):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        counts = np.bincount(draw, minlength=n_clusters).astype(np.float64)
        w = counts[pid_idx]                          # 样本权重 = 其 patient 被抽中的次数
        for m in METHODS:
            ovr, worst, gap, argmins = method_metrics(views, m, w)
            boot[m][b] = (ovr, worst, gap)
            for g in argmins:
                if g:
                    argmin_counts[m][g] += 1
        if (b + 1) % 200 == 0:
            print(f"  ... {b + 1}/{args.n_boot}")

    # ---------- 配对效应 + CI + p ----------
    print(f"\n{'=' * 78}\nE-audit.2  full-target 配对效应（HN − 基线；95% 分位 CI；单侧 p）\n{'=' * 78}")
    print(f"| {'比较':<26s} | {'Δ marginal worst':>16s} | {'95% CI':>18s} | {'p_fair':>7s} | "
          f"{'ΔOverall':>9s} | {'p_noninf':>8s} |")
    results: dict[tuple[str, str], dict] = {}
    for hn in HN_METHODS:
        for base in BASELINES:
            d_worst_obs = obs[hn][1] - obs[base][1]
            d_ovr_obs = obs[hn][0] - obs[base][0]
            d_worst_b = boot[hn][:, 1] - boot[base][:, 1]
            d_ovr_b = boot[hn][:, 0] - boot[base][:, 0]
            lo, hi = np.percentile(d_worst_b, [2.5, 97.5])
            # 零假设中心化（plan §3.3）：T* = Δ* − Δ̄_obs
            t_worst = d_worst_b - d_worst_obs
            p_fair = (1 + np.sum(t_worst >= d_worst_obs)) / (args.n_boot + 1)
            # 非劣：H0: ΔOverall ≤ δ_NI ⇒ p = P(T* ≥ Δ̄_obs − δ_NI)
            t_ovr = d_ovr_b - d_ovr_obs
            p_noninf = (1 + np.sum(t_ovr >= d_ovr_obs - DELTA_NI)) / (args.n_boot + 1)
            results[(hn, base)] = {
                "d_worst": d_worst_obs, "ci": (lo, hi), "p_fair": p_fair,
                "d_overall": d_ovr_obs, "p_noninf": p_noninf,
                "ci_excludes_0": bool(lo > 0 or hi < 0),
            }
            print(f"| {hn + ' vs ' + base:<26s} | {d_worst_obs:>+16.4f} | "
                  f"[{lo:>+7.4f},{hi:>+7.4f}] | {p_fair:>7.4f} | {d_ovr_obs:>+9.4f} | {p_noninf:>8.4f} |")

    # ---------- go/no-go 判决 ----------
    print(f"\n{'=' * 78}\nE-audit.2  OOD go/no-go 判决（plan §3.1，预先冻结）\n{'=' * 78}")
    go = False
    for hn in GO_NOGO_HN:
        pos_both = all(results[(hn, b)]["d_worst"] > 0 for b in BASELINES)
        any_ci = any(results[(hn, b)]["ci_excludes_0"] for b in BASELINES)
        verdict = pos_both and any_ci
        go = go or verdict
        print(f"  {hn:<12s}: 对 ERM 与 SWAD 均为正 = {pos_both} | 至少一个 95%CI 不含 0 = {any_ci} "
              f"⇒ {'满足' if verdict else '不满足'}")
    print(f"\n  ⇒ **{'GO' if go else 'NO-GO'}**：" + (
        "启动 E2 范围 OOD 训练（ERM/C-Head/C-Deep/C-Full）。" if go else
        "M→C 降为「资源约束下的探索性机制跟进」，**不新训范围网格**（plan §3.1）。"))

    # ---------- E-audit.3 动态-min 校准自检 ----------
    print(f"\n{'=' * 78}\nE-audit.3  动态-min 校准自检（只估覆盖率/偏差，**不切换主终点**）\n{'=' * 78}")
    n_argmin = args.n_boot * len(FOLD_SEEDS)         # 每 replicate 每 fold 各贡献一次 argmin
    print(f"\n[argmin 稳定性] 各候选组当选 worst 的频率（{args.n_boot} replicate × {len(FOLD_SEEDS)} fold）：")
    for m in METHODS:
        top = sorted(argmin_counts[m].items(), key=lambda kv: -kv[1])
        share = ", ".join(f"{g}={c / n_argmin:.1%}" for g, c in top if c > 0)
        dom = top[0][1] / n_argmin
        # 主导组占比高 ⇒ 落在校准自检的 `separated` 情景（Δ 覆盖率≈名义）；低 ⇒ 近 `tied`（Δ 保守、欠覆盖）
        regime = "separated（Δ 判决可信）" if dom >= 0.9 else (
            "介于两者之间（谨慎解读）" if dom >= 0.6 else "tied（Δ 保守/欠覆盖，慎判 null）")
        print(f"  {m:<12s}: {share}   → 主导组 {dom:.1%}，近似 {regime}")

    print("\n[min 算子偏差] level 侧（min 天然向下有偏）与**配对差值**侧（确认性估计量）：")
    print(f"  {'方法/比较':<26s} {'obs':>9s} {'boot 均值':>10s} {'偏差':>9s}")
    for m in METHODS:
        bias = boot[m][:, 1].mean() - obs[m][1]
        print(f"  {m + ' (worst level)':<26s} {obs[m][1]:>9.4f} {boot[m][:, 1].mean():>10.4f} {bias:>+9.4f}")
    for hn in GO_NOGO_HN:
        for base in BASELINES:
            d_obs = results[(hn, base)]["d_worst"]
            d_mean = (boot[hn][:, 1] - boot[base][:, 1]).mean()
            print(f"  {hn + ' vs ' + base + ' (Δ)':<26s} {d_obs:>+9.4f} {d_mean:>+10.4f} "
                  f"{d_mean - d_obs:>+9.4f}")
    print("\n  解读：level 侧偏差反映 min 算子挑负向噪声；**配对差值**侧偏差小 ⇒ 该偏差在配对中大部分抵消，"
          "\n  确认性估计量（Δ）受影响有限。argmin 若高度集中于单一组 ⇒ 动态 min ≈ 固定 min，更稳。")

    # ---------- 落盘 ----------
    out = {
        "estimator": "full-target（每 source fold×seed 评完整 target，(f,s) 等权平均）",
        "inference": f"患者级配对 cluster bootstrap B={args.n_boot}，动态 argmin，候选组见 E-audit.1 manifest",
        "n_samples": int(len(y)), "n_clusters": int(n_clusters), "delta_ni": DELTA_NI,
        "point_estimates": {m: {"overall": obs[m][0], "marginal_worst": obs[m][1], "gap": obs[m][2]}
                            for m in METHODS},
        "paired_effects": {f"{hn}_vs_{b}": {k: (list(v) if isinstance(v, tuple) else v)
                                            for k, v in results[(hn, b)].items()}
                           for hn in HN_METHODS for b in BASELINES},
        "go_nogo": {"verdict": "GO" if go else "NO-GO", "criterion": GO_NOGO_HN},
        "dynamic_min_calibration": {
            "argmin_share": {m: {g: c / n_argmin for g, c in argmin_counts[m].items() if c}
                             for m in METHODS},
            "argmin_observed_per_fold": obs_argmin,
            "level_bias": {m: float(boot[m][:, 1].mean() - obs[m][1]) for m in METHODS},
            "delta_bias": {f"{hn}_vs_{b}": float((boot[hn][:, 1] - boot[b][:, 1]).mean()
                                                 - results[(hn, b)]["d_worst"])
                           for hn in HN_METHODS for b in BASELINES},
        },
    }
    out_path = ABLATION_DIR / "eaudit_m2c_full_target_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已落盘结果 → {out_path}")


if __name__ == "__main__":
    main()
