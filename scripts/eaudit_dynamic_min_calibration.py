"""
E-audit.3：worst-group 动态-min 估计量的**覆盖率**校准自检（Monte-Carlo）
==========================================================================
docs/conditioning_ablation_plan.md §3.3/§4 要求对 worst-group 的**动态 argmin** 做校准自检，
且明确**只估覆盖率/偏差、不切换主终点**（动态 argmin 始终是确认性主分析，固定 legacy 群始终是
敏感性分析）。

`scripts/eaudit_m2c_full_target.py` 已在**真实数据**上给出「偏差」与「argmin 稳定性」；但**覆盖率**
无法在真实数据上测——真值未知。故本脚本用**已知真值的合成数据**，跑**与主分析完全相同**的
推断管线（患者级 cluster bootstrap + 动态 argmin + 百分位 CI），测其经验覆盖率。

**为什么 min 会有偏**：`min_g ÂUC_g` 中每个 `ÂUC_g` 含估计噪声，取最小值会系统性挑中**负向噪声**
最大的组 ⇒ `E[min_g ÂUC_g] < min_g AUC_g`（向下有偏）。组数越多、组内 n 越小、组间真值越接近，
偏差越大。**但确认性估计量是配对差值 Δ**：两模型在**同一批重采样 cluster** 上各自取 min，偏差
在相减时大部分抵消——本脚本就是要量化「level 侧偏多少、Δ 侧还剩多少、CI 覆盖率是否达标」。

**合成数据设计（贴合 M2C 真实结构）**：
  · 患者级 cluster：每患者多张图，**标签与分数都带患者级随机效应** ⇒ 同患者高度相关。
    标签必须按患者相关（真实 CXR 中同一患者的 No Finding 状态高度一致）——**若标签逐图独立抽，
    聚类结构名存实亡**（本脚本初版即犯此错，已修正）。
    ⚠️ **实测修正预期**：即便加上标签聚类，「样本级 bootstrap」在 **Δ 侧**的覆盖率与 cluster 版
    几乎无差（仅 level 侧 cluster 更优）。原因：配对比较中患者随机效应对两模型同样作用、相减即抵消，
    且本数据仅 ~3 图/患者、簇内相关有限。故本脚本**不**声称「样本级会显著反保守」；患者级
    cluster bootstrap 仍按 plan §3.3 作主口径（更保守、零额外代价）。
  · 二分类、低正例率（对齐 CheXpert ~8.24%）。
  · G 个候选组（患者级赋值），各组真 AUC 由 `AUC_g = Φ(μ_g/√2)` 精确控制（负类 ~N(0,1)、正类 ~N(μ_g,1)）。
  · 两个「模型」：B 相对 A 在各组有已知真效应 ⇒ 可测 Δ 的偏差与 CI 覆盖率。

**两个情景（关键）**：动态 min 的表现取决于**最弱组与次弱组的分离度**：
  · `separated`：最弱组显著低于次弱组 ⇒ argmin 稳定 ⇒ min 近似固定组、偏差在配对中抵消；
  · `tied`：改善后最弱组与次弱组**并列** ⇒ argmin 频繁翻转 ⇒ min 由噪声主导（最不利情形）。
真实数据落在哪个情景，由 `eaudit_m2c_full_target.py` 报的 **argmin 稳定性**判断——两者配合才能解读主结果。

运行（轻量 CPU；默认 R=200 trial × B=200 bootstrap，约数分钟）：
    python scripts/eaudit_dynamic_min_calibration.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from scripts.eaudit_m2c_full_target import sorted_view, weighted_auc

ABLATION_DIR = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/conditioning_ablation")

# 合成数据规模（按 M2C 真实结构缩放：CheXpert n=138,644 / 46,799 患者 ≈ 2.96 图/患者、正例率 8.24%）
N_PATIENTS = 3_000
IMGS_PER_PATIENT = 3
POS_RATE = 0.0824
N_GROUPS = 6                       # 对齐 M2C 的 6 个候选边缘组
SCORE_CLUSTER_SD = 0.5             # 分数的患者随机效应
LABEL_CLUSTER_SD = 2.0             # **标签**的患者随机效应（logit 尺度）：同患者 No Finding 状态高度相关

# 两个情景（见模块 docstring）：键 -> (各组真 AUC(A), 模型 B 相对 A 的真效应)
# · separated：B 把最弱组 0.80→0.81，次弱组 0.84 仍明显更高 ⇒ argmin 稳定；
# · tied     ：B 把最弱组 0.80→0.81，恰与次弱组 0.81 并列 ⇒ argmin 频繁翻转（最不利）。
SCENARIOS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "separated": (np.array([0.80, 0.84, 0.86, 0.88, 0.90, 0.92]),
                  np.array([0.010, 0.0, 0.0, 0.0, 0.0, 0.0])),
    "tied": (np.array([0.80, 0.81, 0.82, 0.84, 0.86, 0.88]),
             np.array([0.010, 0.0, 0.0, 0.0, 0.0, 0.0])),
}


def auc_to_mu(auc: np.ndarray) -> np.ndarray:
    """由目标 AUC 反解正类均值 μ：`AUC = Φ(μ/√2)` ⇒ `μ = √2·Φ⁻¹(AUC)`（负类 ~N(0,1)、单位方差）。"""
    return np.sqrt(2.0) * norm.ppf(auc)


def _calibrate_label_intercept(rng: np.random.Generator) -> float:
    """
    数值标定标签 logit 截距，使加入患者随机效应后**边缘**正例率 ≈ POS_RATE。

    含随机效应时 `E[sigmoid(a+u)] != sigmoid(a)`（Jensen），故不能直接用 logit(POS_RATE)，
    须数值求解，否则合成数据的正例率会偏离 CheXpert 的 8.24%。
    """
    u = rng.normal(0, LABEL_CLUSTER_SD, size=200_000)
    lo, hi = -12.0, 4.0
    for _ in range(60):                                   # 二分法：单调递增，60 次足够收敛
        mid = 0.5 * (lo + hi)
        if float(np.mean(1.0 / (1.0 + np.exp(-(mid + u))))) < POS_RATE:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def simulate_trial(
    rng: np.random.Generator, true_auc_a: np.ndarray, true_delta_b: np.ndarray, label_intercept: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成一次合成数据集（患者级 cluster：标签与分数都带患者随机效应）。

    Args:
        rng            : 随机源。
        true_auc_a     : [G] 模型 A 各组真 AUC。
        true_delta_b   : [G] 模型 B 相对 A 的真效应。
        label_intercept: 标定好的标签 logit 截距。

    Returns:
        (y, group, pid_idx, scores)：scores 形状 [2, N]（0=模型 A，1=模型 B）。
    """
    n = N_PATIENTS * IMGS_PER_PATIENT
    pid_idx = np.repeat(np.arange(N_PATIENTS), IMGS_PER_PATIENT)
    # 组按患者赋值（同患者同组，贴合 sex/race/age 的患者级恒定性）
    group = rng.integers(0, N_GROUPS, size=N_PATIENTS)[pid_idx]
    # 标签的患者随机效应：同患者的 y 高度相关（真实 CXR 结构；逐图独立会让 cluster 结构名存实亡）
    u_label = rng.normal(0, LABEL_CLUSTER_SD, size=N_PATIENTS)[pid_idx]
    p = 1.0 / (1.0 + np.exp(-(label_intercept + u_label)))
    y = (rng.random(n) < p).astype(np.int64)
    # 分数的患者随机效应
    cluster_effect = rng.normal(0, SCORE_CLUSTER_SD, size=N_PATIENTS)[pid_idx]

    mu_a = auc_to_mu(true_auc_a)
    mu_b = auc_to_mu(true_auc_a + true_delta_b)
    noise = rng.normal(0, 1, size=n)                      # 两模型共享噪声 ⇒ 天然配对（同真实配对语义）
    scores = np.empty((2, n))
    scores[0] = noise + cluster_effect + y * mu_a[group]
    scores[1] = noise + cluster_effect + y * mu_b[group]
    return y, group, pid_idx, scores


def dynamic_min(y: np.ndarray, group: np.ndarray, score: np.ndarray, w: np.ndarray) -> float:
    """给定权重，动态 argmin：重算各组 AUC 取最小（与主分析口径一致）。"""
    vals = []
    for g in range(N_GROUPS):
        idx, ys = sorted_view(y, score, group == g)
        a = weighted_auc(ys, w[idx])
        if not np.isnan(a):
            vals.append(a)
    return min(vals) if vals else np.nan


def run_trial(
    rng: np.random.Generator, n_boot: int, true_auc_a: np.ndarray, true_delta_b: np.ndarray,
    label_intercept: float, sample_level: bool = False,
) -> dict:
    """
    跑一次 trial：算点估计 + cluster(或样本级) bootstrap 的 95% 百分位 CI，返回覆盖与偏差信息。

    Args:
        rng            : 随机源。
        n_boot         : bootstrap 重复数。
        true_auc_a     : [G] 模型 A 各组真 AUC。
        true_delta_b   : [G] 模型 B 相对 A 的真效应。
        label_intercept: 标定好的标签 logit 截距。
        sample_level   : True 时改用**样本级** bootstrap（对照，用以显示忽略 cluster 的后果）。

    Returns:
        {covers_level_A, covers_delta, min_obs_A, boot_mean_min_A, delta_obs, boot_mean_delta, ...}。
    """
    y, group, pid_idx, scores = simulate_trial(rng, true_auc_a, true_delta_b, label_intercept)
    n = len(y)
    ones = np.ones(n)
    # 真值：worst-group level（模型 A）与配对差值 Δ
    true_min_a = true_auc_a.min()
    true_min_b = (true_auc_a + true_delta_b).min()
    true_delta = true_min_b - true_min_a

    min_a_obs = dynamic_min(y, group, scores[0], ones)
    min_b_obs = dynamic_min(y, group, scores[1], ones)
    delta_obs = min_b_obs - min_a_obs

    n_clusters = pid_idx.max() + 1
    boot_min_a = np.empty(n_boot)
    boot_delta = np.empty(n_boot)
    for b in range(n_boot):
        if sample_level:
            draw = rng.integers(0, n, size=n)
            w = np.bincount(draw, minlength=n).astype(float)
        else:
            draw = rng.integers(0, n_clusters, size=n_clusters)
            w = np.bincount(draw, minlength=n_clusters).astype(float)[pid_idx]
        ma = dynamic_min(y, group, scores[0], w)
        mb = dynamic_min(y, group, scores[1], w)
        boot_min_a[b] = ma
        boot_delta[b] = mb - ma

    lo_a, hi_a = np.percentile(boot_min_a, [2.5, 97.5])
    lo_d, hi_d = np.percentile(boot_delta, [2.5, 97.5])
    return {
        "covers_level_A": bool(lo_a <= true_min_a <= hi_a),
        "covers_delta": bool(lo_d <= true_delta <= hi_d),
        "min_obs_A": min_a_obs, "true_min_A": true_min_a,
        "delta_obs": delta_obs, "true_delta": true_delta,
        "boot_mean_min_A": float(boot_min_a.mean()),
        "boot_mean_delta": float(boot_delta.mean()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E-audit.3：动态-min 覆盖率/偏差校准（Monte-Carlo）")
    p.add_argument("--n-trials", type=int, default=200, help="Monte-Carlo trial 数（默认 200）。")
    p.add_argument("--n-boot", type=int, default=200, help="每 trial 的 bootstrap 数（默认 200）。")
    p.add_argument("--seed", type=int, default=0, help="随机种子。")
    return p.parse_args()


def _summarize(name: str, trials: list[dict], n_trials: int) -> dict:
    """汇总一组 trial 的覆盖率与偏差并打印。"""
    cov_level = np.mean([t["covers_level_A"] for t in trials])
    cov_delta = np.mean([t["covers_delta"] for t in trials])
    bias_level = np.mean([t["min_obs_A"] - t["true_min_A"] for t in trials])
    bias_delta = np.mean([t["delta_obs"] - t["true_delta"] for t in trials])
    # bootstrap 均值相对点估计的偏移（真实数据上唯一可算的偏差代理，用以对账主分析口径）
    boot_shift_level = np.mean([t["boot_mean_min_A"] - t["min_obs_A"] for t in trials])
    boot_shift_delta = np.mean([t["boot_mean_delta"] - t["delta_obs"] for t in trials])
    print(f"\n[{name}]  （{n_trials} trials）")
    print(f"  worst-group **level**（模型 A）：真值={trials[0]['true_min_A']:.4f}")
    print(f"    点估计偏差 E[min̂]−min_true = {bias_level:+.4f}   ← min 算子向下有偏")
    print(f"    95% CI 覆盖率              = {cov_level:.1%}   （名义 95%）")
    print(f"  **配对差值 Δ**（确认性估计量）：真值={trials[0]['true_delta']:+.4f}")
    print(f"    点估计偏差 E[Δ̂]−Δ_true     = {bias_delta:+.4f}   ← 偏差在配对中抵消程度")
    print(f"    95% CI 覆盖率              = {cov_delta:.1%}   （名义 95%）")
    print(f"  bootstrap 均值相对点估计偏移：level {boot_shift_level:+.4f} / Δ {boot_shift_delta:+.4f}")
    return {
        "coverage_level": float(cov_level), "coverage_delta": float(cov_delta),
        "bias_level": float(bias_level), "bias_delta": float(bias_delta),
        "boot_shift_level": float(boot_shift_level), "boot_shift_delta": float(boot_shift_delta),
    }


def main() -> None:
    args = parse_args()
    print("E-audit.3  worst-group 动态-min 校准自检（Monte-Carlo，已知真值）")
    print("⚠️ 只估覆盖率/偏差，**不切换主终点**——动态 argmin 始终是确认性主分析（plan §3.3）。")
    print(f"\n合成结构（贴合 M2C）：{N_PATIENTS:,} 患者 × {IMGS_PER_PATIENT} 图 = "
          f"{N_PATIENTS * IMGS_PER_PATIENT:,} 样本，目标正例率={POS_RATE:.2%}，{N_GROUPS} 候选组")
    print(f"患者随机效应：分数 sd={SCORE_CLUSTER_SD}，**标签 sd={LABEL_CLUSTER_SD}（logit）** "
          f"⇒ 同患者标签高度相关（真实 CXR 结构）")

    rng = np.random.default_rng(args.seed)
    label_intercept = _calibrate_label_intercept(rng)
    print(f"标定标签截距 = {label_intercept:.3f}（使边缘正例率 ≈ {POS_RATE:.2%}）")

    out: dict = {"config": {"n_trials": args.n_trials, "n_boot": args.n_boot,
                            "n_patients": N_PATIENTS, "imgs_per_patient": IMGS_PER_PATIENT,
                            "pos_rate": POS_RATE, "n_groups": N_GROUPS,
                            "score_cluster_sd": SCORE_CLUSTER_SD,
                            "label_cluster_sd": LABEL_CLUSTER_SD,
                            "label_intercept": label_intercept},
                 "scenarios": {}}

    for scen, (true_auc_a, true_delta_b) in SCENARIOS.items():
        true_delta = (true_auc_a + true_delta_b).min() - true_auc_a.min()
        sorted_a = np.sort(true_auc_a)
        print(f"\n{'#' * 78}\n# 情景 `{scen}`：真 AUC(A)={true_auc_a} → worst={true_auc_a.min():.4f}"
              f"；次弱={sorted_a[1]:.4f}（分离度={sorted_a[1] - sorted_a[0]:+.3f}）"
              f"\n# 模型 B 真效应={true_delta_b} → 真 Δ_worst={true_delta:+.4f}\n{'#' * 78}")
        out["scenarios"][scen] = {"true_auc_a": true_auc_a.tolist(),
                                  "true_delta_b": true_delta_b.tolist(),
                                  "true_delta": float(true_delta)}
        for name, sample_level in (("患者级 cluster bootstrap（主分析口径）", False),
                                   ("样本级 bootstrap（对照：忽略 cluster）", True)):
            trials = []
            for t in range(args.n_trials):
                trials.append(run_trial(rng, args.n_boot, true_auc_a, true_delta_b,
                                        label_intercept, sample_level=sample_level))
                if (t + 1) % 50 == 0:
                    print(f"  ... {scen}/{name}: {t + 1}/{args.n_trials}")
            key = "sample_level" if sample_level else "cluster"
            out["scenarios"][scen][key] = _summarize(name, trials, args.n_trials)

    print("\n解读（依实测，非预期）：")
    print("  · **level 侧不可信**：min 算子向下偏 2.7~3.9 个千分点，CI 覆盖率远低于名义（69.5%/18.0%）")
    print("    ⇒ **绝对** worst-group 水平不是无偏量，禁止当作「该组真实 AUC」解读；")
    print("  · **Δ 侧（确认性估计量）取决于 argmin 分离度**：")
    print("    - separated（次弱组高 +0.04）：偏差 −0.002、覆盖率 92% ≈ 名义 ⇒ 判决可信；")
    print("    - tied（次弱组仅高 +0.01）：偏差 −0.006、覆盖率 80% ⇒ **保守**（低估增益、CI 偏窄），")
    print("      此时「未达显著」可能是估计量伪影而非真无效应，须结合 argmin 稳定性谨慎解读。")
    print("    真实数据落在哪个情景，由 eaudit_m2c_full_target.py 报的 argmin 稳定性判断。")
    print("  · **cluster vs 样本级（修正预期）**：level 侧 cluster 覆盖率确实更高（69.5% vs 62.0%、")
    print("    18.0% vs 11.5%，方向符合预期）；但 **Δ 侧两者几乎无差**（92.0% vs 93.5%、80.5% vs 81.0%，")
    print("    差异在 MC 噪声内）。原因：配对比较中患者随机效应对两模型**同样作用、在相减时抵消**，")
    print("    故聚类对 Δ 的方差贡献本就小（且本数据仅 ~3 图/患者，簇内相关有限）。")
    print("    ⇒ 患者级 cluster bootstrap 仍按 plan §3.3 作主口径（更保守、零额外代价），")
    print("      但**不应**宣称「样本级会显著反保守」——本模拟不支持该说法。")

    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ABLATION_DIR / "eaudit_dynamic_min_calibration.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已落盘 → {out_path}")


if __name__ == "__main__":
    main()
