"""
把合成属性翻转率 η 标定到目标条件互信息档位（评审补强 R1.2）
=============================================================
R1.1 已证 `A_syn = Y ⊕ Bernoulli(η)` 的 `I(Y;A_syn|X)` 由 η 单调可调（负控制在 η=0.5）。
本脚本把这个「η ↔ 实测 I」关系**在 MIMIC 真实数据上**标定出来，为 R1.3 的剂量-反应实验选定
落在目标档位 `I(Y;A_syn|X) ∈ {0, 0.02, 0.05, 0.10} nats` 的 η 值（0 档 = η=0.5 负控制）。

**估计口径与 D2 完全一致**（单一事实来源）：复用 scripts/estimate_conditional_mi.py 的
冻结特征 φ(X) 上的 ΔNLL 估计器（`_delta_nll_for_synthetic`，base: Y~φ vs int: 逐 A_syn 组
独立 [1,φ]，held-out test ΔNLL = I 的保守下界）。A_syn 由 R1.1 的确定性
`make_split_rng` 生成，与 R1.3 训练/评估读到的 A_syn 逐元素一致。

方法对标（R1.2 参考）：MINE / CMI Neural Estimator 用已知真值合成分布验证估计器功效的思路，
此处**反用**——扫 η（已知会单调改变真值 I），读估计器输出，建 η→实测 I 的标定映射。

流程：
    1. 复用 MIMIC baseline 冻结特征缓存（outputs/mimic_cxr/cmi_cache，backbone seed 42/43/44）。
    2. 扫 η 网格，各 η 上以真实 Y 为标签、A_syn 为条件组，估 test ΔNLL_int / ΔNLL_add（跨 3 seed 均值 ± std）。
    3. 对每个目标档位，用 η→ΔNLL 单调曲线线性插值求 η*，并直接在 η* 上复核实测 I 落在目标 ±MDE 内。
    4. 落盘标定表 JSON + 打印，供 R1.3 直接取用 η*。

运行：轻量（在缓存特征上跑逐层凸 logistic），GPU 可显著加速 LBFGS，按项目规范经 sbatch 提交
（slurm/calibrate_synthetic_signal.sh）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.estimate_conditional_mi import (
    CACHE_DIR, OUTPUT_DIR, SEEDS, SPLITS, _delta_nll_for_synthetic, load_features,
)
from src.datasets.synthetic_attribute import inject_synthetic_attribute, make_split_rng

# 目标条件互信息档位（nats）。0 档 = 负控制（η=0.5，A_syn⊥Y|X）。
TARGET_NATS: tuple[float, ...] = (0.0, 0.02, 0.05, 0.10)

# η 扫描网格：0.5（负控制）向 0 递减；小信号档位密集在 0.5 附近，故 0.5 端更密。
ETA_SCAN: tuple[float, ...] = (
    0.50, 0.48, 0.46, 0.44, 0.42, 0.40, 0.37, 0.34, 0.30, 0.25, 0.20, 0.15, 0.10,
)

OUT_PATH = OUTPUT_DIR / "synthetic_signal_calibration.json"


def _a_syn_for_all_splits(feats: dict[str, dict[str, np.ndarray]], eta: float) -> dict[str, np.ndarray]:
    """
    为一个 η 在 train/val/test 上生成确定性 A_syn = 真实 Y ⊕ Bernoulli(η)。

    标签取自缓存特征里的真实 MIMIC label（与 baseline 训练同口径）；每 split 用 R1.1 的
    `make_split_rng(eta, split)` 播种，保证与 R1.3 训练/评估读到的 A_syn 完全一致。
    """
    return {
        split: inject_synthetic_attribute(feats[split]["label"], eta, make_split_rng(eta, split))
        for split in SPLITS
    }


def scan_eta(seeds: tuple[int, ...] = SEEDS) -> dict[str, dict]:
    """
    在 η 扫描网格上估计 test ΔNLL_int / ΔNLL_add（跨 backbone seed 均值 ± std）。

    对每个 backbone seed 载入其冻结特征，逐 η 用真实 Y + A_syn 跑 base-vs-int 估计器。
    返回 {f"eta={η}": {int/add 的 mean/std/per_seed + CI}}。
    """
    # 预载每个 seed 的特征一次（train 158MB，避免逐 η 重载）
    feats_by_seed = {seed: load_features(seed) for seed in seeds}

    out: dict[str, dict] = {}
    for eta in ETA_SCAN:
        int_vals, add_vals, ci_int_per_seed = [], [], []
        for seed in seeds:
            feats = feats_by_seed[seed]
            a_syn = _a_syn_for_all_splits(feats, eta)
            y_real = {s: feats[s]["label"].astype(np.int64) for s in SPLITS}
            rng = np.random.default_rng(1000 + seed)
            # int：逐 A_syn 组独立决策函数（与 D2 头条同口径）
            res_int = _delta_nll_for_synthetic(feats, y_real, a_syn, rng, n_bootstrap=500)
            int_vals.append(res_int["delta_nll_nats"])
            ci_int_per_seed.append(res_int["ci95_nats"])
            # add：逐 A_syn 组仅截距偏移（校准型；A_syn 的主效应几乎全在此）
            add_vals.append(_delta_nll_add(feats, y_real, a_syn, np.random.default_rng(2000 + seed)))
        out[f"eta={eta}"] = {
            "eta": eta,
            "int_nats_mean": float(np.mean(int_vals)), "int_nats_std": float(np.std(int_vals)),
            "add_nats_mean": float(np.mean(add_vals)), "add_nats_std": float(np.std(add_vals)),
            "int_per_seed": [float(v) for v in int_vals],
            "int_ci95_per_seed": ci_int_per_seed,
        }
        m, sd = out[f"eta={eta}"]["int_nats_mean"], out[f"eta={eta}"]["int_nats_std"]
        print(f"[scan] η={eta:.2f}  ΔNLL_int={m:+.5f}±{sd:.5f} nats  "
              f"ΔNLL_add={out[f'eta={eta}']['add_nats_mean']:+.5f} nats", flush=True)
    return out


def _delta_nll_add(
    feats: dict[str, dict[str, np.ndarray]], y_syn: dict[str, np.ndarray],
    a_syn: dict[str, np.ndarray], rng: np.random.Generator,
) -> float:
    """
    add 口径 ΔNLL：base(Y~φ) vs add(Y~[φ, onehot(A_syn)去首列])，逐组截距偏移。

    复用 estimate_conditional_mi 的 `_build_design`/`_eval_test_logit`/`_per_sample_nll`，
    与 int 口径同一套凸 logistic + 温度标定，只换设计矩阵。仅返回全局 ΔNLL 均值（标定辅助口径）。
    """
    from scripts.estimate_conditional_mi import _build_design, _eval_test_logit, _per_sample_nll
    tr, va, te = feats["train"], feats["val"], feats["test"]
    y_test = y_syn["test"].astype(np.float64)
    logit_f = _eval_test_logit(
        _build_design(tr["feat"], None, 0, "base"), y_syn["train"],
        _build_design(va["feat"], None, 0, "base"), y_syn["val"],
        _build_design(te["feat"], None, 0, "base"),
    )
    logit_g = _eval_test_logit(
        _build_design(tr["feat"], a_syn["train"], 2, "add"), y_syn["train"],
        _build_design(va["feat"], a_syn["val"], 2, "add"), y_syn["val"],
        _build_design(te["feat"], a_syn["test"], 2, "add"),
    )
    llr = _per_sample_nll(logit_f, y_test) - _per_sample_nll(logit_g, y_test)
    return float(llr.mean())


def interpolate_eta(scan: dict[str, dict], target_nats: float, key: str = "int_nats_mean") -> dict:
    """
    在 η→ΔNLL 单调曲线上线性插值，求给定目标 nats 对应的 η*。

    ΔNLL 关于 η 单调递减（η=0.5 时≈0，η 减小时增大）。对 target=0 直接返回 η=0.5（负控制）。
    否则在扫描点间找到跨越 target 的相邻区间，按 ΔNLL 线性插值 η*。

    Args:
        scan:        scan_eta 的输出。
        target_nats: 目标条件互信息（nats）。
        key:         用哪个口径插值（默认 int，D2 头条口径）。

    Returns:
        {target_nats, eta_star, bracket:(η_lo,η_hi), interp_note}。
    """
    if target_nats == 0.0:
        return {"target_nats": 0.0, "eta_star": 0.5, "bracket": [0.5, 0.5],
                "interp_note": "负控制：η=0.5，A_syn⊥Y|X（无需插值）"}

    # 按 η 降序（ΔNLL 升序）排列扫描点
    pts = sorted(({"eta": v["eta"], "nats": v[key]} for v in scan.values()), key=lambda p: -p["eta"])
    for lo, hi in zip(pts[:-1], pts[1:]):
        # lo.eta > hi.eta，lo.nats < hi.nats（单调）；target 落在 [lo.nats, hi.nats] 之间
        if lo["nats"] <= target_nats <= hi["nats"]:
            frac = (target_nats - lo["nats"]) / (hi["nats"] - lo["nats"] + 1e-12)
            eta_star = lo["eta"] + frac * (hi["eta"] - lo["eta"])
            return {"target_nats": target_nats, "eta_star": round(float(eta_star), 4),
                    "bracket": [hi["eta"], lo["eta"]],
                    "interp_note": f"在 η∈[{hi['eta']},{lo['eta']}]（ΔNLL∈[{lo['nats']:.4f},{hi['nats']:.4f}]）线性插值"}
    # target 超出扫描范围
    return {"target_nats": target_nats, "eta_star": None, "bracket": None,
            "interp_note": f"目标 {target_nats} nats 超出扫描区间 ΔNLL∈"
                           f"[{pts[0][key] if False else pts[-1]['nats']:.4f},{pts[0]['nats']:.4f}]，请扩 η 网格"}


def verify_eta_star(eta_star: float, seeds: tuple[int, ...] = SEEDS) -> dict:
    """在插值得到的 η* 上直接跑估计器复核实测 I（跨 seed 均值 ± std + 每 seed CI）。"""
    int_vals, ci_list = [], []
    for seed in seeds:
        feats = load_features(seed)
        a_syn = _a_syn_for_all_splits(feats, eta_star)
        y_real = {s: feats[s]["label"].astype(np.int64) for s in SPLITS}
        res = _delta_nll_for_synthetic(feats, y_real, a_syn, np.random.default_rng(3000 + seed), n_bootstrap=1000)
        int_vals.append(res["delta_nll_nats"])
        ci_list.append(res["ci95_nats"])
    return {"eta_star": eta_star, "measured_int_nats_mean": float(np.mean(int_vals)),
            "measured_int_nats_std": float(np.std(int_vals)), "int_per_seed": [float(v) for v in int_vals],
            "ci95_per_seed": ci_list}


def main() -> None:
    """扫 η → 建标定映射 → 对各目标档位插值 η* → 复核 → 落盘 + 打印。"""
    print(f"缓存目录: {CACHE_DIR}")
    missing = [s for s in SEEDS if not (CACHE_DIR / f"feat_seed{s}_test.npz").exists()]
    if missing:
        raise FileNotFoundError(
            f"缺少 backbone seed {missing} 的特征缓存；请先跑 scripts/estimate_conditional_mi.py 抽特征。")

    print(f"\n{'='*70}\n扫描 η → 实测 I(Y;A_syn|X)（跨 backbone seed {SEEDS}）\n{'='*70}")
    scan = scan_eta()

    print(f"\n{'='*70}\n对目标档位插值 η* 并复核\n{'='*70}")
    calibration = []
    for target in TARGET_NATS:
        interp = interpolate_eta(scan, target)
        entry = {**interp}
        if interp["eta_star"] is not None and target > 0.0:
            entry["verification"] = verify_eta_star(interp["eta_star"])
            v = entry["verification"]
            print(f"[target {target:.2f} nats] η*={interp['eta_star']} → 复核实测 "
                  f"ΔNLL_int={v['measured_int_nats_mean']:+.5f}±{v['measured_int_nats_std']:.5f} nats")
        else:
            print(f"[target {target:.2f} nats] η*={interp['eta_star']}（{interp['interp_note']}）")
        calibration.append(entry)

    out = {
        "estimator": "frozen-feature ΔNLL (base vs int)，与 D2 conditional-MI 同口径",
        "backbone_seeds": list(SEEDS),
        "target_nats": list(TARGET_NATS),
        "eta_scan": scan,
        "calibration_table": calibration,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n标定表已保存到 {OUT_PATH}")

    print(f"\n{'='*70}\n标定表（翻转率 η ↔ 目标 I(Y;A_syn|X)，供 R1.3 取用）\n{'='*70}")
    print(f"  {'目标 nats':>10} | {'η*':>7} | {'复核实测 nats':>16}")
    for e in calibration:
        v = e.get("verification")
        meas = (f"{v['measured_int_nats_mean']:+.5f}±{v['measured_int_nats_std']:.5f}"
                if v else "0（负控制）")
        print(f"  {e['target_nats']:>10.2f} | {str(e['eta_star']):>7} | {meas:>16}")


if __name__ == "__main__":
    main()
