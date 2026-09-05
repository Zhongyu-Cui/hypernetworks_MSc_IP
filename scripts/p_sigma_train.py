"""
实验 P·训练噪声 σ_train 估计（多 replicate）
============================================
主结果（`scripts/p_pred_attr_analysis.py`）的 bootstrap CI 只含**评估不确定性**：它回答
「换一批测试样本，这个 Δ 会变多少」，不回答「同样的数据、同样的配置**重训一次**，Δ 会变多少」。
后者是 σ_train，需要多个只差训练随机种子的完整 CV-OOF 重复。

设计（与 slurm/p_pred_attr.sh 的 TRIAL 参数配套）：
    seed = 42 + fold + 100 * trial      trial 0 = 主结果，trial 1/2 = 额外 replicate
每个 trial 都是一整套 5 折 OOF ⇒ 每个 trial 给出一个完整的 averaging 指标估计。

**配对口径（本脚本的主输出）**：对每个 trial 各算一次配对差 Δ_t = metric(臂) − metric(参照)，
再取跨 trial 的均值与标准差。配对能抵消 trial 间的共同波动（如某个 seed 整体训得好），
比单臂方差更贴近「重跑一次，这个对比会变多少」。这与实验 G 用配对差矩阵估 σ_train 的口径一致。

判据：**|mean Δ| < σ_train ⇒ 该效应不超训练噪声**，无论 bootstrap CI 说什么。

运行（轻量 CPU）：
    python scripts/p_sigma_train.py --dataset fitzpatrick
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.build_oof_results_averaging import Spec, Views, clusters
from scripts.p_pred_attr_analysis import (
    DATASET_SPECS, METRIC_KEYS, arms_and_contrasts, resolve_tags,
)
from src.training.harness.predictions import load_predictions

N_FOLDS = 5
FOLD_SEED_BASE = 42
TRIAL_STRIDE = 100          # 与 slurm/p_pred_attr.sh 的 seed 公式一致


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    ap = argparse.ArgumentParser(description="实验 P：σ_train（训练噪声）估计")
    ap.add_argument("--dataset", required=True, choices=sorted(DATASET_SPECS))
    ap.add_argument("--trials", type=int, nargs="+", default=[0, 1, 2],
                    help="参与估计的 trial 号（缺省 0 1 2）。")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="只看这些方法（缺省 = 有完整 replicate 的全部方法）。")
    ap.add_argument("--cond", default="age", choices=("age", "sex_age"),
                    help="条件通路：age=历史臂；sex_age=HAM 双属性臂（method 带 _sexage 后缀），"
                         "结果写 p_sigma_train_sexage.json。")
    return ap.parse_args()


def load_trial_folds(spec: Spec, method: str, tag: str, trial: int) -> list[dict] | None:
    """
    载入某方法某 trial 的 5 折预测（保持逐折结构）。

    Args:
        spec  : 该库的寻址规格。
        method: 方法名。
        tag   : config_tag。
        trial : trial 号（seed = 42 + fold + 100*trial）。

    Returns:
        逐折 dict 列表；任一折缺失则返回 None。
    """
    out: list[dict] = []
    for fold in range(N_FOLDS):
        seed = FOLD_SEED_BASE + fold + TRIAL_STRIDE * trial
        path = spec.pred_dir / f"{method}_{tag}_seed{seed}_overall.npz"
        if not path.exists():
            return None
        y, s, a = load_predictions(path)
        y = np.asarray(y).astype(int)
        s = np.asarray(s, dtype=np.float64)
        if spec.filter_age_neg1:
            keep = a["age"].astype(int) >= 0
            y, s = y[keep], s[keep]
            a = {k: v[keep] for k, v in a.items()}
        out.append({"y": y, "s": s, "attrs": a})
    return out


def main() -> None:
    args = parse_args()
    spec = DATASET_SPECS[args.dataset]
    if args.cond == "sex_age" and args.dataset != "ham10000":
        raise ValueError("--cond sex_age 只有 HAM10000 有双属性对照臂")
    pred_arms, reference, _, _ = arms_and_contrasts(args.cond)
    tags = resolve_tags(spec, args.dataset, args.cond)

    # 逐 (方法, trial) 算一套 averaging 指标
    per_method: dict[str, dict[int, tuple]] = {}
    for method, tag in tags.items():
        if args.methods and method not in args.methods:
            continue
        for trial in args.trials:
            folds = load_trial_folds(spec, method, tag, trial)
            if folds is None:
                continue
            _, offs = clusters(spec, folds)
            views = Views(spec, folds)
            per_method.setdefault(method, {})[trial] = views.compute(offs, np.ones(offs[-1]))

    complete = {m: t for m, t in per_method.items() if len(t) >= 2}
    if not complete:
        print("没有任何方法具备 ≥2 个 trial，无法估计 σ_train。")
        return

    print(f"\n{'=' * 92}\n实验 P · {spec.name} · cond={args.cond} · σ_train（训练噪声）"
          f"  trials={sorted(set().union(*[set(t) for t in complete.values()]))}\n{'=' * 92}")
    print(f"{'方法':<24}{'trials':>7}" + "".join(f"{k:>21}" for k in METRIC_KEYS))
    single: dict[str, dict[str, float]] = {}
    for method, by_trial in complete.items():
        row = f"{method:<24}{len(by_trial):>7}"
        single[method] = {}
        for j, key in enumerate(METRIC_KEYS):
            vals = np.array([by_trial[t][j] for t in sorted(by_trial)])
            sd = float(vals.std(ddof=1))
            single[method][key] = sd
            row += f"{vals.mean():.4f}±{sd:.4f}".rjust(21)
        print(row)

    # ---- 配对差的 σ_train（主输出）----
    print(f"\n{'-' * 92}\n配对差的 σ_train：Δ = 臂 − 参照，逐 trial 各算一次，再求跨 trial 均值±SD\n"
          f"判据：|mean Δ| < σ_train ⇒ 该效应不超训练噪声\n{'-' * 92}")
    paired: dict[str, dict[str, dict]] = {}
    # 实验臂 = 本 cond 的四个 pred 臂；参照 = 该 cond 的 GT/基线 + 全部 pred 臂（含跨臂对比）
    arms = [m for m in complete if m in pred_arms]
    refs = [m for m in complete if m in reference or m in pred_arms]
    for arm in arms:
        for ref in refs:
            if arm == ref:
                continue
            shared = sorted(set(complete[arm]) & set(complete[ref]))
            if len(shared) < 2:
                continue
            entry: dict[str, dict] = {}
            print(f"\n{arm}  vs  {ref}   （{len(shared)} 个 trial 配对）")
            for j, key in enumerate(METRIC_KEYS):
                deltas = np.array([complete[arm][t][j] - complete[ref][t][j] for t in shared])
                mean_d, sd_d = float(deltas.mean()), float(deltas.std(ddof=1))
                ratio = abs(mean_d) / sd_d if sd_d > 0 else float("inf")
                entry[key] = {"mean_delta": mean_d, "sigma_train": sd_d, "ratio": ratio,
                              "per_trial": deltas.tolist()}
                verdict = "超噪声" if ratio > 1.0 else "不超噪声"
                print(f"  {key:<18} Δ={mean_d:+.4f}  σ_train={sd_d:.4f}  "
                      f"|Δ|/σ={ratio:.2f}  {verdict}")
            paired[f"{arm}_vs_{ref}"] = entry

    out = spec.cfg.parent / (
        "p_sigma_train.json" if args.cond == "age" else "p_sigma_train_sexage.json")
    out.write_text(json.dumps({
        "dataset": args.dataset, "cond": args.cond, "trials": args.trials, "config": tags,
        "single_arm_sigma": single, "paired": paired,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main()
