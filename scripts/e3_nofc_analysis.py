"""
E1 敏感性实验③分析：fc-off 后超网络是否激活 conv
====================================================
消费 `train_condnet.py --diagnostics` 落盘的 diag_logs/ + 预测 + checkpoint，回答：
关掉 fc 条件化后，超网络是否被逼着把 age 条件化路由进 conv（区分「fc 挤占」vs「优化/参数化瓶颈」）。

三部分：
  A. **诊断轨迹**（逐 epoch，跨 15 run 对齐平均）：conv ρ^between / 梯度范数(A/B) / 权重范数(‖A‖)。
  B. **最终 conv ρ^between**（从 best_overall checkpoint 算，与 E1 fc-on 版同口径对比）——
     判「fc 挤占」：nofc 的 conv ρ^between 是否显著大于 fc-on 版、是否接近 fc-on 的 fc ρ^between。
  C. **行为**（averaging 口径 test）：nofc 的 Overall/worst vs ERM 与 vs E1 fc-on；
     + 属性置换（conv-only 的 age 依赖，与 E1②knockout 的 fc_off 变体互证）。

用法（纯 CPU，实验室机器）：
    python scripts/e3_nofc_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.e1_ham_analysis import (
    CV_DIR, HAM_AGE_NAMES, N_FOLDS, SEEDS, SPLIT_DIR, load_candidate_groups,
)
from scripts.eaudit_m2c_full_target import sorted_view, weighted_auc
from src.training.harness.predictions import load_predictions

TAG = "lr1e-04_wd1e-04"                         # E1 为 C-Deep/C-Full 选定的 config（exp3 沿用）
NOFC = {"condnet_deep_nofc": "C-Deep-noFC", "condnet_full_nofc": "C-Full-noFC"}
FCON = {"condnet_deep_nofc": "condnet_deep", "condnet_full_nofc": "condnet_full"}  # 对应 fc-on
NAME2CODE = {f"age:{v}": k for k, v in HAM_AGE_NAMES.items()}


# ============================================================
# A. 诊断轨迹（逐 epoch 跨 run 对齐平均）
# ============================================================
def load_diag_runs(cell: str) -> list[list[dict]]:
    """载入某 cell 的 15 条 diag 日志（每条 = 一个 run 的逐 epoch 记录）。"""
    runs = []
    for k in range(N_FOLDS):
        for seed in SEEDS:
            p = CV_DIR / f"fold{k}" / "diag_logs" / f"{cell}_{TAG}_seed{seed}.jsonl"
            runs.append([json.loads(line) for line in p.read_text().splitlines()])
    return runs


def trajectory(cell: str) -> None:
    """打印跨 run 对齐的逐 epoch 轨迹（各 epoch 对当时仍在训的 run 平均）。"""
    runs = load_diag_runs(cell)
    max_ep = max(len(r) for r in runs)
    print(f"\n  {NOFC[cell]}（{len(runs)} run，最长 {max_ep} epoch；各 epoch 对仍在训 run 平均）")
    print(f"    {'epoch':>5s} | {'conv ρ^btw':>10s} | {'grad_A':>7s} | {'grad_B':>7s} | "
          f"{'‖A‖':>6s} | {'n_run':>5s} | {'val_ov':>6s} | {'val_wc':>6s}")
    for e in range(max_ep):
        rows = [r[e] for r in runs if e < len(r)]
        def m(key): return float(np.mean([x[key] for x in rows]))
        if e < 5 or e == max_ep - 1 or e % 5 == 4:
            print(f"    {e + 1:>5d} | {m('conv_rho_between_mean'):>10.4f} | {m('gradnorm_A_mean'):>7.4f} | "
                  f"{m('gradnorm_B_conv_mean'):>7.4f} | {m('wnorm_A'):>6.3f} | {len(rows):>5d} | "
                  f"{m('val_overall_auc'):>6.4f} | {m('val_worst_case_auc'):>6.4f}")


# ============================================================
# B. 最终 conv ρ^between（从 best_overall checkpoint 算）
# ============================================================
def final_rho(cell: str) -> dict:
    """跨 15 run 平均的最终 conv ρ / ρ^between（best_overall checkpoint 口径，与 E1 §4 对齐）。"""
    from scripts.e1_rho_and_permutation import compute_rho
    from src.models.resnet18_condnet import build_base_fc_seed
    from src.training.train_condnet import DATASET_SPECS

    spec = DATASET_SPECS["ham10000"]
    location = "deep_nofc" if "deep" in cell else "full_nofc"
    rhos, betws = [], []
    for k in range(N_FOLDS):
        for seed in SEEDS:
            ckpt = CV_DIR / f"fold{k}" / f"{cell}_{TAG}_seed{seed}_best_overall.pth"
            model = spec.cond_factory(location, build_base_fc_seed("ham10000", fold=k, seed=seed), False, False)
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            model.eval()
            r = compute_rho(model)                          # nofc ⇒ 只有 conv 层
            conv = [v for kk, v in r.items() if kk != "fc"]
            rhos.append(np.mean([v["rho"] for v in conv]))
            betws.append(np.mean([v["rho_between"] for v in conv]))
            del model
    return {"conv_rho_mean": float(np.mean(rhos)), "conv_rho_between_mean": float(np.mean(betws)),
            "conv_rho_between_sd": float(np.std(betws))}


# ============================================================
# C. 行为：averaging Overall/worst + 置换
# ============================================================
def load_cell_folds(cell: str) -> list[dict]:
    """载入某 cell 的逐折 test 预测（overall checkpoint），保逐折结构。"""
    folds = []
    for k in range(N_FOLDS):
        df = pd.read_csv(SPLIT_DIR / f"fold{k}" / "test.csv")
        df = df[df["age_group"] >= 0].reset_index(drop=True)
        rec = {"y": df["label"].values.astype(int), "age": df["age_group"].values.astype(int),
               "lesion": df["lesion_id"].values, "scores": {}}
        for seed in SEEDS:
            p = CV_DIR / f"fold{k}" / "predictions" / f"{cell}_{TAG}_seed{seed}_overall.npz"
            yy, ss, attrs = load_predictions(p)
            assert np.array_equal(np.asarray(yy).astype(int), df["label"].values.astype(int))
            assert np.array_equal(attrs["age"].astype(int), df["age_group"].values.astype(int))
            rec["scores"][seed] = np.asarray(ss, dtype=np.float64)
        folds.append(rec)
    return folds


def averaging_metrics(folds: list[dict], groups: list[str], age_feed=None) -> tuple[float, float]:
    """averaging 口径 (Overall, marginal worst)；age_feed=None 用真实 age（分组键恒真实）。"""
    per_group = {g: [] for g in groups}
    overalls = []
    for k, f in enumerate(folds):
        y, real_age = f["y"], f["age"]
        fed = real_age if age_feed is None else age_feed[k]
        for seed in SEEDS:
            s = f["scores"][seed] if age_feed is None else f["scores_table"][seed][np.arange(len(y)), fed]
            i, ys = sorted_view(y, s, np.ones(len(y), bool))
            a = weighted_auc(ys, np.ones(len(y))[i])
            if not np.isnan(a):
                overalls.append(a)
            for g in groups:
                m = real_age == NAME2CODE[g]
                ig, yg = sorted_view(y, s, m)
                ag = weighted_auc(yg, np.ones(len(y))[ig])
                if not np.isnan(ag):
                    per_group[g].append(ag)
    means = {g: float(np.mean(v)) for g, v in per_group.items() if v}
    return float(np.mean(overalls)), float(min(means.values()))


def main() -> None:
    groups = load_candidate_groups()
    print("=" * 90)
    print("E1 敏感性实验③：fc-off 后超网络是否激活 conv")
    print("=" * 90)

    # ---------- A. 诊断轨迹 ----------
    print("\n" + "=" * 90 + "\nA. 诊断轨迹（逐 epoch）：关 fc 后 conv 是否随训练被激活\n" + "=" * 90)
    for cell in NOFC:
        trajectory(cell)

    # ---------- B. 最终 conv ρ^between vs E1 fc-on ----------
    print("\n" + "=" * 90 + "\nB. 最终 conv ρ^between（best_overall checkpoint）vs E1 fc-on 版\n" + "=" * 90)
    e1_rho = json.loads((CV_DIR / "e1_rho_between.json").read_text())     # E1 fc-on 聚合
    def fcon_conv_betw(fcon_cell):
        vals = [v["rho_between"] for kk, v in e1_rho.items()
                if kk.startswith(f"{fcon_cell}|") and not kk.endswith("|fc")]
        return float(np.mean(vals))
    def fcon_fc_betw(fcon_cell):
        return e1_rho[f"{fcon_cell}|fc"]["rho_between"]
    print(f"  {'cell':<13s} | {'conv ρ^btw (nofc)':>18s} | {'conv ρ^btw (fc-on)':>18s} | "
          f"{'倍数':>5s} | {'fc-on 的 fc ρ^btw':>16s}")
    rho_out = {}
    for cell, fcon in FCON.items():
        fr = final_rho(cell)
        rho_out[cell] = fr
        nofc_b = fr["conv_rho_between_mean"]
        fcon_b = fcon_conv_betw(fcon)
        fc_b = fcon_fc_betw(fcon)
        print(f"  {NOFC[cell]:<13s} | {nofc_b:>18.4f} | {fcon_b:>18.4f} | {nofc_b / fcon_b:>5.1f}× | "
              f"{fc_b:>16.4f}")

    # ---------- C. 行为：averaging + 置换 ----------
    print("\n" + "=" * 90 + "\nC. 行为（averaging test）：nofc vs ERM / fc-on\n" + "=" * 90)
    erm_folds = load_cell_folds_named("condnet_erm")
    o_erm, w_erm = averaging_metrics(erm_folds, groups)
    e1res = json.loads((CV_DIR / "e1_ham_results_averaging.json").read_text())["point_estimates"]
    print(f"  {'cell':<13s} | {'Overall':>8s} | {'worst':>8s} | {'Δworst vs ERM':>13s} | {'fc-on worst':>11s}")
    print(f"  {'ERM':<13s} | {o_erm:>8.4f} | {w_erm:>8.4f} | {'—':>13s} | {'—':>11s}")
    beh_out = {"ERM": {"overall": o_erm, "worst": w_erm}}
    for cell, fcon in FCON.items():
        folds = load_cell_folds(cell)
        o, w = averaging_metrics(folds, groups)
        fcon_w = e1res["C-Deep" if "deep" in cell else "C-Full"]["marginal_worst"]
        beh_out[NOFC[cell]] = {"overall": o, "worst": w, "d_worst_vs_erm": w - w_erm}
        print(f"  {NOFC[cell]:<13s} | {o:>8.4f} | {w:>8.4f} | {w - w_erm:>+13.4f} | {fcon_w:>11.4f}")

    out = {"tag": TAG, "final_rho": rho_out, "behavior": beh_out,
           "erm": {"overall": o_erm, "worst": w_erm}}
    p = CV_DIR / "e3_nofc_results.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已落盘 → {p}")


def load_cell_folds_named(cell: str) -> list[dict]:
    """ERM 的逐折预测（config = ERM 选定 lr3e-05_wd1e-04）。"""
    erm_tag = json.loads((CV_DIR / "selected_configs.json").read_text())["config"]["condnet_erm"]
    folds = []
    for k in range(N_FOLDS):
        df = pd.read_csv(SPLIT_DIR / f"fold{k}" / "test.csv")
        df = df[df["age_group"] >= 0].reset_index(drop=True)
        rec = {"y": df["label"].values.astype(int), "age": df["age_group"].values.astype(int),
               "lesion": df["lesion_id"].values, "scores": {}}
        for seed in SEEDS:
            p = CV_DIR / f"fold{k}" / "predictions" / f"{cell}_{erm_tag}_seed{seed}_overall.npz"
            yy, ss, attrs = load_predictions(p)
            rec["scores"][seed] = np.asarray(ss, dtype=np.float64)
        folds.append(rec)
    return folds


if __name__ == "__main__":
    main()
