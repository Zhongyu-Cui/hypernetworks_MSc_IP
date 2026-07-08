"""
R3.1 —— 为 D2 条件互信息表逐行补 95% bootstrap 置信区间
========================================================

评审意见：D2 表原来只给光秃点估计（且部分为负，说明估计器在该尺度被方差主导），
「没检测到」≠「确认为零」。本脚本把已落盘的逐 seed test-bootstrap CI（各数据集
`conditional_mi.json` 里的 `ci95_nats`）合并成每个 (数据集, 敏感轴) 一行的
「点估计 ± 95%CI」，供 D2 表替换裸点估计。

统计口径（诚实地把两层不确定性都算进来）：
    - 估计量：held-out ΔNLL（int 模型）= I(Y;A|X) 的估计。真值 MI ≥ 0，故**负点估计
      = 真信号 ≈ 0 且估计器方差/过拟合主导**（held-out ΔNLL 在 g 过拟合时向下偏），
      这正是「被方差主导」的直接读数。
    - 层内（within-seed）不确定性：test 集 1000× bootstrap，从 JSON 的 ci95_nats 反推
      SE = (hi − lo) / (2 × 1.96)。
    - 层间（between-seed）不确定性：3 个 backbone seed 的点估计样本方差。
    - 合并：Rubin's rules（多重插补合并规则）——
        point = mean_k θ_k
        W     = mean_k SE_k^2                    （层内平均方差）
        B     = var_k θ_k (ddof=1)               （层间方差）
        T     = W + (1 + 1/K) · B                （总方差）
        ν     = (K−1) · (1 + W / ((1+1/K)·B))^2  （Rubin 自由度）
        95%CI = point ± t_{ν,0.975} · sqrt(T)
      当 B≪W 时 ν→∞（t→z=1.96）；当 B 主导时 ν→K−1=2（t 变宽），如实反映 seed 少。
    - PAPILA 为单次全-CV 池化运行（无多 seed），直接用其单run bootstrap CI。

用法：
    python -m scripts.summarize_conditional_mi_ci
输出：
    outputs/conditional_mi_ci_summary.json（机器可读）+ 终端可读表（可直接贴进 D2）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import t as student_t

OUTPUTS_ROOT = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs")
Z975 = 1.959963984540054  # 标准正态 0.975 分位（bootstrap CI 反推 SE 用）

# (显示名, 目录, 是否多-seed)；敏感轴由各 JSON 的 E1 keys 决定
DATASETS = [
    ("MIMIC-CXR", "mimic_cxr", True),
    ("CheXpert", "chexpert_cxr", True),
    ("Fitzpatrick17k", "fitzpatrick", True),
    ("HAM10000", "ham10000", True),
    ("PAPILA", "papila", False),
]


def _se_from_ci(lo: float, hi: float) -> float:
    """由对称 95% bootstrap CI 反推标准误 SE。"""
    return (hi - lo) / (2.0 * Z975)


def combine_rubin(points: list[float], ses: list[float]) -> dict:
    """
    用 Rubin's rules 合并多 seed 的点估计与层内 SE，返回合并点估计 + 95%CI + 自由度。

    Args:
        points: 各 seed 的 ΔNLL 点估计。
        ses:    各 seed 的层内（test-bootstrap）标准误。

    Returns:
        dict：point / ci95 / t_df / within_var / between_var / total_var。
    """
    pts = np.asarray(points, dtype=np.float64)
    se = np.asarray(ses, dtype=np.float64)
    k = pts.size
    point = float(pts.mean())
    within = float((se ** 2).mean())                      # W
    between = float(pts.var(ddof=1)) if k > 1 else 0.0    # B
    total = within + (1.0 + 1.0 / k) * between            # T
    if between > 0 and k > 1:
        # Rubin 自由度：seed 少且层间主导时收缩到 k−1
        nu = (k - 1) * (1.0 + within / ((1.0 + 1.0 / k) * between)) ** 2
    else:
        nu = np.inf
    tcrit = float(student_t.ppf(0.975, nu)) if np.isfinite(nu) else Z975
    half = tcrit * float(np.sqrt(total))
    return {
        "point": point,
        "ci95": [point - half, point + half],
        "t_df": (float(nu) if np.isfinite(nu) else None),
        "within_var": within,
        "between_var": between,
        "total_var": total,
        "n_seeds": int(k),
    }


def _verdict(point: float, ci_lo: float, ci_hi: float) -> str:
    """
    据点估计与 CI 给判定。核心：MI 真值≥0，故只有 CI 下界 > 0 才算「确认 > 0」；
    CI 含 0 = 不可判定（灰区）；负点估计且 CI 不含 0 = 落在噪声地板（方差主导，真值≈0）。
    """
    if ci_lo > 0:
        return "confirmed > 0"
    if ci_hi < 0:
        return "noise floor (est. variance-dominated; true I≈0)"
    return "undecidable (CI spans 0)"


def process_dataset(display: str, subdir: str, multiseed: bool) -> list[dict]:
    """读单个数据集 JSON，对每个敏感轴给合并 CI 行。"""
    path = OUTPUTS_ROOT / subdir / "conditional_mi.json"
    data = json.load(open(path, encoding="utf-8"))
    rows: list[dict] = []

    if multiseed:
        per_seed = data["per_seed"]
        seeds = list(per_seed.keys())
        specs = list(per_seed[seeds[0]]["E1"].keys())
        for spec in specs:
            points, ses = [], []
            for s in seeds:
                g = per_seed[s]["E1"][spec]["int"]
                lo, hi = g["ci95_nats"]
                points.append(float(g["delta_nll_nats"]))
                ses.append(_se_from_ci(lo, hi))
            comb = combine_rubin(points, ses)
            comb.update({"dataset": display, "axis": spec,
                         "verdict": _verdict(comb["point"], comb["ci95"][0], comb["ci95"][1])})
            rows.append(comb)
    else:
        # PAPILA：单次全-CV 池化，直接用其单run bootstrap CI
        for spec, g in data["E1"].items():
            lo, hi = g["int"]["ci95_nats"]
            point = float(g["int"]["delta_nll_nats"])
            rows.append({
                "dataset": display, "axis": spec, "point": point,
                "ci95": [float(lo), float(hi)], "t_df": None,
                "within_var": _se_from_ci(lo, hi) ** 2, "between_var": None,
                "total_var": _se_from_ci(lo, hi) ** 2, "n_seeds": 1,
                "verdict": _verdict(point, lo, hi),
            })
    return rows


def mde_from_dose_response(display: str, subdir: str) -> dict:
    """
    R3.2 —— 从 E3 dose-response 提取每数据集的最小可检测效应（MDE）与正控制探针功效。

    E3 在 seed 42 上以概率 α 用与图像无关的随机属性覆盖标签（注入强度 α 的纯条件信号），
    报告每档「实测 ΔNLL（nats）+ bootstrap CI + 是否 CI 下界>0（detected）」。定义：
        - MDE = 首个 detected=True 档的**实测 ΔNLL**（nats）= 探针在本数据规模下能可靠检出的
          最小注入信号；其下最大未检出档给出下括。
        - 正控制（Y XOR r，强纯条件信号）的 ΔNLL = 探针功效上确认（必须 detected）。
        - 若 detected 序列随 α 非单调（先 True 后 False），标记 probe_unreliable=True
          （n 太小、单档实现噪声压过信号，MDE 不可定义）——用于把该数据集判为「灰区」。

    Returns:
        dict：mde_nats / mde_bracket_nats / positive_control / probe_unreliable。
    """
    data = json.load(open(OUTPUTS_ROOT / subdir / "conditional_mi.json", encoding="utf-8"))
    e3 = data["E3_sensitivity"]
    pc = e3["positive_control_xor"]
    # 按 α 数值排序
    items = sorted(e3["dose_response"].items(), key=lambda kv: float(kv[0].split("=")[1]))
    alphas = [float(k.split("=")[1]) for k, _ in items]
    measured = [float(v["delta_nll_nats"]) for _, v in items]
    detected = [bool(v["detected"]) for _, v in items]

    # 单调性：一旦 True 应保持 True；否则探针在该规模不可靠
    first_true = next((i for i, d in enumerate(detected) if d), None)
    probe_unreliable = first_true is not None and not all(detected[first_true:])

    if first_true is None:
        mde, lower = None, measured[-1]
    else:
        mde = measured[first_true]
        lower = measured[first_true - 1] if first_true > 0 else None

    return {
        "dataset": display,
        "mde_nats": mde,
        "mde_bracket_nats": [lower, mde],           # (最大未检出, 最小检出)
        "mde_alpha": (alphas[first_true] if first_true is not None else None),
        "positive_control_nats": pc["delta_nll_nats"],
        "positive_control_detected": bool(pc["detected"]),
        "probe_unreliable": probe_unreliable,
    }


def main() -> None:
    """跑全部数据集 → 落 JSON + 打印可贴进 D2 的表（R3.1 CI + R3.2 MDE）。"""
    all_rows: list[dict] = []
    for display, subdir, multiseed in DATASETS:
        all_rows.extend(process_dataset(display, subdir, multiseed))

    mde_rows = [mde_from_dose_response(display, subdir) for display, subdir, _ in DATASETS]

    out_path = OUTPUTS_ROOT / "conditional_mi_ci_summary.json"
    json.dump({"rows": all_rows, "mde": mde_rows,
               "method": "Rubin's rules over per-seed test-bootstrap CIs; MDE from E3 dose-response (seed 42)"},
              open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # R3.1 CI 表
    print(f"\n{'数据集':<16}{'轴':<9}{'I(Y;A|X) 点估计 ± 95%CI (nats)':<40}{'判定'}")
    print("-" * 100)
    for r in all_rows:
        lo, hi = r["ci95"]
        df_s = f"df={r['t_df']:.1f}" if r["t_df"] else "single-run"
        est = f"{r['point']:+.5f}  [{lo:+.5f}, {hi:+.5f}]"
        print(f"{r['dataset']:<16}{r['axis']:<9}{est:<40}{r['verdict']}  ({df_s})")

    # R3.2 MDE 表
    print(f"\n{'数据集':<16}{'MDE (nats)':<28}{'正控制 XOR ΔNLL':<18}{'探针'}")
    print("-" * 80)
    for m in mde_rows:
        lo, hi = m["mde_bracket_nats"]
        lo_s = f"{lo:+.5f}" if lo is not None else "—"
        hi_s = f"{hi:+.5f}" if hi is not None else ">max"
        bracket = f"{lo_s} < MDE ≤ {hi_s}"
        probe = "不可靠(非单调)" if m["probe_unreliable"] else "有功效"
        print(f"{m['dataset']:<16}{bracket:<28}{m['positive_control_nats']:+.4f}{'':<10}{probe}")
    print(f"\n已保存: {out_path}")


if __name__ == "__main__":
    main()
