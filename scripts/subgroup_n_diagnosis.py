"""
R4.2 —— 评估端子群样本量诊断：界定 HAM canonical worst-group 口径的可信下限
==========================================================================

动机（评审 R4.2）：HAM worst-group = 14 个 canonical 子群 AUC 取 min，其中 8 个 sex×age joint
格样本极小（最小 27 眼）。worst-group 的高方差（D3 表 ERM ±0.15）究竟是「模型不稳」还是
「评估格太小、AUC 本身估不准」？本脚本对每个子群格报 **n / n_pos / n_neg / AUC / AUC 估计 SE**，
明确哪些格进入「噪声地板」（AUC 估计 SE 大到无信息），从而界定 canonical joint 口径的可信下限。

SE 三口径互证（对 seed-ensemble 分数）：
    - Hanley–McNeil 解析 SE：AUC 的经典闭式 SE（依赖 n_pos, n_neg, AUC），min(n_pos,n_neg) 小时暴涨。
    - DeLong 解析 SE：结构分量协方差（与 HM 独立推导，互证）。
    - Bootstrap SE：格内 case-resampling 2000 次（对退化格更稳）。
判据：SE ≥ 0.10（⇒ 95%CI 半宽 ≥ ~0.20，AUC estimate ±0.2 基本无信息）标为「噪声地板」。

用法：python -m scripts.subgroup_n_diagnosis
输出：outputs/ham10000/subgroup_n_diagnosis.json + 终端表。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.build_results_tables import DATASETS, _load, _ham_ensemble_scores
from src.training.harness.subgroup_auc import subgroup_auc_vector

OUT = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs")
NOISE_FLOOR_SE = 0.10   # AUC 估计 SE ≥ 此值 ⇒ 该格无信息（噪声地板）


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    """
    Hanley–McNeil (1982) AUC 解析 SE。Q1=AUC/(2−AUC)、Q2=2AUC²/(1+AUC)。
    n_pos 或 n_neg = 0 时不可定义，返回 nan。
    """
    if n_pos < 1 or n_neg < 1:
        return float("nan")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (auc * (1.0 - auc) + (n_pos - 1) * (q1 - auc * auc)
           + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)
    return float(np.sqrt(var)) if var > 0 else float("nan")


def bootstrap_auc_se(y: np.ndarray, s: np.ndarray, n_boot: int = 2000,
                     rng: np.random.Generator | None = None) -> float:
    """格内 case-resampling bootstrap 的 AUC SE（退化重采样跳过）。"""
    rng = rng or np.random.default_rng(0)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yi = y[idx]
        if len(np.unique(yi)) < 2:
            continue
        aucs.append(roc_auc_score(yi, s[idx]))
    return float(np.std(aucs, ddof=1)) if len(aucs) > 1 else float("nan")


def diagnose_ham() -> list[dict]:
    """对 HAM 14 canonical 子群逐格给 n / n_pos / n_neg / AUC / 三口径 SE / 噪声地板标记。"""
    key = "ham10000"
    y_true, ens, _seed, attrs = _ham_ensemble_scores()
    s = ens["erm"]  # 用 ERM ensemble 作代表；n/方差地板是评估端属性，方法间同量级
    vec = subgroup_auc_vector(key, y_true, s, sex=attrs["sex"], age=attrs["age"])

    # 重建每格掩码（与 subgroup_auc_vector 同口径的键解析）
    sex_name = {0: "Male", 1: "Female"}
    age_name = {0: "20-40", 1: "40-60", 2: "60-80", 3: "80+"}
    masks: dict[str, np.ndarray] = {}
    for sv, sn in sex_name.items():
        masks[f"sex:{sn}"] = attrs["sex"] == sv
    for av, an in age_name.items():
        masks[f"age:{an}"] = attrs["age"] == av
    for sv, sn in sex_name.items():
        for av, an in age_name.items():
            masks[f"sex:{sn}|age:{an}"] = (attrs["sex"] == sv) & (attrs["age"] == av)

    rows: list[dict] = []
    for gname, (auc, n) in vec.items():
        m = masks[gname]
        yy, ss = y_true[m], s[m]
        n_pos = int((yy == 1).sum())
        n_neg = int((yy == 0).sum())
        hm = hanley_mcneil_se(auc, n_pos, n_neg) if auc is not None else float("nan")
        boot = bootstrap_auc_se(yy, ss) if (n_pos >= 1 and n_neg >= 1) else float("nan")
        # DeLong 单 AUC SE：直接从 fast delong 的对角
        dl_se = _delong_single_se(yy, ss) if (n_pos >= 1 and n_neg >= 1) else float("nan")
        is_joint = "|" in gname
        # 噪声地板：任一可算 SE ≥ 阈值，或 min(pos,neg) < 10
        se_max = np.nanmax([x for x in (hm, boot, dl_se) if x is not None])
        noise_floor = bool((np.isfinite(se_max) and se_max >= NOISE_FLOOR_SE)
                           or min(n_pos, n_neg) < 10)
        rows.append({
            "subgroup": gname, "is_joint": is_joint, "n": int(n),
            "n_pos": n_pos, "n_neg": n_neg,
            "auc": (None if auc is None else round(float(auc), 4)),
            "se_hanley_mcneil": (None if not np.isfinite(hm) else round(hm, 4)),
            "se_delong": (None if not np.isfinite(dl_se) else round(dl_se, 4)),
            "se_bootstrap": (None if not np.isfinite(boot) else round(boot, 4)),
            "noise_floor": noise_floor,
        })
    return rows


def _delong_single_se(y: np.ndarray, s: np.ndarray) -> float:
    """单条 ROC 的 DeLong AUC SE（复用 fast delong；退化时 nan）。"""
    from src.training.harness.significance import _fast_delong
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(-y, kind="mergesort")
    n_pos = int((y == 1).sum())
    preds = np.vstack((s, s))[:, order]  # 复用双行接口，取对角
    _aucs, cov = _fast_delong(preds, n_pos)
    var = float(cov[0, 0])
    return float(np.sqrt(var)) if var > 0 else float("nan")


def other_datasets_min_cell() -> list[dict]:
    """其余数据集 worst-group 各格最小 n（说明 joint 小格问题主要属 HAM）。"""
    out = []
    for name in ("mimic", "chexpert", "fitzpatrick", "papila"):
        spec = DATASETS[name]
        if spec["oof"]:
            continue  # PAPILA 走 OOF，另计
        y, s, a = _load(spec["pdir"], "erm", spec["cfg"]["erm"], 42, spec["ham_filter"])
        vec = subgroup_auc_vector(spec["key"], y, s, **{k: a[k] for k in spec["attrs"]})
        ns = [n for _k, (_auc, n) in vec.items()]
        pos = []
        # 最小格的 n_pos 需重建掩码；这里只给最小 n 作量级对照
        out.append({"dataset": name, "n_subgroups": len(vec),
                    "min_cell_n": int(min(ns)), "median_cell_n": int(np.median(ns))})
    return out


def main() -> None:
    """HAM 逐格诊断 + 其余数据集最小格 n 对照 → JSON + 表。"""
    rows = diagnose_ham()
    others = other_datasets_min_cell()

    print(f"\nHAM canonical 14 子群评估端诊断（seed-ensemble；噪声地板阈 SE≥{NOISE_FLOOR_SE} 或 min(pos,neg)<10）")
    print(f"{'子群':<24}{'n':>5}{'pos':>5}{'neg':>5}{'AUC':>8}{'SE_HM':>8}{'SE_DL':>8}{'SE_boot':>9}  地板")
    print("-" * 90)
    for r in rows:
        f = lambda x: ("  -  " if x is None else f"{x:.4f}")
        flag = "⚠️噪声地板" if r["noise_floor"] else ""
        print(f"{r['subgroup']:<24}{r['n']:>5}{r['n_pos']:>5}{r['n_neg']:>5}"
              f"{f(r['auc']):>8}{f(r['se_hanley_mcneil']):>8}{f(r['se_delong']):>8}"
              f"{f(r['se_bootstrap']):>9}  {flag}")

    n_floor = sum(r["noise_floor"] for r in rows)
    n_joint_floor = sum(r["noise_floor"] and r["is_joint"] for r in rows)
    print(f"\n噪声地板格数: {n_floor}/14（其中 joint {n_joint_floor}/8）")
    print("\n其余数据集 worst-group 最小格 n 对照：")
    for o in others:
        print(f"  {o['dataset']:<12} 子群数={o['n_subgroups']} 最小格 n={o['min_cell_n']} 中位 n={o['median_cell_n']}")

    out_path = OUT / "ham10000" / "subgroup_n_diagnosis.json"
    json.dump({"ham_subgroups": rows, "n_noise_floor": n_floor,
               "n_joint_noise_floor": n_joint_floor, "noise_floor_se": NOISE_FLOOR_SE,
               "other_datasets_min_cell": others},
              open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n已保存: {out_path}")


if __name__ == "__main__":
    main()
