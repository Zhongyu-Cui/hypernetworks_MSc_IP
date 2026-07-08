"""
R3.4 —— age 分箱敏感性：检验 MIMIC / CheXpert 的 `I(Y;age|X)≈0` 是否为分箱伪影
================================================================================

动机（评审 R3.4）：D2 的 age 条件互信息在缓存里是**二值化 @60** 的单一分箱结果。若换分箱
就翻盘，则「null」只是分箱选择的伪影。本脚本在**同一份冻结特征缓存**上，用 ≥2 种年龄分箱
重估 `I(Y;age|X)`，看 null 是否稳健。

复用 estimate_conditional_mi 的原始件（`_build_design` / `_eval_test_logit` / `_per_sample_nll`）
与 summarize_conditional_mi_ci 的 Rubin 合并。缓存里的 `age` 已是 @60 二值，故从 split CSV 按
**行序**（缓存以 shuffle=False 抽取，已逐行核对 age/label 完全对齐）取回**原始 age** 再重新分箱。

分箱方案（阈值/分位边界均由 train 原始 age 定，跨 split 一致）：
    bin2@50 / bin2@60（基线复现）/ bin2@70   —— 三个二值阈值
    bin3_tertile  —— 3 分位（3 组）
    bin4_quartile —— 4 分位（4 组）
判据：若各分箱下 `I(Y;age|X)` 的 95%CI 一律含 0 或落负地板（= 未超 MDE），则 null 稳健、非分箱伪影。

用法：python -m scripts.agebin_sensitivity_cmi   （用缓存特征，GPU 上数分钟）
输出：outputs/agebin_sensitivity_cmi.json + 终端表。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.estimate_conditional_mi import _build_design, _eval_test_logit, _per_sample_nll
from scripts.summarize_conditional_mi_ci import combine_rubin, _se_from_ci

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs")
SEEDS = (42, 43, 44)
SPLITS = ("train", "val", "test")
N_BOOTSTRAP = 1000

DATASETS = [
    ("MIMIC-CXR", "mimic_cxr", "mimic_cxr_baseline.yaml"),
    ("CheXpert", "chexpert_cxr", "chexpert_baseline.yaml"),
]


def load_raw_age(subdir: str, cfgname: str) -> dict[str, np.ndarray]:
    """从 split CSV 按行序取回原始 age（缓存已逐行对齐验证）。"""
    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / cfgname, encoding="utf-8"))
    split_dir = REPO_ROOT / cfg["data"]["split_dir"]
    return {sp: pd.read_csv(split_dir / f"{sp}.csv")["age"].to_numpy(dtype=np.float64)
            for sp in SPLITS}


def make_binnings(train_age: np.ndarray) -> dict[str, dict]:
    """据 train 原始 age 定义各分箱的边界与组数（跨 split 用同一套边界）。"""
    q = lambda ps: np.quantile(train_age, ps)
    return {
        "bin2@50":       {"edges": [50.0],            "n_groups": 2},
        "bin2@60":       {"edges": [60.0],            "n_groups": 2},  # 基线复现
        "bin2@70":       {"edges": [70.0],            "n_groups": 2},
        "bin3_tertile":  {"edges": list(q([1/3, 2/3])), "n_groups": 3},
        "bin4_quartile": {"edges": list(q([.25, .5, .75])), "n_groups": 4},
    }


def bin_age(age: np.ndarray, edges: list[float]) -> np.ndarray:
    """按升序 edges 把连续 age 映射到 0..len(edges) 的组 id。"""
    return np.digitize(age, np.asarray(edges, dtype=np.float64)).astype(np.int64)


def load_feats(subdir: str, seed: int) -> dict[str, dict[str, np.ndarray]]:
    """载入某 seed 的 train/val/test 冻结特征缓存。"""
    cache = OUTPUTS_ROOT / subdir / "cmi_cache"
    out = {}
    for sp in SPLITS:
        with np.load(cache / f"feat_seed{seed}_{sp}.npz") as z:
            out[sp] = {k: z[k] for k in z.files}
    return out


def age_delta_nll(
    feats: dict, gid: dict[str, np.ndarray], n_groups: int, rng: np.random.Generator,
) -> tuple[float, tuple[float, float]]:
    """
    对给定 age 分箱（组 id gid，n_groups 组）估计 I(Y;age|X) 的 int-model ΔNLL + test bootstrap CI。
    f: Y~φ；int: 逐组独立 [1,φ]。ΔNLL = mean(NLL_f − NLL_int)。
    """
    tr, va, te = feats["train"], feats["val"], feats["test"]
    y_test = te["label"].astype(np.float64)
    logit_f = _eval_test_logit(
        _build_design(tr["feat"], None, 0, "base"), tr["label"],
        _build_design(va["feat"], None, 0, "base"), va["label"],
        _build_design(te["feat"], None, 0, "base"),
    )
    logit_g = _eval_test_logit(
        _build_design(tr["feat"], gid["train"], n_groups, "int"), tr["label"],
        _build_design(va["feat"], gid["val"], n_groups, "int"), va["label"],
        _build_design(te["feat"], gid["test"], n_groups, "int"),
    )
    llr = _per_sample_nll(logit_f, y_test) - _per_sample_nll(logit_g, y_test)
    point = float(llr.mean())
    boot = np.array([llr[rng.integers(0, llr.size, llr.size)].mean() for _ in range(N_BOOTSTRAP)])
    ci = np.percentile(boot, [2.5, 97.5])
    return point, (float(ci[0]), float(ci[1]))


def verdict(lo: float, hi: float) -> str:
    """三态判定（同 R3.1）：CI 下界>0=确认>0；CI 全<0=噪声地板；跨 0=不可判定。"""
    if lo > 0:
        return "confirmed>0"
    if hi < 0:
        return "noise-floor(≈0)"
    return "undecidable(spans 0)"


def main() -> None:
    """两数据集 × 各分箱 × 3 seed → Rubin 合并 → 表 + JSON。"""
    results: list[dict] = []
    for display, subdir, cfgname in DATASETS:
        raw_age = load_raw_age(subdir, cfgname)
        binnings = make_binnings(raw_age["train"])
        feats_by_seed = {s: load_feats(subdir, s) for s in SEEDS}

        for bname, bspec in binnings.items():
            gid = {sp: bin_age(raw_age[sp], bspec["edges"]) for sp in SPLITS}
            points, ses = [], []
            for s in SEEDS:
                rng = np.random.default_rng(s)
                pt, (lo, hi) = age_delta_nll(feats_by_seed[s], gid, bspec["n_groups"], rng)
                points.append(pt)
                ses.append(_se_from_ci(lo, hi))
            comb = combine_rubin(points, ses)
            row = {
                "dataset": display, "binning": bname, "n_groups": bspec["n_groups"],
                "edges": [round(float(e), 1) for e in bspec["edges"]],
                "point": comb["point"], "ci95": comb["ci95"],
                "verdict": verdict(comb["ci95"][0], comb["ci95"][1]),
            }
            results.append(row)
            lo, hi = comb["ci95"]
            print(f"{display:<11}{bname:<15}groups={bspec['n_groups']} "
                  f"edges={row['edges']!s:<22} I(Y;age|X)={comb['point']:+.5f} "
                  f"[{lo:+.5f},{hi:+.5f}]  {row['verdict']}")

    out_path = OUTPUTS_ROOT / "agebin_sensitivity_cmi.json"
    json.dump({"results": results, "n_bootstrap": N_BOOTSTRAP, "seeds": list(SEEDS)},
              open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n已保存: {out_path}")


if __name__ == "__main__":
    main()
