"""
E-audit.1：label-only 冻结 worst-group 候选组与样本门槛
=======================================================
条件化范围消融方案（docs/conditioning_ablation_plan.md §3.3 / §4）要求：

  「候选组由 regime 固定、全模型共用，设最低样本门槛（**E-audit 依 label 计数定，禁按模型效应调**）」

本脚本即该冻结步骤。它**只读 split CSV 的 label 与属性列**，**从不接触任何模型预测/checkpoint**——
这是「禁按模型效应调门槛」的实现保证：门槛在看到任何模型效应之前就被冻结、并落盘成 manifest，
后续 E1/E2 的 worst-group 动态 argmin 一律从该 manifest 读候选组，杜绝事后按结果挑组。

两个 regime 的候选组（对齐 plan §3.3「HAM 候选组 = age 4 组；M→C = sex∪race∪age」）：

  · **HAM-ID**：age 4 组（**仅 age**，不含 sex）——CondNet 只条件化 age（HAM 诊断 sex 轴 I≈0、
    age 轴 I(Y;age|X)>0）。⚠️ 这比既有 cv_oof_report 的 "marginal"（HAM = sex 2 + age 4 = 6 组）
    **更窄**，是本消融特有的候选集，故必须显式冻结、不能复用既有 marginal。
  · **M2C-OOD**：sex(2) + race(2) + age(2) = 6 个边缘组（排除 sex×race×age 三阶交叉——joint 小格
    噪声大，plan 以 marginal worst 为主终点）。评估集 = **完整 CheXpert target**（full-target 口径，
    5 折 test 并集），故各折候选组计数相同。

门槛规则（**预先冻结、label-only**）：`min(n_pos, n_neg) >= MIN_CLASS_N = 50`。
理由（纯统计、与模型无关）：组内 AUC 的 Hanley–McNeil 标准误有保守上界
`SE <= sqrt(A(1-A)/min(n_pos,n_neg))`；取 A≈0.8、min_class_n=50 得 `SE <~ 0.057`，即单组 AUC 的
噪声量级不至于压倒 plan §3.3 关心的效应量级（小效应 0.005 / 强效应 0.010 是**组间配对差值**、
其配对 bootstrap 噪声远小于单组边缘 SE）。门槛只用于**剔除本质上估不稳的组**，不承担任何选择功能。

输出：`outputs/conditioning_ablation/eaudit_candidate_groups.json`（冻结 manifest）。

运行（轻量、纯 CSV，实验室机器可直接跑）：
    python scripts/eaudit_freeze_candidate_groups.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS = REPO_ROOT / "data" / "splits"
OUT_DIR = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/conditioning_ablation")

# ============================================================
# 冻结常量（改动须走方案修订，不得因结果好坏调整）
# ============================================================
# 最低样本门槛：组内两类各至少 50 例（见模块 docstring 的 Hanley–McNeil 论证）
MIN_CLASS_N: int = 50

# CXR age 二值化阈值（与 configs/*_baseline.yaml attributes.age.age_threshold 一致）
CXR_AGE_THRESHOLD: int = 60

# 候选组键名与 src/training/harness/subgroup_auc.py 的子群键**严格一致**（避免定义漂移）
HAM_AGE_NAMES: dict[int, str] = {0: "20-40", 1: "40-60", 2: "60-80", 3: "80+"}
CXR_SEX_NAMES: dict[int, str] = {0: "Male", 1: "Female"}
CXR_RACE_NAMES: dict[int, str] = {0: "White", 1: "Non-White"}
CXR_AGE_NAMES: dict[int, str] = {0: "<60", 1: ">=60"}


def _count_group(df: pd.DataFrame, mask: pd.Series) -> dict[str, int]:
    """给定布尔掩码，返回该组的 label-only 计数 {n, n_pos, n_neg}。"""
    sub = df[mask]
    n_pos = int(sub["label"].sum())
    return {"n": int(len(sub)), "n_pos": n_pos, "n_neg": int(len(sub) - n_pos)}


def _load_pooled_cv_test(split_subdir: str, cv: int = 5) -> pd.DataFrame:
    """
    读 cv{K} 五折 test 的并集。五折 test 互斥且并集 = 全库，故该并集即：
      · HAM-ID  ：pooled-OOF 评估集；
      · M2C-OOD ：完整 CheXpert target（full-target 口径）。
    """
    frames = [pd.read_csv(SPLITS / split_subdir / f"cv{cv}" / f"fold{k}" / "test.csv") for k in range(cv)]
    return pd.concat(frames, ignore_index=True)


def freeze_ham_id() -> dict:
    """
    HAM-ID regime：候选组 = age 4 组（排除 age_group==-1 的 0-20 组，对齐 MEDFAIR 与既有口径）。

    Returns:
        该 regime 的 manifest 段（含候选组计数与门槛判定）。
    """
    df = _load_pooled_cv_test("ham10000")
    df = df[df["age_group"] >= 0]                     # 0-20 排除组不是候选（embedding 亦不吃 -1）
    groups: dict[str, dict] = {}
    for a, name in HAM_AGE_NAMES.items():
        counts = _count_group(df, df["age_group"] == a)
        counts["passes"] = min(counts["n_pos"], counts["n_neg"]) >= MIN_CLASS_N
        groups[f"age:{name}"] = counts
    return {
        "regime": "HAM-ID",
        "eval_set": "pooled-OOF（cv5 五折 test 并集，过滤 age_group>=0）",
        "n_total": int(len(df)),
        "positive_rate": round(float(df["label"].mean()), 4),
        "candidate_axes": ["age"],
        "groups": groups,
    }


def freeze_m2c_ood() -> dict:
    """
    M2C-OOD regime：候选组 = sex ∪ race ∪ age 的 6 个边缘组；评估集 = 完整 CheXpert target。

    Returns:
        该 regime 的 manifest 段（含候选组计数与门槛判定）。
    """
    df = _load_pooled_cv_test("chexpert_nofinding")
    df["age_bin"] = (df["age"] >= CXR_AGE_THRESHOLD).astype(int)
    groups: dict[str, dict] = {}
    for col, names in (("sex", CXR_SEX_NAMES), ("race", CXR_RACE_NAMES), ("age_bin", CXR_AGE_NAMES)):
        axis = "age" if col == "age_bin" else col      # 键名统一为 age:（与 subgroup_auc 一致）
        for v, name in names.items():
            counts = _count_group(df, df[col] == v)
            counts["passes"] = min(counts["n_pos"], counts["n_neg"]) >= MIN_CLASS_N
            groups[f"{axis}:{name}"] = counts
    return {
        "regime": "M2C-OOD",
        "eval_set": "完整 CheXpert target（full-target 口径；cv5 五折 test 并集）",
        "n_total": int(len(df)),
        "positive_rate": round(float(df["label"].mean()), 4),
        "candidate_axes": ["sex", "race", "age"],
        "groups": groups,
    }


def _print_regime(section: dict) -> None:
    """打印单个 regime 的候选组计数表与门槛判定。"""
    print(f"\n{'=' * 78}\n{section['regime']}  |  评估集：{section['eval_set']}")
    print(f"n_total={section['n_total']:,}  正例率={section['positive_rate']:.4f}  "
          f"候选轴={'∪'.join(section['candidate_axes'])}\n{'-' * 78}")
    print(f"  {'候选组':<16s} {'n':>9s} {'n_pos':>8s} {'n_neg':>9s} {'min(class)':>11s}  门槛")
    for key, c in section["groups"].items():
        mn = min(c["n_pos"], c["n_neg"])
        print(f"  {key:<16s} {c['n']:>9,} {c['n_pos']:>8,} {c['n_neg']:>9,} {mn:>11,}  "
              f"{'✓ 通过' if c['passes'] else '✗ 剔除'}")
    n_pass = sum(c["passes"] for c in section["groups"].values())
    print(f"  → {n_pass}/{len(section['groups'])} 组通过门槛 min(n_pos,n_neg)>={MIN_CLASS_N}")


def main() -> None:
    """冻结两个 regime 的候选组与门槛，打印计数表并落盘 manifest。"""
    print("E-audit.1  label-only 冻结 worst-group 候选组与样本门槛")
    print(f"门槛规则（预先冻结、仅依 label 计数）：min(n_pos, n_neg) >= {MIN_CLASS_N}")
    print("⚠️ 本脚本不读取任何模型预测/checkpoint —— 门槛在看到模型效应前冻结。")

    manifest = {
        "min_class_n": MIN_CLASS_N,
        "rule": "min(n_pos, n_neg) >= min_class_n",
        "label_only": True,
        "cxr_age_threshold": CXR_AGE_THRESHOLD,
        "note": ("候选组键名与 src/training/harness/subgroup_auc.py 子群键一致；"
                 "HAM-ID 候选集（age 4 组）比既有 cv_oof_report 的 marginal（含 sex）更窄，为本消融特有。"),
        "regimes": {},
    }
    for section in (freeze_ham_id(), freeze_m2c_ood()):
        _print_regime(section)
        manifest["regimes"][section["regime"]] = section

    # 门槛不应把任何候选组剔空（若剔空则 regime 的 worst-group 无定义，须回方案修订）
    for name, section in manifest["regimes"].items():
        assert any(c["passes"] for c in section["groups"].values()), f"{name} 无任何候选组通过门槛"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "eaudit_candidate_groups.json"
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已冻结 manifest → {out_path}")


if __name__ == "__main__":
    main()
