"""
E1 第二级选择：每个 cell 按「五折 validation marginal-worst 均值」选 config
===========================================================================
docs/conditioning_ablation_plan.md §3.2 的**两级选择**：
  ① 每个 config 内以 **validation Overall AUC** 选 epoch（= 训练时的早停 / best_overall checkpoint）；
  ② 每个 cell 以**五折 validation marginal-worst 均值**选 config，**平手时以 validation Overall 破局**。
  · **pooled-OOF / test 只用于最终评估，不参与任何选择**——本脚本只读 val 日志，从不碰 test 预测。

**两个易错点（本脚本显式规避）**：
1. **不能用 val 日志里的 `worst_case_auc` 字段**：它是对**全部 14 个子群**（含 sex×age 交叉）取 min，
   而本消融的候选集是 **E-audit.1 冻结的「仅 age 4 组」**（HAM-ID regime，比既有 marginal 含 sex 更窄）。
   必须从 `subgroup_auc` 按冻结 manifest 重算，否则口径漂移（实测同一 epoch：全 14 组 min=0.700，
   age 4 组 min=0.769）。
2. **epoch 必须取「val Overall 最优」那一条**，而非最后一条或 val-worst 最优那条——与 best_overall
   checkpoint（主结果所用）严格对应，否则选择口径与评估口径不一致。

输出：`outputs/conditioning_ablation/ham10000/cv5/selected_configs.json`（供确认阶段补 seed 43/44）。

运行（轻量，实验室机器）：
    python scripts/e1_select_config.py --dataset ham10000
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from src.paths import OUTPUTS_DIR

ABLATION_ROOT = OUTPUTS_DIR / "conditioning_ablation"
MANIFEST = ABLATION_ROOT / "eaudit_candidate_groups.json"

CELLS = ("condnet_erm", "condnet_head", "condnet_deep", "condnet_full")
N_FOLDS = 5
SEARCH_SEED = 42
# 数据集 → (outputs 子目录, E-audit.1 manifest regime 名)。
# ⚠️ MIMIC 的消融产物落 mimic_cxr/（与 train_condnet 的 output_subdir 一致，非 "mimic"）。
# ⚠️ MIMIC 的 config 选择**只用 MIMIC validation**（plan §3.2 零泄漏，CheXpert 不参与选择）；
#    候选组键与 M2C-OOD manifest 的 6 个边缘组一致（sex∪race∪age），MIMIC val 上各组样本充足必过门槛，
#    故直接复用 M2C-OOD 的边缘组键（键定义与数据集无关，门槛在 CheXpert target 上已验证全过）。
DATASET_SPEC = {
    "ham10000": ("ham10000", "HAM-ID"),
    "mimic": ("mimic_cxr", "M2C-OOD"),
}


def load_candidate_groups(dataset: str) -> list[str]:
    """从 E-audit.1 冻结 manifest 读该 regime 的候选组（只取通过门槛者）。"""
    manifest = json.loads(MANIFEST.read_text())
    regime = manifest["regimes"][DATASET_SPEC[dataset][1]]
    return [g for g, c in regime["groups"].items() if c["passes"]]


def best_epoch_record(log_path: Path) -> dict:
    """
    读一条 run 的 val 日志，返回 **val Overall AUC 最优** 那个 epoch 的记录
    （对应 best_overall checkpoint，§3.2 的第①级选择）。
    """
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"空 val 日志：{log_path}")
    return max(records, key=lambda r: r["overall_auc"])


def marginal_worst(record: dict, candidate_groups: list[str]) -> float:
    """
    在**冻结候选组**上重算 marginal worst = min_g AUC_g。

    ⚠️ 不用 record["worst_case_auc"]（那是全 14 子群含 joint 的 min，口径不同）。

    Args:
        record          : 某 epoch 的 val 记录。
        candidate_groups: E-audit.1 冻结的候选组名。

    Returns:
        候选组上的最小 AUC；某组缺失/为 None 时跳过（不应发生，会断言）。
    """
    vals = []
    for g in candidate_groups:
        auc = record["subgroup_auc"].get(g)
        if auc is not None:
            vals.append(float(auc))
    assert vals, f"候选组在 val 记录中全缺失：{candidate_groups}"
    return min(vals)


def collect(dataset: str, cell: str, config_tag: str, candidate_groups: list[str]) -> tuple[float, float, int]:
    """
    汇总某 (cell, config) 的五折 val 指标。

    Returns:
        (五折 val marginal-worst 均值, 五折 val Overall 均值, 实际折数)。
    """
    worsts, overalls = [], []
    for fold in range(N_FOLDS):
        log = (ABLATION_ROOT / DATASET_SPEC[dataset][0] / f"cv{N_FOLDS}" / f"fold{fold}" / "val_logs"
               / f"{cell}_{config_tag}_seed{SEARCH_SEED}.jsonl")
        if not log.exists():
            raise FileNotFoundError(f"缺 val 日志：{log}")
        rec = best_epoch_record(log)
        worsts.append(marginal_worst(rec, candidate_groups))
        overalls.append(float(rec["overall_auc"]))
    return float(np.mean(worsts)), float(np.mean(overalls)), len(worsts)


# 6 配置网格的 tag 形状（lr3e-05_wd1e-04 等）。用它过滤，因为 glob 的 `{cell}_*` 会连带匹配到
# 派生臂的日志：E3 敏感性实验把 C-Deep/C-Full 的 no-fc 变体写成 `condnet_deep_nofc_<tag>`，
# 前缀与 `condnet_deep` 相同，不过滤就会让本函数返回 7 个 tag 并触发下游断言。
CONFIG_TAG_RE = re.compile(r"^lr[0-9eE.+-]+_wd[0-9eE.+-]+$")


def discover_configs(dataset: str, cell: str) -> list[str]:
    """从 fold0 的 val_logs 发现该 cell 跑过的全部 config_tag（应为 6 个）。"""
    d = ABLATION_ROOT / DATASET_SPEC[dataset][0] / f"cv{N_FOLDS}" / "fold0" / "val_logs"
    tags = sorted({p.name[len(cell) + 1:-len(f"_seed{SEARCH_SEED}.jsonl")]
                   for p in d.glob(f"{cell}_*_seed{SEARCH_SEED}.jsonl")})
    return [t for t in tags if CONFIG_TAG_RE.match(t)]


def main() -> None:
    p = argparse.ArgumentParser(description="E1 §3.2 第二级选择：五折 val marginal-worst 均值选 config")
    p.add_argument("--dataset", default="ham10000", choices=sorted(DATASET_SPEC))
    args = p.parse_args()

    groups = load_candidate_groups(args.dataset)
    print("E1 config 选择（plan §3.2 两级选择）")
    print(f"  ① epoch = val Overall AUC 最优（对应 best_overall checkpoint）")
    print(f"  ② config = 五折 val **marginal-worst 均值**最大，平手用 val Overall 破局")
    print(f"  候选组（E-audit.1 冻结，{DATASET_SPEC[args.dataset][1]}）= {groups}")
    print("  ⚠️ 只读 val 日志；pooled-OOF/test 不参与选择\n")

    selected: dict[str, str] = {}
    detail: dict[str, dict] = {}
    for cell in CELLS:
        tags = discover_configs(args.dataset, cell)
        assert len(tags) == 6, f"{cell} 期望 6 个 config，实得 {len(tags)}：{tags}"
        rows = []
        for tag in tags:
            w, o, n = collect(args.dataset, cell, tag, groups)
            rows.append((tag, w, o, n))
        # 排序：marginal-worst 均值优先，平手用 Overall 破局（§3.2）
        rows_sorted = sorted(rows, key=lambda r: (-r[1], -r[2]))
        best = rows_sorted[0]
        selected[cell] = best[0]
        detail[cell] = {"selected": best[0],
                        "candidates": [{"config": t, "val_marginal_worst_mean": w,
                                        "val_overall_mean": o, "n_folds": n} for t, w, o, n in rows]}
        print(f"--- {cell} ---")
        print(f"  {'config':<18s} {'val marg-worst(5折均)':>20s} {'val Overall(5折均)':>18s}")
        for t, w, o, n in rows_sorted:
            mark = "  ← 选定" if t == best[0] else ""
            print(f"  {t:<18s} {w:>20.4f} {o:>18.4f}{mark}")

    out = ABLATION_ROOT / DATASET_SPEC[args.dataset][0] / f"cv{N_FOLDS}" / "selected_configs.json"
    out.write_text(json.dumps({"dataset": args.dataset, "rule": "五折 val marginal-worst 均值最大，平手用 val Overall",
                               "candidate_groups": groups, "config": selected, "detail": detail},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n选定 config：{json.dumps(selected, ensure_ascii=False)}")
    print(f"已落盘 → {out}")


if __name__ == "__main__":
    main()
