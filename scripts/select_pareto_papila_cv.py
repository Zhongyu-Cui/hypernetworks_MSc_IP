"""
PAPILA 全-CV 折聚合 Minimax-Pareto 选择器（协议 C5.5，PAPILA 特例）
====================================================================
背景：PAPILA test≈84 眼极小，C5 走「全 CV：6 配置 × 5 折都跑」（用户裁定）。搜索阶段每方法
产出 6 config × 5 fold = 30 条 val 日志（fold↔seed 一一映射，seed=42+fold）。

标准 `harness/pareto.py` 的 `select_config` 把每条 val 日志当独立候选（config×fold），
`select_minimax` 只会挑单个「幸运折」的 config——对 PAPILA 小样本不稳健。本脚本改为
**先按 config 跨 5 折聚合子群 AUC 向量（逐键取折均值）→ 6 个聚合候选 → 再 minimax**，
使超参选择建立在折平均之上（对齐方案文档「5 折 OOF 池化抢功效」精神），而非单折噪声。

聚合候选的 worst-case 定义为「折均子群向量的最小值」（与 Pareto 在同一折均向量上做支配一致）。

用法：
  python scripts/select_pareto_papila_cv.py --method erm       [--selection worstcase]
  python scripts/select_pareto_papila_cv.py --method hyperhead
  # 缺省对 4 方法各跑一次
"""

import argparse
from collections import defaultdict
from pathlib import Path

from src.training.harness.pareto import (
    Candidate, Selection, SELECTIONS, load_candidates, select_minimax,
)

DEFAULT_VAL_LOG_DIR = Path(
    "/vol/biomedic2/bglocker_studproj/zc125/outputs/papila/val_logs"
)
DEFAULT_METHODS = ("erm", "hyperhead", "hyperfusion", "hyperadapt")


def _mean_ignore_none(values: list[float | None]) -> float | None:
    """对一组可能含 None 的值取均值；全为 None 时返回 None（该子群在所有折均无效）。"""
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def aggregate_folds(candidates: list[Candidate]) -> list[Candidate]:
    """
    把 config×fold 的候选按 config_tag 跨折聚合为「每 config 一个」的折均候选。

    逐子群键取折均（忽略 None）；overall = 折均 overall；worst-case = 折均子群向量的最小值
    （只在有非 None 键时定义，否则 None）。聚合候选的 seed/epoch 无单折意义，置 -1 作占位。

    Args:
        candidates: 同一方法、config×fold 的塌缩候选列表（每折一个，seed=42+fold）。

    Returns:
        每 config_tag 一个的折均候选列表（长度 = 配置数，正常为 6）。
    """
    by_config: dict[str, list[Candidate]] = defaultdict(list)
    for cand in candidates:
        by_config[cand.config_tag].append(cand)

    aggregated: list[Candidate] = []
    for config_tag, fold_cands in sorted(by_config.items()):
        # 固定子群键顺序（取首折的键序；各折键集应一致）
        keys = list(fold_cands[0].subgroup_auc.keys())
        mean_subgroup: dict[str, float | None] = {
            k: _mean_ignore_none([c.subgroup_auc.get(k) for c in fold_cands]) for k in keys
        }
        valid_vals = [v for v in mean_subgroup.values() if v is not None]
        worst = min(valid_vals) if valid_vals else None
        mean_overall = sum(c.overall_auc for c in fold_cands) / len(fold_cands)
        aggregated.append(Candidate(
            method=fold_cands[0].method, config_tag=config_tag, seed=-1, epoch=-1,
            selection=fold_cands[0].selection,
            overall_auc=mean_overall, worst_case_auc=worst, subgroup_auc=mean_subgroup,
        ))
    return aggregated


def select_config_cv(
    val_log_dir: Path, method: str, selection: Selection,
) -> tuple[str, str]:
    """
    端到端：为某方法做 PAPILA 全-CV 折聚合 Minimax-Pareto 选择。

    Args:
        val_log_dir: PAPILA val 日志目录（含 30 条 <method>_<config>_seed4{2..6}.jsonl）。
        method     : 目标方法名。
        selection  : model selection 策略（默认 worstcase）。

    Returns:
        (选定 config_tag, 人类可读摘要文本)。
    """
    # seeds=None 全纳入（5 折 = seed 42..46 全取）；每折先各自塌缩到选定 epoch
    fold_cands = load_candidates(val_log_dir, method, selection, seeds=None)
    n_folds = len({c.seed for c in fold_cands})
    agg = aggregate_folds(fold_cands)
    sel = select_minimax(agg)

    lines = [
        f"[PAPILA 全-CV 折聚合 Minimax-Pareto | method={method} selection={selection}] "
        f"{len(fold_cands)} 折候选（{n_folds} 折 × {len(agg)} config）→ 折均 {len(agg)} 候选",
        f"  选定配置: {sel.selected.config_tag}  "
        f"(折均 worst={sel.selected.worst_case_auc:.4f}, overall={sel.selected.overall_auc:.4f})",
        "  折均候选（按 worst 降序）:",
    ]
    for cand in sorted(agg, key=lambda c: (c.worst_case_auc or -1.0), reverse=True):
        mark = " ←选定" if cand.config_tag == sel.selected.config_tag else ""
        w = "None" if cand.worst_case_auc is None else f"{cand.worst_case_auc:.4f}"
        lines.append(f"    {cand.config_tag}: worst={w} overall={cand.overall_auc:.4f}{mark}")
    return sel.selected.config_tag, "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """解析命令行参数（--method 可选；缺省对 4 方法各跑；--selection / --val-log-dir 可选）。"""
    p = argparse.ArgumentParser(description="PAPILA 全-CV 折聚合 Minimax-Pareto 选择（C5.5）")
    p.add_argument("--method", type=str, default=None,
                   help="目标方法（erm/hyperhead/hyperfusion/hyperadapt）；缺省全跑。")
    p.add_argument("--selection", type=str, default="worstcase", choices=list(SELECTIONS),
                   help="model selection 策略（默认 worstcase）。")
    p.add_argument("--val-log-dir", type=str, default=str(DEFAULT_VAL_LOG_DIR),
                   help="PAPILA val 日志目录。")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    val_log_dir = Path(args.val_log_dir)
    methods = [args.method] if args.method else list(DEFAULT_METHODS)
    for method in methods:
        print(f"############################## {method} ##############################")
        _config_tag, summary = select_config_cv(val_log_dir, method, args.selection)
        print(summary)
        print()


if __name__ == "__main__":
    main()
