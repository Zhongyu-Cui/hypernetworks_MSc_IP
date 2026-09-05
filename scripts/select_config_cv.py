"""
CV-OOF 折均 marginal-worst config 选择器（方案 S1，5 数据集通用）
==================================================================
`docs/oof_selection_rollout_plan.md` 三决策之②：**config 选择（6→1）用「折均 marginal-worst」**
（minimax，平手用折均 Overall 破）。本脚本把 `select_pareto_papila_cv.py`（PAPILA 专用、canonical
worst + Pareto 前沿）泛化为**数据集参数化**、且把选择信号从 canonical-worst 换成 **marginal-worst**。

**三决策口径（与方案文档一致）**：
  ① 早停 / epoch：Overall AUC argmax（本脚本用 `selection="overall"` 逐折塌缩到该 epoch）。
  ② config 选择：逐 config 跨 5 折求「val marginal-worst」折均 → 取最高（平手用折均 Overall 破）。
  ③ 评估：不在此，见 `cv_oof_report.py`（pooled-OOF canonical-worst）。

**marginal-worst**（与 canonical-worst 的区别，方案文档 §0）：只在**单属性子群**（子群键不含 `|`）
上取最差 AUC，**丢弃样本极少的交叉小格**（如 HAM 的 `sex:Female|age:80+` n≈27）。它是**稳定的选择
信号**（逐-epoch 抖动 −35%）；交叉子群的真实公平留到 pooled-OOF 上用 canonical-worst 报（那里 n 够大）。
对只有单一属性轴的 Fitzpatrick（6 肤色）marginal == canonical（无 `|` 键），退化为同一量，安全。

**为何是纯 argmax 而非 Pareto-then-minimax**：方案 S1 明确写「求折均 → minimax 选最高，平手用折均
Overall 破」——即在折均候选上直接取 marginal-worst 最大者。Pareto 前沿是多目标去支配，这里选择目标
已被 marginal-worst 标量单一化，故直接 argmax(marginal_worst, overall) 即忠实实现，无需前沿步骤。

用法：
  python scripts/select_config_cv.py --dataset ham10000            # 4 方法各选一个 config
  python scripts/select_config_cv.py --dataset papila --method erm
  python scripts/select_config_cv.py --self-test                   # 逻辑自检
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.training.harness.pareto import Candidate, load_candidates
from src.training.harness.run import Selection, swad_method_name
from src.paths import OUTPUTS_DIR

OUTPUTS_ROOT = OUTPUTS_DIR

# 数据集 → val 日志目录（PAPILA 原生全-CV 无 cv5 子目录；余者走 cv5/）。
# 键为 subgroup_auc 的数据集 key；值为 outputs 下相对目录。
DATASET_VAL_LOG_DIR: dict[str, str] = {
    "ham10000": "ham10000/cv5/val_logs",
    "papila": "papila/val_logs",
    "fitzpatrick": "fitzpatrick/cv5/val_logs",
    "chexpert": "chexpert_cxr/cv5/val_logs",
    "mimic": "mimic_cxr/cv5/val_logs",
}

# 需做 config 选择的方法（swad/roc 派生自 erm、复用 erm 选定 config，故不单独选）。
# `groupdro` 是独立训练的基线（自己的 6 配置搜索），故与 erm/HN 同列参与选择。
DEFAULT_METHODS = ("erm", "groupdro", "hyperhead", "hyperfusion", "hyperadapt")


def _mean_ignore_none(values: list[float | None]) -> float | None:
    """对一组可能含 None 的值取均值；全 None 返回 None。"""
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def marginal_worst(subgroup_auc: dict[str, float | None]) -> float | None:
    """
    折均子群 AUC 向量的 **marginal-worst**：只取单属性子群（键不含 `|`）的最小非 None AUC。

    Args:
        subgroup_auc: 子群键 -> AUC（或 None）的字典（通常为跨折折均后的向量）。

    Returns:
        marginal 子群里的最小 AUC；无有效 marginal 子群时 None。
    """
    vals = [v for k, v in subgroup_auc.items() if v is not None and "|" not in k]
    return min(vals) if vals else None


def aggregate_folds(candidates: list[Candidate]) -> list[Candidate]:
    """
    把 config×fold 候选按 config_tag 跨折聚合为「每 config 一个」的折均候选。

    逐子群键取折均（忽略 None）；overall = 折均 overall；worst_case_auc 字段在此填 **marginal-worst**
    （折均向量的 marginal 最小值），使下游 argmax 直接读该字段即用 marginal 信号。seed/epoch 置 -1。

    Args:
        candidates: 同一方法、已按 selection 塌缩到选定 epoch 的 config×fold 候选列表。

    Returns:
        每 config_tag 一个的折均候选（worst_case_auc = marginal-worst）。
    """
    by_config: dict[str, list[Candidate]] = defaultdict(list)
    for cand in candidates:
        by_config[cand.config_tag].append(cand)

    aggregated: list[Candidate] = []
    for config_tag, fold_cands in sorted(by_config.items()):
        keys = list(fold_cands[0].subgroup_auc.keys())  # 固定键序（各折应一致）
        mean_sub: dict[str, float | None] = {
            k: _mean_ignore_none([c.subgroup_auc.get(k) for c in fold_cands]) for k in keys
        }
        mean_overall = sum(c.overall_auc for c in fold_cands) / len(fold_cands)
        aggregated.append(Candidate(
            method=fold_cands[0].method, config_tag=config_tag, seed=-1, epoch=-1,
            selection=fold_cands[0].selection, overall_auc=mean_overall,
            worst_case_auc=marginal_worst(mean_sub), subgroup_auc=mean_sub,
        ))
    return aggregated


def select_config_cv(
    val_log_dir: Path, method: str, epoch_selection: Selection = "overall",
) -> tuple[str, str]:
    """
    端到端：为某方法做「折均 marginal-worst」config 选择。

    Args:
        val_log_dir    : 该数据集 CV val 日志目录（含 <method>_<config>_seed4{2..6}.jsonl）。
        method         : 目标方法名。
        epoch_selection: 逐折塌缩到哪个 epoch 的 model selection（决策①，默认 overall=Overall AUC 早停）。

    Returns:
        (选定 config_tag, 人类可读摘要文本)。
    """
    # seeds=None 全纳入 5 折；每折先按 epoch_selection 塌缩到选定 epoch 的子群向量
    fold_cands = load_candidates(val_log_dir, method, epoch_selection, seeds=None)
    n_folds = len({c.seed for c in fold_cands})
    agg = aggregate_folds(fold_cands)
    # 纯 argmax：marginal-worst 最高；平手用折均 Overall 破（None marginal 以 -inf 垫底）
    selected = max(agg, key=lambda c: (
        float("-inf") if c.worst_case_auc is None else c.worst_case_auc, c.overall_auc))

    lines = [
        f"[CV 折均 marginal-worst | method={method} epoch_sel={epoch_selection}] "
        f"{len(fold_cands)} 折候选（{n_folds} 折 × {len(agg)} config）→ 折均 {len(agg)} 候选",
        f"  选定配置: {selected.config_tag}  "
        f"(折均 marginal-worst={_fmt(selected.worst_case_auc)}, overall={selected.overall_auc:.4f})",
        "  折均候选（按 marginal-worst 降序）:",
    ]
    for cand in sorted(agg, key=lambda c: (c.worst_case_auc or -1.0), reverse=True):
        mark = " ←选定" if cand.config_tag == selected.config_tag else ""
        lines.append(
            f"    {cand.config_tag}: marginal-worst={_fmt(cand.worst_case_auc)} "
            f"overall={cand.overall_auc:.4f}{mark}")
    return selected.config_tag, "\n".join(lines)


def _fmt(v: float | None) -> str:
    """AUC 格式化（None → 'None'）。"""
    return "None" if v is None else f"{v:.4f}"


def resolve_val_log_dir(dataset: str, override: str | None) -> Path:
    """把 --dataset 解析为 val 日志目录（--val-log-dir 显式覆盖优先）。"""
    if override is not None:
        return Path(override)
    if dataset not in DATASET_VAL_LOG_DIR:
        raise ValueError(f"未知数据集 {dataset!r}，合法：{tuple(DATASET_VAL_LOG_DIR)}")
    return OUTPUTS_ROOT / DATASET_VAL_LOG_DIR[dataset]


# ============================================================
# self-test：marginal-worst 忽略 joint 小格、平手用 overall 破、Fitz 退化一致
# ============================================================
def _mk(config_tag: str, sub: dict[str, float | None], overall: float) -> Candidate:
    """构造折均候选（worst_case_auc 由 marginal_worst 填充，模拟 aggregate_folds 输出）。"""
    return Candidate(method="erm", config_tag=config_tag, seed=-1, epoch=-1,
                     selection="overall", overall_auc=overall,
                     worst_case_auc=marginal_worst(sub), subgroup_auc=dict(sub))


def _select(cands: list[Candidate]) -> str:
    """复用与主逻辑相同的 argmax。"""
    return max(cands, key=lambda c: (
        float("-inf") if c.worst_case_auc is None else c.worst_case_auc, c.overall_auc)).config_tag


def _selftest() -> None:
    """覆盖：marginal 丢 joint、canonical 会误选而 marginal 不、平手 overall 破、单轴退化。"""
    # cfgA 的 joint 小格极低(0.50)但 marginal 都高(min 0.85)；cfgB marginal 差(min 0.80)。
    # canonical-worst 会因 cfgA 的 joint 0.50 而选 cfgB；marginal-worst 应正确选 cfgA(0.85>0.80)。
    cfgA = _mk("A", {"sex:M": 0.88, "sex:F": 0.85, "age:0": 0.90,
                     "sex:F|age:3": 0.50}, overall=0.87)
    cfgB = _mk("B", {"sex:M": 0.82, "sex:F": 0.80, "age:0": 0.86,
                     "sex:F|age:3": 0.83}, overall=0.84)
    assert marginal_worst(cfgA.subgroup_auc) == 0.85
    assert marginal_worst(cfgB.subgroup_auc) == 0.80
    assert _select([cfgA, cfgB]) == "A", "marginal-worst 应忽略 joint 小格选 cfgA"

    # 平手（marginal-worst 同为 0.80）→ 用 overall 破，选 overall 更高者
    t1 = _mk("t1", {"sex:M": 0.80, "sex:F": 0.90}, overall=0.85)
    t2 = _mk("t2", {"sex:M": 0.80, "sex:F": 0.82}, overall=0.81)
    assert _select([t1, t2]) == "t1", "marginal 平手应用 overall 破"

    # 单轴数据集（Fitzpatrick：无 joint 键）→ marginal == canonical，正常选最高 min
    fa = _mk("fa", {"skin:I": 0.90, "skin:VI": 0.86}, overall=0.90)
    fb = _mk("fb", {"skin:I": 0.92, "skin:VI": 0.83}, overall=0.91)
    assert marginal_worst(fa.subgroup_auc) == 0.86 and marginal_worst(fb.subgroup_auc) == 0.83
    assert _select([fa, fb]) == "fa"

    # aggregate_folds：两折折均正确（逐键平均、marginal 忽略 joint）
    f0 = Candidate("erm", "c", 42, 1, "overall", 0.80,
                   0.7, {"sex:M": 0.80, "sex:F": 0.70, "sex:F|age:3": 0.40})
    f1 = Candidate("erm", "c", 43, 1, "overall", 0.90,
                   0.6, {"sex:M": 0.90, "sex:F": 0.60, "sex:F|age:3": 0.60})
    agg = aggregate_folds([f0, f1])
    assert len(agg) == 1
    a = agg[0]
    assert abs(a.overall_auc - 0.85) < 1e-9
    assert abs(a.subgroup_auc["sex:M"] - 0.85) < 1e-9
    assert abs(a.subgroup_auc["sex:F"] - 0.65) < 1e-9      # (0.70+0.60)/2
    assert abs(a.worst_case_auc - 0.65) < 1e-9             # marginal min，忽略 joint 折均 0.50
    print("select_config_cv self-test 全部通过 ✓（marginal 丢 joint / 平手 overall 破 / 单轴退化 / 折均）")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser(description="CV-OOF 折均 marginal-worst config 选择（S1）")
    p.add_argument("--dataset", type=str, default=None,
                   help=f"数据集 key，合法：{tuple(DATASET_VAL_LOG_DIR)}")
    p.add_argument("--method", type=str, default=None,
                   help="目标方法（缺省对 erm/hyperhead/hyperfusion/hyperadapt 各跑）")
    p.add_argument("--val-log-dir", type=str, default=None,
                   help="显式 val 日志目录（覆盖 --dataset 的默认解析）")
    p.add_argument("--epoch-selection", type=str, default="overall",
                   choices=("overall", "worstcase"),
                   help="逐折塌缩到哪个 epoch（决策①，默认 overall=Overall AUC 早停）")
    p.add_argument("--write-json", type=str, default=None,
                   help="把选定配置写入 JSON（含 swad/roc=erm 派生），供 cv_oof_report 消费")
    p.add_argument("--merge", action="store_true",
                   help="--write-json 时若目标已存在，**合并**而非覆盖：只写入本次算出的方法键，"
                        "保留文件里其它已冻结的键（如实验 G 的 <hn>_swad）。补跑单个方法时必用。")
    p.add_argument("--self-test", action="store_true", help="跑逻辑自检后退出")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        _selftest()
        return
    if args.dataset is None and args.val_log_dir is None:
        raise SystemExit("需 --dataset 或 --val-log-dir（或 --self-test）")
    val_log_dir = resolve_val_log_dir(args.dataset, args.val_log_dir)
    if not val_log_dir.is_dir():
        raise SystemExit(f"val 日志目录不存在：{val_log_dir}")
    methods = [args.method] if args.method else list(DEFAULT_METHODS)
    print(f"# 数据集={args.dataset}  val_log_dir={val_log_dir}\n")
    selected: dict[str, str] = {}
    for method in methods:
        print(f"############################## {method} ##############################")
        try:
            config_tag, summary = select_config_cv(val_log_dir, method, args.epoch_selection)
        except FileNotFoundError as exc:
            # 该数据集尚未跑该方法（如只有部分数据集有 groupdro）：跳过而非中断整批选择
            print(f"  [跳过] {exc}\n")
            continue
        print(summary)
        selected[method] = config_tag
        print()
    print("# ===== 选定配置汇总（可直接填入 cv_oof_report CFG）=====")
    for method, tag in selected.items():
        print(f"SELECTED[{method}] = {tag}")

    if args.write_json is not None:
        # swad/roc 派生自 erm、复用其选定 config（与 ham_cv_oof CFG 约定一致）
        payload = dict(selected)
        if "erm" in selected:
            payload.setdefault("swad", selected["erm"])
            payload.setdefault("roc", selected["erm"])
        # 实验 G（HN×SWAD 融合）：<hn>_swad 派生自对应 HN、复用其选定 config——与「SWAD 搭 ERM
        # 的车、不单独搜参」同规程，避免为融合臂引入额外的超参选择自由度。
        # 纯命名派生：若该 HN 未跑 --swad，此 key 只是无对应预测、被下游 load 时跳过。
        for hn in ("hyperhead", "hyperfusion", "hyperadapt"):
            if hn in selected:
                payload.setdefault(swad_method_name(hn), selected[hn])
        out = Path(args.write_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        if args.merge and out.exists():
            # 合并：以既有文件为底，只覆盖本次算出的键，其余（如实验 G 的 <hn>_swad）原样保留
            old = json.loads(out.read_text())["config"]
            merged = {**old, **payload}
            new_keys = [k for k in payload if k not in old]
            print(f"\n# 合并模式：既有 {len(old)} 键 + 本次 {len(payload)} 键 "
                  f"→ {len(merged)} 键（新增 {new_keys}）")
            payload = merged
        out.write_text(json.dumps({"dataset": args.dataset, "config": payload}, indent=2))
        print(f"\n# 已写入选定配置 JSON：{out}")


if __name__ == "__main__":
    main()
