"""
框架 1（ID）的 Holm 校正判决
================================
第 4 章 §4.3.3 的规则「一个框架把一组臂与一组参照相比时，p 值在该集合内做 Holm 校正」
适用于框架 1，但 `build_oof_results_averaging.py` 只落盘未校正的 95% 区间。本脚本读取该 JSON
里的**精确 percentile bootstrap p**（两侧、中心化到观测 Δ），按数据集分别做 Holm。

**family 的范围**取「本框架实际报告并解释的比较」= 每库 11 个：
5 个臂对 ERM（SWAD / GroupDRO / 三个 HN）+ 3 个 HN 对 SWAD + 3 个 HN 对 GroupDRO。
另打印 family=5（仅 vs ERM）与 family=9（3 HN × 3 基线）两种更窄定义下的判决，
用以检查结论是否对 family 的划法敏感——**若三者一致，则该判决与这个划分选择无关**。

运行（秒级）：
    PYTHONPATH=. python scripts/id_holm_correction.py
"""

from __future__ import annotations

import json
from pathlib import Path
from src.paths import OUTPUTS_DIR

OUTPUTS = OUTPUTS_DIR
DATASETS = [("HAM10000", "HAM10000"), ("Fitzpatrick", "Fitzpatrick17k"),
            ("MIMIC", "MIMIC-CXR"), ("CheXpert", "CheXpert")]
HN = [("hyperhead", "HyperHead"), ("hyperfusion", "HyperFusion"),
      ("hyperadapt", "HyperAdapt")]
BASE = [("erm", "ERM"), ("swad", "SWAD"), ("groupdro", "GroupDRO")]
METRIC = "marginal_worst"          # 主终点；次要终点不并入主终点的 family
# 次要终点：组间 gap 与固定操作点上的 accuracy 类读数。它们**各自**在同一个 11 比较族内
# 校正（同一族划法，只是换一个终点），**不与主终点合并**——合并会改变主终点的校正基数，
# 进而改动第 5 章「12 个比较无一存活」这条核心陈述。数字源见 build_group_levels_averaging.py。
SECONDARY = [
    ("marginal_gap", "marginal gap (AUC)"),
    ("youden_val:worst_accuracy", "worst-group accuracy"),
    ("youden_val:worst_balanced_accuracy", "worst-group balanced accuracy"),
    ("youden_val:gap_accuracy", "accuracy gap"),
    ("youden_val:gap_balanced_accuracy", "balanced accuracy gap"),
]


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm 步降法，返回逐检验是否拒绝。一旦某一步不拒绝，其后全部不拒绝。"""
    order = sorted(pvals.items(), key=lambda kv: kv[1])
    out, prev = {}, True
    for i, (k, p) in enumerate(order):
        prev = prev and p <= alpha / (len(order) - i)
        out[k] = prev
    return out


def comparisons(ds: str, res: dict, scope: str,
                metric: str = METRIC, sexage_keys: bool = True,
                ) -> dict[str, tuple[float, float, bool]]:
    """
    按 scope 取出该库的比较集合。

    Args:
        ds     : 数据集键。
        res    : 该库的结果字典（含 `vs`）。
        scope  : "11" / "9" / "5"，见模块 docstring。
        metric : 终点键（缺省主终点）。
        sexage_keys: 结果 JSON 的 HAM HN 臂是否带 `_sexage` 后缀。
                 `oof_results_averaging.json` 带；`group_levels_averaging.json` 已在
                 生成时归一为无后缀的报告臂名，故取 False。

    Returns:
        {名称: (delta, p, 未校正是否显著)}。
    """
    sx = "_sexage" if (ds == "HAM10000" and sexage_keys) else ""
    out: dict[str, tuple[float, float, bool]] = {}
    arms_vs_erm = [("swad", "SWAD"), ("groupdro", "GroupDRO")] + HN
    if scope in ("11", "5"):
        for a, al in arms_vs_erm:
            k = f"{a}{sx}_vs_erm" if (a, al) in HN else f"{a}_vs_erm"
            e = res["vs"][k][metric]
            out[f"{al} - ERM"] = (e["delta"], e["p"], e["sig"])
    if scope in ("11", "9"):
        for h, hl in HN:
            for b, bl in BASE:
                if scope == "11" and b == "erm":
                    continue                      # vs ERM 已在上面加入
                e = res["vs"][f"{h}{sx}_vs_{b}"][metric]
                out[f"{hl} - {bl}"] = (e["delta"], e["p"], e["sig"])
    return out


def main() -> None:
    res_all = json.loads(
        (OUTPUTS / "conditioning_ablation" / "oof_results_averaging.json").read_text())
    if "p" not in res_all["HAM10000"]["vs"]["swad_vs_erm"][METRIC]:
        raise SystemExit("JSON 缺 p 字段，请先跑 build_oof_results_averaging.py")

    verdicts: dict[str, dict[str, bool]] = {}
    for ds, name in DATASETS:
        comp = comparisons(ds, res_all[ds], "11")
        rej = holm({k: v[1] for k, v in comp.items()})
        verdicts[name] = rej
        print(f"\n=== {name}  (Holm within {len(comp)}, alpha=0.05) ===")
        for k, (d, p, raw) in sorted(comp.items(), key=lambda kv: kv[1][1]):
            flag = "  <-- 与未校正判决不同" if raw != rej[k] else ""
            print(f"   {k:<26} d={d:+.4f}  p={p:.4g}  "
                  f"raw={'sig' if raw else 'n.s.':<5} Holm={'sig' if rej[k] else 'n.s.'}{flag}")

    print("\n" + "=" * 70)
    print("family 划法敏感性（主终点，HN vs ERM 的判决）")
    print("=" * 70)
    for ds, name in DATASETS:
        line = [f"{name:<15}"]
        for scope in ("5", "9", "11"):
            comp = comparisons(ds, res_all[ds], scope)
            rej = holm({k: v[1] for k, v in comp.items()})
            hits = [hl for _, hl in HN if rej.get(f"{hl} - ERM", False)]
            line.append(f"family={scope}: {hits if hits else 'none'}")
        print("   ".join(line))

    out = OUTPUTS / "analysis" / "id_holm_verdicts.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdicts, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已落盘 -> {out}")

    sec_path = OUTPUTS / "conditioning_ablation" / "group_levels_averaging.json"
    if not sec_path.exists():
        print(f"[跳过次要终点] 缺 {sec_path}（先跑 build_group_levels_averaging.py）")
        return
    sec_all = json.loads(sec_path.read_text())
    sec_verdicts: dict[str, dict[str, dict[str, bool]]] = {}
    print("\n" + "=" * 70)
    print("次要终点（各自在同一个 11 比较族内 Holm，不与主终点合并）")
    print("=" * 70)
    for metric, label in SECONDARY:
        sec_verdicts[metric] = {}
        row = [f"{label:<30}"]
        for ds, name in DATASETS:
            if ds not in sec_all:
                continue
            comp = comparisons(ds, sec_all[ds], "11", metric=metric, sexage_keys=False)
            rej = holm({k: v[1] for k, v in comp.items()})
            sec_verdicts[metric][name] = rej
            # 只数「HN vs 基线」，因为这才是本报告要判决的对象
            hn_hits = sum(1 for k, v in rej.items()
                          if v and k.split(" - ")[0].startswith("Hyper"))
            row.append(f"{name}: {hn_hits}/9")
        print("   ".join(row))
    sec_out = OUTPUTS / "analysis" / "id_holm_verdicts_secondary.json"
    sec_out.write_text(json.dumps(sec_verdicts, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    print(f"\n已落盘 -> {sec_out}")


if __name__ == "__main__":
    main()
