"""
把 `group_levels_averaging.json` 渲染成报表（Markdown 文档 + 备用 LaTeX 片段）
==============================================================================
`build_group_levels_averaging.py` 只落 JSON；本脚本负责成表，杜绝手抄数字。产出两份：

  1. **Markdown**（缺省 `docs/between_group_gaps_and_accuracy.md`）——分隔线以上自动生成、
     重跑会覆盖，以下留给人工分析节（同既有文档的约定）。
  2. **LaTeX 片段**（`outputs/conditioning_ablation/group_tables_latex.tex`）——两张主表的
     tabular 体，等到动报告时直接取用。本脚本**不**改动 `report/` 下任何文件。

表：
  A  组间 gap 主表        四库 × 六臂：Overall AUC / worst-group AUC / marginal gap / canonical gap
  B  逐组 accuracy 主表   四库 × 六臂：Overall / worst-group / gap，accuracy 与 balanced accuracy
  C  逐组 AUC 明细        每库一块，六个边缘子群
  D  逐组 accuracy 明细   每库一块，六个边缘子群 × (accuracy, balanced accuracy)
  E  阈值规则敏感性       每库：三条规则各自的 worst-group accuracy 与 gap
  F  对 ERM 的差与区间    gap 与 accuracy 两类终点的 Δ、95% CI、是否显著

**显著性口径**：这些都是**次要终点**，按第 4 章 §4.3.3 的约定不并入主终点的 Holm family，
因此表里的显著性是**未校正**的，与 Overall AUC 同待遇；读时须按 exploratory 对待。

运行：
    python scripts/build_group_tables.py
    python scripts/build_group_tables.py --rule fpr20_val     # 换主口径重出表
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.build_group_levels_averaging import CLF_REPORT, THRESHOLD_RULES
from src.paths import OUTPUTS_DIR, REPO_ROOT

# 报告里的库名与出现顺序（PAPILA 已删；两个 OOD regime 单列一节）
ID_REGIMES = [("HAM10000", "HAM10000"), ("Fitzpatrick", "Fitzpatrick17k"),
              ("MIMIC", "MIMIC-CXR"), ("CheXpert", "CheXpert")]
OOD_REGIMES = [("M2C_OOD", "MIMIC→CheXpert"), ("C2M_OOD", "CheXpert→MIMIC")]
ARM_LABEL = {"erm": "ERM", "swad": "SWAD", "groupdro": "GroupDRO", "hyperhead": "HyperHead",
             "hyperfusion": "HyperFusion", "hyperadapt": "HyperAdapt"}
ARM_ORDER = ("erm", "swad", "groupdro", "hyperhead", "hyperfusion", "hyperadapt")
RULE_LABEL = {"youden_val": "Youden J（val）", "fpr20_val": "整体 FPR=0.2（val）",
              "half": "prob=0.5（原生）"}
CLF_LABEL = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy",
             "sensitivity": "Sensitivity", "specificity": "Specificity"}
AUTOGEN_MARK = "<!-- 以上表格由 scripts/build_group_tables.py 自动生成，重跑会覆盖至本分隔线 -->"


def fmt(v: float | None, nd: int = 4) -> str:
    """数值格式化；None（无定义）显示为破折号。"""
    return "—" if v is None else f"{v:.{nd}f}"


def fmt_delta(entry: dict | None) -> str:
    """把一条 Δ 记录渲染成 `+0.0123 [lo, hi]`，显著者加粗。"""
    if entry is None or entry.get("delta") is None:
        return "—"
    d = entry["delta"]
    ci = entry.get("ci")
    body = f"{d:+.4f}" + ("" if ci is None else f" [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    return f"**{body}**" if entry.get("sig") else body


def md_table(header: list[str], rows: list[list[str]]) -> str:
    """拼一张 Markdown 表。"""
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def scalar(res: dict, arm: str, key: str) -> float | None:
    """取某臂某标量终点（缺臂或缺值时 None）。"""
    p = res["point"].get(arm)
    return None if p is None else p["scalars"].get(key)


def _stacked(results: dict, regimes: list[tuple[str, str]],
             cols: list[tuple[str, str]]) -> str:
    """
    跨库主表的统一版式：数据集作**行组**（而非把四个库横铺成二十列，那样没法读）。

    Args:
        results: 全部 regime 的结果。
        regimes: [(regime 键, 展示名)]。
        cols   : [(标量终点键, 列名)]。

    Returns:
        Markdown 表。
    """
    rows = []
    for key, label in regimes:
        res = results.get(key)
        if res is None:
            continue
        for i, arm in enumerate(ARM_ORDER):
            if arm not in res["point"]:
                continue
            rows.append([label if i == 0 else "", ARM_LABEL[arm]]
                        + [fmt(scalar(res, arm, ck)) for ck, _cl in cols])
    return md_table(["数据集", "方法"] + [cl for _ck, cl in cols], rows)


def table_a(results: dict, regimes: list[tuple[str, str]]) -> str:
    """表 A：组间 gap 主表（AUC 侧）。"""
    return _stacked(results, regimes, [
        ("overall", "Overall AUC"), ("marginal_worst", "WG AUC"),
        ("marginal_gap", "marg gap"), ("gap", "canon gap")])


def table_b(results: dict, regimes: list[tuple[str, str]], rule: str) -> str:
    """表 B：逐组 accuracy 主表（Overall / worst-group / gap，accuracy 与 balanced accuracy）。"""
    return _stacked(results, regimes, [
        (f"{rule}:overall_accuracy", "Overall acc"), (f"{rule}:worst_accuracy", "WG acc"),
        (f"{rule}:gap_accuracy", "acc gap"),
        (f"{rule}:overall_balanced_accuracy", "Overall bacc"),
        (f"{rule}:worst_balanced_accuracy", "WG bacc"),
        (f"{rule}:gap_balanced_accuracy", "bacc gap")])


def table_c(res: dict) -> str:
    """表 C：逐组 AUC 明细（边缘子群）。"""
    groups = _marginal_groups(res, "per_group_auc")
    header = ["方法"] + [g.split(":", 1)[1] for g in groups] + ["Worst", "Gap"]
    rows = []
    for arm in ARM_ORDER:
        p = res["point"].get(arm)
        if p is None:
            continue
        vals = [fmt(p["per_group_auc"].get(g)) for g in groups]
        rows.append([ARM_LABEL[arm]] + vals +
                    [fmt(scalar(res, arm, "marginal_worst")), fmt(scalar(res, arm, "marginal_gap"))])
    return md_table(header, rows)


def table_d(res: dict, rule: str, metric: str) -> str:
    """表 D：逐组阈值指标明细（某规则某指标）。"""
    groups = _marginal_groups(res, "per_group_clf", rule)
    if not groups:
        return "_（该规则下无可用数据）_\n"
    header = ["方法"] + [g.split(":", 1)[1] for g in groups] + ["Overall", "Worst", "Gap"]
    rows = []
    for arm in ARM_ORDER:
        p = res["point"].get(arm)
        if p is None or rule not in p.get("per_group_clf", {}):
            continue
        per = p["per_group_clf"][rule]
        vals = [fmt(per.get(g, {}).get(metric)) for g in groups]
        ov = ("overall_accuracy" if metric == "accuracy" else
              "overall_balanced_accuracy" if metric == "balanced_accuracy" else None)
        rows.append([ARM_LABEL[arm]] + vals +
                    [fmt(scalar(res, arm, f"{rule}:{ov}")) if ov else "—",
                     fmt(scalar(res, arm, f"{rule}:worst_{metric}")),
                     fmt(scalar(res, arm, f"{rule}:gap_{metric}"))])
    return md_table(header, rows)


def table_e(res: dict) -> str:
    """表 E：三条阈值规则的敏感性（worst-group accuracy 与 gap）。"""
    header = ["方法"] + [f"{RULE_LABEL[r]} {c}" for r in THRESHOLD_RULES for c in ("WG acc", "gap")]
    rows = []
    for arm in ARM_ORDER:
        if arm not in res["point"]:
            continue
        row = [ARM_LABEL[arm]]
        for r in THRESHOLD_RULES:
            row += [fmt(scalar(res, arm, f"{r}:worst_accuracy")),
                    fmt(scalar(res, arm, f"{r}:gap_accuracy"))]
        rows.append(row)
    return md_table(header, rows)


def table_f(res: dict, rule: str, base: str = "erm") -> str:
    """表 F：各臂对某基线的 Δ 与 95% CI（gap 与 accuracy 类终点；未做多重校正）。"""
    keys = [("marginal_gap", "marginal gap (AUC)"), ("gap", "canonical gap (AUC)"),
            (f"{rule}:worst_accuracy", "worst-group accuracy"),
            (f"{rule}:gap_accuracy", "accuracy gap"),
            (f"{rule}:worst_balanced_accuracy", "worst-group balanced acc"),
            (f"{rule}:gap_balanced_accuracy", "balanced acc gap")]
    header = ["终点"] + [ARM_LABEL[a] for a in ARM_ORDER if a != base and a in res["point"]]
    rows = []
    for key, label in keys:
        row = [label]
        for arm in ARM_ORDER:
            if arm == base or arm not in res["point"]:
                continue
            row.append(fmt_delta(res["vs"].get(f"{arm}_vs_{base}", {}).get(key)))
        rows.append(row)
    return md_table(header, rows)


def _marginal_groups(res: dict, field: str, rule: str | None = None) -> list[str]:
    """从任一臂取边缘子群键序（各臂 schema 相同）。"""
    for arm in ARM_ORDER:
        p = res["point"].get(arm)
        if p is None:
            continue
        d = p[field] if rule is None else p[field].get(rule)
        if d:
            return [g for g in d if "|" not in g]
    return []


def thresholds_note(res: dict, rule: str) -> str:
    """把各臂逐折阈值列出来——它们的离散程度本身就是「不能池化」的证据。"""
    lines = []
    for arm in ARM_ORDER:
        p = res["point"].get(arm)
        if p is None or rule not in p.get("thresholds", {}):
            continue
        ts = ", ".join(f"{t:.3f}" for t in p["thresholds"][rule])
        lines.append(f"- {ARM_LABEL[arm]}：{ts}")
    return "\n".join(lines) + "\n"


def latex_main_tables(results: dict, rule: str) -> str:
    """两张主表的 LaTeX tabular 体（备用；本脚本不动 report/）。"""
    out = ["% 由 scripts/build_group_tables.py 生成；粘进报告前请再核对列宽与表头",
           "% 表 A：Overall / worst-group / between-group gap（marginal 分区）"]
    out.append(r"\begin{tabular}{l" + "ccc" * len(ID_REGIMES) + "}")
    out.append(r"\toprule")
    out.append("& " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{lab}}}" for _k, lab in ID_REGIMES)
               + r" \\")
    out.append(" ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}"
                        for i in range(len(ID_REGIMES))))
    out.append("Method & " + " & ".join(["Ov & WG & Gap"] * len(ID_REGIMES)) + r" \\")
    out.append(r"\midrule")
    for arm in ARM_ORDER:
        cells = []
        for key, _lab in ID_REGIMES:
            res = results.get(key)
            cells += ([fmt(scalar(res, arm, k), 3) for k in
                       ("overall", "marginal_worst", "marginal_gap")]
                      if res and arm in res["point"] else ["--"] * 3)
        out.append(rf"\{ARM_LABEL[arm]} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", "",
            "% 表 B：逐组 accuracy（阈值规则 = " + rule + "）"]
    out.append(r"\begin{tabular}{l" + "ccc" * len(ID_REGIMES) + "}")
    out.append(r"\toprule")
    out.append("& " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{lab}}}" for _k, lab in ID_REGIMES)
               + r" \\")
    out.append(" ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}"
                        for i in range(len(ID_REGIMES))))
    out.append("Method & " + " & ".join(["Ov & WG & Gap"] * len(ID_REGIMES)) + r" \\")
    out.append(r"\midrule")
    for arm in ARM_ORDER:
        cells = []
        for key, _lab in ID_REGIMES:
            res = results.get(key)
            cells += ([fmt(scalar(res, arm, f"{rule}:{k}"), 3) for k in
                       ("overall_accuracy", "worst_accuracy", "gap_accuracy")]
                      if res and arm in res["point"] else ["--"] * 3)
        out.append(rf"\{ARM_LABEL[arm]} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def build_markdown(results: dict, rule: str) -> str:
    """拼出整份 Markdown 报表。"""
    md: list[str] = []
    md.append("# 组间差异（gap）与阈值上的逐组 accuracy\n")
    md.append("> 零重训，在既有 CV-OOF 预测上按 **averaging 口径**（逐折算 → 折间平均 → 再跨组取 "
              "min/max）重算。数字源 `outputs/conditioning_ablation/group_levels_averaging.json`，"
              "由 `scripts/build_group_levels_averaging.py` 产出并对既有 "
              "`oof_results_averaging.json` 做过逐位回归自检。\n")
    md.append(f"> **主阈值规则 = {RULE_LABEL[rule]}**：每折在该折**验证集**上选一个**组无关**的"
              "全局阈值，施加到该折全部子群，指标折间平均。阈值从不接触评估数据。\n")
    md.append("> **显著性未做多重校正**：gap 与 accuracy 都是次要终点，按第 4 章 §4.3.3 不并入"
              "主终点的 Holm family，读时按 exploratory 对待。加粗 = 95% CI 不含 0。\n")

    md.append("\n## A. 组间 gap（AUC 侧，同分布）\n")
    md.append("`marg gap` = 边缘分区的组间极差（与主终点 worst-group 同分区，**本次新增**）；"
              "`canon gap` = 含 joint 交叉格的极差（既有 JSON 里的 `gap`）。\n")
    md.append(table_a(results, ID_REGIMES))

    md.append("\n## B. 逐组 accuracy（同分布）\n")
    md.append("Accuracy 与 balanced accuracy **必须成对读**：CheXpert 患病率 8.2%，"
              "全判负即得 91.8% 的 accuracy，单看它几乎只是在报各组的负样本占比。\n")
    md.append(table_b(results, ID_REGIMES, rule))

    for key, label in ID_REGIMES:
        res = results.get(key)
        if res is None:
            continue
        md.append(f"\n## {label} 明细\n")
        md.append(f"n={res['n']:,}，聚类单位数={res['n_clusters']:,}，bootstrap={res['n_boot']} 次。\n")
        md.append("\n### 逐子群 AUC\n")
        md.append(table_c(res))
        for metric in ("accuracy", "balanced_accuracy"):
            md.append(f"\n### 逐子群 {CLF_LABEL[metric]}（{RULE_LABEL[rule]}）\n")
            md.append(table_d(res, rule, metric))
        md.append("\n### 阈值规则敏感性（worst-group accuracy 与 gap）\n")
        md.append(table_e(res))
        md.append(f"\n### 逐折阈值（{RULE_LABEL[rule]}，概率尺度）\n")
        md.append("折间阈值的离散程度本身说明**不能**把五折拼起来定一个阈值："
                  "那会把逐折的校准漂移当成信号。\n")
        md.append(thresholds_note(res, rule))
        md.append("\n### 对 ERM 的差与 95% CI\n")
        md.append(table_f(res, rule))

    ood = [(k, lab) for k, lab in OOD_REGIMES if k in results]
    if ood:
        md.append("\n## 跨数据集迁移（附：口径与第 5 章 OOD 表不同）\n")
        md.append("⚠️ 这两个 regime 走的是 **cv5 averaging 口径（5 个 replicate）**，与第 5 章 OOD 表"
                  "所用的 15-replicate 方差分解口径**不是同一套**，两处数字不可直接互比。"
                  "阈值取**源域**验证集的操作点——目标域没有验证集，部署时能用的只有源域那个阈值。"
                  "该口径下无 GroupDRO 预测。\n")
        md.append(table_a(results, ood))
        md.append("\n")
        md.append(table_b(results, ood, rule))

    md.append(f"\n{AUTOGEN_MARK}\n")
    md.append("\n## 结论（人工分析节）\n")
    return "\n".join(md)


def main() -> None:
    """读 JSON，写 Markdown 与 LaTeX 片段。"""
    ap = argparse.ArgumentParser(description="把逐组水平 JSON 渲染成报表")
    ap.add_argument("--json", default=None, help="输入 JSON（缺省 outputs/conditioning_ablation/）")
    ap.add_argument("--rule", default=THRESHOLD_RULES[0], choices=list(THRESHOLD_RULES),
                    help="主阈值规则（缺省 youden_val）")
    ap.add_argument("--out-md", default=None, help="输出 Markdown（缺省 docs/…）")
    ap.add_argument("--out-tex", default=None, help="输出 LaTeX 片段（缺省 outputs/…）")
    args = ap.parse_args()

    src = (OUTPUTS_DIR / "conditioning_ablation" / "group_levels_averaging.json"
           if args.json is None else Path(args.json))
    results = json.loads(src.read_text())
    out_md = (REPO_ROOT / "docs" / "between_group_gaps_and_accuracy.md"
              if args.out_md is None else Path(args.out_md))
    out_tex = (OUTPUTS_DIR / "conditioning_ablation" / "group_tables_latex.tex"
               if args.out_tex is None else Path(args.out_tex))

    body = build_markdown(results, args.rule)
    # 保留人工分析节：只覆盖自动生成标记以上的内容
    if out_md.exists() and AUTOGEN_MARK in out_md.read_text():
        tail = out_md.read_text().split(AUTOGEN_MARK, 1)[1]
        body = body.split(AUTOGEN_MARK, 1)[0] + AUTOGEN_MARK + tail
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(body)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(latex_main_tables(results, args.rule))
    print(f"已写出 {out_md}\n已写出 {out_tex}（备用，未改动 report/）")


if __name__ == "__main__":
    main()
