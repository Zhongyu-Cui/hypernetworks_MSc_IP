"""生成报告附录 A（Additional Results）的全部 LaTeX 表。

用法：
    python scripts/build_appendix_a_tables.py [--out <path.tex>]

数据源（全部为既有产物，本脚本只读不写）：
  * ``outputs/conditioning_ablation/oof_results_averaging.json``  —— ID 与 OOD 的四终点水平与两两比较
  * ``outputs/conditioning_ablation/group_levels_averaging.json`` —— 逐子群 AUC、阈值族读数、逐折阈值
  * ``outputs/<ds>/cv5/val_logs/*.jsonl``                         —— 6 配置超参搜索的验证集读数
  * ``outputs/ood_cxr/variance_decomposition_family/``            —— OOD 逐 replicate 值
  * ``outputs/conditioning_ablation/<ds>/cv5/e1_*.json``          —— 消融的逐层 rho 与三项干预
  * ``outputs/analysis/g_fusion/fusion_2x2*.json``                —— 融合臂 2x2
  * ``outputs/analysis/named_hn_rho.json``                        —— 12 个命名臂的逐层 rho
  * ``data/splits/<ds>/cv5/fold*/test.csv``                       —— 子群样本量与患病率
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.paths import OUTPUTS_DIR as OUTPUTS, REPO_ROOT

SPLITS = REPO_ROOT / "data" / "splits"

# 报告中的库名 -> (产物 JSON 里的键, split 目录名)
DATASETS: dict[str, tuple[str, str]] = {
    "HAM10000": ("HAM10000", "ham10000"),
    "Fitzpatrick17k": ("Fitzpatrick", "fitzpatrick17k"),
    "MIMIC-CXR": ("MIMIC", "mimic_cxr_nofinding"),
    "CheXpert": ("CheXpert", "chexpert_nofinding"),
}

# 报告中的臂名 -> 产物键。HAM 的三个 HN 走 sex+age 双属性键（见第 4 章）。
ARMS: list[tuple[str, str]] = [
    (r"\ERM", "erm"),
    (r"\SWAD", "swad"),
    (r"\GroupDRO", "groupdro"),
    (r"\HyperHead", "hyperhead"),
    (r"\HyperFusion", "hyperfusion"),
    (r"\HyperAdapt", "hyperadapt"),
]
HN_KEYS = {"hyperhead", "hyperfusion", "hyperadapt"}


def arm_key(ds_key: str, key: str) -> str:
    """HAM10000 上三个超网络臂改用 sex+age 双属性的产物键。

    只有 ``oof_results_averaging.json``、``named_hn_rho.json`` 与 ``selected_configs.json``
    需要这层映射；``group_levels_averaging.json`` 已经把 sex+age 臂存在无后缀的键下。
    """
    if ds_key == "HAM10000" and key in HN_KEYS:
        return f"{key}_sexage"
    return key


def load(path: Path) -> dict:
    """读取一个 JSON 产物。"""
    return json.loads(path.read_text())


def fmt(value: float | None, digits: int = 3) -> str:
    """把一个数格式化为定宽小数，None 与 NaN 渲染为短横线。"""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "---"
    return f"{value:.{digits}f}"


def signed(value: float | None, digits: int = 4) -> str:
    """带正负号的差值格式，供 Delta 列使用。"""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "---"
    return f"${value:+.{digits}f}$"


def pval(p: float | None) -> str:
    """p 值格式：小于 0.001 时写成 ``<0.001``，否则给三位小数。"""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "---"
    if p < 1e-3:
        return r"$<$0.001"
    return f"{p:.3f}"


def ci(lo: float, hi: float, digits: int = 4) -> str:
    """把一对分位点渲染成区间。"""
    return f"$[{lo:+.{digits}f},\, {hi:+.{digits}f}]$"


def table(caption: str, label: str, spec: str, header: list[str], rows: list,
          size: str = r"\small", colsep: float | None = None,
          placement: str = "htbp", header_rule: str | None = None,
          long: bool = False) -> str:
    """把表头与行渲染成一个完整的 LaTeX 表。

    Args:
        caption    : 表题。
        label      : \\label 名。
        spec       : tabular 列型。
        header     : 表头单元格。
        rows       : 数据行，元素为单元格列表；字面量 ``"MID"`` 渲染成一条 \\midrule。
        size       : 字号命令。
        colsep     : \\tabcolsep 的 pt 值，None 表示不改。
        placement  : 浮动体位置，long=True 时忽略。
        header_rule: 表头下方额外的 \\cmidrule 串，供两层表头使用。
        long       : 行数超过一页时用 longtable，使表能跨页续排。

    Returns:
        完整的 LaTeX 片段。
    """
    head_cells = " & ".join(header) + r" \\"
    body = []
    for row in rows:
        body.append(r"\midrule" if row == "MID" else " & ".join(row) + r" \\")
    if long:
        # 字号与列距必须写在 center 造出的组里面，否则会漏到表后面的正文段落上
        parts = [r"\begin{center}", size]
        if colsep is not None:
            parts.append(f"\\setlength{{\\tabcolsep}}{{{colsep}pt}}")
        parts += [f"\\begin{{longtable}}{{{spec}}}",
                  f"\\caption{{{caption}}}\\label{{{label}}}\\\\",
                  r"\toprule", head_cells]
        if header_rule:
            parts.append(header_rule)
        parts += [r"\midrule", r"\endfirsthead",
                  r"\multicolumn{%d}{l}{\emph{\tablename~\thetable, continued}}\\" % len(header),
                  r"\toprule", head_cells]
        if header_rule:
            parts.append(header_rule)
        parts += [r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
        parts += body
        parts += [r"\end{longtable}", r"\end{center}", ""]
        return "\n".join(parts)
    parts = [f"\\begin{{table}}[{placement}]", r"\centering",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}", size]
    if colsep is not None:
        parts.append(f"\\setlength{{\\tabcolsep}}{{{colsep}pt}}")
    parts += [f"\\begin{{tabular}}{{{spec}}}", r"\toprule", head_cells]
    if header_rule:
        parts.append(header_rule)
    parts.append(r"\midrule")
    parts += body
    parts += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(parts)


# =========================================================================
# A.1 子群构成：逐子群样本量与患病率
# =========================================================================

def subgroup_frame(ds_key: str, split_dir: str) -> pd.DataFrame:
    """把一个库的五个 test 折拼回全量，并补出与产物同名的子群列。"""
    frames = [pd.read_csv(SPLITS / split_dir / "cv5" / f"fold{k}" / "test.csv")
              for k in range(5)]
    df = pd.concat(frames, ignore_index=True)
    if ds_key == "HAM10000":
        # 第 4 章的排除：最年轻的年龄档不参与评估
        df = df[df["age_group"] >= 0].copy()
        bands = {0: "20-40", 1: "40-60", 2: "60-80", 3: "80+"}
        df["sex_g"] = df["sex"].map({0: "sex:Male", 1: "sex:Female"})
        df["age_g"] = df["age_group"].map(lambda k: f"age:{bands[k]}")
        df["axes"] = list(zip(df["sex_g"], df["age_g"]))
    elif ds_key == "Fitzpatrick":
        roman = {0: "I", 1: "II", 2: "III", 3: "IV", 4: "V", 5: "VI"}
        df["skin_g"] = df["skin"].map(lambda k: f"skin:{roman[k]}")
        df["axes"] = list(zip(df["skin_g"]))
    else:
        df["sex_g"] = df["sex"].map({0: "sex:Male", 1: "sex:Female"})
        df["race_g"] = df["race"].map({0: "race:White", 1: "race:Non-White"})
        df["age_g"] = df["age"].map(lambda a: "age:>=60" if a >= 60 else "age:<60")
        df["axes"] = list(zip(df["sex_g"], df["race_g"], df["age_g"]))
    return df


def subgroup_counts(ds_key: str, split_dir: str) -> dict[str, tuple[int, float]]:
    """返回 {子群名: (样本量, 患病率)}，marginal 组与 canonical 格各一份。"""
    df = subgroup_frame(ds_key, split_dir)
    out: dict[str, tuple[int, float]] = {}
    axis_cols = [c for c in ("sex_g", "race_g", "age_g", "skin_g") if c in df]
    for col in axis_cols:                       # marginal 分区
        for name, sub in df.groupby(col):
            out[name] = (len(sub), float(sub["label"].mean()))
    if len(axis_cols) > 1:                      # canonical 分区（单轴库两者重合）
        for name, sub in df.groupby("axes"):
            out["|".join(name)] = (len(sub), float(sub["label"].mean()))
    return out


def texify_group(name: str) -> str:
    """把产物里的子群键渲染成表里可读的名字。"""
    return (name.replace("|", r" $\times$ ").replace(":", ": ")
            .replace(">=", r"$\geq$").replace("<", "$<$"))


def build_composition() -> str:
    """A.1：四个库的逐子群样本量与患病率，marginal 组与 canonical 格并列。"""
    rows: list[list[str]] = []
    for i, (ds, (ds_key, split_dir)) in enumerate(DATASETS.items()):
        if i:
            rows.append("MID")
        rows.append([r"\multicolumn{4}{l}{\emph{%s}}" % ds])
        counts = subgroup_counts(ds_key, split_dir)
        marg = [g for g in counts if "|" not in g]
        canon = [g for g in counts if "|" in g]
        for g in marg + canon:
            n, prev = counts[g]
            rows.append([texify_group(g),
                         "canonical" if "|" in g else "marginal",
                         f"{n:,}", f"{100 * prev:.1f}"])
    caption = (r"Subgroup composition of the four datasets, taken over the union of "
               r"the five held-out folds, which is the whole of each dataset. "
               r"\emph{Prev.} is the positive rate of the binary target inside the "
               r"subgroup. On Fitzpatrick17k the two partitions coincide, because "
               r"there is a single attribute axis.")
    return table(caption, "tab:app-composition", "llrr",
                 ["Subgroup", "Partition", r"$n$", r"Prev. (\%)"], rows,
                 size=r"\footnotesize", long=True)




# =========================================================================
# A.2 超参搜索：6 个配置的验证集 worst-group AUC，与每格选中的配置
# =========================================================================

GRID = ["lr3e-05_wd1e-04", "lr3e-05_wd1e-03", "lr1e-04_wd1e-04",
        "lr1e-04_wd1e-03", "lr3e-04_wd1e-04", "lr3e-04_wd1e-03"]
GRID_LABEL = {"lr3e-05_wd1e-04": r"$3\!\times\!10^{-5}$, $10^{-4}$",
              "lr3e-05_wd1e-03": r"$3\!\times\!10^{-5}$, $10^{-3}$",
              "lr1e-04_wd1e-04": r"$10^{-4}$, $10^{-4}$",
              "lr1e-04_wd1e-03": r"$10^{-4}$, $10^{-3}$",
              "lr3e-04_wd1e-04": r"$3\!\times\!10^{-4}$, $10^{-4}$",
              "lr3e-04_wd1e-03": r"$3\!\times\!10^{-4}$, $10^{-3}$"}

def build_selected() -> str:
    """A.2：每 (库, 方法) 选中的配置。

    只报选中结果而不报 6 个候选的验证值：``val_logs`` 里有四个格的日志在选择做完之后被
    下游作业重跑覆盖过，因此现在从日志重算出的候选值已不是当初做选择时读到的那一份。
    """
    rows: list[list[str]] = []
    chosen = {ds: load(OUTPUTS / SEARCH_DIR[ds_key] / "cv5" / "selected_configs.json")["config"]
              for ds, (ds_key, _) in DATASETS.items()}
    for name, key in ARMS:
        cells = [GRID_LABEL[chosen[ds][arm_key(ds_key, key)]]
                 for ds, (ds_key, _) in DATASETS.items()]
        rows.append([name] + cells)
    caption = (r"The configuration selected for each dataset and method, given as "
               r"learning rate and weight decay. Selection maximises the "
               r"fold-averaged validation marginal worst-group AUC over the six "
               r"candidates of \cref{tab:training-config}, as described in "
               r"\cref{sec:exp1-id}. \SWAD\ is not searched separately and reuses "
               r"the \ERM\ configuration, and \HASWAD\ reuses the \HyperAdapt\ one. "
               r"On HAM10000 the three hypernetworks condition on sex and age jointly.")
    header = ["Method"] + list(DATASETS)
    return table(caption, "tab:app-selected", "l" + "c" * len(DATASETS), header, rows,
                 size=r"\small", colsep=6.0)


SEARCH_DIR = {"HAM10000": "ham10000", "Fitzpatrick": "fitzpatrick",
              "MIMIC": "mimic_cxr", "CheXpert": "chexpert_cxr"}


# =========================================================================
# A.3 同分布：四个终点的完整水平，含 canonical 分区
# =========================================================================

ID_ENDPOINTS = [("overall", "Overall"), ("marginal_worst", "WG marg"),
                ("canonical_worst", "WG canon"), ("marginal_gap", "Gap marg"),
                ("gap", "Gap canon")]


def build_id_levels() -> str:
    """A.3：ID 全终点水平表，补上正文只给 marginal 的 canonical 读数。"""
    oof = load(OUTPUTS / "conditioning_ablation" / "oof_results_averaging.json")
    glv = load(OUTPUTS / "conditioning_ablation" / "group_levels_averaging.json")
    rows: list[list[str]] = []
    for i, (ds, (ds_key, _)) in enumerate(DATASETS.items()):
        if i:
            rows.append("MID")
        rows.append([r"\multicolumn{6}{l}{\emph{%s}}" % ds])
        for name, key in ARMS:
            k = arm_key(ds_key, key)
            point = oof[ds_key]["point"][k]
            scal = glv[ds_key]["point"][key]["scalars"]
            cells = []
            for field, _ in ID_ENDPOINTS:
                value = point.get(field, scal.get(field))
                cells.append(fmt(value))
            rows.append([name] + cells)
    caption = (r"In-distribution levels on both partitions, averaged over the five "
               r"folds. \emph{WG} is worst-group AUC and \emph{Gap} is the "
               r"between-group gap \eqref{eq:gap}, each read on the marginal and on "
               r"the canonical partition of \cref{sec:notation}. The marginal "
               r"columns are those of \cref{tab:id-levels}. On Fitzpatrick17k the "
               r"two partitions coincide.")
    header = ["Method"] + [h for _, h in ID_ENDPOINTS]
    return table(caption, "tab:app-id-levels", "lccccc", header, rows,
                 size=r"\footnotesize", colsep=5.0, long=True)


def holm_reject(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm 逐步向下校正：返回每个比较是否在水平 alpha 上被拒绝。

    Args:
        pvals: 一个比较族内的未校正 p 值，顺序与调用方的比较列表一致。
        alpha: 族错误率，默认 0.05。

    Returns:
        与 pvals 等长的布尔列表，True 表示该比较在校正后仍显著。
    """
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    reject = [False] * m
    # 按 p 值升序逐个比较 alpha/(m-r)，一旦不通过即停止（Holm 的 step-down 规则）
    for rank, idx in enumerate(order):
        if pvals[idx] < alpha / (m - rank):
            reject[idx] = True
        else:
            break
    return reject


# 第 4 章说明的 ID 家族：五个臂对 ERM，加三个 HN 各对 SWAD 与 GroupDRO，共 11 个比较
ID_FAMILY: list[tuple[str, str]] = [
    ("swad", "erm"), ("groupdro", "erm"), ("hyperhead", "erm"),
    ("hyperfusion", "erm"), ("hyperadapt", "erm"),
    ("hyperhead", "swad"), ("hyperfusion", "swad"), ("hyperadapt", "swad"),
    ("hyperhead", "groupdro"), ("hyperfusion", "groupdro"), ("hyperadapt", "groupdro"),
]
def pair(arm: str, ref: str) -> str:
    """渲染一个两两比较的名字，宏后补 ``\\`` 以免 LaTeX 吞掉宏名后的空格。"""
    return PRETTY[arm] + r"\ $-$ " + PRETTY[ref]


PRETTY = {"erm": r"\ERM", "swad": r"\SWAD", "groupdro": r"\GroupDRO",
          "hyperhead": r"\HyperHead", "hyperfusion": r"\HyperFusion",
          "hyperadapt": r"\HyperAdapt"}


def build_id_pairwise(field: str, label: str, tag: str, primary: bool) -> str:
    """A.4：ID 家族的全部两两比较，给 Delta、区间、未校正 p 与 Holm 校正后的判决。

    Args:
        field: 终点在产物里的键名，例如 ``marginal_worst`` 或 ``marginal_gap``。
        label: 终点在表题里的英文写法。
        tag: 表 label 的后缀，拼成 ``tab:app-id-pairwise-<tag>``。
        primary: 该终点是否为主终点，只影响表题措辞；两类终点一律各自校正。

    Returns:
        整张 longtable 的 LaTeX 源码。
    """
    oof = load(OUTPUTS / "conditioning_ablation" / "oof_results_averaging.json")
    glv = load(OUTPUTS / "conditioning_ablation" / "group_levels_averaging.json")
    rows: list[list[str]] = []
    for i, (ds, (ds_key, _)) in enumerate(DATASETS.items()):
        if i:
            rows.append("MID")
        rows.append([r"\multicolumn{5}{l}{\emph{%s}}" % ds])
        # marginal_gap 只落在 group_levels 产物里，其余终点两处都有，故按顺序回落
        found: list[tuple[tuple[str, str], dict]] = []
        for arm, ref in ID_FAMILY:
            pair_key = f"{arm_key(ds_key, arm)}_vs_{arm_key(ds_key, ref)}"
            entry = oof[ds_key]["vs"].get(pair_key)
            if entry is None or field not in entry:
                entry = glv[ds_key]["vs"].get(f"{arm}_vs_{ref}")
            if entry is None or field not in entry:
                continue
            found.append(((arm, ref), entry[field]))
        # 每个 (数据集, 终点) 自成一个 11 元比较族，Holm 在族内做
        verdicts = holm_reject([rec["p"] for _, rec in found])
        for ((arm, ref), rec), keep in zip(found, verdicts):
            rows.append([pair(arm, ref), signed(rec["delta"]),
                         ci(rec["ci"][0], rec["ci"][1]), pval(rec["p"]),
                         "yes" if keep else "no"])
    role = "primary" if primary else "secondary"
    caption = (rf"In-distribution pairwise comparisons on {label}, with 95 per cent "
               rf"paired cluster bootstrap intervals. The $p$ column is uncorrected "
               rf"and the last column gives the verdict after the Holm correction "
               rf"applied within the eleven comparisons this framework carries on "
               rf"each dataset, this being a {role} endpoint corrected within its "
               rf"own set as \cref{{sec:estimands}} requires.")
    return table(caption, f"tab:app-id-pairwise-{tag}", "lcccc",
                 ["Comparison", r"$\Delta$", "95\\% CI", r"$p$", "Sig."], rows,
                 size=r"\scriptsize", colsep=4.0, long=True)




# =========================================================================
# A.5 逐子群 AUC 水平
# =========================================================================

def build_per_group_auc() -> str:
    """A.5：每个子群在六个臂下的折平均 AUC，marginal 与 canonical 一并给出。"""
    glv = load(OUTPUTS / "conditioning_ablation" / "group_levels_averaging.json")
    rows: list[list[str]] = []
    for i, (ds, (ds_key, _)) in enumerate(DATASETS.items()):
        if i:
            rows.append("MID")
        rows.append([r"\multicolumn{7}{l}{\emph{%s}}" % ds])
        point = glv[ds_key]["point"]
        groups = list(point["erm"]["per_group_auc"].keys())
        for g in groups:
            cells = [fmt(point[key]["per_group_auc"].get(g)) for _, key in ARMS]
            rows.append([texify_group(g)] + cells)
    caption = (r"Group AUC of every subgroup under every arm, averaged over the five "
               r"folds. The worst entry of a column is the worst-group AUC reported "
               r"in \cref{tab:id-levels} for that arm, read on the marginal groups, "
               r"and the canonical rows give the same reading on the crossed cells.")
    header = ["Subgroup"] + [n for n, _ in ARMS]
    return table(caption, "tab:app-per-group", "l" + "c" * len(ARMS), header, rows,
                 size=r"\scriptsize", colsep=3.5, long=True)


# =========================================================================
# A.6 固定操作点上的完整读数
# =========================================================================

DEC_ROWS = [("overall_accuracy", "Overall accuracy"),
            ("worst_accuracy", "Worst-group accuracy"),
            ("gap_accuracy", "Accuracy gap"),
            ("overall_balanced_accuracy", "Overall balanced accuracy"),
            ("worst_balanced_accuracy", "Worst-group balanced accuracy"),
            ("gap_balanced_accuracy", "Balanced accuracy gap"),
            ("worst_sensitivity", "Worst-group sensitivity"),
            ("worst_specificity", "Worst-group specificity")]


def build_decision_levels() -> str:
    """A.6：Youden 操作点上的全部读数，补上正文未给的 overall 与两个类条件率。"""
    glv = load(OUTPUTS / "conditioning_ablation" / "group_levels_averaging.json")
    rows: list[list[str]] = []
    for i, (ds, (ds_key, _)) in enumerate(DATASETS.items()):
        if i:
            rows.append("MID")
        rows.append([r"\multicolumn{7}{l}{\emph{%s}}" % ds])
        for field, name in DEC_ROWS:
            cells = [fmt(glv[ds_key]["point"][key]["scalars"]
                         .get(f"youden_val:{field}")) for _, key in ARMS]
            rows.append([name] + cells)
    caption = (r"All readings at the fixed operating point of "
               r"\cref{sec:fairness-metrics}, averaged over the five folds on the "
               r"marginal partition. The four worst-group and gap rows for accuracy "
               r"and balanced accuracy are those of \cref{tab:id-decision}. The "
               r"overall rows and the two class-conditional rates are given here "
               r"only. A smaller value is better in the two gap rows.")
    header = ["Reading"] + [n for n, _ in ARMS]
    return table(caption, "tab:app-decision", "l" + "c" * len(ARMS), header, rows,
                 size=r"\scriptsize", colsep=3.5, long=True)


def build_thresholds() -> str:
    """A.7：每折上 Youden 规则返回的阈值，说明第 4 章的折间漂移。"""
    glv = load(OUTPUTS / "conditioning_ablation" / "group_levels_averaging.json")
    rows: list[list[str]] = []
    for ds, (ds_key, _) in DATASETS.items():
        thr = glv[ds_key]["point"]["erm"]["thresholds"]["youden_val"]
        rows.append([ds] + [f"{t:.3f}" for t in thr]
                    + [f"{max(thr) / max(min(thr), 1e-9):.1f}"])
    caption = (r"The threshold the Youden rule returns on the validation split of "
               r"each fold, under \ERM. The last column is the ratio of the largest "
               r"to the smallest, which is why the threshold is selected inside a "
               r"fold rather than once on the five folds pooled "
               r"(\cref{sec:estimands}).")
    header = ["Dataset"] + [f"Fold {k}" for k in range(5)] + ["Ratio"]
    return table(caption, "tab:app-thresholds", "lccccc" + "c", header, rows,
                 size=r"\small", colsep=5.0)


# =========================================================================
# A.8 迁移：逐 replicate 值与方差分量
# =========================================================================

OOD_FILES = {"marginal_worst": ("mw", "variance_decomposition_marginal_worst.json"),
             "overall": ("ov", "variance_decomposition_overall.json")}
DIRECTIONS = [("M2C", r"MIMIC-CXR $\rightarrow$ CheXpert"),
              ("C2M", r"CheXpert $\rightarrow$ MIMIC-CXR")]
OOD_ARMS = ["erm", "swad", "groupdro", "hyperhead", "hyperfusion", "hyperadapt"]


def ood_json(metric: str) -> dict:
    """读一个终点的 OOD 方差分解产物。"""
    sub, name = OOD_FILES[metric]
    return load(OUTPUTS / "ood_cxr" / "variance_decomposition_family" / sub / name)


def build_ood_replicates() -> str:
    """A.8：每个臂在 5 折 x 3 seed 上的 15 个 replicate 值，两个方向各一块。"""
    data = ood_json("marginal_worst")
    rows: list[list[str]] = []
    for i, (code, label) in enumerate(DIRECTIONS):
        if i:
            rows.append("MID")
        rows.append([r"\multicolumn{4}{l}{\emph{%s}}" % label])
        per_cell = data[code]["per_cell"]
        for key in OOD_ARMS:
            cells = per_cell[key]
            for fold, seeds in enumerate(cells):
                rows.append([PRETTY[key], f"Fold {fold}"]
                            + [fmt(v, 4) for v in seeds])
    caption = (r"Every replicate behind \cref{tab:ood-main}, on marginal worst-group "
               r"AUC. Each row is one source fold and the three columns are the three "
               r"training seeds run on it, so an arm contributes fifteen models per "
               r"direction and every one of them is evaluated on the entire target "
               r"dataset. The averages of these entries are the levels of "
               r"\cref{tab:ood-main}.")
    header = ["Method", "Fold", "Seed 1", "Seed 2", "Seed 3"]
    return table(caption, "tab:app-ood-replicates", "llccc", header, rows,
                 size=r"\scriptsize", colsep=5.0, long=True)


def variance_components(cells_a: list[list[float]],
                        cells_b: list[list[float]]) -> dict[str, float]:
    """按 \\eqref{eq:anova-model} 对一对臂的逐格差做两因子方差分解。"""
    d = [[a - b for a, b in zip(ra, rb)] for ra, rb in zip(cells_a, cells_b)]
    n_f, n_s = len(d), len(d[0])
    grand = sum(sum(r) for r in d) / (n_f * n_s)
    row = [sum(r) / n_s for r in d]
    col = [sum(d[f][s] for f in range(n_f)) / n_f for s in range(n_s)]
    ss_f = n_s * sum((r - grand) ** 2 for r in row)
    ss_s = n_f * sum((c - grand) ** 2 for c in col)
    ss_e = sum((d[f][s] - row[f] - col[s] + grand) ** 2
               for f in range(n_f) for s in range(n_s))
    ms_f, ms_s = ss_f / (n_f - 1), ss_s / (n_s - 1)
    ms_e = ss_e / ((n_f - 1) * (n_s - 1))
    var_f, var_s = max((ms_f - ms_e) / n_s, 0.0), max((ms_s - ms_e) / n_f, 0.0)
    return {"mean": grand, "var_fold": var_f, "var_seed": var_s, "var_resid": ms_e,
            "sigma_train": math.sqrt(var_s + ms_e)}


def build_ood_variance(metric: str, label: str, tag: str) -> str:
    """A.9：18 个对比的方差分量、噪声地板与比值，第 4 章定义了却未在正文出现。"""
    data = ood_json(metric)
    rows: list[list[str]] = []
    for i, (code, dir_label) in enumerate(DIRECTIONS):
        if i:
            rows.append("MID")
        rows.append([r"\multicolumn{7}{l}{\emph{%s}}" % dir_label])
        per_cell = data[code]["per_cell"]
        for arm in ("hyperhead", "hyperfusion", "hyperadapt"):
            for ref in ("erm", "swad", "groupdro"):
                vc = variance_components(per_cell[arm], per_cell[ref])
                ratio = abs(vc["mean"]) / vc["sigma_train"] if vc["sigma_train"] else float("nan")
                rows.append([pair(arm, ref),
                             signed(vc["mean"]),
                             fmt(math.sqrt(vc["var_fold"]), 4),
                             fmt(math.sqrt(vc["var_seed"]), 4),
                             fmt(math.sqrt(vc["var_resid"]), 4),
                             fmt(vc["sigma_train"], 4),
                             fmt(ratio, 2)])
    caption = (rf"Variance decomposition of the transfer comparisons on {label}. "
               r"The three standard deviations are the square roots of the components "
               r"of \eqref{eq:anova-model}, estimated from the fifteen paired "
               r"differences of \cref{tab:app-ood-replicates}. The floor "
               r"$\sigtrain$ of \eqref{eq:sigma-train} keeps the seed and the residual "
               r"component and leaves out the fold one. The last column is the ratio "
               r"plotted in \cref{fig:ood-ratio}, and the floor condition asks whether "
               r"it reaches one.")
    header = ["Comparison", r"$\bar{d}$", r"$\hat{\sigma}_{\text{fold}}$",
              r"$\hat{\sigma}_{\text{seed}}$", r"$\hat{\sigma}_{\varepsilon}$",
              r"$\sigtrain$", r"$|\bar{d}|/\sigtrain$"]
    return table(caption, f"tab:app-ood-variance-{tag}", "lcccccc", header, rows,
                 size=r"\scriptsize", colsep=4.0, long=True)


def read_family_readout() -> dict[tuple[str, str], list[dict]]:
    """解析 ``family_readout.txt``，取出 18 个对比的区间与 Holm p。

    这两个量在方差分解 JSON 里只对 ERM 参照存了一份，而框架实际报告的是三个参照，
    完整的一份只落在这个 readout 里，也正是 :math:`\\cref{fig:ood-ratio}` 画的那一份。
    """
    path = (OUTPUTS / "ood_cxr" / "variance_decomposition_family" / "family_readout.txt")
    blocks: dict[tuple[str, str], list[dict]] = {}
    key: tuple[str, str] | None = None
    for line in path.read_text().splitlines():
        line = line.rstrip()
        if line.startswith("==="):
            code, metric = line.strip("= ").split(" \u00b7 ")
            key = (code, metric)
            blocks[key] = []
        elif key and line.startswith(("HyperHead", "HyperFusion", "HyperAdapt")):
            parts = line.split()
            blocks[key].append({
                "arm": parts[0], "ref": parts[2], "delta": float(parts[3]),
                "ci": parts[4], "p": parts[5], "sig": parts[6],
                "sigma": float(parts[7]), "ratio": float(parts[8]),
                "floor": parts[9]})
    return blocks


def build_ood_pairwise() -> str:
    """A.10：迁移方向上 18 个对比的 Delta、区间、Holm p 与噪声地板判决。"""
    blocks = read_family_readout()
    rows: list[list[str]] = []
    first = True
    for metric, label in (("marginal_worst", "Marginal worst-group AUC"),
                          ("overall", "Overall AUC")):
        for code, dir_label in DIRECTIONS:
            if not first:
                rows.append("MID")
            first = False
            rows.append([r"\multicolumn{6}{l}{\emph{%s, %s}}" % (label, dir_label)])
            for rec in blocks[(code, metric)]:
                lo, hi = rec["ci"].strip("[]").split(",")
                rows.append([
                    pair(rec["arm"].lower(), rec["ref"].lower()),
                    "$%+.4f$" % rec["delta"],
                    "$[%s,\\, %s]$" % (lo, hi),
                    rec["p"] if "e" not in rec["p"] else r"$<$0.001",
                    "yes" if rec["sig"] == "yes" else "no",
                    "yes" if rec["floor"] == "yes" else "no"])
    caption = (r"Every comparison the transfer framework carries, on both endpoints "
               r"and in both directions. The interval is the 95 per cent paired "
               r"cluster bootstrap over the patients of the target dataset and the "
               r"$p$-value is Holm-corrected within the nine comparisons of one "
               r"direction. \emph{Floor} records whether the effect also clears "
               r"$\sigtrain$, taken from \cref{tab:app-ood-variance-wg} for the "
               r"worst-group blocks and from \cref{tab:app-ood-variance-ov} for "
               r"the Overall ones. \Cref{fig:ood-ratio} plots the first of those "
               r"two ratios. An effect is accepted "
               r"only when both columns read yes in both directions.")
    return table(caption, "tab:app-ood-pairwise", "lccccc",
                 ["Comparison", r"$\Delta$", "95\\% CI", r"$p$", "Sig.", "Floor"],
                 rows, size=r"\scriptsize", colsep=4.0, long=True)


# =========================================================================
# A.11 消融的补充读数
# =========================================================================

CELLS = [("condnet_erm", r"\ERM"), ("condnet_head", r"\CHead"),
         ("condnet_deep", r"\CDeep"), ("condnet_full", r"\CFull")]
ABL = OUTPUTS / "conditioning_ablation"


def build_ablation_rho() -> str:
    """A.11：消融各 cell 的逐层 rho 与 rho_between，正文只给了按 stage 平均的版本。"""
    rows: list[list[str]] = []
    for i, (ds, sub) in enumerate((("HAM10000", "ham10000"), ("MIMIC-CXR", "mimic_cxr"))):
        if i:
            rows.append("MID")
        rows.append([r"\multicolumn{4}{l}{\emph{%s}}" % ds])
        name = "e1_rho_between.json" if sub == "ham10000" else "e2_rho_between.json"
        data = load(ABL / sub / "cv5" / name)
        for cell_key, cell_name in CELLS:
            layers = [(k.split("|", 1)[1], v) for k, v in data.items()
                      if k.startswith(cell_key + "|")]
            for j, (layer, rec) in enumerate(layers):
                rows.append([cell_name if j == 0 else "",
                             layer.replace("_", r"\_"),
                             fmt(rec.get("rho"), 4), fmt(rec.get("rho_between"), 4)])
    caption = (r"Per-layer conditioning strength of every ablation cell, averaged "
               r"over folds and seeds. Both quantities of \cref{sec:rho-def} are "
               r"given, since $\rhotot$ records how far the layer moved and only "
               r"$\rhobet$ shows that the movement depends on the attribute. "
               r"\Cref{fig:rho} averages these entries within a stage. The ablation "
               r"conditions on age alone, which is why its \CFull\ profile is not "
               r"directly comparable with the \HyperAdapt\ profile of "
               r"\cref{tab:app-named-rho} on HAM10000.")
    return table(caption, "tab:app-ablation-rho", "llcc",
                 ["Cell", "Layer", r"$\rhotot$", r"$\rhobet$"], rows,
                 size=r"\scriptsize", colsep=5.0, long=True)


def build_ablation_interventions() -> str:
    """A.12：三项干预的完整读数，正文只给了其中一部分。"""
    base = load(ABL / "ham10000" / "cv5" / "e1_ham_results_averaging.json")["point_estimates"]
    alt = load(ABL / "ham10000" / "cv5" / "e1_ham_results_averaging_worstcase.json")["point_estimates"]
    nofc = load(ABL / "ham10000" / "cv5" / "e3_nofc_results.json")
    knock = load(ABL / "ham10000" / "cv5" / "e2_knockout_permutation_overall.json")["results"]

    rows: list[list[str]] = []
    rows.append([r"\multicolumn{5}{l}{\emph{Checkpoint at best validation Overall AUC}}"])
    for key in ("ERM", "C-Head", "C-Deep", "C-Full"):
        rec = base[key]
        rows.append([key.replace("C-", r"\C"), fmt(rec["overall"], 4),
                     fmt(rec["marginal_worst"], 4), fmt(rec["gap"], 4), ""])
    rows.append("MID")
    rows.append([r"\multicolumn{5}{l}{\emph{Checkpoint at best validation worst-group AUC}}"])
    for key in ("ERM", "C-Head", "C-Deep", "C-Full"):
        rec = alt.get(key)
        if rec is None:
            continue
        rows.append([key.replace("C-", r"\C"), fmt(rec["overall"], 4),
                     fmt(rec["marginal_worst"], 4), fmt(rec["gap"], 4), ""])
    rows.append("MID")
    rows.append([r"\multicolumn{5}{l}{\emph{Retrained with the head pathway disabled}}"])
    for key, label in (("C-Deep-noFC", r"\CDeep"), ("C-Full-noFC", r"\CFull")):
        rec = nofc["behavior"][key]
        cell = "condnet_deep_nofc" if "Deep" in key else "condnet_full_nofc"
        rho = nofc["final_rho"][cell]
        rows.append([label, fmt(rec["overall"], 4), fmt(rec["worst"], 4), "---",
                     fmt(rho["conv_rho_between_mean"], 4)])
    caption = (r"The ablation cells under the two checkpoint rules and after "
               r"retraining without the head pathway, on HAM10000 in distribution. "
               r"The last column is the convolutional $\rhobet$ the retrained cells "
               r"reach, which is what the third intervention of "
               r"\cref{sec:exp3-ablation} is read on together with the endpoints. "
               r"The gap is on the canonical partition.")
    return table(caption, "tab:app-ablation-interventions", "lcccc",
                 ["Cell", "Overall", "Worst-group", "Gap", r"conv $\rhobet$"], rows,
                 size=r"\small", colsep=5.0)


def build_knockout_overall() -> str:
    """A.13：通路 knockout 的 Overall 口径，正文 tab:knockout 只给 worst-group。"""
    knock = load(ABL / "ham10000" / "cv5" / "e2_knockout_permutation_overall.json")["results"]
    names = {"full": "both", "conv_off": "head only", "fc_off": "convolution only"}
    rows: list[list[str]] = []
    for cell in ("C-Deep", "C-Full"):
        for j, (suffix, label) in enumerate(names.items()):
            rec = knock.get(f"{cell}|{suffix}")
            if rec is None:
                continue
            sigma = (rec["delta_overall"] / rec["overall_perm_sd"]
                     if rec["overall_perm_sd"] else float("nan"))
            rows.append([cell.replace("C-", r"\C") if j == 0 else "", label,
                         signed(rec["delta_overall"]),
                         fmt(rec["overall_perm_sd"], 4), f"{sigma:.1f}$\\sigma$"])
        if cell == "C-Deep":
            rows.append("MID")
    caption = (r"Pathway knockout read on Overall AUC, which is the counterpart of "
               r"\cref{tab:knockout}. $\Delta$ is the difference between the real "
               r"attribute and one permuted across lesions, and the spread is the "
               r"standard deviation over the permutations.")
    return table(caption, "tab:app-knockout-overall", "llccc",
                 ["Cell", "Pathways active", r"$\Delta$ Overall", "Spread", "Distance"],
                 rows, size=r"\small", colsep=6.0)


# =========================================================================
# A.14 融合臂与机制的补充
# =========================================================================

FUSION_DS = [("HAM10000", "fusion_2x2.json", "HAM10000"),
             ("Fitzpatrick17k", "fusion_2x2.json", "Fitzpatrick"),
             ("MIMIC-CXR", "fusion_2x2_cxr.json", "MIMIC"),
             ("CheXpert", "fusion_2x2_cxr.json", "CheXpert")]


def build_fusion() -> str:
    """A.14：2x2 设计的四个 cell 与配置匹配下的交互项，两个终点。"""
    rows: list[list[str]] = []
    cache: dict[str, dict] = {}
    for ds, fname, key in FUSION_DS:
        if fname not in cache:
            cache[fname] = load(OUTPUTS / "analysis" / "g_fusion" / fname)
        block = cache[fname][key]
        matched = block["config_matched"].get("by_endpoint")
        for endpoint, field in (("Marginal worst-group AUC", "marginal worst"),
                                ("Overall AUC", "Overall")):
            cells = [fmt(block["point"][a][field], 4)
                     for a in ("erm", "swad", "hyperadapt", "hyperadapt_swad")]
            if matched is None:                 # 四臂同 config 时无需匹配
                inc = ["---", "---", "---"]
            else:
                rec = matched[endpoint.split()[0] if field == "Overall" else field]
                inc = [signed(rec["erm_side_increment"]),
                       signed(rec["hn_side_increment"]),
                       signed(rec["delta_int_matched"])]
            rows.append([ds if endpoint.startswith("Marginal") else "",
                         endpoint] + cells + inc)
    caption = (r"The $2\times2$ design on both endpoints. The four level columns take "
               r"each arm at its own selected configuration and the three increment "
               r"columns take all four cells at one configuration, which is the "
               r"comparison $\dint$ of \eqref{eq:delta-int} is read on. On "
               r"Fitzpatrick17k all four arms already share a configuration, so no "
               r"matched increment is reported. The worst-group rows are those of "
               r"\cref{tab:interaction}.")
    header = ["Dataset", "Endpoint", r"\ERM", r"\SWAD", r"\HyperAdapt", "fused",
              "on \\ERM", "on \\HyperAdapt", r"$\dint$"]
    return table(caption, "tab:app-fusion", "ll" + "c" * 7, header, rows,
                 size=r"\scriptsize", colsep=3.0)


def build_named_rho() -> str:
    """A.15：三个命名超网络在四个库上的逐层 rho，正文 tab:named-rho 只给汇总。"""
    data = load(OUTPUTS / "analysis" / "named_hn_rho.json")
    layers = list(data["MIMIC-CXR"]["hyperadapt"]["per_layer"].keys())
    rows: list[list[str]] = []
    for layer in layers:
        cells = []
        for ds, (ds_key, _) in DATASETS.items():
            key = "hyperadapt_sexage" if ds_key == "HAM10000" else "hyperadapt"
            rec = data[ds]["per_layer"].get(layer) if False else \
                data[ds][key]["per_layer"].get(layer)
            cells += [fmt(rec.get("rho"), 4), fmt(rec.get("rho_between"), 4)] \
                if rec else ["---", "---"]
        rows.append([layer.replace("_", r"\_")] + cells)
    caption = (r"Per-layer conditioning strength of \HyperAdapt\ on all four "
               r"datasets, averaged over the five folds. \Cref{tab:named-rho} gives "
               r"the convolutional range and the head value of these profiles. The "
               r"head carries between roughly five and one hundred times more "
               r"attribute-dependent offset than any convolution, the ratio "
               r"depending on the dataset.")
    header = ["Layer"] + ["\\multicolumn{2}{c}{%s}" % d for d in DATASETS]
    rule = "".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i) for i in range(4))
    sub = [""] + [r"$\rhotot$", r"$\rhobet$"] * 4
    return table(caption, "tab:app-named-rho", "l" + "cc" * 4, header,
                 [sub] + rows, size=r"\scriptsize", colsep=3.0, header_rule=rule,
                 long=True)


def main() -> None:
    """按附录 A 的小节顺序生成全部表，每张表一个文件，供章节正文就地 \\input。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "report" / "sections" / "appendix_a"))
    ap.add_argument("--only", default=None, help="只生成一张表，用于逐张核对")
    args = ap.parse_args()

    builders: dict[str, object] = {
        "composition": build_composition,
        "selected": build_selected,
        "id_levels": build_id_levels,
        "id_pairwise_wg": lambda: build_id_pairwise(
            "marginal_worst", "marginal worst-group AUC", "wg", True),
        "id_pairwise_ov": lambda: build_id_pairwise(
            "overall", "Overall AUC", "ov", False),
        "id_pairwise_canon": lambda: build_id_pairwise(
            "canonical_worst", "canonical worst-group AUC", "canon", False),
        "id_pairwise_gap": lambda: build_id_pairwise(
            "marginal_gap", "the marginal between-group gap", "gap", False),
        "per_group_auc": build_per_group_auc,
        "decision_levels": build_decision_levels,
        "thresholds": build_thresholds,
        "ood_replicates": build_ood_replicates,
        "ood_variance_wg": lambda: build_ood_variance(
            "marginal_worst", "marginal worst-group AUC", "wg"),
        "ood_variance_ov": lambda: build_ood_variance(
            "overall", "Overall AUC", "ov"),
        "ood_pairwise": build_ood_pairwise,
        "ablation_rho": build_ablation_rho,
        "ablation_interventions": build_ablation_interventions,
        "knockout_overall": build_knockout_overall,
        "fusion": build_fusion,
        "named_rho": build_named_rho,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [args.only] if args.only else list(builders)
    banner = "% 由 scripts/build_appendix_a_tables.py 生成，请勿手改。\n"
    for name in names:
        (out_dir / f"{name}.tex").write_text(banner + builders[name]())
    print(f"wrote {len(names)} tables to {out_dir}")


if __name__ == "__main__":
    main()
