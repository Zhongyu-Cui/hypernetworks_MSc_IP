"""
跨数据集 / 跨架构 worst-group 公平性汇总表（回答核心论点）
==========================================================
论点：**敏感属性条件化超网络（HN）能否跨架构与数据集稳定改善医学图像分类的 worst-group
fairness？** 本脚本在**已落盘的 CV-OOF 预测**上（零重训）一次性算出四份支撑数据，全部相对
ERM 基线：

  D1  每数据集，各 HN 相对 ERM，在**各敏感边缘子群**上 AUC / TPR / FPR / specificity /
      sensitivity / PPV / NPV 的差（阈值相关指标在「各方法各自校准到整体 FPR=0.2」的可比操作点上）。
  D2  每数据集，各方法（ERM + 3 HN + SWAD/ROC）的 **worst-group 组别**（marginal 与 canonical 两口径）。
  D3  每数据集，各 HN 在 **5 折 OOF 每一折**上 worst-group AUC 相对 ERM 的差值（考察逐折稳定性）。
  D4  每数据集，各 HN 的 (best-group AUC − worst-group AUC) gap 相对 ERM 的差，**恒等分解**为
      Δgap = Δbest − Δworst（best-group AUC 差 减 worst-group AUC 差；判「抬起弱组」还是「压低强组」）。

**口径（2026-07-19 已由 pooled-OOF 改为 averaging，与 `docs/oof_regime_results.md` /
`scripts/build_oof_results_averaging.py` 权威口径一致）**：
  - 评估用 **averaging**：每折各自算子群指标，再折间平均（`mean_f metric(g,f)`）。**永不跨折
    池化 raw logit**——因各折是不同模型、早停轮数异质使 logit 尺度漂移，池化会对排序指标注入
    系统性负偏（Parker 2007 / Airola 2010；本项目实测 HN 受害远重于 ERM，曾制造高置信假阳性）。
    zero 重训——同一批逐折预测，仅改聚合方式。D3 逐折算（本就是 averaging 家族，对该伪影免疫）。
  - worst/best 用「**先折间平均后 min/max**」：`min_g[mean_f AUC(g,f)]`（Jensen：先 min 会放大向下偏，
    且「模型在哪组最弱」应是模型属性、不该每折换答案）。某折某格只剩一类 ⇒ 该折该指标无定义、跳过用余折平均。
  - 子群集合复用 `subgroup_auc.subgroup_masks`（单一事实来源）。**headline 用 marginal 边缘子群**
    （joint 小格在阈值/单折下退化成噪声）；D2 额外给 canonical（含 joint）worst 组别以透明核查。
  - AUC 阈值无关；TPR/FPR/spec/sens/PPV/NPV 需阈值 → **每折各方法各自** `threshold_at_overall_fpr(·,0.2)`
    校准操作点、算子群指标、再折间平均（整体 FPR 对齐 0.2，隔离「整体操作点差异」，只留子群差异）。
    注：TPR≡sensitivity、specificity≡1−FPR（按要求全列出，含冗余项）。

用法：
  python scripts/build_worstgroup_crossdataset_tables.py                 # 全 5 数据集 → md + csv
  python scripts/build_worstgroup_crossdataset_tables.py --dataset ham10000
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.training.harness.predictions import load_predictions
from src.training.harness.subgroup_auc import subgroup_auc_vector
from src.utils.operating_point_fairness import (
    to_prob, subgroup_classification_metrics, threshold_at_overall_fpr,
)
from scripts.cv_oof_report import (
    OUTPUTS_ROOT, FOLD_SEEDS, METHOD_ORDER, METHOD_LABEL, HN,
    SCORE_IS_PROB, DATASET_SPEC,
)

# 目标 FPR 操作点（各方法各自校准到此整体 FPR，得到跨方法可比的阈值相关指标）
TARGET_FPR = 0.2
# 阈值相关子群指标纳入门限（该类样本 < 此值则该指标记 None，避免小格退化率）
MIN_CLASS_N = 10
# 输出根：CSV 落集群 outputs（非版本控制），md 汇总另写 docs/
CSV_ROOT = OUTPUTS_ROOT / "analysis" / "crossds_worstgroup"

DATASETS_DEFAULT = ("ham10000", "papila", "fitzpatrick", "chexpert", "mimic")


# ============================================================
# 数据加载：pooled-OOF（D1/D2/D4）与逐折（D3）
# ============================================================
def _pred_dir(ds: str) -> Path:
    """该数据集 CV 预测目录（复用 cv_oof_report 的 DATASET_SPEC 相对路径）。"""
    rel_pred, _ds_key, _f = DATASET_SPEC[ds]
    return OUTPUTS_ROOT / rel_pred


def _load_cfg(ds: str) -> dict[str, str]:
    """读该数据集 S1 选定配置 selected_configs.json → {method: config_tag}。"""
    # papila 的 selected_configs.json 在 outputs/papila/ 下；其余在 <ds>/cv5/ 下。
    rel_pred, _ds_key, _f = DATASET_SPEC[ds]
    sel_path = _pred_dir(ds).parent / "selected_configs.json"
    return json.loads(sel_path.read_text())["config"]


def available_methods(ds: str) -> tuple[list[str], dict[str, str], int]:
    """
    该数据集实际有预测可载入的方法列表（按 METHOD_ORDER）+ {method: config_tag} + pooled-OOF 总 n。

    n 仅供打印参考（averaging 与 pooling 用满同一批样本，n 相同）。
    """
    pred_dir = _pred_dir(ds)
    cfg = _load_cfg(ds)
    methods: list[str] = []
    tags: dict[str, str] = {}
    for method in METHOD_ORDER:
        tag = cfg.get(method)
        if tag is None:
            continue
        path0 = pred_dir / f"{method}_{tag}_seed{FOLD_SEEDS[0]}_overall.npz"
        if not path0.exists():
            print(f"  [跳过] {ds}/{method} 缺预测：{path0.name}")
            continue
        methods.append(method)
        tags[method] = tag
    assert methods, f"{ds} 无任何方法预测可载入"
    n = sum(len(load_fold(ds, methods[0], tags[methods[0]], seed)[0]) for seed in FOLD_SEEDS)
    return methods, tags, n


def load_fold(ds: str, method: str, tag: str, seed: int
              ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """载入某方法单折 test 预测（应用该数据集 OOF 过滤器，如 HAM 排 age=-1）。"""
    _rel, _ds_key, ds_filter = DATASET_SPEC[ds]
    path = _pred_dir(ds) / f"{method}_{tag}_seed{seed}_overall.npz"
    y, s, a = load_predictions(path)
    mask = ds_filter(a) if ds_filter is not None else np.ones(len(y), dtype=bool)
    y = np.asarray(y)[mask]; s = np.asarray(s)[mask]
    a = {k: np.asarray(v)[mask] for k, v in a.items()}
    return y, s, a


# ============================================================
# 子群指标提取（marginal 边缘子群，键不含 '|'）
# ============================================================
def marginal_auc(ds_key: str, y: np.ndarray, s: np.ndarray, attrs: dict[str, np.ndarray]
                 ) -> "dict[str, float]":
    """各**边缘**子群 AUC（阈值无关）；退化组（AUC=None）剔除。返回 {子群键: auc}。"""
    vec = subgroup_auc_vector(ds_key, y, s, **attrs)
    return {k: a for k, (a, _n) in vec.items() if a is not None and "|" not in k}


def canonical_auc(ds_key: str, y: np.ndarray, s: np.ndarray, attrs: dict[str, np.ndarray]
                  ) -> "dict[str, float]":
    """各子群 AUC（含 joint 交叉，canonical 口径）；退化组剔除。"""
    vec = subgroup_auc_vector(ds_key, y, s, **attrs)
    return {k: a for k, (a, _n) in vec.items() if a is not None}


def _argmin(d: "dict[str, float]") -> "tuple[str, float]":
    """字典 argmin（返回 (键, 值)）；空字典返回 ('N/A', nan)。"""
    if not d:
        return "N/A", float("nan")
    k = min(d, key=d.get)
    return k, d[k]


def _argmax(d: "dict[str, float]") -> "tuple[str, float]":
    """字典 argmax。"""
    if not d:
        return "N/A", float("nan")
    k = max(d, key=d.get)
    return k, d[k]


def averaged_group_auc(ds: str, ds_key: str, method: str, tag: str, *, canonical: bool
                       ) -> "dict[str, float]":
    """
    averaging 口径的各子群 AUC：**逐折算 AUC → 折间平均**（跳过无定义折）。

    Args:
        canonical: True 含 joint 交叉子群；False 仅 marginal 边缘子群（键不含 '|'）。

    Returns:
        {子群键: 折均 AUC}（至少一折有定义的子群才入表）。
    """
    per: dict[str, list[float]] = {}
    for seed in FOLD_SEEDS:
        y, s, a = load_fold(ds, method, tag, seed)
        vec = subgroup_auc_vector(ds_key, y, s, **a)
        for key, (auc, _n) in vec.items():
            if auc is None:
                continue
            if not canonical and "|" in key:
                continue
            per.setdefault(key, []).append(auc)
    return {g: float(np.mean(v)) for g, v in per.items() if v}


def averaged_clf(ds: str, ds_key: str, method: str, tag: str, score_is_prob: bool
                 ) -> "dict[str, dict[str, float | None]]":
    """
    averaging 口径各**边缘**子群 7 指标：**每折各自校准 FPR=0.2 阈值 → 算子群指标 → 折间平均**。

    AUC 折均另由 averaged_group_auc 取（阈值无关）；tpr≡sensitivity、fpr≡1−specificity（冗余列出）；
    某折某子群该指标无定义（分母 0）则跳过该折、用余折平均。

    Returns:
        {子群键: {auc, tpr, fpr, specificity, sensitivity, ppv, npv}}。
    """
    acc: dict[str, dict[str, list[float]]] = {}
    for seed in FOLD_SEEDS:
        y, s, a = load_fold(ds, method, tag, seed)
        prob = to_prob(s, score_is_prob)
        thr = threshold_at_overall_fpr(y, prob, TARGET_FPR)   # 逐折自校准操作点
        clf = subgroup_classification_metrics(ds_key, y, prob, a, thr, marginal_only=True)
        for key, m in clf.items():
            d = acc.setdefault(key, {})
            for name, val in (("sensitivity", m.sensitivity), ("specificity", m.specificity),
                              ("ppv", m.ppv), ("npv", m.npv)):
                if val is not None:
                    d.setdefault(name, []).append(val)
    aucs = averaged_group_auc(ds, ds_key, method, tag, canonical=False)
    out: dict[str, dict[str, float | None]] = {}
    for key, d in acc.items():
        sens = float(np.mean(d["sensitivity"])) if d.get("sensitivity") else None
        spec = float(np.mean(d["specificity"])) if d.get("specificity") else None
        out[key] = {
            "auc": aucs.get(key),
            "sensitivity": sens,
            "tpr": sens,                                       # TPR ≡ sensitivity
            "specificity": spec,
            "fpr": (1.0 - spec) if spec is not None else None, # FPR ≡ 1 − specificity
            "ppv": float(np.mean(d["ppv"])) if d.get("ppv") else None,
            "npv": float(np.mean(d["npv"])) if d.get("npv") else None,
        }
    return out


# ============================================================
# D1：各 HN vs ERM 逐子群 7 指标差
# ============================================================
METRIC7 = ("auc", "tpr", "fpr", "specificity", "sensitivity", "ppv", "npv")


def compute_d1(ds: str, ds_key, tags, methods) -> list[dict]:
    """D1（averaging）：各 HN 相对 ERM，在各边缘子群上 7 指标折均的差。返回长表行列表。"""
    erm_clf = averaged_clf(ds, ds_key, "erm", tags["erm"], SCORE_IS_PROB["erm"])
    rows: list[dict] = []
    for hn in [m for m in HN if m in methods]:
        hn_clf = averaged_clf(ds, ds_key, hn, tags[hn], SCORE_IS_PROB[hn])
        for grp in erm_clf:
            for met in METRIC7:
                e = erm_clf[grp][met]
                h = hn_clf.get(grp, {}).get(met)
                diff = (h - e) if (e is not None and h is not None) else None
                rows.append({
                    "dataset": ds, "subgroup": grp, "hn": METHOD_LABEL[hn], "metric": met,
                    "erm": e, "hn_val": h, "diff": diff,
                })
    return rows


# ============================================================
# D2：worst-group 组别（marginal + canonical）
# ============================================================
def compute_d2(ds, ds_key, tags, methods) -> list[dict]:
    """D2（averaging）：各方法折均 AUC 上的 marginal / canonical worst-group 组别与 AUC。"""
    rows = []
    for m in methods:
        mg, ma = _argmin(averaged_group_auc(ds, ds_key, m, tags[m], canonical=False))
        cg, ca = _argmin(averaged_group_auc(ds, ds_key, m, tags[m], canonical=True))
        rows.append({
            "dataset": ds, "method": METHOD_LABEL[m],
            "marg_worst_group": mg, "marg_worst_auc": ma,
            "canon_worst_group": cg, "canon_worst_auc": ca,
        })
    return rows


# ============================================================
# D3：逐折 worst-group AUC 差（HN − ERM），marginal 口径
# ============================================================
def compute_d3(ds, ds_key, methods) -> list[dict]:
    """D3：各 HN 在 5 折每折上 marginal worst-group AUC 相对 ERM 的差值。"""
    cfg = _load_cfg(ds)
    # 预算 ERM 各折 worst（各折用 ERM 自身 argmin worst 子群）
    erm_worst = []
    for seed in FOLD_SEEDS:
        y, s, a = load_fold(ds, "erm", cfg["erm"], seed)
        _g, w = _argmin(marginal_auc(ds_key, y, s, a))
        erm_worst.append(w)
    rows = []
    for hn in [m for m in HN if m in methods]:
        for k, seed in enumerate(FOLD_SEEDS):
            y, s, a = load_fold(ds, hn, cfg[hn], seed)
            _g, w = _argmin(marginal_auc(ds_key, y, s, a))
            rows.append({
                "dataset": ds, "hn": METHOD_LABEL[hn], "fold": k,
                "worst_erm": erm_worst[k], "worst_hn": w, "diff": w - erm_worst[k],
            })
    return rows


# ============================================================
# D4：gap 分解  Δgap = Δbest − Δworst（marginal，pooled-OOF）
# ============================================================
def compute_d4(ds, ds_key, tags, methods) -> list[dict]:
    """D4（averaging）：各 HN 的 best−worst gap 相对 ERM 的差，恒等分解为 Δbest − Δworst。"""
    erm_aucs = averaged_group_auc(ds, ds_key, "erm", tags["erm"], canonical=False)
    _bg, best_erm = _argmax(erm_aucs)
    _wg, worst_erm = _argmin(erm_aucs)
    gap_erm = best_erm - worst_erm
    rows = []
    for hn in [m for m in HN if m in methods]:
        aucs = averaged_group_auc(ds, ds_key, hn, tags[hn], canonical=False)
        _bg, best_hn = _argmax(aucs)
        _wg, worst_hn = _argmin(aucs)
        gap_hn = best_hn - worst_hn
        d_best = best_hn - best_erm
        d_worst = worst_hn - worst_erm
        d_gap = gap_hn - gap_erm
        rows.append({
            "dataset": ds, "hn": METHOD_LABEL[hn],
            "best_erm": best_erm, "worst_erm": worst_erm, "gap_erm": gap_erm,
            "best_hn": best_hn, "worst_hn": worst_hn, "gap_hn": gap_hn,
            "d_best": d_best, "d_worst": d_worst, "d_gap": d_gap,
        })
    return rows


# ============================================================
# 输出：CSV + Markdown
# ============================================================
def _f(v, nd=4) -> str:
    """数值格式化（None/nan → 'N/A'）。"""
    if v is None:
        return "N/A"
    if isinstance(v, float) and np.isnan(v):
        return "N/A"
    return f"{v:.{nd}f}"


def _sign(v, nd=4) -> str:
    """带符号格式化（差值列用）。"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:+.{nd}f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    """把行列表写 CSV（列取首行键序）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


# D1 展示列顺序（对齐需求：AUC / TPR / FPR / specificity / sensitivity / PPV / NPV）
D1_COLS = (("auc", "ΔAUC"), ("tpr", "ΔTPR"), ("fpr", "ΔFPR"),
           ("specificity", "Δspec"), ("sensitivity", "Δsens"),
           ("ppv", "ΔPPV"), ("npv", "ΔNPV"))


def md_d1(md: list[str], ds: str, rows: list[dict]) -> None:
    """D1 markdown：每 (数据集, HN) 一张表，行=边缘子群，列=7 指标的差（HN − ERM）。"""
    groups = list(dict.fromkeys(r["subgroup"] for r in rows))
    hns = list(dict.fromkeys(r["hn"] for r in rows))
    idx = {(r["subgroup"], r["hn"], r["metric"]): r["diff"] for r in rows}
    for hn in hns:
        md.append(f"\n#### {ds} · {hn} − ERM，各边缘子群 7 指标差\n")
        md.append("| 子群 | " + " | ".join(lbl for _m, lbl in D1_COLS) + " |")
        md.append("|------|" + "------|" * len(D1_COLS))
        for g in groups:
            # AUC 阈值无关（越大越好）；TPR/sens/spec/PPV/NPV 越大越好；FPR 越小越好
            cells = [_sign(idx.get((g, hn, m)), 4) for m, _lbl in D1_COLS]
            md.append(f"| {g} | " + " | ".join(cells) + " |")


def md_d2(md: list[str], rows: list[dict]) -> None:
    """D2 markdown：worst-group 组别表（全数据集合一）。"""
    md.append("\n### D2 · 各方法 worst-group 组别（averaging，折均 AUC 上取 argmin）\n")
    md.append("| 数据集 | 方法 | marginal worst 组 | AUC | canonical worst 组 | AUC |")
    md.append("|--------|------|-------------------|-----|--------------------|-----|")
    for r in rows:
        md.append(f"| {r['dataset']} | {r['method']} | {r['marg_worst_group']} | "
                  f"{_f(r['marg_worst_auc'])} | {r['canon_worst_group']} | {_f(r['canon_worst_auc'])} |")


def md_d3(md: list[str], rows: list[dict]) -> None:
    """D3 markdown：逐折 worst-group ΔAUC 表（5 折 + 均值）。"""
    md.append("\n### D3 · 逐折 worst-group AUC 差（HN − ERM，marginal，每折各自 argmin worst）\n")
    md.append("| 数据集 | HN | fold0 | fold1 | fold2 | fold3 | fold4 | 均值 |")
    md.append("|--------|----|-------|-------|-------|-------|-------|------|")
    key = {}
    for r in rows:
        key.setdefault((r["dataset"], r["hn"]), {})[r["fold"]] = r["diff"]
    for (ds, hn), folds in key.items():
        vals = [folds.get(k) for k in range(len(FOLD_SEEDS))]
        mean = float(np.mean([v for v in vals if v is not None]))
        md.append(f"| {ds} | {hn} | " + " | ".join(_sign(v, 3) for v in vals) +
                  f" | {_sign(mean, 3)} |")


def md_d4(md: list[str], rows: list[dict]) -> None:
    """D4 markdown：gap 分解表。"""
    md.append("\n### D4 · best−worst gap 相对 ERM 的差及分解 Δgap = Δbest − Δworst（marginal，averaging）\n")
    md.append("| 数据集 | HN | gap_ERM | gap_HN | Δgap | Δbest | Δworst | 主导 |")
    md.append("|--------|----|---------|--------|------|-------|--------|------|")
    for r in rows:
        # 判读：Δgap<0=gap 收窄（更公平）；由 Δworst>0(抬弱组) 还是 Δbest<0(压强组) 主导
        if abs(r["d_best"]) >= abs(r["d_worst"]):
            drive = "best 侧" + ("↓压低" if r["d_best"] < 0 else "↑抬高")
        else:
            drive = "worst 侧" + ("↑抬高" if r["d_worst"] > 0 else "↓压低")
        md.append(f"| {r['dataset']} | {r['hn']} | {_f(r['gap_erm'])} | {_f(r['gap_hn'])} | "
                  f"{_sign(r['d_gap'])} | {_sign(r['d_best'])} | {_sign(r['d_worst'])} | {drive} |")


def main() -> None:
    parser = argparse.ArgumentParser(description="跨数据集/架构 worst-group 公平汇总（4 份支撑数据）")
    parser.add_argument("--dataset", nargs="*", default=None,
                        help=f"子集（默认全部 {DATASETS_DEFAULT}）")
    parser.add_argument("--csv-dir", type=str, default=str(CSV_ROOT))
    args = parser.parse_args()
    datasets = tuple(args.dataset) if args.dataset else DATASETS_DEFAULT
    csv_dir = Path(args.csv_dir)

    all_d1, all_d2, all_d3, all_d4 = [], [], [], []
    md_d1_blocks: list[str] = ["\n### D1 · 各 HN vs ERM 逐子群 7 指标差（每 (数据集,HN) 一张表）\n"
                               "> 阈值相关指标（TPR/FPR/spec/sens/PPV/NPV）取各方法各自校准到整体 "
                               f"FPR={TARGET_FPR} 的可比操作点；AUC 阈值无关；TPR≡sensitivity、FPR≡1−specificity。\n"
                               "> 除 FPR「越小越好」外各列均「越大越好」；正号=HN 优于 ERM。数值全量另见 "
                               "`d1_per_group_metric_diffs.csv`。"]

    for ds in datasets:
        print(f"# === {ds} ===")
        _rel, ds_key, _f = DATASET_SPEC[ds]
        methods, tags, n = available_methods(ds)
        print(f"  averaging OOF n={n:,}  方法={[METHOD_LABEL[m] for m in methods]}")

        d1 = compute_d1(ds, ds_key, tags, methods);  all_d1 += d1
        d2 = compute_d2(ds, ds_key, tags, methods);  all_d2 += d2
        d3 = compute_d3(ds, ds_key, methods);        all_d3 += d3
        d4 = compute_d4(ds, ds_key, tags, methods);  all_d4 += d4
        md_d1(md_d1_blocks, ds, d1)

    # CSV
    write_csv(csv_dir / "d1_per_group_metric_diffs.csv", all_d1)
    write_csv(csv_dir / "d2_worstgroup_identity.csv", all_d2)
    write_csv(csv_dir / "d3_perfold_worstgroup_diff.csv", all_d3)
    write_csv(csv_dir / "d4_gap_decomposition.csv", all_d4)
    print(f"\n[CSV] 已写入 {csv_dir}/ （d1..d4）")

    # Markdown（拼到 stdout，供人工核对/贴入文档）
    md: list[str] = ["## 跨数据集 / 跨架构 worst-group 公平性支撑表（CV-OOF）\n"]
    md += md_d1_blocks
    md_d2(md, all_d2)
    md_d3(md, all_d3)
    md_d4(md, all_d4)
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
