"""
CV-OOF 池化评估 / worst-group 显著性报告器（方案 S2，5 数据集通用）
====================================================================
`docs/oof_selection_rollout_plan.md` 三决策之③：**评估/报告用 pooled-OOF 上的 canonical worst-group
+ Overall + gap（样本级配对 bootstrap）**。本脚本把 `scripts/ham_cv_oof_significance.py`（HAM 专用）
泛化为 **`--dataset` 参数化**，供全 5 数据集复用同一评估口径。

**pooled-OOF**：把选定 config 的 5 折 disjoint test 预测拼成全数据集 OOF（每样本恰被其留出折预测
一次，从未参与选择 → 三重无泄漏）。少数子群评估 n 放大约 5×，抬起单-split 的噪声地板。

**选定 config 来源**：`--config-json`（`select_config_cv.py --write-json` 的输出），或 `--config
method=tag` 显式覆盖。swad/roc 复用 erm 的选定 config。

**口径注脚**：早停=Overall AUC、config 选择=折均 marginal-worst、报告=canonical-worst on pooled-OOF。

用法：
  # 用 S1 选定配置（推荐）
  python scripts/cv_oof_report.py --dataset ham10000 --config-json outputs/ham10000/cv5/selected_configs.json
  # 或显式指定（如对账路线 A 旧单-split 配置）
  python scripts/cv_oof_report.py --dataset ham10000 \
      --config erm=lr1e-04_wd1e-04 swad=lr1e-04_wd1e-04 roc=lr1e-04_wd1e-04 \
               hyperhead=lr1e-04_wd1e-03 hyperfusion=lr3e-04_wd1e-03 hyperadapt=lr3e-04_wd1e-04
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.metrics import roc_auc_score

from src.training.harness.predictions import load_predictions
from src.training.harness.subgroup_auc import subgroup_auc_vector, subgroup_masks
from src.training.harness.significance import (
    worst_and_gap, paired_bootstrap_diff, delong_test,
)
from src.utils.operating_point_fairness import (
    to_prob, _clf_metrics, subgroup_classification_metrics,
    worst_group_clf, clf_gap, threshold_at_overall_fpr,
)
from src.paths import OUTPUTS_DIR

OUTPUTS_ROOT = OUTPUTS_DIR
FOLD_SEEDS = (42, 43, 44, 45, 46)   # fold k ↔ seed 42+k（CV 搜索约定）

# `hyperadapt_swad` = 实验 G（HN×SWAD 融合）的融合臂，**追加在末尾**使既有方法顺序/行序不变；
# 仅在 selected_configs.json 含其 key 且预测存在时才被载入（available_methods 会跳过缺失者）。
METHOD_ORDER = ("erm", "swad", "roc", "groupdro",
                "hyperhead", "hyperfusion", "hyperadapt", "hyperadapt_swad")
METHOD_LABEL = {"erm": "ERM", "swad": "SWAD", "roc": "ROC", "groupdro": "GroupDRO",
                "hyperhead": "HyperHead", "hyperfusion": "HyperFusion", "hyperadapt": "HyperAdapt",
                "hyperadapt_swad": "HyperAdapt+SWAD"}
# ⚠️ `HN` 定义的是**已冻结主矩阵**里「HN vs ERM」的对比集（D1/D3/D4 逐 HN 出表），
# 故融合臂**不**加入——实验 G 的 2×2 与交互对比在其专属分析脚本里做，不改写主矩阵结论表。
HN = ("hyperhead", "hyperfusion", "hyperadapt")
# `groupdro` 是**训练时用子群标签**的基线（Sagawa ICLR 2020），加入基线列后 HN 会额外与它对比——
# 这正是它存在的意义：把「用属性 vs 不用属性」从「HN vs 全属性盲基线」的混淆对比里解耦出来。
BASE = ("erm", "swad", "roc", "groupdro")
# ROC 落已调整概率，余方法落 logits（决定 to_prob 是否先 sigmoid）
SCORE_IS_PROB = {m: (m == "roc") for m in METHOD_ORDER}


def _ham_age_filter(attrs: dict[str, np.ndarray]) -> np.ndarray:
    """HAM：过滤 age_group==-1（0-20 排除组），与 HN 训练/单-split 口径一致。"""
    return np.asarray(attrs["age"]) >= 0


# 数据集 → (预测目录相对 outputs, subgroup key, OOF 过滤器或 None)
# subgroup key 用于 subgroup_auc_vector/subgroup_masks；npz attr 键名与其 kwargs 同名，直接 splat。
DATASET_SPEC: dict[str, tuple[str, str, Callable[[dict[str, np.ndarray]], np.ndarray] | None]] = {
    "ham10000":    ("ham10000/cv5/predictions",    "ham10000",    _ham_age_filter),
    "papila":      ("papila/predictions",          "papila",      None),
    "fitzpatrick": ("fitzpatrick/cv5/predictions", "fitzpatrick", None),
    "chexpert":    ("chexpert_cxr/cv5/predictions", "chexpert",   None),
    "mimic":       ("mimic_cxr/cv5/predictions",   "mimic",       None),
}


def _sel(method: str) -> str:
    """ROC/SWAD 仅落 overall 预测；HN/ERM 亦取 overall selection（决策①早停口径）。"""
    return "overall"


def pool_oof(
    pred_dir: Path, method: str, tag: str,
    ds_filter: Callable[[dict[str, np.ndarray]], np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    池化某方法 5 折 disjoint test 预测为全数据集 OOF（每样本恰 1 个预测，来自其留出折）。

    Args:
        pred_dir : 该数据集 CV 预测目录（含 <method>_<tag>_seed{42+k}_overall.npz）。
        method   : 方法名。
        tag      : 该方法选定 config_tag。
        ds_filter: OOF 过滤器（如 HAM 排 age=-1）；None 表示不过滤。

    Returns:
        (y_true_pool [N], score_pool [N], attrs_pool {attr -> [N]})，按折序拼接。
    """
    ys, ss = [], []
    pooled_attrs: dict[str, list[np.ndarray]] = {}
    for seed in FOLD_SEEDS:
        path = pred_dir / f"{method}_{tag}_seed{seed}_{_sel(method)}.npz"
        y, s, a = load_predictions(path)
        mask = ds_filter(a) if ds_filter is not None else np.ones(len(y), dtype=bool)
        ys.append(np.asarray(y)[mask]); ss.append(np.asarray(s)[mask])
        for k, v in a.items():
            pooled_attrs.setdefault(k, []).append(np.asarray(v)[mask])
    y = np.concatenate(ys); s = np.concatenate(ss)
    attrs = {k: np.concatenate(v) for k, v in pooled_attrs.items()}
    return y, s, attrs


def load_all_pooled(
    pred_dir: Path, cfg: dict[str, str],
    ds_filter: Callable[[dict[str, np.ndarray]], np.ndarray] | None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    """
    载入全部可用方法的池化 OOF，**硬断言样本对齐**（配对 bootstrap/DeLong 正确性前提）。

    某方法若缺预测文件（如某数据集尚未派生 ROC），跳过并记入 missing，而非中止——便于半成品数据集
    也能出 perf 表。各方法按同一折序、同一过滤，故 y_true 应逐元素相等。

    Returns:
        (y_true [N], scores {method -> [N]}, attrs {attr -> [N]}, methods 实际可用方法列表)。
    """
    y_ref = attrs = None
    scores: dict[str, np.ndarray] = {}
    methods: list[str] = []
    for method in METHOD_ORDER:
        tag = cfg.get(method)
        if tag is None:
            continue
        path0 = pred_dir / f"{method}_{tag}_seed{FOLD_SEEDS[0]}_{_sel(method)}.npz"
        if not path0.exists():
            print(f"  [跳过] {method} 缺预测：{path0.name}")
            continue
        y_m, s_m, a_m = pool_oof(pred_dir, method, tag, ds_filter)
        if y_ref is None:
            y_ref, attrs = y_m, a_m
        else:
            assert y_m.shape == y_ref.shape and np.array_equal(y_m, y_ref), (
                f"{method} 池化 OOF 与首方法样本不对齐（配对检验非法）：{y_m.shape} vs {y_ref.shape}")
            for k in attrs:
                assert np.array_equal(a_m[k], attrs[k]), f"{method} 属性 {k} 与首方法不对齐"
        scores[method] = s_m
        methods.append(method)
    assert y_ref is not None, "无任何方法预测可载入"
    return y_ref, scores, attrs, methods


def _marg_worst_from_vec(vec) -> float | None:
    """从 (AUC,n) 向量取 marginal worst-group（仅单属性子群，排 joint `|`）。"""
    vals = [a for k, (a, _n) in vec.items() if a is not None and "|" not in k]
    return min(vals) if vals else None


def perf_table(ds_key: str, y, scores, attrs, methods) -> None:
    """D-A1：池化 OOF 上各方法 Overall / canonical worst / marginal worst / gap（单点估计）。"""
    print(f"\n### D-A1 {ds_key} CV-OOF 性能与公平（池化全 {len(y):,} 样本，单点估计）")
    print("| 方法 | Overall AUC | canonical worst | marginal worst | AUC gap |")
    print("|------|-------------|-----------------|----------------|---------|")
    for method in methods:
        s = scores[method]
        ov = roc_auc_score(y, s)
        vec = subgroup_auc_vector(ds_key, y, s, **attrs)
        cw, gap = worst_and_gap(vec)
        mw = _marg_worst_from_vec(vec)
        print(f"| {METHOD_LABEL[method]} | {ov:.4f} | {_fmt(cw)} | {_fmt(mw)} | {_fmt(gap)} |")


def diag_table(ds_key: str, y, scores, attrs, methods) -> None:
    """D-A2：逐子群 n / n_pos / n_neg / AUC / bootstrap-SE 诊断（首方法代表分数）——坐实评估-n 地板。"""
    s = scores[methods[0]]
    masks = subgroup_masks(ds_key, **attrs)
    rng = np.random.default_rng(0)
    print(f"\n### D-A2 {ds_key} CV-OOF 逐子群样本量诊断（{METHOD_LABEL[methods[0]]} 代表分数，池化全 {len(y):,}）")
    print("| 子群 | n | n_pos | n_neg | AUC | boot-SE | 地板 |")
    print("|------|---|-------|-------|-----|---------|------|")
    for name, m in masks.items():
        yi, si = y[m], s[m]
        n, n_pos = int(m.sum()), int(yi.sum())
        n_neg = n - n_pos
        if n_pos == 0 or n_neg == 0:
            print(f"| {name} | {n} | {n_pos} | {n_neg} | N/A | N/A | ⚠️退化 |")
            continue
        auc = roc_auc_score(yi, si)
        boots, idx = [], np.arange(n)
        for _ in range(500):
            b = rng.choice(idx, size=n, replace=True)
            if 0 < yi[b].sum() < n:
                boots.append(roc_auc_score(yi[b], si[b]))
        se = float(np.std(boots)) if boots else float("nan")
        floor = "⚠️噪声地板" if (se >= 0.10 or min(n_pos, n_neg) < 10) else ""
        print(f"| {name} | {n} | {n_pos} | {n_neg} | {auc:.3f} | {se:.3f} | {floor} |")


def _canon_worst(ds_key, y, s, attrs):
    """canonical worst-group（含 joint）的 metric 闭包，供 paired_bootstrap_diff。"""
    def metric(idx):
        vec = subgroup_auc_vector(ds_key, y[idx], s[idx], **{k: v[idx] for k, v in attrs.items()})
        return worst_and_gap(vec)[0]
    return metric


def _marg_worst(ds_key, y, s, attrs):
    """marginal worst-group（排 joint）的 metric 闭包，隔离小格噪声。"""
    def metric(idx):
        vec = subgroup_auc_vector(ds_key, y[idx], s[idx], **{k: v[idx] for k, v in attrs.items()})
        return _marg_worst_from_vec(vec)
    return metric


def sig_tables(ds_key: str, y, scores, attrs, methods, n_boot: int = 2000) -> None:
    """D-A3/D-A4：worst-group 样本级配对 bootstrap（canonical + marginal）+ Overall DeLong（HN vs 基线）。

    n_boot：配对 bootstrap 重采样次数（默认 2000）。巨大 n 数据集（如 MIMIC n≈199k、CheXpert n≈138k）
    上单次 subgroup-AUC 重算很贵，可降到 1000 —— 此时 bootstrap CI 已 ±0.002~0.003、verdict 不变。
    """
    n = len(y)
    rng = np.random.default_rng(0)
    hns = [m for m in HN if m in methods]
    bases = [m for m in BASE if m in methods]
    if not hns or not bases:
        print("\n[显著性跳过] HN 或基线方法预测不全。")
        return

    def _sig_block(title: str, closure) -> None:
        print(title)
        print("| HN \\ 基线 | " + " | ".join(f"vs {METHOD_LABEL[b]}" for b in bases) + " |")
        print("|" + "-----------|" * (len(bases) + 1))
        for hn in hns:
            cells = []
            for base in bases:
                d = paired_bootstrap_diff(n, closure(y, scores[hn], attrs),
                                          closure(y, scores[base], attrs), n_boot=n_boot, rng=rng)
                cells.append(f"{d.point:+.4f} [{d.ci_low:+.4f},{d.ci_high:+.4f}] "
                             f"{'**显著**' if d.significant else 'n.s.'}")
            print(f"| {METHOD_LABEL[hn]} | " + " | ".join(cells) + " |")

    _sig_block(
        f"\n### D-A3a {ds_key} CV-OOF canonical worst-group 配对 bootstrap（HN − 基线，n={n:,}，{n_boot} 次）",
        lambda y, s, a: _canon_worst(ds_key, y, s, a))
    _sig_block(
        f"\n### D-A3b {ds_key} CV-OOF marginal worst-group 配对 bootstrap（隔离 joint 噪声，n={n:,}）",
        lambda y, s, a: _marg_worst(ds_key, y, s, a))

    # Overall AUC DeLong
    print(f"\n### D-A4 {ds_key} CV-OOF Overall AUC DeLong（HN vs 基线，池化 OOF n={n:,}）")
    print("| HN \\ 基线 | " + " | ".join(f"vs {METHOD_LABEL[b]}" for b in bases) + " |")
    print("|" + "-----------|" * (len(bases) + 1))
    for hn in hns:
        cells = []
        for base in bases:
            r = delong_test(y, scores[hn], scores[base])
            cells.append(f"Δ={r.diff:+.4f} (p={r.p_value:.4f}{', **显著**' if r.significant else ''})")
        print(f"| {METHOD_LABEL[hn]} | " + " | ".join(cells) + " |")


def clinical_tables(ds_key: str, y, scores, attrs, methods) -> None:
    """D-A5/D-A6：临床四联指标（sensitivity/specificity/PPV/NPV）在两操作点（原生 0.5 / 固定 FPR=0.2）。"""
    for target_desc, get_thr in (
        ("原生阈值 prob=0.5", None),
        ("固定整体 FPR=0.2（specificity≈0.8，跨方法可比）", 0.2),
    ):
        print(f"\n### D-A5 {ds_key} CV-OOF 整体临床四联（{target_desc}，n={len(y):,}）")
        print("| 方法 | Sensitivity | Specificity | PPV | NPV | 用阈值 |")
        print("|------|-------------|-------------|-----|-----|--------|")
        for method in methods:
            prob = to_prob(scores[method], SCORE_IS_PROB[method])
            thr = 0.5 if get_thr is None else threshold_at_overall_fpr(y, prob, get_thr)
            m = _clf_metrics(y, (prob >= thr).astype(int))
            thr_show = "0.5(prob)" if get_thr is None else f"p={thr:.3f}"
            print(f"| {METHOD_LABEL[method]} | {_fmt(m.sensitivity)} | {_fmt(m.specificity)} | "
                  f"{_fmt(m.ppv)} | {_fmt(m.npv)} | {thr_show} |")

        print(f"\n### D-A6 {ds_key} CV-OOF worst-group 临床指标（{target_desc}，marginal 组，min_class_n=10）")
        print("| 方法 | worst Sens | Sens gap | worst Spec | Spec gap | worst PPV | worst NPV |")
        print("|------|-----------|----------|-----------|----------|-----------|-----------|")
        for method in methods:
            prob = to_prob(scores[method], SCORE_IS_PROB[method])
            thr = 0.5 if get_thr is None else threshold_at_overall_fpr(y, prob, get_thr)
            rates = subgroup_classification_metrics(ds_key, y, prob, attrs, thr, marginal_only=True)
            print(f"| {METHOD_LABEL[method]} | {_fmt(worst_group_clf(rates, 'sensitivity'))} | "
                  f"{_fmt(clf_gap(rates, 'sensitivity'))} | {_fmt(worst_group_clf(rates, 'specificity'))} | "
                  f"{_fmt(clf_gap(rates, 'specificity'))} | {_fmt(worst_group_clf(rates, 'ppv'))} | "
                  f"{_fmt(worst_group_clf(rates, 'npv'))} |")


def _fmt(v: float | None) -> str:
    """AUC/指标格式化（None → 'N/A'）。"""
    return "N/A" if v is None else f"{v:.4f}"


def resolve_cfg(args: argparse.Namespace) -> dict[str, str]:
    """从 --config-json 与 --config CLI 覆盖解析出 {method: config_tag}（CLI 覆盖优先）。"""
    cfg: dict[str, str] = {}
    if args.config_json is not None:
        data = json.loads(Path(args.config_json).read_text())
        cfg.update(data["config"] if "config" in data else data)
    for item in (args.config or []):
        method, _, tag = item.partition("=")
        if not tag:
            raise SystemExit(f"--config 项格式应为 method=tag：{item!r}")
        cfg[method] = tag
    if not cfg:
        raise SystemExit("需 --config-json 或 --config 指定选定配置")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="CV-OOF 池化评估 / worst-group 显著性（S2，通用）")
    parser.add_argument("--dataset", required=True, choices=tuple(DATASET_SPEC))
    parser.add_argument("--config-json", type=str, default=None,
                        help="select_config_cv --write-json 的输出 JSON")
    parser.add_argument("--config", nargs="*", default=None,
                        help="显式 method=tag 覆盖（如 erm=lr1e-04_wd1e-04）")
    parser.add_argument("--pred-dir", type=str, default=None, help="显式预测目录（覆盖默认）")
    parser.add_argument("--section", choices=("perf", "diag", "sig", "clinical", "all"), default="all")
    parser.add_argument("--n-boot", type=int, default=2000,
                        help="配对 bootstrap 次数（默认 2000；巨大 n 数据集可降 1000，CI 仍 ±0.002~0.003）")
    args = parser.parse_args()

    rel_pred, ds_key, ds_filter = DATASET_SPEC[args.dataset]
    pred_dir = Path(args.pred_dir) if args.pred_dir else OUTPUTS_ROOT / rel_pred
    cfg = resolve_cfg(args)

    print(f"# 数据集={args.dataset}  pred_dir={pred_dir}")
    print(f"# 选定配置：" + "  ".join(f"{m}={cfg[m]}" for m in METHOD_ORDER if m in cfg))
    y, scores, attrs, methods = load_all_pooled(pred_dir, cfg, ds_filter)
    print(f"[CV-OOF 池化] n={len(y):,}  阳性率={y.mean():.3f}  "
          f"（{len(FOLD_SEEDS)} 折 disjoint test，{len(methods)} 方法样本对齐已断言）")

    if args.section in ("perf", "all"):
        perf_table(ds_key, y, scores, attrs, methods)
    if args.section in ("diag", "all"):
        diag_table(ds_key, y, scores, attrs, methods)
    if args.section in ("sig", "all"):
        sig_tables(ds_key, y, scores, attrs, methods, n_boot=args.n_boot)
    if args.section in ("clinical", "all"):
        clinical_tables(ds_key, y, scores, attrs, methods)


if __name__ == "__main__":
    main()
