"""
D 汇总数据生成：ID 性能/公平表（D3/D5）+ OOD 表（D4）+ HAM 显著性（D6）
====================================================================
从落盘预测 npz 统一重算所有结果表，口径与训练/选择一致（`subgroup_auc_vector` 数据集感知 +
`worst_and_gap` + `mean_ci` Student-t 95%CI）。输出 markdown 便于粘入 `docs/results_summary.md`。

- ID（D3/D5）：每数据集 × 6 方法，5-seed 均值±95%CI 的 Overall / worst-group / AUC gap。
  PAPILA 走全-CV：5 折 disjoint test 池化成全 420 眼 OOF，报池化点估计（无 seed CI，footnote 说明）。
  HAM：基线(ERM/SWAD/ROC) 含 age_group=-1(n=995)，HN 过滤(n=969)；已验证过滤后逐元素对齐，
       故 HAM 全方法统一过滤 age>=0 到 969，保证同 test 集可比 + 配对 bootstrap 合法。
- OOD（D4）：MIMIC↔CheXpert 双向，5 方法 × 5 seed 均值±95%CI。
- D6：HAM HyperFusion vs {ERM,SWAD,ROC} 的 worst-group 配对 bootstrap（样本×种子两层不确定度）
      + Overall AUC DeLong（逐 seed 平均 z/p 的代表 seed）。

用法：python scripts/build_results_tables.py [--section id|ood|sig|all]
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.training.harness.predictions import load_predictions
from src.training.harness.subgroup_auc import subgroup_auc_vector, SUBGROUP_VECTOR_DIM
from src.training.harness.significance import (
    worst_and_gap, mean_ci, paired_bootstrap_diff, make_subgroup_metric_multiseed, delong_test,
)

# R0.1 回归卫生：凡走 canonical worst-group 口径的数据集，其子群向量**必须含 joint 格**（键带 "|"）。
# 单属性数据集（fitzpatrick 仅 skin）无 joint、mimic_synth 仅 2 组，故豁免。用于 _metrics_one 断言，
# 锁死「全部主表走 canonical subgroup_auc_vector（含 joint）」，杜绝 C6 边缘-only vs D4 canonical 漂移复发。
_CANONICAL_JOINT_REQUIRED = {"mimic", "chexpert", "ham10000", "papila"}

OUT = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs")
SEEDS = (42, 43, 44, 45, 46)
METHOD_ORDER = ("erm", "swad", "roc", "hyperhead", "hyperfusion", "hyperadapt")
METHOD_LABEL = {"erm": "ERM", "swad": "SWAD", "roc": "ROC",
                "hyperhead": "HyperHead", "hyperfusion": "HyperFusion", "hyperadapt": "HyperAdapt"}

# 每数据集：subgroup 键、predictions 目录、所需 attrs、各方法 Pareto 选定 config_tag、是否 OOF 池化
DATASETS: dict[str, dict] = {
    "ham10000": dict(
        key="ham10000", pdir=OUT / "ham10000" / "predictions", attrs=("sex", "age"), oof=False,
        ham_filter=True,
        cfg={"erm": "lr1e-04_wd1e-04", "swad": "lr1e-04_wd1e-04", "roc": "lr1e-04_wd1e-04",
             "hyperhead": "lr1e-04_wd1e-03", "hyperfusion": "lr3e-04_wd1e-03",
             "hyperadapt": "lr3e-04_wd1e-04"}),
    "mimic": dict(
        key="mimic", pdir=OUT / "mimic_cxr" / "predictions", attrs=("sex", "race", "age"), oof=False,
        ham_filter=False,
        cfg={"erm": "lr3e-04_wd1e-03", "swad": "lr3e-04_wd1e-03", "roc": "lr3e-04_wd1e-03",
             "hyperhead": "lr1e-04_wd1e-04", "hyperfusion": "lr1e-04_wd1e-04",
             "hyperadapt": "lr3e-04_wd1e-04"}),
    "chexpert": dict(
        key="chexpert", pdir=OUT / "chexpert_cxr" / "predictions", attrs=("sex", "race", "age"), oof=False,
        ham_filter=False,
        cfg={"erm": "lr3e-04_wd1e-03", "swad": "lr3e-04_wd1e-03", "roc": "lr3e-04_wd1e-03",
             "hyperhead": "lr3e-05_wd1e-03", "hyperfusion": "lr1e-04_wd1e-04",
             "hyperadapt": "lr3e-04_wd1e-03"}),
    "fitzpatrick": dict(
        key="fitzpatrick", pdir=OUT / "fitzpatrick" / "predictions", attrs=("skin",), oof=False,
        ham_filter=False,
        cfg={"erm": "lr1e-04_wd1e-03", "swad": "lr1e-04_wd1e-03", "roc": "lr1e-04_wd1e-03",
             "hyperhead": "lr1e-04_wd1e-04", "hyperfusion": "lr3e-04_wd1e-04",
             "hyperadapt": "lr1e-04_wd1e-03"}),
    "papila": dict(
        key="papila", pdir=OUT / "papila" / "predictions", attrs=("sex", "age"), oof=True,
        ham_filter=False,
        cfg={"erm": "lr3e-04_wd1e-04", "swad": "lr3e-04_wd1e-04", "roc": "lr3e-04_wd1e-04",
             "hyperhead": "lr3e-04_wd1e-04", "hyperfusion": "lr1e-04_wd1e-03",
             "hyperadapt": "lr3e-04_wd1e-04"}),
}


SELECTION = "overall"   # 由 --selection 覆盖（overall / worstcase）；ROC/SWAD 只有 overall 落盘


def _sel_for(method: str) -> str:
    """ROC/SWAD 仅落 overall 预测（派生自 ERM overall），恒用 overall；余用全局 SELECTION。"""
    return "overall" if method in ("roc", "swad") else SELECTION


def _load(pdir: Path, method: str, tag: str, seed: int, ham_filter: bool):
    """加载单 run 预测，返回 (y, s, attrs_dict)；HAM 过滤 age>=0 使全方法同 test 集。"""
    y, s, a = load_predictions(pdir / f"{method}_{tag}_seed{seed}_{_sel_for(method)}.npz")
    if ham_filter:
        m = a["age"] >= 0
        y, s = y[m], s[m]
        a = {k: v[m] for k, v in a.items()}
    return np.asarray(y), np.asarray(s), {k: np.asarray(v) for k, v in a.items()}


def _assert_canonical_vector(key: str, vec) -> None:
    """
    R0.1 断言：vec 是 canonical 子群向量（固定 schema、含 joint），非边缘-only 子集。

    两条守卫，任一不满足即 AssertionError（防口径漂移）：
      1. 维度 == SUBGROUP_VECTOR_DIM[key]（边缘-only 会短于 canonical，如 HAM 6<14）。
      2. 多属性数据集（_CANONICAL_JOINT_REQUIRED）必须含至少一个 joint 键（含 "|"）。
    """
    expected = SUBGROUP_VECTOR_DIM.get(key)
    assert expected is None or len(vec) == expected, (
        f"[R0.1] {key} worst-group 口径漂移：子群向量维度 {len(vec)} ≠ canonical {expected}"
        "（疑似退回边缘-only 口径）。全部主表须走 canonical subgroup_auc_vector。")
    if key in _CANONICAL_JOINT_REQUIRED:
        assert any("|" in k for k in vec), (
            f"[R0.1] {key} 子群向量缺 joint 格（无 '|' 键）：canonical 口径必须含 joint，"
            "疑似退回边缘-only。")


def _metrics_one(key: str, y, s, attrs) -> tuple[float, float | None, float | None]:
    """单 run 的 (Overall, worst-group, gap)，worst/gap 用数据集感知 canonical subgroup 向量（含 joint）。"""
    ov = roc_auc_score(y, s)
    vec = subgroup_auc_vector(key, y, s, **{k: attrs[k] for k in attrs})
    _assert_canonical_vector(key, vec)   # R0.1：锁死 canonical 口径，杜绝边缘-only 漂移
    w, g = worst_and_gap(vec)
    return ov, w, g


def _fmt(mc) -> str:
    """MeanCI → 'mean±half' 文本。"""
    return f"{mc.mean:.4f}±{mc.half_width:.4f}"


def id_tables() -> None:
    """D3/D5：各数据集 6 方法的 Overall / worst-group / gap（5-seed 均值±95%CI 或 PAPILA OOF 点估计）。"""
    for ds, spec in DATASETS.items():
        print(f"\n### {ds}  (subgroup 轴: {', '.join(spec['attrs'])}"
              f"{'; 全-CV OOF 池化 n=420' if spec['oof'] else '; 5-seed on fixed test'})")
        print(f"| 方法 | Overall AUC | Worst-group AUC | AUC gap |")
        print(f"|------|-------------|-----------------|---------|")
        for method in METHOD_ORDER:
            tag = spec["cfg"][method]
            if spec["oof"]:
                # 池化 5 折 disjoint test → 全 420 眼一次算
                ys, ss = [], []
                pool_attrs = {k: [] for k in spec["attrs"]}
                for sd in SEEDS:
                    y, s, a = _load(spec["pdir"], method, tag, sd, spec["ham_filter"])
                    ys.append(y); ss.append(s)
                    for k in spec["attrs"]:
                        pool_attrs[k].append(a[k])
                y = np.concatenate(ys); s = np.concatenate(ss)
                A = {k: np.concatenate(v) for k, v in pool_attrs.items()}
                ov, w, g = _metrics_one(spec["key"], y, s, A)
                gs = "N/A" if g is None else f"{g:.4f}"
                print(f"| {METHOD_LABEL[method]} | {ov:.4f} | {w:.4f} | {gs} | ")
            else:
                O, W, G = [], [], []
                for sd in SEEDS:
                    y, s, a = _load(spec["pdir"], method, tag, sd, spec["ham_filter"])
                    ov, w, g = _metrics_one(spec["key"], y, s, a)
                    O.append(ov); W.append(w); G.append(g)
                print(f"| {METHOD_LABEL[method]} | {_fmt(mean_ci(O))} | "
                      f"{_fmt(mean_ci(W))} | {_fmt(mean_ci(G))} |")
        if spec["oof"]:
            print("> PAPILA 为全-CV：数字为 5 折 disjoint test 池化的全 420 眼 OOF 点估计（非 seed 均值）。")


def ood_table() -> None:
    """D4：MIMIC↔CheXpert 双向 OOD，5 方法 × 5 seed 均值±95%CI。"""
    root = OUT / "ood_cxr"
    for direction, key in (("mimic2chexpert", "chexpert"), ("chexpert2mimic", "mimic")):
        print(f"\n### OOD {direction}  (目标子群轴口径={key})")
        print(f"| 方法 | Overall AUC | Worst-group AUC | AUC gap |")
        print(f"|------|-------------|-----------------|---------|")
        for method in ("erm", "swad", "hyperhead", "hyperfusion", "hyperadapt"):
            O, W, G = [], [], []
            for sd in SEEDS:
                y, s, a = load_predictions(root / direction / f"{method}_seed{sd}.npz")
                ov, w, g = _metrics_one(key, np.asarray(y), np.asarray(s),
                                        {k: np.asarray(v) for k, v in a.items()})
                O.append(ov); W.append(w); G.append(g)
            print(f"| {METHOD_LABEL[method]} | {_fmt(mean_ci(O))} | "
                  f"{_fmt(mean_ci(W))} | {_fmt(mean_ci(G))} |")


def ham_significance() -> None:
    """D6：HAM HyperFusion vs {ERM,SWAD,ROC} —— worst-group 配对 bootstrap + Overall DeLong。"""
    spec = DATASETS["ham10000"]
    key, pdir = spec["key"], spec["pdir"]
    # 载入各方法 5-seed 预测（HAM 过滤 age>=0；全方法逐元素对齐，共享同一 y_true）
    def load_all(method):
        tag = spec["cfg"][method]
        ys, ss, A = [], [], None
        for sd in SEEDS:
            y, s, a = _load(pdir, method, tag, sd, True)
            ys.append(s)
            if A is None: y0, A = y, a
        return y0, ys, A
    HN = ("hyperhead", "hyperfusion", "hyperadapt")
    BASE = ("erm", "swad", "roc")
    y_true = load_all("erm")[0]
    scores = {m: load_all(m)[1] for m in HN + BASE}
    attrs = load_all("erm")[2]

    def marginal_worst_metric(y, s_list, rng_):
        """闭包：随机抽 seed → 仅边缘子群(sex/age，排 joint)最小 AUC，隔离小格噪声。"""
        n_seed = len(s_list)
        def metric(idx):
            s = s_list[int(rng_.integers(0, n_seed))]
            vec = subgroup_auc_vector(key, y[idx], s[idx],
                                      sex=attrs["sex"][idx], age=attrs["age"][idx])
            vals = [a for k, (a, _n) in vec.items() if a is not None and "|" not in k]
            return min(vals) if vals else None
        return metric

    # (1) worst-group 配对 bootstrap：3 HN × 3 基线（canonical 14 子群，含 joint 小格）
    print("\n### D6-a HAM worst-group 配对 bootstrap（HN − 基线，canonical 14 子群，样本×种子两层，2000 次）")
    print("| HN \\ 基线 | vs ERM | vs SWAD | vs ROC |")
    print("|-----------|--------|---------|--------|")
    rng = np.random.default_rng(0)
    for hn in HN:
        cells = []
        for base in BASE:
            m_a = make_subgroup_metric_multiseed(key, y_true, scores[hn], attrs, "worst", rng)
            m_b = make_subgroup_metric_multiseed(key, y_true, scores[base], attrs, "worst", rng)
            d = paired_bootstrap_diff(len(y_true), m_a, m_b, n_boot=2000, rng=rng)
            cells.append(f"{d.point:+.3f} [{d.ci_low:+.3f},{d.ci_high:+.3f}] "
                         f"{'**显著**' if d.significant else 'n.s.'}")
        print(f"| {METHOD_LABEL[hn]} | {cells[0]} | {cells[1]} | {cells[2]} |")

    # (2) 边缘 worst-group 配对 bootstrap：3 HN vs ERM（隔离 joint 小格噪声，更敏感）
    print("\n### D6-b HAM 边缘 worst-group 配对 bootstrap（HN − ERM，仅 sex/age 6 组，隔离 joint 噪声）")
    print("| 对比 | Δ 边缘worst | 95% CI | p | 判定 |")
    print("|------|-------------|--------|---|------|")
    for hn in HN:
        m_a = marginal_worst_metric(y_true, scores[hn], rng)
        m_b = marginal_worst_metric(y_true, scores["erm"], rng)
        d = paired_bootstrap_diff(len(y_true), m_a, m_b, n_boot=2000, rng=rng)
        print(f"| {METHOD_LABEL[hn]} vs ERM | {d.point:+.4f} | "
              f"[{d.ci_low:+.4f}, {d.ci_high:+.4f}] | {d.p_value:.3f} | "
              f"{'**显著**' if d.significant else 'n.s.'} |")

    # (3) Overall AUC DeLong：3 HN × 3 基线（逐 seed 配对，报均值）
    print("\n### D6-c HAM Overall AUC DeLong（HN vs 基线，5-seed 平均 ΔAUC / p）")
    print("| HN \\ 基线 | vs ERM | vs SWAD | vs ROC |")
    print("|-----------|--------|---------|--------|")
    for hn in HN:
        cells = []
        for base in BASE:
            diffs, ps = [], []
            for i in range(len(SEEDS)):
                r = delong_test(y_true, scores[hn][i], scores[base][i])
                diffs.append(r.diff); ps.append(r.p_value)
            cells.append(f"{np.mean(diffs):+.4f} (p={np.mean(ps):.3f})")
        print(f"| {METHOD_LABEL[hn]} | {cells[0]} | {cells[1]} | {cells[2]} |")


def _ham_ensemble_scores() -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, list[np.ndarray]], dict[str, np.ndarray]]:
    """
    R4.1 用：载入 HAM 全方法 5-seed 预测，构建 **seed-ensemble**（概率空间均值）分数。

    ensemble 把逐 seed 的模型随机性平均掉（方差 ~1/√5），用于**更高功效**的显著性分析：
    logit 尺度方法（ERM/SWAD/HN）先过 sigmoid 再跨 seed 平均，ROC 本就是概率、直接平均。
    AUC/worst-group 是秩泛函，sigmoid 单调不改单 seed 排序，跨 seed 平均则真正降方差。

    Returns:
        (y_true, ens[method]->[N] 集成分数, seed_scores[method]->list[[N]] 逐 seed 概率, attrs)。
    """
    spec = DATASETS["ham10000"]
    pdir = spec["pdir"]

    def load_all(method: str) -> tuple[np.ndarray, list[np.ndarray], dict]:
        tag = spec["cfg"][method]
        ys, A, y0 = [], None, None
        for sd in SEEDS:
            y, s, a = _load(pdir, method, tag, sd, True)
            ys.append(s if method == "roc" else 1.0 / (1.0 + np.exp(-s)))  # 概率空间
            if A is None:
                y0, A = y, a
        return y0, ys, A

    methods = ("erm", "swad", "roc", "hyperhead", "hyperfusion", "hyperadapt")
    y_true, _, attrs = load_all("erm")
    seed_scores = {m: load_all(m)[1] for m in methods}
    ens = {m: np.mean(seed_scores[m], axis=0) for m in methods}
    return y_true, ens, seed_scores, attrs


def ham_significance_highpower() -> None:
    """
    D6-HP（R4.1）：HAM 更高功效显著性 —— 对 **seed-ensemble** 分数做样本级配对 bootstrap /
    DeLong，作为 5-seed「pick-1-seed 两层 bootstrap」（D6-a/b/c）的功效补强。

    动机（评审 R4.1）：D6 在 n=5 seed + joint 小格下功效不足、无一 p<0.05。此处走协议
    sanction 的「跨 seed 池化 test 样本级 bootstrap」路径（非重训 15 seed）：seed-ensemble
    把模型方差降 ~√5，样本级配对 bootstrap / DeLong 只保留 **test 采样**不确定度，功效更高。
    与 D6-a/b/c 并列——若 ensemble 下仍 n.s.，说明瓶颈是评估端小样本（R4.2），非 seed 数。

    口径说明：ensemble-DeLong 的 p 是「集成分数在此 test 集上的 Overall AUC 差异」显著性
    （test 采样口径，条件于 ensemble、不再传播 seed 方差）；配对 bootstrap 同理。这是比
    D6-c「逐 seed DeLong 平均 p」更高功效但更窄口径的一击，两者并置读。
    """
    key = "ham10000"
    y_true, ens, _seed_scores, attrs = _ham_ensemble_scores()
    n = len(y_true)
    HN = ("hyperhead", "hyperfusion", "hyperadapt")
    BASE = ("erm", "swad", "roc")

    def canon_worst(s: np.ndarray) -> MetricFnLocal:
        def metric(idx: np.ndarray):
            vec = subgroup_auc_vector(key, y_true[idx], s[idx],
                                      sex=attrs["sex"][idx], age=attrs["age"][idx])
            return worst_and_gap(vec)[0]
        return metric

    def marg_worst(s: np.ndarray) -> MetricFnLocal:
        def metric(idx: np.ndarray):
            vec = subgroup_auc_vector(key, y_true[idx], s[idx],
                                      sex=attrs["sex"][idx], age=attrs["age"][idx])
            vals = [a for k, (a, _n) in vec.items() if a is not None and "|" not in k]
            return min(vals) if vals else None
        return metric

    rng = np.random.default_rng(0)
    # (a) canonical worst-group（含 joint 小格）
    print("\n### D6-HP-a HAM seed-ensemble worst-group 样本级配对 bootstrap（canonical 14 子群，HN − 基线，2000 次）")
    print("| HN \\ 基线 | vs ERM | vs SWAD | vs ROC |")
    print("|-----------|--------|---------|--------|")
    for hn in HN:
        cells = []
        for base in BASE:
            d = paired_bootstrap_diff(n, canon_worst(ens[hn]), canon_worst(ens[base]), n_boot=2000, rng=rng)
            cells.append(f"{d.point:+.3f} [{d.ci_low:+.3f},{d.ci_high:+.3f}] "
                         f"{'**显著**' if d.significant else 'n.s.'}")
        print(f"| {METHOD_LABEL[hn]} | {cells[0]} | {cells[1]} | {cells[2]} |")

    # (b) 边缘 worst-group（隔离 joint 噪声）
    print("\n### D6-HP-b HAM seed-ensemble 边缘 worst-group 样本级配对 bootstrap（仅 sex/age 6 组，HN − ERM）")
    print("| 对比 | Δ 边缘worst | 95% CI | p | 判定 |")
    print("|------|-------------|--------|---|------|")
    for hn in HN:
        d = paired_bootstrap_diff(n, marg_worst(ens[hn]), marg_worst(ens["erm"]), n_boot=2000, rng=rng)
        print(f"| {METHOD_LABEL[hn]} vs ERM | {d.point:+.4f} | "
              f"[{d.ci_low:+.4f}, {d.ci_high:+.4f}] | {d.p_value:.3f} | "
              f"{'**显著**' if d.significant else 'n.s.'} |")

    # (c) Overall AUC DeLong on ensemble
    print("\n### D6-HP-c HAM seed-ensemble Overall AUC DeLong（HN vs 基线，集成分数上 test 采样口径）")
    print("| HN \\ 基线 | vs ERM | vs SWAD | vs ROC |")
    print("|-----------|--------|---------|--------|")
    for hn in HN:
        cells = []
        for base in BASE:
            r = delong_test(y_true, ens[hn], ens[base])
            cells.append(f"Δ={r.diff:+.4f} (p={r.p_value:.4f}{', **显著**' if r.significant else ''})")
        print(f"| {METHOD_LABEL[hn]} | {cells[0]} | {cells[1]} | {cells[2]} |")


# 局部类型别名（避免从 significance 导 MetricFn；仅注解用）
MetricFnLocal = "callable"


def _selftest_canonical() -> None:
    """
    R0.1 回归测试：验证 canonical worst-group 口径守卫。

    正向：每数据集用随机小样本跑 _metrics_one，断言通过（canonical 向量维度对、含 joint）。
    负向：手工构造「边缘-only」向量（删掉 joint 键），断言 _assert_canonical_vector 会 fire——
          证明守卫确实能抓住「退回边缘-only」的口径漂移（防 C6 vs D4 并存复发）。
    """
    from collections import OrderedDict
    rng = np.random.default_rng(0)
    # 各数据集所需属性的取值域（与 subgroup_auc schema 一致）
    attr_domains = {
        "mimic": {"sex": 2, "race": 2, "age": 2},
        "chexpert": {"sex": 2, "race": 2, "age": 2},
        "ham10000": {"sex": 2, "age": 4},
        "fitzpatrick": {"skin": 6},
        "papila": {"sex": 2, "age": 2},
    }
    n = 400
    for key, doms in attr_domains.items():
        y = rng.integers(0, 2, n)
        s = y * 0.7 + rng.standard_normal(n)   # 有信号，保证多数格可算 AUC
        attrs = {a: rng.integers(0, d, n) for a, d in doms.items()}
        ov, w, g = _metrics_one(key, y, s, attrs)   # 正向：不得 raise
        assert 0.0 <= ov <= 1.0 and (w is None or 0.0 <= w <= 1.0), (key, ov, w)

    # 负向：边缘-only 向量（无 joint 键）对 joint-required 数据集必须 fire
    marginal_only = OrderedDict([("sex:Male", (0.9, 100)), ("sex:Female", (0.8, 100)),
                                 ("age:<60", (0.85, 100)), ("age:>=60", (0.82, 100))])
    for key in ("mimic", "ham10000", "papila"):
        try:
            _assert_canonical_vector(key, marginal_only)
            raise AssertionError(f"[R0.1] {key} 边缘-only 向量未被守卫拦截（守卫失效！）")
        except AssertionError as e:
            assert "R0.1" in str(e), f"意外的 AssertionError：{e}"
    # fitzpatrick 单属性豁免 joint 要求，但维度守卫仍在（6 维）；给错维度应 fire
    try:
        _assert_canonical_vector("fitzpatrick", marginal_only)  # 4≠6
        raise AssertionError("[R0.1] fitzpatrick 维度守卫未 fire")
    except AssertionError as e:
        assert "R0.1" in str(e), e

    print("R0.1 canonical worst-group 口径守卫 self-test 全部通过 ✓"
          "（5 数据集正向走 canonical + 边缘-only/错维度被拦截）")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--section", choices=("id", "ood", "sig", "selftest", "all"), default="all")
    p.add_argument("--selection", choices=("overall", "worstcase"), default="overall",
                   help="HN/ERM checkpoint 选择口径（ROC/SWAD 恒 overall）。")
    args = p.parse_args()
    global SELECTION
    SELECTION = args.selection
    print(f"[selection={SELECTION}]")
    if args.section == "selftest":
        _selftest_canonical()
        return
    if args.section in ("id", "all"):
        print("\n========== D3/D5 ID 性能与公平表 ==========")
        id_tables()
    if args.section in ("ood", "all"):
        print("\n========== D4 OOD 表 ==========")
        ood_table()
    if args.section in ("sig", "all"):
        print("\n========== D6 HAM 显著性 ==========")
        ham_significance()
        print("\n========== D6-HP HAM 更高功效显著性（R4.1 seed-ensemble）==========")
        ham_significance_highpower()


if __name__ == "__main__":
    main()
