"""
实验 P 分析：Predicted-Attribute HyperAdapt 的 CV-OOF averaging 读数
===================================================================
方案 `docs/predicted_attribute_hyperadapt_plan.md` §2/§5。口径与主线完全一致：

  · **averaging**（逐折算指标 → 折间平均 → 再 min/max 取组），不做 pooled（已知负偏）；
  · 配对 cluster bootstrap（Fitzpatrick 每 md5hash 唯一 ⇒ 退化为样本级，正确）；
  · 分组键恒用**真值**敏感属性（p̂ 只作模型输入）。

组件直接复用 `scripts/build_oof_results_averaging.py`（Spec/Views/load_folds/clusters），
但方法表与 tag 映射在本脚本内独立给出——**不写入共享的 selected_configs.json**，
避免污染既有分析脚本的方法遍历。

预注册的对比（见方案 §2）：
  P-H1 非劣：hyperadapt_pred  vs hyperadapt(GT)
  P-H2 有效：hyperadapt_pred  vs erm
  P-H3 路由 vs 容量：hyperadapt_pred vs hyperadapt_predperm / hyperadapt_predconst
  消融    ：hyperadapt_predhard vs hyperadapt_pred

运行（轻量 CPU）：
    python scripts/p_pred_attr_analysis.py --dataset fitzpatrick
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.build_oof_results_averaging import (
    N_BOOT, Spec, Views, clusters, load_folds,
)

OUTPUTS = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs")

# 实验 P 的四个臂 + 三个参照（GT-HyperAdapt / ERM / SWAD）
PRED_ARMS = ("hyperadapt_pred", "hyperadapt_predhard", "hyperadapt_predperm", "hyperadapt_predconst")
# groupdro = 训练时用**真值**子群标签做重加权的第 4 基线（见 docs/groupdro_baseline_5datasets.md）。
# 与 pred 臂构成「真值重加权 vs 预测条件化」的对照。
REFERENCE = ("erm", "swad", "hyperadapt", "groupdro")

# ---- HAM 双属性（sex+age）条件化版（`--cond sex_age`，仅 ham10000）----
# 四个 pred 臂改用 sex+age 两份 p̂ 条件化（method 加 `_sexage` 后缀），GT 参照相应换成
# `hyperadapt_sexage`（GT 双属性臂）。age-only 的 pred 臂进入参照集，构成本臂最核心的
# 跨臂对比：**只差条件输入是否含 sex**（两臂同 config、同 split/seed、同 p̂ 的 age 分量）。
SEXAGE_SUFFIX = "_sexage"
PRED_ARMS_SEXAGE = tuple(a + SEXAGE_SUFFIX for a in PRED_ARMS)
REFERENCE_SEXAGE = ("erm", "swad", "hyperadapt_sexage", "groupdro",
                    "hyperadapt", "hyperadapt_pred", "hyperadapt_predconst")

# 预注册对比：(实验臂, 参照臂)
CONTRASTS = (
    ("hyperadapt_pred", "hyperadapt"),        # P-H1 非劣（属性标注是否可省）
    ("hyperadapt_pred", "erm"),               # P-H2 软路由是否有效
    ("hyperadapt_pred", "swad"),              # 与最强 ID 基线的对照
    ("hyperadapt_pred", "hyperadapt_predperm"),   # P-H3 路由信息 vs 容量
    ("hyperadapt_pred", "hyperadapt_predconst"),  # P-H3 容量上界
    ("hyperadapt_predhard", "hyperadapt_pred"),   # 软 vs 硬消融
    # 解释性对比：GT 臂与两个「无样本级信息」的臂各自对 ERM 的增量。
    # 若 const/perm 对 ERM 的增量与 GT 臂相当，则 HyperAdapt 的 worst-group 增益可由
    # 「多余容量 + 训练噪声」解释，与属性信息无关。
    ("hyperadapt", "erm"),
    ("hyperadapt_predconst", "erm"),
    ("hyperadapt_predperm", "erm"),
    # 事后追加（**不进 Holm family**，family 恒为前 N_PREREGISTERED 个）：
    # 与 GroupDRO 的对照——两种「用属性」的方式：真值重加权 vs 预测条件化。
    ("hyperadapt_pred", "groupdro"),
    ("hyperadapt", "groupdro"),
    ("groupdro", "erm"),
)

# `--cond sex_age` 的对比表：与 age-only 版逐项同构（前 6 个仍是预注册 family），
# 只把 pred 臂换成 `_sexage`、GT 参照换成 `hyperadapt_sexage`；末尾追加两个跨臂对比。
CONTRASTS_SEXAGE = (
    ("hyperadapt_pred_sexage", "hyperadapt_sexage"),        # P-H1 非劣（对 GT 双属性臂）
    ("hyperadapt_pred_sexage", "erm"),                      # P-H2 软路由是否有效
    ("hyperadapt_pred_sexage", "swad"),                     # 与最强 ID 基线的对照
    ("hyperadapt_pred_sexage", "hyperadapt_predperm_sexage"),   # P-H3 路由信息 vs 容量
    ("hyperadapt_pred_sexage", "hyperadapt_predconst_sexage"),  # P-H3 容量上界
    ("hyperadapt_predhard_sexage", "hyperadapt_pred_sexage"),   # 软 vs 硬消融
    # 解释性对比（不进 Holm family）
    ("hyperadapt_sexage", "erm"),
    ("hyperadapt_predconst_sexage", "erm"),
    ("hyperadapt_predperm_sexage", "erm"),
    ("hyperadapt_pred_sexage", "groupdro"),
    ("hyperadapt_sexage", "groupdro"),
    ("groupdro", "erm"),
    # 跨臂：条件输入加不加 sex（两臂 config-matched，唯一差异就是 sex 的 p̂）
    ("hyperadapt_pred_sexage", "hyperadapt_pred"),
    ("hyperadapt_predconst_sexage", "hyperadapt_predconst"),
)


def arms_and_contrasts(cond: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple, str]:
    """
    按条件通路给出 (pred 臂, 参照臂, 对比表, 结果文件名)。

    Args:
        cond: "age"（历史臂）或 "sex_age"（HAM 双属性臂）。

    Returns:
        (PRED_ARMS, REFERENCE, CONTRASTS, 输出 json 文件名)。
    """
    if cond == "age":
        return PRED_ARMS, REFERENCE, CONTRASTS, "p_pred_attr_results.json"
    return (PRED_ARMS_SEXAGE, REFERENCE_SEXAGE, CONTRASTS_SEXAGE,
            "p_pred_attr_sexage_results.json")

# 与 build_oof_results_averaging.SPECS 的对应条目逐字段一致（同一寻址与口径）。
# 注意 PAPILA：CV 产物不放 cv5 子目录（output_dir 不随 cv 变，见 train_papila_*）。
DATASET_SPECS: dict[str, Spec] = {
    "fitzpatrick": Spec("Fitzpatrick", "fitzpatrick/cv5/predictions", "fitzpatrick",
                        "fitzpatrick/cv5", None, None, False),
    "ham10000": Spec("HAM10000", "ham10000/cv5/predictions", "ham10000",
                     "ham10000/cv5", "ham10000", "lesion_id", True),
    "mimic": Spec("MIMIC", "mimic_cxr/cv5/predictions", "mimic",
                  "mimic_cxr/cv5", "mimic_cxr_nofinding", "patient_id", False),
    "chexpert": Spec("CheXpert", "chexpert_cxr/cv5/predictions", "chexpert",
                     "chexpert_cxr/cv5", "chexpert_nofinding", "patient_id", False),
    "papila": Spec("PAPILA", "papila/predictions", "papila",
                   "papila", "papila", "pid", False),
}
METRIC_KEYS = ("overall", "canonical_worst", "marginal_worst", "gap")


# 预注册的**主要终点**：worst-group（canonical）。多重校正的 family 限定为
# CONTRASTS 中的前 6 个预注册对比（后 3 个是解释性对比，不进 family）。
PRIMARY_METRIC = "canonical_worst"
N_PREREGISTERED = 6


def bootstrap_p_value(diff: np.ndarray) -> float:
    """
    配对 bootstrap 的双侧 p 值：p = 2·min(P(d≤0), P(d≥0))，下限截到 1/n_boot。

    Args:
        diff: [n_boot] 配对差的 bootstrap 分布。

    Returns:
        双侧 p 值 ∈ (0, 1]。
    """
    n = len(diff)
    p = 2.0 * min(float((diff <= 0).mean()), float((diff >= 0).mean()))
    return float(min(1.0, max(1.0 / n, p)))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """
    Holm–Bonferroni 逐步降序校正（family-wise error rate 控制）。

    Args:
        p_values: {对比名 -> 原始 p 值}。

    Returns:
        {对比名 -> 校正后 p 值}（单调化后截到 1.0）。
    """
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(items):
        running = max(running, (m - rank) * p)      # 单调化
        adjusted[name] = float(min(1.0, running))
    return adjusted


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    ap = argparse.ArgumentParser(description="实验 P（Predicted-Attribute HyperAdapt）结果分析")
    ap.add_argument("--dataset", default="fitzpatrick", choices=sorted(DATASET_SPECS))
    ap.add_argument("--n_boot", type=int, default=N_BOOT, help="配对 bootstrap 重采样次数。")
    ap.add_argument("--cond", default="age", choices=("age", "sex_age"),
                    help="条件通路：age=历史臂（p̂ 只含 age）；sex_age=HAM 双属性臂（p̂ 含 sex+age，"
                         "method 带 _sexage 后缀），结果写 p_pred_attr_sexage_results.json。")
    ap.add_argument("--gt-config-matched", action="store_true",
                    help="仅 --cond sex_age：把 GT 参照 hyperadapt_sexage 的 config 换成与 pred 臂"
                         "相同的 lr3e-05_wd1e-04（GT 臂自选的是 lr3e-05_wd1e-03），"
                         "做 P-H1 的**选参去混淆**敏感性。")
    return ap.parse_args()


def resolve_tags(spec: Spec, dataset: str, cond: str = "age",
                 gt_config_matched: bool = False) -> dict[str, str]:
    """
    给出方法 -> config_tag 的映射。

    参照臂的 tag 取自该数据集既有的 selected_configs.json；实验 P 的四个臂**复用 GT-HyperAdapt
    的选定配置**（方案 §5：保证「仅属性来源」的单变量差异、且算力中性）。

    `cond="sex_age"`（HAM 双属性臂）沿用**同一个** `hyperadapt` 的 config（lr3e-05_wd1e-04），
    使 sex+age pred 臂与 age-only pred 臂天然 config-matched，跨臂 Δ 的唯一来源是条件输入。
    GT 参照 `hyperadapt_sexage` 默认用它自己的选定配置（协议口径）；`gt_config_matched=True`
    时改用与 pred 臂相同的 config，做 P-H1 的选参去混淆敏感性。

    Args:
        spec             : 该数据集的 Spec。
        dataset          : 数据集名。
        cond             : "age" 或 "sex_age"。
        gt_config_matched: 见上（只影响 hyperadapt_sexage）。

    Returns:
        {method -> config_tag}。

    Raises:
        KeyError: selected_configs.json 缺 hyperadapt。
    """
    arms, reference, _, _ = arms_and_contrasts(cond)
    cfg = json.loads(spec.cfg.read_text())["config"]
    tags = {m: cfg[m] for m in reference if m in cfg}
    if "hyperadapt" not in cfg:
        raise KeyError(f"{dataset}: selected_configs.json 缺 hyperadapt，无法确定实验 P 的配置")
    for arm in arms:
        tags[arm] = cfg["hyperadapt"]
    if cond == "sex_age":
        # age-only pred 臂（跨臂对比的参照）同样跟随 hyperadapt 的 config
        for arm in PRED_ARMS:
            if arm in reference:
                tags[arm] = cfg["hyperadapt"]
        if gt_config_matched:
            tags["hyperadapt_sexage"] = cfg["hyperadapt"]
    return tags


def main() -> None:
    args = parse_args()
    spec = DATASET_SPECS[args.dataset]
    if args.cond == "sex_age" and args.dataset != "ham10000":
        raise ValueError("--cond sex_age 只有 HAM10000 有双属性对照臂")
    pred_arms, reference, contrast_pairs, out_name = arms_and_contrasts(args.cond)
    tags = resolve_tags(spec, args.dataset, args.cond, args.gt_config_matched)

    views: dict[str, Views] = {}
    folds_by: dict[str, list[dict]] = {}
    for method, tag in tags.items():
        folds = load_folds(spec, method, tag)
        if folds is None:
            print(f"[跳过] {method}（{tag}）：5 折预测不完整")
            continue
        folds_by[method] = folds
        views[method] = Views(spec, folds)
    missing = [m for m in pred_arms if m not in views]
    if missing:
        print(f"⚠️ 缺失实验臂：{missing}")

    cidx, offs = clusters(spec, folds_by["erm"])
    n_clusters = int(cidx.max() + 1)
    ones = np.ones(offs[-1])
    obs = {m: views[m].compute(offs, ones) for m in views}

    # 配对 cluster bootstrap：全方法共用同一批重采样索引（配对）
    rng = np.random.default_rng(42)
    boot: dict[str, list[tuple]] = {m: [] for m in views}
    for _ in range(args.n_boot):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        weights = np.bincount(draw, minlength=n_clusters).astype(float)[cidx]
        for m in views:
            boot[m].append(views[m].compute(offs, weights))
    boot_arr = {m: np.array([r[:4] for r in boot[m]]) for m in views}

    print(f"\n{'=' * 96}")
    print(f"实验 P · {spec.name} · cond={args.cond} · CV-OOF averaging（n={offs[-1]:,}，"
          f"cluster={n_clusters:,}，bootstrap={args.n_boot}）"
          + ("  [GT 参照 config-matched]" if args.gt_config_matched else ""))
    print(f"{'=' * 96}")
    print(f"{'方法':<26}{'Overall':>10}{'worst(canon)':>15}{'worst(marg)':>14}{'gap':>10}{'无定义格':>10}")
    for m in list(reference) + list(pred_arms):
        if m not in obs:
            continue
        o = obs[m]
        print(f"{m:<26}{o[0]:>10.4f}{o[1]:>15.4f}{o[2]:>14.4f}{o[3]:>10.4f}{o[4]:>10d}")

    print(f"\n{'-' * 96}\n配对对比（Δ = 实验臂 − 参照臂，95% CI 来自配对 cluster bootstrap）\n{'-' * 96}")
    contrasts: dict[str, dict] = {}
    for arm, base in contrast_pairs:
        if arm not in views or base not in views:
            continue
        entry: dict[str, dict] = {}
        print(f"\n{arm}  vs  {base}")
        for j, key in enumerate(METRIC_KEYS):
            diff = boot_arr[arm][:, j] - boot_arr[base][:, j]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            delta = obs[arm][j] - obs[base][j]
            sig = bool(lo > 0 or hi < 0)
            p_raw = bootstrap_p_value(diff)
            entry[key] = {"delta": float(delta), "ci": [float(lo), float(hi)],
                          "sig": sig, "p_raw": p_raw}
            print(f"  {key:<18} Δ={delta:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]  "
                  f"p={p_raw:.3f}  {'显著' if sig else 'n.s.'}")
        contrasts[f"{arm}_vs_{base}"] = entry

    # 输出与该 regime 的 selected_configs.json 同目录（PAPILA 无 cv5 子目录，故不能硬编码 cv5）
    # ---- 主要终点的多重校正（Holm，family = 前 N_PREREGISTERED 个预注册对比）----
    family = {}
    for arm, base in contrast_pairs[:N_PREREGISTERED]:
        key = f"{arm}_vs_{base}"
        if key in contrasts:
            family[key] = contrasts[key][PRIMARY_METRIC]["p_raw"]
    holm = holm_adjust(family) if family else {}
    if holm:
        print(f"\n{'-' * 96}\n主要终点（{PRIMARY_METRIC}）的 Holm 校正"
              f"（family = {len(family)} 个预注册对比）\n{'-' * 96}")
        for key in family:
            print(f"  {key:<50} p_raw={family[key]:.3f}  p_holm={holm[key]:.3f}  "
                  f"{'显著' if holm[key] < 0.05 else 'n.s.'}")
            contrasts[key][PRIMARY_METRIC]["p_holm"] = holm[key]

    # config-matched 敏感性另存，不覆盖协议口径的主结果
    if args.gt_config_matched:
        out_name = out_name.replace(".json", "_gtmatched.json")
    out_path = spec.cfg.parent / out_name
    payload = {
        "dataset": args.dataset, "cond": args.cond,
        "gt_config_matched": bool(args.gt_config_matched),
        "n": int(offs[-1]), "n_clusters": n_clusters,
        "n_boot": args.n_boot, "config": tags,
        "point": {m: dict(zip(METRIC_KEYS, [float(x) for x in obs[m][:4]]),
                          undefined_fold_cells=int(obs[m][4])) for m in obs},
        "contrasts": contrasts,
        "holm_primary": {"metric": PRIMARY_METRIC, "family_size": len(family), "p_holm": holm},
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n结果已写入 {out_path}")


if __name__ == "__main__":
    main()
