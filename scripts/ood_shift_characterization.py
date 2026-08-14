"""
OOD 偏移刻画（MIMIC ↔ CheXpert）—— v2 实验 D3 的第一部分
==========================================================
`docs/ood_experiment_v2_preregistration.md` §6 · D3。**在谈"鲁棒性"之前先刻画偏移的类型与强度**
（Moreno-Torres et al. 2012 的 dataset shift 类型学；WILDS, Koh et al. ICML 2021 的子群报告规范）。
不这样做，"某方法 OOD 更好"就无法归因——不知道它到底在抵抗哪一种偏移。

本脚本只做**纯统计、零 GPU** 的三项（covariate shift 需图像特征，另见
`ood_domain_discriminability.py`）：

  1. **Prior shift**   ΔP(Y)：两库整体患病率之差。本对最突出的偏移——MIMIC ~30% vs CheXpert ~6–8%。
  2. **属性构成偏移**  ΔP(A)：各 marginal 子群（sex / race / age）的人群占比之差。
  3. **条件患病率偏移** ΔP(Y|A)：各子群内患病率之差；**并检查子群间的排序是否翻转**——
     若某子群在 source 中患病率高于总体、在 target 中反而低于总体，即 **concept shift 的迹象**
     （P(Y|A) 的方向改变，而非仅整体平移）。

**为何这对本项目要紧**：AUC 对患病率不敏感，故 prior shift 不会直接反映在 §1.6/§1.7 的数字上，
却会让**校准**崩塌（见 D2）。而 ΔP(Y|A) 的方向翻转若存在，则属性—标签关系跨库改变，正是
"属性条件化模型（HN）在 OOD 下可能有用"的唯一机制性理由；若不存在，HN 的 OOD 增益就缺少机制解释。

口径：子群 schema 取自 `harness.subgroup_auc.subgroup_masks`（单一事实来源）；总体 = 5 折 test 并集
（= 全数据集，与评估集一致）。

运行（轻量 CPU）：
    python scripts/ood_shift_characterization.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from src.training.harness.subgroup_auc import subgroup_masks

# 中文字体（.ttc 只注册首个 face，family 名为 "Noto Sans CJK JP"）
for _p, _n in (("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "Noto Sans CJK JP"),
               ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", "Droid Sans Fallback")):
    if Path(_p).exists():
        font_manager.fontManager.addfont(_p)
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REPO = Path("/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP")
SPLITS = REPO / "data" / "splits"
OUT_DIR = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/ood_cxr/shift_characterization")
N_FOLDS = 5

# (数据集键, split 子目录, 显示名)
DATASETS = (("mimic", "mimic_cxr_nofinding", "MIMIC-CXR"),
            ("chexpert", "chexpert_nofinding", "CheXpert"))
# dataviz 分类槽位 1/2
COLORS = {"MIMIC-CXR": "#2a78d6", "CheXpert": "#eb6834"}
INK, MUTED = "#0b0b0b", "#52514e"


def load_full(split_sub: str) -> pd.DataFrame:
    """载入某数据集 5 折 test 的并集（= 全数据集，与 OOD 评估集同源）。"""
    parts = [pd.read_csv(SPLITS / split_sub / "cv5" / f"fold{k}" / "test.csv")
             for k in range(N_FOLDS)]
    df = pd.concat(parts, ignore_index=True)
    assert df["patient_id"].nunique() == df.groupby("patient_id").ngroups
    return df


def age_bins(age: np.ndarray) -> np.ndarray:
    """CXR 口径的在线 age 分箱：<60 → 0，>=60 → 1（与 subgroup_masks 一致）。"""
    return (age >= 60).astype(int)


def marginal_stats(df: pd.DataFrame, ds_key: str) -> dict[str, dict[str, float]]:
    """
    算某数据集的整体患病率与各 marginal 子群的 (占比, 组内患病率)。

    Args:
        df: split CSV（含 label/sex/race/age 列）。
        ds_key: subgroup_masks 的数据集键（"mimic" / "chexpert"）。

    Returns:
        {子群名: {"share": 人群占比, "prev": 组内患病率, "n": 样本数}}，
        另含键 "__all__" 承载整体统计。
    """
    y = df["label"].values.astype(int)
    masks = subgroup_masks(ds_key, sex=df["sex"].values.astype(int),
                           race=df["race"].values.astype(int),
                           age=age_bins(df["age"].values))
    out = {"__all__": {"share": 1.0, "prev": float(y.mean()), "n": int(len(y))}}
    for g, m in masks.items():
        if "|" in g:                       # 只看 marginal（joint 小格噪声大，且非本诊断重点）
            continue
        out[g] = {"share": float(m.mean()), "prev": float(y[m].mean()), "n": int(m.sum())}
    return out


def build_table(stats: dict[str, dict]) -> pd.DataFrame:
    """把两库统计并成对照表，附 Δ 与「相对总体的方向」以便查 concept shift。"""
    (ka, na), (kb, nb) = [(k, n) for k, _, n in DATASETS][:1] + [(k, n) for k, _, n in DATASETS][1:]
    a, b = stats[na], stats[nb]
    rows = []
    for g in a:
        if g == "__all__":
            continue
        # 「相对总体」= 组内患病率 − 该库整体患病率；符号翻转 ⇒ concept shift 迹象
        rel_a = a[g]["prev"] - a["__all__"]["prev"]
        rel_b = b[g]["prev"] - b["__all__"]["prev"]
        rows.append({
            "子群": g,
            f"{na} 占比": a[g]["share"], f"{nb} 占比": b[g]["share"],
            "Δ占比": b[g]["share"] - a[g]["share"],
            f"{na} 患病率": a[g]["prev"], f"{nb} 患病率": b[g]["prev"],
            "Δ患病率": b[g]["prev"] - a[g]["prev"],
            f"{na} 相对总体": rel_a, f"{nb} 相对总体": rel_b,
            "方向翻转": bool(np.sign(rel_a) != np.sign(rel_b)),
        })
    return pd.DataFrame(rows)


def plot_shift(stats: dict[str, dict], table: pd.DataFrame, path: Path) -> None:
    """三面板：整体患病率、逐子群人群占比、逐子群组内患病率（均为两库并置对照）。"""
    names = [n for _, _, n in DATASETS]
    groups = table["子群"].tolist()
    y = np.arange(len(groups))
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 0.44 * len(groups) + 3.2),
                             gridspec_kw={"width_ratios": [0.6, 1.0, 1.0]})

    # --- 面板 1：整体患病率（prior shift 的核心一眼图） ---
    ax = axes[0]
    prevs = [stats[n]["__all__"]["prev"] for n in names]
    ax.bar(names, prevs, color=[COLORS[n] for n in names], width=0.55,
           edgecolor="white", linewidth=0.8)
    for i, v in enumerate(prevs):
        ax.annotate(f"{v:.1%}", (i, v), ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_ylabel("整体患病率 P(Y=1)", fontsize=9, color=MUTED)
    ax.set_title(f"Prior shift：ΔP(Y) = {prevs[1] - prevs[0]:+.1%}", fontsize=10, color=INK)
    ax.set_ylim(0, max(prevs) * 1.28)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)

    # --- 面板 2/3：逐子群占比与组内患病率 ---
    for ax, col_tpl, title in ((axes[1], "{} 占比", "属性构成偏移 ΔP(A)"),
                               (axes[2], "{} 患病率", "条件患病率偏移 ΔP(Y|A)")):
        for i, n in enumerate(names):
            ax.scatter(table[col_tpl.format(n)], y + (0.15 if i else -0.15), s=44,
                       color=COLORS[n], label=n, zorder=3, edgecolors="white", linewidths=1.1)
        # 连线强调同一子群的两库落差
        for j in range(len(groups)):
            xa = table[col_tpl.format(names[0])].iloc[j]
            xb = table[col_tpl.format(names[1])].iloc[j]
            ax.plot([xa, xb], [y[j] - 0.15, y[j] + 0.15], color="#c9c8c2", linewidth=1.4, zorder=2)
        ax.set_yticks(y, groups, fontsize=8.5)
        ax.set_title(title, fontsize=10, color=INK)
        ax.legend(fontsize=8.5, frameon=False, loc="lower right")
        ax.grid(axis="x", color="#e6e5e1", linewidth=0.8)
        ax.set_axisbelow(True)
    axes[2].axvline(stats[names[0]]["__all__"]["prev"], color=COLORS[names[0]],
                    linestyle="--", linewidth=1.0, alpha=0.7)
    axes[2].axvline(stats[names[1]]["__all__"]["prev"], color=COLORS[names[1]],
                    linestyle="--", linewidth=1.0, alpha=0.7)

    for a in axes:
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.tick_params(colors=MUTED, labelsize=8.5)
    fig.suptitle("MIMIC ↔ CheXpert 偏移刻画（虚线 = 各库整体患病率）", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="MIMIC↔CheXpert 偏移刻画（prior / 构成 / 条件患病率）")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    for ds_key, split_sub, name in DATASETS:
        df = load_full(split_sub)
        stats[name] = marginal_stats(df, ds_key)
        print(f"{name}: n={stats[name]['__all__']['n']:,}  "
              f"整体患病率={stats[name]['__all__']['prev']:.4f}")

    table = build_table(stats)
    na, nb = [n for _, _, n in DATASETS]
    pa, pb = stats[na]["__all__"]["prev"], stats[nb]["__all__"]["prev"]
    print(f"\n{'=' * 100}\nPrior shift: P(Y) {pa:.4f} → {pb:.4f}  (Δ={pb - pa:+.4f}, "
          f"比值={pb / pa:.2f}×)\n{'=' * 100}")
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", lambda v: f"{v:+.4f}"):
        print(table.to_string(index=False))

    flips = table[table["方向翻转"]]["子群"].tolist()
    print(f"\nconcept shift 迹象（组内患病率相对总体的方向翻转）："
          f"{flips if flips else '无 —— 各子群相对总体的方向在两库一致'}")

    plot_shift(stats, table, args.out_dir / "shift_characterization.png")
    payload = {"prior_shift": {"source": pa, "target": pb, "delta": pb - pa, "ratio": pb / pa},
               "per_dataset": stats, "table": table.to_dict(orient="records"),
               "direction_flips": flips}
    (args.out_dir / "shift_characterization.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n图 → {args.out_dir / 'shift_characterization.png'}")
    print(f"已落盘 → {args.out_dir / 'shift_characterization.json'}")


if __name__ == "__main__":
    main()
