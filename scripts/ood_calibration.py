"""
OOD 校准评估（MIMIC ↔ CheXpert）—— v2 实验 D2 / 假设 H3
=========================================================
`docs/ood_experiment_v2_preregistration.md` §6 · D2、§4 · H3。

**为什么必须做**：TRIPOD+AI（Collins et al., BMJ 2024）与 CLAIM 都要求预测模型的**外部验证必报
校准**，而非仅报判别（AUC）。本项目现行 OOD 结论只报 AUC。而 `ood_shift_characterization.py`
已实测：MIMIC 患病率 30.2% vs CheXpert 8.2%（**3.7 倍的 prior shift**）——AUC 对患病率不敏感，
**校准却会系统性崩塌**。只报 AUC 会让"模型跨库迁移良好"的结论严重失真。

**四个指标**（整体 + 逐 marginal 子群）：
  · **CITL**（calibration-in-the-large）= mean(p̂) − mean(y)：整体偏移量，prior shift 的直接后果。
  · **calibration slope** = logit(y) ~ logit(p̂) 的回归斜率；1 = 完美，<1 = 过度自信。
  · **ECE**（expected calibration error，等频分箱）与 **Brier**。

**原样 vs 重校准**：同时报两套。重校准用 **isotonic 回归 + 评估集内 5 折交叉拟合**——即在 target 上
分 5 折，用 4 折拟合校准映射、predict 剩下 1 折，**每个样本的校准分数都来自未见过它的映射**，故
不泄漏、也**不需要额外划 val split**。

**H3 的两条判据**：
  1. HyperAdapt 的 OOD 校准不劣于 ERM；
  2. **判别优势（若有）在重校准后仍存在**——isotonic 是单调映射，理论上不改变 AUC，故此处的作用是
     检验"HyperAdapt 的优势是否只是分数刻度差异"。若重校准后 marginal-worst 的 Δ 显著缩小，
     说明原优势相当部分来自校准而非判别。

口径：averaging（逐折算 → 折间平均），与主结果同源；子群 schema 取自 `harness.subgroup_auc`。

运行（轻量 CPU，无需 GPU）：
    python scripts/ood_calibration.py
    python scripts/ood_calibration.py --direction M2C
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from scripts.eaudit_m2c_full_target import sorted_view, weighted_auc
from src.training.harness.predictions import load_predictions
from src.training.harness.subgroup_auc import subgroup_masks

for _p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"):
    if Path(_p).exists():
        font_manager.fontManager.addfont(_p)
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUTS = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs")
OUT_DIR = OUTPUTS / "ood_cxr" / "calibration"
FOLD_SEEDS = (42, 43, 44, 45, 46)
N_BINS = 10
RECAL_FOLDS = 5

DIRECTIONS = {
    "M2C": {"cfg_dir": "mimic_cxr", "ood_sub": "mimic2chexpert", "tgt_key": "chexpert",
            "title": "MIMIC → CheXpert", "src_prev": 0.3020, "tgt_prev": 0.0824},
    "C2M": {"cfg_dir": "chexpert_cxr", "ood_sub": "chexpert2mimic", "tgt_key": "mimic",
            "title": "CheXpert → MIMIC", "src_prev": 0.0824, "tgt_prev": 0.3020},
}
# 追加实验 G 的融合臂（HyperAdapt+SWAD）与 GroupDRO 基线——追加不插入，既有方法顺序/配色不变。
# OOD v2 偏离 #4：再追加 HyperHead / HyperFusion 两臂（恢复条件化深度轴）——同样是**追加不插入**，
# 既有方法顺序/配色逐位不变。
METHODS = ("erm", "swad", "hyperadapt", "hyperadapt_swad", "groupdro",
           "hyperhead", "hyperfusion")
COLORS = {"erm": "#2a78d6", "hyperadapt": "#eb6834", "swad": "#1baf7a",
          "hyperadapt_swad": "#8b53c9", "groupdro": "#c2185b",
          "hyperhead": "#d4a017", "hyperfusion": "#00838f"}
LABELS = {"erm": "ERM", "swad": "SWAD", "hyperadapt": "HyperAdapt",
          "hyperadapt_swad": "HyperAdapt+SWAD", "groupdro": "GroupDRO",
          "hyperhead": "HyperHead", "hyperfusion": "HyperFusion"}
INK, MUTED = "#0b0b0b", "#52514e"


# calibration slope 的合理范围；超出即视为拟合退化（如 logit(p̂) 方差过小），报 nan 而非天文数字
SLOPE_SANE_MAX = 20.0


def sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 logistic 函数（scipy.expit，两分支均不溢出）。"""
    return expit(np.asarray(z, dtype=np.float64))


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """
    算四个校准指标。

    Args:
        y: [N] 0/1 真实标签。
        p: [N] 预测概率 ∈ (0,1)。

    Returns:
        {"citl", "slope", "ece", "brier"}；slope 在类别单一时为 nan。
    """
    p = np.clip(p, 1e-6, 1 - 1e-6)
    citl = float(p.mean() - y.mean())
    brier = float(np.mean((p - y) ** 2))

    # calibration slope：y ~ logit(p̂) 的一元 logistic 回归斜率。
    # 用 sklearn 的 lbfgs（无正则）而非手写牛顿法——后者在 logit(p̂) 方差极小时会发散到 1e8 量级。
    x = np.log(p / (1 - p))
    if len(np.unique(y)) < 2 or np.std(x) < 1e-8:
        slope = float("nan")
    else:
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        lr.fit(x.reshape(-1, 1), y)
        slope = float(lr.coef_[0, 0])
        if not np.isfinite(slope) or abs(slope) > SLOPE_SANE_MAX:
            slope = float("nan")                       # 拟合退化，不报假数字

    # ECE：等频分箱（对极端偏斜的患病率比等宽分箱稳）
    order = np.argsort(p)
    ece = 0.0
    for chunk in np.array_split(order, N_BINS):
        if len(chunk) == 0:
            continue
        ece += len(chunk) / len(p) * abs(p[chunk].mean() - y[chunk].mean())
    return {"citl": citl, "slope": slope, "ece": float(ece), "brier": brier}


def cross_fit_isotonic(y: np.ndarray, p: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    评估集内 5 折交叉拟合的 isotonic 重校准：每个样本的校准分数都来自未含它的映射。

    Args:
        y: [N] 0/1 标签。
        p: [N] 原始预测概率。
        seed: KFold 打乱种子。

    Returns:
        [N] 重校准后的概率。
    """
    out = np.empty_like(p)
    kf = KFold(n_splits=RECAL_FOLDS, shuffle=True, random_state=seed)
    for tr, te in kf.split(p):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p[tr], y[tr])
        out[te] = iso.predict(p[te])
    return np.clip(out, 1e-6, 1 - 1e-6)


def marginal_worst_auc(y: np.ndarray, s: np.ndarray, attrs: dict, ds_key: str) -> float:
    """marginal 子群上的最差 AUC（与主结果口径一致）。"""
    kw = {k: np.asarray(v).astype(int) for k, v in attrs.items()
          if k in ("sex", "race", "age", "skin")}
    vals = []
    for g, m in subgroup_masks(ds_key, **kw).items():
        if "|" in g:
            continue
        _, ys = sorted_view(y, s, m)
        v = weighted_auc(ys, np.ones(len(ys)))
        if not np.isnan(v):
            vals.append(v)
    return float(min(vals)) if vals else float("nan")


def weighted_calibration(y: np.ndarray, p: np.ndarray, w: np.ndarray,
                         edges: np.ndarray) -> tuple[float, float, float]:
    """
    bootstrap 权重下的 (CITL, ECE, Brier)。分箱边界由**原始**分数预先定死，
    使每次重采样落在同一组箱内（否则箱定义随样本变动，ECE 不可比）。
    """
    sw = w.sum()
    citl = float((w * p).sum() / sw - (w * y).sum() / sw)
    brier = float((w * (p - y) ** 2).sum() / sw)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    ece = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        wb = w[m].sum()
        if wb <= 0:
            continue
        ece += wb / sw * abs((w[m] * p[m]).sum() / wb - (w[m] * y[m]).sum() / wb)
    return citl, float(ece), brier


def bootstrap_calibration_deltas(direction: str, cfg: dict, n_boot: int) -> dict:
    """
    校准指标的**配对患者级 cluster bootstrap**（H3 判定所需的 CI）。

    对 CITL / ECE / Brier 报 Δ(方法 − ERM) 的 95% CI；**slope 不做 CI**
    （加权 logistic 回归在每次重采样上重拟合成本过高，且 slope 已由点估计充分表达）。
    口径 = averaging：逐折算 → 折间平均，与主结果同源。

    Args:
        direction: "M2C" / "C2M"。
        cfg: {方法: config_tag}。
        n_boot: 重采样次数。

    Returns:
        {f"{方法}_vs_erm": {指标: {delta, ci, sig}}}。
    """
    from scripts.build_oof_results_averaging import SPECS, clusters, load_folds

    spec = next(s for s in SPECS if s.name == f"{direction}_OOD")
    folds_by, data = {}, {}
    for m in METHODS:
        fl = load_folds(spec, m, cfg[m])
        if fl is None:
            return {}
        folds_by[m] = fl
        # 逐折缓存 (y, p_raw, 分箱边界)；p_cal 不进 bootstrap（isotonic 已由交叉拟合固定）
        data[m] = []
        for f in fl:
            y = f["y"].astype(np.float64)
            p = sigmoid(f["s"])
            edges = np.unique(np.quantile(p, np.linspace(0, 1, N_BINS + 1)))
            edges[0], edges[-1] = -np.inf, np.inf
            data[m].append((y, p, edges))

    cidx, offs = clusters(spec, folds_by["erm"])
    n_cl = int(cidx.max() + 1)

    def stat(m: str, w_all: np.ndarray) -> np.ndarray:
        """该方法在给定权重下的折均 (CITL, ECE, Brier)。"""
        rows = [weighted_calibration(y, p, w_all[offs[k]:offs[k + 1]], e)
                for k, (y, p, e) in enumerate(data[m])]
        return np.mean(rows, axis=0)

    ones = np.ones(offs[-1])
    obs = {m: stat(m, ones) for m in METHODS}
    rng = np.random.default_rng(42)
    boot = {m: [] for m in METHODS}
    for _ in range(n_boot):
        draw = rng.integers(0, n_cl, size=n_cl)
        w = np.bincount(draw, minlength=n_cl).astype(float)[cidx]
        for m in METHODS:
            boot[m].append(stat(m, w))
    B = {m: np.asarray(boot[m]) for m in METHODS}

    keys = ("citl", "ece", "brier")
    out = {}
    for m in METHODS:
        if m == "erm":
            continue
        e = {}
        for j, k in enumerate(keys):
            d = B[m][:, j] - B["erm"][:, j]
            lo, hi = np.percentile(d, [2.5, 97.5])
            e[k] = {"delta": float(obs[m][j] - obs["erm"][j]),
                    "ci": [float(lo), float(hi)], "sig": bool(lo > 0 or hi < 0)}
        # CITL 是**有符号**量（正=高估、负=低估），其 Δ 的符号不代表优劣；
        # 校准好坏须看 |CITL| 是否更接近 0，故另报 Δ|CITL|（<0 = 校准更好）。
        d_abs = np.abs(B[m][:, 0]) - np.abs(B["erm"][:, 0])
        lo, hi = np.percentile(d_abs, [2.5, 97.5])
        e["abs_citl"] = {"delta": float(abs(obs[m][0]) - abs(obs["erm"][0])),
                         "ci": [float(lo), float(hi)], "sig": bool(lo > 0 or hi < 0)}
        out[f"{m}_vs_erm"] = e
    return out


def run_method(direction: str, method: str, tag: str) -> dict | None:
    """跑一个方法的逐折校准评估（原样 + 重校准），返回折间平均结果。"""
    d = DIRECTIONS[direction]
    pred_dir = OUTPUTS / "ood_cxr" / d["ood_sub"] / "cv5" / "predictions"
    per_fold_raw, per_fold_cal, curves = [], [], []
    auc_raw, auc_cal = [], []
    sub_raw: dict[str, list[float]] = {}

    for seed in FOLD_SEEDS:
        p_path = pred_dir / f"{method}_{tag}_seed{seed}_overall.npz"
        if not p_path.exists():
            return None
        y, logits, attrs = load_predictions(p_path)
        y = np.asarray(y).astype(int)
        p = sigmoid(np.asarray(logits, dtype=np.float64))
        p_cal = cross_fit_isotonic(y, p)

        per_fold_raw.append(calibration_metrics(y, p))
        per_fold_cal.append(calibration_metrics(y, p_cal))
        auc_raw.append(marginal_worst_auc(y, np.asarray(logits, np.float64), attrs, d["tgt_key"]))
        auc_cal.append(marginal_worst_auc(y, p_cal, attrs, d["tgt_key"]))

        # 逐子群 CITL（只在原样分数上报——重校准后按定义已被拉回）
        kw = {k: np.asarray(v).astype(int) for k, v in attrs.items()
              if k in ("sex", "race", "age")}
        for g, m in subgroup_masks(d["tgt_key"], **kw).items():
            if "|" in g:
                continue
            sub_raw.setdefault(g, []).append(float(p[m].mean() - y[m].mean()))

        # 校准曲线（等频 10 箱）
        order = np.argsort(p)
        pts = [(float(p[c].mean()), float(y[c].mean()))
               for c in np.array_split(order, N_BINS) if len(c)]
        curves.append(pts)

    def avg(rows: list[dict]) -> dict[str, float]:
        return {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}

    return {
        "config": tag,
        "raw": avg(per_fold_raw), "recalibrated": avg(per_fold_cal),
        "marginal_worst_auc_raw": float(np.nanmean(auc_raw)),
        "marginal_worst_auc_recal": float(np.nanmean(auc_cal)),
        "subgroup_citl_raw": {g: float(np.mean(v)) for g, v in sub_raw.items()},
        "curve": [[float(np.mean([c[i][0] for c in curves])),
                   float(np.mean([c[i][1] for c in curves]))] for i in range(N_BINS)],
    }


def plot_direction(direction: str, res: dict, path: Path) -> None:
    """三面板：校准曲线 / 四指标对照 / 逐子群 CITL。"""
    d = DIRECTIONS[direction]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6),
                             gridspec_kw={"width_ratios": [1.0, 1.15, 1.0]})

    # --- 面板 1：校准曲线 ---
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1.2, label="完美校准")
    hi = 0.0
    for m, r in res.items():
        c = np.array(r["curve"])
        ax.plot(c[:, 0], c[:, 1], marker="o", markersize=4.5, linewidth=1.8,
                color=COLORS[m], label=LABELS[m], zorder=3)
        hi = max(hi, c[:, 0].max(), c[:, 1].max())
    ax.axhline(d["tgt_prev"], color="#e34948", linestyle=":", linewidth=1.3,
               label=f"target 实际患病率 {d['tgt_prev']:.1%}")
    ax.set_xlim(0, hi * 1.08)
    ax.set_ylim(0, hi * 1.08)
    ax.set_xlabel("预测概率（分箱均值）", fontsize=9, color=MUTED)
    ax.set_ylabel("实际患病比例", fontsize=9, color=MUTED)
    ax.set_title("OOD 校准曲线（原样迁移）", fontsize=10, color=INK)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.grid(color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)

    # --- 面板 2：四指标，原样 vs 重校准 ---
    ax = axes[1]
    keys = ("citl", "slope", "ece", "brier")
    names = ("CITL\n(0=完美)", "slope\n(1=完美)", "ECE\n(0=完美)", "Brier\n(越小越好)")
    xs = np.arange(len(keys))
    w = 0.26
    for i, m in enumerate(res):
        ax.bar(xs + (i - 1) * w, [res[m]["raw"][k] for k in keys], w * 0.9,
               color=COLORS[m], label=f"{LABELS[m]}（原样）", edgecolor="white", linewidth=0.7)
        ax.bar(xs + (i - 1) * w, [res[m]["recalibrated"][k] for k in keys], w * 0.9,
               facecolor="none", edgecolor=COLORS[m], linewidth=1.4, linestyle="--",
               label=f"{LABELS[m]}（重校准）")
    ax.axhline(0, color=INK, linewidth=0.9)
    ax.axhline(1, color=MUTED, linewidth=0.8, linestyle=":")     # slope 的完美参考
    ax.set_xticks(xs, names, fontsize=8)
    ax.set_title("校准指标：原样（实心） vs 重校准（虚框）", fontsize=10, color=INK)
    ax.legend(fontsize=6.8, frameon=False, ncol=2, loc="upper right")
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)

    # --- 面板 3：逐子群 CITL ---
    ax = axes[2]
    groups = list(next(iter(res.values()))["subgroup_citl_raw"].keys())
    y = np.arange(len(groups))
    for i, m in enumerate(res):
        ax.scatter([res[m]["subgroup_citl_raw"][g] for g in groups],
                   y + (i - 1) * 0.2, s=42, color=COLORS[m], label=LABELS[m],
                   zorder=3, edgecolors="white", linewidths=1.0)
    ax.axvline(0, color=INK, linewidth=1.0)
    ax.set_yticks(y, groups, fontsize=8.5)
    ax.set_xlabel("CITL = mean(p̂) − mean(y)（>0 = 系统性高估）", fontsize=9, color=MUTED)
    ax.set_title("逐 marginal 子群校准偏移", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)

    for a in axes:
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.tick_params(colors=MUTED, labelsize=8.5)
    fig.suptitle(f"{d['title']}  外部验证校准（source π={d['src_prev']:.1%} → "
                 f"target π={d['tgt_prev']:.1%}）", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="OOD 校准评估（TRIPOD+AI 外部验证要求）")
    ap.add_argument("--direction", choices=("M2C", "C2M"), default=None)
    ap.add_argument("--n-boot", type=int, default=1000, help="配对 cluster bootstrap 次数")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    payload = {}
    for direction, d in DIRECTIONS.items():
        if args.direction and direction != args.direction:
            continue
        cfg = json.loads((OUTPUTS / d["cfg_dir"] / "cv5" / "selected_configs.json").read_text())["config"]
        res = {}
        for m in METHODS:
            r = run_method(direction, m, cfg[m])
            if r is not None:
                res[m] = r
        if not res:
            print(f"[跳过] {direction}: 无可用预测")
            continue
        # 嵌套存放：per_method 只含方法条目，供绘图遍历；bootstrap 另置一层，避免污染
        payload[direction] = {"per_method": res}

        print(f"\n{'=' * 104}\n{d['title']}   source π={d['src_prev']:.1%} → "
              f"target π={d['tgt_prev']:.1%}\n{'=' * 104}")
        print(f"{'方法':<12s} {'CITL':>10s} {'slope':>9s} {'ECE':>9s} {'Brier':>9s}"
              f" | {'CITL(重校准)':>13s} {'slope(重校准)':>14s} {'ECE(重校准)':>12s}"
              f" | {'margWorst原样':>13s} {'margWorst重校准':>15s}")
        for m, r in res.items():
            print(f"{LABELS[m]:<12s} {r['raw']['citl']:>+10.4f} {r['raw']['slope']:>9.3f} "
                  f"{r['raw']['ece']:>9.4f} {r['raw']['brier']:>9.4f} | "
                  f"{r['recalibrated']['citl']:>+13.4f} {r['recalibrated']['slope']:>14.3f} "
                  f"{r['recalibrated']['ece']:>12.4f} | "
                  f"{r['marginal_worst_auc_raw']:>13.4f} {r['marginal_worst_auc_recal']:>15.4f}")

        # H3 判定所需的配对 CI（|CITL| 与 ECE 越小越好，故 Δ<0 = 校准更好）
        deltas = bootstrap_calibration_deltas(direction, cfg, args.n_boot)
        if deltas:
            payload[direction]["bootstrap_vs_erm"] = deltas
            print(f"\n  配对 cluster bootstrap（Δ = 方法 − ERM）")
            print(f"    判读：**|CITL| / ECE / Brier 的 Δ<0 = 校准更好**；"
                  f"citl 一行是有符号量，仅供参考，不代表优劣")
            for key, e in deltas.items():
                print(f"    [{key}]")
                for k in ("abs_citl", "ece", "brier", "citl"):
                    v = e[k]
                    lo, hi = v["ci"]
                    better = ""
                    if k != "citl" and v["sig"]:
                        better = "  ← 校准更好" if v["delta"] < 0 else "  ← 校准更差"
                    print(f"      {k:<9s} {v['delta']:>+9.4f} [{lo:>+8.4f},{hi:>+8.4f}]  "
                          f"{'**显著**' if v['sig'] else 'n.s.':<8s}{better}")

        p = args.out_dir / f"{direction.lower()}_calibration.png"
        plot_direction(direction, res, p)
        print(f"\n   图 → {p}")

    out = args.out_dir / "calibration.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已落盘 → {out}")


if __name__ == "__main__":
    main()
