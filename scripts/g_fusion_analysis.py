"""
实验 G · G3.2–G3.4：HyperAdapt × SWAD 的 2×2 主表、交互对比与折间方差
====================================================================
方案：`docs/hyperadapt_swad_fusion_plan.md`。四臂 2×2（{ERM, HyperAdapt} × {best-ckpt, +SWAD}），
产出三块：

  G3.2  **2×2 主表**：四臂 × {Overall, canonical worst, marginal worst, gap} 点估计。
  G3.3  **交互对比**：Δ_int = (HA+SWAD − SWAD) − (HA − ERM)，配对 cluster bootstrap + Holm 校正；
        另给三个单臂对比（HA+SWAD vs ERM / vs SWAD / vs HA）。
  G3.4  **折间方差**（H2）：各臂逐折 marginal worst 的 SD，以及
        `SD(HA+SWAD) − SD(HA)` 与 `SD(SWAD) − SD(ERM)` 的对比（SWAD 对谁的方差压得更狠）。

**口径**：复用 `build_oof_results_averaging` 的 Spec/Views/clusters（averaging：逐折算 → 折间平均
→ 再 min/max 取组）与患者/病灶级配对 cluster bootstrap（全方法共用同一批重采样索引）。

🛑 **结论分层（预注册，见方案 §6.5 的 MDE 结果，不得违反）**：
  · **worst-group 终点的交互检验没有功效**（MDE 0.045–0.089 ≈ 本数据集最大真效应 0.046–0.058）
    ⇒ 该终点上的 `Δ_int ≈ 0` **只能报为「无功效 null」**，不得表述为「证明两者无交互」。
  · **Overall 终点有功效**（MDE 0.007–0.012 ≪ SWAD 主效应 0.019–0.024）⇒ 可给实质结论。
  · **G3.4 的折间 SD 仅描述性**（5 折，无检验功效），不做显著性宣称。

运行（轻量 CPU，数分钟）：
    python scripts/g_fusion_analysis.py
    python scripts/g_fusion_analysis.py --n-boot 200 --datasets HAM10000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.build_oof_results_averaging import (
    OUTPUTS, SPECS, Spec, Views, build_masks, clusters, load_folds,
)
from scripts.eaudit_m2c_full_target import sorted_view, weighted_auc

ARMS = ("erm", "swad", "hyperadapt", "hyperadapt_swad")
ARM_LABEL = {"erm": "ERM", "swad": "SWAD", "hyperadapt": "HyperAdapt",
             "hyperadapt_swad": "HyperAdapt+SWAD"}
ENDPOINTS = ("Overall", "canonical worst", "marginal worst", "gap")
PRIMARY_IDX = 2                                   # marginal worst
# 结论分层：仅 Overall 终点有功效（方案 §6.5）
POWERED = {"Overall": True, "canonical worst": False, "marginal worst": False, "gap": False}


def per_fold_marginal_worst(spec: Spec, folds: list[dict]) -> list[float]:
    """
    逐折的 marginal worst-group AUC（供 G3.4 算折间 SD；口径同 D3 逐折表）。

    Args:
        spec : 数据集规格。
        folds: 该方法的逐折预测。

    Returns:
        长度 = 折数的列表；某折若所有边缘组都无定义则记 NaN。
    """
    out: list[float] = []
    for f in folds:
        vals = []
        for g, m in build_masks(spec, f["attrs"]).items():
            if "|" in g:                          # 只取边缘组（joint 小格在单折下退化为噪声）
                continue
            _, ys = sorted_view(f["y"], f["s"], m)      # ys = 该组样本按分数排序后的标签
            a = weighted_auc(ys, np.ones(len(ys)))      # 未加权 = 该折该组的普通 AUC
            if not np.isnan(a):
                vals.append(a)
        out.append(float(np.min(vals)) if vals else float("nan"))
    return out


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm–Bonferroni 校正：按 p 升序，第 i 个（0-based）乘 (m−i)，再取累积最大值保证单调。"""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj: dict[str, float] = {}
    run = 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, p * (m - i)))
        adj[k] = run
    return adj


def boot_p(dist: np.ndarray) -> float:
    """bootstrap 双侧 p：2·min(P(d<0), P(d>0))，下限为 1/B（B 次重采样的分辨率上限）。"""
    b = len(dist)
    p = 2.0 * min((dist < 0).mean(), (dist > 0).mean())
    return float(max(p, 1.0 / b))


def main() -> None:
    ap = argparse.ArgumentParser(description="实验 G 的 2×2 主表 / 交互对比 / 折间方差")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--datasets", nargs="+", default=["HAM10000", "Fitzpatrick"])
    ap.add_argument("--out", default=str(OUTPUTS / "analysis" / "g_fusion" / "fusion_2x2.json"))
    args = ap.parse_args()

    result: dict = {}
    for spec in SPECS:
        if spec.name not in args.datasets:
            continue
        cfg = json.loads(spec.cfg.read_text())["config"]
        folds = {}
        for a in ARMS:
            if a in cfg and (fl := load_folds(spec, a, cfg[a])) is not None:
                folds[a] = fl
        print(f"\n{'=' * 100}\n{spec.name}   臂: {[ARM_LABEL[a] for a in folds]}\n{'=' * 100}")
        missing = [a for a in ARMS if a not in folds]
        if missing:
            print(f"  [缺臂] {missing} → 跳过该数据集"); continue

        views = {a: Views(spec, folds[a]) for a in folds}
        cidx, offs = clusters(spec, folds["erm"])
        n_cl = int(cidx.max() + 1)
        obs = {a: np.array(views[a].compute(offs, np.ones(offs[-1]))[:4]) for a in folds}

        rng = np.random.default_rng(42)            # 与权威脚本同种子
        boot = {a: np.empty((args.n_boot, 4)) for a in folds}
        for b in range(args.n_boot):
            draw = rng.integers(0, n_cl, size=n_cl)
            w = np.bincount(draw, minlength=n_cl).astype(float)[cidx]
            for a in folds:
                boot[a][b] = views[a].compute(offs, w)[:4]

        # ---------------- G3.2 · 2×2 主表 ----------------
        print(f"\n【G3.2 · 2×2 主表】config: " +
              "  ".join(f"{ARM_LABEL[a]}={cfg[a]}" for a in ARMS))
        print(f"  {'臂':<18s} " + " ".join(f"{e:>16s}" for e in ENDPOINTS))
        for a in ARMS:
            print(f"  {ARM_LABEL[a]:<18s} " + " ".join(f"{obs[a][i]:>16.4f}" for i in range(4)))
        print(f"\n  2×2 视图（{ENDPOINTS[PRIMARY_IDX]} / Overall）：")
        print(f"  {'':<14s} {'best-ckpt':>18s} {'+SWAD':>18s} {'SWAD 增量':>18s}")
        for base, der in (("erm", "swad"), ("hyperadapt", "hyperadapt_swad")):
            row = (f"  {ARM_LABEL[base]:<14s} "
                   f"{obs[base][PRIMARY_IDX]:>8.4f}/{obs[base][0]:<9.4f} "
                   f"{obs[der][PRIMARY_IDX]:>8.4f}/{obs[der][0]:<9.4f} "
                   f"{obs[der][PRIMARY_IDX] - obs[base][PRIMARY_IDX]:>+8.4f}/"
                   f"{obs[der][0] - obs[base][0]:<+9.4f}")
            print(row)

        # ---------------- G3.3 · 对比 + 交互 ----------------
        contrasts = {
            "HA+SWAD − ERM":  [("hyperadapt_swad", 1), ("erm", -1)],
            "HA+SWAD − SWAD": [("hyperadapt_swad", 1), ("swad", -1)],
            "HA+SWAD − HA":   [("hyperadapt_swad", 1), ("hyperadapt", -1)],
            "Δ_int = (HA+SWAD−SWAD)−(HA−ERM)":
                [("hyperadapt_swad", 1), ("swad", -1), ("hyperadapt", -1), ("erm", 1)],
        }
        ds_res: dict = {"config": {a: cfg[a] for a in ARMS},
                        "point": {a: dict(zip(ENDPOINTS, obs[a].tolist())) for a in ARMS},
                        "contrasts": {}}
        for ei, ename in enumerate(ENDPOINTS):
            raw_p, cache = {}, {}
            for label, expr in contrasts.items():
                d = float(sum(c * obs[m][ei] for m, c in expr))
                dist = sum(c * boot[m][:, ei] for m, c in expr)
                lo, hi = np.percentile(dist, [2.5, 97.5])
                cache[label] = {"delta": d, "ci": [float(lo), float(hi)], "p": boot_p(dist)}
                raw_p[label] = cache[label]["p"]
            adj = holm(raw_p)                      # Holm 在同一终点的 4 个对比家族内校正
            tag = "有功效" if POWERED[ename] else "🛑无功效(见§6.5)"
            print(f"\n【G3.3 · 对比】终点 = {ename}  [{tag}]")
            print(f"  {'对比':<34s} {'Δ':>9s} {'95%CI':>20s} {'p':>8s} {'p_Holm':>8s} {'判定':>8s}")
            for label in contrasts:
                c = cache[label]
                c["p_holm"] = adj[label]
                sig = "显著" if adj[label] < 0.05 else "n.s."
                print(f"  {label:<34s} {c['delta']:>+9.4f} "
                      f"[{c['ci'][0]:>+7.4f},{c['ci'][1]:>+7.4f}] "
                      f"{c['p']:>8.3f} {adj[label]:>8.3f} {sig:>8s}")
            ds_res["contrasts"][ename] = cache

        # ---------------- G3.4 · 折间方差（描述性）----------------
        print(f"\n【G3.4 · 折间方差（描述性，H2）】逐折 marginal worst-group AUC")
        pf = {a: per_fold_marginal_worst(spec, folds[a]) for a in ARMS}
        print(f"  {'臂':<18s} " + " ".join(f"{'f'+str(k):>8s}" for k in range(5)) +
              f" {'均值':>9s} {'SD':>9s}")
        for a in ARMS:
            v = np.array(pf[a], dtype=float)
            print(f"  {ARM_LABEL[a]:<18s} " + " ".join(f"{x:>8.4f}" for x in v) +
                  f" {np.nanmean(v):>9.4f} {np.nanstd(v, ddof=1):>9.4f}")
        sd = {a: float(np.nanstd(np.array(pf[a], dtype=float), ddof=1)) for a in ARMS}
        d_hn = sd["hyperadapt_swad"] - sd["hyperadapt"]
        d_erm = sd["swad"] - sd["erm"]
        print(f"\n  SWAD 对折间 SD 的作用：HN 侧 {d_hn:>+.4f}   ERM 侧 {d_erm:>+.4f}   "
              f"差 {d_hn - d_erm:>+.4f}")
        print(f"  → H2 方向：{'支持（SWAD 对 HN 压方差更狠）' if d_hn < d_erm else '不支持'}"
              f"（**描述性**，5 折无检验功效）")
        ds_res["fold_sd"] = {"per_fold": pf, "sd": sd,
                             "d_hn": d_hn, "d_erm": d_erm, "diff": d_hn - d_erm}

        # ---------------- G3.5 · 配置匹配对照（方案 §2.1，零训练）----------------
        # 朴素 Δ_int 的混杂：ERM/SWAD 用 ERM 的选定 config，HA/HA+SWAD 用 HA 的选定 config。
        # 两个「SWAD 增量」因此测在**不同 lr/wd** 上，而 SWAD 的收益本就依赖训练噪声水平
        # （lr 越小轨迹越平滑、可平均掉的噪声越少）。ERM 搜索阶段带 SWAD 跑满 6 配置，
        # 故可白拿 erm/swad 在 **HA 的 config** 上的一对，把 ERM 侧增量也测在同一 config 上。
        if cfg["erm"] != cfg["hyperadapt"]:
            mt = cfg["hyperadapt"]
            fe, fs = load_folds(spec, "erm", mt), load_folds(spec, "swad", mt)
            if fe is None or fs is None:
                print(f"\n【G3.5 · 配置匹配对照】缺 erm/swad @ {mt} 预测 → 跳过")
            else:
                ve, vs = Views(spec, fe), Views(spec, fs)
                oe = np.array(ve.compute(offs, np.ones(offs[-1]))[:4])
                os_ = np.array(vs.compute(offs, np.ones(offs[-1]))[:4])
                be, bs = np.empty((args.n_boot, 4)), np.empty((args.n_boot, 4))
                rng2 = np.random.default_rng(42)          # 同种子 ⇒ 与上面的抽样逐次一致（配对）
                for b in range(args.n_boot):
                    draw = rng2.integers(0, n_cl, size=n_cl)
                    w = np.bincount(draw, minlength=n_cl).astype(float)[cidx]
                    be[b] = ve.compute(offs, w)[:4]
                    bs[b] = vs.compute(offs, w)[:4]
                print(f"\n【G3.5 · 配置匹配对照】ERM/SWAD 改用 HA 的 config = {mt}"
                      f"（原为 {cfg['erm']}）")
                print(f"  {'终点':<18s} {'ERM侧SWAD增量':>16s} {'HN侧SWAD增量':>16s} "
                      f"{'匹配Δ_int':>12s} {'95%CI':>20s} {'p':>7s}")
                matched: dict = {}
                for ei, ename in enumerate(ENDPOINTS):
                    inc_erm = os_[ei] - oe[ei]
                    inc_hn = obs["hyperadapt_swad"][ei] - obs["hyperadapt"][ei]
                    dist = ((boot["hyperadapt_swad"][:, ei] - boot["hyperadapt"][:, ei])
                            - (bs[:, ei] - be[:, ei]))
                    lo, hi = np.percentile(dist, [2.5, 97.5])
                    p = boot_p(dist)
                    print(f"  {ename:<18s} {inc_erm:>+16.4f} {inc_hn:>+16.4f} "
                          f"{inc_hn - inc_erm:>+12.4f} [{lo:>+7.4f},{hi:>+7.4f}] {p:>7.3f}")
                    matched[ename] = {"erm_side_increment": float(inc_erm),
                                      "hn_side_increment": float(inc_hn),
                                      "delta_int_matched": float(inc_hn - inc_erm),
                                      "ci": [float(lo), float(hi)], "p": p}
                ds_res["config_matched"] = {"tag": mt, "by_endpoint": matched}
        else:
            print(f"\n【G3.5 · 配置匹配对照】四臂本就同 config ({cfg['erm']}) ⇒ 朴素 Δ_int 已无混杂")
            ds_res["config_matched"] = {"tag": cfg["erm"], "note": "四臂同 config，无需匹配"}

        result[spec.name] = ds_res

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已落盘 → {out}")


if __name__ == "__main__":
    main()
