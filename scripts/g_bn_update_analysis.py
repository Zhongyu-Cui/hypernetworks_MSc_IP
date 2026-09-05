"""
实验 G · BN 对齐（update_bn）的 **ID 侧**影响分析
=================================================
配套 `scripts/g_bn_update_swad.py`。把两个**权重平均派生**臂（SWAD、HyperAdapt+SWAD）的取数
换到 BN 与官方 SWAD 对齐后的预测（`outputs/<ds>/cv5_bnupd/predictions/`），
**ERM / HyperAdapt 仍读原目录**——它们不是平均权重、不受 BN 重估影响，作逐位不变的对照。

产出三块（口径与 `g_fusion_analysis.py` 完全一致：averaging，逐折算 → 折间平均 → 再 min/max 取组）：

  1. **逐臂 BN 前后对比**：Overall / canonical worst / marginal worst / gap 的点估计与差值。
  2. **2×2 主表在新 BN 处理下的重算**，含 `Δ_int = (HA+SWAD − SWAD) − (HA − ERM)`
     ——检验报告 §4.3「四库无正交互」是否被 BN 处理改变。
  3. **配对 cluster bootstrap**（患者/病灶级，全臂共用同一批重采样索引）给 BN 前后差值的 CI。

> ⚠️ **判读限制沿用报告 §4.5**：HAM / Fitzpatrick 的 worst-group 终点**无检验功效**
> （MDE 0.045–0.089），其 null 只能报「测不出」。本脚本不改变功效状况。

运行（轻量 CPU）：
    python scripts/g_bn_update_analysis.py
    python scripts/g_bn_update_analysis.py --datasets HAM10000 --n-boot 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.build_oof_results_averaging import (
    OUTPUTS, SPECS, Spec, Views, clusters, load_folds,
)

ARMS = ("erm", "swad", "hyperadapt", "hyperadapt_swad")
# BN 重估只作用于权重平均派生臂；其余臂读原目录 ⇒ 对照逐位不变
BNUPD_ARMS = ("swad", "hyperadapt_swad")
METRIC_NAMES = ("Overall", "canonical worst", "marginal worst", "gap")
OUT_DIR = OUTPUTS / "analysis" / "g_fusion"
# 只有这四个 ID regime 有 cv5_bnupd 产物（OOD 另由校准脚本处理）
ID_SPECS = ("HAM10000", "Fitzpatrick", "MIMIC", "CheXpert")


def bnupd_spec(spec: Spec) -> Spec:
    """同一 regime 的 bnupd 版 Spec：只把预测目录换成 `cv5_bnupd`，其余口径逐位不变。"""
    rel = spec.pred_dir.relative_to(OUTPUTS).as_posix().replace("/cv5/", "/cv5_bnupd/")
    return Spec(spec.name + "_bnupd", rel, spec.ds_key,
                spec.cfg.parent.relative_to(OUTPUTS).as_posix(),
                spec.split_sub, spec.cluster_col, spec.filter_age_neg1)


def load_arm(spec: Spec, spec_bn: Spec, method: str, tag: str,
             bnupd: bool) -> list[dict] | None:
    """按臂选目录载入 5 折预测：bnupd 且属权重平均派生臂 → bnupd 目录，否则原目录。"""
    use = spec_bn if (bnupd and method in BNUPD_ARMS) else spec
    return load_folds(use, method, tag)


def run_spec(spec: Spec, n_boot: int) -> dict | None:
    """跑一个数据集：BN 前后各算一套四指标，并对差值做配对 cluster bootstrap。"""
    spec_bn = bnupd_spec(spec)
    if not spec_bn.pred_dir.exists():
        print(f"[跳过] {spec.name}: 无 {spec_bn.pred_dir}")
        return None
    cfg = json.loads(spec.cfg.read_text())["config"]

    folds: dict[tuple[str, bool], list[dict]] = {}
    views: dict[tuple[str, bool], Views] = {}
    for method in ARMS:
        if method not in cfg:
            continue
        # 只有权重平均派生臂需要第二套视图；ERM/HyperAdapt 的 bnupd 版与原版逐位相同，不重复载入
        for bn in ((False, True) if method in BNUPD_ARMS else (False,)):
            fl = load_arm(spec, spec_bn, method, cfg[method], bn)
            if fl is None:
                print(f"[跳过] {spec.name}: 缺 {method} (bnupd={bn}) 预测")
                return None
            folds[(method, bn)] = fl
            views[(method, bn)] = Views(spec, fl)

    # 聚类索引取自 ERM（BN 前后同一批样本、同一行序）
    cidx, offs = clusters(spec, folds[("erm", False)])
    n_cl = int(cidx.max() + 1)
    ones = np.ones(offs[-1])

    def vec(method: str, bn: bool, w: np.ndarray) -> np.ndarray:
        """该臂的四指标向量（Overall, canonical worst, marginal worst, gap）。"""
        return np.asarray(views[(method, bn)].compute(offs, w)[:4])

    obs = {(m, bn): vec(m, bn, ones) for (m, bn) in views}

    def dint(w: np.ndarray, bn: bool) -> np.ndarray:
        """Δ_int = (HA+SWAD − SWAD) − (HA − ERM)；ERM/HA 恒取原目录。"""
        return ((vec("hyperadapt_swad", bn, w) - vec("swad", bn, w))
                - (vec("hyperadapt", False, w) - vec("erm", False, w)))

    obs_dint = {bn: dint(ones, bn) for bn in (False, True)}

    # --- 配对 cluster bootstrap：全臂共用同一批重采样索引 ---
    rng = np.random.default_rng(42)
    boot: dict[str, list[np.ndarray]] = {}
    for _ in range(n_boot):
        draw = rng.integers(0, n_cl, size=n_cl)
        w = np.bincount(draw, minlength=n_cl).astype(float)[cidx]
        for method in BNUPD_ARMS:
            boot.setdefault(method, []).append(vec(method, True, w) - vec(method, False, w))
        boot.setdefault("dint", []).append(dint(w, True) - dint(w, False))

    def ci_block(samples: list[np.ndarray], point: np.ndarray) -> dict:
        arr = np.asarray(samples)
        out = {}
        for j, name in enumerate(METRIC_NAMES):
            lo, hi = np.percentile(arr[:, j], [2.5, 97.5])
            out[name] = {"delta": float(point[j]), "ci": [float(lo), float(hi)],
                         "sig": bool(lo > 0 or hi < 0)}
        return out

    return {
        "dataset": spec.name, "n_boot": n_boot, "n_clusters": n_cl,
        "config": {m: cfg[m] for m in ARMS if m in cfg},
        "point": {f"{m}{'_bnupd' if bn else ''}": {n: float(v) for n, v in zip(METRIC_NAMES, obs[(m, bn)])}
                  for (m, bn) in obs},
        "dint": {("bnupd" if bn else "orig"): {n: float(v) for n, v in zip(METRIC_NAMES, obs_dint[bn])}
                 for bn in (False, True)},
        "bn_effect": {m: ci_block(boot[m], obs[(m, True)] - obs[(m, False)]) for m in BNUPD_ARMS},
        "dint_shift": ci_block(boot["dint"], obs_dint[True] - obs_dint[False]),
    }


def print_spec(r: dict) -> None:
    """打印一个数据集的 BN 前后对照。"""
    print(f"\n{'=' * 104}\n{r['dataset']}   聚类数={r['n_clusters']:,}   "
          f"（ERM / HyperAdapt 不受 BN 重估影响，逐位不变）\n{'=' * 104}")
    print(f"{'臂':<22s} " + " ".join(f"{n:>16s}" for n in METRIC_NAMES))
    for m in ARMS:
        row = r["point"].get(m)
        if row:
            print(f"{m:<22s} " + " ".join(f"{row[n]:>16.4f}" for n in METRIC_NAMES))
        if m in BNUPD_ARMS:
            rb = r["point"].get(f"{m}_bnupd")
            if rb:
                print(f"{m + ' (BN对齐)':<22s} " + " ".join(f"{rb[n]:>16.4f}" for n in METRIC_NAMES))

    print(f"\n  BN 重估的效应（BN对齐 − 原），配对 cluster bootstrap 95% CI")
    for m in BNUPD_ARMS:
        print(f"    [{m}]")
        for n in METRIC_NAMES:
            e = r["bn_effect"][m][n]
            lo, hi = e["ci"]
            print(f"      {n:<18s} {e['delta']:>+9.4f} [{lo:>+8.4f},{hi:>+8.4f}] "
                  f"{'**显著**' if e['sig'] else 'n.s.'}")

    print(f"\n  交互项 Δ_int = (HA+SWAD − SWAD) − (HA − ERM)")
    print(f"    {'终点':<18s} {'原 BN 处理':>12s} {'BN 对齐后':>12s} {'移动量':>10s} {'95% CI':>21s}")
    for n in METRIC_NAMES:
        s = r["dint_shift"][n]
        lo, hi = s["ci"]
        print(f"    {n:<18s} {r['dint']['orig'][n]:>+12.4f} {r['dint']['bnupd'][n]:>+12.4f} "
              f"{s['delta']:>+10.4f} [{lo:>+8.4f},{hi:>+8.4f}] "
              f"{'**显著**' if s['sig'] else 'n.s.'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="实验 G：BN 对齐（update_bn）的 ID 侧影响")
    ap.add_argument("--datasets", nargs="*", default=None, help=f"缺省全部：{ID_SPECS}")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "bn_update_id.json")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    wanted = tuple(args.datasets) if args.datasets else ID_SPECS
    payload = {}
    for spec in SPECS:
        if spec.name not in wanted:
            continue
        r = run_spec(spec, args.n_boot)
        if r is None:
            continue
        payload[spec.name] = r
        print_spec(r)

    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已落盘 → {args.out}")


if __name__ == "__main__":
    main()
