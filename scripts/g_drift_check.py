"""
实验 G · G1.4 / G2.4：重跑 HyperAdapt 臂的**漂移核对**（新产物 vs `_pre_G_backup` 快照）
========================================================================================
实验 G 为了在同一条权重轨迹上派生 SWAD 平均模型，须以**相同的 (method, config, seed)** 重跑
HyperAdapt，因而原地覆写了已冻结的 cv5 产物（D1–D4 结论表的数据来源）。G0.5 已先做快照。
本脚本比较「新跑」与「快照」在 averaging 口径各终点上的差，回答两件事：

  1. **训练非确定性有多大**（cuDNN 算法选择 / DataLoader worker 顺序 ⇒ 同 seed 不逐位复现）；
  2. **既有结论是否被动摇**——若某终点漂移量级接近文档中的效应量（如 HAM 的 HN Δworst≈0.02），
     则 D1–D4 的相应数字须标注不确定性。

**口径**：复用 `build_oof_results_averaging` 的 Spec/Views/clusters（averaging：逐折算 → 折间平均
→ 再 min/max 取组），与权威分析同源，不新增统计口径。除终点差外另报**逐折 logit 相关**，
用以区分「模型几乎相同、指标微动」与「模型实质不同」。

⚠️ 本脚本**只读**，不改动任何产物。

运行（轻量 CPU，秒级）：
    python scripts/g_drift_check.py                       # HAM + Fitzpatrick
    python scripts/g_drift_check.py --datasets HAM10000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from scripts.build_oof_results_averaging import (
    OUTPUTS, SPECS, Spec, Views, clusters, load_folds,
)

ENDPOINT_NAMES = ("Overall", "canonical worst", "marginal worst", "gap")
# 参照：文档中 HN 相对 ERM 的 worst-group 效应量级（averaging 口径），用于判断漂移是否可忽略。
# ⚠️ 两个 CXR 库的 HN 效应本就极小（MIMIC·HyperAdapt +0.0022 / CheXpert·HyperFusion +0.0029，
# 均「显著但 |Δ|<0.005」，靠 n≥138k 的功效撑出来的），故此处比值天然会很大——这本身就是
# 「显著 ≠ 重要」的量化体现，不代表 CXR 的漂移绝对量比皮肤镜大。
REFERENCE_EFFECT = {"HAM10000": 0.021, "Fitzpatrick": 0.017,
                    "MIMIC": 0.0022, "CheXpert": 0.0029}


def backup_spec(spec: Spec) -> Spec:
    """把某数据集的 Spec 复制一份、预测目录改指向 `_pre_G_backup/predictions`（其余口径不变）。"""
    rel = spec.pred_dir.relative_to(OUTPUTS)                     # 如 ham10000/cv5/predictions
    bak = rel.parent / "_pre_G_backup" / "predictions"
    return Spec(spec.name + "(backup)", str(bak), spec.ds_key,
                str(spec.cfg.parent.relative_to(OUTPUTS)),
                spec.split_sub, spec.cluster_col, spec.filter_age_neg1)


def main() -> None:
    ap = argparse.ArgumentParser(description="实验 G 重跑 HyperAdapt 臂的漂移核对（只读）")
    ap.add_argument("--datasets", nargs="+", default=["HAM10000", "Fitzpatrick"])
    ap.add_argument("--method", default="hyperadapt", help="被覆写的方法名")
    ap.add_argument("--out", default=str(OUTPUTS / "analysis" / "g_fusion" / "drift_check.json"))
    args = ap.parse_args()

    result: dict = {}
    for spec in SPECS:
        if spec.name not in args.datasets:
            continue
        tag = json.loads(spec.cfg.read_text())["config"][args.method]
        bspec = backup_spec(spec)
        new = load_folds(spec, args.method, tag)
        old = load_folds(bspec, args.method, tag)
        print(f"\n{'=' * 84}\n{spec.name}  method={args.method}  tag={tag}\n{'=' * 84}")
        if new is None or old is None:
            print(f"  [跳过] 新产物={'有' if new else '缺'} / 快照={'有' if old else '缺'}")
            continue

        # --- 终点差（averaging 口径，无 bootstrap：这里比的是点估计的漂移）---
        cidx, offs = clusters(spec, new)
        ones = np.ones(offs[-1])
        v_new = Views(spec, new).compute(offs, ones)
        v_old = Views(bspec, old).compute(offs, ones)
        print(f"  {'终点':<18s} {'快照':>9s} {'新跑':>9s} {'漂移Δ':>9s}")
        ds: dict = {"tag": tag, "endpoints": {}}
        for i, nm in enumerate(ENDPOINT_NAMES):
            d = v_new[i] - v_old[i]
            print(f"  {nm:<18s} {v_old[i]:>9.4f} {v_new[i]:>9.4f} {d:>+9.4f}")
            ds["endpoints"][nm] = {"backup": v_old[i], "new": v_new[i], "drift": d}

        # --- 逐折 logit 一致性：区分「模型几乎相同」与「模型实质不同」---
        print(f"\n  {'fold':<6s} {'Pearson r':>10s} {'Spearman ρ':>11s} {'max|Δlogit|':>12s}")
        cors = []
        for k, (fo, fn) in enumerate(zip(old, new)):
            assert np.array_equal(fo["y"], fn["y"]), f"fold{k}: 标签不对齐，无法比较"
            r = float(pearsonr(fo["s"], fn["s"])[0])
            rho = float(spearmanr(fo["s"], fn["s"])[0])
            mx = float(np.max(np.abs(fo["s"] - fn["s"])))
            cors.append({"fold": k, "pearson": r, "spearman": rho, "max_abs_dlogit": mx})
            print(f"  {k:<6d} {r:>10.4f} {rho:>11.4f} {mx:>12.4f}")
        ds["per_fold_logit"] = cors

        # --- 裁定：漂移是否小到不影响既有结论 ---
        mw = abs(ds["endpoints"]["marginal worst"]["drift"])
        ref = REFERENCE_EFFECT.get(spec.name)
        if ref is not None:
            ratio = mw / ref
            verdict = ("可忽略（<20% 效应量）" if ratio < 0.2 else
                       "需在报告中标注" if ratio < 0.5 else "🛑 与效应量同量级，既有结论须复核")
            print(f"\n  marginal-worst 漂移 |Δ|={mw:.4f} vs 参照效应量 {ref:.3f} "
                  f"({ratio:.0%}) → {verdict}")
            ds["verdict"] = {"drift_abs": mw, "reference_effect": ref,
                             "ratio": ratio, "verdict": verdict}
        result[spec.name] = ds

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已落盘 → {out}")


if __name__ == "__main__":
    main()
