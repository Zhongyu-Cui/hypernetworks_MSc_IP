"""
E1 ρ_l 聚合：把逐 run 的 e1_rho{,_worstcase}.json 汇总成 plan §4 的 per-cell 层级表
=================================================================================
读 `e1_rho{suffix}.json`（build_rho_only/build_logit_tables 落盘的逐 (cell,fold,seed) ρ），
按 cell 内各层对「5 折 × 3 seed」求均值，产出 {cell|layer -> {rho, rho_between, ratio}}，
并按 conv/fc 汇总每档 cell 的「fc ρ_between」「conv ρ_between 均值」「fc/conv 倍数」。

用法：
    python scripts/e1_rho_aggregate.py --checkpoint worstcase
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.e1_ham_analysis import CV_DIR, NICE

CELLS = ("condnet_head", "condnet_deep", "condnet_full")


def aggregate(checkpoint: str) -> dict:
    """把逐 run ρ 按 cell|layer 跨 fold×seed 平均。"""
    suf = "" if checkpoint == "overall" else f"_{checkpoint}"
    rho_all = json.loads((CV_DIR / f"e1_rho{suf}.json").read_text())
    # 收集每个 (cell, layer) 的逐 run rho / rho_between
    acc: dict[str, dict[str, list[float]]] = {}
    for run_key, layers in rho_all.items():
        cell = run_key.split("|")[0]
        for lname, v in layers.items():
            key = f"{cell}|{lname}"
            acc.setdefault(key, {"rho": [], "rho_between": []})
            # 兼容两种结构：新版 {rho, rho_between} dict；旧版仅 float rho（无 rho_between）
            if isinstance(v, dict):
                acc[key]["rho"].append(v["rho"])
                acc[key]["rho_between"].append(v["rho_between"])
            else:
                acc[key]["rho"].append(float(v))
                acc[key]["rho_between"].append(float("nan"))
    out = {}
    for key, d in acc.items():
        rho = float(np.mean(d["rho"]))
        betw = float(np.mean(d["rho_between"]))
        out[key] = {"rho": rho, "rho_between": betw,
                    "ratio": betw / rho if rho else float("nan"),
                    "n_runs": len(d["rho"])}
    return out


def summarize(agg: dict) -> None:
    """按 cell 打印 fc vs conv 的 ρ_between 对比（回答『用了 fc 以外的层吗』）。"""
    print(f"| {'cell':<7s} | {'fc ρ':>7s} | {'fc ρ_betw':>9s} | {'conv ρ_betw(均)':>14s} | {'fc/conv 倍':>9s} |")
    for cell in CELLS:
        fc = agg.get(f"{cell}|fc")
        convs = [v["rho_between"] for k, v in agg.items()
                 if k.startswith(f"{cell}|") and not k.endswith("|fc")]
        if fc is None:
            continue
        conv_mean = float(np.mean(convs)) if convs else float("nan")
        ratio = fc["rho_between"] / conv_mean if convs and conv_mean else float("nan")
        print(f"| {NICE[cell]:<7s} | {fc['rho']:>7.4f} | {fc['rho_between']:>9.4f} | "
              f"{conv_mean:>14.4f} | {ratio:>9.1f} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", choices=("overall", "worstcase"), default="overall")
    args = ap.parse_args()
    agg = aggregate(args.checkpoint)
    suf = "" if args.checkpoint == "overall" else f"_{args.checkpoint}"
    p = CV_DIR / f"e1_rho_between{suf}.json"
    p.write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"checkpoint={args.checkpoint}  已落盘 → {p}\n")
    summarize(agg)


if __name__ == "__main__":
    main()
