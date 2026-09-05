"""
OOD 确认性家族（第 4 章 §4.4）：3 个超网络 × 3 个基线 = **每方向 9 个对比**。

`ood_variance_decomposition.py` 只算 vs ERM，但它已把逐方法的 bootstrap 分布落盘
（`boot` 键，全方法共用同一批重采样索引），故任意一对方法的配对差就是 `B[m1] − B[m2]`，
无须重跑 bootstrap。H2 的 sigma_train 同理由 `per_cell` 的 [fold, trial] 矩阵直接算。

Holm 在**每个方向自己的 9 个检验内**校正（第 4 章 §4.4.2 的措辞）。

运行（秒级）：
    PYTHONPATH=. python scripts/ood_confirmatory_family.py --src <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

HN = ("hyperhead", "hyperfusion", "hyperadapt")
BASE = ("erm", "swad", "groupdro")
LABEL = {"erm": "ERM", "swad": "SWAD", "groupdro": "GroupDRO",
         "hyperhead": "HyperHead", "hyperfusion": "HyperFusion",
         "hyperadapt": "HyperAdapt"}


def anova_sigma_train(d: np.ndarray) -> float:
    """每格 1 观测的双因子随机效应分解，返回 sigma_train（seed 分量 + 残差）。"""
    f, t = d.shape
    grand = d.mean()
    row, col = d.mean(axis=1), d.mean(axis=0)
    ms_f = t * ((row - grand) ** 2).sum() / (f - 1)
    ms_t = f * ((col - grand) ** 2).sum() / (t - 1)
    ms_e = ((d - row[:, None] - col[None, :] + grand) ** 2).sum() / ((f - 1) * (t - 1))
    return float(np.sqrt(max((ms_t - ms_e) / f, 0.0) + max(ms_e, 0.0)))


def holm(pvals: dict[str, float]) -> dict[str, bool]:
    """Holm 校正，alpha=0.05，返回逐检验是否拒绝。"""
    order = sorted(pvals.items(), key=lambda kv: kv[1])
    out, prev = {}, True
    for i, (k, p) in enumerate(order):
        prev = prev and p <= 0.05 / (len(order) - i)
        out[k] = prev
    return out


def run(path: Path, metric: str) -> None:
    """打印一个指标下两个方向各 9 个对比的 H1 / H2 判决。"""
    payload = json.loads(path.read_text())
    for dk in ("M2C", "C2M"):
        e = payload[dk]
        boot = {m: np.asarray(v) for m, v in e["boot"].items()}
        cell = {m: np.asarray(v) for m, v in e["per_cell"].items()}
        rows, pvals = {}, {}
        for h in HN:
            for b in BASE:
                key = f"{h}_vs_{b}"
                diff = boot[h] - boot[b]
                delta = float(cell[h].mean() - cell[b].mean())
                lo, hi = np.percentile(diff, [2.5, 97.5])
                se = (hi - lo) / (2 * 1.96)
                pvals[key] = float(2 * (1 - norm.cdf(abs(delta) / se))) if se > 0 else 1.0
                st = anova_sigma_train(cell[h] - cell[b])
                rows[key] = (delta, float(lo), float(hi), st, abs(delta) > st)
        rej = holm(pvals)
        print(f"\n=== {dk} · {metric} ===")
        print(f"{'comparison':<28s}{'delta':>10s}{'95% CI':>24s}{'Holm p':>10s}"
              f"{'H1':>6s}{'sigma_tr':>10s}{'|d|/s':>8s}{'H2':>5s}")
        for k, (d, lo, hi, st, exc) in rows.items():
            h, b = k.split("_vs_")
            print(f"{LABEL[h]+' - '+LABEL[b]:<28s}{d:>+10.4f}"
                  f"  [{lo:>+7.4f},{hi:>+7.4f}]{pvals[k]:>10.3g}"
                  f"{'yes' if rej[k] else 'no':>6s}{st:>10.4f}{abs(d)/st:>8.2f}"
                  f"{'yes' if exc else 'no':>5s}")
        n_sig = sum(rej.values())
        n_pos_sig = sum(1 for k, v in rows.items() if rej[k] and v[0] > 0)
        print(f"  significant {n_sig}/9, of which improvements {n_pos_sig}; "
              f"clearing sigma_train {sum(v[4] for v in rows.values())}/9")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    args = ap.parse_args()
    for metric, sub in (("marginal_worst", "mw"), ("overall", "ov")):
        run(args.src / sub / f"variance_decomposition_{metric}.json", metric)


if __name__ == "__main__":
    main()
