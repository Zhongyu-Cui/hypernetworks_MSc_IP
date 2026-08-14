"""
实验 G · G3.1：worst-group 主要终点的**最小可检出效应（MDE）**
================================================================
**跑分析前必须先完成**（方案 `docs/hyperadapt_swad_fusion_plan.md` §5 冻结规程）：把 H1 的零假设
写成 `Δ_int = 0 ± MDE`，而不是含糊的「Δ_int ≈ 0」。动机是本项目的历史教训——路线 A 之前，
worst-group 的「无差异」结论一度建立在**没有功效**的比较上（评估-n 地板），事后才发现测不出。
若 MDE 大于任何合理的交互效应量，本实验只能给「无功效」读数，须在报告中明说、不得包装成 null 判决。

**口径**：与权威分析**完全同源**——直接复用 `build_oof_results_averaging` 的
Spec / load_folds / Views / clusters（averaging：逐折算 → 折间平均 → 再 min/max 取组）与
**患者/病灶级配对 cluster bootstrap**（全方法共用同一批重采样索引，只重采样 cluster）。
本脚本不新增任何统计口径，只是把同一套 bootstrap 分布拿来算 SE 而非 CI。

**MDE 定义**：双侧 α=0.05、功效 1−β 下，`MDE = (z_{1-α/2} + z_{1-β}) · SE`
（80% 功效 ⇒ 2.802·SE；90% ⇒ 3.242·SE）。SE 取配对 bootstrap 分布的标准差。

**为什么要用代理对比**：Δ_int = (HA+SWAD − SWAD) − (HA − ERM) 里的 HA+SWAD 尚未训练完，
无法直接算其 SE。故用**同结构的四臂差中差代理**（如 (HyperFusion − SWAD) − (HyperAdapt − ERM)）
估计尺度——它与 Δ_int 同为「两个配对差之差、同一批 OOF 样本、同一批 bootstrap 抽样」。
⚠️ 代理**偏保守**：真实的 HA+SWAD 由 HA 自身的权重轨迹派生，与 HA 臂高度正相关，
`Var(D1−D2)=Var(D1)+Var(D2)−2Cov` 中的 Cov 会更大 ⇒ 真 SE 应**小于**代理 SE。
故「代理 MDE 已小于目标效应」是充分的功效证据；反之则须谨慎解读。

运行（轻量 CPU，约数分钟）：
    python scripts/g_mde_worstgroup.py                     # HAM + Fitzpatrick
    python scripts/g_mde_worstgroup.py --n-boot 200        # 快速试跑
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.build_oof_results_averaging import (
    OUTPUTS, SPECS, Spec, Views, clusters, load_folds,
)

# 主要终点在 Views.compute 返回元组中的下标（0=Overall,1=canonical worst,2=marginal worst,3=gap）
ENDPOINTS = {"overall": 0, "canonical_worst": 1, "marginal_worst": 2}
PRIMARY = "marginal_worst"                       # 方案 §5：主要终点 = marginal worst-group AUC
Z_ALPHA = 1.959964                               # 双侧 α=0.05
Z_POWER = {"80": 0.841621, "90": 1.281552}       # 键即功效百分数（用于 mde_80 / mde_90 字段名）
# 实验 G 涉及的方法（HA+SWAD 尚未落盘时自动跳过）
G_METHODS = ("erm", "swad", "hyperadapt", "hyperfusion", "hyperhead", "hyperadapt_swad")


def bootstrap_endpoints(
    spec: Spec, methods: dict[str, list[dict]], n_boot: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    对每个方法跑配对 cluster bootstrap，返回各终点的观测值与 bootstrap 分布。

    全方法共用同一批重采样索引（配对），故任意两方法之差的 SE 已扣除共同的样本波动。

    Args:
        spec   : 数据集规格（决定聚类键、子群 schema、过滤规则）。
        methods: {method -> 逐折预测}（load_folds 的返回）。
        n_boot : bootstrap 次数。

    Returns:
        (obs, boot)：obs[m] = [3] 各终点观测值；boot[m] = [n_boot, 3] bootstrap 分布。
    """
    views = {m: Views(spec, fl) for m, fl in methods.items()}
    cidx, offs = clusters(spec, methods["erm"])
    n_cl = int(cidx.max() + 1)
    ones = np.ones(offs[-1])

    idx = list(ENDPOINTS.values())
    obs = {m: np.array(views[m].compute(offs, ones), dtype=object)[idx].astype(float)
           for m in views}
    boot = {m: np.empty((n_boot, len(idx))) for m in views}
    rng = np.random.default_rng(42)               # 与权威脚本同种子，抽样序列一致
    for b in range(n_boot):
        draw = rng.integers(0, n_cl, size=n_cl)
        w = np.bincount(draw, minlength=n_cl).astype(float)[cidx]
        for m in views:
            r = views[m].compute(offs, w)
            boot[m][b] = [r[i] for i in idx]
    return obs, boot


def contrast_stats(
    obs: dict[str, np.ndarray], boot: dict[str, np.ndarray], expr: list[tuple[str, float]],
    endpoint: str,
) -> dict | None:
    """
    计算一个线性对比（如差、差中差）的点估计、SE 与 MDE。

    Args:
        obs / boot: bootstrap_endpoints 的返回。
        expr      : [(method, 系数)] 线性组合，如差中差 = [(a,+1),(b,-1),(c,-1),(d,+1)]。
        endpoint  : ENDPOINTS 的键。

    Returns:
        {delta, se, ci, mde_80, mde_90}；任一方法缺失则返回 None。
    """
    if any(m not in obs for m, _ in expr):
        return None
    j = list(ENDPOINTS).index(endpoint)
    delta = float(sum(c * obs[m][j] for m, c in expr))
    dist = sum(c * boot[m][:, j] for m, c in expr)
    se = float(np.std(dist, ddof=1))
    lo, hi = (float(x) for x in np.percentile(dist, [2.5, 97.5]))
    return {"delta": delta, "se": se, "ci": [lo, hi],
            **{f"mde_{k}": (Z_ALPHA + z) * se for k, z in Z_POWER.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description="实验 G 的 worst-group MDE（averaging + cluster bootstrap）")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--datasets", nargs="+", default=["HAM10000", "Fitzpatrick"])
    ap.add_argument("--out", default=str(OUTPUTS / "analysis" / "g_fusion" / "mde.json"))
    args = ap.parse_args()

    result: dict = {}
    for spec in SPECS:
        if spec.name not in args.datasets:
            continue
        cfg = json.loads(spec.cfg.read_text())["config"]
        methods = {}
        for m in G_METHODS:
            if m in cfg and (fl := load_folds(spec, m, cfg[m])) is not None:
                methods[m] = fl
        print(f"\n{'=' * 92}\n{spec.name}  可用方法: {list(methods)}\n{'=' * 92}")
        if "erm" not in methods:
            print("  [跳过] 无 ERM")
            continue

        obs, boot = bootstrap_endpoints(spec, methods, args.n_boot)

        # 参照效应量：SWAD−ERM 是本项目唯一「显著且有实质幅度」的 worst-group 赢家，
        # 用它给「多大算真效应」定标尺。
        refs = {
            "SWAD − ERM (已知真效应参照)":       [("swad", 1.0), ("erm", -1.0)],
            "HyperAdapt − ERM (HA 主效应)":      [("hyperadapt", 1.0), ("erm", -1.0)],
            "HyperAdapt − SWAD":                 [("hyperadapt", 1.0), ("swad", -1.0)],
        }
        # Δ_int 的同结构四臂代理（HA+SWAD 未落盘时用）；若已落盘则直接算真 Δ_int。
        proxies = {
            "代理 Δint: (HyperFusion−SWAD)−(HA−ERM)":
                [("hyperfusion", 1.0), ("swad", -1.0), ("hyperadapt", -1.0), ("erm", 1.0)],
            "代理 Δint: (HyperHead−SWAD)−(HA−ERM)":
                [("hyperhead", 1.0), ("swad", -1.0), ("hyperadapt", -1.0), ("erm", 1.0)],
            "真 Δint: (HA+SWAD−SWAD)−(HA−ERM)":
                [("hyperadapt_swad", 1.0), ("swad", -1.0), ("hyperadapt", -1.0), ("erm", 1.0)],
        }

        ds_res: dict = {"n": int(len(np.concatenate([f["y"] for f in methods["erm"]])))}
        for endpoint in ENDPOINTS:
            star = " ★主要终点" if endpoint == PRIMARY else ""
            print(f"\n--- 终点: {endpoint}{star} ---")
            print(f"  {'对比':<42s} {'Δ':>9s} {'SE':>8s} {'MDE(80%)':>10s} {'MDE(90%)':>10s}")
            for label, expr in {**refs, **proxies}.items():
                st = contrast_stats(obs, boot, expr, endpoint)
                if st is None:
                    continue
                print(f"  {label:<42s} {st['delta']:>+9.4f} {st['se']:>8.4f} "
                      f"{st['mde_80']:>10.4f} {st['mde_90']:>10.4f}")
                ds_res.setdefault(endpoint, {})[label] = st

            # --- 结构估计（比上面的四臂代理更贴切）---------------------------------
            # 恒等式：Δint ≡ (HA+SWAD − HA) − (SWAD − ERM)，即「SWAD 给 HA 的增量」减
            # 「SWAD 给 ERM 的增量」。两项都是**派生臂 − 自身基座**的配对差（派生臂由基座自身的
            # 权重轨迹平均而来 ⇒ 高度正相关、SE 远小于两个独立训练模型之差）。
            # 实测即印证：SE(SWAD−ERM) 明显小于 SE(HA−ERM)。故以 s_d = SE(SWAD−ERM) 作为
            # **单个派生增量**的 SE 尺度，再按两项间的相关性 ρ 给出 Δint 的 SE 区间：
            #     SE(Δint) = s_d · sqrt(2(1−ρ))
            # ρ=0（两个增量互不相关）给上界；ρ=0.5（同一 SWAD 机制作用于两种架构，正相关）给中值。
            # ⚠️ 仍是估计：真 SE 须待 HA+SWAD 落盘后由「真 Δint」行给出。
            sd = ds_res[endpoint].get("SWAD − ERM (已知真效应参照)", {}).get("se")
            if sd is not None:
                print(f"  {'结构估计 SE(Δint)=s_d·√(2(1−ρ)), s_d=SE(SWAD−ERM)':<42s}"
                      f" {'':>9s} {'':>8s}")
                for rho in (0.0, 0.5):
                    se_est = sd * np.sqrt(2 * (1 - rho))
                    m80, m90 = (Z_ALPHA + Z_POWER["80"]) * se_est, (Z_ALPHA + Z_POWER["90"]) * se_est
                    print(f"    ρ={rho:.1f}{'':<36s} {'':>9s} {se_est:>8.4f} {m80:>10.4f} {m90:>10.4f}")
                    ds_res[endpoint][f"结构估计 Δint (rho={rho})"] = {
                        "se": se_est, "mde_80": m80, "mde_90": m90}
        result[spec.name] = ds_res

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已落盘 → {out}")


if __name__ == "__main__":
    main()
