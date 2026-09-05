"""
HAM10000 双属性对照臂 vs age-only 臂（averaging 口径配对 bootstrap）
====================================================================
回答的问题：HAM 的三个 HN 原本**只条件化 age**，而 worst-group 评估口径是 Sex / Age / Sex×Age
三套分组（见 `src/utils/ham10000_fairness.py`）——条件化口径与评估口径不对称，sex 轴与交叉格上的
最差子群，模型从未拿到对应属性。本脚本对比「条件输入补齐为评估分组变量全集」的 `<hn>_sexage` 臂
与原 age-only 臂，判定这个不对称是否解释了 HN 在 worst-group 上的打平结果。

口径与 `scripts/build_oof_results_averaging.py` **完全一致**（直接复用其 Spec / Views / clusters /
load_folds）：逐折算 AUC → 折间平均 → 再 min/max 取组；推断用**病灶级（lesion_id）配对 cluster
bootstrap**，两臂共用同一批重采样索引，故 Δ 的 CI 是配对 CI（消掉共同的抽样噪声）。

与 build_oof_results_averaging 的唯一差异：那里的参照基线是 ERM / SWAD / GroupDRO，本脚本把参照
换成**对应的 age-only HN**，即每对只差「条件输入是否含 sex」这一个变量。

运行（轻量 CPU）：
    python scripts/ham_sexage_vs_age_averaging.py
    python scripts/ham_sexage_vs_age_averaging.py --n-boot 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.build_oof_results_averaging import (
    OUTPUTS, SPECS, Views, clusters, load_folds,
)
from src.training.harness.hparam_grid import iter_hparam_grid

# (双属性臂, 对应 age-only 臂)：唯一差异是条件输入是否含 sex
PAIRS: tuple[tuple[str, str], ...] = (
    ("hyperhead_sexage", "hyperhead"),
    ("hyperfusion_sexage", "hyperfusion"),
    ("hyperadapt_sexage", "hyperadapt"),
)
METRIC_KEYS: tuple[str, ...] = ("overall", "canonical_worst", "marginal_worst", "gap")


def parse_args() -> argparse.Namespace:
    """解析命令行参数（--n-boot 配对 bootstrap 重采样次数）。"""
    parser = argparse.ArgumentParser(description="HAM sexage 臂 vs age-only 臂（averaging 配对 bootstrap）")
    parser.add_argument("--n-boot", type=int, default=1000, help="bootstrap 重采样次数（默认 1000）")
    parser.add_argument("--write-json", type=str, default=None, help="结果落盘路径（可选）")
    parser.add_argument("--config-matched", action="store_true",
                        help="额外跑 **config-matched 敏感性**：两臂各自按协议选 config 时，Δ 里混进了"
                             "超参差异；本模式逐 config（6 个）在**同一超参**下比两臂，只看点估计的"
                             "符号一致性，判断 Δ 是条件输入造成的还是选参造成的。")
    return parser.parse_args()


def config_matched_report(spec, cfg_tags: tuple[str, ...] | None = None) -> None:
    """
    逐 config 在同一超参下比较两臂（点估计，无 bootstrap）。

    Args:
        spec     : HAM10000 的 regime 规格（复用 build_oof_results_averaging.SPECS[0]）。
        cfg_tags : 要比的 config tag；缺省=超参网格全部 6 个。
    """
    if cfg_tags is None:
        cfg_tags = tuple(c.tag for c in iter_hparam_grid())
    print(f"\n{'=' * 96}\nconfig-matched 敏感性（同一超参下 sexage − age-only，点估计）\n{'=' * 96}")
    print(f"{'方法对':<32s}{'config':<18s}{'ΔOverall':>10s}{'Δcanon_w':>10s}{'Δmarg_w':>10s}")
    for sexage, age_only in PAIRS:
        deltas: dict[str, list[float]] = {k: [] for k in ("overall", "canonical_worst", "marginal_worst")}
        for tag in cfg_tags:
            pair_obs = {}
            for method in (sexage, age_only):
                folds = load_folds(spec, method, tag)
                if folds is None:
                    break
                _, offs = clusters(spec, folds)
                pair_obs[method] = Views(spec, folds).compute(offs, np.ones(offs[-1]))
            if len(pair_obs) < 2:
                print(f"{sexage + ' − ' + age_only:<32s}{tag:<18s}  [缺预测，跳过]")
                continue
            d = [pair_obs[sexage][j] - pair_obs[age_only][j] for j in range(3)]
            for k, v in zip(deltas, d):
                deltas[k].append(v)
            print(f"{sexage + ' − ' + age_only:<32s}{tag:<18s}{d[0]:>+10.4f}{d[1]:>+10.4f}{d[2]:>+10.4f}")
        for key, vals in deltas.items():
            if vals:
                n_pos = sum(1 for v in vals if v > 0)
                print(f"{'  → ' + key:<44s}均值 {np.mean(vals):+.4f}  正号 {n_pos}/{len(vals)}")
        print()


def main() -> None:
    args = parse_args()
    spec = SPECS[0]                                   # HAM10000
    assert spec.name == "HAM10000", f"SPECS[0] 应为 HAM10000，实为 {spec.name}"
    cfg = json.loads(spec.cfg.read_text())["config"]

    # 载入两臂共 6 个方法的逐折预测（缺任一折即报错，避免静默半份数据）
    methods = [m for pair in PAIRS for m in pair]
    folds_by: dict[str, list[dict]] = {}
    for method in methods:
        folds = load_folds(spec, method, cfg[method])
        if folds is None:
            raise SystemExit(f"缺 {method} 的 5 折预测（config={cfg.get(method)}）")
        folds_by[method] = folds
    views = {m: Views(spec, f) for m, f in folds_by.items()}

    # 聚类键（lesion_id）与折偏移：由任一方法的折结构恢复，各臂样本集逐位一致
    cidx, offs = clusters(spec, folds_by["hyperadapt"])
    n_clusters = int(cidx.max() + 1)
    ones = np.ones(offs[-1])

    obs = {m: views[m].compute(offs, ones) for m in views}

    # 配对 bootstrap：所有方法共用同一 draw ⇒ Δ 的 CI 已消掉共同抽样噪声
    rng = np.random.default_rng(42)
    boot: dict[str, list[tuple]] = {m: [] for m in views}
    for _ in range(args.n_boot):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        weights = np.bincount(draw, minlength=n_clusters).astype(float)[cidx]
        for m in views:
            boot[m].append(views[m].compute(offs, weights))
    boot_arr = {m: np.array([r[:4] for r in boot[m]]) for m in views}

    print(f"HAM10000 averaging 口径 | n={offs[-1]:,} 张 / {n_clusters:,} 病灶 | "
          f"配对 cluster bootstrap B={args.n_boot}\n")
    print(f"{'方法':<20s}{'Overall':>9s}{'canon_w':>9s}{'marg_w':>9s}{'gap':>9s}   config")
    for method in methods:
        p = obs[method]
        print(f"{method:<20s}{p[0]:>9.4f}{p[1]:>9.4f}{p[2]:>9.4f}{p[3]:>9.4f}   {cfg[method]}")

    results: dict[str, dict] = {}
    print(f"\n{'对比 (sexage − age-only)':<34s}{'指标':<18s}{'Δ':>9s}   95%CI              显著?")
    for sexage, age_only in PAIRS:
        entry: dict[str, dict] = {}
        for j, key in enumerate(METRIC_KEYS):
            diff = boot_arr[sexage][:, j] - boot_arr[age_only][:, j]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            delta = float(obs[sexage][j] - obs[age_only][j])
            sig = bool(lo > 0 or hi < 0)
            entry[key] = {"delta": delta, "ci": [float(lo), float(hi)], "sig": sig}
            label = f"{sexage} − {age_only}" if j == 0 else ""
            print(f"{label:<34s}{key:<18s}{delta:>+9.4f}   [{lo:+.4f},{hi:+.4f}]   "
                  f"{'**显著**' if sig else 'n.s.'}")
        results[f"{sexage}_vs_{age_only}"] = entry

    if args.config_matched:
        config_matched_report(spec)

    if args.write_json is not None:
        out = Path(args.write_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n": int(offs[-1]), "n_clusters": n_clusters, "n_boot": args.n_boot,
            "config": {m: cfg[m] for m in methods},
            "point": {m: dict(zip(METRIC_KEYS, [float(v) for v in obs[m][:4]])) for m in methods},
            "vs": results,
        }
        out.write_text(json.dumps(payload, indent=2))
        print(f"\n已落盘 → {out}")


if __name__ == "__main__":
    main()
