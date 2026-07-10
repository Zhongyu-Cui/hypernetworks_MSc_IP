"""
模型选择策略探针（Overall / Minimax-Pareto / DTO）+ 选择信号有效性验证
===================================================================
在**已落盘预测**上（零训练）回答两问：
  (1) MEDFAIR 的三种选择策略（Overall / Minimax-Pareto / DTO，均在 val 子群 AUC 上）在 HAM 上
      是否选出不同 config、各自的 test 公平如何——复刻 MEDFAIR Fig 4 的对比结构。
  (2) **选择信号有效性**：跨 6 config，val 上的各选择信号（overall / canonical-worst / marginal-worst /
      DTO 距离）与 test worst-group 的**秩相关**——检验"哪个 val 信号真能预测 test 公平"。

口径：HAM10000，seed42 搜索阶段的 6 config（单-split，每 config 有 val + test 预测）。
子群向量用项目权威 `subgroup_auc_vector`（canonical 含 sex×age joint；marginal = sex/age 6 组）。
纯后处理；作为"选择策略/信号该如何配置"讨论的实证依据，非正式结果表。

用法：
  python -m scripts.selection_strategy_probe                # 单-split（seed42 6 config）
  python -m scripts.selection_strategy_probe --mode cv      # 全-CV（6 config × 5 折；折均选择 + pooled-OOF）

--mode cv 需先跑 slurm/c1_ham_cv_search.sh（6 config × 5 折）。它做的正是「fold-mean 折均能否把单-split
的负秩相关救成正」这一关键验证：选择信号用**折均 val**（√5 降噪），test 目标用**5 折 OOF 池化**（n≈9707，
评估-n 地板已抬）。两端同时降噪后若秩相关转正 = 选择信号在 CV 下可用。
"""

from __future__ import annotations

import argparse
import glob
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from src.training.harness.subgroup_auc import subgroup_auc_vector

PRED = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/ham10000/predictions")
PRED_CV = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/ham10000/cv5/predictions")
DATASET = "ham10000"
METHODS = ("erm", "hyperhead", "hyperfusion", "hyperadapt")
SEED = 42
FOLDS = (0, 1, 2, 3, 4)   # CV：fold↔seed = 42+fold


def _load_vector(npz_path: Path) -> "OrderedDict[str, tuple[float | None, int]]":
    """从预测 npz 算权威子群 AUC 向量（复用训练同款 subgroup_auc_vector）。"""
    d = np.load(npz_path, allow_pickle=True)
    return subgroup_auc_vector(
        DATASET, d["y_true"], d["y_score"],
        sex=d["attr_sex"], age=d["attr_age"],
    )


def _overall_auc(npz_path: Path) -> float:
    """整体 AUC（不分组）。"""
    from sklearn.metrics import roc_auc_score
    d = np.load(npz_path, allow_pickle=True)
    return float(roc_auc_score(d["y_true"], d["y_score"]))


def _worst(vec: "OrderedDict", marginal: bool) -> float:
    """worst-case AUC。marginal=True 仅取边缘键（无 '|'），隔离 joint 小格噪声。"""
    aucs = [v[0] for k, v in vec.items() if v[0] is not None and (("|" not in k) if marginal else True)]
    return float(min(aucs)) if aucs else float("nan")


def _canonical_keys(vec: "OrderedDict") -> list[str]:
    """canonical 子群键（全部，含 joint），用于 Pareto/DTO 的向量维度。"""
    return [k for k, v in vec.items() if v[0] is not None]


def _config_candidates(method: str) -> "list[tuple[str, dict]]":
    """收集某方法 6 config 的 val/test 指标；返回 [(config_tag, metrics)]。"""
    out = []
    for vp in sorted(glob.glob(str(PRED / f"{method}_lr*_seed{SEED}_val_overall.npz"))):
        tag = re.search(rf"{method}_(lr.*?)_seed{SEED}_val_overall.npz", vp).group(1)
        tp = PRED / f"{method}_{tag}_seed{SEED}_overall.npz"
        if not tp.exists():
            continue
        vval = _load_vector(Path(vp))
        vtest = _load_vector(tp)
        out.append((tag, {
            "val_vec": vval, "test_vec": vtest,
            "val_overall": _overall_auc(Path(vp)), "test_overall": _overall_auc(tp),
            "val_cworst": _worst(vval, marginal=False), "val_mworst": _worst(vval, marginal=True),
            "test_cworst": _worst(vtest, marginal=False), "test_mworst": _worst(vtest, marginal=True),
        }))
    return out


def _foldmean_vector(vecs: list) -> "OrderedDict[str, tuple[float | None, int]]":
    """把多个折的 val 子群向量按 key 求折均 AUC（缺该折则跳过），返回同结构向量。"""
    keys = list(vecs[0].keys())
    out: "OrderedDict[str, tuple[float | None, int]]" = OrderedDict()
    for k in keys:
        aucs = [v[k][0] for v in vecs if k in v and v[k][0] is not None]
        ns = [v[k][1] for v in vecs if k in v]
        out[k] = (float(np.mean(aucs)) if aucs else None, int(np.sum(ns)))
    return out


def _cv_config_candidates(method: str) -> "list[tuple[str, dict]]":
    """
    CV 版候选：每 config 取 5 折 → 折均 val 向量（选择信号）+ 5 折 OOF 池化 test（评估目标）。
    只保留 5 折齐全的 config。返回 [(config_tag, metrics)]，metrics 结构对齐单-split 版。
    """
    # 发现该方法在 cv5 下出现过的所有 config tag
    tags = sorted({
        re.search(rf"{method}_(lr.*?)_seed\d+_val_overall.npz", Path(p).name).group(1)
        for p in glob.glob(str(PRED_CV / f"{method}_lr*_seed*_val_overall.npz"))
    })
    out = []
    for tag in tags:
        val_vecs, test_ys, test_ss, test_sex, test_age = [], [], [], [], []
        ok = True
        for fold in FOLDS:
            seed = 42 + fold
            vp = PRED_CV / f"{method}_{tag}_seed{seed}_val_overall.npz"
            tp = PRED_CV / f"{method}_{tag}_seed{seed}_overall.npz"
            if not (vp.exists() and tp.exists()):
                ok = False
                break
            val_vecs.append(_load_vector(vp))
            td = np.load(tp, allow_pickle=True)
            test_ys.append(td["y_true"]); test_ss.append(td["y_score"])
            test_sex.append(td["attr_sex"]); test_age.append(td["attr_age"])
        if not ok:
            continue
        from sklearn.metrics import roc_auc_score
        # 折均 val 向量（选择信号：逐子群 AUC 跨 5 折平均，√5 降噪）
        fm_val = _foldmean_vector(val_vecs)
        # val 整体 AUC = 各折 val overall 的折均（比在折均向量上重算稳）
        val_overall = float(np.mean([roc_auc_score(v["y_true"], v["y_score"]) for v in
                                     (np.load(PRED_CV / f"{method}_{tag}_seed{42 + f}_val_overall.npz",
                                              allow_pickle=True) for f in FOLDS)]))
        # 5 折 disjoint test 池化成 OOF（评估目标：n≈9707，评估-n 地板已抬）
        y = np.concatenate(test_ys); s = np.concatenate(test_ss)
        sex = np.concatenate(test_sex); age = np.concatenate(test_age)
        pooled_vec = subgroup_auc_vector(DATASET, y, s, sex=sex, age=age)
        out.append((tag, {
            "val_vec": fm_val, "test_vec": pooled_vec,
            "val_overall": val_overall, "test_overall": float(roc_auc_score(y, s)),
            "val_cworst": _worst(fm_val, marginal=False), "val_mworst": _worst(fm_val, marginal=True),
            "test_cworst": _worst(pooled_vec, marginal=False), "test_mworst": _worst(pooled_vec, marginal=True),
        }))
    return out


def _pareto_pick(cands: list) -> str:
    """Minimax-Pareto：在 canonical 子群向量的 Pareto 前沿里选 val worst-case 最高者。"""
    keys = _canonical_keys(cands[0][1]["val_vec"])
    vecs = {tag: np.array([m["val_vec"][k][0] for k in keys]) for tag, m in cands}
    front = []
    for tag, v in vecs.items():
        dominated = any((vecs[o] >= v).all() and (vecs[o] > v).any() for o in vecs if o != tag)
        if not dominated:
            front.append(tag)
    # 前沿里 val worst-case 最高；平手用 val overall 破
    md = dict(cands)
    return max(front, key=lambda t: (md[t]["val_cworst"], md[t]["val_overall"]))


def _dto_pick(cands: list) -> tuple[str, dict]:
    """DTO：到 val 乌托邦点（各子群跨 config 的最大 AUC）的归一化欧氏距离最小者。"""
    keys = _canonical_keys(cands[0][1]["val_vec"])
    M = np.array([[m["val_vec"][k][0] for k in keys] for _, m in cands])   # [n_cfg, n_sub]
    utopia = M.max(axis=0)
    # 逐子群 min-max 归一化（跨 config），避免某子群量纲主导
    rng = np.maximum(M.max(axis=0) - M.min(axis=0), 1e-8)
    Mn = (M - M.min(axis=0)) / rng
    un = (utopia - M.min(axis=0)) / rng
    dist = np.sqrt(((un - Mn) ** 2).sum(axis=1))
    dto = {cands[i][0]: float(dist[i]) for i in range(len(cands))}
    return cands[int(dist.argmin())][0], dto


def main() -> None:
    """三选择器对比 + 选择信号秩相关（单-split 或全-CV 折均）。"""
    ap = argparse.ArgumentParser(description="模型选择策略探针 + 选择信号有效性验证")
    ap.add_argument("--mode", choices=["single", "cv"], default="single",
                    help="single=单-split 6config(seed42)；cv=全-CV 6config×5折(折均 val + pooled OOF)")
    args = ap.parse_args()
    collect = _cv_config_candidates if args.mode == "cv" else _config_candidates
    header = ("HAM10000 · 6-config 全-CV（折均 val 选择 + 5 折 OOF 池化 test）· 三策略对比"
              if args.mode == "cv" else
              "HAM10000 · 6-config 单-split（seed42）· 三种模型选择策略对比（test 指标）")
    print("=" * 92)
    print(header)
    print("=" * 92)
    print(f"{'method':12s}{'strategy':10s}{'pick config':18s}{'test overall':>13}{'test c-worst':>13}{'test m-worst':>13}")
    pooled = []  # 收集所有 method×config 点做秩相关
    for method in METHODS:
        cands = collect(method)
        if len(cands) < 2:
            print(f"{method:12s} <不足 2 config，跳过>")
            continue
        md = dict(cands)
        pick_overall = max(cands, key=lambda c: c[1]["val_overall"])[0]
        pick_pareto = _pareto_pick(cands)
        pick_dto, dto_scores = _dto_pick(cands)
        for strat, tag in [("Overall", pick_overall), ("Pareto", pick_pareto), ("DTO", pick_dto)]:
            m = md[tag]
            print(f"{method:12s}{strat:10s}{tag:18s}{m['test_overall']:13.4f}"
                  f"{m['test_cworst']:13.4f}{m['test_mworst']:13.4f}")
        print("-" * 92)
        for tag, m in cands:
            pooled.append({
                "val_overall": m["val_overall"], "val_cworst": m["val_cworst"], "val_mworst": m["val_mworst"],
                "val_dto_neg": -dto_scores[tag],   # 距离越小越好 → 取负使"越大越好"与其它信号同向
                "test_cworst": m["test_cworst"], "test_mworst": m["test_mworst"],
            })

    # ---- 选择信号有效性：val 信号 vs test worst-group 秩相关 ----
    print("\n" + "=" * 92)
    print(f"选择信号有效性（跨全部 {len(pooled)} 个 method×config 点，Spearman ρ；越接近 +1 越能预测 test 公平）")
    print("=" * 92)
    def col(k): return [p[k] for p in pooled]
    pairs = [
        ("val overall      → test canonical-worst", "val_overall", "test_cworst"),
        ("val canonical-worst → test canonical-worst", "val_cworst", "test_cworst"),
        ("val marginal-worst  → test marginal-worst ", "val_mworst", "test_mworst"),
        ("val DTO(−dist)    → test canonical-worst", "val_dto_neg", "test_cworst"),
        ("val marginal-worst  → test canonical-worst", "val_mworst", "test_cworst"),
    ]
    for label, vk, tk in pairs:
        rho, p = spearmanr(col(vk), col(tk))
        print(f"  {label:44s}  ρ={rho:+.3f}  (p={p:.3f})")
    print("\n读法：ρ 越高 = 该 val 信号越能预测 test 公平、越适合当选择信号；接近 0 或负 = 在选噪声。")


if __name__ == "__main__":
    main()
