"""
复核：既有 pooled-OOF 结论在 **averaging** 口径下是否成立
=========================================================
E1 分析中实测发现「池化各折 raw logit 算一个 AUC」会被**跨折校准漂移**污染，且污染程度**因方法
系统性不同** ⇒ 组间比较可能测的是漂移而非真效应（HAM 上曾据此产生高置信假阳性：H2 假「确认」
Holm=0.006，真值 +0.0075/p=0.70）。这是**学界已知缺陷**：
  · Parker, Günter & Bedo (2007), *BMC Bioinformatics* 8:326 —— pooled AUC 严重负偏（无信号数据 <0.3）；
  · Airola et al. (2010), *JMLR W&CP* 8:3–13 ——「pooling 假设各折分类器来自同一总体……计算 AUC 时更可疑，
    **因为部分正负对由不同折的样本构成**」，推荐 **averaging**；
  · Forman & Scholz (2010), *SIGKDD Explorations* 12(1):49–57。

`docs/oof_regime_results.md` 的**全部 ID 结果**与 **M→C legacy k→k** 均建立在池化口径上，故须复核。
本脚本对每个数据集、每个方法的**选定 config**，用**同一批预测**分别计算：
  · **pooled**（旧）：5 折预测拼接 → 一个 AUC；
  · **averaging**（新主口径）：逐折算 AUC → 折间平均（**无跨折正负对**，免疫漂移）；
并报「HN − ERM」的效应在两口径下是否**变号/改变结论**。

> 既有预测的布局是 `seed=42+fold`（fold↔seed 绑定，见 cv_oof_report.FOLD_SEEDS），每 (method,config)
> 恰好 5 条预测、每条对应一折的 test ⇒ 正是 averaging 所需结构，无需重训。
> **诊断量**：一并报「跨折 logit 中位数 SD」——它是漂移强度的直接度量，可解释哪些方法受害最重。

运行（轻量 CPU）：
    python scripts/review_pooled_vs_averaging.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUTPUTS = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs")
FOLD_SEEDS = (42, 43, 44, 45, 46)          # fold k ↔ seed 42+k（既有 CV 搜索约定）
METHODS = ("erm", "swad", "hyperhead", "hyperfusion", "hyperadapt")

# 数据集 → (预测目录, 边缘候选组构造器)。候选组 = 单属性边缘组（排 joint），与 oof_regime_results 的
# "marginal worst" 口径一致（**注意**：这与条件化范围消融 E1 的「仅 age 4 组」不同，此处复核的是既有结论）。
DATASETS = {
    "ham10000": ("ham10000/cv5/predictions", "ham"),
    "fitzpatrick": ("fitzpatrick/cv5/predictions", "fitz"),
    "mimic": ("mimic_cxr/cv5/predictions", "cxr"),
    "chexpert": ("chexpert_cxr/cv5/predictions", "cxr"),
    "m2c_ood": ("ood_cxr/mimic2chexpert/cv5/predictions", "cxr"),
}
CONFIG_SRC = {"ham10000": "ham10000", "fitzpatrick": "fitzpatrick",
              "mimic": "mimic_cxr", "chexpert": "chexpert_cxr", "m2c_ood": "mimic_cxr"}

# 聚类键来源：split 目录 + 列名。**cluster bootstrap 必需**（同患者/病灶多图相关，样本级会反保守）。
# Fitzpatrick 每 md5hash 唯一、无患者结构 ⇒ 无聚类（样本级即正确）。
# 注：ERM/SWAD 的 HAM 预测含 age=-1（n=1990），HN 已在训练侧过滤（n=1943）；load_folds 对 HAM 统一
# 过滤 age>=0 后，两者与 CSV 逐元素对齐（已断言），故 lesion_id 可按位置恢复。
CLUSTER_SRC: dict[str, tuple[str, str] | None] = {
    "ham10000": ("ham10000", "lesion_id"),
    "fitzpatrick": None,
    "mimic": ("mimic_cxr_nofinding", "patient_id"),
    "chexpert": ("chexpert_nofinding", "patient_id"),
    "m2c_ood": ("chexpert_nofinding", "patient_id"),   # target=CheXpert
}
SPLITS = Path("/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/data/splits")
N_BOOT = 1000


def marginal_masks(kind: str, attrs: dict) -> dict[str, np.ndarray]:
    """构造**边缘**子群掩码（排 joint 交叉），与既有 marginal-worst 口径一致。"""
    m: dict[str, np.ndarray] = {}
    if kind == "ham":
        for v, n in {0: "Male", 1: "Female"}.items():
            m[f"sex:{n}"] = attrs["sex"].astype(int) == v
        for v, n in {0: "20-40", 1: "40-60", 2: "60-80", 3: "80+"}.items():
            m[f"age:{n}"] = attrs["age"].astype(int) == v
    elif kind == "fitz":
        for v in range(6):
            m[f"skin:{v}"] = attrs["skin"].astype(int) == v
    else:  # cxr
        for v, n in {0: "Male", 1: "Female"}.items():
            m[f"sex:{n}"] = attrs["sex"].astype(int) == v
        for v, n in {0: "White", 1: "Non-White"}.items():
            m[f"race:{n}"] = attrs["race"].astype(int) == v
        for v, n in {0: "<60", 1: ">=60"}.items():
            m[f"age:{n}"] = attrs["age"].astype(int) == v
    return m


def _sel(method: str) -> str:
    return "averaged" if False else "overall"          # 既有预测：SWAD 亦落 overall 槽


def load_folds(pred_dir: Path, method: str, tag: str, kind: str) -> list[dict] | None:
    """载入某方法 5 折预测（保持逐折结构）。缺文件返回 None。"""
    from src.training.harness.predictions import load_predictions
    out = []
    for seed in FOLD_SEEDS:
        p = pred_dir / f"{method}_{tag}_seed{seed}_{_sel(method)}.npz"
        if not p.exists():
            return None
        y, s, a = load_predictions(p)
        y = np.asarray(y).astype(int); s = np.asarray(s, float)
        if kind == "ham":                              # 既有 HAM OOF 口径：排 age_group==-1
            keep = a["age"].astype(int) >= 0
            y, s = y[keep], s[keep]
            a = {k: v[keep] for k, v in a.items()}
        out.append({"y": y, "s": s, "attrs": a})
    return out


def metrics(folds: list[dict], kind: str) -> tuple[float, float, float, float, float]:
    """
    返回 (pooled Overall, pooled marg-worst, avg Overall, avg marg-worst, 跨折 logit 中位数 SD)。

    · pooled  ：5 折预测拼接后算一个 AUC（旧口径，含跨折正负对）；
    · averaging：逐折算 AUC → 折间平均（新主口径，无跨折对）。worst 取「先平均后 min」。
    """
    # pooled
    Y = np.concatenate([f["y"] for f in folds])
    S = np.concatenate([f["s"] for f in folds])
    A = {k: np.concatenate([f["attrs"][k] for f in folds]) for k in folds[0]["attrs"]}
    pm = marginal_masks(kind, A)
    p_ov = roc_auc_score(Y, S)
    p_w = min(roc_auc_score(Y[m], S[m]) for m in pm.values()
              if m.sum() and len(np.unique(Y[m])) > 1)
    # averaging（先平均后 min）
    ov, per = [], {}
    for f in folds:
        ov.append(roc_auc_score(f["y"], f["s"]))
        for g, m in marginal_masks(kind, f["attrs"]).items():
            if m.sum() and len(np.unique(f["y"][m])) > 1:
                per.setdefault(g, []).append(roc_auc_score(f["y"][m], f["s"][m]))
    a_ov = float(np.mean(ov))
    a_w = float(min(np.mean(v) for v in per.values()))
    drift = float(np.std([np.median(f["s"]) for f in folds]))
    return p_ov, p_w, a_ov, a_w, drift


def load_clusters(ds: str, kind: str, folds: list[dict]) -> tuple[np.ndarray, np.ndarray] | None:
    """
    按行序从 fold test.csv 恢复聚类键（patient_id / lesion_id），并硬校验对齐。

    Returns:
        (cluster_idx [N], offsets [K+1])；无聚类结构（Fitzpatrick）返回 None。
    """
    src = CLUSTER_SRC[ds]
    if src is None:
        return None
    sub, col = src
    keys, offs = [], [0]
    for k, f in enumerate(folds):
        df = pd.read_csv(SPLITS / sub / "cv5" / f"fold{k}" / "test.csv")
        if kind == "ham":
            df = df[df["age_group"] >= 0].reset_index(drop=True)
        assert len(df) == len(f["y"]), f"{ds} fold{k}: CSV {len(df)} vs 预测 {len(f['y'])} 行数不符"
        assert np.array_equal(df["label"].values.astype(int), f["y"]), f"{ds} fold{k}: label 不对齐"
        keys.append(df[col].values)
        offs.append(offs[-1] + len(df))
    _, cidx = np.unique(np.concatenate(keys), return_inverse=True)
    return cidx, np.array(offs)


def _avg_metrics_w(folds: list[dict], kind: str, offs: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """给定全局样本权重，算 averaging 口径的 (Overall, marginal worst)（先平均后 min）。"""
    from scripts.eaudit_m2c_full_target import sorted_view, weighted_auc
    ov, per = [], {}
    for k, f in enumerate(folds):
        wk = w[offs[k]:offs[k + 1]]
        i, ys = sorted_view(f["y"], f["s"], np.ones(len(f["y"]), bool))
        a = weighted_auc(ys, wk[i])
        if not np.isnan(a):
            ov.append(a)
        for g, m in marginal_masks(kind, f["attrs"]).items():
            ig, yg = sorted_view(f["y"], f["s"], m)
            ag = weighted_auc(yg, wk[ig])
            if not np.isnan(ag):
                per.setdefault(g, []).append(ag)
    return float(np.mean(ov)), float(min(np.mean(v) for v in per.values()))


def significance(ds: str, kind: str, all_folds: dict[str, list[dict]], n_boot: int) -> dict:
    """
    averaging 口径下的 **配对 cluster bootstrap** 显著性（HN/SWAD − ERM，worst-group，双侧 95% CI）。

    所有方法共用**同一批重采样 cluster 索引**（配对）；只重采样 cluster、不重采样 seed。
    Fitzpatrick 无聚类结构 ⇒ 退化为样本级重采样（正确，因每 md5hash 唯一）。
    """
    ref = all_folds["erm"]
    cl = load_clusters(ds, kind, ref)
    if cl is None:                                    # 无聚类 ⇒ 样本级
        n = sum(len(f["y"]) for f in ref)
        cidx = np.arange(n)
        offs = np.array([0] + list(np.cumsum([len(f["y"]) for f in ref])))
    else:
        cidx, offs = cl
    n_cl = int(cidx.max() + 1)
    rng = np.random.default_rng(42)
    boot: dict[str, list[float]] = {m: [] for m in all_folds}
    for _ in range(n_boot):
        draw = rng.integers(0, n_cl, size=n_cl)
        w = np.bincount(draw, minlength=n_cl).astype(float)[cidx]
        for m, fl in all_folds.items():
            boot[m].append(_avg_metrics_w(fl, kind, offs, w)[1])
    out = {}
    for m in all_folds:
        if m == "erm":
            continue
        d = np.array(boot[m]) - np.array(boot["erm"])
        obs = _avg_metrics_w(all_folds[m], kind, offs, np.ones(offs[-1]))[1] - \
              _avg_metrics_w(ref, kind, offs, np.ones(offs[-1]))[1]
        lo, hi = np.percentile(d, [2.5, 97.5])
        out[m] = {"delta": float(obs), "ci": [float(lo), float(hi)],
                  "sig": bool(lo > 0 or hi < 0)}
    return out


def main() -> None:
    print("复核：既有 pooled-OOF 结论在 averaging 口径下是否成立")
    print("依据：Parker2007 / Airola2010 —— pooled AUC 有悲观负偏（跨折正负对，各折模型尺度不可比）\n")
    summary: dict = {}
    for ds, (rel, kind) in DATASETS.items():
        pred_dir = OUTPUTS / rel
        cfg_path = OUTPUTS / CONFIG_SRC[ds] / "cv5" / "selected_configs.json"
        if not pred_dir.exists() or not cfg_path.exists():
            print(f"[跳过] {ds}: 缺预测或 config")
            continue
        cfg = json.loads(cfg_path.read_text())["config"]
        print(f"{'=' * 100}\n{ds}   （config 源: {CONFIG_SRC[ds]}）\n{'=' * 100}")
        print(f"| {'方法':<12s} | {'pooled Ov':>9s} {'pooled worst':>12s} | "
              f"{'avg Ov':>7s} {'avg worst':>9s} | {'worst 差(avg−pool)':>17s} | {'跨折logitSD':>11s} |")
        rows: dict = {}
        all_folds: dict[str, list[dict]] = {}
        for me in METHODS:
            if me not in cfg:
                continue
            folds = load_folds(pred_dir, me, cfg[me], kind)
            if folds is None:
                print(f"| {me:<12s} | {'缺预测':>9s} |")
                continue
            all_folds[me] = folds
            p_ov, p_w, a_ov, a_w, drift = metrics(folds, kind)
            rows[me] = {"pooled_overall": p_ov, "pooled_worst": p_w,
                        "avg_overall": a_ov, "avg_worst": a_w, "fold_logit_sd": drift}
            print(f"| {me:<12s} | {p_ov:>9.4f} {p_w:>12.4f} | {a_ov:>7.4f} {a_w:>9.4f} | "
                  f"{a_w - p_w:>+17.4f} | {drift:>11.2f} |")
        # HN − ERM 的效应在两口径下是否变号 / 改变结论；并给 averaging 口径的配对 cluster bootstrap
        if "erm" in rows:
            cl_desc = "样本级（无聚类结构）" if CLUSTER_SRC[ds] is None else f"{CLUSTER_SRC[ds][1]} 级"
            print(f"\n  配对 cluster bootstrap（averaging 口径，B={N_BOOT}，{cl_desc}，全方法共用同一批索引）")
            sig = significance(ds, kind, all_folds, N_BOOT)
            print(f"  {'HN/SWAD − ERM（worst）':<26s} {'pooled Δ':>9s} {'avg Δ':>9s} "
                  f"{'avg 95%CI':>20s} {'显著?':>7s}  变号?")
            for me in ("swad", "hyperhead", "hyperfusion", "hyperadapt"):
                if me not in rows:
                    continue
                dp = rows[me]["pooled_worst"] - rows["erm"]["pooled_worst"]
                da = sig[me]["delta"]
                lo, hi = sig[me]["ci"]
                flip = "⚠️ 变号" if np.sign(dp) != np.sign(da) and abs(dp) > 1e-4 and abs(da) > 1e-4 else ""
                rows[me]["avg_ci"] = sig[me]["ci"]
                rows[me]["avg_sig"] = sig[me]["sig"]
                print(f"  {me:<26s} {dp:>+9.4f} {da:>+9.4f} [{lo:>+7.4f},{hi:>+7.4f}] "
                      f"{'**显著**' if sig[me]['sig'] else '  n.s.':>7s}  {flip}")
        summary[ds] = rows
        print()
    p = OUTPUTS / "conditioning_ablation" / "review_pooled_vs_averaging.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已落盘 → {p}")


if __name__ == "__main__":
    main()
