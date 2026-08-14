"""
在 **averaging** 口径下重算全部 CV-OOF 结果（5 ID 数据集 + 2 OOD 方向）
=======================================================================
用于把 `docs/oof_regime_results.md` 从 **pooled-OOF** 口径整体重构到 **averaging** 口径。

**为什么换口径（文献依据）**：pooled AUC 把 5 折**不同模型**的分数拼起来排序，构造出**跨折正负对**，
要求各折模型分数尺度可比；而早停轮数异质使各折 logit 尺度漂移（HAM 实测跨折 logit 中位数差 19 个单位）。
这是**学界已知缺陷**：
  · **Parker, Günter & Bedo (2007)**, *BMC Bioinformatics* 8:326 —— pooled AUC 严重**负偏**
    （无信号数据 AUC 掉到 <0.3 而非 0.5）；**排序类指标(AUC)受害远重于 accuracy**。
  · **Airola et al. (2010)**, *JMLR W&CP* 8:3–13 ——「pooling 假设各折分类器来自同一总体……计算 AUC 时
    该假设更可疑，**因为部分正负对由不同折的样本构成**」；推荐 **averaging** 或 LPOCV。
  · **Forman & Scholz (2010)**, *SIGKDD Explorations* 12(1):49–57。
本项目实测：污染程度**因方法系统性不同**（ERM −0.01~−0.05 vs C-Deep −0.19）⇒ 组间比较会测成漂移；
且曾据此产生高置信假阳性（Holm=0.006 的假「确认」，真值 p=0.70）。

**averaging 不损失评估 n**：抬 n 的功劳来自 **CV 本身**（每样本轮到一次 test），非 pooling；AUC 方差由
**类别样本数**支配（`Var≈c/n_pos+c'/n_neg`）而非「对数」，5 折各估再平均(÷5)与池化精度相当，
**实测 SE 反而小 8–28%**。LPOCV 需按对重训、算力不可行，故取 averaging。

**口径细则**：
  · **worst/best 用「先平均后 min/max」**：`min_g[mean_f AUC(g,f)]`。Jensen 恒有
    `mean_f[min_g …] ≤ min_g[mean_f …]`；选前者会让每折在最噪处取极值、放大 min 的向下偏，
    且「模型在哪组最弱」应是模型属性、不该每折换答案。
  · **子群 schema 取自 `harness.subgroup_auc.subgroup_masks`**（单一事实来源，杜绝定义漂移）：
    canonical = 全部子群（含 joint 交叉）；marginal = 仅单属性边缘组（排含 `|` 的 joint）。
  · **推断 = 患者/病灶级配对 cluster bootstrap**（全方法共用同一批重采样索引；只重采样 cluster）。
    Fitzpatrick 每 md5hash 唯一、无聚类结构 ⇒ 退化为样本级（正确）。
  · ⚠️ **averaging 的已知代价（Airola 明示，本脚本如实报告）**：小 joint 格可能某折只剩一类 ⇒ 该折
    AUC 无定义。处理 = **跳过该折、用其余折平均**，并统计 `undefined_cells` 一并报出。
    实测 HAM joint(sex×age) 8 格×5 折 = 40 个中 **1 个**无定义（fold4 sex=Male&age=20-40，n=127/n_pos=0）。

运行（轻量 CPU）：
    python scripts/build_oof_results_averaging.py            # 全部
    python scripts/build_oof_results_averaging.py --dataset ham10000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.eaudit_m2c_full_target import sorted_view, weighted_auc
from src.training.harness.predictions import load_predictions
from src.training.harness.subgroup_auc import subgroup_masks

OUTPUTS = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs")
SPLITS = Path("/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/data/splits")
FOLD_SEEDS = (42, 43, 44, 45, 46)                 # fold k ↔ seed 42+k（既有 CV 布局）
# 末尾追加实验 G（HN×SWAD 融合）的融合臂——**追加而非插入**，既有方法的出现顺序不变。
# 该方法仅在 selected_configs.json 含其 key 且预测存在时才被载入，否则静默跳过（见 run_spec）。
METHODS = ("erm", "swad", "roc", "groupdro",
           "hyperhead", "hyperfusion", "hyperadapt", "hyperadapt_swad")
# 配对 bootstrap 的参照基线：新增 `groupdro`（训练时用子群标签的基线），使
# 「HN vs GroupDRO」= 同样用属性、只差 conditioning vs reweighting 的干净对比。
BASELINES = ("erm", "swad", "groupdro")
N_BOOT = 1000


class Spec:
    """一个 regime 的完整寻址与口径规格。"""

    def __init__(self, name: str, pred_rel: str, ds_key: str, cfg_rel: str,
                 split_sub: str | None, cluster_col: str | None, filter_age_neg1: bool) -> None:
        self.name = name
        self.pred_dir = OUTPUTS / pred_rel
        self.ds_key = ds_key                      # subgroup_masks 的数据集键
        self.cfg = OUTPUTS / cfg_rel / "selected_configs.json"
        self.split_sub = split_sub                # 聚类键来源（None = 无聚类结构）
        self.cluster_col = cluster_col
        self.filter_age_neg1 = filter_age_neg1    # HAM：排 age_group==-1


SPECS = [
    Spec("HAM10000", "ham10000/cv5/predictions", "ham10000", "ham10000/cv5",
         "ham10000", "lesion_id", True),
    Spec("PAPILA", "papila/predictions", "papila", "papila",
         "papila", "pid", False),
    Spec("Fitzpatrick", "fitzpatrick/cv5/predictions", "fitzpatrick", "fitzpatrick/cv5",
         None, None, False),
    Spec("MIMIC", "mimic_cxr/cv5/predictions", "mimic", "mimic_cxr/cv5",
         "mimic_cxr_nofinding", "patient_id", False),
    Spec("CheXpert", "chexpert_cxr/cv5/predictions", "chexpert", "chexpert_cxr/cv5",
         "chexpert_nofinding", "patient_id", False),
    Spec("M2C_OOD", "ood_cxr/mimic2chexpert/cv5/predictions", "chexpert", "mimic_cxr/cv5",
         "chexpert_nofinding", "patient_id", False),
    Spec("C2M_OOD", "ood_cxr/chexpert2mimic/cv5/predictions", "mimic", "chexpert_cxr/cv5",
         "mimic_cxr_nofinding", "patient_id", False),
]


def load_folds(spec: Spec, method: str, tag: str) -> list[dict] | None:
    """载入某方法 5 折预测（保持逐折结构）。缺任一折返回 None。"""
    out = []
    for seed in FOLD_SEEDS:
        p = spec.pred_dir / f"{method}_{tag}_seed{seed}_overall.npz"
        if not p.exists():
            return None
        y, s, a = load_predictions(p)
        y = np.asarray(y).astype(int)
        s = np.asarray(s, dtype=np.float64)
        if spec.filter_age_neg1:                  # HAM：ERM/SWAD 预测含 age=-1，HN 已在训练侧过滤
            keep = a["age"].astype(int) >= 0
            y, s = y[keep], s[keep]
            a = {k: v[keep] for k, v in a.items()}
        out.append({"y": y, "s": s, "attrs": a})
    return out


def build_masks(spec: Spec, attrs: dict) -> "dict[str, np.ndarray]":
    """用 harness 的单一事实来源构造子群掩码。"""
    kw = {k: np.asarray(v).astype(int) for k, v in attrs.items() if k in
          ("sex", "race", "age", "skin")}
    return subgroup_masks(spec.ds_key, **kw)


def clusters(spec: Spec, folds: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """按行序恢复聚类键并硬校验对齐；无聚类结构时退化为样本级。"""
    sizes = [len(f["y"]) for f in folds]
    offs = np.array([0] + list(np.cumsum(sizes)))
    if spec.split_sub is None or spec.cluster_col is None:
        return np.arange(offs[-1]), offs                    # 样本级
    keys = []
    for k, f in enumerate(folds):
        df = pd.read_csv(SPLITS / spec.split_sub / "cv5" / f"fold{k}" / "test.csv")
        if spec.filter_age_neg1:
            df = df[df["age_group"] >= 0].reset_index(drop=True)
        assert len(df) == len(f["y"]), f"{spec.name} fold{k}: CSV {len(df)} vs 预测 {len(f['y'])}"
        assert np.array_equal(df["label"].values.astype(int), f["y"]), f"{spec.name} fold{k}: label 不对齐"
        keys.append(df[spec.cluster_col].values)
    _, cidx = np.unique(np.concatenate(keys), return_inverse=True)
    return cidx, offs


class Views:
    """逐折预排序视图（**永不跨折拼接分数** —— averaging 免疫跨折校准漂移的原因）。"""

    def __init__(self, spec: Spec, folds: list[dict]) -> None:
        self.groups: list[str] = list(build_masks(spec, folds[0]["attrs"]).keys())
        self.v: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        for k, f in enumerate(folds):
            self.v[(k, "__all__")] = sorted_view(f["y"], f["s"], np.ones(len(f["y"]), bool))
            for g, m in build_masks(spec, f["attrs"]).items():
                self.v[(k, g)] = sorted_view(f["y"], f["s"], m)
        self.n_folds = len(folds)

    def compute(self, offs: np.ndarray, w: np.ndarray) -> tuple[float, float, float, float, int]:
        """
        averaging：逐折算 → 折间平均 → 再 min/max 取组。

        Returns:
            (Overall, canonical worst, marginal worst, gap(canonical), 无定义折-格计数)。
        """
        ov: list[float] = []
        per: dict[str, list[float]] = {g: [] for g in self.groups}
        undef = 0
        for k in range(self.n_folds):
            wk = w[offs[k]:offs[k + 1]]
            i, ys = self.v[(k, "__all__")]
            a = weighted_auc(ys, wk[i])
            if not np.isnan(a):
                ov.append(a)
            for g in self.groups:
                ig, yg = self.v[(k, g)]
                ag = weighted_auc(yg, wk[ig])
                if np.isnan(ag):
                    undef += 1                     # 该折该格只剩一类 ⇒ 跳过（Airola 提示的 averaging 代价）
                else:
                    per[g].append(ag)
        means = {g: float(np.mean(v)) for g, v in per.items() if v}
        marg = {g: v for g, v in means.items() if "|" not in g}
        canon_w = min(means.values())
        canon_b = max(means.values())
        return (float(np.mean(ov)), canon_w, min(marg.values()), canon_b - canon_w, undef)


def run_spec(spec: Spec, n_boot: int) -> dict | None:
    """跑一个 regime：点估计 + 配对 cluster bootstrap（参照基线见 BASELINES：ERM / SWAD / GroupDRO）。"""
    if not spec.pred_dir.exists() or not spec.cfg.exists():
        print(f"[跳过] {spec.name}: 缺预测或 config")
        return None
    cfg = json.loads(spec.cfg.read_text())["config"]
    views: dict[str, Views] = {}
    folds_by: dict[str, list[dict]] = {}
    for me in METHODS:
        if me not in cfg:
            continue
        fl = load_folds(spec, me, cfg[me])
        if fl is None:
            continue
        folds_by[me] = fl
        views[me] = Views(spec, fl)
    if "erm" not in views:
        print(f"[跳过] {spec.name}: 无 ERM")
        return None
    cidx, offs = clusters(spec, folds_by["erm"])
    n_cl = int(cidx.max() + 1)
    ones = np.ones(offs[-1])

    obs = {m: views[m].compute(offs, ones) for m in views}
    rng = np.random.default_rng(42)
    boot: dict[str, list[tuple]] = {m: [] for m in views}
    for _ in range(n_boot):
        draw = rng.integers(0, n_cl, size=n_cl)
        w = np.bincount(draw, minlength=n_cl).astype(float)[cidx]
        for m in views:
            boot[m].append(views[m].compute(offs, w))
    B = {m: np.array([r[:4] for r in boot[m]]) for m in views}

    res = {"n": int(offs[-1]), "n_clusters": n_cl, "config": {m: cfg[m] for m in views},
           "point": {m: {"overall": obs[m][0], "canonical_worst": obs[m][1],
                         "marginal_worst": obs[m][2], "gap": obs[m][3],
                         "undefined_fold_cells": obs[m][4]} for m in views},
           "vs": {}}
    for base in BASELINES:
        if base not in views:
            continue
        for m in views:
            if m == base:
                continue
            e = {}
            for j, key in enumerate(("overall", "canonical_worst", "marginal_worst", "gap")):
                d = B[m][:, j] - B[base][:, j]
                lo, hi = np.percentile(d, [2.5, 97.5])
                e[key] = {"delta": float(obs[m][j] - obs[base][j]), "ci": [float(lo), float(hi)],
                          "sig": bool(lo > 0 or hi < 0)}
            res["vs"][f"{m}_vs_{base}"] = e
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="averaging 口径重算全部 CV-OOF 结果")
    ap.add_argument("--dataset", default=None, help="只跑某 regime（缺省=全部）")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    result_path = OUTPUTS / "conditioning_ablation" / "oof_results_averaging.json"
    # ⚠️ `--dataset X` 只重算一个 regime：必须以既有 JSON 为底做**合并**，否则落盘会把其余 6 个
    # regime 整体抹掉（本脚本原实现即如此，2026-08-11 补 GroupDRO 时踩到并修复）。
    out: dict = {}
    if args.dataset and result_path.exists():
        out = json.loads(result_path.read_text())
        print(f"[合并模式] 以既有 {len(out)} 个 regime 为底，只重算 {args.dataset}")
    for spec in SPECS:
        if args.dataset and spec.name.lower() != args.dataset.lower():
            continue
        print(f"\n{'=' * 96}\n{spec.name}\n{'=' * 96}")
        r = run_spec(spec, args.n_boot)
        if r is None:
            continue
        out[spec.name] = r
        print(f"n={r['n']:,}  clusters={r['n_clusters']:,}  config={r['config']}")
        print(f"| {'方法':<12s} | {'Overall':>8s} | {'canon worst':>11s} | {'marg worst':>10s} | "
              f"{'gap':>7s} | 无定义折-格")
        for m, p in r["point"].items():
            print(f"| {m:<12s} | {p['overall']:>8.4f} | {p['canonical_worst']:>11.4f} | "
                  f"{p['marginal_worst']:>10.4f} | {p['gap']:>7.4f} | {p['undefined_fold_cells']}")
        print(f"\n  {'比较':<26s} {'Δmarg worst':>12s} {'95%CI':>20s} {'显著?':>8s}")
        for k, v in r["vs"].items():
            mw = v["marginal_worst"]
            lo, hi = mw["ci"]
            print(f"  {k:<26s} {mw['delta']:>+12.4f} [{lo:>+7.4f},{hi:>+7.4f}] "
                  f"{'**显著**' if mw['sig'] else 'n.s.':>8s}")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已落盘 → {result_path}（含 regime：{list(out)}）")


if __name__ == "__main__":
    main()
