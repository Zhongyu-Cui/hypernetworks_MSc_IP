"""
E2 主分析（M→C OOD）：full-target + 双侧 H1/H2 + 训练随机性稳健性
==================================================================
docs/conditioning_ablation_plan.md §3.3（2026-07-16 修订版）的落地。消费
`scripts/run_e2_ood_full_target.py` 落盘的 4 cell × 5 fold × 3 seed 预测（各评完整 CheXpert target）。

**估计量 = full-target 配对效应**：每 (fold,seed) 模型评完整 target、对 15 个 (f,s) **等权平均**
（无跨折池化 ⇒ 免疫 pooled-OOF 的跨折校准漂移伪影）。worst = **`min_g[mean_{f,s} AUC(g,f,s)]`**
（先平均后 min）。候选组 = E-audit.1 冻结的 M2C-OOD 6 边缘组（sex∪race∪age）。

**主分析推断 = 患者级配对 cluster bootstrap**（B=1000；全 cell 共用同一批重采样 patient；
只重采样 patient、不重采样 (f,s)）。

**判决（§3.3 修订，E2 双侧）**：
- **H1 = C-Deep − C-Head、H2 = C-Full − C-Deep**：报**双侧 95% CI** + family 内 **Holm**（对双侧 p）。
  三分：CI 全>0 且 Overall 非劣 ⇒「改善」；CI 全<0 ⇒「有害」；含 0 ⇒「不确定」（不得写「无效应」）。
- **Overall 非劣**：单侧 δ_NI=−0.005（方向性命题，保持单侧）。
- **历史正向复现族（已缩为 2 项）**：主 = **C-Full vs ERM**；次 = **C-Full vs SWAD**；双侧 + Holm。
  ⚠️「胜 SWAD」不可与「胜 ERM」等价（M→C 中 SWAD 本身劣于 ERM）。C-Deep vs ERM/SWAD 仅描述性。

**训练随机性稳健性分析（§3.3 强制并报）**：主 bootstrap 固定 15 个模型 ⇒ CI 是「给定这 15 个模型」的
条件性 CI，训练随机性不在内。故并报：① **fold-level 效应**（每 fold 内先平均 3 seed，得 5 个 fold 效应）
+ 符号一致性 + 范围；② **leave-one-fold-out** 是否变号。判读：若患者 bootstrap 显著但 fold 效应正负
混杂/去一折变号 ⇒ 只能称「目标样本层面显著，但训练稳定性不足」。

SWAD 基线：M→C 的 SWAD 预测来自既有 full-target（E-audit）目录，与本 condnet 评估**同一 target、同行序**
（均由 chexpert_full_target_test.csv 构建），可直接配对。

运行（轻量 CPU）：
    python scripts/e2_ood_analysis.py --n-boot 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.eaudit_m2c_full_target import sorted_view, weighted_auc
from src.training.harness.predictions import load_extra, load_predictions

OUTPUTS = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs")
E2_PRED = OUTPUTS / "conditioning_ablation" / "ood_e2_mimic2chexpert" / "predictions"
SWAD_PRED = OUTPUTS / "ood_cxr" / "mimic2chexpert" / "cv5_full_target" / "predictions"  # E-audit full-target
MANIFEST = OUTPUTS / "conditioning_ablation" / "eaudit_candidate_groups.json"

TAG = "lr1e-04_wd1e-03"                     # 四 cell 全选
CELLS = ("condnet_erm", "condnet_head", "condnet_deep", "condnet_full")
NICE = {"condnet_erm": "ERM", "condnet_head": "C-Head", "condnet_deep": "C-Deep", "condnet_full": "C-Full"}
N_FOLDS = 5
SEEDS = (42, 43, 44)
FOLD_SEEDS_SWAD = (42, 43, 44, 45, 46)      # SWAD full-target 用 seed=42+fold（E-audit 布局）
DELTA_NI = -0.005
AGE_THRESHOLD = 60


def load_candidate_groups() -> list[str]:
    m = json.loads(MANIFEST.read_text())["regimes"]["M2C-OOD"]
    return [g for g, c in m["groups"].items() if c["passes"]]


def _masks(attrs: dict, groups: list[str]) -> dict[str, np.ndarray]:
    """构造 6 边缘组掩码（sex/race/age，age 由 Dataset 已二值化）。"""
    names = {"sex": {0: "Male", 1: "Female"}, "race": {0: "White", 1: "Non-White"},
             "age": {0: "<60", 1: ">=60"}}
    out = {}
    for axis, mp in names.items():
        col = np.asarray(attrs[axis]).astype(int)
        for v, nm in mp.items():
            key = f"{axis}:{nm}"
            if key in groups:
                out[key] = col == v
    return out


def load_e2() -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """载入 E2 四 cell × 15 (f,s) 预测。Returns (y, patient_id, groups_masks, scores{(cell,f,s)})。"""
    y_ref = pid_ref = attrs_ref = None
    scores = {}
    for cell in CELLS:
        for k in range(N_FOLDS):
            for s in SEEDS:
                p = E2_PRED / f"{cell}_{TAG}_fold{k}_seed{s}.npz"
                y, sc, attrs = load_predictions(p)
                pid = load_extra(p)["patient_id"]
                if y_ref is None:
                    y_ref, pid_ref, attrs_ref = np.asarray(y).astype(int), pid, attrs
                else:
                    assert np.array_equal(np.asarray(y).astype(int), y_ref)
                    assert np.array_equal(pid, pid_ref)
                scores[(cell, k, s)] = np.asarray(sc, dtype=np.float64)
    groups = load_candidate_groups()
    return y_ref, pid_ref, _masks(attrs_ref, groups), scores


def load_swad(y_ref: np.ndarray, pid_ref: np.ndarray) -> dict:
    """载入 SWAD full-target 预测（5 个 fold，seed=42+fold）；对齐校验。Returns {(fold): score}。"""
    from src.training.run_ood_cxr_full_target import load_selected_config
    tag = load_selected_config("mimic")["swad"]
    out = {}
    for k, seed in enumerate(FOLD_SEEDS_SWAD):
        p = SWAD_PRED / f"swad_{tag}_seed{seed}_overall.npz"
        y, sc, _ = load_predictions(p)
        assert np.array_equal(np.asarray(y).astype(int), y_ref), f"SWAD {p.name} 与 target 不对齐"
        assert np.array_equal(load_extra(p)["patient_id"], pid_ref)
        out[k] = np.asarray(sc, dtype=np.float64)
    return out


class Views:
    """预排序视图：每 (cell,f,s) × {overall,各组} 折内排序。SWAD 亦纳入（key=('swad',k,None)）。"""

    def __init__(self, y, masks, scores, swad) -> None:
        self.masks = masks
        self.groups = list(masks)
        full = np.ones(len(y), bool)
        self.v = {}
        for key, sc in scores.items():
            self.v[(key, "__all__")] = sorted_view(y, sc, full)
            for g, m in masks.items():
                self.v[(key, g)] = sorted_view(y, sc, m)
        for k, sc in swad.items():
            kk = ("swad", k, None)
            self.v[(kk, "__all__")] = sorted_view(y, sc, full)
            for g, m in masks.items():
                self.v[(kk, g)] = sorted_view(y, sc, m)

    def _fs_metric(self, key, w) -> tuple[float, dict[str, float]]:
        """单 (cell,f,s) 的 (Overall, {组->AUC})。"""
        idx, ys = self.v[(key, "__all__")]
        ov = weighted_auc(ys, w[idx])
        per = {}
        for g in self.groups:
            ig, yg = self.v[(key, g)]
            a = weighted_auc(yg, w[ig])
            if not np.isnan(a):
                per[g] = a
        return ov, per

    def cell(self, cell, w, members) -> tuple[float, float]:
        """
        full-target 估计量：对 members 里的 (f,s) 求各量 → 平均 → worst=先平均后 min。

        members: (cell,f,s) key 列表（子集，供 LOFO / fold-level）。
        Returns (Overall, marginal worst)。
        """
        ovs, per_acc = [], {g: [] for g in self.groups}
        for key in members:
            ov, per = self._fs_metric(key, w)
            if not np.isnan(ov):
                ovs.append(ov)
            for g, a in per.items():
                per_acc[g].append(a)
        means = {g: float(np.mean(v)) for g, v in per_acc.items() if v}
        return float(np.mean(ovs)), float(min(means.values()))


def _members(cell, folds=range(N_FOLDS), seeds=SEEDS):
    if cell == "swad":
        return [("swad", k, None) for k in folds]
    return [(cell, k, s) for k in folds for s in seeds]


def _ci_p(boot_d: np.ndarray, obs: float, n_boot: int) -> tuple[list[float], float, bool]:
    """双侧：返回 (95%CI, 双侧 p, 是否显著)。双侧 p 由中心化 bootstrap 两尾。"""
    lo, hi = np.percentile(boot_d, [2.5, 97.5])
    t = boot_d - obs
    p_two = 2 * min((1 + np.sum(t >= obs)) / (n_boot + 1), (1 + np.sum(t <= obs)) / (n_boot + 1))
    return [float(lo), float(hi)], float(min(1.0, p_two)), bool(lo > 0 or hi < 0)


def _p_noninf(boot_d: np.ndarray, obs: float, n_boot: int) -> float:
    """Overall 非劣单侧：H0 ΔOverall ≤ δ_NI。"""
    t = boot_d - obs
    return float((1 + np.sum(t >= obs - DELTA_NI)) / (n_boot + 1))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items); out = {}; prev = 0.0
    for i, (k, p) in enumerate(items):
        prev = min(1.0, max(prev, (m - i) * p)); out[k] = prev
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="E2 M→C OOD 主分析")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    y, pid, masks, scores = load_e2()
    swad = load_swad(y, pid)
    print("E2 主实验 B（M→C OOD）：full-target + 双侧 H1/H2 + 训练随机性稳健性")
    print(f"  n={len(y):,}  患者={len(np.unique(pid)):,}  正例率={y.mean():.4f}  候选组={list(masks)}")
    views = Views(y, masks, scores, swad)
    ones = np.ones(len(y))
    all_cells = CELLS + ("swad",)

    # ---------- 点估计 ----------
    obs = {c: views.cell(c, ones, _members(c)) for c in all_cells}
    print(f"\n{'=' * 74}\nE2 full-target 点估计（15 个 (f,s) 等权平均；SWAD 5 fold）\n{'=' * 74}")
    print(f"| {'cell':<8s} | {'Overall':>8s} | {'marg worst':>10s} |")
    for c in all_cells:
        print(f"| {NICE.get(c, c.upper()):<8s} | {obs[c][0]:>8.4f} | {obs[c][1]:>10.4f} |")

    # ---------- 患者级配对 cluster bootstrap ----------
    print(f"\n患者级配对 cluster bootstrap（B={args.n_boot}）...")
    uniq, pidx = np.unique(pid, return_inverse=True)
    n_cl = len(uniq)
    rng = np.random.default_rng(args.seed)
    boot = {c: np.empty((args.n_boot, 2)) for c in all_cells}
    for b in range(args.n_boot):
        w = np.bincount(rng.integers(0, n_cl, n_cl), minlength=n_cl).astype(float)[pidx]
        for c in all_cells:
            boot[c][b] = views.cell(c, w, _members(c))
        if (b + 1) % 250 == 0:
            print(f"  ... {b+1}/{args.n_boot}")

    # ---------- 双侧 H1/H2 + Overall 非劣 ----------
    fam = {"H1: C-Deep − C-Head": ("condnet_deep", "condnet_head"),
           "H2: C-Full − C-Deep": ("condnet_full", "condnet_deep")}
    print(f"\n{'=' * 74}\nH1/H2 双侧（Holm，m=2）+ Overall 单侧非劣\n{'=' * 74}")
    res = {}
    for name, (a, bl) in fam.items():
        dw = obs[a][1] - obs[bl][1]
        do = obs[a][0] - obs[bl][0]
        bw = boot[a][:, 1] - boot[bl][:, 1]
        bo = boot[a][:, 0] - boot[bl][:, 0]
        ci, p_two, sig = _ci_p(bw, dw, args.n_boot)
        pni = _p_noninf(bo, do, args.n_boot)
        res[name] = {"d_worst": dw, "ci": ci, "p_two": p_two, "sig_worst": sig,
                     "d_overall": do, "p_noninf": pni, "overall_noninf": pni < 0.05}
        print(f"  {name}: Δworst={dw:+.4f} CI=[{ci[0]:+.4f},{ci[1]:+.4f}] p2={p_two:.3f} "
              f"| ΔOv={do:+.4f} p_noninf={pni:.3f}")
    holm_p = holm({k: v["p_two"] for k, v in res.items()})
    print("\n  三分判决（Holm 校正后双侧 CI + Overall 非劣）：")
    for k, v in res.items():
        hp = holm_p[k]
        lo, hi = v["ci"]
        if lo > 0 and v["overall_noninf"]:
            verdict = "**改善**"
        elif hi < 0:
            verdict = "**有害**"
        else:
            verdict = "不确定（不得写「无效应」）"
        v["holm"] = hp
        print(f"  {k}: Holm p2={hp:.3f} → {verdict}")

    # ---------- 历史正向复现（C-Full vs ERM 主 / vs SWAD 次）----------
    print(f"\n{'=' * 74}\n历史正向复现（C-Full vs ERM 主证据 / vs SWAD 次；双侧+Holm）\n{'=' * 74}")
    rep = {}
    for base in ("condnet_erm", "swad"):
        dw = obs["condnet_full"][1] - obs[base][1]
        bw = boot["condnet_full"][:, 1] - boot[base][:, 1]
        ci, p_two, sig = _ci_p(bw, dw, args.n_boot)
        rep[f"C-Full vs {NICE.get(base, base.upper())}"] = {"d": dw, "ci": ci, "p_two": p_two}
    hrep = holm({k: v["p_two"] for k, v in rep.items()})
    for k, v in rep.items():
        print(f"  {k}: Δworst={v['d']:+.4f} CI=[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] "
              f"Holm p2={hrep[k]:.3f} {'**显著**' if hrep[k] < 0.05 else 'n.s.'}")
        v["holm"] = hrep[k]
    print("  （描述性）C-Deep vs ERM/SWAD：",
          f"vs ERM Δworst={obs['condnet_deep'][1]-obs['condnet_erm'][1]:+.4f}",
          f"vs SWAD Δworst={obs['condnet_deep'][1]-obs['swad'][1]:+.4f}")

    # ---------- 训练随机性稳健性 ----------
    print(f"\n{'=' * 74}\n训练随机性稳健性（主 CI 固定 15 模型 ⇒ 条件性；下列并报）\n{'=' * 74}")
    robust = {}
    for name, (a, bl) in fam.items():
        # fold-level 效应：每 fold 内先平均 3 seed，得 5 个 fold 效应（worst 先平均后 min）
        fold_eff = []
        for k in range(N_FOLDS):
            wa = views.cell(a, ones, _members(a, folds=[k]))[1]
            wb = views.cell(bl, ones, _members(bl, folds=[k]))[1]
            fold_eff.append(wa - wb)
        fold_eff = np.array(fold_eff)
        # LOFO：去掉每一折后重算全效应
        lofo = []
        for k in range(N_FOLDS):
            keep = [f for f in range(N_FOLDS) if f != k]
            wa = views.cell(a, ones, _members(a, folds=keep))[1]
            wb = views.cell(bl, ones, _members(bl, folds=keep))[1]
            lofo.append(wa - wb)
        lofo = np.array(lofo)
        n_pos = int((fold_eff > 0).sum())
        robust[name] = {"fold_effects": fold_eff.tolist(), "fold_sign_pos": n_pos,
                        "fold_range": [float(fold_eff.min()), float(fold_eff.max())],
                        "lofo": lofo.tolist(), "lofo_sign_flips": int((np.sign(lofo) != np.sign(res[name]["d_worst"])).sum())}
        print(f"  {name}:")
        print(f"    5 fold 效应={np.array2string(fold_eff, precision=4, floatmode='fixed')} "
              f"(正号 {n_pos}/5, 范围[{fold_eff.min():+.4f},{fold_eff.max():+.4f}])")
        print(f"    LOFO={np.array2string(lofo, precision=4, floatmode='fixed')} "
              f"(去某折变号次数 {robust[name]['lofo_sign_flips']}/5)")

    out = {"n": int(len(y)), "n_patients": int(n_cl), "n_boot": args.n_boot,
           "point": {NICE.get(c, c.upper()): {"overall": obs[c][0], "marginal_worst": obs[c][1]} for c in all_cells},
           "hypotheses": res, "replication": rep, "robustness": robust}
    p = OUTPUTS / "conditioning_ablation" / "e2_ood_results.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已落盘 → {p}")


if __name__ == "__main__":
    main()
