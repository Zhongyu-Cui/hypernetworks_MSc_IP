"""
E1 机制诊断：逐层有效偏移 ρ_l + 属性置换评估（plan §4，复用 checkpoint、零额外训练）
====================================================================================
plan §4「机制诊断（仅两项）」的落地：

**① 逐层有效偏移** `ρ_l = ‖W_l ⊙ M_l‖_F / ‖W_l‖_F`（fc 加性用 `‖ΔW‖_F/‖W‖_F`）
   —— 确认模型**实际使用了哪些条件化层、多深**（挂了 adapter ≠ 用了 adapter）。
   **无需图像**：条件属性只有 age（4 个取值），`M_l = A·B` 只由 age 决定 ⇒ 每层仅 4 个不同的 M_l，
   直接从 checkpoint 的生成器算即可（纯 CPU）。
   口径（plan §6.1）：**先逐单位求 Frobenius norm 再对单位平均**（非「先平均偏移再求 norm」）；
   主结果**按群体等权平均**（4 个 age 组等权，避免大群体主导）。

**② 属性置换评估** —— 真实 A vs 置换 A，确认收益（若有）**依赖正确属性**、而非单纯多了可训练参数。
   · **置换单位 = 病灶**（HAM）：同一 lesion 的所有图整组分到同一个被置换的 age，否则会给同一病灶
     喂进自相矛盾的属性、产生数据中不存在的组合。
   · **子群划分永远用真实属性**（plan §4 硬约束）——置换只改**喂给模型的** age，不改分组键。
   · **所有 cell 用相同置换索引**（配对比较）。`n_perm=20`，报分布。
   · **免重复前向的关键**：模型输出只依赖 (image, age)，age 仅 4 值 ⇒ 预计算每图在 4 个 age 下的
     logits 表，任意置换退化为**查表**。ERM（location=none）忽略 age ⇒ 其 4 列恒等，置换必然无效应
     —— 这构成一个**内建正确性自检**。

估计量与 E1 主分析一致：**averaging**（逐折算指标→折间平均，无跨折池化；见 e1_ham_analysis.py 的
文献依据 Parker2007/Airola2010）+ worst = `min_g[mean_f AUC(g,f)]`（先平均后 min）。

用法：
    # ① GPU：预计算 logits 表（4 cell × 5 折 × 3 seed × 4 age 取值）+ ρ_l，经 sbatch
    python scripts/e1_rho_and_permutation.py --mode logits
    # ②' CPU-only：仅算 ρ_l（不建 logits 表，无需 GPU；因 ρ_l 只依赖生成器权重）
    python scripts/e1_rho_and_permutation.py --mode rho
    # ③ CPU：ρ_l 报告 + 置换分析（需先有 logits 表）
    python scripts/e1_rho_and_permutation.py --mode analyze

**`--checkpoint {overall,worstcase}`**：选评估用 checkpoint。overall=val Overall 最优 epoch（**主结果**）；
worstcase=val worst-case AUC 最优 epoch（训练策略敏感性实验）。ρ_l/logits 表/置换的落盘文件名按此加后缀，
overall 保持原名不覆盖主结果。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.e1_ham_analysis import (
    CELLS, CV_DIR, HAM_AGE_NAMES, N_FOLDS, NICE, SEEDS, SPLIT_DIR, load_candidate_groups,
)
from scripts.eaudit_m2c_full_target import sorted_view, weighted_auc

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGIT_DIR = CV_DIR / "age_logit_tables"
N_PERM = 20
LOCATION_OF = {"condnet_erm": "none", "condnet_head": "head",
               "condnet_deep": "deep", "condnet_full": "full"}


# ============================================================
# ① 逐层有效偏移 ρ_l（纯 CPU，无需图像）
# ============================================================
@torch.no_grad()
def compute_rho(model: torch.nn.Module) -> dict[str, dict[str, float]]:
    """
    算该模型每个**条件化层**的两个偏移量（对 4 个 age 组等权）：

    · **`rho`（总偏移量级）** = `E_a[‖ΔW_l(a)‖_F] / ‖W_l‖_F` —— 该层权重被改动了多少。
    · **`rho_between`（属性特异性偏移）** =
      `sqrt(E_a‖ΔW_l(a) − mean_a ΔW_l‖_F²) / ‖W_l‖_F` —— 偏移**随属性变化**的部分有多大。

    ⚠️ **为什么必须两个都报（manipulation check 的核心）**：`rho` 只量偏移**大小**，
    **区分不了**「所有属性产生**相同**偏移」（⇒ 该层退化为**静态 adapter/重参数化**）
    与「真正的属性特异性偏移」；**只有 `rho_between` 才证明该层在按属性分化**。

    **实现委托给 `ResNet18CondNet.conditioning_rho`（单一事实来源，防漂移）**；本函数只负责
    喂入 HAM 的 4 个 age 取值嵌入。conv 乘性 ΔW=W⊙M、fc 加性 ΔW=A_fc·B_fc（详见模型方法）。

    Returns:
        {模块名 -> {"rho": …, "rho_between": …}}；按前向顺序。
    """
    ages = torch.arange(len(HAM_AGE_NAMES), dtype=torch.long)     # 4 个 age 取值
    emb = model.patient_embed(ages)                               # [4, embed_dim]
    return model.conditioning_rho(emb)


def _rho_suffix(checkpoint: str) -> str:
    """checkpoint 落盘后缀：overall 保持原名（不覆盖主结果），worstcase 加 _worstcase。"""
    return "" if checkpoint == "overall" else f"_{checkpoint}"


def _build_cond_model(spec, cell: str, k: int, seed: int, tag: str, checkpoint: str) -> torch.nn.Module:
    """
    在 CPU 上构造某 (cell, fold, seed) 的 condnet 并加载指定 checkpoint 的权重（canonical base state）。

    Args:
        checkpoint: "overall" 或 "worstcase"，决定加载 `_best_{checkpoint}.pth`。

    Returns:
        eval 模式的模型（仍在 CPU）。
    """
    from src.models.resnet18_condnet import build_base_fc_seed

    ckpt = CV_DIR / f"fold{k}" / f"{cell}_{tag}_seed{seed}_best_{checkpoint}.pth"
    model = spec.cond_factory(LOCATION_OF[cell],
                              build_base_fc_seed("ham10000", fold=k, seed=seed), False, False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def build_rho_only(selected: dict[str, str], checkpoint: str = "overall") -> None:
    """
    **纯 CPU** 计算 ρ_l 并落盘（不建 logits 表、不需 GPU）。

    ρ_l 只依赖条件生成器（age 仅 4 取值），与图像无关，故无需前向、无需 GPU。用于快速复核
    「换 checkpoint 后各层实际使用了多少」而不必重跑整套 logits 表。ERM（无条件通路）跳过。
    """
    from src.training.train_condnet import DATASET_SPECS

    spec = DATASET_SPECS["ham10000"]
    rho_all: dict[str, dict] = {}
    for k in range(N_FOLDS):
        for cell in CELLS:
            tag = selected[cell]
            for seed in SEEDS:
                model = _build_cond_model(spec, cell, k, seed, tag, checkpoint)
                if model.needs_embedding:
                    rho_all[f"{cell}|fold{k}|seed{seed}"] = compute_rho(model)
                del model
    out = CV_DIR / f"e1_rho{_rho_suffix(checkpoint)}.json"
    out.write_text(json.dumps(rho_all, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"ρ_l（{checkpoint}，纯 CPU）已落盘 → {out}  （{len(rho_all)} 个条件化 run）")


# ============================================================
# ② logits 表预计算（GPU）
# ============================================================
@torch.no_grad()
def build_logit_tables(selected: dict[str, str], checkpoint: str = "overall", batch_size: int = 128) -> None:
    """
    对每个 (cell, fold, seed)，算该折 test 全部图像在 **4 个 age 取值**下的 logits，存 [n, 4] 表。
    有了它，任意 age 置换只是查表 ⇒ 20 次置换零额外前向。同时落盘 ρ_l。

    batch_size：集群 gpus24 用 128；本地 GTX 1050 Ti(4GB) 建议 32 防 OOM。
    """
    from torch.utils.data import DataLoader
    from src.models.resnet18_condnet import build_base_fc_seed
    from src.training.train_condnet import DATASET_SPECS, load_config

    if not torch.cuda.is_available():
        raise RuntimeError("需 GPU（经 sbatch 提交）。")
    dev = torch.device("cuda")
    spec = DATASET_SPECS["ham10000"]
    cfg = load_config(spec.config_path)
    _, eval_t = spec.build_transforms(cfg)
    LOGIT_DIR.mkdir(parents=True, exist_ok=True)
    rho_all: dict[str, dict] = {}

    for k in range(N_FOLDS):
        split_dir = REPO_ROOT / "data" / "splits" / "ham10000" / "cv5" / f"fold{k}"
        _, _, test_set = spec.build_datasets(cfg, split_dir, eval_t, eval_t)
        loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        for cell in CELLS:
            tag = selected[cell]
            for seed in SEEDS:
                model = _build_cond_model(spec, cell, k, seed, tag, checkpoint)
                # ρ_l 只需生成器，CPU 上算（ERM 无条件通路 ⇒ 跳过）
                if model.needs_embedding:
                    rho_all[f"{cell}|fold{k}|seed{seed}"] = compute_rho(model)
                model.to(dev)
                cols = []
                for a in range(len(HAM_AGE_NAMES)):
                    outs = []
                    for batch in loader:
                        images = batch[0].to(dev, non_blocking=True)
                        av = torch.full((images.shape[0],), a, dtype=torch.long, device=dev)
                        lg = model(images, av) if model.needs_embedding else model(images)
                        outs.append(lg.squeeze(1).cpu().numpy())
                    cols.append(np.concatenate(outs))
                table = np.stack(cols, axis=1)                     # [n, 4]
                np.save(LOGIT_DIR / f"{cell}_fold{k}_seed{seed}{_rho_suffix(checkpoint)}.npy", table)
                print(f"  {cell:<14s} fold{k} seed{seed}: logits 表 {table.shape} 已存")
                del model
                torch.cuda.empty_cache()
    rho_out = CV_DIR / f"e1_rho{_rho_suffix(checkpoint)}.json"
    rho_out.write_text(json.dumps(rho_all, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nρ_l 已落盘 → {rho_out}")


# ============================================================
# ③ 分析（CPU）
# ============================================================
def _worst_and_overall(
    folds: list[dict], groups: list[str], cell: str, age_feed: dict[int, np.ndarray] | None,
) -> tuple[float, float]:
    """
    averaging 口径下算 (Overall, marginal worst)。

    Args:
        age_feed: {fold -> [n_k] 喂给模型的 age}；None = 用真实 age。**分组键恒用真实 age**。

    Returns:
        (Overall, worst)。
    """
    name2code = {f"age:{v}": kk for kk, v in HAM_AGE_NAMES.items()}
    per_group: dict[str, list[float]] = {g: [] for g in groups}
    overalls: list[float] = []
    for k, f in enumerate(folds):
        y = f["y"]
        real_age = f["age"]                                       # 分组键：永远真实
        fed = real_age if age_feed is None else age_feed[k]
        for seed in SEEDS:
            tbl = f["tables"][(cell, seed)]                       # [n, 4]
            s = tbl[np.arange(len(y)), fed]                       # 按喂入的 age 查表
            i, ys = sorted_view(y, s, np.ones(len(y), bool))
            a = weighted_auc(ys, np.ones(len(y))[i])
            if not np.isnan(a):
                overalls.append(a)
            for g in groups:
                m = real_age == name2code[g]
                ig, yg = sorted_view(y, s, m)
                ag = weighted_auc(yg, np.ones(len(y))[ig])
                if not np.isnan(ag):
                    per_group[g].append(ag)
    means = {g: float(np.mean(v)) for g, v in per_group.items() if v}
    return float(np.mean(overalls)), float(min(means.values()))


def analyze(selected: dict[str, str], checkpoint: str = "overall") -> None:
    """ρ_l 报告 + 属性置换评估。"""
    groups = load_candidate_groups()
    suf = _rho_suffix(checkpoint)
    # 载入逐折数据 + logits 表
    folds: list[dict] = []
    for k in range(N_FOLDS):
        df = pd.read_csv(SPLIT_DIR / f"fold{k}" / "test.csv")
        df = df[df["age_group"] >= 0].reset_index(drop=True)
        rec = {"y": df["label"].values.astype(int), "age": df["age_group"].values.astype(int),
               "lesion": df["lesion_id"].values, "tables": {}}
        for cell in CELLS:
            for seed in SEEDS:
                rec["tables"][(cell, seed)] = np.load(LOGIT_DIR / f"{cell}_fold{k}_seed{seed}{suf}.npy")
        folds.append(rec)

    # ---------- ① ρ_l ----------
    rho_all = json.loads((CV_DIR / f"e1_rho{suf}.json").read_text())
    print(f"\n{'=' * 78}\n① 逐层有效偏移 ρ_l = ‖W⊙M‖_F/‖W‖_F（4 age 组等权平均；跨 5 折 × 3 seed 平均）\n{'=' * 78}")
    print("  含义：挂了 adapter ≠ 用了 adapter。ρ_l≈0 ⇒ 该层条件化实际未被使用。")
    for cell in CELLS:
        keys = [k for k in rho_all if k.startswith(f"{cell}|")]
        if not keys:
            print(f"\n  {NICE[cell]}: 无条件化通路（ERM）")
            continue
        layers = list(rho_all[keys[0]].keys())
        print(f"\n  {NICE[cell]}（{len(layers)} 个条件化模块）:")
        for lname in layers:
            # 兼容两种结构：新版 {rho, rho_between} dict；旧版仅 float rho
            entries = [rho_all[k][lname] for k in keys]
            rhos = [e["rho"] if isinstance(e, dict) else float(e) for e in entries]
            if isinstance(entries[0], dict):
                betws = [e["rho_between"] for e in entries]
                print(f"    {lname:<24s} ρ={np.mean(rhos):.4f} ± {np.std(rhos):.4f}   "
                      f"ρ^between={np.mean(betws):.4f} ± {np.std(betws):.4f}")
            else:
                print(f"    {lname:<24s} ρ={np.mean(rhos):.4f} ± {np.std(rhos):.4f}")

    # ---------- ② 属性置换 ----------
    print(f"\n{'=' * 78}\n② 属性置换评估（病灶级整组置换 age，n_perm={N_PERM}；**分组键恒用真实 age**）\n{'=' * 78}")
    rng = np.random.default_rng(0)
    # 病灶级置换：打乱 lesion→age 的映射（同 lesion 所有图整组同值）；全 cell 共用同一批置换（配对）
    perms: list[dict[int, np.ndarray]] = []
    for _ in range(N_PERM):
        pm: dict[int, np.ndarray] = {}
        for k, f in enumerate(folds):
            les, inv = np.unique(f["lesion"], return_inverse=True)
            les_age = np.array([f["age"][f["lesion"] == l][0] for l in les])
            pm[k] = rng.permutation(les_age)[inv]                 # 整组置换
        perms.append(pm)

    print(f"| {'cell':<8s} | {'真实 age':>18s} | {'置换 age (均值±SD)':>24s} | {'Δ(真实−置换)':>14s} |")
    print(f"| {'':<8s} | {'Overall':>8s}{'worst':>10s} | {'Overall':>11s}{'worst':>13s} | {'Overall':>6s}{'worst':>8s} |")
    out: dict = {}
    for cell in CELLS:
        o_real, w_real = _worst_and_overall(folds, groups, cell, None)
        o_p, w_p = [], []
        for pm in perms:
            o, w = _worst_and_overall(folds, groups, cell, pm)
            o_p.append(o); w_p.append(w)
        d_o, d_w = o_real - np.mean(o_p), w_real - np.mean(w_p)
        out[NICE[cell]] = {"overall_real": o_real, "worst_real": w_real,
                           "overall_perm_mean": float(np.mean(o_p)), "worst_perm_mean": float(np.mean(w_p)),
                           "overall_perm_sd": float(np.std(o_p)), "worst_perm_sd": float(np.std(w_p)),
                           "delta_overall": float(d_o), "delta_worst": float(d_w)}
        print(f"| {NICE[cell]:<8s} | {o_real:>8.4f}{w_real:>10.4f} | "
              f"{np.mean(o_p):.4f}±{np.std(o_p):.4f} {np.mean(w_p):.4f}±{np.std(w_p):.4f} | "
              f"{d_o:>+6.4f}{d_w:>+8.4f} |")
    print("\n  判读：Δ≈0 ⇒ 打乱属性对模型无影响 ⇒ 该 cell **并未真正利用属性信息**（增益/参数与属性无关）；")
    print("        Δ>0 ⇒ 依赖正确属性。**ERM 的 Δ 必须恒为 0**（忽略 age）——这是内建正确性自检。")
    erm_d = abs(out["ERM"]["delta_overall"]) + abs(out["ERM"]["delta_worst"])
    assert erm_d < 1e-9, f"ERM 对 age 置换有反应（Δ={erm_d:.2e}），说明查表/置换实现有误"
    print(f"  [OK] ERM 置换 Δ = {erm_d:.1e}（恒 0，自检通过）")

    p = CV_DIR / f"e1_permutation{suf}.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已落盘 → {p}")


def main() -> None:
    ap = argparse.ArgumentParser(description="E1 机制诊断：ρ_l + 属性置换")
    ap.add_argument("--mode", choices=("logits", "rho", "analyze"), required=True,
                    help="logits=GPU 建 logits 表+ρ_l；rho=纯 CPU 仅算 ρ_l；analyze=ρ_l 报告+置换")
    ap.add_argument("--checkpoint", choices=("overall", "worstcase"), default="overall",
                    help="评估 checkpoint：overall（主结果）或 worstcase（训练策略敏感性）。")
    ap.add_argument("--batch-size", type=int, default=128,
                    help="logits 模式的 batch：集群 gpus24 用 128；本地 1050 Ti(4GB) 用 32。")
    args = ap.parse_args()
    selected = json.loads((CV_DIR / "selected_configs.json").read_text())["config"]
    if args.mode == "logits":
        build_logit_tables(selected, checkpoint=args.checkpoint, batch_size=args.batch_size)
    elif args.mode == "rho":
        build_rho_only(selected, checkpoint=args.checkpoint)
    else:
        analyze(selected, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
