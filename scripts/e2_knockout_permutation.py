"""
E1 敏感性实验②：通路 knockout × 属性置换（fc vs conv 的年龄依赖归因）
=====================================================================
docs/conditioning_ablation_plan.md E1 的补充。E1 的 ρ_l 从**权重层面**指出条件化几乎全在 fc；
本实验从**行为层面**（预测对 age 输入的依赖）直接验证：对 C-Deep / C-Full 做两种 knockout，
再各自比较「真实 age vs 置换 age」的性能差 Δ。

两种 knockout（均置零生成器，精确、可逆，不改结构）：
  · **fc_off（ΔW_fc = 0）**：置零 `fc_A_gen`（weight+bias）⇒ a_fc=0 ⇒ ΔW_fc=0 ⇒ fc 退化为
    **属性无关的共享 base 头**；conv 条件化保留。
  · **conv_off（M_conv = 0）**：置零 `A_gen`（shared-A，weight+bias）⇒ a_shared=0 ⇒ M=0 ⇒ 所有
    条件化 conv 退化为**普通卷积**；fc 条件化保留。
  （`adapted_conv2d(M=0)`＝普通卷积、`adapted_linear(ΔW=0)`＝普通线性，见 resnet18_hyperadapt。）

对照基线 `full`（两条通路都开）= 标准 E1 置换。判读：
  · **fc_off 后 Δ(真实−置换) 塌到 ~0** ⇒ 年龄依赖主要由 **fc** 承担；
  · **fc_off 后 Δ 仍明显 > 0** ⇒ **conv 虽 ρ 小、却有实际功能**（行为上仍依赖 age）。
  · conv_off 作互补对照：若 Δ 几乎不掉（≈full）⇒ conv 对行为无贡献，与 ρ 一致。

估计量与 E1 一致：averaging（逐折算指标→折间平均）、age-4 候选组、病灶级整组置换 age（n_perm=20，
与 e1_rho_and_permutation 同一 rng）、分组键恒用真实 age。checkpoint 默认 **best_overall**（E1 主结果、
§4 ρ 表口径；实验①已证换 best_worstcase 不改变机制图景）。

内建自检：`full` 变体的 Δ 应复现 e1_permutation.json（同 checkpoint）——验证 knockout 管线与
canonical 置换管线一致。

用法（本地 GPU 即可，无需排队；GTX 1050 Ti/batch128 足够）：
    python scripts/e2_knockout_permutation.py --checkpoint overall
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.e1_ham_analysis import CV_DIR, HAM_AGE_NAMES, N_FOLDS, NICE, SEEDS, SPLIT_DIR, load_candidate_groups
from scripts.e1_rho_and_permutation import LOCATION_OF, N_PERM, _build_cond_model, _worst_and_overall

# 只对两个含 conv 条件化的 cell 做 knockout（C-Head 无 conv、ERM 无条件化）
KNOCKOUT_CELLS = ("condnet_deep", "condnet_full")
VARIANTS = ("full", "fc_off", "conv_off")   # full=两通路开；fc_off=ΔW_fc=0；conv_off=M_conv=0


def _zero_generator(model: torch.nn.Module, which: str) -> None:
    """
    原地置零指定条件生成器（knockout）。

    Args:
        which: "fc"  → 置零 fc_A_gen（ΔW_fc=0，关 fc 条件化）；
               "conv"→ 置零 A_gen.*（shared-A，M_conv=0，关全部 conv 条件化）。
    ⚠️ 用 startswith 精确区分：`fc_A_gen.` 与 `A_gen.` 的名字都含子串 "A_gen"。
    """
    prefix = "fc_A_gen." if which == "fc" else "A_gen."
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.startswith(prefix):
                p.zero_()


@torch.no_grad()
def build_variant_tables(
    checkpoint: str, batch_size: int = 128,
) -> list[dict]:
    """
    对 KNOCKOUT_CELLS × 5 折 × 3 seed × 3 变体，算「每图 × 4 age」的 [n,4] logits 表。

    每个 (cell,fold,seed) 只从 NFS 载入一次 checkpoint，随后在内存里对生成器做三种 knockout
    （原状态缓存、变体间复原），避免重复 I/O。

    Returns:
        folds[k] = {"y","age","lesion","tables"{(cell,variant,seed) -> [n,4]}}。
    """
    from torch.utils.data import DataLoader
    from src.training.train_condnet import DATASET_SPECS, load_config

    if not torch.cuda.is_available():
        raise RuntimeError("需 GPU（本地 GTX 1050 Ti 即可，无需排队）。")
    dev = torch.device("cuda")
    spec = DATASET_SPECS["ham10000"]
    cfg = load_config(spec.config_path)
    _, eval_t = spec.build_transforms(cfg)
    selected = json.loads((CV_DIR / "selected_configs.json").read_text())["config"]

    folds: list[dict] = []
    for k in range(N_FOLDS):
        split_dir = SPLIT_DIR / f"fold{k}"
        df = pd.read_csv(split_dir / "test.csv")
        df = df[df["age_group"] >= 0].reset_index(drop=True)
        _, _, test_set = spec.build_datasets(cfg, split_dir, eval_t, eval_t)
        loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        rec = {"y": df["label"].values.astype(int), "age": df["age_group"].values.astype(int),
               "lesion": df["lesion_id"].values, "tables": {}}
        for cell in KNOCKOUT_CELLS:
            tag = selected[cell]
            for seed in SEEDS:
                model = _build_cond_model(spec, cell, k, seed, tag, checkpoint)
                # 缓存原始生成器参数（fc_A_gen / A_gen.*），供变体间复原
                orig = {n: p.detach().clone() for n, p in model.named_parameters()
                        if n.startswith("fc_A_gen.") or n.startswith("A_gen.")}
                model.to(dev)
                for variant in VARIANTS:
                    # 复原 → 施加当前 knockout
                    with torch.no_grad():
                        for n, p in model.named_parameters():
                            if n in orig:
                                p.copy_(orig[n])
                    if variant == "fc_off":
                        _zero_generator(model, "fc")
                    elif variant == "conv_off":
                        _zero_generator(model, "conv")
                    # 前向：4 个 age 取值
                    cols = []
                    for a in range(len(HAM_AGE_NAMES)):
                        outs = []
                        for batch in loader:
                            images = batch[0].to(dev, non_blocking=True)
                            av = torch.full((images.shape[0],), a, dtype=torch.long, device=dev)
                            outs.append(model(images, av).squeeze(1).cpu().numpy())
                        cols.append(np.concatenate(outs))
                    rec["tables"][(cell, variant, seed)] = np.stack(cols, axis=1)   # [n,4]
                del model, orig
                torch.cuda.empty_cache()
        folds.append(rec)
        print(f"  fold{k}: 变体 logits 表完成（{len(KNOCKOUT_CELLS)} cell × {len(VARIANTS)} 变体 × {len(SEEDS)} seed）")
    return folds


def _perm_test(folds: list[dict], groups: list[str], cell: str, variant: str) -> dict:
    """
    对某 (cell,variant) 做真实 vs 置换 age 的评估（复用 _worst_and_overall 的 averaging 口径）。

    置换 = 病灶级整组置换 age（n_perm=20，与 e1_rho_and_permutation 同一 rng seed=0），分组键恒真实。
    """
    # 为该 variant 构造 _worst_and_overall 期望的 folds 视图：tables 键为 (cell, seed)
    view = [{"y": f["y"], "age": f["age"], "lesion": f["lesion"],
             "tables": {(cell, s): f["tables"][(cell, variant, s)] for s in SEEDS}} for f in folds]
    o_real, w_real = _worst_and_overall(view, groups, cell, None)
    # 病灶级整组置换（与 e1 完全一致的生成方式）
    rng = np.random.default_rng(0)
    perms = []
    for _ in range(N_PERM):
        pm = {}
        for kk, f in enumerate(view):
            les, inv = np.unique(f["lesion"], return_inverse=True)
            les_age = np.array([f["age"][f["lesion"] == l][0] for l in les])
            pm[kk] = rng.permutation(les_age)[inv]
        perms.append(pm)
    o_p, w_p = [], []
    for pm in perms:
        o, w = _worst_and_overall(view, groups, cell, pm)
        o_p.append(o); w_p.append(w)
    return {"overall_real": o_real, "worst_real": w_real,
            "overall_perm_mean": float(np.mean(o_p)), "worst_perm_mean": float(np.mean(w_p)),
            "overall_perm_sd": float(np.std(o_p)), "worst_perm_sd": float(np.std(w_p)),
            "delta_overall": float(o_real - np.mean(o_p)), "delta_worst": float(w_real - np.mean(w_p))}


def main() -> None:
    ap = argparse.ArgumentParser(description="E1②：通路 knockout × 属性置换")
    ap.add_argument("--checkpoint", choices=("overall", "worstcase"), default="overall")
    ap.add_argument("--batch-size", type=int, default=128,
                    help="本地 GTX 1050 Ti(4GB) 建议 32；集群 gpus24 可 128。")
    args = ap.parse_args()
    groups = load_candidate_groups()
    print("E1 敏感性实验②：通路 knockout × 属性置换")
    print(f"  checkpoint={args.checkpoint}  cells={[NICE[c] for c in KNOCKOUT_CELLS]}  候选组={groups}")
    print("  变体：full(两通路开) / fc_off(ΔW_fc=0) / conv_off(M_conv=0)\n")

    folds = build_variant_tables(args.checkpoint, batch_size=args.batch_size)

    print(f"\n{'=' * 90}")
    print("置换评估 Δ(真实 age − 置换 age)：Δ→0 ⇒ 该通路关掉后模型不再依赖 age")
    print(f"{'=' * 90}")
    print(f"| {'cell':<7s} | {'variant':<9s} | {'真实 Ov/worst':>16s} | {'置换 Ov/worst(均)':>18s} | "
          f"{'Δoverall':>8s} | {'Δworst':>8s} |")
    out: dict = {"checkpoint": args.checkpoint, "candidate_groups": groups, "results": {}}
    for cell in KNOCKOUT_CELLS:
        for variant in VARIANTS:
            r = _perm_test(folds, groups, cell, variant)
            out["results"][f"{NICE[cell]}|{variant}"] = r
            print(f"| {NICE[cell]:<7s} | {variant:<9s} | {r['overall_real']:.4f}/{r['worst_real']:.4f}  | "
                  f"{r['overall_perm_mean']:.4f}/{r['worst_perm_mean']:.4f}     | "
                  f"{r['delta_overall']:>+8.4f} | {r['delta_worst']:>+8.4f} |")
        print(f"| {'':<7s} | {'':<9s} | {'':>16s} | {'':>18s} | {'':>8s} | {'':>8s} |")

    # 判读摘要：fc_off 相对 full 的 Δworst 保留比例
    print(f"\n{'=' * 90}\n判读（Δworst 保留比例 = variant Δworst / full Δworst）\n{'=' * 90}")
    for cell in KNOCKOUT_CELLS:
        full_dw = out["results"][f"{NICE[cell]}|full"]["delta_worst"]
        for variant in ("fc_off", "conv_off"):
            dw = out["results"][f"{NICE[cell]}|{variant}"]["delta_worst"]
            frac = dw / full_dw if abs(full_dw) > 1e-9 else float("nan")
            tag = ("关 fc" if variant == "fc_off" else "关 conv")
            print(f"  {NICE[cell]:<7s} {tag}: Δworst {dw:+.4f}（full {full_dw:+.4f} 的 {frac:.0%}）"
                  + ("  ⇒ age 依赖主要在被关的通路" if frac < 0.3 else
                     "  ⇒ 被关通路非唯一来源" if frac < 0.7 else "  ⇒ 被关通路贡献很小"))

    p = CV_DIR / f"e2_knockout_permutation_{args.checkpoint}.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已落盘 → {p}")


if __name__ == "__main__":
    main()
