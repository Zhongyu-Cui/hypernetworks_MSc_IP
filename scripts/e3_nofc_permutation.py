"""
E1 敏感性实验③补充：no-fc 训练模型的真实 vs 置换 age 比较（与消融口径一致）
=============================================================================
对实验③重训的 `condnet_deep_nofc` / `condnet_full_nofc`（关 fc 条件化、conv-only、从头训练）做
属性置换评估，**口径与 E1/E2 完全一致**：averaging（逐折算指标→折间平均）、age-4 候选组、
病灶级整组置换 age（n_perm=20、rng seed=0、同 e1_rho_and_permutation）、分组键恒用真实 age、
best_overall checkpoint。

判读：Δ(真实−置换)>0 ⇒ 模型行为上依赖 age。因实验③已证 conv 在操作点 ρ 很小、行为退回近属性无关，
预期 Δ 很小（接近 E2 的 fc_off 变体：conv-only 的 age 依赖弱）。与 E1 fc-on 版（e1_permutation.json）
及 E2 knockout 的 fc_off 变体并排对比。

反事实 logits 表用**本地 GTX 1050 Ti**（batch 32）前向，避开集群排队。

用法：
    python scripts/e3_nofc_permutation.py --batch-size 32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.e1_ham_analysis import CV_DIR, HAM_AGE_NAMES, N_FOLDS, SEEDS, SPLIT_DIR, load_candidate_groups
from scripts.e1_rho_and_permutation import N_PERM, _worst_and_overall

TAG = "lr1e-04_wd1e-04"                          # 实验③沿用 E1 选定 config
NOFC = {"condnet_deep_nofc": ("deep_nofc", "C-Deep-noFC"),
        "condnet_full_nofc": ("full_nofc", "C-Full-noFC")}


@torch.no_grad()
def build_tables(batch_size: int) -> list[dict]:
    """
    对 nofc 两 cell × 5 折 × 3 seed，算「每图 × 4 age 取值」的 [n,4] logits 表（best_overall checkpoint）。
    本地 GPU 前向；同时携带 y/age/lesion 供 averaging 置换。
    """
    from torch.utils.data import DataLoader
    from src.models.resnet18_condnet import build_base_fc_seed
    from src.training.train_condnet import DATASET_SPECS, load_config

    if not torch.cuda.is_available():
        raise RuntimeError("需 GPU（本地 GTX 1050 Ti 即可，无需排队）。")
    dev = torch.device("cuda")
    spec = DATASET_SPECS["ham10000"]
    cfg = load_config(spec.config_path)
    _, eval_t = spec.build_transforms(cfg)

    folds: list[dict] = []
    for k in range(N_FOLDS):
        split_dir = SPLIT_DIR / f"fold{k}"
        df = pd.read_csv(split_dir / "test.csv")
        df = df[df["age_group"] >= 0].reset_index(drop=True)
        _, _, test_set = spec.build_datasets(cfg, split_dir, eval_t, eval_t)
        loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        rec = {"y": df["label"].values.astype(int), "age": df["age_group"].values.astype(int),
               "lesion": df["lesion_id"].values, "tables": {}}
        for cell, (location, _nice) in NOFC.items():
            for seed in SEEDS:
                ckpt = CV_DIR / f"fold{k}" / f"{cell}_{TAG}_seed{seed}_best_overall.pth"
                model = spec.cond_factory(location, build_base_fc_seed("ham10000", fold=k, seed=seed), False, False)
                model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
                model.eval().to(dev)
                cols = []
                for a in range(len(HAM_AGE_NAMES)):
                    outs = []
                    for batch in loader:
                        images = batch[0].to(dev, non_blocking=True)
                        av = torch.full((images.shape[0],), a, dtype=torch.long, device=dev)
                        outs.append(model(images, av).squeeze(1).cpu().numpy())
                    cols.append(np.concatenate(outs))
                rec["tables"][(cell, seed)] = np.stack(cols, axis=1)      # [n,4]
                del model
                torch.cuda.empty_cache()
        folds.append(rec)
        print(f"  fold{k}: nofc logits 表完成（2 cell × 3 seed）")
    return folds


def perm_test(folds: list[dict], groups: list[str], cell: str) -> dict:
    """真实 vs 病灶级整组置换 age（n_perm=20、rng seed=0，与 e1_rho_and_permutation 一致）。"""
    o_real, w_real = _worst_and_overall(folds, groups, cell, None)
    rng = np.random.default_rng(0)
    perms = []
    for _ in range(N_PERM):
        pm = {}
        for kk, f in enumerate(folds):
            les, inv = np.unique(f["lesion"], return_inverse=True)
            les_age = np.array([f["age"][f["lesion"] == l][0] for l in les])
            pm[kk] = rng.permutation(les_age)[inv]
        perms.append(pm)
    o_p, w_p = [], []
    for pm in perms:
        o, w = _worst_and_overall(folds, groups, cell, pm)
        o_p.append(o); w_p.append(w)
    return {"overall_real": o_real, "worst_real": w_real,
            "overall_perm_mean": float(np.mean(o_p)), "worst_perm_mean": float(np.mean(w_p)),
            "overall_perm_sd": float(np.std(o_p)), "worst_perm_sd": float(np.std(w_p)),
            "delta_overall": float(o_real - np.mean(o_p)), "delta_worst": float(w_real - np.mean(w_p))}


def main() -> None:
    ap = argparse.ArgumentParser(description="E1③补充：no-fc 模型真实 vs 置换 age")
    ap.add_argument("--batch-size", type=int, default=32, help="本地 1050 Ti(4GB) 用 32")
    args = ap.parse_args()
    groups = load_candidate_groups()
    print("E1 敏感性实验③补充：no-fc 训练模型的属性置换评估（与 E1/E2 口径一致）")
    print(f"  候选组={groups}  n_perm={N_PERM}  病灶级整组置换  averaging  best_overall\n")

    folds = build_tables(args.batch_size)

    print(f"\n{'=' * 92}")
    print("置换评估 Δ(真实 age − 置换 age)：Δ→0 ⇒ 模型行为上不依赖 age")
    print(f"{'=' * 92}")
    print(f"| {'cell':<13s} | {'真实 Ov/worst':>16s} | {'置换 Ov/worst(均±SD)':>26s} | "
          f"{'Δoverall':>8s} | {'Δworst':>8s} | {'worst σ':>7s} |")
    out = {"tag": TAG, "candidate_groups": groups, "n_perm": N_PERM, "results": {}}
    for cell, (_loc, nice) in NOFC.items():
        r = perm_test(folds, groups, cell)
        out["results"][nice] = r
        sd = r["worst_perm_sd"]
        z = r["delta_worst"] / sd if sd > 1e-9 else float("nan")
        print(f"| {nice:<13s} | {r['overall_real']:.4f}/{r['worst_real']:.4f}  | "
              f"{r['overall_perm_mean']:.4f}/{r['worst_perm_mean']:.4f} (±{sd:.4f}) | "
              f"{r['delta_overall']:>+8.4f} | {r['delta_worst']:>+8.4f} | {z:>6.1f}σ |")

    # 并排对比：E1 fc-on 版（e1_permutation.json）+ E2 knockout 的 fc_off 变体（conv-only）
    print(f"\n{'=' * 92}\n并排对比（同口径）\n{'=' * 92}")
    e1 = json.loads((CV_DIR / "e1_permutation.json").read_text())
    e2p = CV_DIR / "e2_knockout_permutation_overall.json"
    e2 = json.loads(e2p.read_text())["results"] if e2p.exists() else {}
    print(f"  {'对象':<28s} | {'Δoverall':>8s} | {'Δworst':>8s}")
    for cell, (_loc, nice) in NOFC.items():
        base = "C-Deep" if "deep" in cell else "C-Full"
        print(f"  {nice + ' (③conv-only,从头训练)':<28s} | {out['results'][nice]['delta_overall']:>+8.4f} | "
              f"{out['results'][nice]['delta_worst']:>+8.4f}")
        print(f"  {base + ' (①fc-on 全通路)':<28s} | {e1[base]['delta_overall']:>+8.4f} | {e1[base]['delta_worst']:>+8.4f}")
        km = e2.get(f"{base}|fc_off")
        if km:
            print(f"  {base + '|fc_off (②推理时关fc)':<28s} | {km['delta_overall']:>+8.4f} | {km['delta_worst']:>+8.4f}")
        print()

    p = CV_DIR / "e3_nofc_permutation.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已落盘 → {p}")


if __name__ == "__main__":
    main()
