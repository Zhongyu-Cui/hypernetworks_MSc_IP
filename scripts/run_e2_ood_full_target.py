"""
E2 主实验 B：ResNet18CondNet 四 cell 的 M→C **full-target** OOD 评估
====================================================================
docs/conditioning_ablation_plan.md E2 的推理侧落地。对每个 MIMIC-source 的 condnet cell 模型
（5 fold × 3 seed = 15 个），在**完整 CheXpert target**（138,644 张）上评估、落盘逐样本预测
（含 patient_id 供患者级 cluster bootstrap）。

**估计量 = full-target 配对效应**（plan §3.3）：每 source fold×seed 评完整 target、对 (f,s) 等权平均
——**按构造无跨折池化**，天然免疫 pooled-OOF 的跨折校准漂移伪影（见 oof_regime_results.md §0）。

**零泄漏**：source=MIMIC 的模型与 config 均只由 MIMIC 自身选定（`mimic_cxr/cv5/selected_configs.json`，
四 cell 全选 lr1e-04_wd1e-03）；CheXpert 仅作最终评估、不参与任何选择。

**checkpoint 布局**：E2 用 fold/seed **解耦**布局
`conditioning_ablation/mimic_cxr/cv5/fold{k}/{cell}_{tag}_seed{s}_best_overall.pth`
（区别于既有 OOD 驱动的 fold≡seed 布局）。

**复用已验证的推理路径**：target loader / GPU 快速失败 / file_system 共享策略均复用
`run_ood_cxr_full_target`（E-audit.2 作业 72196 已验证正确），只替换模型工厂与 checkpoint 寻址，
不引入自建缓存（避免在最耗成本处引入 correctness-critical 新组件）。

用法（重型 GPU 推理，经 sbatch；--cell 供 array 按 cell 并行）：
    python scripts/run_e2_ood_full_target.py --cell condnet_full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.models.resnet18_condnet import build_base_fc_seed
from src.training.harness.predictions import save_predictions
from src.training.harness.train_loop import (
    evaluate, forward_image_only, forward_sex_race_age, unpack_sex_race_age,
)
from src.training.run_ood_cxr_full_target import build_full_target_loader  # sets file_system + GPU 检查
from src.training.train_condnet import DATASET_SPECS

OUTPUTS = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/conditioning_ablation")
CV_DIR = OUTPUTS / "mimic_cxr" / "cv5"
PRED_DIR = OUTPUTS / "ood_e2_mimic2chexpert" / "predictions"
CELLS = ("condnet_erm", "condnet_head", "condnet_deep", "condnet_full")
LOCATION_OF = {"condnet_erm": "none", "condnet_head": "head",
               "condnet_deep": "deep", "condnet_full": "full"}
N_FOLDS = 5
SEEDS = (42, 43, 44)
CRITERION = torch.nn.BCEWithLogitsLoss()


def run_cell(cell: str) -> None:
    """对某 cell 的 15 个 (fold,seed) 模型，各评完整 CheXpert target，落盘预测（含 patient_id）。"""
    if not torch.cuda.is_available():
        raise RuntimeError("需 GPU（经 sbatch 提交）。")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    sel = json.loads((CV_DIR / "selected_configs.json").read_text())["config"]
    tag = sel[cell]
    spec = DATASET_SPECS["mimic"]
    location = LOCATION_OF[cell]
    forward_fn = forward_image_only if location == "none" else spec.attr_forward_fn

    loader, df = build_full_target_loader(target="chexpert")   # 复用 E-audit 缓存的完整 target CSV
    patient_id = df["patient_id"].values
    print(f"完整 target: n={len(df):,}  唯一患者={df['patient_id'].nunique():,}  "
          f"正例率={df['label'].mean():.4f}  cell={cell} config={tag}")

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for fold in range(N_FOLDS):
        for seed in SEEDS:
            ckpt = CV_DIR / f"fold{fold}" / f"{cell}_{tag}_seed{seed}_best_overall.pth"
            if not ckpt.exists():
                raise FileNotFoundError(f"缺 checkpoint：{ckpt}")
            # base_fc_seed 对评估无影响（checkpoint 全量覆盖权重）；pretrained=False 免下载 ImageNet
            model = spec.cond_factory(location, build_base_fc_seed("mimic", fold, seed), False, False)
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            model.to("cuda").eval()
            er = evaluate(model, loader, CRITERION, "chexpert", unpack_sex_race_age, forward_fn)
            assert len(er.labels) == len(patient_id), "预测数与 target 行数不一致"
            npz = PRED_DIR / f"{cell}_{tag}_fold{fold}_seed{seed}.npz"
            save_predictions(npz, er.labels, er.logits, er.attrs, extra={"patient_id": patient_id})
            wc = f"{er.wc_auc:.4f}" if er.wc_auc is not None else "N/A"
            print(f"  {cell} fold{fold} seed{seed}: Overall={er.overall_auc:.4f} worst={wc} "
                  f"n={len(er.labels):,} → {npz.name}")
            del model
            torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser(description="E2 M→C full-target OOD 评估（condnet 四 cell）")
    p.add_argument("--cell", choices=CELLS, default=None, help="只跑指定 cell（缺省=全部）。")
    args = p.parse_args()
    for cell in ((args.cell,) if args.cell else CELLS):
        run_cell(cell)
    print("\nE2 full-target OOD 推理完成。分析见 scripts/e2_ood_analysis.py")


if __name__ == "__main__":
    main()
