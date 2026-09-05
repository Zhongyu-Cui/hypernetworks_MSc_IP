"""
实验 P·OOD：pred-attr HyperAdapt 的 full-target 跨库评估（MIMIC↔CheXpert）
=========================================================================
按 **OOD v2 预注册**（`docs/ood_experiment_v2_preregistration.md`）的规程，把实验 P 的
pred-attr 臂纳入跨库评估：

  · **主估计量 = full-target**：每个 source `(fold, trial)` 模型评**完整** target，
    逐 (f,t) 算指标再等权平均，**从不跨折池化**（与 `run_ood_cxr_full_target.py` 同口径）；
  · **主指标 = marginal worst-group AUC**；配对 cluster bootstrap 用患者级聚类键；
  · **config 只由 source 自身选定**（source 的 `cv5/selected_configs.json` 的 hyperadapt 项），
    target 全程不参与训练/选择。

**与 ID 版的唯一差异：条件输入来自「源库训练的 g」对目标库的预测**（`scripts/build_attr_predictions_cross.py`）。
这是 g+HyperAdapt 作为**单一方法整体迁移**的正确形态——部署时既没有目标库的属性标注，
也不会在目标库上重训 g。`const` 对照臂的先验向量同样取自 **source train** 的 p̂ 均值。

⚠️ **对 OOD v2 预注册的一处编码偏离**：v2 用 `seed = 42 + fold + 10×trial`；实验 P 的 replicate
在 ID 阶段已按 `seed = 42 + fold + 100×trial` 产出（见 `scripts/p_sigma_train.py`）。两者功能等价
（都是「同 config 只换训练种子」），本脚本用 `--trial-stride` 参数化，默认 100 以复用既有 checkpoint。

用法（重型 GPU 推理，经 sbatch 提交）：
    python -m src.training.run_ood_cxr_pred --direction m2c --attr_mode soft --trials 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch.multiprocessing
from torch.utils.data import DataLoader

# 全 target 推理 + 多 worker 会耗尽 fd（见 run_ood_cxr_full_target 头注）
torch.multiprocessing.set_sharing_strategy("file_system")

from src.datasets.mimic_cxr_dataset import MIMICCXRDataset
from src.datasets.soft_attr_wrapper import (
    ATTR_MODES, SoftAttrDataset, align_probs_by_key, materialize_probs,
)
from src.training.eval_ood_cxr import build_eval_transform, evaluate_ood, load_config
from src.training.harness.predictions import save_predictions
from src.training.harness.train_loop import unpack_sex_race_age_soft
from src.training.run_ood_cxr_full_target import build_full_target_csv, load_selected_config

OUTPUTS_ROOT = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs")
OUTPUT_DIR = {"mimic": OUTPUTS_ROOT / "mimic_cxr", "chexpert": OUTPUTS_ROOT / "chexpert_cxr"}
AXES = ("sex", "race", "age")          # 顺序 = SoftPatientEmbedding 的输入顺序
KEY_COL = "image_path"
CV = 5
DIRECTIONS = {"m2c": ("mimic", "chexpert"), "c2m": ("chexpert", "mimic")}
METHOD_BY_MODE = {
    "soft": "hyperadapt_pred", "hard": "hyperadapt_predhard",
    "perm": "hyperadapt_predperm", "const": "hyperadapt_predconst",
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser(description="实验 P·OOD full-target 评估（pred-attr 臂）")
    p.add_argument("--direction", choices=sorted(DIRECTIONS), default="m2c")
    p.add_argument("--attr_mode", choices=ATTR_MODES, default="soft")
    p.add_argument("--trials", default="0", help="逗号分隔的 trial 编号。")
    p.add_argument("--trial-stride", type=int, default=100,
                   help="seed = 42 + fold + stride×trial；实验 P 的 replicate 用 100（见头注）。")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8,
                   help="全 target 推理是 I/O 密集，worker 太少会让 GPU 饿死。")
    p.add_argument("--skip-existing", action="store_true", default=True)
    return p.parse_args()


def load_cross_predictions(source: str, target: str, fold: int) -> tuple[np.ndarray, dict, dict]:
    """
    读某 source fold 的 g 对完整 target 的属性预测。

    Args:
        source / target: 方向。
        fold           : source 的折号（决定用哪个 g）。

    Returns:
        (keys, {轴 -> [N,C] 概率}, {轴 -> [C] source-train 先验})。

    Raises:
        FileNotFoundError: 跨库 p̂ 未生成。
    """
    path = OUTPUTS_ROOT / "ood_cxr" / f"{source}2{target}" / "attr_pred" / f"fold{fold}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"缺跨库 p̂ {path}；先跑 scripts/build_attr_predictions_cross.py "
            f"--source {source} --target {target}")
    data = np.load(path, allow_pickle=True)
    keys = data["keys"].astype(str)
    probs = {a: data[f"prob_{a}"].astype(np.float32) for a in AXES}
    priors = {a: data[f"prior_{a}"].astype(np.float32) for a in AXES}
    return keys, probs, priors


def build_soft_full_target_loader(
    source: str, target: str, fold: int, attr_mode: str, seed: int,
    batch_size: int, num_workers: int,
) -> tuple[DataLoader, pd.DataFrame]:
    """
    构建**完整 target** 的 soft 条件输入 loader（shuffle=False，行序 == CSV 行序）。

    Args:
        source / target: 方向。
        fold           : source 折号（决定用哪个 g 的预测）。
        attr_mode      : 条件输入模式（soft/hard/perm/const）。
        seed           : 该 run 的 seed（perm 的置换种子据此派生，保证可复现）。
        batch_size / num_workers: DataLoader 参数。

    Returns:
        (loader, df)：df 供按位置取 patient_id 作 cluster 键。
    """
    cfg = load_config(target)
    csv_path = build_full_target_csv(target)
    df = pd.read_csv(csv_path)
    base = MIMICCXRDataset(
        csv_path, transform=build_eval_transform(cfg),
        image_size=cfg["data"]["image_size"],
        age_threshold=cfg["attributes"]["age"]["age_threshold"])
    assert len(base) == len(df), "Dataset 长度与 CSV 行数不一致（行序对齐假设被破坏）"

    keys = df[KEY_COL].astype(str).to_numpy()
    pred_keys, probs_by_axis, priors = load_cross_predictions(source, target, fold)
    materialized: list[np.ndarray] = []
    for axis_index, axis in enumerate(AXES):
        probs = align_probs_by_key(keys, pred_keys, probs_by_axis[axis])
        # 各轴用不同置换种子，避免三轴同步置换（那会保留轴间相关、削弱对照的破坏力）
        materialized.append(materialize_probs(
            probs, attr_mode, prior=priors[axis], seed=seed * 100 + 10 * (axis_index + 1)))
    loader = DataLoader(SoftAttrDataset(base, materialized), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, pin_memory=True)
    return loader, df


def main() -> None:
    args = parse_args()
    source, target = DIRECTIONS[args.direction]
    trials = tuple(int(x) for x in args.trials.split(","))
    method = METHOD_BY_MODE[args.attr_mode]
    tag = load_selected_config(source)["hyperadapt"]      # pred 臂复用 GT-HyperAdapt 的选定 config
    out_pred = OUTPUTS_ROOT / "ood_cxr" / f"{source}2{target}" / "cv5_full_target" / "predictions"

    print(f"\n{'=' * 74}\n实验 P·OOD full-target: {source} → {target}"
          f"\nmethod={method}  attr_mode={args.attr_mode}  config={tag}  trials={trials}"
          f"  (seed = 42 + fold + {args.trial_stride}×trial)\n输出: {out_pred}\n{'=' * 74}")

    for trial in trials:
        for fold in range(CV):
            seed = 42 + fold + args.trial_stride * trial
            npz_path = out_pred / f"{method}_{tag}_seed{seed}_overall.npz"
            if args.skip_existing and npz_path.exists():
                print(f"  {method:<22} fold{fold} trial{trial} seed{seed}: 已存在，跳过")
                continue
            ckpt = OUTPUT_DIR[source] / "cv5" / f"{method}_{tag}_seed{seed}_best_overall.pth"
            if not ckpt.exists():
                raise FileNotFoundError(f"缺少 checkpoint：{ckpt}")
            loader, df = build_soft_full_target_loader(
                source, target, fold, args.attr_mode, seed, args.batch_size, args.num_workers)
            result = evaluate_ood(method, ckpt, source=source, target=target,
                                  test_loader=loader, report=False,
                                  unpack_override=unpack_sex_race_age_soft)
            er = result.eval_result
            patient_id = df["patient_id"].values
            assert len(er.labels) == len(patient_id), "预测数与 CSV 行数不一致"
            # 分组属性进 attrs；soft_* 条件输入与 patient_id 同走 extra（不污染 fairness kwargs）
            extra = {"patient_id": patient_id,
                     **{k: v for k, v in er.attrs.items() if k.startswith("soft_")}}
            attrs = {k: v for k, v in er.attrs.items() if not k.startswith("soft_")}
            save_predictions(npz_path, er.labels, er.logits, attrs, extra=extra)
            wc = f"{er.wc_auc:.4f}" if er.wc_auc is not None else "N/A"
            print(f"  {method:<22} fold{fold} trial{trial} seed{seed}: "
                  f"Overall={er.overall_auc:.4f} worst={wc}  n={len(er.labels):,} → {npz_path.name}")


if __name__ == "__main__":
    main()
