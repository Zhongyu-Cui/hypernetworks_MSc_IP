"""
实验 P·OOD：用**源库的 g** 预测**目标库**的属性（跨库 p̂）
=========================================================
OOD 设定下，g + HyperAdapt 作为**单一方法整体迁移**到目标库：目标库的条件输入必须由
**源库训练的 g** 给出（部署时拿不到目标库的属性标注，也不会在目标库上重训 g）。
本脚本据此为每个 source fold 生成一份「该折的 g 对完整 target 的预测」。

与 ID 版（`build_attr_predictions.py`）的关系：
  · probe 的拟合过程**逐字相同**（同特征、同 StandardScaler、同 C、同 lbfgs），且仍只用
    source fold-k 的 train 拟合、source val 定温度 ⇒ 目标库全程不参与任何拟合，**零泄漏**；
  · 只是预测对象从「source 自己的三个 split」换成「完整 target」。
  · 特征可直接复用两库已缓存的 `features.npz`——特征提取器是**冻结的 ImageNet 权重**，
    与库无关、与标签无关，故同一个 probe 可以直接作用于目标库特征。

输出：`outputs/ood_cxr/<src>2<tgt>/attr_pred/fold{k}.npz`
      keys = target 的 image_path（与 full-target CSV 对齐用），prob_<axis> / gt_<axis>，
      外加 `prior_<axis>`（source train 的 p̂ 均值，供 const 对照臂使用）。

用法（轻量，CPU 即可；特征已缓存）：
    python scripts/build_attr_predictions_cross.py --source mimic --target chexpert
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_attr_predictions import (
    DATASET_SPECS, axis_labels, read_fold_frames,
)
from src.training.attr_predictor import PROBE_C, _softmax, _to_logit_matrix, fit_temperature
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs")
# full-target CSV 由 run_ood_cxr_full_target.build_full_target_csv 生成并缓存于此
ABLATION_DIR = OUTPUTS_ROOT / "conditioning_ablation"
CV = 5


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    ap = argparse.ArgumentParser(description="实验 P·OOD：源库 g → 目标库属性预测")
    ap.add_argument("--source", required=True, choices=("mimic", "chexpert"))
    ap.add_argument("--target", required=True, choices=("mimic", "chexpert"))
    ap.add_argument("--cv", type=int, default=CV)
    return ap.parse_args()


def load_feature_map(spec: dict) -> dict[str, np.ndarray]:
    """读该库已缓存的冻结特征（key -> [512]）。"""
    cache = OUTPUTS_ROOT / spec["attr_root"] / "features.npz"
    if not cache.exists():
        raise FileNotFoundError(f"缺特征缓存 {cache}；先跑 build_attr_predictions.py --dataset <ds>")
    data = np.load(cache, allow_pickle=True)
    return dict(zip(data["keys"].astype(str), data["features"].astype(np.float32)))


def full_target_frame(target: str, cv: int) -> pd.DataFrame:
    """
    读完整 target 的 CSV（cv{K} 五折 test 的并集）。

    与 `run_ood_cxr_full_target.build_full_target_csv` 同一份缓存文件；不存在时按同样规则重建，
    保证行序与 OOD 评估 loader 完全一致（p̂ 靠 image_path 键 join，行序不是对齐依据，但保持一致更安全）。

    Args:
        target: 目标库名。
        cv    : 折数。

    Returns:
        完整 target 的 DataFrame。
    """
    csv_path = ABLATION_DIR / f"{target}_full_target_test.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    split_sub = DATASET_SPECS[target]["split_dir"]
    frames = [pd.read_csv(REPO_ROOT / split_sub / f"cv{cv}" / f"fold{k}" / "test.csv")
              for k in range(cv)]
    df = pd.concat(frames, ignore_index=True)
    assert not df["image_path"].duplicated().any(), "五折 test 并集出现重复 image_path"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def main() -> None:
    args = parse_args()
    if args.source == args.target:
        raise ValueError("source 与 target 必须不同")
    src_spec, tgt_spec = DATASET_SPECS[args.source], DATASET_SPECS[args.target]
    axes_spec: dict[str, dict] = src_spec["axes"]

    src_features = load_feature_map(src_spec)
    tgt_features = load_feature_map(tgt_spec)
    tgt_df = full_target_frame(args.target, args.cv)
    tgt_keys = tgt_df["image_path"].astype(str).to_numpy()
    tgt_x = np.stack([tgt_features[k] for k in tgt_keys])
    print(f"[实验 P·OOD] {args.source} → {args.target}  完整 target n={len(tgt_keys):,}  "
          f"轴={list(axes_spec)}")

    out_root = OUTPUTS_ROOT / "ood_cxr" / f"{args.source}2{args.target}" / "attr_pred"
    out_root.mkdir(parents=True, exist_ok=True)
    metrics: dict = {"source": args.source, "target": args.target, "folds": {}}

    for fold in range(args.cv):
        frames = read_fold_frames(src_spec, args.cv, fold)
        payload: dict[str, np.ndarray] = {}
        fold_metrics: dict[str, dict] = {}

        for axis, axis_spec in axes_spec.items():
            # --- 与 ID 版逐字相同的 probe 拟合（只用 source 的 train / val）---
            x_tr = np.stack([src_features[k] for k in frames["train"][src_spec["key_col"]].astype(str)])
            x_va = np.stack([src_features[k] for k in frames["val"][src_spec["key_col"]].astype(str)])
            y_tr = axis_labels(frames["train"], axis_spec)
            y_va = axis_labels(frames["val"], axis_spec)

            scaler = StandardScaler().fit(x_tr)
            clf = LogisticRegression(C=PROBE_C, max_iter=2000).fit(scaler.transform(x_tr), y_tr)
            num_classes = axis_spec["num_classes"]
            logit_va = _to_logit_matrix(clf.decision_function(scaler.transform(x_va)), num_classes)
            temperature = fit_temperature(logit_va, y_va)

            # --- 预测完整 target（目标库从未参与拟合）---
            logit_tgt = _to_logit_matrix(clf.decision_function(scaler.transform(tgt_x)), num_classes)
            prob_tgt = _softmax(logit_tgt, temperature=temperature).astype(np.float32)
            gt_tgt = axis_labels(tgt_df, tgt_spec["axes"][axis])

            # const 对照臂的先验：仍取 **source train** 的 p̂ 均值（部署时只能用源库统计量）
            logit_tr = _to_logit_matrix(clf.decision_function(scaler.transform(x_tr)), num_classes)
            prior = _softmax(logit_tr, temperature=temperature).mean(axis=0).astype(np.float32)

            payload[f"prob_{axis}"] = prob_tgt
            payload[f"gt_{axis}"] = gt_tgt
            payload[f"prior_{axis}"] = prior
            acc = float((prob_tgt.argmax(axis=1) == gt_tgt).mean())
            entropy = float((-prob_tgt * np.log(np.clip(prob_tgt, 1e-12, None))).sum(axis=1).mean())
            fold_metrics[axis] = {"temperature": temperature, "target_accuracy": acc,
                                  "target_mean_entropy": entropy, "prior": prior.tolist()}
            print(f"  fold{fold} · {axis:5s}: T={temperature:.3f}  "
                  f"target acc={acc:.3f}  熵={entropy:.3f}")

        np.savez(out_root / f"fold{fold}.npz",
                 keys=np.asarray(tgt_keys, dtype=object), **payload)
        metrics["folds"][f"fold{fold}"] = fold_metrics

    (out_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n跨库 p̂ 已写入 {out_root}")


if __name__ == "__main__":
    main()
