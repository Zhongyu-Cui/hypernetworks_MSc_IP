"""
生成子群分类器 g 的属性预测 p̂（实验 P：Predicted-Attribute HyperAdapt）
======================================================================
方案见 `docs/predicted_attribute_hyperadapt_plan.md` §3.2。对每个 CV fold、**每个敏感属性轴**：

  1. 在该 fold 的 **train** 上拟合 probe（冻结 ImageNet 特征 + 多项 logistic，见
     `src/training/attr_predictor.py`）；
  2. 用同一个 probe 对 train/val/test 全部前向出 p̂（train 侧为 in-sample，**不做交叉拟合**——
     g + HyperAdapt 视为单一方法）；
  3. 在 val 上拟合温度 T（逐轴独立）并施于三个 split；
  4. 落盘 p̂（按样本唯一键索引，多轴写同一个 .npz）与诊断指标。

**特征只抽一次**：backbone 冻结 ⇒ 特征与 fold、与属性轴都无关，全数据集抽一次缓存到
`<attr_pred_root>/features.npz`，各 fold 各轴只重拟合 probe（CPU 秒级）。

**数据集差异全部收敛到 `DATASET_SPECS`**：唯一键列、属性轴（含连续 age 的二分箱阈值）、
行过滤（HAM 的 age_group==-1）、图像 loader（CXR 的 16-bit vs 自然 RGB）、评估 transform
（须与该数据集训练侧的 eval transform 逐项一致）。

用法（特征抽取需 GPU）：
    python scripts/build_attr_predictions.py --dataset fitzpatrick
    python scripts/build_attr_predictions.py --dataset mimic --batch_size 128
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from torchvision import transforms

from src.training.attr_predictor import (
    extract_features, fit_attr_probe, load_cxr_16bit, load_rgb, save_fold_predictions,
)
from src.paths import OUTPUTS_DIR

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = OUTPUTS_DIR
SPLIT_NAMES = ("train", "val", "test")


# ============================================================
# 各数据集的评估 transform（必须与该数据集训练侧 eval transform 逐项一致）
# ============================================================
def _norm(cfg: dict) -> transforms.Normalize:
    """从 yaml 取 ImageNet 归一化。"""
    n = cfg["transforms"]["normalize"]
    return transforms.Normalize(mean=n["mean"], std=n["std"])


def tf_fitzpatrick(cfg: dict) -> transforms.Compose:
    """Fitzpatrick：源图已 224，不 resize。"""
    return transforms.Compose([transforms.ToTensor(), _norm(cfg)])


def tf_resize_crop(cfg: dict) -> transforms.Compose:
    """HAM：Resize(256) → CenterCrop(224)（MEDFAIR 口径）。"""
    e = cfg["transforms"]["eval"]
    return transforms.Compose([
        transforms.Resize(tuple(e["resize"])), transforms.CenterCrop(e["center_crop"]),
        transforms.ToTensor(), _norm(cfg),
    ])


def tf_cxr(cfg: dict) -> transforms.Compose:
    """MIMIC / CheXpert：Grayscale(3) → Resize(224)（16-bit 已在 loader 里归一化）。"""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(cfg["transforms"]["eval"]["resize"]),
        transforms.ToTensor(), _norm(cfg),
    ])


# ============================================================
# 数据集规格
# ============================================================
# axes: 轴名 -> {col: CSV 列名, num_classes: 类别数, threshold: 可选（连续列的二分箱阈值）}
# 轴名须与训练侧 attrs 键一致（sex/race/age/skin）；row_filter 用于 HAM 排除 age_group==-1。
# path_resolver: 可选，CSV 的 image_path -> 实际读取路径。CXR7-1M 的 split CSV 存 512x512 路径，
#   而训练侧 MIMICCXRDataset 按 cfg.data.image_size 替换成 224x224；特征抽取必须**同口径**，
#   否则 g 看的是与主模型不同分辨率的图。样本键仍用 CSV 原始 image_path（与训练侧 join 一致）。
DATASET_SPECS: dict[str, dict] = {
    "fitzpatrick": {
        "config": "configs/fitzpatrick_baseline.yaml",
        "split_dir": "data/splits/fitzpatrick17k",
        "attr_root": "fitzpatrick/cv5/attr_pred",
        "key_col": "md5hash",
        "axes": {"skin": {"col": "skin", "num_classes": 6}},
        "transform": tf_fitzpatrick,
        "image_loader": load_rgb,
        "row_filter": None,
    },
    "ham10000": {
        "config": "configs/ham10000_baseline.yaml",
        "split_dir": "data/splits/ham10000",
        "attr_root": "ham10000/cv5/attr_pred",
        "key_col": "image_id",
        "axes": {"age": {"col": "age_group", "num_classes": 4}},
        "transform": tf_resize_crop,
        "image_loader": load_rgb,
        # HAM：age_group==-1（0-20 排除组）不参与，与训练脚本的过滤口径一致
        "row_filter": {"col": "age_group", "min": 0},
    },
    # HAM 双属性对照臂（实验 P · sex+age 条件化）：与上面 ham10000 的唯一差异是多了 sex 轴，
    # 且写到**独立目录** attr_pred_sexage/，绝不改写支撑已冻结 age-only 结果的 attr_pred/。
    # 键列 / 行过滤 / transform / loader 与 ham10000 逐字段一致 ⇒ 同一份特征、同一套 probe 规程，
    # 该目录里的 prob_age 与 attr_pred/ 的 prob_age 逐位相同（已核对）。
    "ham10000_sexage": {
        "config": "configs/ham10000_baseline.yaml",
        "split_dir": "data/splits/ham10000",
        "attr_root": "ham10000/cv5/attr_pred_sexage",
        "key_col": "image_id",
        "axes": {
            "sex": {"col": "sex", "num_classes": 2},
            "age": {"col": "age_group", "num_classes": 4},
        },
        "transform": tf_resize_crop,
        "image_loader": load_rgb,
        # 与 age-only 臂同一过滤口径：age_group==-1 不参与（两臂样本集逐样本可比）
        "row_filter": {"col": "age_group", "min": 0},
    },
    "mimic": {
        "config": "configs/mimic_cxr_baseline.yaml",
        "split_dir": "data/splits/mimic_cxr_nofinding",
        "attr_root": "mimic_cxr/cv5/attr_pred",
        "key_col": "image_path",              # CXR split CSV 无单图 id，image_path 唯一
        "axes": {
            "sex": {"col": "sex", "num_classes": 2},
            "race": {"col": "race", "num_classes": 2},
            # age 在 CSV 里是连续值，训练侧由 MIMICCXRDataset 按 threshold 在线二分箱，此处同口径
            "age": {"col": "age", "num_classes": 2, "threshold": 60},
        },
        "transform": tf_cxr,
        "image_loader": load_cxr_16bit,
        "row_filter": None,
        "path_resolver": lambda p: p.replace("512x512", "224x224"),
    },
    "chexpert": {
        "config": "configs/chexpert_baseline.yaml",
        "split_dir": "data/splits/chexpert_nofinding",
        "attr_root": "chexpert_cxr/cv5/attr_pred",
        "key_col": "image_path",
        "axes": {
            "sex": {"col": "sex", "num_classes": 2},
            "race": {"col": "race", "num_classes": 2},
            "age": {"col": "age", "num_classes": 2, "threshold": 60},
        },
        "transform": tf_cxr,
        "image_loader": load_cxr_16bit,
        "row_filter": None,
        "path_resolver": lambda p: p.replace("512x512", "224x224"),
    },
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="生成子群分类器 g 的属性预测 p̂（实验 P）")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_SPECS))
    parser.add_argument("--cv", type=int, default=5, help="CV 折数（沿用 splits 下的 cv{K}）。")
    parser.add_argument("--batch_size", type=int, default=128, help="特征抽取的前向 batch。")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader 工作进程数。")
    parser.add_argument("--force_features", action="store_true", help="忽略特征缓存，强制重抽。")
    return parser.parse_args()


def load_cfg(spec: dict) -> dict:
    """读取该数据集的训练 yaml（transform / 归一化口径的来源）。"""
    with open(REPO_ROOT / spec["config"], "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_fold_frames(spec: dict, cv: int, fold: int) -> dict[str, pd.DataFrame]:
    """
    读取某 fold 的三个 split CSV 并施加行过滤。

    Args:
        spec: 数据集配置。
        cv  : 折数 K。
        fold: 折号 k。

    Returns:
        {"train"/"val"/"test" -> DataFrame}。
    """
    split_dir = REPO_ROOT / spec["split_dir"] / f"cv{cv}" / f"fold{fold}"
    frames: dict[str, pd.DataFrame] = {}
    for name in SPLIT_NAMES:
        df = pd.read_csv(split_dir / f"{name}.csv")
        rf = spec["row_filter"]
        if rf is not None:
            # 与训练脚本的过滤口径严格一致（否则 p̂ 覆盖不到 / 多覆盖训练侧样本）
            df = df[df[rf["col"]] >= rf["min"]].reset_index(drop=True)
        frames[name] = df
    return frames


def axis_labels(df: pd.DataFrame, axis_spec: dict) -> np.ndarray:
    """
    从 DataFrame 取某轴的真值类别索引（连续列按 threshold 二分箱）。

    Args:
        df       : split 的 DataFrame。
        axis_spec: {col, num_classes, threshold?}。

    Returns:
        [N] int64 类别索引。
    """
    col = df[axis_spec["col"]]
    if "threshold" in axis_spec:
        return (col.astype(float).values >= axis_spec["threshold"]).astype(np.int64)
    return col.astype(np.int64).values


def build_feature_cache(
    spec: dict, cv: int, transform, batch_size: int, num_workers: int, force: bool,
) -> dict[str, np.ndarray]:
    """
    抽取（或复用缓存）全数据集的冻结特征，返回 key -> 512-d 特征。

    Args:
        spec / cv / transform: 数据集配置、折数、评估 transform。
        batch_size / num_workers: DataLoader 参数。
        force: True 时忽略缓存重抽。

    Returns:
        {key -> [512] float32}。
    """
    cache_path = OUTPUTS_ROOT / spec["attr_root"] / "features.npz"
    key_col = spec["key_col"]

    rows: list[pd.DataFrame] = []
    for fold in range(cv):
        rows.extend(read_fold_frames(spec, cv, fold).values())
    catalog = pd.concat(rows, ignore_index=True).drop_duplicates(subset=[key_col])
    keys = catalog[key_col].astype(str).to_numpy()
    paths = catalog["image_path"].astype(str).to_numpy()
    resolver = spec.get("path_resolver")
    if resolver is not None:
        paths = np.array([resolver(p) for p in paths])

    if cache_path.exists() and not force:
        cached = np.load(cache_path, allow_pickle=True)
        cached_keys = cached["keys"].astype(str)
        if set(cached_keys) >= set(keys):
            print(f"  复用特征缓存 {cache_path}（{len(cached_keys):,} 个样本）")
            return dict(zip(cached_keys, cached["features"].astype(np.float32)))
        print("  ⚠️ 特征缓存的键集合不覆盖当前 split，重新抽取")

    print(f"  抽取冻结 ImageNet 特征：{len(keys):,} 张图 ...")
    features = extract_features(paths, transform, batch_size=batch_size,
                                num_workers=num_workers, image_loader=spec["image_loader"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, keys=np.asarray(keys, dtype=object), features=features)
    print(f"  特征已缓存 -> {cache_path}  shape={features.shape}")
    return dict(zip(keys, features))


def main() -> None:
    args = parse_args()
    spec = DATASET_SPECS[args.dataset]
    key_col = spec["key_col"]
    axes_spec: dict[str, dict] = spec["axes"]
    cfg = load_cfg(spec)

    print(f"[实验 P] 数据集={args.dataset}  轴="
          f"{ {a: s['num_classes'] for a, s in axes_spec.items()} }  cv={args.cv}")
    transform = spec["transform"](cfg)
    feature_map = build_feature_cache(
        spec, args.cv, transform, args.batch_size, args.num_workers, args.force_features)

    attr_root = OUTPUTS_ROOT / spec["attr_root"]
    all_metrics: dict = {
        "dataset": args.dataset,
        "axes": {a: s["num_classes"] for a, s in axes_spec.items()},
        "probe": "frozen ImageNet ResNet-18 features + multinomial logistic (in-sample on train)",
        "folds": {},
    }

    for fold in range(args.cv):
        frames = read_fold_frames(spec, args.cv, fold)
        features = {
            name: np.stack([feature_map[k] for k in df[key_col].astype(str)])
            for name, df in frames.items()
        }
        fold_metrics: dict[str, dict] = {}
        per_split_axes: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
            name: {} for name in frames
        }

        for axis, axis_spec in axes_spec.items():
            labels = {name: axis_labels(df, axis_spec) for name, df in frames.items()}
            result = fit_attr_probe(features, labels, num_classes=axis_spec["num_classes"])
            for name in frames:
                per_split_axes[name][axis] = (result.splits[name].probs, labels[name])
            fold_metrics[axis] = result.to_metrics()
            tr, te = result.splits["train"], result.splits["test"]
            print(f"  fold{fold} · {axis:5s}: T={result.temperature:.3f}  "
                  f"acc train={tr.accuracy:.3f} / test={te.accuracy:.3f} (Δ={tr.accuracy-te.accuracy:+.3f})  "
                  f"熵 train={tr.mean_entropy:.3f} / test={te.mean_entropy:.3f}  ECE test={te.ece:.3f}")

        for name, df in frames.items():
            save_fold_predictions(attr_root / f"fold{fold}", name,
                                  keys=df[key_col].astype(str).to_numpy(),
                                  axes=per_split_axes[name])
        all_metrics["folds"][f"fold{fold}"] = {"axes": fold_metrics}

    metrics_path = attr_root / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\n指标已写入 {metrics_path}")

    for axis in axes_spec:
        acc_gaps = [m["axes"][axis]["train"]["accuracy"] - m["axes"][axis]["test"]["accuracy"]
                    for m in all_metrics["folds"].values()]
        ent_gaps = [m["axes"][axis]["test"]["mean_entropy"] - m["axes"][axis]["train"]["mean_entropy"]
                    for m in all_metrics["folds"].values()]
        print(f"in-sample 尖锐度诊断 · {axis}: accuracy 差 {np.mean(acc_gaps):+.3f}±{np.std(acc_gaps):.3f}，"
              f"熵差(test-train) {np.mean(ent_gaps):+.3f}±{np.std(ent_gaps):.3f}")


if __name__ == "__main__":
    main()
