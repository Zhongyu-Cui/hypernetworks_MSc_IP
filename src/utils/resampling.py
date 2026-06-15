"""
敏感组 × 标签平衡重采样
=======================
为 MIMIC-CXR No Finding 训练构建 `WeightedRandomSampler`，按 (sex, race, age_group, label)
联合分组（十六格）对训练样本做带放回的加权采样，缓解两个数据侧公平性病因：
  (a) 优化器忽视小子群；(b) 各组阳性率差异（主要由 age 驱动）制造的基率/阈值偏差。

强度由温度 alpha ∈ [0, 1] 控制，样本权重 w_i ∝ (1 / n_{cell(i)})^alpha：
  - alpha = 0   → 各样本等权，即自然分布（等价于 shuffle=True，无重采样）
  - alpha = 1   → 各 cell 等概率（完全平衡，最大程度过采样稀有 cell）
  - 0 < alpha < 1 → 介于两者之间的温和平衡

约定：
  * **只对训练集使用**。val / test 必须保持人群真实分布，否则评估指标失去可比性。
  * 与现有框架的唯一区别是把 train DataLoader 的 `shuffle=True` 换成 `sampler=<本采样器>`，
    其余（模型、优化器、batch_size、增强、早停、seed）全部不变，保证干净的 A/B 对比。
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

# 平衡维度名 -> MIMICCXRDataset 上对应的属性数组名
_DIM_TO_ATTR = {
    "sex": "sexes",
    "race": "races",
    "age": "age_groups",
    "label": "labels",
}


def _cell_ids(dataset, dims: list[str]) -> np.ndarray:
    """
    把每个样本的所选维度组合编码成一个整数 cell id。

    Args:
        dataset: MIMICCXRDataset 实例，需含 sexes / races / age_groups / labels 数组。
        dims   : 参与分组的维度名列表，取值来自 {"sex","race","age","label"}。

    Returns:
        [N] int64 数组，每个元素为该样本所属 cell 的编号。
    """
    cols = []
    for dim in dims:
        if dim not in _DIM_TO_ATTR:
            raise ValueError(f"未知的平衡维度 {dim!r}，可选：{list(_DIM_TO_ATTR)}")
        arr = np.asarray(getattr(dataset, _DIM_TO_ATTR[dim])).astype(np.int64)
        cols.append(arr)
    # 各维度均为 0/1 二值，按位混合编码成唯一 cell id（dims 顺序决定编码，不影响分组结果）
    cell = np.zeros(len(cols[0]), dtype=np.int64)
    for arr in cols:
        cell = cell * (int(arr.max()) + 1) + arr
    return cell


def build_group_label_sampler(
    dataset,
    dims: list[str],
    alpha: float,
    verbose: bool = True,
) -> WeightedRandomSampler:
    """
    构建按 (敏感组 × 标签) cell 加权的 WeightedRandomSampler（带放回）。

    样本权重 w_i ∝ (1 / n_{cell(i)})^alpha；num_samples = len(dataset)，故每 epoch 仍抽取
    与原始训练集等量的样本，epoch 长度不变。

    Args:
        dataset: 训练集 MIMICCXRDataset 实例。
        dims   : 平衡维度，如 ["sex","race","age","label"]（十六格联合平衡）。
        alpha  : 平衡强度温度 ∈ [0,1]；0=自然分布，1=各 cell 等概率。
        verbose: 是否打印各 cell 的样本数与有效过采样倍数（便于日志核对）。

    Returns:
        WeightedRandomSampler，可直接传给 DataLoader 的 sampler 参数（此时须 shuffle=False）。
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha 必须在 [0,1]，得到 {alpha}")

    cell = _cell_ids(dataset, dims)
    unique_cells, counts = np.unique(cell, return_counts=True)
    count_of = dict(zip(unique_cells.tolist(), counts.tolist()))

    # 每个 cell 的权重 ∝ (1/n_cell)^alpha；样本权重 = 其所在 cell 的权重
    n_per_sample = np.array([count_of[c] for c in cell], dtype=np.float64)
    sample_weights = n_per_sample ** (-alpha)

    if verbose:
        n_total = len(cell)
        # 抽到样本 i 的概率 = w_i / Σ_j w_j；分母即所有样本权重之和
        denom = float(sample_weights.sum())
        print(f"[resampling] dims={dims}  alpha={alpha}  cells={len(unique_cells)}  "
              f"num_samples/epoch={n_total:,}")
        print(f"[resampling] {'cell_id':>8}{'n':>9}{'每图期望抽样次数/epoch':>24}")
        for c, n in sorted(count_of.items()):
            w = n ** (-alpha)
            per_img = n_total * w / denom   # = num_samples * P(单张该 cell 图被抽中)
            print(f"[resampling] {c:>8}{n:>9}{per_img:>24.2f}")

    weights_tensor = torch.as_tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=len(dataset),
        replacement=True,
    )


# ============================================================
# 自检：打印真实 train 分布下各强度的采样倍数
# ============================================================
if __name__ == "__main__":
    from pathlib import Path

    from src.datasets.mimic_cxr_dataset import MIMICCXRDataset

    REPO_ROOT = Path(__file__).resolve().parents[2]
    split_dir = REPO_ROOT / "data" / "splits" / "mimic_cxr_nofinding"
    # transform=None：自检只用属性数组，不读图像
    ds = MIMICCXRDataset(split_dir / "train.csv", transform=None,
                         image_size="224x224", age_threshold=60)
    print(f"Train set: {len(ds):,} images\n")
    for a in (1.0, 0.5):
        build_group_label_sampler(ds, dims=["sex", "race", "age", "label"], alpha=a, verbose=True)
        print()
