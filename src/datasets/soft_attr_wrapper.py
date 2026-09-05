"""
SoftAttrDataset：给任一现有 Dataset 追加子群分类器 g 的属性概率 p̂
==================================================================
方案见 `docs/predicted_attribute_hyperadapt_plan.md` §4。实验 P 的训练侧需要每个样本带上
g 输出的 [C] 概率向量作为超网络的条件输入，而各数据集的 Dataset 类返回元组形状各异。
本模块用**包装**而非改写各 Dataset 来实现：`SoftAttrDataset` 持有 base dataset 与一份
按行序对齐好的 [N, C] 概率矩阵，`__getitem__` 返回 `(*base_item, prob)`。

⚠️ **对齐必须按样本唯一键 join，禁止按行号**：训练脚本可能就地过滤 base dataset
（如 HAM 的 `age_group>=0`），行序与 CSV 不再一致。构造时传入与 base **当前行序**一致的
键数组（由训练脚本从 base dataset 的路径/属性直接导出），本类负责 join 并断言无缺失。

四种条件输入模式（方案 §3.4 与 §6 的对照臂），在构造时一次性物化成概率矩阵，
训练过程中不再变动，保证同一 run 内条件输入完全确定：
  · soft  : g 的校准概率（主线）
  · hard  : argmax(p̂) 的 one-hot（消融：软概率是否比硬标签好）
  · perm  : 把 p̂ 的行在整个 split 内随机置换（保边际分布、破坏与样本的对应）——
            分离「路由信息」与「纯容量」，是判定任何正向读数的必需对照
  · const : 所有样本用同一个 train 集先验向量（纯容量上界对照）
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

ATTR_MODES = ("soft", "hard", "perm", "const")


def align_probs_by_key(
    keys: Sequence[str], pred_keys: np.ndarray, pred_probs: np.ndarray,
) -> np.ndarray:
    """
    按唯一键把 g 的预测概率对齐到 base dataset 的当前行序。

    Args:
        keys      : [N] base dataset 当前行序对应的样本键。
        pred_keys : [M] 预测文件中的键。
        pred_probs: [M, C] 预测概率。

    Returns:
        [N, C] float32，行序与 keys 一致。

    Raises:
        ValueError: 预测文件的键有重复，或 base 中存在预测文件未覆盖的样本。
    """
    if len(pred_keys) != len(set(pred_keys.tolist())):
        raise ValueError("属性预测文件的 keys 存在重复，无法按键 join")
    index = {k: i for i, k in enumerate(pred_keys.tolist())}
    missing = [k for k in keys if k not in index]
    if missing:
        raise ValueError(
            f"有 {len(missing)} 个样本在属性预测文件中缺失（示例 {missing[:3]}）；"
            "检查 fold / split 是否配对，以及过滤口径是否与生成脚本一致"
        )
    rows = np.array([index[k] for k in keys], dtype=np.int64)
    return pred_probs[rows].astype(np.float32)


def materialize_probs(
    probs: np.ndarray, mode: str, prior: np.ndarray | None = None, seed: int = 0,
) -> np.ndarray:
    """
    按条件输入模式把校准概率物化成实际喂给超网络的矩阵。

    Args:
        probs : [N, C] 校准后的 g 概率（soft）。
        mode  : ATTR_MODES 之一。
        prior : [C] const 模式使用的先验向量（通常为 train 集 p̂ 均值）。
        seed  : perm 模式的置换随机种子（与 run seed 绑定，保证可复现）。

    Returns:
        [N, C] float32。

    Raises:
        ValueError: mode 非法，或 const 模式未提供 prior。
    """
    if mode not in ATTR_MODES:
        raise ValueError(f"未知 attr_mode={mode!r}，合法：{ATTR_MODES}")
    if mode == "soft":
        return probs.astype(np.float32)
    if mode == "hard":
        hard = np.zeros_like(probs, dtype=np.float32)
        hard[np.arange(len(probs)), probs.argmax(axis=1)] = 1.0
        return hard
    if mode == "perm":
        rng = np.random.default_rng(seed)
        return probs[rng.permutation(len(probs))].astype(np.float32)
    if prior is None:
        raise ValueError("const 模式必须提供 prior 向量")
    return np.tile(np.asarray(prior, dtype=np.float32), (len(probs), 1))


class SoftAttrDataset(Dataset):
    """
    包装 base dataset，在每个样本的返回元组末尾追加**一份或多份**属性概率向量。

    单轴（Fitzpatrick skin / HAM age / PAPILA age）追加 1 个向量；多轴（MIMIC / CheXpert 的
    sex/race/age）按给定顺序追加多个向量，顺序须与训练侧 unpack 回调的解包顺序一致。

    Args:
        base : 被包装的 Dataset（返回元组，如 Fitzpatrick 的 (image, label, skin)）。
        probs: [N, C] 概率矩阵，或按轴顺序排列的矩阵列表（均已按 base 行序对齐并物化）。

    每个 __getitem__ 返回: (*base[idx], prob_1, ..., prob_M)，每个 prob 为 [C_i] float32 Tensor。
    """

    def __init__(self, base: Dataset, probs: "np.ndarray | list[np.ndarray]") -> None:
        matrices = [probs] if isinstance(probs, np.ndarray) else list(probs)
        for matrix in matrices:
            if len(base) != len(matrix):
                raise ValueError(f"base 长度 {len(base)} 与概率矩阵 {len(matrix)} 不一致")
        self.base = base
        self.probs = [torch.from_numpy(np.ascontiguousarray(m, dtype=np.float32))
                      for m in matrices]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple:
        item = self.base[idx]
        if not isinstance(item, tuple):
            item = (item,)
        return (*item, *[m[idx] for m in self.probs])
