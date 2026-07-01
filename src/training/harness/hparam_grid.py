"""
超参网格遍历器（比较协议 A0.1）
==============================
把比较协议 §3「训练与模型选择」规定的超参搜索网格参数化为一个可复用模块，
供所有数据集的训练脚本（ERM + 3 HN）与 SLURM array 脚本共用。

**权威规格**（docs/comparison_protocol.md §3）：
    lr ∈ {3e-5, 1e-4, 3e-4} × wd ∈ {1e-4, 1e-3} = 6 配置，搜索阶段每配置 1 seed；
    固定单配置 (lr=1e-4, wd=1e-4) 作为网格中心点。

本模块只负责「枚举 6 个配置并赋予稳定的 index / 文件名 tag」这一件事（A0.1）：
  - **index**：0..5 的确定序号，供 SLURM array（`--array=0-5`）把 array task 映射到唯一配置；
    顺序固定为 lr 外层、wd 内层，改动会破坏已提交作业与 checkpoint 的对应关系，勿轻易调整。
  - **tag**：形如 `lr3e-05_wd1e-04` 的文件名安全字符串，供 checkpoint / 日志命名，
    与 index 一一对应且人类可读（Pareto 选择阶段据此回溯配置）。

不负责：子群 AUC 向量日志（A0.2）、seed 参数化（A0.3）、脚本改造（A0.4）——各自独立实现。

命令行自查（不参与训练，仅供人工核对网格）：
    python -m src.training.harness.hparam_grid            # 打印全部 6 个配置
    python -m src.training.harness.hparam_grid --index 2  # 打印 array index=2 的配置
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterator

# ============================================================
# 权威网格定义（比较协议 §3）
# ============================================================
# 顺序即 index 顺序：lr 为外层循环、wd 为内层循环。**此顺序被 SLURM array index 依赖，
# 一旦有作业按某 index 提交/落盘，就不能再改动这两个元组的顺序或元素。**
LEARNING_RATES: tuple[float, ...] = (3e-5, 1e-4, 3e-4)
WEIGHT_DECAYS: tuple[float, ...] = (1e-4, 1e-3)

# 网格中心点（固定单配置）：既是网格的一员，也是 pilot / sanity check 时的默认落点。
CENTER_LR: float = 1e-4
CENTER_WD: float = 1e-4


def _fmt_hparam(value: float) -> str:
    """
    把学习率 / 权重衰减格式化为紧凑、文件名安全、无歧义的字符串（如 3e-05 / 1e-04）。

    用 `:.0e` 科学计数法保证 3e-5 与 1e-4 这类跨数量级取值长度一致、可读且不含小数点歧义，
    便于拼进 checkpoint 文件名与日志键。

    Args:
        value: 待格式化的超参数值（正浮点数）。

    Returns:
        形如 "3e-05" 的字符串。
    """
    return f"{value:.0e}"


@dataclass(frozen=True)
class HParamConfig:
    """
    单个超参搜索候选配置。

    frozen=True 使其可哈希、不可变，可安全用作 dict 键或放入集合（Pareto 选择阶段按配置聚合时用）。

    Attributes:
        index: 网格内的确定序号（0..5），供 SLURM array 索引；顺序为 lr 外层、wd 内层。
        lr: 学习率（AdamW）。
        wd: 权重衰减（AdamW）。
    """

    index: int
    lr: float
    wd: float

    @property
    def is_center(self) -> bool:
        """该配置是否为网格中心点 (lr=1e-4, wd=1e-4)。"""
        return self.lr == CENTER_LR and self.wd == CENTER_WD

    @property
    def tag(self) -> str:
        """
        文件名安全的配置标识，形如 `lr3e-05_wd1e-04`，与 index 一一对应。

        用于 checkpoint / 日志命名，使 Pareto 选择阶段能从文件名无歧义地回溯到 (lr, wd)。
        """
        return f"lr{_fmt_hparam(self.lr)}_wd{_fmt_hparam(self.wd)}"

    def describe(self) -> str:
        """人类可读的一行描述，供命令行自查与训练日志抬头打印。"""
        center = "  (center)" if self.is_center else ""
        return f"[{self.index}] {self.tag}  lr={self.lr:.0e}  wd={self.wd:.0e}{center}"


def grid_size() -> int:
    """网格配置总数（= len(LEARNING_RATES) × len(WEIGHT_DECAYS) = 6）。SLURM array 上界据此设置。"""
    return len(LEARNING_RATES) * len(WEIGHT_DECAYS)


def iter_hparam_grid() -> Iterator[HParamConfig]:
    """
    按确定顺序（lr 外层、wd 内层）遍历全部 6 个超参配置。

    Yields:
        HParamConfig，index 从 0 递增到 grid_size()-1。
    """
    index = 0
    for lr in LEARNING_RATES:
        for wd in WEIGHT_DECAYS:
            yield HParamConfig(index=index, lr=lr, wd=wd)
            index += 1


def get_hparam_config(index: int) -> HParamConfig:
    """
    取网格中第 `index` 个配置（供 SLURM array：`--array=0-5` → `$SLURM_ARRAY_TASK_ID`）。

    Args:
        index: 网格序号，须落在 [0, grid_size())。

    Returns:
        对应的 HParamConfig。

    Raises:
        IndexError: index 超出 [0, grid_size()) 范围（避免 array 上界配错时静默取错配置）。
    """
    if not 0 <= index < grid_size():
        raise IndexError(
            f"超参网格 index={index} 越界，合法范围 [0, {grid_size()})；"
            f"检查 SLURM --array 上界是否与 grid_size()={grid_size()} 一致。"
        )
    # 直接由 index 反解 (lr, wd)，避免遍历；与 iter_hparam_grid 的 lr 外层/wd 内层顺序严格一致
    lr = LEARNING_RATES[index // len(WEIGHT_DECAYS)]
    wd = WEIGHT_DECAYS[index % len(WEIGHT_DECAYS)]
    return HParamConfig(index=index, lr=lr, wd=wd)


def _parse_args() -> argparse.Namespace:
    """解析命令行自查参数。"""
    parser = argparse.ArgumentParser(
        description="超参网格自查工具（比较协议 A0.1）：打印网格或某 index 的配置"
    )
    parser.add_argument(
        "--index", type=int, default=None,
        help="只打印该 array index 对应的配置；缺省时打印全部 6 个配置。",
    )
    return parser.parse_args()


def main() -> None:
    """命令行入口：打印网格供人工核对（不参与训练）。"""
    args = _parse_args()
    if args.index is not None:
        print(get_hparam_config(args.index).describe())
        return
    print(f"超参搜索网格（比较协议 §3）：共 {grid_size()} 个配置，"
          f"SLURM array 用 --array=0-{grid_size() - 1}")
    for cfg in iter_hparam_grid():
        print("  " + cfg.describe())


if __name__ == "__main__":
    main()
