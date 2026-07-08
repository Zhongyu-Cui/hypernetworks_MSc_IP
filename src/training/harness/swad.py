"""
SWAD 权重平均（比较协议 A3.1）
=============================
比较协议 §1 把 **SWAD**（Cha et al., NeurIPS 2021, *Domain Generalization by Seeking Flat
Minima*）列为独立基线（ERM 训练 + 权重平均，不与 HN 组合、不单独搜参）。本模块实现 A3.1
「权重平均包装器（epoch 区间检测 + 平均）」两件事，与训练流程解耦、可单测：

  1. `select_swad_interval` —— **loss valley 区间检测**：从每 epoch 的验证损失序列，按 SWAD 的
     三个超参 (N_s, N_e, r) 找出「谷底起点 t_s」到「过拟合确认前的终点 t_e」。
  2. `average_state_dicts`  —— **权重平均**：把 [t_s, t_e] 各 epoch 的模型 state_dict 逐张量求均值。

**loss valley 语义（对齐 SWAD 论文 §3.2 的 overfit-aware sampling，epoch 粒度离线版）**：
  - 平滑：`s(i) = mean(loss[i-N_s+1 : i+1])`（宽 N_s 的尾窗滑动平均，抑制单 epoch 噪声）。
  - 起点 `t_s` = 平滑损失最小的 epoch（谷底 / 收敛点）；`L_min = s(t_s)`。
  - 阈值 `thr = r · L_min`（r>1，容忍谷底附近的小幅回升仍属「平坦谷」）。
  - 终点 `t_e`：从 t_s 之后逐 epoch 看 `s(i) > thr`；连续 N_e 个 epoch 越界即判「谷已死」（过拟合），
    回退到越界前最后一个好 epoch 作为 t_e；期间若损失回落到阈值内则计数清零（仍在谷内）。
    始终不越界则 t_e = 末 epoch。
  - 平均区间 = `[t_s, t_e]` 闭区间的**全部** epoch 权重（SWAD 平均谷内所有采样点，非仅端点）。

**默认超参**：N_s=3, N_e=6, r=1.3（SWAD 论文 DomainBed 设置）。本项目 ≤30 epoch、逐 epoch 评估，
epoch 粒度足够；A3.2 负责把训练循环每 epoch 的 (val_loss, state_dict) 喂进来并落地平均权重。

**权重平均的 buffer 处理**：浮点张量（含 conv/fc 权重、BN 的 running_mean/var）逐张量求均值；
非浮点 buffer（如 BN 的 `num_batches_tracked`，int）取区间内**最后一个** epoch 的值（均值无意义）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# SWAD 论文 DomainBed 默认超参
DEFAULT_N_CONVERGE: int = 3     # N_s：平滑窗宽 / 谷底收敛判据
DEFAULT_N_TOLERANCE: int = 6    # N_e：过拟合容忍（连续越界确认谷死）
DEFAULT_TOLERANCE_RATIO: float = 1.3  # r：阈值倍率（thr = r·L_min）


@dataclass(frozen=True)
class SWADInterval:
    """
    loss valley 区间检测结果。

    Attributes:
        t_s        : 谷底起点 epoch 索引（0-based，指向输入损失序列）。
        t_e        : 谷终点 epoch 索引（0-based，闭区间）。
        l_min      : 谷底平滑损失 s(t_s)。
        threshold  : 过拟合阈值 r·L_min。
        smoothed   : 平滑后的损失序列（长度同输入）。
        n_averaged : 区间内 epoch 数（= t_e - t_s + 1）。
    """

    t_s: int
    t_e: int
    l_min: float
    threshold: float
    smoothed: list[float]
    n_averaged: int

    def epoch_indices(self) -> list[int]:
        """区间内的全部 epoch 索引（0-based，闭区间 [t_s, t_e]）。"""
        return list(range(self.t_s, self.t_e + 1))


def _trailing_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """
    宽 `window` 的尾窗滑动平均：s(i) = mean(values[max(0,i-window+1) : i+1])。

    尾窗（非居中）避免用到未来 epoch，与在线检测语义一致；前 window-1 个点用可用范围求均值。

    Args:
        values: [T] 一维损失序列。
        window: 窗宽（≥1）。

    Returns:
        [T] 平滑后序列。
    """
    if window <= 1:
        return values.astype(float).copy()
    out = np.empty(len(values), dtype=float)
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out[i] = values[lo:i + 1].mean()
    return out


def select_swad_interval(
    val_losses: "list[float] | np.ndarray",
    *,
    n_converge: int = DEFAULT_N_CONVERGE,
    n_tolerance: int = DEFAULT_N_TOLERANCE,
    tolerance_ratio: float = DEFAULT_TOLERANCE_RATIO,
) -> SWADInterval:
    """
    A3.1（其一）：从每 epoch 验证损失检测 SWAD 的权重平均区间 [t_s, t_e]。

    语义见模块 docstring。索引均 0-based，指向 `val_losses`（第 i 项 = 第 i 个 epoch 的 val loss）。

    Args:
        val_losses     : 各 epoch 的验证损失（越小越好）；长度 ≥1。
        n_converge     : N_s，平滑窗宽（≥1）。
        n_tolerance    : N_e，过拟合确认所需的连续越界 epoch 数（≥1）。
        tolerance_ratio: r，阈值倍率（>1）。

    Returns:
        SWADInterval。

    Raises:
        ValueError: 输入为空、含非有限值，或超参非法。
    """
    arr = np.asarray(val_losses, dtype=float)
    if arr.size == 0:
        raise ValueError("select_swad_interval 收到空 val_losses。")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"val_losses 含非有限值：{arr!r}")
    if n_converge < 1 or n_tolerance < 1:
        raise ValueError(f"n_converge/n_tolerance 须 ≥1，收到 {n_converge}/{n_tolerance}")
    if tolerance_ratio <= 1.0:
        raise ValueError(f"tolerance_ratio 须 >1，收到 {tolerance_ratio}")

    smoothed = _trailing_moving_average(arr, n_converge)
    t_s = int(np.argmin(smoothed))           # 谷底 = 平滑损失最小 epoch
    l_min = float(smoothed[t_s])
    threshold = tolerance_ratio * l_min

    # 谷终点：从 t_s+1 起，连续 n_tolerance 个 epoch 越界即判谷死，回退到越界前最后一个好 epoch
    t_e = len(arr) - 1
    consecutive = 0
    for i in range(t_s + 1, len(arr)):
        if smoothed[i] > threshold:
            consecutive += 1
            if consecutive >= n_tolerance:
                t_e = i - n_tolerance      # 越界串开始前的最后一个好 epoch
                break
        else:
            consecutive = 0               # 回落到阈值内 → 仍在谷内，计数清零
    # t_e 不应早于 t_s（当过拟合紧随谷底时回退可能越过 t_s）
    t_e = max(t_e, t_s)

    return SWADInterval(
        t_s=t_s, t_e=t_e, l_min=l_min, threshold=threshold,
        smoothed=[float(x) for x in smoothed], n_averaged=t_e - t_s + 1,
    )


def average_state_dicts(
    state_dicts: "list[dict[str, torch.Tensor]]",
) -> "dict[str, torch.Tensor]":
    """
    A3.1（其二）：把多个模型 state_dict 逐张量求均值（SWAD 权重平均）。

    浮点张量（conv/fc 权重、BN running_mean/var 等）取算术平均；非浮点 buffer（如 int 型
    `num_batches_tracked`）均值无意义，取列表中**最后一个** state_dict 的值。所有 state_dict 须同键。

    Args:
        state_dicts: 待平均的 state_dict 列表（长度 ≥1，均在 CPU 或同一 device；键集须一致）。

    Returns:
        平均后的 state_dict（新张量，不改动输入；device/dtype 随输入）。

    Raises:
        ValueError: 列表为空，或各 state_dict 键集不一致。
    """
    if not state_dicts:
        raise ValueError("average_state_dicts 收到空列表。")
    ref_keys = set(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], start=1):
        if set(sd.keys()) != ref_keys:
            raise ValueError(f"第 {i} 个 state_dict 键集与首个不一致（无法平均）。")

    n = len(state_dicts)
    averaged: dict[str, torch.Tensor] = {}
    for key in state_dicts[0].keys():
        ref = state_dicts[0][key]
        if torch.is_floating_point(ref):
            acc = torch.zeros_like(ref, dtype=torch.float64)
            for sd in state_dicts:
                acc += sd[key].to(torch.float64)
            averaged[key] = (acc / n).to(ref.dtype)
        else:
            # 非浮点 buffer（如 num_batches_tracked）：均值无意义，取最后一个的值
            averaged[key] = state_dicts[-1][key].clone()
    return averaged


def swad_average_over_interval(
    val_losses: "list[float] | np.ndarray",
    state_dicts: "list[dict[str, torch.Tensor]]",
    *,
    n_converge: int = DEFAULT_N_CONVERGE,
    n_tolerance: int = DEFAULT_N_TOLERANCE,
    tolerance_ratio: float = DEFAULT_TOLERANCE_RATIO,
) -> "tuple[dict[str, torch.Tensor], SWADInterval]":
    """
    端到端便捷入口：检测 loss valley 区间，再平均该区间内各 epoch 的权重（A3.2 直接调用）。

    `val_losses[i]` 与 `state_dicts[i]` 须一一对应（同为第 i 个 epoch 的产物、等长）。

    Args:
        val_losses : 各 epoch 验证损失（等长于 state_dicts）。
        state_dicts: 各 epoch 的 state_dict（每 epoch 训练后保存的模型权重）。
        n_converge / n_tolerance / tolerance_ratio: SWAD 超参。

    Returns:
        (平均后的 state_dict, SWADInterval)。

    Raises:
        ValueError: val_losses 与 state_dicts 长度不一致。
    """
    if len(val_losses) != len(state_dicts):
        raise ValueError(
            f"val_losses({len(val_losses)}) 与 state_dicts({len(state_dicts)}) 长度须一致。"
        )
    interval = select_swad_interval(
        val_losses, n_converge=n_converge, n_tolerance=n_tolerance,
        tolerance_ratio=tolerance_ratio,
    )
    selected = [state_dicts[i] for i in interval.epoch_indices()]
    return average_state_dicts(selected), interval


# ============================================================
# self-test：区间检测语义 + 权重平均正确性 + 退化/校验
# ============================================================
def _selftest() -> None:
    """验证 loss valley 检测（谷底/谷死/回落清零/单调）、state_dict 平均（浮点/int buffer）、端到端。"""
    # --- 区间检测：平坦谷后过拟合，n_converge=1 无平滑、确定性可断言 ---
    losses = [1.0, 0.7, 0.5, 0.45, 0.46, 0.47, 0.6, 0.75, 0.9]
    #          0    1    2    3(谷) 4     5     6越界 7越界  8
    iv = select_swad_interval(losses, n_converge=1, n_tolerance=2, tolerance_ratio=1.3)
    assert iv.t_s == 3, iv.t_s                     # 谷底 index 3 (0.45)
    assert abs(iv.l_min - 0.45) < 1e-12
    assert abs(iv.threshold - 0.585) < 1e-12       # 1.3*0.45
    # i=4(0.46),5(0.47) 在阈值内；i=6(0.6),7(0.75) 连续 2 次越界 → 谷死，t_e = 7-2 = 5
    assert iv.t_e == 5, iv.t_e
    assert iv.epoch_indices() == [3, 4, 5]

    # --- 回落清零：谷内短暂越界后回落，不应过早判谷死 ---
    losses2 = [1.0, 0.5, 0.62, 0.5, 0.51, 0.9, 0.95]  # thr=1.3*0.5=0.65
    #           0    1(谷)2     3    4     5越界 6越界
    iv2 = select_swad_interval(losses2, n_converge=1, n_tolerance=2, tolerance_ratio=1.3)
    assert iv2.t_s == 1, iv2.t_s
    # i=2(0.62)<=0.65, i=3,4 内, i=5(0.9)越界count1, i=6(0.95)越界count2→谷死 t_e=6-2=4
    assert iv2.t_e == 4, iv2.t_e

    # --- 单调下降：谷底 = 末 epoch，区间退化为单点 ---
    mono = [1.0, 0.8, 0.6, 0.4, 0.3]
    ivm = select_swad_interval(mono, n_converge=1, n_tolerance=2, tolerance_ratio=1.3)
    assert ivm.t_s == 4 and ivm.t_e == 4 and ivm.n_averaged == 1

    # --- 平滑生效：单 epoch 尖峰被 N_s 窗抹平，谷底不被噪声带偏 ---
    noisy = [1.0, 0.5, 0.9, 0.48, 0.49, 0.5, 0.52]  # index2 尖峰
    ivs = select_swad_interval(noisy, n_converge=3, n_tolerance=6, tolerance_ratio=1.3)
    assert len(ivs.smoothed) == len(noisy)
    assert ivs.smoothed[2] != noisy[2]             # 已平滑

    # --- 权重平均：浮点取均值、int buffer 取最后一个 ---
    sds = [
        {"w": torch.tensor([2.0, 4.0]), "bn.num_batches_tracked": torch.tensor(10)},
        {"w": torch.tensor([4.0, 8.0]), "bn.num_batches_tracked": torch.tensor(20)},
        {"w": torch.tensor([6.0, 12.0]), "bn.num_batches_tracked": torch.tensor(30)},
    ]
    avg = average_state_dicts(sds)
    assert torch.allclose(avg["w"], torch.tensor([4.0, 8.0])), avg["w"]          # 均值
    assert avg["bn.num_batches_tracked"].item() == 30                            # 最后一个
    assert avg["w"].dtype == torch.float32                                       # dtype 保持
    # 不改动输入
    assert torch.allclose(sds[0]["w"], torch.tensor([2.0, 4.0]))
    # 键集不一致报错
    try:
        average_state_dicts([{"w": torch.zeros(2)}, {"v": torch.zeros(2)}])
        raise AssertionError("键集不一致应报错")
    except ValueError:
        pass
    try:
        average_state_dicts([]); raise AssertionError("空列表应报错")
    except ValueError:
        pass

    # --- 端到端：区间检测 + 平均一致 ---
    losses_e2e = [1.0, 0.7, 0.5, 0.45, 0.46, 0.47, 0.6, 0.75, 0.9]
    sds_e2e = [{"w": torch.tensor([float(i)])} for i in range(len(losses_e2e))]
    avg_sd, iv_e2e = swad_average_over_interval(
        losses_e2e, sds_e2e, n_converge=1, n_tolerance=2, tolerance_ratio=1.3)
    # 区间 [3,4,5] → 平均 w = mean(3,4,5) = 4
    assert iv_e2e.epoch_indices() == [3, 4, 5]
    assert abs(avg_sd["w"].item() - 4.0) < 1e-12
    # 长度不一致报错
    try:
        swad_average_over_interval([0.1, 0.2], sds_e2e); raise AssertionError("长度不一致应报错")
    except ValueError:
        pass

    # --- 输入校验 ---
    for bad in [lambda: select_swad_interval([]),
                lambda: select_swad_interval([0.1, float("inf")]),
                lambda: select_swad_interval([0.1, 0.2], tolerance_ratio=0.9)]:
        try:
            bad(); raise AssertionError("应报错")
        except ValueError:
            pass

    print("swad A3.1 self-test 全部通过 ✓（谷底/谷死/回落清零/单调/平滑 + 浮点均值/int buffer + 端到端 + 校验）")


if __name__ == "__main__":
    _selftest()
