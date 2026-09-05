"""
Run 命名与 seed 参数化（比较协议 A0.3）
======================================
把「一次训练运行」唯一标识为三元组 **(method, config_tag, seed)**，并据此统一生成
checkpoint 路径与 val 日志路径，使超参搜索（6 config × 1 seed）与确认（1 config × 5 seed）
的产物互不覆盖、可被 A1 Pareto / A2 显著性按文件名无歧义寻址。

**为什么必须引入 config_tag**：改造前各脚本 checkpoint 命名形如
`hyperfusion_age_malignant_seed44_best_overall.pth`——**不含超参配置**。一旦按协议跑 6 个
lr/wd 配置，它们会写到同一文件、互相覆盖。本模块的 `run_id` 把 config_tag 嵌入命名
（`<method>_<config_tag>_seed<seed>`），从根上消除覆盖。

**seed 规程（对齐协议 §3）**：
  - 搜索阶段 SEEDS_SEARCH = (42,)：每个 lr/wd 配置只跑 1 seed，产出 Pareto 候选池。
  - 确认阶段 SEEDS_CONFIRM = (42,43,44,45,46)：Pareto 选定的配置再跑 5 seed，报均值±95%CI。

本模块只做「命名 + 播种」，不 import 训练/日志逻辑（保持无环）；val_log.py 反向 import 本模块的
`run_id`，使 checkpoint 与 val 日志共用同一 run 标识（单一事实来源）。
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from src.training.harness.hparam_grid import HParamConfig

# ============================================================
# 常量：方法集、seed 规程、model selection 策略
# ============================================================
# 需要训练的方法（ERM + GroupDRO + 3 HN）。SWAD/ROC 由 ERM 派生（不单独训练，故不在此列）。
# `groupdro`（Sagawa et al. ICLR 2020）是**训练时使用子群标签**的基线，与 ERM 单变量差异仅在损失，
# 补上「用属性但不条件化函数」这一对照格（见 harness/groupdro.py 模块文档）。
TRAINABLE_METHODS: tuple[str, ...] = ("erm", "groupdro", "hyperhead", "hyperfusion", "hyperadapt")


def swad_method_name(base_method: str) -> str:
    """
    某方法派生出的 SWAD 变体的 method 名（命名规约的**单一事实来源**，训练侧与分析侧共用）。

    ERM 派生的 SWAD 是**协议 §1 的独立基线**，历史上就叫 `"swad"`，其 checkpoint / 预测 /
    val 日志已大量落盘并被 A1/A2/D 各分析脚本按该名寻址，故保持不变。
    非 ERM 方法（**实验 G**：HN×SWAD 融合，见 docs/hyperadapt_swad_fusion_plan.md）派生的 SWAD
    必须换名为 `"<method>_swad"`——否则同一数据集下只要 HN 的选定超参与 ERM 的选定超参**恰好相同**，
    run_id=(method, config_tag, seed) 就会完全撞车、静默覆写 ERM-SWAD 基线（HAM cv5 即为实例：
    HyperAdapt 选定 `lr3e-05_wd1e-04`，而 ERM 搜索阶段带 SWAD 跑满 6 配置，
    `swad_lr3e-05_wd1e-04` 已存在）。

    Args:
        base_method: 派生来源的方法名（如 "erm" / "hyperadapt" / "hyperadapt_frozen"）。

    Returns:
        SWAD 变体的 method 名："swad"（base 为 erm 时）或 f"{base_method}_swad"。
    """
    return "swad" if base_method == "erm" else f"{base_method}_swad"

# seed 规程（比较协议 §3）
SEEDS_SEARCH: tuple[int, ...] = (42,)                    # 超参搜索：每配置 1 seed
SEEDS_CONFIRM: tuple[int, ...] = (42, 43, 44, 45, 46)    # 确认：选定配置 5 seed

# 两种 model selection 策略（各存一个 checkpoint）
Selection = Literal["overall", "worstcase"]
SELECTIONS: tuple[Selection, ...] = ("overall", "worstcase")

# run_id 解析正则：config_tag 恒以 "lr" 开头、含 "_wd"，据此从 method 与 seed 之间切出
_RUN_ID_RE = re.compile(r"^(?P<method>.+?)_(?P<config_tag>lr.+?_wd.+?)_seed(?P<seed>\d+)$")


def seed_everything(seed: int) -> None:
    """
    播种 Python / NumPy / PyTorch(CPU+CUDA) 的随机数发生器，使各 seed 的运行可区分且可复现。

    仅播种 RNG，不强制 cudnn deterministic（那会显著拖慢卷积训练；本比较只需不同 seed 间
    可区分、同 seed 可复现的统计口径，无需逐 bit 确定性）。

    Args:
        seed: 随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class RunSpec:
    """
    一次训练运行的唯一标识 (method, config_tag, seed)，外加数据集名（用于日志自解释）。

    Attributes:
        dataset   : 数据集名（{"mimic","chexpert","ham10000","fitzpatrick"}）。
        method    : 方法名（TRAINABLE_METHODS 之一）。
        config_tag: 超参配置标识（HParamConfig.tag，如 "lr1e-04_wd1e-04"）。
        seed      : 随机种子。
    """

    dataset: str
    method: str
    config_tag: str
    seed: int

    @classmethod
    def from_hparam(
        cls, dataset: str, method: str, hparam: HParamConfig, seed: int,
    ) -> "RunSpec":
        """从 HParamConfig 直接构造（训练脚本入口常用）。"""
        return cls(dataset=dataset, method=method, config_tag=hparam.tag, seed=seed)

    @property
    def run_id(self) -> str:
        """本运行的 run_id（见模块级 run_id 函数）。"""
        return run_id(self.method, self.config_tag, self.seed)

    def checkpoint_path(self, output_dir: Path | str, selection: Selection) -> Path:
        """本运行某 selection 策略的 checkpoint 路径（见模块级 checkpoint_path 函数）。"""
        return checkpoint_path(output_dir, self.method, self.config_tag, self.seed, selection)


def run_id(method: str, config_tag: str, seed: int) -> str:
    """
    组装 run_id：`<method>_<config_tag>_seed<seed>`，唯一标识一次训练运行。

    checkpoint 与 val 日志均以此为基名，保证 method×config×seed 三元组唯一寻址。

    Args:
        method    : 方法名（如 "hyperfusion"）。
        config_tag: 超参配置标识（如 "lr1e-04_wd1e-04"）。
        seed      : 随机种子。

    Returns:
        形如 "hyperfusion_lr1e-04_wd1e-04_seed44" 的字符串。
    """
    return f"{method}_{config_tag}_seed{seed}"


def parse_run_id(rid: str) -> tuple[str, str, int]:
    """
    从 run_id 反解 (method, config_tag, seed)，供 A1/A2 从文件名回溯运行标识。

    依赖 config_tag 恒以 "lr" 开头、含 "_wd"（hparam_grid.HParamConfig.tag 的固定形态）。

    Args:
        rid: run_id 字符串（可含 _best_overall 等后缀前先自行去除，或传入纯 run_id）。

    Returns:
        (method, config_tag, seed)。

    Raises:
        ValueError: rid 不符合 run_id 形态。
    """
    m = _RUN_ID_RE.match(rid)
    if m is None:
        raise ValueError(f"无法解析 run_id: {rid!r}（期望 <method>_lr..._wd..._seed<int>）")
    return m.group("method"), m.group("config_tag"), int(m.group("seed"))


def checkpoint_path(
    output_dir: Path | str, method: str, config_tag: str, seed: int, selection: Selection,
) -> Path:
    """
    某运行、某 model selection 策略的 checkpoint 路径：
    `<output_dir>/<run_id>_best_<selection>.pth`。

    Args:
        output_dir: 该数据集 outputs 根（如 outputs/ham10000）。
        method / config_tag / seed: 见 run_id。
        selection : "overall"（Overall AUC 选择 / 早停）或 "worstcase"（worst-case AUC 选择）。

    Returns:
        checkpoint 的绝对/相对 Path（不创建目录）。
    """
    if selection not in SELECTIONS:
        raise ValueError(f"未知 selection {selection!r}，合法：{SELECTIONS}")
    return Path(output_dir) / f"{run_id(method, config_tag, seed)}_best_{selection}.pth"


# ============================================================
# self-test
# ============================================================
def _selftest() -> None:
    """验证 run_id 组装/反解 round-trip、路径命名、seed 规程、seed_everything 可复现。"""
    # run_id round-trip（覆盖所有方法 × 全网格 tag）
    from src.training.harness.hparam_grid import iter_hparam_grid
    for method in TRAINABLE_METHODS:
        for cfg in iter_hparam_grid():
            for seed in SEEDS_CONFIRM:
                rid = run_id(method, cfg.tag, seed)
                assert parse_run_id(rid) == (method, cfg.tag, seed), rid

    # 派生 SWAD 方法名（实验 G）：run_id 仍可逆，且与 ERM-SWAD 基线在**同 config_tag** 下不撞名。
    # 回归意义：正则的 method 组非贪婪、config_tag 锚定 "lr"，故 "<method>_swad" 不会被误切；
    # 同 tag 不撞名是硬要求——HAM cv5 下 HyperAdapt 选定 lr3e-05_wd1e-04，而 ERM 搜索带 SWAD 跑满
    # 6 配置，swad_lr3e-05_wd1e-04 已存在，若命名不隔离会静默覆写基线。
    assert swad_method_name("erm") == "swad"                     # 向后兼容：既有产物名不变
    for base in ("hyperhead", "hyperfusion", "hyperadapt", "hyperadapt_frozen"):
        derived = swad_method_name(base)
        assert derived == f"{base}_swad", derived
        for cfg in iter_hparam_grid():
            for seed in SEEDS_CONFIRM:
                rid = run_id(derived, cfg.tag, seed)
                assert parse_run_id(rid) == (derived, cfg.tag, seed), rid
                # 同 config_tag / 同 seed 下与 ERM-SWAD、与派生来源本身均不撞名
                assert rid != run_id("swad", cfg.tag, seed)
                assert rid != run_id(base, cfg.tag, seed)

    # 唯一性：method×config×seed 三元组两两不同名（消除覆盖）
    ids = {
        run_id(m, c.tag, s)
        for m in TRAINABLE_METHODS for c in iter_hparam_grid() for s in SEEDS_CONFIRM
    }
    assert len(ids) == len(TRAINABLE_METHODS) * 6 * len(SEEDS_CONFIRM), len(ids)

    # checkpoint 路径含 config_tag 与 selection，两种 selection 不撞名
    p_ov = checkpoint_path("outputs/ham10000", "hyperfusion", "lr1e-04_wd1e-04", 44, "overall")
    p_wc = checkpoint_path("outputs/ham10000", "hyperfusion", "lr1e-04_wd1e-04", 44, "worstcase")
    assert p_ov.name == "hyperfusion_lr1e-04_wd1e-04_seed44_best_overall.pth"
    assert p_ov != p_wc

    # RunSpec.from_hparam 便捷构造
    cfg2 = next(iter_hparam_grid())
    spec = RunSpec.from_hparam("fitzpatrick", "erm", cfg2, 42)
    assert spec.run_id == f"erm_{cfg2.tag}_seed42"
    assert spec.checkpoint_path("outputs/fitzpatrick", "overall").name.endswith("_best_overall.pth")

    # seed 规程符合协议
    assert SEEDS_SEARCH == (42,) and SEEDS_CONFIRM == (42, 43, 44, 45, 46)

    # seed_everything 可复现：同 seed 两次取样一致，不同 seed 不同
    seed_everything(42); a = (random.random(), float(np.random.rand()), torch.rand(1).item())
    seed_everything(42); b = (random.random(), float(np.random.rand()), torch.rand(1).item())
    seed_everything(43); c = (random.random(), float(np.random.rand()), torch.rand(1).item())
    assert a == b and a != c

    print("run self-test 全部通过 ✓（run_id round-trip + 派生 SWAD 名隔离 + 唯一性 + 路径命名 "
          "+ seed 规程 + 播种可复现）")


if __name__ == "__main__":
    _selftest()
