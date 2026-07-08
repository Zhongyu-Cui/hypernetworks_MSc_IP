"""
合成侧信道属性注入机制（评审补强 R1.1）
=======================================
为「合成信号剂量-反应」实验（docs/review_followup_tasks.md R1）提供**可控强度**的敏感属性。
核心论点需要属性携带图像 X 之外的**残余标签信息**（`I(Y;A|X)>0`），这与主流 spurious-correlation
基准（Colored-MNIST / Waterbirds / Spawrious，把属性**画进图像**、故 `I(Y;A|X)≈0`）设定相反，
**不能直接照搬**。本模块因此把属性构造成**旁路侧信道（side channel）**——一个不进图像、只作 HN
条件输入的标量：

    A_syn = Y ⊕ Bernoulli(η)                      （η = 翻转率 flip rate）

对标学界成熟范式：
  - 标签噪声翻转率（label-noise flip rate）——把干净标签以概率 η 翻转。
  - 特权信息 LUPI（Vapnik & Vashist 2009）/ TRAM（Collier et al. 2022）——A_syn 是训练/推理时
    可得、但**不在图像里**的旁路信号，携带 X 无法表达的标签相关信息。

**为什么强度由 η 单调可调（本模块的正确性核心）**：翻转噪声 B(η) 与图像 X **独立**，故
    I(Y;A_syn|X) = E_X[ H(A_syn|X) − H(A_syn|Y,X) ] = E_X[ H(A_syn|X) ] − h_b(η)
其中 h_b 是二元熵、H(A_syn|Y,X)=H(B(η))=h_b(η)（给定 Y 后 A_syn 只剩翻转噪声）。于是
  - η → 0.5：B(η) 为公平硬币、与一切独立 ⇒ A_syn ⊥ (Y, X) ⇒ I(Y;A_syn)=I(Y;A_syn|X)=0（负控制）。
  - η → 0  ：A_syn ≈ Y ⇒ H(A_syn|X)→H(Y|X)、h_b(η)→0 ⇒ I(Y;A_syn|X) → H(Y|X)（残余熵上限）。
  - η ∈ (0, 0.5) 上 I(Y;A_syn|X) 关于 η **严格单调递减**（h_b 单调、且 A_syn 边际越接近 Y 越强）。

据此，扫 η 即在**单一数据集内**注入已知强度的条件信号，把「iff」从跨数据集相关升级为
因果剂量-反应（R1 动机）。真实到目标 nats 档位的标定见 scripts/calibrate_synthetic_signal.py（R1.2）。

本模块只做「注入机制 + 确定性 A_syn 生成」，不含任何训练/模型逻辑（保持无依赖、可单元测试）。
"""

from __future__ import annotations

import numpy as np

# A_syn 是二值属性，用于 HN 单属性通路（复用 num_age=2 的 *Age 变体）与子群划分命名。
SYNTH_ATTR_NAMES: dict[int, str] = {0: "Asyn=0", 1: "Asyn=1"}


def inject_synthetic_attribute(
    labels: np.ndarray, eta: float, rng: np.random.Generator,
) -> np.ndarray:
    """
    构造侧信道合成属性 A_syn = Y ⊕ Bernoulli(η)。

    每个样本独立地以概率 η 翻转其标签得到 A_syn；翻转噪声与图像 X 完全独立，故 A_syn 携带的
    Y 相关信息是 X 之外的残余信息（`I(Y;A_syn|X)>0`，除非 η=0.5）。**不改动任何图像**。

    Args:
        labels: [N] 二值标签 Y（0/1，整数或可转整数）。
        eta:    翻转率 η ∈ [0, 0.5]。η=0 ⇒ A_syn=Y（信号最强）、η=0.5 ⇒ A_syn⊥Y（信号 0）。
        rng:    numpy 随机数发生器（传入固定 seed 的 Generator 以保证 A_syn 确定、可复现）。

    Returns:
        [N] int64 的 A_syn（0/1）。

    Raises:
        ValueError: eta 不在 [0, 0.5]，或 labels 含非 {0,1} 取值。
    """
    if not (0.0 <= eta <= 0.5):
        raise ValueError(f"翻转率 eta 须在 [0, 0.5]，得到 {eta}")
    y = np.asarray(labels).astype(np.int64)
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("labels 必须是二值 {0,1}")
    # Bernoulli(η) 翻转掩码；与 X 独立 ⇒ A_syn 的条件信号强度纯由 η 决定
    flips = (rng.random(y.shape[0]) < eta).astype(np.int64)
    return (y ^ flips).astype(np.int64)


def make_split_rng(eta: float, split: str, base_seed: int = 20260706) -> np.random.Generator:
    """
    为某 (η, split) 生成**确定性**随机数发生器，使 A_syn 在训练 / 评估 / MI 标定三处完全一致。

    A_syn 必须对每个 (η, split) 唯一确定：训练循环、conditional-MI 标定、剂量-反应汇总都要读到
    **同一个** A_syn，否则信号强度对不上。用 (base_seed, η, split) 派生 seed 保证这一点，且不同
    η / split 之间互相独立（避免不同档位共用同一翻转掩码引入伪相关）。

    Args:
        eta:       翻转率（并入 seed，使不同档位 A_syn 独立）。
        split:     'train' / 'val' / 'test'（并入 seed，使各 split A_syn 独立）。
        base_seed: 实验级基种子（默认 R1 建立日期 2026-07-06）。

    Returns:
        np.random.Generator（SeedSequence 由 base_seed + η*1e6 取整 + split 哈希派生）。
    """
    split_code = {"train": 1, "val": 2, "test": 3}.get(split)
    if split_code is None:
        raise ValueError(f"未知 split {split!r}，合法：train/val/test")
    # η*1e6 取整并入 entropy，避免浮点相等问题；split_code 区分三集
    seed_seq = np.random.SeedSequence([base_seed, int(round(eta * 1_000_000)), split_code])
    return np.random.default_rng(seed_seq)


# ============================================================
# self-test：证明 R1.1 完成判据
#   ① η=0.5 时 A_syn ⊥ Y | X（条件独立）；② η 减小时条件依赖单调增；③ 确定性可复现。
# ============================================================
def _plugin_mi_bits(y: np.ndarray, a: np.ndarray) -> float:
    """plug-in 估计 I(Y;A)（bits），仅用于 self-test 的量级比较（plug-in 对 MI 有小正偏）。"""
    n = y.shape[0]
    joint = np.zeros((2, 2))
    for iy in (0, 1):
        for ia in (0, 1):
            joint[iy, ia] = np.sum((y == iy) & (a == ia))
    p = joint / n
    py = p.sum(axis=1, keepdims=True)
    pa = p.sum(axis=0, keepdims=True)
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / (py @ pa)[mask])))


def _conditional_mi_bits(y: np.ndarray, a: np.ndarray, x_bin: np.ndarray) -> float:
    """
    分层 plug-in 估计 I(Y;A|X)（bits）：按 X 的离散分箱分层，箱内算 I(Y;A)，按箱概率加权求和。
    这是「给定 X 后 A 与 Y 的残余依赖」的模型无关近似，直接对应 R1.1 判据里的「|X」。
    """
    n = y.shape[0]
    total = 0.0
    for b in np.unique(x_bin):
        m = x_bin == b
        if m.sum() < 2:
            continue
        total += (m.sum() / n) * _plugin_mi_bits(y[m], a[m])
    return total


def _selftest() -> None:
    """构造 (X,Y) 使 Y 部分可由 X 预测，扫 η 验证 R1.1 三条判据。"""
    rng = np.random.default_rng(0)
    n = 200_000

    # 构造一个部分可预测 Y 的潜变量 X：Y|X=x ~ Bernoulli(sigmoid 型)，X 分 10 箱
    x_latent = rng.standard_normal(n)
    p_y_given_x = 1.0 / (1.0 + np.exp(-1.2 * x_latent))     # X 携带部分 Y 信息（H(Y|X)>0）
    y = (rng.random(n) < p_y_given_x).astype(np.int64)
    x_bin = np.clip(np.digitize(x_latent, np.quantile(x_latent, np.linspace(0, 1, 11)[1:-1])), 0, 9)

    # 残余熵 H(Y|X) 作为 η→0 时 I(Y;A_syn|X) 的理论上限，用于合理性对照
    h_y_given_x = 0.0
    for b in range(10):
        m = x_bin == b
        q = y[m].mean()
        if 0 < q < 1:
            h_y_given_x += (m.sum() / n) * (-q * np.log2(q) - (1 - q) * np.log2(1 - q))

    eta_grid = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    cond_mi = []
    for eta in eta_grid:
        a_syn = inject_synthetic_attribute(y, eta, np.random.default_rng(100))
        cond_mi.append(_conditional_mi_bits(y, a_syn, x_bin))

    # 判据①：η=0.5 时 A_syn ⊥ Y | X（条件 MI ≈ 0，仅剩 plug-in 有限样本正偏）
    assert cond_mi[-1] < 0.01, f"η=0.5 条件 MI 应≈0，得到 {cond_mi[-1]:.4f} bits"
    # 判据②：η 减小 ⇒ 条件依赖单调增（cond_mi 关于 η 严格递减）
    for i in range(len(eta_grid) - 1):
        assert cond_mi[i] > cond_mi[i + 1] - 1e-6, (
            f"单调性破坏：η={eta_grid[i]} I={cond_mi[i]:.4f} 应 >= η={eta_grid[i+1]} I={cond_mi[i+1]:.4f}"
        )
    # 判据②补充：η=0 时 A_syn=Y ⇒ I(Y;A_syn|X) ≈ H(Y|X)（残余熵上限，允许 plug-in 小偏差）
    assert abs(cond_mi[0] - h_y_given_x) < 0.02, (
        f"η=0 条件 MI 应≈H(Y|X)={h_y_given_x:.4f}，得到 {cond_mi[0]:.4f} bits"
    )
    # 边际独立性交叉验证：η=0.5 时 I(Y;A_syn) 边际也≈0
    a_half = inject_synthetic_attribute(y, 0.5, np.random.default_rng(7))
    assert _plugin_mi_bits(y, a_half) < 0.001, "η=0.5 边际 MI 也应≈0"

    # 判据③：确定性——同 (η,split) 两次调用 make_split_rng 得到完全相同的 A_syn
    a1 = inject_synthetic_attribute(y, 0.1, make_split_rng(0.1, "train"))
    a2 = inject_synthetic_attribute(y, 0.1, make_split_rng(0.1, "train"))
    assert np.array_equal(a1, a2), "同 (η,split) 的 A_syn 必须逐元素一致（确定性）"
    # 不同 split / η 的翻转掩码独立（A_syn 不应恒等）
    a3 = inject_synthetic_attribute(y, 0.1, make_split_rng(0.1, "val"))
    assert not np.array_equal(a1, a3), "不同 split 的 A_syn 应独立"

    # 输入校验
    for bad_eta in (-0.1, 0.6):
        try:
            inject_synthetic_attribute(y, bad_eta, rng); assert False
        except ValueError:
            pass

    print("synthetic_attribute self-test 全部通过 ✓")
    print(f"  H(Y|X) 上限 = {h_y_given_x:.4f} bits")
    print("  I(Y;A_syn|X) vs η（bits，应随 η 增大单调降到 0）:")
    for eta, mi in zip(eta_grid, cond_mi):
        print(f"    η={eta:.2f}  I(Y;A_syn|X)={mi:.4f}")


if __name__ == "__main__":
    _selftest()
