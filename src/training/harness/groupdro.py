"""
GroupDRO 训练目标（比较协议 §1 基线补强：**训练时使用子群标签**的基线）
========================================================================
Sagawa et al., *Distributionally Robust Neural Networks for Group Shifts*, ICLR 2020
（arXiv:1911.08731）的 **online greedy GroupDRO**（论文 Algorithm 1）实现。

**为什么补这个方法（协议缺口）**：协议 §1 的三个基线里 ERM / SWAD 完全不看属性，ROC 虽用属性
但只在**已训练模型的分数上做事后阈值调整**（零训练）。于是「训练时用子群标签」这一格里**只有 3 个
HN**，导致「HN 用属性 vs 基线不用属性」是**混淆对比**：HN 赢时分不清赢在超网络架构还是赢在见到了
属性；HN 打平时（本项目五库中四库的主结论）也无法排除「是这套属性接入方式没实现好」。
GroupDRO 恰好补上**用子群标签、但不条件化函数**的一格：

    |            | 不用属性        | 用属性                              |
    |------------|----------------|-------------------------------------|
    | 改损失/权重 | ERM, SWAD      | **GroupDRO（本模块）**               |
    | 改决策阈值  | —              | ROC                                 |
    | 改函数(条件化)| —            | HyperHead / HyperFusion / HyperAdapt |

理论上二者依赖的信号也不同：HN 的条件化要兑现增益**必须** `I(Y;A|X)>0`（否则盲池化已是 Bayes 最优）；
GroupDRO 的重加权**不需要**条件信号，它只是拿 Overall 换 worst-group。故在 `I(Y;A|X)≈0` 的数据集上
「HN 打平 ERM 而 GroupDRO 仍可能抬 worst-group」是可检验的预测。

算法（论文 Algorithm 1，逐 batch）：
    per-sample 损失 → 按组求组内均值 L_g（batch 内无该组样本时记 0）
    q_g ← q_g · exp(η · (L_g + C/√n_g)) ，随后归一化       # 指数梯度上升（对抗侧）
    robust_loss = Σ_g q_g · L_g                             # 模型侧照常反传
其中 η=step_size（论文默认 0.01），C=group_adjustment（论文 §5 的泛化调整项，默认 0=关闭）。
**与参考实现（Sagawa 官方 `LossComputer`）逐行对齐**：空组的组损失记 0、其 q 乘 exp(η·C/√n_g)
（C=0 时即 ×1 不变），q 的更新在 `no_grad` 下用 detach 的损失做，不参与反传。

**分组定义 = 评估侧 canonical joint 子群**（`harness.subgroup_auc.subgroup_masks` 是单一事实来源）：
训练所优化的组与报告 worst-group 所 min 的组必须是同一批，否则「优化了 A 组、报告 B 组」不可解释。
故本模块不另立分组表，而是复用 `subgroup_masks` 的**交叉键**（含 `|` 的键）作为划分；对只有单一属性轴
的 Fitzpatrick（无交叉键）退化为 6 个肤色子群。构造时断言该批键在样本上**互斥且完备**（每样本恰属
一组）——HAM 的 `age_group==-1`（0-20 排除组）不属任何交叉键，故 HAM 侧训练须先过滤 `age>=0`
（与 3 个 HN 训练脚本同口径），否则此处显式报错而非静默漏样本。

各数据集组数：mimic/chexpert = Sex×Race×Age 8；ham10000 = Sex×Age 8；papila = Sex×Age 4；
fitzpatrick = Skin 6。

自检：`python -m src.training.harness.groupdro`
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.harness.subgroup_auc import subgroup_masks

# 论文默认超参（Sagawa et al. 2020 §4 / 官方实现 `--robust_step_size 0.01`）。
# **不进比较协议的超参网格**：网格是 lr×wd 6 配置，全方法一致；若为 GroupDRO 额外搜 η，
# 它就比其它方法多一个选择自由度，破坏公平比较。故固定为论文默认，敏感性另做（CLI 可覆盖）。
DEFAULT_STEP_SIZE: float = 0.01
DEFAULT_GROUP_ADJUSTMENT: float = 0.0

# 各数据集构造分组所需的属性键（与 subgroup_masks 的 kwargs 同名，也与 train_loop 的 attrs 键同名）
GROUP_ATTRS: dict[str, tuple[str, ...]] = {
    "mimic": ("sex", "race", "age"),
    "chexpert": ("sex", "race", "age"),
    "ham10000": ("sex", "age"),
    "papila": ("sex", "age"),
    "fitzpatrick": ("skin",),
}


class GroupIndexer:
    """
    把 `subgroup_masks` 的 canonical 子群定义复用为**训练侧的互斥完备划分**，逐 batch 给出组序号。

    键的选取：优先取交叉键（含 `|`，即 Sex×Race×Age / Sex×Age），它们天然互斥完备；单一属性轴的
    数据集（Fitzpatrick）无交叉键，退化为全部边缘键（6 个肤色组，同样互斥完备）。键顺序取自
    `subgroup_masks` 的固定 schema（与数据取值无关），故组序号在所有 batch / 折 / 方法间一致。
    """

    def __init__(self, dataset: str) -> None:
        """
        Args:
            dataset: {"mimic","chexpert","ham10000","fitzpatrick","papila"} 之一。

        Raises:
            ValueError: 数据集未在 GROUP_ATTRS 中登记。
        """
        if dataset not in GROUP_ATTRS:
            raise ValueError(f"GroupDRO 未登记数据集 {dataset!r}，合法：{tuple(GROUP_ATTRS)}")
        self.dataset = dataset
        self.attr_names: tuple[str, ...] = GROUP_ATTRS[dataset]
        # 用零长度数组探出固定 schema 的键顺序（subgroup_masks 的键与数据取值无关，见其模块文档）
        empty = {name: np.zeros(0, dtype=int) for name in self.attr_names}
        all_keys = list(subgroup_masks(dataset, **empty).keys())
        joint = [k for k in all_keys if "|" in k]
        self.keys: list[str] = joint if joint else all_keys

    @property
    def n_groups(self) -> int:
        """划分的组数（= worst-group 报告所 min 的交叉子群数）。"""
        return len(self.keys)

    def index_numpy(self, **attrs: np.ndarray) -> np.ndarray:
        """
        给定 numpy 属性数组，返回每样本的组序号 [N]（int64）。

        Args:
            **attrs: 该数据集所需的属性数组（键见 `attr_names`），长度一致。

        Returns:
            [N] 组序号，取值 ∈ [0, n_groups)。

        Raises:
            ValueError: 存在样本不属于任何组或同时属于多组（划分被破坏）。
                HAM 上最常见的成因是未过滤 `age_group==-1`。
        """
        kw = {name: np.asarray(attrs[name]).astype(int) for name in self.attr_names}
        masks = subgroup_masks(self.dataset, **kw)
        stacked = np.stack([masks[k] for k in self.keys])          # [G, N]
        hits = stacked.sum(axis=0)
        if not np.all(hits == 1):
            bad = int((hits != 1).sum())
            raise ValueError(
                f"GroupDRO 分组被破坏：{bad}/{len(hits)} 个样本不恰属于一个组"
                f"（dataset={self.dataset}，组数={self.n_groups}）。"
                f"HAM 上通常是未过滤 age_group==-1（0-20 排除组）——该组不在 canonical 交叉子群内。"
            )
        return stacked.argmax(axis=0).astype(np.int64)

    def index(self, attrs: "dict[str, torch.Tensor]") -> torch.Tensor:
        """
        train_loop 的 attrs（CPU Tensor 字典）→ 组序号 LongTensor [B]。

        Args:
            attrs: {属性名 -> [B] Tensor}，须含 `attr_names` 全部键。

        Returns:
            [B] int64 Tensor（CPU；调用方自行搬 device）。
        """
        missing = [k for k in self.attr_names if k not in attrs]
        if missing:
            raise ValueError(f"attrs 缺少 GroupDRO 所需属性 {missing}（dataset={self.dataset}）")
        np_attrs = {k: attrs[k].detach().cpu().numpy() for k in self.attr_names}
        return torch.from_numpy(self.index_numpy(**np_attrs))

    def counts_numpy(self, **attrs: np.ndarray) -> np.ndarray:
        """训练集各组样本数 [G]（用于 group adjustment 的 C/√n_g 与日志诊断）。"""
        idx = self.index_numpy(**attrs)
        return np.bincount(idx, minlength=self.n_groups).astype(np.int64)

    def counts_from_dataset(self, dataset_obj) -> np.ndarray:
        """
        直接从 Dataset 对象的属性数组统计各组样本数（免去调用方逐数据集拼 kwargs）。

        依赖各 Dataset 的**既有命名约定**（与 `src/utils/resampling._DIM_TO_ATTR` 同一套）：
        `sex→sexes` / `race→races` / `age→age_groups` / `skin→skins`。

        Args:
            dataset_obj: 已构建（且已按需过滤）的训练集 Dataset。

        Returns:
            [G] 各组样本数。
        """
        arrays = {name: np.asarray(getattr(dataset_obj, ATTR_ARRAY_NAME[name]))
                  for name in self.attr_names}
        return self.counts_numpy(**arrays)


# 属性名 → Dataset 上的数组属性名（与 src/utils/resampling._DIM_TO_ATTR 保持一致；
# 该模块的 dims 与本模块的 attr_names 用的是同一批名字，故两处必须同步）
ATTR_ARRAY_NAME: dict[str, str] = {
    "sex": "sexes", "race": "races", "age": "age_groups", "skin": "skins",
}


class GroupDROObjective:
    """
    可调用的 GroupDRO 训练目标（替换 train_loop 里的 `criterion(logits, labels)`）。

    只改**训练损失**，不改模型、不改验证/测试评估：val 仍用普通 BCE 均值算 loss（早停与 SWAD 口径
    保持与全部其它方法一致，协议决策① = Overall AUC 早停）。

    Attributes:
        indexer   : 分组器（组定义 = 评估侧 canonical 交叉子群）。
        q         : [G] 对抗侧组权重（单纯形上），随训练在线更新。
        step_size : η，指数梯度步长。
        adjustment: [G] 组调整项 C/√n_g（C=0 时全 0）。
    """

    def __init__(
        self,
        dataset: str,
        *,
        step_size: float = DEFAULT_STEP_SIZE,
        group_adjustment: float = DEFAULT_GROUP_ADJUSTMENT,
        train_group_counts: np.ndarray | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        """
        Args:
            dataset          : 数据集 key（决定分组 schema）。
            step_size        : η，q 的指数梯度步长（论文默认 0.01）。
            group_adjustment : C，泛化调整系数；>0 时按 C/√n_g 给小组额外加权（需 train_group_counts）。
            train_group_counts: [G] 训练集各组样本数；仅 group_adjustment>0 时必需。
            device           : q 与损失所在设备。

        Raises:
            ValueError: group_adjustment>0 但未提供 train_group_counts，或计数长度不匹配。
        """
        self.indexer = GroupIndexer(dataset)
        n_groups = self.indexer.n_groups
        self.device = torch.device(device)
        self.step_size = float(step_size)
        # q 初始化为均匀分布（= 训练起点等价于「各组等权」，而非等样本权重的 ERM）
        self.q = torch.full((n_groups,), 1.0 / n_groups, device=self.device, dtype=torch.float32)
        self.criterion_none = nn.BCEWithLogitsLoss(reduction="none")

        if group_adjustment and group_adjustment > 0.0:
            if train_group_counts is None:
                raise ValueError("group_adjustment>0 需提供 train_group_counts（用于 C/√n_g）")
            counts = np.asarray(train_group_counts, dtype=np.float64)
            if counts.shape != (n_groups,):
                raise ValueError(f"train_group_counts 形状应为 ({n_groups},)，实为 {counts.shape}")
            # 空组的 √n_g=0 会得到 inf：用 1 兜底（该组训练中永不出现，权重无意义）
            safe = np.where(counts > 0, counts, 1.0)
            adj = float(group_adjustment) / np.sqrt(safe)
        else:
            adj = np.zeros(n_groups, dtype=np.float64)
        self.adjustment = torch.tensor(adj, device=self.device, dtype=torch.float32)
        self.group_adjustment = float(group_adjustment)
        self.train_group_counts = (None if train_group_counts is None
                                   else np.asarray(train_group_counts, dtype=np.int64))

        # epoch 级统计（仅日志用）：各组累计损失与样本数，以及「出现空组的 batch 数」
        # （空组是组损失抖动的来源；组均匀采样下应恒为 0，standard 采样下会显著 >0）
        self._epoch_loss_sum = np.zeros(n_groups, dtype=np.float64)
        self._epoch_count = np.zeros(n_groups, dtype=np.int64)
        self._epoch_batches = 0
        self._epoch_batches_with_empty_group = 0

    @property
    def n_groups(self) -> int:
        """组数。"""
        return self.indexer.n_groups

    def __call__(
        self, logits: torch.Tensor, targets: torch.Tensor, attrs: "dict[str, torch.Tensor]",
    ) -> torch.Tensor:
        """
        一个 batch 的 GroupDRO 目标值（可反传），并就地更新 q。

        Args:
            logits : [B,1] 模型输出（未过 sigmoid）。
            targets: [B,1] float 标签（0/1），与 train_loop 的 `label_col` 同形。
            attrs  : train_loop 的属性字典（CPU Tensor），用于取组序号。

        Returns:
            标量 Tensor：robust_loss = Σ_g q_g · L_g。
        """
        per_sample = self.criterion_none(logits, targets).squeeze(1)        # [B]
        group_idx = self.indexer.index(attrs).to(per_sample.device)         # [B]
        onehot = F.one_hot(group_idx, self.n_groups).float()                # [B,G]

        counts = onehot.sum(dim=0)                                          # [G] batch 内各组样本数
        # 空组：分母置 1 → 组损失 = 0（与 Sagawa 官方 LossComputer.compute_group_avg 一致）
        denom = counts + (counts == 0).float()
        group_loss = (onehot * per_sample.unsqueeze(1)).sum(dim=0) / denom  # [G]

        # --- 对抗侧：指数梯度上升更新 q（不参与反传）---
        with torch.no_grad():
            adjusted = group_loss.detach() + self.adjustment.to(group_loss.device)
            self.q = self.q.to(group_loss.device)
            self.q = self.q * torch.exp(self.step_size * adjusted)
            self.q = self.q / self.q.sum()

            # epoch 统计（按真实样本数加权，空组不计）
            c = counts.detach().cpu().numpy()
            self._epoch_loss_sum += (group_loss.detach().cpu().numpy() * c)
            self._epoch_count += c.astype(np.int64)
            self._epoch_batches += 1
            self._epoch_batches_with_empty_group += int((c == 0).any())

        return torch.dot(self.q, group_loss)

    def reset_epoch_stats(self) -> None:
        """清空 epoch 级组损失统计（每个 epoch 开始时调用）。"""
        self._epoch_loss_sum[:] = 0.0
        self._epoch_count[:] = 0
        self._epoch_batches = 0
        self._epoch_batches_with_empty_group = 0

    def epoch_summary(self) -> str:
        """
        本 epoch 的组损失与当前 q 的单行摘要（供训练日志观察「哪个组被顶起来了」）。

        Returns:
            形如 `GroupDRO | worst=sex:F|age:80+ L=0.51 q=0.21 | q范围[0.07,0.21]` 的字符串。
        """
        seen = self._epoch_count > 0
        if not seen.any():
            return "GroupDRO | 本 epoch 无样本统计"
        mean_loss = np.full(self.n_groups, np.nan)
        mean_loss[seen] = self._epoch_loss_sum[seen] / self._epoch_count[seen]
        worst = int(np.nanargmax(mean_loss))
        q = self.q.detach().cpu().numpy()
        empty = (f"{self._epoch_batches_with_empty_group}/{self._epoch_batches}"
                 if self._epoch_batches else "0/0")
        return (f"GroupDRO | 组损失最高={self.indexer.keys[worst]} "
                f"L={mean_loss[worst]:.4f} q={q[worst]:.3f} | "
                f"q∈[{q.min():.3f},{q.max():.3f}] (均匀={1.0 / self.n_groups:.3f}) | "
                f"空组 batch={empty}")

    def describe(self) -> str:
        """构造参数与分组的多行摘要（训练开始时打印一次）。"""
        lines = [
            f"GroupDRO: {self.n_groups} 组（= canonical 交叉子群，与 worst-group 报告口径同源）",
            f"          η(step_size)={self.step_size}  C(group_adjustment)={self.group_adjustment}",
        ]
        if self.train_group_counts is not None:
            pairs = ", ".join(f"{k}:{n}" for k, n in
                              zip(self.indexer.keys, self.train_group_counts.tolist()))
            lines.append(f"          训练集各组 n = {pairs}")
        return "\n".join(lines)


# ============================================================
# self-test：分组划分正确性 + q 的指数梯度动力学 + 与 ERM 的退化一致性
# ============================================================
def _selftest() -> None:
    """
    覆盖四件事：
      1. 分组 = canonical 交叉子群，且互斥完备；HAM 的 age=-1 会显式报错（不静默漏样本）。
      2. η=0 时 q 恒均匀，且在**各组等大**时 robust_loss ≡ 普通 BCE 均值（GroupDRO 退化为 ERM）。
      3. η>0 时 q 单调偏向高损失组（指数梯度上升方向正确），并与手算 softmax 形式逐位一致。
      4. group_adjustment>0 时小组获得更大权重（C/√n_g 的方向正确）。
    """
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # --- 1. 分组正确性 ---
    idxr = GroupIndexer("ham10000")
    assert idxr.n_groups == 8, idxr.n_groups
    assert all("|" in k for k in idxr.keys), idxr.keys
    n = 400
    sex = rng.integers(0, 2, n)
    age = rng.integers(0, 4, n)
    gid = idxr.index_numpy(sex=sex, age=age)
    assert gid.min() >= 0 and gid.max() < 8
    # 与手工混合进制编码一致（键序 = sex 外层、age 内层，见 subgroup_auc._ham_masks）
    assert np.array_equal(gid, sex * 4 + age), "组序号与 subgroup_masks 键序不一致"
    # age=-1（0-20 排除组）不属任何交叉子群 → 必须报错
    try:
        idxr.index_numpy(sex=np.array([0]), age=np.array([-1]))
        raise AssertionError("age=-1 未触发划分破坏报错")
    except ValueError as e:
        assert "age_group==-1" in str(e)
    # 其余数据集组数
    assert GroupIndexer("mimic").n_groups == 8 and GroupIndexer("chexpert").n_groups == 8
    assert GroupIndexer("papila").n_groups == 4
    fitz = GroupIndexer("fitzpatrick")
    assert fitz.n_groups == 6 and all("|" not in k for k in fitz.keys)  # 单轴退化为 6 肤色组

    # --- 2. η=0 → q 均匀；各组等大时 robust_loss == 普通 BCE 均值 ---
    b = 64
    logits = torch.randn(b, 1)
    targets = torch.randint(0, 2, (b, 1)).float()
    # 构造各组恰好等大的 batch（8 组 × 8 样本）
    g_balanced = np.repeat(np.arange(8), b // 8)
    attrs_bal = {"sex": torch.from_numpy(g_balanced // 4), "age": torch.from_numpy(g_balanced % 4)}
    obj0 = GroupDROObjective("ham10000", step_size=0.0)
    loss0 = obj0(logits, targets, attrs_bal)
    plain = nn.BCEWithLogitsLoss()(logits, targets)
    assert torch.allclose(obj0.q, torch.full((8,), 0.125)), obj0.q
    assert torch.allclose(loss0, plain, atol=1e-6), (loss0.item(), plain.item())

    # --- 3. η>0 → q 偏向高损失组，且与手算指数梯度逐位一致 ---
    eta = 0.5
    obj = GroupDROObjective("ham10000", step_size=eta)
    # 让组 7 的损失显著更高：其样本 logit 与标签强烈相反
    logits2 = torch.full((b, 1), -4.0)
    targets2 = torch.zeros(b, 1)
    targets2[g_balanced == 7] = 1.0                     # 组 7 全部预测错
    _ = obj(logits2, targets2, attrs_bal)
    q = obj.q.numpy()
    assert int(np.argmax(q)) == 7, q
    # 手算：q ∝ exp(η·L_g)（初始均匀 → 一步后正比于 exp）
    per = nn.BCEWithLogitsLoss(reduction="none")(logits2, targets2).squeeze(1).numpy()
    manual_L = np.array([per[g_balanced == g].mean() for g in range(8)])
    manual_q = np.exp(eta * manual_L) / np.exp(eta * manual_L).sum()
    assert np.allclose(q, manual_q, atol=1e-6), (q, manual_q)
    # 多步后仍单调偏向组 7（累积）
    q_before = obj.q[7].item()
    for _ in range(5):
        _ = obj(logits2, targets2, attrs_bal)
    assert obj.q[7].item() > q_before, (q_before, obj.q[7].item())
    assert "GroupDRO |" in obj.epoch_summary()
    obj.reset_epoch_stats()
    assert "无样本统计" in obj.epoch_summary()

    # --- 4. group adjustment：小组（n 小）拿到更大权重 ---
    counts = np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000, 25])   # 组 7 最小
    obj_adj = GroupDROObjective("ham10000", step_size=eta, group_adjustment=1.0,
                                train_group_counts=counts)
    # 全组损失相同 → 差异只能来自 C/√n_g
    same_logits = torch.zeros(b, 1)
    same_targets = torch.ones(b, 1)
    _ = obj_adj(same_logits, same_targets, attrs_bal)
    assert int(np.argmax(obj_adj.q.numpy())) == 7, obj_adj.q
    assert "训练集各组 n" in obj_adj.describe()

    print("groupdro self-test 全部通过 ✓（分组=canonical交叉子群/age=-1报错/η=0退化为ERM/"
          "指数梯度方向与手算一致/group adjustment 方向正确）")


if __name__ == "__main__":
    _selftest()
