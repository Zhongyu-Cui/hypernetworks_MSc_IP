"""
OOD 属性 knockout：判别「机制 A（属性迁移）」vs「机制 B（条件化的正则副作用）」
==============================================================================
`docs/ood_experiment_v2_results.md` §5.1 列出 HyperAdapt 在 OOD 上可能有优势的三条机制。其中：

  · **机制 A**：把 source 上学到的属性条件化优势迁移到 target（需 `I(Y;A|X)>0` 且无 concept shift）。
  · **机制 B**：条件化带来的**正则/容量**效应，**与属性信息无关**。

两者在 AUC 上看起来一样，**只有反事实干预能分开**：拿同一份 HyperAdapt 权重，把喂给超网络的属性
换掉再推断——

  · `real`     ：真实属性（= 既有 full-target 预测，直接复用，不重算）
  · `permute`  ：**全局置换**属性（打破 属性↔图像 配对，但**保持属性边缘分布**）
  · `constant` ：全部置为 target 的 **majority cell**（连边缘分布也抹掉，模型只见「典型病人」）
  · `marginal` ：**输出边缘化**（推荐的严格版）——对 sex×race×age 的 8 种组合各前向一次，
    按 target 经验先验 `p(a)` **加权平均概率**：`p(y|x) = Σ_a p(a)·p(y|x,a)`。

**为何 `marginal` 才是「移除属性信息」的正确定义**：①②③ 式的「在参数空间取平均」（属性 embedding
均值 / profile 均值 / Δθ 均值）得到的是一个**平均模型**，而「不知道属性」在数学上是**边缘化**，
要的是**平均预测**。fuse MLP 含 ReLU、Δθ = A⊗B 是双线性 ⇒ 参数空间平均与预测平均**不等价**，
且只有边缘化有明确概率语义。
`constant` 其实是边缘化的**退化特例**（先验塌缩成点质量 `p(a)=δ_{a*}`）——这正是它的偏倚来源：
majority cell 的 age 恰好等于主指标 worst 组（`age:>=60`）的取值，使该组保留了正确的 age。

**成本**：`marginal` 需 8 次前向，但本作业实测瓶颈在 **NFS 读图（GPU 利用率恒 0%）**，
8 倍计算落在空转的 GPU 上，墙钟时间接近不变。

判读（Δ = HyperAdapt − ERM，marginal worst AUC）：
  · real 下 Δ>0 而 permute/constant 下 Δ→0  ⇒ 增益**依赖属性** ⇒ 机制 A 存活。
  · 三种条件下 Δ 都差不多            ⇒ 增益**与属性无关** ⇒ 机制 B（正则/容量）。

**关键实现点：干预必须包在 `forward_fn` 上，不能包在 `unpack_fn` 上。**
`harness.train_loop.evaluate` 用 `unpack_fn` 解出的 attrs **同时**用于两件事：喂模型（经 forward_fn）
与收集分组键（`attrs_all`）。若在 unpack 处替换，子群分组也会用到假属性，评估口径就被污染了。
包在 forward_fn 上则只改「模型看到什么」，**分组键恒为真实属性**——这是本实验成立的前提。

置换用**全局**而非 batch 内：loader `shuffle=False` ⇒ 行序 == CSV 行序，故可预先对整份属性数组做
一次置换，再按游标逐 batch 取用。batch 内置换会受 batch 局部属性分布影响，不可取。
每个 (fold, trial) 用不同的置换种子，15 个 replicate 天然平滑掉置换随机性。

用法（重型 GPU 推理，经 sbatch）：
    python -m src.training.run_ood_attr_knockout --mode permute
    python -m src.training.run_ood_attr_knockout --mode constant --direction m2c
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from src.training.harness.predictions import save_predictions
from src.training.harness.train_loop import forward_sex_race_age
from src.training.run_ood_cxr_full_target import (
    OUTPUTS_ROOT, build_full_target_loader, checkpoint_path, fold_trial_seeds,
    load_selected_config,
)
from src.training.eval_ood_cxr import evaluate_ood

METHOD = "hyperadapt"          # ERM/SWAD 是 attribute-blind，knockout 对其无定义
ATTR_KEYS = ("sex", "race", "age")


def majority_cell(attrs: dict[str, np.ndarray]) -> dict[str, int]:
    """target 上最常见的 (sex, race, age) 组合——constant 模式把所有样本都置为它。"""
    keys = np.stack([attrs[k] for k in ATTR_KEYS], axis=1)
    uniq, cnt = np.unique(keys, axis=0, return_counts=True)
    top = uniq[cnt.argmax()]
    return {k: int(top[i]) for i, k in enumerate(ATTR_KEYS)}


def build_intervened_attrs(true_attrs: dict[str, np.ndarray], mode: str,
                           seed: int) -> dict[str, np.ndarray]:
    """
    生成**全局**干预后的属性数组（与 CSV 行序对齐）。

    Args:
        true_attrs: 真实属性，各为 [N] 整数数组。
        mode: "permute"（整体置换，保持边缘分布）或 "constant"（全置 majority cell）。
        seed: 置换随机种子（每个 (fold, trial) 不同）。

    Returns:
        同形状的干预后属性字典。
    """
    n = len(next(iter(true_attrs.values())))
    if mode == "permute":
        # 三个属性用**同一个**置换：保持属性之间的联合结构，只打破「属性↔图像」的配对。
        # 若各属性独立置换，会额外破坏 sex/race/age 的相关性，混入第二个变化源。
        perm = np.random.default_rng(seed).permutation(n)
        return {k: np.asarray(v)[perm] for k, v in true_attrs.items()}
    if mode == "constant":
        cell = majority_cell(true_attrs)
        return {k: np.full(n, cell[k], dtype=np.int64) for k in ATTR_KEYS}
    raise ValueError(f"未知 mode {mode!r}")


def empirical_prior(true_attrs: dict[str, np.ndarray]) -> dict[tuple[int, int, int], float]:
    """
    target 上 (sex, race, age) 8 个 cell 的经验先验 p(a)，用于输出边缘化。

    用 target 而非 source 的分布：本实验评估的是模型在 target 上的表现，
    「不知道属性时的最优预测」应当对 **target 的属性分布**积分。
    """
    keys = np.stack([true_attrs[k] for k in ATTR_KEYS], axis=1)
    n = len(keys)
    prior = {}
    for combo in itertools.product((0, 1), repeat=3):
        prior[combo] = float((keys == np.array(combo)).all(axis=1).sum() / n)
    assert abs(sum(prior.values()) - 1.0) < 1e-9, "先验未归一"
    return prior


def make_marginalized_forward(prior: dict[tuple[int, int, int], float]):
    """
    输出边缘化的 forward：`p(y|x) = Σ_a p(a)·p(y|x,a)`，对 8 种属性组合各前向一次。

    **平均概率而非平均 logit**：`Σ_a p(a)·σ(z_a)` 才是边缘化的定义；平均 logit 无概率语义。
    两者一般**不是单调等价**，但在本项目的实际量级下差异可忽略——实测 8 个属性组合间的 logit
    离散度很小（真实 vs 置换属性的中位 |Δlogit| ≈ 0.07），此时两种平均的排序 Kendall τ ≈ 0.9999
    （逆序对 4e-05）；离散度大时才会分开（τ 降至 0.87）。故此处选概率平均是出于**定义正确性**，
    并非因为它会改变本实验的 AUC。

    下游 `evaluate` 按 logit 处理（BCEWithLogitsLoss / AUC），故末尾转回 logit；
    转换前 clamp，避免概率贴近 0/1 时 logit 爆炸。

    Args:
        prior: {(sex, race, age): p}，来自 `empirical_prior`。

    Returns:
        可传给 `evaluate` 的 forward_fn。
    """
    combos = list(prior)

    def fwd(model, images: torch.Tensor, attrs: dict) -> torch.Tensor:
        b, dev = images.shape[0], images.device
        # ⚠️ 累加器必须**沿用模型输出的形状**（本项目 num_classes=1 ⇒ [B,1]）。
        # 若初始化成 [B] 再与 [B,1] 相加会**广播成 [B,B]**，下游 BCEWithLogitsLoss 才报错。
        prob: torch.Tensor | None = None
        for combo in combos:
            w = prior[combo]
            if w == 0.0:                       # 空 cell 跳过，省一次前向
                continue
            fake_attrs = {k: torch.full((b,), v, dtype=torch.long, device=dev)
                          for k, v in zip(ATTR_KEYS, combo)}
            p = torch.sigmoid(forward_sex_race_age(model, images, fake_attrs))
            prob = w * p if prob is None else prob + w * p
        assert prob is not None, "先验全为 0，无有效属性组合"
        assert prob.shape[0] == b, f"边缘化输出形状异常：{tuple(prob.shape)}（batch={b}）"
        prob = prob.clamp(1e-6, 1.0 - 1e-6)
        return torch.log(prob / (1.0 - prob))

    return fwd


def make_knockout_forward(fake: dict[str, np.ndarray]):
    """
    包装 forward：**只替换喂给模型的属性**，不触碰 evaluate 收集的分组键。

    loader `shuffle=False` ⇒ 按游标逐 batch 切出对应的假属性片段。

    Args:
        fake: 全局干预后的属性数组（与 CSV 行序对齐）。

    Returns:
        可传给 `evaluate` 的 forward_fn。
    """
    cursor = {"i": 0}

    def fwd(model, images: torch.Tensor, attrs: dict) -> torch.Tensor:
        b = images.shape[0]
        i = cursor["i"]
        sl = slice(i, i + b)
        cursor["i"] = i + b
        dev = images.device
        fake_attrs = {k: torch.as_tensor(fake[k][sl], dtype=torch.long, device=dev)
                      for k in ATTR_KEYS}
        return forward_sex_race_age(model, images, fake_attrs)

    return fwd


def run(direction: str, mode: str, trials: tuple[int, ...], num_workers: int) -> None:
    """对每个 (fold, trial) 用干预后的属性在完整 target 上推断并落盘。"""
    source, target = ("mimic", "chexpert") if direction == "m2c" else ("chexpert", "mimic")
    cfg = load_selected_config(source)
    tag = cfg[METHOD]
    out_pred = (OUTPUTS_ROOT / "ood_cxr" / f"{source}2{target}" /
                f"cv5_knockout_{mode}" / "predictions")
    print(f"\n{'=' * 78}\n属性 knockout: {source} → {target}  mode={mode}  method={METHOD}"
          f"\n输出: {out_pred}\n{'=' * 78}")

    loader, df = build_full_target_loader(target=target, num_workers=num_workers)
    patient_id = df["patient_id"].values
    true_attrs = {"sex": df["sex"].values.astype(int),
                  "race": df["race"].values.astype(int),
                  # age 在线分箱，与 MIMICCXRDataset / subgroup_masks 的 60 岁阈值一致
                  "age": (df["age"].values >= 60).astype(int)}
    prior = None
    if mode == "constant":
        print(f"majority cell = {majority_cell(true_attrs)}")
    elif mode == "marginal":
        prior = empirical_prior(true_attrs)
        print("target 经验先验 p(sex,race,age)：")
        for k, v in sorted(prior.items(), key=lambda kv: -kv[1]):
            print(f"    {k} → {v:.4f}")

    for fold, trial, seed in fold_trial_seeds(trials):
        npz_path = out_pred / f"{METHOD}_{tag}_seed{seed}_overall.npz"
        if npz_path.exists():
            print(f"  fold{fold} trial{trial} seed{seed}: 已存在，跳过")
            continue
        ckpt = checkpoint_path(source, METHOD, tag, seed)
        if not ckpt.exists():
            raise FileNotFoundError(f"缺少 checkpoint：{ckpt}")
        if mode == "marginal":
            fwd = make_marginalized_forward(prior)
        else:
            fwd = make_knockout_forward(build_intervened_attrs(true_attrs, mode, seed=1000 + seed))
        result = evaluate_ood(METHOD, ckpt, source=source, target=target,
                              test_loader=loader, report=False, forward_override=fwd)
        er = result.eval_result
        assert len(er.labels) == len(patient_id), "预测数与 CSV 行数不一致"
        # 断言：收集到的分组键仍是**真实**属性（干预只应作用于模型输入）
        assert np.array_equal(np.asarray(er.attrs["sex"]).astype(int), true_attrs["sex"]), \
            "分组键被污染：attrs 不再是真实属性，knockout 包装位置有误"
        save_predictions(npz_path, er.labels, er.logits, er.attrs,
                         extra={"patient_id": patient_id})
        wc = f"{er.wc_auc:.4f}" if er.wc_auc is not None else "N/A"
        print(f"  fold{fold} trial{trial} seed{seed}: Overall={er.overall_auc:.4f} "
              f"worst={wc} → {npz_path.name}")


def main() -> None:
    p = argparse.ArgumentParser(description="OOD 属性 knockout（机制 A vs B）")
    p.add_argument("--direction", choices=("m2c", "c2m"), default="m2c")
    p.add_argument("--mode", choices=("permute", "constant", "marginal"), required=True)
    p.add_argument("--trials", default="0,1,2")
    p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用：拒绝静默退化到 CPU（坏 GPU 节点请 --exclude 换节点）。")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    run(args.direction, args.mode, tuple(int(t) for t in args.trials.split(",")),
        args.num_workers)
    print("\nknockout 推理完成。分析见 scripts/ood_knockout_analysis.py")


if __name__ == "__main__":
    main()
