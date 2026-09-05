"""
跨数据集 OOD 评估（CXR：MIMIC ↔ CheXpert）（比较协议 A5.1 / A5.2）
================================================================
比较协议 §2「OOD 仅 CXR 一对，双向做」：加载在**源**数据集 A 训练好的模型，在**目标**数据集 B 的
测试集上评估（No Finding 二分类），报告 Overall / worst-group AUC / AUC gap 与分组公平性。MIMIC 与
CheXpert 来自同一 CXR7-1M master、字段完全同构（sex/race/age、16-bit PNG），故 Dataset / 公平性 /
模型架构全量复用 MIMIC 实现——OOD 唯一变化是「权重来自 A、测试集来自 B」。

  A5.1 `evaluate_ood`      —— 单向：加载 A 模型 → B 测试集评估（本文件核心）。
  A5.2 `evaluate_ood_pair` —— 双向参数化：给定两端 checkpoint，跑 A→B 与 B→A。

**方法注册表**：ERM/SWAD 用 image-only 架构（forward 只吃图像；SWAD 是 ERM 的权重平均，架构相同）；
HyperHead/HyperFusion/HyperAdapt 用对应 HN 架构（forward 吃 sex/race/age）。HN 构造用 `pretrained=False`
（权重全由 checkpoint 提供，避免评估时无谓下载 ImageNet 权重）。**ROC 不在此**：它是分数后处理、无独立
OOD 模型（OOD-ROC = 对 OOD ERM 分数再跑 roc.apply_roc，属 D 阶段可选，不在 A5）。

运行（单向）：
    python -m src.training.eval_ood_cxr --method erm --source mimic --target chexpert \
        --checkpoint outputs/mimic_cxr/erm_lr1e-04_wd1e-04_seed42_best_overall.pth
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.mimic_cxr_dataset import MIMICCXRDataset
from src.models.resnet18_pretrained import ResNet18Pretrained
from src.models.resnet18_hyperhead import ResNet18HyperHead
from src.models.resnet18_hyperfusion import ResNet18HyperFusion
from src.models.resnet18_hyperadapt import ResNet18HyperAdapt, ResNet18HyperAdaptSoftPatient
from src.training.harness.train_loop import (
    DEVICE, EvalResult, evaluate, forward_image_only, forward_sex_race_age, forward_soft_patient,
    unpack_sex_race_age, unpack_sex_race_age_soft,
)
from src.utils.mimic_fairness import print_mimic_fairness_report

REPO_ROOT = Path(__file__).resolve().parents[2]

# 数据集 → 训练配置（提供 split_dir / image_size / age_threshold / eval transform）
DATASET_CONFIG: dict[str, Path] = {
    "mimic": REPO_ROOT / "configs" / "mimic_cxr_baseline.yaml",
    "chexpert": REPO_ROOT / "configs" / "chexpert_baseline.yaml",
}

# 方法 → (模型工厂, forward 回调)。CXR 三属性通路（sex/race/age）。
METHOD_REGISTRY: dict[str, tuple[Callable[[], nn.Module], Callable]] = {
    "erm": (lambda: ResNet18Pretrained(num_classes=1), forward_image_only),
    "swad": (lambda: ResNet18Pretrained(num_classes=1), forward_image_only),  # SWAD=ERM 权重平均
    # GroupDRO：训练时用子群标签重加权损失，**架构与 ERM 完全相同**（属性只进损失、不进 forward）
    # ⇒ 推理侧是纯 image-only，与 ERM/SWAD 同一注册形态。
    "groupdro": (lambda: ResNet18Pretrained(num_classes=1), forward_image_only),
    "hyperhead": (lambda: ResNet18HyperHead(num_classes=1), forward_sex_race_age),
    "hyperfusion": (lambda: ResNet18HyperFusion(num_classes=1, pretrained=False), forward_sex_race_age),
    "hyperadapt": (lambda: ResNet18HyperAdapt(num_classes=1, pretrained=False), forward_sex_race_age),
    # 实验 G 的融合臂：架构与 hyperadapt 完全相同，只是权重来自 SWAD loss-valley 平均
    # （checkpoint 后缀 _averaged，见 run_ood_cxr_full_target.checkpoint_path）。
    "hyperadapt_swad": (lambda: ResNet18HyperAdapt(num_classes=1, pretrained=False), forward_sex_race_age),
    # 实验 P 的 pred-attr 臂（docs/predicted_attribute_hyperadapt_plan.md）：条件输入是**概率**而非
    # 索引，故模型换成期望嵌入版、forward 换成 forward_soft_patient；四个 attr_mode 共用同一架构
    # （差别只在条件输入如何物化，由 loader 侧决定），故注册为四个同构条目。
    # ⚠️ 这些方法要求 test_loader 产出 SoftAttrDataset 包装的 8 元组，并给 evaluate_ood 传
    # `unpack_override=unpack_sex_race_age_soft`（见 run_ood_cxr_pred.py）。
    **{m: (lambda: ResNet18HyperAdaptSoftPatient(num_classes=1, pretrained=False), forward_soft_patient)
       for m in ("hyperadapt_pred", "hyperadapt_predhard",
                 "hyperadapt_predperm", "hyperadapt_predconst")},
}


def load_config(dataset: str) -> dict:
    """读取目标数据集的训练配置（用于构建同款 eval transform 与 test loader）。"""
    if dataset not in DATASET_CONFIG:
        raise ValueError(f"未知数据集 {dataset!r}，OOD 仅支持 {sorted(DATASET_CONFIG)}。")
    with open(DATASET_CONFIG[dataset], "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_eval_transform(cfg: dict) -> transforms.Compose:
    """构建评估 transform（与训练一致：Grayscale(3) → Resize → ToTensor → Normalize，不增强）。"""
    t_cfg = cfg["transforms"]
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(t_cfg["eval"]["resize"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=t_cfg["normalize"]["mean"], std=t_cfg["normalize"]["std"]),
    ])


def build_target_test_loader(
    dataset: str, batch_size: int = 128, num_workers: int = 4,
    *, cv: int = 0, fold: int | None = None,
) -> DataLoader:
    """
    构建目标数据集 B 的 test DataLoader（复用 MIMICCXRDataset；两 CXR 数据集 schema 相同）。

    Args:
        dataset    : 目标数据集名（"mimic" / "chexpert"）。
        batch_size : 推理 batch。
        num_workers: DataLoader worker 数。
        cv / fold  : CV-OOF OOD 用——给定 cv>=2 与 fold∈[0,cv) 时读 `cv{cv}/fold{fold}/test.csv`
                     （目标该折 test），供 source 逐折 checkpoint → target 逐折 test 池化成 target OOF。
                     缺省（cv=0）读单-split `test.csv`（与旧单-split OOD 兼容）。

    Returns:
        目标 test 集的 DataLoader（不打乱）。
    """
    cfg = load_config(dataset)
    split_dir = REPO_ROOT / cfg["data"]["split_dir"]
    image_size = cfg["data"]["image_size"]
    age_threshold = cfg["attributes"]["age"]["age_threshold"]
    if cv and cv >= 2:
        if fold is None or not (0 <= fold < cv):
            raise ValueError(f"cv={cv} 需配合合法 fold ∈ [0,{cv})")
        test_csv = split_dir / f"cv{cv}" / f"fold{fold}" / "test.csv"
    else:
        test_csv = split_dir / "test.csv"
    test_set = MIMICCXRDataset(
        test_csv, transform=build_eval_transform(cfg),
        image_size=image_size, age_threshold=age_threshold,
    )
    return DataLoader(test_set, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


@dataclass
class OODResult:
    """OOD 单向评估结果（源 A 模型 → 目标 B 测试集）。"""

    method: str
    source: str
    target: str
    checkpoint: str
    overall_auc: float
    worst_case_auc: float | None
    eval_result: EvalResult   # 完整 EvalResult（含子群向量 / logits / labels / attrs，供下游）


def evaluate_ood(
    method: str,
    checkpoint_path: str | Path,
    source: str,
    target: str,
    *,
    test_loader: DataLoader | None = None,
    report: bool = True,
    forward_override: Callable | None = None,
    unpack_override: Callable | None = None,
) -> OODResult:
    """
    A5.1：加载源 A 训练的模型，在目标 B 的测试集上评估并报告公平性。

    评估设备固定为 harness.DEVICE（评估内部把数据搬到 DEVICE，模型须一致）。

    Args:
        method         : {"erm","swad","hyperhead","hyperfusion","hyperadapt"} 之一（见注册表）。
        checkpoint_path: 源 A 训练好的模型权重（state_dict .pth）。
        source         : 源数据集名（仅用于记录/打印）。
        target         : 目标数据集名（决定 test loader）。
        test_loader    : 可选，直接注入目标 test loader（缺省则按 target 构建；测试/复用时用）。
        report         : 是否打印分组公平性报告。
        forward_override: 可选，替换该方法默认的 forward 回调。用于**属性 knockout 反事实**
            （`run_ood_attr_knockout.py`）——只改「模型看到什么属性」，而 `evaluate` 收集的
            分组键仍来自 unpack_fn 的真实属性，故评估口径不受污染。缺省 None = 原行为。
        unpack_override : 可选，替换默认的 `unpack_sex_race_age`。实验 P 的 pred-attr 臂需要
            `unpack_sex_race_age_soft`（loader 多产出三份属性概率）。**分组键仍取真值 sex/race/age**，
            评估口径不变。缺省 None = 原行为。

    Returns:
        OODResult。

    Raises:
        ValueError: method 未注册。
    """
    if method not in METHOD_REGISTRY:
        raise ValueError(f"未知 method {method!r}，可用：{sorted(METHOD_REGISTRY)}。")
    build_model, forward_fn = METHOD_REGISTRY[method]
    if forward_override is not None:
        forward_fn = forward_override      # knockout：只改模型输入，不改分组键

    model = build_model().to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)

    if test_loader is None:
        test_loader = build_target_test_loader(target)

    criterion = nn.BCEWithLogitsLoss()
    # subgroup_auc / 公平性口径：两 CXR 数据集共用 mimic 向量（target 传 mimic/chexpert 皆可）
    unpack_fn = unpack_override if unpack_override is not None else unpack_sex_race_age
    er = evaluate(model, test_loader, criterion, target, unpack_fn, forward_fn)

    wc_str = f"{er.wc_auc:.4f}" if er.wc_auc is not None else "N/A"
    print(f"\n{'#' * 70}\n# OOD: {source} 模型 → {target} 测试集  (method={method})\n{'#' * 70}")
    print(f"checkpoint: {Path(checkpoint_path).name}")
    print(f"Test loss={er.loss:.4f}  accuracy={er.acc:.2f}%  Overall AUC={er.overall_auc:.4f}  "
          f"worst-group AUC={wc_str}")
    if report:
        print_mimic_fairness_report(
            y_true=er.labels, y_score=er.logits,
            sex=er.attrs["sex"], race=er.attrs["race"], age=er.attrs["age"], threshold=0.0,
        )
    return OODResult(
        method=method, source=source, target=target, checkpoint=str(checkpoint_path),
        overall_auc=er.overall_auc, worst_case_auc=er.wc_auc, eval_result=er,
    )


def evaluate_ood_pair(
    method: str,
    checkpoint_mimic: str | Path,
    checkpoint_chexpert: str | Path,
    **kwargs,
) -> tuple[OODResult, OODResult]:
    """
    A5.2：双向 OOD（MIMIC↔CheXpert）。用 MIMIC 训练的权重评估 CheXpert 测试集，反之亦然。

    Args:
        method             : 方法名。
        checkpoint_mimic   : 在 MIMIC 上训练的权重（用于 MIMIC→CheXpert）。
        checkpoint_chexpert: 在 CheXpert 上训练的权重（用于 CheXpert→MIMIC）。
        **kwargs           : 透传给 evaluate_ood（如 test_loader 不适用双向，勿传）。

    Returns:
        (mimic→chexpert 的 OODResult, chexpert→mimic 的 OODResult)。
    """
    m2c = evaluate_ood(method, checkpoint_mimic, source="mimic", target="chexpert", **kwargs)
    c2m = evaluate_ood(method, checkpoint_chexpert, source="chexpert", target="mimic", **kwargs)
    return m2c, c2m


def _parse_args() -> argparse.Namespace:
    """命令行参数（单向 OOD 评估）。"""
    p = argparse.ArgumentParser(description="CXR 跨数据集 OOD 评估（MIMIC↔CheXpert）")
    p.add_argument("--method", required=True, choices=sorted(METHOD_REGISTRY))
    p.add_argument("--checkpoint", required=True, help="源数据集训练好的权重 .pth")
    p.add_argument("--source", required=True, choices=sorted(DATASET_CONFIG))
    p.add_argument("--target", required=True, choices=sorted(DATASET_CONFIG))
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    """命令行入口：单向 OOD 评估。"""
    args = _parse_args()
    if args.source == args.target:
        raise SystemExit("OOD 的 --source 与 --target 必须不同（MIMIC↔CheXpert）。")
    loader = build_target_test_loader(args.target, args.batch_size, args.num_workers)
    evaluate_ood(args.method, args.checkpoint, args.source, args.target, test_loader=loader)


if __name__ == "__main__":
    main()
