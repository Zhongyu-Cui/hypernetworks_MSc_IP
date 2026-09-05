"""
补落盘 SWAD 系臂的 **val 逐样本预测**（纯推理，不训练、不改动任何 checkpoint）
==============================================================================
`train_loop._finalize_swad` 在验证集上跑了 `evaluate`，但只把**子群 AUC 向量**写进 val 日志，
没有调 `_save_eval_predictions` 落逐样本 npz；test 那一侧则落了。结果是 ERM / GroupDRO /
三个 HN 都有 `*_val_overall.npz`，唯独 `swad` 与 `hyperadapt_swad` 四个库全缺。

阈值类指标（accuracy 等）要求**阈值在 val 上选、绝不碰评估数据**，缺 val 预测的臂就无法进表，
而 SWAD 恰是主报告里唯一给出真实收益的基线。本脚本从已存在的 `<run_id>_averaged.pth`
重跑一次验证集推理把 npz 补上，逐字复用 harness 的 `evaluate_checkpoint` /
`_save_eval_predictions`，保证产物与正常训练流程同构（同命名、同 attrs/extra 分流）。

**自检**：同一 checkpoint 同时在 test 上重跑一遍，与既有 test 预测比对，用来挡住
「模型构造 / transform / loader 与当年训练不一致」这类静默漂移。既有 npz **只读不覆盖**。
判据落在决策相关量上（y_true 完全一致 + Overall AUC 差 ≤1e-4 + prob=0.5 判定翻转率 ≤0.5%），
**不**要求 logits 逐位相等——同一权重换一块 GPU 重跑，卷积算法与 TF32 会让 logits 差到 1e-2
量级，而装错模型会差 O(1)，两者相隔几个数量级。详见 `verify_against_stored`。

用法（需 GPU；MIMIC/CheXpert 每 (臂,折) 数分钟）：
    python scripts/dump_swad_val_predictions.py --dataset ham10000 --folds 0 1 2 3 4
    python scripts/dump_swad_val_predictions.py --dataset mimic --arms swad --folds 0
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable

import numpy as np
import torch.multiprocessing
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from src.models.resnet18_hyperadapt import (
    ResNet18HyperAdapt, ResNet18HyperAdaptAge, ResNet18HyperAdaptSkin,
)
from src.models.resnet18_pretrained import ResNet18Pretrained
from src.paths import OUTPUTS_DIR
from src.training.harness.predictions import (
    default_prediction_path, load_predictions, save_predictions, val_prediction_path,
)
from src.training.harness.run import run_id, seed_everything
from src.training.harness.train_loop import (
    DEVICE, _save_eval_predictions, evaluate_checkpoint,
    forward_age, forward_image_only, forward_sex_race_age, forward_skin,
    unpack_sex_age, unpack_sex_race_age, unpack_skin,
)

# DataLoader 的多进程默认用 file_descriptor 共享张量，MIMIC/CheXpert 这种数万张的评估集
# 会把句柄耗光（"Too many open files"，实测 MIMIC 五折全崩在 test 自检那一趟）。
# file_system 策略改用共享内存文件名传递，不吃 fd 配额。
torch.multiprocessing.set_sharing_strategy("file_system")

FOLD_SEED_BASE = 42                       # fold k ↔ seed 42+k（CV 搜索约定）
# 各库 outputs 子目录（selected_configs.json 的所在处，用于取 config tag）
OUTPUT_SUBDIR = {"ham10000": "ham10000", "fitzpatrick": "fitzpatrick",
                 "mimic": "mimic_cxr", "chexpert": "chexpert_cxr"}


@dataclass(frozen=True)
class ArmSpec:
    """一个 (数据集, SWAD 臂) 的推理规格：从哪个训练模块取 loader/回调、怎么造模型。"""

    module: str                                        # 提供 loader / transform / 公平性回调的训练模块
    unpack_fn: Callable                                # batch → (images, labels, attrs)
    forward_fn: Callable                               # (model, images, attrs) → logits
    build_model: Callable[[dict, ModuleType], nn.Module]


# `swad` 派生自 ERM，`hyperadapt_swad` 派生自 HyperAdapt——模型结构、loader 与回调必须与
# 派生来源的训练脚本 main() 完全一致，否则 state_dict 装不上或前向语义不同（test 自检会抓到）。
ARMS: dict[tuple[str, str], ArmSpec] = {
    ("ham10000", "swad"): ArmSpec(
        "src.training.train_ham10000_resnet18", unpack_sex_age, forward_image_only,
        lambda cfg, m: ResNet18Pretrained(num_classes=1)),
    ("fitzpatrick", "swad"): ArmSpec(
        "src.training.train_fitzpatrick_resnet18", unpack_skin, forward_image_only,
        lambda cfg, m: ResNet18Pretrained(num_classes=1)),
    ("mimic", "swad"): ArmSpec(
        "src.training.train_mimic_resnet18", unpack_sex_race_age, forward_image_only,
        lambda cfg, m: ResNet18Pretrained(num_classes=1)),
    ("chexpert", "swad"): ArmSpec(
        "src.training.train_chexpert_resnet18", unpack_sex_race_age, forward_image_only,
        lambda cfg, m: ResNet18Pretrained(num_classes=1)),
    # HAM 的融合臂是 **age 条件**版（`hyperadapt_swad`，非 `_sexage`），与选定 config 一致
    ("ham10000", "hyperadapt_swad"): ArmSpec(
        "src.training.train_ham10000_hyperadapt", unpack_sex_age, forward_age,
        lambda cfg, m: ResNet18HyperAdaptAge(
            num_classes=1, num_age=m.NUM_AGE, pretrained=cfg["model"]["pretrained"])),
    ("fitzpatrick", "hyperadapt_swad"): ArmSpec(
        "src.training.train_fitzpatrick_hyperadapt", unpack_skin, forward_skin,
        lambda cfg, m: ResNet18HyperAdaptSkin(
            num_classes=1, num_skin=m.NUM_SKIN, pretrained=cfg["model"]["pretrained"])),
    ("mimic", "hyperadapt_swad"): ArmSpec(
        "src.training.train_mimic_hyperadapt", unpack_sex_race_age, forward_sex_race_age,
        lambda cfg, m: ResNet18HyperAdapt(num_classes=1, pretrained=cfg["model"]["pretrained"])),
    ("chexpert", "hyperadapt_swad"): ArmSpec(
        "src.training.train_chexpert_hyperadapt", unpack_sex_race_age, forward_sex_race_age,
        lambda cfg, m: ResNet18HyperAdapt(num_classes=1, pretrained=cfg["model"]["pretrained"])),
}
DEFAULT_ARMS = ("swad", "hyperadapt_swad")


def selected_tag(dataset: str, arm: str, cv: int) -> str:
    """
    从该库的 `selected_configs.json` 取该臂的 Pareto 选定配置 tag。

    Args:
        dataset: {"ham10000","fitzpatrick","mimic","chexpert"} 之一。
        arm    : "swad" 或 "hyperadapt_swad"。
        cv     : 折数 K（定位 `cv{K}/selected_configs.json`）。

    Returns:
        形如 "lr3e-04_wd1e-03" 的配置 tag。

    Raises:
        KeyError: 该 JSON 里没有这个臂。
    """
    path = OUTPUTS_DIR / OUTPUT_SUBDIR[dataset] / f"cv{cv}" / "selected_configs.json"
    config = json.loads(path.read_text())["config"]
    if arm not in config:
        raise KeyError(f"{path} 无臂 {arm!r}（有：{sorted(config)}）")
    return str(config[arm])


def verify_against_stored(
    output_dir: Path, arm: str, tag: str, seed: int,
    y_new: np.ndarray, s_new: np.ndarray, auc_atol: float, flip_rtol: float,
    recheck_path: Path | None,
) -> bool:
    """
    把本次 test 推理与既有 test 预测比对（既有 npz **只读不覆盖**）。

    判据落在**决策相关量**上，而不是 logits 的逐位相等：同一权重在不同 GPU 上重跑，
    卷积会走不同的 cuDNN 算法、Ampere 及以后默认对 conv 启用 TF32，逐层累积后 logits
    可以差到 1e-2 量级（相对误差 ~1e-3）。这不说明装错了模型——**装错模型会差 O(1)**。
    真正要保证的是「本次重跑与当年落盘描述的是同一个分类器」，故判据取
      ① y_true 完全一致（loader 顺序 / split 没变）；
      ② Overall AUC 差 ≤ auc_atol（排序结构一致）；
      ③ prob=0.5 处的判定翻转率 ≤ flip_rtol（阈值行为一致）。

    Args:
        output_dir  : 该库该 CV regime 的 outputs 目录。
        arm/tag/seed: 定位既有预测。
        y_new/s_new : 本次推理的标签与 logits。
        auc_atol    : Overall AUC 允许的绝对偏差。
        flip_rtol   : prob=0.5 判定翻转率上限（0.005 = 0.5%）。
        recheck_path: 非 None 时把本次 test 预测另存到此路径（**不覆盖**既有），供离线复核。

    Returns:
        True 表示一致；False 表示不一致（调用方据此判失败）。
    """
    stored = default_prediction_path(output_dir, arm, tag, seed, "overall")
    if not stored.exists():
        print(f"  [自检跳过] 无既有 test 预测: {stored.name}")
        return True
    y_old, s_old, _ = load_predictions(stored)
    y_old_i, y_new_i = np.asarray(y_old).astype(int), np.asarray(y_new).astype(int)
    if len(y_old) != len(y_new):
        print(f"  ❌ 自检失败: 样本数 {len(y_old)} vs {len(y_new)}")
        return False
    if not np.array_equal(y_old_i, y_new_i):
        print("  ❌ 自检失败: y_true 不一致（loader 顺序或 split 变了）")
        return False
    s_old_f, s_new_f = np.asarray(s_old, float), np.asarray(s_new, float)
    dmax = float(np.max(np.abs(s_old_f - s_new_f)))
    auc_old, auc_new = roc_auc_score(y_old_i, s_old_f), roc_auc_score(y_old_i, s_new_f)
    d_auc = abs(float(auc_old) - float(auc_new))
    flip = float(np.mean((s_old_f >= 0.0) != (s_new_f >= 0.0)))
    ok = d_auc <= auc_atol and flip <= flip_rtol
    print(f"  {'✓' if ok else '❌'} 自检: AUC {auc_old:.6f} vs {auc_new:.6f} (Δ={d_auc:.2e}) | "
          f"判定翻转率={flip:.4%} | max|Δlogit|={dmax:.3e}（仅诊断，见 docstring）")
    if recheck_path is not None:
        save_predictions(recheck_path, y_new_i, s_new_f, {})
        print(f"  已另存本次 test 预测供离线复核: {recheck_path.name}")
    return ok


def dump_one(
    dataset: str, arm: str, fold: int, cv: int, batch_size: int | None,
    auc_atol: float, flip_rtol: float, verify: bool, overwrite: bool, keep_recheck: bool,
) -> bool:
    """
    补落盘单个 (数据集, 臂, 折) 的 val 预测，并按需做 test 自检。

    Args:
        dataset/arm/fold/cv: 定位一次 run。
        batch_size  : 覆盖评估 batch（None = 用 config 的值）。
        auc_atol    : test 自检的 Overall AUC 容差。
        flip_rtol   : test 自检的判定翻转率上限。
        verify      : 是否跑 test 自检（关掉可省一半推理时间）。
        overwrite   : val npz 已存在时是否重跑覆盖。
        keep_recheck: 是否把自检用的 test 预测另存为 `*_overall_recheck.npz`。

    Returns:
        True 表示成功（含「已存在而跳过」）；False 表示自检失败或缺 checkpoint。
    """
    spec = ARMS[(dataset, arm)]
    module = importlib.import_module(spec.module)
    seed = FOLD_SEED_BASE + fold
    tag = selected_tag(dataset, arm, cv)

    cfg = module.load_config(module.CONFIG_PATH)
    if batch_size is not None:
        cfg["training"]["batch_size"] = batch_size
    split_dir, output_dir = module.resolve_paths(cfg, cv, fold)
    # SWAD 权重不是 harness 的 `_best_<selection>.pth`，而是 _finalize_swad 直接存的平均权重。
    # 用 `_averaged.pth`（**不是** `_averaged_bnupd.pth`）——既有 test 预测由前者产出，
    # 后者是 BN 重估复核实验的副产品，混用会让自检必然失败。
    ckpt = output_dir / f"{run_id(arm, tag, seed)}_averaged.pth"
    if not ckpt.exists():
        print(f"  [跳过] 缺 checkpoint: {ckpt}")
        return False

    out_val = val_prediction_path(output_dir, arm, tag, seed)
    if out_val.exists() and not overwrite:
        print(f"  [跳过] val 预测已存在: {out_val.name}（--overwrite 可重跑）")
        return True

    print(f"\n{'=' * 78}\n[{dataset} · {arm} · fold{fold} · seed{seed} · {tag}]\n{'=' * 78}")
    seed_everything(seed)                     # 与训练脚本一致：构建 loader 前播种
    train_tf, eval_tf = module.build_transforms(cfg)
    _train_loader, val_loader, test_loader = module.get_dataloaders(
        cfg, split_dir, train_tf, eval_tf)
    model = spec.build_model(cfg, module).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()

    er_val = evaluate_checkpoint(model, ckpt, val_loader, criterion, module.DATASET,
                                 spec.unpack_fn, spec.forward_fn, module._fairness_report,
                                 label=f"[val] {arm}/averaged")
    _save_eval_predictions(out_val, er_val)
    print(f"  已落盘 val 预测: {out_val.name}  (n={len(er_val.labels)})")

    if not verify:
        return True
    er_test = evaluate_checkpoint(model, ckpt, test_loader, criterion, module.DATASET,
                                  spec.unpack_fn, spec.forward_fn, module._fairness_report,
                                  label=f"[test 自检] {arm}/averaged")
    recheck = (output_dir / "predictions" / f"{run_id(arm, tag, seed)}_overall_recheck.npz"
               if keep_recheck else None)
    return verify_against_stored(output_dir, arm, tag, seed, er_test.labels, er_test.logits,
                                 auc_atol, flip_rtol, recheck)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    ap = argparse.ArgumentParser(description="补落盘 SWAD 系臂的 val 预测（纯推理）")
    ap.add_argument("--dataset", required=True, choices=sorted(OUTPUT_SUBDIR))
    ap.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS), choices=list(DEFAULT_ARMS),
                    help=f"要补的臂（缺省 {DEFAULT_ARMS}）。")
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="折号列表。")
    ap.add_argument("--cv", type=int, default=5, help="折数 K。")
    ap.add_argument("--batch_size", type=int, default=None, help="覆盖评估 batch（默认用 config）。")
    ap.add_argument("--auc-atol", type=float, default=1e-4,
                    help="test 自检的 Overall AUC 容差（同权重跨 GPU 重跑的合理量级）。")
    ap.add_argument("--flip-rtol", type=float, default=5e-3,
                    help="test 自检的 prob=0.5 判定翻转率上限。")
    ap.add_argument("--keep-recheck", action="store_true",
                    help="把自检用的 test 预测另存为 *_overall_recheck.npz（既有 npz 仍不覆盖）。")
    ap.add_argument("--no-verify", action="store_true", help="跳过 test 自检（省一半推理时间）。")
    ap.add_argument("--overwrite", action="store_true", help="val npz 已存在时重跑覆盖。")
    return ap.parse_args()


def main() -> None:
    """按 (臂 × 折) 逐个补落盘，末尾汇总；任一自检失败则以非零码退出。"""
    args = parse_args()
    print(f"设备: {DEVICE}")
    failed: list[str] = []
    for arm in args.arms:
        for fold in args.folds:
            # 逐 (臂,折) 兜错：共享盘偶发 "broken data stream" 之类的瞬时故障只该让这一项失败，
            # 不该带走同一作业里其余的臂（实测 CheXpert fold1 就撞上过一次）
            try:
                ok = dump_one(args.dataset, arm, fold, args.cv, args.batch_size,
                              args.auc_atol, args.flip_rtol, not args.no_verify,
                              args.overwrite, args.keep_recheck)
            except Exception as exc:                       # noqa: BLE001（有意兜住全部运行期故障）
                print(f"  ❌ {arm}/fold{fold} 异常: {type(exc).__name__}: {exc}")
                ok = False
            if not ok:
                failed.append(f"{arm}/fold{fold}")
    print(f"\n{'=' * 78}")
    if failed:
        print(f"❌ 失败或跳过 {len(failed)} 项: {', '.join(failed)}")
        raise SystemExit(1)
    print(f"✓ {args.dataset}: {len(args.arms)} 臂 × {len(args.folds)} 折 全部完成")


if __name__ == "__main__":
    main()
