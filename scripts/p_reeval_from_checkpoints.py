"""
实验 P·从已训 checkpoint 重新评估并落盘预测（不重训）
====================================================
用途：训练本身成功、但 `run_training` 收尾阶段的落盘失败时（如公平性报告回调的参数名写错），
checkpoint 已保存而 test/val 预测未落盘。本脚本按既有 checkpoint 重跑评估与落盘，
**逐字复用 harness 的 evaluate_checkpoint / _save_eval_predictions**，
保证产物与正常训练流程逐位同构（同命名、同 attrs/extra 分流）。

只做推理，不训练、不改动任何 checkpoint。省下的是重训成本（MIMIC 20 run ≈ 13 GPU-hours）。

用法（需 GPU；MIMIC 每 (臂,折) 约数分钟）：
    python scripts/p_reeval_from_checkpoints.py --dataset mimic --fold 0
    python scripts/p_reeval_from_checkpoints.py --dataset mimic --fold 0 --modes soft hard
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch.nn as nn

from src.datasets.soft_attr_wrapper import ATTR_MODES
from src.training.harness.hparam_grid import get_hparam_config
from src.training.harness.predictions import default_prediction_path, val_prediction_path
from src.training.harness.run import SELECTIONS, checkpoint_path, seed_everything
from src.training.harness.train_loop import (
    DEVICE, _save_eval_predictions, evaluate_checkpoint,
)

# 各库的 pred-attr 训练模块（提供 spec / loader / 回调 / 模型工厂）
MODULES = {
    "mimic": ("src.training.train_cxr_hyperadapt_pred", "mimic"),
    "chexpert": ("src.training.train_cxr_hyperadapt_pred", "chexpert"),
    "ham10000": ("src.training.train_ham10000_hyperadapt_pred", None),
    "fitzpatrick": ("src.training.train_fitzpatrick_hyperadapt_pred", None),
}
FOLD_SEED_BASE = 42        # fold k ↔ seed 42+k（与既有 CV 布局一致）
# 各库 pred 臂使用的超参网格序号 = 该库 GT-HyperAdapt 的 Pareto 选定配置（与训练脚本的默认值一致）
CONFIG_INDEX = {"mimic": 3, "chexpert": 4, "ham10000": 0, "fitzpatrick": 5}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    ap = argparse.ArgumentParser(description="实验 P：从已训 checkpoint 重新评估并落盘预测")
    ap.add_argument("--dataset", required=True, choices=sorted(MODULES))
    ap.add_argument("--fold", type=int, required=True, help="折号 k ∈ [0,5)。")
    ap.add_argument("--cv", type=int, default=5, help="折数 K。")
    ap.add_argument("--modes", nargs="+", default=list(ATTR_MODES),
                    help=f"要处理的条件输入模式（缺省全部 {ATTR_MODES}）。")
    ap.add_argument("--batch_size", type=int, default=None, help="覆盖评估 batch（默认用 config）。")
    ap.add_argument("--cond", type=str, default="age",
                    help="仅 HAM 支持：条件通路（age / sex_age）。双属性对照臂用 sex_age，"
                         "会取 attr_pred_sexage/ 的 p̂ 并按 <method>_sexage 命名寻址 checkpoint。")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    module_name, dataset_arg = MODULES[args.dataset]
    module = __import__(module_name, fromlist=["*"])
    criterion = nn.BCEWithLogitsLoss()
    seed = FOLD_SEED_BASE + args.fold

    # 通用 CXR 脚本把库差异收在 spec 里；单库脚本没有这层
    spec = module.DATASET_SPECS[dataset_arg] if dataset_arg is not None else None
    cfg = module.load_config(spec["config"] if spec is not None else module.CONFIG_PATH)
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size

    tag = get_hparam_config(CONFIG_INDEX[args.dataset]).tag
    dataset_key = spec["dataset"] if spec is not None else module.DATASET

    # 只有 HAM 的 pred 脚本支持多条件通路（以模块级 COND_AXES 标识）；其余库忽略 --cond
    cond_kwargs = {"cond": args.cond} if hasattr(module, "COND_AXES") else {}
    if cond_kwargs and args.cond not in module.COND_AXES:
        raise ValueError(f"--cond={args.cond} 不被 {module_name} 支持（合法：{tuple(module.COND_AXES)}）")
    paths = (module.resolve_paths(spec, cfg, args.cv, args.fold) if spec is not None
             else module.resolve_paths(cfg, args.cv, args.fold, **cond_kwargs))
    split_dir, output_dir, attr_pred_dir = paths
    train_tf, eval_tf = module.build_transforms(cfg)

    print(f"[实验 P·重评估] dataset={args.dataset} fold={args.fold} seed={seed} config={tag}")
    for mode in args.modes:
        method = (module.method_of(mode, args.cond) if cond_kwargs
                  else module.METHOD_BY_MODE[mode])
        seed_everything(seed)
        loaders = (module.get_dataloaders(spec, cfg, split_dir, attr_pred_dir,
                                          train_tf, eval_tf, mode, seed) if spec is not None
                   else module.get_dataloaders(cfg, split_dir, attr_pred_dir,
                                               train_tf, eval_tf, mode, seed, **cond_kwargs))
        _, val_loader, test_loader = loaders
        unpack_fn = (module.unpack_fn_of_cond(args.cond) if cond_kwargs else module.UNPACK_FN)
        forward_fn = (module.forward_fn_of_cond(args.cond) if cond_kwargs else module.FORWARD_FN)
        model = (module.make_model(cfg, spec) if spec is not None
                 else module.MODEL_FACTORY(cfg, **cond_kwargs)).to(DEVICE)

        for selection in SELECTIONS:
            ckpt = checkpoint_path(output_dir, method, tag, seed, selection)
            if not ckpt.exists():
                print(f"  [跳过] 缺 checkpoint: {ckpt.name}")
                continue
            er = evaluate_checkpoint(model, ckpt, test_loader, criterion, dataset_key,
                                     unpack_fn, forward_fn,
                                     module._fairness_report, label=f"{method}/{selection}")
            out = default_prediction_path(output_dir, method, tag, seed, selection)
            _save_eval_predictions(out, er)
            print(f"  已落盘 test 预测: {out.name}")

        # val 预测（供 ROC 后处理）：始终取 overall-selection checkpoint，与 run_training 一致
        ckpt_overall = checkpoint_path(output_dir, method, tag, seed, "overall")
        if ckpt_overall.exists():
            er_val = evaluate_checkpoint(model, ckpt_overall, val_loader, criterion, dataset_key,
                                         unpack_fn, forward_fn,
                                         module._fairness_report, label=f"[val] {method}/overall")
            out_val = val_prediction_path(output_dir, method, tag, seed)
            _save_eval_predictions(out_val, er_val)
            print(f"  已落盘 val 预测: {out_val.name}")
        del model


if __name__ == "__main__":
    main()
