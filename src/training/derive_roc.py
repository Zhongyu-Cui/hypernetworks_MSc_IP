"""
ROC (Reject-Option Classification) 派生后处理（协议 C*.8，数据集无关）
====================================================================
从已训练 ERM 的落盘预测（test overall + val_overall）后处理派生 ROC 基线：对每个确认 seed，
用 val 预测自动选二分敏感轴、定 deprived 子群、选临界区半宽 θ（val 上恒不劣 ERM），再把 θ 应用到
test 预测得调整后分数，落盘为 method="roc" 的预测（供 A2 显著性 / D 汇总）。

ROC 复用 ERM 预测、零训练（协议 §3「SWAD/ROC 搭 ERM 的车」），故为 CPU 轻量任务，在实验室机器
直接运行，不占集群。前置：ERM 5-seed 确认训练已产出
`outputs/<dataset>/predictions/erm_<config_tag>_seed<seed>_{overall,val_overall}.npz`。

用法示例（HAM，Pareto 选定 config_tag=lr1e-04_wd1e-04）：
  python -m src.training.derive_roc --dataset ham10000 --config-tag lr1e-04_wd1e-04
"""

import argparse
from pathlib import Path

import numpy as np

from src.training.harness.predictions import (
    default_prediction_path, load_predictions, save_predictions, val_prediction_path,
)
from src.training.harness.roc import apply_roc_auto, roc_fairness_report

DEFAULT_SEEDS = (42, 43, 44, 45, 46)   # SEEDS_CONFIRM
DEFAULT_OUTPUTS_ROOT = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs")


def parse_args() -> argparse.Namespace:
    """解析命令行参数（--dataset / --config-tag 必填；--method / --seeds / --output-dir 可选）。"""
    parser = argparse.ArgumentParser(description="ROC 派生后处理（协议 C*.8，数据集无关）")
    parser.add_argument("--dataset", type=str, required=True,
                        help="数据集名（ham10000/mimic/chexpert/fitzpatrick/papila），用于 ROC 轴集合与预测寻址。")
    parser.add_argument("--config-tag", type=str, required=True,
                        help="源 ERM 的 Pareto 选定超参配置标识，如 lr1e-04_wd1e-04（与预测文件名一致）。")
    parser.add_argument("--method", type=str, default="erm",
                        help="ROC 后处理的源方法名（默认 erm；ROC 恒复用 ERM 预测）。")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                        help="确认阶段 seed 列表（默认 42 43 44 45 46）。")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="数据集 outputs 目录（默认 /vol/.../outputs/<dataset>）；预测在其 predictions/ 子目录。")
    return parser.parse_args()


def _resolve_output_dir(dataset: str, output_dir: str | None) -> Path:
    """解析数据集 outputs 目录（缺省用 DEFAULT_OUTPUTS_ROOT / <dataset>）。"""
    return Path(output_dir) if output_dir is not None else DEFAULT_OUTPUTS_ROOT / dataset


def derive_roc_for_seed(
    dataset: str, method: str, config_tag: str, seed: int, output_dir: Path,
) -> None:
    """
    对单个 seed 派生 ROC：读 test overall + val_overall 预测 → apply_roc_auto → 打印 before/after
    → 落盘 method="roc" 的调整后 test 预测。

    Args:
        dataset   : 数据集名（决定 ROC 可用二分轴集合）。
        method    : 源方法（erm）。
        config_tag: Pareto 选定超参配置标识。
        seed      : 确认阶段 seed。
        output_dir: 数据集 outputs 目录（predictions 在其子目录）。
    """
    test_path = default_prediction_path(output_dir, method, config_tag, seed, "overall")
    val_path = val_prediction_path(output_dir, method, config_tag, seed)
    if not test_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"缺少预测文件（seed={seed}）：\n  test={test_path}\n  val ={val_path}\n"
            f"请先完成 ERM 5-seed 确认训练（C*.6）。")

    y_test_true, y_test_logits, test_attrs = load_predictions(test_path)
    y_val_true, y_val_logits, val_attrs = load_predictions(val_path)

    result = apply_roc_auto(
        dataset,
        y_val_true=y_val_true, y_val_logits=y_val_logits, val_attrs=val_attrs,
        y_test_true=y_test_true, y_test_logits=y_test_logits, test_attrs=test_attrs,
    )
    print(f"\n===== seed {seed}  (源 {method}_{config_tag}) =====")
    roc_fairness_report(result)

    # 落盘 method="roc" 的 test 预测（调整后分数），供 A2 显著性 / D 汇总；selection 沿用 "overall"
    roc_pred_path = default_prediction_path(output_dir, "roc", config_tag, seed, "overall")
    save_predictions(roc_pred_path, y_test_true, result.adjusted_test_scores, test_attrs)
    print(f"已落盘 ROC 预测: {roc_pred_path.name}")


def main() -> None:
    args = parse_args()
    output_dir = _resolve_output_dir(args.dataset, args.output_dir)
    print(f"ROC 派生（协议 C*.8）  dataset={args.dataset}  method={args.method}  "
          f"config_tag={args.config_tag}  seeds={args.seeds}")
    print(f"predictions 目录: {output_dir / 'predictions'}")

    for seed in args.seeds:
        derive_roc_for_seed(args.dataset, args.method, args.config_tag, seed, output_dir)

    print(f"\n全部 {len(args.seeds)} 个 seed 的 ROC 派生完成。")


if __name__ == "__main__":
    main()
