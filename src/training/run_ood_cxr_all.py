"""
C6 OOD 全量评估驱动（CXR：MIMIC ↔ CheXpert 双向，5 方法 × 5 seed）
====================================================================
比较协议工作清单 C6.1（MIMIC→CheXpert）/ C6.2（CheXpert→MIMIC）。对每个方向、每个方法
（ERM/SWAD + 3 HN；**ROC 不在此**，属分数后处理）、每个确认 seed(42–46)，加载源数据集训练好的
确认配置 checkpoint，在目标数据集 test 集评估，并把 OOD 预测（logits/labels/attrs）落盘为 npz，
供 D 阶段汇总（Overall / worst-group / gap，5-seed 均值±CI）。

复用 `eval_ood_cxr.evaluate_ood`（架构/公平性/loader 全量复用 MIMIC 实现）；目标 test loader 每方向
只构建一次、跨方法/seed 复用（省 I/O）。checkpoint 命名：SWAD=`_averaged.pth`，余=`_best_overall.pth`。

用法：
  python -m src.training.run_ood_cxr_all --direction both
  python -m src.training.run_ood_cxr_all --direction m2c   # 仅 MIMIC→CheXpert
"""

import argparse
from pathlib import Path

from src.training.eval_ood_cxr import build_target_test_loader, evaluate_ood
from src.training.harness.predictions import save_predictions

OUTPUTS_ROOT = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs")
SEEDS = (42, 43, 44, 45, 46)
METHODS = ("erm", "swad", "hyperhead", "hyperfusion", "hyperadapt")

# 各数据集 Pareto 选定的确认配置（config_tag），源自协议清单 C2.5 / C3.5
CONFIRMED_CONFIG: dict[str, dict[str, str]] = {
    "mimic": {
        "erm": "lr3e-04_wd1e-03", "swad": "lr3e-04_wd1e-03",
        "hyperhead": "lr1e-04_wd1e-04", "hyperfusion": "lr1e-04_wd1e-04",
        "hyperadapt": "lr3e-04_wd1e-04",
    },
    "chexpert": {
        "erm": "lr3e-04_wd1e-03", "swad": "lr3e-04_wd1e-03",
        "hyperhead": "lr3e-05_wd1e-03", "hyperfusion": "lr1e-04_wd1e-04",
        "hyperadapt": "lr3e-04_wd1e-03",
    },
}

# 数据集名 → outputs 子目录（子群键=mimic/chexpert，但目录带 _cxr 后缀）
OUTPUT_DIR = {"mimic": OUTPUTS_ROOT / "mimic_cxr", "chexpert": OUTPUTS_ROOT / "chexpert_cxr"}


def checkpoint_path(source: str, method: str, seed: int) -> Path:
    """解析源数据集上确认配置的 checkpoint 路径（SWAD=_averaged，余=_best_overall）。"""
    config_tag = CONFIRMED_CONFIG[source][method]
    suffix = "averaged" if method == "swad" else "best_overall"
    return OUTPUT_DIR[source] / f"{method}_{config_tag}_seed{seed}_{suffix}.pth"


def run_direction(source: str, target: str) -> None:
    """
    跑单方向 OOD（source 模型 → target test），5 方法 × 5 seed，逐 run 落盘 OOD 预测 npz。

    Args:
        source: 源数据集名（权重来源）。
        target: 目标数据集名（test 集来源）。
    """
    out_dir = OUTPUTS_ROOT / "ood_cxr" / f"{source}2{target}"
    print(f"\n{'=' * 70}\nOOD 方向: {source} → {target}   输出目录: {out_dir}\n{'=' * 70}")
    test_loader = build_target_test_loader(target)  # 每方向构建一次，跨方法/seed 复用

    for method in METHODS:
        for seed in SEEDS:
            ckpt = checkpoint_path(source, method, seed)
            if not ckpt.exists():
                raise FileNotFoundError(f"缺少 checkpoint：{ckpt}")
            result = evaluate_ood(
                method, ckpt, source=source, target=target,
                test_loader=test_loader, report=False,
            )
            er = result.eval_result
            npz_path = out_dir / f"{method}_seed{seed}.npz"
            save_predictions(npz_path, er.labels, er.logits, er.attrs)
            wc = f"{er.wc_auc:.4f}" if er.wc_auc is not None else "N/A"
            print(f"  {method:<12} seed{seed}: Overall={er.overall_auc:.4f} "
                  f"worst={wc}  → {npz_path.name}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数（--direction m2c/c2m/both）。"""
    p = argparse.ArgumentParser(description="C6 OOD 全量评估（MIMIC↔CheXpert 双向）")
    p.add_argument("--direction", choices=("m2c", "c2m", "both"), default="both",
                   help="OOD 方向：m2c=MIMIC→CheXpert，c2m=CheXpert→MIMIC，both=双向。")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.direction in ("m2c", "both"):
        run_direction("mimic", "chexpert")
    if args.direction in ("c2m", "both"):
        run_direction("chexpert", "mimic")
    print("\nC6 OOD 全量评估完成。")


if __name__ == "__main__":
    main()
