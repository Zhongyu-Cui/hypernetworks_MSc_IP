"""
CV-OOF OOD 全量评估驱动（CXR：MIMIC ↔ CheXpert 双向，5 方法 × 5 折）
=====================================================================
`docs/oof_selection_rollout_plan.md` R-MIMIC 末项：MIMIC↔CheXpert OOD 用**各自 OOF-选定 config 的逐折
checkpoint** 跨库评估。本驱动是 `run_ood_cxr_all.py`（单-split）的 CV-OOF 版：

  对每个方向 source→target、每个方法（ERM/SWAD + 3 HN；ROC 是分数后处理不在此）、每折 k（seed=42+k）：
    加载 source **cv5 fold-k** 的选定 config checkpoint → 在 target **cv5 fold-k** test 上评估 → 落盘预测。
  5 折 disjoint target-fold test 并集 = 全 target，故池化 = **target OOF**（每 target 样本恰被一个
  source-fold 模型预测一次），与 ID 的 CV-OOF 同构，可直接用 cv_oof_report 池化 + 显著性。

**与单-split OOD 的隔离**：预测写 `outputs/ood_cxr/{s}2{t}/cv5/predictions/`，配置读各数据集
`cv5/selected_configs.json`（S1 折均 marginal-worst 选定）。checkpoint 命名 SWAD=`_averaged`，余=`_best_overall`。

**为何 source-fold-k → target-fold-k**：source 与 target 是不同数据集、无泄漏，fold 对应仅决定「哪个
source 模型预测哪批 target 样本」；取 disjoint 的 target-fold-k 使 5 折池化恰好覆盖全 target 一次
（OOF 式），比「每个 source 模型都过全 target 再平均」省 5× 推理且天然可池化做样本级显著性。

预测文件名 `{method}_{source_config}_seed{k}_overall.npz`——与 cv_oof_report.pool_oof 的寻址一致，
故 OOD 报告可复用：
  python scripts/cv_oof_report.py --dataset chexpert \
      --pred-dir outputs/ood_cxr/mimic2chexpert/cv5/predictions \
      --config <method>=<source_config> ... --section perf,sig

用法（重型 GPU 推理，经 sbatch 提交）：
  python -m src.training.run_ood_cxr_cv --direction both
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.training.eval_ood_cxr import build_target_test_loader, evaluate_ood
from src.training.harness.predictions import save_predictions

OUTPUTS_ROOT = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs")
FOLD_SEEDS = (42, 43, 44, 45, 46)   # fold k ↔ seed 42+k
CV = 5
# 末尾追加实验 G 的融合臂（HyperAdapt+SWAD）——追加不插入，既有方法顺序不变。
METHODS = ("erm", "swad", "hyperhead", "hyperfusion", "hyperadapt", "hyperadapt_swad")  # ROC 是分数后处理，不在此

# 数据集名（子群键）→ outputs 子目录（目录带 _cxr 后缀）
OUTPUT_DIR = {"mimic": OUTPUTS_ROOT / "mimic_cxr", "chexpert": OUTPUTS_ROOT / "chexpert_cxr"}


def load_selected_config(dataset: str) -> dict[str, str]:
    """读某数据集 cv5 的 S1 选定配置（{method: config_tag}）。"""
    path = OUTPUT_DIR[dataset] / "cv5" / "selected_configs.json"
    return json.loads(path.read_text())["config"]


def checkpoint_path(source: str, method: str, config_tag: str, seed: int) -> Path:
    """
    源数据集 cv5 该折选定 config 的 checkpoint。

    **凡权重平均派生的方法**（ERM 派生的 `swad`、实验 G 的 `<hn>_swad`）都只有单一平均模型
    `_averaged.pth`；其余方法取 `_best_overall.pth`。
    """
    suffix = "averaged" if method.endswith("swad") else "best_overall"
    return OUTPUT_DIR[source] / "cv5" / f"{method}_{config_tag}_seed{seed}_{suffix}.pth"


def run_direction(source: str, target: str, methods: tuple[str, ...] = METHODS,
                  config_override: dict[str, str] | None = None,
                  out_subdir: str = "cv5") -> None:
    """
    跑单方向 CV-OOF OOD（source cv5 逐折模型 → target cv5 逐折 test），逐 run 落盘预测。

    Args:
        source: 源数据集名（权重来源）。
        target: 目标数据集名（test 集来源）。
        methods: 要评估的方法子集；缺省 = 全部 5 个方法。
        config_override: {方法: config_tag}，覆盖 `selected_configs.json` 的选定配置。
            用于**超参匹配对照**（如让 ERM 走 HyperAdapt 的 lr/wd），破除「HN 赢是因为选到了
            更好超参」这一混杂。缺省 = 不覆盖。
        out_subdir: 预测输出的子目录名；对照实验须传非 "cv5" 的名字以**隔离主结果**。
    """
    src_cfg = dict(load_selected_config(source))
    if config_override:
        src_cfg.update(config_override)
    out_pred = OUTPUTS_ROOT / "ood_cxr" / f"{source}2{target}" / out_subdir / "predictions"
    print(f"\n{'=' * 70}\nCV-OOF OOD: {source} → {target}   输出: {out_pred}\n{'=' * 70}")
    # 每折的 target test loader 只建一次，跨方法复用
    fold_loaders = {k: build_target_test_loader(target, cv=CV, fold=k) for k in range(CV)}

    for method in methods:
        tag = src_cfg[method]
        for fold, seed in enumerate(FOLD_SEEDS):
            ckpt = checkpoint_path(source, method, tag, seed)
            if not ckpt.exists():
                raise FileNotFoundError(f"缺少 checkpoint：{ckpt}")
            result = evaluate_ood(
                method, ckpt, source=source, target=target,
                test_loader=fold_loaders[fold], report=False,
            )
            er = result.eval_result
            npz_path = out_pred / f"{method}_{tag}_seed{seed}_overall.npz"
            save_predictions(npz_path, er.labels, er.logits, er.attrs)
            wc = f"{er.wc_auc:.4f}" if er.wc_auc is not None else "N/A"
            print(f"  {method:<12} fold{fold} seed{seed}: Overall={er.overall_auc:.4f} "
                  f"worst={wc}  n={len(er.labels):,}  → {npz_path.name}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数（--direction m2c/c2m/both）。"""
    p = argparse.ArgumentParser(description="CV-OOF OOD 全量评估（MIMIC↔CheXpert 双向）")
    p.add_argument("--direction", choices=("m2c", "c2m", "both"), default="both",
                   help="m2c=MIMIC→CheXpert，c2m=CheXpert→MIMIC，both=双向。")
    p.add_argument("--methods", default=None,
                   help=f"逗号分隔的方法子集（缺省=全部 {','.join(METHODS)}）。")
    p.add_argument("--config-m2c", default=None,
                   help="m2c 方向的 config 覆盖，形如 'erm=lr1e-04_wd1e-03'（逗号分隔多项）。")
    p.add_argument("--config-c2m", default=None,
                   help="c2m 方向的 config 覆盖，形如 'erm=lr3e-04_wd1e-04'。")
    p.add_argument("--out-subdir", default="cv5",
                   help="预测输出子目录；对照实验须传非 cv5 的名字以隔离主结果。")
    return p.parse_args()


def parse_override(spec: str | None) -> dict[str, str]:
    """把 'erm=lr1e-04_wd1e-03,swad=...' 解析成 {方法: config_tag}。"""
    if not spec:
        return {}
    out = {}
    for item in spec.split(","):
        method, _, tag = item.partition("=")
        if not tag:
            raise ValueError(f"config 覆盖项格式应为 '方法=config_tag'，收到 {item!r}")
        out[method.strip()] = tag.strip()
    return out


def main() -> None:
    args = parse_args()
    methods = tuple(m.strip() for m in args.methods.split(",")) if args.methods else METHODS
    if args.direction in ("m2c", "both"):
        run_direction("mimic", "chexpert", methods=methods,
                      config_override=parse_override(args.config_m2c),
                      out_subdir=args.out_subdir)
    if args.direction in ("c2m", "both"):
        run_direction("chexpert", "mimic", methods=methods,
                      config_override=parse_override(args.config_c2m),
                      out_subdir=args.out_subdir)
    print("\nCV-OOF OOD 全量评估完成。池化报告见 cv_oof_report --pred-dir .../cv5/predictions。")


if __name__ == "__main__":
    main()
