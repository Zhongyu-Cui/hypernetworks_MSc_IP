"""
E-audit.2（推理侧）：M→C OOD **full-target** 评估驱动
=====================================================
条件化范围消融方案（docs/conditioning_ablation_plan.md §3.3）把 OOD 主估计量由 legacy **k→k**
改为 **full-target**：

  · **legacy k→k**（`run_ood_cxr_cv.py`）：source fold-k 模型 → 只评 target fold-k test，5 折 disjoint
    并集 = target OOF（每个 target 样本恰被 1 个 source 模型预测 1 次）。省 5× 推理，但**每个模型只在
    1/5 的 target 上被测**，模型间比较混入了「各自被分到哪批 target 样本」的差异。
  · **full-target**（本脚本）：**每个 source fold×seed 模型都评完整 target**（全 138,644 张），
    对 `(f,s)` 等权平均得配对效应。所有模型在**同一批完整 target 样本**上被比较，配对更干净，
    是 plan 冻结的**确认性主估计量**；k→k 降为 legacy sensitivity。

历史 M→C 胜利（HyperFusion/HyperAdapt marginal worst 显著超 ERM 与 SWAD，见 docs/oof_regime_results.md）
是 legacy k→k 口径的结论，**须在本口径下复核**才作定论——这正是 E-audit 的 go/no-go 依据（plan §3.1）。

**零泄漏**：source=MIMIC 的模型与 config 选择只用 MIMIC 自身（cv5 selected_configs.json，S1 折均
marginal-worst，全程只看 MIMIC validation）；CheXpert **仅作最终评估**，不参与任何训练/选择。故在
完整 CheXpert（含其 train/val 段）上评估**不构成泄漏**——这些数据对 MIMIC 模型而言完全未见。

**patient_id 落盘**：plan §3.3 的推断是**患者级配对 cluster bootstrap**，故预测须带聚类键。
MIMICCXRDataset 不过滤行、且 loader `shuffle=False`，行序 == CSV 行序，故按位置附加 patient_id
（附 len 断言）。经 save_predictions 的 `extra` 通道存盘，不污染 attrs。

用法（重型 GPU 推理，经 sbatch 提交；--method 供 SLURM array 按方法并行）：
    python -m src.training.run_ood_cxr_full_target                 # 全部 5 方法
    python -m src.training.run_ood_cxr_full_target --method erm    # 单方法（array 用）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader

# DataLoader worker 默认用 file_descriptor 策略共享张量：每个 batch 占若干 fd，本作业按方法并行
# （同一节点可能并发多个任务 × 多 worker）时会耗尽 fd 上限，报
# 「RuntimeError: Too many open files」（首次提交作业 72185 的 task 1/2 即因此挂掉）。
# 改用 file_system 策略：共享内存经 /dev/shm 命名对象而非 fd 传递，不受 ulimit -n 约束。
torch.multiprocessing.set_sharing_strategy("file_system")

from src.datasets.mimic_cxr_dataset import MIMICCXRDataset
from src.training.eval_ood_cxr import build_eval_transform, load_config
from src.training.eval_ood_cxr import evaluate_ood
from src.training.harness.predictions import save_predictions
from src.paths import OUTPUTS_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = OUTPUTS_DIR
ABLATION_DIR = OUTPUTS_ROOT / "conditioning_ablation"

FOLD_SEEDS = (42, 43, 44, 45, 46)   # fold k ↔ seed 42+k（沿用既有 cv5 checkpoint 命名）
CV = 5
# 末尾追加实验 G 的融合臂（HyperAdapt+SWAD）与 GroupDRO 基线——追加不插入，既有方法顺序不变。
METHODS = ("erm", "swad", "hyperhead", "hyperfusion", "hyperadapt", "hyperadapt_swad", "groupdro")

SOURCE = "mimic"
TARGET = "chexpert"
OUTPUT_DIR = {"mimic": OUTPUTS_ROOT / "mimic_cxr", "chexpert": OUTPUTS_ROOT / "chexpert_cxr"}
SPLIT_SUBDIR = {"mimic": "mimic_cxr_nofinding", "chexpert": "chexpert_nofinding"}


def build_full_target_csv(target: str = TARGET, cv: int = CV) -> Path:
    """
    生成（并缓存）完整 target 的 test CSV = cv{K} 五折 test 的并集。

    五折 test 互斥且并集 = 全库，故该并集即完整 target（CheXpert n=138,644）。写到 outputs 下的
    消融目录而非 data/splits（后者进 git；这是可由五折确定性重建的派生索引，不入库）。

    Args:
        target: 目标数据集名。
        cv    : 折数。

    Returns:
        完整 target CSV 的路径（已存在则直接复用）。
    """
    out_csv = ABLATION_DIR / f"{target}_full_target_test.csv"
    if out_csv.exists():
        return out_csv
    split_dir = REPO_ROOT / "data" / "splits" / SPLIT_SUBDIR[target]
    frames = [pd.read_csv(split_dir / f"cv{cv}" / f"fold{k}" / "test.csv") for k in range(cv)]
    df = pd.concat(frames, ignore_index=True)
    # 五折 test 应互斥（并集无重复图像），否则 full-target 会重复计入某些样本
    assert not df["image_path"].duplicated().any(), "五折 test 并集出现重复 image_path，折划分有误"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"已生成完整 target CSV: {out_csv}  n={len(df):,}")
    return out_csv


def build_full_target_loader(
    target: str = TARGET, batch_size: int = 128, num_workers: int = 4,
) -> tuple[DataLoader, pd.DataFrame]:
    """
    构建**完整 target** 的评估 loader（shuffle=False，行序 == CSV 行序）。

    Returns:
        (loader, df)：df 为该 CSV 的 DataFrame，供按位置取 patient_id 作 cluster 键。
    """
    cfg = load_config(target)
    csv_path = build_full_target_csv(target)
    df = pd.read_csv(csv_path)
    test_set = MIMICCXRDataset(
        csv_path, transform=build_eval_transform(cfg),
        image_size=cfg["data"]["image_size"],
        age_threshold=cfg["attributes"]["age"]["age_threshold"],
    )
    assert len(test_set) == len(df), "Dataset 长度与 CSV 行数不一致（行序对齐假设被破坏）"
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    return loader, df


def load_selected_config(dataset: str) -> dict[str, str]:
    """读某数据集 cv5 的 S1 选定配置（{method: config_tag}）——只用 source 自身的选择结果。"""
    path = OUTPUT_DIR[dataset] / "cv5" / "selected_configs.json"
    return json.loads(path.read_text())["config"]


def checkpoint_path(source: str, method: str, config_tag: str, seed: int) -> Path:
    """
    源数据集 cv5 该折选定 config 的 checkpoint。

    **凡权重平均派生的方法**（ERM 派生的 `swad`，以及实验 G 的 `<hn>_swad` 融合臂）都只有单一
    平均模型 `_averaged.pth`（无 overall/worstcase 之分）；其余方法取 `_best_overall.pth`。
    """
    suffix = "averaged" if method.endswith("swad") else "best_overall"
    return OUTPUT_DIR[source] / "cv5" / f"{method}_{config_tag}_seed{seed}_{suffix}.pth"


def fold_trial_seeds(trials: tuple[int, ...], cv: int = CV) -> list[tuple[int, int, int]]:
    """
    展开 (fold, trial, seed) 三元组。seed 编码见 `docs/ood_experiment_v2_preregistration.md` §2.1：
    `seed = 42 + fold + 10 × trial`（trial 0 = 既有 42–46，trial 1/2 = 52–56 / 62–66）。
    """
    return [(f, t, 42 + f + 10 * t) for t in trials for f in range(cv)]


def run_full_target(methods: tuple[str, ...], source: str = SOURCE, target: str = TARGET,
                    trials: tuple[int, ...] = (0,), skip_existing: bool = True,
                    num_workers: int = 4) -> None:
    """
    对每个 method、每个 source (fold, trial)，在**完整 target** 上评估并落盘预测（含 patient_id）。

    Args:
        methods: 要跑的方法子集（供 SLURM array 按方法并行）。
        source: 权重来源数据集。
        target: 评估数据集。
        trials: 要跑的 trial 编号；缺省 (0,) = 原行为。
        skip_existing: 已存在的预测直接跳过（M→C trial 0 已由 E-audit.2 产出，避免重复推断）。
        num_workers: DataLoader 工作进程数。**全 target 推理是 I/O 密集**（MIMIC 19.9 万张
            从 NFS 读），worker 太少会让 GPU 饿死——作业 73789 在 deepmedic2 上实测 GPU 利用率
            恒 0%、CPU 时间仅涨 7.5%，单个模型 >31 min 仍未完成。负载高的节点上须调高此值。
    """
    src_cfg = load_selected_config(source)
    out_pred = OUTPUTS_ROOT / "ood_cxr" / f"{source}2{target}" / "cv5_full_target" / "predictions"
    print(f"\n{'=' * 74}\nfull-target OOD: {source} → {target}（每个 source fold×trial 评完整 target）"
          f"\ntrials={trials}  methods={methods}\n输出: {out_pred}\n{'=' * 74}")

    loader, df = build_full_target_loader(target=target, num_workers=num_workers)
    patient_id = df["patient_id"].values
    print(f"完整 target: n={len(df):,}  唯一患者={df['patient_id'].nunique():,}  "
          f"正例率={df['label'].mean():.4f}")

    for method in methods:
        tag = src_cfg[method]
        for fold, trial, seed in fold_trial_seeds(trials):
            npz_path = out_pred / f"{method}_{tag}_seed{seed}_overall.npz"
            if skip_existing and npz_path.exists():
                print(f"  {method:<12} fold{fold} trial{trial} seed{seed}: 已存在，跳过")
                continue
            ckpt = checkpoint_path(source, method, tag, seed)
            if not ckpt.exists():
                raise FileNotFoundError(f"缺少 checkpoint：{ckpt}")
            result = evaluate_ood(method, ckpt, source=source, target=target,
                                  test_loader=loader, report=False)
            er = result.eval_result
            assert len(er.labels) == len(patient_id), "预测数与 CSV 行数不一致（patient_id 对齐失败）"
            save_predictions(npz_path, er.labels, er.logits, er.attrs,
                             extra={"patient_id": patient_id})
            wc = f"{er.wc_auc:.4f}" if er.wc_auc is not None else "N/A"
            print(f"  {method:<12} fold{fold} trial{trial} seed{seed}: "
                  f"Overall={er.overall_auc:.4f} worst={wc}  n={len(er.labels):,} → {npz_path.name}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数（--method 限定单方法，供 SLURM array 并行）。"""
    p = argparse.ArgumentParser(description="CXR OOD full-target 评估（E-audit.2 / OOD v2 阶段 3）")
    p.add_argument("--method", choices=METHODS, default=None,
                   help="只跑指定方法（缺省=全部 5 方法）。SLURM array 按方法并行时用。")
    p.add_argument("--direction", choices=("m2c", "c2m"), default="m2c",
                   help="m2c=MIMIC→CheXpert（缺省，原行为）；c2m=CheXpert→MIMIC。")
    p.add_argument("--trials", default="0",
                   help="逗号分隔的 trial 编号（seed = 42 + fold + 10×trial）；缺省 '0' = 原行为。")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader 进程数；全 target 推理是 I/O 密集，高负载节点建议 8-12。")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="即使预测已存在也重算（缺省跳过，避免重复的全 target 推断）。")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # 快速失败：坏 GPU 节点上 CUDA 初始化会静默失败（torch 把可见设备数置 0），harness 的
    # DEVICE 便退化为 CPU——693k 次前向在 CPU 上等同挂死，且日志看不出异常（作业 72191 的
    # mira05 即如此）。宁可立刻报错、换节点重投，也不要静默跑 CPU。
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA 不可用：本作业为重型 GPU 推理，拒绝静默退化到 CPU。"
            "多为坏 GPU 节点（如 mira05 的 'CUDA unknown error'），请用 --exclude 换节点重投。"
        )
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    methods = (args.method,) if args.method else METHODS
    source, target = ("mimic", "chexpert") if args.direction == "m2c" else ("chexpert", "mimic")
    trials = tuple(int(t) for t in args.trials.split(","))
    run_full_target(methods, source=source, target=target, trials=trials,
                    skip_existing=not args.no_skip_existing, num_workers=args.num_workers)
    print("\nfull-target OOD 推理完成。分析见 scripts/eaudit_m2c_full_target.py")


if __name__ == "__main__":
    main()
