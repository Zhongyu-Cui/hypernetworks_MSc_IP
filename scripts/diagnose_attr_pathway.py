"""
超网络变体诊断脚本：属性通路是否"活着" + EqOdds 是否为阈值/基率伪影
=====================================================================
针对已训好的 MIMIC-CXR No Finding 模型（seed 42, best_overall checkpoint），在测试集上做两项诊断：

诊断 1（属性翻转敏感度）——回答"模型是否在真正使用敏感属性"
    对每个含属性的变体（HyperHead / HyperAdapt / HyperFusion / AttrConcat），保持图像不变，
    分别翻转 sex / race / age / 全部三者，测量输出 logit 的变化幅度。若变化幅度相对 logit
    本身的动态范围接近 0，说明属性通路已坍缩成恒等映射（模型实际忽略了属性），那么
    "公平性未提升"的根因是优化目标而非属性信号弱。同时对比变体与 baseline 的 logit 差异，
    验证"变体≈baseline"。

诊断 2（逐组阈值 EqOdds 复算）——回答"Age 维度的 EqOdds gap 有多少是阈值/基率伪影"
    现有报告用全局固定阈值（logit>0）算 TPR/FPR。No Finding（健康）阳性率随年龄下降，
    >=60 组基率低，固定阈值下 TPR 必然偏低。本诊断对每组改用"按组内真实阳性率定阈值"
    （prevalence-matched）后重算 TPR/FPR/gap，并报告各组 AUC（排序质量），以剥离基率伪影、
    看真实可修复的 gap 有多大。

运行方式:
    推理任务（对测试集多次前向），按项目规范通过 sbatch 提交到 SLURM 集群
    （见 slurm/diagnose_attr_pathway.sh）。
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.mimic_cxr_dataset import MIMICCXRDataset
from src.models.resnet18_pretrained import ResNet18Pretrained
from src.models.resnet18_attrconcat import ResNet18AttrConcat
from src.models.resnet18_hyperhead import ResNet18HyperHead
from src.models.resnet18_hyperadapt import ResNet18HyperAdapt
from src.models.resnet18_hyperfusion import ResNet18HyperFusion

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "mimic_cxr_baseline.yaml"
OUTPUT_DIR = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/mimic_cxr")
SEED = 42  # 用 seed 42 的 best_overall checkpoint 做诊断

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 含属性的变体: name -> (构造器, checkpoint 文件名)
ATTR_MODELS = {
    "HyperHead":   (ResNet18HyperHead,   f"hyperhead_no_finding_seed{SEED}_best_overall.pth"),
    "HyperAdapt":  (ResNet18HyperAdapt,  f"hyperadapt_no_finding_seed{SEED}_best_overall.pth"),
    "HyperFusion": (ResNet18HyperFusion, f"hyperfusion_no_finding_seed{SEED}_best_overall.pth"),
    "AttrConcat":  (ResNet18AttrConcat,  f"attrconcat_no_finding_seed{SEED}_best_overall.pth"),
}
BASELINE_CKPT = f"resnet18_no_finding_seed{SEED}_best_overall.pth"

GROUP_NAMES = {
    "Sex":  {0: "Male",  1: "Female"},
    "Race": {0: "White", 1: "Non-White"},
    "Age":  {0: "<60",   1: ">=60"},
}


def build_eval_loader() -> DataLoader:
    """构建与训练 eval 完全一致的测试集 DataLoader。"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    t_cfg = cfg["transforms"]
    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(t_cfg["eval"]["resize"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=t_cfg["normalize"]["mean"], std=t_cfg["normalize"]["std"]),
    ])
    split_dir = REPO_ROOT / cfg["data"]["split_dir"]
    test_set = MIMICCXRDataset(
        split_dir / "test.csv", transform=eval_transform,
        image_size=cfg["data"]["image_size"], age_threshold=cfg["attributes"]["age"]["age_threshold"],
    )
    return DataLoader(
        test_set, batch_size=cfg["training"]["batch_size"], shuffle=False,
        num_workers=cfg["dataloader"]["num_workers"], pin_memory=cfg["dataloader"]["pin_memory"],
    )


@torch.no_grad()
def collect_logits(
    model: nn.Module,
    loader: DataLoader,
    takes_attrs: bool,
    flip_sex: bool = False,
    flip_race: bool = False,
    flip_age: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    在测试集上收集 logit 与 (label, sex, race, age)。

    Args:
        takes_attrs: 模型 forward 是否接收属性（baseline 为 False，只吃图像）。
        flip_sex/flip_race/flip_age: 是否在送入模型前翻转对应属性（仅影响模型输入，
            返回的属性数组始终为真实值，供分组使用）。

    Returns:
        (logits, labels, sexes, races, ages)，均为 [N] numpy 数组。
    """
    model.eval()
    out_logits, out_labels, out_sex, out_race, out_age = [], [], [], [], []
    for images, labels, sexes, races, ages in loader:
        images = images.to(DEVICE, non_blocking=True)
        if takes_attrs:
            in_sex  = (1 - sexes) if flip_sex  else sexes
            in_race = (1 - races) if flip_race else races
            in_age  = (1 - ages)  if flip_age  else ages
            logits = model(images, in_sex.to(DEVICE), in_race.to(DEVICE), in_age.to(DEVICE))
        else:
            logits = model(images)
        out_logits.append(logits.squeeze(1).cpu().numpy())
        out_labels.append(labels.numpy())
        out_sex.append(sexes.numpy())
        out_race.append(races.numpy())
        out_age.append(ages.numpy())
    return (
        np.concatenate(out_logits), np.concatenate(out_labels),
        np.concatenate(out_sex), np.concatenate(out_race), np.concatenate(out_age),
    )


def tpr_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """给定二值预测，返回 (TPR, FPR)。"""
    pos = y_true == 1
    neg = y_true == 0
    tpr = float(y_pred[pos].mean()) if pos.sum() > 0 else float("nan")
    fpr = float(y_pred[neg].mean()) if neg.sum() > 0 else float("nan")
    return tpr, fpr


def threshold_for_rate(scores: np.ndarray, target_rate: float) -> float:
    """
    返回一个阈值，使 (scores > thr) 的占比≈target_rate（prevalence-matched 工作点）。
    target_rate 为该组预测为正的目标比例（取该组真实阳性率）。
    """
    if target_rate <= 0:
        return float(np.max(scores)) + 1.0   # 全判负
    if target_rate >= 1:
        return float(np.min(scores)) - 1.0   # 全判正
    # 第 (1-target_rate) 分位数即为使正例占比=target_rate 的阈值
    return float(np.quantile(scores, 1.0 - target_rate))


def eqodds_by_threshold_mode(
    logits: np.ndarray, labels: np.ndarray, group_ids: np.ndarray, names: dict,
) -> dict:
    """
    对一个分组维度，分别在两种工作点下报告各组 AUC / 基率 / TPR / FPR / EqOdds gap：
      - global  : 全局固定阈值 logit>0（复现现有报告）
      - prevmatch: 每组按组内真实阳性率定阈值（剥离基率/阈值伪影）
    """
    result = {"global": {}, "prevmatch": {}, "group_auc": {}, "base_rate": {}}
    global_pred = (logits > 0.0).astype(int)

    tprs_g, fprs_g, tprs_p, fprs_p = {}, {}, {}, {}
    for gid, gname in names.items():
        mask = group_ids == gid
        if mask.sum() == 0:
            continue
        yt, sc = labels[mask], logits[mask]
        base_rate = float((yt == 1).mean())
        result["base_rate"][gname] = base_rate
        result["group_auc"][gname] = (
            float(roc_auc_score(yt, sc)) if len(np.unique(yt)) > 1 else float("nan")
        )
        # 全局阈值
        tpr, fpr = tpr_fpr(yt, global_pred[mask])
        result["global"][gname] = {"tpr": tpr, "fpr": fpr}
        tprs_g[gname], fprs_g[gname] = tpr, fpr
        # 逐组 prevalence-matched 阈值
        thr = threshold_for_rate(sc, base_rate)
        tpr_p, fpr_p = tpr_fpr(yt, (sc > thr).astype(int))
        result["prevmatch"][gname] = {"tpr": tpr_p, "fpr": fpr_p, "thr": thr}
        tprs_p[gname], fprs_p[gname] = tpr_p, fpr_p

    def _gap(d: dict) -> float:
        vals = [v for v in d.values() if not np.isnan(v)]
        return float(max(vals) - min(vals)) if len(vals) > 1 else float("nan")

    # EqOdds gap = max(TPR gap, FPR gap)，与项目 eqodd_by_group 口径一致
    result["global_gap"] = max(_gap(tprs_g), _gap(fprs_g))
    result["prevmatch_gap"] = max(_gap(tprs_p), _gap(fprs_p))
    return result


def run_diagnostic_1(loader: DataLoader, baseline_logits: np.ndarray) -> dict:
    """诊断 1：属性翻转敏感度 + 与 baseline 的差异。"""
    print("\n" + "=" * 78)
    print("诊断 1：属性翻转敏感度（模型是否真在用敏感属性）")
    print("=" * 78)
    report = {}
    for name, (ctor, ckpt_name) in ATTR_MODELS.items():
        ckpt_path = OUTPUT_DIR / ckpt_name
        if not ckpt_path.exists():
            print(f"\n[{name}] 跳过：找不到 checkpoint {ckpt_path}")
            continue
        model = ctor(num_classes=1).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))

        orig, labels, sex, race, age = collect_logits(model, loader, takes_attrs=True)
        flip_s, *_ = collect_logits(model, loader, takes_attrs=True, flip_sex=True)
        flip_r, *_ = collect_logits(model, loader, takes_attrs=True, flip_race=True)
        flip_a, *_ = collect_logits(model, loader, takes_attrs=True, flip_age=True)
        flip_all, *_ = collect_logits(
            model, loader, takes_attrs=True, flip_sex=True, flip_race=True, flip_age=True
        )

        logit_std = float(orig.std())
        def _summ(delta: np.ndarray) -> dict:
            d = np.abs(delta)
            return {
                "mean_abs_delta": float(d.mean()),
                "median_abs_delta": float(np.median(d)),
                "ratio_to_logit_std": float(d.mean() / (logit_std + 1e-12)),
                "pred_flip_frac": float(((orig > 0) != ((orig + delta) > 0)).mean()),
            }

        # 变体 vs baseline：原始 logit 的差异 + 各自 AUC
        diff_to_base = np.abs(orig - baseline_logits)
        entry = {
            "logit_std": logit_std,
            "auc": float(roc_auc_score(labels, orig)),
            "vs_baseline_mean_abs_logit_diff": float(diff_to_base.mean()),
            "vs_baseline_corr": float(np.corrcoef(orig, baseline_logits)[0, 1]),
            "flip_sex": _summ(flip_s - orig),
            "flip_race": _summ(flip_r - orig),
            "flip_age": _summ(flip_a - orig),
            "flip_all": _summ(flip_all - orig),
        }
        report[name] = entry

        print(f"\n[{name}]  AUC={entry['auc']:.4f}   logit std={logit_std:.3f}")
        print(f"  与 baseline 比：mean|Δlogit|={entry['vs_baseline_mean_abs_logit_diff']:.4f}, "
              f"corr={entry['vs_baseline_corr']:.4f}")
        print(f"  {'翻转项':<10}{'mean|Δlogit|':>14}{'相对logit_std':>16}{'预测翻转占比':>14}")
        for flip_name, key in [("sex", "flip_sex"), ("race", "flip_race"),
                               ("age", "flip_age"), ("全部", "flip_all")]:
            s = entry[key]
            print(f"  {flip_name:<10}{s['mean_abs_delta']:>14.4f}"
                  f"{s['ratio_to_logit_std']:>16.4f}{s['pred_flip_frac']:>14.4f}")
    return report


def run_diagnostic_2(loader: DataLoader) -> dict:
    """诊断 2：逐组阈值 EqOdds 复算（全局阈值 vs prevalence-matched）。"""
    print("\n" + "=" * 78)
    print("诊断 2：逐组阈值 EqOdds 复算（剥离基率/阈值伪影）")
    print("=" * 78)

    models = {"Baseline": (ResNet18Pretrained, BASELINE_CKPT, False)}
    for name, (ctor, ckpt) in ATTR_MODELS.items():
        models[name] = (ctor, ckpt, True)

    report = {}
    for name, (ctor, ckpt_name, takes_attrs) in models.items():
        ckpt_path = OUTPUT_DIR / ckpt_name
        if not ckpt_path.exists():
            print(f"\n[{name}] 跳过：找不到 checkpoint {ckpt_path}")
            continue
        model = ctor(num_classes=1).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        logits, labels, sex, race, age = collect_logits(model, loader, takes_attrs=takes_attrs)

        report[name] = {}
        print(f"\n[{name}]  Overall AUC={roc_auc_score(labels, logits):.4f}")
        for dim, gids in [("Sex", sex), ("Race", race), ("Age", age)]:
            res = eqodds_by_threshold_mode(logits, labels, gids, GROUP_NAMES[dim])
            report[name][dim] = res
            print(f"  --- {dim} ---")
            for gname in GROUP_NAMES[dim].values():
                if gname not in res["global"]:
                    continue
                g, p = res["global"][gname], res["prevmatch"][gname]
                print(f"    {gname:<10} 基率={res['base_rate'][gname]:.3f} "
                      f"AUC={res['group_auc'][gname]:.4f} | "
                      f"全局阈值 TPR={g['tpr']:.3f} FPR={g['fpr']:.3f} | "
                      f"逐组阈值 TPR={p['tpr']:.3f} FPR={p['fpr']:.3f}")
            print(f"    EqOdds gap: 全局阈值={res['global_gap']:.4f}  "
                  f"逐组(prevalence-matched)阈值={res['prevmatch_gap']:.4f}")
    return report


def main() -> None:
    print(f"Device: {DEVICE}   Seed checkpoint: {SEED} (best_overall)")
    loader = build_eval_loader()
    print(f"Test set: {len(loader.dataset):,} images")

    # baseline logits 供诊断 1 做"变体≈baseline"对比
    base_model = ResNet18Pretrained(num_classes=1).to(DEVICE)
    base_model.load_state_dict(
        torch.load(OUTPUT_DIR / BASELINE_CKPT, map_location=DEVICE, weights_only=True)
    )
    baseline_logits, base_labels, *_ = collect_logits(base_model, loader, takes_attrs=False)
    print(f"Baseline AUC (ref): {roc_auc_score(base_labels, baseline_logits):.4f}  "
          f"(logit std={baseline_logits.std():.3f})")

    diag1 = run_diagnostic_1(loader, baseline_logits)
    diag2 = run_diagnostic_2(loader)

    out_json = OUTPUT_DIR / "diagnostic_attr_pathway.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"diagnostic_1": diag1, "diagnostic_2": diag2}, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_json}")


if __name__ == "__main__":
    main()
