"""
Train ResNet18 on HAM10000 with GroupDRO (malignant, image-only backbone + 子群感知损失)
========================================================================================
比较协议 §1 的**第 4 个基线**：训练时使用子群标签，但**不条件化函数**。
模型与 ERM 完全相同（image-only `ResNet18Pretrained`，不吃 sex/age），**唯一差异是损失**——
把 BCE 均值换成 Sagawa et al. (ICLR 2020) 的 online greedy GroupDRO（见 harness/groupdro.py）。

**它填的对照格**（导师指出的协议缺口）：ERM/SWAD 不用属性、ROC 只做事后阈值调整，于是「训练时用
子群标签」这一格里只有 3 个 HN ⇒ 「HN 用属性 vs 基线不用属性」是混淆对比。GroupDRO 用属性但只改
**损失权重**（reweighting），HN 用属性改**函数本身**（conditioning）。二者对信号的要求不同：
conditioning 要兑现增益必须 `I(Y;A|X)>0`，reweighting 不需要——它只是拿 Overall 换 worst-group。

**与 ERM 的对齐**：同 backbone / 同 transforms / 同 AdamW(lr,wd 来自同一 6 配置网格) / 同 batch=128 /
同早停（val Overall AUC, patience=5）/ 同双 checkpoint。以下差异是**有意的、已记录**的：
  1. **组均匀采样（默认开）**：论文 Algorithm 1 的第一步就是 `g ~ Uniform(1..m)` 后再从该组取样，
     官方实现即 `--reweight_groups`（与 `--robust` 同时使用），故**它是 GroupDRO 的组成部分而非
     额外技巧**。实现上复用 `src/utils/resampling.build_group_label_sampler(dims=[sex,age], alpha=1)`
     （权重 ∝ 1/n_g，每 epoch 样本数不变）。故本方法与 ERM 的差异是「损失 + 采样」两处，二者**共同**
     构成 GroupDRO；若要拆开看，`--groupdro_sampler standard` 给出「只有对抗损失、批次仍自然分布」
     的消融臂。
  2. **训练/评估集过滤 `age_group>=0`**（train 7958→7775 等）。GroupDRO 需要每个训练样本恰属一个
     canonical 交叉子群，而 0-20 组（age_group=-1）不在 Sex×Age 8 格内。这与 3 个 HN 训练脚本的
     口径**完全一致**（HN 的 embedding 同样吃不了 -1），且 CV-OOF 评估侧本来就 `age>=0` 过滤，
     故不引入新的口径分叉；相对 ERM 少 2.3% 训练样本这一点在报告中如实标注。
  3. **η（GroupDRO 步长）固定为论文默认 0.01，不进超参网格**。否则 GroupDRO 比其它 5 个方法多一个
     选择自由度，破坏「同一 6 配置网格 + 同一选择规程」的公平比较。`--groupdro_eta` 仅供敏感性分析。

分组 = `subgroup_masks("ham10000")` 的 Sex×Age 8 个交叉子群，与报告 canonical worst-group 所 min
的组**同源**（优化目标与评估目标一致，否则「优化 A 组、报告 B 组」不可解释）。

运行方式：重型任务，经 sbatch 提交（复用 slurm/c1_ham_cv_search.sh，见文件头示例）。
"""

import argparse
from pathlib import Path

import numpy as np
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.ham10000_dataset import HAM10000Dataset
from src.models.resnet18_pretrained import ResNet18Pretrained
from src.training.harness.groupdro import (
    DEFAULT_GROUP_ADJUSTMENT, DEFAULT_STEP_SIZE, GroupDROObjective, GroupIndexer,
)
from src.training.harness.hparam_grid import get_hparam_config, grid_size
from src.training.harness.run import seed_everything
from src.training.harness.train_loop import DEVICE, forward_image_only, run_training, unpack_sex_age
from src.utils.ham10000_fairness import print_ham10000_fairness_report
from src.utils.resampling import build_group_label_sampler

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "ham10000_baseline.yaml"
OUTPUT_DIR = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/ham10000")

DATASET = "ham10000"
METHOD = "groupdro"


def load_config(path: Path) -> dict:
    """读取 YAML 训练配置。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """解析命令行参数（超参网格 / seed / CV 折 / GroupDRO 自身两个超参）。"""
    parser = argparse.ArgumentParser(
        description="Train ResNet18 on HAM10000 with GroupDRO (subgroup-aware baseline)")
    parser.add_argument("--config_index", type=int, default=2,
                        help=f"超参网格配置序号 0–{grid_size() - 1}；缺省 2=中心点 (lr=1e-4, wd=1e-4)。")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子；缺省取 ham10000_baseline.yaml 的 training.seeds[0]")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="覆盖 config 的 batch_size（与其它方法等 batch 对照时用）。")
    parser.add_argument("--cv", type=int, default=0,
                        help="K 折 lesion 级 GroupKFold（OOF regime 用 5）；0=单次 80/10/10（默认）。")
    parser.add_argument("--fold", type=int, default=None,
                        help="--cv>0 时指定折号 k ∈ [0,K)。")
    parser.add_argument("--groupdro_sampler", type=str, default="balanced",
                        choices=("balanced", "standard"),
                        help="训练批次采样：balanced=组均匀采样（论文 Algorithm 1 / 官方 "
                             "--reweight_groups，默认）；standard=自然分布 shuffle（消融臂，"
                             "只保留对抗损失）。standard 时 method 记为 groupdro_nobal，产物与主臂隔离。")
    parser.add_argument("--groupdro_eta", type=float, default=DEFAULT_STEP_SIZE,
                        help=f"GroupDRO 步长 η（默认 {DEFAULT_STEP_SIZE}=Sagawa 论文默认）。"
                             f"**不进超参网格**——仅供敏感性分析，改动会使本方法比其它方法多一个"
                             f"选择自由度。")
    parser.add_argument("--groupdro_adjustment", type=float, default=DEFAULT_GROUP_ADJUSTMENT,
                        help=f"GroupDRO 泛化调整系数 C（默认 {DEFAULT_GROUP_ADJUSTMENT}=关闭）。"
                             f">0 时按 C/√n_g 额外上调小组权重（论文 §5）。")
    return parser.parse_args()


def resolve_paths(cfg: dict, cv: int, fold: int | None) -> tuple[Path, Path]:
    """
    按 --cv/--fold 解析 (split_dir, output_dir)。CV 时 split 取 cv{K}/fold{k}，output_dir 重定向到
    OUTPUT_DIR/cv{K}，与单次划分完全隔离（与 ERM/HN 脚本同约定）。
    """
    base = REPO_ROOT / cfg["data"]["split_dir"]
    if cv and cv >= 2:
        if fold is None or not (0 <= fold < cv):
            raise ValueError(f"--cv={cv} 需配合合法 --fold ∈ [0,{cv})")
        return base / f"cv{cv}" / f"fold{fold}", OUTPUT_DIR / f"cv{cv}"
    return base, OUTPUT_DIR


def build_transforms(cfg: dict) -> tuple[transforms.Compose, transforms.Compose]:
    """构建训练 / 评估 transform（与 HAM ERM baseline 严格一致）。"""
    t_cfg = cfg["transforms"]
    norm_mean, norm_std = t_cfg["normalize"]["mean"], t_cfg["normalize"]["std"]
    train_c, eval_c = t_cfg["train"], t_cfg["eval"]

    train_ops: list = [
        transforms.Resize(tuple(train_c["resize"])),
        transforms.RandomCrop(train_c["random_crop"]),
    ]
    if train_c.get("horizontal_flip", False):
        train_ops.append(transforms.RandomHorizontalFlip())
    train_ops.append(transforms.RandomRotation(degrees=train_c["rotation_degrees"]))
    train_ops += [transforms.ToTensor(), transforms.Normalize(mean=norm_mean, std=norm_std)]
    train_transform = transforms.Compose(train_ops)

    eval_transform = transforms.Compose([
        transforms.Resize(tuple(eval_c["resize"])),
        transforms.CenterCrop(eval_c["center_crop"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])
    return train_transform, eval_transform


def _filter_age_valid(dataset: HAM10000Dataset) -> int:
    """
    就地过滤 `age_group==-1`（0-20 排除组），只保留 age_group∈{0..3}。返回剩余样本数。

    与 3 个 HN 训练脚本同一实现口径：GroupDRO 要求每样本恰属一个 Sex×Age 交叉子群，而 -1 不在
    canonical 8 格内（harness.groupdro 会显式报错而非静默漏样本）。
    """
    mask = np.asarray(dataset.age_groups).astype(int) >= 0
    dataset.image_paths = dataset.image_paths[mask]
    dataset.labels = dataset.labels[mask]
    dataset.sexes = dataset.sexes[mask]
    dataset.age_groups = dataset.age_groups[mask]
    return int(mask.sum())


def get_dataloaders(
    cfg: dict, split_dir: Path,
    train_transform: transforms.Compose, eval_transform: transforms.Compose,
    sampler_mode: str = "balanced",
) -> tuple[DataLoader, DataLoader, DataLoader, HAM10000Dataset]:
    """
    构建 train/val/test DataLoader（三 split 均过滤 age_group>=0），并返回 train_set 供统计各组 n。

    Args:
        sampler_mode: "balanced"=train 用组均匀 WeightedRandomSampler（论文 Algorithm 1 的
            `g ~ Uniform`，dims=[sex,age] 8 格、alpha=1、每 epoch 样本数不变）；
            "standard"=自然分布 shuffle（消融臂）。val/test 恒为自然分布、不采样。

    Returns:
        (train_loader, val_loader, test_loader, train_set)。
    """
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["dataloader"]["num_workers"]
    pin_memory = cfg["dataloader"]["pin_memory"]

    train_set = HAM10000Dataset(split_dir / "train.csv", transform=train_transform)
    val_set = HAM10000Dataset(split_dir / "val.csv", transform=eval_transform)
    test_set = HAM10000Dataset(split_dir / "test.csv", transform=eval_transform)

    n_tr, n_va, n_te = _filter_age_valid(train_set), _filter_age_valid(val_set), _filter_age_valid(test_set)
    print(f"  过滤 age_group>=0 后: train {n_tr:,} / val {n_va:,} / test {n_te:,}（已排除 0-20）")

    if sampler_mode == "balanced":
        # 组均匀采样：dims=[sex,age] 恰为 GroupDRO 的 8 个 canonical 交叉子群（不含 label——
        # 平衡的是**组**而非类别，与 GroupDRO 的组定义严格同源）
        sampler = build_group_label_sampler(train_set, dims=["sex", "age"], alpha=1.0, verbose=True)
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=pin_memory)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader, train_set


def _fairness_report(y_true, y_score, attrs) -> None:
    """HAM 公平性报告回调：从 attrs 取 sex / age(=age_group) 分组键。"""
    print_ham10000_fairness_report(
        y_true=y_true, y_score=y_score, sex=attrs["sex"], age_group=attrs["age"], threshold=0.0,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(CONFIG_PATH)
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    seed = args.seed if args.seed is not None else cfg["training"]["seeds"][0]
    hparam = get_hparam_config(args.config_index)
    split_dir, output_dir = resolve_paths(cfg, args.cv, args.fold)

    seed_everything(seed)
    print(f"Loading HAM10000 (malignant, GroupDRO baseline)  "
          f"split_dir={split_dir.name}  output_dir={output_dir.name}  "
          f"sampler={args.groupdro_sampler}  η={args.groupdro_eta}  C={args.groupdro_adjustment}...")
    train_transform, eval_transform = build_transforms(cfg)
    train_loader, val_loader, test_loader, train_set = get_dataloaders(
        cfg, split_dir, train_transform, eval_transform, sampler_mode=args.groupdro_sampler)

    # 训练集各组 n：用于 group adjustment 的 C/√n_g，并作为日志诊断（最小格通常是 Female|80+）
    train_group_counts = GroupIndexer(DATASET).counts_numpy(
        sex=np.asarray(train_set.sexes), age=np.asarray(train_set.age_groups))
    objective = GroupDROObjective(
        DATASET, step_size=args.groupdro_eta, group_adjustment=args.groupdro_adjustment,
        train_group_counts=train_group_counts, device=DEVICE,
    )

    # 消融臂（关组均匀采样）另立 method 名，产物与主臂完全隔离、互不覆盖
    method = METHOD if args.groupdro_sampler == "balanced" else f"{METHOD}_nobal"

    run_training(
        dataset=DATASET, method=method, output_dir=output_dir, cfg=cfg, hparam=hparam, seed=seed,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        build_model=lambda: ResNet18Pretrained(num_classes=1),
        unpack_fn=unpack_sex_age, forward_fn=forward_image_only,
        fairness_report_fn=_fairness_report,
        objective=objective,
    )


if __name__ == "__main__":
    main()
