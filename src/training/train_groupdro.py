"""
GroupDRO 基线通用训练入口（MIMIC / CheXpert / Fitzpatrick）
=====================================================================
比较协议 §1 第 4 基线 GroupDRO（Sagawa et al., ICLR 2020）在**除 HAM 外四个数据集**上的训练入口。
算法本体见 `src/training/harness/groupdro.py`；HAM 有更早写成的专用脚本
`train_ham10000_groupdro.py`（多一步 `age_group>=0` 过滤），其已完成的作业 75498 保持原样不动。

**为什么是一个参数化脚本而不是四份复制**：各数据集之间只有「配置路径 / transforms / DataLoader
构建 / 公平性报告 / unpack 回调 / 分组维度」不同，而这些**在各自的 ERM 脚本里已经存在且是单一
事实来源**。本脚本按 `--dataset` 从对应 ERM 模块取这些部件，只做两件 GroupDRO 特有的事：

  1. 把 train loader 换成**组均匀采样**（论文 Algorithm 1 的 `g ~ Uniform(1..m)`，
     即官方 `--reweight_groups`）：复用 `resampling.build_group_label_sampler(dims=<组维度>, alpha=1)`；
  2. 把训练目标换成 `GroupDROObjective`（`run_training(objective=...)`）。

由此保证「GroupDRO 与该数据集 ERM 的差异**只有损失+采样**」这一单变量性质**由构造保证**，
而不是靠人工比对两份脚本是否抄对（这正是四份复制最容易出错的地方）。

**分组**：`GroupIndexer` 复用 `subgroup_masks` 的 canonical 交叉子群 ⇒ 训练所优化的组 =
报告 worst-group 所 min 的组。各库组数：MIMIC/CheXpert = Sex×Race×Age 8；Fitzpatrick = Skin 6；
三库均无 HAM 那样的「排除组」，故无需过滤（构造时的互斥完备断言会兜底）。

**超参**：lr/wd 走同一 6 配置网格；η 固定论文默认 0.01、C=0，**不进网格**（否则本方法比其它方法
多一个选择自由度）。

运行（复用各库既有的 CV 搜索 array 脚本，接口一致 `--config_index/--seed/--cv/--fold`）：
    sbatch --partition=gpus24 --job-name=mimic_gdro_cvsearch \
      --output=logs/mimic_gdro_cvsearch.%N.%A_%a.log \
      --export=ALL,PY_SCRIPT=src/training/train_groupdro.py,GDRO_DATASET=mimic \
      slurm/c2_mimic_cv_search.sh
⚠️ 数据集必须经环境变量 **`GDRO_DATASET`** 传入（不是 `DATASET`）：array 脚本以固定的
`--config_index/--seed/--cv/--fold` 调用 `$PY_SCRIPT`，没有位置塞 `--dataset`，故本文件的
`--dataset` 缺省值读该环境变量（见 parse_args）。直接命令行运行时用 `--dataset` 亦可。
"""

import argparse
import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from torch.utils.data import DataLoader

from src.models.resnet18_pretrained import ResNet18Pretrained
from src.training.harness.groupdro import (
    DEFAULT_GROUP_ADJUSTMENT, DEFAULT_STEP_SIZE, GroupDROObjective, GroupIndexer,
)
from src.training.harness.hparam_grid import get_hparam_config, grid_size
from src.training.harness.run import seed_everything
from src.training.harness.train_loop import (
    DEVICE, UnpackFn, forward_image_only, run_training,
    unpack_sex_age, unpack_sex_race_age, unpack_skin,
)
from src.utils.resampling import build_group_label_sampler

METHOD = "groupdro"


@dataclass(frozen=True)
class DatasetSpec:
    """
    一个数据集的 GroupDRO 训练规格：从哪个 ERM 模块借部件、按什么维度分组、如何解析路径。

    Attributes:
        erm_module : 该数据集 ERM 训练脚本的模块路径（提供 CONFIG_PATH / load_config /
            build_transforms / get_dataloaders / _fairness_report / OUTPUT_DIR）。
        group_dims : 组均匀采样的分组维度，须与 `groupdro.GROUP_ATTRS[ds_key]` 一致
            （前者喂 resampling 的 cell 编码，后者喂 GroupIndexer 的 canonical 子群，
            二者是同一个划分的两种编码；main() 里有断言）。
        unpack_fn  : loader 元组 → (images, labels, attrs) 的回调。
        cv_output_subdir: CV 时 output_dir 是否加 `cv{K}` 子目录。
    """

    erm_module: str
    group_dims: tuple[str, ...]
    unpack_fn: UnpackFn
    cv_output_subdir: bool


DATASET_SPECS: dict[str, DatasetSpec] = {
    "mimic": DatasetSpec("src.training.train_mimic_resnet18", ("sex", "race", "age"),
                         unpack_sex_race_age, True),
    "chexpert": DatasetSpec("src.training.train_chexpert_resnet18", ("sex", "race", "age"),
                            unpack_sex_race_age, True),
    "fitzpatrick": DatasetSpec("src.training.train_fitzpatrick_resnet18", ("skin",),
                               unpack_skin, True),
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数（数据集 + 与各库 ERM 脚本一致的网格/seed/CV 接口 + GroupDRO 自身超参）。"""
    p = argparse.ArgumentParser(description="GroupDRO baseline (subgroup-aware) 通用训练入口")
    # 各库既有的 CV array 脚本以 `python $PY_SCRIPT --config_index .. --seed .. --cv .. --fold ..`
    # 固定调用、无处塞 --dataset，故额外支持环境变量 GDRO_DATASET（经 sbatch --export 传入；
    # 环境变量不含逗号/空格，规避 --export 的取值限制）。CLI 显式给出时优先。
    p.add_argument("--dataset", type=str, default=os.environ.get("GDRO_DATASET"),
                   choices=tuple(DATASET_SPECS),
                   help="数据集 key（缺省读环境变量 GDRO_DATASET）。"
                        "HAM10000 用专用脚本 train_ham10000_groupdro.py")
    p.add_argument("--config_index", type=int, default=2,
                   help=f"超参网格配置序号 0–{grid_size() - 1}；缺省 2=中心点 (lr=1e-4, wd=1e-4)。")
    p.add_argument("--seed", type=int, default=None, help="随机种子；缺省取该库 yaml 的 training.seeds[0]")
    p.add_argument("--batch_size", type=int, default=None, help="覆盖 config 的 batch_size。")
    p.add_argument("--cv", type=int, default=0, help="K 折 GroupKFold（OOF regime 用 5）；0=单-split。")
    p.add_argument("--fold", type=int, default=None, help="--cv>0 时指定折号 k ∈ [0,K)。")
    p.add_argument("--groupdro_sampler", type=str, default="balanced",
                   choices=("balanced", "standard"),
                   help="balanced=组均匀采样（论文 Algorithm 1 / 官方 --reweight_groups，默认）；"
                        "standard=自然分布 shuffle（消融臂，method 记为 groupdro_nobal）。")
    p.add_argument("--groupdro_eta", type=float, default=DEFAULT_STEP_SIZE,
                   help=f"GroupDRO 步长 η（默认 {DEFAULT_STEP_SIZE}=论文默认）。**不进超参网格**。")
    p.add_argument("--groupdro_adjustment", type=float, default=DEFAULT_GROUP_ADJUSTMENT,
                   help=f"泛化调整系数 C（默认 {DEFAULT_GROUP_ADJUSTMENT}=关闭），>0 时按 C/√n_g 上调小组。")
    return p.parse_args()


def resolve_paths(mod, spec: DatasetSpec, cfg: dict, cv: int, fold: int | None) -> tuple[Path, Path]:
    """
    解析 (split_dir, output_dir)，逐库复用其 ERM 模块自己的解析函数以免口径漂移。

    Args:
        mod : 该数据集的 ERM 模块。
        spec: 数据集规格。
        cfg : 已读入的 yaml 配置。
        cv / fold: CV 折设置（0 = 单-split）。

    Returns:
        (split_dir, output_dir)。
    """
    # 四库的 ERM 模块都提供 resolve_paths（同时给出 split_dir 与 output_dir）
    return mod.resolve_paths(cfg, cv, fold)


def build_loaders(
    mod, cfg: dict, split_dir: Path,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    调该数据集 ERM 模块的 `get_dataloaders` 建三个 loader（各库签名统一为 4 个位置参数）。

    Returns:
        (train_loader, val_loader, test_loader)。
    """
    train_transform, eval_transform = mod.build_transforms(cfg)
    loaders = mod.get_dataloaders(cfg, split_dir, train_transform, eval_transform)
    return loaders[0], loaders[1], loaders[2]


def rebuild_train_loader_balanced(
    cfg: dict, train_loader: DataLoader, group_dims: tuple[str, ...],
) -> DataLoader:
    """
    把 ERM 的 `shuffle=True` train loader 换成**组均匀采样**（带放回，每 epoch 样本数不变）。

    alpha=1 时样本权重 w_i ∝ 1/n_{g(i)} ⇒ 每组总权重相等 ⇒ 每次抽样命中各组概率均为 1/G，
    即论文 Algorithm 1 的 `g ~ Uniform(1..m)` 后 `(x,y) ~ P_g`。

    Args:
        cfg         : yaml 配置（取 batch_size / num_workers / pin_memory）。
        train_loader: ERM 建好的训练 loader（只借它的 dataset）。
        group_dims  : 分组维度（= GroupDRO 的组定义）。

    Returns:
        采样器已替换的新 train loader。
    """
    sampler = build_group_label_sampler(
        train_loader.dataset, dims=list(group_dims), alpha=1.0, verbose=True)
    return DataLoader(
        train_loader.dataset, batch_size=cfg["training"]["batch_size"], sampler=sampler,
        num_workers=cfg["dataloader"]["num_workers"], pin_memory=cfg["dataloader"]["pin_memory"],
    )


def main() -> None:
    args = parse_args()
    if args.dataset is None:
        raise SystemExit("需 --dataset 或环境变量 GDRO_DATASET（合法："
                         f"{tuple(DATASET_SPECS)}）")
    spec = DATASET_SPECS[args.dataset]
    mod = importlib.import_module(spec.erm_module)

    cfg = mod.load_config(mod.CONFIG_PATH)
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    seed = args.seed if args.seed is not None else cfg["training"]["seeds"][0]
    hparam = get_hparam_config(args.config_index)
    split_dir, output_dir = resolve_paths(mod, spec, cfg, args.cv, args.fold)

    indexer = GroupIndexer(args.dataset)
    # 采样维度（resampling 的 cell 编码）与分组维度（GroupIndexer 的 canonical 子群）必须同源，
    # 否则会出现「按 A 划分采样、按 B 划分加权」的错配
    assert tuple(spec.group_dims) == indexer.attr_names, (spec.group_dims, indexer.attr_names)

    seed_everything(seed)                                   # 规程：建 loader 前播种
    print(f"Loading {args.dataset} (GroupDRO baseline)  split_dir={split_dir.name}  "
          f"output_dir={output_dir.name}  sampler={args.groupdro_sampler}  "
          f"η={args.groupdro_eta}  C={args.groupdro_adjustment}...")
    train_loader, val_loader, test_loader = build_loaders(mod, cfg, split_dir)
    if args.groupdro_sampler == "balanced":
        train_loader = rebuild_train_loader_balanced(cfg, train_loader, spec.group_dims)
    print(f"  Train: {len(train_loader.dataset):,} | Val: {len(val_loader.dataset):,} | "
          f"Test: {len(test_loader.dataset):,}")

    objective = GroupDROObjective(
        args.dataset, step_size=args.groupdro_eta, group_adjustment=args.groupdro_adjustment,
        train_group_counts=indexer.counts_from_dataset(train_loader.dataset), device=DEVICE,
    )
    # 消融臂（关组均匀采样）另立 method 名，产物与主臂完全隔离
    method = METHOD if args.groupdro_sampler == "balanced" else f"{METHOD}_nobal"

    run_training(
        dataset=args.dataset, method=method, output_dir=output_dir, cfg=cfg, hparam=hparam,
        seed=seed, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        build_model=lambda: ResNet18Pretrained(num_classes=1),
        unpack_fn=spec.unpack_fn, forward_fn=forward_image_only,
        fairness_report_fn=mod._fairness_report,
        objective=objective,
    )


if __name__ == "__main__":
    main()
