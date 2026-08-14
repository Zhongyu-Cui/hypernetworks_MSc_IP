"""
统一训练入口：ResNet18CondNet 条件化范围消融（ERM / C-Head / C-Deep / C-Full）
==============================================================================
条件化范围消融方案（docs/conditioning_ablation_plan.md）的**唯一训练脚本**：一个入口驱动
四个 cell（`--location` ∈ {none,head,deep,full}）× 四个数据集（`--dataset`），全部走统一
harness.run_training，只在数据集维度分派 transforms / dataloader / 属性过滤 / 公平性报告。

与既有各 train_*_hyperadapt.py 的关键区别：
  1. 模型统一为 ResNet18CondNet（location 开关），而非固定注入深度的 HyperAdapt；
  2. **canonical base state（硬约束，plan §1）**：所有 cell（含比较用 ERM=location "none"）在同一
     (dataset, fold, seed) 下用 build_base_fc_seed 派生的确定性种子构建，保证 base backbone / base fc
     逐元素相等——否则「Δθ=0 等价 ERM」不对*比较用* ERM 成立。故本消融的 ERM **不用**旧 baseline
     （ResNet18Pretrained），而用 location="none" 的 CondNet，与 conditioning cell 共享同一 base state。

方法命名：method = f"condnet_{location}"（none→"condnet_erm"），run_id 据此唯一寻址、与既有
结果互不覆盖；产物落 outputs/<dataset>[/cv{K}]，可直接进 CV-OOF 选择/评估管线。

运行方式：重型任务，经 sbatch 提交（见 slurm/train_condnet.sh）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.fitzpatrick_dataset import FitzpatrickDataset
from src.datasets.ham10000_dataset import HAM10000Dataset
from src.datasets.mimic_cxr_dataset import MIMICCXRDataset
from src.models.resnet18_condnet import (
    ResNet18CondNet,
    ResNet18CondNetAge,
    ResNet18CondNetSkin,
    build_base_fc_seed,
)
from src.training.harness.hparam_grid import get_hparam_config, grid_size
from src.training.harness.run import run_id, seed_everything
from src.training.harness.train_loop import (
    forward_age,
    forward_image_only,
    forward_skin,
    forward_sex_race_age,
    run_training,
    unpack_skin,
    unpack_sex_age,
    unpack_sex_race_age,
)
from src.utils.fitzpatrick_fairness import print_fitzpatrick_fairness_report
from src.utils.ham10000_fairness import print_ham10000_fairness_report
from src.utils.mimic_fairness import print_mimic_fairness_report

REPO_ROOT = Path(__file__).resolve().parents[2]
# 消融产物独立成根，与既有各数据集 outputs（旧 fold↔seed 绑定布局）物理隔离，互不覆盖
OUTPUT_ROOT = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/conditioning_ablation")

# location → method 命名（none = 比较用 ERM，与 conditioning cell 共享 canonical base state）
LOCATION_TO_METHOD = {
    "none": "condnet_erm",
    "head": "condnet_head",
    "deep": "condnet_deep",
    "full": "condnet_full",
    # E1 敏感性实验③（fc-off）：关 fc 条件化、只留 conv，检验超网络是否被逼激活 conv
    "deep_nofc": "condnet_deep_nofc",
    "full_nofc": "condnet_full_nofc",
}


# ============================================================
# 公平性报告回调（依数据集）
# ============================================================
def _fairness_ham(y_true, y_score, attrs) -> None:
    """HAM：sex / age(=age_group) 分组。"""
    print_ham10000_fairness_report(
        y_true=y_true, y_score=y_score, sex=attrs["sex"], age_group=attrs["age"], threshold=0.0,
    )


def _fairness_fitz(y_true, y_score, attrs) -> None:
    """Fitzpatrick：6 肤色子群（不报 EqOdd）。"""
    print_fitzpatrick_fairness_report(
        y_true=y_true, y_score=y_score, skin=attrs["skin"], threshold=0.0,
    )


def _fairness_cxr(y_true, y_score, attrs) -> None:
    """MIMIC / CheXpert：sex / race / age 分组。"""
    print_mimic_fairness_report(
        y_true=y_true, y_score=y_score,
        sex=attrs["sex"], race=attrs["race"], age=attrs["age"], threshold=0.0,
    )


# ============================================================
# transforms（依数据集，照搬各 baseline）
# ============================================================
def _transforms_ham(cfg: dict) -> tuple[transforms.Compose, transforms.Compose]:
    """HAM：Resize(256)→RandomCrop(224)→Flip→Rotation；eval Resize(256)→CenterCrop(224)。"""
    t = cfg["transforms"]
    mean, std = t["normalize"]["mean"], t["normalize"]["std"]
    tr, ev = t["train"], t["eval"]
    ops: list = [transforms.Resize(tuple(tr["resize"])), transforms.RandomCrop(tr["random_crop"])]
    if tr.get("horizontal_flip", False):
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.RandomRotation(degrees=tr["rotation_degrees"]))
    ops += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    eval_t = transforms.Compose([
        transforms.Resize(tuple(ev["resize"])), transforms.CenterCrop(ev["center_crop"]),
        transforms.ToTensor(), transforms.Normalize(mean=mean, std=std),
    ])
    return transforms.Compose(ops), eval_t


def _transforms_fitz(cfg: dict) -> tuple[transforms.Compose, transforms.Compose]:
    """Fitzpatrick：224 原生，train Flip→Rotation，无 Resize/Crop。"""
    t = cfg["transforms"]
    mean, std = t["normalize"]["mean"], t["normalize"]["std"]
    ops: list = []
    if t["train"].get("horizontal_flip", False):
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.RandomRotation(degrees=t["train"]["rotation_degrees"]))
    ops += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    eval_t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(ops), eval_t


def _transforms_cxr(cfg: dict) -> tuple[transforms.Compose, transforms.Compose]:
    """MIMIC / CheXpert：Grayscale(3)→Resize(224)→Rotation（不水平翻转）。"""
    t = cfg["transforms"]
    mean, std = t["normalize"]["mean"], t["normalize"]["std"]
    train_t = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(t["train"]["resize"]),
        transforms.RandomRotation(degrees=t["train"]["rotation_degrees"]),
        transforms.ToTensor(), transforms.Normalize(mean=mean, std=std),
    ])
    eval_t = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(t["eval"]["resize"]),
        transforms.ToTensor(), transforms.Normalize(mean=mean, std=std),
    ])
    return train_t, eval_t


# ============================================================
# 属性有效性过滤（仅 HAM：排除 age_group==-1）
# ============================================================
def _filter_age_valid(dataset: HAM10000Dataset) -> int:
    """就地过滤 age_group==-1（0-20），只保留 age_group∈{0..3}（embedding 不能吃 -1）。返回剩余数。"""
    mask = np.asarray(dataset.age_groups).astype(int) >= 0
    dataset.image_paths = dataset.image_paths[mask]
    dataset.labels = dataset.labels[mask]
    dataset.sexes = dataset.sexes[mask]
    dataset.age_groups = dataset.age_groups[mask]
    return int(mask.sum())


# ============================================================
# 数据集规格（dispatch 表）
# ============================================================
class DatasetSpec:
    """封装某数据集的：config 路径、输出根、Dataset 构建、CondNet 工厂、harness 回调。"""

    def __init__(
        self,
        name: str,
        config_name: str,
        output_subdir: str,
        build_transforms: Callable[[dict], tuple[transforms.Compose, transforms.Compose]],
        build_datasets: Callable[[dict, Path, transforms.Compose, transforms.Compose], tuple],
        unpack_fn: Callable,
        attr_forward_fn: Callable,
        fairness_fn: Callable,
        cond_factory: Callable,
    ) -> None:
        self.name = name
        self.config_path = REPO_ROOT / "configs" / config_name
        self.output_dir = OUTPUT_ROOT / output_subdir
        self.build_transforms = build_transforms
        self.build_datasets = build_datasets
        self.unpack_fn = unpack_fn
        self.attr_forward_fn = attr_forward_fn
        self.fairness_fn = fairness_fn
        self.cond_factory = cond_factory


def _build_ham_datasets(cfg, split_dir, train_t, eval_t):
    """HAM：三 split 过滤 age_group>=0。"""
    tr = HAM10000Dataset(split_dir / "train.csv", transform=train_t)
    va = HAM10000Dataset(split_dir / "val.csv", transform=eval_t)
    te = HAM10000Dataset(split_dir / "test.csv", transform=eval_t)
    n_tr, n_va, n_te = _filter_age_valid(tr), _filter_age_valid(va), _filter_age_valid(te)
    print(f"  过滤 age_group>=0 后: train {n_tr:,} / val {n_va:,} / test {n_te:,}（已排除 0-20）")
    return tr, va, te


def _build_fitz_datasets(cfg, split_dir, train_t, eval_t):
    """Fitzpatrick：无需过滤。"""
    return (FitzpatrickDataset(split_dir / "train.csv", transform=train_t),
            FitzpatrickDataset(split_dir / "val.csv", transform=eval_t),
            FitzpatrickDataset(split_dir / "test.csv", transform=eval_t))


def _build_cxr_datasets(cfg, split_dir, train_t, eval_t):
    """MIMIC / CheXpert：age 由 age_threshold 二值化（MIMICCXRDataset 内部处理）。"""
    image_size = cfg["data"]["image_size"]
    age_threshold = cfg["attributes"]["age"]["age_threshold"]
    kw = dict(image_size=image_size, age_threshold=age_threshold)
    return (MIMICCXRDataset(split_dir / "train.csv", transform=train_t, **kw),
            MIMICCXRDataset(split_dir / "val.csv", transform=eval_t, **kw),
            MIMICCXRDataset(split_dir / "test.csv", transform=eval_t, **kw))


def _cond_factory_age(location, base_fc_seed, pretrained, freeze):
    return ResNet18CondNetAge(location=location, num_age=4, pretrained=pretrained,
                              base_fc_seed=base_fc_seed, freeze_backbone=freeze)


def _cond_factory_skin(location, base_fc_seed, pretrained, freeze):
    return ResNet18CondNetSkin(location=location, num_skin=6, pretrained=pretrained,
                               base_fc_seed=base_fc_seed, freeze_backbone=freeze)


def _cond_factory_patient(location, base_fc_seed, pretrained, freeze):
    return ResNet18CondNet(location=location, num_sex=2, num_race=2, num_age=2,
                           pretrained=pretrained, base_fc_seed=base_fc_seed, freeze_backbone=freeze)


DATASET_SPECS: dict[str, DatasetSpec] = {
    "ham10000": DatasetSpec(
        "ham10000", "ham10000_baseline.yaml", "ham10000",
        _transforms_ham, _build_ham_datasets, unpack_sex_age, forward_age, _fairness_ham,
        _cond_factory_age,
    ),
    "fitzpatrick": DatasetSpec(
        "fitzpatrick", "fitzpatrick_baseline.yaml", "fitzpatrick",
        _transforms_fitz, _build_fitz_datasets, unpack_skin, forward_skin, _fairness_fitz,
        _cond_factory_skin,
    ),
    "mimic": DatasetSpec(
        "mimic", "mimic_cxr_baseline.yaml", "mimic_cxr",
        _transforms_cxr, _build_cxr_datasets, unpack_sex_race_age, forward_sex_race_age, _fairness_cxr,
        _cond_factory_patient,
    ),
    "chexpert": DatasetSpec(
        "chexpert", "chexpert_baseline.yaml", "chexpert_cxr",
        _transforms_cxr, _build_cxr_datasets, unpack_sex_race_age, forward_sex_race_age, _fairness_cxr,
        _cond_factory_patient,
    ),
}


# ============================================================
# 逐 epoch 机制诊断探针（E1 敏感性实验③：fc-off 后超网络是否激活 conv）
# ============================================================
# 单属性数据集的属性取值数（ρ 探针需枚举全部取值算逐层偏移）；多属性(cxr)不支持本探针。
SINGLE_ATTR_CARDINALITY = {"ham10000": 4, "fitzpatrick": 6}


class CondNetDiagnostics:
    """
    逐 epoch 探针，把三类量落到独立日志 `<output_dir>/diag_logs/<run_id>.jsonl`（不动 val 日志 schema）：

      1. **conv ρ / ρ^between**（对所有条件化 conv 层取均值 + 逐层）——超网络是否/多深地激活 conv；
      2. **conv adapter 梯度范数**（shared-A vs B_gen 分列，**裁剪前真实梯度**，epoch 内逐 batch 均值）
         ——A=0 初始化 ⇒ B 初始梯度为 0 的串行依赖是否被打破（优化瓶颈的直接信号，训练后不可复原）；
      3. **conv adapter 权重范数**（‖A_gen‖ / ‖B_gen‖）——零初始化的 shared-A 是否离开 0。

    仅对含 conv 条件化的单属性模型有意义（deep/full/deep_nofc/full_nofc × HAM/Fitzpatrick）。

    Args:
        model      : 已 .to(device) 的 ResNet18CondNet（含 conv 条件化通路）。
        num_attr   : 属性取值数 K（ρ 探针枚举 0..K-1 喂入条件编码器）。
        diag_path  : 诊断日志落盘路径（每 epoch 一条 JSON 行）。
    """

    def __init__(self, model: torch.nn.Module, num_attr: int, diag_path: Path) -> None:
        self.model = model
        self.num_attr = num_attr
        self.path = diag_path
        # 按参数名归组：shared-A（A_gen.*）vs conv 的 B 生成器（*.B_gen_conv* / *.B_gen_down）
        self.a_params = [p for n, p in model.named_parameters() if n.startswith("A_gen.")]
        self.b_params = [p for n, p in model.named_parameters()
                         if (".B_gen_conv" in n) or (".B_gen_down" in n)]
        assert self.a_params and self.b_params, "无 conv 条件化通路，诊断探针不适用（需 deep/full[_nofc]）"
        self._reset()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()                     # 本 run 重跑时清空旧诊断日志

    def _reset(self) -> None:
        """清空本 epoch 的梯度范数累积器。"""
        self._gsum_a = 0.0
        self._gsum_b = 0.0
        self._n_batches = 0

    @staticmethod
    def _group_grad_norm(params: list[torch.nn.Parameter]) -> float:
        """一组参数的梯度 L2 范数（跳过 grad=None）。"""
        sq = 0.0
        for p in params:
            if p.grad is not None:
                sq += float(p.grad.detach().pow(2).sum())
        return sq ** 0.5

    @staticmethod
    def _group_weight_norm(params: list[torch.nn.Parameter]) -> float:
        """一组参数的权重 L2 范数。"""
        return float(sum(float(p.detach().pow(2).sum()) for p in params)) ** 0.5

    @torch.no_grad()
    def on_after_backward(self, model: torch.nn.Module) -> None:
        """裁剪前逐 batch 累积 conv adapter 梯度范数（A / B 分列）。"""
        self._gsum_a += self._group_grad_norm(self.a_params)
        self._gsum_b += self._group_grad_norm(self.b_params)
        self._n_batches += 1

    @torch.no_grad()
    def epoch_record(self, epoch: int, er) -> None:
        """算本 epoch 的 ρ / 权重范数 + 收尾梯度范数均值，落盘一条 JSON，随后重置累积器。"""
        device = next(self.model.parameters()).device
        ages = torch.arange(self.num_attr, dtype=torch.long, device=device)
        emb = self.model.patient_embed(ages)                  # [K, embed_dim]
        rho = self.model.conditioning_rho(emb)                # {模块名 -> {rho, rho_between}}
        conv = {k: v for k, v in rho.items() if k != "fc"}
        n_b = max(1, self._n_batches)
        rec = {
            "epoch": epoch,
            "val_overall_auc": float(er.overall_auc),
            "val_worst_case_auc": (None if er.wc_auc is None else float(er.wc_auc)),
            "conv_rho_mean": float(np.mean([v["rho"] for v in conv.values()])),
            "conv_rho_between_mean": float(np.mean([v["rho_between"] for v in conv.values()])),
            "fc_rho": (rho["fc"]["rho"] if "fc" in rho else None),
            "fc_rho_between": (rho["fc"]["rho_between"] if "fc" in rho else None),
            "gradnorm_A_mean": self._gsum_a / n_b,            # 裁剪前真实梯度（epoch 内均值）
            "gradnorm_B_conv_mean": self._gsum_b / n_b,
            "wnorm_A": self._group_weight_norm(self.a_params),
            "wnorm_B_conv": self._group_weight_norm(self.b_params),
            "conv_rho_between_by_layer": {k: v["rho_between"] for k, v in conv.items()},
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._reset()


# ============================================================
# CLI + 主流程
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet18CondNet (条件化范围消融)")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_SPECS),
                        help="数据集名。")
    parser.add_argument("--location", required=True, choices=sorted(LOCATION_TO_METHOD),
                        help="条件化范围 cell：none(ERM)/head(C-Head)/deep(C-Deep)/full(C-Full)。")
    parser.add_argument("--config_index", type=int, default=2,
                        help=f"超参网格配置序号 0–{grid_size() - 1}；缺省 2=中心点 (lr=1e-4, wd=1e-4)。")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子；缺省取 config 的 training.seeds[0]。")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="覆盖 config 的 batch_size（重型 HN 显存对照须等 batch）。")
    parser.add_argument("--cv", type=int, default=0,
                        help="K 折 CV（路线 A 用 5）；0=单次划分（默认）。")
    parser.add_argument("--fold", type=int, default=None,
                        help="--cv>0 时指定折号 k ∈ [0,K)。")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="冻结 regime（冻结 backbone 卷积/BN，仅训生成器 + fc）。method 加 _frozen 后缀。")
    parser.add_argument("--diagnostics", action="store_true",
                        help="E1③ 逐 epoch 机制探针：conv ρ/ρ^between + adapter 梯度范数(A/B 分列) + "
                             "权重范数，落 diag_logs/<run_id>.jsonl。仅含 conv 条件化的单属性模型可用。")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    """读取 YAML 训练配置。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(spec: DatasetSpec, cfg: dict, cv: int, fold: int | None) -> tuple[Path, Path]:
    """
    按 --cv/--fold 解析 (split_dir, output_dir)。

    ⚠️ **output_dir 必须含 fold**（与既有 train_ham10000_*.py 的关键差异）：harness 的
    `run_id = <method>_<config_tag>_seed<seed>` **不含 fold**，故既有 CV 脚本被迫用 `seed=42+fold`
    把折号编进 seed（见 slurm/c1_ham_cv_search.sh 注释、scripts/cv_oof_report.py 的 FOLD_SEEDS），
    等于**把 fold 与 train_seed 绑死、无法同折多 seed**。本消融要求 fold 与 seed **解耦**
    （plan §1 canonical base state；§3「统一 3 seed」⇒ 5 折 × 3 seed），若沿用旧布局，
    `(fold=0,seed=42)` 与 `(fold=1,seed=42)` 会写到同一预测路径而**静默互相覆盖**。
    故把 fold 放进 output_dir，run_id 在该目录内即唯一。

    Returns:
        (split_dir, output_dir)；CV 时 output_dir = <ablation根>/<dataset>/cv{K}/fold{k}。
    """
    base = REPO_ROOT / cfg["data"]["split_dir"]
    if cv and cv >= 2:
        if fold is None or not (0 <= fold < cv):
            raise ValueError(f"--cv={cv} 需配合合法 --fold ∈ [0,{cv})")
        return base / f"cv{cv}" / f"fold{fold}", spec.output_dir / f"cv{cv}" / f"fold{fold}"
    return base, spec.output_dir


def main() -> None:
    args = parse_args()
    spec = DATASET_SPECS[args.dataset]
    cfg = load_config(spec.config_path)
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    seed = args.seed if args.seed is not None else cfg["training"]["seeds"][0]
    hparam = get_hparam_config(args.config_index)
    split_dir, output_dir = resolve_paths(spec, cfg, args.cv, args.fold)

    # canonical base state 种子：fold 未指定（单-split）记为 0
    fold = args.fold if (args.cv and args.cv >= 2) else 0
    base_fc_seed = build_base_fc_seed(args.dataset, fold=fold, seed=seed)

    method = LOCATION_TO_METHOD[args.location]
    if args.freeze_backbone:
        method += "_frozen"

    # 播种须在建 loader 前（harness 约定）
    seed_everything(seed)
    print(f"Loading {args.dataset} | location={args.location} (method={method}) | "
          f"split_dir={split_dir.name} output_dir={output_dir.name} | "
          f"base_fc_seed={base_fc_seed} (dataset={args.dataset}, fold={fold}, seed={seed})")

    train_t, eval_t = spec.build_transforms(cfg)
    train_set, val_set, test_set = spec.build_datasets(cfg, split_dir, train_t, eval_t)

    bs = cfg["training"]["batch_size"]
    nw = cfg["dataloader"]["num_workers"]
    pin = cfg["dataloader"]["pin_memory"]
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin)

    # ERM（location "none"）忽略属性 → forward_image_only；conditioning cell 用数据集属性 forward
    forward_fn = forward_image_only if args.location == "none" else spec.attr_forward_fn
    pretrained = cfg["model"]["pretrained"]

    # E1③ 逐 epoch 机制探针（可选）：须模型建成后挂钩，故用容器在 build_model 内捕获实例
    diag_holder: dict[str, CondNetDiagnostics] = {}

    def _build_with_diag():
        model = spec.cond_factory(args.location, base_fc_seed, pretrained, args.freeze_backbone)
        if args.diagnostics:
            if args.dataset not in SINGLE_ATTR_CARDINALITY:
                raise ValueError(f"--diagnostics 仅支持单属性数据集 {sorted(SINGLE_ATTR_CARDINALITY)}，"
                                 f"不支持 {args.dataset}（多属性，ρ 探针无法枚举取值）")
            diag_path = output_dir / "diag_logs" / f"{run_id(method, hparam.tag, seed)}.jsonl"
            diag_holder["d"] = CondNetDiagnostics(
                model, num_attr=SINGLE_ATTR_CARDINALITY[args.dataset], diag_path=diag_path)
            print(f"[E1③ 诊断探针 ON] conv ρ/梯度/权重 → {diag_path}")
        return model

    on_after_backward = (lambda m: diag_holder["d"].on_after_backward(m)) if args.diagnostics else None
    epoch_diag_fn = (lambda ep, er: diag_holder["d"].epoch_record(ep, er)) if args.diagnostics else None

    run_training(
        dataset=spec.name, method=method, output_dir=output_dir, cfg=cfg, hparam=hparam, seed=seed,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        build_model=_build_with_diag,
        unpack_fn=spec.unpack_fn, forward_fn=forward_fn, fairness_report_fn=spec.fairness_fn,
        on_after_backward=on_after_backward, epoch_diag_fn=epoch_diag_fn,
    )


if __name__ == "__main__":
    main()
