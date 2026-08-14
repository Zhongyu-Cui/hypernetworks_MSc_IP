"""
估计各数据集的 Conditional V-information  I_V(A → Y | φ(X))
=================================================================================================
「属性在图像之上还带多少可用标签信息」的闸门 —— 完全基于已确立方法，不含自创量。

引用（三篇，闸门的全部理论依据）：
  - V-information（可用信息）：Xu, Zhao, Song, Stewart, Ermon, ICLR 2020, arXiv:2002.10689。
      用一个函数族 V 重定义信息：V-entropy  H_V(Y|R) = inf_{f∈V} E[−log f[R](Y)]，
      V-information  I_V(R→Y) = H_V(Y) − H_V(Y|R)。V=全函数时退回香农互信息。
  - Conditional probing / conditional V-information：Hewitt, Ethayarajh, Liang, Manning,
      EMNLP 2021, aclanthology 2021.emnlp-main.122，Eq.(2)：给定 baseline B，训两个 probe
      （输入 [B;φ] 与 [B;0]），留出集负交叉熵相减，估计
          I_V(φ→Y|B) = H_V(Y|B) − H_V(Y|B,φ)。
  - Control task（探针功效对照）：Hewitt & Liang, EMNLP 2019, aclanthology D19-1275。

判据（HN 有用的**必要条件**，模型无关）：
  比较基准是 attribute-blind 池化 baseline。HN 要赢它，必要前提是「给定图像 X 后属性 A 仍带
  可用标签信息」，即 conditional V-information  I_V(A→Y|φ(X)) > 0。若 ≤ 0，池化已用尽 V 能用
  的信息，任何属性条件架构（全部 HN 变体）顶多退化成它、不可能超过。本脚本全程不含 HN，只在
  冻结 baseline 特征 φ(X) 上测这个量，把「信号弱」与「机制没学到」切开。

本脚本的角色映射（把 Hewitt Eq.2 的 baseline 换成图像、被研究表征换成属性）：
      baseline B = φ(X)（冻结图像特征）,  被研究表征 = 属性 A
      I_V(A→Y|φ(X)) = H_V(Y|φ(X)) − H_V(Y|φ(X),A)
                    = [留出集 NLL of  Y~φ]  −  [留出集 NLL of  Y~[φ,A]]

probe 族 V（Hewitt 明确要求「因 theory-external 理由选」，此处选来匹配 HN 容量）：
      V_int : 逐组条件线性（每组独立 [1,φ] 块设计）—— HN 生成逐组参数的直接对应，
              **主口径**（对共享 backbone 的 HN 是可用增益的紧上界）。
      V_add : concat 线性 [φ, onehot(A)] —— 只吃逐组截距 = 患病率/校准位移，**次口径**（分解用）。
两者都是合法的 conditional V-information，仅声明的 V 不同。

估计与无泄漏：probe 在 train 上拟合（凸 LBFGS，逼近 Eq.3 的 inf），L2 在 val 上按 NLL 选、
温度在 val 上标定，V-entropy 在 **留出集 test** 上评估。留出交叉熵之差只会**低估** A 的贡献
（含 A 的 probe 总可忽略 A 退化成 baseline），故 ≈0 是保守读数。结果一律以 **bits** 汇报（nats/ln2）。

OOF 口径（对齐 docs/oof_regime_results.md 与 scripts/cv_oof_report.py）：
  用 cv5 的 5 折 image-only baseline（erm 选定 config，seed=42+fold，见 selected_configs.json）。
  对每折 f：用 M_f 抽 train_f/val_f/test_f 冻结特征 → train_f 拟合 probe、val_f 选/标定
  → test_f 逐样本 NLL。5 折 test 逐样本 NLL 池化（每样本恰被其留出折预测一次 = OOF）
  → 全数据集 conditional V-info + 样本级 bootstrap CI。每样本的留出读数都来自一个从未见过它
  的折的 probe（与 OOF 报告同为三重无泄漏）。

Control task（Hewitt & Liang 2019，兜住唯一假阴风险）：
  Y' = Y ⊕ r，r~Bernoulli(0.5) 与图像无关。以 A_syn=r 在 V_int 下算 conditional V-info，
  它**必须**显著为正（probe 族对强条件信号有功效）；否则本次真实读数不可信、作废。

运行：特征抽取是 GPU 前向（每折抽 train/val/test，共 5×|数据集|），按项目规范用 sbatch 提交，
见 slurm/estimate_conditional_v_information.sh；probe 拟合在缓存特征上跑，随作业一并完成。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from src.models.resnet18_pretrained import ResNet18Pretrained

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs")

# 大数据集（MIMIC/CheXpert ~12–17 万/折）多 worker 抽特征时，默认 file_descriptor 共享策略会
# 耗尽文件句柄（RuntimeError: Too many open files）。改用 file_system 策略（错误信息推荐做法）。
torch.multiprocessing.set_sharing_strategy("file_system")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS = 5                      # cv5：seed=42+fold
FEATURE_DIM = 512               # resnet18 avgpool 输出维度
L2_GRID = (0.1, 1.0, 10.0, 100.0)  # 在 val 上按 NLL 选 L2，让 V-entropy 的 inf 逼得尽量紧
LN2 = float(np.log(2.0))
EPS = 1e-12


# ============================================================
# 数据集注册表 —— 每个数据集的 Dataset / 属性 / checkpoint / split 差异集中于此
# ============================================================
@dataclass
class DatasetSpec:
    """描述一个数据集接入 conditional-V-info 闸门所需的全部信息。"""
    name: str                                  # 目录名（= outputs 下子目录）
    out_dir: Path                              # 结果输出目录
    ckpt_dir: Path                             # cv5 erm baseline checkpoint 所在目录
    split_dir: Path                            # cv5 splits 根（其下 fold{f}/{train,val,test}.csv）
    config_yaml: Path                          # 训练配置（取 transforms/dataloader）
    modality: str                              # 'cxr'(灰度 16-bit) 或 'rgb'
    build_dataset: Callable                    # (csv_path, transform, cfg) -> torch Dataset
    unpack_batch: Callable                     # (batch) -> (images, dict[str, LongTensor])
    attr_keys: tuple[str, ...]                 # 缓存里保存的属性键
    attr_specs: tuple[str, ...]                # 要报告的属性规格
    group_fn: Callable                         # (data_dict, spec) -> (gid[N], n_groups)
    filter_fn: Callable                        # (data_dict, spec) -> bool mask[N]（哪些样本进入该 spec）


# ---------- 各数据集的 Dataset 构造与 batch 解包 ----------
def _cxr_dataset(csv_path: Path, transform, cfg: dict):
    """MIMIC / CheXpert：16-bit 灰度 PNG，MIMICCXRDataset 内部做 min-max 归一化。"""
    from src.datasets.mimic_cxr_dataset import MIMICCXRDataset
    return MIMICCXRDataset(
        csv_path, transform=transform,
        image_size=cfg["data"]["image_size"],
        age_threshold=cfg["attributes"]["age"]["age_threshold"],
    )


def _cxr_unpack(batch):
    """(image, label, sex, race, age_group) -> (image, {label,sex,race,age})。"""
    images, label, sex, race, age = batch
    return images, {"label": label, "sex": sex, "race": race, "age": age}


def _ham_dataset(csv_path: Path, transform, cfg: dict):
    from src.datasets.ham10000_dataset import HAM10000Dataset
    return HAM10000Dataset(csv_path, transform=transform)


def _ham_unpack(batch):
    """(image, label, sex, age_group) -> (image, {label,sex,age_group})。"""
    images, label, sex, age_group = batch
    return images, {"label": label, "sex": sex, "age_group": age_group}


def _fitz_dataset(csv_path: Path, transform, cfg: dict):
    from src.datasets.fitzpatrick_dataset import FitzpatrickDataset
    return FitzpatrickDataset(csv_path, transform=transform)


def _fitz_unpack(batch):
    """(image, label, skin) -> (image, {label,skin})。"""
    images, label, skin = batch
    return images, {"label": label, "skin": skin}


def _papila_dataset(csv_path: Path, transform, cfg: dict):
    from src.datasets.papila_dataset import PAPILADataset
    return PAPILADataset(csv_path, transform=transform, cache=False)


def _papila_unpack(batch):
    """(image, label, sex, age_group) -> (image, {label,sex,age_group})。"""
    images, label, sex, age_group = batch
    return images, {"label": label, "sex": sex, "age_group": age_group}


# ---------- 各数据集的分组编码与样本过滤 ----------
def _cxr_group(d: dict[str, np.ndarray], spec: str) -> tuple[np.ndarray, int]:
    """MIMIC/CheXpert：sex/race/age 各 2 组，joint = sex*4+race*2+age（8 组）。"""
    if spec == "joint":
        return (d["sex"] * 4 + d["race"] * 2 + d["age"]).astype(np.int64), 8
    return d[spec].astype(np.int64), 2


def _cxr_filter(d: dict[str, np.ndarray], spec: str) -> np.ndarray:
    return np.ones(d["label"].shape[0], dtype=bool)   # 无排除组


def _ham_group(d: dict[str, np.ndarray], spec: str) -> tuple[np.ndarray, int]:
    """HAM：sex(2, 全量)、age(4, 过滤 age_group>=0)、joint=sex*4+age(8, 过滤)。"""
    if spec == "sex":
        return d["sex"].astype(np.int64), 2
    if spec == "age":
        return d["age_group"].astype(np.int64), 4
    if spec == "joint":
        return (d["sex"] * 4 + d["age_group"]).astype(np.int64), 8
    raise ValueError(spec)


def _ham_filter(d: dict[str, np.ndarray], spec: str) -> np.ndarray:
    """age/joint 需排除 age_group==-1（0-20 组，embedding/编码不能吃 -1）。"""
    if spec in ("age", "joint"):
        return d["age_group"] >= 0
    return np.ones(d["label"].shape[0], dtype=bool)


def _fitz_group(d: dict[str, np.ndarray], spec: str) -> tuple[np.ndarray, int]:
    """Fitzpatrick：skin(6 型)、skin_bin(浅 I–III=0 / 深 IV–VI=1)。"""
    if spec == "skin":
        return d["skin"].astype(np.int64), 6
    if spec == "skin_bin":
        return (d["skin"] >= 3).astype(np.int64), 2
    raise ValueError(spec)


def _fitz_filter(d: dict[str, np.ndarray], spec: str) -> np.ndarray:
    return np.ones(d["label"].shape[0], dtype=bool)   # -1 已在 split 阶段排除


def _papila_group(d: dict[str, np.ndarray], spec: str) -> tuple[np.ndarray, int]:
    """PAPILA：sex(2)、age(2, <=60/>60)、joint=sex*2+age(4)。"""
    if spec == "age":
        return d["age_group"].astype(np.int64), 2
    if spec == "joint":
        return (d["sex"] * 2 + d["age_group"]).astype(np.int64), 4
    return d["sex"].astype(np.int64), 2


def _papila_filter(d: dict[str, np.ndarray], spec: str) -> np.ndarray:
    return np.ones(d["label"].shape[0], dtype=bool)


def build_registry() -> dict[str, DatasetSpec]:
    """构建 5 数据集的注册表。checkpoint 命名统一为 erm_<config>_seed<42+fold>_best_overall.pth。"""
    reg: dict[str, DatasetSpec] = {}
    reg["mimic_cxr"] = DatasetSpec(
        name="mimic_cxr", out_dir=OUTPUTS_ROOT / "mimic_cxr",
        ckpt_dir=OUTPUTS_ROOT / "mimic_cxr" / "cv5",
        split_dir=REPO_ROOT / "data/splits/mimic_cxr_nofinding" / "cv5",
        config_yaml=REPO_ROOT / "configs/mimic_cxr_baseline.yaml", modality="cxr",
        build_dataset=_cxr_dataset, unpack_batch=_cxr_unpack,
        attr_keys=("sex", "race", "age"), attr_specs=("sex", "race", "age", "joint"),
        group_fn=_cxr_group, filter_fn=_cxr_filter)
    reg["chexpert_cxr"] = DatasetSpec(
        name="chexpert_cxr", out_dir=OUTPUTS_ROOT / "chexpert_cxr",
        ckpt_dir=OUTPUTS_ROOT / "chexpert_cxr" / "cv5",
        split_dir=REPO_ROOT / "data/splits/chexpert_nofinding" / "cv5",
        config_yaml=REPO_ROOT / "configs/chexpert_baseline.yaml", modality="cxr",
        build_dataset=_cxr_dataset, unpack_batch=_cxr_unpack,
        attr_keys=("sex", "race", "age"), attr_specs=("sex", "race", "age", "joint"),
        group_fn=_cxr_group, filter_fn=_cxr_filter)
    reg["ham10000"] = DatasetSpec(
        name="ham10000", out_dir=OUTPUTS_ROOT / "ham10000",
        ckpt_dir=OUTPUTS_ROOT / "ham10000" / "cv5",
        split_dir=REPO_ROOT / "data/splits/ham10000" / "cv5",
        config_yaml=REPO_ROOT / "configs/ham10000_baseline.yaml", modality="rgb",
        build_dataset=_ham_dataset, unpack_batch=_ham_unpack,
        attr_keys=("sex", "age_group"), attr_specs=("sex", "age", "joint"),
        group_fn=_ham_group, filter_fn=_ham_filter)
    reg["fitzpatrick"] = DatasetSpec(
        name="fitzpatrick", out_dir=OUTPUTS_ROOT / "fitzpatrick",
        ckpt_dir=OUTPUTS_ROOT / "fitzpatrick" / "cv5",
        split_dir=REPO_ROOT / "data/splits/fitzpatrick17k" / "cv5",
        config_yaml=REPO_ROOT / "configs/fitzpatrick_baseline.yaml", modality="rgb",
        build_dataset=_fitz_dataset, unpack_batch=_fitz_unpack,
        attr_keys=("skin",), attr_specs=("skin", "skin_bin"),
        group_fn=_fitz_group, filter_fn=_fitz_filter)
    reg["papila"] = DatasetSpec(
        name="papila", out_dir=OUTPUTS_ROOT / "papila",
        ckpt_dir=OUTPUTS_ROOT / "papila",                    # PAPILA baseline 在扁平目录
        split_dir=REPO_ROOT / "data/splits/papila" / "cv5",
        config_yaml=REPO_ROOT / "configs/papila_baseline.yaml", modality="rgb",
        build_dataset=_papila_dataset, unpack_batch=_papila_unpack,
        attr_keys=("sex", "age_group"), attr_specs=("sex", "age", "joint"),
        group_fn=_papila_group, filter_fn=_papila_filter)
    return reg


def load_erm_config(spec: DatasetSpec) -> str:
    """读 selected_configs.json 的 erm 条目（cv5 选定的 baseline 超参 config_tag）。"""
    for cand in (spec.ckpt_dir / "selected_configs.json", spec.out_dir / "selected_configs.json"):
        if cand.exists():
            return json.load(open(cand))["config"]["erm"]
    raise FileNotFoundError(f"找不到 {spec.name} 的 selected_configs.json")


# ============================================================
# Stage A —— 冻结特征抽取（唯一需要 GPU 的部分）
# ============================================================
def build_eval_transform(spec: DatasetSpec, cfg: dict) -> transforms.Compose:
    """按数据集口径构建 eval（无增强、确定性）transform，与训练时的评估口径一致。"""
    norm = cfg["transforms"]["normalize"]
    eval_cfg = cfg["transforms"].get("eval", {})
    steps: list = []
    if spec.modality == "cxr":
        # CXR：Dataset 已把 16-bit min-max 成干净 8-bit 单通道；这里灰度转 3 通道 + Resize
        steps.append(transforms.Grayscale(num_output_channels=3))
        steps.append(transforms.Resize(eval_cfg["resize"]))
    else:
        # RGB：自然图。若配置给了 resize/center_crop 则用（HAM/PAPILA），否则 224 原生（Fitz）
        if eval_cfg.get("resize") is not None:
            steps.append(transforms.Resize(tuple(eval_cfg["resize"])))
        if eval_cfg.get("center_crop") is not None:
            steps.append(transforms.CenterCrop(eval_cfg["center_crop"]))
    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(mean=norm["mean"], std=norm["std"]))
    return transforms.Compose(steps)


def build_loader(spec: DatasetSpec, fold: int, split: str, cfg: dict) -> DataLoader:
    """为某折某 split 构建确定性 eval DataLoader（train 也用 eval transform 以便特征可复现）。"""
    csv_path = spec.split_dir / f"fold{fold}" / f"{split}.csv"
    dataset = spec.build_dataset(csv_path, build_eval_transform(spec, cfg), cfg)
    return DataLoader(
        dataset, batch_size=cfg["training"]["batch_size"], shuffle=False,
        num_workers=cfg["dataloader"]["num_workers"], pin_memory=cfg["dataloader"]["pin_memory"],
    )


def checkpoint_path(spec: DatasetSpec, config_tag: str, fold: int) -> Path:
    """cv5 baseline checkpoint 路径：seed=42+fold，best_overall（Overall 早停口径）。"""
    return spec.ckpt_dir / f"erm_{config_tag}_seed{42 + fold}_best_overall.pth"


@torch.no_grad()
def extract_split_features(
    spec: DatasetSpec, config_tag: str, fold: int, split: str, cfg: dict,
) -> dict[str, np.ndarray]:
    """
    用第 fold 折的 image-only baseline checkpoint，在某 split 上抽 512 维冻结特征 + logit + 属性。

    Returns:
        dict：feat[float16,(N,512)] / logit[float32,(N,)] / label + 各属性[int64,(N,)]。
    """
    ckpt = checkpoint_path(spec, config_tag, fold)
    if not ckpt.exists():
        raise FileNotFoundError(f"找不到 baseline checkpoint: {ckpt}")

    model = ResNet18Pretrained(num_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()

    captured: dict[str, torch.Tensor] = {}

    def _hook(_m: nn.Module, _i: tuple, out: torch.Tensor) -> None:
        captured["feat"] = out.flatten(1)               # [B,512,1,1] -> [B,512]

    handle = model.backbone.avgpool.register_forward_hook(_hook)

    loader = build_loader(spec, fold, split, cfg)
    feats, logits = [], []
    attr_buffers: dict[str, list] = {k: [] for k in ("label", *spec.attr_keys)}
    for batch in loader:
        images, attrs = spec.unpack_batch(batch)
        images = images.to(DEVICE, non_blocking=True)
        logit = model(images).squeeze(1)
        feats.append(captured["feat"].half().cpu().numpy())
        logits.append(logit.float().cpu().numpy())
        for k in attr_buffers:
            attr_buffers[k].append(attrs[k].numpy())
    handle.remove()

    out = {"feat": np.concatenate(feats).astype(np.float16),
           "logit": np.concatenate(logits).astype(np.float32)}
    for k, chunks in attr_buffers.items():
        out[k] = np.concatenate(chunks).astype(np.int64)
    return out


def cache_path(spec: DatasetSpec, fold: int, split: str) -> Path:
    """某 (fold, split) 特征缓存的 npz 路径。"""
    return spec.out_dir / "cvi_cache" / f"feat_fold{fold}_{split}.npz"


def ensure_features(spec: DatasetSpec, config_tag: str, cfg: dict) -> None:
    """对所有 (fold, split) 抽取并缓存冻结特征（已存在则跳过）。"""
    (spec.out_dir / "cvi_cache").mkdir(parents=True, exist_ok=True)
    for fold in range(N_FOLDS):
        for split in ("train", "val", "test"):
            path = cache_path(spec, fold, split)
            if path.exists():
                print(f"[extract] 缓存已存在，跳过: {path.name}")
                continue
            print(f"[extract] fold={fold} split={split} ...", flush=True)
            data = extract_split_features(spec, config_tag, fold, split, cfg)
            np.savez_compressed(path, **data)
            print(f"[extract]   -> {path.name}  N={len(data['label']):,}")


def load_fold_features(spec: DatasetSpec, fold: int) -> dict[str, dict[str, np.ndarray]]:
    """载入某折 train/val/test 特征缓存，返回 {split: {key: array}}。"""
    out = {}
    for split in ("train", "val", "test"):
        with np.load(cache_path(spec, fold, split)) as npz:
            out[split] = {k: npz[k] for k in npz.files}
    return out


# ============================================================
# 浅 probe：凸 logistic（LBFGS）+ 温度标定 —— 逼近 V-entropy 的 inf
# （与 estimate_conditional_mi.py 完全一致的估计器内核，已验证正确）
# ============================================================
def _fit_logistic(
    x_train: torch.Tensor, y_train: torch.Tensor,
    x_val: torch.Tensor, y_val: torch.Tensor, l2: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """全批 LBFGS 拟合 logistic（凸，收敛干净）+ val 温度标定，返回 (weight, bias, temperature)。"""
    dim = x_train.shape[1]
    linear = nn.Linear(dim, 1).to(DEVICE)
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS(linear.parameters(), lr=1.0, max_iter=200,
                                  history_size=20, line_search_fn="strong_wolfe")

    def _closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = bce(linear(x_train).squeeze(1), y_train) + l2 * linear.weight.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(_closure)
    with torch.no_grad():
        weight = linear.weight.detach().squeeze(0).clone()
        bias = linear.bias.detach().clone()
        val_logit = x_val @ weight + bias

    log_t = torch.zeros(1, device=DEVICE, requires_grad=True)
    temp_opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

    def _temp_closure() -> torch.Tensor:
        temp_opt.zero_grad()
        loss = bce(val_logit / torch.exp(log_t), y_val)
        loss.backward()
        return loss

    temp_opt.step(_temp_closure)
    return weight, bias, float(torch.exp(log_t.detach()))


def _per_sample_nll(logit: np.ndarray, y: np.ndarray) -> np.ndarray:
    """逐样本 BCE（nats）。logit 为温度标定后 logit，y 为 0/1。"""
    return np.logaddexp(0.0, logit) - y * logit


def _onehot(group_id: np.ndarray, n_groups: int) -> np.ndarray:
    """整数组 id -> one-hot [N, n_groups]（float32）。"""
    oh = np.zeros((group_id.shape[0], n_groups), dtype=np.float32)
    oh[np.arange(group_id.shape[0]), group_id] = 1.0
    return oh


def _build_design(
    feat: np.ndarray, group_id: Optional[np.ndarray], n_groups: int, mode: str,
) -> np.ndarray:
    """
    构建 probe 输入设计矩阵。

    mode:
      'base'  = [φ]                         —— H_V(Y|φ) 的输入（baseline probe）。
      'add'   = [φ, onehot(组)去首列]        —— V_add：concat 线性，逐组截距（校准位移）。
      'int'   = 逐组独立 [1, φ] 块设计        —— V_int：逐组条件线性（逐组不同决策函数）。
    """
    feat = feat.astype(np.float32)
    n = feat.shape[0]
    if mode == "base":
        return feat
    if mode == "add":
        return np.concatenate([feat, _onehot(group_id, n_groups)[:, 1:]], axis=1)
    if mode == "int":
        block = np.zeros((n, n_groups * FEATURE_DIM), dtype=np.float32)
        rows = np.arange(n)
        for g in range(n_groups):
            mask = group_id == g
            block[np.ix_(rows[mask], np.arange(g * FEATURE_DIM, (g + 1) * FEATURE_DIM))] = feat[mask]
        return np.concatenate([block, _onehot(group_id, n_groups)], axis=1)
    raise ValueError(f"未知 mode: {mode}")


def _eval_test_logit(
    design_train: np.ndarray, y_train: np.ndarray,
    design_val: np.ndarray, y_val: np.ndarray, design_test: np.ndarray,
) -> np.ndarray:
    """在 L2 网格上拟合 + val 温度标定，按 val NLL 选最优，返回 test 上校准后 logit。"""
    xtr = torch.from_numpy(design_train).to(DEVICE)
    ytr = torch.from_numpy(y_train.astype(np.float32)).to(DEVICE)
    xva = torch.from_numpy(design_val).to(DEVICE)
    yva = torch.from_numpy(y_val.astype(np.float32)).to(DEVICE)
    xte = torch.from_numpy(design_test).to(DEVICE)

    best_val_nll, best_test_logit = float("inf"), None
    for l2 in L2_GRID:
        weight, bias, temperature = _fit_logistic(xtr, ytr, xva, yva, l2)
        with torch.no_grad():
            val_logit = (xva @ weight + bias) / temperature
            val_nll = float(nn.functional.binary_cross_entropy_with_logits(val_logit, yva))
            if val_nll < best_val_nll:
                best_val_nll = val_nll
                best_test_logit = ((xte @ weight + bias) / temperature).cpu().numpy()
    return best_test_logit.astype(np.float64)


# ============================================================
# OOF conditional V-information：逐折算逐样本 llr，池化成全数据集读数
# ============================================================
def _fold_llr(
    fold_feats: dict[str, dict[str, np.ndarray]], spec: DatasetSpec,
    attr_spec: str, mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    单折内：拟合 baseline probe(Y~φ) 与条件 probe(Y~[φ,A])，返回 test 上逐样本 llr 与组 id。

    llr = NLL_base − NLL_cond（nats，逐样本）；均值 = 该折 conditional V-info（未池化）。
    对含排除组的 spec（HAM age/joint），三 split 均先按 filter_fn 过滤。
    """
    tr, va, te = fold_feats["train"], fold_feats["val"], fold_feats["test"]

    def _view(d: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = spec.filter_fn(d, attr_spec)
        gid, n_g = spec.group_fn({k: v[mask] for k, v in d.items()}, attr_spec)
        return d["feat"][mask], d["label"][mask], gid, n_g

    ftr, ytr, gtr, n_g = _view(tr)
    fva, yva, gva, _ = _view(va)
    fte, yte, gte, _ = _view(te)

    logit_base = _eval_test_logit(
        _build_design(ftr, None, 0, "base"), ytr,
        _build_design(fva, None, 0, "base"), yva,
        _build_design(fte, None, 0, "base"))
    logit_cond = _eval_test_logit(
        _build_design(ftr, gtr, n_g, mode), ytr,
        _build_design(fva, gva, n_g, mode), yva,
        _build_design(fte, gte, n_g, mode))

    llr = _per_sample_nll(logit_base, yte.astype(np.float64)) - \
        _per_sample_nll(logit_cond, yte.astype(np.float64))
    return llr, gte


def _summarize_oof(
    llr: np.ndarray, gid: np.ndarray, n_bootstrap: int, rng: np.random.Generator,
) -> dict:
    """把池化 OOF 逐样本 llr 汇总为 conditional V-info（bits）+ bootstrap CI + 逐组 + 少数群加权。"""
    global_nats = float(llr.mean())
    boot = np.array([llr[rng.integers(0, llr.size, llr.size)].mean() for _ in range(n_bootstrap)])
    ci_nats = np.percentile(boot, [2.5, 97.5])

    per_group, group_means = {}, []
    for g in np.unique(gid):
        m = gid == g
        gm = float(llr[m].mean())
        per_group[str(int(g))] = {"cvi_bits": gm / LN2, "n": int(m.sum())}
        group_means.append(gm)
    minority_weighted = float(np.mean(group_means)) if group_means else float("nan")

    return {
        "cvi_bits": global_nats / LN2,
        "cvi_nats": global_nats,
        "ci95_bits": [float(ci_nats[0] / LN2), float(ci_nats[1] / LN2)],
        "detected": bool(ci_nats[0] > 0),           # CI 下界 >0 ⟹ 有可用信号
        "per_group": per_group,
        "minority_weighted_bits": minority_weighted / LN2,
        "n_total": int(llr.size),
    }


def conditional_v_information_oof(
    spec: DatasetSpec, attr_spec: str, all_feats: list[dict],
    n_bootstrap: int, rng: np.random.Generator,
) -> dict:
    """
    某属性规格的 OOF conditional V-information：逐折算 llr → 5 折 test 池化 → 汇总（V_add + V_int）。
    """
    result: dict = {"attr_spec": attr_spec}
    for mode in ("add", "int"):
        llrs, gids = [], []
        for fold in range(N_FOLDS):
            llr, gid = _fold_llr(all_feats[fold], spec, attr_spec, mode)
            llrs.append(llr)
            gids.append(gid)
        pooled_llr = np.concatenate(llrs)
        pooled_gid = np.concatenate(gids)
        key = "V_int" if mode == "int" else "V_add"
        result[key] = _summarize_oof(pooled_llr, pooled_gid, n_bootstrap, rng)
    return result


# ============================================================
# Control task（Hewitt & Liang 2019）：Y'=Y⊕r，探针功效对照
# ============================================================
def control_task_oof(
    spec: DatasetSpec, all_feats: list[dict], n_bootstrap: int, seed: int = 12345,
) -> dict:
    """
    正控制：r~Bernoulli(0.5) 与图像无关，Y'=Y⊕r，A_syn=r（2 组）在 V_int 下算 conditional V-info。
    r 完全决定 label 翻转且独立于图像 ⟹ [φ,r] 交互能大幅优于单独 φ ⟹ 应读出大正值。
    若不显著为正，说明 probe 族对强条件信号都无功效，本次真实读数不可信。
    """
    rng = np.random.default_rng(seed)
    llrs, gids = [], []
    for fold in range(N_FOLDS):
        ff = all_feats[fold]
        syn = {}
        for split in ("train", "val", "test"):
            d = ff[split]
            n = d["label"].shape[0]
            r = rng.integers(0, 2, n).astype(np.int64)
            syn[split] = {"feat": d["feat"], "label": (d["label"] ^ r).astype(np.int64),
                          "_r": r}
        # 用临时 spec 语义：组 = r
        tr, va, te = syn["train"], syn["val"], syn["test"]
        logit_base = _eval_test_logit(
            _build_design(tr["feat"], None, 0, "base"), tr["label"],
            _build_design(va["feat"], None, 0, "base"), va["label"],
            _build_design(te["feat"], None, 0, "base"))
        logit_cond = _eval_test_logit(
            _build_design(tr["feat"], tr["_r"], 2, "int"), tr["label"],
            _build_design(va["feat"], va["_r"], 2, "int"), va["label"],
            _build_design(te["feat"], te["_r"], 2, "int"))
        llr = _per_sample_nll(logit_base, te["label"].astype(np.float64)) - \
            _per_sample_nll(logit_cond, te["label"].astype(np.float64))
        llrs.append(llr)
        gids.append(te["_r"])
    pooled_llr = np.concatenate(llrs)
    pooled_gid = np.concatenate(gids)
    return _summarize_oof(pooled_llr, pooled_gid, n_bootstrap, np.random.default_rng(seed))


# ============================================================
# 编排
# ============================================================
def main() -> None:
    """抽取 cv5 baseline 特征（缺则补）→ 逐属性算 OOF conditional V-info → control task → 落 JSON。"""
    parser = argparse.ArgumentParser(description="OOF conditional V-information 闸门")
    parser.add_argument("--dataset", required=True,
                        choices=["mimic_cxr", "chexpert_cxr", "ham10000", "fitzpatrick", "papila"])
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()

    spec = build_registry()[args.dataset]
    config_tag = load_erm_config(spec)
    print(f"Device: {DEVICE} | dataset={spec.name} | baseline erm config={config_tag} | folds={N_FOLDS}")

    with open(spec.config_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ---------- Stage A：特征抽取（GPU）----------
    ensure_features(spec, config_tag, cfg)

    # ---------- Stage B：OOF conditional V-info（CPU/GPU 浅 probe）----------
    all_feats = [load_fold_features(spec, fold) for fold in range(N_FOLDS)]
    rng = np.random.default_rng(0)

    headline = {}
    for attr_spec in spec.attr_specs:
        res = conditional_v_information_oof(spec, attr_spec, all_feats, args.n_bootstrap, rng)
        headline[attr_spec] = res
        vi, va = res["V_int"], res["V_add"]
        print(f"[CVI] {attr_spec:6s} V_int={vi['cvi_bits']:+.5f} bits "
              f"CI[{vi['ci95_bits'][0]:+.5f},{vi['ci95_bits'][1]:+.5f}] detected={vi['detected']} "
              f"| minority-w={vi['minority_weighted_bits']:+.5f} | V_add={va['cvi_bits']:+.5f}")

    # ---------- Control task ----------
    ctrl = control_task_oof(spec, all_feats, args.n_bootstrap)
    print(f"[CTRL] Y⊕r 正控制：V_int={ctrl['cvi_bits']:+.5f} bits "
          f"CI[{ctrl['ci95_bits'][0]:+.5f},{ctrl['ci95_bits'][1]:+.5f}] detected={ctrl['detected']} "
          f"（应为大正值；否则读数不可信）")

    out = {
        "dataset": spec.name,
        "method": "conditional V-information (Hewitt et al. 2021, EMNLP; Xu et al. 2020, ICLR)",
        "regime": "cv5 OOF (5 folds pooled), baseline=erm image-only",
        "baseline_config": config_tag,
        "unit": "bits",
        "primary": "V_int (group-conditional linear probe family)",
        "conditional_v_information": headline,
        "control_task_Y_xor_r": ctrl,
    }
    out_path = spec.out_dir / "conditional_v_information.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_path}")


if __name__ == "__main__":
    main()
