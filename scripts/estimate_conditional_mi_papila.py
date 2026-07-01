"""
估计 PAPILA 青光眼上的条件互信息 I(Y;A|X)，模型无关地检验「sex/age 信号是否弱」
================================================================================================

本脚本是 HAM10000 版 scripts/estimate_conditional_mi_ham10000.py 的 PAPILA 平移：判据、四层
设计 (E0–E3)、估计器、决策规则全部保留。**关键适配 = 小样本 OOF 池化**：PAPILA 仅 420 眼-图、
单折 test≈84，直接在单 test 上估计功效极差（见 docs/papila_preprocessing_plan.md §6 的 E3 硬约束
警告）。因此本脚本在 **5 折患者级 GroupKFold** 上逐折拟合，并把每折 test 的逐样本量（ΔNLL、
image-only logit、解码概率）**池化为 OOF（out-of-fold）全集**——每张图恰被其留出折评估一次，
合并后所有 E0–E3 统计都在完整 420 上进行，最大化本数据规模下的统计功效。

判据（见项目记忆 mimic-fairness-signal-reframing / 仓库 CLAUDE.md「诊断方法重构」节）：
    HN 要赢 attribute-blind 池化基线，必要条件是 I(Y;A|X) > 0（给定图像后，子群仍需不同
    决策函数）。若该量 = 0，盲池化已是 Bayes 最优，任何条件化架构（含全部 HN 变体）顶多
    退化成它、不可能超过。本脚本全程不含 HN，只估计这个量本身。

任务/属性：
    Y = 青光眼（排除 suspect，阳性率 ~20.7%）
    A = sex（Male=0/Female=1，2 组）、age_group（<=60=0/>60=1，**2 组**）、
        joint = sex*2+age_group（4 组）；backbone = 5 折 image-only
        resnet18_glaucoma_cv5fold{k}_seed42_best_overall.pth

与 HAM 模板的差异：
    ① 折级 OOF 池化取代「单 split × 多 seed」：循环 5 折、每折用自己的 train/val 拟合浅头、
       预测自己的 test，再把 test 级逐样本量按折拼接成 420 的 OOF 全集。
    ② age_group 为**二值**（无 -1 排除组）：N_AGE=2, N_JOINT=4；无需 filter_valid_age。
       E2 中 age 与 sex 同走 2×2 CMH（含方向 MH-OR），仅 joint(4) 走方向无关多类 χ²。
    ③ E3 灵敏度在 OOF 全集上做（pooled 420），更贴近 5 折合并的实际操作规模。

四层互补设计（每层偏差方向不同，三角验证）：
    E0  边际量标定（不当判据，只解释）：I(Y;A) 边际 MI；I(A;X) 可解码度。
    E1  主估计器：冻结特征 φ(X) 上的条件似然比 ΔNLL（= I(Y;A|X) 估计，OOF 池化）。
    E2  无参数对照：image-only logit ŝ 分层，箱内查 A 与 Y 的条件关联（OOF 池化）。
    E3  灵敏度标定：正控制 Y'=Y XOR r（必须读出大 ΔNLL，否则探针失灵）+ 剂量-反应功效曲线
        （标定本数据规模下的最小可检测效应 MDE）。

决策规则（对接 reframing 三分支）：
    OOF 全局 ΔNLL CI 上界≈0 且 E2 各箱无显著、E3 证明探针有功效
        → 「信号弱」成立，HN 注定无收益。
    全局>0（尤其 age 轴）→ 信号在、HN 没用上 → 机制/架构有可争取空间。
    E3 正控制不达功效 → 判「样本规模下不可判定」，不强行下结论。

运行方式：本机有 GPU 即可直接跑（特征抽取 = 5 折各 420 次前向，极轻量）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.papila_dataset import PAPILADataset
from src.models.resnet18_pretrained import ResNet18Pretrained

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "papila_baseline.yaml"
OUTPUT_DIR = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/papila")
CACHE_DIR = OUTPUT_DIR / "cmi_cache"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS = 5
SEED = 42                       # baseline checkpoint 的 seed
SPLITS = ("train", "val", "test")
FEATURE_DIM = 512               # resnet18 avgpool 输出维度
N_AGE = 2                       # age_group 组数（<=60 / >60）
N_JOINT = 4                     # sex(2) × age(2)

SEX_NAMES = {0: "Male", 1: "Female"}
AGE_NAMES = {0: "<=60", 1: ">60"}
JOINT_NAMES = {s * 2 + a: f"{SEX_NAMES[s][0]}&{AGE_NAMES[a]}" for s in (0, 1) for a in range(2)}
EPS = 1e-12


# ============================================================
# Stage A —— 特征抽取（唯一需要 GPU 的部分，逐折）
# ============================================================
def build_loader(fold: int, split: str, cfg: dict) -> DataLoader:
    """为某折某 split 构建 eval（无增强、确定性）DataLoader，用于抽取冻结特征。"""
    t_cfg = cfg["transforms"]
    # train split 也用 eval transform（无翻转/旋转），保证特征对每张图确定可缓存复现。
    # 源图已是 256 缓存，Resize((256,256))→CenterCrop(224) 与训练评估口径一致。
    eval_transform = transforms.Compose([
        transforms.Resize(tuple(t_cfg["eval"]["resize"])),
        transforms.CenterCrop(t_cfg["eval"]["center_crop"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=t_cfg["normalize"]["mean"], std=t_cfg["normalize"]["std"]),
    ])
    split_dir = REPO_ROOT / cfg["data"]["split_dir"] / f"cv{N_FOLDS}" / f"fold{fold}"
    dataset = PAPILADataset(split_dir / f"{split}.csv", transform=eval_transform, cache=False)
    return DataLoader(
        dataset, batch_size=cfg["training"]["batch_size"], shuffle=False,
        num_workers=cfg["dataloader"]["num_workers"], pin_memory=cfg["dataloader"]["pin_memory"],
    )


@torch.no_grad()
def extract_split_features(fold: int, split: str, cfg: dict) -> dict[str, np.ndarray]:
    """用某折 image-only best_overall checkpoint，在该折某 split 上抽取冻结特征 + logit。"""
    ckpt_path = OUTPUT_DIR / f"resnet18_glaucoma_cv{N_FOLDS}fold{fold}_seed{SEED}_best_overall.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到 baseline checkpoint: {ckpt_path}")

    model = ResNet18Pretrained(num_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()

    captured: dict[str, torch.Tensor] = {}

    def _hook(_module: nn.Module, _inp: tuple, out: torch.Tensor) -> None:
        captured["feat"] = out.flatten(1)

    handle = model.backbone.avgpool.register_forward_hook(_hook)

    loader = build_loader(fold, split, cfg)
    feats, logits, labels, sexes, age_groups = [], [], [], [], []
    for images, label, sex, age_group in loader:        # PAPILADataset 返回 4 元组
        images = images.to(DEVICE, non_blocking=True)
        logit = model(images).squeeze(1)
        feats.append(captured["feat"].half().cpu().numpy())
        logits.append(logit.float().cpu().numpy())
        labels.append(label.numpy())
        sexes.append(sex.numpy())
        age_groups.append(age_group.numpy())
    handle.remove()

    return {
        "feat":      np.concatenate(feats).astype(np.float16),
        "logit":     np.concatenate(logits).astype(np.float32),
        "label":     np.concatenate(labels).astype(np.int64),
        "sex":       np.concatenate(sexes).astype(np.int64),
        "age_group": np.concatenate(age_groups).astype(np.int64),
    }


def cache_path(fold: int, split: str) -> Path:
    """返回某 (fold, split) 特征缓存的 npz 路径。"""
    return CACHE_DIR / f"feat_fold{fold}_{split}.npz"


def ensure_features(cfg: dict) -> None:
    """对所有 (fold, split) 抽取并缓存特征（已存在则跳过）。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for fold in range(N_FOLDS):
        for split in SPLITS:
            path = cache_path(fold, split)
            if path.exists():
                print(f"[extract] 缓存已存在，跳过: {path.name}")
                continue
            print(f"[extract] fold={fold} split={split} ...", flush=True)
            data = extract_split_features(fold, split, cfg)
            np.savez_compressed(path, **data)
            print(f"[extract]   -> {path.name}  N={len(data['label']):,}")


def load_fold(fold: int) -> dict[str, dict[str, np.ndarray]]:
    """载入某折的 train/val/test 特征缓存，返回 {split: {key: array}}。"""
    out = {}
    for split in SPLITS:
        with np.load(cache_path(fold, split)) as npz:
            out[split] = {k: npz[k] for k in npz.files}
    return out


# ============================================================
# 浅头拟合（GPU 上的凸 logistic / multinomial，LBFGS）+ 温度标定
# ============================================================
def _fit_logistic(x_train, y_train, x_val, y_val, l2):
    """全批 LBFGS 拟合二值 logistic（凸）+ val 温度标定。返回 (weight, bias, temperature)。"""
    dim = x_train.shape[1]
    linear = nn.Linear(dim, 1).to(DEVICE)
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS(linear.parameters(), lr=1.0, max_iter=200,
                                  history_size=20, line_search_fn="strong_wolfe")

    def _closure():
        optimizer.zero_grad()
        loss = bce(linear(x_train).squeeze(1), y_train) + l2 * linear.weight.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(_closure)
    with torch.no_grad():
        weight = linear.weight.detach().squeeze(0).clone()
        bias = linear.bias.detach().clone()
        val_logit = (x_val @ weight + bias)

    log_t = torch.zeros(1, device=DEVICE, requires_grad=True)
    temp_opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

    def _temp_closure():
        temp_opt.zero_grad()
        loss = bce(val_logit / torch.exp(log_t), y_val)
        loss.backward()
        return loss

    temp_opt.step(_temp_closure)
    return weight, bias, float(torch.exp(log_t.detach()))


def _fit_multinomial(x_train, y_train, n_classes, l2):
    """全批 LBFGS 拟合 K 路 softmax 线性头（凸）。返回 (weight[K,D], bias[K])。"""
    dim = x_train.shape[1]
    linear = nn.Linear(dim, n_classes).to(DEVICE)
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    ce = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS(linear.parameters(), lr=1.0, max_iter=200,
                                  history_size=20, line_search_fn="strong_wolfe")

    def _closure():
        optimizer.zero_grad()
        loss = ce(linear(x_train), y_train) + l2 * linear.weight.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(_closure)
    return linear.weight.detach().clone(), linear.bias.detach().clone()


def _per_sample_nll(logit: np.ndarray, y: np.ndarray) -> np.ndarray:
    """逐样本二值 BCE（nats）。"""
    return np.logaddexp(0.0, logit) - y * logit


def _onehot(group_id: np.ndarray, n_groups: int) -> np.ndarray:
    """整数组 id -> one-hot [N, n_groups]（float32）。"""
    oh = np.zeros((group_id.shape[0], n_groups), dtype=np.float32)
    oh[np.arange(group_id.shape[0]), group_id] = 1.0
    return oh


def _build_design(feat, group_id, n_groups, mode):
    """构建 base / add（逐组截距）/ int（逐组独立 φ 块 = 真条件信号）三种设计矩阵之一。"""
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


L2_GRID = (0.1, 1.0, 10.0, 100.0)


def _eval_test_logit(design_train, y_train, design_val, y_val, design_test, l2_grid=L2_GRID):
    """L2 网格上拟合二值 logistic（train）+ 温度标定（val），按 val NLL 选优，返回 test 校准 logit。"""
    xtr = torch.from_numpy(design_train).to(DEVICE)
    ytr = torch.from_numpy(y_train.astype(np.float32)).to(DEVICE)
    xva = torch.from_numpy(design_val).to(DEVICE)
    yva = torch.from_numpy(y_val.astype(np.float32)).to(DEVICE)
    xte = torch.from_numpy(design_test).to(DEVICE)

    best_val_nll, best_test_logit = float("inf"), None
    for l2 in l2_grid:
        weight, bias, temperature = _fit_logistic(xtr, ytr, xva, yva, l2)
        with torch.no_grad():
            val_logit = ((xva @ weight + bias) / temperature)
            val_nll = float(nn.functional.binary_cross_entropy_with_logits(val_logit, yva))
            if val_nll < best_val_nll:
                best_val_nll = val_nll
                best_test_logit = ((xte @ weight + bias) / temperature).cpu().numpy()
    return best_test_logit.astype(np.float64)


# ============================================================
# OOF 工具：逐折拟合 + 池化
# ============================================================
def _group_ids(d: dict[str, np.ndarray], attr_spec: str) -> tuple[np.ndarray, int]:
    """按属性规格返回 (组 id, 组数)。"""
    if attr_spec == "sex":
        return d["sex"].astype(np.int64), 2
    if attr_spec == "age":
        return d["age_group"].astype(np.int64), N_AGE
    if attr_spec == "joint":
        return (d["sex"] * N_AGE + d["age_group"]).astype(np.int64), N_JOINT
    raise ValueError(f"未知 attr_spec: {attr_spec}")


def _name(attr_spec: str, g: int) -> str:
    """组 id -> 可读名。"""
    return {"sex": SEX_NAMES, "age": AGE_NAMES, "joint": JOINT_NAMES}[attr_spec].get(g, str(g))


# ============================================================
# E0 —— 边际量标定（OOF 池化）
# ============================================================
def _entropy_bits(counts: np.ndarray) -> float:
    """由计数得到熵（bits）。"""
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def marginal_mi_bits(y: np.ndarray, a: np.ndarray) -> float:
    """plug-in 估计 I(Y;A)（bits）；对 MI 有正偏，仅作量级参考。"""
    y_vals, a_vals = np.unique(y), np.unique(a)
    n = y.shape[0]
    joint = np.zeros((y_vals.size, a_vals.size))
    for i, yv in enumerate(y_vals):
        for j, av in enumerate(a_vals):
            joint[i, j] = np.sum((y == yv) & (a == av))
    p_joint = joint / n
    p_y = p_joint.sum(axis=1, keepdims=True)
    p_a = p_joint.sum(axis=0, keepdims=True)
    mask = p_joint > 0
    return float(np.sum(p_joint[mask] * np.log2(p_joint[mask] / (p_y @ p_a)[mask])))


def attribute_decodability_oof(folds: list[dict], attr_spec: str, n_classes: int) -> dict:
    """
    I(A;X) 的 OOF 预测式下界 + macro-OvR AUC + top-1 acc：图像特征能多大程度解码属性 A。
    逐折用 train 拟合 softmax 头、预测该折 test，池化 OOF probs 后在完整 420 上评估。
    """
    ce_loss = nn.CrossEntropyLoss()
    oof_prob, oof_true = [], []
    for fold in folds:
        tr, va, te = fold["train"], fold["val"], fold["test"]
        xtr = torch.from_numpy(tr["feat"].astype(np.float32)).to(DEVICE)
        gid_tr, _ = _group_ids(tr, attr_spec)
        ytr = torch.from_numpy(gid_tr).to(DEVICE)
        xva = torch.from_numpy(va["feat"].astype(np.float32)).to(DEVICE)
        gid_va, _ = _group_ids(va, attr_spec)
        yva_t = torch.from_numpy(gid_va).to(DEVICE)
        xte = torch.from_numpy(te["feat"].astype(np.float32)).to(DEVICE)
        gid_te, _ = _group_ids(te, attr_spec)

        best_val_ce, best_logits = float("inf"), None
        for l2 in L2_GRID:
            weight, bias = _fit_multinomial(xtr, ytr, n_classes, l2)
            with torch.no_grad():
                val_ce = float(ce_loss(xva @ weight.t() + bias, yva_t))
                if val_ce < best_val_ce:
                    best_val_ce = val_ce
                    best_logits = (xte @ weight.t() + bias).cpu().numpy()
        z = best_logits - best_logits.max(axis=1, keepdims=True)
        probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        oof_prob.append(probs)
        oof_true.append(gid_te)

    probs = np.concatenate(oof_prob)
    a_true = np.concatenate(oof_true)
    h_a = _entropy_bits(np.bincount(a_true, minlength=n_classes))
    ce_bits = float(-np.mean(np.log(probs[np.arange(len(a_true)), a_true] + EPS)) / np.log(2))
    top1_acc = float(np.mean(probs.argmax(axis=1) == a_true))
    try:
        if n_classes == 2:
            macro_auc = float(roc_auc_score(a_true, probs[:, 1]))
        else:
            macro_auc = float(roc_auc_score(a_true, probs, multi_class="ovr", average="macro"))
    except ValueError:
        macro_auc = float("nan")
    return {"macro_ovr_auc": macro_auc, "top1_acc": top1_acc, "H(A)_bits": h_a,
            "I(A;X)_lowerbound_bits": max(h_a - ce_bits, 0.0)}


# ============================================================
# E1 —— 条件似然比 ΔNLL（= I(Y;A|X) 估计，OOF 池化）
# ============================================================
def conditional_mi_gain_oof(
    folds: list[dict], attr_spec: str, n_bootstrap: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    OOF 估计 I(Y;A|X)：逐折拟合 f(Y~φ) 与 g(add/int)，预测各折 test，池化逐样本 llr=NLL_f-NLL_g
    成完整 420 的 OOF，再算全局 ΔNLL + bootstrap CI + 逐子群 + 向少数群加权。
    """
    rng = rng or np.random.default_rng(0)
    n_g = {"sex": 2, "age": N_AGE, "joint": N_JOINT}[attr_spec]

    oof = {"llr_add": [], "llr_int": [], "y": [], "gid": [],
           "logit_f": [], "logit_add": [], "logit_int": []}
    for fold in folds:
        tr, va, te = fold["train"], fold["val"], fold["test"]
        gid_tr, _ = _group_ids(tr, attr_spec)
        gid_va, _ = _group_ids(va, attr_spec)
        gid_te, _ = _group_ids(te, attr_spec)
        y_test = te["label"].astype(np.float64)

        logit_f = _eval_test_logit(
            _build_design(tr["feat"], None, 0, "base"), tr["label"],
            _build_design(va["feat"], None, 0, "base"), va["label"],
            _build_design(te["feat"], None, 0, "base"))
        nll_f = _per_sample_nll(logit_f, y_test)

        oof["y"].append(y_test)
        oof["gid"].append(gid_te)
        oof["logit_f"].append(logit_f)
        for mode in ("add", "int"):
            logit_g = _eval_test_logit(
                _build_design(tr["feat"], gid_tr, n_g, mode), tr["label"],
                _build_design(va["feat"], gid_va, n_g, mode), va["label"],
                _build_design(te["feat"], gid_te, n_g, mode))
            oof[f"llr_{mode}"].append(nll_f - _per_sample_nll(logit_g, y_test))
            oof[f"logit_{mode}"].append(logit_g)

    y = np.concatenate(oof["y"])
    gid = np.concatenate(oof["gid"])
    result = {"attr_spec": attr_spec, "n_groups": n_g, "n_oof": int(len(y)),
              "auc_f": float(roc_auc_score(y, np.concatenate(oof["logit_f"])))}
    for mode in ("add", "int"):
        llr = np.concatenate(oof[f"llr_{mode}"])
        global_nats = float(llr.mean())
        boot = np.array([llr[rng.integers(0, llr.size, llr.size)].mean()
                         for _ in range(n_bootstrap)])
        ci = np.percentile(boot, [2.5, 97.5])
        per_group, group_means = {}, []
        for g in range(n_g):
            mask = gid == g
            if mask.sum() == 0:
                continue
            gm = float(llr[mask].mean())
            per_group[_name(attr_spec, g)] = {"delta_nll_nats": gm, "n": int(mask.sum())}
            group_means.append(gm)
        result[mode] = {
            "auc_g": float(roc_auc_score(y, np.concatenate(oof[f"logit_{mode}"]))),
            "delta_nll_nats": global_nats,
            "delta_nll_bits": global_nats / np.log(2),
            "ci95_nats": [float(ci[0]), float(ci[1])],
            "per_group": per_group,
            "minority_weighted_nats": float(np.mean(group_means)) if group_means else float("nan"),
        }
    return result


# ============================================================
# E2 —— 预后分数分层内的属性效应（OOF 池化）
# ============================================================
def stratified_cmh(score, y, a, n_bins=5):
    """二值属性版：ŝ 分层后 A–Y 条件关联（逐箱 2×2 Pearson χ² 求和=方向无关主判据 + CMH/MH-OR）。"""
    edges = np.quantile(score, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bin_id = np.clip(np.digitize(score, edges[1:-1]), 0, n_bins - 1)
    a_sum = e_sum = var_sum = den_or_num = den_or_den = 0.0
    agg_chi2, agg_df = 0.0, 0
    per_bin = []
    for b in range(n_bins):
        m = bin_id == b
        yt, at = y[m], a[m]
        nn_ = float(m.sum())
        n11 = float(np.sum((at == 1) & (yt == 1)))
        n10 = float(np.sum((at == 1) & (yt == 0)))
        n01 = float(np.sum((at == 0) & (yt == 1)))
        n00 = float(np.sum((at == 0) & (yt == 0)))
        row1, row0, col1 = n11 + n10, n01 + n00, n11 + n01
        if nn_ < 2 or row1 == 0 or row0 == 0 or col1 == 0 or col1 == nn_:
            per_bin.append({"bin": b, "n": int(nn_), "skipped": True})
            continue
        a_sum += n11
        e_sum += row1 * col1 / nn_
        var_sum += row1 * row0 * col1 * (nn_ - col1) / (nn_ * nn_ * (nn_ - 1))
        den_or_num += n11 * n00 / nn_
        den_or_den += n10 * n01 / nn_
        agg_chi2 += nn_ * (n11 * n00 - n10 * n01) ** 2 / (row1 * row0 * col1 * (nn_ - col1))
        agg_df += 1
        per_bin.append({"bin": b, "n": int(nn_), "y_rate_a1": n11 / row1, "y_rate_a0": n01 / row0,
                        "n_a1": int(row1), "n_a0": int(row0)})
    cmh_stat = (abs(a_sum - e_sum) - 0.5) ** 2 / var_sum if var_sum > 0 else float("nan")
    cmh_p = float(chi2.sf(cmh_stat, df=1)) if np.isfinite(cmh_stat) else float("nan")
    mh_or = float(den_or_num / den_or_den) if den_or_den > 0 else float("nan")
    agg_p = float(chi2.sf(agg_chi2, df=agg_df)) if agg_df > 0 else float("nan")
    return {"agg_chi2": float(agg_chi2), "agg_df": int(agg_df), "agg_p": agg_p,
            "cmh_stat": float(cmh_stat), "cmh_p": cmh_p, "mh_odds_ratio": mh_or, "per_bin": per_bin}


def stratified_multiclass_effect(score, y, group, n_groups, n_bins=5):
    """多类属性（joint 4 类）版：方向无关逐箱 (≤K)×2 Pearson χ² 求和。"""
    edges = np.quantile(score, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bin_id = np.clip(np.digitize(score, edges[1:-1]), 0, n_bins - 1)
    agg_chi2, agg_df = 0.0, 0
    per_bin = []
    for b in range(n_bins):
        m = bin_id == b
        yt, gt = y[m], group[m]
        nb = int(m.sum())
        present = [g for g in range(n_groups) if np.any(gt == g)]
        rec = {"bin": b, "n": nb}
        if nb >= 2 and (yt == 1).any() and (yt == 0).any() and len(present) >= 2:
            table = np.array([[np.sum((gt == g) & (yt == 1)),
                               np.sum((gt == g) & (yt == 0))] for g in present], dtype=np.float64)
            row = table.sum(axis=1, keepdims=True)
            col = table.sum(axis=0, keepdims=True)
            expected = row @ col / nb
            valid = expected > 0
            chi_b = float(np.sum((table[valid] - expected[valid]) ** 2 / expected[valid]))
            df_b = (table.shape[0] - 1) * (table.shape[1] - 1)
            agg_chi2 += chi_b
            agg_df += df_b
            rec.update({"chi2": chi_b, "df": df_b, "n_groups_present": len(present)})
        else:
            rec["skipped"] = True
        per_bin.append(rec)
    agg_p = float(chi2.sf(agg_chi2, df=agg_df)) if agg_df > 0 else float("nan")
    return {"agg_chi2": float(agg_chi2), "agg_df": int(agg_df), "agg_p": agg_p, "per_bin": per_bin}


def pool_oof_test(folds: list[dict]) -> dict[str, np.ndarray]:
    """把 5 折 test 的 image-only logit/label/sex/age_group 池化成完整 420 的 OOF（供 E2）。"""
    keys = ("logit", "label", "sex", "age_group")
    return {k: np.concatenate([f["test"][k] for f in folds]) for k in keys}


# ============================================================
# E3 —— 灵敏度标定（正控制 + 剂量-反应，OOF 池化）
# ============================================================
def _oof_delta_nll_synth(folds, y_syn_per_fold, a_syn_per_fold, rng, n_bootstrap=1000):
    """对每折合成 (y, 二值 a) 跑 base vs int，池化 OOF llr，返回 ΔNLL + CI + detected。"""
    oof_llr = []
    for fold, y_syn, a_syn in zip(folds, y_syn_per_fold, a_syn_per_fold):
        tr, va, te = fold["train"], fold["val"], fold["test"]
        logit_f = _eval_test_logit(
            _build_design(tr["feat"], None, 0, "base"), y_syn["train"],
            _build_design(va["feat"], None, 0, "base"), y_syn["val"],
            _build_design(te["feat"], None, 0, "base"))
        logit_g = _eval_test_logit(
            _build_design(tr["feat"], a_syn["train"], 2, "int"), y_syn["train"],
            _build_design(va["feat"], a_syn["val"], 2, "int"), y_syn["val"],
            _build_design(te["feat"], a_syn["test"], 2, "int"))
        y_test = y_syn["test"].astype(np.float64)
        oof_llr.append(_per_sample_nll(logit_f, y_test) - _per_sample_nll(logit_g, y_test))
    llr = np.concatenate(oof_llr)
    boot = np.array([llr[rng.integers(0, llr.size, llr.size)].mean() for _ in range(n_bootstrap)])
    ci = np.percentile(boot, [2.5, 97.5])
    return {"delta_nll_nats": float(llr.mean()), "ci95_nats": [float(ci[0]), float(ci[1])],
            "detected": bool(ci[0] > 0)}


def sensitivity_calibration_oof(folds: list[dict], seed: int = 12345) -> dict:
    """E3：正控制 Y'=Y XOR r + 剂量-反应（注入强度 α 的合成条件信号），全部 OOF 池化。"""
    rng = np.random.default_rng(seed)

    # 正控制：r 与图像无关，Y'=Y XOR r；a_syn=r
    r_per_fold, yxor_per_fold = [], []
    for fold in folds:
        r = {s: rng.integers(0, 2, fold[s]["label"].shape[0]).astype(np.int64) for s in SPLITS}
        yxor = {s: (fold[s]["label"] ^ r[s]).astype(np.int64) for s in SPLITS}
        r_per_fold.append(r)
        yxor_per_fold.append(yxor)
    positive_control = _oof_delta_nll_synth(folds, yxor_per_fold, r_per_fold, rng)

    # 剂量-反应：a_syn 独立随机；以概率 α 用 a_syn 覆盖标签
    a_per_fold = [{s: rng.integers(0, 2, fold[s]["label"].shape[0]).astype(np.int64)
                   for s in SPLITS} for fold in folds]
    dose_response = {}
    for alpha in (0.0, 0.02, 0.05, 0.1, 0.2, 0.4):
        y_per_fold = []
        for fold, a_syn in zip(folds, a_per_fold):
            y_alpha = {}
            for s in SPLITS:
                n = fold[s]["label"].shape[0]
                replace = rng.random(n) < alpha
                y_s = fold[s]["label"].copy()
                y_s[replace] = a_syn[s][replace]
                y_alpha[s] = y_s.astype(np.int64)
            y_per_fold.append(y_alpha)
        dose_response[f"alpha={alpha}"] = _oof_delta_nll_synth(folds, y_per_fold, a_per_fold, rng)
    return {"positive_control_xor": positive_control, "dose_response": dose_response}


# ============================================================
# 编排
# ============================================================
def main() -> None:
    """抽取特征（缺则补）→ 跑 E0–E3（OOF 池化）→ 落 JSON + 打印可读小结。"""
    print(f"Device: {DEVICE}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ---------- Stage A：特征抽取（GPU，逐折）----------
    ensure_features(cfg)
    folds = [load_fold(k) for k in range(N_FOLDS)]
    oof_te = pool_oof_test(folds)         # 完整 420 的 image-only OOF（供 E0 边际 / E2）
    attr_specs = ["sex", "age", "joint"]
    rng = np.random.default_rng(SEED)

    # ---------- E0 ----------
    print(f"\n{'=' * 70}\nE0 边际量标定（OOF 池化 N={len(oof_te['label'])}）\n{'=' * 70}", flush=True)
    e0 = {"marginal_mi_bits": {}, "decodability": {}}
    joint_oof = (oof_te["sex"] * N_AGE + oof_te["age_group"]).astype(np.int64)
    e0["marginal_mi_bits"]["sex"] = marginal_mi_bits(oof_te["label"], oof_te["sex"])
    e0["marginal_mi_bits"]["age"] = marginal_mi_bits(oof_te["label"], oof_te["age_group"])
    e0["marginal_mi_bits"]["joint"] = marginal_mi_bits(oof_te["label"], joint_oof)
    e0["decodability"]["sex"] = attribute_decodability_oof(folds, "sex", 2)
    e0["decodability"]["age"] = attribute_decodability_oof(folds, "age", N_AGE)
    e0["decodability"]["joint"] = attribute_decodability_oof(folds, "joint", N_JOINT)
    for attr in attr_specs:
        d = e0["decodability"][attr]
        print(f"[E0] {attr:5s}  I(Y;A)={e0['marginal_mi_bits'][attr]:.4f} bits | "
              f"I(A;X)>= {d['I(A;X)_lowerbound_bits']:.3f} bits "
              f"(macro-AUC {d['macro_ovr_auc']:.3f}, top1 {d['top1_acc']:.3f})")

    # ---------- E1 ----------
    print(f"\n{'=' * 70}\nE1 条件似然比 ΔNLL = I(Y;A|X) 估计（OOF 池化）\n{'=' * 70}", flush=True)
    e1 = {}
    for spec in attr_specs:
        e1[spec] = conditional_mi_gain_oof(folds, spec, rng=rng)
        g = e1[spec]["int"]
        print(f"[E1] {spec:5s} ΔNLL_int={g['delta_nll_nats']:+.5f} nats "
              f"CI[{g['ci95_nats'][0]:+.5f},{g['ci95_nats'][1]:+.5f}] "
              f"| minority-weighted={g['minority_weighted_nats']:+.5f} "
              f"| auc_f={e1[spec]['auc_f']:.3f} auc_g={g['auc_g']:.3f}")

    # ---------- E2 ----------
    print(f"\n{'=' * 70}\nE2 预后分数分层内属性效应（OOF 池化）\n{'=' * 70}", flush=True)
    e2 = {}
    e2["sex"] = stratified_cmh(oof_te["logit"], oof_te["label"], oof_te["sex"])
    e2["age"] = stratified_cmh(oof_te["logit"], oof_te["label"], oof_te["age_group"])
    e2["joint"] = stratified_multiclass_effect(oof_te["logit"], oof_te["label"], joint_oof, N_JOINT)
    for attr in ("sex", "age"):
        print(f"[E2] {attr:5s} aggχ²={e2[attr]['agg_chi2']:.2f}(df={e2[attr]['agg_df']}) "
              f"agg_p={e2[attr]['agg_p']:.3g} | CMH_p={e2[attr]['cmh_p']:.3g} "
              f"MH-OR={e2[attr]['mh_odds_ratio']:.3f}")
    print(f"[E2] joint aggχ²={e2['joint']['agg_chi2']:.2f}(df={e2['joint']['agg_df']}) "
          f"agg_p={e2['joint']['agg_p']:.3g}")

    # ---------- E3 ----------
    print(f"\n{'=' * 70}\nE3 灵敏度标定（OOF 池化，full 420）\n{'=' * 70}", flush=True)
    e3 = sensitivity_calibration_oof(folds)
    pc = e3["positive_control_xor"]
    print(f"[E3] 正控制 Y XOR r: ΔNLL={pc['delta_nll_nats']:.4f} nats "
          f"CI[{pc['ci95_nats'][0]:.4f},{pc['ci95_nats'][1]:.4f}] detected={pc['detected']}")
    for k, v in e3["dose_response"].items():
        print(f"[E3] {k:12s} ΔNLL={v['delta_nll_nats']:+.5f} "
              f"CI[{v['ci95_nats'][0]:+.5f},{v['ci95_nats'][1]:+.5f}] detected={v['detected']}")

    # ---------- 落盘 + 头条 ----------
    headline = {spec: {
        "delta_nll_int_nats": e1[spec]["int"]["delta_nll_nats"],
        "ci95_nats": e1[spec]["int"]["ci95_nats"],
        "minority_weighted_nats": e1[spec]["int"]["minority_weighted_nats"],
        "delta_nll_add_nats": e1[spec]["add"]["delta_nll_nats"],
    } for spec in attr_specs}

    out = {
        "config": {"n_folds": N_FOLDS, "seed": SEED, "l2_grid": list(L2_GRID),
                   "feature_dim": FEATURE_DIM, "task": "glaucoma_binary",
                   "attributes": "sex (2), age_group (2), joint sex×age (4)",
                   "design": "OOF pooled across 5 patient-grouped folds (N=420)"},
        "headline_I(Y;A|X)_estimate": headline,
        "E0": e0, "E1": e1, "E2": e2, "E3_sensitivity": e3,
    }
    out_path = OUTPUT_DIR / "conditional_mi.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_path}")

    print(f"\n{'=' * 70}\n头条：I(Y;A|X) 估计（int 模型 ΔNLL，OOF 池化 N=420，nats）\n{'=' * 70}")
    for spec in attr_specs:
        h = headline[spec]
        sig = "信号在" if h["ci95_nats"][0] > 0 else "≈0(CI含0)"
        print(f"  {spec:6s} 全局 {h['delta_nll_int_nats']:+.5f} "
              f"CI[{h['ci95_nats'][0]:+.5f},{h['ci95_nats'][1]:+.5f}] "
              f"| 少数群加权 {h['minority_weighted_nats']:+.5f}  -> {sig}")


if __name__ == "__main__":
    main()
