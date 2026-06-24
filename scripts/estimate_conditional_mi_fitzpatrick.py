"""
估计 Fitzpatrick17k malignant 上的条件互信息 I(Y;A|X)，模型无关地检验「肤色信号是否弱」
================================================================================================

本脚本是 MIMIC-CXR 版 scripts/estimate_conditional_mi.py 的严格平移：判据、四层设计
(E0–E3)、估计器、决策规则全部保留，仅按 Fitzpatrick 的「单一 6 类肤色属性」改写三处
(见下「与 MIMIC 的差异」)。

判据（见项目记忆 mimic-fairness-signal-reframing / 仓库 CLAUDE.md「诊断方法重构」节）：
    HN 要赢 attribute-blind 池化基线，必要条件是 I(Y;A|X) > 0（给定图像后，子群仍需不同
    决策函数）。若该量 = 0，盲池化已是 Bayes 最优，任何条件化架构（含全部 HN 变体）顶多
    退化成它、不可能超过。本脚本全程不含 HN，只估计这个量本身，把「信号弱」与「机制没学到」
    彻底切开。

任务/属性（与 MIMIC 的差异）：
    Y = malignant（0=benign/non-neoplastic, 1=malignant，恶性率 ~13.5%）
    A = skin（Fitzpatrick I–VI → 0–5，单轴 6 子群；MIMIC 是 sex/race/age 三个二值 + joint-8）
    backbone = image-only resnet18_malignant_seed{42,43,44}_best_overall.pth
    定位：肤色在皮肤镜图像中视觉强可解码、且与恶性率有梯度关联，预期是「强属性正控制」——
    与 MIMIC「信号弱」判决互为正/负对照，同时验证本探针在真有信号时能读出「强」。

三处按多类属性的必要适配（其余照搬 MIMIC 版）：
    E0 解码度：二值 logistic+roc_auc → 6 路 softmax 线性头；MI 下界仍 H(skin)−CE(bits)，
               AUC 换 macro-OvR AUC + top-1 acc。
    E1 设计矩阵：_build_design 的 add/int 对 n_groups 已通用 → 直接 n_groups=6，组 id=skin。
               另跑 skin_bin（浅 I–III=0 / 深 IV–VI=1，2 组）作可与 MIMIC 横比的二值视图。
    E2 CMH：主判据(方向无关)推广为每箱 (≤6)×2 Pearson χ² 求和；方向性 MH-OR 仅在 skin_bin 上算。

四层互补设计（每层偏差方向不同，三角验证）：
    E0  边际量标定（不当判据，只解释）：I(Y;skin) 边际 MI；I(skin;X) 可解码度（解释胸/皮像
        已吸收多少 A，A 越能从 X 解码，条件信号越易被压向 0）。
    E1  主估计器：冻结特征 φ(X) 上的条件似然比 ΔNLL（= I(Y;A|X) 估计）。
        f: Y~φ；g_add: Y~[φ, group]（逐组截距=校准型增益）；g_int: 每组独立[1,φ]（真条件信号）。
        held-out ΔNLL 是 I 的保守（下界向）估计：g 欠拟合只会低估。报全局/逐组/向少数群加权。
    E2  无参数对照：image-only logit ŝ=p(Y|X) 分层，箱内查 A 与 Y 的条件关联。
    E3  灵敏度标定：正控制 Y'=Y XOR r（必须读出大 ΔNLL，否则探针失灵）+ 剂量-反应功效曲线
        （注入强度 α 的合成条件信号，标定本数据规模下的最小可检测效应 MDE）。

决策规则（对接 reframing 三分支）：
    全局 ΔNLL CI 上界≈0 且不集中于小子群、E2 各箱无显著、E3 证明探针有功效
        → 「信号弱」成立，HN 注定无收益。
    全局≈0 但集中于数据稀缺子群（深肤色 V/VI）→ 唯一能怪机制的区间。
    全局>0（本数据集的预期方向）→ 信号在、HN 没用上 → 机制/架构有可争取空间。

运行方式：
    特征抽取是对 ~16k 张图各一次前向（GPU），按项目规范 sbatch 提交，见
    slurm/estimate_conditional_mi_fitzpatrick.sh。E0–E3 分析在缓存特征上跑，随作业一并完成。
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

from src.datasets.fitzpatrick_dataset import FitzpatrickDataset
from src.models.resnet18_pretrained import ResNet18Pretrained

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "fitzpatrick_baseline.yaml"
OUTPUT_DIR = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/fitzpatrick")
CACHE_DIR = OUTPUT_DIR / "cmi_cache"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = (42, 43, 44)            # backbone best_overall checkpoint 的 seed
SPLITS = ("train", "val", "test")
FEATURE_DIM = 512               # resnet18 avgpool 输出维度
N_SKIN = 6                      # Fitzpatrick 肤色 I–VI

# 肤色取值名（与 fitzpatrick_fairness 口径一致：skin = fitzpatrick_scale - 1）
SKIN_NAMES = {0: "I", 1: "II", 2: "III", 3: "IV", 4: "V", 5: "VI"}
EPS = 1e-12


# ============================================================
# Stage A —— 特征抽取（唯一需要 GPU 的部分）
# ============================================================
def build_loader(split: str, cfg: dict) -> DataLoader:
    """为指定 split 构建 eval（无增强、确定性）DataLoader，用于抽取冻结特征。"""
    norm = cfg["transforms"]["normalize"]
    # 注意：train split 也用 eval transform（无翻转/旋转），保证特征对每张图确定、可缓存复现。
    # Fitzpatrick preproc_224x224 为自然 RGB（uint8），不需要 CXR 的 Grayscale/Resize。
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=norm["mean"], std=norm["std"]),
    ])
    split_dir = REPO_ROOT / cfg["data"]["split_dir"]
    dataset = FitzpatrickDataset(split_dir / f"{split}.csv", transform=eval_transform)
    return DataLoader(
        dataset, batch_size=cfg["training"]["batch_size"], shuffle=False,
        num_workers=cfg["dataloader"]["num_workers"], pin_memory=cfg["dataloader"]["pin_memory"],
    )


@torch.no_grad()
def extract_split_features(seed: int, split: str, cfg: dict) -> dict[str, np.ndarray]:
    """
    用 seed 的 image-only best_overall checkpoint，在一个 split 上抽取冻结特征。

    Args:
        seed:  backbone checkpoint 的 seed。
        split: 'train' / 'val' / 'test'。
        cfg:   已解析的 baseline 配置。

    Returns:
        dict，键为 feat[float16,(N,512)] / logit[float32,(N,)] / label,skin[int64,(N,)]。
        feat 为 avgpool 后 512 维倒数第二层特征，logit 为 image-only 模型对该图的输出
        logit（= E2 的预后分数 ŝ 来源）。
    """
    ckpt_path = OUTPUT_DIR / f"resnet18_malignant_seed{seed}_best_overall.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到 baseline checkpoint: {ckpt_path}")

    model = ResNet18Pretrained(num_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()

    # forward hook 捕获 avgpool 输出（[B,512,1,1] -> [B,512]），与 logit 同一次前向得到
    captured: dict[str, torch.Tensor] = {}

    def _hook(_module: nn.Module, _inp: tuple, out: torch.Tensor) -> None:
        captured["feat"] = out.flatten(1)

    handle = model.backbone.avgpool.register_forward_hook(_hook)

    loader = build_loader(split, cfg)
    feats, logits, labels, skins = [], [], [], []
    for images, label, skin in loader:                 # FitzpatrickDataset 返回 3 元组
        images = images.to(DEVICE, non_blocking=True)
        logit = model(images).squeeze(1)               # [B]
        feats.append(captured["feat"].half().cpu().numpy())
        logits.append(logit.float().cpu().numpy())
        labels.append(label.numpy())
        skins.append(skin.numpy())
    handle.remove()

    return {
        "feat":  np.concatenate(feats).astype(np.float16),
        "logit": np.concatenate(logits).astype(np.float32),
        "label": np.concatenate(labels).astype(np.int64),
        "skin":  np.concatenate(skins).astype(np.int64),
    }


def cache_path(seed: int, split: str) -> Path:
    """返回某 (seed, split) 特征缓存的 npz 路径。"""
    return CACHE_DIR / f"feat_seed{seed}_{split}.npz"


def ensure_features(seeds: tuple[int, ...], cfg: dict) -> None:
    """对所有 (seed, split) 抽取并缓存特征（已存在则跳过）。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for split in SPLITS:
            path = cache_path(seed, split)
            if path.exists():
                print(f"[extract] 缓存已存在，跳过: {path.name}")
                continue
            print(f"[extract] seed={seed} split={split} ...", flush=True)
            data = extract_split_features(seed, split, cfg)
            np.savez_compressed(path, **data)
            print(f"[extract]   -> {path.name}  N={len(data['label']):,}")


def load_features(seed: int) -> dict[str, dict[str, np.ndarray]]:
    """载入某 seed 的 train/val/test 特征缓存，返回 {split: {key: array}}。"""
    out = {}
    for split in SPLITS:
        with np.load(cache_path(seed, split)) as npz:
            out[split] = {k: npz[k] for k in npz.files}
    return out


# ============================================================
# 浅头拟合（GPU 上的凸 logistic / multinomial，LBFGS）+ 温度标定
# ============================================================
def _fit_logistic(
    x_train: torch.Tensor, y_train: torch.Tensor,
    x_val: torch.Tensor, y_val: torch.Tensor,
    l2: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    全批 LBFGS 拟合二值 logistic 回归（凸，收敛干净，最大限度排除「没学到」），并在 val 上
    做温度标定（保证 NLL 口径下良好校准）。

    Returns:
        (weight[D], bias[1], temperature)；预测 logit = x @ weight + bias，
        校准后 logit = 该 logit / temperature。
    """
    dim = x_train.shape[1]
    linear = nn.Linear(dim, 1).to(DEVICE)
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS(linear.parameters(), lr=1.0, max_iter=200,
                                  history_size=20, line_search_fn="strong_wolfe")

    def _closure() -> torch.Tensor:
        optimizer.zero_grad()
        logit = linear(x_train).squeeze(1)
        loss = bce(logit, y_train) + l2 * linear.weight.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(_closure)

    with torch.no_grad():
        weight = linear.weight.detach().squeeze(0).clone()
        bias = linear.bias.detach().clone()
        val_logit = (x_val @ weight + bias)

    # 温度标定：在 val 上拟合单标量 T，最小化 NLL(logit / T)
    log_t = torch.zeros(1, device=DEVICE, requires_grad=True)
    temp_opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

    def _temp_closure() -> torch.Tensor:
        temp_opt.zero_grad()
        loss = bce(val_logit / torch.exp(log_t), y_val)
        loss.backward()
        return loss

    temp_opt.step(_temp_closure)
    temperature = float(torch.exp(log_t.detach()))
    return weight, bias, temperature


def _fit_multinomial(
    x_train: torch.Tensor, y_train: torch.Tensor,
    n_classes: int, l2: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    全批 LBFGS 拟合 K 路 softmax 线性头（凸），用于 E0 多类属性可解码度。

    Args:
        x_train:   设计矩阵 [N, D]（float32, DEVICE）。
        y_train:   整数类别标签 [N]（long, DEVICE）。
        n_classes: 类别数（=6）。
        l2:        L2 正则强度（作用于权重）。

    Returns:
        (weight[K, D], bias[K])；logits = x @ weight.t() + bias。val/test 评估在调用处做。
    """
    dim = x_train.shape[1]
    linear = nn.Linear(dim, n_classes).to(DEVICE)
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    ce = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS(linear.parameters(), lr=1.0, max_iter=200,
                                  history_size=20, line_search_fn="strong_wolfe")

    def _closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = ce(linear(x_train), y_train) + l2 * linear.weight.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(_closure)
    return linear.weight.detach().clone(), linear.bias.detach().clone()


def _per_sample_nll(logit: np.ndarray, y: np.ndarray) -> np.ndarray:
    """逐样本二值 BCE（nats）。logit 为已温度标定后的 logit，y 为 0/1。"""
    # 数值稳定的 log(1+exp)：softplus(-logit) + (1-y)*logit
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
    构建三种设计矩阵之一。

    Args:
        feat:     冻结特征 [N, 512]（float32）。
        group_id: 组 id [N]（base 模式可为 None）。
        n_groups: 组数。
        mode:     'base' = [φ]；'add' = [φ, onehot(组)去首列]（逐组截距）；
                  'int'  = 逐组独立 [1, φ] 块设计（逐组不同决策函数 = 真正的条件信号）。

    Returns:
        设计矩阵 [N, D]（float32）。
    """
    feat = feat.astype(np.float32)
    n = feat.shape[0]
    if mode == "base":
        return feat
    if mode == "add":
        # 去掉 one-hot 首列以免与 bias 共线（参照组吸收进 bias）
        return np.concatenate([feat, _onehot(group_id, n_groups)[:, 1:]], axis=1)
    if mode == "int":
        # 块设计：每个组占据自己的 φ 块（其余块为 0）+ 全 one-hot 提供逐组截距。
        # 维度 = n_groups*(512+1)；等价于「逐组独立 logistic + 共享 L2」。
        block = np.zeros((n, n_groups * FEATURE_DIM), dtype=np.float32)
        rows = np.arange(n)
        for g in range(n_groups):
            mask = group_id == g
            block[np.ix_(rows[mask], np.arange(g * FEATURE_DIM, (g + 1) * FEATURE_DIM))] = feat[mask]
        return np.concatenate([block, _onehot(group_id, n_groups)], axis=1)
    raise ValueError(f"未知 mode: {mode}")


L2_GRID = (0.1, 1.0, 10.0, 100.0)  # 在 val 上按 NLL/CE 选 L2，使 ΔNLL 这个下界尽量紧


def _eval_test_logit(
    design_train: np.ndarray, y_train: np.ndarray,
    design_val: np.ndarray, y_val: np.ndarray,
    design_test: np.ndarray, l2_grid: tuple[float, ...] = L2_GRID,
) -> np.ndarray:
    """
    在 L2 网格上各拟合一个二值 logistic（train）+ 温度标定（val），按 val NLL 选最优，返回
    test 上校准后的 logit。给每个头「最强一击」——这样 ΔNLL≈0 才是「信号确实弱」而非「正则没调好」。
    """
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
# E0 —— 边际量标定
# ============================================================
def _entropy_bits(counts: np.ndarray) -> float:
    """由计数得到熵（bits）。"""
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def marginal_mi_bits(y: np.ndarray, a: np.ndarray) -> float:
    """
    plug-in 估计 I(Y;A)（bits）。注意 plug-in 对 MI 有正偏，仅作量级参考。
    对任意基数的 a（含 6 类 skin）通用。
    """
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


def multiclass_decodability(
    feats: dict[str, dict[str, np.ndarray]], n_classes: int = N_SKIN,
) -> dict[str, float]:
    """
    I(skin;X) 的预测式下界 + test macro-OvR AUC + top-1 acc：图像特征能多大程度解码肤色。
    用于解释「皮肤镜图像已吸收多少肤色」（肤色越能从 X 解码，I(Y;skin|X) 越易被压向 0）。

    MIMIC 版属性是二值（用 _eval_test_logit + roc_auc）；此处肤色 6 类，改 6 路 softmax 头，
    在 L2 网格上按 val CE 选优，MI 下界仍为 H(skin) − CE（bits）。
    """
    tr, va, te = feats["train"], feats["val"], feats["test"]
    xtr = torch.from_numpy(tr["feat"].astype(np.float32)).to(DEVICE)
    ytr = torch.from_numpy(tr["skin"].astype(np.int64)).to(DEVICE)
    xva = torch.from_numpy(va["feat"].astype(np.float32)).to(DEVICE)
    yva = va["skin"].astype(np.int64)
    xva_t = torch.from_numpy(yva).to(DEVICE)
    xte = torch.from_numpy(te["feat"].astype(np.float32)).to(DEVICE)
    a_test = te["skin"].astype(np.int64)

    ce_loss = nn.CrossEntropyLoss()
    best_val_ce, best_test_logits = float("inf"), None
    for l2 in L2_GRID:
        weight, bias = _fit_multinomial(xtr, ytr, n_classes, l2)
        with torch.no_grad():
            val_logits = xva @ weight.t() + bias
            val_ce = float(ce_loss(val_logits, xva_t))
            if val_ce < best_val_ce:
                best_val_ce = val_ce
                best_test_logits = (xte @ weight.t() + bias).cpu().numpy()

    # softmax 概率（数值稳定）
    z = best_test_logits - best_test_logits.max(axis=1, keepdims=True)
    probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    # 预测式 MI 下界（bits）：I(skin;X) >= H(skin) − CE，CE 用 test 上交叉熵换底到 bits
    h_a = _entropy_bits(np.bincount(a_test, minlength=n_classes))
    ce_bits = float(-np.mean(np.log(probs[np.arange(len(a_test)), a_test] + EPS)) / np.log(2))
    top1_acc = float(np.mean(probs.argmax(axis=1) == a_test))
    macro_auc = float(roc_auc_score(a_test, probs, multi_class="ovr", average="macro"))
    return {
        "macro_ovr_auc": macro_auc,
        "top1_acc": top1_acc,
        "H(skin)_bits": h_a,
        "I(skin;X)_lowerbound_bits": max(h_a - ce_bits, 0.0),
    }


# ============================================================
# E1 —— 条件似然比 ΔNLL（= I(Y;A|X) 估计）
# ============================================================
def conditional_mi_gain(
    feats: dict[str, dict[str, np.ndarray]], attr_spec: str,
    n_bootstrap: int = 1000, rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    估计某属性规格的 I(Y;A|X)：ΔNLL = NLL_f − NLL_g（test, nats）。

    Args:
        attr_spec: 'skin'（6 组）或 'skin_bin'（浅 I–III=0 / 深 IV–VI=1，2 组）。
        n_bootstrap: 全局 ΔNLL 的 bootstrap 次数（test 上重采样）。

    Returns:
        含 add/int 两种 g 的全局 ΔNLL（nats/bits）+ 95% CI + 逐子群 + 向少数群加权值。
    """
    rng = rng or np.random.default_rng(0)
    tr, va, te = feats["train"], feats["val"], feats["test"]

    def _group_ids(d: dict[str, np.ndarray]) -> tuple[np.ndarray, int]:
        if attr_spec == "skin":
            return d["skin"].astype(np.int64), N_SKIN
        if attr_spec == "skin_bin":
            return (d["skin"] >= 3).astype(np.int64), 2   # 浅(I–III)=0, 深(IV–VI)=1
        raise ValueError(f"未知 attr_spec: {attr_spec}")

    gid_tr, n_g = _group_ids(tr)
    gid_va, _ = _group_ids(va)
    gid_te, _ = _group_ids(te)
    y_test = te["label"].astype(np.float64)

    # f: Y ~ φ
    logit_f = _eval_test_logit(
        _build_design(tr["feat"], None, 0, "base"), tr["label"],
        _build_design(va["feat"], None, 0, "base"), va["label"],
        _build_design(te["feat"], None, 0, "base"),
    )
    nll_f = _per_sample_nll(logit_f, y_test)

    result: dict = {"attr_spec": attr_spec, "n_groups": n_g, "auc_f": float(roc_auc_score(y_test, logit_f))}
    for mode in ("add", "int"):
        logit_g = _eval_test_logit(
            _build_design(tr["feat"], gid_tr, n_g, mode), tr["label"],
            _build_design(va["feat"], gid_va, n_g, mode), va["label"],
            _build_design(te["feat"], gid_te, n_g, mode),
        )
        nll_g = _per_sample_nll(logit_g, y_test)
        llr = nll_f - nll_g                        # 逐样本对数似然比；均值 = ΔNLL = I 估计

        # 全局 ΔNLL + bootstrap CI（test 上重采样样本）
        global_nats = float(llr.mean())
        boot = np.array([llr[rng.integers(0, llr.size, llr.size)].mean() for _ in range(n_bootstrap)])
        ci = np.percentile(boot, [2.5, 97.5])

        # 逐子群 ΔNLL + 向少数群加权（每组等权 = 逐组均值再平均；天然上调小组 V/VI）
        per_group = {}
        group_means = []
        for g in range(n_g):
            mask = gid_te == g
            if mask.sum() == 0:
                continue
            gm = float(llr[mask].mean())
            name = SKIN_NAMES.get(g, str(g)) if attr_spec == "skin" else ("light" if g == 0 else "dark")
            per_group[name] = {"delta_nll_nats": gm, "n": int(mask.sum())}
            group_means.append(gm)
        minority_weighted = float(np.mean(group_means)) if group_means else float("nan")

        result[mode] = {
            "auc_g": float(roc_auc_score(y_test, logit_g)),
            "delta_nll_nats": global_nats,
            "delta_nll_bits": global_nats / np.log(2),
            "ci95_nats": [float(ci[0]), float(ci[1])],
            "per_group": per_group,
            "minority_weighted_nats": minority_weighted,
        }
    return result


# ============================================================
# E2 —— 预后分数分层内的肤色效应（方向无关 K×2 χ² + skin_bin CMH）
# ============================================================
def stratified_skin_effect(
    score: np.ndarray, y: np.ndarray, skin: np.ndarray, n_bins: int = 10,
) -> dict:
    """
    在按预后分数 ŝ 分层后，检验肤色 A 与二值标签 Y 的条件关联。

    原理：ŝ=p(Y|X) 是 X 对 Y 预测内容的一维充分摘要；在 ŝ≈常数的箱内，X 能解释的部分被
    钉住，箱内 A 与 Y 的残余关联即 I(Y;A|X)>0 的模型无关证据。

    与 MIMIC 版差异：肤色 6 类，主判据从「2×2 Pearson χ²」推广为「逐箱 (≤6)×2 列联表
    Pearson χ² 求和」（方向无关，对随 ŝ 翻转的纯交互也敏感，是主判据）；方向性的 CMH/MH-OR
    仅在二值化 skin_bin（深 vs 浅）上算作次要参考。

    Returns:
        - agg_chi2 / agg_df / agg_p：逐箱 (≤6)×2 Pearson χ² 求和 ~ χ²(Σdf)，**方向无关**主判据。
        - cmh_stat / cmh_p / mh_odds_ratio：skin_bin（深=1/浅=0）上的 Cochran–Mantel–Haenszel。
        - per_bin：逐箱诊断（各肤色 Y 率、深/浅 Y 率）。
    """
    edges = np.quantile(score, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bin_id = np.clip(np.digitize(score, edges[1:-1]), 0, n_bins - 1)
    skin_bin = (skin >= 3).astype(np.int64)        # 深(IV–VI)=1, 浅(I–III)=0

    agg_chi2, agg_df = 0.0, 0
    # CMH 累加（skin_bin 二值，同向假设）
    a_sum, e_sum, var_sum = 0.0, 0.0, 0.0
    den_or_num, den_or_den = 0.0, 0.0
    per_bin = []
    for b in range(n_bins):
        m = bin_id == b
        yt, st, sb = y[m], skin[m], skin_bin[m]
        nn_b = int(m.sum())

        # ---- 主判据：该箱 (出现的肤色) × (Y=0/1) Pearson χ² ----
        present = [g for g in range(N_SKIN) if np.any(st == g)]
        # 仅当 Y 两类都出现、且≥2 个肤色出现时该箱才贡献统计量
        if nn_b >= 2 and (yt == 1).any() and (yt == 0).any() and len(present) >= 2:
            table = np.array([[np.sum((st == g) & (yt == 1)),
                               np.sum((st == g) & (yt == 0))] for g in present], dtype=np.float64)
            row = table.sum(axis=1, keepdims=True)
            col = table.sum(axis=0, keepdims=True)
            expected = row @ col / nn_b
            valid = expected > 0
            chi_b = float(np.sum((table[valid] - expected[valid]) ** 2 / expected[valid]))
            df_b = (table.shape[0] - 1) * (table.shape[1] - 1)
            agg_chi2 += chi_b
            agg_df += df_b

        # ---- 次要：skin_bin 上的 2×2 CMH 累加 ----
        n11 = float(np.sum((sb == 1) & (yt == 1)))
        n10 = float(np.sum((sb == 1) & (yt == 0)))
        n01 = float(np.sum((sb == 0) & (yt == 1)))
        n00 = float(np.sum((sb == 0) & (yt == 0)))
        row1, row0 = n11 + n10, n01 + n00
        col1 = n11 + n01
        n = float(nn_b)
        bin_rec = {"bin": b, "n": nn_b}
        if n >= 2 and row1 > 0 and row0 > 0 and col1 > 0 and col1 < n:
            a_sum += n11
            e_sum += row1 * col1 / n
            var_sum += row1 * row0 * col1 * (n - col1) / (n * n * (n - 1))
            den_or_num += n11 * n00 / n
            den_or_den += n10 * n01 / n
            bin_rec.update({"y_rate_dark": n11 / row1, "y_rate_light": n01 / row0,
                            "n_dark": int(row1), "n_light": int(row0)})
        else:
            bin_rec["skipped_cmh"] = True
        per_bin.append(bin_rec)

    cmh_stat = (abs(a_sum - e_sum) - 0.5) ** 2 / var_sum if var_sum > 0 else float("nan")
    cmh_p = float(chi2.sf(cmh_stat, df=1)) if np.isfinite(cmh_stat) else float("nan")
    mh_or = float(den_or_num / den_or_den) if den_or_den > 0 else float("nan")
    agg_p = float(chi2.sf(agg_chi2, df=agg_df)) if agg_df > 0 else float("nan")
    return {
        "agg_chi2": float(agg_chi2), "agg_df": int(agg_df), "agg_p": agg_p,
        "cmh_stat": float(cmh_stat), "cmh_p": cmh_p, "mh_odds_ratio": mh_or,
        "per_bin": per_bin,
    }


# ============================================================
# E3 —— 灵敏度标定（正控制 + 剂量-反应功效曲线）
# ============================================================
def _delta_nll_for_synthetic(
    feats: dict[str, dict[str, np.ndarray]], y_syn: dict[str, np.ndarray],
    a_syn: dict[str, np.ndarray], rng: np.random.Generator,
    n_bootstrap: int = 500,
) -> dict:
    """对一组合成 (标签 y_syn, 二值属性 a_syn) 跑 E1 的 int 模型，返回 ΔNLL + CI。"""
    tr, va, te = feats["train"], feats["val"], feats["test"]
    y_test = y_syn["test"].astype(np.float64)
    logit_f = _eval_test_logit(
        _build_design(tr["feat"], None, 0, "base"), y_syn["train"],
        _build_design(va["feat"], None, 0, "base"), y_syn["val"],
        _build_design(te["feat"], None, 0, "base"),
    )
    logit_g = _eval_test_logit(
        _build_design(tr["feat"], a_syn["train"], 2, "int"), y_syn["train"],
        _build_design(va["feat"], a_syn["val"], 2, "int"), y_syn["val"],
        _build_design(te["feat"], a_syn["test"], 2, "int"),
    )
    llr = _per_sample_nll(logit_f, y_test) - _per_sample_nll(logit_g, y_test)
    boot = np.array([llr[rng.integers(0, llr.size, llr.size)].mean() for _ in range(n_bootstrap)])
    ci = np.percentile(boot, [2.5, 97.5])
    return {"delta_nll_nats": float(llr.mean()), "ci95_nats": [float(ci[0]), float(ci[1])],
            "detected": bool(ci[0] > 0)}


def sensitivity_calibration(
    feats: dict[str, dict[str, np.ndarray]], seed: int = 12345,
) -> dict:
    """
    E3：正控制 + 剂量-反应（与 MIMIC 版逐字一致；属性无关的合成检验）。

    正控制：r~Bernoulli(0.5) 与图像无关，Y'=Y XOR r。a_syn=r、label=Y' 跑 int 模型，
        ΔNLL 必须很大（探针对纯条件信号有功效），否则整套读数不可信。
    剂量-反应：a_syn~Bernoulli(0.5)；以概率 α 把标签替换为 a_syn（注入强度 α 的条件信号），
        画 ΔNLL vs α；CI>0 的最小 α = 最小可检测效应（本数据规模 ~16k 下的灵敏度地板，
        预期比 MIMIC ~199k 更高）。
    """
    rng = np.random.default_rng(seed)
    sizes = {s: feats[s]["label"].shape[0] for s in SPLITS}

    # 正控制
    r = {s: rng.integers(0, 2, sizes[s]).astype(np.int64) for s in SPLITS}
    y_xor = {s: (feats[s]["label"] ^ r[s]).astype(np.int64) for s in SPLITS}
    positive_control = _delta_nll_for_synthetic(feats, y_xor, r, rng)

    # 剂量-反应：a_syn 独立随机；以概率 α 用 a_syn 覆盖标签
    a_syn = {s: rng.integers(0, 2, sizes[s]).astype(np.int64) for s in SPLITS}
    dose_response = {}
    for alpha in (0.0, 0.02, 0.05, 0.1, 0.2):
        y_alpha = {}
        for s in SPLITS:
            replace = rng.random(sizes[s]) < alpha
            y_s = feats[s]["label"].copy()
            y_s[replace] = a_syn[s][replace]
            y_alpha[s] = y_s.astype(np.int64)
        dose_response[f"alpha={alpha}"] = _delta_nll_for_synthetic(feats, y_alpha, a_syn, rng)

    return {"positive_control_xor": positive_control, "dose_response": dose_response}


# ============================================================
# 编排
# ============================================================
def aggregate(values: list[float]) -> dict[str, float]:
    """跨 seed 聚合：返回 mean / std。"""
    arr = np.array(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def main() -> None:
    """抽取特征（缺则补）→ 跑 E0–E3 → 落 JSON + 打印可读小结。"""
    print(f"Device: {DEVICE}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ---------- Stage A：特征抽取（GPU）----------
    ensure_features(SEEDS, cfg)

    # ---------- Stage B：分析（每 seed 跑，最后聚合）----------
    attr_specs = ["skin", "skin_bin"]
    per_seed: dict[int, dict] = {}

    for seed in SEEDS:
        print(f"\n{'=' * 70}\n分析 backbone seed={seed}\n{'=' * 70}", flush=True)
        feats = load_features(seed)
        rng = np.random.default_rng(seed)
        te = feats["test"]

        # E0
        e0 = {
            "marginal_mi_bits_skin": marginal_mi_bits(te["label"], te["skin"]),
            "decodability_skin": multiclass_decodability(feats),
        }
        d = e0["decodability_skin"]
        print(f"[E0] skin  I(Y;skin)={e0['marginal_mi_bits_skin']:.4f} bits | "
              f"I(skin;X)>= {d['I(skin;X)_lowerbound_bits']:.3f} bits "
              f"(macro-AUC {d['macro_ovr_auc']:.3f}, top1 {d['top1_acc']:.3f})")

        # E1
        e1 = {}
        for spec in attr_specs:
            e1[spec] = conditional_mi_gain(feats, spec, rng=rng)
            g = e1[spec]["int"]
            print(f"[E1] {spec:8s} ΔNLL_int={g['delta_nll_nats']:.5f} nats "
                  f"CI[{g['ci95_nats'][0]:.5f},{g['ci95_nats'][1]:.5f}] "
                  f"| minority-weighted={g['minority_weighted_nats']:.5f}")

        # E2
        e2 = stratified_skin_effect(te["logit"], te["label"], te["skin"])
        print(f"[E2] skin  aggχ²={e2['agg_chi2']:.2f}(df={e2['agg_df']}) agg_p={e2['agg_p']:.3g} "
              f"| skin_bin CMH_p={e2['cmh_p']:.3g} MH-OR={e2['mh_odds_ratio']:.3f}")

        per_seed[seed] = {"E0": e0, "E1": e1, "E2": e2}

    # E3 灵敏度标定：seed 42 一份即可（灵敏度是估计器+数据规模属性，单 backbone 足够）
    print(f"\n{'=' * 70}\nE3 灵敏度标定（seed 42）\n{'=' * 70}", flush=True)
    e3 = sensitivity_calibration(load_features(42))
    pc = e3["positive_control_xor"]
    print(f"[E3] 正控制 Y XOR r: ΔNLL={pc['delta_nll_nats']:.4f} nats "
          f"CI[{pc['ci95_nats'][0]:.4f},{pc['ci95_nats'][1]:.4f}] detected={pc['detected']}")
    for k, v in e3["dose_response"].items():
        print(f"[E3] {k:12s} ΔNLL={v['delta_nll_nats']:.5f} "
              f"CI[{v['ci95_nats'][0]:.5f},{v['ci95_nats'][1]:.5f}] detected={v['detected']}")

    # 跨 seed 聚合 E1 的 int ΔNLL（全局 + 向少数群加权）作为头条
    headline = {}
    for spec in attr_specs:
        headline[spec] = {
            "delta_nll_int_nats": aggregate([per_seed[s]["E1"][spec]["int"]["delta_nll_nats"] for s in SEEDS]),
            "minority_weighted_nats": aggregate([per_seed[s]["E1"][spec]["int"]["minority_weighted_nats"] for s in SEEDS]),
            "delta_nll_add_nats": aggregate([per_seed[s]["E1"][spec]["add"]["delta_nll_nats"] for s in SEEDS]),
        }

    out = {
        "config": {"seeds": list(SEEDS), "l2_grid": list(L2_GRID), "feature_dim": FEATURE_DIM,
                   "task": "malignant_binary", "attribute": "skin (Fitzpatrick I-VI, 6 groups)"},
        "headline_I(Y;A|X)_estimate": headline,
        "per_seed": per_seed,
        "E3_sensitivity": e3,
    }
    out_path = OUTPUT_DIR / "conditional_mi.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_path}")

    print(f"\n{'=' * 70}\n头条：I(Y;skin|X) 估计（int 模型 ΔNLL，跨 {len(SEEDS)} seed 均值±std，nats）\n{'=' * 70}")
    for spec in attr_specs:
        h = headline[spec]
        print(f"  {spec:8s} 全局 {h['delta_nll_int_nats']['mean']:+.5f} ± {h['delta_nll_int_nats']['std']:.5f} "
              f"| 少数群加权 {h['minority_weighted_nats']['mean']:+.5f} ± {h['minority_weighted_nats']['std']:.5f}")


if __name__ == "__main__":
    main()
