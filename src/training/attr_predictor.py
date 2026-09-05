"""
子群分类器 g_φ：冻结 ImageNet 特征 + 多项 logistic probe
=========================================================
方案见 `docs/predicted_attribute_hyperadapt_plan.md` §3.1–3.2。g 与 HyperAdapt 构成**单一方法**
（顺序两阶段 pipeline），因此：

  · g 只在该 fold 的 **train** 上拟合一次，随后对 train/val/test 全部前向出 p̂
    （train 侧即 in-sample 输出，**不做交叉拟合**——p̂ 不是外部无偏估计量，而是 pipeline 内部表示）；
  · g 的容量被**锁死为低容量 probe**（冻结 ImageNet ResNet-18 的 512-d 特征 + 带 L2 的多项 logistic
    回归）：in-sample 与 out-of-sample 的锐度差由 g 的容量决定，低容量 probe 让训练/测试的条件输入
    分布几乎不漂移，也让本方法与「GT 训练 + 预测测试」保持可区分；
  · 温度校准（Guo et al. 2017，单标量 T，在 **val** 上最小化 NLL）只改变 p̂ 的锐度、不改 argmax。
    这一步决定 soft 臂与 hard 臂是否真有差别——未校准的过自信 p̂ 会让二者退化为同一个方法。

特征与 fold 无关（backbone 冻结），故**全数据集只抽一次特征并缓存**，每折只重拟合 probe（CPU 秒级）。

诊断指标（写入 metrics.json，只记录、不作 gate）：逐 split 的 accuracy / macro-F1 / 平均熵 / ECE。
train 与 test 的 accuracy、熵差越大，说明 in-sample p̂ 越尖锐、本方法越向「GT 条件训练」退化，
读结果时必须据此定调。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# probe 的 L2 强度：**固定常数，不进超参搜索**（方案 §5：避免 g 变成第二条 forking path）。
# sklearn 的 C 是正则强度的倒数，1.0 为其默认值。
PROBE_C: float = 1.0
PROBE_MAX_ITER: int = 2000


# ============================================================
# 特征抽取
# ============================================================
def load_rgb(path: str) -> Image.Image:
    """自然 RGB 图像 loader（Fitzpatrick preproc_224x224 / HAM images）。"""
    return Image.open(path).convert("RGB")


def load_cxr_16bit(path: str) -> Image.Image:
    """
    CXR7-1M 的 16-bit PNG loader（与 `MIMICCXRDataset.__getitem__` 逐行一致）。

    CXR7-1M 的 PNG 是 16-bit（PIL mode 'I;16'，像素 0~65535）。**禁用 `.convert("L")`**——
    PIL 的 I;16→L 转换有损，会把 ~3 万灰度级压成 10~120 级、使 AUC 崩到 ~0.67。正确做法是
    按真实动态范围逐图 min-max 归一化后转 8-bit 灰度，交由下游 transform 的 Grayscale(3) 复制通道。

    Args:
        path: PNG 绝对路径。

    Returns:
        mode="L" 的 PIL Image。
    """
    arr = np.array(Image.open(path)).astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return Image.fromarray((arr * 255.0).astype(np.uint8), mode="L")


class _ImageKeyDataset(Dataset):
    """
    仅用于特征抽取的极简 Dataset：按路径读图 + 确定性 transform，不返回标签。

    Args:
        image_paths: [N] 图像绝对路径。
        transform  : 作用在 PIL Image 上的**评估**（确定性）transform；抽特征不可用随机增强。
        loader     : path -> PIL Image 的加载函数（自然图像用 load_rgb，CXR 用 load_cxr_16bit）。
    """

    def __init__(self, image_paths: np.ndarray, transform, loader=load_rgb) -> None:
        self.image_paths = image_paths
        self.transform = transform
        self.loader = loader

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.transform(self.loader(self.image_paths[idx]))


def build_feature_extractor() -> nn.Module:
    """
    构建冻结的 ImageNet 预训练 ResNet-18 特征提取器（fc 换成 Identity，输出 512-d）。

    Returns:
        eval 模式、requires_grad=False、已搬到 DEVICE 的 nn.Module。
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()          # 去掉 1000 类头，直接取 512-d pooled 特征
    for param in model.parameters():
        param.requires_grad = False
    return model.eval().to(DEVICE)


@torch.no_grad()
def extract_features(
    image_paths: np.ndarray,
    transform,
    batch_size: int = 128,
    num_workers: int = 8,
    image_loader=load_rgb,
) -> np.ndarray:
    """
    对给定图像路径批量抽取 512-d 冻结特征。

    Args:
        image_paths : [N] 图像路径。
        transform   : 评估 transform（确定性）。
        batch_size  : 前向 batch 大小。
        num_workers : DataLoader 工作进程数。
        image_loader: path -> PIL Image（自然图像 load_rgb / CXR load_cxr_16bit）。

    Returns:
        [N, 512] float32 特征矩阵（行序与 image_paths 一致）。
    """
    model = build_feature_extractor()
    loader = DataLoader(
        _ImageKeyDataset(image_paths, transform, loader=image_loader),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    features: list[np.ndarray] = []
    for images in loader:
        out = model(images.to(DEVICE, non_blocking=True))
        features.append(out.cpu().numpy().astype(np.float32))
    return np.concatenate(features, axis=0)


# ============================================================
# probe + 温度校准
# ============================================================
def _to_logit_matrix(decision: np.ndarray, num_classes: int) -> np.ndarray:
    """
    把 sklearn `decision_function` 的输出统一成 [N, C] logit 矩阵。

    二分类时 sklearn 返回 [N]（正类的 margin），需展开为两列 [0, margin]，
    softmax 后与 predict_proba 一致。

    Args:
        decision   : decision_function 的原始输出。
        num_classes: 类别数。

    Returns:
        [N, C] logit 矩阵。
    """
    if decision.ndim == 1:
        if num_classes != 2:
            raise ValueError(f"一维 decision 只可能来自二分类，但 num_classes={num_classes}")
        return np.stack([np.zeros_like(decision), decision], axis=1)
    return decision


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """数值稳定的按行 softmax（可选温度缩放 logits/T）。"""
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """
    温度标定（Guo et al. 2017）：在给定集合（本方法用 val）上最小化 NLL，拟合单标量 T。

    T 只缩放 logits、不改变 argmax，故只影响 soft 臂，hard 臂不受任何影响。

    Args:
        logits : [N, C] 未校准 logit。
        y_true : [N] 真值类别索引。

    Returns:
        最优温度 T（限制在 [0.05, 20]，越大越平缓）。
    """
    def nll(log_temperature: float) -> float:
        probs = _softmax(logits, temperature=float(np.exp(log_temperature)))
        picked = probs[np.arange(len(y_true)), y_true]
        return float(-np.log(np.clip(picked, 1e-12, None)).mean())

    # 在 log T 上做一维搜索，天然保证 T>0 且尺度对称
    result = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(result.x))


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """
    多类 ECE（按最大概率分箱的置信度-准确率差的加权平均）。

    Args:
        probs  : [N, C] 概率。
        y_true : [N] 真值类别索引。
        n_bins : 置信度分箱数。

    Returns:
        ECE ∈ [0, 1]，越小越校准。
    """
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    correct = (prediction == y_true).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def mean_entropy(probs: np.ndarray) -> float:
    """平均预测熵（nats）。train 与 test 的熵差是 in-sample 尖锐度的直接诊断量。"""
    return float((-probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1).mean())


@dataclass
class ProbeSplitResult:
    """probe 在某个 split 上的输出与诊断。"""

    probs: np.ndarray                       # [N, C] 校准后概率
    y_true: np.ndarray                      # [N] 真值属性
    accuracy: float
    macro_f1: float
    mean_entropy: float
    ece: float

    def to_metrics(self) -> dict:
        """导出为可 JSON 序列化的指标字典（不含逐样本数组）。"""
        return {
            "n": int(len(self.y_true)),
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "mean_entropy": self.mean_entropy,
            "ece": self.ece,
        }


@dataclass
class ProbeFoldResult:
    """某 fold 上 g 的完整产物：三个 split 的 p̂ + 温度 + 诊断。"""

    temperature: float
    num_classes: int
    prior: np.ndarray                       # [C] train 集 p̂ 的均值（const 对照臂用）
    splits: dict[str, ProbeSplitResult] = field(default_factory=dict)

    def to_metrics(self) -> dict:
        """导出该 fold 的指标字典。"""
        return {
            "temperature": self.temperature,
            "num_classes": self.num_classes,
            "prior_from_train_pred": self.prior.tolist(),
            **{name: split.to_metrics() for name, split in self.splits.items()},
        }


def fit_attr_probe(
    features: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    num_classes: int,
    probe_c: float = PROBE_C,
) -> ProbeFoldResult:
    """
    在 train 上拟合 probe，对 train/val/test 出 p̂，并用 val 做温度校准。

    步骤（对应方案 §3.2 的 4 步规程）：
      1. StandardScaler + 多项 logistic 回归在 **train** 上拟合（in-sample，无交叉拟合）；
      2. 对三个 split 前向得未校准 logits；
      3. 在 **val** 上拟合温度 T（val 不参与 probe 拟合，也是既有流程的选择/早停用集）；
      4. 同一 T 施于三个 split，输出校准后 p̂ 与诊断指标。

    Args:
        features   : {"train"/"val"/"test" -> [N_i, D] 特征}。
        labels     : {"train"/"val"/"test" -> [N_i] 真值属性索引}。
        num_classes: 属性类别数。
        probe_c    : logistic 回归的 C（L2 强度的倒数），固定常数。

    Returns:
        ProbeFoldResult。

    Raises:
        KeyError: features/labels 缺少 train 或 val。
    """
    for required in ("train", "val"):
        if required not in features or required not in labels:
            raise KeyError(f"features/labels 必须包含 {required!r} split")

    scaler = StandardScaler().fit(features["train"])
    clf = LogisticRegression(C=probe_c, max_iter=PROBE_MAX_ITER)
    clf.fit(scaler.transform(features["train"]), labels["train"])

    raw_logits = {
        name: _to_logit_matrix(clf.decision_function(scaler.transform(feat)), num_classes)
        for name, feat in features.items()
    }
    temperature = fit_temperature(raw_logits["val"], labels["val"])

    result = ProbeFoldResult(temperature=temperature, num_classes=num_classes,
                             prior=np.zeros(num_classes, dtype=np.float64))
    for name, logits in raw_logits.items():
        probs = _softmax(logits, temperature=temperature).astype(np.float32)
        y_true = labels[name]
        result.splits[name] = ProbeSplitResult(
            probs=probs,
            y_true=y_true,
            accuracy=float((probs.argmax(axis=1) == y_true).mean()),
            macro_f1=float(f1_score(y_true, probs.argmax(axis=1), average="macro")),
            mean_entropy=mean_entropy(probs),
            ece=expected_calibration_error(probs, y_true),
        )
    result.prior = result.splits["train"].probs.mean(axis=0).astype(np.float64)
    return result


# ============================================================
# 落盘 / 读取
# ============================================================
def save_fold_predictions(
    out_dir: Path, split_name: str, keys: np.ndarray,
    axes: dict[str, tuple[np.ndarray, np.ndarray]],
) -> Path:
    """
    落盘某 fold 某 split 的 p̂（支持多个敏感属性轴，如 MIMIC/CheXpert 的 sex/race/age）。

    Args:
        out_dir   : `<outputs>/<ds>/cv5/attr_pred/fold{k}`（自动创建）。
        split_name: "train"/"val"/"test"。
        keys      : [N] 样本唯一键（Fitzpatrick: md5hash；HAM: image_id；CXR: image_path），
                    用于与训练侧按键 join。
        axes      : {轴名 -> ([N, C] 校准后概率, [N] 真值索引)}。

    Returns:
        写出的 .npz 路径。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{split_name}.npz"
    payload: dict[str, np.ndarray] = {}
    for axis, (probs, gt) in axes.items():
        payload[f"prob_{axis}"] = probs
        payload[f"gt_{axis}"] = gt
    np.savez(path, keys=np.asarray(keys, dtype=object), **payload)
    return path


def load_fold_predictions(path: Path, axis: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    读回某 split 的 p̂。

    Args:
        path: `.../fold{k}/{split}.npz`。
        axis: 属性轴名。

    Returns:
        (keys [N], probs [N, C] float32, gt [N] int64)。
    """
    data = np.load(path, allow_pickle=True)
    return (
        data["keys"].astype(str),
        data[f"prob_{axis}"].astype(np.float32),
        data[f"gt_{axis}"].astype(np.int64),
    )
