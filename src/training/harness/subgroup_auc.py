"""
完整子群 AUC 向量（比较协议 A0.2 之一）
=====================================
比较协议 §3 要求「每候选必须记录**完整子群 AUC 向量**（非仅 worst-case 标量）供 Pareto 使用」。
本模块把各数据集的子群 AUC 计算统一成一个**固定 schema、等长、可对齐**的向量，
供 A0.2 的 val 日志（val_log.py）记录、A1 的 Minimax-Pareto 选择器消费。

**与 worst-case 的一致性约束（本模块的正确性核心）**：
    向量所含子群 = 各数据集 `*_fairness.worst_case_auc` 所 min 的那批子群，因此恒有
        min(向量中非 None 的 AUC) == worst_case_auc(...)
    self-test（`python -m src.training.harness.subgroup_auc`）对四种 schema 逐一断言此等式，
    保证本模块与既有 fairness 模块**不发生定义漂移**（子群集合的单一事实来源仍是 worst_case_auc）。

**固定 schema（区别于 worst_case_auc 用 np.unique）**：子群键从各 fairness 模块导出的 name
字典枚举，而非从数据 `np.unique` 推断。这样即使某子群在某 val 划分里为空 / 单一类别（AUC=None），
其**键仍然存在**，各候选（不同 lr/wd/seed）产出的向量长度与键顺序完全一致——这是 Pareto 前沿
逐维比较的前提。在真实固定 val 集上两种枚举重合（各组均有样本），self-test 已覆盖。

各数据集向量维度：
  mimic / chexpert : Sex(2)+Race(2)+Age(2) 边缘 + Sex×Race×Age(8) = 14
  ham10000         : Sex(2, 含 age=-1) + Age(4, 排 -1) + Sex×Age(8, 排 -1) = 14
  fitzpatrick      : Skin(6) = 6
  papila           : Sex(2)+Age(2) 边缘 + Sex×Age(4) = 8
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

# 复用各 fairness 模块的 name 字典（子群命名的单一事实来源）与 worst_case_auc（一致性断言用）
from src.utils.mimic_fairness import (
    MIMIC_SEX_NAMES, MIMIC_RACE_NAMES, MIMIC_AGE_NAMES,
    worst_case_auc as mimic_worst_case_auc,
)
from src.utils.ham10000_fairness import (
    HAM_SEX_NAMES, HAM_AGE_NAMES,
    worst_case_auc as ham_worst_case_auc,
)
from src.utils.fitzpatrick_fairness import (
    FITZ_SKIN_NAMES,
    worst_case_auc as fitz_worst_case_auc,
)
from src.utils.papila_fairness import (
    PAPILA_SEX_NAMES, PAPILA_AGE_NAMES,
    worst_case_auc as papila_worst_case_auc,
)

# mimic / chexpert 共享同一 fairness 口径（CheXpert 复用 MIMIC 模块）
DATASETS: tuple[str, ...] = ("mimic", "chexpert", "ham10000", "fitzpatrick", "papila")

# 每个数据集子群向量的固定维度（供 val_log / Pareto 侧断言对齐）
SUBGROUP_VECTOR_DIM: dict[str, int] = {
    "mimic": 14, "chexpert": 14, "ham10000": 14, "fitzpatrick": 6, "papila": 8,
}


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """单一子群内的 AUC；子群为空或只含一种类别时返回 None（与各 fairness 模块口径一致）。"""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_score)


def _auc_n(y_true: np.ndarray, y_score: np.ndarray, mask: np.ndarray) -> tuple[float | None, int]:
    """给定布尔掩码，返回该子群的 (AUC 或 None, 样本数)。"""
    n = int(mask.sum())
    if n == 0:
        return None, 0
    return _safe_auc(y_true[mask], y_score[mask]), n


def _as_arrays(
    y_true: np.ndarray, y_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """标签转 int、分数转 float（与 fairness 模块预处理一致）。"""
    return np.asarray(y_true).astype(int), np.asarray(y_score).astype(float)


def _mimic_vector(
    y_true: np.ndarray, y_score: np.ndarray,
    sex: np.ndarray, race: np.ndarray, age: np.ndarray,
) -> "OrderedDict[str, tuple[float | None, int]]":
    """MIMIC / CheXpert 子群向量：Sex/Race/Age 边缘 + Sex×Race×Age 三阶交叉（共 14 项）。"""
    y_true, y_score = _as_arrays(y_true, y_score)
    sex = np.asarray(sex).astype(int)
    race = np.asarray(race).astype(int)
    age = np.asarray(age).astype(int)

    vec: "OrderedDict[str, tuple[float | None, int]]" = OrderedDict()
    # 边缘轴（固定枚举 name 字典的键，不依赖数据取值）
    for s, sname in MIMIC_SEX_NAMES.items():
        vec[f"sex:{sname}"] = _auc_n(y_true, y_score, sex == s)
    for r, rname in MIMIC_RACE_NAMES.items():
        vec[f"race:{rname}"] = _auc_n(y_true, y_score, race == r)
    for a, aname in MIMIC_AGE_NAMES.items():
        vec[f"age:{aname}"] = _auc_n(y_true, y_score, age == a)
    # Sex×Race×Age 三阶交叉（8 项，与 worst_case_auc 的交叉口径一致）
    for s, sname in MIMIC_SEX_NAMES.items():
        for r, rname in MIMIC_RACE_NAMES.items():
            for a, aname in MIMIC_AGE_NAMES.items():
                key = f"sex:{sname}|race:{rname}|age:{aname}"
                vec[key] = _auc_n(y_true, y_score, (sex == s) & (race == r) & (age == a))
    return vec


def _ham_vector(
    y_true: np.ndarray, y_score: np.ndarray,
    sex: np.ndarray, age_group: np.ndarray,
) -> "OrderedDict[str, tuple[float | None, int]]":
    """HAM10000 子群向量：Sex 边缘(含 age=-1) + Age 边缘(排 -1) + Sex×Age(排 -1)，共 14 项。"""
    y_true, y_score = _as_arrays(y_true, y_score)
    sex = np.asarray(sex).astype(int)
    age_group = np.asarray(age_group).astype(int)

    vec: "OrderedDict[str, tuple[float | None, int]]" = OrderedDict()
    # Sex 边缘：与 worst_case_auc 一致，使用全部样本（含 age_group==-1）
    for s, sname in HAM_SEX_NAMES.items():
        vec[f"sex:{sname}"] = _auc_n(y_true, y_score, sex == s)
    # Age 边缘：仅 age_group∈{0..3}（HAM_AGE_NAMES 不含 -1，排除组）
    for a, aname in HAM_AGE_NAMES.items():
        vec[f"age:{aname}"] = _auc_n(y_true, y_score, age_group == a)
    # Sex×Age 交叉（8 项，age 取 0..3，隐式排除 -1）
    for s, sname in HAM_SEX_NAMES.items():
        for a, aname in HAM_AGE_NAMES.items():
            vec[f"sex:{sname}|age:{aname}"] = _auc_n(
                y_true, y_score, (sex == s) & (age_group == a)
            )
    return vec


def _fitzpatrick_vector(
    y_true: np.ndarray, y_score: np.ndarray, skin: np.ndarray,
) -> "OrderedDict[str, tuple[float | None, int]]":
    """Fitzpatrick17k 子群向量：6 个肤色子群（型 I–VI）。"""
    y_true, y_score = _as_arrays(y_true, y_score)
    skin = np.asarray(skin).astype(int)

    vec: "OrderedDict[str, tuple[float | None, int]]" = OrderedDict()
    for k, kname in FITZ_SKIN_NAMES.items():
        vec[f"skin:{kname}"] = _auc_n(y_true, y_score, skin == k)
    return vec


def _papila_vector(
    y_true: np.ndarray, y_score: np.ndarray,
    sex: np.ndarray, age_group: np.ndarray,
) -> "OrderedDict[str, tuple[float | None, int]]":
    """PAPILA 子群向量：Sex(2)+Age(2) 边缘 + Sex×Age(4) 交叉，共 8 项。"""
    y_true, y_score = _as_arrays(y_true, y_score)
    sex = np.asarray(sex).astype(int)
    age_group = np.asarray(age_group).astype(int)

    vec: "OrderedDict[str, tuple[float | None, int]]" = OrderedDict()
    for s, sname in PAPILA_SEX_NAMES.items():
        vec[f"sex:{sname}"] = _auc_n(y_true, y_score, sex == s)
    for a, aname in PAPILA_AGE_NAMES.items():
        vec[f"age:{aname}"] = _auc_n(y_true, y_score, age_group == a)
    for s, sname in PAPILA_SEX_NAMES.items():
        for a, aname in PAPILA_AGE_NAMES.items():
            vec[f"sex:{sname}|age:{aname}"] = _auc_n(
                y_true, y_score, (sex == s) & (age_group == a)
            )
    return vec


def subgroup_auc_vector(
    dataset: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    sex: np.ndarray | None = None,
    race: np.ndarray | None = None,
    age: np.ndarray | None = None,
    skin: np.ndarray | None = None,
) -> "OrderedDict[str, tuple[float | None, int]]":
    """
    计算给定数据集的完整子群 AUC 向量（固定 schema、等长、可跨候选对齐）。

    子群集合与该数据集 `*_fairness.worst_case_auc` 所 min 的子群一致，故
    `min(非 None 的 AUC) == worst_case_auc(...)`（self-test 已断言）。

    Args:
        dataset: {"mimic","chexpert","ham10000","fitzpatrick","papila"} 之一。
        y_true : [N] 0/1 真实标签。
        y_score: [N] 模型输出（logits 或概率，AUC 对单调变换不敏感）。
        sex/race/age/skin: 各数据集所需的敏感属性数组（按数据集取用）：
            mimic/chexpert 需 sex+race+age；ham10000/papila 需 sex+age（age 为 age_group）；
            fitzpatrick 需 skin。age 对 HAM 传 age_group（-1 为排除组）。

    Returns:
        OrderedDict[子群键 -> (AUC 或 None, 样本数)]，键顺序固定、与数据取值无关。

    Raises:
        ValueError: dataset 不识别，或所需属性缺失。
    """
    def _require(name: str, arr: np.ndarray | None) -> np.ndarray:
        if arr is None:
            raise ValueError(f"数据集 {dataset!r} 需要属性 {name!r}，但未提供。")
        return arr

    if dataset in ("mimic", "chexpert"):
        return _mimic_vector(
            y_true, y_score,
            _require("sex", sex), _require("race", race), _require("age", age),
        )
    if dataset == "ham10000":
        return _ham_vector(y_true, y_score, _require("sex", sex), _require("age", age))
    if dataset == "fitzpatrick":
        return _fitzpatrick_vector(y_true, y_score, _require("skin", skin))
    if dataset == "papila":
        return _papila_vector(y_true, y_score, _require("sex", sex), _require("age", age))
    raise ValueError(f"未知数据集 {dataset!r}，合法取值：{DATASETS}")


def vector_worst_case(vec: "OrderedDict[str, tuple[float | None, int]]") -> float | None:
    """
    从子群 AUC 向量取 worst-case（最小非 None AUC）；无有效子群时返回 None。

    与各 fairness 模块的 worst_case_auc 等价（self-test 已断言），便于日志侧直接从
    向量派生 worst-case，无需重复调用 fairness 模块。
    """
    aucs = [auc for auc, _n in vec.values() if auc is not None]
    return min(aucs) if aucs else None


# ============================================================
# self-test：断言「向量 min == 既有 worst_case_auc」，杜绝定义漂移
# ============================================================
def _selftest() -> None:
    """对四种 schema 生成合成数据，断言向量口径与既有 worst_case_auc 完全一致。"""
    rng = np.random.default_rng(0)

    def close(a: float | None, b: float | None) -> bool:
        if a is None or b is None:
            return a is b
        return abs(a - b) < 1e-9

    # --- MIMIC / CheXpert（sex/race/age 三个二值轴）---
    n = 4000
    y = rng.integers(0, 2, n)
    s = 0.6 * y + rng.standard_normal(n)
    sex = rng.integers(0, 2, n); race = rng.integers(0, 2, n); age = rng.integers(0, 2, n)
    for ds in ("mimic", "chexpert"):
        vec = subgroup_auc_vector(ds, y, s, sex=sex, race=race, age=age)
        assert len(vec) == SUBGROUP_VECTOR_DIM[ds], (ds, len(vec))
        assert close(vector_worst_case(vec), mimic_worst_case_auc(y, s, sex, race, age)), ds
    print(f"  mimic/chexpert: dim={len(vec)}  min(vec)==worst_case ✓")

    # --- HAM10000（sex 二值 + age 4 类含 -1 排除组）---
    n = 3000
    y = rng.integers(0, 2, n)
    s = 0.6 * y + rng.standard_normal(n)
    sex = rng.integers(0, 2, n)
    age_group = rng.choice([-1, 0, 1, 2, 3], size=n, p=[0.024, 0.16, 0.45, 0.29, 0.076])
    vec = subgroup_auc_vector("ham10000", y, s, sex=sex, age=age_group)
    assert len(vec) == SUBGROUP_VECTOR_DIM["ham10000"], len(vec)
    assert close(vector_worst_case(vec), ham_worst_case_auc(y, s, sex, age_group))
    print(f"  ham10000      : dim={len(vec)}  min(vec)==worst_case ✓ (age=-1 已按 fairness 口径排除)")

    # --- Fitzpatrick17k（6 类肤色）---
    n = 3000
    y = rng.integers(0, 2, n)
    s = 0.6 * y + rng.standard_normal(n)
    skin = rng.integers(0, 6, n)
    vec = subgroup_auc_vector("fitzpatrick", y, s, skin=skin)
    assert len(vec) == SUBGROUP_VECTOR_DIM["fitzpatrick"], len(vec)
    assert close(vector_worst_case(vec), fitz_worst_case_auc(y, s, skin))
    print(f"  fitzpatrick   : dim={len(vec)}  min(vec)==worst_case ✓")

    # --- PAPILA（sex + age 均二值）---
    n = 1000
    y = rng.integers(0, 2, n)
    s = 0.6 * y + rng.standard_normal(n)
    sex = rng.integers(0, 2, n); age_group = rng.integers(0, 2, n)
    vec = subgroup_auc_vector("papila", y, s, sex=sex, age=age_group)
    assert len(vec) == SUBGROUP_VECTOR_DIM["papila"], len(vec)
    assert close(vector_worst_case(vec), papila_worst_case_auc(y, s, sex, age_group))
    print(f"  papila        : dim={len(vec)}  min(vec)==worst_case ✓")

    print("subgroup_auc self-test 全部通过 ✓")


if __name__ == "__main__":
    _selftest()
