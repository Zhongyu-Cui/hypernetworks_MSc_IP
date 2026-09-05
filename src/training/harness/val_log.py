"""
候选 val 日志（比较协议 A0.2 之二）
==================================
把每个超参候选（config × seed）在**每个 epoch** 的验证集完整子群 AUC 向量，以结构化
JSONL（每行一个 JSON 对象）追加写入日志文件，供 A1 Minimax-Pareto 选择器与 A2 显著性脚本消费。

**为什么记录每 epoch 而非只记最终**：训练脚本按 val overall AUC（早停判据）与 val worst-case
AUC 两种策略各保存一个 checkpoint。把每 epoch 都落盘后，两种 model selection 各自选中的 epoch
= 对应 val 指标 argmax 的那一行，A1 可**无歧义复现**训练时的 checkpoint 选择（同一 val 指标），
无需在日志里另存标志位。日志侧因此保持「哑记录」，选择逻辑全部留给 A1。

**JSONL 而非 CSV**：子群 AUC 是变长嵌套字段（含 None），JSONL 天然表达 null 与嵌套 dict，
append-only 且可被 pandas `read_json(lines=True)` 直接读。

一条记录（ValRecord）的字段见下方 dataclass；子群向量口径见 subgroup_auc.py。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.training.harness.run import run_id
from src.training.harness.subgroup_auc import subgroup_auc_vector, vector_worst_case


@dataclass
class ValRecord:
    """
    单个候选在单个 epoch 的验证集评估记录（一条 JSONL 行）。

    Attributes:
        dataset       : 数据集名（{"mimic","chexpert","ham10000","fitzpatrick"}）。
        method        : 方法名（"erm"/"hyperhead"/"hyperfusion"/"hyperadapt" 等）。
        config_tag    : 超参配置标识（hparam_grid.HParamConfig.tag，如 "lr1e-04_wd1e-04"）。
        lr, wd        : 该配置的学习率 / 权重衰减（冗余存一份，便于日志自解释）。
        seed          : 随机种子。
        epoch         : 该记录对应的训练 epoch（从 1 起）。
        overall_auc   : 验证集 Overall AUC（早停 / best_overall 选择判据）。
        worst_case_auc: 验证集 worst-case AUC（= 子群向量最小非 None 项；best_worstcase 判据）。
        subgroup_auc  : 完整子群 AUC 向量 {子群键 -> AUC 或 None}，键顺序固定（供 Pareto 对齐）。
        subgroup_n    : 各子群样本数 {子群键 -> n}，供可靠性加权 / 小 cell 甄别。
    """

    dataset: str
    method: str
    config_tag: str
    lr: float
    wd: float
    seed: int
    epoch: int
    overall_auc: float
    worst_case_auc: float | None
    subgroup_auc: dict[str, float | None]
    subgroup_n: dict[str, int]

    def to_json_line(self) -> str:
        """序列化为单行 JSON（None -> null）。"""
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "ValRecord":
        """从解析后的 JSON dict 还原（读日志时用）。"""
        return cls(**d)


def record_from_vector(
    vec: "OrderedDict[str, tuple[float | None, int]]",
    *,
    dataset: str,
    method: str,
    config_tag: str,
    lr: float,
    wd: float,
    seed: int,
    epoch: int,
    overall_auc: float,
) -> ValRecord:
    """
    从已算好的子群 AUC 向量构建 ValRecord（不重复计算向量）。

    训练热循环里 evaluate 已算过向量供 worst-case 选择，直接复用，避免每 epoch 二次计算。

    Args:
        vec: subgroup_auc_vector 的返回（OrderedDict[键 -> (AUC|None, n)]）。
        其余: 见 ValRecord / build_val_record。

    Returns:
        填好子群向量与 worst-case 的 ValRecord。
    """
    subgroup_auc = {k: (None if auc is None else float(auc)) for k, (auc, _n) in vec.items()}
    subgroup_n = {k: int(n) for k, (_auc, n) in vec.items()}
    wc = vector_worst_case(vec)
    return ValRecord(
        dataset=dataset, method=method, config_tag=config_tag,
        lr=float(lr), wd=float(wd), seed=int(seed), epoch=int(epoch),
        overall_auc=float(overall_auc),
        worst_case_auc=(None if wc is None else float(wc)),
        subgroup_auc=subgroup_auc, subgroup_n=subgroup_n,
    )


def build_val_record(
    *,
    dataset: str,
    method: str,
    config_tag: str,
    lr: float,
    wd: float,
    seed: int,
    epoch: int,
    y_true: np.ndarray,
    y_score: np.ndarray,
    overall_auc: float,
    sex: np.ndarray | None = None,
    race: np.ndarray | None = None,
    age: np.ndarray | None = None,
    skin: np.ndarray | None = None,
) -> ValRecord:
    """
    从一次验证集预测构建 ValRecord：内部计算完整子群 AUC 向量与 worst-case。

    训练脚本每 epoch 调用一次即可（配合 append_val_record 落盘）。overall_auc 由调用方传入
    （通常训练循环已算好，避免重复计算）；worst_case 与子群向量在此统一由 subgroup_auc 计算，
    保证与 fairness 模块口径一致。

    Args:
        dataset..seed / epoch: 见 ValRecord 字段。
        y_true, y_score      : [N] 验证集真实标签与模型输出。
        overall_auc          : 验证集 Overall AUC（调用方已算，直接透传）。
        sex/race/age/skin    : 该数据集所需敏感属性（按数据集取用，见 subgroup_auc_vector）。

    Returns:
        填好子群向量与 worst-case 的 ValRecord。
    """
    vec = subgroup_auc_vector(
        dataset, y_true, y_score, sex=sex, race=race, age=age, skin=skin,
    )
    return record_from_vector(
        vec, dataset=dataset, method=method, config_tag=config_tag,
        lr=lr, wd=wd, seed=seed, epoch=epoch, overall_auc=overall_auc,
    )


def default_val_log_path(
    output_dir: Path | str, method: str, config_tag: str, seed: int,
) -> Path:
    """
    该候选（method × config × seed）的默认 val 日志路径：
    `<output_dir>/val_logs/<method>_<config_tag>_seed<seed>.jsonl`。

    一个候选的所有 epoch 记录写入同一文件；A1 用 `val_logs/*.jsonl` 通配收集全部候选。
    output_dir 通常是各数据集的 outputs 根（如 outputs/ham10000）。基名复用 run.run_id，
    与 checkpoint 命名共用同一 run 标识（单一事实来源）。
    """
    log_dir = Path(output_dir) / "val_logs"
    return log_dir / f"{run_id(method, config_tag, seed)}.jsonl"


def append_val_record(log_path: Path | str, record: ValRecord) -> None:
    """
    向 JSONL 日志追加一条记录（不存在则建目录与文件）。append 模式，多 epoch 顺序累积。

    Args:
        log_path: 目标 .jsonl 路径（通常来自 default_val_log_path）。
        record  : 待写入的 ValRecord。
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(record.to_json_line() + "\n")


def read_val_log(log_path: Path | str) -> list[ValRecord]:
    """
    读取一个 JSONL 日志文件，返回全部 ValRecord（按写入顺序，即 epoch 递增）。

    Args:
        log_path: .jsonl 路径。

    Returns:
        ValRecord 列表；空文件返回空列表。
    """
    log_path = Path(log_path)
    records: list[ValRecord] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(ValRecord.from_dict(json.loads(line)))
    return records


# ============================================================
# self-test：构建记录 -> 落盘 -> 读回 round-trip；验证选择可复现
# ============================================================
def _selftest() -> None:
    """合成一个候选的多 epoch 记录，验证 JSONL round-trip 与 argmax 选择可复现。"""
    import tempfile

    rng = np.random.default_rng(1)
    n = 1500
    y = rng.integers(0, 2, n)
    sex = rng.integers(0, 2, n)
    age_group = rng.integers(0, 2, n)

    with tempfile.TemporaryDirectory() as td:
        log_path = default_val_log_path(td, method="erm", config_tag="lr1e-04_wd1e-04", seed=42)
        # 模拟 3 个 epoch，分数信噪比递增 -> overall AUC 递增
        for epoch, snr in enumerate([0.3, 0.6, 0.5], start=1):
            score = snr * y + rng.standard_normal(n)
            from sklearn.metrics import roc_auc_score
            rec = build_val_record(
                dataset="ham10000", method="erm", config_tag="lr1e-04_wd1e-04",
                lr=1e-4, wd=1e-4, seed=42, epoch=epoch,
                y_true=y, y_score=score, overall_auc=float(roc_auc_score(y, score)),
                sex=sex, age=age_group,
            )
            append_val_record(log_path, rec)

        recs = read_val_log(log_path)
        assert len(recs) == 3, len(recs)
        # round-trip：字段与子群向量键完全保留
        assert recs[1].config_tag == "lr1e-04_wd1e-04" and recs[1].seed == 42
        assert len(recs[0].subgroup_auc) == 14  # ham10000 维度
        assert list(recs[0].subgroup_auc.keys()) == list(recs[2].subgroup_auc.keys())  # 键对齐
        # best_overall 选择可复现：epoch 2（snr=0.6）overall 最高
        best_overall = max(recs, key=lambda r: r.overall_auc)
        assert best_overall.epoch == 2, best_overall.epoch
        # None 正确 round-trip（若有单类别子群）
        for r in recs:
            for k, v in r.subgroup_auc.items():
                assert v is None or isinstance(v, float)

    print("val_log self-test 全部通过 ✓（JSONL round-trip + argmax 选择可复现 + 键对齐）")


if __name__ == "__main__":
    _selftest()
