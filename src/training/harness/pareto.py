"""
Minimax-Pareto 模型选择器（比较协议 A1.1 + A1.2）
================================================
比较协议 §3「模型选择：Minimax Pareto」——**在超参搜索产生的候选池（config × seed）上**，
按各子群 AUC 向量取 Pareto 最优候选，再从前沿上选 worst-group AUC 最高者，输出选定配置
（`config_tag`）供确认阶段（5 seed）复跑。本模块消费 A0.2 落盘的 val 日志（val_log.py 的
`val_logs/*.jsonl`），产出选定配置，不触碰训练/checkpoint 本身。

**两步（对应两个复选框）**：
  A1.1 `pareto_front`     —— 子群 AUC 向量的 Pareto 前沿计算（支配关系）。
  A1.2 `select_minimax`   —— 前沿上取 worst-group 最优，输出选定候选/配置。

**候选 → 单向量（复现训练时的 checkpoint 选择）**：每个候选（一个 val 日志文件）含每 epoch
一条记录。训练脚本按两种 model selection 各存一个 checkpoint：`overall`（val Overall AUC argmax）
与 `worstcase`（val worst-case AUC argmax）。本模块用同一 argmax 把候选**塌缩到那一个 epoch 的
子群向量**，故 Pareto 选择所用向量 = 该 selection 策略最终会加载的 checkpoint 在 val 上的表现，
两者口径一致、无歧义（见 `collapse_candidate`）。选择在 `selection` 轴上分别进行（overall /
worstcase 各选一次），因为最终测试各自加载对应 checkpoint。

**None 子群的处理（正确性核心）**：某子群 AUC 为 None 当且仅当它在**固定 val 集**上为空或单一
类别——这只取决于 val 划分，**与模型无关**，因此所有候选共享**同一 None 掩码**。本模块据此只在
「所有候选都非 None」的公共子群键上做支配比较，并断言各候选 None 掩码一致（否则 val 日志被污染，
宁可显式报错也不静默错选）。worst-group 标量则复用 val_log 已存的 `worst_case_auc`（= min 非 None）。

命令行（对某数据集某方法选定配置）：
    python -m src.training.harness.pareto --val-log-dir outputs/ham10000/val_logs \
        --method hyperfusion --selection worstcase
无参数运行 = 跑 self-test。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from src.training.harness.run import Selection, SELECTIONS, parse_run_id
from src.training.harness.val_log import ValRecord, read_val_log

# 支配比较的浮点容差：AUC 差异小于此值视为持平（不构成「严格更优」），杜绝浮点噪声制造伪支配。
DOMINANCE_EPS: float = 1e-9


@dataclass(frozen=True)
class Candidate:
    """
    候选池中的一个候选（一次训练运行 = config × seed），已塌缩到选定 epoch。

    Attributes:
        method     : 方法名（"erm"/"hyperhead"/...）。
        config_tag : 超参配置标识（如 "lr1e-04_wd1e-04"），最终选择输出的就是它。
        seed       : 随机种子。
        epoch      : 该 selection 策略选中的 epoch（val 指标 argmax）。
        selection  : 塌缩所用的 model selection 策略（"overall" / "worstcase"）。
        overall_auc: 选中 epoch 的 val Overall AUC。
        worst_case_auc: 选中 epoch 的 val worst-case AUC（可能 None，若全子群无效）。
        subgroup_auc: 选中 epoch 的完整子群 AUC 向量 {键 -> AUC 或 None}，键顺序固定。
    """

    method: str
    config_tag: str
    seed: int
    epoch: int
    selection: Selection
    overall_auc: float
    worst_case_auc: float | None
    subgroup_auc: dict[str, float | None]

    @property
    def run_id(self) -> str:
        """本候选的 run_id（<method>_<config_tag>_seed<seed>）。"""
        from src.training.harness.run import run_id
        return run_id(self.method, self.config_tag, self.seed)


def collapse_candidate(records: list[ValRecord], selection: Selection) -> Candidate:
    """
    把一个候选的多-epoch val 记录塌缩到「该 selection 策略选中的 epoch」的单条记录。

    复现训练脚本的 checkpoint 选择：`overall` → val Overall AUC argmax 的 epoch；
    `worstcase` → val worst-case AUC argmax 的 epoch。worst-case 为 None 的 epoch 不参与
    worstcase 选择（不能作为 best；与训练侧一致——无有效子群的 epoch 不会被存为 best_worstcase）。

    Args:
        records  : 同一候选（同 method/config/seed）的全部 epoch 记录（read_val_log 的返回）。
        selection: "overall" 或 "worstcase"。

    Returns:
        塌缩后的 Candidate。

    Raises:
        ValueError: records 为空、跨候选混入、或 worstcase 选择下无任何有效 epoch。
    """
    if not records:
        raise ValueError("collapse_candidate 收到空记录列表。")
    if selection not in SELECTIONS:
        raise ValueError(f"未知 selection {selection!r}，合法：{SELECTIONS}")

    # 一致性：所有记录须同属一个候选（同 method/config/seed）
    keys = {(r.method, r.config_tag, r.seed) for r in records}
    if len(keys) != 1:
        raise ValueError(f"collapse_candidate 收到跨候选记录：{sorted(keys)}")

    if selection == "overall":
        best = max(records, key=lambda r: r.overall_auc)
    else:  # worstcase：仅在 worst_case_auc 非 None 的 epoch 中取 argmax
        valid = [r for r in records if r.worst_case_auc is not None]
        if not valid:
            raise ValueError(
                f"候选 {records[0].method}/{records[0].config_tag}/seed{records[0].seed} "
                f"在所有 epoch 的 worst-case AUC 均为 None，无法做 worstcase 选择。"
            )
        best = max(valid, key=lambda r: r.worst_case_auc)

    return Candidate(
        method=best.method, config_tag=best.config_tag, seed=best.seed,
        epoch=best.epoch, selection=selection,
        overall_auc=best.overall_auc, worst_case_auc=best.worst_case_auc,
        subgroup_auc=dict(best.subgroup_auc),
    )


def _common_valid_keys(candidates: list[Candidate]) -> list[str]:
    """
    返回所有候选都非 None 的公共子群键（保持首候选的固定顺序），并断言 None 掩码跨候选一致。

    None 只取决于 val 划分（子群空/单类），与模型无关，故各候选 None 掩码应完全相同；不同则说明
    val 日志被污染（如不同候选用了不同 val 集），显式报错而非静默丢维。

    Args:
        candidates: 待比较候选（均已塌缩）。

    Returns:
        公共有效子群键列表（Pareto 逐维比较的维度集合）。

    Raises:
        ValueError: 候选间子群键集合不一致，或 None 掩码不一致。
    """
    if not candidates:
        raise ValueError("_common_valid_keys 收到空候选列表。")
    ref_keys = list(candidates[0].subgroup_auc.keys())
    ref_none = {k for k, v in candidates[0].subgroup_auc.items() if v is None}
    for cand in candidates[1:]:
        if list(cand.subgroup_auc.keys()) != ref_keys:
            raise ValueError(
                f"候选 {cand.run_id} 子群键集合与首候选不一致（val 日志 schema 漂移）。"
            )
        none_mask = {k for k, v in cand.subgroup_auc.items() if v is None}
        if none_mask != ref_none:
            raise ValueError(
                f"候选 {cand.run_id} 的 None 子群掩码与首候选不一致："
                f"{sorted(none_mask ^ ref_none)}（None 应仅取决于固定 val 集，与模型无关）。"
            )
    return [k for k in ref_keys if k not in ref_none]


def _dominates(a: dict[str, float], b: dict[str, float], keys: list[str]) -> bool:
    """
    Pareto 支配（最大化）：a 在每个子群上 ≥ b，且至少一个子群严格 >（超出 DOMINANCE_EPS）。

    Args:
        a, b: 子群键 -> AUC 的字典（均已保证在 keys 上非 None）。
        keys: 参与比较的公共有效子群键。

    Returns:
        a 是否支配 b。
    """
    all_ge = all(a[k] >= b[k] - DOMINANCE_EPS for k in keys)
    any_gt = any(a[k] > b[k] + DOMINANCE_EPS for k in keys)
    return all_ge and any_gt


def pareto_front(candidates: list[Candidate]) -> list[Candidate]:
    """
    A1.1：计算候选池在子群 AUC 向量上的 Pareto 前沿（非被支配集）。

    支配定义见 `_dominates`（子群 AUC 逐维最大化）。比较维度为所有候选的公共有效子群键
    （None 维已按 `_common_valid_keys` 一致性剔除）。

    Args:
        candidates: 已塌缩的候选列表（长度 ≥ 1）。

    Returns:
        前沿候选子集（不被任何其他候选支配），保持输入相对顺序。
    """
    if not candidates:
        raise ValueError("pareto_front 收到空候选列表。")
    keys = _common_valid_keys(candidates)
    vecs = [{k: c.subgroup_auc[k] for k in keys} for c in candidates]  # type: ignore[misc]

    front: list[Candidate] = []
    for i, cand in enumerate(candidates):
        dominated = any(
            j != i and _dominates(vecs[j], vecs[i], keys)
            for j in range(len(candidates))
        )
        if not dominated:
            front.append(cand)
    return front


@dataclass
class ParetoSelection:
    """
    Minimax-Pareto 选择结果（A1.2 输出）。

    Attributes:
        selected     : 选定候选（其 config_tag 即确认阶段要复跑的配置）。
        front        : Pareto 前沿全部候选（供审查）。
        selection    : 本次选择所用的 model selection 策略。
        n_candidates : 候选池大小。
    """

    selected: Candidate
    front: list[Candidate] = field(default_factory=list)
    selection: Selection = "worstcase"
    n_candidates: int = 0

    def describe(self) -> str:
        """人类可读的选择摘要（选定配置 + 前沿概览）。"""
        lines = [
            f"[Minimax-Pareto | selection={self.selection}] "
            f"候选池 {self.n_candidates} 个 → 前沿 {len(self.front)} 个",
            f"  选定配置: {self.selected.config_tag}  "
            f"(seed{self.selected.seed}, epoch{self.selected.epoch}, "
            f"worst={_fmt(self.selected.worst_case_auc)}, overall={self.selected.overall_auc:.4f})",
            "  前沿候选:",
        ]
        for c in sorted(self.front, key=lambda x: (_key_wc(x), x.overall_auc), reverse=True):
            mark = " ←选定" if c.run_id == self.selected.run_id else ""
            lines.append(
                f"    {c.config_tag} seed{c.seed}: "
                f"worst={_fmt(c.worst_case_auc)} overall={c.overall_auc:.4f}{mark}"
            )
        return "\n".join(lines)


def _key_wc(c: Candidate) -> float:
    """worst-case 排序键：None 视为 -inf（无有效子群的候选排最后）。"""
    return float("-inf") if c.worst_case_auc is None else c.worst_case_auc


def _fmt(v: float | None) -> str:
    """AUC 格式化（None → 'None'）。"""
    return "None" if v is None else f"{v:.4f}"


def select_minimax(candidates: list[Candidate]) -> ParetoSelection:
    """
    A1.2：先取 Pareto 前沿，再在前沿上选 worst-group AUC 最高者（minimax），
    平手用 Overall AUC 破。

    Args:
        candidates: 已塌缩的候选列表（同一 selection 策略；长度 ≥ 1）。

    Returns:
        ParetoSelection（含选定候选与前沿）。

    Raises:
        ValueError: 候选池为空。
    """
    if not candidates:
        raise ValueError("select_minimax 收到空候选列表。")
    selection = candidates[0].selection
    front = pareto_front(candidates)
    # minimax：worst-case 最高；平手用 overall 破（worst-case=None 者以 -inf 垫底）
    selected = max(front, key=lambda c: (_key_wc(c), c.overall_auc))
    return ParetoSelection(
        selected=selected, front=front, selection=selection, n_candidates=len(candidates),
    )


def load_candidates(
    val_log_dir: Path | str,
    method: str,
    selection: Selection,
    *,
    seeds: tuple[int, ...] | None = None,
) -> list[Candidate]:
    """
    从 val 日志目录加载某方法的候选池，并按 selection 策略各自塌缩为 Candidate。

    扫描 `<val_log_dir>/*.jsonl`，按 run_id 反解出 (method, config_tag, seed)，只保留目标
    method（可选再按 seeds 过滤，如只取搜索阶段 seed=42）的日志，逐文件塌缩。

    Args:
        val_log_dir: 存放 `*.jsonl` 的目录（如 outputs/ham10000/val_logs）。
        method     : 目标方法名。
        selection  : 塌缩所用 model selection 策略。
        seeds      : 若给定，只纳入这些 seed（超参搜索用 (42,)）；None 表示全纳入。

    Returns:
        候选列表（每个 .jsonl 一个候选）。

    Raises:
        FileNotFoundError: 目录不存在或该 method 下无匹配日志。
    """
    val_log_dir = Path(val_log_dir)
    if not val_log_dir.is_dir():
        raise FileNotFoundError(f"val 日志目录不存在：{val_log_dir}")

    candidates: list[Candidate] = []
    for path in sorted(val_log_dir.glob("*.jsonl")):
        try:
            m, _config_tag, seed = parse_run_id(path.stem)
        except ValueError:
            continue  # 非 run_id 命名的文件，跳过
        if m != method:
            continue
        if seeds is not None and seed not in seeds:
            continue
        records = read_val_log(path)
        if not records:
            continue
        candidates.append(collapse_candidate(records, selection))

    if not candidates:
        raise FileNotFoundError(
            f"{val_log_dir} 下未找到 method={method!r}"
            + (f"、seeds∈{seeds}" if seeds is not None else "")
            + " 的 val 日志。"
        )
    return candidates


def select_config(
    val_log_dir: Path | str,
    method: str,
    selection: Selection = "worstcase",
    *,
    seeds: tuple[int, ...] | None = None,
) -> ParetoSelection:
    """
    端到端：从 val 日志目录为某方法选出 Minimax-Pareto 配置（A1.1+A1.2 组合入口）。

    Args:
        val_log_dir: val 日志目录。
        method     : 目标方法名。
        selection  : model selection 策略（默认 worstcase，与公平性目标一致）。
        seeds      : 纳入的 seed 集（搜索阶段传 (42,)）；None 全纳入。

    Returns:
        ParetoSelection，`.selected.config_tag` 即确认阶段要复跑的配置。
    """
    candidates = load_candidates(val_log_dir, method, selection, seeds=seeds)
    return select_minimax(candidates)


# ============================================================
# self-test：支配关系 + None 掩码一致性 + minimax 选择 + 端到端
# ============================================================
def _mk_candidate(
    config_tag: str, sub: dict[str, float | None], *, seed: int = 42,
    selection: Selection = "worstcase", overall: float | None = None, method: str = "erm",
) -> Candidate:
    """构造合成候选（worst_case = min 非 None，overall 默认取子群均值）。"""
    valid = [v for v in sub.values() if v is not None]
    wc = min(valid) if valid else None
    ov = overall if overall is not None else (sum(valid) / len(valid) if valid else 0.0)
    return Candidate(
        method=method, config_tag=config_tag, seed=seed, epoch=1, selection=selection,
        overall_auc=ov, worst_case_auc=wc, subgroup_auc=dict(sub),
    )


def _selftest() -> None:
    """覆盖：支配判定、前沿、minimax 选择、None 一致性断言、collapse、端到端。"""
    import tempfile
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from src.training.harness.val_log import build_val_record, append_val_record, default_val_log_path

    # --- _dominates：严格支配 / 持平不算 / 混合不支配 ---
    keys = ["a", "b", "c"]
    assert _dominates({"a": .9, "b": .9, "c": .9}, {"a": .8, "b": .8, "c": .8}, keys)
    assert not _dominates({"a": .9, "b": .9, "c": .9}, {"a": .9, "b": .9, "c": .9}, keys)  # 持平
    assert not _dominates({"a": .9, "b": .7, "c": .9}, {"a": .8, "b": .8, "c": .8}, keys)  # 混合
    assert _dominates({"a": .9, "b": .8, "c": .8}, {"a": .8, "b": .8, "c": .8}, keys)      # ≥ 且一维 >

    # --- 前沿 + minimax：构造 4 个候选 ---
    # cA、cB 均被 cD 支配（各维都 ≤ 且 cD 有严格更优维）；cC、cD 互不支配（cC 的 g1 更强、cD 更均衡）
    cA = _mk_candidate("lrA", {"g1": .70, "g2": .70, "g3": .70})   # 被 cB、cD 支配
    cB = _mk_candidate("lrB", {"g1": .80, "g2": .80, "g3": .80})   # 被 cD 支配（cD 各维 ≥ 且严格更优）
    cC = _mk_candidate("lrC", {"g1": .95, "g2": .78, "g3": .90})   # g1 强、g2 弱(0.78)
    cD = _mk_candidate("lrD", {"g1": .82, "g2": .85, "g3": .81})   # 更均衡、worst=0.81 最高
    cands = [cA, cB, cC, cD]
    front = pareto_front(cands)
    front_tags = {c.config_tag for c in front}
    assert "lrA" not in front_tags and "lrB" not in front_tags, "cA/cB 被 cD 支配，不应在前沿"
    assert front_tags == {"lrC", "lrD"}, front_tags
    sel = select_minimax(cands)
    # minimax：前沿上 worst-case 最高者 = cD(0.81) > cC(0.78)
    assert sel.selected.config_tag == "lrD", sel.selected.config_tag

    # --- minimax 平手用 overall 破 ---
    t1 = _mk_candidate("t1", {"g1": .80, "g2": .90, "g3": .95}, overall=0.88)  # worst=0.80
    t2 = _mk_candidate("t2", {"g1": .80, "g2": .82, "g3": .83}, overall=0.82)  # worst=0.80
    # 两者互不支配（t1 在 g2/g3 更强，g1 持平；t2 无一维严格超 t1 → t1 支配 t2? 检查）
    # t1 各维 ≥ t2 且 g2/g3 严格 > → t1 支配 t2，故前沿只剩 t1
    st = select_minimax([t1, t2])
    assert st.selected.config_tag == "t1", st.selected.config_tag

    # --- None 子群：一致掩码可比 ---
    n1 = _mk_candidate("n1", {"g1": .80, "g2": None, "g3": .90})
    n2 = _mk_candidate("n2", {"g1": .85, "g2": None, "g3": .88})
    fr = pareto_front([n1, n2])          # 仅在 g1/g3 上比较，互不支配
    assert len(fr) == 2, len(fr)
    # None 掩码不一致 → 报错
    n3 = _mk_candidate("n3", {"g1": .85, "g2": .70, "g3": .88})
    try:
        pareto_front([n1, n3])
        raise AssertionError("None 掩码不一致应报错")
    except ValueError:
        pass

    # --- collapse_candidate：overall vs worstcase 选不同 epoch ---
    rng = np.random.default_rng(3)
    n = 1200
    y = rng.integers(0, 2, n); sex = rng.integers(0, 2, n); age = rng.integers(0, 2, n)
    recs = []
    for epoch, (snr_ov, snr_wc) in enumerate([(0.7, 0.2), (0.4, 0.6)], start=1):
        # epoch1 overall 高、epoch2 worst 高（用两套分数近似）
        score = snr_ov * y + rng.standard_normal(n)
        rec = build_val_record(
            dataset="papila", method="erm", config_tag="lr1e-04_wd1e-04", lr=1e-4, wd=1e-4,
            seed=42, epoch=epoch, y_true=y, y_score=score,
            overall_auc=float(roc_auc_score(y, score)), sex=sex, age=age,
        )
        # 人工改写 worst_case / overall 使两 selection 分歧：epoch1 overall 高、epoch2 worst 高
        rec = ValRecord(**{**rec.__dict__, "worst_case_auc": (0.60 if epoch == 1 else 0.75),
                           "overall_auc": (0.90 if epoch == 1 else 0.70)})
        recs.append(rec)
    c_ov = collapse_candidate(recs, "overall")
    c_wc = collapse_candidate(recs, "worstcase")
    assert c_ov.epoch == 1 and c_wc.epoch == 2, (c_ov.epoch, c_wc.epoch)

    # --- 端到端 load_candidates + select_config（真实落盘 3 个候选）---
    with tempfile.TemporaryDirectory() as td:
        val_dir = Path(td) / "val_logs"
        for tag, base in [("lr3e-05_wd1e-04", 0.3), ("lr1e-04_wd1e-04", 0.6), ("lr3e-04_wd1e-04", 0.45)]:
            score = base * y + rng.standard_normal(n)
            lr = float(tag.split("_")[0][2:]); wd = float(tag.split("wd")[1])
            rec = build_val_record(
                dataset="papila", method="hyperfusion", config_tag=tag, lr=lr, wd=wd,
                seed=42, epoch=1, y_true=y, y_score=score,
                overall_auc=float(roc_auc_score(y, score)), sex=sex, age=age,
            )
            append_val_record(default_val_log_path(td, "hyperfusion", tag, 42), rec)
        sel_e2e = select_config(val_dir, "hyperfusion", "worstcase", seeds=(42,))
        assert sel_e2e.n_candidates == 3, sel_e2e.n_candidates
        assert sel_e2e.selected.method == "hyperfusion"
        # 只取 seed=42（搜索阶段），其它 method 不混入
        assert all(c.seed == 42 for c in sel_e2e.front)

    print("pareto self-test 全部通过 ✓（支配/前沿/minimax/平手破/None一致性/collapse/端到端）")


def _build_arg_parser() -> argparse.ArgumentParser:
    """命令行入口的参数解析器。"""
    p = argparse.ArgumentParser(
        description="Minimax-Pareto 模型选择（A1）：从 val 日志为某方法选定超参配置"
    )
    p.add_argument("--val-log-dir", type=str, default=None,
                   help="val 日志目录（如 outputs/ham10000/val_logs）；缺省则跑 self-test。")
    p.add_argument("--method", type=str, default=None, help="目标方法名（erm/hyperhead/...）。")
    p.add_argument("--selection", type=str, default="worstcase", choices=list(SELECTIONS),
                   help="model selection 策略（默认 worstcase）。")
    p.add_argument("--seeds", type=int, nargs="*", default=[42],
                   help="纳入的 seed（默认仅 42 = 搜索阶段）；传空 [] 表示全纳入。")
    return p


def main() -> None:
    """命令行：给定 val 日志目录与方法，打印 Minimax-Pareto 选定配置；无参数则 self-test。"""
    args = _build_arg_parser().parse_args()
    if args.val_log_dir is None or args.method is None:
        _selftest()
        return
    seeds = tuple(args.seeds) if args.seeds else None
    sel = select_config(args.val_log_dir, args.method, args.selection, seeds=seeds)
    print(sel.describe())
    print(f"\nSELECTED_CONFIG={sel.selected.config_tag}")


if __name__ == "__main__":
    main()
