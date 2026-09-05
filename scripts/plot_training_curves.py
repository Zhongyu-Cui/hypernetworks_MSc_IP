"""
训练曲线绘制（AUC / loss，均值 / per-seed）
============================================
把 C 阶段「选定配置 5-seed 确认训练」的训练曲线统一可视化，供 `docs/results_summary.md` 与
过程检查复用。**选定配置映射（`SELECTED_CONFIGS`）是本脚本的单一事实来源**，与
`docs/comparison_protocol.md` 的 C1.5/C2.5/C3.5/C4.5/C5.5 一致；重训换配置时只改此处。

两种指标、两种口径：
  - `--metric auc`  ：验证集 Overall AUC 与 worst-case AUC，读自 `outputs/<dir>/val_logs/*.jsonl`
                      （harness 逐 epoch 落盘，权威口径）。
  - `--metric loss` ：train / val BCE loss，解析自 `logs/*.log` SLURM stdout（val 日志不含 loss）。
  - `--mode mean`     ：5×2 网格（5 数据集 × 2 指标），每格 4 方法，逐 epoch 对 seed 求均值，
                        只画「全部 5 seed 仍在训练」的 epoch 段（避免尾部薄样本误导）。
  - `--mode per-seed` ：每个 (数据集×方法) 单独一张图，画出全部 5 seed 各自曲线（不平均，
                        画至各自早停长度），输出到子文件夹。

CV 数据集的 run_id seed 与折一一映射（seed=42+fold），故其「5 seed」实为「5 折」，
方差含折间差异，图注与图例已标注。

用法：
    python -m scripts.plot_training_curves --metric auc  --mode mean
    python -m scripts.plot_training_curves --metric loss --mode per-seed
    python -m scripts.plot_training_curves --all           # 4 种组合一次生成

依赖 matplotlib；中文用系统 Noto Serif CJK 字体（实验室机器 / 集群均有）。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.paths import WORK_ROOT

# ============================================================
# 路径与常量
# ============================================================
ROOT = WORK_ROOT
LOG_DIR = ROOT / "logs"
OUT_ROOT = ROOT / "outputs"

# 选定配置（单一事实来源，对齐 comparison_protocol C*.5）。
# 键 = 规范数据集名；与 val_logs 目录名 / SLURM 日志 dataset= 名的映射见 DATASETS。
SELECTED_CONFIGS: dict[str, dict[str, str]] = {
    "ham10000":    {"erm": "lr1e-04_wd1e-04", "hyperhead": "lr1e-04_wd1e-03", "hyperfusion": "lr3e-04_wd1e-03", "hyperadapt": "lr3e-04_wd1e-04"},
    "mimic":       {"erm": "lr3e-04_wd1e-03", "hyperhead": "lr1e-04_wd1e-04", "hyperfusion": "lr1e-04_wd1e-04", "hyperadapt": "lr3e-04_wd1e-04"},
    "chexpert":    {"erm": "lr3e-04_wd1e-03", "hyperhead": "lr3e-05_wd1e-03", "hyperfusion": "lr1e-04_wd1e-04", "hyperadapt": "lr3e-04_wd1e-03"},
    "fitzpatrick": {"erm": "lr1e-04_wd1e-03", "hyperhead": "lr1e-04_wd1e-04", "hyperfusion": "lr3e-04_wd1e-04", "hyperadapt": "lr1e-04_wd1e-03"},
}


@dataclass(frozen=True)
class DatasetInfo:
    """一个数据集的显示名、val_logs 目录名、SLURM 日志 dataset= 名、是否全-CV。"""

    key: str
    display: str
    val_log_dir: str      # outputs/<val_log_dir>/val_logs/（AUC 口径；含 _cxr 后缀）
    log_name: str         # SLURM 日志 Run 头的 dataset= 值（loss 口径；短名）
    is_cv: bool = False   # 全-CV（seed=fold）


DATASETS: tuple[DatasetInfo, ...] = (
    DatasetInfo("ham10000", "HAM10000  (age 有信号 · I>0)", "ham10000", "ham10000"),
    DatasetInfo("mimic", "MIMIC-CXR  (I≈0 · 干净负控制)", "mimic_cxr", "mimic"),
    DatasetInfo("chexpert", "CheXpert  (I<MDE · 不可判定)", "chexpert_cxr", "chexpert"),
    DatasetInfo("fitzpatrick", "Fitzpatrick17k  (I<MDE · 不可判定)", "fitzpatrick", "fitzpatrick"),
)
DS_BY_KEY: dict[str, DatasetInfo] = {d.key: d for d in DATASETS}

METHODS: tuple[str, ...] = ("erm", "hyperhead", "hyperfusion", "hyperadapt")
METHOD_LABEL: dict[str, str] = {
    "erm": "ERM (baseline)", "hyperhead": "HyperHead (浅)",
    "hyperfusion": "HyperFusion (中)", "hyperadapt": "HyperAdapt (深)",
}
METHOD_LABEL_LONG: dict[str, str] = {
    "erm": "ERM (baseline)", "hyperhead": "HyperHead (浅层 HN)",
    "hyperfusion": "HyperFusion (中层 HN)", "hyperadapt": "HyperAdapt (深层 HN)",
}
SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)

# 方法配色（mean 图，4 方法）与 seed 配色（per-seed 图，5 seed）：高分离度 CVD 友好，无红绿对撞
METHOD_COLOR: dict[str, str] = {"erm": "#2a78d6", "hyperhead": "#eda100", "hyperfusion": "#e34948", "hyperadapt": "#4a3aa7"}
METHOD_MARKER: dict[str, str] = {"erm": "o", "hyperhead": "s", "hyperfusion": "^", "hyperadapt": "D"}
SEED_COLOR: dict[int, str] = {42: "#2a78d6", 43: "#eda100", 44: "#e34948", 45: "#4a3aa7", 46: "#1baf7a"}

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e6e2", "#fcfcfb"
_CJK_FONT = Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")


# ============================================================
# 数据读取
# ============================================================
@dataclass
class Curve:
    """单个 run 的曲线数据：epoch → 各指标。缺失指标为 None。"""

    epochs: list[int] = field(default_factory=list)
    overall_auc: list[float] = field(default_factory=list)
    worst_auc: list[float | None] = field(default_factory=list)
    train_loss: list[float | None] = field(default_factory=list)
    val_loss: list[float | None] = field(default_factory=list)


def load_auc_curve(ds: DatasetInfo, method: str, config_tag: str, seed: int) -> Curve | None:
    """
    从 val 日志 JSONL 读取一个 run 的逐 epoch Overall / worst-case AUC。

    Args:
        ds: 数据集信息（用 val_log_dir 定位）。
        method: 方法名。
        config_tag: 超参配置标识（如 "lr1e-04_wd1e-04"）。
        seed: 随机种子（CV 下即折号 42+fold）。

    Returns:
        Curve（仅填 overall_auc / worst_auc），文件不存在时返回 None。
    """
    path = OUT_ROOT / ds.val_log_dir / "val_logs" / f"{method}_{config_tag}_seed{seed}.jsonl"
    if not path.exists():
        return None
    curve = Curve()
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        curve.epochs.append(int(rec["epoch"]))
        curve.overall_auc.append(float(rec["overall_auc"]))
        curve.worst_auc.append(rec.get("worst_case_auc"))
    return curve


# SLURM 日志解析用正则：头部 Run 行 + 逐 epoch 行
_HDR_RE = re.compile(r"Run\s*:\s*dataset=(\S+)\s+method=(\S+)\s+config=(\S+)\s+seed=(\d+)")
_EPOCH_RE = re.compile(r"^Epoch\s+(\d+)/\d+.*train loss=([\d.]+).*val loss=([\d.]+)")


def load_all_loss_curves() -> dict[tuple[str, str, str, int], Curve]:
    """
    扫描全部 SLURM 日志，解析每个 run 的逐 epoch train / val loss。

    一个 (dataset, method, config, seed) 可能有多份日志（搜索 seed42 与确认 seed42 重复、
    semois 故障重跑等）：**confirm 日志优先**；同类日志取 epoch 更全者。据日志头 dataset= 名索引
    （短名，如 "mimic"），与 SELECTED_CONFIGS / DATASetInfo.log_name 对齐。

    Returns:
        {(log_name, method, config_tag, seed): Curve}（仅填 train_loss / val_loss）。
    """
    runs: dict[tuple[str, str, str, int], Curve] = {}
    is_confirm_flag: dict[tuple[str, str, str, int], bool] = defaultdict(bool)

    for path in sorted(LOG_DIR.glob("*.log")):
        is_confirm = "confirm" in path.name
        header: tuple[str, str, str, int] | None = None
        epochs: list[int] = []
        train: list[float] = []
        val: list[float] = []
        for line in path.read_text(errors="ignore").splitlines():
            if header is None:
                m = _HDR_RE.search(line)
                if m:
                    header = (m.group(1), m.group(2), m.group(3), int(m.group(4)))
                continue
            e = _EPOCH_RE.match(line)
            if e:
                epochs.append(int(e.group(1)))
                train.append(float(e.group(2)))
                val.append(float(e.group(3)))
        if header is None or not epochs:
            continue
        # 去重：confirm 覆盖 search；同类取更长者
        prev = runs.get(header)
        take = (
            prev is None
            or (is_confirm and not is_confirm_flag[header])
            or (is_confirm == is_confirm_flag[header] and len(epochs) > len(prev.epochs))
        )
        if take:
            runs[header] = Curve(epochs=epochs, train_loss=list(train), val_loss=list(val))
            is_confirm_flag[header] = is_confirm
    return runs


# ============================================================
# 字体
# ============================================================
def setup_font() -> None:
    """注册系统 Noto Serif CJK 字体，使中文标题/标签正常渲染（缺字体时回退 DejaVu Sans）。"""
    family = ["DejaVu Sans"]
    if _CJK_FONT.exists():
        fm.fontManager.addfont(str(_CJK_FONT))
        family = [fm.FontProperties(fname=str(_CJK_FONT)).get_name(), "DejaVu Sans"]
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.labelcolor": INK, "font.family": family, "axes.unicode_minus": False,
    })


def _style_axis(ax: "plt.Axes") -> None:
    """统一子图样式：浅色 surface、recessive 网格、去顶右脊。"""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(spine == "")
    ax.tick_params(length=3)


# ============================================================
# 聚合（mean 口径）
# ============================================================
def _series_for(curve: Curve, field_name: str) -> tuple[list[int], list[float]]:
    """从 Curve 取某指标的 (epochs, values)，过滤 None。"""
    ys = getattr(curve, field_name)
    pts = [(e, v) for e, v in zip(curve.epochs, ys) if v is not None]
    return [p[0] for p in pts], [p[1] for p in pts]


def _mean_over_seeds(per_seed: list[tuple[list[int], list[float]]]) -> tuple[list[int], list[float]]:
    """
    逐 epoch 对多个 seed 的曲线求均值，只保留「全部 seed 都覆盖」的 epoch 段。

    Args:
        per_seed: 每个 seed 的 (epochs, values)。

    Returns:
        (epochs, mean_values)；无数据时返回空。
    """
    if not per_seed:
        return [], []
    maps = [dict(zip(xs, ys)) for xs, ys in per_seed]
    full = min(max(m) for m in maps)          # 所有 seed 都覆盖的最大 epoch
    xs = list(range(1, full + 1))
    ys = [sum(m[e] for m in maps) / len(maps) for e in xs]
    return xs, ys


# ============================================================
# 绘图：mean 口径 5×2 网格
# ============================================================
def _collect_mean_series(
    metric: str, loss_runs: dict[tuple[str, str, str, int], Curve],
) -> dict[tuple[str, str, int], tuple[list[int], list[float]]]:
    """
    为 mean 图收集每 (数据集, 方法, 列) 的均值曲线。列：AUC=(overall,worst)、loss=(train,val)。

    Returns:
        {(ds_key, method, col_index): (epochs, mean_values)}；col_index ∈ {0,1}。
    """
    cols = ("overall_auc", "worst_auc") if metric == "auc" else ("train_loss", "val_loss")
    out: dict[tuple[str, str, int], tuple[list[int], list[float]]] = {}
    for ds in DATASETS:
        for method in METHODS:
            tag = SELECTED_CONFIGS[ds.key][method]
            # 取各 seed 的 Curve
            curves: list[Curve] = []
            for seed in SEEDS:
                if metric == "auc":
                    c = load_auc_curve(ds, method, tag, seed)
                else:
                    c = loss_runs.get((ds.log_name, method, tag, seed))
                if c:
                    curves.append(c)
            for col, fname in enumerate(cols):
                per_seed = [_series_for(c, fname) for c in curves]
                per_seed = [ps for ps in per_seed if ps[0]]
                out[(ds.key, method, col)] = _mean_over_seeds(per_seed)
    return out


def plot_mean_grid(metric: str, loss_runs: dict[tuple[str, str, str, int], Curve], out_path: Path) -> None:
    """
    绘制 5×2 均值网格（5 数据集 × 2 指标列），每格 4 方法线。

    Args:
        metric: "auc" 或 "loss"。
        loss_runs: 预解析的 loss run 表（metric="auc" 时忽略）。
        out_path: 输出 PNG 路径。
    """
    series = _collect_mean_series(metric, loss_runs)
    col_titles = ("验证集 Overall AUC", "验证集 worst-case AUC") if metric == "auc" \
        else ("训练集 loss (train BCE)", "验证集 loss (val BCE)")
    fig, axes = plt.subplots(5, 2, figsize=(11, 15.5))
    for row, ds in enumerate(DATASETS):
        for col in (0, 1):
            ax = axes[row][col]
            _style_axis(ax)
            for method in METHODS:
                xs, ys = series[(ds.key, method, col)]
                if xs:
                    ax.plot(xs, ys, color=METHOD_COLOR[method], lw=1.8, marker=METHOD_MARKER[method],
                            ms=4.5, mec="white", mew=0.5, zorder=3)
            if col == 0:
                ax.set_ylabel(ds.display, fontsize=10.5, fontweight="bold", labelpad=8)
            ax.set_title(col_titles[col], fontsize=10, color=MUTED, loc="left", pad=4)
            if row == 4:
                ax.set_xlabel("Epoch", fontsize=9.5)

    handles = [Line2D([0], [0], color=METHOD_COLOR[m], lw=2.2, marker=METHOD_MARKER[m], ms=6,
                      mec="white", mew=0.6, label=METHOD_LABEL[m]) for m in METHODS]
    fig.legend(handles=handles, ncol=4, loc="upper center", frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 0.995))
    what = "AUC" if metric == "auc" else "loss"
    fig.suptitle(f"选定配置的 5-seed 确认训练 {what} 曲线（逐 epoch，均值 over 全存活 seed）",
                 fontsize=13, fontweight="bold", y=0.972)
    fig.text(0.5, 0.005, "注：只画全部 5 seed 仍在训练的 epoch 段（早停致各 seed 长度不一）；"
             "CV regime 下 seed=fold，方差含折间差异。", ha="center", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.012, 1, 0.955))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_path}")


# ============================================================
# 绘图：per-seed（每 数据集×方法 一张）
# ============================================================
def plot_per_seed(metric: str, loss_runs: dict[tuple[str, str, str, int], Curve], out_dir: Path) -> None:
    """
    每个 (数据集×方法) 输出一张图，画全部 5 seed 各自曲线（不平均，画至各自早停）。

    Args:
        metric: "auc" 或 "loss"。
        loss_runs: 预解析的 loss run 表（metric="auc" 时忽略）。
        out_dir: 输出目录（自动创建）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = (("overall_auc", "验证集 Overall AUC"), ("worst_auc", "验证集 worst-case AUC")) if metric == "auc" \
        else (("train_loss", "训练集 loss (train BCE)"), ("val_loss", "验证集 loss (val BCE)"))
    count = 0
    for ds in DATASETS:
        for method in METHODS:
            tag = SELECTED_CONFIGS[ds.key][method]
            present: list[tuple[int, Curve]] = []
            for seed in SEEDS:
                c = load_auc_curve(ds, method, tag, seed) if metric == "auc" \
                    else loss_runs.get((ds.log_name, method, tag, seed))
                if c:
                    present.append((seed, c))
            if not present:
                print(f"  SKIP {ds.key} {method} (no runs)")
                continue
            fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
            for ax, (fname, ylab) in zip(axes, cols):
                _style_axis(ax)
                for seed, curve in present:
                    xs, ys = _series_for(curve, fname)
                    if xs:
                        ax.plot(xs, ys, color=SEED_COLOR[seed], lw=1.6, marker="o", ms=3.5,
                                mec="white", mew=0.4, zorder=3)
                ax.set_title(ylab, fontsize=10.5, color=MUTED, loc="left", pad=4)
                ax.set_xlabel("Epoch", fontsize=9.5)
            handles = [Line2D([0], [0], color=SEED_COLOR[s], lw=2, marker="o", ms=5, mec="white", mew=0.5,
                              label=(f"seed {s} (fold {s - 42})" if ds.is_cv else f"seed {s}")) for s, _ in present]
            fig.legend(handles=handles, ncol=len(present), loc="upper center", frameon=False,
                       fontsize=9.5, bbox_to_anchor=(0.5, 0.995))
            fold = " · seed=fold(0–4)" if ds.is_cv else ""
            what = "AUC" if metric == "auc" else "loss"
            fig.suptitle(f"{ds.display.split('  ')[0]} · {METHOD_LABEL_LONG[method]} · config={tag}{fold}"
                         f"  —— 5 seed 各自 {what}（不平均，画至各自早停）",
                         fontsize=11.5, fontweight="bold", y=0.9)
            fig.tight_layout(rect=(0, 0, 1, 0.86))
            out_path = out_dir / f"{ds.val_log_dir}__{method}.png"
            fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            count += 1
    print(f"共 {count} 张 → {out_dir}")


# ============================================================
# CLI
# ============================================================
def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser(description="绘制选定配置的 5-seed 训练曲线（AUC / loss，均值 / per-seed）")
    p.add_argument("--metric", choices=["auc", "loss"], default="auc", help="指标：auc（读 val_logs）或 loss（解析 SLURM 日志）")
    p.add_argument("--mode", choices=["mean", "per-seed"], default="mean", help="口径：mean 5×2 网格 或 per-seed 每图")
    p.add_argument("--all", action="store_true", help="一次生成 4 种组合（覆盖 --metric/--mode）")
    p.add_argument("--out-dir", type=Path, default=OUT_ROOT, help="输出根目录（默认 outputs/）")
    return p.parse_args()


def _run_one(metric: str, mode: str, loss_runs: dict, out_root: Path) -> None:
    """按 (metric, mode) 生成一份产物。"""
    if mode == "mean":
        name = "training_curves_selected_configs.png" if metric == "auc" else "loss_curves_selected_configs.png"
        plot_mean_grid(metric, loss_runs, out_root / name)
    else:
        sub = "auc_curves_per_seed" if metric == "auc" else "loss_curves_per_seed"
        plot_per_seed(metric, loss_runs, out_root / sub)


def main() -> None:
    """入口：装配字体、按需解析 loss 日志、生成所选产物。"""
    args = _parse_args()
    setup_font()
    combos = [("auc", "mean"), ("auc", "per-seed"), ("loss", "mean"), ("loss", "per-seed")] \
        if args.all else [(args.metric, args.mode)]
    # 仅当需要 loss 时才解析 SLURM 日志（较慢）
    loss_runs = load_all_loss_curves() if any(m == "loss" for m, _ in combos) else {}
    for metric, mode in combos:
        _run_one(metric, mode, loss_runs, args.out_dir)


if __name__ == "__main__":
    main()
