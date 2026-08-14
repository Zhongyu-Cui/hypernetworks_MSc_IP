"""
E1 主分析（HAM ID）：pooled-OOF + H1/H2 确认性检验
==================================================
docs/conditioning_ablation_plan.md §2/§3.3 的落地。消费 `train_condnet.py` 在
`outputs/conditioning_ablation/ham10000/cv5/fold{k}/` 落盘的 4 cell × 5 折 × 3 seed 预测。

**估计量 = averaging（`--estimator averaging`，主口径；对 plan §3「pooled-OOF」的有依据偏离，见下）**：
  · **逐折算指标 → 折间平均**，**永不跨折池化分数**；
  · **3 seed 等权平均**（plan §3「统一 3 seed」）；折与 seed 合起来 = 15 个模型等权平均；
  · **marginal worst-group**：**先平均后 min**（见下「次序」），候选组 = E-audit.1 冻结的 **age 4 组**
    （HAM-ID regime，比既有 cv_oof_report 的 marginal 含 sex 更窄——必须从 manifest 读，勿复用旧口径）；
  · **gap 分解**：Δgap = Δbest − Δworst（防「压低强组」被误判为公平改善）。

**为什么弃用 pooled-OOF（文献依据）**：直接池化各折 raw logit 算一个 AUC，会构造出**跨折正负对**
（折 i 的阳性 vs 折 j 的阴性），而这两个分数来自**不同模型**、尺子不同。这是学界已知缺陷：
  · Parker, Günter & Bedo (2007), *BMC Bioinformatics* 8:326 —— 低信号数据上 pooled AUC 有严重**负偏**
    （无信号数据 AUC 掉到 <0.3 而非 0.5）；
  · Airola et al. (2010), *JMLR W&CP* 8:3–13 ——「pooling 假设各折分类器来自同一总体……计算 AUC 时该假设
    更可疑，因为部分正负对由**不同折**的样本构成」，并推荐 **averaging** 或 LPOCV；
  · Forman & Scholz (2010), *SIGKDD Explorations* 12(1):49–57 —— 各家聚合口径不兼容致文献不可比。
本项目实测印证：早停轮数异质 ⇒ 逐折 logit 尺度漂移达 19 个单位 ⇒ C-Deep 逐折 AUC 均值 0.831 却池化成
0.639；且惩罚因 cell 而异（ERM −0.01~−0.05 vs C-Deep −0.19）⇒ **会把校准漂移测成范围效应**（旧口径下
H2 假阳性「确认」Holm=0.006，真值 +0.0075/p=0.70）。

**averaging 不损失评估 n**：抬高 n 的功劳来自 **CV 本身**（每个样本都轮到一次 test ⇒ 全 9,707 张都有
干净预测），**不是** pooling；两种聚合都用满这 9,707 张。AUC 方差由**类别样本数**支配
（`Var≈c/n_pos+c'/n_neg`）而非「对数」，故 5 折各自估计再平均（÷5）与池化精度相当。**实测反而更紧**：
worst-group 的 bootstrap SE = averaging/池化 = 0.72~0.92×（C-Deep 收益最大 —— 正是漂移最重的 cell，
印证跨折对只注入噪声、不带信息）。LPOCV 需按对重训，本项目算力不可行，故取 averaging。

**⚠️ min 与 average 的次序（不可交换，已冻结选择）**：
  · (A) `mean_f[min_g AUC(g,f)]` 先 min 后平均；(B) `min_g[mean_f AUC(g,f)]` **先平均后 min**（**本脚本采用**）。
  · Jensen 恒有 (A) ≤ (B)。选 (B) 的两个理由：① **(A) 放大 min 的向下偏**——E-audit.3 已实测 min 算子系统性
    挑负向噪声、组内 n 越小偏越狠，而 (A) 让每折（age:20-40 每折仅 ~19 阳性）各自取极值，偏得最厉害；
    (B) 先跨折平均压噪再取 min。② **(B) 才符合语义**——「模型在哪个年龄组最弱」应是模型属性，
    不该每折换一个答案（(A) 允许折 1 挑 20-40、折 2 挑 80+，混的是不同组）。
  · **动态 argmin 仍保留**：每 replicate 内重算 `mean_f AUC(g,f)` 再取 min，argmin 可跨 replicate 移动。

**推断**：**病灶级配对 cluster bootstrap**（plan §3.3）——只重采样 lesion（cluster）、不重采样 seed；
所有 cell 共用**同一批重采样 cluster 索引**（配对）。HAM 45% 样本处在多图病灶中（7,414 lesion /
9,948 图），故必须按病灶聚类，否则 CI 偏窄。
> lesion_id 未随预测落盘，按**行序**从 fold test.csv 恢复（HAM10000Dataset 不改行序、loader
> shuffle=False、age 过滤为布尔掩码亦保序），并以 **sex/age/label 三重逐元素校验**证实对齐。

**确认性判决（plan §3.3，两层分开报）**：
  · **H1 = C-Deep − C-Head**、**H2 = C-Full − C-Deep**，family 内 **Holm** 校正（m=2）；
  · `p_joint = max(p_fair, p_noninf)`：
      - `p_fair`（worst-group 优效，单侧右尾）：H0: Δ≤0，零假设中心化 `T*=Δ*−Δ̄_obs`，
        `p=(1+#{T*≥Δ̄_obs})/(B+1)`；
      - `p_noninf`（Overall 非劣，单侧）：δ_NI=−0.005，H0: ΔOverall ≤ −0.005，**同一批 bootstrap 索引**；
      - **非劣参照 = 被比较的范围模型本身**（H1: C-Deep 不劣于 C-Head；H2: C-Full 不劣于 C-Deep）。
  · **效应分级**（描述性）：|Δ|≥0.005 小但可重复；≥0.010 较强。**不是存在性判据**。

**最小观测有效配置**（纯描述性，判据 = 点估计，plan §2）：按嵌套序 Head→Deep→Full，第一个满足
「worst-group 相对 ERM `Δ̄>0` **且** Overall 非劣 `Δ̄≥−0.005`」的档；CI 一并报告以示透明，
**但不要求 CI 下界>0**（要求 CI 即等于升格为显著性主张，与描述性定位冲突）。

⚠️ **E-audit.3 校准提示**：HAM 规模（9,707、最小组 694）**正是动态-min 偏差/欠覆盖的敏感区间**
（不同于 n 大 15× 的 M2C）。故本脚本必报 **argmin 稳定性**：若 argmin 频繁翻转（近 `tied` 情景），
则「未达显著」可能是估计量保守伪影而非真无效应，须据此收严表述。

运行（轻量 CPU）：
    python scripts/e1_ham_analysis.py --n-boot 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from scripts.eaudit_m2c_full_target import sorted_view, weighted_auc
from src.training.harness.predictions import load_predictions

REPO_ROOT = Path(__file__).resolve().parents[1]
ABLATION_ROOT = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/conditioning_ablation")
DS = "ham10000"
CV_DIR = ABLATION_ROOT / DS / "cv5"
SPLIT_DIR = REPO_ROOT / "data" / "splits" / "ham10000" / "cv5"
MANIFEST = ABLATION_ROOT / "eaudit_candidate_groups.json"

N_FOLDS = 5
SEEDS = (42, 43, 44)                       # plan §3「统一 3 seed」
CELLS = ("condnet_erm", "condnet_head", "condnet_deep", "condnet_full")
NICE = {"condnet_erm": "ERM", "condnet_head": "C-Head", "condnet_deep": "C-Deep", "condnet_full": "C-Full"}
# 嵌套序（描述性「最小观测有效配置」按此序扫）
NESTED = ("condnet_head", "condnet_deep", "condnet_full")
DELTA_NI = -0.005
HAM_AGE_NAMES = {0: "20-40", 1: "40-60", 2: "60-80", 3: "80+"}


def load_candidate_groups() -> list[str]:
    """读 E-audit.1 冻结的 HAM-ID 候选组（age 4 组）。"""
    m = json.loads(MANIFEST.read_text())["regimes"]["HAM-ID"]
    return [g for g, c in m["groups"].items() if c["passes"]]


def load_folds(selected: dict[str, str], ckpt: str = "overall") -> tuple[list[dict], np.ndarray, np.ndarray]:
    """
    载入**保持逐折结构**的预测（averaging 主口径所需；不跨折拼接分数）。

    Args:
        selected: {cell: config_tag}。
        ckpt    : 评估用 checkpoint 选择，"overall"（主结果，val Overall 最优 epoch）
                  或 "worstcase"（val worst-case AUC 最优 epoch，供训练策略敏感性实验）。

    Returns:
        (folds, lesion_idx, offsets)：
          folds[k] = {"y","age","n","scores"{(cell,seed)->[n_k]}}；
          lesion_idx = 全局病灶整数编码（按折序拼接，供 cluster bootstrap 全局重采样）；
          offsets    = 各折在全局权重向量中的起止（offsets[k]:offsets[k+1]）。
    """
    folds: list[dict] = []
    lesion_parts: list[np.ndarray] = []
    for k in range(N_FOLDS):
        df = pd.read_csv(SPLIT_DIR / f"fold{k}" / "test.csv")
        df = df[df["age_group"] >= 0].reset_index(drop=True)   # 与训练侧 _filter_age_valid 同口径
        rec: dict = {"scores": {}}
        for cell in CELLS:
            tag = selected[cell]
            for seed in SEEDS:
                p = CV_DIR / f"fold{k}" / "predictions" / f"{cell}_{tag}_seed{seed}_{ckpt}.npz"
                yy, ss, attrs = load_predictions(p)
                # 行序对齐硬校验（lesion_id 据此按位置恢复）
                assert len(yy) == len(df), f"{p} 长度与 CSV 不符"
                assert np.array_equal(np.asarray(yy).astype(int), df["label"].values.astype(int))
                assert np.array_equal(attrs["age"].astype(int), df["age_group"].values.astype(int))
                assert np.array_equal(attrs["sex"].astype(int), df["sex"].values.astype(int))
                rec["scores"][(cell, seed)] = np.asarray(ss, dtype=np.float64)
                rec["y"] = np.asarray(yy).astype(int)
                rec["age"] = attrs["age"].astype(int)
        rec["n"] = len(df)
        folds.append(rec)
        lesion_parts.append(df["lesion_id"].values)
    lesions = np.concatenate(lesion_parts)
    _, lesion_idx = np.unique(lesions, return_inverse=True)
    offsets = np.cumsum([0] + [f["n"] for f in folds])
    return folds, lesion_idx, offsets


class AvgViews:
    """
    averaging 主口径的预排序视图：每 (fold, cell, seed) × {overall, 各 age 组} 存折内排序索引与标签。
    **永不跨折拼接分数** —— 这正是它免疫跨折校准漂移的原因（无跨折正负对）。
    """

    def __init__(self, folds: list[dict], groups: list[str], offsets: np.ndarray) -> None:
        self.groups = groups
        self.offsets = offsets
        self.n_folds = len(folds)
        name2code = {f"age:{v}": k for k, v in HAM_AGE_NAMES.items()}
        self.v: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        for k, f in enumerate(folds):
            full = np.ones(f["n"], dtype=bool)
            for key, s in f["scores"].items():
                self.v[(k, key, "__all__")] = sorted_view(f["y"], s, full)
                for g in groups:
                    self.v[(k, key, g)] = sorted_view(f["y"], s, f["age"] == name2code[g])

    def cell_metrics(self, cell: str, w: np.ndarray) -> tuple[float, float, float, list[str]]:
        """
        averaging 估计量：逐 (fold, seed) 算各组 AUC → **对 fold×seed 等权平均** → **再** min/max 取组。

        次序 = 「先平均后 min」（见模块 docstring：压 min 的向下偏 + 语义正确）。
        动态 argmin 保留：本函数在给定权重下重算组均值再取 min，故 argmin 随 replicate 移动。

        Args:
            cell: cell 名。
            w   : [N] 全局样本权重（cluster bootstrap 抽中次数；按 offsets 切给各折）。

        Returns:
            (Overall, worst, best, [argmin 组名])；argmin 为单元素列表（估计量已跨 fold/seed 平均）。
        """
        per_group: dict[str, list[float]] = {g: [] for g in self.groups}
        overalls: list[float] = []
        for k in range(self.n_folds):
            wk = w[self.offsets[k]:self.offsets[k + 1]]
            for seed in SEEDS:
                i, ys = self.v[(k, (cell, seed), "__all__")]
                a = weighted_auc(ys, wk[i])
                if not np.isnan(a):
                    overalls.append(a)
                for g in self.groups:
                    ig, yg = self.v[(k, (cell, seed), g)]
                    ag = weighted_auc(yg, wk[ig])
                    if not np.isnan(ag):
                        per_group[g].append(ag)
        means = {g: float(np.mean(v)) for g, v in per_group.items() if v}
        g_min = min(means, key=means.get)
        return (float(np.mean(overalls)), means[g_min], float(max(means.values())), [g_min])


def load_pooled(
    selected: dict[str, str], score_scale: str = "fold_rank", ckpt: str = "overall",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    池化 5 折 OOF。y/attrs/lesion 对所有 cell、seed 相同（同折结构、同行序），只有分数不同。

    ⚠️ **`score_scale="fold_rank"`（默认）修复跨折校准漂移伪影** —— 本项目 pooled-OOF 的一个
    实测缺陷：各折是**不同模型**，其 logit 尺度因早停轮数不同而漂移（实测 C-Deep seed44 逐折
    logit 中位数 [-7.6,-4.0,**-23.0**,-6.0,-11.2]，跨折差 19 个单位）。直接池化 **raw logit** 会把
    「折内排序」与「跨折尺度对齐」混在一起：该 run 逐折 test AUC 均值 0.831（健康）却池化成
    **0.639**（−0.192）。且惩罚随 cell 系统性不同（ERM 仅 −0.01~−0.05，C-Deep 达 −0.19）
    ⇒ **H1/H2 会测成校准漂移而非范围效应**。实测 `corr(跨折 logit 离散度, 池化惩罚) = −0.607`。

    修复 = **折内秩归一**：把每折**全部** test 样本的分数映射到折内百分位 (0,1)。这是折内**严格单调**
    变换 ⇒ **折内 AUC 完全不变**（AUC 只依赖排序），但各折被放到同一可比尺度上。实测把最差池化
    惩罚由 −0.192 压到 −0.051、相关性由 −0.607 降到 −0.136。折间可交换性成立（GroupKFold 对同一
    总体的随机划分），故该变换不引入偏倚。

    Args:
        selected   : {cell: config_tag}。
        score_scale: "fold_rank"（默认，修复后主口径）或 "raw_logit"（旧口径，仅作敏感性对照）。

    Returns:
        (y, lesion_idx, age, scores)：scores[(cell, seed)] = [N] 分数；lesion_idx 为病灶整数编码。
    """
    if score_scale not in ("fold_rank", "raw_logit"):
        raise ValueError(f"未知 score_scale={score_scale!r}")
    y_parts, age_parts, lesion_parts = [], [], []
    score_parts: dict[tuple[str, int], list[np.ndarray]] = {(c, s): [] for c in CELLS for s in SEEDS}

    for k in range(N_FOLDS):
        df = pd.read_csv(SPLIT_DIR / f"fold{k}" / "test.csv")
        df = df[df["age_group"] >= 0].reset_index(drop=True)   # 与训练侧 _filter_age_valid 同口径
        ref_y = ref_age = None
        for cell in CELLS:
            tag = selected[cell]
            for seed in SEEDS:
                p = CV_DIR / f"fold{k}" / "predictions" / f"{cell}_{tag}_seed{seed}_{ckpt}.npz"
                yy, ss, attrs = load_predictions(p)
                # 行序对齐硬校验：CSV 的 label/sex/age 必须与预测逐元素相等（lesion_id 据此按位置恢复）
                assert len(yy) == len(df), f"{p} 长度与 CSV 不符"
                assert np.array_equal(np.asarray(yy).astype(int), df["label"].values.astype(int))
                assert np.array_equal(attrs["age"].astype(int), df["age_group"].values.astype(int))
                assert np.array_equal(attrs["sex"].astype(int), df["sex"].values.astype(int))
                s_arr = np.asarray(ss, dtype=np.float64)
                if score_scale == "fold_rank":
                    # 折内百分位：在该折**全部** test 样本上算秩（不是逐子群），保折内排序、去尺度
                    s_arr = rankdata(s_arr) / (len(s_arr) + 1.0)
                score_parts[(cell, seed)].append(s_arr)
                if ref_y is None:
                    ref_y, ref_age = np.asarray(yy).astype(int), attrs["age"].astype(int)
        y_parts.append(ref_y)
        age_parts.append(ref_age)
        lesion_parts.append(df["lesion_id"].values)

    y = np.concatenate(y_parts)
    age = np.concatenate(age_parts)
    lesions = np.concatenate(lesion_parts)
    _, lesion_idx = np.unique(lesions, return_inverse=True)
    scores = {k: np.concatenate(v) for k, v in score_parts.items()}
    return y, lesion_idx, age, scores


class Views:
    """预排序视图：每个 (cell, seed) × {overall, 各 age 候选组} 存排序索引与标签，供加权 AUC 复用。"""

    def __init__(self, y: np.ndarray, age: np.ndarray, scores: dict, groups: list[str]) -> None:
        self.groups = groups
        name2code = {f"age:{v}": k for k, v in HAM_AGE_NAMES.items()}
        self.masks = {g: age == name2code[g] for g in groups}
        full = np.ones(len(y), dtype=bool)
        self.v: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        for (cell, seed), s in scores.items():
            self.v[(cell, seed, "__all__")] = sorted_view(y, s, full)
            for g in groups:
                self.v[(cell, seed, g)] = sorted_view(y, s, self.masks[g])

    def one(self, cell: str, seed: int, w: np.ndarray) -> tuple[float, float, float, str]:
        """单 (cell,seed) 的 (Overall, worst, best, argmin 组名)，动态 argmin。"""
        idx, ys = self.v[(cell, seed, "__all__")]
        overall = weighted_auc(ys, w[idx])
        aucs, names = [], []
        for g in self.groups:
            i, yy = self.v[(cell, seed, g)]
            a = weighted_auc(yy, w[i])
            if not np.isnan(a):
                aucs.append(a); names.append(g)
        arr = np.asarray(aucs)
        j = int(arr.argmin())
        return overall, float(arr[j]), float(arr.max()), names[j]

    def cell_metrics(self, cell: str, w: np.ndarray) -> tuple[float, float, float, list[str]]:
        """对 3 seed **等权平均**，得 (Overall, worst, best) 与各 seed 的 argmin 组名。"""
        rows = [self.one(cell, s, w) for s in SEEDS]
        m = np.mean([[r[0], r[1], r[2]] for r in rows], axis=0)
        return float(m[0]), float(m[1]), float(m[2]), [r[3] for r in rows]


def _p_one_sided(boot_delta: np.ndarray, obs: float, null_value: float, n_boot: int) -> float:
    """
    单侧右尾 p（零假设中心化，plan §3.3）：H0: Δ ≤ null_value。
    `T* = Δ* − Δ̄_obs`；`p = (1 + #{T* ≥ Δ̄_obs − null_value}) / (B+1)`。
    """
    t = boot_delta - obs
    return float((1 + np.sum(t >= obs - null_value)) / (n_boot + 1))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm 校正（family 内），返回校正后 p。"""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        out[k] = adj
        prev = adj
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="E1 HAM ID 主分析：pooled-OOF + H1/H2")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--estimator", choices=("averaging", "pooled_rank", "pooled_raw"),
                    default="averaging",
                    help="averaging=逐折算指标再平均（**主口径**，文献推荐 Parker2007/Airola2010，"
                         "无跨折正负对）；pooled_rank=折内秩归一后池化（敏感性）；"
                         "pooled_raw=旧口径直接池化 raw logit（敏感性，已证被漂移污染）。")
    ap.add_argument("--checkpoint", choices=("overall", "worstcase"), default="overall",
                    help="评估用 checkpoint：overall=val Overall 最优 epoch（**主结果**）；"
                         "worstcase=val worst-case AUC 最优 epoch（训练策略敏感性实验）。"
                         "⚠️ config 选择不变（仍沿用 §3.2 best_overall 口径选定的 config），"
                         "仅切换最终评估的 checkpoint，以隔离「保哪一档权重」这一单一变量。")
    args = ap.parse_args()

    selected = json.loads((CV_DIR / "selected_configs.json").read_text())["config"]
    groups = load_candidate_groups()
    _DESC = {"averaging": "averaging（逐折算指标→折间平均；**无跨折正负对**，免疫校准漂移）",
             "pooled_rank": "pooled + 折内秩归一（敏感性：保折内排序、去尺度）",
             "pooled_raw": "⚠️ pooled + raw logit（旧口径，已证被跨折漂移污染，仅作对照）"}
    print("E1 主实验 A（HAM ID）：H1/H2 确认性检验")
    print(f"  选定 config: {json.dumps(selected, ensure_ascii=False)}")
    print(f"  候选组（E-audit.1 冻结 HAM-ID）= {groups}")
    print(f"  估计量: {_DESC[args.estimator]}")
    print(f"  评估 checkpoint: {args.checkpoint}"
          + ("（主结果）" if args.checkpoint == "overall"
             else "（val worst-case 最优 epoch；config 选择不变，仅换评估权重）"))

    if args.estimator == "averaging":
        folds, lesion_idx, offsets = load_folds(selected, ckpt=args.checkpoint)
        views = AvgViews(folds, groups, offsets)
        n_tot = int(offsets[-1])
        y_all = np.concatenate([f["y"] for f in folds])
    else:
        y_all, lesion_idx, age, scores = load_pooled(
            selected, score_scale="fold_rank" if args.estimator == "pooled_rank" else "raw_logit",
            ckpt=args.checkpoint)
        views = Views(y_all, age, scores, groups)
        n_tot = len(y_all)
    n_les = int(lesion_idx.max() + 1)
    print(f"  评估集: n={n_tot:,}  病灶={n_les:,}  恶性率={y_all.mean():.4f}  seeds={SEEDS}")
    print("  [OK] 行序对齐经 label/sex/age 三重逐元素校验（lesion_id 按位置恢复）")

    y = y_all
    ones = np.ones(n_tot)

    # ---------- 点估计 ----------
    obs: dict[str, tuple[float, float, float]] = {}
    obs_argmin: dict[str, list[str]] = {}
    print(f"\n{'=' * 78}\nE1 点估计（{args.estimator}，fold×seed 等权平均）\n{'=' * 78}")
    print(f"| {'cell':<7s} | {'Overall':>8s} | {'marg worst':>10s} | {'best':>7s} | {'gap':>7s} | argmin 组(各seed)")
    for c in CELLS:
        o, w_, b, am = views.cell_metrics(c, ones)
        obs[c] = (o, w_, b); obs_argmin[c] = am
        print(f"| {NICE[c]:<7s} | {o:>8.4f} | {w_:>10.4f} | {b:>7.4f} | {b - w_:>7.4f} | "
              f"{'/'.join(sorted(set(x.split(':')[-1] for x in am)))}")

    # ---------- 病灶级配对 cluster bootstrap ----------
    print(f"\n病灶级配对 cluster bootstrap（B={args.n_boot}，只重采样 lesion、不重采样 seed，"
          f"全 cell 共用同一批索引）...")
    rng = np.random.default_rng(args.seed)
    boot: dict[str, np.ndarray] = {c: np.empty((args.n_boot, 3)) for c in CELLS}   # (Overall, worst, best)
    argmin_cnt: dict[str, dict[str, int]] = {c: {g: 0 for g in groups} for c in CELLS}
    for b in range(args.n_boot):
        draw = rng.integers(0, n_les, size=n_les)
        w = np.bincount(draw, minlength=n_les).astype(np.float64)[lesion_idx]
        for c in CELLS:
            o, w_, bst, am = views.cell_metrics(c, w)
            boot[c][b] = (o, w_, bst)
            for g in am:
                argmin_cnt[c][g] += 1
        if (b + 1) % 250 == 0:
            print(f"  ... {b + 1}/{args.n_boot}")

    # ---------- H1 / H2 ----------
    fam = {"H1: C-Deep − C-Head": ("condnet_deep", "condnet_head"),
           "H2: C-Full − C-Deep": ("condnet_full", "condnet_deep")}
    print(f"\n{'=' * 78}\nE1 确认性检验 H1/H2（family 内 Holm，m=2；单侧「前者更优/不劣」）\n{'=' * 78}")
    print(f"| {'假设':<20s} | {'Δworst':>8s} | {'95% CI':>18s} | {'p_fair':>7s} | "
          f"{'ΔOverall':>9s} | {'p_noninf':>8s} | {'p_joint':>7s} |")
    res: dict[str, dict] = {}
    for name, (a, bl) in fam.items():
        d_w = obs[a][1] - obs[bl][1]
        d_o = obs[a][0] - obs[bl][0]
        bw = boot[a][:, 1] - boot[bl][:, 1]
        bo = boot[a][:, 0] - boot[bl][:, 0]
        lo, hi = np.percentile(bw, [2.5, 97.5])
        p_fair = _p_one_sided(bw, d_w, 0.0, args.n_boot)
        p_noninf = _p_one_sided(bo, d_o, DELTA_NI, args.n_boot)
        p_joint = max(p_fair, p_noninf)
        d_gap = (obs[a][2] - obs[a][1]) - (obs[bl][2] - obs[bl][1])
        res[name] = {"d_worst": d_w, "ci": [lo, hi], "p_fair": p_fair, "d_overall": d_o,
                     "p_noninf": p_noninf, "p_joint": p_joint, "d_gap": d_gap}
        print(f"| {name:<20s} | {d_w:>+8.4f} | [{lo:>+7.4f},{hi:>+7.4f}] | {p_fair:>7.4f} | "
              f"{d_o:>+9.4f} | {p_noninf:>8.4f} | {p_joint:>7.4f} |")
    holm_p = holm({k: v["p_joint"] for k, v in res.items()})
    print(f"\nHolm 校正后（family = HAM-ID 范围确认，m=2）：")
    for k, v in res.items():
        v["p_joint_holm"] = holm_p[k]
        verdict = "**确认**" if holm_p[k] < 0.05 else "未达显著"
        grade = ("较强(|Δ|≥0.010)" if abs(v["d_worst"]) >= 0.010 else
                 "小但可重复(|Δ|≥0.005)" if abs(v["d_worst"]) >= 0.005 else "微弱(<0.005)")
        print(f"  {k:<20s}: p_joint={v['p_joint']:.4f} → Holm={holm_p[k]:.4f}  {verdict}   "
              f"[效应分级: {grade}；Δgap={v['d_gap']:+.4f}]")

    # ---------- 最小观测有效配置（描述性）----------
    print(f"\n{'=' * 78}\n最小观测有效配置（**纯描述性，判据=点估计**；CI 仅透明展示，不作要求）\n{'=' * 78}")
    minimal = None
    for c in NESTED:
        dw = obs[c][1] - obs["condnet_erm"][1]
        do = obs[c][0] - obs["condnet_erm"][0]
        bw = boot[c][:, 1] - boot["condnet_erm"][:, 1]
        lo, hi = np.percentile(bw, [2.5, 97.5])
        ok = (dw > 0) and (do >= DELTA_NI)
        if ok and minimal is None:
            minimal = c
        print(f"  {NICE[c]:<7s} vs ERM: Δworst={dw:>+7.4f} [{lo:>+7.4f},{hi:>+7.4f}]  "
              f"ΔOverall={do:>+7.4f}  判据(Δworst>0 且 ΔOverall≥−0.005)={'满足' if ok else '不满足'}")
    print(f"\n  ⇒ 最小观测有效配置 = {NICE[minimal] if minimal else '**所测范围内未观测到有效配置**'}")

    # ---------- 动态-min 稳定性（E-audit.3 联动）----------
    # ⚠️ 分母必须按**该 cell 的实际计数总和**求，不能写死 n_boot×len(SEEDS)：averaging 口径下
    # cell_metrics 已把 fold×seed 平均掉、每 replicate 只产出 **1 个** argmin，而 pooled 口径每
    # replicate 每 seed 各产出 1 个。写死会让占比差 3 倍（曾致占比总和只有 ~33%）。
    print(f"\n{'=' * 78}\n动态-min argmin 稳定性（{args.n_boot} replicate，{args.estimator} 口径）\n{'=' * 78}")
    for c in CELLS:
        top = sorted(argmin_cnt[c].items(), key=lambda kv: -kv[1])
        n_am = max(1, sum(argmin_cnt[c].values()))
        dom = top[0][1] / n_am
        share = ", ".join(f"{g}={v / n_am:.1%}" for g, v in top if v)
        regime = ("separated（Δ 判决可信）" if dom >= 0.9 else
                  "介于两者之间（谨慎）" if dom >= 0.6 else "tied（Δ 保守/欠覆盖，慎判 null）")
        print(f"  {NICE[c]:<7s}: {share}  → 主导 {dom:.1%}，近似 {regime}")
    print("\n  ⚠️ HAM 规模（n=9,707、最小组 694）落在 E-audit.3 校准的敏感区间：若上表近 `tied`，")
    print("     则「未达显著」可能是动态-min 的保守伪影（校准实测 Δ 偏差 −0.006、覆盖率 80%），非真无效应。")

    out = {"selected_config": selected, "candidate_groups": groups,
           "estimator": args.estimator, "checkpoint": args.checkpoint,
           "n": int(n_tot), "n_lesions": n_les, "seeds": list(SEEDS), "n_boot": args.n_boot,
           "point_estimates": {NICE[c]: {"overall": obs[c][0], "marginal_worst": obs[c][1],
                                         "best": obs[c][2], "gap": obs[c][2] - obs[c][1]} for c in CELLS},
           "hypotheses": res,
           "minimal_effective": NICE[minimal] if minimal else None,
           "argmin_share": {NICE[c]: {g: v / max(1, sum(argmin_cnt[c].values()))
                             for g, v in argmin_cnt[c].items() if v} for c in CELLS}}
    ckpt_suffix = "" if args.checkpoint == "overall" else f"_{args.checkpoint}"
    p = CV_DIR / f"e1_ham_results_{args.estimator}{ckpt_suffix}.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已落盘 → {p}")


if __name__ == "__main__":
    main()
