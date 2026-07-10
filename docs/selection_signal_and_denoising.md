# 模型选择策略与选择信号 —— 单-split → CV 双端降噪实证

> 建立日期：2026-07-10。回答「在 worst-group 剧烈震荡的背景下，模型选择策略与选择信号该如何配置」。
> 结论由 `scripts/selection_strategy_probe.py` 在**已落盘预测**上重算（零训练）；CV 数据来自
> HAM 全-CV 6 config × 5 折搜索（作业 70771–70774）。与 `docs/routeA_ham_cv_oof_fairness.md`（评估侧
> worst-group 判决）互补：本文件聚焦**选择侧**。

---

## 0. TL;DR

- **问题**：worst-group AUC 逐 epoch 剧烈震荡（`docs/training_dynamics_diagnostics` 侧 wcVol 0.045–0.23），
  怀疑不是良好的模型选择信号。
- **单-split 实证**：在 HAM 6 config 上，**所有 val 选择信号（overall / worst-case / DTO）与 test
  worst-group 的秩相关全为负**（ρ −0.13 ~ −0.33）——选择环节在**选噪声、甚至反向**；DTO 一度给
  HyperHead 选出 test worst=**0.586** 的灾难 config。
- **CV 双端降噪**（折均 val 选择信号 √5 降噪 + 5 折 OOF 池化 test 评估-n 地板已抬）：**系统性负相关消失**
  （5/5 全负 → 3/5 转正、负的都趋零，ρ −0.12 ~ +0.21），**灾难性选择消失**（DTO 0.586→0.714），
  **三策略大面积一致**。
- **判决**：单-split 的"选择反向有害"主要是**小样本伪影**；**双端降噪把选择从"有害"变为"低风险"**。
  剩余 ρ≈0 是**良性**的——合理 config 的真实 worst-group 近乎并列（0.70–0.79），选择本身低风险。
  **真正的杠杆在信号层降噪（fold-mean + OOF），而非策略层的精巧选择。**

---

## 1. 动机

MEDFAIR（Zong et al., ICLR 2023）证明**模型选择策略比去偏算法更影响 worst-case 公平**：Minimax-Pareto
显著优于 Overall，且几乎不损整体 AUC。但 MEDFAIR 在**单-split + 3 seed** 上做选择，**没有排除小 val 的
选择噪声**——它甚至用 val worst-case AUC 早停（本项目已改用 overall 早停，更稳）。

本项目的训练动力学诊断显示 worst-case 逐 epoch |Δ| 达 0.045–0.23（远大于 Overall ~0.01），且 R4.2 证
worst-group 被 joint 小格（n_pos 3/8/4）噪声地板主导。故提出核心疑问：**val 上的 worst-group 选择信号，
到底能不能预测 test 上的 worst-group？** 若不能，则任何选择策略（Overall/Pareto/DTO）都建在流沙上。

---

## 2. 方法

**三选择器**（均在 val 子群 AUC 向量上，复刻 MEDFAIR 三策略）：
- **Overall**：选 val 整体 AUC 最高的 config。
- **Minimax-Pareto**：在子群 AUC 向量的 Pareto 前沿里选 val worst-case 最高者。
- **DTO**：选到 val 乌托邦点（各子群跨 config 的最大 AUC）归一化欧氏距离最小者。

**选择信号有效性 = 代理秩相关**：跨候选 config，比较 val 选择信号与 test worst-group 的 Spearman ρ。
ρ 越高 = 该 val 信号越能预测 test 公平、越适合当选择信号；≈0 或负 = 在选噪声。

**两种口径**：
| 口径 | 选择信号（val） | 评估目标（test） | 数据 |
|------|----------------|-----------------|------|
| 单-split | seed42 单折 val | 单 split test（995→969 age 过滤，joint 小格 ~26） | 搜索 69577–69580 |
| **CV 双端降噪** | **5 折折均 val（√5 降噪）** | **5 折 OOF 池化（n≈9707，地板已抬）** | 搜索 **70771–70774** |

canonical 子群 = 含 sex×age joint（14 格）；marginal = sex/age 6 组（隔离 joint 小格噪声）。

---

## 3. 结果

### 3.1 选择信号 → test worst-group 秩相关（24 点 = 4 方法 × 6 config）

| val 信号 → test 目标 | 单-split ρ (p) | **CV 折均+OOF ρ (p)** |
|---|---|---|
| val overall → test canonical-worst | −0.191 (0.370) | **−0.018 (0.933)** |
| val canonical-worst → test canonical-worst | −0.255 (0.229) | **+0.043 (0.843)** |
| **val marginal-worst → test marginal-worst** | −0.271 (0.201) | **+0.208 (0.330)** |
| val DTO(−dist) → test canonical-worst | −0.130 (0.545) | **−0.116 (0.590)** |
| val marginal-worst → test canonical-worst | −0.327 (0.119) | **+0.123 (0.568)** |

**读法**：单-split **5/5 全负**（选择反向有害）；CV 双端降噪后**散在 0 附近、3/5 转正、负的都趋零**。
系统性负相关消失。**全部不显著**（p 0.33–0.93，24 点功效低）——严格只能讲"点估计从系统性负变到≈0"，
但"全负 → 转正/趋零"的定性反转是稳的。

### 3.2 三策略选出的 config 与 test 表现（CV 口径）

| 方法 | 策略 | 选中 config | test overall | test c-worst | test m-worst |
|------|------|------------|--------------|--------------|--------------|
| ERM | Overall / DTO | lr1e-04_wd1e-04 | 0.8807 | 0.7727 | 0.7961 |
| ERM | Pareto | lr1e-04_wd1e-03 | 0.8800 | 0.7645 | 0.7981 |
| HyperHead | Overall | lr3e-04_wd1e-03 | 0.8884 | 0.7043 | 0.7314 |
| HyperHead | Pareto / DTO | lr1e-04_wd1e-03 | 0.8764 | 0.7140 | 0.7195 |
| HyperFusion | Overall / Pareto / DTO | lr1e-04_wd1e-04 | 0.8919 | 0.7822 | 0.8186 |
| HyperAdapt | Overall / Pareto | lr3e-04_wd1e-03 | 0.8926 | 0.7899 | 0.8054 |
| HyperAdapt | DTO | lr3e-05_wd1e-03 | 0.8816 | 0.7909 | 0.7974 |

- **三策略大面积一致**（HyperFusion 三者同选；ERM/HyperHead/HyperAdapt 的 Overall≈Pareto/DTO）。
- **灾难性选择消失**：单-split 下 DTO 给 HyperHead 选出 test c-worst=**0.586**（被一个 val 看着不错、
  test 有小格崩坏的 config 骗了）；CV 下 DTO 选的是 **0.714**——OOF 池化杀掉了小格伪影。
- **合理 config 近乎并列**：选出的 test c-worst 都落在 **0.70–0.79** 窄带——config 间真实 worst-group
  差异小。

### 3.3 为什么"修好负相关、却没变强正"——一个良性原因

合理 config 的真实 worst-group 近乎并列（0.70–0.79）。**当候选本身近乎打平时，没有任何选择信号能把
它们排得准（ρ≈0），但同时"选错"的代价也很小**。所以剩余的 ρ≈0 不是坏消息，而是**低风险**：
危险的是**负相关**（系统性选到更差的）——已消除；剩下的**近零相关**只说明选择这件事本身低风险。

---

## 4. 结论：选择策略与选择信号该如何配置

| 决策 | 配置 | 依据 |
|------|------|------|
| **早停 / epoch 选择** | **Overall AUC**（低方差），非 worst-case | worst-case 逐 epoch 抖动大；MEDFAIR 用 worst-case 早停是坑，本项目已改对 |
| **config 选择：降噪** | **CV-OOF + fold-mean（√5）双端降噪** | 把选择从"反向有害(−0.3)"修成"低风险(≈0)"，消除灾难性选择 |
| **config 选择：策略** | 三策略在 CV 下已收敛，**不必纠结**；若挑，**marginal-worst 折均**最不坏(+0.21)，DTO 无优势(−0.12) | 3.1 / 3.2 |
| **评估 / 报告** | **canonical worst-group on pooled-OOF**（真目标），选择用代理、评估用真目标 | 选择与评估分离；`docs/routeA_ham_cv_oof_fairness.md` |
| **优化力度** | **别过度优化 config 选择的公平性**——合理 config 近乎并列，选最稳信号后老实报得到的 worst-group | 3.3 |

**核心判决**：**真正的杠杆在信号层降噪（fold-mean + OOF），而非策略层的精巧选择。** MEDFAIR 的"换策略
改公平"在小数据上被选择噪声吞没；双端降噪解除噪声后，策略之争也随之消解，剩下的是"低风险的近平局选择"。

---

## 5. 诚实边界

- **秩相关全不显著**（24 点，p 0.33–0.93）。只能讲"点估计从系统性负变到≈0"，不能讲"确定转正"。
  但"单-split 5/5 全负 → CV 3/5 转正且负的趋零"的**定性反转**与整条推理链自洽（小样本制造假负相关，
  双端降噪解除之）。
- **仅 HAM 一个数据集**。MEDFAIR 的跨数据集排秩需多库；本文是单库的机制性实证，非跨库结论。
- **config 近乎并列**这一良性解释成立于 HAM 的 6 config 网格；更宽的搜索空间可能拉开 config 差异，
  届时选择信号的有效性需重估。

---

## 6. 可复现

| 产物 | 路径 / 命令 |
|------|------------|
| CV 搜索提交 | `slurm/c1_ham_cv_search.sh`（array 0–29 = 6 config × 5 折；作业 70771–70774） |
| 选择探针 | `scripts/selection_strategy_probe.py`（`--mode single` / `--mode cv`） |
| 单-split 结果 | `python -m scripts.selection_strategy_probe --mode single` |
| CV 双端降噪结果 | `python -m scripts.selection_strategy_probe --mode cv` |
| CV 预测 | `outputs/ham10000/cv5/predictions/`（6 config × 5 折 × 4 方法，120 run，0 错误） |
