# 超网络公平性比较 —— 结果汇总报告

> 生成日期：2026-07-03。本文件是比较协议（`docs/comparison_protocol.md`）**产出（第 6 节）**的落地。
> 所有数字由 `scripts/build_results_tables.py` 从落盘预测 npz 统一重算，口径与训练/模型选择一致
> （数据集感知 `subgroup_auc_vector` + `worst_and_gap` + Student-t 95%CI）。**ID 主口径 = overall
> checkpoint 选择、5-seed 均值±95%CI**（worstcase 选择经核对结论定性一致）；PAPILA 为全-CV，报 5 折
> disjoint test 池化的全 420 眼 OOF 点估计。

---

## D1 报告框架与核心论点

**核心论点**（协议 §0）：超网络（HN）的价值不在超越 SOTA 整体性能，而在**融合图像本身无法表达、
但与诊断相关的敏感属性信息**。故 HN 相对 attribute-blind 池化基线的增益，**当且仅当** `I(Y;A|X) > 0`
（给定图像 X 后属性 A 仍携带子群专属决策结构）时才可能出现。若 `I(Y;A|X) ≈ 0`，盲池化已是 Bayes
最优，任何属性条件化架构顶多退化成它。

**本报告的验证结构**：把 conditional-MI 估计 `I(Y;A|X)`（模型无关的信号闸门）与 6 方法 × 5 数据集
的性能/公平结果**并置**——在信号存在处（HAM-age）检验增益，在信号缺失处解释 null result；并用
**R1 单数据集内合成信号剂量-反应**补上跨数据集观测无法给的**因果充分性**证据（详见 D7-0 与
`docs/r1_synthetic_dose_response_results.md`）。**证据分向陈述**（必要 vs 充分，强度不同）见 **D7-0**。

**对比方法（6）**：3 基线（ERM、SWAD、ROC）+ 3 HN（HyperHead 浅→HyperFusion 中→HyperAdapt 深，
属性介入深度递增）。**数据集（5）**：HAM10000、MIMIC-CXR、CheXpert、Fitzpatrick17k、PAPILA。
**OOD**：仅 MIMIC↔CheXpert 双向。

---

## D2 条件互信息 `I(Y;A|X)`（信号闸门，与性能并置）

模型无关估计（冻结 baseline 特征 φ(X)，比较 `Y~φ(X)` vs `Y~[φ(X),A]` 的 held-out ΔNLL；
int 模型 ΔNLL = `I(Y;A|X)` 估计，单位 nats；正控制探针 E3 已验证各数据集探针有功效）。

**（R3.1 补强）每格改报「点估计 [95% CI]」，不再用裸点估计。** CI 口径：逐 backbone seed 的
test 集 1000× bootstrap CI（各数据集 `conditional_mi.json` 的 `ci95_nats`）经 **Rubin's rules**
合并层内（test 采样）与层间（3 seed）两层不确定性；PAPILA 为单次全-CV 池化，用其单run bootstrap CI。
生成脚本 [scripts/summarize_conditional_mi_ci.py](../scripts/summarize_conditional_mi_ci.py)
→ `outputs/conditional_mi_ci_summary.json`。**判定基于「MI 真值 ≥ 0」这一硬约束**：只有 CI 下界 > 0
才算「确认 > 0」；CI 跨 0 = **不可判定（undecidable）**；负点估计且 CI 全 < 0 = 落在**噪声地板**
（held-out ΔNLL 在 g 过拟合时向下偏，真值 ≈ 0）。

| 数据集 | 敏感轴 | `I(Y;A|X)` 点估计 [95% CI]（nats） | 判定 |
|--------|--------|-----------------------------------|------|
| **HAM10000** | **age** | **+0.01379 [+0.00064, +0.02694]** | **确认 > 0（唯一正信号）** |
| **HAM10000** | **sex×age joint** | **+0.01202 [+0.00009, +0.02395]** | **确认 > 0（含 age 轴）** |
| HAM10000 | sex | +0.00281 [−0.00692, +0.01254] | 不可判定（CI 跨 0） |
| MIMIC-CXR | sex | −0.00099 [−0.00253, +0.00055] | 不可判定（CI 跨 0） |
| MIMIC-CXR | race | −0.00072 [−0.00120, −0.00023] | 噪声地板（真值 ≈ 0） |
| MIMIC-CXR | age | −0.00088 [−0.00174, −0.00002] | 噪声地板（真值 ≈ 0） |
| MIMIC-CXR | joint | −0.00378 [−0.00703, −0.00053] | 噪声地板（真值 ≈ 0） |
| CheXpert | sex | −0.00060 [−0.00197, +0.00077] | 不可判定（CI 跨 0） |
| CheXpert | race | −0.00088 [−0.00230, +0.00055] | 不可判定（CI 跨 0） |
| CheXpert | age | −0.00054 [−0.00135, +0.00026] | 不可判定（CI 跨 0） |
| CheXpert | joint | −0.00638 [−0.01564, +0.00287] | 不可判定（CI 跨 0） |
| Fitzpatrick17k | skin (I–VI) | −0.00182 [−0.01636, +0.01271] | 不可判定（CI 跨 0） |
| Fitzpatrick17k | skin (bin) | +0.00005 [−0.00615, +0.00626] | 不可判定（CI 跨 0） |
| PAPILA | age | +0.01125 [−0.01342, +0.03775] | 不可判定（灰区，见 R3.3） |
| PAPILA | sex | −0.01306 [−0.03660, +0.01076] | 不可判定（灰区，见 R3.3） |
| PAPILA | joint | +0.00805 [−0.04150, +0.05451] | 不可判定（灰区，见 R3.3） |

**关键（据 CI 修订，比旧「全部 ≈0」更诚实）**：
- **仅 HAM-age（及含 age 的 sex×age joint）的 CI 下界 > 0**，是全研究**唯一被置信区间确认的正信号**。
- **MIMIC race/age/joint 的 CI 全 < 0**，落在估计器噪声地板 —— 与「真值 ≈ 0、方差主导」一致（负 ΔNLL
  不代表负 MI，而是 g 在无信号时的过拟合罚）。
- **CheXpert（全轴）、Fitzpatrick、PAPILA、MIMIC-sex、HAM-sex 的 CI 跨 0 = 不可判定**：这些**不是**
  「确认为零」，而是「在本数据规模下未能与零区分」。「不可判定」区间有多大、探针能检出多小的注入信号
  （MDE），见下节 **R3.2**；PAPILA 的灰区定位与退出 null 证据集见 **R3.3**。

据此，**理论预测：仅 HAM 可能出现 HN 增益**（唯一 CI 确认 > 0 的数据集）；其余数据集在其 MDE 之上
无可利用信号，HN 预测打平基线。下文性能结果逐一检验此预测。

### D2-MDE 逐数据集最小可检测效应与正控制探针功效（R3.2）

「不可判定」区间有多宽，取决于探针在该数据规模下能检出多小的注入信号。E3 灵敏度标定（seed 42）
以概率 α 用与图像无关的随机属性覆盖标签（注入强度 α 的**纯条件信号**），量出每数据集的
**最小可检测效应 MDE**（首个 bootstrap CI 下界 > 0 的档的实测 ΔNLL）与正控制（Y XOR r 强信号，
探针必须检出）功效。脚本同 [scripts/summarize_conditional_mi_ci.py](../scripts/summarize_conditional_mi_ci.py)。

| 数据集 | test n | MDE 括区（nats） | 正控制 XOR ΔNLL | 探针功效 | 「null / 不可判定」可信区间 |
|--------|--------|------------------|-----------------|---------|------------------------------|
| MIMIC-CXR | ~17.5k | 0.001（`−0.0009 < MDE ≤ +0.0012`） | +0.257（检出） | ✅ 有功效 | 能检出 ≳0.001 nats；实测全轴 \|I\|≲0.001 且点估计为负 ⇒ **null 可信** |
| CheXpert | ~12.2k | 0.006（`+0.0007 < MDE ≤ +0.0057`） | +0.483（检出） | ✅ 有功效 | 能检出 ≳0.006 nats；实测全轴 \|I\|≲0.001 ⇒ 对 >0.006 的信号 null 可信，更小则不可判定 |
| Fitzpatrick17k | 1602 | 0.015（`+0.0038 < MDE ≤ +0.0150`） | +0.456（检出） | ✅ 有功效 | 仅能检出 ≳0.015 nats；实测 ≈0 ⇒ 对 >0.015 的信号 null 可信 |
| HAM10000 | 969 | **0.05**（`+0.0053 < MDE ≤ +0.0476`） | +0.425（检出） | ✅ 有功效 | 单-seed 探针仅能检出 ≳0.05 nats；**实测 age +0.0138 落在 MDE 括区内** —— 见下方警示 |
| PAPILA | 420 | **不可定义**（检出随 α 非单调：α=0.05 检出、α=0.1 反而未检出） | +0.305（检出） | ⚠️ **不可靠** | n=420 单档实现噪声压过信号 ⇒ **灰区，退出 null 证据集（R3.3）** |

**读法与三点结论**：
1. **MDE 随 √(test n) 收缩**：胸片两库 n 大、MDE 低到 0.001–0.006 nats，故其「未检出」是在**很紧的地板**
   上做出的 —— MIMIC 全轴点估计为负、落在噪声地板，null 结论**可信**；CheXpert 对 >0.006 nats 的信号也
   可信为 null，只有更小的量级不可判定（诚实地不再声称「确认为零」）。
2. **⚠️ HAM-age 的「确认 > 0」是薄的、依赖多-seed 池化**：单-seed E3 探针在 HAM（n=969）的 MDE≈0.05 nats，
   **高于实测 age 信号 +0.0138**；D2 的 CI 之所以能排除 0，是靠 **3-seed Rubin 池化**把区间收窄了约 √3 并
   纳入 seed 间信息。即：同规模的单次运行**检不出** 0.0138 nats 的信号。这不推翻 HAM-age（多-seed CI 是
   合法的不确定性口径），但说明其正信号**幅度小、余量薄**，直接支撑 **R4（扩 seed 提升功效）** 的必要性。
3. **PAPILA 探针在 n=420 下不可靠**（剂量-反应非单调），无法界定 MDE ⇒ 其 `I(Y;A|X)` 估计既不能证实也
   不能证伪，**判为灰区并退出 null 证据集**（R3.3 落实表述订正）。

### D2-agebin age 分箱敏感性：MIMIC/CheXpert 的 null 非分箱伪影（R3.4）

D2 的 age 结果用的是二值化 @60 单一分箱。为排除「null 只是分箱选择的伪影」，在**同一份冻结特征缓存**上
用 5 种分箱（二值阈值 50/60/70 + 3 分位 tertile + 4 分位 quartile；边界由 train 原始 age 定）重估
`I(Y;age|X)`，3 seed Rubin 合并。脚本 [scripts/agebin_sensitivity_cmi.py](../scripts/agebin_sensitivity_cmi.py)
→ `outputs/agebin_sensitivity_cmi.json`。（`bin2@60` 复现 R3.1：MIMIC −0.00088、CheXpert −0.00055，逐位对齐。）

| 数据集 | @50 | @60（基线） | @70 | tertile(3) | quartile(4) |
|--------|-----|-------------|-----|------------|-------------|
| MIMIC-CXR | −0.0009 地板 | −0.0009 地板 | −0.0008 跨0 | −0.0015 地板 | −0.0020 地板 |
| CheXpert | −0.0008 跨0 | −0.0006 跨0 | −0.0008 跨0 | −0.0013 跨0 | −0.0024 跨0 |

**结论**：两数据集在**全部 5 种分箱**下 `I(Y;age|X)` 点估计一律为负、CI 无一「确认 > 0」（MIMIC 多落负噪声
地板、CheXpert 全跨 0）——**null 对分箱稳健，非分箱伪影**。且**越细的分箱（tertile→quartile）点估计越负**
（g 组数增多、无信号时纯过拟合罚加大），进一步佐证不存在被粗分箱掩盖的隐藏 age 信号。

---

## D3 / D5 数据集内 ID 性能与公平（Overall / Worst-group / AUC gap）

每格为 5-seed 均值 ± 95%CI（Student-t）。worst-group = 数据集全子群向量的最小 AUC；gap = max−min。

### HAM10000（sex + age；唯一 `I(Y;A|X)>0`）
| 方法 | Overall AUC | Worst-group AUC | AUC gap |
|------|-------------|-----------------|---------|
| ERM | 0.8827 ± 0.0069 | 0.6988 ± 0.1526 | 0.2902 ± 0.1657 |
| SWAD | 0.8939 ± 0.0046 | 0.7637 ± 0.0427 | 0.2279 ± 0.0636 |
| ROC | 0.8800 ± 0.0088 | 0.6942 ± 0.1490 | 0.2942 ± 0.1627 |
| HyperHead | 0.8997 ± 0.0054 | 0.7579 ± 0.0626 | 0.2356 ± 0.0695 |
| HyperFusion | 0.8919 ± 0.0183 | 0.7362 ± 0.0923 | 0.2442 ± 0.0714 |
| HyperAdapt | **0.9012 ± 0.0116** | 0.7575 ± 0.0360 | 0.2425 ± 0.0360 |

> worst-group 含 8 个 sex×age joint 格（最小 ~26 眼），噪声极大（ERM ±0.15）。用**边缘子群（sex+age，
> 6 组）**隔离 joint 噪声后：ERM 0.754±0.047 / SWAD 0.799±0.029 / HyperHead **0.812±0.021** /
> HyperFusion 0.783±0.062 / HyperAdapt 0.790±0.043 —— **HN 与 SWAD 一致抬升 worst-group 高于 ERM**，
> 方向与 `I(Y;age|X)>0` 相符；但小样本使 CI 仍交叠（见 D6 显著性）。

### MIMIC-CXR（sex + race + age；`I≈0`）
| 方法 | Overall AUC | Worst-group AUC | AUC gap |
|------|-------------|-----------------|---------|
| ERM | 0.8456 ± 0.0039 | 0.8078 ± 0.0079 | 0.0775 ± 0.0085 |
| SWAD | 0.8507 ± 0.0015 | 0.8153 ± 0.0048 | 0.0743 ± 0.0073 |
| ROC | 0.8453 ± 0.0039 | 0.8067 ± 0.0067 | 0.0782 ± 0.0076 |
| HyperHead | 0.8462 ± 0.0015 | 0.8103 ± 0.0056 | 0.0799 ± 0.0094 |
| HyperFusion | 0.8461 ± 0.0021 | 0.8056 ± 0.0072 | 0.0819 ± 0.0080 |
| HyperAdapt | 0.8459 ± 0.0023 | 0.8057 ± 0.0098 | 0.0818 ± 0.0099 |

### CheXpert（sex + race + age；`I≈0`）
| 方法 | Overall AUC | Worst-group AUC | AUC gap |
|------|-------------|-----------------|---------|
| ERM | 0.8689 ± 0.0029 | 0.7778 ± 0.0229 | 0.1142 ± 0.0198 |
| SWAD | **0.8772 ± 0.0027** | 0.7986 ± 0.0145 | 0.0984 ± 0.0159 |
| ROC | 0.8474 ± 0.0170 | 0.7709 ± 0.0165 | 0.1136 ± 0.0187 |
| HyperHead | 0.8715 ± 0.0027 | 0.7789 ± 0.0166 | 0.1132 ± 0.0179 |
| HyperFusion | 0.8730 ± 0.0025 | 0.7948 ± 0.0087 | 0.0977 ± 0.0093 |
| HyperAdapt | 0.8714 ± 0.0020 | 0.7978 ± 0.0241 | 0.0949 ± 0.0263 |

### Fitzpatrick17k（skin I–VI；`I≈0`）
| 方法 | Overall AUC | Worst-group AUC | AUC gap |
|------|-------------|-----------------|---------|
| ERM | 0.8947 ± 0.0128 | 0.7850 ± 0.0494 | 0.1528 ± 0.0420 |
| SWAD | **0.9108 ± 0.0033** | 0.7808 ± 0.0805 | 0.1670 ± 0.0800 |
| ROC | 0.8944 ± 0.0129 | 0.7850 ± 0.0494 | 0.1528 ± 0.0421 |
| HyperHead | 0.8892 ± 0.0138 | 0.7721 ± 0.0320 | 0.1570 ± 0.0336 |
| HyperFusion | 0.8807 ± 0.0071 | 0.7465 ± 0.0762 | 0.1860 ± 0.0764 |
| HyperAdapt | 0.9023 ± 0.0101 | 0.8172 ± 0.0959 | 0.1213 ± 0.0902 |

### PAPILA（sex + age；全-CV OOF n=420；`I` **灰区/不可判定** —— 探针 n=420 不可靠，见 D2-MDE(R3.2)，**不计入 null 证据**）
| 方法 | Overall AUC | Worst-group AUC | AUC gap |
|------|-------------|-----------------|---------|
| ERM | 0.8414 | 0.7952 | 0.0629 |
| SWAD | 0.8593 | 0.8135 | 0.0752 |
| ROC | 0.8414 | 0.7952 | 0.0629 |
| HyperHead | 0.8111 | 0.6592 | 0.1955 |
| HyperFusion | 0.7959 | 0.6652 | 0.1993 |
| HyperAdapt | **0.8603** | 0.7830 | 0.1049 |
> 数字为 5 折 disjoint test 池化的全 420 眼 OOF 点估计（非 seed 均值）。HyperHead/HyperFusion 在 PAPILA
> 明显退化（worst 0.66），HyperAdapt 打平 baseline；无 HN 改善公平。

---

## D4 OOD 性能（CXR：MIMIC ↔ CheXpert 双向）

源数据集确认模型 → 目标数据集 test，5 方法 × 5-seed 均值 ± 95%CI（ROC 属分数后处理，不在 OOD）。

### MIMIC → CheXpert
| 方法 | Overall AUC | Worst-group AUC | AUC gap |
|------|-------------|-----------------|---------|
| ERM | 0.8532 ± 0.0054 | 0.7509 ± 0.0177 | 0.1319 ± 0.0143 |
| SWAD | 0.8526 ± 0.0035 | 0.7315 ± 0.0306 | 0.1533 ± 0.0294 |
| HyperHead | 0.8547 ± 0.0054 | 0.7433 ± 0.0145 | 0.1368 ± 0.0187 |
| HyperFusion | 0.8517 ± 0.0073 | 0.7306 ± 0.0348 | 0.1542 ± 0.0294 |
| HyperAdapt | **0.8561 ± 0.0031** | 0.7401 ± 0.0224 | 0.1464 ± 0.0280 |

### CheXpert → MIMIC
| 方法 | Overall AUC | Worst-group AUC | AUC gap |
|------|-------------|-----------------|---------|
| ERM | 0.8114 ± 0.0040 | 0.7702 ± 0.0107 | 0.0900 ± 0.0072 |
| SWAD | **0.8206 ± 0.0022** | **0.7886 ± 0.0032** | 0.0804 ± 0.0032 |
| HyperHead | 0.8128 ± 0.0012 | 0.7765 ± 0.0064 | 0.0948 ± 0.0077 |
| HyperFusion | 0.8148 ± 0.0014 | 0.7780 ± 0.0060 | 0.0919 ± 0.0079 |
| HyperAdapt | 0.8159 ± 0.0026 | 0.7796 ± 0.0032 | 0.0888 ± 0.0055 |

**OOD 结论**：两方向所有方法 CI 大幅重叠，HN ≈ baseline。唯 **SWAD 在 CheXpert→MIMIC 有小幅域泛化
优势**（Overall 0.821 / worst 0.789，符其权重平均设计），HN 无 OOD 增益——与 MIMIC/CheXpert `I≈0`
（无条件信号可利用）一致。

---

## D6 数据集内显著性（HAM，唯一信号数据集）—— 完整 3 HN × 3 基线矩阵

**全 3 HN × 3 基线矩阵是比较协议的正确设计**（所有方法 × 所有基线全比）。协议 §5 一度把 HAM 显著性缩窄到
「仅 HyperFusion」属**规格/执行失误**——完整方法比较本应覆盖全部 HN；该失误已在 C 阶段统一重训时一并订正。
故此处的全矩阵**不是**看到 Pareto 反转后的数据驱动探索性扩展（**无 forking-path / researcher-degrees-of-freedom**），
Pareto 反转只是**暴露该失误的契机**、非扩展的理由。此外加**边缘 worst-group** 口径（隔离 sex×age joint 小格
~26 眼噪声，见 D6-diag/R4.2）。worst-group 配对 bootstrap = 样本×种子两层不确定度 2000 次；Overall 用 DeLong。
（同时报多个比较仍需多重性校正，见 **D6-FDR/R4.3**——这与是否预设无关。）

**D6-a　worst-group 配对 bootstrap（HN − 基线；canonical 14 子群，含 joint 小格）** —— Δ [95% CI]
| HN \ 基线 | vs ERM | vs SWAD | vs ROC |
|-----------|--------|---------|--------|
| HyperHead | +0.296 [−0.160,+0.403] n.s. | −0.081 [−0.191,+0.186] n.s. | −0.054 [−0.157,+0.414] n.s. |
| HyperFusion | +0.303 [−0.221,+0.376] n.s. | +0.039 [−0.231,+0.177] n.s. | −0.173 [−0.194,+0.415] n.s. |
| HyperAdapt | −0.033 [−0.118,+0.413] n.s. | −0.061 [−0.142,+0.193] n.s. | +0.035 [−0.108,+0.424] n.s. |

**D6-b　边缘 worst-group 配对 bootstrap（HN − ERM；仅 sex/age 6 组，隔离 joint 噪声）**
| 对比 | Δ 边缘worst | 95% CI | p | 判定 |
|------|-------------|--------|---|------|
| HyperHead vs ERM | +0.0452 | [−0.0451, +0.2004] | 0.412 | n.s. |
| HyperFusion vs ERM | +0.0843 | [−0.1263, +0.1744] | 0.680 | n.s. |
| HyperAdapt vs ERM | +0.0668 | [−0.0804, +0.1882] | 0.644 | n.s. |

**D6-c　Overall AUC DeLong（HN vs 基线；5-seed 平均 ΔAUC，p）**
| HN \ 基线 | vs ERM | vs SWAD | vs ROC |
|-----------|--------|---------|--------|
| HyperHead | +0.0170 (p=0.195) | +0.0058 (p=0.507) | +0.0196 (p=0.161) |
| HyperFusion | +0.0092 (p=0.254) | −0.0020 (p=0.317) | +0.0119 (p=0.251) |
| HyperAdapt | +0.0186 (p=0.287) | +0.0074 (p=0.394) | +0.0212 (p=0.250) |

**显著性结论（完整矩阵，D6-a/b/c）**：在 HAM（唯一 `I(Y;A|X)>0`）上，D6-a/b/c（5-seed 两层不确定度）
下**全部 3 HN × 3 基线、跨 canonical worst / 边缘 worst / Overall AUC 三种口径，无一对比达到 p<0.05**。但
**点估计方向系统性为正**——3 HN 相对 ERM 在 Overall（+0.009~+0.019）与边缘 worst（+0.045~+0.084）上一致
为正，符合 `I(Y;age|X)>0` 的预测方向；最接近显著的是 HyperHead/HyperAdapt 的 Overall vs ERM/ROC（p≈0.16~0.29）。

### D6-HP 更高功效补强（R4.1：seed-ensemble 样本级 bootstrap / DeLong）

D6-a/b/c 的功效受限于 **n=5 seed**（两层 bootstrap 每次仅抽 1 个 seed，注入全单-seed 方差）。R4.1 走协议
sanction 的**跨 seed 池化**路径（非重训 15 seed）：把 5 seed 的预测在**概率空间平均**成 seed-ensemble（模型方差
降 ~√5），再做**样本级**配对 bootstrap / DeLong（只保留 test 采样不确定度）。生成于
[scripts/build_results_tables.py](../scripts/build_results_tables.py) `ham_significance_highpower()`（`--section sig`）。

**D6-HP-a　seed-ensemble worst-group 样本级配对 bootstrap（canonical 14 子群，HN − 基线）**
| HN \ 基线 | vs ERM | vs SWAD | vs ROC |
|-----------|--------|---------|--------|
| HyperHead | −0.007 [−0.101,+0.138] n.s. | −0.028 [−0.131,+0.122] n.s. | −0.000 [−0.079,+0.148] n.s. |
| HyperFusion | +0.036 [−0.054,+0.170] n.s. | +0.014 [−0.064,+0.151] n.s. | +0.042 [−0.043,+0.176] n.s. |
| HyperAdapt | +0.035 [−0.035,+0.225] n.s. | +0.014 [−0.070,+0.197] n.s. | +0.042 [−0.035,+0.208] n.s. |

**D6-HP-b　seed-ensemble 边缘 worst-group（仅 sex/age 6 组，HN − ERM）**
| 对比 | Δ | 95% CI | p | 判定 |
|------|---|--------|---|------|
| HyperHead vs ERM | +0.0485 | [−0.0448, +0.1372] | 0.506 | n.s. |
| HyperFusion vs ERM | +0.0355 | [−0.0345, +0.1345] | 0.464 | n.s. |
| HyperAdapt vs ERM | +0.0815 | [−0.0084, +0.1891] | **0.099**（趋势） | n.s. |

**D6-HP-c　seed-ensemble Overall AUC DeLong（HN vs 基线；集成分数上 test 采样口径）**
| HN \ 基线 | vs ERM | vs SWAD | vs ROC |
|-----------|--------|---------|--------|
| HyperHead | Δ+0.0141 (p=0.0051 **显著**) | Δ+0.0110 (p=0.074) | Δ+0.0164 (p=0.0016 **显著**) |
| HyperFusion | Δ+0.0178 (p=0.0113 **显著**) | Δ+0.0147 (p=0.038 **显著**) | Δ+0.0202 (p=0.0055 **显著**) |
| HyperAdapt | Δ+0.0230 (p=0.0001 **显著**) | Δ+0.0199 (p=0.0026 **显著**) | Δ+0.0253 (p<0.0001 **显著**) |

**更高功效判决（D6-HP）**：
1. **Overall AUC 增益从"方向对未达显著"升级为"显著"**：seed-ensemble DeLong 下，全部 3 HN **显著跑赢
   ERM 与 ROC**（p=0.0001~0.0051 vs ERM）；HyperFusion/HyperAdapt 亦显著赢 SWAD，仅 HyperHead vs SWAD
   打平（p=0.074）。即在充分功效下，HAM 上 HN 的**整体判别增益是真实且显著的**（口径：条件于 ensemble 的
   test 采样不确定度，不再传播 seed 方差——比 D6-c 更高功效、更窄口径，两者并置读）。
2. **worst-group 仍不显著、但 CI 收窄约 2×**：ensemble 把 D6-a 的 seed 噪声压掉后（如 HyperAdapt vs ERM
   由 [−0.118,+0.413] 收到 [−0.035,+0.225]），canonical worst 依旧 n.s.，边缘 worst 最好也只到 HyperAdapt
   p=0.099。**这判定了瓶颈所在**：worst-group 的功效不足**主要来自评估端子群小样本**（joint 格 ~26 眼，见
   **R4.2**），**而非 seed 数**——故重训到 15 seed 也无法救 worst-group 显著性，正当化 R4.1 选择 bootstrap
   路径而非扩 seed。
3. **合并判决**：HAM 上 HN 兑现了**显著的 Overall 增益**与**方向一致但受评估-n 限制、未达显著的 worst-group
   改善**，二者都与 `I(Y;age|X)>0` 的预测方向吻合。相较 D6 旧结论（"无一显著"），R4.1 把 Overall 一侧
   由"趋势"升级为"证实"，同时诚实地把 worst-group 的未达显著**归因到评估端小样本**而非机制或信号缺失。

### D6-diag 评估端子群样本量诊断：界定 canonical worst-group 可信下限（R4.2）

R4.1 把 worst-group 的未达显著归因到「评估端小样本」；R4.2 逐格量化坐实之。对 HAM canonical 14 子群
（seed-ensemble/ERM 代表分数）报 **n / n_pos / n_neg / AUC / 三口径 AUC-SE**（Hanley–McNeil 解析 +
DeLong 解析 + 格内 bootstrap，互证），噪声地板阈 = 任一 SE≥0.10 或 min(n_pos,n_neg)<10。脚本
[scripts/subgroup_n_diagnosis.py](../scripts/subgroup_n_diagnosis.py) → `outputs/ham10000/subgroup_n_diagnosis.json`。

| 子群 | n | n_pos | n_neg | AUC | SE(HM / DeLong / boot) | 地板 |
|------|---|-------|-------|-----|------------------------|------|
| sex:Male | 533 | 99 | 434 | 0.909 | 0.020 / 0.015 / 0.015 | |
| sex:Female | 436 | 55 | 381 | 0.883 | 0.030 / 0.022 / 0.022 | |
| age:40-60 | 421 | 39 | 382 | 0.936 | 0.028 / 0.019 / 0.019 | |
| age:60-80 | 296 | 78 | 218 | 0.877 | 0.027 / 0.023 / 0.023 | |
| age:20-40 | 162 | 11 | 151 | 0.786 | 0.083 / 0.075 / 0.076 | (临界) |
| age:80+ | 90 | 26 | 64 | 0.850 | 0.050 / 0.049 / 0.050 | |
| sex:M\|age:40-60 | 217 | 20 | 197 | 0.969 | 0.027 / 0.011 / 0.010 | |
| sex:M\|age:60-80 | 188 | 54 | 134 | 0.884 | 0.031 / 0.028 / 0.028 | |
| sex:F\|age:40-60 | 204 | 19 | 185 | 0.918 | 0.044 / 0.036 / 0.036 | |
| sex:F\|age:60-80 | 108 | 24 | 84 | 0.860 | 0.050 / 0.041 / 0.041 | |
| sex:M\|age:20-40 | 65 | **3** | 62 | 0.828 | **0.149** / 0.058 / 0.056 | ⚠️噪声地板 |
| sex:M\|age:80+ | 63 | 22 | 41 | 0.824 | 0.060 / 0.058 / 0.056 | |
| sex:F\|age:20-40 | 97 | **8** | 89 | 0.805 | 0.096 / 0.091 / 0.093 | ⚠️噪声地板 |
| sex:F\|age:80+ | 27 | **4** | 23 | **1.000** | nan / nan / 0.000* | ⚠️噪声地板 |

\* Female\|80+ 的 bootstrap SE=0 是**退化**（4 正例把 AUC 钉在 1.0，重采样几乎恒 1.0），HM/DeLong SE 直接不可定义——恰是"该格不可信"的最强信号，非"零方差"。

**结论（可信下限）**：
1. **3/14 格进入噪声地板，全部是 joint 小格**（sex×age，n_pos = 3 / 8 / 4）。joint 格 AUC-SE 普遍 0.05–0.15，
   是边缘格（0.02–0.05）的 **2–5 倍**；worst-group = 全格取 min，**恒被这几个高方差小格拖走**，D3 表 ERM
   worst-group ±0.15 的巨幅 CI 由此而来。
2. **canonical-joint worst-group 口径的可信下限被 ~3 个 n_pos<10 的格压破**，其 min-AUC 主要在测噪声、非真实
   最差子群性能。故 **HAM 公平的可信口径应以边缘 worst-group（6 组，最小格 age:20-40 n_pos=11，SE≈0.08 临界
   但可用）为准**——这正是 D3 脚注与 D6-b/D6-HP-b 采用的口径，本诊断为其提供了量化依据。
3. **此问题是 HAM 独有**：其余数据集 worst-group 最小格远大于 HAM（MIMIC 855 / CheXpert 527 / Fitzpatrick 55，
   见 JSON），joint 小格噪声地板不构成威胁；故 canonical-joint 口径仅在 HAM 需降级为边缘口径。

### D6-FDR 多重比较校正（R4.3：Benjamini–Hochberg）

D6-HP 同时做 21 个假设检验（Overall AUC 9 + canonical worst 9 + 边缘 worst 3），须做多重性控制。对全部
比较统一 BH-FDR，**按度量分族**（Overall / canonical worst / 边缘 worst 各成族，跨度量合并 p 无意义）与
**全局单族**（21 合一，更严）两口径并报。脚本
[scripts/ham_multiplicity_fdr.py](../scripts/ham_multiplicity_fdr.py) → `outputs/ham10000/d6_multiplicity_fdr.json`。

> **非 forking-path 声明**：全 3×3 矩阵是比较协议的正确设计；协议 §5 一度缩到「仅 HF」系规格失误、已订正，
> **非**数据驱动的探索性扩展，故不存在 researcher-degrees-of-freedom 的假阳性膨胀。但同时报多比较仍需下方
> 多重性校正——**这与是否预设无关**（预设的全矩阵同样需要 BH-FDR）。

**Overall AUC 族 BH-FDR（q 值 = 校正后 FDR）**
| HN \ 基线 | vs ERM | vs SWAD | vs ROC |
|-----------|--------|---------|--------|
| HyperHead | q=0.008（族内✓/全局✓） | q=0.074（✗/✗） | q=0.005（✓/✓） |
| HyperFusion | q=0.015（✓/✓） | q=0.043 族内✓ / 全局 q=0.100✗ | q=0.008（✓/✓） |
| HyperAdapt | q=0.006（✓/✓） | q=0.006（✓/✓） | q=0.0004（✓/✓） |

（canonical worst 9 格与边缘 worst 3 格的 raw p 全 >0.05——canonical 0.34~0.98、边缘最小 HyperAdapt 0.099——校正后必不存活，从略；完整表见 JSON。）

**多重性校正判决（R4.3）**：
1. **Overall AUC 的 HN 增益经多重性校正后仍稳健显著**：**全部 3 HN 显著跑赢 ERM 与 ROC**，在**族内**（8/9 存活）
   与更严的**全局 21-比较**（7/9 存活）BH-FDR 下**均存活**（q 从 0.0004 到 0.034）。即 R4.1 的"Overall 增益显著"
   **不是多重比较捞出的假阳性**。
2. **唯一被校正削弱的是 vs SWAD**：HyperHead vs SWAD 校正前后均 n.s.（q=0.074）；HyperFusion vs SWAD 族内存活、
   全局边缘失守（q=0.100）；仅 HyperAdapt vs SWAD 稳健存活。符合 SWAD 是强公平基线的既有观察。**注（R5.2）**：
   vs SWAD 比较**不对称**（SWAD 未独立搜参、搭 ERM 车），HN 在此列的胜差本就可能被调优不对称高估——即便如此
   仍多被 FDR 削弱，故主结论不建于此列（见 D7 结论 4 的对称性声明）。
3. **worst-group 全族 21→0 存活**：canonical / 边缘 worst 的 raw p 全 >0.05，校正后无一显著——与 R4.2「评估端
   小样本地板」一致，多重性校正不改此判。

---

## D8 操作点（阈值相关）公平（评审补强 R2）

> D3–D6 的公平只挂 **AUC / worst-group AUC / AUC gap**，三者皆**阈值无关**，带来两处盲区：(i) 公平
> 切面过窄——AUC 只测跨子群「排序」一致，不测「在部署阈值上各子群 TPR/FPR 是否平等」（Equalized
> Odds 关心的操作层）；(ii) **让 Reject-Option（ROC）在结构上失效**——ROC 是阈值上的干预，用阈值
> 无关的 worst-group AUC 选其 margin θ 时 θ*=0 恒不劣，故 ROC 在 D3/D5 上**逐元素等于 ERM**（旧口径
> 4/5 数据集调整量=0），其子群奇偶目的从未被测。
>
> 本节全部在**已落盘预测 npz 上重算，零重训**（`scripts/build_operating_point_tables.py` +
> `src/utils/operating_point_fairness.py`，后者 sklearn 交叉验证）。两个操作点：**①原生阈值 prob=0.5**
> （logits 的 0，ROC 锚点）；**②固定整体 FPR=0.2**（类别不均衡稳健）。**ROC 在此按操作点重选 θ**
> （目标=最小化 val EqOdds gap，deprived 按 val 0.5-TPR 二分；网格含 0 → 不劣 ERM），使其阈值干预
> 真正被选中、被测（R2.3）。EqOdds gap=max(TPR 组间极差, FPR 组间极差)，**越小越公平**；worst-TPR=
> 最小边缘子群 TPR，**越大越好**。边缘子群、min_class_n=10、5-seed 均值±95%CI（PAPILA OOF 池化）。

**D8-a EqOdds gap（越小越好）**
| 数据集 | 操作点 | ERM | SWAD | ROC | HyperHead | HyperFusion | HyperAdapt |
|--------|--------|-----|------|-----|-----------|-------------|------------|
| HAM10000 | @0.5 | 0.489±0.238 | 0.384±0.047 | **0.342±0.117** | 0.496±0.178 | 0.386±0.300 | 0.359±0.132 |
| HAM10000 | @FPR.2 | 0.415±0.088 | 0.351±0.058 | 0.415±0.088 | 0.467±0.025 | 0.431±0.093 | **0.342±0.094** |
| MIMIC | @0.5 | 0.280±0.042 | 0.257±0.016 | **0.087±0.051** | 0.272±0.029 | 0.270±0.036 | 0.269±0.040 |
| MIMIC | @FPR.2 | 0.195±0.022 | 0.189±0.008 | **0.169±0.071** | 0.187±0.021 | 0.181±0.015 | 0.201±0.009 |
| CheXpert | @0.5 | 0.200±0.112 | 0.357±0.047 | **0.078±0.066** | 0.399±0.070 | 0.346±0.062 | 0.365±0.117 |
| CheXpert | @FPR.2 | 0.223±0.030 | 0.203±0.015 | 0.223±0.030 | 0.224±0.021 | **0.202±0.052** | 0.237±0.027 |
| Fitzpatrick | @0.5 | 0.240±0.115 | **0.221±0.075** | 0.276±0.082 | 0.297±0.141 | 0.244±0.048 | 0.258±0.173 |
| Fitzpatrick | @FPR.2 | 0.251±0.071 | 0.217±0.100 | 0.251±0.071 | 0.236±0.093 | 0.201±0.045 | **0.156±0.064** |
| PAPILA(OOF) | @0.5 | 0.100 | 0.101 | 0.100 | 0.185 | 0.543 | 0.147 |
| PAPILA(OOF) | @FPR.2 | 0.094 | 0.112 | 0.094 | 0.280 | 0.543 | 0.238 |

**D8-b worst-group TPR（越大越好）**
| 数据集 | 操作点 | ERM | SWAD | ROC | HyperHead | HyperFusion | HyperAdapt |
|--------|--------|-----|------|-----|-----------|-------------|------------|
| HAM10000 | @0.5 | 0.235±0.123 | 0.346±0.065 | 0.315±0.048 | 0.200±0.257 | 0.091±0.080 | **0.455±0.160** |
| HAM10000 | @FPR.2 | 0.578±0.098 | 0.673±0.062 | 0.578±0.098 | 0.624±0.098 | 0.527±0.147 | **0.669±0.056** |
| MIMIC | @0.5 | 0.456±0.084 | 0.525±0.032 | **0.564±0.097** | 0.479±0.076 | 0.481±0.056 | 0.499±0.104 |
| MIMIC | @FPR.2 | 0.659±0.019 | 0.672±0.006 | 0.666±0.027 | 0.666±0.012 | 0.668±0.010 | 0.657±0.008 |
| CheXpert | @0.5 | 0.044±0.058 | 0.112±0.039 | 0.064±0.064 | 0.045±0.063 | 0.052±0.062 | 0.064±0.068 |
| CheXpert | @FPR.2 | 0.664±0.019 | **0.688±0.005** | 0.664±0.019 | 0.668±0.021 | 0.685±0.037 | 0.657±0.021 |
| Fitzpatrick | @0.5 | 0.513±0.147 | **0.568±0.055** | 0.474±0.164 | 0.419±0.091 | 0.441±0.093 | 0.442±0.134 |
| Fitzpatrick | @FPR.2 | 0.653±0.088 | 0.705±0.099 | 0.653±0.088 | 0.673±0.088 | 0.719±0.071 | **0.768±0.059** |
| PAPILA(OOF) | @0.5 | 0.400 | 0.457 | 0.400 | 0.200 | 0.200 | 0.333 |
| PAPILA(OOF) | @FPR.2 | 0.667 | 0.657 | 0.667 | 0.467 | 0.200 | 0.600 |

**D8-c ROC 操作点重选小结（θ* 由 val EqOdds gap 选定，含 0 → 不劣 ERM）**
| 数据集 | 二分轴 | 平均 θ* | ROC test EqOdds gap @0.5（ERM → ROC） |
|--------|--------|---------|---------------------------------------|
| HAM10000 | age | 0.230 | 0.489 → **0.342** |
| MIMIC | age | 0.160 | 0.280 → **0.087** |
| CheXpert | age | 0.120 | 0.200 → **0.078** |
| Fitzpatrick | skin | 0.110 | 0.240 → 0.276（test 反弹，见下） |
| PAPILA | — | 0.000 | 0.100 → 0.100（退化 ERM，n 太小） |

**操作点结论**：
1. **ROC「等价 ERM」的假象消除（R2.3 核心）**：把 θ 选择目标从阈值无关的 worst-group AUC 换成**操作点
   EqOdds gap**后，ROC 在 3/5 数据集选出 **θ*>0** 并在**其设计目标上兑现大幅奇偶改善**——MIMIC EqOdds gap
   @0.5 **0.280→0.087**（−69%）、CheXpert **0.200→0.078**（−61%）、HAM **0.489→0.342**——这些正是 AUC 口径
   完全看不到（旧 ROC 逐元素=ERM）的公平收益。**这不改变核心论点**（ROC 是后处理阈值干预、非属性条件化
   架构，其收益与 `I(Y;A|X)` 无关），但补上了「ROC 到底做了什么」这一此前缺失的测量。
2. **ROC 的收益锚定在 0.5、不迁移到其它操作点**：在固定 FPR=0.2 处 ROC 的 EqOdds/worst-TPR 多与 ERM 相同
   （reject-option 只翻转 [0.5−θ,0.5+θ] 带内样本，FPR=0.2 阈值落在带外）。这是 reject-option 的**固有性质**
   （只在决策边界邻域起效），现被明确测出而非被 AUC 口径掩盖；也说明部署操作点远离 0.5 时（不均衡 CXR）
   ROC 杠杆有限。Fitzpatrick 上 val 选中的 θ 在 test 反弹（深肤色小组 val→test 泛化差），CI 交叠。
3. **HN 在操作点公平上仍多为 ≈ERM，与信号未达 MDE 一致**：3 个 null 数据集（MIMIC 真值≈0、
   CheXpert/Fitzpatrick 未达 MDE）各 HN 的 EqOdds gap / worst-TPR 与 ERM 大面积 CI 重叠，无稳健改善；
   PAPILA（灰区、不计入判决）的 HyperHead/HyperFusion 反在操作点上明显退化，与其 AUC 退化同源。**唯一信号数据集 HAM**：HyperAdapt 在两操作点
   均给出**最高 worst-TPR**（@0.5 0.455、@FPR.2 0.669）与偏低 EqOdds gap，方向与 D3/D5/D6 的 AUC 口径
   一致——即 HAM 上属性条件化的收益在操作点公平上同样可见（但同样受小样本 CI 限制）。
4. **原生 0.5 worst-TPR 在极不均衡任务上退化**：CheXpert（正例 5.8%）@0.5 各法 worst-TPR≈0.04–0.11（几乎不
   判正），此时 **FPR=0.2 操作点**（worst-TPR≈0.66）才是信息量所在——两操作点互补，报告并置。
5. **SWAD 仍是稳健公平锚点**：在多处（CheXpert @FPR.2 worst-TPR 0.688、HAM @FPR.2）SWAD 给出最好或次好
   的操作点公平，且方差小，与 D3/D5 判断一致。

> 复算脚本：`scripts/build_operating_point_tables.py`（`--target-fpr` 可调）。度量实现与 sklearn 交叉验证：
> `src/utils/operating_point_fairness.py`（`python -m src.utils.operating_point_fairness` self-test）。
> 子群 schema 单一事实来源：`subgroup_auc.subgroup_masks`（与 AUC 口径同批子群）。

---

## D7 结论

### D7-0 证据分向综合（R5.1）——不再用「闭环证实判据」，改分「必要 / 充分」两向如实陈述

核心论点是一条 **iff**：HN 相对 attribute-blind 基线的增益「**当且仅当** `I(Y;A|X)>0`」。iff 有两个方向，
本研究对二者的支持强度**不同**，须分开陈述（避免旧版「5×6 闭环证实判据」的过度概括）：

- **必要方向（增益 ⟸ 信号；「无信号⇒无增益」）——弱到中等支持**。3 个 null 库 HN 一律打平 attribute-blind
  基线：**MIMIC 是干净负控制**（`I` 落噪声地板、真值≈0，R3.1）；**CheXpert / Fitzpatrick 为「不可判定」**
  （点估计≈0 但 < 各自 MDE，R3.2）——性能打平与「信号未超 MDE」一致，但**不构成「确认零信号」的强证据**，
  故必要方向是「与理论一致」而非「严格证实」。PAPILA 因探针不可靠已退出（R3.3）。

- **充分方向（信号 ⟹ 增益；「有信号⇒有增益」）——获得本研究最强的一块证据，但仅限「整体判别」层面**。
  两条独立证据线：
  1. **R1 合成剂量-反应（因果）**：在单一数据集（MIMIC）内注入可控侧信道信号 `A_syn`，**Overall AUC 增益
     随实测 `I(Y;A_syn|X)` 严格单调上升**（Spearman ρ=+1.0；0 档干净打平；0.02/0.05/0.10 nats 档 ΔOverall
     ≈+.016/+.037/+.066，CI 排除 0）——把 iff 从跨数据集相关升级为**单数据集内因果剂量-反应**。
  2. **HAM 自然信号（观测）**：唯一 `I(Y;age|X)>0` 的真实数据集上，3 HN 的 **Overall AUC 增益经 BH-FDR
     多重校正后仍显著跑赢 ERM/ROC**（R4.1/R4.3，全局 FDR 存活 7/9，q≤0.034）。

- **⚠️ 关键限定（充分方向的边界）**：上述充分性证据**只在 Overall / 池化判别层面成立，不延伸到 worst-group
  公平**——R1 的 worst-group(A_syn) 增益平/微负（边际先验型信号只改池化判别、不改组内最差子群排序）。HAM 的
  worst-group 在单-split 下曾受**评估端小样本地板**限制而「未达显著」（R4.2）；**路线 A（`docs/routeA_ham_cv_oof_fairness.md`）
  已把该地板移除**——改 5 折 lesion GroupKFold、5 折 disjoint test 池化成全 9707 OOF，少数格 n 放大 5–10×、
  **进噪声地板的格 3/14→0/14**。地板移除后，**3 HN 的 worst-group 对 ERM 仍打平（点估计一律微负、CI 跨 0）、
  且显著输给 attribute-blind 的 SWAD**——故先前的「未达显著（疑似评估噪声）」正式升级为**有功效的否定**。即：
  **「信号存在 ⇒ 整体判别增益」获因果+观测双向证实，而「信号存在 ⇒ worst-group 公平改善」（本课题最初动机）
  在 HAM 上（评估-n 地板移除后）获有功效的证伪**。

**净表述**：iff 判据在**整体判别增益**层面获必要（弱-中）+ 充分（因果）双向支持；但 HN 把该信号转化为
**worst-group 公平提升**——本项目的核心公平目标——在 HAM 上（路线 A 解除评估-n 限制后）**未实现且被有功效地证伪**
（对 ERM 打平、显著输 SWAD），机制证据（R1/HAM）一致指向信号先验多改池化、少改组内排序。下方 D7-1~5 为分项支撑。

1. **核心论点成立、但增益被小样本噪声压制**：conditional-MI 干净地区分出唯一被 CI 确认有信号的数据集
   （HAM-age，`I(Y;A|X)=+0.0138`，95%CI 下界>0）；其余数据集中 **MIMIC 落在噪声地板（真值≈0）**、
   **CheXpert/Fitzpatrick 为「不可判定」**（点估计 ≈0 但 < 各自 MDE，见 R3.1/R3.2）、
   **PAPILA 为灰区**（探针 n=420 不可靠，已退出 null 证据集，见 R3.3）。**性能侧完全符合预测的方向**——只有 HAM 上
   HN/属性感知方法一致抬升 worst-group（边缘口径 HyperHead 0.812 vs ERM 0.754），其余数据集 HN 一律
   打平或退化。**5-seed 口径 D6-a/b/c 无一对比达 p<0.05**（方向系统性为正，功效不足）；但 **R4.1 更高功效
   补强（D6-HP，seed-ensemble 样本级 bootstrap/DeLong）把 Overall AUC 一侧升级为显著**——全部 3 HN 显著
   跑赢 ERM/ROC（p=0.0001~0.0051 vs ERM），HyperFusion/HyperAdapt 亦显著赢 SWAD；**且经 BH-FDR 多重比较
   校正后仍稳健**（全部 3 HN vs ERM/ROC 在全局 21-比较 FDR 下均存活，q≤0.034，R4.3）。**worst-group 仍不显著
   （CI 收窄 2× 后依旧跨 0，FDR 后 0 存活），且经诊断瓶颈在评估端子群小样本（joint~26，R4.2）而非 seed 数**。
   ⚠️ **路线 A（`docs/routeA_ham_cv_oof_fairness.md`，5 折 CV-OOF 池化 n=9707、少数格 n×5–10、进噪声地板格 3/14→0/14）已移除该评估-n 地板，worst-group 判决翻新**：地板移除后 3 HN 对 ERM 仍打平（点估计**微负**、CI 跨 0）、且**显著输给 SWAD**——上文单-split「HN 一致抬升 worst-group（HyperHead 边缘 0.812 vs ERM 0.754）」实为**噪声地板内的伪抬升**（CV-OOF 下 HyperHead 边缘 worst 0.783 反低于 ERM 0.797）。净修订：HAM 上 HN 的 **Overall 增益经多重性校正证实显著**；worst-group 公平在**评估-n 地板移除后获有功效的否定**（对 ERM 打平、显著输 SWAD）——与 R1 合成（边际先验型信号只改池化、不改组内排序）机制自洽。

2. **null 数据集判决一致（3 库，PAPILA 已剔除）**：MIMIC、CheXpert、Fitzpatrick 上，HN 相对
   attribute-blind 基线 CI 全重叠，无稳健公平改善；Fitzpatrick 的 HyperFusion 甚至退化。这与
   「`I(Y;A|X)` 未超 MDE ⇒ HN 无收益空间」**一致**（MIMIC 为真值≈0 的干净负控制；CheXpert/Fitzpatrick
   为「不可判定」——性能打平与信号未达 MDE 相符，但不构成「确认零信号」的强证据）。
   **PAPILA 不再计入此判决**：其探针在 n=420 下不可靠（R3.2），性能虽同样无 HN 改善（HyperHead/HyperFusion
   退化、HyperAdapt 打平），但既无法证实也无法证伪 `I≈0`，故仅作趋势旁证、不支撑理论（R3.3 双重计账订正）。

3. **属性介入深度轴无单调收益**：在 null 数据集上 ROC→Head→Fusion→Adapt 的递增介入深度未换来递增收益；
   合规 regime 下深层 HN（HyperAdapt）整体 AUC 常最高但公平不稳健，浅层介入亦未系统占优——介入深度不是
   决定因素，**属性信号是否存在**才是。（ROC 作为阈值后处理，其真实作用被阈值无关的 AUC 口径掩盖；
   见 **D8 操作点公平**——改用操作点目标重选 θ 后，ROC 在 EqOdds gap 上兑现大幅奇偶改善，但收益锚定于
   0.5 决策边界、与 `I(Y;A|X)` 正交，不改变上述结论。）

4. **SWAD 是稳健的公平基线**：作为域泛化锚点，SWAD 在多个数据集（CheXpert ID、CheXpert→MIMIC OOD）
   给出与最优 HN 相当或更好的 worst-group，且方差最小——在**无属性信号**时，权重平均比属性条件化更划算。

   > **⚠️ SWAD 比较不对称声明（R5.2）**：**SWAD 未独立搜参——它复用 ERM 的 Pareto 选定配置 (lr,wd)、
   > 在 ERM 训练上做权重平均**（协议 §「省算力」；每数据集仅 ERM+3HN 各自跑 6 配置 Pareto 搜索，SWAD/ROC
   > 「搭 ERM 的车」）。故涉及 SWAD 的对比**非对称**：3 HN 各自被独立调优，SWAD 未被调优。该不对称**两向解读**：
   > (i) **SWAD 取胜处（CheXpert/Fitz Overall、C→M OOD、HAM worst）是保守的**——SWAD 在**自身未调优**的劣势
   > 下仍赢，独立搜参只会让它更强，故「SWAD 是强基线」不因此被高估；(ii) **HN 赢 SWAD 处（D6-HP Overall
   > HyperFusion/HyperAdapt）须谨慎读**——调优不对称偏向 HN，该胜差可能部分是调优假象。**关键：本报告 HN 的
   > 充分性主结论锚定在 HN vs ERM（二者对称调优）+ R1 单数据集内合成（无跨方法调优问题），不依赖「HN 赢 SWAD」**
   > （且 D6-FDR 已显示 vs SWAD 是最弱一列：HyperHead 打平、HyperFusion 全局失守）。因此**结论不重度依赖
   > SWAD 优势，无需补 SWAD 独立搜参**；如后续要把「HN vs SWAD」当独立卖点，则须先补 SWAD 的 6 配置 Pareto
   > 搜索以恢复对称。

5. **方法论要点**：（i）**等-batch 对照**消除了 pilot 阶段 HyperAdapt b32 OOM 假象；（ii）**Pareto 选定
   配置**改变了 pilot 的方法排序（HAM 上 HyperFusion 的"优势"不复现），说明单配置 pilot 结论不可靠；
   （iii）PAPILA **全-CV OOF 池化**（fold↔seed 映射复用全 harness）抑制了 test≈84 的评估噪声；
   （iv）小子群 joint 格是 worst-group 噪声主源，报告需区分边缘 vs 交叉口径。

**一句话**：`I(Y;A|X)>0` 是 HN 相对 attribute-blind 基线取得增益的数据侧闸门；本研究**分向**给出证据——
**充分方向**由 R1 合成剂量-反应（因果）+ HAM 自然信号（BH-FDR 后显著）**证实**，但**仅限 Overall 整体判别**；
**必要方向**由 3 个 null 库（MIMIC 真值≈0、CheXpert/Fitzpatrick 未达 MDE）打平**弱-中支持**；而 HN 把信号转化为 **worst-group 公平提升**这一核心目标：R1 合成为平/微负，**HAM 经路线 A（5 折 CV-OOF、评估-n 地板 3/14→0/14 已移除）获有功效的否定**（对 ERM 打平、显著输 attribute-blind 的 SWAD）——即**未实现且已被证伪**，非「尚未证实」。
PAPILA 因探针不可靠列为灰区、不计入判决。
