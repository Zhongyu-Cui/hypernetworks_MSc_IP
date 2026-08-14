# OOD 实验 v2 · 预注册（MIMIC↔CheXpert）

> **本文件在任何 v2 结果产出之前冻结。** 假设、指标、统计口径、判定规则一经写定不得事后增补或修改；
> 若执行中发现必须偏离，只能**追加**「偏离记录」小节说明偏离内容与理由，不得改写原条目。
>
> 冻结日期：2026-07-28
> 方案来源：本仓库会话计划（`~/.claude/plans/partitioned-splashing-dewdrop.md`）

---

## 0. 动机（为什么要 v2）

现行 OOD 结果（`docs/oof_regime_results.md` §1.6/§1.7）复用 ID 阶段选出的 config，在 source fold-k
的 checkpoint 上评 target fold-k，**只报 AUC**。ID 侧比较协议本身完善；v2 只针对跨分布评估特有的
三项缺口：

1. **训练随机性与数据划分无法分离**：`seed = 42 + fold`。实测同 config/seed/fold 重跑的逐折
   |ΔAUC| 可达 **0.003**，而现存效应仅 **0.0018–0.0019**（M→C +0.0045 在超参对照下已知主要由
   wd 差异贡献）。效应是否在训练噪声之上，当前设计**无法回答**。
2. **ID 选出的 config 在 OOD 上并非最优**：ERM 在 `lr1e-04_wd1e-03`（非 ID 选中）的 M→C marginal
   worst = 0.8157 > ID 选中 `lr1e-04_wd1e-04` 的 0.8114。该"选择损失"从未被量化或披露。
3. **只报判别、不报校准**：MIMIC 正例率 ~30% vs CheXpert ~5.8–8.4%，属极强 **prior shift**。
   AUC 对患病率不敏感而校准会崩塌，现行结论对此沉默（TRIPOD+AI 要求外部验证必报校准）。

另：本会话已完成的 argmin 归属诊断（`outputs/ood_cxr/argmin_attribution/`）显示 **M→C 的 canonical
worst 的 argmin 不稳**（两方法 worst 落在不同 joint 组，bootstrap 主导份额 46%/38%）⇒ 该列不可作
点比较，v2 报告须沿用此判定。

---

## 1. 范围

- **方法（3）**：ERM、SWAD、HyperAdapt。SWAD 由 ERM 的 `--swad` 同 run 派生。
- **方向（2）**：MIMIC→CheXpert（M→C）、CheXpert→MIMIC（C→M）。
- **不含 HyperHead / HyperFusion** ⇒ **v2 不检验「条件化深度」轴**，任何深度结论不得由本实验推出。
- **不含 ROC**（操作点后处理，非 AUC 可比方法）。

### 选择准则

**主比较只用 S1 = training-domain validation**（source val 的折均 marginal-worst，即现行
`scripts/select_config_cv.py`），这是本设定下**唯一可部署**的准则。

> **不把 oracle 当作并列准则的理由**：DomainBed（Gulrajani & Lopez-Paz, ICLR 2021）明确指出
> test-domain（oracle）selection **不是有效的 benchmarking 方案**——它使超参更多的算法占便宜，
> 只应作为 disclaimer 披露。故 oracle 在 v2 中仅用于 D1 的「选择损失」诊断，**不参与方法排名**。

---

## 2. 因子与设计

| 因子 | 水平 |
|---|---|
| 方法 | ERM / SWAD / HyperAdapt |
| 方向 | M→C / C→M |
| 划分 | 5 fold（患者级 GroupKFold，沿用现有 `cv5/`） |
| **trial** | **3（trial 0/1/2）**，`seed = 42 + fold + 10 × trial` |

trial 0（seed 42–46）= 现有产物，全部复用。trial 1 = 52–56，trial 2 = 62–66。
⇒ 每个（方法 × 方向）条件有 **15 个 replicate**。

---

## 3. 估计量与指标

- **主估计量 = full-target**：每个 source `(fold, trial)` 模型评**完整** target，逐 `(f,t)` 算指标
  再等权平均。**从不跨折池化**（`docs/oof_regime_results.md` §2 已证 pooling 制造校准漂移伪影）。
- **k→k averaging 仅用于 D1 的 config 选择**，不进入任何报告数字。
- **主指标**：marginal worst-group AUC。
- **次要指标**：Overall AUC；canonical worst（**必须附 argmin 稳定性检验**）；canonical gap。
- **校准指标**：calibration-in-the-large、calibration slope、ECE、Brier；整体 + 逐 marginal 子群。

---

## 4. 预注册假设

| 编号 | 假设 | 判定方式 |
|---|---|---|
| **H1** | S1 下 HyperAdapt 的 marginal-worst AUC 优于 ERM，在 **两个方向** 复现 | 配对 cluster bootstrap 95% CI 下界 > 0 |
| **H2**（主） | HyperAdapt−ERM 的效应量**超过训练噪声的方差分量** | fold × trial 双因子随机效应分解；效应量 > √(训练方差分量) |
| **H3** | HyperAdapt 的 OOD 校准不劣于 ERM，**且**判别优势（若有）在**重校准后仍存在** | 校准指标 CI 不劣；重校准后重算主指标，CI 下界仍 > 0 |
| **H4** | HyperAdapt 位于 ID–OOD 拟合线**之上**（真 effective robustness），而非仅落在线上更高处 | probit 尺度线性拟合的残差 CI > 0 |

### 判定规则（事前写定）

- **H2 或 H3 任一不成立** ⇒ 现行结论「HyperAdapt 是唯一一致显著为正的 HN」**不可维持**，须按
  准则、训练噪声与指标口径**条件化重述**。
- H1 不成立（即 trial 扩充后不复现）⇒ 直接判定现有 OOD 结论为单-trial 偶然。
- H4 不成立不推翻 H1–H3，但须在报告中声明「OOD 增益不独立于 ID 增益」。

---

## 5. 统计口径

- **配对 cluster bootstrap**：患者级，1000 次，**全方法共用同一批重采样索引**（保证配对）。
  复用 `scripts/build_oof_results_averaging.py` 的 `clusters()`。
- **方差分解**：`(fold, trial)` 双因子随机效应，分别报告划分方差分量与训练方差分量。
- **多重比较**：Holm family = {SWAD, HyperAdapt} × {M→C, C→M} = **4 个检验**，仅在**主指标**上校正。
  诊断量（D1–D3）不进入该 family，且**不作显著性声明**。
- 一律报 **点估计 + 95% CI + 效应量**；不单独呈现 p 值。

---

## 6. 三项诊断（不参与方法排名，不做显著性声明）

- **D1 · 选择损失（oracle disclaimer）**：target 上按折均 marginal-worst 找 oracle config，报
  `oracle − S1`。**标注为不可部署上界**。仅 trial 0，仅 k→k。
- **D2 · 校准**：§3 的四个校准指标 + 校准曲线，整体与逐 marginal 子群。**同时报原样迁移与重校准后**
  （isotonic，在评估集内 5 折交叉拟合，不需额外 split）。
- **D3 · 偏移刻画 + effective robustness**：ΔP(Y)、逐子群 ΔP(Y|A)、ΔP(A)（prior shift）；
  source/target 域判别器 AUC（proxy A-distance，Ben-David et al. 2010）；ID vs OOD 散点 + probit
  拟合（Taori et al. 2020；Miller et al. 2021）。

---

## 7. 执行顺序与依赖

| 阶段 | 内容 | 依赖 | GPU-h |
|---|---|---|---|
| 0 | D2 + D3（用现有 trial 0 预测） | 无 | 0 |
| 1 | trial 1/2 训练（S1 选中 config） | 无 | ≈74 |
| 2 | D1：补全 6 config 网格的 trial 0 checkpoint | 无 | ≈185 |
| 3 | full-target 终评 + H1–H4 判定 | 阶段 1 | ≈37 |

阶段 0 的结果（尤其校准）可能直接改变对现有数字的解读，故先行。

---

## 8. 偏离记录

> 执行中任何对本预注册的偏离**追加**记录于此，不得改写上文。

### 偏离 #1 · H4（effective robustness）在阶段 0 数据下**不可评估**（2026-07-28）

**内容**：H4 原定用 probit 尺度的 ID–OOD 线性拟合残差判定。阶段 0 用现有 trial 0 预测执行后
（`scripts/ood_effective_robustness.py`，30 点/方向）发现**拟合线不成立**：

| 方向 | ID 质量跨度 | probit 拟合 R² | 拟合斜率 |
|---|---|---|---|
| M→C | **0.0106** | **0.153** | −0.764 |
| C→M | **0.0163** | **0.135** | −0.291 |

Taori et al. (2020) / Miller et al. (2021) 的方法前提是模型在 ID 上有**足够大的质量跨度**（其原始
研究覆盖 accuracy ~0.2–0.9 的上百个模型）以定出趋势线。此处全部 30 点挤在 ID ∈ [0.840, 0.851]
的区间内，散布几乎全部来自 fold 间评估噪声，拟合出的**负斜率与低 R² 无实质含义**；据此计算的
残差若用于判定，会得到「看似显著、实则虚假」的结论（实测 M→C 的 HyperAdapt 残差
+0.0094 [+0.0011, +0.0178] 恰会被误判为「线之上」）。

**处置**：脚本内置双重守卫（`MIN_ID_SPAN = 0.05`、`MIN_R2 = 0.25`），不满足即**拒绝判定**，
`above_line` 置 `None`，图与 stdout 均标注「H4 不可评估」。残差数值仍落盘，仅供记录，**不得作结论**。

**H4 的重新安排**：推迟至**阶段 2 之后**。阶段 2 补齐 6 config × 5 fold 的全网格 checkpoint 后，
OOD 点数增至 ~90/方向且 config 跨 lr 3e-05→3e-04，ID 质量跨度将显著扩大，届时重跑并按原判据评定。
若跨度仍不达 `MIN_ID_SPAN`，则 H4 记为「本设定下不可评估」，不作任何 effective robustness 声明。

**对其他假设的影响**：无。H1–H3 不依赖该拟合。

---

### 偏离 #2 · H3 第二条判据（"重校准后优势仍存在"）在 AUC 口径下是同义反复（2026-07-28）

**内容**：H3 原定第二条判据为「判别优势（若有）在重校准后仍存在——重校准后重算主指标，CI 下界仍 > 0」。
执行时发现该判据**在 AUC 口径下不可能证伪**：isotonic（及 Platt）是**单调**映射，而 AUC 只依赖分数
排序，故 AUC 对任何单调重校准**在理论上完全免疫**。

实测证实：重校准后三方法的 marginal-worst AUC **同向**小幅下降，Δ(HyperAdapt − ERM) 几乎不变——

| 方向 | ERM | SWAD | HyperAdapt | Δ(HA−ERM) 原样 → 重校准 |
|---|---|---|---|---|
| M→C | 0.8114 → 0.8072 | 0.8069 → 0.8037 | 0.8159 → 0.8128 | +0.0045 → +0.0055 |
| C→M | 0.7866 → 0.7852 | 0.7854 → 0.7841 | 0.7883 → 0.7869 | +0.0017 → +0.0018 |

三方法同向的微小下降来自**交叉拟合分折 + isotonic 产生并列值（ties）**导致的排序信息损失，
而非校准效应。

**处置（修正判据）**：H3 第二条改为在**校准敏感**的指标上检验，即 **Brier score**（已在 D2 中计算，
含配对 cluster bootstrap CI）。判据改为：「若 HyperAdapt 在 Brier 上显著劣于 ERM，则其 AUC 优势
不能被解读为整体预测质量的优势」。

**执行结果（同时作为 H3 的判定）**：

| 方向 | Δ&#124;CITL&#124; | ΔECE | ΔBrier | 判决 |
|---|---|---|---|---|
| **M→C** | **+0.0068** [+0.0061,+0.0075] | **+0.0068** [+0.0061,+0.0075] | **+0.0040** [+0.0036,+0.0045] | **校准显著更差** |
| C→M | −0.0063 [−0.0068,−0.0058] | −0.0063 [−0.0068,−0.0058] | −0.0012 [−0.0016,−0.0008] | 校准显著更好 |

（Δ = HyperAdapt − ERM；&#124;CITL&#124;/ECE/Brier 的 Δ<0 = 校准更好。SWAD 在**两个方向**校准均显著优于 ERM。）

⇒ **H3 第一条「HyperAdapt 校准不劣于 ERM」在 M→C 上不成立** ⇒ 按 §4 判定规则，现行结论
「HyperAdapt 是唯一一致显著为正的 HN」**须条件化重述**：其 M→C 的 AUC 优势（本已由超参对照证明
主要来自 wd 差异）**伴随显著更差的校准**，在患病率相差 3.7 倍的跨库部署下不构成实用优势。

---

### 偏离 #3 · 范围扩充：加入 **GroupDRO** 第四臂（2026-08-13）

**内容**：§1 冻结的方法集为 ERM / SWAD / HyperAdapt 三臂。现按导师要求补入的第 4 基线
**GroupDRO**（Sagawa et al., ICLR 2020，比较协议 §1「训练时使用子群标签」那一格）扩充为四臂，
两个方向均做。

**理由（为何这不是 forking path）**：GroupDRO 的加入不是在看到 OOD 结果后挑方法，而是补一个
**协议早已规定、ID 侧已在五个数据集完成**（`docs/groupdro_baseline_5datasets.md`）的对照格。
其作用是排除一条替代解释——「HN 在 OOD 上的表现是因为它是唯一见过子群标签的方法」——
没有它，OOD 侧的 HN vs 基线比较与 ID 侧不同构。

**未污染声明（写在任何数字之前）**：本条追加时，GroupDRO 的 OOD 预测**一张都尚未计算**
（仅有 ID 侧 cv5 trial 0 的 checkpoint）。故下述 H5 与 §4 的 H1–H4 同为**事前**假设。

**新增假设**：

| 编号 | 假设 | 判定方式 |
|---|---|---|
| **H5** | S1 下 GroupDRO 的 marginal-worst AUC 优于 ERM，在**两个方向**复现，**且**效应量超过训练噪声 | 与 H1/H2 完全同口径：配对 cluster bootstrap 95% CI 下界 > 0 **且** \|d̄\| > σ_train |

H5 的两条判据分别对应 H1（样本不确定性）与 H2（训练不确定性），**不新设更宽松的判据**。
校准（D2）同口径纳入 GroupDRO，但**不为其新增 H3 型判定**，只作描述性报告。

**统计口径的连带修改（§5 多重比较）**：Holm family 由
`{SWAD, HyperAdapt} × {M→C, C→M}` = **4** 扩为 `{SWAD, HyperAdapt, GroupDRO} × {M→C, C→M}` = **6**，
仅在主指标（marginal worst-group AUC）上校正。**既有 4 项检验须按 6-family 重新校正**；
若有原「Holm 后保持显著」的项在新 family 下不再显著，须在结果文件中如实更新——这是扩充 family
的必然代价，事前接受。

**不适用的部分**：§6 之外的**属性 knockout（结果文件 §5.1b）对 GroupDRO 不适用**——该反事实
干预的是「模型 forward 时看到什么属性」，而 GroupDRO 的子群标签只进损失、推理时架构为纯
image-only（与 ERM 逐位同构），**无干预面**。这一点本身也是结论：GroupDRO 与 HN 用子群标签的
方式不同（训练期重加权 vs 推理期条件化），故二者对同一份属性信息的利用路径不可互换。

**执行**：阶段 1 补 trial 1/2 训练（MIMIC 作业 75757 / CheXpert 75758，各 5 fold × 2 trial，
config 为各自 S1 选中 `lr3e-04_wd1e-04` / `lr1e-04_wd1e-03`，batch 与 trial 0 逐项一致）；
阶段 3 双向 full-target 终评（15 replicate/方向）。trial 0 复用作业 75529/75530 既有 checkpoint。
