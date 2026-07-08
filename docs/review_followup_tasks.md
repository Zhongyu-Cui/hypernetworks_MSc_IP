# 评审补强任务清单（对核心结论的必要补充）

> 建立日期：2026-07-06。对应对 `docs/comparison_protocol.md` 已完成的 A–D 阶段的**评审意见落地**。
> 核心问题：现有证据只对论点的「必要」方向提供弱支持，对「充分」方向（`I(Y;A|X)>0 ⇒ HN 增益`）
> 几乎无支持；且 null 判定缺 MDE/CI、公平只用 AUC 衡量。本清单只列**为支撑核心结论必须补的内容**，
> 不含 nice-to-have。
>
> 状态标记：`[ ]` 未开始 · `[~]` 进行中 · `[x]` 完成。
> 粒度约定：每个复选框 = 一件可独立完成并验证的任务；括注为「完成判据」。
> 优先级：**R1 > R2 > R3 > R4 > R5 > R0**（R1 提供现有设计最缺的充分性因果证据，ROI 最高）。

---

## R1. 合成信号剂量-反应实验（充分性因果证据）—— 最高优先

> 动机：全研究仅 HAM 一个数据集 `I(Y;A|X)>0`，信号强弱与「数据集身份」完全共线，无法把
> 「iff」从跨数据集相关升级为因果。在**单一数据集内注入可控信号**、看 HN 增益是否随之上升，
> 是补上因果链且成本远低于 C 阶段的关键实验。首选 MIMIC（n 大、MDE 小、baseline 稳、`I(Y;A|X)≈0`
> 是干净的负控制起点）。
>
> **学界方法对标（关键设计修正）**：主流可控-信号基准 —— Colored-MNIST（IRM, Arjovsky et al. 2019）、
> Waterbirds（GroupDRO, Sagawa et al. 2020）、Spawrious（2023，fine-grained control）—— 控制的是
> `I(Y;A)` 且**属性画进图像**，故其设定下 `I(Y;A|X)≈0`，与本论点所需相反、**不能直接照搬**。本实验需
> 属性携带 X 之外的残余标签信息（`I(Y;A|X)>0`），故 A 必须作为**旁路侧信息（side channel），不进图像**。
> 拼装三块成熟范式：① 剂量-反应**实验结构**借 spurious-correlation 谱系（Spawrious 的 fine control 为模板，
> 综述 *The Clever Hans Mirage* 2024）；② **注入机制** = 标签噪声（随机翻转率）+ 特权信息（LUPI,
> Vapnik & Vashist 2009；TRAM, *Transfer and Marginalize* 2022，证特权信息对标签噪声鲁棒、预测零开销）；
> ③ **MI 校准**借可控-MI 合成基准思路（MINE, Belghazi et al. 2018；CMI Neural Estimator 2019/2020）。

- [x] R1.1 设计合成属性注入机制 —— 构造侧信道属性 `A_syn = Y ⊕ Bernoulli(η)`，以**标量形式**作为 HN
      条件输入（**不画进图像**，区别于 CMNIST/Waterbirds），使 `I(Y;A_syn|X)` 由翻转率 η 单调可调：
      η→0.5 时 A_syn⊥Y（信号 0）、η→0 时 A_syn≈Y（信号≈`H(Y|X)`）。对标：标签噪声翻转率 + LUPI 特权信息
      设定（TRAM 2022 / Vapnik 2009）。（判据：注入代码 + 单元测试证明 η=0.5 时 A_syn⊥Y|X、η 减小时条件依赖单调增）
      **✅ 完成**：`src/datasets/synthetic_attribute.py`（`inject_synthetic_attribute` + 确定性 `make_split_rng`）；
      `python -m src.datasets.synthetic_attribute` self-test 通过——分层 plug-in 证 η=0.5 时 I(Y;A_syn|X)=0.0000、
      η 减小时严格单调增、η=0 时 =H(Y|X)=0.826 bits（残余熵上限），确定性与输入校验均验证。
- [x] R1.2 用现有 conditional-MI 估计器校准注入强度 —— 把 η 标定到目标 MI 档位
      `I(Y;A_syn|X) ∈ {0, 0.02, 0.05, 0.10} nats`（0 档=负控制）。方法对标：MINE / CMI Neural Estimator
      用已知真值合成分布验证估计器功效的思路，反用于「η ↔ 实测 I」标定。
      （判据：每档实测 ΔNLL 落在目标 ±MDE 内，形成「翻转率 η ↔ 实测 I」标定表）
      **✅ 完成**：`scripts/calibrate_synthetic_signal.py`（复用 D2 冻结特征 ΔNLL 估计器 + R1.1 确定性 A_syn），
      作业 **70114**（gpus24），扫 13 档 η 单调、跨 backbone seed 42/43/44。标定表（int 口径，`outputs/mimic_cxr/synthetic_signal_calibration.json`）：
      **η=0.5→0（负控制）/ η=0.3621→0.02036±0.00034 / η=0.2942→0.04889±0.00019 / η=0.2096→0.09915±0.00054 nats**，
      每档复核实测均在目标 ±0.001 内（远紧于 MIMIC MDE≈0.001–0.002）。
- [x] R1.3 各强度档统一 regime 训练 ERM + 3 HN —— 复用 A0 harness（Pareto 6 配置 → 选定 → ≥5 seed），
      A_syn 作为 HN 的条件属性。（判据：4 档 × 4 方法预测全部落盘、0 错误）**✅ 65 run 全部落盘、0 错误**。
      **基础设施 ✅**：`src/datasets/mimic_synth_dataset.py`（确定性 A_syn，经验翻转率对齐 η）、
      `src/utils/synthetic_fairness.py` + subgroup_auc 注册 `mimic_synth`(dim=2, self-test ✅)、
      `unpack_asyn`/`forward_asyn`、`src/training/train_mimic_synth.py`(4 方法参数化) + `slurm/r1_synth.sh`。
      **训练已提交（2026-07-06，用户裁定：复用 C2 Pareto 配置 + 5 seed，只训 ERM+3HN、不派生 SWAD/ROC）**：
      ERM 跨档不变（忽略 A_syn，用独立 RNG）故只训 1 次（作业 **70120**，idx5 η=0.5）；3 HN × 4 档 × 5 seed：
      HyperHead(idx2,gpus24) **70115**(η0.5)/**70121**(η0.3621)/**70122**(η0.2942)/**70123**(η0.2096)；
      HyperFusion(idx2,gpus24) **70124/70125/70126/70127**；HyperAdapt(idx4,gpus48,bs128 等-batch) **70128/70129/70130/70131**。
      共 65 run，全部避开 semois，运行中。
- [x] R1.4 绘制剂量-反应曲线并下结论 —— (HN − ERM) 的 Overall/worst-group 增益 vs 实测 `I(Y;A_syn|X)`；
      检验增益是否随信号单调上升、0 档是否打平。实验结构对标 spurious-correlation 谱系的「指标 vs 强度」
      扫描（Spawrious fine control）。（判据：曲线图 + 单调性检验写入报告，明确回答「信号存在是否**充分**导致增益」）
      **✅ 完成**：`scripts/analyze_synthetic_dose_response.py` → `outputs/mimic_synth/{dose_response_summary.json,dose_response_curve.png}`；
      报告 **`docs/r1_synthetic_dose_response_results.md`**。**结论**：三 HN 的 **Overall AUC 增益随实测 nats 严格单调上升
      （Spearman ρ=+1.000, p=0；0 档 CI 含 0 干净打平；0.02/0.05/0.10 档 ΔOverall≈+0.016/+0.037/+0.066 CI 排除 0）**
      = **充分性方向获单数据集内因果证据**；worst-group(A_syn) 平/微负（边际先验型信号只改池化判别、不改组内排序，机制自洽）。
      介入深度非决定因素（三 HN 曲线重合）。

> **参考文献（R1）**：
> - Arjovsky et al., *Invariant Risk Minimization*, 2019（Colored-MNIST 可控 spurious 谱系源头）
> - Sagawa et al., *Distributionally Robust Neural Networks* (GroupDRO), 2020（Waterbirds）
> - Lynch et al., *Spawrious: A Benchmark for Fine Control of Spurious Correlation Biases*, 2023
> - *The Clever Hans Mirage: A Survey on Spurious Correlations in ML*, 2024（范式地图）
> - Vapnik & Vashist, *Learning Using Privileged Information* (LUPI), 2009
> - Collier et al., *Transfer and Marginalize (TRAM): Explaining Away Label Noise with Privileged Information*, 2022
> - Belghazi et al., *Mutual Information Neural Estimation* (MINE), 2018
> - *Conditional Mutual Information Neural Estimator*, 2019（arXiv:1911.02277）
> - ⚠️ 引用年份/作者以正式写作时复核 arXiv 原文为准。

---

## R2. 操作点公平指标（让 ROC 有意义 + 公平不只挂 AUC）

> 动机：worst-group AUC / AUC gap 均阈值无关，导致 reject-option（ROC，阈值上的公平干预）
> 在结构上失效（θ*=0 时 ROC≡ERM，θ*>0 时反被 AUC 惩罚），其真实目的从未被测量；且「公平」
> 课题只报 AUC 是过窄切面。本组**零重训**，全部在已落盘预测 npz 上重算。

- [x] R2.1 实现操作点公平度量 —— Equalized Odds gap + 固定 FPR（如 0.2）下的 worst-group TPR
      + 各子群 TPR/FPR 差。（判据：复用/移植 `drafts/fairness_metrics.py`，与 sklearn 交叉验证）
      **✅ 完成**：`src/utils/operating_point_fairness.py`（`subgroup_tpr_fpr`/`equalized_odds_gap`/
      `worst_group_tpr`/`threshold_at_overall_fpr`/`operating_point_report`，两操作点=原生 0.5 + 固定
      FPR=0.2）；子群 schema 复用 `subgroup_auc.subgroup_masks`（新抽出的单一事实来源，AUC 与操作点同批
      子群）；self-test 用 sklearn `confusion_matrix`/`recall_score`/`roc_curve` 逐一交叉验证通过。
      **分数尺度**：logits(ERM/HN)→sigmoid、ROC=已调整概率；原生操作点恒 prob=0.5 对两尺度语义一致。
- [x] R2.2 在全部数据集已落盘预测上重算 —— 5 数据集 × 6 方法 × 5 seed（PAPILA 走 OOF 池化）。
      （判据：新指标表逐格生成、口径与 D3/D5 一致）
      **✅ 完成**：`scripts/build_operating_point_tables.py`（复用 `build_results_tables` 的 DATASETS/cfg/
      SEEDS/`_load`，同 Pareto 配置 + HAM 过滤 age≥0 + PAPILA OOF 池化）→ 报告 D8-a/D8-b 表；ERM 单格
      经手工 sklearn 复核逐位一致。
- [x] R2.3 用操作点指标重评 ROC 并补入报告 —— 使 ROC 在其设计目标（子群奇偶）上可比，
      更新 D3/D5/D6 相关表与结论。（判据：报告新增操作点公平表，ROC 不再等价 ERM 的假象消除）
      **✅ 完成**：`roc.apply_roc_operating_point[_auto]`——θ 选择目标从阈值无关 worst-group AUC 换成
      **val EqOdds gap**（deprived 按 val 0.5-TPR 二分，网格含 0 → 不劣 ERM），纯后处理零重训。ROC 在
      3/5 数据集选出 θ*>0 并在操作点上**大幅收窄 EqOdds gap**（MIMIC 0.280→0.087、CheXpert 0.200→0.078、
      HAM 0.489→0.342），**假象消除**。报告新增 **D8 操作点公平**节（D8-a/b/c + 5 条结论），含
      「ROC 收益锚定 0.5、不迁移到 FPR=0.2」「HN 仍随 `I(Y;A|X)` 兑现/打平」等测量。**注**：ROC 是阈值
      后处理、非属性架构，其收益与核心论点正交，故 D3/D5/D6 的 AUC 主表不改动，操作点结果并列于 D8。

---

## R3. conditional-MI null 判定的可信度（必要性方向补强）

> 动机：D2 表 MIMIC/CheXpert/Fitzpatrick 仅给光秃点估计（且为负，说明估计器在该尺度被方差主导）；
> PAPILA 自认「不可判定」却被算进 null 证据。「没检测到」≠「确认为零」，须逐数据集给 CI+MDE。

- [x] R3.1 D2 每行补 bootstrap CI —— 对 5 数据集全部敏感轴的 `I(Y;A|X)` 估计给 95% CI。
      （判据：D2 表每格 `点估计 ± CI`，无裸点估计）
      **✅ 完成**：`scripts/summarize_conditional_mi_ci.py`（Rubin's rules 合并逐 seed test-bootstrap CI
      的层内+层间不确定性，PAPILA 用单run CI）→ `outputs/conditional_mi_ci_summary.json`；D2 表 16 行全部
      改为「点估计 [95% CI]」+ 三态判定（确认>0 / 噪声地板 / 不可判定）。**结论修订**：唯 **HAM-age
      [+0.00064,+0.02694] 与 sex×age joint [+0.00009,+0.02395] 的 CI 下界 > 0**（确认正信号）；MIMIC
      race/age/joint CI 全 < 0（噪声地板，真值≈0）；**CheXpert 全轴 / Fitzpatrick / PAPILA / MIMIC-sex /
      HAM-sex 的 CI 跨 0 = 不可判定**（非「确认为零」，比旧表诚实）。
- [x] R3.2 逐数据集报告 MDE + 正控制探针 E3 功效数值 —— 明确每数据集探针「能检出多大注入信号」。
      （判据：每数据集一行「MDE=? / E3 能检出注入 I=? 」，据此界定「≈0」的可信区间）
      **✅ 完成**：扩展 `summarize_conditional_mi_ci.py`（从 E3 dose-response 提 MDE=首个 CI 下界>0 档的
      实测 ΔNLL + 正控制功效 + 非单调探针标记）→ 报告新增 **D2-MDE 节**（每数据集一行 MDE 括区 + 正控制
      + null 可信区间）。**关键发现**：MDE 随 √(test n) 收缩——MIMIC 0.001 / CheXpert 0.006 / Fitz 0.015 /
      **HAM 0.05（单-seed）**；⚠️ **HAM-age 实测 +0.0138 低于其单-seed MDE**，D2 的「确认>0」靠 3-seed Rubin
      池化才排 0，余量薄 → 支撑 R4；**PAPILA 探针非单调不可靠（n=420）→ 灰区，退出 null（R3.3）**。
- [x] R3.3 修订 PAPILA 定位 —— 明确标为「灰区、不可判定」并**退出 null 证据集**；
      订正协议/报告中「PAPILA null 判决一致」的双重计账表述。（判据：报告不再用 PAPILA 支持理论）
      **✅ 完成**：results_summary 的 PAPILA D3 表头改「灰区/不可判定·不计入 null 证据」；D7 结论 2「null
      判决一致」由 4 库改 **3 库（MIMIC/CheXpert/Fitz），PAPILA 明剔除**；D7 结论 1、D8 操作点结论 3、
      一句话总结、comparison_protocol 的 C5.8 与 D2 行同步订正——PAPILA 性能打平仅作趋势旁证、不支撑理论。
      grep 复核：全部 PAPILA×null 同现行均已显式标注「灰区/退出/不计入」。
- [x] R3.4 age 分箱敏感性 —— 至少 2 种年龄分箱下重估 CheXpert / MIMIC 的 `I(Y;A|X)`，
      检验 null 是否为分箱伪影。（判据：分箱敏感性小表，结论稳健性说明）
      **✅ 完成**：`scripts/agebin_sensitivity_cmi.py`（同一冻结特征缓存 + 从 split CSV 按行序取回原始 age
      重新分箱，逐行核对 age/label 对齐；GPU 缓存分析 <3min）跑 **5 种分箱**（二值 50/60/70 + tertile + quartile）
      × 3 seed Rubin 合并 → `outputs/agebin_sensitivity_cmi.json` + 报告 **D2-agebin 节**。`bin2@60` 逐位复现
      R3.1。**结论**：两库全部 5 分箱下 `I(Y;age|X)` 一律负、无一「确认>0」（MIMIC 多落负地板、CheXpert 全
      跨 0），**null 对分箱稳健、非伪影**；且越细分箱点估计越负（纯过拟合罚），无隐藏 age 信号。

---

## R4. HAM 充分性方向的功效补强

> 动机：唯一正信号数据集在 9×3 矩阵无一 p<0.05；功效不足既来自 seed 少（n=5），也来自评估端
> 子群样本极小（joint 格 ~26）。须同时从「更多 seed」和「诚实的子群样本量诊断」两侧补强。

- [x] R4.1 扩 seed 重跑 HAM 确认 —— n=5 → ≥15 seed（或对跨 seed 池化 test 做样本级 bootstrap），
      重算 D6 显著性。（判据：D6 矩阵在更高功效下重出，明确增益是否达显著）
      **✅ 完成（走「或」路径：跨 seed 池化样本级 bootstrap，非重训 15 seed）**。裁定依据：D6 已是样本×种子
      两层 bootstrap（每次抽 1/5 seed 注入全单-seed 方差），且 worst-group 瓶颈在评估端小样本、扩 seed 救不了。
      实现 `ham_significance_highpower()`（seed-ensemble=概率空间跨 5 seed 平均→模型方差降√5→样本级配对
      bootstrap/DeLong），报告新增 **D6-HP-a/b/c** + 更高功效判决。**关键结果**：**Overall AUC 升级为显著**
      ——3 HN 全部显著赢 ERM/ROC（p=0.0001~0.0051 vs ERM）、HyperFusion/HyperAdapt 亦显著赢 SWAD；
      **worst-group CI 收窄~2× 后仍 n.s.**（canonical 全跨 0、边缘最好 HyperAdapt p=0.099）→ 判定瓶颈是
      评估端 joint~26 小样本（R4.2），非 seed 数。D7 结论 1 同步修订。
- [x] R4.2 评估端子群样本量诊断 —— 报告每 worst-group 格的 n 与 AUC 估计方差，界定 canonical
      joint 口径的可信下限。（判据：子群 n × 方差表，明确哪些格进入噪声地板）
      **✅ 完成**：`scripts/subgroup_n_diagnosis.py`（HAM 14 格逐格 n/n_pos/n_neg/AUC + 三口径 AUC-SE
      Hanley–McNeil/DeLong/bootstrap 互证，噪声地板阈 SE≥0.10 或 min(pos,neg)<10）→
      `outputs/ham10000/subgroup_n_diagnosis.json` + 报告 **D6-diag 表**。**结论**：**3/14 格入噪声地板、
      全是 joint 小格**（n_pos=3/8/4，含 Female|80+ AUC=1.0 退化），joint 格 SE 0.05–0.15 是边缘格 2–5×；
      worst-group=取 min 恒被这几格拖走（解释 ERM ±0.15）。**canonical-joint 可信下限被压破 → HAM 公平应以
      边缘 worst-group(6 组)为准**（D3 脚注/D6-b/HP-b 口径的量化依据）。此问题 HAM 独有（余库最小格 ≥55）。
- [x] R4.3 多重比较与事后扩展说明 —— 对 D6 的比较做多重性校正（BH-FDR）；**订正**「HF-only → 全矩阵」的
      定性。（判据：报告加多重性校正列 + 非-forking-path 说明）
      **✅ 完成**：`scripts/ham_multiplicity_fdr.py`（收集 D6-HP 全 21 比较 raw p → BH-FDR 按度量分族 + 全局
      单族两口径）→ `outputs/ham10000/d6_multiplicity_fdr.json` + 报告 **D6-FDR 表**。**结果**：Overall AUC
      全 3 HN vs ERM/ROC **BH-FDR 后仍稳健显著**（族内 8/9、全局 21-比较 7/9 存活，q≤0.034）；仅 vs SWAD 被
      削弱（HyperHead/HyperFusion 全局失守）；worst-group 全 21→0 存活（与 R4.2 一致）。
      **⚠️ 按用户订正（2026-07-08）**：「HF-only → 全矩阵」**非** forking-path/探索性扩展——全矩阵本是比较
      协议正确设计，协议 §5 缩到仅 HF 系**规格失误**、C 阶段已订正，Pareto 反转仅是暴露契机。报告 D6 表头 +
      D6-FDR 非-forking 声明 + comparison_protocol D6 行同步改写（**不再**保留「因 pilot HF 领先/因反转故扩展」
      的数据驱动旧措辞）；BH-FDR 仍照做（多重性与是否预设无关）。

---

## R5. 结论表述修订（写作侧）

- [x] R5.1 修订摘要与 D7 —— 去掉「闭环证实了该判据」，改为分「必要 / 充分」两向的表述。
      （判据：结论与 R1–R4 证据强度一致）
      **✅ 完成**：新增 **D7-0 证据分向综合** + 改写 D1 验证结构 + 一句话。**关键——placeholder 措辞已过时,按
      R1–R4 实际强度上调**：充分方向**非「未证」而是「获最强证据」**（R1 合成剂量-反应因果 ρ=+1.0 + HAM
      自然信号 BH-FDR 后显著）;必要方向 3 null 打平**弱-中支持**(MIMIC 干净负控/CheXpert·Fitz 不可判定,非
      「确认零」)。**关键限定**:充分性**仅限 Overall 整体判别**——worst-group 公平(本课题核心动机)因 R1
      worst-group 平 + HAM 评估-n 限制**未获证实**。去掉旧「5×6 闭环证实判据」的过度概括。
- [x] R5.2 SWAD 对称性说明 —— 显式声明 SWAD「搭 ERM 车、未独立搜参」的比较不对称；
      如结论重度依赖 SWAD 优势则补其独立超参搜索。（判据：报告加对称性说明或补搜参结果）
      **✅ 完成（走「加对称性说明」分支，无需补搜参）**。核实：协议 §省算力 + C1.6(69607 ERM+SWAD 同 run)
      证 SWAD 复用 ERM Pareto 配置、未独立搜参。D7 结论 4 加**对称性声明** + D6-FDR 结论 2 加注。**裁定不补
      搜参**：不对称两向解读——(i) SWAD 取胜处(CheXpert/Fitz/OOD)是**保守**的(自身未调优仍赢);(ii) HN 赢
      SWAD 处须谨慎(调优偏 HN)，但**主结论锚定 HN vs ERM(对称调优)+ R1 合成，不依赖 HN 赢 SWAD**(且 FDR 已
      示 vs SWAD 最弱)，故条件「重度依赖 SWAD 优势」不成立、无需补 6 配置搜索。

---

## R0. 口径统一与回归保护（低优先、随手补）

- [x] R0.1 统一 worst-group 口径 —— 断言全部表格走 canonical `subgroup_auc_vector`（含 joint），
      修复 C6 边缘-only vs D4 canonical 曾并存的痕迹。（判据：`build_results_tables.py` 加断言/回归测试）
      **✅ 完成**：`build_results_tables.py` 加 `_assert_canonical_vector`（在**唯一算子** `_metrics_one` 内，
      覆盖 D3/D5 ID + D4 OOD + PAPILA OOF 全路径）——断言子群向量维度==`SUBGROUP_VECTOR_DIM[key]` 且多属性库
      含 joint 键，退回边缘-only 立即 AssertionError。加 `--section selftest` 回归测试（5 库正向走 canonical +
      边缘-only/错维度被拦截，✓ 通过）。复核：`--section id/ood` 重生成全表 0 断言触发、数字与报告逐位一致。

---

**说明**：R1–R3 对应评审意见的「投入产出比最高的三件事」；R4/R5 是其配套的功效与表述补强；
R0 是防回归的卫生项。完成 R1（合成剂量-反应）+ R3（MDE/CI）后，论点的必要方向转「稳」、
充分方向获得现有设计唯一缺失的因果证据，届时方可将结论从「趋势一致」升级为「判据成立」。
