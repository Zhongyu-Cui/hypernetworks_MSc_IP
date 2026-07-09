# 实验 F·Fitzpatrick17k：冻结预训练 backbone regime（skin）

> **一句话结论**：在 **HyperAdapt 论文的本尊数据集 Fitzpatrick17k** 上、用**论文自己的冻结 backbone
> 设定**复刻实验 F。powered CV-OOF（池化 16012）下，冻结 HyperAdapt 的 worst-group AUC 对冻结 ERM
> **+0.021 但不显著（p=0.334）**，AUC gap 反而**变宽**（0.053→0.083）；唯一稳健、极显著的增益仍是
> **Overall AUC（+0.045, p=1e-24，容量效应）**。方向上比 HAM（负）更接近论文（dark-skin / Eopp 名义改善），
> 但**统计上未确立公平赢**——论文的 Fitzpatrick 公平声明在我们的 powered 二分类-AUC 口径下**未获稳健复现**。

---

## 1. 动机

HAM 实验 F（[[见 frozen_regime_ham_experiment.md]]）已证冻结未翻盘。但 HAM **不是** HyperAdapt 论文用的
数据集。Fitzpatrick17k **是**论文三数据集之一，且**肤色 skin 正是论文的公平轴**，论文报告 HyperAdapt
在此**dark-skin F1 +4.9%、Eopp1 −33%**。更关键：**论文的 backbone 本就是冻结的**（Fig 2「frozen
base-model weights」），故**我们的冻结 arm = 论文 Ours 的直接对应，冻结 ERM = 论文 Vanilla**。这让本实验
成为对论文公平声明**在其自有数据集 + 自有 regime**下的最贴近检验。

口径差异（须知）：论文任务是**多分类诊断 + F1/recall/Eopp**；我们是**二分类 malignant + AUC**（+ 附操作点
F1/recall/Eopp 搭桥）。同数据集、同肤色轴，是可达到的最贴近对照。

---

## 2. 设计与执行（与 HAM 实验 F 完全一致）

- 唯一变量 `freeze_backbone`（冻 conv/BN 仿射 θ、BN stats 自适应、只训 adapter+fc）；独立重搜超参。
- **基建**：给 fitz ERM/HyperAdapt(skin) 脚本加 `--freeze_backbone`+`--cv/--fold`；新建 cv5
  StratifiedKFold(skin×label) split（5 折 disjoint 覆盖全 16012，**肤色 VI 单-split 55 → 池化 635，~11× 评估功效**）。
- **执行**：F-search 70564/70565 → Pareto → F-confirm 70576/70577 → F-CV-OOF 70586/70587。
- **Pareto 选定**：erm_frozen=idx5 `lr3e-04_wd1e-03`、hyperadapt_frozen=idx2 `lr1e-04_wd1e-04`。
- 全微调参照（单-split 5-seed，无 cv5）：erm / hyperadapt 均 Pareto idx3 `lr1e-04_wd1e-03`。
- 分析：`scripts/analyze_frozen_regime_fitz.py`（单-split）、`scripts/analyze_frozen_regime_fitz_cvoof.py`（CV-OOF）。

---

## 3. 结果 A：单-split 5-seed（对齐 N=1602）

| Arm | Overall | worst-group | worst std(H1) | wTPR@.2 | Eopp |
|-----|---------|-------------|--------------|---------|------|
| ERM 全微调 | 0.895±.013 | 0.785±.049 | 0.040 | 0.653 | 0.251 |
| HyperAdapt 全微调 | 0.902±.010 | 0.817±.096 | 0.077 | 0.768 | 0.156 |
| ERM 冻结 | 0.798±.001 | 0.665±.033 | 0.026 | 0.511 | 0.289 |
| HyperAdapt 冻结 | 0.852±.010 | 0.717±.071 | 0.057 | 0.602 | 0.227 |

配对(seed42)：冻结 HA−ERM worst +0.031 CI[−0.054,+0.149] ✗跨0；Overall +0.045 **p=3e-4 ✓**。
> 单-split worst CI ±.05–.10（肤色 VI test 仅 55），噪声地板，下节 powered 定音。

---

## 4. 结果 B：CV-OOF 池化终判（N=16012，肤色 VI 评估 n≈635）

| Arm | Overall | worst | gap | wTPR@.2 | Eopp | 逐肤色 AUC I→VI |
|-----|---------|-------|-----|---------|------|-----------------|
| **ERM 冻结**（论文 Vanilla） | 0.8163 | 0.7729 | 0.053 | 0.475 | 0.273 | .826/.817/.819/.789/.819/**.773** |
| **HyperAdapt 冻结**（论文 Ours） | 0.8616 | 0.7941 | 0.083 | 0.574 | 0.214 | .863/.854/.862/.877/.870/**.794** |

配对显著性（冻结 HA − 冻结 ERM，池化 OOF，3000×bootstrap / DeLong）：

| 指标 | Δ | 判决 |
|------|---|------|
| worst-group AUC | **+0.0211** [−0.0335, +0.0855] p=0.334 | **✗ 不显著** |
| Overall AUC (DeLong) | **+0.0453** p=1.4e-24 | ✓ 极显著 |
| AUC gap | 0.053 → **0.083（变宽）** | HN 抬升所有肤色，浅肤色更多 |
| wTPR@.2 / Eopp（点估计） | 0.475→0.574 / 0.273→0.214 | 名义改善（论文方向），但未做配对显著性 |

---

## 5. H1–H4 判决

- **H2（worst-group AUC）——否（但方向为正）**：冻结 HA 对冻结 ERM **+0.021 不显著**（p=0.334）。
  与 HAM（−0.041）不同，此处**符号为正**且逐肤色**每一型都改善**（含最深 VI 0.773→0.794），
  但**未达统计显著**，且 **gap 变宽**（HN 抬浅肤色更多）——不构成稳健公平赢。
- **H3（操作点）——名义正、未显著检验**：冻结 HA 的 wTPR（0.475→0.574）与 Eopp（0.273→0.214）
  名义改善，方向与论文一致；但仅点估计，worst-group AUC（更严的排序公平）已判 n.s.。
- **H4（Overall）——稳健正（容量）**：冻结 HA 对冻结 ERM **+0.045 p=1e-24**，极显著。冻结 ERM
  linear-probe 欠拟合（0.816），HN 的 adapter 容量把**所有**肤色一起抬起——这是**准确率/容量**效应，非公平。
- **H1（方差）**：powered 池化后对所有臂消解，次要。

---

## 6. 结论与和论文的对照

1. **论文 Fitzpatrick 公平声明未获稳健复现**：论文报 HA dark-skin F1 +4.9%、Eopp1 −33%。我们在**同数据集、
   同肤色轴、同冻结 regime**下，dark-skin（VI）AUC +0.021、Eopp −22%（相对）——**方向一致**，但
   **worst-group AUC 增益不显著（p=0.334）**，且 AUC gap 变宽。论文的效应在点估计层面部分再现，
   **但过不了 powered CV-OOF + 配对检验**。
2. **与 HAM 的差异有信息量**：Fitz 冻结 HA worst-group 符号为正（HAM 为负）。可能因肤色比 age 更可从图像
   解码 / 冻结 ERM linear-probe 更弱（0.816），留给 HN 容量的抬升空间更大——但抬升是**全肤色齐涨**（gap 变宽），
   本质仍是 Overall/容量而非缩小组间差。
3. **两数据集一致的硬结论**：**冻结 regime 下 HN 无稳健 worst-group 公平赢**（Fitz +0.021 n.s.、HAM −0.041 n.s.），
   唯一稳健增益是 Overall AUC。这与 `I(Y;A|X)≈0`（skin 作业 67913）、R1、路线 A、HAM 实验 F 全线一致：
   **HN 价值在 Overall（充分性），不在公平**。冻结预训练（论文卖点）不改变此结论。

---

## 7. 与项目主线的关系

实验 F 现覆盖 **HAM(age) + Fitzpatrick(skin)** 两数据集、含论文本尊数据集与本尊 regime。结论稳固：
HyperAdapt 论文的「公平×整体双赢」在 powered 评估下**不复现于我们检验的任一真实医学影像设定**；
论文表面的公平赢，来自(a)多分类低精度 headroom + 阈值类指标、(b)未做 powered 显著性（3-run 无 CI）、
(c)Overall/容量效应被读作公平。见 [[frozen-regime-ham-experiment]] 与 docs/frozen_regime_ham_experiment.md。
