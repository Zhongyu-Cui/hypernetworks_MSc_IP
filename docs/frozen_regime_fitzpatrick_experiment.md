# 实验 F·Fitzpatrick17k：冻结预训练 backbone regime（skin）

> 🛑 **信号闸门口径更新（2026-07-19）**：闸门已由单-split 条件互信息改为 **OOF conditional V-information**
> （Xu 2020 / Hewitt 2021；权威 = `docs/conditional_v_information_gate.md`）。关键变化：
> **HAM-age 由「唯一正信号」退回「未检出」**（+0.0028 bits，CI 触 0）；Fitzpatrick ≈0；
> **CheXpert / MIMIC 全轴 CI<0 = 决定性 0**；唯一 detected 为 **PAPILA-age**（+0.041 bits，但 n=420）。
> control task 五库全部通过 ⇒ 零读数非假阴。本文「HAM-age `I(Y;age|X)>0`」一族表述**已过时**（下方已内联订正）；
> 各处**结论方向不变**，但「有信号」前提须按新闸门重读。


> **⚠️ 2026-07-19 口径修订**：本文档 CV-OOF 数字（结果 B、H2–H4）原用 **pooled**（跨折池化 raw logit +
> DeLong），已整体改为 **averaging**（逐折算→折间平均 + 样本级 bootstrap）。**零重训**。改口径原因见
> [[pooled-oof-calibration-drift-artifact]]。分析脚本 `scripts/analyze_frozen_regime_fitz_cvoof_averaging.py`。
>
> **一句话结论（averaging 口径，修订后）**：在 **HyperAdapt 论文的本尊数据集 Fitzpatrick17k** 上、用**论文
> 自己的冻结 backbone 设定**复刻实验 F。powered CV-OOF averaging（N=16012）下，冻结 HyperAdapt 的 worst-group
> AUC 对冻结 ERM **+0.046，p=0.078（临界，仍不显著）**，AUC gap 仍**微变宽**（0.057→0.064）；唯一稳健、
> 极显著的增益仍是 **Overall AUC（+0.048，bootstrap CI 排除 0，容量效应）**。方向上比 HAM 更强（每一肤色型
> 都升、含最深 VI 0.770→0.816），但 **worst-group AUC 未过显著门槛**——论文的 Fitzpatrick 公平声明在我们的
> powered 二分类-AUC 口径下**未获稳健（显著）复现**。（相对旧 pooled 版：worst-group Δ +0.021→**+0.046**、
> p 0.334→**0.078**，效应更大、更接近显著但未越线；总判决不变。）

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
- 分析：`scripts/analyze_frozen_regime_fitz.py`（单-split）、`scripts/analyze_frozen_regime_fitz_cvoof_averaging.py`
  （**CV-OOF averaging，现口径**；旧 `analyze_frozen_regime_fitz_cvoof.py` 为 pooled，已弃用）。

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

## 4. 结果 B：CV-OOF averaging 终判（N=16012，肤色 VI 折均评估 n≈635/折均 ~127）

| Arm | Overall | worst | gap | wTPR@.2 | Eopp | 逐肤色 AUC I→VI |
|-----|---------|-------|-----|---------|------|-----------------|
| **ERM 冻结**（论文 Vanilla） | 0.8171 | 0.7700 | 0.057 | 0.444 | 0.305 | .827/.818/.819/.792/.818/**.770** |
| **HyperAdapt 冻结**（论文 Ours） | 0.8647 | 0.8156 | 0.064 | 0.534 | 0.317 | .870/.858/.870/.879/.875/**.816** |

配对显著性（冻结 HA − 冻结 ERM，**averaging + 样本级 bootstrap，B=3000**；Overall 亦用 bootstrap CI，弃用池化 DeLong）：

| 指标 | Δ | 判决 |
|------|---|------|
| worst-group AUC | **+0.0456** [−0.0058, +0.1064] p=0.078 | **✗ 不显著（临界）** |
| Overall AUC | **+0.0476** [+0.0391, +0.0561] p<0.001 | ✓ 极显著 |
| AUC gap | 0.057 → **0.064（微变宽）** | HN 抬升所有肤色，浅肤色略多 |
| wTPR@.2 / Eopp（点估计） | 0.444→0.534 / 0.305→**0.317** | wTPR 名义改善、**Eopp 反微升（更差）**——操作点信号混杂，未做配对显著性 |

> **对比旧 pooled 版**：worst-group Δ **+0.021→+0.046**、p **0.334→0.078**（HN 跨折漂移被池化压低，averaging 修复后
> 效应更大、逼近但未越显著线）；逐肤色 VI **.794→.816**；gap 变宽幅度缩小（.083→.064）。Overall 稳健不变。
> **注意 Eopp 由旧版「改善 0.273→0.214」转为 averaging「微升 0.305→0.317」**——操作点指标受阈值/小格影响、
> 池化阈值漂移曾夸大其「改善」，averaging 下 HN 的操作点公平信号并不稳健。

---

## 5. H1–H4 判决

- **H2（worst-group AUC）——否（但方向为正、临界）**：冻结 HA 对冻结 ERM **+0.046，p=0.078 不显著**
  （averaging；旧 pooled +0.021 p=0.334）。逐肤色**每一型都改善**（含最深 VI 0.770→0.816），符号为正且
  比 HAM 更强，但**仍未过显著门槛**，且 **gap 微变宽**（HN 抬浅肤色略多）——不构成稳健公平赢。
- **H3（操作点）——混杂、未解锁公平**：averaging 下冻结 HA 的 wTPR 名义改善（0.444→0.534）但 **Eopp 反微升
  （0.305→0.317，更差）**——与旧 pooled 版「wTPR/Eopp 双改善」不同（池化阈值漂移曾夸大 Eopp 改善）。操作点
  信号在 averaging 下**方向不一致**，未做配对显著性；与 worst-group AUC 的 n.s. 一致：冻结未在公平轴解锁显著优势。
- **H4（Overall）——稳健正（容量）**：冻结 HA 对冻结 ERM **+0.048，bootstrap CI 排除 0**，极显著。冻结 ERM
  linear-probe 欠拟合（0.817），HN 的 adapter 容量把**所有**肤色一起抬起——这是**准确率/容量**效应，非公平。
- **H1（方差）**：powered 口径后对所有臂消解，次要。

---

## 6. 结论与和论文的对照

1. **论文 Fitzpatrick 公平声明未获稳健复现**：论文报 HA dark-skin F1 +4.9%、Eopp1 −33%。我们在**同数据集、
   同肤色轴、同冻结 regime**下，dark-skin（VI）AUC **+0.046**——**方向一致**，但 **worst-group AUC 增益不显著
   （p=0.078，临界未越线）**，AUC gap 微变宽，且 Eopp 点估计不降反微升。论文的效应在点估计层面部分再现，
   **但过不了 powered CV-OOF averaging + 配对检验**。
2. **与 HAM 的差异（averaging 口径下缩小）**：averaging 下 Fitz（+0.046 n.s.）与 HAM（+0.048 marg n.s.）
   worst-group 符号**双双为正、量级相近**——旧 pooled 版「Fitz 正 / HAM 负」的不一致**是池化伪影**，已消。
   两数据集现一致为「正但不显著」。Fitz 抬升是**全肤色齐涨**（gap 变宽），本质仍是 Overall/容量而非缩小组间差。
3. **两数据集一致的硬结论**：**冻结 regime 下 HN 无稳健（显著）worst-group 公平赢**（Fitz +0.046 p=0.078、
   HAM +0.048 marg n.s.，均方向为正但 n.s.），唯一稳健增益是 Overall AUC。这与 `I(Y;A|X)≈0`（skin 作业 67913）、
   R1、[[crossdataset-worstgroup-fairness]]（averaging）、HAM 实验 F 全线一致：**HN 价值在 Overall（充分性），
   不在（显著）公平**。冻结预训练（论文卖点）不改变此结论。

---

## 7. 与项目主线的关系

实验 F 现覆盖 **HAM(age) + Fitzpatrick(skin)** 两数据集、含论文本尊数据集与本尊 regime。结论稳固：
HyperAdapt 论文的「公平×整体双赢」在 powered 评估下**不复现于我们检验的任一真实医学影像设定**；
论文表面的公平赢，来自(a)多分类低精度 headroom + 阈值类指标、(b)未做 powered 显著性（3-run 无 CI）、
(c)Overall/容量效应被读作公平。见 [[frozen-regime-ham-experiment]] 与 docs/frozen_regime_ham_experiment.md。
