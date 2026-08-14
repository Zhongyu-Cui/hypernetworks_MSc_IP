# 实验 F：冻结预训练 backbone regime（HAM10000·age）

> 🛑 **信号闸门口径更新（2026-07-19）**：闸门已由单-split 条件互信息改为 **OOF conditional V-information**
> （Xu 2020 / Hewitt 2021；权威 = `docs/conditional_v_information_gate.md`）。关键变化：
> **HAM-age 由「唯一正信号」退回「未检出」**（+0.0028 bits，CI 触 0）；Fitzpatrick ≈0；
> **CheXpert / MIMIC 全轴 CI<0 = 决定性 0**；唯一 detected 为 **PAPILA-age**（+0.041 bits，但 n=420）。
> control task 五库全部通过 ⇒ 零读数非假阴。本文「HAM-age `I(Y;age|X)>0`」一族表述**已过时**（下方已内联订正）；
> 各处**结论方向不变**，但「有信号」前提须按新闸门重读。


> **⚠️ 2026-07-19 口径修订**：本文档 CV-OOF 数字（结果 B、H1–H4）原用 **pooled**（跨折池化 raw logit
> + DeLong），已整体改为 **averaging**（逐折算→折间平均 + lesion cluster bootstrap）。**零重训**，同一批
> 逐折预测仅改聚合。旧 pooled 版见 git 历史。改口径原因（池化对排序指标的系统性负偏、HN 受害重于 ERM）见
> [[pooled-oof-calibration-drift-artifact]]。分析脚本 `scripts/analyze_frozen_regime_ham_cvoof_averaging.py`。
>
> **一句话结论（averaging 口径，修订后）**：把 HyperAdapt 从「全微调预训练 backbone」改成「冻结
> backbone、只训 adapter+fc」（对齐 Xu et al. 2026 论文设定），在 HAM-age 上**没有翻盘路线 A**。powered
> CV-OOF averaging 下，冻结 HN 的 worst-group AUC 对冻结 ERM **方向为正但不显著**（canon +0.052 / marg
> +0.048，n.s.）；冻结**显著压低 Overall**（容量上限）却**未在 worst-group / 操作点上兑现显著公平赢**。
> 论文的「公平×整体双赢」在 HAM-age 上、无论冻结与否都**未获稳健（显著）复现**——HN 唯一稳健的赢仍只是
> **Overall AUC**。
>
> **相对旧 pooled 版的关键更正**：旧版「冻结 HA worst 0.698 四臂最低、HA−ERM worst −0.041（HN 更差）」
> **系池化伪影**（HN 跨折 logit 漂移最重、被池化负偏压得最狠）。averaging 下冻结 HA canonical worst 实为
> **0.782（四臂最高）**、HA−ERM worst **+0.052（n.s.）**——符号翻正。**总判决（无显著公平赢、价值仅在
> Overall）不变，但「HN 有害/最差」的旧叙述已删。**

---

## 1. 动机与设计

HyperAdapt 论文（arXiv:2601.13094）报告 HyperAdapt 在 Fitzpatrick/ODIR/PAD-UFES-20 上
**公平与整体同升**，且 backbone 是**冻结**的（只训 adapter ϕ）。本项目此前的 HAM-age HN 实验用的是
**全微调预训练 backbone**。实验 F 把「冻结 vs 全微调」作为**唯一变量**做因果归因，检验四个假设：

- **H1（方差）**：冻结是否降低 seed 间 worst-group 方差（论文低方差叙事）。
- **H2（worst-group AUC）**：冻结 HN 是否稳健跑赢冻结 ERM（不同于路线 A 的打平/微负）。
- **H3（操作点）**：AUC 若打平，冻结是否在 F1/recall/Eopp 层面兑现公平（论文吃的是阈值类指标）。
- **H4（Overall 不掉）**：冻结 HN 是否维持 Overall。

### 控制变量

- **唯一变动**：`freeze_backbone`（冻结 conv/BN 仿射 θ，只训 adapter+fc）。其余（属性=age、
  split、CV-OOF 折、评估、batch=128 等-batch）全部对齐路线 A。
- **BN 沿用自适应**：只冻结仿射/卷积权重 θ，BN running stats 仍随 HAM 分布更新（既有 `_freeze_backbone`
  约定），避免 ImageNet BN 统计硬套皮肤图拖垮 AUC，把变量干净隔离为「卷积/线性权重能否被梯度更新」。
- **独立重搜超参**（严谨性要求）：不复用全微调 Pareto config，对 `*_frozen` 走完整协议 §3
  （6-config 搜索 → A1 Minimax-Pareto → 5-seed 确认 → CV-OOF）。

### 冻结后可训练参数

| 方法 | 可训练 | 冻结 |
|------|--------|------|
| ERM 冻结（linear probe） | 513（仅 fc） | 11.18M |
| HyperAdapt 冻结 | 2.76M（adapter+fc） | 11.18M |

---

## 2. 训练与选择（作业号）

| 阶段 | 作业 | 内容 |
|------|------|------|
| F-search | 70530（erm_frozen）/ 70531（hyperadapt_frozen） | 6-config × seed42，单-split |
| F-Pareto | — | Minimax worst-group：**erm_frozen=idx4 `lr3e-04_wd1e-04`**，**hyperadapt_frozen=idx5 `lr3e-04_wd1e-03`**（均落最高 lr 端，印证 linear-probe/adapter 需更大 lr） |
| F-confirm | 70542 / 70543 | 选定 config × 5 seed（42–46），单-split |
| F-CV-OOF | 70552 / 70553 | 选定 config × 5 折 GroupKFold，池化 9707 OOF |

对照臂（全微调，路线 A 已有 config）：ERM idx2 `lr1e-04_wd1e-04`、HyperAdapt idx4 `lr3e-04_wd1e-04`
（**是路线 A 对照 config，与 crossdataset/`oof_results_averaging.json` 的 `selected_configs` 不同**，故本文
全微调数字与那两处不逐位对照）。分析脚本 `scripts/analyze_frozen_regime_ham.py`（单-split）、
`scripts/analyze_frozen_regime_ham_cvoof_averaging.py`（**CV-OOF averaging，现口径**；旧
`analyze_frozen_regime_ham_cvoof.py` 为 pooled，已弃用）。

---

## 3. 结果 A：单-split 5-seed 确认（对齐 969，selection=overall）

| Arm | Overall AUC | worst-group AUC | worst std(H1) | wTPR@FPR.2 | Eopp |
|-----|------------|-----------------|--------------|-----------|------|
| ERM 全微调 | 0.883±.007 | 0.699±.153 | **0.123** | 0.468 | 0.520 |
| HyperAdapt 全微调 | 0.901±.012 | 0.758±.036 | 0.029 | 0.540 | 0.478 |
| ERM 冻结 | 0.848±.008 | 0.728±.007 | **0.005** | 0.358 | 0.595 |
| HyperAdapt 冻结 | 0.887±.012 | 0.752±.091 | 0.073 | 0.473 | 0.539 |

配对（seed42）：冻结 HA−ERM worst Δ+0.085 CI[−0.056,+0.184] ✗跨0；Overall DeLong +0.044 **p=0.001 ✓**。

> ⚠️ 单-split worst-group 落 969 小格（joint cell），CI ±.07–.15，评估-n 噪声大。此处冻结 HA
> 「worst 0.752 > ERM 0.728」符号为**正**——**与下一节 powered averaging 口径一致**（+0.052 n.s.）。
> （历史注：旧 pooled 版本曾称此处被 powered「反转为 −0.041」并当作小格假阳性；现已查明**那个反转本身是
> 池化伪影**——单-split 与 averaging 同为正号，是 pooled 聚合制造了假的符号翻转。见 §6 方法学教训。）

---

## 4. 结果 B：CV-OOF averaging 终判（N=9707，评估-n 地板已消除）

worst-group 主口径 = **canonical**（含 sex×age joint，对齐原脚本 `worst_and_gap`）；附 **marginal**（单属性、
更稳，joint 小格折均噪声大）。gap 为 canonical gap。

| Arm | Overall | canon worst | marg worst | gap | wTPR@.2 | Eopp |
|-----|---------|-------------|------------|-----|---------|------|
| ERM 全微调 | 0.8887 | 0.7603 | 0.8136 | 0.151 | 0.603 | 0.399 |
| HyperAdapt 全微调 | 0.9087 | 0.7779 | 0.8214 | 0.155 | 0.505 | 0.600 |
| ERM 冻结 | 0.8508 | 0.7297 | 0.7446 | 0.138 | 0.491 | 0.483 |
| HyperAdapt 冻结 | 0.8752 | **0.7819** | 0.7925 | 0.127 | 0.411 | 0.665 |

> **对比旧 pooled 版**：冻结 HA canon worst **0.6984→0.7819（+0.083）**（HN 跨折 logit 漂移最重、受池化负偏
> 最狠，averaging 修复后由「四臂最低」变「四臂最高」）；ERM 冻结 −0.010（几乎不变）。这就是把 HA−ERM worst
> 从假性负值抬回正值的机制来源。

配对显著性（**averaging + lesion cluster bootstrap，B=3000**；Overall 亦用 cluster bootstrap CI，弃用池化 DeLong）：

| 对比 | canonical worst Δ | marginal worst Δ | Overall Δ |
|------|-------------------|------------------|-----------|
| 冻结 HA − 冻结 ERM | **+0.052** [−0.044,+0.109] n.s. | **+0.048** [−0.025,+0.083] n.s. | +0.024 [+0.015,+0.034] ✓ |
| 全微调 HA − 全微调 ERM | **+0.018** [−0.077,+0.169] n.s. | **+0.008** [−0.045,+0.051] n.s. | +0.020 [+0.012,+0.028] ✓ |
| HA 冻结 − 全微调（Overall） | — | — | **−0.033** [−0.042,−0.025] ✓（冻结显著更低） |
| ERM 冻结 − 全微调（Overall） | — | — | **−0.038** [−0.048,−0.027] ✓（冻结显著更低） |

> **读法**：两种 regime 下 HN 对同 regime ERM 的 worst-group 均**方向为正但 n.s.**（旧 pooled 版此处为负、
> 系伪影）；Overall 均**显著为正**（HN 补容量）；冻结整体被容量上限**显著压低** Overall（−0.033/−0.038）。
> 结论骨架（无显著公平赢 + Overall 稳健 + 冻结压低容量）与旧版一致，仅「HN worst-group 更差」的符号被更正。

---

## 5. H1–H4 判决

- **H1（方差）——分裂，不支持论文**（源自结果 A 单-split，未受口径影响）：冻结把 *ERM linear-probe*
  的单-split worst 方差从 0.123 砸到 0.005（24×），但对 *HyperAdapt* 反而升高（0.029→0.073，冒坏 seed）。
  低方差只惠及线性探针，**未传导到 HN**；powered 口径下此假设次要。
- **H2（worst-group AUC）——否（无显著公平赢），但方向为正**（averaging 口径更正）：冻结 HA 对冻结 ERM
  worst **canon +0.052 / marg +0.048，均 n.s.**（CI 跨 0）；与全微调（canon +0.018 / marg +0.008，n.s.）**同向**。
  冻结 HA 的 canon worst（0.782）实为**四臂最高**（旧 pooled 版「−0.041、四臂最低」系池化伪影，已更正）。
  即：**HN 微抬 worst-group 但达不到显著**——不构成稳健公平赢，但也**非「HN 有害」**。
- **H3（操作点）——否（未解锁显著公平），点估计仍偏弱**：冻结 HA 的 wTPR@FPR.2=0.411、Eopp=0.665
  在阈值层面**仍差于冻结 ERM**（0.491 / 0.483）——与 worst-group AUC 的正号不一致（操作点更受阈值/小格影响），
  但两者共识是**冻结未在公平轴解锁显著优势**。（旧 pooled 版 wTPR=0.213、Eopp=0.682 的极端值受池化阈值漂移
  夸大，averaging 下已缓和但方向未变；操作点未做配对显著性。）
- **H4（Overall）——部分**：冻结内 HN **显著**高于冻结 ERM（+0.024，linear-probe 欠拟合下 HN 补容量），
  但冻结整体被容量上限**显著**压在全微调之下（−0.033/−0.038）。HN 唯一稳健的赢仍只是 **Overall AUC**。

---

## 6. 结论与方法学教训

1. **冻结 regime 未翻盘路线 A**：HN 的 worst-group 在两种 regime 下对 ERM 都**方向为正但不显著**
   （非旧版所称「≤ ERM/更差」——那是池化伪影），HyperAdapt 论文的「公平×整体双赢」在 HAM-age 上
   **无论冻结与否都未获显著复现**（公平轴 n.s.）。深层遍布式介入（HyperAdapt）Overall 最高的模式在冻结下依旧。
2. **HN 的价值仍只在 Overall AUC**（⚠️ **归因订正**：原写作「`I(Y;age|X)>0` 的充分性兑现」；新 OOF V-info 闸门下
   HAM-age **未检出（≈0）**，故该 Overall 增益应归于 **adapter 容量效应**而非属性信号充分性——与本文 §5 H4
   「linear-probe 欠拟合下 HN 补容量」一致），**worst-group 公平未达显著**——
   与 R1 合成剂量-反应、[[crossdataset-worstgroup-fairness]]（averaging 口径）的既有结论一致。
3. **方法学教训（重要，已修订）**：单-split 显示冻结 HA「worst 0.752 > ERM 0.728」（正号），旧 pooled
   powered CV-OOF **曾把它翻成 −0.041** 并当作「小格假阳性被 powered 纠正」——但 averaging powered 给出
   **+0.052（仍正号）**。**真相：那次符号反转本身是池化聚合的伪影**（跨折 logit 尺度漂移，HN 受害最重），
   不是 powering 纠了单-split 的错。单-split 与 averaging **同为正号**、一致。教训因此更新为：**跨折/跨模型
   聚合口径（averaging vs pooling）比是否 powered 更能左右 worst-group 排序结论**——powered 仍重要（收窄 CI），
   但必须配 averaging，池化 raw logit 会制造假符号。见 [[pooled-oof-calibration-drift-artifact]]。
4. **为何与论文不冲突**：论文赢在(a)多分类低精度大 headroom + 阈值类指标（F1/recall/Eopp），
   (b)最大胜绩来自 PAD-UFES-20 的 26 维**信息性临床属性**（≈HAM-age 机制推广）。HAM-age 是二分类
   AUC 近天花板 + 纯人口学属性 + powered 评估，正是论文优势不成立的场景。

---

## 7. 与项目主线的关系

实验 F 是对「冻结能否救活 HN 公平收益」这一 HyperAdapt 论文诱因的**有功效否定**，补齐了
「全微调 HN 无公平赢」是否只是 regime artifact 的对照缺口。结论：**不是 regime 问题，是信号/评估问题**。
HN 在真实医学影像上公平收益的稀缺性，不因冻结预训练而改变。
