# 实验 F：冻结预训练 backbone regime（HAM10000·age）

> **一句话结论**：把 HyperAdapt 从「全微调预训练 backbone」改成「冻结 backbone、只训 adapter+fc」
> （对齐 Xu et al. 2026 论文设定），在 HAM-age 上**没有翻盘路线 A**。powered CV-OOF 口径下，
> HN 的 worst-group AUC 在两种 regime 下都 ≤ ERM（不显著）；冻结**同时压低** Overall（容量上限）
> 与 worst-group/操作点公平。论文的「公平×整体双赢」在 HAM-age 上、无论冻结与否都**未复现**。

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

对照臂（全微调，路线 A 已有）：ERM idx2 `lr1e-04_wd1e-04`、HyperAdapt idx4 `lr3e-04_wd1e-04`。
分析脚本 `scripts/analyze_frozen_regime_ham.py`（单-split）、`scripts/analyze_frozen_regime_ham_cvoof.py`
（CV-OOF 池化，复现路线 A 数字逐位吻合：ERM 0.8742/0.7796、HA 0.8982/0.7468）。

---

## 3. 结果 A：单-split 5-seed 确认（对齐 969，selection=overall）

| Arm | Overall AUC | worst-group AUC | worst std(H1) | wTPR@FPR.2 | Eopp |
|-----|------------|-----------------|--------------|-----------|------|
| ERM 全微调 | 0.883±.007 | 0.699±.153 | **0.123** | 0.468 | 0.520 |
| HyperAdapt 全微调 | 0.901±.012 | 0.758±.036 | 0.029 | 0.540 | 0.478 |
| ERM 冻结 | 0.848±.008 | 0.728±.007 | **0.005** | 0.358 | 0.595 |
| HyperAdapt 冻结 | 0.887±.012 | 0.752±.091 | 0.073 | 0.473 | 0.539 |

配对（seed42）：冻结 HA−ERM worst Δ+0.085 CI[−0.056,+0.184] ✗跨0；Overall DeLong +0.044 **p=0.001 ✓**。

> ⚠️ 单-split worst-group 落 969 小格（joint cell），CI ±.07–.15，是路线 A 明确点名的**评估-n 噪声地板**。
> 此处冻结 HA 的「worst 0.752 > ERM 0.728」在下一节 powered 口径下**被反转**——典型小格假阳性。

---

## 4. 结果 B：CV-OOF 池化终判（N=9707，evaluation-n 地板已消除）

| Arm | Overall | worst-grp | gap | wTPR@.2 | Eopp |
|-----|---------|-----------|-----|---------|------|
| ERM 全微调 | 0.8742 | **0.7796** | 0.108 | 0.594 | 0.357 |
| HyperAdapt 全微调 | 0.8982 | 0.7468 | 0.178 | 0.377 | 0.526 |
| ERM 冻结 | 0.8455 | 0.7394 | 0.129 | 0.410 | 0.426 |
| HyperAdapt 冻结 | 0.8651 | **0.6984** | 0.170 | **0.213** | **0.682** |

配对显著性（池化 OOF，3000×bootstrap / DeLong）：

| 对比 | worst-group Δ | Overall Δ |
|------|--------------|-----------|
| 冻结 HA − 冻结 ERM | **−0.041** [−0.104,+0.017] p=0.162 ✗（HN 更差，n.s.） | +0.020 p=6.5e-6 ✓ |
| 全微调 HA − 全微调 ERM | **−0.033** [−0.087,+0.057] p=0.643 ✗（HN 更差，n.s.） | +0.024 p=5.7e-8 ✓ |
| HA 冻结 − 全微调（同法） | — | **−0.033 p=3e-16**（冻结显著更低） |
| ERM 冻结 − 全微调（同法） | — | **−0.029 p=8e-9**（冻结显著更低） |

---

## 5. H1–H4 判决

- **H1（方差）——分裂，不支持论文**：冻结把 *ERM linear-probe* 的单-split worst 方差从 0.123 砸到
  0.005（24×），但对 *HyperAdapt* 反而升高（0.029→0.073，冒坏 seed）。低方差只惠及线性探针，
  **未传导到 HN**。且 powered 池化后方差问题对所有臂消失，此假设变得次要。
- **H2（worst-group AUC）——否（powered 口径下更强的否）**：冻结 HA 对冻结 ERM worst **−0.041**
  （n.s.），且冻结 HA 的 worst（0.698）是**四臂最低**。冻结不但没兑现公平赢，反而是最差。
- **H3（操作点）——否，且负**：冻结 HA 的 wTPR@FPR.2=0.213、Eopp=0.682 **全场最差**。
  冻结没在阈值层面解锁公平，反而压低操作点。
- **H4（Overall）——部分**：冻结内 HN 显著高于冻结 ERM（+0.020，linear-probe 欠拟合下 HN 补容量），
  但冻结整体被容量上限显著压在全微调之下（−0.033，p=3e-16）。HN 唯一稳健的赢仍只是 **Overall AUC**。

---

## 6. 结论与方法学教训

1. **冻结 regime 未翻盘路线 A，反而强化之**：HN 的 worst-group 在两种 regime 下都 ≤ ERM（不显著），
   HyperAdapt 论文的「公平×整体双赢」在 HAM-age 上**无论冻结与否都未复现**。深层遍布式介入
   （HyperAdapt）Overall 最高、公平最差的模式在冻结下依旧。
2. **HN 的价值仍只在 Overall AUC**（`I(Y;age|X)>0` 的充分性兑现），**不在 worst-group 公平**——
   与 R1 合成剂量-反应、路线 A CV-OOF 的既有结论一致。
3. **方法学教训（重要）**：单-split 曾显示冻结 HA「worst 0.752 > ERM 0.728」，powered CV-OOF 下
   **反转为 −0.041**。这是小格评估-n 噪声地板制造的假阳性的又一实例——**再次证明 powered 评估
   （池化 OOF）不可省**，单-split worst-group 结论不可信。
4. **为何与论文不冲突**：论文赢在(a)多分类低精度大 headroom + 阈值类指标（F1/recall/Eopp），
   (b)最大胜绩来自 PAD-UFES-20 的 26 维**信息性临床属性**（≈HAM-age 机制推广）。HAM-age 是二分类
   AUC 近天花板 + 纯人口学属性 + powered 评估，正是论文优势不成立的场景。

---

## 7. 与项目主线的关系

实验 F 是对「冻结能否救活 HN 公平收益」这一 HyperAdapt 论文诱因的**有功效否定**，补齐了
「全微调 HN 无公平赢」是否只是 regime artifact 的对照缺口。结论：**不是 regime 问题，是信号/评估问题**。
HN 在真实医学影像上公平收益的稀缺性，不因冻结预训练而改变。
