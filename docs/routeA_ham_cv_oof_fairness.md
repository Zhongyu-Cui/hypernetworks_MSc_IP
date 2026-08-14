# 路线 A：HAM 5 折 CV-OOF —— worst-group 公平的有功效判决

> 🛑 **信号闸门口径更新（2026-07-19）**：闸门已改为 **OOF conditional V-information**
> （权威 = `docs/conditional_v_information_gate.md`）。**HAM-age 由「唯一正信号」退回「未检出」**（≈0）；
> Fitzpatrick ≈0；**CheXpert / MIMIC 全轴 CI<0 = 决定性 0**；唯一 detected 为 **PAPILA-age**（n=420）。
> 本文「唯一 `I(Y;age|X)>0`」等标注**已过时**；**性能结论与聚合口径不受影响**，但「有信号/无信号」的分类须按新闸门重读。


> 建立日期：2026-07-09。对应「优化显著性检验能否用于公平（worst-group）」这一问的落地实验。
> 权威结论并入 `docs/results_summary.md` D7-0；本文件是路线 A 的完整设计 + 数据 + 判决记录。
> 数据由 `scripts/ham_cv_oof_significance.py` 从 `outputs/ham10000/cv5/predictions/` 统一重算。

---

## 1. 动机：为什么 seed-ensemble 救不了 worst-group，而 CV-OOF 能

评审 R4.1（D6-HP）用 **seed-ensemble**（概率空间跨 5 seed 平均，模型方差 ↓√5）把 HAM 的
**Overall AUC** 增益证到显著；但同一套机器对 **worst-group** 仍 n.s.（边缘最好 HyperAdapt p=0.099）。
R4.2 诊断出根因：worst-group 卡在**评估端子群小样本**——单-split test 仅 969（age 有效），
joint 小格 Female|80+ n_pos=4、Male|20-40 n_pos=3 进噪声地板（SE 0.10–0.15，含 AUC=1.0 退化）。

**关键区分**：seed-ensemble 压的是**模型方差**，worst-group 缺的是**评估样本**——两者正交。
你把模型平均得再稳，也无法往一个只有 3–4 个阳性的格里注入信息。故对 worst-group，显著性检验这根
杠杆到头了（换排列/精确检验只给更宽的诚实区间，不会变显著）。

**路线 A = 该问题的评估侧对应解**：把 HAM 从「单 split × 5 seed」改成「5 折 lesion 级 GroupKFold」，
**5 折 disjoint test 池化成全数据集 OOF**（每样本恰被其留出折预测一次，n≈9707），少数格评估 n
放大约 5–10 倍。这是 seed-ensemble（降模型方差）的公平侧对应（**增评估样本**），也是先前在 PAPILA
（C5，打穿 n≈84）已验证过的同一招。

---

## 2. 方法（算力中性、完全隔离）

- **划分**：`scripts/build_ham10000_splits.py --cv 5` 生成 `data/splits/ham10000/cv5/fold{0..4}/`。
  lesion 级 GroupKFold（同病灶整组只进一折的 train/val/test，无泄漏），每折 test≈20%，
  **5 折 test 并集 = 全 9948（硬断言通过）**，池化即全数据集 OOF。
- **算力中性**：**不重做 6-config 超参搜索**，复用 C1.5 已选定的 Pareto 配置（ERM idx2 / HyperHead
  idx3 / HyperFusion idx5 / HyperAdapt idx4）。每方法只跑 5 折 = 原 5-seed **同 run 数**——只是把
  「同 split 多 seed」重排成「5 折 disjoint test」。fold↔seed 一一映射（seed=42+fold）避免 run_id 覆盖。
- **隔离**：CV 产物写 `outputs/ham10000/cv5/`，**不碰已冻结的单-split D3/D6 预测**。
- **显著性口径**：CV-OOF **无 seed-ensemble**（每样本仅 1 个 OOF 预测），显著性来自「池化大 n +
  少数格放大」，做**样本级配对 bootstrap / DeLong**。全方法池化按同折序、同 age 过滤，**y_true 逐元素
  对齐已硬断言**（配对检验合法性前提）。
- **作业**：70451（ERM+SWAD）/ 70452（HyperHead）/ 70453（HyperFusion）/ 70454（HyperAdapt，gpus48
  b128 等-batch），各 array 0–4，0 错误；ROC 由 `derive_roc --output-dir .../cv5` 逐折后处理派生。

---

## 3. 结果

池化 OOF：**n=9,707，阳性率 0.148**，全方法样本对齐已断言。

### D-A1 性能与公平（池化全 9,707，单点估计）
| 方法 | Overall AUC | canonical worst | 边缘 worst | AUC gap |
|------|-------------|-----------------|-----------|---------|
| ERM | 0.8742 | 0.7796 | 0.7967 | 0.1078 |
| SWAD | **0.9018** | **0.8239** | **0.8316** | 0.0983 |
| ROC | 0.8696 | 0.7677 | 0.7900 | 0.1148 |
| HyperHead | 0.8892 | 0.7728 | 0.7830 | 0.1445 |
| HyperFusion | 0.8876 | 0.7704 | 0.7851 | 0.1469 |
| HyperAdapt | 0.8982 | 0.7468 | 0.7741 | 0.1784 |

- 全 3 HN 的 worst-group（两口径）**≤ ERM**；**gap 反而比 ERM 大**（0.14–0.18 vs 0.11）。
- **SWAD（attribute-blind 权重平均）在 Overall 与 worst-group 双轴上都最优**。
- 越深介入越伤公平：**HyperAdapt**（介入最遍布）Overall 最高 0.898、但 worst 最低 0.747、gap 最大 0.178。

### D-A2 逐子群样本量诊断（ERM 代表分数）——地板已抬起
| 子群 | n | n_pos | n_neg | AUC | boot-SE | 地板 |
|------|---|-------|-------|-----|---------|------|
| sex:Male | 5289 | 907 | 4382 | 0.871 | 0.006 | |
| sex:Female | 4418 | 530 | 3888 | 0.874 | 0.009 | |
| age:20-40 | 1630 | 93 | 1537 | 0.797 | 0.021 | |
| age:40-60 | 4478 | 426 | 4052 | 0.881 | 0.010 | |
| age:60-80 | 2905 | 695 | 2210 | 0.859 | 0.008 | |
| age:80+ | 694 | 223 | 471 | 0.803 | 0.017 | |
| sex:Male\|age:20-40 | 706 | 32 | 674 | 0.792 | 0.037 | |
| sex:Male\|age:40-60 | 2241 | 245 | 1996 | 0.875 | 0.014 | |
| sex:Male\|age:60-80 | 1841 | 474 | 1367 | 0.861 | 0.010 | |
| sex:Male\|age:80+ | 501 | 156 | 345 | 0.780 | 0.021 | |
| sex:Female\|age:20-40 | 924 | 61 | 863 | 0.797 | 0.027 | |
| sex:Female\|age:40-60 | 2237 | 181 | 2056 | 0.887 | 0.014 | |
| sex:Female\|age:60-80 | 1064 | 221 | 843 | 0.852 | 0.015 | |
| sex:Female\|age:80+ | 193 | 67 | 126 | 0.856 | 0.029 | |

**对照单-split（R4.2/D6-diag）**：进噪声地板的格 **3/14 → 0/14**；最小格 Female|80+ 由
n=27/n_pos=4/AUC=1.0 退化 → **n=193/n_pos=67/SE=0.029**；全 14 格 SE<0.10 且 min(pos,neg)≥10。
**worst-group 估计现在可信**——「评估-n 地板」这条对冲被兑掉。

### D-A3a canonical worst-group 样本级配对 bootstrap（HN − 基线，n=9,707，2000 次）
| HN \ 基线 | vs ERM | vs SWAD | vs ROC |
|-----------|--------|---------|--------|
| HyperHead | −0.007 [−0.064,+0.046] n.s. | −0.051 [−0.112,−0.002] **显著（HN 更差）** | +0.005 [−0.060,+0.057] n.s. |
| HyperFusion | −0.009 [−0.045,+0.066] n.s. | −0.054 [−0.086,+0.007] n.s. | +0.003 [−0.032,+0.076] n.s. |
| HyperAdapt | −0.033 [−0.085,+0.053] n.s. | −0.077 [−0.144,+0.007] n.s. | −0.021 [−0.081,+0.066] n.s. |

### D-A3b 边缘 worst-group 样本级配对 bootstrap（HN − 基线，仅 sex/age 6 组，n=9,707）
| HN \ 基线 | vs ERM | vs SWAD | vs ROC |
|-----------|--------|---------|--------|
| HyperHead | −0.0138 [−0.0504,+0.0315] n.s. | −0.0486 [−0.0978,−0.0029] **显著（HN 更差）** | −0.0071 n.s. |
| HyperFusion | −0.0117 [−0.0454,+0.0444] n.s. | −0.0465 [−0.0762,−0.0135] **显著（HN 更差）** | −0.0050 n.s. |
| HyperAdapt | −0.0226 [−0.0555,+0.0291] n.s. | −0.0575 [−0.1081,−0.0062] **显著（HN 更差）** | −0.0159 n.s. |

### D-A4 Overall AUC DeLong（HN vs 基线，池化 OOF n=9,707）
| HN \ 基线 | vs ERM | vs SWAD | vs ROC |
|-----------|--------|---------|--------|
| HyperHead | Δ=+0.0150 (p=0.0005 **显著**) | Δ=−0.0126 (p=0.0004 **显著 HN 更差**) | Δ=+0.0196 (p<0.0001 **显著**) |
| HyperFusion | Δ=+0.0133 (p=0.0104 **显著**) | Δ=−0.0142 (p=0.0005 **显著 HN 更差**) | Δ=+0.0180 (p=0.0004 **显著**) |
| HyperAdapt | Δ=+0.0240 (p<0.0001 **显著**) | Δ=−0.0036 (p=0.351 n.s.) | Δ=+0.0286 (p<0.0001 **显著**) |

### D-A5 整体临床四联指标（sensitivity/specificity/PPV/NPV，正类=malignant，阈值相关，复用 `operating_point_fairness`）

阈值无关的 AUC 之外补报阈值相关的四联，两个操作点（对齐 R2/D8）：原生 prob=0.5，与**固定整体 FPR=0.2
（specificity≈0.8，跨方法可比——不均衡任务下 0.5 阈值几乎全判负、sensitivity 失真，不宜跨方法比）**。

**① 原生阈值 prob=0.5**（各方法在 logit=0 的隐含操作点不同，故非公平对照，仅存档）
| 方法 | Sensitivity | Specificity | PPV | NPV |
|------|-------------|-------------|-----|-----|
| ERM | 0.4475 | 0.9571 | 0.6443 | 0.9088 |
| SWAD | 0.5825 | 0.9376 | 0.6186 | 0.9282 |
| ROC | 0.4962 | 0.9360 | 0.5741 | 0.9145 |
| HyperHead | 0.5400 | 0.9369 | 0.5978 | 0.9214 |
| HyperFusion | 0.6945 | 0.8742 | 0.4897 | 0.9428 |
| HyperAdapt | 0.5484 | 0.9397 | 0.6123 | 0.9229 |

**② 固定整体 FPR=0.2（specificity≈0.80，公平对照）** —— 此表才是跨方法可比口径
| 方法 | Sensitivity | Specificity | PPV | NPV |
|------|-------------|-------------|-----|-----|
| ERM | 0.7947 | 0.8011 | 0.4098 | 0.9574 |
| SWAD | **0.8469** | 0.8007 | 0.4248 | 0.9678 |
| ROC | 0.7961 | 0.8005 | 0.4094 | 0.9576 |
| HyperHead | 0.8086 | 0.8007 | 0.4135 | 0.9601 |
| HyperFusion | 0.8100 | 0.8000 | 0.4131 | 0.9604 |
| HyperAdapt | 0.8316 | 0.8008 | 0.4205 | 0.9647 |

- 匹配 specificity 后，**Overall sensitivity 排序 = AUC 排序**（SWAD 0.847 > HyperAdapt 0.832 > HF≈HH 0.81 >
  ERM≈ROC 0.795）：HN 对 ERM 的 sensitivity 小幅↑（+0.01~0.04）就是 Overall AUC 增益的阈值化表达，无新信息。

### D-A6 worst-group 临床指标（边缘 6 组，min_class_n=10）—— 公平视图（点估计）

**固定 FPR=0.2 口径**（specificity≈0.8 公平对照）
| 方法 | worst Sens | Sens gap | worst Spec | Spec gap |
|------|-----------|----------|-----------|----------|
| ERM | 0.6129 | 0.2301 | 0.5435 | 0.3222 |
| SWAD | **0.6989** | 0.2024 | 0.5584 | 0.3140 |
| ROC | 0.6022 | 0.2409 | 0.5350 | 0.3319 |
| HyperHead | **0.4086** | 0.4883 | 0.4904 | 0.4055 |
| HyperFusion | 0.6129 | 0.2346 | 0.5159 | 0.3774 |
| HyperAdapt | **0.3978** | 0.4945 | 0.5648 | 0.3390 |

- **临床上比 AUC 更刺眼**：在最差子群的**恶性检出率（worst-group sensitivity）**上，深层 HN 显著**恶化**——
  HyperHead 0.409 / HyperAdapt 0.398 **低于 ERM 0.613**（漏诊率从 39% 升到 60%），**sensitivity gap 近乎翻倍**
  （ERM 0.23 → HN 0.49）；HyperFusion 打平 ERM。**SWAD（不用属性）worst-group sensitivity 最高 0.699**。
- 即 HN 的 Overall sensitivity 小幅↑（D-A5）是靠**多数/易组**换来的，**最差子群的恶性检出反而更差**——
  worst-group 临床四联把 AUC 口径下的「打平/微负」放大为「在部署最关心的漏诊率上主动变差」。
- ⚠️ 口径：worst-group 四联为池化 OOF 上的**点估计**（未附 bootstrap CI）；worst-group 的统计判决以
  D-A3（AUC 配对 bootstrap）为准，本表是其阈值层的**描述性佐证**，方向一致。

---

## 4. 判决：把 D7-0 的对冲升级为「有功效的否定」

1. **worst-group：HN 无增益，且这是有功效的判决（非评估噪声）**。评估-n 地板移除后（少数格 n
   放大 5–10×、0/14 进地板），3 HN 的 worst-group 对 ERM 打平（点估计一律微负、CI 收窄仍跨 0），
   且**显著输给 attribute-blind 的 SWAD**（边缘口径 3/3 显著、canonical HyperHead 显著）。
   即先前「worst-group 未证实，疑似评估-n 受限」正式升级为 **「即使解除评估-n 限制，HN 仍不能把
   `I(Y;age|X)>0` 转化为 worst-group 公平提升」**（⚠️ 新 OOF V-info 闸门下 **HAM-age 已退回未检出**，见 `docs/conditional_v_information_gate.md`——前提本身不再成立，本节否定结论**方向不变、且更属预期**）。**临床四联（D-A6）同向佐证且更刺眼**：固定
   specificity=0.8 下，深层 HN 的 **worst-group sensitivity（最差子群恶性检出率）反而显著低于 ERM**
   （HyperHead 0.409 / HyperAdapt 0.398 vs ERM 0.613，漏诊率 39%→60%，sensitivity gap 近翻倍），
   SWAD（不用属性）最高——HN 的 Overall 增益是靠多数/易组换来的。

2. **Overall：增益复现且稳健，但以更大 gap 为代价**。HN vs ERM/ROC 的 Overall AUC 显著更高
   （p≤0.01，复现 D6-HP），但 HN 的 AUC gap 反而大于 ERM——整体判别的提升靠拉大组间差距、
   不惠及（甚至损及）最差组。**深度轴**：HyperAdapt Overall 最高却 worst 最低、gap 最大。

3. **HN vs SWAD 的 Overall 优势不稳健**。CV-OOF 下 HN 在 Overall 上反而**显著输给 SWAD**
   （HyperHead/HyperFusion Δ≈−0.013，p<0.001），与单-split D6-HP「HF/HA 赢 SWAD」**符号翻转**，
   印证 R5.2「HN vs SWAD 比较不对称、不作主张」；两个 regime 都稳的是 **HN vs ERM（对称调优）**。

**机制自洽**：这与 R1 合成剂量-反应一致——边际先验型的条件信号只改池化判别、不改组内最差子群排序。
路线 A 把「HN 价值局限于 Overall」从「评估受限的推测」坐实为 **HAM 上有功效的判决**。

---

## 5. 可复现

| 产物 | 路径 |
|------|------|
| CV 折生成 | `scripts/build_ham10000_splits.py --cv 5`（`build_cv_folds`，lesion GroupKFold + OOF 完整性断言） |
| 训练脚本 `--cv/--fold` | `src/training/train_ham10000_{resnet18,hyperhead,hyperfusion,hyperadapt}.py`（`resolve_paths` 隔离 output 到 cv5） |
| 提交脚本 | `slurm/c1_ham_cv_oof.sh`（array 0–4=fold，config 经 --export，排除 semois） |
| OOF 显著性 | `scripts/ham_cv_oof_significance.py`（池化 + 对齐断言 + D-A1~D-A4） |
| 预测 / ROC | `outputs/ham10000/cv5/predictions/`（4 作业 70451–70454 + `derive_roc --output-dir .../cv5`） |
