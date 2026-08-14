# 跨数据集 / 跨架构 worst-group 公平性汇总

> 🛑 **信号闸门口径更新（2026-07-19）**：闸门已由单-split 条件互信息改为 **OOF conditional V-information**
> （Xu 2020 / Hewitt 2021；权威 = `docs/conditional_v_information_gate.md`）。关键变化：
> **HAM-age 由「唯一正信号」退回「未检出」**（+0.0028 bits，CI 触 0）；Fitzpatrick ≈0；
> **CheXpert / MIMIC 全轴 CI<0 = 决定性 0**；唯一 detected 为 **PAPILA-age**（+0.041 bits，但 n=420）。
> control task 五库全部通过 ⇒ 零读数非假阴。本文「HAM-age `I(Y;age|X)>0`」一族表述**已过时**（下方已内联订正）；
> 各处**结论方向不变**，但「有信号」前提须按新闸门重读。


> 建立：2026-07-13。**口径重构：2026-07-19 由 pooled-OOF 改为 averaging**（详见下方「口径」节与
> [pooled_oof_calibration_note] 记忆）。核心问题：**敏感属性条件化超网络（HN）是否能够跨架构和数据集
> 稳定改善医学图像分类的 worst-group fairness？** 本文档在**已落盘的 CV-OOF 预测**上（零重训）汇总四份
> 相对 ERM 基线的支撑数据。
>
> - **生成脚本**：[scripts/build_worstgroup_crossdataset_tables.py](../scripts/build_worstgroup_crossdataset_tables.py)
> - **机器可读 CSV**：`outputs/analysis/crossds_worstgroup/{d1,d2,d3,d4}_*.csv`
> - **口径 = averaging**（**逐折算子群指标 → 折间平均** `mean_f metric(g,f)`；worst/best 用「先平均后
>   min/max」`min_g[mean_f AUC(g,f)]`）。**不再跨折池化 raw logit**——各折是不同模型、早停轮数异质使
>   logit 尺度漂移，池化对排序指标注入系统性负偏（Parker 2007 / Airola 2010；本项目实测 HN 受害远重于
>   ERM，曾据旧口径制造高置信假阳性）。**零重训**：同一批逐折预测，仅改聚合。D2 各方法 marginal-worst
>   AUC 与权威 `outputs/conditioning_ablation/oof_results_averaging.json` **逐位一致**（已核验）。
> - 子群集合复用 [subgroup_auc.subgroup_masks](../src/training/harness/subgroup_auc.py)（单一事实来源）。
>   **headline 用 marginal 边缘子群**（joint 小格在阈值/单折下退化成噪声）；D2 额外给 canonical。
> - AUC 阈值无关；TPR/FPR/spec/sens/PPV/NPV 需阈值 → **每折各方法各自** `threshold_at_overall_fpr(·,0.2)`
>   校准操作点、算子群指标、再折间平均（整体 FPR 对齐 0.2，隔离整体操作点差异，只留子群差异）。TPR≡sensitivity、FPR≡1−specificity。
> - 数据集规模：HAM10000 n=9,707（阳性 14.8%）/ PAPILA n=420（20.7%）/ Fitzpatrick n=16,012（13.5%）/
>   CheXpert n=138,644（8.2%）/ MIMIC n=199,356（30.2%）。属性介入深度递增：HyperHead(浅,仅头) →
>   HyperFusion(中,单 block) → HyperAdapt(深,除 stem 外所有层)。
> - **显著性**（配对 cluster bootstrap，B=1000，Δmarginal-worst vs ERM/SWAD）来自 averaging 权威 JSON，
>   本文表格为描述性点估计、显著性在结论散文中标注。

---

## 结论（先说答案）

**否。HN 未能跨架构与数据集稳定改善 worst-group fairness。** 但相较旧 pooled 版本，**支撑理由已随
averaging 口径修正**（旧版「HyperHead 最浅最有害」「仅 HAM·HyperFusion 实质抬升」两条均系池化伪影，已推翻）。
逐条证据：

1. **稳定性（D3 逐折）仍是最直接的否证**（逐折本就 averaging 家族、对池化漂移免疫，数值与旧版一致）：
   5 折 worst-group ΔAUC 的**折均全部落在 ±0.026 内**，且**逐折符号不稳定**——同一 (数据集, HN) 在不同折
   间正负翻转（HAM·HyperHead fold1 −0.032↔fold4 +0.046；PAPILA·HyperAdapt fold0 −0.107↔fold3 +0.110）。
   没有任何 HN 在任何数据集上给出**逐折一致为正**的 worst-group 改善；折间抖动 ≫ 折均。

2. **无 HN 给出显著且有实质幅度的 worst-group 赢**（配对 bootstrap，averaging）：
   - **HAM**（~~唯一 `I(Y;age|X)>0`~~ → 新 OOF V-info 闸门下 **未检出、≈0**）：averaging 下三 HN 均把 worst 组 AUC **抬高**（Δworst
     HyperHead +0.018 / HyperFusion +0.033 / HyperAdapt +0.021），**但三者 vs ERM 全部 n.s.**；
     唯一显著胜 ERM 的是 **SWAD +0.046（SIG）**。
   - **Fitzpatrick**：三 HN vs ERM 全 n.s.；唯一显著胜 ERM 的又是 **SWAD +0.058（SIG）**；HyperFusion
     反而**显著劣于 SWAD**。
   - **大样本 CXR**：微小但个别显著——**MIMIC·HyperAdapt +0.0022（SIG）**是唯一「深层 HN 显著为正」，
     然 |Δ|<0.005；CheXpert·HyperFusion +0.0029（SIG），CheXpert·HyperHead −0.0040（SIG 但为负）。
     这些显著全靠 n≥138k 的功效，**显著 ≠ 重要**。
   - **PAPILA**：n=420，全部 n.s.，无功效可言。

3. **旧「HyperHead 最浅最有害」被推翻（池化伪影）**：HyperHead 跨折 logit SD 在 HAM 最高，池化负偏最重。
   旧版 HAM·HyperHead worst=0.7195、vs ERM Δworst −0.0552、gap 加宽 +0.039；averaging 修正后
   worst=**0.8159**、Δworst **+0.0182**、gap **收窄 −0.0274**。它既不胜也不害。D2 里 HAM·HyperHead 的
   worst 组虽仍是 age:20-40，但已是**健康 AUC（0.816）**而非塌陷——只是几个健康组里边缘最低者，非「误差再分配」。

4. **「gap 收窄的来源」也随口径改写（D4）**：averaging 下 **HAM 三 HN 全为「worst 侧↑抬高」**（leveling-up，
   Δworst 均 >0），旧版「HyperHead 压低 worst / 仅 HyperFusion 抬升」的分裂叙事消失。PAPILA 少数「worst 侧↓
   压低」但 n.s. 且 n 极小；CXR 双库多为双侧微动。**没有一处兼具「显著 + 抬弱组 + 实质幅度」**。

5. **跨架构仍无一致赢家，深度趋势弱**：HyperAdapt（最深）的 worst 侧移动最常非负（唯一在 MIMIC 显著），
   与「属性条件化越深越能兑现公平」方向一致，但幅度全 <0.005，且三 HN 混杂范围/参数化/容量/初始化多轴，
   不能把这点微弱优势归因给「深度」（受控消融 E1 见 [[conditioning-ablation-e0-eaudit]]，H1/H2 均 n.s.）。

6. **与既有判决一致**：本结果与路线 A（[[routeA-ham-cv-oof-worstgroup]]）、冻结 regime 实验
   （[[frozen-regime-ham-experiment]] / [[frozen-regime-fitzpatrick-experiment]]）的「有功效否定」相互印证——
   移除评估-n 地板后，HN 的 worst-group 对 ERM 大体打平（微正/微负、n.s.），且 ID 下一律不显著优于 SWAD。

**一句话**：worst-group 公平的改善**不稳定、不跨数据集、不跨架构**；averaging 口径下 HN 对 ERM 的 worst-group
改善处处不显著或幅度可忽略，而**唯一稳健且有实质幅度的 worst-group 赢家是 SWAD（HAM/Fitz）**，非任何 HN。

---

<!-- 以下 D1–D4 表格由脚本生成（averaging 口径），直接粘贴；数值全量见对应 CSV。 -->

### D1 · 各 HN vs ERM 逐子群 7 指标差（每 (数据集,HN) 一张表）
> 阈值相关指标（TPR/FPR/spec/sens/PPV/NPV）取**每折各方法各自**校准到整体 FPR=0.2 的操作点、算子群指标后折间平均；AUC 阈值无关；TPR≡sensitivity、FPR≡1−specificity。
> 除 FPR「越小越好」外各列均「越大越好」；正号=HN 优于 ERM。数值全量另见 `d1_per_group_metric_diffs.csv`。

#### ham10000 · HyperHead − ERM，各边缘子群 7 指标差

| 子群 | ΔAUC | ΔTPR | ΔFPR | Δspec | Δsens | ΔPPV | ΔNPV |
|------|------|------|------|------|------|------|------|
| sex:Male | +0.0011 | +0.0067 | +0.0030 | -0.0030 | +0.0067 | -0.0015 | +0.0017 |
| sex:Female | +0.0004 | -0.0058 | -0.0023 | +0.0023 | -0.0058 | +0.0004 | -0.0017 |
| age:20-40 | -0.0216 | -0.1819 | -0.0641 | +0.0641 | -0.1819 | +0.0235 | -0.0100 |
| age:40-60 | -0.0092 | -0.0563 | -0.0052 | +0.0052 | -0.0563 | -0.0064 | -0.0060 |
| age:60-80 | +0.0003 | +0.0428 | +0.0507 | -0.0507 | +0.0428 | -0.0295 | +0.0125 |
| age:80+ | +0.0425 | +0.0575 | +0.0456 | -0.0456 | +0.0575 | -0.0085 | +0.0368 |

#### ham10000 · HyperFusion − ERM，各边缘子群 7 指标差

| 子群 | ΔAUC | ΔTPR | ΔFPR | Δspec | Δsens | ΔPPV | ΔNPV |
|------|------|------|------|------|------|------|------|
| sex:Male | +0.0055 | +0.0231 | +0.0079 | -0.0079 | +0.0231 | -0.0019 | +0.0055 |
| sex:Female | +0.0076 | +0.0212 | -0.0092 | +0.0092 | +0.0212 | +0.0195 | +0.0027 |
| age:20-40 | -0.0064 | -0.1200 | -0.0914 | +0.0914 | -0.1200 | +0.1319 | -0.0050 |
| age:40-60 | -0.0048 | +0.0351 | +0.0116 | -0.0116 | +0.0351 | -0.0103 | +0.0039 |
| age:60-80 | +0.0032 | +0.0321 | +0.0445 | -0.0445 | +0.0321 | -0.0288 | +0.0089 |
| age:80+ | +0.0472 | +0.0343 | +0.0037 | -0.0037 | +0.0343 | +0.0043 | +0.0271 |

#### ham10000 · HyperAdapt − ERM，各边缘子群 7 指标差

| 子群 | ΔAUC | ΔTPR | ΔFPR | Δspec | Δsens | ΔPPV | ΔNPV |
|------|------|------|------|------|------|------|------|
| sex:Male | -0.0079 | -0.0043 | +0.0025 | -0.0025 | -0.0043 | -0.0044 | -0.0009 |
| sex:Female | -0.0048 | -0.0094 | -0.0066 | +0.0066 | -0.0094 | +0.0056 | -0.0016 |
| age:20-40 | +0.0029 | -0.0917 | -0.0640 | +0.0640 | -0.0917 | +0.0647 | -0.0050 |
| age:40-60 | -0.0092 | -0.0753 | -0.0148 | +0.0148 | -0.0753 | +0.0038 | -0.0074 |
| age:60-80 | -0.0086 | +0.0327 | +0.0549 | -0.0549 | +0.0327 | -0.0352 | +0.0084 |
| age:80+ | +0.0212 | +0.0578 | +0.0574 | -0.0574 | +0.0578 | -0.0107 | +0.0346 |

#### papila · HyperHead − ERM，各边缘子群 7 指标差

| 子群 | ΔAUC | ΔTPR | ΔFPR | Δspec | Δsens | ΔPPV | ΔNPV |
|------|------|------|------|------|------|------|------|
| sex:Male | -0.0599 | -0.1200 | +0.1004 | -0.1004 | -0.1200 | -0.1811 | -0.0313 |
| sex:Female | +0.0275 | -0.0214 | +0.0175 | -0.0175 | -0.0214 | -0.0275 | -0.0073 |
| age:<=60 | +0.0036 | +0.0125 | -0.0213 | +0.0213 | +0.0125 | +0.0611 | -0.0073 |
| age:>60 | -0.0375 | -0.0501 | +0.0913 | -0.0913 | -0.0501 | -0.1085 | -0.0219 |

#### papila · HyperFusion − ERM，各边缘子群 7 指标差

| 子群 | ΔAUC | ΔTPR | ΔFPR | Δspec | Δsens | ΔPPV | ΔNPV |
|------|------|------|------|------|------|------|------|
| sex:Male | +0.0203 | +0.0250 | +0.0528 | -0.0528 | +0.0250 | -0.0435 | -0.0124 |
| sex:Female | +0.0140 | +0.0563 | +0.0377 | -0.0377 | +0.0563 | -0.0276 | +0.0112 |
| age:<=60 | +0.0257 | +0.1250 | -0.0060 | +0.0060 | +0.1250 | +0.0254 | +0.0130 |
| age:>60 | -0.0323 | -0.0084 | +0.1058 | -0.1058 | -0.0084 | -0.1095 | +0.0002 |

#### papila · HyperAdapt − ERM，各边缘子群 7 指标差

| 子群 | ΔAUC | ΔTPR | ΔFPR | Δspec | Δsens | ΔPPV | ΔNPV |
|------|------|------|------|------|------|------|------|
| sex:Male | -0.0461 | -0.0700 | +0.1120 | -0.1120 | -0.0700 | -0.1800 | -0.0128 |
| sex:Female | +0.0555 | +0.0683 | +0.0259 | -0.0259 | +0.0683 | -0.0230 | +0.0186 |
| age:<=60 | +0.0327 | +0.0875 | -0.0107 | +0.0107 | +0.0875 | +0.0311 | -0.0027 |
| age:>60 | -0.0123 | +0.0260 | +0.1154 | -0.1154 | +0.0260 | -0.1096 | +0.0190 |

#### fitzpatrick · HyperHead − ERM，各边缘子群 7 指标差

| 子群 | ΔAUC | ΔTPR | ΔFPR | Δspec | Δsens | ΔPPV | ΔNPV |
|------|------|------|------|------|------|------|------|
| skin:I | +0.0116 | -0.0045 | -0.0216 | +0.0216 | -0.0045 | +0.0239 | -0.0001 |
| skin:II | +0.0056 | -0.0028 | -0.0059 | +0.0059 | -0.0028 | +0.0061 | -0.0005 |
| skin:III | +0.0171 | +0.0306 | +0.0095 | -0.0095 | +0.0306 | -0.0037 | +0.0055 |
| skin:IV | +0.0292 | +0.0733 | -0.0073 | +0.0073 | +0.0733 | +0.0322 | +0.0100 |
| skin:V | +0.0221 | +0.0421 | +0.0087 | -0.0087 | +0.0421 | +0.0060 | +0.0048 |
| skin:VI | +0.0266 | +0.0154 | +0.0297 | -0.0297 | +0.0154 | -0.0164 | -0.0003 |

#### fitzpatrick · HyperFusion − ERM，各边缘子群 7 指标差

| 子群 | ΔAUC | ΔTPR | ΔFPR | Δspec | Δsens | ΔPPV | ΔNPV |
|------|------|------|------|------|------|------|------|
| skin:I | +0.0207 | +0.0109 | -0.0168 | +0.0168 | +0.0109 | +0.0225 | +0.0030 |
| skin:II | +0.0072 | +0.0081 | -0.0194 | +0.0194 | +0.0081 | +0.0263 | +0.0027 |
| skin:III | +0.0146 | +0.0263 | +0.0165 | -0.0165 | +0.0263 | -0.0119 | +0.0045 |
| skin:IV | +0.0132 | +0.0733 | +0.0347 | -0.0347 | +0.0733 | -0.0276 | +0.0095 |
| skin:V | +0.0095 | -0.0133 | -0.0036 | +0.0036 | -0.0133 | -0.0042 | -0.0011 |
| skin:VI | +0.0007 | +0.0321 | -0.0071 | +0.0071 | +0.0321 | +0.0028 | +0.0041 |

#### fitzpatrick · HyperAdapt − ERM，各边缘子群 7 指标差

| 子群 | ΔAUC | ΔTPR | ΔFPR | Δspec | Δsens | ΔPPV | ΔNPV |
|------|------|------|------|------|------|------|------|
| skin:I | +0.0015 | -0.0399 | -0.0172 | +0.0172 | -0.0399 | +0.0076 | -0.0080 |
| skin:II | -0.0023 | +0.0094 | +0.0079 | -0.0079 | +0.0094 | -0.0045 | +0.0017 |
| skin:III | +0.0001 | -0.0109 | +0.0112 | -0.0112 | -0.0109 | -0.0173 | -0.0024 |
| skin:IV | +0.0093 | +0.0368 | +0.0105 | -0.0105 | +0.0368 | -0.0061 | +0.0046 |
| skin:V | +0.0098 | +0.0485 | +0.0058 | -0.0058 | +0.0485 | +0.0081 | +0.0060 |
| skin:VI | +0.0170 | -0.0333 | -0.0663 | +0.0663 | -0.0333 | +0.0779 | -0.0010 |

#### chexpert · HyperHead − ERM，各边缘子群 7 指标差

| 子群 | ΔAUC | ΔTPR | ΔFPR | Δspec | Δsens | ΔPPV | ΔNPV |
|------|------|------|------|------|------|------|------|
| sex:Male | -0.0048 | -0.0228 | -0.0054 | +0.0054 | -0.0228 | -0.0004 | -0.0022 |
| sex:Female | -0.0024 | +0.0034 | +0.0090 | -0.0090 | +0.0034 | -0.0078 | +0.0001 |
| race:White | -0.0031 | -0.0132 | -0.0006 | +0.0006 | -0.0132 | -0.0026 | -0.0013 |
| race:Non-White | -0.0053 | -0.0063 | +0.0044 | -0.0044 | -0.0063 | -0.0057 | -0.0008 |
| age:<60 | -0.0043 | -0.0125 | -0.0077 | +0.0077 | -0.0125 | +0.0033 | -0.0021 |
| age:>=60 | -0.0040 | -0.0096 | +0.0056 | -0.0056 | -0.0096 | -0.0078 | -0.0008 |

#### chexpert · HyperFusion − ERM，各边缘子群 7 指标差

| 子群 | ΔAUC | ΔTPR | ΔFPR | Δspec | Δsens | ΔPPV | ΔNPV |
|------|------|------|------|------|------|------|------|
| sex:Male | +0.0001 | +0.0021 | -0.0030 | +0.0030 | +0.0021 | +0.0034 | +0.0003 |
| sex:Female | +0.0023 | +0.0083 | +0.0048 | -0.0048 | +0.0083 | -0.0027 | +0.0008 |
| race:White | +0.0013 | +0.0029 | -0.0006 | +0.0006 | +0.0029 | +0.0013 | +0.0003 |
| race:Non-White | +0.0002 | +0.0106 | +0.0034 | -0.0034 | +0.0106 | -0.0006 | +0.0012 |
| age:<60 | +0.0001 | -0.0033 | -0.0084 | +0.0084 | -0.0033 | +0.0059 | -0.0003 |
| age:>=60 | +0.0029 | +0.0179 | +0.0057 | -0.0057 | +0.0179 | -0.0015 | +0.0010 |

#### chexpert · HyperAdapt − ERM，各边缘子群 7 指标差

| 子群 | ΔAUC | ΔTPR | ΔFPR | Δspec | Δsens | ΔPPV | ΔNPV |
|------|------|------|------|------|------|------|------|
| sex:Male | -0.0030 | -0.0076 | -0.0038 | +0.0038 | -0.0076 | +0.0018 | -0.0007 |
| sex:Female | -0.0024 | +0.0074 | +0.0069 | -0.0069 | +0.0074 | -0.0050 | +0.0007 |
| race:White | -0.0022 | -0.0025 | +0.0005 | -0.0005 | -0.0025 | -0.0011 | -0.0003 |
| race:Non-White | -0.0037 | +0.0029 | +0.0010 | -0.0010 | +0.0029 | -0.0001 | +0.0003 |
| age:<60 | -0.0039 | -0.0045 | -0.0028 | +0.0028 | -0.0045 | +0.0011 | -0.0008 |
| age:>=60 | -0.0023 | +0.0039 | +0.0027 | -0.0027 | +0.0039 | -0.0018 | +0.0002 |

#### mimic · HyperHead − ERM，各边缘子群 7 指标差

| 子群 | ΔAUC | ΔTPR | ΔFPR | Δspec | Δsens | ΔPPV | ΔNPV |
|------|------|------|------|------|------|------|------|
| sex:Male | -0.0010 | +0.0004 | -0.0014 | +0.0014 | +0.0004 | +0.0021 | +0.0003 |
| sex:Female | +0.0006 | +0.0026 | +0.0017 | -0.0017 | +0.0026 | -0.0009 | +0.0011 |
| race:White | -0.0001 | +0.0020 | -0.0001 | +0.0001 | +0.0020 | +0.0008 | +0.0008 |
| race:Non-White | +0.0002 | +0.0007 | +0.0003 | -0.0003 | +0.0007 | -0.0001 | +0.0002 |
| age:<60 | -0.0007 | -0.0036 | -0.0021 | +0.0021 | -0.0036 | +0.0006 | -0.0025 |
| age:>=60 | +0.0004 | +0.0060 | +0.0007 | -0.0007 | +0.0060 | +0.0011 | +0.0017 |

#### mimic · HyperFusion − ERM，各边缘子群 7 指标差

| 子群 | ΔAUC | ΔTPR | ΔFPR | Δspec | Δsens | ΔPPV | ΔNPV |
|------|------|------|------|------|------|------|------|
| sex:Male | -0.0009 | -0.0077 | -0.0071 | +0.0071 | -0.0077 | +0.0065 | -0.0020 |
| sex:Female | +0.0003 | +0.0092 | +0.0086 | -0.0086 | +0.0092 | -0.0066 | +0.0033 |
| race:White | -0.0006 | -0.0002 | -0.0004 | +0.0004 | -0.0002 | +0.0004 | -0.0000 |
| race:Non-White | +0.0004 | +0.0032 | +0.0011 | -0.0011 | +0.0032 | -0.0002 | +0.0015 |
| age:<60 | -0.0011 | -0.0050 | -0.0072 | +0.0072 | -0.0050 | +0.0042 | -0.0028 |
| age:>=60 | +0.0001 | +0.0056 | +0.0029 | -0.0029 | +0.0056 | -0.0022 | +0.0013 |

#### mimic · HyperAdapt − ERM，各边缘子群 7 指标差

| 子群 | ΔAUC | ΔTPR | ΔFPR | Δspec | Δsens | ΔPPV | ΔNPV |
|------|------|------|------|------|------|------|------|
| sex:Male | +0.0004 | -0.0031 | -0.0017 | +0.0017 | -0.0031 | +0.0011 | -0.0009 |
| sex:Female | +0.0018 | +0.0026 | +0.0019 | -0.0019 | +0.0026 | -0.0014 | +0.0009 |
| race:White | +0.0011 | -0.0011 | -0.0002 | +0.0002 | -0.0011 | -0.0002 | -0.0004 |
| race:Non-White | +0.0017 | +0.0022 | +0.0003 | -0.0003 | +0.0022 | +0.0003 | +0.0012 |
| age:<60 | -0.0001 | -0.0053 | -0.0038 | +0.0038 | -0.0053 | +0.0015 | -0.0035 |
| age:>=60 | +0.0022 | +0.0042 | +0.0014 | -0.0014 | +0.0042 | -0.0006 | +0.0011 |

### D2 · 各方法 worst-group 组别（averaging，折均 AUC 上取 argmin）

| 数据集 | 方法 | marginal worst 组 | AUC | canonical worst 组 | AUC |
|--------|------|-------------------|-----|--------------------|-----|
| ham10000 | ERM | age:80+ | 0.7976 | sex:Male\|age:80+ | 0.7877 |
| ham10000 | SWAD | age:80+ | 0.8433 | sex:Male\|age:80+ | 0.8331 |
| ham10000 | ROC | age:80+ | 0.7821 | sex:Male\|age:80+ | 0.7744 |
| ham10000 | HyperHead | age:20-40 | 0.8159 | sex:Male\|age:20-40 | 0.7712 |
| ham10000 | HyperFusion | age:20-40 | 0.8310 | sex:Male\|age:20-40 | 0.8171 |
| ham10000 | HyperAdapt | age:80+ | 0.8188 | sex:Female\|age:80+ | 0.7930 |
| papila | ERM | age:>60 | 0.8277 | sex:Female\|age:>60 | 0.8245 |
| papila | SWAD | age:>60 | 0.8280 | age:>60 | 0.8280 |
| papila | ROC | age:>60 | 0.8277 | sex:Female\|age:>60 | 0.8245 |
| papila | HyperHead | age:>60 | 0.7902 | sex:Male\|age:>60 | 0.7071 |
| papila | HyperFusion | age:>60 | 0.7955 | sex:Male\|age:<=60 | 0.6458 |
| papila | HyperAdapt | age:>60 | 0.8155 | sex:Male\|age:<=60 | 0.6250 |
| fitzpatrick | ERM | skin:VI | 0.8161 | skin:VI | 0.8161 |
| fitzpatrick | SWAD | skin:VI | 0.8739 | skin:VI | 0.8739 |
| fitzpatrick | ROC | skin:VI | 0.8071 | skin:VI | 0.8071 |
| fitzpatrick | HyperHead | skin:VI | 0.8427 | skin:VI | 0.8427 |
| fitzpatrick | HyperFusion | skin:VI | 0.8168 | skin:VI | 0.8168 |
| fitzpatrick | HyperAdapt | skin:VI | 0.8331 | skin:VI | 0.8331 |
| chexpert | ERM | age:>=60 | 0.8377 | sex:Female\|race:Non-White\|age:>=60 | 0.8316 |
| chexpert | SWAD | age:>=60 | 0.8354 | sex:Female\|race:Non-White\|age:>=60 | 0.8317 |
| chexpert | ROC | age:>=60 | 0.8365 | sex:Female\|race:Non-White\|age:>=60 | 0.8299 |
| chexpert | HyperHead | age:>=60 | 0.8337 | sex:Female\|race:Non-White\|age:>=60 | 0.8322 |
| chexpert | HyperFusion | age:>=60 | 0.8406 | sex:Male\|race:White\|age:>=60 | 0.8378 |
| chexpert | HyperAdapt | age:>=60 | 0.8354 | sex:Male\|race:Non-White\|age:>=60 | 0.8313 |
| mimic | ERM | age:>=60 | 0.8178 | sex:Male\|race:White\|age:>=60 | 0.8089 |
| mimic | SWAD | age:>=60 | 0.8194 | sex:Male\|race:White\|age:>=60 | 0.8096 |
| mimic | ROC | age:>=60 | 0.8178 | sex:Male\|race:White\|age:>=60 | 0.8089 |
| mimic | HyperHead | age:>=60 | 0.8182 | sex:Male\|race:White\|age:>=60 | 0.8082 |
| mimic | HyperFusion | age:>=60 | 0.8179 | sex:Male\|race:White\|age:>=60 | 0.8087 |
| mimic | HyperAdapt | age:>=60 | 0.8201 | sex:Male\|race:White\|age:>=60 | 0.8097 |

> **读法**：CXR 两库与 Fitzpatrick 的 worst 组身份**跨方法稳定**（age:>=60 / skin:VI），HN 仅微调其 AUC。
> **HAM**：ERM/SWAD/ROC/HyperAdapt 的 worst 组为 age:80+，HyperHead/HyperFusion 为 age:20-40——但两者
> 的 age:20-40 AUC（0.816 / 0.831）均属**健康量级**（≈ 其余组），是「几个健康组里边缘最低者」，而非旧
> pooled 版本里 age:20-40 塌陷到 0.72 的「误差再分配」（那是池化伪影，见 [[pooled-oof-calibration-drift-artifact]]）。
> PAPILA 的 canonical worst 落在 sex×age 小格（n 极小、AUC 噪声大），marginal 口径更可信。

### D3 · 逐折 worst-group AUC 差（HN − ERM，marginal，每折各自 argmin worst）
> **本表逐折算、对池化漂移免疫**（本就是 averaging 家族），数值与旧版一致，是稳定性否证的核心证据。

| 数据集 | HN | fold0 | fold1 | fold2 | fold3 | fold4 | 均值 |
|--------|----|-------|-------|-------|-------|-------|------|
| ham10000 | HyperHead | +0.031 | -0.032 | -0.042 | +0.041 | +0.046 | +0.009 |
| ham10000 | HyperFusion | +0.037 | -0.015 | -0.022 | +0.008 | +0.002 | +0.002 |
| ham10000 | HyperAdapt | +0.068 | -0.027 | -0.054 | -0.020 | +0.041 | +0.002 |
| papila | HyperHead | -0.067 | -0.173 | +0.006 | +0.080 | +0.024 | -0.026 |
| papila | HyperFusion | +0.024 | -0.092 | +0.027 | +0.076 | -0.011 | +0.005 |
| papila | HyperAdapt | -0.107 | -0.108 | +0.085 | +0.110 | -0.009 | -0.006 |
| fitzpatrick | HyperHead | -0.075 | -0.028 | +0.137 | +0.043 | +0.038 | +0.023 |
| fitzpatrick | HyperFusion | +0.025 | -0.016 | +0.036 | +0.054 | -0.085 | +0.003 |
| fitzpatrick | HyperAdapt | -0.007 | -0.021 | +0.087 | +0.007 | +0.039 | +0.021 |
| chexpert | HyperHead | +0.002 | -0.008 | -0.001 | -0.008 | -0.005 | -0.004 |
| chexpert | HyperFusion | +0.004 | +0.004 | +0.001 | +0.003 | +0.003 | +0.003 |
| chexpert | HyperAdapt | -0.000 | -0.007 | +0.001 | -0.004 | -0.001 | -0.002 |
| mimic | HyperHead | +0.005 | -0.002 | +0.002 | -0.003 | +0.000 | +0.000 |
| mimic | HyperFusion | +0.005 | -0.001 | +0.001 | -0.002 | -0.002 | +0.000 |
| mimic | HyperAdapt | +0.008 | +0.001 | +0.001 | +0.000 | +0.001 | +0.002 |

> **读法（稳定性的核心证据）**：小样本数据集（HAM/PAPILA/Fitz）折间符号剧烈翻转，折内 |Δ| 可达
> 0.05~0.17，**折均被抵消到近 0**——「改善」无法与折间噪声区分。大样本 CXR（CheXpert/MIMIC）折间
> 稳定但**幅度全部 <0.008**，实际可忽略。**没有任何 (数据集,HN) 给出逐折一致为正的 worst-group 抬升。**

### D4 · best−worst gap 相对 ERM 的差及分解 Δgap = Δbest − Δworst（marginal，averaging）

| 数据集 | HN | gap_ERM | gap_HN | Δgap | Δbest | Δworst | 主导 |
|--------|----|---------|--------|------|-------|--------|------|
| ham10000 | HyperHead | 0.1166 | 0.0891 | -0.0274 | -0.0092 | +0.0182 | worst 侧↑抬高 |
| ham10000 | HyperFusion | 0.1166 | 0.0783 | -0.0382 | -0.0048 | +0.0334 | worst 侧↑抬高 |
| ham10000 | HyperAdapt | 0.1166 | 0.0861 | -0.0305 | -0.0092 | +0.0212 | worst 侧↑抬高 |
| papila | HyperHead | 0.0402 | 0.0813 | +0.0411 | +0.0036 | -0.0375 | worst 侧↓压低 |
| papila | HyperFusion | 0.0402 | 0.0982 | +0.0580 | +0.0257 | -0.0323 | worst 侧↓压低 |
| papila | HyperAdapt | 0.0402 | 0.0851 | +0.0449 | +0.0327 | -0.0123 | best 侧↑抬高 |
| fitzpatrick | HyperHead | 0.0904 | 0.0915 | +0.0010 | +0.0277 | +0.0266 | best 侧↑抬高 |
| fitzpatrick | HyperFusion | 0.0904 | 0.1037 | +0.0133 | +0.0141 | +0.0007 | best 侧↑抬高 |
| fitzpatrick | HyperAdapt | 0.0904 | 0.0832 | -0.0072 | +0.0098 | +0.0170 | worst 侧↑抬高 |
| chexpert | HyperHead | 0.0389 | 0.0388 | -0.0001 | -0.0041 | -0.0040 | best 侧↓压低 |
| chexpert | HyperFusion | 0.0389 | 0.0366 | -0.0023 | +0.0006 | +0.0029 | worst 侧↑抬高 |
| chexpert | HyperAdapt | 0.0389 | 0.0373 | -0.0016 | -0.0039 | -0.0023 | best 侧↓压低 |
| mimic | HyperHead | 0.0466 | 0.0455 | -0.0011 | -0.0007 | +0.0004 | best 侧↓压低 |
| mimic | HyperFusion | 0.0466 | 0.0454 | -0.0012 | -0.0011 | +0.0001 | best 侧↓压低 |
| mimic | HyperAdapt | 0.0466 | 0.0443 | -0.0023 | -0.0001 | +0.0022 | worst 侧↑抬高 |

> **读法**：恒等式 `Δgap = Δbest − Δworst` 逐行成立（如 HAM·HyperHead：−0.0092−(+0.0182)=−0.0274）。
> **Δgap<0（收窄）**理想上应由 **Δworst>0（抬起弱组）**驱动。**averaging 口径下 HAM 三 HN 均为「抬弱组
> 收 gap」**（Δworst +0.018~+0.033 主导），这与旧 pooled 版本「HyperHead 压低 worst、加宽 gap」相反——旧
> 结论是池化伪影。但注意：HAM 三 HN 的 Δworst 虽为正，**配对 bootstrap 下 vs ERM 全 n.s.**（见结论 2），
> 且 gap 收窄同时 Δbest 微负（未牺牲强组但也未显著抬弱组）。PAPILA gap 反被 HN 加宽（n 极小、n.s.）；
> CXR 双库多为双侧微动、幅度可忽略。

---

## 复现

```bash
source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
python scripts/build_worstgroup_crossdataset_tables.py            # 全 5 数据集 → 上表 + CSV（averaging 口径）
python scripts/build_worstgroup_crossdataset_tables.py --dataset ham10000   # 单数据集
```

选定 config 来自各数据集 `outputs/<ds>[/cv5]/selected_configs.json`（S1 折均 marginal-worst 选择器产物）。
本文档只做**描述性汇总**（averaging 点估计）；worst-group 的**配对显著性**（cluster bootstrap，B=1000）见
权威 `outputs/conditioning_ablation/oof_results_averaging.json`（生成器
[scripts/build_oof_results_averaging.py](../scripts/build_oof_results_averaging.py)），已在结论散文中引用，
与本文 D2 marginal-worst AUC 逐位一致。

> **口径变更史**：本文档 2026-07-13 初版为 pooled-OOF；2026-07-19 整体改 averaging（零重训、仅改聚合），
> 旧 pooled 版见 git 历史。变更原因（池化对排序指标的系统性负偏、HN 受害重于 ERM、曾制造高置信假阳性）
> 与文献依据见 [[pooled-oof-calibration-drift-artifact]] 与 `docs/oof_regime_results.md`。
