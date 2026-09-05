# 报告写作备忘（叙事主线、章间边界、禁令）

写论文过程中逐步定下来的**跨章约定**。只记「写作时容易忘、忘了会返工」的决定；
LaTeX 机制（编译、引擎、引用宏）见 `report/README.md`，不在此重复。
新定一条约定就追加一行，别散落在各章注释里。

---

## 1. 全文叙事主线（导师定，2026-08-19）

四段式，全文所有章节服从这条线：

```
① 条件化（conditioning）已被证明在多种场景下有助于 bias mitigation   → 给例子文献
② 实现条件化的一种方式是超网络                                      → 给超网络跨领域用例文献
③ 我们的实验结果
④ 没有任何方法给出「统计显著且一致」的结果 —— 与 MEDFAIR 结论一致
```

三条推论，写作时反复对照：

- **定位是「检验一个既有结论能否推广」，不是「我们试了个方法没用」。**
  ~~第 1 章开场从①起笔~~ **已改（用户 2026-09-01）**：第 1 章沿用 interim 的开场
  （子群差异 → 危害与规模 → 缓解三分类 → 为何超网络），①在 §2.2 承担；
  定位由**第 1 章末段**兑现（三框架 + 一致 null 的预告），不靠开场。
- **MEDFAIR 必须在第 2 章就出现**。否则第 6 章说「与 MEDFAIR 一致」会读成事后找靠山。
  ⚠️ **落点已改（2026-09-01）**：原计划的 §2.4 Benchmarks and negative findings **不设**，
  改为 **§2.2 末尾的「基准与其发现」段**（十一算法无一显著优于 ERM + SWAD），详见下方「第 1/2 章结构」。
- **导师主线①的落点同样改了**：不在第 1 章开场，而在 **§2.2 的条件化潜力段**（用户 2026-09-01 定）。
- **④ 的两个关键词是 significant 与 consistent**，分别对应配对 bootstrap 的 p 值、
  与跨 5 库 / 双向 OOD 的方向一致性。§5.1 主表按这两列组织，§6.1 按这两条收口。

**机制分析与方法学发现不在导师主线内，一律降级保留**（不删）：机制分析 → §5.6 一节 + 附录（2026-08-30 重组后的编号）；
方法学发现 → §6.2 一小节或附录。机制分析是回答「是不是实现错了」的唯一材料，不可删。

---

## 2. 章间边界规则

> **第 3 章 Methods = 「是什么」**：模型、估计量、指标的定义。与数据集、折数、统计口径无关，读完能独立实现。
> **第 4 章 Experimental Setup = 「怎么用」**：在哪些库、哪个 regime、什么统计判据下跑这些东西。

**操作判据**（有歧义时按这条切）：

- 凡带下标 **fold / seed / bootstrap / Holm** 的 → 第 4 章
- 凡是**一个公式或一个网络结构** → 第 3 章

由此推出的几处归属（已定，勿再挪）：

| 内容 | 归属 | 另一章怎么处理 |
|---|---|---|
| ρ / ρ_between 公式 | §3.5 | §4.4.6 只留「在此执行 + go/no-go 判定」，公式 `\cref` 回指 |
| knockout 边缘化式 | §3.5 | §4.3.3 保留设计动机与功效论证，式子回指 |
| worst-group 定义式（先折均再取 min） | §3.4 | §4.4.2 表格那行改为回指 |
| marginal / canonical **划分算子** | §3.1 | 各库**具体**几个子群 → §4.3 |
| 超参网格、CV 折数、选择信号 | §4.4.1/4.4.2 | 第 3 章完全不谈选择 |

**第 3 章不出现任何数据集名称**。举例、样本量、子群数一律留给第 4 章。

第 3 章五节：3.1 问题设定与记号 / 3.2 超网络变体（含 3.2.5 ResNet18CondNet）/ 3.3 基线 /
3.4 公平性估计量 / 3.5 机制诊断。

### 第 4 章结构（2026-08-28 定稿，导师口径；勿再改动编号）

**定位**：全章只讲**三个实验框架**与它们共用的数据/方法；工程细节下沉附录，结果数值留第 5 章
（但**事前量**——预注册 MDE、0.005 效应分级、σ_train 口径——属设计，留第 4 章）。
**每个框架一张流程示意图，围绕图讲流程，再讲该框架自己的 evaluation 与统计检验；
一体的内容不拆块**（这是返工两次才定下的：曾把「共用协议」抽成 §4.4 五个小节，导致
读者要在四处之间跳转，已废弃）。

```
4.1 Datasets            ID 与 OOD 共用，故单独成节：tab:datasets + 逐库角色 + 排除/子群 + 图像一段
4.2 Methods Compared    ID 与 OOD 共用。**写作逻辑（2026-08-28 导师二次收紧）= 只做两件事**：
                        ① 结合 tab:methods 说明「被比较的对象有哪些、各跑在哪个框架」；
                        ② 交代第 3 章没给的**必要参数**。
                        🛑 **不写「为什么纳入某基线 / 某 HN」**——第 3 章 §3.2 开头（三基线各自角色）
                        与 §3.4.1（三 HN 五轴同变、排序不能归因深度）已述，重复即删。
                        结构 = 一段导语 + tab:methods（8 臂，按 baselines / HN / HyperAdapt 扩展分三块）
                        + 一段「Settings」（SWAD 谷窗 N_s=3/N_e=6/r=1.3 与 BN 取算术平均、
                        GroupDRO 步长取论文默认且按 canonical 格、HASWAD 复用对应 HyperAdapt 的 config、
                        HApred 的 probe 与两个对照臂 + 跨库 g 随模型迁移）。全节约 1 页（含表）。
                        ⚠️ 连带两处搬家：**Δ_int 式移到 §4.3.2 估计量段**（估计量归口一处）；
                        **implementation checks 段移到附录 B「Implementation Checks」并保留
                        `sec:validity` 这个 label**（第 3 章 §3.3.3 与第 6 章仍 \cref 它）。
4.3 Framework 1 · ID    fig:pipeline-id + 流程 + tab:training-config
                        → 4.3.2 Evaluation(sec:estimands) → 4.3.3 Statistical Inference(sec:stats, tab:mde)
4.4 Framework 2 · OOD   **两小节（2026-08-28 导师二次收紧）**：
                        4.4.1 Pipeline —— fig:pipeline-ood + 与 4.3 的三点差异（3 seed/15 replicate、
                            full-target、σ_train）+ **点明 tab:ood-hypotheses 的两条假设与接受条件**；
                        4.4.2 Evaluation and Statistical Inference —— **即使沿用 §4.3.2 的口径也要
                            把评价指标写出来**（逐 replicate 在完整 target 上算 Overall / marginal worst /
                            canonical worst / gap，marginal worst 为主，再对 15 个 replicate 取平均）
                            + σ_train 的完整推导 + 假设表 + 确认性家族（3 HN × 3 baseline = 每方向 9 对比，Holm）。
                            σ_train 写法（2026-08-28 导师要求讲明白并给文献）：
                            ① 先说 bootstrap 答不了什么（重采样病人时模型是固定的）⇒ 三个 seed 的用途，
                               引 Bouthillier et al. MLSys 2021（ML benchmark 必须计训练方差）；
                            ② 配对差 d_{f,s} 同折同 seed 同目标样本 ⇒ 只剩训练带来的波动；
                            ③ 交叉双因子随机效应模型 + ANOVA 矩估计式（σ²_fold=(MS_f−MS_e)/S 等），
                               负分量截断为 0，引 Searle/Casella/McCulloch, *Variance Components* 1992；
                            ④ **σ_train 只取 seed 与残差、剔除 fold 分量**（fold 反映给了哪份数据、非运行随机性）；
                            ⑤ 每格 1 观测 ⇒ 交互并进残差 ⇒ σ_train 偏高 ⇒ H2 是保守判据；
                            ⑥ 判据 |d̄| > σ_train，另报比值 d̄/σ_train。
                        🛑 **不写预注册假设**（导师，2026-08-28 第三次收紧：三个框架里只有 §4.4
                        还留着假设叙事，不一致）。H1/H2 与 tab:ood-hypotheses **已删**，改为两个**判据**：
                        **interval condition**（paired cluster bootstrap 下显著）与
                        **floor condition**（超过 σ_train），**两个方向都满足才算 effect**。
                        更早删掉的还有 H3(校准)/H4(effective robustness)/H5(GroupDRO) 与偏离叙事——
                        GroupDRO 是 baseline 之一不该单列假设，且 OOD 现已训全部方法。
                        ⚠️ 第 5 章原用 H1/H2 指代这两条（11 处），已同步改成 interval / floor condition，
                        「confirmatory family」改为「the nine comparisons per direction」。
                        🛑 **Attribute knockout 与 Shift/Calibration 两节已从第 4 章删除**（导师，暂不写）。
                        ⚠️ 第 1/5/6 章仍有 knockout 与校准的 todo 残留，若确定全文不写需一并清。
4.5 Framework 3 · 消融  **两小节（2026-08-28 导师三次收紧）**：
                        4.5.1 Design —— fig:design-ablation + tab:ablation-cells + 共同 base state
                            + **本框架考察什么**（scope 与公平性的关系，由两个增量读出）
                            + 「Weight-level diagnostics」段（ρ/ρ_between 属设计，不是附加读数）
                            + 「Three further interventions」段（换 checkpoint / 通路 knockout / 关 fc 重训）；
                        4.5.2 Evaluation and Statistical Inference —— 继承所在框架的口径 +
                            两增量双侧 Holm(family=2) + ρ 的读法（>0 = 用了没用，≈0 = 从未启用）
                            + **三项干预各自怎么判**（换 checkpoint：两种选择规则下都成立才留；
                            knockout：看预测而非权重，残留占比相对属性置换的散布，落在散布内即判惰性；
                            关 fc 重训：同时看 conv 的 ρ_between 能否被驱动、与 val Overall AUC 的代价）。
                        🛑 **不写成「预注册假设」**：这里 H1/H2 是消融要考察的对象不是事前假设。
                        **两个增量已改名 $\dincdeep$ / $\dincfull$**（notation.tex 新增），
                        避免与 §4.4 的 H1/H2 撞名；式中方法名须包 `\text{}`，否则连字符被排成减号。
                        🛑 三项干预**不再单列 Mechanism Diagnostics 小节**——整个框架本身就是机制诊断。
```

**已明确排除（导师，2026-08-28）**：
- **frozen backbone 实验（实验 F）** 不进报告任何一章；
- **ROC 基线**全文删除（第 2/4/6 章的相关句已清；`\ROC` 宏保留但无引用）——
  它是操作点方法、组内 AUC 恒不变，在本文 AUC 口径下无信息；
- **PAPILA 全文删除**（第 4 章数据表/角色段/子群段/MDE 表、附录 transform 表均已清）。
  连带口径更新：**四库、两模态**；「三模态/四个数量级」等说法已改；batch 128 不再需要 PAPILA 例外；
  「小 canonical 格某折 AUC 无定义」的代价改为一般性表述（原以 PAPILA 举例）。

**HAM10000 = sex+age 两轴（2026-08-28 定；双轴 run 已跑完，2026-08-28 核实）**：
`outputs/ham10000/cv5/predictions` 下已有 `hyperhead_sexage` / `hyperfusion_sexage` /
`hyperadapt_sexage`，**6 config 全网格 × 5 折齐全**，`cv5/selected_configs.json` 也已选好三条
（分别 lr3e-04_wd1e-03 / lr1e-04_wd1e-03 / lr3e-05_wd1e-03）。旧的 age-only 结果
（`ResNet18HyperAdaptAge` 一族）**作废，第 5 章 HAM 的 HN 各臂一律用 `*_sexage`**。
报告统一按 **sex+age** 口径：tab:datasets 的 HAM 行 =
`sex, age (4 bins)`，marginal 6 组 / canonical 8 个 sex×age 格，与评估侧
（`subgroup_auc.py` 的 14 维向量）和 GroupDRO 的分组一致，全章无需特例说明。
> ⚠️ 聚合脚本（`build_oof_results_averaging.py` 的 METHODS）与第 5 章取数须指向 `*_sexage`，
> 别再读到旧的 age-only 键。

**OOD 覆盖全部 8 个臂（2026-08-28 核实）**：`outputs/ood_cxr/variance_decomposition_hn6/` 显示
**HyperHead / HyperFusion 也已按 v2 口径跑完双向 OOD**（6 臂 × 2 方向，per_cell = 5 fold × 3 trial
= 15 replicate，σ_train 判据齐全）。⇒ tab:methods 的 **Run in 列已删**（不再有区分度），
§4.4 里「HyperHead/HyperFusion 缺席 ⇒ OOD 不得下深度结论」这条**边界声明已删除**——
现在 OOD 侧同样有完整的深度阶梯。

**HyperAdapt-pred 已做 OOD（2026-08-28 核实）**：`docs/predicted_attribute_hyperadapt_results.md` §11
双向 MIMIC↔CheXpert、v2 口径（full-target、15 replicate、σ_train）。故 tab:methods 该行 Run in =
**ID, OOD**。要点：跨库 p̂ 由**源库训练的 g** 给出（g+HyperAdapt 整体迁移，目标库不参与任何拟合），
`const` 臂先验取 source train；**`perm` 臂只有 ID、OOD 未做**（OOD 六臂 = ERM/SWAD/HA-GT/pred-soft/
pred-const/GroupDRO）。第 4 章 §4.2 与 §4.4 已按此表述。
**融合臂 HASWAD 与预测属性臂 HApred 不单独成节**——它们只是进入 ID/OOD 框架的两个额外方法，
写在 §4.2「两个派生臂」段，特殊之处在对应框架内交代。

**跨章 label 契约**（第 3 章/附录已 \cref 引用，改名即断）：`sec:datasets`、`sec:methods-compared`、
`sec:estimands`、`sec:stats`、`sec:validity`、`sec:exp1-id`、`tab:methods`、`tab:datasets`、
`tab:training-config`、`tab:ablation-cells`。

**排版两个坑**（已踩）：① TikZ 里**不能**用 `cap` 当 style 名（与 `/tikz/cap` 保留键冲突，
xelatex 报 pgfkeys error 但 latexmk 只说「上次有错」，极易误判为编译通过——改用 `lbl`）；
② `\section` 标题过长会与 `Chapter 4. Experimental Setup` 页眉**重叠**（fancyhdr 奇数页同侧），
三个框架节均已用 `\section[短标题]{全标题}`。

### 第 1/2 章结构（2026-09-01 定，迁移自 interim report 后按后文订正）

**来源**：`docs/project_related/interim_report.pdf` 第 1/2 章逐段迁移，数字引用改 natbib 作者-年份制。
**订正原则（用户定）**：**凡与已完成实验冲突处，一律以后文为准**。

```
第 1 章 Introduction    无小节，沿用 interim 九段结构。
                        🛑 **不设 Contributions 与 Report Structure 两节**（用户 2026-09-01 定），
                        贡献与三框架预告压进末段。
                        🛑 **不按导师主线①改开场**（用户定）——①「条件化已被证明有用」改放 §2.2。
第 2 章 Literature Review   2.1 Unfairness in AI for medical imaging
                            2.2 Approaches to improve fairness
                            2.3 Hypernetworks
                        🛑 **不设 §2.4**：MEDFAIR、SWAD、GroupDRO、HN 谱系全部并入 2.2/2.3。
```

**§2.2 承载四件事（顺序即行文顺序）**：① 三分类学综述（pre/in/post，原 interim 内容）；
② **GroupDRO 写在 fairness-aware objective 段**里作为该路线的实例，点明「训练用、推理不用」
这条与条件化模型的分界，并 `\cref{sec:baselines}` 回指；③ **条件化潜力段（导师主线①）**
= FairAdaBN + FairREAD + 指向 §2.3 的 CBN/FiLM，收在「决策规则可以合法地因子群而异」这个前提上；
④ **「基准与其发现」段** = MEDFAIR 十一算法无一显著优于 ERM + **只讲 SWAD**（用户定，GroupDRO 不在此重复）。
第 6 章末段「与 MEDFAIR 一致」的靠山至此建立。

**§2.3 新增 HN 谱系段**（置于 CBN/FiLM 段之后、"The principal value…" 之前）：
HyperFusion（表格临床变量 → 单个 block）对 HyperAdapt（患者属性 → selected layers 低秩偏移），
两者之差即第 3 章的**介入深度轴**，`\cref{ch:methods}` 回指。

**八处按后文订正的冲突（已改，勿回退）**：

| # | interim 原文 | 现在 | 权威 |
|---|---|---|---|
| 1 | 公平性 = 缩小 inter-group gap | worst-group（gap 降为描述量） | §3.1 eq:minimax / §3.4 |
| 2 | 报 sensitivity/specificity/PPV/NPV | 全部终点建在 AUC 上 | §3.4 / §6.2 |
| 3 | group fairness ≡ 正类率均等 | 拆 outcome / performance 两支，取后者 | §3.1「not outcomes but performance」 |
| 4 | 承诺以 calibration 作基线 | 后处理只综述不入对比（组内排序不变），基线改述为「不看属性的」与「训练时看属性的」 | ROC 已全文删除 / §4.2 |
| 5 | 「属性与任务无关」为前提 | 删除该前提 | §2.2 收口 / §6.1 |
| 6 | 「在 MEDFAIR 的 benchmark 上做实验」 | 只沿用其目标/属性定义/age 分箱，管线自建 CV-OOF | §4.1 |
| 7 | 第 1 章无结论预告 | 末段加「no variant improves worst-group both significantly and consistently」 | tab:verdict |
| 8 | 第 3 章 ε「取 §4.3.3 的效应量阈值」 | **删掉该阈值叙事**（effect grading 早已删除，属悬空引用） | §4.3.3 |

**行文规则已回溯适用于第 1/2 章**（用户 2026-09-01 同意）：分号/冒号/破折号/引号与 `deliberately`
在 1/2 章已清零，数值区间改 `60 to 80` / `2016 to 2024` / `5 to 10 per cent`，
`human--AI` → `human-AI`、`column--row` → `column-row`。
⚠️ **`08_appendix.tex` 里还有一处 `deliberately` 未清**（用户明示暂不动）。

**篇幅**：第 1 章 1084 词 / 3 页，第 2 章 3339 词 / 8 页，全文 67 页。

**顺带修的一条引用**：`dwork2012fairness` 原按 interim 引 CoRR 2011，正文写「in 2012」，
apalike 会渲染成 (Dwork et al., 2011)。已改为 ITCS 2012 条目。

### 工程细节一律下沉附录 B（`app:implementation`）

判据：**「不影响某个读数怎么解释」的实现事项 → 附录**；影响解释的留正文。已下沉的有：
逐库 transform 表（`app:preprocessing`）、16-bit 读图实现与实测代价、统一 harness 与等-batch 的
显存混淆由来（`app:harness`）、SWAD 窗口超参 N_s=3/N_e=6/r=1.3 与 BN 两种处理
（`app:swad-details`）、子群 probe 规格（`app:predattr-probe`）、算力（`app:compute`）、
种子与重跑地板实测（`app:reproducibility`）。

### 第 5 章结构（2026-08-31 重排定稿）

**两条硬约束**（导师）：① **一个结论配一个浮动体**——摆出来却不推进任何结论的数据一律下沉附录；
② **正文自足**——正文出现的数必须在正文的表或图里查得到。由 17 个浮动体压到 **12（8 表 + 4 图）**，
正文 3810 → **2675 词**，章长 13 → **11 页**。

```
导语        §5.1「Summary of Findings」已并入章导语，只留 tab:verdict（主线④一页答案）
5.1 In-Distribution      fig:id-delta（两行：Overall / worst-group 的 Δ vs ERM + CI）
                         tab:id-levels（6 臂 × 4 库 × 两终点）
5.2 Out-of-Distribution  tab:ood-main（含迁移落差块）
                         fig:ood-ratio（**新**：2 面板 × 9 对比，画 r=|d̄|/σ_train，r=1 竖线 + 柱端数值）
5.3 Where the Attribute Is Injected
                         tab:ablation（4 cell + 关 head 重训两行 + 两增量）
                         fig:rho（**加第三面板**：conv ρ_between 在三种口径下）
                         tab:knockout
5.4 Extensions           tab:interaction（砍 naive 列）、tab:predattr
5.5 What the Conditioning Does
                         tab:named-rho、fig:hyperplane（**新**：§3.5.2 判别基，四库各一格）
5.6 Summary              **新增**，无浮动体，约 180 词
```

**删除的小节**：§5.2.3（canonical / gap，其结论就是「与主终点一致、无独立发现」）。
**下沉附录 A**：canonical/gap 全表、id-family 配对 CI、ood-family 逐格 Δ、interventions 细表、
trajectory、OOD 几何（类间分离度 / d∥ / Spearman）、超平面的 PCA 对照基与逐子群 panel、
关 fc 重训的逐 epoch 曲线。

**§5.6 Summary 的边界**（导师）：只做「总判决 + 两条独立支撑腿」，
🛑 **不写「没有确立什么」**——那属于第 6 章讨论，写进第 5 章会抢话。

**图表位置**（导师要求与所属小节对齐）。默认 `[t]` 会让浮动体整体排队后移一到两节，两处修：
① `includes.tex` 放宽浮动参数（topnumber=3 / topfraction=0.9 / textfraction=0.07 等，全文生效）；
② 第 5 章全部改 `[!htbp]`，并把 4 个「本节最后一个浮动体」在**源码中提前声明**
（tab:id-levels、fig:ood-ratio、tab:knockout、fig:hyperplane、tab:predattr）。
结果：12 个浮动体全部落在其所属小节的页面上，或与其引用段同页。

**§5.6 的超平面图（fig:hyperplane）**：
- 用 §3.5.2 的 $e_1/e_2$ 判别基出图，**闭合第三章**——此前那两个式子定义了却从未在正文兑现。
- 每格标注**最小组间余弦**与**反事实 AUC 跨度**，把「转了多少」与「后果是零」并排给出。
- 🛑 **HAM 必须用 sex+age 重跑**：现成 `viz_hyperadapt_hyperplane_ham.py` 硬编码
  `ResNet18HyperAdaptAge`（4 组）。新增 `scripts/viz_hyperplane_ham_sexage.py`（8 个 sex×age 组，
  同 schema 落盘，**不动原脚本**，age-only 历史产物仍可复现）。
  新读数：最小余弦 **0.646（49.7°）**、反事实 AUC 跨度 **0.0155**（age-only 版为 0.613 / 0.0099）。
- 其余三库直接复用既有 `hyperplane_arrays_*.npz`，无需重跑。
- 数据为 **fold0 / seed42 单点**，跨 fold/seed 未复核；按导师意见**只在图题里交代取自哪一折一个种子**，
  不写成大段告诫。

**两个必须保留的 label**：`sec:results-main`（第 3 章 §3.2.5 引）、`sec:results-mechanism`（附录 B 引）。

**5.4 与 5.6 的切分按「归属」不按「机制 vs 终点」（2026-08-30 修订，初稿切错）**：
- 框架 3 **自有的**诊断（逐 cell ρ + 三项干预）必须留在 5.4——第 4 章 §4.5.2 明文要求
  「平增量须与该位置的 ρ_between 就地合看」，推后到 5.6 会使 5.4 无法解读；
- 5.6 只放**跨框架跨库**的部分：三个命名 HN 在四库 + 双向 OOD 上的判别面几何，
  它支撑的是**全章** null 而非范围轴 null；且附录 `sec:validity` 已承诺
  「ρ 对 every trained conditioned model 计算，值在 `sec:results-mechanism` 汇报」；
- ρ 出现两次但对象不同：5.4.2 按**位置**读（新加的位置有没有被启用）、
  5.6.1 按**模型**读（部署的模型到底条件化没有）。估计量只在 §3.5.1 定义一次，两处都不重讲。

**下沉附录 A（`app:additional-results`）**：6 配置搜索全表与选中配置、四终点 × 8 臂 × 4 库完整数值、
逐子群 AUC、逐折散点、OOD 逐 replicate 值与方差分量表、逐层 ρ/ρ_between 完整曲线、
knockout 逐子群残留占比。

**排版坑（已踩）**：`\todo{}` 的中文参数里 `\CFull\（` 会把 `\（` 解析成未定义控制序列
（CJK 标点在 XeTeX 下是 letter catcode），须写 `\CFull{}（`；另 emoji 的 U+FE0F 变体选择符
在 Noto Serif CJK 里缺字，正文里的 ⚠ 不要带 FE0F。

---

---

## 2b. 行文规则（第 4 章起全文适用，2026-08-28 导师定）

- 🛑 **不用分号、不用冒号、不用破折号、不用引号**（含 ``…''）。用句号断句，或用
  because / since / thus / whereas 连接。**第 5 章起加严到冒号与破折号**（导师，2026-08-30）；
  数值区间写 `0.36 to 0.40` 而非 `0.36--0.40`。定稿前用下面这段脚本自查（`\eqref{eq:...}` 是唯一豁免）。
- 🛑 **不用 deliberately / on purpose 这类「强调我们是故意这么做」的措辞**，直接给理由。
- **不要花篇幅讲「被避免的做法有多糟」**：只说采用某做法的理由。
  典型返工：§4.3.2 曾用整段讲 pooling 的后果（跨折配对、校准漂移、逐方法污染），
  已压成「逐折算再平均是 CV-AUC 的推荐聚合方式（Airola/Parker/Forman），因为每折是不同模型」。
- **统计节的固定顺序**：先点名方法（paired cluster percentile bootstrap）→ 再讲操作流程
  （重采样单位、B=1000、全臂同一抽样、逐折重算再折间平均、得 Δ 的自举分布）→ 最后给拒绝准则
  （95% percentile CI 不含 0 即在 5% 水平拒绝 Δ=0，等价的双侧 bootstrap p 由中心化分布尾部质量给出，
  事前声明的确认性家族内先做 Holm）。§4.3.3 到此为止，**不再往下写**。
- 🛑 **不写效应分级（|Δ|≥0.005）与 MDE/功效声明**（导师，2026-08-28）：
  ① CV-OOF 框架内**每个 config 只跑一次**，不存在「重跑噪声」，0.005 这条线没有框架内依据；
  ② 显著性检验本身已经在判结果站不站得住，**再对检验做功效检验超出本项目的目标**。
  已删：§4.3.3 的 Effect grading 段、Power 段、tab:mde 整表、附录 B「The re-run floor」段、
  第 6 章讨论里「大库 MDE 声明」那一条。
  ⚠️ **σ_train 不在此列**——它是框架 2 用 5 fold × 3 trial **实测**出来的训练间波动，属实验设计，保留。
  ⚠️ 连带：写作备忘 §4「null 可信度三根支柱」的第 3 根（MDE 声明）作废，
  第 6 章收口只剩「四路径归零」与「ρ_between>0 证明通路启用」两根 + OOD 的 σ_train 判据。
- **tab:training-config 必须体现 GroupDRO 的损失**（BCE 按子群重加权，回指式 eq:groupdro-q），
  否则「所有方法共用同一 regime」会与第 3 章的 GroupDRO 定义冲突。

---

## 3. 叙事禁令

- 🛑 **不写信息闸门（conditional 𝒱-information / 可用信息 / control task）** —— 导师不建议。
  §2.4 旧闸门节已删；`docs/conditional_v_information_gate.md` 只作内部记录，不进报告任何一章。
  相关论证（「为何 null 不是实现失败」）改由**机制诊断**承担：ρ_between > 0 + knockout 反事实 + 超平面惰性。
- 🛑 **不写 R1 合成剂量-反应实验** —— 其标定用 nats、与闸门同源，2026-08-19 定为不进正文。
- 🛑 **不写 Minimax Pareto** —— 那是早期 5-seed pilot 的选择器，正式 CV-OOF regime 用的是
  `scripts/select_config_cv.py` 的**折均 marginal-worst 纯 argmax**。写 Pareto 会与 §4.4 自相矛盾。
- 🛑 **不定义 Equalized Odds** —— 全文终点均基于排序，不报阈值相关指标。若要提只写一句「故不采用」。
- ⚠️ **pooled OOF AUC 一律不作为读数** —— 主口径是 averaging（逐折算再折间平均）。
  旧文档里的 pooled 数字是负偏伪影，引用前先确认来源是 averaging 版。

---

## 4. null 的可信度靠什么撑（写第 6 章时对号入座）

三条（2026-08-28 修订：原第 3 条 MDE 声明已随功效叙事一并删除，改为 OOD 的 σ_train 判据）：

1. 三条互相独立的属性消费路径（GroupDRO / 三档 HN / CondNet 范围轴，另加 HApred 的零信息对照臂）全部归零；
2. 条件化通路确有启用（ρ_between > 0），故不是实现失败；
3. OOD 侧效应须同时越过 bootstrap 与 **σ_train**（5 fold × 3 trial 实测的训练间波动），
   而无一越过 ⇒ 那里的 null 不是抽样噪声造成的。
   🛑 ~~大库 worst-group 有功效（MDE 声明）~~ **已作废，勿再引用**。

---

## 5. 待办

- [x] ~~**文献欠缺是当前最大缺口**~~ **已补齐（2026-09-01，随第 1/2 章迁移）**：`refs.bib` 由 23 → 73 条。
      ① 条件化用于 bias mitigation：FairAdaBN `xu2023fairadabn` / FairREAD `gao2025fairread` /
      Petersen `petersen2024demographically`（均在 §2.2）；② 超网络跨领域用例：HyperSpace
      `joutard2024hyperspace` / Dynamic Filter `debrabandere2016dynamicfilter` / 残差 adapter
      `rebuffi2017adapters` / HyperFormer `mahabadi2021hyperformer` / CBN `devries2017cbn` /
      FiLM `perez2018film`（均在 §2.3）；MEDFAIR `zong2023medfair` 已在 §2.2 单列一段。
- [x] ~~核实 HyperFusion 出处~~ **已定（2026-08-19）**：HyperFusion = **Duenias, Nichyporuk,
      Arbel, Riklin Raviv, *Medical Image Analysis* 102:103503, 2025**（已入 `refs.bib`）。
      原「Ortiz et al., ICLR 2024」是两篇文献被合并的错误，代码 docstring 已订正。
- [ ] **MIP 的出处仍待确认**（改用 MIP 的**原因已定**：HyperFusion 原文的初始化方案
      **无法用于把 downsample 部分超网络化**，故本项目改用 MIP 加性重写——这是有意的修改，
      不是实现偏差，§3.2.3 已按此写；仍缺一句「为何不适用」的确切机制）：HyperFusion 原文用的是 Chang et al. 2019 方差对齐初始化，
      **不含 MIP、也无 E_L2 投影**；我们实现的 MIP 应来自 Ortiz et al. ICLR 2024 关于
      hypernetwork magnitude-invariant parametrization 的工作，但「Modulated Initialisation
      Procedure」这个展开式大概率是错的。确认后补 `refs.bib` 并改 §3.2.3 与代码 docstring。
- [x] ~~核实 HyperAdapt 的「Xu et al. 2026」完整出处~~ **已核实（2026-09-01，arXiv 页面回源）**：
      **Gelei Xu, Yuying Duan, Jun Xia, Ruining Deng, Wei Jin, Yiyu Shi, *Patient-Conditioned Adaptive
      Offsets for Reliable Diagnosis across Subgroups*, arXiv:2601.13094 (cs.CV, 2026-01-19)**，
      DOI 10.48550/arXiv.2601.13094，**仅预印本、无会议/期刊**。要点三条：
      ① **方法名「HyperAdapt」是原文自己的命名**，本文沿用无需另起名；
      ② 原文口径是「generates small residual modulation parameters for **selected layers**
      of a shared backbone」，低秩 + bottleneck 参数化——故 §2.3 写 selected layers，
      「除 stem 外所有层」是**本项目实现的选择**，不是原文口径，勿混写；
      ③ 原文动机与本文 §2.2 收口一致（属性在医学场景常携带诊断信息，抑制它会掉精度）。
      `refs.bib` 的 `xu2026patientoffsets` 已按此更正（此前作者名是凭 interim 参考文献表猜的，六位全错）。
      `drafts/ResNet18_hyperadapt.py` 的 docstring 只写「Xu et al., 2026, arXiv:2601.13094」，与核实结果相容，无须改。
- [ ] `refs.bib` 中 `xu2020theory` / `hewitt2021conditional` 两条已随闸门作废，定稿前删。
- [x] ~~软属性 / 预测属性 HyperAdapt：倾向降至附录~~ **改判（2026-08-28）**：作为额外方法臂进 §4.2，理由见 §2 第 4 章结构节。
- [ ] DeepView 决策边界图：倾向不进报告（超平面可视化已覆盖同一论点）。

---

## 5b. 第 5 章取数与图（2026-08-30 填充完成）

**取数唯一权威**（写作时不得从旧 md 抄数，一律回源）：

| 节 | 数据源 |
|---|---|
| §5.2 ID | `outputs/conditioning_ablation/oof_results_averaging.json`（averaging；HAM 的 HN 取 `*_sexage` 键） |
| §5.3 OOD | `outputs/ood_cxr/variance_decomposition_hn6/variance_decomposition_{marginal_worst,overall}.json` |
| §5.4 消融 | `docs/conditioning_ablation_summary.md` + `conditioning_ablation_e1_{worstcase_checkpoint,pathway_knockout}.md` + `e3_fcoff_training.md` |
| §5.5 派生臂 | `docs/hyperadapt_swad_fusion_full_report.md` §4.3、`docs/predicted_attribute_hyperadapt_results.md` §4.1 |
| §5.6 机制 | `docs/hyperplane_subgroup_visualisation.md` §5.1、`docs/ood_hyperplane_visualisation.md` §5.1、`outputs/analysis/hyperadapt_trajectory/{ham,mimic}_fold0/scalars.json` |

**三张图**由 `scripts/plot_report_ch5_figures.py` 生成到 `report/figures/`（改数据后须重跑）：
`ch5_id_worst.pdf`（四库 Δworst 条形 + 95% CI，两个大库用 5× 放大刻度并写进栏标题）、
`ch5_ood_noise.pdf`（|d̄| vs σ_train 并排条形，双方向）、
`ch5_rho.pdf`（stage 级 ρ_between，**对数纵轴**，否则 conv 被压成零线）。

**⚠️ 写作时实际纠正过的六处口径错误**（都是「凭印象概括」出的，回源后才发现）：
① HN 区间宽度是 0.08–0.12 不是 0.05–0.14；② canonical 口径下 HAM 的 HyperAdapt **低于** ERM
（marginal 口径是高于），三变体排序不跨口径保留；③ 胸片 canonical 最大差 0.0061 ⇒ 写「within 0.007」；
④ OOD 臂间跨度 M2C 0.010 / C2M 0.006 ⇒ 写「at most 0.010」；⑤ OOD 六个 HN 比较里显著的是 **5** 个
不是 4 个（其中 2 个是改善，且同在 M→C 方向）；⑥ σ_train 是效应的 **2–9** 倍不是 2–6 倍。

**OOD 确认性家族已补齐（2026-08-30）**：第 4 章 §4.4 预注册「3 HN × 3 baseline = 每方向 9 对比」，
但 `ood_variance_decomposition.py` 原本只算 vs ERM。已做两处改动：
① 该脚本新增 `boot` 落盘（逐方法的 bootstrap 分布，全方法共用同一批重采样索引 ⇒
**任意一对**方法的配对差 = `B[m1] − B[m2]`，无须重跑 bootstrap）；
② 新增 `scripts/ood_confirmatory_family.py`，由 `boot` + `per_cell` 导出 18 个对比的
Δ / CI / Holm（**在各自方向的 9 个检验内校正**）/ σ_train / r=|d̄|/σ_train。
产物 `outputs/ood_cxr/variance_decomposition_family/{mw,ov}/` + `family_readout.txt`。
vs-ERM 数值与既有 `variance_decomposition_hn6/` 逐位一致（同 BOOT_SEED），可作自检。

> 🛑 **该家族改写了一条结论**。初稿曾写「no hypernetwork clears H2 in either direction, and the
> conclusion does not depend on which baseline」——**后半句是错的**：marginal worst 上
> **18 个对比里有 2 个越过 σ_train，全部是 M→C 方向的 HN vs GroupDRO**
> （HyperHead r=1.07、HyperAdapt r=1.43），但两者在 C→M 塌到 r=0.57 / 0.45。
> 正确表述 = 「因为接受要求两方向同时满足 H1 与 H2，家族内无一被接受」，
> 而不是「无一越过 σ_train」。§5.3.2 已按此重写并加 `tab:ood-family`。

## 5c. 🛑 正文自足原则（导师，2026-08-30）

> **正文里出现的数，必须在正文的表或图里能查到。附录只作补充，不作出处。**
> 允许**整类指标不报**（如校准），但**不允许报了却查无出处**。参照 MEDFAIR 的汇报方式
> （逐库逐方法给多终点表，不在正文里丢下无表可依的配对比较数）。

初稿违反此原则的六处及处理（第 5 章已整改）：

| 处 | 无出处的数 | 处理 |
|---|---|---|
| §5.2.1/5.2.2 | Overall 的 vs-ERM 显著性 | `ch5_id_worst.pdf` **改为两行**（上 Overall / 下 worst-group） |
| §5.2.2 末段 | vs SWAD、vs GroupDRO 的 Δ 与显著性 | 新增 **tab:id-family**（3 HN × 3 baseline × 4 库） |
| §5.2.3 | canonical worst、gap | **tab:id-main 改 MEDFAIR 式**：转置成逐库分块、四终点（Ov / WG marg / WG canon / Gap） |
| §5.4.3 | 两种 checkpoint 下的 fc 与 conv ρ、关 fc 重训后的表现 | 新增 **tab:interventions**（逐通路 × 三种口径 + 重训后两终点） |
| §5.5.1 | 「SWAD 增益 +0.058 传到 +0.045」 | **tab:interaction 补两侧增量列** |
| §5.6.2 | 类间分离度压缩、d∥、两个 Spearman | 新增 **tab:ood-geometry**（2 方向 × 3 方法） |

**仍允许留在正文而无独立表格的三类**（均可由章内图表推出或属单一汇总量）：
① 可从图上读出的量（如 CI 宽度 0.08–0.12）；② 表内两数之比（如 1.0–1.4×、0.29 SD）；
③ 一族检验的单一汇总量（如「Holm 后最小 p=0.38」「18 个里 17 个显著」）。

**§5.6.1 的 ρ 缺口已补齐（2026-08-30）**。初稿写「ρ_between>0 across the four datasets and
all three variants」时，ρ **只对消融的 CondNet cell 在 HAM/MIMIC 上算过**，命名 HN 四库的 ρ
从未计算，附录 `sec:validity` 的「computed for every trained conditioned model」无产物支撑。
现已新增 **`scripts/named_hn_rho.py`** 补跑（**12 组 × 5 折全部成功**，纯 CPU 约 3 分钟，
产物 `outputs/analysis/named_hn_rho.json`），`sec:validity` 的承诺现已兑现，**无须改附录**。

- **定义与 `ResNet18CondNet.conditioning_rho` 逐行一致**（K 个属性取值等权、先逐单位求 F-范数再平均）。
- 三变体的 ΔW 各不相同，分别实现：HyperAdapt = conv 乘性 `W⊙(A·B)` + fc 加性 `A_fc·B_fc`；
  HyperFusion = `layer4[0].downsample` 的**加性残差**，分母取 θ_0。
- 🛑 **HyperHead 的 ρ 无定义**：它**直接生成**整个 fc 权重而非在 base 上加偏移，没有可作分母的 W。
  故只报 ρ_between，分母改用**跨属性平均权重**的范数。第 5 章表题已写明此事，
  **ρ_between 可与另两变体并列读，ρ 不可**。

两条新读数（均已写入 §5.6.1，配 `tab:named-rho`）：
1. **HyperFusion 是 ρ_between 必须与 ρ 分开报的实证**：单一注入点被改动 0.59–2.17 个基权重范数，
   但其中只有 **17–25%** 随属性变化 ⇒ 大部分是**所有子群共享的静态重参数化**，
   只看 ρ 会把条件化强度高估约 5 倍。
2. **HyperAdapt 的 fc 主导在消融之外复现**：四库 fc/conv = **98.2× / 13.1× / 5.7× / 6.8×**，
   且跨库次序与消融一致（HAM 最集中）⇒ 受控消融与部署模型对「条件化去了哪里」给出同一答案。

---

## 5d. Holm 校正的适用范围（2026-08-31 定）

第 4 章 §4.3.3 的规则「一个框架把一组臂与一组参照相比时，p 值在该集合内做 Holm 校正」
**§4.3.3 本身就在框架 1 之内，故框架 1 无须再单独声明集合**（导师）。逐实验判断：

| 实验 | 是否校正 | 理由 |
|---|---|---|
| 框架 1（ID）主终点 | **校正，且是全文唯一承重处** | 12 个 HN-vs-ERM 里有 3 个「显著」、最大 +0.0029、且一个符号相反 = 选择性噪声的典型形态。不校正等于报告三个自己都不信的阳性 |
| 框架 2（OOD） | 校正，**不承重** | 18 个里 17 个 Holm 后仍显著；真正判决的是 floor condition |
| 框架 3 两增量 | 可做，**不承重** | 两个增量是各自预设的独立问题、不是找赢家；p=0.84 / 0.004 离阈值都很远 |
| HApred 20 比较 | 可做，**不承重** | 最小 p=0.38。⚠️ 该臂核心是**等价性**声明，不能靠「校正后不显著」支撑，靠的是效应量与区间宽度 |
| HASWAD 的 Δ_int | **不并入 family** | 零假设本就是可加性；唯一显著值是负的（与超可加相反），已按效应量论证。校正只会让零结果更易成立 |
| 次要终点（Overall / canonical / gap） | **不并入主 family** | 主终点已声明为 marginal worst-group；塞进同一 family 只会稀释主检验 |
| 机制诊断 | 不涉及 | 无假设检验；knockout 的 σ 是置换参照量不是 p 值 |

**family 范围 = 每库 11 个**（5 臂 vs ERM + 3 HN vs SWAD + 3 HN vs GroupDRO），即本框架实际报告并
解释的比较。**已验证对划法不敏感**：family=5 / 9 / 11 三种定义下，HN-vs-ERM 的判决完全一致（全为 n.s.）。
用 Holm（FWER）而非 BH（FDR），因为本文阳性极少且每个都单独承重，FWER 让任何阳性最难成立。

### 执行结果与连带改动

`build_oof_results_averaging.py` 已加 **精确两侧 percentile bootstrap p**
（中心化到观测 Δ，用 `(1+count)/(B+1)` 避免 p=0 破坏 Holm 排序）。重跑 B=1000 后
**点估计 / CI / sig 与旧版逐位一致**，只多出 `p` 字段。判决由 `scripts/id_holm_correction.py`
产出到 `outputs/analysis/id_holm_verdicts.json`。

主终点上 Holm 后的改判（**均以精确 p 为准，非正态近似**）：
- **HN vs ERM 由「3 个显著」变为 0 个**（+0.0029 / −0.0040 CheXpert、+0.0014 MIMIC 全部退出）；
- **SWAD−ERM 在两个皮肤库（+0.046 / +0.058）也退为不显著** —— 这条常被忽略，正文原有的
  「SWAD 是唯一清出自身区间的臂」已据此改写；
- Fitzpatrick 的 HyperFusion−SWAD、MIMIC 的 HyperFusion−SWAD 亦退出；
- **仍显著者**：两个胸片库上 GroupDRO−ERM 为负、三个 HN 显著高于 GroupDRO、
  以及 CheXpert 的 HyperFusion−SWAD（+0.0052，全框架唯一「HN 显著胜非-GroupDRO 基线」）。

`fig:id-delta` 的 worst-group 行改按 Holm 着色、Overall 行保持未校正，**图题写明两行口径不同**。

> 🛑 **必须在正文承认的风险**：这里校正的方向与通常相反——它删掉的是本章仅有的阳性。
> 防守点不是「校正后不显著」，而是：① 规则第 4 章事前声明、三个框架统一适用；
> ② 三个翻转的读数是 +0.0029 / +0.0014 / −0.0040，**校正与否都不构成实践意义的改善**；
> ③ **本章的零结果本就不靠不显著支撑**，靠的是效应量 ≤0.004、符号跨库不一致、迁移侧的
> σ_train 地板、零信息对照臂四条，**全部与校正无关**。

---

---

## 6. 定稿前检查

```bash
grep -rn "todo{" sections/ main.tex                            # 批注清空
grep -rni "pareto\|equalized odds\|V-information\|usable info" sections/   # 禁令残留
```
