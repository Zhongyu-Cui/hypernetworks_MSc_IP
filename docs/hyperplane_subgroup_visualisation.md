# 逐子群分类超平面可视化 —— 权威规格

> **定位**：HyperAdapt 在不同敏感组条件下**如何调整分类超平面**的测量与作图规格。
> 回答的是机制问题（HN 到底动了什么、动了多少、走哪条通路），**不是**性能/公平性结论的来源
> ——后者的权威口径见 `docs/comparison_protocol.md` 与 `docs/oof_regime_results.md`。
>
> **attribute-blind 基线（ERM / SWAD）的零对照版见 §9**——那里没有反事实扫掠、只有一条超平面，
> 本文档 §2–§5 的方法与判读**不适用**于它们。
>
> **实现**：`src/utils/hyperplane_viz.py`（共享数学与渲染）、`src/utils/pathway_audit_report.py`
> （三口径通路审计）、`src/utils/hyperplane_cxr_runner.py`（MIMIC/CheXpert 共享的 CXR 8 子群流程）、
> `scripts/viz_hyperadapt_hyperplane_ham.py`、`scripts/viz_hyperadapt_hyperplane_mimic.py`、
> `scripts/viz_hyperadapt_hyperplane_chexpert.py`、`scripts/viz_hyperadapt_hyperplane_fitzpatrick.py`。
> **产物**：`outputs/{ham10000,mimic_cxr,chexpert_cxr,fitzpatrick}/hyperplane_viz/`。
> **状态**：2026-07-27 建成并完成三口径审计（HAM / MIMIC / CheXpert / Fitzpatrick）；结果均为
> **fold0 / seed42 单点**，跨 fold/seed 复核未做。

---

## 1. 概念与符号

HyperAdapt 的决策（`num_classes=1` 二分类，特征 512 维）：

```
logit(x, a) = < w_eff(a), f(x, a) > + b
```

| 符号 | 含义 | 代码位置 |
|---|---|---|
| `f(x, a)` | penultimate 特征（512 维），**随属性 a 变**（conv 被逐层调制） | `resnet18_hyperadapt.py:614` 的 `torch.flatten(out, 1)` |
| `w_eff(a)` | **有效超平面法向量** = `W + ΔW(a)`，`ΔW = fc_A_gen(emb) @ fc_B_gen(emb)`（加性低秩，rank=4） | `resnet18_hyperadapt.py:616-620` |
| `b` | `fc.bias`，**不随 a 变**（HyperAdapt 不条件化 bias） | 同上 |

`logit = 0` 的集合即**分类超平面**，`w_eff(a)` 是其法向量：方向决定超平面朝向，`‖w_eff‖` 与 `b`
决定其位置。故「HN 如何调整超平面」在数学上等价于「`w_eff(a)` 与 `f(·,a)` 如何随 a 变」。

> **每组只有一个 `w_eff`**：同一属性组内所有样本的 `patient_embed` 输出相同 ⇒ `ΔW` 相同。
> 虽然机制上是逐样本生成，实际每子群只对应一条确定的决策线（HAM 4 条、Fitzpatrick 6 条、MIMIC / CheXpert 各 8 条）。

---

## 2. 方法（三步）

### 2.1 反事实属性扫掠

固定同一批 held-out 图像 X，**逐个假设它属于每个子群各前向一次**，得到 `F_g = f(X, a_g)`
（`[G, N, 512]`）与 `w_g = w_eff(a_g)`（`[G, 512]`）。

**为什么必须反事实**：若改用各样本的真实属性，不同组的点来自不同病人，组间差异分不清是 HN 调制
造成的还是「不同人群的影像本来就不同」（混杂）。固定病人后，组间差异 **100% 归因于 HN 的调制**。

**为什么不能只比较 `w_eff(a)` 的夹角/范数**：那样会完全漏掉 conv 通路。HyperAdapt 的条件化有两条
通路（conv 逐层乘性调制改变特征、fc 加性低秩改变打分表），只看 fc 侧在 MIMIC 上会把模型误判成
「几乎没在调整」。

### 2.2 两套 2D 投影基（同一中心 `mu`）

`mu` = **所有组特征的总均值**（不是各组自身均值，否则各 panel 原点不同、坐标不可比）。

| 基 | 构造 | 语义 |
|---|---|---|
| **discriminative**（主口径） | `e1 = w_bar/‖w_bar‖`；`e2` = 在 **`e1` 的正交补内**对 `{w_g − w_bar}` 做 PCA 的第一主成分 | 横轴 = 诊断方向；纵轴 = 组间调整方向 |
| **pca**（对照） | 对全部特征做 PCA 取前 2 主成分 | 方差最大方向（**非**判别方向） |

> **`e2` 必须先投影到正交补、再做 PCA（deflation），不是「全空间 PCA1 再 Gram-Schmidt」。**
> 前者精确求解 `max_{e ⊥ e1, ‖e‖=1} Σ_g <w_g − w_bar, e>²`；后者只是把一个未受约束的解事后压进
> 正交补——当组间偏离的主方差大部分沿 `e1` 时 `pc1≈e1`，其垂直残量方向由噪声决定，可任意偏离补
> 空间内的真正主方向。
> 🛑 **2026-07-27 修正**：初版实现为后者。实测两法在本项目三个数据集上夹角仅 **1.1–1.6°**、
> `e2` 捕获方差差 **<0.05pp**（理论保证 deflation ≥ Gram-Schmidt，实测符号恒为正），故
> **历史图与结论均不受影响**；改用 deflation 是因其无额外成本且有最优性保证。
> **2026-07-28 补**：上述修正当时只改了实现，**图内 suptitle 仍印着旧的 "Gram-Schmidt
> orthogonalised against x"**（`hyperplane_viz.py` 的 `BASIS_EXPLAIN`）。现已改为
> "computed INSIDE the orthogonal complement of x (deflation, not post-hoc Gram-Schmidt)"，
> 并重跑 Fitzpatrick / CheXpert / MIMIC 三套图（HAM 用自己的 2×4 渲染、不含该句，无需重跑）。
> 三份 `hyperplane_metrics_*.json` 的 md5 与重跑前**逐位一致** ⇒ 纯文字修正，数值零变化。
>
> 同时修正了一个**误报**：初版 panel 内印的 `e2 captures X% of between-group var` 是**全空间**
> 比例，易被读成「e2 展示了 X% 的组间差异」；`e2` 作为纵轴只应对正交补负责。现改报三个量：
> `between_var_along_e1`（沿判别轴的**纯尺度**份额，图上表现为决策线平移而非转向）、
> `between_var_in_perp`、`e2_perp_var_ratio`（`e2` 占**正交补内**的比例），另加
> `e2_singular_value_ratio = s1/s2`（接近 1 时补空间主方向近乎简并、`e2` 方向不稳，读图须谨慎）。

**线性投影是硬要求**：`f = mu + P u` ⇒

```
logit(u) = < P^T w, u > + ( < w, mu > + b )
```

在 2D 上仍是**可解析绘制的直线**，无需 DeepView 的逆映射合成网格近似。t-SNE/UMAP 非线性且无逆映射
⇒ 超平面在其中既非直线也非可写出的曲线，**不能用来画边界**（其合法用途仅为给点云着色看分布）。

### 2.3 Δlogit 通路分解

对参照组 `a_ref` 做代数恒等变形（**非近似**，`b` 自动抵消）：

```
Δlogit = < w_g − w_ref, f(x, a_ref) >  +  < w_g, f(x, a_g) − f(x, a_ref) >
         \______ fc 头贡献 ______/        \______ conv 特征贡献 ______/
```

参照组取样本量最大的子群（HAM `age=1`；Fitzpatrick `skin=1`（型 II）；MIMIC 与 CheXpert 均按
**抽样前** split 规模取 `Male/White/≥60`）。

---

## 3. 五个方法论陷阱与处理

| # | 陷阱 | 处理 | 状态 |
|---|---|---|---|
| 1 | 只看 `w_eff` 会漏掉 conv 通路 | 反事实扫掠同时取 `F_g` 与 `w_g` | 已处理 |
| 2 | **规范自由度（gauge freedom）**：特征云与超平面同时移动，单看线的位移无物理含义 | 渲染**强制**点云与该组决策线同 panel；判读只看「点相对线的位置」 | 已处理 |
| 3 | **投影损失**：`w` 垂直于平面的分量不可见 ⇒「两组线重合」可能只是投影伪影 | 必报 **in-plane 能量** `‖Pᵀw‖/‖w‖`，及 `e2` 占组间方差比 / PCA 解释方差比 | 已处理 |
| 4 | **分解顺序依赖**：`Δlogit` 有两种精确分解，交互项 `I = <Δw, Δf>` 全归 conv（order-1）或全归 fc（order-2），占比读数因此路径依赖 | 三口径并报（order-1 / order-2 / **Shapley** 均分 I，三者均为精确分解），主口径用 Shapley | **2026-07-27 新增** |
| 5 | **跨口径混谈**：logit 幅度口径与权重范数口径、AUC 口径回答不同问题，可给出方向相反的答案 | §4 强制三口径并列 + 对账规则 | **2026-07-27 新增** |

---

## 4. 三口径通路归因与既有诊断的对账（**本文档最重要的一节**）

「哪条通路主导条件化」有**三个互不等价**的口径。混用会得出相反结论。

| 口径 | 测什么 | 出处 | HAM | MIMIC | CheXpert | Fitzpatrick |
|---|---|---|---|---|---|---|
| **A. 权重相对幅度** `ρ = E_a‖ΔW_l(a)‖_F/‖W_l‖_F`（已按每层自身范数归一化） | 参数被改动多少 | `docs/conditioning_ablation_summary.md` §4.1 | **fc 主导，fc/conv 45–60×** | **fc 主导，fc/conv ≈5×** | 未做 | 未做 |
| **B. logit 幅度** `mean\|贡献\|`（§2.3 分解） | 谁把 logit 推得更远 | 本文档 | **fc 66.9%**（Shapley 均值） | **conv 74.0%**（fc 26.0%） | **conv 72.5%**（fc 27.5%），但**逐子群跨度极大 7.9%–71.7%** | **conv 60.7%**（fc 39.3%），fc 占比**随肤色变深单调升** 14%→63% |
| **C. AUC（排序）** 只叠加一条通路后 AUC 变化 | 谁真的改了排序 | 本文档 + 消融 knockout | fc 略大（max\|ΔAUC\| fc 0.0058 / conv 0.0028）；消融：**fc 承担全部 age 依赖，conv 惰性** | **两条通路均 ≈0**（fc 0.0000 / conv 0.0005） | **两条通路均 ≈0**（fc 0.0000 / conv 0.0002） | **两条通路均 ≈0**（fc 0.0013 / conv 0.0015） |

### 4.1 🛑 已修正的错误陈述（2026-07-27）

初版汇报称本工作「独立复现了权重轨迹的 MIMIC conv 主导」。**该说法错误，已撤回。**

- 权重轨迹的 ρ **已做尺寸归一化**，归一化后**两个数据集都是 fc 主导**（旧的「conv 主导 / 随深度增」
  本身就是未归一化的尺寸伪影，见记忆 `hyperadapt-weight-trajectory-viz`）。
- 本工作的口径 B 在 MIMIC 上给出 **conv 74%**，与口径 A **方向相反**，不是吻合。
- **只有 HAM 是真的一致**（口径 A、B、C 都指向 fc）。

### 4.2 口径 A 与 B 方向相反不矛盾

`ρ` 归一化掉了参数量，但**没有归一化层数与功能放大**：

- conv：每层相对改动仅 0.5–1%，但有 **20 层乘性调制**、作用于整张特征图并逐层累积
  ⇒ 实测 MIMIC 特征相对位移 `‖Δf‖/‖f_ref‖` = 1.5–5.4%；
- fc：单层相对改动 2–4%，但只影响最后一次 512 维线性组合。

⇒ **「参数改动相对更小」与「对 logit 贡献更大」可以并存。** 口径 A 不能推出口径 B，反之亦然。

### 4.3 判读规则（强制）

1. **口径 C（AUC）是最终仲裁**。logit 平移**不改排序**，故口径 B 的「主导」不足以支持任何性能/公平性
   论断。MIMIC 上 fc/conv 对 AUC 均 ≈0 ⇒ **「谁主导」在 MIMIC 上是无意义的问题**，不得写成结论。
2. 口径 B 的占比**必须**标注是哪个分解顺序，并附 Shapley 值与 `I/|Δlogit|`。
3. 引用「fc 主导」时**必须**说明口径。与消融/轨迹对账时只能拿口径 A↔A、C↔C 比。

### 4.4 附带发现：方向对、幅度不足（MIMIC）

审计③显示 MIMIC 两条通路的贡献**本身都与标签相关**——该贡献单独作分数的 AUC：
fc-term 0.487–0.824、conv-term 0.763–0.801（多数远离 0.5）。但幅度（fc mean ±0.06、
conv mean −0.11…−0.02 logit）相对 logit 自身尺度太小 ⇒ 对最终排序无影响。

**解读：模型学到的调制方向大体正确，但幅度被优化压到无关紧要。** 与
`docs/conditional_v_information_gate.md`「MIMIC 全轴 CI<0 = 决定性 0」一致，也与消融
`conditioning_ablation_e3_fcoff_training.md`「激活 conv 反而降性能 ⇒ 性能目标自动把它压到最小」自洽。

---

## 5. 结果（fold0 / seed42）

### 5.1 超平面几何

| 量 | HAM (age, 4 组) | MIMIC (sex×race×age, 8 组) | CheXpert (sex×race×age, 8 组) | Fitzpatrick (skin, 6 组) |
|---|---|---|---|---|
| 组间最小余弦（最大夹角） | **0.613（≈52°）** | **0.998（≈3.6°）** | **0.988（≈9.0°）** | **0.954（≈17.5°）** |
| `‖w_eff‖` 范围 | 0.646 – 1.374（**2.1×**） | 0.507 – 0.513（几乎恒定） | 0.457 – 0.512（**1.12×**） | 0.537 – 0.566（**1.05×**，随肤色变深**单调升**） |
| mean logit 范围 | −6.97 → −3.29（**平移 3.7**） | −0.52 → −0.42（平移 0.1） | −3.50 → −3.02（平移 0.48） | −4.15 → −3.62（平移 0.54） |
| 特征相对位移 `‖Δf‖/‖f_ref‖` | 0.356 – 0.403 | 0.015 – 0.054（**小一个量级**） | 0.027 – 0.132 | 0.050 – 0.124 |
| 反事实 AUC 范围 | 0.8828 – 0.8927（跨度 0.010） | 0.8301 – 0.8310（跨度 0.001） | 0.8506 – 0.8509（跨度 **0.0003**） | 0.8946 – 0.8967（跨度 0.002） |
| `fc.bias` | −0.0116（全组共用） | −0.0132（全组共用） | −0.0262（全组共用） | −0.0198（全组共用） |

**HAM 大幅调整超平面、MIMIC 几乎不调整、CheXpert 居中但以「单子群离群」为主（§5.4）、
Fitzpatrick 居中且呈「沿肤色轴单调」（§5.5）**；但**四库的反事实 AUC 都几乎不变** ⇒ 调整全是沿判别
方向的平移/缩放（操作点/校准），**不改组内排序**。

### 5.2 投影基对照（in-plane 能量）

in-plane 能量 `‖Pᵀw‖/‖w‖`（数值为 2026-07-27 改用 deflation 版 `e2` 后重算，四库全部重跑）：

| | discriminative | pca | PCA 基解释方差 |
|---|---|---|---|
| HAM | **0.947 – 1.000** | 0.525 – 0.653 | PC1 15.0% + PC2 5.0% |
| MIMIC | **0.9997 – 0.9999** | **0.380 – 0.390** | PC1 78.6% + PC2 13.5% |
| CheXpert | **0.998 – 1.000** | 0.582 – 0.601 | PC1 77.8% + PC2 16.0% |
| Fitzpatrick | **0.992 – 0.999** | 0.620 – 0.645 | PC1 34.3% + PC2 14.3% |

**默认 PCA 系统性低估条件化幅度**：MIMIC 上它抓住 92% 的**特征方差**却只保留 38% 的**打分表**
（两个主成分与 `w` 近正交），图上各组决策线被压扁成近乎重合。**方差大 ≠ 判别相关。**

#### 组间方差的分解与 `e2` 的稳定性（判别基自身诊断）

| | 沿 `e1`（纯尺度） | 正交补 | `e2` 占正交补 | `s1/s2` |
|---|---|---|---|---|
| HAM | **24.2%** | 75.8% | **80.9%** | **2.70** |
| MIMIC | 2.1% | 97.9% | 41.6% | 1.43 |
| CheXpert | **24.8%** | 75.2% | **78.3%** | **2.22** |
| Fitzpatrick | 1.8% | 98.2% | 48.0% | 1.38 |

两点判读：

1. **「沿 e1」份额 = 纯尺度变化**（各组 `‖w_eff‖` 不同），在图上表现为决策线**平移而非转向**。
   HAM/CheXpert 各约 1/4 的组间差异属于此类；MIMIC/Fitzpatrick 几乎为 0（组间差异全在方向上，
   但绝对幅度本就极小）。
2. **`s1/s2` 是 `e2` 方向的稳定性指标**。HAM 2.70、CheXpert 2.22 ⇒ 补空间内有明确主方向，纵轴
   可解读；**MIMIC 1.43、Fitzpatrick 1.38 ⇒ 前两个主方向方差接近、`e2` 近乎简并，纵轴方向不稳定，
   不得对其做方向性解读**（这本身与两库「组间调整接近各向同性噪声、无突出结构」自洽）。

### 5.3 通路归因（三口径，见 §4）

| | fc 占比 order1 / order2 / Shapley | `I/\|Δlogit\|` | max\|ΔAUC\| fc / conv |
|---|---|---|---|
| HAM | 68.1% / 65.9% / **66.9%** | 0.19 – 0.24× | 0.0058 / 0.0028 |
| MIMIC | 25.7% / 26.3% / **26.0%** | 0.01 – 0.02× | **0.0000 / 0.0005** |
| CheXpert | 27.3% / 27.7% / **27.5%**（跨子群均值） | 0.00 – 0.12× | **0.0000 / 0.0002** |
| Fitzpatrick | 38.3% / 40.4% / **39.3%**（跨子群均值） | 0.02 – 0.07× | **0.0013 / 0.0015** |

**HAM 附带发现**：两条通路**方向相反、部分对冲**（fc +4.1…+5.9 vs conv −2.2…−2.6，净剩 +1.4…+3.7）。

三口径恒等式残差均在 float32 误差量级（HAM ~4e−06、MIMIC ~1e−06、CheXpert ~3e−06）⇒ 分解实现正确。

> ⚠️ CheXpert 的 27.5% 是**跨子群均值，掩盖了极大的异质性**：逐子群 Shapley fc 占比从
> `Female/White/<60` 的 **7.9%** 一路到 `Female/Non-White/≥60` 的 **71.7%**（MIMIC 各组集中在
> 20–30%）。引用 CheXpert 的「conv 主导」必须附这一跨度，否则会掩盖 §5.4 的单子群离群现象。

### 5.4 CheXpert 特有发现：条件化几乎全部集中在**一个**子群

CheXpert 的组间差异**不是弥散的**，而是压倒性地由最小、最弱的子群
`Female / Non-White / ≥60` 独自承担（该组也正是 baseline 台账里 3-seed 一致的最弱交叉子群，
full test n=553）：

| 量 | 其余 7 组 | `Female / Non-White / ≥60` |
|---|---|---|
| `‖w_eff‖` | 0.457 – 0.471 | **0.512**（+12%） |
| 与平均超平面余弦 | 0.9980 – 0.9996 | **0.9916**（唯一明显偏转） |
| `mean Δlogit` vs 参照 | −0.03 … +0.09 | **−0.39** |
| `mean\|fc 贡献\|` | 0.009 – 0.163 | **0.547** |
| Shapley fc 占比 | 7.9% – 39.2% | **71.7%**（唯一 fc 主导组） |
| `I/\|Δlogit\|` | 0.00 – 0.03× | 0.12×（交互项仍小，分解稳） |

**判读（严格按 §4.3 规则）**：这一切都是**口径 B（logit 幅度）**的读数。**口径 C 仲裁的结论是
「什么也没发生」**——该组反事实 AUC 0.8509，与其余组的 0.8506–0.8508 无实质差别，
`max|ΔAUC|` fc 0.0000 / conv 0.0002。**HN 学到的是把该组整体下压约 0.4 logit（操作点/校准偏移），
而非改变组内排序**，与 CheXpert `I(Y;A|X)≈0`（作业 68963）及三 HN 变体「Overall AUC 全部打平
baseline 0.8720」完全自洽。

**不得**由此写成「HN 识别并补偿了最弱子群」——那需要口径 C 上的 AUC 改善，而这里没有。

**与 MIMIC 的对照**：两库同为 CXR、同架构、同 8 子群，MIMIC 的调整均匀弥散且更小
（余弦 ≥0.998、`‖w‖` 几乎恒定），CheXpert 则把同样量级的「无效调整」几乎全部堆到一个小子群上。
可能与 CheXpert 正例率仅 8.2%（该组抽样前 n=1477 中健康仅 86 例）导致该格梯度信号最噪有关，
但**本工作未做统计推断（单 fold / 单 seed），此为观察而非结论**。

### 5.5 Fitzpatrick 特有发现：调整沿肤色轴**单调**，且与子群患病率对齐

CheXpert 的调整堆在单一子群，Fitzpatrick 则相反——**六组的调整沿 Fitzpatrick I→VI 有序排列**
（fold0 test，`skin` 是有序变量，这一点 HAM 的 age 亦然而 CXR 的 sex×race 不然）：

| skin | I | II（参照） | III | IV | V | VI |
|---|---|---|---|---|---|---|
| n（fold0 test） | 589 | 961 | 662 | 557 | 308 | 126 |
| 该组恶性率 | 0.153 | 0.154 | 0.139 | 0.110 | 0.097 | 0.095 |
| `‖w_eff‖` | 0.537 | 0.538 | 0.540 | 0.542 | 0.553 | 0.566 |
| mean logit | −3.78 | −3.62 | −3.73 | −3.95 | −4.15 | −4.10 |
| mean Δlogit vs 参照 | −0.16 | — | −0.11 | −0.34 | −0.54 | −0.48 |
| Shapley fc 占比 | 14.1% | — | 28.3% | 53.2% | 38.1% | 62.9% |
| 反事实 AUC | 0.8967 | 0.8952 | 0.8951 | 0.8954 | 0.8946 | 0.8950 |

三条单调趋势（组级 n=6 的描述性统计，**非推断**）：`‖w_eff‖` 随肤色指数严格递增（Spearman 1.00）；
`mean logit` 与该组恶性率同向（Pearson r=0.95）；fc 通路占比随肤色变深上升（Spearman 0.90）。

**判读（严格按 §4.3 规则）**：HN 学到的是**把深肤色组整体下压约 0.5 logit**，方向与这些组更低的
恶性率一致——即**组先验（患病率）的校准，而非组内判别规则的改变**。**口径 C 仍判定「没改排序」**：
六组反事实 AUC 全在 0.8946–0.8967（跨度 0.002），`max|ΔAUC|` fc 0.0013 / conv 0.0015。

这与 `docs/r1_synthetic_dose_response_results.md` 的因果实验互相印证：R1 注入的**边际先验型**信号
同样只抬 Overall、不动 worst-group 组内排序。Fitzpatrick 在此提供了一个**真实数据上的同型案例**。

**不得**由此写成「HN 补偿了深肤色组的诊断劣势」——那需要口径 C 的组内 AUC 改善，这里没有；
且与 `I(Y;skin|X)≈0`（作业 67913）及等-batch b128 复核「HN worst-case ≈ baseline」完全自洽。

---

## 6. 实施与复现

```bash
# HAM（4 age 组；两套基合成一张 2x4 图）
PYTHONPATH=. python scripts/viz_hyperadapt_hyperplane_ham.py \
    --ckpt  <outputs>/ham10000/cv5/hyperadapt_lr3e-05_wd1e-04_seed42_best_overall.pth \
    --split_dir data/splits/ham10000/cv5/fold0 --batch_size 32

# MIMIC（8 子群；两套基分开两张图）
PYTHONPATH=. python scripts/viz_hyperadapt_hyperplane_mimic.py \
    --ckpt  <outputs>/mimic_cxr/cv5/hyperadapt_lr1e-04_wd1e-03_seed42_best_overall.pth \
    --split_dir data/splits/mimic_cxr_nofinding/cv5/fold0 --n_samples 4000 --batch_size 32

# CheXpert（8 子群；与 MIMIC 共用 src/utils/hyperplane_cxr_runner.py，仅 spec 不同）
PYTHONPATH=. python scripts/viz_hyperadapt_hyperplane_chexpert.py \
    --ckpt  <outputs>/chexpert_cxr/cv5/hyperadapt_lr3e-04_wd1e-04_seed42_best_overall.pth \
    --split_dir data/splits/chexpert_nofinding/cv5/fold0 --n_samples 4000 --batch_size 32

# Fitzpatrick（6 肤色组；两套基分开两张图，2x3 排布；fold0 test 仅 3203 张 ⇒ 全量不抽样）
PYTHONPATH=. python scripts/viz_hyperadapt_hyperplane_fitzpatrick.py \
    --ckpt  <outputs>/fitzpatrick/cv5/hyperadapt_lr3e-04_wd1e-03_seed42_best_overall.pth \
    --split_dir data/splits/fitzpatrick17k/cv5/fold0 --batch_size 64
```

- **配置与 fold 对应**：配置取 `cv5/selected_configs.json`（HAM `lr3e-05_wd1e-04`、
  MIMIC `lr1e-04_wd1e-03`、CheXpert `lr3e-04_wd1e-04`、Fitzpatrick `lr3e-04_wd1e-03`）；
  **seed42 ↔ fold0**，必须用对应 fold 的 held-out test（避免泄漏）。
- **特征抽取**：`avgpool` 上挂 forward hook，不改动模型代码。
- **HAM**：三 split 过滤 `age_group>=0`（`AgeEmbedding` 不接受 −1），fold0 test 1990 → **1943**；
  用 `Subset` 内存过滤，不落临时 CSV。
- **MIMIC**：fold0 test 39,872 张，全量 ×8 组代价过高 ⇒ 按 `label×sex×race×age` **16 格分层抽样**
  （默认 n=4000，每格 250）。分层保证小子群（`Male/Non-White/<60` split 内仅 1,647）也有支撑。
  ⚠️ 逐组 AUC 是在这个**平衡子样本**上算的，**不等于** split 全量指标。
- **CheXpert**：fold0 test 27,729 张，同 16 格分层抽样；但正例率仅 **8.3%**，`label=1` 的小格取不满
  配额（`Female/Non-White/≥60` 全 split 仅 86 例）⇒ 各格实际不等量，n=4000 请求实得 **3,568**，
  抽样后整体健康率 0.439。**逐组 AUC 是在这个平衡子样本上算的，不等于 split 全量指标**
  （0.851 vs 台账 test 0.872 的差异主要来源于此，不是模型退化）。
- **Fitzpatrick**：`preproc_224x224` 为自然 RGB（uint8），eval transform 仅 `ToTensor`+`Normalize`
  （224 原生，无 Resize/Crop，与训练脚本一致）；split 构建阶段已排除 `fitzpatrick_scale==-1`，
  **无需**像 HAM 那样内存过滤。fold0 test 3,203 张 **全量扫掠不抽样** ⇒ 逐组 AUC 即 split 全量指标。
  ⚠️ 型 VI 仅 126 张（恶性 12 例），该组任何逐组读数的噪声本就很大。
- **成本**：纯推理，本地 GTX 1050 Ti 即可。HAM ~2 min；MIMIC ~15 min；CheXpert ~13 min；
  Fitzpatrick ~4 min。不需上集群。
- **16-bit PNG**：MIMIC / CheXpert 均由 `MIMICCXRDataset` 内部处理，禁用 `convert("L")`
  （见仓库 `CLAUDE.md`）。

---

## 7. 判读规则与禁止说法

**必报**（四项缺一不可）：
1. in-plane 能量（逐组）；
2. 判别基诊断：`between_var_along_e1` / `between_var_in_perp` / `e2_perp_var_ratio` / `s1/s2`；
   PCA 基诊断：两个主成分的解释方差比；
3. 通路占比的三口径 + `I/|Δlogit|` + `max|ΔAUC|`；
4. 抽样口径（是否分层、实得 n、逐组 AUC 是否为子样本上的值）。

**禁止说法**：

| ❌ 不得说 | 理由 |
|---|---|
| 「两组决策线重合 ⇒ 超平面相同」 | 投影损失；须先看 in-plane 能量 |
| 「超平面转了 X° / 平移了 Y」（不带点云） | 规范自由度；只有点相对线的位置有含义 |
| 「conv 主导 / fc 主导」（不标口径） | 三口径可给出相反答案（§4） |
| 由口径 B（logit 幅度）推出性能或公平性结论 | logit 平移不改排序；须用口径 C |
| 用 t-SNE/UMAP 画决策边界 | 非线性、无逆映射；边界不可解析绘制 |
| 「e2 捕获了 X% 的组间差异」 | `e2` 只对**正交补**负责；沿 `e1` 的份额由横轴承担（§5.2） |
| 对 `s1/s2` 接近 1 的库（MIMIC 1.43 / Fitzpatrick 1.38）解读纵轴方向 | `e2` 近乎简并、方向不稳定 |

---

## 8. 边界与未做

- **单 fold（fold0）、单 seed（42）**。方向性结论（MIMIC 几乎不调整、判别基优于默认 PCA、
  顺序敏感性可忽略）幅度悬殊因而稳；但逐子群占比数值、HAM 的 66.9% 与「61%→73% 随年龄递增」
  的单调性，需 5 fold × 5 seed 复核方可写入论文。
- 只做了 **HyperAdapt**。HyperHead/HyperFusion 可复用同一模块（HyperHead 特征与属性无关 ⇒ conv 项
  恒为 0，是天然的零对照），尚未跑。
- 只做了 **HAM + MIMIC + CheXpert + Fitzpatrick**。PAPILA 未做。
- CheXpert 的「条件化集中于单一小子群」（§5.4）是 **fold0/seed42 单点观察**，且该组正是抽样后
  样本最少的格（n=336）；是否稳定需 5 fold × 5 seed 复核，**现阶段不得写入论文结论**。
- Fitzpatrick 的三条单调趋势（§5.5）是 **n=6 组级描述性相关**，单 fold/单 seed、无 CI、无置换基线；
  且最深的型 VI 仅 126 张。「HN 学到组先验校准」的因果解释需 5 fold × 5 seed + 与实际患病率的
  对照实验才能坐实。
- 未做**统计推断**：所有数字为点估计，无 CI / 无置换检验。与消融的置换基线（σ 口径）不可直接比强弱。

---

## 9. attribute-blind 零对照：ERM / SWAD 的单-panel 版（2026-07-28）

**实现**：`src/utils/blind_hyperplane_viz.py` + `scripts/viz_blind_hyperplane.py`（四库统一入口）。
**产物**：`outputs/{ham10000,fitzpatrick,mimic_cxr,chexpert_cxr}/hyperplane_viz/blind_hyperplane_{swad,erm}_<tag>.png`
（+ metrics JSON / arrays npz）。同为 **fold0 / seed42 单点**，配置取 `cv5/selected_configs.json`。

### 9.1 为什么不能照搬 §2 那套图

ERM / SWAD 都是 image-only（SWAD = ERM + loss-valley 区间权重平均），forward **不吃属性**。于是：

| | HyperAdapt（§2–§5） | ERM / SWAD（本节） |
|---|---|---|
| 反事实属性扫掠 | 有（每子群一次前向） | **不存在**：一张图只有一个特征、一个 logit |
| 超平面 | 每子群一条 `w_eff(a)` | **全局唯一一条 `w`** |
| Δlogit 通路分解 | fc / conv 两项 | **两项恒为 0**，不做通路图 |
| 纵轴 `e2` | 组间 `w_g − w_bar` 的主方向 | 组间**特征均值** `mu_g − mu` 的主方向 |
| 组间位移的归因 | **HN 的条件化**（病人固定） | **数据本身**（不同人群影像分布不同） |

`build_discriminative_basis` 在此会因 `w_g − w_bar ≡ 0` 直接抛错——这不是 bug，是模型没有该自由度。

⚠️ **两套图的「组间位移」不可混谈**：HN 图里的位移是模型造出来的，这里的位移是数据自带的。
也**不得**说 ERM/SWAD「调整了超平面」——它根本无可调之处。

### 9.2 两个在本版中**退化为恒等式**的量（不可当诊断读）

1. **in-plane 能量 ≡ 1.000**：`e1 = w/‖w‖` ⇒ `Pᵀw = (‖w‖, 0)`，判别方向必然完整落在平面内。
   在 §5.2 它是真诊断（PCA 基下会掉到 0.38），这里是构造保证，图内已标注 `by construction`。
2. **决策线严格垂直**：同一原因 ⇒ 线方程退化成 `u1 = −const/‖w‖`。故「点相对线的位置」
   完全由横坐标决定 = logit 符号。

仍然有效的诊断：`between_var_along_e1`（组间均值差异中属**纯得分平移**的份额）、`e2_perp_var_ratio`、
`s1/s2`。注意 `s1/s2` 在本版普遍**比 HyperAdapt 版高得多**（HAM 4.76、MIMIC 6.66 vs HN 版的 2.70、1.43）
——组间**特征均值**差异有明确主方向，而组间 `w` 差异在 MIMIC/Fitzpatrick 上近乎各向同性噪声。

### 9.3 渲染约定（与 §2 不同，读图前必看）

- **单 panel**：全部样本一张散点 + 唯一决策线（用户口径选择；代价是子群点云互相遮挡）。
- **颜色 = 子群**，用**单 hue ordinal ramp**，非分类色板。理由：散点属 `--pairs all` 口径，
  8 个分类色无法通过 CVD/normal-vision 地板（dataviz skill；已用 `validate_palette.js --ordinal` 验证
  4 级与 6 级 ramp 全项 PASS）。HAM `age_group` / Fitzpatrick `skin` **本就是有序属性**，
  sequential 是正解；**CXR 的 race×age 并非有序量**，ramp 仅作稳定区分（图注已声明），
  `sex` 另由 **marker 形状**承载，且每子群均有直接文字标签 ⇒ identity 不由颜色单独承载。
- **标签 y 不逐点着色**（颜色通道已给子群），改由每子群的 **y0/y1 质心对**表达：
  空心 = y0 质心、实心 = y1 质心，两者**沿判别轴的间距 = 该子群的类间分离度**。
- 样本选取与 HyperAdapt 版**逐位一致**（同 split/fold/seed；HAM 同样过滤 `age_group>=0` 得 1943；
  CXR 同 16 格分层抽样、同配额与 seed）⇒ 两套图画的是同一批图像，可直接并排对照。
  ⚠️ 故 CXR 的逐组 AUC 仍是**平衡子样本**上的值，不等于台账全量指标。

### 9.4 读数（fold0 / seed42）

| | HAM10000 | Fitzpatrick17k | MIMIC-CXR | CheXpert |
|---|---|---|---|---|
| n（本图） | 1,943（全量，filtered） | 3,203（全量） | 4,000（分层） | 3,568（分层） |
| Overall AUC SWAD / ERM | **0.9141 / 0.8797** | **0.9259 / 0.8938** | 0.8276 / 0.8253 | 0.8535 / 0.8573 |
| 组间均值差异沿 `e1` SWAD / ERM | 33.4% / 29.5% | 15.2% / 6.7% | 10.9% / 8.1% | 18.5% / 13.6% |
| `s1/s2` SWAD / ERM | 4.76 / 4.44 | 1.64 / 1.72 | 6.66 / 5.58 | 4.26 / 4.65 |
| 最弱子群（SWAD） | age 20-40 **0.839** | 型 VI **0.842**（n=126） | Male/Non-White/<60 **0.796** | Female/Non-White/≥60 **0.785** |

三点观察（**均为单 fold/单 seed 的描述性读数，无 CI**）：

1. **各库最弱子群与台账一致**：HAM 的 20-40、CheXpert 的 Female/Non-White/≥60、Fitzpatrick 的型 VI
   都复现了各自 outputs 台账里的最弱格，说明本图的逐组读数与既有评估口径自洽。
2. **CXR 两库的组间排布主要由 age 驱动**：8 个质心沿 `e2` 明显分成 `<60` 与 `≥60` 两簇，
   与两库台账「主公平轴 = Age」一致；race/sex 在图上造成的分离小得多。
3. **SWAD 在两个皮肤镜库上明显高于 ERM**（HAM +0.034、Fitzpatrick +0.032），CXR 两库则打平
   （±0.004）——方向与 `docs/oof_regime_results.md`「SWAD 仍 ID 最强」一致；但**这是单 fold
   单 seed 的点估计，不能替代该文档的 5 折口径**。

### 9.5 逐子群 panel 版 + 组间差异的置换检验（2026-07-28 追加）

**产物**：同目录 `blind_subgroup_panels_{swad,erm}_<tag>.png`（与 §2 的 HyperAdapt 图同布局，
可并排对照）。每 panel = 一个子群的**真实成员**（无反事实），点按 label 着色（同一套
`CLASS_COLORS`），**决策线在每个 panel 完全相同**——不是 bug，模型只有一条。
每 panel 叠：该子群 y0/y1 质心（类别色）+ **全体 pooled 的 y0/y1 质心（灰，各 panel 相同）**，
两者之差即该组的组间位移。

**位移分解（panel 标题的两个数）**：以全体在各轴上的 SD 为尺度，
- `d∥` = 沿 `e1` 的质心位移 ⇒ **纯得分/操作点偏移，不改组内排序**；
- `d⊥` = 沿 `e2`（判别方向的正交补）的位移 ⇒ **对该分类器完全不可见**。

**置换检验**（`permutation_test_between_group`，B=2000，纯 numpy 秒级）：打乱子群标签重算
between-group sum of squares，分别检验 total / 沿 `e1` / 正交补三部分。

| | HAM10000 | Fitzpatrick17k | MIMIC-CXR | CheXpert |
|---|---|---|---|---|
| `p_total`（SWAD / ERM） | 0.0005 / 0.0005 | 0.0005 / 0.0005 | 0.0005 / 0.0005 | 0.0005 / 0.0005 |
| **正交补占组间方差** SWAD / ERM | **70.4% / 73.9%** | **80.5% / 90.5%** | **89.1% / 91.9%** | **81.8% / 86.7%** |
| `d∥` 范围（SWAD） | **−0.28 … +0.79** | −0.17 … +0.11 | −0.27 … +0.29 | −0.42 … +0.29 |
| `d⊥` 范围（SWAD） | −0.36 … +0.84 | −0.44 … +0.74 | −0.34 … +0.30 | −0.37 … +0.49 |

> `p=0.0005` 是 B=2000 的分辨率地板（1/2001），即「所有置换都没达到观测值」。
> ⚠️ **n 在 1,943–4,000 之间，组间差异必然显著；`p` 本身不带信息量，判读必须看效应量**（`d∥`/`d⊥`）。

**四点判读（fold0/seed42 单点，无 CI）**：

1. **组间差异 70–92% 正交于判别方向**（八个配置全部如此）。即：attribute-blind 模型的特征空间里
   子群**分得很开**，但分开的方向绝大部分**与打分无关**。这是「`I(A;X)>0` 但 `I(Y;A|X)≈0`」的
   直接几何图像，与 §5 的 HyperAdapt 结论（条件化全花在沿判别方向的平移/缩放上、不改排序）互补。
2. **Fitzpatrick 是最干净的「可分但无关」样本**：`|d∥| ≤ 0.17` 而 `d⊥` 从型 I 的 −0.44 单调走到
   型 V 的 +0.74 ⇒ 肤色在特征空间沿正交方向强烈分离（视觉上本就可解码），却几乎不动得分。
   与 `I(Y;skin|X)≈0`（作业 67913）完全自洽。
3. **HAM 是唯一 `d∥` 有实质且单调分量的库**：age 20-40 → 80+ 的 `d∥` 为 −0.22 / −0.28 / +0.34 / +0.79，
   与各组恶性率 0.043 / 0.098 / 0.218 / 0.301 同向 ⇒ 模型已把年龄相关的**风险先验**编码进得分。
   HAM 也正是四库中唯一曾检出 age 条件信号的库——但这是 **n=1 的对应，不构成规律**。
4. **CXR 两库由 age 主导**：`<60` 组 `d∥>0`、`≥60` 组 `d∥<0`，race/sex 造成的位移小得多；
   CheXpert 上最负的两格是 `Male/Non-White/≥60`（−0.42）与 `Female/Non-White/≥60`（−0.40）。
   ⚠️ **CheXpert 的 `d∥` 含抽样伪影**：16 格分层抽样在小格取不满（这两格 no-finding 率仅 0.271/0.256，
   其余格 0.500），类别构成差异本身就会推动 `d∥`；MIMIC 八格全为 0.500，其 `d∥` 无此污染。

**附带观察（四库一致，但机制未查）**：正交补占比 SWAD 一致**低于** ERM（70.4<73.9、80.5<90.5、
89.1<91.9、81.8<86.7）⇒ SWAD 的组间差异相对更多落在判别方向上。四库同向故不像噪声，
但也可能只是两者 `w` 方向不同导致的分解差异，**未做因果检验**。

### 9.6 🛑 判读红线：组间可分 ≠ 需要条件化

这套图最容易被误读成「组间差别显著 ⇒ 需要属性条件化」。**不成立**：

- 组间点云分得开，测的是 **`I(A;X) > 0`**（属性能否从图像读出）。本项目四库**全部**满足，
  且早已知道——Fitzpatrick 的肤色 `I(skin;X) ≥ 0.37 bits`、CXR 的性别可从胸片解码
  （见仓库 `CLAUDE.md` 通路诊断节：XOR-sex 正控制正是因此被污染）。
- HN 有收益的必要条件是 **`I(Y;A|X) > 0`**（给定图像后属性对**标签**还有用）。这是另一个量，
  已由 OOF conditional V-information 闸门测过：**五库中仅 PAPILA-age 检出**
  （见 `docs/conditional_v_information_gate.md`）。
- Fitzpatrick 就是两者分离的现成反例：`I(A;X)` 大、`I(Y;A|X)≈0`、HN 无收益。

**因此本节的图不能用来论证条件化的必要性**；它的正当用途是反过来——量化「组间差异有多大比例
落在与判别无关的方向上」，为 `I(Y;A|X)≈0` 提供几何解释。图内 suptitle 已写入这条红线。

### 9.7 未做

- 只做了 ERM / SWAD 两个 blind 方法；ROC（后处理）未做——其超平面与 ERM 相同，只改阈值，
  在本图上表现为决策线的**平移**，可作为便宜的后续补充。
- 单 fold / 单 seed，无 CI。§9.5 的置换检验只检「组间差异是否为 0」，**不**给 `d∥`/`d⊥` 的 CI；
  跨 fold/seed 的稳定性未查（§9.5 判读 3、4 与「SWAD 正交补占比更低」尤需复核）。
- 单 panel 布局（§9.3）下 CXR 8 子群的点云互相遮挡严重（选定口径的已知代价），
  组间比较应以质心与右侧表为准；panel 版（§9.5）无此问题。
- 逐组 `d∥` 在 CheXpert 上与分层抽样后的类别构成纠缠（§9.5 判读 4），未做去混杂；
  若要干净读数应改为按各组真实患病率加权或全量扫掠。

---

## 10. 相关文档

- **本方法的 OOD 扩展（MIMIC↔CheXpert，ERM/SWAD/HyperAdapt 两方向）**：
  `docs/ood_hyperplane_visualisation.md` —— 同帧 ID↔OOD 叠加。关键补充：超平面只由 checkpoint 决定，
  跨域**逐位不变**，故迁移在本坐标里只能表现为点云平移/形变；实测迁移损失全落在**类间分离度**上
  （与 `d∥` 无关，Spearman +0.94 vs +0.03），且 HN 的条件化跨库**原样搬运**（逐组 Δlogit r=0.98–0.99）
  却**同样惰性**（口径 C max|ΔAUC| ≤0.0007）。
- 性能/公平性权威口径：`docs/comparison_protocol.md`、`docs/oof_regime_results.md`
- 信号闸门：`docs/conditional_v_information_gate.md`
- 条件化范围消融与 fc/conv 归因（口径 A、C）：`docs/conditioning_ablation_summary.md` §4、
  `conditioning_ablation_e1_pathway_knockout.md`、`conditioning_ablation_e3_fcoff_training.md`
- 权重空间轨迹（口径 A）：`scripts/viz_hyperadapt_weight_trajectory.py`、
  `scripts/viz_hyperadapt_trajectory_mimic.py`
- DeepView 决策边界（非线性、含逆映射近似）：`scripts/deepview_{ham,mimic}_hyperadapt_lib.py`
