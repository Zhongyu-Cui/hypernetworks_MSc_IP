# 条件化位置 × 参数共享 机制解剖实验方案

> 🛑 **信号闸门口径更新（2026-07-19）**：闸门已由单-split 条件互信息改为 **OOF conditional V-information**
> （Xu 2020 / Hewitt 2021；权威 = `docs/conditional_v_information_gate.md`）。关键变化：
> **HAM-age 由「唯一正信号」退回「未检出」**（+0.0028 bits，CI 触 0）；Fitzpatrick ≈0；
> **CheXpert / MIMIC 全轴 CI<0 = 决定性 0**；唯一 detected 为 **PAPILA-age**（+0.041 bits，但 n=420）。
> control task 五库全部通过 ⇒ 零读数非假阴。本文「HAM-age `I(Y;age|X)>0`」一族表述**已过时**（下方已内联订正）；
> 各处**结论方向不变**，但「有信号」前提须按新闸门重读。


> 修订日期：2026-07-14（v4，九轮内部审阅 + 八轮外部评审全部并入）。本文件是「为什么 HyperHead /
> HyperFusion / HyperAdapt 表现不同」这一研究问题的**实验规格**。**状态（外部评审 #四.1 校准）：确认性骨架
> 已冻结**（研究问题、终点层级、主对比、Holm 族、统计判决、bootstrap 算法、OOD 估计量均已定死）；**以下仍待
> 冻结、须在 E0 前定稿**：精确 cell manifest（当前 HAM「~21 唯一 cell」仍是估数，§9）、**共享轴 budget-matching**
> （`S-indep-budget / S-sharedA-budget / S-global-trunk-budget` 的 hidden/projection 宽度与确定性搜索算法及容差；
> **注：范围端点 `param-matched-to-ds` 已冻结 h=17，见 §3.2/§4.3**）、S-indep 生成器参数量（新实现后统计，§4.3）、
> n_pos/n_neg 门槛（E-audit 后终定）、**动态-min 校准的资源上限数值/基准规模/时间单位与 σ 降阶规则**（§11，E-audit 定）。
> **建议 E0 前先人工产出一版 cell manifest**（不需等模型代码），cell 身份定死前预算不算真正冻结。剩余为**实现验收项**
> （模块 manifest 断言、bootstrap 脚本、null 自检等，见 §12）。
>
> **v4 外部评审（第十轮）三大结构性修订**（已核对代码属实、生成器参数量逐一复算吻合）：
> ① **OOD 确认性主估计量由 k→k 改为 full-target**（每 source fold×seed 评完整 target、配对效应等权平均；
>    k→k 降为 legacy sensitivity，另报 5-fold logit ensemble）——§7.1/§7.4/§8.2/§9。
> ② **引入 canonical ERM base state**（每 `(dataset,fold,seed)` 冻结 `base_backbone/base_fc/data_order/aug`
>    RNG，所有 cell 加载同一 base 再初始化 adapter），并把 `split_id=fold` 与 `train_seed∈{0..4}` 解耦——
>    否则「Δθ=0 等价 ERM」只对该 cell 自身成立、不对用于比较的 ERM 成立——§1.3/§6.3/§7.5。
> ③ **容量/DoF 由「每层 rank×C」升级为五量表并冻结初始化数值**（HyperHead 生成器 34,201 / HyperFusion
>    8,550,816 / HyperAdapt 2,764,388 可训练参数，实测吻合）——§0.2/§4.3。
> 与主比较协议 `docs/comparison_protocol.md`（六方法主比较）平行、互补：主比较回答「HN 在各数据集
> 上是否/何时有收益」，本方案回答「HN 内部的**条件化范围**与**参数共享结构**如何决定其效用与
> 公平性」。评估复用主比较的 CV-OOF / val 选配置 / Overall·worst·gap，本方案**只新增模型侧的受控
> 消融设计与若干统计/诊断加强**，评估管线主体不改。
>
> 状态标记：`[ ]` 未开始 · `[~]` 进行中 · `[x]` 完成。每个复选框 = 一件可独立完成并验证的任务。

---

## 0. 研究问题与核心诊断

### 0.1 研究问题（RQ）

> **条件化的「位置/范围」与「参数共享方式」如何影响超网络的整体效用与群体公平性？
> 为什么 HyperHead、HyperFusion、HyperAdapt 表现不同？**

- **RQ1（范围）**：把条件化从深层单点逐步扩展到浅层、铺满更多层，是否带来效用/公平的额外收益？
- **RQ2（共享）**：跨层共享条件生成器（shared-A）相对每层独立生成，**是否改善 worst-group fairness**（确认性）；
  **训练波动降低是否能作为其机制解释**（次要机制证据）——**不以「或」并列两个成功出口**。
- **RQ3（归因）**：原三种模型的差异，来自**条件化范围**、**参数化形式（低秩 vs 直接生成）**、
  **调制形式（加性 vs 乘性）**、**容量**，还是**初始化**？**（地位：范围/共享轴由 RQ1/RQ2 承担确认性推断；
  参数化/调制/初始化的机制归因为预先规定的受控对照，报双侧效应+CI、不进 Holm 族、结论止于「与机制解释一致」，
  见 §4.4——RQ3 是描述性机制解剖，不作「确认某因素导致差异」的确认性声称。）**

### 0.2 核心诊断：现有三方法的比较是完全混杂的

把三个模型的实现（`src/models/resnet18_hyperhead.py`、`resnet18_hyperfusion.py`、
`resnet18_hyperadapt.py`）逐轴对齐：

| 轴 | HyperHead | HyperFusion | HyperAdapt |
|---|---|---|---|
| ① 条件化范围 | 仅 fc（head-only，不改 backbone） | 仅 `layer4[0].downsample`（单点 backbone） | 除 stem 外每个 conv **＋ fc**（最遍布） |
| ② 参数化 / 共享 | 直接生成完整 fc 权重 | 直接生成完整 1×1 conv 权重 | 低秩 A·B；A 按 stage 共享、B 每层独立 |
| ③ 调制形式 | **绝对生成完整头（无共享基础头，非加性残差）** | 加性残差 θ=θ₀+Δθ | conv 乘性 (1+M)、fc 加性 |
| ④ **每样本生成参数维度** | 513 | **131 072** | A 坐标 3,844 + B 坐标 17,408 = **21,252** |
| ⑤ 初始化 | 末层 std=1e-3（绝对头，**不与 baseline 等价**） | MIP=Kaiming×0.01（**近似**非严格）+ E_L2 | A=0, B=Kaiming（Δθ=0 严格） |

> ⚠️ 轴④是**每样本生成的输出维度**，**不等于**超网络可训练参数量或有效容量。实际容量另用**五量**
> 表征，见 §4.3。**冻结实测值（rank=4，代码复算）**：
>
> | 模型 | 可训练生成器参数 | 每样本发射坐标 | 实现调制张量元素 | 可识别低秩流形 DoF |
> |---|---:|---:|---:|---:|
> | HyperHead | 34,201 | 513 | **513（512 权重 + 1 bias）** | **513** |
> | HyperFusion | **8,550,816** | 131,072 | 131,072 | 131,072（满秩 1×1，bias=False） |
> | HyperAdapt（命名模型 = **含 fc** = L-all+head 位置） | 2,764,388 | 21,252（A/B 因子） | **1,393,152**（Σ C_out·C_in + fc 512） | **19,648**（见下 stage-coupled 公式） |
>
> **⚠️ 此行是含 fc 的命名 HyperAdapt（= §3 桥接 L-all+head）、且为 CXR(PatientEmbedding) 口径**；**§4 共享主轴的
> S-sharedA ≡ L-all-conv 不含 fc、数字不同**（生成器 CXR 2,499,680 / **HAM 2,495,552**、发射 19,200 / DoF 19,136），
> 且**编码器按数据集不同**——共享轴首战 HAM 用 2,495,552，见 §4.3 分数据集表，两者勿混。
>
> **HyperAdapt DoF 必须用 stage-coupled 公式（外部评审 #3，`Σ_l r(C_in+C_out−r)` 是 independent-A 上界、
> 会重复计 A 与 −r²）**：因 A_s 按 stage 共享、每 stage 只有一组 A_s 与一次 GL(r) 基变换不可识别性，正确的
> 每-stage 流形维数 ≈ `r·C_out,s + Σ_{l∈s} r·C_in,l − r²`。ResNet-18/rank=4 实算：conv **19,136** + fc adapter
> **512**（ΔW 形状 1×512、rank≤1）= **19,648**；`independent-A 上界`同口径为 34,512（仅作对照上界，非实际）。
>
> **关键反直觉**：HyperFusion 每样本只改单点 shortcut，但因其生成器要直接吐出 131,072 个满权重系数，
> **可训练生成器参数（8.55M）反而是 HyperAdapt（2.76M）的 3 倍、HyperHead（34k）的 250 倍**——「注入范围
> 最广」与「生成器容量最大」是**两个不同的轴**，命名模型比较把它们混在一起，故预算匹配须按五量分别报、
> 不能只对齐单一数字（§4.3）。

**没有任何一对方法只差一个轴。** 现有观察（如「HAM-age 上深层 HN 兑现公平收益、浅层 HyperHead
弱」，记忆 `ham10000-conditional-mi-age-signal`）无法归因——可能是范围、可能是容量（④ 差 256 倍）、
可能是初始化。本方案通过「只改一个轴、其余全锁死」的受控网格分离它们。

### 0.3 信号地图与研究定位（**权威依据：`docs/oof_regime_results.md`，CV-OOF 严格口径，2026-07-10**）

> ⚠️ **口径说明**：本节数字取自 `oof_regime_results.md`（5 折 CV-OOF 池化、患者/病灶级、样本级配对
> bootstrap）。它**取代** `results_summary.md` 的 D4（旧单-split / 5-seed OOD 口径，已过时——早前据
> D4 得出「MIMIC 是负控制、OOD HN 不胜」是**错误**结论）。

条件信号 `I(Y;A|X)`、ID 与 OOD 现状：

| 数据集 | n(OOF) | `I(Y;A\|X)` | ID worst-group vs ERM | OOD | 对本研究的意义 |
|---|---|---|---|---|---|
| **HAM-age** | 9,707 | **age >0** | 全 n.s.（HyperFusion 单点转正 +0.015/+0.044） | — | ID 最清晰条件信号，范围/共享轴 ID 主战场 |
| **MIMIC** | 199,356 | **弱 >0** | **HyperAdapt marginal +0.003 显著**（幅度小；余 ≤ERM，HyperHead 显著劣） | **M→C 深层 HN 稳健胜（legacy k→k，待 full-target 复核）** | **legacy k→k 下唯一 HN 稳健胜 SWAD 处**（见下；full-target 复核前不作定论） |
| CheXpert | 138,644 | ≈0 | 大 n 下 HN 显著微劣（Δ≈−0.005） | C→M 全 HN 显著劣 | 干净负控制 |
| Fitzpatrick | 16,012 | ≈0 | 全 +0.004~0.011 n.s.（论文本尊数据集） | — | 便宜负控制 |
| PAPILA | 420 | ≈0 | 全 ≤ERM n.s.（功效低） | — | 功效不足，仅趋势旁证 |

**关键正向结果（本研究最强单点证据，MIMIC→CheXpert OOD）**：深层 HN
（HyperFusion **marginal worst +0.0058 vs ERM / +0.0079 vs SWAD**；HyperAdapt **+0.0061 vs ERM /
+0.0082 vs SWAD**，均显著）**同时超越 ERM 与 SWAD**，HyperAdapt 连 Overall 亦显著超两者；head-only 的
HyperHead 全 n.s.。反向 C→M（source=CheXpert，`I≈0`）无信号可迁移，全 HN 显著更差 = **强方向不对称**。

**范围 × 信号 × 泛化——一个待检验的方向性假设（非既成规律，表述已收严）**：
- **现有结果呈现方向性假设**：在 HAM 与 MIMIC 中，**能修改 backbone 特征的方法**（HyperFusion/HyperAdapt）
  相对 **head-only**（HyperHead）更有利；`I≈0` 时（CheXpert/Fitz/PAPILA）HN 打平或大 n 下显著微劣。
  **但该模式在 HAM 尚不稳健**（ID 整体未显著），**且命名模型比较本身混杂**（尚不能排除容量、初始化与
  参数化差异）——**本实验正是要对其做受控检验**。（术语：HyperFusion 只条件化单个高层 shortcut、HyperAdapt
  才广泛条件化，不宜统称「最深」；统一用 **backbone-level** vs **head-only conditioning**。）
- **「`I(Y;A|X)>0` + backbone 层条件化 + OOD 压力，三者缺一则 HN 退化为盲基线」目前是待检验的经验假设**
  （consistent with 现有 OOD 数据），**不是**已成立的规律。SWAD 仍是 ID 下最稳的公平方法。

**本研究定位（据此更新）**：机制解剖既要**解释 null**（ID worst-group 为何普遍不胜），又要**解剖一个在 legacy
k→k 口径下观察到的候选正向信号（外部评审第五次 #小4：full-target 尚未复核，不预设「真实」）**——即在 M→C OOD
这个 **legacy k→k 口径下唯一显示稳健正向、待 full-target 复核（§11 E-audit-OOD）的候选正向 regime** 里，**若该
效应在 full-target 下复现**，定位**其 worst-group 稳健赢的最小有效配置**（哪一档范围 / 何种共享是保留该效应的必要
组成、或单独足以恢复该效应）；**若不复现，则转为「该效应为何仅 k→k 口径出现」的机制解释**。**表述边界（#12）**：
受控消融至多能说明「**在本数据集、ResNet-18 与所测模型族内**，某范围/共享是否为保留该效应的必要成分、或是否单独
足以恢复」，**不得**上推为「医学图像 HN 的一般必要/充分条件」。诚实边界：ID 下增益仍可忽略/不稳健，候选正向证据主要在 OOD。

---

## 1. 统一可配置模型 `ResNet18CondNet`

**一次实现、全实验共用。** 改造 `resnet18_hyperadapt.py`，暴露正交开关，其余锁死，消除 ③（形式）⑤
（初始化）与编码器的混杂、并固定每层 rank。**注意（#7）**：④ **总容量**无法被范围轴同时锁死——扩大范围
必然增加 adapter 层数/生成器参数/DoF/FLOPs，故范围轴结论只表述为「固定每层 rank 与参数化机制下扩大覆盖
范围的**总效应**」（详见 §3.2 容量混杂说明与端点敏感性）。

### 1.1 锁死的公共设定

| 项 | 固定值 |
|---|---|
| 条件编码器 | `AgeEmbedding`/`SkinEmbedding`/`PatientEmbedding`（cat_embed=16, out_dim=128），所有 cell 相同 |
| 调制形式 | conv 乘性低秩 (1+M)、fc 加性低秩（桥接 §4.4 除外，专扫形式） |
| 初始化（**数值冻结，#4；near-zero 与 MIP 是两种不同初始化，禁止混写**） | **① residual 低秩**：shared-A=0（严格 Δθ=0），B/hidden 用 `kaiming_normal_(fan_in,linear)`；**② near-zero（full-additive 用）**：生成器末层 `Normal(0, σ=1e-3)`、bias=0；**③ MIP（初始化消融的对照臂，非 near-zero）**：生成器末层按 `kaiming_uniform_(a=√5)` 后 `×mip_scale=0.01`、bias=0——②③ 的 hidden 层 / bias / E_L2 之外一切初始化**完全相同**，初始化消融正是 ②vs③；**④ absolute head（HH）**：末层 `Normal(0, σ=1e-3)`、bias=0（无共享基础头，**不等价 baseline**）。**等价容差（混合，防 base logit≈0 时相对误差爆炸）**：以 baseline 前向为参照，`residual_zero_exact` 断言 `‖Δlogit‖_∞ < 1e-6`（严格）；`residual_near_zero` 断言 `‖Δlogit‖_∞ ≤ atol + rtol·‖logit_base‖_∞`（`atol=1e-3, rtol=1e-2`）**且** `‖ΔW‖_F/‖W‖_F < 1e-2`（权重侧双查）。 |
| **canonical base state（#2，硬约束）** | 每 `(dataset, fold, seed)` 先生成唯一 `base_backbone_state`（ImageNet 预训练，跨 cell 天然相同）＋ `base_fc_state`（**冻结的 fc 初始化**，所有 cell 与比较用 ERM 共享同一份，消除模块注册顺序导致的 RNG 错配）＋ `data_order_seed / augmentation_seed`（派生 RNG）。所有 cell 加载同一 base 后再初始化 adapter——否则「Δθ=0 等价 ERM」只对该 cell 自身成立、不对比较 ERM 成立。**`split_id=fold` 与 `train_seed∈{0..4}` 解耦**（弃用 `seed=42+fold`）。 |
| 因子秩 | conv rank=4；head 用 `factor_width=4`（见 §5.2，**fc 秩≤1，不与 conv rank 同义**） |
| 训练 regime | 对齐现有 HAM HN：`BCEWithLogitsLoss`、`AdamW`、grad_clip=1.0、bs=128、≤30 epoch |
| **模型选择 checkpoint** | **主结果一律用 `best_overall`**（对齐主协议 checkpoint 准则）。`best_worstcase` 仅保存**作诊断**，**禁止**依据 test/OOF 在两者间选优；若要研究 checkpoint 准则须作独立消融 |

### 1.2 两个开关

- **`location`** ∈ {`head`, `ds`, `layer4`, `layer34`, `all-conv`, `all+head`}（§3）。
- **`sharing`** ∈ {`indep`, `sharedA`, `global-trunk`}（§4）。

### 1.3 工作项
- [ ] E0.1 `ResNet18CondNet` 骨架 + `location` 开关（含 `all-conv`/`all+head` 拆分）
- [ ] E0.2 `sharing` 开关（indep / sharedA / global-trunk）
- [ ] E0.3 桥接开关：调制形式（add/mul）、参数化（full/low-rank）、初始化（MIP·E_L2/近零）
- [ ] E0.4 结构自检（**按 cell 分级定义初始等价，#14**，不能一律「严格等价 baseline」）：
      · `residual_zero_exact`（residual low-rank，Δθ=0 严格等价）；
      · `residual_near_zero`（full-additive / MIP，输出差 < 预设容差，近似等价）；
      · `absolute_head_near_zero_logits`（HH-absolute：无共享基础头，**不与 baseline 等价**，只验 logit
        分布稳定不爆炸）。另出前向/反向 OK、五容量量（§4.3）。
- [ ] E0.5 接入统一 harness，复用 predictions/OOF 落盘；固定只落 `best_overall` 为主预测
- [ ] E0.6 **canonical base state 机制（#2）**：实现 `build_base_state(dataset, fold, seed)`，落盘
      `base_backbone_state / base_fc_state / data_order_seed / augmentation_seed`；所有 cell（含比较用
      ERM）在 adapter 初始化**前**加载同一 `base_fc_state`；`split_id` 与 `train_seed` 解耦、各随机源
      （init / 数据顺序 / 增强 / sampler / worker）用可复现的**派生 RNG**。**单元测试断言**：同 `(fold,seed)`
      下 residual cell 的 base fc 与比较 ERM 的 fc **逐元素相等**（否则 Δθ=0 不等价 ERM）。

---

## 2. 因子分解总览

```
                         sharing (仅低秩) →
 location ↓     indep     sharedA(HyperAdapt-style)   global-trunk
  head         ────  桥接: HyperHead 系列（§4.4.1，factor_width）  ────
  ds           ────  桥接: HyperFusion 系列（§4.4.2，容量/形式/init(MIP,E_L2) 四分离）  ─
  layer4          ·            ·                          ·
  layer34         ·            ·                          ·
  all-conv     S-indep      S-sharedA                 S-global-trunk   ← 共享主轴（§4）
                                │
                        容量轴 rank∈{1,4,16}（§5）
  all+head     ── 完整 HyperAdapt 位置的桥接（conv 全 + fc），不进范围主轴 ──
```

---

## 3. 主轴一：条件化**范围**（非严格深度）

固定 sharing=sharedA、乘性低秩、rank=4、Δθ≈0，只改注入范围。

| Cell | 条件化位置 | 角色 |
|---|---|---|
| ERM | 无 | 参照零点 |
| L-head | 仅 fc（factor_width=4） | head-only（不改 backbone）；对齐 HyperHead 介入范围 |
| L-ds | 仅 `layer4[0].downsample`（256→512, 1×1） | **HyperFusion 原始注入点**（最小 backbone 条件化） |
| L-layer4 | layer4 两 block 全 conv | 深层单 stage |
| L-layer34 | layer3 + layer4 conv | 高两层 |
| **L-all-conv** | **layer1–4 全 conv、fc 不条件化** | **范围主轴终点** |
| L-all+head | layer1–4 全 conv **＋ fc adapter** | 完整 HyperAdapt 位置的**桥接**（不进范围主轴） |

> **精确模块 manifest（保证严格嵌套 L-ds ⊂ L-layer4 ⊂ L-layer34 ⊂ L-all-conv —— 报告规范修补）**：
> - **L-layer4** 含 layer4 两 block 的所有 conv **＋ layer4[0].downsample**（故 L-ds ⊂ L-layer4 成立）。
> - **L-layer34** 含 layer3+4 全 conv **＋ 二者的 shortcut downsample**。
> - **L-all-conv** 含 layer1–4 全 conv **＋ layer2/3/4 的所有 downsample**（layer1 无 downsample）。
> - 所有 cell **一律排除 stem（conv1/bn1）与 fc**，除非名称含 head。
> - E0 须让模型**打印挂 adapter 的模块名**，并在单元测试中 **assert 集合嵌套关系**——否则漏挂某 shortcut
>   会破坏「范围增量」的严格嵌套。

> **#1 修正**：原 HyperAdapt 同时条件化「所有残差 conv **＋ fc**」。若范围终点含 fc，则
> `Δ_early = L-all − L-layer34` 会同时混入 layer1–2 conv 与 fc 两个变化，破坏单因子解释。故拆分：
> 范围主轴终点用 **L-all-conv**（fc 保持普通、非条件化）；**L-all+head** 单列为「完整 HyperAdapt
> 位置」的桥接。共享轴（§4）固定在 **L-all-conv**，fc 一律普通不变。

### 3.1 两个对照
- **对照 A（性质差异）= L-head vs L-ds**（head 在池化特征、无空间维；backbone conv 在特征图上）。
- **对照 B（范围扩展）= L-ds → L-layer4 → L-layer34 → L-all-conv**。

### 3.2 判定：增量分析（不要求严格单调）
- Δ_stage4 = L-layer4 − L-ds　→ shortcut 单点是否足够、铺满 stage4 是否有增益
- Δ_stage3 = L-layer34 − L-layer4
- Δ_early  = L-all-conv − L-layer34
- （桥接）Δ_head = L-all+head − L-all-conv　→ 加 fc adapter 的边际效应

> ⚠️ **范围轴未消除容量混杂（#7）**：L-ds→L-all-conv 同时增加 adapter 层数、生成器参数、每样本调制
> DoF、FLOPs。故范围轴**固定每层 rank 与参数化机制**，控制了 ③ 形式 / ⑤ 初始化 / 编码器 / 每层 rank，
> **但未控制总容量 ④**。结论只能表述为「**固定每层 rank 与参数化机制下，扩大条件化覆盖范围的总效应**」，
> **不得**称「纯条件化位置的因果效应」。**端点容量敏感性**：加 **`L-all-conv-param-matched-to-ds`**——**在保持
> stage-shared A + layer-specific B 拓扑不变的前提下（外部评审第七次 #1：删「共享 trunk」措辞、与 §4.3 一致、
> 严禁 global-trunk），把每个 A/B 生成器替换为统一深度的瓶颈 MLP `Linear(128,h)→ReLU→Linear(h,d_l)`，只搜瓶颈
> 宽度 h 使生成器总参数匹配 `N_gen^HAM(L-ds)=415,040`**。**冻结实测**：**h=17 → 414,791（−0.06%，唯一落入 ±2%；
> h=16=−5.4%、h=18=+5.3% 均超）**，故 **h 冻结=17**；A 生成器末层零初始化、B 用 Kaiming（同主轴）、含 bias、
> 激活 ReLU；**不保留「不可达改 global-trunk」的实施出口**。瓶颈 MLP 改了生成器深度与优化几何，故该 cell 只作
> **参数量敏感性对照、非纯容量控制**（调制层数/DoF 仍与 L-ds 不同）。

### 3.3 工作项
- [ ] E2.1–E2.5 L-head / L-ds / L-layer4 / L-layer34 / L-all-conv 训练 + OOF（HAM）
- [ ] E2.6 L-all+head 桥接 + L-all-conv-param-matched-to-ds 端点敏感性训练 + OOF（目标 = N_gen(L-ds)、±2%）
- [ ] E2.7 增量 Δ 分析 + 配对 cluster-bootstrap 差值 CI（§7.4）

---

## 4. 主轴二：参数共享（只比同一低秩形式，固定 L-all-conv）

固定 location=L-all-conv、乘性低秩、rank=4、同 init/编码器，**只改生成器如何共享**。

| Cell | A / B 生成器组织 |
|---|---|
| S-indep | 每层独立 A_l、B_l |
| S-sharedA | A 按 stage 跨层共享、B 每层独立（**HyperAdapt-style backbone sharing**；固定 L-all-conv 不含 fc，**非**完整 HyperAdapt——后者由 L-all+head 表示） |
| S-global-trunk | 全局共享条件主干 z(a)=g(a)，每层保留轻量投影头 A_l=P_l^A(z), B_l=P_l^B(z)（**不广播同一调制**——各层 C_in/C_out/kernel 语义不同） |

### 4.1 主对照
**S-indep vs S-sharedA**（识别性最强：除 A 是否跨层共享外全同）。

### 4.2 两套结果
- **(a) 固定 rank=4**：**共享方式的主要因果证据**（同表示秩比结构）。
- **(b) 参数预算匹配**：**容量敏感性分析**（非无混杂因果比较；仅匹配参数个数 ≠ 匹配有效容量）。

### 4.3 参数预算匹配的可执行定义（#8 修正：模板须保留各自共享拓扑）
1. **统一 MLP 深度、但各保留共享拓扑**：预算匹配版三结构可共享「两层 MLP 深度」，但**共享拓扑必须
   与各自语义一致，不得统一成同一个全局 trunk**（否则破坏实验因子）：
   - **S-indep-budget**：**每层独立** g_l(a)→A_l,B_l，**不得共享 trunk**（否则就不再是「每层独立」）。
   - **S-sharedA-budget**：每 stage 共享 A 生成 trunk、B 仍逐层独立。
   - **S-global-trunk-budget**：全网共享 z(a)=g(a)、各层仅投影头。
2. **hidden 位置明确**：匹配只调各自 trunk 宽度与 projection 宽度，**rank 恒 =4**（不靠改 rank 匹配）。
3. **预算锚点必须按数据集分别计算（外部评审第六次 #1：编码器不同——HAM `AgeEmbedding` 18,752 ≠ CXR
   `PatientEmbedding` 22,880 ≠ Fitz `SkinEmbedding` 18,784；此前的 2,499,680/2,764,388 是 CXR 值，误用作 HAM
   锚点会差 ~4k）。§4 共享轴 + 预算匹配为 HAM 首战（§8.1），故 HAM 锚点为准**：

   | cell（rank=4，含条件编码器口径） | HAM(Age) | Fitz(Skin) | CXR(Patient) |
   |---|---:|---:|---:|
   | S-sharedA ≡ L-all-conv（无 fc） | **2,495,552** | 2,495,584 | 2,499,680 |
   | L-all+head（含 fc） | 2,760,260 | 2,760,292 | 2,764,388 |
   | L-ds | **415,040** | 415,072 | 419,168 |

   - **共享轴锚点（HAM）`N_sharing-anchor^HAM` = 2,495,552**——仅用于 S-sharedA-budget / S-indep-budget /
     S-global-trunk-budget 互相匹配。
   - **范围端点锚点（HAM）`N_range-anchor^HAM` = N_gen^HAM(L-ds) = 415,040**（L-ds 的 A/B 生成器
     `(128+1)×(4×512+4×256)=396,288` + AgeEmbedding 18,752）——仅用于 `L-all-conv-param-matched-to-ds`，
     **禁止用 2,495,552 或 CXR 的 2,499,680 作该 cell 目标**。**实现冻结（外部评审第七次 #1，与 §3.2 一致）**：保持
     stage-shared A + layer-B 拓扑，各生成器换成瓶颈 MLP `Linear(128,h)→ReLU→Linear(h,d_l)`、**h 冻结=17**
     （→414,791，−0.06% ∈ ±2%；h=16/18 均超），**不保留改 global-trunk 的出口**。**口径统一**：全文容量数字均
     **含条件编码器**；若改不含编码器口径须全表同步。
   - **拓扑约束（外部评审第六次 #1）**：`L-all-conv-param-matched-to-ds` **必须保持与原生 L-all-conv 相同的
     stage-shared A + layer-specific B 拓扑**，只缩各生成器 MLP 的 hidden/projection width；**不得改成 global-trunk**
     （否则同时改容量与共享方式）。若必须改 global-trunk 才能达标，则改称「范围＋共享拓扑联合敏感性」、不再作容量匹配旁证。
4. **可行性预表**：训练前产出各结构「最小输出投影参数 ≥? 目标预算」可行性表并冻结目标；若某结构最小
   投影已超目标预算，记录处理方式（放宽到就近可达值并标注），不硬凑。
5. **五容量量并报（外部评审 #3/#四1；「参数个数相似」≠「有效容量匹配」；§4 主轴 cell 一律不含 fc）**：
   ① 可训练生成器参数量；② 每样本发射因子坐标；③ 实现调制张量元素数；④ 可识别低秩流形 DoF（**S-indep 用
   `Σ_l r(C_in+C_out−r)`；S-sharedA 用 stage-coupled `Σ_s[r·C_out,s+Σ_{l∈s}r·C_in,l−r²]`**）；⑤ FLOPs/峰值显存。
   **§4 cell 冻结实测（rank=4，L-all-conv 不含 fc；生成器参数列为 HAM(Age) 口径，§4 首战在 HAM；括号 = CXR(Patient)）**：

   | cell | 生成器参数 HAM(CXR) | 发射坐标 | 调制元素 | 低秩流形 DoF |
   |---|---:|---:|---:|---:|
   | **S-sharedA ≡ L-all-conv** | **2,495,552**（2,499,680） | **19,200** | **1,392,640** | **19,136**（stage-coupled） |
   | **S-indep ≡ L-all-conv** | 待新实现统计（HAM/CXR 各计） | **34,304** | 1,392,640 | **34,000**（每层独立） |
   | L-all+head（= 完整 HyperAdapt，**含 fc**，§3 桥接，非 §4 主轴） | 2,760,260（2,764,388） | 21,252 | 1,393,152 | 19,648 |

   > **⚠️ §0.2 表的「HyperAdapt」行 = 命名模型 = L-all+head 位置（含 fc）、CXR 口径 = 2,764,388/21,252/1,393,152/
   > 19,648**；**§4 共享主轴的 S-sharedA ≡ L-all-conv 不含 fc**，须用上表（共享轴首战 HAM = **2,495,552**；CXR
   > 2,499,680），两者勿混。

### 4.4 原模型桥接（锚点 vs 受控，分开命名 —— #3/#13 修正）

统一模型下所有 cell 用 16→128 编码器；原方法的编码器/初始化/共享头未必相同，故**历史锚点**与
**受控桥接**分开命名，锚点只作展示、不承担单因子推断。

> **RQ3 机制归因 = 描述性、非确认性（外部评审第四次 #4，全项目一致口径）**：本节所有桥接对照（HyperHead
> absolute/residual、HyperFusion 容量/形式/初始化、L-all+head vs L-all-conv 等）**均为预先规定的受控机制对照**，
> **报双侧效应 + 未调整 CI、不进 Holm 族、不承担确认性成功判决**。故 RQ3 结论止于「**结果与某机制解释一致**」，
> **不得**声称「确认了 参数化/形式/初始化 导致三模型差异」。承担确认性推断的只有 RQ1（范围）与 RQ2（共享）的
> 预注册主对比。若日后要对某一条机制对照作更强声明，须**单独预注册**为关键次要比较并保证功效。

**4.4.1 HyperHead 桥接（固定 fc）**
- `HyperHead-original`：现有历史结果（锚点）
- `HH-absolute-controlled`：统一编码器下直接生成完整 fc 权重（替换）
- `HH-residual-full-controlled`：生成完整残差 ΔW，加到基 fc
- `HH-residual-factorized-controlled`：= L-head（factor_width=4，复用）

**4.4.2 HyperFusion 桥接（固定 L-ds，四因子分离）**
> **公共控制冻结（#3）**：容量对照与形式对照的**固定初始化 = 统一近零、固定 E_L2 = off**（与主轴 L-ds 一致，
> 使 `HF-lowrank-mul` 严格复用 L-ds）。
> **E_L2 操作精确公式冻结（外部评审第四次 #小5，防误写成 L2 penalty / unit-norm / LayerNorm）**：E_L2=on 时对
> 条件向量 γ 做 `γ' = √d · γ / max(‖γ‖_2, 1e-6)`（d=embed_dim；即投到半径 √d 的球面、切断 magnitude
> proportionality）；E_L2=off 时 `γ'=γ`。与现有 `resnet18_hyperfusion.py` 实现一致。
- 锚点：`HyperFusion-original`（full-additive + MIP + E_L2，历史结果，不作单因子推断）
- **容量对照**（固定加性、近零、E_L2 off）：`HF-full-add-ctrl` vs `HF-lowrank-add-ctrl`
- **形式对照**（固定 rank=4、近零、E_L2 off）：`HF-lowrank-add` vs `HF-lowrank-mul`（= L-ds）
- ⚠️ **manifest 去重（本轮发现）**：`HF-lowrank-add-ctrl`（容量·低秩加性）**≡** `HF-lowrank-add`（形式·加性）
  ——同为 L-ds+低秩 rank4+加性+近零+E_L2 off，**是同一 cell、合并复用不重训**；`HF-lowrank-mul` ≡ L-ds。
- **初始化对照（冻结 3-cell，三 cell 共用同一参数化 = L-ds + full-additive，只改初始化与 E_L2 —— #6/#7）**：

  | Cell | 固定结构 | 输出初始化 | E_L2 |
  |---|---|---|---|
  | HF-init-zero-E | L-ds + full-additive | 近零 | on |
  | HF-init-MIP-E | L-ds + full-additive | MIP | on |
  | HF-init-MIP-noE | L-ds + full-additive | MIP | off |

  **三 cell 全部固定 full-additive**（否则初始化效应会混入参数化差异）。**这 3 cell 的初始化对照 = 预先规定的
  受控机制对照（外部评审第四次 #4：不称「确认性」，无对应 Holm 族）**：初始化效应 = 前两者（zero-E vs MIP-E）；
  E_L2 效应 = 后两者（MIP-E vs MIP-noE）；**报双侧效应 + 未调整 CI，不承担确认性成功判决**。禁止把「MIP+E_L2」
  整体效果解释为初始化效果。
  ⚠️ **修订第八轮「第四格不运行」（manifest 发现）**：第四格「近零 + E_L2 off + full-additive」**≡
  `HF-full-add-ctrl`（容量对照满秩臂），仍会被训练**（作为容量对照），**只是不进初始化受控对照的这 3 cell**——
  并非「不运行」。它复用容量对照的那次训练，不重复提交。

### 4.5 工作项
- [ ] E3.1 S-indep（rank=4）训练 + OOF
- [ ] E3.2 S-global-trunk（rank=4）训练 + OOF
- [ ] E3.3 参数预算匹配可行性表 + 三结构预算匹配版训练
- [ ] E3.4 五容量量统计与并置
- [ ] E3.5 HyperHead 桥接（HH-absolute / HH-residual-full；factorized 复用 L-head）
- [ ] E3.6 HyperFusion 桥接（容量/形式/初始化三对照 + 锚点展示）
- [ ] E3.7 S-indep vs S-sharedA 主对照显著性 + 稳定性（§6.3、§7）

---

## 5. 容量轴（固定 L-all-conv + sharedA，乘性）

### 5.1 rank 扫描
默认扫 rank ∈ {1, 4, 16}，称**低秩容量敏感性曲线**。**rank=32 触发条件（#5，用 validation 避免 test/OOF
适应性）**：仅当 **HAM locked、seed0 的 validation 折均 worst-group：rank16 相对 rank4 改善 ≥ +0.005** 才
补跑。**若改用 pooled-OOF 触发，则 rank32 明确属数据依赖的探索性实验、不进确认性容量结论**（优先 validation 触发）。**明确不含 all-stage
full-direct**（逐样本直生全层完整残差 → 生成器末层数亿参数、显存/优化几何与 rank≤16 不同量级，且直接
全生成与低秩因子生成优化几何不同，不能视为 rank 连续增大的同一模型）。full-direct 只在 head/ds 出现（§4.4）。

### 5.2 术语（#12 修正）
rank 容量曲线**仅针对卷积条件化主轴**。二分类 fc 权重形状 1×512、矩阵秩≤1，head 里的「4」是 A/B
因子生成的**潜变量宽度 `factor_width`**，与 conv 的 rank 不同义；L-head vs L-ds 是**性质对照**、
不用于同 rank 容量推断。

### 5.3 工作项（外部评审第四次 #小3：编号改 E5.x，避免与 §11 的 E4=M→C 冲突；rank/容量属 §11 的 E5 阶段）
- [ ] E5.1 rank=1；E5.2 rank=16（rank=4 复用 S-sharedA）；E5.3（触发才跑）rank=32
- [ ] E5.4 容量曲线拟合与解释

---

## 6. 机制诊断

置换/常量/knockout/偏移范数为 eval 时复用 checkpoint、零额外训练；伪属性臂需新训。

### 6.1 逐层偏移范数（升级）
- **主指标 = 有效偏移** ρ_l = ‖W_l ⊙ M_l‖_F / ‖W_l‖_F（乘性相对原权重的实际改变量），fc 加性用 ‖ΔW‖/‖W‖。
- 辅助 ‖M_l‖_F / √(C_out·C_in)。
- **群体差异 D_l（须归一化 + 更名，外部评审 #三.3）**：偏移大但各组几乎相同 ⇒ 用的是容量、非属性差异。
  - **主指标必须相对化**：`D^rel_l = E_i ‖ΔW_l(a) − ΔW_l(a')‖_F / ‖W_l‖_F`——不同 stage 通道数差异巨大，
    原始 Frobenius norm 会随矩阵规模天然增长、无法形成有意义的深度曲线；原始 D_l 仅作附录。
  - **更名（避免误称因果）**：把「固定图像+另两属性、只切当前属性」的操作称 **「受控属性输入切换
    (controlled attribute-input contrast)」**，**不称「反事实(counterfactual)」**——更换 race/sex 而保持医学
    影像不变**没有真实因果解释**，只是模型输入敏感性实验。
  - **HAM（单属性 age）**：观察性即无混杂，`D^rel_l = E_a‖ΔW_l(a) − 均值_a ΔW_l‖_F / ‖W_l‖_F`。
  - **CXR（sex/race/age 三属性）**：**主分析用逐样本受控输入切换**——固定图像与另外两个属性，只替换当前
    属性的合法取值，算生成偏移的组间 dispersion（相对 ‖W_l‖_F），再对**患者等权**平均。例：
    `D^rel_{l,sex} = E_i‖ΔW_l(M, r_i, a_i) − ΔW_l(F, r_i, a_i)‖_F / ‖W_l‖_F`。直接按观察到的组求均值会把 sex
    差异混入 race/age 分布差异，故观察性口径仅作附录旁证。
  - **口径歧义显式声明**：ρ_l 与 D_l 均**先逐样本求 Frobenius norm 再对单位平均**（非「先平均偏移再求
    norm」——两者解释不同）。
  - **主结果按群体等权平均**（避免大群体主导），另附样本频率加权；joint 组合仅在样本充足时补充。
- **stage-knockout（含 BN 重校准对照 —— #11 修正）**：test 时逐 stage 关 adapter。因关闭会改变中间
  特征分布而 BN running-stats 仍是 adapter-on 时积累的，故每个 knockout 报**两个**数：① raw knockout；
  ② 关 adapter 后重估 BN 统计再评估。**BN 重估数据严格限定（#8）**：ID 用**本折训练集**；**OOD（M→C）
  只能用 MIMIC source-train 重估、禁用 CheXpert target-test**——若用 CheXpert 无标签数据即变成 test-time
  adaptation / UDA，须另立 regime。**BN 重估显式流程冻结（外部评审 #三.4 + 第六次 #3：必须 `momentum=None` 才是
  cumulative，否则仍是 EMA——仅清 `num_batches_tracked` 不切换累计模式）**，对所有 BN 层：

  ```python
  model.eval()
  for bn in batch_norm_layers:          # 仅 BN 层
      saved_momentum[bn] = bn.momentum
      bn.reset_running_stats()          # running_mean/var 复位 + num_batches_tracked=0
      bn.momentum = None                # ★ 切换为 cumulative（否则默认 EMA，非单遍累计）
      bn.train()                        # 只 BN 进 train mode，采集统计
  with torch.inference_mode():
      run_one_pass(source_train_loader) # 单遍、确定性 eval transform、shuffle=False、drop_last=False、batch=128
  # 恢复 bn.momentum 与各模块 train/eval 模式
  ```

  **遍数 = 1 pass**；措辞「**单遍累计 BN 统计**」而非「精确总体方差」——cumulative BN 是对 batch 统计累计，末 batch
  尺寸不同时不严格等于全样本总体矩（若要严格总体须自累计一/二阶矩，本方案不要求，故 **batch size 冻结=128、
  写入 manifest**，因 batch/末 batch 影响结果）。knockout 只作**后验必要性诊断**，**不替代**从头训练的范围消融。

### 6.2 属性来源五臂（子群指标一律用真实属性分组 —— #10 修正）

| 训练 | 评估 | 目的 |
|---|---|---|
| 真实 A | 真实 A | 主模型 |
| 真实 A | 置换 A | 是否依赖正确属性 |
| 真实 A | 常量 A | 个性化是否必要 |
| **伪 A** | 伪 A | 随机分组动态条件化：随机条件的噪声/正则效应 |
| **常量 A** | **常量 A** | **static-adapter 架构对照**：保留生成器与额外参数、但只用一个条件点（生成器变静态 adapter） |

> **#10/#7：伪 A 训练不单独等于「额外容量效应」**——它同时含「多一条条件通路」＋「按随机组动态变参」＋
> 「随机条件噪声/正则」。故加**「常量 A 训练」臂**，称 **static-adapter architecture control**（生成器参数
> 都在、但只访问一个条件点，embedding 未访问方向无有效功能，≈由生成器参数化的静态 adapter）。三个差值
> **收严表述**：真实 A − 常量 A 训练 = **动态真实属性条件化的增量**；伪 A − 常量 A 训练 = **随机动态条件化的
> 增量/干扰**；常量 A 训练 − ERM = **静态 adapter / 额外参数化的增量**（**不称「纯参数容量效应」**）。
> **评估时的常量 A ≠ 常量 A 训练**（前者测已训模型去个性化，后者从头训练），二者不可互替。
>
> **常量 A 训练用专门可学习 null 条件（避免任意选真实类别 + 防退化，#8）**：若统一喂「年轻/男性/某 race」，
> 结果会受该类别 embedding 初始化与语义影响；而若向**无偏置、齐次**的生成器输入**全零向量**，输出可能永久
> 为 0 ⇒ static-adapter 退化成 ERM（并非「静态 adapter」）。故用**一个独立的、可学习的
> `null_condition_embedding`**：**相同 null token 定义 + 相同初始化规则 + 每个训练运行独立优化 + 该运行内
> 所有样本共享同一 null embedding**（可学习向量经训练后不可能跨 fold/seed 数值完全相同，故**不要求跨运行
> 数值一致**；若确需跨运行完全一致则改用固定非零向量而非可学习 embedding）。既不对应任何真实类别，又能形成
> 真正非零的静态 adapter。**E0.4 单元测试须断言：常量-A 训练后 adapter 偏移可非零**（否则等于没加）。
> 而真实-A 模型的**常量-A 评估**仍可遍历所有合法属性值，用于分析预测对参考类别的敏感性（两者用途不同）。

- **子群划分永远用真实属性**，不得用被置换后的属性分组。
- **置换单位 = 病灶/患者（外部评审，硬约束）**：**HAM 病灶级、CXR 患者级置换**——同一患者多图/同一病灶多图
  必须整组分到同一被置换属性值，否则会给同一患者输入自相矛盾属性、产生不现实伪条件；**所有模型与所有重复
  用相同置换索引**（配对）。
- **CXR 置换须分解**：单独置换 sex / race / age、联合置换全部、以及各属性的所有常量值；**置换重复
  次数冻结 = 20（`n_perm=20`）、报分布**，不只跑一次随机打乱。
- **伪属性生成——HAM 与 CXR 分别定义（#4 修正，原「患者级固定 + 每图标签层等比例」在 CXR 不可同时满足，
  因同一 CXR 患者可同时有 No-Finding 阳/阴图）**：
  - **HAM**（病灶标签固定）：按病灶赋值，**在标签层内按相同类别比例分配**，P(Ã|Y=0)≈P(Ã|Y=1) 由构造成立。
  - **CXR**：**患者级约束随机化**——① 每患者只一个伪属性；② 伪属性边缘分布 ≈ 真实属性；③ 优化/重复随机
    分配，使各伪属性组的阳/阴图数、患者阳性率分布尽量平衡；④ 设接受阈值（image-level MI、患者阳性率
    标准化差 均 < 预设值）；⑤ 不达标就重新生成。
  - 公共约束：类别数同真实、边缘频率近似、训练内固定不逐 epoch 重采样、train/val/test 各自独立生成；
    MI / Cramér's V 仅作事后验证、非独立性唯一依据。
- **伪属性 seed 与训练 seed 一一配对（冻结值）**：伪属性 seed 与训练 seed **按 0,1,2 一一配对、不做笛卡尔积**。
  伪/static-A 训练臂**仅在 HAM 跑、仅 L-ds 与 L-all-conv 两结构、seed {0,1,2}**，总计固定
  **2 结构 × 2 臂 × 3 seed × 5 折 = 60 run**（§8.2/§9）。

### 6.3 训练稳定性 = fold × seed **交叉**，逐 cell 方差（#9 修正）

所有 fold 用相同若干 seed ⇒ fold 与 seed 交叉。**须分别回答两个问题、都报告**：
1. **绝对稳定性**：每个 cell 自身 metric 的**逐 cell 跨 seed 方差**（不能只用全 cell 共享的 v_s，
   那只估共同 seed 波动，无法比较 σ²_seed,indep vs σ²_seed,sharedA）。**方法冻结（外部评审第四次 #小6，指定
   主方法、去二选一）**：**主方法 = 直接比较每个 fold 内各 cell 的跨 seed 样本方差**（对 fold 取平均、bootstrap
   over fold 附 CI）；**敏感性分析 = cell-specific seed random-effect 方差分量模型**（混合模型），**不作主判据**——
   避免共享降方差时在两法间选报。
2. **效应稳定性**：cell 相对同一参照的配对差 Δ_{c,f,s}=M_{c,f,s}−M_{ref,f,s} 的方差（配对抵消 fold
   难度）。

- 方差分量**附 CI**；**不以点估计声称**共享降方差。5 folds×3 seeds 估计偏少，**若稳定性是核心主张，
  共享轴加到 5 seed**。
- **参照模型须同 fold×seed 存在（#3，硬约束）**：配对差 Δ_{c,f,s}=M_{c,f,s}−M_{ref,f,s} 要求 ERM（及
  作 OOD 基线的 SWAD）在**每个 cell 用到的相同 fold 与相同 seed**下都有预测，否则 seed 方差无法配对
  抵消、也无法判断共享是否比 ERM 更稳。故 **HAM ERM 须按与主 cell 相同的 5 folds × 5 seeds 重训**、
  **MIMIC ERM 补齐与其确认相同的 seeds**、**SWAD 从对应 ERM seed 轨迹派生**；缺失 seed 必须补训、计入
  预算（现有 CV-OOF ERM 为每 (config,fold) 单 seed，不足）。
- worst-group 身份跨 fold 可变，故同时报：① 模型自身 worst-group AUC；② 固定 ERM-worst 群上的 AUC；
  ③ 每真实群体 AUC 向量；④ worst 群身份跨 fold 变动频率。

### 6.4 工作项
- [ ] E1.1 有效偏移 ρ_l / 群体差异 D_l / 深度曲线落盘
- [ ] E1.2 stage-knockout（raw + BN 重校准）评估路径
- [ ] E1.3 置换（分属性+联合+常量，多次重复）/ 常量 A 评估路径（子群按真实 A）
- [ ] E1.4 伪属性生成器（分组单位赋值 + 标签层内等比例 + 关联事后验证）+ **伪-A 训练臂 + 常量-A 训练臂**（§6.2 五臂）
- [ ] E1.5 fold×seed 交叉方差：逐 cell 绝对稳定性 + 配对差效应稳定性（均附 CI；ERM 须同 fold×seed）
- [ ] E1.6 worst-group bootstrap 估计量预固化（每次重算 argmin=主；固定 ERM-worst 群=次要；身份变动频率）

---

## 7. 统计方案（预注册，#7/#8 修正）

### 7.1 终点层级（预注册，避免多重成功自由度）

两个 regime 各设一个主终点（分开预注册，不合并成功判定）：
- **主终点 A（ID 信号）**：HAM 上 **age 的 marginal worst-group AUC**。
- **主终点 B（OOD 正向）**：**MIMIC→CheXpert 的 marginal worst-group AUC**（相对 ERM **与** SWAD）。
  —— 这是 **legacy k→k 口径下唯一显示稳健正向、待 full-target 复核（§11）的候选正向 regime**（§0.3），也是
  范围/共享效应最真实的落点；本方案要回答该（待复核的）胜利
  **需要多大范围、何种共享**。
- **主要效用约束（非劣，参照 = 被比较模型本身，见 §7.3 表）**：结构确认性比较的 Overall 非劣参照是
  **被比较的结构模型**（p_noninf 用 L-all-conv 不劣于 L-ds、S-sharedA 不劣于 S-indep），已并入该比较的
  p_joint。**「同时不劣于 ERM」的地位（#6）= 关键次要效用检查，非确认性成功门槛**：结构确认性成功**仅由对应
  结构比较的 `p_joint_holm` + 点估计阈值决定**；相对 ERM 的 Overall 非劣**除非预先加入对应 Holm 族，否则不作
  额外确认性否决**（防结果出来后自由选择是否用 ERM 非劣否决结构结论）。OOD 中 cell vs ERM/SWAD 的 p_joint
  各自已含对应基线非劣。
- **次要终点**：全部 marginal worst / canonical worst / gap 及其分解 / seed 方差 / Overall 增益 /
  C→M 反向（负控制方向）/ ID 其余数据集。

### 7.2 判决规则（预注册；显著性由 `p_joint_holm` 把关，CI 描述方向与幅度 —— #4 统一命名）
**p 值命名统一（全表一致，防止误用 fairness-only Holm 再单独看 Overall CI）**：
- `p_fair` = worst-group 单侧 p；`p_noninf` = Overall 非劣单侧 p（T_NI=ΔOverall+0.005, H0:T_NI≤0，同 bootstrap 索引）；
- **`p_joint = max(p_fair, p_noninf)`**（intersection-union）；**`p_joint_holm`** = p_joint 在预注册族内的 Holm 校正值。
- **每个确认性比较报告**：配对 cluster-bootstrap 效应 + 未调整 95% CI（描述）、p_fair、p_noninf、p_joint、p_joint_holm。

- **两类成功分开定义（外部评审 #2：历史胜利 +0.006~0.008 不可能过 +0.01 门槛，不能与结构确认同尺）**：
  - **结构性确认成功**（范围/共享主对比、论文主结论）须**同时** **`p_joint_holm < 0.05`** ＋ **点估计 ≥ +0.01**。
  - **历史统计胜利复现**（判定 cell 是否保留历史 OOD 胜利）**仅须** **`p_joint_holm(vs ERM) < 0.05` 且
    `p_joint_holm(vs SWAD) < 0.05`**（p_joint 已含各自 Overall 非劣）；**不要求 +0.01**，是否达 +0.01
    **单独分级报告**（见 §7.3 ②）。
  - `CI 下界 > 0` 仅作方向稳定性描述，**不单独承担显著性判决**。
- **稳健实质改善**（更强、少见）：另加 **未调整 CI 下界 > +0.01**。
- **实质等效**：worst-group 差值 **TOST / 双侧 90% CI 完全落入 [−0.01, +0.01]**。
- 其余 = 不确定。（Holm 调整 CI 实现较复杂，本方案用「未调整 CI 描述 + p_joint_holm 判显著」，全表一致。）

### 7.3 预注册主对比 + 多重校正族（#5：分离 RQ 与「保留历史胜利」）
- **位置主对比**：**L-ds vs L-all-conv**（扩大条件化范围是否有价值）。
- **共享主对比（真正的 RQ2）**：**S-sharedA vs S-indep**（shared-A 是否优于每层独立生成）——这是共享
  轴**唯一的主统计假设**。
- 两个主对比在两个主 regime（HAM-ID、M→C-OOD）各预注册一次；OOD 里另加 **L-head vs L-all-conv**
  （HyperHead 无益 vs backbone-level 取胜的范围落差，§0.3）。
- **OOD 成功分两级（历史胜利 +0.006~0.008 < +0.01 阈值，不能用同一条尺）**：
  - **① 历史统计胜利复现（交集，用 p_joint_holm）**：**p_joint_holm(vs ERM) < 0.05 且 p_joint_holm(vs SWAD)
    < 0.05**（每个 vs-基线比较的 p_joint 已含各自 Overall 非劣、再在历史复现族内 Holm；两比较均须过）。
    = 「复现统计正向且同时胜两基线」——判定 cell 是否保留历史 OOD 胜利。
  - **② 达实际意义阈值**：进一步 **相对 ERM 与 SWAD 的点估计均 ≥ +0.01**（外部评审 #1：两基线都须达阈值，
    否则「同时胜两基线」的实质意义含糊；不采「仅对更强基线达阈值」）；更强版 **未调整 CI 下界均 > +0.01**。
  - 报告时**诚实三分**：统计稳健但幅度小（仅①）/ 点估计达阈值 / CI 整体超阈值。历史 M→C 胜利属①、未达②。
- **两类问题不混为一个主假设**：`S-sharedA vs S-indep` 是**共享 RQ 的主对比**；`vs ERM/SWAD` 是**判定该
  cell 是否保留历史 OOD 胜利的辅助条件（用①级）**，二者分列、不并入同一假设。
- **Holm 校正族逐项冻结（#2，不到分析时再定；每假设为单侧「前者更优」）**：

  | 族 | 成员比较 | m |
  |---|---|---|
  | HAM 范围确认 | {L-all-conv − L-ds} | 1 |
  | HAM 共享确认 | {S-sharedA − S-indep} | 1 |
  | OOD 范围确认 | {L-all-conv − L-ds, L-all-conv − L-head} | 2 |
  | OOD 历史胜利复现 | **{L-ds, L-layer4, L-layer34, L-all-conv} × {ERM, SWAD}** | **8** |
  | OOD 共享确认 | {S-sharedA − S-indep} | 1 |

  **去重（#2）**：`S-sharedA vs ERM/SWAD` 只在**历史胜利复现族**（因 S-sharedA≡L-all-conv，已含
  L-all-conv×{ERM,SWAD}）里检验，**不再进 OOD 共享族**——同一比较不在两族各校正一次。共享模型「是否保留历史
  胜利」**复用历史复现族的校正结果**。**L-head 不进有序 backbone 阶梯族**（只作 head-only 参照，§7.6）。
  确认性比较均**单侧**（前者更优）；其余增量分析（Δ_stage* 等）用双侧或仅报 CI、不进确认性族。
  > **FWER 控制范围声明（外部评审第七次 #小3，须写进论文方法）**：**FWER 在每个预注册 RQ×regime 族内控制**
  > （上表逐族 Holm），**不是**对全文所有确认性声明做单一全局 FWER 控制；跨族解读须注明这一多重性边界。

- **Overall 非劣的多重性 + 参照对象（#3）**：对每个确认性比较定义**联合 p** `p_joint = max(p_fairness,
  p_noninf)` 并**对 p_joint 做 Holm**（点估计 ≥ +0.01 作额外效果量门槛）。**非劣参照 = 被比较的模型本身**
  （结构对结构，非一律 vs ERM）：

  | 公平比较 | Overall 非劣参照 |
  |---|---|
  | L-all-conv vs L-ds | L-all-conv 不劣于 **L-ds** |
  | S-sharedA vs S-indep | S-sharedA 不劣于 **S-indep** |
  | cell vs ERM | cell 不劣于 **ERM** |
  | cell vs SWAD | cell 不劣于 **SWAD** |

  「同时不劣于 ERM」作为**另设的绝对效用要求**、不与结构非劣混同。**非劣 p 公式**：`T_NI = ΔOverall + 0.005`，
  检验 `H0: T_NI ≤ 0`，**与 fairness p 用同一批 cluster-bootstrap 索引**。

### 7.4 显著性检验（cluster bootstrap + p 值算法 + worst-group 估计量口径 —— #3/#6/#8/#11）
- **主推断 = 患者/病灶级 cluster bootstrap**（同一单位可有多图，按单图重采样会低估不确定性）：
  **HAM 病灶级、CXR 患者级、Fitzpatrick 图像级**；**所有 cell 用相同 cluster 与相同重采样索引**（配对）。
- **单侧 p 值算法冻结（#3，写进代码规格）**：① 配对 cluster bootstrap；② 用观察效应对 bootstrap 分布做
  **零假设中心化**（`T*_null = Δ* − Δ_obs`）；③ 单侧 p 用 **+1 修正**：`p = (1 + #{T*_null ≥ Δ_obs}) / (B+1)`；
  ④ **固定 bootstrap seed、所有模型同一批 cluster 索引**；⑤ **确认性分析 B = 5000（固定，默认不提前停止）**——
  含 MIMIC；**E-audit 优先靠向量化 + 预计算 group indices + 并行把 5000 次运行时间压到可接受**（这是首选方案）。
  - **仅当 MIMIC 5000 固定确实不可行，才用严格族级顺序停止（#3，唯一算法、消除隐性选择）**：① MC p 用
    **Clopper–Pearson 区间**；② 检查节点 **{1000,2000,3000,4000,5000}** 与 **max B=5000** 预先固定，对「5 节点 ×
    该 Holm 族全部假设」做 **Bonferroni 覆盖校正**的区间；③ **仅当该 Holm 族的所有假设的最终拒绝/不拒绝状态，
    在其 p 值区间任取值下均不改变时才停**（即整族判决稳定，非单假设稳定）；④ 否则**必运行到 5000**。
    **禁止**「结果刚好显著就停」。
- **marginal worst 候选群体集合与口径冻结（#1/#6，资格只按真实标签/属性、不按模型）**：
  - **口径原则：候选集合由 regime 固定，与 cell 是否条件化无关（#1 修正）**。ERM/SWAD 不条件化任何属性，但
    须用与所有 HN cell **完全相同**的公平指标；故**不得**用「该 cell 条件化的轴」表述。
    - **HAM-ID（主终点 A）**：`𝒢_HAM = {20-40, 40-60, 60-80, 80+}`，`worst_A = min_{g∈𝒢_HAM} AUC_g`（age 4 组）。
      **sex marginal 属次要分析**、不进主终点。（订正原「6 组」笔误。）
    - **M→C（主终点 B）**：`𝒢_{M→C} = 𝒢_sex ∪ 𝒢_race ∪ 𝒢_age = {M,F}∪{White,Non-White}∪{<60,≥60}`，
      `worst_B = min_{g∈𝒢_{M→C}} AUC_g`。
    - **同一 regime 内 ERM、SWAD、所有 HN cell、所有 seed 共享同一 𝒢**（否则 worst-group 不可比）。
  - **`unknown/other` 默认不进候选**（主分析排除；预注册，不看结果决定）。
  - **最低样本阈值 = n_pos ≥ 20 且 n_neg ≥ 20（暂定，E-audit 后正式定，一旦开跑不改）**；不达标的群不进候选。
    **E-audit 调门槛只能依据真实标签/属性计数、有效 bootstrap 率、数值可计算性（#5）；禁止依据「哪个阈值下模型
    差异更大/p 更显著/CI 更有利」**——否则门槛沦为数据驱动的结果筛选。
  - **所有模型 / seed / bootstrap 共用同一候选集合**（否则 worst-group 不可比）。
- **worst-group bootstrap 估计量必须预先固化（#11）**：
  - **主终点**：每次重采样**重新计算 marginal argmin**（`min_g AUC_g^(b)`，模型自身 worst，群体身份可跳变）。
    **条件切换（预注册，外部评审第三次 #3；主判据用 `p_fair`，见 §11）**：**若 §11 E-audit 的动态-min 校准自检
    不合格**（`Pr(p_fair<0.05)` 的 95% 二项 CI 上界 > 0.06），则确认性主终点**切换为下方「固定 ERM-worst 群」**、
    动态 argmin 降为描述性——单一 fallback、不留双出口。**再兜底（外部评审第七次 #4）**：若门槛序列 20/30/40 全失败
    或 `g*` 自身有效重采样率 <95%，该 regime **不作确认性 worst-group 推断、只报描述性**（§11 全失败最终出口）。
  - **关键次要终点（亦为校准不良时的确认性 fallback）= 固定 ERM-worst 群 `g*`（外部评审第四次 #3，精确定义、
    杜绝群体选择自由度）**：
    - **选择数据 = legacy ERM 预测**（**在新 cell 训练前**用现有 ERM 预测冻结 `g*`，之后**不随新 canonical ERM /
      seed / bootstrap 重新选择**——避免在同一 test/target 上「选群 + 检验」的循环忽略群体选择不确定性；推断解释为
      「对预先选定弱组的条件性效应」）。
    - **公式须匹配 legacy 文件结构（外部评审第五次 #2：legacy ERM 是每 fold 单 run、`seed=42+fold`、fold 与 seed
      绑定，不存在 crossed F×S 网格，故不能对不存在的 seed 维求平均）**：
      - **HAM**：用 legacy **pooled-OOF** ERM 分数——`g*_HAM = argmin_{g∈𝒢_HAM} AUC_g(y_OOF, p^OOF_{ERM,legacy})`
        （每样本仅一个 held-out fold 预测，**不对 seed 平均**）。
      - **M→C**：**先完成 E-audit-OOD**（5 个 legacy source-fold ERM checkpoint 各预测完整 target），再
        `g*_M2C = argmin_{g∈𝒢_{M→C}} (1/5)Σ_{f=1..5} AUC_g^{ERM,legacy,f}`（**仅对 source fold 平均、无 seed 维**）。
        用 full-target legacy ERM 冻结 OOD 弱组，也与「OOD 确认性主口径 = full-target」一致。
    - **平局**：并列时取候选集合 `𝒢` 预注册顺序中的**第一个**（`𝒢` 顺序本身预注册、不看结果）。
    - **操作顺序冻结**：① 定候选集合 + n_pos/n_neg 门槛 → ② 完成 legacy full-target 推理（E-audit-OOD）→
      ③ 冻结 `g*` → ④ 才开始新 canonical ERM 与新 cell 训练。
    - **按 regime 分别执行**：HAM 与 M→C **各自独立校准、各自独立决定是否切换**（不联动）。
    - **解释边界（外部评审第七次 #小4，写进论文）**：`g*` 用 legacy ERM 在**同一** HAM/CheXpert 样本上选弱组、
      再在该样本上推断，**仍不含群体选择不确定性**；只能称「**对预先选定弱组的条件性效应**」，**不得**把 fallback
      描述为完全无选择偏差的独立确认。
  - 同时报 worst-group **身份变化频率**。二者在代码中**预先固化**，不在分析时临时选择。
  - **无效重采样处理（#9，代码固化）**：某 replicate 使任一候选群体无阳/无阴 ⇒ AUC 不可算 ⇒ 该 replicate
    **判无效、重抽**直至得到预设数量有效 replicate；**报告有效重采样率**；**有效率 <95%（即无效率 >5%）时该
    worst-group CI 不作确认性证据**。（HAM 主终点仅在 age marginal **4 组**上，出问题概率低，但仍须固化。）
- **CI 构造方法冻结 = percentile（#2，非 basic/BCa/正态）**：确认性效应用 **cluster-bootstrap percentile CI**——
  95% CI 取 bootstrap 效应分布的 **2.5% / 97.5% 分位数**；TOST 用 **5% / 95% 分位数**构成双侧 90% CI。
  （percentile 简单可复现；不选 BCa 以免额外 cluster 级 jackknife 成本。）
- gap 做 **Δgap = Δbest − Δworst** 分解，防止把「压低强组」误判为公平改善。
- **DeLong 作辅助**（其默认观测独立，CXR 上违反，主推断用患者级配对 cluster bootstrap）。

### 7.5 多-seed 主效应与 CI（estimand 唯一冻结 —— #1/#2）
主表里的单个 ΔAUC 与 CI（**禁止**同患者多 seed 预测纵向拼成 3×/5× 独立样本、禁止临时选最好 seed、
禁止无说明地平均 logit）：
- **确认性主 CI（唯一方法，不留分析后选择自由度）**：每个 cluster-bootstrap replicate 内用**同一组预注册
  seed**，分别算每个 seed 的配对效应、对 seed **等权平均**；**bootstrap 只重采样患者/病灶 cluster、不重采样
  seed**。点估计 **Δ̄ = (1/S)Σ_s Δ_s**。解释：**「给定预注册训练 seed 集合时，平均训练运行效应的测试样本
  不确定性」**。（**删除**上一版「若需要…再对 seed 外层重采样 **或** 混合模型」的可选项——seed 数仅 3/5、
  可重采样单位过少，seed bootstrap **不作主 CI**。）
  - **OOD（full-target）特化（外部评审 #1）**：ID 只对 seed 平均；**OOD 对 `(fold f, seed s)` 双重等权平均**
    `Δ̄ = mean_{f,s} Δ_{f,s}`，bootstrap 只重采样完整 target 患者 cluster、**不重采样 (f,s)**（同 §8.2）。
  - **OOD 主 CI 的推断边界（外部评审第六次 #统计，须写进论文方法）**：OOD 主 CI **条件于固定的 MIMIC source
    cohort、fold 与预注册 seed 集**，**主要反映 CheXpert target 患者的抽样不确定性**，**不覆盖重新抽取 source
    cohort 的不确定性**（非设计错误，是必要的推断边界声明）。
- **训练随机性另行报告（稳定性/敏感性，不进主 CI）**：seed-specific 效应、fold×seed 方差分量、以及平均
  效应的混合模型 CI 作为补充；样本不确定性与训练随机性**不混成一个 CI**。
- **次要结果 = seed ensemble**：同样本平均多 seed 的 logit 后再算 AUC，**明确标 ensemble-OOF**，不与主结果混称。

**共同 seed 交集规则（#2 + 外部评审第三次 #1：同一「族/判定」内所有 cell 必须用同一 seed estimand，不能因
L-all-conv 兼作 S-sharedA 而独享 5 seed）**：不同 cell 的 seed 数不同（范围 cell 3、共享 cell 5、ERM 5），配对
差**必须用同一 seed 集**，且**按分析冻结、不按可用 seed 数临时取交集**：

| 分析 | seed 集（冻结） |
|---|---|
| OOD 范围主对比（L-ds/…/L-all-conv vs 基线或彼此） | **{0,1,2}** |
| **OOD 历史胜利复现族（m=8）** | **{0,1,2}（全 cell 统一，含 L-all-conv）** |
| **最小有效范围判定** | **{0,1,2}（全 cell 统一）** |
| HAM 范围主对比 | {0,1,2} |
| 共享主对比 S-sharedA vs S-indep | {0,1,2,3,4} |
| L-all-conv / S-sharedA 额外稳定性（§6.3） | {0,1,2,3,4}（**仅稳定性，不进上面任何配对/复现判定**） |

- **关键**：L-all-conv 在历史复现族与最小范围判定里**只用 seed {0,1,2}**（`Δ̄_c = (1/15)Σ_{f=1..5}Σ_{s=0..2}
  Δ_{c,f,s}`），使 4 个范围 cell 的 estimand 完全对齐；seed 3/4 仅供共享主对比与稳定性诊断，**不得**让 L-all-conv
  在与 L-ds/L-layer4/L-layer34 同场比较时获得更稳的 5-seed 均值。
- 所有配对差用**相同 fold、相同 seed、相同样本**。分析脚本须对「同一族/判定内所有 cell 的 seed 集合完全一致」
  下 assertion（不只是「配对两侧一致」）。

### 7.6 探索性定位与 RQ 地位（报告规范 B + #4 修正：最小范围只在 backbone 阶梯内）
- **最小有效 backbone 范围（L-head 不进有序阶梯）**：只在**严格嵌套的 backbone 阶梯**
  `L-ds ⊂ L-layer4 ⊂ L-layer34 ⊂ L-all-conv`（外部评审补齐 L-layer34）内找 **最小有效 cell =
  `p_joint_holm(vs ERM) < 0.05` 且 `p_joint_holm(vs SWAD) < 0.05`（p_joint 已含各自 Overall 非劣）且**点估计
  相对 ERM 与 SWAD 均 ≥ +0.01**（外部评审第三次 #内2：与 §7.3 ②级一致，两基线都须达）的最小测试 backbone cell**。
  - **范围阈值 = suffix-success 规则（外部评审第三次 #内3，严格定义）**：存在某档 `k∈{ds,layer4,layer34,all-conv}`
    使得**所有比 k 小的范围全失败、k 及所有更大范围全成功**，则称 k 为**有序范围阈值**（例：ds/layer4 失败、layer34/
    all 成功 ⇒ 阈值=layer34；四档全成功 ⇒ 阈值=ds）。
  - **非 suffix-monotone**（如 ds 失败/layer4 成功/layer34 失败/all 成功）⇒ 只报「最小通过标准的**已测试 backbone
    cell**」、**不称范围阈值**。
- **head-only 对照单列**：`L-head vs L-all-conv` 回答「head-only 与 backbone-level 是否不同」，是**另一维
  性质结果**，**不参与**「最小范围」的有序判定（head/backbone 性质不同、不构成嵌套）。
- **RQ2 稳定性地位（报告规范 B）**：**shared-A 对 worst-group AUC 的影响 = 确认性结果**；**seed 方差是否
  下降 = 次要机制证据**。**不得仅因方差下降就宣布 RQ2 得到正向公平结论**。

---

## 8. 数据集范围与执行优先级

评估沿用**各数据集现有的无泄漏 CV 划分 + pooled-OOF**（HAM 病灶级 GroupKFold、CXR 患者级分组、
Fitzpatrick 图像级 StratifiedKFold —— **不统称 GroupKFold**）。

### 8.1 第一优先级 A：HAM10000（age，ID 信号主战场）
完整跑：范围轴（§3，含 L-all+head 桥接）+ 共享轴（§4，含预算匹配 + 5-seed 稳定性 + 两桥接）+
rank（§5）+ 全部诊断（§6）。ID 下唯一清晰 `I(Y;A|X)>0` 处；主终点 A。

### 8.2 第一优先级 B：MIMIC（ID）→ CheXpert（OOD 候选正向，主终点 B，待 full-target 复核）
**这是 legacy k→k 口径下唯一显示稳健正向、待 full-target 复核（§11 E-audit-OOD）的候选正向 regime（§0.3），与
HAM 并列一等。** 在 MIMIC（ID）上训练**缩减网格**，再逐折
source→target 池化做 M→C OOD 评估（复用 `src/training/run_ood_cxr_cv.py`，CV-OOF 口径，非旧
`eval_ood_cxr.py`）。**缩减网格 = 6 个唯一 cell（外部评审：补 L-layer34 补齐严格嵌套阶梯）**——因范围轴
本身固定 sharing=sharedA，故 **S-sharedA ≡ L-all-conv**，两轴共用该 cell：
1. L-head　2. L-ds　3. L-layer4　4. **L-layer34**　5. **L-all-conv（= S-sharedA）**　6. S-indep
- **范围阶梯**（cell 1–5）：`L-ds ⊂ L-layer4 ⊂ L-layer34 ⊂ L-all-conv` **严格嵌套**，回答 M→C worst-group
  稳健赢**从哪一档范围开始出现**、增益究竟来自仅 stage4 / 增 stage3 / 还是须增 layer1–2——补 L-layer34 后
  才可正当称「最小有效范围」（否则只能称「最小已测试粗粒度配置」）。M→C 是全项目 **legacy k→k 下唯一显示稳健
  正向、待 full-target 复核的候选正向 regime**，不宜在此省。
- **共享**（cell 5 vs 6）：S-sharedA vs S-indep（OOD 泛化下 shared-A 是否为稳健赢的必要成分）。
- **OOD 原模型桥接（外部评审 #三.2：补 RQ3 在 OOD 上的完整性；locked-only、不做六配置搜索）**：范围/共享
  网格只覆盖「统一低秩乘性」族，无法回答**原三种模型为何在 M→C 表现不同**（`L-head`=残差因子化头≠原
  absolute HyperHead；`L-ds`=低秩乘性≠原 full-additive+MIP+E_L2 HyperFusion；`L-all-conv` 缺原 HyperAdapt 的
  fc adapter）。故补 **3 个 OOD 桥接 cell（外部评审第三次 #4：三个全必跑、无「可选」，使三种原方法各有受控 OOD
  bridge、且 cell manifest 在 E0 前可定死）**，**仅 locked 配置 + 共同 seed {0,1,2}**，各 5 折×3 seed = 15 run、
  共 **45 run**：
  - **L-all+head**（近原 HyperAdapt）：L-all-conv **＋ fc adapter** = 完整 HyperAdapt 位置——否则 L-all-conv 若未
    复现 HyperAdapt 的 OOD 收益，无法判断是否只是漏掉 fc adapter。
  - **HF-full-add-MIP-E**（近原 HyperFusion 机制）：L-ds + full-additive + MIP + E_L2。
  - **HH-absolute-controlled**（近原 HyperHead 机制）：统一编码器下 absolute full-head。
  - **地位**：这些桥接**不进确认性 Holm 族**（不做六配置搜索、单 locked），只作 RQ3 归因的**描述性/机制证据**，
    结论表述限「统一编码器下、locked 配置的机制对照」。**若要对 `L-all+head − L-all-conv`（fc adapter 的边际
    OOD 效应）作更强声明，须预注册为一个单独的关键次要比较**（否则一律停在描述性）。
- **诊断臂分配已定死（无运行后扩展）**：**伪-A 训练、static-A 训练两臂仅在 HAM 执行**；**MIMIC 只做
  真实 / 置换 / 常量 A 三个 eval 臂 + 偏移范数 + knockout**（全部复用真实-A checkpoint、零额外训练）。
- **HAM 五臂结构与 seed 冻结（#1）**：结构 = **L-ds 与 L-all-conv**（同 shared-A 机制、只改范围，最适合回答
  「真实属性依赖是否随范围扩大而增强」；共享效应已由 S-indep vs S-sharedA 确认性比较 + 稳定性诊断承担，不再
  让五臂兼第二问）；训练 seed = **{0,1,2}**；每结构各训伪-A 与 static-A ⇒
  **2 结构 × 2 臂 × 3 seed × 5 折 = 60 run**（§9）。
- **OOD 确认性主估计量 = full-target 配对效应（外部评审 #1，取代 k→k 拼接）**：k→k 口径虽保证每个 target
  样本只被预测一次，但 **source fold 与 target fold 无自然对应**——最终 AUC 是「5 个不同模型各服务一批患者」
  的拼接系统性能，不是可部署模型或 ensemble 的性能，换 source–target 排列结果会变。改为：
  - **主估计量**：**每个 source `fold f × seed s` 模型都评完整 target 集**，得配对效应
    `Δ_{f,s} = M(y_target, p^cell_{f,s}) − M(y_target, p^ref_{f,s})`，再对 `(f,s)` 等权平均得点估计
    `Δ̄ = mean_{f,s} Δ_{f,s}`。**患者级 bootstrap 每次重采样完整 target 患者集，在同一索引下重算所有模型的
    效应**（cell 与 ref 用同一批患者、同一 `(f,s)`，配对抵消）。
  - **次要 ensemble 估计量**：先对 5 个 source fold 的 logit 平均 `z̄_s(x) = (1/5)Σ_f z_{f,s}(x)`，再算 target
    AUC（更贴近实际部署的单一系统）；按 seed 报分布。
  - **legacy sensitivity**：保留 k→k 拼接 OOF（现有 `run_ood_cxr_cv.py` 实现）作稳健性旁证，**不再作确认性
    主结果**；历史 M→C「胜利」须在 full-target 新口径下**重新确认**（§0.3 数字来自 k→k，改口径后可能不完全复现，
    E4 须复算并如实报告）。
  - **实现改动**：`run_ood_cxr_cv.py` 增 `--estimand full_target|kk`，full_target 让每个 `(f,s)` checkpoint
    评完整 target（推理量 = cell 数 × fold × seed 次全 target 前向）；加 assertion「主估计量 = full_target」。
- **模型选择只用 MIMIC 的 validation（#9）**：**禁止**看 CheXpert OOD 结果后回选 cell/config；target
  CheXpert 仅用于最终 OOD 评估（否则目标域调参泄漏）。
- **跨数据集属性契约（#2/#5，OOD 专项冻结的具体编码表）**：属性既是**模型输入**又是**分组键**，MIMIC 与
  CheXpert 必须共用同一份映射（**同一 JSON、非两套代码**；引用 `build_{mimic,chexpert}_splits_nofinding.py`
  的编码 + `run_ood_cxr_cv.py`）：

  | 属性 | 模型输入编码 | fairness 候选组 |
  |---|---|---|
  | sex | Male=0, Female=1, unknown=2 | {Male, Female} |
  | race | White=0, Non-White=1, unknown=2 | {White, Non-White} |
  | age | `<60`=0, `≥60`=1, unknown=2 | {`<60`, `≥60`} |

  - **age 统一到 `<60/≥60` 二值**（MIMIC/CheXpert 现均用此阈值，见 oof_regime「age≥60 最弱格」），OOD 两端一致。
  - **race 合并规则**：CheXpert 原始 race 标签按 `build_chexpert_splits` 的 White / Non-White 归并（与 MIMIC 同规则）；
    **现有 split 已在预处理阶段排除 sex/race=Unknown**，故 unknown token 主要作 source-unseen 的安全兜底（当前
    两库排除口径一致、无 source-unseen 类别）。
  - **模型输入可用 unknown，但 unknown 不进确认性 worst-group 候选**（§7.4）；空值/Other/Unknown 统一映射到
    `unknown` token；**所有映射仅由 source 训练前规则决定，禁止看 CheXpert 结果后修改**。
- **反向 C→M（#1 修正，采方案 A）**：C→M 需在 **CheXpert 上训练**对应受控网格（MIMIC checkpoint 无法
  产出 C→M），额外约 200 run。**本方案不训 CheXpert 受控网格**；C→M 仅**复用现有原始三方法的历史
  结果作方法级负控制背景**（验证方向不对称），**不承担受控范围/共享推断**。

### 8.3 第二优先级：null / 负控制（便宜、验证边界）
- **Fitzpatrick17k**（skin，便宜弱信号）：ERM / L-head / L-ds / L-all-conv。目的**不是**预设「一定
  打平」，而是**检验弱信号环境下扩大范围是否无效/微弱有害/意外收益**（现有三 HN +0.004~+0.011 n.s.）。
- **CheXpert ID（默认不跑，条件触发 —— #5）**：默认**不纳入预算/执行**；**仅当 HAM 或 M→C 主结果出现
  确认性范围效应**时，才补 CheXpert ID 端点验证，并**标为条件触发的次要分析**（`I≈0` 大 n 下 HN 显著微劣，
  作「充分性反方向」）。

---

## 9. 算力预算（seed 分配已明确，#1 修正）

**seed 补给对象 = locked 公共配置（情况 B）**，避免主结果无 seed 稳定性：
1. **搜索**：每 cell 6 配置（lr×wd）× 5 折 × **seed0** = **30 run**。
2. **locked 主结果**：预冻结公共配置补 **seed1、seed2** × 5 折 = **10 run** ⇒ 普通 cell **40 run**。
3. **fold-local tuned**：**仅用 seed0**（单 seed，明确只作描述性结果，不做 tuned 的 seed 稳定性）。
4. **共享核心 cell**（S-indep / S-sharedA）需 5 seed：locked 配置再补 seed3、seed4 × 5 折 ⇒ **50 run**。

> 结论：普通 cell 40 run、共享核心 cell 50 run；**locked 才有 3–5 seed 的主结果稳定性，tuned 恒单 seed
> 描述性**。若某处要求 tuned 也多 seed，须显式另加 ~10 run/cell，不得再笼统写「40」。
>
> **MIMIC tuned ERM 对称性冻结（外部评审第六次 #3，防 tuned-HN vs locked-ERM 非对称描述性比较）**：MIMIC HN cell 的
> 40 run 含 seed0 六配置搜索、会顺带产出 fold-local tuned 描述性结果，但 **MIMIC ERM 只补 locked（+25，§9），不做
> tuned ERM/SWAD**。故规定：**MIMIC 的 tuned 描述性结果只作各 cell 自身旁证、不与 ERM/SWAD 作任何比较**（HN-vs-基线
> 一律用 locked）。如日后确需 MIMIC tuned HN-vs-ERM 比较，须显式把 ERM 预算改成 +25/+50（与 HAM 一致）再启。

**6 配置搜索网格冻结表（可复现性；fold-local tuned 依赖此网格，全 cell 搜索空间完全一致）**——
`config_index = lr_outer × wd_inner`，引用冻结 JSON `configs/search_grid.json`（记其哈希，防脚本默认值漂移）：

| idx | lr | wd |  | idx | lr | wd |
|---|---|---|---|---|---|---|
| 0 | 3e-5 | 1e-4 | | 3 | 1e-4 | 1e-3 |
| 1 | 3e-5 | 1e-3 | | 4 | 3e-4 | 1e-4 |
| **2** | **1e-4** | **1e-4** | | 5 | 3e-4 | 1e-3 |

**locked = idx2（lr1e-04/wd1e-04）**，与协议 C1.5 一致。

**locked 公共配置冻结表（#1；locked 是 6 配置网格的 idx2，故 seed0 复用搜索、每 cell 不另加 5 折训练）**
——「所有 cell 一致」= **同一数据集/regime 内一致**，不强制 HAM 与 MIMIC 用相同 batch/训练长度：

| Dataset/regime | locked lr | wd | batch | epochs | scheduler | adapter/backbone LR | patience |
|---|---|---|---|---|---|---|---|
| HAM10000 | 1e-4 | 1e-4 | 128 | ≤30 | 无（AdamW 定 LR） | 同一 LR（单优化器） | 5（best_overall 早停） |
| MIMIC / M→C | 1e-4 | 1e-4 | 128（gpus48） | ≤30 | 同上 | 同一 LR | 5 |
| Fitzpatrick | 1e-4 | 1e-4 | 128 | ≤30 | 同上 | 同一 LR | 5 |

- **locked = 网格中心 `lr1e-04/wd1e-04`**（在 6 配置内 = 搜索 idx2，故 seed0 直接复用搜索运行，不另占预算）。
- 公共固定项（全数据集）：ImageNet 预训练 backbone、WeightedRandomSampler 默认 OFF、增强按各数据集
  CLAUDE.md、grad_clip=1.0、`BCEWithLogitsLoss`、weight decay 不排除任何参数组（backbone 与生成器同规则）。
- 数据集间**允许不同**的仅 batch/epochs（受显存/规模约束），其余在同一 regime 内一致。

**config 选择协议（#9，须与 §11-E0.5 一并定稿；术语已收严）**：
- **locked = 结构主证据**：全 cell 用**预先冻结**的公共 lr/wd 比较，不涉及任何数据依赖的 config 选择。
- **fold-local tuned**（**不叫 nested CV**）：每个 outer fold 只用**该折自己的 validation** 选 config。
  这是 **fold-local / outer-fold-local tuning**，**不是**严格 nested CV（后者须在 outer-train 内部再切
  inner folds；只有真正建立 inner CV 时才用该术语）。
- 若仍用**全局折均**选一个 config，则**明确标注为「内部 CV 调参后的描述性最佳可达」**——因轮转 CV 中
  某样本作 fold-k 的 test 时可能在其他 fold 充当 val 并参与全局 config 选择，全局折均 tuned pooled-OOF
  **不是**严格无泄漏的外部泛化估计。**主结论一律以 locked 为准。**
- **OOD 目标域零泄漏（#9）**：MIMIC 模型/config 选择只用 MIMIC validation，CheXpert 仅作最终评估。

**精确 cell manifest = E0 交付物（唯一真值源，替代「约 17 cell」估数 —— #manifest）**：E0 生成
`configs/cell_manifest.{json,csv}`，逐 cell 列 `cell_id / location / sharing / rank(或 factor_width) /
add·mul / full·lowrank / init / E_L2 / 复用哪个已有 cell / seed 集合 / 确认性·探索性`。**已识别的复用/去重**
（不重复训练）：`L-all-conv ≡ S-sharedA ≡ rank4`、`HH-residual-factorized ≡ L-head`、`HF-lowrank-mul ≡ L-ds`、
`HF-lowrank-add-ctrl ≡ HF-lowrank-add`、`HF-init-zero-noE 第四格 ≡ HF-full-add-ctrl`。**最终精确预算由 manifest
的唯一 cell 数派生**（下方为量级估计，以 manifest 为准）：

**HAM 主网格（外部评审第四次 #2 重算，旧「18–20 cell / 760 run」少算）**：逐项去重 = 范围及端点 **7**（head、
ds、layer4、layer34、all-conv、all+head、param-matched-to-ds）+ 共享新增 **2**（S-indep、S-global-trunk）+ 预算匹配
**3** + HyperHead 桥接新增 **2**（HH-absolute-ctrl、HH-residual-full-ctrl）+ HyperFusion 桥接新增 **~5** + rank 新增
**2**（rank1、rank16）≈ **21 唯一 cell**；其中 **L-all-conv 与 S-indep 各 50 run、余 19 各 40 run** ⇒
2×50 + 19×40 = **~860 run**；**＋ ERM 补训 +25（仅 locked 5×5）/ +50（含 tuned 描述性搜索 30+20）**（canonical base
state 弃用 `seed=42+fold`、共享 base_fc/data_order/aug，**旧 ERM seed0 不可复用**，除非逐元素+RNG 审计证明恰等价）
⇒ **885 / 910**；**＋ 伪/static-A 五臂固定 60 run** ⇒ **945 / 970**；**＋ rank32 触发则再 ~40**。SWAD 由对应 25/50
条 ERM 轨迹派生、不增训练。**最终精确数以 E0 的 cell manifest 为准**（仅当 manifest 证明某预算匹配 cell 与已有
cell 完全重复才可降到 ~820）；全文 HAM 量统一以「**~860 主网格（+ERM 25/50 + 五臂 60）≈ 945–970**」为准，旧
「~720/~760/18–20 cell/+ERM 20」估数作废。

**MIMIC 缩减网格（主终点 B，OOD，外部评审：+L-layer34）**：**6 唯一 cell** 中 **L-head/L-ds/L-layer4/
L-layer34 各 40、L-all-conv(=S-sharedA)/S-indep 各 50** ⇒ 4×40 + 2×50 = **260 run**；**＋ ERM 补训 = 25**
（外部评审第三次 #4：locked 5 fold×5 seed，旧 seed0 因 canonical base state 不可复用；SWAD 派生不另占）⇒
**~285 run**；**＋ OOD 桥接（外部评审 #三.2 + 第三次 #4：三个全必跑，使 manifest E0 前定死、三种原方法各有受控
OOD bridge）L-all+head / HF-full-add-MIP-E / HH-absolute 各 locked-only × 5 折 × seed{0,1,2} = 各 15，共
**45 run**** ⇒ **~330 run**（桥接不进 Holm 族）。**MIMIC 单 run 重（n≈199k）**，走 gpus48、array 分批、留足时限
（曾因 2000 次 bootstrap 超时降 1000 次）。**OOD 推理（full-target 口径，每次评完整 target，外部评审 #1）**：
HN cell 4×5×3 + 2×5×5 = **110 次**；**＋ ERM/SWAD 参照各 5 fold×5 seed = 各 25**；**＋ 3 桥接 ×5 折×seed{0,1,2}
= 45** ⇒ **~205 次 full-target 推理**（比 k→k 拼接的每次 1/5-target 贵 5×，但仍是推理、成本远低于训练；须在任务
生成与完整性检查中精确计入）；C→M 复用历史三方法、不新训。Fitzpatrick 4 cell × 40 ≈ 160；CheXpert ID（可选）。

**分阶段**：先 HAM 范围轴 + 共享主轴（含预算匹配、5-seed 稳定性、诊断）拿 ID 主结论 → **MIMIC 缩减
范围/共享网格 + M→C OOD**（M→C 主分析与机制结论：full-target 复核 + 机制，最高价值）→ HAM 桥接 + rank →
Fitzpatrick/CheXpert 负控制。

---

## 10. 最终结构（速查）

**范围主轴**：ERM · L-head · L-ds · L-layer4 · L-layer34 · **L-all-conv**（+ L-all+head 桥接）
**共享主轴**（固定 L-all-conv）：S-indep · S-sharedA · S-global-trunk（rank=4 主，预算匹配为敏感性）
**HyperHead 桥接**：HyperHead-original（锚点）· HH-absolute-ctrl · HH-residual-full-ctrl · HH-residual-factorized(=L-head)
**HyperFusion 桥接**（固定 L-ds）：锚点 HyperFusion-original · 容量对照 · 形式对照 · 初始化 3-cell（全 full-additive：zero-E / MIP-E / MIP-noE；初始化=前两、E_L2=后两）
**容量分析**（L-all-conv+sharedA）：rank 1·4·16（rank32 触发=HAM val rank16−rank4≥+.005）　**（不含 all-stage full-direct）**
**诊断**：ρ_l(群体等权主+频率加权附) · D_l · stage-knockout(raw + BN 重估；OOD 限 source-only) · **五臂**(真实/置换分属性多次/常量A 评估 + 伪A 训练 + 常量A 训练=static-adapter/null 条件；伪·static-A 训练默认仅HAM、取两个真不同结构) · fold×seed 逐 cell 方差(ERM 须同 fold×seed)
**两个主 regime**（worst 候选集 regime 固定，全模型/seed 共用）：A = HAM-ID（𝒢=age 4 组）；**B = M→C OOD（𝒢=sex∪race∪age；legacy k→k 下唯一显示稳健正向、待 full-target 复核；OOD 属性契约冻结）**
**OOD 估计量**：**full-target 配对效应 Δ̄=mean_{f,s}Δ_{f,s} 为确认性主口径**（每 source fold×seed 评完整 target，患者级 bootstrap 重采样完整 target；k→k 拼接=legacy sensitivity，5-fold logit ensemble=次要）
**OOD 两级成功**：①统计胜利=**p_joint_holm(vs ERM)<.05 且 p_joint_holm(vs SWAD)<.05**（p_joint 已含各自 Overall 非劣）；②达阈值=点估计≥.01（历史 k→k 胜利属①未达②，须在 full-target 口径复算）
**seed/CI**：搜索 seed0；locked 补 seed1-2(普通40run)/seed3-4(共享核心50run)；tuned 恒 seed0。主 CI=cluster bootstrap **只重采样患者/病灶、不重采样 seed**，B=5000(MIMIC 超时用预注册顺序停止,max5000)；p=零假设中心化+(1+·)/(B+1)；**seed 集按分析冻结(非按可用数取交集)**：OOD 范围/历史复现族(m=8)/最小范围判定**全 cell 统一 {0,1,2}**(含 L-all-conv，seed3-4 仅供共享主对比{0-4}与稳定性，不进范围场比较)
**统计**：确认性成功=**p_joint_holm<.05 + 点估计≥.01**（p_joint=max(p_fair,p_noninf)，非劣参照=被比较模型；CI 只描述方向）；共享主对比=S-sharedA vs S-indep；n_pos/n_neg≥20 候选门槛(E-audit 定,禁按模型效应)；Holm 族逐项冻结(去重:S-sharedA vs 基线只在历史复现族)；locked(主)+fold-local tuned(描述性,非 nested CV)

---

## 11. 交付与执行顺序

> **E-audit 阶段执行顺序冻结（外部评审第五/六次：M→C 口径=full-target；且门槛/𝒢/`g*` 顺序不得循环——门槛只依
> label/cluster、与模型预测无关，故先定死再选 `g*`）**：**① Label-only audit**（仅用 cluster + 真实标签 + 属性算
> 群体计数与 bootstrap 有效率，**不用任何模型预测**）→ **② 按预注册规则冻结 n_pos/n_neg 门槛与最终 𝒢** →
> ③ E-audit-OOD full-target 推理（下条）→ **④ 按最终 𝒢 冻结 `g*`（§7.4）** → ⑤ 用 **HAM legacy-OOF + M→C
> full-target** 做效应 CI/功效/动态-min 校准。**门槛在②冻结后不再回改，故 `g*` 不会因门槛变化而重选（去循环）。**
> **门槛调整规则冻结（外部评审第六次 #2，禁参考模型效应）**：默认 **20/20**；若某确认性候选群有效率 <95%，**仅**从
> 预注册序列 **20→30→40** 取第一个使**全部**确认性候选群有效率 ≥95% 的门槛；不看模型效应/p/CI。
> **全失败最终出口冻结（外部评审第七次 #4，闭合所有分支；患者/病灶级聚集可能提高门槛也救不回）**：若 20/30/40 **均
> 不能**使最终候选集全部达 95%，则该 regime 的**动态-worst 终点自动判「不可确认」**、切**预冻结固定 `g*`**（§7.4）；
> **若 `g*` 自身有效率也 <95%，该 regime 不作确认性 worst-group AUC 推断、只报描述性结果**。
- [ ] **E-audit-label（步骤①②，零推理）**：用 cluster+真实标签+属性算各候选群 n_pos/n_neg 与 cluster-bootstrap
  有效重采样率；按上方规则冻结门槛与 𝒢。**这是唯一可回改门槛处，且只依 label 计数、与模型无关。**
- [ ] **E-audit-effect（步骤⑤，低成本、零重训）**：门槛与 `g*` 冻结后，用 **HAM legacy pooled-OOF + M→C
  full-target** 预测，cluster bootstrap 估两主对比 CI 典型宽度、检查 ±0.01 等效功效、估 bootstrap 运行时间（尤其
  MIMIC full-target）。**提前发现「跑完但 CI 宽到无法判定」风险。**
- [ ] **E-audit-OOD：full-target 复核现有 checkpoint（外部评审 #三.1，零重训、千次训练前必做）**：用**现有**
  ERM/SWAD/HyperHead/HyperFusion/HyperAdapt 的 cv5 checkpoint（seed 42–46）**重跑 full-target OOD 推理**
  （`run_ood_cxr_cv.py --estimand full_target`）。**pilot 的作用域严格冻结（外部评审第三次 #2：pilot 不得决定
  是否保留 M→C 主终点，否则构成对已观察目标域结果的数据依赖选择）**：
  - **M→C 无论 pilot 正负都保留为确认性主 regime**（本决定预先冻结、不看 pilot 结果改）。
  - pilot **只**回答：(a) 旧 k→k 正向是否依赖任意 fold 配对；(b) full-target 下效应量与功效/CI 宽度；(c) **论文中
    该结果称「正向机制解剖」还是「旧结果未复现的机制解释」**（仅措辞层级，非删/换/降级主终点）。
  - **诚实定性**：M→C 因旧 k→k 表现良好而入选、效应仅 +0.006~0.008，故它是**对已观察目标域结果、预先规定分析
    规则的内部机制跟进（pre-specified internal mechanistic follow-up）**，**不称「独立外部复现」**。
  - **算力取消条款去后门（外部评审第四次 #小2）**：**默认不取消**；**仅当在查看 pilot 结果之前**已确认算力不足，
    才**按预先规定的资源规则**取消后续 M→C 网格，并**改标为「资源约束下的条件性/探索性机制研究」**（禁止「看到
    pilot 为负」再决定取消——那是结果驱动执行选择）。取消与否一律不改「非独立复现」的话术。
- [ ] **E-audit：worst-group 动态 min 校准自检（外部评审 #五 + 第三/四次；单一 fallback、算法完全冻结）**：动态
  argmin 在两群接近并频繁互换时非光滑、普通 bootstrap 可能失准。**自检算法冻结（外部评审第四次 #小1，全部数值定死、
  防可复现漏洞）**：
  - 用真实标签/群体/基础分数 `z`（= legacy ERM 的 logit）构造两同等期望性能伪模型 `z_1=z+ε_1, z_2=z+ε_2`；
  - **噪声抽样单位 = cluster 相关（外部评审第七次 #2，不能逐图独立加噪破坏患者内相关）**：`ε_{ij}=u_i+v_{ij}`，
    `u_i~Normal(0, ρσ²)` 为患者/病灶级共享扰动、`v_{ij}~Normal(0,(1−ρ)σ²)` 为图像级扰动，**方差比 ρ 冻结=0.5**；
    `σ` 网格 = {0.05, 0.10, 0.20, 0.40}（总方差）。（HAM 病灶级、CXR 患者级为聚集单位。）
  - **worst-case σ 选择 = 确定性 swap-rate 预扫描（唯一算法，外部评审第八次 A1：删「或最接近 50%」与「资源超限才
    单档」双出口——同一输入必返回唯一 σ、且只校准一档）**：对四档 σ 做**相同次数、不含 bootstrap** 的廉价预扫描，
    算两伪模型 **worst-group 身份不同的比例（swap rate）**；**选 swap rate 最高的 σ；并列时按预注册网格顺序
    {0.05,0.10,0.20,0.40} 取第一个**。**全程不看任何 p 值**（无结果导向）。**无论运行时间基准结果如何，确认性校准
    只对这一档执行；运行时间基准仅用于资源安排、不改统计判决算法**。**恒能选出一档并跑校准，无「无结果」分支**；
    **另报 z 的两弱群确定性 AUC 差**（<0.002 = 真近并列；≥0.002 = argmin 本就平滑），仅作场景描述、不参与选择或判定。
  - **主判据 = `p_fair`（外部评审第七次 #2，非 `p_joint`）**：fallback 针对的是**公平终点动态 `min_g AUC_g` 的
    非光滑性**；`p_joint=max(p_fair,p_noninf)` 混入 Overall 非劣、在 HAM 功效不足时 `p_noninf` 会让联合更保守、
    **掩盖 `p_fair` 的假阳性膨胀**。故**合格主判据 = `Pr(p_fair<0.05)` 的 95% 二项 CI 上界 ≤ 0.06**；`Pr(p_joint<0.05)`
    的拒绝率**另报作整体 sanity**，不作主判。
  - **校准参数冻结（对选出的单一 σ）**：外层模拟 = **5000**；**内层 bootstrap = 固定 B=999**；**所有模拟复用同一组
    预生成 cluster 索引**；**恰好前 200 次**额外用 **B=5000** 做敏感性复核；**模拟 seed = 20260714**。**先做运行时间
    基准仅供资源安排**（资源上限具体值/基准规模/单位见文首「待冻结项」，E-audit 定，**不改被校准的 σ 数量或判决算法**）。
  - **B=999 与 B=5000 一致性判据精确化（外部评审第七次 #小1，两者外层 n 不同、不能直接比二项 CI）**：**在同一「前
    200 次」模拟上逐次比较 `p_fair<0.05` 的拒绝决定**，**不一致比例 > 1% ⇒ 校准不合格**（保守启用 fallback）。
  - **HAM 与 M→C 分别执行、分别判定**（不联动）。
  - **唯一 fallback（不保留两出口）**：某 regime 不合格 ⇒ 该 regime **确认性主终点切「固定 ERM-worst 群」worst-group
    AUC**（§7.4；`g*` 用 legacy ERM 预冻结），动态 min **降为描述性**；两者不再并列作确认性。（相同预测的退化 null
    仅作 sanity——Δ≡0、p≈1，**不用于估 α**。）
- [ ] **E0** `ResNet18CondNet`（location/sharing/桥接开关 + 模块 manifest 嵌套 assert + 分级初始等价自检 +
  null_condition 可学习自检 + 接入 harness + checkpoint/config 选择定稿）
  **＋ 生成去重后的精确 cell manifest `configs/cell_manifest.{json,csv}`**（唯一真值源，配置相同的 cell 复用
  同一训练结果、不重复训练；据此派生最终精确预算，替代量级估数）
- [ ] **E1** 诊断挂钩（ρ_l / D_l / stage-knockout(含 BN 重校准,OOD source-only) / 五臂(含常量-A 训练) /
  伪属性生成器 / bootstrap p 值公式 + 候选群体集合 + 估计量固化）
- [ ] **E2** HAM 范围轴（§3，含 L-all+head）—— `ssh biomedia-slurm` 后 `sbatch`，`--exclude` semois
- [ ] **E3** HAM 共享轴（§4，含预算匹配 + 5-seed 稳定性 + 两桥接）
- [ ] **E4** **M→C 受控实验**（MIMIC ID 缩减范围/共享网格 + M→C OOD 主分析与机制结论：full-target 复核 + 机制）**＋ C→M 历史方法级背景整理**
  （不训 CheXpert 受控网格，§8.2）—— 走 gpus48、留足时限
- [ ] **E5** HAM 容量 / 形式轴 + 桥接（§5 + §4.4）
- [ ] **E6** Fitzpatrick 负控制；**满足预注册触发条件（HAM 或 M→C 出现确认性范围效应）时补 CheXpert ID
  端点验证**（§8.3，条件触发的次要分析）
- [ ] **E7** 汇总报告 `docs/conditioning_ablation_results.md`

---

## 12. 冻结状态与实现验收项

**确认性骨架已冻结**（九轮内部审阅 + 八轮外部评审并入；仍待冻结项见文首状态说明）。下列为 **E0/E1 实现时
的验收项（confirmatory skeleton frozen · implementation to verify）**，随代码落地打勾。仅 **n_pos/n_neg 门槛**
及文首所列 manifest/预算匹配/global-trunk 结构保留 E0 前最后一次定稿，此外确认性部分全部锁定。

> **本节只保留「第十轮（外部评审）」落地状态**；**第二~九轮内部审阅的逐条日志已迁至
> [`docs/conditioning_ablation_changelog.md`](conditioning_ablation_changelog.md)**（外部评审 #四.2）。凡历史
> 日志与本方案冲突处一律以本方案为准（changelog 顶部已列被覆盖的旧数字对照）。

**第十轮审阅（外部评审，八次；已核对代码属实、生成器参数量与 DoF 逐一复算吻合）落地状态**：

*第一次外部评审（3 硬阻塞 + 机制诊断 + 文档冲突）*：
- [ ] **OOD 主估计量 k→k → full-target**（#外1，§7.1/§7.4/§7.5/§8.2/§9/§10）：每 source fold×seed 评完整
  target、配对效应 `Δ_{f,s}` 对 (f,s) 等权平均；患者级 bootstrap 重采样完整 target；k→k=legacy sensitivity；
  次要 5-fold logit ensemble；`run_ood_cxr_cv.py` 增 `--estimand full_target|kk` + assertion；历史 M→C 胜利须
  在新口径复算（§0.3 数字来自 k→k）。
- [ ] **canonical ERM base state + fold/seed 解耦**（#外2，§1.1/§1.3-E0.6/§6.3/§7.5）：每 `(dataset,fold,seed)`
  冻结 `base_backbone/base_fc/data_order/aug`；所有 cell（含比较 ERM）加载同一 `base_fc_state` 再初始化 adapter；
  弃用 `seed=42+fold`、`split_id=fold` 与 `train_seed∈{0..4}` 解耦、各随机源用派生 RNG；单元测试断言 residual
  cell 的 base fc 与比较 ERM 逐元素相等（否则 Δθ=0 不等价 ERM）。
- [ ] **容量五量表 + 初始化数值冻结**（#外3，§0.2/§1.1/§4.3）：五列容量 + 冻结实测（HH 34,201 / HF 8,550,816 /
  HA 2,764,388）；near-zero σ=1e-3、bias=0、混合容差；HyperHead 定性为 absolute full-head（无共享基础头）。
- [ ] **MIMIC 补 L-layer34**（#外4，§3/§7.6/§8.2/§9）：缩减网格 5→6 cell，补齐严格嵌套阶梯。
- [ ] **机制诊断补严 + 文档冲突清理**（#外5/#外6，§6.1/§6.2/全文）：置换单位病灶/患者级、`n_perm=20`；轮数、
  预算 720→760、`p_Holm`→`p_joint_holm`、第四格措辞等。

*第二次外部评审（4 硬问题 + 补强 + 文档管理；本轮）*：
- [ ] **Holm 族补 L-layer34**（#外7，§7.3）：OOD 历史复现族 m=6→**8**（{L-ds,L-layer4,L-layer34,L-all-conv}×
  {ERM,SWAD}）——否则 §7.6 用 L-layer34 判最小范围却无其预注册 Holm 结果；②级「点估计≥+0.01」明确 **vs ERM 与
  SWAD 两者都须达**。
- [ ] **§7.2/§7.3 成功规则分离**（#外8，§7.2）：**结构性确认成功**=p_joint_holm<.05 且点估计≥+0.01；**历史统计
  胜利复现**=仅 p_joint_holm(vs ERM/SWAD)<.05、**不要求 +0.01**（历史 +0.006~0.008 本就过不了）、+0.01 单独分级。
- [ ] **DoF 数学订正**（#外9，§0.2/§4.3）：HyperHead 实现/DoF=**513**（含 bias，非 512）；HyperAdapt shared-A
  用 **stage-coupled 公式** `Σ_s[r·C_out,s+Σ_{l∈s}r·C_in,l−r²]`=conv 19,136+fc 512=**19,648**（`Σ_l r(C_in+C_out−r)`
  是 independent-A 上界 34,512、会重复计 A 与 −r²）；realized=1,393,152。
- [ ] **near-zero 与 MIP 分写 + 混合容差**（#外10，§1.1）：near-zero=`Normal(0,1e-3)`；MIP=`Kaiming×0.01`（两者
  是初始化消融的对照臂，除 hidden/bias/E_L2 外全同）；`residual_near_zero` 改 `‖Δz‖≤atol+rtol‖z_base‖`（atol=1e-3,
  rtol=1e-2）+ 权重侧 `‖ΔW‖_F/‖W‖_F<1e-2` 双查，防 base logit≈0 时相对误差爆炸。
- [ ] **D_l 归一化 + 更名**（#外11，§6.1）：主指标 `D^rel_l=‖·‖_F/‖W_l‖_F`（原始 norm 随通道数增长）；「反事实」
  →「受控属性输入切换 controlled attribute-input contrast」（换属性保图无真实因果解释）。
- [ ] **BN 重估显式流程**（#外12，§6.1）：reset_running_stats + 清零 num_batches_tracked + BN train mode + 其余
  freeze + 单遍；措辞「单遍累计 BN 统计」非「精确总体方差」。
- [ ] **E-audit 前 full-target 复核 + 软化定位**（#外13，§0.3/§8.2/§11）：用现有 checkpoint 零重训跑 full-target
  OOD，未复核前措辞改「legacy k→k 下唯一显示稳健正向、待 full-target 复核的候选正向 regime」。
- [ ] **worst-group 动态 min 校准自检**（#外14，§11 E-audit）：相同预测 null + 近并列群 null 两项；失准则改
  同时置信区间/固定弱组。
- [ ] **RQ3 OOD locked-only 桥接**（#外15，§8.2/§9）：加 L-all+head + HF-full-add-MIP-E + HH-absolute
  （**第三次评审已改为三个全必跑、固定 45 run**，见下 #内4），locked+seed{0,1,2}、不进 Holm 族、仅描述性机制
  证据（否则 L-all-conv 未复现 HyperAdapt 增益无法归因 fc）。
- [ ] **状态改「确认性骨架已冻结」**（#外16，文首/§12）：manifest/预算匹配/global-trunk 结构/n_pos_neg 门槛仍待
  E0 前定稿；建议 E0 前先人工出一版 cell manifest。
- [ ] **历史日志迁 changelog**（#外17，§12）：第二~九轮移至 `conditioning_ablation_changelog.md`。

*第三次外部评审（4 冻结前必改 + 6 内部一致性；总体 9/10）*：
- [ ] **seed estimand 统一（外部评审第三次 #1，最重要）**（§7.5/§10）：OOD 历史复现族(m=8)与最小范围判定**全 cell
  统一 seed {0,1,2}**（含兼作 S-sharedA 的 L-all-conv，`Δ̄_c=(1/15)Σ_{f,s≤2}Δ_{c,f,s}`）；seed3-4 仅供共享主对比
  {0-4}与稳定性，**不进范围场比较**——否则 L-all-conv 独享 5-seed 更稳均值、破坏最小范围 estimand 可比性。分析
  脚本 assert「同族/判定内全 cell seed 集一致」。
- [ ] **pilot 不决定 M→C 保留（#2）**（§8.2/§11）：M→C 无论 pilot 正负都保留为确认性主 regime；pilot 只定
  论文措辞（正向机制解剖 vs 旧结果未复现的机制解释）；诚实定性为「预先规定规则的**内部机制跟进**」非「独立复现」。
- [ ] **动态 min 校准自检可真检 α + 单一 fallback（#3）**（§11）：伪模型 `z+ε_1 / z+ε_2`（调噪使两弱群 AUC 近）、
  ≥5000 模拟记 `Pr(p_joint<.05)`、拒绝率 95% 二项 CI 上界≤0.06 判合格；**唯一 fallback**：不合格则确认性主终点切
  「固定 ERM-worst 群」、动态 min 降描述性（删「或同时置信区间」的双出口）；相同预测退化 null 仅 sanity 不估 α。
- [ ] **ERM 预算：canonical base state 使旧 seed0 不可复用（#4）**（§9）：ERM 补训 **+25（仅 locked 5×5）** 或
  **+50（含 tuned 描述性搜索 30+20）**，非原「+20」；SWAD 派生。
- [ ] **内部一致性 6 项**（#内1-6）：§7.1/§10「唯一稳健取胜」→软化(#内1)；§7.6 +0.01 明确 vs ERM 与 SWAD 均满足
  (#内2)；「有序转折」改 **suffix-success 严格定义**(#内3)；HH-absolute OOD 桥接**改必跑**、MIMIC 桥接固定 45 run
  (#内4)；OOD 推理预算含桥接 = **~205 次**(#内5)；外部评审轮数「一→三」(#内6)。

*第四次外部评审（4 实质 + 6 收口；总体 9.2/10）*：
- [ ] **S-sharedA 容量扣除 fc（外部评审第四次 #1）**（§0.2/§4.3）：§4 主轴 S-sharedA≡L-all-conv **不含 fc**，
  生成器 = 2,764,388 − fc 264,708 = **2,499,680**、发射 **19,200**、调制 **1,392,640**、DoF **19,136**；预算匹配锚点
  两个预算锚点分离：共享轴锚点 2,499,680（sharedA/indep/global-trunk），范围端点 `L-all-conv-param-matched-to-ds`
  锚点 = N_gen(L-ds)±2%（§4.3 #外五1，非 2,499,680）；§0.2「HyperAdapt」行注明含 fc 的 L-all+head、S-indep 发射 34,304/DoF 34,000。
- [ ] **HAM 预算重算（#2）**（§9）：21 唯一 cell、2×50+19×40 = **~860 主网格**；+ERM 25/50 ⇒ 885/910；+五臂 60
  ⇒ **945/970**；+rank32 触发 ~40；旧「760/18–20 cell」作废，精确以 manifest 为准。
- [ ] **固定 ERM-worst 群精确定义（#3）**（§7.4）：`g*` 用 **legacy ERM 预测**在新 cell 训练前冻结、平局取 𝒢
  预注册序首、**按 regime 分别校准/切换**，不随新 ERM/seed/bootstrap 重选。（⚠️ **本条原写的 ID `argmin mean_s`、
  OOD `argmin mean_{f,s}` 公式已由第五次评审 #2 覆盖**——legacy ERM 无 crossed F×S，正确为 HAM pooled-OOF argmin、
  M→C `argmin_g (1/5)Σ_f AUC^{ERM,legacy,f}`，见 §7.4 现行文本。）
- [ ] **RQ3 机制对照 = 描述性（#4，用户裁定采「更简洁」）**（§4.4/§4.4.2）：所有桥接（HH/HF 容量·形式·初始化、
  L-all+head vs L-all-conv）报双侧效应+CI、**不进 Holm 族、不承担确认性**；RQ3 止于「与机制解释一致」，不称「确认」；
  确认性只有 RQ1/RQ2。
- [ ] **6 收口**：动态-min 校准算法全冻结（σ 网格{.05,.1,.2,.4}、近似阈 <0.002、恰 5000 次、模拟 seed
  20260714、分 regime）(#小1)；pilot 取消条款去后门（仅查看前按资源规则）(#小2)；§5 工作项 E4.x→**E5.x**(#小3)；
  「唯一正向 regime/正向主结论」→「候选正向/复核+机制」(#小4)；E_L2 公式 `√d·γ/max(‖γ‖₂,1e-6)` 写入规格(#小5)；
  稳定性主方法=fold 内跨 seed 样本方差直接比较、混合模型作敏感性(#小6)。

*第五次外部评审（3 实质 + 4 收口；总体 9.4/10）*：
- [ ] **budget-reduced 锚点分离（外部评审第五次 #1）**（§3.2/§4.3/§9）：`L-all-conv-budget-reduced` 更名
  **`L-all-conv-param-matched-to-ds`**，目标 = **N_gen(L-ds)±2%**（非 2,499,680）；共享轴锚点仍 2,499,680；两锚点勿混。
- [ ] **g\* 公式匹配 legacy 结构（#2）**（§7.4）：legacy ERM 是每 fold 单 run、`seed=42+fold`、无 crossed F×S；
  改 HAM=pooled-OOF argmin（不对 seed 平均）、M→C=E-audit-OOD 后 `argmin_g (1/5)Σ_f AUC^{ERM,legacy,f}`（仅 source
  fold 平均）；操作顺序：定候选+门槛→full-target 推理→冻结 g\*→训练。
- [ ] **动态-min 校准可执行 + 全跳过闭合（#3）**（§11）：内层 bootstrap B=999（非 5000）+ 复用预生成索引 + 小撮
  B=5000 复核。（⚠️ **本条「四档全跳过→取差值最小档」的 σ 选择已被第七次评审 #3 覆盖为 swap-rate 预扫描单一机制**，
  见 §11 现行文本；「无空分支」结论仍成立。）
- [ ] **4 收口**：E-audit 顺序=群体门槛→full-target 推理→冻结 g\*→用 HAM-OOF+M→C full-target 做 CI/功效/校准(#小1)；
  BN 重估用确定性 eval transform、无增强、shuffle=False、固定 batch、drop_last=False(#小2)；外部评审轮数「三→四」(#小3)；
  §0.3「解剖真实正向信号」→「legacy k→k 下观察到的候选正向信号、full-target 复现才定位/否则转机制解释」(#小4)。

*第六次外部评审（2 硬 + 2 实质 + 3 文档；总体 9.5/10）*：
- [ ] **生成器参数按数据集分别计算（外部评审第六次 #1，硬）**（§4.3/§8.1）：编码器不同（Age 18,752 / Skin 18,784 /
  Patient 22,880），§4 共享轴首战在 HAM，故 **HAM 锚点为准：N_sharing^HAM = 2,495,552、N_range^HAM = N_gen^HAM(L-ds)
  = 415,040**；2,499,680/2,764,388 是 CXR 值；全文容量数字含编码器口径统一。**param-matched-to-ds 必须保持 shared-A
  拓扑**（只缩 hidden/projection width、不得改 global-trunk，否则同改容量+共享）。
- [ ] **E-audit 门槛/𝒢/g\* 去循环（#2，硬）**（§11）：拆 **label-only audit**（仅 cluster+标签+属性算计数/有效率、
  无模型预测）→ 按预注册规则冻结门槛（20/20，不足则 20→30→40 序列取首个使全确认性候选群有效率≥95%）与 𝒢 →
  full-target 推理 → 冻结 g\* → 效应 CI/功效/校准；门槛冻结后不回改，g\* 不重选。
- [ ] **BN 必须 momentum=None（#3）**（§6.1）：仅 reset + 清 num_batches_tracked 仍是 EMA；补 `bn.momentum=None`
  才 cumulative；给出 save/restore 完整代码；batch=128 写入 manifest。
- [ ] **动态-min 校准补冻结（#4）**（§11）：「两弱群接近」判据基于**基础分数 z 的确定性群 AUC**（σ 不改之）；
  **恰好前 200 次** B=5000；**先做运行时间基准**。（⚠️ **本条原写「σ=0.40 单一 worst-case」「方向不一致按不合格」已被
  第七次评审 #2/#3/#小1 覆盖**：worst-case σ 改 **swap-rate 预扫描**选、主判据改 **`p_fair`**、一致性改「同一前 200 次逐
  sim 比 `p_fair<.05`、不一致 >1% 判不合格」，见 §11 现行文本。）
- [ ] **3 文档**：外部评审轮数「四→五」(#文1)；§12 第四次 g\* 旧公式标注「已被第五次覆盖」(#文2)；MIMIC tuned 描述性
  结果不与 ERM/SWAD 比较（去 tuned-HN vs locked-ERM 非对称）(#文3)；另加 OOD 主 CI 推断边界声明（§7.5，条件于固定
  source cohort，反映 target 抽样不确定性）。

*第七次外部评审（4 实质 + 4 小；总体 9.6/10）*：
- [ ] **§3.2/§4.3 拓扑冲突消除 + 瓶颈冻结（外部评审第七次 #1）**（§3.2/§4.3）：§3.2 删「共享 trunk」措辞、与 §4.3
  一致（保持 stage-shared A + layer-B，严禁 global-trunk）；`param-matched-to-ds` 各生成器换瓶颈 MLP
  `Linear(128,h)→ReLU→Linear(h,d_l)`、**h 冻结=17**（→414,791，−0.06% ∈ ±2%；h=16/18 均超）、A 末层零初始化 + B
  Kaiming + bias + ReLU、无 global-trunk 出口；仍称参数量敏感性对照非纯容量控制。
- [ ] **动态-min 校准主判据改 `p_fair` + cluster 噪声（#2）**（§11/§7.4）：主判据 = `Pr(p_fair<0.05)` 95% 二项 CI
  上界 ≤0.06（`p_joint` 另报作 sanity，因 `p_noninf` 在 HAM 功效不足时会掩盖 `p_fair` 假阳性膨胀）；噪声 cluster
  相关 `ε_ij=u_i+v_ij`、方差比 ρ=0.5。
- [ ] **worst-case σ = swap-rate 预扫描（#3）**（§11）：不含 bootstrap 廉价预扫、只算 worst-group 身份交换率、选
  交换率**最高**的 σ（全程不看 p 值），取代默认 σ=0.40。（⚠️ **本条原写「最高/最接近 50%」双规则已被第八次评审 A1
  收敛为唯一规则**：只取最高、并列按网格序 {0.05,0.10,0.20,0.40} 取首、且只校准这一档，见 §11 现行文本。）
- [ ] **全分支闭合（#4）**（§11/§7.4）：门槛 20/30/40 全失败 ⇒ 动态-worst 判「不可确认」→ 切固定 `g*`；`g*` 自身
  有效率 <95% ⇒ 该 regime 不作确认性 worst-group 推断、只描述性。
- [ ] **4 小**：B=999/B=5000 一致性改「同一前 200 次逐 sim 比 `p_fair<.05` 决定、不一致 >1% 判不合格」(#小1)；外部
  评审轮数「五→七」(#小2)；Holm FWER 声明「按 RQ×regime 族内控制、非全局」(#小3)；`g*` 解释边界「不称无选择偏差
  独立确认」(#小4)。

*第八次外部评审（1 A 类阻塞 + 0 B新增 + 0 C；除 A1 外全通过，判「未通过：1 A 类」）*：
- [ ] **A1：动态-min 校准 σ 选择收敛为唯一确定性算法（§11）**：删「或最接近 50%」与「资源超限才单档」双出口——
  **只取 swap-rate 最高的 σ、并列按网格序 {0.05,0.10,0.20,0.40} 取首、无论运行时间只校准这一档**（运行时间基准仅供
  资源安排、不改判决算法）。验收：同一输入返回唯一 σ、明确并列、资源基准不改被校准 σ 数、每 regime 单一合格/不合格
  判决、不允许人工覆盖。
- 评审确认 B1–B8（cell manifest / 共享轴预算匹配 / 五容量量 / 群体门槛 / full-target+g\* / canonical base state /
  bootstrap+fallback 自动化 / BN 重估）**均已在方案中有明确阶段与验收路径**，属 E0/E1/E-audit 实现验收项、非设计缺陷；
  C 类新增 = 0。

**第二~九轮内部审阅的逐条落地日志已迁出**，见 [`docs/conditioning_ablation_changelog.md`](conditioning_ablation_changelog.md)（外部评审 #四.2：历史日志从主规格移出，主文件只保留当前有效规格 + 第十轮，杜绝旧数字被脚本/工具误检索）。凡与本方案冲突处一律以本方案为准。

> **工作规范**（对齐根目录 `CLAUDE.md`）：E0/E1（代码）完成后报实现逻辑并核对无冲突；E2–E5（训练）
> 先报本次训练内容，**必须先 `ssh biomedia-slurm` 再 `sbatch`**，提交后查状态、节点故障换节点；
> E7 单独在 `docs/` 建 md；每完成一项即在本文件打勾。
