# 条件化范围 消融实验方案

> 🛑 **信号闸门口径更新（2026-07-19）**：闸门已由单-split 条件互信息改为 **OOF conditional V-information**
> （Xu 2020 / Hewitt 2021；权威 = `docs/conditional_v_information_gate.md`）。关键变化：
> **HAM-age 由「唯一正信号」退回「未检出」**（+0.0028 bits，CI 触 0）；Fitzpatrick ≈0；
> **CheXpert / MIMIC 全轴 CI<0 = 决定性 0**；唯一 detected 为 **PAPILA-age**（+0.041 bits，但 n=420）。
> control task 五库全部通过 ⇒ 零读数非假阴。本文「HAM-age `I(Y;age|X)>0`」一族表述**已过时**（下方已内联订正）；
> 各处**结论方向不变**，但「有信号」前提须按新闸门重读。


> **RQ：敏感属性条件化从分类头扩展到深层、再到完整 backbone 时，是否改善整体性能与 worst-group fairness？**

本方案只研究**条件化范围**一个轴。参数共享、rank/容量、参数化/调制/初始化归因均不作为变量（固定锁死）。
与主比较协议 `docs/comparison_protocol.md`（六方法「HN 是否有收益」）平行互补。相较 HyperAdapt 原论文的
depth ablation，本方案多出**公平终点、严格 OOF 推断、跨数据集 OOD 验证**三项贡献。

被剔除的共享/容量/归因内容与全部实现级细节归档于 `conditioning_ablation_plan_full_v4_reference.md`。
状态标记：`[ ]` 未开始 · `[~]` 进行中 · `[x]` 完成。

---

## 0. 研究问题与信号地图

### 0.1 研究问题
把敏感属性条件化从仅影响分类头，逐步扩展进 backbone，是否带来额外效用与公平收益？增益来自深层条件化就够，
还是须铺满全 backbone？

三个现成模型（HyperHead/HyperFusion/HyperAdapt）在范围、参数化、调制、容量、初始化多轴同时不同、不可比。
本方案**固定其余全部轴、只改条件化范围**，得到严格嵌套、逐档可比的范围阶梯。

### 0.2 信号地图（权威：`docs/oof_regime_results.md`，**averaging 口径，2026-07-16 重构版**）

> 🛑 **本节已于 2026-07-16 同步至 averaging 口径。** 此前所有引用 pooled-OOF 的数字与结论已作废——
> pooled AUC 有学界已知的悲观负偏（Parker 2007 / Airola 2010，详见 `oof_regime_results.md` §0），
> 且**污染程度因方法而异** ⇒ 会把校准漂移测成范围效应。

| 数据集 | `I(Y;A\|X)` | 现状（averaging，Δ=marginal worst vs ERM） | 定位 |
|---|---|---|---|
| **HAM-age** | ~~age >0~~ → **未检出（≈0）** | 三 HN 全 **n.s.**（+0.018~+0.033，CI ±0.05~0.08）；只有 SWAD 显著（+0.046） | ID 主战场（**新闸门下无可用信号**，null 更属预期） |
| **MIMIC→CheXpert** | ~~弱 >0~~ → **决定性 0** | **只有 HyperAdapt 显著（+0.0045）**；HyperHead/Fusion n.s. | OOD 主战场（go/no-go = **GO**） |
| Fitzpatrick | ≈0（新闸门确认） | 三 HN 全 n.s. | 条件触发负控制 |

#### 既有命名模型的 worst-group 全景（averaging + 患者/病灶级配对 cluster bootstrap，B=1000）

| regime | HyperHead | HyperFusion | **HyperAdapt** |
|---|---|---|---|
| HAM (I>0) | +0.0182 n.s. | +0.0334 n.s. | +0.0212 n.s. |
| Fitzpatrick (I≈0) | +0.0266 n.s. | +0.0007 n.s. | +0.0170 n.s. |
| PAPILA (I≈0) | −0.0375 n.s. | −0.0323 n.s. | −0.0123 n.s. |
| MIMIC (I 弱>0) | +0.0004 n.s. | +0.0001 n.s. | **+0.0022 显著** |
| CheXpert (I≈0) | **−0.0040 显著劣** | **+0.0029 显著优** | −0.0023 n.s. |
| **M→C OOD** | +0.0020 n.s. | +0.0015 n.s. | **+0.0045 显著** |
| **C→M OOD** | −0.0001 n.s. | −0.0002 n.s. | **+0.0017 显著** |

**换口径后被推翻的三条旧结论**（同一批预测、只改聚合 ⇒ 差异纯属口径伪影）：
1. ~~「深层 HN（HyperFusion/HyperAdapt）在 M→C 稳健胜」~~ → **只有 HyperAdapt 胜**；
   HyperFusion `+0.0058 显著 → +0.0015 n.s.`（约 3/4 是漂移）。
2. ~~「C→M 全 HN 显著更差」~~ → HyperHead/HyperFusion **打平**，HyperAdapt 反而 **+0.0017 显著优**。
3. ~~「强烈方向不对称（M→C 胜 / C→M 全输）」~~ → **不成立**：HyperAdapt **两个方向都显著为正**。

> ⚠️ **对本方案范围轴的直接影响（表述边界）**：
> - **H1（C-Deep − C-Head）没有既有经验证据支持其方向**：averaging 下 head-only 的 HyperHead 与单点
>   backbone 的 HyperFusion **在所有 regime 都统计不可分**（两者各 regime 均 n.s. 或量级相当）。
>   ⇒ **E2 须按真正的双侧探索对待、不得预期正向。**
> - **唯一与「范围越深越好」方向一致的线索**：**HyperAdapt（条件化最遍布）是唯一一致显著为正的 HN**
>   （MIMIC +0.0022、M→C +0.0045、C→M +0.0017，且从未显著劣）。**但幅度全 <0.005（微弱档）**，
>   且三处显著都出现在 **n≥138k** ⇒ **统计显著 ≠ 实际重要**。
> - **且该线索不能归因给范围**：三个命名模型在范围/参数化/调制/容量/初始化上**同时不同（混杂）**。
>   **而受控消融（E1，唯一只改范围）H1/H2 均 n.s.**，其 **ρ_l 更显示 conv adapter 的偏移仅 1.3~2%、
>   fc 达 ~100%**（条件化几乎全由 fc 承担）⇒ **HyperAdapt 的微弱优势很可能来自范围以外的轴**
>   （容量/参数化/初始化）。这恰恰**强化**了受控消融的必要性，也是目前最站得住的表述。
> - **不可单独引「HN 胜 SWAD」作证据**：M→C 中三 HN 均显著胜 SWAD，但**此 regime 下 SWAD 本身显著
>   劣于 ERM（−0.0044）** ⇒「胜 SWAD」部分只是「SWAD 偏弱」。

> **表述边界（避免因果过声）**：受控消融至多说明「在本数据集、ResNet-18 与所测模型族内，扩大到某一条件化
> 范围是否带来**可重复的增量效用**，以及哪一档是所测范围中的**最小观测有效配置**」。H1/H2 是范围间的**优效
> 比较**，**不能**证明某些层「必要」（Deep 不显著优于 Head 也可能只是功效不足）。全文**禁用**「必要成分 /
> 充分条件 / 范围阈值」等措辞。
> **归因澄清（修正本文件此前的一次错误）**：曾把「M→C legacy k→k 与 full-target 的三处变号」归因于
> **估计量选择**（k→k vs full-target）。**实为聚合方式**（pooling vs averaging）——k→k 预测改用 averaging
> 后即与 full-target 吻合（HyperHead +0.0020 vs +0.0022；HyperFusion +0.0015 vs −0.0005；
> HyperAdapt +0.0045 vs +0.0031）。full-target 因**逐 (f,s) 算指标再平均、从不跨折池化**而天然免疫。

---

## 1. 模型：四个严格嵌套的条件化范围

统一模型 `ResNet18CondNet`（改造 `resnet18_hyperadapt.py`），只暴露 `location` 开关，其余锁死。

| Cell | 条件化范围 | 作用 |
|---|---|---|
| ERM | 无条件化 | 基线 |
| **C-Head** | 仅 fc | 最小范围（head-only，与 HyperHead **条件化位置对齐**） |
| **C-Deep** | fc ＋ layer4 全部 conv | 深层范围 |
| **C-Full** | fc ＋ layer1–4 全部 conv | 全范围（与 HyperAdapt **条件化覆盖位置对齐**） |

严格嵌套 **C-Head ⊂ C-Deep ⊂ C-Full**。fc 条件化在三档间恒定，故增量干净：
`C-Deep − C-Head` = 加 layer4 conv、`C-Full − C-Deep` = 加 layer1–3 conv。

> **仅位置对齐、非原模型复现（C3）**：四个 cell 一律用统一的**残差低秩 adapter**（sharedA、rank=4、Δθ≈0）。
> C-Head 用残差低秩 fc adapter，**不同于**原 HyperHead 的直接生成完整分类头；C-Full 覆盖 HyperAdapt 的条件化
> 位置，但不含其原始编码器/初始化细节。故只称「条件化位置对齐」，不称复现原模型。

**锁死的公共设定**（不作变量）：统一条件编码器（16→128）、参数共享固定 sharedA、conv rank=4、conv 乘性低秩 /
fc 加性低秩、初始化令 Δθ≈0、训练 regime 对齐现有 HAM HN（BCE、AdamW、bs=128、≤30 epoch）、主结果用
`best_overall` checkpoint。

> **canonical base state（硬约束）**：每 `(dataset, fold, seed)` 冻结同一份 base backbone / base fc / 数据顺序 /
> 增强 RNG，所有 cell（含比较用 ERM）加载同一 base 再初始化 adapter——否则「Δθ=0 等价 ERM」不对比较 ERM 成立。

> **容量随范围增大是范围干预的组成部分**，不作参数匹配（参数匹配会引入 rank/embedding/generator topology 新
> 变化，反而破坏「只研究范围」）。直接**报告各档 adapter 参数量与 FLOPs**即可。

---

## 2. 研究假设（每个 regime 内两项）

- **H1（深层 vs 仅头）= C-Deep − C-Head**：敏感属性是否需要进入 backbone，而非只影响最终决策边界。
- **H2（全范围 vs 深层）= C-Full − C-Deep**：向中浅层继续扩展是否产生额外效用，还是深层条件化已足够。

H1、H2 是本轴仅有的两个确认性主假设，在每个 regime 内做 Holm 校正（§3.3 family 表）。**不做 suffix-success
阶梯判定。**

> **最小观测有效配置（纯描述性，判据 = 点估计）**：按嵌套序 Head→Deep→Full，**第一个**满足「worst-group
> 点估计相对 ERM `Δ̄ > 0` **且** Overall 点估计非劣 `Δ̄_Overall ≥ −δ_NI(−0.005)`」的档；**CI 一并报告以示透明，
> 但不要求 CI 下界 >0**（要求 CI 即等于把它升格为显著性主张，与描述性定位冲突）。无一档满足 ⇒ 报「所测范围内
> 未观测到有效配置」。**不承担显著性判决、不进 Holm、不作因果/必要性解读**；进入该档的**增量是否显著**由 H1/H2
> 的相邻比较回答，二者不混同。

---

## 3. 数据集与统计方案

评估沿用各数据集现有无泄漏 CV 划分（HAM 病灶级 GroupKFold、CXR 患者级分组）。**统一 3 seed。**

> 🛑 **口径变更（2026-07-16）：~~pooled-OOF~~ → `averaging`**（对本文件原定 pooled-OOF 的**有依据偏离**）。
> **理由**：pooled AUC 把 5 折**不同模型**的分数拼起来排序、构造出**跨折正负对**，要求各折尺度可比；
> 而早停轮数异质使 logit 尺度漂移，**且污染程度因方法系统性不同** ⇒ **会把校准漂移测成范围效应**
> （E1 实测：旧口径给出 H2 假阳性「确认」Holm=0.006，真值 +0.0075/p=0.70）。这是**学界已知缺陷**：
> Parker/Günter/Bedo (2007) *BMC Bioinformatics* 8:326；Airola et al. (2010) *JMLR W&CP* 8:3–13
> （推荐 **averaging** 或 LPOCV）；Forman & Scholz (2010) *SIGKDD Explorations* 12(1):49–57。
> **不损失评估 n**：抬 n 的功劳来自 **CV 本身**（每样本轮到一次 test），非 pooling；AUC 方差由类别样本数
> 支配（`Var≈c/n_pos+c'/n_neg`）而非「对数」，5 折各估再平均(÷5)与池化精度相当，**实测 SE 反而小 8–28%**。
> LPOCV 需按对重训、算力不可行，故取 averaging。详见 `docs/oof_regime_results.md` §0。
>
> **口径细则**：① **worst/best 用「先平均后 min/max」**：`min_g[mean_f AUC(g,f)]`（Jensen 恒有
> `mean_f[min_g …] ≤ min_g[mean_f …]`；「先 min」会让每折在最噪处取极值、放大 min 的向下偏，且
> 「模型在哪组最弱」应是模型属性、不该每折换答案）；**动态 argmin 仍保留**（每 replicate 重算组均值再取 min）。
> ② ⚠️ **averaging 的已知代价**（Airola 明示）：小格某折只剩一类 ⇒ 该折 AUC 无定义，处理 = 跳过该折、
> 用其余折平均，并报计数（HAM 实测 1/40 折-格无定义）。
> 实现：`scripts/e1_ham_analysis.py --estimator {averaging(主) | pooled_rank | pooled_raw}`，三口径并排可查。

### 3.1 两个主实验
- **A（ID）= HAM10000（age）**：四个模型全跑。回答范围能否利用明确的 ID 条件信号。
- **B（OOD）= MIMIC→CheXpert**：**先用现有 checkpoint 复核 full-target 效应**（E-audit），达 go/no-go 才训练
  四个范围模型、做 M→C OOD 评估。回答范围是否影响跨数据集 worst-group 稳健性。
  - **OOD go/no-go 条件（预先冻结，属算力决策非结果判决）**：若现有 HyperFusion 或 HyperAdapt 中**至少一个**
    模型，其 full-target marginal worst 相对 ERM **与** SWAD 均为正，**且至少一个比较的 95% CI 不含 0**，则启动
    范围 OOD 训练；否则 M→C 降为「资源约束下的探索性机制跟进」，不新训范围网格。
  - **OOD 零泄漏**：MIMIC 模型/config 选择只用 MIMIC validation，CheXpert 仅作最终评估。
- **负控制（条件触发，2026-07-16 重设计）**：
  > 🛑 **旧设计**：「仅当 HAM 或 M→C 出现确认性范围效应，就训练 **Fitzpatrick** 四档」。**两处均已改**。
  - **触发条件收紧（加入实践阈值）**：
    | E2 结果 | 动作 |
    |---|---|
    | 无确认性结果，**或** 效应 `\|Δworst\| < 0.005`（实践阈值） | **直接跳过负控制** |
    | 校正后显著 **且** `Δworst ≥ 0.005` | 启动负控制 |
    > 理由：E1/E2 的效应若本就在「微弱档」以下，负控制无从区分「真无效」与「噪声」，纯属浪费算力。
  - **负控制改为 CheXpert ID 范围网格（+ 顺带 C→M）**，**取代 Fitzpatrick**：
    CheXpert 与 E2 的 **影像模态、任务（No Finding）、敏感属性（sex/race/age）、样本规模（10 万级）
    全部相同**，**只改变训练域与条件信号背景**（`I(Y;A|X)≈0`）⇒ 比跨到皮肤镜的 Fitzpatrick
    **更接近真正的负控制**（后者同时改变模态/任务/属性/规模，是多因素混杂的对照）。
    训练完 CheXpert 四档后，**C→M 只增加推理成本**，还能顺带检验 averaging 复核中
    **「HyperAdapt 双向微正（M→C +0.0045、C→M +0.0017）」是否真由范围造成**。
  - **Fitzpatrick 降为可选的跨模态外部验证**，**不再承担主要负控制角色**。
  - PAPILA **不新增范围训练**（n=420，功效不足）。

### 3.2 模型选择协议（全 cell 同等调参，防「Full 因优化更难被误读为范围无效」）
- **所有 cell 使用完全相同的 LR×WD 搜索空间、fold、seed 与训练预算**（搜索网格见参考文件；locked 中心配置
  与主协议 C1.5 一致）。
- **两级选择**：① 每个 config 内以 **validation Overall AUC** 选 epoch（early stop）；② 每个 cell 以**五折
  validation marginal-worst 均值**选 config，平手时以 **validation Overall** 破局。
- **test（averaging 口径）只用于最终评估**，不参与任何选择。**M→C 的 config 选择仅用 MIMIC validation**，CheXpert
  不参与任何选择。

### 3.3 终点与推断
- **指标**：Overall AUC、marginal worst-group AUC（HAM 候选组 = age 4 组；M→C = sex∪race∪age）、gap 分解
  （Δgap = Δbest − Δworst，防「压低强组」误判为公平改善）。
- **推断 = 患者/病灶级配对 cluster bootstrap**（所有 cell 用相同 cluster 与重采样索引；只重采样 cluster、
  不重采样 seed；replicate 内对 3 seed 等权平均）。
- **OOD 估计量 = full-target 配对效应**：每 source `fold×seed` 评完整 target、对 `(f,s)` 等权平均；k→k 拼接降
  为 legacy sensitivity。历史 M→C 胜利须在此口径下重新确认。

> 🛑 **主分析 CI 的性质必须写明（2026-07-16 补）**：上述 bootstrap **只重采样目标患者、固定这 15 个已训模型
> （5 fold × 3 seed）** ⇒ 它给出的是**「给定这 15 个模型时、针对目标患者总体」的条件性 CI**，
> **训练随机性不在 CI 内**。CheXpert 目标 n=138,644 ⇒ **患者抽样误差极小**，显著性可能主要由少数训练运行
> 驱动却**不进 CI**。**不推翻主分析**（它回答的问题是明确的），但**必须并报下列预先规定的稳健性分析**：

**训练随机性稳健性分析（E2 起强制，预先规定）**：
1. **fold-level 配对效应**：每个 source fold 内**先对 3 seed 平均**，得 **5 个 fold-level 效应**；
   报 **符号一致性**（5 折中同号几折）与**取值范围**。
2. **leave-one-fold-out**：逐折剔除后重算效应，报是否变号。
3. **双层 bootstrap（目标患者重采样 + source fold 分块重采样、seed 嵌套）**——把训练随机性纳入 CI。
4. **判读规则（冻结）**：若**患者 bootstrap 显著、但 fold 效应正负混杂或去掉某一折即变号**
   ⇒ 只能表述为「**目标样本层面显著，但训练稳定性不足**」，**不得**称「范围扩大带来改善」。

> 🛑 **E2 起改为双侧（2026-07-16 修订，解决本文件的一处内部矛盾）**：§0.2 已写明「E2 须按真正的双侧
> 探索、不得预期正向」，但本节此前仍把 H1/H2 定义为**单侧「前者更优」**——两者不能并存。
> **裁定：E1 保留原单侧预设、不追溯修改**（其结果已按单侧报出，见 §5 E1 节）；**E2 改为双侧**。
> 理由：averaging 复核后，H1 的正方向**已无既有经验支持**（HyperHead 与 HyperFusion 在所有 regime
> 统计不可分），若仍用单侧检验，则**先验地排除了「扩大范围反而有害」这一同样合理的可能**。

**确认性判决（两层，分开报告）**：
- **① 统计确认**
  - **E1（HAM-ID，已完成，单侧原预设）**：`p_joint_holm < 0.05`，`p_joint = max(p_fair, p_noninf)`；
    `p_fair` 单侧右尾 `H0: Δ_fair ≤ 0`，零假设中心化 `T*_b = Δ*_b − Δ̄_obs`，
    `p_fair = (1 + #{T*_b ≥ Δ̄_obs})/(B+1)`。
  - **E2（M→C-OOD，双侧）**：报 **H1/H2 的双侧 95% CI**，family 内 **Holm 校正**（对双侧 p）。
    **三分判决（预先冻结）**：
    | 情形 | 判决 |
    |---|---|
    | 校正后 CI **全 > 0** 且 Overall 非劣 | **「扩大范围带来改善」** |
    | 校正后 CI **全 < 0** | **「扩大范围有害」** |
    | 其余（CI 含 0） | **「不确定」**——不得表述为「无效应」 |
  - **`p_noninf`（Overall 非劣，恒为单侧）**：界值 **δ_NI = −0.005**，`H0: ΔOverall ≤ −0.005`；
    与范围检验用**同一批 cluster-bootstrap 索引**。**非劣参照 = 被比较的范围模型本身**
    （H1: C-Deep 不劣于 C-Head；H2: C-Full 不劣于 C-Deep）。
    > 非劣是**方向性命题**（「不比它差太多」），故保持单侧；范围效应本身双侧。二者不矛盾。
- **② 效应分级**（描述性，不作存在性门槛）：`|Δ| ≥ 0.005` = 小但可重复；`≥ 0.010` = 较强实际效应。
  **不是范围效应是否存在的判据**；主结论以 ① 为准，②只作幅度分级。

**Holm family（逐项冻结）**：

| Family | 成员 | m | 侧 |
|---|---|---|---|
| HAM-ID 范围确认（E1，**已完成**） | H1: C-Deep−C-Head；H2: C-Full−C-Deep | 2 | 单侧（原预设） |
| **M→C-OOD 范围确认（E2）** | H1: C-Deep−C-Head；H2: C-Full−C-Deep | 2 | **双侧** |
| **M→C-OOD 历史正向复现（E2，已缩减）** | **C-Full vs ERM（主）；C-Full vs SWAD（次）** | **2** | 双侧 |

> 🛑 **历史复现族已由 4 项缩减为 2 项（2026-07-16）**：原设 `{C-Deep, C-Full} × {vs ERM, vs SWAD}`。
> 但 averaging 复核后，**历史上真正留下正向证据的只有 HyperAdapt**（条件化最遍布 ⇒ 对应 **C-Full**）；
> **HyperFusion 由「显著」变为 n.s.** ⇒ **C-Deep 已没有明确的「历史胜利」需要复现**。
> 故：**主复现 = C-Full vs ERM**；**次要基线 = C-Full vs SWAD**；**C-Deep vs ERM/SWAD 只作描述性结果**
> （它是否足够，由 H1/H2 回答，不必再单列确认族）。
> **且「胜 SWAD」不可与「胜 ERM」等价解释**：M→C 中 **SWAD 本身显著劣于 ERM（−0.0044）**
> ⇒ **C-Full vs ERM 是主证据，SWAD 仅作竞争基线**。
> 历史复现族与 H1/H2 族**分列、不混同**：前者判「是否保留历史 OOD 正向」，后者判范围间增量。
> FWER 按 family 内 Holm 控制，不做全局校正。

**worst-group 估计量（单一确认性路径，无数据依赖切换）**：
- **动态 argmin**（每 replicate 重算 `min_g AUC_g`）**始终是确认性主分析**。
- **固定 legacy ERM-worst 群**（用 legacy ERM 预冻结 `g*`）**始终是敏感性分析**。
- **校准自检只用于解释动态 min 的覆盖率与偏差，不切换主终点**。候选组由 regime 固定、全模型共用，设最低样本
  门槛（E-audit 依 label 计数定，禁按模型效应调）。

---

## 4. 机制诊断（仅两项）

全部复用范围 cell 的 checkpoint，零额外训练：

> 🛑 **ρ_l 已由「附加机制图」升格为 manipulation check（2026-07-16）**：E1 证明**「允许某层条件化」
> ≠「模型实际使用该层」**（新增 conv adapter 的偏移仅 1.3~2%，fc 却达 ~100%）。故 ρ_l 是**解释 H1/H2
> 的必要条件**，而非可选补充。**E4 汇总必须分两层报告**：
> | 层次 | 问题 | 指标 |
> |---|---|---|
> | **① 范围分配效应** | 启用 Head/Deep/Full 后**性能**如何变化 | H1/H2 的 Δworst / ΔOverall |
> | **② 范围实际使用** | 新增层的调制**是否真的变大、且随属性分化** | ρ_l 与 **ρ^between_l** |
>
> **若 E2 的卷积 ρ_l 仍停在 E1 水平（~1-2%），正确结论是**：
> 「**在统一训练设定下，扩大可用条件化范围未改变模型行为，优化仍主要把条件信息放在分类头**」，
> **而不是**「深层位置本身无效」——后者是未经检验的过度推断（该位置从未被真正启用）。
- **逐层有效偏移（两个量，缺一不可）**，对属性取值等权平均、先逐单位求 norm 再平均：
  - **`ρ_l = E_a‖ΔW_l(a)‖_F / ‖W_l‖_F`（总偏移量级）**：该层权重被改动了多少。
  - **`ρ^between_l = sqrt(E_a‖ΔW_l(a) − mean_a ΔW_l‖_F²) / ‖W_l‖_F`（属性特异性偏移）**：偏移中
    **随属性变化**的部分有多大。
  > 🛑 **为什么必须两个都报（2026-07-16 补，修补 E1 的一处实质漏洞）**：**`ρ_l` 只量偏移大小，
  > 区分不了「所有属性产生*相同*偏移」（⇒ 该层退化为**静态 adapter / 重参数化**、根本没在做属性条件化）
  > 与「真正的属性特异性偏移」**。**只有 `ρ^between_l` 能证明该层在按属性分化**：
  > 若 `ρ_l` 大而 `ρ^between_l ≈ 0` ⇒ 该层**并未条件化**，只是把权重挪了个位置。
  > 零训练成本（条件仅 age 4 取值 ⇒ 每层只有 4 个不同 ΔW，纯 CPU 可算）。
- **属性置换评估**：真实 A vs 置换 A。子群划分**永远用真实属性**。**置换须整组、按数据集**：
  **MIMIC 在患者级联合置换完整 `(sex, race, age)` 元组**（不分别置换，以免制造数据中不存在的属性组合）；
  **HAM 在病灶级置换 age**。所有 cell 用相同置换索引（配对）。
  > 🛑 **该项能证明什么、不能证明什么（2026-07-16 收紧）**：推理时置换只能证明
  > **「模型没有忽略属性输入」**（Δ>0 ⇒ 打乱属性会掉性能）。它**不能排除**「性能差异来自额外参数量
  > 或训练正则化」——因为置换**不改变模型的参数量与训练过程**。
  > ⇒ **禁用**「排除了『只是多了可训练参数』」这一表述；**改为**「排除了『模型完全忽略属性输入』」。
  > 若 E2 真出现正向范围效应、且需作**纯机制归因**，才追加参数匹配或伪属性训练臂；否则**不扩张实验**。

> stage-knockout、BN 重估、常量属性多臂、伪属性重训练、D_l 多属性切换曲线**均降为补充材料/不做**——它们不是
> 回答主 RQ 的必要条件，容易把方案重新扩展成多个机制问题。

---

## 5. 工作清单

- [x] E0 `ResNet18CondNet`：`location` 开关（Head/Deep/Full）+ 结构自检（Δθ≈0 等价、模块嵌套 assert）+
      canonical base state + 接入 harness；报告各档 adapter 参数量与 FLOPs
      · 模型 `src/models/resnet18_condnet.py`（`ResNet18CondNet` + Age/Skin 子类，`location`∈{none,head,deep,full}）
      · 统一训练入口 `src/training/train_condnet.py` + `slurm/train_condnet.sh`（4 数据集 dispatch，走 harness.run_training）
      · 自检全过：权重侧 Δθ=0 bitwise、logits 近零一致、模块嵌套 head⊂deep(5conv)⊂full(19conv)、canonical base fc 跨 cell 逐元素相等
      · adapter 参数量（HAM/Age 口径）：ERM 0 / C-Head 283,460 / C-Deep 1,604,420 / C-Full 2,760,260（= plan §4.3 冻结值 L-all+head 2,760,260 ✓）
- [x] E-audit：label-only 冻结候选组与样本门槛 → full-target 复核现有 checkpoint 的 M→C 效应（判 go/no-go，
      §3.1）→ worst-group 动态-min 校准自检（**只估覆盖率/偏差，不切换主终点**）
      · **E-audit.1**（`scripts/eaudit_freeze_candidate_groups.py` → `outputs/conditioning_ablation/eaudit_candidate_groups.json`）：
        label-only 冻结。门槛 `min(n_pos,n_neg)>=50`（Hanley–McNeil SE 上界论证）。HAM-ID=age 4 组（4/4 过）、
        M2C-OOD=sex∪race∪age 6 边缘组（6/6 过），**无组被剔除**。
      · **E-audit.2**（作业 72196，`src/training/run_ood_cxr_full_target.py` + `scripts/eaudit_m2c_full_target.py`）：
        **判决 = GO**（HyperAdapt 对 ERM 与 SWAD 均为正且 CI 不含 0）⇒ 启动 E2 范围 OOD 训练。
        ⚠️ **但 full-target 复核实质性推翻了 legacy k→k 的「深层 HN 胜」图景**（详见 §0.2 修订）。
      · **E-audit.3**（`scripts/eaudit_dynamic_min_calibration.py`）：M2C 实测 argmin **100% 恒为 age:>=60**
        ⇒ 落在 `separated` 情景、动态 min≈固定 min，实测 min 算子偏差仅 **+0.0002**、Δ 判决可信。
        ⚠️ 校准模拟规模（9,000 样本）≈ **HAM** 规模而非 M2C（138,644）：其悲观数字（level 偏差 −0.027~−0.037、
        Δ 覆盖率 80~92%）**适用于 E1(HAM)、不适用于 E2(M2C)**——M2C 的 n 大 15×，组 AUC 估计精确、min 选择偏差消失。
- [x] E1 **主实验 A（HAM ID）**：ERM/C-Head/C-Deep/C-Full 训练 + OOF + H1/H2 检验 + ρ_l + 属性置换
      · **训练完成**（2026-07-15）：搜索 4 cell × 6 config × 5 折（作业 72210/11/12/20，120 run，57 min）
        + 确认 4 cell × 5 折 × seed{43,44}（作业 72357/65/66/67，40 run）。全 160 run 零报错。
        分区实测定为 **gpus24 + bs=128 全 cell 等 batch**（探针 72207：C-Full 峰值仅 11.08 GiB/47%，
        **推翻**了「HyperAdapt 类需 gpus48/bs32」的既有约定）。
      · **选定 config**（§3.2 五折 val marginal-worst 均值）：ERM `lr3e-05_wd1e-04` / C-Head `lr1e-04_wd1e-03`
        / C-Deep `lr1e-04_wd1e-04` / C-Full `lr1e-04_wd1e-04`（四 cell 各不相同 ⇒ 印证「全 cell 同等调参」必要）。
      · **H1/H2 判决（修复口径 fold_rank）**：**两者均未达显著**（见下方 §3.3 附录）。
      · **待办**：ρ_l 逐层有效偏移、属性置换（均复用 checkpoint、零额外训练）。

> ### 🛑 E1 发现：pooled-OOF 的**跨折校准漂移伪影**（影响面超出本方案）
> **现象**：C-Deep 的 val marginal-worst 最高（0.8169，四 cell 之首），pooled test 却最低（0.7088）。
> **根因（非 bug，是口径缺陷）**：5 折是**5 个不同模型**，logit 尺度因**早停轮数不同**而漂移
> （C-Deep seed44 fold2 训 21 epoch、其余 7–9 ⇒ 逐折 logit 中位数 `[-7.6,-4.0,-23.0,-6.0,-11.2]`，
> 跨折差 19 个单位）。直接池化 **raw logit** 把「折内排序」与「跨折尺度对齐」混在一起：该 run
> 逐折 test AUC 均值 **0.831（健康）**却池化成 **0.639（−0.192）**。且惩罚**因 cell 系统性不同**
> （ERM 仅 −0.01~−0.05，C-Deep 达 −0.19）⇒ **H1/H2 会测成校准漂移而非范围效应**。
> 实测 `corr(跨折 logit 离散度, 池化惩罚) = −0.607`。
> **这是学界已知缺陷**（发现后经检索核实，非本项目特有；术语 = **pooling vs averaging**、
> pooled AUC 的 **pessimistic bias**）：**Parker/Günter/Bedo (2007)** *BMC Bioinformatics* 8:326
> （无信号数据 pooled AUC 掉到 <0.3 而非 0.5；**排序类指标受害远重于 accuracy**）；
> **Airola et al. (2010)** *JMLR W&CP* 8:3–13（「部分正负对由**不同折**的样本构成」，推荐 **averaging**/LPOCV）；
> **Forman & Scholz (2010)** *SIGKDD Explorations* 12(1):49–57。本项目实测的是该缺陷的**新近因**
> （Parker 的近因是 stratification，本项目是**早停轮数异质**）。
> **修复 = 采用文献推荐的 `averaging`**（逐折算指标→折间平均，**无跨折正负对**）。
> （中途曾自创「折内秩归一 + 仍池化」作折中：把最差惩罚由 −0.192 压到 −0.051、相关性 −0.607→−0.136，
> 但**仍残留约 −0.01 的悲观偏**且非文献标准解 ⇒ **已降为敏感性口径** `--estimator pooled_rank`。）
> **后果量化（旧口径会制造高置信假阳性）**：
>
> | 口径 | C-Deep worst | H2=C-Full−C-Deep | Holm | 判决 | 最小有效配置 |
> |---|---|---|---|---|---|
> | `pooled_raw`（旧） | 0.7088 | **+0.0914** | **0.0060** | **「确认」❌ 假阳性** | C-Full |
> | `pooled_rank`（折中） | 0.8002 | +0.0075 | 0.6953 | 未达显著 | 无 |
> | **`averaging`（主口径）** | **0.8225** | **−0.0011** | **0.8392** | **未达显著 ✅** | **无** |
>
> ✅ **波及的既有结论已于 2026-07-16 全部复核完毕**（`scripts/build_oof_results_averaging.py`，零重训、
> 同一批预测只改聚合）：`docs/oof_regime_results.md` **已按 averaging 整体重构 = 唯一权威**。
> **推翻三条旧核心结论**：①「深层 HN 在 M→C 稳健胜」→ 只有 HyperAdapt 胜（HyperFusion +0.0058 显著
> → +0.0015 n.s.）；②「C→M 全 HN 显著更差」→ 打平/HyperAdapt 反而显著微优；③「强烈方向不对称」→ 不成立。
> **污染与数据集大小强相关**：HAM(9.7k) 跨折 logit SD 达 2.59、worst 修正 −0.023~−0.096；
> MIMIC(199k)/CheXpert(138k) 仅 −0.0005~−0.009（大 n 近乎免疫）。

#### 🛑 估计量变更：pooled-OOF → **averaging**（有依据偏离 plan §3，2026-07-15）

**依据**：pooled AUC 的悲观负偏是**学界已知缺陷**，非本项目特有：
- **Parker, Günter & Bedo (2007)**, *BMC Bioinformatics* 8:326 —— 低信号数据上 pooled AUC 严重负偏
  （无信号数据 AUC 掉到 <0.3 而非 0.5）；**排序类指标（AUC）受害远重于 accuracy**。
- **Airola et al. (2010)**, *JMLR W&CP* 8:3–13 ——「pooling 假设各折分类器来自**同一总体**……计算 AUC 时
  该假设**更可疑，因为部分正负对由不同折的样本构成**」，推荐 **averaging** 或 LPOCV。
- **Forman & Scholz (2010)**, *SIGKDD Explorations* 12(1):49–57 —— 各家聚合口径不兼容致文献不可比。

即本项目实测的伪影是该已知缺陷的**一个新近因**（Parker 的近因是 stratification；本项目是**早停轮数异质**
⇒ logit 尺度漂移）。**LPOCV 需按对重训、算力不可行**，故取 averaging。

**averaging 不损失评估 n**（回应「池化抬 n」的初衷）：抬 n 的功劳来自 **CV 本身**（每样本轮到一次 test
⇒ 全 9,707 张都有干净预测），**不是** pooling；两种聚合都用满这 9,707 张。AUC 方差由**类别样本数**支配
（`Var≈c/n_pos+c'/n_neg`）而非「对数」，故 5 折各自估计再平均（÷5）与池化精度相当。**实测反而更紧**：
worst-group bootstrap SE 之比 averaging/池化 = **0.72~0.92×**（C-Deep 收益最大——正是漂移最重的 cell
⇒ 跨折对只注入噪声、不带信息）。Airola 反对 averaging 的理由（小折 AUC 算不出）**在 HAM 不成立**：
每折每 age 组两类齐全。

**冻结的次序选择**：worst = **`min_g[mean_f AUC(g,f)]`（先平均后 min）**，非 `mean_f[min_g …]`。
Jensen 恒有后者 ≤ 前者；选前者因 ①「先 min」会让每折（age:20-40 每折仅 ~19 阳性）在最噪处取极值、
放大 E-audit.3 实测的 min 向下偏；②「模型在哪个年龄组最弱」应是模型属性，不该每折换答案。
**动态 argmin 仍保留**（每 replicate 重算组均值再取 min）。

实现：`scripts/e1_ham_analysis.py --estimator {averaging(主) | pooled_rank | pooled_raw}`，三口径并排可查。

#### E1 H1/H2 结果（**averaging 主口径**，B=1000 病灶级配对 cluster bootstrap，n=9,707 / 7,280 病灶）

| cell | Overall | marginal worst | gap | Δworst vs ERM |
|---|---|---|---|---|
| ERM | 0.8925 | **0.8227** | 0.0887 | — |
| C-Head | 0.8954 | 0.8194 | 0.0950 | −0.0033 n.s. |
| C-Deep | 0.8947 | 0.8225 | 0.0889 | −0.0002 n.s. |
| C-Full | 0.8963 | 0.8214 | 0.0912 | −0.0013 n.s. |

| 假设 | Δworst | 95% CI | p_fair | ΔOverall | p_joint | **Holm** | 判决 |
|---|---|---|---|---|---|---|---|
| H1: C-Deep − C-Head | +0.0030 | [−0.015,+0.029] | 0.458 | −0.0007 | 0.458 | 0.839 | 未达显著 |
| H2: C-Full − C-Deep | −0.0011 | [−0.046,+0.020] | 0.420 | +0.0016 | 0.420 | 0.839 | 未达显著 |

- **两个确认性假设均未达显著，效应量级均 <0.005（「微弱」档）**；三档 Overall 相对 ERM 均**微正**
  （+0.002~+0.004）⇒ **扩大条件化范围没掉整体性能，但也没换来 worst-group 公平收益**。
- **最小观测有效配置 = 无**（三档 Δworst vs ERM 均 <0）⇒ 报「所测范围内未观测到有效配置」。
- **口径稳健性**：H1/H2 在 averaging 与 pooled_rank 下**符号相反但均 n.s.**（H1 +0.0030 vs −0.0042；
  H2 −0.0011 vs +0.0075），差异全在噪声内 ⇒ **「无范围效应」这一结论对口径选择稳健**；
  唯 `pooled_raw` 给出假阳性「确认」（见上）。
- **argmin 稳定性（tied，须据此收严表述）**：argmin 在 **age:80+ 与 age:20-40 之间频繁翻转**
  （ERM 53%/47%、C-Head 66%/34%、C-Deep 67%/33%、C-Full 56%/43%）⇒ 落在 E-audit.3 校准的
  **`tied`~中间**区间，Δ 估计**偏保守**（实测偏差 −0.006、覆盖率 80%）。故上述 null **不可**解读为
  「范围确无效应」，只能说「**未观测到可检出的增量**」。

#### E1 机制诊断（plan §4，作业 72403，复用 checkpoint 零额外训练）

**① 逐层有效偏移（4 age 组等权 × 5 折 × 3 seed 平均）—— manipulation check**

| cell | 层 | `ρ`（总偏移） | **`ρ^between`（属性特异）** | between/ρ |
|---|---|---|---|---|
| C-Head | **fc** | 1.066 | **0.704** | 66.0% |
| C-Deep | layer4 全部 conv | 0.020 | **0.016** | ~77% |
| C-Deep | **fc** | 1.083 | **0.753** | 69.5% |
| C-Full | layer1→layer4 conv | 0.0126→0.0163 | **0.0096→0.0121** | ~75% |
| C-Full | **fc** | 0.954 | **0.638** | 66.9% |

> **🔑 这是 H1/H2 null 的机制解释（且已通过 `ρ^between` 的 manipulation check）**：
> - **fc 在做真正的属性条件化，不是静态重参数化**：其 `ρ^between ≈ 0.64~0.75`（属性特异性偏移达
>   **基权重范数的 ~70%**）。⚠️ 这一步是**必需的**——单看 `ρ≈1` **无法**区分「真条件化」与
>   「所有属性产生相同偏移的静态 adapter」；`ρ^between` 才排除了后者。
> - **conv 的属性特异性偏移仅 0.0096~0.016，比 fc 小约 45~60 倍** ⇒ **条件化几乎全部由 fc 承担，
>   backbone conv 的调制近乎可忽略**。而 fc 在三个 conditioning cell 间**恒定**（§1 的设计）
>   ⇒ 范围扩展多加的 conv adapter **优化并未真正用起来**，三档行为本就差别极小。
> - **「挂了 adapter」≠「用了 adapter」** ⇒ H1/H2 的 null **与机制自洽，不是单纯功效不足的偶然**。
> - **正确表述**：「在**当前统一低秩训练设定**下，扩大可用条件化范围**未改变模型行为**，优化仍主要把
>   条件信息放在分类头」；**不得**说「深层位置本身无效」——**该位置从未被真正启用**，后者是过度推断。
> - **另**：ρ 与 ρ^between 均随深度**单调递增**（layer1 0.0096 → layer4 0.0121）⇒ **越浅的层被用得越少**。
> - **附带观察**：`between/ρ` 在所有层稳定在 **66~77%** ⇒ 各层偏移都以**属性特异成分为主**，
>   约 1/4~1/3 是跨属性共有的静态成分。
> 实现：`scripts/e1_rho_and_permutation.py:compute_rho`（零训练成本：条件仅 age 4 取值 ⇒ 每层只有
> 4 个不同 ΔW，纯 CPU）；数字见 `outputs/conditioning_ablation/ham10000/cv5/e1_rho_between.json`。

**② 属性置换（病灶级整组置换 age，n_perm=20，分组键恒用真实 age）**

| cell | 真实 age (Ov / worst) | 置换 age (均值±SD) | **Δ(真实−置换)** |
|---|---|---|---|
| **ERM** | 0.8925 / 0.8227 | 0.8925±0.0000 / 0.8227±0.0000 | **+0.0000 / −0.0000** ✅ 自检 |
| C-Head | 0.8954 / 0.8194 | 0.8835±0.0017 / 0.8108±0.0055 | +0.0119 / **+0.0086** |
| C-Deep | 0.8947 / 0.8224 | 0.8746±0.0028 / 0.7979±0.0078 | +0.0202 / **+0.0245** |
| C-Full | 0.8963 / 0.8213 | 0.8816±0.0019 / 0.8025±0.0087 | +0.0147 / **+0.0188** |

- **ERM 的 Δ 恒为 0（实测 1.1e-16）**——内建正确性自检通过（ERM 忽略 age，置换必然无效应）。
- **三个 conditioning cell 的 Δ 全部 >0**（打乱即掉 0.019~0.025）⇒ **排除「模型完全忽略属性输入」**。
  > 🛑 **表述边界（2026-07-16 收紧）**：推理时置换**不改变模型的参数量与训练过程**，故它
  > **不能排除**「性能差异来自额外参数量或训练正则化」。**禁用**「排除了『只是多了可训练参数』」
  > 这一（此前误用的）说法。纯机制归因须待 E2 出现正向范围效应后，再追加参数匹配或伪属性训练臂。

> **🔑 E1 的核心矛盾（写作要点）**：**模型确实在用 age（置换会掉），但用了 age 却并不比
> attribute-blind 的 ERM 更好**（三档 Δworst vs ERM 全 ≈0）。
> 🛑 **收紧后的表述（2026-07-16；旧稿「瓶颈在数据侧的信号强度、不在架构侧的机制」判断过强，已作废）**：
> > **模型能够利用 age 条件（ρ^between 证明 fc 确在按属性分化、置换证明预测确实依赖它），但在
> > **当前统一低秩训练设定**下，条件化主要集中于**分类头**，未转化为相对 ERM 的净收益。
> > **数据条件信号缺乏**（新 OOF V-info 闸门：HAM-age **未检出、≈0**；原写「>0 但弱」已过时）**与深层调制利用不足**（conv 的 ρ^between 比 fc 小
> > 45~60 倍）**均可能构成限制**，本实验无法区分二者。**
>
> ⇒ 不可单方面归因于数据侧：E1 同时发现 **conv adapter 基本未被启用** ⇒ **优化/参数化也可能是瓶颈**。
> （此前呼应记忆 `mimic-fairness-signal-reframing` 的单侧归因已修正。）
> 实现：`scripts/e1_rho_and_permutation.py`（logits 表预计算使 20 次置换退化为查表，零额外前向）。

> #### E1 敏感性实验（探究「训练策略是否造成 conv adapter 未被启用」，2026-07-16）
> 三个补充实验，均**未推翻** E1 结论（条件化几乎全在 fc、conv 行为近乎惰性），确证「conv 未被启用」
> **不是** checkpoint 选择、事后归因、或 fc 挤占的伪影，而是当前统一低秩训练设定的稳健属性：
> - **① 换 `best_worstcase` checkpoint 复评**（`docs/conditioning_ablation_e1_worstcase_checkpoint.md`）：
>   config 不变、仅换评估权重。conv ρ^between 仍 1~2%、fc/conv 仍 48~63×；H1/H2 仍 n.s.；
>   若有别，反而 fc 占比更高、C-Full worst 更差。脚本加 `--checkpoint`；新增纯 CPU `--mode rho`。
> - **② 通路 knockout × 置换**（`docs/conditioning_ablation_e1_pathway_knockout.md`，
>   `scripts/e2_knockout_permutation.py`）：对 C-Deep/C-Full 关 fc（ΔW_fc=0）/ 关 conv（M_conv=0）后各自
>   比较真实 vs 置换 age。**关 fc 后 Δoverall 塌到 ≈0、C-Deep worst 塌到 0.9σ** ⇒ age 依赖主要由 fc 承担；
>   唯 **C-Full worst 残留 +0.0084 但仅 1.6σ**（未过置换噪声）⇒ 深层 conv 至多有微弱、不稳健的 worst 功能。
>   `full` 变体精确复现 `e1_permutation.json`（管线自检）。
> - **③ 关 fc 从头训练 C-Deep/C-Full**（`docs/conditioning_ablation_e3_fcoff_training.md`，
>   `scripts/e3_nofc_analysis.py`；新增 location `deep_nofc`/`full_nofc` + `--diagnostics` 逐 epoch 探针）：
>   检验 conv 能否被逼激活。**答：能被优化但无用**——关 fc 后 conv ρ^between 随训练增长、shared-A 离零、
>   **B 梯度解锁（否定「B 永冻」硬瓶颈）**，但**激活 conv 越多性能越差**；早停操作点上 conv ρ^between 仅
>   1.0~1.4× fc-on（仍比 fc 小 35~60×），行为退回近属性无关（C-Deep-noFC≈ERM、C-Full-noFC 劣于 ERM）。
>   ⇒ **移除 fc 是丢失条件化而非重路由进 conv；fc 几乎是唯一能有用表达弱 age 信号的通路。**
>   **指向**：要 conv 承载有用条件化须改**参数化本身**（乘性→加性/更高 rank/判别方向正则），非仅关 fc/调训练时长。
- [x] E2 **主实验 B（M→C OOD）**：160 run 训练 + 60 次 full-target OOD 评估 + 双侧 H1/H2 + 稳健性 + ρ^between
      （作业：搜索 72434/35/38/39、确认 72640/41/42/49、OOD 72715；脚本 `run_e2_ood_full_target.py`/`e2_ood_analysis.py`）
- [~] E3 负控制（条件触发）：**判定 = 跳过**（依据见下 E2 §5 结论：E2 无「校正后显著且 Δworst≥0.005」的结果）
- [x] E4 汇总：两 regime 的 H1/H2 判决 + 最小观测有效范围 + 写作 → **`docs/conditioning_ablation_summary.md`**

**分阶段**：HAM ID 主结论 ✓ → M→C full-target OOD 范围结论 ✓ → 负控制（**未触发，跳过**）→ 汇总（下一步）。

#### E2 结果（M→C OOD，full-target + 患者级配对 cluster bootstrap，B=1000，n=138,644 / 46,799 患者）

| cell | Overall | marginal worst |
|---|---|---|
| ERM | 0.8503 | 0.8132 |
| C-Head | 0.8504 | 0.8131 |
| **C-Deep** | 0.8503 | **0.8143** |
| C-Full | 0.8497 | 0.8126 |
| SWAD | 0.8482 | 0.8085 |

| 假设（双侧） | Δworst | 95% CI | Holm p2 | ΔOverall | 三分判决 |
|---|---|---|---|---|---|
| **H1: C-Deep − C-Head** | +0.0012 | [+0.0005,+0.0020] | 0.004 | −0.0001（非劣） | **「改善」**（但见稳健性↓） |
| **H2: C-Full − C-Deep** | −0.0016 | [−0.0024,−0.0008] | 0.004 | −0.0006（非劣） | **「有害」** |

**历史正向复现（缩为 C-Full）**：**C-Full vs ERM = −0.0005 n.s.**（Holm p2=0.324）⇒ **HyperAdapt 位置未复现
legacy M→C 正向**；C-Full vs SWAD = +0.0041 显著，但 **M→C 中 SWAD 本身劣于 ERM**（0.8085 vs 0.8132）
⇒「胜 SWAD」只是 SWAD 偏弱、**不作 HN 有效证据**。（描述性）C-Deep vs ERM = +0.0011。

> 🛑 **改双侧的价值即刻兑现**：H2 是**负向且 CI 全<0**（「有害」）。若沿用旧的单侧「前者更优」，H2 会被
> 记成 n.s.、**漏掉这个退化方向**。这印证了 §3.3「E2 双侧」修订的必要性。

> ⚠️ **两个诚实边界（必须随结论一起表述）**：
> 1. **所有效应 |Δ| ≤ 0.0016，远在「微弱档」（<0.005）以下** ⇒ **统计可检出但实践可忽略**。n=138,644
>    把千分位差别推成显著；worst-group AUC 差 0.001 无临床/实践意义。
> 2. **H1 的「改善」训练稳定性不足**（§3.3 强制稳健性分析当场抓住）：H1 的 **5 个 fold 效应 =
>    [+0.0007,+0.0084,+0.00002,−0.0015,−0.0015]，仅 3/5 正、LOFO 去某折即变号（1/5）**。主 bootstrap
>    的紧 CI 是**「给定这 15 个模型」的条件性 CI**，训练随机性不在内。**按 §3.3 判读：只能称「目标样本
>    层面显著，但训练稳定性不足」，不得称「范围扩大带来改善」。** 对比 H2（fold 1/5 正、LOFO 0/5 变号）
>    在训练随机性下更稳、但同样幅度可忽略。

#### E2 ρ / ρ^between（manipulation check，与 HAM 关键性不同）

| | fc ρ | fc ρ^between | conv ρ | conv ρ^between | fc/conv 比 |
|---|---|---|---|---|---|
| **HAM (E1)** | ~1.0 | ~0.70 | 0.013~0.02 | 0.010~0.016 | **45~60×** |
| **MIMIC (E2)** | **0.02~0.04** | **0.02~0.04** | 0.005~0.010 | 0.004~0.009 | **仅 ~5×** |

- **MIMIC 上「条件化」整体极弱**：fc ρ 仅 0.04（HAM ~1.0，**弱 ~25×**）⇒ 与 MIMIC 弱信号（`I(Y;A|X)` 弱）
  自洽——**信号弱 ⇒ 模型几乎不条件化**。
- **但那点微弱偏移确是属性特异的**（ρ^between/ρ ≈ 0.9~0.97）⇒ 非静态重参数化、是真条件化，只是量级微不足道。
- **两数据集的 null 同向不同因**：HAM = 「fc 强条件化、conv 未被启用」；MIMIC = 「信号太弱、哪里都几乎
  不条件化」。⇒ ρ^between 的价值：**没有它只会笼统说「conv 没用」，看不出 MIMIC 其实是整体弱**。

#### E3 触发判定（依 §3.3 冻结规则）

E2 **无任何「校正后显著且 `Δworst ≥ 0.005`」的结果**（H1 名义改善但仅 +0.0012 且训练不稳；H2 有害；
C-Full vs ERM n.s.）⇒ **触发条件不满足 ⇒ 跳过 E3 负控制**（CheXpert ID 范围网格 + C→M 不训练）。
理由（§3.3）：效应本就在「微弱档」以下，负控制无从区分「真无效」与「噪声」，纯属浪费算力。
