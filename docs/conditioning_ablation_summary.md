# 条件化范围消融 —— 汇总报告（E4）

> 🛑 **信号闸门口径更新（2026-07-19）**：闸门已由单-split 条件互信息改为 **OOF conditional V-information**
> （Xu 2020 / Hewitt 2021；权威 = `docs/conditional_v_information_gate.md`）。关键变化：
> **HAM-age 由「唯一正信号」退回「未检出」**（+0.0028 bits，CI 触 0）；Fitzpatrick ≈0；
> **CheXpert / MIMIC 全轴 CI<0 = 决定性 0**；唯一 detected 为 **PAPILA-age**（+0.041 bits，但 n=420）。
> control task 五库全部通过 ⇒ 零读数非假阴。本文「HAM-age `I(Y;age|X)>0`」一族表述**已过时**（下方已内联订正）；
> 各处**结论方向不变**，但「有信号」前提须按新闸门重读。


> 建立 2026-07-19。方案规格：`docs/conditioning_ablation_plan.md`（权威，含逐项判决与工作清单）。
> 本文是**面向读者的收尾叙事**，把 E0–E3 串成一条结论线；所有数字均可回溯到方案文档与
> `outputs/conditioning_ablation/` 下的结果 JSON。

---

## 0. 一句话结论

**在受控消融下（锁死参数共享 / rank / 参数化 / 初始化 / 编码器，只改条件化范围），把敏感属性条件化
从分类头逐步扩展进 backbone，在 HAM10000（ID）与 MIMIC→CheXpert（OOD）上均不带来实践意义的
worst-group 公平收益。** 命名模型「越深越好」的表象来自范围以外的混杂轴，而非范围本身。

---

## 1. 研究问题与设计

**RQ**：敏感属性条件化从仅影响分类头（head-only），扩展到深层单 stage、再到完整 backbone 时，
是否改善整体性能与 worst-group fairness？增益来自深层就够，还是须铺满全 backbone？

三个现成模型（HyperHead / HyperFusion / HyperAdapt）在**范围、参数化、调制、容量、初始化**多轴
同时不同、不可比。本方案用**统一模型 `ResNet18CondNet`**、只暴露 `location` 一个开关，得到严格嵌套、
逐档可比的四个 cell：

| cell | 条件化范围 | 对齐 |
|---|---|---|
| **ERM** | 无 | 参照零点 |
| **C-Head** | 仅 fc | head-only（HyperHead 位置） |
| **C-Deep** | fc ＋ layer4 全 conv | 深层单 stage |
| **C-Full** | fc ＋ layer1–4 全 conv | 全 backbone（HyperAdapt 位置） |

严格嵌套 C-Head ⊂ C-Deep ⊂ C-Full；fc 条件化三档恒定 ⇒ 增量干净。锁死设定经 adapter 参数量
交叉验证（C-Full = 2,760,260，与方案 §4.3 冻结值逐位吻合）。

**两个确认性主假设**：H1 = C-Deep − C-Head（属性是否需进 backbone）；H2 = C-Full − C-Deep（向中浅层
继续扩展是否有额外效用）。

---

## 2. 关键方法学基建（贡献，非仅工具）

本方案相较 HyperAdapt 原论文的 depth ablation，多出三项方法学贡献，且过程中发现并修复了一个
**波及全项目**的评估缺陷：

### 2.1 canonical base state（E0）
每 `(dataset, fold, seed)` 用确定性种子冻结同一份 base backbone / base fc，使所有 cell（含比较用 ERM）
的 base 权重逐元素相等——否则「Δθ=0 等价 ERM」只对 cell 自身成立、不对比较 ERM 成立。

### 2.2 🛑 pooled-OOF 的悲观负偏（学界已知缺陷，本项目实测并改口径）
E1 分析中发现：直接**池化各折 raw logit 算一个 AUC** 会构造出**跨折正负对**（折 i 阳性 vs 折 j 阴性），
而各折是**不同模型**、logit 尺度因早停轮数异质而漂移。这是学界已知缺陷：
**Parker/Günter/Bedo (2007)** *BMC Bioinformatics* 8:326；**Airola et al. (2010)** *JMLR W&CP* 8:3–13（推荐
**averaging**）；**Forman & Scholz (2010)** *SIGKDD Explorations* 12(1):49–57。
- **后果**：污染程度**因方法系统性不同**（HAM 上 ERM −0.01~−0.05 vs C-Deep −0.19）⇒ 会把校准漂移
  测成范围效应。旧口径下 H2 曾给出**假阳性「确认」**（Holm=0.006，averaging 下真值 +0.0075/p=0.70）。
- **修复 = averaging**（逐折算指标→折间平均，无跨折正负对）。**不损失评估 n**（抬 n 靠 CV 本身，非
  pooling；实测 SE 反而小 8–28%）。
- **波及既有结论已全部复核**：`docs/oof_regime_results.md` 已按 averaging 整体重构，推翻旧版三条 OOD
  核心结论（「深层 HN 稳健胜」「C→M 全 HN 更差」「强方向不对称」——详见该文档）。

### 2.3 worst-group 动态-min 校准（E-audit.3）
worst = `min_g AUC_g` 对含噪组 AUC **向下有偏**。Monte-Carlo 校准显示：HAM 规模（n≈9.7k）的 level 侧
偏差 −0.03~−0.04、覆盖率低至 18%（`tied` 情景）；**配对差值 Δ** 侧偏差小得多（−0.006、覆盖率 80~92%）。
⇒ 小数据集的 null **只能表述为「未观测到可检出的增量」，不得说「范围确无效应」**。

### 2.4 患者/病灶级配对 cluster bootstrap
预处理只做「患者级分组划分」（防跨 split 泄漏），**未按患者去重**（CheXpert 83% 样本属多图患者）
⇒ 必须按 cluster 重采样，否则 CI 偏窄。

---

## 3. 两个主实验的结果

### 3.1 E1 — HAM10000（ID）

> ⚠️ **订正**：本节原称 HAM 为「`I(Y;age|X)>0` 的明确条件信号」。新 OOF V-info 闸门下
> **HAM-age 未检出（≈0）** ⇒ H1/H2 的 null **更属预期**，与 §4.2 的机制诊断自洽（模型在 fc 努力条件化一个
> 几乎不含可用信息的属性）。结论方向不变。

**averaging 主口径，B=1000 病灶级 cluster bootstrap，n=9,707 / 7,280 病灶**：

| cell | Overall | marginal worst | Δworst vs ERM |
|---|---|---|---|
| ERM | 0.8925 | 0.8227 | — |
| C-Head | 0.8954 | 0.8194 | −0.0033 n.s. |
| C-Deep | 0.8947 | 0.8225 | −0.0002 n.s. |
| C-Full | 0.8963 | 0.8214 | −0.0013 n.s. |

- **H1 = +0.0030（Holm 0.839）、H2 = −0.0011（Holm 0.839）⇒ 均未达显著。**
- **最小观测有效配置 = 无**（三档 Δworst vs ERM 均 <0）。扩范围不掉整体（Overall 微升），也不换公平。
- **口径稳健**：averaging 与 pooled_rank 下 H1/H2 符号相反但均 n.s.，差异全在噪声内。

### 3.2 E2 — MIMIC→CheXpert（OOD，弱条件信号，go/no-go=GO）

**full-target（每 fold×seed 评完整 target、等权平均），B=1000 患者级 cluster bootstrap，n=138,644**：

| cell | Overall | marginal worst |
|---|---|---|
| ERM | 0.8503 | 0.8132 |
| C-Head | 0.8504 | 0.8131 |
| C-Deep | 0.8503 | 0.8143 |
| C-Full | 0.8497 | 0.8126 |
| SWAD | 0.8482 | 0.8085 |

**双侧判决（§3.3 修订：E2 改双侧）**：
| 假设 | Δworst | 95% CI | Holm p2 | 三分判决 |
|---|---|---|---|---|
| H1: C-Deep − C-Head | +0.0012 | [+0.0005,+0.0020] | 0.004 | 「改善」* |
| H2: C-Full − C-Deep | −0.0016 | [−0.0024,−0.0008] | 0.004 | **「有害」** |
| C-Full vs ERM（复现） | −0.0005 | [−0.0016,+0.0005] | 0.324 | n.s. |

三条边界**必须随结论一起表述**：
1. **所有 |Δ| ≤ 0.0016 < 0.005（微弱档）⇒ 统计可检出但实践可忽略**（n=138k 把千分位推成显著）。
2. **H1 的「改善」训练稳定性不足**：5 fold 效应仅 3/5 正、LOFO 去一折即变号。主 CI 是「给定这 15 个
   模型」的**条件性** CI，训练随机性不在内 ⇒ 只能称「目标样本层面显著、训练稳定性不足」。
3. **C-Full 未复现 legacy M→C 正向**（vs ERM n.s.）；「胜 SWAD」只因 SWAD 本身劣于 ERM，不作证据。
> **改双侧的价值即刻兑现**：H2 负向「有害」，旧单侧「前者更优」会漏掉这个退化方向。

---

## 4. 机制诊断：为什么范围轴是 null（本方案最有价值的产出）

范围轴 null 有两种可能——(a) 深层位置本身无用，(b) 深层位置从未被优化真正启用。**ρ 诊断把二者分开**。

### 4.1 逐层有效偏移 ρ 与属性特异偏移 ρ^between
- **ρ** = `E_a‖ΔW_l(a)‖_F/‖W_l‖_F`（该层被改多少）；
- **ρ^between** = `sqrt(E_a‖ΔW_l(a)−mean_a ΔW_l‖_F²)/‖W_l‖_F`（偏移中**随属性变化**的部分）。
- ⚠️ **只报 ρ 不够**（外部评审指出）：ρ 大也可能是「所有属性产生相同偏移」= 静态重参数化、根本没
  条件化；**只有 ρ^between 证明按属性分化**。

| | fc ρ | fc ρ^between | conv ρ | conv ρ^between | fc/conv 比 |
|---|---|---|---|---|---|
| **HAM (E1)** | ~1.0 | ~0.70 | 0.013~0.02 | 0.010~0.016 | 45~60× |
| **MIMIC (E2)** | 0.02~0.04 | 0.02~0.04 | 0.005~0.010 | 0.004~0.009 | ~5× |

### 4.2 两数据集的 null 同向但**不同因**
- **HAM**：**fc 强条件化（ρ^between~0.70）、backbone conv 近乎未被启用**（fc/conv 45~60×）。
  范围扩展多加的 conv adapter **优化根本没用起来** ⇒ 三档行为本就差别极小。
- **MIMIC**：**信号太弱、哪里都几乎不条件化**（fc ρ 仅 0.04，比 HAM 弱 25×），但那点微弱偏移仍属性
  特异（ρ^between/ρ≈0.9~0.97，是真条件化）。
- ⇒ ρ^between 的价值：**没有它只会笼统说「conv 没用」，看不出 MIMIC 其实是整体弱**。

### 4.3 属性置换（行为层面佐证）
真实 A vs 病灶/患者级整组置换 A：三个 conditioning cell 的 Δ 全 >0（HAM worst +0.009~+0.025）⇒
**排除「模型完全忽略属性输入」**。⚠️ 但置换不改变参数量/训练过程，**不能**排除「性能差异来自额外
参数量或正则化」。

### 4.4 三个敏感性实验：确证「conv 未启用」非伪影，并精细归因（HAM，best_overall 口径）

E1 的机制结论——「深层 conv 未被启用」——面临三种「其实是别的原因」的替代解释。三个敏感性实验
逐一排除，把结论从「conv ρ 小」推进到「conv 通路可优化但对本任务无用」。

**① 换 `best_worstcase` checkpoint 复评**（`docs/conditioning_ablation_e1_worstcase_checkpoint.md`）
——排除「是 checkpoint 选择策略的伪影」。只改评估所用的 checkpoint（两份在同一次训练并行保存），
config 不变。结果：
- conv `ρ^between` 几乎不变（C-Deep 0.0158→0.0167、C-Full 0.0108→0.0122），**fc/conv 仍 48~63×**；
- **fc 的偏移不降反升**（若有区别是把条件化**更集中到 fc**，与「worst-case 选点会调动深层」相反）；
- H1/H2 仍全部不显著；`best_worstcase` 整体压低性能且 **C-Full worst 掉到 0.795、gap 张到 0.107**。
⇒ **「深层 conv 未启用」不随「保哪一档权重」改变，是训练设定的稳健属性。**

**② 通路 knockout × 属性置换**（`docs/conditioning_ablation_e1_pathway_knockout.md`）
——从**行为层面**（预测对 age 的依赖）独立验证 ρ 的**权重层面**结论。对 C-Deep/C-Full 分别关 fc 条件化
（仅留 conv）或关 conv（仅留 fc），比较真实 vs 病灶级置换 age 的 Δ：

| cell | 变体 | Δworst | σ | 保留比例 |
|---|---|---|---|---|
| C-Deep | full（两通路） | +0.0245 | 3.1σ | 100% |
| | **仅 conv（关 fc）** | **+0.0018** | 0.9σ | **7%** |
| | 仅 fc（关 conv） | +0.0211 | 3.3σ | 86% |
| C-Full | full | +0.0188 | 2.2σ | 100% |
| | **仅 conv（关 fc）** | **+0.0084** | 1.6σ | **45%** |
| | 仅 fc（关 conv） | +0.0161 | 2.7σ | 86% |

⇒ **Overall 的 age 依赖 100% 由 fc 承担**（关 fc 后 Δoverall≈0）；worst-group 上 **C-Deep 的 conv 完全惰性**
（关 fc 后塌到 0.9σ 噪声内），唯 **C-Full 的 conv 残留 +0.0084 但仅 1.6σ、未过噪声**（至多暗示性）。
`full` 变体精确复现主置换结果（管线自检）。

**③ 关 fc 从头训练 C-Deep/C-Full**（`docs/conditioning_ablation_e3_fcoff_training.md`）
——排除「fc 只是阻力最小路径、conv 本可承担」。新增 `deep_nofc`/`full_nofc`（`condition_fc=False`），
其余全同 E1，检验超网络能否被「逼」着把 age 路由进 conv：
- **conv 可被优化（否定「梯度锁死」硬瓶颈）**：关 fc 后逐 epoch conv `ρ^between` 单调增长
  （Deep 0.012→0.052）、shared-A 离开零（‖A‖ 0.6→2.1）、**B 生成器梯度从 ~0 解锁**（0.004→0.031）；
- **但激活 conv 与性能反向**：conv 越活，val Overall/worst 越差（Deep ep10 掉到 0.862/0.585）；
- 早停在 val-Overall 峰值（~epoch 3）选操作点，此处 conv `ρ^between` 与 fc-on 版**几乎相同**（1.0~1.4×）、
  仍比 fc 小 35~60×；行为退回**近属性无关**（**C-Deep-noFC worst 0.828 ≈ ERM 0.823；C-Full-noFC 0.806 < ERM**）。

**三实验闭环的精细判决**：移除 fc **不是**把条件化重路由进 conv，**而是丢失条件化**。低秩**乘性 conv 调制**
可优化，但它学到的东西**对本任务无用**（激活反而降性能），故性能目标自动把 conv 压到最小、即便 fc 不在。
**fc 的加性低秩头几乎是这套超网络唯一能有用表达（弱）age 信号的通路。**
> **指向**：要让 conv 承载有用条件化，须改**参数化本身**（乘性→加性 / 更高 rank / 判别方向正则），
> 而非仅关 fc、调 checkpoint 或训练时长。

---

## 5. 表述边界（禁止过声）

1. 受控消融至多说明「**在本数据集、ResNet-18 与所测模型族内**，扩大到某范围是否带来可重复的增量
   效用」。H1/H2 是范围间**优效/劣效比较**，**不能**证明某些层「必要」或「无用」——全文禁用「必要
   成分 / 充分条件 / 范围阈值」。
2. **「深层位置本身无效」是过度推断**：ρ 诊断显示该位置**从未被真正启用**。正确表述 = 「在当前统一
   低秩训练设定下，扩大可用条件化范围未改变模型行为」。
3. 小数据集（HAM）的 null 因动态-min 偏保守，只能说「未观测到可检出的增量」。
4. E2 的「显著」在大 n 下**统计可检出 ≠ 实践重要**。

---

## 6. 与命名模型比较的关系（为什么受控消融是必要的）

`docs/oof_regime_results.md`（averaging 权威版）显示命名模型排序 **HyperAdapt（最遍布）> HyperFusion
≈ HyperHead**，方向似乎支持「越深越好」；HyperAdapt 是唯一在 MIMIC-ID/M→C/C→M 一致显著为正的 HN
（但幅度全 <0.005）。**然而三者在范围/参数化/容量/初始化上同时不同、混杂**。

本受控消融（唯一只改范围）在 HAM 与 M→C **均判 null**，且 ρ 诊断显示深层 conv 未被启用
⇒ **HyperAdapt 的微弱一致优势很可能来自范围以外的轴（容量/参数化/初始化），而非「条件化更深」本身。**
这正是命名模型比较无法回答、而受控消融能回答的问题。

---

## 7. 工作清单最终状态

| 项 | 状态 | 结论 |
|---|---|---|
| E0 模型 + 自检 + canonical base state + harness | ✅ | 锁死设定经参数量逐位验证 |
| E-audit（候选组冻结 / full-target go-nogo / 动态-min 校准） | ✅ | go/no-go=GO |
| E1 主实验 A（HAM ID） | ✅ | H1/H2 均 n.s.；最小有效配置=无；ρ 释因 |
| E2 主实验 B（M→C OOD） | ✅ | 双侧 H1「改善」但训练不稳、H2「有害」，均实践可忽略；C-Full 未复现 |
| E3 负控制 | ⏭ 跳过 | E2 无「校正后显著且 Δworst≥0.005」结果，触发不满足 |
| E4 汇总 | ✅ | 本文档 |

---

## 8. 关键产物索引

- **方案/判决权威**：`docs/conditioning_ablation_plan.md`
- **跨方法 averaging 权威结果**：`docs/oof_regime_results.md`
- **E1 敏感性实验**：`docs/conditioning_ablation_e1_{worstcase_checkpoint,pathway_knockout}.md`、`conditioning_ablation_e3_fcoff_training.md`
- **结果 JSON**：`outputs/conditioning_ablation/e2_ood_results.json`、`ham10000/cv5/{e1_ham_results_averaging,e1_rho_between}.json`、`mimic_cxr/cv5/e2_rho_between.json`
- **模型/训练/分析脚本**：`src/models/resnet18_condnet.py`、`src/training/{train_condnet,run_e2_ood_full_target}.py`、`scripts/{e1_ham_analysis,e2_ood_analysis,e1_rho_and_permutation,build_oof_results_averaging}.py`
