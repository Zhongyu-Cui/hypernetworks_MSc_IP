# OOD 实验 v2 结果（MIMIC↔CheXpert）

> 预注册：`docs/ood_experiment_v2_preregistration.md`（含两条偏离记录）。
> 口径：**full-target** 估计量 · **5 fold × 3 trial = 15 replicate** · 患者级配对 cluster bootstrap。
> 日期：2026-07-29

---

## 0. 一句话结论

> **HyperAdapt 在 MIMIC↔CheXpert 上的 OOD worst-group 优势不成立。** 四条预注册假设 **H1/H2/H3 全部
> 否定、H4 不可评估**：优势在两方向未一致复现（C→M **变号为显著劣**）、效应量在两方向均**小于训练
> 噪声**、且 M→C 的校准显著更差。现行 `docs/oof_regime_results.md` §1.6/§1.7 的 OOD 结论须撤回重述。

---

## 1. 假设判定总表

| 假设 | 内容 | 判定 |
|---|---|---|
| **H1** | S1 下 HyperAdapt marginal-worst 优于 ERM，**两方向复现** | ❌ **不成立**（C→M 为 −0.0025） |
| **H2** | 效应量超过训练噪声方差分量 | ❌ **不成立**（两方向 \|d̄\| < σ_train） |
| **H3** | 校准不劣于 ERM，且优势在重校准后仍在 | ❌ **不成立**（M→C 校准显著更差） |
| **H4** | 位于 ID–OOD 拟合线之上 | ⚠️ **不可评估**（见偏离 #1） |

---

> **📌 2026-08-14 范围扩充**：按预注册**偏离 #3** 加入 **GroupDRO** 第四臂（双向、同口径 15
> replicate），新增 **H5** 并将 Holm family 由 4 扩为 **6**。**既有 4 项检验在新 family 下全部保持
> 显著，本文件所有数字逐位不变**；GroupDRO 的结果与判定见 `docs/ood_experiment_v2_groupdro.md`
> （一句话：两方向均显著劣于 ERM，且是全 v2 唯一超过训练噪声的效应——一个负效应）。

## 2. 主结果（marginal worst-group AUC，full-target，15 replicate）

| 方向 | ERM | SWAD | HyperAdapt | GroupDRO（偏离 #3 追加） |
|---|---|---|---|---|
| M→C | 0.8125 | 0.8078 | **0.8156** | 0.8056 |
| C→M | **0.7882** | 0.7875 | 0.7857 | 0.7821 |

| 方向 | 比较 | Δ | 95% CI | H1（样本） | \|d̄\| | **σ_train** | t | **H2（训练）** |
|---|---|---|---|---|---|---|---|---|
| M→C | HyperAdapt − ERM | **+0.0031** | [+0.0019,+0.0043] | 显著 | 0.0031 | **0.0086** | 1.27 | ❌ **未超过噪声** |
| M→C | SWAD − ERM | −0.0047 | [−0.0061,−0.0033] | 显著劣 | 0.0047 | 0.0067 | −2.11 | ❌ 未超过噪声 |
| C→M | HyperAdapt − ERM | **−0.0025** | [−0.0032,−0.0019] | **显著劣** | 0.0025 | **0.0064** | −0.95 | ❌ 未超过噪声 |
| C→M | SWAD − ERM | −0.0007 | [−0.0013,−0.0001] | 显著劣 | 0.0007 | 0.0072 | −0.31 | ❌ 未超过噪声 |

Holm 校正（family = 2 方法 × 2 方向 = 4）：**四项全部保持显著**——但见 §4 的关键说明。

---

## 3. 🛑 C→M 方向变号，且**由 trial 扩充所致**

| C→M 口径 | Δ(HyperAdapt − ERM) | 判决 |
|---|---|---|
| 旧：k→k averaging，**trial 0 单 trial** | **+0.0017** | 显著**优** |
| full-target，**trial 0 单 trial** | +0.0023 | 正 |
| **full-target，15 replicate（本实验）** | **−0.0025** | **显著劣** |

第 1→2 行说明**换估计量不改变符号**（+0.0017 → +0.0023）；第 2→3 行说明**加入 trial 1/2 后翻转**
（+0.0023 → −0.0025）。⇒ **变号的原因是训练随机性，不是口径**。

这正是 v2 引入 trial 的目的所在：现行设计 `seed = 42 + fold` 把训练随机性与数据划分绑死，**单 trial
的 OOD 结论不可靠**——本例中它给出了**方向相反**的答案。

---

## 4. ⚠️ 方法学要点：bootstrap「显著」与训练噪声无关

四项比较的 bootstrap CI 全部显著、Holm 后全部保持——但**同样这四项，效应量全都小于训练噪声**
（t 比值 0.31–2.11，均未达常规阈值）。两者不矛盾，因为它们量的是**不同的不确定性**：

- **患者级 cluster bootstrap** 只回答「换一批病人」，n≈14–20 万 ⇒ CI 极窄（±0.0006~0.0012）；
- **fold × trial 方差分解** 回答「换一次训练随机种子」，σ_train ≈ 0.006–0.009，**比效应量大 2–3 倍**。

⇒ **在本设定下，只报 bootstrap CI 会系统性高估 OOD 结论的可靠性。** 任何声称 Δ<0.005 量级 OOD
差异的结论，都必须同时报告训练侧不确定性，否则不可采信。

---

## 5. 支撑性诊断（阶段 0，零 GPU）

### 5.1 偏移刻画：几乎全是 prior shift，**无 concept shift**

| 偏移类型 | 量级 |
|---|---|
| **Prior shift** | P(Y) **30.2% → 8.2%**（3.7 倍） |
| 属性构成 ΔP(A) | ≤0.059（age），其余 <0.05 |
| **Concept shift** | **无迹象**——6 个 marginal 子群相对总体的患病率方向在两库**完全一致** |

**对项目核心论点的意义**（🛑 本段初稿的因果归因有误，已订正）：

HN 在 OOD 上可能有优势的机制有三条，必须分开讨论：

| 机制 | 需要的条件 | 本项目是否满足 |
|---|---|---|
| **A. 把 ID 的属性条件化优势迁移过去**（主路径） | ① `I(Y;A|X) > 0`；② 无 concept shift | ① **不满足**；② 满足 |
| B. 条件化的正则/容量效应（与属性无关） | — | **✅ 已由 §5.1b knockout 确认为主导**（占 M→C 增益约 90%）；另已知 legacy +0.0045 中 +0.0043 来自 wd |
| C. 面对 concept shift 的适应性 | 存在 concept shift + 目标端适应手段 | 不适用（无 concept shift，且 zero-shot 无适应手段） |

> **初稿错误**：曾写「无 concept shift ⇒ HN 唯一机制没有用武之地」，这把机制 C 当成了唯一路径。
> 实际上主路径是 **A**，而**无 concept shift 恰恰是 A 的有利条件**——属性—标签关系跨库稳定，
> 意味着 source 上学到的属性条件化**能够**迁移到 target。

**订正后的推理**：机制 A 断在条件 ①，不在 ②。MIMIC 与 CheXpert 的 `I(Y;A|X)` 均为**决定性的 0**
（两库全轴 CI 上界 < 检出地板）⇒ **source 域上 HN 相对 attribute-blind ERM 就没有优势**。

> **无 concept shift 不是 HyperAdapt 失败的原因，而是让「没有优势」忠实地传导到了 OOD。**
> 属性—标签关系确实稳定、确实可迁移，只是被迁移的那个量的值是零。

它在论证中的作用是**排除一条替代性辩护**：不能说「HyperAdapt 的 OOD 表现差是因为目标域属性—标签
关系变了、它没跟上」——关系没变，此路不通。

**由此得到的可检验预测**：若 `I(Y;A|X) > 0` **且**无 concept shift，则 HN 的优势应在 OOD 上**完整
保留**。项目已有的 R1 合成信号注入（`A_syn = Y ⊕ Bern(η)`，已证因果剂量-反应）正好制造出条件 ①。
**把合成信号版本拿去做跨库 OOD**，即可直接检验机制 A——增益按预期保留则 A 成立，衰减则另有因素。
这比在零信号上测迁移的信息量大得多。

**另一处补充**：无 concept shift 也解释了校准结果——各子群相对总体的患病率方向跨库一致，偏移基本是
**全局平移**，故**单一的全局单调重校准即可修复**（实测 ECE 降 40–50 倍）。若存在 concept shift，
全局重校准将不足，须分子群校准。

### 5.1b 属性 knockout 反事实：机制 A 的直接检验（M→C）

§5.1 把可能的机制拆成 A/B 后，**只有反事实干预能把两者分开**：拿**同一份** HyperAdapt 权重、在
**同一批**样本上，只改「模型看到什么属性」再推断。架构/容量/超参因此全部被消掉，唯一变化是属性信息。

**两个 knockout 条件**（各 5 fold × 3 trial = 15 replicate，full-target）：

| 条件 | 做法 | 反事实含义 |
|---|---|---|
| `permute` | 全局置换属性（保边缘分布与属性间联合结构） | 给**错误**属性 ⇒ 模型施加**错误方向**的调制 |
| **`marginal`** | **输出边缘化** `p(y\|x)=Σ_a p(a)·p(y\|x,a)`，8 组合按 target 经验先验加权平均概率 | **移除**属性信息的**正确定义** |

> **实现要点**：干预包在 `forward_fn` 而非 `unpack_fn` 上——`harness.evaluate` 用 unpack 的 attrs
> **同时**喂模型与收集分组键，若在 unpack 处替换则子群分组也会用假属性。包在 forward 上则
> **分组键恒为真实属性**（脚本内建断言核验）。
>
> 另跑过 `constant`（全置 majority cell），但它只是 `marginal` 把先验塌缩成点质量的**退化特例**，
> 且该点的 age 恰好等于主指标 worst 组的取值（碰巧对齐）；`marginal` 就位后不再提供独立信息，
> 已从分析中移除（产物保留于 `cv5_knockout_constant/` 供追溯）。

**结果（marginal worst AUC，Δ = HyperAdapt − ERM；ERM = 0.8125）**：

| 属性条件 | HyperAdapt | Δ vs ERM | 95% CI | σ_train | t |
|---|---|---|---|---|---|
| 真实 | 0.8156 | +0.0031 | [+0.0019,+0.0043] | 0.0086 | 1.27 |
| 置换 | 0.8149 | +0.0024 | [+0.0013,+0.0035] | 0.0089 | 0.96 |
| **边缘化** | 0.8154 | +0.0029 | [+0.0018,+0.0040] | 0.0086 | 1.16 |

**属性信息的净贡献**（配对差）：

| 配对差 | Δ | 95% CI | σ_train | t | 判决 |
|---|---|---|---|---|---|
| 真实 − 置换 | +0.0008 | [+0.0003,+0.0012] | 0.0004 | +6.55 | 显著（**高估**：含错配惩罚） |
| **真实 − 边缘化** | **+0.0003** | **[−0.0002,+0.0006]** | 0.0004 | +2.66 | **n.s.，且未超训练噪声** |

⇒ **判决：机制 A 证据弱。** 严格口径（边缘化）下属性信息的净贡献 **+0.0003、CI 含 0、未超训练噪声**。
`permute` 的 +0.0008 里约六成是「给错属性造成的额外伤害」，不应计入属性的贡献——两种 knockout 的
分歧由边缘化裁决，真值紧贴「无贡献」一端。

> **量化结论**：M→C 上 HyperAdapt 的 +0.0031 增益中，属性信息贡献约 **+0.0003（不显著）**，
> **其余约 90% 来自与属性无关的架构效应（机制 B）**；而 +0.0031 本身也小于 σ_train = 0.0086。
> 这与 `I(Y;A|X)` 为决定性 0 的闸门判决一致。

**两个副产品**：

1. **配对反事实的检出功效高于闸门**。注意 σ_train 差 20 倍：`real−knockout` 是 0.0004，
   `knockout−ERM` 是 0.0086~0.0089——同权重同样本的反事实把共同变异消得极干净。故
   conditional V-information 判的「决定性 0」应理解为**低于该方法的检出地板**，而非严格为零。
2. **条件化真实发生、但无用**。产物验证：置换属性后 **100%** 样本 logit 改变，中位 |Δlogit| = 0.0696。
   超网络确实在按属性调制参数，只是该调制对排序几乎无贡献——与权重轨迹分析的
   「条件化了但属性无用」互为独立佐证。

---

### 5.2 校准（TRIPOD+AI 要求，现行 OOD 结论完全未报）

原样跨库迁移的校准严重失准，且被 AUC 完全掩盖：

| 方向 | CITL | ECE | 重校准后 ECE |
|---|---|---|---|
| M→C | +0.10 ~ +0.13（系统性高估） | 0.097–0.125 | 0.0023–0.0025 |
| C→M | −0.15 ~ −0.17（系统性低估） | 0.151–0.166 | 0.0026–0.0033 |

HyperAdapt vs ERM（配对 bootstrap）：**M→C 校准显著更差**（Δ\|CITL\| +0.0068、ΔECE +0.0068、
ΔBrier +0.0040），C→M 显著更好。**SWAD 在两方向校准均显著优于 ERM**——这是它此前未被看到的优点
（现行只报 AUC，而 AUC 上 SWAD 两方向皆劣）。

### 5.3 argmin 归属

M→C 的 **canonical worst 的 argmin 不稳**（两方法 worst 落在不同 joint 组，bootstrap 主导份额仅
46%/38%）⇒ 该列不可作点比较。marginal worst 的 argmin 极稳（`age:>=60`，5/5 折、bootstrap 100%）。

---

## 6. 对现有文档的修订要求

`docs/oof_regime_results.md` 需按本文件修订：

1. **§1.7（C→M）**「HyperAdapt +0.0017 显著优」→ **撤回**，改为 15-replicate full-target 下
   **−0.0025 显著劣**。
2. **§1.6（M→C）**「HyperAdapt +0.0045 显著优」→ 保留数值但须注明：(a) 超参对照显示其中 +0.0043
   来自 wd 差异；(b) full-target 15-replicate 下降至 +0.0031 且**小于训练噪声**；(c) 校准显著更差。
3. **§3.2**「HyperAdapt 是唯一一致显著为正的 HN」→ **撤回**。两个 OOD 方向已不一致（一正一负）。
4. **§1.6 canonical worst 列** → 加 argmin 不稳警告。
5. 全表须补充说明：bootstrap CI 未含训练侧不确定性（§4）。

---

## 7. 边界与未决

1. **不含 HyperHead / HyperFusion**（用户指定范围）⇒ **无法在 OOD 上检验条件化深度轴**，本文件的
   任何结论都不得外推为「深度」结论。
2. **H4 未评估**：需阶段 2（全 config 网格）提供 ID 质量跨度后重跑。
3. **ID 侧协议未改动**，ID 结论不受本实验影响。
4. **选择损失（D1）未做**：阶段 2 未执行。已知的存在性证据是 ERM@`lr1e-04_wd1e-03` 的 M→C
   marginal worst 0.8157 > ID 选中配置的 0.8114，即 ID 选出的 config 在 OOD 上并非最优。

---

## 8. 产物

| 内容 | 路径 |
|---|---|
| H1/H2 主分析 | `outputs/ood_cxr/variance_decomposition/` |
| **属性 knockout（§5.1b）** | `outputs/ood_cxr/knockout/` |
| 校准（D2） | `outputs/ood_cxr/calibration/` |
| 偏移刻画（D3） | `outputs/ood_cxr/shift_characterization/` |
| effective robustness | `outputs/ood_cxr/effective_robustness/` |
| argmin 归属 | `outputs/ood_cxr/argmin_attribution/` |
| 超参匹配对照 | `outputs/ood_cxr/hparam_matched/` |
| full-target 预测 | `outputs/ood_cxr/{m2c,c2m}/cv5_full_target/predictions/` |
| **超平面几何可视化**（机制侧，非本文件结论来源） | `outputs/ood_cxr/hyperplane_viz/{m2c,c2m}/`，报告 `docs/ood_hyperplane_visualisation.md` |

脚本：`scripts/ood_variance_decomposition.py`、`ood_calibration.py`、
`ood_shift_characterization.py`、`ood_effective_robustness.py`、`ood_argmin_attribution.py`、
`ood_hparam_matched_control.py`、**`ood_knockout_analysis.py`**；
训练 `slurm/ood_trial.sh`，推理 `slurm/oodv2_full_target.sh`、**`slurm/ood_knockout.sh`**
（+ `src/training/run_ood_attr_knockout.py`）。

**回归测试**：M→C trial 0 的 full-target Δ = **+0.0031**，与 `oof_regime_results.md` §1.6 注中既有
full-target 交叉验证值 **+0.0031 逐位吻合** ⇒ `(fold, trial)` 泛化未改变既有口径。
