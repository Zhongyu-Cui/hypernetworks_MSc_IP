# Conditional V-information 信号闸门（基于已确立方法的重述）

> 建立日期：2026-07-19。本文档取代旧「四层 E0–E3 条件互信息诊断」作为**信号闸门的权威方法说明**。
> 旧脚本 `scripts/estimate_conditional_mi*.py` 保留供对账，但主口径改为本文所述的 conditional V-information。
> 实现：`scripts/estimate_conditional_v_information.py`（统一 `--dataset`）；作业脚本
> `slurm/estimate_conditional_v_information.sh`。

---

## 1. 闸门要回答的问题与判据

比较基准是 **attribute-blind 池化 baseline**（忽略属性 A、对所有人用同一预测器）。HN 的全部价值在于
让预测器依赖 A。故有一个**训练任何 HN 之前**就能回答的是非题：

> 给定图像 X 之后，属性 A 还携带图像没给的、与标签 Y 相关、且**一个 HN 容量的模型用得上**的信息吗？

- **若否** → 池化已用尽可用信息，任何属性条件架构（全部 HN 变体）顶多退化成它、不可能赢 → HN 注定无收益，根因在数据侧。
- **若是** → 信号在、HN 没接住 → 机制/架构问题。

这是 HN 有用的**必要条件**，且测它**不需要训练任何 HN**（模型无关）。

---

## 2. 为什么不用互信息 I(Y;A|X)，而用 conditional V-information

条件互信息在原理上对，但用作闸门有两个致命问题（Hewitt et al. 2021 对前者、McAllester & Stratos 2020 对后者）：

1. **MI 不区分「信息在不在」与「信息用不用得上」**：加密文档与其明文和标签的互信息相等（数据处理不等式 / 双射不变性），但 ResNet 不会解密。闸门要问的恰是「HN 容量的模型能否用上 A」。
2. **MI 估计被诅咒**：任何分布无关的高置信 MI 下界，从 N 样本最多到 O(ln N)（旧 E0 的 plug-in 边际 MI 正踩此坑，已弃用）。

**V-information**（Xu et al. 2020）把「可用性」写进定义：用一个函数族 V 重定义信息。

- **V-entropy**：`H_V(Y|R) = inf_{f∈V} E[−log f[R](Y)]`（用 V 中最好的函数读 R 预测 Y 能达到的最小交叉熵）。
- **V-information**：`I_V(R→Y) = H_V(Y) − H_V(Y|R)`。V=全函数时退回香农 MI；V=逻辑回归时是「逻辑回归能用上的信息」。

**Conditional V-information / conditional probing**（Hewitt et al. 2021, EMNLP, Eq.2）给定 baseline B：
```
I_V(φ→Y|B) = H_V(Y|B) − H_V(Y|B,φ)     估计式：Perf([B;φ]) − Perf([B;0])
```
其中 Perf = 留出集负交叉熵，probe 在 train 上拟合。

---

## 3. 映射到本项目

把 Eq.2 的角色对调：**baseline B = 冻结图像特征 φ(X)，被研究表征 = 属性 A**：

```
闸门量：  I_V(A→Y|φ(X)) = H_V(Y|φ(X)) − H_V(Y|φ(X),A)
                        = [留出集 NLL of  Y~φ]  −  [留出集 NLL of  Y~[φ,A]]
```
直觉：先把图像白送给 probe，A 只能因「图像解释不了的那部分 Y」拿分。**≥0 在极限成立；留出估计只会低估**
（含 A 的 probe 总可忽略 A 退化成 baseline），故 ≈0 是保守读数，CI 下界 >0 才判「有可用信号」。一律以 **bits** 汇报。

### probe 族 V（Hewitt 要求「因 theory-external 理由选」→ 选来匹配 HN 容量）

| 族 | 设计矩阵 | 含义 | 角色 |
|----|---------|------|------|
| **V_int**（主口径） | 逐组独立 `[1, φ]` 块设计 | 逐组不同决策函数 | HN 生成逐组参数的对应 = **紧上界** |
| V_add（次口径） | `[φ, onehot(A)]` 线性 | 仅逐组截距 = 患病率/校准位移 | 分解：多少只是先验位移 |

两者都是合法 conditional V-information，仅声明的 V 不同。

---

## 4. 估计流程与无泄漏（OOF 口径）

对齐 `docs/oof_regime_results.md` / `scripts/cv_oof_report.py`：

- backbone = cv5 的 5 折 **image-only erm baseline**（选定 config 见各 `selected_configs.json`，seed=42+fold）。
- 对每折 f：用 M_f 抽 train_f/val_f/test_f 冻结特征（avgpool 512 维）→ **train_f 拟合 probe**（凸 LBFGS，逼近 Eq.3 的 inf）、**val_f 选 L2 + 温度标定**、**test_f 逐样本 NLL**。
- 5 折 test 逐样本 NLL 池化（每样本恰被其留出折预测一次 = OOF）→ 全数据集 conditional V-info + 样本级 bootstrap CI。
- 每样本的留出读数都来自一个**从未见过它**的折的 probe（选择/标定只用 val、评估只用 OOF test，三重无泄漏）。

### Control task（Hewitt & Liang 2019，兜住唯一假阴风险）

probe 族若整体太弱，会即便有信号也读 0。构造 `Y'=Y⊕r`（r~Bernoulli(0.5) 与图像无关），以 A_syn=r
在 V_int 下算 conditional V-info。r 完全决定翻转且独立于图像 ⟹ **必须读出大正值**；否则 probe 族无功效、本次真实读数作废。

---

## 5. 诚实的边界

1. **条件在 φ(X) 而非 X**：测的是 `I_V(A→Y|φ(X))`。对共享该 backbone 的 HN（尤其 HyperHead）是紧的；对 modulate backbone 的 HyperAdapt/HyperFusion，φ 可能逐组变，是同口径近似而非严格上界。因所有被比 HN 共用同一 backbone 的 φ，近似对基线/HN 一视同仁，作为比较闸门仍公平。
2. **留出 V-info 可略负**：按 Xu/Hewitt 惯例解释为 ≈0。
3. **读数依赖 V**：已显式声明 V=逐组线性；若要弱化「选 V」主观性，可加 Voita & Titov 2020 的 MDL/在线编码版作稳健性附录。

---

## 6. 结果（OOF，V_int 主口径，bits）

> 每格：`V_int (bits)  CI95  | detected(CI>0) | minority-weighted | V_add`。detected=CI 下界>0 ⟹ 有可用信号。
> Control task 应对所有数据集读出大正值（探针功效）。

| 数据集 | 属性 | V_int (bits) | CI95 | detected | minority-w | V_add | control task |
|--------|------|-------------|------|:--------:|-----------|-------|--------------|
| **PAPILA** (n=420) | sex | +0.0207 | [−0.010, +0.057] | ✗ | +0.033 | +0.003 | +0.476 [+0.35,+0.59] ✓ |
| | age | **+0.0413** | [+0.010, +0.074] | **✓** | +0.036 | +0.002 | |
| | joint | +0.0226 | [−0.028, +0.078] | ✗ | +0.019 | +0.004 | |
| **HAM10000** (n≈9.7k) | sex | −0.0016 | [−0.0032, −0.0001] | ✗ | −0.0016 | +0.00001 | +0.619 [+0.60,+0.63] ✓ |
| | age | +0.0028 | [−0.0003, +0.0058] | ✗ | +0.0054 | +0.0001 | |
| | joint | +0.0008 | [−0.0031, +0.0047] | ✗ | +0.0023 | +0.0001 | |
| **Fitzpatrick17k** (n=16k) | skin | −0.0028 | [−0.0052, −0.0005] | ✗ | −0.0044 | +0.00002 | +0.657 [+0.64,+0.67] ✓ |
| | skin_bin | −0.0010 | [−0.0021, +0.0002] | ✗ | −0.0011 | +0.00002 | |
| **CheXpert** (n=138k) | sex | −0.0008 | [−0.0010, −0.0006] | ✗ | −0.0009 | +0.00001 | +0.705 [+0.701,+0.709] ✓ |
| | race | −0.0012 | [−0.0014, −0.0009] | ✗ | −0.0022 | +0.00000 | |
| | age | −0.0008 | [−0.0011, −0.0006] | ✗ | −0.0009 | +0.00000 | |
| | joint | −0.0092 | [−0.0099, −0.0084] | ✗ | −0.0138 | +0.00001 | |
| **MIMIC-CXR** (n=199k) | sex | −0.0006 | [−0.0007, −0.0004] | ✗ | −0.0006 | +0.00002 | +0.374 [+0.370,+0.378] ✓ |
| | race | −0.0008 | [−0.0010, −0.0006] | ✗ | −0.0015 | +0.00001 | |
| | age | −0.0010 | [−0.0012, −0.0008] | ✗ | −0.0012 | +0.00002 | |
| | joint | −0.0038 | [−0.0042, −0.0033] | ✗ | −0.0057 | +0.00003 | |

**已出小结（PAPILA/HAM/Fitzpatrick）**：
- **control task 全部大正值且 detected**（+0.48~+0.66 bits）→ 三数据集的 probe 族都对强条件信号有功效，≈0 读数可信（非假阴）。
- **PAPILA age 是目前唯一 detected 的真实轴**（+0.041 bits, CI>0）；sex/joint 不显著。
- **HAM/Fitzpatrick 所有真实属性 conditional V-info 均 ≈0**（|值|<0.003 bits，CI 贴 0 或微负）。
  - ⚠️ **注意与旧单-split conditional-MI 的差异**：旧口径曾把 **HAM-age 判为「唯一正信号」**（记忆
    `ham10000-conditional-mi-age-signal`）；在**本 OOF 口径**下 HAM-age 仅 +0.0028 bits，**CI[−0.0003,+0.0058] 恰好含 0、未 detected**。
    这与路线 A / averaging 系列「OOF/powered 口径洗掉 HAM-age worst-group 增益」的既有发现方向一致——OOF 闸门给出比单-split 更保守的读数。
  - V_add 一律≈0（无逐组截距/患病率位移信号），说明即便有微弱效应也非校准型。

### 全 5 数据集总判决（OOF conditional V-information）

| 数据集 | detected 的真实属性轴 | 判决 |
|--------|----------------------|------|
| **PAPILA** | **age**（+0.041 bits, CI>0） | age 有小而可检出的可用信号（⚠️ n=420 小样本，最弱稳健性） |
| HAM10000 | 无（age +0.003, CI 触 0） | ≈0 |
| Fitzpatrick17k | 无（全 ≤0） | ≈0 |
| **CheXpert** | 无（全 **CI<0**） | **0，决定性** |
| **MIMIC-CXR** | 无（全 **CI<0**） | **0，决定性** |

- **control task 五个数据集全部大正值 detected**（+0.37~+0.71 bits）→ 所有 probe 族都有功效，∴ 真实属性的 ≈0/负读数一律可信为「无可用信号」，非假阴。
- **CheXpert/MIMIC 的 V_int 在 n=13万~20万下 CI 完全落在 0 以下**：负的留出 conditional V-info 表示逐组条件 probe 比池化 probe **泛化更差**（额外容量在无真实信号时只带来微弱过拟合代价）——按 Xu/Hewitt 惯例即判「无可用信息」，是比「微小正」更干净的「信号=0」证据。joint（8 组、设计维最高）最负，正是无信号时容量惩罚的签名。
- **唯一 detected 的真实轴是 PAPILA-age**；旧单-split 曾判为正的 **HAM-age 在 OOF 口径下退回未 detected**（原因见 §8）。

### §8 单-split 与 OOF 差异的来源（HAM-age 为例）

旧单-split conditional-MI 报 HAM-age = **+0.0199 bits（detected）**，本 OOF 口径 = **+0.0028 bits（未 detected）**。差异源于「单-split→OOF」这一步**同时改了三样**：

1. **φ 不同（最根本）**：conditional V-info 定义在冻结特征 φ 上（`I_V(A→Y|φ)`）。旧用单-split pilot 配置（lr1e-4/wd1e-4）的一个 φ；新用 OOF 选定配置（lr3e-4/wd1e-3）的 5 折 φ。不同 config/训练子集 → 不同 φ → 字面上是两个不同的量。
2. **留出规模 969 → 9707**：旧在一个 10% 切片评（3 seed 共用同一 test，有效独立点≈969）；OOF 全量池化。CI 收紧 ~3×，点估计从「幸运切片」向真值（≈0）回归。
3. **聚合**：旧同一 split 上 3-seed 平均（只压 backbone 初始化噪声）；OOF 逐样本 disjoint 折池化（既扩样本又无泄漏）。

结论方向与路线 A/averaging 系列的「单-split 伪抬升 → OOF 洗掉」母题一致；OOF 是更可信读数。若需干净隔离「纯池化」项，可固定同一 config 只换评估口径重跑（预期仍 ≈0）。

---

## 7. 引用

- Xu, Zhao, Song, Stewart, Ermon (2020). *A Theory of Usable Information Under Computational Constraints.* ICLR. arXiv:2002.10689.
- Hewitt, Ethayarajh, Liang, Manning (2021). *Conditional probing: measuring usable information beyond a baseline.* EMNLP. aclanthology 2021.emnlp-main.122.
- Hewitt & Liang (2019). *Designing and Interpreting Probes with Control Tasks.* EMNLP. aclanthology D19-1275.
- McAllester & Stratos (2020). *Formal Limitations on the Measurement of Mutual Information.* AISTATS.（说明为何弃用 plug-in MI）
- Voita & Titov (2020). *Information-Theoretic Probing with Minimum Description Length.* EMNLP.（可选稳健性变体）
