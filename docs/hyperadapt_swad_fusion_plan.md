# 实验 G · HyperAdapt × SWAD 融合方案（精简版）

> 建立：2026-08-04。状态：**训练与主分析已完成**（G0/G1/G2/G3.0–G3.6 完成，余 G3.7/G3.8）。
> 结果报告：[hyperadapt_swad_fusion.md](hyperadapt_swad_fusion.md)。
> 定位：**独立追加实验**，与实验 E（条件化范围消融）、F（冻结 regime）平级。
> **不动**已冻结的 6 方法主矩阵与 D1–D4 结论表——协议 §1 明文写「SWAD 不与 HN 组合」，
> 本实验是对该条的**受控破例**，结论单独成文，不回写主矩阵。
>
> 权威依赖：评估口径见 [routeA_ham_cv_oof_fairness.md](routeA_ham_cv_oof_fairness.md) 与
> [crossdataset_worstgroup_fairness.md](crossdataset_worstgroup_fairness.md)（averaging 口径）；
> 信号闸门见 [conditional_v_information_gate.md](conditional_v_information_gate.md)。

---

## 1. 动机与**诚实的预期**

### 1.1 出发点

现有 averaging 口径读数（Δ marginal worst-group AUC vs ERM）：

| 数据集 | SWAD | HyperAdapt | 备注 |
|---|---|---|---|
| HAM10000 | **+0.046（显著）** | +0.021（n.s.） | 唯一显著胜 ERM 的是 SWAD |
| Fitzpatrick17k | **+0.058（显著）** | n.s.，worst 的 seed 间 std ±0.069 | 同上 |

即：**SWAD 是目前唯一稳健且有实质幅度的 worst-group 赢家，而 HN 全线打平。**
自然的问题是两者能否叠加。

### 1.2 SWAD+CORAL 先例给出的**定量零假设**

SWAD 原论文（Cha et al., NeurIPS 2021）在 DomainBed 上把 SWAD 套到 CORAL 上取得其最佳配置，
定性模式是：**SWAD 给 CORAL 的增量 ≈ 它给 ERM 的增量，两因子近似可加**，而 **CORAL 单独相对
ERM 已经是正的**。

> ⚠️ 该论文的具体数值须回查原文 Table 后再引用，本文档暂不写死数字。

把这条模式套到我们这里，得到的**不是**"会更好"，而是：

$$\text{HA+SWAD} \;\approx\; \text{SWAD} + (\text{HA} - \text{ERM}) \;\approx\; \text{SWAD}$$

因为 HA − ERM ≈ 0。**可加性作用在 0 上还是 0。** 这条先例的价值在于把"大概率 null"从直觉
升级为**可预注册的定量零假设**（见 §5 的 Δ_int = 0 ± MDE）。

### 1.3 机制上像不像？——不像，三处结构性不对称

按 SWAD 论文自己的泛化界分解（目标域风险 ≤ 源域经验风险 + 平坦性项 + 域散度项 + 复杂度项）：

| 方法 | 打界里哪一项 | 层次 | 推理时是否需要组信息 |
|---|---|---|---|
| SWAD | 平坦性项 | 优化 / 权重聚合 | 否 |
| CORAL | 域散度项 | **损失层**（加正则、不加参数） | 否 |
| HyperAdapt | 近似误差项，**当且仅当 I(Y;A\|X)>0** | **架构层**（+1.3–2.8M 参数，逐样本生成权重） | **是** |

- **(a) 打的项不同**：CORAL 与 SWAD 攻击的是同一个界里两个**不同的加性项**，这是它们可加的机制来源。
  HyperAdapt 攻击第三项，而闸门判定该项在 HAM-age / Fitz-skin 上 **≈0**；更糟的是它对平坦性项的
  作用**反向**（更多参数、逐样本权重 ⇒ 极小值更锐、方差更大，D3 逐折符号翻转即证据）。
  ⇒ 融合不是"两个正效应相加"，而是"**SWAD 给 HN 收拾它自己制造的方差**"。
- **(b) DomainBed 里没有 HN 的对应物**：CORAL/IRM/MMD/Mixup 全属"训练时用组信息、推理时盲"；
  HyperAdapt 推理时必须吃属性。**SWAD 论文的证据覆盖不到我们这一格。**
- **(c) 被平均的参数 ≠ 被使用的参数**：见 §4.1，这是 HN 特有的问题，CORAL 不存在。

### 1.4 三条待检验假设

| 编号 | 假设 | 先验 |
|---|---|---|
| **H1**（均值/加性） | HA+SWAD 的 worst-group 优于 SWAD 单独 | 低（见 §1.2） |
| **H2**（方差/稳定化，**主打**） | SWAD 对 HN 的**折间方差**削减幅度大于对 ERM | 中，机制上最站得住 |
| **H3**（正交） | 两者皆 null ⇒ **SWAD 收益与架构正交、与属性条件化无交互** | 中高 |

**任一结果都有明确读数**（§7 判读表），不押注单一方向。H3 是与"属性无可用信息"闸门互相印证的
阴性证据，可直接并入既有证据链，不是浪费。

---

## 2. 设计：2×2 因子设计

|  | 普通 best-ckpt 选择 | + SWAD loss-valley 权重平均 |
|---|---|---|
| **ERM**（属性盲） | ✅ 已有 | ✅ 已有 |
| **HyperAdapt**（属性条件） | ✅ 已有 | 🆕 **本实验唯一要补的格** |

核心统计量是**交互对比**：

```
Δ_int = (HA+SWAD − SWAD) − (HA − ERM)
```

比单臂比较信息量高，且堵死事后叙事。

### 2.1 免费的额外对照（零训练）

ERM 搜索阶段是带 `SWAD=1` 跑满 6 config × 5 fold 的，故 `outputs/{ham10000,fitzpatrick}/cv5/predictions/`
下 **6 个配置的 SWAD 预测全部已存在**。于是可白拿一个控制：

- **配置匹配的 SWAD 对照**：取与 HyperAdapt **同 (lr, wd)** 的 `swad_<HA的tag>`，
  隔离"超参不同"这一混杂，使 `HA+SWAD − SWAD` 的差不掺配置差异。
- HAM：HA 用 `lr3e-05_wd1e-04` ⇒ 对照取 `swad_lr3e-05_wd1e-04`（主口径 SWAD 是 `lr3e-04_wd1e-03`）。
- Fitz：HA 与 SWAD 恰**同为** `lr3e-04_wd1e-03` ⇒ 天然匹配，无需额外对照。

**纯分析项，不增加任何 run。**

---

## 3. 范围与算力

| 项 | 决定 |
|---|---|
| 数据集 | **HAM10000（主）+ Fitzpatrick17k（复现）**。这两库正是 SWAD 唯一显著获胜处、n 足够（9,707 / 16,012） |
| 不做 | MIMIC / CheXpert（闸门决定性 0，只能确认 null，n 大但无新信息）；**PAPILA（n=420 无功效）** |
| 不做 | GroupDRO 正对照臂、iteration 级密采样 SWAD —— 均已论证有价值，但本轮为控制工作量**明确移出范围**，记入 §8 后续 |
| run 数 | **5 折 × 2 库 = 10 run** |
| 单 run 增量成本 | SWAD 逐 epoch 缓存 CPU 权重：13.9M 参数 × 4B × ≤30 epoch ≈ **1.7 GB 内存**，gpus48（125 GB）无压力 |

---

## 4. 两个必须知道的技术要点

### 4.1 在哪个参数空间做平均（W vs Θ）

HyperAdapt 的有效权重（[resnet18_hyperadapt.py:346](../src/models/resnet18_hyperadapt.py#L346)）：

$$\theta_\ell(a) = \Theta_\ell \odot \big(1 + A\cdot B_\ell(e(a))\big)$$

调制量 M = A·B 对参数**双线性**，故 `mean(A)·mean(B) ≠ mean(A·B)`——
**朴素权重空间平均，平均的不是模型实际使用的那组权重**。SWAD 的平坦极小理论说的是"损失所在的
那个参数空间"，对 HN 而言是 θ(a) 而非 (A, B)。CORAL 不存在这个问题（单一函数，"平均的参数 =
使用的参数"），这正是 §1.3(c) 的不对称。

- **变体 W（权重空间，literal SWAD）＝ 本轮主口径**：逐张量平均全部参数（含超网），零改动，
  直接走现成 `swad_average_over_interval`。与 ERM-SWAD 操作**字面同构**，唯一严格可比的口径。
- **变体 Θ（有效参数空间）＝ 本轮后置**：属性低基数离散（HAM age 4 格 / Fitz skin 6 格），
  可对每格物化 M_ℓ(a)、Δfc(a)，谷内逐 epoch 平均后冻成查表 adapter（≈20–30 MB）。
  **触发条件（预先写死）**：仅当 **W 口径打平 SWAD（H1 为 null）** 时才做——因为那时 Θ 才有
  判别力（Θ > W 即证明"双线性错配是 HN 无法从权重平均获益的原因"，是可单独成节的方法学发现）。
  若 W 已经赢，Θ 不影响结论，不做。

### 4.2 BN 统计（本轮后置）

SWAD 原论文跑在 DomainBed 的 **冻结 BN** ResNet 上，直接平均权重无问题。本项目**不冻 BN**，
`average_state_dicts` 把 `running_mean/var` 一并平均（[swad.py:177](../src/training/harness/swad.py#L177)）是近似。
对 HyperAdapt 更甚：conv 输出**逐样本**调制后才进 BN，BN 统计与 adapter 强耦合。

- **本轮主口径保持现状（不重估 BN）**，以与已冻结的 ERM-SWAD 数字严格可比。
- **后置敏感性分析**：post-hoc `update_bn`（一次训练集前向，**零重训**，可直接吃已有
  `*_averaged.pth`），且必须**对称**施加到 ERM-SWAD 与 HA-SWAD 两臂。
  **触发条件**：H1 或 H2 出现任何显著效应时，用它排除"效应来自 BN 统计失配"这一替代解释。

---

## 5. 规程冻结（跑之前不得再改）

| 项 | 决定 | 理由 |
|---|---|---|
| 超参 | **不新搜参**，HA+SWAD 复用 cv5 已选 HyperAdapt 配置 | 完全对齐"SWAD 搭 ERM 的车、不单独搜参"的既有规程；新搜参引入不可比的选择自由度 |
| ↳ HAM | `lr3e-05_wd1e-04` = **CONFIG_INDEX 0** | 权威 = `outputs/ham10000/cv5/selected_configs.json` |
| ↳ Fitzpatrick | `lr3e-04_wd1e-03` = **CONFIG_INDEX 5** | 权威 = `outputs/fitzpatrick/cv5/selected_configs.json` |
| SWAD 超参 | N_s=3 / N_e=6 / r=1.3，**不调** | 与 ERM-SWAD 逐位一致，否则交互项不干净 |
| batch | **128（等 batch）**，gpus48 | 重型 HN 显存对照必须等 batch（曾被 b32 OOM 伪影坑过） |
| 节点 | `--exclude=semois` | C 阶段既定纪律 |
| 评估口径 | **CV-OOF 5 折 + averaging 聚合**（逐折算指标 → 折间平均） | 现行权威口径；池化 raw logit 对 HN 有系统性负偏 |
| 主要终点 | marginal worst-group AUC | 与 D1–D4 一致 |
| 次要终点 | Overall AUC |  |
| 第三终点 | **折间 SD（H2 主战场）** |  |
| 检验 | 配对 cluster bootstrap B=1000；对比 {vs ERM, vs SWAD, vs HA, Δ_int}；**Holm 校正** | 复用 [harness/significance.py](../src/training/harness/significance.py) |
| **功效** | **跑分析前先算 MDE**，把 H1 零假设写成 `Δ_int = 0 ± MDE` | 避免重蹈"评估-n 地板 → 无功效否定"的老坑 |
| 配对基准 | 2×2 内的 **HA 臂用本次 run 的产物**（与 SWAD 同一权重轨迹，配对最干净）；旧冻结值仅作漂移核对 | 见 §8 覆盖风险 |

---

## 6. 工作清单

> 规范：完成一项即把 `[ ]` 改为 `[x]` 并在行尾补作业号 / 关键产物路径。

### G0 · 代码实现（实验室机器，轻量）

- [x] **G0.1 修 SWAD method 命名冲突**（**阻断项，必须先做**）
      [harness/train_loop.py:310](../src/training/harness/train_loop.py#L310) 的 `_finalize_swad` 把 method
      硬编码成 `"swad"`；HN 开 `--swad` 会产出同样叫 `swad` 的 checkpoint / 预测 / val 日志，
      分析脚本按 `method="swad"` 收集会把 HA-SWAD 误当 ERM-SWAD。
      改为 `swad_method = "swad" if method == "erm" else f"{method}_swad"`，
      并把 `method != "erm"` 的警告改成信息性说明。
- [x] **G0.2 验证 `parse_run_id` 能反解新方法名**
      `hyperadapt_swad_lr3e-05_wd1e-04_seed42` 须解出 `("hyperadapt_swad", "lr3e-05_wd1e-04", 42)`；
      在 [harness/run.py](../src/training/harness/run.py) 的 selftest 里补一条断言。
- [x] **G0.3 HyperAdapt 训练脚本加 `--swad` 开关**（harness 侧已支持，纯透传）
      [train_ham10000_hyperadapt.py](../src/training/train_ham10000_hyperadapt.py) +
      [train_fitzpatrick_hyperadapt.py](../src/training/train_fitzpatrick_hyperadapt.py)；
      顺带给 hyperhead / hyperfusion 一并加（成本为零，便于后续扩展）。
- [x] **G0.4 分析侧注册新方法名 `hyperadapt_swad`**
      [scripts/build_oof_results_averaging.py:54](../scripts/build_oof_results_averaging.py#L54) 的 `METHODS`、
      [scripts/build_worstgroup_crossdataset_tables.py](../scripts/build_worstgroup_crossdataset_tables.py)、
      [scripts/cv_oof_report.py](../scripts/cv_oof_report.py)。
- [x] **G0.5 备份将被覆盖的既有产物**（**提交作业前必做**）
      本次 run 会以相同 (method=hyperadapt, config, seed) 重写已冻结的 cv5 产物。备份：
      `outputs/{ham10000,fitzpatrick}/cv5/predictions/hyperadapt_<tag>_seed4*.npz`、
      对应 `*_best_{overall,worstcase}.pth` 与 val 日志 → `cv5/_pre_G_backup/`。
- [x] **G0.6 冒烟测试**：本地/gpus 单折 1–2 epoch 跑通，确认产出
      `hyperadapt_swad_<tag>_seed42_overall.npz` 且 **未**污染 `swad_*` 命名空间。

### G1 · HAM10000（5 run）

- [x] **G1.1 提交**（`ssh biomedia-slurm` 后 `sbatch`；复用现成脚本，不新建 SLURM 文件）
      ```bash
      sbatch --partition=gpus48 --job-name=g1_ham_ha_swad \
             --output=/vol/biomedic2/bglocker_studproj/zc125/logs/g1_ham_ha_swad.%N.%A_%a.log \
             --export=ALL,PY_SCRIPT=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/src/training/train_ham10000_hyperadapt.py,CONFIG_INDEX=0,BATCH=128,SWAD=1 \
             slurm/c1_ham_cv_oof.sh
      ```
      （`c1_ham_cv_oof.sh` array 0–4 = fold，SEED = 42+fold）
- [x] **G1.2 检查运行状态**：`squeue -u $USER` / `/vol/biomedic3/bin/lazyslurm`；
      GPU 节点偶发错误时换节点（`--nodelist` / `--exclude`）重跑该折。 —— 5 折全部 R→完成于 loki，无节点故障，未换节点
- [x] **G1.3 产物齐全性核验**：5 折的 `hyperadapt_swad_lr3e-05_wd1e-04_seed4{2..6}_overall.npz` 全在，
      OOF 并集 = 9,707（过滤 age_group≥0 后），0 错误日志。 —— 作业 **74958**，5/5 融合臂预测 + 5/5 平均权重，异常签名 0；谷宽 4–9 epoch 无退化
- [x] **G1.4 漂移核对**：新跑出的 `hyperadapt` 臂与 G0.5 备份的旧值比对；
      若 worst-group 漂移 >0.005 需在报告中说明（非确定性来源），2×2 仍用本次 run 的 HA 臂。

### G2 · Fitzpatrick17k（5 run） —— 见 §6.6，Δmarginal-worst=+0.0028（14% 效应量，可忽略）
- [x] **G2.1 提交**（复用 `c4_fitz_cv_search.sh`，用 array 子集只跑选定 config 的 5 折：
      task = config_index + 6×fold ⇒ config 5 对应 `5,11,17,23,29`）
      ```bash
      sbatch --partition=gpus48 --job-name=g2_fitz_ha_swad --array=5,11,17,23,29 \
             --output=/vol/biomedic2/bglocker_studproj/zc125/logs/g2_fitz_ha_swad.%N.%A_%a.log \
             --export=ALL,PY_SCRIPT=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/src/training/train_fitzpatrick_hyperadapt.py,BATCH=128,SWAD=1 \
             slurm/c4_fitz_cv_search.sh
      ```
- [x] **G2.2 检查运行状态**（同 G1.2） —— 5 折并行于 loki，无故障
- [x] **G2.3 产物齐全性核验**：5 折 `hyperadapt_swad_lr3e-04_wd1e-03_seed4{2..6}_overall.npz` 全在，
      OOF 并集 = 16,012。 —— 作业 **74965**，5/5 融合臂预测 + 5/5 平均权重，异常签名 0；谷宽 2–11 epoch
- [x] **G2.4 漂移核对**（同 G1.4）

### G3 · 分析与写作 —— 见 §6.6，Δmarginal-worst=−0.0024（14% 效应量，可忽略）
- [x] **G3.0 让 `hyperadapt_swad` 的 config key 落地**（⚠️ 改为**增量合并**，见 §6.7：直接重生成会因
      重跑覆写 val 日志而使 Fitzpatrick 的 hyperadapt 选择翻转 = 循环论证；冻结版存 `.pre_G`）
- [x] **G3.1 先算 MDE**（用现有 OOF 预测，跑分析前完成），写死 H1 的可检出下限。
- [x] **G3.2 2×2 主表**：两库 × 四臂 × {marginal worst-group AUC, Overall AUC}，averaging 口径。 —— `scripts/g_fusion_analysis.py`，见 §6.8
- [x] **G3.3 交互项检验**：Δ_int 的配对 cluster bootstrap（B=1000）+ Holm 校正。 —— 同上；⚠️ 朴素 Δ_int 在 HAM 上是 config 混杂伪影，须以 G3.5 匹配版为准
- [x] **G3.4 H2 方差表**：四臂各自的**折间 SD**，以及 `SD(HA+SWAD) − SD(HA)` vs `SD(SWAD) − SD(ERM)`。 —— 结果**两库不一致**（Fitz 支持 / HAM 不支持），仅描述性
- [x] **G3.5 配置匹配 SWAD 对照**（§2.1，零训练）：HAM 用 `swad_lr3e-05_wd1e-04` 复核结论不变。 —— **关键**：证明 HAM 的朴素 Δ_int 系 lr 混杂，匹配后归零
- [x] **G3.6 撰写报告** `docs/hyperadapt_swad_fusion.md`（结论先行 + 2×2 表 + 交互项 + 方差表 +
      判读表落位 + 与 CORAL 先例的对照讨论）。 —— [hyperadapt_swad_fusion.md](hyperadapt_swad_fusion.md)
- [ ] **G3.7 回查 SWAD 原论文 Table**，核对 §1.2 的 CORAL 数字后再写入报告（不以记忆入档）。
- [ ] **G3.8 在 [comparison_protocol.md](comparison_protocol.md) 加一行指针**（标注为实验 G、
      对 §1「SWAD 不与 HN 组合」的受控破例，主矩阵不变）。

### G4 · 条件触发项（**已判定不触发、关闭**，理由见结果报告 §7）

- [ ] **G4.1 Θ 空间平均变体** —— 触发：G3 显示 W 口径打平 SWAD（H1 null）。
      新建 `src/training/harness/swad_hyper.py`（枚举属性格 → 逐 epoch 物化 M_ℓ(a)/Δfc(a) →
      谷内平均 → 冻结查表 wrapper）+ selftest（断言恒等超网下退化为 W 平均）。**需重跑 10 run**
      （epoch 权重未持久化）。
- [ ] **G4.2 BN 后校准敏感性** —— 触发：H1 或 H2 出现显著效应。
      新建 `scripts/recalibrate_bn.py`（post-hoc `update_bn`，**零重训**），对 ERM-SWAD 与 HA-SWAD
      **对称**施加。

---

## 6.5 🛑 G3.1 MDE 结果（2026-08-04，**训练落盘前完成，构成预注册**）

脚本 [scripts/g_mde_worstgroup.py](../scripts/g_mde_worstgroup.py)，B=1000 配对 cluster bootstrap，
口径与权威分析同源（复用 `build_oof_results_averaging` 的 Spec/Views/clusters）。
`MDE = (z_{0.975} + z_{1-β})·SE`。口径自验：HAM `SWAD−ERM = +0.0457` 与 D 系列文档的 +0.046 吻合。

**Δ_int 的 SE 怎么估**：由恒等式 `Δ_int ≡ (HA+SWAD − HA) − (SWAD − ERM)`，两项都是
「派生臂 − 自身基座」的配对差（派生臂由基座自身权重轨迹平均而来 ⇒ 高度正相关、SE 远小于
两个独立训练模型之差；实测 `SE(SWAD−ERM)=0.016 < SE(HA−ERM)=0.023` 即印证）。
故取 `s_d = SE(SWAD−ERM)`，`SE(Δ_int) = s_d·√(2(1−ρ))`；ρ=0 保守、ρ=0.5 乐观。
四臂代理（HyperFusion/HyperHead 版）因用两个独立训练的 HN，**系统性高估**，仅作上界参考。

### 主要终点 marginal worst-group AUC

| 数据集 | 参照真效应 SWAD−ERM | MDE(80%) ρ=0.5 | MDE(80%) ρ=0 | 四臂代理 MDE(80%) |
|---|---|---|---|---|
| HAM10000 | **+0.0457** | 0.0454 | 0.0642 | 0.076–0.081 |
| Fitzpatrick17k | **+0.0578** | 0.0630 | 0.0891 | 0.085–0.109 |

### 次要终点 Overall AUC

| 数据集 | 参照真效应 SWAD−ERM | MDE(80%) ρ=0.5 | MDE(80%) ρ=0 |
|---|---|---|---|
| HAM10000 | +0.0191 | 0.0088 | 0.0124 |
| Fitzpatrick17k | +0.0244 | 0.0068 | 0.0096 |

### 🛑 结论：**worst-group 上的 H1 交互检验没有功效，Overall 上有**

- **worst-group**：MDE ≈ 0.045–0.089，而本数据集里**最大的已知真效应**（SWAD 主效应）也才
  0.046–0.058。即：只有当「SWAD 给 HA 的增量」比「SWAD 给 ERM 的增量」还大出**整整一个 SWAD
  主效应**时，我们才有 80% 把握测出来。任何现实量级的交互都测不出。
  ⇒ **预注册裁定：worst-group 上的 `Δ_int ≈ 0` 只能报为「无功效的 null」，不得表述为
  「证明两者无交互」，也不得作为 H3 的证据。** 这正是本项目「评估-n 地板 → 无功效否定」老坑的复现，
  区别是这次**在跑之前就查出来了**。
- **Overall**：MDE ≈ 0.007–0.012，远小于 SWAD 主效应（0.019–0.024）⇒ **H1-on-Overall 是有功效的
  真检验**，可给出实质结论。故报告中 Overall 与 worst-group 的证据强度必须分开陈述。
- **H2（方差）**：5 折下的折间 SD 差异无正式检验功效，**只能描述性呈现**（点估计 + 逐折散点），
  不做显著性宣称。
- **是否仍值得跑**：值得。10 run 成本已支出，Overall 终点有功效，H2 描述性证据对「HN 不稳定」
  这一既有短板有直接价值；但**必须按上面的分层口径写结论**。

> 完整数值（含 canonical worst、各对比 SE/CI）：`outputs/analysis/g_fusion/mde.json`。

---

## 6.6 G1.4 / G2.4 漂移核对结果（2026-08-04）

脚本 [scripts/g_drift_check.py](../scripts/g_drift_check.py)（只读）。比较「实验 G 重跑的
HyperAdapt 臂」与 `_pre_G_backup` 快照，口径同 averaging。

| 数据集 | ΔOverall | Δcanonical worst | **Δmarginal worst** | Δgap | 占参照效应量 |
|---|---|---|---|---|---|
| HAM10000 | +0.0038 | +0.0060 | **+0.0028** | −0.0007 | 14% |
| Fitzpatrick17k | −0.0052 | −0.0024 | **−0.0024** | +0.0063 | 14% |

**终点漂移可忽略**（均 <20% 效应量）⇒ D1–D4 的相应数字不受动摇，2×2 可照常用新 HA 臂。

### 🛑 但逐折 logit 揭示了一件更值得记的事

| 数据集 | 逐折 Pearson r（新 vs 旧） |
|---|---|
| HAM10000 | 0.956 – 0.998 |
| **Fitzpatrick17k** | **0.737 – 0.901** |

**同 config、同 seed 重跑，得到的却是实质不同的模型**（Fitz 尤甚）。根因经核查是
**checkpoint 选择放大了非确定性**：val AUC 曲线在顶部很平且有噪声，argmax 会选到不同 epoch——
HAM fold1 选 11→7、fold4 选 9→16；Fitz fold3 选 5→9；即便选到同一 epoch，val AUC 也可差 0.018
（Fitz fold0：0.9217 vs 0.9034）。聚合终点之所以稳，是 5 折平均把它抹平了。

三条推论：

1. **「同 config+seed」在本项目里不等于可复现**。任何**单次运行**之间 |Δ| < ~0.005 的差异都在
   重跑噪声内，不应解读为方法差异。
2. **强化了 2×2 的配对规则**（§5 最后一行）：HA 臂必须用**本次 run** 的产物——HA+SWAD 由同一条
   权重轨迹派生，配对差扣掉了这整个噪声源；若拿旧快照当 HA 臂，等于把重跑噪声灌进 Δ_int。
3. 这本身是 **H2 的旁证**：HN 的不稳定不止在折间，**同一 split 重跑就已经不稳**——正是 SWAD
   权重平均意在压制的那类噪声。（描述性证据，非检验。）

> 完整数值：`outputs/analysis/g_fusion/drift_check.json`。

---

## 6.7 🛑 G3.0 踩到的坑：重生成 `selected_configs.json` 会**循环论证**

原计划（G3.0）是训练完成后跑 `select_config_cv.py --write-json` 让 `hyperadapt_swad` 的 config key
落地。**实际执行时发现不能这么做**：

config 选择是拿 **6 个配置的 val 日志**比出来的，而实验 G 的重跑**覆写了选中那一档
（`hyperadapt`）的 val 日志**。重生成的结果：

| 数据集 | `hyperadapt` 选中 config | 结果 |
|---|---|---|
| HAM10000 | `lr3e-05_wd1e-04` → 不变 | 纯增量，安全 |
| **Fitzpatrick17k** | `lr3e-04_wd1e-03` → **`lr1e-04_wd1e-03`（翻转）** | **不可采纳** |

翻转的唯一成因就是我们自己的重跑（覆写档的数值略降，与 §6.6 的 ΔOverall −0.0052 一致），
argmax 随之换档。采纳它意味着：**用实验 G 的重跑结果去改动实验 G 的前提**（循环论证），
且会使刚跑完的 5 个 Fitz run 所用的 config 变成"非选中"，实验自相矛盾。

**处置**：不重跑选择，改为**增量合并**——保留冻结的既有 6 项逐位不变，只按命名规则补
`<hn>_swad = <hn>` 的冻结值。冻结版另存为 `selected_configs.json.pre_G`。已断言既有项未被改动。

**两条更一般的教训**（值得写进方法学讨论）：

1. **Fitzpatrick 的 HyperAdapt config 选择对训练非确定性不稳健**：同一套配置、同一批 seed，
   仅因重跑就换了赢家。这说明 6 配置间的 val 差异小于重跑噪声，"Pareto 选中配置"这件事本身
   带有不可忽略的随机性——与 §6.6 的 logit 相关 0.74 互为印证。
2. **任何会被重跑覆写的中间产物，都不能作为下游选择的输入而不加保护**。本项目此后凡"重跑
   已冻结 run"的操作，都应同时冻结其派生的选择结果（本次靠 `.pre_G` 留档 + 增量合并做到）。

---

## 7. 判读表（预先写死，防事后叙事）

> ⚠️ **判读表按 §6.5 的 MDE 结果分层生效**：worst-group 行的"Δ_int ≈ 0"一律读作**无功效 null**，
> 只有 **Overall 终点**上的 null 才是有功效的实质结论。

| 结果 | 结论 | 去向 |
|---|---|---|
| Δ_int > 0 且显著，两库均复现 | **H1 成立**：属性条件化需平坦极小才兑现 | 触发 G4.2 排除 BN 解释 → 扩到 MIMIC/CheXpert → 进论文主线 |
| Δ_int ≈ 0，但 HA+SWAD 折间 SD 显著低于 HA（且降幅大于 ERM→SWAD） | **H2 成立**：SWAD 是 HN 的稳定器；HN 最致命的"不稳定"短板可被消除 | 触发 G4.2 → 重画 D3 逐折图 → 单独一节 |
| Δ_int ≈ 0 且方差无差异 | **H3 成立**：SWAD 收益与架构正交、与属性条件化无交互 | 触发 G4.1（查是否为双线性错配所致）→ 归入阴性证据链，与 E/F 并列 |
| G4.1 中 Θ > W 显著 | **机制发现**：双线性错配阻断了 HN 的权重平均收益 | 方法学贡献，单独成节 |
| Δ_int < 0 显著 | SWAD 与 HN 相互干扰 | 查 BN 统计失配（G4.2）与谷检测是否被 HN 的噪声 val loss 带偏 |

---

## 8. 已知坑与风险

1. **覆盖既有冻结产物**（最高优先）：本 run 以相同 (method, config, seed) 重跑 `hyperadapt`，
   会覆写 cv5 预测 / ckpt / val 日志，而 D1–D4 表依赖它们。⇒ **G0.5 备份是硬前置**。
2. **method 命名冲突**：不做 G0.1 就提交，`swad_<HA的tag>` 会被 HA-SWAD 覆写，
   直接污染 ERM-SWAD 基线（HAM 的 `swad_lr3e-05_wd1e-04` 恰好存在，**会真的撞车**）。
3. **cv5 配置 ≠ 顶层 5-seed 配置**：必须以 `cv5/selected_configs.json` 为准
   （如 HAM HyperAdapt 顶层是 `lr3e-04_wd1e-04`，cv5 是 `lr3e-05_wd1e-04`）。
4. **等 batch 纪律**：HyperAdapt 必须 b128 + gpus48；b32 曾造成 OOM 伪影导致错误的"负结果"。
5. **功效**：worst-group 在 HAM/Fitz 上的 MDE 可能大于任何合理的交互效应 ⇒ **G3.1 先算 MDE**，
   若 MDE 过大则本实验只能给"无功效"读数，须在报告中明说，不得包装成 null 判决。
6. 🛑 **`build_oof_results_averaging.py --dataset X` 会覆写整份权威 JSON**：其 `main()` 无条件把
   本次运行收集到的 `out` 写到 `outputs/conditioning_ablation/oof_results_averaging.json`，
   而 `--dataset` 只是**过滤要跑哪个 regime**——单库运行会产出只含该库的 JSON 并**整份覆盖**，
   其余 4 个 regime 的结果直接丢失。⇒ G3 重算时**必须不带 `--dataset` 跑全量**（或先备份该 JSON）。
   本轮 G0.4 因此**未**运行该脚本，改用单元级验证证明注册为 no-op。
7. **谷检测对 HN 是否稳健**：HN 的 val loss 噪声更大，loss valley 的 t_s/t_e 可能被带偏；
   G3 需记录两库 10 个 run 的 `swad_interval`，若区间长度异常（如恒为 1）须专门讨论。

---

## 9. 本轮明确移出范围（记档，供后续）

| 项 | 价值 | 为何本轮不做 |
|---|---|---|
| GroupDRO+SWAD 正对照臂 | 高——照搬 CORAL 的角色（损失层用组信息、推理盲）。若 SWAD 与 GroupDRO 可加而与 HA 不可加，即把责任精确定位到"架构式条件化"，使 null 结果可解释 | +10 run + 新方法实现，工作量超本轮预算 |
| iteration 级密采样 SWAD | 中高——原版 SWAD 是逐 iteration 密采样，我们现在是 epoch 粒度＝弱化版；若 H2 为真，采样密度正是放大杠杆 | 须在线累加实现（list 缓存在 MIMIC 上会爆内存），且会使 HA-SWAD 与既有 ERM-SWAD 不可比、需同步重跑 ERM |
| MIMIC / CheXpert 扩展 | 低——闸门决定性 0，只能确认 null | 留作 H1 成立时的确认步骤 |

---

## 10. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-08-04 | 建立。初版含 GroupDRO 正对照 + Θ 并列主口径 + iteration 级采样，为控制工作量精简为「仅 HA+SWAD、W 主口径、HAM+Fitz、10 run」，其余移入 §9 / §G4 条件触发。 |
