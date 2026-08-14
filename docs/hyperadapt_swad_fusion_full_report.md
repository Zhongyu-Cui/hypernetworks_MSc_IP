# 实验 G · HyperAdapt × SWAD 融合：全量实验报告（配置 / 方法 / 流程 / 结果）

> 日期：2026-08-09。**范围**：4 数据集 ID（HAM10000、Fitzpatrick17k、MIMIC-CXR、CheXpert）
> + MIMIC↔CheXpert 双向 OOD。
>
> **与既有文档的关系**
> - [hyperadapt_swad_fusion_plan.md](hyperadapt_swad_fusion_plan.md)：方案与预注册（功效声明 §6.5、
>   漂移核对 §6.6、config 落地坑 §6.7）。本文件**不修改**其中任何预注册条目。
> - [hyperadapt_swad_fusion.md](hyperadapt_swad_fusion.md)：**两库（HAM + Fitzpatrick）ID 结果报告**，
>   2026-08-04 完成。本文件是其**范围扩展**，不是替代；两库的数字逐位沿用。
>   扩展后有一条读数发生变化（H2，见 §5.4），已在该节明确标注。
> - [ood_experiment_v2_preregistration.md](ood_experiment_v2_preregistration.md) /
>   [ood_experiment_v2_results.md](ood_experiment_v2_results.md)：OOD 部分**严格遵循 v2 口径**
>   （full-target 估计量、5 fold × 3 trial = 15 replicate、σ_train 判据）。v2 的预注册条目同样未改动。
>
> **定位**：对比较协议 §1「SWAD 不与 HN 组合」的受控破例，属独立追加实验。
> 结论单独成文，**不回写**已冻结的 6 方法主矩阵与 D1–D4 表。

---

## 0. 摘要

本实验用 2×2 因子设计检验一个问题：**SWAD（loss-valley 权重平均）的收益，能否传递到
HyperAdapt（属性条件化超网络）架构上，并产生超越两者单独使用的增量？**

四个数据集的 ID 结果与双向 OOD 结果一致指向：

1. **传递成立，超越不成立。** 在 SWAD 自身有实质收益的唯一数据集（Fitzpatrick，worst-group +0.058）上，
   该收益近乎原样传递给 HyperAdapt（+0.045），交互项 n.s.；但融合臂与 SWAD 单独**统计上无法区分**。
2. **其余三库 SWAD 自身收益≈0，故无收益可传递。** 配置匹配后，四库交互项无一为显著正。
3. **朴素差中差在三库中均受 config 混杂污染**，且混杂幅度随 ERM/HA 两侧学习率差异单调变化。
4. **OOD（双向）上所有 AUC 效应均未超过训练噪声 σ_train**，包括融合臂；融合臂的效应两方向变号。
5. **校准侧曾是唯一候选正面读数，2026-08-09 补做 full-target × 15 replicate + σ_train 后被否定。**
   主要终点 ECE 的 8 个检验（4 对比 × 2 方向）**无一超过训练噪声**，尽管 bootstrap CI 全部 Holm 后显著；
   且「SWAD 的校准收益向 HN 等量传递」这一旧读数**不复现**（两方向 Δ_int 变号）。见 §5.3。
   本实验因此**没有任何终点上的正面结论**。
6. 校准侧新增两条**描述性**读数：SWAD 因子把 ECE 的训练间抖动压低约 40–60%（两方向 4/4 一致），
   但同时使校准斜率更过度自信（4/4 方向一致，2/4 超过 σ_train）。
7. **实现稳健性**：本项目的 SWAD 对 BatchNorm 统计做算术平均，而官方 SWAD 用 `update_bn` 重估。
   2026-08-09 补做对齐后重推理（ID 四库 + 双向 OOD），**所有结论逐条不变**（§4.7、§5.3.5）；
   BN 失配只解释了 slope 恶化的**一部分**（M→C 上大幅缓解、C→M 上未缓解）。

---

## 1. 实验配置

### 1.1 四臂（2×2 因子）

| 臂 | 属性条件化 | 权重选择 | 来源 |
|---|---|---|---|
| ERM | 无 | best-ckpt（val 早停 argmax） | 复用既有 CV-OOF 产物 |
| SWAD | 无 | loss-valley 权重平均 | 复用（ERM 同 run 派生） |
| HyperAdapt | 有（sex/race/age） | best-ckpt | 复用 |
| **HyperAdapt+SWAD** | 有 | loss-valley 权重平均 | **本实验新训** |

四格中三格复用既有产物，仅融合臂为新训练。

### 1.2 数据集与规模

| 数据集 | n（评估） | 敏感属性 | 子群数 | 分组单位 |
|---|---|---|---|---|
| HAM10000 | 9,707 | sex × age_group | 2×4 | lesion 级 |
| Fitzpatrick17k | 16,012 | skin type（单轴） | 6 | 图像级 |
| MIMIC-CXR | 199,356 | sex × race × age | 2×2×2 | 患者级 |
| CheXpert | 138,644 | sex × race × age | 2×2×2 | 患者级 |

任务均为二分类（皮肤镜 malignant；胸片 No Finding）。预处理、标签口径、split 策略沿用
仓库 [CLAUDE.md](../CLAUDE.md) 既有规格，**本实验未作任何改动**。

### 1.3 超参与 config

**超参不新搜**：融合臂复用对应 HyperAdapt 的 cv5 选定 config（同「SWAD 搭 ERM 的车」规程）。
SWAD 超参沿用 `N_s=3 / N_e=6 / r=1.3`，不调。

| 数据集 | ERM / SWAD config | HyperAdapt / 融合臂 config | 两侧差异 |
|---|---|---|---|
| HAM10000 | lr3e-04_wd1e-03 | lr3e-05_wd1e-04 | **lr 差 10×** |
| Fitzpatrick17k | lr3e-04_wd1e-03 | lr3e-04_wd1e-03 | **无（四臂同 config）** |
| MIMIC-CXR | lr1e-04_wd1e-04 | lr1e-04_wd1e-03 | 仅 wd 差 |
| CheXpert | lr1e-04_wd1e-04 | lr3e-04_wd1e-04 | **lr 差 3×** |

> config 差异不是设计缺陷，而是各方法按 S1 准则（source val 折均 marginal-worst）**各自选中**的结果。
> 但它使朴素差中差不可解释，故 §2.4 的配置匹配对照是必需项而非可选项。

### 1.4 算力与作业

| 阶段 | 作业号 | 规模 | 分区 |
|---|---|---|---|
| ID · HAM10000 | 74958 | 5 折 | gpus48 / b128 |
| ID · Fitzpatrick17k | 74965 | 5 折 | gpus48 / b128 |
| ID · MIMIC-CXR | 75205 | 5 折 | gpus48 / b128 |
| ID · CheXpert | 75206 | 5 折 | gpus48 / b128 |
| OOD · trial1/2 源模型（MIMIC） | 75216 | 5 折 × 2 trial | gpus48 / b128 |
| OOD · trial1/2 源模型（CheXpert） | 75226 | 5 折 × 2 trial | gpus48 / b128 |
| OOD · full-target 推理 | 75240 | 2 方向 × 15 replicate | gpus24 |
| OOD · 配折推理（供校准） | 75271 | 2 方向 × 5 折 | gpus24 |

合计新增训练 **40 run**、推理 **40 次**。
2026-08-09 的校准口径升级（§5.3）**未新增任何 GPU 作业**——纯 CPU 分析，在实验室机器上直接跑，
消费的是上表 75216/75226/75240 已落盘的 full-target 预测。

同日的 BN 对齐复核（§4.7 / §5.3.5）**不重训**，只做无梯度前向 + 重推理：

| 阶段 | 作业号 | 内容 |
|---|---|---|
| BN 重估 + ID 推理 | 75313 / 75314 / 75315 / 75316 | HAM / Fitz / MIMIC / CheXpert，各 10 ckpt + 10 预测 |
| BN 重估(trial1/2) + OOD 推理 | 75317 / 75318 | 双向各 20 ckpt + 30 full-target 预测 |

> 首次提交 75315/75316 时 `--export=ALL,STAGE='id,ood',TRIALS='0,1,2'` 被 sbatch 按逗号切成变量，
> 只跑了 `STAGE=id` / `TRIALS=0`（ID 部分正确、OOD 缺失）。已改为冒号分隔并在脚本内转换，
> 75317/75318 补齐。**sbatch `--export` 的值不能含逗号**。所有作业排除 `semois`（GPU device-handle 故障）；
OOD 推理另排除 `mira05`（长时推理下 GPU lost，v2 记录）。**batch 统一 128**——项目记忆记载
HyperAdapt 在 batch 不等时会产生显存混淆伪影，重型 HN 的对照必须等 batch。

---

## 2. 实验方法

### 2.1 评估口径

- **CV-OOF 5 折** + **averaging**：逐折算指标 → 折间平均 → 再 min/max 取组。
  **不使用池化（pooled）口径**——项目已确证池化 raw logit 会因跨折校准漂移制造负偏
  （学界已知缺陷，Parker 2007 / Airola 2010）。
- **终点四项**：Overall AUC、canonical worst-group AUC、**marginal worst-group AUC（主要终点）**、gap。
- **统计**：患者/病灶级**配对 cluster bootstrap**，B=1000，全臂共用同一批重采样索引；
  Holm 校正（每终点 4 个对比）。

### 2.2 交互项定义

`Δ_int = (HA+SWAD − SWAD) − (HA − ERM)`

即「SWAD 加在 HN 基座上的增量」减去「SWAD 加在 ERM 基座上的增量」。
`Δ_int > 0` 表示超可加（H1 成立）；`≈ 0` 表示可加；`< 0` 表示互相干扰。

### 2.3 预注册假设

| 假设 | 内容 |
|---|---|
| **H1** | Δ_int > 0 且跨数据集复现（融合超越两者单独） |
| **H2** | Δ_int ≈ 0 但 SWAD 压低 HN 折间方差的幅度大于压低 ERM 的幅度 |
| **H3** | Δ_int ≈ 0 且方差无一致差异（两因子正交） |

### 2.4 config 匹配对照（关键设计）

朴素差中差把「SWAD 在 config A 上的增量」与「SWAD 在 config B 上的增量」相减，混杂的是
**超参**而非架构。对照做法：利用 ERM 超参搜索阶段已落盘的产物（ERM 搜索带 `--swad` 跑满 6 配置
⇒ 每个 config 的 ERM-SWAD 对都已存在），**零训练成本**地把 ERM 侧也测到 HN 的 config 上，
再算差中差。此为方案 §2.1 预先设计的「免费对照」。

### 2.5 功效预注册（MDE）

在训练结果落盘**之前**计算最小可检测效应 MDE(80%)，写死后不得事后调整。
结构估计给出 ρ=0（保守）与 ρ=0.5（现实相关性）两档。

### 2.6 漂移核对

同 config / 同 seed 重跑并非逐位可复现（val AUC 曲线顶部平坦 + argmax 选到不同 epoch）。
故对每个数据集比较「快照 vs 新跑」的终点差与逐折 logit 相关，量化重跑噪声的绝对幅度，
并以之为尺子判断效应是否可忽略。

### 2.7 OOD 方法（严格遵循 v2）

- **估计量 = full-target**：每个 source (fold, trial) 模型评**完整** target 集，逐 (f,t) 算指标
  再等权平均，从不跨折池化。
- **replicate = 5 fold × 3 trial = 15**，`seed = 42 + fold + 10 × trial`。
  trial 的存在理由：v2 实测同 config/seed/fold 重跑的逐折 |ΔAUC| 可达 0.003，
  而待检验效应仅 0.0018–0.0019 —— **不补独立 trial 就无法判断效应是否在训练噪声之上**。
- **σ_train 判据（H2）**：效应量 |d̄| 必须超过训练随机性方差分量 σ_train 才算数。
  仅有 bootstrap 显著性**不充分**——bootstrap 按患者重采样，不含训练随机性。
- **选择准则 = S1（training-domain validation）**，即 source val 折均 marginal-worst。
  不用 oracle 选择：DomainBed（Gulrajani & Lopez-Paz, ICLR 2021）指出 test-domain selection
  不是有效的 benchmarking 方案，只应作 disclaimer 披露。
- **校准**：MIMIC 正例率 30.2% vs CheXpert 8.2%，属强 prior shift；AUC 对患病率不敏感而校准会崩塌，
  故 TRIPOD+AI 要求外部验证必报校准。指标 CITL / slope / ECE / Brier，并报交叉拟合 isotonic 重校准后的值。

---

## 3. 实验流程

### 3.1 执行顺序

1. **代码实现（G0）**：融合臂训练路径 + 派生模型命名隔离 + 分析脚本。
2. **功效预注册（G3.1）**：先算 MDE 并写死，再落训练结果。
3. **训练（G1/G2 + 本轮扩展）**：四库各 5 折。
4. **漂移核对（G1.4/G2.4 + 扩展）**：每库比较快照 vs 新跑。
5. **config key 增量合并**：融合臂的 config 键**追加**进 `selected_configs.json`，
   既有选择逐位不变（冻结版留 `.pre_G` / `.pre_G_backup`）。
6. **ID 分析（G3.2–G3.5）**：2×2 主表 → 对比与交互 → 折间方差 → 配置匹配对照。
7. **OOD 扩展**：补 trial1/2 源模型（20 run）→ full-target 推理（30 次）→ 方差分解 →
   配折推理（10 次）→ 校准分析。
8. **校准口径升级（2026-08-09）**：把校准从 cv5 配折 × 5 replicate 换到 full-target × 15 replicate，
   补 σ_train 检验（§5.3）。纯分析，无新训练/推理。

### 3.2 产物核验规程

每个训练作业结束后核验四项，全部通过方进入分析：
预测文件数 = 折数、平均权重文件数 = 折数、异常签名（Traceback / OOM / RuntimeError / Killed）计数为 0、
SWAD 谷区间实质检出（非退化为单点）。

四库谷宽实测：HAM 4–9、Fitzpatrick 2–11、MIMIC 3–9、CheXpert 4–10 个 epoch 参与平均，
**无一退化为单点** ⇒ 方案 §8 担心的「HN 的 val loss 噪声带偏谷检测」未发生。

### 3.3 覆写风险与处置（四次）

本实验涉及大量既有冻结产物的复用，共遇到四次覆写风险，均在造成损失前拦下：

| # | 风险 | 处置 |
|---|---|---|
| 1 | `_finalize_swad` 把 method 硬编码为 `"swad"`，HN 开 `--swad` 会**静默覆写 ERM-SWAD 基线** | 改为 `swad_method_name()` 并加回归自测 |
| 2 | 融合臂训练会覆写 `hyperadapt` 的 cv5 产物 | 训练前备份 + MD5 清单 |
| 3 | OOD trial1/2 训练会覆写 `hyperadapt` 的 trial1/2 产物（**v2 结果所依赖**） | 备份两库各 60 文件（2.5 GB）+ MD5 清单 |
| 4 | `c5_ood_cv.sh` 硬跑全部方法、无过滤，会覆写既有 OOD 预测 | 作业 75270 在建 dataloader 阶段取消（按 mtime 核实 **0 文件被改动**）；脚本增加 `OOD_METHODS` 过滤，缺省行为不变 |

> 方法学结论：**派生模型的命名必须与基座隔离**；**任何复用冻结产物的实验，都应在提交前
> 显式枚举它会写入哪些路径**。风险 #1 与 #4 若未拦下，都会静默污染已发表结论。

### 3.4 代码改动清单

| 文件 | 改动 | 兼容性 |
|---|---|---|
| `harness/run.py` | `swad_method_name()` 命名隔离 | 新增，ERM 路径不变 |
| `run_ood_cxr_full_target.py` | METHODS 追加融合臂；checkpoint 后缀改 `method.endswith("swad")` | 追加不插入 |
| `run_ood_cxr_cv.py` | 同上 | 同上 |
| `eval_ood_cxr.py` | METHOD_REGISTRY 注册融合臂 | 追加 |
| `ood_variance_decomposition.py` / `ood_calibration.py` | METHODS / COLORS / LABELS 追加 | 既有方法顺序与配色不变 |
| `slurm/oodv2_full_target.sh` | 新增 `METHOD_OVERRIDE`（单方法双向） | 未设时 v2 原逻辑逐位不变 |
| `slurm/c5_ood_cv.sh` | 新增 `OOD_METHODS` 过滤 | 未设时行为不变 |
| `g_drift_check.py` | 补 CXR 两库参照效应量 | 数据补充 |
| `g_ood_calibration_full_target.py` | **新增**：校准的 full-target × 15 replicate + σ_train；后加 `--bnupd` 换取数目录 | 独立脚本；复用 `ood_calibration` 的指标实现与 `ood_variance_decomposition` 的 ANOVA，两者**均未改动** |
| `g_bn_update_swad.py` / `slurm/g_bn_update.sh` | **新增**：BN 与官方 SWAD 对齐（`update_bn`）+ 重推理 | 全新产物路径（`*_averaged_bnupd.pth` / `cv5_bnupd/` / `cv5_full_target_bnupd/`），既有冻结产物零覆盖 |
| `g_bn_update_analysis.py` | **新增**：BN 对齐的 ID 侧影响（四库 2×2 + Δ_int 移动量） | 复用 `build_oof_results_averaging` 的 Spec/Views/clusters，该模块未改动 |

---

## 4. 实验结果 · ID（四数据集）

### 4.1 2×2 主表（averaging 口径点估计）

**HAM10000**（n=9,707）

| 臂 | config | Overall | canonical worst | marginal worst | gap |
|---|---|---|---|---|---|
| ERM | lr3e-04_wd1e-03 | 0.8938 | 0.7877 | 0.7976 | 0.1348 |
| SWAD | lr3e-04_wd1e-03 | **0.9129** | **0.8331** | **0.8433** | **0.1024** |
| HyperAdapt | lr3e-05_wd1e-04 | 0.8910 | 0.7990 | 0.8217 | 0.1190 |
| HyperAdapt+SWAD | lr3e-05_wd1e-04 | 0.8917 | 0.7994 | 0.8257 | 0.1185 |

**Fitzpatrick17k**（n=16,012，**四臂同 config**，无混杂）

| 臂 | Overall | marginal worst（=canonical，单轴） | gap |
|---|---|---|---|
| ERM | 0.8962 | 0.8161 | 0.0904 |
| SWAD | **0.9205** | 0.8739 | **0.0578** |
| HyperAdapt | 0.8935 | 0.8306 | 0.0895 |
| **HyperAdapt+SWAD** | 0.9197 | **0.8752** | 0.0657 |

**MIMIC-CXR**（n=199,356）

| 臂 | config | Overall | canonical worst | marginal worst | gap |
|---|---|---|---|---|---|
| ERM | lr1e-04_wd1e-04 | 0.8456 | 0.8089 | 0.8178 | 0.0634 |
| SWAD | lr1e-04_wd1e-04 | **0.8474** | **0.8096** | **0.8194** | 0.0627 |
| HyperAdapt | lr1e-04_wd1e-03 | 0.8461 | 0.8090 | 0.8192 | 0.0630 |
| HyperAdapt+SWAD | lr1e-04_wd1e-03 | 0.8469 | 0.8085 | 0.8187 | 0.0644 |

**CheXpert**（n=138,644）

| 臂 | config | Overall | canonical worst | marginal worst | gap |
|---|---|---|---|---|---|
| ERM | lr1e-04_wd1e-04 | 0.8701 | 0.8316 | 0.8377 | 0.0516 |
| SWAD | lr1e-04_wd1e-04 | 0.8699 | 0.8317 | 0.8354 | 0.0551 |
| HyperAdapt | lr3e-04_wd1e-04 | 0.8677 | 0.8335 | 0.8359 | 0.0490 |
| **HyperAdapt+SWAD** | lr3e-04_wd1e-04 | **0.8729** | **0.8412** | **0.8424** | **0.0485** |

> ⚠️ 除 Fitzpatrick 外，各库上下两组 config 不同 ⇒ **不可直接做差中差**，须读 §4.3。

### 4.2 Fitzpatrick：唯一干净的传递证据

四臂同 config，无混杂，故此库的读数可直接解释：

| 基座 | Δ marginal worst | Δ Overall |
|---|---|---|
| ERM → SWAD | +0.0578 | +0.0244 |
| HyperAdapt → HA+SWAD | **+0.0446** | **+0.0263** |

两个增量量级相当，即**可加性的直接证据**。对比检验：

| 对比 | Δ marginal worst | 95% CI | p_Holm |
|---|---|---|---|
| HA+SWAD − ERM | +0.0591 | [+0.016, +0.106] | **0.042 显著** |
| HA+SWAD − SWAD | +0.0013 | [−0.031, +0.031] | 1.000 n.s. |
| HA+SWAD − HA | +0.0446 | [+0.012, +0.080] | **0.016 显著** |
| **Δ_int** | −0.0132 | [−0.066, +0.036] | 1.000 n.s. |

读法：融合臂显著优于 ERM、也显著优于 HyperAdapt 单独，**但与 SWAD 单独打平**；交互项为零。
⇒ 全部增量来自 SWAD 因子，HN 因子贡献 0，两者不互相干扰。

### 4.3 config 匹配对照：朴素交互项在三库中均受污染

| 数据集 | 朴素 Δ_int（主要终点） | 匹配后 Δ_int | 匹配后 p | 两侧 lr 差异 |
|---|---|---|---|---|
| HAM10000 | −0.0184（Overall，**Holm .004**） | **+0.0017** | 0.580 | **10×** |
| CheXpert | +0.0088（marg worst，**Holm .004**） | **+0.0022** | 0.204 | **3×** |
| MIMIC-CXR | −0.0021（marg worst，Holm .024） | −0.0025 | 0.002 | 仅 wd |
| Fitzpatrick | −0.0132（n.s.） | 不适用（本就同 config） | — | 无 |

**两项观察：**

1. **HAM 与 CheXpert 的朴素显著性均在匹配后消失**，且 HAM 的符号翻转。若无此对照，
   两库会各自得出一个**方向相反且高置信**的错误结论。
2. **混杂幅度随两侧 lr 差异单调变化**：lr 差 10× 的 HAM 受污染最重，差 3× 的 CheXpert 次之，
   仅 wd 差的 MIMIC 几乎不受影响。这为「SWAD 收益由学习率主导」提供了一条**独立于 HAM 的**证据。

**匹配后各库 SWAD 增量（主要终点 marginal worst）**：

| 数据集 | ERM 侧 | HN 侧 | Δ_int | 判定 |
|---|---|---|---|---|
| Fitzpatrick | +0.0578 | +0.0446 | −0.0132 | n.s. |
| HAM10000 | +0.0021 | +0.0040 | +0.0019 | n.s.（两侧均≈0） |
| CheXpert | +0.0043 | +0.0065 | +0.0022 | n.s. |
| MIMIC-CXR | +0.0020 | −0.0005 | **−0.0025** | **p=0.002 显著为负** |

**H1（Δ_int > 0 且跨库复现）在四库中无一支持。**

MIMIC 的显著负交互需谨慎解读：|Δ|=0.0025，**小于该库 SWAD 主效应本身（+0.0020）的量级**，
且仅为重跑漂移（0.0009）的 2.8 倍。它是「n=199k 带来的功效」而非「实质重要性」的体现。

### 4.4 折间方差（H2，描述性）

逐折 marginal worst-group AUC 的标准差（5 折）：

| 数据集 | ERM | SWAD | HyperAdapt | HA+SWAD | SWAD 对 ERM 侧 | SWAD 对 HN 侧 | 差 | 方向 |
|---|---|---|---|---|---|---|---|---|
| Fitzpatrick | 0.0794 | 0.0486 | 0.0940 | 0.0536 | −0.0308 | **−0.0404** | −0.0096 | 支持 |
| CheXpert | 0.0096 | 0.0137 | 0.0127 | 0.0120 | +0.0041 | −0.0007 | −0.0048 | 支持 |
| MIMIC-CXR | 0.0026 | 0.0048 | 0.0053 | 0.0041 | +0.0022 | −0.0011 | −0.0033 | 支持 |
| HAM（匹配 @ lr3e-05） | 0.0658 | 0.0584 | 0.0452 | 0.0489 | −0.0075 | +0.0037 | +0.0112 | 不支持 |

> 🛑 **相对两库版报告的读数变化**：[hyperadapt_swad_fusion.md](hyperadapt_swad_fusion.md) §4 基于
> HAM + Fitzpatrick 判「H2 两库不一致、不作宣称」。补齐 CXR 两库后变为 **4 库中 3 库方向一致**
> （SWAD 压低 HN 折间抖动的幅度大于压低 ERM 的幅度）。
>
> **但这仍不构成宣称**：5 折无检验功效，四项差值均为**描述性**读数，且未做多重比较控制。
> 恰当表述为「H2 方向获多数数据集的一致描述性支持，但未获检验」，
> **不得**表述为「H2 成立」。

### 4.5 功效声明（预注册，不得违反）

| 数据集 | 终点 | MDE(80%) | 参照真效应（SWAD−ERM） | 判定 |
|---|---|---|---|---|
| HAM / Fitzpatrick | Overall | 0.007–0.012 | 0.019–0.024 | 有功效 |
| HAM / Fitzpatrick | marginal worst | 0.045–0.089 | 0.046–0.058 | 🛑 **无功效** |
| MIMIC-CXR | marginal worst | **0.0015**（ρ=0.5）/ 0.0021（ρ=0） | 0.0016 | 有功效 |
| CheXpert | marginal worst | **0.0041**（ρ=0.5）/ 0.0057（ρ=0） | −0.0023 | 有功效 |

**两库版报告的核心局限（worst-group 无功效）在 CXR 两库上被解决**：MDE 下降 10–40 倍。

> ⚠️ **但必须同时限定**：CXR 两库的 **SWAD 主效应本身≈0**（MIMIC +0.0016，CheXpert −0.0023）。
> 也就是说，这里的交互检验虽功效充足，测的却是一个**退化情形**——没有多少「SWAD 收益」可供传递。
> 「有功效」与「所测现象存在」是两件事，不可混淆。

事后核对：观测到的 SE(Δ_int, marginal worst, HAM) ≈ 0.016，与预注册结构估计 ρ=0.5 档的 0.0162
几乎逐位吻合 ⇒ 功效预估方法本身得到验证。

### 4.6 漂移核对

| 数据集 | Δ marginal worst（快照 vs 新跑） | 占参照效应量 | 逐折 logit Pearson r |
|---|---|---|---|
| HAM10000 | −0.0024 | 14% | 0.960–0.998 |
| Fitzpatrick17k | +0.0028 | 14% | **0.74–0.90** |
| MIMIC-CXR | −0.0009 | **40%** | 0.947–0.990 |
| CheXpert | +0.0005 | 17% | 0.896–0.978 |

CXR 两库的漂移**绝对量更小**（0.0005–0.0009 vs 皮肤镜 0.0024–0.0028），符合 n 更大的预期。

> **但 MIMIC 的相对量达 40%**，因为主矩阵中「MIMIC·HyperAdapt +0.0022（唯一深层 HN 显著为正）」
> 这一效应本身极小。换言之，**该已发表的显著结果仅为重跑噪声的 2.4 倍**。
> 这为 [crossdataset_worstgroup_fairness.md](crossdataset_worstgroup_fairness.md) 中已有的
> 「显著 ≠ 重要」提示补了一个量化刻度。

### 4.7 BN 统计与官方 SWAD 对齐后的复核（ID 侧，2026-08-09）

**背景**：本项目的权重平均（`harness/swad.py::average_state_dicts`）对 BatchNorm 的
`running_mean` / `running_var` 也做**算术平均**；官方 SWAD（khanrc/swad，`domainbed/trainer.py`）
在平均后调用 `swa_utils.update_bn(train_iter, model, n_steps=500)` 用训练数据**重估**。
后者是 SWA 原论文（Izmailov et al., 2018）的标准要求——平均权重产生的激活分布不等于任一单点的，
`running_var` 尤其不可线性平均。故本节把 BN 处理与官方对齐后重推理，检验结论是否稳健。

**做法**：消费既有 `*_averaged.pth`，**不重训**；`update_bn` 为无梯度前向，每臂用**自己训练时的
loader**（HAM 两臂训练集不同）。**只有两个权重平均派生臂被影响**，ERM / HyperAdapt 读原预测，
逐位不变（已在输出中核对）。

| 数据集 | SWAD marg worst | HA+SWAD marg worst | 四终点最大 \|Δ\| | Δ_int(marg worst) 移动量 |
|---|---|---|---|---|
| HAM10000 | 0.8433 → 0.8443 | 0.8257 → 0.8251 | 0.0024 | −0.0015 n.s. |
| Fitzpatrick17k | 0.8739 → 0.8707 | 0.8752 → 0.8700 | **0.0072** | −0.0020 n.s. |
| MIMIC-CXR | 0.8194 → 0.8185 | 0.8187 → 0.8179 | 0.0009 | +0.0000 n.s. |
| CheXpert | 0.8354 → 0.8348 | 0.8424 → 0.8424 | 0.0005 | +0.0005 n.s. |

**判定：ID 侧全部结论不变。** 四库最大变化 0.0072（Fitzpatrick 的 gap），Δ_int 的移动量
**四库全部 n.s.**。CXR 两库出现的几个「显著」（如 MIMIC SWAD marginal worst −0.0009）
量级都在重跑漂移（§4.6，0.0005–0.0009）之内，是 n=14–20 万的功效而非实质。

> 本表只读**移动量**；Δ_int 的绝对值须按 §4.3 的 **config 匹配**口径解读（此处为朴素口径）。
> config 混杂对 BN 前后是共同的，故移动量不受其影响。

---

## 5. 实验结果 · OOD（MIMIC↔CheXpert 双向）

### 5.1 口径

严格遵循 v2：full-target 估计量、15 replicate、S1 选择准则、σ_train 判据。
融合臂在两方向各产出 15 个 full-target 预测，与 ERM / SWAD / HyperAdapt 三个对照臂**逐位对齐**。

### 5.2 marginal worst-group AUC 与 σ_train 判据

| 方向 | ERM | SWAD | HyperAdapt | HA+SWAD |
|---|---|---|---|---|
| M→C | 0.8125 | 0.8078 | **0.8156** | 0.8103 |
| C→M | 0.7882 | 0.7875 | 0.7857 | **0.7910** |

| 方向 | 比较 | Δ | bootstrap 判定 | σ_train | t | **H2（超过训练噪声？）** |
|---|---|---|---|---|---|---|
| M→C | SWAD − ERM | −0.0047 | Holm 后显著 | 0.0067 | −2.11 | ❌ |
| M→C | HyperAdapt − ERM | +0.0031 | Holm 后显著 | 0.0086 | 1.27 | ❌ |
| M→C | HA+SWAD − ERM | −0.0023 | Holm 后显著 | 0.0070 | −1.25 | ❌ |
| C→M | SWAD − ERM | −0.0007 | Holm 后显著 | 0.0072 | −0.31 | ❌ |
| C→M | HyperAdapt − ERM | −0.0025 | Holm 后显著 | 0.0064 | −0.95 | ❌ |
| C→M | HA+SWAD − ERM | +0.0028 | Holm 后显著 | 0.0040 | 1.56 | ❌ |

**六个检验全部「Holm 后仍显著」，但无一超过训练噪声。** 这与 v2 对 HyperAdapt 的判决同一模式，
也再次说明：**cluster bootstrap 的显著性在本设定下不足以支撑结论**——它按患者重采样，
不含训练随机性，而训练随机性才是这个尺度上的主导项。

**两点补充观察：**

1. **SWAD 在 OOD 上未提供 worst-group 收益，M→C 方向为负（−0.0047）。**
   作为域泛化方法这与其定位相悖，但与 ID 侧发现一致：SWAD 收益依赖操作点，
   而 CXR 两库的选中操作点上它本就≈0。**没有收益可供传递。**
2. **融合臂的效应两方向变号**（M→C −0.0023 / C→M +0.0028），
   正是 v2 用以否定 HyperAdapt H1 的「不跨方向复现」。

由点估计导出的交互项：M→C `Δ_int = −0.0006`，C→M `Δ_int = +0.0060`，
**均远小于 σ_train 且符号相反** ⇒ **OOD 上不支持任何交互结论。**

### 5.3 校准（强 prior shift 下的次要终点）

source π=30.2% → target π=8.2%（M→C）与其反向，属强 prior shift。

> **口径升级（2026-08-09，原 §7.2 未决项 1）**：本节已由 **full-target 估计量 × 5 fold × 3 trial
> = 15 replicate** 重做，与 §5.2 的 AUC 主分析**逐位同口径**，并补上 v2 对 AUC 强制要求的
> **σ_train 检验**。旧版基于 cv5 配折产物、仅 trial 0（5 replicate）、无 σ_train，其两条读数的下场见 §5.3.3。
> 脚本 [g_ood_calibration_full_target.py](../scripts/g_ood_calibration_full_target.py)，
> **无新增训练**（消费作业 75216/75226/75240 的既有预测）。`ood_calibration.py`（v2 D2）未改动。

#### 5.3.1 点估计（15 replicate 等权平均）

| 方向 | 指标 | ERM | SWAD | HyperAdapt | HA+SWAD |
|---|---|---|---|---|---|
| M→C | ECE（=\|CITL\|） | 0.1069 | **0.0991** | 0.1177 | **0.0991** |
| M→C | slope（1=完美） | 1.094 | 0.843 | 1.027 | 0.838 |
| M→C | Brier | **0.0780** | 0.0816 | 0.0829 | 0.0819 |
| C→M | ECE（=\|CITL\|） | 0.1601 | 0.1482 | 0.1571 | **0.1473** |
| C→M | slope | 0.718 | 0.594 | 0.715 | 0.654 |
| C→M | Brier | 0.1826 | 0.1762 | 0.1830 | **0.1747** |

> **ECE 与 |CITL| 在本设定下逐位相等**——强 prior shift 使各箱偏移同号，ECE 退化为整体偏移的绝对值。
> 二者**不是两份独立证据**。（点估计与旧 cv5 口径有 0.005–0.02 的差异，源于估计量与 replicate 数变更。）

> **重校准后四臂差异消失**：交叉拟合 isotonic 后 ECE 全部落到 0.0008–0.0011（两方向），
> 且 marginal worst AUC 重校准前后仅差 ≤0.0015（并列分数所致）。
> ⇒ 校准差异是**可由单层单调映射修复的刻度问题，不触及判别**，§5.2 的 AUC 结论不受影响。

#### 5.3.2 σ_train 判据：主要终点 ECE 上**八个检验全部未过关**

| 方向 | 对比 | ΔECE | 95% CI（H1） | σ_train | t | H2 |
|---|---|---|---|---|---|---|
| M→C | SWAD − ERM | −0.0078 | [−0.0082, −0.0074] | 0.0247 | −1.11 | ❌ |
| M→C | HyperAdapt − ERM | +0.0109 | [+0.0106, +0.0111] | 0.0339 | +1.24 | ❌ |
| M→C | HA+SWAD − ERM | −0.0078 | [−0.0082, −0.0074] | 0.0229 | −1.31 | ❌ |
| M→C | **Δ_int** | −0.0108 | [−0.0111, −0.0105] | 0.0401 | −1.04 | ❌ |
| C→M | SWAD − ERM | −0.0119 | [−0.0123, −0.0115] | 0.0205 | −2.25 | ❌ |
| C→M | HyperAdapt − ERM | −0.0030 | [−0.0032, −0.0028] | 0.0281 | −0.40 | ❌ |
| C→M | HA+SWAD − ERM | −0.0128 | [−0.0132, −0.0124] | 0.0265 | −1.87 | ❌ |
| C→M | **Δ_int** | +0.0021 | [+0.0019, +0.0023] | 0.0367 | +0.19 | ❌ |

（ΔECE<0 = 校准更好；Δ_int = (HA+SWAD − HA) − (SWAD − ERM)，与 §2.2 定义代数等价。）

**八个 H1 检验全部 Holm 后显著（p≈0），却无一超过训练噪声**——bootstrap CI 宽度约 0.0008，
而 σ_train 为 0.020–0.040，**相差 25–50 倍**。这与 §5.2 AUC 的模式逐条重合，
是「cluster bootstrap 显著性不是充分判据」的第二份、且更极端的实证。

#### 5.3.3 旧读数（cv5 × 5 replicate）在新口径下的下场

| 旧读数 | 新口径判定 |
|---|---|
| ① v2 记录的 HyperAdapt 校准劣化，在融合臂上不再出现 | **方向部分复现，但不成立** |
| ② SWAD 的校准收益向 HN 传递量级一致（两方向差同为 +0.0012） | 🛑 **不复现，撤回** |

① M→C 上 HyperAdapt 仍劣于 ERM（+0.0109）、融合臂转为优于 ERM（−0.0078），方向与旧读数一致；
但两者 |d̄| 均远小于各自 σ_train（0.0339 / 0.0229）。且 **C→M 上 HyperAdapt 本就不劣于 ERM（−0.0030）**，
「HyperAdapt OOD 校准劣化」本身**不跨方向复现**。

② full-target 下两方向的传递差**变号**：

| 方向 | SWAD 给 ERM 的 ECE 改善 | 给 HyperAdapt 的改善 | Δ_int | σ_train |
|---|---|---|---|---|
| M→C | −0.0078 | −0.0186 | **−0.0108** | 0.0401 |
| C→M | −0.0119 | −0.0098 | **+0.0021** | 0.0367 |

旧口径的「两方向差同为 +0.0012」是 5-replicate 下的巧合。
⇒ **「可加性在校准上比在 AUC 上呈现得更清晰」这一表述撤回**；校准上的 Δ_int 与 AUC 上一样，
两方向变号且远低于训练噪声。

#### 5.3.4 两条新增读数（描述性，跨方向一致）

**① SWAD 因子稳定校准**：ECE 在 15 个 (fold, trial) cell 上的 SD，两方向 4/4 一致被 SWAD 压低：

| 方向 | ERM | SWAD | HyperAdapt | HA+SWAD |
|---|---|---|---|---|
| M→C | 0.0200 | **0.0083** | 0.0170 | **0.0125** |
| C→M | 0.0229 | **0.0144** | 0.0304 | **0.0121** |

与 ID 侧 §4.4 的 H2（SWAD 压低折间抖动）方向一致，并在 OOD 校准上跨两方向复现。
**仍属描述性**——15 个 cell、未做方差比检验、未控多重比较，不得表述为「已证」。

**② 但 SWAD 让校准斜率更过度自信**：|slope − 1| 在四个「加 SWAD」对比中 **4/4 为正**（更远离 1），
其中 **2 个超过 σ_train**（C→M 的 SWAD−ERM +0.1239 > 0.0776；M→C 的 HA+SWAD−HA +0.1160 > 0.0684）。
slope 由 1.094→0.843（M→C）、0.718→0.594（C→M）。
⇒ SWAD 把**整体偏移**（CITL/ECE）拉近的同时**压缩了分数分布**。
这是本实验**唯二超过训练噪声的校准读数，且方向为负**——即 SWAD 在 OOD 校准上并非纯改善。

#### 5.3.5 BN 与官方 SWAD 对齐后的复核（2026-08-09）

§5.3.4② 的自然机制假说是 **BN 统计失配**（背景与做法见 §4.7）：平均权重后 BN running stats
不再匹配，典型后果正是「判别不变、刻度偏移」。本节把 BN 处理与官方对齐后在双向 OOD 上重推理。
ERM / HyperAdapt 读原预测、逐位不变（已核对：ECE / slope / Brier / marginal worst 四项全等）。

**点估计（原 → BN 对齐）**

| 方向 | 臂 | ECE | slope | Brier | marginal worst |
|---|---|---|---|---|---|
| M→C | SWAD | 0.0991 → 0.0971 | 0.843 → **0.880** | 0.0816 → 0.0799 | 0.8078 → 0.8067 |
| M→C | HA+SWAD | 0.0991 → 0.0990 | 0.838 → **0.869** | 0.0819 → 0.0809 | 0.8103 → 0.8089 |
| C→M | SWAD | 0.1482 → 0.1505 | 0.594 → 0.585 | 0.1762 → 0.1770 | 0.7875 → 0.7870 |
| C→M | HA+SWAD | 0.1473 → 0.1549 | 0.654 → **0.679** | 0.1747 → 0.1783 | 0.7910 → 0.7897 |

**判别几乎不动**（marginal worst 变化 ≤0.0014），与「BN 失配是刻度问题」一致。

**slope 恶化的 σ_train 判据（|slope−1| 的 Δ，>0 = SWAD 让斜率更远离 1）**

| 方向 | 对比 | 原 d̄ | 新 d̄ | 新 σ_train | 原判 / 新判 |
|---|---|---|---|---|---|
| M→C | SWAD − ERM | +0.0442 | **+0.0068** | 0.1093 | ❌ / ❌ |
| M→C | HA+SWAD − HA | +0.1160 | +0.0856 | 0.0711 | ✅ / **✅** |
| C→M | SWAD − ERM | +0.1239 | +0.1337 | 0.0691 | ✅ / **✅** |
| C→M | HA+SWAD − HA | +0.0613 | +0.0364 | 0.0897 | ❌ / ❌ |

**判定：假说部分成立，但不足以解释该现象。**

1. **M→C 上 BN 失配确实是主要来源**：SWAD−ERM 的 slope 恶化被消掉 **85%**（+0.0442→+0.0068）。
2. **C→M 上不成立**：SWAD−ERM 反而略微更差（+0.1239→+0.1337）。
3. **整体判决逐条不变**：四个「加 SWAD」对比的 |slope−1| 变化仍 **4/4 为正**、**2/4 超过 σ_train**，
   且超噪声的仍是原来那两个。⇒ §5.3.4② **保留**，只是不能全部归因于 BN。

**主要终点 ECE 亦不变**：八个检验的 |d̄| 仍**无一超过 σ_train**（新 |d̄| 0.0022–0.0097 vs
σ_train 0.019–0.038）；§5.3.2 的判决与 §5.3.3 对两条旧读数的撤回**均不受影响**。
§5.3.4① 的抖动压低在对齐后同样成立且略强（ECE 的 15-cell SD：M→C ERM 0.0200 / SWAD 0.0086 /
HA 0.0170 / HA+SWAD 0.0108；C→M 0.0229 / 0.0123 / 0.0304 / 0.0098，仍 4/4 一致）。

---

## 6. 结论

### 6.1 对预注册假设的判定

| 假设 | 判定 | 依据 |
|---|---|---|
| **H1**（融合超越两者单独） | **未获支持** | 四库 config 匹配后 Δ_int 无一显著为正；Fitzpatrick 上融合臂与 SWAD 单独打平（+0.0013 n.s.）；OOD 两方向变号且均低于 σ_train |
| **H2**（SWAD 更强地稳定 HN 方差） | **方向获多数支持，未获检验** | ID 四库中 3 库方向一致；5 折无功效，仅描述性。OOD 校准侧另见 §5.3.4①：SWAD 压低 ECE 训练间抖动，两方向 4/4 一致，但 HN 侧压得**不比** ERM 侧更狠（M→C 0.0170→0.0125 vs 0.0200→0.0083），故只支持「SWAD 稳定方差」、不支持 H2 的「对 HN 更强」 |
| **H3**（两因子正交可加） | **与数据一致** | Fitzpatrick 干净证据 + 三库匹配后 Δ_int≈0 |

### 6.2 主结论（谨慎表述）

**HyperAdapt + SWAD ≈ SWAD。融合是可加的，但因 HyperAdapt 自身增量为 0，可加的结果是追平 SWAD、
而非超越。** 这与预注册的定量零假设 `HA+SWAD ≈ SWAD + (HA − ERM) ≈ SWAD` 逐条吻合。

范围扩展到四库 + 双向 OOD 后，该结论**未被推翻，且获得了功效更强的支持**：
CXR 两库把 worst-group 的 MDE 降低了 10–40 倍，在此功效下仍未出现正交互。

### 6.3 附带结论

1. **SWAD 的收益由操作点（尤其学习率）主导，而非架构。** 证据现有三条：
   HAM 上 lr3e-04→+0.046 而 lr3e-05→+0.002；CheXpert 上 lr1e-04→−0.0023 而 lr3e-04→+0.0043（符号翻转）；
   config 混杂幅度随两侧 lr 差异单调变化。
   ⇒ 项目论断「SWAD 是最稳健的 worst-group 公平性基线」应限定为「**在其选中操作点上**最稳健」。
2. **差中差必须匹配 config**，否则会得到方向相反的高置信错误结论（HAM、CheXpert 两次实证）。
3. **cluster bootstrap 显著性在本设定下不是充分判据**，必须配 σ_train
   （OOD AUC 六个检验全中；补做后 OOD 校准 ECE 八个检验又全中，CI 宽度与 σ_train 相差 25–50 倍）。
4. **replicate 数不足会制造「跨方向一致」的假象**：旧 cv5 × 5 replicate 口径下
   「SWAD 的校准收益传递差两方向同为 +0.0012」看似干净复现，扩到 15 replicate 后直接变号（§5.3.3）。
   ⇒ 跨方向/跨数据集的一致性本身**不能**替代噪声尺度的核查。
5. **本项目 SWAD 相对官方的实现简化不影响任何结论**（§4.7 / §5.3.5）。已实测的最大偏离是
   **BN 统计处理**（算术平均 vs `update_bn` 重估）：对齐后 ID 四终点变化 ≤0.0072、Δ_int 移动量全 n.s.，
   OOD 的 ECE/σ_train 判决与 slope 判决逐条不变。BN 失配只解释了 slope 恶化的一部分
   （M→C 消掉 85%，C→M 未缓解），故该现象另有来源。

---

## 7. 限制与未决

### 7.1 限制

- **HAM / Fitzpatrick 的 worst-group 终点无检验功效**（MDE 0.045–0.089 ≈ 参照效应量）。
  这两库的 null 只能报「测不出」，不得表述为「证明无交互」。
- **CXR 两库虽有功效，但所测为退化情形**（SWAD 主效应本身≈0）。
  「在功效充足下未见交互」的说服力，受限于「本就没有收益可供传递」这一前提。
- **OOD 校准读数已补 σ_train 检验并被否定**（§5.3.2，2026-08-09）。原「唯一可能给出正面结论的方向」
  就此关闭 ⇒ **本实验在 ID 与 OOD、AUC 与校准的所有主要终点上均无正面结论**。
  校准侧仅余两条**描述性**读数（§5.3.4），不得作结论使用。
- **单一 HN 架构**：仅测 HyperAdapt。条件化深度轴（HyperHead / HyperFusion）未纳入，
  故本实验**不能**推出任何关于「条件化深度 × SWAD」的结论。
- **SWAD 为 epoch 粒度**，非原版的 iteration 级密采样（官方每 `checkpoint_freq` 步一个 segment，
  采样点多一到两个数量级）。**这一条仍未被检验**；另一处实现偏离（BN 统计）已对齐验证、
  结论不变（§4.7 / §5.3.5）。
- **未做 GroupDRO 正对照臂**，故无法回答「SWAD 是否与*任何*方法都可加」。
- 四库中 HN 增量为 0 是本结论的前提；在信号闸门为正的数据集上（本项目暂无）结论未必成立。

### 7.2 未决项（按优先级）

1. ~~**把 OOD 校准分析扩到 full-target × 15 replicate**，补 σ_train 检验。~~
   **✅ 2026-08-09 完成**（§5.3）。结果为**否定**：八个 ECE 检验无一超过 σ_train，
   旧「传递量级一致」读数不复现。产物见 §8。
2. ~~若 1 成立，考虑 BN 统计后校准（G4.2）作为机制侧追问。~~
   **前提未成立 ⇒ 取消**（1 为否定结果，无待解释的正面效应）。
3. **（现列首位）系统的 lr 扫描**，以确证 §6.3 的第 1 条附带结论
   （SWAD 收益由操作点主导；现有证据为 3 个数据集上的零散观察）。
4. **§5.3.4② 的机制追问**：SWAD 为何在压低 |CITL| 的同时压缩分数分布、使 slope 更远离 1。
   **第一个候选（BN 统计失配）已于 2026-08-09 检验：部分成立但不充分**（§5.3.5）——
   M→C 上消掉 85%、C→M 上未缓解，判决不变。若继续追，下一个候选是
   **epoch 粒度 vs 官方 iteration 级密采样**（§7.1），成本高于 BN 对齐（需重训）。低优先。

---

## 8. 产物索引

| 类别 | 路径 |
|---|---|
| ID 分析（两库） | `outputs/analysis/g_fusion/{fusion_2x2,mde,drift_check}.json` |
| ID 分析（CXR 两库） | `outputs/analysis/g_fusion/{fusion_2x2_cxr,mde_cxr,drift_check_cxr}.json` |
| OOD full-target 预测 | `outputs/ood_cxr/{mimic2chexpert,chexpert2mimic}/cv5_full_target/predictions/` |
| OOD 配折预测 | `outputs/ood_cxr/{mimic2chexpert,chexpert2mimic}/cv5/predictions/` |
| OOD 方差分解 | `outputs/ood_cxr/variance_decomposition_g/` |
| OOD 校准（cv5 配折，v2 D2 口径） | `outputs/ood_cxr/calibration_g/` |
| **OOD 校准（full-target × 15 replicate + σ_train）** | `outputs/ood_cxr/calibration_g_full_target/{calibration_full_target.json,calibration_full_target_ece.png}` |
| **BN 对齐后的 checkpoint** | `outputs/{ham10000,fitzpatrick,mimic_cxr,chexpert_cxr}/cv5/*_averaged_bnupd.pth`（80 个） |
| **BN 对齐后的 ID 预测 / 分析** | `outputs/<ds>/cv5_bnupd/predictions/`；`outputs/analysis/g_fusion/bn_update_id.json` |
| **BN 对齐后的 OOD 预测 / 校准** | `outputs/ood_cxr/<dir>/cv5_full_target_bnupd/predictions/`；`outputs/ood_cxr/calibration_g_full_target_bnupd/` |
| 冻结产物备份 | `outputs/{ham10000,fitzpatrick,mimic_cxr,chexpert_cxr}/cv5/_pre_G_backup/`（含 MD5 清单） |

**生成脚本**：[g_fusion_analysis.py](../scripts/g_fusion_analysis.py)（2×2 / 对比 / 交互 / 方差 / 配置匹配）、
[g_mde_worstgroup.py](../scripts/g_mde_worstgroup.py)（功效）、[g_drift_check.py](../scripts/g_drift_check.py)（漂移）、
[ood_variance_decomposition.py](../scripts/ood_variance_decomposition.py)、
[ood_calibration.py](../scripts/ood_calibration.py)（cv5 口径，未改动）、
[g_ood_calibration_full_target.py](../scripts/g_ood_calibration_full_target.py)（full-target × 15 replicate + σ_train，2026-08-09 新增；`--bnupd` 走 BN 对齐取数）、
[g_bn_update_swad.py](../scripts/g_bn_update_swad.py)（BN 与官方 SWAD 对齐 + 重推理，配 [slurm/g_bn_update.sh](../slurm/g_bn_update.sh)）、
[g_bn_update_analysis.py](../scripts/g_bn_update_analysis.py)（BN 对齐的 ID 侧影响）。
