# OOD 实验 v2 · HyperHead / HyperFusion 两臂与条件化深度轴（MIMIC↔CheXpert）

> 预注册：`docs/ood_experiment_v2_preregistration.md` **偏离 #4**（范围扩充 + H6/H7 + Holm family 6→10）。
> 主文件：`docs/ood_experiment_v2_results.md`（ERM / SWAD / HyperAdapt）· 第四臂：`docs/ood_experiment_v2_groupdro.md`。
> 口径：**full-target** 估计量 · **5 fold × 3 trial = 15 replicate** · 患者级配对 cluster bootstrap。
> 日期：2026-08-28

---

## 0. 一句话结论

> **H6（HyperHead）与 H7（HyperFusion）双双否定，每条的两项判据、两个方向共八项全败。**
> 二者的 marginal-worst 在 M→C 为微正（+0.0010 n.s. / +0.0016 显著）、在 C→M **均显著为劣**
> （−0.0022 / −0.0015），方向未复现；效应量在四个格子里全部**远小于训练噪声**（|t| = 0.34–1.31）。
> 至此 **OOD v2 六臂完成，全部正面效应无一超过训练噪声**——唯一超噪声的仍是 GroupDRO 的那个**负**效应。

**对项目主论点的作用**：偏离 #4 之前，OOD 侧只有三个 HN 中的一个（HyperAdapt）在场，§1 因此禁止
任何深度结论。补齐后，**ID 侧早已成立的判决「属性条件化的深度不是决定因素」在 OOD 侧得到独立复现**：
三个介入深度递增的 HN 在跨库部署下彼此差异全在噪声内，且都没有相对 attribute-blind ERM 的稳健优势。

---

## 1. 假设判定

| 编号 | 假设（偏离 #4 事前写定） | 判定 |
|---|---|---|
| **H6** | HyperHead marginal-worst 优于 ERM，**两方向复现**，**且** \|d̄\| > σ_train | ❌ **四项全败** |
| **H7** | HyperFusion 同上 | ❌ **四项全败** |

判据与 H1/H2（HyperAdapt）、H5（GroupDRO）逐字同构，未新设更宽松标准。

---

## 2. 主结果（marginal worst-group AUC，full-target，15 replicate）

### 2.1 六臂点估计

| 方向 | ERM | SWAD | GroupDRO | **HyperHead** | **HyperFusion** | HyperAdapt |
|---|---|---|---|---|---|---|
| M→C | 0.8125 | 0.8078 | 0.8056 | **0.8135** | **0.8141** | 0.8156 |
| C→M | **0.7882** | 0.7875 | 0.7821 | **0.7860** | **0.7867** | 0.7857 |

（config：M→C 两臂均 `lr1e-04_wd1e-04`；C→M HyperHead `lr3e-04_wd1e-04`、HyperFusion `lr1e-04_wd1e-04`，
均为各自 cv5 `selected_configs.json` 的 S1 折均 marginal-worst 选中值。）

### 2.2 两臂 vs ERM

| 方向 | 比较 | Δ | 95% CI | H1（样本） | \|d̄\| | **σ_train** | t | **H2（训练）** |
|---|---|---|---|---|---|---|---|---|
| M→C | HyperHead − ERM | **+0.0010** | [−0.0003, +0.0023] | **n.s.** | 0.0010 | **0.0089** | 0.34 | ❌ 未超过噪声 |
| M→C | HyperFusion − ERM | **+0.0016** | [+0.0003, +0.0028] | 显著 | 0.0016 | **0.0065** | 0.60 | ❌ 未超过噪声 |
| C→M | HyperHead − ERM | **−0.0022** | [−0.0028, −0.0017] | **显著劣** | 0.0022 | **0.0051** | −1.31 | ❌ 未超过噪声 |
| C→M | HyperFusion − ERM | **−0.0015** | [−0.0019, −0.0011] | **显著劣** | 0.0015 | **0.0035** | −1.18 | ❌ 未超过噪声 |

**变号模式与 HyperAdapt 同形**：M→C 微正、C→M 显著劣。主文件 §3 已证 HyperAdapt 的这一变号由
trial 扩充（训练随机性）造成而非口径；两个新臂独立重现同一模式，**支持「M→C 的微正读数是方向-特异
的偶然配合，而非 HN 的稳定属性」这一解读**。

### 2.3 Overall AUC（次要指标）

| 方向 | ERM | SWAD | GroupDRO | **HyperHead** | **HyperFusion** | HyperAdapt |
|---|---|---|---|---|---|---|
| M→C | 0.8505 | 0.8477 | 0.8428 | **0.8505** | **0.8507** | 0.8522 |
| C→M | 0.8192 | 0.8200 | 0.8136 | **0.8170** | **0.8180** | 0.8170 |

| 方向 | 比较 | Δ | 95% CI | H1 | \|d̄\| vs σ_train | H2 |
|---|---|---|---|---|---|---|
| M→C | HyperHead − ERM | −0.0000 | [−0.0007, +0.0006] | n.s. | 0.0000 vs 0.0071 | ❌ |
| M→C | HyperFusion − ERM | +0.0002 | [−0.0004, +0.0008] | n.s. | 0.0002 vs 0.0056 | ❌ |
| C→M | HyperHead − ERM | −0.0022 | [−0.0025, −0.0018] | 显著劣 | 0.0022 vs 0.0049 | ❌ |
| C→M | HyperFusion − ERM | −0.0012 | [−0.0014, −0.0009] | 显著劣 | 0.0012 vs 0.0032 | ❌ |

M→C 的 Overall 是**精确打平**（Δ ≈ 0.0000 / +0.0002，均 n.s.）——即 M→C 那点 marginal-worst 微正
**没有伴随任何整体判别力变化**。

### 2.4 Holm 校正（family 由 6 扩为 10）

偏离 #4 事先声明「既有 6 项须按新 family 重新校正，若失显著须如实更新」。**实测无需更新**：

| 检验 | p | Holm 后（family=10） |
|---|---|---|
| C2M:groupdro − erm | 0 | 保持显著 |
| M2C:groupdro − erm | 2.2e−16 | 保持显著 |
| **C2M:hyperfusion − erm** | **2.2e−16** | **显著（劣）** |
| C2M:hyperadapt − erm | 2.2e−15 | 保持显著 |
| **C2M:hyperhead − erm** | **1.2e−14** | **显著（劣）** |
| M2C:swad − erm | 1.1e−11 | 保持显著 |
| M2C:hyperadapt − erm | 3.3e−07 | 保持显著 |
| **M2C:hyperfusion − erm** | **0.0115** | **显著（正，但未超噪声）** |
| C2M:swad − erm | 0.0184 | 保持显著 |
| **M2C:hyperhead − erm** | **0.1475** | **不显著** |

**回归核验**：既有四臂的 12 个点估计与 6 个 Δ 值（marginal-worst 与 Overall）与主文件 §2、GroupDRO
文件 §2 **逐位一致**（脚本级 bit-for-bit 比对，见 §6）⇒ 加入两臂未扰动既有口径。

---

## 3. 探索性：条件化深度轴（**不进 Holm family，不作显著性声明，不作因果声明**）

偏离 #4 解除了 §1 的深度禁令，但把深度问题限定为**描述性**。属性介入深度递增序列为
**HyperHead（仅分类头）< HyperFusion（单个 block 的 downsample）< HyperAdapt（除 stem 外所有层）**。

| 指标 / 方向 | Head | Fusion | Adapt | 是否随深度单调 |
|---|---|---|---|---|
| marginal-worst，M→C | +0.0010 | +0.0016 | +0.0031 | **✅ 单调 ↑** |
| marginal-worst，C→M | −0.0022 | −0.0015 | −0.0025 | ❌ 不单调（Fusion 居中最优） |
| Overall，M→C | −0.0000 | +0.0002 | +0.0017 | ✅ 单调 ↑ |
| Overall，C→M | −0.0022 | −0.0012 | −0.0022 | ❌ 不单调 |

**读法（务必按此限定）**：M→C 上两个指标都恰好随深度单调递增，看起来很像「介入越深越好」；但

1. **四个格子的全部 12 个 Δ 无一超过训练噪声**（σ_train ≈ 0.0032–0.0089，Δ 最大 0.0031）
   ⇒ 这条「单调」落在噪声带内部，**不构成趋势证据**；
2. **反方向 C→M 直接不单调**，若深度真是驱动因素，不应只在一个方向成立；
3. 三臂**不是受控的深度阶梯**——参数量（11.2M / 19.7M / 含低秩生成器）、调制形式（生成分类头权重 /
   单 block 加性 HyperFusion / 卷积乘性 + FC 加性低秩）、初始化与选中的 wd 都同时不同
   （`docs/conditioning_ablation_summary.md` §1）。受控的深度轴由实验 E 的 `ResNet18CondNet` 负责，
   其结论（H1/H2 均 n.s.）与本处的「噪声带内」读数**一致**。

⇒ **结论仍是「深度非决定因素」**，本节只是在 OOD 侧补上了此前缺席的两格，使该判决在 ID 与 OOD
两个 regime 上同构。

---

## 4. 校准（D2，描述性；偏离 #4 明确不为两臂新增 H3 型判定）

**点估计**（full-target，15 replicate，原样迁移）：

| 方向 | 方法 | CITL | ECE | Brier | slope |
|---|---|---|---|---|---|
| M→C | ERM | +0.1069 | 0.1069 | 0.0780 | 1.094 |
| M→C | **HyperHead** | +0.1192 | 0.1192 | 0.0842 | 1.031 |
| M→C | **HyperFusion** | +0.1145 | 0.1145 | 0.0822 | 1.024 |
| C→M | ERM | −0.1601 | 0.1601 | 0.1826 | 0.718 |
| C→M | **HyperHead** | −0.1649 | 0.1649 | 0.1863 | 0.714 |
| C→M | **HyperFusion** | −0.1673 | 0.1673 | 0.1868 | 0.700 |

**Δ = 该臂 − ERM**（Δ<0 = 校准更好；全部为患者级配对 bootstrap 显著，但**全部未超训练噪声**）：

| 方向 | 方法 | Δ\|CITL\| | ΔECE | ΔBrier |
|---|---|---|---|---|
| M→C | HyperHead | **+0.0124** | +0.0124 | +0.0062 |
| M→C | HyperFusion | **+0.0076** | +0.0076 | +0.0042 |
| C→M | HyperHead | **+0.0048** | +0.0048 | +0.0037 |
| C→M | HyperFusion | **+0.0072** | +0.0072 | +0.0042 |

🛑 **两臂与 HyperAdapt / GroupDRO 的模式不同：它们在两个方向、三个指标上全部劣于 ERM，没有任何
一个方向占优。** 主文件偏离 #2 与 GroupDRO 文件 §3 观察到的是「一个方向更差、另一个更好」，据此
提出「方向性校准差异更像方法与偏移方向的偶然配合」；HyperHead/HyperFusion 提供的是**更强的版本**
——属性条件化在这两种实现下对跨库校准是**单向的净损失**。但四个格子**全部未超训练噪声**，故此处
只作描述，不升格为判定。

参照：SWAD 在两个方向的 |CITL|/ECE 均显著优于 ERM（−0.0079 / −0.0119），是六臂中唯一在校准上
双向占优者——与它作为域泛化锚点的定位一致。

---

## 5. 执行记录

| 阶段 | 内容 | 作业 |
|---|---|---|
| 1 | trial 1/2 训练，40 run（2 方法 × 2 库 × 5 fold × 2 trial） | **77963**（MIMIC-HH，task 0–3）+ **77981–77984** |
| 3 | 双向 full-target 终评，60 predictions（M→C trial 0 的 10 个复用） | **78098**（HyperHead）/ **78099**（HyperFusion） |

**config 与 batch**：均取各自 cv5 S1 选中 config；**batch 不覆盖**（核对 trial 0 的 cv 搜索日志为
`BATCH=config`，即 128），等-batch 条件与 trial 0 逐项一致。

### 5.1 ⚠️ 硬件异质性披露

本次 trial 1/2 的绝大部分 run 跑在 `gpus` 分区的 **Tesla P40（Pascal, sm_61）** 上，而既有四臂的
trial 1/2 跑在 `gpus24`/`gpus48` 的 RTX 3090/4090/A6000 上。提交时 `gpus24`/`gpus48` 被其他用户占满，
`gpus` 分区空闲。差异来源为 fp32 舍入与 Ampere 上 cuDNN 的 TF32 卷积路径，**属训练噪声范畴**
（种子、config、batch、数据划分全部固定），且项目既有各臂本就跨 mira01/04/09/10/deepmedic2/lora
多种卡型混跑。**但这是本次新引入的臂间差异，如实记录，不宣称其无害。** 若需排除，可在同一卡型上
重跑两臂的 15 replicate 复核。

### 5.2 节点故障记录（工程）

- **monal05**（GTX 1080 Ti）：预检探测通过，实跑时 5 个 task 全部 `CUDA error: unknown error` 崩溃，
  其后落到该节点的 task 静默降级为 `Device: cpu`。**间歇性故障，已加入排除列表。**
- **lory**（`gpus` 分区 16 卡）：`Failed to get device handle for GPU 0`，`torch.cuda.is_available()`
  返回 False，与 semois 同款。**已加入排除列表。**
- **mira05**：复现已知的 CUDA 静默降 CPU。
- **处置**：给 `slurm/ood_trial.sh` 补了 **CUDA 预检**（不可用即非零退出）。推理侧
  `run_ood_cxr_full_target.py` 早有同款守卫，训练侧此前没有——正是它让坏节点上的 run 静默烧机时
  而日志表面无异常。**此后所有训练作业受同一守卫保护。**

---

## 6. 产物与复现

| 内容 | 路径 |
|---|---|
| 预测（60 个 npz，含 patient_id） | `outputs/ood_cxr/{mimic2chexpert,chexpert2mimic}/cv5_full_target/predictions/{hyperhead,hyperfusion}_*` |
| 主分析（marginal-worst / overall） | `outputs/ood_cxr/variance_decomposition_hn6/` |
| 校准（D2） | `outputs/ood_cxr/calibration_hn6/` |

```bash
# 主分析（family = 10；--methods 直接决定 family 大小）
python scripts/ood_variance_decomposition.py --metric marginal_worst \
    --out-dir outputs/ood_cxr/variance_decomposition_hn6 \
    --methods erm,swad,hyperadapt,groupdro,hyperhead,hyperfusion
python scripts/ood_variance_decomposition.py --metric overall --out-dir ... --methods ...
# 校准（两臂进 CONTRASTS 但不进 HOLM_KEYS ⇒ 实验 G 的 8 检验 family 不变）
python scripts/g_ood_calibration_full_target.py --out-dir outputs/ood_cxr/calibration_hn6
```

**回归核验**（已执行，bit-for-bit）：既有臂 `erm/swad/hyperadapt/hyperadapt_swad/groupdro` 的
marginal-worst、Overall 点估计与 Δ，以及 CITL/ECE/Brier/slope 点估计与 Δ，在加入两臂前后
**逐位一致**（对比 `variance_decomposition_gdro/` 与 `calibration_g_full_target/`）。
