# PAPILA 预处理方案（MEDFAIR 对齐 + 本项目惯例）

> 任务：眼底图青光眼二分类，敏感属性 = 性别 + 年龄；用于测试超网络（HyperNetwork）对公平性的提升。
> 本方案对齐 MEDFAIR（Zong et al., ICLR 2023）对 PAPILA 的处理，并沿用本项目已有的
> HAM10000 / Fitzpatrick17k / MIMIC-CXR 三套预处理惯例（`build_*_splits.py` → split CSV →
> `*_dataset.py` → 条件互信息筛查 → baseline 训练）。

---

## 0. 数据实测核对（已在 `/vol/biodata/data/PAPILA` 上验证，与 MEDFAIR 完全吻合）

| 项 | 实测值 | MEDFAIR (Table A3 / B.1.2) |
|----|--------|----------------------------|
| 原始眼-图总数 | **488**（244 患者 × 双眼 OD/OS） | — |
| 排除 suspect 后 | **420 眼-图 / 210 患者**，阳性(青光眼)=87 | 420 images / 210 patients |
| 全 488 诊断分布 | 0 健康=333 / 1 青光眼=87 / 2 可疑=68 | — |
| 缺失敏感属性 | **0 行**（Age/Gender 全有） | 「removing those missing」后 420 |
| 性别 Gender=0(男) | 146/420 = 34.76%，青光眼率 0.240 | Male 34.76% (23.97% unhealthy) |
| 性别 Gender=1(女) | 274/420 = 65.24%，青光眼率 0.190 | Female 65.24% (18.98%) |
| 年龄 0–60 组 | 青光眼率 ≈0.065 | Age Group 0: 8.43% unhealthy |
| 年龄 60+ 组 | 青光眼率 ≈0.304 | Age Group 1: 29.75% unhealthy |

**关键事实**：被排除的 34 个患者恰好是「双眼均为 suspect」者（68 = 34×2），剩余 210 人双眼均非
suspect（420 = 210×2）。这正是 MEDFAIR 的 420/210 来源——可作为脚本硬断言锚点。

**图像规格**：`FundusImages/RET<3位ID><OD|OS>.jpg`，**2576×1934 RGB JPEG (uint8, 0–255)**，
普通自然图像。**不需要 CXR 那套 16-bit min-max 处理**（与 HAM10000 / Fitzpatrick 同，见仓库
`CLAUDE.md` 与记忆 [[cxr7-1m-png-is-16bit]] 的反面：PAPILA 走自然图像通路）。

**临床表读取坑**：`ClinicalData/patient_data_od.xlsx`(右眼) 与 `patient_data_os.xlsx`(左眼)
是 **xlsx**；合并表头占前 2 行，第 3 行为 "ID"，数据从第 4 行起；A 列(ID)无表头名。
`medimg` 环境已补装 **openpyxl 3.1.5**，`pd.read_excel(..., header=1)` 可直接用
（用 `df.rename(columns={df.columns[0]:'ID'})` 命名 ID 列）。列顺序：
`ID, Age, Gender, Diagnosis, dioptre_1, dioptre_2, astigmatism, Phakic/Pseudophakic,
Pneumatic(IOP), Perkins, Pachymetry, Axial_Length, VF_MD`。

---

## 1. 任务与标签（对齐 MEDFAIR）

- **目标 Y**：青光眼二分类。`label = 1` ⇔ `Diagnosis==1`(glaucomatous)，`label = 0` ⇔
  `Diagnosis==0`(non-glaucomatous)。
- **排除 suspect**：`Diagnosis==2`（68 眼-图）整体剔除——与 MEDFAIR「exclude the suspect
  label class」一致。剔除后 **420 眼-图 / 210 患者，阳性率 20.7%**。
- 保留原始 `Diagnosis` 列入 CSV 备查；若后期想做 3 类或「suspect 并入阴性」的消融，可在上层重映射。

---

## 2. 样本筛选与防泄漏划分（本项目惯例 = 按患者分组）

- **建模单元 = 单眼图**（OD、OS 各算一个样本），共 420。
- **无需丢缺失**：Age/Gender 零缺失（不同于 HAM 要丢 67 行）。
- **划分：患者级 `GroupShuffleSplit`，比例 70/10/20**（`groups=pid`，`RANDOM_STATE=42`）。
  - 比例**用 MEDFAIR 的 70/10/20**（注意：这**不同于** HAM/Fitz 的 80/10/10，因为 MEDFAIR
    对 PAPILA 明确用 70/10/20）。
  - 同一患者双眼**绝不跨 split**——与 HAM「按 `lesion_id` 分组」、MIMIC/CheXpert「同病人不跨
    split」同一原则（左右眼高度相关，图像级随机划分会泄漏、虚高指标）。
  - 7 个患者左右眼标签不一致：按眼保留各自标签，两眼仍随患者同进一个 split。

> ⚠️ **小样本警告（贯穿全方案）**：test ≈ 20%×420 ≈ **84 眼-图 / 42 患者**。这是本项目所有
> 数据集里**最小的**（HAM test≈995 已触发 MDE 警告）。**强烈建议**对 PAPILA 改用
> **患者级 K 折交叉验证（GroupKFold, 患者为组，建议 5 折）** 取代单次 70/10/20，
> 以稳定训练与（尤其）条件互信息估计的统计功效。单次划分仅用于和 MEDFAIR 数字对表。

---

## 3. 敏感属性编码（双属性两轴；无种族/肤色）

| 属性 | 编码 | 备注 |
|------|------|------|
| `sex` | Gender：**0=男, 1=女** | 已由青光眼率与 MEDFAIR 对齐验证 |
| `age_group` | **0 = Age ≤ 60，1 = Age > 60** | 对齐 MEDFAIR「0–60 / 60+」（边界 60 归入组 0，使组0=178 与 MEDFAIR 42.38% 吻合）；原始 `age` 连续值同时入 CSV |
| `joint` | `sex*2 + age_group` ∈ {0,1,2,3} | 4 个交叉子群，供 WeightedRandomSampler 与分组评估 |

- PAPILA **无种族/肤色标注**，公平性分组只能用 sex 与 age（与 HAM10000 同构）。
- **模型接口适配**：现有属性融合模型 forward 签名为 `(image, sex, race)`。迁到 PAPILA 时
  **race 槽位填 `age_group`**（与 HAM 把第二轴用 age_group 的做法一致），即实际跑
  `(image, sex, age_group)`。baseline（image-only）不受影响。
- **强弱轴预判**：年龄是强轴（青光眼率 0–60:6.5% vs 60+:30.4%，差距巨大），性别弱
  （24.0% vs 19.0%）。这与记忆 [[ham10000-conditional-mi-age-signal]]「age 有
  I(Y;A|X)>0」的格局类似，age 轴最有希望让 HN 兑现公平收益。

---

## 4. 图像变换（对齐 MEDFAIR 2D 管线）

自然 RGB，`Image.open(path).convert("RGB")`，**禁用任何 16-bit 处理**。

- **train**：`Resize((256,256))` → `RandomCrop(224)` → `RandomHorizontalFlip()` →
  `RandomRotation(15)` → `ToTensor` → ImageNet `Normalize(mean=[.485,.456,.406],
  std=[.229,.224,.225])`。
- **val/test**：`Resize((256,256))` → `CenterCrop(224)` → `ToTensor` → ImageNet Normalize。
- 与 HAM 同属「源图 ≠224 需先 Resize 降采样」一类（区别于 Fitzpatrick 的 224 原生）。
- ⚠️ 眼底特有提示：水平翻转会交换 OD/OS 的左右外观，但视杯/视盘形态对青光眼判别基本左右无偏，
  且 MEDFAIR 统一用 hflip——**保留 hflip 以与 MEDFAIR 可比**，仅在此标注。
- **I/O 优化（推荐）**：原图 2576×1934、共 420 张、约 1.2 GB，每 epoch 解码偏重。建议预降采样到
  256×256 写入个人目录 **`data/processed/papila/fundus_256/RET***.jpg`**（禁止改动只读原始数据），
  split CSV 的 `image_path` 指向该缓存；或沿用 `UTKFaceCSVDataset(cache=True)` 的内存缓存
  （420×256×256×3 ≈ 82 MB，可整库常驻）。

---

## 5. 类别 / 子群不均衡处理（本项目重采样试验场）

- 阳性率 20.7%（类不均衡）+ 子群不均衡（age 轴尤甚）。
- 沿用项目 **WeightedRandomSampler** 惯例：按 **4 个 `joint` 子群**（或 `joint×label` 8 格）
  反频率加权采样，`alpha∈{1.0,0.5}` 调温（同记忆 [[mimic-resampling-plan]] 的十六格做法，
  PAPILA 缩为四/八格）。baseline 先不带 sampler，再加 1.0 对照。

---

## 6. 条件互信息筛查（HN 的必要条件闸门，**先于任何 HN**）

移植 `scripts/estimate_conditional_mi_ham10000.py` 为 `estimate_conditional_mi_papila.py`，
保留 E0–E3 四层设计、估计器与决策规则。判据（见记忆 [[mimic-fairness-signal-reframing]]）：
**HN 要赢 attribute-blind 池化基线，必要条件是 I(Y;A|X) > 0**；若 ≈0 则盲基线已是 Bayes 最优，
所有 HN 变体注定无收益。

- A ∈ {`sex`(2 类，2×2 CMH)、`age_group`(2 类)、`joint`(4 类，多类 softmax + 方向无关 χ²)}。
- backbone 用 ImageNet 预训练 ResNet-18 多 seed 提特征（见 §7）。
- ⚠️ **E3 在 PAPILA 上从「健康性检查」升级为硬约束**：test n≈84 远小于 HAM(≈995)，最小可检测
  效应(MDE)会很高。**务必先跑干净正控制**；若正控制都检不出，只能下「此样本规模下不可判定」，
  **不得误读为「信号弱」**（这是 [[fitzpatrick-conditional-mi-weak]] 的型 VI 假阳性教训）。
  → 因此本步**强制配合 §2 的患者级 K 折**，把多折结果合并以换取功效。

---

## 7. Backbone 与训练（小样本防过拟合）

- **Backbone：ImageNet 预训练 ResNet-18**（对齐 MEDFAIR）。PAPILA n=420 极小，**不建议**从零训
  PreactivResNet18——MEDFAIR 明确用「轻量 + 预训练」防过拟合。
- BCE 损失；早停监控 **val worst-case AUC**（MEDFAIR 结论：模型选择策略显著影响最差子群表现，
  按 worst-case/Pareto 选模型比按 overall 更公平且几乎不损 overall）。
- 多 seed（42/43/44）取均值±方差，对齐项目其它数据集。
- 集群训练：`num_workers=4–8`、`pin_memory=True`；数据小，单卡 `gpus24` 足够。

---

## 8. 评估（MEDFAIR 指标 + 项目 `fairness_metrics.py`）

- **效用**：overall AUC。
- **群体公平**：子群 AUC（按 sex / age_group / joint）、最大-最小 **AUC gap**。
- **Max-Min 公平**：**worst-case 子群 AUC**。
- **EqOdd**（二值属性）。
- 复用 `src/fairness_metrics.py`（把 `print_fairness_report` 的 race 实参传 `age_group`）。
- 模型选择同时报 overall 选模型与 worst-case 选模型两套，便于和 MEDFAIR 对照。

---

## 9. 产出物（与项目目录职责一致）

| 文件 | 作用 |
|------|------|
| `scripts/build_papila_splits.py` | 读 xlsx → 排除 suspect → 编码 sex/age_group → 患者级 GroupShuffleSplit(70/10/20) / 可选 GroupKFold → 写 split CSV。**硬断言：420 眼-图 / 210 患者 / 阳性 87**（对齐 MEDFAIR） |
| `data/splits/papila/{train,val,test}.csv` | 字段：`image_path, label, sex, age, age_group, joint, pid, eye, diagnosis` |
| `src/datasets/papila_dataset.py` | `PAPILADataset`，`__getitem__ → (image, label, sex, age_group)`；自然 RGB convert；支持 `cache=True` |
| `configs/papila_baseline.yaml` | §4 变换 + §7 训练超参 |
| `scripts/estimate_conditional_mi_papila.py` | §6 的 I(Y;A|X) 四层筛查（K 折合并） |
| `scripts/preview_papila.py`（可选） | 数据概览/抽样可视化 |
| `data/processed/papila/fundus_256/`（可选） | 256×256 预降采样缓存 |

---

## 10. 与本项目三套已有方案的差异速查

---

## 11. 执行状态与 baseline 结果（2026-06-24）

**已落地**：`scripts/build_papila_splits.py`（单次 70/10/20 + `--cv 5` 患者级 GroupKFold）、
`src/datasets/papila_dataset.py`、`src/utils/papila_fairness.py`、
`src/training/train_papila_resnet18.py`（`--cv/--fold`）、`configs/papila_baseline.yaml`、
`slurm/train_papila_resnet18.sh`。256 缓存 420 张已生成于 `data/processed/papila/fundus_256/`。

**baseline（ImageNet 预训练 ResNet-18，image-only，seed 42，5 折 GroupKFold 测试集）**：

| 指标 | 5 折均值 ± std |
|------|----------------|
| Overall AUC | **0.828 ± 0.045** |
| Worst-case AUC（Sex/Age/Sex×Age minimax） | 0.638 ± 0.114 |
| Sex AUC gap | 0.111 |
| Age AUC gap | 0.143 |

- Overall AUC 0.83 略高于 MEDFAIR ERM（Table A8 PAPILA ~0.65–0.81），管线 sanity 通过。
- **worst-case AUC 方差极大（±0.11，fold0 甚至 0.50）**——Sex×Age 最小子群仅 6 眼，再次印证
  小样本必须看 5 折合并、单折不可信。**age gap > sex gap**，与 age 为强轴一致。
- overall 选模型与 worst-case 选模型结果几乎相同（val 仅 44 眼，worst-case 选择信号太弱）。
- 下一步：`estimate_conditional_mi_papila.py` 验 I(Y;A|X)（尤其 age 轴）→ 决定是否上 HN。

---

## 12. 弱信号验证结果 I(Y;A|X)（OOF 池化 5 折，N=420，2026-06-24）

脚本 `scripts/estimate_conditional_mi_papila.py`（HAM 版平移 + **5 折 OOF 池化**抢小样本功效）。
结果 `outputs/papila/conditional_mi.json`。判据见记忆 [[mimic-fairness-signal-reframing]]。

**E3 探针有效性（前提）**：正控制 Y XOR r 的 ΔNLL=0.305，CI[0.215,0.392]，detected=True
→ 探针对纯条件信号有功效，读数可信。但剂量-反应 **MDE≈0.05 nats 且抖动**（α=0.05/0.2/0.4 检出、
α=0.1 反而未检出）——n=420 下 E1 对 <0.05 nats 的小效应基本无力。

**E1 主估计器（int 模型 ΔNLL，全部 CI 含 0）**：
| 属性 | ΔNLL (nats) | 95% CI | 判读 |
|------|-------------|--------|------|
| sex | −0.013 | [−0.037, +0.011] | ≈0 |
| age | +0.011 | [−0.013, +0.038] | ≈0（**点估计 < MDE，无力分辨**） |
| joint | +0.008 | [−0.042, +0.055] | ≈0 |

**E0 可解码度**：age 从眼底图**高度可解码**（I(age;X)≥0.161 bits，macro-AUC 0.774），sex 不可
解码（AUC 0.498）。→ 图像已吸收大量 age 信息，残余可被 HN 利用的条件信号被压向 0。

**E2 非参对照（关键分歧）**：在 image-only 预后分数 ŝ 分层后查 A–Y 残余关联——
| 属性 | agg χ²(df) p | CMH_p | MH-OR | 判读 |
|------|--------------|-------|-------|------|
| sex | 0.43 | 0.22 | 0.67 | 无关联，与 E1 一致 |
| **age** | **0.0014** | **0.00014** | **3.61** | **显著！同分数箱内 >60 患病几率约 3.6×** |
| joint | 0.0038 | — | — | 显著（age 驱动） |

**结论（诚实版）**：
- **sex**：E1≈0 且 E2 不显著 → 信号弱，**HN 在 sex 轴注定无收益**（与 MIMIC/Fitz 同）。
- **age**：**矛盾信号**。E2 非参检验给出强条件关联（标量 ŝ 之上 age 仍显著，MH-OR 3.6），
  但 E1 在全特征 φ(X) 上 ≈0——因为 age 已被眼底图高度编码（E0），φ(X) 几乎吸收了 age-相关风险，
  留给 HN 的残余条件信号很小；加之 n=420 使 E1 的 MDE≈0.05 > age 点估计 0.011，**无力判定**
  是否残留可利用信号。这区别于 HAM10000（[[ham10000-conditional-mi-age-signal]] age 有可检测
  I(Y;age|X)>0、深层 HN 兑现收益）：**PAPILA age 更像「被图像吸收 + 样本不足」的不可判定区**。
  🛑 **2026-07-19 反转**：新 OOF conditional V-information 闸门下，**PAPILA-age 反成全研究唯一 detected 的真实轴**
  （+0.041 bits，CI>0），而 **HAM-age 退回未检出**——本段的对比关系已颠倒，见 `docs/conditional_v_information_gate.md`。
- **决策**：sex 轴放弃 HN。age 轴是唯一候选但证据弱于 HAM，可低成本试一把 age-条件 HN
  作为验证，但需管理预期——baseline overall AUC 0.83 已不低，HN 大概率打平而非超过。

---

| 维度 | PAPILA | HAM10000 | Fitzpatrick17k | MIMIC-CXR |
|------|--------|----------|----------------|-----------|
| 模态 | 眼底 RGB | 皮肤镜 RGB | 皮肤镜 RGB | 胸片 16-bit |
| 源分辨率 | 2576×1934(需 Resize) | 600×450(需 Resize) | 224 原生 | 512/224 |
| 划分单位/分组键 | 患者 `pid`(双眼) | 病灶 `lesion_id` | 图像级随机 | 病人 `subject_id` |
| 划分比例 | **70/10/20** | 80/10/10 | 80/10/10 | 80/10/10 |
| 敏感属性 | sex + age(2 箱) | sex + age(4 箱) | 肤色(6) | sex/race/age |
| 样本量 | **420（最小，建议 K 折）** | 9948 | 16012 | ~370k |
| 16-bit 处理 | 否 | 否 | 否 | **是** |
| 对账锚点 | 420/210/pos87 | 9948 | 16012 | — |
