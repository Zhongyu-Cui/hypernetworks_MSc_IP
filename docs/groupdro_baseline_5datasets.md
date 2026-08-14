# GroupDRO 基线 —— 五数据集全覆盖（ID，CV-OOF regime）

> 状态：**全部完成**（2026-08-11）。作业 **75498**(HAM) / **75529**(MIMIC) / **75530**(CheXpert) /
> **75531**(Fitzpatrick) / **75532**(PAPILA)，各 6 配置 × 5 折 = 30 run，**150/150 成功、0 错误**。
> HAM 的实现细节与逐子群深挖见 `docs/groupdro_baseline_ham.md`；本文件是**五库汇总与结论**。
> 口径与既有 OOF regime 完全一致（折均 marginal-worst 选 config → averaging → 病灶/患者级配对
> cluster bootstrap），未新增任何口径。

---

## 1. 一句话结论

**GroupDRO 在五个数据集上没有任何一处显著改善 worst-group；在两个胸片库上对所有指标显著变差。
三个 HN 相对 GroupDRO 从不更差，在 MIMIC/CheXpert 上全指标显著更优。** ⇒ 「HN 的表现是否只是
因为它看到了敏感属性」这一混淆被决定性排除：**同样在训练时使用子群标签的标准方法并没有更好。**

## 2. 实现与设置（权威见 `harness/groupdro.py`）

- 算法：Sagawa et al. (ICLR 2020) online greedy GroupDRO，`q_g ← q_g·exp(η·L_g)` 归一化后
  `robust_loss = Σ q_g L_g`；与官方 `LossComputer` 逐行对齐。
- **分组 = 评估侧 canonical 交叉子群**（复用 `subgroup_masks`）⇒ 优化的组 = 报告 worst-group 的组。
- **组均匀采样默认开**（论文 Algorithm 1 的 `g ~ Uniform`，官方 `--reweight_groups`），
  复用 `resampling.build_group_label_sampler(alpha=1)`。
- η=0.01（论文默认）、C=0，**不进超参网格**；lr/wd 走与其它方法**同一** 6 配置网格。
- 入口：HAM 用 `train_ham10000_groupdro.py`（多一步 `age_group>=0` 过滤）；其余四库用数据集
  参数化的 `train_groupdro.py`（从各库 ERM 模块借 transforms/loaders/公平性回调，
  **"与 ERM 只差损失+采样"由构造保证**）。

| 数据集 | 组 | 训练 n（fold0） | 最小组 | 选定 config |
|---|---|---|---|---|
| HAM10000 | Sex×Age 8 | 6,807 | Female\|80+ 133 | lr3e-04_wd1e-03 |
| MIMIC | Sex×Race×Age 8 | 139,270 | M\|Non-White\|<60 6,231 | lr3e-04_wd1e-04 |
| CheXpert | Sex×Race×Age 8 | 96,874 | F\|Non-White\|<60 4,499 | lr1e-04_wd1e-03 |
| Fitzpatrick | Skin 6 | 11,528 | 型VI 458 | lr1e-04_wd1e-04 |
| PAPILA | Sex×Age 4 | 292 | Male\|≤60 48 | lr3e-04_wd1e-03 |

## 3. 结果

### 3.1 主表（averaging 口径）

| 数据集 | 方法 | Overall | canon worst | marg worst | gap |
|---|---|---|---|---|---|
| **HAM10000** (n=9,707) | ERM | 0.8938 | 0.7877 | 0.7976 | 0.1348 |
| | SWAD | **0.9129** | **0.8331** | **0.8433** | 0.1024 |
| | **GroupDRO** | 0.8813 | 0.7816 | 0.8128 | 0.1237 |
| | HyperFusion | 0.8996 | 0.8171 | 0.8310 | **0.0923** |
| **MIMIC** (n=199,356) | ERM | 0.8456 | 0.8089 | 0.8178 | 0.0634 |
| | SWAD | 0.8474 | 0.8096 | 0.8194 | 0.0627 |
| | **GroupDRO** | **0.8377** | **0.7995** | **0.8096** | 0.0648 |
| | HyperAdapt | **0.8461** | **0.8090** | 0.8192 | 0.0630 |
| **CheXpert** (n=138,644) | ERM | 0.8701 | 0.8316 | 0.8377 | 0.0516 |
| | SWAD | 0.8699 | 0.8317 | 0.8354 | 0.0551 |
| | **GroupDRO** | **0.8565** | **0.8115** | **0.8215** | 0.0637 |
| | HyperFusion | **0.8712** | **0.8378** | **0.8406** | 0.0477 |
| **Fitzpatrick** (n=16,012) | ERM | 0.8962 | 0.8161 | 0.8161 | 0.0904 |
| | SWAD | **0.9205** | **0.8739** | **0.8739** | 0.0578 |
| | **GroupDRO** | 0.8934 | 0.8583 | 0.8583 | **0.0551** |
| | HyperHead | 0.9114 | 0.8427 | 0.8427 | 0.0915 |
| **PAPILA** (n=420) | ERM | 0.8448 | **0.8245** | 0.8277 | 0.0621 |
| | SWAD | 0.8592 | 0.8280 | 0.8280 | 0.1357 |
| | **GroupDRO** | 0.8460 | 0.7875 | 0.8072 | 0.1059 |
| | HyperAdapt | **0.8701** | 0.6250 | 0.8155 | 0.3227 |

（完整 7 方法 × 5 库见 `outputs/conditioning_ablation/oof_results_averaging.json`。）

### 3.2 GroupDRO 的配对比较（Δ [95% CI]，`*` = 显著）

| 数据集 | 比较 | ΔOverall | Δcanon worst | Δmarg worst |
|---|---|---|---|---|
| HAM | GroupDRO − ERM | **−0.0124**[−0.022,−0.003]* | −0.0061[−0.122,+0.079] | +0.0152[−0.041,+0.053] |
| MIMIC | GroupDRO − ERM | **−0.0079**[−0.009,−0.007]* | **−0.0094**[−0.012,−0.007]* | **−0.0082**[−0.010,−0.007]* |
| CheXpert | GroupDRO − ERM | **−0.0136**[−0.015,−0.012]* | **−0.0201**[−0.032,−0.008]* | **−0.0162**[−0.020,−0.013]* |
| Fitzpatrick | GroupDRO − ERM | −0.0028[−0.009,+0.005] | **+0.0422**[−0.026,+0.104] | **+0.0422**[−0.026,+0.104] |
| PAPILA | GroupDRO − ERM | +0.0012[−0.048,+0.054] | −0.0370[−0.144,+0.058] | −0.0205[−0.097,+0.059] |

**五库 worst-group 无一显著为正**；Fitzpatrick 的 +0.042 是最大的正效应但 CI 含 0；
MIMIC/CheXpert 三项指标**全部显著为负**。

### 3.3 HN vs GroupDRO（同样吃属性，只差 conditioning vs reweighting）

| 数据集 | ΔOverall（3 HN 范围） | Δcanon worst | 判读 |
|---|---|---|---|
| HAM | +0.010 ~ **+0.018*** | −0.010 ~ +0.035（全 n.s.） | HN Overall 更优 |
| MIMIC | **+0.0075 ~ +0.0084***（全显著） | **+0.0088 ~ +0.0095***（全显著） | **HN 全指标显著更优** |
| CheXpert | **+0.0100 ~ +0.0147***（全显著） | +0.0207 ~ **+0.0262*** | **HN 全指标显著/近显著更优** |
| Fitzpatrick | +0.000 ~ **+0.018*** | −0.042 ~ −0.016（全 n.s.） | Overall HN 优、worst 打平 |
| PAPILA | −0.000 ~ +0.024（n.s.） | −0.163 ~ −0.080（n.s., 极噪） | 打平（n=420 无功效） |

**HN 在任何数据集的任何指标上都没有显著输给 GroupDRO。**

### 3.4 机制：逐子群 AUC 揭示两种截然不同的行为

**(a) 组间结构存在时 → 真的重分配（Fitzpatrick，肤色 VI 是天然稀缺组）**

| skin | I | II | III | IV | V | **VI (n=458)** | gap |
|---|---|---|---|---|---|---|---|
| ERM | 0.8898 | 0.8906 | 0.9060 | 0.9050 | 0.9065 | **0.8161** | 0.0904 |
| GroupDRO | 0.8832 | 0.8888 | 0.8992 | 0.8964 | 0.9134 | **0.8583** | **0.0551** |

多数肤色型全部小幅下降、最弱的型 VI **+0.042**，gap 收窄到 0.0551（**全方法最低，低于 SWAD 的 0.0578**）。
这是教科书式的 GroupDRO 行为——只是幅度不足以显著。

**(b) 组间无可利用结构时 → 全面均匀劣化（MIMIC，14 个子群无一例外）**

| | Male | Female | White | Non-White | <60 | ≥60 | M\|W\|≥60 | F\|W\|≥60 |
|---|---|---|---|---|---|---|---|---|
| ERM | 0.8359 | 0.8552 | 0.8411 | 0.8533 | 0.8644 | 0.8178 | 0.8089 | 0.8233 |
| GroupDRO | 0.8269 | 0.8486 | 0.8325 | 0.8479 | 0.8563 | 0.8096 | 0.7995 | 0.8155 |
| Δ | −0.009 | −0.007 | −0.009 | −0.005 | −0.008 | −0.008 | −0.009 | −0.008 |

**没有任何重分配**，14 个子群一致下降约 0.008——这正是 `I(Y;A|X)≈0` 体制的指纹：组之间不存在
可交易的差异化结构，minimax 重加权只能付出优化代价、买不到任何东西。

**(c) HAM：抬地板变挪地板**（详见 HAM 专文）——把 ERM 最弱格 Male|80+ 0.788→0.808 抬起，
但 Male|20-40 从 0.837 跌到 0.782 接管地板，canonical worst 净变化 −0.006。

### 3.5 对抗权重 q 的动力学（选定 config，5 折训练末）

| 数据集 | q 范围（均匀值） | 倍差 | 判读 |
|---|---|---|---|
| HAM | [0.072, 0.230] (0.125) | ~3× | 活跃 |
| MIMIC | [0.027, 0.320] (0.125) | 6–12× | 很活跃 |
| CheXpert | [0.010, 0.621] (0.125) | **最高 50×** | 极端集中（单组吃掉 48–62% 权重） |
| Fitzpatrick | [0.079, 0.284] (0.167) | ~3× | 活跃 |
| PAPILA | [0.236, 0.270] (0.250) | **1.1×** | **几乎不动** |

η 既没有小到退化成 ERM（除 PAPILA），也没有大到只训一个组——**η=0.01 的选择在四个库上是合理的**。
CheXpert 的极端集中（单组 q>0.6）与它最差的结果同时出现，提示那里 η 其实**偏大**。

## 4. 解读：这次实验为论文买到了什么

**① 决定性排除了「HN 靠属性占便宜」的混淆。** 这是加这个基线的首要目的，现在有了直接答案：
同样在训练时使用子群标签的 GroupDRO，在五个库上**没有一处显著改善 worst-group**，在两个最大的
库上**全指标显著变差**，而 HN 从不显著输给它。故 HN 的（打平或小幅领先的）表现**不是**"用了属性"
带来的红利。

**② null 结果从"我们的方法没赢"升级为"这类方法都赢不了"。** 原先四个 null 库的论证依赖
「我们实现的 HN 没有改善」+ V-information 闸门读数；现在加上「**文献标配的子群感知训练方法在
同一 regime 下同样没有改善、甚至显著更差**」这一独立证据。MIMIC 的 14 子群一致劣化尤其干净——
它不是"没抬起弱组"，而是**根本没有可重分配的结构**，与 `I(Y;A|X)≈0` 的读数直接吻合。

**③ 属性不是杠杆，泛化才是。** 五个库里唯一稳定改善 worst-group 的仍是 **SWAD**（HAM +0.046 显著、
Fitzpatrick 0.816→0.874），而它完全不看属性。GroupDRO 只在 Fitzpatrick 做到了"正确形状"的
重分配（gap 全方法最低），但幅度不显著，且以 Overall 为代价。

**④ 一个方法学发现（值得单独一段写进论文）：目标与指标错配。** GroupDRO 最小化的是 worst-group
**损失**，而公平性报告的是 worst-group **AUC**。在类别极不均衡的医学数据上二者可以严重不一致：
- Fitzpatrick 上 q 顶到最高的是 **skin:II**（最大的组），而 AUC 最弱的是 skin:VI；
- MIMIC 上损失最高的是 `Female|White|≥60`，AUC 最弱的却是 `Male|White|≥60`；
- HAM 上损失最高的是 `Male|40-60`（多数组）。

即 GroupDRO 把权重投向"BCE 最难拟合"的组，而不是"排序能力最差"的组。这解释了为什么它的
重分配即使发生也常常没落到被报告的那一格上——**这是把 DRO 类方法迁到 AUC 口径公平性任务时
的一个结构性问题，不是本实现的缺陷**（AUC 不可分解为逐样本损失，无法直接做成 DRO 目标）。

## 5. 局限（须在论文中如实写）

1. **η 未搜**（固定论文默认 0.01，为与其它方法同一网格）。q 动力学显示四库合理、**CheXpert 偏大**、
   **PAPILA 偏小**——PAPILA 仅 5 batch/epoch × ≤15 epoch ≈ 75 步更新，q 几乎不动，其"GroupDRO"
   实质是**组均衡 ERM**，该库结果不应作为 GroupDRO 本身的证据。
2. **HAM 训练集少 2.3%**（排 `age_group==-1`，与 3 个 HN 同口径）；其余四库无此差异。
3. **组均匀采样与对抗损失未拆开**。`--groupdro_sampler standard`（`groupdro_nobal`）已实现未运行。
4. **`drop_last=False` 的尾批**：MIMIC 每 epoch 1089 批中末批仅 7 个样本，8 组必然缺席（日志
   `空组 batch=1/1089`）。占比 0.09%，影响可忽略，但严格实现应 `drop_last=True`。
   Fitzpatrick 同理 1/91。其余库为 0。
5. ~~**未做 OOD**~~ → **✅ 已完成（2026-08-14）**：MIMIC↔CheXpert 双向、full-target、15 replicate
   （作业 75757/75778 训练 + 75829 推理）。**两方向 marginal-worst 与 Overall 均显著劣于 ERM**
   （marg −0.0069 / −0.0061），且 C→M 的 Overall 降幅是**整个 OOD v2 唯一超过训练噪声的效应**。
   报告见 `docs/ood_experiment_v2_groupdro.md`，预注册见 `ood_experiment_v2_preregistration.md` 偏离 #3。

## 6. 复现

```bash
# HAM（专用脚本）
sbatch --partition=gpus24 --export=ALL,PY_SCRIPT=.../train_ham10000_groupdro.py slurm/c1_ham_cv_search.sh
# 其余四库（通用入口，数据集经环境变量传入）
sbatch --partition=gpus24 --export=ALL,PY_SCRIPT=.../train_groupdro.py,GDRO_DATASET=mimic       slurm/c2_mimic_cv_search.sh
sbatch --partition=gpus24 --export=ALL,PY_SCRIPT=.../train_groupdro.py,GDRO_DATASET=chexpert    slurm/c3_chexpert_cv_search.sh
sbatch --partition=gpus24 --export=ALL,PY_SCRIPT=.../train_groupdro.py,GDRO_DATASET=fitzpatrick slurm/c4_fitz_cv_search.sh
sbatch --partition=gpus24 --export=ALL,PY_SCRIPT=.../train_groupdro.py,GDRO_DATASET=papila      slurm/c_search_papila_cv.sh

# 选 config（逐库，--merge 保住既有键）+ 全量重算
for d in ham10000 mimic chexpert fitzpatrick papila; do
  python scripts/select_config_cv.py --dataset $d --method groupdro \
    --write-json <该库 selected_configs.json> --merge
done
python scripts/build_oof_results_averaging.py
```
