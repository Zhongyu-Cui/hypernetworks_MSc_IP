# 比较协议（修订版）

> 🛑 **信号闸门口径更新（2026-07-19）**：闸门已由单-split 条件互信息改为 **OOF conditional V-information**
> （Xu 2020 / Hewitt 2021；权威 = `docs/conditional_v_information_gate.md`）。关键变化：
> **HAM-age 由「唯一正信号」退回「未检出」**（+0.0028 bits，CI 触 0）；Fitzpatrick ≈0；
> **CheXpert / MIMIC 全轴 CI<0 = 决定性 0**；唯一 detected 为 **PAPILA-age**（+0.041 bits，但 n=420）。
> control task 五库全部通过 ⇒ 零读数非假阴。本文「HAM-age `I(Y;age|X)>0`」一族表述**已过时**（下方已内联订正）；
> 各处**结论方向不变**，但「有信号」前提须按新闸门重读。


> 定稿日期：2026-07-01。替代 interim report §3.1.3。本文件是实验设计的**权威规格**；
> 精简版进度表见根目录 `CLAUDE.md`「比较协议与进度」节。

## 0. 核心论点（重定位）

超网络的价值不在于超越 SOTA 的整体性能，而在于**融合图像本身无法表达、但与诊断相关的
敏感属性信息**。因此 HN 的增益应当**当且仅当** `I(Y;A|X) > 0` 时出现。本比较以此为主线：
在信号存在处（HAM-age）验证增益，在信号缺失处解释 null result。conditional-MI 估计
`I(Y;A|X)` 由旁支**升为贯穿性解释变量**，与性能结果并列报告。

## 1. 对比方法（7 个：4 基线 + 3 HN）

| # | 方法 | 类型 | 用属性 | 属性介入位置 | 角色 |
|---|------|------|--------|-------------|------|
| 1 | ResNet-18 (ERM) | 基线 | ✗ | — | 参照零点 |
| 2 | SWAD | 基线 | ✗ | 训练（权重平均） | SOTA 锚点 / 域泛化 |
| 3 | ROC (Reject-Option) | 基线 | ✓ | 决策边界后处理 | 属性感知、最浅介入 |
| 4 | **GroupDRO** | **基线** | **✓** | **训练损失（组重加权）** | **属性感知、非条件化对照** |
| 5 | HyperHead | HN | ✓ | 分类头权重 | 浅层 HN |
| 6 | HyperAdapt | HN | ✓ | 各层低秩残差（除 stem） | 深层 HN（介入最遍布） |
| 7 | HyperFusion | HN | ✓ | 末层 downsampling 权重 | 中层 HN（单 block） |

- **ROC 复用 ERM 的已训练模型**做后处理，不单独训练。
- **SWAD 不与 HN 组合**，仅作独立基线（ERM 训练 + 权重平均）。
- **GroupDRO（Sagawa et al., ICLR 2020）单独训练**（自己的 6 配置搜索），实现见
  `src/training/harness/groupdro.py`。**加入动机（2026-08-11，导师建议）**：原三基线里 ERM/SWAD
  完全不看属性、ROC 只做事后阈值调整 ⇒「训练时用子群标签」这一格里只有 3 个 HN，
  「HN 用属性 vs 基线不用属性」是**混淆对比**。GroupDRO 填的正是「用属性但**不条件化函数**」一格：

  |               | 不用属性     | 用属性                                |
  |---------------|-------------|--------------------------------------|
  | 改损失/权重    | ERM, SWAD   | **GroupDRO**                         |
  | 改决策阈值     | —           | ROC                                  |
  | 改函数（条件化）| —           | HyperHead / HyperFusion / HyperAdapt |

  二者对信号的要求不同且可检验：conditioning 要兑现增益**必须** `I(Y;A|X)>0`；reweighting
  **不需要**条件信号（只是拿 Overall 换 worst-group）。故「四个 null 数据集上 HN 打平 ERM 而
  GroupDRO 仍抬 worst-group」是本协议对闸门论证的一次 out-of-sample 检验。
- 属性介入深度递增（ROC→Head→Fusion→Adapt），构成「属性用在越深处是否越有效」对照轴，
  服务 interim report §3.1.4 第 3 个研究问题。GroupDRO **不在该深度轴上**（它不介入函数）。
- `attrconcat` **移出主协议**（不属三设计之一），仅作可选 ablation。

## 2. 数据集 × 敏感属性 × OOD

| 数据集 | 主敏感轴 | 交叉子群 | OOD |
|--------|---------|---------|-----|
| MIMIC-CXR | Sex / Race / Age | Sex×Age | ✓ ↔ CheXpert |
| CheXpert | Sex / Race / Age | Sex×Age | ✓ ↔ MIMIC |
| HAM10000 | **Age**（信号轴）/ Sex | — | ✗ |
| Fitzpatrick17k | Skin tone (I–VI) | — | ✗ |
| PAPILA | Sex / Age | — | ✗ |

- **OOD 仅 CXR 一对**，双向做：train-MIMIC→test-CheXpert 与反向，复用已建 `*_nofinding` split。
- 其余三数据集只做 ID，报告中明确「无天然 OOD 对，故不构造」。

## 3. 训练与模型选择

**Minimax-Pareto 是在超参搜索产生的候选池上做选择**（不是单配置的 checkpoint 选择），
因此强制引入超参搜索。规程：

- **超参搜索网格**（适中）：`lr ∈ {3e-5, 1e-4, 3e-4} × wd ∈ {1e-4, 1e-3}` = 6 配置，
  搜索阶段每配置 1 seed；固定单配置（lr=1e-4, wd=1e-4）作为网格中心点。
- **确认阶段**：Pareto 选出的配置再跑 **5 seed**，报 test 均值 ± 95% CI。
- **模型选择：Minimax Pareto** —— 在候选池（config × seed）的各子群 AUC 向量上取 Pareto
  最优候选，再选 worst-group AUC 最高者。**每候选必须记录完整子群 AUC 向量**（非仅
  worst-case 标量）供 Pareto 使用。
- **省算力**：ROC 是 ERM 预测的后处理（零训练）；SWAD 套在 ERM 训练上（不单独搜参）。
  故每数据集只有 **ERM + 3 HN = 4 个训练 campaign** 需搜参，SWAD/ROC 搭 ERM 的车。
- 重型 HN 显存对照**必须等 batch**（曾被 HyperAdapt b32 OOM 假象坑过）。

> ⚠️ **现有 run 不合规**：截至 2026-07-01 的训练是固定单配置、仅 3 seed、只存 worst-case
> 标量 checkpoint 的 **pilot**，不构成 Pareto 候选池，**不进正式结果**。基线与 HN 一并需在
> 上述统一 regime 下**重训**（见工作清单 C）。

## 4. 评估度量

- Overall AUC、各子群 AUC、**AUC gap**（max−min）、**worst-group AUC**。
- 全数据集报 ID；CXR 额外报 OOD。
- 交叉子群仅在样本量支撑处（CXR 的 Sex×Age）报告。

## 5. 统计检验（仅数据集内）

- **砍掉跨数据集 Friedman/Nemenyi**（N=5、4/5 为 null，功效不足且掩盖唯一真实效应）。
- **数据集内**：≥5 seed 的 worst-group AUC / AUC gap 报**均值 ± 95% CI**。
  - HAM：HyperFusion vs {ERM, SWAD, ROC} 做**配对 bootstrap CI / DeLong**，证明提升非种子噪声。
  - null 数据集：用 CI 重叠 + `I(Y;A|X)≈0` 论证「确实打平、根因数据侧」。

## 6. 产出

按方法给出 ID（全部）+ OOD（CXR）性能表、子群公平度量表、数据集内显著性、以及**按
worst-group AUC 与 AUC gap 的 per-dataset 排名**（不做跨数据集聚合排名）。conditional-MI
表与性能结论并置解释。

## 7. 相对原 interim report 的改动

保留 SWAD；删跨数据集统计检验；OOD 收缩到 CXR；新增数据集内多-seed 显著性；conditional-MI
升为主线；attrconcat 降为可选 ablation。

---

## 工作清单与进度

状态标记：`[ ]` 未开始 · `[~]` 进行中 · `[x]` 完成

> **粒度约定**：每个复选框 = 一件可独立完成并验证的任务。**工作规范**（各阶段完成后如何
> 报告、C 阶段的 SLURM 提交纪律等）见根目录 `CLAUDE.md`「工作规范」节。

### A. 基础设施（一次实现，全数据集共用）—— 最高优先
- [x] A0.1 训练 harness：超参网格遍历器（6 配置）—— `src/training/harness/hparam_grid.py`
- [x] A0.2 训练 harness：每候选完整子群 AUC 向量写入 val 日志 —— `harness/subgroup_auc.py` + `harness/val_log.py`
- [x] A0.3 训练 harness：seed 参数化（5 seed，贯通训练与 checkpoint 命名）—— `harness/run.py`（run_id + seed_everything + SEEDS_SEARCH/CONFIRM）
- [x] A0.4 改造现有各训练脚本调用统一 harness —— `harness/train_loop.py`（run_training + unpack/forward 回调）；5 数据集 ×{ERM,HyperHead,HyperFusion,HyperAdapt} 共 16 脚本全部接入（attrconcat/frozen 为 ablation 不在主协议、未改）
- [x] A0.5 每 run 落盘 test 逐样本预测（A0 延伸，服务 A2/A4/A5/D）—— `harness/predictions.py`（save/load + `default_prediction_path` predictions/<run_id>_<selection>.npz）；`run_training` 两 selection + `_finalize_swad` 均已落盘
- [x] A1.1 Pareto 选择器：子群 AUC 向量的 Pareto 前沿计算 —— `harness/pareto.py`（`pareto_front` + `_dominates`，None 掩码模型无关一致性断言）
- [x] A1.2 Pareto 选择器：前沿上选 worst-group 最优并输出选定配置 —— `harness/pareto.py`（`select_minimax`/`select_config`，worst-case 最高、平手用 overall 破）
- [x] A2.1 显著性：多-seed 汇总（worst-group AUC / AUC gap，均值±95%CI）—— `harness/significance.py`（`mean_ci` Student-t + `worst_and_gap` 向量派生 + `aggregate_seed_scalars` 剔 None）
- [x] A2.2 显著性：配对 bootstrap CI（方法 vs 基线差值）—— `harness/significance.py`（`paired_bootstrap_diff` 索引闭包解耦 + `make_subgroup_metric`/`_multiseed` 样本×种子两层不确定度）
- [x] A2.3 显著性：AUC 差异 DeLong 检验 —— `harness/significance.py`（`delong_test` fast DeLong/Sun&Xu2014，midrank，AUC 与 sklearn 一致、SE 与 bootstrap 交叉验证）
- [x] A3.1 SWAD：权重平均包装器（epoch 区间检测 + 平均）—— `harness/swad.py`（`select_swad_interval` loss valley N_s/N_e/r + `average_state_dicts` 浮点均值/int buffer 取末 + `swad_average_over_interval`）
- [x] A3.2 SWAD：接入 ERM 训练流程（数据集无关）—— `harness/train_loop.py`（`run_training(swad=True)` 逐 epoch 缓存 CPU 权重 + `_finalize_swad` 派生 method="swad" checkpoint/val 日志、val+test 评估）
- [x] A4.1 ROC：reject-option 后处理实现（近边界预测调整）—— `harness/roc.py`（`reject_option_adjust` 临界区翻转 favor deprived + `binarize_axis_by_performance` 逐轴按val子群AUC二分 + `select_margin` val调θ恒不劣ERM）
- [x] A4.2 ROC：接入评估/输出管线（数据集无关）—— `harness/roc.py`（`apply_roc` val定deprived/选θ→test全子群评估 + `apply_roc_auto` 自动选轴 + `roc_fairness_report` before/after）
- [x] A5.1 OOD：跨数据集评估脚本（加载 A 模型 → B 测试集评估）—— `src/training/eval_ood_cxr.py`（`evaluate_ood` + METHOD_REGISTRY(erm/swad/3HN) + `build_target_test_loader` 复用 MIMICCXRDataset，harness.evaluate + mimic 公平性）
- [x] A5.2 OOD：双向参数化（MIMIC↔CheXpert）—— `src/training/eval_ood_cxr.py`（`evaluate_ood_pair` 用 mimic 权重评 chexpert / 反向，方向+checkpoint+目标loader 已验证）

### B. 补全 HN 训练脚本（代码层）
- [x] B1 Fitzpatrick HyperHead 训练脚本 —— 模型 `ResNet18HyperHeadSkin`（+`SkinHyperHeadNet`）+ `train_fitzpatrick_hyperhead.py` + slurm（skin 条件，bs=128）
- [x] B2 Fitzpatrick HyperFusion 训练脚本 —— 模型 `ResNet18HyperFusionSkin`（+`SkinEmbedding`）+ `train_fitzpatrick_hyperfusion.py` + slurm（skin 条件，bs=128）
- [x] B3 PAPILA HyperHead 训练脚本 —— 复用 `ResNet18HyperHeadAge(num_age=2)` + `train_papila_hyperhead.py` + slurm（age 条件，bs=64，支持 --cv/--fold）
- [x] B4 PAPILA HyperAdapt 训练脚本 —— 复用 `ResNet18HyperAdaptAge(num_age=2)` + `train_papila_hyperadapt.py` + slurm（age 条件，--batch_size 32 覆盖，支持 --cv/--fold）

### C. 统一重训（大头；HAM 优先）
每数据集 8 个原子任务：4 方法各一次超参搜索训练 → Pareto 选择 → 5-seed 确认 → 派生 SWAD → 派生 ROC。

**C0 提交脚手架（C 阶段共用，一次实现）**
- [x] C0.1 搜索用通用 SLURM：`slurm/c_search.sh`（array 0–5→config_index，seed=42；分区/日志/PY_SCRIPT 由 sbatch+--export 注入）
- [x] C0.2 确认用通用 SLURM：`slurm/c_confirm.sh`（array 0–4→SEEDS_CONFIRM，CONFIG_INDEX 经 --export 传 Pareto 选定值；SWAD=1 供 ERM 确认同产 SWAD）
- [x] C0.3 ERM 脚本加 `--swad` 开关（透传 run_training(swad=True)，C*.7 不另占作业）
- [x] C0.4 val 预测落盘：train_loop 用 overall checkpoint 落 `<run_id>_val_overall.npz`（`predictions.val_prediction_path`），供 ROC 定 deprived/θ
- [x] C0.5 ROC 派生 CLI：`src/training/derive_roc.py`（读 test+val 预测→apply_roc_auto→落 method=roc 预测，CPU 后处理，端到端 smoke 通过）

**C1 HAM10000（优先）**
- [x] C1.1 ERM 超参搜索训练（6 配置 ×1 seed）—— 作业 **69577**（gpus24），全 6/6 完成，val overall AUC 0.897–0.912
- [x] C1.2 HyperHead 超参搜索训练（6 配置）—— 作业 **69578**（gpus24），全 6/6 完成
- [x] C1.3 HyperFusion 超参搜索训练（6 配置）—— 作业 **69579**（gpus24），全 6/6 完成
- [x] C1.4 HyperAdapt 超参搜索训练（6 配置）—— 作业 **69580**（gpus48 / bs128，等 batch 对照），全 6/6 完成、无 OOM
- [x] C1.5 各方法 Pareto 选择最优配置（selection=worstcase, seed42）—— 选定：**ERM `lr1e-04_wd1e-04`(idx2, val-worst0.831)**、**HyperHead `lr1e-04_wd1e-03`(idx3, 0.820)**、**HyperFusion `lr3e-04_wd1e-03`(idx5, 0.843 最高)**、**HyperAdapt `lr3e-04_wd1e-04`(idx4, 0.840)**。深层 HN(Fusion/Adapt) val-worst 领先，与 HAM-age 信号存在一致。
- [x] C1.6 选定配置 5-seed 确认训练 —— 作业 **69607**(ERM+SWAD,gpus24) / **69608**(HyperHead,gpus24) / **69609**(HyperFusion,gpus24) / **69610**(HyperAdapt,gpus48 b128 等-batch)，array 0–4=seed42-46；ERM seed45 遇 semois GPU 故障(89min/epoch)已排除 semois 重跑(69643)。5 方法各 5 seed 预测齐全
- [x] C1.7 派生 SWAD（ERM 配置 + 权重平均，5 seed）—— 含在 C1.6 的 ERM 确认（作业 69607/69643 带 SWAD=1），5 seed 平均权重 + overall 预测齐全
- [x] C1.8 派生 ROC（ERM 5-seed 预测后处理）—— `derive_roc --dataset ham10000 --config-tag lr1e-04_wd1e-04`，5 seed 全部落盘 `roc_lr1e-04_wd1e-04_seed4?_overall.npz`（二分轴=age，θ* val 选定恒不劣 ERM）
- [x] **C1.9 GroupDRO 基线（§1 新增第 4 基线，导师建议）** —— 实现 `harness/groupdro.py` +
  `train_ham10000_groupdro.py`；作业 **75498**（gpus24，6 config × 5 折 = 30/30 成功）；折均
  marginal-worst 选定 `lr3e-04_wd1e-03`；averaging 评估已并入 `oof_results_averaging.json`。
  **结论：Overall 显著劣于 ERM（−0.0124）、worst-group 未抬（canon −0.006 / marg +0.015 均 n.s.）；
  3 HN vs GroupDRO 的 worst-group 全 n.s. 而 Overall 显著更优 ⇒「HN 靠属性占便宜」的混淆被解除。**
  完整报告见 `docs/groupdro_baseline_ham.md`

**C2 MIMIC-CXR**
- [x] C2.1 ERM 超参搜索训练（6 配置）—— seed42 全 6 配置预测齐全（Pareto 见 C2.5）
- [x] C2.2 HyperHead 超参搜索训练（6 配置）—— seed42 全 6 配置预测齐全
- [x] C2.3 HyperFusion 超参搜索训练（6 配置）—— seed42 全 6 配置预测齐全
- [x] C2.4 HyperAdapt 超参搜索训练（6 配置）—— seed42 全 6 配置预测齐全（gpus48 b128 等-batch）
- [x] C2.5 各方法 Pareto 选择最优配置（worstcase, seed42）—— ERM `lr3e-04_wd1e-03`(idx5,val-worst0.809) / HyperHead `lr1e-04_wd1e-04`(idx2,0.806) / HyperFusion `lr1e-04_wd1e-04`(idx2,0.807) / HyperAdapt `lr3e-04_wd1e-04`(idx4,0.809)。4 方法 worst 挤在 0.806–0.809，与 MIMIC I(Y;A|X)≈0（HN≈baseline）先验一致
- [x] C2.6 选定配置 5-seed 确认训练 —— 作业 **69707**(ERM+SWAD) / **69708**(HyperHead) / **69709**(HyperFusion,gpus24) / **69710**(HyperAdapt,gpus48)，array 0–4=seed42-46，已排除 semois；5 方法 × 5 seed 预测齐全
- [x] C2.7 派生 SWAD（5 seed）—— 含在 C2.6 的 ERM 确认（69707 带 SWAD=1），5 seed overall 预测齐全
- [x] C2.8 派生 ROC（后处理）—— `derive_roc --dataset mimic --config-tag lr3e-04_wd1e-03 --output-dir outputs/mimic_cxr`（⚠️ subgroup 键=`mimic` 但 outputs 目录=`mimic_cxr`，须显式 --output-dir；HAM 因同名未暴露）。5 seed 全落盘，θ* 多为 0/极小=模型已均衡无腾挪空间
- [x] **C2.9 GroupDRO 基线（§1 第 4 基线）** —— 通用入口 `train_groupdro.py`（GDRO_DATASET=mimic），作业 **75529**（6 config × 5 折 = 30/30 成功）；折均 marginal-worst 选定 `lr3e-04_wd1e-04`；averaging 已并入 `oof_results_averaging.json`。**结论：**三指标全部显著劣于 ERM**（Overall −0.0079 / canon worst −0.0094 / marg −0.0082）；14 个子群一致下降 ~0.008、**无任何重分配** = I(Y;A|X)≈0 的指纹；3 HN 相对 GroupDRO 全指标显著更优**。汇总见 `docs/groupdro_baseline_5datasets.md`

**C3 CheXpert**
- [x] C3.1 ERM 超参搜索训练（6 配置）—— 作业 **69762**（gpus24）
- [x] C3.2 HyperHead 超参搜索训练（6 配置）—— 作业 **69763**（gpus24）
- [x] C3.3 HyperFusion 超参搜索训练（6 配置）—— 作业 **69764**（gpus24）
- [x] C3.4 HyperAdapt 超参搜索训练（6 配置）—— 作业 **69765**（gpus48 b128 等-batch）；均已排除 semois
- [x] C3.5 各方法 Pareto 选择最优配置（worstcase, seed42）—— ERM `lr3e-04_wd1e-03`(idx5,val-worst0.751) / HyperHead `lr3e-05_wd1e-03`(idx1,0.730) / HyperFusion `lr1e-04_wd1e-04`(idx2,0.761) / HyperAdapt `lr3e-04_wd1e-03`(idx5,0.755)
- [x] C3.6 选定配置 5-seed 确认训练 —— 作业 **69792**(ERM+SWAD) / **69793**(HyperHead) / **69794**(HyperFusion,gpus24) / **69795**(HyperAdapt,gpus48)，array 0–4=seed42-46，已排除 semois；5 方法 × 5 seed 预测齐全，0 错误日志、节点无 semois
- [x] C3.7 派生 SWAD（5 seed）—— 含在 C3.6 的 ERM 确认（69792 带 SWAD=1），5 seed overall 预测齐全
- [x] C3.8 派生 ROC（后处理）—— `derive_roc --dataset chexpert --config-tag lr3e-04_wd1e-03 --output-dir outputs/chexpert_cxr`，5 seed 全落盘 `roc_lr3e-04_wd1e-03_seed4?_overall.npz`（自动选轴 age×4/sex×1，θ* val 上恒不劣 ERM；test 上 Overall 略降、gap 无稳健收窄=模型已近均衡、与 CheXpert I(Y;A|X)≈0 一致）
- [x] **C3.9 GroupDRO 基线（§1 第 4 基线）** —— 通用入口 `train_groupdro.py`（GDRO_DATASET=chexpert），作业 **75530**（6 config × 5 折 = 30/30 成功）；折均 marginal-worst 选定 `lr1e-04_wd1e-03`；averaging 已并入 `oof_results_averaging.json`。**结论：**三指标全部显著劣于 ERM**（−0.0136 / −0.0201 / −0.0162）；q 极端集中（单组 0.48–0.62）提示 η 偏大；3 HN 相对 GroupDRO 全指标显著更优**。汇总见 `docs/groupdro_baseline_5datasets.md`

**C4 Fitzpatrick17k**
- [x] C4.1 ERM 超参搜索训练（6 配置）—— 作业 **69814**（gpus24），全 6/6 完成，val overall AUC 0.914–0.923
- [x] C4.2 HyperHead 超参搜索训练（6 配置）—— 作业 **69815**（gpus24），全 6/6 完成
- [x] C4.3 HyperFusion 超参搜索训练（6 配置）—— 作业 **69816**（gpus24），全 6/6 完成
- [x] C4.4 HyperAdapt 超参搜索训练（6 配置）—— 作业 **69826**（gpus48 b128 等-batch），全 6/6 完成、无 OOM
- [x] C4.5 各方法 Pareto 选择最优配置（worstcase, seed42）—— ERM `lr1e-04_wd1e-03`(idx3,val-worst0.9025) / HyperHead `lr1e-04_wd1e-04`(idx2,0.9004) / HyperFusion `lr3e-04_wd1e-04`(idx4,0.9110 最高) / HyperAdapt `lr1e-04_wd1e-03`(idx3,0.9073)。val-worst 全挤在 0.900–0.911，深层 HN 仅微弱领先，与 Fitzpatrick I(Y;skin|X)≈0（各法打平）先验一致
- [x] C4.6 选定配置 5-seed 确认训练 —— 作业 **69838**(ERM+SWAD,gpus24) / **69839**(HyperHead,gpus24) / **69840**(HyperFusion,gpus24) / **69841**(HyperAdapt,gpus48 b128 等-batch)，array 0–4=seed42-46，已排除 semois；5 方法 × 5 seed 预测齐全、0 错误
- [x] C4.7 派生 SWAD（5 seed）—— 含在 C4.6 的 ERM 确认（69838 带 SWAD=1），5 seed overall 预测齐全
- [x] C4.8 派生 ROC（5-seed 后处理）—— `derive_roc --dataset fitzpatrick --config-tag lr1e-04_wd1e-03`，5 seed 全落盘（二分轴=skin，θ* val 上恒不劣 ERM；test Overall/worst/gap 几乎不动=无腾挪空间）。**6 方法 5-seed test 汇总（null 结果）**：ERM Overall0.895/worst0.785/gap0.153；SWAD 0.911/0.781/0.167；ROC≡ERM；HyperHead 0.889/0.772/0.157；HyperFusion 0.881/0.747/0.186（fairness 最差、与 HAM 相反）；HyperAdapt 0.902/0.817/0.121（名义最优但 worst std±0.069 极大不稳健）。各法 worst CI 大幅重叠、Pareto val-worst 排序未传导到 test=小子群 val→test 噪声，与 I(Y;skin|X)≈0 打平预期一致
- [x] **C4.9 GroupDRO 基线（§1 第 4 基线）** —— 通用入口 `train_groupdro.py`（GDRO_DATASET=fitzpatrick），作业 **75531**（6 config × 5 折 = 30/30 成功）；折均 marginal-worst 选定 `lr1e-04_wd1e-04`；averaging 已并入 `oof_results_averaging.json`。**结论：**唯一「正确形状」的重分配**：型VI 0.816→0.858（+0.042，n.s.）、gap 0.090→0.055（全方法最低），Overall −0.003 n.s.；仍无显著 worst-group 收益**。汇总见 `docs/groupdro_baseline_5datasets.md`

**C5 PAPILA** — ⚠️ **regime 决策：全 CV（用户裁定 2026-07-03）**。PAPILA test≈84 眼极小，
`docs/papila_preprocessing_plan.md` 明确以患者级 5 折 GroupKFold 取代单次 split。C5 走「6 配置 × 5 折都跑」：
搜索每方法 30 run（`slurm/c_search_papila_cv.sh`，array 0–29，`config=idx%6, fold=idx//6, seed=42+fold`；
**fold↔seed 一一映射**——harness run_id=(method,config,seed) 不含 fold，5 折共用 seed42 会互相覆盖，故按折错开 seed，
零核心改动即复用 Pareto/SWAD/ROC/显著性全链）。全-CV 下选定 config 的 5 折即确认分布，**C5.6 无需再训**。
Pareto 用 `scripts/select_pareto_papila_cv.py`（先按 config 跨 5 折聚合子群向量取折均→6 聚合候选→minimax，
避免选到单折噪声）。最终 test 用 5 折 disjoint OOF 池化成全 420 眼（已验证：各折 84 眼、阳性 19/14/16/19/19、池化=420）。
- [x] C5.1 ERM 超参搜索训练（6 配置 ×5 折=30 run）—— 作业 **69860**（gpus24 +SWAD），全 30/30 完成，OOF 池化=420 眼验证通过、0 错误
- [x] C5.2 HyperHead 超参搜索训练（30 run）—— 作业 **69890**（gpus24），全 30/30 完成
- [x] C5.3 HyperFusion 超参搜索训练（30 run）—— 作业 **69891**（gpus24），全 30/30 完成
- [x] C5.4 HyperAdapt 超参搜索训练（30 run）—— 作业 **69892**（gpus48 b64 等-batch），全 30/30 完成、无 OOM
- [x] C5.5 各方法折聚合 Pareto 选择最优配置（worstcase，折均）—— ERM `lr3e-04_wd1e-04`(idx4,折均worst0.9042) / HyperHead `lr3e-04_wd1e-04`(idx4,0.8747) / HyperFusion `lr1e-04_wd1e-03`(idx3,0.8742) / HyperAdapt `lr3e-04_wd1e-04`(idx4,0.8908)。config 间折均 worst 跨度极大（低 lr 掉 0.50=小子群随机），高 lr 稳健更优=小样本噪声
- [x] C5.6 选定配置确认（全-CV：选定 config 的 5 折已在搜索中训完，无需新作业）
- [x] C5.7 派生 SWAD（逐折）—— ERM 搜索 69860 带 SWAD=1，30/30 SWAD 预测齐全（选定 config lr3e-04_wd1e-04 的 5 折即用）
- [x] C5.8 派生 ROC（逐折后处理）—— `derive_roc --dataset papila --config-tag lr3e-04_wd1e-04`，5 折全落盘（轴=age，θ*=0 全折 ROC≡ERM=无腾挪）。**6 方法 OOF 池化(n=420) test**：ERM Overall0.841/worst0.798/gap0.060；SWAD 0.859/**0.831**/0.046(worst 最优)；ROC≡ERM；HyperHead 0.811/0.742/0.103；HyperFusion 0.796/**0.670**/0.192(退化最重)；HyperAdapt **0.860**/0.811/0.063(Overall 最优、打平 baseline)。**无 HN 稳健改善公平**：HyperAdapt 打平 ERM/SWAD、HyperHead/HyperFusion 退化。⚠️ **R3.3 订正**：PAPILA 探针在 n=420 下不可靠（剂量-反应非单调，见 results_summary D2-MDE），`I(Y;A|X)` **不可判定=灰区**，**已退出 null 证据集**——此处性能打平仅作趋势旁证、不作 null 判决支撑（区别于 MIMIC 真值≈0 / CheXpert·Fitz 未达 MDE；区别于 HAM 弱正）
- [x] **C5.9 GroupDRO 基线（§1 第 4 基线）** —— 通用入口 `train_groupdro.py`（GDRO_DATASET=papila），作业 **75532**（6 config × 5 折 = 30/30 成功）；折均 marginal-worst 选定 `lr3e-04_wd1e-03`；averaging 已并入 `oof_results_averaging.json`。**结论：全指标 n.s.（canon worst −0.037）；⚠️ 仅 ~75 步 q 更新、q∈[0.236,0.270] 几乎不动 ⇒ 实质是组均衡 ERM，**不作为 GroupDRO 本身的证据****。汇总见 `docs/groupdro_baseline_5datasets.md`

**C6 OOD 评估（CXR）** —— 驱动 `src/training/run_ood_cxr_all.py`（复用 A5 `evaluate_ood`）+ `slurm/c6_ood_cxr.sh`；
5 方法(ERM/SWAD/3HN，ROC 属分数后处理不在此) × 5 seed × 2 方向 = 50 次推理，用确认配置 checkpoint
（SWAD=_averaged，余=_best_overall），OOD 预测落盘 `outputs/ood_cxr/<src>2<tgt>/`。作业 **69980**（gpus，both 双向）。
- [x] C6.1 MIMIC→CheXpert OOD 评估 —— 作业 **69980**（gpus），25/25 落盘。**5-seed 均值**：ERM Overall0.853/worst0.817/gap0.045；SWAD 0.853/0.811/0.051；HyperHead 0.855/0.819/0.045；HyperFusion 0.852/0.810/0.055；HyperAdapt 0.856/0.818/0.049。全法 ±1σ 内重叠，HN≈baseline
- [x] C6.2 CheXpert→MIMIC OOD 评估 —— 作业 **69980**（gpus），25/25 落盘。**5-seed 均值**：ERM Overall0.811/worst0.786/gap0.042；**SWAD 0.821/0.797/0.038（微弱领先，符域泛化设计）**；HyperHead 0.813/0.788/0.042；HyperFusion 0.815/0.790/0.042；HyperAdapt 0.816/0.791/0.041。**无 HN 稳健改善 OOD 泛化/公平**，与 MIMIC/CheXpert I(Y;A|X)≈0（无条件信号可利用）一致；SWAD 作为域泛化锚点在 C→M 有小幅优势

- [x] **C6.3 GroupDRO 双向 OOD（OOD v2 偏离 #3 的第四臂）** —— 训练作业 **75757**(MIMIC)/**75778**(CheXpert) 补 trial 1/2，推理作业 **75829**（full-target 双向，各 15 replicate，30/30 落盘）。**H5 决定性否定：两方向 marginal-worst 与 Overall 均显著劣于 ERM**（marg −0.0069 / −0.0061；Overall −0.0077 / −0.0055），**C→M 的 Overall 降幅是整个 OOD v2 唯一 \|d̄\|>σ_train 的效应**（正面效应至今无一超噪声）。Holm family 4→6 后既有 4 项全部保持显著、既有数字逐位不变。⇒ **「HN 靠属性占便宜」的替代解释在 ID 与 OOD 两个 regime 均被排除**。报告 `docs/ood_experiment_v2_groupdro.md`

### D. 汇总与写作（在 `docs/` 单独建 md 报告）—— 全部完成，报告 `docs/results_summary.md`；数据由 `scripts/build_results_tables.py` 统一重算
- [x] D1 建立 `docs/results_summary.md` 报告框架 —— 含 D1 论点 + D2–D7 全表；主口径 overall 选择、5-seed±95%CI
- [x] D2 汇总 conditional-MI 表并与性能并置 —— **唯 HAM-age +0.0138（95%CI 下界>0）可检测**；MIMIC 真值≈0（噪声地板）、CheXpert/Fitz 不可判定（<MDE）、**PAPILA 灰区退出 null（R3.3）**。CI/MDE 补强见 R3.1/R3.2（results_summary D2/D2-MDE）
- [x] D3 生成 ID 性能表（全数据集）—— 5 数据集 × 6 方法 Overall AUC（`subgroup_auc_vector` 统一口径）
- [x] D4 生成 OOD 性能表（CXR）—— 双向；⚠️ 用 canonical `subgroup_auc_vector`（含 sex×age joint）算 worst，与 C6 清单的边缘-only 数字不同，**以 D4 为准**（M→C worst≈0.73–0.75、C→M worst≈0.77–0.79）
- [x] D5 生成子群公平度量表 —— worst-group/gap 已并入 D3 各表；HAM 另报边缘 vs joint 口径隔离小格噪声
- [x] D6 生成数据集内显著性 —— **完整 3 HN × 3 基线矩阵**（此为比较协议正确设计=全方法×全基线；协议 §5 一度缩到「仅 HF」系**规格失误**、C 阶段重训时已订正，**非** Pareto 反转后的数据驱动探索扩展/forking-path，Pareto 反转仅是暴露契机；另加边缘 worst 口径隔离 joint 小格噪声）：worst-group 配对 bootstrap（canonical + 边缘）+ Overall DeLong。**5-seed 口径 D6-a/b/c 无一 p<0.05**（方向系统性为正）；**R4.1 更高功效补强 D6-HP + R4.3 BH-FDR：Overall AUC 全 3 HN 显著赢 ERM/ROC（全局 FDR 存活 7/9），worst-group 仍 n.s.（瓶颈=评估端 joint 小样本 R4.2，非 seed 数）**
- [x] D7 撰写结论 —— 核心论点成立（唯 HAM 有信号且方向正确）但增益被小样本压制；4 null 数据集干净打平；SWAD 是无信号时的稳健公平基线；介入深度非决定因素
- [x] **D8 实验总览（跨实验串联叙事）** —— `docs/experiments_overview.md`（2026-08-14）。把最终成立的四组实验按逻辑结构串成一条证据链：**ID CV-OOF regime（5 库 × 7 方法）→ OOD 实验 v2（双向 · 15 replicate · σ_train · 属性 knockout）→ 受控条件化范围消融（E0–E4）→ HyperAdapt×SWAD 融合（2×2）**，含 GroupDRO 第四基线的双 regime 排除性证据与四条方法学产出（pooled-AUC 负偏 / bootstrap 不含训练噪声 / 差中差须匹配 config / DRO 目标-指标错配）。⚠️ ID 数字一律以 `outputs/conditioning_ablation/oof_results_averaging.json`（2026-08-11）为准——`oof_regime_results.md` 的 **HyperAdapt 一列**因 2026-08-04 预测重写而略有失配（如 MIMIC Δmarg-worst 该文 +0.0022 vs JSON +0.0014），**方向与判决未变**

**起手顺序**：A0.1 → A0.2 → A0.3 → A0.4 → A1.1 → A1.2 → A2.1–A2.3（A0 是前提；让重训一次
到位地产出 Pareto 可选、可做显著性的候选池，避免重训两遍）→ A3/A4/A5 → B → C（HAM 优先）→ D。
**注意**：A2 无法再用现有 HAM 数值预演（那些不合规），须等 C1 的 HAM 重训出结果。

## 现状快照（2026-07-01）

**代码层**已实现的方法（⚠️ 对应的训练结果均为固定单配置、3-seed pilot，**不合规、须重训**）：

| 数据集 | 代码已实现 | 代码待补 |
|--------|-----------|---------|
| MIMIC-CXR | ERM, attrconcat, HyperHead, HyperAdapt(+frozen), HyperFusion | SWAD, ROC |
| CheXpert | ERM, HyperHead, HyperAdapt, HyperFusion | SWAD, ROC |
| HAM10000 | ERM, HyperHead, HyperAdapt, HyperFusion | SWAD, ROC |
| Fitzpatrick17k | ERM, HyperHead, HyperAdapt, HyperFusion | SWAD, ROC |
| PAPILA | ERM, HyperHead, HyperAdapt, HyperFusion | SWAD, ROC |

> 注：B1–B4（2026-07-01 完成）补齐 Fitzpatrick 的 HyperHead/HyperFusion 与 PAPILA 的
> HyperHead/HyperAdapt 后，五数据集的 ERM + 3 HN 训练脚本已全部就位；SWAD/ROC 为 harness 派生
> （A3/A4，数据集无关），无需逐数据集补脚本。至此 C 阶段统一重训所需的代码层已完整。

信号闸门（与模型选择无关，**可复用**）：五数据集均已估计。
🛑 **2026-07-19 口径更新**：已由单-split 条件互信息改为 **OOF conditional V-information**
（权威 `docs/conditional_v_information_gate.md`）。新读数：**唯 PAPILA-age detected**（+0.041 bits，n=420）；
**HAM-age 退回未检出（≈0）**；Fitzpatrick ≈0；**CheXpert / MIMIC 全轴 CI<0 = 决定性 0**。
（旧表述「仅 HAM-age > 0，余者 ≈ 0」已作废。）
