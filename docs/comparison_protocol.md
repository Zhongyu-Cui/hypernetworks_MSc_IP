# 比较协议（修订版）

> 定稿日期：2026-07-01。替代 interim report §3.1.3。本文件是实验设计的**权威规格**；
> 精简版进度表见根目录 `CLAUDE.md`「比较协议与进度」节。

## 0. 核心论点（重定位）

超网络的价值不在于超越 SOTA 的整体性能，而在于**融合图像本身无法表达、但与诊断相关的
敏感属性信息**。因此 HN 的增益应当**当且仅当** `I(Y;A|X) > 0` 时出现。本比较以此为主线：
在信号存在处（HAM-age）验证增益，在信号缺失处解释 null result。conditional-MI 估计
`I(Y;A|X)` 由旁支**升为贯穿性解释变量**，与性能结果并列报告。

## 1. 对比方法（6 个：3 基线 + 3 HN）

| # | 方法 | 类型 | 用属性 | 属性介入位置 | 角色 |
|---|------|------|--------|-------------|------|
| 1 | ResNet-18 (ERM) | 基线 | ✗ | — | 参照零点 |
| 2 | SWAD | 基线 | ✗ | 训练（权重平均） | SOTA 锚点 / 域泛化 |
| 3 | ROC (Reject-Option) | 基线 | ✓ | 决策边界后处理 | 属性感知、最浅介入 |
| 4 | HyperHead | HN | ✓ | 分类头权重 | 浅层 HN |
| 5 | HyperAdapt | HN | ✓ | 各层低秩残差（除 stem） | 深层 HN（介入最遍布） |
| 6 | HyperFusion | HN | ✓ | 末层 downsampling 权重 | 中层 HN（单 block） |

- **ROC 复用 ERM 的已训练模型**做后处理，不单独训练。
- **SWAD 不与 HN 组合**，仅作独立基线（ERM 训练 + 权重平均）。
- 属性介入深度递增（ROC→Head→Fusion→Adapt），构成「属性用在越深处是否越有效」对照轴，
  服务 interim report §3.1.4 第 3 个研究问题。
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
- [ ] A1.1 Pareto 选择器：子群 AUC 向量的 Pareto 前沿计算
- [ ] A1.2 Pareto 选择器：前沿上选 worst-group 最优并输出选定配置
- [ ] A2.1 显著性：多-seed 汇总（worst-group AUC / AUC gap，均值±95%CI）
- [ ] A2.2 显著性：配对 bootstrap CI（方法 vs 基线差值）
- [ ] A2.3 显著性：AUC 差异 DeLong 检验
- [ ] A3.1 SWAD：权重平均包装器（epoch 区间检测 + 平均）
- [ ] A3.2 SWAD：接入 ERM 训练流程（数据集无关）
- [ ] A4.1 ROC：reject-option 后处理实现（近边界预测调整）
- [ ] A4.2 ROC：接入评估/输出管线（数据集无关）
- [ ] A5.1 OOD：跨数据集评估脚本（加载 A 模型 → B 测试集评估）
- [ ] A5.2 OOD：双向参数化（MIMIC↔CheXpert）

### B. 补全 HN 训练脚本（代码层）
- [ ] B1 Fitzpatrick HyperHead 训练脚本
- [ ] B2 Fitzpatrick HyperFusion 训练脚本
- [ ] B3 PAPILA HyperHead 训练脚本
- [ ] B4 PAPILA HyperAdapt 训练脚本

### C. 统一重训（大头；HAM 优先）
每数据集 8 个原子任务：4 方法各一次超参搜索训练 → Pareto 选择 → 5-seed 确认 → 派生 SWAD → 派生 ROC。

**C1 HAM10000（优先）**
- [ ] C1.1 ERM 超参搜索训练（6 配置 ×1 seed）
- [ ] C1.2 HyperHead 超参搜索训练（6 配置）
- [ ] C1.3 HyperFusion 超参搜索训练（6 配置）
- [ ] C1.4 HyperAdapt 超参搜索训练（6 配置）
- [ ] C1.5 各方法 Pareto 选择最优配置
- [ ] C1.6 选定配置 5-seed 确认训练
- [ ] C1.7 派生 SWAD（ERM 配置 + 权重平均，5 seed）
- [ ] C1.8 派生 ROC（ERM 5-seed 预测后处理）

**C2 MIMIC-CXR**
- [ ] C2.1 ERM 超参搜索训练（6 配置）
- [ ] C2.2 HyperHead 超参搜索训练（6 配置）
- [ ] C2.3 HyperFusion 超参搜索训练（6 配置）
- [ ] C2.4 HyperAdapt 超参搜索训练（6 配置）
- [ ] C2.5 各方法 Pareto 选择最优配置
- [ ] C2.6 选定配置 5-seed 确认训练
- [ ] C2.7 派生 SWAD（5 seed）
- [ ] C2.8 派生 ROC（后处理）

**C3 CheXpert**
- [ ] C3.1 ERM 超参搜索训练（6 配置）
- [ ] C3.2 HyperHead 超参搜索训练（6 配置）
- [ ] C3.3 HyperFusion 超参搜索训练（6 配置）
- [ ] C3.4 HyperAdapt 超参搜索训练（6 配置）
- [ ] C3.5 各方法 Pareto 选择最优配置
- [ ] C3.6 选定配置 5-seed 确认训练
- [ ] C3.7 派生 SWAD（5 seed）
- [ ] C3.8 派生 ROC（后处理）

**C4 Fitzpatrick17k**
- [ ] C4.1 ERM 超参搜索训练（6 配置）
- [ ] C4.2 HyperHead 超参搜索训练（6 配置）
- [ ] C4.3 HyperFusion 超参搜索训练（6 配置）
- [ ] C4.4 HyperAdapt 超参搜索训练（6 配置）
- [ ] C4.5 各方法 Pareto 选择最优配置
- [ ] C4.6 选定配置 5-seed 确认训练
- [ ] C4.7 派生 SWAD（5 seed）
- [ ] C4.8 派生 ROC（后处理）

**C5 PAPILA**
- [ ] C5.1 ERM 超参搜索训练（6 配置）
- [ ] C5.2 HyperHead 超参搜索训练（6 配置）
- [ ] C5.3 HyperFusion 超参搜索训练（6 配置）
- [ ] C5.4 HyperAdapt 超参搜索训练（6 配置）
- [ ] C5.5 各方法 Pareto 选择最优配置
- [ ] C5.6 选定配置 5-seed 确认训练
- [ ] C5.7 派生 SWAD（5 seed）
- [ ] C5.8 派生 ROC（后处理）

**C6 OOD 评估（CXR）**
- [ ] C6.1 MIMIC→CheXpert OOD 评估
- [ ] C6.2 CheXpert→MIMIC OOD 评估

### D. 汇总与写作（在 `docs/` 单独建 md 报告）
- [ ] D1 建立 `docs/results_summary.md` 报告框架
- [ ] D2 汇总 conditional-MI 表并与性能并置
- [ ] D3 生成 ID 性能表（全数据集）
- [ ] D4 生成 OOD 性能表（CXR）
- [ ] D5 生成子群公平度量表（AUC gap / worst-group AUC）
- [ ] D6 生成 per-dataset 数据集内显著性与排名
- [ ] D7 撰写结论（conditional-MI 解释框架 + 各数据集判决）

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
| Fitzpatrick17k | ERM, HyperAdapt | HyperHead, HyperFusion, SWAD, ROC |
| PAPILA | ERM, HyperFusion | HyperHead, HyperAdapt, SWAD, ROC |

conditional-MI（`I(Y;A|X)`，与模型选择无关，**可复用**）：五数据集均已估计。仅
**HAM-age > 0**，余者 ≈ 0（含 MIMIC / CheXpert / Fitzpatrick，均低于检出地板）。
