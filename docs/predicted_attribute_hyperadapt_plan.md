# 方案：以子群分类器的 softmax 预测替代真值敏感属性（Predicted-Attribute HyperAdapt）

> 状态：**方案稿，尚未实现**（2026-08-18）。定位为比较协议之外的新增实验线 **P**（Predicted attributes）。
> 依赖既有资产：`src/models/resnet18_hyperadapt.py`、`src/training/harness/*`、`data/splits/*/cv5/`、
> 各数据集 `outputs/*/cv5/selected_configs.json`。

---

## 1. 想法与定位

**做法**：训练一个小的子群分类器 g_φ(x) → 敏感属性的 softmax 分布 p̂ ∈ Δ^{C}；HyperAdapt 的条件输入
由真值属性 A 改为 p̂，**训练与测试两阶段都用 p̂**（不再触碰 GT 属性作为模型输入）。

### 1.1 必须先讲清的理论约束（决定怎么读结果）

p̂ = g(x) 是图像的可测函数 ⇒ 在信息论意义上 **I(Y; p̂ | X) = 0 恒成立**，无论 g 多准。
因此本方案**不可能**兑现本项目主线论点里的那条增益通道（「HN 的价值在于注入图像无法表达的属性信息」，
见 `docs/conditional_v_information_gate.md`）。这不是缺陷，而是本实验的定义域：它把 HyperAdapt
从「信息注入器」改造成 **纯粹的软路由 / 条件计算机制**（soft mixture-of-experts：按预测子群把样本
路由到不同的有效参数 θ + Δθ(p̂)）。它能且只能带来两类效应：

1. **归纳偏置/优化效应**：显式的子群路由可能让不同子群的决策面分开学，从而抬升 worst-group
   （与 clustering-DRO / JTT / EIIL 一族同源），即使不引入新信息；
2. **容量效应**：多出的超网络参数本身可能改变拟合，与属性语义无关。

⇒ **对照臂 (iv)(v) 是本方案的必需项，不是可选项**（§6），否则任何正向读数都无法区分 1 与 2。

并且，由于 g 与 HyperAdapt 被定义为**一个方法**（§3.2），上述判断对整个 pipeline 成立：
无论 g 内部多复杂，pipeline 的条件输入始终是 x 的函数，信息约束不变。

### 1.2 两条独立的价值主张（建议都写进结论）

- **部署价值（fairness without demographics）**：真实场景常拿不到 GT 敏感属性。若 pred-attr 臂
  与 GT-attr 臂打平，则「HyperAdapt 需要属性标注」这一部署障碍被移除，是可发表的正面结论
  （即使两者都只是打平 ERM）。
- **机制价值**：本项目已确证「四库无属性信号 ⇒ HN 一致 null」。pred-attr 臂给出了一个
  **不含任何属性信息**的 HyperAdapt 版本；它与 GT 臂的差 = 属性信息的贡献，它与 ERM 的差 =
  架构/路由/容量的贡献。这正好补上既有消融（E 系列条件化范围、G 系列 HN×SWAD）缺的一格。

---

## 2. 预注册假设与判决规则

以 CV-OOF averaging 口径（逐折算指标→折间平均，`scripts/build_oof_results_averaging.py` 同款）为主，
worst-group AUC 为主终点、Overall AUC 为约束终点。σ_train = 同配置多 seed 的训练间噪声，作噪声地板。

| 编号 | 假设 | 判决 |
|------|------|------|
| **P-H1** | pred-attr HyperAdapt 的 worst-group AUC ≥ GT-attr HyperAdapt（非劣，margin = σ_train） | 支持「属性标注可省」 |
| **P-H2** | pred-attr HyperAdapt 的 worst-group AUC > ERM，且超出 σ_train | 软路由本身有效（需 (iv)(v) 排除容量解释） |
| **P-H3** | pred-attr ≈ permuted-pred（对照 iv） | 判定 P-H2 的效应为容量伪影，非路由 |
| **P-H4**（预期）| 在四个 null 数据集上 pred-attr 与 GT-attr 一样打平 ERM | 与闸门理论一致；此时 P-H1 以「双双 null」形式成立，结论仍成立但强度弱 |

**预登记的最可能读数**：HAM/Fitzpatrick 上 g 对 age/skin 有中等以上精度，p̂ 高度确定 ⇒ 条件向量近似
GT 的确定性副本 ⇒ pred 臂 ≈ GT 臂 ≈ ERM（三方打平）。这仍是有价值的「非劣」结论，须在开工前
承认，避免事后叙事漂移。

---

## 3. 方法规格

### 3.1 子群分类器 g_φ

- **固定为低容量 probe（不设第二档）**：冻结 ImageNet 预训练 ResNet-18 → 512-d 特征 →
  多项 logistic 回归（每个敏感轴一个 softmax 头），带 L2 正则。特征只需抽一次
  （GPU 几分钟/数据集），probe 用 CPU 秒级训练。参数量 ~512×C，名副其实的 small classifier。
  **为什么锁死低容量**：本方案的 g 在训练集上 in-sample 出 p̂（§3.2），in-sample 与 out-of-sample
  的锐度差完全由 g 的容量决定——低容量 probe 的 train/test 准确率差通常只有几个百分点，
  条件输入分布几乎不漂移；换成全微调 CNN 则 train 侧会逼近 100% 准确、p̂ 近 one-hot，
  训练期条件输入退化成 GT 的副本（后果见 §3.2 末）。故 g 的容量在本方法里是**方法定义的一部分**，
  不作为自由旋钮。若 probe 精度贴近先验基线（即 g 几乎没学到东西），不是升级 g，
  而是记录下来并按对照臂 (v) 解释（详见 §7）。
- **多属性数据集**：一个 backbone + 多头（MIMIC/CheXpert: sex 2 / race 2 / age 2；HAM: age 4（+sex 2 备用）；
  Fitzpatrick: skin 6；PAPILA: sex 2 / age 2）。头之间独立 softmax，不做联合 |A| 类分类
  （联合类会在稀有交叉格上退化，且与既有 `PatientEmbedding` 的逐轴结构不匹配）。
- **损失**：CrossEntropy（类不均衡轴如 Fitzpatrick VI 用 class-balanced 权重，仅作用于 g）。
- **审计指标（必须落盘并写进报告）**：逐轴 accuracy / macro-F1 / one-vs-rest AUC / **ECE**，
  以及混淆矩阵。g 的质量是本实验唯一的自变量来源，读数必须可核。

### 3.2 训练与预测规程：g 在训练集上直接训练、in-sample 出 p̂

**方法定义**：g_φ 与 HyperAdapt 构成**单一方法**（顺序两阶段 pipeline），不是"主模型 + 外部辅助标注"。
因此 g 只在该 fold 的训练集上训练一次，并对 train/val/test 一并前向给出 p̂；训练与测试两阶段
都消费 g 的输出。**不做交叉拟合**——cross-fitting 是为"把 p̂ 当作无偏的外部估计量"服务的，
而本方法根本不宣称 p̂ 无偏：它就是这个 pipeline 内部的一个中间表示。

```
对每个 outer fold k ∈ {0..4}（沿用 data/splits/<ds>/cv5/fold{k}）：
  1. 在 fold-k 的 train 上训练 g（低容量 probe，§3.1）；
  2. 用同一个 g 对 train / val / test 全部前向 → p̂（train 侧即 in-sample 输出）；
  3. 温度校准：在 val 上拟合逐轴温度 T（temperature scaling），同一 T 施于 train/val/test 三者
     （val 不参与 g 的拟合，也本来就是本项目的选择/早停用集，不破坏方法的单一性）；
  4. 落盘 p̂ + 审计指标。
```

每数据集 g 的训练次数：**5 次（每 outer fold 一次）**，而非 25 次；`build_attr_predictions.py`
相应简化，无 inner 折编排。

**必须记录的诊断（只记录、不设 gate）**：逐轴的 train accuracy vs test accuracy、
p̂ 的平均熵（train vs test）、ECE。它们不改变方法，但决定结果怎么读：

- 两者接近（预期，probe 容量低）⇒ 训练/测试的条件输入同分布，Stage 1 是一个干净的独立实验；
- 两者相差大（g 记住了训练样本）⇒ 训练期 p̂ → GT one-hot，**Stage 1 在极限上退化为 Stage 0**
  （GT 条件训练 + 预测条件测试），两个 Stage 的读数会趋同。这不是错误，但报告里必须点明，
  否则会把"Stage 1 ≈ Stage 0"误读成独立复现。低容量 probe 正是让两个 Stage 保持可区分的手段。

（这一 in-sample 用法与 JTT 等两阶段方法一致：第一阶段模型在训练集上的自身输出直接驱动第二阶段。）

### 3.2b g 的实测审计（Fitzpatrick17k，2026-08-18）

`outputs/fitzpatrick/cv5/attr_pred/metrics.json`。6 类肤色，5 折逐折拟合：

| fold | 温度 T | acc(train) | acc(test) | Δacc | 熵(train) | 熵(test) | ECE(test) |
|------|-------|-----------|----------|------|----------|---------|-----------|
| 0 | 1.603 | 0.536 | 0.439 | +0.097 | 1.329 | 1.321 | 0.040 |
| 1 | 1.591 | 0.544 | 0.421 | +0.122 | 1.326 | 1.321 | 0.035 |
| 2 | 1.475 | 0.538 | 0.421 | +0.118 | 1.296 | 1.280 | 0.036 |
| 3 | 1.720 | 0.546 | 0.424 | +0.123 | 1.358 | 1.350 | 0.023 |
| 4 | 1.650 | 0.543 | 0.424 | +0.119 | 1.341 | 1.337 | 0.022 |

读数：

1. **g 明显有效但远非完美**：test acc 0.42–0.44 vs 多数类基线 0.30（macro-F1 0.41 vs 多数类的 0.08）。
   这正是本实验最有信息量的工作点——p̂ 既不是常数先验（否则退化为对照臂 v），也不是 GT 的副本
   （否则退化为 GT 臂），soft 与 hard 之间存在实质差别。
2. **in-sample 尖锐度问题几乎不存在**：acc 差 +0.116±0.010，而**熵差仅 −0.008±0.004**
   （train 侧甚至略高于 test）。低容量 probe 的设计目的达到：训练与测试的条件输入分布同尺度，
   Stage 1 是一个干净的独立实验，未向「GT 条件训练」退化。
3. **温度校准确实在起作用**：拟合出的 T≈1.5–1.7（>1 = 原始 probe 过自信），校准后 ECE 0.02–0.04，
   平均熵 1.32 nats（均匀分布上界 log 6 = 1.79）。p̂ 相当"软"，soft/hard 消融有意义。

### 3.3 软条件通路：期望嵌入（Expected Embedding）

现有条件通路是 `nn.Embedding` 查表（`PatientEmbedding` / `SkinEmbedding` / `AgeEmbedding`，
`src/models/resnet18_hyperadapt.py:125-238`）。改造为**概率加权的嵌入期望**：

```
e = p̂ᵀ E        # p̂ ∈ R^{B×C}, E = embedding.weight ∈ R^{C×d}  →  e ∈ R^{B×d}
profile = fuse(concat_axes(e_1, ..., e_M))     # fuse MLP 结构完全不变
```

性质：**p̂ = one-hot 时逐位等价于原 `nn.Embedding` 查表**，因此
(a) GT 臂与 pred 臂共用同一模型类与同一 state_dict schema；
(b) 已训好的 GT-HyperAdapt checkpoint 可**零重训**直接吃软属性（→ §6 Stage 0 先导实验）；
(c) Δθ 的零初始化保证仍成立，起点等价于纯 ResNet-18，不破坏既有初始化论证。

实现落点：新增 `SoftAttrEmbedding`（通用：给定各轴 num_classes 列表 + cat_embed_dim + out_dim），
以及子类 `ResNet18HyperAdaptSoft*`，只替换 `self.patient_embed`，其余 A_gen/B_gen/逐样本卷积/fc
机制原样复用（与既有 `ResNet18HyperAdaptSkin/Age` 的扩展方式一致）。

### 3.4 条件输入的三种模式（同一套代码，开关切换）

| 模式 | 条件输入 | 用途 |
|------|----------|------|
| `gt` | GT 的 one-hot | 现状复现（等价性回归测试必须逐位对齐既有结果） |
| `soft`（**主线**）| 校准后的 p̂ | 用户提出的方法 |
| `hard` | argmax(p̂) 的 one-hot | 消融：软概率是否比硬标签好（噪声吸收 vs 信息丢失） |

附加旋钮：温度 τ 扫描（τ ∈ {0.5, 1（校准值）, 2}）——把「锐度」变成一条可控轴，τ→∞ 时 p̂ → 常数
先验，模型连续退化到「ERM + 多余容量」，与对照臂 (v) 相接，是很干净的连续性检验。**τ 扫描列为可选**，
主线只跑校准值。

### 3.5 可选变体：联合训练（P-B）

既然 g + HyperAdapt 被定义为一个方法，一个更彻底的版本是**联合训练**：
总损失 `L = BCE(y, f(x; θ+Δθ(p̂))) + λ · CE(a, g(x))`，g 与主模型同步更新
（可共享 backbone 或保持独立编码器）。

- 优点：真正端到端，g 的表示会被主任务塑形，不再是纯属性分类器。
- 代价：训练期条件输入是**移动靶**（p̂ 随 g 更新而漂移），早期噪声大；λ 成为新超参；
  与 GT 臂的"单变量差异"论证被破坏。
- 结论：**默认走顺序式（P-A，§3.2）**；P-B 仅在 P-A 给出正向读数、需要进一步加强时再做，
  列入可选清单 P3.5。


---

## 4. 工程落地（文件级改动清单）

**新增**

| 路径 | 内容 |
|------|------|
| `src/models/soft_attr_embedding.py` | `SoftAttrEmbedding`；可选也放 `hard_from_soft` / 温度施加工具 |
| `src/models/resnet18_hyperadapt.py`（改）| 追加 `ResNet18HyperAdaptSoftAge` / `SoftSkin` / `SoftPatient` 三个子类（只换 `patient_embed`） |
| `src/training/attr_predictor.py` | g_φ 的定义、特征抽取、probe 训练、cross-fit 编排、温度校准、审计指标 |
| `scripts/build_attr_predictions.py` | 逐数据集×逐 fold（每 fold 训 1 次 g）跑 §3.2 规程，落盘 p̂ 与审计 JSON |
| `src/datasets/soft_attr_wrapper.py` | `SoftAttrDataset`：包装任一现有 Dataset，按**唯一键**追加 p̂ 向量 |
| `src/training/train_ham10000_hyperadapt_pred.py` 等 | 每数据集一个薄训练脚本（复制 GT 版，改 build_model / unpack / forward / method 名） |
| `slurm/p_pred_attr.sh` | 提交脚本（沿用 `c1_ham_cv_oof.sh` 的 array=fold、SEED=42+FOLD 约定） |
| `docs/predicted_attribute_hyperadapt_results.md` | 结果报告（D 阶段规范） |

**改动（小、且必须向后兼容）**

- `src/training/harness/train_loop.py`
  - 新增 `unpack_*_soft` / `forward_soft` 回调；
  - `evaluate()` 在 `subgroup_auc_vector(dataset, ..., **attrs_np)` 前**按白名单过滤** attrs
    （`{"sex","race","age","skin","a_syn"}`）。**这是硬性必要改动**：当前实现把 attrs 全量 splat，
    多出的 `soft_*` 键会直接 TypeError（`train_loop.py:245`）。
  - 把 `soft_*` 概率经 `save_predictions(..., extra=...)` 落进 test 预测 npz（既有 `extra` 机制
    `predictions.py:76-105` 正好为此设计），便于事后审计路由与子群的对应关系。
- 其余 harness（pareto / significance / cv_oof_report / subgroup_auc）**零改动**，只要 method 名新增。

**产物命名**（与既有隔离，遵循 `run_id = <method>_<config_tag>_seed<seed>`）

```
method = hyperadapt_pred      # 主线 soft
         hyperadapt_predhard  # hard 消融
         hyperadapt_predperm  # 对照 (iv) 置换
         hyperadapt_predconst # 对照 (v) 常数先验
产物根 = outputs/<dataset>/cv5/            （与 GT 臂同址，靠 method 名区分，直接进既有 OOF 分析）
p̂ 落盘 = outputs/<dataset>/cv5/attr_pred/fold{k}/{train,val,test}.npz
         keys: key（image_id / patient_id+eye 等唯一键）, prob_<axis> [N,C], gt_<axis> [N], temperature_<axis>
审计    = outputs/<dataset>/cv5/attr_pred/metrics.json
```

**对齐规程（第二大坑）**：HAM 训练脚本会**就地过滤** `age_group>=0`（`train_ham10000_hyperadapt.py:112`），
行序会变。因此 `SoftAttrDataset` **必须按唯一键 join**（HAM: `image_id`；CXR: `ImagePath`/`dicom_id`；
Fitzpatrick: `md5hash`；PAPILA: `RET<ID><眼别>`），**禁止按行号对齐**，并在构造时断言键集合完全一致、
无重复、无缺失。

---

## 5. 评估口径（不变，照搬既有规范）

- **分组永远用 GT 属性**：p̂ 只作模型输入。worst-group / EqOdds / 四联指标的分组键一律 GT，
  否则跨方法不可比（并且用 p̂ 分组会把 g 的误差算进公平指标，制造无法解释的读数）。
- CV-OOF **averaging** 为主口径（池化 raw logit 有已知负偏，见 `docs/` 内 pooled 伪影记录）。
- 显著性：`harness/significance.py` 的配对 bootstrap（HAM 按 lesion 聚类、CXR 按 patient 聚类）；
  与 σ_train 噪声地板对比后才下结论。
- 模型选择：主线**复用 GT-HyperAdapt 已选定的 Pareto 配置**（`cv5/selected_configs.json`：
  HAM `lr3e-05_wd1e-04`、MIMIC `lr1e-04_wd1e-03`），保证「仅属性来源」单变量差异、且算力中性
  （5 折 × 1 配置 = 一次 array 提交）。**可选加强**：为 pred 臂独立跑 6 配置 Pareto 搜索
  （+30 run/数据集），仅在主线读数临界时补。
- 等-batch纪律：HyperAdapt 逐样本卷积核显存大，pred 臂与 GT 臂**必须同 batch_size（HAM=128, gpus48）**，
  否则重蹈 b32 OOM 混淆伪影。
- **g 的超参不进搜索网格**：probe 的 L2 强度按固定默认值（或在 val 属性 NLL 上一次性定死，对全部臂与全部 fold 通用），避免把 g 变成第二条 forking path。

---

## 6. 实验矩阵与阶段

**Stage 0（先导，≈零训练成本，强烈建议先做）**
用已训好的 GT-HyperAdapt checkpoint，**只在推理时**把条件输入换成 p̂（soft / hard），跑 val+test。
因 §3.3 的等价嵌入设计，无需重训。产出「部署时属性不可得」的即时读数：若 AUC 几乎不掉，说明
条件通路对属性噪声鲁棒，Stage 1 的预期即为非劣；若明显掉，说明路由对属性取值敏感，Stage 1 更有看头。
成本：g 的训练 + 若干次前向 ≈ 1 GPU-hour/数据集。
注意 Stage 0 与 Stage 1 的关系随 g 的 in-sample 过拟合程度而趋同（§3.2 末），低容量 probe 下两者可区分。

**Stage 1（主线，用户提出的方法）**：训练+测试都用 p̂，5 折 CV-OOF。

| 臂 | method | 是否新训 | 说明 |
|----|--------|---------|------|
| (i) GT-HyperAdapt | `hyperadapt` | 否（已有） | 参照 |
| (ii) ERM | `erm` | 否（已有） | 参照 |
| (iii) **pred-soft** | `hyperadapt_pred` | 5 折 | 主线 |
| (iv) pred-permuted | `hyperadapt_predperm` | 5 折 | p̂ 在 train/test 内**跨样本置换**（保边际分布、破对应）⇒ 分离路由 vs 容量 |
| (v) pred-const | `hyperadapt_predconst` | 5 折 | 条件输入固定为训练集边际先验向量 ⇒ 纯容量上界 |
| (vi) pred-hard | `hyperadapt_predhard` | 5 折 | 可选消融 |

**数据集优先级**
1. **HAM10000**（age，主战场，cv5/选配/分析脚本齐备；g 对 age 的可预测性中等，p̂ 有真实不确定性 → 信息量最大）
2. **Fitzpatrick17k**（skin，g 精度最高 ⇒ p̂≈GT，是「预测能否替代真值」的最强检验）
3. **MIMIC-CXR**（可选，阴性对照：race 从 CXR 高度可预测但 I(Y;A|X)=0，预期 pred/GT 双双 null）

**算力估算**：每数据集 Stage 1 = 4 臂 × 5 折 = 20 run，与既有单个 C 阶段方法同量级
（HAM HyperAdapt 单折 gpus48 bs128 与既有 c1 作业同规格）。HAM+Fitzpatrick 合计 ≈ 40 run + g 的训练。

---

## 7. 风险与失败模式（预判 + 缓解）

| 风险 | 表现 | 缓解 |
|------|------|------|
| in-sample p̂ 过尖锐（train/test 锐度失配）| 训练期条件输入≈GT，Stage 1 向 Stage 0 退化 | g 锁死为低容量 probe（§3.1）；诊断记录 train/test accuracy 与熵差，并在报告中据此定调 |
| g 过度自信 | p̂ 近 one-hot，soft≡hard，实验退化 | 温度校准 + 报告 ECE；必要时看 τ 扫描 |
| g 太弱 | p̂ ≈ 常数先验，模型退化为 ERM+容量 | 与对照臂 (v) 对撞即可识别；报告 g 的 accuracy 与先验基线差 |
| 「p̂ 只是 X 的函数」被误读为信息增益 | 结论叙事错误 | §1.1 写进报告首段；任何正向读数须先过 (iv)(v) |
| 交叉格稀疏（HAM Female|80+）| worst-group 噪声地板 | 已由 cv5 OOF 抬升评估 n；沿用 min_class_n 门限 |
| 多重比较 | 4 臂 × 多指标 | 沿用 Holm/BH（`scripts/ham_multiplicity_fdr.py` 口径） |

---

## 8. 工作清单

- [x] P0.1 `SoftAttrEmbedding` + `ResNet18HyperAdaptSoft*` 子类，附**等价性单测**
      （one-hot 输入与 GT 模型逐位一致：embedding 层 max|Δ|=0，整模型 max|Δlogit|=0）
- [x] P0.2 `src/training/attr_predictor.py`：特征抽取 + probe（每 fold 一次）+ 温度校准 + 审计/诊断指标
- [x] P0.3 `scripts/build_attr_predictions.py`：Fitzpatrick 逐 fold 落盘 p̂（train 侧 in-sample）与
      metrics.json（HAM spec 已写好，未跑）
- [x] P0.4 `src/datasets/soft_attr_wrapper.py`（按唯一键 join，含一致性断言 + 四种 attr_mode 物化）
- [x] P0.5 harness：`GROUP_ATTR_KEYS` 白名单过滤 + soft 键经 `extra` 落盘
      （train_loop / predictions / subgroup_auc 三个自检全绿，既有路径无回归）
- [~] P1.0 **Stage 0**：按用户决定**跳过**，直接做 Stage 1
- [x] P1.1 Stage 1 HAM 4 臂 × 5 折（20/20 成功；worst-group 方向与 Fitzpatrick 相反、Holm 后全 n.s.）
- [x] P1.2 Stage 1 Fitzpatrick 4 臂 × 5 折（`slurm/p_pred_attr.sh`，gpus48，bs128 等-batch）
- [x] P2.1 OOF averaging 分析脚本 `scripts/p_pred_attr_analysis.py`（复用 build_oof_results_averaging
      的 Spec/Views，方法表独立，不写入共享 selected_configs.json）
- [x] P2.2 报告 `docs/predicted_attribute_hyperadapt_results.md`（Fitzpatrick 完成；全部预注册假设未获支持，副产品：零信息对照臂解释了 HyperAdapt 的 worst-group 微增益）
- [ ] （可选）P3.1 τ 扫描；P3.2 pred 臂独立 Pareto 搜索；P3.3 MIMIC 阴性对照；
      P3.4 k-means 伪属性（语义无关软路由）对照；P3.5 联合训练变体 P-B（§3.5）
- [x] P-extra 五库全覆盖（MIMIC / CheXpert / PAPILA 亦完成，100 run 全成功）
- [x] P-extra2 **HAM 双属性（sex+age）条件化版**（开放项 5 的落实）：`--cond sex_age`，4 臂 × 5 折 + 3 臂 × 5 折 × 2 replicate = 50 run，与 age-only 臂 config-matched（lr3e-05_wd1e-04）；50/50 run 成功，**结论 null、与 age-only 臂逐条一致**，报告 `docs/predicted_attribute_hyperadapt_ham_sexage.md`

---

## 9. 待拍板的开放项

1. **数据集范围**：HAM+Fitzpatrick（推荐）／只 HAM／再加 MIMIC 阴性对照。
2. ~~g 的档位~~ **已定**：冻结特征 + logistic probe，在 train 上直接训练、in-sample 出 p̂，g+HyperAdapt 视为单一方法（§3.2）。
3. **超参**：复用 GT 臂选定 config（推荐，单变量对照 + 算力中性）／为 pred 臂独立搜 6 配置。
4. **是否先做 Stage 0**（推荐做，近乎免费且能预判 Stage 1 读数）。
5. ~~**是否把 HAM 的 sex 轴一并条件化**~~ **已定（2026-09-01）**：主线三 HN 已在
   `docs/ham_sexage_conditioning_arm.md` 补齐 sex+age 对照臂，为对齐口径，pred 线同样补跑
   `--cond sex_age`（age-only 臂保留不变，两臂 config-matched，唯一差异=条件输入是否含 sex）。
   见 `docs/predicted_attribute_hyperadapt_ham_sexage.md`。
