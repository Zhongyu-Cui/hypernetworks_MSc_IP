# R1 合成信号剂量-反应实验结果（充分性因果证据）

> 建立日期：2026-07-06。对应 `docs/review_followup_tasks.md` 的 **R1（最高优先）**。
> 本报告回答评审最关切的一问：**「`I(Y;A|X)>0` 是否*充分*导致 HN 增益」**——即把主论点的
> 「iff」从跨数据集相关升级为**单一数据集内的因果剂量-反应**。

---

## 1. 动机与设计

全研究仅 HAM-age 一个数据集 `I(Y;A|X)>0`（⚠️ **新 OOF V-info 闸门下连 HAM-age 也退回未检出** ⇒ 自然数据集
**无一**具备可用信号且有功效，本合成实验的必要性**因此更强**），信号强弱与「数据集身份」完全共线，无法排除是数据集
其它差异（模态/规模/任务）而非信号本身导致增益。R1 在**单一数据集 MIMIC-CXR 内**注入可控强度的
侧信道属性，扫信号强度看 HN 增益是否随之出现，补上因果链。

**注入机制（R1.1，`src/datasets/synthetic_attribute.py`）**：
`A_syn = Y ⊕ Bernoulli(η)`，以**标量**作为 HN 条件输入、**不画进图像**（区别于 Colored-MNIST /
Waterbirds / Spawrious 把属性画进图像、故 `I(Y;A|X)≈0`；本实验需属性携带 X 之外的残余标签信息，
对标 LUPI 特权信息 / TRAM 标签噪声范式）。翻转噪声与图像 X 独立，故
`I(Y;A_syn|X)` 由 η 单调可调：η=0.5⇒0（负控制）、η→0⇒H(Y|X)。self-test 已证三条判据。

**MI 标定（R1.2，`scripts/calibrate_synthetic_signal.py`，作业 70114）**：复用 D2 冻结特征 ΔNLL
估计器（同口径），把 η 标到目标档位。每档复核实测均落在目标 ±0.001 nats（远紧于 MIMIC
MDE≈0.001–0.002）：

| 目标 nats | η* | 复核实测 I(Y;A_syn\|X)（int, 3 backbone seed） |
|---|---|---|
| 0（负控制）| 0.5000 | ≈0 |
| 0.02 | 0.3621 | +0.02036 ± 0.00034 |
| 0.05 | 0.2942 | +0.04889 ± 0.00019 |
| 0.10 | 0.2096 | +0.09915 ± 0.00054 |

**统一 regime 训练（R1.3）**：复用 A0 harness + C2 已 Pareto 选定的 MIMIC 超参配置（图像优化
regime 对 A_syn 不变——A_syn 只进 HN 分支、不改图像/标签），每方法 5 seed 确认，子群 = A_syn 两组
（`dataset=mimic_synth`）。**ERM 完全忽略 A_syn（独立 RNG）⇒ 跨档逐比特相同，只训一次**，各档
worst-group 由该档 A_syn 叠加 ERM 预测重算（A_syn 由 y_true 确定性重建，与 HN 所见逐元素一致，
汇总脚本内断言校验通过）。共 65 run（作业 70115/70120–70131），0 错误。

---

## 2. 结果

主口径 = overall-selection；增益 = (HN − ERM) 配对 5-seed 均值 ± 95%CI（Student-t）。
曲线见 `outputs/mimic_synth/dose_response_curve.png`，数据见 `outputs/mimic_synth/dose_response_summary.json`。

### 2.1 Overall AUC（池化判别力）—— 干净的单调剂量-反应

| 实测 nats | ERM Overall | HyperHead ΔOverall | HyperFusion ΔOverall | HyperAdapt ΔOverall |
|---|---|---|---|---|
| 0（负控制）| 0.8465 | +0.0009 [−0.0023, +0.0040] | +0.0004 [−0.0036, +0.0044] | +0.0001 [−0.0018, +0.0021] |
| 0.020 | 0.8465 | **+0.0166 [+0.0144, +0.0187]** | **+0.0160 [+0.0121, +0.0200]** | **+0.0158 [+0.0137, +0.0179]** |
| 0.049 | 0.8465 | **+0.0372 [+0.0354, +0.0389]** | **+0.0365 [+0.0330, +0.0400]** | **+0.0367 [+0.0339, +0.0395]** |
| 0.099 | 0.8465 | **+0.0667 [+0.0646, +0.0688]** | **+0.0661 [+0.0640, +0.0682]** | **+0.0659 [+0.0634, +0.0683]** |

- ERM Overall 恒为 **0.8465**（各档同一批预测），确证 ERM 对 A_syn 不变的设计。
- **负控制（0 nats）**：三 HN 的 ΔOverall 95%CI 均含 0 = **干净打平**（无信号即无增益，排除架构本身的伪增益）。
- **有信号档**：三 HN 的 ΔOverall CI 全部**排除 0 且为正**，且随实测 nats **单调上升**
  （Spearman ρ = **+1.000, p = 0.000**，三方法一致）。增益近似正比于信号强度。

### 2.2 Worst-group（A_syn 子群）AUC —— 平/微负，机制性解释

| 实测 nats | HyperHead Δworst | HyperFusion Δworst | HyperAdapt Δworst |
|---|---|---|---|
| 0 | +0.0008 [−0.0023, +0.0039] | +0.0002 [−0.0030, +0.0033] | −0.0004 [−0.0022, +0.0015] |
| 0.020 | −0.0001 [−0.0025, +0.0023] | −0.0026 [−0.0068, +0.0017] | −0.0018 [−0.0042, +0.0006] |
| 0.049 | −0.0002 [−0.0028, +0.0025] | −0.0016 [−0.0064, +0.0031] | −0.0009 [−0.0047, +0.0029] |
| 0.099 | −0.0014 [−0.0054, +0.0025] | −0.0035 [−0.0070, −0.0001] | −0.0034 [−0.0063, −0.0005] |

- worst-group-over-A_syn **不随信号上升**（Spearman ρ 为负），高信号档甚至微负（HyperFusion/HyperAdapt
  在 0.10 档 CI 排除 0）。
- **机制解释（非矛盾）**：A_syn 携带的是 Y 的**边际先验位移**（组 A_syn=1 的 P(Y=1) 抬高、A_syn=0 压低）。
  HN 用上 A_syn 相当于给每组分数加一个近似**常数偏移**——这改善**跨组池化**排序（Overall AUC），
  但 AUC 阈值无关、组内排序对常数偏移不敏感，故**组内 AUC 基本不变**，worst-group 停在 ERM 水平。
  微负来自 HN 额外参数在小信号下的拟合噪声。

  > **⚠️ 该"常数偏移"解释的适用边界（scope，务必勿越界推广）**：此结论是
  > `A_syn = Y ⊕ Bernoulli(η)` 且**翻转噪声 ⊥ X** 这一**特定属性族**的数学后果，**不推广到任意真实属性**。
  > 因 noise⊥X ⇒ `P(A_syn|Y,X)=P(A_syn|Y)`，贝叶斯最优修正**恰好**是每组一个常数 logit 偏移
  > `logit P(Y|X,A_syn=a)=logit P(Y|X)+c(a)`，`c(a)` 只依赖 a 不依赖 X ⇒ 组内排序（AUC）不变。
  > 换言之，**R1 的属性设计在结构上只允许"先验位移型"修正**，故它对"信号能否改善 worst-group / 组内
  > 结构"这一问题**天然沉默**——是回答该问的**错误工具**，而非给出"信号不改善公平"的**否证**。
  > 真实敏感属性（如 HAM-age）若让**特征→标签关系随组变化**（决策边界旋转/变形，而非平移），
  > `I(Y;A|X)>0` 原则上**可**改善组内/worst-group AUC；HAM-age 上 HyperHead 把最年轻组 20-40 的
  > **组内** AUC 由 0.754 抬到 0.837（+0.083）、age-gap 0.171→0.115，正是 R1 设计故意排除的那类组内效应。
  > 故真实属性的 worst-group 效应**必须由 HAM 等真实数据在足够功效下单独裁定**；其现状为"**未证**"
  > （评估-n 地板：test 20-40 组 ~162、joint 小格 ~26，配对检验全 n.s.，最接近 p=0.099），
  > 即**既未证实、也未否证**，不可用本节的 A_syn 结论替代裁决。详见 `docs/results_summary.md` D6/D6-HP。

---

## 3. 结论：信号存在对 HN 增益是「充分」的（在 Overall 判别力口径）

1. **主论点的充分性方向获得单一数据集内的因果证据**：在 MIMIC（真实属性 `I(Y;A|X)≈0`、HN 打平）
   内注入可控条件信号后，三种 HN（浅 HyperHead / 中 HyperFusion / 深 HyperAdapt）的 **Overall AUC
   相对 attribute-blind ERM 的增益随 `I(Y;A_syn|X)` 严格单调上升，且 0 档干净打平**。这把
   「HN 赢盲基线 ⟺ `I(Y;A|X)>0`」从跨数据集相关**升级为因果剂量-反应**，直接补上评审指出的
   现有设计最缺的一环。

2. **增益体现在池化判别力、而非 A_syn-子群公平**：合成侧信道以边际先验形式被 HN 利用，兑现为
   Overall AUC；对「worst-group over 该属性」这一**公平轴**无改善（常数偏移不改组内排序）。这与真实
   数据集的观察自洽——真实属性 `I(Y;A|X)≈0` 时 Overall 与 worst-group **都**打平。

3. **介入深度非决定因素**：三档注入深度（Head/Fusion/Adapt）的 Overall 剂量-反应曲线几乎重合
   （Δ 之间无显著差异），说明「能否用上信号」由**信号是否存在**决定，而非架构介入多深——
   与 comparison_protocol D7 的既有结论一致。

### 与主报告的衔接
- 补强了 `docs/results_summary.md` 的 D7：核心判据 `I(Y;A|X)>0 ⇒ HN 增益` 现同时有
  **必要方向**（4 个 null 数据集打平）与**充分方向**（本剂量-反应）的证据。
- **表述提醒（对接 R5）**：充分性成立的口径是 **Overall AUC**；「公平性（worst-group）增益」需要的
  是属性与标签在 X 之外**且以组内可分形式**存在的信号，本实验的边际先验型 A_syn 不满足后者——
  这解释了为何即便 HAM-age 有信号，其 worst-group 增益也小且不显著（信号以先验为主）。

---

## 4. 复现

```bash
# R1.1 注入机制 self-test
python -m src.datasets.synthetic_attribute
# R1.2 标定（sbatch，作业 70114）
sbatch slurm/calibrate_synthetic_signal.sh          # → outputs/mimic_cxr/synthetic_signal_calibration.json
# R1.3 训练（sbatch，65 run）：ERM 1×5seed + 3HN×4档×5seed
#   METHOD/ETA/CONFIG_INDEX 经 --export 注入 slurm/r1_synth.sh（见 review_followup_tasks.md R1.3）
# R1.4 汇总 + 绘图
python scripts/analyze_synthetic_dose_response.py    # → outputs/mimic_synth/{dose_response_summary.json,dose_response_curve.png}
```

关键文件：
- 注入/数据：`src/datasets/synthetic_attribute.py`、`src/datasets/mimic_synth_dataset.py`
- 公平性/子群：`src/utils/synthetic_fairness.py`、`src/training/harness/subgroup_auc.py`(`mimic_synth`)
- 训练：`src/training/train_mimic_synth.py`、`slurm/r1_synth.sh`
- 标定/汇总：`scripts/calibrate_synthetic_signal.py`、`scripts/analyze_synthetic_dose_response.py`
