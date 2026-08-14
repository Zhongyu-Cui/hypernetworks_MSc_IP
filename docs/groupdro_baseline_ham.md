# GroupDRO 基线（HAM10000，ID，CV-OOF regime）

> 状态：**已完成**（2026-08-11）。作业 **75498**（gpus24，array 0–29，30/30 成功、0 错误）。
> 实现：`src/training/harness/groupdro.py` + `src/training/train_ham10000_groupdro.py`。
> 评估口径：与既有 OOF regime **完全一致**（折均 marginal-worst 选 config → averaging 逐折算再平均
> → 病灶级配对 cluster bootstrap），未新增任何口径。

---

## 1. 为什么加这个基线

导师建议：*"I would highly recommend you to run at least one baseline (e.g., GroupDRO) which uses
subgroup information for your thesis."*

协议原三基线中 ERM / SWAD 完全不看属性，ROC 只在**已训练模型的分数上**做事后阈值调整（零训练）。
于是「训练时使用子群标签」这一格里**只有 3 个 HN** ⇒「HN 用属性 vs 基线不用属性」是**混淆对比**：

|                | 不用属性     | 用属性                                  |
|----------------|-------------|----------------------------------------|
| 改损失 / 权重   | ERM, SWAD   | **GroupDRO ← 本次补上**                 |
| 改决策阈值      | —           | ROC                                    |
| 改函数（条件化）| —           | HyperHead / HyperFusion / HyperAdapt   |

两条通路对信号的要求不同，构成**可检验的预测**：conditioning 要兑现增益**必须** `I(Y;A|X)>0`
（否则属性盲的池化模型已是 Bayes 最优）；reweighting **不需要**条件信号——它只是把 Overall 换成
worst-group。故「HN 打平 ERM，而 GroupDRO 仍能抬 worst-group」在原则上完全可能，本实验即检验之。

## 2. 实现要点（权威见 `harness/groupdro.py` 模块文档）

Sagawa et al., ICLR 2020（arXiv:1911.08731）的 online greedy GroupDRO，逐 batch：

```
per-sample BCE → 组内均值 L_g（batch 内空组记 0）
q_g ← q_g · exp(η·(L_g + C/√n_g))，归一化        # 对抗侧，no_grad + detach
robust_loss = Σ_g q_g · L_g                       # 模型侧反传
```

- **分组 = 评估侧 canonical 交叉子群**：`GroupIndexer` 直接复用 `harness.subgroup_auc.subgroup_masks`
  的交叉键（HAM = Sex×Age 8 组），保证**优化的组 = 报告 worst-group 所 min 的组**，且构造时断言
  互斥完备（`age_group==-1` 会显式报错而非静默漏样本）。
- **组均匀采样（默认开）**：论文 Algorithm 1 第一步即 `g ~ Uniform(1..m)`，官方实现为
  `--reweight_groups`（与 `--robust` 同用）⇒ 它是方法的组成部分。复用
  `src/utils/resampling.build_group_label_sampler(dims=[sex,age], alpha=1.0)`，每 epoch 抽样数不变。
  实测最小格 Female|80+（train n=133）获 6.4× 过采样，每 batch 期望 16 张、空组概率 3.8e-8
  （若关掉：期望 2.5 张、**8% 的 batch 完全缺席**，这正是组均匀采样要解决的抖动问题）。
- **η 固定 0.01（论文默认），不进超参网格**；C=0。否则 GroupDRO 比其它 6 个方法多一个选择自由度。
- 训练目标经 `run_training(objective=...)` 注入，**val/test 评估、早停、SWAD loss 谷一律仍用普通 BCE**，
  模型选择口径与全部其它方法逐位一致。

**与 ERM 的差异清单**（除损失+采样外全部对齐：同 backbone、同 transforms、同 AdamW 网格、
同 batch=128、同早停 patience=5、同双 checkpoint）：
1. 组均匀采样（方法组成部分，见上）；
2. 三 split 过滤 `age_group>=0`（fold0: train 6969→6807，少 2.3%）——与 3 个 HN 训练脚本同口径，
   评估侧本就过滤，不引入新分叉；
3. η/C 为方法自身超参，固定不搜。

## 3. 实验设置

| 项 | 取值 |
|---|---|
| 超参搜索 | lr∈{3e-5,1e-4,3e-4} × wd∈{1e-4,1e-3} = 6 配置 × 5 折 = 30 run（同其它方法） |
| 折↔seed | fold k ↔ seed 42+k（既有 CV 约定） |
| config 选择 | 折均 **marginal-worst**（S1，`scripts/select_config_cv.py`），平手用折均 Overall 破 |
| 评估 | **averaging**（逐折算 AUC → 折间平均 → 再 min/max 取组），n=9,707 / 7,280 病灶簇 |
| 推断 | 病灶级配对 cluster bootstrap（1000 次，全方法共用同一批重采样索引） |

**选定配置 `lr3e-04_wd1e-03`**（折均 marginal-worst 0.8456 / Overall 0.9065），
即网格中 **wd 最强的一档**——与 Sagawa 论文「GroupDRO 需要强正则才不退化」的处方方向一致
（ERM 亦选中同一配置，故两者的 lr/wd 完全可比）。

| config | 折均 marg-worst | 折均 Overall |
|---|---|---|
| **lr3e-04_wd1e-03** | **0.8456** | 0.9065 |
| lr3e-05_wd1e-04 | 0.8311 | 0.8978 |
| lr3e-05_wd1e-03 | 0.8285 | 0.8981 |
| lr1e-04_wd1e-03 | 0.8243 | 0.8964 |
| lr1e-04_wd1e-04 | 0.8115 | 0.8984 |
| lr3e-04_wd1e-04 | 0.7670 | 0.8962 |

## 4. 结果

### 4.1 主表（averaging，n=9,707）

| 方法 | Overall | canonical worst | marginal worst | gap |
|---|---|---|---|---|
| ERM | 0.8938 | 0.7877 | 0.7976 | 0.1348 |
| **SWAD** | **0.9129** | **0.8331** | **0.8433** | 0.1024 |
| ROC | 0.8841 | 0.7744 | 0.7821 | 0.1452 |
| **GroupDRO** | 0.8813 | 0.7816 | 0.8128 | 0.1237 |
| HyperHead | 0.8941 | 0.7712 | 0.8159 | 0.1441 |
| HyperFusion | 0.8996 | 0.8171 | 0.8310 | **0.0923** |
| HyperAdapt | 0.8910 | 0.7990 | 0.8217 | 0.1190 |

### 4.2 GroupDRO 的配对比较（Δ 与 95% CI，`*`=显著）

| 比较 | ΔOverall | Δcanonical worst | Δmarginal worst |
|---|---|---|---|
| GroupDRO − ERM | **−0.0124** [−0.022,−0.003]* | −0.0061 [−0.122,+0.079] | +0.0152 [−0.041,+0.053] |
| GroupDRO − SWAD | **−0.0315** [−0.040,−0.023]* | −0.0514 [−0.163,+0.047] | −0.0305 [−0.076,+0.004] |
| HyperHead − GroupDRO | **+0.0128** [+0.003,+0.023]* | −0.0104 [−0.110,+0.123] | +0.0030 [−0.051,+0.057] |
| HyperFusion − GroupDRO | **+0.0182** [+0.008,+0.028]* | +0.0354 [−0.092,+0.151] | +0.0182 [−0.051,+0.070] |
| HyperAdapt − GroupDRO | +0.0097 [−0.001,+0.020] | +0.0173 [−0.121,+0.131] | +0.0089 [−0.024,+0.063] |

### 4.3 逐子群 AUC（averaging）——机制证据

| 子群 (train n) | ERM | SWAD | **GroupDRO** | HyperHead | HyperFusion | HyperAdapt |
|---|---|---|---|---|---|---|
| sex:Male | 0.8894 | 0.9129 | 0.8750 | 0.8906 | 0.8949 | 0.8838 |
| sex:Female | 0.8940 | 0.9075 | 0.8852 | 0.8945 | 0.9016 | 0.8947 |
| age:20-40 | 0.8374 | 0.8577 | 0.8217 | 0.8159 | 0.8310 | 0.8505 |
| age:40-60 | 0.9142 | 0.9303 | 0.8967 | 0.9050 | 0.9094 | 0.9075 |
| age:60-80 | 0.8678 | 0.8902 | 0.8606 | 0.8681 | 0.8711 | 0.8643 |
| **age:80+** | 0.7976 | 0.8433 | **0.8128** | 0.8401 | 0.8448 | 0.8217 |
| M\|20-40 (510) | 0.8367 | 0.8752 | **0.7816** ← 新最弱格 | 0.7712 | 0.8171 | 0.8055 |
| M\|40-60 (1563) | 0.9039 | 0.9234 | 0.8855 | 0.8938 | 0.9080 | 0.8926 |
| M\|60-80 (1287) | 0.8755 | 0.8992 | 0.8624 | 0.8702 | 0.8699 | 0.8625 |
| **M\|80+ (357)** | **0.7877** ← ERM 最弱格 | 0.8331 | **0.8083** | 0.8536 | 0.8374 | 0.8284 |
| F\|20-40 (678) | 0.8501 | 0.8489 | 0.8544 | 0.8374 | 0.8457 | 0.8535 |
| F\|40-60 (1566) | 0.9225 | 0.9354 | 0.9053 | 0.9154 | 0.9082 | 0.9180 |
| F\|60-80 (713) | 0.8470 | 0.8722 | 0.8547 | 0.8630 | 0.8720 | 0.8674 |
| F\|80+ (133) | 0.8042 | 0.8482 | 0.8190 | 0.7949 | 0.8447 | 0.7990 |

**对抗权重 q 确实活跃**（非退化）：选定配置 5 折训练末 q 展开到 `[0.072, 0.230]`（均匀=0.125），
最大/最小约 **3×**；各折被顶起的组不同（fold0/2 Male|40-60、fold1/4 Female|40-60、fold3 Male|60-80）。

## 5. 解读

**① GroupDRO 按设计工作了，但没有抬起 worst-group。** 逐子群表清楚显示它做了正确的事：把 ERM 的
最弱格 Male|80+ 从 0.7877 抬到 0.8083（+0.021）、Female|80+ +0.015、age:80+ 边缘 +0.015，代价是几乎
所有多数格下降。但 **canonical worst 反而 −0.006**——因为**最弱格换人了**：Male|20-40（train n=510）
从 0.8367 掉到 0.7816，成为新的地板。这是 minimax 目标在小格噪声下的典型行为：抬起当前最弱组之后，
下一个小格立刻接管地板，"抬地板"变成"挪地板"。

**② 付了 Overall 的代价却没买到公平。** Overall −0.0124（**显著**）、marginal worst +0.015（n.s.）、
canonical worst −0.006（n.s.）。robust optimization 的代价如期出现，收益没有。

**③ 混淆被解除——这是本实验对论文的核心贡献。** 现在有了同样在训练时看到子群标签的基线：
- 3 个 HN 相对 GroupDRO 在 **worst-group 上全部 n.s.**（+0.003 ~ +0.035）；
- HN 在 **Overall 上显著优于 GroupDRO**（HyperHead +0.013、HyperFusion +0.018，均显著）。

⇒ 「HN 打平/不劣于基线，是不是因为它偷偷靠属性占便宜？」这个质疑可以正面回答：**同样吃属性的
GroupDRO 并没有更好，反而在 Overall 上显著更差**。HN 的表现不是"用了属性"带来的，而属性本身
（无论以 reweighting 还是 conditioning 的形式使用）在 HAM 上都没能兑换成 worst-group 公平。

**④ 与主结论一致并强化之。** 五个方法里唯一**显著**抬起 worst-group 的仍是 **SWAD（+0.0457 vs ERM）**,
而它**完全不看属性**。结合闸门读数（HAM-age 在 OOF V-information 下退回未检出 ≈0），本实验给出
一个新的、有功效的旁证：在条件信号 ≈0 的体制下，**属性信息不是 worst-group 公平的有效杠杆，
泛化/平坦性才是**。注意这条比原先的论证更强——它不再只是"我们的 HN 没赢"，而是"文献标配的
子群感知训练方法在同一 regime 下同样没赢"。

## 6. 局限（须在论文中如实写）

1. **η 未搜**（固定论文默认 0.01）。这是为公平比较刻意付出的代价；若 η 严重失配，本结论对 GroupDRO
   不利。缓解证据：q 展开到 3×，说明 η 既非过小（退化为 ERM）亦非过大（退化为只训一个组）。
2. **训练集少 2.3%**（排 `age_group==-1`），与 3 个 HN 同口径，但相对 ERM/SWAD/ROC 是一处差异。
3. **组均匀采样与对抗损失未拆开**。已实现 `--groupdro_sampler standard` 消融臂（method 记为
   `groupdro_nobal`，产物隔离），**尚未运行**；跑了才能分辨收益/代价各来自哪一半。
4. **单数据集**。HAM 是五库中唯一曾报到弱正 age 信号者，故是最有利于"属性有用"的场地；其余四库
   若要补 GroupDRO，训练脚本可直接仿写（harness 与分组器均已数据集参数化）。
5. 空组兜底（L_g=0、q 乘子 1）沿用官方实现，存在对缺席组的相对降权偏差；组均匀采样下
   P(空组)=3.8e-8 故无影响，但 `groupdro_nobal` 臂若要跑，应先补 count-aware EMA 或至少加空组计数诊断。

## 7. 复现

```bash
# 1) 训练（6 配置 × 5 折）
ssh biomedia-slurm
sbatch --partition=gpus24 --job-name=ham_gdro_cvsearch \
  --output=/vol/biomedic2/bglocker_studproj/zc125/logs/ham_gdro_cvsearch.%N.%A_%a.log \
  --export=ALL,PY_SCRIPT=/vol/.../src/training/train_ham10000_groupdro.py \
  /vol/.../slurm/c1_ham_cv_search.sh

# 2) 折均 marginal-worst 选 config，并**合并**进 selected_configs.json（勿覆盖实验 G 的键）
python scripts/select_config_cv.py --dataset ham10000 --method groupdro \
  --write-json outputs/ham10000/cv5/selected_configs.json --merge

# 3) averaging 口径评估（不带 --dataset 即全量重算；带则走新加的合并模式）
python scripts/build_oof_results_averaging.py
```

## 8. 本次附带的两处工程订正

1. **`build_oof_results_averaging.py --dataset X` 会抹掉其余 6 个 regime**（原实现从空 dict 起手后整体
   落盘）。本次踩到并修复为**合并模式**（以既有 JSON 为底只重算目标 regime）。此前若有人用过
   `--dataset` 跑过一次，落盘的 JSON 就只剩一个 regime——建议复核依赖该文件的下游文档。
2. **`hyperadapt` 的 CV 预测在 2026-08-04（实验 G 期间）被重写过**，导致 `oof_results_averaging.json`
   （Jul 16 生成）与磁盘上的预测文件已不一致。本次全量重算一并刷新：HAM hyperadapt
   canonical worst **0.7930 → 0.7990**、Overall 0.8873 → 0.8910；Fitzpatrick / MIMIC / CheXpert 的
   hyperadapt 亦有 0.002–0.006 量级漂移，**其它方法逐位不变**。旧文件已备份为
   `oof_results_averaging.json.pre_groupdro`。⚠️ `docs/oof_regime_results.md` 与
   `docs/crossdataset_worstgroup_fairness.md` 中的 HyperAdapt 数字需按新值复核（结论方向不变：
   全部为 n.s. 的小幅差异）。
