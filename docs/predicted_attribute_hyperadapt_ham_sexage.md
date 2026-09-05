# 实验 P 扩展：HAM10000 的 sex+age 双属性 Predicted-Attribute HyperAdapt

> 方案：`docs/predicted_attribute_hyperadapt_plan.md`（开放项 5 的落实）。
> 参照臂：age-only pred 线见 `docs/predicted_attribute_hyperadapt_results.md`；
> GT 侧的同类对照见 `docs/ham_sexage_conditioning_arm.md`。
> 分析：`scripts/p_pred_attr_analysis.py --dataset ham10000 --cond sex_age`、
> `scripts/p_sigma_train.py --dataset ham10000 --cond sex_age`。

---

## 1. 为什么补这一臂

实验 P 在 HAM 上只把 **age** 的 p̂ 喂给 HyperAdapt，而 worst-group 的评估口径是
**Sex / Age / Sex×Age 三套分组**——条件化口径与评估口径不对称，sex 轴与交叉格上的最差子群，
模型从未拿到对应属性。GT 侧已在 2026-08-28 用 `--cond sex_age` 把这个混淆实测排除
（`docs/ham_sexage_conditioning_arm.md`：90 run，null）。本臂在 **pred 线**做同一件事，
使两条线的条件化口径一致，且把「pred 线的 null 是不是因为少喂了 sex」这个替代解释一并排除。

---

## 2. 实验规格

| 项 | 取值 |
|---|---|
| 条件输入 | g 对 **sex（2 类）与 age_group（4 类）** 的温度校准 softmax，两份都进条件通路 |
| 模型 | `ResNet18HyperAdaptSoftSexAge`（条件通路 = `SoftSexAgeEmbedding`，期望嵌入 e=p̂ᵀE 后拼接） |
| 四臂 | `soft`（主线）/ `hard`（argmax one-hot）/ `perm`（逐轴独立行置换）/ `const`（train 先验） |
| method 名 | `hyperadapt_pred{,hard,perm,const}_sexage` |
| 超参 | **lr3e-05_wd1e-04（config_index 0）= 与 age-only pred 臂完全相同** |
| batch | 128（与 GT 臂等 batch） |
| 划分 / seed | cv5 lesion 级 GroupKFold，seed=42+fold（replicate 再 +100·trial） |
| 评估 | CV-OOF **averaging**，lesion 级配对 cluster bootstrap n=1000，**分组键恒用真值 sex/age** |
| 规模 | 4 臂 × 5 折 = 20 run（主结果）+ soft/hard/const × 5 折 × 2 trial = 30 run（σ_train），共 **50 run** |

**为什么不用 GT 双属性臂自选的 config**：`hyperadapt_sexage` 的 Pareto 选定配置是 lr3e-05_wd1e-03，
若 pred 双属性臂跟它，则「pred_sexage vs pred(age-only)」这个最核心的跨臂对比里会混进超参差异
（正是 `docs/ham_sexage_conditioning_arm.md` §4.3 踩过的坑）。本臂固定跟 age-only pred 臂同 config，
使跨臂 Δ 的唯一来源是条件输入；对 GT 双属性臂的 P-H1 对比则同时给出两种口径
（协议口径 = GT 用自选 config；`--gt-config-matched` = GT 也取 lr3e-05_wd1e-04）。

### 子群分类器 g

p̂ 由 `scripts/build_attr_predictions.py --dataset ham10000_sexage` 生成，落在**独立目录**
`outputs/ham10000/cv5/attr_pred_sexage/`（不改写支撑 age-only 已冻结结果的 `attr_pred/`）。
特征缓存与 age-only 臂同一份（冻结 ImageNet ResNet-18 512-d），probe 规程不变，
故**该目录里的 `prob_age` 与 age-only 目录逐位相同（已核对，max|Δ|=0）**——
两臂唯一的差异就是多了一份 sex 的 p̂。

| 轴 | 温度 T | acc train / test | 熵 train / test | ECE test |
|---|---|---|---|---|
| sex | 1.47–1.83 | 0.734–0.740 / 0.665–0.676 | 0.58–0.61 / 0.58–0.61 | 0.011–0.031 |
| age | 1.74–2.00 | 0.625–0.638 / 0.479–0.499 | 1.10–1.13 / 1.10–1.13 | 0.020–0.044 |

in-sample 尖锐度诊断（train−test accuracy 差）：sex **+0.066±0.005**、age **+0.142±0.013**；
熵差（test−train）两轴都 ≈0（|Δ|≤0.01）⇒ 训练/测试的条件输入分布几乎不漂移，
**没有 PAPILA 那种 in-sample 泄漏退化**（该库 probe 完美分离，见主结果 §6）。
sex 的 test accuracy 0.67 高于多数类基线（≈0.54）⇒ g 确实从皮肤镜图像里读出了一些性别信息。

---

## 3. 结果

**规模**：50/50 run 全部成功（作业 78860/78861/78867/78868 = 四臂 trial 0；78869–78874 =
soft/hard/const 的 trial 1/2），无 CUDA / OOM / 收尾回调异常。
产物：`outputs/ham10000/cv5/{predictions,p_pred_attr_sexage_results.json,
p_pred_attr_sexage_results_gtmatched.json,p_sigma_train_sexage.json}`。

### 3.1 点估计（CV-OOF averaging，n=9,707，cluster=7,280 lesion）

| 方法 | Overall | worst(canon) | worst(marg) | gap |
|---|---|---|---|---|
| erm | 0.8938 | 0.7877 | 0.7976 | 0.1348 |
| swad | **0.9129** | **0.8331** | **0.8433** | 0.1024 |
| groupdro | 0.8813 | 0.7816 | 0.8128 | 0.1237 |
| hyperadapt（GT, age） | 0.8910 | 0.7990 | 0.8217 | 0.1190 |
| hyperadapt_sexage（GT, sex+age） | 0.8980 | 0.7820 | 0.8253 | 0.1390 |
| hyperadapt_pred（age-only） | 0.8937 | 0.8118 | 0.8364 | 0.0974 |
| hyperadapt_predconst（age-only） | 0.8874 | 0.7860 | 0.8184 | 0.1258 |
| **hyperadapt_pred_sexage** | 0.8940 | 0.7947 | 0.8227 | 0.1171 |
| hyperadapt_predhard_sexage | 0.8939 | 0.8071 | 0.8212 | 0.1042 |
| hyperadapt_predperm_sexage | 0.8942 | 0.7961 | 0.8263 | 0.1176 |
| hyperadapt_predconst_sexage | 0.8955 | 0.7931 | 0.8252 | 0.1218 |

**四个 sexage 臂彼此落在 worst(canon) 0.793–0.807 / Overall 0.8939–0.8955 的带内**——
含两个**不携带任何样本级属性信息**的对照臂（perm / const）。与 age-only 线、与 CXR 两库的读数
是同一个模式。

### 3.2 主要终点（worst-group canonical）的预注册 family

| 对比 | Δ | 95% CI | p_raw | p_holm |
|---|---|---|---|---|
| pred_sexage − hyperadapt_sexage（P-H1） | +0.0127 | [−0.062, +0.051] | 0.998 | 1.000 |
| pred_sexage − erm（P-H2） | +0.0070 | [−0.143, +0.076] | 0.878 | 1.000 |
| pred_sexage − swad | −0.0383 | [−0.172, +0.046] | 0.276 | 1.000 |
| pred_sexage − predperm_sexage（P-H3） | −0.0013 | [−0.036, +0.015] | 0.682 | 1.000 |
| pred_sexage − predconst_sexage（P-H3） | +0.0016 | [−0.030, +0.024] | 0.784 | 1.000 |
| predhard_sexage − pred_sexage（软 vs 硬） | +0.0123 | [−0.036, +0.057] | 0.868 | 1.000 |

**6 个预注册对比 Holm 后无一显著**（p_raw 最小 0.276）。与 age-only 臂（p_raw 最小 0.216）同判。
P-H1 的 config-matched 敏感性（GT 也取 lr3e-05_wd1e-04）几乎不动：Δ=+0.0122，
Overall −0.0033（协议口径 −0.0040）⇒ 该对比不受选参影响。

### 3.3 核心跨臂对比：条件输入加不加 sex

两臂 config-matched、split/seed 相同、`prob_age` 逐位相同，唯一差异是多一份 sex 的 p̂：

| 指标 | Δ = pred_sexage − pred(age-only) | 95% CI | p | σ_train（3 trial） | \|Δ\|/σ |
|---|---|---|---|---|---|
| Overall | +0.0003 | [−0.005, +0.005] | 0.908 | 0.0022 | 1.16 |
| worst(canon) | −0.0171 | [−0.062, +0.028] | 0.356 | 0.0149 | 0.58 |
| worst(marg) | −0.0137 | [−0.042, +0.016] | 0.288 | 0.0100 | 0.25 |
| gap | +0.0197 | [−0.025, +0.068] | 0.318 | 0.0201 | 0.57 |

**全部 n.s.，worst-group 两个口径都是（不显著的）负号**。3-trial 配对差把 Overall 从单次的
+0.0003 抬到 +0.0025（σ=0.0022，比值 1.16，刚过噪声线）——两次估计差一个量级本身就说明这点
Overall 差由训练随机性支配；即便按 +0.0025 算，「加 sex」也只值 0.25 个百分点，而在 worst-group
上连方向都不占优。

### 3.4 σ_train：50 run 中 30 run 用于训练噪声

单臂跨-trial 波动（3 trial 均值±SD）：

| 方法 | Overall | worst(canon) | worst(marg) |
|---|---|---|---|
| pred_sexage | 0.8949±0.0010 | 0.7917±0.0168 | 0.8240±0.0028 |
| predhard_sexage | 0.8926±0.0050 | 0.7919±0.0207 | 0.8209±0.0060 |
| predconst_sexage | 0.8949±0.0012 | 0.7817±0.0177 | 0.8263±0.0076 |
| pred（age-only） | 0.8924±0.0019 | 0.8004±0.0106 | 0.8266±0.0100 |
| predconst（age-only） | 0.8906±0.0036 | 0.7898±0.0071 | 0.8153±0.0069 |

**worst(canon) 的单臂 σ 就有 0.007–0.021**，而三个 sexage 臂的 3-trial 均值只差 0.010
（0.7817–0.7919）⇒ 臂间差异整体在训练噪声之内。主要终点上，`pred_sexage` 对
`predconst_sexage` 的 3-trial 配对差 +0.0100（σ=0.0076，比值 1.33）是唯一略越训练噪声的
worst-group 读数，但它的 bootstrap CI 是 [−0.030,+0.024]、p=0.784 ⇒ **评估不确定性下毫无支持**，
两道关卡没有同时通过。

### 3.5 三处 bootstrap 显著读数的处置

1. `pred_sexage − swad` Overall **−0.0188**（p=0.001）：SWAD 在 HAM 上一贯最强，与既有结论一致，
   不是本臂的新信息。
2. `pred_sexage − groupdro` Overall **+0.0127**（p=0.006）：由 `groupdro − erm` = **−0.0124**
   （p=0.012）驱动，即 GroupDRO 自己更差，与 `docs/groupdro_baseline_5datasets.md` 一致。
3. `predconst_sexage − predconst(age-only)` Overall **+0.0081**（p=0.002，|Δ|/σ_train=1.19）：
   **两个都是零信息臂**，差异只可能来自条件通路多出的 256 个 fuse 参数与不同随机初始化。
   若真是容量效应，配对的 soft 对应当同幅上移，实测只有 **+0.0003**；且 3-trial 均值把它压到
   +0.0042。⇒ 判为不稳定的训练实现差异，**不构成任何「属性信息有用」的证据**（恰相反：
   能显著动 Overall 的那一对，条件输入里连样本级信息都没有）。

---

## 4. 结论

1. **null，与 age-only 臂同判**。把实验 P 的条件输入从 age 的 p̂ 补齐为 sex+age 的 p̂ 后，
   主要终点（worst-group canonical）的 6 个预注册对比 Holm 后仍无一显著，且
   pred 与 perm/const 两个零信息臂的差 ≤0.002。
2. **「pred 线只喂了 age」这个替代解释被实测排除**。条件化口径补齐到评估分组变量全集后，
   worst-group 两个口径的变化都是不显著的负号（−0.017 / −0.014），Overall 变化 +0.0003。
   这与 GT 线的同类对照（`docs/ham_sexage_conditioning_arm.md`，三 HN 全 null）逐条呼应：
   **在 HAM 上，条件化 age 还是 sex+age，不改变任何结论**。
3. **g 确实读到了 sex**（test acc 0.665–0.676 vs 多数类 0.54）却没用上——与 MIMIC「三轴都预测得准
   （0.78–0.86）却对任务毫无帮助」是同一现象的第三个数据点，符合 OOF conditional V-information
   闸门对 HAM sex/age 两轴的读数（皆无可用信号）。
4. **写作口径**：本臂可与 `docs/ham_sexage_conditioning_arm.md` 合并声明——
   「无论真值属性还是预测属性、无论条件化 age 还是 sex+age，HAM 上 HN 的 worst-group 都打平」，
   不必再把「pred 线只条件化 age」当作设计限制声明。⚠️ 不要把 §3.5 的三处显著读数当正面结果引用。
