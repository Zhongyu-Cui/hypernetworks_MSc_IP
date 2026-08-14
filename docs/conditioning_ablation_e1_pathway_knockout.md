# E1 敏感性实验②：通路 knockout × 属性置换（fc vs conv 的年龄依赖归因）

> **归属**：`docs/conditioning_ablation_plan.md` E1 的补充敏感性实验（承接实验①
> `conditioning_ablation_e1_worstcase_checkpoint.md`）。
> **动机**：E1 的 ρ_l 从**权重层面**指出条件化几乎全在 fc（conv 的 `ρ^between` 比 fc 小 45~60 倍）。
> 但 ρ 只量「权重被改动多少」，不直接等于「行为上依赖多少」——conv 的 ρ 虽小，会不会仍有实际功能？
> 本实验从**行为层面**（预测对 age 输入的依赖）直接检验：对 C-Deep / C-Full 各做两种通路 knockout，
> 再比较「真实 age vs 置换 age」的性能差 Δ。
>
> **一句话结论**：**年龄依赖主要由 fc 承担，conv 行为上近乎惰性。** 关掉 fc 条件化后，整体 AUC 层面
> 真实/置换 age 几乎无差别（Δoverall≈0）；worst-group 层面 C-Deep 也塌到噪声内。唯一残留是
> **C-Full 的 worst-group**：关 fc 后仍留 +0.0084，但仅 **1.6σ、未明显超过置换噪声**，至多是
> 「深层 conv 或有微弱 worst-group 功能」的**暗示性**证据，不构成 conv 有稳健功能的证明。

---

## 1. 实验设计

对含 conv 条件化的两个 cell（C-Deep、C-Full）做两种 **通路 knockout**（均置零生成器，精确、可逆、不改结构）：

| 变体 | 操作 | 语义 | 保留的通路 |
|---|---|---|---|
| `full` | 不动 | 两条通路都开（= 标准 E1 置换） | fc + conv |
| **`fc_off`** | 置零 `fc_A_gen`（weight+bias）⇒ a_fc=0 ⇒ **ΔW_fc=0** | fc 退化为**属性无关的共享 base 头** | **仅 conv** |
| **`conv_off`** | 置零 `A_gen`（shared-A，weight+bias）⇒ a_shared=0 ⇒ **M_conv=0** | 所有条件化 conv 退化为**普通卷积** | **仅 fc** |

> 数学等价：`adapted_conv2d(M=0)` = 普通卷积、`adapted_linear(ΔW=0)` = 普通线性
> （见 `resnet18_hyperadapt.py`）。故置零对应的生成器即精确关闭该通路的条件化，其余权重（已训练的
> backbone / base fc）保持不变。`A_gen.` 与 `fc_A_gen.` 用 `startswith` 精确区分（都含子串 "A_gen"）。

- 每个变体重算「每图 × 4 age 取值」的 [n,4] logits 表，再做**病灶级整组置换 age**（n_perm=20，与
  `e1_rho_and_permutation` 同一 rng）；**分组键恒用真实 age**；估计量 = averaging（逐折算指标→折间平均）。
- **checkpoint = best_overall**（E1 主结果、§4 ρ 表口径；实验①已证换 best_worstcase 不改变机制图景）。
- **判读逻辑**：
  - `fc_off` 下 Δ(真实−置换) = **conv 单独**的 age 依赖（fc 已不能响应 age）。→0 ⇒ age 依赖不在 conv。
  - `conv_off` 下 Δ = **fc 单独**的 age 依赖。
- **计算**：本地 GTX 1050 Ti（batch 32），**不排队**；全程约 50 min（弱 GPU 计算受限）。

> **内建自检（已通过）**：`full` 变体的 Δ 精确复现 `e1_permutation.json`（C-Deep +0.0202/+0.0245、
> C-Full +0.0147/+0.0188，逐位一致）⇒ knockout 管线与 canonical 置换管线完全一致。

---

## 2. 结果

Δ = 真实 age − 置换 age（正 = 打乱 age 会掉性能 = 依赖 age）。`σ` = Δworst / 20 次置换的 SD（衡量
是否超过置换噪声，非正式检验）。**保留比例** = 该变体 Δworst ÷ full 的 Δworst。

| cell | variant | Δoverall | Δworst | worst σ | Δworst 保留比例 |
|---|---|---|---|---|---|
| **C-Deep** | full | +0.0202 | +0.0245 | 3.1σ | 100%（基线） |
| | **fc_off**（仅 conv） | **−0.0010** | **+0.0018** | **0.9σ** | **7%** |
| | conv_off（仅 fc） | +0.0194 | +0.0211 | 3.3σ | 86% |
| **C-Full** | full | +0.0147 | +0.0188 | 2.2σ | 100%（基线） |
| | **fc_off**（仅 conv） | **+0.0019** | **+0.0084** | **1.6σ** | **45%** |
| | conv_off（仅 fc） | +0.0120 | +0.0161 | 2.7σ | 86% |

### 2.1 整体 AUC（Overall）——两个 cell 都是「fc 承担全部」
- **`fc_off` 后 Δoverall 塌到 ≈0**（C-Deep −0.0010、C-Full +0.0019）：关掉 fc 条件化，模型在整体 AUC
  层面**不再依赖 age**。⇒ **Overall 的年龄依赖 100% 由 fc 承担，conv 行为上惰性。**
- `conv_off`（仅 fc）几乎保住 full 的 Δoverall（C-Deep 96%、C-Full 82%）⇒ 与上一致。

### 2.2 worst-group——C-Deep 同样在 fc；C-Full 有微弱残留但未过噪声
- **C-Deep**：`fc_off` 后 Δworst 从 +0.0245 塌到 **+0.0018（0.9σ，噪声内）**⇒ layer4 conv 单独
  **不使用 age**；worst-group 的年龄依赖也几乎全在 fc（`conv_off` 保留 86%、3.3σ）。
- **C-Full**：`fc_off` 后 Δworst 仍留 **+0.0084（full 的 45%）**，但**仅 1.6σ、未明显超过置换噪声**。
  ⇒ layer1–4 conv 合起来**或**对 worst-group 有微弱 age 功能，但证据**弱、不稳健**，不能据此断言
  「conv 有实际功能」。`conv_off`（仅 fc）保留 86%（2.7σ）⇒ 即便在 C-Full，fc 仍是主承担者。

### 2.3 一个附带观察：C-Full 两通路部分冗余
C-Full 的 worst-group 保留比例 **fc_off 45% + conv_off 86% = 131% > 100%**（C-Deep 为 7%+86%=93%≈100%
可加）。>100% 说明两条通路**不是简单可加、存在冗余**：都各自编码了一部分相同的 age 信号，关掉其一时
另一条能部分补偿。⚠️ 但因 C-Full `fc_off` 仅 1.6σ，此冗余观察**不宜过度解读**。

---

## 3. 结论

1. **回答用户的二分判据**：
   - **「关闭 fc 后真实/置换 age 几乎无差别」→ 成立**（C-Deep worst 0.9σ、两 cell 的 Overall Δ≈0）
     ⇒ **年龄依赖主要由 fc 承担**。这从**行为层面**独立印证了 ρ_l 的**权重层面**结论。
   - **「仍有明显差异 ⇒ conv 虽 ρ 小却有实际功能」→ 仅 C-Full 的 worst-group 有微弱迹象**
     （+0.0084），但 **1.6σ 未过置换噪声**，属**暗示性**而非确证。C-Deep 的 conv 则行为上完全惰性。
2. **ρ 与行为一致，且行为层面更细**：ρ_l 说 conv 的属性特异偏移仅 fc 的 ~2%；knockout 证实这在
   Overall 上对应「conv 零行为贡献」，在 worst-group 上对应「C-Deep 零、C-Full 至多微弱且不稳健」。
   ⇒ **「挂了 adapter ≠ 用了 adapter」在行为层面成立**：范围扩展多加的 conv adapter 基本没被优化用起来。
3. **与 H1/H2 null 自洽**：即便 C-Full 的 conv 对 worst-group 有微弱 age 功能，也没转化为相对 ERM 的
   worst-group 增益（H1/H2 均 n.s.）——因 age 条件信号本身缺乏（新 OOF V-info 闸门：**未检出、≈0**；原写「>0 但小」已过时），且其微弱依赖主要经 fc 起作用。
4. **对后续实验的指向**：若要让深层 conv 真正承担 age 条件化，需从**参数化/优化侧**（conv adapter 初始化
   尺度、rank、分层学习率、给 conv 通路更强梯度信号）入手，而非仅调 checkpoint（实验①）或事后 knockout。
   本实验为该方向提供了**基线归因**：当前设定下 conv 通路的行为贡献接近 0。

---

## 4. 复现

```bash
source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate \
    /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
cd $PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# 本地 GTX 1050 Ti（不排队），batch 32 防 4GB OOM；集群 gpus24 可 --batch-size 128
python scripts/e2_knockout_permutation.py --checkpoint overall --batch-size 32
```

**新增脚本**：`scripts/e2_knockout_permutation.py`
- 复用 `_build_cond_model`（实验①加入）载 checkpoint、`_worst_and_overall`（e1_rho_and_permutation）
  的 averaging 置换评估、`load_candidate_groups` 的 age-4 候选组。
- knockout 经 `_zero_generator(model, "fc"|"conv")` 原地置零生成器，变体间从缓存复原、无需重载 checkpoint。
- `full` 变体复现 `e1_permutation.json` 作内建自检。

**输出**：`outputs/conditioning_ablation/ham10000/cv5/e2_knockout_permutation_overall.json`
（每 cell×变体的 Δoverall/Δworst + 真实/置换均值与 SD）。

> 如需换 best_worstcase 复核：`--checkpoint worstcase`（实验①已证机制图景不随 checkpoint 变，预期同结论）。
