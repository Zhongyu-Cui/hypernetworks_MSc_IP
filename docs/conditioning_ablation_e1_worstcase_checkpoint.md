# E1 敏感性实验①：换用 `best_worstcase` checkpoint 复评

> **归属**：`docs/conditioning_ablation_plan.md`（条件化范围消融）E1 的补充敏感性实验。
> **动机**：E1 主结果（`best_overall` checkpoint）的机制诊断显示——**扩大条件化范围时，分类头（fc）
> 以外的 conv adapter 几乎没被用起来**（conv 的 `ρ^between` 比 fc 小 45~60 倍）。本实验检验这一现象
> 是否由**训练/选择策略**造成：若改保留 **val worst-case AUC 最优**那一档权重（`best_worstcase`
> checkpoint）而非 val Overall 最优（`best_overall`），深层 conv 是否会被更多地启用、公平结论是否改变。
>
> **一句话结论**：**不会**。换 `best_worstcase` 后，条件化仍几乎全部由 fc 承担（conv `ρ^between`
> 仍仅 0.011~0.017、fc/conv 倍数仍 48~63×），H1/H2 仍全部不显著；若有变化，反而是 fc 占比更高、
> C-Full 的 worst-group 更差。**「深层 conv 未被启用」不是 checkpoint 选择策略的伪影。**

---

## 1. 实验设计（隔离单一变量）

- **只改一处**：最终评估所用的 checkpoint 由 `best_overall`（val Overall 最优 epoch）换成
  `best_worstcase`（val worst-case AUC 最优 epoch）。两份 checkpoint 在**同一次训练**中并行保存，
  故 backbone、数据顺序、增强、config、seed 全部相同——纯粹是「保训练轨迹上的哪一档权重」之差。
- **config 选择不变**：仍沿用 §3.2 在 `best_overall` 口径下选定的四 cell config
  （ERM `lr3e-05_wd1e-04` / C-Head `lr1e-04_wd1e-03` / C-Deep `lr1e-04_wd1e-04` /
  C-Full `lr1e-04_wd1e-04`）。这样 H1/H2/ρ 的差异**只归因于 checkpoint**，不掺入「换 config」的混杂。
- **评估口径不变**：averaging 主口径、age-4 候选组、病灶级配对 cluster bootstrap（B=1000）、
  统一 3 seed（42/43/44）、5 折 CV-OOF，全部与 E1 主分析一致。

> **⚠️ 一处口径不对齐（须在解读时记住）**：`best_worstcase` checkpoint 在训练时是按 **val 全 14 子群**
> （sex / age / sex×age 交叉）的 worst-case AUC 选 epoch 的（`train_loop.py` 的 `er.wc_auc`），
> 而 E1 评估终点是 **age-4 组** 的 marginal worst。二者候选集不同（plan §3.2 注①已指出这一漂移），
> 故 `best_worstcase` 并非「针对本评估终点最优」的 checkpoint，只是「针对训练侧 14 子群 worst 最优」。
> 这恰是把它当作**训练策略变量**来观察的意义所在，而非当作更优评估器。

---

## 2. 性能结果（H1/H2，averaging 主口径）

### 2.1 四 cell 点估计

| cell | Overall (overall→worstcase) | marginal worst (overall→worstcase) | gap (overall→worstcase) |
|---|---|---|---|
| ERM    | 0.8925 → **0.8836** | 0.8227 → **0.8101** | 0.0887 → 0.0933 |
| C-Head | 0.8954 → 0.8889 | 0.8194 → 0.8188 | 0.0950 → 0.0873 |
| C-Deep | 0.8947 → 0.8902 | 0.8225 → 0.8165 | 0.0889 → 0.0914 |
| C-Full | 0.8963 → **0.8841** | 0.8214 → **0.7953** | 0.0912 → **0.1067** |

- `best_worstcase` 整体把 **Overall 与 worst 都压低了**（ERM worst 0.823→0.810、C-Full worst 0.821→0.795）
  ——符合预期：worst-case 选点常落在更早/欠拟合的 epoch，容量未充分利用。
- **C-Full 掉得最狠**（worst −0.026、gap 由 0.091 张到 0.107）：范围最广的 cell 在 worst-case 选点下
  反而 worst-group 最差，与「深层范围带来公平收益」相反。

### 2.2 确认性检验 H1/H2

| 假设 | checkpoint | Δworst | 95% CI | ΔOverall | Holm | 判决 |
|---|---|---|---|---|---|---|
| **H1: C-Deep − C-Head** | overall | +0.0030 | [−0.0154, +0.0291] | −0.0007 | 0.839 | 未达显著 |
| | **worstcase** | **−0.0023** | [−0.0210, +0.0246] | +0.0013 | **1.000** | **未达显著** |
| **H2: C-Full − C-Deep** | overall | −0.0011 | [−0.0459, +0.0201] | +0.0016 | 0.839 | 未达显著 |
| | **worstcase** | **−0.0212** | [−0.0684, +0.0149] | −0.0061 | **1.000** | **未达显著** |

- **两个 checkpoint 下 H1/H2 均不显著**，范围结论稳健：扩大条件化范围没换来 worst-group 公平收益。
- worstcase 下 H2 的点估计 **更负**（−0.021，「较强」量级），但 CI 仍含 0、Holm=1.0，方向为「扩到全范围
  反而更差」而非改善——与用户设想的「worst-case 选点会让深层更有用」相反。
- **最小观测有效配置**：overall = **无**；worstcase = **C-Head**（C-Head/C-Deep 对 ERM 的 Δworst 点估计
  为正、C-Full 转负）。但这是纯描述性点估计判据，三档对 ERM 的 CI 均含 0，不构成显著改善。

---

## 3. 机制诊断 ρ_l（本实验的核心）

逐层有效偏移，4 age 组等权 × 5 折 × 3 seed 平均。两个量：`ρ`＝总偏移量级、
`ρ^between`＝属性特异性偏移（唯一能证明「按属性分化」的量，见 plan §4）。

### 3.1 fc vs conv 汇总（回答「用了 fc 以外的层吗」）

| cell | checkpoint | fc ρ | **fc ρ^between** | **conv ρ^between（均）** | **fc/conv 倍数** |
|---|---|---|---|---|---|
| C-Head | overall | 1.066 | 0.704 | — | — |
| | **worstcase** | 1.335 | **0.894** | — | — |
| C-Deep | overall | 1.083 | 0.753 | 0.0158 | 47.6× |
| | **worstcase** | 1.120 | **0.803** | **0.0167** | **48.2×** |
| C-Full | overall | 0.954 | 0.638 | 0.0108 | 59.2× |
| | **worstcase** | 1.154 | **0.773** | **0.0122** | **63.4×** |

- **conv 的属性特异性偏移几乎不变**（C-Deep 0.0158→0.0167、C-Full 0.0108→0.0122），仍在 1~2% 量级，
  比 fc 小 **48~63 倍**（overall 是 48~59 倍）。**换 checkpoint 没让深层 conv 被更多地启用。**
- **fc 的偏移不降反升**（fc ρ^between 三档均↑）：`best_worstcase` 若有区别，是把条件化**更集中到 fc**，
  而非铺到 conv。**与「worst-case 选点会调动深层」的设想相反。**

### 3.2 C-Full 逐层（worstcase，仍单调递增且全程极小）

| 层 | ρ | ρ^between |
|---|---|---|
| layer1（4 conv 均） | ~0.014 | ~0.0106 |
| layer2（5 conv 均） | ~0.015 | ~0.0109 |
| layer3（5 conv 均） | ~0.018 | ~0.0131 |
| layer4（5 conv 均） | ~0.019 | ~0.0138 |
| **fc** | **1.154** | **0.773** |

- 与 overall 同构：**越浅的层被用得越少**（layer1 0.0107 → layer4 0.0139），全部 conv 合起来仍不到 fc 的 2%。
- ⇒ **「挂了 adapter ≠ 用了 adapter」在 worstcase 下同样成立**；范围扩展多加的 conv adapter 优化依旧
  没真正用起来，故 H1/H2 的 null 与机制自洽，非单纯功效不足。

---

## 4. 属性置换评估

病灶级整组置换 age（n_perm=20，与 E1 主分析同一 rng），分组键恒用真实 age。Δ = 真实 − 置换。

| cell | Δoverall (overall→worstcase) | Δworst (overall→worstcase) |
|---|---|---|
| **ERM**（自检） | +0.0000 → **+0.0000** | −0.0000 → **+0.0000** |
| C-Head | +0.0119 → +0.0142 | +0.0086 → +0.0050 |
| C-Deep | +0.0202 → +0.0248 | +0.0245 → +0.0277 |
| C-Full | +0.0147 → +0.0193 | +0.0188 → **+0.0023** |

- **ERM 的 Δ 恒为 0（实测 1.1e-16）**——内建正确性自检通过（ERM 忽略 age）。
- **三个 conditioning cell 在 worstcase 下 Δ 仍 >0**（打乱 age 即掉性能）⇒ **换 checkpoint 后模型仍在
  使用 age 输入**，与 overall 结论一致，「模型完全忽略属性输入」被排除。
- Overall 的 Δ 三档均↑（worstcase 略更依赖 age 输入）；唯 **C-Full 的 worst Δ 掉到 +0.0023**——因
  best_worstcase 的 C-Full worst-group 本身被压低（0.795，见 §2.1），worst 组内已接近噪声地板。
- **注**：置换只能证「模型没忽略 age 输入」，不能排除「差异来自额外参数量/训练正则化」
  （置换不改参数量与训练过程）——表述边界同 E1 主分析。

> 计算：本地 GTX 1050 Ti（batch 32）建 worstcase logits 表 + CPU 置换分析，**未占用集群队列**
> （原集群作业 72548 因队列拥堵改为本地执行）。落盘 `e1_permutation_worstcase.json`。

---

## 5. 结论

1. **训练/选择策略不是「深层 conv 未被启用」的原因。** 换成 `best_worstcase` checkpoint 后，
   conv 的 `ρ^between` 仍仅 1~2%、fc/conv 倍数仍 48~63×，机制图景与 `best_overall` 几乎一致。
   优化把条件信息放在分类头，是这套**统一低秩参数化 + 训练设定**的稳健属性，不随「保哪一档权重」改变。
2. **公平结论稳健**：H1/H2 在两个 checkpoint 下均不显著；`best_worstcase` 若有区别，是整体压低性能、
   且让 C-Full 的 worst-group 更差（与「深层范围改善公平」相反）。
3. **对后续小实验的指向**：既然「换 checkpoint」这一训练策略无效，若要让深层 conv 真正被启用，
   需从**参数化/优化侧**入手（如 conv adapter 的初始化尺度、rank、学习率分层、是否给 conv 通路更强的
   梯度信号），而非仅调 checkpoint 选择。

---

## 6. 复现

```bash
source /vol/biomedic2/bglocker_studproj/zc125/software/miniconda3/bin/activate \
    /vol/biomedic2/bglocker_studproj/zc125/envs/medimg
export PYTHONPATH=/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
cd $PYTHONPATH

# ① H1/H2（CPU，复用已落盘的 _worstcase.npz test 预测）
python scripts/e1_ham_analysis.py --n-boot 1000 --checkpoint worstcase

# ② ρ_l（纯 CPU，从 best_worstcase.pth 生成器计算，无需 GPU）
python scripts/e1_rho_and_permutation.py --mode rho --checkpoint worstcase
python scripts/e1_rho_aggregate.py --checkpoint worstcase

# ③ 置换（需 GPU logits 表）：本地 GTX 1050 Ti（batch 32，不排队）建表 + CPU 分析
python scripts/e1_rho_and_permutation.py --mode logits  --checkpoint worstcase --batch-size 32
python scripts/e1_rho_and_permutation.py --mode analyze --checkpoint worstcase
# （集群版：ssh biomedia-slurm 'sbatch slurm/e1_rho_logits.sh worstcase'，队列拥堵时改本地）
```

**改动的脚本**（均新增 `--checkpoint {overall,worstcase}`，默认 overall、不影响既有主结果）：
- `scripts/e1_ham_analysis.py`：test 预测按 `_{ckpt}.npz` 加载；输出 `e1_ham_results_averaging_worstcase.json`。
- `scripts/e1_rho_and_permutation.py`：新增纯 CPU 的 `--mode rho`；`.pth`/logits 表/落盘文件名按 checkpoint 加后缀。
- `scripts/e1_rho_aggregate.py`（新建）：把逐 run ρ 汇总成 per-cell 层级表 `e1_rho_between_worstcase.json`。
- `slurm/e1_rho_logits.sh`：接受可选参数 `[overall|worstcase]`（默认 overall）。

**输出文件**（`outputs/conditioning_ablation/ham10000/cv5/`，均已生成）：
`e1_ham_results_averaging_worstcase.json`、`e1_rho_worstcase.json`、`e1_rho_between_worstcase.json`、
`age_logit_tables/*_worstcase.npy`（60 张）、`e1_permutation_worstcase.json`。
