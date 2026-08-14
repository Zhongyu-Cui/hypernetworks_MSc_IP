# E1 敏感性实验③：训练 fc-off 的 C-Deep / C-Full——超网络能否激活 conv？

> **归属**：`docs/conditioning_ablation_plan.md` E1 的补充敏感性实验（承接实验①
> `..._e1_worstcase_checkpoint.md`、实验② `..._e1_pathway_knockout.md`）。
> **动机**：E1 与实验②证明条件化几乎全在 fc、conv 行为近乎惰性。本实验**关掉 fc 条件化、只留 conv、
> 从头训练** C-Deep / C-Full，检验超网络是否被"逼"着把 age 条件化路由进 conv——以区分两种解释：
> **(a) fc 挤占**（conv 本可承担、只是 fc 是阻力最小路径）vs **(b) 优化/参数化瓶颈**（低秩乘性 conv
> 通路本身难优化或无用）。
>
> **一句话结论**：**都不完全是——是一个更精细的判决。** conv **可以**被优化（关 fc 后 ρ^between 随训练
> 增长、shared-A 离开零、B 生成器梯度解锁），故**不是**「B 永远拿不到梯度」的硬瓶颈；**但**把 conv 推得
> 越活、性能反而越差，故早停在 val-Overall 最优点选中的**操作点上 conv 激活量与 fc-on 版几乎相同
> （1.0~1.4×），仍比 fc 小 35~60 倍**，且行为退回**近乎属性无关**（C-Deep-noFC≈ERM、C-Full-noFC 劣于
> ERM）。⇒ **移除 fc 不会把条件化重路由进 conv，而是丢失条件化**；fc 不只是阻力最小路径，而几乎是这套
> 低秩超网络**唯一能有用地表达（弱）age 信号的通路**。

---

## 1. 实验设计

- **唯一变量 = 关 fc**：新增 location `deep_nofc` / `full_nofc`，与 E1 的 `deep`/`full` 唯一差别是
  `condition_fc=False`（fc 退化为属性无关共享 base 头），conv 条件化范围不变。
- **其他全同 E1**：同 config（**lr1e-04_wd1e-04 = E1 为 C-Deep/C-Full 选定的 config**）、同 5 折、
  同 3 seed（42/43/44）、同 regime（BCE/AdamW/bs128/≤30 epoch/早停 patience5）、canonical base state。
  60 run（2 cell × 5 折 × 3 seed），gpus24，零报错。
- **逐 epoch 机制探针**（`--diagnostics`，落 `diag_logs/<run_id>.jsonl`）：conv ρ/ρ^between（均值+逐层）、
  conv adapter 梯度范数（shared-A / B_gen 分列，**裁剪前真实梯度**）、权重范数 ‖A‖/‖B‖。
- 初始等价自检：`deep_nofc`/`full_nofc` 初始 |Δlogit vs ERM|=4e-7（fc 无 adapter + shared-A 零初始 ⇒ Δθ≈0）。

---

## 2. 结果

### A. 诊断轨迹（逐 epoch，跨 15 run 对齐平均）——conv **可以**被激活，但激活越多性能越差

| cell | epoch | conv ρ^between | grad_A | grad_B | ‖A‖ | val Overall | val worst |
|---|---|---|---|---|---|---|---|
| **C-Deep-noFC** | 1 | 0.0119 | 0.058 | 0.004 | 0.63 | 0.8934 | 0.6818 |
| | 5 | 0.0280 | 0.099 | 0.017 | 1.19 | 0.8927 | 0.6780 |
| | 10 | 0.0477 | 0.108 | 0.024 | 1.71 | **0.8617** | **0.5846** |
| **C-Full-noFC** | 1 | 0.0080 | 0.232 | 0.010 | 0.68 | 0.8859 | 0.6686 |
| | 5 | 0.0126 | 0.285 | 0.031 | 1.28 | 0.8835 | 0.6941 |
| | 10 | 0.0160 | 0.250 | 0.035 | 1.75 | 0.8500 | 0.6616 |

- **conv 通路是可优化的（否定纯梯度锁瓶颈）**：关 fc 后 conv ρ^between 随训练**单调增长**（Deep 0.012→0.052、
  Full 0.008→0.020）；**shared-A 离开零初始**（‖A‖ 0.6→1.7~2.1）；**B 生成器梯度从 ~0 解锁并增长**
  （grad_B 0.004→0.025~0.031）。⇒ 「A=0 ⇒ ∂L/∂B=0」的串行依赖**确被打破**，B 通路并非永久冻结。
- **但激活 conv 与性能反向**：随 conv ρ 增长到后期，**val Overall 与 val worst 双双下降**
  （Deep ep10 掉到 0.862/0.585）。⇒ 更多 conv 调制 = 更差性能，conv 学到的乘性调制**不编码有用的
  subgroup 结构**（与新 OOF V-info 闸门判 HAM-age **未检出、≈0** 一致）。

### B. 操作点（best_overall checkpoint）的 conv ρ^between——**几乎没比 fc-on 多**

早停按 val Overall 选 checkpoint，**平均选在 epoch ~3**（Deep 3.3 / Full 3.1，即 val-Overall 峰值处），
此时 conv 尚未大幅激活：

| cell | conv ρ^btw (nofc, 操作点) | conv ρ^btw (fc-on, E1) | 倍数 | 对比：fc-on 的 **fc** ρ^btw |
|---|---|---|---|---|
| **C-Deep-noFC** | 0.0217 | 0.0158 | **1.4×** | 0.7527 |
| **C-Full-noFC** | 0.0102 | 0.0108 | **1.0×** | 0.6384 |

- 操作点上 nofc 的 conv ρ^between 与 fc-on 版**几乎相同**（Deep 1.4×、Full 1.0×）；仍比 fc-on 的 fc
  ρ^between **小 35~60 倍**。⇒ **移除 fc 并未在操作点上把条件化有效转移进 conv**。
- 轨迹（A）显示 conv **能**涨到更高（Deep 0.05），但那发生在**性能已劣化的后期**，被早停排除在操作点外。

### C. 行为（averaging 口径 test）——退回近乎属性无关

| cell | Overall | marginal worst | Δworst vs ERM | 对比 fc-on worst |
|---|---|---|---|---|
| ERM | 0.8925 | 0.8227 | — | — |
| **C-Deep-noFC** | 0.8974 | 0.8277 | **+0.0050** | 0.8225（fc-on C-Deep） |
| **C-Full-noFC** | 0.8948 | 0.8057 | **−0.0170** | 0.8214（fc-on C-Full） |

- **C-Deep-noFC ≈ ERM**（worst +0.005，实为打平）：关 fc 后 layer4-only conv 在操作点几乎不带来 age
  条件化，模型退回属性无关基线。
- **C-Full-noFC 劣于 ERM**（worst −0.017）：铺满 conv 的额外容量在操作点**帮倒忙**（conv 调制注入噪声、
  未编码有用 age 结构），worst-group 反被压低——与 A 中「conv 越活越差」一致。

### D. 属性置换（真实 vs 置换 age，与 E1/E2 口径一致）

对 nofc 模型做病灶级整组置换 age（n_perm=20、rng seed=0、averaging、age-4 候选组、分组键恒真实、
best_overall），从**行为层面**测 conv-only 模型是否依赖 age。σ = Δworst / 20 次置换 SD。

| 对象 | Δoverall | Δworst | worst σ |
|---|---|---|---|
| **C-Deep-noFC**（③ conv-only，从头训练） | +0.0089 | **+0.0077** | 1.6σ |
| C-Deep（① fc-on 全通路） | +0.0202 | +0.0245 | — |
| C-Deep\|fc_off（② 推理时关 fc） | −0.0010 | +0.0018 | — |
| **C-Full-noFC**（③ conv-only，从头训练） | +0.0100 | **+0.0062** | 1.0σ |
| C-Full（① fc-on 全通路） | +0.0147 | +0.0188 | — |
| C-Full\|fc_off（② 推理时关 fc） | +0.0019 | +0.0084 | — |

- **nofc 模型确有小幅 age 依赖**（Δ>0）：conv-only 从头训练并非完全属性无关，行为上仍经 conv 用了一点 age。
- **训练时关 fc > 推理时关 fc**：C-Deep-noFC 的 Δworst +0.0077 ≈ 推理时关 fc（② +0.0018）的 **4 倍**
  ⇒ **从头训练确实把更多 age 依赖逼进了 conv**（呼应实验③正文：conv 能被激活）。
- **但仍弱**：conv-only 的 Δworst 只有 fc-on 全通路的 **~1/3**（Deep 0.0077/0.0245、Full 0.0062/0.0188），
  且统计弱（1.0~1.6σ，未明显超过置换噪声，Full 尤甚）。⇒ 逼进 conv 的 age 依赖**弱且不稳健**，与「激活了
  但无用、行为退回近属性无关」（§2C：Deep≈ERM、Full<ERM）自洽。

---

## 3. 结论

1. **回答「超网络能否激活 conv」**：**能，但没用。** 关 fc 后 conv 确实可被优化（ρ^between 增长、
   shared-A 离零、B 梯度解锁）——**否定了「B 永远拿不到梯度」的硬优化瓶颈**；**但**把 conv 推得越活性能越差，
   故性能最优的操作点上 conv 激活量与 fc-on 几乎相同、仍远小于 fc。
2. **两种解释的精细判决**：
   - **不是纯「fc 挤占一个本可用的 conv」(a)**：移除 fc 后 conv 并未有用地接管——行为退回近乎属性无关
     （Deep≈ERM、Full<ERM），而非恢复 age 条件化。
   - **也不是纯「梯度锁死」(b)**：梯度确实流入、conv 确能被激活。
   - **真相**：低秩**乘性 conv 调制**是可优化的，但它学到的东西**对本任务无用**（激活它反而降性能）；
     性能目标因此自动把 conv 压到最小，即便 fc 不在。**fc 的加性低秩头 adapter 几乎是这套超网络唯一能
     有用表达（弱）age 信号的通路。**
   - **行为置换佐证（§2D）**：conv-only 从头训练确实比"推理时关 fc"（实验②）多逼进了一点 age 依赖
     （Deep Δworst 0.0077 ≈ 4× 实验②的 0.0018），但只有 fc-on 全通路的 ~1/3、且弱（1.0~1.6σ）
     ⇒ 逼进 conv 的 age 依赖弱且不稳健，与"激活了但无用"一致。
3. **与实验①②闭环**：①换 checkpoint 不改机制、②行为 knockout 证 age 依赖在 fc、③关 fc 训练证 conv
   无法有用接管——三者一致指向：**在当前统一低秩训练设定与 HAM 弱 age 信号下，条件化只在 fc 有效**。
4. **对"从参数化侧逼 conv"的指向**：既然 conv 通路可优化但所学无用，单纯"逼它激活"无益；若要 conv 承载
   有用条件化，需改**参数化本身**（如乘性调制→加性、更高 rank、或让 conv 调制对齐 age-判别方向的正则），
   而非仅移除 fc 或调 checkpoint/训练时长。

---

## 4. 复现

```bash
# 训练（集群，60 run；唯一变量=关 fc，config/折/seed/regime 全同 E1）
ssh biomedia-slurm 'cd .../hypernetworks_MSc_IP
  sbatch --export=ALL,DATASET=ham10000,LOCATION=deep_nofc,CONFIG_INDEX=2 slurm/e3_condnet_nofc.sh
  sbatch --export=ALL,DATASET=ham10000,LOCATION=full_nofc,CONFIG_INDEX=2 slurm/e3_condnet_nofc.sh'
# 分析（纯 CPU）：轨迹 + 最终 ρ vs fc-on + 行为 vs ERM
python scripts/e3_nofc_analysis.py
# 属性置换（本地 GPU 建反事实 logits 表，batch32 不排队）：真实 vs 置换 age
python scripts/e3_nofc_permutation.py --batch-size 32
```

**代码改动**（均向后兼容，默认不影响既有实验）：
- `src/models/resnet18_condnet.py`：新增 location `deep_nofc`/`full_nofc`（condition_fc=False）；新增
  device-aware `conditioning_rho(embeddings)`（`scripts/e1_rho_and_permutation.compute_rho` 现委托它，
  单一事实来源，已验证 E1 ρ 数字逐值不变）。
- `src/training/harness/train_loop.py`：`run_training`/`train_one_epoch` 加可选 `on_after_backward`
  （裁剪前梯度探针）+ `epoch_diag_fn`（逐 epoch 诊断回调）；不改 val 日志 schema。
- `src/training/train_condnet.py`：`LOCATION_TO_METHOD` 加两 nofc；新增 `CondNetDiagnostics`（三诊断量
  → 独立 `diag_logs/`）+ `--diagnostics` flag。
- `slurm/e3_condnet_nofc.sh`（array 0-14 = 5 折×3 seed）；`scripts/e3_nofc_analysis.py`（轨迹/ρ/行为分析）；
  `scripts/e3_nofc_permutation.py`（§2D 属性置换，本地 GPU）。

**输出**：`outputs/conditioning_ablation/ham10000/cv5/`：`*/diag_logs/condnet_{deep,full}_nofc_*.jsonl`
（逐 epoch 诊断，30 条）、`*/predictions/*_nofc_*.npz`、`e3_nofc_results.json`、`e3_nofc_permutation.json`。

> **一处诚实边界**：早停按 val Overall 选操作点，故本结论描述的是"性能最优点上 conv 未被有用激活"。
> 轨迹显示训练更久 conv 会更活但性能更差——这本身即证据（激活 conv 不利），非口径遗漏。
