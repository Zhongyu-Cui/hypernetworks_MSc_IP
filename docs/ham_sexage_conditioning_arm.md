# HAM10000 双属性（sex + age）条件化对照臂

**日期**：2026-08-28　**作业**：77993 / 77994 / 77995（各 array 0–29，共 90 run）
**结论一句话**：给 HAM 的三个 HN 把条件输入从 age-only 补齐为 sex+age，**worst-group 与 Overall 均无稳健变化**——
条件化口径与评估口径的不对称**不是** HN 在 HAM 上打平基线的原因。

---

## 1. 动机：一个口径不对称

HAM 的三个 HN（HyperHead / HyperFusion / HyperAdapt）此前**只条件化 `age_group`**，理由写在各训练脚本
docstring 里：「HAM 诊断 I(Y;age|X)>0、sex≈0」。但 worst-group 的**评估**口径（`src/utils/ham10000_fairness.py`
的 `worst_case_auc`）是 **Sex / Age / Sex×Age 三套分组**——即 sex 轴与交叉格上的最差子群，模型从未拿到对应属性。

这构成一个可检验的替代解释：HN 的 worst-group 打平，可能不是「属性无用」，而是「模型压根没拿到决定 worst-group
的那个属性」。本臂把条件输入补齐为**评估分组变量的全集**，与 age-only 臂构成单变量对照。

另需注明：选 age 而排除 sex 的原始依据（单-split 条件互信息）在 2026-07-19 换成 OOF conditional V-information
闸门后**已退回未检出（HAM-age ≈ 0）**（见 `docs/conditional_v_information_gate.md`）。即当前读数下 sex 与 age
同属无可用信号，本臂的预期结果就是 null；它的价值在于把「口径不对称」这个混淆**实测排除**，而不是靠论证排除。

## 2. 实现

单变量对照：与 age-only 臂的唯一差异是条件输入，其余（样本集、`age_group>=0` 过滤、超参网格、split/seed 规程、
评估口径）全部一致。因此不新建脚本，而是给三个现有 HAM HN 训练脚本加 `--cond {age,sex_age}`，缺省 `age` 保持历史行为。

| 文件 | 新增 | 条件通路 | Δparams |
|---|---|---|---|
| `src/models/resnet18_hyperhead.py` | `SexAgeHyperHeadNet` + `ResNet18HyperHeadSexAge` | `sex_emb(2×4)`⊕`age_emb(4×4)` → MLP → fc 权重/偏置 | +264 |
| `src/models/resnet18_hyperfusion.py` | `SexAgeEmbedding` + `ResNet18HyperFusionSexAge` | `sex_emb(2×16)`⊕`age_emb(4×16)` → 两层 fuse → 128 维 profile | +2,080 |
| `src/models/resnet18_hyperadapt.py` | `SexAgeEmbedding` + `ResNet18HyperAdaptSexAge` | 同上 | +2,080 |

条件通路 = MIMIC 的 `PatientEmbedding`（sex+race+age）去掉 race 支路；输出维度不变，故下游机制
（HyperAdapt 的 `A_gen`/`B_gen`/逐样本卷积、HyperFusion 的 `layer4[0].downsample` 注入点、HyperHead 的 `bmm` 头）
原样复用。harness 侧新增 `forward_sex_age`；`unpack_sex_age` 不变（它本来就解包 sex，只是此前 sex 仅供评估）。
method 名记为 `<method>_sexage`，与 age-only 产物按 `run_id=(method, config_tag, seed)` 隔离。
SLURM 侧 `slurm/c1_ham_cv_search.sh` 加可选 `COND` 注入。

**提交前验证**：① 三模型前向/反向正常，sex embedding 拿到非零梯度；② Δθ≈0 初始化保持（HyperAdapt 跨属性
logit std=1.6e-8、HyperHead 1.9e-2，与 age-only 版同量级）；③ 真数据 fold0 过滤后 train 6,807 / val 957 /
test 1,943，与 age-only 臂历史日志逐位一致 ⇒ 两臂样本集完全可比。

## 3. 训练

按 CV-OOF 标准：每方法 6 config（lr∈{3e-5,1e-4,3e-4}×wd∈{1e-4,1e-3}）× 5 折 lesion 级 GroupKFold，
`SEED=42+FOLD`，共 90 run，全部正常结束（无 CPU 回退、无 OOM/CUDA 报错）。
> 首次提交（77988–77990）有两个 task 落到 **mira05，CUDA 初始化失败退回 CPU**，已取消并加
> `--exclude=semois,mira05` 重提为 77993–77995。mira05 是已知坏节点，后续 HAM 提交沿用该排除。

config 选择走 `select_config_cv.py`（折均 marginal-worst，平手用折均 Overall 破），结果写入
`outputs/ham10000/cv5/selected_configs.json`：`hyperhead_sexage=lr3e-04_wd1e-03`、
`hyperfusion_sexage=lr1e-04_wd1e-03`、`hyperadapt_sexage=lr3e-05_wd1e-03`。

## 4. 结果（averaging 口径，n=9,707 张 / 7,280 病灶）

### 4.1 点估计

| 方法 | Overall | canon worst | marg worst | gap |
|---|---|---|---|---|
| ERM | 0.8938 | 0.7877 | 0.7976 | 0.1348 |
| SWAD | 0.9129 | 0.8331 | 0.8433 | 0.1024 |
| GroupDRO | 0.8813 | 0.7816 | 0.8128 | 0.1237 |
| HyperHead (age) | 0.8941 | 0.7712 | 0.8159 | 0.1441 |
| **HyperHead (sex+age)** | 0.9031 | 0.8073 | 0.8210 | 0.1140 |
| HyperFusion (age) | 0.8996 | 0.8171 | 0.8310 | 0.0923 |
| **HyperFusion (sex+age)** | 0.8976 | 0.8117 | 0.8302 | 0.1150 |
| HyperAdapt (age) | 0.8910 | 0.7990 | 0.8217 | 0.1190 |
| **HyperAdapt (sex+age)** | 0.8980 | 0.7820 | 0.8253 | 0.1390 |

三个 sexage 臂 vs ERM 的 marginal-worst 增量（+0.023 / +0.033 / +0.028）与 age-only 臂（+0.018 / +0.033 / +0.024）
几乎重合，且**全部 n.s.**；SWAD 仍是唯一显著优于 ERM 的方法（+0.046）。

### 4.2 配对对照（sexage − age-only，病灶级配对 cluster bootstrap，B=1000）

| 对比 | Overall | canon worst | marg worst |
|---|---|---|---|
| HyperHead | **+0.0090** [+0.0008,+0.0173] | +0.0360 [−0.0439,+0.1256] n.s. | +0.0052 [−0.0362,+0.0760] n.s. |
| HyperFusion | −0.0019 [−0.0098,+0.0064] n.s. | −0.0054 [−0.0514,+0.0816] n.s. | −0.0008 [−0.0368,+0.0476] n.s. |
| HyperAdapt | **+0.0069** [+0.0002,+0.0137] | −0.0170 [−0.0738,+0.0545] n.s. | +0.0036 [−0.0561,+0.0365] n.s. |

**worst-group 全 n.s.，且方向不一致**（canonical：+0.036 / −0.005 / −0.017）。两处 Overall 显著但幅度 <0.01，
且见下节——它主要来自选参差异，不是条件输入。

### 4.3 config-matched 敏感性（同一超参下比两臂，点估计）

两臂各自按协议独立选 config，Δ 里因此混进了超参差异。逐 config 在**同一超参**下比较，6 个 config 的符号来回摇摆：

| 方法对 | ΔOverall 均值（正号数） | Δcanon_w 均值（正号数） | Δmarg_w 均值（正号数） |
|---|---|---|---|
| HyperHead | +0.0017 (3/6) | +0.0161 (4/6) | +0.0120 (4/6) |
| HyperFusion | −0.0003 (4/6) | −0.0019 (2/6) | +0.0018 (3/6) |
| HyperAdapt | +0.0005 (4/6) | −0.0110 (2/6) | −0.0086 (3/6) |

**§4.2 的两处 Overall 显著（+0.009 / +0.007）在 config-matched 下塌到 +0.0017 / +0.0005** ⇒ 它是选参差异，
不是加 sex 的效果。worst-group 侧均值全在 ±0.017 内、符号 2/6–4/6，即随机摇摆。

## 5. 结论与对写作的影响

1. **口径不对称混淆被实测排除**。把条件输入补齐为评估分组变量全集后，HN 的 worst-group 没有稳健变化。
   此前「HN 在 HAM 上 worst-group 打平 ERM、显著输 SWAD」的结论**不受这个不对称影响**。
2. **与 V-information 闸门读数一致**：HAM 的 sex 与 age 轴都无可用条件信号，补属性不产生增益，属闸门预期。
   本臂把这一预期从「论证」升级为「实测」，且是在**三种注入深度**上同时成立。
3. **写作口径**：介绍 HAM HN 时不必再声明「只条件化 age」为设计限制，可直接引本臂说明「条件化 age 还是 sex+age
   不改变结论」。⚠️ 但**不要**把 §4.2 的两处 Overall 显著当作正面结果报——§4.3 已证其为选参伪影；这也是又一次
   「单一口径下的小幅显著在敏感性检查下蒸发」的实例（同类教训见 `docs/frozen_regime_*.md`、实验 G）。

## 6. 复现

```bash
# 训练（每方法 6 config × 5 折 = 30 run）
ssh biomedia-slurm
cd /vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP
sbatch --partition=gpus24 --exclude=semois,mira05 --job-name=ham_hh_sexage \
  --output=/vol/.../logs/ham_hh_sexage_cvsearch.%N.%A_%a.log \
  --export=ALL,PY_SCRIPT=/vol/.../src/training/train_ham10000_hyperhead.py,COND=sex_age \
  slurm/c1_ham_cv_search.sh          # hyperfusion 同；hyperadapt 用 gpus48 + BATCH=128

# config 选择（三个方法各一次，合并写入 selected_configs.json）
python scripts/select_config_cv.py --dataset ham10000 --method hyperhead_sexage \
  --write-json /vol/.../outputs/ham10000/cv5/selected_configs.json --merge

# 汇总（新 method 已登记进 build_oof_results_averaging.METHODS）
python scripts/build_oof_results_averaging.py --dataset ham10000

# 本文档的 §4.2 / §4.3
python scripts/ham_sexage_vs_age_averaging.py --config-matched \
  --write-json /vol/.../outputs/ham10000/cv5/sexage_vs_age_averaging.json
```
