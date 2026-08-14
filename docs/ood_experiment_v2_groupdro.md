# OOD 实验 v2 · GroupDRO 第四臂（MIMIC↔CheXpert）

> 预注册：`docs/ood_experiment_v2_preregistration.md` **偏离 #3**（范围扩充 + H5 + Holm family 4→6）。
> 主文件：`docs/ood_experiment_v2_results.md`（ERM / SWAD / HyperAdapt 三臂）。
> ID 侧对照：`docs/groupdro_baseline_5datasets.md`（五库 ID regime）。
> 口径：**full-target** 估计量 · **5 fold × 3 trial = 15 replicate** · 患者级配对 cluster bootstrap。
> 日期：2026-08-14

---

## 0. 一句话结论

> **H5 决定性否定。** GroupDRO 的 OOD worst-group AUC 在**两个方向**都**显著劣于 ERM**
> （M→C −0.0069、C→M −0.0061），Overall 亦然（−0.0077 / −0.0055）。这与它在 ID 侧五库的表现
> 一致，且**是整个 OOD v2 中唯一一个效应量超过训练噪声的比较**（C→M Overall）——**正面效应至今
> 一个都没超过噪声，唯一超过的是负面效应**。

**对项目主论点的作用**：此前"HN 相对基线的表现不是因为它是唯一见过子群标签的方法"这一辩护
**只有 ID 侧证据**。补上 OOD 后，该替代解释在**两个 regime、两个方向**上均被排除——训练期使用
子群标签（GroupDRO）在跨库部署下不但没有优势，反而是四臂中唯一稳定为负的。

---

## 1. 假设判定

| 编号 | 假设（偏离 #3 事前写定） | 判定 |
|---|---|---|
| **H5** | S1 下 GroupDRO marginal-worst 优于 ERM，**两方向复现**，**且** \|d̄\| > σ_train | ❌ **两条判据 × 两个方向，四项全败** |

H5 要求 bootstrap CI 下界 > 0 **且** 效应量超过训练噪声。实测两方向 CI **上界均 < 0**（即显著为**劣**），
方向判据未满足即已判负；噪声判据在主指标上亦未满足。

---

## 2. 主结果（full-target，15 replicate）

### 2.1 marginal worst-group AUC（主指标）

| 方向 | ERM | SWAD | HyperAdapt | **GroupDRO** |
|---|---|---|---|---|
| M→C | 0.8125 | 0.8078 | **0.8156** | **0.8056**（四臂最低） |
| C→M | **0.7882** | 0.7875 | 0.7857 | **0.7821**（四臂最低） |

| 方向 | Δ(GroupDRO−ERM) | 95% CI | \|d̄\| | **σ_train** | t | H2 判据 |
|---|---|---|---|---|---|---|
| M→C | **−0.0069** | [−0.0086, −0.0053] | 0.0069 | 0.0104 | −2.04 | ❌ 未超噪声 |
| C→M | **−0.0061** | [−0.0067, −0.0056] | 0.0061 | 0.0065 | −3.64 | ❌ 未超噪声 |

### 2.2 Overall AUC（次要指标）

| 方向 | ERM | SWAD | HyperAdapt | **GroupDRO** |
|---|---|---|---|---|
| M→C | 0.8505 | 0.8477 | 0.8522 | **0.8428** |
| C→M | 0.8192 | 0.8200 | 0.8170 | **0.8136** |

| 方向 | Δ(GroupDRO−ERM) | 95% CI | \|d̄\| | σ_train | t | H2 判据 |
|---|---|---|---|---|---|---|
| M→C | −0.0077 | [−0.0086, −0.0067] | 0.0077 | 0.0084 | −3.11 | ❌ 未超噪声 |
| **C→M** | **−0.0055** | [−0.0060, −0.0052] | 0.0055 | 0.0054 | −3.98 | ✅ **超过噪声** |

> 🛑 **这是 OOD v2 全部比较中唯一一个 \|d̄\| > σ_train 的条目。** 主文件 §4 的方法学要点
> （"bootstrap 显著 ≠ 超过训练噪声"）在此得到一个**反向的印证**：当效应真的够大时，两个判据会一致。
> 而它恰好是一个**性能下降**，说明该框架并非对一切效应都判"未超噪声"——判据有区分力，不是钝器。

### 2.3 Holm 校正（family 由 4 扩为 6）

偏离 #3 事先声明"既有 4 项须按新 family 重新校正，若失去显著性须如实更新"。**实测无需更新**：

| 检验 | p | Holm 后（family=6） |
|---|---|---|
| C2M:groupdro − erm | 0 | 保持显著 |
| M2C:groupdro − erm | 2.2e−16 | 保持显著 |
| C2M:hyperadapt − erm | 2.2e−15 | 保持显著 |
| M2C:swad − erm | 1.1e−11 | 保持显著 |
| M2C:hyperadapt − erm | 3.3e−07 | 保持显著 |
| C2M:swad − erm | 0.018 | 保持显著 |

**回归核验**：ERM/SWAD/HyperAdapt 的 8 个点估计与 4 个 Δ 值与主文件 §2 **逐位一致**
（如 M→C HyperAdapt−ERM = +0.0031、C→M = −0.0025）⇒ 加入第四臂未扰动既有口径。

---

## 3. 校准（D2，描述性；偏离 #3 明确不为 GroupDRO 新增 H3 型判定）

**点估计**（full-target，15 replicate）：

| 方向 | 方法 | CITL | ECE | Brier | slope |
|---|---|---|---|---|---|
| M→C | ERM | +0.1069 | 0.1069 | 0.0780 | 1.094 |
| M→C | GroupDRO | +0.1242 | 0.1243 | 0.0903 | 0.955 |
| C→M | ERM | −0.1601 | 0.1601 | 0.1826 | 0.718 |
| C→M | GroupDRO | −0.1481 | 0.1496 | 0.1816 | 0.668 |

**Δ = GroupDRO − ERM**（Δ<0 = 校准更好）：

| 方向 | Δ\|CITL\| | ΔECE | ΔBrier | 超噪声？ |
|---|---|---|---|---|
| M→C | **+0.0174** [+0.0170,+0.0177] | **+0.0174** | **+0.0123** | ❌ 全部未超 |
| C→M | **−0.0120** [−0.0122,−0.0117] | **−0.0105** | −0.0010 | ❌ 全部未超 |

⇒ **一个方向显著更差、另一个显著更好，均在训练噪声之内**——与 HyperAdapt 在主文件偏离 #2 中
呈现的模式**同形**。两个方法各自独立地重现了这一模式，支持如下解读：**跨库校准的方向性差异
更像是方法与偏移方向的偶然配合，而非任一方法的稳定属性**；单方向的校准优势不应被当作卖点。

Holm family 保持实验 G 的 8 个检验不变（GroupDRO 的 contrast 进对比清单但不进 family），
故 `docs/hyperadapt_swad_fusion_full_report.md` 的既有校准结论**逐位不受影响**（已核验）。

---

## 4. ID → OOD 的传导

| 数据集 / 方向 | Δmarg worst(GroupDRO−ERM) |
|---|---|
| **ID** MIMIC | −0.0082 [−0.010,−0.007]* |
| **ID** CheXpert | −0.0162 [−0.020,−0.013]* |
| **OOD** M→C（源 = MIMIC） | −0.0069 [−0.0086,−0.0053]* |
| **OOD** C→M（源 = CheXpert） | −0.0061 [−0.0067,−0.0056]* |

两点观察：

1. **劣势按"源域训练库"传导，且方向一致**：MIMIC 上训出的模型带着 MIMIC 的劣势去 CheXpert，
   CheXpert 上训出的带着自己的劣势去 MIMIC，四个数字全为负。这与主文件 §5.1 的判断一致——
   **无 concept shift 意味着 source 域的相对优劣被忠实地搬运到 target**，只是这次搬运的是劣势。
2. **量级不成比例**：CheXpert 的 ID 劣势（−0.0162）是 MIMIC（−0.0082）的两倍，但其 OOD 劣势
   （−0.0061）反而略小于 M→C（−0.0069）。⇒ ID 劣势的大小**不能**线性外推到 OOD；跨库评估会
   压缩方法间差异（目标域自身的难度成为共同的主导项）。这一点对任何"用 ID 差距预估 OOD 差距"
   的做法都是警告。

---

## 5. 边界与未决

1. **η 未搜**（固定论文默认 0.01，为与其它方法共用同一 6 配置网格）。ID 侧 q 动力学诊断显示
   **CheXpert 的 η 偏大**（单组吃掉 48–62% 权重），故 C→M 方向的负结果**可能被 η 放大**，
   不应解读为"GroupDRO 这个算法在 OOD 上不可能有效"，只能说**在与其它方法同等超参预算下无效**。
2. **属性 knockout 不适用**：GroupDRO 的子群标签只进损失、推理端与 ERM 逐位同构，**无干预面**
   （偏离 #3 已记录）。这本身是个结论：GroupDRO 与 HN 使用子群标签的路径（训练期重加权
   vs 推理期条件化）不可互换，故主文件 §5.1b 那套反事实无法迁移到它身上。
3. **不含 HyperHead / HyperFusion**（沿用主文件范围）⇒ 本文件不得推出任何"条件化深度"结论。
4. **D1 选择损失未做**：GroupDRO 的 6 config × 5 fold trial-0 checkpoint 已全部就位，
   若要补 oracle disclaimer 只需推理、无需重训。

---

## 6. 产物与复现

| 内容 | 路径 |
|---|---|
| H1/H2/H5 方差分解（marginal worst + overall） | `outputs/ood_cxr/variance_decomposition_gdro/` |
| 校准（full-target，含 GroupDRO contrast） | `outputs/ood_cxr/calibration_gdro_ft/` |
| full-target 预测（各方向 15 个） | `outputs/ood_cxr/{m2c,c2m}/cv5_full_target/predictions/groupdro_*.npz` |
| trial 1/2 checkpoint | `outputs/{mimic_cxr,chexpert_cxr}/cv5/groupdro_*_seed{52..56,62..66}_*.pth` |

```bash
# 阶段 1：trial 1/2 训练（trial 0 复用 ID 阶段作业 75529/75530 的 checkpoint）
sbatch --partition=gpus24 --array=0-9%3 \
  --export=ALL,PY_SCRIPT=.../src/training/train_groupdro.py,CONFIG_INDEX=4,GDRO_DATASET=mimic \
  slurm/ood_trial.sh                                    # MIMIC 作业 75757
sbatch --partition=gpus24 --array=0-9%3 \
  --export=ALL,PY_SCRIPT=.../src/training/train_groupdro.py,CONFIG_INDEX=3,GDRO_DATASET=chexpert \
  slurm/ood_trial.sh                                    # CheXpert 作业 75778

# 阶段 3：full-target 双向推理（作业 75829）
sbatch --partition=gpus24 --array=0-1 \
  --export=ALL,METHOD_OVERRIDE=groupdro,NUM_WORKERS=8 slurm/oodv2_full_target.sh

# 分析
python scripts/ood_variance_decomposition.py --methods erm,swad,hyperadapt,groupdro \
  --metric marginal_worst --out-dir outputs/ood_cxr/variance_decomposition_gdro
python scripts/ood_variance_decomposition.py --methods erm,swad,hyperadapt,groupdro \
  --metric overall --out-dir outputs/ood_cxr/variance_decomposition_gdro
python scripts/g_ood_calibration_full_target.py --out-dir outputs/ood_cxr/calibration_gdro_ft
```

### 代码改动（均为追加，既有方法顺序/配色/family 不变）

| 文件 | 改动 |
|---|---|
| `src/training/eval_ood_cxr.py` | METHOD_REGISTRY 注册 `groupdro` → image-only 架构 + `forward_image_only` |
| `src/training/run_ood_cxr_full_target.py` | METHODS 末尾追加 `groupdro`（checkpoint 后缀走 `best_overall`） |
| `scripts/ood_variance_decomposition.py` | METHODS/配色/标签追加；**新增 `--methods`** 使 Holm family 大小由参数而非列表内容决定，并在启动时打印 family 大小 |
| `scripts/ood_calibration.py` | METHODS 追加（但该脚本读 legacy k→k 预测，GroupDRO 无此产物，见下） |
| `scripts/g_ood_calibration_full_target.py` | CONTRASTS 追加 `groupdro_vs_erm`；**HOLM_KEYS 不动**（family 保持 8） |

---

## 7. ⚠️ 两处静默失败（工程教训，已避开但值得记录）

1. **`ood_calibration.py` 静默丢方法**：它读 legacy **k→k** 预测目录（`ood_cxr/*/cv5/predictions`），
   而 GroupDRO 只有 full-target 预测 ⇒ `run_method` 命中 `return None`，该方法被**静默移出结果**，
   退出码 0、无任何警告。**必须核对输出里的方法名**，不能以"跑成功了"为准。
2. **`g_ood_calibration_full_target.py` 的对比清单写死**：METHODS 是导入的（GroupDRO 的数据确实
   被读入并算了点估计），但 CONTRASTS 是实验 G 的硬编码列表 ⇒ 点估计有、对比没有。

两处的共同形态是**"成功执行但产出不含目标对象"**。同类脚本若以"文件缺失即跳过"为容错策略，
应至少 `print` 一条跳过记录（`ood_variance_decomposition.py` 就是这么做的，因此一眼可见）。
