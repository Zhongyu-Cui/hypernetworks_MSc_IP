# Attribute-Conditioned Hypernetworks for Subgroup Fairness in Medical Image Classification

Imperial College London · MSc Individual Project 的代码仓库。

本项目检验一个具体主张：把患者的敏感属性（性别、种族、年龄、肤色）通过**超网络**
（hypernetwork）喂给分类器，让网络的权重随属性变化，是否能在不牺牲整体判别力的前提下
抬高**最差子群**的表现。四个公开去标识化数据集、八个训练臂、三个实验框架，全部在
5 折交叉验证的 out-of-fold 口径下评估，并配套一组从权重与决策面直接读取的机制诊断，
用来区分「属性没进到网络里」和「属性进去了但没有用」。

## 目录

- [数据集与训练臂](#数据集与训练臂)
- [仓库结构](#仓库结构)
- [环境准备](#环境准备)
- [数据与划分](#数据与划分)
- [框架 1：同分布评估](#框架-1同分布评估)
- [框架 2：跨数据集迁移](#框架-2跨数据集迁移)
- [框架 3：条件化范围消融](#框架-3条件化范围消融)
- [HyperAdapt 的两个扩展臂](#hyperadapt-的两个扩展臂)
- [机制诊断与报告插图](#机制诊断与报告插图)
- [产物布局](#产物布局)

## 数据集与训练臂

| 数据集 | n | 目标 | 患病率 | 敏感属性 | 划分单位 |
|---|---:|---|---:|---|---|
| HAM10000 | 9,707 | malignant | 14.5% | sex、age（4 箱） | lesion |
| Fitzpatrick17k | 16,012 | malignant | 13.5% | skin type（I–VI） | image |
| MIMIC-CXR | 199,356 | No Finding | 30.2% | sex、race、age（2 箱） | patient |
| CheXpert | 138,644 | No Finding | 8.2% | sex、race、age（2 箱） | patient |

二分类目标、属性定义与年龄分箱对齐 MEDFAIR 基准。八个臂：

| 臂 | 属性用在哪里 | 推理时需要属性 |
|---|---|---|
| ERM | 不使用 | — |
| SWAD | 不使用（权重平均，由 ERM 逐折派生） | — |
| GroupDRO | 训练损失按子群重加权 | — |
| HyperHead | 条件化分类头 | ✓ |
| HyperFusion | 条件化一个深层 block | ✓ |
| HyperAdapt | 条件化除 stem 外所有层 | ✓ |
| HyperAdapt+SWAD | 条件化 + 权重平均 | ✓ |
| HyperAdapt(pred) | 条件输入改为从图像预测的属性 | — |

## 仓库结构

```
src/
├── paths.py              所有绝对路径的唯一来源（见「环境准备」）
├── models/               ResNet-18 与四个条件化变体（含消融专用的 CondNet）
├── datasets/             四个数据集的 Dataset 与软属性包装
├── training/
│   ├── harness/          共用训练环、run 命名、超参网格、子群 AUC、SWAD、GroupDRO、bootstrap
│   ├── train_<库>_<方法>.py   训练入口，每个 (数据集, 方法) 一个
│   ├── train_condnet.py       框架 3 的统一入口（一个开关切换条件化范围）
│   └── run_ood_cxr_*.py       跨库评估驱动
├── utils/                各库 fairness 口径、阈值上的操作点与分类指标、超平面可视化
scripts/                  数据划分、配置选择、各框架分析出表、绘图
slurm/                    SLURM 作业脚本（训练与重推理都经此提交）
configs/                  四个数据集的基础配置
data/splits/              5 折交叉验证的划分索引（CSV，随仓库分发）
report/                   报告 LaTeX 源码
docs/                     实验过程记录与方案文档
```

> `docs/` 是研究过程的完整记录，其中一部分实验（PAPILA、ROC 后处理、合成属性注入、
> 信息闸门估计等）没有进入最终报告，相应代码已从本仓库移除，可从提交历史取回。
> 代码与报告以 `report/` 与本文件为准。

## 环境准备

Python 3.11。

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

所有脚本以仓库根为导入根，运行前先设好 `PYTHONPATH`：

```bash
cd /path/to/hypernetworks_MSc_IP
export PYTHONPATH="$(pwd)"
```

路径全部集中在 [`src/paths.py`](src/paths.py)，用环境变量覆盖即可换机器，不必改代码：

| 环境变量 | 含义 | 缺省 |
|---|---|---|
| `HN_WORK_ROOT` | 产物根目录 | 仓库的祖父目录 |
| `HN_OUTPUTS` / `HN_LOGS` | 直接指定 outputs / logs | `$HN_WORK_ROOT/{outputs,logs}` |
| `HN_CXR_ROOT` | CXR7-1M 根（MIMIC-CXR 与 CheXpert 图像 + master CSV） | 共享盘位置 |
| `HN_CHEXPERT_META` | CheXpert-Plus 元数据目录（`chexbert_labels.zip`） | 共享盘位置 |
| `HN_HAM_ROOT` | HAM10000 根 | 共享盘位置 |
| `HN_FITZ_ROOT` | Fitzpatrick17k 根 | 共享盘位置 |

SLURM 作业另有三个：`HN_REPO_ROOT`（仓库位置）、`HN_CONDA_ACTIVATE` 与 `HN_CONDA_ENV`
（conda 环境；activate 文件不存在时自动跳过，直接用当前解释器）。作业脚本的
`--output` 写成相对路径，提交前在提交目录下 `mkdir -p logs`。

训练必须提交到 GPU 集群，分析与可视化脚本只用 CPU。

条件化模型逐样本生成一套权重，故 `HyperAdapt` 与 `CondNet` 各 cell 在同一 batch 下的显存
需求远高于基线：全流程用的 batch=128 在 4 GB 卡上放不下（实测 OOM 于第一层 adapted conv），
这也是 HN 臂投 gpus48 的原因。降 batch 可以在任何卡上跑起来（实测 batch=16 正常训练），
但那会破坏与基线的等-batch 对照，只适合冒烟测试。

## 数据与划分

四个数据集均为公开去标识化数据，需自行取得访问权限后放到本地，再用环境变量指到对应位置。
仓库内 `data/splits/` 已含现成的 5 折划分索引，直接复现报告结果不需要重新生成；
要从原始数据重建（划分按上表的单位分组，杜绝同病灶/同患者跨折泄漏）：

```bash
python scripts/build_ham10000_splits.py
python scripts/build_mimic_splits_nofinding.py
python scripts/build_chexpert_splits_nofinding.py
python scripts/build_fitzpatrick_splits.py        # 先建单次划分
python scripts/build_fitzpatrick_cv_splits.py     # 再由它派生 5 折
```

## 框架 1：同分布评估

每个 (数据集, 方法) 跑 6 个超参配置 × 5 折，用验证集 worst-group AUC 折均选一个配置，
再把 5 折互斥的 test 预测拼成 out-of-fold 结果。

**1. 训练**（每个数据集有一个作业脚本，`PY_SCRIPT` 指训练入口，可写相对路径）

```bash
# HAM10000 · ERM（加 SWAD=1 顺带派生 SWAD 基线）
sbatch --partition=gpus24 --job-name=ham_erm_cvsearch \
       --output=logs/ham_erm_cvsearch.%N.%A_%a.log \
       --export=ALL,PY_SCRIPT=src/training/train_ham10000_resnet18.py,SWAD=1 \
       slurm/c1_ham_cv_search.sh

# HAM10000 · HyperAdapt（sex+age 条件化；HN 臂用 gpus48 保证等 batch）
sbatch --partition=gpus48 --job-name=ham_ha_cvsearch \
       --output=logs/ham_ha_cvsearch.%N.%A_%a.log \
       --export=ALL,PY_SCRIPT=src/training/train_ham10000_hyperadapt.py,BATCH=128,COND=sex_age \
       slurm/c1_ham_cv_search.sh

# GroupDRO 复用同一 array 脚本，数据集经 DATASET 注入
sbatch --partition=gpus24 --job-name=mimic_gdro_cvsearch \
       --output=logs/mimic_gdro_cvsearch.%N.%A_%a.log \
       --export=ALL,PY_SCRIPT=src/training/train_groupdro.py,GDRO_DATASET=mimic \
       slurm/c2_mimic_cv_search.sh
```

其余数据集换作业脚本即可：MIMIC 用 `c2_mimic_cv_search.sh`，CheXpert 用
`c3_chexpert_cv_search.sh`，Fitzpatrick 用 `c4_fitz_cv_search.sh`。

**2. 选配置 → 3. 出结果 → 4. 统计判决**

```bash
python scripts/select_config_cv.py --dataset ham10000 --merge   # 写 selected_configs.json
python scripts/build_oof_results_averaging.py                    # 主口径：逐折算再折间平均
python scripts/id_holm_correction.py                             # Holm 校正后的判决
# 单库的逐子群明细；--config-json 必填，指向第 2 步写出的那个文件
python scripts/cv_oof_report.py --dataset ham10000 --section all \
       --config-json "${HN_OUTPUTS:-$PWD/../../outputs}/ham10000/cv5/selected_configs.json"
```

> 评估口径是 **averaging**（逐折算指标再平均），不是跨折池化原始分数——后者会因逐折
> 校准漂移制造假阳性。唯一例外是 `cv_oof_report.py`：它打印的是**池化**视图，只作逐子群
> 明细的交叉核对，不要当作报告结果。`--section all` 会连阈值类指标一起算，比其它 section
> 慢得多。

**5. 组间差异与阈值上的逐组 accuracy**（次要终点，零重训）

```bash
# 一次性前置：SWAD 系臂当年只落了 test 预测，val 预测要从平均权重补推理（需 GPU）
# 四个库各提交一次，DATASET ∈ {ham10000, fitzpatrick, mimic, chexpert}
sbatch --job-name=swadval_ham --output=logs/swadval_ham.%N.%A_%a.log \
       --export=ALL,DATASET=ham10000 slurm/dump_swad_val.sh

python scripts/build_group_levels_averaging.py                   # 逐组水平 + gap + 逐组 accuracy
python scripts/build_group_tables.py                             # 出 docs/ 的表与备用 LaTeX
python scripts/build_group_tables.py --rule fpr20_val            # 换阈值规则重出表（敏感性）
```

> 阈值**逐折在该折验证集上选**（组无关的单一全局阈值，主口径 Youden J），施加到该折所有子群，
> 指标折间平均——阈值从不接触评估数据，也不跨折池化。三条阈值规则（`youden_val` /
> `fpr20_val` / `half`）并列是敏感性检查，不是可挑选的口径。
>
> 这一步一次覆盖四个同分布数据集与两个迁移方向，迁移侧沿用 CV-OOF 的跨库预测，
> 阈值取**源域**验证集——目标域没有验证集，部署时能用的只有源域定出的操作点。
> 两类终点都是次要终点，显著性不并入主终点的 Holm family，按 exploratory 读。

## 框架 2：跨数据集迁移

MIMIC ↔ CheXpert 双向。每折三个种子共 15 个模型，每个模型评**整个**目标数据集，
效应要同时通过配对 bootstrap 区间与训练噪声地板 σ_train，且两个方向都通过才算数。

```bash
# 1. 补足种子（fold×seed 解耦；trial 0 即框架 1 已训的模型，不重跑）
sbatch --partition=gpus24 --job-name=oodv2_mimic_erm \
       --output=logs/oodv2_mimic_erm.%N.%A_%a.log \
       --export=ALL,PY_SCRIPT=src/training/train_mimic_resnet18.py,CONFIG_INDEX=2,SWAD=1 \
       slurm/ood_trial.sh

# 2. 全靶评估（两个方向）
sbatch --partition=gpus24 --job-name=oodv2_ft \
       --output=logs/oodv2_ft.%N.%A_%a.log slurm/oodv2_full_target.sh

# 3. 方差分量 + σ_train：两个终点各跑一次，落到确认性家族的 mw/ 与 ov/ 子目录
FAM="${HN_OUTPUTS:-$PWD/../../outputs}/ood_cxr/variance_decomposition_family"
ARMS=erm,swad,groupdro,hyperhead,hyperfusion,hyperadapt
python scripts/ood_variance_decomposition.py --metric marginal_worst --methods $ARMS --out-dir $FAM/mw
python scripts/ood_variance_decomposition.py --metric overall        --methods $ARMS --out-dir $FAM/ov

# 4. 确认性家族：一次调用同时给出两个方向各 9 个对比的 Δ / CI / Holm / σ_train
python scripts/ood_confirmatory_family.py --src $FAM
```

## 框架 3：条件化范围消融

`CondNet` 用一个开关产生四个嵌套的 cell（ERM / 只条件化 fc / fc+layer4 / fc+layer1-4），
四个 cell 共享同一份确定性初始化的基座权重，因此差异只来自条件化范围。

```bash
# 1. 六配置搜索（每个 cell 一次提交，LOCATION ∈ none/head/deep/full）
sbatch --partition=gpus48 --job-name=e1_ham_full_search \
       --output=logs/e1_ham_full_search.%N.%A_%a.log \
       --export=ALL,DATASET=ham10000,LOCATION=full slurm/e1_condnet_search.sh
python scripts/e1_select_config.py --dataset ham10000

# 2. 选定配置补到 3 个种子
sbatch --partition=gpus48 --job-name=e1_ham_full_confirm \
       --output=logs/e1_ham_full_confirm.%N.%A_%a.log \
       --export=ALL,DATASET=ham10000,LOCATION=full,CONFIG_INDEX=2 slurm/e1_condnet_confirm.sh

# 3. 同分布主分析（两个增量的 Holm 家族）
python scripts/e1_ham_analysis.py

# 4. 迁移侧：M→C 全靶评估与分析
sbatch --partition=gpus24 slurm/e2_ood_full_target.sh
python scripts/e2_ood_analysis.py
```

三项介入（报告 §5.4）：

```bash
# ① 换检查点：以验证 worst-group 最优的 epoch 重读全部结论
python scripts/e1_ham_analysis.py --checkpoint worstcase
python scripts/e1_rho_aggregate.py --checkpoint worstcase

# ② 通路 knockout：固定权重下逐一关掉 head / conv 通路，与属性置换的扰动比较
python scripts/e2_knockout_permutation.py

# ③ 关掉 head 通路重训，看 conv 通路能否被优化驱动起来
sbatch --export=ALL,DATASET=ham10000,LOCATION=full_nofc,CONFIG_INDEX=2 slurm/e3_condnet_nofc.sh
python scripts/e3_nofc_analysis.py
```

## HyperAdapt 的两个扩展臂

**预测属性**：属性改由冻结 ImageNet 特征上的 logistic probe 从图像预测，
另配两个零信息对照臂（`perm` 打乱预测、`const` 用训练集边缘先验）。

```bash
sbatch --export=ALL,DS=mimic slurm/p_attr_pred.sh              # 逐折拟合 probe，落盘 p̂
sbatch --partition=gpus48 \
       --export=ALL,PY_SCRIPT=src/training/train_fitzpatrick_hyperadapt_pred.py,ATTR_MODE=soft \
       slurm/p_pred_attr.sh                                     # ATTR_MODE ∈ soft/hard/perm/const
python scripts/p_pred_attr_analysis.py --dataset fitzpatrick
python scripts/p_ood_analysis.py --direction m2c
```

**权重平均**：HyperAdapt 臂加 `SWAD=1` 训练即得（method 记为 `hyperadapt_swad`），
BN 统计另按官方 `update_bn` 重估一份做对照。

```bash
sbatch --export=ALL,DATASET=ham10000,STAGE=id slurm/g_bn_update.sh
python scripts/g_fusion_analysis.py            # 2×2 主表与交互对比
python scripts/g_mde_worstgroup.py             # worst-group 的最小可检出效应
```

## 机制诊断与报告插图

这些脚本回答「属性到底有没有进到网络里」，只读已训练的权重或预测，多数不需要图像。

```bash
# 逐层条件化强度 ρ 与 ρ_between（12 个训练臂 × 4 数据集，纯 CPU，不读图像）
python scripts/named_hn_rho.py

# 消融各 cell 的逐层 ρ。逐 run 的 ρ 有两条产出路径：
#   --mode rho 纯 CPU、只读 checkpoint（画 ρ 图足够）
#   GPU 作业则额外建 logits 表（通路 knockout 分析需要它）
python scripts/e1_rho_and_permutation.py --mode rho     # 纯 CPU
sbatch slurm/e1_rho_logits.sh overall                   # GPU，同时落盘 logits 表
python scripts/e1_rho_aggregate.py                      # 聚合（两条路径产出的都可）

# 决策超平面的逐子群反事实扫掠（判别轴 × 组间调整轴，附反事实 AUC）
python scripts/viz_hyperplane_ham_sexage.py                     # HAM（报告用的 sex+age 版）
python scripts/viz_hyperadapt_hyperplane_mimic.py               # MIMIC
python scripts/viz_hyperadapt_hyperplane_chexpert.py            # CheXpert
python scripts/viz_hyperadapt_hyperplane_fitzpatrick.py         # Fitzpatrick
python scripts/viz_blind_hyperplane.py --dataset ham --method erm   # 属性盲基线的对照（--method erm|swad）
# 迁移几何复用同分布产物，须先跑上面两个 CXR 脚本
python scripts/viz_ood_hyperplane_cxr.py --direction m2c

# 属性扫掠下的 Δθ 权重轨迹（不需要图像）
python scripts/viz_hyperadapt_weight_trajectory.py              # HAM，age 连续扫掠
python scripts/viz_hyperadapt_trajectory_mimic.py               # MIMIC，三属性

# 训练曲线，以及报告第 5 章的四张图
python scripts/plot_training_curves.py
python scripts/plot_report_ch5_figures.py                       # 写入 report/figures/
```

超平面脚本默认读各数据集选定配置的 fold 0 检查点，用 `--ckpt` 可指定其它检查点。

## 产物布局

模型权重、逐折预测与分析 JSON 都写在仓库外的 `$HN_OUTPUTS`（缺省 `../../outputs`），
不进版本控制：

```
outputs/
├── <数据集>/cv5/                 检查点、val 日志、predictions/、selected_configs.json
├── conditioning_ablation/        框架 3 的四个 cell、主口径结果 JSON、逐组水平/gap/accuracy JSON
├── ood_cxr/{m2c,c2m}/            跨库预测与方差分解
└── analysis/                     ρ、Holm 判决、超平面与轨迹产物
```

一次运行由三元组 `(method, config_tag, seed)` 唯一标识，检查点与 val 日志同名，
因此六个配置、多个种子、多个折的产物互不覆盖。同分布的种子由折号决定（seed = 42 + fold），
迁移框架另加两个独立种子（seed = 42 + fold + 10 × trial）以分离训练随机性与划分随机性。
