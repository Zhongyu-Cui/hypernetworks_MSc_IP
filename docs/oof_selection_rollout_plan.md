# CV-OOF 模型选择方案 —— 铺开工作清单（可打勾，供新 session 执行）

> 建立日期：2026-07-10。本清单把「Overall 早停 + 折均 marginal-worst 选 config + pooled-OOF 报公平」
> 方案铺开到全 5 数据集。**自包含**：新 session 无需上下文即可执行；决策依据见
> `docs/selection_signal_and_denoising.md`（选择信号实证）、`docs/routeA_ham_cv_oof_fairness.md`（HAM OOF 判决）。
> 状态标记：`[ ]` 未开始 · `[~]` 进行中 · `[x]` 完成。每个复选框 = 一件可独立完成并验证的任务。

---

## 0. 方案速览（三决策 × 三信号 × 三数据）

| 决策 | 信号 | 数据 | 现行是否已对 |
|------|------|------|-------------|
| ① epoch/早停 | **Overall AUC**（patience 5） | 该折 val | ✅ **已是默认（configs `monitor: val_auc`），本方案不改** |
| ② config 选择（6→1） | **折均 marginal-worst**（minimax，平手用折均 Overall 破） | 5 折 val 折均 | ⚠️ 需改（现 PAPILA 用 canonical、HAM 复用单-split） |
| ③ 评估/报告 | **canonical worst-group** + Overall + gap（样本级配对 bootstrap） | 5 折 **OOF 池化** test | ⚠️ 需铺开 |

**铁律**：选择只用 val，评估只用 pooled-OOF test，两者永不交叉。每折 train/val/test 互斥；OOF = 每样本恰被其
留出折预测一次；config 全局选一个、只用 val 选 → OOF 从未参与选择（三重无泄漏）。

**marginal-worst** = 只在单属性子群取最差 AUC（HAM: sex 2 + age 4 = 6 组），**丢弃样本极少的交叉小格**
（如 F|80+ n=27/pos4）。它是**稳定的选择信号**（逐-epoch 抖动 −35%、秩相关从有害 −0.27 变无害 +0.10）；
交叉子群的真实公平**留到 pooled-OOF 上用 canonical-worst 报**（那里 n 够大、小格不再进噪声地板）。

---

## 1. 现状快照（2026-07-10）

| 数据集 | CV 折 | train 脚本 --cv | 6×5 CV 搜索 | 每-run 成本 | 备注 |
|--------|-------|-----------------|-------------|-------------|------|
| **HAM10000** | ✅ cv5 | ✅ 全 4 | ✅ **作业 70771–74 已完成** | ~3 min | 只差重选+报告 |
| **PAPILA** | ✅ cv5 | ✅ 全 4 | ✅ 已完成（原生全-CV） | ~2 min | 只差切换选择信号 |
| **Fitzpatrick17k** | ❌ | 部分（resnet18/hyperadapt 有；hyperhead/hyperfusion 无） | ❌ | ~3 min | 便宜 |
| **CheXpert** | ❌ | ❌ 全无 | ❌ | ~25 min | ~50 GPU-时 |
| **MIMIC-CXR** | ❌ | ❌ 全无 | ❌ | ~90 min | ~180 GPU-时，最后做 |

已有可复用资产：`slurm/c_search_papila_cv.sh`、`slurm/c1_ham_cv_search.sh`（array 0–29 模板）；
`scripts/select_pareto_papila_cv.py`（折均 canonical minimax，改 marginal）；`scripts/ham_cv_oof_significance.py`
（HAM OOF 池化+显著性，泛化为通用）；`scripts/selection_strategy_probe.py`（选择信号诊断）。

---

## 2. Phase S — 选择/评估基础设施（数据集无关，一次实现，最高优先）

- [x] **S1 折均 marginal-worst 选择器** `scripts/select_config_cv.py` ✅ 2026-07-10
      读 `outputs/<ds>/cv5/val_logs/` → 逐 config 对 5 折的 **val marginal-worst**（子群向量丢 `|` 键取 min）
      求折均 → minimax 选最高，平手用折均 val Overall 破 → 打印选定 config_tag。参考 `select_pareto_papila_cv.py`
      （把 canonical 改 marginal + dataset 参数化）。**判据**：5 数据集通用、`--self-test` 通过、对 HAM 输出一个确定 config。
      （实现：纯 argmax(marginal-worst, overall)；`--write-json` 落选定配置供 S2 消费；HAM 选定 erm=lr3e-04_wd1e-03 等，
      订正了路线 A 复用单-split 的捷径。）
- [x] **S2 通用 OOF 报告器** `scripts/cv_oof_report.py` ✅ 2026-07-10
      泛化 `ham_cv_oof_significance.py`：`--dataset` 参数化 → 池化选定 config 的 5 折 test 预测 → OOF 完整性断言
      （并集=全集、y_true 逐元素对齐）→ 算 canonical worst / marginal worst / Overall / gap → 样本级配对 bootstrap
      （HN/SWAD/ROC vs ERM）+ Overall DeLong。**判据**：HAM 复现 `routeA` 的 D-A1 数字（ERM Overall 0.874 等）。
      （回归验证：旧配置下 ROC 精确复现 0.8696；ERM/HN 因作业 70771–74 重训而略变 = 管线正确。`--config-json`/`--config` 消费选定配置。）
- [ ] **S3（可选）** 把 marginal+3ep-MA 稳健信号并入 `selection_strategy_probe.py`（`--signal {canonical,marginal,marginal_ma}`），供选择信号诊断，非主链必需。

---

## 3. Phase D — CV 折生成（build 脚本加 --cv 5）

- [x] **D1** `build_mimic_splits_nofinding.py --cv 5` ✅ 2026-07-10：患者级 GroupKFold（`patient_id` 分组），生成 `data/splits/mimic_cxr_nofinding/cv5/fold{0..4}/`，断言 5 折 test 并集=199356=全集、折内无患者跨 split（正例率 ~30%）。
- [x] **D2** `build_chexpert_splits_nofinding.py --cv 5` ✅ 2026-07-10：患者级 GroupKFold（`patient_id` 分组），生成 `data/splits/chexpert_nofinding/cv5/fold{0..4}/`，断言 5 折 test 并集=138644=全集、折内无患者跨 split。
- [x] **D3** `scripts/build_fitzpatrick_cv_splits.py`（实验 F 遗留，图像级 StratifiedKFold）：✅ CV 折已在
      `data/splits/fitzpatrick17k/cv5/fold{0..4}/`，已验证并集=16012、5 折 test disjoint、折内互斥。

---

## 4. Phase T — 训练脚本 CV 化（补 --cv/--fold）

仿 `train_ham10000_*.py` / `train_fitzpatrick_resnet18.py` 的 `resolve_paths`（CV 产物写 `outputs/<ds>/cv5/`，
不碰单-split 结果；seed=42+fold）。

- [x] **T1** `train_mimic_{resnet18,hyperhead,hyperfusion,hyperadapt}.py` 加 `--cv/--fold` ✅ 2026-07-10（镜像 resolve_paths，CV 产物写 outputs/mimic_cxr/cv5/；语法+--cv 已验证）。
- [x] **T2** `train_chexpert_{resnet18,hyperhead,hyperfusion,hyperadapt}.py` 加 `--cv/--fold` ✅ 2026-07-10（镜像 resolve_paths，CV 产物写 outputs/chexpert_cxr/cv5/；语法+--cv 已验证）。
- [x] **T3** `train_fitzpatrick_{hyperhead,hyperfusion}.py` 加 `--cv/--fold` ✅ 2026-07-10（resnet18/hyperadapt 已有；镜像 resolve_paths，CV 产物写 cv5/，语法+--help 已验证）。

---

## 5. Phase R — 逐数据集执行（按成本从低到高；每完成一个停下汇报）

> **C 阶段训练纪律**（CLAUDE.md）：**先 `ssh biomedia-slurm` 再 `sbatch`**；重型 HN **等-batch**
> （HyperAdapt bs128 → gpus48）；**`--exclude=semois`**；**预先报告** + **提交后查状态**（节点故障换 `--nodelist`）。
> 每方法一次提交 = array 0–29（6 config × 5 折）；ERM 带 `SWAD=1` 逐折派生；ROC 用 `derive_roc --output-dir .../cv5` 逐折后处理。

- **R-HAM**（数据已备，仅需 S1/S2）✅ 2026-07-10（结果见 `docs/oof_regime_results.md`）
  - [x] 6-config × 5 折 CV 搜索（作业 **70771–74**，120 run，0 错误）
  - [x] S1 折均 marginal-worst 重选 config（订正捷径：erm→lr3e-04_wd1e-03、hyperfusion→lr1e-04_wd1e-04、hyperadapt→lr3e-05_wd1e-04）
  - [x] S2 出 OOF 报告（canonical worst + 显著性）。**判决：config 修正未推翻路线 A**——HyperFusion worst-group 单点转正但配对 bootstrap n.s.；SWAD 仍最强；HyperFusion 仅 Overall 对 ERM 显著 +0.011。ROC 对新 config 重派生；删除未选中 config checkpoint 释放 12.35 GB。
- **R-PAPILA**（全-CV 已备，仅切换选择信号）✅ 2026-07-10（结果见 `docs/oof_regime_results.md`）
  - [x] S1 marginal-worst 重选（erm/hyperhead/hyperfusion=lr3e-04_wd1e-04、hyperadapt=lr3e-04_wd1e-03；ROC 已最新无需重派）
  - [x] S2 出 OOF 报告。**判决同 HAM/路线 A**：全 3 HN worst-group ≤ ERM 且 n.s.（n=420 功效低）、SWAD 最强、无 HN Overall 赢；joint 小格 Male|≤60(n_pos4) 仍进地板→公平以 marginal 口径读。删除未选中 config checkpoint 释放 12.35 GB。
- **R-Fitzpatrick**（便宜；需 D3 + T3 先就位）✅ 2026-07-10（结果见 `docs/oof_regime_results.md`）
  - [x] 新建 `slurm/c4_fitz_cv_search.sh`（仿 `c1_ham_cv_search.sh`）
  - [x] 提交 4 方法 6×5 CV 搜索（作业 **70963–66**，120 run，0 错误；HyperAdapt bs128 等-batch → gpus48）
  - [x] S1 选（erm=lr3e-04_wd1e-03 等）+ ROC 派生 + S2 报。**判决同 HAM/PAPILA/路线 A**：论文本尊数据集上全 3 HN worst-group 对 ERM 仅 +0.004~0.011 且 n.s.、对 SWAD 一律显著劣；仅 HyperFusion Overall +0.013 显著。删除未选中 config checkpoint 释放 12.35 GB。
- **R-CheXpert**（~50 GPU-时；需 D2 + T2）✅ 2026-07-10（结果见 `docs/oof_regime_results.md`）
  - [x] 新建 `slurm/c3_chexpert_cv_search.sh`
  - [x] 提交 4 方法 6×5 CV 搜索（作业 **71099–71102**，120 run，0 错误）
  - [x] S1 选（erm=lr1e-04_wd1e-04 等）+ ROC 派生 + S2 报。**判决：最弱信号数据集（I(Y;A|X)≈0）上 HN 对 ERM 在 Overall 与 marginal worst-group 均统计显著为负（Δ≈−0.005，n=138k 使微差可检出，实际可忽略）**；SWAD 打平 ERM 仍优于 HN。坐实充分性反方向。删除未选中 config checkpoint 释放 12.35 GB。
- **R-MIMIC**（~180 GPU-时，最后；需 D1 + T1）—— ID 完成 ✅ 2026-07-12（结果见 `docs/oof_regime_results.md`）
  - [x] 新建 `slurm/c2_mimic_cv_search.sh`
  - [x] 提交 4 方法 6×5 CV 搜索（作业 **71235–71238**，120 run；1 瞬时 IO 故障换节点重跑 71395 补齐）
  - [x] S1 选（erm 等=lr1e-04_wd1e-04、hyperadapt=lr1e-04_wd1e-03）+ ROC 派生 + S2 报（sig 用 1000 次 = 作业 71587，n=199k 上 2000 次超时限）。**判决：唯一出现 HN 显著 worst-group 赢的数据集——HyperAdapt(最深) marginal worst +0.003 显著优 ERM、Overall +0.0015 显著优，但幅度可忽略且对 SWAD 不赢；HyperHead(最浅)显著劣。SWAD 不败。** 删除未选中 config checkpoint 释放 12.35 GB（保留选定 config 供 OOD）。
  - [x] ✅ **OOD（双向）2026-07-13**：`run_ood_cxr_cv.py`（source cv5 逐折 checkpoint → target cv5 逐折 test 池化=target OOF）+ `build_target_test_loader(cv/fold)` + `slurm/c5_ood_cv.sh`；作业 71635（推理）+ 71668/71669（sig 1000 次）。**判决：强烈方向不对称——MIMIC→CheXpert 深层 HN(HyperFusion/HyperAdapt) marginal worst-group 显著超 ERM 与 SWAD、HyperAdapt Overall 亦然（全研究唯一 HN 稳健胜 SWAD 处）；CheXpert→MIMIC 全 HN 显著输。赢的方向=属性信号源(MIMIC I(Y;A|X)>0)方向。** ⚠️ **本条已双重作废**：(a) averaging 口径推翻「强烈方向不对称」(见 `oof_regime_results.md` §2)；(b) 新 OOF V-info 闸门判 **MIMIC 全轴决定性 0**，不存在「属性信号源方向」。 详见 `docs/oof_regime_results.md` OOD 节。

---

## 6. Phase A — 汇总重算（OOF 口径）

- [ ] **A1** 用 S2 的 OOF 口径重算 `results_summary.md` **D3/D5**（5 数据集 × 6 方法 Overall / canonical-worst / gap）。
- [ ] **A2** 重算 **D6 显著性**（pooled-OOF 样本级配对 bootstrap + DeLong，全 3 HN × 3 基线矩阵）。
- [ ] **A3** 每表脚注标注三决策口径：**早停=Overall、config 选择=折均 marginal-worst、报告=canonical-worst on pooled-OOF**；
      并注明相对旧单-split 口径的差异（尤其 worst-group 判决可能翻新，参照 HAM 路线 A：单-split 伪抬升→OOF 打平/微负）。

---

## 7. Phase W — 文档

- [ ] **W1** 新建 `docs/oof_regime_results.md`：全 5 数据集 OOF 口径结果 + 与旧单-split 的 diff + 结论修订。
- [ ] **W2** 更新 `docs/comparison_protocol.md` 现状快照：主口径切到 CV-OOF，注明单-split 仅作交叉验证。

---

## 8. 起手顺序与依赖

```
S1,S2（选择器/报告器）─┐
                      ├─→ R-HAM（数据已备，先跑通全链，验证 S1/S2）→ R-PAPILA
D3,T3 ─→ R-Fitz ──────┘
D2,T2 ─→ R-CheXpert
D1,T1 ─→ R-MIMIC（最后，最贵）
                      └─→ A1,A2,A3 → W1,W2
```

**先做 S1+S2+R-HAM**：HAM 数据已在，能立刻打通「折均 marginal 选 config → OOF 报告」全链并对账路线 A，
作为其余数据集的模板与回归基准。**便宜的 Fitz 次之**，CheXpert、MIMIC 因算力最后。

**每完成一个数据集（R-*）停下汇报**（`slurm-workflow-prefs` 纪律）。
