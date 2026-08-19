# 备忘：把 Attribute Knockout 反事实扩到 ID CV-OOF 主矩阵

> 建立日期：2026-08-19。**状态：待办（未执行）**。
> 起因：论文叙事已决定**不引入 conditional V-information 信号闸门**（导师不建议），
> R1 合成剂量-反应实验一并移出论文。原本由「闸门 + R1」承担的两项论证
> ——(a) 为什么一致 null 不是实现失败；(b) HN 实现本身有效（正控制）——
> 需改由**不含信息论的经验证据**承担，其中 knockout 反事实是新机制脊的主承重。

---

## 1. 为什么要补

新的解释脊（见 report §4.4.5 + §4.6）是四条证据收敛：

1. 四种消费敏感属性的路径（ROC / GroupDRO / 三档 HN / CondNet 范围轴）全部打平 ERM；
2. **knockout 反事实 `real − marginal ≈ 0`**，且配对设计消掉共同变异 ⇒ 高功效的零；
3. 权重轨迹 / `ρ_between > 0` ⇒ 模型确实按属性生成不同权重；
4. 超平面可视化 ⇒ 条件化发生但判别面几乎不动。

**问题**：第 2 条目前**只覆盖 OOD**。现有实现是
`scripts/ood_knockout_analysis.py`（OOD v2 双向）与
`scripts/e2_knockout_permutation.py`（范围消融的 M→C 格）。
ID CV-OOF 的五库主矩阵（Study 1）**没有 knockout 读数**，
而 Study 1 恰恰是全文的参照矩阵。机制脊在主矩阵上留空是叙事弱点。

## 2. 为什么便宜

knockout 是**纯推理**：拿现成 checkpoint，只改「模型看到什么属性」再前传，
**不重训**。三个推理条件：

- `real`：真实属性；
- `permute`：属性全局置换（保边缘分布）；
- `marginal`：`p(y|x) = Σ_a p(a) p(y|x,a)`，即「移除属性信息」的正确定义。

`real − marginal` = 属性信息的净贡献；`real − permute` 额外惩罚「喂错属性」，会高估效应。
成本量级：单卡几分钟 / 库 × 方法 × fold；MIMIC/CheXpert 稍重（n≈14–20 万）但仍是推理级。

## 3. 执行清单（待做）

- [ ] 复用 `scripts/ood_knockout_analysis.py` 的三条件推理逻辑，抽出与 OOD 无关的核心，
      新建 ID 版（暂名 `scripts/id_knockout_analysis.py`），输入 = cv5 各 fold 的 HN checkpoint
- [ ] 覆盖范围：5 数据集 × 3 HN（HyperHead / HyperFusion / HyperAdapt）× 5 fold
      （HASWAD 可选；ERM/SWAD/GroupDRO 无属性输入，不适用）
- [ ] 口径与 Study 1 完全一致：**averaging 聚合**（逐折算→折间平均），
      marginal 划分为主终点、canonical 为次；配对 cluster bootstrap（患者/病灶级）
- [ ] 选中配置取 `cv5/selected_configs.json`，**不得**用顶层 5-seed 单-split 配置
      （见记忆 `oof-regime-selected-configs`）
- [ ] 报告 `real − marginal` 与 `real − permute` 两个量，明确前者为主口径
- [ ] 量化配对反事实的功效优势：其 σ 应比臂间比较小一个量级，需实测给出数值
- [ ] 结果写入 `docs/id_knockout_results.md`，并接进 report §4.4.3（估计量）与 §5 机制节

## 4. 注意事项

- **不要**在任何产物里重新引入 nats / 互信息 / V-information 口径；本实验的表述
  全部停留在「AUC 差值」层面，这正是它替代闸门的价值所在。
- `marginal` 的 `p(a)` 用**训练集**边缘分布还是**评估折**边缘分布，需事前声明并全库统一
  （建议训练集，与「模型被训练时所见的先验」一致）。
- 若某子群在某折只剩一类，AUC 无定义——沿用 Study 1 既有的缺格处理，不得改口径。

## 5. 相关文档

- `docs/ood_experiment_v2_results.md`（OOD knockout 现有读数）
- `docs/conditioning_ablation_e1_pathway_knockout.md`（通路 knockout × 属性置换）
- `docs/hyperplane_subgroup_visualisation.md`、`docs/oof_regime_results.md`
