# 条件化消融实验方案 —— 历史审阅落地日志（changelog）

> 本文件是 `conditioning_ablation_plan.md` 的**历史审阅日志归档**（第二~九轮内部审阅）。
> **⚠️ 这些是历史记录，不是当前有效规格**：凡与主方案 `conditioning_ablation_plan.md` 的 v4 说明或
> 「第十轮（外部评审）」冲突处，**一律以主方案为准**。已知被后续轮次覆盖的旧内容（**勿据此编码**）：
> - MIMIC「5 唯一 cell / 240 run / OOD 推理 95 次」→ 主方案 **6 cell + 3 必跑桥接 / ~330 run（260 网格 +
>   ERM 25 + 桥接 45）/ full-target 推理 ~205 次**；
> - HAM「~720/~760 run / 18–20 cell / +ERM 20」→ **~860 主网格（21 唯一 cell，2×50+19×40）/ +ERM 25（仅 locked）
>   或 50（含 tuned）/ +五臂 60 ≈ 945–970**（canonical base state 使旧 ERM seed0 不可复用；精确以 manifest 为准）；
> - S-sharedA 容量含 fc（19,648 / 2,764,388）→ **不含 fc：DoF 19,136 / 生成器 2,499,680**（§4 主轴 L-all-conv 无 fc）；
> - OOD「k→k 为主口径」→ **full-target 为确认性主估计量、k→k 降 legacy**；
> - 「第四格本方案不运行」→ **训练但不进初始化确认性对照**；
> - 散见 `p_Holm 双基线 + Overall 另列` → 统一 **`p_joint_holm`**；
> - OOD 历史胜利复现族 `m=6` → **`m=8`（含 L-layer34）**；
> - 「cell vs 基线用 seed 交集」→ **OOD 复现族/最小范围判定全 cell 统一 {0,1,2}**（含 L-all-conv）。
>
> 第十轮（外部评审）的落地状态保留在主方案 §12，不迁到此处。

---

**第二轮审阅（14 条）已全部并入规格**，其中 6 个硬阻塞的落地状态：
1. **[x] C→M 需额外训练问题**（#1）：已采**方案 A**（§8.2）——C→M 仅用历史三方法作背景负控制、不训
   CheXpert 受控网格。需在分析时严格标注其为「方法级背景」，不进受控推断。
2. **[x] MIMIC 唯一 cell = 5**（#2，§8.2/§9）：S-sharedA≡L-all-conv；预算 3×40+2×50+ERM20 ≈ **240 run**、OOD 推理 95 次。
   （**已被第十轮覆盖：6 cell / ~280 run / full-target ~160 次**。）
3. **[x] 多-seed ERM/SWAD 补训**（#3，§6.3/§9）：HAM ERM 5 fold×5 seed、MIMIC ERM 补确认 seed、
   SWAD 派生——须计入预算并在 E2/E4 前完成。
4. **[x] 判决用 Holm-p 把关**（#4→第四轮更新，§7.2）：确认性成功=p_Holm<.05 + 点估计≥.01 + Overall 非劣；
   CI 只描述方向/幅度（**非**「全用 CI」）。（**已被第八/九/十轮统一为 `p_joint_holm`**。）
5. **[x] OOD 交集成功 + 分离共享 RQ 与历史胜利**（#5，§7.3）：主对比 S-sharedA vs S-indep；vs ERM/SWAD
   为辅助条件；Holm 族含两基线比较。
6. **[x] MIP scale 与 E_L2 分离**（#6，§4.4.2）：预注册主/辅初始化对比，禁整体归因。

**第二轮其余项已并入**：范围轴容量混杂表述（#7）· 预算模板保共享拓扑（#8）· fold-local tuning 非 nested
CV + 目标域零泄漏（#9）· 常量-A 训练第五臂（#10）· worst-group bootstrap 估计量预固化（#11）· 必要/充分
表述收严（#12）· backbone-level 术语（#13）· 按 cell 分级初始等价自检（#14）· 主 checkpoint 只用 best_overall。

**第三轮审阅（9 条 + 2 报告规范）落地状态**：
- [x] **seed 补给对象 = locked（情况 B）**（#1，§9）：搜索 seed0、locked 补 seed1-2/3-4、tuned 恒单 seed；
  普通 40 / 共享核心 50 run；tuned 多 seed 须显式另加。
- [x] **多-seed estimand 冻结**（#2→第四轮更新，§7.5）：主=平均训练运行效应 Δ̄，主 CI=**固定 seed 集合下的
  cluster bootstrap（只重采样患者/病灶、不重采样 seed）**；次=ensemble-OOF；禁纵向拼接/选最好 seed。
- [x] **OOD 两级成功**（#3，§7.1/§7.3）：①历史胜利复现（**p_Holm<.05 双基线**（第四轮更新）+ Overall 非劣
  vs 两基线）与 ②达 +0.01 阈值分离；OOD Overall 非劣须 vs ERM **且** vs SWAD。
- [x] **CXR 伪属性患者级约束随机化**（#4，§6.2）：HAM 标签层等比例、CXR 患者级约束随机化 + 接受阈值；
  伪属性 seed 与训练 seed 1:1 配对不做笛卡尔积。
- [x] **最小有效范围预定义 + 非单调处理**（#5，§7.6）：比较族预固定、非单调只报「最小通过 cell」不称阈值。
- [x] **MIMIC 五臂预算对齐**（#6，§8.2）：伪/常量-A 训练默认仅 HAM；若上 MIMIC 须 2 臂×2 真不同结构×seed×折
  另计（S-sharedA≡L-all-conv 不算两结构）。
- [x] static-adapter 控制表述收严（#7，§6.2）· OOD BN 重估限 source-only（#8，§6.1）· bootstrap 无效
  replicate 重抽+有效率（#9，§7.4）· D_l 群体等权主 + 频率加权附（规范 A，§6.1）· RQ2 方差为机制次要、
  不单凭方差判正（规范 B，§7.6）。

**第五轮审阅（8 + 3 文档冲突 + 1 审计）落地状态**：
- [x] **locked 配置冻结表**（#1，§9）：全 regime locked=网格中心 lr1e-04/wd1e-04（在 6 配置内、seed0 复用搜索）；
  batch/epochs 允许跨数据集不同、同 regime 内一致。
- [x] **Holm 族逐项冻结**（#2，§7.3）：5 个族 + 各 m 明确列表；确认性单侧；L-head 不进 backbone 阶梯族。
  （**已被第十轮更新：OOD 历史复现族 m=6→8，含 L-layer34**。）
- [x] **bootstrap p 值算法冻结**（#3，§7.4）：零假设中心化 + (1+·)/(B+1) + 固定 seed/索引 + ≥5000（MIMIC 用
  向量化/自适应停止或据实记录 B）；Overall 非劣并入 `p_joint=max(p_fair,p_noninf)` 一并 Holm。
- [x] **最小有效范围限 backbone 阶梯**（#4，§7.6）：只在 L-ds⊂L-layer4⊂L-all-conv 内；L-head 单列性质对照。
  （**已被第十轮更新：阶梯补 L-layer34**。）
- [x] **OOD 预测聚合口径 + assertion**（#5，§8.2）：source fold k→target fold k、target 样本每 seed 恰一次、
  五折互斥并集全 target、禁纵向拼接。（**已被第十轮覆盖：改 full-target 主口径，k→k 降 legacy**。）
- [x] **marginal worst 候选群体集合冻结**（#6，§7.4）：固定候选列表、排除 unknown、最低正/负阈值、资格只按
  真实标签/属性、全模型/seed/bootstrap 共用。
- [x] **HyperFusion 初始化定稿 3-cell**（#7，§4.4.2）：zero-E / MIP-E / MIP-noE；rank=32 触发条件（rank16−rank4
  ≥+.005）；CheXpert ID 明确为可选补充。
- [x] **null_condition 可学习防退化**（#8，§6.2）：独立可学习 null embedding + 单元测试断言训练后偏移非零。
- [x] 文档冲突清理：§12「判决全用 CI」→Holm-p；「两层 bootstrap」→固定 seed 集 cluster bootstrap；E4「C→M
  负控制」→「M→C 受控 + C→M 历史背景」；RQ2「或降低波动」→公平确认性/方差次要机制。
- [x] **E-audit 设计阶段精度审计**（§11，零新训）：CI 宽度 / ±0.01 等效功效 / 5000 次运行时间 / 无效重采样率。

**第四轮审阅（5 + 2 + 1）落地状态**：
- [x] **主 CI 唯一冻结**（#1，§7.5）：cluster bootstrap **只重采样患者/病灶、不重采样 seed**；删「或混合模型」
  可选项；训练随机性另报为稳定性、不进主 CI。
- [x] **共同 seed 交集**（#2，§7.5）：范围对比 {0,1,2}、共享对比 {0-4}、cell vs 基线取交集；脚本 assertion。
- [x] **Holm-p 闭合多重性**（#3，§7.2/§7.3）：确认性成功=**p_Holm<.05 + 点估计≥.01 + Overall 非劣**；
  CI 下界仅描述方向；OOD 交集须两比较均过 Holm。（**已统一为 `p_joint_holm`**。）
- [x] **算力更正**（#4，§9）：MIMIC ~240 run + OOD 推理 95 次；HAM ~720 run（+伪/static-A 臂）。
  （**已被第十轮覆盖：MIMIC ~280（+桥接 ~310–325）run / full-target ~160 次；HAM ~760 run**。）
- [x] **跨数据集属性契约**（#5，§8.2）：age 统一 bin/编码、公共类别集、冻结 unknown token、source-only 规则、
  禁看 target 改映射；引用 `run_ood_cxr_cv.py` 映射。
- [x] 范围 cell 精确模块 manifest + 嵌套 assertion（§3）· 常量-A 训练用 null/零条件向量（§6.2）·
  §0.3 改为「方向性假设 / 待检验经验假设」（措辞）。

**第七轮审阅（6 + 小修正）落地状态**：
- [x] **候选群门槛填值 + HAM 组数订正**（#1，§7.4）：n_pos/n_neg ≥ 20（E-audit 后终定）；HAM 主终点 = age
  **4 组** min（订正「6 组」）；写出 worst 公式（HAM=age 4、CXR=sex∪race∪age）。
- [x] **OOD Holm 族去重**（#2，§7.3）：S-sharedA vs ERM/SWAD 只在历史复现族；OOD 共享族 m=1。
- [x] **p_joint 非劣参照 = 被比较模型**（#3，§7.1/§7.3）：结构对结构非劣；T_NI=ΔOverall+0.005、H0:T_NI≤0、
  同 bootstrap 索引。
- [x] **MIMIC bootstrap 定死**（#4，§7.4）：B=5000 固定；超时才用预注册顺序停止（检查节点 + max5000，禁「刚好
  显著就停」）；E-audit 措辞改「有效率 ≥95%」。
- [x] **删剩余可选性**（#5）：伪/static-A 训练仅 HAM（§8.2）；CheXpert ID 默认不跑、条件触发（§8.3）；rank32
  用 HAM val 触发（§5.1）。
- [x] **桥接细节封死**（#6）：HyperFusion 3-cell 全 full-additive（§4.4.2）；null embedding 同定义/同 init/
  逐运行独立优化（§6.2）。
- [x] 小修正：head-only 术语替代「最浅」（§0.2/§0.3/§3）· §7.6 RQ2 去重段 · §12 拆「设计冻结/实现验收」。

**第八轮审阅（3 + 4 + 1）落地状态**：
- [x] **HAM 五臂冻结**（#1，§8.2）：结构 = L-ds & L-all-conv、seed = {0,1,2}、2 结构×2 臂×3 seed×5 折 = 60 run。
- [x] **跨库属性契约具体编码表**（#2，§8.2）：sex/race/age 三行编码 + 候选组 + race 合并引用 build 脚本 +
  unknown 兜底；共用同一 JSON。
- [x] **MIMIC bootstrap 定死**（#3，§7.4）：默认 B=5000（E-audit 靠向量化压时间）；仅不可行时用严格族级顺序
  停止（Clopper–Pearson + Bonferroni 覆盖 + 整族判决稳定 + max5000）。
- [x] **成功判据统一 `p_joint_holm`**（#4，§7.2/§7.3）：p_fair/p_noninf/p_joint/p_joint_holm 命名；成功=p_joint_holm<.05+点估计≥.01。
- [x] **E-audit 门槛调整禁用模型效应**（#5，§7.4）：只依标签/属性计数 + 有效率 + 可计算性。
- [x] **命名/清单清理**（#6/#7）：S-sharedA=「HyperAdapt-style backbone sharing」（完整 HyperAdapt=L-all+head）；
  E6 CheXpert 条件触发；HyperFusion 第四格「本方案不运行」。（**已被第十轮更正为「训练但不进初始化确认性对照」**。）
- [x] **6 配置搜索网格冻结表 + JSON 哈希**（可复现性，§9）：idx0–5 明表，locked=idx2，引用 `configs/search_grid.json`。

**第九轮审阅（3 + 3 + 1）落地状态**：
- [x] **worst-group 候选集合 regime 固定**（#1，§7.4）：𝒢_HAM=age 4 组、𝒢_{M→C}=sex∪race∪age；与 cell 是否
  条件化无关，ERM/SWAD/全 HN/全 seed 共用。删「该 cell 条件化的轴」表述。
- [x] **bootstrap CI = percentile**（#2，§7.4）：95%=2.5/97.5 分位，TOST 90%=5/95 分位。
- [x] **HyperFusion 容量/形式对照公共控制**（#3，§4.4.2）：统一近零 + E_L2 off（使 HF-lowrank-mul 严复用 L-ds）。
- [x] **全文统一 p_joint_holm**（#4，§7.2/§7.3/§7.6/§10）：含 fairness + 非劣，不再 p_Holm 双基线 + Overall 另列。
- [x] **HAM 五臂全局统一**（#5，§6.2/§8.2/§9）：L-ds & L-all-conv、seed{0,1,2}、伪属性 seed 一一配对、固定 60 run。
- [x] **ERM 非劣地位 = 关键次要、非确认性门槛**（#6，§7.1/§7.3）：结构成功只由结构比较 p_joint_holm 决定。
- [x] **精确 cell manifest（E0 交付）**（#manifest，§9/§11）：去重（HF-lowrank-add-ctrl≡HF-lowrank-add、
  第四格≡HF-full-add-ctrl 等）；据 manifest 派生精确预算。
