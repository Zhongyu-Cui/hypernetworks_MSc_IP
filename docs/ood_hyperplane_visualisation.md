# OOD 超平面可视化（MIMIC↔CheXpert）—— ERM / SWAD / HyperAdapt

> **定位**：把 `docs/hyperplane_subgroup_visualisation.md` 的方法搬到 `docs/ood_experiment_v2_results.md`
> 的设定上——**source 训练的权重，评 target 的数据**，覆盖 3 方法 × 2 方向。
> 回答的是**机制/几何**问题（跨库迁移在模型自己的判别坐标里长什么样），
> **不是**性能/公平性结论的来源——后者权威口径见 `docs/ood_experiment_v2_results.md`。
>
> **实现**：`src/utils/ood_hyperplane_viz.py`（同帧叠加 + 位移度量）、
> `scripts/viz_ood_hyperplane_cxr.py`（统一驱动）；target 侧标准图复用
> `src/utils/hyperplane_cxr_runner.py` 与 `scripts/viz_blind_hyperplane.py`（仅换 checkpoint）。
> **产物**：`outputs/ood_cxr/hyperplane_viz/{m2c,c2m}/`。
> **状态**：2026-08-02 建成，6 个 (方向 × 方法) 全部跑通。结果均为
> **fold0 / trial 0 / seed42 单点**，跨 fold/seed/trial 复核未做。

---

## 0. 一句话结论

> 跨库迁移在这套坐标里**只做了两件事**：把特征云沿判别轴平移了 0.14–0.25 SD（**校准**，不改排序），
> 并把类间分离度压缩了 0.09–0.29 SD（**判别**，这才是 AUC 掉的地方）。**超平面本身一动没动。**
> HyperAdapt 的属性条件化**原样搬到了 target**（逐子群 Δlogit 与 ID 的 r=0.98–0.99），
> 且**在 target 上同样惰性**（反事实 AUC 跨度 ≤0.0008、口径 C max\|ΔAUC\| ≤0.0007）——
> 包括那条在 CheXpert 上学到的 `Female/Non-White/≥60` −0.39 logit 下压，
> 被逐字搬到 MIMIC（−0.28），而该组在 MIMIC 恰是 ≥60 里健康率**最高**的一格 ⇒ **方向反了**。

---

## 1. 三个参照系（读任何数字前必须分清）

| 记号 | 权重 | 数据 | 用途 |
|---|---|---|---|
| **src-ID** | source | source cv5/fold0 test | 同帧叠加的左 panel（**模型不变**，数据换库） |
| **OOD** | **source** | **target cv5/fold0 test** | 本工作现算 |
| **tgt-ID** | target | target cv5/fold0 test | **迁移落差**参照（**数据不变**，模型换库） |

- `OOD` 与 `tgt-ID` 用**同一批 target 图像**（同 fold、同 16 格分层抽样配额与 seed=42）⇒ 可逐点配对；
  `tgt-ID` 直接从既有 ID 产物 npz 读出，**零额外 GPU**。
- `src-ID` 与 `OOD` 用**同一份权重** ⇒ 组间/域间差异 100% 来自数据。
- 🛑 **`OOD − src-ID` 是被混杂的量**（跨数据集比 AUC，把「target 本身多难」和「迁移掉多少」搅在一起）；
  **只有 `OOD − tgt-ID` 才是迁移落差的干净读数**。图中三栏条形图的第 2 栏就是前者、第 3 栏是后者，
  并列画出正是为了强制分辨这两者。本文档下方一律以第 3 栏为准。

---

## 2. 方法

### 2.1 一个恒等式：超平面在 ID 与 OOD 之间**逐位相同**

- ERM/SWAD 的 `w`、HyperAdapt 的 `w_eff(a) = W + ΔW(a)` 与 `fc.bias` **只由 checkpoint 决定**，
  与评估数据无关；
- ⇒ ID 与 OOD 的决策线是**同一条（组）线**，变的只有特征云 `f(x)`。

因此「迁移出了什么问题」在这套坐标里**必然**表现为**点云相对一条不动的线平移/形变**，
而不是「边界变了」。这不是经验发现，是构造性的，图内 suptitle 已写明。

> **自带验证**：HyperAdapt 的判别基 `P` 只由 `w_eff` 构造 ⇒ OOD 运行自算的 `P` 必须与 ID 存盘的
> 逐位相同。实测两方向的 `basis_diagnostics` 与 ID 完全一致
> （MIMIC `e2_perp_var_ratio` 0.416 / `s1/s2` 1.4256；CheXpert 0.783 / 2.2168）⇒ 同帧构造正确。
> 代码内另有 `assert_same_frame`（比 \|cos\|，容忍 SVD 符号自由度）作 checkpoint 一致性断言。

### 2.2 同帧构造（精确，无需重跑 ID）

投影基取 **src-ID 那次运行的 `(P, mu)`**（已落盘），OOD 特征用同一 `P`、同一 `mu` 投影：

```
U_id  = (f_id  − mu_id) @ P_id      （ID 运行已存盘，直接读）
U_ood = (f_ood − mu_id) @ P_id      （本工作现算）
```

HyperAdapt 的 ID npz 存的是 8 组**反事实**扫掠 `[8, N, ·]`；部署时每样本只走**自己真实属性**那条，
故取 `logits[group_ids[i], i]` 得**事实**口径，与 blind 模型走同一套下游代码。

### 2.3 两个位移量（尺度 = ID 云在各轴上的 SD）

| 量 | 含义 | 能不能解释 AUC |
|---|---|---|
| `d∥` | 沿 `e1`（判别轴）的质心平移 | **不能**——纯操作点/校准偏移，单调平移不改任何组内排序 |
| `d⊥` | 沿 `e2`（判别轴正交补）的平移 | **不能**——对该分类器完全不可见 |
| **类间分离度** `sep` = (y1 质心 − y0 质心) 沿 `e1` | 两类被推开多远 | **能**——这是唯一与判别挂钩的量 |

---

## 3. 三个方法学陷阱与处理

| # | 陷阱 | 处理 |
|---|---|---|
| 1 | **分层抽样抹掉了 `P(Y)`**：16 格分层把 y=1 率拉到 0.44–0.50，`ood_experiment_v2_results.md` §5.1 的「P(Y) 30.2%→8.2%」在图里**根本看不到** ⇒ 图上的 `d∥` 不能当部署时的操作点漂移 | 另报**零 GPU 的自然患病率对照**（§4.2）：直接读既有预测落盘（source fold0 test vs **完整 target**，均**未抽样**），给出真实患病率下的同向读数 |
| 2 | **跨数据集 AUC 差被混杂**（见 §1） | 三栏条形图并列 `d∥` / 混杂差 / **受控迁移落差**，结论一律只引第 3 栏 |
| 3 | **`d⊥` 跨方法不可比**：`e2` 是每个模型自己的轴（HyperAdapt 由 `w_g − w_bar` 定，blind 由组间特征均值定），不同 checkpoint 的 `e2` 指向不同方向 | `d⊥` 只在**同一模型的 ID vs OOD 之间**读；**禁止**比较不同方法 `d⊥` 的符号或大小 |

---

## 4. 结果（fold0 / trial 0 / seed42）

### 4.1 几何与迁移落差（16 格分层子样本）

| 方向 | 方法 | src-ID AUC | **OOD** | tgt-ID | **迁移落差**<br>OOD−tgt-ID | `d∥` (SD) | `d⊥` (SD) | 类间分离度 ID→OOD |
|---|---|---|---|---|---|---|---|---|
| M→C | ERM | 0.8253 | 0.8229 | 0.8573 | **−0.0345** | −0.248 | +0.235 | 1.12 → 0.89 |
| M→C | SWAD | 0.8276 | 0.8335 | 0.8535 | **−0.0200** | −0.239 | +0.273 | 1.12 → 0.94 |
| M→C | HyperAdapt | 0.8310 | 0.8417 | 0.8509 | **−0.0092** | −0.216 | −0.216 | 1.15 → 1.06 |
| C→M | ERM | 0.8573 | 0.7970 | 0.8253 | **−0.0282** | +0.144 | −0.310 | 1.23 → 1.11 |
| C→M | SWAD | 0.8535 | 0.7932 | 0.8276 | **−0.0343** | +0.170 | −0.294 | 1.21 → 1.02 |
| C→M | HyperAdapt | 0.8509 | 0.7897 | 0.8310 | **−0.0413** | +0.201 | +0.194 | 1.19 → 0.90 |

> ⚠️ M→C 的 `OOD` 列（0.823–0.842）看起来**高于** `src-ID`（0.825–0.831），这**不是**「迁移反而变好」——
> CheXpert 的 16 格分层子样本被抽成近平衡（y=1 率 0.439），比 MIMIC 的子样本容易。**正是陷阱 #2。**
> 同一批图像上 `tgt-ID` 是 0.851–0.857，落差仍为负。

### 4.2 自然患病率对照（**未抽样**：source fold0 test vs 完整 target；零 GPU）

| 方向 | 方法 | 患病率 ID→OOD | 校准**所需** logit 平移 | **实测**平移 | 占比 | 平均预测概率−患病率<br>ID → OOD |
|---|---|---|---|---|---|---|
| M→C | ERM | 0.299 → 0.082 | −1.56 | −0.51 | 33% | +0.013 → **+0.122** |
| M→C | SWAD | 0.299 → 0.082 | −1.56 | −0.69 | 44% | −0.000 → **+0.093** |
| M→C | HyperAdapt | 0.299 → 0.082 | −1.56 | −0.56 | 36% | +0.039 → **+0.149** |
| C→M | ERM | 0.083 → 0.302 | +1.56 | +0.51 | 33% | +0.009 → **−0.147** |
| C→M | SWAD | 0.083 → 0.302 | +1.56 | +0.75 | 48% | +0.006 → **−0.146** |
| C→M | HyperAdapt | 0.083 → 0.302 | +1.56 | +1.02 | 65% | +0.009 → **−0.151** |

「所需平移」= `logit(π_target) − logit(π_source)`，即若模型在 source 上已校准、要在 target 上仍校准
所必须的整体 logit 偏移。

### 4.3 逐子群（OOD 事实口径）

| 方向 | 方法 | OOD 最弱子群 | AUC | 逐组迁移落差 均值[范围] | 逐组 `d∥` 范围 |
|---|---|---|---|---|---|
| M→C | ERM | Male / White / ≥60 | 0.7593 | −0.0396 [−0.065, −0.018] | [−0.415, −0.105] |
| M→C | SWAD | Male / White / ≥60 | 0.7715 | −0.0215 [−0.043, −0.006] | [−0.431, −0.070] |
| M→C | HyperAdapt | Female / Non-White / ≥60 | 0.7812 | −0.0125 [−0.030, **+0.001**] | [−0.416, −0.069] |
| C→M | ERM | Male / White / ≥60 | 0.7708 | −0.0215 [−0.047, −0.013] | [−0.091, +0.461] |
| C→M | SWAD | Male / White / ≥60 | 0.7612 | −0.0289 [−0.047, −0.008] | [−0.071, +0.478] |
| C→M | HyperAdapt | Male / White / ≥60 | 0.7776 | −0.0322 [−0.047, −0.014] | [−0.024, +0.477] |

**6 个 cell 中 5 个的 OOD 最弱子群都是 `Male/White/≥60`**，与两库台账「主公平轴 = Age」一致。

---

## 5. 五个发现

### 5.1 迁移损失**全部**落在类间分离度上，与 `d∥` 无关

跨 6 个 cell 的组级相关（**n=6 描述性统计，非推断，无 CI**）：

| 相关 | Pearson | Spearman |
|---|---|---|
| Δ类间分离度 ↔ 迁移落差 | **+0.837** | **+0.943** |
| \|`d∥`\| ↔ 迁移落差 | +0.211 | **+0.029** |

⇒ **判别轴上的平移预测不了任何 AUC 变化，类间分离度的压缩几乎完全预测了它。**
这是 `hyperplane_subgroup_visualisation.md` §4.3「口径 B ≠ 口径 C」在 OOD 设定下的直接量化：
`d∥` 属口径 B（logit 幅度），只能谈校准；能谈性能的只有分离度/AUC。

### 5.2 两个方向都只走了 1/3–2/3 的应有平移，方向对、幅度不足

模型的分数**确实**朝正确方向动（M→C 降、C→M 升），但只走了所需量的 33%–65%，
残余系统偏差 **M→C 高估 +0.09…+0.15 / C→M 低估 −0.146…−0.151**。

与 `ood_experiment_v2_results.md` §5.2 的 CITL 读数**同号同量级**
（该文档报 M→C CITL +0.10~+0.13、C→M −0.15~−0.17）⇒ 本工作的自然患病率读数与既有校准分析自洽。

> ⚠️ 「占比」列不是校准优劣的排序依据：三方法的 ID 起点 mean logit 不同
> （C→M：−3.50 / −4.18 / −4.33），故走得多不等于落点准——三者的 OOD 残余偏差几乎相同（−0.146…−0.151）。
> **方法间的校准比较必须以 `ood_experiment_v2_results.md` §5.2 的 15-replicate 配对 bootstrap 为准**，
> 本表是单 seed 单 fold 的描述性读数（且「平均预测概率−患病率」≠ CITL，定义不同）。

### 5.3 HyperAdapt 的条件化**原样搬到了 target**

逐子群 Δlogit（相对参照组 `Male/White/≥60`）在 ID 与 OOD 之间几乎逐位重合：

| 方向 | ID 上的 8 个 Δlogit | OOD 上的 8 个 Δlogit | Pearson r |
|---|---|---|---|
| M→C | −0.052, 0, +0.010, −0.025, −0.093, −0.084, −0.068, −0.020 | −0.058, 0, −0.006, −0.031, −0.105, −0.096, −0.080, −0.033 | **0.992** |
| C→M | +0.002, 0, +0.091, +0.030, +0.070, +0.008, −0.031, **−0.390** | +0.011, 0, +0.087, +0.039, +0.086, −0.004, +0.044, **−0.281** | **0.980** |

通路占比同样稳定（Shapley fc 占比：MIMIC 26.0% → M→C 23.7%；CheXpert 27.5% → C→M 24.6%），
逐子群 fc 占比的形状也保住了（CheXpert 的 `F/NW/≥60` 71.7% → MIMIC 上 67.4%）。

**这是预期的**：`w_eff(a)` 只由 checkpoint 决定，conv 调制也是固定的乘性掩码；
只要没有 concept shift（`ood_experiment_v2_results.md` §5.1 已证无），条件化就会照搬。

### 5.4 ……而它在 target 上**同样惰性**

| 口径 | MIMIC ID | **M→C OOD** | CheXpert ID | **C→M OOD** |
|---|---|---|---|---|
| 反事实 AUC 跨度（8 组） | 0.0009 | **0.0007** | 0.0003 | **0.0008** |
| 口径 C `max\|ΔAUC\|` fc / conv | 0.0000 / 0.0005 | **0.0001 / 0.0004** | 0.0000 / 0.0002 | **0.0001 / 0.0007** |

⇒ 换一个库、换 3.7 倍的患病率，**条件化对排序的贡献依然是 0**。
与 `ood_experiment_v2_results.md` §5.1b 的 knockout 判决（属性信息净贡献 +0.0003、CI 含 0）
及 `docs/conditional_v_information_gate.md`「MIMIC/CheXpert 全轴 CI<0」完全自洽，
且是**独立于预测台账的第三份证据**（这里用的是反事实扫掠，不是 knockout）。

### 5.5 🛑 搬过去的那条偏移，在 target 上方向是**反的**

CheXpert 上 HyperAdapt 学到的最显著动作是把 `Female/Non-White/≥60` 整体下压 **−0.39 logit**
（`hyperplane_subgroup_visualisation.md` §5.4 的「条件化几乎全集中在一个子群」）。
迁到 MIMIC 后这条偏移被逐字保留（−0.28）。但两库该组的实际健康率排位**不同**：

| 子群（≥60 四格） | CheXpert 健康率 | MIMIC 健康率 |
|---|---|---|
| Male / White / ≥60 | 0.0574 | 0.2262 |
| Male / Non-White / ≥60 | **0.0550（最低）** | 0.2443 |
| Female / White / ≥60 | 0.0565 | 0.2421 |
| **Female / Non-White / ≥60** | 0.0582 | **0.2875（最高）** |

在 MIMIC 上该组健康率是四格里**最高**的，却被压低 0.28 logit ⇒ **组先验方向搞反了**。

**判读（严格按口径规则）**：这是**口径 B** 的读数。**口径 C 判定「什么也没发生」**——
该组在 C→M 上的反事实 AUC 与其余组无实质差别（跨度 0.0008），它不是 C→M 最弱格。
故**不得**写成「HN 因为搬错先验而在 C→M 上变差」。它的正当含义是：
**prior shift 下最该被重新估计的那部分（组先验）恰恰是 HN 完全没有重估的部分**——
zero-shot 迁移没有任何目标端适应手段，条件化只能照搬 source 的先验。
这为 `ood_experiment_v2_results.md` §5.1 机制表的「机制 C 不适用（无目标端适应手段）」提供了几何图像。

---

## 6. 与 `ood_experiment_v2_results.md` 的对账

| 该文档的结论 | 本工作的对应观察 | 是否一致 |
|---|---|---|
| §5.1 无 concept shift | 条件化模式跨库照搬（r=0.98–0.99）；子群健康率的 `<60 高 / ≥60 低` 秩序两库一致 | ✅ |
| §5.1b 属性信息净贡献 ≈0（+0.0003，CI 含 0） | 反事实扫掠：OOD 上 8 组 AUC 跨度 ≤0.0008、口径 C ≤0.0007 | ✅ 独立佐证 |
| §5.1b 条件化真实发生但无用（100% 样本 logit 改变，中位 \|Δlogit\|=0.0696） | 逐组 Δlogit 幅度 0.01–0.28，口径 C 仍 ≈0 | ✅ 同构 |
| §5.2 M→C CITL +0.10~+0.13（高估）、C→M −0.15~−0.17（低估） | 自然患病率残余偏差 +0.093~+0.149 / −0.146~−0.151 | ✅ 同号同量级 |
| §5.2 M→C HyperAdapt 校准显著更差 | M→C 残余偏差 HyperAdapt +0.149 > ERM +0.122 | ⚠️ **同向但不构成复核**（单 seed、不同定义） |
| §5.2 C→M HyperAdapt 校准显著更好 | C→M 残余偏差 −0.151 vs ERM −0.147（几无差别） | ⚠️ **未复现**——单 seed 分辨不了此量级；以该文档为准 |
| §1/§2 H1 不成立：M→C +0.0031 优、C→M −0.0025 劣（方向不一致） | 受控迁移落差：M→C HyperAdapt **−0.0092（三方法最小）**、C→M **−0.0413（三方法最大）** | ✅ **方向不对称同号**（但口径、n 均不同，非复核） |

🛑 **本文档的任何数字都不得用来判定 H1–H4。** 该文档的主估计量是 full-target × 5 fold × 3 trial
= 15 replicate，而这里是 fold0/trial 0/seed42 单点、且在 16 格分层子样本上；
§4 已证「效应量 < 训练噪声 σ_train ≈ 0.006–0.009」，而本表多数差异正落在该量级内。

---

## 7. 判读规则与禁止说法

**必报**（四项缺一不可）：
1. 用的是哪个参照系（src-ID / OOD / tgt-ID），以及 AUC 差是混杂的还是受控的；
2. 抽样口径（16 格分层、实得 n、y=1 率被拉平这一事实）；
3. `d∥`/`d⊥` 的 SD 尺度基准（ID 云）与类间分离度；
4. 与 full-target 15-replicate 台账的口径差异声明。

| ❌ 不得说 | 理由 |
|---|---|
| 「OOD 下超平面变了 / HN 在 target 上调整了边界」 | 线只由 checkpoint 决定，ID/OOD 逐位相同（§2.1） |
| 用 `d∥` 或 `d⊥` 解释 AUC / worst-group 差异 | 单调平移不改排序；`d⊥` 对 `w` 不可见（§2.3、§5.1） |
| 「M→C 迁移后 AUC 反而变高了」 | 跨数据集比较被分层抽样与库难度混杂（§4.1 注、陷阱 #2） |
| 比较不同方法的 `d⊥` 符号/大小 | `e2` 是各模型自己的轴，跨 checkpoint 不可比（陷阱 #3） |
| 由 §5.5 推出「HN 因搬错先验而在 C→M 变差」 | 那是口径 B；口径 C 判定该组无事发生 |
| 拿本文档数字复核 H1–H4，或与台账数字并列 | 单点 vs 15-replicate、分层子样本 vs full-target（§6） |
| 「组间/域间分得开 ⇒ 需要属性条件化」 | 继承 §9.6 红线：那测的是 `I(A;X)`，HN 要的是 `I(Y;A|X)` |

---

## 8. 边界与未做

- **单 fold（fold0）、单 trial（0）、单 seed（42）**，无 CI、无置换检验。§5.1 的相关是 **n=6 的组级
  描述性统计**；§5.3 的 r=0.98/0.99 是 **n=8 组级**读数。跨 fold/seed/trial 稳定性**未查**。
- **不含 HyperHead / HyperFusion**（沿用 OOD v2 的范围限定）⇒ **不检验条件化深度轴**，
  任何深度结论不得由本工作推出。ROC 亦未做（后处理只改阈值，在本图上表现为决策线平移）。
- **分层抽样**是为保证 8 个子群都有支撑而必须的，代价是抹掉 `P(Y)`；§4.2 的自然患病率对照只在
  **整体**层面补回，**逐子群**的自然患病率读数未做（需全量扫掠，HyperAdapt 代价 ×8）。
- 只做了 **ID↔OOD 同帧**一种叠加。`tgt-ID` 只以数字进表，**未**与 OOD 同帧作图
  （两者 `w` 不同 ⇒ 不存在共同的判别帧，强行同帧会误导）。
- §5.5 的「先验方向反了」是 **fold0/seed42 单点观察**，且该格在 CheXpert 上本就是抽样后最小的格
  （n=336）；是否稳定需 5 fold × 5 seed 复核，**现阶段不得写入论文结论**。

---

## 9. 复现

```bash
# 6 个 (方向 × 方法) 全跑（medimg env，纯推理，本地 GTX 1050 Ti 即可）
PYTHONPATH=. python scripts/viz_ood_hyperplane_cxr.py

# 单个 cell
PYTHONPATH=. python scripts/viz_ood_hyperplane_cxr.py --direction m2c --method hyperadapt

# 只重画逐子群条形图（改渲染后省一遍前向，从既有 metrics JSON 读）
PYTHONPATH=. python scripts/viz_ood_hyperplane_cxr.py --rerender_bars
```

- **前置**：需先有两库自训模型的 ID 产物（`outputs/{mimic_cxr,chexpert_cxr}/hyperplane_viz/*.npz`），
  由 `viz_hyperadapt_hyperplane_{mimic,chexpert}.py` 与 `viz_blind_hyperplane.py` 生成；
  缺失时脚本**快速失败**并给出提示。
- **checkpoint 寻址**与 `src/training/run_ood_cxr_full_target.py::checkpoint_path` 同一规则
  （source `cv5/selected_configs.json` 的 S1 选中配置，SWAD=`_averaged`、余 `_best_overall`，seed42↔fold0）
  ⇒ 本图用的权重与 OOD 台账**逐位一致**。
- **`--seed` 必须保持 42**：否则 target 上抽到的不是同一批图像，`OOD` 与 `tgt-ID` 不再可配对。
- **成本**：HyperAdapt 每方向 ~13–15 min（8 子群 × 3.5k–4k 张），blind 每 cell ~2 min；
  全套 ~35 min，**不需上集群**。

### 产物清单（每 cell）

| 文件 | 内容 |
|---|---|
| `ood_domain_shift_<tag>.png` | **ID ǀ OOD 同帧叠加**（本工作核心图） |
| `ood_subgroup_shift_<tag>.png` | 逐子群三栏：`d∥` ǀ 混杂 AUC 差 ǀ **受控迁移落差** |
| `ood_hyperplane_metrics_<tag>.json` | 域级位移 + 三参照系逐子群表 + 自然患病率对照 |
| `hyperplane_{discriminative,pca}_<tag>.png`、`delta_logit_pathway_<tag>.png` | HyperAdapt：target 上的 8 子群标准图（复用既有渲染） |
| `blind_hyperplane_<method>_<tag>.png`、`blind_subgroup_panels_<method>_<tag>.png` | ERM/SWAD：target 上的标准图（复用既有渲染） |

---

## 10. 相关文档

- 方法学母本（ID 版，含五个陷阱与三口径规则）：`docs/hyperplane_subgroup_visualisation.md`
- OOD 性能/校准权威口径：`docs/ood_experiment_v2_results.md`、`docs/ood_experiment_v2_preregistration.md`
- 信号闸门：`docs/conditional_v_information_gate.md`
- ID 侧性能/公平性权威口径：`docs/comparison_protocol.md`、`docs/oof_regime_results.md`
