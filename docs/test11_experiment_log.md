# test11 实验日志 —— 锦标赛淘汰制 vs 快速二阶段筛选

## 0. 系列定位（★根本方法改进，不沿用 test7 编号★）

选择/筛选机制属进化框架的**根本方法**层，与 test7 系列（适应度/观测/环境迭代）
性质不同，独立为 **test11 系列**。test7g（单阶段全评 10 局、无 CRN 的旧基准）与
test7h（现行 CRN 两阶段快速筛选）**仅作外部参照线**，不在本系列内续训。
代码基座 = test7h v2 全套（适应度 v2 W3/CAP4、CRN 公共随机数、弱连接屏蔽 20%、
类正态变异），**唯一实验变量是筛选器**（`evaluate_population_gpu` 分派，
test11.py:1100 附近）。

## 1. 研究问题

test7 系列现行"快速二阶段筛选"（K1=3 全种群粗筛保 410 → K2=10 精评）是否可以被
**锦标赛淘汰制**替代：每轮全体幸存者加赛 K 局新 CRN 库、指标加权累计后**两两配对**
淘汰一半，直到剩 ELITE_SIZE。锦标赛是否**更快**（每代成本、达标代数）、**更稳妥**
（选择保真度、曲线稳定性、终局质量）？

## 2. 两臂设计（预注册）

| | A 臂 two_stage（现行） | B 臂 tournament（实验） |
|---|---|---|
| 机制 | 排序截断两阶段 | 逐轮加赛随机配对淘汰赛 |
| 局数 | 阶段1 全体×3 → 阶段2 幸存410×10 | 每轮 ×3，2048→1024→512→256 |
| 每代局次 | 10244 | 10752（**+5%**） |
| 精英判据 | 累计 13 局 | 累计 9 局（可用 `--tourn-final-pass` +3） |
| 淘汰证据 | 1638 个休 3 局、154 个休 13 局 | 第1轮出局休 3 局、第2轮 6 局、第3轮 9 局 |
| 配对随机性 | 无（全局排序） | 有（同 pair 强强相遇可爆冷，见下） |

共同起点：`--seed 42`（初始化+进化全局 RNG，test7h 没有的新开关）+ 同 CRN_SEED
20260827（每代库逐位相同）。除筛选器外代码路径逐位一致；two_stage 模式在 test11
内即 test7h 原路径（selfcheck 校验）。

**已知语义差异（selfcheck 5 验证）**：随机配对淘汰赛允许爆冷——16 体 toy 例中
真值第 7 名晋级精英而第 8/10 名首轮出局。这不是 bug，是锦标赛的多样性效应；
其代价/收益正是本实验要度量的问题。`--tourn-pairing rank`（按累计分排序砍半）
为消融变体，只在 Phase 0 离线对比，不花训练预算。

## 3. 实验流程（run_test11_ab.bat 一键链）

| 阶段 | 内容 | 预计耗时 |
|---|---|---|
| [1/5] | Phase 0 保真度，test7h 收敛种群（`experiments/test11_selection_fidelity.py`，8 trial × 真值 24 库，库规范化+记忆化省 ~50% 算力） | ~1.5 h |
| [2/5] | Phase 0，test7g 中期种群（次要，4 trial × 16 库） | ~45 min |
| [3/5] | A 臂从零 20 代 | ~1.5 h |
| [4/5] | B 臂从零 20 代 | ~1.5 h |
| [5/5] | 终局验收（`experiments/test11_ab_verdict.py`，40 库配对复评 + 判定表） | ~25 min |

**启动前置条件：GPU 空闲（test7h 103-153 跑完后）。** 门判定不过自动停止。
产物：`test11_econ_{two_stage,tourn_k3}_*`、`results/test11_selection_fidelity_{t7h,t7g}.json/png`、
`results/test11_ab_verdict.json/png`、`logs/test11_*.log`。

Phase 0 真值口径：全体 ×24（或 16）库 CRN 评估 = 每个体真值 food/适应度；
4 选择器 × T trial（gen=5000+t 库），指标 = 选中精英集真值 food 均值（主）、
与真值 top-256 重合率、真 top-16 覆盖、选中集内 Kendall τ。

## 4. 预注册验收标准（20 代单 seed 口径）

### 门（Phase 0，tourn_random vs two_stage 逐 trial 配对差）
- 配对差 ≥ −0.3 → PASS；[−0.5, −0.3) → RISKY（放行+警告）；< −0.5 → **STOP**（训练臂不启动）。

### 快
- B 每代评估耗时 ≤ 1.10×A（账面 +5%）。
- B@20 的 EliteFood ≥ A@20 − 0.5（不落后）；达 θ=min(30, A@20) 的代数 B ≤ A。

### 稳
- Phase 0：B 选中集真值分 ≥ A−0.3 且 trial 间 std ≤ 1.2×A（本轮主要统计证据）。
- 在线：avg_food 无 ≥10 代冻结窗口（7g 教训：avg 冻结 25 代）；EliteFood 最大回撤 B ≤ A+0.5；
  终局逐库 margin 的 std B ≤ A（淘汰赛不应引入更大库间波动）。

### 优（终局 40 库同库配对，库内分 = 精英集平均适应度）
- B 显著优：胜率 ≥ 90% 且 margin ≥ +1.5；等效：|margin| < 0.5；B 更差：胜率 ≤ 10% 且 margin ≤ −1.5；其余不确定。

### 结论矩阵
- B 显著优，或等效且更快/更稳 → **锦标赛列为转正候选**（建议 51 代复核后再改默认）。
- 等效无优势 → 保留两阶段，归档结论。
- B 更差 → 归档（Phase 0 数据用于解释：爆冷代价 vs 递增证据收益）。

### 限制声明
- 单 seed × 20 代只覆盖快速爬升期（7g 参照：gen20 avg_food ≈ 15，仍在爬升）；
  平台期行为不在本实验范围，Phase 0 用收敛种群（test7h checkpoint）补这一维度的统计证据。
- 终局复评样本 = 各臂最终 256 精英 + best_model 单体，非全程多次采样。

## 5. 参照线（外部，不重跑）

- test7g 原始曲线：40 平台（best≈40 含极值虚增 +7，真实 ≈33），avg 自 gen23 冻结 25 代
  （`experiments/diagnose_test7g_plateau.py`）。
- test7h 续训（7g@51 断点起，v2）：EliteFood 32.2(gen51) → 35.3(gen102)，尾 12 代 ≈35.1，
  每代 eval 220-530s（`test7h_econ_history.json`）。
- 本实验从零跑 v2 基线本身是**新数据**（现行曲线是 7g v1 前 50 代 + 7h v2 续训的混合）。

## 6. 结果（待填）

### Phase 0 保真度（results/test11_selection_fidelity_t7h.json / _t7g.json）

| testbed | selector | food_mean | ±std | overlap | top16cov | tau | pairedΔ |
|---|---|---|---|---|---|---|---|
| t7h | two_stage | 待填 | | | | | ref |
| t7h | tourn_random | 待填 | | | | | |
| t7h | tourn_rank | 待填 | | | | | |
| t7h | tourn_final | 待填 | | | | | |
| t7g | （同上） | 待填 | | | | | |

门判定：待填。

### 训练曲线（20 代）

| 臂 | EliteFood 终值 | 尾5均值 | 达 θ 代数 | 冻结窗口 | 最大回撤 | s/gen |
|---|---|---|---|---|---|---|
| A two_stage | 待填 | | | | | |
| B tourn_k3 | 待填 | | | | | |

### 终局 40 库配对验收

margin（B−A）= 待填 ± 待填 | 胜率 = 待填 | 判定 = 待填 | 结论 = 待填
