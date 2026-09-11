# test12 实验日志 —— ego 观测重写 + 适应度多因子化（当前主线，留守根目录）

> 2026-08-28 起。基于 test7h 全套机制做"感知重写"：把食物感知改为 ego 帧
> 方位+距离双通道，并把习惯/效率因子逐步并入适应度（v3→v7）。
> 与 test7b 并列为仓库仅有的两条活跃实验线（留守根目录）。

## 0. 继承关系与归档位置

```
test7h v2 (CRN 筛选方法论, experiments/test7_series/test7h.py)
   └→ test12 (观测重写 + 适应度 v3→v7, 本日志) —— 留守根目录
        ├→ 外围 A/B（习惯化 E1/E2/E2c、教师标定、孤岛诊断）在 experiments/
        └→ deploy_test12/ （AutoDL 云端部署包）
```

- **起点**：test7h 外围 A/B 全线失败（混沌吸引子硬抵抗、imitC 长跑卡 29、
  prox 无效）→ 预注册判定转向"观测增维/重写"路线（见
  docs/test7_series_experiment_log.md §2）。
- 机制全套继承：CRN 公共随机数、两阶段淘汰、类正态变异、曼哈顿观测度量、
  评估期弱连接屏蔽。
- 文件：根目录 `test12.py`（活跃）、`test12_econ_*` / `test12s1_*` /
  `test12s2_*`（运行产物）、`run_test12_smoke.bat`（冒烟）、
  `deploy_test12/`（云端部署）。

## 1. 主要目的

不改环境规则、只改感知与适应度：突破 7h 系平台（7g 口径 ~40 真实 ≈33 /
7h 口径 EliteFood ~35），消除"空间挤压自撞"主死因，让有序行为与追食行为
共存生长。

## 2. 观测编码演进（obs[8:16]，32 维不变，OBS_ENC_VERSION 守卫）

对照实验定位：`experiments/diag_test12_vs_7h.py`。

| 版本 | 编码 | 结果 |
|---|---|---|
| v1 | 绝对系 + 原始 sig | 卡 1.0 食 |
| v2 | 绝对系 + sig×K | 仍卡 1.0（量级无关） |
| v3（现行） | **相对系 ego**：obs[8:12] 四相对方位信号 ×K（前/右/后/左随头转，sig=clamp(û·dir,0,1)×K）+ obs[12:16] 四方位距离倒数（sig×K/曼哈顿距离） | 突破——方位+距离双通道可精确恢复食物相对向量 |

关键认知：绝对系要求网络先学会"绝对方位 ⊗ 头朝向"绑定，随机初网络零初始
相关、选择无梯度；相对系 ego 与 7h 扇区同为自体系、同输入量级（对准 ≈K）。
'OBS_ENC_VERSION' 断点校验：7h 旧 checkpoint 编码不同但维度相同，resume-pop /
seed-model 按编码版本拒绝混用。

## 3. 适应度演进（FITNESS_VERSION 3→7）

| 版本 | 形式 | 要点 |
|---|---|---|
| v3 | food + 孤岛折减 | 孤岛惩罚：每次吃食采样点蛇头可达空间 < ISLAND_THRESHOLD×(总空间−蛇长) 记局触发，按触发率线性折减（v1"任一局触发整体×0.1"实测与食物数正相关、反向压选择，已废）；动机=7h 行为学"主死因=空间挤压自撞" |
| v4 | + te 项 | 转弯效率项 TURN_EFF_W=3.0（继承 7h v2 校准） |
| v5 | food + eff | te 转为遥测 + 精英配额（TURN_EFF_W 注释"v5 起休眠"）；v5.1 三因素行为遥测（edge/conn/straight） |
| v6 | food + eff + 0.6·H | 加法习惯项（E2c 预注册对照验证） |
| **v7（现行）** | **乘法式** `fitness = food × (1 + W_EFF·eff + W_STRAIGHT·straight + W_EDGE·edge_share) × conn` | 乘法意图：习惯/效率因子随食物等比放大——加法 tie-breaker（v6 预算 <1 食）实测搬不动后期坏行为，乘法在 50+ 食物段 20% 习惯折损即 10+ 分；括号内各项∈[0,1]、因子非负 → fitness ≥ food×conn，无需守卫项；conn=蛇身外自由空间连通块数倒数 1/n（单连通=1，块越多折越重，严格惩罚尾部占用）；v7 起食物与习惯不再有字典序保证（设计意图本身） |

## 4. 习惯化研究线（E1/E2/E2c，experiments/exp_habit_*.py）

- **E1 指标判别力**（exp_habit_metrics.py）：教师解法器（全局图+抢先规划）
  应表现出高 edge_pref / 高 conn，best 模型（局部观测）作对照——同库 40 局
  验证三因素指标的方向与区分度。
- **E2 习惯配额 A/B**（exp_habit_ab.py，预注册）：习惯配额全关 vs
  HABIT_EDGE/CONN/STRAIGHT_ELITE 各 8，256 种群×12 代×种子{0,1} 同 CRN；
  P1 食物不伤 / P2 有效 / P3 健康（撞己占比不升）。
- **E2c 适应度 v6 vs v5 演化对照**（exp_fitness_v6_ab.py，预注册）：
  256 种群×20 代×种子{0,1}；P1 食物不损（修正 E2 对标判据的方向错误）/
  P2 习惯有效 / P3 健康 / P4 食物优先保持（Kendall τ）。
- **净结论**：配额与加法习惯项均搬不动后期坏行为 → v7 改乘法式把三因素
  并入适应度（HABIT_W 注释"v6 加法习惯项已由 v7 乘法式取代"；行为习惯
  不再走配额）。
- 可视化：plot_habit_curves.py（results/habit_curves.png，best vs 教师
  逐存亡步三因素曲线）、demo_history_plot.py（过程图右栏"适应度来源堆叠"
  的合成 30 代视觉验收，results/demo_v6_history.png）。

## 5. 参照线与诊断工具

- **教师解法器**（全局图+抢先规划，40/40 通关）= "物理可达上限"参照；
  diag_teacher12_fitness.py 做适应度逐项分解（food/eff/te/孤岛各 Term 的
  量级、饱和与区分度检查）。
- **孤岛研究**：diag_test12_island.py、diag_test12_island_teacher.py、
  eval_best12_islands.py（best 模型 N 局终局报告 + 每局孤岛曲线
  reach_ratio 吃食采样点）。

## 6. 运行记录与当前状态

- 本地长跑：test12_econ_history.json（主跑）、`--name s1` / `--name s2`
  独立前缀跑（test12s1/s2_* 产物，test12s2_econ_history.png）；econ 真实
  过程图 results/econ_real_history.png（过程图右栏已支持 v7 乘法归因分解：
  food×conn / eff / 少转弯 / 边角四份之和恰=总分）。
- 云端：deploy_test12/（README_AutoDL.md、run_test12.sh、requirements.txt），
  2026-08-29 起部署 AutoDL 长跑；run_test12_smoke.bat 本地冒烟。
- **当前状态（2026-08-31 工作区）**：v7 乘法式为现行适应度；s1/s2 前缀跑、
  展示图与 test12.py v7 头注为本日志撰写时尚未提交的最新进展（随本次整理
  一并入库）。长跑/云端结果归档后再补"结果与结论"终稿。

## 7. 结果与结论终稿（2026-09-02 补写：Phase-0 相位归因）

test12 主跑 170 代 elite ~32.7 / best ~38.9（每 30 代约 +3 缓爬）的"后期
上不去"，经 `experiments/exp_phase_diagnostic.py`（Phase-0，预注册门 P0）
归因为**后期游戏段模式缺失**（前后期质变），获 P0-CONFIRM(STRONG)：

- test12 冠军（latest_gen_best）从 len≤30 的教师身体直接出生续评与自然局
  同段完全一致（ratio 1.00）；**len 40 起崩塌**（coil 0.25 / teacher 0.50）、
  len 50 深崩（0.33-0.67），死因 91-98% 空间挤压自撞。
- test7b 冠军（67 食）崩得更早（len≥30 ratio≈0.00，8-15 步即死），且其
  自然局转弯密度全部长度段恒 0.81-0.85——追食模式从未切换。
- 教师解法器（回路+安全捷径双模式）同状态同钟可达 93 分 ⇒ 瓶颈是网络
  策略类缺"模式切换"，非物理上限、非适应度噪声（7g 已排除）、非新局部
  最优（C 线语境）。

即：v3→v7 的适应度演进与 E2/E2c 的习惯塑形都在"教一个只会追食的策略
更守规矩"，而它缺的是**按游戏阶段换挡的表达结构**——适应度侧杠杆已到
天花板。后续走 test14 期相激素路线（局内动态调制，见
docs/test14_experiment_log.md），本文件不再更新。
