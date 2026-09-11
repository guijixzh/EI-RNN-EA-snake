# test16b 实验日志 —— 评估提速（fast-eval）+ 阶段2 对半精评（halving）

**日期**：2026-09-05 ｜ **代码**：`test16b.py`（test16a 子版本，基因组/CRN/适应度公式逐字一致）
**一句话**：不换实验、不改进化语义，把云端 simp 臂（pop4096, K1=12→保2048→K2=24, 精英1024=1/4）
单代 eval ~1320s 的成本砍到 ~1/4.4（fast-eval）并再省 19% 评估局数（halving）；
`--resume-pop test16a_simp_checkpoint.pth` 可直接从 gen210 断点续训。

---

## 0. 背景与瓶颈诊断

云端 test16a `--fit-mode simple` 长跑至 Gen 210（BestFood≈55，σ_ε≈1.09），单代 eval 1142–1403s，
剩余 110 代 ≈ 40 GPU·h，且 S>75 后自适应 K2 会进一步加码。表面瓶颈是"抗噪多次筛选"
（每代 98,304 局 = 阶段1 4096×12 + 阶段2 2048×24），实测真正瓶颈是**每环境步的调度开销**：

- `_fitness_simple` 只吃 4 列（food/steps_last/act1/act2），其余 15 列全是遥测，
  但遥测全部跑在每步热路径上：teacher.act 每步（mismatch）、reach_ratio 在每次吃食触发
  满轮 100 次 max_pool、habit_conn_score 每 4 步最多 100 轮 min_pool + 每 8 轮一次同步；
- obs40 的 16 扇区射线是 8×10 步 ×2 组的 Python 循环（~960 op/步），
  且 test16a.py:940-951 食物方位块存在**逐字重复执行两遍**的冗余；
- 每步 3 次强制 CPU-GPU 同步（`al.any()`/`ate_now.any()`/`all_done().item()`）；
- launch-bound 下墙钟 ∝ 扫描次数，分块内"全死才 break"被苟活个体拖尾。

**剖面实证**（本地 RTX 5070 Laptop，pop64, K1=4, K2=8, `--profile-eval`，同步计时）：

| 路径 | 单代总时 | obs | fwd | teacher | edge(含conn) | reach | step |
|---|---|---|---|---|---|---|---|
| 基线（test16a 逐字路径） | 9.2s | **75.7%** | 4.4% | 13.9% | 2.8% | 1.0% | 2.2% |
| fast-eval | 1.7s（5.4×） | 45.6% | 25.1% | 11.4% | 4.1% | 1.1% | 12.7% |

基线 obs+遥测+同步占比 **93.4%**（预注册判据 ≥70%，达成）；fast-eval 后 obs 绝对值
7.0s→0.78s（9×）。

## 1. 改动清单

### A 层 `--fast-eval`（默认开；`--no-fast-eval` 逐字回退 test16a 路径）
1. `_obs40_fast`：16 扇区射线 → 预计算 [4,8,G,2] 偏移表一次 gather + cumsum 首命中
   （~960 op → ~40 op）；自检 8 对拍 `_obs40` **逐位一致**（随机状态 + 四朝向遍历）。
2. 删除食物方位重复执行块。
3. 遥测降频不减列：teacher/8 步、edge/4 步（新增 `edge_steps` 归一化分母，
   straight 分母 `beh_steps` 保持全存活步）、conn/32 步、reach/16 步窗口近似；
   列 5/11/12/13/14/15/16/18 全保留，**采样口径变化**（均值统计一致，逐值可微移）。
4. 消除每步 `al.any()`/`ate_now.any()` 门控同步（掩码无条件累计）；`all_done` 每 8 步一查
   （死个体冻结，最多多跑 7 步，逐行指标不变）。
5. `EVAL_MEM_FRAC` 0.55→0.85（关 fast 时自动回 0.55，保证分块形状与 test16a 一致）；
   分块 OOM 自动减半重试。
6. 阶段2 分块按阶段1 实测 steps_last 降序组装（长命个体聚一块；行→bank 映射与分组
   无关，纯调度）。

### B 层 `--stage2-halving`（默认开；`--no-stage2-halving` 关）
阶段2 幸存者全体先打 A=6 局 → 按 key(K1+A) 取前 ELITE_SIZE 再打 B=18 局。
**精英最终 K = K1+A+B = 36 与 test16a 完全一致（精英评估预算一食不动）**，
注定落选的后一半幸存者每体省 18 局。云端档阶段2 49,152→30,720 局（-37.5%），
全代 98,304→79,872 局（-18.8%）。`--stage2a-eps/--stage2b-eps` 可调（对齐精英 K=K1+K2）。
备选 `--k2-floor-dynamic`（默认关，halving 关闭时生效）：自适应 K2 地板改用
need=⌈(SEL_CV·S/δ)²⌉−K1（σ_ε≤δ 预注册判据；S=55 时 K=20 而非 36）。

### 其他
`--profile-eval` 分块计时探针；`--seed` 固定 CPU+CUDA 种子；`run_seeded.py`
给无 --seed 的 test16a 注入同种子（A/B 对照用）。

## 2. 兼容性验证

- `python test16b.py --selfcheck`：自检 0–7 全过 + 新增自检 8（fast obs 逐位等价）。
- `--smoke`（fast+halving 全开）与 `--smoke --no-fast-eval --no-stage2-halving`（B0 路径）均通。
- AUTO_RESUME 续训冒烟：从 smoke 断点 next_gen=3 正确接续。
- **test16a 断点导入**：`load_checkpoint7('test16a_checkpoint.pth')`（econ 臂 gen40,
  pop4096）通过全部版本守卫（N/OBS_DIM/ACTION_DIM/BRAIN_VERSION='sparse1'/
  OBS_ENC_VERSION）并成功 unpack；simp 断点为同构 payload，云端直接
  `--fit-mode simple --resume-pop test16a_simp_checkpoint.pth` 续训（FITNESS_VERSION=9 一致，
  best 追踪无缝）。

## 3. 同种子 20 代对照（本地 RTX 5070 Laptop 8GB，从随机初始化）

配置（漏斗比例镜像云端 1/4 口径）：pop=256 → 保 128 → 精英 64，K1=4，K2=8，20 代，
种子 42；B2 用 A=2/B=6（精英 K=12 与基线 K1+K2=12 对齐）。

| 臂 | 代码 | 开关 | 总eval | vs A | 末代Best | 峰Best | 末代Elite | 末代Avg |
|---|---|---|---|---|---|---|---|---|
| A | test16a | 原样 | 791.0s | 1.00× | 6.58 | 7.67 | 4.23 | 1.63 |
| B0 | test16b | 全关 | 699.2s | 1.13× | **6.58** | **7.67** | **4.23** | **1.63** |
| B1 | test16b | 仅 fast | 158.5s | **4.99×** | **6.58** | **7.67** | **4.23** | **1.63** |
| B2 | test16b | fast+halving | 186.7s | **4.24×** | 11.92 | 11.92 | 3.93 | 1.72 |

（对照图：`results/test16b_arms_20gen.png`）

**判据逐条核销**（预注册于计划）：
1. **A ≡ B0**：20 代 × 全部 history 键（含 best_epref/best_conn 遥测）**逐位一致**——
   重构零回归。2 代门禁（pop64）同样全键一致。
2. **B1 ≡ A**：20 代全部适应度键（best_food/fit/avg/elite/seen/unseen/turneff/straight/
   fit_parts_*/k2）逐位一致；仅 best_epref/best_conn 因 edge/conn 采样口径微移
   （如 best_epref 0.342→0.331）——选择零影响。
3. **B2 无系统性回退**：末代 BestFood 11.92 vs 基线 6.58（单种子，运气主导，只主张
   "无损伤"，不主张"更优"）；EliteFood 3.93 vs 4.23、AvgFood 1.72 vs 1.63 在同带内；
   且 B2 更早收敛到低转弯高效形态（Gen 2 起 BestStraight≈0.8、TurnEff=4.0）。
4. **提速**：B1 4.99×（≥4× ✓）；B2 4.24× vs A（≥4× ✓；B2 局数少 19% 但多一轮扫描 +
   其蛇更好→回合更长，小规模下墙钟被行为长度混杂）。
5. **剖面**：基线 obs+遥测 93.4% ≥ 70% ✓；fast 后 obs 75.7%→45.6%（绝对 9×）✓。

### 口径变更记录（B2，云端直改已获同意）
- 阶段2 落选半区指标 = (K1·m1 + A·m2a)/(K1+A)（K=18@云端），其 history 的
  AvgFood/Die 率等均值口径随之微移；精英与 best 追踪 K=36 不变。
- fast 遥测列（teacher/edge/conn/reach）为采样均值；卡55监控的 eshare/conn 趋势保留。
- halving 模式下精英 K 固定 36：若 S>75，σ_ε 将超 δ=1.5（test16a 会自适应加码到
  K2≤128）；如需维持分辨率，提高 `--stage2b-eps`（如 S≈100 时 B=46 → K=64）。

## 4. 云端直连用法（AutoDL）

```bash
# 从 test16a simp 断点 gen210 直接续训（fast+halving 默认开）
python test16b.py --fit-mode simple --resume-pop test16a_simp_checkpoint.pth

# 仅提速不改筛选口径
python test16b.py --fit-mode simple --resume-pop test16a_simp_checkpoint.pth --no-stage2-halving

# 需要剖面时加 --profile-eval（同步计时，仅看占比）
```

**预期**：单代 1320s → ~240-290s（fast ≈4.4-5×，halving 再省 19% 局数；实际受云端卡
单核频率与行为长度影响）。剩余 110 代 ≈ **7-9h**（原 40h）。若 S>75 后需抗噪加码，
叠加 `--stage2b-eps` 上调（成本线性可控）。续训非比特级可复现（断点只存 CPU RNG，
CUDA 流从当前种子重排——test16a 自身续训即如此，非 test16b 引入）。

## 5. 结论与后续

- 瓶颈诊断（调度开销而非局数/FLOPs）被剖面证实：obs 75.7% + 遥测 17.7%。
- fast-eval 在选择零影响（逐位）下提速 ~4.4-5×；halving 在精英预算不动的前提下
  省 18.8% 局数且 20 代无损伤。
- 本地 8GB 卡 eval 批 ~6104 行（EVAL_MEM_FRAC 0.85 自动），更大显存云端卡分块
  更大、扫描更少，收益只增不减。
- 未做（明确排除）：CUDA Graph / torch.compile 深改、FRAME_RATE/STARVE_SLOPE/CRN
  动力学改动、test16a 本体与断点的任何修改。

---

## 6. 320 代终局模型 1000 局基准（2026-09-06，本地 RTX 5070 Laptop）

**代码**：`test16b_benchmark.py`（与 `test7b_benchmark.py` 同款口径；评估路径逐字复用
`_eval_sweep_chunk_fast` + CRN bank，与进化测量同数学；CRN 种子流 gen=987654/stage=9
与训练流（gen 0..319 × stage 1/2）不冲突，跨进程可复现）。运行：
`python test16b_benchmark.py`（两模型各 1000 局共 ~124s；
产物 `results/test16b_bench1k.json`、日志 `logs/test16b_bench1k.log`）。

云端 simp 臂 320 代两个模型（fast+halving 全开，gen210 断点续训）：

| 模型 | 训练期估计(K=36) | 1000局 mean | median | std | P5/P95 | max | 撞墙 | 撞己 | 饿死 | 均存活步 | 直行率 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| simp_best（gen272, 全局最优） | 65.0 | **62.67** | 64 | 9.10 | 55 / 71 | 75 | 36.4% | 62.3% | 1.3% | 1829 | 0.182 |
| simp_latest（gen319, 末代最优） | 60.6 | **56.56** | 59 | 12.33 | 41 / 68 | 75 | 26.3% | 69.8% | 3.9% | 1035 | 0.272 |
| （参考）7b_latest，1000局 | 67.0* | 61.35 | 62 | 7.13 | — / 71 | 79 | 22.6% | 76.8% | 0.6% | — | — |

（*7b 元数据为其训练期 best 追踪值；基准数字取自 `results/test7b_weakmask_bench.json`。）

**判读**：
1. **best 模型实测 62.7 ≈ 训练期 65.0 − 2.3**：K=36 估计的每代 σ_ε≈9.1/√36≈1.5 食，
   回撤 ~1.5σ，属小样本上偏的正常回归；62.7 已**略超 7b 的 61.35**（+1.3 分）。
2. **latest 模型实测 56.6 = 训练期 60.6 − 4.0（~2.6σ）**：末代 best 的 K=36 分数显著
   偏乐观（history 中 elite_food 53.6 与 best_food 60.6 的 7 分落差同样提示）——
   best 追踪的 LCB 折价不足以完全吸掉侥幸高分。**跨模型比较应用 best 而非 latest**。
3. 行为形态：best 更"稳"（存活 1829 步、直行率 0.18、撞墙 36%），latest 更"急"
   （1035 步、直行率 0.27、饿死 3.9%）；7b 以撞己死为主（76.8%），16b 撞墙占比
   明显更高（36% vs 23%）。16b 的 obs40 洪水稀缺通道未把撞墙率压到 7b 以下。

**结论**：16b（稀疏固定扇入 + 4× 柱数 + obs40）在 1000 局实测下达到了与 7b 相当
（略优）的终局性能；且 16b 评估扫描显著更快（1000 局 76s vs 7b 当时 108s）。

## 7. einbrain 可视化升级：兼容 16 系列模型（2026-09-06）

改动（`einbrain/io.py` + `einbrain/vis.py`）：

1. **`io.load_brain_state` 支持 sparse1 基因组**：`rec_idx/rec_w`（槽位制）scatter 成
   稠密 `M_rec/W_rec`（重复源权重叠加，与 gather-乘-归约前向数学等价），脑对象标记
   `sparse_rec/rec_fanin/rec_unique_edges`。新增 `io.load_model_any(path)`：用模型自带
   config 构造 cfg（本包 Config 兜底），任意血统一行加载。
2. **`vis` 向量化与自适应**：社区发现/拓扑图/连接矩阵的 N² 边提取全部 numpy 向量化
   （N=1024 可用，原实现 ~1M 次 Python 循环不可用）；大 N 节点尺寸缩小、输入边按
   |W| 截 top-500（全画 ~6k 条糊成绿墙）、rec 边 top-320。
3. **游玩可视化**：新增 `visualize_model_play(model_path)`——16 系列（sparse1/obs40）
   走 `play_model_16series`（借 test16b 的 BatchedSnakeEnv+deliberate_batch 逐字复刻
   评估语义，支持 `--bank-seed` CRN 复现固定局）；稠密旧血统走 einbrain 标量环境。
   修复：`speed=0` 时不可调 `plt.pause(0)`（事件循环 timeout=0=无限等待，曾挂死）。
4. **CLI**：`python -m einbrain.vis <model.pth> [--play] [--topology] [--matrices]
   [--save-prefix P] [--bank-seed N]`。

**验证**：
- **等价对拍**（`experiments/verify_vis16_equiv.py`）：test16b 稀疏前向 vs einbrain
  稠密化前向，同 obs 流 40 步动作全一致、E 范数对齐到 1e-6（fp32/CPU）。
- 16b：拓扑图/连接矩阵正常出图（`results/brain16b_simp_best_{topology,matrices}.png`），
  play 桥接跑通（bank_seed=7 一局 120 步 food=10）。
- 回归：7b 拓扑/矩阵/play 正常；`einbrain/tests/test_smoke.py` 全过。

### 7.1 实时可视化服务器 tools/brain_visualizer.py 同步兼容 16 系列（2026-09-06）

`tools/brain_visualizer.py`（SSE 实时脑活动面板）同日升级，改动：

1. **引擎注册**：`ENGINES` 头部新增 `Engine("16b", "test16b.py", BRAIN_VERSION=='sparse1')`
   ——必须排在 test12 之前（16 系列也含 OBS_ENC_VERSION，会被 12 的 detect 误吞）。
   默认模型优先级 16b > 12 > 7g > …，下拉框新增"16b 最优 (1024 稀疏)"并扫描
   `test16*_best*.pth`。
2. **稀疏基因组适配**：`BrainGeneAdapter` 对 rec_idx/rec_w 槽位 scatter 稠密化
   （重复源叠加，与 einbrain.io 同数学），布局/边提取/弱连接屏蔽与稠密血统同口径；
   `_run_episode_gene` 走 test16b 自己的 BatchedSnakeEnv + forward_batch（press 疲劳
   形态自动识别），对局动力学与评估逐字一致。
3. **布局向量化**：compute_layout 的三处 N² Python 循环（社区图构建/rec top-K 边/
   3D 每柱 top-10 边）全部 numpy/torch 向量化——N=1024 布局计算 7s（原实现
   ~3×10⁶ 次循环不可用）；init payload ~700KB，SSE 帧 E/I/τ 均为 1024 维。
4. **前端修复与动态化**（`static/visualizer.js` + `index.html`）：
   - 修复 `?model=16b` URL 预选只改下拉框不切后端的缺口（现走与 onchange 相同的
     `/api/init?model=` 路径）；
   - 拓扑面板副标题由静态 "In·24" 改为按 meta 动态（16b 显示
     "In·40 → 柱(1024 稀疏K=16) → Out·3"）；meta 新增 sparse/sparse_fanin 字段。
5. **验证**：引擎检测（16b/12/7b 各归其位）；16b 全链路（init 布局 → SSE 帧
   8 秒 59 帧、score 18 → 浏览器页面截图确认各面板正常、16b 实游玩到 34 分）；
   7b 切换回归正常。

### 7.2 40 维显示修复、obs40 新通道语义探针与单步调试控件（2026-09-06）

**A. 40 维显示修复**（`static/visualizer.js`）：`S.obsMode` 原把非 24 维一律折叠
为 32 模式，导致 16b 的 [32:40) 八通道在三处丢失/错位——
①输入面板只建 32 条（且 [8:16) 用了 32proj 标签，而 16b 该段是 32ego1 语义）；
②拓扑图输入节点 32..39 显示 undefined；③3D 弹簧图输入分类墙按 24 维旧分组、
后 16 个输入坍缩在墙心。修复：新增 OBS40 标签组（[0:32]=32ego1 同构 +
钟压/尾相·前右后左/洪·前左右），obsMode 三态（24/32/40），所有 `=== 32` 判断
改为维度自适应（输入条数、拓扑标签、热图分组白线、分类墙分组、5 射线视线守卫）。

**B. obs40 新通道语义探针**（`experiments/probe_obs40_semantics.py`，受控摆放实证）：
1. **[33:37] 尾相对方位 = 正确的自我系投影，非 bug**：4 朝向 × 8 方位共 32 例
   全部与几何预期一致（对角位两通道各 0.5）。"激活通道随蛇头朝向而变"正是 ego
   语义的定义（如朝东时屏幕"右前"=ego 左前，因朝东的左侧是北）。与用户预期的
   差异：①参照系是自我系而非屏幕绝对系（OBS_FOOD_FRAME='ego' 训练约定，绝对系
   实测无梯度）；②该组**不含距离项**（纯 L1 归一方位占比，尺度不变），与食物
   通道 [12:16] 的距离项不同族。显示侧已把组名标注为"尾相对方位 自我系"。
2. **[37:40] 洪水稀缺 = 相对稀缺度而非绝对可达面积，两个观察均属实**：
   - 空盘基线稀缺 ≈0.02–0.04（设计意图为 0）：蛇头自身格子占位挡洪，而空盘
     参考 cap 允许穿过——与头位置无关的小幅常量偏置（≤4%）。
   - "左右空间差大数值差小"：cap=**同位置空盘** 7 步洪泛可达数（位置归一化，
     test15 明确设计"与头在盘何处无关"）——贴墙侧绝对面积 53 vs 中央 80，
     稀缺度同为 0.036。通道度量的是"自身身体造成的空间损失"（自困风险），
     不是绝对前瞻面积；种子=各向邻格（非蛇头）、深度 7、值域 [0,1] 高=堵死。
   - 影响评估：训练 320 代与部署同语义，无 train/deploy 漂移；若要改为绝对
     可达面积/蛇头源洪泛，属 OBS_ENC_VERSION 破坏性变更需重训，不建议轻动。
   显示侧组名已标注"高=空间少"。

**C. 单步调试控件**（brain_visualizer.py + 前端）：
- **🔁 自动下一局开关**：关闭后局终挂起（状态栏提示"手动模式：点 🔄 开始
  下一局"），🔄/切模型唤醒；默认开（原行为）。
- **⏭ 游戏步进 / ⏩ 网络步进**：许可制闸口——游戏步=一个环境步（K=5 次网络
  更新后走一步）；网络步=一次内部 E-I 更新（棋盘不动，E/I/τ/部分 logits 倾向
  逐帧演化，满 K 次环境走一步）。点步进隐含解除暂停；⏸ 暂停/▶ 继续退出单步
  回连续。修复两个实现坑：①模式切换时旧闸口会吞掉新单位的许可或死等——闸口
  带单位、模式变更即放行、换单位清旧许可；②帧的 steps/score 取自增前的值，
  连续模式不可见但单步模式计数滞后一步——改为先更新计数再构帧。
- **验证**：4 次网络步进步数不变、第 5 次 +1、游戏步每次 +1；手动模式挂起
  6s+ 局数冻结、🔄 后新局；5a 旧血统路径步进许可驱动、可中断、无死锁。
- **追加修复（用户报告"点单步后一直显示已发送"）**：①帧内新增 `paused/stepping`
  状态字段，前端看门狗 3s 无帧时区分「单步等待中：点击 ⏭/⏩ 继续 / 已暂停 /
  后端断开」——此前单步挂起会被误报为"后端可能已退出"；②SSE CLOSED 后自动
  2s 重连（此前后端重启后页面永久失联，须手动刷新）。

### 7.3 帧错位 bug 修复与尾方位通道独立核查（2026-09-06，用户报告驱动）

用户观察：蛇朝南（屏幕下）时尾在"左后"，输入面板却激活"尾相右"。核查结论
（`experiments/probe_tail_ego_check.py`）：

1. **编码语义正确**：改用与代码无关的物理叉积定义（physical_left=(-dc,dr)，
   真实世界校验：朝北左=屏幕左…朝南左=屏幕右）重跑合成用例，4 朝向 × 8 方位
   32/32 全对；对角位两通道各 0.5 也正确。
2. **真实 bug：可视化帧错位**（test7a 时代就有）：`_run_episode_gene/_run_episode`
   中帧的 `obs` 取自 `env.step` 之前，而帧的 `body/food/dir`（游戏盘渲染用）读自
   `env.step` 之后——盘面比观测超前一步。连续播放不可见；蛇转弯的瞬间用户拿
   盘面几何对照"尾相"通道必然对不上（真实对局 92 帧核查 90 帧不符，错位模式
   =每帧实际值恰为下一帧期望值经朝向旋转的平移）。修复：step 前快照
   body/food/dir 放入帧，与 obs 严格同刻；修复后 91/91 全对。
3. **剩余的"左右相反"感是自我系心智旋转**：蛇朝南时蛇的右手边=屏幕左侧
   （想象自己面朝屏幕下方站立）。对照表：朝北左=屏幕左；朝南左=屏幕右；
   朝东左=屏幕上；朝西左=屏幕下。编码遵循"蛇的左右"，与食物方位通道
   （[8:16]，同为 ego）一致——若改成屏幕绝对系反而与训练语义漂移。

## 8. 16b vs 7b 结构指标横向对比（2026-09-06）

**代码**：`experiments/structure_16b_vs_7b.py`（16b best/latest + 7b latest 三模型；
产物 `results/structure_16b_vs_7b.{json,png}`）。口径：rec 图取唯一边（16b 槽位去重后），
权重口径统一 |W|>0.01 为强边（两者强边占比均 ~97.5%，近无差异）；图指标 community 用
greedy_modularity（同算法同权重口径横比）。

| 指标 | 16b_best | 16b_latest | 7b_latest |
|---|---|---|---|
| 柱数 N / obs 维 | 1024 / 40 | 1024 / 40 | 256 / 32 |
| rec 唯一边数 / 密度 | 16264 / **0.0155** | 16267 / 0.0155 | 11289 / **0.173** |
| 平均扇入（入度） | 15.9（K=16 槽，去重亏空仅 0.12） | 15.9 | 44.1 |
| 入度 cv / max | 0.249 / 29（=Poisson(16) 理论值） | 0.246 / 32 | 0.134 / 65 |
| \|W_rec\| mean / p99 / 基尼 | 0.252 / 0.92 / 0.436 | 0.264 / 0.96 / 0.437 | 0.315 / 1.22 / 0.455 |
| E 边占比（W>0） | 50.1% | 50.2% | 50.0% |
| tau_e mean±std | 0.717±0.319 | 0.711±0.345 | 0.720±0.376 |
| w_ei / w_ie mean | 2.02 / 2.00 | 2.00 / 2.02 | 2.10 / 2.02 |
| 输入扇入/柱（M_in 行和） | 6.57（零输入柱 0.2%） | 6.42 | 4.99（0.4%） |
| 读出：每动作柱数 / 柱覆盖率 | 190/183/169 / 44% | — | 51/56/46 / 48% |
| 模块度 Q（greedy） | **0.219**（14 社区，最大 12.7%） | 0.218（13） | **0.113**（6，最大 23.4%） |
| 聚类系数 / 互惠率 | **0.030 / 0.015** | 0.030 / 0.015 | **0.316 / 0.172** |
| 平均最短路 / 小世界 σ | 2.34 / 0.97 | 2.34 / 0.97 | 1.68 / 1.00 |
| 度同配性 / SCC 数 | −0.016 / 1（全连通） | −0.016 / 1 | −0.022 / 1 |

**四个核心结论**：

1. **连接预算同量级，分配方式相反**：16b 并未减少总突触（16.3k ≈ 7b 的 11.3k），
   而是把预算从"少柱高扇入"（256 柱 × 平均 44 输入）改为"多柱低扇入"
   （1024 柱 × 固定 16 输入）。同等连接预算下性能持平（62.7 vs 61.4），说明
   贪吃蛇解对扇入分布不敏感、对总突触量敏感。
2. **16b 的循环拓扑在统计上仍是随机图**：入度 cv=0.249≈Poisson(16) 理论值（1/√16）、
   聚类 0.030、互惠 0.015（几乎无双向边）、小世界 σ=0.97、同配≈0——与稀疏随机初始化
   无统计差异。320 代进化主要雕刻**权重幅度与输入/输出掩码**，未重排稀疏拓扑
   （固定扇入 + 低槽位重连率下拓扑漂移极慢）。其更高的模块度 Q（0.219 vs 0.113）
   是**稀疏度伪影**（稀疏随机图 greedy_modularity 天然切出更多小社区），非进化产物。
3. **7b 是"稠密循环"结构**：聚类 0.316、互惠 0.172——互连丰富的循环子网络
   （平均最短路仅 1.68）。两种基因组先验演化出的宏观图结构完全不同。
4. **单柱层面的收敛高度同构**：E/I 符号各半、|W| 基尼 0.44/0.46、tau 均值 0.72、
   w_ei≈2.0 全部一致——这些是任务解的内在需求，与基因组范式（稀疏/稠密）无关；
   差异只在权重绝对尺度（7b 的 p99 大 ~30%，可能为高扇入下的代偿）。

**展望**：若要让"结构进化"真正发挥作用（而非仅权重进化），16 系需提高 rec_idx
槽位重连变异率，或引入模块化/结构化奖励；当前 K=16 随机拓扑已是足够的"哑容器"。
另：16b 撞墙死占比（36%）显著高于 7b（23%），值得结合 obs40 洪水通道做后续归因。

---

## 9. 扇入扩容：K=16 最优模型 → K=32 种子续训（2026-09-06）

### 9.1 动机与改造语义

§8 的结论（320 代拓扑仍是随机图、进化只雕刻权重与掩码、K=16 槽位是"哑容器"）指向
容量瓶颈假设：**每柱 16 个循环输入槽限制了最高性能**。本轮把历史最优
`test16b_simp_best_model.pth`（gen272，训练期 65.0 / 千局 62.67）改造为 K=32 模型作为
种子续训。改造语义：

- 前缀保留：`rec_idx/rec_w[..., :16]` 原样保留（原回路一针不动）；
- 新增 16 槽：源默认 **fresh 采样**（避开该行已连过的源，唯一边直接扩容，
  对症"狭窄"；`--new-source random` 则与 `random_init` 完全同款），自连禁止同款兜底；
- 新槽权重默认 **零**（einbrain/io.py 同款语义："槽位权重为 0 的连接 M=1 但 W=0，
  W\*M=0 等效断开"）→ **扩容后行为与原模型严格等价**，新槽交给进化接管
  （rec_w 权重噪声 + 槽位重连变异）；`--init noise`（σ=0.05）为"立即有弱信号"变体；
- `config['REC_FANIN']` 如实更新，`--seed-model` / `--resume-pop` / AUTO_RESUME 全链路一致。

前向天然兼容变 K（`forward_batch` 的 K 从张量形状取），改造工具只需做槽位手术——
新工具 **`experiments/expand_fanin.py`**（支持 best-model 与全种群断点双输入，
断点模式同步扩容 `pop` 与 `best_state`，云端 320 代断点扩容后 `--resume-pop`
即全种群 4096 个体无缝续训）。

### 9.2 自检三道门（预注册，工具 `--check` 默认开）

| 门 | 内容 | 实测（best 模型 & mB0 断点 best_state） |
|---|---|---|
| P1 解析 | 稠密化后 W_rec 逐位相等、M_rec 为超集、16384 条新边权重恰为 0 | **全过**（唯一边 16,264→32,648，扇入 13..16→29..32） |
| P2 行为 | 同观测/同 press 驱动 200 步对拍 | **max\|ΔE\|=0.0，argmax 翻转 0**（严格逐位等价） |
| P3 守卫负例 | 原 K=16 模型载入 K=32 run 必须被显式拒绝 | **过**（警告"请先扩容"并忽略种子，run 优雅降级为随机初始化） |

noise 变体走弱校验（旧边权重不被扰动），实测 max|ΔE|=0.077、3 次近平局翻转——
行为有小扰动，属预期。

### 9.3 test16b.py 配套改动（默认行为均不变）

1. `--seed-pop`：种子模型广播至全种群（血统迁移式注入；触发的 SEED_POP 分支原已存在）；
2. `--topo-mut`：覆盖 `TOPOLOGY_MUT_PROB`（默认 0.05 不变）——§8 展望的"提高槽位
   重连率"实验臂入口，扩槽位 + 提重连率才可能真正重排拓扑；
3. `load_best_state` 增 REC_FANIN 显式守卫：K 不符明确报错拒绝（原先静默通过后在
   `set_individual_from_state` 抛难懂的形状错）；
4. **循环连接 + 参数分布实时遥测**（2026-09-06 补，test16b/test16c 同款）：每代双行输出——
   主行含 best 个体 `recNZ`（非零槽占比）/`recEdges`（唯一边数）/`newSlotW`（槽位 ≥
   EXPAND_K_OLD=16 的 \|rec_w\| 质量占比，仅 K>16 显示）；次行 `[params]` 含种群层面
   `newSlotWPop`（新槽权 pop 均值，附 max）/`newNZ`（新槽非零率）与 G2 基因分布
   （tau/w_ei/w_ie 的 pop 均值±标准差）。全部落 history 12 键（rec_nz/rec_edges/
   rec_newmass/rec_newmass_pop/rec_newmass_popmax/rec_newnz_pop/tau_mean/tau_std/
   wei_mean/wei_std/wie_mean/wie_std），旧断点续训自动 None 补齐不错位。
   判定②"接管信号"的 best 层与种群层读数齐全：best 层 0.0% 平台期属预期（best 被
   种子的未变异精英拷贝垄断，recEdges 恒定即字节级未变证据），种群层 newNZ/nP 才是
   噪声注入与选择清零的分水岭。16c 原生 K=64 run 的分区基线：newSlotW(Pop)≈75%、
   newNZ≈100%（随机 init 全非零），偏离基线即选择在雕刻槽位使用。

### 9.4 冒烟（本地 RTX 5070 Laptop，各 1 代，pop16/pop64）

| 冒烟 | 命令要点 | 结果 |
|---|---|---|
| 种子注入（主路径） | `--fanin 32 --seed-model test16b_simp_best_e32.pth --pop 16 --gens 1` | ✅ gen0 BestFood **60.27**（种子个体健在），32768 槽位恰半非零（原16+新16零权） |
| 新旗标 | `--seed-pop --topo-mut 0.15` | ✅ 全种群广播（AvgFood 60.62≈克隆），旗标无报错 |
| 断点扩容（路径A） | 扩容 mB0 断点 → `--resume-pop ... --pop 64 --gens 3` | ✅ next_gen=2→3 无缝接续，K=32 全程无错 |
| 守卫负例 | `--fanin 32 --seed-model`（原 K=16 模型） | ✅ 明确拒绝并降级随机初始化，退出码 0 |

### 9.5 云端长训启动（预注册）

> **2026-09-06 更新**：云端 320 代全种群断点 `test16b_simp_checkpoint.pth` 误删，
> 路径A 暂不可行，**路径B（扩容种子注入）转正为主路径**。`test16b_simp_best_model.pth`
> 与 `test16b_simp_latest_gen_best.pth` 仍在（best/latest 是独立文件，不受断点删除影响）；
> 路径A 待未来出现新的全种群断点（如 e32 长跑自身的断点）后仍可用于"扩容续训"范式。

```bash
# 路径B（现主路径）：随机种群 + 扩容种子注入（本地产物已就绪，上传云端即用）
python test16b.py --fit-mode simple --fanin 32 --seed-model test16b_simp_best_e32.pth --name e32

# 变体：新槽噪声初始化（立即有弱信号，行为有小扰动）
python test16b.py --fit-mode simple --fanin 32 --seed-model test16b_simp_best_e32_noise.pth --name e32n

# 可选臂：种子注入 + 提高槽位重连率（§8 展望）
python test16b.py --fit-mode simple --fanin 32 --seed-model test16b_simp_best_e32.pth \
    --topo-mut 0.15 --name e32r

# 路径A（暂不可行，留档）：全种群断点扩容续训——保留 4096 成熟个体与全部进化历史
# python experiments/expand_fanin.py --src <全种群断点>.pth --fanin 32 --out <扩容断点>.pth
# python test16b.py --fit-mode simple --fanin 32 --resume-pop <扩容断点>.pth --name e32
```

路径B 与路径A 的差异：从"4096 成熟个体 + 历史曲线全保留"变为"1 个 65 分种子 +
4095 随机个体"，血统靠选择从 row 0 扩散（精英 1/4 截断下 gen0 必保留，冒烟实测
gen0 BestFood 60.27 即种子个体）；best 追踪与 history 从 0 重新记录。
路径B 转正后，"扩容种子 vs 从零训 K=32"的对照价值上升——如算力允许建议同步起一个
无种子对照（`--fanin 32 --name c32`）。

**判定**：① 种子承接——e32 臂 gen0/gen1 best_food 应在种子水平（60±2，冒烟实测
60.27/59.12）；② 接管信号——实时日志 `newSlotW` 列（history 键 `rec_newmass`）
> 0 且随代数上升（zero 补槽 gen0 恒为 0.0%，冒烟实测一致）；③ 终局对比——
千局基准对照 simp 320 代的 62.67，c32 对照回答"扩容种子是否优于从零训 32"
（c32 的 newSlotW 天然在 ~50% 附近随机波动，是 e32 的天然基线读数）。

**风险**：test15 两次温启动失败（成熟回路重构成本）不可直接类比——那两次是新增
输入维/新基因，本次观测不变、零权补槽严格等价，不存在破坏成熟回路的通道；真实风险
是**新槽休眠**（重连率低则新槽长期零权），已由 `--topo-mut` 臂覆盖。

### 9.6 中断与接续（K=32 种子 run）

AUTO_RESUME 默认开启：中断后**重跑同一条启动命令**（提高 `--gens`）即从
`test16b_e32_checkpoint.pth` 的 next_gen 接续，种群/history/best/RNG 状态全恢复；
种子注入只在无断点的首次启动分支执行，续跑不会重复注入（`--seed-model` 留在
命令里无害）。两个要点：

- `--name e32 --fit-mode simple` 决定断点文件名，续跑必须一致；
- **`--fanin 32` 必须带上**——2026-09-06 起 `load_checkpoint7` 增设扇入硬守卫：
  断点 K 与当前 run 不符直接 sys.exit 并提示（不可返回 None，否则 AUTO_RESUME
  会当"无断点"全新初始化并覆盖断点；冒烟实测负例报错清晰、断点 md5 未动）。
  遗漏的 `--elite` 等种群参数同理会被既有配置守卫拦下。

断点按 `CHECKPOINT_INTERVAL` 逐代间隔落盘（Ctrl+C 与正常结束也各存一次），
硬杀最多损失最近区间的进度；每代 best/latest 模型独立落盘，不依赖断点间隔。
