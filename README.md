# SiNNtry — 用神经进化训练「E-I 皮质柱脑区」玩贪吃蛇

一个把**兴奋-抑制（E-I）皮质柱脑区**当作决策器官、用 GPU 全并行神经进化在贪吃蛇上
迭代了 16 代实验的研究项目。最终沉淀为两部分：

- **`snake_std.py`** —— 标准实现程序（单文件训练入口）：以 test16 系列为基座，
  把三套观测环境、多套适应度公式、多套筛选方案与激素/轮换等系统开关全部 CLI 化；
- **`einbrain/`** —— 经过考验有效的通用库：统一配置、环境、脑模型、E-I 动力学、
  进化 / PPO / NEAT 三条训练范式、IO 与可视化。

> **English abstract.** SiNNtry trains an excitatory–inhibitory (E-I) cortical-column
> recurrent network to play Snake via fully GPU-parallel neuroevolution. The flagship
> single-file trainer `snake_std.py` exposes three observation encodings (32-dim ego,
> 32-dim 8-sector projection, 40-dim enriched), several fitness formulas (simple
> efficiency — the recommended default, multiplicative econ, tuple, robust-min),
> selection schemes (single-stage, two-stage, stage-2 halving, adaptive-K2 with LCB
> key), and system toggles (hormone modulation, group-freeze rotation, sensory/motor
> pool constraints, fixed-map) behind CLI flags. A verified 1000-game model
> (`test16b_simp_best_model.pth`, 62.7 mean food, vs 61.4 for the previous champion)
> ships in the repo. Research logs live in `docs/` (Chinese).

---

## 目录

- [设计想法](#设计想法)
- [重要结论](#重要结论)
- [安装](#安装)
- [快速开始](#快速开始)
- [标准实现全开关矩阵](#标准实现全开关矩阵snake_stdpy)
- [断点续训与可复现](#断点续训与可复现)
- [使用 einbrain 库](#使用-einbrain-库)
- [预训练模型](#预训练模型)
- [仓库结构](#仓库结构)
- [研究文档](#研究文档)
- [环境与已知边界](#环境与已知边界)
- [第三方与致谢](#第三方与致谢)

---

## 设计想法

### 1. 脑模型：E-I 皮质柱 + K 倍帧率思考

决策器官不是策略网络，而是一个**皮层柱风格的循环动力系统**：

- 每柱一对状态变量 `E`（兴奋）、`I`（抑制），离散代数更新
  `E = σ(total_in + τ_e·E − w_ei·I)`、`I = σ(w_ie·E)`；动作 logits 由读出矩阵从 `E` 读出。
- 环境一步 = **K=5 次内部迭代**（输入按 0.9ᵏ 衰减，logits 求和后 argmax）——给低频
  决策一个高频"思考"窗口，`E/I` 状态跨步持续，形成真正的时序记忆。
- 种群不是参数向量列表，而是**堆叠成批张量**（`GeneStack`）：整代 4096 个个体在
  GPU 上一次前向、一次交叉变异，进化循环里没有逐个体 Python 开销。

### 2. 基因组：稀疏固定扇入

把 N×N 稠密循环权重换成**每柱 K 个槽位**（`rec_idx [N,K]` + `rec_w [N,K]`）：结构本身
参与进化（槽位重连变异），前向即 gather-乘-归约；自检包含稀疏前向与稠密参考的逐位
等价验证。**推荐 N=256 / K=96**（无损覆盖冠军网络逐行非零连接数 max=65；降到 64
亦无明显损失）。

### 3. 评估：CRN 公共随机数

比较两个个体的适应度时，**食物序列本身不该是噪声源**。每代每阶段预生成公共落子流
（per-generation banks），同库个体面对完全相同的地图序列——同库比较无食物运气差异，
跨代用种子序列可复现。这是后续一切"千局基准"可信的前提。

### 4. 选择：两阶段淘汰 + 对半精评 + LCB

- **两阶段筛选**：全种群 ×K1 局（同库精确可比）→ 保前 STAGE2_KEEP → 幸存者 ×K2 局
  精评。单阶段全局排序的 rank-1 复评漂移实测 28.7↔39.6，两阶段把它消掉。
- **对半精评（halving）**：阶段 2 幸存者先全体 ×A 局，只有前一半（有精英资格的）
  再打 B 局——注定落选的半区每体省 18 局，精英评估预算与原方案完全对齐
  （K=K1+A+B 不变）。
- **LCB 选择键**：不同局数下测得的适应度不可直接比。选择键 = base − λ·CV·max(food,1)/√K，
  把"以更低 K 测出的高分"按方差折价；配合**自适应 K2**（σ_ε≤δ 分辨率判据动态加密
  评估），解决"高分是评估噪声"的假冠军问题。

### 5. 适应度：少整形，多杠杆

- **`simple`（推荐，默认）**：`food + k·food/steps_last`——效率最简口径。
  **全部最优模型（7b 冠军 61.4、16b_simp 62.7、cheat7b 通关）均出自此口径。**
- `econ`（v7 乘法式，对照保留）：`food × (1 + W_EFF·eff + W_STRAIGHT·straight +
  W_EDGE·edge) × conn`——习惯因子随食物等比放大的设计很优雅，但**其相对 simple 的
  额外增益未获证据支持**，仅保留作对照口径。
- `tuple`：早期词典序口径，保留作对照。
- **鲁棒最小值评估**：每个体以原权重 + σ=1e-3 权重噪声共 R 份副本评估，取**最差副本**
  ——过滤"混沌彩票解"（对特定权重实现过拟合的个体）。
- 12 代实验换来的方法论结论：**行为改造的杠杆优先在观测/环境端，适应度整形是弱杠杆**
  （外围 A/B 实验群全部 NULL，见 docs/test7 系日志）。

### 6. 观测工程：32 维为主，40 维作迁移性扩展

- `32ego1`（推荐，默认）：食物以**自我中心系**给方位+距离（绝对系被实测证明选择
  无梯度），加自身/障碍 8 扇区；
- `32proj7b`：8 扇区欧氏投影（食物永远可见，经典口径）；
- `40tailflood1`：在 32ego 之上追加**饥饿钟压力、尾四方位、前/左/右 7 步有限洪水
  稀缺度**。实测结论：**增维到 40 没有明显突破与优化，只是一定程度增加了跨盘面
  可迁移性**——新增通道真实承重（冷启动消融：关掉 23.08 食 → 开着 0.42 食），
  但不足以改变格局。

### 7. 系统开关：默认全部关闭（未证明有益，保留接口）

- **激素系统**（`--train-hormone`）：前馈网读 `[E;I;total]` → 每类激素**同帧单柱
  门控释放**（严格超过阈值才释放，零初始化=无操作）→ 沿循环拓扑扩散衰减 → 调制
  τ_e。**默认关闭**：判定为 G1-NULL（"收益"是精英统计通胀而非能力增益，见 test14 日志）。
- **训练轮换**（`--cycle-pattern`）：参数分 G1（结构）/G2（动力学）/G3（激素）组，
  可逐代指定激活组、冻结组单亲遗传不变异。**默认两组全开（无冻结）**——冻结交替
  未证明有益；test5a 式交替需显式 `"G2;G1"`。
- 其余开关（感觉-运动池约束、固定地图、疲劳、弱连接屏蔽等）同样默认关闭，作为
  已验证机制的可选接口。

### 8. fast-eval：把瓶颈从数学搬到调度

逐环境步 ~1200 次 Python 级 torch 调用是真正的瓶颈。fast-eval 把 16 扇区射线改为
偏移表一次 gather + cumsum 首命中（~960 → ~40 op/步）、遥测降频不减列、消除每步强制
同步——**~5× 提速且适应度路径逐位不变**（自检对拍保证"提速不改选择"）。

---

## 重要结论

以下三条是全部实验收敛后最值得带走的实证发现：

### 1. 标杆成绩与外部参考模型对比（千局随机地图口径）

| 玩家 | 观测/网络 | 成绩（千局随机盘） | 备注 |
|---|---|---|---|
| **test7b 冠军** | 32proj，稠密 N=256 | mean 61.35 / median 62（训练期 67.0） | 效率适应度首冠军 |
| **test7b（展示素材评测）** | 同上 | mean 61.1 / median 62 / **单盘最高 80** | B=1000 并发评测，最佳盘渲染成 `7b_best_random.mp4` |
| **test16b_simp（旗舰）** | 40 维，稀疏 K=16, N=1024 | mean 62.67 / median 64 / max 75 | 现基准 |
| Ackeraa nn_97（外部参考） | 前馈 [32,12,8,4] | mean 27.3 / median 8 / max 78 | 40 局 CRN 同库、免饿死钟口径；双峰：要么早死要么大赢 |
| chuyangliu/snake（外部参考） | 图搜索（BFS+哈密顿） | mean 48.9 / max 98（整版通关） | 40 局 CRN、含饿死钟；作"物理可达上限"参考解法器 |

解读：进化得到的 E-I 循环网络显著强于同量级前馈学习网络（Ackeraa），但距离图搜索
的"理想解法器"仍有本质差距——**瓶颈在观测/表征，不在搜索**：千局死亡以撞墙 36% /
自撞 62% 为主，且后期（蛇长 40+）的坍缩 91–98% 源于空间挤压自撞（test12 期相归因），
即网络"看不见"死区，而搜索式解法器看得见。

### 2. 模型近似前向模型：历史记忆的影响极小

`experiments/test16_series/exp_7b_memory_probe_random.py`（对 cheat7b 通关模型的
记忆探针，全随机变体）：把 15 个状态输入槽位以**有放回随机重抽**构造 16 步序列，
测量动作一致率与 logits 的首末步差异、跨序列末步两两距离。

**结论：模型近似一个前向（反应式）模型——输出极大程度上由当前输入决定，极小受
历史影响。** K=5 帧思考与 E/I 循环状态确实携带时序信息，但对动作决策的贡献与
"只看当前帧"几乎无法区分。这对"循环网络学到了长时规划"的直觉是一个重要的反例
（也解释了为什么观测端的信息直供如此有效）。

### 3. 网络稳定性对结构扰动高度不敏感

三类扰动实验一致表明**性能不依赖个别精确权重，容量冗余很大**：

- **稀疏固定扇入**：同突触预算下，稀疏 K=16/N=1024（循环密度 1.6%）与稠密 N=256
  相当且略优（62.7 vs 61.4）——97.4% 的循环边为强边，结构本身不是瓶颈；
- **弱权重删除**：删最弱 0–30% 循环连接 ×1000 局基准，均值 61.35 → 61.05/60.67/61.66
  （差 <0.7，组内 std≈7–8）；10–40% 弱连接屏蔽波动 ±1.8 食——剪枝不改变分数，
  只重排死因结构；
- **权重噪声**：σ=1e-3 微扰副本与原副本成绩差异在噪声量级内（鲁棒最小值口径的
  作用仅是过滤极少数"混沌彩票解"，而非普遍提升稳定性）。

---

## 安装

```bash
git clone <repo-url> && cd SiNNtry
conda create -n snake python=3.13 -y && conda activate snake

# GPU（推荐，任意 ≥8GB 显存 CUDA 卡；CPU 亦可跑通但慢）
pip install torch --index-url https://download.pytorch.org/whl/cu130
# 或 CPU 版：pip install torch

pip install numpy matplotlib networkx
pip install python-louvain        # 可选：可视化社区检测
```

验证环境：

```bash
python snake_std.py --selfcheck     # 14 组自检（适应度公式/CRN 确定性/稀疏等价/
                                    # 观测等价/激素等价/鲁棒机制/池约束/固定地图）
python snake_std.py --smoke         # 3 代端到端冒烟
```

Windows 可直接双击 `run_snake_std_smoke.bat`（自动定位 conda 环境并跑自检+冒烟）。

---

## 快速开始

### 训练（默认 = 推荐配置）

```bash
python snake_std.py --gens 320
# 产出：snake_std_checkpoint.pth（原子断点）、snake_std_best_model.pth、
#       snake_std_latest_gen_best.pth、snake_std_history.json/png
```

默认配置：**32ego 观测、simple 适应度、N=256 / K=96**、pop 4096 / 精英 1024、
K1=12 → 保 2048 → 对半精评（A=6/B=18）、LCB 选择键、CRN 开、fast-eval 开。
激素/轮换/池约束等全部关闭（未证明有益，见[重要结论](#重要结论)与开关矩阵）。

### 播放与可视化

```bash
# 回放本仓库自带的旗舰模型（obs40/N1024 口径，需显式对齐参数）
python -m einbrain.vis test16b_simp_best_model.pth --play --bank-seed 7
python -m einbrain.vis test16b_simp_best_model.pth --topology --matrices --save-prefix results/brain

# 回放自己训练出的最优模型（默认口径）
python snake_std.py --play
```

Web 实时脑活动可视化（拓扑/柱状/热图/3D + 单步调试）：

```bash
python tools/brain_visualizer.py          # 启动后浏览器打开本机服务地址
```

模型下拉框自动扫描根目录与 `artifacts/*/`；支持稠密（test4b–7）与稀疏（16 系列）
全部血统，逐神经元活动实时渲染。

### 千局基准

```bash
python experiments/test16_series/test16b_benchmark.py        # 16b 基准模型 vs 7b 参照
python experiments/test7_series/test7b_benchmark.py          # 经典 7b 冠军
```

---

## 标准实现全开关矩阵（snake_std.py）

### 观测环境 `--obs`

| 取值 | 编码版本 | 说明 |
|---|---|---|
| `32ego`（**默认**） | 32ego1 | test12 ego 编码（食物方位+距离倒数，自我中心系） |
| `32proj` | 32proj7b | test7b 8 扇区欧氏投影（食物永远可见） |
| `40` | 40tailflood1 | 32ego1 + 饥饿钟/尾方位/洪水稀缺 8 通道——无明显突破，仅小幅提升可迁移性（`--no-new-obs` 置零作 A0 对照） |
| `24` | 24ray1 | 早期射线观测（休眠，兼容历史模型） |

### 适应度 `--fit-mode` 及修饰

| 取值 | 公式 |
|---|---|
| **`simple`（默认，推荐）** | v9：`food + k·food/steps_last`（`--simple-eff-w`）——**全部最优模型出自此口径** |
| `econ` | v7 乘法式：`food×(1+W_EFF·eff+W_STRAIGHT·straight+W_EDGE·edge)×conn`——乘法设计的额外增益未获证据支持，对照保留 |
| `tuple` | test7 词典序 `(food, unseen, −seen)` |
| `--robust-eval R` | v20 鲁棒最小值：R 份副本（权重噪声 σ=`--robust-sigma`）取最差 |
| `--te-elite N` | te=步数/转弯数 配额外精英配额 |
| `--sel-lcb λ --sel-cv cv` | LCB 选择键（0=关） |

### 筛选方案

| 方案 | 开关 |
|---|---|
| 单阶段（全种群 ×K1 全局排序） | `--no-stage2` |
| 二阶段（K1 → 保 STAGE2_KEEP → K2） | 默认；`--stage1-eps/--stage2-eps/--stage2-keep` |
| 二阶段+对半精评 | 默认开；`--no-stage2-halving` 关闭，`--stage2a-eps/--stage2b-eps` 可调 |
| 自适应 K2 | 默认开；`--no-k2-adapt/--res-target/--k2-max/--k2-floor-dynamic` |

### 系统开关（默认全部关闭）

| 开关 | 说明 |
|---|---|
| `--train-hormone` | 激素系统：单柱门控释放 + 沿稀疏拓扑扩散调制 τ_e。默认关：判定 G1-NULL（精英通胀非能力增益） |
| `--cycle-pattern` | 训练轮换：`"G2,G1"`=两组每代全开（同默认）；`"G2;G1"`=真交替冻结（未证明有益） |
| `--pools` | 感觉-运动池约束：输入/输出列限制在池内（`--sensory-frac/--motor-frac`），强制隐藏层结构 |
| `--fixed-map` | 固定单张地图训练（`--map-seed`），通关特训口径 |
| `--turn-gain / --turn-decay` | 转向疲劳（默认关） |
| `--no-crn / --crn-seed` | 公共随机数开关与种子 |
| `--weak-mask-frac` | 评估期弱连接屏蔽 |
| `--no-fast-eval` | 关闭 fast-eval（回退逐字对拍路径） |
| `--no-one-sided-death` | 关闭单侧转弯判死 |
| `--columns / --fanin` | 柱数 N（默认 256）/ 扇入 K（默认 96，≥64 无明显损失） |
| `--pop / --elite / --gens` | 种群 / 精英（亲本）数 / 代数 |
| `--seed-model [--seed-pop]` | 种子模型注入（单行/全种群克隆） |
| `--name` | 输出文件前缀（多臂实验隔离） |

---

## 断点续训与可复现

- **原子断点**：`snake_std_checkpoint.pth` 含全种群基因、RNG 状态、历史与 best 追踪，
  Ctrl+C 安全；`--gens` 提高后重跑即续训。
- **跨断点导入**：`--resume-pop <16系列checkpoint.pth>` 可导入任意 16 系列全种群断点
  （版本守卫：N/OBS 编码/BRAIN_VERSION/扇入 K/激素开关逐项校验，拒绝静默错配）。
- **同种子对照**：`--seed 42` 固定 CPU+CUDA RNG。
- **FITNESS_VERSION 口径**：simple=9（默认）/ econ=8（`--fit-mode econ`）；
  开启鲁棒评估=21、激素=22（口径变化，导入旧断点时 best 追踪按规则重置）。

---

## 使用 einbrain 库

`einbrain/` 是从 16 代实验中蒸馏出的通用包，三条已验证训练范式共享同一 E-I 核心：

```python
import einbrain
from einbrain import Config

cfg = Config()                       # 统一配置（脑/环境/进化/PPO/GPU/NEAT）

best, history = einbrain.run_evolution(cfg)          # CPU 进化（test5d 语义）
best, history = einbrain.run_training(cfg)           # PPO（test6 语义）
from einbrain.gpu import run_training_gpu
best, history = run_training_gpu(cfg)                # GPU 全并行进化（test7 语义）

# 任意血统一行加载（稠密 test4b–7 与稀疏 16 系列自动识别）
brain, cfg, meta = einbrain.io.load_model_any('test16b_simp_best_model.pth')
```

另有完整 NEAT 引擎（创新号+物种化+历史标记交叉，`einbrain.neat`）驱动同一脑模型
（见 `experiments/test8_neat/`）。

---

## 预训练模型

| 模型 | 位置 | 口径 |
|---|---|---|
| **test16b_simp**（旗舰） | `test16b_simp_best_model.pth`（根目录） | 千局均值 **62.7**；N=1024 稀疏 K=16，obs40 |
| test7b 冠军 | `artifacts/test7b/test7b_best_model.pth` | 训练期 67.0 / 千局 61.4；N=256 稠密，obs32 投影 |
| cheat7b 通关模型 | `artifacts/test16c_cheat7b/` | 7b 迁移稀疏 K=96 + 固定地图整版通关 |
| 16b 变体（mB0/mB1/e32 扇入扩容） | `artifacts/test16b/` | 见 docs/test16b 日志 |
| test12/14/15 各世代 best | `artifacts/test12…15/` | 32ego → 40tailflood 演化链 |
| 早期（test4b–test8） | `models/` | LSTM 预训练柱 / 24 维时代 |

---

## 仓库结构

```
SiNNtry/
├── snake_std.py                  ★ 标准实现（唯一训练入口）
├── test16b_simp_best_model.pth   ★ 基准模型
├── einbrain/                     通用库（config/env/brain/dynamics/evolve/neat/ppo/gpu/io/vis）
├── experiments/                  历史实验源码（按谱系归档）
│   ├── early/                      test1–6 家族
│   ├── test7_series/               test7→7h + 7b 冠军线 + 外围 A/B 分析
│   ├── test8_neat/                 NEAT × EI-RNN
│   ├── test10_lunar/ test11/ test12/ test13_ppo/ test14/ test15/
│   ├── test16_series/              16 系列全家 + 基准/扩容/迁移/对拍
│   └── run_seeded.py               旧脚本种子注入运行器
├── artifacts/                    各世代模型/曲线产物（按世代分目录）
├── tools/                        brain_visualizer 实时可视化 + video/ 视频制作
├── bench/                        性能基准脚本
├── models/                       早期模型权重
├── results/                      基准 json / 曲线 / 诊断图
└── docs/                         实验日志与研究文档（中文）
```

历史脚本统一**从仓库根目录运行**，例如：

```bash
python experiments/test7_series/test7b_benchmark.py
python experiments/test12/test12.py --selfcheck
python experiments/test16_series/test16c_cheat7b.py --smoke
```

---

## 研究文档

全部为中文，按主题成稿（含每步的预注册判定门与 NULL 结果）：

| 文档 | 内容 |
|---|---|
| `docs/test_evolution_analysis.md` | test1–8 全演进分析（为什么是 E-I 柱 + 三范式） |
| `docs/test7_series_experiment_log.md` | GPU 主线 test7→7h；CRN/两阶段/类正态变异的来源 |
| `docs/test11_experiment_log.md` | 锦标赛 vs 两阶段筛选（两阶段胜出） |
| `docs/test12_experiment_log.md` | ego 观测重写 + 适应度 v3→v7；后期瓶颈归因 |
| `docs/test13_ppo_experiment_log.md` | 冠军剪枝 + PPO 微调为何不可行；弱剪枝不敏感实验 |
| `docs/test14_experiment_log.md` | 激素 v1 与"精英通胀"判定方法学 |
| `docs/test15_experiment_log.md` | 观测 32→40 增维：标定、A/B、冷启动消融 |
| `docs/test16b_experiment_log.md` | 稀疏基因组、fast-eval、对半精评、结构对比、扇入扩容 |
| `docs/standard_implementation_log.md` | **snake_std 标准实现**：开关矩阵 + 14 组自检 + 等价门 |

---

## 环境与已知边界

- **开发环境**：Python 3.13 / torch 2.9（CUDA 13）/ numpy 2.2；Windows 与 Linux 均可。
  更低 Python 版本未测。
- **显存**：默认 pop4096/N256 在 8GB 卡上余量充足（评估分块自动按显存折算，OOM 自动
  减半重试）；obs40/N1024 口径（旗舰模型血统）同样 8GB 可用；激素开启约 +1.1GB。
- **可复现性**：训练受 GPU 非确定性影响，`--seed` 保证同机同库确定；跨机型逐位一致
  不做承诺，千局统计口径可复现（`results/*.json`）。
- **32ego 历史模型**：test12/14/15 的 32ego 权重是稠密基因组，不能注入 16 系列稀疏
  种群（按设计拒绝）；32ego 观测用于训练新的稀疏世代。
- `test16c_cheat7b` 的 7b 稠密→稀疏迁移器未并入标准实现，复现请用原脚本。
- N=1024 柱数方向：同连接预算下未见优化，且因算力条件未完成完整实验——结论以
  N=256 为准。

---

## 第三方与致谢

- [Ackeraa/snake](https://github.com/Ackeraa/snake)：外部贪吃蛇 AI（`third_party/ackeraa/`，
  前馈网络 nn_97），用作同环境对拍的参考模型，版权归原作者所有；
- [chuyangliu/snake](https://github.com/chuyangliu/snake)：GitHub 上的贪吃蛇通关脚本
  （图搜索），移植为参考解法器（`experiments/test7_series/ref_solver.py` 的
  GraphAgent/CycleSolver），用于适应度权重标定与"物理可达上限"参照；
- 项目全部研究日志、判定方法（预注册 A/B 门、跨库配对消融、等价门）与 NULL 结果
  一并开源——负结果也是结果。

## License

[MIT](LICENSE)
