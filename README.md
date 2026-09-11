# SiNNtry — EI-RNN（E-I 皮质柱脑区）研究项目

基于进化 / 强化学习训练**兴奋-抑制（E-I）皮质柱脑区**的时序决策模型。该项目经过十余个迭代实验，沉淀出经过考验有效的 EI-RNN 基础架构（`einbrain/` 包）。

## 目录结构

```
SiNNtry/
├── snake_std.py      ★ 标准实现程序（根目录唯一训练入口，以 16b 为基座：三种观测环境 ×
│                       多适应度公式 × 多筛选方案 × 激素/轮换等系统开关，
│                       详见 docs/standard_implementation_log.md）
├── test16b_simp_best_model.pth   ★ 当前基准模型（千局 62.67，略超 7b 61.35）
├── einbrain/         经过考验有效的 EI-RNN 整合通用包
│   ├── config.py       统一配置（脑/环境/进化/PPO/GPU/NEAT）
│   ├── env.py          贪吃蛇环境（24 维射线观测） + RaySnakeEnv（32 维 8 方向观测） + ProjSnakeEnv（test7b 投影观测复刻）
│   ├── brain.py        统一 EIBrainRegion 脑模型
│   ├── dynamics.py     共享 E-I 核心数学
│   ├── evolve.py       CPU 进化训练（test5d 语义）
│   ├── neat.py         真正 NEAT 引擎（创新号+物种化+历史标记交叉，test8 专用）
│   ├── ppo.py          PPO 强化学习（test6 语义）
│   ├── gpu.py          GPU 全并行进化（test7 语义）+ BatchedRaySnakeEnv
│   ├── io.py           统一保存/加载，兼容三格式模型互作种子
│   ├── deliberation.py K 帧思考
│   └── vis.py          可视化（16 系列模块自动定位 experiments/test16_series/）
├── artifacts/        ★ 实验产物归档（按世代分目录）：test7b/ test12/ test14/ test15/
│                     test16/ test16a/ test16b/ test16c/ test16c_cheat7b/
│                     存放各世代 *_best_model.pth / *_history.json|png 等
├── experiments/      ★ 历史实验源码归档（按谱系主题子目录，脚本按仓库根相对路径运行）：
│   ├── early/           test1..test6 全家族（含 smoke/lunar 变体）+ 早期诊断脚本
│   ├── test7_series/    test7→7h 主线 + test7b 冠军线 + benchmark/评测 + 外围 A/B 分析
│   ├── test8_neat/      真正 NEAT × EI-RNN
│   ├── test10_lunar/    LunarLander 移植（含云端部署包）
│   ├── test11/          锦标赛 vs 两阶段筛选
│   ├── test12/          ego 观测 + 适应度 v7 主线 + 习惯/孤岛诊断
│   ├── test13_ppo/      冠军剪枝 + PPO 微调（已结题）
│   ├── test14/          期相激素 v1 + 激素 A/B 判定
│   ├── test15/          观测增维 32→40 + obs40 标定/判定
│   ├── test16_series/   test16→16a→16b→16c→16c_cheat7b + 基准/扩容/迁移/对拍
│   └── run_seeded.py    旧脚本种子注入运行器
├── tools/            实时脑活动可视化服务器（brain_visualizer + static，模型扫描覆盖根目录与 artifacts/）
│   └── video/           视频制作（make_intro_video / render_win_video / tune_topology）
├── media/            成品视频（不入库）
├── deploy_test12/    test12 的 AutoDL 云端部署包
├── deploy_test15/    test15 观测增维的 AutoDL 云端部署包（cold1 断点续训）
├── bench/             性能基准（列数扫描 / A-B 对比 / 加速比）
├── models/            训练好的模型权重（test4b/5/5a/5d/6/7/8 等，含早期 LSTM 预训练）
├── results/           训练曲线/基准/诊断等数据与图（含 test9/ 子目录）
├── logs/              运行日志
└── docs/              实验演进分析与各主题实验日志
```

## 实验日志索引（docs/）

| 日志 | 主题 | 状态 |
|---|---|---|
| `docs/test_evolution_analysis.md` | test1–8 全演进分析 + §4 后续演进索引 | 持续维护 |
| `docs/test7_series_experiment_log.md` | GPU 进化主线 test7→7h + 外围 A/B 实验群 | 已归档（7b 冠军线活跃） |
| `docs/test9_experiment_log.md` | 链接影响/剪枝分析 | 已结题 |
| `docs/test10_lunar_experiment_log.md` | LunarLander 二度移植 | 云端长跑，结果未归档 |
| `docs/test11_experiment_log.md` | 锦标赛筛选 vs 两阶段 | 已结题（保留两阶段） |
| `docs/test12_experiment_log.md` | ego 观测 + 适应度 v7（当前主线） | 已结题（§7 归因终稿：后期模式缺失，转 test14） |
| `docs/test13_ppo_experiment_log.md` | 冠军脑剪枝固化 + PPO 微调 | 已结题（判定不可行） |
| `docs/test14_experiment_log.md` | 期相激素 v1：局内动态调制 | 已结题（G1-NULL：精英通胀非能力增益） |
| `docs/test15_experiment_log.md` | 观测增维 32→40（信息直供 + Phase A 标定 + 温启动 A/B） | 活跃 |
| `docs/test16b_experiment_log.md` | 评估提速 fast-eval + 阶段2 对半精评；§6 320 代终局 1000 局基准（best 62.7 略超 7b 61.4）、§7 einbrain 16 系列可视化、§8 16b vs 7b 结构对比（同连接预算/随机拓扑 vs 稠密循环）、§9 扇入扩容 K=16→32 种子续训（`experiments/expand_fanin.py`，零权补槽严格等价 + 三道自检门） | 活跃 |
| `docs/standard_implementation_log.md` | **snake_std 标准实现程序**：三观测环境 × 多适应度 × 多筛选 × 激素/轮换/池/固定地图开关；14 组自检 + 等价门（默认=16b 逐位） | 活跃（标准入口） |

## 快速上手（snake_std 标准实现，推荐入口）

以 test16b 为基座的单文件标准程序，四种能力面全由 CLI 选择（完整矩阵见
`docs/standard_implementation_log.md`）：

```bash
# 默认 = 16b 行为（obs40 + econ v7/LCB + 二阶段+对半精评 + fast-eval）
python snake_std.py --gens 320

# 三种观测环境：40tailflood1（默认）/ 32ego1（test12）/ 32proj7b（test7b）
python snake_std.py --obs 32ego
python snake_std.py --obs 32proj

# 适应度公式：econ 乘法式（默认）/ simple 最简 / tuple 词典序 / 鲁棒最小值
python snake_std.py --fit-mode simple
python snake_std.py --robust-eval 2

# 筛选方案：二阶段+对半精评（默认）/ 单阶段 / 不精评 / 自适应 K2
python snake_std.py --no-stage2

# 系统开关：激素系统 / 训练轮换 / 感觉-运动池 / 固定地图
python snake_std.py --train-hormone --cycle-pattern "G1,G2,G3"
python snake_std.py --pools
python snake_std.py --fixed-map

# 断点续训（16 系列断点互认）与播放
python snake_std.py --resume-pop <16系列checkpoint.pth>
python snake_std.py --play
```

自检（14 组）一键冒烟：`run_snake_std_smoke.bat`。

## 快速上手（einbrain 包）

```python
import einbrain
from einbrain import Config

cfg = Config()

# 1) CPU 进化（test5d 语义）
best_brain, history = einbrain.run_evolution(cfg)

# 2) PPO 强化学习（test6 语义，可加载 test5d 模型作种子）
best_brain, history = einbrain.run_training(cfg)

# 3) GPU 全并行进化（test7 语义）
from einbrain.gpu import run_training_gpu
best_state, history = run_training_gpu(cfg)

# 加载已有最优模型并游玩
from einbrain import io, SnakeEnv, deliberate_action
brain, food, steps = io.load_best_model_brain('test5d_best_model.pth', cfg)

# 任意血统一行加载（含 test16 系列稀疏基因组，自动稠密化等价展开）
from einbrain import io
brain, cfg, meta = io.load_model_any('test16b_simp_best_model.pth')
```

模型可视化 CLI（兼容 test5d/6/7 稠密与 test16 系列稀疏基因组）：

```bash
python -m einbrain.vis test16b_simp_best_model.pth --play --bank-seed 7   # 游玩（16 系列走 test16b 评估语义）
python -m einbrain.vis test16b_simp_best_model.pth --topology --matrices --save-prefix results/brain16
```

三种训练模式的详细对比与设计动机见 `docs/test_evolution_analysis.md`。

## 环境

- Python 3.13，conda 环境 `env_torch`（`env_torch`）
- 依赖：`torch` / `numpy` / `matplotlib` / `networkx`；PPO 用 `gymnasium`（实验脚本 test3/test5a_lunar）；可视化可选 `python-louvain`
- GPU 模式（test7）依赖 CUDA（自动检测，可用 `cfg.DEVICE='cpu'` 回退）

## 运行历史实验

历史脚本按谱系归档于 `experiments/` 主题子目录（模型/断点/产物路径按仓库根
相对解析，**从仓库根目录运行**）：

```bash
python experiments/early/test5d.py                  # 最优 CPU 进化版
python experiments/early/test6.py                   # PPO
python experiments/test7_series/test7h.py --smoke   # GPU CRN 筛选版自检
python experiments/test7_series/test7b_benchmark.py # 7b 冠军千局基准
python experiments/test8_neat/test8.py --smoke      # 真正 NEAT × EI-RNN 自检
python experiments/test12/test12.py --selfcheck     # ego 观测主线自检
python experiments/test16_series/test16b.py --smoke # 16b（snake_std 的基座）自检
```

## test8 — 真正 NEAT 与 EI-RNN 的契合度验证

`experiments/test8_neat/test8.py` 用**完整的 NEAT**（创新号 + 物种化/相容性距离 + 历史标记交叉 + add-connection/add-node 结构生长）驱动 EI-RNN 进化，验证二者契合度：

- **环境**：K=5（`FRAME_RATE=5`）贪吃蛇，新 **32 维头朝向相对 8 方向观测**（`RaySnakeEnv` / `BatchedRaySnakeEnv`）：蛇首方向 4 one-hot + 蛇尾方向（末节移动朝向）4 one-hot + 食物 8 扇区 + 自身 8 扇区 + 8 方向障碍距离倒数（身后槽约定为 `sqrt(蛇身长度/格子度)`，避免冗余不变量）。
- **启动**：256 柱 `INIT_DENSITY=0.15` 稀疏随机拓扑，为每条连接分配创新号；此后 add-connection / add-node 单调生长。
- **契合度指标**：每代物种数（自适应阈值收敛于 `SPECIES_TARGET`）、平均激活连接/柱数（结构生长）、BestFood 曲线。
- **性能**：POP=1024/N=256/K=5 下每代约 10–45s（`evolve` 父本继承 + 每 `RE_SPECIATE_INTERVAL` 代全量再物种化、`SPECIES_CAP` 兜底；`obs()` 全向量化；`decode_population` 批量 scatter；断点紧凑数组）。
- **运行**：`python test8.py --device cuda --pop 2048 --gens 100`；断点续训 / 最优模型保存 / `--play` 播放与 `einbrain.io` 完全兼容。

> 注意：各世代产物统一归档于 `artifacts/<世代>/`（基准模型 test16b_simp_best_model.pth 留守根目录）；历史脚本按仓库根相对路径解析，从仓库根目录运行即可。
