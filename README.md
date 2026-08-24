# SiNNtry — EI-RNN（E-I 皮质柱脑区）研究项目

基于进化 / 强化学习训练**兴奋-抑制（E-I）皮质柱脑区**的时序决策模型。该项目经过十余个迭代实验，沉淀出经过考验有效的 EI-RNN 基础架构（`einbrain/` 包）。

## 目录结构

```
SiNNtry/
├── einbrain/         ★ 经过考验有效的 EI-RNN 整合通用包（供后续调用/实验）
│   ├── config.py       统一配置（脑/环境/进化/PPO/GPU/NEAT）
│   ├── env.py          贪吃蛇环境（24 维射线观测） + RaySnakeEnv（32 维 8 方向观测）
│   ├── brain.py        统一 EIBrainRegion 脑模型
│   ├── dynamics.py     共享 E-I 核心数学
│   ├── evolve.py       CPU 进化训练（test5d 语义）
│   ├── neat.py         真正 NEAT 引擎（创新号+物种化+历史标记交叉，test8 专用）
│   ├── ppo.py          PPO 强化学习（test6 语义）
│   ├── gpu.py          GPU 全并行进化（test7 语义）+ BatchedRaySnakeEnv
│   ├── io.py           统一保存/加载，兼容三格式模型互作种子
│   ├── deliberation.py K 帧思考
│   └── vis.py          可视化
├── experiments/      历史实验脚本（test1..test8 及 test5b/test5c，归档保留）
├── bench/             性能基准（列数扫描 / A-B 对比 / 加速比）
├── tools/             实时脑活动可视化服务器（brain_visualizer + static）
├── models/            训练好的模型权重（test4b/5/5a/5d/6/7 等，含早期 LSTM 预训练）
├── results/           训练曲线/拓扑等可视化输出图
├── logs/              运行日志
└── docs/              实验演进分析文档
```

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
```

三种训练模式的详细对比与设计动机见 `docs/test_evolution_analysis.md`。

## 环境

- Python 3.13，conda 环境 `env_torch`（`env_torch`）
- 依赖：`torch` / `numpy` / `matplotlib` / `networkx`；PPO 用 `gymnasium`（实验脚本 test3/test5a_lunar）；可视化可选 `python-louvain`
- GPU 模式（test7）依赖 CUDA（自动检测，可用 `cfg.DEVICE='cpu'` 回退）

## 运行历史实验

历史脚本归档于 `experiments/`，保持原样可复现：

```bash
cd experiments
python test5d.py                    # 最优进化版
python test5a_smoke_test.py         # 快速自检
python test6.py                     # PPO
python test7.py --smoke             # GPU 自检
python test8.py --smoke             # 真正 NEAT × EI-RNN（K=5，32 维观测）自检
```

## test8 — 真正 NEAT 与 EI-RNN 的契合度验证

`experiments/test8.py` 用**完整的 NEAT**（创新号 + 物种化/相容性距离 + 历史标记交叉 + add-connection/add-node 结构生长）驱动 EI-RNN 进化，验证二者契合度：

- **环境**：K=5（`FRAME_RATE=5`）贪吃蛇，新 **32 维头朝向相对 8 方向观测**（`RaySnakeEnv` / `BatchedRaySnakeEnv`）：蛇首方向 4 one-hot + 蛇尾方向（末节移动朝向）4 one-hot + 食物 8 扇区 + 自身 8 扇区 + 8 方向障碍距离倒数（身后槽约定为 `sqrt(蛇身长度/格子度)`，避免冗余不变量）。
- **启动**：256 柱 `INIT_DENSITY=0.15` 稀疏随机拓扑，为每条连接分配创新号；此后 add-connection / add-node 单调生长。
- **契合度指标**：每代物种数（自适应阈值收敛于 `SPECIES_TARGET`）、平均激活连接/柱数（结构生长）、BestFood 曲线。
- **性能**：POP=1024/N=256/K=5 下每代约 10–45s（`evolve` 父本继承 + 每 `RE_SPECIATE_INTERVAL` 代全量再物种化、`SPECIES_CAP` 兜底；`obs()` 全向量化；`decode_population` 批量 scatter；断点紧凑数组）。
- **运行**：`python test8.py --device cuda --pop 2048 --gens 100`；断点续训 / 最优模型保存 / `--play` 播放与 `einbrain.io` 完全兼容。

> 注意：历史脚本中的模型/断点路径按当时约定解析（相对运行目录）。当前模型权重统一存放在 `models/`；若需在历史脚本中加载模型，请将其复制到运行目录，或改用 `einbrain.io` 的路径解析。
