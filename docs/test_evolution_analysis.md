# EI-RNN 实验演进分析

> 逐文件分析本仓库各 `test` 脚本的**内容与目的**，梳理 EI-RNN（E-I 皮质柱脑区）从零到成熟的技术演进主线，并说明最终沉淀为 `einbrain/` 整合包的基础架构。

---

## 1. 技术演进主线

```
test1 (LSTM预训练)
   └→ test2 (CartPole, 冻结柱+可塑外模块) ── base ability done
        └→ test3 (LunarLander, 规模扩展)
             └→ test4 (贪吃蛇, 真E-I动力学+拓扑进化) ── EI-RNN 诞生
                  ├→ test4a (+激素+价值头TD)
                  │     └→ test4b (工程化重构, 修复转圈) ── 首个可信版本
                  │          └→ test4c (纯预测编码链接训练分析)
                  └→ test5 (移除PC与拉马克, 解锁Wei/Wie, K帧思考)
                       ├→ test5_fast (并行评估+两阶段筛选)
                       ├→ test5a (交替冻结进化 G1/G2/G3)
                       │     ├→ test5a_lunar (LunarLander移植)
                       │     ├→ test5b (CNN前端, 失败)
                       │     └→ test5c (CNN前端, 失败)
                       └→ test5d v2 (三元组筛选+单柱激素门控+动态变异) ── ★最优进化版
                            ├→ test6 (PPO梯度训练, 替代进化)
                            └→ test7 (GPU全向量化进化, 去激素)
```

- **★ 核心结论**：经过多轮验证有效的 EI-RNN 基础架构，最终沉淀在 `test5d`（CPU 进化）、`test6`（PPO）、`test7`（GPU 进化）三套实现中，三者共用同一套 E-I 动力学。该基础架构已被抽离为 `einbrain/` 整合通用包。

---

## 2. 逐文件分析

### 2.1 预训练与早期验证

| 文件 | 内容 | 目的 / 结论 |
|---|---|---|
| `test1.py` | 32 维 LSTM 皮质柱在 4 个时序任务（正弦波 / 延迟 XOR / 去噪 / Lorenz）上多任务预训练 | 为脑区提供**冻结的时序特征编码器**。产出 `cortical_column_pretrained.pth`，后续 test2/test3 加载后冻结 LSTM 参数。 |
| `test2.py` | CartPole-v1：32 柱脑区 + 稀疏内部连接 `M`（connectivity=0.1）+ 预测编码微调 + 软拉马克遗传 + 自适应变异率 | 验证核心设计假设：**预训练柱冻结、外部模块可演化**。内置 3 项架构检验（柱权重一致性 / 外模块演化偏移 / 拓扑稀疏性）。git 提交标记 "base ability done"。 |
| `test3.py` | LunarLander-v3：脑区扩展到 64 柱 / 8 维输入 / 4 维输出 | 验证**规模扩展**可行性；引入 80% 交叉 / 20% 克隆的进化策略。 |

### 2.2 EI-RNN 诞生（贪吃蛇）

| 文件 | 内容 | 目的 / 结论 |
|---|---|---|
| `test4.py` | 自研轻量贪吃蛇环境 + **真正的 E-I 动力学**（代数式：`E=σ(输入+τ·E-ωei·I)`，`I=σ(ωie·E)`）+ 拓扑掩码（`M_in/M_rec/M_out`）作为基因型参与进化（NEAT 风格） | 首个真正的 **EI-RNN**。验证稀疏拓扑 + E-I 柱在序列决策任务上可进化出有效策略。 |
| `test4a.py` | 在 test4 基础上新增：**激素控制器**（前馈网络输出每柱兴奋/抑制激素，沿拓扑扩散）、价值头 + 输出侧 TD 预测编码、增强拉马克遗传（只回传表征/激素参数，**不回传 W_out/W_value**，避免污染进化方向） | 引入**神经调节**（neuromodulation）机制。 |

### 2.3 工程化与"可信版本"

| 文件 | 内容 | 目的 / 结论 |
|---|---|---|
| `test4b.py` | 全面重构：`Config` 集中管理参数、256 柱、`W_rec_eff/W_out_eff/M_norm` 缓存、24 维射线观测、**动作疲劳**（纯连续次数）、检查点/断点续训/最优模型/种子继承、fp16 序列化 | **首个工程化可信版本**。git 提交标记 "优化与修复了转圈问题"。`test4b_best_model.pth` 为重要中间成果。 |
| `test4b_orig.py` | test4b 优化前的原始实现（无缓存、深拷贝克隆、更慢） | 与 test4b 对照，供基准测试（`bench_test4b.py`）验证优化不改数值语义。 |
| `test4c.py` | 加载 test4b 最优模型，**仅用预测编码误差在线训练**链接权重（W_pred 充分训练 + W_in 慢速微调），冻结拓扑与 EI 柱自身参数；含 `--zero-interference` 对照模式 | 验证**预测编码对已进化模型的附加价值**（是否值得在行为模型上继续做 PC 训练）。 |

### 2.4 简化与加速

| 文件 | 内容 | 目的 / 结论 |
|---|---|---|
| `test5.py` | **移除预测编码与拉马克遗传**（git 提交标记 "PC 与拉马克 is a lie"）、解开 `w_ei/w_ie` 逐柱进化封印、引入 **K 倍帧率思考**（`FRAME_RATE=5`，观测逐次衰减、logits 平均） | 用消融实验证明：进化 + K 帧思考足够，PC 与拉马克并非必要。 |
| `test5_fast.py` | test5 的加速基建：**A 方案多进程并行评估 + C 方案两阶段快速筛选**（初筛 → 前 K 名精评），专属断点/最优模型路径 | 将 256 柱 / 2048 种群的训练时间压缩到可接受范围。 |
| `test5a.py` | **交替冻结进化（B 方案）**：参数分 G1 结构 / G2 动力学 / G3 激素三组，按 `CYCLE_PATTERN` 周期轮换激活，冻结组不交叉不变异 | 抵抗解锁 Wei/Wie 后的**自由度爆炸**。 |

### 2.5 移植与失败尝试

| 文件 | 内容 | 目的 / 结论 |
|---|---|---|
| `test5a_lunar.py` | 将 test5a 方法论移植到 LunarLander-v3（8 维 obs / 4 维 action，`FRAME_RATE=1`，去疲劳，动态变异率） | 验证 EI-RNN 跨环境可迁移。 |
| `test5b/` | **CNN 前端尝试 1**：冻结全局共享的预训练 CNNEncoder（10×10×5 网格 → 24 维语义特征）喂给 EI 脑区；含 `env.py / cnn.py / expert_ai.py / collect_data.py / train_cnn.py` 全链路 | 试图让 EI-RNN 直接处理**像素级输入**。**头注释明确标记：失败**。 |
| `test5c/` | **CNN 前端尝试 2**：CNN 投影改为 16 维 + BatchNorm1d（修复 Tanh 压窄导致"永远直行"退化），去掉疲劳；`collect_crisis.py` 用随机游走蛇生成危机态数据 | **再次失败**。教训：EI 脑区适合低维语义观测，像素级感知需另行设计。 |

### 2.6 最优进化版与范式切换

| 文件 | 内容 | 目的 / 结论 |
|---|---|---|
| `test5d.py` | **最优进化版 v2**：(1) 进化筛选改为三元组 `(food, seen, unseen)`，25 分前后切换排序压力（低分保"看见秒吃"，高分保"看不见活得久"，防长蛇追食自杀）；(2) 激素网络默认不训练 + **单柱释放门控**；(3) 动态变异（拓扑余弦退火 + 动力学指数衰减） | **当前最佳、验证最充分的 CPU 进化版 EI-RNN**，被 test6/test7 用作种子模型。 |
| `test6.py` | **PPO 强化学习**：同一脑结构改为可微 `forward_ppo`/`forward_ppo_k`（显式状态 + 截断 BPTT），新增 value head `V/b_v`；16 向量化环境替代 2048 个体种群；seen/unseen 每步奖励 + 空转/饥饿惩罚 | 从"进化"切换为"梯度 RL"范式，信息利用率提升数个量级。 |
| `test7.py` | **GPU 全并行进化**：`GeneStack` 把整个种群堆叠为 `[POP,N,N]` 批量张量，交叉/变异全向量化；`BatchedSnakeEnv` 批量并行游戏；**彻底移除激素支路**（与 test5d 零激素动力学一致）；显存自适应分块 | 将进化训练速度提升至全 GPU 级别。 |

### 2.7 基础设施 / 工具

| 文件 | 内容 | 目的 |
|---|---|---|
| `bench/bench_nscan.py` | NUM_COLUMNS 可扩展性扫描（内存 / 纯前向 / 综合评估单步耗时） | 指导柱数选择。 |
| `bench/bench_test4b.py` | test4b_orig vs test4b 的 A/B 基准 + 数值一致性 + 同种子结果一致性 | 验证优化不改语义。 |
| `bench/bench_test5_fast.py` | 串行全量 vs 并行全量 vs 并行+筛选 三种评估模式加速比 | 验证加速基建收益。 |
| `tools/brain_visualizer.py` + `tools/static/` | HTTP + SSE 实时脑活动可视化服务器（拓扑、神经元活动热力图、观测条、动作潜力），前端为无 CDN 原生页面 | 观察训练好的大脑在线游玩时的内部动力学。 |
| `tools/start_server.bat` | 自动定位 env_torch 环境并启动可视化服务器 | 一键启动。 |
| `experiments/*_smoke_test.py` | 各主要实验的快速自检脚本（架构不变量、冻结组逻辑、评估口径一致性、检查点往返、小规模跑通） | 回归保障。 |

---

## 3. 基础架构沉淀（einbrain 包）

经过考验有效的 EI-RNN 基础架构被单独保存为 `einbrain/` 通用包，三套训练范式共用同一脑模型：

```
einbrain/
├── config.py        # 统一 Config（脑/环境/进化/PPO/GPU 全参数）
├── env.py           # SnakeEnv（24维射线观测 + 空转/饥饿）
├── brain.py         # EIBrainRegion（权重/掩码/tau_e/w_ei/w_ie/激素/疲劳/价值头）
├── dynamics.py      # 共享 E-I 核心数学（ei_update/激素/疲劳），单脑与 GPU 批量共用
├── evolve.py        # CPU 进化：evaluate/evolve_topology/并行评估/两阶段筛选/run_evolution
├── ppo.py           # PPO：forward_ppo_k/GAE/ppo_update/run_training
├── gpu.py           # GPU：GeneStack/BatchedSnakeEnv/run_training_gpu
├── io.py            # 统一保存/加载，兼容 test5d/6/7 三种模型格式，互作种子
├── deliberation.py  # K 帧思考 deliberate_action
└── vis.py           # 训练曲线 / 贪吃蛇游玩 / 拓扑可视化
```

三种训练模式：

| 入口 | 范式 | 对应实验 |
|---|---|---|
| `run_evolution(cfg)` | CPU 进化（多进程 + 筛选） | test5d |
| `run_training(cfg)` | PPO 强化学习（K 帧思考 + 截断 BPTT） | test6 |
| `run_training_gpu(cfg)` | GPU 全向量化进化 | test7 |

> 注意：`test5d_best_model.pth` 带有**已训练的激素权重**，仅 `EIBrainRegion`（CPU/PPO 路径）能利用；GPU 路径为无激素架构，其与单脑前向的等价性仅在激素权重为 0（如 test7 模型）时严格成立（已由单元测试验证，最大误差 0.0）。
