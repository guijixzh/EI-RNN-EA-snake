# test13 实验日志 —— 冠军脑弱剪枝拓扑固化 + PPO 强化学习微调

> 2026-08-30 ~ 08-31。范式交叉实验：进化冠军脑（test7b）→ 20% 弱剪枝+
> 拓扑永久固化 → einbrain PPO 梯度微调。**总判定：不可行**（提分失败、
> 纯侵蚀）；两条工程结论已沉淀进 einbrain。详细数据表见
> `results/test13_ppo_summary.md`。

## 0. 继承关系与归档位置

```
test7b 冠军模型 (根目录 test7b_latest_gen_best.pth, 进化期 Food=67)
   └→ test13 (弱剪枝拓扑固化 + PPO 微调, 本日志)
        ├→ einbrain PPO（test6 语义）+ ProjSnakeEnv（test7b 观测接口精确复刻）
        └→ 前置基准：test7b_benchmark.py --weak-mask-frac（弱剪枝 0–30% 基准）
```

- **归档位置**：`experiments/test13_ppo/`（test13_ppo.py、全部模型
  seed/best/smoke .pth、history json/png 含 noanchor 对照）。
- 环境等价性对拍脚本：`experiments/verify_proj_env.py`——ProjSnakeEnv 与
  test7b BatchedSnakeEnv 共用同一 CPU Generator 逐步对拍，43.6 万步 PASS、
  obs 最大差 2.4e-07（PPO 微调期与进化期输入分布一致）。

## 1. 主要目的

验证"进化找到的冠军脑能否再用梯度 RL 继续提分"：剪掉最弱的 20% 连接并
永久固化拓扑（掩码置 0 后 forward_ppo 梯度路径 W*M 无法复活，双重固化；
G2 动力学 tau_e/w_ei/w_ie 仍可学习），PPO 奖惩（吃食 +1 / 死亡 −1 / 饥饿
步惩罚），K=5 帧思考与 test7b 推理口径一致，周期评估走 test7b GPU 批量
评估（与 1000 局基准同口径）。

## 2. 过程与重要更新

### 实验 1：事后弱剪枝 0/10/20/30% × 1000 局基准

| 剪枝 | 均值 | 中位 | 撞墙% | 自撞% |
|---|---|---|---|---|
| 0% | 61.35 | 62 | 22.6 | 76.8 |
| 10% | 61.05 | 62 | 22.2 | 77.1 |
| 20% | 60.67 | 62 | 14.6 | 85.0 |
| 30% | 61.66 | 62 | 30.9 | 68.1 |

**结论**：事后弱剪枝对分数无显著影响（均值差 <0.7，组内 std≈7–8，n=1000
标准误 ≈0.25）；test7h 的"+1.8 分"不迁移——那是与剪枝共进化的种群去内噪
收益。剪枝不改变分数但**重排队死因结构**（20% 把撞墙死亡换给自撞，30%
反向）。数据：results/test7b_weakmask_bench.json。

### 实验 2：拓扑固化 + PPO 微调（贪心 128 局 EvalFood 轨迹）

| 配置 | LR | KL 锚 | 温度 | 轨迹 | 结果 |
|---|---|---|---|---|---|
| A | 1e-4 | ✓ | 1.0 | 62→30@20 | 崩溃 |
| B | 3e-5 | ✓ | 1.0 | 58.8→55.1@60 | −9.3/100 |
| C | 1e-5 | β=.05* | 1.0 | 61.9→56.8@160 | −3.2/100 |
| D | 2e-5 | β=.2* | 0.5 | 61.3→54.8@240 | −3.0/100 |
| E（无锚定） | 2e-5 | — | 0.5 | 60.1（冻结验证）→52.1@180→34.7@2000 | 平台 ~34 |
| E'（真锚定） | 2e-5 | β=.2 | 0.5 | 61.2→57.8@200→52.5@1600（缓降平台） | −0.6/100 |

\* 配置 C/D/E 的"锚定"事后发现无效：`save_brain_state(fp32)` 与原脑共享
存储，anchor 与 live brain 同体，KL(new−anchor)≡0。E' 为修复后的真锚定。

### 关键发现

1. **PPO 对该冠军脑只有侵蚀、没有改进**。六股机制：采样行为(45 步)与贪心
   (1100 步)鸿沟、value 头冷启动、value 梯度流经共享主干（DETACH_VALUE_TRUNK
   修复）、优势归一化在收敛点放大噪声、势奖励恒定推向均匀、γ=0.99 信用
   视野(~460 步) << 局长(~1100 步)。
2. **真 KL 锚定把侵蚀速率降 3 倍以上**（−3.4 → −0.6 /100 iter）并稳住
   平台，但不产生改进——信噪比差 2–3 个数量级，锚定球内净漂移仍为负。
3. **`save_brain_state` 存储共享 bug（已修复）**：fp32 镜像与原脑共享参数
   存储，曾导致 best_brain/"锦"随续训原地漂移；best 模型文件一度保存了
   退化权重+种子元数据。修复后 E' 的 best 镜像 256 局=60.75、1000 局=60.70
   （种子水平），保护机制真实生效。
4. **"完成贪吃蛇"（100 分）不可达**：32 维局部扇区观测 + 冻结拓扑不携带
   全局占用/规划所需信息——这是表示天花板，不是优化问题。

### 最终 1000 局基准（test13_ppo_best_model.pth = 真种子镜像，含 20% 固化剪枝）

| 再剪枝 | 均值 | 中位 | 最大 |
|---|---|---|---|
| 0% | 60.70 | 62 | 81 |
| 0.2（再剪 20%） | 58.72 | 61 | 78 |

## 3. 总判定与工程沉淀

- **总判定**：拓扑固化 + PPO 微调在当前任务形态下不可行（提分失败、纯
  侵蚀）；"100 分"属表示天花板。
- **有效工程结论**：① 真 KL 锚定是侵蚀的有效减速带；② 快照隔离必须显式
  clone（einbrain/io.py 修复）；③ 代码沉淀：einbrain/env.py::ProjSnakeEnv、
  einbrain/ppo.py（ENV_CLS/EVAL_FN/KL_ANCHOR_BETA/SAMPLE_TEMP/ADV_NORMALIZE/
  POLICY_WARMUP_ITERS/设备支持）、einbrain/brain.py（forward_ppo tau 顺序
  修复 + DETACH_VALUE_TRUNK）。

## 4. 产物索引

- `experiments/test13_ppo/`：test13_ppo.py（CLI 全开关）、
  test13_ppo_best_model.pth（=固化种子镜像，可比基准用）、
  test13_ppo_history.{json,png}（E' 真锚定）、test13_ppo_history_noanchor.*
  （E 无锚定）、test13_ppo_seed.pth、test13_smoke_*（冒烟）。
- `results/`：test13_ppo_summary.md（详细数据表）、
  test13_ppo_best_bench.json（E' 终评）、test7b_weakmask_bench.json（实验 1）。
- `logs/`：test13_ppo_run.log、test13_ppo_run_noanchor.log。
