# snake_std — 贪吃蛇进化标准实现程序（实验日志）

状态：**活跃（标准入口）** ｜ 基座：test16b ｜ 日期：2026-09-12
环境：conda `env_torch`（`python`），Python 3.13 + PyTorch（CUDA 自动检测）

## 1. 定位

把 16 系列全部已验证机制收敛为**单一可配置入口** `snake_std.py`：观测环境、适应度
公式、筛选方案、系统开关四个能力面全部由 CLI 选择。基因组/CRN 与 16 系列逐字
同源（`BRAIN_VERSION='sparse1'`），16 系列断点可直接续训。

> **2026-09-12 实证回调**：推荐默认已从 16b 原口径改为
> **simple 适应度 × 32ego 观测 × N=256 / K=96**——全部最优模型（7b 61.4 /
> 16b_simp 62.7 / cheat7b 通关）均出自 simple；观测增维到 40 无明显突破
> （仅小幅提升跨盘面可迁移性）；N=1024 在固定扇入限制总连接数与 256 相当的
> 口径下未见优化且未条件完成完整实验；K=96 无损覆盖冠军连接数（≥64 无明显
> 损失）。复现 16b 原口径请显式传参：
> `--obs 40 --columns 1024 --fanin 16 --fit-mode econ`。

四个能力面与其来源谱系：

| 能力面 | 选项 | 来源 |
|---|---|---|
| 观测环境 | `--obs 40 / 32ego / 32proj / 24` | test15/16 系（40tailflood1）／test12（32ego1）／test7b（32proj7b）／test7（休眠） |
| 适应度公式 | `--fit-mode econ / simple / tuple` ＋ `--robust-eval R` ＋ `--te-elite N` ＋ LCB `--sel-lcb/--sel-cv` | test12 v7／test7b v9／test7 词典序／16c_cheat7b v20／7h 配额／16a v8 |
| 筛选方案 | `--no-stage2`（单阶段）／默认二阶段／`--stage2-halving`（对半精评）／`--no-k2-adapt` 等（自适应K2） | test7g／test7h v2→test16／test16b B 层／test16a |
| 系统开关 | `--train-hormone` ＋ `--cycle-pattern` ＋ `--pools` ＋ `--fixed-map` ＋ 疲劳/CRN/弱掩码/单侧判死/fast-eval；**研究选项（test17 系列，默认关）**：`--kframe-input/--kframe-read`（K帧首帧输入/尾帧读出消融）＋ `--euler-leak`（E 泄漏积分器，st σ迹负反馈，τ≤1）＋ `--train-wii/--train-wee`（I/E 对称自项基因） | einbrain brain+dynamics 批量移植／test5a 轮换／test16c／16c_cheat7b／test7 系／test17+17A+17B'（见 docs/test17_kft_experiment_log.md，fork `experiments/test17_kft/test17b_leak.py` 为血统源） |

## 2. 开关矩阵（标准命令）

```bash
PY=python

# 推荐起点（默认：32ego + simple v9 + N256/K96 + 二阶段 + 对半精评 + fast-eval）
$PY -X utf8 -u snake_std.py --gens 320

# 复现 16b 原口径（obs40 + econ v8 + N1024/K16）
$PY -X utf8 -u snake_std.py --obs 40 --columns 1024 --fanin 16 --fit-mode econ --gens 320

# 观测环境三选一
$PY -X utf8 -u snake_std.py --obs 32proj       # test7b 8 扇区投影（32 维）

# 适应度方案
$PY -X utf8 -u snake_std.py --fit-mode simple          # v9 最简 food+k·eff
$PY -X utf8 -u snake_std.py --fit-mode tuple           # test7 词典序
$PY -X utf8 -u snake_std.py --robust-eval 2            # v20 鲁棒最小值（σ=1e-3）
$PY -X utf8 -u snake_std.py --te-elite 24              # te 配额精英

# 筛选方案
$PY -X utf8 -u snake_std.py --no-stage2                # 单阶段（全种群 ×K1 排序）
$PY -X utf8 -u snake_std.py --no-stage2-halving        # 二阶段但不精评
$PY -X utf8 -u snake_std.py --no-k2-adapt --stage1-eps 12 --stage2-eps 24

# 系统开关
$PY -X utf8 -u snake_std.py --train-hormone --cycle-pattern "G1,G2,G3"
$PY -X utf8 -u snake_std.py --cycle-pattern "G2,G1"    # 默认轮换（可任意序列）
$PY -X utf8 -u snake_std.py --pools --sensory-frac 0.5 --motor-frac 0.25
$PY -X utf8 -u snake_std.py --fixed-map --map-seed 20260908

# 组合示例：固定地图 + 鲁棒评估 + 池约束通关特训
$PY -X utf8 -u snake_std.py --fixed-map --robust-eval 2 --pools --fanin 64 --columns 256

# 断点续训（16 系列互认）
$PY -X utf8 -u snake_std.py --resume-pop artifacts/../test16a_simp_checkpoint.pth
$PY -X utf8 -u snake_std.py --play    # 播放 snake_std_best_model.pth（或改 cfg.BEST_MODEL_PATH）
```

## 3. 实现要点

### 3.1 三种观测环境
- **40**（`40tailflood1`）：`_obs40`/`_obs40_fast` 原样（32ego1 前 32 通道 + 钟压/尾方位/洪水稀缺 [32:40]）。`--no-new-obs` 置零新通道（A0 对照）。
- **32ego**（`32ego1`）：**与 obs40 共用同一实现**——`OBS_DIM=32` 时 `_obs40`/`_obs40_fast` 的新通道块被 `OBS_DIM >= 40` 门控整块跳过，前 32 通道逐位相同（自检 9 证明）。注意：test12/14/15 的 32ego 旧模型是稠密基因组，不可载入稀疏标准程序（`load_migratable_state` 按设计拒绝）；32ego 环境用于训练新的稀疏世代。
- **32proj**（`32proj7b`）：`_obs32`/`_obs32_fast`/`_d8_vectors` 从 test16c_cheat7b.py 移植（8 扇区欧氏投影 `K·max(0,v·û)/|û|/|v|²`，食物永远可见）。
- checkpoint 守卫按 `OBS_ENC_VERSION` 字符串比较（`load_checkpoint7`/`load_best_state`/`--resume-pop` 三处），三种编码天然互斥互认。

### 3.2 激素系统（批量移植，`--train-hormone`）
16b 前向无激素支路（仅零填充兼容字段）。从 `einbrain/brain.py forward` + `einbrain/dynamics.py` 移植**批量版**到 GeneStack/forward_batch：
- 激素基因（G3 组）：`Wh1 [B,H,3N]`、`b_h1`、`W_excit/W_inhib [B,N,H]`、`b_excit/b_inhib [B,N]`，零初始化（零初始化 = 无操作，test14 教训：logit 0 不越过门控阈值）。仅 `TRAIN_HORMONE_NET=True` 时分配；`individual_state` 开启时存真实基因、关闭时零填充（16 系列格式兼容）。
- 前向支路：`hh=relu(Wh1·[E;I;total])` → 单柱头 logits → **单柱门控释放**（每类激素同帧单柱、严格 > 阈值、强度 sigmoid(max)，批量=按行 argmax/scatter）→ **稀疏扩散** `h ← (1−decay)·cmd + decay·((1−d)·h + d·M_norm·h)`，其中 `M_norm·h` 用扇入 gather 复用 rec 前向模式（`_mnorm_slot = 1/行内不同目标数`，O(B·N·K)，免建 B×N×N）→ `tau_e += EXCIT_GAIN·he − INHIB_GAIN·hi`。
- 进化：G3 逐元素掩码交叉 + `HORMONE_MUT_FRAC/HORMONE_MUT_STD`×s 噪声变异（einbrain evolve.py 同语义）；`_freeze_active_groups` 已有 G3 丢弃逻辑（ hormone 关时）。
- 状态：`he/hi [B,N]` 每局零初始化、随 K 帧内部迭代逐帧更新（与 einbrain 单脑 forward 一致）；`forward_batch/deliberate_batch` 统一返回 6 元组，关闭时 he/hi=None 且**不进支路（逐位=16b 前向，自检 11 证明）**。
- 守卫：断点 `TRAIN_HORMONE_NET` 与当前 run 不符 → `sys.exit`（拒绝静默丢弃激素基因）；激素开启但轮换无 G3 → 打印提示（激素永远冻结在零）。

### 3.3 筛选方案派发
- `evaluate_population_gpu`：`two_stage = not NO_STAGE2 and STAGE2_KEEP ≥ ELITE and K1>0 and K2>K1`，否则单阶段返回。
- `run_training`：`halving = STAGE2_HALVING and not NO_STAGE2 and STAGE2_KEEP > ELITE and K1>0`。
- 鲁棒评估 `_eval_pop_robust` 包装全部 5 个评估调用点（二阶段 1/2、对半 2a/2b、单阶段 1）：R=1 零开销透传；R≥2 时分块复制 R 份（副本 0 原权重，副本 1..R-1 对 W_in/W_out/rec_w/b_out 加 σ 噪声、tau_e 加噪后 clamp），评估后按 food 取最差副本整行指标。

### 3.4 其余
- `--cycle-pattern "G1,G2,G3"`：解析为 `CYCLE_PATTERN=[('G1','G2','G3')]`（逐代取模）。
- `--fixed-map`：`_crn_seed` 的 base 与 gen 钉死为 `(MAP_SEED, MAP_GEN)`——全部阶段/局共用一张地图（CRN 机制不变）。
- `--pools`：`_pool_masks`（POOL_SEED 确定性）作用于 `random_init` 的 M_in/M_out 与拓扑变异的 M_in/M_out 翻转守卫（池外翻转无效）。
- 输出前缀 `snake_std_*`（`--name`/`--fit-mode` 派生 arm 前缀同 16b）。
- FITNESS_VERSION：econ=8 / simple=9 不变（16b 断点互认、best 追踪保留）；`--robust-eval≥2` → 21、`--train-hormone` → 22（评估口径变化 → 导入旧断点时 best 追踪按既有规则重置，种群/历史保留）。

## 4. 验证记录（2026-09-12，env_torch / RTX 5070 Laptop）

### 4.1 自检 `--selfcheck`（18 组全过；2026-09-12 test17→17A→17B' 逐次增补 15/16/17/18 组）
原 16b 8 组（食物编码/习惯三因素/单侧判死/v7 公式+杠杆+分解/变异分布/弱掩码/CRN 确定性/稀疏-稠密前向等价/重连变异/LCB/`_obs40_fast`≡`_obs40`）＋ 新增 8 组：

| # | 检查 | 结果 |
|---|---|---|
| 9 | 32ego ≡ obs40 前 32 通道（slow+fast，共享 CRN 库 30 步随机对局） | max\|Δ\| = 0 逐位一致 ✅ |
| 10 | 32proj ≡ test16c_cheat7b._obs32（25 步随机对局 slow+fast 对拍） | max\|Δ\| = 0 逐位一致 ✅ |
| 11 | 激素零初始化 ≡ 无激素（logits/E 逐位同、he 恒零）；非零激素两跑确定性 | Δ=0 / Δ=0；与无激素差异 1.63e-04 > 0 ✅ |
| 12 | 鲁棒评估机制：σ=0 时 R=2 ≡ R=1 逐位；σ>0 副本行为差异 3.9e+01 > 0 | ✅ |
| 13 | 池约束完整性：初始化 + 全强度变异后池外非零 = 0/0 | ✅ |
| 14 | 固定地图：FIXED_MAP 跨代同流 / 关闭后跨代换流 | ✅ |
| 15 | test17 K帧消融：first+tail ≡ 手写参考循环逐位；decay+sum ≡ 原公式逐位（FATIGUE_TURN_GAIN=0.5 覆盖 press 置零支路）；四模式动作非退化 | ✅ |
| 16 | test17A w_ii/自连：off 不分配 pack 无键/同 seed 核心一致；零 w_ii ≡ 无 w_ii 逐位；非零确定性 + I 状态差异 + 二帧 logits 差异 >0；自连开启全强度重连自环 >0 且无越界（自检6 同步条件化） | ✅ |
| 17 | test17B' E 泄漏积分器：保留比值（first 调度尾/首 ≈0.10，对照旧架构 ~1e-5）；零输入 200 帧有界；st σ迹负反馈生效且方向正确（高 st→小 τ→E 增长更慢）；st∈[0,1] 有界；确定性；τclamp≤1 | ✅ |
| 18 | test17B' E 对称自项 w_ee：off 不分配/同 seed 核心一致；零 w_ee ≡ 无 w_ee 逐位；非零确定性 + 生效差异 >0；individual_state 往返 | ✅ |

### 4.2 冒烟矩阵（--smoke 3 代端到端，14 项全过）
默认(=16b)、`--obs 32ego`、`--obs 32proj`、`--train-hormone --cycle-pattern G1,G2,G3`、`--robust-eval 2`、`--pools`、`--fixed-map`、`--no-stage2`、`--fit-mode simple`、`--fit-mode tuple`、`--no-stage2-halving`、`--kframe-input first --kframe-read tail`（test17）、`--allow-self-conn --train-wii`（test17A）、`--euler-leak --train-wii --train-wee`（test17B'）——全部训练完成、checkpoint/best/history 落盘正确。
（注：连续冒烟共用 `snake_std_smoke_*` 文件时会触发激素/开关断点守卫拦截——守卫按设计工作，矩阵用 `--name` 隔离后全过。）

### 4.3 等价门（显式 16b 口径 = 16b 行为；2026-09-12 默认回调前为默认配置）
`snake_std.py --smoke --seed 42 --obs 40 --columns 1024 --fanin 16 --fit-mode econ`
vs `test16b.py --smoke --seed 42 --columns 1024 --fanin 16 --fit-mode econ`：history 的
`best_food / avg_food / best_fit / best_seen / elite_food` 全轨迹逐位一致 → **PASS**（test17/17A/17B' 合并后复跑仍 PASS）。

**移植保真门（test17B' 合并验证）**：`snake_std.py --smoke --euler-leak --train-wii
--train-wee`（其余同上，`--name` 隔离）vs 血统源 `experiments/test17_kft/test17b_leak.py`
同口径冒烟：history 五轨迹逐位一致 → **PASS**——标准实现的研究选项与实验 fork 数值完全等价。

### 4.4 向后兼容
- `--resume-pop test16b_smoke_checkpoint.pth`：16b 断点（gen3）导入 → 续训至 gen5 ✅
- `--play` 加载基准模型 `test16b_simp_best_model.pth`（千局 62.67）：正常游玩（200 步 Food=12，截断口径）✅

### 4.5 示例种子权重与 7b 血统直读（2026-09-12 test17B' 合并后增补）

- **根目录示例权重**：`cheat7b_win_model.pth`（后训练/特化，稀疏 K=96 直读）、
  `test7b_base_model.pth`（7b 基模，稠密基因组）。`load_seed_state_any` 为
  `--seed-model` / `--play`（新增 `--play-model` 任意路径）统一入口：稀疏格式
  走原守卫直读；7b 稠密（W_rec/M_rec）自动走 `load_best_state_7b_dense_as_sparse`
  转换（算法移植自 test16c_cheat7b：掩码内 top-|W| 防幽灵 + 无自连重定向），
  要求 `--obs 32proj`。7b 基模转换**无损**（行非零 max 67 ≤ K=96）。
- 保真实测（标准实现全路径，10 局随机图）：test7b 基模 54.6 食（千局 61.4 同水位）、
  cheat7b 68.0 食（千局重评 63.2/中位 74 同水位）。
- 已知特性：cheat7b 在个别随机图有零分盘（README 结论 1 已记录的特化尾部代价）；
  以它为种子的单图微型冒烟可能恰好命中零分盘（gen0/stage1/ep0 即一例，
  ep1-3 同引擎 28-30 食），非机制缺陷。

## 5. 与 16 系列脚本的关系

- test16 系列脚本归档于 `experiments/test16_series/`（test16b.py 保留为参考基线：
  `test16b_benchmark.py` 千局基准与等价对照仍在用，模块装载路径已指向归档位置）；
  新实验一律从根目录 `snake_std.py` 起步。
- test16c（池约束）/ test16c_cheat7b（固定地图+鲁棒+7b种子）的特殊机制已并入
  标准程序，但 cheat7b 的「7b 稠密→稀疏迁移器」（`load_best_state_7b_sparse`）
  与 N=256/K=96 专属配置未并入——复现 cheat7b 请仍用
  `experiments/test16_series/test16c_cheat7b.py`（其 SEED_MODEL_PATH 指向
  `artifacts/test7b/`）。
- 轮换系统沿用 test16 的 2 代 `('G2','G1')` 默认（test5a 的 5 槽复杂轮换可用
  `--cycle-pattern` 重建）。

## 6. 已知边界

- 32ego/32proj 模式下 `--no-new-obs` 无意义（新通道块本就不存在），不报错但不生效。
- 激素开启时前向每帧多 2 次 bmm（[B,H,3N]×1 与 2 次 [B,N,H]），显存约 +1.1GB（pop4096/N1024/fp16）；建议配合 `--eval-batch` 或默认自动分块。
- 鲁棒评估 R 倍放大评估局数（精英 K=R×(K1+K2) 的算力口径，不是局数口径——每副本各打满 K 局）。
- `OBS_MODE='24'` 休眠路径未列入自检矩阵（test7 血统遗留，历史模型兼容用）。
