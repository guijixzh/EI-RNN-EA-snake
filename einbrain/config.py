"""统一配置类。

整合 test5d（CPU 进化）、test6（PPO 强化学习）、test7（GPU 全并行进化）
三套已过考验的 EI-RNN 实现的所有超参数。不同训练模式只取各自需要的子集，
保证一个 Config 对象即可驱动三种训练管线。

来源对照：
    - 脑结构 / E-I 动力学 / 激素 / 疲劳 / K 帧思考 / 进化筛选：test5d.py
    - PPO 超参 / 每步奖励 / 空转 / 饥饿：test6.py
    - GPU 并行 / 精度 / 显存：test7.py
"""
from __future__ import annotations


class Config:
    # ==================== 进化参数（test5d）====================
    POP_SIZE = 2048
    GENERATIONS = 100
    ELITE_SIZE = 256
    MUT_RATE = 0.05
    TOPOLOGY_MUT_PROB = 0.05
    WEIGHT_MUT_FRAC = 0.2
    WEIGHT_MUT_STD = 0.1
    TAU_E_MUT_STD = 0.05
    HORMONE_MUT_FRAC = 0.1
    HORMONE_MUT_STD = 0.05

    # --- 动态变异控制（余弦退火 + 指数衰减）---
    EVO_COS_MODE = 'anneal'        # 'anneal' 单调 1→0 | 'oscillate' 周期振荡
    EVO_COS_PERIOD = 100
    EVO_DYN_DECAY_TAU = 33

    # ==================== 真正 NEAT 进化参数（test8）====================
    # 创新号 / 物种化 / 历史标记交叉（见 einbrain.neat）
    COMPAT_THRESHOLD_INIT = 1.5     # 初始相容性阈值（自适应调整；随机初始对距离约 1.7）
    COMPAT_C1 = 1.0                 # excess 系数
    COMPAT_C2 = 1.0                 # disjoint 系数
    COMPAT_C3 = 0.4                 # 权重差系数
    SPECIES_TARGET = 8              # 目标物种数（阈值自适应锚点）
    SPECIES_ELITE = 1               # 每物种保底精英数
    SPECIES_CAP = 64                # 全量再物种化物种数硬上限（防 O(POP²) 复发）
    RE_SPECIATE_INTERVAL = 5        # 每 N 代全量再物种化；其余代后代继承父本物种
    ADD_CONN_PROB = 0.6             # add-connection 相对概率
    ADD_NODE_PROB = 0.3             # add-node 相对概率（其余为 disable）
    REENABLE_PROB = 0.1             # 重新启用被禁用连接的概率

    # --- 激素网络（G3）---
    TRAIN_HORMONE_NET = False      # False = 保持 0 初始化，G3 永久冻结
    HORMONE_GATE_THRESHOLD = 0.0   # 单柱释放门控：最大 logit 严格 > 阈值才释放

    # ==================== PPO 超参（test6）====================
    N_ENVS = 16                    # 向量化并行环境数（替代种群规模）
    ROLLOUT_LEN = 128
    TOTAL_ITERATIONS = 2000
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPS = 0.2
    VALUE_COEF = 0.5
    ENTROPY_COEF = 0.01
    PPO_EPOCHS = 4
    MINIBATCH_SIZE = 256
    LR = 3e-4
    LR_DECAY = True
    MAX_GRAD_NORM = 0.5

    # --- 每步奖励（test6）---
    UNSEEN_STEP_REWARD = 0.01
    SEEN_STEP_REWARD = 0.0
    EAT_REWARD = 1.0
    DEATH_REWARD = -1.0

    # --- 空转惩罚（防原地打转白赚步奖励）---
    LOITER_WINDOW = 12
    LOITER_REPEAT = 3
    LOITER_PENALTY = -0.1

    # --- 饿死截断 / 饥饿惩罚 ---
    STARVE_BIAS = 12               # steps_without_food > 2*len(body)+STARVE_BIAS
    HUNGER_WINDOW = 16
    HUNGER_STEP_PENALTY = 0.05

    # ==================== 环境参数 ====================
    GRID_SIZE = 10
    EVAL_EPISODES = 5
    MAX_STEPS = 500
    EVAL_INTERVAL = 20             # PPO 周期评估间隔（迭代数）

    # ==================== 脑结构参数 ====================
    NUM_COLUMNS = 256
    OBS_DIM = 24   # 3 食物方向 bit + 1 食物距离 + 5 射线×2 + 8 自体感知桶 + 2 尾巴局部坐标
    ACTION_DIM = 3
    INIT_DENSITY = 0.15
    MASK_FREEZE = True             # PPO：冻结拓扑掩码，只训练权重

    # ==================== E-I 动力学参数 ====================
    BASE_TAU_E = 0.7
    TAU_E_NOISE = 0.1
    TAU_E_MIN = 0.001
    TAU_E_MAX = 2.0
    W_EI = 2.0
    W_IE = 2.0
    W_EI_MUT_STD = 0.1
    W_IE_MUT_STD = 0.1
    W_EI_MIN = 0.0
    W_EI_MAX = 6.0
    W_IE_MIN = 0.0
    W_IE_MAX = 6.0

    # --- 短期 tau 调制 ---
    SHORT_TERM_GAIN = -0.2
    SHORT_TERM_DECAY = 0.3

    # ==================== 激素系统参数 ====================
    HORMONE_DECAY = 0.95
    EXCIT_HORMONE_GAIN = 0.25
    INHIB_HORMONE_GAIN = 0.50
    EXCIT_DIFFUSION = 0.15
    INHIB_DIFFUSION = 0.40
    HORMONE_NET_HIDDEN = 32

    # ==================== 动作疲劳（纯连续次数）====================
    FATIGUE_GAIN = 1e-5
    FATIGUE_THRESHOLD = 4
    FATIGUE_MAX = 5.0

    # ==================== K 倍帧率思考 ====================
    FRAME_RATE = 5
    INPUT_DECAY = 0.9

    # ==================== 进化筛选策略（test5d v2）====================
    LONG_SNAKE_SCORE_THRESHOLD = 3.0
    CYCLE_PATTERN = [('G2', 'G1', 'G3')]

    # ==================== 加速训练（test5_fast / test5d）====================
    PARALLEL_EVAL = True
    NUM_WORKERS = 0                # 0 = 自动 min(cpu_count, 16)
    SCREEN_ENABLE = True
    SCREEN_EPISODES = 2
    SCREEN_MULTIPLIER = 3
    SCREEN_AUTO_FALLBACK = True

    # ==================== GPU 全并行（test7）====================
    DEVICE = 'auto'                # 'auto' | 'cuda' | 'cpu'
    USE_FP16 = True
    EVAL_BATCH = 0                 # 0 = 按显存自动估算
    EVAL_MEM_FRAC = 0.55
    PRINT_HISTORY_EVERY = 1

    # ==================== 输出 / 断点 / 种子 ====================
    CHECKPOINT_PATH = 'checkpoint.pth'
    BEST_MODEL_PATH = 'best_model.pth'
    AUTO_RESUME = True
    CHECKPOINT_INTERVAL = 5
    SEED_FROM_BEST = True          # 进化：用已有最优模型作为种群种子
    SEED_MODEL_PATH = 'test5d_best_model.pth'    # test7 种子来源
    SEED_MODEL_PATH2 = 'test6_best_model.pth'    # test7 备选种子来源
    TEST5D_MODEL_PATH = 'test5d_best_model.pth'  # test6 PPO 种子来源
    SEED_FROM_TEST5D = True


def make_smoke_config():
    """小规模快速自检配置（各模式通用，跑得快）。"""
    cfg = Config()
    cfg.POP_SIZE = 16
    cfg.GENERATIONS = 3
    cfg.ELITE_SIZE = 6
    cfg.NUM_COLUMNS = 24
    cfg.EVAL_EPISODES = 1
    cfg.MAX_STEPS = 60
    cfg.FRAME_RATE = 2
    cfg.CHECKPOINT_INTERVAL = 2
    cfg.PARALLEL_EVAL = False
    cfg.SCREEN_ENABLE = False
    cfg.SEED_FROM_BEST = False
    cfg.SEED_FROM_TEST5D = False
    cfg.N_ENVS = 4
    cfg.ROLLOUT_LEN = 8
    cfg.TOTAL_ITERATIONS = 2
    cfg.MINIBATCH_SIZE = 32
    cfg.EVAL_BATCH = 16
    cfg.CHECKPOINT_PATH = 'smoke_checkpoint.pth'
    cfg.BEST_MODEL_PATH = 'smoke_best.pth'
    return cfg
