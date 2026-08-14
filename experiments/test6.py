# ==========================================
# test6.py —— PPO 强化学习版（单网络 + 16 向量化环境 + K 倍帧率思考）
#
# 对比 test5d（进化搜索）的核心改进：
#  1. 范式切换：进化（无梯度，2048 个体每代仅 1 个标量适应度）→ PPO（梯度，
#     每步交互都产生可微梯度信号），信息利用率提升几个数量级。
#  2. 每步奖励区分（用户指定）：
#       - 没看到食物（unseen）：正奖励（+0.01），鼓励巡航探索
#       - 看到食物（seen）  ：0.0（低于 unseen 保留"看到更低"语义，但非负值——
#                             负值会训练蛇"远离食物"导致空转，已在修复中消除）
#       - 吃食物：+1.0；死亡/饿死/超时：-1.0
#  2b. 空转惩罚（防原地打转）：蛇头在 LOITER_WINDOW 步内重复 ≥ LOITER_REPEAT 次
#      判定为空转，之后每步 -LOITER_PENALTY（默认 -0.1）；正常巡航不受罚。
#  3. 自身八桶输入改造（用户指定）：原来用"网格对角线常数"（grid*sqrt(2)≈14.1）
#     归一化近端度，短蛇时所有 body 节 closeness 挤在高位无法区分；
#     改为"按自身长度归一化"，短蛇/长蛇各节相对位置信息清晰可辨。
#  4. K 倍帧率思考（与 test5d 的 FRAME_RATE 语义完全一致，默认 K=5）：
#     - rollout 采样：K 次内部 E-I 迭代，观测逐次衰减 ×decay^k，logits 平均后采样
#     - PPO 更新：从 buffer 的「初始状态」重放 K 次迭代算 new_logits（同口径）
#     - 评估 & 种子基线 & 播放：全部 K=5 统一口径，杜绝"假分数"继承
#  5. 掩码：加载 test5d_best_model.pth 已进化拓扑（M_*）并冻结，只训练权重。
#  6. 截断 BPTT：显式状态（E/I/short_term/hormone/fatigue）逐样本传入，
#     每样本独立反传（重放 K 步时梯度流经 K 次迭代，内存 O(K·B)），可任意 minibatch。
#
# 奖励机制说明（防刷死循环）：
#  - unseen 正奖励仅 +0.01/步，500 步封顶 +5；若长期不吃会触发
#    "steps_without_food > 2*len(body)+STARVE_BIAS(12)" 饿死截断并 -1；
#  - 空转（蛇头 12 步窗口重复 ≥3 次）每步 -0.1，绕圈净亏 → 摒弃；
#  - 最优策略 = 快速吃掉食物（+1） + 积极巡航找到新食物（+0.01 步）。
# ==========================================

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import math
import random
import time
import os
import sys
import collections
import matplotlib.pyplot as plt


# ==========================================
# 0. 全局配置类
# ==========================================
class Config:
    # --- PPO 超参 ---
    N_ENVS = 16                # 向量化并行环境数（替代 2048 个体种群）
    ROLLOUT_LEN = 128          # 每轮采样步数（总样本 = N_ENVS × ROLLOUT_LEN）
    TOTAL_ITERATIONS = 2000    # 训练迭代轮数
    GAMMA = 0.99               # 折扣因子
    GAE_LAMBDA = 0.95          # GAE 平滑系数
    CLIP_EPS = 0.2             # PPO clip 阈值
    VALUE_COEF = 0.5           # 价值损失权重
    ENTROPY_COEF = 0.01        # 熵奖励权重（鼓励探索）
    PPO_EPOCHS = 4             # 每轮数据重复更新次数
    MINIBATCH_SIZE = 256       # mini-batch 大小
    LR = 3e-4                  # Adam 学习率
    LR_DECAY = True            # 学习率线性衰减到 0
    MAX_GRAD_NORM = 0.5        # 梯度裁剪

    # --- K 倍帧率思考（与 test5d 完全一致的推理方式）---
    FRAME_RATE = 5             # 游戏每前进一步，AI 内部更新 K 次（思考）
    INPUT_DECAY = 0.9          # 思考期间外部输入逐次衰减系数

    # --- 每步奖励（核心设计：seen/unseen 区分）---
    # 该步"开始时的观测"决定：
    #   unseen （看不到食物）              → 正奖励，鼓励巡航探索
    #   seen   （看到食物，射线食物信号>0）→ 0.0（低于 unseen 保留"看到更低"
    #            语义，但不再是负值——负值会训练蛇"远离食物"导致空转）
    # 吃食物 +1.0；死亡/饿死/超时 -1.0
    UNSEEN_STEP_REWARD = 0.01
    SEEN_STEP_REWARD = 0.0
    EAT_REWARD = 1.0
    DEATH_REWARD = -1.0

    # --- 空转惩罚（防止蛇原地打转白赚步奖励）---
    # 检测最近 LOITER_WINDOW 步内蛇头位置：同一格出现 ≥ LOITER_REPEAT 次
    # 判定为空转，之后每步额外加 LOITER_PENALTY（-0.1）。
    # 空转 12 步净亏 -1.2，远超 unseen 白赚的 +0.12 → 绕圈变成亏本行为；
    # 正常巡航蛇头持续移动，不重复 → 不受罚。
    LOITER_WINDOW = 12
    LOITER_REPEAT = 3
    LOITER_PENALTY = -0.1

    # --- 饿死截断阈值：steps_without_food > 2*len(body) + STARVE_BIAS ---
    # 收紧：20 → 12（初始蛇 16 步不吃即截断，空转更快终止且折扣优势转负）
    STARVE_BIAS = 12

    # --- 饥饿惩罚（防"绕大圈赚步奖励不追食物"）---
    # 连续没吃食物 ≥ HUNGER_WINDOW 步后，每步额外 penalty：
    #   -HUNGER_STEP_PENALTY × (steps_without_food - HUNGER_WINDOW)
    # 逐级累加：40 步没吃 → -0.05×(40-16) = -1.2，远超 unseen 白赚 +0.4 → 
    # 绕圈变亏本；追食物重置惩罚 + 拿 +1 → 唯一划算路径。
    # 不依赖头位置（大圈 4×4 绕一圈 8~12 步照样被罚），任意圈型无盲区。
    HUNGER_WINDOW = 16
    HUNGER_STEP_PENALTY = 0.05

    # --- 评估 ---
    EVAL_INTERVAL = 20         # 每 N 轮用 argmax 策略评估
    EVAL_EPISODES = 5

    # --- 环境参数 ---
    GRID_SIZE = 10
    MAX_STEPS = 500            # 单局最大步数（超时截断）

    # --- 脑结构参数（与 test5d 完全一致，保证种子可加载）---
    NUM_COLUMNS = 256
    # 3 食物方向 bit + 1 食物距离 + 5 射线×2 + 8 自体感知桶 + 2 尾巴局部坐标 = 24
    OBS_DIM = 24
    ACTION_DIM = 3
    INIT_DENSITY = 0.15
    MASK_FREEZE = True         # 冻结拓扑掩码（保留 test5d 进化成果，只训权重）

    # --- E-I 动力学参数 ---
    BASE_TAU_E = 0.7
    TAU_E_NOISE = 0.1
    TAU_E_MIN = 0.001
    TAU_E_MAX = 2.0
    W_EI = 2.0
    W_IE = 2.0
    W_EI_MIN = 0.0
    W_EI_MAX = 6.0
    W_IE_MIN = 0.0
    W_IE_MAX = 6.0

    # --- 短期 tau 调制 ---
    SHORT_TERM_GAIN = -0.2
    SHORT_TERM_DECAY = 0.3

    # --- 激素系统（默认冻结，输出恒 0；解冻需配合可微软释放）---
    TRAIN_HORMONE_NET = False
    HORMONE_DECAY = 0.95
    EXCIT_HORMONE_GAIN = 0.25
    INHIB_HORMONE_GAIN = 0.50
    EXCIT_DIFFUSION = 0.15
    INHIB_DIFFUSION = 0.40
    HORMONE_NET_HIDDEN = 32
    HORMONE_GATE_THRESHOLD = 0.0

    # --- 动作疲劳参数（纯连续次数，防止蛇反复转圈自杀）---
    FATIGUE_GAIN = 1e-5
    FATIGUE_THRESHOLD = 4
    FATIGUE_MAX = 5.0

    # --- 断点 / 种子 ---
    CHECKPOINT_PATH = 'test6_checkpoint.pth'
    BEST_MODEL_PATH = 'test6_best_model.pth'
    AUTO_RESUME = True
    CHECKPOINT_INTERVAL = 50       # 每 N 轮自动保存断点
    SEED_FROM_TEST5D = True        # 加载 test5d 已进化的拓扑+权重作为初始化
    TEST5D_MODEL_PATH = 'test5d_best_model.pth'


# ==========================================
# 1. 轻量级贪吃蛇环境（射线视野 + 无奖励设计）
#    test6 改动：
#      - step 返回 (next_obs, ate, done, truncated)，done=撞死，truncated=饿死/超时
#      - 八桶自体感知归一化基准由 "网格对角线常数" 改为 "自身长度"
# ==========================================
class SnakeEnv:
    def __init__(self, grid_size=10, max_steps=500, cfg=None):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.cfg = cfg          # 可空；空转/饿死参数从 cfg 读，缺省用默认
        self.reset()

    def reset(self):
        self.head = (self.grid_size // 2, self.grid_size // 2)
        self.dir = random.choice(((0, 1), (1, 0), (0, -1), (-1, 0)))
        self.body = [self.head, (self.head[0] - self.dir[0], self.head[1] - self.dir[1])]
        self._place_food()
        self.food_count = 0
        self.steps = 0
        self.steps_without_food = 0
        # --- 空转检测：最近 LOITER_WINDOW 步蛇头位置环形缓冲 ---
        if self.cfg is not None:
            window = max(2, int(getattr(self.cfg, 'LOITER_WINDOW', 12)))
        else:
            window = 12
        self.head_history = collections.deque(maxlen=window)
        self.loiter_now = False
        return self._get_obs()

    def _place_food(self):
        while True:
            self.food = (random.randint(0, self.grid_size - 1),
                         random.randint(0, self.grid_size - 1))
            if self.food not in self.body:
                break

    def _cast_ray(self, direction, tail_included=True):
        """沿 direction 发射射线，返回 (自由路径长度比, 食物信号)。"""
        max_dist = self.grid_size
        obstacle_dist = max_dist
        food_dist = -1

        body_set = self.body if tail_included else self.body[:-1]

        for step in range(1, max_dist + 1):
            r = self.head[0] + direction[0] * step
            c = self.head[1] + direction[1] * step

            if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
                obstacle_dist = step
                break
            if (r, c) in body_set:
                obstacle_dist = step
                break
            if (r, c) == self.food and food_dist < 0:
                food_dist = step

        free_path_length = obstacle_dist / max_dist
        if food_dist > 0:
            food_signal = 1.0 - (food_dist / max_dist)
        else:
            food_signal = 0.0
        return free_path_length, food_signal

    def _get_obs(self, will_eat=False):
        """构造 24 维观测。

        布局（与 test5d 一致，仅八桶归一化改为自身长度）：
        [0:3]    食物方向 bit（前 / 左前 / 右前）
        [3]      食物距离（欧氏距离，归一化到 [0,1]）
        [4:14]   5 条射线 × 2（自由路径比, 食物信号）
        [14:22]  自体感知：蛇头朝向参考系下 8 个方向桶，
                 每桶 = 该方向最近身体节的"近端度"。
                 ★ test6 改造：用『自身长度』归一化（原为网格对角常数），
                   短蛇时各节 distance/身长 差异明显，长蛇时反映全身占比位置。
        [22:24]  蛇尾局部坐标
        """
        obs = np.zeros(self.obs_dim, dtype=np.float32)

        left_dir = (-self.dir[1], self.dir[0])
        right_dir = (self.dir[1], -self.dir[0])
        dx = self.food[0] - self.head[0]
        dy = self.food[1] - self.head[1]
        if (dx * self.dir[0] + dy * self.dir[1]) > 0:       obs[0] = 1.0
        if (dx * left_dir[0] + dy * left_dir[1]) > 0:       obs[1] = 1.0
        if (dx * right_dir[0] + dy * right_dir[1]) > 0:     obs[2] = 1.0
        obs[3] = min(math.hypot(dx, dy) / (self.grid_size * math.sqrt(2)), 1.0)

        ray_dirs = [
            left_dir,
            (self.dir[0] + left_dir[0], self.dir[1] + left_dir[1]),
            self.dir,
            (self.dir[0] + right_dir[0], self.dir[1] + right_dir[1]),
            right_dir,
        ]
        tail_included = will_eat
        for i, rd in enumerate(ray_dirs):
            free_path, food_sig = self._cast_ray(rd, tail_included=tail_included)
            obs[4 + i * 2]     = free_path
            obs[4 + i * 2 + 1] = food_sig

        # --- 自体感知：8 方向桶（蛇头朝向参考系，逆时针为正 = 左侧）---
        dxh, dyh = self.dir  # 单位方向向量
        # ★ test6 改造：归一化基准 = 自身长度（身体节数），而非网格对角常数
        self_len = max(1.0, float(len(self.body) - 1))   # 身体节数（不含蛇头）
        for seg in self.body[1:]:  # 排除蛇头自身
            wx = seg[0] - self.head[0]
            wy = seg[1] - self.head[1]
            # 旋转到蛇头朝向参考系：x=前方, y=左方
            rot_x = wx * dxh + wy * dyh
            rot_y = -wx * dyh + wy * dxh
            ang = math.degrees(math.atan2(rot_y, rot_x))  # [-180, 180]
            if ang < 0:
                ang += 360.0
            bucket = int((ang + 22.5) // 45) % 8
            closeness = 1.0 - min(math.hypot(wx, wy) / self_len, 1.0)
            if closeness > obs[14 + bucket]:
                obs[14 + bucket] = closeness

        # --- 蛇尾局部坐标（追尾策略的关键线索）---
        tail = self.body[-1]
        wx = tail[0] - self.head[0]
        wy = tail[1] - self.head[1]
        obs[22] = (wx * dxh + wy * dyh) / self.grid_size
        obs[23] = (-wx * dyh + wy * dxh) / self.grid_size

        return obs

    def step(self, action):
        """执行动作。

        返回 (next_obs, ate_food, done, truncated)：
          done     = 撞墙 / 撞到身体（真正死亡，价值引导 0）
          truncated = 饿死（steps_without_food 超时）或超过 MAX_STEPS
                    （蛇还活着但局被截断，价值用 V(next) bootstrap）
        """
        if action == 1:
            self.dir = (-self.dir[1], self.dir[0])
        elif action == 2:
            self.dir = (self.dir[1], -self.dir[0])

        next_head = (self.head[0] + self.dir[0], self.head[1] + self.dir[1])
        self.steps += 1
        self.steps_without_food += 1

        will_eat = (next_head == self.food)
        body_to_check = self.body if will_eat else self.body[:-1]
        if (next_head[0] < 0 or next_head[0] >= self.grid_size or
            next_head[1] < 0 or next_head[1] >= self.grid_size or
            next_head in body_to_check):
            return self._get_obs(will_eat), False, True, False

        self.body.insert(0, next_head)
        self.head = next_head

        # 更新蛇头历史并判定空转（吃食物也会记录，但正常追食路径头不重复）
        self.head_history.append(self.head)
        if self.cfg is not None:
            repeat_thr = max(2, int(getattr(self.cfg, 'LOITER_REPEAT', 3)))
        else:
            repeat_thr = 3
        if len(self.head_history) >= repeat_thr:
            counts = {}
            for h in self.head_history:
                counts[h] = counts.get(h, 0) + 1
            self.loiter_now = max(counts.values()) >= repeat_thr
        else:
            self.loiter_now = False

        ate_food = False
        if self.head == self.food:
            self.food_count += 1
            ate_food = True
            self.steps_without_food = 0
            self._place_food()
        else:
            self.body.pop()

        # 饿死截断：2*len(body) + STARVE_BIAS（收紧，空转更快终止）
        if self.cfg is not None:
            starve_bias = int(getattr(self.cfg, 'STARVE_BIAS', 12))
        else:
            starve_bias = 12
        if self.steps_without_food > 2 * len(self.body) + starve_bias:
            return self._get_obs(), False, False, True

        if self.steps >= self.max_steps:
            return self._get_obs(), ate_food, False, True

        return self._get_obs(), ate_food, False, False

    obs_dim = 24  # 观测维度（类属性常量，与 Config.OBS_DIM 一致）


def _obs_sees_food(obs):
    """判断观测是否"看到食物"（5 条射线任一食物信号 > 0）。

    批量版本：传入 [B, 24] 数组时返回 [B] 布尔向量。
    """
    o = np.asarray(obs)
    if o.ndim == 1:
        return bool(np.any(o[5:14:2] > 0.0))
    return np.any(o[:, 5:14:2] > 0.0, axis=1)


# ==========================================
# 2. E-I 皮质柱脑区模型（PPO 可微版本）
#    test6 改动：
#      - 新增 value head（V, b_v）
#      - forward_ppo：单步批量可微前向（1 步 BPTT）
#      - forward_ppo_k：K 倍帧率思考的可微前向（K 次内部迭代 + logits 平均）
#      - 掩码冻结（不参与梯度）
#      - 激素默认冻结（跳过计算；解冻用可微 softmax 释放）
# ==========================================
class EIBrainRegion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.N = cfg.NUM_COLUMNS
        self.obs_dim = cfg.OBS_DIM
        self.action_dim = cfg.ACTION_DIM
        self.train_hormone = bool(getattr(cfg, 'TRAIN_HORMONE_NET', False))

        # --- 基因型：拓扑掩码（冻结，不参与梯度）---
        self.M_in = (torch.rand(self.N, self.obs_dim) < cfg.INIT_DENSITY).float()
        self.M_rec = (torch.rand(self.N, self.N) < cfg.INIT_DENSITY).float()
        torch.diagonal(self.M_rec).zero_()
        self.M_out = (torch.rand(self.action_dim, self.N) < cfg.INIT_DENSITY).float()

        # --- 表现型：突触权重（可训练）---
        self.W_in = nn.Parameter(torch.randn(self.N, self.obs_dim) * 0.1)
        self.W_rec = nn.Parameter(torch.randn(self.N, self.N) * 0.05)
        self.W_out = nn.Parameter(torch.randn(self.action_dim, self.N) * 0.1)
        self.b_out = nn.Parameter(torch.zeros(self.action_dim))

        # --- 每柱体 tau_e 初始值（可训练）---
        self.tau_e_init = nn.Parameter(
            cfg.BASE_TAU_E + (torch.rand(self.N) * 2 - 1) * cfg.TAU_E_NOISE
        )

        # --- 每柱体 Wei / Wie（可训练）---
        self.w_ei = nn.Parameter(torch.full((self.N,), cfg.W_EI))
        self.w_ie = nn.Parameter(torch.full((self.N,), cfg.W_IE))

        # --- 激素调控前馈网络（默认冻结=全 0，输出恒 0）---
        hormone_input_dim = self.N * 3
        self.W_hormone1 = nn.Parameter(torch.zeros(cfg.HORMONE_NET_HIDDEN, hormone_input_dim))
        self.b_hormone1 = nn.Parameter(torch.zeros(cfg.HORMONE_NET_HIDDEN))
        self.W_excit = nn.Parameter(torch.zeros(self.N, cfg.HORMONE_NET_HIDDEN))
        self.b_excit = nn.Parameter(torch.zeros(self.N))
        self.W_inhib = nn.Parameter(torch.zeros(self.N, cfg.HORMONE_NET_HIDDEN))
        self.b_inhib = nn.Parameter(torch.zeros(self.N))

        # --- ★ value head（PPO critic）---
        self.V = nn.Parameter(torch.zeros(self.N))
        self.b_v = nn.Parameter(torch.zeros(1))

        # --- 归一化扩散矩阵缓存（激素解冻时用；现算也可）---
        self.register_buffer('M_norm', torch.zeros(self.N, self.N))
        self.refresh_cached()

    def refresh_cached(self):
        with torch.no_grad():
            deg = self.M_rec.sum(dim=1, keepdim=True) + 1e-8
            self.M_norm.copy_(self.M_rec / deg)

    def forward_ppo(self, obs_t, E, I, short_term, horm_e, horm_i, counts):
        """单步批量可微前向（1 步 BPTT，状态全部显式传入）。

        参数：
          obs_t     : [B, obs_dim] 观测张量
          E, I      : [B, N] 兴奋/抑制态（必须 detach，来自上一步输出）
          short_term: [B, N] 短期 tau 调制状态（detach）
          horm_e    : [B, N] 兴奋激素浓度（detach）；train_hormone=False 时传 None
          horm_i    : [B, N] 抑制激素浓度（detach）；同上
          counts    : [B, action_dim] 连续动作计数（detach 常数）

        返回 (logits[B,3], value[B], E_next, I_next, short_next, he_next, hi_next)
          E/I/short/hormone 的 next 均以 detach 张量返回，供下一步作为输入。
        计算图的源头只有：obs_t + 可训练参数 → 本步 logits/value。
        """
        E = E.detach()
        I = I.detach()
        if short_term is not None:
            short_term = short_term.detach()
        if horm_e is not None:
            horm_e = horm_e.detach()
        if horm_i is not None:
            horm_i = horm_i.detach()
        counts = counts.detach()

        # 1. 外部与循环输入（掩码现算；掩码不参与梯度，梯度只流向 W）
        ext_in = torch.matmul(obs_t, (self.W_in * self.M_in).T)      # [B, N]
        rec_in = torch.matmul(E, (self.W_rec * self.M_rec).T)        # [B, N]
        total_in = ext_in + rec_in

        # 2. 激素（默认冻结跳过；解冻用可微 softmax 加权释放）
        if self.train_hormone and horm_e is not None:
            cfg = self.cfg
            hormone_input = torch.cat([E, I, total_in], dim=-1)      # [B, 3N]
            h_hidden = torch.relu(torch.matmul(hormone_input, self.W_hormone1.T) + self.b_hormone1)
            excit_logits = torch.matmul(h_hidden, self.W_excit.T) + self.b_excit  # [B, N]
            inhib_logits = torch.matmul(h_hidden, self.W_inhib.T) + self.b_inhib
            gate_thr = float(getattr(cfg, 'HORMONE_GATE_THRESHOLD', 0.0))
            # 可微软释放：softmax 权重 × 门控 × sigmoid(最大 logit)
            excit_w = torch.softmax(excit_logits, dim=-1)
            max_e = excit_logits.max(dim=-1, keepdim=True).values
            gate_e = (max_e > gate_thr).float()
            excit_cmd = excit_w * gate_e * torch.sigmoid(max_e)
            inhib_w = torch.softmax(inhib_logits, dim=-1)
            max_i = inhib_logits.max(dim=-1, keepdim=True).values
            gate_i = (max_i > gate_thr).float()
            inhib_cmd = inhib_w * gate_i * torch.sigmoid(max_i)

            horm_e_next = (1 - cfg.HORMONE_DECAY) * excit_cmd + \
                cfg.HORMONE_DECAY * ((1 - cfg.EXCIT_DIFFUSION) * horm_e +
                                     cfg.EXCIT_DIFFUSION * torch.matmul(horm_e, self.M_norm.T))
            horm_i_next = (1 - cfg.HORMONE_DECAY) * inhib_cmd + \
                cfg.HORMONE_DECAY * ((1 - cfg.INHIB_DIFFUSION) * horm_i +
                                     cfg.INHIB_DIFFUSION * torch.matmul(horm_i, self.M_norm.T))
            horm_e_next = horm_e_next.detach()
            horm_i_next = horm_i_next.detach()
        else:
            horm_e_next = None
            horm_i_next = None
            horm_e = horm_i = None

        # 3. 短期状态（历史 detach，只保留本步梯度）
        if short_term is not None:
            short_next = (self.cfg.SHORT_TERM_DECAY * short_term +
                          (1 - self.cfg.SHORT_TERM_DECAY) * E).detach()
        else:
            short_next = None
            short_term = torch.zeros_like(E)

        # 4. 有效 tau_e
        horm_e_eff = horm_e if horm_e is not None else torch.zeros_like(E)
        horm_i_eff = horm_i if horm_i is not None else torch.zeros_like(E)
        effective_tau_e = self.tau_e_init + \
            self.cfg.SHORT_TERM_GAIN * short_term + \
            self.cfg.EXCIT_HORMONE_GAIN * horm_e_eff - \
            self.cfg.INHIB_HORMONE_GAIN * horm_i_eff
        effective_tau_e = torch.clamp(effective_tau_e, self.cfg.TAU_E_MIN, self.cfg.TAU_E_MAX)

        # 4b. 有效 Wei / Wie（边界保护）
        w_ei_eff = torch.clamp(self.w_ei, self.cfg.W_EI_MIN, self.cfg.W_EI_MAX)   # [N]
        w_ie_eff = torch.clamp(self.w_ie, self.cfg.W_IE_MIN, self.cfg.W_IE_MAX)

        # 5. E-I 离散代数更新（w_ei [N] 广播到 [B, N]）
        E_new = torch.sigmoid(total_in + effective_tau_e * E - w_ei_eff * I)      # [B, N]
        I_new = torch.sigmoid(w_ie_eff * E_new)                                   # [B, N]

        # 6. 动作输出（W_out*M_out 现算；梯度只流向 W_out/b_out）
        action_logits = torch.matmul(E_new, (self.W_out * self.M_out).T) + self.b_out  # [B, 3]

        # 7. 动作疲劳抑制（counts 为 detach 常数，仅作偏置）
        fatigue = torch.relu(counts - self.cfg.FATIGUE_THRESHOLD) * self.cfg.FATIGUE_GAIN
        fatigue = torch.clamp(fatigue, max=self.cfg.FATIGUE_MAX)
        action_logits = action_logits - fatigue

        # 8. value head（critic）
        value = torch.matmul(E_new, self.V.unsqueeze(1)).squeeze(1) + self.b_v   # [B]

        return (action_logits, value, E_new.detach(), I_new.detach(),
                short_next, horm_e_next, horm_i_next)


def forward_ppo_k(brain, obs_t, E, I, short_term, horm_e, horm_i, counts,
                  K=None, decay=None):
    """K 倍帧率思考的可微前向（与 test5d 的 deliberate_action 语义一致）。

    - 观测逐次衰减：obs_t * (decay^k)，E/I/short/hormone 跨 K 次迭代连续传递
    - logits 取 K 次平均（avg_logits）；value 取最后一次（critic 用最终状态估值）
    - 返回 (avg_logits[B,3], value[B], E_f, I_f, short_f, he_f, hi_f)，
      迭代后的 final 状态供下一步推进（已 detach，与 forward_ppo 语义一致）

    计算图包含 K 次迭代；用作训练前向时梯度会流经整个 K 步思考链
    （内存 O(K·B)，K 默认 5）。
    """
    cfg = brain.cfg
    if K is None:
        K = max(1, int(getattr(cfg, 'FRAME_RATE', 1)))
    if decay is None:
        decay = float(getattr(cfg, 'INPUT_DECAY', 0.9))

    logits_sum = None
    E_f, I_f = E, I
    short_f = short_term
    he_f, hi_f = horm_e, horm_i
    last_value = None

    for k in range(K):
        scaled_obs = obs_t * (decay ** k)
        logits, last_value, E_f, I_f, short_f, he_f, hi_f = brain.forward_ppo(
            scaled_obs, E_f, I_f, short_f, he_f, hi_f, counts)
        if logits_sum is None:
            logits_sum = logits
        else:
            logits_sum = logits_sum + logits

    avg_logits = logits_sum / K
    return avg_logits, last_value, E_f, I_f, short_f, he_f, hi_f


def update_counts(counts, action):
    """更新连续动作计数（detach 纯 buffer 操作，不影响梯度）。

    counts : [B, 3] 当前计数
    action : [B] long 采样动作（或 [B,1]）
    返回   : [B, 3] 更新后的计数（选定动作 +1，其余清零）
    """
    counts = counts.detach()
    if action.dim() == 2 and action.shape[1] == 1:
        act_idx = action.squeeze(1)      # [B]
    else:
        act_idx = action
    cur = counts.gather(1, act_idx.unsqueeze(1)).squeeze(1) + 1.0
    new = torch.zeros_like(counts)
    new.scatter_(1, act_idx.unsqueeze(1), cur.unsqueeze(1))
    return new


def make_zero_states(brain, batch=1):
    """创建初始零状态（每局/每 env 起点）。"""
    N = brain.N
    E = torch.zeros(batch, N)
    I = torch.zeros(batch, N)
    short = torch.zeros(batch, N)
    if brain.train_hormone:
        he = torch.zeros(batch, N)
        hi = torch.zeros(batch, N)
    else:
        he = None
        hi = None
    counts = torch.zeros(batch, 3)
    return E, I, short, he, hi, counts


def trainable_parameters(brain, cfg):
    """返回参与 PPO 梯度更新的参数列表（掩码冻结，激素按开关）。"""
    params = [brain.W_in, brain.W_rec, brain.W_out, brain.b_out,
              brain.tau_e_init, brain.w_ei, brain.w_ie,
              brain.V, brain.b_v]
    if bool(getattr(cfg, 'TRAIN_HORMONE_NET', False)):
        params += [brain.W_hormone1, brain.b_hormone1,
                   brain.W_excit, brain.b_excit,
                   brain.W_inhib, brain.b_inhib]
    return params


# ==========================================
# 3. 检查点 / 断点 / 种子（与 test5d 格式兼容）
# ==========================================
def _config_dict(cfg):
    merged = {}
    merged.update(vars(cfg.__class__))
    merged.update(vars(cfg))
    return {k: v for k, v in merged.items() if not k.startswith('__')}


def save_brain_state(brain, use_half=True):
    """打包遗传属性为可序列化 dict（test5d 格式 + value head）。"""
    dtype = torch.float16 if use_half else torch.float32
    with torch.no_grad():
        return {
            'N': brain.N,
            'M_in': brain.M_in.to(torch.uint8),
            'M_rec': brain.M_rec.to(torch.uint8),
            'M_out': brain.M_out.to(torch.uint8),
            'W_in': brain.W_in.data.to(dtype),
            'W_rec': brain.W_rec.data.to(dtype),
            'W_out': brain.W_out.data.to(dtype),
            'b_out': brain.b_out.data.to(dtype),
            'tau_e_init': brain.tau_e_init.data.to(dtype),
            'w_ei': brain.w_ei.data.to(dtype),
            'w_ie': brain.w_ie.data.to(dtype),
            'W_hormone1': brain.W_hormone1.data.to(dtype),
            'b_hormone1': brain.b_hormone1.data.to(dtype),
            'W_excit': brain.W_excit.data.to(dtype),
            'b_excit': brain.b_excit.data.to(dtype),
            'W_inhib': brain.W_inhib.data.to(dtype),
            'b_inhib': brain.b_inhib.data.to(dtype),
            # test6 新增：value head
            'V': brain.V.data.to(dtype),
            'b_v': brain.b_v.data.to(dtype),
        }


def load_brain_state(state, cfg):
    """从 dict 重建 EIBrainRegion（test5d 格式兼容，缺 V/b_v 时默认零）。"""
    new = EIBrainRegion.__new__(EIBrainRegion)
    nn.Module.__init__(new)

    new.cfg = cfg
    new.N = int(state['N'])
    new.obs_dim = cfg.OBS_DIM
    new.action_dim = cfg.ACTION_DIM
    new.train_hormone = bool(getattr(cfg, 'TRAIN_HORMONE_NET', False))

    new.M_in = state['M_in'].float()
    new.M_rec = state['M_rec'].float()
    new.M_out = state['M_out'].float()

    new.W_in = nn.Parameter(state['W_in'].float())
    new.W_rec = nn.Parameter(state['W_rec'].float())
    new.W_out = nn.Parameter(state['W_out'].float())
    new.b_out = nn.Parameter(state['b_out'].float())
    new.tau_e_init = nn.Parameter(state['tau_e_init'].float())
    new.w_ei = nn.Parameter(state['w_ei'].float())
    new.w_ie = nn.Parameter(state['w_ie'].float())

    new.W_hormone1 = nn.Parameter(state['W_hormone1'].float())
    new.b_hormone1 = nn.Parameter(state['b_hormone1'].float())
    new.W_excit = nn.Parameter(state['W_excit'].float())
    new.b_excit = nn.Parameter(state['b_excit'].float())
    new.W_inhib = nn.Parameter(state['W_inhib'].float())
    new.b_inhib = nn.Parameter(state['b_inhib'].float())

    # test6：value head（旧 test5d 模型无此字段 → 默认零初始化）
    new.V = nn.Parameter(state.get('V', torch.zeros(new.N)).float())
    new.b_v = nn.Parameter(state.get('b_v', torch.zeros(1)).float())

    new.register_buffer('M_norm', torch.zeros(new.N, new.N))
    new.refresh_cached()
    return new


def save_best_model(path, brain, cfg, food, steps):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    torch.save({
        'brain': save_brain_state(brain, use_half=False),
        'food': float(food),
        'steps': float(steps),
        'config': _config_dict(cfg),
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }, path)


def load_best_model_brain(path, cfg):
    """从最优模型文件恢复 (brain, food, steps)；不兼容时返回 None。"""
    if not os.path.exists(path):
        return None
    try:
        data = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"警告: 最优模型 {path} 读取失败 ({e})，已忽略种子")
        return None

    saved_cfg = data.get('config', {})
    if saved_cfg:
        if (saved_cfg.get('NUM_COLUMNS') != cfg.NUM_COLUMNS or
                saved_cfg.get('OBS_DIM') != cfg.OBS_DIM or
                saved_cfg.get('ACTION_DIM') != cfg.ACTION_DIM):
            print(f"警告: 最优模型 {path} 与当前配置不匹配 (N/OBS/ACTION_DIM)，已忽略种子")
            return None

    brain = load_brain_state(data['brain'], cfg)
    return brain, float(data.get('food', -1.0)), float(data.get('steps', 0.0))


def save_checkpoint(path, cfg, next_iter, brain, optimizer, history,
                    best_brain, best_food, best_steps):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    payload = {
        'next_iter': int(next_iter),
        'brain': save_brain_state(brain, use_half=False),
        'optimizer': optimizer.state_dict(),
        'history': history,
        'best_brain': save_brain_state(best_brain, use_half=False) if best_brain is not None else None,
        'best_food': float(best_food),
        'best_steps': float(best_steps),
        'config': _config_dict(cfg),
        'random_state': random.getstate(),
        'torch_rng_state': torch.get_rng_state(),
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    tmp_path = path + '.tmp'
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    print(f"  [Checkpoint] 断点已保存 -> {path} (next_iter={next_iter})")


def load_checkpoint(path, cfg):
    """读取断点；文件不存在或配置不兼容时返回 None。"""
    if not os.path.exists(path):
        return None

    data = torch.load(path, map_location='cpu', weights_only=False)
    saved_cfg = data.get('config', {})
    if saved_cfg:
        if (saved_cfg.get('NUM_COLUMNS') != cfg.NUM_COLUMNS or
                saved_cfg.get('OBS_DIM') != cfg.OBS_DIM or
                saved_cfg.get('ACTION_DIM') != cfg.ACTION_DIM):
            print(f"警告: 断点 {path} 与当前配置不匹配 (N/OBS/ACTION_DIM)，已忽略")
            return None

    brain = load_brain_state(data['brain'], cfg)
    return {
        'next_iter': int(data['next_iter']),
        'brain': brain,
        'optimizer': data.get('optimizer'),
        'history': data.get('history', {}),
        'best_brain': (load_brain_state(data['best_brain'], cfg)
                       if data.get('best_brain') is not None else None),
        'best_food': float(data.get('best_food', -1.0)),
        'best_steps': float(data.get('best_steps', 0.0)),
    }


# ==========================================
# 4. 评估 / GAE / PPO 更新
# ==========================================
def evaluate(brain, env, cfg, episodes=5):
    """argmax 贪心评估（K=FRAME_RATE 思考）。返回 (avg_food, avg_steps)。"""
    brain.eval()
    total_foods = []
    total_steps = []
    with torch.no_grad():
        for _ in range(episodes):
            E, I, short, he, hi, counts = make_zero_states(brain, 1)
            obs = env.reset()
            ep_food = 0
            steps = 0
            done = truncated = False
            while not done and not truncated and steps < cfg.MAX_STEPS:
                obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                logits, _, E, I, short, he, hi = forward_ppo_k(
                    brain, obs_t, E, I, short, he, hi, counts)
                action = int(logits.squeeze(0).argmax().item())
                counts = update_counts(counts, torch.tensor([[action]]))
                obs, ate, done, truncated = env.step(action)
                if ate:
                    ep_food += 1
                steps += 1
            total_foods.append(ep_food)
            total_steps.append(steps)
    brain.train()
    return float(np.mean(total_foods)), float(np.mean(total_steps))


def evaluate_k1_probe(brain, env, cfg, episodes=5):
    """K=1 单次思考评估（诊断用：对比 K=FRAME_RATE 的差异）。
    仅用于理解思考带来的效果，不参与训练/选择。
    """
    brain.eval()
    total_foods = []
    with torch.no_grad():
        for _ in range(episodes):
            E, I, short, he, hi, counts = make_zero_states(brain, 1)
            obs = env.reset()
            ep_food = 0
            steps = 0
            done = truncated = False
            while not done and not truncated and steps < cfg.MAX_STEPS:
                obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                logits, _, E, I, short, he, hi = forward_ppo_k(
                    brain, obs_t, E, I, short, he, hi, counts, K=1, decay=1.0)
                action = int(logits.squeeze(0).argmax().item())
                counts = update_counts(counts, torch.tensor([[action]]))
                obs, ate, done, truncated = env.step(action)
                if ate:
                    ep_food += 1
                steps += 1
            total_foods.append(ep_food)
    brain.train()
    return float(np.mean(total_foods))


def compute_gae(rew_buf, val_buf, mask_buf, last_val, cfg):
    """GAE 优势估计。

    rew_buf/val_buf/mask_buf: [T, B]
    last_val: [B] —— rollout 末尾状态的 value（no_grad）
    mask=0 表示该步真死亡（boot 0）；mask=1 表示可继续（含 truncated，boot value）
    """
    T, B = rew_buf.shape
    adv = torch.zeros_like(rew_buf)
    gae = 0.0
    # 最后一步的 boot：若最后一步死亡（mask[T-1]=0）则 boot=0，否则 boot=last_val
    next_val = last_val * mask_buf[T - 1]
    for t in reversed(range(T)):
        if t == T - 1:
            nv = next_val
        else:
            nv = val_buf[t + 1]
        delta = rew_buf[t] + cfg.GAMMA * nv * mask_buf[t] - val_buf[t]
        gae = delta + cfg.GAMMA * cfg.GAE_LAMBDA * mask_buf[t] * gae
        adv[t] = gae
    ret = adv + val_buf
    return adv, ret


def ppo_update(brain, optimizer, cfg,
               obs_buf, act_buf, logp_buf, adv_buf, ret_buf,
               E0_buf, I0_buf, short0_buf, he0_buf, hi0_buf, counts0_buf):
    """PPO clip 更新（K=FRAME_RATE 思考重放）。

    每个样本都以 buffer 中的「初始状态」为起点，用 forward_ppo_k 重放
    K 次内部迭代得到当前策略的 avg_logits → new_logp → ratio，
    与 rollout 采样时的思考口径完全一致（观测同样逐次衰减 ×decay^k）。
    返回 (policy_loss, value_loss, entropy) 均值。
    """
    T, B, obs_dim = obs_buf.shape
    N = brain.N
    n = T * B
    K = max(1, int(getattr(cfg, 'FRAME_RATE', 1)))
    decay = float(getattr(cfg, 'INPUT_DECAY', 0.9))

    obs_flat = obs_buf.reshape(n, obs_dim)
    act_flat = act_buf.reshape(n)
    logp_flat = logp_buf.reshape(n)
    adv_flat = adv_buf.reshape(n)
    ret_flat = ret_buf.reshape(n)

    E0_flat = E0_buf.reshape(n, N)
    I0_flat = I0_buf.reshape(n, N)
    short0_flat = short0_buf.reshape(n, N)
    counts0_flat = counts0_buf.reshape(n, 3)
    he0_flat = he0_buf.reshape(n, N) if he0_buf is not None else None
    hi0_flat = hi0_buf.reshape(n, N) if hi0_buf is not None else None

    # 优势归一化（稳定训练）
    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    idx = np.arange(n)
    mb_size = int(cfg.MINIBATCH_SIZE)
    losses = {'policy': [], 'value': [], 'entropy': []}

    for _ in range(cfg.PPO_EPOCHS):
        random.shuffle(idx)
        for s in range(0, n, mb_size):
            mb = idx[s:s + mb_size]
            mb_t = torch.from_numpy(mb)

            logits, val, *_ = forward_ppo_k(
                brain, obs_flat[mb_t],
                E0_flat[mb_t], I0_flat[mb_t], short0_flat[mb_t],
                he0_flat[mb_t] if he0_flat is not None else None,
                hi0_flat[mb_t] if hi0_flat is not None else None,
                counts0_flat[mb_t],
                K=K, decay=decay)

            dist = Categorical(logits=logits)
            new_logp = dist.log_prob(act_flat[mb_t])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logp - logp_flat[mb_t])
            adv_mb = adv_flat[mb_t]
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1.0 - cfg.CLIP_EPS, 1.0 + cfg.CLIP_EPS) * adv_mb
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (val - ret_flat[mb_t]).pow(2).mean()

            loss = policy_loss + cfg.VALUE_COEF * value_loss - cfg.ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in trainable_parameters(brain, cfg) if p.grad is not None],
                cfg.MAX_GRAD_NORM)
            optimizer.step()

            # 参数边界保护（数据级 clamp，不影响图）
            with torch.no_grad():
                brain.tau_e_init.data.clamp_(cfg.TAU_E_MIN, cfg.TAU_E_MAX)
                brain.w_ei.data.clamp_(cfg.W_EI_MIN, cfg.W_EI_MAX)
                brain.w_ie.data.clamp_(cfg.W_IE_MIN, cfg.W_IE_MAX)

            losses['policy'].append(float(policy_loss.item()))
            losses['value'].append(float(value_loss.item()))
            losses['entropy'].append(float(entropy.item()))

    return (float(np.mean(losses['policy'])), float(np.mean(losses['value'])),
            float(np.mean(losses['entropy'])))


# ==========================================
# 5. 可视化
# ==========================================
def render_snake_game(env):
    grid = np.zeros((env.grid_size, env.grid_size, 3))
    grid[env.food[0], env.food[1]] = [1, 1, 0]
    for seg in env.body:
        if 0 <= seg[0] < env.grid_size and 0 <= seg[1] < env.grid_size:
            grid[seg[0], seg[1]] = [0, 0, 1]
    if 0 <= env.head[0] < env.grid_size and 0 <= env.head[1] < env.grid_size:
        grid[env.head[0], env.head[1]] = [1, 0, 0]
    return grid


def plot_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    iters = history.get('iter', [])

    axes[0, 0].plot(iters, history.get('mean_rew', []), color='blue', marker='.', markersize=3)
    axes[0, 0].set_title("Mean Episode Reward")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(iters, history.get('mean_len', []), color='green', marker='.', markersize=3)
    axes[0, 1].set_title("Mean Episode Length")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].grid(True, alpha=0.3)

    # eval_food 每 EVAL_INTERVAL 轮才记录一次，x 轴需用对应的 eval_iter
    eval_food = history.get('eval_food', [])
    eval_iters = history.get('eval_iter', [])
    if len(eval_iters) != len(eval_food):
        # 旧断点无 eval_iter 字段时兜底：按记录顺序递增
        eval_iters = list(range(len(eval_food)))
    axes[1, 0].plot(eval_iters, eval_food, color='red', marker='o', markersize=4)
    axes[1, 0].set_title("Best Eval Food (avg over episodes)")
    axes[1, 0].set_xlabel("Iteration (every EVAL_INTERVAL)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(iters, history.get('policy_loss', []), label='policy', color='purple', alpha=0.7)
    axes[1, 1].plot(iters, history.get('value_loss', []), label='value', color='orange', alpha=0.7)
    axes[1, 1].plot(iters, history.get('entropy', []), label='entropy', color='brown', alpha=0.7)
    axes[1, 1].set_title("PPO Losses")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_best_brain_play(brain, cfg, max_steps=300):
    env = SnakeEnv(grid_size=cfg.GRID_SIZE, max_steps=max_steps)
    E, I, short, he, hi, counts = make_zero_states(brain, 1)
    obs = env.reset()

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    img = ax.imshow(render_snake_game(env), cmap='viridis', vmin=0, vmax=1)
    ax.set_title("Best Brain Playing Snake (PPO)")
    ax.axis('off')

    steps = 0
    done = truncated = False
    with torch.no_grad():
        while not done and not truncated and steps < max_steps:
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
            logits, _, E, I, short, he, hi = forward_ppo_k(
                brain, obs_t, E, I, short, he, hi, counts)
            action = int(logits.squeeze(0).argmax().item())
            counts = update_counts(counts, torch.tensor([[action]]))

            obs, ate, done, truncated = env.step(action)
            steps += 1

            img.set_data(render_snake_game(env))
            ax.set_title(f"Step: {steps} | Score: {len(env.body) - 2}")
            fig.canvas.draw_idle()
            plt.pause(0.05)

    print(f"\nGame Over! Final Score: {len(env.body) - 2} | Survived Steps: {steps}")
    plt.ioff()
    plt.show()


# ==========================================
# 6. 主训练循环
# ==========================================
def run_training(cfg):
    """PPO 主训练循环（K 倍帧率思考，全程统一口径）。

    - rollout：no_grad 下 forward_ppo_k 采样（K=FRAME_RATE 思考）
    - buffer   ：存每步「初始状态」（E0/I0/short0/counts0）供 PPO 重放
    - ppo_update：从初始状态重放 K 次迭代算 new_logp（与 rollout 同口径）
    - 评估/种子基线：K=FRAME_RATE 统一口径
    """
    t_program = time.perf_counter()

    # ---- 断点续训 ----
    start_iter = 0
    brain = None
    optimizer = None
    ckpt = None
    history = {'iter': [], 'mean_rew': [], 'mean_len': [],
               'eval_food': [], 'eval_iter': [],
               'policy_loss': [], 'value_loss': [], 'entropy': []}
    best_brain = None
    best_food = -1.0
    best_steps = 0.0

    if cfg.AUTO_RESUME and os.path.exists(cfg.CHECKPOINT_PATH):
        ckpt = load_checkpoint(cfg.CHECKPOINT_PATH, cfg)
        if ckpt is not None:
            start_iter = ckpt['next_iter']
            brain = ckpt['brain']
            history = ckpt['history']
            best_brain = ckpt['best_brain']
            best_food = ckpt['best_food']
            best_steps = ckpt['best_steps']
            print(f"\n=== 检测到断点 [{cfg.CHECKPOINT_PATH}] ===")
            print(f"  {cfg.TOTAL_ITERATIONS} 轮中已完成 {start_iter} 轮 -> 从第 {start_iter} 轮接续 | "
                  f"历史最优: Food={best_food:.1f}")

    # ---- 新建或种子加载 ----
    if brain is None:
        print("=== test6: PPO 强化学习（K 倍帧率思考）===")
        print(f"  N_ENVS={cfg.N_ENVS}, ROLLOUT_LEN={cfg.ROLLOUT_LEN}, "
              f"TOTAL_ITERATIONS={cfg.TOTAL_ITERATIONS}")
        print(f"  K=FRAME_RATE={cfg.FRAME_RATE}, INPUT_DECAY={cfg.INPUT_DECAY}")
        print(f"  Step reward: unseen(+{cfg.UNSEEN_STEP_REWARD}) / "
              f"seen({cfg.SEEN_STEP_REWARD}) / eat(+{cfg.EAT_REWARD}) / "
              f"death({cfg.DEATH_REWARD})")
        print(f"  Self-bucket normalization: by body length")

        if cfg.SEED_FROM_TEST5D:
            seed_result = load_best_model_brain(cfg.TEST5D_MODEL_PATH, cfg)
            if seed_result is not None:
                seed, seed_food, seed_steps = seed_result
                brain = seed
                # ★ 种子基线用 test6 自身口径（K=FRAME_RATE）重评：
                #   test5d 的 food 是用 K=5 + 旧八桶编码评出的，不能直接继承。
                #   重评得到诚实基线，PPO 一旦超过真实分数就更新 Best。
                eval_env_seed = SnakeEnv(grid_size=cfg.GRID_SIZE, max_steps=cfg.MAX_STEPS)
                base_food, base_steps = evaluate(brain, eval_env_seed, cfg,
                                                 episodes=max(cfg.EVAL_EPISODES, 5))
                best_food = base_food
                best_steps = base_steps
                # 深拷贝快照（EIBrainRegion 无 clone，用 save/load 重建独立实例）
                best_brain = load_brain_state(save_brain_state(brain, use_half=False), cfg)
                print(f"  [Seed] 已加载 test5d 模型 {cfg.TEST5D_MODEL_PATH} "
                      f"(文件标注 Food={seed_food:.1f}，K=5+旧编码口径；不可直接继承)")
                print(f"  [Seed] 用 test6 K={cfg.FRAME_RATE} 口径重评基线: "
                      f"Food={base_food:.1f}, Steps={base_steps:.1f}")
                print(f"  [Seed] 拓扑掩码冻结={cfg.MASK_FREEZE}，仅训练权重/tau/wei/wie/value")
            else:
                print(f"  [Seed] 未找到可用 test5d 模型，随机初始化")
                brain = EIBrainRegion(cfg)
        else:
            brain = EIBrainRegion(cfg)

    assert brain is not None, "brain 初始化失败"

    # ---- 优化器 ----
    if optimizer is None:
        optimizer = torch.optim.Adam(trainable_parameters(brain, cfg), lr=cfg.LR)
        if ckpt is not None and ckpt.get('optimizer') is not None:
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
            except Exception as e:
                print(f"警告: 优化器状态加载失败 ({e})，已重新初始化")

    # ---- 环境与状态 ----
    B = cfg.N_ENVS
    N = brain.N
    T = cfg.ROLLOUT_LEN
    envs = [SnakeEnv(grid_size=cfg.GRID_SIZE, max_steps=cfg.MAX_STEPS, cfg=cfg) for _ in range(B)]
    obs_stack = np.stack([e.reset() for e in envs], axis=0).astype(np.float32)  # [B, 24]
    E, I, short, he, hi, counts = make_zero_states(brain, B)

    def _make_buffers():
        """每步存「初始状态 + 观测 + 动作/logp/value/奖励/mask」。
        初始状态（E0/I0/short0/counts0）供 ppo_update 以相同初始条件
        重放 K 次思考，得到与 rollout 同口径的 new_logp。
        """
        obs_buf = torch.zeros(T, B, cfg.OBS_DIM, dtype=torch.float32)
        act_buf = torch.zeros(T, B, dtype=torch.long)
        logp_buf = torch.zeros(T, B, dtype=torch.float32)
        val_buf = torch.zeros(T, B, dtype=torch.float32)
        rew_buf = torch.zeros(T, B, dtype=torch.float32)
        mask_buf = torch.zeros(T, B, dtype=torch.float32)
        E0_buf = torch.zeros(T, B, N, dtype=torch.float32)
        I0_buf = torch.zeros(T, B, N, dtype=torch.float32)
        short0_buf = torch.zeros(T, B, N, dtype=torch.float32)
        counts0_buf = torch.zeros(T, B, 3, dtype=torch.float32)
        if brain.train_hormone:
            he0_buf = torch.zeros(T, B, N, dtype=torch.float32)
            hi0_buf = torch.zeros(T, B, N, dtype=torch.float32)
        else:
            he0_buf = None
            hi0_buf = None
        return (obs_buf, act_buf, logp_buf, val_buf, rew_buf, mask_buf,
                E0_buf, I0_buf, short0_buf, he0_buf, hi0_buf, counts0_buf)

    (obs_buf, act_buf, logp_buf, val_buf, rew_buf, mask_buf,
     E0_buf, I0_buf, short0_buf, he0_buf, hi0_buf, counts0_buf) = _make_buffers()

    # ---- 主循环 ----
    cur_iter = None
    try:
        for it in range(start_iter, cfg.TOTAL_ITERATIONS):
            cur_iter = it
            t_iter = time.perf_counter()

            # ---- Rollout：收集 T×B 步轨迹（K 倍帧率思考采样）----
            ep_rew = np.zeros(B, dtype=np.float64)
            ep_len = np.zeros(B, dtype=np.int64)
            ep_infos = []

            for t in range(T):
                obs_t = torch.from_numpy(obs_stack)
                # 记录本步初始状态（供 ppo_update 重放 K 次思考）
                E0_buf[t] = E
                I0_buf[t] = I
                short0_buf[t] = short
                if he0_buf is not None:
                    he0_buf[t] = he
                    hi0_buf[t] = hi
                counts0_buf[t] = counts

                # rollout 不需要梯度：旧 logp 与旧 value 存储时都 detach，
                # 避免 ppo_update 多次 backward 引用已释放的 rollout 计算图。
                # K 倍帧率思考：内部迭代 K 次，观测逐次衰减，logits 平均后采样。
                with torch.no_grad():
                    logits, val, E, I, short, he, hi = forward_ppo_k(
                        brain, obs_t, E, I, short, he, hi, counts)

                dist = Categorical(logits=logits)
                act = dist.sample()
                logp = dist.log_prob(act)

                obs_buf[t] = obs_t
                act_buf[t] = act
                logp_buf[t] = logp.detach()
                val_buf[t] = val.detach()

                # ---- 每步奖励：seen/unseen 区分（该步开始时的观测决定）----
                seen_mask = _obs_sees_food(obs_stack)          # [B] bool
                step_rew = np.where(
                    seen_mask, cfg.SEEN_STEP_REWARD, cfg.UNSEEN_STEP_REWARD)  # [B]

                # ---- 执行动作（疲劳计数更新在动作采样后；E/I/short 已是思考后状态）----
                counts = update_counts(counts, act)

                act_np = act.numpy()
                next_obs = np.zeros_like(obs_stack)
                for i in range(B):
                    a = int(act_np[i])
                    o2, ate, done, truncated = envs[i].step(a)
                    r = float(step_rew[i])
                    # ★ 空转检测：蛇头在窗口内重复 ≥ 阈值 → 每步额外惩罚
                    if envs[i].loiter_now:
                        r += cfg.LOITER_PENALTY
                    # ★ 饥饿惩罚：连续没吃 ≥ HUNGER_WINDOW 步后逐级累加
                    #（大转圈、食物在视野不追都被无盲区覆盖）
                    hunger = envs[i].steps_without_food - cfg.HUNGER_WINDOW
                    if hunger > 0:
                        r -= cfg.HUNGER_STEP_PENALTY * float(hunger)
                    if ate:
                        r += cfg.EAT_REWARD
                    if done or truncated:
                        r += cfg.DEATH_REWARD
                    next_obs[i] = o2
                    rew_buf[t, i] = r
                    mask_buf[t, i] = 0.0 if done else 1.0

                    ep_rew[i] += r
                    ep_len[i] += 1

                    if done or truncated:
                        ep_infos.append((float(ep_rew[i]), int(ep_len[i])))
                        # 重置该 env 的状态
                        next_obs[i] = envs[i].reset()
                        E[i].zero_()
                        I[i].zero_()
                        short[i].zero_()
                        if he0_buf is not None:
                            he[i].zero_()
                            hi[i].zero_()
                        counts[i].zero_()
                        ep_rew[i] = 0.0
                        ep_len[i] = 0

                obs_stack = next_obs

            # ---- 末尾 boot value ----
            with torch.no_grad():
                obs_t = torch.from_numpy(obs_stack)
                _, last_val, _, _, _, _, _ = forward_ppo_k(
                    brain, obs_t, E, I, short, he, hi, counts)

            # ---- GAE ----
            adv_buf, ret_buf = compute_gae(rew_buf, val_buf, mask_buf, last_val, cfg)

            # ---- PPO 更新（学习率线性衰减）----
            if cfg.LR_DECAY:
                progress = (it + 1) / cfg.TOTAL_ITERATIONS
                for g in optimizer.param_groups:
                    g['lr'] = cfg.LR * (1.0 - 0.9 * progress)

            p_loss, v_loss, ent = ppo_update(
                brain, optimizer, cfg,
                obs_buf, act_buf, logp_buf, adv_buf, ret_buf,
                E0_buf, I0_buf, short0_buf, he0_buf, hi0_buf, counts0_buf)

            # ---- 统计 ----
            mean_rew = float(np.mean([e[0] for e in ep_infos])) if ep_infos else 0.0
            mean_len = float(np.mean([e[1] for e in ep_infos])) if ep_infos else 0.0
            history['iter'].append(it)
            history['mean_rew'].append(mean_rew)
            history['mean_len'].append(mean_len)
            history['policy_loss'].append(p_loss)
            history['value_loss'].append(v_loss)
            history['entropy'].append(ent)

            iter_time = time.perf_counter() - t_iter
            print(f"Iter {it + 1}/{cfg.TOTAL_ITERATIONS} | "
                  f"MeanRew: {mean_rew:6.2f} | MeanLen: {mean_len:5.1f} | "
                  f"PolicyL: {p_loss:.4f} | ValL: {v_loss:.4f} | Ent: {ent:.4f} | "
                  f"{iter_time:.1f}s",
                  end="")

            # ---- 周期评估（K=FRAME_RATE 统一口径）----
            if (it + 1) % cfg.EVAL_INTERVAL == 0:
                eval_env = SnakeEnv(grid_size=cfg.GRID_SIZE, max_steps=cfg.MAX_STEPS)
                eval_food, eval_steps = evaluate(brain, eval_env, cfg, episodes=cfg.EVAL_EPISODES)
                history['eval_food'].append(eval_food)
                history['eval_iter'].append(it)
                if eval_food > best_food:
                    best_food = eval_food
                    best_steps = eval_steps
                    # 深拷贝快照：拷贝全部参数与掩码
                    st = save_brain_state(brain, use_half=False)
                    best_brain = load_brain_state(st, cfg)
                    save_best_model(cfg.BEST_MODEL_PATH, brain, cfg, best_food, best_steps)
                print(f" | EvalFood: {eval_food:.1f} (Best: {best_food:.1f})", end="")

            print()

            # ---- 周期断点 ----
            if (it + 1) % cfg.CHECKPOINT_INTERVAL == 0:
                save_checkpoint(cfg.CHECKPOINT_PATH, cfg, it + 1, brain, optimizer,
                                history, best_brain, best_food, best_steps)

    except KeyboardInterrupt:
        print("\n训练被中断 (Ctrl+C)，正在保存断点以供下次自动接续...")
        save_checkpoint(cfg.CHECKPOINT_PATH, cfg, cur_iter, brain, optimizer,
                        history, best_brain, best_food, best_steps)
        print(f"断点已保存: {cfg.CHECKPOINT_PATH} (下次运行将从第 {cur_iter} 轮接续)")
        sys.exit(0)

    # ---- 训练完成 ----
    print(f"\nTotal runtime: {time.perf_counter() - t_program:.1f}s")

    if best_brain is None:
        best_brain = brain
    if best_food < 0:
        eval_env = SnakeEnv(grid_size=cfg.GRID_SIZE, max_steps=cfg.MAX_STEPS)
        best_food, best_steps = evaluate(best_brain, eval_env, cfg,
                                         episodes=max(cfg.EVAL_EPISODES, 5))
    save_best_model(cfg.BEST_MODEL_PATH, best_brain, cfg, best_food, best_steps)
    print(f"最优模型已保存: {cfg.BEST_MODEL_PATH} (Food={best_food:.1f}, Steps={best_steps:.1f})")

    if os.path.exists(cfg.CHECKPOINT_PATH):
        os.remove(cfg.CHECKPOINT_PATH)
        print(f"训练已完成，已删除临时断点: {cfg.CHECKPOINT_PATH}")

    plot_history(history)

    print("\n--- Best Brain Summary ---")
    print(f"Input connections active:   {best_brain.M_in.sum().item():.0f}/{best_brain.N * best_brain.obs_dim}")
    print(f"Internal connections active: {best_brain.M_rec.sum().item():.0f}/{best_brain.N * best_brain.N}")
    print(f"Output connections active:   {best_brain.M_out.sum().item():.0f}/{best_brain.action_dim * best_brain.N}")
    print(f"tau_e range: [{best_brain.tau_e_init.min().item():.3f}, {best_brain.tau_e_init.max().item():.3f}]")
    print(f"Wei range:   [{best_brain.w_ei.min().item():.3f}, {best_brain.w_ei.max().item():.3f}]")
    print(f"Wie range:   [{best_brain.w_ie.min().item():.3f}, {best_brain.w_ie.max().item():.3f}]")

    print("\nLaunching visualization for the best individual...")
    visualize_best_brain_play(best_brain, cfg, max_steps=300)


if __name__ == "__main__":
    run_training(Config())