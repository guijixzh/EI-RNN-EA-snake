
"""
test5a_lunar.py
===============
将 test5a.py 的方法体系（E-I 皮质柱脑区 + 交替冻结进化 + K 倍帧率思考 +
两阶段筛选 + 多进程并行评估 + 断点续训）应用于 test3.py 的 LunarLander-v3 任务。

相对 test5a.py 的主要改动：
  1. 环境：SnakeEnv -> gymnasium "LunarLander-v3"（OBS_DIM=8, ACTION_DIM=4）
  2. K（FRAME_RATE）= 1：每次游戏步 AI 仅做 1 次内部更新
  3. 移除动作疲劳机制（consecutive_counts / update_fatigue / forward 疲劳抑制）
  4. 评估指标：食物数 -> 平均奖励 (avg_reward, avg_steps)
  5. 网络规模缩小：NUM_COLUMNS=64（先验证可跑通，之后再扩大）
  6. 种群规模缩小：POP_SIZE=512, EVAL_EPISODES=3, GENERATIONS=100
  7. 检查点 / 最优模型路径独立：test5a_lunar_checkpoint.pth / test5a_lunar_best_model.pth
"""

import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import time
import os
import sys
import multiprocessing as mp
import networkx as nx
import matplotlib.colors as mcolors
import gymnasium as gym


# ==========================================
# 0. 全局配置类（所有重要参数集中管理）
# ==========================================
class Config:
    # --- 进化参数 ---
    POP_SIZE = 2048                     # test5a 为 2048，缩小以便先跑通
    GENERATIONS = 100
    ELITE_SIZE = 64                    # 精英数 = POP_SIZE // 8
    MUT_RATE = 0.05
    TOPOLOGY_MUT_PROB = 0.05
    WEIGHT_MUT_FRAC = 0.2
    WEIGHT_MUT_STD = 0.1
    TAU_E_MUT_STD = 0.05
    HORMONE_MUT_FRAC = 0.1
    HORMONE_MUT_STD = 0.05

    # --- 动态进化率（test5 系新增：抑制训练后期震荡）---
    # 结构组 G1（拓扑/权重）采用 cos 余弦衰减；动力学 G2 与激素 G3 采用指数衰减，
    # 均按 progress = gen / (GENERATIONS-1) 计算，前期大范围探索、后期精细收敛。
    EXP_DECAY = 2.5                # 指数衰减率：末期 f ≈ e^-2.5 ≈ 8%
    MUT_MIN = 0.005                # 所有动态变异率的下限，防止完全停摆

    # --- 防退化软淘汰（test5"单侧转弯判死"的 LunarLander 等价物）---
    DEGENERATE_ACTION_RATIO = 0.9     # 单一动作占比超过该比例
    DEGENERATE_REWARD_THRESHOLD = -190 # 且平均奖励低于该值时触发退化惩罚
    DEGENERATE_PENALTY = 100.0        # 触发后退化惩罚量：从 reward 扣减，
                                      # 使单策略个体显著劣于多样性个体（软淘汰），
                                      # 同时保留真实 steps，避免 BestSteps 被 99999 污染

    # --- 环境参数 ---
    ENV_NAME = 'LunarLander-v3'        # test3.py 使用的环境
    EVAL_EPISODES = 10                 # 全量评估局数，大幅降低评估方差
    MAX_STEPS = 500                    # 低于 LunarLander 默认 1000 上限

    # --- 脑结构参数 ---
    NUM_COLUMNS = 256                   # test5a 为 256，先验证可跑通
    # LunarLander 观测：位置x/y + 速度vx/vy + 角度 + 角速度 + 腿接触(2) = 8
    OBS_DIM = 8
    ACTION_DIM = 4                     # 0=无操作 1=左推进 2=主推进 3=右推进
    INIT_DENSITY = 0.15

    # --- E-I 动力学参数 ---
    BASE_TAU_E = 0.7
    TAU_E_NOISE = 0.1
    TAU_E_MIN = 0.001
    TAU_E_MAX = 2.0
    W_EI = 2.0
    W_IE = 2.0
    # Wei / Wie 进化范围与变异强度（逐柱体可进化参数，与 tau_e 同类）
    W_EI_MUT_STD = 0.1
    W_IE_MUT_STD = 0.1
    W_EI_MIN = 0.0
    W_EI_MAX = 6.0
    W_IE_MIN = 0.0
    W_IE_MAX = 6.0

    # --- 短期 tau 调制 ---
    SHORT_TERM_GAIN = -0.2
    SHORT_TERM_DECAY = 0.3

    # --- 激素系统参数 ---
    HORMONE_DECAY = 0.95
    EXCIT_HORMONE_GAIN = 0.25
    INHIB_HORMONE_GAIN = 0.50
    EXCIT_DIFFUSION = 0.15
    INHIB_DIFFUSION = 0.40
    HORMONE_NET_HIDDEN = 32

    # --- K 倍帧率思考（帧数倍率）---
    FRAME_RATE = 1                     # K=1：每次游戏步 AI 仅内部更新 1 次
    INPUT_DECAY = 0.9                  # 思考期间外部输入逐次衰减系数（K=1 时无实际影响）

    # --- 加速训练配置 ---
    # A 方案：多进程并行评估
    PARALLEL_EVAL = True               # 总开关；设为 False 则回退到串行评估
    NUM_WORKERS = 0                    # 并行进程数；0 = 自动取 os.cpu_count()
    # C 方案：两阶段快速筛选（初筛 SCREEN_EPISODES 局 → 前 K 名精评至满局）
    SCREEN_ENABLE = False              # 关闭筛选：全量 10 局评估，与 test5 一致，最稳定
    SCREEN_EPISODES = 2                # 初筛局数（仅 SCREEN_ENABLE=True 时生效）
    SCREEN_MULTIPLIER = 3              # 精评覆盖倍数：K = ELITE_SIZE × SCREEN_MULTIPLIER
    SCREEN_AUTO_FALLBACK = True        # 初筛种群整体过弱时自动回退全量评估

    # --- 检查点 / 断点续训 / 最优模型 / 种子继承（本代码专属路径）---
    CHECKPOINT_PATH = 'test5a_lunar_checkpoint.pth'
    BEST_MODEL_PATH = 'test5a_lunar_best_model.pth'
    CHECKPOINT_FALLBACK = None
    BEST_MODEL_FALLBACK = None
    AUTO_RESUME = True
    CHECKPOINT_INTERVAL = 5
    SEED_FROM_BEST = True

    # --- B 方案：交替冻结进化（现改为 all 全量激活，与 test5_fast 一致）---
    # 个体参数按功能耦合拆为三组；FREEZE_SCHEME='all' 时每组每代都交叉+变异。
    #   G1 结构组：拓扑掩码 M_* + 突触权重 W_*/b_out
    #   G2 动力学组：tau_e_init + w_ei + w_ie
    #   G3 激素组：W_hormone1/b_hormone1/W_excit/b_excit/W_inhib/b_inhib
    FREEZE_SCHEME = 'all'
    CYCLE_PATTERN = [('G2',), ('G1',), ('G2', 'G3'), ('G1',), ('G3',)]
    G1_INTERVAL = 3      # soft 备选
    G2_INTERVAL = 1      # soft 备选
    G3_INTERVAL = 5      # soft 备选


# ==========================================
# 1. E-I 皮质柱脑区模型
#    （tau_e / Wei / Wie 均进化为逐柱体参数 + 激素调控；已移除疲劳机制）
# ==========================================
class EIBrainRegion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.N = cfg.NUM_COLUMNS
        self.obs_dim = cfg.OBS_DIM
        self.action_dim = cfg.ACTION_DIM

        # --- 基因型：拓扑掩码 ---
        self.M_in = (torch.rand(self.N, self.obs_dim) < cfg.INIT_DENSITY).float()
        self.M_rec = (torch.rand(self.N, self.N) < cfg.INIT_DENSITY).float()
        torch.diagonal(self.M_rec).zero_()
        self.M_out = (torch.rand(self.action_dim, self.N) < cfg.INIT_DENSITY).float()

        # --- 表现型：突触权重 ---
        self.W_in = nn.Parameter(torch.randn(self.N, self.obs_dim) * 0.1)
        self.W_rec = nn.Parameter(torch.randn(self.N, self.N) * 0.05)
        self.W_out = nn.Parameter(torch.randn(self.action_dim, self.N) * 0.1)
        self.b_out = nn.Parameter(torch.zeros(self.action_dim))  # 输出偏置项

        # --- 每柱体 tau_e 初始值 ---
        self.tau_e_init = nn.Parameter(
            cfg.BASE_TAU_E + (torch.rand(self.N) * 2 - 1) * cfg.TAU_E_NOISE
        )

        # --- 每柱体 Wei / Wie（E-I 耦合强度，可进化）---
        self.w_ei = nn.Parameter(torch.full((self.N,), cfg.W_EI))
        self.w_ie = nn.Parameter(torch.full((self.N,), cfg.W_IE))

        # --- 激素调控前馈网络 ---
        hormone_input_dim = self.N * 3
        self.W_hormone1 = nn.Parameter(torch.zeros(cfg.HORMONE_NET_HIDDEN, hormone_input_dim))
        self.b_hormone1 = nn.Parameter(torch.zeros(cfg.HORMONE_NET_HIDDEN))
        self.W_excit = nn.Parameter(torch.zeros(self.N, cfg.HORMONE_NET_HIDDEN))
        self.b_excit = nn.Parameter(torch.zeros(self.N))
        self.W_inhib = nn.Parameter(torch.zeros(self.N, cfg.HORMONE_NET_HIDDEN))
        self.b_inhib = nn.Parameter(torch.zeros(self.N))

        # --- 运行时状态（非进化参数；疲劳计数器已移除）---
        self.register_buffer('hormone_excit', torch.zeros(self.N))
        self.register_buffer('hormone_inhib', torch.zeros(self.N))
        self.register_buffer('short_term_state', torch.zeros(self.N))

        self.baseline = None

        # --- 缓存：掩码权重与归一化扩散矩阵（forward 中复用） ---
        self.register_buffer('W_rec_eff', torch.zeros(self.N, self.N))
        self.register_buffer('W_out_eff', torch.zeros(self.action_dim, self.N))
        self.register_buffer('M_norm', torch.zeros(self.N, self.N))
        self.refresh_cached()

    def reset_runtime(self):
        """每局开始前重置激素与短期状态"""
        self.hormone_excit.zero_()
        self.hormone_inhib.zero_()
        self.short_term_state.zero_()

    def forward(self, obs_t, E_prev, I_prev):
        # 1. 外部与循环输入
        ext_in = torch.matmul(self.W_in * self.M_in, obs_t)
        rec_in = torch.matmul(self.W_rec_eff, E_prev)
        total_in = ext_in + rec_in

        # 2. 激素调控前馈网络
        hormone_input = torch.cat([E_prev, I_prev, total_in], dim=-1)
        h_hidden = torch.relu(torch.matmul(self.W_hormone1, hormone_input) + self.b_hormone1)
        excit_cmd = torch.relu(torch.matmul(self.W_excit, h_hidden) + self.b_excit)
        inhib_cmd = torch.relu(torch.matmul(self.W_inhib, h_hidden) + self.b_inhib)

        # 3. 激素沿拓扑扩散 + 长期衰减（M_norm 缓存）
        self.hormone_excit = (1 - self.cfg.HORMONE_DECAY) * excit_cmd + \
            self.cfg.HORMONE_DECAY * (
                (1 - self.cfg.EXCIT_DIFFUSION) * self.hormone_excit +
                self.cfg.EXCIT_DIFFUSION * torch.matmul(self.M_norm, self.hormone_excit)
            )

        self.hormone_inhib = (1 - self.cfg.HORMONE_DECAY) * inhib_cmd + \
            self.cfg.HORMONE_DECAY * (
                (1 - self.cfg.INHIB_DIFFUSION) * self.hormone_inhib +
                self.cfg.INHIB_DIFFUSION * torch.matmul(self.M_norm, self.hormone_inhib)
            )

        # 4. 短期状态
        self.short_term_state = self.cfg.SHORT_TERM_DECAY * self.short_term_state + \
            (1 - self.cfg.SHORT_TERM_DECAY) * E_prev

        # 5. 有效 tau_e
        effective_tau_e = self.tau_e_init + \
            self.cfg.SHORT_TERM_GAIN * self.short_term_state + \
            self.cfg.EXCIT_HORMONE_GAIN * self.hormone_excit - \
            self.cfg.INHIB_HORMONE_GAIN * self.hormone_inhib
        effective_tau_e = torch.clamp(effective_tau_e, self.cfg.TAU_E_MIN, self.cfg.TAU_E_MAX)

        # 5b. 有效 Wei / Wie（进化参数，运行时仅做边界保护）
        w_ei_eff = torch.clamp(self.w_ei, self.cfg.W_EI_MIN, self.cfg.W_EI_MAX)
        w_ie_eff = torch.clamp(self.w_ie, self.cfg.W_IE_MIN, self.cfg.W_IE_MAX)

        # 6. E-I 离散代数更新
        E_new = torch.sigmoid(total_in + effective_tau_e * E_prev - w_ei_eff * I_prev)
        I_new = torch.sigmoid(w_ie_eff * E_new)

        # 7. 动作输出 (添加偏置, W_out_eff 缓存；疲劳机制已移除)
        action_logits = torch.matmul(self.W_out_eff, E_new) + self.b_out

        return action_logits, E_new, I_new

    def refresh_cached(self):
        """刷新前向缓存：W_rec*M_rec、W_out*M_out 与归一化扩散矩阵 M_norm。"""
        with torch.no_grad():
            self.W_rec_eff.copy_(self.W_rec.data * self.M_rec)
            self.W_out_eff.copy_(self.W_out.data * self.M_out)
            deg = self.M_rec.sum(dim=1, keepdim=True) + 1e-8
            self.M_norm.copy_(self.M_rec / deg)

    def save_genetic_baseline(self):
        """保存遗传基线（零拷贝优化）。"""
        self.baseline = {
            'W_in': self.W_in.data,
            'W_rec': self.W_rec.data,
            'W_out': self.W_out.data,
            'b_out': self.b_out.data,
            'M_in': self.M_in.clone(), 'M_rec': self.M_rec.clone(), 'M_out': self.M_out.clone(),
            'tau_e_init': self.tau_e_init.data,
            'w_ei': self.w_ei.data,
            'w_ie': self.w_ie.data,
            'W_hormone1': self.W_hormone1.data, 'b_hormone1': self.b_hormone1.data,
            'W_excit': self.W_excit.data, 'b_excit': self.b_excit.data,
            'W_inhib': self.W_inhib.data, 'b_inhib': self.b_inhib.data,
        }

    def clone(self):
        """轻量克隆：只复制遗传基因（权重/掩码）与基线，远快于 copy.deepcopy。"""
        new = EIBrainRegion.__new__(EIBrainRegion)
        nn.Module.__init__(new)

        new.cfg = self.cfg
        new.N = self.N
        new.obs_dim = self.obs_dim
        new.action_dim = self.action_dim

        # 基因型掩码
        new.M_in = self.M_in.clone()
        new.M_rec = self.M_rec.clone()
        new.M_out = self.M_out.clone()

        # 表现型权重
        new.W_in = nn.Parameter(self.W_in.data.clone())
        new.W_rec = nn.Parameter(self.W_rec.data.clone())
        new.W_out = nn.Parameter(self.W_out.data.clone())
        new.b_out = nn.Parameter(self.b_out.data.clone())
        new.tau_e_init = nn.Parameter(self.tau_e_init.data.clone())

        # 每柱体 Wei / Wie
        new.w_ei = nn.Parameter(self.w_ei.data.clone())
        new.w_ie = nn.Parameter(self.w_ie.data.clone())

        # 激素调控网络
        new.W_hormone1 = nn.Parameter(self.W_hormone1.data.clone())
        new.b_hormone1 = nn.Parameter(self.b_hormone1.data.clone())
        new.W_excit = nn.Parameter(self.W_excit.data.clone())
        new.b_excit = nn.Parameter(self.b_excit.data.clone())
        new.W_inhib = nn.Parameter(self.W_inhib.data.clone())
        new.b_inhib = nn.Parameter(self.b_inhib.data.clone())

        # 运行时状态（置零）
        new.register_buffer('hormone_excit', torch.zeros(self.N))
        new.register_buffer('hormone_inhib', torch.zeros(self.N))
        new.register_buffer('short_term_state', torch.zeros(self.N))
        new.register_buffer('W_rec_eff', torch.zeros(self.N, self.N))
        new.register_buffer('W_out_eff', torch.zeros(self.action_dim, self.N))
        new.register_buffer('M_norm', torch.zeros(self.N, self.N))

        new.baseline = self.baseline  # 引用共享（安全，见 test5a docstring）

        new.refresh_cached()
        return new

    def restore_genetic_baseline(self):
        if self.baseline:
            self.W_in.data = self.baseline['W_in'].clone()
            self.W_rec.data = self.baseline['W_rec'].clone()
            self.W_out.data = self.baseline['W_out'].clone()
            self.b_out.data = self.baseline['b_out'].clone()
            self.M_in = self.baseline['M_in'].clone()
            self.M_rec = self.baseline['M_rec'].clone()
            self.M_out = self.baseline['M_out'].clone()
            self.tau_e_init.data = self.baseline['tau_e_init'].clone()
            self.w_ei.data = self.baseline['w_ei'].clone()
            self.w_ie.data = self.baseline['w_ie'].clone()
            self.W_hormone1.data = self.baseline['W_hormone1'].clone()
            self.b_hormone1.data = self.baseline['b_hormone1'].clone()
            self.W_excit.data = self.baseline['W_excit'].clone()
            self.b_excit.data = self.baseline['b_excit'].clone()
            self.W_inhib.data = self.baseline['W_inhib'].clone()
            self.b_inhib.data = self.baseline['b_inhib'].clone()
            self.refresh_cached()


# ==========================================
# 1b. K 倍帧率思考：AI 与游戏环境的交互接口
#      （K=1：每次游戏步仅 1 次内部更新）
# ==========================================
def deliberate_action(brain, obs, E, I, K=None, decay=None):
    """游戏环境每前进一步，AI 在固定观测上做 K 次内部更新后给出动作。

    K 默认取 cfg.FRAME_RATE（本脚本为 1）。
    """
    cfg = brain.cfg
    if K is None:
        K = cfg.FRAME_RATE
    if decay is None:
        decay = cfg.INPUT_DECAY

    obs_np = np.asarray(obs, dtype=np.float32)
    scales = [decay ** k for k in range(K)]
    logits_sum = None
    with torch.no_grad():
        for k in range(K):
            obs_t = torch.from_numpy(obs_np * scales[k])
            logits, E, I = brain(obs_t, E, I)
            if logits_sum is None:
                logits_sum = logits
            else:
                logits_sum = logits_sum + logits

    avg_logits = logits_sum / K
    action = torch.argmax(avg_logits).item()
    return action, avg_logits, E, I


# ==========================================
# 2. 评估与进化逻辑
# ==========================================
def evaluate_individual(brain, env, render=False, max_steps=None, episodes=None):
    """评估单个个体指定局数（LunarLander：平均奖励 + 平均存活步数）。

    - episodes=None 时取 cfg.EVAL_EPISODES。
    - 防退化软淘汰（test5"单侧转弯判死"的 LunarLander 等价物）：
      统计全部局汇总的动作分布，若单一动作占比 > DEGENERATE_ACTION_RATIO
      且平均奖励 < DEGENERATE_REWARD_THRESHOLD，则从 avg_reward 扣减
      DEGENERATE_PENALTY（软淘汰），抑制"一直主推进 / 一直无操作"
      这类单一策略爆炸式增长导致的伪高分震荡，同时保留真实 avg_steps。
    """
    cfg = brain.cfg
    if max_steps is None:
        max_steps = cfg.MAX_STEPS
    if episodes is None:
        episodes = cfg.EVAL_EPISODES

    total_rewards = []
    total_steps_list = []
    total_action_counts = [0] * brain.action_dim   # 全部局汇总的动作计数

    for ep in range(episodes):
        brain.reset_runtime()
        obs, _ = env.reset()
        E = torch.zeros(brain.N)
        I = torch.zeros(brain.N)

        ep_reward = 0.0
        steps = 0
        done = False

        while not done and steps < max_steps:
            # K 倍帧率思考：内部更新 K 次后给出实际动作（K=1）
            action, avg_logits, E, I = deliberate_action(brain, obs, E, I)
            total_action_counts[action] += 1

            next_obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            obs = next_obs
            steps += 1

        total_rewards.append(ep_reward)
        total_steps_list.append(steps)

    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps_list)

    # --- 防退化软淘汰：单一动作占比过高 + 平均奖励为负 -> 惩罚 reward ---
    # 惩罚式替代（而非 -1e9/99999 判死）：从 avg_reward 扣减 DEGENERATE_PENALTY，
    # 让"一直单动作"的退化个体 reward 显著低于多样性个体，自然退出最优/精英池
    # （软淘汰），同时保留真实 avg_steps，避免 BestSteps 被 99999 污染。
    total_actions = sum(total_action_counts)
    if total_actions > 0:
        max_ratio = max(total_action_counts) / total_actions
        deg_ratio = float(getattr(cfg, 'DEGENERATE_ACTION_RATIO', 0.9))
        deg_reward = float(getattr(cfg, 'DEGENERATE_REWARD_THRESHOLD', 0.0))
        if max_ratio > deg_ratio and avg_reward < deg_reward:
            avg_reward = avg_reward - float(getattr(cfg, 'DEGENERATE_PENALTY', 100.0))

    if render:
        print(f"  [Render] Reward: {avg_reward:.1f}, Steps: {avg_steps:.1f}")

    return avg_reward, avg_steps


def _freeze_active_groups(gen, cfg):
    """按 FREEZE_SCHEME 计算第 gen 代激活的参数组集合（同 test5a）。"""
    scheme = getattr(cfg, 'FREEZE_SCHEME', 'all')
    if scheme == 'all':
        return frozenset({'G1', 'G2', 'G3'})
    if scheme == 'hard':
        return frozenset({('G1', 'G2', 'G3')[gen % 3]})
    if scheme == 'cycle':
        pattern = getattr(cfg, 'CYCLE_PATTERN', [('G1', 'G2', 'G3')])
        return frozenset(pattern[gen % len(pattern)])
    active = set()
    if gen % max(1, int(getattr(cfg, 'G1_INTERVAL', 3))) == 0:
        active.add('G1')
    if gen % max(1, int(getattr(cfg, 'G2_INTERVAL', 1))) == 0:
        active.add('G2')
    if gen % max(1, int(getattr(cfg, 'G3_INTERVAL', 5))) == 0:
        active.add('G3')
    return frozenset(active)


def _dynamic_mut_rates(gen, cfg):
    """动态进化率：按 progress = gen/(GENERATIONS-1) 计算各组变异强度。

    - G1 结构组（拓扑掩码翻转 + 突触权重）采用 cos 余弦衰减：
      f(p) = 0.5*(1+cos(pi*p))，p: 0->1 单调降，训练后期结构趋于稳定。
    - G2 动力学组（tau_e/w_ei/w_ie）与 G3 激素组采用指数衰减：
      f(p) = exp(-EXP_DECAY*p)，后期保留约 e^-2.5 ≈ 8% 的精细微调。
    - 所有强度取下限 MUT_MIN，防止训练末期完全停摆。
    """
    total = max(1, cfg.GENERATIONS - 1)
    p = min(1.0, max(0.0, gen / total))

    cos_f = 0.5 * (1.0 + math.cos(math.pi * p))
    exp_f = math.exp(-float(getattr(cfg, 'EXP_DECAY', 2.5)) * p)
    mmin = float(getattr(cfg, 'MUT_MIN', 0.005))

    return {
        'topo_prob': max(mmin, cfg.TOPOLOGY_MUT_PROB * cos_f),  # G1 拓扑变异触发概率（cos）
        'topo_rate': max(mmin, cfg.MUT_RATE * cos_f),           # G1 拓扑边翻转比例（cos）
        'w_std':     max(mmin, cfg.WEIGHT_MUT_STD * cos_f),     # G1 权重高斯噪声标准差（cos）
        'w_frac':    max(mmin, cfg.WEIGHT_MUT_FRAC * cos_f),    # G1 权重噪声掩码比例（cos）
        'tau_std':   max(mmin, cfg.TAU_E_MUT_STD * exp_f),      # G2 tau_e 变异强度（指数）
        'wei_std':   max(mmin, cfg.W_EI_MUT_STD * exp_f),       # G2 Wei 变异强度（指数）
        'wie_std':   max(mmin, cfg.W_IE_MUT_STD * exp_f),       # G2 Wie 变异强度（指数）
        'horm_std':  max(mmin, cfg.HORMONE_MUT_STD * exp_f),    # G3 激素权重噪声（指数）
        'horm_frac': max(mmin, cfg.HORMONE_MUT_FRAC * exp_f),   # G3 激素掩码比例（指数）
    }


def evolve_topology(population, metrics_list, cfg, gen=0, best_idx=None):
    """进化下一代（全量激活 + 动态进化率 + 最优个体强保护）。

    - FREEZE_SCHEME='all'：G1/G2/G3 每代全部参与交叉与变异（与 test5_fast 一致）。
    - 动态进化率：见 _dynamic_mut_rates，前期大范围探索、后期精细收敛。
    - best_idx 非 None 时，将本代最优个体 population[best_idx] 的
      无变异克隆显式置于下一代种群第 0 位，保证其基因完全不变地跨代存活。
    """
    active = _freeze_active_groups(gen, cfg)
    has_g1 = 'G1' in active
    has_g2 = 'G2' in active
    has_g3 = 'G3' in active
    rates = _dynamic_mut_rates(gen, cfg)

    sorted_indices = sorted(
        range(len(metrics_list)),
        key=lambda i: (metrics_list[i][0], -metrics_list[i][1]),
        reverse=True
    )

    elite_idx = sorted_indices[:cfg.ELITE_SIZE]
    elites = [population[i] for i in elite_idx]   # 原个体引用（供交叉采样）

    # ---- 精英强保护：最优个体完全不变地进入下一代第 0 位 ----
    new_pop = []
    protected = set()
    if best_idx is not None:
        new_pop.append(population[best_idx].clone())
        protected.add(int(best_idx))
    # 其余精英照旧克隆保留（跳过已被保护的最优个体，避免重复）
    for i in elite_idx:
        if int(i) not in protected:
            new_pop.append(population[i].clone())
    new_pop = new_pop[:cfg.ELITE_SIZE]

    while len(new_pop) < len(population):
        p1, p2 = random.sample(elites, 2)
        child = p1.clone()
        N = child.N

        with torch.no_grad():
            # ---- G1 交叉：结构组（掩码 + 权重 + 输出偏置）----
            if has_g1:
                col_mask = torch.rand(N) > 0.5
                row_mask = col_mask.unsqueeze(1)
                col_mask_2d = col_mask.unsqueeze(0)
                same_p1 = row_mask & col_mask_2d
                same_p2 = (~row_mask) & (~col_mask_2d)

                child.W_in.data = torch.where(col_mask.unsqueeze(1), p1.W_in.data, p2.W_in.data)
                child.M_in = torch.where(col_mask.unsqueeze(1), p1.M_in, p2.M_in)

                child.W_rec.data = torch.where(same_p1, p1.W_rec.data,
                                      torch.where(same_p2, p2.W_rec.data,
                                          torch.where(torch.rand_like(p1.W_rec.data) > 0.5,
                                                      p1.W_rec.data, p2.W_rec.data)))
                child.M_rec = torch.where(same_p1, p1.M_rec,
                                 torch.where(same_p2, p2.M_rec,
                                     torch.where(torch.rand_like(p1.M_rec) > 0.5,
                                                 p1.M_rec, p2.M_rec)))

                child.W_out.data = torch.where(col_mask.unsqueeze(0), p1.W_out.data, p2.W_out.data)
                child.M_out = torch.where(col_mask.unsqueeze(0), p1.M_out, p2.M_out)

                # 输出偏置归 G1：与 W_out 同步混合
                mask_out = torch.rand_like(p1.b_out.data) > 0.5
                child.b_out.data = torch.where(mask_out, p1.b_out.data, p2.b_out.data)

            # ---- G2 交叉：动力学组（tau_e / Wei / Wie）----
            if has_g2:
                col_mask2 = torch.rand(N) > 0.5
                child.tau_e_init.data = torch.where(col_mask2, p1.tau_e_init.data, p2.tau_e_init.data)
                child.w_ei.data = torch.where(col_mask2, p1.w_ei.data, p2.w_ei.data)
                child.w_ie.data = torch.where(col_mask2, p1.w_ie.data, p2.w_ie.data)

            # ---- G3 交叉：激素调控网络 ----
            if has_g3:
                for attr in ['W_hormone1', 'b_hormone1', 'W_excit', 'b_excit', 'W_inhib', 'b_inhib']:
                    p1_t = getattr(p1, attr).data
                    p2_t = getattr(p2, attr).data
                    mask = torch.rand_like(p1_t) > 0.5
                    getattr(child, attr).data = torch.where(mask, p1_t, p2_t)

        # ---- 变异（逐组独立触发，冻结组一律跳过；强度采用动态进化率）----
        with torch.no_grad():
            if has_g1:
                if random.random() < rates['topo_prob']:
                    m_attr = random.choice(['M_in', 'M_rec', 'M_out'])
                    m_tensor = getattr(child, m_attr)
                    mut_mask = torch.rand_like(m_tensor) < rates['topo_rate']
                    m_tensor[mut_mask] = 1.0 - m_tensor[mut_mask]

                for attr in ['W_in', 'W_rec', 'W_out', 'b_out']:
                    w_tensor = getattr(child, attr).data
                    noise = torch.randn_like(w_tensor) * rates['w_std']
                    noise_mask = torch.rand_like(w_tensor) < rates['w_frac']
                    setattr(child, attr, nn.Parameter(w_tensor + noise * noise_mask))

            if has_g2:
                tau_noise = torch.randn_like(child.tau_e_init.data) * rates['tau_std']
                child.tau_e_init.data = torch.clamp(
                    child.tau_e_init.data + tau_noise,
                    cfg.TAU_E_MIN, cfg.TAU_E_MAX
                )

                w_ei_noise = torch.randn_like(child.w_ei.data) * rates['wei_std']
                child.w_ei.data = torch.clamp(
                    child.w_ei.data + w_ei_noise,
                    cfg.W_EI_MIN, cfg.W_EI_MAX
                )
                w_ie_noise = torch.randn_like(child.w_ie.data) * rates['wie_std']
                child.w_ie.data = torch.clamp(
                    child.w_ie.data + w_ie_noise,
                    cfg.W_IE_MIN, cfg.W_IE_MAX
                )

            if has_g3:
                for attr in ['W_hormone1', 'b_hormone1', 'W_excit', 'b_excit', 'W_inhib', 'b_inhib']:
                    w = getattr(child, attr).data
                    noise = torch.randn_like(w) * rates['horm_std']
                    mask = torch.rand_like(w) < rates['horm_frac']
                    getattr(child, attr).data = w + noise * mask

        child.refresh_cached()
        child.save_genetic_baseline()
        new_pop.append(child)

    return new_pop


# ==========================================
# 3. 可视化
# ==========================================
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history['gen'], history['best_reward'], label='Best Reward', color='red', marker='o', markersize=3)
    ax1.plot(history['gen'], history['avg_reward'], label='Avg Reward', color='blue', alpha=0.6)
    # LunarLander 大于 200 算解决（test3.py 及格线）
    ax1.axhline(y=200, color='green', linestyle='--', label='Solved Threshold (200)')
    ax1.set_title("Evolution Progress — Reward (Primary Criterion)")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Average Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['gen'], history['best_steps'], label='Best Steps', color='green', marker='s', markersize=3)
    ax2.set_title("Best Individual Steps")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Steps Survived")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_best_brain_play(brain, cfg, max_steps=500):
    """在 LunarLander-v3 中实时渲染最优脑区的操控表现。"""
    env = gym.make(cfg.ENV_NAME, render_mode="human")
    brain.restore_genetic_baseline()
    brain.reset_runtime()

    obs, _ = env.reset()
    E = torch.zeros(brain.N)
    I = torch.zeros(brain.N)

    total_reward = 0.0
    steps = 0
    done = False

    while not done and steps < max_steps:
        # K 倍帧率思考：内部更新 K 次后给出实际动作（K=1）
        action, avg_logits, E, I = deliberate_action(brain, obs, E, I)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        obs = next_obs
        steps += 1

    print(f"\nGame Over! Total Reward: {total_reward:.1f} | Survived Steps: {steps}")
    env.close()


try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False
    print("Warning: python-louvain not installed. Using basic community detection.")


# ==========================================
# 3b. 拓扑可视化 - 方案 A/B（更直观的连接展示）
# ==========================================
def _norm_edge_weights(w):
    """将权重绝对值归一化到 [0,1]，用于线宽/透明度映射"""
    w = np.abs(w)
    wmin, wmax = w.min(), w.max()
    if wmax > wmin:
        return (w - wmin) / (wmax - wmin)
    return np.zeros_like(w)


def _community_order(brain, partition):
    """按社区分组、组内按 tau_e 排序，返回柱节点排列顺序与分组边界"""
    tau = brain.tau_e_init.detach().numpy()
    comm_of_col = np.array([partition.get(f"Col_{i}", -1) for i in range(brain.N)], dtype=int)

    cols_by_comm = {}
    for i in range(brain.N):
        cols_by_comm.setdefault(int(comm_of_col[i]), []).append(i)
    comms = sorted(cols_by_comm.keys(), key=lambda c: -len(cols_by_comm[c]))

    order = []
    boundaries = []
    for c in comms:
        cols = sorted(cols_by_comm[c], key=lambda i: tau[i])
        boundaries.append(len(order))
        order.extend(cols)
    return np.array(order), comm_of_col, comms, np.array(boundaries)


def plot_topology_layered(brain, cfg, partition):
    """方案 A：三层流水线拓扑图（输入层 → 柱层 → 输出层）"""
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch, Patch, Rectangle
    from matplotlib.lines import Line2D

    N = brain.N
    tau = brain.tau_e_init.detach().numpy()
    order, comm_of_col, comms, boundaries = _community_order(brain, partition)

    W_rec_np = brain.W_rec.detach().numpy()
    M_rec_np = brain.M_rec.numpy()

    # --- 1. 社区锚点：按 2D 网格排列 ---
    n_comms = len(comms)
    ncols = max(1, int(np.ceil(np.sqrt(n_comms))))
    nrows = max(1, int(np.ceil(n_comms / ncols)))
    anchor_step_x = 0.90
    anchor_step_y = 0.90
    anchor_x0 = 1.30
    total_height = nrows * anchor_step_y
    anchor = {}
    for idx_blk, comm_id in enumerate(comms):
        r = idx_blk // ncols
        c = idx_blk % ncols
        anchor[int(comm_id)] = (anchor_x0 + c * anchor_step_x,
                                total_height - (r + 0.5) * anchor_step_y)

    # --- 2. 块内 spring_layout ---
    block_radius = 0.34 * anchor_step_x
    MIN_NODE_DIST = 0.12
    pos_col = {}
    for idx_blk, comm_id in enumerate(comms):
        start = int(boundaries[idx_blk])
        end = int(boundaries[idx_blk + 1]) if idx_blk + 1 < len(boundaries) else N
        block = [int(col) for col in order[start:end]]
        if len(block) == 1:
            sub_pos = {block[0]: (0.0, 0.0)}
        else:
            sub = nx.DiGraph()
            for col in block:
                sub.add_node(col)
            for i in block:
                for j in block:
                    if M_rec_np[i, j] > 0:
                        sub.add_edge(j, i, weight=abs(W_rec_np[i, j]))
            seed = 42 + int(comm_id)
            try:
                sub_pos = nx.spring_layout(sub, k=0.6, iterations=150, seed=seed)
            except Exception:
                sub_pos = {col: (random.uniform(-1, 1), random.uniform(-1, 1))
                           for col in block}
        xs = [p[0] for p in sub_pos.values()]
        ys = [p[1] for p in sub_pos.values()]
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        span = max(float(np.max(xs) - np.min(xs)),
                   float(np.max(ys) - np.min(ys)), 1e-6)
        scale = (2.0 * block_radius) / span
        ax0, ay0 = anchor[int(comm_id)]
        for col in block:
            px, py = sub_pos[col]
            pos_col[col] = (ax0 + (px - cx) * scale,
                            ay0 + (py - cy) * scale)

    # --- 2b. 柱节点最小间距约束 ---
    for idx_blk, comm_id in enumerate(comms):
        start = int(boundaries[idx_blk])
        end = int(boundaries[idx_blk + 1]) if idx_blk + 1 < len(boundaries) else N
        block = [int(col) for col in order[start:end]]
        if len(block) < 2:
            continue
        for _ in range(40):
            moved = False
            for a_idx in range(len(block)):
                for b_idx in range(a_idx + 1, len(block)):
                    a, b = block[a_idx], block[b_idx]
                    xa, ya = pos_col[a]
                    xb, yb = pos_col[b]
                    dx = xa - xb
                    dy = ya - yb
                    dist = math.hypot(dx, dy)
                    if dist < MIN_NODE_DIST and dist > 1e-9:
                        push = (MIN_NODE_DIST - dist) / 2.0
                        ux, uy = dx / dist, dy / dist
                        pos_col[a] = (xa + ux * push, ya + uy * push)
                        pos_col[b] = (xb - ux * push, yb - uy * push)
                        moved = True
            if not moved:
                break

    # --- 输入/输出层 y 坐标 ---
    OUT_X = anchor_x0 + ncols * anchor_step_x + 0.5
    y_in = {j: (j + 1.0) / (brain.obs_dim + 1.0) * total_height for j in range(brain.obs_dim)}
    y_out = {i: (i + 1.0) / (brain.action_dim + 1.0) * total_height for i in range(brain.action_dim)}

    community_colors = list(mcolors.TABLEAU_COLORS.values())
    tau_norm = plt.Normalize(cfg.TAU_E_MIN, cfg.TAU_E_MAX)
    in_degree = brain.M_rec.sum(dim=0).numpy()

    fig, ax = plt.subplots(figsize=(17, max(8, total_height + 2.0)))

    # --- 边：输入 → 柱（直线，绿色） ---
    W_in_abs = _norm_edge_weights(brain.W_in.detach().numpy())
    for i in range(N):
        for j in range(brain.obs_dim):
            if brain.M_in[i, j] > 0:
                nw = W_in_abs[i, j]
                x1, y1 = pos_col[i]
                ax.plot([0.0, x1], [y_in[j], y1], color='green',
                        lw=0.3 + 2.2 * nw, alpha=0.15 + 0.6 * nw,
                        solid_capstyle='round', zorder=1)

    # --- 边：柱 → 输出（直线，品红） ---
    W_out_abs = _norm_edge_weights(brain.W_out.detach().numpy())
    for i in range(brain.action_dim):
        for j in range(N):
            if brain.M_out[i, j] > 0:
                nw = W_out_abs[i, j]
                x1, y1 = pos_col[j]
                ax.plot([x1, OUT_X], [y1, y_out[i]], color='magenta',
                        lw=0.3 + 2.2 * nw, alpha=0.15 + 0.6 * nw,
                        solid_capstyle='round', zorder=1)

    # --- 边：柱 → 柱（top-K 贝塞尔弧线，红=兴奋 / 蓝=抑制） ---
    rec_edges = [(i, j, W_rec_np[i, j])
                 for i in range(N) for j in range(N) if M_rec_np[i, j] > 0]
    rec_edges.sort(key=lambda e: -abs(e[2]))
    top_k = min(180, len(rec_edges))
    if top_k > 0:
        rec_abs = np.abs([e[2] for e in rec_edges[:top_k]])
        w_max, w_min = rec_abs.max(), rec_abs.min()
        for i, j, w in rec_edges[:top_k]:
            nw = (abs(w) - w_min) / (w_max - w_min + 1e-8)
            x0p, y0p = pos_col[j]
            x1p, y1p = pos_col[i]
            dist = np.hypot(x1p - x0p, y1p - y0p)
            curve = 0.12 + 0.10 * dist
            mid_x = (x0p + x1p) / 2.0
            mid_y = (y0p + y1p) / 2.0
            ctrl = (mid_x + curve, mid_y + curve)
            verts = [(x0p, y0p), ctrl, (x1p, y1p)]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            ax.add_patch(PathPatch(Path(verts, codes), facecolor='none',
                                   edgecolor='red' if w >= 0 else 'blue',
                                   lw=0.3 + 2.0 * nw, alpha=0.10 + 0.55 * nw, zorder=2))

    # --- 柱节点 ---
    for col in range(N):
        x, y = pos_col[col]
        tau_color = plt.cm.plasma(tau_norm(tau[col]))
        ax.scatter(x, y, s=78, color=tau_color, edgecolors='black', linewidths=0.8, zorder=3)
        ax.scatter(x, y, s=48, color=tau_color, zorder=4)
        in_n = int(in_degree[col])
        if in_n > 0:
            freq_norm = min(in_n, 20) / 20.0
            brightness = 0.55 + 0.45 * freq_norm
            ax.scatter(x, y, s=6.0,
                       color=(brightness, brightness, brightness), zorder=5)

    # --- 输入 / 输出节点 ---
    for j in range(brain.obs_dim):
        ax.scatter(0.0, y_in[j], marker='s', s=140, color='limegreen', edgecolors='black', zorder=3)
        ax.text(0.0, y_in[j], f"In_{j}", fontsize=7, ha='right', va='center')
    act_names = ['None', 'Left', 'Main', 'Right']
    for i in range(brain.action_dim):
        ax.scatter(OUT_X, y_out[i], marker='D', s=160, color='orange', edgecolors='black', zorder=3)
        ax.text(OUT_X, y_out[i], f"Out_{i}\n{act_names[i]}", fontsize=8, ha='left', va='center')

    # --- 社区标注 ---
    for idx_blk, comm_id in enumerate(comms):
        ax0, ay0 = anchor[int(comm_id)]
        ax.text(ax0, ay0 + block_radius + 0.05, f"C{comm_id}", fontsize=9,
                ha='center', va='bottom',
                color=community_colors[int(comm_id) % len(community_colors)], fontweight='bold')

    # --- 区域背景与楼层标签 ---
    ax.add_patch(Rectangle((anchor_x0 - 0.35, -0.35),
                           ncols * anchor_step_x + 0.7, total_height + 0.7,
                           facecolor='lightgray', alpha=0.18, zorder=0,
                           edgecolor='gray', linestyle='--', linewidth=0.8))
    ax.text((anchor_x0 + (ncols - 1) * anchor_step_x + anchor_x0) / 2.0,
            total_height + 0.42, 'Cortical Columns (E-I units)',
            ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.text(0.0, total_height + 0.42, 'Input', ha='center', va='bottom',
            fontsize=12, fontweight='bold')
    ax.text(OUT_X, total_height + 0.42, 'Output', ha='center', va='bottom',
            fontsize=12, fontweight='bold')

    # --- 图例 / colorbar / 装饰 ---
    handles = [
        Patch(facecolor='limegreen', edgecolor='black', label='Input node (8 obs)'),
        Patch(facecolor='orange', edgecolor='black', label='Output node (4 actions)'),
        Patch(facecolor='white', edgecolor='black', label='Column node (ring+fill=tau_e, white-dot brightness=in-degree freq)'),
        Line2D([0], [0], color='green', lw=2, label='Input→Column edge'),
        Line2D([0], [0], color='magenta', lw=2, label='Column→Output edge'),
        Line2D([0], [0], color='red', lw=2, label='Recurrent E (W>0)'),
        Line2D([0], [0], color='blue', lw=2, label='Recurrent I (W<0)'),
    ]
    ax.legend(handles=handles, loc='lower left', fontsize=8, framealpha=0.9)

    sm = plt.cm.ScalarMappable(cmap='plasma', norm=tau_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label('tau_e_init', fontsize=10)

    ax.set_xlim(-1.1, OUT_X + 0.9)
    ax.set_ylim(-0.8, total_height + 0.8)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("A) Evolved Brain Topology — Three-Layer Pipeline\n"
                 "Columns spread in 2D by community (islands) | E=red / I=blue | "
                 "thickness/alpha ∝ |W| | top-K recurrent edges", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_connectivity_matrices(brain, cfg, partition):
    """方案 B：连接矩阵仪表盘（M_in / M_rec / M_out 热力图）"""
    N = brain.N
    order, comm_of_col, comms, boundaries = _community_order(brain, partition)

    W_in_map = (brain.W_in * brain.M_in).detach().numpy()
    W_rec_map = (brain.W_rec * brain.M_rec).detach().numpy()
    W_out_map = (brain.W_out * brain.M_out).detach().numpy()

    M_in_sorted = W_in_map[order, :]
    M_rec_sorted = W_rec_map[np.ix_(order, order)]
    M_out_sorted = W_out_map[:, order]

    rec_vmax = float(np.abs(M_rec_sorted).max()) + 1e-8

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))

    im0 = axes[0].imshow(M_in_sorted, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[0].set_title("Input W_in (Column × Feature)", fontsize=12)
    axes[0].set_xlabel("Input feature (0-7)")
    axes[0].set_ylabel("Column (community/τ sorted)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(M_rec_sorted, aspect='auto', cmap='RdBu', interpolation='nearest',
                         vmin=-rec_vmax, vmax=rec_vmax)
    axes[1].set_title("Recurrent W_rec (Target × Source)\nE=red / I=blue", fontsize=12)
    axes[1].set_xlabel("Source column")
    axes[1].set_ylabel("Target column")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    for b in boundaries[1:]:
        line_pos = int(b) - 0.5
        axes[1].axhline(y=line_pos, color='white', lw=1.0, alpha=0.9)
        axes[1].axvline(x=line_pos, color='white', lw=1.0, alpha=0.9)

    im2 = axes[2].imshow(M_out_sorted, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[2].set_title("Output W_out (Action × Column)", fontsize=12)
    axes[2].set_xlabel("Column (community/τ sorted)")
    axes[2].set_ylabel("Action (0=None, 1=Left, 2=Main, 3=Right)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    fig.suptitle("B) Connectivity Matrices — sorted by functional community & tau_e", fontsize=14)
    plt.show()


def visualize_brain_ecosystem(brain, env, cfg):
    print("\n=== Generating Brain Ecosystem Visualization ===")

    # --- 1. 拓扑网络图 ---
    G = nx.DiGraph()
    for i in range(brain.obs_dim): G.add_node(f"In_{i}", layer='input')
    for i in range(brain.N): G.add_node(f"Col_{i}", layer='column')
    for i in range(brain.action_dim): G.add_node(f"Out_{i}", layer='output')

    for i in range(brain.N):
        for j in range(brain.obs_dim):
            if brain.M_in[i, j] > 0:
                G.add_edge(f"In_{j}", f"Col_{i}", weight=abs(brain.W_in[i, j].item()))
    for i in range(brain.N):
        for j in range(brain.N):
            if brain.M_rec[i, j] > 0:
                G.add_edge(f"Col_{j}", f"Col_{i}", weight=abs(brain.W_rec[i, j].item()))
    for i in range(brain.action_dim):
        for j in range(brain.N):
            if brain.M_out[i, j] > 0:
                G.add_edge(f"Col_{j}", f"Out_{i}", weight=abs(brain.W_out[i, j].item()))

    try:
        partition = community_louvain.best_partition(G.to_undirected())
    except ImportError:
        communities = nx.algorithms.community.greedy_modularity_communities(G.to_undirected())
        partition = {}
        for i, com in enumerate(communities):
            for node in com:
                partition[node] = i

    plot_topology_layered(brain, cfg, partition)
    plot_connectivity_matrices(brain, cfg, partition)

    # --- 2. 运行时内部状态 + 动作潜力热力图 ---
    brain.restore_genetic_baseline()
    brain.reset_runtime()
    obs, _ = env.reset()
    E = torch.zeros(brain.N)
    I = torch.zeros(brain.N)
    E_history, excit_history, inhib_history, tau_history = [], [], [], []
    logits_history = []
    actions = []
    done = False
    steps = 0

    while not done and steps < 100:
        action, avg_logits, E, I = deliberate_action(brain, obs, E, I)
        logits_history.append(avg_logits.numpy().copy())

        E_history.append(E.numpy().copy())
        excit_history.append(brain.hormone_excit.numpy().copy())
        inhib_history.append(brain.hormone_inhib.numpy().copy())
        eff_tau = brain.tau_e_init + \
            cfg.SHORT_TERM_GAIN * brain.short_term_state + \
            cfg.EXCIT_HORMONE_GAIN * brain.hormone_excit - \
            cfg.INHIB_HORMONE_GAIN * brain.hormone_inhib
        tau_history.append(torch.clamp(eff_tau, cfg.TAU_E_MIN, cfg.TAU_E_MAX).detach().numpy().copy())
        actions.append(action)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        obs = next_obs
        steps += 1

    E_history = np.array(E_history)
    excit_history = np.array(excit_history)
    inhib_history = np.array(inhib_history)
    tau_history = np.array(tau_history)
    logits_history = np.array(logits_history)

    fig, axes = plt.subplots(5, 1, figsize=(14, 20))

    im0 = axes[0].imshow(E_history.T, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[0].set_title("Column Excitation (E) Over Time", fontsize=13)
    axes[0].set_ylabel("Columns")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(tau_history.T, aspect='auto', cmap='plasma', interpolation='nearest')
    axes[1].set_title("Effective tau_e (init + short-term + hormone)", fontsize=13)
    axes[1].set_ylabel("Columns")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(excit_history.T, aspect='auto', cmap='Reds', interpolation='nearest')
    axes[2].set_title("Excitatory Hormone (small diffusion)", fontsize=13)
    axes[2].set_ylabel("Columns")
    plt.colorbar(im2, ax=axes[2])

    im3 = axes[3].imshow(inhib_history.T, aspect='auto', cmap='Blues', interpolation='nearest')
    axes[3].set_title("Inhibitory Hormone (large diffusion)", fontsize=13)
    axes[3].set_ylabel("Columns")
    plt.colorbar(im3, ax=axes[3])

    im4 = axes[4].imshow(logits_history.T, aspect='auto', cmap='RdBu', interpolation='nearest', vmin=-3, vmax=3)
    axes[4].set_title("Net Action Logits (K-Frame Avg Logits)", fontsize=13)
    axes[4].set_ylabel("Action (0=None, 1=Left, 2=Main, 3=Right)")
    axes[4].set_xlabel("Time Steps (Game Steps)")
    plt.colorbar(im4, ax=axes[4])

    for t, act in enumerate(actions):
        axes[4].plot(t, act, 'k*', markersize=10, markeredgecolor='yellow', markeredgewidth=0.5)

    action_changes = [i for i in range(1, len(actions)) if actions[i] != actions[i - 1]]
    for ax in axes:
        for t in action_changes:
            ax.axvline(x=t, color='lime', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    # --- 3. tau_e / Wei / Wie 进化分布 ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 14))
    param_specs = [
        (axes[0], brain.tau_e_init.data.numpy(), cfg.BASE_TAU_E, 'purple',
         'tau_e init', 'Evolved tau_e per Column'),
        (axes[1], brain.w_ei.data.numpy(), cfg.W_EI, 'coral',
         'Wei', 'Evolved Wei (E→I coupling)'),
        (axes[2], brain.w_ie.data.numpy(), cfg.W_IE, 'teal',
         'Wie', 'Evolved Wie (I→E coupling)'),
    ]
    for ax, data, base, color, ylabel, title in param_specs:
        ax.bar(range(brain.N), data, color=color, alpha=0.7)
        ax.axhline(y=base, color='gray', linestyle='--',
                   label=f'Base {ylabel}={base}')
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Column Index")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Evolved Per-Column Parameters", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


# ==========================================
# 4. 检查点 / 断点续训 / 最优模型
# ==========================================
def _config_dict(cfg):
    """收集生效的配置为可序列化 dict（实例属性优先于类属性）。

    沿 type(cfg) 的 MRO 逐类收集类属性（支持 Config 子类继承覆盖），
    再以实例属性覆盖同名类属性，确保序列化与断点校验反映真正生效的配置。
    """
    merged = {}
    for cls in type(cfg).mro():
        if cls is object:
            continue
        merged.update({k: v for k, v in vars(cls).items() if not k.startswith('__')})
    merged.update(vars(cfg))
    return merged


def save_brain_state(brain, use_half=True):
    """将单个个体的遗传属性打包为可序列化 dict（同 test5a）。"""
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
        }


def load_brain_state(state, cfg):
    """从 save_brain_state 的 dict 重建 EIBrainRegion 个体（同 test5a）。"""
    new = EIBrainRegion.__new__(EIBrainRegion)
    nn.Module.__init__(new)

    new.cfg = cfg
    new.N = int(state['N'])
    new.obs_dim = cfg.OBS_DIM
    new.action_dim = cfg.ACTION_DIM

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

    new.register_buffer('hormone_excit', torch.zeros(new.N))
    new.register_buffer('hormone_inhib', torch.zeros(new.N))
    new.register_buffer('short_term_state', torch.zeros(new.N))
    new.register_buffer('W_rec_eff', torch.zeros(new.N, new.N))
    new.register_buffer('W_out_eff', torch.zeros(new.action_dim, new.N))
    new.register_buffer('M_norm', torch.zeros(new.N, new.N))

    new.baseline = None

    new.refresh_cached()
    new.save_genetic_baseline()
    return new


def save_checkpoint(path, cfg, next_gen, population, history,
                    cum_eval_time=0.0, cum_evolve_time=0.0,
                    best_brain=None, best_reward=-1e9, best_steps=0.0):
    """保存训练断点：先写 .tmp 临时文件，再原子替换正式文件。"""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    payload = {
        'next_gen': next_gen,
        'population': [save_brain_state(ind) for ind in population],
        'history': history,
        'cum_eval_time': float(cum_eval_time),
        'cum_evolve_time': float(cum_evolve_time),
        'best_brain': save_brain_state(best_brain) if best_brain is not None else None,
        'best_reward': float(best_reward),
        'best_steps': float(best_steps),
        'config': _config_dict(cfg),
        'random_state': random.getstate(),
        'torch_rng_state': torch.get_rng_state(),
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    tmp_path = path + '.tmp'
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    print(f"  [Checkpoint] 断点已保存 -> {path} "
          f"(next_gen={next_gen}, saved_at={payload['saved_at']})")


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

    population = [load_brain_state(s, cfg) for s in data['population']]
    best_brain = (load_brain_state(data['best_brain'], cfg)
                  if data.get('best_brain') is not None else None)

    random.setstate(data['random_state'])
    torch.set_rng_state(data['torch_rng_state'])

    return {
        'next_gen': int(data['next_gen']),
        'population': population,
        'history': data['history'],
        'cum_eval_time': float(data.get('cum_eval_time', 0.0)),
        'cum_evolve_time': float(data.get('cum_evolve_time', 0.0)),
        'best_brain': best_brain,
        'best_reward': float(data.get('best_reward', -1e9)),
        'best_steps': float(data.get('best_steps', 0.0)),
    }


def save_best_model(path, brain, cfg, reward, steps):
    """训练完成时保存最优个体（fp32 全精度，体积小，便于直接加载复用）。"""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    torch.save({
        'brain': save_brain_state(brain, use_half=False),
        'reward': float(reward),
        'steps': float(steps),
        'config': _config_dict(cfg),
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }, path)


def load_best_model_brain(path, cfg):
    """从 save_best_model 保存的最优模型文件中恢复个体与其指标。"""
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
    return brain, float(data.get('reward', -1e9)), float(data.get('steps', 0.0))


# ==========================================
# 5. 加速训练：多进程并行评估（A 方案）+ 两阶段快速筛选（C 方案）
# ==========================================
def _worker_cfg_from_dict(cfg_dict):
    """从配置 dict 重建一个轻量 cfg 对象（用于子进程 worker）。"""
    tmp = type('_RtCfg', (), {})()
    for k, v in cfg_dict.items():
        setattr(tmp, k, v)
    return tmp


def _eval_worker_serialize(task):
    """多进程 worker：分片批处理，评估一个分片内的所有个体（LunarLander 版）。"""
    items, cfg_dict, episodes, max_steps = task
    rt_cfg = _worker_cfg_from_dict(cfg_dict)

    # 每个 worker 进程的随机种子彼此不同（Windows spawn 下需差异化）
    random.seed(os.getpid())

    env = gym.make(rt_cfg.ENV_NAME)
    results = []
    try:
        for idx, genes in items:
            brain = load_brain_state(genes, rt_cfg)
            results.append((idx, evaluate_individual(brain, env,
                                                     max_steps=max_steps,
                                                     episodes=episodes)))
    finally:
        env.close()
    return results


def _eval_batch(inds, cfg, pool, episodes, env=None):
    """对一组个体评估指定局数并返回 (reward, steps) 列表（同 test5a）。"""
    if pool is not None:
        n = len(inds)
        if n == 0:
            return []
        cfg_dict = _config_dict(cfg)
        n_chunks = max(1, min(pool._processes, n))
        chunk_size = (n + n_chunks - 1) // n_chunks
        chunks = []
        for c in range(n_chunks):
            lo = c * chunk_size
            hi = min(lo + chunk_size, n)
            items = [(i, save_brain_state(inds[i], use_half=False))
                     for i in range(lo, hi)]
            chunks.append((items, cfg_dict, episodes, cfg.MAX_STEPS))

        flat = []
        for chunk_results in pool.map(_eval_worker_serialize, chunks):
            flat.extend(chunk_results)
        flat.sort(key=lambda x: x[0])
        return [m for _, m in flat]

    results = []
    for ind in inds:
        results.append(evaluate_individual(ind, env, episodes=episodes))
    return results


def evaluate_population(population, cfg, env, pool=None):
    """统一评估入口：结合 A 方案（并行）与 C 方案（两阶段筛选）。

    调度逻辑（SCREEN_ENABLE=True 且 1 <= SCREEN_EPISODES < EVAL_EPISODES 时启用 C）：
    1. 阶段 1：全种群初筛 SCREEN_EPISODES 局；
    2. 自适应保护：快筛最高奖励<=0（整体过弱无法区分）时回退全量评估；
    3. 阶段 2：按 (reward, -steps) 排序，取前 K = ELITE_SIZE × SCREEN_MULTIPLIER
       名个体补评 (EVAL_EPISODES - SCREEN_EPISODES) 局；
    4. 汇总：精评个体 = 初筛局 + 补评局加权平均；非精评个体 = 仅初筛局分数。
    """
    episodes = cfg.EVAL_EPISODES
    screen_ep = int(getattr(cfg, 'SCREEN_EPISODES', 1))

    use_screen = (bool(getattr(cfg, 'SCREEN_ENABLE', False)) and
                  screen_ep >= 1 and screen_ep < episodes)

    if not use_screen:
        return _eval_batch(population, cfg, pool, episodes, env)

    # --- 阶段 1：全种群快筛 ---
    quick = _eval_batch(population, cfg, pool, screen_ep, env)

    # --- 自适应保护：整体过弱时回退全量评估 ---
    if bool(getattr(cfg, 'SCREEN_AUTO_FALLBACK', True)) and max(m[0] for m in quick) <= 0:
        return _eval_batch(population, cfg, pool, episodes, env)

    # --- 阶段 2：前 K 名进入精评 ---
    k = cfg.ELITE_SIZE * int(getattr(cfg, 'SCREEN_MULTIPLIER', 3))
    k = max(1, min(k, len(population)))
    if k >= len(population):
        return _eval_batch(population, cfg, pool, episodes, env)

    order = sorted(range(len(population)),
                   key=lambda i: (quick[i][0], -quick[i][1]),
                   reverse=True)
    refine_idx = set(order[:k])
    refine_list = [population[i] for i in range(len(population)) if i in refine_idx]

    extra = _eval_batch(refine_list, cfg, pool, episodes - screen_ep, env)

    # --- 汇总 ---
    metrics = [None] * len(population)
    ref_pos = 0
    for i in range(len(population)):
        if i in refine_idx:
            f_q, s_q = quick[i]
            f_e, s_e = extra[ref_pos]
            ref_pos += 1
            metrics[i] = ((f_q * screen_ep + f_e * (episodes - screen_ep)) / episodes,
                          (s_q * screen_ep + s_e * (episodes - screen_ep)) / episodes)
        else:
            metrics[i] = quick[i]
    return metrics


def make_eval_pool(cfg):
    """按配置创建评估进程池；PARALLEL_EVAL=False 时返回 None（串行）。"""
    if not cfg.PARALLEL_EVAL or cfg.POP_SIZE < 2:
        return None
    n_workers = int(getattr(cfg, 'NUM_WORKERS', 0))
    if n_workers <= 0:
        n_workers = min(os.cpu_count() or 1, 16)
    n_workers = max(1, min(n_workers, cfg.POP_SIZE))
    return mp.Pool(n_workers)


# ==========================================
# 6. 主循环（断点续训 + 每 N 代自动保存 + 完成保存最优模型）
# ==========================================
if __name__ == "__main__":
    cfg = Config()
    env = gym.make(cfg.ENV_NAME)

    t_program = time.perf_counter()

    # ---- 断点自动接续 ----
    start_gen = 0
    population = None
    history = {'gen': [], 'best_reward': [], 'avg_reward': [], 'best_steps': []}
    cum_eval_time = 0.0
    cum_evolve_time = 0.0
    best_ever_brain = None
    best_ever_reward = -1e9
    best_ever_steps = 0.0

    if cfg.AUTO_RESUME:
        ckpt_path = cfg.CHECKPOINT_PATH
        if (not os.path.exists(ckpt_path) and
                getattr(cfg, 'CHECKPOINT_FALLBACK', None) and
                os.path.exists(cfg.CHECKPOINT_FALLBACK)):
            print(f"[Migrate] 未发现新断点 {ckpt_path}，检测到旧版断点 "
                  f"{cfg.CHECKPOINT_FALLBACK}，将自动迁移接续")
            ckpt_path = cfg.CHECKPOINT_FALLBACK
        if os.path.exists(ckpt_path):
            ckpt = load_checkpoint(ckpt_path, cfg)
            if ckpt is not None:
                start_gen = ckpt['next_gen']
                population = ckpt['population']
                history = ckpt['history']
                cum_eval_time = ckpt['cum_eval_time']
                cum_evolve_time = ckpt['cum_evolve_time']
                best_ever_brain = ckpt['best_brain']
                best_ever_reward = ckpt['best_reward']
                best_ever_steps = ckpt['best_steps']
                last_done = history['gen'][-1] + 1 if history['gen'] else 0
                print(f"\n=== 检测到断点 [{ckpt_path}] ===")
                print(f"  {cfg.GENERATIONS} 代中已完成 {last_done} 代 -> 从第 {start_gen} 代接续 | "
                      f"历史最优: Reward={best_ever_reward:.1f}, Steps={best_ever_steps:.1f} | "
                      f"已耗时: {cum_eval_time + cum_evolve_time:.1f}s")

    if population is None:
        try:
            t_init_start = time.perf_counter()
            print("Initializing Population...")
            print(f"  Observation: LunarLander-v3 {cfg.OBS_DIM} dim | Actions: {cfg.ACTION_DIM}")
            print(f"  K-frame deliberation: FRAME_RATE={cfg.FRAME_RATE}, INPUT_DECAY={cfg.INPUT_DECAY}")
            print(f"  Fatigue mechanism: disabled")
            if getattr(cfg, 'FREEZE_SCHEME', 'cycle') == 'cycle':
                pattern = getattr(cfg, 'CYCLE_PATTERN', [('G2',), ('G1',), ('G2', 'G3'), ('G1',), ('G3',)])
                names = {'G1': '结构', 'G2': '动力学', 'G3': '激素'}
                shown = ['+'.join((f"{g}({names[g]})" for g in slot)) for slot in pattern]
                print(f"  Freeze evolution: scheme=cycle, 周期序列: {' -> '.join(shown)}")
            else:
                print(f"  Freeze evolution: scheme={cfg.FREEZE_SCHEME}, "
                      f"G1(结构)每{getattr(cfg, 'G1_INTERVAL', 3)}代 / "
                      f"G2(动力学)每{getattr(cfg, 'G2_INTERVAL', 1)}代 / "
                      f"G3(激素)每{getattr(cfg, 'G3_INTERVAL', 5)}代")
            population = [EIBrainRegion(cfg) for _ in range(cfg.POP_SIZE)]
            for ind in population:
                ind.save_genetic_baseline()

            # ---- 种子继承：以已有最优模型作为种群的精英个体 ----
            if cfg.SEED_FROM_BEST:
                seed_path = cfg.BEST_MODEL_PATH
                if (not os.path.exists(seed_path) and
                        getattr(cfg, 'BEST_MODEL_FALLBACK', None) and
                        os.path.exists(cfg.BEST_MODEL_FALLBACK)):
                    print(f"  [Seed] 未发现新最优模型 {seed_path}，改用旧版 "
                          f"{cfg.BEST_MODEL_FALLBACK} 作为种子")
                    seed_path = cfg.BEST_MODEL_FALLBACK
                seed_result = load_best_model_brain(seed_path, cfg)
                if seed_result is not None:
                    seed, seed_reward, seed_steps = seed_result
                    population[0] = seed
                    best_ever_brain = seed
                    best_ever_reward = seed_reward
                    best_ever_steps = seed_steps
                    print(f"  [Seed] 已注入最优模型 {seed_path} "
                          f"作为种群种子（上轮 Reward={seed_reward:.1f}, Steps={seed_steps:.1f}；"
                          f"其余 {cfg.POP_SIZE - 1} 个体随机初始化）")
                else:
                    print(f"  [Seed] 未发现可用的最优模型种子，全新随机初始化")

            print(f"  Population init done in {time.perf_counter() - t_init_start:.1f}s")
        except KeyboardInterrupt:
            print("\n种群初始化期间被中断（此时尚无断点可续，下次运行将重新初始化）")
            sys.exit(0)

        # 初始断点
        save_checkpoint(cfg.CHECKPOINT_PATH, cfg, start_gen, population, history,
                        cum_eval_time, cum_evolve_time,
                        best_ever_brain, best_ever_reward, best_ever_steps)

    # ---- 评估进程池 ----
    eval_pool = make_eval_pool(cfg)
    if eval_pool is not None:
        print(f"[Parallel] 多进程并行评估已启用："
              f"{eval_pool._processes} workers "
              f"(SCREEN={cfg.SCREEN_ENABLE}, SCREEN_EPISODES={cfg.SCREEN_EPISODES}, "
              f"SCREEN_MULTIPLIER={cfg.SCREEN_MULTIPLIER})")

    # ---- 进化主循环 ----
    cur_gen = None
    try:
        for gen in range(start_gen, cfg.GENERATIONS):
            cur_gen = gen
            t_gen_start = time.perf_counter()
            metrics = evaluate_population(population, cfg, env, eval_pool)
            eval_time = time.perf_counter() - t_gen_start
            cum_eval_time += eval_time

            best_idx = max(range(len(metrics)),
                           key=lambda i: (metrics[i][0], -metrics[i][1]))
            best_reward = metrics[best_idx][0]
            best_steps = metrics[best_idx][1]
            avg_reward = np.mean([m[0] for m in metrics])

            history['gen'].append(gen)
            history['best_reward'].append(best_reward)
            history['avg_reward'].append(avg_reward)
            history['best_steps'].append(best_steps)

            best_brain = population[best_idx]

            # 跨代跟踪历史最优个体
            if (best_reward > best_ever_reward or
                    (best_reward == best_ever_reward and best_steps < best_ever_steps)):
                best_ever_reward = best_reward
                best_ever_steps = best_steps
                best_ever_brain = best_brain.clone()

            if gen < cfg.GENERATIONS - 1:
                t_ev_start = time.perf_counter()
                population = evolve_topology(population, metrics, cfg,
                                             gen=gen, best_idx=best_idx)
                evolve_time = time.perf_counter() - t_ev_start
                cum_evolve_time += evolve_time
            else:
                evolve_time = 0.0

            active_names = {'G1': '结构', 'G2': '动力学', 'G3': '激素'}
            active_groups = ''.join(sorted(f"{g}({active_names[g]})" for g in _freeze_active_groups(gen, cfg))) or '无'

            print(f"Gen {gen+1}/{cfg.GENERATIONS} | "
                  f"BestReward: {best_reward:.1f} | BestSteps: {best_steps:.1f} | AvgReward: {avg_reward:.1f} | "
                  f"Active: {active_groups} | "
                  f"Edges: {best_brain.M_in.sum().item():.0f}-in / "
                  f"{best_brain.M_rec.sum().item():.0f}-rec / "
                  f"{best_brain.M_out.sum().item():.0f}-out | "
                  f"tau_e: [{best_brain.tau_e_init.min().item():.3f}, "
                  f"{best_brain.tau_e_init.max().item():.3f}] | "
                  f"Wei: [{best_brain.w_ei.min().item():.3f}, "
                  f"{best_brain.w_ei.max().item():.3f}] | "
                  f"Wie: [{best_brain.w_ie.min().item():.3f}, "
                  f"{best_brain.w_ie.max().item():.3f}] | "
                  f"eval {eval_time:.1f}s / evolve {evolve_time:.1f}s")

            # 每 CHECKPOINT_INTERVAL 代自动保存一次
            if (gen + 1) % cfg.CHECKPOINT_INTERVAL == 0:
                save_checkpoint(cfg.CHECKPOINT_PATH, cfg, gen + 1, population,
                                history, cum_eval_time, cum_evolve_time,
                                best_ever_brain, best_ever_reward, best_ever_steps)

    except KeyboardInterrupt:
        if eval_pool is not None:
            eval_pool.terminate()
            eval_pool.join()
        nxt = cur_gen if cur_gen is not None else start_gen
        print("\n训练被中断 (Ctrl+C)，正在保存断点以供下次自动接续...")
        save_checkpoint(cfg.CHECKPOINT_PATH, cfg, nxt, population,
                        history, cum_eval_time, cum_evolve_time,
                        best_ever_brain, best_ever_reward, best_ever_steps)
        print(f"断点已保存: {cfg.CHECKPOINT_PATH} (下次运行将从第 {nxt} 代接续)")
        sys.exit(0)

    # 正常结束：关闭进程池
    if eval_pool is not None:
        eval_pool.close()
        eval_pool.join()
    env.close()

    print(f"\nTotal runtime: {time.perf_counter() - t_program:.1f}s "
          f"(eval {cum_eval_time:.1f}s / evolve {cum_evolve_time:.1f}s)")

    # ---- 训练完成：保存最优模型 ----
    if best_ever_brain is None:
        best_ever_brain = population[0]
    save_best_model(cfg.BEST_MODEL_PATH, best_ever_brain, cfg,
                    best_ever_reward, best_ever_steps)
    print(f"\n最优模型已保存: {cfg.BEST_MODEL_PATH} "
          f"(Reward={best_ever_reward:.1f}, Steps={best_ever_steps:.1f})")

    # 训练已全部完成，删除临时断点释放空间
    if os.path.exists(cfg.CHECKPOINT_PATH):
        os.remove(cfg.CHECKPOINT_PATH)
        print(f"训练已完成，已删除临时断点: {cfg.CHECKPOINT_PATH}")

    plot_history(history)

    best_brain = best_ever_brain

    print("\n--- Best Brain Summary ---")
    print(f"Input connections active:   {best_brain.M_in.sum().item():.0f}/{best_brain.N * best_brain.obs_dim}")
    print(f"Internal connections active: {best_brain.M_rec.sum().item():.0f}/{best_brain.N * best_brain.N}")
    print(f"Output connections active:   {best_brain.M_out.sum().item():.0f}/{best_brain.action_dim * best_brain.N}")
    print(f"tau_e range: [{best_brain.tau_e_init.min().item():.3f}, {best_brain.tau_e_init.max().item():.3f}]")
    print(f"Wei range:   [{best_brain.w_ei.min().item():.3f}, {best_brain.w_ei.max().item():.3f}]")
    print(f"Wie range:   [{best_brain.w_ie.min().item():.3f}, {best_brain.w_ie.max().item():.3f}]")
    print(f"Evolved Output Bias: {best_brain.b_out.data.numpy()}")
    print(f"Hormone net L1 norm: W_hormone1={best_brain.W_hormone1.data.abs().sum().item():.4f}, "
          f"W_excit={best_brain.W_excit.data.abs().sum().item():.4f}, "
          f"W_inhib={best_brain.W_inhib.data.abs().sum().item():.4f}")

    print("\nLaunching visualization for the best individual...")
    env_viz = gym.make(cfg.ENV_NAME)
    visualize_best_brain_play(best_brain, cfg, max_steps=500)
    visualize_brain_ecosystem(best_brain, env_viz, cfg)
    env_viz.close()