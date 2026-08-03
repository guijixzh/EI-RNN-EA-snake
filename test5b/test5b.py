"""test5b：CNN 前端 + E-I 皮质柱进化（完整参考 test5a.py 重建）。

相对 test5a.py 的改动（本质仅更换输入模块）：
  1. 输入：10x10x5 网格 -> CNNEncoder（冻结，全局共享）-> 24 维投影特征
  2. EI 输入维度保持 24（CNN 投影头输出）；不再继承 test5a 模型
     （两位的 24 维语义不同：test5a=手工特征布局，test5b=CNN 语义特征）
  3. K 帧思考：每个游戏步 CNN 只算 1 次特征，K 帧内特征复用
  4. 其余进化机制与 test5a 完全一致：多进程并行评估（A 方案）、
     两阶段筛选（C 方案）、G1/G2/G3 冻结交替、断点续训、最优模型保存
  5. seed 继承 / 版本迁移全部禁用：全新随机初始化，test5b 专用 checkpoint/best 路径

多进程说明：test5a 的多线程/多进程评估本身没有问题（问题此前出在
env.py 的 _place_food 在满盘时随机重试死循环，已修复为空格枚举）。
因此本文件恢复 test5a 的 mp.Pool 并行评估。

用法：
    python test5b/test5b.py            # 断点自动接续或全新训练
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

# 同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import SnakeEnv
from cnn import CNNEncoder, load_cnn_encoder


# ==========================================
# 0. 全局配置类（test5a 全套参数 + CNN 专属；禁用 seed/fallback）
# ==========================================
class Config:
    # --- 进化参数（与 test5a 一致） ---
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

    # --- 环境参数 ---
    GRID_SIZE = 10
    EVAL_EPISODES = 5
    MAX_STEPS = 500

    # --- 脑结构参数 ---
    NUM_COLUMNS = 256
    OBS_DIM = 24            # CNN 投影头输出维度（语义特征，非 test5a 手工特征）
    ACTION_DIM = 3
    INIT_DENSITY = 0.15

    # --- E-I 动力学参数（与 test5a 一致） ---
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

    # --- 激素系统参数 ---
    HORMONE_DECAY = 0.95
    EXCIT_HORMONE_GAIN = 0.25
    INHIB_HORMONE_GAIN = 0.50
    EXCIT_DIFFUSION = 0.15
    INHIB_DIFFUSION = 0.40
    HORMONE_NET_HIDDEN = 32

    # --- 动作疲劳参数 ---
    FATIGUE_GAIN = 0.01
    FATIGUE_THRESHOLD = 4
    FATIGUE_MAX = 5.0

    # --- K 倍帧率思考 ---
    FRAME_RATE = 5
    INPUT_DECAY = 0.9

    # --- 交替冻结进化（与 test5a 一致） ---
    FREEZE_SCHEME = 'cycle'
    CYCLE_PATTERN = [('G2',), ('G1',), ('G2', 'G3'), ('G1',), ('G3',)]
    G1_INTERVAL = 3
    G2_INTERVAL = 1
    G3_INTERVAL = 5

    # --- 加速训练配置（与 test5a 一致） ---
    PARALLEL_EVAL = True
    NUM_WORKERS = 0            # 0 = 自动取 min(os.cpu_count(), 16)
    SCREEN_ENABLE = True
    SCREEN_EPISODES = 2
    SCREEN_MULTIPLIER = 3
    SCREEN_AUTO_FALLBACK = True

    # --- 检查点 / 最优模型（test5b 专用路径；禁用旧版迁移与种子继承） ---
    CHECKPOINT_PATH = 'test5b/test5b_checkpoint.pth'
    BEST_MODEL_PATH = 'test5b/test5b_best_model.pth'
    CNN_PATH = 'test5b/cnn_encoder.pth'
    CHECKPOINT_FALLBACK = None      # 禁用：不迁移 test5a 断点（24 维语义不同）
    BEST_MODEL_FALLBACK = None      # 禁用：不注入 test5a 最优模型作种子
    AUTO_RESUME = True
    CHECKPOINT_INTERVAL = 5
    SEED_FROM_BEST = False          # 禁用：全新随机初始化


# ==========================================
# 1. E-I 皮质柱脑区模型（test5a 逻辑；输入改为 CNN 投影特征）
# ==========================================
class EIBrainRegion(nn.Module):
    def __init__(self, cfg, cnn=None):
        super().__init__()
        self.cfg = cfg
        self.N = cfg.NUM_COLUMNS
        self.obs_dim = cfg.OBS_DIM
        self.action_dim = cfg.ACTION_DIM

        # --- CNN 前端（冻结，全局共享，非进化参数） ---
        self.cnn = cnn if cnn is not None else CNNEncoder()
        for p in self.cnn.parameters():
            p.requires_grad = False
        self.cnn.eval()

        # --- 基因型：拓扑掩码 ---
        self.M_in = (torch.rand(self.N, self.obs_dim) < cfg.INIT_DENSITY).float()
        self.M_rec = (torch.rand(self.N, self.N) < cfg.INIT_DENSITY).float()
        torch.diagonal(self.M_rec).zero_()
        self.M_out = (torch.rand(self.action_dim, self.N) < cfg.INIT_DENSITY).float()

        # --- 表现型：突触权重 ---
        self.W_in = nn.Parameter(torch.randn(self.N, self.obs_dim) * 0.1)
        self.W_rec = nn.Parameter(torch.randn(self.N, self.N) * 0.05)
        self.W_out = nn.Parameter(torch.randn(self.action_dim, self.N) * 0.1)
        self.b_out = nn.Parameter(torch.zeros(self.action_dim))

        # --- 每柱体 tau_e ---
        self.tau_e_init = nn.Parameter(
            cfg.BASE_TAU_E + (torch.rand(self.N) * 2 - 1) * cfg.TAU_E_NOISE
        )

        # --- 每柱体 Wei / Wie ---
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

        # --- 运行时状态（非进化参数） ---
        self.register_buffer('hormone_excit', torch.zeros(self.N))
        self.register_buffer('hormone_inhib', torch.zeros(self.N))
        self.register_buffer('short_term_state', torch.zeros(self.N))
        self.register_buffer('consecutive_counts', torch.zeros(self.action_dim))

        self.baseline = None

        # --- 缓存 ---
        self.register_buffer('W_rec_eff', torch.zeros(self.N, self.N))
        self.register_buffer('W_out_eff', torch.zeros(self.action_dim, self.N))
        self.register_buffer('M_norm', torch.zeros(self.N, self.N))
        self.refresh_cached()

    def reset_runtime(self):
        self.hormone_excit.zero_()
        self.hormone_inhib.zero_()
        self.short_term_state.zero_()
        self.consecutive_counts.zero_()

    def encode_grid(self, grid_t):
        """网格(1,10,10,5) -> 24 维投影特征（冻结 CNN，单前向）。"""
        with torch.no_grad():
            feat = self.cnn(grid_t)   # (1, 24)
        return feat

    def forward(self, obs_in, E_prev, I_prev):
        """obs_in: (24,) 已由 CNN 投影的特征向量（每游戏步计算一次）。"""
        # 1. 外部与循环输入
        ext_in = torch.matmul(self.W_in * self.M_in, obs_in)
        rec_in = torch.matmul(self.W_rec_eff, E_prev)
        total_in = ext_in + rec_in

        # 2. 激素调控前馈网络
        hormone_input = torch.cat([E_prev, I_prev, total_in], dim=-1)
        h_hidden = torch.relu(torch.matmul(self.W_hormone1, hormone_input) + self.b_hormone1)
        excit_cmd = torch.relu(torch.matmul(self.W_excit, h_hidden) + self.b_excit)
        inhib_cmd = torch.relu(torch.matmul(self.W_inhib, h_hidden) + self.b_inhib)

        # 3. 激素沿拓扑扩散 + 长期衰减
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

        # 5b. 有效 Wei / Wie
        w_ei_eff = torch.clamp(self.w_ei, self.cfg.W_EI_MIN, self.cfg.W_EI_MAX)
        w_ie_eff = torch.clamp(self.w_ie, self.cfg.W_IE_MIN, self.cfg.W_IE_MAX)

        # 6. E-I 离散代数更新
        E_new = torch.sigmoid(total_in + effective_tau_e * E_prev - w_ei_eff * I_prev)
        I_new = torch.sigmoid(w_ie_eff * E_new)

        # 7. 动作输出
        action_logits = torch.matmul(self.W_out_eff, E_new) + self.b_out

        # 8. 动作疲劳抑制
        fatigue = torch.relu(self.consecutive_counts - self.cfg.FATIGUE_THRESHOLD) * self.cfg.FATIGUE_GAIN
        fatigue = torch.clamp(fatigue, max=self.cfg.FATIGUE_MAX)
        action_logits = action_logits - fatigue

        return action_logits, E_new, I_new

    def update_fatigue(self, action):
        with torch.no_grad():
            cur = float(self.consecutive_counts[action]) + 1.0
            self.consecutive_counts.zero_()
            self.consecutive_counts[action] = cur

    def refresh_cached(self):
        with torch.no_grad():
            self.W_rec_eff.copy_(self.W_rec.data * self.M_rec)
            self.W_out_eff.copy_(self.W_out.data * self.M_out)
            deg = self.M_rec.sum(dim=1, keepdim=True) + 1e-8
            self.M_norm.copy_(self.M_rec / deg)

    def save_genetic_baseline(self):
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
        """轻量克隆（CNN 共享引用，只复制 EI 基因）。"""
        new = EIBrainRegion.__new__(EIBrainRegion)
        nn.Module.__init__(new)

        new.cfg = self.cfg
        new.N = self.N
        new.obs_dim = self.obs_dim
        new.action_dim = self.action_dim
        new.cnn = self.cnn  # 共享冻结 CNN

        new.M_in = self.M_in.clone()
        new.M_rec = self.M_rec.clone()
        new.M_out = self.M_out.clone()

        new.W_in = nn.Parameter(self.W_in.data.clone())
        new.W_rec = nn.Parameter(self.W_rec.data.clone())
        new.W_out = nn.Parameter(self.W_out.data.clone())
        new.b_out = nn.Parameter(self.b_out.data.clone())
        new.tau_e_init = nn.Parameter(self.tau_e_init.data.clone())
        new.w_ei = nn.Parameter(self.w_ei.data.clone())
        new.w_ie = nn.Parameter(self.w_ie.data.clone())

        new.W_hormone1 = nn.Parameter(self.W_hormone1.data.clone())
        new.b_hormone1 = nn.Parameter(self.b_hormone1.data.clone())
        new.W_excit = nn.Parameter(self.W_excit.data.clone())
        new.b_excit = nn.Parameter(self.b_excit.data.clone())
        new.W_inhib = nn.Parameter(self.W_inhib.data.clone())
        new.b_inhib = nn.Parameter(self.b_inhib.data.clone())

        new.register_buffer('hormone_excit', torch.zeros(self.N))
        new.register_buffer('hormone_inhib', torch.zeros(self.N))
        new.register_buffer('short_term_state', torch.zeros(self.N))
        new.register_buffer('consecutive_counts', torch.zeros(self.action_dim))
        new.register_buffer('W_rec_eff', torch.zeros(self.N, self.N))
        new.register_buffer('W_out_eff', torch.zeros(self.action_dim, self.N))
        new.register_buffer('M_norm', torch.zeros(self.N, self.N))

        new.baseline = self.baseline

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
# 2. K 倍帧率思考（CNN 特征每游戏步算 1 次，K 帧复用）
# ==========================================
def deliberate_action(brain, grid_t, E, I, K=None, decay=None):
    """每个游戏步：CNN 算 1 次特征，K 帧内部更新复用特征，输出平均 logits。"""
    cfg = brain.cfg
    if K is None:
        K = cfg.FRAME_RATE
    if decay is None:
        decay = cfg.INPUT_DECAY

    # CNN 特征：每游戏步只算 1 次（冻结，no_grad）
    feat = brain.encode_grid(grid_t)          # (1, 24)
    scales = [decay ** k for k in range(K)]
    logits_sum = None
    with torch.no_grad():
        for k in range(K):
            obs_t = feat * scales[k]          # 衰减施加在特征上
            logits, E, I = brain(obs_t.squeeze(0), E, I)
            if logits_sum is None:
                logits_sum = logits
            else:
                logits_sum = logits_sum + logits

    avg_logits = logits_sum / K
    action = torch.argmax(avg_logits).item()
    return action, avg_logits, E, I


# ==========================================
# 3. 评估与进化逻辑（test5a 逻辑；输入改为 CNN 网格特征）
# ==========================================
def evaluate_individual(brain, env, render=False, max_steps=None, episodes=None):
    """评估单个个体指定局数（与 test5a 语义一致）。

    输入为 10x10x5 网格 -> CNN 特征 -> EI。
    """
    cfg = brain.cfg
    if max_steps is None:
        max_steps = cfg.MAX_STEPS
    if episodes is None:
        episodes = cfg.EVAL_EPISODES

    total_foods = []
    total_steps_list = []
    total_action_counts = [0, 0, 0]

    for ep in range(episodes):
        brain.reset_runtime()
        env.reset()
        E = torch.zeros(brain.N)
        I = torch.zeros(brain.N)

        ep_food = 0
        steps = 0
        done = False

        while not done and steps < max_steps:
            grid_t = torch.from_numpy(env._get_grid_state()).unsqueeze(0)  # (1,10,10,5)
            action, avg_logits, E, I = deliberate_action(brain, grid_t, E, I)

            brain.update_fatigue(action)
            total_action_counts[action] += 1

            _, ate_food, done = env.step(action)
            if ate_food:
                ep_food += 1
            steps += 1

        total_foods.append(ep_food)
        total_steps_list.append(steps)

    avg_food = np.mean(total_foods)
    avg_steps = np.mean(total_steps_list)

    # 硬性淘汰：如果只向一侧转弯，直接判定为最差适应度（test5a 同源逻辑）
    turn_lim = max(episodes, 1)
    if (total_action_counts[1] > turn_lim or total_action_counts[2] > turn_lim) and \
       (total_action_counts[1] == 0 or total_action_counts[2] == 0):
        avg_food = 0
        avg_steps = 99999

    if render:
        print(f"  [Render] Food: {avg_food:.1f}, Steps: {avg_steps:.1f}")

    return avg_food, avg_steps


def _freeze_active_groups(gen, cfg):
    """按 FREEZE_SCHEME 计算第 gen 代激活的参数组集合（与 test5a 一致）。"""
    scheme = getattr(cfg, 'FREEZE_SCHEME', 'cycle')
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


def evolve_topology(population, metrics_list, cfg, gen=0):
    """进化下一代（与 test5a 完全一致：G1/G2/G3 软冻结交替）。"""
    active = _freeze_active_groups(gen, cfg)
    has_g1 = 'G1' in active
    has_g2 = 'G2' in active
    has_g3 = 'G3' in active

    sorted_indices = sorted(
        range(len(metrics_list)),
        key=lambda i: (metrics_list[i][0], -metrics_list[i][1]),
        reverse=True
    )

    elite_idx = sorted_indices[:cfg.ELITE_SIZE]
    elites = [population[i].clone() for i in elite_idx]

    new_pop = [e.clone() for e in elites]

    while len(new_pop) < len(population):
        p1, p2 = random.sample(elites, 2)
        child = p1.clone()
        N = child.N

        with torch.no_grad():
            # G1 交叉：结构组（掩码 + 权重 + 输出偏置）
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

                mask_out = torch.rand_like(p1.b_out.data) > 0.5
                child.b_out.data = torch.where(mask_out, p1.b_out.data, p2.b_out.data)

            # G2 交叉：动力学组
            if has_g2:
                col_mask2 = torch.rand(N) > 0.5
                child.tau_e_init.data = torch.where(col_mask2, p1.tau_e_init.data, p2.tau_e_init.data)
                child.w_ei.data = torch.where(col_mask2, p1.w_ei.data, p2.w_ei.data)
                child.w_ie.data = torch.where(col_mask2, p1.w_ie.data, p2.w_ie.data)

            # G3 交叉：激素组
            if has_g3:
                for attr in ['W_hormone1', 'b_hormone1', 'W_excit', 'b_excit', 'W_inhib', 'b_inhib']:
                    p1_t = getattr(p1, attr).data
                    p2_t = getattr(p2, attr).data
                    mask = torch.rand_like(p1_t) > 0.5
                    getattr(child, attr).data = torch.where(mask, p1_t, p2_t)

        with torch.no_grad():
            # G1 变异：拓扑 + 权重
            if has_g1:
                if random.random() < cfg.TOPOLOGY_MUT_PROB:
                    m_attr = random.choice(['M_in', 'M_rec', 'M_out'])
                    m_tensor = getattr(child, m_attr)
                    mut_mask = torch.rand_like(m_tensor) < cfg.MUT_RATE
                    m_tensor[mut_mask] = 1.0 - m_tensor[mut_mask]

                for attr in ['W_in', 'W_rec', 'W_out', 'b_out']:
                    w_tensor = getattr(child, attr).data
                    noise = torch.randn_like(w_tensor) * cfg.WEIGHT_MUT_STD
                    noise_mask = torch.rand_like(w_tensor) < cfg.WEIGHT_MUT_FRAC
                    setattr(child, attr, nn.Parameter(w_tensor + noise * noise_mask))

            # G2 变异：动力学
            if has_g2:
                tau_noise = torch.randn_like(child.tau_e_init.data) * cfg.TAU_E_MUT_STD
                child.tau_e_init.data = torch.clamp(
                    child.tau_e_init.data + tau_noise,
                    cfg.TAU_E_MIN, cfg.TAU_E_MAX
                )
                w_ei_noise = torch.randn_like(child.w_ei.data) * cfg.W_EI_MUT_STD
                child.w_ei.data = torch.clamp(
                    child.w_ei.data + w_ei_noise,
                    cfg.W_EI_MIN, cfg.W_EI_MAX
                )
                w_ie_noise = torch.randn_like(child.w_ie.data) * cfg.W_IE_MUT_STD
                child.w_ie.data = torch.clamp(
                    child.w_ie.data + w_ie_noise,
                    cfg.W_IE_MIN, cfg.W_IE_MAX
                )

            # G3 变异：激素
            if has_g3:
                for attr in ['W_hormone1', 'b_hormone1', 'W_excit', 'b_excit', 'W_inhib', 'b_inhib']:
                    w = getattr(child, attr).data
                    noise = torch.randn_like(w) * cfg.HORMONE_MUT_STD
                    mask = torch.rand_like(w) < cfg.HORMONE_MUT_FRAC
                    getattr(child, attr).data = w + noise * mask

        child.refresh_cached()
        child.save_genetic_baseline()
        new_pop.append(child)

    return new_pop


# ==========================================
# 4. 可视化（简化：进化曲线 + 简单渲染）
# ==========================================
def plot_history(history, save_path='test5b/evolution.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history['gen'], history['best_food'], label='Best Food', color='red', marker='o', markersize=3)
    ax1.plot(history['gen'], history['avg_food'], label='Avg Food', color='blue', alpha=0.6)
    ax1.set_title("Evolution Progress — Food Count (Primary Criterion)")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Food Eaten")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['gen'], history['best_steps'], label='Best Steps', color='green', marker='s', markersize=3)
    ax2.set_title("Best Individual Steps (Lower = Better, among top food)")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Steps Survived")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"[Plot] 进化曲线已保存 -> {save_path}")
    plt.close(fig)


def visualize_best_brain_play(brain, cfg, max_steps=300):
    env = SnakeEnv(grid_size=cfg.GRID_SIZE)
    brain.restore_genetic_baseline()
    brain.reset_runtime()

    env.reset()
    E = torch.zeros(brain.N)
    I = torch.zeros(brain.N)

    steps = 0
    done = False
    while not done and steps < max_steps:
        grid_t = torch.from_numpy(env._get_grid_state()).unsqueeze(0)
        action, avg_logits, E, I = deliberate_action(brain, grid_t, E, I)
        brain.update_fatigue(action)
        _, _, done = env.step(action)
        steps += 1

    print(f"\n[Render] Game Over! Final Score: {len(env.body) - 2} | "
          f"Survived Steps: {steps}")


# ==========================================
# 5. 检查点 / 断点续训 / 最优模型（test5b 专用；适配 CNN 共享）
# ==========================================
def _config_dict(cfg):
    merged = {}
    merged.update(vars(cfg.__class__))
    merged.update(vars(cfg))
    return {k: v for k, v in merged.items() if not k.startswith('__')}


def save_brain_state(brain, use_half=True):
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


def load_brain_state(state, cfg, cnn):
    new = EIBrainRegion.__new__(EIBrainRegion)
    nn.Module.__init__(new)

    new.cfg = cfg
    new.N = int(state['N'])
    new.obs_dim = cfg.OBS_DIM
    new.action_dim = cfg.ACTION_DIM
    new.cnn = cnn  # 共享冻结 CNN

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
    new.register_buffer('consecutive_counts', torch.zeros(new.action_dim))
    new.register_buffer('W_rec_eff', torch.zeros(new.N, new.N))
    new.register_buffer('W_out_eff', torch.zeros(new.action_dim, new.N))
    new.register_buffer('M_norm', torch.zeros(new.N, new.N))

    new.baseline = None
    new.refresh_cached()
    new.save_genetic_baseline()
    return new


def save_checkpoint(path, cfg, next_gen, population, history,
                    cum_eval_time=0.0, cum_evolve_time=0.0,
                    best_brain=None, best_food=-1.0, best_steps=0.0):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    payload = {
        'next_gen': next_gen,
        'population': [save_brain_state(ind) for ind in population],
        'history': history,
        'cum_eval_time': float(cum_eval_time),
        'cum_evolve_time': float(cum_evolve_time),
        'best_brain': save_brain_state(best_brain) if best_brain is not None else None,
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
    print(f"  [Checkpoint] 断点已保存 -> {path} "
          f"(next_gen={next_gen}, saved_at={payload['saved_at']})")


def load_checkpoint(path, cfg, cnn):
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

    population = [load_brain_state(s, cfg, cnn) for s in data['population']]
    best_brain = (load_brain_state(data['best_brain'], cfg, cnn)
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
        'best_food': float(data.get('best_food', -1.0)),
        'best_steps': float(data.get('best_steps', 0.0)),
    }


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


def load_best_model_brain(path, cfg, cnn):
    if not os.path.exists(path):
        return None

    try:
        data = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"警告: 最优模型 {path} 读取失败 ({e})，已忽略")
        return None

    saved_cfg = data.get('config', {})
    if saved_cfg:
        if (saved_cfg.get('NUM_COLUMNS') != cfg.NUM_COLUMNS or
                saved_cfg.get('OBS_DIM') != cfg.OBS_DIM or
                saved_cfg.get('ACTION_DIM') != cfg.ACTION_DIM):
            print(f"警告: 最优模型 {path} 与当前配置不匹配 (N/OBS/ACTION_DIM)，已忽略")
            return None

    brain = load_brain_state(data['brain'], cfg, cnn)
    return brain, float(data.get('food', -1.0)), float(data.get('steps', 0.0))


# ==========================================
# 6. 加速训练：多进程并行评估（A 方案）+ 两阶段快速筛选（C 方案）
#    （与 test5a 完全一致；worker 内加载一次冻结 CNN 并缓存）
# ==========================================
def _worker_cfg_from_dict(cfg_dict):
    tmp = type('_RtCfg', (), {})()
    for k, v in cfg_dict.items():
        setattr(tmp, k, v)
    return tmp


# 每个 worker 进程内的 CNN 缓存（进程级全局，避免每个分片重复加载）
_WORKER_CNN_CACHE = {}


def _get_worker_cnn(cnn_path):
    if '_cnn' not in _WORKER_CNN_CACHE:
        _WORKER_CNN_CACHE['_cnn'] = load_cnn_encoder(cnn_path)
    return _WORKER_CNN_CACHE['_cnn']


def _eval_worker_serialize(task):
    """多进程 worker：分片批处理，评估一个分片内的所有个体。

    输入 task = ([(idx, 基因 dict), ...], 配置 dict, episodes, max_steps)。
    - 基因经 save_brain_state(use_half=False) 全精度传输；
    - 每个 worker 只接收一个分片（IPC 次数 = worker 数而非个体数），
      共享一个 SnakeEnv 与一份冻结 CNN；逐个体重建脑区实例；
    - 随机种子用 os.getpid() 差异化。
    """
    items, cfg_dict, episodes, max_steps = task
    rt_cfg = _worker_cfg_from_dict(cfg_dict)

    random.seed(os.getpid())

    cnn = _get_worker_cnn(rt_cfg.CNN_PATH)
    env = SnakeEnv(grid_size=rt_cfg.GRID_SIZE)
    results = []
    for idx, genes in items:
        brain = load_brain_state(genes, rt_cfg, cnn)
        results.append((idx, evaluate_individual(brain, env,
                                                 max_steps=max_steps,
                                                 episodes=episodes)))
    return results


def _eval_batch(inds, cfg, pool, episodes, env=None):
    """对一组个体评估指定局数并返回 (food, steps) 列表。"""
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
    """统一评估入口：A 方案（并行）+ C 方案（两阶段筛选），与 test5a 一致。"""
    episodes = cfg.EVAL_EPISODES
    screen_ep = int(getattr(cfg, 'SCREEN_EPISODES', 1))

    use_screen = (bool(getattr(cfg, 'SCREEN_ENABLE', False)) and
                  screen_ep >= 1 and screen_ep < episodes)

    if not use_screen:
        return _eval_batch(population, cfg, pool, episodes, env)

    quick = _eval_batch(population, cfg, pool, screen_ep, env)

    if bool(getattr(cfg, 'SCREEN_AUTO_FALLBACK', True)) and max(m[0] for m in quick) <= 0:
        return _eval_batch(population, cfg, pool, episodes, env)

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
# 7. 主循环（test5a 逻辑；移除种子继承/版本迁移；CNN 前置加载）
# ==========================================
if __name__ == "__main__":
    cfg = Config()
    env = SnakeEnv(grid_size=cfg.GRID_SIZE)

    # ---- 前置校验：冻结 CNN 必须存在 ----
    if not os.path.exists(cfg.CNN_PATH):
        print(f"[FATAL] CNN 编码器不存在: {cfg.CNN_PATH}")
        print("请先运行: python test5b/train_cnn.py --data test5b/dataset.npz "
              "--out test5b/cnn_encoder.pth")
        sys.exit(1)
    cnn = load_cnn_encoder(cfg.CNN_PATH)
    print(f"[CNN] 已加载冻结编码器 {cfg.CNN_PATH}")

    t_program = time.perf_counter()

    # ---- 断点自动接续（仅 test5b 专用路径，无版本迁移） ----
    start_gen = 0
    population = None
    history = {'gen': [], 'best_food': [], 'avg_food': [], 'best_steps': []}
    cum_eval_time = 0.0
    cum_evolve_time = 0.0
    best_ever_brain = None
    best_ever_food = -1.0
    best_ever_steps = 0.0

    if cfg.AUTO_RESUME:
        ckpt_path = cfg.CHECKPOINT_PATH
        if os.path.exists(ckpt_path):
            ckpt = load_checkpoint(ckpt_path, cfg, cnn)
            if ckpt is not None:
                start_gen = ckpt['next_gen']
                population = ckpt['population']
                history = ckpt['history']
                cum_eval_time = ckpt['cum_eval_time']
                cum_evolve_time = ckpt['cum_evolve_time']
                best_ever_brain = ckpt['best_brain']
                best_ever_food = ckpt['best_food']
                best_ever_steps = ckpt['best_steps']
                last_done = history['gen'][-1] + 1 if history['gen'] else 0
                print(f"\n=== 检测到断点 [{ckpt_path}] ===")
                print(f"  {cfg.GENERATIONS} 代中已完成 {last_done} 代 -> 从第 {start_gen} 代接续 | "
                      f"历史最优: Food={best_ever_food:.1f}, Steps={best_ever_steps:.1f} | "
                      f"已耗时: {cum_eval_time + cum_evolve_time:.1f}s")

    if population is None:
        try:
            t_init_start = time.perf_counter()
            print("Initializing Population...")
            print(f"  输入: CNN(10x10x5) -> {cfg.OBS_DIM} 维语义特征 -> EI({cfg.NUM_COLUMNS} 柱)")
            print(f"  K-frame deliberation: FRAME_RATE={cfg.FRAME_RATE}, INPUT_DECAY={cfg.INPUT_DECAY}")
            if getattr(cfg, 'FREEZE_SCHEME', 'cycle') == 'cycle':
                pattern = getattr(cfg, 'CYCLE_PATTERN', [('G2',), ('G1',), ('G2', 'G3'), ('G1',), ('G3',)])
                names = {'G1': '结构', 'G2': '动力学', 'G3': '激素'}
                shown = ['+'.join((f"{g}({names[g]})" for g in slot)) for slot in pattern]
                print(f"  Freeze evolution: scheme=cycle, 周期序列: {' -> '.join(shown)}")
            else:
                print(f"  Freeze evolution: scheme={cfg.FREEZE_SCHEME}")
            print(f"  Seed: 禁用（全新随机初始化，{cfg.POP_SIZE} 个体）")
            population = [EIBrainRegion(cfg, cnn=cnn) for _ in range(cfg.POP_SIZE)]
            for ind in population:
                ind.save_genetic_baseline()

            print(f"  Population init done in {time.perf_counter() - t_init_start:.1f}s")
        except KeyboardInterrupt:
            print("\n种群初始化期间被中断（此时尚无断点可续，下次运行将重新初始化）")
            sys.exit(0)

        # 初始断点：保证刚启动即被中断也不丢失种群
        save_checkpoint(cfg.CHECKPOINT_PATH, cfg, start_gen, population, history,
                        cum_eval_time, cum_evolve_time,
                        best_ever_brain, best_ever_food, best_ever_steps)

    # ---- 评估进程池（A 方案并行） ----
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
            best_food = metrics[best_idx][0]
            best_steps = metrics[best_idx][1]
            avg_food = np.mean([m[0] for m in metrics])

            history['gen'].append(gen)
            history['best_food'].append(best_food)
            history['avg_food'].append(avg_food)
            history['best_steps'].append(best_steps)

            best_brain = population[best_idx]

            # 跨代跟踪历史最优个体
            if (best_food > best_ever_food or
                    (best_food == best_ever_food and best_steps < best_ever_steps)):
                best_ever_food = best_food
                best_ever_steps = best_steps
                best_ever_brain = best_brain.clone()

            if gen < cfg.GENERATIONS - 1:
                t_ev_start = time.perf_counter()
                population = evolve_topology(population, metrics, cfg, gen=gen)
                evolve_time = time.perf_counter() - t_ev_start
                cum_evolve_time += evolve_time
            else:
                evolve_time = 0.0

            active_names = {'G1': '结构', 'G2': '动力学', 'G3': '激素'}
            active_groups = ''.join(sorted(f"{g}({active_names[g]})"
                                           for g in _freeze_active_groups(gen, cfg))) or '无'

            print(f"Gen {gen+1}/{cfg.GENERATIONS} | "
                  f"BestFood: {best_food:.1f} | BestSteps: {best_steps:.1f} | "
                  f"AvgFood: {avg_food:.1f} | Active: {active_groups} | "
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
                                best_ever_brain, best_ever_food, best_ever_steps)

    except KeyboardInterrupt:
        if eval_pool is not None:
            eval_pool.terminate()
            eval_pool.join()
        nxt = cur_gen if cur_gen is not None else start_gen
        print("\n训练被中断 (Ctrl+C)，正在保存断点以供下次自动接续...")
        save_checkpoint(cfg.CHECKPOINT_PATH, cfg, nxt, population,
                        history, cum_eval_time, cum_evolve_time,
                        best_ever_brain, best_ever_food, best_ever_steps)
        print(f"断点已保存: {cfg.CHECKPOINT_PATH} (下次运行将从第 {nxt} 代接续)")
        sys.exit(0)

    # 正常结束：关闭进程池
    if eval_pool is not None:
        eval_pool.close()
        eval_pool.join()

    print(f"\nTotal runtime: {time.perf_counter() - t_program:.1f}s "
          f"(eval {cum_eval_time:.1f}s / evolve {cum_evolve_time:.1f}s)")

    # ---- 训练完成：保存 test5b 专用最优模型 ----
    if best_ever_brain is None:
        best_ever_brain = population[0]
    save_best_model(cfg.BEST_MODEL_PATH, best_ever_brain, cfg,
                    best_ever_food, best_ever_steps)
    print(f"\n最优模型已保存: {cfg.BEST_MODEL_PATH} "
          f"(Food={best_ever_food:.1f}, Steps={best_ever_steps:.1f})")

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

    print("\nRendering best individual...")
    visualize_best_brain_play(best_brain, cfg, max_steps=300)