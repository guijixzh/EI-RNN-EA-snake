import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import time
import networkx as nx
import matplotlib.colors as mcolors

# ==========================================
# 0. 全局配置类（所有重要参数集中管理）
# ==========================================
class Config:
    # --- 进化参数 ---
    POP_SIZE = 1024
    GENERATIONS = 2
    ELITE_SIZE = 64
    MUT_RATE = 0.05            # 拓扑变异率（位翻转概率）
    TOPOLOGY_MUT_PROB = 0.05   # 发生拓扑变异的个体概率
    WEIGHT_MUT_FRAC = 0.2      # 权重变异比例
    WEIGHT_MUT_STD = 0.1       # 权重变异标准差
    TAU_E_MUT_STD = 0.05       # tau_e 初始值变异标准差
    HORMONE_MUT_FRAC = 0.1     # 激素网络变异比例
    HORMONE_MUT_STD = 0.05     # 激素网络变异标准差

    # --- 环境参数 ---
    GRID_SIZE = 10
    EVAL_EPISODES = 5
    MAX_STEPS = 500

    # --- 脑结构参数 ---
    NUM_COLUMNS = 64
    OBS_DIM = 24
    ACTION_DIM = 3
    INIT_DENSITY = 0.15

    # --- E-I 动力学参数 ---
    BASE_TAU_E = 0.7           # tau_e 基准值
    TAU_E_NOISE = 0.1          # tau_e 初始随机范围（小范围）
    TAU_E_MIN = 0.2            # tau_e 下限
    TAU_E_MAX = 1.2            # tau_e 上限
    W_EI = 2.0                 # 抑制强度
    W_IE = 2.0                 # 兴奋触发抑制的强度

    # --- 短期 tau 调制（柱体自身状态，迅速消失）---
    SHORT_TERM_GAIN = 0.2      # 短期状态对 tau_e 的影响强度
    SHORT_TERM_DECAY = 0.3     # 短期状态保留率（越小消失越快）

    # --- 激素系统参数（长期 tau 调制）---
    HORMONE_DECAY = 0.95       # 激素保留率（越大越持久）
    EXCIT_HORMONE_GAIN = 0.25  # 兴奋激素对 tau_e 的增强
    INHIB_HORMONE_GAIN = 0.50  # 抑制激素对 tau_e 的削弱
    EXCIT_DIFFUSION = 0.15     # 兴奋激素沿拓扑扩散程度（小范围）
    INHIB_DIFFUSION = 0.40     # 抑制激素沿拓扑扩散程度（大范围）
    HORMONE_NET_HIDDEN = 32    # 激素前馈网络隐藏层大小

    # --- 局部预测编码学习 ---
    PC_LR = 0.001
    PC_DECAY = 0.0001
    LAMARCKIAN_FACTOR = 0.2    # 软拉马克遗传比例


# ==========================================
# 1. 轻量级贪吃蛇环境（去除所有奖励设计）
# ==========================================
class SnakeEnv:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.head = (self.grid_size // 2, self.grid_size // 2)
        self.dir = random.choice(((0, 1), (1, 0), (0, -1), (-1, 0)))
        self.body = [self.head, (self.head[0] - self.dir[0], self.head[1] - self.dir[1])]
        self._place_food()
        self.food_count = 0
        self.steps = 0
        self.steps_without_food = 0
        return self._get_obs()

    def _place_food(self):
        while True:
            self.food = (random.randint(0, self.grid_size - 1),
                         random.randint(0, self.grid_size - 1))
            if self.food not in self.body:
                break

    def _get_obs(self):
        obs = np.zeros(6, dtype=np.float32)
        left_dir = (-self.dir[1], self.dir[0])
        right_dir = (self.dir[1], -self.dir[0])
        for i, d in enumerate([self.dir, left_dir, right_dir]):
            next_pos = (self.head[0] + d[0], self.head[1] + d[1])
            if (next_pos[0] < 0 or next_pos[0] >= self.grid_size or
                next_pos[1] < 0 or next_pos[1] >= self.grid_size or
                next_pos in self.body):
                obs[i] = 1.0
        dx = self.food[0] - self.head[0]
        dy = self.food[1] - self.head[1]
        if (dx * self.dir[0] + dy * self.dir[1]) > 0: obs[3] = 1.0
        if (dx * left_dir[0] + dy * left_dir[1]) > 0: obs[4] = 1.0
        if (dx * right_dir[0] + dy * right_dir[1]) > 0: obs[5] = 1.0

        forward = self.dir
        back = (-self.dir[0], -self.dir[1])
        left = (-self.dir[1], self.dir[0])
        right = (self.dir[1], -self.dir[0])
        offsets = [
            (forward[0] + left[0], forward[1] + left[1]), forward,
            (forward[0] + right[0], forward[1] + right[1]),
            left, (0, 0), right,
            (back[0] + left[0], back[1] + left[1]), back,
            (back[0] + right[0], back[1] + right[1])
        ]
        local_vision = []
        for r_off, c_off in offsets:
            r, c = self.head[0] + r_off, self.head[1] + c_off
            is_obstacle = 0.0
            is_food = 0.0
            if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
                is_obstacle = 1.0
            elif (r, c) in self.body:
                is_obstacle = 1.0
            elif (r, c) == self.food:
                is_food = 1.0
            local_vision.extend([is_obstacle, is_food])
        obs = np.concatenate([obs, np.array(local_vision, dtype=np.float32)])
        return obs

    def step(self, action):
        """去除所有奖励设计，仅返回 (obs, ate_food, done)"""
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
            return self._get_obs(), False, True

        self.body.insert(0, next_head)
        self.head = next_head

        ate_food = False
        if self.head == self.food:
            self.food_count += 1
            ate_food = True
            self.steps_without_food = 0
            self._place_food()
        else:
            self.body.pop()

        if self.steps_without_food > len(self.body) + 20:
            return self._get_obs(), False, True

        return self._get_obs(), ate_food, False


# ==========================================
# 2. E-I 皮质柱脑区模型（含 tau_e 进化 + 激素调控前馈网络）
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
        self.W_pred = nn.Parameter(torch.randn(self.N, self.N) * 0.1)

        # --- 每柱体 tau_e 初始值（小范围随机，受进化迭代）---
        self.tau_e_init = nn.Parameter(
            cfg.BASE_TAU_E + (torch.rand(self.N) * 2 - 1) * cfg.TAU_E_NOISE
        )

        # --- 激素调控前馈网络（初始化为 0，由进化驱动）---
        # 输入：网络整体状态 [E, I, total_in]  (N*3 维)
        # 输出：每个柱体的兴奋/抑制激素释放量
        hormone_input_dim = self.N * 3
        self.W_hormone1 = nn.Parameter(torch.zeros(cfg.HORMONE_NET_HIDDEN, hormone_input_dim))
        self.b_hormone1 = nn.Parameter(torch.zeros(cfg.HORMONE_NET_HIDDEN))
        self.W_excit = nn.Parameter(torch.zeros(self.N, cfg.HORMONE_NET_HIDDEN))
        self.b_excit = nn.Parameter(torch.zeros(self.N))
        self.W_inhib = nn.Parameter(torch.zeros(self.N, cfg.HORMONE_NET_HIDDEN))
        self.b_inhib = nn.Parameter(torch.zeros(self.N))

        # --- 运行时状态（非进化参数）---
        self.register_buffer('hormone_excit', torch.zeros(self.N))
        self.register_buffer('hormone_inhib', torch.zeros(self.N))
        self.register_buffer('short_term_state', torch.zeros(self.N))

        self.w_ei = cfg.W_EI
        self.w_ie = cfg.W_IE
        self.baseline = None

    def reset_runtime(self):
        """每局开始前重置激素与短期状态"""
        self.hormone_excit.zero_()
        self.hormone_inhib.zero_()
        self.short_term_state.zero_()

    def forward(self, obs_t, E_prev, I_prev):
        # 1. 外部与循环输入
        ext_in = torch.matmul(self.W_in * self.M_in, obs_t)
        rec_in = torch.matmul(self.W_rec * self.M_rec, E_prev)
        total_in = ext_in + rec_in

        # 2. 激素调控前馈网络：接收网络整体状态 → 决定在何处释放何种激素
        hormone_input = torch.cat([E_prev, I_prev, total_in], dim=-1)  # (N*3,)
        h_hidden = torch.relu(torch.matmul(self.W_hormone1, hormone_input) + self.b_hormone1)
        excit_cmd = torch.relu(torch.matmul(self.W_excit, h_hidden) + self.b_excit)  # 兴奋激素释放量
        inhib_cmd = torch.relu(torch.matmul(self.W_inhib, h_hidden) + self.b_inhib)  # 抑制激素释放量

        # 3. 激素沿拓扑网络扩散 + 长期衰减
        #    兴奋激素扩散范围小，抑制激素扩散范围大
        with torch.no_grad():
            deg = self.M_rec.sum(dim=1, keepdim=True) + 1e-8
            M_norm = self.M_rec / deg  # 行归一化的拓扑扩散矩阵

        self.hormone_excit = (1 - self.cfg.HORMONE_DECAY) * excit_cmd + \
            self.cfg.HORMONE_DECAY * (
                (1 - self.cfg.EXCIT_DIFFUSION) * self.hormone_excit +
                self.cfg.EXCIT_DIFFUSION * torch.matmul(M_norm, self.hormone_excit)
            )

        self.hormone_inhib = (1 - self.cfg.HORMONE_DECAY) * inhib_cmd + \
            self.cfg.HORMONE_DECAY * (
                (1 - self.cfg.INHIB_DIFFUSION) * self.hormone_inhib +
                self.cfg.INHIB_DIFFUSION * torch.matmul(M_norm, self.hormone_inhib)
            )

        # 4. 短期状态（柱体自身状态，迅速消失）
        self.short_term_state = self.cfg.SHORT_TERM_DECAY * self.short_term_state + \
            (1 - self.cfg.SHORT_TERM_DECAY) * E_prev

        # 5. 有效 tau_e = 初始值 + 短期调制 + 长期激素调制
        effective_tau_e = self.tau_e_init + \
            self.cfg.SHORT_TERM_GAIN * self.short_term_state + \
            self.cfg.EXCIT_HORMONE_GAIN * self.hormone_excit - \
            self.cfg.INHIB_HORMONE_GAIN * self.hormone_inhib
        effective_tau_e = torch.clamp(effective_tau_e, self.cfg.TAU_E_MIN, self.cfg.TAU_E_MAX)

        # 6. E-I 离散代数更新
        E_new = torch.sigmoid(total_in + effective_tau_e * E_prev - self.w_ei * I_prev)
        I_new = torch.sigmoid(self.w_ie * E_new)

        # 7. 预测编码
        pred_in = torch.matmul(self.W_pred, E_prev)
        error = total_in - pred_in

        # 8. 动作输出
        action_logits = torch.matmul(self.W_out * self.M_out, E_new)

        return action_logits, E_new, I_new, error, total_in

    def apply_pc_update(self, E_prev, error, total_in, obs_t):
        """局部预测编码微调（生命周期内学习）"""
        with torch.no_grad():
            lr = self.cfg.PC_LR
            decay = self.cfg.PC_DECAY
            self.W_pred += lr * torch.outer(error, E_prev) - decay * self.W_pred
            self.W_in += lr * torch.outer(error, obs_t) - decay * self.W_in

    def save_genetic_baseline(self):
        self.baseline = {
            'W_in': self.W_in.data.clone(), 'W_rec': self.W_rec.data.clone(),
            'W_out': self.W_out.data.clone(), 'W_pred': self.W_pred.data.clone(),
            'M_in': self.M_in.clone(), 'M_rec': self.M_rec.clone(), 'M_out': self.M_out.clone(),
            'tau_e_init': self.tau_e_init.data.clone(),
            'W_hormone1': self.W_hormone1.data.clone(), 'b_hormone1': self.b_hormone1.data.clone(),
            'W_excit': self.W_excit.data.clone(), 'b_excit': self.b_excit.data.clone(),
            'W_inhib': self.W_inhib.data.clone(), 'b_inhib': self.b_inhib.data.clone(),
        }

    def restore_genetic_baseline(self):
        if self.baseline:
            self.W_in.data = self.baseline['W_in'].clone()
            self.W_rec.data = self.baseline['W_rec'].clone()
            self.W_out.data = self.baseline['W_out'].clone()
            self.W_pred.data = self.baseline['W_pred'].clone()
            self.M_in = self.baseline['M_in'].clone()
            self.M_rec = self.baseline['M_rec'].clone()
            self.M_out = self.baseline['M_out'].clone()
            self.tau_e_init.data = self.baseline['tau_e_init'].clone()
            self.W_hormone1.data = self.baseline['W_hormone1'].clone()
            self.b_hormone1.data = self.baseline['b_hormone1'].clone()
            self.W_excit.data = self.baseline['W_excit'].clone()
            self.b_excit.data = self.baseline['b_excit'].clone()
            self.W_inhib.data = self.baseline['W_inhib'].clone()
            self.b_inhib.data = self.baseline['b_inhib'].clone()


# ==========================================
# 3. 评估与进化逻辑
#    适应度标准：优先吃子数最多，其次步数最少
# ==========================================
def evaluate_individual(brain, env, render=False, max_steps=None):
    cfg = brain.cfg
    if max_steps is None:
        max_steps = cfg.MAX_STEPS

    original_baseline = None
    if not render and brain.baseline:
        original_baseline = {k: v.clone() for k, v in brain.baseline.items()}

    total_foods = []
    total_steps_list = []

    for _ in range(cfg.EVAL_EPISODES):
        brain.restore_genetic_baseline()
        brain.reset_runtime()
        obs = env.reset()
        E = torch.zeros(brain.N)
        I = torch.zeros(brain.N)

        ep_food = 0
        steps = 0
        done = False

        while not done and steps < max_steps:
            obs_t = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                E_old = E.clone()
                logits, E, I, error, total_in = brain(obs_t, E, I)
                action = torch.argmax(logits).item()

            if not render:
                brain.apply_pc_update(E_old, error, total_in, obs_t)

            next_obs, ate_food, done = env.step(action)
            if ate_food:
                ep_food += 1
            obs = next_obs
            steps += 1

        total_foods.append(ep_food)
        total_steps_list.append(steps)

    avg_food = np.mean(total_foods)
    avg_steps = np.mean(total_steps_list)

    # 软拉马克遗传：融入部分生命周期学习成果
    if not render and original_baseline:
        with torch.no_grad():
            lf = cfg.LAMARCKIAN_FACTOR
            brain.W_in.data = (1 - lf) * original_baseline['W_in'] + lf * brain.W_in.data
            brain.W_pred.data = (1 - lf) * original_baseline['W_pred'] + lf * brain.W_pred.data
            brain.save_genetic_baseline()

    if render:
        print(f"  [Render] Food: {avg_food:.1f}, Steps: {avg_steps:.1f}")

    # 返回 (吃子数, 步数) 元组，用于字典序比较
    return avg_food, avg_steps


def evolve_topology(population, metrics_list, cfg):
    """
    进化：按 (吃子数降序, 步数升序) 字典序筛选精英，
    然后柱体级交叉 + 变异（含 tau_e 初始值与激素网络）。
    """
    # 字典序排序：吃子多的优先，吃子相同则步数少的优先
    sorted_indices = sorted(
        range(len(metrics_list)),
        key=lambda i: (metrics_list[i][0], -metrics_list[i][1]),
        reverse=True
    )

    elite_idx = sorted_indices[:cfg.ELITE_SIZE]
    elites = [copy.deepcopy(population[i]) for i in elite_idx]

    new_pop = [copy.deepcopy(e) for e in elites]

    while len(new_pop) < len(population):
        p1, p2 = random.sample(elites, 2)
        child = copy.deepcopy(p1)
        N = child.N

        # --- 柱体级交叉 ---
        col_mask = torch.rand(N) > 0.5
        row_mask = col_mask.unsqueeze(1)    # (N, 1)
        col_mask_2d = col_mask.unsqueeze(0) # (1, N)
        same_p1 = row_mask & col_mask_2d
        same_p2 = (~row_mask) & (~col_mask_2d)

        with torch.no_grad():
            # W_in, M_in: (N, obs_dim) — 按行(柱体)交叉
            child.W_in.data = torch.where(col_mask.unsqueeze(1), p1.W_in.data, p2.W_in.data)
            child.M_in = torch.where(col_mask.unsqueeze(1), p1.M_in, p2.M_in)

            # W_rec, M_rec: (N, N) — 模块级交叉
            child.W_rec.data = torch.where(same_p1, p1.W_rec.data,
                                  torch.where(same_p2, p2.W_rec.data,
                                      torch.where(torch.rand_like(p1.W_rec.data) > 0.5,
                                                  p1.W_rec.data, p2.W_rec.data)))
            child.M_rec = torch.where(same_p1, p1.M_rec,
                             torch.where(same_p2, p2.M_rec,
                                 torch.where(torch.rand_like(p1.M_rec) > 0.5,
                                             p1.M_rec, p2.M_rec)))

            # W_out, M_out: (action_dim, N) — 按列(柱体)交叉
            child.W_out.data = torch.where(col_mask.unsqueeze(0), p1.W_out.data, p2.W_out.data)
            child.M_out = torch.where(col_mask.unsqueeze(0), p1.M_out, p2.M_out)

            # W_pred: (N, N) — 按行(柱体)交叉
            child.W_pred.data = torch.where(col_mask.unsqueeze(1), p1.W_pred.data, p2.W_pred.data)

            # tau_e_init: (N,) — 按柱体交叉
            child.tau_e_init.data = torch.where(col_mask, p1.tau_e_init.data, p2.tau_e_init.data)

            # 激素网络: 均匀交叉
            for attr in ['W_hormone1', 'b_hormone1', 'W_excit', 'b_excit', 'W_inhib', 'b_inhib']:
                p1_t = getattr(p1, attr).data
                p2_t = getattr(p2, attr).data
                mask = torch.rand_like(p1_t) > 0.5
                getattr(child, attr).data = torch.where(mask, p1_t, p2_t)

        # --- 变异 ---
        with torch.no_grad():
            # 拓扑变异（生长/剪切连接）
            if random.random() < cfg.TOPOLOGY_MUT_PROB:
                m_attr = random.choice(['M_in', 'M_rec', 'M_out'])
                m_tensor = getattr(child, m_attr)
                mut_mask = torch.rand_like(m_tensor) < cfg.MUT_RATE
                m_tensor[mut_mask] = 1.0 - m_tensor[mut_mask]

            # 突触权重高斯变异
            for attr in ['W_in', 'W_rec', 'W_out', 'W_pred']:
                w_tensor = getattr(child, attr).data
                noise = torch.randn_like(w_tensor) * cfg.WEIGHT_MUT_STD
                noise_mask = torch.rand_like(w_tensor) < cfg.WEIGHT_MUT_FRAC
                setattr(child, attr, nn.Parameter(w_tensor + noise * noise_mask))

            # tau_e 初始值变异
            tau_noise = torch.randn_like(child.tau_e_init.data) * cfg.TAU_E_MUT_STD
            child.tau_e_init.data = torch.clamp(
                child.tau_e_init.data + tau_noise,
                cfg.TAU_E_MIN, cfg.TAU_E_MAX
            )

            # 激素网络变异
            for attr in ['W_hormone1', 'b_hormone1', 'W_excit', 'b_excit', 'W_inhib', 'b_inhib']:
                w = getattr(child, attr).data
                noise = torch.randn_like(w) * cfg.HORMONE_MUT_STD
                mask = torch.rand_like(w) < cfg.HORMONE_MUT_FRAC
                getattr(child, attr).data = w + noise * mask

        child.save_genetic_baseline()
        new_pop.append(child)

    return new_pop


# ==========================================
# 4. 可视化
# ==========================================
def plot_history(history):
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
    plt.show()


def render_snake_game(env):
    grid = np.zeros((env.grid_size, env.grid_size, 3))
    grid[env.food[0], env.food[1]] = [1, 1, 0]
    for seg in env.body:
        if 0 <= seg[0] < env.grid_size and 0 <= seg[1] < env.grid_size:
            grid[seg[0], seg[1]] = [0, 0, 1]
    if 0 <= env.head[0] < env.grid_size and 0 <= env.head[1] < env.grid_size:
        grid[env.head[0], env.head[1]] = [1, 0, 0]
    return grid


def visualize_best_brain_play(brain, cfg, max_steps=300):
    env = SnakeEnv(grid_size=cfg.GRID_SIZE)
    brain.restore_genetic_baseline()
    brain.reset_runtime()

    obs = env.reset()
    E = torch.zeros(brain.N)
    I = torch.zeros(brain.N)

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    img = ax.imshow(render_snake_game(env), cmap='viridis', vmin=0, vmax=1)
    ax.set_title("Best Brain Playing Snake")
    ax.axis('off')

    steps = 0
    done = False

    while not done and steps < max_steps:
        obs_t = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            logits, E, I, error, total_in = brain(obs_t, E, I)
            action = torch.argmax(logits).item()

        next_obs, ate_food, done = env.step(action)
        obs = next_obs
        steps += 1

        img.set_data(render_snake_game(env))
        ax.set_title(f"Step: {steps} | Score: {len(env.body) - 2}")
        fig.canvas.draw_idle()
        plt.pause(0.1)

    print(f"\nGame Over! Final Score: {len(env.body) - 2} | Survived Steps: {steps}")
    plt.ioff()
    plt.show()


try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False
    print("Warning: python-louvain not installed. Using basic community detection.")


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

    if HAS_LOUVAIN:
        partition = community_louvain.best_partition(G.to_undirected())
    else:
        communities = nx.algorithms.community.greedy_modularity_communities(G.to_undirected())
        partition = {}
        for i, com in enumerate(communities):
            for node in com:
                partition[node] = i

    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
    community_colors = list(mcolors.TABLEAU_COLORS.values())
    node_colors = [community_colors[partition.get(node, 0) % 10] for node in G.nodes()]
    degrees = dict(G.degree())
    node_sizes = [degrees[node] * 80 + 100 for node in G.nodes()]

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray', width=0.5)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")
    plt.title("Evolved Brain Topology & Functional Modules", fontsize=16)
    plt.axis("off")

    # --- 2. 运行时内部状态 + 激素 + tau_e 热力图 ---
    brain.restore_genetic_baseline()
    brain.reset_runtime()
    obs = env.reset()
    E = torch.zeros(brain.N)
    I = torch.zeros(brain.N)
    E_history, excit_history, inhib_history, tau_history = [], [], [], []
    actions = []
    done = False
    steps = 0

    while not done and steps < 100:
        obs_t = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            E_old = E.clone()
            logits, E, I, error, total_in = brain(obs_t, E, I)
            action = torch.argmax(logits).item()

        E_history.append(E.numpy().copy())
        excit_history.append(brain.hormone_excit.numpy().copy())
        inhib_history.append(brain.hormone_inhib.numpy().copy())
        # 记录有效 tau_e
        eff_tau = brain.tau_e_init + \
            cfg.SHORT_TERM_GAIN * brain.short_term_state + \
            cfg.EXCIT_HORMONE_GAIN * brain.hormone_excit - \
            cfg.INHIB_HORMONE_GAIN * brain.hormone_inhib
        tau_history.append(torch.clamp(eff_tau, cfg.TAU_E_MIN, cfg.TAU_E_MAX).detach().numpy().copy())
        actions.append(action)
        next_obs, ate_food, done = env.step(action)
        obs = next_obs
        steps += 1

    E_history = np.array(E_history)
    excit_history = np.array(excit_history)
    inhib_history = np.array(inhib_history)
    tau_history = np.array(tau_history)

    fig, axes = plt.subplots(4, 1, figsize=(14, 16))

    # E 状态
    im0 = axes[0].imshow(E_history.T, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[0].set_title("Column Excitation (E) Over Time", fontsize=13)
    axes[0].set_ylabel("Columns")
    plt.colorbar(im0, ax=axes[0])

    # 有效 tau_e
    im1 = axes[1].imshow(tau_history.T, aspect='auto', cmap='plasma', interpolation='nearest')
    axes[1].set_title("Effective tau_e (init + short-term + hormone)", fontsize=13)
    axes[1].set_ylabel("Columns")
    plt.colorbar(im1, ax=axes[1])

    # 兴奋激素
    im2 = axes[2].imshow(excit_history.T, aspect='auto', cmap='Reds', interpolation='nearest')
    axes[2].set_title("Excitatory Hormone (small diffusion)", fontsize=13)
    axes[2].set_ylabel("Columns")
    plt.colorbar(im2, ax=axes[2])

    # 抑制激素
    im3 = axes[3].imshow(inhib_history.T, aspect='auto', cmap='Blues', interpolation='nearest')
    axes[3].set_title("Inhibitory Hormone (large diffusion)", fontsize=13)
    axes[3].set_ylabel("Columns")
    axes[3].set_xlabel("Time Steps")
    plt.colorbar(im3, ax=axes[3])

    # 标记动作切换点
    action_changes = [i for i in range(1, len(actions)) if actions[i] != actions[i - 1]]
    for ax in axes:
        for t in action_changes:
            ax.axvline(x=t, color='r', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

    # --- 3. tau_e_init 分布 ---
    plt.figure(figsize=(10, 4))
    plt.bar(range(brain.N), brain.tau_e_init.data.numpy(), color='purple', alpha=0.7)
    plt.axhline(y=cfg.BASE_TAU_E, color='gray', linestyle='--', label=f'Base tau_e={cfg.BASE_TAU_E}')
    plt.title("Evolved tau_e Initial Values per Column", fontsize=14)
    plt.xlabel("Column Index")
    plt.ylabel("tau_e init")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==========================================
# 5. 主循环
# ==========================================
if __name__ == "__main__":
    cfg = Config()
    env = SnakeEnv(grid_size=cfg.GRID_SIZE)

    print("Initializing Population...")
    population = [EIBrainRegion(cfg) for _ in range(cfg.POP_SIZE)]
    for ind in population:
        ind.save_genetic_baseline()

    history = {'gen': [], 'best_food': [], 'avg_food': [], 'best_steps': []}

    for gen in range(cfg.GENERATIONS):
        metrics = []
        for ind in population:
            m = evaluate_individual(ind, env)
            metrics.append(m)

        # 字典序选优：吃子最多 → 步数最少
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
        print(f"Gen {gen+1}/{cfg.GENERATIONS} | "
              f"BestFood: {best_food:.1f} | BestSteps: {best_steps:.1f} | AvgFood: {avg_food:.1f} | "
              f"Edges: {best_brain.M_in.sum().item():.0f}-in / "
              f"{best_brain.M_rec.sum().item():.0f}-rec / "
              f"{best_brain.M_out.sum().item():.0f}-out | "
              f"tau_e: [{best_brain.tau_e_init.min().item():.3f}, "
              f"{best_brain.tau_e_init.max().item():.3f}]")

        if gen < cfg.GENERATIONS - 1:
            population = evolve_topology(population, metrics, cfg)

    plot_history(history)

    # 选出最终最优个体
    best_idx = max(range(len(metrics)),
                   key=lambda i: (metrics[i][0], -metrics[i][1]))
    best_brain = population[best_idx]

    print("\n--- Best Brain Summary ---")
    print(f"Input connections active:   {best_brain.M_in.sum().item():.0f}/{best_brain.N * best_brain.obs_dim}")
    print(f"Internal connections active: {best_brain.M_rec.sum().item():.0f}/{best_brain.N * best_brain.N}")
    print(f"Output connections active:   {best_brain.M_out.sum().item():.0f}/{best_brain.action_dim * best_brain.N}")
    print(f"tau_e range: [{best_brain.tau_e_init.min().item():.3f}, {best_brain.tau_e_init.max().item():.3f}]")
    print(f"Hormone net L1 norm: W_hormone1={best_brain.W_hormone1.data.abs().sum().item():.4f}, "
          f"W_excit={best_brain.W_excit.data.abs().sum().item():.4f}, "
          f"W_inhib={best_brain.W_inhib.data.abs().sum().item():.4f}")

    print("\nLaunching visualization for the best individual...")
    visualize_best_brain_play(best_brain, cfg, max_steps=300)
    visualize_brain_ecosystem(best_brain, env, cfg)