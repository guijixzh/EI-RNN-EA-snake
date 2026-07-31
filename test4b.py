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
    NUM_COLUMNS = 64
    OBS_DIM = 13              # 3 食物方向 + 5 射线×2 = 13
    ACTION_DIM = 3
    INIT_DENSITY = 0.15

    # --- E-I 动力学参数 ---
    BASE_TAU_E = 0.7
    TAU_E_NOISE = 0.1
    TAU_E_MIN = 0.001
    TAU_E_MAX = 2.0
    W_EI = 2.0
    W_IE = 2.0

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

    # --- 局部预测编码学习 ---
    PC_LR = 0.001
    PC_DECAY = 0.0001
    LAMARCKIAN_FACTOR = 0.2

    # --- 动作疲劳参数（彻底重做：仅基于连续次数）---
    FATIGUE_GAIN = 0.01        # 每超过阈值一次，增加的疲劳抑制量
    FATIGUE_THRESHOLD = 4     # 允许连续转向的次数（如设为2，则第3次同方向转弯开始受惩罚）
    FATIGUE_MAX = 5.0         # 疲劳上限，防止无限增大


# ==========================================
# 1. 轻量级贪吃蛇环境（射线视野 + 无奖励设计）
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

    def _cast_ray(self, direction):
        max_dist = self.grid_size
        obstacle_dist = max_dist
        food_dist = -1

        for step in range(1, max_dist + 1):
            r = self.head[0] + direction[0] * step
            c = self.head[1] + direction[1] * step

            if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
                obstacle_dist = step
                break
            if (r, c) in self.body:
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

    def _get_obs(self):
        obs = np.zeros(13, dtype=np.float32)

        left_dir = (-self.dir[1], self.dir[0])
        right_dir = (self.dir[1], -self.dir[0])
        dx = self.food[0] - self.head[0]
        dy = self.food[1] - self.head[1]
        if (dx * self.dir[0] + dy * self.dir[1]) > 0:       obs[0] = 1.0
        if (dx * left_dir[0] + dy * left_dir[1]) > 0:       obs[1] = 1.0
        if (dx * right_dir[0] + dy * right_dir[1]) > 0:     obs[2] = 1.0

        ray_dirs = [
            left_dir,
            (self.dir[0] + left_dir[0], self.dir[1] + left_dir[1]),
            self.dir,
            (self.dir[0] + right_dir[0], self.dir[1] + right_dir[1]),
            right_dir,
        ]
        for i, rd in enumerate(ray_dirs):
            free_path, food_sig = self._cast_ray(rd)
            obs[3 + i * 2]     = free_path
            obs[3 + i * 2 + 1] = food_sig

        return obs

    def step(self, action):
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

        if self.steps_without_food > 2*len(self.body) + 20:
            return self._get_obs(), False, True

        return self._get_obs(), ate_food, False


# ==========================================
# 2. E-I 皮质柱脑区模型
#    （含 tau_e 进化 + 激素调控 + 纯次数疲劳 + 输出偏置）
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
        self.b_out = nn.Parameter(torch.zeros(self.action_dim)) # 输出偏置项

        # --- 每柱体 tau_e 初始值 ---
        self.tau_e_init = nn.Parameter(
            cfg.BASE_TAU_E + (torch.rand(self.N) * 2 - 1) * cfg.TAU_E_NOISE
        )

        # --- 激素调控前馈网络 ---
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
        self.register_buffer('consecutive_counts', torch.zeros(self.action_dim)) # 纯次数计数器

        self.w_ei = cfg.W_EI
        self.w_ie = cfg.W_IE
        self.baseline = None

    def reset_runtime(self):
        """每局开始前重置激素、短期状态与连续动作计数"""
        self.hormone_excit.zero_()
        self.hormone_inhib.zero_()
        self.short_term_state.zero_()
        self.consecutive_counts.zero_()

    def forward(self, obs_t, E_prev, I_prev):
        # 1. 外部与循环输入
        ext_in = torch.matmul(self.W_in * self.M_in, obs_t)
        rec_in = torch.matmul(self.W_rec * self.M_rec, E_prev)
        total_in = ext_in + rec_in

        # 2. 激素调控前馈网络
        hormone_input = torch.cat([E_prev, I_prev, total_in], dim=-1)
        h_hidden = torch.relu(torch.matmul(self.W_hormone1, hormone_input) + self.b_hormone1)
        excit_cmd = torch.relu(torch.matmul(self.W_excit, h_hidden) + self.b_excit)
        inhib_cmd = torch.relu(torch.matmul(self.W_inhib, h_hidden) + self.b_inhib)

        # 3. 激素沿拓扑扩散 + 长期衰减
        with torch.no_grad():
            deg = self.M_rec.sum(dim=1, keepdim=True) + 1e-8
            M_norm = self.M_rec / deg

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

        # 4. 短期状态
        self.short_term_state = self.cfg.SHORT_TERM_DECAY * self.short_term_state + \
            (1 - self.cfg.SHORT_TERM_DECAY) * E_prev

        # 5. 有效 tau_e
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

        # 8. 动作输出 (添加偏置)
        action_logits = torch.matmul(self.W_out * self.M_out, E_new) + self.b_out
        
        # 9. 动作疲劳抑制 (纯基于连续次数)
        # fatigue = max(0, 连续次数 - 阈值) * 增益
        fatigue = torch.relu(self.consecutive_counts - self.cfg.FATIGUE_THRESHOLD) * self.cfg.FATIGUE_GAIN
        fatigue = torch.clamp(fatigue, max=self.cfg.FATIGUE_MAX)
        action_logits = action_logits - fatigue

        return action_logits, E_new, I_new, error, total_in

    def update_fatigue(self, action):
        """更新连续动作计数。一旦切换动作，其他动作计数清零。"""
        with torch.no_grad():
            mask = torch.ones(self.action_dim, dtype=torch.bool)
            mask[action] = False
            self.consecutive_counts[mask] = 0.0
            self.consecutive_counts[action] += 1.0

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
            'b_out': self.b_out.data.clone(),
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
            self.b_out.data = self.baseline['b_out'].clone()
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
    total_action_counts = torch.zeros(cfg.ACTION_DIM) # 统计5局总动作分布

    for ep in range(cfg.EVAL_EPISODES):
        # 仅在第1局恢复出厂设置，后续局保留上一局学到的权重
        if ep == 0:
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

            brain.update_fatigue(action)
            total_action_counts[action] += 1

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

    # 硬性淘汰：如果只向一侧转弯，直接判定为最差适应度
    if (total_action_counts[1] > 5 or total_action_counts[2] > 5) and \
       (total_action_counts[1] == 0 or total_action_counts[2] == 0):
        avg_food = 0
        avg_steps = 99999  # 强制在字典序排序中垫底

    # 软拉马克遗传：融入5局累积的生命周期学习成果
    if not render and original_baseline:
        with torch.no_grad():
            lf = cfg.LAMARCKIAN_FACTOR
            brain.W_in.data = (1 - lf) * original_baseline['W_in'] + lf * brain.W_in.data
            brain.W_pred.data = (1 - lf) * original_baseline['W_pred'] + lf * brain.W_pred.data
            brain.save_genetic_baseline()

    if render:
        print(f"  [Render] Food: {avg_food:.1f}, Steps: {avg_steps:.1f}")

    return avg_food, avg_steps


def evolve_topology(population, metrics_list, cfg):
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

        col_mask = torch.rand(N) > 0.5
        row_mask = col_mask.unsqueeze(1)
        col_mask_2d = col_mask.unsqueeze(0)
        same_p1 = row_mask & col_mask_2d
        same_p2 = (~row_mask) & (~col_mask_2d)

        with torch.no_grad():
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

            child.W_pred.data = torch.where(col_mask.unsqueeze(1), p1.W_pred.data, p2.W_pred.data)

            child.tau_e_init.data = torch.where(col_mask, p1.tau_e_init.data, p2.tau_e_init.data)

            for attr in ['W_hormone1', 'b_hormone1', 'W_excit', 'b_excit', 'W_inhib', 'b_inhib', 'b_out']:
                p1_t = getattr(p1, attr).data
                p2_t = getattr(p2, attr).data
                mask = torch.rand_like(p1_t) > 0.5
                getattr(child, attr).data = torch.where(mask, p1_t, p2_t)

        with torch.no_grad():
            if random.random() < cfg.TOPOLOGY_MUT_PROB:
                m_attr = random.choice(['M_in', 'M_rec', 'M_out'])
                m_tensor = getattr(child, m_attr)
                mut_mask = torch.rand_like(m_tensor) < cfg.MUT_RATE
                m_tensor[mut_mask] = 1.0 - m_tensor[mut_mask]

            for attr in ['W_in', 'W_rec', 'W_out', 'W_pred', 'b_out']:
                w_tensor = getattr(child, attr).data
                noise = torch.randn_like(w_tensor) * cfg.WEIGHT_MUT_STD
                noise_mask = torch.rand_like(w_tensor) < cfg.WEIGHT_MUT_FRAC
                setattr(child, attr, nn.Parameter(w_tensor + noise * noise_mask))

            tau_noise = torch.randn_like(child.tau_e_init.data) * cfg.TAU_E_MUT_STD
            child.tau_e_init.data = torch.clamp(
                child.tau_e_init.data + tau_noise,
                cfg.TAU_E_MIN, cfg.TAU_E_MAX
            )

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

        brain.update_fatigue(action)

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


# ==========================================
# 4b. 拓扑可视化 - 方案 A/B（更直观的连接展示）
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
    boundaries = []  # 每个分组在 order 中的起始索引
    for c in comms:
        cols = sorted(cols_by_comm[c], key=lambda i: tau[i])
        boundaries.append(len(order))
        order.extend(cols)
    return np.array(order), comm_of_col, comms, np.array(boundaries)


def plot_topology_layered(brain, cfg, partition):
    """方案 A：三层流水线拓扑图（输入层 → 柱层 → 输出层）

    - 信息流方向：左 → 右，一目了然
    - 柱层在中间二维区域内展开：每个功能社区占据一个"岛屿"区块，
      块内节点用 spring_layout 二维散布，直观展示内部环路形态
    - 柱节点：外环=社区，内填充=tau_e_init (plasma)，白点=递归入度
    - 边：绿色=输入→柱，品红=柱→输出；递归边红色=兴奋(正W)/蓝色=抑制(负W)
    - 线宽与透明度 ∝ |W|，递归边只显示 top-K 强连接防毛团
    """
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch, Patch, Rectangle
    from matplotlib.lines import Line2D

    N = brain.N
    tau = brain.tau_e_init.detach().numpy()
    order, comm_of_col, comms, boundaries = _community_order(brain, partition)

    W_rec_np = brain.W_rec.detach().numpy()
    M_rec_np = brain.M_rec.numpy()

    # --- 1. 社区锚点：按 2D 网格排列，覆盖柱层中间区域 ---
    n_comms = len(comms)
    ncols = max(1, int(np.ceil(np.sqrt(n_comms))))
    nrows = max(1, int(np.ceil(n_comms / ncols)))
    anchor_step_x = 0.90
    anchor_step_y = 0.90
    anchor_x0 = 1.30          # 柱层区域左边界
    total_height = nrows * anchor_step_y
    anchor = {}
    for idx_blk, comm_id in enumerate(comms):
        r = idx_blk // ncols
        c = idx_blk % ncols
        anchor[int(comm_id)] = (anchor_x0 + c * anchor_step_x,
                                total_height - (r + 0.5) * anchor_step_y)

    # --- 2. 块内 spring_layout：社区内部节点 2D 展开 ---
    block_radius = 0.34 * anchor_step_x
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

    # --- 输入/输出层 y 坐标（覆盖同一垂直范围） ---
    OUT_X = anchor_x0 + ncols * anchor_step_x + 0.5   # 输出层 x 坐标
    y_in = {j: (j + 1.0) / (brain.obs_dim + 1.0) * total_height for j in range(brain.obs_dim)}
    y_out = {i: (i + 1.0) / (brain.action_dim + 1.0) * total_height for i in range(brain.action_dim)}

    community_colors = list(mcolors.TABLEAU_COLORS.values())
    tau_norm = plt.Normalize(cfg.TAU_E_MIN, cfg.TAU_E_MAX)
    in_degree = brain.M_rec.sum(dim=0).numpy()  # 每柱递归入度

    fig, ax = plt.subplots(figsize=(17, max(8, total_height + 2.0)))

    # --- 边：输入 → 柱（直线，绿色，强度∝线宽） ---
    W_in_abs = _norm_edge_weights(brain.W_in.detach().numpy())
    for i in range(N):
        for j in range(brain.obs_dim):
            if brain.M_in[i, j] > 0:
                nw = W_in_abs[i, j]
                x1, y1 = pos_col[i]
                ax.plot([0.0, x1], [y_in[j], y1], color='green',
                        lw=0.3 + 2.2 * nw, alpha=0.15 + 0.6 * nw,
                        solid_capstyle='round', zorder=1)

    # --- 边：柱 → 输出（直线，品红，强度∝线宽） ---
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
            curve = 0.12 + 0.10 * dist          # 跨社区长距离边明显外凸
            mid_x = (x0p + x1p) / 2.0
            mid_y = (y0p + y1p) / 2.0
            ctrl = (mid_x + curve, mid_y + curve)
            verts = [(x0p, y0p), ctrl, (x1p, y1p)]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            ax.add_patch(PathPatch(Path(verts, codes), facecolor='none',
                                   edgecolor='red' if w >= 0 else 'blue',
                                   lw=0.3 + 2.0 * nw, alpha=0.10 + 0.55 * nw, zorder=2))

    # --- 柱节点：外环=社区 + 内圆=tau_e + 白点=入度 ---
    for col in range(N):
        x, y = pos_col[col]
        comm = comm_of_col[col]
        ring_color = community_colors[comm % len(community_colors)] if comm >= 0 else 'gray'
        tau_color = plt.cm.plasma(tau_norm(tau[col]))
        ax.scatter(x, y, s=78, color=ring_color, edgecolors='black', linewidths=0.8, zorder=3)
        ax.scatter(x, y, s=48, color=tau_color, zorder=4)
        in_n = int(in_degree[col])
        if in_n > 0:
            ax.scatter(x, y, s=4.0 + 2.0 * min(in_n, 20), color='white', zorder=5)

    # --- 输入 / 输出节点 ---
    for j in range(brain.obs_dim):
        ax.scatter(0.0, y_in[j], marker='s', s=140, color='limegreen', edgecolors='black', zorder=3)
        ax.text(0.0, y_in[j], f"In_{j}", fontsize=7, ha='right', va='center')
    act_names = ['Fwd', 'Left', 'Right']
    for i in range(brain.action_dim):
        ax.scatter(OUT_X, y_out[i], marker='D', s=160, color='orange', edgecolors='black', zorder=3)
        ax.text(OUT_X, y_out[i], f"Out_{i}\n{act_names[i]}", fontsize=8, ha='left', va='center')

    # --- 社区标注（锚点上方） ---
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
        Patch(facecolor='limegreen', edgecolor='black', label='Input node (13 obs)'),
        Patch(facecolor='orange', edgecolor='black', label='Output node (3 actions)'),
        Patch(facecolor='white', edgecolor='black', label='Column node (ring=community, fill=tau_e, dot=in-degree)'),
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
    """方案 B：连接矩阵仪表盘（M_in / M_rec / M_out 热力图）

    柱节点均按「社区 → tau_e」重排，直观显示功能模块方块结构。
    M_rec 用 RdBu：红=兴奋(正 W)，蓝=抑制(负 W)，白=无连接。
    """
    N = brain.N
    order, comm_of_col, comms, boundaries = _community_order(brain, partition)

    # 权重×掩码矩阵
    W_in_map = (brain.W_in * brain.M_in).detach().numpy()
    W_rec_map = (brain.W_rec * brain.M_rec).detach().numpy()
    W_out_map = (brain.W_out * brain.M_out).detach().numpy()

    M_in_sorted = W_in_map[order, :]
    M_rec_sorted = W_rec_map[np.ix_(order, order)]
    M_out_sorted = W_out_map[:, order]

    rec_vmax = float(np.abs(M_rec_sorted).max()) + 1e-8

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))

    # --- M_in ---
    im0 = axes[0].imshow(M_in_sorted, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[0].set_title("Input W_in (Column × Feature)", fontsize=12)
    axes[0].set_xlabel("Input feature (0-12)")
    axes[0].set_ylabel("Column (community/τ sorted)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # --- M_rec ---
    im1 = axes[1].imshow(M_rec_sorted, aspect='auto', cmap='RdBu', interpolation='nearest',
                         vmin=-rec_vmax, vmax=rec_vmax)
    axes[1].set_title("Recurrent W_rec (Target × Source)\nE=red / I=blue", fontsize=12)
    axes[1].set_xlabel("Source column")
    axes[1].set_ylabel("Target column")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # 社区分隔线（白色细线）
    for b in boundaries[1:]:
        line_pos = int(b) - 0.5
        axes[1].axhline(y=line_pos, color='white', lw=1.0, alpha=0.9)
        axes[1].axvline(x=line_pos, color='white', lw=1.0, alpha=0.9)

    # --- M_out ---
    im2 = axes[2].imshow(M_out_sorted, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[2].set_title("Output W_out (Action × Column)", fontsize=12)
    axes[2].set_xlabel("Column (community/τ sorted)")
    axes[2].set_ylabel("Action (0=Fwd, 1=Left, 2=Right)")
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
        import community as community_louvain
        partition = community_louvain.best_partition(G.to_undirected())
    except ImportError:
        communities = nx.algorithms.community.greedy_modularity_communities(G.to_undirected())
        partition = {}
        for i, com in enumerate(communities):
            for node in com:
                partition[node] = i

    # --- 方案 A：三层流水线拓扑图（方向 / E-I 符号 / 强度 / 模块） ---
    plot_topology_layered(brain, cfg, partition)

    # --- 方案 B：连接矩阵仪表盘（M_in / M_rec / M_out 精确视图） ---
    plot_connectivity_matrices(brain, cfg, partition)

    # --- 2. 运行时内部状态 + 动作潜力热力图 ---
    brain.restore_genetic_baseline()
    brain.reset_runtime()
    obs = env.reset()
    E = torch.zeros(brain.N)
    I = torch.zeros(brain.N)
    E_history, excit_history, inhib_history, tau_history = [], [], [], []
    logits_history = []  # NEW: 记录叠加了疲劳的动作净潜力
    actions = []
    done = False
    steps = 0

    while not done and steps < 100:
        obs_t = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            E_old = E.clone()
            logits, E, I, error, total_in = brain(obs_t, E, I)
            action = torch.argmax(logits).item()

        # 记录最终的 logits（它已经包含了疲劳的惩罚项）
        logits_history.append(logits.numpy().copy())
        
        brain.update_fatigue(action)

        E_history.append(E.numpy().copy())
        excit_history.append(brain.hormone_excit.numpy().copy())
        inhib_history.append(brain.hormone_inhib.numpy().copy())
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

    # ---- NEW: 动作净潜力 (叠加疲劳) 热力图 ----
    # 使用 RdBu (红蓝) 颜色映射，0为中心。蓝色代表正向潜力，红色代表被抑制。
    im4 = axes[4].imshow(logits_history.T, aspect='auto', cmap='RdBu', interpolation='nearest', vmin=-3, vmax=3)
    axes[4].set_title("Net Action Logits (Base Logits - Fatigue Penalty)", fontsize=13)
    axes[4].set_ylabel("Action (0=Fwd, 1=Left, 2=Right)")
    axes[4].set_xlabel("Time Steps")
    plt.colorbar(im4, ax=axes[4])

    # 在图中用黑色星号标记实际被选中的动作
    for t, act in enumerate(actions):
        axes[4].plot(t, act, 'k*', markersize=10, markeredgecolor='yellow', markeredgewidth=0.5)

    action_changes = [i for i in range(1, len(actions)) if actions[i] != actions[i - 1]]
    for ax in axes:
        for t in action_changes:
            ax.axvline(x=t, color='lime', linestyle='--', alpha=0.5)

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
    print(f"  Observation: 5 rays x 2 + 3 food direction = {cfg.OBS_DIM} dim")
    print(f"  Action fatigue: gain={cfg.FATIGUE_GAIN}, threshold={cfg.FATIGUE_THRESHOLD}, max={cfg.FATIGUE_MAX}")
    population = [EIBrainRegion(cfg) for _ in range(cfg.POP_SIZE)]
    for ind in population:
        ind.save_genetic_baseline()

    history = {'gen': [], 'best_food': [], 'avg_food': [], 'best_steps': []}

    for gen in range(cfg.GENERATIONS):
        metrics = []
        for ind in population:
            m = evaluate_individual(ind, env)
            metrics.append(m)

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

    best_idx = max(range(len(metrics)),
                   key=lambda i: (metrics[i][0], -metrics[i][1]))
    best_brain = population[best_idx]

    print("\n--- Best Brain Summary ---")
    print(f"Input connections active:   {best_brain.M_in.sum().item():.0f}/{best_brain.N * best_brain.obs_dim}")
    print(f"Internal connections active: {best_brain.M_rec.sum().item():.0f}/{best_brain.N * best_brain.N}")
    print(f"Output connections active:   {best_brain.M_out.sum().item():.0f}/{best_brain.action_dim * best_brain.N}")
    print(f"tau_e range: [{best_brain.tau_e_init.min().item():.3f}, {best_brain.tau_e_init.max().item():.3f}]")
    print(f"Evolved Output Bias: {best_brain.b_out.data.numpy()}")
    print(f"Hormone net L1 norm: W_hormone1={best_brain.W_hormone1.data.abs().sum().item():.4f}, "
          f"W_excit={best_brain.W_excit.data.abs().sum().item():.4f}, "
          f"W_inhib={best_brain.W_inhib.data.abs().sum().item():.4f}")

    print("\nLaunching visualization for the best individual...")
    visualize_best_brain_play(best_brain, cfg, max_steps=300)
    visualize_brain_ecosystem(best_brain, env, cfg)