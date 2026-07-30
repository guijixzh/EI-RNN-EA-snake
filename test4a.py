import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import time
import networkx as nx
import matplotlib.colors as mcolors

# ==========================================
# 1. 轻量级贪吃蛇环境 (保持不变)
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
            self.food = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if self.food not in self.body:
                break

    def _get_obs(self):
        obs = np.zeros(6, dtype=np.float32)
        left_dir = (-self.dir[1], self.dir[0])
        right_dir = (self.dir[1], -self.dir[0])
        for i, d in enumerate([self.dir, left_dir, right_dir]):
            next_pos = (self.head[0] + d[0], self.head[1] + d[1])
            if next_pos[0] < 0 or next_pos[0] >= self.grid_size or next_pos[1] < 0 or next_pos[1] >= self.grid_size or next_pos in self.body:
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
            (forward[0] + left[0], forward[1] + left[1]), forward, (forward[0] + right[0], forward[1] + right[1]),
            left, (0, 0), right,
            (back[0] + left[0], back[1] + left[1]), back, (back[0] + right[0], back[1] + right[1])
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
        if action == 1: self.dir = (-self.dir[1], self.dir[0])
        elif action == 2: self.dir = (self.dir[1], -self.dir[0])
        prev_head = self.head
        next_head = (self.head[0] + self.dir[0], self.head[1] + self.dir[1])
        self.steps += 1
        self.steps_without_food += 1
        will_eat = (next_head == self.food)
        body_to_check = self.body if will_eat else self.body[:-1]
        if next_head[0] < 0 or next_head[0] >= self.grid_size or next_head[1] < 0 or next_head[1] >= self.grid_size or next_head in body_to_check:
            return self._get_obs(), -1000, True
        self.body.insert(0, next_head)
        self.head = next_head
        reward = -0.1
        prev_dist = abs(prev_head[0] - self.food[0]) + abs(prev_head[1] - self.food[1])
        curr_dist = abs(next_head[0] - self.food[0]) + abs(next_head[1] - self.food[1])
        if curr_dist < prev_dist: reward += 0.5
        if self.head == self.food:
            self.food_count += 1
            reward = 20
            self.steps_without_food = 0
            self._place_food()
        else:
            self.body.pop()
        if self.steps_without_food > len(self.body) + 20:
            reward = -10
            return self._get_obs(), reward, True
        return self._get_obs(), reward, False


# ==========================================
# 2. 激素控制器 (前馈网络)  [新增]
# ==========================================
class HormoneController(nn.Module):
    """
    前馈网络：接收全局脑状态 → 输出每个柱体的 (兴奋性激素, 抑制性激素) 释放量。
    - 输入: [mean_E, mean_I, mean_|error|, std_E, mean_tau, active_fraction]
    - 输出: 2*N 维 (N 个柱体各得 H_exc, H_inh)
    两种激素具有不同的刺激增益和扩散系数 (在 EIBrainRegion 中配置)。
    """
    def __init__(self, num_columns, state_dim=6, hidden_dim=32):
        super().__init__()
        self.num_columns = num_columns
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_columns * 2)

    def forward(self, global_state):
        h = torch.relu(self.fc1(global_state))
        out = torch.sigmoid(self.fc2(h))
        H_exc = out[:self.num_columns]
        H_inh = out[self.num_columns:]
        return H_exc, H_inh


# ==========================================
# 3. E-I 皮质柱与脑区模型 (增强版)
# ==========================================
class EIBrainRegion(nn.Module):
    def __init__(self, num_columns=64, obs_dim=24, action_dim=3, init_density=0.15):
        super().__init__()
        self.N = num_columns
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # --- 基因型：拓扑掩码 ---
        self.M_in = (torch.rand(num_columns, obs_dim) < init_density).float()
        self.M_rec = (torch.rand(num_columns, num_columns) < init_density).float()
        torch.diagonal(self.M_rec).zero_()
        self.M_out = (torch.rand(action_dim, num_columns) < init_density).float()

        # --- 表现型：突触权重 ---
        self.W_in = nn.Parameter(torch.randn(num_columns, obs_dim) * 0.1)
        self.W_rec = nn.Parameter(torch.randn(num_columns, num_columns) * 0.05)
        self.W_out = nn.Parameter(torch.randn(action_dim, num_columns) * 0.1)
        self.W_pred = nn.Parameter(torch.randn(num_columns, num_columns) * 0.1)

        # --- [新增1] 每柱体 tau_e 随机初始化 ---
        self.tau_e_base = nn.Parameter(torch.rand(num_columns) * 0.6 + 0.4)  # ∈ [0.4, 1.0]

        # --- [新增3] 价值头 (用于 W_out 的预测编码 / TD 学习) ---
        self.W_value = nn.Parameter(torch.randn(num_columns) * 0.05)

        # --- [新增2] 激素控制器 ---
        self.hormone_ctrl = HormoneController(num_columns, state_dim=6, hidden_dim=32)

        # --- 激素状态 (长期，跨时间步持续) ---
        self.H_exc_state = torch.zeros(num_columns)
        self.H_inh_state = torch.zeros(num_columns)

        # --- 激素超参数：两种激素刺激强度与扩散程度不同 ---
        self.h_exc_gain = 1.2       # 兴奋性激素：温和增益
        self.h_inh_gain = 2.0       # 抑制性激素：强抑制
        self.h_exc_diffusion = 0.4  # 兴奋性激素：广泛扩散 (类比谷氨酸溢出)
        self.h_inh_diffusion = 0.1  # 抑制性激素：局部作用 (类比 GABA 局部调控)
        self.hormone_decay = 0.95   # 慢衰减 → 长期影响

        # E-I 固定参数
        self.w_ei = 2.0
        self.w_ie = 2.0

        # 扩散核 (基于拓扑)
        self.register_buffer('diffusion_kernel', self._build_diffusion_kernel())

        self.baseline = None

    def _build_diffusion_kernel(self):
        """基于 M_rec 拓扑构建对称扩散核 (行归一化)。"""
        A = self.M_rec.clone()
        A_sym = (A + A.T) / 2.0
        row_sum = A_sym.sum(dim=1, keepdim=True) + 1e-8
        return A_sym / row_sum

    def reset_runtime_state(self):
        """每局游戏开始时重置瞬态状态。"""
        self.H_exc_state = torch.zeros(self.N)
        self.H_inh_state = torch.zeros(self.N)

    def _get_global_state(self, E, I, error, tau_eff):
        """提取 6 维全局状态供激素控制器使用。"""
        gs = torch.zeros(6)
        gs[0] = E.mean()
        gs[1] = I.mean()
        gs[2] = error.abs().mean()
        gs[3] = E.std()
        gs[4] = tau_eff.mean()
        gs[5] = (E > 0.5).float().mean()
        return gs

    def _compute_tau_eff(self, E_prev):
        """
        有效 tau_e = tau_e_base × 短期调制 × 长期(激素)调制
        - 短期: 当前 E 状态影响 (高活跃 → 略降 tau，防止过兴奋)
        - 长期: 激素状态影响 (兴奋激素 ↑tau → 记忆巩固; 抑制激素 ↓tau)
        """
        short_mod = 1.0 - 0.3 * (E_prev - 0.5)               # ∈ ~[0.85, 1.15]
        long_mod = 1.0 + 0.4 * self.H_exc_state - 0.4 * self.H_inh_state
        long_mod = torch.clamp(long_mod, 0.3, 2.0)
        tau_eff = self.tau_e_base * short_mod * long_mod
        return torch.clamp(tau_eff, 0.05, 0.9)

    def _diffuse_hormones(self):
        """沿拓扑网络扩散两种激素 (扩散系数不同)。"""
        self.H_exc_state = (1 - self.h_exc_diffusion) * self.H_exc_state + \
                           self.h_exc_diffusion * torch.matmul(self.diffusion_kernel, self.H_exc_state)
        self.H_inh_state = (1 - self.h_inh_diffusion) * self.H_inh_state + \
                           self.h_inh_diffusion * torch.matmul(self.diffusion_kernel, self.H_inh_state)

    def forward(self, obs_t, E_prev, I_prev):
        # 1. 有效 tau_e (受短期状态 + 长期激素影响)
        tau_eff = self._compute_tau_eff(E_prev)

        # 2. 外部 & 循环输入
        ext_in = torch.matmul(self.W_in * self.M_in, obs_t)
        rec_in = torch.matmul(self.W_rec * self.M_rec, E_prev)
        total_in = ext_in + rec_in

        # 3. 激素调制兴奋性 (两种激素刺激强度不同)
        exc_mod = 1.0 + self.h_exc_gain * self.H_exc_state
        inh_mod = 1.0 + self.h_inh_gain * self.H_inh_state
        modulated_in = total_in * exc_mod / inh_mod

        # 4. E-I 离散更新 (使用每柱体独立 tau_e)
        E_new = torch.sigmoid(modulated_in + tau_eff * E_prev - self.w_ei * I_prev)
        I_new = torch.sigmoid(self.w_ie * E_new)

        # 5. 预测编码：预测其他柱体的输入
        pred_in = torch.matmul(self.W_pred, E_prev)
        error = total_in - pred_in

        # 6. 动作输出 & 状态价值
        action_logits = torch.matmul(self.W_out * self.M_out, E_new)
        value = torch.dot(self.W_value, E_new)

        # 7. 激素控制器：根据全局状态决定激素释放
        global_state = self._get_global_state(E_new, I_new, error, tau_eff)
        H_exc_release, H_inh_release = self.hormone_ctrl(global_state)

        # 8. 激素状态更新 (慢积分 → 长期影响)
        self.H_exc_state = self.hormone_decay * self.H_exc_state + (1 - self.hormone_decay) * H_exc_release
        self.H_inh_state = self.hormone_decay * self.H_inh_state + (1 - self.hormone_decay) * H_inh_release

        # 9. 沿拓扑扩散
        self._diffuse_hormones()

        return action_logits, E_new, I_new, error, total_in, value, tau_eff

    # --- 局部预测编码：输入侧 (W_pred, W_in) ---
    def apply_pc_update_input(self, E_prev, error, obs_t, lr=0.001, decay=0.0001):
        with torch.no_grad():
            self.W_pred += lr * torch.outer(error, E_prev) - decay * self.W_pred
            self.W_in += lr * torch.outer(error, obs_t) - decay * self.W_in

    # --- [新增3] 预测编码：输出侧 (W_out, W_value) 基于 TD 误差 ---
    def apply_pc_update_output(self, E_prev, action, reward, value, next_value, done,
                               lr_out=0.0001, lr_value=0.002, gamma=0.95, decay=0.0001):
        with torch.no_grad():
            target = reward + (gamma * next_value if not done else 0.0)
            td_error = target - value
            
            # 修复：td_error 此时是 Python float，使用原生 max/min 进行裁剪
            td_error = max(-10.0, min(10.0, td_error))

            # W_out: TD 误差驱动，强化/抑制所采取的动作
            action_onehot = torch.zeros(self.action_dim)
            action_onehot[action] = 1.0
            update = lr_out * td_error * action_onehot.unsqueeze(1) * E_prev.unsqueeze(0)
            self.W_out.data += update * self.M_out - decay * self.W_out.data

            # W_value: 向 TD 目标靠拢
            self.W_value.data -= lr_value * (-td_error) * E_prev  # gradient of ½(target-V)²
            self.W_value.data -= decay * self.W_value.data

            # tau_e_base 慢速适应：受激素长期暴露影响 (拉马克可遗传)
            tau_lr = 0.0008
            self.tau_e_base.data += tau_lr * (self.H_exc_state - self.H_inh_state) * 0.1
            self.tau_e_base.data = torch.clamp(self.tau_e_base.data, 0.1, 2.0)

    def save_genetic_baseline(self):
        self.baseline = {
            'W_in': self.W_in.data.clone(),
            'W_rec': self.W_rec.data.clone(),
            'W_out': self.W_out.data.clone(),
            'W_pred': self.W_pred.data.clone(),
            'W_value': self.W_value.data.clone(),
            'tau_e_base': self.tau_e_base.data.clone(),
            'M_in': self.M_in.clone(),
            'M_rec': self.M_rec.clone(),
            'M_out': self.M_out.clone(),
            'diffusion_kernel': self.diffusion_kernel.clone(),
            'hormone_ctrl': {k: v.clone() for k, v in self.hormone_ctrl.state_dict().items()},
        }

    def restore_genetic_baseline(self):
        if self.baseline:
            self.W_in.data = self.baseline['W_in'].clone()
            self.W_rec.data = self.baseline['W_rec'].clone()
            self.W_out.data = self.baseline['W_out'].clone()
            self.W_pred.data = self.baseline['W_pred'].clone()
            self.W_value.data = self.baseline['W_value'].clone()
            self.tau_e_base.data = self.baseline['tau_e_base'].clone()
            self.M_in = self.baseline['M_in'].clone()
            self.M_rec = self.baseline['M_rec'].clone()
            self.M_out = self.baseline['M_out'].clone()
            self.diffusion_kernel = self.baseline['diffusion_kernel'].clone()
            self.hormone_ctrl.load_state_dict(self.baseline['hormone_ctrl'])
            self.reset_runtime_state()


# ==========================================
# 4. 进化算法与评估逻辑
# ==========================================
def evaluate_individual(brain, env, render=False, max_steps=500):
    original_baseline = copy.deepcopy(brain.baseline) if brain.baseline else None

    total_rewards = []
    total_foods = []

    for _ in range(5):
        brain.restore_genetic_baseline()
        obs = env.reset()
        E = torch.zeros(brain.N)
        I = torch.zeros(brain.N)

        ep_reward = 0
        ep_food = 0
        done = False
        steps = 0

        # 延迟 TD: 存储上一步数据
        prev_td_data = None  # (E_prev, action, reward, value, done)

        while not done and steps < max_steps:
            obs_t = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                E_old = E.clone()
                logits, E, I, error, total_in, value, tau_eff = brain(obs_t, E, I)
                action = torch.argmax(logits).item()

            next_obs, reward, done = env.step(action)

            # --- 输入侧预测编码 (当前步) ---
            brain.apply_pc_update_input(E_old, error, obs_t)

            # --- 输出侧 TD 更新 (用上一步数据，当前 value 作为 next_value) ---
            if prev_td_data is not None:
                p_Eprev, p_action, p_reward, p_value, p_done = prev_td_data
                brain.apply_pc_update_output(p_Eprev, p_action, p_reward, p_value,
                                             value.item(), p_done)

            prev_td_data = (E_old, action, reward, value.item(), done)

            if reward >= 10:
                ep_food += 1
            ep_reward += reward
            obs = next_obs
            steps += 1

        # 处理最后一步 (done=True, next_value=0)
        if prev_td_data is not None:
            p_Eprev, p_action, p_reward, p_value, p_done = prev_td_data
            brain.apply_pc_update_output(p_Eprev, p_action, p_reward, p_value, 0.0, True)

        total_rewards.append(ep_reward)
        total_foods.append(ep_food)

    avg_reward = np.mean(total_rewards)
    avg_food = np.mean(total_foods)

    # --- [新增3] 增强拉马克遗传：40% 知识回传 (原 20%)，且涵盖所有可塑参数 ---
    # --- 【修复】调整拉马克遗传：只回传内部表征和激素参数，不回传 W_out 和 W_value ---
    if not render and original_baseline:
        with torch.no_grad():
            lf = 0.2  # Lamarckian factor
            # 回传感知与预测权重
            brain.W_in.data = (1 - lf) * original_baseline['W_in'] + lf * brain.W_in.data
            brain.W_pred.data = (1 - lf) * original_baseline['W_pred'] + lf * brain.W_pred.data
            # 回传激素相关参数
            brain.tau_e_base.data = (1 - lf) * original_baseline['tau_e_base'] + lf * brain.tau_e_base.data
            for k in brain.hormone_ctrl.state_dict():
                orig = original_baseline['hormone_ctrl'][k]
                curr = brain.hormone_ctrl.state_dict()[k]
                brain.hormone_ctrl.state_dict()[k].copy_((1 - lf) * orig + lf * curr)
            
            # 【注意】千万不要回传 W_out 和 W_value！
            # 生命周期内的 TD 学习很容易陷入局部最优（转圈保命），
            # 如果回传会污染整个种群的进化方向。
            brain.W_out.data = original_baseline['W_out']
            brain.W_value.data = original_baseline['W_value']
            
            brain.save_genetic_baseline()
    
    fitness = avg_food + avg_reward / 100

    if render:
        print(f"  [Render] Food: {avg_food:.1f}, Reward: {avg_reward:.1f}, Fitness: {fitness:.1f}")

    return fitness


def evolve_topology(population, fitnesses, elite_size=64, mut_rate=0.05):
    elite_idx = np.argsort(fitnesses)[-elite_size:]
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
            # W_rec, M_rec
            child_W_rec = torch.where(same_p1, p1.W_rec.data,
                              torch.where(same_p2, p2.W_rec.data,
                                  torch.where(torch.rand_like(p1.W_rec.data) > 0.5, p1.W_rec.data, p2.W_rec.data)))
            child.W_rec.data = child_W_rec
            child.M_rec = torch.where(same_p1, p1.M_rec,
                              torch.where(same_p2, p2.M_rec,
                                  torch.where(torch.rand_like(p1.M_rec) > 0.5, p1.M_rec, p2.M_rec)))

            # W_in, M_in
            child.W_in.data = torch.where(col_mask.unsqueeze(1), p1.W_in.data, p2.W_in.data)
            child.M_in = torch.where(col_mask.unsqueeze(1), p1.M_in, p2.M_in)

            # W_out, M_out
            child.W_out.data = torch.where(col_mask.unsqueeze(0), p1.W_out.data, p2.W_out.data)
            child.M_out = torch.where(col_mask.unsqueeze(0), p1.M_out, p2.M_out)

            # W_pred
            child.W_pred.data = torch.where(col_mask.unsqueeze(1), p1.W_pred.data, p2.W_pred.data)

            # [新增1] tau_e_base (1D, 按柱体交叉)
            child.tau_e_base.data = torch.where(col_mask, p1.tau_e_base.data, p2.tau_e_base.data)

            # [新增3] W_value (1D, 按柱体交叉)
            child.W_value.data = torch.where(col_mask, p1.W_value.data, p2.W_value.data)

            # [新增2] 激素控制器权重交叉 (均匀混合)
            for k in child.hormone_ctrl.state_dict():
                child.hormone_ctrl.state_dict()[k].copy_(
                    0.5 * p1.hormone_ctrl.state_dict()[k] + 0.5 * p2.hormone_ctrl.state_dict()[k])

        # --- 变异 ---
        with torch.no_grad():
            if random.random() < 0.05:
                m_attr = random.choice(['M_in', 'M_rec', 'M_out'])
                m_tensor = getattr(child, m_attr)
                mut_mask = torch.rand_like(m_tensor) < mut_rate
                m_tensor[mut_mask] = 1.0 - m_tensor[mut_mask]
                if m_attr == 'M_rec':
                    torch.diagonal(child.M_rec).zero_()

            # 重建扩散核以匹配可能变异的 M_rec
            child.diffusion_kernel = child._build_diffusion_kernel()

            # 权重高斯变异
            for attr in ['W_in', 'W_rec', 'W_out', 'W_pred', 'W_value', 'tau_e_base']:
                w_tensor = getattr(child, attr)
                noise = torch.randn_like(w_tensor) * 0.1
                noise_mask = torch.rand_like(w_tensor) < 0.2
                setattr(child, attr, nn.Parameter(w_tensor.data + noise * noise_mask))

            # 激素控制器变异
            for k in child.hormone_ctrl.state_dict():
                child.hormone_ctrl.state_dict()[k].add_(
                    torch.randn_like(child.hormone_ctrl.state_dict()[k]) * 0.05)

        child.save_genetic_baseline()
        new_pop.append(child)

    return new_pop


# ==========================================
# 5. 可视化
# ==========================================
def plot_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history['gen'], history['best'], label='Best Fitness', color='red', marker='o')
    plt.plot(history['gen'], history['avg'], label='Avg Fitness', color='blue', alpha=0.6)
    plt.title("Evolution Progress (EI Columns + Hormones + Predictive W_out + Enhanced Lamarckian)")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True, alpha=0.3)
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


def visualize_best_brain_play(brain, grid_size=10, max_steps=300):
    env = SnakeEnv(grid_size=grid_size)
    brain.restore_genetic_baseline()
    obs = env.reset()
    E = torch.zeros(brain.N)
    I = torch.zeros(brain.N)

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    img = ax.imshow(render_snake_game(env), cmap='viridis', vmin=0, vmax=1)
    ax.set_title("Best Brain Playing Snake")
    ax.axis('off')

    total_reward = 0
    steps = 0
    done = False

    while not done and steps < max_steps:
        obs_t = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            logits, E, I, error, total_in, value, tau_eff = brain(obs_t, E, I)
            action = torch.argmax(logits).item()
        next_obs, reward, done = env.step(action)
        total_reward += reward
        obs = next_obs
        steps += 1
        img.set_data(render_snake_game(env))
        ax.set_title(f"Step: {steps} | Score: {len(env.body) - 2} | Reward: {total_reward:.1f}")
        fig.canvas.draw_idle()
        plt.pause(0.1)

    print(f"\nGame Over! Score: {len(env.body) - 2} | Reward: {total_reward:.1f} | Steps: {steps}")
    plt.ioff()
    plt.show()


try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False


def visualize_brain_ecosystem(brain, env):
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
            for node in com: partition[node] = i

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

    # --- 2. 运行时活动热力图 ---
    obs = env.reset()
    E = torch.zeros(brain.N)
    I = torch.zeros(brain.N)
    E_history = []
    H_exc_history = []
    H_inh_history = []
    tau_history = []
    actions = []
    done = False
    steps = 0

    while not done and steps < 100:
        obs_t = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            logits, E, I, error, total_in, value, tau_eff = brain(obs_t, E, I)
            action = torch.argmax(logits).item()
        E_history.append(E.numpy().copy())
        H_exc_history.append(brain.H_exc_state.numpy().copy())
        H_inh_history.append(brain.H_inh_state.numpy().copy())
        tau_history.append(tau_eff.numpy().copy())
        actions.append(action)
        next_obs, reward, done = env.step(action)
        obs = next_obs
        steps += 1

    E_history = np.array(E_history)
    H_exc_history = np.array(H_exc_history)
    H_inh_history = np.array(H_inh_history)
    tau_history = np.array(tau_history)

    # --- 3. 综合动态图 ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 3a. 柱体兴奋活动
    ax = axes[0, 0]
    im = ax.imshow(E_history.T, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(im, ax=ax, label="E state")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Column #")
    ax.set_title("Column Excitation Dynamics")
    action_changes = [i for i in range(1, len(actions)) if actions[i] != actions[i - 1]]
    for t in action_changes:
        ax.axvline(x=t, color='r', linestyle='--', alpha=0.5)

    # 3b. 兴奋性激素动态
    ax = axes[0, 1]
    im = ax.imshow(H_exc_history.T, aspect='auto', cmap='Oranges', interpolation='nearest')
    plt.colorbar(im, ax=ax, label="H_exc")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Column #")
    ax.set_title("Excitatory Hormone (wide diffusion, mild gain)")

    # 3c. 抑制性激素动态
    ax = axes[1, 0]
    im = ax.imshow(H_inh_history.T, aspect='auto', cmap='Purples', interpolation='nearest')
    plt.colorbar(im, ax=ax, label="H_inh")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Column #")
    ax.set_title("Inhibitory Hormone (local diffusion, strong gain)")

    # 3d. 有效 tau_e 动态
    ax = axes[1, 1]
    im = ax.imshow(tau_history.T, aspect='auto', cmap='coolwarm', interpolation='nearest')
    plt.colorbar(im, ax=ax, label="tau_eff")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Column #")
    ax.set_title("Effective tau_e (base × short-term × hormone)")

    plt.tight_layout()
    plt.show()

    # --- 4. tau_e_base 分布 ---
    plt.figure(figsize=(10, 4))
    plt.bar(range(brain.N), brain.tau_e_base.data.numpy(), color='steelblue', alpha=0.7)
    plt.xlabel("Column #")
    plt.ylabel("tau_e_base (genetic)")
    plt.title("Per-Column tau_e_base Distribution (Random Init + Hormone-Adapted)")
    plt.axhline(y=brain.tau_e_base.data.mean().item(), color='r', linestyle='--', label=f"Mean={brain.tau_e_base.data.mean().item():.3f}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==========================================
# 6. 主循环
# ==========================================
if __name__ == "__main__":
    POP_SIZE = 1024
    GENERATIONS = 20
    env = SnakeEnv(grid_size=10)

    print("Initializing Population...")
    population = [EIBrainRegion(num_columns=64, init_density=0.15) for _ in range(POP_SIZE)]
    for ind in population:
        ind.save_genetic_baseline()

    history = {'gen': [], 'best': [], 'avg': []}

    for gen in range(GENERATIONS):
        fitnesses = []
        for ind in population:
            fit = evaluate_individual(ind, env)
            fitnesses.append(fit)

        fitnesses = np.array(fitnesses)
        best_fit = np.max(fitnesses)
        avg_fit = np.mean(fitnesses)
        history['gen'].append(gen)
        history['best'].append(best_fit)
        history['avg'].append(avg_fit)

        best_idx = np.argmax(fitnesses)
        best_brain = population[best_idx]
        print(f"Gen {gen + 1}/{GENERATIONS} | Best: {best_fit:.1f} | Avg: {avg_fit:.1f} | "
              f"Edges: {best_brain.M_in.sum().item():.0f}-in / {best_brain.M_rec.sum().item():.0f}-rec / "
              f"{best_brain.M_out.sum().item():.0f}-out | "
              f"tau_e: {best_brain.tau_e_base.data.mean().item():.3f}±{best_brain.tau_e_base.data.std().item():.3f}")

        population = evolve_topology(population, fitnesses, elite_size=64, mut_rate=0.05)

    plot_history(history)

    best_brain = population[np.argmax(fitnesses)]
    print("\nLaunching visualization for the best individual...")
    visualize_best_brain_play(best_brain, grid_size=10, max_steps=300)

    print("\n--- Best Brain Structure ---")
    print(f"Input edges:  {best_brain.M_in.sum().item():.0f}")
    print(f"Recurrent edges: {best_brain.M_rec.sum().item():.0f}")
    print(f"Output edges: {best_brain.M_out.sum().item():.0f}")
    print(f"tau_e_base: mean={best_brain.tau_e_base.data.mean().item():.3f}, "
          f"std={best_brain.tau_e_base.data.std().item():.3f}, "
          f"min={best_brain.tau_e_base.data.min().item():.3f}, "
          f"max={best_brain.tau_e_base.data.max().item():.3f}")

    visualize_brain_ecosystem(best_brain, env)