import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import copy
import random

# ==========================================
# 1. 轻量级贪吃蛇环境 (纯Python实现)
# ==========================================
class SnakeEnv:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()
        
    def reset(self):
        self.head = (self.grid_size//2, self.grid_size//2)
        self.dir = random.choice(((0,1),(1,0),(0,-1),(-1,0)))
        self.body = [self.head, (self.head[0] - self.dir[0], self.head[1] - self.dir[1])]
        self._place_food()
        self.food_count = 0
        self.steps = 0
        self.steps_without_food = 0
        return self._get_obs()
        
    def _place_food(self):
        while True:
            self.food = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            if self.food not in self.body:
                break
                
    def _get_obs(self):
        # --- 1. 原始的 6 维基础观测 ---
        obs = np.zeros(6, dtype=np.float32)
        
        left_dir = (-self.dir[1], self.dir[0])
        right_dir = (self.dir[1], -self.dir[0])
        
        for i, d in enumerate([self.dir, left_dir, right_dir]):
            next_pos = (self.head[0]+d[0], self.head[1]+d[1])
            if next_pos[0]<0 or next_pos[0]>=self.grid_size or next_pos[1]<0 or next_pos[1]>=self.grid_size or next_pos in self.body:
                obs[i] = 1.0
                
        dx = self.food[0] - self.head[0]
        dy = self.food[1] - self.head[1]
        if (dx*self.dir[0] + dy*self.dir[1]) > 0: obs[3] = 1.0
        if (dx*left_dir[0] + dy*left_dir[1]) > 0: obs[4] = 1.0
        if (dx*right_dir[0] + dy*right_dir[1]) > 0: obs[5] = 1.0
        
        # --- 2. 新增：跟随头部的 3x3 局部视野 ---
        # 计算相对于当前朝向的四个方向向量
        forward = self.dir
        back = (-self.dir[0], -self.dir[1])
        left = (-self.dir[1], self.dir[0])
        right = (self.dir[1], -self.dir[0])
        
        # 3x3 视野的 9 个格子相对偏移量 (按行排列)
        # [左前, 直前, 右前, 左, 中(头), 右, 左后, 后, 右后]
        offsets = [
            (forward[0]+left[0], forward[1]+left[1]), forward, (forward[0]+right[0], forward[1]+right[1]),
            left, (0, 0), right,
            (back[0]+left[0], back[1]+left[1]), back, (back[0]+right[0], back[1]+right[1])
        ]
        
        local_vision = []
        for r_off, c_off in offsets:
            r, c = self.head[0] + r_off, self.head[1] + c_off
            
            is_obstacle = 0.0
            is_food = 0.0
            
            # 判断是否出界或撞到自己
            if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
                is_obstacle = 1.0
            elif (r, c) in self.body:
                is_obstacle = 1.0
            elif (r, c) == self.food:
                is_food = 1.0
                
            # 每个格子输出 2 个数值
            local_vision.extend([is_obstacle, is_food])
            
        # 拼接原始观测与局部视野 (6 + 18 = 24 维)
        obs = np.concatenate([obs, np.array(local_vision, dtype=np.float32)])
        return obs
        
    def step(self, action):
        # action: 0=直走, 1=左转, 2=右转
        if action == 1: self.dir = (-self.dir[1], self.dir[0])
        elif action == 2: self.dir = (self.dir[1], -self.dir[0])

        prev_head = self.head
        next_head = (self.head[0]+self.dir[0], self.head[1]+self.dir[1])
        self.steps += 1
        self.steps_without_food += 1
        
        # 判断死亡
        will_eat = (next_head == self.food)
        body_to_check = self.body if will_eat else self.body[:-1]
        if next_head[0]<0 or next_head[0]>=self.grid_size or next_head[1]<0 or next_head[1]>=self.grid_size or next_head in body_to_check:
            return self._get_obs(), -1000, True
            
        self.body.insert(0, next_head)
        self.head = next_head
        
        # 判断吃食物
        reward = -0.1 # 鼓励快速找食物

        # 计算移动前后与食物的曼哈顿距离
        prev_dist = abs(prev_head[0] - self.food[0]) + abs(prev_head[1] - self.food[1])
        curr_dist = abs(next_head[0] - self.food[0]) + abs(next_head[1] - self.food[1])

        if curr_dist < prev_dist:
            reward += 0.5

        if self.head == self.food:
            self.food_count += 1
            reward = 20
            self.steps_without_food = 0
            self._place_food()
        else:
            self.body.pop() # 没吃到，尾巴跟着动
            
        # 防止无限循环
        if self.steps_without_food > len(self.body)+20:
            reward = -10
            return self._get_obs(), reward, True
            
        return self._get_obs(), reward, False

# ==========================================
# 2. E-I 皮质柱与脑区模型
# ==========================================
class EIBrainRegion(nn.Module):
    def __init__(self, num_columns=20, obs_dim=24, action_dim=3, init_density=0.3):
        super().__init__()
        self.N = num_columns
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # --- 基因型：拓扑掩码 (决定结构) ---
        self.M_in = (torch.rand(num_columns, obs_dim) < init_density).float()
        self.M_rec = (torch.rand(num_columns, num_columns) < init_density).float()
        torch.diagonal(self.M_rec).zero_() # 不允许自环
        self.M_out = (torch.rand(action_dim, num_columns) < init_density).float()
        
        # --- 表现型：突触权重 (可塑) ---
        self.W_in = nn.Parameter(torch.randn(num_columns, obs_dim) * 0.1)
        self.W_rec = nn.Parameter(torch.randn(num_columns, num_columns) * 0.05)
        self.W_out = nn.Parameter(torch.randn(action_dim, num_columns) * 0.1)
        self.W_pred = nn.Parameter(torch.randn(num_columns, num_columns) * 0.1) # 预测其他柱体的状态
        
        # 内部 E-I 动力学参数 (固定基因，模拟保守的生物柱体结构)
        self.tau_e = 0.7 # 兴奋性记忆保持
        self.w_ei = 2.0  # 抑制强度
        self.w_ie = 2.0  # 兴奋触发抑制的强度
        
        self.baseline = None

    def forward(self, obs_t, E_prev, I_prev):
        # 1. 外部输入 (应用拓扑掩码)
        ext_in = torch.matmul(self.W_in * self.M_in, obs_t) # (N,)
        # 2. 内部循环输入 (应用拓扑掩码)
        rec_in = torch.matmul(self.W_rec * self.M_rec, E_prev) # (N,)
        
        total_in = ext_in + rec_in
        
        # 3. E-I 离散代数更新
        E_new = torch.sigmoid(total_in + self.tau_e * E_prev - self.w_ei * I_prev)
        I_new = torch.sigmoid(self.w_ie * E_new)
        
        # 4. 预测编码：预测其他柱体应当的输入
        pred_in = torch.matmul(self.W_pred, E_prev)
        error = total_in - pred_in
        
        # 5. 动作输出
        action_logits = torch.matmul(self.W_out * self.M_out, E_new)
        
        return action_logits, E_new, I_new, error, total_in

    # 局部预测编码微调 (生命周期内学习)
    def apply_pc_update(self, E_prev, error, total_in, obs_t, lr=0.001, decay=0.0001):
        with torch.no_grad():
            # 更新预测权重和输入权重，带突触稳态衰减
            self.W_pred += lr * torch.outer(error, E_prev) - decay * self.W_pred
            self.W_in += lr * torch.outer(error, obs_t) - decay * self.W_in
            # 注意：W_rec 不进行局部学习，保持进化算法的纯粹拓扑控制

    def save_genetic_baseline(self):
        self.baseline = {
            'W_in': self.W_in.data.clone(), 'W_rec': self.W_rec.data.clone(),
            'W_out': self.W_out.data.clone(), 'W_pred': self.W_pred.data.clone(),
            'M_in': self.M_in.clone(), 'M_rec': self.M_rec.clone(), 'M_out': self.M_out.clone()
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

# ==========================================
# 3. 进化算法与评估逻辑
# ==========================================
def evaluate_individual(brain, env, render=False, max_steps=500):
    original_baseline = {k: v.clone() for k, v in brain.baseline.items()} if brain.baseline else None
    
    total_rewards = []
    total_foods = []
    
    for _ in range(5): # 玩5局取平均，平衡评估速度与准确性
        brain.restore_genetic_baseline()
        obs = env.reset()
        E = torch.zeros(brain.N)
        I = torch.zeros(brain.N)
        
        ep_reward = 0
        ep_food = 0
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            obs_t = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                # 1. 先保存旧状态 (E_prev)
                E_old = E.clone() 
                
                # 2. 前向传播，此时 E 被更新为 E_new
                logits, E, I, error, total_in = brain(obs_t, E, I)
                action = torch.argmax(logits).item()
            
            # 3. 生命周期内学习：传入正确的 E_old 作为 E_prev
            brain.apply_pc_update(E_old, error, total_in, obs_t)
            
            next_obs, reward, done = env.step(action)
            
            # 统计吃子数 (假设吃到食物的 reward 为 10)
            if reward >= 10:
                ep_food += 1
                
            ep_reward += reward
            obs = next_obs
            steps += 1
            
        total_rewards.append(ep_reward)
        total_foods.append(ep_food)

    avg_reward = np.mean(total_rewards)
    avg_food = np.mean(total_foods)

    # 软拉马克遗传：融入20%的生命周期学习成果
    if not render and original_baseline:
        with torch.no_grad():
            lamarckian_factor = 0.2
            brain.W_in.data = (1-lamarckian_factor)*original_baseline['W_in'] + lamarckian_factor*brain.W_in.data
            brain.W_pred.data = (1-lamarckian_factor)*original_baseline['W_pred'] + lamarckian_factor*brain.W_pred.data
            brain.save_genetic_baseline()
            
    # 构建复合适应度：优先吃子数，其次看分数
    fitness = avg_food + avg_reward/100
    
    # 如果是渲染模式，打印单局信息以便观察
    if render:
        print(f"  [Render] Food Eaten: {avg_food:.1f}, Reward: {avg_reward:.1f}, Fitness: {fitness:.1f}")
            
    return fitness

def evolve_topology(population, fitnesses, elite_size=64, mut_rate=0.05):
    elite_idx = np.argsort(fitnesses)[-elite_size:]
    elites = [copy.deepcopy(population[i]) for i in elite_idx]
    
    new_pop = [copy.deepcopy(e) for e in elites] # 精英直接保留
    
    while len(new_pop) < len(population):
        p1, p2 = random.sample(elites, 2)
        child = copy.deepcopy(p1)
        N = child.N
        
        # --- 核心改进：柱体级（模块级）交叉 ---
        # 生成一个布尔掩码，决定每个柱体来自父本1还是父本2
        col_mask = torch.rand(N) > 0.5 
        
        row_mask = col_mask.unsqueeze(1) # (N, 1)
        col_mask_2d = col_mask.unsqueeze(0) # (1, N)
        
        # 两人来自同一父本的连接，保留；来自不同父本的连接，各取一半概率
        same_p1 = row_mask & col_mask_2d
        same_p2 = (~row_mask) & (~col_mask_2d)
        
        child_W_rec = torch.where(same_p1, p1.W_rec.data, 
                        torch.where(same_p2, p2.W_rec.data, 
                            torch.where(torch.rand_like(p1.W_rec.data) > 0.5, p1.W_rec.data, p2.W_rec.data)))
        child.W_rec.data = child_W_rec
        with torch.no_grad():
            # 1. W_in, M_in: (N, obs_dim) -> 按行(柱体)交叉
            child.W_in.data = torch.where(col_mask.unsqueeze(1), p1.W_in.data, p2.W_in.data)
            child.M_in = torch.where(col_mask.unsqueeze(1), p1.M_in, p2.M_in)
            
            # 2. W_rec, M_rec: (N, N) -> 使用前面算好的掩码交叉
            child.W_rec.data = child_W_rec
            child.M_rec = torch.where(same_p1, p1.M_rec, 
                             torch.where(same_p2, p2.M_rec, 
                                 torch.where(torch.rand_like(p1.M_rec) > 0.5, p1.M_rec, p2.M_rec)))
            
            # 3. W_out, M_out: (action_dim, N) -> 按列(柱体)交叉
            child.W_out.data = torch.where(col_mask.unsqueeze(0), p1.W_out.data, p2.W_out.data)
            child.M_out = torch.where(col_mask.unsqueeze(0), p1.M_out, p2.M_out)
            
            # 4. W_pred: (N, N) -> 按行(柱体)交叉
            child.W_pred.data = torch.where(col_mask.unsqueeze(1), p1.W_pred.data, p2.W_pred.data)
                
        # --- 变异：主要翻转拓扑掩码位 (生长/剪切连接)，同时轻微扰动权重 ---
        with torch.no_grad():
            if random.random() < 0.05: # 5%概率发生拓扑变异
                m_attr = random.choice(['M_in', 'M_rec', 'M_out'])
                m_tensor = getattr(child, m_attr)
                mut_mask = torch.rand_like(m_tensor) < mut_rate
                m_tensor[mut_mask] = 1.0 - m_tensor[mut_mask] # 0变1，1变0
                
            # 权重高斯变异 (只扰动被激活的连接)
            for attr in ['W_in', 'W_rec', 'W_out', 'W_pred']:
                w_tensor = getattr(child, attr)
                noise = torch.randn_like(w_tensor) * 0.1
                # 只在 20% 的连接上施加噪声
                noise_mask = torch.rand_like(w_tensor) < 0.2
                setattr(child, attr, nn.Parameter(w_tensor.data + noise * noise_mask))
                
        child.save_genetic_baseline()
        new_pop.append(child)
        
    return new_pop

# ==========================================
# 4. 训练主循环与可视化
# ==========================================
def plot_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history['gen'], history['best'], label='Best Fitness', color='red', marker='o')
    plt.plot(history['gen'], history['avg'], label='Avg Fitness', color='blue', alpha=0.6)
    plt.title("Evolution Progress (E-I Columns + Topology NEAT + Soft Lamarckian)")
    plt.xlabel("Generation")
    plt.ylabel("Snake Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

import time
import matplotlib.pyplot as plt

def render_snake_game(env):
    """将当前环境状态转换为 matplotlib 可渲染的图像矩阵"""
    grid = np.zeros((env.grid_size, env.grid_size, 3)) # RGB图像
    # 食物渲染为黄色
    grid[env.food[0], env.food[1]] = [1, 1, 0]
    # 蛇身渲染为蓝色
    for seg in env.body:
        if 0 <= seg[0] < env.grid_size and 0 <= seg[1] < env.grid_size:
            grid[seg[0], seg[1]] = [0, 0, 1]
    # 蛇头渲染为红色
    if 0 <= env.head[0] < env.grid_size and 0 <= env.head[1] < env.grid_size:
        grid[env.head[0], env.head[1]] = [1, 0, 0]
    return grid

def visualize_best_brain_play(brain, grid_size=10, max_steps=300):
    """可视化最优大脑玩贪吃蛇的过程"""
    env = SnakeEnv(grid_size=grid_size)
    
    # 恢复到最优的遗传基线
    brain.restore_genetic_baseline()
    
    obs = env.reset()
    E = torch.zeros(brain.N)
    I = torch.zeros(brain.N)
    
    # 设置 matplotlib 画布
    plt.ion() # 开启交互模式
    fig, ax = plt.subplots(figsize=(6, 6))
    img = ax.imshow(render_snake_game(env), cmap='viridis', vmin=0, vmax=1)
    ax.set_title("Best Brain Playing Snake")
    ax.axis('off')
    
    total_reward = 0
    steps = 0
    done = False
    
    while not done and steps < max_steps:
        obs_t = torch.tensor(obs, dtype=torch.float32)
        
        # 前向传播，选择动作
        with torch.no_grad():
            logits, E, I, error, total_in = brain(obs_t, E, I)
            action = torch.argmax(logits).item()
        
        # 执行动作 (这里不再进行生命周期内学习，纯粹展示其学到的基线能力)
        next_obs, reward, done = env.step(action)
        total_reward += reward
        obs = next_obs
        steps += 1
        
        # 更新画面
        img.set_data(render_snake_game(env))
        ax.set_title(f"Step: {steps} | Score: {len(env.body)-2} | Total Reward: {total_reward:.1f}")
        fig.canvas.draw_idle()
        plt.pause(0.1) # 控制帧率 (0.1秒一帧)
        
    print(f"\nGame Over! Final Score: {len(env.body)-2} | Total Reward: {total_reward:.1f} | Survived Steps: {steps}")
    plt.ioff() # 关闭交互模式
    plt.show()

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter

# 尝试导入最佳社区发现算法，若无则使用内置算法
try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False
    print("Warning: python-louvain not installed. Using basic community detection.")

def visualize_brain_ecosystem(brain, env):
    print("\n=== Generating Brain Ecosystem Visualization ===")
    
    # 1. 构建网络图
    G = nx.DiGraph()
    
    # 添加节点 (分层：输入层、柱体层、输出层)
    for i in range(brain.obs_dim): G.add_node(f"In_{i}", layer='input')
    for i in range(brain.N): G.add_node(f"Col_{i}", layer='column')
    for i in range(brain.action_dim): G.add_node(f"Out_{i}", layer='output')
        
    # 添加边 (仅添加掩码为1的活跃边)
    edges = []
    # Input -> Column
    for i in range(brain.N):
        for j in range(brain.obs_dim):
            if brain.M_in[i, j] > 0:
                G.add_edge(f"In_{j}", f"Col_{i}", weight=abs(brain.W_in[i,j].item()))
                edges.append((f"In_{j}", f"Col_{i}"))
    # Column -> Column (Recurrent)
    for i in range(brain.N):
        for j in range(brain.N):
            if brain.M_rec[i, j] > 0:
                G.add_edge(f"Col_{j}", f"Col_{i}", weight=abs(brain.W_rec[i,j].item()))
                edges.append((f"Col_{j}", f"Col_{i}"))
    # Column -> Output
    for i in range(brain.action_dim):
        for j in range(brain.N):
            if brain.M_out[i, j] > 0:
                G.add_edge(f"Col_{j}", f"Out_{i}", weight=abs(brain.W_out[i,j].item()))
                edges.append((f"Col_{j}", f"Out_{i}"))

    # 2. 社区发现 (寻找自发形成的功能分区)
    if HAS_LOUVAIN:
        # 使用 Louvain 算法寻找密集连接的子图 (模块)
        partition = community_louvain.best_partition(G.to_undirected())
    else:
        # Fallback: 使用 networkx 自带的贪婪模块度算法
        communities = nx.algorithms.community.greedy_modularity_communities(G.to_undirected())
        partition = {}
        for i, com in enumerate(communities):
            for node in com: partition[node] = i

    # 3. 绘图布局 (使用弹簧布局，让连接密集的节点靠在一起)
    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
    
    # 颜色映射
    community_colors = list(mcolors.TABLEAU_COLORS.values())
    node_colors = [community_colors[partition.get(node, 0) % 10] for node in G.nodes()]
    
    # 节点大小映射 (出入度越大，节点越大，模拟重要枢纽)
    degrees = dict(G.degree())
    node_sizes = [degrees[node] * 80 + 100 for node in G.nodes()]
    
    # 边的透明度映射 (权重越大越深)
    edge_weights = nx.get_edge_attributes(G, 'weight')
    edge_alphas = [min(0.8, w * 2) for w in edge_weights.values()]

    plt.figure(figsize=(14, 10))
    
    # 绘制边
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray', width=0.5)
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors='black')
    
    # 绘制标签
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")
    
    plt.title("Evolved Brain Topology & Functional Modules (Communities)", fontsize=16)
    plt.axis("off")
    
    # ==========================================
    # 4. 绘制运行时的内部状态热力图 (有序性展示)
    # ==========================================
    plt.figure(figsize=(12, 4))
    
    # 跑一局游戏，记录 E 状态
    obs = env.reset()
    E = torch.zeros(brain.N)
    I = torch.zeros(brain.N)
    E_history = []
    actions = []
    done = False
    steps = 0
    
    while not done and steps < 100:
        obs_t = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            logits, E, I, error, total_in = brain(obs_t, E, I)
            action = torch.argmax(logits).item()
        
        E_history.append(E.numpy().copy())
        actions.append(action)
        next_obs, reward, done = env.step(action)
        obs = next_obs
        steps += 1

    E_history = np.array(E_history) # shape: (timesteps, num_columns)
    
    # 绘制热力图
    plt.imshow(E_history.T, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label="Excitation Level (E state)")
    plt.xlabel("Time Steps (in one game)", fontsize=12)
    plt.ylabel("Cortical Columns (Neurons)", fontsize=12)
    plt.title("Dynamic Activity of Columns Over Time (Ordered Patterns)", fontsize=14)
    
    # 标记动作切换点
    action_changes = [i for i in range(1, len(actions)) if actions[i] != actions[i-1]]
    for t in action_changes:
        plt.axvline(x=t, color='r', linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    POP_SIZE = 1024 #
    GENERATIONS = 60 #
    env = SnakeEnv(grid_size=10)
    
    print("Initializing Population...")
    population = [EIBrainRegion(num_columns=64, init_density=0.15) for _ in range(POP_SIZE)]
    for ind in population: ind.save_genetic_baseline()
    
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
        print(f"Gen {gen+1}/{GENERATIONS} | Best: {best_fit:.1f} | Avg: {avg_fit:.1f} | Active Edges: {population[np.argmax(fitnesses)].M_in.sum().item():.0f}-in / {population[np.argmax(fitnesses)].M_rec.sum().item():.0f}-rec / {population[np.argmax(fitnesses)].M_out.sum().item():.0f}-out")
        
        population = evolve_topology(population, fitnesses, elite_size=64, mut_rate=0.05)

    plot_history(history)

    # 打印最佳个体的结构演化结果
    best_brain = population[np.argmax(fitnesses)]

    # 启动游戏可视化
    print("\nLaunching visualization for the best individual...")
    visualize_best_brain_play(best_brain, grid_size=10, max_steps=300)
    
    print("\n--- Best Brain Topology ---")
    print(f"Input connections active: {best_brain.M_in.sum().item()}/120")
    print(f"Internal connections active: {best_brain.M_rec.sum().item()}/380")
    print(f"Output connections active: {best_brain.M_out.sum().item()}/60")

    visualize_brain_ecosystem(best_brain, env)