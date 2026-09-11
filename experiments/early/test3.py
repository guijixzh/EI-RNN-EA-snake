import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import copy
import os

# ==========================================
# 1. 定义皮质柱与脑区结构
# ==========================================
class CorticalColumn(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        for param in self.lstm.parameters():
            param.requires_grad = False

class BrainRegion(nn.Module):
    def __init__(self, num_columns=64, obs_dim=8, action_dim=4, hidden_size=32, connectivity=0.1):
        super().__init__()
        self.num_columns = num_columns
        self.hidden_size = hidden_size
        
        self.column = CorticalColumn(hidden_size)
        
        self.W_in = nn.Parameter(torch.randn(num_columns, obs_dim) * 0.1)
        self.M = nn.Parameter(torch.randn(num_columns, num_columns * hidden_size) * 0.05)
        m_mask = (torch.rand(num_columns, num_columns * hidden_size) < connectivity).float()
        self.register_buffer('M_mask', m_mask)
        
        self.W_pred = nn.Parameter(torch.randn(num_columns, hidden_size) * 0.1)
        self.W_out_local = nn.Parameter(torch.randn(num_columns, hidden_size) * 0.1)
        self.W_out = nn.Parameter(torch.randn(action_dim, num_columns) * 0.1)

        self.baseline = None

    def forward(self, obs, h_prev, c_prev):
        column_inputs = torch.matmul(self.W_in, obs) 
        h_flat = h_prev.view(-1) 
        internal_inputs = torch.matmul(self.M * self.M_mask, h_flat)
        
        x = (column_inputs + internal_inputs).unsqueeze(1).unsqueeze(2)
        out, (h_new, c_new) = self.column.lstm(x, (h_prev.unsqueeze(0), c_prev.unsqueeze(0)))
        
        out_squeezed = out.squeeze(1)
        column_outputs = torch.sum(out_squeezed * self.W_out_local, dim=1) 
        action_logits = torch.matmul(self.W_out, column_outputs)
        
        return action_logits, h_new.squeeze(0), c_new.squeeze(0), out_squeezed, column_inputs

    def apply_pc_update(self, prev_out, current_input, obs, lr=0.001, decay=0.0001):
        with torch.no_grad():
            pred_input = torch.sum(prev_out * self.W_pred, dim=1)
            error = current_input - pred_input 
            self.W_pred += lr * (prev_out * error.unsqueeze(1)) - decay * self.W_pred
            self.W_in += lr * torch.outer(error, obs) - decay * self.W_in

    def save_genetic_baseline(self):
        self.baseline = {name: param.data.clone() for name, param in self.named_parameters()}

    def restore_genetic_baseline(self):
        if self.baseline is not None:
            for name, param in self.named_parameters():
                if name in self.baseline:
                    param.data = self.baseline[name].clone()

# ==========================================
# 2. 进化算法与评估逻辑
# ==========================================
def evaluate_individual(brain, env, render=False):
    original_baseline = {k: v.clone() for k, v in brain.baseline.items()} if brain.baseline else None
    
    rewards = []
    # LunarLander 初始随机性大，评估5局取平均
    for i in range(5):
        brain.restore_genetic_baseline()
        obs, _ = env.reset()
        h = torch.zeros(brain.num_columns, brain.hidden_size)
        c = torch.zeros(brain.num_columns, brain.hidden_size)
        prev_out = torch.zeros(brain.num_columns, brain.hidden_size)
        
        total_reward = 0
        done = False
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                logits, h, c, out, col_in = brain(obs_tensor, h, c)
                action = torch.argmax(logits).item()
            
            brain.apply_pc_update(prev_out, col_in.detach(), obs_tensor, lr=0.001)
            prev_out = out.detach()
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            obs = next_obs
        rewards.append(total_reward)

    avg_reward = np.mean(rewards)

    if not render:
        with torch.no_grad():
            lamarckian_factor = 0.3 # 任务变难，适当提高拉马克遗传比例
            for name, param in brain.named_parameters():
                if name in original_baseline:
                    if name == 'M_mask': continue
                    new_weight = (1 - lamarckian_factor) * original_baseline[name] + lamarckian_factor * param.data
                    param.data = new_weight
            brain.save_genetic_baseline()
            
    return avg_reward

def mutate_weights(weights, mask=None, mutation_rate=0.1, mutation_scale=0.1):
    mutated = weights.clone()
    mask_tensor = (torch.rand_like(mutated) < mutation_rate)
    noise = torch.randn_like(mutated) * mutation_scale
    mutated += mask_tensor * noise
    if mask is not None:
        mutated = mutated * mask
    return mutated

def crossover(parent1, parent2):
    """模块级交叉：保证语义对齐，以皮质柱为单位进行重组"""
    child = copy.deepcopy(parent1)
    num_cols = child.num_columns
    
    # 随机生成一个布尔掩码，决定每个柱体来自父本1还是父本2
    # 形状: (32,)
    col_mask = torch.rand(num_cols) > 0.5
    
    with torch.no_grad():
        # 1. W_in: (32, 4) -> 按行(柱体)交叉
        child.W_in.data = torch.where(col_mask.unsqueeze(1), parent1.W_in.data, parent2.W_in.data)
        
        # 2. W_out_local: (32, 32) -> 按行(柱体)交叉
        # 这代表该柱体如何从LSTM隐状态中提取信息
        child.W_out_local.data = torch.where(col_mask.unsqueeze(1), parent1.W_out_local.data, parent2.W_out_local.data)
        
        # 3. M: (32, 1024) -> 按行(柱体)交叉
        # 这代表该柱体接收来自其他所有柱体的内部投射
        child.M.data = torch.where(col_mask.unsqueeze(1), parent1.M.data, parent2.M.data)
        # 注意：M矩阵的列代表了上游柱体，严格来说上下游应该对应，但在进化中，
        # 允许下游混合而上游不混合，可以产生新的信息处理回路，且不会导致单个柱体内部计算崩溃。
        
        # 4. W_out: (2, 32) -> 按列(柱体)交叉
        # 这代表全局动作如何读取该柱体的输出
        child.W_out.data = torch.where(col_mask.unsqueeze(0), parent1.W_out.data, parent2.W_out.data)
        
        # 5. W_pred: (32, 32) -> 按行(柱体)交叉
        child.W_pred.data = torch.where(col_mask.unsqueeze(1), parent1.W_pred.data, parent2.W_pred.data)

    return child

def run_evolution(generations=80, pop_size=64, elite_size=32):
    # 升级环境为 LunarLander
    env = gym.make("LunarLander-v3")
    
    # 扩展脑区：64个柱体，8维输入，4维输出
    population = [BrainRegion(num_columns=64, obs_dim=8, action_dim=4, connectivity=0.1) for _ in range(pop_size)]
    
    if os.path.exists("cortical_column_pretrained.pth"):
        print("Loaded pre-trained Cortical Column weights.")
        base_state = torch.load("cortical_column_pretrained.pth")
        lstm_state_dict = {}
        for k, v in base_state.items():
            if k.startswith("lstm."):
                new_key = k[5:]
                lstm_state_dict[new_key] = v
                
        for ind in population:
            ind.column.lstm.load_state_dict(lstm_state_dict)
            ind.save_genetic_baseline()
    else:
        print("Warning: Pre-trained weights not found.")
        for ind in population:
            ind.save_genetic_baseline()

    history = {'gen': [], 'best': [], 'avg': []}
    
    for gen in range(generations):
        fitnesses = []
        for individual in population:
            fitnesses.append(evaluate_individual(individual, env))
        
        fitnesses = np.array(fitnesses)
        best_fit = np.max(fitnesses)
        avg_fit = np.mean(fitnesses)
        history['gen'].append(gen)
        history['best'].append(best_fit)
        history['avg'].append(avg_fit)
        print(f"Gen {gen+1}/{generations} | Best: {best_fit:.1f} | Avg: {avg_fit:.1f}")
        
        elite_idx = np.argsort(fitnesses)[-elite_size:]
        elites = [copy.deepcopy(population[i]) for i in elite_idx]
        
        new_population = []
        new_population.extend(elites)
        
        while len(new_population) < pop_size:
            p1, p2 = np.random.choice(elites, 2, replace=False)
            
            # 80% 概率进行模块交叉，20% 概率直接克隆父本（仅靠变异探索）
            if np.random.rand() < 0.8:
                child = crossover(p1, p2)
            else:
                child = copy.deepcopy(p1) # 或 p2
            
            # 自适应变异
            progress = gen / generations
            current_mut_rate = max(0.01, 0.1 * (1 - progress))
            current_mut_scale = max(0.01, 0.1 * (1 - progress))
            
            with torch.no_grad():
                child.W_in.data = mutate_weights(child.W_in.data, mutation_rate=current_mut_rate, mutation_scale=current_mut_scale)
                child.M.data = mutate_weights(child.M.data, mask=child.M_mask, mutation_rate=current_mut_rate, mutation_scale=current_mut_scale)
                child.W_pred.data = mutate_weights(child.W_pred.data, mutation_rate=current_mut_rate, mutation_scale=current_mut_scale)
                child.W_out_local.data = mutate_weights(child.W_out_local.data, mutation_rate=current_mut_rate, mutation_scale=current_mut_scale)
                child.W_out.data = mutate_weights(child.W_out.data, mutation_rate=current_mut_rate, mutation_scale=current_mut_scale)
            child.save_genetic_baseline()
            new_population.append(child)
            
        population = new_population
        
    env.close()
    return population, history

# ==========================================
# 3. 运行与可视化展示
# ==========================================
def visualize_best(best_brain):
    env = gym.make("LunarLander-v3", render_mode="human")
    reward = evaluate_individual(best_brain, env, render=True)
    print(f"Visualized Best Brain Reward: {reward}")
    env.close()

def plot_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history['gen'], history['best'], label='Best Fitness', color='red', marker='o')
    plt.plot(history['gen'], history['avg'], label='Avg Fitness', color='blue', alpha=0.6)
    # 添加及格线 (LunarLander 大于 200 算解决)
    plt.axhline(y=200, color='green', linestyle='--', label='Solved Threshold (200)')
    plt.title("Evolution Progress on LunarLander-v3")
    plt.xlabel("Generation")
    plt.ylabel("Average Reward (5 episodes)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # 增加代数到80代，LunarLander需要更多时间演化
    final_population, history = run_evolution(generations=30, pop_size=256, elite_size=32)
    plot_history(history)
    
    print("Re-evaluating final population to find the absolute best...")
    env = gym.make("LunarLander-v3")
    best_reward = -1000
    best_brain = None
    for ind in final_population:
        r = evaluate_individual(ind, env, render=True)
        if r > best_reward:
            best_reward = r
            best_brain = ind
    env.close()
    
    print(f"Found best with reward: {best_reward}")
    print("Launching visualization...")
    visualize_best(best_brain)