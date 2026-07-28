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
    def __init__(self, num_columns=32, obs_dim=4, action_dim=2, hidden_size=32, connectivity=0.1):
        super().__init__()
        self.num_columns = num_columns
        self.hidden_size = hidden_size
        
        self.column = CorticalColumn(hidden_size)
        
        # 外部连接
        self.W_in = nn.Parameter(torch.randn(num_columns, obs_dim) * 0.1)
        
        # 内部连接 M (稀疏化: 只保留 connectivity 比例的连接)
        self.M = nn.Parameter(torch.randn(num_columns, num_columns * hidden_size) * 0.05)
        # 生成稀疏掩码，并在内存中固定下来
        m_mask = (torch.rand(num_columns, num_columns * hidden_size) < connectivity).float()
        self.register_buffer('M_mask', m_mask)
        
        self.W_pred = nn.Parameter(torch.randn(num_columns, hidden_size) * 0.1)
        self.W_out_local = nn.Parameter(torch.randn(num_columns, hidden_size) * 0.1)
        self.W_out = nn.Parameter(torch.randn(action_dim, num_columns) * 0.1)

        self.baseline = None

    def forward(self, obs, h_prev, c_prev):
        column_inputs = torch.matmul(self.W_in, obs) 
        
        h_flat = h_prev.view(-1) 
        # 应用稀疏掩码，切断不需要的连接
        internal_inputs = torch.matmul(self.M * self.M_mask, h_flat)
        
        x = (column_inputs + internal_inputs).unsqueeze(1).unsqueeze(2)
        out, (h_new, c_new) = self.column.lstm(x, (h_prev.unsqueeze(0), c_prev.unsqueeze(0)))
        
        out_squeezed = out.squeeze(1)
        column_outputs = torch.sum(out_squeezed * self.W_out_local, dim=1) 
        action_logits = torch.matmul(self.W_out, column_outputs)
        
        return action_logits, h_new.squeeze(0), c_new.squeeze(0), out_squeezed, column_inputs

    # 局部预测编码微调 (加入权值衰减防爆炸)
    def apply_pc_update(self, prev_out, current_input, obs, lr=0.001, decay=0.0001):
        with torch.no_grad():
            pred_input = torch.sum(prev_out * self.W_pred, dim=1)
            error = current_input - pred_input 
            
            # 更新权重并减去衰减项 (突触稳态)
            self.W_pred += lr * (prev_out * error.unsqueeze(1)) - decay * self.W_pred
            self.W_in += lr * torch.outer(error, obs) - decay * self.W_in
            # 注意：M矩阵在生命周期内不进行PC微调，保持进化算法的纯净性

    def save_genetic_baseline(self):
        self.baseline = {name: param.data.clone() for name, param in self.named_parameters()}

    def restore_genetic_baseline(self):
        if self.baseline is not None:
            for name, param in self.named_parameters():
                if name in self.baseline:
                    param.data = self.baseline[name].clone()

# ==========================================
# 2. 进化算法与评估逻辑 (核心改进)
# ==========================================
def evaluate_individual(brain, env, render=False):
    # 评估时，记录原始基线
    original_baseline = {k: v.clone() for k, v in brain.baseline.items()} if brain.baseline else None
    
    rewards = []
    # 玩3局游戏
    for i in range(3):
        # 每局开始前，恢复到出生基线，保证3局测试的公平性
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
            
            # 局内学习
            brain.apply_pc_update(prev_out, col_in.detach(), obs_tensor, lr=0.001)
            prev_out = out.detach()
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            obs = next_obs
        rewards.append(total_reward)

    avg_reward = np.mean(rewards)

    # ---- 软化拉马克遗传 ----
    # 不直接使用最后一局的权重，而是将“原始基线”与“最后一局微调后的权重”做滑动平均
    if not render:
        with torch.no_grad():
            lamarckian_factor = 0.2 # 保留80%旧基线，融入20%新学习
            for name, param in brain.named_parameters():
                if name in original_baseline:
                    # 确保稀疏掩码不参与平均
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
    # 如果提供了结构掩码(如M_mask)，确保被剪枝的连接不会因为变异而复活
    if mask is not None:
        mutated = mutated * mask
    return mutated

def crossover(parent1, parent2):
    child = copy.deepcopy(parent1)
    with torch.no_grad():
        for param_c, param_p1, param_p2 in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
            mask = torch.rand_like(param_c) > 0.5
            param_c.data = torch.where(mask, param_p1.data, param_p2.data)
    return child

def run_evolution(generations=50, pop_size=64, elite_size=4):
    env = gym.make("CartPole-v1")
    
    # 初始化种群，降低内部连接率到 10% (小世界先验)
    population = [BrainRegion(connectivity=0.1) for _ in range(pop_size)]
    
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
            # 锦标赛选择：随机抽3个精英，选最好的两个作为父母
            tournament1 = np.random.choice(elites, 3, replace=False)
            p1 = max(tournament1, key=lambda x: x.fitness if hasattr(x, 'fitness') else 0) # 需要在评估时记录fitness
            
            # 为了简单起见，这里直接用之前的随机选择也是可以的，只要精英池大了就不会有大问题
            p1, p2 = np.random.choice(elites, 2, replace=False)
            
            child = crossover(p1, p2)
            with torch.no_grad():
                child.W_in.data = mutate_weights(child.W_in.data, mutation_rate=0.1, mutation_scale=0.1)
                # 对 M 矩阵变异时，传入其掩码，维持稀疏性
                child.M.data = mutate_weights(child.M.data, mask=child.M_mask, mutation_rate=0.1, mutation_scale=0.1)
                child.W_pred.data = mutate_weights(child.W_pred.data, mutation_rate=0.1, mutation_scale=0.1)
                child.W_out_local.data = mutate_weights(child.W_out_local.data, mutation_rate=0.1, mutation_scale=0.1)
                child.W_out.data = mutate_weights(child.W_out.data, mutation_rate=0.1, mutation_scale=0.1)
            child.save_genetic_baseline()
            new_population.append(child)
            
        population = new_population
        
    env.close()
    return population, history

# ==========================================
# 3. 运行与可视化展示
# ==========================================
def visualize_best(best_brain):
    env = gym.make("CartPole-v1", render_mode="human")
    # 在可视化时，render=True 会跳过拉马克权重覆盖，保持其最佳状态
    reward = evaluate_individual(best_brain, env, render=True)
    print(f"Visualized Best Brain Reward: {reward}")
    env.close()

def plot_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history['gen'], history['best'], label='Best Fitness', color='red', marker='o')
    plt.plot(history['gen'], history['avg'], label='Avg Fitness', color='blue', alpha=0.6)
    plt.title("Evolution Progress (Sparse + Soft Lamarckian + Homeostasis)")
    plt.xlabel("Generation")
    plt.ylabel("CartPole Reward (Max 500)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # 适当增加代数以适应新的平滑机制
    final_population, history = run_evolution(generations=500, pop_size=64, elite_size=32)
    plot_history(history)
    
    # 找出历史最佳的个体 (不仅仅是最后一代的最后一个)
    # 因为我们在评估时改变了 baseline，为了保证展示效果，直接取最后一代适应度最高的
    # 重新评估一次最后一代的种群以获取确切的最佳个体
    print("Re-evaluating final population to find the absolute best...")
    env = gym.make("CartPole-v1")
    best_reward = -1
    best_brain = None
    for ind in final_population:
        r = evaluate_individual(ind, env, render=True) # render=True 防止覆盖基线
        if r > best_reward:
            best_reward = r
            best_brain = ind
    env.close()
    
    print(f"Found best with reward: {best_reward}")
    print("Launching visualization...")
    visualize_best(best_brain)