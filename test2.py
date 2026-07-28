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
        # 冻结LSTM参数，使其作为固定的通用计算单元
        for param in self.lstm.parameters():
            param.requires_grad = False

class BrainRegion(nn.Module):
    def __init__(self, num_columns=32, obs_dim=4, action_dim=2, hidden_size=32):
        super().__init__()
        self.num_columns = num_columns
        self.hidden_size = hidden_size
        
        # 1. 核心柱体
        self.column = CorticalColumn(hidden_size)
        
        # 2. 可进化与可塑的连接权重 (基因型)
        self.W_in = nn.Parameter(torch.randn(num_columns, obs_dim) * 0.1)
        self.M = nn.Parameter(torch.randn(num_columns, num_columns * hidden_size) * 0.05)
        self.W_pred = nn.Parameter(torch.randn(num_columns, hidden_size) * 0.1)
        self.W_out_local = nn.Parameter(torch.randn(num_columns, hidden_size) * 0.1)
        self.W_out = nn.Parameter(torch.randn(action_dim, num_columns) * 0.1)

    def forward(self, obs, h_prev, c_prev):
        # 1. 计算外部输入到各个柱体的信号: (32, 4) x (4,) -> (32,)
        column_inputs = torch.matmul(self.W_in, obs) 
        
        # 2. 计算柱体间内部连接信号: (32, 1024) x (1024,) -> (32,)
        h_flat = h_prev.view(-1) 
        internal_inputs = torch.matmul(self.M, h_flat)
        
        # 3. 融合信号: (32,) -> (32, 1, 1) 即 (batch_size=32, seq_len=1, input_size=1)
        x = (column_inputs + internal_inputs).unsqueeze(1).unsqueeze(2)
        
        # 4. 并行通过32个柱体 (h_prev: (32,32) -> (1,32,32) 即 (num_layers=1, batch_size=32, hidden_size=32))
        out, (h_new, c_new) = self.column.lstm(x, (h_prev.unsqueeze(0), c_prev.unsqueeze(0)))
        
        # 5. 局部解码: out shape is (32, 1, 32) -> squeeze(1) -> (32, 32)
        out_squeezed = out.squeeze(1)
        # (32, 32) * (32, 32) 按位相乘后，在维度1求和 -> (32,)
        column_outputs = torch.sum(out_squeezed * self.W_out_local, dim=1) 
        
        # 6. 全局动作输出: (2, 32) x (32,) -> (2,)
        action_logits = torch.matmul(self.W_out, column_outputs)
        
        return action_logits, h_new.squeeze(0), c_new.squeeze(0), out_squeezed, column_inputs

    # 局部预测编码微调 (修正了维度对齐，确保32个柱体独立计算误差和更新)
    def apply_pc_update(self, prev_out, current_input, obs, lr=0.002):
        with torch.no_grad():
            # prev_out: (32, 32), W_pred: (32, 32) -> 对位相乘求和 -> pred_input: (32,)
            pred_input = torch.sum(prev_out * self.W_pred, dim=1)
            error = current_input - pred_input # (32,)
            
            # 更新预测权重 (32, 32)
            self.W_pred += lr * (prev_out * error.unsqueeze(1))
            # 更新输入映射权重 (32, 4)
            self.W_in += lr * torch.outer(error, obs)

# ==========================================
# 2. 进化算法与评估逻辑 (仅修改 apply_pc_update 的调用处)
# ==========================================
def evaluate_individual(brain, env, render=False):
    obs, _ = env.reset()
    h = torch.zeros(brain.num_columns, brain.hidden_size)
    c = torch.zeros(brain.num_columns, brain.hidden_size)
    prev_out = torch.zeros(brain.num_columns, brain.hidden_size)
    
    total_reward = 0
    done = False
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        
        # 前向传播
        with torch.no_grad():
            logits, h, c, out, col_in = brain(obs_tensor, h, c)
            action = torch.argmax(logits).item()
        
        # 局部预测编码微调 (生命周期内学习)
        # 传入 obs_tensor 以便更新 W_in
        brain.apply_pc_update(prev_out, col_in.detach(), obs_tensor, lr=0.002)
        prev_out = out.detach()
        
        # 执行动作
        next_obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        obs = next_obs
        
    return total_reward

def mutate_weights(weights, mutation_rate=0.1, mutation_scale=0.1):
    """对张量进行高斯变异"""
    mutated = weights.clone()
    mask = torch.rand_like(mutated) < mutation_rate
    noise = torch.randn_like(mutated) * mutation_scale
    mutated += mask * noise
    return mutated

def crossover(parent1, parent2):
    """均匀交叉产生后代"""
    child = copy.deepcopy(parent1)
    with torch.no_grad():
        for param_c, param_p1, param_p2 in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
            mask = torch.rand_like(param_c) > 0.5
            param_c.data = torch.where(mask, param_p1.data, param_p2.data)
    return child

def run_evolution(generations=15, pop_size=64, elite_size=4):
    env = gym.make("CartPole-v1")
    
    # 初始化种群
    population = [BrainRegion() for _ in range(pop_size)]
    
    # 尝试加载第一步预训练的柱体权重
    if os.path.exists("cortical_column_pretrained.pth"):
        print("Loaded pre-trained Cortical Column weights.")
        base_state = torch.load("cortical_column_pretrained.pth")
        
        # 提取并清理 LSTM 的权重键 (去掉 "lstm." 前缀，并忽略 "decoder." 的权重)
        lstm_state_dict = {}
        for k, v in base_state.items():
            if k.startswith("lstm."):
                # 截取 "lstm." 之后的字符串作为新的键名
                new_key = k[5:]
                lstm_state_dict[new_key] = v
                
        for ind in population:
            ind.column.lstm.load_state_dict(lstm_state_dict)
    else:
        print("Warning: Pre-trained weights not found. Using random LSTM weights.")

    history = {'gen': [], 'best': [], 'avg': []}
    
    for gen in range(generations):
        # 1. 评估阶段 (含生命周期内PC微调)
        fitnesses = []
        for i, individual in enumerate(population):
            # 每个个体玩3次游戏取平均，减少随机性
            rewards = [evaluate_individual(individual, env) for _ in range(3)]
            fitnesses.append(np.mean(rewards))
        
        fitnesses = np.array(fitnesses)
        
        # 记录统计
        best_fit = np.max(fitnesses)
        avg_fit = np.mean(fitnesses)
        history['gen'].append(gen)
        history['best'].append(best_fit)
        history['avg'].append(avg_fit)
        print(f"Gen {gen+1}/{generations} | Best: {best_fit:.1f} | Avg: {avg_fit:.1f}")
        
        # 2. 选择与拉马克遗传
        elite_idx = np.argsort(fitnesses)[-elite_size:]
        elites = [copy.deepcopy(population[i]) for i in elite_idx] # 保留微调后的权重(拉马克)
        
        # 3. 产生下一代
        new_population = []
        # 精英直接保留
        new_population.extend(elites)
        
        # 生成剩余个体
        while len(new_population) < pop_size:
            p1, p2 = np.random.choice(elites, 2, replace=False)
            child = crossover(p1, p2)
            # 变异 (柱体LSTM不变异，仅变异外部连接)
            with torch.no_grad():
                child.W_in.data = mutate_weights(child.W_in.data)
                child.M.data = mutate_weights(child.M.data)
                child.W_pred.data = mutate_weights(child.W_pred.data)
                child.W_out_local.data = mutate_weights(child.W_out_local.data)
                child.W_out.data = mutate_weights(child.W_out.data)
            new_population.append(child)
            
        population = new_population
        
    env.close()
    return population, history

# ==========================================
# 3. 运行与可视化展示
# ==========================================
def visualize_best(best_brain):
    env = gym.make("CartPole-v1", render_mode="human")
    reward = evaluate_individual(best_brain, env, render=True)
    print(f"Visualized Best Brain Reward: {reward}")
    env.close()

def plot_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history['gen'], history['best'], label='Best Fitness', color='red', marker='o')
    plt.plot(history['gen'], history['avg'], label='Avg Fitness', color='blue', alpha=0.6)
    plt.title("Evolution Progress (EA + Predictive Coding Plasticity)")
    plt.xlabel("Generation")
    plt.ylabel("CartPole Reward (Max 500)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # 运行进化 (减少代数以保证代码运行时间可接受，通常15代即可看到明显趋势)
    final_population, history = run_evolution(generations=100, pop_size=64, elite_size=4)
    
    # 绘制进化曲线
    plot_history(history)
    
    # 获取最佳个体并展示其游玩效果
    best_brain = final_population[-1] # 最后一个通常是历史最优之一
    print("Launching visualization for the best individual...")
    visualize_best(best_brain)