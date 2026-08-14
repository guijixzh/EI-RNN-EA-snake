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
    original_baseline = {k: v.clone() for k, v in brain.baseline.items()} if brain.baseline else None
    
    rewards = []
    for i in range(3):#固定三局？
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
            
            # ---- 核心修改：奖励塑形 ----
            # obs[0] 是小车位置 (-2.4 到 2.4), obs[2] 是杆子角度 (-0.21 到 0.21 弧度)
            # 强烈惩罚偏离中心和角度倾斜，迫使网络学习"回中"与"保持垂直"
            pos_penalty = abs(next_obs[0]) / 2.4
            ang_penalty = abs(next_obs[2]) / 0.21
            shaped_reward = reward - (pos_penalty * 0.5) - (ang_penalty * 0.5)
            
            total_reward += shaped_reward
            done = terminated or truncated
            obs = next_obs
        rewards.append(total_reward)

    avg_reward = np.mean(rewards)

    # ---- 软化拉马克遗传 ----
    if not render:
        with torch.no_grad():
            lamarckian_factor = 0.2 
            for name, param in brain.named_parameters():
                if name in original_baseline:
                    if name == 'M_mask': continue
                    
                    # 核心修复：只对在生命周期内被 PC 机制真正修改过的参数进行拉马克平滑
                    # LSTM 等冻结参数跳过，避免浮点运算引入的精度噪音
                    if name in ['W_in', 'W_pred']:
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
            p1, p2 = np.random.choice(elites, 2, replace=False)
            child = crossover(p1, p2)
            
            # ---- 核心修改：自适应变异率 ----
            # 随着代数增加，变异率和变异尺度衰减，帮助网络在后期稳定收敛
            progress = gen / generations
            current_mut_rate = max(0.01, 0.1 * (1 - progress))  # 最低保留0.01
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

#检验部分
import torch.distributions as dist

def verify_model_architecture(best_brain):
    print("\n" + "="*50)
    print("开始模型架构与演化有效性验证...")
    print("="*50)
    
    # 1. 验证皮质柱参数是否被冻结且一致
    print("\n[检验 1] 皮质柱权重一致性验证")
    if os.path.exists("cortical_column_pretrained.pth"):
        base_state = torch.load("cortical_column_pretrained.pth")
        lstm_state_dict = {}
        for k, v in base_state.items():
            if k.startswith("lstm."):
                lstm_state_dict[k[5:]] = v
                
        current_lstm_state = best_brain.column.lstm.state_dict()
        all_match = True
        for k, v in current_lstm_state.items():
            if not torch.allclose(v, lstm_state_dict[k], atol=1e-7):
                print(f"  -> 错误! 键 {k} 的权重与预训练模型不一致！")
                all_match = False
                break
        if all_match:
            print("  -> 成功! 演化后的皮质柱权重与预训练初始权重 100% 一致。参数已被完美冻结。")
    else:
        print("  -> 跳过: 未找到预训练权重文件。")

    # 2. 验证外部模块是否真的发生了演化
    print("\n[检验 2] 外部连接模块演化验证")
    # 生成一个全新的、未训练的脑区作为基准
    baseline_brain = BrainRegion(connectivity=0.1)
    
    modules_to_check = ['W_in', 'M', 'W_pred', 'W_out_local', 'W_out']
    for name in modules_to_check:
        # 获取演化后和初始的权重
        evolved_param = getattr(best_brain, name).data
        initial_param = getattr(baseline_brain, name).data
        
        # 计算权重的变化量 (L2 范数)
        diff = torch.norm(evolved_param - initial_param).item()
        # 计算权重的当前标准差
        std = torch.std(evolved_param).item()
        
        print(f"  -> 模块 {name:<15} | 演化偏移量: {diff:.4f} | 当前权重标准差: {std:.4f}")
        
    # 3. 验证内部拓扑稀疏性
    print("\n[检验 3] 内部连接矩阵 M 的拓扑稀疏性验证")
    M_data = best_brain.M.data
    M_mask = best_brain.M_mask
    
    # 计算非零元素比例
    total_elements = M_mask.numel()
    active_elements = torch.sum(M_mask).item()
    sparsity = active_elements / total_elements
    
    # 验证被掩码切断的位置，权重是否真的为 0 (或者由于浮点误差接近0)
    # 注意：由于我们在变异时传入了mask，被屏蔽的连接应该始终保持初始值或被强制归零
    masked_weights = M_data * (1 - M_mask) # 获取被屏蔽位置的权重
    max_masked_weight = torch.max(torch.abs(masked_weights)).item()
    
    print(f"  -> 设定稀疏率: 0.10 | 实际有效连接比例: {sparsity:.4f}")
    print(f"  -> 被掩码切断连接的最大残留权重: {max_masked_weight:.8f} (应接近0)")
    
    print("\n" + "="*50)
    print("验证结束。")
    print("="*50)

if __name__ == "__main__":
    # 适当增加代数以适应新的平滑机制
    final_population, history = run_evolution(generations=80, pop_size=512, elite_size=32)#200 256 32 即可
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
    verify_model_architecture(best_brain)