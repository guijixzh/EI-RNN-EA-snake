import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 定义皮质柱 (32维 LSTM)
# ==========================================
class CorticalColumn(nn.Module):
    def __init__(self, input_size=1, hidden_size=32):
        super(CorticalColumn, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        # 解码器仅用于训练时计算Loss，训练完成后将被丢弃
        self.decoder = nn.Linear(hidden_size, 1) 

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x)
        # 使用线性层将隐状态解码为预测值，仅为了计算Loss
        pred = self.decoder(out)
        return pred, out, (h_n, c_n)

# ==========================================
# 2. 任务生成器
# ==========================================
def gen_sine_wave(batch_size, seq_len):
    t = np.linspace(0, 20, seq_len)
    freqs = np.random.uniform(0.5, 2.0, batch_size)
    phases = np.random.uniform(0, 2*np.pi, batch_size)
    x = np.sin(2 * np.pi * freqs[:, None] * t + phases[:, None])
    # 预测下一时刻
    inputs = x[:, :-1, None]
    targets = x[:, 1:, None]
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

def gen_delayed_xor(batch_size, seq_len, delay=5):
    # 1. 生成基础的 0/1 随机序列
    raw_x = np.random.randint(0, 2, (batch_size, seq_len))
    
    # 2. 连续化：使用指数平滑让阶跃变成带有过渡的脉冲
    # alpha 越小，过渡越平缓
    alpha = 0.3 
    smooth_x = np.zeros_like(raw_x, dtype=np.float32)
    smooth_x[:, 0] = raw_x[:, 0]
    for i in range(1, seq_len):
        smooth_x[:, i] = alpha * smooth_x[:, i-1] + (1 - alpha) * raw_x[:, i]
        
    # 将尺度从 [0, 1] 映射到 [-1, 1]，与其他任务的输入尺度对齐
    inputs = (smooth_x * 2 - 1)[:, :, None]
    
    # 3. 加入低频背景噪声，避免网络在静态区间“休眠”
    noise = np.random.normal(0, 0.05, (batch_size, seq_len, 1)).astype(np.float32)
    inputs = inputs + noise
    
    # 4. 计算目标：当前平滑信号与 delay 步前平滑信号的逻辑异或
    # 为了保持目标也是连续平滑的，我们对 raw_x 进行异或，然后再平滑
    raw_y = np.zeros_like(raw_x)
    for i in range(delay, seq_len):
        raw_y[:, i] = raw_x[:, i] ^ raw_x[:, i-delay]
        
    smooth_y = np.zeros_like(raw_y, dtype=np.float32)
    smooth_y[:, 0] = raw_y[:, 0]
    for i in range(1, seq_len):
        smooth_y[:, i] = alpha * smooth_y[:, i-1] + (1 - alpha) * raw_y[:, i]
        
    # 将目标尺度映射到 [-1, 1]
    targets = (smooth_y * 2 - 1)[:, :, None]
    
    return torch.tensor(inputs), torch.tensor(targets)

def gen_signal_denoising(batch_size, seq_len):
    t = np.linspace(0, 20, seq_len)
    # 干净信号：不同频率的正弦波组合
    clean = np.sin(2 * np.pi * 1.0 * t) + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    clean = np.tile(clean, (batch_size, 1))
    # 加入高斯噪声
    noise = np.random.normal(0, 0.5, (batch_size, seq_len))
    noisy = clean + noise
    inputs = noisy[:, :-1, None]
    targets = clean[:, 1:, None] # 预测下一步的干净信号
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

def gen_lorenz_system(batch_size, seq_len):
    # 简化版的Lorenz吸引子 X维度演化
    dt = 0.02
    # 确保 x, y, z 都是 (batch_size, seq_len) 的形状
    x = np.zeros((batch_size, seq_len))
    y = np.zeros((batch_size, seq_len))
    z = np.zeros((batch_size, seq_len))
    
    # 初始状态加点随机性
    x[:, 0] = np.random.uniform(-1, 1, batch_size)
    y[:, 0] = np.random.uniform(-1, 1, batch_size)
    z[:, 0] = np.random.uniform(0, 1, batch_size)
    
    # 标准的洛伦兹迭代
    sigma, rho, beta = 10.0, 28.0, 2.66667
    for i in range(1, seq_len):
        dx = sigma * (y[:, i-1] - x[:, i-1])
        dy = x[:, i-1] * (rho - z[:, i-1]) - y[:, i-1]
        dz = x[:, i-1] * y[:, i-1] - beta * z[:, i-1]
        
        x[:, i] = x[:, i-1] + dx * dt
        y[:, i] = y[:, i-1] + dy * dt
        z[:, i] = z[:, i-1] + dz * dt
        
    # 归一化 x 序列
    x = (x - x.mean()) / (x.std() + 1e-8)
    
    inputs = x[:, :-1, None].astype(np.float32)
    targets = x[:, 1:, None].astype(np.float32)
    return torch.tensor(inputs), torch.tensor(targets)

# ==========================================
# 3. 训练循环
# ==========================================
def train_column():
    column = CorticalColumn()
    optimizer = optim.Adam(column.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    tasks = {
        "Sine Wave": gen_sine_wave,
        "Delayed XOR": gen_delayed_xor,
        "Denoising": gen_signal_denoising,
        "Lorenz Dyn.": gen_lorenz_system
    }

    epochs = 2000
    batch_size = 64
    seq_len = 100
    history = {name: [] for name in tasks}

    print("Starting Cortical Column Pre-training...")
    for epoch in range(epochs):
        epoch_losses = {name: 0 for name in tasks}
        
        # 交替训练各个任务
        for name, gen_func in tasks.items():
            x, y = gen_func(batch_size, seq_len)
            
            optimizer.zero_grad()
            pred, _, _ = column(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_losses[name] = loss.item()
            history[name].append(loss.item())

        if (epoch + 1) % 20 == 0:
            avg_loss = np.mean([epoch_losses[t] for t in tasks])
            print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f} | " + 
                  " | ".join([f"{t}: {l:.4f}" for t, l in epoch_losses.items()]))

    # 保存训练好的权重（包含LSTM和Decoder，后续使用时只取LSTM）
    torch.save(column.state_dict(), "cortical_column_pretrained.pth")
    print("Training complete. Weights saved to 'cortical_column_pretrained.pth'")
    
    return column, history

# ==========================================
# 4. 评估与可视化
# ==========================================
def evaluate_and_plot(column):
    column.eval()
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Cortical Column (32-Dim LSTM) Multi-Task Performance", fontsize=16)
    
    tasks = [
        ("Sine Wave Prediction", gen_sine_wave),
        ("Delayed XOR (Delay=5)", gen_delayed_xor),
        ("Signal Denoising", gen_signal_denoising),
        ("Lorenz System Evolution", gen_lorenz_system)
    ]

    with torch.no_grad():
        for ax, (name, gen_func) in zip(axs.flat, tasks):
            x, y = gen_func(1, 100) # 生成1个样本
            pred, hidden_states, _ = column(x)
            
            # 将数据转换为numpy用于绘图
            x_np = x[0, :, 0].numpy()
            y_np = y[0, :, 0].numpy()
            pred_np = pred[0, :, 0].numpy()
            
            ax.plot(y_np, label='Target', color='blue', alpha=0.7)
            ax.plot(pred_np, label='Prediction', color='red', linestyle='--')
            
            # 对于去噪任务，额外画出输入噪声
            if "Denoising" in name:
                ax.plot(x_np, label='Noisy Input', color='gray', alpha=0.3)
                
            ax.set_title(name)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# 运行训练和展示
if __name__ == "__main__":
    model, history = train_column()
    evaluate_and_plot(model)