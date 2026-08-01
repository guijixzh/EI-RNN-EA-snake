# -*- coding: utf-8 -*-
"""
test4c.py — 纯预测编码链接权重训练
=====================================================
读取 test4b 训练得到的最优模型 (test4b_best_model.pth)，
在 **保持拓扑关系与 EI 皮质柱自身参数不变** 的前提下，
仅用 **预测编码误差** 在线训练优化内部链接权重：

        pred_in  = W_pred @ E_prev           (预测上一时刻柱活动)
        error    = total_in - pred_in        (预测误差)
        L        = mean(error^2)             (预测编码目标)

可训练链接权重（预测回路）：
    W_in   (输入→柱链接, lr = PC_LR × W_IN_LR_SCALE 慢速微调,
             梯度受 M_in 掩码约束, 拓扑/行为影响最小)
    W_pred (预测链接,    全连接无拓扑掩码, 以 PC_LR 充分训练;
             仅参与预测误差、不影响行为输出)

冻结（EI 皮质柱自身参数 / 拓扑 / 柱间循环 / 输出层，训练前后逐元素校验）：
    M_in / M_rec / M_out  —— 拓扑掩码，永不变
    W_rec  —— 柱间循环链接权重（EI 皮质柱自身动力学核心，预测编码不改写）
    tau_e_init / 激素网络 / w_ei / w_ie —— EI 柱自身参数
    W_out / b_out         —— 输出层不参与预测误差，保持最优模型原值

用法：
    python test4c.py                          # 默认训练（W_pred + W_in 慢速微调）
    python test4c.py --epochs 500             # 指定训练局数
    python test4c.py --zero-interference      # 零干扰对照：仅训练 W_pred（行为不变）

展示内容：
    1) 训练过程的指标变化：预测误差 loss 曲线 + 周期评估 Food / Steps 曲线
    2) 训练前后完整游玩对比（取消 300 步限制，玩到环境自然结束）：
        统计对比图（食物 / 存活步数 / 动作分布）+ 并排全程游玩动画

直接复用 test4b 已有模块（Config / SnakeEnv / EIBrainRegion /
load_best_model_brain / render_snake_game 等）。
"""
import argparse
import os
import random
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from test4b import (Config, SnakeEnv, EIBrainRegion,
                    load_best_model_brain, render_snake_game)

# =================================================
# 0. test4c 配置
# =================================================
BEST_MODEL_PATH = 'test4b_best_model.pth'   # test4b 保存的最优模型文件

PC_EPOCHS       = 2000      # 预测编码训练局数
PC_LR           = 0.0001    # Adam 学习率
PC_WEIGHT_DECAY = 0.00001   # 权重衰减
W_REC_FROZEN    = True      # 冻结 W_rec（柱间循环=EI 皮质动力学核心，不参与预测编码更新）
W_IN_LR_SCALE   = 0.1       # W_in（输入链接）学习率缩放：慢速微调，最大限度保护最优行为
EVAL_INTERVAL   = 20        # 每 N 局做一次纯游玩评估
EVAL_EPISODES   = 3         # 每次评估 / 前后对比的游玩局数
PLAY_MAX_STEPS  = 5000      # 取消 300 步限制：允许环境自然结束为止
TRAIN_MAX_STEPS = 5000      # 训练同样不受 300 步限制
SEED            = 114       # 固定随机种子，保证前后对比可复现

# =================================================
# 1. 训练专用前向（使 W_rec 可反传，行为与 test4b.forward 数学等价）
# =================================================
def train_forward(brain, obs_t, E_prev, I_prev):
    """训练专用前向，与 test4b.EIBrainRegion.forward 数学上完全等价。

    关键差异：
    - rec_in 不再使用缓存 W_rec_eff，而是现算 W_rec*M_rec@E_prev，
      使 W_rec 能参与反向传播（缓存只含 .data，会切断计算图）；
    - 将不参与预测误差的运行时动力学（激素 / 短期 / tau / E-I 更新 /
      动作输出）隔离在 no_grad 块内，保证只有预测回路权重收到梯度；
    - E_prev 输入为无梯度张量（上一时步 E_new 来自 no_grad），自然截断 BPTT。

    返回与 forward 相同: (action_logits, E_new, I_new, error, total_in)
    """
    cfg = brain.cfg

    # --- 1. 总输入（保留梯度） ---
    ext_in = torch.matmul(brain.W_in * brain.M_in, obs_t)
    rec_in = torch.matmul(brain.W_rec * brain.M_rec, E_prev)
    total_in = ext_in + rec_in

    # --- 2-6. 运行时动力学（无需梯度，隔离在 no_grad 块） ---
    with torch.no_grad():
        hormone_input = torch.cat([E_prev, I_prev, total_in], dim=-1)
        h_hidden = torch.relu(torch.matmul(brain.W_hormone1, hormone_input) + brain.b_hormone1)
        excit_cmd = torch.relu(torch.matmul(brain.W_excit, h_hidden) + brain.b_excit)
        inhib_cmd = torch.relu(torch.matmul(brain.W_inhib, h_hidden) + brain.b_inhib)

        brain.hormone_excit = (1 - cfg.HORMONE_DECAY) * excit_cmd + \
            cfg.HORMONE_DECAY * (
                (1 - cfg.EXCIT_DIFFUSION) * brain.hormone_excit +
                cfg.EXCIT_DIFFUSION * torch.matmul(brain.M_norm, brain.hormone_excit)
            )
        brain.hormone_inhib = (1 - cfg.HORMONE_DECAY) * inhib_cmd + \
            cfg.HORMONE_DECAY * (
                (1 - cfg.INHIB_DIFFUSION) * brain.hormone_inhib +
                cfg.INHIB_DIFFUSION * torch.matmul(brain.M_norm, brain.hormone_inhib)
            )

        brain.short_term_state = cfg.SHORT_TERM_DECAY * brain.short_term_state + \
            (1 - cfg.SHORT_TERM_DECAY) * E_prev

        effective_tau_e = brain.tau_e_init + \
            cfg.SHORT_TERM_GAIN * brain.short_term_state + \
            cfg.EXCIT_HORMONE_GAIN * brain.hormone_excit - \
            cfg.INHIB_HORMONE_GAIN * brain.hormone_inhib
        effective_tau_e = torch.clamp(effective_tau_e, cfg.TAU_E_MIN, cfg.TAU_E_MAX)

        E_new = torch.sigmoid(total_in + effective_tau_e * E_prev - brain.w_ei * I_prev)
        I_new = torch.sigmoid(brain.w_ie * E_new)

        fatigue = torch.clamp(
            torch.relu(brain.consecutive_counts - cfg.FATIGUE_THRESHOLD) * cfg.FATIGUE_GAIN,
            max=cfg.FATIGUE_MAX)
        action_logits = torch.matmul(brain.W_out_eff, E_new) + brain.b_out - fatigue

    # --- 7. 预测编码（需要梯度：唯一回传路径） ---
    pred_in = torch.matmul(brain.W_pred, E_prev)
    error = total_in - pred_in

    return action_logits, E_new, I_new, error, total_in

# =================================================
# 2. 纯游玩评估（无学习，取消 300 步限制）
# =================================================
def play_evaluate(brain, env, cfg, n_episodes=EVAL_EPISODES,
                  max_steps=PLAY_MAX_STEPS, verbose=False):
    """纯游玩评估：不学习、无 300 步限制，每局玩到环境自然结束。

    返回 (avg_food, avg_steps, action_counts)
    """
    foods, steps_list = [], []
    action_counts = [0, 0, 0]

    for ep in range(n_episodes):
        brain.reset_runtime()
        obs = env.reset()
        E = torch.zeros(brain.N)
        I = torch.zeros(brain.N)
        ep_food, steps, done = 0, 0, False

        while not done and steps < max_steps:
            obs_t = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                logits, E, I, error, total_in = brain(obs_t, E, I)
                action = int(torch.argmax(logits).item())

            brain.update_fatigue(action)
            action_counts[action] += 1

            next_obs, ate, done = env.step(action)
            if ate:
                ep_food += 1
            obs = next_obs
            steps += 1

        foods.append(ep_food)
        steps_list.append(steps)

    avg_food = float(np.mean(foods))
    avg_steps = float(np.mean(steps_list))
    if verbose:
        print(f"    Food per episode: {foods} | Steps per episode: {steps_list}")
    return avg_food, avg_steps, action_counts

# =================================================
# 3. 纯预测编码训练主循环
# =================================================
def train_predictive_coding(brain, env, cfg):
    """纯预测编码在线训练：最小化预测误差。

    可训练链接权重：
      - W_pred：预测链接，以 PC_LR 充分训练（仅参与预测误差，不影响行为输出）
      - W_in  ：输入链接，以 PC_LR × W_IN_LR_SCALE 慢速微调（最大限度保护最优行为）
      - W_rec ：柱间循环链接，默认冻结（W_REC_FROZEN=True）—— EI 皮质柱自身
                动力学不做任何预测编码改写；仅当 W_REC_FROZEN=False 时才会
                纳入训练（注意：会改变 EI 皮质柱间动力学，可能破坏最优行为）

    返回 (loss_history, eval_history)
    """
    # --- 冻结全部参数，仅放开预测回路链接权重 ---
    for p in brain.parameters():
        p.requires_grad = False

    # W_pred：预测链接，充分训练
    brain.W_pred.requires_grad = True
    # W_in：输入链接，慢速微调（scale=0 时完全冻结，即零干扰对照模式）
    brain.W_in.requires_grad = True

    param_groups = [
        {'params': [brain.W_pred], 'lr': PC_LR, 'weight_decay': PC_WEIGHT_DECAY},
        {'params': [brain.W_in], 'lr': PC_LR * W_IN_LR_SCALE,
         'weight_decay': PC_WEIGHT_DECAY},
    ]
    if not W_REC_FROZEN:
        brain.W_rec.requires_grad = True
        param_groups.append({'params': [brain.W_rec],
                             'lr': PC_LR * W_IN_LR_SCALE,
                             'weight_decay': PC_WEIGHT_DECAY})

    optimizer = torch.optim.Adam(param_groups, lr=PC_LR)
    grad_params = ([brain.W_pred, brain.W_in] +
                   ([brain.W_rec] if not W_REC_FROZEN else []))

    loss_history = []                       # 每局平均预测误差
    eval_history = {'ep': [], 'food': [], 'steps': []}

    t0 = time.perf_counter()
    for ep in range(1, PC_EPOCHS + 1):
        brain.reset_runtime()
        obs = env.reset()
        E = torch.zeros(brain.N)
        I = torch.zeros(brain.N)
        ep_losses = []
        steps, done = 0, False

        while not done and steps < TRAIN_MAX_STEPS:
            obs_t = torch.tensor(obs, dtype=torch.float32)
            logits, E, I, error, total_in = train_forward(brain, obs_t, E, I)
            loss = torch.mean(error ** 2)

            optimizer.zero_grad()
            loss.backward()
            with torch.no_grad():
                # 拓扑约束：仅掩码覆盖的连接允许更新（拓扑关系不变）
                if brain.W_in.grad is not None:
                    brain.W_in.grad.mul_(brain.M_in)
                if brain.W_rec.grad is not None:
                    brain.W_rec.grad.mul_(brain.M_rec)
                torch.nn.utils.clip_grad_norm_(grad_params, max_norm=1.0)
            optimizer.step()
            # 刷新缓存（W_rec_eff 等），使推理 forward 与训练权重同步
            brain.refresh_cached()

            ep_losses.append(float(loss.item()))

            action = int(torch.argmax(logits).item())
            brain.update_fatigue(action)

            next_obs, ate, done = env.step(action)
            obs = next_obs
            steps += 1

        loss_history.append(float(np.mean(ep_losses)))

        if ep % EVAL_INTERVAL == 0 or ep == PC_EPOCHS:
            ev_food, ev_steps, _ = play_evaluate(brain, env, cfg,
                                                 n_episodes=EVAL_EPISODES,
                                                 max_steps=PLAY_MAX_STEPS)
            eval_history['ep'].append(ep)
            eval_history['food'].append(ev_food)
            eval_history['steps'].append(ev_steps)
            print(f"  [Eval @ ep {ep}/{PC_EPOCHS}] Food={ev_food:.2f} "
                  f"Steps={ev_steps:.0f} | Loss={loss_history[-1]:.4f} | "
                  f"Elapsed={time.perf_counter() - t0:.1f}s")

    print(f"  训练完成: {PC_EPOCHS} 局, 总耗时 {time.perf_counter() - t0:.1f}s")
    return loss_history, eval_history

# =================================================
# 4. 可视化：训练指标变化
# =================================================
def plot_training_curves(loss_history, eval_history,
                         save_path='test4c_training_curves.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左：预测误差 loss
    ax1.plot(np.arange(1, len(loss_history) + 1), loss_history,
             color='royalblue', lw=1.0)
    ax1.set_title("Training Metrics — Prediction Error (MSE)")
    ax1.set_xlabel("Training Episode")
    ax1.set_ylabel("Mean Prediction Loss")
    ax1.grid(True, alpha=0.3)

    # 右：周期评估 Food / Steps（取消 300 步限制）
    eps = eval_history['ep']
    ax2.plot(eps, eval_history['food'], 'o-', color='red', label='Avg Food')
    ax2.set_xlabel("Training Episode")
    ax2.set_ylabel("Avg Food", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_title("Periodic Evaluation (no 300-step limit)")
    ax2.grid(True, alpha=0.3)

    ax3 = ax2.twinx()
    ax3.plot(eps, eval_history['steps'], 's--', color='green', label='Avg Steps')
    ax3.set_ylabel("Avg Steps", color='green')
    ax3.tick_params(axis='y', labelcolor='green')

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

# =================================================
# 5. 可视化：训练前后游玩统计对比
# =================================================
def plot_before_after_comparison(before_metrics, after_metrics,
                                 save_path='test4c_before_after.png'):
    before_food, before_steps, before_counts = before_metrics
    after_food, after_steps, after_counts = after_metrics

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. 平均食物
    axes[0].bar(['Before PC', 'After PC'], [before_food, after_food],
                color=['tab:gray', 'tab:red'])
    for i, v in enumerate([before_food, after_food]):
        axes[0].text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    axes[0].set_title("Avg Food per Game")
    axes[0].set_ylabel("Food")

    # 2. 平均存活步数
    axes[1].bar(['Before PC', 'After PC'], [before_steps, after_steps],
                color=['tab:gray', 'tab:blue'])
    for i, v in enumerate([before_steps, after_steps]):
        axes[1].text(i, v + 1, f"{v:.0f}", ha='center', fontweight='bold')
    axes[1].set_title("Avg Steps Survived")
    axes[1].set_ylabel("Steps")

    # 3. 动作分布
    labels = ['Fwd', 'Left', 'Right']
    x = np.arange(len(labels))
    w = 0.35
    axes[2].bar(x - w / 2, before_counts, w, label='Before PC', color='tab:gray')
    axes[2].bar(x + w / 2, after_counts, w, label='After PC', color='tab:orange')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_title("Action Distribution")
    axes[2].legend()

    for ax in axes:
        ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

# =================================================
# 6. 并排全程游玩动画（训练前 vs 训练后，取消 300 步限制）
# =================================================
def play_snake_animation(brain_before, brain_after, cfg,
                         max_steps=PLAY_MAX_STEPS, draw_every=5, interval=0.02):
    """并排播放训练前 / 训练后各一局的完整游玩。
    取消 300 步限制，直到双方都自然结束（撞墙/撞己/饿死）。
    两个环境使用相同随机种子初始化，保证起始局面一致、对比公平。
    """
    env_b = SnakeEnv(cfg.GRID_SIZE)
    env_a = SnakeEnv(cfg.GRID_SIZE)

    random.seed(SEED)
    obs_b = env_b.reset()
    random.seed(SEED)
    obs_a = env_a.reset()

    brain_before.reset_runtime()
    brain_after.reset_runtime()
    E_b = torch.zeros(brain_before.N)
    I_b = torch.zeros(brain_before.N)
    E_a = torch.zeros(brain_after.N)
    I_a = torch.zeros(brain_after.N)

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    img1 = ax1.imshow(render_snake_game(env_b), cmap='viridis', vmin=0, vmax=1)
    img2 = ax2.imshow(render_snake_game(env_a), cmap='viridis', vmin=0, vmax=1)
    ax1.set_title("Before PC Training")
    ax2.set_title("After PC Training")
    for ax in (ax1, ax2):
        ax.axis('off')

    done_a = done_b = False
    steps = 0
    while (not done_a or not done_b) and steps < max_steps:
        if not done_b:
            obs_b_t = torch.tensor(obs_b, dtype=torch.float32)
            with torch.no_grad():
                logits_b, E_b, I_b, _, _ = brain_before(obs_b_t, E_b, I_b)
                action_b = int(torch.argmax(logits_b).item())
            brain_before.update_fatigue(action_b)
            obs_b, _, done_b = env_b.step(action_b)

        if not done_a:
            obs_a_t = torch.tensor(obs_a, dtype=torch.float32)
            with torch.no_grad():
                logits_a, E_a, I_a, _, _ = brain_after(obs_a_t, E_a, I_a)
                action_a = int(torch.argmax(logits_a).item())
            brain_after.update_fatigue(action_a)
            obs_a, _, done_a = env_a.step(action_a)

        steps += 1
        if steps % draw_every == 0 or (done_a and done_b):
            img1.set_data(render_snake_game(env_b))
            img2.set_data(render_snake_game(env_a))
            ax1.set_title(f"Before PC | Step {steps} | Food {env_b.food_count}")
            ax2.set_title(f"After PC  | Step {steps} | Food {env_a.food_count}")
            fig.canvas.draw_idle()
            plt.pause(interval)

    print(f"\n[Playback] Before: Score={len(env_b.body) - 2}, "
          f"Food={env_b.food_count}, Steps={env_b.steps}, Done={done_b}")
    print(f"[Playback] After:  Score={len(env_a.body) - 2}, "
          f"Food={env_a.food_count}, Steps={env_a.steps}, Done={done_a}")
    plt.ioff()
    plt.show()

# =================================================
# 7. 主流程
# =================================================
def main(epochs=None, zero_interference=False):
    global PC_EPOCHS, W_IN_LR_SCALE
    if epochs is not None:
        PC_EPOCHS = epochs
    if zero_interference:
        W_IN_LR_SCALE = 0.0

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    cfg = Config()
    env = SnakeEnv(grid_size=cfg.GRID_SIZE)

    # ---------- 1. 加载最优模型 ----------
    print("=" * 60)
    print("加载 test4b 最优模型 ...")
    loaded = load_best_model_brain(BEST_MODEL_PATH, cfg)
    if loaded is None:
        print(f"错误：无法加载最优模型 {BEST_MODEL_PATH}。")
        return
    brain_before, saved_food, saved_steps = loaded
    print(f"  最优模型加载成功 (test4b 历史记录 Food={saved_food:.1f}, "
          f"Steps={saved_steps:.1f})")
    print(f"  拓扑: M_in={brain_before.M_in.sum().item():.0f} / "
          f"M_rec={brain_before.M_rec.sum().item():.0f} / "
          f"M_out={brain_before.M_out.sum().item():.0f}")

    # 独立加载第二份作为训练对象（与 before 完全独立，两侧互不影响）
    brain_train, _, _ = load_best_model_brain(BEST_MODEL_PATH, cfg)
    brain_train.baseline = None  # 纯预测编码训练不使用遗传基线恢复

    # 记录待校验的冻结参数（训练后进行逐元素一致性校验）
    frozen_check = {
        'M_in': brain_train.M_in.clone(),
        'M_rec': brain_train.M_rec.clone(),
        'M_out': brain_train.M_out.clone(),
        'tau_e_init': brain_train.tau_e_init.data.clone(),
        'W_hormone1': brain_train.W_hormone1.data.clone(),
        'b_hormone1': brain_train.b_hormone1.data.clone(),
        'W_excit': brain_train.W_excit.data.clone(),
        'b_excit': brain_train.b_excit.data.clone(),
        'W_inhib': brain_train.W_inhib.data.clone(),
        'b_inhib': brain_train.b_inhib.data.clone(),
        'W_out': brain_train.W_out.data.clone(),
        'b_out': brain_train.b_out.data.clone(),
    }
    if W_REC_FROZEN:
        frozen_check['W_rec'] = brain_train.W_rec.data.clone()
    if W_IN_LR_SCALE == 0.0:
        frozen_check['W_in'] = brain_train.W_in.data.clone()

    # ---------- 2. 训练前纯游玩评估（取消 300 步限制） ----------
    print("\n=== 训练前纯游玩评估（无 300 步限制） ===")
    before_metrics = play_evaluate(brain_before, env, cfg,
                                   EVAL_EPISODES, PLAY_MAX_STEPS, verbose=True)
    print(f"  Before: Food={before_metrics[0]:.2f}, "
          f"Steps={before_metrics[1]:.1f}, Actions={before_metrics[2]}")

    # ---------- 3. 纯预测编码训练 ----------
    mode_desc = ("零干扰对照：仅训练 W_pred（W_in 学习率=0）"
                 if W_IN_LR_SCALE == 0.0
                 else "W_pred + W_in 慢速微调")
    print(f"\n=== 纯预测编码训练（{mode_desc}） ===")
    print(f"  冻结: 拓扑掩码 M_* / W_rec(柱间循环) / tau_e_init / "
          f"激素网络 / W_out+b_out")
    print(f"  W_pred 学习率={PC_LR}, W_in 学习率={PC_LR * W_IN_LR_SCALE:.5f}"
          f" (缩放 {W_IN_LR_SCALE})")
    print(f"  (注意: 预测编码只优化内部预测回路，不直接优化行为目标，"
          f"因此 Food 曲线可能非单调)")
    loss_history, eval_history = train_predictive_coding(brain_train, env, cfg)

    # ---------- 4. 训练后纯游玩评估 ----------
    print("\n=== 训练后纯游玩评估（无 300 步限制） ===")
    after_metrics = play_evaluate(brain_train, env, cfg,
                                  EVAL_EPISODES, PLAY_MAX_STEPS, verbose=True)
    print(f"  After:  Food={after_metrics[0]:.2f}, "
          f"Steps={after_metrics[1]:.1f}, Actions={after_metrics[2]}")

    # ---------- 5. 冻结参数一致性校验 ----------
    print("\n=== 冻结参数一致性校验 ===")
    freeze_ok = True
    with torch.no_grad():
        for name, saved in frozen_check.items():
            cur = getattr(brain_train, name)
            ok = torch.equal(saved, cur)
            if not ok:
                freeze_ok = False
                print(f"  [FAIL] {name} 发生变化！")
    if freeze_ok:
        print("  全部冻结参数与训练前完全一致 OK  "
              "(拓扑 / W_rec / tau_e_init / 激素网络 / W_out / b_out 未变)")
    else:
        print("  [WARN] 检测到冻结参数变化，请检查！")

    # ---------- 6. 链接权重变化统计 ----------
    with torch.no_grad():
        d_in = (brain_train.W_in - brain_before.W_in).abs()
        d_rec = (brain_train.W_rec - brain_before.W_rec).abs()
        d_pred = (brain_train.W_pred - brain_before.W_pred).abs()
    print("\n=== 链接权重变化统计（训练前 -> 训练后） ===")
    print(f"  |ΔW_in|:   mean={d_in.mean().item():.5f}  max={d_in.max().item():.5f}")
    print(f"  |ΔW_rec|:  mean={d_rec.mean().item():.5f}  max={d_rec.max().item():.5f}")
    print(f"  |ΔW_pred|: mean={d_pred.mean().item():.5f}  max={d_pred.max().item():.5f}")

    # ---------- 7. 展示 ----------
    print("\n=== 展示训练指标变化 ===")
    plot_training_curves(loss_history, eval_history)

    print("\n=== 展示前后游玩统计对比 ===")
    plot_before_after_comparison(before_metrics, after_metrics)

    print("\n=== 并排全程游玩动画（取消 300 步限制） ===")
    try:
        play_snake_animation(brain_before, brain_train, cfg)
    except Exception as e:
        print(f"  [WARN] 动画播放失败（{e}），跳过动画。")

    print("\n完成：已展示训练指标变化与前后游玩对比。")


if __name__ == '__main__':
    _parser = argparse.ArgumentParser(description='test4c: 纯预测编码链接权重训练')
    _parser.add_argument('--epochs', type=int, default=None,
                         help='预测编码训练局数（默认取 PC_EPOCHS）')
    _parser.add_argument('--zero-interference', action='store_true',
                         help='零干扰对照模式：仅训练 W_pred（W_in 学习率=0），'
                              '行为理论上完全不变，用于验证 PC 劣化来自 W_in 扰动')
    _args = _parser.parse_args()
    main(epochs=_args.epochs, zero_interference=_args.zero_interference)