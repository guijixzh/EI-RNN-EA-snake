"""PPO 强化学习训练管线（test6 语义）。

单网络 + N 个向量化环境 + K 倍帧率思考 + 截断 BPTT。
    - rollout 用 no_grad 的 forward_ppo_k（K 次内部迭代，观测逐次衰减，logits 平均）
    - buffer 存每步『初始状态』，ppo_update 从初始状态重放 K 次思考算 new_logp
    - 每步奖励区分 seen/unseen + 空转/饥饿惩罚（见 Config）
"""
from __future__ import annotations

import collections
import os
import random
import sys
import time

import numpy as np
import torch
from torch.distributions import Categorical

from .brain import EIBrainRegion
from .env import SnakeEnv, _obs_sees_food
from . import io

# ==================== 可微前向（K 帧思考）====================


def forward_ppo_k(brain, obs_t, E, I, short_term, horm_e, horm_i, counts,
                  K=None, decay=None):
    """K 倍帧率思考的可微前向（与 deliberate_action 语义一致）。

    - 观测逐次衰减 obs_t * (decay^k)，E/I/short/hormone 跨 K 次迭代连续传递
    - logits 取 K 次平均；value 取最后一次
    - 返回 (avg_logits[B,A], value[B], E_f, I_f, short_f, he_f, hi_f)

    训练用前向时梯度会流经整个 K 步思考链（内存 O(K·B)）。
    """
    cfg = brain.cfg
    if K is None:
        K = max(1, int(getattr(cfg, 'FRAME_RATE', 1)))
    if decay is None:
        decay = float(getattr(cfg, 'INPUT_DECAY', 0.9))

    logits_sum = None
    E_f, I_f = E, I
    short_f = short_term
    he_f, hi_f = horm_e, horm_i
    last_value = None

    for k in range(K):
        scaled_obs = obs_t * (decay ** k)
        logits, last_value, E_f, I_f, short_f, he_f, hi_f = brain.forward_ppo(
            scaled_obs, E_f, I_f, short_f, he_f, hi_f, counts)
        if logits_sum is None:
            logits_sum = logits
        else:
            logits_sum = logits_sum + logits

    avg_logits = logits_sum / K
    return avg_logits, last_value, E_f, I_f, short_f, he_f, hi_f


def update_counts(counts, action):
    """更新连续动作计数（detach 纯 buffer 操作，不影响梯度）。"""
    counts = counts.detach()
    if action.dim() == 2 and action.shape[1] == 1:
        act_idx = action.squeeze(1)
    else:
        act_idx = action
    cur = counts.gather(1, act_idx.unsqueeze(1)).squeeze(1) + 1.0
    new = torch.zeros_like(counts)
    new.scatter_(1, act_idx.unsqueeze(1), cur.unsqueeze(1))
    return new


def make_zero_states(brain, batch=1):
    """创建初始零状态（每局/每 env 起点）。"""
    N = brain.N
    E = torch.zeros(batch, N)
    I = torch.zeros(batch, N)
    short = torch.zeros(batch, N)
    if brain.train_hormone:
        he = torch.zeros(batch, N)
        hi = torch.zeros(batch, N)
    else:
        he = None
        hi = None
    counts = torch.zeros(batch, 3)
    return E, I, short, he, hi, counts


def trainable_parameters(brain, cfg):
    """返回参与 PPO 梯度更新的参数列表（掩码冻结，激素按开关）。"""
    params = [brain.W_in, brain.W_rec, brain.W_out, brain.b_out,
              brain.tau_e_init, brain.w_ei, brain.w_ie,
              brain.V, brain.b_v]
    if bool(getattr(cfg, 'TRAIN_HORMONE_NET', False)):
        params += [brain.W_hormone1, brain.b_hormone1,
                   brain.W_excit, brain.b_excit,
                   brain.W_inhib, brain.b_inhib]
    return params


# ==================== 评估 ====================


def evaluate(brain, env, cfg, episodes=5):
    """argmax 贪心评估（K=FRAME_RATE 思考）。返回 (avg_food, avg_steps)。"""
    brain.eval()
    total_foods = []
    total_steps = []
    with torch.no_grad():
        for _ in range(episodes):
            E, I, short, he, hi, counts = make_zero_states(brain, 1)
            obs = env.reset()
            ep_food = 0
            steps = 0
            done = truncated = False
            while not done and not truncated and steps < cfg.MAX_STEPS:
                obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                logits, _, E, I, short, he, hi = forward_ppo_k(
                    brain, obs_t, E, I, short, he, hi, counts)
                action = int(logits.squeeze(0).argmax().item())
                counts = update_counts(counts, torch.tensor([[action]]))
                obs, ate, done, truncated = env.step(action)
                if ate:
                    ep_food += 1
                steps += 1
            total_foods.append(ep_food)
            total_steps.append(steps)
    brain.train()
    return float(np.mean(total_foods)), float(np.mean(total_steps))


def evaluate_k1_probe(brain, env, cfg, episodes=5):
    """K=1 单次思考评估（诊断用）。"""
    brain.eval()
    total_foods = []
    with torch.no_grad():
        for _ in range(episodes):
            E, I, short, he, hi, counts = make_zero_states(brain, 1)
            obs = env.reset()
            ep_food = 0
            steps = 0
            done = truncated = False
            while not done and not truncated and steps < cfg.MAX_STEPS:
                obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                logits, _, E, I, short, he, hi = forward_ppo_k(
                    brain, obs_t, E, I, short, he, hi, counts, K=1, decay=1.0)
                action = int(logits.squeeze(0).argmax().item())
                counts = update_counts(counts, torch.tensor([[action]]))
                obs, ate, done, truncated = env.step(action)
                if ate:
                    ep_food += 1
                steps += 1
            total_foods.append(ep_food)
    brain.train()
    return float(np.mean(total_foods))


# ==================== GAE / PPO 更新 ====================


def compute_gae(rew_buf, val_buf, mask_buf, last_val, cfg):
    """GAE 优势估计。

    rew_buf/val_buf/mask_buf: [T, B]；last_val: [B]
    mask=0 表示该步真死亡（boot 0）；mask=1 表示可继续（含 truncated，boot value）。
    """
    T, B = rew_buf.shape
    adv = torch.zeros_like(rew_buf)
    gae = 0.0
    next_val = last_val * mask_buf[T - 1]
    for t in reversed(range(T)):
        if t == T - 1:
            nv = next_val
        else:
            nv = val_buf[t + 1]
        delta = rew_buf[t] + cfg.GAMMA * nv * mask_buf[t] - val_buf[t]
        gae = delta + cfg.GAMMA * cfg.GAE_LAMBDA * mask_buf[t] * gae
        adv[t] = gae
    ret = adv + val_buf
    return adv, ret


def ppo_update(brain, optimizer, cfg,
               obs_buf, act_buf, logp_buf, adv_buf, ret_buf,
               E0_buf, I0_buf, short0_buf, he0_buf, hi0_buf, counts0_buf):
    """PPO clip 更新（K=FRAME_RATE 思考重放）。返回 (policy, value, entropy) 均值。"""
    T, B, obs_dim = obs_buf.shape
    N = brain.N
    n = T * B
    K = max(1, int(getattr(cfg, 'FRAME_RATE', 1)))
    decay = float(getattr(cfg, 'INPUT_DECAY', 0.9))

    obs_flat = obs_buf.reshape(n, obs_dim)
    act_flat = act_buf.reshape(n)
    logp_flat = logp_buf.reshape(n)
    adv_flat = adv_buf.reshape(n)
    ret_flat = ret_buf.reshape(n)

    E0_flat = E0_buf.reshape(n, N)
    I0_flat = I0_buf.reshape(n, N)
    short0_flat = short0_buf.reshape(n, N)
    counts0_flat = counts0_buf.reshape(n, 3)
    he0_flat = he0_buf.reshape(n, N) if he0_buf is not None else None
    hi0_flat = hi0_buf.reshape(n, N) if hi0_buf is not None else None

    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    idx = np.arange(n)
    mb_size = int(cfg.MINIBATCH_SIZE)
    losses = {'policy': [], 'value': [], 'entropy': []}

    for _ in range(cfg.PPO_EPOCHS):
        random.shuffle(idx)
        for s in range(0, n, mb_size):
            mb = idx[s:s + mb_size]
            mb_t = torch.from_numpy(mb)

            logits, val, *_ = forward_ppo_k(
                brain, obs_flat[mb_t],
                E0_flat[mb_t], I0_flat[mb_t], short0_flat[mb_t],
                he0_flat[mb_t] if he0_flat is not None else None,
                hi0_flat[mb_t] if hi0_flat is not None else None,
                counts0_flat[mb_t], K=K, decay=decay)

            dist = Categorical(logits=logits)
            new_logp = dist.log_prob(act_flat[mb_t])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logp - logp_flat[mb_t])
            adv_mb = adv_flat[mb_t]
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1.0 - cfg.CLIP_EPS, 1.0 + cfg.CLIP_EPS) * adv_mb
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (val - ret_flat[mb_t]).pow(2).mean()

            loss = policy_loss + cfg.VALUE_COEF * value_loss - cfg.ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in trainable_parameters(brain, cfg) if p.grad is not None],
                cfg.MAX_GRAD_NORM)
            optimizer.step()

            with torch.no_grad():
                brain.tau_e_init.data.clamp_(cfg.TAU_E_MIN, cfg.TAU_E_MAX)
                brain.w_ei.data.clamp_(cfg.W_EI_MIN, cfg.W_EI_MAX)
                brain.w_ie.data.clamp_(cfg.W_IE_MIN, cfg.W_IE_MAX)

            losses['policy'].append(float(policy_loss.item()))
            losses['value'].append(float(value_loss.item()))
            losses['entropy'].append(float(entropy.item()))

    return (float(np.mean(losses['policy'])), float(np.mean(losses['value'])),
            float(np.mean(losses['entropy'])))


# ==================== 主循环 ====================


def _make_buffers(brain, cfg, T, B):
    obs_buf = torch.zeros(T, B, cfg.OBS_DIM, dtype=torch.float32)
    act_buf = torch.zeros(T, B, dtype=torch.long)
    logp_buf = torch.zeros(T, B, dtype=torch.float32)
    val_buf = torch.zeros(T, B, dtype=torch.float32)
    rew_buf = torch.zeros(T, B, dtype=torch.float32)
    mask_buf = torch.zeros(T, B, dtype=torch.float32)
    N = brain.N
    E0_buf = torch.zeros(T, B, N, dtype=torch.float32)
    I0_buf = torch.zeros(T, B, N, dtype=torch.float32)
    short0_buf = torch.zeros(T, B, N, dtype=torch.float32)
    counts0_buf = torch.zeros(T, B, 3, dtype=torch.float32)
    if brain.train_hormone:
        he0_buf = torch.zeros(T, B, N, dtype=torch.float32)
        hi0_buf = torch.zeros(T, B, N, dtype=torch.float32)
    else:
        he0_buf = None
        hi0_buf = None
    return (obs_buf, act_buf, logp_buf, val_buf, rew_buf, mask_buf,
            E0_buf, I0_buf, short0_buf, he0_buf, hi0_buf, counts0_buf)


def run_training(cfg, visualize=False):
    """PPO 主训练循环（K 倍帧率思考，全程统一口径）。返回 (best_brain, history)。"""
    from .vis import plot_history, visualize_best_brain_play

    t_program = time.perf_counter()

    start_iter = 0
    brain = None
    optimizer = None
    ckpt = None
    history = {'iter': [], 'mean_rew': [], 'mean_len': [],
               'eval_food': [], 'eval_iter': [],
               'policy_loss': [], 'value_loss': [], 'entropy': []}
    best_brain = None
    best_food = -1.0
    best_steps = 0.0

    if cfg.AUTO_RESUME and os.path.exists(cfg.CHECKPOINT_PATH):
        ckpt = _load_ppo_checkpoint(cfg)
        if ckpt is not None:
            start_iter = ckpt['next_iter']
            brain = ckpt['brain']
            history = ckpt['history']
            best_brain = ckpt['best_brain']
            best_food = ckpt['best_food']
            best_steps = ckpt['best_steps']
            print(f"\n=== 检测到断点 [{cfg.CHECKPOINT_PATH}] ===")
            print(f"  {cfg.TOTAL_ITERATIONS} 轮中已完成 {start_iter} 轮 -> 从第 {start_iter} 轮接续 | "
                  f"历史最优: Food={best_food:.1f}")

    if brain is None:
        print("=== einbrain PPO 强化学习（K 倍帧率思考）===")
        print(f"  N_ENVS={cfg.N_ENVS}, ROLLOUT_LEN={cfg.ROLLOUT_LEN}, "
              f"TOTAL_ITERATIONS={cfg.TOTAL_ITERATIONS}")
        print(f"  K=FRAME_RATE={cfg.FRAME_RATE}, INPUT_DECAY={cfg.INPUT_DECAY}")

        if cfg.SEED_FROM_TEST5D:
            seed_result = io.load_best_model_brain(cfg.TEST5D_MODEL_PATH, cfg)
            if seed_result is not None:
                seed, seed_food, seed_steps = seed_result
                brain = seed
                eval_env_seed = SnakeEnv(grid_size=cfg.GRID_SIZE, max_steps=cfg.MAX_STEPS)
                base_food, base_steps = evaluate(brain, eval_env_seed, cfg,
                                                 episodes=max(cfg.EVAL_EPISODES, 5))
                best_food = base_food
                best_steps = base_steps
                best_brain = io.load_brain_state(
                    io.save_brain_state(brain, use_half=False), cfg)
                print(f"  [Seed] 已加载 {cfg.TEST5D_MODEL_PATH} 并用本口径重评基线: "
                      f"Food={base_food:.1f}, Steps={base_steps:.1f}")
            else:
                print(f"  [Seed] 未找到可用种子模型，随机初始化")
                brain = EIBrainRegion(cfg)
        else:
            brain = EIBrainRegion(cfg)

    assert brain is not None, "brain 初始化失败"

    if optimizer is None:
        optimizer = torch.optim.Adam(trainable_parameters(brain, cfg), lr=cfg.LR)
        if ckpt is not None and ckpt.get('optimizer') is not None:
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
            except Exception as e:
                print(f"警告: 优化器状态加载失败 ({e})，已重新初始化")

    B = cfg.N_ENVS
    N = brain.N
    T = cfg.ROLLOUT_LEN
    envs = [SnakeEnv(grid_size=cfg.GRID_SIZE, max_steps=cfg.MAX_STEPS, cfg=cfg) for _ in range(B)]
    obs_stack = np.stack([e.reset() for e in envs], axis=0).astype(np.float32)
    E, I, short, he, hi, counts = make_zero_states(brain, B)

    (obs_buf, act_buf, logp_buf, val_buf, rew_buf, mask_buf,
     E0_buf, I0_buf, short0_buf, he0_buf, hi0_buf, counts0_buf) = _make_buffers(
        brain, cfg, T, B)

    cur_iter = None
    try:
        for it in range(start_iter, cfg.TOTAL_ITERATIONS):
            cur_iter = it
            t_iter = time.perf_counter()

            ep_rew = np.zeros(B, dtype=np.float64)
            ep_len = np.zeros(B, dtype=np.int64)
            ep_infos = []

            for t in range(T):
                obs_t = torch.from_numpy(obs_stack)
                E0_buf[t] = E
                I0_buf[t] = I
                short0_buf[t] = short
                if he0_buf is not None:
                    he0_buf[t] = he
                    hi0_buf[t] = hi
                counts0_buf[t] = counts

                with torch.no_grad():
                    logits, val, E, I, short, he, hi = forward_ppo_k(
                        brain, obs_t, E, I, short, he, hi, counts)

                dist = Categorical(logits=logits)
                act = dist.sample()
                logp = dist.log_prob(act)

                obs_buf[t] = obs_t
                act_buf[t] = act
                logp_buf[t] = logp.detach()
                val_buf[t] = val.detach()

                seen_mask = _obs_sees_food(obs_stack)
                step_rew = np.where(seen_mask, cfg.SEEN_STEP_REWARD, cfg.UNSEEN_STEP_REWARD)

                counts = update_counts(counts, act)

                act_np = act.numpy()
                next_obs = np.zeros_like(obs_stack)
                for i in range(B):
                    a = int(act_np[i])
                    o2, ate, done, truncated = envs[i].step(a)
                    r = float(step_rew[i])
                    if envs[i].loiter_now:
                        r += cfg.LOITER_PENALTY
                    hunger = envs[i].steps_without_food - cfg.HUNGER_WINDOW
                    if hunger > 0:
                        r -= cfg.HUNGER_STEP_PENALTY * float(hunger)
                    if ate:
                        r += cfg.EAT_REWARD
                    if done or truncated:
                        r += cfg.DEATH_REWARD
                    next_obs[i] = o2
                    rew_buf[t, i] = r
                    mask_buf[t, i] = 0.0 if done else 1.0

                    ep_rew[i] += r
                    ep_len[i] += 1

                    if done or truncated:
                        ep_infos.append((float(ep_rew[i]), int(ep_len[i])))
                        next_obs[i] = envs[i].reset()
                        E[i].zero_()
                        I[i].zero_()
                        short[i].zero_()
                        if he0_buf is not None:
                            he[i].zero_()
                            hi[i].zero_()
                        counts[i].zero_()
                        ep_rew[i] = 0.0
                        ep_len[i] = 0

                obs_stack = next_obs

            with torch.no_grad():
                obs_t = torch.from_numpy(obs_stack)
                _, last_val, _, _, _, _, _ = forward_ppo_k(
                    brain, obs_t, E, I, short, he, hi, counts)

            adv_buf, ret_buf = compute_gae(rew_buf, val_buf, mask_buf, last_val, cfg)

            if cfg.LR_DECAY:
                progress = (it + 1) / cfg.TOTAL_ITERATIONS
                for g in optimizer.param_groups:
                    g['lr'] = cfg.LR * (1.0 - 0.9 * progress)

            p_loss, v_loss, ent = ppo_update(
                brain, optimizer, cfg,
                obs_buf, act_buf, logp_buf, adv_buf, ret_buf,
                E0_buf, I0_buf, short0_buf, he0_buf, hi0_buf, counts0_buf)

            mean_rew = float(np.mean([e[0] for e in ep_infos])) if ep_infos else 0.0
            mean_len = float(np.mean([e[1] for e in ep_infos])) if ep_infos else 0.0
            history['iter'].append(it)
            history['mean_rew'].append(mean_rew)
            history['mean_len'].append(mean_len)
            history['policy_loss'].append(p_loss)
            history['value_loss'].append(v_loss)
            history['entropy'].append(ent)

            iter_time = time.perf_counter() - t_iter
            print(f"Iter {it + 1}/{cfg.TOTAL_ITERATIONS} | "
                  f"MeanRew: {mean_rew:6.2f} | MeanLen: {mean_len:5.1f} | "
                  f"PolicyL: {p_loss:.4f} | ValL: {v_loss:.4f} | Ent: {ent:.4f} | "
                  f"{iter_time:.1f}s", end="")

            if (it + 1) % cfg.EVAL_INTERVAL == 0:
                eval_env = SnakeEnv(grid_size=cfg.GRID_SIZE, max_steps=cfg.MAX_STEPS)
                eval_food, eval_steps = evaluate(brain, eval_env, cfg, episodes=cfg.EVAL_EPISODES)
                history['eval_food'].append(eval_food)
                history['eval_iter'].append(it)
                if eval_food > best_food:
                    best_food = eval_food
                    best_steps = eval_steps
                    best_brain = io.load_brain_state(io.save_brain_state(brain, use_half=False), cfg)
                    io.save_best_model(cfg.BEST_MODEL_PATH, brain, cfg, best_food, best_steps)
                print(f" | EvalFood: {eval_food:.1f} (Best: {best_food:.1f})", end="")

            print()

            if (it + 1) % cfg.CHECKPOINT_INTERVAL == 0:
                _save_ppo_checkpoint(cfg, it + 1, brain, optimizer,
                                     history, best_brain, best_food, best_steps)

    except KeyboardInterrupt:
        print("\n训练被中断 (Ctrl+C)，正在保存断点以供下次自动接续...")
        _save_ppo_checkpoint(cfg, cur_iter, brain, optimizer,
                             history, best_brain, best_food, best_steps)
        sys.exit(0)

    print(f"\nTotal runtime: {time.perf_counter() - t_program:.1f}s")

    if best_brain is None:
        best_brain = brain
    if best_food < 0:
        eval_env = SnakeEnv(grid_size=cfg.GRID_SIZE, max_steps=cfg.MAX_STEPS)
        best_food, best_steps = evaluate(best_brain, eval_env, cfg,
                                         episodes=max(cfg.EVAL_EPISODES, 5))
    io.save_best_model(cfg.BEST_MODEL_PATH, best_brain, cfg, best_food, best_steps)
    print(f"最优模型已保存: {cfg.BEST_MODEL_PATH} (Food={best_food:.1f}, Steps={best_steps:.1f})")

    if os.path.exists(cfg.CHECKPOINT_PATH):
        os.remove(cfg.CHECKPOINT_PATH)
        print(f"训练已完成，已删除临时断点: {cfg.CHECKPOINT_PATH}")

    if visualize:
        plot_history(history)
        visualize_best_brain_play(best_brain, cfg, max_steps=300)

    return best_brain, history


# ==================== PPO 断点（含 optimizer 状态）====================


def _save_ppo_checkpoint(cfg, next_iter, brain, optimizer, history,
                         best_brain, best_food, best_steps):
    parent = os.path.dirname(os.path.abspath(cfg.CHECKPOINT_PATH))
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {
        'next_iter': int(next_iter),
        'brain': io.save_brain_state(brain, use_half=False),
        'optimizer': optimizer.state_dict(),
        'history': history,
        'best_brain': io.save_brain_state(best_brain, use_half=False) if best_brain is not None else None,
        'best_food': float(best_food),
        'best_steps': float(best_steps),
        'config': io.config_dict(cfg),
        'random_state': random.getstate(),
        'torch_rng_state': torch.get_rng_state(),
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    tmp_path = cfg.CHECKPOINT_PATH + '.tmp'
    torch.save(payload, tmp_path)
    os.replace(tmp_path, cfg.CHECKPOINT_PATH)
    print(f"  [Checkpoint] 断点已保存 -> {cfg.CHECKPOINT_PATH} (next_iter={next_iter})")


def _load_ppo_checkpoint(cfg):
    if not os.path.exists(cfg.CHECKPOINT_PATH):
        return None
    data = torch.load(cfg.CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    saved_cfg = data.get('config', {})
    if saved_cfg and (saved_cfg.get('NUM_COLUMNS') != cfg.NUM_COLUMNS or
                      saved_cfg.get('OBS_DIM') != cfg.OBS_DIM or
                      saved_cfg.get('ACTION_DIM') != cfg.ACTION_DIM):
        print(f"警告: 断点 {cfg.CHECKPOINT_PATH} 与当前配置不匹配 (N/OBS/ACTION_DIM)，已忽略")
        return None
    return {
        'next_iter': int(data['next_iter']),
        'brain': io.load_brain_state(data['brain'], cfg),
        'optimizer': data.get('optimizer'),
        'history': data.get('history', {}),
        'best_brain': (io.load_brain_state(data['best_brain'], cfg)
                       if data.get('best_brain') is not None else None),
        'best_food': float(data.get('best_food', -1.0)),
        'best_steps': float(data.get('best_steps', 0.0)),
    }
