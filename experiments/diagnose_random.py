# -*- coding: utf-8 -*-
"""
diagnose_random.py — E4 随机脑行为学化验（不跑进化，分钟级）

在两种 OBS_MODE（24 维 test7 观测 vs 32 维投影观测）下，用同一套随机初始化种群测量：
  1. 存活步数分布（均值/中位/P90）——解释 gen1 的 34 步 vs 5.8 步差距
  2. 每局动作切换率与动作熵——随机脑是否"只会直行"
  3. 输入驱动力 |W_in_eff @ obs| 的均值/方差——观测信号强度差异
  4. 观测→动作敏感度：对 obs 加 ±10% 噪声后 argmax 翻转率
  5. gen1 风格指标：随机种群 BestFood/BestSeen/BestUnseen（两种口径对照）

用法: python experiments/diagnose_random.py [POP] [EPISODES]
"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util

_SPEC = importlib.util.spec_from_file_location('t7a', os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test7a.py'))
t7a = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(t7a)


def measure(mode, pop_size, episodes, device, food_scale=1.0):
    cfg = t7a.Config()
    cfg.DEVICE = str(device)
    cfg.OBS_MODE = mode
    cfg.OBS_DIM = 24 if mode == '24' else 32
    cfg.USE_FP16 = False
    cfg.POP_SIZE = pop_size

    torch.manual_seed(1234)
    pop = t7a.GeneStack(cfg, device=device)
    pop.random_init()
    pop.refresh_eff()

    env = t7a.BatchedSnakeEnv(cfg, pop_size, device)
    B, N, A = pop_size, cfg.NUM_COLUMNS, cfg.ACTION_DIM
    dev = device

    alive_steps = np.zeros(B)
    switch_rates = []
    act_counts = np.zeros((B, A))
    food_tot = np.zeros(B)
    seen_tot = np.zeros(B)
    unseen_tot = np.zeros(B)
    input_abs_sum = []
    input_var = []
    flip_rate_sum, flip_trials = 0.0, 0

    for _ in range(episodes):
        env.reset()
        E = torch.zeros(B, N, device=dev)
        I = torch.zeros(B, N, device=dev)
        st = torch.zeros(B, N, device=dev)
        cts = torch.zeros(B, A, device=dev)
        prev_act = None
        switches = 0
        steps_taken = 0

        for _t in range(cfg.MAX_STEPS):
            al = env.alive
            obs = env.obs()
            if food_scale != 1.0:
                obs = obs.clone()
                obs[:, 8:16] = obs[:, 8:16] * food_scale
            sees = env.sees_food(obs)
            seen_tot += (al & sees).float().cpu().numpy()
            unseen_tot += (al & (~sees)).float().cpu().numpy()

            # 输入驱动力
            with torch.no_grad():
                drive = torch.bmm(pop.W_in_eff, obs.unsqueeze(-1).float()).squeeze(-1)
            input_abs_sum.append(drive.abs().mean().item())
            input_var.append(drive.var().item())

            act, E, I, st = t7a.deliberate_batch(pop, obs, E, I, st, cts, cfg)
            cts = t7a.update_fatigue(cts, act)

            # 观测→动作敏感度：扰动后 argmax 翻转率（每 20 步采样一次）
            if _t % 20 == 0:
                with torch.no_grad():
                    noisy = obs * (1.0 + 0.2 * (torch.rand_like(obs) - 0.5))
                    logits_sum = None
                    E2, I2, st2 = E, I, st
                    for k in range(cfg.FRAME_RATE):
                        o2 = noisy * (cfg.INPUT_DECAY ** k)
                        lg, E2, I2, st2 = t7a.forward_batch(pop, o2, E2, I2, st2, cts, cfg)
                        logits_sum = lg if logits_sum is None else logits_sum + lg
                    flip_rate_sum += (torch.argmax(logits_sum, 1) != act).float().mean().item()
                    flip_trials += 1

            if prev_act is not None:
                switches += (act != prev_act)[al].float().sum().item()
            prev_act = act.clone()
            act_counts[torch.arange(B), act.cpu()] += al.cpu().numpy()
            steps_taken += 1

            env.step(act)
            food_tot += (al & env.ate).float().cpu().numpy()
            alive_steps += al.float().cpu().numpy()
            if env.all_done():
                break
        switch_rates.append(switches / max(1.0, float(alive_steps.sum())))

    # 动作熵（按个体平均计数的全局分布）
    pc = act_counts / np.clip(act_counts.sum(axis=1, keepdims=True), 1e-9, None)
    ent = -(pc * np.log(np.clip(pc, 1e-9, None))).sum(axis=1)

    key = t7a._make_key_fn(cfg)
    mn = np.stack([food_tot / episodes, seen_tot / episodes, unseen_tot / episodes,
                   np.zeros(B), np.zeros(B)], axis=1)
    bi = max(range(B), key=lambda i: key(*mn[i]))

    return {
        'mode': mode,
        'food_scale': food_scale,
        'alive_mean': float(alive_steps.mean() / episodes),
        'alive_median': float(np.median(alive_steps / episodes)),
        'alive_p90': float(np.percentile(alive_steps / episodes, 90)),
        'switch_rate': float(np.mean(switch_rates)),
        'act_entropy': float(ent.mean()),
        'drive_abs': float(np.mean(input_abs_sum)),
        'drive_var': float(np.mean(input_var)),
        'obs_flip_rate': float(flip_rate_sum / max(1, flip_trials)),
        'best_food': float(mn[bi][0]),
        'best_seen': float(mn[bi][1]),
        'best_unseen': float(mn[bi][2]),
        'food_mean': float((food_tot / episodes).mean()),
    }


def main():
    pop_size = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 可选: 只跑 32proj 并给食物扇区乘缩放系数 (验证信号幅度假设)
    scales = [float(s) for s in sys.argv[3].split(',')] if len(sys.argv) > 3 else [1.0]
    only32 = len(sys.argv) > 3
    print(f"[diag] device={device}  POP={pop_size}  EPISODES={episodes}  scales={scales}\n")
    print(f"{'指标':<22}{'24维(test7)':>14}{'32维(投影)':>14}")
    print('-' * 52)
    rows = []
    for mode in ('24', '32proj') if not only32 else ('32proj',):
        for sc in scales:
            rows.append(measure(mode, pop_size, episodes, device, food_scale=sc))
    keys = ['alive_mean', 'alive_median', 'alive_p90', 'switch_rate', 'act_entropy',
            'drive_abs', 'drive_var', 'obs_flip_rate', 'best_food', 'best_seen',
            'best_unseen', 'food_mean']
    names = {'alive_mean': '平均存活步数', 'alive_median': '存活中位数', 'alive_p90': '存活P90',
             'switch_rate': '动作切换率', 'act_entropy': '动作熵', 'drive_abs': '|输入驱动|均值',
             'drive_var': '输入驱动方差', 'obs_flip_rate': '扰动翻转率', 'best_food': '随机种群BestFood',
             'best_seen': 'BestSeen', 'best_unseen': 'BestUnseen', 'food_mean': '种群平均Food'}
    print(f"\n各运行: " + " | ".join(
        f"{r['mode']}" + (f"x{r['food_scale']:g}" if r['food_scale'] != 1.0 else '') for r in rows))
    for k in keys:
        print(f"{names[k]:<22}" + "".join(f"{r[k]:>14.4f}" for r in rows))


if __name__ == '__main__':
    main()
