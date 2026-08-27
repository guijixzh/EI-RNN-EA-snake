# -*- coding: utf-8 -*-
"""
diagnose_blocks.py — 通道块可读性化验（回答"哪些输入白输了"）

POP 个随机个体在 32proj 观测（food=8 默认）下运行，对 5 个通道块分别测量：
  - 平均|obs|：该块信号强度
  - 驱动贡献 |W_in_eff·obs_block|：该块对网络输入的实际贡献
  - 置零翻转率：该块置 0 后动作 argmax 翻转的概率（网络当前是否在用它）
  - 放大翻转率：该块 ×4 后动作翻转概率（随机网络能否读它；食物块=已验证可读基准）

用法: python experiments/diagnose_blocks.py [POP] [EPISODES]
"""
import os
import sys

import numpy as np
import torch

import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location('t7a', os.path.join(ROOT, 'test7a.py'))
t7a = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(t7a)

BLOCKS = [('头onehot', 0, 4), ('尾onehot', 4, 8), ('食物x8', 8, 16),
          ('自身', 16, 24), ('障碍', 24, 32)]


def main():
    pop_size = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    ck_path = sys.argv[3] if len(sys.argv) > 3 else None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = t7a.Config()
    cfg.DEVICE = str(device)
    cfg.USE_FP16 = False
    cfg.POP_SIZE = pop_size

    if ck_path and os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location='cpu', weights_only=False)
        saved = ck.get('config', {})
        for k in ('OBS_FOOD_SCALE', 'OBS_SELF_SCALE', 'OBS_OBSTACLE_SCALE'):
            if k in saved:
                setattr(cfg, k, saved[k])
        pop = t7a.GeneStack(cfg, device=device)
        pop.unpack(ck['pop'])
        pop_size = pop.P
        cfg.POP_SIZE = pop_size
        print(f"[blocks] 已加载进化种群 {ck_path} (next_gen={ck.get('next_gen')})")
    else:
        torch.manual_seed(1234)
        pop = t7a.GeneStack(cfg, device=device)
        pop.random_init()
    pop.refresh_eff()

    env = t7a.BatchedSnakeEnv(cfg, pop_size, device)
    B, N, A = pop_size, cfg.NUM_COLUMNS, cfg.ACTION_DIM
    dev = device

    abs_sum = np.zeros(32)
    obs_steps = 0
    drive_sum = {name: 0.0 for name, _, _ in BLOCKS}
    zero_flips = {name: 0 for name, _, _ in BLOCKS}
    amp_flips = {name: 0 for name, _, _ in BLOCKS}
    trials = 0

    for _ep in range(episodes):
        env.reset()
        E = torch.zeros(B, N, device=dev)
        I = torch.zeros(B, N, device=dev)
        st = torch.zeros(B, N, device=dev)
        cts = torch.zeros(B, A, device=dev)

        for _t in range(cfg.MAX_STEPS):
            al = env.alive
            obs = env.obs()
            abs_sum += obs.abs().mean(dim=0).cpu().numpy() * float(al.float().mean())
            obs_steps += 1

            act, E, I, st = t7a.deliberate_batch(pop, obs, E, I, st, cts, cfg)
            cts = t7a.update_fatigue(cts, act)

            # 每步对每块做驱动贡献/置零/放大敏感性测试（重算 K 帧思考和原动作比）
            if _t % 10 == 0 and al.any():
                trials += 1
                with torch.no_grad():
                    for name, lo, hi in BLOCKS:
                        ob = torch.zeros_like(obs)
                        ob[:, lo:hi] = obs[:, lo:hi]
                        d = torch.bmm(pop.W_in_eff, ob.unsqueeze(-1).float()).squeeze(-1)
                        drive_sum[name] += d.abs().mean().item()
                for name, lo, hi in BLOCKS:
                    for mode in ('zero', 'amp'):
                        o2 = obs.clone()
                        if mode == 'zero':
                            o2[:, lo:hi] = 0.0
                        else:
                            o2[:, lo:hi] = o2[:, lo:hi] * 4.0
                        with torch.no_grad():
                            ls = None
                            E2, I2, st2 = E, I, st
                            for k in range(cfg.FRAME_RATE):
                                ok = o2 * (cfg.INPUT_DECAY ** k)
                                lg, E2, I2, st2 = t7a.forward_batch(pop, ok, E2, I2, st2, cts, cfg)
                                ls = lg if ls is None else ls + lg
                            flipped = (torch.argmax(ls, 1) != act).float()
                            w = al.float()
                            frac = (flipped * w).sum().item() / max(1.0, w.sum().item())
                            if mode == 'zero':
                                zero_flips[name] += frac
                            else:
                                amp_flips[name] += frac
            if env.all_done():
                break

    print(f"[blocks] device={device} POP={pop_size} eps={episodes} trials={trials}\n")
    print(f"{'块':<10}{'平均|obs|':>10}{'驱动贡献':>10}{'置零翻转率':>12}{'放大x4翻转率':>14}")
    print('-' * 58)
    for name, lo, hi in BLOCKS:
        print(f"{name:<10}{abs_sum[lo:hi].mean() / max(1, obs_steps):>10.3f}"
              f"{drive_sum[name] / max(1, trials):>10.4f}"
              f"{zero_flips[name] / max(1, trials):>12.4f}"
              f"{amp_flips[name] / max(1, trials):>14.4f}")
    print("\n判读: 放大翻转率低=随机网络读不动该块(白输入); 置零翻转率高=进化已在使用该块。")
    print("食物x8 块为已验证可读基准。")


if __name__ == '__main__':
    main()
