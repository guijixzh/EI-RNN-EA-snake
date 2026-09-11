# -*- coding: utf-8 -*-
"""
analyze_signal_threshold.py — 信号自举阈值精确测量
（验证判据: 输入信号振幅 a × W_in 初始化标准差 σ_win 产生的 logit 扰动 > 决策边界 M）

测量内容（随机初始化网络，POP 256，32proj 观测，全部缩放=1，手动注入振幅）：
  1. 决策边界 M = K帧求和 logits 的 top1-top2 差分布
  2. 单通道级联增益 g：把指定通道从 0 置为 a 后 |Δlogit| 分布
  3. 翻转率曲线 P(argmax 翻转) 随 a
  4. 阈值 a*（翻转率>=5% 的最小振幅）与 a*·σ_win、M/g 的对照

用法: python experiments/analyze_signal_threshold.py [POP]
"""
import os
import sys

import numpy as np
import torch

import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SPEC = importlib.util.spec_from_file_location('t7a', os.path.join(ROOT, 'experiments', 'test7_series', 'test7a.py'))
t7a = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(t7a)

AMPLITUDES = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
CHANNELS = [(8, '食物·前'), (24, '障碍·前'), (0, '头方向bit')]
SIGMA_WIN = 0.1   # W_in 初始化标准差（Config: W_IN_STD）


def deliberate_logits(pop, obs, E, I, st, cts, cfg):
    ls = None
    for k in range(cfg.FRAME_RATE):
        o = obs * (cfg.INPUT_DECAY ** k)
        lg, E, I, st = t7a.forward_batch(pop, o, E, I, st, cts, cfg)
        ls = lg if ls is None else ls + lg
    return ls


def main():
    pop_size = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = t7a.Config()
    cfg.DEVICE = str(device)
    cfg.USE_FP16 = False
    cfg.POP_SIZE = pop_size
    cfg.OBS_FOOD_SCALE = 1.0
    cfg.OBS_SELF_SCALE = 1.0
    cfg.OBS_OBSTACLE_SCALE = 1.0

    torch.manual_seed(1234)
    pop = t7a.GeneStack(cfg, device=device)
    pop.random_init()
    pop.refresh_eff()
    env = t7a.BatchedSnakeEnv(cfg, pop_size, device)
    B, N, A = pop_size, cfg.NUM_COLUMNS, cfg.ACTION_DIM
    dev = device

    margins = []
    dlogit = {(ch, a): [] for ch, _ in CHANNELS for a in AMPLITUDES}
    win = {(ch, a): 0 for ch, _ in CHANNELS for a in AMPLITUDES}
    trials = 0

    for _ep in range(2):
        env.reset()
        E = torch.zeros(B, N, device=dev)
        I = torch.zeros(B, N, device=dev)
        st = torch.zeros(B, N, device=dev)
        cts = torch.zeros(B, A, device=dev)
        for _t in range(cfg.MAX_STEPS):
            al = env.alive
            obs = env.obs()
            base = deliberate_logits(pop, obs, E, I, st, cts, cfg)
            act, E, I, st = t7a.deliberate_batch(pop, obs, E, I, st, cts, cfg)
            cts = t7a.update_fatigue(cts, act)
            if _t % 5 == 0 and al.any():
                trials += 1
                top2 = torch.topk(base, 2, dim=1).values
                margins.append((top2[:, 0] - top2[:, 1])[al].cpu().numpy())
                for ch, _name in CHANNELS:
                    for a in AMPLITUDES:
                        o2 = obs.clone()
                        o2[:, ch] = o2[:, ch] + a
                        ls = deliberate_logits(pop, o2, E, I, st, cts, cfg)
                        dlogit[(ch, a)].append((ls - base)[al].abs().cpu().numpy())
                        win[(ch, a)] += int(((ls.argmax(1) != base.argmax(1)) & al).sum())
            env.step(act)
            if env.all_done():
                break

    margins = np.concatenate(margins)
    n_ind = trials * pop_size
    print(f"[threshold] POP={pop_size} 采样步={trials} | σ_win={SIGMA_WIN}")
    print(f"决策边界 M (top1-top2): median={np.median(margins):.4f} "
          f"P25={np.percentile(margins,25):.4f} P75={np.percentile(margins,75):.4f} "
          f"P90={np.percentile(margins,90):.4f}\n")

    hdr = f"{'振幅a':>7}{'a·σ_win':>9}"
    for ch, name in CHANNELS:
        hdr += f"{'|Δlogit|中位':>12}{name + '翻转率':>12}"
    print(hdr)
    for a in AMPLITUDES:
        row = f"{a:>7.2f}{a * SIGMA_WIN:>9.3f}"
        for ch, _name in CHANNELS:
            dl = np.concatenate(dlogit[(ch, a)])
            row += f"{np.median(dl):>12.4f}{win[(ch, a)] / n_ind:>12.4f}"
        print(row)

    print()
    results = {}
    for ch, name in CHANNELS:
        a_star = None
        for a in AMPLITUDES:
            if win[(ch, a)] / n_ind >= 0.05:
                a_star = a
                break
        dl32 = np.concatenate(dlogit[(ch, 3.2)])
        g = np.median(dl32) / 3.2
        results[ch] = (a_star, g)
        if a_star is not None:
            print(f"{name}: 阈值 a*≈{a_star}  (a*·σ_win={a_star * SIGMA_WIN:.3f}, "
                  f"g={g:.4f}, M/g={np.median(margins) / g:.2f})")
        else:
            print(f"{name}: 全部振幅翻转率<5% (g={g:.4f}, M/g={np.median(margins) / g:.2f} — 理论阈值超出扫描范围)")

    print(f"\n判据验证: a*·σ_win vs M/g —— 两者应同量级（a*·σ_win 是输入侧乘积, "
          f"M/g 是网络侧阈值换算回振幅）。")
    print(f"对照当前默认(×8)通道幅度: 食物均值0.78(峰值8.0) 自身~1.0 障碍~1.82。")


if __name__ == '__main__':
    main()
