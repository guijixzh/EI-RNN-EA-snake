# -*- coding: utf-8 -*-
"""
measure_lunar_threshold.py — LunarLander-v3 各输入通道自举阈值实测
（test10 定标前置实验；方法移植自 experiments/analyze_signal_threshold.py）

判据: 归一化观测中通道振幅 a × σ_win(0.1) 引起的 K 帧求和 logits 扰动
      必须超过决策边界 M (top1-top2)，随机种群才存在可被选择梯度放大的反射。

测量方式：
  1. N=256、σ_win=0.1、INIT_DENSITY=0.15 的随机 EI-RNN 种群（与 test7b/test10 一致）
  2. 64 个并行 LunarLander-v3 环境以随机策略滚动采样真实状态分布
  3. 观测先逐维归一到 ±1（/OBS_RANGES），再对单通道注入振幅 a，扫描翻转率
  4. 输出每通道阈值 a*（翻转率>=5% 的最小振幅）与推荐 CHANNEL_SCALES

用法: python experiments/measure_lunar_threshold.py [POP]
"""
import json
import os
import sys

import numpy as np
import torch

import gymnasium as gym

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 与 test10_lunar.py 保持一致（归一化物理范围）
OBS_RANGES = [1.5, 1.5, 5.0, 5.0, np.pi, 10.0, 1.0, 1.0]
CHANNEL_NAMES = ['x位置', 'y高度', 'vx横速', 'vy纵速', '角度', '角速度', '左腿', '右腿']
AMPLITUDES = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
SIGMA_WIN = 0.1
NUM_COLUMNS = 256
INIT_DENSITY = 0.15
BASE_TAU_E = 0.7
W_EI = W_IE = 2.0
FRAME_RATE = 5
INPUT_DECAY = 0.9
SHORT_TERM_GAIN = -0.2
SHORT_TERM_DECAY = 0.3


def build_pop(B, device):
    g = torch.Generator(device='cpu').manual_seed(1234)
    torch.manual_seed(1234)
    O, A, N = 8, 4, NUM_COLUMNS
    st = {}
    st['M_in'] = (torch.rand(B, N, O, generator=g) < INIT_DENSITY).float().to(device)
    M_rec = (torch.rand(B, N, N, generator=g) < INIT_DENSITY).float()
    M_rec[:, torch.eye(N, dtype=torch.bool)] = 0.0
    st['M_rec'] = M_rec.to(device)
    st['M_out'] = (torch.rand(B, A, N, generator=g) < INIT_DENSITY).float().to(device)
    st['W_in'] = (torch.randn(B, N, O, generator=g) * SIGMA_WIN).to(device)
    st['W_rec'] = (torch.randn(B, N, N, generator=g) * 0.05).to(device)
    st['W_out'] = (torch.randn(B, A, N, generator=g) * SIGMA_WIN).to(device)
    st['b_out'] = torch.zeros(B, A, device=device)
    st['tau_e'] = (BASE_TAU_E + (torch.rand(B, N, generator=g) * 2 - 1) * 0.1).to(device)
    st['w_ei'] = torch.full((B, N), W_EI, device=device)
    st['w_ie'] = torch.full((B, N), W_IE, device=device)
    for k in ('M_in', 'M_rec', 'M_out', 'W_in', 'W_rec', 'W_out'):
        st[k + '_eff'] = st[k]  # 同名缓存，前向直接用 eff 权重
    return st


def forward(pop, obs, E, I):
    ext = torch.bmm(pop['W_in_eff'], obs.unsqueeze(-1)).squeeze(-1)
    rec = torch.bmm(pop['W_rec_eff'], E.unsqueeze(-1)).squeeze(-1)
    total = ext + rec
    E_new = torch.sigmoid(total + pop['tau_e'] * E - pop['w_ei'] * I)
    I_new = torch.sigmoid(pop['w_ie'] * E_new)
    logits = torch.bmm(pop['W_out_eff'], E_new.unsqueeze(-1)).squeeze(-1) + pop['b_out']
    return logits, E_new, I_new


def deliberate_logits(pop, obs, E, I):
    ls = None
    for k in range(FRAME_RATE):
        lg, E, I = forward(pop, obs * (INPUT_DECAY ** k), E, I)
        ls = lg if ls is None else ls + lg
    return ls


def normalize(obs_np):
    return obs_np / np.asarray(OBS_RANGES, dtype=np.float32)[None, :]


def main():
    pop_size = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B = pop_size
    pop = build_pop(B, device)

    envs = [gym.make('LunarLander-v3') for _ in range(B)]
    obs_np = np.stack([e.reset(seed=i)[0] for i, e in enumerate(envs)]).astype(np.float32)

    margins = []
    dlogit = {(ch, a): [] for ch in range(8) for a in AMPLITUDES}
    wins = {(ch, a): 0 for ch in range(8) for a in AMPLITUDES}
    trials = 0
    steps_per_sample = 5

    E = torch.zeros(B, NUM_COLUMNS, device=device)
    I = torch.zeros(B, NUM_COLUMNS, device=device)
    for ep in range(3):
        for t in range(300):
            obs = torch.from_numpy(normalize(obs_np)).to(device)
            base = deliberate_logits(pop, obs, E, I)
            if t % steps_per_sample == 0:
                trials += 1
                top2 = torch.topk(base, 2, dim=1).values
                margins.append((top2[:, 0] - top2[:, 1]).cpu().numpy())
                with torch.no_grad():
                    for ch in range(8):
                        for a in AMPLITUDES:
                            o2 = obs.clone()
                            o2[:, ch] = o2[:, ch] + a
                            ls = deliberate_logits(pop, o2, E, I)
                            dlogit[(ch, a)].append((ls - base).abs().cpu().numpy())
                            wins[(ch, a)] += int((ls.argmax(1) != base.argmax(1)).sum())
            actions = np.random.randint(0, 4, size=B)
            new_obs = np.zeros_like(obs_np)
            for i, e in enumerate(envs):
                o, r, term, trunc, _ = e.step(actions[i])
                if term or trunc:
                    o, _ = e.reset()
                new_obs[i] = o
            obs_np = new_obs.astype(np.float32)

    margins = np.concatenate(margins)
    n_ind = trials * B
    print(f"[threshold] POP={B} 采样步={trials} | sigma_win={SIGMA_WIN} | 归一化单位")
    print(f"决策边界 M (top1-top2): median={np.median(margins):.4f} "
          f"P25={np.percentile(margins,25):.4f} P75={np.percentile(margins,75):.4f} "
          f"P90={np.percentile(margins,90):.4f}\n")

    results = {'M_median': float(np.median(margins)), 'channels': {}}
    for ch in range(8):
        a_star = None
        for a in AMPLITUDES:
            if wins[(ch, a)] / n_ind >= 0.05:
                a_star = a
                break
        dl16 = np.concatenate(dlogit[(ch, 1.6)])
        ggain = np.median(dl16) / 1.6
        name = CHANNEL_NAMES[ch]
        row = f"{name:>6}: "
        row += " ".join(f"{a:.2f}->{wins[(ch,a)]/n_ind:.3f}" for a in AMPLITUDES)
        print(row)
        rec = max(2.0 * a_star, 1.0) if a_star is not None else 12.8
        results['channels'][name] = {
            'a_star': a_star, 'a_star_sigma': None if a_star is None else a_star * SIGMA_WIN,
            'gain': float(ggain), 'M_over_g': float(np.median(margins) / ggain),
            'recommended_scale': float(rec),
        }
        if a_star is not None:
            print(f"          阈值 a*≈{a_star} (a*·σ_win={a_star*SIGMA_WIN:.3f}, "
                  f"g={ggain:.4f}, M/g={np.median(margins)/ggain:.2f}) -> 推荐 scale {rec}")
        else:
            print(f"          全部振幅翻转率<5% (g={ggain:.4f}, "
                  f"M/g={np.median(margins)/ggain:.2f}) -> 推荐 scale 上限 12.8")

    out_dir = os.path.join(ROOT, 'results')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'test10_channel_threshold.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存 -> results/test10_channel_threshold.json")

    for e in envs:
        e.close()


if __name__ == '__main__':
    main()
