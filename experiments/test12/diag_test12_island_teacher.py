# 诊断 2：用 VectorCycleTeacher（有序参考解法器）驱动 test12 环境，
# 检验"好蛇"（按设计不困死自己）是否被孤岛惩罚误伤 → 反向压选择。
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch

from test12 import (Config, BatchedSnakeEnv, VectorCycleTeacher, reach_ratio,
                    make_banks, _fitness_econ)


def run_teacher_episodes(cfg, n_ep, gen=0):
    dev = torch.device(cfg.DEVICE) if cfg.DEVICE else torch.device('cpu')
    banks = make_banks(cfg, gen, 1, n_ep, dev)
    bank = {'stream': torch.stack([b['stream'] for b in banks]),
            'dir0': torch.tensor([b['dir0'] for b in banks], dtype=torch.long, device=dev)}
    B = n_ep
    env = BatchedSnakeEnv(cfg, B, dev)
    env.reset(bank=bank)
    teacher = VectorCycleTeacher(cfg.GRID_SIZE, dev)
    ar = torch.arange(B, device=dev)
    min_reach = torch.full((B,), 1e9, device=dev)
    food = torch.zeros(B, device=dev)
    for t in range(cfg.MAX_STEPS):
        al = env.alive
        necks = env.body[ar, 1]
        act = teacher.act(env.head, env.food, env.body, env.body_len, env.dir_idx, necks)
        env.step(act)
        ate = al & env.ate
        food += ate.float()
        if ate.any():
            rr = reach_ratio(env)
            min_reach = torch.where(ate & (rr < min_reach), rr, min_reach)
        if env.all_done():
            break
    flag = (min_reach < cfg.ISLAND_THRESHOLD) & (food > 0)
    return (food.cpu().numpy(), min_reach.cpu().numpy().clip(max=1.0),
            flag.cpu().numpy())


def main():
    cfg = Config()
    cfg.MAX_STEPS = 100000
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.DEVICE = dev
    food, minr, flag = run_teacher_episodes(cfg, n_ep=40)
    print(f"教师(有序解法器) 40 局 | 阈值 {cfg.ISLAND_THRESHOLD} 罚 {cfg.ISLAND_PENALTY}")
    print(f"food: 中位 {np.median(food):.0f} 均值 {food.mean():.1f} max {food.max():.0f}")
    print(f"min_reach: 中位 {np.median(minr):.2f} | min {minr.min():.2f}")
    print(f"孤岛触发率(局级): {flag.mean():.1%}  触发局的 food 均值 {food[flag].mean() if flag.any() else '-':.1f}")
    # 个体级后果模拟：13 局(K1=3+K2=10)教师水平个体，只要 ≥1 局触发 → 全部 ×0.1
    p = max(flag.mean(), 1e-9)
    p_any = 1 - (1 - p) ** 13
    print(f"若每局触发率 {p:.1%}，13 局个体'任一局触发'概率 ≈ {p_any:.1%}")
    # 排序反转点：被标记的 F 食蛇 vs 未标记的 1 食蛇
    for F in (2, 3, 5, 8):
        m_hi = np.zeros(15); m_hi[0] = F; m_hi[3] = F * 10; m_hi[10] = F * 10 / 3.0
        m_lo = np.zeros(15); m_lo[0] = 1; m_lo[3] = 10; m_lo[10] = 10 / 3.0
        f_hi_flag = _fitness_econ(np.concatenate([m_hi[:14], [1.0]]), cfg)
        f_lo_clean = _fitness_econ(np.concatenate([m_lo[:14], [0.0]]), cfg)
        print(f"  被标记 {F} 食蛇 fit={f_hi_flag:.2f}  vs  未标记 1 食蛇 fit={f_lo_clean:.2f}"
              f"  → {'反转' if f_hi_flag < f_lo_clean else '正常'}")


if __name__ == '__main__':
    main()
