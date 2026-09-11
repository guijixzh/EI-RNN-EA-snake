# 评估 test12 best 模型：N 局终局报告 + 每局孤岛曲线（吃食采样点 reach_ratio）
# 用法: python experiments/eval_best12_islands.py [--model artifacts/test12/test12_econ_best_model.pth] [--episodes 10]
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

import test12 as t12

DEATH = {0: '存活(到步数上限)', 1: '撞墙', 2: '撞己', 3: '饿死'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='artifacts/test12/test12_econ_best_model.pth')
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--max-steps', type=int, default=100000)
    args = ap.parse_args()

    cfg = t12.Config()
    cfg.MAX_STEPS = args.max_steps
    res = t12.load_best_state(args.model, cfg)
    if res is None:
        sys.exit('[错误] 模型与当前配置/编码不兼容（见上方警告）')
    st, food0, steps0 = res
    dev = t12._resolve_device(cfg)
    B = args.episodes
    pop = t12.GeneStack(cfg, B=B, device=dev)
    pop.random_init()
    for i in range(B):
        pop.set_individual_from_state(i, st)
    pop.refresh_eff()
    if getattr(cfg, 'USE_FP16', True):
        pop.fp16()
        pop.refresh_eff()

    banks = t12.make_banks(cfg, 0, 1, B, dev)
    bank = {'stream': torch.stack([b['stream'] for b in banks]),
            'dir0': torch.tensor([b['dir0'] for b in banks],
                                 dtype=torch.long, device=dev)}
    env = t12.BatchedSnakeEnv(cfg, B, dev)
    env.reset(bank=bank)
    E = torch.zeros(B, pop.N, dtype=pop.dtype, device=dev)
    I = torch.zeros_like(E)
    stt = torch.zeros_like(E)
    press = torch.zeros(B, dtype=torch.float32, device=dev)

    thr = float(cfg.ISLAND_THRESHOLD)
    pen = float(cfg.ISLAND_PENALTY)
    curves = [[] for _ in range(B)]
    for _ in range(cfg.MAX_STEPS):
        al = env.alive
        obs = env.obs().to(pop.dtype)
        act, E, I, stt = t12.deliberate_batch(pop, obs, E, I, stt, press, cfg)
        env.step(act)
        ate = al & env.ate
        if ate.any():
            rr = t12.reach_ratio(env)
            for i in range(B):
                if bool(ate[i]):
                    curves[i].append(float(rr[i]))
        if env.all_done():
            break

    print(f"\n模型: {args.model} (载入时 food={food0:.2f}) | enc="
          f"{cfg.OBS_ENC_VERSION}/{getattr(cfg, 'OBS_FOOD_FRAME', 'ego')} | "
          f"孤岛阈值 {thr} | 惩罚率口径 pen={pen}（该模型训练时 pen=1.0=关闭，"
          f"以下为按默认 0.1 的假想折减；触发口径与训练一致=局级二值："
          f"该局任一采样点低于阈值 → 该局记 1）")
    print(f"{'局':>3} | {'food':>4} | {'蛇长':>3} | {'步数':>6} | 终局 | "
          f"孤岛曲线(*=低于阈值{thr}) | 局标记 | 局惩罚后系数")
    flags = []
    for i in range(B):
        cur = curves[i]
        n = len(cur)
        flag = 1 if any(v < thr for v in cur) else 0
        factor = 1.0 - (1.0 - pen) * flag      # 局级二值 → 触发局 ×pen
        flags.append(flag)
        curve_s = ' '.join(f"{v:.2f}{'*' if v < thr else ''}" for v in cur) or '(未吃食)'
        print(f"{i:>3} | {n:>4} | {int(env.body_len[i]):>3} | {int(env.steps[i]):>6} | "
              f"{DEATH[int(env.died[i])]} | {curve_s} | {flag:>2} | ×{factor:.2f}")
    flags = np.array(flags)
    print(f"\n汇总: 局均触发率={flags.mean():.2%} | "
          f"个体级惩罚后系数 = 1−(1−pen)×{flags.mean():.2f} = "
          f"×{1 - (1 - pen) * flags.mean():.3f} | "
          f"零触发局={int((flags == 0).sum())}/{B}")


if __name__ == '__main__':
    main()
