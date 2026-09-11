# E1：习惯三因素指标判别力验证（设计文档 experiments 方案）。
# 教师解法器（全局回+捷径规划）理应表现出高 edge_pref / 高 conn；
# best 模型（局部观测）作为对照。同库 40 局，指标方向与区分度即为判定。
# 用法: python experiments/exp_habit_metrics.py [--n 40] [--model artifacts/test12/test12_econ_best_model.pth]
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch

import test12 as t12


def run_tracked(model_pop=None, teacher=None, cfg=None, bank=None, B=0, dev=None):
    """同库跑策略，逐存活步累计三因素遥测（与 _eval_sweep_chunk 同口径）。"""
    env = t12.BatchedSnakeEnv(cfg, B, dev)
    env.reset(bank=bank)
    ar = torch.arange(B, device=dev)
    ep_sum = torch.zeros(B, device=dev)   # edge_share（与适应度同源）
    conn_hit = torch.zeros(B, device=dev)
    conn_n = torch.zeros(B, device=dev)
    steps = torch.zeros(B, device=dev)
    turns = torch.zeros(B, device=dev)
    food = torch.zeros(B, device=dev)
    E = torch.zeros(B, cfg.NUM_COLUMNS,
                    dtype=torch.float16 if getattr(cfg, 'USE_FP16', True) else torch.float32,
                    device=dev)
    I = torch.zeros_like(E)
    stt = torch.zeros_like(E)
    press = torch.zeros(B, dtype=torch.float32, device=dev)
    for t in range(cfg.MAX_STEPS):
        al = env.alive
        obs = env.obs().to(E.dtype)
        if teacher is not None:
            necks = env.body[ar, 1]
            act = teacher.act(env.head, env.food, env.body, env.body_len,
                              env.dir_idx, necks)
        else:
            act, E, I, stt = t12.deliberate_batch(model_pop, obs, E, I, stt,
                                                  press, cfg)
        ep_sum += al.float() * t12.habit_edge_share(env)
        if t % 4 == 0:
            conn_hit += al.float() * t12.habit_conn_score(env)
            conn_n += al.float()
        steps += al.float()
        turns += (al & (act != 0)).float()
        env.step(act)
        food += (al & env.ate).float()
        if env.all_done():
            break
    alive = steps > 0
    return dict(
        food=food.cpu().numpy(),
        epref=(ep_sum / steps.clamp(min=1)).cpu().numpy(),
        conn=np.where(alive.cpu().numpy(), (conn_hit / conn_n.clamp(min=1)).cpu().numpy(), np.nan),
        straight=(1 - turns / steps.clamp(min=1)).cpu().numpy(),
    )


def report(name, r):
    f = r['food']
    sel = f > 0
    print(f"\n===== {name}（N={len(f)}，food>0 局 {int(sel.sum())}）=====")
    H = (r['epref'] + np.nan_to_num(r['conn'], nan=1.0) + r['straight']) / 3.0
    fit = f + 0.3 * f / 200.0 + 0.6 * H     # eff 项用整局近似，仅作量级参照
    print(f"food 中位 {np.median(f):.0f} | "
          f"edge_share 均值 {r['epref'][sel].mean():.3f} | "
          f"conn_score 均值 {np.nanmean(r['conn'][sel]):.3f} | "
          f"straight 均值 {r['straight'][sel].mean():.3f} | "
          f"H 均值 {H[sel].mean():.3f} | v6fit 近似均值 {fit[sel].mean():.1f}")
    return {k: (v[sel].mean() if k != 'conn' else np.nanmean(v[sel]))
            for k, v in r.items() if k != 'food'} | {'food': f[sel].mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--model', default='artifacts/test12/test12_econ_best_model.pth')
    args = ap.parse_args()

    cfg = t12.Config()
    cfg.MAX_STEPS = 100000
    dev = t12._resolve_device(cfg)
    B = args.n
    banks = t12.make_banks(cfg, 0, 1, B, dev)
    bank = {'stream': torch.stack([b['stream'] for b in banks]),
            'dir0': torch.tensor([b['dir0'] for b in banks],
                                 dtype=torch.long, device=dev)}

    teacher = t12.VectorCycleTeacher(cfg.GRID_SIZE, dev)
    r_t = run_tracked(teacher=teacher, cfg=cfg, bank=bank, B=B, dev=dev)
    s_t = report('教师（全局回+捷径）', r_t)

    res = t12.load_best_state(args.model, cfg)
    if res is None:
        print('[跳过] best 模型不兼容')
        return
    st = res[0]
    pop = t12.GeneStack(cfg, B=B, device=dev)
    pop.random_init()
    for i in range(B):
        pop.set_individual_from_state(i, st)
    pop.refresh_eff()
    if getattr(cfg, 'USE_FP16', True):
        pop.fp16()
        pop.refresh_eff()
    r_m = run_tracked(model_pop=pop, cfg=cfg, bank=bank, B=B, dev=dev)
    s_m = report(f'best 模型（{args.model}）', r_m)

    print("\n===== E1 判定 =====")
    print(f"edge_share: 教师 {s_t['epref']:.3f} vs 模型 {s_m['epref']:.3f} "
          f"→ {'符合预期(教师更高)' if s_t['epref'] > s_m['epref'] else '不符合预期'}")
    print(f"conn_score: 教师 {s_t['conn']:.3f} vs 模型 {s_m['conn']:.3f} "
          f"→ {'符合预期(教师更高)' if s_t['conn'] > s_m['conn'] else '不符合预期'}")
    print(f"straight:  教师 {s_t['straight']:.3f} vs 模型 {s_m['straight']:.3f} "
          f"→ {'符合预期(教师更高)' if s_t['straight'] > s_m['straight'] else '不符合预期'}")


if __name__ == '__main__':
    main()
