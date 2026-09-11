# E2c：适应度 v6（food+eff+0.6·H）vs v5（food+eff）演化对照（预注册）。
# 256 种群 × 20 代 × 种子 {0,1}，同 CRN。预注册判定（公式重定义后首次演化验证）：
#   P1 食物不损（单侧不劣，修正 E2 对称判据的方向错误）：
#      gen-20 avg_food(v6) ≥ avg_food(v5) − 0.5 且 best_food(v6) ≥ best_food(v5) − max(5%, 1食)
#   P2 习惯有效：gen-20 精英组 H(v6) > H(v5) + 0.05（分因素报告）
#   P3 健康：精英组撞己占比 v6 ≤ v5 + 0.1
#   P4 食物优先保持：精英组 Kendall τ(food, fitness) v6 ≥ v5 − 0.02
#      （习惯项没有把低食个体抬进精英）
# 用法: python experiments/exp_fitness_v6_ab.py [--gens 20] [--seeds 0 1]
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch

import test12 as t12


def make_cfg(v6):
    cfg = t12.Config()
    cfg.POP_SIZE, cfg.NUM_COLUMNS = 256, 96
    cfg.ELITE_SIZE, cfg.STAGE1_EPS, cfg.EVAL_EPISODES, cfg.STAGE2_KEEP = 64, 2, 4, 64
    cfg.MAX_STEPS, cfg.EVAL_BATCH = 3000, 256
    cfg.HABIT_W = 0.6 if v6 else 0.0
    cfg.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    return cfg


def kendall_tau(a, b):
    n = len(a)
    c = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            c += s
    return c / (n * (n - 1) * 0.5)


def run_arm(cfg, seed, gens):
    torch.manual_seed(seed)
    dev = t12._resolve_device(cfg)
    pop = t12.GeneStack(cfg, device=dev)
    pop.random_init()
    curve = []
    for gen in range(gens):
        metrics, order = t12.evaluate_population_gpu(pop, cfg, gen=gen)
        mn = metrics.numpy()
        curve.append((mn[:, 0].max(), mn[:, 0].mean()))
        print(f"  gen {gen}: best={mn[:, 0].max():.2f} avg={mn[:, 0].mean():.3f}",
              flush=True)
        if gen < gens - 1:
            pop = t12.evolve_topology_gpu(pop, metrics, cfg, gen=gen, order=order)
    elite = mn[order[:cfg.ELITE_SIZE]]
    sel = elite[:, 0] > 0
    H = (elite[:, 18] + elite[:, 16] + elite[:, 17]) / 3.0
    tau = kendall_tau(elite[sel, 0], (elite[sel, 0] + 0.3 * elite[sel, 0]
                                     / np.maximum(elite[sel, 3], 1)
                                     + cfg.HABIT_W * H[sel]))
    return (np.array(curve),
            dict(H=float(H[sel].mean()), epref=float(elite[sel, 18].mean()),
                 conn=float(elite[sel, 16].mean()),
                 straight=float(elite[sel, 17].mean()),
                 food=float(elite[sel, 0].mean()),
                 self_die=float(mn[:, 6].mean()), tau=float(tau)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gens', type=int, default=20)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    args = ap.parse_args()

    results = {}
    for arm, v6 in (('v5', False), ('v6', True)):
        cfg = make_cfg(v6)
        for seed in args.seeds:
            print(f"\n=== 臂 {arm} seed {seed}（HABIT_W={cfg.HABIT_W}）===",
                  flush=True)
            curve, em = run_arm(cfg, seed, args.gens)
            results[(arm, seed)] = (curve, em)

    print("\n================ E2c 预注册判定 ================")
    p1_ok = p2_ok = p3_ok = p4_ok = True
    for seed in args.seeds:
        c5, m5 = results[('v5', seed)]
        c6, m6 = results[('v6', seed)]
        d_avg = c6[-1, 1] - c5[-1, 1]
        d_best = c6[-1, 0] - c5[-1, 0]
        ok1 = (d_avg >= -0.5) and (d_best >= -max(0.05 * max(c5[-1, 0], 1), 1.0))
        p1_ok &= ok1
        print(f"[P1 食物不损] seed{seed}: Δavg={d_avg:+.3f}(≥-0.5) "
              f"Δbest={d_best:+.2f}(≥-{max(0.05 * max(c5[-1, 0], 1), 1.0):.2f}) "
              f"→ {'PASS' if ok1 else 'FAIL'}")
        dH = m6['H'] - m5['H']
        ok2 = dH > 0.05
        p2_ok &= ok2
        print(f"[P2 习惯有效] seed{seed}: H {m5['H']:.3f}→{m6['H']:.3f} "
              f"(Δ{dH:+.3f}>0.05) → {'PASS' if ok2 else 'FAIL'} | "
              f"分因素 edge {m5['epref']:.2f}→{m6['epref']:.2f} conn "
              f"{m5['conn']:.2f}→{m6['conn']:.2f} straight "
              f"{m5['straight']:.2f}→{m6['straight']:.2f}")
        ok3 = m6['self_die'] <= m5['self_die'] + 0.1
        p3_ok &= ok3
        print(f"[P3 健康  ] seed{seed}: 撞己 {m5['self_die']:.2f}→{m6['self_die']:.2f} "
              f"→ {'PASS' if ok3 else 'FAIL'}")
        dtau = m6['tau'] - m5['tau']
        ok4 = dtau >= -0.02
        p4_ok &= ok4
        print(f"[P4 食物优先] seed{seed}: τ {m5['tau']:.3f}→{m6['tau']:.3f} "
              f"(Δ{dtau:+.3f}≥-0.02) → {'PASS' if ok4 else 'FAIL'}")
    ok = p1_ok and p2_ok and p3_ok and p4_ok
    print(f"\nE2c 总判定: {'通过' if ok else '未通过'}"
          f"（P1={p1_ok} P2={p2_ok} P3={p3_ok} P4={p4_ok}）")


if __name__ == '__main__':
    main()
