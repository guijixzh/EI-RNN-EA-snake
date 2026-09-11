# E2：习惯三分立配额 A/B（预注册实验）。
# 臂 OFF = 适应度 v5（food+eff），习惯配额全关；
# 臂 ALL = 同适应度 + HABIT_EDGE/CONN/STRAIGHT_ELITE 各 8。
# 种子 {0,1}，256 种群 × 12 代，同 CRN。
# 预注册判定（实验前定，不可事后改）：
#   P1 不干扰：同种子 gen-12 |Δavg_food| ≤ 0.5 且 |Δbest_food| ≤ max(5%, 1食)；
#   P2 有效：ALL 臂 gen-12 精英组（适应度 top-ELITE）三因素指标均值
#      均 > OFF 臂 + 0.05（绝对值）；
#   P3 健康：ALL 臂终局撞己占比不高于 OFF + 0.1。
# 用法: python experiments/exp_habit_ab.py [--gens 12] [--seeds 0 1]
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch

import test12 as t12


def make_cfg(habit):
    cfg = t12.Config()
    cfg.POP_SIZE, cfg.NUM_COLUMNS = 256, 96
    cfg.ELITE_SIZE, cfg.STAGE1_EPS, cfg.EVAL_EPISODES, cfg.STAGE2_KEEP = 64, 2, 4, 64
    cfg.MAX_STEPS, cfg.EVAL_BATCH = 3000, 256
    n = 8 if habit else 0
    cfg.HABIT_EDGE_ELITE = n
    cfg.HABIT_CONN_ELITE = n
    cfg.HABIT_STRAIGHT_ELITE = n
    cfg.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    return cfg


def probe_quota_fires():
    """快速探针：同种子 1 代小种群，习惯配额开/关的子代必须不同（否则配额未生效，
    直接中止避免无效对照）。"""
    sums = []
    for habit in (False, True):
        cfg = make_cfg(habit)
        cfg.POP_SIZE, cfg.ELITE_SIZE = 64, 16
        cfg.STAGE1_EPS, cfg.EVAL_EPISODES, cfg.STAGE2_KEEP = 1, 2, 16
        cfg.EVAL_BATCH = 64
        torch.manual_seed(99)
        dev = t12._resolve_device(cfg)
        pop = t12.GeneStack(cfg, device=dev)
        pop.random_init()
        metrics, order = t12.evaluate_population_gpu(pop, cfg, gen=0)
        pop2 = t12.evolve_topology_gpu(pop, metrics, cfg, gen=0, order=order)
        sums.append(float(pop2.W_in.float().abs().sum()))
    fired = abs(sums[0] - sums[1]) > 1e-6
    print(f"[探针] 配额开/关子代 W_in 绝对值和: {sums[0]:.4f} vs {sums[1]:.4f} → "
          f"{'配额已生效' if fired else '配额未生效（中止）'}")
    return fired


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
    # 终代精英组三因素（适应度 top-ELITE，同种子同 CRN 可比）
    elite = mn[order[:cfg.ELITE_SIZE]]
    sel = elite[:, 0] > 0
    return (np.array(curve),
            dict(epref=float(elite[sel, 15].mean()),
                 conn=float(elite[sel, 16].mean()),
                 straight=float(elite[sel, 17].mean()),
                 self_die=float(mn[:, 6].mean())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gens', type=int, default=12)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    args = ap.parse_args()

    if not probe_quota_fires():
        sys.exit('[中止] 习惯配额未生效，修复后再跑（避免无效对照）')

    results = {}
    for arm, habit in (('OFF', False), ('ALL', True)):
        cfg = make_cfg(habit)
        for seed in args.seeds:
            print(f"\n=== 臂 {arm} seed {seed} ===", flush=True)
            curve, elite_m = run_arm(cfg, seed, args.gens)
            results[(arm, seed)] = (curve, elite_m)

    print("\n================ E2 预注册判定 ================")
    print("（conn 天花板处理：OFF 精英 conn≥0.995 时指标饱和，该因素本阶段")
    print("  不可判（任何选择都无法 >1.0），标记 CEILING 不计入 P2 失败）")
    p1_ok, p2_ok, p2_n = True, True, 0
    p3_ok = True
    for seed in args.seeds:
        c_off, m_off = results[('OFF', seed)]
        c_all, m_all = results[('ALL', seed)]
        d_avg = abs(c_all[-1, 1] - c_off[-1, 1])
        d_best = abs(c_all[-1, 0] - c_off[-1, 0])
        ok1 = d_avg <= 0.5 and d_best <= max(0.05 * max(c_off[-1, 0], 1), 1.0)
        p1_ok &= ok1
        print(f"[P1 不干扰] seed{seed}: Δavg={d_avg:.3f} (≤0.5) "
              f"Δbest={d_best:.2f} → {'PASS' if ok1 else 'FAIL'}")
        for k in ('epref', 'conn', 'straight'):
            d = m_all[k] - m_off[k]
            if m_off[k] >= 0.995:
                print(f"[P2 有效  ] seed{seed} {k}: OFF {m_off[k]:.3f} → "
                      f"ALL {m_all[k]:.3f} CEILING（饱和，不可判）")
                continue
            p2_n += 1
            ok = d > 0.05
            p2_ok &= ok
            print(f"[P2 有效  ] seed{seed} {k}: OFF {m_off[k]:.3f} → "
                  f"ALL {m_all[k]:.3f} (Δ{d:+.3f} >0.05) {'PASS' if ok else 'FAIL'}")
        ok3 = m_all['self_die'] <= m_off['self_die'] + 0.1
        p3_ok &= ok3
        print(f"[P3 健康  ] seed{seed}: 撞己占比 OFF {m_off['self_die']:.2f} vs "
              f"ALL {m_all['self_die']:.2f} → {'PASS' if ok3 else 'FAIL'}")
    verdict = ('通过' if (p1_ok and p2_ok and p3_ok) else
               '未通过' if p2_n > 0 else '通过（可判因素均 PASS，conn 饱和）')
    print(f"\nE2 总判定: {verdict}（P1={p1_ok} P2={p2_ok} 可判因素数={p2_n} P3={p3_ok}）")


if __name__ == '__main__':
    main()
