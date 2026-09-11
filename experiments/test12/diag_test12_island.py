# 诊断 test12 "best food 卡 1.0"：孤岛惩罚是否与食物数正相关、是否反向压选择。
# 用法: python experiments/diag_test12_island.py [--gens 6]
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch

from test12 import (Config, GeneStack, evaluate_population_gpu,
                    evolve_topology_gpu, _fitness_econ, _resolve_device)


def rankdata(x):
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x))
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gens', type=int, default=6)
    args = ap.parse_args()

    cfg = Config()
    cfg.POP_SIZE, cfg.NUM_COLUMNS = 256, 96
    cfg.ELITE_SIZE, cfg.STAGE1_EPS, cfg.EVAL_EPISODES, cfg.STAGE2_KEEP = 64, 2, 4, 64
    cfg.MAX_STEPS, cfg.EVAL_BATCH = 3000, 256
    dev = _resolve_device(cfg)
    print(f"device={dev} pop={cfg.POP_SIZE} K1={cfg.STAGE1_EPS} K2={cfg.EVAL_EPISODES} "
          f"island: thr={cfg.ISLAND_THRESHOLD} pen={cfg.ISLAND_PENALTY}")

    torch.manual_seed(12)
    pop = GeneStack(cfg, device=dev)
    pop.random_init()

    for gen in range(args.gens):
        metrics, order = evaluate_population_gpu(pop, cfg, gen=gen)
        mn = metrics.numpy()
        food, flag, minr = mn[:, 0], mn[:, 14], mn[:, 13]
        fit = np.array([_fitness_econ(mn[i], cfg) for i in range(len(mn))])

        print(f"\n== gen {gen} ==")
        print(f"  island 触发率(个体级,任一局): {flag.mean():.2%} | "
              f"food==1: {(food == 1).mean():.2%} food>=2: {(food >= 2).mean():.2%} "
              f"food==0: {(food == 0).mean():.2%}")
        for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 1e9)):
            sel = (food >= lo) & (food < hi)
            if sel.sum() > 0:
                print(f"  food∈[{lo},{hi if hi < 1e8 else '∞'}) n={int(sel.sum()):4d} "
                      f"flag率={flag[sel].mean():.2%} min_reach中位={np.median(minr[sel]):.2f}")
        print(f"  spearman(fitness, food)={spearman(fit, food):+.3f} | "
              f"spearman(flag, food)={spearman(flag, food):+.3f}")
        top_fit = np.argsort(-fit)[:32]
        top_food = np.argsort(-food)[:32]
        print(f"  适应度Top32 平均food={food[top_fit].mean():.2f} | "
              f"纯food Top32 的平均fitness={fit[top_food].mean():.2f} "
              f"(对照: 全体平均fitness={fit.mean():.2f})")
        no_pen = np.array([_fitness_econ(np.concatenate([mn[i][:14], [0.0]]), cfg)
                           for i in range(len(mn))])
        print(f"  若无惩罚 spearman={spearman(no_pen, food):+.3f} | "
              f"惩罚后 Top32 food 均值 {food[top_fit].mean():.2f} vs "
              f"无惩罚 Top32 food 均值 {food[np.argsort(-no_pen)[:32]].mean():.2f}")
        pop = evolve_topology_gpu(pop, metrics, cfg, gen=gen, order=order)


if __name__ == '__main__':
    main()
