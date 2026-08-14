# -*- coding: utf-8 -*-
"""test5_fast.py 加速效果基准。

对比三种评估模式（同一固定种群、固定评估局数预算）：
  1. 串行全量  PARALLEL_EVAL=False, SCREEN_ENABLE=False
  2. 并行全量  PARALLEL_EVAL=True , SCREEN_ENABLE=False
  3. 并行+筛选 PARALLEL_EVAL=True , SCREEN_ENABLE=True

仅做评估阶段计时（含多进程建池/序列化分发），不包含进化演化。
"""
import os
import sys
import time
import random
import torch

EXPERIMENTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiments')
if EXPERIMENTS not in sys.path:
    sys.path.insert(0, EXPERIMENTS)

import test5_fast as M


def build_pop(cfg, seed=2026):
    random.seed(seed)
    torch.manual_seed(seed)
    pop = [M.EIBrainRegion(cfg) for _ in range(cfg.POP_SIZE)]
    for ind in pop:
        ind.save_genetic_baseline()
    return pop


def bench_once(cfg, pop, env, label, pool=None):
    t0 = time.perf_counter()
    metrics = M.evaluate_population(pop, cfg, env, pool)
    dt = time.perf_counter() - t0
    avg_food = float(sum(m[0] for m in metrics) / len(metrics))
    print(f"  {label:<22} {dt:8.2f}s   avg_food={avg_food:.3f}")
    return dt


def main():
    torch.set_num_threads(1)  # 稳定单线程基线，减少环境噪声

    cfg = M.Config()
    cfg.POP_SIZE = 128
    cfg.EVAL_EPISODES = 5
    cfg.SCREEN_EPISODES = 1
    cfg.SCREEN_MULTIPLIER = 3
    cfg.SCREEN_ENABLE = True
    cfg.MAX_STEPS = 200          # 缩短步数，加快基准
    cfg.ELITE_SIZE = 16

    # 与 make_eval_pool 相同的默认 worker 数计算
    n_workers = min(os.cpu_count() or 1, 16, cfg.POP_SIZE)
    print(f"=== test5_fast 基准：POP={cfg.POP_SIZE}, "
          f"EVAL_EPISODES={cfg.EVAL_EPISODES}, 默认进程数={n_workers} ===")
    print(f"    每代局数预算：串行全量={cfg.POP_SIZE * cfg.EVAL_EPISODES}, "
          f"并行+筛选≈{cfg.POP_SIZE * cfg.SCREEN_EPISODES + cfg.ELITE_SIZE * cfg.SCREEN_MULTIPLIER * (cfg.EVAL_EPISODES - cfg.SCREEN_EPISODES)}")

    env = M.SnakeEnv(grid_size=cfg.GRID_SIZE)

    # 同一固定种群，三种模式各测一轮
    pop = build_pop(cfg)
    print("\n--- 串行全量 ---")
    t_serial = bench_once(cfg, pop, env, "serial-full", pool=None)

    cfg.PARALLEL_EVAL = True
    pool = M.make_eval_pool(cfg)
    print("\n--- 并行全量 (A) ---")
    t_par_full = bench_once(cfg, pop, env, f"parallel-full({pool._processes}p)", pool=pool)

    cfg.SCREEN_ENABLE = True
    print("\n--- 并行+筛选 (A+C) ---")
    t_par_screen = bench_once(cfg, pop, env, f"parallel+screen({pool._processes}p)", pool=pool)

    print("\n--- 加速比 ---")
    print(f"  A       vs 串行全量 : {t_serial / t_par_full:6.2f}x")
    print(f"  A+C     vs 串行全量 : {t_serial / t_par_screen:6.2f}x")
    print(f"  (若为单核机器，并行加速比会接近 1.x)")

    if pool is not None:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()