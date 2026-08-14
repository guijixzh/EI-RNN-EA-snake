# -*- coding: utf-8 -*-
"""A/B 基准测试：test4b_orig.py (原版) vs test4b.py (已优化)

指标：
  1) forward 平均步耗时（2000 步，纯大脑计算，无环境）
  2) 相同随机种子下 3 代小规模进化的耗时（初始化 / 评估 / 进化）
  3) 正确性：两版在相同 seed 下应产生完全一致的结果（best_food/best_steps）
"""
import importlib.util
import os
import time
import random
import torch
import numpy as np

EXPERIMENTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiments')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_forward_bench(mod, steps=2000, seed=0):
    """只测大脑 forward 的平均耗时（排除 SnakeEnv 开销）"""
    torch.manual_seed(seed)
    cfg = mod.Config()
    brain = mod.EIBrainRegion(cfg)
    brain.save_genetic_baseline()
    obs = torch.zeros(cfg.OBS_DIM)
    E = torch.zeros(cfg.NUM_COLUMNS)
    I = torch.zeros(cfg.NUM_COLUMNS)

    with torch.no_grad():
        for _ in range(50):  # warmup
            logits, E, I, err, tin = brain(obs, E, I)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            logits, E, I, err, tin = brain(obs, E, I)
    dt = time.perf_counter() - t0
    return dt / steps


def consistency_check(mod):
    """跑 3 步 forward，返回 E 与 logits 快照（用于两版逐值比较）"""
    torch.manual_seed(7)
    cfg = mod.Config()
    brain = mod.EIBrainRegion(cfg)
    brain.save_genetic_baseline()
    brain.reset_runtime()
    E = torch.zeros(brain.N)
    I = torch.zeros(brain.N)
    obs = torch.rand(brain.obs_dim)
    snapshots = []
    with torch.no_grad():
        for _ in range(3):
            logits, E, I, err, tin = brain(obs, E, I)
            snapshots.append((E.clone(), logits.clone()))
    return snapshots


def run_evolution(mod, pop_size=64, elite=8, gens=3, episodes=2, max_steps=200, seed=1234):
    """相同 seed 下跑小规模进化，统计各阶段耗时与结果"""
    random.seed(seed)
    torch.manual_seed(seed)

    cfg = mod.Config()
    cfg.POP_SIZE = pop_size
    cfg.ELITE_SIZE = elite
    cfg.GENERATIONS = gens
    cfg.EVAL_EPISODES = episodes
    cfg.MAX_STEPS = max_steps

    t0 = time.perf_counter()
    population = [mod.EIBrainRegion(cfg) for _ in range(pop_size)]
    for ind in population:
        ind.save_genetic_baseline()
    init_t = time.perf_counter() - t0

    env = mod.SnakeEnv(grid_size=cfg.GRID_SIZE)
    cum_eval = 0.0
    cum_evolve = 0.0
    best_history = []

    for gen in range(gens):
        t = time.perf_counter()
        metrics = [mod.evaluate_individual(ind, env) for ind in population]
        cum_eval += time.perf_counter() - t

        best_idx = max(range(len(metrics)),
                       key=lambda i: (metrics[i][0], -metrics[i][1]))
        best_history.append((metrics[best_idx][0], metrics[best_idx][1]))

        if gen < gens - 1:
            t = time.perf_counter()
            population = mod.evolve_topology(population, metrics, cfg)
            cum_evolve += time.perf_counter() - t

    return {
        'init': init_t,
        'eval': cum_eval,
        'evolve': cum_evolve,
        'total_work': init_t + cum_eval + cum_evolve,
        'best_history': best_history,
        'best_food': best_history[-1][0],
        'best_steps': best_history[-1][1],
    }


def fmt(sec):
    return f"{sec * 1000:.1f} ms" if sec < 1 else f"{sec:.2f} s"


def main():
    orig = load_module("test4b_orig", os.path.join(EXPERIMENTS, "test4b_orig.py"))
    opt = load_module("test4b_opt", os.path.join(EXPERIMENTS, "test4b.py"))

    # ---------- 1) forward 步耗时 ----------
    t_orig = run_forward_bench(orig)
    t_opt = run_forward_bench(opt)
    print("=" * 74)
    print("1) Forward 平均单步耗时 (2000 步, 排除环境开销)")
    print(f"   原版   : {t_orig * 1e6:8.1f} µs/step")
    print(f"   优化版 : {t_opt * 1e6:8.1f} µs/step   ({t_orig / t_opt:.2f}x 提速)")

    # 外推到 POP_SIZE=2048、100 代、5 局/个体、平均 300 步/局的评估总步数
    total_steps = 2048 * 100 * 5 * 300
    print(f"   [外推] 2048 pop × 100 gen × 5 ep × 300 steps 的纯评估耗时:")
    print(f"   原版   : {total_steps * t_orig / 3600:.1f} 小时")
    print(f"   优化版 : {total_steps * t_opt / 3600:.1f} 小时")

    # ---------- 2) 正确性：3 步 forward 快照逐值对比 ----------
    snap_orig = consistency_check(orig)
    snap_opt = consistency_check(opt)
    max_diff = 0.0
    for (E_o, L_o), (E_n, L_n) in zip(snap_orig, snap_opt):
        max_diff = max(max_diff,
                       float((E_o - E_n).abs().max()),
                       float((L_o - L_n).abs().max()))
    print("=" * 74)
    print(f"2) 正确性 (3 步 forward 快照最大偏差): {max_diff:.3e} "
          f"({'一致' if max_diff < 1e-6 else '存在差异!'})")

    # ---------- 3) 相同 seed 下 3 代小规模进化 ----------
    r_orig = run_evolution(orig)
    r_opt = run_evolution(opt)
    same = (r_orig['best_food'] == r_opt['best_food'] and
            r_orig['best_steps'] == r_opt['best_steps'])
    print("=" * 74)
    print("3) 相同 seed 下 3 代小规模进化 (POP=64, GEN=3, EP=2, MAX_STEPS=200)")
    print(f"   {'指标':<10} {'原版':>16} {'优化版':>16} {'提速':>8}")
    print(f"   {'初始化':<10} {fmt(r_orig['init']):>16} {fmt(r_opt['init']):>16} "
          f"{r_orig['init'] / r_opt['init']:>6.2f}x")
    print(f"   {'评估总计':<10} {fmt(r_orig['eval']):>16} {fmt(r_opt['eval']):>16} "
          f"{r_orig['eval'] / r_opt['eval']:>6.2f}x")
    print(f"   {'进化总计':<10} {fmt(r_orig['evolve']):>16} {fmt(r_opt['evolve']):>16} "
          f"{r_orig['evolve'] / r_opt['evolve']:>6.2f}x")
    print(f"   {'总工作量':<10} {fmt(r_orig['total_work']):>16} {fmt(r_opt['total_work']):>16} "
          f"{r_orig['total_work'] / r_opt['total_work']:>6.2f}x")
    print(f"\n   末代 best_food / best_steps:")
    print(f"   原版   : {r_orig['best_food']:.1f} / {r_orig['best_steps']:.1f}")
    print(f"   优化版 : {r_opt['best_food']:.1f} / {r_opt['best_steps']:.1f}")
    print(f"   结果一致性: {'完全一致' if same else '不一致!!'}")

    print("=" * 74)


if __name__ == "__main__":
    main()