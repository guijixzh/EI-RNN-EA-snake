# -*- coding: utf-8 -*-
"""NUM_COLUMNS (N) 扫描基准：实测 N=64/128/256 对单步耗时与内存的影响。

指标：
  1) 单个体参数+缓存总内存（MB），外推 POP_SIZE=2048 总内存
  2) 纯 forward 平均单步耗时（µs/step）
  3) 综合评估单步耗时（含 SnakeEnv step + PC 更新，µs/step）
  4) 外推 2048 pop × 100 gen × 5 ep × 300 steps 的评估总时长
"""
import importlib.util
import os
import time
import random
import torch

EXPERIMENTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiments')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def brain_memory_bytes(brain):
    total = 0
    for p in brain.parameters():
        total += p.numel() * p.element_size()
    for b in brain.buffers():
        total += b.numel() * b.element_size()
    return total


def forward_bench(brain, cfg, steps=2000, seed=0):
    torch.manual_seed(seed)
    obs = torch.zeros(cfg.OBS_DIM)
    E = torch.zeros(brain.N)
    I = torch.zeros(brain.N)
    with torch.no_grad():
        for _ in range(50):
            logits, E, I, err, tin = brain(obs, E, I)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            logits, E, I, err, tin = brain(obs, E, I)
    return (time.perf_counter() - t0) / steps


def eval_workload(mod, N, episodes=2, max_steps=200, repeats=3, seed=0):
    """调用优化版 evaluate_individual，统计综合单步耗时（含环境+PC更新）"""
    cfg = mod.Config()
    cfg.NUM_COLUMNS = N
    cfg.EVAL_EPISODES = episodes
    cfg.MAX_STEPS = max_steps

    torch.manual_seed(seed)
    random.seed(seed)
    brain = mod.EIBrainRegion(cfg)
    brain.save_genetic_baseline()
    env = mod.SnakeEnv(grid_size=cfg.GRID_SIZE)

    t0 = time.perf_counter()
    for _ in range(repeats):
        mod.evaluate_individual(brain, env)
    total = time.perf_counter() - t0

    # 估计每步综合成本（用每次评估的平均步数）
    avg_steps = 0.0
    for _ in range(5):
        brain.restore_genetic_baseline()
        brain.reset_runtime()
        obs = env.reset()
        E = torch.zeros(brain.N)
        I = torch.zeros(brain.N)
        done = False
        steps = 0
        while not done and steps < max_steps:
            obs_t = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                logits, E, I, err, tin = brain(obs_t, E, I)
                action = torch.argmax(logits).item()
            next_obs, _, done = env.step(action)
            obs = next_obs
            steps += 1
        avg_steps += steps
    avg_steps /= 5

    total_steps_run = repeats * episodes * avg_steps
    per_step = total / total_steps_run
    return per_step, avg_steps


def main():
    torch.set_num_threads(4)  # 固定线程数，减少噪音
    mod = load_module("test4b_opt", os.path.join(EXPERIMENTS, "test4b.py"))
    POP = 2048
    GEN = 100
    EP = 5
    STEPS_PER_EP = 300

    print("=" * 78)
    print("NUM_COLUMNS (N) 扫描基准 — 优化版 test4b.py")
    print("=" * 78)

    results = []
    for N in [64, 128, 256]:
        cfg = mod.Config()
        cfg.NUM_COLUMNS = N

        torch.manual_seed(0)
        brain = mod.EIBrainRegion(cfg)
        brain.save_genetic_baseline()

        mem = brain_memory_bytes(brain)
        t_fwd = forward_bench(brain, cfg)
        t_eval_step, avg_steps = eval_workload(mod, N)

        pop_mem = mem * POP / (1024 ** 3)
        ext_hours = t_eval_step * POP * GEN * EP * STEPS_PER_EP / 3600

        results.append((N, mem, t_fwd, t_eval_step, avg_steps, pop_mem, ext_hours))
        print(f"\nN = {N}:")
        print(f"  单个体参数+缓存内存 : {mem / 1024:.1f} KB  "
              f"(POP={POP} 总内存约 {pop_mem:.2f} GB)")
        print(f"  纯 forward           : {t_fwd * 1e6:.1f} µs/step")
        print(f"  综合评估(含环境/PC)  : {t_eval_step * 1e6:.1f} µs/step  "
              f"(实测每局平均 {avg_steps:.0f} 步)")
        print(f"  外推 2048×100×5×300  : 约 {ext_hours:.1f} 小时 (评估阶段)")

    print("\n" + "=" * 78)
    print("增长倍数 (相对 N=64):")
    base = results[0]
    print(f"  {'N':<6} {'内存/个':>10} {'纯forward':>12} {'综合评估步':>12} {'外推时长':>10}")
    for (N, mem, t_fwd, t_eval, _, _, ext) in results:
        print(f"  {N:<6} {mem / base[1]:>9.2f}x {t_fwd / base[2]:>11.2f}x "
              f"{t_eval / base[3]:>11.2f}x {ext / base[6]:>9.2f}x (={ext:.0f}h)")
    print(f"\n  [参考] 若为纯 O(N²)，N=128 应为 4.00x, N=256 应为 16.00x")
    print("=" * 78)


if __name__ == "__main__":
    main()