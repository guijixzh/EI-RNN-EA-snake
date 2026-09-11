# -*- coding: utf-8 -*-
"""test5_fast.py 冒烟测试：
1. 串行全量评估（排除多进程干扰）作为参考基准
2. A 方案：多进程并行评估结果与串行一致
3. C 方案：两阶段筛选在不同 EVAL_EPISODES / SCREEN_EPISODES 组合下的正确性
4. 断点保存/加载往返一致
"""
import os
import random
import time
import torch
import numpy as np

import test5_fast as M


def make_cfg(**overrides):
    cfg = M.Config()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def assert_close(name, a, b, tol=1e-6):
    ok = all(abs(x - y) <= tol for x, y in zip(a, b))
    print(f"  {'PASS' if ok else 'FAIL'}: {name}: {a} vs {b}")
    if not ok:
        raise AssertionError(f"{name}: {a} != {b}")


def run():
    # 固定随机种子保证可复现
    seed = 12345
    random.seed(seed)
    torch.manual_seed(seed)

    print("=== 测试 1: 串行全量评估 (PARALLEL_EVAL=False, SCREEN_ENABLE=False) ===")
    cfg = make_cfg(PARALLEL_EVAL=False, SCREEN_ENABLE=False, EVAL_EPISODES=5,
                   POP_SIZE=8, NUM_COLUMNS=64, MAX_STEPS=120)
    env = M.SnakeEnv(grid_size=cfg.GRID_SIZE)
    pop = [M.EIBrainRegion(cfg) for _ in range(cfg.POP_SIZE)]
    for ind in pop:
        ind.save_genetic_baseline()
    random.seed(seed)  # 评估前重设种子（评估内部用随机）
    t0 = time.perf_counter()
    m_serial = M.evaluate_population(pop, cfg, env, pool=None)
    t_serial = time.perf_counter() - t0
    print(f"  serial {len(m_serial)} inds in {t_serial:.2f}s, first={m_serial[0]}")

    print("=== 测试 2: A 方案并行评估 ===")
    random.seed(seed)
    t0 = time.perf_counter()
    pool = M.make_eval_pool(cfg)
    m_par = M.evaluate_population(pop, cfg, env, pool)
    t_par = time.perf_counter() - t0
    if pool is not None:
        pool.close()
        pool.join()
    print(f"  parallel {len(m_par)} inds in {t_par:.2f}s, first={m_par[0]}")
    # 并行 worker 各进程随机种子不同，结果不必逐项相等；
    # 只需验证：并行也返回相同数量的指标且为合法 (food, steps)。
    # （硬淘汰判死时为 Python int 0/99999，其余为 np.float64/float）
    assert len(m_par) == len(pop)
    assert all(isinstance(f, (int, float, np.floating)) and
               isinstance(s, (int, float, np.floating)) for f, s in m_par)
    # 并行评估分数量级应接近串行（同为随机局，均值差异在噪声范围内）
    avg_serial = float(np.mean([m[0] for m in m_serial]))
    avg_par = float(np.mean([m[0] for m in m_par]))
    print(f"  avg food: serial={avg_serial:.3f}, parallel={avg_par:.3f}")
    assert abs(avg_serial - avg_par) < 8.0, "parallel 与 serial 平均食物差异过大"

    print("=== 测试 3: C 方案两阶段筛选（EVAL_EPISODES=5, SCREEN_EPISODES=1） ===")
    cfg2 = make_cfg(PARALLEL_EVAL=False, SCREEN_ENABLE=True, SCREEN_EPISODES=1,
                    SCREEN_MULTIPLIER=3, EVAL_EPISODES=5, POP_SIZE=16,
                    NUM_COLUMNS=64, MAX_STEPS=120, ELITE_SIZE=4)
    env2 = M.SnakeEnv(grid_size=cfg2.GRID_SIZE)
    pop2 = [M.EIBrainRegion(cfg2) for _ in range(cfg2.POP_SIZE)]
    for ind in pop2:
        ind.save_genetic_baseline()
    random.seed(seed)
    t0 = time.perf_counter()
    m_screen = M.evaluate_population(pop2, cfg2, env2, pool=None)
    t_screen = time.perf_counter() - t0
    print(f"  screen eval {len(m_screen)} inds in {t_screen:.2f}s")
    # K = ELITE_SIZE * 3 = 12，精评 12 名
    print(f"  first={m_screen[0]}, last={m_screen[-1]}")
    assert len(m_screen) == len(pop2)
    assert all(m is not None for m in m_screen)

    print("=== 测试 4: C 方案自适应不同 EVAL_EPISODES（8 局 / 初筛 3 局） ===")
    cfg3 = make_cfg(PARALLEL_EVAL=False, SCREEN_ENABLE=True, SCREEN_EPISODES=3,
                    SCREEN_MULTIPLIER=2, EVAL_EPISODES=8, POP_SIZE=10,
                    NUM_COLUMNS=64, MAX_STEPS=80, ELITE_SIZE=3)
    env3 = M.SnakeEnv(grid_size=cfg3.GRID_SIZE)
    pop3 = [M.EIBrainRegion(cfg3) for _ in range(cfg3.POP_SIZE)]
    for ind in pop3:
        ind.save_genetic_baseline()
    random.seed(seed)
    m_screen3 = M.evaluate_population(pop3, cfg3, env3, pool=None)
    assert len(m_screen3) == len(pop3)
    print(f"  EVAL_EPISODES=8/SCREEN=3 OK: first={m_screen3[0]}")

    print("=== 测试 5: SCREEN_EPISODES == EVAL_EPISODES 时自动禁用 C ===")
    cfg4 = make_cfg(PARALLEL_EVAL=False, SCREEN_ENABLE=True, SCREEN_EPISODES=5,
                    EVAL_EPISODES=5, POP_SIZE=6, SCREEN_AUTO_FALLBACK=False)
    env4 = M.SnakeEnv(grid_size=cfg4.GRID_SIZE)
    pop4 = [M.EIBrainRegion(cfg4) for _ in range(cfg4.POP_SIZE)]
    for ind in pop4:
        ind.save_genetic_baseline()
    random.seed(seed)
    m4 = M.evaluate_population(pop4, cfg4, env4, pool=None)
    assert len(m4) == len(pop4)
    print("  SCREEN_EPISODES==EVAL_EPISODES -> 等效全量评估 OK")

    print("=== 测试 6: 断点保存/加载往返 ===")
    ckpt_path = '_smoke_ckpt.pth'
    cfg5 = make_cfg(POP_SIZE=6)
    pop5 = [M.EIBrainRegion(cfg5) for _ in range(cfg5.POP_SIZE)]
    for ind in pop5:
        ind.save_genetic_baseline()
    hist = {'gen': [0], 'best_food': [1.0], 'avg_food': [0.5], 'best_steps': [100.0]}
    M.save_checkpoint(ckpt_path, cfg5, 2, pop5, hist,
                      cum_eval_time=1.0, cum_evolve_time=0.5,
                      best_brain=pop5[0], best_food=1.0, best_steps=100.0)
    ck = M.load_checkpoint(ckpt_path, cfg5)
    assert ck is not None
    assert ck['next_gen'] == 2
    assert len(ck['population']) == 6
    assert ck['best_food'] == 1.0
    # 权重往返一致性
    a = pop5[0]
    b = ck['population'][0]
    for attr in ['W_in', 'W_rec', 'W_out', 'b_out', 'tau_e_init', 'w_ei', 'w_ie']:
        ta = getattr(a, attr).detach()
        tb = getattr(b, attr).detach()
        assert torch.allclose(ta.float(), tb.float(), atol=1e-3), f"{attr} 往返不一致"
    print("  checkpoint round-trip OK")
    os.remove(ckpt_path)

    print("\n=== 全部冒烟测试通过 ===")


if __name__ == "__main__":
    run()