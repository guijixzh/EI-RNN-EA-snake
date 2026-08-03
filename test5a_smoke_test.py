# -*- coding: utf-8 -*-
"""test5a.py 冒烟测试：验证固定周期交替冻结（cycle）核心逻辑。

1. _freeze_active_groups 四种 scheme 的激活组正确性
2. G1 独占不变式：cycle 任意代 G1 不与 G2/G3 共存
3. 冻结行为：G1 冻结代，子代 G1 参数与父代 p1 完全一致
4. 短代进化运行（极小配置走 6 代，覆盖 5 个 cycle 阶段 + 一个完整周期）
"""
import copy
import random

import torch

import test5a
from test5a import Config, EIBrainRegion, SnakeEnv, evolve_topology, _freeze_active_groups


def make_mini_cfg():
    """构造极小配置（Config 实例属性覆盖类属性，不污染 test5a.Config）。"""
    cfg = Config()
    cfg.POP_SIZE = 16
    cfg.ELITE_SIZE = 4
    cfg.GENERATIONS = 6
    cfg.NUM_COLUMNS = 8
    cfg.EVAL_EPISODES = 1
    cfg.SCREEN_EPISODES = 1
    cfg.SCREEN_ENABLE = False
    cfg.PARALLEL_EVAL = False  # 冒烟测试走串行评估
    cfg.AUTO_RESUME = False
    cfg.SEED_FROM_BEST = False
    cfg.CHECKPOINT_PATH = 'test5a_smoke_checkpoint.pth'
    cfg.BEST_MODEL_PATH = 'test5a_smoke_best_model.pth'
    cfg.FREEZE_SCHEME = 'cycle'
    cfg.CYCLE_PATTERN = [('G2',), ('G1',), ('G2', 'G3'), ('G1',), ('G3',)]
    return cfg


def test_active_groups():
    cfg = make_mini_cfg()
    scheme_all = copy.copy(cfg)
    scheme_all.FREEZE_SCHEME = 'all'
    scheme_hard = copy.copy(cfg)
    scheme_hard.FREEZE_SCHEME = 'hard'
    scheme_soft = copy.copy(cfg)
    scheme_soft.FREEZE_SCHEME = 'soft'

    # --- cycle（默认）：固定周期 G2 / G1 / G2G3 / G1 / G3 ---
    expected_cycle = [
        frozenset({'G2'}),
        frozenset({'G1'}),
        frozenset({'G2', 'G3'}),
        frozenset({'G1'}),
        frozenset({'G3'}),
    ]
    pattern = cfg.CYCLE_PATTERN
    for g in range(10):
        exp = frozenset(pattern[g % len(pattern)])
        assert _freeze_active_groups(g, cfg) == exp, f"gen{g}(cycle) 应为 {exp}"

    # --- G1 独占不变式：任意代若含 G1，则不含 G2/G3 ---
    for g in range(20):
        act = _freeze_active_groups(g, cfg)
        if 'G1' in act:
            assert 'G2' not in act and 'G3' not in act, f"gen{g}(cycle) 违反 G1 独占：{act}"

    # --- all：等价原 test5_fast ---
    for g in range(6):
        assert _freeze_active_groups(g, scheme_all) == frozenset({'G1', 'G2', 'G3'}), f"gen{g}(all) 应全激活"

    # --- hard：严格轮换 ---
    assert _freeze_active_groups(0, scheme_hard) == frozenset({'G1'}), "gen0(hard) 应为 G1"
    assert _freeze_active_groups(1, scheme_hard) == frozenset({'G2'}), "gen1(hard) 应为 G2"
    assert _freeze_active_groups(2, scheme_hard) == frozenset({'G3'}), "gen2(hard) 应为 G3"
    assert _freeze_active_groups(3, scheme_hard) == frozenset({'G1'}), "gen3(hard) 应为 G1"

    # --- soft：取模独立激活（备选） ---
    assert _freeze_active_groups(0, scheme_soft) == frozenset({'G1', 'G2', 'G3'}), "gen0(soft) 应全激活"
    assert _freeze_active_groups(1, scheme_soft) == frozenset({'G2'}), "gen1(soft) 应为 G2"

    print("[PASS] _freeze_active_groups 四种 scheme 行为正确 + cycle G1 独占不变式成立")


def test_freeze_behavior():
    """cycle 方案 gen=1（仅 G1 激活）时，G2/G3 冻结：
    child 的 G2/G3 参数必须与 p1 完全一致。"""
    cfg = make_mini_cfg()
    torch.manual_seed(123)
    pop = [EIBrainRegion(cfg) for _ in range(8)]
    metrics = [(random.random() * 3, random.random() * 100) for _ in range(8)]

    # gen=1：仅 G1 激活（cycle 第二个槽位）
    new_pop = evolve_topology(pop, metrics, cfg, gen=1)

    # 构造 elite 索引（与 evolve_topology 相同的排序规则）
    sorted_idx = sorted(range(len(metrics)),
                        key=lambda i: (metrics[i][0], -metrics[i][1]),
                        reverse=True)
    elite_idx = sorted_idx[:cfg.ELITE_SIZE]
    elites = [pop[i] for i in elite_idx]

    # 逐一核对后代（索引 >= ELITE_SIZE）G2/G3 冻结性：必须与某精英完全一致
    for offspring in new_pop[cfg.ELITE_SIZE:]:
        g2_matches = g3_matches = False
        for e in elites:
            if torch.equal(offspring.tau_e_init.data, e.tau_e_init.data) and \
               torch.equal(offspring.w_ei.data, e.w_ei.data) and \
               torch.equal(offspring.w_ie.data, e.w_ie.data):
                g2_matches = True
            if torch.equal(offspring.W_hormone1.data, e.W_hormone1.data) and \
               torch.equal(offspring.b_hormone1.data, e.b_hormone1.data) and \
               torch.equal(offspring.W_excit.data, e.W_excit.data) and \
               torch.equal(offspring.b_excit.data, e.b_excit.data) and \
               torch.equal(offspring.W_inhib.data, e.W_inhib.data) and \
               torch.equal(offspring.b_inhib.data, e.b_inhib.data):
                g3_matches = True
            if g2_matches and g3_matches:
                break
        assert g2_matches, "G2 冻结失败：子代动力学参数被交叉/变异修改"
        assert g3_matches, "G3 冻结失败：子代激素参数被交叉/变异修改"

    # G1 应确实发生了变异/交叉（至少多数子代与各精英都有差异）
    changed = 0
    for offspring in new_pop[cfg.ELITE_SIZE:]:
        if not any(torch.equal(offspring.M_in, e.M_in) and
                   torch.equal(offspring.M_rec, e.M_rec) and
                   torch.equal(offspring.M_out, e.M_out) and
                   torch.equal(offspring.W_in.data, e.W_in.data) and
                   torch.equal(offspring.W_rec.data, e.W_rec.data) and
                   torch.equal(offspring.W_out.data, e.W_out.data) and
                   torch.equal(offspring.b_out.data, e.b_out.data)
                   for e in elites):
            changed += 1
    assert changed > 0, "G1 激活失败：所有子代结构参数与精英完全相同"
    print(f"[PASS] 冻结行为正确（gen=1: G1 激活、G2/G3 冻结；{changed} 个子代 G1 已变异）")


def test_short_train_run():
    """极小配置跑 6 代（覆盖 5 个 cycle 阶段 + 完整周期），验证无异常。"""
    cfg = make_mini_cfg()
    env = SnakeEnv(grid_size=cfg.GRID_SIZE)
    torch.manual_seed(42)
    random.seed(42)

    pop = [EIBrainRegion(cfg) for _ in range(cfg.POP_SIZE)]
    for ind in pop:
        ind.save_genetic_baseline()

    seen_phases = set()
    for gen in range(cfg.GENERATIONS):
        metrics = [test5a.evaluate_individual(ind, env) for ind in pop]
        best_food = max(m[0] for m in metrics)
        if gen < cfg.GENERATIONS - 1:
            pop = evolve_topology(pop, metrics, cfg, gen=gen)
        act = tuple(sorted(_freeze_active_groups(gen, cfg)))
        seen_phases.add(act)
        print(f"  gen={gen} 完成, best_food={best_food:.1f}, active={act}")

    assert ('G2',) in seen_phases and ('G1',) in seen_phases and \
           ('G2', 'G3') in seen_phases and ('G3',) in seen_phases, \
        f"6 代未覆盖全部 cycle 阶段: {seen_phases}"
    print(f"[PASS] 短代进化运行正常（覆盖 {len(seen_phases)} 个 cycle 阶段: {sorted(seen_phases)}）")


if __name__ == "__main__":
    test_active_groups()
    test_freeze_behavior()
    test_short_train_run()
    print("\n=== 全部冒烟测试通过 ===")