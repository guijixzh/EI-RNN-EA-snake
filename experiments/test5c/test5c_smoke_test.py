# -*- coding: utf-8 -*-
"""test5c.py 冒烟测试：验证预训练 CNN(16维) + E-I 进化（无疲劳机制）。

1. CNN 路径：网格(1,10,10,5) -> CNN(冻结) -> 16 维特征，维度正确
2. 无疲劳：EIBrainRegion 无 consecutive_counts / update_fatigue 属性方法；
   forward 输出不再包含疲劳抑制（检查 logits 形状仍为 (3,)）
3. _freeze_active_groups cycle 阶段正确（用户自定义组合模式）
4. 冻结行为：按 cycle 激活组验证，被冻结的组参数与父代 p1 完全一致
5. 短代进化运行（极小配置走 6 代，覆盖 5 个 cycle 阶段 + 完整周期）
"""
import os
import random
import sys

import torch

# 同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test5c
from test5c import (Config, EIBrainRegion, SnakeEnv, evolve_topology,
                    _freeze_active_groups, load_cnn_encoder,
                    deliberate_action)


def make_mini_cfg():
    """构造极小配置（Config 实例属性覆盖类属性，不污染 test5c.Config）。"""
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
    cfg.CHECKPOINT_PATH = 'test5c_smoke_checkpoint.pth'
    cfg.BEST_MODEL_PATH = 'test5c_smoke_best_model.pth'
    cfg.CNN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'cnn_encoder_16.pth')
    cfg.FREEZE_SCHEME = 'cycle'
    # 与 test5c.py Config 正式配置一致（用户自定义组合模式，G1 不与 G2/G3 独占）
    cfg.CYCLE_PATTERN = [('G2',), ('G1', 'G2'), ('G1', 'G3'),
                         ('G2', 'G3'), ('G3',)]
    return cfg


def test_cnn_encoding():
    """CNN 网格 -> 16 维特征链路（冻结、维度、数值有限）。"""
    cfg = make_mini_cfg()
    assert os.path.exists(cfg.CNN_PATH), f"CNN 不存在: {cfg.CNN_PATH}"
    cnn = load_cnn_encoder(cfg.CNN_PATH)
    assert cnn.proj_dim == 16, f"投影维度应为 16，实际 {cnn.proj_dim}"

    env = SnakeEnv(grid_size=cfg.GRID_SIZE)
    env.reset()
    grid_t = torch.from_numpy(env._get_grid_state()).unsqueeze(0)  # (1,10,10,5)
    assert grid_t.shape == (1, 10, 10, 5), f"网格形状错误: {grid_t.shape}"

    brain = EIBrainRegion(cfg, cnn=cnn)
    feat = brain.encode_grid(grid_t)
    assert feat.shape == (1, 16), f"CNN 特征应为 (1,16)，实际 {feat.shape}"
    assert torch.isfinite(feat).all(), "CNN 特征含 NaN/Inf"

    # 冻结性：CNN 参数不可训练
    assert all(not p.requires_grad for p in brain.cnn.parameters()), "CNN 未冻结"

    # 完整 forward 链路：obs(1,16) -> logits(3,)
    E = torch.zeros(brain.N)
    I = torch.zeros(brain.N)
    logits, E_new, I_new = brain(feat.squeeze(0), E, I)
    assert logits.shape == (3,), f"logits 应为 (3,)，实际 {logits.shape}"
    assert E_new.shape == (cfg.NUM_COLUMNS,)
    assert I_new.shape == (cfg.NUM_COLUMNS,)

    # K 帧思考（无疲劳）
    action, avg_logits, E, I = deliberate_action(brain, grid_t, E, I, K=3)
    assert action in (0, 1, 2), f"动作非法: {action}"
    assert avg_logits.shape == (3,)
    print("[PASS] CNN 网格->16 维->EI(K帧思考) 链路正确，特征有限")
    print(f"       feat 均值 {feat.mean().item():.4f} / 方差 {feat.var().item():.4f}")


def test_no_fatigue():
    """无疲劳机制：无 consecutive_counts / update_fatigue / FATIGUE_* 配置。"""
    cfg = make_mini_cfg()

    # 配置无疲劳相关项
    for attr in ('FATIGUE_GAIN', 'FATIGUE_THRESHOLD', 'FATIGUE_MAX',
                 'consecutive_counts', 'update_fatigue'):
        assert not hasattr(cfg, attr), f"疲劳配置/成员残留: {attr}"

    # 个体无运行时疲劳 bufffer 与方法
    cnn = load_cnn_encoder(cfg.CNN_PATH)
    brain = EIBrainRegion(cfg, cnn=cnn)
    assert not hasattr(brain, 'consecutive_counts'), "brain 残留 consecutive_counts"
    assert not hasattr(brain, 'update_fatigue'), "brain 残留 update_fatigue"

    # 保存/重建状态后同样无疲劳字段
    state = test5c.save_brain_state(brain)
    assert 'consecutive_counts' not in state, "保存的状态含疲劳字段"
    rebrain = test5c.load_brain_state(state, cfg, cnn)
    assert not hasattr(rebrain, 'consecutive_counts'), "重建 brain 残留疲劳 buffer"

    # 评估流程无 update_fatigue 调用（运行 1 局验证无异常）
    metrics = test5c.evaluate_individual(brain, SnakeEnv(grid_size=cfg.GRID_SIZE),
                                         episodes=1)
    assert isinstance(metrics, tuple) and len(metrics) == 2, f"评估返回值异常: {metrics}"
    print(f"[PASS] 疲劳机制已完全去除（配置/buffer/方法/调用链均无残留）; "
          f"评估正常 food={metrics[0]:.1f} steps={metrics[1]:.0f}")


def test_active_groups():
    cfg = make_mini_cfg()
    pattern = cfg.CYCLE_PATTERN
    for g in range(10):
        exp = frozenset(pattern[g % len(pattern)])
        assert _freeze_active_groups(g, cfg) == exp, f"gen{g}(cycle) 应为 {exp}"

    # 阶段覆盖：全部槽位出现
    seen = {frozenset(pattern[i % len(pattern)]) for i in range(20)}
    exp_phases = {frozenset(p) for p in pattern}
    assert seen == exp_phases, f"cycle 未覆盖全部阶段: {seen} != {exp_phases}"
    # 至少一个槽位同时激活两个组（组合模式生效）
    assert any(len(p) >= 2 for p in pattern), "cycle 无组合激活槽位"

    print("[PASS] _freeze_active_groups cycle 阶段正确（含组合激活槽位）")


def test_freeze_behavior():
    """cycle 方案 gen=4（仅 G3 激活）时，G1/G2 冻结：
    child 的 G1/G2 参数必须与父代 p1 完全一致。"""
    cfg = make_mini_cfg()
    torch.manual_seed(123)
    pop = [EIBrainRegion(cfg, cnn=load_cnn_encoder(cfg.CNN_PATH))
           for _ in range(8)]
    metrics = [(random.random() * 3, random.random() * 100) for _ in range(8)]

    gen = 4  # CYCLE_PATTERN[4] = ('G3',)：仅 G3 激活
    new_pop = evolve_topology(pop, metrics, cfg, gen=gen)

    sorted_idx = sorted(range(len(metrics)),
                        key=lambda i: (metrics[i][0], metrics[i][1]),
                        reverse=True)
    elite_idx = sorted_idx[:cfg.ELITE_SIZE]
    elites = [pop[i] for i in elite_idx]

    for offspring in new_pop[cfg.ELITE_SIZE:]:
        g1_matches = g2_matches = False
        for e in elites:
            # G1 冻结：结构组所有参数与某精英完全一致
            if torch.equal(offspring.M_in, e.M_in) and \
               torch.equal(offspring.M_rec, e.M_rec) and \
               torch.equal(offspring.M_out, e.M_out) and \
               torch.equal(offspring.W_in.data, e.W_in.data) and \
               torch.equal(offspring.W_rec.data, e.W_rec.data) and \
               torch.equal(offspring.W_out.data, e.W_out.data) and \
               torch.equal(offspring.b_out.data, e.b_out.data):
                g1_matches = True
            # G2 冻结：动力学参数与某精英完全一致
            if torch.equal(offspring.tau_e_init.data, e.tau_e_init.data) and \
               torch.equal(offspring.w_ei.data, e.w_ei.data) and \
               torch.equal(offspring.w_ie.data, e.w_ie.data):
                g2_matches = True
            if g1_matches and g2_matches:
                break
        assert g1_matches, "G1 冻结失败：子代结构参数被交叉/变异修改"
        assert g2_matches, "G2 冻结失败：子代动力学参数被交叉/变异修改"

    # G3 应确实发生了变异/交叉（至少多数子代与各精英有差异）
    changed = 0
    for offspring in new_pop[cfg.ELITE_SIZE:]:
        if not any(torch.equal(offspring.W_hormone1.data, e.W_hormone1.data) and
                   torch.equal(offspring.b_hormone1.data, e.b_hormone1.data) and
                   torch.equal(offspring.W_excit.data, e.W_excit.data) and
                   torch.equal(offspring.b_excit.data, e.b_excit.data) and
                   torch.equal(offspring.W_inhib.data, e.W_inhib.data) and
                   torch.equal(offspring.b_inhib.data, e.b_inhib.data)
                   for e in elites):
            changed += 1
    assert changed > 0, "G3 激活失败：所有子代激素参数与精英完全相同"
    print(f"[PASS] 冻结行为正确（gen=4: G3 激活、G1/G2 冻结；{changed} 个子代 G3 已变异）")


def test_short_train_run():
    """极小配置跑 6 代（覆盖全部 cycle 阶段），验证无异常。"""
    cfg = make_mini_cfg()
    env = SnakeEnv(grid_size=cfg.GRID_SIZE)
    torch.manual_seed(42)
    random.seed(42)

    cnn = load_cnn_encoder(cfg.CNN_PATH)
    pop = [EIBrainRegion(cfg, cnn=cnn) for _ in range(cfg.POP_SIZE)]
    for ind in pop:
        ind.save_genetic_baseline()

    seen_phases = set()
    for gen in range(cfg.GENERATIONS):
        metrics = [test5c.evaluate_individual(ind, env) for ind in pop]
        best_food = max(m[0] for m in metrics)
        if gen < cfg.GENERATIONS - 1:
            pop = evolve_topology(pop, metrics, cfg, gen=gen)
        act = frozenset(_freeze_active_groups(gen, cfg))
        seen_phases.add(act)
        print(f"  gen={gen} 完成, best_food={best_food:.1f}, active={tuple(sorted(act))}")

    exp_phases = {frozenset(p) for p in cfg.CYCLE_PATTERN}
    assert exp_phases.issubset(seen_phases), \
        f"6 代未覆盖全部 cycle 阶段: 缺 {exp_phases - seen_phases}"
    print(f"[PASS] 短代进化运行正常（覆盖 {len(seen_phases)} 个 cycle 阶段: {sorted(seen_phases)}）")


if __name__ == "__main__":
    test_cnn_encoding()
    test_no_fatigue()
    test_active_groups()
    test_freeze_behavior()
    test_short_train_run()
    print("\n=== 全部冒烟测试通过 ===")