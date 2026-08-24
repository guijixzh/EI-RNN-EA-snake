"""test8 相关单元测试：RaySnakeEnv 观测 / CPU-GPU 一致性 / NEAT 引擎。

运行：
    python einbrain/tests/test_neat.py
"""
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from einbrain import Config  # noqa: E402
from einbrain.env import RaySnakeEnv  # noqa: E402


def _pass(name):
    print(f"[PASS] {name}")


def _smoke_cfg():
    cfg = Config()
    cfg.OBS_DIM = 32
    cfg.POP_SIZE = 16
    cfg.NUM_COLUMNS = 24
    cfg.INIT_DENSITY = 0.15
    return cfg


def test_raysnake_env_invariants():
    import random
    random.seed(7)
    env = RaySnakeEnv(grid_size=10, max_steps=200)
    for _ in range(4):
        obs = env.reset()
        assert obs.shape == (32,)
        assert abs(obs[0:4].sum() - 1) < 1e-6, "蛇首方向 one-hot 恰 1 位"
        assert abs(obs[4:8].sum() - 1) < 1e-6, "蛇尾方向 one-hot 恰 1 位"
        assert abs(obs[8:16].sum() - 1) < 1e-6, "食物恰落在 1 个扇区"
        assert obs[16 + 4] == 1.0, "颈节恒在正后方 → 自身[后]=1"
        assert abs(obs[24 + 4] - math.sqrt(len(env.body) / env.grid_size)) < 1e-6, "身后障碍 = sqrt(蛇身长度/格子度)"
        for v in obs[24:32]:
            assert 0 < v <= 1.0, "障碍倒数 ∈ (0,1]"
        for _ in range(30):
            obs, _, d, t = env.step(np.random.choice([0, 1, 2]))
            if d or t:
                break
    _pass("RaySnakeEnv 32 维观测不变量")


def test_cpu_gpu_obs_parity():
    from einbrain.gpu import BatchedRaySnakeEnv

    class Cfg:
        GRID_SIZE = 10
        MAX_STEPS = 500
        EVAL_EPISODES = 1

    ce = RaySnakeEnv(grid_size=10, max_steps=500)
    ce.head = (5, 5)
    ce.dir = (0, 1)
    ce.body = [(5, 5), (5, 4), (5, 3), (4, 3), (4, 4)]
    ce.food = (7, 5)
    obs_cpu = ce._get_obs()

    ge = BatchedRaySnakeEnv(Cfg(), 1, torch.device('cpu'))
    ge.head = torch.tensor([[5, 5]])
    ge.dir_idx = torch.tensor([0])
    body = torch.zeros(1, ge.MAXLEN, 2, dtype=torch.long)
    for i, (r, c) in enumerate([(5, 5), (5, 4), (5, 3), (4, 3), (4, 4)]):
        body[0, i] = torch.tensor([r, c])
    ge.body = body
    ge.body_len = torch.tensor([5])
    ge.food = torch.tensor([[7, 5]])
    obs_gpu = ge.obs()[0].numpy()

    diff = float(np.abs(obs_cpu - obs_gpu).max())
    assert diff < 1e-5, f"CPU/GPU 观测不一致: {diff}"
    _pass(f"RaySnakeEnv CPU/GPU 同状态观测一致（max diff = {diff}）")


def test_innovation_registry():
    from einbrain.neat import InnovationRegistry
    reg = InnovationRegistry()
    a = reg.get(1, 2)
    b = reg.get(1, 2)
    c = reg.get(2, 1)
    assert a == b and a != c
    assert reg.pair(a) == (1, 2)
    assert reg.counter == 2
    _pass("创新号注册表唯一性 + 反查")


def test_neat_mutations_and_crossover():
    from einbrain.neat import (NEATPopulation, crossover, mutate_add_connection,
                               mutate_add_node, mutate_disable)
    cfg = _smoke_cfg()
    pop = NEATPopulation(cfg)
    pop.init_random()
    g1, g2 = pop.genomes[0], pop.genomes[1]
    n0 = len(g1.conns)

    assert mutate_add_connection(g1, cfg), "add-connection 应成功"
    assert len(g1.conns) == n0 + 1, "add-connection 新增 1 条"

    n1 = len(g1.conns)
    disabled_before = sum(1 for _, e in g1.conns.values() if not e)
    assert mutate_add_node(g1, cfg), "add-node 应成功"
    disabled_after = sum(1 for _, e in g1.conns.values() if not e)
    assert disabled_after == disabled_before + 1, "add-node 恰好禁用 1 条原连接"
    assert n1 <= len(g1.conns) <= n1 + 2, "add-node 新增 0~2 条（存在平行连接时覆盖）"

    n2 = len(g1.conns)
    assert mutate_disable(g1, cfg), "disable 应成功"
    assert len(g1.conns) == n2, "disable 不改基因数"

    child = crossover(g1, g2)
    i1 = set(g1.conns.keys())
    i2 = set(g2.conns.keys())
    assert set(child.conns.keys()) == (i1 & i2) | (i1 - i2), "交叉后代基因 = 匹配 + fitter 独有"
    _pass("NEAT add-connection/add-node/disable/历史标记交叉")


def test_decode_and_decode_population():
    from einbrain.neat import NEATPopulation, decode_population
    from einbrain.gpu import GeneStack
    cfg = _smoke_cfg()
    pop = NEATPopulation(cfg)
    pop.init_random()
    st = pop.genomes[0].decode()
    assert st['M_in'].shape == (24, 32)
    assert st['M_rec'].shape == (24, 24)
    assert st['M_out'].shape == (3, 24)
    assert st['M_rec'].diag().sum() == 0, "M_rec 无自环"

    tens = decode_population(pop.genomes, cfg)
    gs = GeneStack(cfg, B=cfg.POP_SIZE, device=torch.device('cpu'))
    for k in GeneStack.GENES:
        setattr(gs, k, tens[k])
    gs.refresh_eff()
    assert gs.W_in_eff.shape == (16, 24, 32)
    _pass("decode → GeneStack 批量加载")


def test_population_evolve_and_roundtrip():
    from einbrain.neat import NEATPopulation
    cfg = _smoke_cfg()
    cfg.POP_SIZE = 12
    pop = NEATPopulation(cfg)
    pop.init_random()
    rng = np.random.default_rng(0)
    fitness = rng.random(12)
    health = pop.evolve(fitness, gen=0)
    assert len(pop.genomes) == 12
    assert {'n_species', 'avg_conns', 'avg_cols'} <= set(health.keys())
    assert health['n_species'] >= 1

    payload = pop.pack()
    pop2 = NEATPopulation(cfg)
    pop2.unpack(payload, cfg)
    assert len(pop2.genomes) == 12
    assert pop2.registry.counter == pop.registry.counter
    assert set(pop2.genomes[0].conns.keys()) == set(pop.genomes[0].conns.keys())
    _pass("NEATPopulation evolve + pack/unpack 往返")


def test_decode_batch_matches_per_genome():
    from einbrain.neat import NEATPopulation, decode_population
    cfg = _smoke_cfg()
    cfg.POP_SIZE = 8
    pop = NEATPopulation(cfg)
    pop.init_random()
    tens = decode_population(pop.genomes, cfg)
    for p, g in enumerate(pop.genomes):
        st = g.decode()
        assert torch.equal(tens['M_in'][p], st['M_in'])
        assert torch.equal(tens['M_rec'][p], st['M_rec'])
        assert torch.equal(tens['M_out'][p], st['M_out'])
        assert torch.equal(tens['W_in'][p], st['W_in'])
        assert torch.equal(tens['W_rec'][p], st['W_rec'])
        assert torch.equal(tens['W_out'][p], st['W_out'])
        assert torch.allclose(tens['tau_e'][p], st['tau_e_init'])
        assert torch.allclose(tens['w_ei'][p], st['w_ei'])
        assert torch.allclose(tens['w_ie'][p], st['w_ie'])
    _pass("decode_population 批量构建 == 逐个体 decode")


def test_compat_vectorized_matches_reference():
    from einbrain.neat import NEATPopulation, compatibility_distance
    cfg = _smoke_cfg()
    cfg.POP_SIZE = 12
    pop = NEATPopulation(cfg)
    pop.init_random()

    def compat_ref(g1, g2):
        c1, c2, c3 = cfg.COMPAT_C1, cfg.COMPAT_C2, cfg.COMPAT_C3
        i1, i2 = set(g1.conns.keys()), set(g2.conns.keys())
        max1, max2 = max(i1) if i1 else -1, max(i2) if i2 else -1
        excess = len([x for x in (i1 - i2) if x > max2]) + \
                 len([x for x in (i2 - i1) if x > max1])
        dn = len((i1 - i2) | (i2 - i1)) - excess
        m = i1 & i2
        wbar = (sum(abs(g1.conns[i][0] - g2.conns[i][0]) for i in m) / len(m)) if m else 0.0
        return (c1 * excess + c2 * dn) / max(len(i1), len(i2), 1) + c3 * wbar

    import itertools
    for a, b in itertools.combinations(pop.genomes[:6], 2):
        assert abs(compatibility_distance(a, b, cfg) - compat_ref(a, b)) < 1e-5
        assert abs(compatibility_distance(a, b, cfg) -
                   compatibility_distance(b, a, cfg)) < 1e-6, "对称性"
    assert compatibility_distance(pop.genomes[0], pop.genomes[0], cfg) == 0.0
    _pass("compatibility_distance 向量化 == 参考实现 + 对称性")


def test_speciation_strategy_and_cap():
    from einbrain.neat import NEATPopulation
    cfg = _smoke_cfg()
    cfg.POP_SIZE = 16
    cfg.SPECIES_CAP = 4
    cfg.RE_SPECIATE_INTERVAL = 2
    pop = NEATPopulation(cfg)
    pop.init_random()
    rng = np.random.default_rng(1)
    h0 = pop.evolve(rng.random(16), gen=0)   # gen0: 全量再物种化
    assert h0['n_species'] <= cfg.SPECIES_CAP, "SPECIES_CAP 生效"
    h1 = pop.evolve(rng.random(16), gen=1)   # gen1: 父本继承分组
    assert 1 <= h1['n_species'] <= cfg.SPECIES_CAP
    h2 = pop.evolve(rng.random(16), gen=2)   # gen2: 再物种化
    assert h2['n_species'] <= cfg.SPECIES_CAP
    # 后代物种_id 非空（父本继承链）
    assert all(g.species_id is not None for g in pop.genomes)
    _pass("父本继承 + 周期再物种化 + SPECIES_CAP 兜底")


def test_neat_gpu_eval_counts_food():
    from einbrain.gpu import GeneStack
    from einbrain.neat import NEATPopulation, decode_population
    from experiments.test8 import evaluate_population_ray

    cfg = _smoke_cfg()
    cfg.POP_SIZE = 8
    cfg.EVAL_EPISODES = 1
    cfg.MAX_STEPS = 50
    cfg.DEVICE = 'cpu'
    cfg.USE_FP16 = False
    cfg.FRAME_RATE = 2
    pop = NEATPopulation(cfg)
    pop.init_random()
    tens = decode_population(pop.genomes, cfg)
    gs = GeneStack(cfg, B=cfg.POP_SIZE, device=torch.device('cpu'))
    for k in GeneStack.GENES:
        setattr(gs, k, tens[k])
    metrics = evaluate_population_ray(gs, cfg)
    assert metrics.shape == (8, 3)
    assert torch.isfinite(metrics).all(), "评估指标含 NaN"
    _pass("NEAT GPU 批量评估（含 food 计数）跑通")


if __name__ == '__main__':
    test_raysnake_env_invariants()
    test_cpu_gpu_obs_parity()
    test_innovation_registry()
    test_neat_mutations_and_crossover()
    test_decode_and_decode_population()
    test_population_evolve_and_roundtrip()
    test_decode_batch_matches_per_genome()
    test_compat_vectorized_matches_reference()
    test_speciation_strategy_and_cap()
    test_neat_gpu_eval_counts_food()
    print("\n=== test8 NEAT 单元测试全部通过 ===")
