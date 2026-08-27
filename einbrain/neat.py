"""真正的 NEAT（NeuroEvolution of Augmenting Topologies）进化引擎，驱动 EI-RNN。

相对 einbrain.evolve（NEAT 风格掩码翻转）的关键差异：
    - 创新号（innovation number）：全局 (src,dst) → id 注册表，跨代持久化，
      使交叉可按『历史标记』对齐基因。
    - 结构生长：add-connection（新增连接）/ add-node（拆分连接 → 激活闲置柱）
      取代随机掩码翻转，拓扑在进化中单调累积。
    - 物种化（speciation）：按相容性距离聚类，适应度共享保护创新。
    - 交叉（crossover）：按创新号匹配，disjoint/excess 基因来自更优父本。

性能（POP=1024 / N=256 规模化）：
    - 基因组缓存排序 innov 数组 + 对齐权重/使能数组，变异时失效；
      compatibility_distance 用 numpy searchsorted 向量化 → ~0.3ms/对。
    - 繁殖时后代继承父本物种；仅每 RE_SPECIATE_INTERVAL 代全量再物种化
      （保持阈值自适应），其余代按 species_id 分组 → 每代 O(POP×物种数)。
    - SPECIES_CAP 兜底：全量再物种化时物种超上限即并入最近物种，杜绝 O(POP²)。
    - decode_population 批量 scatter 构建，消除逐基因组标量张量赋值。

与 EI-RNN 的对应关系（256 柱方案启动）：
    - 节点：O 输入 + N 兴奋-抑制柱（固定 256）+ A 输出。
    - 连接基因：input→column / column→column（无自环）/ column→output。
    - 每柱动力学属性（tau_e / w_ei / w_ie）作为节点基因参与进化。
    - decode() 将基因组还原为 einbrain.gpu.GeneStack 可批量加载的状态 dict。

初始拓扑 = test5d 的 INIT_DENSITY 稀疏随机连通（为每条连接分配创新号），
NEAT 的 add-connection / add-node 在此基础上继续生长。
"""
from __future__ import annotations

import random

import numpy as np
import torch

from .evolve import _dynamic_mutation_rates


class InnovationRegistry:
    """全局创新号注册表：(src,dst) → id；跨代/跨个体共享，保证基因可对齐。"""

    def __init__(self):
        self.pair_to_innov = {}
        self.innov_to_pair = {}
        self.counter = 0
        self._arrays = None          # 缓存：排序 innov → (src, dst) 数组

    def get(self, src, dst):
        key = (src, dst)
        innov = self.pair_to_innov.get(key)
        if innov is None:
            innov = self.counter
            self.counter += 1
            self.pair_to_innov[key] = innov
            self.innov_to_pair[innov] = key
            self._arrays = None
        return innov

    def pair(self, innov):
        return self.innov_to_pair.get(innov)

    def _ensure_arrays(self):
        """返回 (sorted_innovs, srcs, dsts) numpy 数组，供向量化解码。"""
        if self._arrays is None:
            itp = self.innov_to_pair
            keys = np.fromiter(itp.keys(), dtype=np.int64, count=len(itp))
            vals = np.array(list(itp.values()), dtype=np.int64).reshape(-1, 2)
            order = np.argsort(keys)
            keys = keys[order]
            vals = vals[order]
            self._arrays = (keys, vals[:, 0], vals[:, 1])
        return self._arrays

    def state(self):
        return {'pair_to_innov': self.pair_to_innov,
                'innov_to_pair': self.innov_to_pair,
                'counter': self.counter}

    def load_state(self, st):
        self.pair_to_innov = st['pair_to_innov']
        self.innov_to_pair = st['innov_to_pair']
        self.counter = st['counter']
        self._arrays = None


class Genome:
    """EI-RNN 的 NEAT 基因组。

    连接以 {innov: (weight, enabled)} 存储，src/dst 由注册表反查（省内存）。
    tau_e / w_ei / w_ie 为 [N] 张量（每柱动力学属性，视为节点基因）。

    性能缓存：`_ensure_cache()` 惰性构建『排序 innov + 对齐权重/使能』numpy
    数组，供向量化相容距离 / 批量解码 / 健康指标复用；变异后失效。
    """

    def __init__(self, cfg, registry, conns, tau_e, w_ei, w_ie):
        self.cfg = cfg
        self.registry = registry
        self.conns = dict(conns)          # innov -> (weight, enabled)
        self.tau_e = tau_e
        self.w_ei = w_ei
        self.w_ie = w_ie
        self.fitness = None
        self.species_id = None
        self.generation = 0
        self._cache = None

    # ---------- 属性 ----------
    @property
    def O(self):
        return self.cfg.OBS_DIM

    @property
    def N(self):
        return self.cfg.NUM_COLUMNS

    @property
    def A(self):
        return self.cfg.ACTION_DIM

    def _ensure_cache(self):
        """(sorted_innovs[np.int64], weights[np.float32], enabled[np.bool])。"""
        if self._cache is None:
            c = self.conns
            innovs = np.fromiter(c.keys(), dtype=np.int64, count=len(c))
            weights = np.fromiter((v[0] for v in c.values()), dtype=np.float32, count=len(c))
            enabled = np.fromiter((v[1] for v in c.values()), dtype=np.bool_, count=len(c))
            order = np.argsort(innovs)
            self._cache = (innovs[order], weights[order], enabled[order])
        return self._cache

    def _invalidate(self):
        self._cache = None

    def active_conns(self):
        return self._ensure_cache()[0][self._ensure_cache()[2]].tolist()

    def n_active_conns(self):
        return int(self._ensure_cache()[2].sum())

    def n_active_columns(self):
        """有入向启连连接的柱数（dst ∈ [O, O+N) 的启连连接的去重源计数）。"""
        innovs, _, enabled = self._ensure_cache()
        act = innovs[enabled]
        if act.size == 0:
            return 0
        _, srcs, dsts = self.registry._ensure_arrays()
        locs = np.searchsorted(_, act)
        d = dsts[locs]
        cols = d[(d >= self.O) & (d < self.O + self.N)] - self.O
        return int(np.unique(cols).size)

    def active_columns(self):
        innovs, _, enabled = self._ensure_cache()
        act = innovs[enabled]
        if act.size == 0:
            return []
        _, srcs, dsts = self.registry._ensure_arrays()
        locs = np.searchsorted(_, act)
        d = dsts[locs]
        cols = d[(d >= self.O) & (d < self.O + self.N)] - self.O
        return sorted(np.unique(cols).tolist())

    def clone(self):
        g = Genome(self.cfg, self.registry, self.conns,
                   self.tau_e.clone(), self.w_ei.clone(), self.w_ie.clone())
        g.fitness = self.fitness
        g.species_id = self.species_id
        g.generation = self.generation
        return g

    # ---------- 解码 → GeneStack 状态 dict ----------
    def decode(self):
        O, N, A = self.O, self.N, self.A
        M_in = torch.zeros(N, O)
        M_rec = torch.zeros(N, N)
        M_out = torch.zeros(A, N)
        W_in = torch.zeros(N, O)
        W_rec = torch.zeros(N, N)
        W_out = torch.zeros(A, N)
        for innov, (w, e) in self.conns.items():
            if not e:
                continue
            s, d = self.registry.pair(innov)
            if s < O and d < O + N:                    # input → column
                i, j = d - O, s
                M_in[i, j] = 1.0
                W_in[i, j] = w
            elif d < O + N:                            # column → column
                i, j = d - O, s - O
                M_rec[i, j] = 1.0
                W_rec[i, j] = w
            else:                                      # column → output
                a, i = d - (O + N), s - O
                M_out[a, i] = 1.0
                W_out[a, i] = w
        return {
            'N': N,
            'M_in': M_in, 'M_rec': M_rec, 'M_out': M_out,
            'W_in': W_in, 'W_rec': W_rec, 'W_out': W_out,
            'b_out': torch.zeros(A),
            'tau_e_init': self.tau_e.clone(),
            'w_ei': self.w_ei.clone(),
            'w_ie': self.w_ie.clone(),
            # 以下为 einbrain.io.load_brain_state 兼容字段（NEAT 无激素支路，补零）
            'W_hormone1': torch.zeros(self.cfg.HORMONE_NET_HIDDEN, N * 3),
            'b_hormone1': torch.zeros(self.cfg.HORMONE_NET_HIDDEN),
            'W_excit': torch.zeros(N, self.cfg.HORMONE_NET_HIDDEN),
            'b_excit': torch.zeros(N),
            'W_inhib': torch.zeros(N, self.cfg.HORMONE_NET_HIDDEN),
            'b_inhib': torch.zeros(N),
            'V': torch.zeros(N),
            'b_v': torch.zeros(1),
        }


# ==================== 初始化（256 柱方案）====================


def random_genome(cfg, registry, rng=random):
    """test5d INIT_DENSITY 稀疏随机拓扑 + 创新号分配。"""
    O, N, A = cfg.OBS_DIM, cfg.NUM_COLUMNS, cfg.ACTION_DIM
    tau = (cfg.BASE_TAU_E + (torch.rand(N) * 2 - 1) * cfg.TAU_E_NOISE
           ).clamp(cfg.TAU_E_MIN, cfg.TAU_E_MAX)
    w_ei = torch.full((N,), cfg.W_EI)
    w_ie = torch.full((N,), cfg.W_IE)

    conns = {}
    for i in range(N):
        for j in range(O):
            if rng.random() < cfg.INIT_DENSITY:
                innov = registry.get(j, O + i)
                conns[innov] = (rng.gauss(0.0, 0.1), True)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if rng.random() < cfg.INIT_DENSITY:
                innov = registry.get(O + i, O + j)
                conns[innov] = (rng.gauss(0.0, 0.05), True)
    for a in range(A):
        for i in range(N):
            if rng.random() < cfg.INIT_DENSITY:
                innov = registry.get(O + i, O + N + a)
                conns[innov] = (rng.gauss(0.0, 0.1), True)
    return Genome(cfg, registry, conns, tau, w_ei, w_ie)


# ==================== 变异算子 ====================


def mutate_add_connection(g, cfg, rng=random):
    """新增一条未存在的连接（input→col / col→col / col→output）。"""
    O, N, A = g.O, g.N, g.A
    present = set(g.conns.keys())
    for _ in range(60):
        t = rng.randrange(3)
        if t == 0:
            src, dst = rng.randrange(O), O + rng.randrange(N)
        elif t == 1:
            i, j = rng.randrange(N), rng.randrange(N)
            if i == j:
                continue
            src, dst = O + i, O + j
        else:
            src, dst = O + rng.randrange(N), O + N + rng.randrange(A)
        innov = g.registry.get(src, dst)
        if innov not in present:
            g.conns[innov] = (rng.gauss(0.0, 0.1), True)
            g._invalidate()
            return True
    return False


def mutate_add_node(g, cfg, rng=random):
    """add-node：拆分一条启用连接，插入一个中继柱。

    原连接禁用；新增 src→col（权重 1.0）与 col→dst（权重=原权重）两条连接。
    256 柱 INIT_DENSITY 方案下所有柱均已激活，故中继柱从非 src/dst 的任意柱
    中选择（避免自环），等效于在固定 N 柱架构中新增一个两跳中继节点。
    """
    enabled = [(innov, w) for innov, (w, e) in g.conns.items() if e]
    if not enabled:
        return False
    innov, w = rng.choice(enabled)
    src, dst = g.registry.pair(innov)
    candidates = [c for c in range(g.N) if (g.O + c) not in (src, dst)]
    if not candidates:
        return False
    col = rng.choice(candidates)
    g.conns[innov] = (w, False)
    in_innov = g.registry.get(src, g.O + col)
    out_innov = g.registry.get(g.O + col, dst)
    g.conns[in_innov] = (1.0, True)
    g.conns[out_innov] = (w, True)
    g._invalidate()
    return True


def mutate_disable(g, cfg, rng=random):
    """随机禁用一条启用连接。"""
    enabled = [innov for innov, (_, e) in g.conns.items() if e]
    if not enabled:
        return False
    innov = rng.choice(enabled)
    w, _ = g.conns[innov]
    g.conns[innov] = (w, False)
    g._invalidate()
    return True


def mutate_reenable(g, rng=random):
    """随机重新启用一条被禁用的连接。"""
    disabled = [innov for innov, (_, e) in g.conns.items() if not e]
    if not disabled:
        return False
    innov = rng.choice(disabled)
    w, _ = g.conns[innov]
    g.conns[innov] = (w, True)
    g._invalidate()
    return True


def mutate_weights(g, cfg, mut, rng=random):
    """权重扰动（NEAT：小概率整体重置，否则高斯扰动）。"""
    frac = mut['weight_mut_frac']
    std = mut['weight_mut_std']
    reset_std = std * 3.0
    for innov in list(g.conns.keys()):
        w, e = g.conns[innov]
        if rng.random() < frac:
            if rng.random() < 0.1:
                w = rng.gauss(0.0, reset_std)
            else:
                w += rng.gauss(0.0, std)
        g.conns[innov] = (w, e)
    g._invalidate()


def mutate_node_attrs(g, cfg, mut, rng=random):
    """每柱动力学属性（G2：tau_e / w_ei / w_ie）扰动。"""
    with torch.no_grad():
        g.tau_e = torch.clamp(g.tau_e + torch.randn(g.N) * mut['tau_e_mut_std'],
                              cfg.TAU_E_MIN, cfg.TAU_E_MAX)
        g.w_ei = torch.clamp(g.w_ei + torch.randn(g.N) * mut['w_ei_mut_std'],
                             cfg.W_EI_MIN, cfg.W_EI_MAX)
        g.w_ie = torch.clamp(g.w_ie + torch.randn(g.N) * mut['w_ie_mut_std'],
                             cfg.W_IE_MIN, cfg.W_IE_MAX)


def mutate_genome(g, cfg, mut, rng=random):
    """组合变异：拓扑（add-conn/add-node/disable）+ 权重 + 动力学属性。"""
    if rng.random() < mut['topo_mut_prob']:
        r = rng.random()
        if r < cfg.ADD_CONN_PROB:
            mutate_add_connection(g, cfg, rng)
        elif r < cfg.ADD_CONN_PROB + cfg.ADD_NODE_PROB:
            mutate_add_node(g, cfg, rng)
        else:
            mutate_disable(g, cfg, rng)
    if rng.random() < getattr(cfg, 'REENABLE_PROB', 0.1):
        mutate_reenable(g, rng)
    mutate_weights(g, cfg, mut, rng)
    mutate_node_attrs(g, cfg, mut, rng)


# ==================== 相容性距离与交叉 ====================


def compatibility_distance(g1, g2, cfg):
    """NEAT 相容性距离：excess/disjoint（按基因数归一）+ 匹配基因权重差。

    使用缓存排序 innov 数组 + numpy searchsorted 全向量化：
        matching   = 创新号交集（历史标记对齐）
        excess     = 超出对方最大创新号的非匹配基因
        disjoint   = 范围内非匹配基因
        Wbar       = 匹配基因权重差均值
    归一化用两基因组中较大的基因数（标准 NEAT）：本库基因组在 256 柱下
    连接数 ~10⁴，基因数归一使 add-connection 带来的单条差异相对权重噪声
    可忽略，物种由整体相似度聚类，配合自适应阈值收敛。
    """
    c1, c2, c3 = cfg.COMPAT_C1, cfg.COMPAT_C2, cfg.COMPAT_C3
    i1, w1a, _ = g1._ensure_cache()
    i2, w2a, _ = g2._ensure_cache()
    n1, n2 = i1.size, i2.size

    locs = np.searchsorted(i2, i1)
    valid = locs < n2
    match1 = valid & (i2[np.clip(locs, 0, n2 - 1)] == i1)
    matching = int(match1.sum())

    max1 = int(i1[-1]) if n1 else -1
    max2 = int(i2[-1]) if n2 else -1
    excess_1 = int((i1 > max2).sum())
    excess_2 = int((i2 > max1).sum())
    disjoint_n = (n1 - matching - excess_1) + (n2 - matching - excess_2)

    if matching:
        mw1 = w1a[match1]
        mw2 = w2a[np.clip(locs, 0, n2 - 1)[match1]]
        wbar = float(np.abs(mw1 - mw2).mean())
    else:
        wbar = 0.0

    norm = max(n1, n2, 1)
    return (c1 * (excess_1 + excess_2) + c2 * disjoint_n) / norm + c3 * wbar


def crossover(g1, g2, rng=random):
    """历史标记交叉。g1 为更优父本；匹配基因随机继承，disjoint/excess 取自 g1。

    被继承的 disabled 基因有 25% 概率重新启用（NEAT 惯例，保留创新潜质）。
    每柱动力学属性按列随机继承（对齐 test5d G2 交叉语义）。
    """
    f_innovs = set(g1.conns.keys())
    o_innovs = set(g2.conns.keys())
    new_conns = {}

    for innov in f_innovs & o_innovs:
        w1, e1 = g1.conns[innov]
        w2, e2 = g2.conns[innov]
        if rng.random() < 0.5:
            w, e = w1, e1
        else:
            w, e = w2, e2
        if not e and rng.random() < 0.25:
            e = True
        new_conns[innov] = (w, e)

    for innov in f_innovs - o_innovs:
        new_conns[innov] = g1.conns[innov]

    col_mask = torch.rand(g1.N) > 0.5
    tau = torch.where(col_mask, g1.tau_e, g2.tau_e)
    w_ei = torch.where(col_mask, g1.w_ei, g2.w_ei)
    w_ie = torch.where(col_mask, g1.w_ie, g2.w_ie)

    g = Genome(g1.cfg, g1.registry, new_conns, tau, w_ei, w_ie)
    g.generation = max(g1.generation, g2.generation)
    return g


# ==================== 种群 / 物种化 ====================


class NEATPopulation:
    """NEAT 种群容器：初始化 / 物种化 / 繁殖。

    用法（test8）：
        pop = NEATPopulation(cfg); pop.init_random()
        ... 解码评估得到 fitnesses ...
        pop.evolve(fitnesses, gen)

    物种化策略（加速）：
        - 每 RE_SPECIATE_INTERVAL 代做一次全量再物种化（阈值自适应）；
        - 其余代后代继承父本物种，按 species_id 分组（O(POP)）；
        - SPECIES_CAP 兜底：全量再物种化时物种超上限即并入最近物种。
    """

    def __init__(self, cfg, registry=None, genomes=None):
        self.cfg = cfg
        self.registry = registry if registry is not None else InnovationRegistry()
        self.genomes = genomes if genomes is not None else []
        self.species = {}
        self.compat_threshold = float(getattr(cfg, 'COMPAT_THRESHOLD_INIT', 1.0))

    def init_random(self, rng=random):
        self.genomes = [random_genome(self.cfg, self.registry, rng)
                        for _ in range(self.cfg.POP_SIZE)]

    # ---------- 物种化 ----------
    def _compat(self, g1, g2):
        return compatibility_distance(g1, g2, self.cfg)

    def speciate(self):
        """全量再物种化：按相容性距离聚类，SPECIES_CAP 兜底防 O(POP²)。"""
        cfg = self.cfg
        cap = int(getattr(cfg, 'SPECIES_CAP', 64))
        for g in self.genomes:
            g.species_id = None
        new_species = {}
        for g in self.genomes:
            if not new_species:
                new_species[0] = {'rep': g, 'members': [g]}
                g.species_id = 0
                continue
            best_sid, best_d = 0, None
            for sid, sp in new_species.items():
                d = self._compat(sp['rep'], g)
                if best_d is None or d < best_d:
                    best_d, best_sid = d, sid
            if best_d < self.compat_threshold or len(new_species) >= cap:
                new_species[best_sid]['members'].append(g)
                g.species_id = best_sid
            else:
                sid = len(new_species)
                new_species[sid] = {'rep': g, 'members': [g]}
                g.species_id = sid
        self.species = new_species

        target = int(getattr(cfg, 'SPECIES_TARGET', 8))
        n = len(self.species)
        if n > target:
            self.compat_threshold *= 1.03
        elif n < target and n > 0:
            self.compat_threshold *= 0.97

    def _group_by_species(self):
        """按物种_id 分组（父本继承代，无距离计算）。"""
        groups = {}
        for g in self.genomes:
            sid = g.species_id if g.species_id is not None else 0
            groups.setdefault(sid, []).append(g)
        new = {}
        for sid, members in groups.items():
            members.sort(key=lambda g: -g.fitness)
            k = len(new)
            new[k] = {'rep': members[0], 'members': members}
            for m in members:
                m.species_id = k
        return new

    # ---------- 繁殖 ----------
    def evolve(self, fitnesses, gen=0, rng=random):
        cfg = self.cfg
        for g, f in zip(self.genomes, fitnesses):
            g.fitness = float(f)

        interval = int(getattr(cfg, 'RE_SPECIATE_INTERVAL', 5))
        if gen % interval == 0:
            self.speciate()
        else:
            self.species = self._group_by_species()
        health = self.health()

        # 每物种适应度 = 共享后总和（fitness / 规模 → 平均），据此分配子代预算
        stats = []
        total_adj = 0.0
        for sid, sp in self.species.items():
            members = sp['members']
            n = len(members)
            adj = sum(max(0.0, m.fitness) for m in members) / n
            stats.append({'sid': sid, 'members': members, 'adj': adj, 'n': n})
            total_adj += adj

        mut = _dynamic_mutation_rates(cfg, gen)

        counts = []
        for st in stats:
            if total_adj > 0 and st['adj'] > 0:
                counts.append(max(1, int(round(st['adj'] / total_adj * cfg.POP_SIZE))))
            else:
                counts.append(1)
        diff = cfg.POP_SIZE - sum(counts)
        if diff != 0 and stats:
            order = sorted(range(len(stats)), key=lambda i: -stats[i]['adj'])
            i = 0
            while diff != 0:
                idx = order[i % len(order)]
                if diff > 0:
                    counts[idx] += 1
                    diff -= 1
                else:
                    if counts[idx] > 1:
                        counts[idx] -= 1
                        diff += 1
                i += 1

        # 全局精英（跨物种，原样保留）；各物种其余配额全部产出变异后代。
        # 后代继承父本物种（加速物种化），保证单例物种也有变异压力。
        n_elite_global = min(int(getattr(cfg, 'SPECIES_ELITE', 1)), len(self.genomes))
        order_global = sorted(range(len(self.genomes)), key=lambda i: -self.genomes[i].fitness)
        new_genomes = []
        for k in range(n_elite_global):
            elite = self.genomes[order_global[k]].clone()
            new_genomes.append(elite)

        for st, cnt in zip(stats, counts):
            members = sorted(st['members'], key=lambda g: -g.fitness)
            pool = members[:max(1, len(members) // 2)]
            for _ in range(cnt):
                p1 = rng.choice(pool)
                if len(pool) > 1:
                    p2 = rng.choice(pool)
                    child = crossover(p1, p2, rng)
                else:
                    child = p1.clone()
                mutate_genome(child, cfg, mut, rng)
                child.generation = gen + 1
                child.species_id = p1.species_id
                new_genomes.append(child)

        # 精确到 POP_SIZE
        if len(new_genomes) > cfg.POP_SIZE:
            new_genomes = new_genomes[:cfg.POP_SIZE]
        while len(new_genomes) < cfg.POP_SIZE:
            new_genomes.append(rng.choice(self.genomes).clone())

        self.genomes = new_genomes
        self.species = {}
        return health

    # ---------- 断点 ----------
    def pack(self):
        return {
            'registry': self.registry.state(),
            'genomes': [
                {'innovs': g._ensure_cache()[0], 'weights': g._ensure_cache()[1],
                 'enabled': g._ensure_cache()[2],
                 'tau_e': g.tau_e.tolist(), 'w_ei': g.w_ei.tolist(), 'w_ie': g.w_ie.tolist(),
                 'fitness': g.fitness, 'generation': g.generation,
                 'species_id': g.species_id}
                for g in self.genomes
            ],
        }

    def unpack(self, payload, cfg):
        self.cfg = cfg
        self.registry.load_state(payload['registry'])
        self.genomes = []
        for rec in payload['genomes']:
            innovs = rec['innovs']
            weights = rec['weights']
            enabled = rec['enabled']
            conns = {int(innovs[k]): (float(weights[k]), bool(enabled[k]))
                     for k in range(len(innovs))}
            g = Genome(cfg, self.registry, conns,
                       torch.tensor(rec['tau_e']),
                       torch.tensor(rec['w_ei']), torch.tensor(rec['w_ie']))
            g.fitness = rec['fitness']
            g.generation = rec.get('generation', 0)
            g.species_id = rec.get('species_id')
            self.genomes.append(g)
        self.species = {}

    # ---------- 契合度健康指标 ----------
    def health(self):
        n_conns = [g.n_active_conns() for g in self.genomes]
        n_cols = [g.n_active_columns() for g in self.genomes]
        return {
            'n_species': len(self.species),
            'avg_conns': sum(n_conns) / len(n_conns),
            'avg_cols': sum(n_cols) / len(n_cols),
        }


def decode_population(genomes, cfg):
    """整代解码为 GeneStack 可直接赋值的堆叠张量 dict（批量 scatter 构建）。

    返回 {M_in/W_in: [P,N,O], M_rec/W_rec: [P,N,N], M_out/W_out: [P,A,N],
          b_out: [P,A], tau_e/w_ei/w_ie: [P,N]}。

    利用每基因组缓存的排序 innov 数组 + 注册表 innov→(src,dst) 数组，
    一次向量化映射 + index_put_ 落盘，消除逐基因组的标量张量赋值。
    """
    P = len(genomes)
    N, O, A = cfg.NUM_COLUMNS, cfg.OBS_DIM, cfg.ACTION_DIM
    out = {
        'M_in': torch.zeros(P, N, O), 'W_in': torch.zeros(P, N, O),
        'M_rec': torch.zeros(P, N, N), 'W_rec': torch.zeros(P, N, N),
        'M_out': torch.zeros(P, A, N), 'W_out': torch.zeros(P, A, N),
        'b_out': torch.zeros(P, A),
        'tau_e': torch.zeros(P, N), 'w_ei': torch.zeros(P, N), 'w_ie': torch.zeros(P, N),
    }
    if P == 0:
        return out

    gi_in, ii_in, jj_in, ww_in = [], [], [], []
    gi_rec, ii_rec, jj_rec, ww_rec = [], [], [], []
    gi_out, aa_out, ii_out, ww_out = [], [], [], []

    for p, g in enumerate(genomes):
        innovs, weights, enabled = g._ensure_cache()
        if enabled.size == 0:
            continue
        act_i = innovs[enabled]
        act_w = weights[enabled]
        keys, srcs, dsts = g.registry._ensure_arrays()
        locs = np.searchsorted(keys, act_i)
        src = srcs[locs]
        dst = dsts[locs]
        # input → column
        m_in = (src < O) & (dst < O + N)
        gi_in.append(np.full(int(m_in.sum()), p, dtype=np.int64))
        ii_in.append(dst[m_in] - O)
        jj_in.append(src[m_in])
        ww_in.append(act_w[m_in])
        # column → column（dst ∈ [O, O+N) 且 src ≥ O）
        m_rec = (dst < O + N) & ~m_in
        gi_rec.append(np.full(int(m_rec.sum()), p, dtype=np.int64))
        ii_rec.append(dst[m_rec] - O)
        jj_rec.append(src[m_rec] - O)
        ww_rec.append(act_w[m_rec])
        # column → output
        m_out = ~m_in & ~m_rec
        gi_out.append(np.full(int(m_out.sum()), p, dtype=np.int64))
        aa_out.append(dst[m_out] - (O + N))
        ii_out.append(src[m_out] - O)
        ww_out.append(act_w[m_out])

        out['tau_e'][p] = g.tau_e
        out['w_ei'][p] = g.w_ei
        out['w_ie'][p] = g.w_ie

    def _scatter(mask_t, w_t, gis, is_, js, ws):
        if gis:
            gi = torch.from_numpy(np.concatenate(gis))
            ii = torch.from_numpy(np.concatenate(is_))
            jj = torch.from_numpy(np.concatenate(js))
            wv = torch.from_numpy(np.concatenate(ws)).float()
            mask_t.index_put_((gi, ii, jj), torch.ones_like(wv))
            w_t.index_put_((gi, ii, jj), wv)

    _scatter(out['M_in'], out['W_in'], gi_in, ii_in, jj_in, ww_in)
    _scatter(out['M_rec'], out['W_rec'], gi_rec, ii_rec, jj_rec, ww_rec)
    if gi_out:
        gi = torch.from_numpy(np.concatenate(gi_out))
        aa = torch.from_numpy(np.concatenate(aa_out))
        ii = torch.from_numpy(np.concatenate(ii_out))
        wv = torch.from_numpy(np.concatenate(ww_out)).float()
        out['M_out'].index_put_((gi, aa, ii), torch.ones_like(wv))
        out['W_out'].index_put_((gi, aa, ii), wv)

    return out
