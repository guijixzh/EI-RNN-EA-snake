"""test4_32dim 消融实验：隔离 N/门控/排序键 三个变量。

在 test4 原始架构（N=64, 32维, 无预测编码, 无拉马克）上：
  A: 原始（对照）
  B: + influence 门控
  C: + test9a 排序键 (food, -seen, unseen)

每个配置跑 20 代，对比 BestFood 趋势。
"""

import copy, math, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from einbrain.env import RaySnakeEnv, ray_obs_sees_food


# ---- test4 脑模型（无预测编码，简化版用于消融）----
class Brain(nn.Module):
    def __init__(self, N=64, obs_dim=32, act_dim=3, density=0.15):
        super().__init__()
        self.N, self.obs_dim, self.action_dim = N, obs_dim, act_dim
        self.M_in = (torch.rand(N, obs_dim) < density).float()
        self.M_rec = (torch.rand(N, N) < density).float()
        torch.diagonal(self.M_rec).zero_()
        self.M_out = (torch.rand(act_dim, N) < density).float()
        self.W_in = nn.Parameter(torch.randn(N, obs_dim) * 0.1)
        self.W_rec = nn.Parameter(torch.randn(N, N) * 0.05)
        self.W_out = nn.Parameter(torch.randn(act_dim, N) * 0.1)
        self.tau_e, self.w_ei, self.w_ie = 0.7, 2.0, 2.0
        self.baseline = None
        # 门控状态
        self.act_col = torch.zeros(N)
        self._acc_col = torch.zeros(N); self._acc_n = 0
        self.gate_active = False
        self.M0_rec = torch.tensor(0.0)

    def forward(self, obs_t, E, I):
        ext = torch.matmul(self.W_in * self.M_in, obs_t)
        rec = torch.matmul(self.W_rec * self.M_rec, E)
        total = ext + rec
        E_new = torch.sigmoid(total + self.tau_e * E - self.w_ei * I)
        I_new = torch.sigmoid(self.w_ie * E_new)
        logits = torch.matmul(self.W_out * self.M_out, E_new)
        return logits, E_new, I_new

    def save_baseline(self):
        self.baseline = {k: v.clone() if torch.is_tensor(v) else v
                         for k, v in [('W_in', self.W_in.data), ('W_rec', self.W_rec.data),
                                       ('W_out', self.W_out.data), ('M_in', self.M_in),
                                       ('M_rec', self.M_rec), ('M_out', self.M_out)]}

    def restore_baseline(self):
        for k in ['W_in', 'W_rec', 'W_out']:
            getattr(self, k).data = self.baseline[k].clone()
        for k in ['M_in', 'M_rec', 'M_out']:
            setattr(self, k, self.baseline[k].clone())

    # ---- 门控 ----
    def begin_activity(self):
        self._acc_col.zero_(); self._acc_n = 0

    def acc_activity(self, E):
        self._acc_col += E.abs(); self._acc_n += 1

    def end_activity(self):
        if self._acc_n > 0:
            self.act_col = 0.5 * self.act_col + 0.5 * (self._acc_col / self._acc_n)

    def refresh_gate(self, delta):
        if delta <= 0 or not self.gate_active:
            return
        with torch.no_grad():
            if self.M0_rec.item() == 0:
                inf = self.W_rec.data.abs() * self.act_col.unsqueeze(0)
                vals = inf[self.M_rec > 0]
                self.M0_rec = vals.median() if vals.numel() > 0 else torch.tensor(1e-6)
            inf = self.W_rec.data.abs() * self.act_col.unsqueeze(0)
            self.M_rec = self.M_rec * (inf > delta * self.M0_rec).float()
            torch.diagonal(self.M_rec).zero_()


# ---- 评估 ----
def evaluate(brain, env, use_gate=False, max_steps=1000, episodes=5):
    foods, seen_all, unseen_all = [], [], []
    for _ in range(episodes):
        brain.restore_baseline()
        obs = env.reset()
        E = torch.zeros(brain.N); I = torch.zeros(brain.N)
        f = s = u = 0; done = False; steps = 0
        while not done and steps < max_steps:
            obs_t = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                logits, E, I = brain(obs_t, E, I)
            action = torch.argmax(logits).item()
            if use_gate:
                brain.acc_activity(E)
            next_obs, ate, done, trunc = env.step(action)
            done = done or trunc
            if ray_obs_sees_food(obs): s += 1
            else: u += 1
            if ate: f += 1
            obs = next_obs; steps += 1
        foods.append(f); seen_all.append(s); unseen_all.append(u)
    if use_gate:
        brain.end_activity()
    return float(np.mean(foods)), float(np.mean(seen_all)), float(np.mean(unseen_all))


# ---- 进化 ----
def evolve(pop, fitnesses, elite_size=64, mut_rate=0.05):
    idx = np.argsort(fitnesses)[-elite_size:]
    elites = [copy.deepcopy(pop[i]) for i in idx]
    new_pop = [copy.deepcopy(e) for e in elites]
    while len(new_pop) < len(pop):
        p1, p2 = random.sample(elites, 2)
        child = copy.deepcopy(p1); N = child.N
        cm = torch.rand(N) > 0.5; r = cm.unsqueeze(1); c = cm.unsqueeze(0)
        sp1 = r & c; sp2 = (~r) & (~c)
        with torch.no_grad():
            child.W_in.data = torch.where(cm.unsqueeze(1), p1.W_in.data, p2.W_in.data)
            child.M_in = torch.where(cm.unsqueeze(1), p1.M_in, p2.M_in)
            child.W_rec.data = torch.where(sp1, p1.W_rec.data,
                torch.where(sp2, p2.W_rec.data, torch.where(torch.rand_like(p1.W_rec.data)>.5, p1.W_rec.data, p2.W_rec.data)))
            child.M_rec = torch.where(sp1, p1.M_rec, torch.where(sp2, p2.M_rec,
                torch.where(torch.rand_like(p1.M_rec)>.5, p1.M_rec, p2.M_rec)))
            child.W_out.data = torch.where(cm.unsqueeze(0), p1.W_out.data, p2.W_out.data)
            child.M_out = torch.where(cm.unsqueeze(0), p1.M_out, p2.M_out)
        with torch.no_grad():
            if random.random() < 0.05:
                m_attr = random.choice(['M_in', 'M_rec', 'M_out'])
                m = getattr(child, m_attr)
                flip_mask = torch.rand_like(m) < mut_rate
                m[flip_mask] = 1.0 - m[flip_mask]
            for a in ['W_in', 'W_rec', 'W_out']:
                w = getattr(child, a).data
                noise = torch.randn_like(w) * 0.1
                w += noise * (torch.rand_like(w) < 0.2)
        child.save_baseline()
        new_pop.append(child)
    return new_pop


# ---- 主循环 ----
def run_ablation(name, cfg, fitness_fn, use_gate=False):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    N = cfg['N']; POP = cfg['POP']; GEN = cfg['GEN']
    density = cfg['density']; max_steps = cfg['max_steps']
    delta_end = cfg.get('delta_end', 0.4)
    gate_start = cfg.get('gate_start', 0.3)
    gate_end = cfg.get('gate_end', 0.7)

    random.seed(42); torch.manual_seed(42)
    env = RaySnakeEnv(grid_size=10, max_steps=max_steps)
    pop = [Brain(N=N, obs_dim=32, act_dim=3, density=density) for _ in range(POP)]
    for b in pop: b.save_baseline()
    if use_gate:
        for b in pop: b.gate_active = True

    history = []
    t0 = time.perf_counter()
    for gen in range(GEN):
        tg = time.perf_counter()
        # 门控
        if use_gate:
            frac = gen / max(1, GEN-1)
            if frac < gate_start: delta = 0
            elif frac >= gate_end: delta = delta_end
            else: delta = delta_end * (frac - gate_start) / (gate_end - gate_start)
            if delta > 0:
                for b in pop: b.refresh_gate(delta)

        results = [fitness_fn(b, env, use_gate) for b in pop]
        te = time.perf_counter() - tg

        foods = [r[0] for r in results]
        best_idx = int(np.argmax(foods))
        bf = foods[best_idx]
        af = float(np.mean(foods))
        ar = int(pop[best_idx].M_rec.sum().item())
        history.append({'gen': gen, 'best_food': bf, 'avg_food': af, 'act_rec': ar, 'delta': delta if use_gate else 0})
        print(f"  Gen {gen+1:2d}/{GEN} | BestFood: {bf:.2f} | AvgFood: {af:.2f} | ActRec: {ar} | {te:.1f}s")

        if gen < GEN - 1:
            if fitness_fn == fitness_key_a:
                fitnesses = fitness_sort_key_a(results)
            else:
                fitnesses = fitness_sort_by_food(results)
            pop = evolve(pop, fitnesses, cfg['elite_size'])

    total = time.perf_counter() - t0
    bf0 = history[0]['best_food']; bf_end = history[-1]['best_food']
    af0 = history[0]['avg_food']; af_end = history[-1]['avg_food']
    best_bf = max(h['best_food'] for h in history)
    print(f"\n  汇总: BestFood {bf0:.2f}→{bf_end:.2f} (峰值{best_bf:.2f}) | AvgFood {af0:.2f}→{af_end:.2f} | {total:.0f}s")
    return history


def fitness_standard(brain, env, use_gate=False):
    """test4 标准 fitness: food + reward/100"""
    f, s, u = evaluate(brain, env, use_gate=use_gate)
    return (f, s, u)


def fitness_key_a(brain, env, use_gate=False):
    """test9a 排序键: (food, -seen, unseen)"""
    f, s, u = evaluate(brain, env, use_gate=use_gate)
    return (f, s, u)


def fitness_sort_by_food(results):
    """A 排序: 只按 food"""
    return [r[0] for r in results]


def fitness_sort_key_a(results):
    """C 排序: test9a 排序键，返回可比较的标量"""
    def key(r):
        f, s, u = r
        if f > 3.0: return (f * 1e6 + u * 1e3 - s)
        return (f * 1e6 - s * 1e3 + u)
    return [key(r) for r in results]


if __name__ == '__main__':
    cfg = {'N': 64, 'POP': 1024, 'GEN': 20, 'density': 0.15,
           'max_steps': 1000, 'elite_size': 64, 'delta_end': 0.4,
           'gate_start': 0.3, 'gate_end': 0.7}

    print("=== test4_32dim 消融: 隔离门控 vs 排序键 ===")
    print(f"N={cfg['N']}  POP={cfg['POP']}  GEN={cfg['GEN']}  32维环境")

    # A: 原始对照
    h_a = run_ablation("A: 原始 test4 fitness", cfg, fitness_standard, use_gate=False)

    # B: + influence 门控
    h_b = run_ablation("B: + influence 门控", cfg, fitness_standard, use_gate=True)

    # C: + test9a 排序键
    h_c = run_ablation("C: + test9a 排序键", cfg, fitness_key_a, use_gate=False)

    # 对比表
    print(f"\n{'='*60}")
    print(f"{'':>20s} {'gen0':>8s} {'gen10':>8s} {'gen19':>8s} {'峰值':>8s}")
    for name, h in [('A 原始', h_a), ('B +门控', h_b), ('C +排序键', h_c)]:
        bf = [x['best_food'] for x in h]
        print(f"  {name:>12s}  {bf[0]:>8.2f} {bf[9]:>8.2f} {bf[19]:>8.2f} {max(bf):>8.2f}")
