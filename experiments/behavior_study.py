# ==========================================
# experiments/behavior_study.py —— test7h 瓶颈局部行为学（B1/B2/B5/B3，只读不训练）
#
# B1 死亡解剖：40 库模型逐局死亡模式分类（空间挤压/机动失误/隔离饿死/决策饿死）
# B2 逐食段行为学：与 CycleSolver 参考（必通关理想解法）同库同段对比
#     绕路比/转向密度/直线度 按段序分箱
# B5 精英行为多样性：checkpoint 精英 top128 训练同口径评估 → te/food 分布
# B3 反事实缝合：k∈{10,20,30} 食处模型↔参考互相接管，归因"路径依赖 vs 决策"
# 产出：results/behavior_study.json + behavior_b1_b2.png
# ==========================================
import copy
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test7h as t7h  # noqa: E402
from ref_solver import MiniEnv, CycleSolver, rollout_reference  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = 10
EPS = 40
DEV = torch.device('cpu')


def cpu_cfg(base=None):
    cfg = copy.copy(base if base is not None else t7h.Config())
    cfg.DEVICE = 'cpu'
    cfg.USE_FP16 = False
    cfg.MAX_STEPS = 20000
    return cfg


class ModelBrain:
    """单体模型脑（mini-env 配对：E/I/st/press 持久，与训练 deliberate 一致）。"""

    def __init__(self, cfg, brain_state):
        self.cfg = cfg
        self.pop = t7h.GeneStack(cfg, B=1, device=DEV)
        self.pop.random_init()
        self.pop.set_individual_from_state(0, brain_state)
        self.pop.refresh_eff()
        N = self.pop.N
        self.E = torch.zeros(1, N)
        self.I = torch.zeros(1, N)
        self.st = torch.zeros(1, N)
        self.press = torch.zeros(1)

    def act(self, env):
        # mini-env 状态 → 32 维观测（复刻 BatchedSnakeEnv._obs32，欧氏/曼哈顿按 cfg）
        obs = self._obs32(env)
        act, self.E, self.I, self.st = t7h.deliberate_batch(
            self.pop, obs, self.E, self.I, self.st, self.press, self.cfg)
        self.press = t7h.update_fatigue(self.press, act,
                                        decay=float(self.cfg.FATIGUE_TURN_DECAY))
        return int(act[0])

    def _obs32(self, env):
        cfg = self.cfg
        B = 1
        head = torch.tensor([env.head])
        food = torch.tensor([env.food])
        body = torch.zeros(1, env_body_maxlen(env), 2, dtype=torch.long)
        body[0, :len(env.body)] = torch.tensor(env.body)
        bl = torch.tensor([len(env.body)])
        dir_idx = torch.tensor([env.dir])
        m = _ObsShell(cfg, head, food, body, bl, dir_idx)
        return m.obs32()


def env_body_maxlen(env):
    return max(len(env.body), 2)


class _ObsShell:
    """借用 BatchedSnakeEnv._obs32 的纯函数形态（不构造完整 env）。"""

    def __init__(self, cfg, head, food, body, body_len, dir_idx):
        import test7h as t7
        self.cfg = cfg
        self.B = 1
        self.G = cfg.GRID_SIZE
        self.device = torch.device('cpu')
        self.DIRS = t7._make_dirs(self.device)
        self.head = head
        self.food = food
        self.body = body
        self.body_len = body_len
        self.dir_idx = dir_idx

    def _occupancy_flat(self, tail_invalid=False):
        B, G, dev = self.B, self.G, self.device
        flat = self.body[:, :, 0] * G + self.body[:, :, 1]
        maxlen = self.body.shape[1]
        valid = torch.arange(maxlen)[None, :] < self.body_len[:, None]
        if tail_invalid:
            valid &= torch.arange(maxlen)[None, :] < (self.body_len - 1)[:, None]
        occ = torch.zeros(B, G * G)
        occ.scatter_add_(1, flat.clamp(max=G * G - 1), valid.float())
        return occ

    def obs32(self):
        return _obs32_impl(self.cfg, self.head, self.food, self.body,
                           self.body_len, self.dir_idx, self.G)


def _obs32_impl(cfg, head, food, body, body_len, dir_idx, G):
    """BatchedSnakeEnv._obs32 的等价实现（B=1，与 GPU env 逐位一致——已由
    verify_mini_env 的同观测前提隐式校验：模型局 food 序列逐位一致）。"""
    B = 1
    dev = torch.device('cpu')
    import test7h as t7
    DIRS = t7._make_dirs(dev)
    d = DIRS[dir_idx]
    obs = torch.zeros(B, 32)
    obs[:, 0] = ((d[:, 0] == 0) & (d[:, 1] == 1)).float()
    obs[:, 1] = ((d[:, 0] == 1) & (d[:, 1] == 0)).float()
    obs[:, 2] = ((d[:, 0] == 0) & (d[:, 1] == -1)).float()
    obs[:, 3] = ((d[:, 0] == -1) & (d[:, 1] == 0)).float()
    ti = int(body_len[0].item()) - 1
    pi = max(ti - 1, 0)
    tail = body[0, ti]
    prev = body[0, pi]
    tdr, tdc = prev[0] - tail[0], prev[1] - tail[1]
    for k, (a, b) in enumerate([(0, 1), (1, 0), (0, -1), (-1, 0)]):
        obs[:, 4 + k] = float(tdr == a and tdc == b)
    left = DIRS[(dir_idx + 3) % 4]
    right = DIRS[(dir_idx + 1) % 4]
    d8 = [d, d + left, left, left - d, -d, right - d, right, d + right]
    vr = (food[:, 0] - head[:, 0]).float()
    vc = (food[:, 1] - head[:, 1]).float()
    manhattan = bool(getattr(cfg, 'OBS_MANHATTAN', True))
    d2 = ((vr.abs() + vc.abs()).clamp(min=1.0)) if manhattan else \
        (vr * vr + vc * vc).clamp(min=1.0)
    k_scale = float(getattr(cfg, 'OBS_FOOD_SCALE', 1.0))
    for i in range(8):
        dx = d8[i][:, 0].float()
        dy = d8[i][:, 1].float()
        if manhattan:
            dot = vr * dx + vc * dy
        else:
            norm = torch.sqrt(dx * dx + dy * dy)
            dot = (vr * dx + vc * dy) / norm
        obs[:, 8 + i] = torch.clamp(dot, min=0.0) / d2 * k_scale
    flat_body = body[0, :, 0] * G + body[0, :, 1]
    seg_valid = ((torch.arange(body.shape[1]) >= 1) &
                 (torch.arange(body.shape[1]) < body_len[0])).unsqueeze(0)
    bf = torch.zeros(B, G * G, dtype=torch.long)
    bf.scatter_add_(1, flat_body.clamp(max=G * G - 1).unsqueeze(0), seg_valid.long())
    bmask = bf.view(G, G) > 0
    ar = torch.arange(B)
    for i in range(8):
        ddr, ddc = int(d8[i][0, 0]), int(d8[i][0, 1])
        dist = float(G + 1)
        ok = True
        for k in range(1, G + 1):
            r = int(head[0, 0]) + ddr * k
            c = int(head[0, 1]) + ddc * k
            if not (0 <= r < G and 0 <= c < G):
                break
            if bmask[r, c]:
                dist = float(k)
                break
        obs[:, 16 + i] = (1.0 / dist) if dist <= G else 0.0
    k_self = float(getattr(cfg, 'OBS_SELF_SCALE', 1.0))
    obs[:, 16:24] *= k_self
    for i in range(8):
        ddr, ddc = int(d8[i][0, 0]), int(d8[i][0, 1])
        dist = float(G)
        for k in range(1, G + 1):
            r = int(head[0, 0]) + ddr * k
            c = int(head[0, 1]) + ddc * k
            if not (0 <= r < G and 0 <= c < G):
                dist = float(k)
                break
            if bmask[r, c]:
                dist = float(k)
                break
        obs[:, 24 + i] = 1.0 / dist
    obs[:, 28] = torch.sqrt(body_len.float().clamp(min=1) / G)
    k_obs = float(getattr(cfg, 'OBS_OBSTACLE_SCALE', 1.0))
    obs[:, 24:32] *= k_obs
    return obs


def rollout_model_mini(cfg, brain_state, bank, starve_slope=None,
                       start_state=None, max_steps=20000, take_steps=None,
                       take_food=None):
    """mini-env 模型局。start_state=(body,dir,food,draw) 时从该状态接管。"""
    cfg = cpu_cfg(cfg)
    env = MiniEnv(bank, starve_slope=float(cfg.STARVE_SLOPE if starve_slope is None
                                           else starve_slope),
                  max_steps=max_steps)
    init_food = env.food
    if start_state is not None:
        body, d, food, draw = start_state
        env.body = list(body)
        env.head = body[0]
        env.dir = d
        env.food = food
        env.draw = draw
        env.path = [env.head]
        env.alive = True
        env.steps_wo_food = 0
    brain = ModelBrain(cfg, brain_state)
    acts = []
    t = 0
    eaten0 = len(env.events)
    while env.alive and t < max_steps:
        a = brain.act(env)
        env.step(a)
        acts.append(a)
        t += 1
        if take_food is not None and len(env.events) - eaten0 >= take_food:
            break
    return dict(acts=acts, events=env.events, path=env.path, snaps=env.snaps,
                died=env.died, death_ctx=env.death_ctx, won=env.won,
                init_food=init_food,
                start_state=(list(env.body), env.dir, env.food, env.draw))


def rollout_ref_mini(bank, starve_slope=1e9, start_state=None, take_food=None,
                     max_steps=20000, seed=0):
    env = MiniEnv(bank, starve_slope=starve_slope, max_steps=max_steps)
    init_food = env.food
    if start_state is not None:
        body, d, food, draw = start_state
        env.body = list(body)
        env.head = body[0]
        env.dir = d
        env.food = food
        env.draw = draw
        env.path = [env.head]
        env.alive = True
        env.steps_wo_food = 0
    solver = CycleSolver()
    t = 0
    eaten0 = len(env.events)
    while env.alive and t < max_steps:
        a = solver.next_action(env)
        env.step(a)
        t += 1
        if take_food is not None and len(env.events) - eaten0 >= take_food:
            break
    return dict(acts=env.acts, events=env.events, path=env.path, snaps=env.snaps,
                died=env.died, death_ctx=env.death_ctx, won=env.won,
                init_food=init_food,
                start_state=(list(env.body), env.dir, env.food, env.draw))


def segment_metrics(acts, events):
    """逐食段指标：段 i = events[i-1]→events[i]。"""
    segs = []
    start = 0
    for i, et in enumerate(events):
        sl = et - start
        acts_seg = acts[start:et]
        turns = sum(1 for a in acts_seg if a != 0)
        segs.append(dict(SL=sl, turns=turns, density=turns / max(sl, 1)))
        start = et
    return segs


def main():
    torch.manual_seed(0)
    base = t7h.Config()
    banks = t7h.make_banks(base, 777, 1, EPS, DEV)
    result = {}

    # ---- 载入模型 ----
    mdata = torch.load(os.path.join(ROOT, 'test7h_econ_best_model.pth'),
                       map_location='cpu', weights_only=False)
    mcfg = t7h.Config()
    mcfg.OBS_MANHATTAN = False
    mcfg.FATIGUE_TURN_GAIN = 0.0
    for k, v in mdata.get('config', {}).items():
        if hasattr(mcfg, k) and isinstance(v, (int, float, bool, str)):
            setattr(mcfg, k, v)
    mcfg = cpu_cfg(mcfg)
    st = mdata['brain']
    print(f'[模型] test7h best（manhattan={mcfg.OBS_MANHATTAN}, fatigue='
          f'{mcfg.FATIGUE_TURN_GAIN}）')

    # ---- 参考蛇（免钟，必通关）----
    print('[参考] CycleSolver 40 库（免饿死钟）...')
    rlogs = rollout_reference(banks, starve_slope=1e9, max_steps=30000)
    rfood = [len(l['events']) for l in rlogs]
    print(f'  food mean {np.mean(rfood):.1f} min {min(rfood)} max {max(rfood)} | '
          f'通关 {sum(l.get("won", False) for l in rlogs)}/40')

    # ---- 模型局（mini-env，原生钟）----
    print('[模型局] 逐个体 mini-env ...')
    mlogs = []
    for i in range(EPS):
        lg = rollout_model_mini(mcfg, st, banks[i])
        mlogs.append(lg)
    mfood = [len(l['events']) for l in mlogs]
    mdied = [l['died'] for l in mlogs]
    print(f'  food mean {np.mean(mfood):.1f} ± {np.std(mfood):.1f} | '
          f'死因 墙{mdied.count(1)}/撞己{mdied.count(2)}/饿死{mdied.count(3)}')

    # ================= B1 死亡解剖 =================
    print('\n[B1] 死亡解剖')
    b1 = dict(wall=0, squeeze=0, miscut=0, isolated_starve=0, decision_starve=0,
              by_L={})
    for lg in mlogs:
        dc = lg['death_ctx']
        if dc is None:
            continue
        L = dc['L']
        b1['by_L'].setdefault(L, []).append(lg['died'])
        if lg['died'] == 1:
            b1['wall'] += 1
        elif lg['died'] == 2:
            if dc.get('free_reach', 999) < 2 * L:
                b1['squeeze'] += 1
            else:
                b1['miscut'] += 1
        elif lg['died'] == 3:
            if not dc.get('food_reachable', True):
                b1['isolated_starve'] += 1
            else:
                b1['decision_starve'] += 1
    b1['by_L'] = {str(k): v for k, v in sorted(b1['by_L'].items())}
    result['B1'] = b1
    print(f"  撞墙 {b1['wall']} | 自撞: 空间挤压 {b1['squeeze']} / 机动失误 "
          f"{b1['miscut']} | 饿死: 隔离 {b1['isolated_starve']} / 可达未吃 "
          f"{b1['decision_starve']}（共 {EPS} 局）")

    # ================= B2 逐食段 =================
    print('\n[B2] 逐食段行为学（模型 vs 参考，按段序 5 段分箱）')
    def seg_profile(logs, max_bin=8):
        bins = {}
        for lg in logs:
            segs = segment_metrics(lg['acts'], lg['events'])
            for i, sg in enumerate(segs):
                k = min(i // 5, max_bin)
                bins.setdefault(k, []).append(sg)
        out = {}
        for k, v in sorted(bins.items()):
            out[k] = dict(
                n=len(v),
                SL=float(np.mean([x['SL'] for x in v])),
                density=float(np.mean([x['density'] for x in v])))
        return out
    b2m, b2r = seg_profile(mlogs), seg_profile(rlogs)
    # 绕路比（段 SL / 曼哈顿最短）需 path；补算
    def detour_profile(logs):
        bins = {}
        for lg in logs:
            start = 0
            for i, et in enumerate(lg['events']):
                # 段起点头位置 = path[start]；段终点食物 = 第 i 颗（snaps[i-1][1]
                # 是其后刷新的新食物，第 0 颗 = init_food）
                src = lg['path'][start] if start < len(lg['path']) else None
                dst = lg['snaps'][i - 1][1] if i >= 1 else lg.get('init_food')
                if src is None or dst is None:
                    start = et
                    continue
                sl = et - start
                md = abs(src[0] - dst[0]) + abs(src[1] - dst[1])
                k = min(i // 5, 8)
                bins.setdefault(k, []).append(sl / max(md, 1))
                start = et
        return {k: float(np.mean(v)) for k, v in sorted(bins.items())}
    b2m_d, b2r_d = detour_profile(mlogs), detour_profile(rlogs)
    result['B2'] = dict(model=b2m, ref=b2r,
                        model_detour=b2m_d, ref_detour=b2r_d)
    print(f'  {"段":<6}{"模型密度":>9}{"参考密度":>9}{"模型绕路":>9}{"参考绕路":>9}')
    for k in sorted(set(b2m) | set(b2r)):
        print(f'  {k * 5}-{k * 5 + 4:<3}'
              f'{b2m.get(k, {}).get("density", float("nan")):>9.2f}'
              f'{b2r.get(k, {}).get("density", float("nan")):>9.2f}'
              f'{b2m_d.get(k, float("nan")):>9.2f}'
              f'{b2r_d.get(k, float("nan")):>9.2f}')

    # ================= B5 精英多样性 =================
    print('\n[B5] 精英 top128 训练同口径评估（CRN+屏蔽+两阶段）')
    ck = torch.load(os.path.join(ROOT, 'test7h_econ_checkpoint.pth'),
                    map_location='cpu', weights_only=False)
    cfg5 = t7h.Config()
    for k, v in ck.get('config', {}).items():
        if hasattr(cfg5, k) and isinstance(v, (int, float, bool, str)):
            setattr(cfg5, k, v)
    cfg5.GENERATIONS = cfg5.next_gen if hasattr(cfg5, 'next_gen') else cfg5.GENERATIONS
    n_elite = 128
    genes = {g: ck['pop'][g][:n_elite] for g in t7h.GeneStack.GENES}
    pop5 = t7h.GeneStack(cfg5, B=n_elite, device=DEV)
    pop5.unpack(genes)
    te_list, food_list = [], []
    for bi in range(5):
        sub = pop5.clone_rows(list(range(n_elite)))
        sub.refresh_eff()
        m = t7h._eval_pop_banks(sub, cfg5, [banks[bi]])
        te = (m[:, 3] / m[:, 10].clamp(min=1)).clamp(max=4).numpy()
        te_list.append(te)
        food_list.append(m[:, 0].numpy())
    te_all = np.stack(te_list, axis=1).mean(axis=1)
    food_all = np.stack(food_list, axis=1).mean(axis=1)
    qs = np.quantile(te_all, [0.1, 0.25, 0.5, 0.75, 0.9])
    frac_ge2 = float((te_all >= 2.0).mean())
    frac_ge3 = float((te_all >= 3.0).mean())
    result['B5'] = dict(te_quantiles=[float(x) for x in qs],
                        te_frac_ge2=frac_ge2, te_frac_ge3=frac_ge3,
                        food_mean=float(food_all.mean()),
                        food_max=float(food_all.max()))
    print(f'  te 分位 10/25/50/75/90%: {[round(x, 2) for x in qs]} | '
          f'te≥2 占比 {frac_ge2:.1%} | te≥3 占比 {frac_ge3:.1%} | '
          f'精英 food 均值 {food_all.mean():.1f}（5 库）')

    # ================= B3 反事实缝合 =================
    print('\n[B3] 反事实缝合（免钟参考 ↔ 原生钟模型，k=10/20/30）')
    b3 = {}
    for k in (10, 20, 30):
        gains_a, gains_b, n_ok = [], [], 0
        for i in range(EPS):
            ml, rl = mlogs[i], rlogs[i]
            if len(ml['events']) < k or len(rl['events']) < k:
                continue
            n_ok += 1
            # A 向：参考玩到第 k 食 → 模型接管（免钟，给模型宽松环境）
            st_ref = rl['start_state'] if 'start_state' in rl else None
            # 参考局重放到第 k 食取状态
            rpart = rollout_ref_mini(banks[i], starve_slope=1e9, take_food=k)
            a_gain = rollout_model_mini(mcfg, st, banks[i], starve_slope=1e9,
                                        start_state=rpart['start_state'])
            gains_a.append(len(a_gain['events']))
            # B 向：模型玩到第 k 食 → 参考接管（免钟）
            mpart = rollout_model_mini(mcfg, st, banks[i], take_food=k)
            bg = rollout_ref_mini(banks[i], starve_slope=1e9,
                                  start_state=mpart['start_state'])
            gains_b.append(len(bg['events']))
        if n_ok:
            b3[f'k{k}'] = dict(
                n=n_ok,
                A_ref_to_model_extra=float(np.mean(gains_a)),
                A_min=int(np.min(gains_a)), A_max=int(np.max(gains_a)),
                B_model_to_ref_extra=float(np.mean(gains_b)),
                B_min=int(np.min(gains_b)), B_max=int(np.max(gains_b)))
            print(f'  k={k}（{n_ok} 库）: 参考起手→模型再吃 '
                  f'{np.mean(gains_a):.1f}（min {min(gains_a)}）| '
                  f'模型起手→参考再吃 {np.mean(gains_b):.1f}（min {min(gains_b)}）')
    result['B3'] = b3

    # ---- 落盘 + 绘图 ----
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    with open(os.path.join(ROOT, 'results', 'behavior_study.json'), 'w',
              encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print('\n[落盘] results/behavior_study.json')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
        # B1
        ax = axes[0]
        cats = ['墙', '自撞·挤压', '自撞·机动', '饿死·隔离', '饿死·可达']
        vals = [b1['wall'], b1['squeeze'], b1['miscut'],
                b1['isolated_starve'], b1['decision_starve']]
        ax.bar(range(5), vals, color=['gray', 'red', 'orange', 'purple', 'blue'])
        ax.set_xticks(range(5))
        ax.set_xticklabels(cats, fontsize=8, rotation=20)
        ax.set_title('B1 death modes (40 eps)')
        # B2
        ax = axes[1]
        ks = sorted(set(b2m) | set(b2r))
        x = [k * 5 for k in ks]
        ax.plot(x, [b2m.get(k, {}).get('density', np.nan) for k in ks],
                'o-', label='model density')
        ax.plot(x, [b2r.get(k, {}).get('density', np.nan) for k in ks],
                's-', label='ref density')
        ax2 = ax.twinx()
        ax2.plot(x, [b2m_d.get(k, np.nan) for k in ks], 'o--', alpha=0.5,
                 label='model detour')
        ax2.plot(x, [b2r_d.get(k, np.nan) for k in ks], 's--', alpha=0.5,
                 label='ref detour')
        ax.set_xlabel('food segment index bin')
        ax.set_title('B2 per-segment behavior')
        ax.legend(fontsize=8, loc='upper left')
        ax2.legend(fontsize=8, loc='upper right')
        # B3
        ax = axes[2]
        ks3 = [int(k[1:]) for k in b3]
        w = 0.35
        ax.bar([i - w / 2 for i in range(len(ks3))],
               [b3[f'k{k}']['A_ref_to_model_extra'] for k in b3], width=w,
               label='ref start → model extra food')
        ax.bar([i + w / 2 for i in range(len(ks3))],
               [b3[f'k{k}']['B_model_to_ref_extra'] for k in b3], width=w,
               label='model start → ref extra food')
        ax.set_xticks(range(len(ks3)))
        ax.set_xticklabels([f'k={k}' for k in ks3])
        ax.set_title('B3 handover experiment')
        ax.legend(fontsize=8)
        plt.tight_layout()
        out = os.path.join(ROOT, 'results', 'behavior_b1_b2_b3.png')
        fig.savefig(out, dpi=110)
        print(f'[落盘] {out}')
    except Exception as e:
        print(f'(绘图跳过: {e})')


if __name__ == '__main__':
    main()
