# ==========================================
# experiments/diagnose_test7g_plateau.py —— test7g 40 分平台只读诊断（不训练）
#
# 背景：7g 二阶段（三项和式适应度）从 20 代种子续训 100 代，30 代后平台 ~40，
# 且蛇复现 7b 式摇头（斜线/方波线）导致空间破碎。本脚本只读现有模型做克隆评估：
#   P1 摇头定位：转向密度按蛇长段（<15 / 15-30 / >30 / 末食后）+ 直行游程直方图
#      → 区分"长蛇段/末食后漂变"（嫌疑1：压力真空）vs 全程性回退
#   P2 reach 判别力：吃食序号-reach 曲线（验证前 ~25 次饱和稀释）；
#      适应度口径反演：现状 / 无reach / 仅序号>25 / 全程采样 四口径的 Kendall-τ
#      与"少食高reach 压过多食低reach"反转实例 → 验证嫌疑2（兑换率退化）
#   P3 噪声地板：克隆 40 局 food 的 mean/σ/SEM（对照平台区边际信号 ~1.2 分）；
#      checkpoint 精英 top64 × 10 局：个体间 vs 个体内方差分解
#   P4 死因/效率/口袋食物：死因分布、steps_last、食物落点连通域
#      （蛇长 40+ 时食物落入口袋的概率与"吃口袋拉低 avg_reach"悖论）
#
# 对象：test7g_econ_best_model.pth   二阶段续训 best（40 平台产物）
#       test7g_econ_best_100gen.pth  一阶段 economy best（30 平台产物，对照）
#       test7b_best_model.pth        7b 67 分模型（"摇头但高分"基线，对照）
#       test7g_econ_checkpoint copy.pth  读 next_gen/config 确认续训参数 + 精英 top64
# 产出：results/test7g_plateau_diagnosis.json + .png
# ==========================================
import argparse
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments', 'test7_series'))  # 归档后模块路径
import test7g as t7  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 模型保存 config 中与评估相关的标量键（缺失时用 test7g 默认值）
_EVAL_KEYS = [
    'GRID_SIZE', 'NUM_COLUMNS', 'OBS_DIM', 'ACTION_DIM', 'OBS_MODE',
    'OBS_MANHATTAN', 'OBS_FOOD_SCALE', 'OBS_SELF_SCALE', 'OBS_OBSTACLE_SCALE',
    'FATIGUE_TURN_GAIN', 'FATIGUE_TURN_DECAY', 'FRAME_RATE', 'INPUT_DECAY',
    'SHORT_TERM_GAIN', 'SHORT_TERM_DECAY', 'BASE_TAU_E', 'TAU_E_NOISE',
    'TAU_E_MIN', 'TAU_E_MAX', 'W_EI', 'W_IE', 'STARVE_SLOPE',
    'USE_FP16', 'MAX_STEPS', 'REACH_W', 'FOOD_EFF_WEIGHT',
]


def build_cfg(saved_cfg):
    """从模型保存的 config 构造评估配置。

    关键：OBS_MANHATTAN / FATIGUE_TURN_GAIN 是 7g 才有的键，pre-7g 模型
    （如 7b）保存的 config 里没有——默认回退 False / 0.0（7b 旧疲劳对
    交替摆动不触发，等效 0），否则会把 7b 脑错误地放进曼哈顿观测里评估。
    """
    cfg = t7.Config()
    cfg.OBS_MANHATTAN = False
    cfg.FATIGUE_TURN_GAIN = 0.0
    for k in _EVAL_KEYS:
        if k in saved_cfg:
            setattr(cfg, k, saved_cfg[k])
    return cfg


def load_model(path):
    data = torch.load(path, map_location='cpu', weights_only=False)
    return data.get('brain'), data.get('config', {}), data


def make_pop(cfg, st, B, dev):
    pop = t7.GeneStack(cfg, B=B, device=dev)
    pop.random_init()
    for i in range(B):
        pop.set_individual_from_state(i, st)
    if cfg.USE_FP16:
        pop.fp16()
    pop.refresh_eff()
    return pop


# ---------- BFS 助手（与训练 reach_ratio 同语义：尾格可走、十字 4 连通）----------
def _free_grid(env):
    occ = env._occupancy_flat(tail_invalid=True).view(env.B, 1, env.G, env.G)
    return occ < 0.5


def _bfs_mask(env, free, seeds):
    """从 seeds[B,2] 出发的自由格连通域掩码（种子若被占则自身不计入）。"""
    B, G, dev = env.B, env.G, env.device
    reach = torch.zeros(B, 1, G, G, device=dev)
    reach[torch.arange(B, device=dev), 0, seeds[:, 0], seeds[:, 1]] = 1.0
    mp = torch.nn.functional.max_pool2d
    for _ in range(G * G):
        cross = torch.maximum(mp(reach, (3, 1), stride=1, padding=(1, 0)),
                              mp(reach, (1, 3), stride=1, padding=(0, 1)))
        reach = cross * free
    return reach


def reach_and_pocket(env):
    """一次返回：头部可达域占比 / 新食物是否在头部连通域 / 食物连通域占比。"""
    free = _free_grid(env)
    free_cnt = free.sum(dim=(1, 2, 3)).float().clamp(min=1.0)
    head_m = _bfs_mask(env, free, env.head)
    food_m = _bfs_mask(env, free, env.food)
    reach = (head_m > 0).float().sum(dim=(1, 2, 3)) / free_cnt
    # 食物格自由 ⇒ 在头部连通域内 ⇔ 头部掩码覆盖食物格（比计数相等更稳）
    ar = torch.arange(env.B, device=env.device)
    head_has_food = head_m[ar, 0, env.food[:, 0], env.food[:, 1]] > 0
    pocket = (food_m > 0).float().sum(dim=(1, 2, 3)) / free_cnt
    return reach, head_has_food, pocket


# ---------- 轨迹采集滚动评估 ----------
def rollout(cfg, pop, episodes, reach_every=10, verbose_tag=''):
    """克隆评估 episodes 局，逐局逐个体记录轨迹统计。

    返回 rows: 每局每个体一条 dict（food/steps_last/died/分段转向/吃食明细…），
    以及 pooled 的直行游程 Counter 与吃食序号-reach 明细。
    """
    dev = pop.device
    B = pop.P
    half = torch.float16 if cfg.USE_FP16 else torch.float32
    env = t7.BatchedSnakeEnv(cfg, B, dev)
    rows = []
    run_hist = Counter()
    eat_detail = []  # (tag_row, ordinal, reach, head_reaches, pocket_ratio)

    for ep in range(episodes):
        env.reset()
        E = torch.zeros(B, pop.N, dtype=half, device=dev)
        I = torch.zeros(B, pop.N, dtype=half, device=dev)
        st = torch.zeros(B, pop.N, dtype=half, device=dev)
        press = torch.zeros(B, dtype=torch.float32, device=dev)

        acts, lens, alives = [], [], []
        ordinal = np.zeros(B, dtype=np.int64)
        last_eat_t = np.full(B, -1, dtype=np.int64)
        eats = [[] for _ in range(B)]  # 每个体 [(ordinal, reach, head_reach, pocket)]
        reach_cont_sum = torch.zeros(B, dtype=torch.float64, device=dev)
        reach_cont_n = torch.zeros(B, dtype=torch.float64, device=dev)

        t = -1
        for t in range(cfg.MAX_STEPS):
            al = env.alive
            if not bool(al.any()):
                break
            obs = env.obs().to(half)
            act, E, I, st = t7.deliberate_batch(pop, obs, E, I, st, press, cfg)
            press = t7.update_fatigue(press, act, decay=float(cfg.FATIGUE_TURN_DECAY))
            env.step(act)
            ate = al & env.ate
            acts.append(act.tolist())
            lens.append(env.body_len.tolist())
            alives.append(al.tolist())
            if ate.any():
                rr, same, pocket = reach_and_pocket(env)
                ate_idx = ate.nonzero(as_tuple=True)[0].tolist()
                rr_l, same_l, pk_l = rr.tolist(), same.tolist(), pocket.tolist()
                for b in ate_idx:
                    ordinal[b] += 1
                    last_eat_t[b] = t
                    eats[b].append((ordinal[b], rr_l[b], bool(same_l[b]), pk_l[b]))
            if t % reach_every == 0:
                free = _free_grid(env)
                free_cnt = free.sum(dim=(1, 2, 3)).float().clamp(min=1.0)
                hm = _bfs_mask(env, free, env.head)
                rr = (hm > 0).float().sum(dim=(1, 2, 3)) / free_cnt
                af = al.float().to(torch.float64)
                reach_cont_sum += rr.to(torch.float64) * af
                reach_cont_n += af

        T = len(acts)  # 循环可能在顶部 alive 检查处 break，t+1 会多记 1 步
        if T == 0:
            for b in range(B):
                rows.append(dict(food=0, steps=0, steps_last=0, died=0,
                                 turns_total=0, steps_seg=[0, 0, 0, 0],
                                 turns_seg=[0, 0, 0, 0], runs_le2_frac=0.0,
                                 runs_median=0.0, reach_eat=0.0, reach25=None,
                                 reach_cont=0.0, eats_eat=[], eats_same=[],
                                 eats_pocket=[]))
            continue
        acts_np = np.array(acts, dtype=np.int64)
        lens_np = np.array(lens, dtype=np.int64)
        alives_np = np.array(alives, dtype=bool)

        for b in range(B):
            a = acts_np[:, b]
            ln = lens_np[:, b]
            av = alives_np[:, b]
            le = int(last_eat_t[b])
            # 分段：末食后（post）优先，其余按蛇长 <15 / 15-30 / >30
            seg = np.where(ln < 15, 0, np.where(ln < 30, 1, 2))
            post = np.arange(T) > le
            steps_seg = [int(((seg == s) & av & ~post).sum()) for s in range(3)]
            turns_seg = [int(((a != 0) & (seg == s) & av & ~post).sum()) for s in range(3)]
            steps_post = int((post & av).sum())
            turns_post = int(((a != 0) & post & av).sum())
            # 直行游程（存活步内，转向/死亡截断）
            runs = []
            run = 0
            for tt in range(T):
                if not av[tt]:
                    if run:
                        runs.append(run)
                    run = 0
                    continue
                if a[tt] == 0:
                    run += 1
                else:
                    if run:
                        runs.append(run)
                    run = 0
            if run:
                runs.append(run)
            run_hist.update(runs)
            food = len(eats[b])
            steps_last = (le + 1) if le >= 0 else 0
            reach_eat = float(np.mean([e[1] for e in eats[b]])) if eats[b] else 0.0
            reach25 = ([e[1] for e in eats[b] if e[0] > 25]
                       if any(e[0] > 25 for e in eats[b]) else None)
            rows.append(dict(
                food=food, steps=int(av.sum()), steps_last=steps_last,
                died=int(env.died[b].item()),
                turns_total=int((a != 0)[av].sum()),
                steps_seg=steps_seg + [steps_post],
                turns_seg=turns_seg + [turns_post],
                runs_le2_frac=float(np.mean([r <= 2 for r in runs])) if runs else 0.0,
                runs_median=float(np.median(runs)) if runs else 0.0,
                reach_eat=reach_eat,
                reach25=(float(np.mean(reach25)) if reach25 else None),
                reach_cont=float((reach_cont_sum[b] / max(reach_cont_n[b].item(), 1.0)).item()),
                eats_eat=[e[1] for e in eats[b]],
                eats_same=[e[2] for e in eats[b]],
                eats_pocket=[e[3] for e in eats[b]],
            ))
            eat_detail.extend(
                (len(rows) - 1, e[0], e[1], e[2], e[3]) for e in eats[b])
        if verbose_tag:
            print(f"  [{verbose_tag}] episode {ep + 1}/{episodes} 完成 "
                  f"({time.perf_counter() - _T0:.0f}s)")
    return rows, run_hist, eat_detail


# ---------- 适应度口径 ----------
def fitness_variants(row_or_mean, reach_w=0.25, eff_w=0.3):
    """返回四种口径的适应度（输入含 food/steps_last/reach_eat/reach25/reach_cont）。"""
    food = row_or_mean['food']
    eff = food / max(row_or_mean['steps_last'], 1.0)
    base = food + eff_w * eff
    r25 = row_or_mean.get('reach25')
    return {
        'cur': base + reach_w * food * row_or_mean['reach_eat'],
        'noreach': base,
        'reach25': base + (reach_w * food * r25 if r25 is not None else 0.0),
        'cont': base + reach_w * food * row_or_mean['reach_cont'],
    }


def kendall(a, b):
    n = len(a)
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0:
                c += 1
            elif s < 0:
                d += 1
    return (c - d) / max(c + d, 1)


def summarize_rows(rows):
    food = np.array([r['food'] for r in rows], dtype=float)
    seg_steps = np.sum([r['steps_seg'] for r in rows], axis=0)
    seg_turns = np.sum([r['turns_seg'] for r in rows], axis=0)
    dens = seg_turns / np.maximum(seg_steps, 1)
    pocket_same = [s for r in rows for s in r['eats_same']]
    pocket_v = [p for r in rows for p in r['eats_pocket']]
    died = Counter(r['died'] for r in rows)
    return dict(
        n=len(rows), food_mean=float(food.mean()), food_std=float(food.std()),
        food_sem=float(food.std() / math.sqrt(len(rows))),
        food_max=int(food.max()), food_min=int(food.min()),
        turn_density=dict(early=float(dens[0]), mid=float(dens[1]),
                          late=float(dens[2]), post=float(dens[3])),
        straight_run_le2_frac=float(np.mean([r['runs_le2_frac'] for r in rows])),
        straight_run_median=float(np.median([r['runs_median'] for r in rows])),
        reach_eat_mean=float(np.mean([r['reach_eat'] for r in rows])),
        reach_cont_mean=float(np.mean([r['reach_cont'] for r in rows])),
        pocket_spawn_unreachable_frac=float(np.mean([not s for s in pocket_same]))
        if pocket_same else 0.0,
        pocket_ratio_mean=float(np.mean(pocket_v)) if pocket_v else 0.0,
        died={str(k): int(v) for k, v in died.items()},
        steps_last_mean=float(np.mean([r['steps_last'] for r in rows])),
    )


def eat_ordinal_curve(eat_detail, bin_w=5, max_ord=45):
    """吃食序号 → (reach 均值, 口袋占比, 不可达占比) 分箱曲线。"""
    bins = {}
    for _, o, rr, same, pk in eat_detail:
        k = min((o - 1) // bin_w, max_ord // bin_w)
        bins.setdefault(k, []).append((rr, same, pk))
    out = []
    for k in sorted(bins):
        v = bins[k]
        out.append(dict(
            bin=f'{k * bin_w + 1}-{(k + 1) * bin_w}',
            n=len(v),
            reach=float(np.mean([x[0] for x in v])),
            unreachable_frac=float(np.mean([not x[1] for x in v])),
            pocket=float(np.mean([x[2] for x in v])),
        ))
    return out


# ---------- 主流程 ----------
_T0 = time.perf_counter()


def main():
    ap = argparse.ArgumentParser(description='test7g 40 分平台只读诊断')
    ap.add_argument('--episodes', type=int, default=40, help='克隆评估局数')
    ap.add_argument('--elite-n', type=int, default=64, help='checkpoint 精英评估数')
    ap.add_argument('--elite-episodes', type=int, default=10)
    ap.add_argument('--reach-every', type=int, default=10, help='全程 reach 采样间隔步数')
    ap.add_argument('--device', type=str, default='auto')
    ap.add_argument('--skip-elite', action='store_true')
    args = ap.parse_args()

    torch.manual_seed(20260827)
    dev = torch.device(args.device if args.device != 'auto'
                       else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'[诊断] device = {dev}')

    result = {}

    # ---- 0. checkpoint 参数确认 ----
    ck_path = os.path.join(ROOT, 'test7g_econ_checkpoint copy.pth')
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location='cpu', weights_only=False)
        hist = ck.get('history', {})
        cfgk = ck.get('config', {})
        result['checkpoint'] = dict(
            next_gen=int(ck.get('next_gen', -1)),
            hist_len=len(hist.get('gen', [])),
            best_food=float(ck.get('best_food', -1)),
            tail_best_food=[round(x, 1) for x in hist.get('best_food', [])[-12:]],
            tail_avg_food=[round(x, 2) for x in hist.get('avg_food', [])[-12:]],
            config={k: cfgk.get(k) for k in
                    ['FIT_MODE', 'FATIGUE_TURN_GAIN', 'REACH_W', 'FOOD_EFF_WEIGHT',
                     'TURN_COST', 'OBS_MANHATTAN', 'GENERATIONS', 'POP_SIZE',
                     'EVAL_EPISODES', 'SEED_FROM_BEST', 'STARVE_SLOPE']},
            saved_at=ck.get('saved_at'),
        )
        print('[checkpoint copy]', json.dumps(result['checkpoint'], ensure_ascii=False,
                                              indent=1, default=str))
    else:
        result['checkpoint'] = None
        print('[checkpoint copy] 不存在，跳过')

    # ---- 1. 三个 best 模型克隆评估 ----
    models = [
        ('7g_phase2_best(40平台)', 'test7g_econ_best_model.pth'),
        ('7g_phase1_econ(30平台)', 'test7g_econ_best_100gen.pth'),
        ('7b_67分基线', 'test7b_best_model.pth'),
    ]
    per_model_rows = {}
    for tag, fn in models:
        path = os.path.join(ROOT, fn)
        if not os.path.exists(path):
            print(f'[跳过] {fn} 不存在')
            continue
        st, scfg, meta = load_model(path)
        if st is None:
            print(f'[跳过] {fn} 无 brain')
            continue
        cfg = build_cfg(scfg)
        cfg.MAX_STEPS = min(cfg.MAX_STEPS, 20000)
        cfg.DEVICE = str(dev)
        if dev.type != 'cuda':
            cfg.USE_FP16 = False
        pop = make_pop(cfg, st, B=args.episodes, dev=dev)
        print(f'\n[评估] {tag} <- {fn} | obs_manhattan={cfg.OBS_MANHATTAN} '
              f'fatigue={cfg.FATIGUE_TURN_GAIN} food_scale={cfg.OBS_FOOD_SCALE}')
        # B=episodes 个克隆并行跑 1 局 = episodes 局独立样本
        rows, run_hist, eat_detail = rollout(cfg, pop, episodes=1,
                                             reach_every=args.reach_every,
                                             verbose_tag=tag)
        per_model_rows[tag] = rows
        result.setdefault('models', {})[tag] = dict(
            file=fn,
            saved_food=float(meta.get('food', -1)),
            summary=summarize_rows(rows),
            eat_ordinal_curve=eat_ordinal_curve(eat_detail),
            run_hist={str(k): int(v) for k, v in sorted(run_hist.items()) if v >= 3},
            # 每局四口径适应度 → 口径间 Kendall-τ（局作为候选）
        )
        foods = [r['food'] for r in rows]
        fv = [fitness_variants(r) for r in rows]
        keys = list(fv[0].keys())
        taus = {}
        for k in keys:
            taus[f'{k}_vs_food'] = kendall([r['food'] for r in rows],
                                           [f[k] for f in fv])
        result['models'][tag]['fitness_tau_vs_food'] = taus
        # 反转局例：food 差 >=2 但 cur 口径次序反转
        inv = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                df = rows[i]['food'] - rows[j]['food']
                if df >= 2 and fv[i]['cur'] < fv[j]['cur']:
                    inv.append((rows[i]['food'], rows[j]['food'],
                                round(fv[i]['cur'], 2), round(fv[j]['cur'], 2),
                                round(rows[i]['reach_eat'], 3), round(rows[j]['reach_eat'], 3)))
        inv.sort(key=lambda x: -(x[0] - x[1]))
        result['models'][tag]['inversions_top5'] = inv[:5]
        result['models'][tag]['inversions_count'] = len(inv)
        s = result['models'][tag]['summary']
        print(f"  food {s['food_mean']:.1f} ± {s['food_std']:.1f} (SEM {s['food_sem']:.2f}) "
              f"max {s['food_max']} | 转向密度 早{s['turn_density']['early']:.2f} "
              f"中{s['turn_density']['mid']:.2f} 晚{s['turn_density']['late']:.2f} "
              f"末食后{s['turn_density']['post']:.2f} | 游程≤2比例 "
              f"{s['straight_run_le2_frac']:.2f} | reach(食时){s['reach_eat_mean']:.3f} "
              f"(全程){s['reach_cont_mean']:.3f} | 食物落点不可达 "
              f"{s['pocket_spawn_unreachable_frac']:.2f} | 反转局例 {len(inv)}")

    # ---- 2. checkpoint 精英 top64（按保存顺序即上代适应度降序）----
    if not args.skip_elite and os.path.exists(ck_path) and ck is not None:
        scfg = ck.get('config', {})
        cfg = build_cfg(scfg)
        cfg.MAX_STEPS = min(cfg.MAX_STEPS, 20000)
        cfg.DEVICE = str(dev)
        if dev.type != 'cuda':
            cfg.USE_FP16 = False
        n = min(args.elite_n, ck['pop']['W_in'].shape[0])
        genes = {g: ck['pop'][g][:n] for g in t7.GeneStack.GENES}
        pop = t7.GeneStack(cfg, B=n, device=dev)
        pop.unpack(genes)
        if not cfg.USE_FP16:
            pop.fp32()  # checkpoint 种群按 half 打包，fp32 评估需整体转回
        pop.refresh_eff()
        print(f"\n[精英评估] checkpoint 前 {n} 行（=上代适应度降序精英）× "
              f"{args.elite_episodes} 局")
        rows, run_hist, eat_detail = rollout(cfg, pop, episodes=args.elite_episodes,
                                             reach_every=args.reach_every,
                                             verbose_tag='elite')
        # 按（个体 = 行号）聚合：rows 顺序为 ep*n + b
        elites = []
        for b in range(n):
            sub = [rows[ep * n + b] for ep in range(args.elite_episodes)]
            m = dict(
                food=float(np.mean([r['food'] for r in sub])),
                steps_last=float(np.mean([r['steps_last'] for r in sub])),
                reach_eat=float(np.mean([r['reach_eat'] for r in sub])),
                reach25=(float(np.mean([r['reach25'] for r in sub if r['reach25'] is not None]))
                         if any(r['reach25'] is not None for r in sub) else None),
                reach_cont=float(np.mean([r['reach_cont'] for r in sub])),
                food_std=float(np.std([r['food'] for r in sub])),
            )
            elites.append(m)
        foods = [e['food'] for e in elites]
        fv = [fitness_variants(e) for e in elites]
        taus = {f'{k}_vs_food': kendall(foods, [f[k] for f in fv])
                for k in fv[0]}
        inv = []
        for i in range(n):
            for j in range(i + 1, n):
                df = elites[i]['food'] - elites[j]['food']
                if df >= 1.0 and fv[i]['cur'] < fv[j]['cur']:
                    inv.append((round(elites[i]['food'], 1), round(elites[j]['food'], 1),
                                round(fv[i]['cur'], 2), round(fv[j]['cur'], 2),
                                round(elites[i]['reach_eat'], 3), round(elites[j]['reach_eat'], 3)))
        inv.sort(key=lambda x: -(x[0] - x[1]))
        result['elites'] = dict(
            n=n, episodes=args.elite_episodes,
            food_mean=float(np.mean(foods)), food_std_between=float(np.std(foods)),
            food_std_within_mean=float(np.mean([e['food_std'] for e in elites])),
            food_sem_within=float(np.mean([e['food_std'] for e in elites])
                                  / math.sqrt(args.elite_episodes)),
            top8_food=[round(f, 1) for f in foods[:8]],
            top8_reach=[round(e['reach_eat'], 3) for e in elites[:8]],
            fitness_tau_vs_food=taus,
            inversions_count=len(inv), inversions_top5=inv[:5],
        )
        e = result['elites']
        print(f"  精英间 food σ {e['food_std_between']:.2f} vs 个体内 SEM "
              f"{e['food_sem_within']:.2f} | τ(cur,food)="
              f"{taus['cur_vs_food']:.3f} | 食物差≥1 被 reach 反转对 {len(inv)}")
        print(f"  top8 food: {e['top8_food']} | top8 reach(食时): {e['top8_reach']}")

    # ---- 3. 落盘 ----
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    out_json = os.path.join(ROOT, 'results', 'test7g_plateau_diagnosis.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1, default=str)
    print(f'\n[落盘] {out_json}')

    # ---- 4. 绘图 ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(18, 9))
        tags = [t for t, _ in models if t in result.get('models', {})]
        # (1) 分段转向密度
        ax = axes[0][0]
        ph = ['early', 'mid', 'late', 'post']
        w = 0.8 / max(len(tags), 1)
        for i, tg in enumerate(tags):
            d = result['models'][tg]['summary']['turn_density']
            ax.bar([x + i * w for x in range(4)],
                   [d[p] for p in ph], width=w, label=tg)
        ax.set_xticks(range(4))
        ax.set_xticklabels(['len<15', '15-30', '>30', 'post-last-food'])
        ax.set_ylabel('turn density')
        ax.set_title('P1 turn density by phase')
        ax.legend(fontsize=8)
        # (2) 直行游程分布
        ax = axes[0][1]
        for tg in tags:
            rh = result['models'][tg]['run_hist']
            xs = sorted(int(k) for k in rh)
            ys = [rh[str(x)] for x in xs]
            ax.plot(xs, ys, marker='.', label=tg)
        ax.set_yscale('log')
        ax.set_xlabel('straight run length (cells)')
        ax.set_ylabel('count')
        ax.set_title('P1 straight-run histogram')
        ax.legend(fontsize=8)
        # (3) reach 随吃食序号
        ax = axes[0][2]
        for tg in tags:
            cur = result['models'][tg]['eat_ordinal_curve']
            xs = list(range(len(cur)))
            ax.plot(xs, [c['reach'] for c in cur], marker='o', ms=3, label=tg + ' reach')
        ax.set_xlabel('eat ordinal bin (5)')
        ax.set_ylabel('reach at eat')
        ax.set_title('P2 reach vs eat ordinal')
        ax.legend(fontsize=8)
        # (4) 食物落点不可达占比随序号
        ax = axes[1][0]
        for tg in tags:
            cur = result['models'][tg]['eat_ordinal_curve']
            ax.plot([c['unreachable_frac'] for c in cur], marker='s', ms=3, label=tg)
        ax.set_xlabel('eat ordinal bin (5)')
        ax.set_ylabel('food spawned unreachable')
        ax.set_title('P4 pocket-food fraction')
        ax.legend(fontsize=8)
        # (5) 每局 food 分布
        ax = axes[1][1]
        for tg in tags:
            ax.hist([r['food'] for r in per_model_rows[tg]], alpha=0.5, label=tg, bins=15)
        if 'elites' in result:
            ax.axvline(result['elites']['food_mean'], color='k', ls='--',
                       label='elite mean')
        ax.set_xlabel('food per episode')
        ax.set_title('P3 food distribution (40 eps)')
        ax.legend(fontsize=8)
        # (6) 精英 food vs reach 散点
        ax = axes[1][2]
        if 'elites' in result:
            xs = [e['food'] for e in elites]
            ys = [e['reach_eat'] for e in elites]
            sc = ax.scatter(xs, ys, c=[f['cur'] for f in fv], cmap='viridis', s=18)
            fig.colorbar(sc, ax=ax, label='fitness(cur)')
        ax.set_xlabel('elite food (10-ep mean)')
        ax.set_ylabel('reach at eat')
        ax.set_title('P2 elites: food vs reach')
        plt.tight_layout()
        out_png = os.path.join(ROOT, 'results', 'test7g_plateau_diagnosis.png')
        fig.savefig(out_png, dpi=110)
        plt.close(fig)
        print(f'[落盘] {out_png}')
    except Exception as e:
        print(f'(绘图跳过: {e})')


if __name__ == '__main__':
    main()
