#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test18 门1/门2：FormulaTeacherV5 移植正确性验证（权威基准 = 根目录
SiNNtry_V5_final_formula_run.py，逐位对拍）。

门1：≥2000 个分层状态（蛇长层 3-10/11-30/31-60/61-90/91-98）上，fork 的
     FormulaTeacherV5.act 与 V5 脚本 choose_actions 动作逐位一致。
     状态来源两路：(a) V5 脚本自身 SnakeBatch 闭环轨迹的真实状态；
                  (b) 随机自回避蛇合成状态（保证长蛇层配额）。
门2：教师驱动 fork BatchedSnakeEnv 闭环 100 局（starve=3，同 V5 协议），
     food_mean 与 V5 脚本自跑（同局数/种子）统计对照（环境 RNG 流不同，
     逐局不可比，均值应同分布）。

用法：
  python exp18_teacher_gate.py [--states 2000] [--episodes 100] [--seed 20260912]
  python exp18_teacher_gate.py --skip-gate2          # 只跑门1
产物：results/test18_formula_distill/gate_result.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _serpentine_paths(G):
    """4 个朝向的蛇形哈密顿路径（相邻格连续），用于构造长蛇前缀。"""
    paths = []
    base = []
    for r in range(G):
        cols = range(G) if r % 2 == 0 else range(G - 1, -1, -1)
        base.extend((r, c) for c in cols)
    paths.append(base)
    paths.append(base[::-1])
    # 转置版本
    tb = []
    for c in range(G):
        rows = range(G) if c % 2 == 0 else range(G - 1, -1, -1)
        tb.extend((r, c) for r in rows)
    paths.append(tb)
    paths.append(tb[::-1])
    return paths


def synth_states(n, layers, rng, G=10):
    """分层合成状态：蛇长 ≤60 用随机自回避游走（拓扑多样）；>60 用蛇形
    哈密顿路径随机前缀（随机 SAW 在 10×10 上几乎不可能达到 90+ 节）。
    返回 (body, dir_idx, food) 列表；rng 为 random.Random。"""
    DIRS4 = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    serp = _serpentine_paths(G)
    out = []
    for i in range(n):
        lo, hi = layers[i % len(layers)]
        target = rng.randint(max(2, lo), hi)
        body = None
        if target > 60:
            for _ in range(300):
                p = rng.choice(serp)
                start = rng.randint(0, G * G - target)
                body = list(p[start:start + target])
                if len(body) == target:
                    break
            if body is None or len(body) != target:
                continue
        else:
            for _ in range(300):
                cand_body = [(rng.randrange(G), rng.randrange(G))]
                cur = rng.randrange(4)
                while len(cand_body) < target:
                    cand = []
                    for turn in (0, -1, 1):
                        nd = (cur + turn) % 4
                        nxt = (cand_body[-1][0] + DIRS4[nd][0],
                               cand_body[-1][1] + DIRS4[nd][1])
                        if (0 <= nxt[0] < G and 0 <= nxt[1] < G
                                and nxt not in cand_body):
                            cand.append((nd, nxt))
                    if not cand:
                        break
                    nd, nxt = rng.choice(cand)
                    cand_body.append(nxt)
                    cur = nd
                if len(cand_body) == target:
                    body = cand_body
                    break
            if body is None:
                continue
        v = (body[0][0] - body[1][0], body[0][1] - body[1][1])
        if v not in DIRS4:
            continue
        free = [(r, c) for r in range(G) for c in range(G)
                if (r, c) not in body]
        f = rng.choice(free) if free else (0, 0)
        out.append((body, DIRS4.index(v), f))
    return out


def main():
    ap = argparse.ArgumentParser(description='test18 公式教师移植正确性门')
    ap.add_argument('--states', type=int, default=2000, help='门1 合成状态数')
    ap.add_argument('--rollout-states', type=int, default=600,
                    help='门1 V5 真实轨迹捕获状态数')
    ap.add_argument('--episodes', type=int, default=100, help='门2 局数')
    ap.add_argument('--seed', type=int, default=20260912)
    ap.add_argument('--skip-gate2', action='store_true')
    ap.add_argument('--out', default=str(_ROOT / 'results' / 'test18_formula_distill'
                                         / 'gate_result.json'))
    args = ap.parse_args()

    fork = _load_module('test18a_formula', _HERE / 'test18a_formula.py')
    v5 = _load_module('sinntry_v5_final',
                      _ROOT / 'experiments' / 'v5_formula' / 'SiNNtry_V5_final_formula_run.py')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[gate] device={device}')
    G = 10
    layers = ((3, 10), (11, 30), (31, 60), (61, 90), (91, 98))
    teacher = fork.FormulaTeacherV5(G, device)
    report = {'device': str(device), 'gate1': {}, 'gate2': {}}
    t_start = time.time()

    # ---------------- 门1a：合成分层状态 vs V5 choose_actions ----------------
    import random as _random
    rng = _random.Random(args.seed)
    synth = synth_states(args.states, layers, rng, G)
    n_synth = len(synth)
    body = torch.zeros(n_synth, G * G, 2, dtype=torch.long)
    blen = torch.zeros(n_synth, dtype=torch.long)
    darr = torch.zeros(n_synth, dtype=torch.long)
    food = torch.zeros(n_synth, 2, dtype=torch.long)
    for i, (b, d0, f) in enumerate(synth):
        blen[i] = len(b)
        for j, cell in enumerate(b):
            body[i, j, 0], body[i, j, 1] = cell
        food[i, 0], food[i, 1] = f
        darr[i] = d0
    head = body[:, 0].clone()

    env = v5.SnakeBatch(n_synth)                       # V5 权威动作：状态灌入其 env
    env.body = body.clone()
    env.body_len = blen.clone()
    env.head = head.clone()
    env.food = food.clone()
    env.dir_idx = darr.clone()
    env.alive = torch.ones(n_synth, dtype=torch.bool)
    t0 = time.time()
    ref = v5.choose_actions(env).numpy()
    t_ref = time.time() - t0
    t0 = time.time()
    got = teacher.act(head.to(device), food.to(device), body.to(device),
                      blen.to(device), darr.to(device)).cpu().numpy()
    t_got = time.time() - t0
    diff = int((ref != got).sum())
    report['gate1']['synth'] = {
        'n': n_synth, 'mismatch': diff,
        'per_layer': [int(((ref != got) & (blen.numpy() >= lo) & (blen.numpy() <= hi)).sum())
                      for lo, hi in layers],
        'len_dist': [int(((blen.numpy() >= lo) & (blen.numpy() <= hi)).sum())
                     for lo, hi in layers],
    }
    print(f"[门1a 合成] {n_synth} 态 mismatch={diff} "
          f"(V5 {t_ref:.2f}s / fork {t_got:.2f}s)")
    if diff:
        bad = np.flatnonzero(ref != got)[:5]
        for i in bad:
            print(f"  首个不一致 idx={i} len={int(blen[i])} ref={ref[i]} got={got[i]} "
                  f"food={food[i].tolist()} dir={int(darr[i])}")

    # ---------------- 门1b：V5 真实闭环轨迹状态（按蛇长层配额捕获）----------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2 ** 32 - 1))
    renv = v5.SnakeBatch(64)
    cap_every, want = 5, args.rollout_states
    per_layer_want = [want // 5] * 5
    per_layer_want[4] += want - sum(per_layer_want)
    cap = {'body': [], 'blen': [], 'dir': [], 'food': [], 'act': []}
    got_layers = [0] * 5
    steps = 0
    while steps < 8000 and any(g < w for g, w in zip(got_layers, per_layer_want)):
        acts = v5.choose_actions(renv)
        alive = renv.alive.numpy()
        if steps % cap_every == 0:
            idx = np.flatnonzero(alive)
            for i in idx:
                L = int(renv.body_len[i])
                lay = next((k for k, (lo, hi) in enumerate(layers)
                            if lo <= L <= hi), 0 if L < 3 else 4)
                if lay >= len(per_layer_want) or got_layers[lay] >= per_layer_want[lay]:
                    continue
                cap['body'].append(renv.body[i, :L].clone())
                cap['blen'].append(L)
                cap['dir'].append(int(renv.dir_idx[i]))
                cap['food'].append(renv.food[i].clone())
                cap['act'].append(int(acts[i]))
                got_layers[lay] += 1
        renv.step(acts)
        steps += 1
        if renv.all_done():
            break
    n_real = len(cap['act'])
    rb = torch.zeros(n_real, G * G, 2, dtype=torch.long)
    for i, b in enumerate(cap['body']):
        rb[i, :b.shape[0]] = b
    rl = torch.tensor(cap['blen'], dtype=torch.long)
    rd = torch.tensor(cap['dir'], dtype=torch.long)
    rf = torch.stack(cap['food']).long()
    ra = np.array(cap['act'])
    got2 = teacher.act(rb[:, 0].to(device), rf.to(device), rb.to(device),
                       rl.to(device), rd.to(device)).cpu().numpy()
    diff2 = int((ra != got2).sum())
    report['gate1']['rollout'] = {'n': n_real, 'mismatch': diff2,
                                  'rollout_steps': steps}
    print(f"[门1b 真实轨迹] {n_real} 态（{steps} 步闭环内捕获）mismatch={diff2}")

    gate1_pass = (diff == 0 and diff2 == 0 and n_synth >= 1000 and n_real >= 200)
    report['gate1']['pass'] = bool(gate1_pass)

    # 覆盖统计：拓扑算符活跃度（含 R=0 候选的状态数 / 追尾模式状态数）
    sub = min(400, n_synth)
    r_filter = 0
    tail_mode = 0
    for i in range(sub):
        b = [(int(r), int(c)) for r, c in body[i].numpy()[:int(blen[i])]]
        a_ref, cand = _scalar_candidates(b, int(darr[i]),
                                         (int(food[i][0]), int(food[i][1])), G)
        if any((not c['R']) for c in cand):
            r_filter += 1
        safe = [c for c in cand if c['R']] or cand
        if not any(c['G'] for c in safe):
            tail_mode += 1
    report['gate1']['coverage'] = {
        'sampled': sub,
        'any_candidate_R0': r_filter,
        'food_unreachable_state': tail_mode,
    }
    print(f"[门1 覆盖] {sub} 态中含 R=0 候选: {r_filter} | "
          f"食物全不可达(追尾模式): {tail_mode}")

    # ---------------- 门2：教师驱动 fork 环闭环 ----------------
    gate2_pass = None
    if not args.skip_gate2:
        cfg = fork.Config()
        cfg.GRID_SIZE = G
        cfg.STARVE_SLOPE = 3.0                      # 对齐 V5/cheat7b 口径
        cfg.MAX_STEPS = 8000
        cfg.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        dev = fork._resolve_device(cfg)
        torch.manual_seed(args.seed + 1)
        fenv = fork.BatchedSnakeEnv(cfg, args.episodes, dev)
        fenv.reset()
        fteacher = fork.FormulaTeacherV5(G, dev)
        ar = torch.arange(args.episodes, device=dev)
        t0 = time.time()
        n_steps = 0
        for _ in range(cfg.MAX_STEPS):
            if fenv.all_done():
                break
            necks = fenv.body[ar, 1]
            a = fteacher.act(fenv.head, fenv.food, fenv.body, fenv.body_len,
                             fenv.dir_idx, necks)
            fenv.step(a)
            n_steps += 1
        t_roll = time.time() - t0
        food_f = (fenv.body_len - 2).cpu().numpy().astype(int)
        died_f = fenv.died.cpu().numpy().astype(int)
        fork_stats = {
            'episodes': args.episodes,
            'food_mean': float(food_f.mean()),
            'food_median': float(np.median(food_f)),
            'food_min': int(food_f.min()), 'food_max': int(food_f.max()),
            'perfect98': int((food_f >= 98).sum()),
            'death_wall': int((died_f == 1).sum()),
            'death_self': int((died_f == 2).sum()),
            'death_starve': int((died_f == 3).sum()),
            'steps': n_steps, 'runtime_sec': t_roll,
        }
        t0 = time.time()
        ref_run = v5.run(args.episodes, args.seed, 8000, 3.0)
        t_v5 = time.time() - t0
        ref_stats = {k: ref_run[k] for k in
                     ('food_mean', 'food_median', 'food_min', 'food_max',
                      'perfect98', 'death_wall', 'death_self', 'death_starve')}
        ref_stats['runtime_sec'] = t_v5
        dmean = abs(fork_stats['food_mean'] - ref_stats['food_mean'])
        gate2_pass = bool(dmean <= 4.0 and fork_stats['food_mean'] >= 90.0)
        report['gate2'] = {'fork_env': fork_stats, 'v5_env': ref_stats,
                           'mean_abs_diff': dmean, 'pass': gate2_pass}
        print(f"[门2 fork环境] mean={fork_stats['food_mean']:.2f} "
              f"median={fork_stats['food_median']:.0f} min/max="
              f"{fork_stats['food_min']}/{fork_stats['food_max']} "
              f"98食={fork_stats['perfect98']} 死(W/S/St)="
              f"{fork_stats['death_wall']}/{fork_stats['death_self']}/"
              f"{fork_stats['death_starve']} 步数={n_steps} {t_roll:.1f}s")
        print(f"[门2 V5环境] mean={ref_stats['food_mean']:.2f} "
              f"median={ref_stats['food_median']:.0f} "
              f"98食={ref_stats['perfect98']} {t_v5:.1f}s")
        print(f"[门2] |Δmean|={dmean:.2f} → {'PASS' if gate2_pass else 'FAIL'}")

    report['total_sec'] = time.time() - t_start
    all_pass = gate1_pass and (gate2_pass if gate2_pass is not None else True)
    report['pass'] = bool(all_pass)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding='utf-8')
    print(f"[gate] 总判定 {'PASS' if all_pass else 'FAIL'} | "
          f"结果已保存 {out} ({report['total_sec']:.1f}s)")
    sys.exit(0 if all_pass else 1)


def _scalar_candidates(body, d0, food, G=10):
    """镜像 formula_v5_scalar 的候选构造，返回 (动作, 候选字典列表)。"""
    DIRS4 = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    not_right = not_left = 0
    for r in range(G):
        for c in range(G):
            bit = 1 << (r * G + c)
            if c < G - 1:
                not_right |= bit
            if c > 0:
                not_left |= bit
    length = len(body)
    pref = [0] * (length + 1)
    mask = 0
    for j, (r, c) in enumerate(body):
        mask |= 1 << (r * G + c)
        pref[j + 1] = mask
    head = body[0]
    head_bit = 1 << (head[0] * G + head[1])
    food_idx = food[0] * G + food[1]
    candidates = []
    for act_i, turn in enumerate((0, -1, 1)):
        nd = (d0 + turn) % 4
        nh = (head[0] + DIRS4[nd][0], head[1] + DIRS4[nd][1])
        if nh[0] < 0 or nh[0] >= G or nh[1] < 0 or nh[1] >= G:
            continue
        ni = nh[0] * G + nh[1]
        eat = (nh == food)
        collision = (pref[length] if eat else pref[length - 1]) & ~head_bit
        if (collision >> ni) & 1:
            continue
        blocked = pref[length - 1] if eat else pref[max(length - 2, 0)]
        ft = body[length - 1] if eat else body[max(length - 2, 0)]
        ti = ft[0] * G + ft[1]
        free = ((1 << (G * G)) - 1) & ~blocked
        reach = 1 << ni
        for _ in range(G * G):
            nb = ((reach << G) | (reach >> G)
                  | ((reach & not_right) << 1) | ((reach & not_left) >> 1)) & free
            nr = reach | nb
            if nr == reach:
                break
            reach = nr
        candidates.append({'act': act_i,
                           'R': bool((reach >> ti) & 1),
                           'G': bool((reach >> food_idx) & 1),
                           'DF': abs(nh[0] - food[0]) + abs(nh[1] - food[1]),
                           'DT': abs(nh[0] - ft[0]) + abs(nh[1] - ft[1])})
    if not candidates:
        return 0, []
    safe = [q for q in candidates if q['R']] or candidates
    pool = [q for q in safe if q['G']]
    if pool:
        bv = min(q['DF'] for q in pool)
        best = sorted([q for q in pool if q['DF'] == bv], key=lambda q: q['act'])
    else:
        bv = max(q['DT'] for q in safe)
        best = sorted([q for q in safe if q['DT'] == bv], key=lambda q: q['act'])
    return best[0]['act'], candidates


if __name__ == '__main__':
    main()
