#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test18 闭环基准评测：模型 × {闭环分数, 死因, on-policy 分蛇长公式一致度,
bank 分层一致度}；附 V5 公式同协议参照行。

口径：starve=3（V5/cheat7b 协议）、随机地图（非 CRN）、有状态闭环（E/I/st
跨步持续——真实部署口径，与训练期评估一致）。

用法：
  python exp18_benchmark.py --model ../../test7b_base_model.pth \
      --model test18a_fd_scratch_best_model.pth [--episodes 100] [--seed 20260913]
      [--bank formula_bank_32proj_v1.pt] [--out benchmark_xxx.json]
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


def _load_fork():
    spec = importlib.util.spec_from_file_location('test18a_formula',
                                                  _HERE / 'test18a_formula.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def eval_model(fork, cfg, device, state, episodes, seed, teacher):
    """单模型闭环评测：有状态 rollout + 分蛇长 on-policy 一致度。"""
    torch.manual_seed(seed)
    G = cfg.GRID_SIZE
    pop = fork.GeneStack(cfg, B=1, device=device)
    pop.random_init()                      # 先分配张量（run_training 同序）
    pop.set_individual_from_state(0, state)
    pop.refresh_eff()
    if bool(getattr(cfg, 'USE_FP16', True)):
        pop.fp16()
    popb = pop[torch.zeros(episodes, dtype=torch.long, device=device)]
    popb.refresh_eff()
    env = fork.BatchedSnakeEnv(cfg, episodes, device)
    env.reset()
    half = torch.float16 if bool(getattr(cfg, 'USE_FP16', True)) else torch.float32
    N = cfg.NUM_COLUMNS
    E = torch.zeros(episodes, N, dtype=half, device=device)
    I = torch.zeros(episodes, N, dtype=half, device=device)
    st = torch.zeros(episodes, N, dtype=half, device=device)
    press = torch.zeros(episodes, dtype=half, device=device)
    ar = torch.arange(episodes, device=device)
    layers = fork._FORMULA_BANK_LAYERS
    lay_agree = torch.zeros(5, dtype=torch.float64)
    lay_steps = torch.zeros(5, dtype=torch.float64)
    t0 = time.time()
    with torch.no_grad():
        for t in range(cfg.MAX_STEPS):
            if env.all_done():
                break
            al = env.alive
            obs = env.obs().to(half)
            act, E, I, st, _, _ = fork.deliberate_batch(popb, obs, E, I, st, press, cfg)
            press = fork.update_fatigue(press, act, decay=float(cfg.FATIGUE_TURN_DECAY))
            f_act = teacher.act(env.head, env.food, env.body, env.body_len,
                                env.dir_idx, env.body[ar, 1])
            bl = env.body_len
            for li in range(5):
                lo, hi = layers[li]
                # 分层一致度（每步全量统计，评测规模下可承受）
                m = al & (bl >= lo) & (bl <= hi)
                lay_agree[li] += float((m & (act == f_act)).sum())
                lay_steps[li] += float(m.sum())
            env.step(act)
    food = (env.body_len - 2).cpu().numpy().astype(int)
    died = env.died.cpu().numpy().astype(int)
    agree = (lay_agree / lay_steps.clamp(min=1)).tolist()
    cover = (lay_steps / lay_steps.sum().clamp(min=1)).tolist()
    return {
        'episodes': episodes,
        'food_mean': float(food.mean()), 'food_median': float(np.median(food)),
        'food_std': float(food.std(ddof=1)) if episodes > 1 else 0.0,
        'food_min': int(food.min()), 'food_max': int(food.max()),
        'ge62': int((food >= 62).sum()), 'ge80': int((food >= 80).sum()),
        'ge90': int((food >= 90).sum()), 'perfect98': int((food >= 98).sum()),
        'death_wall': int((died == 1).sum()),
        'death_self': int((died == 2).sum()),
        'death_starve': int((died == 3).sum()),
        'onpolicy_agree_by_layer': agree,
        'onpolicy_layer_coverage': cover,
        'onpolicy_agree_total': float(
            (lay_agree.sum() / lay_steps.sum().clamp(min=1))),
        'runtime_sec': time.time() - t0,
    }


def bank_eval(fork, cfg, device, state, bank):
    """bank 分层一致率（零状态单步决策）。"""
    fork._FORMULA_BANK = bank
    pop = fork.GeneStack(cfg, B=1, device=device)
    pop.random_init()
    pop.set_individual_from_state(0, state)
    pop.refresh_eff()
    obs_full = fork.formula_bank_obs(bank, cfg, device)
    want = bank['action']
    half = torch.float16 if bool(getattr(cfg, 'USE_FP16', True)) else torch.float32
    out = {}
    with torch.no_grad():
        one = pop[0:1]
        if one.W_in_eff is None or one.W_out_eff is None:
            one.refresh_eff()
        view = fork._ExpandedOnePop(one, want.shape[0])
        m_all = want.shape[0]
        acts = torch.zeros(m_all, dtype=torch.long, device=device)
        chunk = 4096
        for lo in range(0, m_all, chunk):
            hi = min(lo + chunk, m_all)
            m = hi - lo
            o = obs_full[lo:hi].to(view.W_in_eff.dtype)
            E = torch.zeros(m, pop.N, dtype=half, device=device)
            I = torch.zeros(m, pop.N, dtype=half, device=device)
            st = torch.zeros(m, pop.N, dtype=half, device=device)
            pr = torch.zeros(m, dtype=half, device=device)
            a, _, _, _, _, _ = fork.deliberate_batch(view, o, E, I, st, pr, cfg)
            acts[lo:hi] = a
        bl = bank['body_len']
        per = {}
        for li, (lo2, hi2) in enumerate(fork._FORMULA_BANK_LAYERS):
            msk = (bl >= lo2) & (bl <= hi2)
            per[str(li)] = float((acts[msk] == want[msk]).float().mean())
        out['bank_agree_total'] = float((acts == want).float().mean())
        out['bank_agree_by_layer'] = per
    return out


def main():
    ap = argparse.ArgumentParser(description='test18 闭环基准评测')
    ap.add_argument('--model', action='append', required=True,
                    help='模型路径（可多次）；7b 稠密基因组自动转换')
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--seed', type=int, default=20260913)
    ap.add_argument('--obs', type=str, default='32proj')
    ap.add_argument('--starve', type=float, default=3.0)
    ap.add_argument('--bank', type=str, default=str(_HERE / 'formula_bank_32proj_v1.pt'))
    ap.add_argument('--skip-v5-ref', action='store_true')
    ap.add_argument('--device', type=str, default=None,
                    help='cuda|cpu（默认自动；GPU 训练期间可用 cpu 避免抢卡）')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    fork = _load_fork()
    cfg = fork.Config()
    cfg.OBS_MODE = args.obs
    fork.apply_obs_mode(cfg)
    cfg.STARVE_SLOPE = args.starve
    cfg.MAX_STEPS = 8000
    cfg.DEVICE = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = fork._resolve_device(cfg)
    teacher = fork.FormulaTeacherV5(cfg.GRID_SIZE, device)

    bank = None
    if args.bank and Path(args.bank).exists():
        bank = fork.load_formula_bank(args.bank, cfg, device)
        print(f'[bench] bank 载入 {args.bank}（{bank["body_len"].shape[0]} 态）')
    else:
        print('[bench] bank 未提供或不存在，跳过 bank 一致率')

    report = {'protocol': {'obs': cfg.OBS_MODE, 'starve': args.starve,
                           'episodes': args.episodes, 'seed': args.seed},
              'models': {}}
    for path in args.model:
        name = Path(path).stem
        seed_st = fork.load_seed_state_any(path, cfg, verbose=False)
        if seed_st is None:
            print(f'[bench][跳过] {path} 无法加载')
            continue
        st, s_food, s_steps = seed_st
        r = eval_model(fork, cfg, device, st, args.episodes, args.seed, teacher)
        r['checkpoint_food'] = float(s_food)
        if bank is not None:
            r.update(bank_eval(fork, cfg, device, st, bank))
        report['models'][name] = r
        agree_str = ' '.join(f'L{i}:{a:.3f}' for i, a in
                             enumerate(r['onpolicy_agree_by_layer']))
        print(f"[bench] {name}: food {r['food_mean']:.2f} (中位 {r['food_median']:.0f} "
              f"min/max {r['food_min']}/{r['food_max']} 98食 {r['perfect98']}) | "
              f"死(W/S/St) {r['death_wall']}/{r['death_self']}/{r['death_starve']} | "
              f"on-policy 总一致 {r['onpolicy_agree_total']:.3f} [{agree_str}]"
              + (f" | bank {r.get('bank_agree_total', float('nan')):.4f}"
                 if bank is not None else ''))

    if not args.skip_v5_ref:
        spec = importlib.util.spec_from_file_location(
            'sinntry_v5_final',
            _ROOT / 'experiments' / 'v5_formula' / 'SiNNtry_V5_final_formula_run.py')
        v5 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(v5)
        ref = v5.run(args.episodes, args.seed, 8000, args.starve)
        report['v5_reference'] = {k: ref[k] for k in
                                  ('food_mean', 'food_median', 'food_std',
                                   'food_min', 'food_max', 'perfect98',
                                   'death_wall', 'death_self', 'death_starve')}
        print(f"[bench] V5公式参照: food {ref['food_mean']:.2f} "
              f"(中位 {ref['food_median']:.0f}, 98食 {ref['perfect98']})")

    out = Path(args.out) if args.out else (
        _ROOT / 'results' / 'test18_formula_distill' / f'benchmark_{args.seed}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding='utf-8')
    print(f'[bench] 报告已保存 {out}')


if __name__ == '__main__':
    main()
