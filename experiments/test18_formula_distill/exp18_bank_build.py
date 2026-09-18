#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test18 公式状态库构建：FormulaTeacherV5 闭环 rollout 采集原始盘面状态，
按蛇长分层（3-10/11-30/31-60/61-90/91-98）配额，存 (盘面, 教师动作)。

库语义（与 fork 训练侧约定一致）：
- 存原始盘面（body/body_len/head/dir_idx/food）+ 教师动作 + 层号 + 观测编码
  版本；观测在训练侧用现行编码器现算（防编码漂移，加载时校验版本）。
- 库一旦生成不可变 → bank 一致率零评估噪声、免疫"早死刷一致率"。

用法：
  python exp18_bank_build.py [--obs 32proj] [--quota 1200,1000,900,700,296]
                             [--batch-eps 128] [--seed 20260913] [--starve 3]
                             [--out formula_bank_32proj_v1.pt]
产物：experiments/test18_formula_distill/<out> +
      results/test18_formula_distill/bank_build_<tag>.json
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


def main():
    ap = argparse.ArgumentParser(description='test18 公式状态库构建')
    ap.add_argument('--obs', type=str, default='32proj',
                    choices=['32proj', '32ego', '40'])
    ap.add_argument('--quota', type=str, default='1200,1000,900,700,296',
                    help='五层配额（蛇长 3-10/11-30/31-60/61-90/91-98）')
    ap.add_argument('--batch-eps', type=int, default=128)
    ap.add_argument('--seed', type=int, default=20260913)
    ap.add_argument('--starve', type=float, default=3.0)
    ap.add_argument('--cap-every', type=int, default=5, help='同局采样间隔（步）')
    ap.add_argument('--max-batches', type=int, default=8)
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    fork = _load_fork()
    cfg = fork.Config()
    cfg.OBS_MODE = args.obs
    fork.apply_obs_mode(cfg)
    cfg.STARVE_SLOPE = args.starve
    cfg.MAX_STEPS = 8000
    cfg.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = fork._resolve_device(cfg)
    G = cfg.GRID_SIZE
    quota = [int(x) for x in args.quota.split(',')]
    assert len(quota) == 5, '配额须为五层'
    layers = fork._FORMULA_BANK_LAYERS
    print(f'[bank] obs={cfg.OBS_MODE}({cfg.OBS_ENC_VERSION}) device={device} '
          f'配额={quota} starve={args.starve} cap_every={args.cap_every}')

    torch.manual_seed(args.seed)
    teacher = fork.FormulaTeacherV5(G, device)
    cap = {k: [] for k in ('body', 'blen', 'dir', 'food', 'act', 'layer')}
    got = [0] * 5
    t0 = time.time()
    for batch in range(args.max_batches):
        if all(g >= q for g, q in zip(got, quota)):
            break
        env = fork.BatchedSnakeEnv(cfg, args.batch_eps, device)
        env.reset()
        ar = torch.arange(args.batch_eps, device=device)
        step_i = 0
        while step_i < cfg.MAX_STEPS and not env.all_done():
            necks = env.body[ar, 1]
            acts = teacher.act(env.head, env.food, env.body, env.body_len,
                               env.dir_idx, necks)
            if step_i % args.cap_every == 0:
                alive = env.alive
                bl = env.body_len
                for lay in range(5):
                    if got[lay] >= quota[lay]:
                        continue
                    lo, hi = layers[lay]
                    sel = torch.nonzero(alive & (bl >= lo) & (bl <= hi)).flatten()
                    need = quota[lay] - got[lay]
                    if sel.numel() > need:
                        perm = torch.randperm(sel.numel(), device=device)[:need]
                        sel = sel[perm]
                    for i in sel.tolist():
                        L = int(bl[i])
                        cap['body'].append(env.body[i, :L].clone())
                        cap['blen'].append(L)
                        cap['dir'].append(int(env.dir_idx[i]))
                        cap['food'].append(env.food[i].clone())
                        cap['act'].append(int(acts[i]))
                        cap['layer'].append(lay)
                        got[lay] += 1
            env.step(acts)
            step_i += 1
        layer_str = ' '.join(f'L{i}{layers[i][0]}-{layers[i][1]}:{got[i]}/{quota[i]}'
                             for i in range(5))
        print(f'[bank] batch {batch + 1}: {step_i} 步 | 已采 {layer_str}')

    M = len(cap['act'])
    if M == 0 or any(g < q for g, q in zip(got, quota)):
        print(f'[bank][警告] 配额未满足：{got} / {quota}（继续用已采状态）')
    body = torch.zeros(M, G * G, 2, dtype=torch.long)
    for i, b in enumerate(cap['body']):
        body[i, :b.shape[0]] = b
    bank = {
        'obs_enc_version': cfg.OBS_ENC_VERSION,
        'grid_size': G,
        'body': body,
        'body_len': torch.tensor(cap['blen'], dtype=torch.long),
        'head': body[:, 0].clone(),
        'dir_idx': torch.tensor(cap['dir'], dtype=torch.long),
        'food': torch.stack(cap['food']).long(),
        'action': torch.tensor(cap['act'], dtype=torch.long),
        'layer': torch.tensor(cap['layer'], dtype=torch.long),
        'layers': layers,
        'obs': None,
    }
    # 有效性自证：库内动作重放 = 教师重算（防捕获时序错位——存的就是决策时刻状态）
    chk_env = fork.BatchedSnakeEnv(cfg, min(M, 2048), device)
    nchk = 0
    bad = 0
    for lo in range(0, M, 2048):
        hi = min(lo + 2048, M)
        m = hi - lo
        if chk_env.B != m:
            chk_env = fork.BatchedSnakeEnv(cfg, m, device)
        chk_env.reset()
        chk_env.head = bank['head'][lo:hi].to(device)
        chk_env.body = bank['body'][lo:hi].to(device)
        chk_env.body_len = bank['body_len'][lo:hi].to(device)
        chk_env.dir_idx = bank['dir_idx'][lo:hi].to(device)
        chk_env.food = bank['food'][lo:hi].to(device)
        ar2 = torch.arange(m, device=device)
        re_act = teacher.act(chk_env.head, chk_env.food, chk_env.body,
                             chk_env.body_len, chk_env.dir_idx,
                             chk_env.body[ar2, 1])
        bad += int((re_act != bank['action'][lo:hi].to(device)).sum())
        nchk += m
    verdict = 'PASS' if bad == 0 else 'FAIL（捕获错位！）'
    print(f'[bank] 动作重放校验：{nchk - bad}/{nchk} 一致（{verdict}）')

    out_name = args.out or f'formula_bank_{args.obs}_v1.pt'
    out = _HERE / out_name
    fork.save_formula_bank(str(out), bank)
    act_np = bank['action'].numpy()
    stats = {
        'total': M, 'per_layer': got, 'quota': quota,
        'layers': [list(l) for l in layers],
        'act_dist': {str(a): int((act_np == a).sum()) for a in (0, 1, 2)},
        'obs_enc_version': cfg.OBS_ENC_VERSION,
        'starve_slope': args.starve, 'seed': args.seed,
        'replay_check_bad': bad, 'runtime_sec': time.time() - t0,
        'out': str(out),
    }
    rep = _ROOT / 'results' / 'test18_formula_distill' / f'bank_build_{args.obs}.json'
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                   encoding='utf-8')
    print(f'[bank] 完成：{M} 态 → {out}')
    print(f'[bank] 动作分布 直/左/右 = '
          f"{stats['act_dist']['0']}/{stats['act_dist']['1']}/{stats['act_dist']['2']}"
          f' | {stats["runtime_sec"]:.1f}s | 报告 {rep}')


if __name__ == '__main__':
    main()
