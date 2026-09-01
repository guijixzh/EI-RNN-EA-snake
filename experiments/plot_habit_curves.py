# 习惯三因素局内实时曲线：best 模型 vs 教师解法器，同库 10 局。
# 每局逐存活步记录：edge_pref（外环-内部占用差）、conn（蛇身外空间单连通 0/1，
# 严格含尾）、straight（累计 1−转弯/步数）。死亡即曲线终止。
# 输出: results/habit_curves.png（2 行×3 列：上=模型 下=教师）
# 用法: python experiments/plot_habit_curves.py [--n 10] [--model test12_econ_best_model.pth]
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
import torch

import test12 as t12


def run_tracked(model_pop=None, teacher=None, cfg=None, bank=None, B=0, dev=None):
    """同库跑策略，返回逐步序列 {factor: [B, T]}（死亡后 NaN）。"""
    env = t12.BatchedSnakeEnv(cfg, B, dev)
    env.reset(bank=bank)
    ar = torch.arange(B, device=dev)
    T = 0
    series = None
    ep = [dict(epref=[], conn=[], straight=[]) for _ in range(B)]
    E = torch.zeros(B, cfg.NUM_COLUMNS,
                    dtype=torch.float16 if getattr(cfg, 'USE_FP16', True) else torch.float32,
                    device=dev)
    I = torch.zeros_like(E)
    stt = torch.zeros_like(E)
    press = torch.zeros(B, dtype=torch.float32, device=dev)
    turns = torch.zeros(B, device=dev)
    steps = torch.zeros(B, device=dev)
    food = torch.zeros(B, device=dev)
    t = 0
    while t < cfg.MAX_STEPS:
        al = env.alive
        obs = env.obs().to(E.dtype)
        if teacher is not None:
            necks = env.body[ar, 1]
            act = teacher.act(env.head, env.food, env.body, env.body_len,
                              env.dir_idx, necks)
        else:
            act, E, I, stt = t12.deliberate_batch(model_pop, obs, E, I, stt,
                                                  press, cfg)
        epv = t12.habit_edge_score(env) / (16.0 * env.body_len.float().clamp(min=1))
        cv = t12.habit_conn_score(env)
        steps += al.float()
        turns += (al & (act != 0)).float()
        st_rate = 1.0 - turns / steps.clamp(min=1.0)
        for i in range(B):
            if bool(al[i]):
                ep[i]['epref'].append(float(epv[i]))
                ep[i]['conn'].append(float(cv[i]))
                ep[i]['straight'].append(float(st_rate[i]))
        env.step(act)
        food += (al & env.ate).float()
        t += 1
        if env.all_done():
            break
    T = max(len(e['epref']) for e in ep)
    out = {}
    for k in ('epref', 'conn', 'straight'):
        arr = np.full((B, T), np.nan)
        for i, e in enumerate(ep):
            arr[i, :len(e[k])] = e[k]
        out[k] = arr
    out['food'] = food.cpu().numpy()
    out['steps'] = steps.cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--model', default='test12_econ_best_model.pth')
    ap.add_argument('--out', default='results/habit_curves.png')
    args = ap.parse_args()

    cfg = t12.Config()
    cfg.MAX_STEPS = 100000
    dev = t12._resolve_device(cfg)
    B = args.n
    banks = t12.make_banks(cfg, 0, 1, B, dev)
    bank = {'stream': torch.stack([b['stream'] for b in banks]),
            'dir0': torch.tensor([b['dir0'] for b in banks],
                                 dtype=torch.long, device=dev)}

    res = t12.load_best_state(args.model, cfg)
    if res is None:
        sys.exit('[错误] best 模型不兼容')
    st = res[0]
    pop = t12.GeneStack(cfg, B=B, device=dev)
    pop.random_init()
    for i in range(B):
        pop.set_individual_from_state(i, st)
    pop.refresh_eff()
    if getattr(cfg, 'USE_FP16', True):
        pop.fp16()
        pop.refresh_eff()

    print('跑 best 模型 10 局…', flush=True)
    r_m = run_tracked(model_pop=pop, cfg=cfg, bank=bank, B=B, dev=dev)
    print('跑教师 10 局…', flush=True)
    teacher = t12.VectorCycleTeacher(cfg.GRID_SIZE, dev)
    r_t = run_tracked(teacher=teacher, cfg=cfg, bank=bank, B=B, dev=dev)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    titles = {'epref': '因素1 edge_share（圈层总分 16/8/4/2/1 ÷(16×蛇长)）',
              'conn': '因素2 conn（自由空间连通块倒数 1/n，严格含尾）',
              'straight': '因素3 straight（累计 1−转弯/步）'}
    fig, axes = plt.subplots(2, 3, figsize=(19, 8), sharex='col')
    for row, (name, r) in enumerate((('best 模型', r_m), ('教师解法器', r_t))):
        food = r['food']
        for col, k in enumerate(('epref', 'conn', 'straight')):
            ax = axes[row][col]
            arr = r[k]
            for i in range(B):
                y = arr[i]
                x = np.arange(len(y))
                ax.plot(x, y, lw=0.9, alpha=0.75,
                        label=f'局{i} food={int(food[i])}')
            ax.set_title(f"{name} — {titles[k]}", fontsize=10)
            ax.grid(alpha=0.3)
            ax.axhline(0, color='gray', lw=0.6)
            if col == 0:
                ax.set_ylabel(name, fontsize=11)
            if row == 1:
                ax.set_xlabel('局内步数')
            if col == 2 and row == 0:
                ax.legend(fontsize=6.5, loc='lower left', ncol=2)
            if k == 'conn':
                ax.set_ylim(-0.02, 1.06)
    fig.suptitle(f'习惯三因素局内实时变化（同库 {B} 局；曲线终点=该局死亡步）', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=110)
    print(f'已保存: {args.out}')
    for name, r in (('模型', r_m), ('教师', r_t)):
        print(f"{name}: food 均值 {np.nanmean(r['food']):.1f} | "
              f"edge_share 均值 {np.nanmean(r['epref']):.3f} | "
              f"conn 均值 {np.nanmean(r['conn']):.3f} | "
              f"straight 终值均值 {np.nanmean(r['straight']):.3f}")


if __name__ == '__main__':
    main()
