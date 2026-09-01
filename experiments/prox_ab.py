# ==========================================
# experiments/prox_ab.py —— 追食重塑形双臂 × 10 代（从 anneal_FREE 末代出发）
#
# 诊断：C 线 28-29 瓶颈 = 有序但追食回路退化（模仿 90 代稀释）。prox
# （存活期间 1/食物距离均值，指标列[4]，每步已测）从未进适应度——本实验
# 以 W_p∈{1.5, 3.0} 重装稠密追食梯度。
# 判定（预注册）：任一臂 10 代内 EliteFood > 30（对照基线 FREE 臂
# 25.8→26.3，+0.4/10 代）→ 转长跑；双臂无效 → 观测增维。
# 产出：results/prox_ab.json + prox_ab.png
# ==========================================
import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments', 'test7_series'))  # 归档后模块路径
import test7h as t7h  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENS = 10
ARMS = {'P15': 1.5, 'P30': 3.0}
CKPT = 'test7h_anneal_FREE.pth'


def te_of(mn, cap):
    return np.where((mn[:, 0] > 0) & (mn[:, 10] > 0),
                    np.minimum(mn[:, 3] / np.maximum(mn[:, 10], 1.0), cap), 0.0)


def run_arm(tag, w, ck, cfg0):
    cfg = copy.copy(cfg0)
    cfg.PROX_W = w
    device = t7h._resolve_device(cfg)
    pop = t7h.GeneStack(cfg, device=device)
    pop.unpack(ck['pop'])
    start_gen = int(ck['next_gen'])
    cap = float(cfg.TURN_EFF_CAP)
    hist = []
    for gen in range(start_gen, start_gen + GENS):
        t0 = time.time()
        metrics, order = t7h.evaluate_population_gpu(pop, cfg, gen=gen)
        mn = metrics.numpy()
        elite_idx = order[:cfg.ELITE_SIZE]
        te_elite = te_of(mn[elite_idx], cap)
        row = dict(
            gen=gen,
            elite_food=float(np.mean(mn[elite_idx, 0])),
            best_food=float(np.max(mn[:, 0])),
            avg_food=float(np.mean(mn[:, 0])),
            elite_te_med=float(np.median(te_elite)),
            elite_prox=float(np.mean(mn[elite_idx, 4])),
            elite_dens=float(np.mean((mn[elite_idx, 8] + mn[elite_idx, 9]) /
                                     np.maximum(mn[elite_idx, 1] + mn[elite_idx, 2], 1))),
            eval_s=time.time() - t0)
        hist.append(row)
        print(f'  [{tag}] gen{gen}: EliteFood {row["elite_food"]:.2f} | '
              f'BestFood {row["best_food"]:.1f} | AvgFood {row["avg_food"]:.2f} '
              f'| 精英te {row["elite_te_med"]:.2f} | prox {row["elite_prox"]:.3f} '
              f'| 密度 {row["elite_dens"]:.2f} | {row["eval_s"]:.0f}s', flush=True)
        if gen < start_gen + GENS - 1:
            pop = t7h.evolve_topology_gpu(pop, metrics, cfg, gen=gen, order=order)
    return hist, pop, cfg


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(os.path.join(ROOT, CKPT), map_location='cpu', weights_only=False)
    cfg0 = t7h.Config()
    for k, v in ck.get('config', {}).items():
        if hasattr(cfg0, k) and isinstance(v, (int, float, bool, str)):
            setattr(cfg0, k, v)
    print(f'[prox-ab] device={device} | 从 next_gen={ck["next_gen"]} 双臂各 {GENS} 代')
    result = {}
    ends = {}
    for tag, w in ARMS.items():
        print()
        print(f'[臂 {tag}] PROX_W={w}')
        hist, pop_end, cfg_end = run_arm(tag, w, ck, cfg0)
        result[tag] = hist
        ends[tag] = (pop_end, cfg_end)
    with open(os.path.join(ROOT, 'results', 'prox_ab.json'), 'w',
              encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print()
    print('[落盘] results/prox_ab.json')
    for tag, (pop_end, cfg_end) in ends.items():
        payload = {
            'next_gen': int(ck['next_gen']) + GENS,
            'pop': pop_end.pack(),
            'history': {'gen': [r['gen'] for r in result[tag]],
                        'best_food': [r['best_food'] for r in result[tag]],
                        'avg_food': [r['avg_food'] for r in result[tag]],
                        'best_seen': [], 'best_unseen': [],
                        'elite_food': [r['elite_food'] for r in result[tag]],
                        'best_fit': [], 'best_turneff': []},
            'config': t7h._cfg_dict(cfg_end),
            'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        torch.save(payload, os.path.join(ROOT, f'test7h_prox_{tag}.pth'))
        print(f'[落盘] test7h_prox_{tag}.pth')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
        for tag, color in (('P15', 'red'), ('P30', 'blue')):
            h = result[tag]
            g = [r['gen'] for r in h]
            axes[0].plot(g, [r['elite_food'] for r in h], 'o-', color=color,
                         label=f'{tag} EliteFood')
            axes[0].plot(g, [r['avg_food'] for r in h], 's--', color=color,
                         alpha=0.5, label=f'{tag} AvgFood')
            axes[1].plot(g, [r['elite_prox'] for r in h], 'o-', color=color,
                         label=f'{tag} elite prox')
            axes[1].plot(g, [r['elite_dens'] for r in h], 's--', color=color,
                         label=f'{tag} elite density')
        axes[0].axhline(30, color='k', ls=':', lw=0.8)
        axes[0].set_title('EliteFood / AvgFood (prox shaping)')
        axes[1].set_title('prox / density')
        for ax in axes:
            ax.set_xlabel('generation'); ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(os.path.join(ROOT, 'results', 'prox_ab.png'), dpi=110)
        print('[落盘] results/prox_ab.png')
    except Exception as e:
        print(f'(绘图跳过: {e})')


if __name__ == '__main__':
    main()
