# ==========================================
# experiments/anneal_ab.py —— 模仿退火双臂 × 10 代（从 imitC gen-230 断点出发）
#
# 背景：教师 v1 @原生钟 mean 93.1 / median 98（无天花板）；C 线 28.5 瓶颈
# 的候选原因是模仿绑定（W=2.0 罚偏离 ≤2 分）压制食物梯度。
# 臂：ANNEAL05 = W=0.5（温和退火）；FREE = W=0（完全放开）。
# te 项（3 分）+ te 配额（32）继续保留，防混沌回退。
# 判定（预注册）：任一臂 10 代内 BestFood > 31 → 转长跑；双臂无效 → 观测增维。
# 产出：results/anneal_ab.json + anneal_ab.png
# ==========================================
import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test7h as t7h  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENS = 10
ARMS = {'ANNEAL05': 0.5, 'FREE': 0.0}
CKPT = 'test7h_imitC_checkpoint.pth'


def te_of(mn, cap):
    return np.where((mn[:, 0] > 0) & (mn[:, 10] > 0),
                    np.minimum(mn[:, 3] / np.maximum(mn[:, 10], 1.0), cap), 0.0)


def run_arm(tag, w, ck, cfg0):
    cfg = copy.copy(cfg0)
    cfg.IMITATION_W = w
    device = t7h._resolve_device(cfg)
    pop = t7h.GeneStack(cfg, device=device)
    pop.unpack(ck['pop'])
    start_gen = int(ck['next_gen'])
    key_fn = t7h._make_key_fn(cfg)
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
            elite_mis=float(np.mean(mn[elite_idx, 12])),
            elite_dens=float(np.mean((mn[elite_idx, 8] + mn[elite_idx, 9]) /
                                     np.maximum(mn[elite_idx, 1] + mn[elite_idx, 2], 1))),
            eval_s=time.time() - t0)
        hist.append(row)
        print(f'  [{tag}] gen{gen}: EliteFood {row["elite_food"]:.2f} | '
              f'BestFood {row["best_food"]:.1f} | 精英te {row["elite_te_med"]:.2f} '
              f'| mis {row["elite_mis"]:.2f} | 密度 {row["elite_dens"]:.2f} | '
              f'{row["eval_s"]:.0f}s', flush=True)
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
    print(f'[anneal-ab] device={device} | 从 next_gen={ck["next_gen"]} 双臂各 {GENS} 代')
    result = {}
    ends = {}
    for tag, w in ARMS.items():
        print(f'\n[臂 {tag}] IMITATION_W={w}')
        hist, pop_end, cfg_end = run_arm(tag, w, ck, cfg0)
        result[tag] = hist
        ends[tag] = (pop_end, cfg_end)
    with open(os.path.join(ROOT, 'results', 'anneal_ab.json'), 'w',
              encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print('\n[落盘] results/anneal_ab.json')
    # 保存各臂末代种群
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
        torch.save(payload, os.path.join(ROOT, f'test7h_anneal_{tag}.pth'))
        print(f'[落盘] test7h_anneal_{tag}.pth')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
        for tag, color in (('ANNEAL05', 'red'), ('FREE', 'blue')):
            h = result[tag]
            g = [r['gen'] for r in h]
            axes[0].plot(g, [r['best_food'] for r in h], 'o-', color=color,
                         label=f'{tag} BestFood')
            axes[0].plot(g, [r['elite_food'] for r in h], 's--', color=color,
                         alpha=0.6, label=f'{tag} EliteFood')
            axes[1].plot(g, [r['elite_dens'] for r in h], 'o-', color=color,
                         label=f'{tag} elite turn density')
        axes[0].axhline(31, color='k', ls=':', lw=0.8)
        axes[1].axhline(0.30, color='k', ls=':', lw=0.8)
        axes[0].set_title('BestFood / EliteFood (anneal)')
        axes[1].set_title('elite turn density (chaos ≈0.85, order ≈0.28)')
        for ax in axes:
            ax.set_xlabel('generation'); ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(os.path.join(ROOT, 'results', 'anneal_ab.png'), dpi=110)
        print('[落盘] results/anneal_ab.png')
    except Exception as e:
        print(f'(绘图跳过: {e})')


if __name__ == '__main__':
    main()
