# ==========================================
# experiments/te_quota_ab.py —— te-配额精英 A/B 小实验（双臂 × 8 代，GPU）
#
# 假设（行为学 B5）：精英 te≥2 占比 0% = 选择无料可选；te-配额给稀有有序
# 变异体进入繁殖池的通道。
# 协议：同一 checkpoint（next_gen 断点）出发，双臂各 8 代：
#   OFF: TE_ELITE=0（现行）；ON: TE_ELITE=32。
# 同代同 CRN 库（make_banks 按 gen/stage/ep 定种子）→ 双臂逐代可比。
# 逐代记录：EliteFood、精英 te 分位、全种群 te≥2 / te≥1.5 计数、te 配额
# 实际注入数、BestFood。
# 产出：results/te_quota_ab.json + te_quota_ab.png
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
GENS = 8
ARMS = {'OFF': 0, 'ON': 32}


def te_of(mn, cap):
    return np.where((mn[:, 0] > 0) & (mn[:, 10] > 0),
                    np.minimum(mn[:, 3] / np.maximum(mn[:, 10], 1.0), cap), 0.0)


def run_arm(tag, te_elite, ck, cfg0, banks_seed):
    cfg = copy.copy(cfg0)
    cfg.TE_ELITE = te_elite
    cfg.CHECKPOINT_PATH = f'test7h_teab_{tag}.pth'
    cfg.BEST_MODEL_PATH = f'test7h_teab_{tag}_best.pth'
    cfg.LATEST_GEN_BEST_MODEL_PATH = f'test7h_teab_{tag}_latest.pth'
    cfg.AUTO_RESUME = True
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
        te_all = te_of(mn, cap)
        te_elite_vals = te_all[elite_idx]
        row = dict(
            gen=gen,
            elite_food=float(np.mean(mn[elite_idx, 0])),
            avg_food=float(np.mean(mn[:, 0])),
            best_food=float(np.max(mn[:, 0])),
            elite_te_med=float(np.median(te_elite_vals)),
            elite_te_max=float(np.max(te_elite_vals)),
            elite_te_ge2=int((te_elite_vals >= 2).sum()),
            elite_te_ge15=int((te_elite_vals >= 1.5).sum()),
            pop_te_ge2=int((te_all >= 2).sum()),
            pop_te_ge15=int((te_all >= 1.5).sum()),
            pop_te_max=float(np.max(te_all)),
            eval_s=time.time() - t0)
        # 注入数（配额个体与适应度精英的差集）
        te_order = sorted(range(cfg.POP_SIZE), key=lambda i: te_all[i], reverse=True)
        extra = [i for i in te_order if i not in set(elite_idx)][:te_elite]
        row['quota_injected'] = len(extra)
        hist.append(row)
        print(f'  [{tag}] gen{gen}: EliteFood {row["elite_food"]:.2f} | '
              f'精英te中位 {row["elite_te_med"]:.2f}/max {row["elite_te_max"]:.2f} '
              f'| 精英te≥2 {row["elite_te_ge2"]} | 种群te≥2 {row["pop_te_ge2"]} '
              f'| 配额注入 {row["quota_injected"]} | {row["eval_s"]:.0f}s', flush=True)
        pop = t7h.evolve_topology_gpu(pop, metrics, cfg, gen=gen, order=order)
    return hist


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[te-ab] device={device}')
    ck = torch.load(os.path.join(ROOT, 'test7h_econ_checkpoint.pth'),
                    map_location='cpu', weights_only=False)
    cfg0 = t7h.Config()
    for k, v in ck.get('config', {}).items():
        if hasattr(cfg0, k) and isinstance(v, (int, float, bool, str)):
            setattr(cfg0, k, v)
    print(f'[te-ab] 从 next_gen={ck["next_gen"]} 双臂各 {GENS} 代 | '
          f'arms={ARMS}')
    result = {}
    for tag, te in ARMS.items():
        print(f'\n[臂 {tag}] TE_ELITE={te}')
        result[tag] = run_arm(tag, te, ck, cfg0, None)
    with open(os.path.join(ROOT, 'results', 'te_quota_ab.json'), 'w',
              encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print('[落盘] results/te_quota_ab.json')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
        for tag, color in (('OFF', 'gray'), ('ON', 'red')):
            h = result[tag]
            g = [r['gen'] for r in h]
            axes[0].plot(g, [r['elite_food'] for r in h], 'o-', color=color,
                         label=f'{tag} EliteFood')
            axes[1].plot(g, [r['elite_te_med'] for r in h], 'o-', color=color,
                         label=f'{tag} elite te med')
            axes[1].plot(g, [r['elite_te_max'] for r in h], '--', color=color,
                         label=f'{tag} elite te max')
            axes[2].plot(g, [r['pop_te_ge2'] for r in h], 'o-', color=color,
                         label=f'{tag} pop te≥2')
            axes[2].plot(g, [r['elite_te_ge2'] for r in h], 's--', color=color,
                         label=f'{tag} elite te≥2')
        axes[0].set_title('EliteFood'); axes[0].legend(fontsize=8)
        axes[1].set_title('elite te'); axes[1].legend(fontsize=8)
        axes[2].set_title('te≥2 counts'); axes[2].legend(fontsize=8)
        for ax in axes:
            ax.set_xlabel('generation')
        plt.tight_layout()
        fig.savefig(os.path.join(ROOT, 'results', 'te_quota_ab.png'), dpi=110)
        print('[落盘] results/te_quota_ab.png')
    except Exception as e:
        print(f'(绘图跳过: {e})')


if __name__ == '__main__':
    main()
