# ==========================================
# experiments/imitation_ab.py —— 模仿引导适应度 3 臂 × 10 代验证（GPU，全内存）
#
# 假设：te-配额 28 代证明精英池内重组无法转型（B3X：策略-身体共适应，
# 中途换风格必死）→ 需要稠密行为梯度。教师=VectorCycleTeacher（40/40 通关，
# 向量化，评估内嵌零额外扫描）。
# 臂设计（同代同 CRN 库， banks 按 gen 对齐）：
#   A 对照：现 checkpoint 种群，IMITATION_W=0
#   B：    现 checkpoint 种群，IMITATION_W=2.0 —— 测"能否把现有脑拉出混沌"
#   C：    随机初始化，      IMITATION_W=2.0 —— 测"从出生模仿"（避开混沌吸引子）
# 判定（预注册）：B 或 C 在 10 代内精英 mismatch 0.65→<0.45 且精英 te 中位
# ≥1.35、EliteFood ≥33 → 模仿通道有效；C 起效 B 不起效 → 混沌吸引子为硬
# 阻力（转型须从出生塑形）；均无效 → 转岛模型路线。
# 产出：results/imitation_ab.json + imitation_ab.png
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
BASE_GEN = 130          # 与现 checkpoint 的 next_gen 对齐 → 同代同 CRN 库


def te_of(mn, cap):
    return np.where((mn[:, 0] > 0) & (mn[:, 10] > 0),
                    np.minimum(mn[:, 3] / np.maximum(mn[:, 10], 1.0), cap), 0.0)


def run_arm(tag, pop, cfg, start_gen):
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
            avg_food=float(np.mean(mn[:, 0])),
            best_food=float(np.max(mn[:, 0])),
            elite_te_med=float(np.median(te_elite)),
            elite_te_max=float(np.max(te_elite)),
            elite_mis=float(np.mean(mn[elite_idx, 12])),
            pop_mis=float(np.mean(mn[:, 12])),
            pop_te_ge2=int((te_of(mn, cap) >= 2).sum()),
            eval_s=time.time() - t0)
        hist.append(row)
        print(f'  [{tag}] gen{gen}: EliteFood {row["elite_food"]:.2f} | '
              f'BestFood {row["best_food"]:.1f} | 精英te中位 {row["elite_te_med"]:.2f} '
              f'| 精英mis {row["elite_mis"]:.2f} | 种群mis {row["pop_mis"]:.2f} | '
              f'{row["eval_s"]:.0f}s', flush=True)
        if gen < start_gen + GENS - 1:
            pop = t7h.evolve_topology_gpu(pop, metrics, cfg, gen=gen, order=order)
    return hist, pop


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[imitation-ab] device={device}')
    ck = torch.load(os.path.join(ROOT, 'test7h_econ_checkpoint.pth'),
                    map_location='cpu', weights_only=False)
    cfg0 = t7h.Config()
    for k, v in ck.get('config', {}).items():
        if hasattr(cfg0, k) and isinstance(v, (int, float, bool, str)):
            setattr(cfg0, k, v)
    cfg0.IMITATION_W = 0.0
    start_gen = int(ck['next_gen'])
    print(f'[imitation-ab] 基准 gen={BASE_GEN}（checkpoint next_gen={start_gen}）| '
          f'{GENS} 代/臂 | TE_ELITE={cfg0.TE_ELITE} IMITATION_W arms: A=0/B=2/C=2(rand)')

    result = {}
    # ---- A 对照 ----
    print('\n[臂 A] checkpoint + W=0（对照）')
    popA = t7h.GeneStack(cfg0, device=device)
    popA.unpack(ck['pop'])
    histA, _ = run_arm('A', popA, cfg0, start_gen)
    result['A'] = histA
    # ---- B checkpoint + 模仿 ----
    print('\n[臂 B] checkpoint + W=2.0')
    cfgB = copy.copy(cfg0)
    cfgB.IMITATION_W = 2.0
    popB = t7h.GeneStack(cfgB, device=device)
    popB.unpack(ck['pop'])
    histB, popB_end = run_arm('B', popB, cfgB, start_gen)
    result['B'] = histB
    # ---- C 随机 + 模仿 ----
    print('\n[臂 C] 随机初始化 + W=2.0')
    cfgC = copy.copy(cfgB)
    torch.manual_seed(20260828)
    popC = t7h.GeneStack(cfgC, device=device)
    popC.random_init()
    histC, popC_end = run_arm('C', popC, cfgC, BASE_GEN)
    result['C'] = histC

    with open(os.path.join(ROOT, 'results', 'imitation_ab.json'), 'w',
              encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print('\n[落盘] results/imitation_ab.json')

    # ---- 保存 B/C 末代种群（供后续长跑续接）----
    for tag, pop, cfg in (('B', popB_end, cfgB), ('C', popC_end, cfgC)):
        payload = {
            'next_gen': BASE_GEN + GENS,
            'pop': pop.pack(),
            'history': {'gen': [r['gen'] for r in result[tag]],
                        'best_food': [r['best_food'] for r in result[tag]],
                        'avg_food': [r['avg_food'] for r in result[tag]],
                        'best_seen': [], 'best_unseen': [],
                        'elite_food': [r['elite_food'] for r in result[tag]],
                        'best_fit': [], 'best_turneff': []},
            'config': t7h._cfg_dict(cfg),
            'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        torch.save(payload, os.path.join(ROOT, f'test7h_imit_{tag}.pth'))
        print(f'[落盘] test7h_imit_{tag}.pth（{tag} 臂末代种群）')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
        for tag, color in (('A', 'gray'), ('B', 'red'), ('C', 'blue')):
            h = result[tag]
            g = [r['gen'] for r in h]
            axes[0].plot(g, [r['elite_mis'] for r in h], 'o-', color=color,
                         label=f'{tag} elite mismatch')
            axes[1].plot(g, [r['elite_te_med'] for r in h], 'o-', color=color,
                         label=f'{tag} elite te med')
            axes[2].plot(g, [r['elite_food'] for r in h], 'o-', color=color,
                         label=f'{tag} EliteFood')
        axes[0].axhline(0.45, color='k', ls=':', lw=0.8)
        axes[0].set_title('elite action-mismatch vs teacher')
        axes[1].axhline(1.35, color='k', ls=':', lw=0.8)
        axes[1].set_title('elite te median')
        axes[2].set_title('EliteFood')
        for ax in axes:
            ax.set_xlabel('generation'); ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(os.path.join(ROOT, 'results', 'imitation_ab.png'), dpi=110)
        print('[落盘] results/imitation_ab.png')
    except Exception as e:
        print(f'(绘图跳过: {e})')


if __name__ == '__main__':
    main()
