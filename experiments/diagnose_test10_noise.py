# -*- coding: utf-8 -*-
"""
diagnose_test10_noise.py — test10_lunar 噪声主导假设的定量诊断（E1-E3）

背景：训练在 avg≈-140 平台停滞，best 在 -11~144 震荡。假设：2 局评估噪声淹没选择信号
+ 个体间种子不公平 + 悬停局部最优。

实验（全部基于现有 checkpoint，只读，不改训练代码）：
  E0 平台期统计：checkpoint 训练历史近 30 代 avg/best 均值方差（量化用户观测）
  E1 单策略噪声谱：best 克隆 60 份并行各评 1 局（独立种子）→ 单局 reward 分布 +
     终局普查（≥200 及格 / 软着陆 / 坠毁 / 500步截断悬停）+ 主引擎占空比
  E2 克隆零假设检验：best 克隆 256 槽评 4 局独立种子 → k=1/2/3/4 局平均的估计量宽度
     收缩曲线（=现行选择器看到的噪声）；CRN 模式（同代同种子）对照剩余宽度
  E3 精英排序稳定性：checkpoint 种群前 8 名各评 16 局，bootstrap 2/3/4/8/16 局 →
     Kendall-τ 与 top-1 命中率 → 决定第二阶段精评局数 K2

用法: python experiments/diagnose_test10_noise.py
"""
import importlib.util
import json
import os
import random
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location('t10', os.path.join(ROOT, 'experiments', 'test10_lunar', 'test10_lunar.py'))
t10 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(t10)

# 终局类别: 0=及格着陆(>=200) 1=软着陆(term, 0..200) 2=坠毁(term, <0)
#           3=gym截断 4=我们 500 步截断(悬停)
KIND_NAMES = ['及格着陆', '软着陆', '坠毁', 'gym截断', '悬停截断@500']


def run_episodes(pop, cfg, n_eps, seed_mode, max_steps=None, tag=''):
    """驱动 B 个槽位（各含一个个体）各跑 n_eps 局，返回逐局 reward/类别/步数/主引擎次数。"""
    B = pop.P
    dev = pop.device
    half = pop.dtype
    max_steps = max_steps or cfg.MAX_STEPS
    ranges = np.asarray(cfg.OBS_RANGES, dtype=np.float32)[None, :]
    scales = np.asarray(cfg.CHANNEL_SCALES, dtype=np.float32)[None, :]

    import gymnasium as gym
    envs = [gym.make(cfg.ENV_ID) for _ in range(B)]
    rng = random.Random(777 + abs(hash(tag)) % 10000)

    rets = np.zeros((B, n_eps))
    kinds = np.full((B, n_eps), -1, dtype=np.int8)
    steps_rec = np.zeros((B, n_eps))
    main_rec = np.zeros((B, n_eps))

    t0 = __import__('time').perf_counter()
    for ep in range(n_eps):
        if seed_mode == 'crn':
            s = rng.randrange(2 ** 31)
            seeds = [s] * B
        else:
            seeds = [rng.randrange(2 ** 31) for _ in range(B)]
        obs_np = np.stack([e.reset(seed=s)[0] for e, s in zip(envs, seeds)]).astype(np.float32)

        E = torch.zeros(B, pop.N, dtype=half, device=dev)
        I = torch.zeros(B, pop.N, dtype=half, device=dev)
        stt = torch.zeros(B, pop.N, dtype=half, device=dev)
        cts = torch.zeros(B, pop.A, dtype=half, device=dev)
        done = np.zeros(B, dtype=bool)
        ep_ret = np.zeros(B)
        steps = np.zeros(B, dtype=np.int64)
        main = np.zeros(B, dtype=np.int64)

        for t in range(max_steps):
            o = ((obs_np / ranges) * scales).astype(np.float32)
            obs = torch.from_numpy(o).to(dev, dtype=half)
            act, E, I, stt = t10.deliberate_batch(pop, obs, E, I, stt, cts, cfg)
            cts = t10.update_fatigue(cts, act)
            act_np = act.cpu().numpy()
            for i in range(B):
                if done[i]:
                    continue
                a = int(act_np[i])
                if a == 2:
                    main[i] += 1
                o2, r, term, trunc, _ = envs[i].step(a)
                ep_ret[i] += r
                steps[i] += 1
                if term or trunc:
                    done[i] = True
                    rets[i, ep] = ep_ret[i]
                    steps_rec[i, ep] = steps[i]
                    main_rec[i, ep] = main[i]
                    if term:
                        if ep_ret[i] >= cfg.SUCCESS_REWARD:
                            kinds[i, ep] = 0
                        elif ep_ret[i] >= 0:
                            kinds[i, ep] = 1
                        else:
                            kinds[i, ep] = 2
                    else:
                        kinds[i, ep] = 3
                else:
                    obs_np[i] = o2
            if done.all():
                break
        for i in range(B):
            if not done[i]:
                rets[i, ep] = ep_ret[i]
                steps_rec[i, ep] = steps[i]
                main_rec[i, ep] = main[i]
                kinds[i, ep] = 4
        if (ep + 1) % 4 == 0:
            print(f"  [{tag}] {ep + 1}/{n_eps} 局完成 "
                  f"({__import__('time').perf_counter() - t0:.0f}s)")

    for e in envs:
        e.close()
    return rets, kinds, steps_rec, main_rec


def clones_pop(cfg, st, B, device):
    pop = t10.GeneStack(cfg, B=B, device=device)
    pop.random_init()
    for i in range(B):
        pop.set_individual_from_state(i, st)
    pop.refresh_eff()
    if cfg.USE_FP16:
        pop.fp16()
        pop.refresh_eff()
    return pop


def main():
    ck_path = os.path.join(ROOT, 'test10_lunar_checkpoint.pth')
    if not os.path.exists(ck_path):
        print('未找到 test10_lunar_checkpoint.pth')
        sys.exit(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = t10.Config()
    cfg.DEVICE = str(device)

    print(f'加载 checkpoint（{ck_path}）...')
    ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    results = {}

    # ---- E0 平台期统计 ----
    h = ck['history']
    ar = np.asarray(h['avg_reward'])
    br = np.asarray(h['best_reward'])
    t = slice(max(0, len(ar) - 30), len(ar))
    e0 = {
        'gens_done': int(ck['next_gen']),
        'avg_mean_30': float(ar[t].mean()), 'avg_std_30': float(ar[t].std()),
        'best_mean_30': float(br[t].mean()), 'best_std_30': float(br[t].std()),
        'best_min_30': float(br[t].min()), 'best_max_30': float(br[t].max()),
        'pop_gap_30': float((br[t] - ar[t]).mean()),
    }
    results['E0_plateau'] = e0
    print(f"\n[E0] 已完成 {e0['gens_done']} 代 | 近30代 avg={e0['avg_mean_30']:.1f}±{e0['avg_std_30']:.1f} "
          f"best={e0['best_mean_30']:.1f}±{e0['best_std_30']:.1f} "
          f"范围[{e0['best_min_30']:.0f},{e0['best_max_30']:.0f}] "
          f"种群 best-avg 差距={e0['pop_gap_30']:.1f}")

    # ---- E1 单策略噪声谱（best 克隆 ×60 局）----
    print('\n[E1] best 策略 60 局噪声谱...')
    st_best = ck['best_state']
    pop1 = clones_pop(cfg, st_best, 60, device)
    r1, k1, s1, m1 = run_episodes(pop1, cfg, 1, 'indep', tag='E1')
    rets = r1[:, 0]
    kinds = k1[:, 0]
    steps = s1[:, 0]
    mains = m1[:, 0]
    census = {KIND_NAMES[k]: int((kinds == k).sum()) for k in range(5)}
    band = {}
    for k in range(5):
        m = kinds == k
        if m.any():
            band[KIND_NAMES[k]] = {'n': int(m.sum()),
                                   'mean': float(rets[m].mean()),
                                   'min': float(rets[m].min()), 'max': float(rets[m].max())}
    duty = float((mains / np.clip(steps, 1, None)).mean())
    e1 = {
        'mean': float(rets.mean()), 'std': float(rets.std()),
        'p10': float(np.percentile(rets, 10)), 'p50': float(np.percentile(rets, 50)),
        'p90': float(np.percentile(rets, 90)),
        'sem_2ep': float(rets.std() / np.sqrt(2)),
        'census': census, 'band': band,
        'main_duty': duty, 'mean_steps': float(steps.mean()),
    }
    results['E1_single_policy'] = e1
    print(f"  reward mean={e1['mean']:.1f} std={e1['std']:.1f} "
          f"P10/50/90 = {e1['p10']:.0f}/{e1['p50']:.0f}/{e1['p90']:.0f}")
    print(f"  2局评估 SEM ≈ {e1['sem_2ep']:.1f}")
    print(f"  终局普查: {census}")
    for kk, vv in band.items():
        print(f"    {kk}: n={vv['n']} mean={vv['mean']:.1f} 范围[{vv['min']:.0f},{vv['max']:.0f}]")
    print(f"  主引擎占空比={duty:.2f} 平均步数={e1['mean_steps']:.0f}")

    # ---- E2 克隆零假设（256 槽 ×4 局独立 + CRN 对照）----
    print('\n[E2] 克隆 256 槽 × 4 局独立种子（现行方案的估计量噪声）...')
    pop2 = clones_pop(cfg, st_best, 256, device)
    r2, _, _, _ = run_episodes(pop2, cfg, 4, 'indep', tag='E2-indep')
    e2 = {'est_std': {}, 'est_width': {}}
    blocks = {1: [(0,), (1,), (2,), (3,)],
              2: [(0, 1), (2, 3)],
              3: [(0, 1, 2), (1, 2, 3)],
              4: [(0, 1, 2, 3)]}
    for k, idxs in blocks.items():
        ests = np.concatenate([rets_k.mean(axis=1) for rets_k in
                               (r2[:, list(ix)] for ix in idxs)])
        e2['est_std'][str(k)] = float(ests.std())
        e2['est_width'][str(k)] = float(np.percentile(ests, 95) - np.percentile(ests, 5))
    results['E2_clone_null'] = e2
    print('  k局平均估计量: ' + ' | '.join(
        f"k={k} std={e2['est_std'][str(k)]:.1f} (P95-P5={e2['est_width'][str(k)]:.1f})"
        for k in (1, 2, 3, 4)))

    print('  CRN 对照（同代同种子）×2 局...')
    r2c, _, _, _ = run_episodes(pop2, cfg, 2, 'crn', tag='E2-crn')
    crn_spread = {str(ep): float(r2c[:, ep].std()) for ep in range(2)}
    e2['crn_spread'] = crn_spread
    print(f"  CRN 同种子下 256 槽分布 std: {[f'{v:.3f}' for v in crn_spread.values()]} "
          f"(理论≈0，残差=fp16 非确定性)")

    # ---- E3 精英排序稳定性（前 8 名 × 16 局）----
    print('\n[E3] checkpoint 前 8 名精英 × 16 局...')
    packed = ck['pop']
    sub = {g: packed[g][:8].clone() for g in t10.GeneStack.GENES}
    pop3 = t10.GeneStack(cfg, B=8, device=device)
    pop3.unpack(sub)
    pop3.refresh_eff()
    r3, k3, _, _ = run_episodes(pop3, cfg, 16, 'indep', tag='E3')
    true_mean = r3.mean(axis=1)
    ref_rank = np.argsort(-true_mean)
    e3 = {'elite_16ep_mean': [float(x) for x in true_mean],
          'metrics': {}}
    rng = random.Random(42)
    n_boot = 2000
    for k in (2, 3, 4, 8, 16):
        taus, top1 = [], []
        for _ in range(n_boot):
            idx = rng.choices(range(16), k=k)
            est = r3[:, idx].mean(axis=1)
            rank = np.argsort(-est)
            # top-1 命中
            top1.append(rank[0] == ref_rank[0])
            # Kendall tau（8 个体 O(64)）
            conc = disc = 0
            for a in range(8):
                for b in range(a + 1, 8):
                    sa, sb = est[a] - est[b], true_mean[a] - true_mean[b]
                    if sa * sb > 0:
                        conc += 1
                    elif sa * sb < 0:
                        disc += 1
            taus.append((conc - disc) / (conc + disc))
        e3['metrics'][str(k)] = {'tau': float(np.mean(taus)), 'top1': float(np.mean(top1))}
    results['E3_rank_stability'] = e3
    print(f"  8 精英 16 局真实均值: {np.round(true_mean, 1)}")
    for k in (2, 3, 4, 8, 16):
        mm = e3['metrics'][str(k)]
        print(f"  k={k:>2}: Kendall-τ={mm['tau']:.3f}  top-1 命中率={mm['top1']:.3f}")

    # ---- 汇总对照 ----
    print('\n===== 汇总 =====')
    print(f"种群可用信号（近30代 best-avg 均值差距）: {e0['pop_gap_30']:.1f} 分")
    print(f"现行 2 局估计量噪声 std: {e2['est_std']['2']:.1f} 分  "
          f"(2048 抽样纯噪声极值≈{e2['est_std']['2'] * 3.2:.0f} 分)")
    print(f"单局 σ: {e1['std']:.1f} | 悬停+坠毁占比: "
          f"{(census.get('悬停截断@500', 0) + census.get('坠毁', 0)) / 60:.0%}")

    out = os.path.join(ROOT, 'results', 'test10_noise_diagnosis.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'\n结果已保存 -> {out}')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        colors = ['green', 'olive', 'red', 'gray', 'orange']
        for k in range(5):
            m = kinds == k
            if m.any():
                axes[0].hist(rets[m], bins=20, alpha=0.6, label=KIND_NAMES[k], color=colors[k])
        axes[0].axvline(200, color='green', ls='--', alpha=0.5)
        axes[0].set_title('E1: best 策略 60 局 reward 分布')
        axes[0].set_xlabel('episode reward')
        axes[0].legend(fontsize=8)
        ks = [1, 2, 3, 4]
        axes[1].plot(ks, [e2['est_std'][str(k)] for k in ks], 'o-', label='实测估计量 std')
        sig = e1['std'] / np.sqrt(ks)
        axes[1].plot(ks, sig, 'k--', alpha=0.5, label=r'$\sigma/\sqrt{k}$ 理论')
        axes[1].axhline(e0['pop_gap_30'], color='red', ls=':', label='种群 best-avg 差距')
        axes[1].set_title('E2: 克隆零假设 — 噪声 vs 局数')
        axes[1].set_xlabel('评估局数 k')
        axes[1].legend(fontsize=8)
        ks3 = [2, 3, 4, 8, 16]
        axes[2].plot(ks3, [e3['metrics'][str(k)]['tau'] for k in ks3], 's-', label='Kendall-τ')
        axes[2].plot(ks3, [e3['metrics'][str(k)]['top1'] for k in ks3], 'o-', label='top-1 命中率')
        axes[2].axhline(0.9, color='gray', ls=':', alpha=0.6)
        axes[2].set_title('E3: 精英排序稳定性 vs 局数')
        axes[2].set_xlabel('评估局数 k')
        axes[2].legend(fontsize=8)
        plt.tight_layout()
        png = out.replace('.json', '.png')
        fig.savefig(png, dpi=100)
        plt.close(fig)
        print(f'图已保存 -> {png}')
    except Exception as e:
        print(f'(绘图跳过: {e})')


if __name__ == '__main__':
    main()
