# ==========================================
# test11 Phase 0 —— 筛选保真度离线实验（只读 checkpoint，不训练）
#
# 问题：给定同一批真实个体，锦标赛淘汰制选出的精英集，是否与现行两阶段
# 筛选一样好（甚至更好/更稳）？——在训练之前用统计手段回答，省掉盲跑。
#
# 协议（预注册，docs/test11_experiment_log.md）：
#   1. 加载真实 checkpoint 种群（不改动任何基因）；
#   2. 真值：全体 ×GT 库（gen=999, stage=0）CRN 评估 → 每个体真值 food/适应度；
#   3. T 个 trial（gen=5000+t），每 trial 用 test11 真实选择器代码模拟 4 种筛选：
#      two_stage / tourn_random / tourn_rank / tourn_final（=random+精英终轮加赛）；
#      库按 (gen,stage,E) 规范化 + (库,个体集) 记忆化——首轮 K=3 库三者天然
#      共享（同 gen 同 stage），每 trial 实际新增评估 ≈2 万局次而非 4.3 万；
#   4. 指标：选中精英集的真值 food 均值（主）、与真值 top-N 重合率、
#      真 top-16 覆盖、选中集内 Kendall τ（选择序 vs 真值序）；
#   5. 门判定（对 tourn_random vs two_stage 的逐 trial 配对差）：
#      配对均值 ≥ -0.3 → PASS(0)；[-0.5,-0.3) → RISKY(0，打印警告)；
#      < -0.5 → STOP(2)，调用方（run_test11_ab.bat）应停止训练臂。
#
# 用法（须等 GPU 空闲，如 test7h 103-153 训练结束后）：
#   python experiments/test11_selection_fidelity.py \
#       --checkpoint test7h_econ_checkpoint.pth --tag t7h --trials 8 --gt-banks 24
#   python experiments/test11_selection_fidelity.py \
#       --checkpoint "test7g_econ_checkpoint copy.pth" --tag t7g --trials 4 --gt-banks 16
# ==========================================
import argparse
import copy
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments', 'test11'))  # 归档后模块路径
import test11 as t11  # noqa: E402


def cfg_from_ckpt(ck):
    """从 checkpoint 的 config 快典重建 cfg：test11 缺省值打底，旧键覆盖；
    旧版本（7g）没有的新键保持 test11 默认，WEAK_MASK_FRAC 缺失视为 0
    （与该种群训练时的评估口径一致）。"""
    cfg = t11.Config()
    stored = ck.get('config', {}) or {}
    for k, v in stored.items():
        if k.isupper() and not k.startswith('_'):
            try:
                setattr(cfg, k, v)
            except Exception:
                pass
    cfg.AUTO_RESUME = False
    if 'WEAK_MASK_FRAC' not in stored:
        cfg.WEAK_MASK_FRAC = 0.0
    return cfg


class _IdView:
    """包装 GeneStack 并携带全局序号：list 索引 → 嵌套视图（选择器用），
    tensor 索引 → 映射回全局序号后的真切片（_eval_pop_banks 分块用）。"""

    def __init__(self, base, ids):
        self.base = base
        self.ids = list(ids)
        self.P = len(self.ids)
        self.device = base.device

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            g = torch.tensor([self.ids[i] for i in idx.tolist()],
                             dtype=torch.long, device=self.base.device)
            return self.base[g]
        if isinstance(idx, (list, tuple)):
            return _IdView(self.base, [self.ids[i] for i in idx])
        return _IdView(self.base, [self.ids[idx]])

    def fp16(self):        # 基座已 fp16，幂等空操作
        pass


def install_cached_eval():
    """库规范化（同 (CRN_SEED,gen,stage,E) 复用同一组 bank 对象）+
    (bank 对象组, 个体全局序号) 级记忆化。返回 (原始评估函数, 记忆化包装,
    bank 规范化, 统计 dict)。恢复时把 t11.make_banks/_eval_pop_banks 还原。"""
    stats = {'slots': 0, 'hits': 0, 'miss': 0}
    orig_make_banks = t11.make_banks
    orig_eval = t11._eval_pop_banks
    bank_canon = {}
    cache = {}

    def canon_make_banks(cfg, gen, stage, episodes, device):
        key = (int(cfg.CRN_SEED), int(gen), int(stage), int(episodes))
        if key not in bank_canon:
            bank_canon[key] = orig_make_banks(cfg, gen, stage, episodes, device)
        return bank_canon[key]

    def cached_eval(pop, cfg, banks):
        ids = tuple(getattr(pop, 'ids', range(pop.P)))
        key = (tuple(id(b) for b in banks), ids)
        if key in cache:
            stats['hits'] += 1
            return cache[key]
        stats['miss'] += 1
        stats['slots'] += pop.P * len(banks)
        out = orig_eval(pop, cfg, banks)
        cache[key] = out
        return out

    t11.make_banks = canon_make_banks
    t11._eval_pop_banks = cached_eval
    return orig_eval, cached_eval, canon_make_banks, stats


def kendall_tau(rank_a, rank_b):
    """rank_x[i] = 个体 i 在口径 x 下的名次（0=最好）。O(n²) 向量化。"""
    a = list(rank_a)
    b = list(rank_b)
    n = len(a)
    iu = [(i, j) for i in range(n) for j in range(i + 1, n)]
    s = sum(1 if (a[i] - a[j]) * (b[i] - b[j]) > 0
            else (-1 if (a[i] - a[j]) * (b[i] - b[j]) < 0 else 0)
            for i, j in iu)
    return s / (n * (n - 1) / 2)


def run_selector(view, cfg_base, mode, pairing, final_pass, gen_t, elite):
    cfg = copy.copy(cfg_base)
    cfg.SELECTION_MODE = mode
    cfg.TOURN_PAIRING = pairing
    cfg.TOURN_FINAL_PASS = final_pass
    _, order = t11.evaluate_population_gpu(view, cfg, gen=gen_t)
    return list(order[:elite])


def main():
    ap = argparse.ArgumentParser(description='test11 Phase 0 筛选保真度（只读）')
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--tag', type=str, default='t7h')
    ap.add_argument('--trials', type=int, default=8)
    ap.add_argument('--gt-banks', type=int, default=24)
    ap.add_argument('--gen-base', type=int, default=5000)
    ap.add_argument('--device', type=str, default='auto')
    args = ap.parse_args()

    dev = torch.device(args.device) if args.device != 'auto' else (
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f'[Phase0] checkpoint = {args.checkpoint} | device = {dev} | '
          f'trials = {args.trials} | gt_banks = {args.gt_banks}')

    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = cfg_from_ckpt(ck)
    elite = int(cfg.ELITE_SIZE)
    pop = t11.GeneStack(cfg, device=dev)
    pop.unpack(ck['pop'])
    if getattr(cfg, 'USE_FP16', True):
        pop.fp16()
    P = pop.P
    print(f'[Phase0] 种群 P={P} | 精英 N={elite} | next_gen={ck.get("next_gen")} | '
          f'FITNESS_VERSION={getattr(cfg, "FITNESS_VERSION", 1)} | '
          f'WEAK_MASK={getattr(cfg, "WEAK_MASK_FRAC", 0):.2f}')

    orig_eval, cached_eval, canon_make_banks, stats = install_cached_eval()

    t0 = time.perf_counter()
    # ---- 真值：全体 ×GT 库 ----
    view = _IdView(pop, list(range(P)))
    banks_gt = canon_make_banks(cfg, 999, 0, args.gt_banks, dev)
    m_gt = orig_eval(view, cfg, banks_gt)
    true_food = m_gt[:, 0].numpy()
    key_fn = t11._make_key_fn(cfg)
    mnp = m_gt.numpy()
    true_fit = [float(key_fn(mnp[i])) for i in range(P)]
    gt_order = sorted(range(P), key=lambda i: true_fit[i], reverse=True)
    true_rank = [0] * P
    for r, i in enumerate(gt_order):
        true_rank[i] = r
    gt_top = set(gt_order[:elite])
    gt_top16 = set(gt_order[:16])
    print(f'[Phase0] 真值完成（{time.perf_counter() - t0:.0f}s）：'
          f'food mean={true_food.mean():.2f} std={true_food.std():.2f} | '
          f'top-{elite} 真值均值={true_food[list(gt_top)].mean():.2f} | '
          f'top-16 均值={true_food[list(gt_top16)].mean():.2f}')

    selectors = [
        ('two_stage', 'two_stage', 'random', False),
        ('tourn_random', 'tournament', 'random', False),
        ('tourn_rank', 'tournament', 'rank', False),
        ('tourn_final', 'tournament', 'random', True),
    ]
    sel_ids = {name: [] for name, *_ in selectors}
    for tr in range(args.trials):
        t1 = time.perf_counter()
        s0 = stats['slots']
        gen_t = args.gen_base + tr
        for name, mode, pairing, fpass in selectors:
            sel_ids[name].append(run_selector(view, cfg, mode, pairing, fpass,
                                              gen_t, elite))
        print(f'[Phase0] trial {tr + 1}/{args.trials} 完成 '
              f'({time.perf_counter() - t1:.0f}s, 新增局次 {stats["slots"] - s0})')

    # ---- 指标汇总 ----
    def metrics_of(ids_list):
        foods = [float(true_food[ids].mean()) for ids in ids_list]
        overlaps = [len(set(ids) & gt_top) / elite for ids in ids_list]
        t16 = [len(set(ids) & gt_top16) / 16 for ids in ids_list]
        # tau：选择器第 i 顺位选中者的真值名次序列 vs 完美序列——选择序与
        # 真值序完全一致时=1（注意不能先 sorted，否则恒为 1）
        taus = [kendall_tau(list(range(elite)),
                            [true_rank[i] for i in ids])
                for ids in ids_list]
        return foods, overlaps, t16, taus

    result = {
        'meta': {
            'checkpoint': args.checkpoint, 'tag': args.tag, 'P': P, 'elite': elite,
            'gt_banks': args.gt_banks, 'trials': args.trials,
            'gen_base': args.gen_base, 'device': str(dev),
            'fitness_version': getattr(cfg, 'FITNESS_VERSION', 1),
            'weak_mask_frac': getattr(cfg, 'WEAK_MASK_FRAC', 0.0),
            'crn_seed': int(cfg.CRN_SEED),
            'cache_slots_total': stats['slots'],
            'cache_hits': stats['hits'], 'cache_miss': stats['miss'],
            'elapsed_s': None,
        },
        'truth': {
            'food_mean': float(true_food.mean()), 'food_std': float(true_food.std()),
            'gt_top_food_mean': float(true_food[list(gt_top)].mean()),
            'gt_top16_food_mean': float(true_food[list(gt_top16)].mean()),
        },
        'selectors': {},
    }
    for name, *_ in selectors:
        foods, overlaps, t16, taus = metrics_of(sel_ids[name])
        result['selectors'][name] = {
            'food_mean_by_trial': foods,
            'food_mean': sum(foods) / len(foods),
            'food_std': (sum((f - sum(foods) / len(foods)) ** 2 for f in foods)
                         / (len(foods) - 1)) ** 0.5 if len(foods) > 1 else 0.0,
            'overlap_mean': sum(overlaps) / len(overlaps),
            'top16cov_mean': sum(t16) / len(t16),
            'tau_mean': sum(taus) / len(taus),
        }
    # 逐 trial 配对差（各选择器 − two_stage，同 trial 同库布局）
    ref = result['selectors']['two_stage']['food_mean_by_trial']
    for name, *_ in selectors:
        if name == 'two_stage':
            continue
        diffs = [a - b for a, b in
                 zip(result['selectors'][name]['food_mean_by_trial'], ref)]
        result['selectors'][name]['paired_diff_by_trial'] = diffs
        result['selectors'][name]['paired_diff_mean'] = sum(diffs) / len(diffs)

    result['meta']['elapsed_s'] = time.perf_counter() - t0

    # ---- 门判定（主选择器 tourn_random vs two_stage）----
    pd = result['selectors']['tourn_random']['paired_diff_mean']
    print('\n===== Phase 0 汇总（选中精英集真值 food）=====')
    print(f'{"selector":<14}{"food_mean":>10}{"±std":>8}{"overlap":>9}'
          f'{"top16cov":>10}{"tau":>8}{"pairedΔ":>10}')
    for name, *_ in selectors:
        r = result['selectors'][name]
        pdd = f"{r.get('paired_diff_mean', 0):+.3f}" if name != 'two_stage' else '  ref'
        print(f'{name:<14}{r["food_mean"]:>10.3f}{r["food_std"]:>8.3f}'
              f'{r["overlap_mean"]:>9.3f}{r["top16cov_mean"]:>10.3f}'
              f'{r["tau_mean"]:>8.3f}{pdd:>10}')
    if pd >= -0.3:
        verdict, code = 'PASS', 0
    elif pd >= -0.5:
        verdict, code = 'RISKY', 0
    else:
        verdict, code = 'STOP', 2
    result['gate'] = {'paired_diff_mean': pd, 'verdict': verdict}
    print(f'\n[门判定] tourn_random − two_stage 配对差 = {pd:+.3f} → {verdict} '
          f'(阈值: ≥-0.3 PASS / [-0.5,-0.3) RISKY / <-0.5 STOP)')

    os.makedirs('results', exist_ok=True)
    out_json = f'results/test11_selection_fidelity_{args.tag}.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f'[Phase0] 结果已写入 {out_json} '
          f'(总耗时 {result["meta"]["elapsed_s"]:.0f}s, 实际评估局次 {stats["slots"]})')

    # ---- PNG ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        names = [n for n, *_ in selectors]
        xs = range(len(names))
        means = [result['selectors'][n]['food_mean'] for n in names]
        stds = [result['selectors'][n]['food_std'] for n in names]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        ax1.bar(xs, means, yerr=stds, capsize=4,
                color=['gray', 'tab:red', 'tab:orange', 'tab:purple'])
        ax1.set_xticks(list(xs), names, rotation=15)
        ax1.set_ylabel('selected elite true food (GT banks)')
        ax1.set_title(f'Selection fidelity — {args.tag} (P={P}, elite={elite})')
        ax1.grid(True, axis='y', alpha=0.3)
        ovs = [result['selectors'][n]['overlap_mean'] for n in names]
        ax2.bar(xs, ovs, color=['gray', 'tab:red', 'tab:orange', 'tab:purple'])
        ax2.set_xticks(list(xs), names, rotation=15)
        ax2.set_ylabel(f'overlap with true top-{elite}')
        ax2.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        out_png = f'results/test11_selection_fidelity_{args.tag}.png'
        fig.savefig(out_png, dpi=100)
        plt.close(fig)
        print(f'[Phase0] 图已保存 {out_png}')
    except Exception as e:  # noqa: BLE001
        print(f'(matplotlib 跳过: {e})')

    sys.exit(code)


if __name__ == '__main__':
    main()
