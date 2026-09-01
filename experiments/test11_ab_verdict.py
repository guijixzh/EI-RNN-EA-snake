# ==========================================
# test11 终局验收 —— A/B 两臂配对复评 + 预注册判定表
#
# 输入：两臂产物 test11_econ_{tag}_checkpoint.pth / _history.json
#   （A 臂 tag=two_stage，B 臂 tag=tourn_k3，见 run_test11_ab.bat）
# 步骤：
#   1. 曲线指标：终值/尾5均值 EliteFood、达阈值代数、avg_food 冻结窗口、
#      EliteFood 最大回撤、每代评估耗时（cum_eval_time/代数）；
#   2. 精英集复原：各臂 checkpoint 的最终种群 + 该臂自己的筛选器在
#      gen=next_gen-1 重放（CRN 确定性 → 与训练末代选择逐位一致）；
#   3. 同 40 CRN 库配对复评：两精英集逐库得分（库内=精英平均适应度/food），
#      margin_b = B(b) − A(b) → 胜率/均值/σ；
#      预注册（沿 solver_reference 范式）：
#        B 显著优 = 胜率 ≥ 90% 且 margin ≥ +1.5；等效 = |margin| < 0.5；
#        B 更差 = 胜率 ≤ 10% 且 margin ≤ −1.5；其余 → 不确定需复核。
#   4. best_model 单体配对复评（辅助证据）。
# 输出：results/test11_ab_verdict.json + 曲线/margin 图 + 控制台判定表。
# 用法（两臂训练完成后）：
#   python experiments/test11_ab_verdict.py --gens 20
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
    cfg = t11.Config()
    stored = ck.get('config', {}) or {}
    for k, v in stored.items():
        if k.isupper() and not k.startswith('_'):
            try:
                setattr(cfg, k, v)
            except Exception:
                pass
    cfg.AUTO_RESUME = False
    return cfg


def load_arm(tag, dev):
    """返回 (cfg, pop, ck, history)。"""
    base = f'test11_econ_{tag}'
    with open(base + '_history.json', encoding='utf-8') as f:
        hist = json.load(f)
    ck = torch.load(base + '_checkpoint.pth', map_location='cpu', weights_only=False)
    cfg = cfg_from_ckpt(ck)
    cfg.SELECTION_MODE = 'tournament' if tag.startswith('tourn') else 'two_stage'
    pop = t11.GeneStack(cfg, device=dev)
    pop.unpack(ck['pop'])
    if getattr(cfg, 'USE_FP16', True):
        pop.fp16()
    return cfg, pop, ck, hist


def curve_stats(hist, theta):
    gen = hist['gen']
    elite = [x if x is not None else float('nan') for x in hist['elite_food']]
    avg = hist['avg_food']
    n = len(gen)
    tail5 = sum(x for x in elite[-5:]) / max(len(elite[-5:]), 1)
    g2t = None
    for i, x in enumerate(elite):
        if x == x and x >= theta:
            g2t = gen[i]
            break
    freeze10 = 0
    for i in range(n - 10):
        if avg[i + 10] - avg[i] < 0.1:
            freeze10 += 1
    peak, dd = float('-inf'), 0.0
    for x in elite:
        peak = max(peak, x)
        dd = max(dd, peak - x)
    return {'final_elite_food': elite[-1], 'tail5_elite_food': tail5,
            'gens_to_theta': g2t, 'theta': theta,
            'avg_freeze10_windows': freeze10, 'elite_max_drawdown': dd,
            'n_gens': n}


def main():
    ap = argparse.ArgumentParser(description='test11 A/B 终局验收')
    ap.add_argument('--gens', type=int, default=20)
    ap.add_argument('--tag-a', type=str, default='two_stage')
    ap.add_argument('--tag-b', type=str, default='tourn_k3')
    ap.add_argument('--gt-banks', type=int, default=40)
    ap.add_argument('--win-threshold', type=float, default=0.90)
    ap.add_argument('--margin-threshold', type=float, default=1.5)
    ap.add_argument('--equiv-threshold', type=float, default=0.5)
    ap.add_argument('--device', type=str, default='auto')
    args = ap.parse_args()

    dev = torch.device(args.device) if args.device != 'auto' else (
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f'[Verdict] A={args.tag_a}  B={args.tag_b} | device={dev} | '
          f'GT banks={args.gt_banks}')
    t0 = time.perf_counter()

    cfgA, popA, ckA, histA = load_arm(args.tag_a, dev)
    cfgB, popB, ckB, histB = load_arm(args.tag_b, dev)
    elite_n = int(cfgA.ELITE_SIZE)
    gens_done = min(len(histA['gen']), len(histB['gen']))
    if len(histA['gen']) != len(histB['gen']):
        print(f'  警告：两臂代数不同 A={len(histA["gen"])} B={len(histB["gen"])}，'
              f'曲线对比取前 {gens_done} 代')

    # ---- 1. 曲线 ----
    eliteA = [x if x is not None else float('nan') for x in histA['elite_food']]
    theta = min(30.0, eliteA[-1])
    statsA = curve_stats(histA, theta)
    statsB = curve_stats(histB, theta)
    evalA = float(ckA.get('cum_eval_time', 0)) / max(int(ckA.get('next_gen', 1)), 1)
    evalB = float(ckB.get('cum_eval_time', 0)) / max(int(ckB.get('next_gen', 1)), 1)

    # ---- 2. 精英集复原（各臂自己的筛选器重放末代）----
    gen_last = int(ckA['next_gen']) - 1
    _, orderA = t11.evaluate_population_gpu(popA, copy.copy(cfgA), gen=gen_last)
    _, orderB = t11.evaluate_population_gpu(popB, copy.copy(cfgB), gen=gen_last)
    selA, selB = list(orderA[:elite_n]), list(orderB[:elite_n])
    inter = len(set(selA) & set(selB))
    print(f'[Verdict] 末代(gen={gen_last})精英集复原完成：'
          f'|A∩B|={inter}/{elite_n}（重合率 {inter / elite_n:.1%}）')

    # ---- 3. 同库逐库配对复评 ----
    banks = t11.make_banks(cfgA, 777, 9, args.gt_banks, dev)
    subA = popA[selA]
    subB = popB[selB]
    keyA = t11._make_key_fn(cfgA)
    fitA = []   # [bank] -> (mean fitness, mean food)
    fitB = []
    for ep in range(args.gt_banks):
        ma = t11._eval_pop_banks(subA, cfgA, [banks[ep]]).numpy()
        mb = t11._eval_pop_banks(subB, cfgB, [banks[ep]]).numpy()
        fA = [float(keyA(ma[i])) for i in range(elite_n)]
        fB = [float(keyA(mb[i])) for i in range(elite_n)]
        fitA.append((sum(fA) / elite_n, float(ma[:, 0].mean())))
        fitB.append((sum(fB) / elite_n, float(mb[:, 0].mean())))
    margins = [b[0] - a[0] for a, b in zip(fitA, fitB)]
    margins_food = [b[1] - a[1] for a, b in zip(fitA, fitB)]
    win = sum(1 for m in margins if m > 0) / len(margins)
    mmean = sum(margins) / len(margins)
    mstd = (sum((m - mmean) ** 2 for m in margins) / (len(margins) - 1)) ** 0.5
    winfood = sum(1 for m in margins_food if m > 0) / len(margins_food)
    mfood = sum(margins_food) / len(margins_food)

    # ---- 4. best_model 单体配对 ----
    def best_scores(tag, cfg):
        res = t11.load_best_state(f'test11_econ_{tag}_best_model.pth', cfg)
        if res is None:
            return None
        st, _, _ = res
        p1 = t11.GeneStack(cfg, B=1, device=dev)
        p1.random_init()
        p1.set_individual_from_state(0, st)
        p1.refresh_eff()
        if getattr(cfg, 'USE_FP16', True):
            p1.fp16()
            p1.refresh_eff()
        sc_f, sc_fd = [], []
        for ep in range(args.gt_banks):
            m1 = t11._eval_pop_banks(p1, cfg, [banks[ep]]).numpy()[0]
            sc_f.append(float(keyA(m1)))
            sc_fd.append(float(m1[0]))
        return sc_f, sc_fd

    bA = best_scores(args.tag_a, cfgA)
    bB = best_scores(args.tag_b, cfgB)
    best_margin = best_win = None
    if bA and bB:
        bm = [x - y for x, y in zip(bB[0], bA[0])]
        best_margin = sum(bm) / len(bm)
        best_win = sum(1 for m in bm if m > 0) / len(bm)

    # ---- 预注册判定 ----
    if win >= args.win_threshold and mmean >= args.margin_threshold:
        quality = 'B 显著优'
    elif win <= 1 - args.win_threshold and mmean <= -args.margin_threshold:
        quality = 'B 更差（A 显著优）'
    elif abs(mmean) < args.equiv_threshold:
        quality = '等效'
    else:
        quality = '不确定（需复核）'
    speed_ok = evalB <= 1.10 * evalA
    noninf = (statsB['final_elite_food'] >= statsA['final_elite_food'] - 0.5
              or statsB['tail5_elite_food'] >= statsA['tail5_elite_food'] - 0.5)
    g2tB, g2tA = statsB['gens_to_theta'], statsA['gens_to_theta']
    faster_progress = (g2tB is not None and (g2tA is None or g2tB <= g2tA))

    print('\n===== test11 A/B 终局验收判定表 =====')
    print(f'{"指标":<34}{"A two_stage":>16}{"B tourn_k3":>16}{"判定":>14}')
    print(f'{"每代评估耗时 (s/gen)":<30}{evalA:>16.1f}{evalB:>16.1f}'
          f'{"OK" if speed_ok else "SLOW":>14} (门: B≤1.10×A)')
    print(f'{"EliteFood 终值":<30}{statsA["final_elite_food"]:>16.2f}'
          f'{statsB["final_elite_food"]:>16.2f}'
          f'{"OK" if noninf else "LAG":>14} (门: B≥A-0.5)')
    print(f'{"EliteFood 尾5均值":<30}{statsA["tail5_elite_food"]:>16.2f}'
          f'{statsB["tail5_elite_food"]:>16.2f}')
    print(f'{"达阈值 θ={:.1f} 代数".format(theta):<28}{str(g2tA):>16}{str(g2tB):>16}'
          f'{"OK" if faster_progress else "SLOWER":>14}')
    print(f'{"avg 冻结10代窗口数":<28}{statsA["avg_freeze10_windows"]:>16}'
          f'{statsB["avg_freeze10_windows"]:>16}')
    print(f'{"EliteFood 最大回撤":<29}{statsA["elite_max_drawdown"]:>16.2f}'
          f'{statsB["elite_max_drawdown"]:>16.2f}')
    label_ov = f'精英集重合 |A∩B|/{elite_n}'
    print(f'{label_ov:<30}{inter:>16}')
    print('-' * 90)
    print(f'终局精英集同 {args.gt_banks} 库配对（适应度）：B−A margin = {mmean:+.3f} '
          f'± {mstd:.3f} | B 胜率 = {win:.0%} | food 口径 margin = {mfood:+.3f} '
          f'(胜率 {winfood:.0%})')
    if best_margin is not None:
        print(f'best_model 单体配对：B−A margin = {best_margin:+.3f} '
              f'(胜率 {best_win:.0%})')
    print(f'[质量判定] {quality} '
          f'(显著优: 胜率≥{args.win_threshold:.0%} 且 margin≥+{args.margin_threshold}; '
          f'等效: |margin|<{args.equiv_threshold})')
    overall = ('B 转正候选' if (quality == 'B 显著优'
                                or (quality == '等效' and speed_ok
                                    and (noninf or faster_progress)))
               else ('保留 two_stage' if quality in ('B 更差（A 显著优）', '等效')
                     else '需复核'))
    print(f'[结论] {overall}')

    result = {
        'meta': {'gens': args.gens, 'gt_banks': args.gt_banks,
                 'tag_a': args.tag_a, 'tag_b': args.tag_b,
                 'elite_n': elite_n, 'sel_overlap': inter,
                 'elapsed_s': time.perf_counter() - t0},
        'curves': {'A': statsA, 'B': statsB},
        'speed': {'eval_s_per_gen_A': evalA, 'eval_s_per_gen_B': evalB,
                  'ratio': evalB / evalA, 'gate_1.10x': speed_ok},
        'paired': {'margin_mean': mmean, 'margin_std': mstd, 'win_rate': win,
                   'margin_food_mean': mfood, 'win_rate_food': winfood,
                   'margins_by_bank': margins,
                   'best_model_margin': best_margin,
                   'best_model_win_rate': best_win},
        'gates': {'quality': quality, 'noninferior': noninf,
                  'faster_progress': faster_progress, 'overall': overall},
    }
    os.makedirs('results', exist_ok=True)
    with open('results/test11_ab_verdict.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print('[Verdict] 结果已写入 results/test11_ab_verdict.json')

    # ---- 图 ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
        ax1.plot(histA['gen'], histA['elite_food'], label=f'A {args.tag_a}',
                 color='tab:blue', marker='o', markersize=3)
        ax1.plot(histB['gen'], histB['elite_food'], label=f'B {args.tag_b}',
                 color='tab:red', marker='s', markersize=3)
        ax1.plot(histA['gen'], histA['avg_food'], label='A avg', color='tab:blue',
                 alpha=0.4, lw=1)
        ax1.plot(histB['gen'], histB['avg_food'], label='B avg', color='tab:red',
                 alpha=0.4, lw=1)
        ax1.axhline(theta, color='gray', ls='--', lw=0.8, label=f'θ={theta:.1f}')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Food')
        ax1.set_title('test11 A/B curves')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax2.hist(margins, bins=16, color='tab:green', alpha=0.75)
        ax2.axvline(0, color='k', lw=1)
        ax2.axvline(mmean, color='tab:red', ls='--',
                    label=f'mean={mmean:+.2f}, win={win:.0%}')
        ax2.set_xlabel('per-bank margin (B − A), elite-set mean fitness')
        ax2.set_title('Paired 40-bank re-eval')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig('results/test11_ab_verdict.png', dpi=100)
        plt.close(fig)
        print('[Verdict] 图已保存 results/test11_ab_verdict.png')
    except Exception as e:  # noqa: BLE001
        print(f'(matplotlib 跳过: {e})')


if __name__ == '__main__':
    main()
