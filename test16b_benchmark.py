"""test16b 模型快速基准测试：对给定模型并发跑 1000 局，输出平均吃子数等统计。

与 test7b_benchmark.py 同款口径（每模型 1000 局、独立随机落子流、并发批扫描），
评估路径逐字复用 test16b 的 fast-eval 扫描（_eval_sweep_chunk_fast + CRN bank），
与进化期测量同数学：

    python test16b_benchmark.py                 # 默认 simp best + latest_gen_best 各 1000 局
    python test16b_benchmark.py --games 100     # 快速抽查

CRN 种子流取 gen=BENCH_GEN/stage=9（训练只用 gen 0..319 × stage 1/2，不冲突），
跨进程确定可复现。
"""
import os, sys, time, json, argparse, importlib.util

ROOT = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('t16b', os.path.join(ROOT, 'test16b.py'))
t16b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t16b)

import torch
import numpy as np

# 基准局种子流标识（避开训练的 gen×stage 空间）
BENCH_GEN = 987654
BENCH_STAGE = 9

# 7b 1000 局基准（results/test7b_weakmask_bench.json, 2026-08-25）供横向对照
REF_7B = {'model': 'test7b_latest_gen_best', 'mean_food': 61.353, 'median_food': 62.0,
          'std_food': 7.129}


def _stack_banks(cfg, banks, dev):
    return {'stream': torch.stack([b['stream'] for b in banks]),
            'dir0': torch.tensor([b['dir0'] for b in banks], dtype=torch.long, device=dev)}


def benchmark(model_path, num_games=1000, device=None, chunk_eps=1000):
    cfg = t16b.Config()
    cfg.EVAL_EPISODES = 1
    cfg.SEED_FROM_BEST = False

    dev = t16b._resolve_device(cfg) if device is None else torch.device(device)
    print(f"device = {dev}  |  model = {model_path}  |  games = {num_games}")

    res = t16b.load_best_state(model_path, cfg)
    if res is None:
        print(f"错误: 无法加载模型 {model_path}")
        sys.exit(1)
    st, saved_food, saved_steps = res
    print(f"模型元数据: Food={saved_food:.1f}, Steps={saved_steps:.1f}")

    # 单个体基因栈 → 全行同状态（局维折叠：批维 = 局数）
    pop = t16b.GeneStack(cfg, B=1, device=dev)
    pop.random_init()
    pop.set_individual_from_state(0, st)
    if getattr(cfg, 'USE_FP16', True):
        pop.fp16()

    max_rows = t16b._auto_eval_batch(cfg, dev)
    chunk_eps = max(32, min(chunk_eps, num_games, max_rows))
    print(f"并发批大小 = {chunk_eps} 局/扫描（auto 上限 {max_rows}）")

    all_food, all_wall, all_self, all_starve, all_alive = [], [], [], [], []
    all_steps, all_last, all_straight, all_mis = [], [], [], []
    t0 = time.perf_counter()
    done = 0
    while done < num_games:
        E = min(chunk_eps, num_games - done)
        banks = [t16b.make_bank(cfg, BENCH_GEN, BENCH_STAGE, done + e, dev) for e in range(E)]
        bank = _stack_banks(cfg, banks, dev)
        sub = pop[[0] * E]           # 高级索引=拷贝；行 e 即第 done+e 局
        sub.refresh_eff()
        m = t16b._eval_sweep_chunk_fast(sub, cfg, bank).cpu().numpy()   # [E,19]
        all_food.extend(m[:, 0].tolist())
        all_wall.extend(m[:, 5].tolist())
        all_self.extend(m[:, 6].tolist())
        all_starve.extend(m[:, 7].tolist())
        all_steps.extend((m[:, 1] + m[:, 2]).tolist())   # 总存活步数
        all_last.extend(m[:, 3].tolist())                # 最后一食步
        all_straight.extend(m[:, 17].tolist())
        all_mis.extend(m[:, 12].tolist())
        done += E
        print(f"  进度: {done}/{num_games}（累计 {time.perf_counter() - t0:.1f}s）")

    elapsed = time.perf_counter() - t0
    food_arr = np.array(all_food)
    steps_arr = np.array(all_steps)
    last_arr = np.array(all_last)
    pct = {f'P{p}': float(np.percentile(food_arr, p)) for p in (5, 25, 50, 75, 90, 95, 99)}
    out = {
        'model': os.path.basename(model_path),
        'num_games': num_games,
        'elapsed_s': round(elapsed, 1),
        'train_food': saved_food,
        'mean_food': float(food_arr.mean()),
        'std_food': float(food_arr.std()),
        'median_food': float(np.median(food_arr)),
        'min_food': float(food_arr.min()),
        'max_food': float(food_arr.max()),
        'pct': pct,
        'die_wall': float(np.mean(np.array(all_wall) > 0)),
        'die_self': float(np.mean(np.array(all_self) > 0)),
        'die_starve': float(np.mean(np.array(all_starve) > 0)),
        'survive_cap': float(np.mean((np.array(all_wall) + np.array(all_self)
                                      + np.array(all_starve)) == 0)),
        'mean_steps': float(steps_arr.mean()),
        'mean_steps_last_food': float(last_arr.mean()),
        'mean_straight': float(np.mean(all_straight)),
        'mean_mismatch': float(np.mean(all_mis)),
        'food_all': [float(x) for x in food_arr.tolist()],
    }
    print(f"\n--- 基准结果 ({num_games} 局, {elapsed:.1f}s) ---")
    print(f"  吃子: mean={out['mean_food']:.2f}  median={out['median_food']:.1f}  "
          f"std={out['std_food']:.2f}  min={out['min_food']:.0f}  max={out['max_food']:.0f}")
    print(f"  分位: " + "  ".join(f"{k}={v:.0f}" for k, v in pct.items()))
    print(f"  死因: 撞墙 {out['die_wall']:.1%} | 撞己 {out['die_self']:.1%} | "
          f"饿死 {out['die_starve']:.1%} | 存活到上限 {out['survive_cap']:.1%}")
    print(f"  步数: 总存活 mean={out['mean_steps']:.0f}  末食 mean={out['mean_steps_last_food']:.0f}  "
          f"直行率={out['mean_straight']:.3f}")
    print(f"  对照: 训练期估计 {saved_food:.1f}（K=36 局均值）| "
          f"7b 1000 局 mean={REF_7B['mean_food']:.2f} median={REF_7B['median_food']:.1f}")
    return out


def main():
    ap = argparse.ArgumentParser(description='test16b 模型 1000 局基准（fast-eval 扫描）')
    ap.add_argument('--models', nargs='+', default=[
        os.path.join(ROOT, 'test16b_simp_best_model.pth'),
        os.path.join(ROOT, 'test16b_simp_latest_gen_best.pth'),
    ], help='模型文件路径（默认 simp best + latest_gen_best）')
    ap.add_argument('--games', type=int, default=1000)
    ap.add_argument('--device', default=None)
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'test16b_bench1k.json'))
    args = ap.parse_args()

    torch.manual_seed(20260906)
    results = []
    for mp in args.models:
        r = benchmark(mp, num_games=args.games, device=args.device)
        results.append(r)
        print()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'ref_7b_1000g': REF_7B, 'results': results}, f,
                  ensure_ascii=False, indent=1)
    print(f"已写入 {args.out}")


if __name__ == '__main__':
    main()
