"""test7b 模型快速基准测试：对给定模型并发跑 1000 局，输出平均吃子数等统计。

支持 --weak-mask-frac 评估期弱连接屏蔽（与 test12.apply_weak_mask 同款逐行
kthvalue 逻辑，作用副本权重，不回写模型文件）。
"""
import os, sys, time, math, argparse, importlib.util, json

ROOT = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('t7b', os.path.join(ROOT, 'test7b.py'))
t7b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t7b)

import torch
import numpy as np


def apply_weak_mask_state(st, frac):
    """对单个体状态 dict 的 W_rec 做弱连接屏蔽（逐行 kthvalue，与 test12 一致）。

    返回剪枝的连接数（原地修改 st['W_rec']，不碰 M_rec/其他权重）。
    """
    if frac <= 0:
        return 0
    W = st['W_rec'].float()
    M = st['M_rec'].float()
    mag = W.abs() * M
    keep = M > 0
    n_pruned = 0
    for r in range(W.shape[0]):
        m_r = mag[r][keep[r]]
        if m_r.numel() == 0:
            continue
        kk = int(math.ceil(m_r.numel() * frac))
        if kk <= 0:
            continue
        thr = torch.kthvalue(m_r, kk).values
        zero = keep[r] & (mag[r] <= thr)
        W[r] = torch.where(zero, torch.zeros_like(W[r]), W[r])
        n_pruned += int(zero.sum().item())
    st['W_rec'] = W
    return n_pruned


def benchmark(model_path, num_games=1000, device=None, weak_mask_frac=0.0):
    cfg = t7b.Config()
    cfg.EVAL_EPISODES = 1
    cfg.SEED_FROM_BEST = False

    dev = t7b._resolve_device(cfg) if device is None else torch.device(device)
    print(f"device = {dev}  |  model = {model_path}  |  games = {num_games}  |  "
          f"weak_mask_frac = {weak_mask_frac}")

    res = t7b.load_best_state(model_path, cfg)
    if res is None:
        print(f"错误: 无法加载模型 {model_path}")
        sys.exit(1)
    st, saved_food, saved_steps = res
    print(f"模型元数据: Food={saved_food:.1f}, Steps={saved_steps:.1f}")

    n_pruned = apply_weak_mask_state(st, weak_mask_frac)
    n_active = int((st['M_rec'].float() > 0).sum().item())
    if weak_mask_frac > 0:
        print(f"弱连接屏蔽: 剪掉 {n_pruned}/{n_active} 条 W_rec 连接 "
              f"({n_pruned / max(n_active, 1) * 100:.1f}%)")

    batch = min(t7b._auto_eval_batch(cfg, dev), num_games)
    print(f"并发批大小 = {batch}")

    pop = t7b.GeneStack(cfg, B=batch, device=dev)
    pop.random_init()
    for i in range(batch):
        pop.set_individual_from_state(i, st)
    pop.refresh_eff()
    if getattr(cfg, 'USE_FP16', True):
        pop.fp16()
        pop.refresh_eff()

    all_food = []
    all_wall = []
    all_self = []
    all_starve = []
    t0 = time.perf_counter()

    done = 0
    while done < num_games:
        B = min(batch, num_games - done)
        sub = pop[:B]
        metrics = t7b._eval_chunk(sub, cfg)
        mn = metrics.numpy()
        all_food.extend(mn[:, 0].tolist())
        all_wall.extend(mn[:, 5].tolist())
        all_self.extend(mn[:, 6].tolist())
        all_starve.extend(mn[:, 7].tolist())
        done += B
        sys.stdout.write(f"\r  进度: {done}/{num_games}")
        sys.stdout.flush()

    elapsed = time.perf_counter() - t0
    print(f"\n--- 基准结果 ({num_games} 局, {elapsed:.1f}s, weak_mask_frac={weak_mask_frac}) ---")
    food_arr = np.array(all_food)
    wall_arr = np.array(all_wall)
    self_arr = np.array(all_self)
    starve_arr = np.array(all_starve)
    print(f"  平均吃子:  {food_arr.mean():.3f}  (std={food_arr.std():.3f})")
    print(f"  中位吃子:  {np.median(food_arr):.1f}")
    print(f"  最大吃子:  {food_arr.max():.0f}")
    print(f"  死因分布:  撞墙 {wall_arr.mean()*100:.1f}%  自撞 {self_arr.mean()*100:.1f}%  饿死 {starve_arr.mean()*100:.1f}%")
    pcts = [25, 50, 75, 90, 95, 99]
    print(f"  百分位:    " + "  ".join(f"P{p}={np.percentile(food_arr, p):.0f}" for p in pcts))

    return {
        'model': model_path,
        'weak_mask_frac': weak_mask_frac,
        'num_games': num_games,
        'elapsed_s': elapsed,
        'n_pruned_rec': n_pruned,
        'mean_food': float(food_arr.mean()),
        'std_food': float(food_arr.std()),
        'median_food': float(np.median(food_arr)),
        'max_food': float(food_arr.max()),
        'pct': {f'P{p}': float(np.percentile(food_arr, p)) for p in pcts},
        'die_wall': float(wall_arr.mean()),
        'die_self': float(self_arr.mean()),
        'die_starve': float(starve_arr.mean()),
        'food_all': all_food,
    }


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='test7b 模型基准测试 (1000 局并发)')
    ap.add_argument('model', nargs='?', default='artifacts/test7b/test7b_latest_gen_best.pth',
                    help='模型 .pth 文件路径 (默认 artifacts/test7b/test7b_latest_gen_best.pth)')
    ap.add_argument('-n', '--num-games', type=int, default=1000, help='总局数 (默认 1000)')
    ap.add_argument('-d', '--device', type=str, default=None, help='设备 (默认 auto)')
    ap.add_argument('-w', '--weak-mask-frac', type=float, nargs='+', default=[0.0],
                    help='弱连接屏蔽比例，可多个 (默认 0；如 0 0.1 0.2 0.3)')
    ap.add_argument('-o', '--out-json', type=str, default=None,
                    help='结果落盘 JSON 路径 (可选)')
    args = ap.parse_args()

    results = []
    for frac in args.weak_mask_frac:
        r = benchmark(args.model, num_games=args.num_games, device=args.device,
                      weak_mask_frac=frac)
        results.append(r)
        print()

    if args.out_json:
        parent = os.path.dirname(os.path.abspath(args.out_json))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已写入 {args.out_json}")
