"""test7b 模型快速基准测试：对给定模型并发跑 1000 局，输出平均吃子数等统计。"""
import os, sys, time, argparse, importlib.util

ROOT = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('t7b', os.path.join(ROOT, 'test7b.py'))
t7b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t7b)

import torch
import numpy as np


def benchmark(model_path, num_games=1000, device=None):
    cfg = t7b.Config()
    cfg.EVAL_EPISODES = 1
    cfg.SEED_FROM_BEST = False

    dev = t7b._resolve_device(cfg) if device is None else torch.device(device)
    print(f"device = {dev}  |  model = {model_path}  |  games = {num_games}")

    res = t7b.load_best_state(model_path, cfg)
    if res is None:
        print(f"错误: 无法加载模型 {model_path}")
        sys.exit(1)
    st, saved_food, saved_steps = res
    print(f"模型元数据: Food={saved_food:.1f}, Steps={saved_steps:.1f}")

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
    print(f"\n\n--- 基准结果 ({num_games} 局, {elapsed:.1f}s) ---")
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


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='test7b 模型基准测试 (1000 局并发)')
    ap.add_argument('model', nargs='?', default='test7b_latest_gen_best.pth',
                    help='模型 .pth 文件路径 (默认 test7b_latest_gen_best.pth)')
    ap.add_argument('-n', '--num-games', type=int, default=1000, help='总局数 (默认 1000)')
    ap.add_argument('-d', '--device', type=str, default=None, help='设备 (默认 auto)')
    args = ap.parse_args()
    benchmark(args.model, num_games=args.num_games, device=args.device)
