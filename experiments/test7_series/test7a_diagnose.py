"""test7a 死因诊断：对给定模型并发跑 N 局，按死因/蛇长分桶输出结论表。

核心问题（对应训练瓶颈分析）：
  Q1 饿死是"长蛇重排预算不足"还是"绕圈不敢吃"？
     -> 按蛇长分桶的饿死直方图 + 最后一颗食物周期内的最小接近距离分布
        (min-dist 大 => 纯绕圈病态；min-dist 小 => 差一步/预算不足)
  Q2 浪费步数占比：最后一周期浪费步数 / 饿死预算(2L+20)
  Q3 自撞/撞墙死亡时的蛇长分布（确认 55+ 区间是否主导）

用法:
  python test7a_diagnose.py test7a_v5a_best_model.pth [-n 1000]
"""
import os
import sys
import time
import argparse
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location('t7a', os.path.join(ROOT, 'experiments', 'test7_series', 'test7a.py'))
t7a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t7a)

import numpy as np
import torch


def diagnose(model_path, num_games=1000, device=None):
    cfg = t7a.Config()
    cfg.EVAL_EPISODES = 1
    cfg.SEED_FROM_BEST = False

    dev = t7a._resolve_device(cfg) if device is None else torch.device(device)
    print(f"device = {dev}  |  model = {model_path}  |  games = {num_games}")

    res = t7a.load_best_state(model_path, cfg)
    if res is None:
        print(f"错误: 无法加载模型 {model_path}")
        sys.exit(1)
    st, saved_food, saved_steps = res
    print(f"模型元数据: Food={saved_food:.1f}, Steps={saved_steps:.1f}")

    batch = min(t7a._auto_eval_batch(cfg, dev), num_games)
    print(f"并发批大小 = {batch}")

    pop = t7a.GeneStack(cfg, B=batch, device=dev)
    pop.random_init()
    for i in range(batch):
        pop.set_individual_from_state(i, st)
    pop.refresh_eff()
    if getattr(cfg, 'USE_FP16', True):
        pop.fp16()
        pop.refresh_eff()

    N = pop.N
    half = pop.dtype

    # 逐局采集（每局一批，单个体并行）
    recs = []   # dict per game: died, food, body_len, starve_wasted, starve_budget, starve_mindist
    t0 = time.perf_counter()
    done = 0
    while done < num_games:
        B = min(batch, num_games - done)
        sub = pop[:B]
        env = t7a.BatchedSnakeEnv(cfg, B, dev)
        E = torch.zeros(B, N, dtype=half, device=dev)
        I = torch.zeros(B, N, dtype=half, device=dev)
        stt = torch.zeros(B, N, dtype=half, device=dev)
        cts = torch.zeros(B, 3, dtype=half, device=dev)

        food_cnt = torch.zeros(B, dtype=torch.float32, device=dev)
        # 最后一颗食物周期内: 距食物最小欧氏距离 / 浪费步数（吃子时重置）
        mind = torch.full((B,), 1e9, dtype=torch.float32, device=dev)

        for t in range(cfg.MAX_STEPS):
            al = env.alive
            if not bool(al.any().item()):
                break
            obs = env.obs().to(half)
            fr = (env.food[:, 0] - env.head[:, 0]).float()
            fc = (env.food[:, 1] - env.head[:, 1]).float()
            dist = torch.sqrt(fr * fr + fc * fc).clamp(min=1.0)
            mind = torch.minimum(mind, torch.where(al, dist, mind))

            act, E, I, stt = t7a.deliberate_batch(sub, obs, E, I, stt, cts, cfg)
            cts = t7a.update_fatigue(cts, act)
            env.step(act)
            food_cnt += (al & env.ate).float()
            mind = torch.where(env.ate, torch.full_like(mind, 1e9), mind)

        fc = food_cnt.cpu().numpy()
        died = env.died.cpu().numpy()
        blen = env.body_len.cpu().numpy()
        swf = env.steps_wo_food.cpu().numpy().astype(np.float32)
        md = np.minimum(mind.cpu().numpy(), 99.0)

        for i in range(B):
            budget = 2.0 * blen[i] + 20.0
            recs.append(dict(died=int(died[i]), food=float(fc[i]), body_len=int(blen[i]),
                             wasted=float(swf[i]), budget=budget,
                             mindist=float(md[i]) if died[i] == 3 else None))
        done += B
        sys.stdout.write(f"\r  进度: {done}/{num_games}")
        sys.stdout.flush()
    print(f"\n耗时 {time.perf_counter() - t0:.1f}s")

    d = {k: np.array([r[k] if r[k] is not None else np.nan for r in recs])
         for k in ('died', 'food', 'body_len', 'wasted', 'budget', 'mindist')}
    n = len(recs)
    died = d['died'].astype(int)
    names = {0: '存活', 1: '撞墙', 2: '自撞', 3: '饿死'}

    print(f"\n===== 总览 ({n} 局) =====")
    print(f"  平均吃子 {d['food'].mean():.2f}  中位 {np.median(d['food']):.0f}  "
          f"最大 {d['food'].max():.0f}")
    for code, nm in names.items():
        cnt = int((died == code).sum())
        if cnt:
            sel = died == code
            print(f"  {nm}: {cnt} ({cnt / n * 100:.1f}%)  "
                  f"吃子 mean={d['food'][sel].mean():.1f}  "
                  f"蛇长 P25={np.percentile(d['body_len'][sel], 25):.0f} "
                  f"P50={np.percentile(d['body_len'][sel], 50):.0f} "
                  f"P75={np.percentile(d['body_len'][sel], 75):.0f}")

    print(f"\n===== Q1: 自撞蛇长分桶 =====")
    bins = [0, 10, 20, 30, 40, 50, 55, 60, 65, 100]
    lbls = ['<10', '10-20', '20-30', '30-40', '40-50', '50-55', '55-60', '60-65', '65+']
    for code in (1, 2, 3):
        sel = died == code
        if not sel.any():
            continue
        h, _ = np.histogram(d['body_len'][sel], bins=bins)
        tot = sel.sum()
        row = '  '.join(f"{l}:{v}({v / tot * 100:.0f}%)" for l, v in zip(lbls, h) if v > 0)
        print(f"  {names[code]}: {row}")

    print(f"\n===== Q1/Q3: 饿死案例最后一食物周期 min-dist 与浪费比 =====")
    sel = died == 3
    if sel.any():
        md = d['mindist'][sel]
        wr = d['wasted'][sel] / d['budget'][sel]
        print(f"  min-dist: P10={np.percentile(md, 10):.1f} P25={np.percentile(md, 25):.1f} "
              f"P50={np.percentile(md, 50):.1f} P75={np.percentile(md, 75):.1f} "
              f"P90={np.percentile(md, 90):.1f}")
        print(f"  浪费比(浪费步/预算): P50={np.percentile(wr, 50):.2f} "
              f"P90={np.percentile(wr, 90):.2f}")
        # 交叉表: min-dist 桶 x 蛇长桶（只看饿死）
        md_bins = [0, 1.5, 3.5, 6.5, 99]
        md_lbls = ['<=1(差一步)', '1-3(贴身)', '3-6(徘徊)', '>6(远离)']
        h, _ = np.histogram(md, bins=md_bins)
        tot = sel.sum()
        print(f"  min-dist 分桶: " + '  '.join(f"{l}:{v}({v / tot * 100:.0f}%)"
                                              for l, v in zip(md_lbls, h)))
        # 绕圈判定示例：min-dist > 3 且浪费比 >= 0.9（有预算却不接近）
        coward = (md > 3.0) & (wr >= 0.9)
        print(f"  绕圈嫌疑(min-dist>3 且 浪费比>=0.9): {int(coward.sum())} / {tot} "
              f"({coward.mean() * 100:.1f}% of 饿死, {coward.sum() / n * 100:.1f}% of 总局)")
        stuck = (md <= 2.0) & (wr >= 0.99)
        print(f"  预算不足嫌疑(min-dist<=2 且 浪费比~1): {int(stuck.sum())} / {tot} "
              f"({stuck.mean() * 100:.1f}% of 饿死)")

    out = os.path.join(ROOT, 'results', 'test7a_diagnose_result.pth')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({'model': model_path, 'n': n, 'recs': recs}, out)
    print(f"\n原始记录已保存: {out}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='test7a 死因诊断')
    ap.add_argument('model', help='模型 .pth 文件路径')
    ap.add_argument('-n', '--num-games', type=int, default=1000)
    ap.add_argument('-d', '--device', type=str, default=None)
    args = ap.parse_args()
    diagnose(args.model, num_games=args.num_games, device=args.device)
