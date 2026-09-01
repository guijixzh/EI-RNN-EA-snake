# ==========================================
# test7c_fatigue_scan.py —— 转向疲劳零训练扫描
#
# 加载 test7b 最优模型（权重不动），在 gain×decay 网格上并发评估，
# 量化转向疲劳对现有"密集摆动"策略的即时行为改变幅度：
#   - gain=0 一组即 test7b 现状基线
#   - 选参标准：转向占比显著下降（目标减半）且平均吃子不显著下滑
# ==========================================
import os, sys, time, importlib.util

ROOT = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('t7c', os.path.join(ROOT, 'test7c.py'))
t7c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t7c)

import torch
import numpy as np

GAINS = [0.0, 0.1, 0.2, 0.4, 0.8]
DECAYS = [0.5, 0.7, 0.9]
GAMES = 200
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'test7b_best_model.pth'


def run_grid(model_path, games=GAMES):
    cfg = t7c.Config()
    cfg.EVAL_EPISODES = 1
    cfg.SEED_FROM_BEST = False
    dev = t7c._resolve_device(cfg)

    res = t7c.load_best_state(model_path, cfg)
    if res is None:
        print(f"错误: 无法加载模型 {model_path}")
        sys.exit(1)
    st, saved_food, _ = res
    print(f"model = {model_path}  (saved Food={saved_food:.1f})  games/grid = {games}")

    batch = min(t7c._auto_eval_batch(cfg, dev), games)
    pop = t7c.GeneStack(cfg, B=batch, device=dev)
    pop.random_init()
    for i in range(batch):
        pop.set_individual_from_state(i, st)
    pop.refresh_eff()
    if getattr(cfg, 'USE_FP16', True):
        pop.fp16()
        pop.refresh_eff()

    print(f"{'gain':>5} {'decay':>6} | {'food':>7} {'P50':>4} {'P90':>4} | "
          f"{'turn%':>6} | 撞墙%  自撞%  饿死%")
    print('-' * 66)
    for gain in GAINS:
        for decay in DECAYS:
            cfg.FATIGUE_TURN_GAIN = gain
            cfg.FATIGUE_TURN_DECAY = decay
            foods, turns, steps_all = [], [], []
            wall = self_ = starve = 0.0
            t0 = time.perf_counter()
            done = 0
            while done < games:
                B = min(batch, games - done)
                sub = pop[:B]
                m = t7c._eval_chunk(sub, cfg).numpy()
                foods.extend(m[:, 0].tolist())
                wall += m[:, 5].sum()
                self_ += m[:, 6].sum()
                starve += m[:, 7].sum()
                turns.extend((m[:, 8] + m[:, 9]).tolist())   # 每局转向次数
                done += B
            fa = np.array(foods)
            ta = np.array(turns)
            # 每局步数 ≈ seen+unseen（每存活步计一次），转向占比 = 转向数/步数
            # _eval_chunk 未返回步数列，此处用转向数绝对值对比（各组同口径可比）
            print(f"{gain:>5.2f} {decay:>6.2f} | {fa.mean():>7.2f} "
                  f"{np.median(fa):>4.0f} {np.percentile(fa, 90):>4.0f} | "
                  f"{ta.mean():>6.1f} | {wall/games*100:>5.1f} {self_/games*100:>5.1f} {starve/games*100:>5.1f}"
                  f"   [{time.perf_counter()-t0:.0f}s]")


if __name__ == '__main__':
    run_grid(MODEL)
