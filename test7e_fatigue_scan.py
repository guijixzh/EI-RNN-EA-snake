# ==========================================
# test7e_fatigue_scan.py —— 疲劳机制现状定量刻画（零训练）
#
# 在两个决策强度迥异的模型上扫 gain×decay 网格：
#   test7b_latest_gen_best（强决策：转向边际均值 6.9）
#   test7d_econ_best       （弱决策：转向边际均值 1.0）
# 输出：平均吃子 / 转向密度 / 短直行游程占比 / 死因
# 回答：绝对减益式疲劳对强决策策略还有多少实际影响力；
#       gain 的最优点是否随模型 logit 尺度漂移（量纲错配证据）。
# ==========================================
import os, sys, time, importlib.util

ROOT = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('t7e', os.path.join(ROOT, 'test7e.py'))
t7e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t7e)

import numpy as np
import torch

GAINS = [0.0, 0.1, 0.2, 0.4, 0.8]
DECAYS = [0.3, 0.6, 0.9]
GAMES = 200


def run_grid(model_path, games=GAMES):
    cfg = t7e.Config()
    cfg.EVAL_EPISODES = 1
    cfg.SEED_FROM_BEST = False
    dev = t7e._resolve_device(cfg)
    res = t7e.load_best_state(model_path, cfg)
    if res is None:
        print(f'错误: 无法加载 {model_path}')
        sys.exit(1)
    st, saved_food, _ = res

    B = games
    pop = t7e.GeneStack(cfg, B=B, device=dev)
    pop.random_init()
    for i in range(B):
        pop.set_individual_from_state(i, st)
    pop.refresh_eff()
    pop.fp16(); pop.refresh_eff()
    env = t7e.BatchedSnakeEnv(cfg, B, dev)
    G, N, K = cfg.GRID_SIZE, pop.N, cfg.FRAME_RATE
    half = torch.float16

    print(f'\n########## model = {os.path.basename(model_path)} (saved Food={saved_food:.0f}) '
          f'games={games} ##########')
    print(f"{'gain':>5} {'decay':>6} | {'food':>7} {'P50':>4} | {'密度':>6} {'游程≤2':>6} | "
          f"撞墙%  自撞%  饿死%  [t]")
    print('-' * 70)
    for gain in GAINS:
        for decay in DECAYS:
            cfg.FATIGUE_TURN_GAIN = gain
            cfg.FATIGUE_TURN_DECAY = decay
            env.reset()
            E = torch.zeros(B, N, dtype=half, device=dev)
            I = torch.zeros(B, N, dtype=half, device=dev)
            stt = torch.zeros(B, N, dtype=half, device=dev)
            press = torch.zeros(B, dtype=torch.float32, device=dev)
            act_buf = torch.zeros(B, dtype=torch.long, device=dev) - 1
            run_len = torch.zeros(B, dtype=torch.long, device=dev)
            tot_turn = np.zeros(B); tot_step = np.zeros(B)
            short_runs = 0; total_runs = 0
            foods = np.zeros(B)
            t0 = time.perf_counter()
            for t in range(cfg.MAX_STEPS):
                al = env.alive
                if not al.any():
                    break
                obs = env.obs().to(half)
                act, E, I, stt = t7e.deliberate_batch(pop, obs, E, I, stt, press, cfg)
                press = t7e.update_fatigue(press, act, decay=decay)
                # 游程统计（先读后写）
                ended = al & (act != 0) & (act_buf == 0)
                rl = run_len[ended].cpu().numpy()
                short_runs += int((rl <= 2).sum()); total_runs += int(len(rl))
                cont = (act == act_buf) & (act == 0) & al
                run_len = torch.where(cont, run_len + 1,
                                      torch.where(al & (act == 0),
                                                  torch.ones_like(run_len),
                                                  torch.zeros_like(run_len)))
                act_buf = torch.where(al, act, act_buf)
                tot_turn += (al & (act != 0)).float().cpu().numpy()
                tot_step += al.float().cpu().numpy()
                env.step(act)
                foods += (al & env.ate).float().cpu().numpy()
            dens = tot_turn.sum() / max(tot_step.sum(), 1)
            wall = (env.died == 1).float().mean().item() * 100
            self_ = (env.died == 2).float().mean().item() * 100
            starve = (env.died == 3).float().mean().item() * 100
            sr = short_runs / max(total_runs, 1) * 100
            print(f"{gain:>5.2f} {decay:>6.2f} | {foods.mean():>7.2f} {np.median(foods):>4.0f} | "
                  f"{dens:>6.3f} {sr:>5.1f}% | "
                  f"{wall:>5.1f} {self_:>5.1f} {starve:>5.1f}  [{time.perf_counter()-t0:.0f}s]")


if __name__ == '__main__':
    for m in ['test7b_latest_gen_best.pth', 'test7d_econ_best_model.pth']:
        run_grid(m)
