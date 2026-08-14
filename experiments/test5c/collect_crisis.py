# -*- coding: utf-8 -*-
"""危急局面数据采集（test5c A2）：随机游走蛇 + 贪心教师软标签。

问题背景：原 dataset.npz 来自贪心专家轨迹，专家几乎从不走到
"前方 1 格就是墙/蛇身"的危急局面（它早已转向），导致 CNN 对
避障（危险信号）编码严重欠采样，EI 网络学不到"快转弯保命"。

方案：用随机游走蛇跑局，蛇会频繁撞向墙/蛇身，自然产生大量
各类危急状态；在每步用 expert_ai.ExpertSnakeAI 打软标签。

输出 test5c/crisis_dataset.npz：
    {
      'grids':   np.float32 (N,10,10,5),
      'softs':   np.float32 (N,3),
      'actions': np.int64 (N,),
      'stats':   {...}
    }

用法：
    python test5c/collect_crisis.py --episodes 200 --out test5c/crisis_dataset.npz
"""
import argparse
import os
import random
import sys
import time
import numpy as np

# 同目录 + test5b 教师模块
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(_THIS_DIR), 'test5b'))

from env import SnakeEnv
from expert_ai import ExpertSnakeAI, _softmax


def run_crisis_game(env, expert, max_steps=300, record_grid=True):
    """随机游走蛇跑一局，每步教师打软标签。

    随机动作：0=Fwd, 1=Left, 2=Right 等概率 → 蛇频繁撞墙/撞身，
    大量危急状态。
    """
    env.reset()
    grids, softs, actions = [], [], []
    done = False
    steps = 0
    while not done and steps < max_steps:
        if record_grid:
            grids.append(env._get_grid_state())
        # 教师对"当前状态"打分（含当前朝向与蛇身），与 collect_data 一致
        scores, expert_action = expert.act(env)
        softs.append(_softmax(scores, temperature=1.0))
        # 实际执行随机动作（产生危急状态），而非教师的决策动作
        rand_action = random.randint(0, 2)
        actions.append(rand_action)  # 软标签用教师，硬标签用实际动作？——见下

        _, _, done = env.step(rand_action)
        steps += 1
    return {
        'grids': grids,
        'softs': softs,
        'actions': actions,
        'foods': env.food_count,
        'steps': steps,
    }


def collect_crisis(episodes=200, out_path='crisis_dataset.npz', seed=777,
                   probe=False):
    random.seed(seed)
    np.random.seed(seed)
    t0 = time.perf_counter()

    print(f"[Crisis] 随机游走蛇采集 {episodes} 局危急数据 | out={out_path}",
          flush=True)

    env = SnakeEnv(grid_size=10)
    expert = ExpertSnakeAI(grid_size=10)

    # ---- Probe ----
    t_probe = time.perf_counter()
    r = run_crisis_game(env, expert, max_steps=300, record_grid=True)
    probe_dt = time.perf_counter() - t_probe
    print(f"[Probe] 1 局 {probe_dt:.1f}s | 步数 {r['steps']} | "
          f"食物 {r['foods']} | 单步 {probe_dt/max(1,r['steps'])*1000:.2f}ms",
          flush=True)
    print(f"[Probe] 预计 {episodes} 局 ≈ {probe_dt*episodes/60:.1f} 分钟",
          flush=True)
    if probe:
        print("[Probe] 自检模式：结束。", flush=True)
        return None

    # ---- 采集 ----
    all_records = []
    progress_gap = max(1, episodes // 20)
    next_report = progress_gap
    t_run = time.perf_counter()

    for ep in range(episodes):
        rec = run_crisis_game(env, expert, max_steps=300, record_grid=True)
        all_records.append(rec)
        done = ep + 1
        if done >= next_report or done >= episodes:
            elapsed = time.perf_counter() - t_run
            print(f"  已采集 {done}/{episodes} 局 | "
                  f"耗时 {elapsed:.0f}s | {done/elapsed:.0f} 局/s", flush=True)
            next_report += progress_gap

    # ---- 汇总统计 ----
    foods = [r['foods'] for r in all_records]
    steps_list = [r['steps'] for r in all_records]
    avg_food = float(np.mean(foods))
    avg_steps = float(np.mean(steps_list))
    death_rate = float(np.mean([1 if s < 300 else 0 for s in steps_list]))
    n_samples = sum(len(r['grids']) for r in all_records)

    print(f"[Crisis] 完成 {len(all_records)} 局 | 样本 {n_samples} | "
          f"平均食物 {avg_food:.1f} | 死局率 {death_rate*100:.1f}%", flush=True)

    # ---- 打包 ----
    print("[Crisis] 打包数据 ...", flush=True)
    grids_list = [np.asarray(r['grids'], dtype=np.float32) for r in all_records]
    softs_list = [np.asarray(r['softs'], dtype=np.float32) for r in all_records]
    grids = np.concatenate(grids_list, axis=0)
    softs = np.concatenate(softs_list, axis=0)
    del grids_list, softs_list

    # 硬标签：用教师的决策动作（argmax 软标签），更符合"教师监督"语义。
    # （随机动作只用于制造危急状态轨迹，不作为标签。）
    actions = np.argmax(softs, axis=1).astype(np.int64)

    np.savez(out_path, grids=grids, softs=softs, actions=actions,
             stats={'episodes': len(all_records),
                    'n_samples': int(n_samples),
                    'avg_food': avg_food,
                    'avg_steps': float(avg_steps),
                    'max_food': int(max(foods)),
                    'min_food': int(min(foods)),
                    'death_rate': float(death_rate),
                    'collect_seconds': float(time.perf_counter() - t0)})
    print(f"[Crisis] 已保存 -> {out_path} "
          f"(总耗时 {time.perf_counter()-t0:.1f}s)", flush=True)
    return {'grids': grids, 'softs': softs, 'actions': actions}


def merge_datasets(expert_path, crisis_path, out_path,
                   expert_frac=0.7, crisis_frac=0.3):
    """合并专家数据与危急数据（按比例混合，危急数据过采样保命技能）。

    最终混合比例：专家 crisis × crisis_frac / expert 基数。
    若 crisis 数据量 < expert×frac/crisis_frac，则 crisis 全部保留
    并按比例复制达到目标；否则随机抽到目标量。
    """
    d1 = np.load(expert_path, allow_pickle=True)
    d2 = np.load(crisis_path, allow_pickle=True)
    g1, s1, a1 = d1['grids'], d1['softs'], d1['actions']
    g2, s2, a2 = d2['grids'], d2['softs'], d2['actions']

    n1, n2 = len(g1), len(g2)
    # 目标 crisis 数量 = 使 crisis 占总量 crisis_frac
    # g2_target = n1 * crisis_frac / max(expert_frac, 1e-8)
    g2_target = int(n1 * crisis_frac / max(expert_frac, 1e-8))
    g2_target = min(g2_target, n2)
    if g2_target < n2:
        idx = np.random.permutation(n2)[:g2_target]
        g2, s2, a2 = g2[idx], s2[idx], a2[idx]
    elif g2_target > n2 and n2 > 0:
        # 复制直到达到目标
        reps = int(np.ceil(g2_target / n2))
        g2 = np.tile(g2, (reps, 1, 1, 1))[:g2_target]
        s2 = np.tile(s2, (reps, 1))[:g2_target]
        a2 = np.tile(a2, (reps,))[:g2_target]

    grids = np.concatenate([g1, g2], axis=0)
    softs = np.concatenate([s1, s2], axis=0)
    actions = np.concatenate([a1, a2], axis=0)

    # 全局洗牌
    perm = np.random.permutation(len(grids))
    grids, softs, actions = grids[perm], softs[perm], actions[perm]

    np.savez(out_path, grids=grids, softs=softs, actions=actions,
             stats={'n_expert': int(n1), 'n_crisis': int(len(g2)),
                    'n_total': int(len(grids))})
    print(f"[Merge] 专家 {n1} + 危急 {len(g2)} = {len(grids)} -> {out_path}")
    print(f"[Merge] 混合后动作分布: "
          f"Fwd {np.mean(actions==0)*100:.1f}% / "
          f"Left {np.mean(actions==1)*100:.1f}% / "
          f"Right {np.mean(actions==2)*100:.1f}%")
    return {'grids': grids, 'softs': softs, 'actions': actions}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--out', type=str, default='test5c/crisis_dataset.npz')
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()

    if args.probe:
        collect_crisis(args.episodes, args.out, args.seed, probe=True)
    else:
        # 先采集，再合并（一次命令完成）
        crisis = collect_crisis(args.episodes, args.out, args.seed)
        merge_datasets('test5b/dataset.npz', args.out,
                       'test5c/dataset_combined.npz')