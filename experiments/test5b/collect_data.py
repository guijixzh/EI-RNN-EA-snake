"""串行数据采集：用 ExpertSnakeAI 生成 (网格, 软标签, 动作) 数据集。

用法：
    python collect_data.py --episodes 500 --out dataset.npz
    python collect_data.py --probe                       # 快速自检：跑 1 局

纯串行设计（本机 Windows + Anaconda 多进程环境不稳定，坚决不用进程池）：
    - 主循环简单 for，每跑 5% 局打印一次进度（带 flush）
    - worker 异常直接抛到主进程打印，绝不静默挂死
    - 单局约 0.5s → 500 局约 4 分钟，一次性采集完全可接受

输出 dataset.npz (dict)：
    {
      'grids':  np.float32 (N, 10, 10, 5),   # channel_last
      'softs':  np.float32 (N, 3),           # 专家启发式分数 softmax 软标签
      'actions': np.int64 (N,),              # 最终 argmax 动作
      'stats':  {'episodes', 'n_samples', 'avg_food', 'avg_steps',
                 'max_food', 'min_food', 'death_rate', 'collect_seconds'}
    }
"""
import argparse
import os
import random
import sys
import time
import numpy as np

from env import SnakeEnv
from expert_ai import ExpertSnakeAI, run_expert_game


# ==========================================
# 主流程（纯串行）
# ==========================================
def collect_dataset(episodes=500, out_path='dataset.npz', seed=12345,
                    probe=False):
    random.seed(seed)
    np.random.seed(seed)
    t0 = time.perf_counter()

    print(f"[Collect] 串行采集 {episodes} 局 | out={out_path}", flush=True)

    # ---- Probe：先跑 1 局计时 ----
    env = SnakeEnv(grid_size=10)
    expert = ExpertSnakeAI(grid_size=10)
    t_probe = time.perf_counter()
    probe_rec = run_expert_game(env, expert, max_steps=2000, record_grid=True)
    probe_dt = time.perf_counter() - t_probe
    print(f"[Probe] 1 局 {probe_dt:.1f}s | 步数 {probe_rec['steps']} | "
          f"食物 {probe_rec['foods']} | 单步 "
          f"{probe_dt/max(1,probe_rec['steps'])*1000:.2f}ms", flush=True)
    est = probe_dt * episodes
    print(f"[Probe] 预计 {episodes} 局串行 ≈ {est/60:.1f} 分钟 | "
          f"样本约 {probe_rec['steps']*episodes:,}", flush=True)
    if probe:
        print("[Probe] 自检模式：结束。", flush=True)
        return None

    # ---- 采集 ----
    all_records = []
    progress_gap = max(1, episodes // 20)  # 每 5% 打印一次
    next_report = progress_gap
    t_run = time.perf_counter()
    print("[Collect] 开始采集 ...", flush=True)

    for ep in range(episodes):
        r = run_expert_game(env, expert, max_steps=2000, record_grid=True)
        all_records.append(r)
        done = ep + 1
        if done >= next_report or done >= episodes:
            elapsed = time.perf_counter() - t_run
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  已采集 {done}/{episodes} 局 | "
                  f"耗时 {elapsed:.0f}s | {rate:.0f} 局/s", flush=True)
            next_report += progress_gap

    # ---- 汇总统计 ----
    foods = [r['foods'] for r in all_records]
    steps_list = [r['steps'] for r in all_records]
    avg_food = float(np.mean(foods))
    avg_steps = float(np.mean(steps_list))
    max_food = int(np.max(foods))
    min_food = int(np.min(foods))
    death_rate = float(np.mean([1 if r['steps'] < 2000 else 0
                                for r in all_records]))
    n_samples = sum(len(r['grids']) for r in all_records)

    print(f"[Collect] 完成 {len(all_records)} 局 | 样本 {n_samples} | "
          f"平均食物 {avg_food:.1f} | 最大 {max_food} 最小 {min_food} | "
          f"死局率 {death_rate*100:.1f}%", flush=True)

    # ---- 打包 ----
    print("[Collect] 正在打包数据 ...", flush=True)
    grids_list = []
    softs_list = []
    actions_list = []
    for r in all_records:
        grids_list.append(np.asarray(r['grids'], dtype=np.float32))
        softs_list.append(np.asarray(r['softs'], dtype=np.float32))
        actions_list.append(np.asarray(r['actions'], dtype=np.int64))
    grids = np.concatenate(grids_list, axis=0)
    softs = np.concatenate(softs_list, axis=0)
    actions = np.concatenate(actions_list, axis=0)
    del grids_list, softs_list, actions_list

    data = {
        'grids': grids,
        'softs': softs,
        'actions': actions,
        'stats': {
            'episodes': len(all_records),
            'n_samples': int(n_samples),
            'avg_food': avg_food,
            'avg_steps': avg_steps,
            'max_food': max_food,
            'min_food': min_food,
            'death_rate': death_rate,
            'collect_seconds': float(time.perf_counter() - t0),
        },
    }

    print(f"[Collect] 正在写入 {out_path} ...", flush=True)
    t_save = time.perf_counter()
    np.savez(out_path, grids=grids, softs=softs, actions=actions,
             stats=data['stats'])
    print(f"[Collect] 已保存 -> {out_path} "
          f"(写入 {time.perf_counter()-t_save:.1f}s) | "
          f"总耗时 {time.perf_counter()-t0:.1f}s", flush=True)
    return data


def load_dataset(path):
    """加载 npz 数据集。"""
    d = np.load(path, allow_pickle=True)
    out = {}
    for k in d.files:
        if k == 'stats':
            out[k] = d[k].item()
        else:
            out[k] = d[k]
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--out', type=str, default='dataset.npz')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--probe', action='store_true',
                        help='快速自检：跑 1 局验证环境与速度后退出')
    args = parser.parse_args()
    collect_dataset(args.episodes, args.out, args.seed, probe=args.probe)