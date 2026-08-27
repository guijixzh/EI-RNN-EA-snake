# ==========================================
# test8.py —— 验证 NEAT 与 EI-RNN 的契合度
#
#   - 真正的 NEAT（创新号 + 物种化 + 历史标记交叉 + add-connection/add-node）
#     驱动 EI-RNN（256 柱稀疏拓扑，INIT_DENSITY=0.15 启动）进化。
#   - 环境：K=5（FRAME_RATE=5）贪吃蛇，新 32 维头朝向相对 8 方向观测
#     （RaySnakeEnv / BatchedRaySnakeEnv）。
#   - 评估：GPU 全向量化（GeneStack + BatchedRaySnakeEnv），K 帧思考、
#     三元组 (food, seen, unseen) 筛选、单侧转弯判死、饥饿截断、动作疲劳。
#   - 契合度健康指标：每代物种数、平均激活连接数、平均激活柱数。
# ==========================================

import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from einbrain import Config, io
from einbrain.evolve import _selection_key
from einbrain.gpu import (GeneStack, _auto_eval_batch, _resolve_device,
                          deliberate_batch, forward_batch, update_fatigue,
                          BatchedRaySnakeEnv)
from einbrain.neat import NEATPopulation, decode_population


# ==========================================
# 0. 配置
# ==========================================
class NEATConfig(Config):
    OBS_DIM = 32          # 蛇首方向4 + 蛇尾方向4 + 食物8 + 自身8 + 障碍倒数8
    ACTION_DIM = 3
    FRAME_RATE = 5        # K=5
    GRID_SIZE = 10
    # NEAT 参数（COMPAT_*/SPECIES_*/ADD_*/REENABLE_*）继承自 einbrain.Config

    # --- 输出 ---
    CHECKPOINT_PATH = 'test8_checkpoint.pth'
    BEST_MODEL_PATH = 'test8_best_model.pth'
    CHECKPOINT_INTERVAL = 5
    PRINT_HISTORY_EVERY = 1
    AUTO_RESUME = True
    SEED_FROM_BEST = False

    # --- GPU ---
    DEVICE = 'auto'
    USE_FP16 = True
    EVAL_BATCH = 0


# ==========================================
# 1. GPU 批量评估（BatchedRaySnakeEnv，修正 food 计数）
# ==========================================
def _eval_chunk_ray(pop, cfg):
    B = pop.P
    dev = pop.device
    N = pop.N
    A = pop.A
    env = BatchedRaySnakeEnv(cfg, B, dev)
    half = torch.float16 if getattr(cfg, 'USE_FP16', True) else torch.float32

    tot_food = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_seen = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_unseen = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_act1 = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_act2 = torch.zeros(B, dtype=torch.float32, device=dev)

    for _ in range(cfg.EVAL_EPISODES):
        env.reset()
        E = torch.zeros(B, N, dtype=half, device=dev)
        I = torch.zeros(B, N, dtype=half, device=dev)
        st = torch.zeros(B, N, dtype=half, device=dev)
        cts = torch.zeros(B, A, dtype=half, device=dev)

        for _ in range(cfg.MAX_STEPS):
            al = env.alive
            obs = env.obs().to(half)
            sees = env.sees_food(obs)
            tot_seen += (al & sees).float()
            tot_unseen += (al & (~sees)).float()

            act, E, I, st = deliberate_batch(pop, obs, E, I, st, cts, cfg)
            cts = update_fatigue(cts, act)
            tot_act1 += (al & (act == 1)).float()
            tot_act2 += (al & (act == 2)).float()

            env.step(act)
            tot_food += (al & env.ate).float()
            if env.all_done():
                break

    E_ = float(cfg.EVAL_EPISODES)
    metrics = torch.stack((tot_food, tot_seen, tot_unseen), dim=1) / E_

    turn_lim = max(cfg.EVAL_EPISODES, 1)
    c1 = tot_act1.cpu().numpy()
    c2 = tot_act2.cpu().numpy()
    m = metrics.cpu().numpy()
    death = ((c1 > turn_lim) | (c2 > turn_lim)) & ((c1 == 0) | (c2 == 0))
    m[death] = (0.0, 99999.0, 0.0)
    return torch.from_numpy(m).float()


def evaluate_population_ray(pop, cfg):
    """全种群评估：按显存分块并行，返回 [P,3] = (food, seen, unseen)。"""
    if getattr(cfg, 'USE_FP16', True):
        pop.fp16()
    batch = _auto_eval_batch(cfg, pop.device)
    if batch >= pop.P:
        pop.refresh_eff()
        return _eval_chunk_ray(pop, cfg)

    metrics = torch.zeros(pop.P, 3)
    for lo in range(0, pop.P, batch):
        sub = pop[slice(lo, min(lo + batch, pop.P))]
        sub.refresh_eff()
        m = _eval_chunk_ray(sub, cfg)
        metrics[lo:lo + sub.P] = m.cpu()
    return metrics


# ==========================================
# 2. 断点
# ==========================================
def save_neat_checkpoint(path, cfg, next_gen, pop_genes, history,
                         best_state, best_food, best_seen, best_unseen):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {
        'next_gen': int(next_gen),
        'population': pop_genes.pack(),
        'history': history,
        'best_state': best_state,
        'best_food': float(best_food),
        'best_seen': float(best_seen),
        'best_unseen': float(best_unseen),
        'config': io.config_dict(cfg),
        'random_state': random.getstate(),
        'torch_rng_state': torch.get_rng_state(),
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    tmp = path + '.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, path)
    print(f"  [Checkpoint] 断点已保存 -> {path} (next_gen={next_gen})")


def load_neat_checkpoint(path, cfg):
    if not os.path.exists(path):
        return None
    try:
        data = torch.load(path, map_location='cpu', weights_only=False)
    except Exception:
        return None
    saved = data.get('config', {})
    if saved and (saved.get('NUM_COLUMNS') != cfg.NUM_COLUMNS or
                  saved.get('OBS_DIM') != cfg.OBS_DIM):
        print(f"  警告: 断点 {path} 与当前配置不匹配 (N/OBS/ACTION_DIM)，已忽略")
        return None
    return data


# ==========================================
# 3. 主循环
# ==========================================
def run_training_neat(cfg, visualize=False):
    device = _resolve_device(cfg)
    print(f"[NEAT×EI-RNN] device = {device}  "
          f"({'GPU: ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU 回退'})")
    t_program = time.perf_counter()

    pop_genes = NEATPopulation(cfg)
    history = {'gen': [], 'best_food': [], 'avg_food': [],
               'best_seen': [], 'best_unseen': [],
               'n_species': [], 'avg_conns': [], 'avg_cols': []}
    start_gen = 0
    best_state = None
    best_food = -1.0
    best_seen = 0.0
    best_unseen = 0.0

    if cfg.AUTO_RESUME:
        ck = load_neat_checkpoint(cfg.CHECKPOINT_PATH, cfg)
        if ck is not None:
            start_gen = int(ck['next_gen'])
            pop_genes.unpack(ck['population'], cfg)
            history = ck['history']
            best_state = ck.get('best_state')
            best_food = float(ck.get('best_food', -1.0))
            best_seen = float(ck.get('best_seen', 0.0))
            best_unseen = float(ck.get('best_unseen', 0.0))
            if best_state is not None:
                random.setstate(ck['random_state'])
                torch.set_rng_state(ck['torch_rng_state'])
            print(f"=== 检测到断点 [{cfg.CHECKPOINT_PATH}] ===")
            print(f"  已完成 {start_gen} 代 -> 从第 {start_gen} 代接续 | "
                  f"历史最优: Food={best_food:.1f}")

    if not pop_genes.genomes:
        print("初始化 NEAT 种群（256 柱稀疏拓扑 + 创新号注册表）...")
        pop_genes.init_random()
        save_neat_checkpoint(cfg.CHECKPOINT_PATH, cfg, 0, pop_genes, history,
                             best_state, best_food, best_seen, best_unseen)

    gs = GeneStack(cfg, device=device)

    try:
        for gen in range(start_gen, cfg.GENERATIONS):
            t_eval = time.perf_counter()
            tensors = decode_population(pop_genes.genomes, cfg)
            for k in GeneStack.GENES:
                setattr(gs, k, tensors[k].to(device))
            metrics = evaluate_population_ray(gs, cfg)
            eval_time = time.perf_counter() - t_eval

            mn = metrics.numpy()
            threshold = float(cfg.LONG_SNAKE_SCORE_THRESHOLD)
            best_idx = max(range(cfg.POP_SIZE),
                           key=lambda i: _selection_key(
                               (mn[i, 0], mn[i, 1], mn[i, 2]), threshold))
            b_food, b_seen, b_unseen = (float(mn[best_idx, 0]),
                                        float(mn[best_idx, 1]),
                                        float(mn[best_idx, 2]))
            avg_food = float(np.mean(mn[:, 0]))

            if (b_food > best_food or
                    (b_food == best_food and best_food >= 0 and
                     ((b_food > threshold and b_unseen > best_unseen) or
                      (b_food <= threshold and b_seen < best_seen)))):
                best_food = b_food
                best_seen = b_seen
                best_unseen = b_unseen
                # 在 evolve() 替换种群之前抓取当前代最优基因组
                best_state = pop_genes.genomes[best_idx].decode()

            t_ev = time.perf_counter()
            health = pop_genes.evolve(mn[:, 0], gen=gen)
            evolve_time = time.perf_counter() - t_ev

            history['gen'].append(gen)
            history['best_food'].append(b_food)
            history['avg_food'].append(avg_food)
            history['best_seen'].append(b_seen)
            history['best_unseen'].append(b_unseen)
            history['n_species'].append(health['n_species'])
            history['avg_conns'].append(health['avg_conns'])
            history['avg_cols'].append(health['avg_cols'])

            if (gen + 1) % cfg.PRINT_HISTORY_EVERY == 0:
                print(f"Gen {gen + 1}/{cfg.GENERATIONS} | "
                      f"BestFood: {b_food:.2f} | BestSeen: {b_seen:.1f} | "
                      f"BestUnseen: {b_unseen:.1f} | AvgFood: {avg_food:.2f} | "
                      f"Species: {health['n_species']} | "
                      f"Conns: {health['avg_conns']:.0f} | Cols: {health['avg_cols']:.0f} | "
                      f"eval {eval_time:.1f}s / evolve {evolve_time:.1f}s")

            if (gen + 1) % cfg.CHECKPOINT_INTERVAL == 0:
                save_neat_checkpoint(cfg.CHECKPOINT_PATH, cfg, gen + 1, pop_genes,
                                     history, best_state, best_food, best_seen, best_unseen)

    except KeyboardInterrupt:
        nxt = gen + 1 if 'gen' in dir() else start_gen
        print("\n训练被中断 (Ctrl+C)，正在保存断点...")
        save_neat_checkpoint(cfg.CHECKPOINT_PATH, cfg, nxt, pop_genes,
                             history, best_state, best_food, best_seen, best_unseen)
        sys.exit(0)

    print(f"\nTotal runtime: {time.perf_counter() - t_program:.1f}s")

    if best_state is None:
        best_state = pop_genes.genomes[0].decode()
    brain = io.load_brain_state(best_state, cfg)
    io.save_best_model(cfg.BEST_MODEL_PATH, brain, cfg, best_food, best_seen + best_unseen)
    print(f"\n最优模型已保存: {cfg.BEST_MODEL_PATH} "
          f"(Food={best_food:.2f}, Seen={best_seen:.1f}, Unseen={best_unseen:.1f})")

    if os.path.exists(cfg.CHECKPOINT_PATH):
        os.remove(cfg.CHECKPOINT_PATH)
        print(f"训练已完成，已删除临时断点: {cfg.CHECKPOINT_PATH}")

    print("\n--- NEAT × EI-RNN 契合度总结 ---")
    print(f"物种数范围: {min(history['n_species'])} ~ {max(history['n_species'])}"
          f"（目标 {cfg.SPECIES_TARGET}，>1 说明物种化在保护多样性）")
    print(f"平均激活连接/柱: {history['avg_conns'][0]:.0f} -> {history['avg_conns'][-1]:.0f}"
          f" / {history['avg_cols'][0]:.0f} -> {history['avg_cols'][-1]:.0f}"
          f"（add-node/add-connection 结构生长）")
    print(f"创新号注册表规模: {pop_genes.registry.counter} 条 (src,dst) 基因")

    if visualize:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            ax = axes[0]
            ax.plot(history['gen'], history['best_food'], label='Best Food', color='red', marker='o', markersize=3)
            ax.plot(history['gen'], history['avg_food'], label='Avg Food', color='blue', alpha=0.6)
            ax.set_title("NEAT × EI-RNN — Food Count")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Food")
            ax.legend(); ax.grid(True, alpha=0.3)
            ax = axes[1]
            ax.plot(history['gen'], history['n_species'], color='green', marker='^', markersize=3)
            ax.axhline(cfg.SPECIES_TARGET, color='gray', linestyle='--', label='target')
            ax.set_title("Species Count (speciation health)")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Species")
            ax.legend(); ax.grid(True, alpha=0.3)
            ax = axes[2]
            ax.plot(history['gen'], history['avg_conns'], label='active conns', color='purple', marker='s', markersize=3)
            ax.plot(history['gen'], history['avg_cols'], label='active cols', color='orange', marker='o', markersize=3)
            ax.set_title("Topology Growth (NEAT augmenting)")
            ax.set_xlabel("Generation")
            ax.set_ylabel("count")
            ax.legend(); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig('test8_history.png', dpi=100)
            plt.close(fig)
            print("历史曲线已保存: test8_history.png")
        except Exception as e:
            print(f"(matplotlib 曲线跳过: {e})")

    return brain, history


# ==========================================
# 4. 播放最优模型（CPU RaySnakeEnv）
# ==========================================
def play_best(cfg, max_steps=300):
    from einbrain import EIBrainRegion
    from einbrain.deliberation import deliberate_action
    from einbrain.env import RaySnakeEnv

    res = io.load_best_model_brain(cfg.BEST_MODEL_PATH, cfg)
    if res is None:
        print("无最优模型可播放")
        return
    brain, food, steps = res
    env = RaySnakeEnv(grid_size=cfg.GRID_SIZE, max_steps=cfg.MAX_STEPS, cfg=cfg)
    brain.reset_runtime()
    obs = env.reset()
    E = torch.zeros(brain.N)
    I = torch.zeros(brain.N)
    G = cfg.GRID_SIZE

    def render():
        g = [['.' for _ in range(G)] for _ in range(G)]
        g[env.food[0]][env.food[1]] = '*'
        for i, (r, c) in enumerate(env.body):
            g[r][c] = 'H' if i == 0 else '#'
        print('  ' + '\n  '.join(''.join(row) for row in g))

    ep_food = 0
    done = False
    for s in range(max_steps):
        action, _, E, I = deliberate_action(brain, obs, E, I)
        brain.update_fatigue(action)
        obs, ate, done, trunc = env.step(action)
        done = done or trunc
        if ate:
            ep_food += 1
        if s % 10 == 0:
            print(f"\n--- Step {s} (score {ep_food}) ---")
            render()
        if done:
            print(f"\n--- 死亡 @ step {s} ---")
            render()
            break
    print(f"\nPlay done: Food={ep_food}, Steps={s + 1}")


# ==========================================
# 5. 入口
# ==========================================
def make_smoke_config():
    cfg = NEATConfig()
    cfg.POP_SIZE = 16
    cfg.GENERATIONS = 2
    cfg.NUM_COLUMNS = 24
    cfg.INIT_DENSITY = min(0.15, 40.0 / cfg.NUM_COLUMNS)
    cfg.EVAL_EPISODES = 1
    cfg.MAX_STEPS = 60
    cfg.FRAME_RATE = 5
    cfg.DEVICE = 'cpu'
    cfg.USE_FP16 = False
    cfg.CHECKPOINT_INTERVAL = 1
    cfg.PRINT_HISTORY_EVERY = 1
    cfg.CHECKPOINT_PATH = 'test8_smoke_checkpoint.pth'
    cfg.BEST_MODEL_PATH = 'test8_smoke_best.pth'
    cfg.SPECIES_TARGET = 4
    return cfg


def main():
    ap = argparse.ArgumentParser(description='test8 — 真正 NEAT × EI-RNN（K=5, 32 维射线观测）')
    ap.add_argument('--smoke', action='store_true', help='小规模快速自检')
    ap.add_argument('--gens', type=int, default=None)
    ap.add_argument('--pop', type=int, default=None)
    ap.add_argument('--columns', type=int, default=None)
    ap.add_argument('--episodes', type=int, default=None)
    ap.add_argument('--max-steps', type=int, default=None)
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--play', action='store_true', help='加载最优模型播放一局')
    args = ap.parse_args()

    if args.smoke:
        cfg = make_smoke_config()
    else:
        cfg = NEATConfig()
    if args.gens:
        cfg.GENERATIONS = args.gens
    if args.pop:
        cfg.POP_SIZE = args.pop
    if args.columns:
        cfg.NUM_COLUMNS = args.columns
        cfg.INIT_DENSITY = min(0.15, 40.0 / cfg.NUM_COLUMNS)
    if args.episodes:
        cfg.EVAL_EPISODES = args.episodes
    if args.max_steps:
        cfg.MAX_STEPS = args.max_steps
    if args.device:
        cfg.DEVICE = args.device

    if args.play:
        play_best(cfg)
        return

    run_training_neat(cfg, visualize=False)


if __name__ == '__main__':
    main()
