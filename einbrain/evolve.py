"""进化训练管线（test5d 语义）。

包含：单个体评估、精英排序键、交替冻结进化（G1 结构 / G2 动力学 / G3 激素）、
多进程并行评估 + 两阶段快速筛选，以及完整的 run_evolution 主循环。
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np
import torch

from .brain import EIBrainRegion
from .deliberation import deliberate_action
from .env import SnakeEnv, _obs_sees_food
from . import io

# ==================== 评估 ====================


def evaluate_individual(brain, env, render=False, max_steps=None, episodes=None):
    """评估单个个体指定局数，返回三元组 (avg_food, avg_seen, avg_unseen)。

    - seen   = 看到食物时走的步数（越短越好 → 快速吃子）
    - unseen = 没看到食物时走的步数（越长越好 → 巡航探索/活得久）
    - 硬性淘汰『单侧转弯判死』：阈值随局数等比缩放。
    """
    cfg = brain.cfg
    if max_steps is None:
        max_steps = cfg.MAX_STEPS
    if episodes is None:
        episodes = cfg.EVAL_EPISODES

    total_foods = []
    total_seen_list = []
    total_unseen_list = []
    total_action_counts = [0, 0, 0]

    for ep in range(episodes):
        brain.reset_runtime()
        obs = env.reset()
        E = torch.zeros(brain.N)
        I = torch.zeros(brain.N)

        ep_food = 0
        ep_seen = 0
        ep_unseen = 0
        steps = 0
        done = False

        while not done and steps < max_steps:
            if _obs_sees_food(obs):
                ep_seen += 1
            else:
                ep_unseen += 1

            action, avg_logits, E, I = deliberate_action(brain, obs, E, I)
            brain.update_fatigue(action)
            total_action_counts[action] += 1

            next_obs, ate_food, done, truncated = env.step(action)
            done = done or truncated
            if ate_food:
                ep_food += 1
            obs = next_obs
            steps += 1

        total_foods.append(ep_food)
        total_seen_list.append(ep_seen)
        total_unseen_list.append(ep_unseen)

    avg_food = np.mean(total_foods)
    avg_seen = np.mean(total_seen_list)
    avg_unseen = np.mean(total_unseen_list)

    turn_lim = max(episodes, 1)
    if (total_action_counts[1] > turn_lim or total_action_counts[2] > turn_lim) and \
       (total_action_counts[1] == 0 or total_action_counts[2] == 0):
        avg_food = 0
        avg_seen = 99999
        avg_unseen = 0

    if render:
        print(f"  [Render] Food: {avg_food:.1f}, Seen: {avg_seen:.1f}, Unseen: {avg_unseen:.1f}")

    return avg_food, avg_seen, avg_unseen


def _selection_key(metric, threshold):
    """进化筛选排序键（25 分前后切换筛选压力）。

    food <= threshold：key=(food, -seen, unseen)  先保『看见秒吃』（seen 短）
    food >  threshold：key=(food, unseen, -seen)  先保『看不见活得久』（unseen 长）
    """
    food, seen, unseen = metric
    if food > threshold:
        return (food, unseen, -seen)
    return (food, -seen, unseen)


# ==================== 进化算子 ====================


def _freeze_active_groups(gen, cfg):
    """按 CYCLE_PATTERN 周期计算第 gen 代激活的参数组集合。"""
    pattern = getattr(cfg, 'CYCLE_PATTERN', [('G2',), ('G1',)])
    active = set(pattern[gen % len(pattern)])
    if not bool(getattr(cfg, 'TRAIN_HORMONE_NET', False)):
        active.discard('G3')
    return frozenset(active)


def _dynamic_mutation_rates(cfg, gen):
    """test5d v2 动态变异：拓扑余弦退火，动力学指数衰减。"""
    cos_mode = getattr(cfg, 'EVO_COS_MODE', 'anneal')
    if cos_mode == 'oscillate':
        period = max(1, int(getattr(cfg, 'EVO_COS_PERIOD', 100)))
        cos_factor = 0.5 + 0.5 * math.cos(2.0 * math.pi * gen / period)
    else:
        cos_factor = 0.5 + 0.5 * math.cos(math.pi * gen / max(1, float(cfg.GENERATIONS)))
    dyn_tau = max(1e-6, float(getattr(cfg, 'EVO_DYN_DECAY_TAU', 33)))
    dyn_factor = math.exp(-gen / dyn_tau)
    return {
        'topo_mut_prob': cfg.TOPOLOGY_MUT_PROB * cos_factor,
        'mask_mut_rate': cfg.MUT_RATE * cos_factor,
        'weight_mut_frac': cfg.WEIGHT_MUT_FRAC * cos_factor,
        'weight_mut_std': cfg.WEIGHT_MUT_STD * cos_factor,
        'tau_e_mut_std': cfg.TAU_E_MUT_STD * dyn_factor,
        'w_ei_mut_std': cfg.W_EI_MUT_STD * dyn_factor,
        'w_ie_mut_std': cfg.W_IE_MUT_STD * dyn_factor,
    }


def evolve_topology(population, metrics_list, cfg, gen=0):
    """进化下一代（B 方案：软冻结交替优化 G1/G2/G3 三组参数）。

    被冻结的组不交叉、不变异，原样继承父代 p1（避免交叉制造失配嵌合体）。
    """
    active = _freeze_active_groups(gen, cfg)
    has_g1 = 'G1' in active
    has_g2 = 'G2' in active
    has_g3 = 'G3' in active
    mut = _dynamic_mutation_rates(cfg, gen)

    threshold = float(getattr(cfg, 'LONG_SNAKE_SCORE_THRESHOLD', 3.0))
    sorted_indices = sorted(
        range(len(metrics_list)),
        key=lambda i: _selection_key(metrics_list[i], threshold),
        reverse=True)

    elite_idx = sorted_indices[:cfg.ELITE_SIZE]
    elites = [population[i].clone() for i in elite_idx]
    new_pop = [e.clone() for e in elites]

    while len(new_pop) < len(population):
        p1, p2 = random.sample(elites, 2)
        child = p1.clone()
        N = child.N

        with torch.no_grad():
            if has_g1:
                col_mask = torch.rand(N) > 0.5
                row_mask = col_mask.unsqueeze(1)
                col_mask_2d = col_mask.unsqueeze(0)
                same_p1 = row_mask & col_mask_2d
                same_p2 = (~row_mask) & (~col_mask_2d)

                child.W_in.data = torch.where(col_mask.unsqueeze(1), p1.W_in.data, p2.W_in.data)
                child.M_in = torch.where(col_mask.unsqueeze(1), p1.M_in, p2.M_in)
                child.W_rec.data = torch.where(same_p1, p1.W_rec.data,
                                      torch.where(same_p2, p2.W_rec.data,
                                          torch.where(torch.rand_like(p1.W_rec.data) > 0.5,
                                                      p1.W_rec.data, p2.W_rec.data)))
                child.M_rec = torch.where(same_p1, p1.M_rec,
                                 torch.where(same_p2, p2.M_rec,
                                     torch.where(torch.rand_like(p1.M_rec) > 0.5,
                                                 p1.M_rec, p2.M_rec)))
                child.W_out.data = torch.where(col_mask.unsqueeze(0), p1.W_out.data, p2.W_out.data)
                child.M_out = torch.where(col_mask.unsqueeze(0), p1.M_out, p2.M_out)
                mask_out = torch.rand_like(p1.b_out.data) > 0.5
                child.b_out.data = torch.where(mask_out, p1.b_out.data, p2.b_out.data)

            if has_g2:
                col_mask2 = torch.rand(N) > 0.5
                child.tau_e_init.data = torch.where(col_mask2, p1.tau_e_init.data, p2.tau_e_init.data)
                child.w_ei.data = torch.where(col_mask2, p1.w_ei.data, p2.w_ei.data)
                child.w_ie.data = torch.where(col_mask2, p1.w_ie.data, p2.w_ie.data)

            if has_g3:
                for attr in ['W_hormone1', 'b_hormone1', 'W_excit', 'b_excit', 'W_inhib', 'b_inhib']:
                    p1_t = getattr(p1, attr).data
                    p2_t = getattr(p2, attr).data
                    mask = torch.rand_like(p1_t) > 0.5
                    getattr(child, attr).data = torch.where(mask, p1_t, p2_t)

        with torch.no_grad():
            if has_g1:
                if random.random() < mut['topo_mut_prob']:
                    m_attr = random.choice(['M_in', 'M_rec', 'M_out'])
                    m_tensor = getattr(child, m_attr)
                    mut_mask = torch.rand_like(m_tensor) < mut['mask_mut_rate']
                    m_tensor[mut_mask] = 1.0 - m_tensor[mut_mask]
                for attr in ['W_in', 'W_rec', 'W_out', 'b_out']:
                    w_tensor = getattr(child, attr).data
                    noise = torch.randn_like(w_tensor) * mut['weight_mut_std']
                    noise_mask = torch.rand_like(w_tensor) < mut['weight_mut_frac']
                    setattr(child, attr, torch.nn.Parameter(w_tensor + noise * noise_mask))

            if has_g2:
                tau_noise = torch.randn_like(child.tau_e_init.data) * mut['tau_e_mut_std']
                child.tau_e_init.data = torch.clamp(
                    child.tau_e_init.data + tau_noise, cfg.TAU_E_MIN, cfg.TAU_E_MAX)
                w_ei_noise = torch.randn_like(child.w_ei.data) * mut['w_ei_mut_std']
                child.w_ei.data = torch.clamp(
                    child.w_ei.data + w_ei_noise, cfg.W_EI_MIN, cfg.W_EI_MAX)
                w_ie_noise = torch.randn_like(child.w_ie.data) * mut['w_ie_mut_std']
                child.w_ie.data = torch.clamp(
                    child.w_ie.data + w_ie_noise, cfg.W_IE_MIN, cfg.W_IE_MAX)

            if has_g3:
                for attr in ['W_hormone1', 'b_hormone1', 'W_excit', 'b_excit', 'W_inhib', 'b_inhib']:
                    w = getattr(child, attr).data
                    noise = torch.randn_like(w) * cfg.HORMONE_MUT_STD
                    mask = torch.rand_like(w) < cfg.HORMONE_MUT_FRAC
                    getattr(child, attr).data = w + noise * mask

        child.refresh_cached()
        child.save_genetic_baseline()
        new_pop.append(child)

    return new_pop


# ==================== 并行评估 + 两阶段筛选 ====================


def _worker_cfg_from_dict(cfg_dict):
    """从配置 dict 重建轻量 cfg 对象（用于子进程 worker）。"""
    tmp = type('_RtCfg', (), {})()
    for k, v in cfg_dict.items():
        setattr(tmp, k, v)
    return tmp


def _eval_worker_serialize(task):
    """多进程 worker：分片批处理，评估一个分片内的所有个体。

    task = ([(idx, 基因 dict), ...], 配置 dict, episodes, max_steps)。
    """
    items, cfg_dict, episodes, max_steps = task
    rt_cfg = _worker_cfg_from_dict(cfg_dict)
    random.seed(os.getpid())
    env = SnakeEnv(grid_size=rt_cfg.GRID_SIZE)
    results = []
    for idx, genes in items:
        brain = io.load_brain_state(genes, rt_cfg)
        results.append((idx, evaluate_individual(brain, env,
                                                 max_steps=max_steps,
                                                 episodes=episodes)))
    return results


def _eval_batch(inds, cfg, pool, episodes, env=None):
    """对一组个体评估指定局数并返回指标列表（pool 不为 None 时并行）。"""
    if pool is not None:
        n = len(inds)
        if n == 0:
            return []
        cfg_dict = io.config_dict(cfg)
        n_chunks = max(1, min(pool._processes, n))
        chunk_size = (n + n_chunks - 1) // n_chunks
        chunks = []
        for c in range(n_chunks):
            lo = c * chunk_size
            hi = min(lo + chunk_size, n)
            items = [(i, io.save_brain_state(inds[i], use_half=False))
                     for i in range(lo, hi)]
            chunks.append((items, cfg_dict, episodes, cfg.MAX_STEPS))
        flat = []
        for chunk_results in pool.map(_eval_worker_serialize, chunks):
            flat.extend(chunk_results)
        flat.sort(key=lambda x: x[0])
        return [m for _, m in flat]

    results = []
    for ind in inds:
        results.append(evaluate_individual(ind, env, episodes=episodes))
    return results


def evaluate_population(population, cfg, env, pool=None):
    """统一评估入口：A 方案（并行）+ C 方案（两阶段筛选）。"""
    episodes = cfg.EVAL_EPISODES
    screen_ep = int(getattr(cfg, 'SCREEN_EPISODES', 1))

    use_screen = (bool(getattr(cfg, 'SCREEN_ENABLE', False)) and
                  screen_ep >= 1 and screen_ep < episodes)

    if not use_screen:
        return _eval_batch(population, cfg, pool, episodes, env)

    quick = _eval_batch(population, cfg, pool, screen_ep, env)

    if bool(getattr(cfg, 'SCREEN_AUTO_FALLBACK', True)) and max(m[0] for m in quick) <= 0:
        return _eval_batch(population, cfg, pool, episodes, env)

    k = cfg.ELITE_SIZE * int(getattr(cfg, 'SCREEN_MULTIPLIER', 3))
    k = max(1, min(k, len(population)))
    if k >= len(population):
        return _eval_batch(population, cfg, pool, episodes, env)

    threshold = float(getattr(cfg, 'LONG_SNAKE_SCORE_THRESHOLD', 3.0))
    order = sorted(range(len(population)),
                   key=lambda i: _selection_key(quick[i], threshold), reverse=True)
    refine_idx = set(order[:k])
    refine_list = [population[i] for i in range(len(population)) if i in refine_idx]

    extra = _eval_batch(refine_list, cfg, pool, episodes - screen_ep, env)

    metrics = [None] * len(population)
    ref_pos = 0
    for i in range(len(population)):
        if i in refine_idx:
            f_q, se_q, un_q = quick[i]
            f_e, se_e, un_e = extra[ref_pos]
            ref_pos += 1
            w_q = screen_ep / episodes
            w_e = (episodes - screen_ep) / episodes
            metrics[i] = (f_q * w_q + f_e * w_e,
                          se_q * w_q + se_e * w_e,
                          un_q * w_q + un_e * w_e)
        else:
            metrics[i] = quick[i]
    return metrics


def make_eval_pool(cfg):
    """按配置创建评估进程池；PARALLEL_EVAL=False 时返回 None（串行）。"""
    if not cfg.PARALLEL_EVAL or cfg.POP_SIZE < 2:
        return None
    n_workers = int(getattr(cfg, 'NUM_WORKERS', 0))
    if n_workers <= 0:
        n_workers = min(os.cpu_count() or 1, 16)
    n_workers = max(1, min(n_workers, cfg.POP_SIZE))
    return mp.Pool(n_workers)


# ==================== 主循环 ====================


def run_evolution(cfg, visualize=False):
    """进化主训练循环（断点续训 + 每 N 代自动保存 + 完成保存最优模型）。

    返回 (best_brain, history)。
    """
    from .vis import plot_history, visualize_best_brain_play

    env = SnakeEnv(grid_size=cfg.GRID_SIZE)
    t_program = time.perf_counter()

    start_gen = 0
    population = None
    history = {'gen': [], 'best_food': [], 'avg_food': [],
               'best_seen': [], 'best_unseen': []}
    cum_eval_time = 0.0
    cum_evolve_time = 0.0
    best_ever_brain = None
    best_ever_food = -1.0
    best_ever_steps = 0.0
    best_ever_seen = 0.0
    best_ever_unseen = 0.0

    if cfg.AUTO_RESUME and os.path.exists(cfg.CHECKPOINT_PATH):
        ckpt = io.load_checkpoint(cfg.CHECKPOINT_PATH, cfg)
        if ckpt is not None:
            start_gen = ckpt['next_gen']
            population = ckpt['population']
            history = ckpt['history']
            cum_eval_time = ckpt['cum_eval_time']
            cum_evolve_time = ckpt['cum_evolve_time']
            best_ever_brain = ckpt['best_brain']
            best_ever_food = ckpt['best_food']
            best_ever_steps = ckpt['best_steps']
            if best_ever_brain is not None:
                best_ever_seen = float(getattr(best_ever_brain, '_track_seen', 0.0))
                best_ever_unseen = float(getattr(best_ever_brain, '_track_unseen', 0.0))
            history.setdefault('best_seen', list(history.get('best_steps', [])))
            history.setdefault('best_unseen', list(history.get('best_steps', [])))
            print(f"\n=== 检测到断点 [{cfg.CHECKPOINT_PATH}] ===")
            print(f"  {cfg.GENERATIONS} 代中已完成 {history['gen'][-1] + 1 if history['gen'] else 0} 代"
                  f" -> 从第 {start_gen} 代接续 | "
                  f"历史最优: Food={best_ever_food:.1f}, Steps={best_ever_steps:.1f}")

    if population is None:
        print("Initializing Population...")
        population = [EIBrainRegion(cfg) for _ in range(cfg.POP_SIZE)]
        for ind in population:
            ind.save_genetic_baseline()

        if cfg.SEED_FROM_BEST:
            seed_result = io.load_best_model_brain(cfg.BEST_MODEL_PATH, cfg)
            if seed_result is not None:
                seed, seed_food, seed_steps = seed_result
                population[0] = seed
                best_ever_brain = seed
                best_ever_food = seed_food
                best_ever_steps = seed_steps
                print(f"  [Seed] 已注入最优模型 {cfg.BEST_MODEL_PATH} 作为种群种子"
                      f"（上轮 Food={seed_food:.1f}, Steps={seed_steps:.1f}）")
            else:
                print(f"  [Seed] 未发现可用的最优模型种子，全新随机初始化")

        io.save_checkpoint(cfg.CHECKPOINT_PATH, cfg, start_gen, population, history,
                           cum_eval_time, cum_evolve_time,
                           best_ever_brain, best_ever_food, best_ever_steps)

    eval_pool = make_eval_pool(cfg)
    if eval_pool is not None:
        print(f"[Parallel] 多进程并行评估已启用：{eval_pool._processes} workers")

    cur_gen = None
    try:
        for gen in range(start_gen, cfg.GENERATIONS):
            cur_gen = gen
            t_gen_start = time.perf_counter()
            metrics = evaluate_population(population, cfg, env, eval_pool)
            eval_time = time.perf_counter() - t_gen_start
            cum_eval_time += eval_time

            threshold = float(getattr(cfg, 'LONG_SNAKE_SCORE_THRESHOLD', 3.0))
            best_idx = max(range(len(metrics)),
                           key=lambda i: _selection_key(metrics[i], threshold))
            best_food = metrics[best_idx][0]
            best_seen = metrics[best_idx][1]
            best_unseen = metrics[best_idx][2]
            avg_food = np.mean([m[0] for m in metrics])

            history['gen'].append(gen)
            history['best_food'].append(best_food)
            history['avg_food'].append(avg_food)
            history['best_seen'].append(best_seen)
            history['best_unseen'].append(best_unseen)

            best_brain = population[best_idx]

            prev_seen = float(getattr(best_ever_brain, '_track_seen', best_ever_seen))
            prev_unseen = float(getattr(best_ever_brain, '_track_unseen', best_ever_unseen))
            if (best_food > best_ever_food or
                    (best_food == best_ever_food and
                     ((best_food > threshold and best_unseen > prev_unseen) or
                      (best_food <= threshold and best_seen < prev_seen)))):
                best_ever_food = best_food
                best_ever_steps = best_seen + best_unseen
                best_ever_seen = best_seen
                best_ever_unseen = best_unseen
                best_ever_brain = best_brain.clone()
                best_ever_brain._track_seen = best_seen
                best_ever_brain._track_unseen = best_unseen

            if gen < cfg.GENERATIONS - 1:
                t_ev_start = time.perf_counter()
                population = evolve_topology(population, metrics, cfg, gen=gen)
                evolve_time = time.perf_counter() - t_ev_start
                cum_evolve_time += evolve_time
            else:
                evolve_time = 0.0

            print(f"Gen {gen+1}/{cfg.GENERATIONS} | "
                  f"BestFood: {best_food:.1f} | BestSeen: {best_seen:.1f} | "
                  f"BestUnseen: {best_unseen:.1f} | AvgFood: {avg_food:.1f} | "
                  f"eval {eval_time:.1f}s / evolve {evolve_time:.1f}s")

            if (gen + 1) % cfg.CHECKPOINT_INTERVAL == 0:
                io.save_checkpoint(cfg.CHECKPOINT_PATH, cfg, gen + 1, population,
                                   history, cum_eval_time, cum_evolve_time,
                                   best_ever_brain, best_ever_food, best_ever_steps)

    except KeyboardInterrupt:
        if eval_pool is not None:
            eval_pool.terminate()
            eval_pool.join()
        nxt = cur_gen if cur_gen is not None else start_gen
        print("\n训练被中断 (Ctrl+C)，正在保存断点以供下次自动接续...")
        io.save_checkpoint(cfg.CHECKPOINT_PATH, cfg, nxt, population,
                           history, cum_eval_time, cum_evolve_time,
                           best_ever_brain, best_ever_food, best_ever_steps)
        sys.exit(0)

    if eval_pool is not None:
        eval_pool.close()
        eval_pool.join()

    print(f"\nTotal runtime: {time.perf_counter() - t_program:.1f}s "
          f"(eval {cum_eval_time:.1f}s / evolve {cum_evolve_time:.1f}s)")

    if best_ever_brain is None:
        best_ever_brain = population[0]
    io.save_best_model(cfg.BEST_MODEL_PATH, best_ever_brain, cfg,
                       best_ever_food, best_ever_steps)
    print(f"\n最优模型已保存: {cfg.BEST_MODEL_PATH} "
          f"(Food={best_ever_food:.1f}, Steps={best_ever_steps:.1f})")

    if os.path.exists(cfg.CHECKPOINT_PATH):
        os.remove(cfg.CHECKPOINT_PATH)
        print(f"训练已完成，已删除临时断点: {cfg.CHECKPOINT_PATH}")

    if visualize:
        plot_history(history)
        visualize_best_brain_play(best_ever_brain, cfg, max_steps=300)

    return best_ever_brain, history
