# -*- coding: utf-8 -*-
"""
test5a_lunar_smoke_test.py
==========================
test5a_lunar.py 的小规模冒烟测试：
以极小种群/极少代数跑完整个训练循环（评估 -> 交叉变异 -> 断点保存/加载 ->
最优模型保存/加载），确认管道无 bug，随后删除临时产物。
"""
import os
import gymnasium as gym
import numpy as np
import test5a_lunar as m


class SmokeConfig(m.Config):
    POP_SIZE = 16
    ELITE_SIZE = 4
    GENERATIONS = 2
    EVAL_EPISODES = 2
    SCREEN_EPISODES = 1
    MAX_STEPS = 30
    PARALLEL_EVAL = False          # 冒烟测试走串行，避免进程池开销
    SCREEN_AUTO_FALLBACK = False   # 禁用回退，强制走两阶段筛选路径
    CHECKPOINT_PATH = 'test5a_lunar_smoke_ckpt.pth'
    BEST_MODEL_PATH = 'test5a_lunar_smoke_best.pth'


def main():
    cfg = SmokeConfig()
    env = gym.make(cfg.ENV_NAME)

    # ---- 完整训练循环（2 代）----
    population = [m.EIBrainRegion(cfg) for _ in range(cfg.POP_SIZE)]
    for ind in population:
        ind.save_genetic_baseline()

    history = {'gen': [], 'best_reward': [], 'avg_reward': [], 'best_steps': []}
    for gen in range(cfg.GENERATIONS):
        metrics = m.evaluate_population(population, cfg, env, None)
        best_idx = max(range(len(metrics)), key=lambda i: (metrics[i][0], -metrics[i][1]))
        history['gen'].append(gen)
        history['best_reward'].append(metrics[best_idx][0])
        history['avg_reward'].append(float(np.mean([x[0] for x in metrics])))
        history['best_steps'].append(metrics[best_idx][1])
        population = m.evolve_topology(population, metrics, cfg, gen=gen)
    print(f"训练循环 OK | best_reward={history['best_reward']} | avg_reward={history['avg_reward']}")

    # ---- 断点 保存/加载 往返 ----
    m.save_checkpoint(cfg.CHECKPOINT_PATH, cfg, cfg.GENERATIONS, population, history,
                      best_brain=population[0], best_reward=history['best_reward'][-1])
    ck = m.load_checkpoint(cfg.CHECKPOINT_PATH, cfg)
    assert ck is not None and len(ck['population']) == cfg.POP_SIZE, "断点加载失败"
    assert ck['next_gen'] == cfg.GENERATIONS, "next_gen 不一致"
    print(f"断点往返 OK | loaded_pop={len(ck['population'])} | next_gen={ck['next_gen']}")

    # ---- 最优模型 保存/加载 往返 ----
    m.save_best_model(cfg.BEST_MODEL_PATH, population[0], cfg, 123.0, 45.0)
    br = m.load_best_model_brain(cfg.BEST_MODEL_PATH, cfg)
    assert br is not None, "最优模型加载失败"
    assert abs(br[1] - 123.0) < 1e-6, "最优模型 reward 往返不一致"
    print(f"最优模型往返 OK | reward={br[1]} | steps={br[2]}")

    # ---- 评估既有加载模型（确认模型可用）----
    r, s = m.evaluate_individual(br[0], env, max_steps=20, episodes=1)
    print(f"加载模型复评 OK | reward={r:.2f} | steps={s}")

    env.close()

    # ---- 清理临时产物 ----
    for p in (cfg.CHECKPOINT_PATH, cfg.BEST_MODEL_PATH, cfg.CHECKPOINT_PATH + '.tmp'):
        if os.path.exists(p):
            os.remove(p)
    print("SMOKE TRAIN OK (临时产物已清理)")


if __name__ == "__main__":
    main()