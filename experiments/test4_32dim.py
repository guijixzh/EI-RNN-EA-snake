# ==========================================
# test4_32dim.py — test4 架构 + 32 维新环境对照实验
#
# 目的：用 test4 的原始架构（N=64，M/W 独立，掩码翻转，预测编码）
# 在 32 维 RaySnakeEnv 下跑 20 代，验证 EI-RNN 从零训练是否可行。
#
# 对照组：test4 用 24 维环境成功从零训练。
# 本实验把环境换成 32 维，其他保持 test4 原样。
# ==========================================

import copy
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from einbrain.env import RaySnakeEnv, ray_obs_sees_food


# ==========================================
# 1. 脑模型（test4 原始 EIBrainRegion）
# ==========================================
class EIBrainRegion(nn.Module):
    def __init__(self, num_columns=64, obs_dim=32, action_dim=3, init_density=0.15):
        super().__init__()
        self.N = num_columns
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # 基因型：拓扑掩码（独立于权重）
        self.M_in = (torch.rand(num_columns, obs_dim) < init_density).float()
        self.M_rec = (torch.rand(num_columns, num_columns) < init_density).float()
        torch.diagonal(self.M_rec).zero_()
        self.M_out = (torch.rand(action_dim, num_columns) < init_density).float()

        # 表现型：突触权重
        self.W_in = nn.Parameter(torch.randn(num_columns, obs_dim) * 0.1)
        self.W_rec = nn.Parameter(torch.randn(num_columns, num_columns) * 0.05)
        self.W_out = nn.Parameter(torch.randn(action_dim, num_columns) * 0.1)
        self.W_pred = nn.Parameter(torch.randn(num_columns, num_columns) * 0.1)

        # 固定 E-I 动力学参数
        self.tau_e = 0.7
        self.w_ei = 2.0
        self.w_ie = 2.0

        self.baseline = None

    def forward(self, obs_t, E_prev, I_prev):
        ext_in = torch.matmul(self.W_in * self.M_in, obs_t)
        rec_in = torch.matmul(self.W_rec * self.M_rec, E_prev)
        total_in = ext_in + rec_in
        E_new = torch.sigmoid(total_in + self.tau_e * E_prev - self.w_ei * I_prev)
        I_new = torch.sigmoid(self.w_ie * E_new)
        pred_in = torch.matmul(self.W_pred, E_prev)
        error = total_in - pred_in
        action_logits = torch.matmul(self.W_out * self.M_out, E_new)
        return action_logits, E_new, I_new, error, total_in

    def apply_pc_update(self, E_prev, error, total_in, obs_t, lr=0.001, decay=0.0001):
        with torch.no_grad():
            self.W_pred += lr * torch.outer(error, E_prev) - decay * self.W_pred
            self.W_in += lr * torch.outer(error, obs_t) - decay * self.W_in

    def save_genetic_baseline(self):
        self.baseline = {
            'W_in': self.W_in.data.clone(), 'W_rec': self.W_rec.data.clone(),
            'W_out': self.W_out.data.clone(), 'W_pred': self.W_pred.data.clone(),
            'M_in': self.M_in.clone(), 'M_rec': self.M_rec.clone(),
            'M_out': self.M_out.clone()
        }

    def restore_genetic_baseline(self):
        if self.baseline:
            self.W_in.data = self.baseline['W_in'].clone()
            self.W_rec.data = self.baseline['W_rec'].clone()
            self.W_out.data = self.baseline['W_out'].clone()
            self.W_pred.data = self.baseline['W_pred'].clone()
            self.M_in = self.baseline['M_in'].clone()
            self.M_rec = self.baseline['M_rec'].clone()
            self.M_out = self.baseline['M_out'].clone()


# ==========================================
# 2. 评估（32 维环境 + food 计数）
# ==========================================
def evaluate_individual(brain, env, max_steps=1000, episodes=5):
    original_baseline = {k: v.clone() for k, v in brain.baseline.items()} if brain.baseline else None
    total_foods = []
    total_steps = []

    for _ in range(episodes):
        brain.restore_genetic_baseline()
        obs = env.reset()
        E = torch.zeros(brain.N)
        I = torch.zeros(brain.N)
        ep_food = 0
        steps = 0
        done = False

        while not done and steps < max_steps:
            obs_t = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                E_old = E.clone()
                logits, E, I, error, total_in = brain(obs_t, E, I)
                action = torch.argmax(logits).item()
            brain.apply_pc_update(E_old, error, total_in, obs_t)
            next_obs, ate, done, trunc = env.step(action)
            done = done or trunc
            if ate:
                ep_food += 1
            obs = next_obs
            steps += 1

        total_foods.append(ep_food)
        total_steps.append(steps)

    # 软拉马克遗传
    if original_baseline:
        with torch.no_grad():
            factor = 0.2
            brain.W_in.data = (1 - factor) * original_baseline['W_in'] + factor * brain.W_in.data
            brain.W_pred.data = (1 - factor) * original_baseline['W_pred'] + factor * brain.W_pred.data
            brain.save_genetic_baseline()

    avg_food = np.mean(total_foods)
    avg_steps = np.mean(total_steps)
    return avg_food, avg_steps


# ==========================================
# 3. 进化（test4 原始逻辑）
# ==========================================
def evolve_topology(population, fitnesses, elite_size=64, mut_rate=0.05):
    elite_idx = np.argsort(fitnesses)[-elite_size:]
    elites = [copy.deepcopy(population[i]) for i in elite_idx]
    new_pop = [copy.deepcopy(e) for e in elites]

    while len(new_pop) < len(population):
        p1, p2 = random.sample(elites, 2)
        child = copy.deepcopy(p1)
        N = child.N

        col_mask = torch.rand(N) > 0.5
        row_mask = col_mask.unsqueeze(1)
        col_mask_2d = col_mask.unsqueeze(0)
        same_p1 = row_mask & col_mask_2d
        same_p2 = (~row_mask) & (~col_mask_2d)

        with torch.no_grad():
            child.W_in.data = torch.where(col_mask.unsqueeze(1), p1.W_in.data, p2.W_in.data)
            child.M_in = torch.where(col_mask.unsqueeze(1), p1.M_in, p2.M_in)

            child_rec = torch.where(same_p1, p1.W_rec.data,
                           torch.where(same_p2, p2.W_rec.data,
                               torch.where(torch.rand_like(p1.W_rec.data) > 0.5, p1.W_rec.data, p2.W_rec.data)))
            child.W_rec.data = child_rec
            child.M_rec = torch.where(same_p1, p1.M_rec,
                         torch.where(same_p2, p2.M_rec,
                             torch.where(torch.rand_like(p1.M_rec) > 0.5, p1.M_rec, p2.M_rec)))

            child.W_out.data = torch.where(col_mask.unsqueeze(0), p1.W_out.data, p2.W_out.data)
            child.M_out = torch.where(col_mask.unsqueeze(0), p1.M_out, p2.M_out)

            child.W_pred.data = torch.where(col_mask.unsqueeze(1), p1.W_pred.data, p2.W_pred.data)

        # 变异
        with torch.no_grad():
            if random.random() < 0.05:
                m_attr = random.choice(['M_in', 'M_rec', 'M_out'])
                m_tensor = getattr(child, m_attr)
                mut_mask = torch.rand_like(m_tensor) < mut_rate
                m_tensor[mut_mask] = 1.0 - m_tensor[mut_mask]

            for attr in ['W_in', 'W_rec', 'W_out', 'W_pred']:
                w_tensor = getattr(child, attr)
                noise = torch.randn_like(w_tensor) * 0.1
                noise_mask = torch.rand_like(w_tensor) < 0.2
                setattr(child, attr, nn.Parameter(w_tensor.data + noise * noise_mask))

        child.save_genetic_baseline()
        new_pop.append(child)

    return new_pop


# ==========================================
# 4. 主循环
# ==========================================
def main():
    N = 64
    OBS_DIM = 32
    ACTION_DIM = 3
    POP_SIZE = 1024
    ELITE_SIZE = 64
    GENERATIONS = 20
    INIT_DENSITY = 0.15
    STARVE_BIAS = 20
    MAX_STEPS = 1000

    print(f"test4_32dim: N={N}, OBS_DIM={OBS_DIM}, POP={POP_SIZE}, GEN={GENERATIONS}")
    print(f"  环境: RaySnakeEnv 32维, STARVE_BIAS={STARVE_BIAS}, MAX_STEPS={MAX_STEPS}")
    print(f"  架构: test4 原始 (M/W独立, 掩码翻转, 预测编码, 软拉马克)")

    random.seed(42)
    torch.manual_seed(42)

    env = RaySnakeEnv(grid_size=10, max_steps=MAX_STEPS)
    population = [EIBrainRegion(N, OBS_DIM, ACTION_DIM, INIT_DENSITY) for _ in range(POP_SIZE)]
    for ind in population:
        ind.save_genetic_baseline()

    history = {'gen': [], 'best_food': [], 'avg_food': [], 'avg_steps': [],
               'best_steps': [], 'active_rec': []}

    t0 = time.perf_counter()
    for gen in range(GENERATIONS):
        tg = time.perf_counter()

        fitnesses = []
        food_list = []
        steps_list = []
        for ind in population:
            f, s = evaluate_individual(ind, env, max_steps=MAX_STEPS)
            fitnesses.append(f)
            food_list.append(f)
            steps_list.append(s)

        best_idx = np.argmax(fitnesses)
        best_food = food_list[best_idx]
        avg_food = np.mean(food_list)
        avg_steps = np.mean(steps_list)
        best_steps = steps_list[best_idx]
        ar = int(population[best_idx].M_rec.sum().item())

        te = time.perf_counter() - tg

        history['gen'].append(gen)
        history['best_food'].append(best_food)
        history['avg_food'].append(avg_food)
        history['best_steps'].append(best_steps)
        history['avg_steps'].append(avg_steps)
        history['active_rec'].append(ar)

        print(f"Gen {gen+1}/{GENERATIONS} | BestFood: {best_food:.2f} | AvgFood: {avg_food:.2f} | "
              f"AvgSteps: {avg_steps:.0f} | ActRec: {ar} | {te:.1f}s")

        if gen < GENERATIONS - 1:
            population = evolve_topology(population, fitnesses, ELITE_SIZE)

    total = time.perf_counter() - t0
    print(f"\n总耗时: {total:.1f}s = {total/60:.1f}min")

    # 保存结果
    os.makedirs(os.path.join(_REPO, 'results'), exist_ok=True)
    torch.save(history, os.path.join(_REPO, 'results', 'test4_32dim_history.pth'))
    print(f"结果已保存: results/test4_32dim_history.pth")

    # 打印汇总
    print(f"\n=== test4_32dim 汇总 ===")
    print(f"BestFood: gen0={history['best_food'][0]:.2f} -> gen{GENERATIONS-1}={history['best_food'][-1]:.2f}")
    print(f"AvgFood:  gen0={history['avg_food'][0]:.2f} -> gen{GENERATIONS-1}={history['avg_food'][-1]:.2f}")
    print(f"ActRec:   gen0={history['active_rec'][0]} -> gen{GENERATIONS-1}={history['active_rec'][-1]}")


if __name__ == '__main__':
    main()
