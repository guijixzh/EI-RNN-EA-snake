# 诊断 3：test12 "卡 1.0" 三臂对照（同种子/同规模/同食物流 CRN）
#   A = test7h 原版（基线）
#   B = test12 关孤岛惩罚（--island-penalty 1.0 等价）
#   C = test12 关惩罚 + _obs32 换回 test7h 食物编码（隔离观测编码变量）
# 用法: python experiments/diag_test12_vs_7h.py [--gens 12]
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'experiments', 'test7_series'))  # 归档后模块路径
import numpy as np
import torch

import test7h as th
import test12 as t12


def make_cfg(mod, frame='abs'):
    cfg = mod.Config()
    cfg.POP_SIZE, cfg.NUM_COLUMNS = 256, 96
    cfg.ELITE_SIZE, cfg.STAGE1_EPS, cfg.EVAL_EPISODES, cfg.STAGE2_KEEP = 64, 2, 4, 64
    cfg.MAX_STEPS, cfg.EVAL_BATCH = 3000, 256
    if hasattr(cfg, 'OBS_FOOD_FRAME'):
        cfg.OBS_FOOD_FRAME = frame
    return cfg


def run_arm(label, mod, cfg, gens, patch_obs7h=False):
    if patch_obs7h:
        t12.BatchedSnakeEnv._obs32 = th.BatchedSnakeEnv._obs32
    torch.manual_seed(0)
    pop = mod.GeneStack(cfg, device=mod._resolve_device(cfg))
    pop.random_init()
    print(f"\n=== 臂 {label} ===")
    print(f"gen | best_food | mean_food | food>0 | top64_food")
    for gen in range(gens):
        metrics, order = evaluate(mod, pop, cfg, gen)
        mn = metrics.numpy()
        food = mn[:, 0]
        top64 = np.asarray(order[:64])
        print(f"{gen:3d} | {food.max():9.2f} | {food.mean():9.3f} | "
              f"{(food > 0).mean():6.2%} | {food[top64].mean():.2f}")
        if gen < gens - 1:
            pop = mod.evolve_topology_gpu(pop, metrics, cfg, gen=gen, order=order)
    if patch_obs7h:  # 还原
        import importlib
        importlib.reload(t12)


def evaluate(mod, pop, cfg, gen):
    return mod.evaluate_population_gpu(pop, cfg, gen=gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gens', type=int, default=12)
    ap.add_argument('--arms', type=str, default='ABC')
    args = ap.parse_args()

    cfgA = make_cfg(th)
    cfgB = make_cfg(t12)
    cfgC = make_cfg(t12)
    cfgD = make_cfg(t12, frame='ego')
    for cfg in (cfgA, cfgB, cfgC, cfgD):
        cfg.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if 'A' in args.arms:
        run_arm('A test7h 原版', th, cfgA, args.gens)
    if 'B' in args.arms:
        run_arm('B test12 v2 默认（abs+分级惩罚开）', t12, cfgB, args.gens)
    if 'C' in args.arms:
        run_arm('C test12 惩罚关+7h食物编码', t12, cfgC, args.gens, patch_obs7h=True)
    if 'D' in args.arms:
        run_arm('D test12 v2 ego帧（前/右/后/左+距离）', t12, cfgD, args.gens)


if __name__ == '__main__':
    main()
