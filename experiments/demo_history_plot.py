# 示例图：合成 30 代历史（真实量级），调用 test12.plot_history_png 确认
# 过程图右栏（适应度来源堆叠：food/eff/habit）的视觉效果。
# 用法: python experiments/demo_history_plot.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import test12 as t12

rng = np.random.default_rng(3)
G = 30
gens = list(range(1, G + 1))

# 合成：best_food 5→55 缓升带噪声；习惯三因素从低基线爬升到教师量级；
# eff 项随 food/steps 自然变化；总分 = food + 0.3eff + 0.6H
best_food = 5 + 50 * (1 - np.exp(-np.array(gens) / 9)) + rng.normal(0, 1.2, G)
best_food = np.clip(best_food, 1, None)
edge = np.clip(0.50 + 0.08 * np.log1p(np.array(gens)) / np.log(31) + rng.normal(0, 0.015, G), 0, 1)
conn = np.clip(0.55 + 0.10 * (1 - np.exp(-np.array(gens) / 8)) + rng.normal(0, 0.02, G), 0, 1)
straight = np.clip(0.30 + 0.45 * (1 - np.exp(-np.array(gens) / 7)) + rng.normal(0, 0.02, G), 0, 1)
steps = best_food * 4.5 + 12
p_food = best_food
p_eff = 0.3 * best_food / steps
best_fit = p_food * conn * (1 + 4.0 * p_eff / np.maximum(p_food, 1)
                         + 0.5 * straight + 0.5 * edge)

# v7 份额分解（与公式严格一致）：基础=food×conn，各因子项=基础×该因子，
# 四份之和=总分；noconn=conn=1 假想适应度，总分与 noconn 之差=单连通折扣
_t_eff = 0.3 * p_food / steps            # W_EFF·eff
_t_st = 0.5 * straight
_t_ed = 0.5 * edge
_f0 = (p_food * conn).tolist()
_pe = (_f0 * _t_eff).tolist()
_ps = (_f0 * _t_st).tolist()
_pd = (_f0 * _t_ed).tolist()
_noconn = (p_food * (1 + _t_eff + _t_st + _t_ed)).tolist()

history = {
    'gen': gens,
    'best_food': best_food.tolist(),
    'avg_food': (best_food * 0.28).tolist(),
    'elite_food': (best_food * 0.82).tolist(),
    'best_fit': best_fit,
    'best_turneff': (1.5 + 0.03 * np.array(gens)).tolist(),
    'best_epref': edge.tolist(),
    'best_conn': conn.tolist(),
    'best_straight': straight.tolist(),
    'fit_parts_food': _f0,
    'fit_parts_eff': _pe,
    'fit_parts_straight': _ps,
    'fit_parts_edge': _pd,
    'fit_parts_noconn': _noconn,
}

cfg = t12.Config()
cfg.CHECKPOINT_PATH = 'results/demo_v6_checkpoint.pth'   # → results/demo_v6_history.png
t12.plot_history_png(cfg, history)
print('示例图已保存: results/demo_v6_history.png（合成数据，仅确认视觉效果）')
