"""test13: test7b 最优模型 → 弱剪枝拓扑固化 → PPO 强化学习微调。

管线（einbrain PPO，test6 语义 + test7b 环境口径）：
    1. 加载 test7b 最后一代最优模型（einbrain io 兼容 test5d/7 格式，V/b_v 价值头零初始化）
    2. 20% 弱剪枝并拓扑固化：|W_rec| 最弱的 frac 比例连接 W 置零且 M_rec 置 0
       —— forward_ppo 梯度路径是 W*M，掩码置零后 PPO 无法复活被剪连接；
       trainable_parameters 本就不含掩码，双重固化。G2 动力学（tau_e/w_ei/w_ie）仍可学习
    3. ProjSnakeEnv（test7b _obs32 投影观测 + 3·len+20 饿死阈值的精确复刻，等价性
       已由 experiments/verify_proj_env.py 对拍验证）
    4. PPO：吃食 +1 / 死亡 -1 / 饥饿步惩罚，K=5 帧思考与 test7b 推理口径一致
    5. 周期评估走 test7b GPU 批量评估（GeneStack+_eval_chunk，与 1000 局基准同口径）

用法:
    python test13_ppo.py                # 正式长跑（默认 2000 iter）
    python test13_ppo.py --smoke        # 冒烟自检（30 iter）
    断点自动续训（test13_ppo_checkpoint.pth）；--fresh 忽略断点并重建种子
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_spec = importlib.util.spec_from_file_location('t7b', os.path.join(ROOT, 'experiments', 'test7_series', 'test7b.py'))
t7b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t7b)

from einbrain import io as eio
from einbrain.brain import EIBrainRegion
from einbrain.config import Config
from einbrain.env import ProjSnakeEnv
from einbrain import ppo as eppo


# ==================== 拓扑固化 ====================

def bake_weak_mask(brain, frac):
    """弱剪枝 + 拓扑固化：W_rec 最弱 frac 比例连接置零，对应 M_rec 置 0（永久移除）。

    返回剪掉的连接数。
    """
    if frac <= 0:
        return 0
    with torch.no_grad():
        W = brain.W_rec.data.float()
        M = brain.M_rec
        mag = W.abs() * M
        keep = M > 0
        n_active = int(keep.sum().item())
        n_pruned = 0
        for r in range(W.shape[0]):
            m_r = mag[r][keep[r]]
            if m_r.numel() == 0:
                continue
            kk = int(math.ceil(m_r.numel() * frac))
            if kk <= 0:
                continue
            thr = torch.kthvalue(m_r, kk).values
            zero = keep[r] & (mag[r] <= thr)
            W[r] = torch.where(zero, torch.zeros_like(W[r]), W[r])
            M[r][zero] = 0.0   # 只清本行的被剪位置（M[zero] 会按行选择语义清掉整行！）
            n_pruned += int(zero.sum().item())
        brain.W_rec.data = W.to(brain.W_rec.dtype)
        brain.refresh_cached()
    print(f"[固化] 弱剪枝 {frac:.0%}: 剪掉 {n_pruned}/{n_active} 条 W_rec 连接 "
          f"({n_pruned / max(n_active, 1) * 100:.1f}%)，M_rec 已同步置 0")
    return n_pruned


# ==================== test7b 口径的批量评估（EVAL_FN 钩子） ====================

def batched_eval(brain, cfg, B=128):
    """把当前脑快照复制进 test7b GeneStack，GPU 批量各跑 1 局。

    与 test7b_benchmark 1000 局基准同口径（同环境、同观测、同饿死阈值），
    返回 (mean_food, mean_steps)。
    """
    bcfg = t7b.Config()
    bcfg.EVAL_EPISODES = 1
    bcfg.SEED_FROM_BEST = False
    dev = t7b._resolve_device(bcfg)

    st = eio.save_brain_state(brain, use_half=False)
    pop = t7b.GeneStack(bcfg, B=B, device=dev)
    pop.random_init()
    for i in range(B):
        pop.set_individual_from_state(i, st)
    pop.refresh_eff()
    if getattr(bcfg, 'USE_FP16', True):
        pop.fp16()
        pop.refresh_eff()

    with torch.no_grad():
        metrics = t7b._eval_chunk(pop, bcfg)
    mn = metrics.cpu().numpy()
    return float(mn[:, 0].mean()), float(mn[:, 3].mean())


# ==================== 主入口 ====================

def build_cfg(args):
    cfg = Config()
    # --- 脑结构：与 test7b 模型逐位对齐 ---
    cfg.NUM_COLUMNS = 256
    cfg.OBS_DIM = 32
    cfg.ACTION_DIM = 3
    cfg.OBS_FOOD_SCALE = 8.0
    cfg.OBS_SELF_SCALE = 8.0
    cfg.OBS_OBSTACLE_SCALE = 8.0
    cfg.STARVE_SLOPE = 3.0
    cfg.ENV_CLS = ProjSnakeEnv
    cfg.SEES_FOOD_FN = None            # 投影观测食物恒可见，seen/unseen 塑形不启用
    cfg.EVAL_FN = batched_eval         # test7b 口径批量评估
    cfg.PPO_DEVICE = args.device

    # --- 环境口径：test7b 语义 ---
    cfg.GRID_SIZE = 10
    cfg.MAX_STEPS = 5000

    # --- 奖励：吃食 +1 / 死亡 -1 / 饥饿步惩罚；无空转惩罚（test7b 无此项）---
    cfg.SEEN_STEP_REWARD = 0.0
    cfg.UNSEEN_STEP_REWARD = 0.0
    cfg.EAT_REWARD = 1.0
    cfg.DEATH_REWARD = -1.0
    cfg.LOITER_PENALTY = 0.0
    cfg.HUNGER_WINDOW = 16
    cfg.HUNGER_STEP_PENALTY = 0.05

    # --- PPO 超参 ---
    cfg.FRAME_RATE = 5                 # 与 test7b deliberate K=5 一致
    cfg.INPUT_DECAY = 0.9
    cfg.N_ENVS = args.n_envs
    cfg.ROLLOUT_LEN = args.rollout
    cfg.TOTAL_ITERATIONS = args.iters
    cfg.LR = args.lr
    cfg.ENTROPY_COEF = args.entropy_coef
    cfg.MINIBATCH_SIZE = args.minibatch
    cfg.PPO_EPOCHS = args.ppo_epochs
    cfg.KL_ANCHOR_BETA = args.kl_beta
    cfg.ADV_NORMALIZE = not args.no_adv_norm
    cfg.DETACH_VALUE_TRUNK = True     # value 梯度不侵蚀共享主干
    cfg.SAMPLE_TEMP = args.sample_temp
    cfg.POLICY_WARMUP_ITERS = args.warmup
    cfg.EVAL_INTERVAL = args.eval_interval
    cfg.CHECKPOINT_INTERVAL = args.ckpt_interval
    cfg.EVAL_EPISODES = 3
    cfg.SEED_FROM_TEST5D = True
    cfg.AUTO_RESUME = not args.fresh

    # --- 输出 ---
    prefix = 'test13_smoke' if args.smoke else 'test13_ppo'
    cfg.CHECKPOINT_PATH = f'{prefix}_checkpoint.pth'
    cfg.BEST_MODEL_PATH = f'{prefix}_best_model.pth'
    cfg.TEST5D_MODEL_PATH = f'{prefix}_seed.pth'
    return cfg, prefix


def prepare_seed(cfg, args, prefix):
    """加载 test7b 模型 → 弱剪枝拓扑固化 → 存为 PPO 种子文件（einbrain 格式）。"""
    seed_path = cfg.TEST5D_MODEL_PATH
    if not args.fresh and os.path.exists(seed_path):
        print(f"[种子] 已存在 {seed_path}，跳过重建（--fresh 重建）")
        return

    t7b_cfg = t7b.Config()
    res = t7b.load_best_state(args.model, t7b_cfg)
    if res is None:
        print(f"错误: 无法加载 test7b 模型 {args.model}")
        sys.exit(1)
    st, saved_food, _ = res
    print(f"[种子] 加载 {args.model} (进化期 Food={saved_food:.1f})")

    brain = eio.load_brain_state(st, cfg)
    bake_weak_mask(brain, args.weak_mask_frac)

    # 固化后用 test7b 口径快速自检（B=64）
    check = batched_eval(brain, cfg, B=64)
    print(f"[种子] 固化后 64 局自检: mean_food={check[0]:.2f}")

    eio.save_best_model(seed_path, brain, cfg, check[0], check[1])
    print(f"[种子] 已保存 -> {seed_path}")


def save_history_json(history, prefix):
    path = f'{prefix}_history.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False)
    print(f"历史已写入 {path}")


def plot_history_png(history, prefix):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    it = history['iter']
    axes[0, 0].plot(it, history['mean_rew'])
    axes[0, 0].set_title('mean_reward')
    axes[0, 1].plot(it, history['mean_len'])
    axes[0, 1].set_title('mean_episode_len')
    axes[1, 0].plot(history['eval_iter'], history['eval_food'], 'o-')
    axes[1, 0].set_title('eval_food (test7b 128局口径)')
    axes[1, 1].plot(it, history['entropy'])
    axes[1, 1].set_title('entropy')
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = f'{prefix}_history.png'
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"曲线已写入 {path}")


def main():
    ap = argparse.ArgumentParser(description='test13: test7b 弱剪枝拓扑固化 + PPO 微调')
    ap.add_argument('--model', default='test7b_latest_gen_best.pth', help='test7b 模型路径')
    ap.add_argument('--weak-mask-frac', type=float, default=0.20, help='弱剪枝比例（默认 0.20）')
    ap.add_argument('--iters', type=int, default=2000, help='PPO 总迭代数')
    ap.add_argument('--lr', type=float, default=2e-5, help='学习率（微调用小步长）')
    ap.add_argument('--n-envs', type=int, default=32, help='并行环境数')
    ap.add_argument('--rollout', type=int, default=128, help='rollout 长度')
    ap.add_argument('--entropy-coef', type=float, default=0.005, help='熵系数')
    ap.add_argument('--minibatch', type=int, default=1024, help='PPO minibatch 大小')
    ap.add_argument('--ppo-epochs', type=int, default=2, help='每 iter PPO epoch 数')
    ap.add_argument('--kl-beta', type=float, default=0.2, help='种子策略 KL 锚定系数（0=关）')
    ap.add_argument('--no-adv-norm', action='store_true', help='关闭优势归一化（保留奖励自然尺度）')
    ap.add_argument('--sample-temp', type=float, default=0.5, help='rollout 采样温度（<1 更贴近贪心）')
    ap.add_argument('--warmup', type=int, default=50, help='策略预热期（仅训 critic 的 iter 数）')
    ap.add_argument('--eval-interval', type=int, default=20, help='评估间隔（迭代）')
    ap.add_argument('--ckpt-interval', type=int, default=20, help='断点间隔（迭代）')
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='PPO 训练设备')
    ap.add_argument('--seed', type=int, default=None, help='随机种子')
    ap.add_argument('--fresh', action='store_true', help='忽略断点并重建种子')
    ap.add_argument('--smoke', action='store_true', help='冒烟自检（30 iter 小配置）')
    args = ap.parse_args()

    if args.smoke:
        args.iters = 30
        args.n_envs = 8
        args.rollout = 64
        args.eval_interval = 10
        args.ckpt_interval = 10

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    cfg, prefix = build_cfg(args)
    prepare_seed(cfg, args, prefix)

    t0 = time.perf_counter()
    best_brain, history = eppo.run_training(cfg)

    save_history_json(history, prefix)
    plot_history_png(history, prefix)

    # 最终：最优模型的 test7b 口径批量评估（B=256）
    if best_brain is not None:
        food, steps = batched_eval(best_brain, cfg, B=256)
        print(f"\n最终评估（test7b 口径 256 局）: mean_food={food:.2f}, mean_steps={steps:.1f}")
    print(f"总耗时: {time.perf_counter() - t0:.1f}s")


if __name__ == '__main__':
    main()
