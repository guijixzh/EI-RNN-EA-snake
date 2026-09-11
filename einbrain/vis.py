"""可视化：训练曲线、贪吃蛇游玩、脑拓扑结构。

兼容两代基因组：
    - test5d/6/7 系（稠密 W_rec/M_rec，24 射线 / 32 投影观测）
    - test16 系列（稀疏 rec_idx/rec_w 固定扇入，40 维观测）——io.load_brain_state
      稠密化等价展开后走同一套拓扑/矩阵绘制；游玩走 test16b 语义桥接
      （play_model_16series，BatchedSnakeEnv + deliberate_batch 逐字复刻评估路径）。

命令行：
    python -m einbrain.vis <model.pth> [--play] [--topology] [--matrices]
                            [--save-prefix results/brain] [--bank-seed 7]
"""
from __future__ import annotations

import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from .env import SnakeEnv
from .deliberation import deliberate_action


def plot_history(history):
    """进化训练曲线（best_food / avg_food / best_seen / best_unseen）。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history['gen'], history['best_food'], label='Best Food', color='red', marker='o', markersize=3)
    ax1.plot(history['gen'], history['avg_food'], label='Avg Food', color='blue', alpha=0.6)
    ax1.set_title("Evolution Progress — Food Count (Primary Criterion)")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Food Eaten")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['gen'], history['best_seen'], label='Best Seen Steps', color='green', marker='s', markersize=3)
    ax2.plot(history['gen'], history['best_unseen'], label='Best Unseen Steps', color='purple', marker='^', markersize=3)
    ax2.set_title("Best Individual Seen/Unseen Steps\n(Seen lower=better, Unseen higher=better)")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Steps")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_ppo_history(history):
    """PPO 训练曲线（reward / eval_food / 各项 loss）。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(history['iter'], history['mean_rew'], label='Mean Reward', color='royalblue', marker='o', markersize=2)
    ax.set_title("PPO — Mean Episode Reward")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(history['eval_iter'], history['eval_food'], label='Eval Food', color='red', marker='s', markersize=3)
    ax.set_title("PPO — Eval Food (K=FRAME_RATE)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Food")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(history['iter'], history['policy_loss'], label='Policy', color='red', alpha=0.8)
    ax.plot(history['iter'], history['value_loss'], label='Value', color='blue', alpha=0.8)
    ax.plot(history['iter'], history['entropy'], label='Entropy', color='green', alpha=0.8)
    ax.set_title("PPO — Losses")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def render_snake_game(env):
    grid = np.zeros((env.grid_size, env.grid_size, 3))
    grid[env.food[0], env.food[1]] = [1, 1, 0]
    for seg in env.body:
        if 0 <= seg[0] < env.grid_size and 0 <= seg[1] < env.grid_size:
            grid[seg[0], seg[1]] = [0, 0, 1]
    if 0 <= env.head[0] < env.grid_size and 0 <= env.head[1] < env.grid_size:
        grid[env.head[0], env.head[1]] = [1, 0, 0]
    return grid


def visualize_best_brain_play(brain, cfg, max_steps=300):
    """可视化最优大脑玩贪吃蛇。"""
    env = SnakeEnv(grid_size=cfg.GRID_SIZE)
    brain.restore_genetic_baseline()
    brain.reset_runtime()

    obs = env.reset()
    E = torch.zeros(brain.N)
    I = torch.zeros(brain.N)

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    img = ax.imshow(render_snake_game(env), cmap='viridis', vmin=0, vmax=1)
    ax.set_title("Best Brain Playing Snake")
    ax.axis('off')

    steps = 0
    done = False

    while not done and steps < max_steps:
        action, avg_logits, E, I = deliberate_action(brain, obs, E, I)
        brain.update_fatigue(action)
        next_obs, ate_food, done, truncated = env.step(action)
        done = done or truncated
        obs = next_obs
        steps += 1

        img.set_data(render_snake_game(env))
        ax.set_title(f"Step: {steps} | Score: {len(env.body) - 2}")
        fig.canvas.draw_idle()
        plt.pause(0.1)

    print(f"\nGame Over! Final Score: {len(env.body) - 2} | Survived Steps: {steps}")
    plt.ioff()
    plt.show()


# ==================== 脑拓扑可视化（需 networkx）====================


def _norm_edge_weights(w):
    w = np.abs(w)
    wmin, wmax = w.min(), w.max()
    if wmax > wmin:
        return (w - wmin) / (wmax - wmin)
    return np.zeros_like(w)


def _community_order(brain, partition):
    tau = brain.tau_e_init.detach().numpy()
    comm_of_col = np.array([partition.get(f"Col_{i}", -1) for i in range(brain.N)], dtype=int)
    cols_by_comm = {}
    for i in range(brain.N):
        cols_by_comm.setdefault(int(comm_of_col[i]), []).append(i)
    comms = sorted(cols_by_comm.keys(), key=lambda c: -len(cols_by_comm[c]))
    order = []
    boundaries = []
    for c in comms:
        cols = sorted(cols_by_comm[c], key=lambda i: tau[i])
        boundaries.append(len(order))
        order.extend(cols)
    return np.array(order), comm_of_col, comms, np.array(boundaries)


def _brain_numpy(brain):
    """一次性别把脑张量转 numpy（大 N 下避免逐元素 .item() 循环）。"""
    return {
        'M_in': brain.M_in.detach().numpy(),
        'M_out': brain.M_out.detach().numpy(),
        'W_in': brain.W_in.detach().numpy(),
        'W_out': brain.W_out.detach().numpy(),
        'M_rec': brain.M_rec.detach().numpy(),
        'W_rec': brain.W_rec.detach().numpy(),
        'tau': brain.tau_e_init.detach().numpy(),
    }


def detect_communities(brain):
    """Louvain 社区发现（无 python-louvain 时回退贪心模块度）。
    边提取向量化（16 系列 N=1024 时 N² Python 双循环不可用）。"""
    import networkx as nx
    A = _brain_numpy(brain)
    G = nx.DiGraph()
    for i in range(brain.obs_dim):
        G.add_node(f"In_{i}", layer='input')
    for i in range(brain.N):
        G.add_node(f"Col_{i}", layer='column')
    for i in range(brain.action_dim):
        G.add_node(f"Out_{i}", layer='output')
    ri, ci = np.nonzero(A['M_in'])
    G.add_weighted_edges_from(
        ((f"In_{j}", f"Col_{i}", abs(A['W_in'][i, j])) for i, j in zip(ri, ci)))
    rt, ct = np.nonzero(A['M_rec'])
    G.add_weighted_edges_from(
        ((f"Col_{j}", f"Col_{i}", abs(A['W_rec'][i, j])) for i, j in zip(rt, ct)))
    ro, co = np.nonzero(A['M_out'])
    G.add_weighted_edges_from(
        ((f"Col_{j}", f"Out_{i}", abs(A['W_out'][i, j])) for i, j in zip(ro, co)))
    try:
        import community as community_louvain
        return community_louvain.best_partition(G.to_undirected())
    except ImportError:
        communities = nx.algorithms.community.greedy_modularity_communities(G.to_undirected())
        partition = {}
        for i, com in enumerate(communities):
            for node in com:
                partition[node] = i
        return partition


def plot_topology_layered(brain, cfg, partition):
    """方案 A：三层流水线拓扑图（输入层 → 柱层 → 输出层）。

    柱层按社区分块二维展开；柱节点 tau_e 上色；递归边红=兴奋/蓝=抑制，
    线宽∝|W|，只显示 top-K 强连接防毛团。
    """
    import random
    import math
    import networkx as nx
    import matplotlib.colors as mcolors
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch, Patch, Rectangle
    from matplotlib.lines import Line2D

    N = brain.N
    A = _brain_numpy(brain)
    tau = A['tau']
    order, comm_of_col, comms, boundaries = _community_order(brain, partition)

    W_rec_np = A['W_rec']
    M_rec_np = A['M_rec']

    n_comms = len(comms)
    ncols = max(1, int(np.ceil(np.sqrt(n_comms))))
    nrows = max(1, int(np.ceil(n_comms / ncols)))
    anchor_step_x = 0.90
    anchor_step_y = 0.90
    anchor_x0 = 1.30
    total_height = nrows * anchor_step_y
    anchor = {}
    for idx_blk, comm_id in enumerate(comms):
        r = idx_blk // ncols
        c = idx_blk % ncols
        anchor[int(comm_id)] = (anchor_x0 + c * anchor_step_x,
                                total_height - (r + 0.5) * anchor_step_y)

    block_radius = 0.34 * anchor_step_x
    pos_col = {}
    for idx_blk, comm_id in enumerate(comms):
        start = int(boundaries[idx_blk])
        end = int(boundaries[idx_blk + 1]) if idx_blk + 1 < len(boundaries) else N
        block = [int(col) for col in order[start:end]]
        if len(block) == 1:
            sub_pos = {block[0]: (0.0, 0.0)}
        else:
            sub = nx.DiGraph()
            for col in block:
                sub.add_node(col)
            for i in block:
                for j in block:
                    if M_rec_np[i, j] > 0:
                        sub.add_edge(j, i, weight=abs(W_rec_np[i, j]))
            seed = 42 + int(comm_id)
            try:
                sub_pos = nx.spring_layout(sub, k=0.6, iterations=150, seed=seed)
            except Exception:
                sub_pos = {col: (random.uniform(-1, 1), random.uniform(-1, 1))
                           for col in block}
        xs = [p[0] for p in sub_pos.values()]
        ys = [p[1] for p in sub_pos.values()]
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        span = max(float(np.max(xs) - np.min(xs)),
                   float(np.max(ys) - np.min(ys)), 1e-6)
        scale = (2.0 * block_radius) / span
        ax0, ay0 = anchor[int(comm_id)]
        for col in block:
            px, py = sub_pos[col]
            pos_col[col] = (ax0 + (px - cx) * scale, ay0 + (py - cy) * scale)

    MIN_NODE_DIST = 0.12
    for idx_blk, comm_id in enumerate(comms):
        start = int(boundaries[idx_blk])
        end = int(boundaries[idx_blk + 1]) if idx_blk + 1 < len(boundaries) else N
        block = [int(col) for col in order[start:end]]
        if len(block) < 2:
            continue
        for _ in range(40):
            moved = False
            for a_idx in range(len(block)):
                for b_idx in range(a_idx + 1, len(block)):
                    a, b = block[a_idx], block[b_idx]
                    xa, ya = pos_col[a]
                    xb, yb = pos_col[b]
                    dx = xa - xb
                    dy = ya - yb
                    dist = math.hypot(dx, dy)
                    if dist < MIN_NODE_DIST and dist > 1e-9:
                        push = (MIN_NODE_DIST - dist) / 2.0
                        ux, uy = dx / dist, dy / dist
                        pos_col[a] = (xa + ux * push, ya + uy * push)
                        pos_col[b] = (xb - ux * push, yb - uy * push)
                        moved = True
            if not moved:
                break

    OUT_X = anchor_x0 + ncols * anchor_step_x + 0.5
    y_in = {j: (j + 1.0) / (brain.obs_dim + 1.0) * total_height for j in range(brain.obs_dim)}
    y_out = {i: (i + 1.0) / (brain.action_dim + 1.0) * total_height for i in range(brain.action_dim)}

    community_colors = list(mcolors.TABLEAU_COLORS.values())
    tau_norm = plt.Normalize(cfg.TAU_E_MIN, cfg.TAU_E_MAX)
    in_degree = M_rec_np.sum(axis=0)

    # 大 N（16 系列 N=1024）自适应：节点缩小、top-K 边放宽，防过绘
    big_brain = N > 512
    s_main, s_inner, s_dot = (78, 48, 6.0) if not big_brain else (34, 20, 3.0)

    fig, ax = plt.subplots(figsize=(17, max(8, total_height + 2.0)))

    W_in_abs = _norm_edge_weights(A['W_in'])
    ri, ci = np.nonzero(A['M_in'])
    w_in_vals = W_in_abs[ri, ci]
    # 大 N 时输入边按 |W| 截 top-K（16 系列 ~6k 条全画会糊成绿墙）
    in_top = min(len(w_in_vals), 500 if big_brain else 1 << 30)
    in_keep = np.argsort(-w_in_vals)[:in_top] if in_top < len(w_in_vals) else np.arange(len(w_in_vals))
    in_alpha_base = 0.06 if big_brain else 0.15
    for kk in in_keep.tolist():
        i, j = int(ri[kk]), int(ci[kk])
        nw = W_in_abs[i, j]
        x1, y1 = pos_col[i]
        ax.plot([0.0, x1], [y_in[j], y1], color='green',
                lw=0.3 + 2.2 * nw, alpha=in_alpha_base + 0.5 * nw,
                solid_capstyle='round', zorder=1)

    W_out_abs = _norm_edge_weights(A['W_out'])
    ro, co = np.nonzero(A['M_out'])
    for i, j in zip(ro.tolist(), co.tolist()):
        nw = W_out_abs[i, j]
        x1, y1 = pos_col[j]
        ax.plot([x1, OUT_X], [y1, y_out[i]], color='magenta',
                lw=0.3 + 2.2 * nw, alpha=0.15 + 0.6 * nw, solid_capstyle='round', zorder=1)

    # rec 边提取向量化（N=1024 时 N² Python 双循环不可用）
    rt, ct = np.nonzero(M_rec_np)
    rec_w_list = W_rec_np[rt, ct]
    top_k = min(320 if big_brain else 180, len(rec_w_list))
    if top_k > 0:
        keep = np.argsort(-np.abs(rec_w_list))[:top_k]
        rec_abs = np.abs(rec_w_list[keep])
        w_max, w_min = rec_abs.max(), rec_abs.min()
        for kk in keep.tolist():
            i, j, w = int(rt[kk]), int(ct[kk]), float(rec_w_list[kk])
            nw = (abs(w) - w_min) / (w_max - w_min + 1e-8)
            x0p, y0p = pos_col[j]
            x1p, y1p = pos_col[i]
            dist = np.hypot(x1p - x0p, y1p - y0p)
            curve = 0.12 + 0.10 * dist
            mid_x = (x0p + x1p) / 2.0
            mid_y = (y0p + y1p) / 2.0
            ctrl = (mid_x + curve, mid_y + curve)
            verts = [(x0p, y0p), ctrl, (x1p, y1p)]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            ax.add_patch(PathPatch(Path(verts, codes), facecolor='none',
                                   edgecolor='red' if w >= 0 else 'blue',
                                   lw=0.3 + 2.0 * nw, alpha=0.10 + 0.55 * nw, zorder=2))

    for col in range(N):
        x, y = pos_col[col]
        comm = comm_of_col[col]
        ring_color = community_colors[comm % len(community_colors)] if comm >= 0 else 'gray'
        tau_color = plt.cm.plasma(tau_norm(tau[col]))
        ax.scatter(x, y, s=s_main, color=ring_color, edgecolors='black', linewidths=0.5, zorder=3)
        ax.scatter(x, y, s=s_inner, color=tau_color, zorder=4)
        in_n = int(in_degree[col])
        if in_n > 0:
            freq_norm = min(in_n, 20) / 20.0
            brightness = 0.55 + 0.45 * freq_norm
            ax.scatter(x, y, s=s_dot, color=(brightness, brightness, brightness), zorder=5)

    for j in range(brain.obs_dim):
        ax.scatter(0.0, y_in[j], marker='s', s=140, color='limegreen', edgecolors='black', zorder=3)
        ax.text(0.0, y_in[j], f"In_{j}", fontsize=7, ha='right', va='center')
    act_names = ['Fwd', 'Left', 'Right']
    for i in range(brain.action_dim):
        ax.scatter(OUT_X, y_out[i], marker='D', s=160, color='orange', edgecolors='black', zorder=3)
        ax.text(OUT_X, y_out[i], f"Out_{i}\n{act_names[i]}", fontsize=8, ha='left', va='center')

    for idx_blk, comm_id in enumerate(comms):
        ax0, ay0 = anchor[int(comm_id)]
        ax.text(ax0, ay0 + block_radius + 0.05, f"C{comm_id}", fontsize=9,
                ha='center', va='bottom',
                color=community_colors[int(comm_id) % len(community_colors)], fontweight='bold')

    ax.add_patch(Rectangle((anchor_x0 - 0.35, -0.35),
                           ncols * anchor_step_x + 0.7, total_height + 0.7,
                           facecolor='lightgray', alpha=0.18, zorder=0,
                           edgecolor='gray', linestyle='--', linewidth=0.8))
    ax.text((anchor_x0 + (ncols - 1) * anchor_step_x + anchor_x0) / 2.0,
            total_height + 0.42, 'Cortical Columns (E-I units)',
            ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.text(0.0, total_height + 0.42, 'Input', ha='center', va='bottom',
            fontsize=12, fontweight='bold')
    ax.text(OUT_X, total_height + 0.42, 'Output', ha='center', va='bottom',
            fontsize=12, fontweight='bold')

    handles = [
        Patch(facecolor='limegreen', edgecolor='black', label='Input node'),
        Patch(facecolor='orange', edgecolor='black', label='Output node (3 actions)'),
        Patch(facecolor='white', edgecolor='black', label='Column node (ring=community, fill=tau_e, dot=in-degree)'),
        Line2D([0], [0], color='green', lw=2, label='Input→Column edge'),
        Line2D([0], [0], color='magenta', lw=2, label='Column→Output edge'),
        Line2D([0], [0], color='red', lw=2, label='Recurrent E (W>0)'),
        Line2D([0], [0], color='blue', lw=2, label='Recurrent I (W<0)'),
    ]
    ax.legend(handles=handles, loc='lower left', fontsize=8, framealpha=0.9)

    sm = plt.cm.ScalarMappable(cmap='plasma', norm=tau_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label('tau_e_init', fontsize=10)

    ax.set_xlim(-1.1, OUT_X + 0.9)
    ax.set_ylim(-0.8, total_height + 0.8)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("Evolved Brain Topology — Three-Layer Pipeline\n"
                 "Columns spread in 2D by community (islands) | E=red / I=blue | "
                 "thickness/alpha ∝ |W| | top-K recurrent edges", fontsize=13)
    plt.tight_layout()


def plot_connectivity_matrices(brain, cfg, partition):
    """方案 B：连接矩阵仪表盘（M_in / M_rec / M_out 热力图）。

    16 系列（sparse_rec）经 io 稠密化后同样适用；rec 矩阵即
    scatter 展开视图（重复源权重已叠加）。"""
    N = brain.N
    A = _brain_numpy(brain)
    order, comm_of_col, comms, boundaries = _community_order(brain, partition)

    W_in_map = A['W_in'] * A['M_in']
    W_rec_map = A['W_rec'] * A['M_rec']
    W_out_map = A['W_out'] * A['M_out']

    M_in_sorted = W_in_map[order, :]
    M_rec_sorted = W_rec_map[np.ix_(order, order)]
    M_out_sorted = W_out_map[:, order]

    rec_vmax = float(np.abs(M_rec_sorted).max()) + 1e-8

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))

    im0 = axes[0].imshow(M_in_sorted, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[0].set_title("Input W_in (Column × Feature)", fontsize=12)
    axes[0].set_xlabel("Input feature")
    axes[0].set_ylabel("Column (community/τ sorted)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(M_rec_sorted, aspect='auto', cmap='RdBu', interpolation='nearest',
                         vmin=-rec_vmax, vmax=rec_vmax)
    axes[1].set_title("Recurrent W_rec (Target × Source)\nE=red / I=blue", fontsize=12)
    axes[1].set_xlabel("Source column")
    axes[1].set_ylabel("Target column")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    for b in boundaries[1:]:
        line_pos = int(b) - 0.5
        axes[1].axhline(y=line_pos, color='white', lw=1.0, alpha=0.9)
        axes[1].axvline(x=line_pos, color='white', lw=1.0, alpha=0.9)

    im2 = axes[2].imshow(M_out_sorted, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[2].set_title("Output W_out (Action × Column)", fontsize=12)
    axes[2].set_xlabel("Column (community/τ sorted)")
    axes[2].set_ylabel("Action")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    fig.suptitle("Connectivity Matrices — sorted by functional community & tau_e", fontsize=14)


def visualize_brain_ecosystem(brain, cfg, save_prefix=None):
    """综合展示：三层拓扑图 + 连接矩阵。

    save_prefix 给定时存 PNG 而非弹窗（多模型批量出图用）。
    16 系列（brain.sparse_rec=True，经 io 稠密化）自动标注来源与扇入。"""
    print("\n=== Generating Brain Ecosystem Visualization ===")
    tag = ''
    if getattr(brain, 'sparse_rec', False):
        tag = (f" [sparse1 genome, fan-in K={brain.rec_fanin}, "
               f"{brain.rec_unique_edges}/{brain.rec_slots} unique slots densified]")
        print(f"  16 系列稀疏基因组{tag}")
    partition = detect_communities(brain)
    plot_topology_layered(brain, cfg, partition)
    if save_prefix:
        plt.savefig(save_prefix + '_topology.png', dpi=150, bbox_inches='tight')
        print(f"  已保存 {save_prefix}_topology.png")
    plot_connectivity_matrices(brain, cfg, partition)
    if save_prefix:
        plt.savefig(save_prefix + '_matrices.png', dpi=150, bbox_inches='tight')
        print(f"  已保存 {save_prefix}_matrices.png")
    if not save_prefix:
        plt.show()


# ==================== 模型文件直读可视化（test5d/6/7 稠密 + test16 系列稀疏）====================


def _load_test16_module():
    """定位仓库根的 test16b.py（16 系列评估语义的权威实现：obs40 + BatchedSnakeEnv）。"""
    import importlib.util
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(root, 'experiments', 'test16_series', 'test16b.py')
    if not os.path.exists(p):
        for alt in (os.path.join('experiments', 'test16_series', 'test16a.py'),
                        os.path.join('experiments', 'test16_series', 'test16.py')):
            p2 = os.path.join(root, alt)
            if os.path.exists(p2):
                p = p2
                break
    spec = importlib.util.spec_from_file_location('_t16_bridge', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _render_batched_env(env):
    """BatchedSnakeEnv（B=1）当前局面 → RGB 网格。"""
    G = env.G
    grid = np.zeros((G, G, 3))
    f = env.food[0].tolist()
    grid[f[0], f[1]] = [1, 1, 0]
    b = env.body[0, :int(env.body_len[0].item())].tolist()
    for i, (r, c) in enumerate(b):
        if 0 <= r < G and 0 <= c < G:
            grid[r, c] = [0.9, 0.1, 0.1] if i == 0 else [0, 0, 1]
    return grid


def play_model_16series(model_path, max_steps=400, speed=0.02, bank_seed=None):
    """16 系列（sparse1 / obs40）模型游玩可视化：借 test16b 的 BatchedSnakeEnv
    （B=1）+ GeneStack + deliberate_batch 逐字复刻评估语义（obs40/fast 路径、
    fp16、FRAME_RATE 思考），matplotlib 实时渲染。

    bank_seed 给定时用 CRN bank 复现确定一局（gen=bank_seed, stage=9, ep=0）。
    返回 (food, steps)。"""
    t16 = _load_test16_module()
    cfg = t16.Config()
    res = t16.load_best_state(model_path, cfg)
    if res is None:
        raise RuntimeError(f"无法加载 16 系列模型 {model_path}（版本守卫未过）")
    st, saved_food, _ = res
    print(f"16 系列模型: {os.path.basename(model_path)}（训练期 Food={saved_food:.1f}）")
    dev = t16._resolve_device(cfg)
    pop = t16.GeneStack(cfg, B=1, device=dev)
    pop.random_init()
    pop.set_individual_from_state(0, st)
    if getattr(cfg, 'USE_FP16', True):
        pop.fp16()
    pop.refresh_eff()

    bank = None
    if bank_seed is not None:
        bank = t16.make_bank(cfg, int(bank_seed), 9, 0, dev)
    env = t16.BatchedSnakeEnv(cfg, 1, dev)
    env.reset(bank=bank)
    E = torch.zeros(1, pop.N, dtype=pop.dtype, device=dev)
    I = torch.zeros(1, pop.N, dtype=pop.dtype, device=dev)
    stt = torch.zeros(1, pop.N, dtype=pop.dtype, device=dev)
    press = torch.zeros(1, dtype=torch.float32, device=dev)

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6.4))
    img = ax.imshow(_render_batched_env(env), vmin=0, vmax=1)
    ax.set_title("16-series Brain Playing Snake")
    ax.axis('off')
    ep_food, s = 0, 0
    for s in range(max_steps):
        obs = env.obs().to(pop.dtype)
        act, E, I, stt = t16.deliberate_batch(pop, obs, E, I, stt, press, cfg)
        press = t16.update_fatigue(press, act, decay=float(cfg.FATIGUE_TURN_DECAY))
        env.step(act)
        if bool(env.ate[0]):
            ep_food += 1
        img.set_data(_render_batched_env(env))
        died = int(env.died[0].item())
        ttl = (f"Step {s + 1} | Score {ep_food} | " +
               ({1: 'died: wall', 2: 'died: self', 3: 'died: starve'}.get(died) or 'alive'))
        ax.set_title(ttl)
        fig.canvas.draw_idle()
        # 注意 speed=0 时不可调 plt.pause(0)——事件循环 timeout=0 语义为无限等待
        if speed > 0:
            plt.pause(speed)
        if bool(env.all_done()):
            break
    plt.ioff()
    plt.show()
    print(f"Play done: Food={ep_food}, Steps={s + 1}")
    return ep_food, s + 1


def visualize_model_play(model_path, max_steps=400, speed=0.02, bank_seed=None):
    """模型文件直读游玩可视化：自动识别血统。

    - 16 系列（稀疏 rec_idx/obs40）→ play_model_16series（test16b 语义桥接）
    - 稠密旧血统（24 射线 / 32 投影）→ einbrain 标量环境 + EIBrainRegion 前向"""
    data = torch.load(model_path, map_location='cpu', weights_only=False)
    brain_state = data.get('brain', {})
    if 'rec_idx' in brain_state and 'rec_w' in brain_state:
        return play_model_16series(model_path, max_steps=max_steps, speed=speed,
                                   bank_seed=bank_seed)

    from . import io as _io
    from .deliberation import deliberate_action
    loaded = _io.load_model_any(model_path)
    if loaded is None:
        raise RuntimeError(f"无法加载模型 {model_path}")
    brain, cfg, meta = loaded
    print(f"稠密血统模型: {os.path.basename(model_path)} "
          f"(N={brain.N}, obs_dim={brain.obs_dim}, 训练期 Food={meta['food']:.1f})")
    if brain.obs_dim == 32:
        from .env import ProjSnakeEnv
        env = ProjSnakeEnv(grid_size=cfg.GRID_SIZE, max_steps=max_steps, cfg=cfg)
    else:
        from .env import SnakeEnv
        env = SnakeEnv(grid_size=cfg.GRID_SIZE, max_steps=max_steps, cfg=cfg)

    brain.restore_genetic_baseline()
    brain.reset_runtime()
    obs = env.reset()
    E = torch.zeros(brain.N)
    I = torch.zeros(brain.N)

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6.4))
    img = ax.imshow(render_snake_game(env), cmap='viridis', vmin=0, vmax=1)
    ax.set_title("Best Brain Playing Snake")
    ax.axis('off')
    steps, done = 0, False
    while not done and steps < max_steps:
        action, avg_logits, E, I = deliberate_action(brain, obs, E, I)
        brain.update_fatigue(action)
        next_obs, ate_food, done, truncated = env.step(action)
        done = done or truncated
        obs = next_obs
        steps += 1
        img.set_data(render_snake_game(env))
        ax.set_title(f"Step: {steps} | Score: {len(env.body) - 2}")
        fig.canvas.draw_idle()
        if speed > 0:            # speed=0 时 pause(0) 会进入无限事件循环
            plt.pause(speed)
    print(f"\nGame Over! Final Score: {len(env.body) - 2} | Survived Steps: {steps}")
    plt.ioff()
    plt.show()
    return len(env.body) - 2, steps


if __name__ == '__main__':
    import argparse
    from . import io as _io

    ap = argparse.ArgumentParser(
        description='einbrain 模型可视化（兼容 test5d/6/7 稠密与 test16 系列稀疏基因组）')
    ap.add_argument('model', help='模型文件（*_best_model.pth 等）')
    ap.add_argument('--play', action='store_true', help='游玩可视化（实时渲染）')
    ap.add_argument('--topology', action='store_true', help='三层拓扑图')
    ap.add_argument('--matrices', action='store_true', help='连接矩阵仪表盘')
    ap.add_argument('--save-prefix', default=None,
                    help='给定时存 PNG 不弹窗（如 results/brain16）')
    ap.add_argument('--max-steps', type=int, default=400)
    ap.add_argument('--speed', type=float, default=0.02, help='游玩渲染间隔秒')
    ap.add_argument('--bank-seed', type=int, default=None,
                    help='16 系列 CRN 局种子（复现确定一局）')
    args = ap.parse_args()

    if args.play:
        visualize_model_play(args.model, max_steps=args.max_steps,
                             speed=args.speed, bank_seed=args.bank_seed)
    if args.topology or args.matrices or not args.play:
        loaded = _io.load_model_any(args.model)
        if loaded is None:
            raise SystemExit(1)
        brain, cfg, meta = loaded
        print(f"模型: {os.path.basename(args.model)} | N={brain.N} obs={brain.obs_dim} "
              f"act={brain.action_dim} | 训练期 Food={meta['food']:.1f} "
              f"| sparse_rec={meta['sparse_rec']}"
              + (f" (K={meta['rec_fanin']})" if meta['sparse_rec'] else ""))
        partition = detect_communities(brain)
        if args.topology or not (args.play or args.matrices):
            plot_topology_layered(brain, cfg, partition)
            if args.save_prefix:
                plt.savefig(args.save_prefix + '_topology.png', dpi=150, bbox_inches='tight')
                print(f"已保存 {args.save_prefix}_topology.png")
        if args.matrices or not (args.play or args.topology):
            plot_connectivity_matrices(brain, cfg, partition)
            if args.save_prefix:
                plt.savefig(args.save_prefix + '_matrices.png', dpi=150, bbox_inches='tight')
                print(f"已保存 {args.save_prefix}_matrices.png")
        if not args.save_prefix:
            plt.show()
