"""可视化：训练曲线、贪吃蛇游玩、脑拓扑结构。"""
from __future__ import annotations

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


def detect_communities(brain):
    """Louvain 社区发现（无 python-louvain 时回退贪心模块度）。"""
    import networkx as nx
    G = nx.DiGraph()
    for i in range(brain.obs_dim):
        G.add_node(f"In_{i}", layer='input')
    for i in range(brain.N):
        G.add_node(f"Col_{i}", layer='column')
    for i in range(brain.action_dim):
        G.add_node(f"Out_{i}", layer='output')
    for i in range(brain.N):
        for j in range(brain.obs_dim):
            if brain.M_in[i, j] > 0:
                G.add_edge(f"In_{j}", f"Col_{i}", weight=abs(brain.W_in[i, j].item()))
    for i in range(brain.N):
        for j in range(brain.N):
            if brain.M_rec[i, j] > 0:
                G.add_edge(f"Col_{j}", f"Col_{i}", weight=abs(brain.W_rec[i, j].item()))
    for i in range(brain.action_dim):
        for j in range(brain.N):
            if brain.M_out[i, j] > 0:
                G.add_edge(f"Col_{j}", f"Out_{i}", weight=abs(brain.W_out[i, j].item()))
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
    tau = brain.tau_e_init.detach().numpy()
    order, comm_of_col, comms, boundaries = _community_order(brain, partition)

    W_rec_np = brain.W_rec.detach().numpy()
    M_rec_np = brain.M_rec.numpy()

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
    in_degree = brain.M_rec.sum(dim=0).numpy()

    fig, ax = plt.subplots(figsize=(17, max(8, total_height + 2.0)))

    W_in_abs = _norm_edge_weights(brain.W_in.detach().numpy())
    for i in range(N):
        for j in range(brain.obs_dim):
            if brain.M_in[i, j] > 0:
                nw = W_in_abs[i, j]
                x1, y1 = pos_col[i]
                ax.plot([0.0, x1], [y_in[j], y1], color='green',
                        lw=0.3 + 2.2 * nw, alpha=0.15 + 0.6 * nw, solid_capstyle='round', zorder=1)

    W_out_abs = _norm_edge_weights(brain.W_out.detach().numpy())
    for i in range(brain.action_dim):
        for j in range(N):
            if brain.M_out[i, j] > 0:
                nw = W_out_abs[i, j]
                x1, y1 = pos_col[j]
                ax.plot([x1, OUT_X], [y1, y_out[i]], color='magenta',
                        lw=0.3 + 2.2 * nw, alpha=0.15 + 0.6 * nw, solid_capstyle='round', zorder=1)

    rec_edges = [(i, j, W_rec_np[i, j]) for i in range(N) for j in range(N) if M_rec_np[i, j] > 0]
    rec_edges.sort(key=lambda e: -abs(e[2]))
    top_k = min(180, len(rec_edges))
    if top_k > 0:
        rec_abs = np.abs([e[2] for e in rec_edges[:top_k]])
        w_max, w_min = rec_abs.max(), rec_abs.min()
        for i, j, w in rec_edges[:top_k]:
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
        ax.scatter(x, y, s=78, color=ring_color, edgecolors='black', linewidths=0.8, zorder=3)
        ax.scatter(x, y, s=48, color=tau_color, zorder=4)
        in_n = int(in_degree[col])
        if in_n > 0:
            freq_norm = min(in_n, 20) / 20.0
            brightness = 0.55 + 0.45 * freq_norm
            ax.scatter(x, y, s=6.0, color=(brightness, brightness, brightness), zorder=5)

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
    plt.show()


def plot_connectivity_matrices(brain, cfg, partition):
    """方案 B：连接矩阵仪表盘（M_in / M_rec / M_out 热力图）。"""
    N = brain.N
    order, comm_of_col, comms, boundaries = _community_order(brain, partition)

    W_in_map = (brain.W_in * brain.M_in).detach().numpy()
    W_rec_map = (brain.W_rec * brain.M_rec).detach().numpy()
    W_out_map = (brain.W_out * brain.M_out).detach().numpy()

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
    plt.show()


def visualize_brain_ecosystem(brain, cfg):
    """综合展示：三层拓扑图 + 连接矩阵。"""
    print("\n=== Generating Brain Ecosystem Visualization ===")
    partition = detect_communities(brain)
    plot_topology_layered(brain, cfg, partition)
    plot_connectivity_matrices(brain, cfg, partition)
