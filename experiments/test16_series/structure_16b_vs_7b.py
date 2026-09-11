"""16b（稀疏固定扇入基因组）vs 7b（稠密掩码基因组）脑结构指标横向对比。

对象（同代最优个体、各自 1000 局实测过的模型）：
    - test16b_simp_best_model.pth        N=1024, rec_idx/rec_w K=16, obs=40
    - artifacts/test16b/test16b_simp_latest_gen_best.pth   同上（末代）
    - artifacts/test7b/test7b_latest_gen_best.pth         N=256,  稠密 W_rec/M_rec,     obs=32（1000 局 mean 61.35）

指标分层：
    A 规模与密度：N、rec 边数/密度、有效（|W|>0.01）边占比、输入/输出连接度
    B 度分布：rec 入度（fan-in）/出度（去重后唯一源）、M_in 行度、M_out 列度
    C 权重：|W| 分布（分位/基尼系数/top1% 份额）、E/I 符号占比
    D 单元动力学：tau_e / w_ei / w_ie 分布
    E 图结构：Louvain(greedy) 模块度 Q、社区数/最大社区占比、WCC/SCC、
      平衡互惠率、度同配性、平均最短路（最大 WCC）、小世界 sigma、hub 集中度
    F 稀疏专属（16b）：零权槽位占比、重复源合并数、扇入亏空

输出：results/structure_16b_vs_7b.json + results/structure_16b_vs_7b.png
用法：python experiments/structure_16b_vs_7b.py
"""
import os, sys, json
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

from einbrain import io as eio

MODELS = {
    '16b_best': os.path.join(ROOT, 'test16b_simp_best_model.pth'),
    '16b_latest': os.path.join(ROOT, 'artifacts/test16b/test16b_simp_latest_gen_best.pth'),
    '7b_latest': os.path.join(ROOT, 'artifacts/test7b/test7b_latest_gen_best.pth'),
}

EFF_W_THR = 0.01     # |W|>该阈值视为有效边（两模型统一口径）


def gini(x):
    """基尼系数（权重集中度；0=完全均匀，1=全部集中一条边）。"""
    x = np.sort(np.abs(np.asarray(x, dtype=np.float64)))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def small_world_sigma(G):
    """小世界系数 sigma = (C/C_rand)/(L/L_rand)；sigma>1 具小世界性。
    C=聚类（无向化），L=平均最短路（最大连通分量）。随机图同 N 同边数 ×3 取均值。"""
    Gu = G.to_undirected()
    if Gu.number_of_nodes() < 8:
        return float('nan')
    C = nx.transitivity(Gu)
    lcc = max(nx.connected_components(Gu), key=len)
    L = nx.average_shortest_path_length(Gu.subgraph(lcc))
    n, m = Gu.number_of_nodes(), Gu.number_of_edges()
    Cs, Ls = [], []
    for seed in range(3):
        rng = np.random.default_rng(1000 + seed)
        Gr = nx.gnm_random_graph(n, m, seed=int(rng.integers(1 << 30)))
        lcc_r = max(nx.connected_components(Gr), key=len)
        Cs.append(nx.transitivity(Gr))
        Ls.append(nx.average_shortest_path_length(Gr.subgraph(lcc_r)))
    Cr, Lr = float(np.mean(Cs)), float(np.mean(Ls))
    if Cr <= 0 or Lr <= 0 or L <= 0:
        return float('nan')
    return float((C / Cr) / (L / Lr))


def graph_metrics(M_rec, W_rec):
    """循环连接图结构指标（有向加权=|W|）。"""
    N = M_rec.shape[0]
    rt, ct = np.nonzero(M_rec)
    w = np.abs(W_rec[rt, ct])
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    G.add_weighted_edges_from(zip(rt.tolist(), ct.tolist(), w.tolist()))

    # 社区：greedy_modularity（两模型同算法同权重口径，可横比）
    part_sets = nx.algorithms.community.greedy_modularity_communities(
        G.to_undirected(), weight='weight')
    sizes = np.array([len(c) for c in part_sets])
    Q = float(nx.algorithms.community.modularity(
        G.to_undirected(), part_sets, weight='weight'))

    Gu = G.to_undirected()
    lcc = max(nx.connected_components(Gu), key=len)
    sub = Gu.subgraph(lcc)
    L = nx.average_shortest_path_length(sub) if len(lcc) > 1 else float('nan')
    C = nx.transitivity(Gu)
    n_scc = nx.number_strongly_connected_components(G)
    recip = float(nx.overall_reciprocity(G)) if G.number_of_edges() else float('nan')
    try:
        assort = float(nx.degree_assortativity_coefficient(Gu))
    except Exception:
        assort = float('nan')
    sigma = small_world_sigma(G)
    indeg = np.array([d for _, d in G.in_degree()], dtype=float)
    return {
        'modularity_Q': Q,
        'n_communities': int(len(sizes)),
        'largest_comm_frac': float(sizes.max() / N),
        'comm_sizes_top5': sorted(int(s) for s in sizes)[-5:],
        'n_wcc': int(nx.number_connected_components(Gu)),
        'lcc_frac': float(len(lcc) / N),
        'n_scc': int(n_scc),
        'reciprocity': recip,
        'assortativity_deg': assort,
        'avg_path_lcc': float(L),
        'clustering_transitivity': float(C),
        'small_world_sigma': sigma,
        'max_in_degree': float(indeg.max()),
        'mean_in_degree': float(indeg.mean()),
    }


def analyze(tag, path):
    data = torch.load(path, map_location='cpu', weights_only=False)
    st = data['brain']
    cfgd = data.get('config', {})
    N = int(st['N'])
    sparse = 'rec_idx' in st and 'rec_w' in st and 'W_rec' not in st

    M_in = st['M_in'].float().numpy()
    M_out = st['M_out'].float().numpy()
    W_in = st['W_in'].float().numpy()
    W_out = st['W_out'].float().numpy()
    tau = st['tau_e_init'].float().numpy()
    w_ei = st['w_ei'].float().numpy()
    w_ie = st['w_ie'].float().numpy()

    extra = {}
    if sparse:
        rec_idx = st['rec_idx'].long()
        rec_w = st['rec_w'].float()
        K = rec_idx.shape[1]
        W_rec = torch.zeros(N, N).scatter_add_(1, rec_idx, rec_w).numpy()
        M_rec = torch.zeros(N, N).scatter_(1, rec_idx, 1.0).numpy()
        # 稀疏专属：唯一源 fan-in、零权槽位
        uniq_per_row = np.array([len(set(rec_idx[i].tolist())) for i in range(N)])
        zero_slot_frac = float((rec_w.abs() < 1e-6).float().mean())
        extra = {
            'rec_fanin_K': int(K),
            'unique_source_fanin_mean': float(uniq_per_row.mean()),
            'unique_source_fanin_min': int(uniq_per_row.min()),
            'unique_source_fanin_max': int(uniq_per_row.max()),
            'zero_weight_slot_frac': zero_slot_frac,
            'slots': int(rec_idx.numel()),
        }
    else:
        W_rec = st['W_rec'].float().numpy()
        M_rec = st['M_rec'].float().numpy()

    Weff = W_rec * M_rec
    eff_mask = M_rec > 0
    eff_strong = eff_mask & (np.abs(Weff) > EFF_W_THR)

    # 度分布（唯一源：M_rec 本身已去重；行和=出度（target 行），列和=入度）
    out_deg = M_rec.sum(axis=1)
    in_deg = M_rec.sum(axis=0)
    in_deg_strong = eff_strong.sum(axis=0)
    out_deg_strong = eff_strong.sum(axis=1)

    rec_w_vals = Weff[eff_mask]
    m_in_deg = M_in.sum(axis=1)
    m_out_deg_per_action = M_out.sum(axis=1)    # 每动作的读出柱数 [A]
    m_out_col_frac = float((M_out.sum(axis=0) > 0).mean())   # 被 ≥1 动作读出的柱占比
    top1p_share = float(np.sort(np.abs(rec_w_vals))[-max(1, len(rec_w_vals) // 100):].sum()
                        / np.abs(rec_w_vals).sum())

    res = {
        'tag': tag,
        'file': os.path.basename(path),
        'saved_food': float(data.get('food', -1)),
        'sparse_genome': sparse,
        'N': N,
        'obs_dim': int(M_in.shape[1]),
        'action_dim': int(M_out.shape[0]),
        'init_density_cfg': cfgd.get('INIT_DENSITY'),
        'n_rec_edges': int(eff_mask.sum()),
        'rec_density': float(eff_mask.sum() / (N * (N - 1))),
        'n_rec_edges_strong': int(eff_strong.sum()),
        'strong_frac': float(eff_strong.sum() / max(eff_mask.sum(), 1)),
        'n_in_edges': int(M_in.sum()),
        'in_fanin_mean': float(m_in_deg.mean()),
        'in_fanin_zero_frac': float((m_in_deg == 0).mean()),
        'out_readout_per_action': m_out_deg_per_action.astype(int).tolist(),
        'out_readout_col_frac': m_out_col_frac,
        # 度分布
        'rec_out_deg_mean': float(out_deg.mean()),
        'rec_out_deg_std': float(out_deg.std()),
        'rec_in_deg_mean': float(in_deg.mean()),
        'rec_in_deg_std': float(in_deg.std()),
        'rec_in_deg_cv': float(in_deg.std() / max(in_deg.mean(), 1e-9)),
        'rec_in_deg_max': float(in_deg.max()),
        'rec_in_deg_top10pct_share': float(
            np.sort(in_deg)[-max(1, N // 10):].sum() / max(in_deg.sum(), 1)),
        'rec_in_deg_strong_mean': float(in_deg_strong.mean()),
        'rec_out_deg_strong_mean': float(out_deg_strong.mean()),
        # 权重
        'rec_absw_mean': float(np.abs(rec_w_vals).mean()),
        'rec_absw_p50': float(np.percentile(np.abs(rec_w_vals), 50)),
        'rec_absw_p99': float(np.percentile(np.abs(rec_w_vals), 99)),
        'rec_gini': gini(rec_w_vals),
        'rec_top1pct_weight_share': top1p_share,
        'rec_exc_frac': float((rec_w_vals > 0).mean()),
        'w_in_absw_mean': float(np.abs(W_in[M_in > 0]).mean()) if (M_in > 0).any() else 0.0,
        'w_out_absw_mean': float(np.abs(W_out[M_out > 0]).mean()) if (M_out > 0).any() else 0.0,
        # 单元动力学
        'tau_mean': float(tau.mean()), 'tau_std': float(tau.std()),
        'tau_p5': float(np.percentile(tau, 5)), 'tau_p95': float(np.percentile(tau, 95)),
        'w_ei_mean': float(w_ei.mean()), 'w_ie_mean': float(w_ie.mean()),
        'w_ei_std': float(w_ei.std()), 'w_ie_std': float(w_ie.std()),
    }
    res.update(extra)
    res['graph'] = graph_metrics(M_rec, W_rec)

    # 供画图的原始数组
    dists = {
        'rec_in_deg': in_deg, 'rec_out_deg': out_deg,
        'rec_absw': np.abs(rec_w_vals), 'tau': tau,
        'm_in_deg': m_in_deg,
    }
    return res, dists


def main():
    all_res, all_dists = {}, {}
    for tag, path in MODELS.items():
        print(f"[{tag}] {os.path.basename(path)} ...")
        res, dists = analyze(tag, path)
        all_res[tag] = res
        all_dists[tag] = dists
        g = res['graph']
        print(f"  N={res['N']} rec_edges={res['n_rec_edges']} (强 {res['strong_frac']:.1%}) "
              f"density={res['rec_density']:.4f} | in-deg cv={res['rec_in_deg_cv']:.2f} "
              f"max={res['rec_in_deg_max']:.0f} top10% share={res['rec_in_deg_top10pct_share']:.2f}")
        print(f"  Q={g['modularity_Q']:.3f} comms={g['n_communities']} "
              f"largest={g['largest_comm_frac']:.1%} | recip={g['reciprocity']:.3f} "
              f"L={g['avg_path_lcc']:.2f} C={g['clustering_transitivity']:.3f} "
              f"sigma={g['small_world_sigma']:.2f} assort={g['assortativity_deg']:.3f}")

    out_json = os.path.join(ROOT, 'results', 'structure_16b_vs_7b.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(all_res, f, ensure_ascii=False, indent=1)
    print(f"已写入 {out_json}")

    # ---------- 对比图 ----------
    colors = {'16b_best': 'tab:red', '16b_latest': 'tab:orange', '7b_latest': 'tab:blue'}
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    ax = axes[0, 0]
    for tag, d in all_dists.items():
        v = d['rec_in_deg']
        ax.hist(v, bins=np.arange(-0.5, max(v.max(), 20) + 1.5, 1), density=True,
                alpha=0.5, label=f"{tag} (N={all_res[tag]['N']})", color=colors[tag])
    ax.set_title('Rec in-degree (fan-in) dist')
    ax.set_xlabel('in-degree (unique sources)'); ax.legend(fontsize=8)

    ax = axes[0, 1]
    for tag, d in all_dists.items():
        v = d['rec_absw']
        ax.hist(v, bins=80, range=(0, np.percentile(all_dists['16b_best']['rec_absw'], 99.5)),
                density=True, alpha=0.5, label=tag, color=colors[tag])
    ax.set_title('|W_rec| dist (existing edges)')
    ax.set_xlabel('|W|'); ax.legend(fontsize=8)

    ax = axes[0, 2]
    for tag, d in all_dists.items():
        v = np.sort(d['rec_absw'])[::-1]
        ax.plot(np.arange(1, len(v) + 1) / len(v), np.cumsum(v) / v.sum(),
                label=f"{tag} gini={all_res[tag]['rec_gini']:.2f}", color=colors[tag])
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='uniform')
    ax.set_title('Lorenz curve of |W_rec|'); ax.set_xlabel('edge frac (sorted)')
    ax.set_ylabel('weight frac'); ax.legend(fontsize=8)

    ax = axes[0, 3]
    for tag, d in all_dists.items():
        ax.hist(d['tau'], bins=40, alpha=0.5, label=tag, color=colors[tag])
    ax.set_title('tau_e_init dist'); ax.set_xlabel('tau'); ax.legend(fontsize=8)

    ax = axes[1, 0]
    for tag, d in all_dists.items():
        v = d['m_in_deg']
        ax.hist(v, bins=np.arange(-0.5, 21.5, 1), density=True, alpha=0.5,
                label=f"{tag} (O={all_res[tag]['obs_dim']})", color=colors[tag])
    ax.set_title('Input fan-in per column (M_in row sum)')
    ax.set_xlabel('#input features'); ax.legend(fontsize=8)

    # 度-度散点（hub 结构）：in vs out degree
    ax = axes[1, 1]
    for tag, d in all_dists.items():
        ax.scatter(d['rec_out_deg'], d['rec_in_deg'], s=4, alpha=0.35,
                   label=tag, color=colors[tag])
    ax.set_title('Column degree map (out vs in)')
    ax.set_xlabel('out-degree'); ax.set_ylabel('in-degree'); ax.legend(fontsize=8)

    # 关键结构指标条形对比
    ax = axes[1, 2]
    tags = list(MODELS.keys())
    keys = ['modularity_Q', 'reciprocity', 'assortativity_deg']
    x = np.arange(len(keys))
    w_ = 0.26
    for i, tag in enumerate(tags):
        vals = [all_res[tag]['graph'][k] for k in keys]
        ax.bar(x + (i - 1) * w_, vals, w_, label=tag, color=colors[tag])
    ax.set_xticks(x); ax.set_xticklabels(['Q (modularity)', 'reciprocity', 'assort.'])
    ax.set_title('Graph structure (higher = more structured)')
    ax.legend(fontsize=8)

    ax = axes[1, 3]
    keys2 = ['rec_density', 'strong_frac', 'rec_in_deg_top10pct_share',
             'largest_comm_frac']
    lbls2 = ['rec density', '|W|>0.01 frac', 'top10% indeg share', 'largest comm frac']
    x = np.arange(len(keys2))
    for i, tag in enumerate(tags):
        vals = [all_res[tag]['graph'][k] if k == 'largest_comm_frac'
                else all_res[tag][k] for k in keys2]
        ax.bar(x + (i - 1) * w_, vals, w_, label=tag, color=colors[tag])
    ax.set_xticks(x); ax.set_xticklabels(lbls2, fontsize=7, rotation=12)
    ax.set_title('Connectivity concentration')
    ax.legend(fontsize=8)

    fig.suptitle('Brain structure: test16b (sparse fan-in K=16, N=1024, obs40) '
                 'vs test7b (dense mask, N=256, obs32)', fontsize=13)
    plt.tight_layout()
    out_png = os.path.join(ROOT, 'results', 'structure_16b_vs_7b.png')
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    print(f"已写入 {out_png}")


if __name__ == '__main__':
    main()
