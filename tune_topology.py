# ==========================================
# tune_topology.py —— 拓扑弹簧布局调参（生图脚本）
#
# ★★★ 拓扑弹簧默认参数 = 下面最顶部的 TOPO_PARAMS 类，直接改值保存即可 ★★★
#   1. 改值 → 运行 python tune_topology.py → 看 topology_tune.png
#   2. 调确定后什么都不用做：render_win_video.py 出片时自动读取这份
#      TOPO_PARAMS（同一份参数，无需搬运）
#   3. CLI 只做临时实验用（--rep-mult 4 之类），不会改动文件里的默认值
#
# 三组弹簧相互独立：
#   W_rec : 柱-柱贴珠边（L0_REC / K_REC）
#   W_in  : 输入通道→柱（L0_IN / K_REC×K_IN_MULT）
#   W_out : 动作←柱  （L0_OUT / K_REC×K_OUT_MULT）
# ==========================================

# ##########################################
# ▼▼▼ TOPO_PARAMS —— 拓扑弹簧默认参数（改这里） ▼▼▼
# ##########################################
class TOPO_PARAMS:
    # ---- 迭代 / 退火 ----
    SEED = 7            # 布局随机种子（同参数同形状）
    ITERS = 2500        # 退火迭代轮数（越大越收敛，越慢）
    T0 = 0.020          # 初始温度（步长系数，线性退火到 0）
    MAX_STEP = 0.02     # 单轮最大位移（防发散）
    MARGIN = 0.04       # 自由柱活动边界
    CENTER_PULL = 0.0  # 微弱向心力（仅收编完全无约束的散点）
    # ---- 节点 ----
    NODE_D = 0.008      # 小球直径（布局单位；柱 marker ≈0.0065）
    # ---- W_rec 组：柱-柱贴珠边 ----
    L0_REC = 4*NODE_D     # 原长 = 小球直径（贴珠）；改 --node-d 时若未单独指定则跟随
    K_REC = 0.0001        # 基础刚度（实际 ×(0.35+0.65·|w|归一)）
    REP_MULT = 4.0      # 硬核斥力倍率（d⁻³ 衰减；d=NODE_D 处力度=K_REC·NODE_D×此值）
    N_REC_COMPUTE = 64  # 布局计算弹簧数/柱（决定平衡形状；≤扇入96）
    N_REC_SHOW = 8      # 显示连线数/柱（取计算边中 |w| 前8，防线条过密）
    # ---- W_in 组：输入通道→柱 ----
    L0_IN = 0.0         # 原长（0=吸附到输入钉扎点）
    K_IN_MULT = 60.0    # 刚度 = K_REC × 此值
    N_IN_FAN = 2        # 每通道连出的 top 柱数
    # ---- W_out 组：动作←柱 ----
    L0_OUT = 0.0        # 原长（0=吸附到输出钉扎点）
    K_OUT_MULT = 80.0   # 刚度 = K_REC × 此值
    N_OUT_FAN = 3       # 每动作连入的 top 柱数
    # ---- 钉扎位置 ----
    IN_X = 0.16         # 输入钉扎列 x（标签在其左侧）
    OUT_X = 0.94        # 输出钉扎列 x
    OUT_SPAN = 0.40     # 输出三点上下总跨度（上=左转 / 中=直行 / 下=右转）
    # ---- 纵向展开力场（抗重叠）：自平衡、位置依赖 ——
    #      以珠云当前竖直中心为界、云自身半高归一：F = SPREAD·wprof(x)·4u(1−u)·sign(u)；
    #      力在云的上/下边缘与中轴处都趋于 0，净外力≈0（无整体漂移），
    #      小球停在「弹簧拉力 = 场力」的内部平衡位置 ----
    SPREAD = 0.0005      # 强度（0=关闭；越大纵向越展开）
    SPREAD_MID = 0.5   # 力场中心 x（≈(IN_X+OUT_X)/2）
    SPREAD_SIGMA = 0.35 # 高斯宽度：越大两侧衰减越慢
    # ---- 动量阻尼（抗震荡+加速收敛）：v = DAMP·v + F·LR，pos += v ----
    DAMP = 0.85         # 速度保留率（=1−阻尼）：震荡大就调低(如0.7)，收敛慢就调高(如0.95)
    # ---- 弛豫收敛（退火后继续迭代到力平衡，消灭"冻在半路"的小球） ----
    RELAX_LR = 0.05     # 弛豫步长系数
    RELAX_ITERS = 30000 # 弛豫轮数上限
    RELAX_TOL = 5e-5    # 收敛判据：最大单步位移低于此值视为平衡
    # ---- 仅调参图显示 ----
    VIEW_RAND_LIT = True  # 小球随机点亮（方便数珠子/看重叠；不影响视频）
# ##########################################
# ▲▲▲ TOPO_PARAMS 结束 ▲▲▲
# ##########################################

import argparse
import os
import sys
import time

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def main():
    # 延迟导入：保证 TOPO_PARAMS 先定义（render_win_video 会读取它）
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import render_win_video as rwv

    ap = argparse.ArgumentParser(description='拓扑弹簧布局调参（快速单图）')
    ap.add_argument('--model', default='16c_cheat7b_win_model.pth')
    ap.add_argument('--out', default='topology_tune.png')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--iters', type=int, default=None)
    ap.add_argument('--t0', type=float, default=None)
    ap.add_argument('--max-step', type=float, default=None)
    ap.add_argument('--margin', type=float, default=None)
    ap.add_argument('--center-pull', type=float, default=None)
    ap.add_argument('--node-d', type=float, default=None,
                    help='小球直径（--l0-rec 未指定时 rec 原长跟随此值）')
    ap.add_argument('--l0-rec', type=float, default=None)
    ap.add_argument('--k-rec', type=float, default=None)
    ap.add_argument('--rep-mult', type=float, default=None)
    ap.add_argument('--n-rec-compute', type=int, default=None)
    ap.add_argument('--n-rec-show', type=int, default=None)
    ap.add_argument('--n-in-fan', type=int, default=None)
    ap.add_argument('--n-out-fan', type=int, default=None)
    ap.add_argument('--l0-in', type=float, default=None)
    ap.add_argument('--k-in-mult', type=float, default=None)
    ap.add_argument('--l0-out', type=float, default=None)
    ap.add_argument('--k-out-mult', type=float, default=None)
    ap.add_argument('--in-x', type=float, default=None)
    ap.add_argument('--out-x', type=float, default=None)
    ap.add_argument('--spread', type=float, default=None)
    ap.add_argument('--spread-sigma', type=float, default=None)
    ap.add_argument('--no-labels', action='store_true')
    args = ap.parse_args()

    # CLI 临时覆盖（只影响本次运行，不改 TOPO_PARAMS 文件值）
    def set(name, val):
        if val is not None:
            setattr(rwv, name, val)
    set('LAYOUT_SEED', args.seed)
    set('LAYOUT_ITERS', args.iters)
    set('LAYOUT_T0', args.t0)
    set('LAYOUT_MAX_STEP', args.max_step)
    set('LAYOUT_MARGIN', args.margin)
    set('LAYOUT_CENTER_PULL', args.center_pull)
    set('NODE_D', args.node_d)
    if args.l0_rec is not None:
        rwv.L0_REC = args.l0_rec
    elif args.node_d is not None:
        rwv.L0_REC = args.node_d          # 默认 rec 原长 = 球径
    set('K_REC', args.k_rec)
    set('REP_MULT', args.rep_mult)
    set('N_REC_COMPUTE', args.n_rec_compute)
    set('N_REC_SHOW', args.n_rec_show)
    set('N_IN_FAN', args.n_in_fan)
    set('N_OUT_FAN', args.n_out_fan)
    if args.l0_in is not None:
        rwv.L0_IN = args.l0_in
    set('K_IN_MULT', args.k_in_mult)
    if args.l0_out is not None:
        rwv.L0_OUT = args.l0_out
    set('K_OUT_MULT', args.k_out_mult)
    set('IN_X', args.in_x)
    set('OUT_X', args.out_x)
    set('SPREAD', args.spread)
    set('SPREAD_SIGMA', args.spread_sigma)

    torch.manual_seed(0)
    data = torch.load(args.model, map_location='cpu', weights_only=False)
    st = data['brain']

    t0 = time.time()
    L = rwv.precompute_layout(st)
    print(f'布局完成: 退火{rwv.LAYOUT_ITERS}轮 + 弛豫{L.get("relax_iters")}轮'
          f'（共 {time.time() - t0:.1f}s）')
    print(f'W_rec : 计算{rwv.N_REC_COMPUTE}/柱 + 显示{rwv.N_REC_SHOW}/柱  '
          f'L0={rwv.L0_REC:.4f}  K={rwv.K_REC:.2g}')
    print(f'W_in  : fan={rwv.N_IN_FAN}/通道   L0={rwv.L0_IN:.4f}  '
          f'K={rwv.K_REC * rwv.K_IN_MULT:.2f}  (=K_REC×{rwv.K_IN_MULT:g})')
    print(f'W_out : fan={rwv.N_OUT_FAN}/动作   L0={rwv.L0_OUT:.4f}  '
          f'K={rwv.K_REC * rwv.K_OUT_MULT:.2f}  (=K_REC×{rwv.K_OUT_MULT:g})')
    print(f'硬核斥力={rwv.REP_MULT * rwv.K_REC * rwv.NODE_D ** 4:.3e} (d⁻³)  '
          f'temp0={rwv.LAYOUT_T0}  maxstep={rwv.LAYOUT_MAX_STEP}  '
          f'margin={rwv.LAYOUT_MARGIN}  center={rwv.LAYOUT_CENTER_PULL}  '
          f'seed={rwv.LAYOUT_SEED}')
    print('（以上参数即 render_win_video.py 出片所用参数）')

    fig, ax = plt.subplots(figsize=(14, 9), dpi=120, facecolor=rwv.BG)
    ax.set_facecolor(rwv.BG)
    for sp in ax.spines.values():
        sp.set_color(rwv.GRID)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.add_collection(LineCollection(L['edges'], colors=L['edge_rgba'],
                                     linewidths=L['edge_lws']))
    ax.add_collection(LineCollection(L['in_segs'], colors=L['in_rgba'],
                                     linewidths=L['in_lws']))
    ax.add_collection(LineCollection(L['out_segs'], colors=L['out_rgba'],
                                     linewidths=L['out_lws']))
    cx, cy = L['col_xy']
    if TOPO_PARAMS.VIEW_RAND_LIT:
        rng_lit = np.random.default_rng(TOPO_PARAMS.SEED)
        lit = rng_lit.uniform(0.25, 1.0, len(cx))
        ax.scatter(cx, cy, s=26, c=rwv.CMAP_ACT(lit), zorder=4,
                   edgecolors='#3A3A3A', linewidths=0.5)
    else:
        ax.scatter(cx, cy, s=26, c=rwv.DIM, zorder=4,
                   edgecolors='#3A3A3A', linewidths=0.5)
    in_base = np.array([rwv._hex2rgb(rwv.OBS32_GROUP_COLORS[ch])
                        for ch in range(32)])
    ax.scatter(L['in_xy'][:, 0], L['in_xy'][:, 1], s=30, c=in_base,
               marker='s', zorder=5)
    ax.scatter(L['out_xy'][:, 0], L['out_xy'][:, 1], s=42,
               c=rwv._hex2rgb(rwv.ACCENT_OUT)[None, :], marker='s', zorder=5)
    if not args.no_labels:
        for ch, (x, y) in enumerate(L['in_xy']):
            ax.text(x - 0.012, y, f'{ch} {rwv.OBS32_PROJ_LABELS[ch]}',
                    color=rwv.OBS32_GROUP_COLORS[ch], fontsize=7,
                    family='Microsoft YaHei', va='center', ha='right')
        for a, (x, y) in enumerate(L['out_xy']):
            ax.text(x + 0.012, y, rwv.ACTION_ZH[a], color=rwv.ACCENT_OUT,
                    fontsize=9, family='Microsoft YaHei', va='center')
    ax.set_title(f'{os.path.basename(args.model)} · iters={rwv.LAYOUT_ITERS} '
                 f'seed={rwv.LAYOUT_SEED} · rec L0={rwv.L0_REC:.4f} K={rwv.K_REC:g}'
                 f' | in L0={rwv.L0_IN:g} ×{rwv.K_IN_MULT:g}'
                 f' | out L0={rwv.L0_OUT:g} ×{rwv.K_OUT_MULT:g}',
                 color=rwv.FG, fontsize=11, family='monospace', loc='left')
    fig.savefig(args.out, facecolor=rwv.BG)
    print('已保存:', args.out)


if __name__ == '__main__':
    main()
