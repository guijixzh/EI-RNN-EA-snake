# ==========================================
# render_win_video.py —— 通关版本游戏过程视频化（1080P）v3
#
# 布局：游戏本体（左上） | 网络拓扑（右上） | 柱激活滚窗热力图（下方横贯）
# 风格：深色底 + 丰富色板（激活 magma、输入青、输出琥珀、obs 组别彩标）
#
# v3 修正（对照 v2）：
#   1. 配色升级：热力图/拓扑激活改 magma 色图；obs 32 通道组别标签用通用
#      后端同款彩色（头绿/尾青/食黄/身紫/障蓝）；输入节点=组别彩标方块、
#      输出节点=琥珀方块（亮度实时调制），输入边发光跟随通道组色
#   2. 弹簧力学规格化：W_rec 原长 = 小球直径（贴珠网格）；W_in/W_out 原长 = 0
#      且刚度 = W_rec 的 10 倍（吸附到钉扎点）
#   3. 实时化：一视频帧 = 一个"思考帧"（神经网络计算步）。每环境步播 5 帧，
#      热力图/拓扑每思考帧推进一次；棋盘在前 4 帧静止、第 5 帧随决策一起移动；
#      输出亮度显示 5 帧累加 logits 的实时 softmax（决策逐渐成形）
#   4. rec 内部边按权重符号分 E/I 双色（正=兴奋橙红 / 负=抑制蓝），
#      三组边线宽均 ∝ |w|；初始神经元强度：出片默认真实全零启动（E0_INIT='zero'，
#      原始通关轨迹 98/98/1825 步），--preview 静态样张默认随机点亮（看图用），
#      可用 --e0 覆盖
#
# 复用：test16c_cheat7b 的回放骨架（win 模型 + make_bank 固定地图 + forward_batch）
# 用法：
#   python render_win_video.py --preview                  # 首/中/末样张 PNG 快速确认
#   python render_win_video.py                            # 全量渲染 win_run_1080p_v2.mp4
#   python render_win_video.py --model 16c_cheat7b_efficient_model.pth --min-food 97
#   python render_win_video.py --fps 30 --skip 6          # 紧凑版
# ==========================================

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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

_repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_repo, 'experiments', 'test16_series'))
import test16c_cheat7b as sim

# ==========================================
# ▼▼▼ 调参区（全部可视化参数集中在此） ▼▼▼
# ==========================================
# ---- 配色（深色底 + 丰富色板） ----
BG = '#0A0A0A'          # 背景
FG = '#E6E6E6'          # 主文字
GRID = '#262626'        # 网格/边框
DIM = '#3A3A3A'         # 弱化元素（柱节点默认色）
FOOD = '#D7263D'        # 食物红
ACCENT_IN = "#28713E"   # 输入侧青（钉扎列/输入边）
ACCENT_OUT = "#AA7C40"  # 输出侧琥珀（钉扎列/输出边/动作文本）
RING = "#EBE6E6"        # 动作定型高亮环（白）
RING_TRY = '#8A929E'    # 决策未定型时随队环（灰）
BLUE_RING = "#4C9AFF"   # act 帧内选中动作的蓝色圆圈
BLUE_RING_S = 240       # 蓝圈面积（比节点高亮圆稍大）
# 连线亮度（三组边统一）：平时暗，活动时按 |w| 与源端激活变亮
EDGE_DIM_BASE = 0.06    # 静默基底
EDGE_DIM_GAIN = 0.20    # 静默时 |w| 亮度系数
EDGE_ACT_GAIN = 0.85    # 激活增量系数（×|w|×源端激活）
# 热力图列重排：对称化 → 谱嵌入 → k-means 簇排序 → 簇内 τ 排序 → 相邻交换微调
HEAT_CLUSTER = True
HEAT_CLUSTERS = 12      # 簇数（谱嵌入维度）
HEAT_SEED = 0           # k-means 种子（确定性）
HEAT_LOCAL_PASSES = 30  # 相邻交换微调的最大轮数
# 激活色图：magma（黑→紫→橙→黄），下端抬到 0.08 让静默柱在深底上隐约可辨
_magma = plt.get_cmap('magma')
CMAP_ACT = LinearSegmentedColormap.from_list(
    'magma08', _magma(np.linspace(0.08, 1.0, 256)))
# rec 循环驱动色图：暗中心发散（负端=蓝叉净抑制，正端=magma 净兴奋，与 E 面板同族）
CMAP_REC = LinearSegmentedColormap.from_list(
    'magma_fork', ['#DFF2FF', '#63A8FF', '#3550B0', '#141040', '#0A0A0A',
                   '#4A1060', '#C13B7B', '#FC8961', '#FCFDBF'])
# obs 32 通道组别标签色（照搬 tools/static/visualizer.js OBS32_PROJ_GROUPS）
GRP_HEAD = '#3fb950'; GRP_TAIL = '#3fe0c8'; GRP_FOOD = '#ffd23f'
GRP_BODY = '#d07ff5'; GRP_WALL = '#4aa8ff'

# ---- 蛇身样式（对齐 visualizer.js，白化） ----
SNAKE_C_HEAD = (1.00, 1.00, 1.00)     # 头端颜色（最亮白）
SNAKE_C_TAIL = (0.333, 0.333, 0.353)  # 尾端颜色（暗灰）
SNAKE_FILL = 0.88       # 方块边长占格子比例
BODY_EDGE_A = 0.27      # 方块黑描边透明度
POLY_ALPHA = 0.45       # 折线不透明度
POLY_LW_FRAC = 0.11     # 折线宽 = 0.11 × 格宽
EYE_SIDE = 0.17         # 眼睛垂直偏移 × 格宽
EYE_FWD = 0.19          # 眼睛前向偏移 × 格宽
EYE_R = 0.21            # 眼睛半径 × 格宽
FOOD_R = 0.32           # 食物半径 × 格宽

# ---- 拓扑弹性平衡网格 ----
# ★ 拓扑弹簧默认参数的唯一修改处：tune_topology.py 最顶部的 TOPO_PARAMS 类。
#   下面是从那份参数来的只读映射，出片自动与调参图同参，不要在这里改数值。
from tune_topology import TOPO_PARAMS as _TP
LAYOUT_SEED = _TP.SEED
LAYOUT_ITERS = _TP.ITERS
LAYOUT_T0 = _TP.T0
LAYOUT_MAX_STEP = _TP.MAX_STEP
LAYOUT_MARGIN = _TP.MARGIN
LAYOUT_CENTER_PULL = _TP.CENTER_PULL
NODE_D = _TP.NODE_D
L0_REC = _TP.L0_REC
K_REC = _TP.K_REC
REP_MULT = _TP.REP_MULT
N_REC_COMPUTE = _TP.N_REC_COMPUTE
N_REC_SHOW = _TP.N_REC_SHOW
L0_IN = _TP.L0_IN
K_IN_MULT = _TP.K_IN_MULT
N_IN_FAN = _TP.N_IN_FAN
L0_OUT = _TP.L0_OUT
K_OUT_MULT = _TP.K_OUT_MULT
N_OUT_FAN = _TP.N_OUT_FAN
IN_X = _TP.IN_X
OUT_X = _TP.OUT_X
OUT_SPAN = _TP.OUT_SPAN
SPREAD = _TP.SPREAD
SPREAD_MID = _TP.SPREAD_MID
SPREAD_SIGMA = _TP.SPREAD_SIGMA
DAMP = _TP.DAMP
RELAX_LR = _TP.RELAX_LR
RELAX_ITERS = _TP.RELAX_ITERS
RELAX_TOL = _TP.RELAX_TOL
IN_EDGE_FLOW = True     # 输入边随通道值实时发光

# ---- rec 内部边：E/I 双色 + 权重决定线宽 ----
COLOR_REC_E = "#6A3C34" # 兴奋性（rec_w > 0）静默暗色
COLOR_REC_I = "#38496C" # 抑制性（rec_w < 0）静默暗色
COLOR_REC_E_HI = "#FF7A50"  # 兴奋边激活亮色
COLOR_REC_I_HI = "#63A8FF"  # 抑制边激活亮色
COLOR_OUT_HI = "#FFC96B"    # 输出边激活亮色
CMAP_W = LinearSegmentedColormap.from_list(
    'ei_w', [COLOR_REC_I_HI, '#000000', COLOR_REC_E_HI])  # 连接矩阵：蓝=抑制 红=兴奋
LW_REC_BASE = 0.35      # rec 边线宽 = BASE + GAIN × |w|归一
LW_REC_GAIN = 1.60
LW_IO_BASE = 0.45       # in/out 边线宽同式
LW_IO_GAIN = 1.30

# ---- 初始神经元强度 ----
E0_INIT = 'zero'        # 出片默认：真实全零启动（原始通关轨迹 98/98/1825 步）
E0_INIT_PREVIEW = 'random'  # --preview 静态样张默认：随机点亮便于看图（不影响出片）
E0_SEED = 3             # 随机种子（random 时改变轨迹，须仍通关才可用）
E0_LO, E0_HI = 0.20, 0.80   # 随机初始 E 的取值范围

# ---- 面板 ----
HEAT_WINDOW = 708       # 热力图滚窗请求值；实际自动钳到面板整宽像素列（1帧=1px 像素对齐）
FPS = 60                # 视频帧率
FRAMES_PER_BRAIN = 2    # 每个思考帧（神经步）占的视频帧数（脑面板/热力图推进节奏）
FRAMES_PER_GAME = 10    # 每个游戏步占的视频帧数（棋盘跳格节奏；=FRAMES_PER_BRAIN×思考帧数 时为自然同步）
HOLD = 30               # 末帧定格
LABEL_FS = 6.5          # 输入语义标签字号
G = 10
BOARD_SPAN = G + 1.2    # 棋盘坐标跨度（-0.6 .. G-0.4）
ACTION_NAMES = ('FWD', 'LEFT', 'RIGHT')
ACTION_ZH = ('直行', '左转', '右转')
DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))   # (d_row, d_col)：右/下/左/上

# ---- obs32('32proj7b') 32 通道语义标签（照搬 tools/static/visualizer.js OBS32_PROJ_LABELS） ----
OBS32_PROJ_LABELS = [
    "头·右", "头·下", "头·左", "头·上",
    "尾·右", "尾·下", "尾·左", "尾·上",
    "食·前", "食·左前", "食·左", "食·左后", "食·后", "食·右后", "食·右", "食·右前",
    "身·前", "身·左前", "身·左", "身·左后", "身·后", "身·右后", "身·右", "身·右前",
    "障·前", "障·左前", "障·左", "障·左后", "障·后·√len", "障·右后", "障·右", "障·右前",
]
OBS32_GROUP_COLORS = ([GRP_HEAD] * 4 + [GRP_TAIL] * 4 + [GRP_FOOD] * 8
                      + [GRP_BODY] * 8 + [GRP_WALL] * 8)
# ==========================================
# ▲▲▲ 调参区结束 ▲▲▲
# ==========================================


def _hex2rgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


# ==========================================
# 1. 回放采集（复用 test16c_cheat7b 骨架）
#    一视频帧 = 一个思考帧（网络计算步）：
#    - 每环境步依次记录 K 个思考帧（末帧E / 实时Σlogits softmax / 衰减后输入）
#    - 前 K-1 帧棋盘保持移动前局面，第 K 帧随决策一起切到移动后局面
# ==========================================
def rollout(cfg, st, max_steps, e0_init=None, bank=None):
    e0_init = e0_init or E0_INIT
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg.DEVICE = str(dev)
    pop = sim.GeneStack(cfg, B=1, device=dev)
    pop.random_init()
    pop.set_individual_from_state(0, st)
    pop.refresh_eff()
    if getattr(cfg, 'USE_FP16', True):
        pop.fp16(); pop.refresh_eff()
    if bank is None:
        bank = sim.make_bank(cfg, cfg.MAP_GEN, 1, 0, dev)
    env = sim.BatchedSnakeEnv(cfg, 1, dev)
    env.reset(bank=bank)
    E = torch.zeros(1, pop.N, dtype=pop.dtype, device=dev)
    if e0_init == 'random':
        g = torch.Generator(device='cpu').manual_seed(E0_SEED)
        E = (torch.rand(1, pop.N, generator=g) * (E0_HI - E0_LO) + E0_LO
             ).to(dev, pop.dtype)
    I = torch.zeros_like(E); stt = torch.zeros_like(E)
    cts = torch.zeros(1, pop.A, dtype=pop.dtype, device=dev)
    K = int(cfg.FRAME_RATE)
    fanin = pop.rec_idx.shape[-1]
    frames = []
    won = False
    score = 0
    n_steps = 0

    def snap(food_override=None):
        return dict(
            body=env.body[0, :env.body_len[0]].cpu().numpy().copy(),
            food=(env.food[0].cpu().numpy().copy() if food_override is None
                  else food_override),
            head=env.head[0].cpu().numpy().copy(),
            dir=int(env.dir_idx[0]),
            score=score, step=int(env.steps[0]),
        )

    for _ in range(max_steps):
        if not bool(env.alive.any()):
            break
        obs = env.obs().to(pop.dtype)
        obs_np = obs[0].float().cpu().numpy().copy()
        pre = snap()
        logits_sum = None
        for k in range(K):
            o = obs * (cfg.INPUT_DECAY ** k)
            Eg = torch.gather(E.unsqueeze(-1).expand(E.shape[0], -1, fanin),
                              1, pop.rec_idx)
            rec = (Eg * pop.rec_w).sum(-1)     # 循环驱动电流（本帧 E 的输入，带符号）
            logits, E, I, stt = sim.forward_batch(pop, o, E, I, stt, cts, cfg)
            logits_sum = logits if logits_sum is None else logits_sum + logits
            frames.append(dict(
                e=E[0].float().cpu().numpy().copy(),
                rec=rec[0].float().cpu().numpy().copy(),
                obs=obs_np * (float(cfg.INPUT_DECAY) ** k),  # 网络该帧实际输入
                probs=torch.softmax(logits_sum[0].float(), dim=0).cpu().numpy().copy(),
                k=k, act=-1,
                act_running=int(torch.argmax(logits_sum[0])),
                **pre))
        act = torch.argmax(logits_sum, dim=1)
        cts = sim.update_fatigue(cts, act)
        env.step(act)
        score += int(env.ate.sum())
        won_now = bool(env.won[0])
        post = snap(np.array([-9.0, -9.0]) if won_now else None)
        frames[-1]['act'] = int(act[0])
        frames[-1].update(post)
        n_steps += 1
        if won_now:
            won = True
            break
    # 终态帧（定格素材；盘满无食物）
    final_board = snap(np.array([-9.0, -9.0]) if won else None)
    frames.append(dict(e=frames[-1]['e'], rec=frames[-1]['rec'],
                       obs=frames[-1]['obs'],
                       probs=frames[-1]['probs'],
                       k=K - 1, act=-1,
                       act_running=-1, **final_board))
    assert len(frames) == n_steps * K + 1, \
        f'帧数 {len(frames)} != 步数 {n_steps} × {K} + 1'
    return frames, won, n_steps


# ==========================================
# 2. 拓扑布局：弹性平衡网格（确定性）
#    力学规格：W_rec 原长=小球直径（贴珠）；W_in/W_out 原长=0、刚度×10（吸附钉扎点）
#    256 柱自由；32 输入钉左列、3 输出钉右列；温度线性退火
# ==========================================
def precompute_layout(st):
    N, K = int(st['N']), int(st['K'])
    n_in, n_out = 32, 3
    total = N + n_in + n_out
    rng = np.random.default_rng(LAYOUT_SEED)

    in_xy = np.stack([np.full(n_in, IN_X), np.linspace(0.90, 0.10, n_in)], 1)
    # 输出钉扎：直行居中，左转在上、右转在下（与动作索引 0/1/2 对应）
    out_y = np.array([0.5,                              # 0 直行
                      0.5 + OUT_SPAN / 2,               # 1 左转
                      0.5 - OUT_SPAN / 2])              # 2 右转
    out_xy = np.stack([np.full(n_out, OUT_X), out_y], 1)

    pos = np.zeros((total, 2), dtype=np.float64)
    pos[N:] = np.concatenate([in_xy, out_xy], 0)
    ang = np.linspace(0.0, 2 * np.pi, N, endpoint=False) + np.pi / 2
    pos[:N, 0] = 0.55 + 0.30 * np.cos(ang)
    pos[:N, 1] = 0.50 + 0.30 * np.sin(ang)
    pos[:N] += rng.normal(0.0, 0.02, (N, 2))

    # ---- 边集合（全局编号：柱 0..N-1，输入 N..N+31，输出 N+32..） ----
    # 计算与显示分离：
    #   计算弹簧 = 每柱 |w| top-N_REC_COMPUTE（决定平衡形状）；
    #   显示连线 = 其中每柱前 N_REC_SHOW 条（|w| 更强的子集，防线条过密）；
    # 同时记录权重符号（E/I）供双色渲染
    wr = st['rec_w'].float().abs()
    wr_raw = st['rec_w'].float()
    kk = min(K, N_REC_COMPUTE + 2)                # 多取 2 槽备滤自环/零权
    vals, idxs = torch.topk(wr, kk, dim=1)        # [N,kk]
    raw_g = torch.gather(wr_raw, 1, idxs)
    comp = [[] for _ in range(N)]                 # 每柱计算边 (src,w,sign)
    for d in range(N):
        row_seen = set()
        for j in range(kk):
            if len(comp[d]) >= N_REC_COMPUTE:
                break
            w = float(vals[d, j])
            if w <= 0:
                break
            s = int(st['rec_idx'][d, int(idxs[d, j])])
            if s == d or s in row_seen:
                continue
            row_seen.add(s)
            comp[d].append((s, w, float(torch.sign(raw_g[d, j]))))
    c_pairs = [(s, d) for d in range(N) for (s, w, g) in comp[d]]
    c_w = [w for d in range(N) for (s, w, g) in comp[d]]
    c_s = [g for d in range(N) for (s, w, g) in comp[d]]
    s_pairs = [(s, d) for d in range(N) for (s, w, g) in comp[d][:N_REC_SHOW]]
    s_w = [w for d in range(N) for (s, w, g) in comp[d][:N_REC_SHOW]]
    s_sign = [g for d in range(N) for (s, w, g) in comp[d][:N_REC_SHOW]]
    # 计算弹簧（受力用）
    sp_src = np.array([p[0] for p in c_pairs], dtype=np.int64)
    sp_dst = np.array([p[1] for p in c_pairs], dtype=np.int64)
    c_wn = np.array(c_w)
    c_wn = c_wn / max(c_wn.max(), 1e-9)
    c_sign = np.array(c_s)
    # 显示连线（画图用，计算边的子集）
    r_src = np.array([p[0] for p in s_pairs], dtype=np.int64)
    r_dst = np.array([p[1] for p in s_pairs], dtype=np.int64)
    r_wn = np.array(s_w)
    r_wn = r_wn / max(r_wn.max(), 1e-9)
    r_sign = np.array(s_sign)
    rec_rgba = np.zeros((len(r_src), 4))
    rec_rgba[r_sign > 0, :3] = _hex2rgb(COLOR_REC_E)
    rec_rgba[r_sign <= 0, :3] = _hex2rgb(COLOR_REC_I)
    rec_rgba[:, 3] = EDGE_DIM_BASE + EDGE_DIM_GAIN * r_wn   # 静默基底亮度（视频内随激活变亮）

    min_w = st['W_in'].float().abs() * st['M_in'].float()      # [256,32]
    tk = min_w.topk(N_IN_FAN, dim=0)
    i_cols = tk.indices.T.numpy()                               # [32,fan]
    i_w = tk.values.T.numpy()
    i_w = i_w / np.clip(i_w.max(1, keepdims=True), 1e-9, None)
    i_ch = np.repeat(np.arange(n_in), N_IN_FAN)
    i_col = i_cols.flatten()
    i_wn = i_w.flatten()

    out_w = st['W_out'].float().abs() * st['M_out'].float()     # [3,256]
    tk_o = out_w.topk(N_OUT_FAN, dim=1)
    o_cols = tk_o.indices.numpy()                               # [3,fan]
    o_w = tk_o.values.numpy()
    o_w = o_w / np.clip(o_w.max(1, keepdims=True), 1e-9, None)
    o_act = np.repeat(np.arange(n_out), N_OUT_FAN)
    o_col = o_cols.flatten()
    o_wn = o_w.flatten()

    in_rgba = np.zeros((len(i_ch), 4))
    in_rgba[:, :3] = _hex2rgb(ACCENT_IN)
    in_rgba[:, 3] = 0.12 + 0.78 * i_wn           # 亮度 ∝ |w|
    out_rgba = np.zeros((len(o_act), 4))
    out_rgba[:, :3] = _hex2rgb(ACCENT_OUT)
    out_rgba[:, 3] = EDGE_DIM_BASE + EDGE_DIM_GAIN * o_wn  # 静默基底亮度

    e_src = np.concatenate([sp_src, N + i_ch, N + n_in + o_act])
    e_dst = np.concatenate([sp_dst, i_col, o_col])
    e_l0 = np.concatenate([np.full(len(sp_src), L0_REC),
                           np.full(len(i_ch), L0_IN),
                           np.full(len(o_act), L0_OUT)])
    e_k = np.concatenate([K_REC * (0.35 + 0.65 * c_wn),
                          K_REC * K_IN_MULT * (0.35 + 0.65 * i_wn),
                          K_REC * K_OUT_MULT * (0.35 + 0.65 * o_wn)])
    # d⁻³ 硬核斥力：d=NODE_D 处力度 = REP_MULT·K_REC·NODE_D（与弹簧同量级），
    # 远离时按立方快速衰减，只防渲染重叠、不扭曲整体网格
    rep = REP_MULT * K_REC * NODE_D ** 4

    # ---- 弹性平衡迭代 ----
    center = np.array([0.55, 0.50])

    def _forces():
        diff = pos[e_dst] - pos[e_src]
        dist = np.sqrt((diff ** 2).sum(1)) + 1e-9
        f = (e_k * (dist - e_l0) / dist)[:, None] * diff
        F = np.zeros_like(pos)
        F[:, 0] += (np.bincount(e_src, weights=f[:, 0], minlength=total)
                    - np.bincount(e_dst, weights=f[:, 0], minlength=total))
        F[:, 1] += (np.bincount(e_src, weights=f[:, 1], minlength=total)
                    - np.bincount(e_dst, weights=f[:, 1], minlength=total))
        delta = pos[:N][:, None, :] - pos[:N][None, :, :]       # [N,N,2]
        d2 = (delta ** 2).sum(-1) + 1e-9
        F[:N] += ((rep / d2 ** 1.5)[..., None] * delta).sum(1)
        # 纵向展开力场（自平衡，位置依赖）：以珠云当前竖直中心为界、云自身半高归一，
        # F = SPREAD·wprof(x)·4u(1−u)·sign(u)。力在云的上/下边缘与中轴处都趋于 0，
        # 净外力≈0（无整体漂移），小球停在「弹簧拉力 = 场力」的内部平衡位置
        if SPREAD > 0:
            wprof = np.exp(-0.5 * ((pos[:N, 0] - SPREAD_MID) / SPREAD_SIGMA) ** 2)
            yc = pos[:N, 1].mean()
            half = max(float(pos[:N, 1].max() - yc),
                       float(yc - pos[:N, 1].min()), 1e-3)
            u = (pos[:N, 1] - yc) / half                            # -1..1
            lobe = np.clip(4.0 * np.abs(u) * (1.0 - np.abs(u)), 0.0, 1.0)
            F[:N, 1] += SPREAD * wprof * lobe * np.sign(u)
        F[:N] += LAYOUT_CENTER_PULL * (center - pos[:N])
        return F

    # 动量阻尼（重球法）：v = DAMP·v + F·系数。阻尼吃掉过冲振荡，
    # 稳态速度放大 1/(1−DAMP) 倍 —— 小刚度参数下收敛显著加快
    vel = np.zeros_like(pos)

    for it in range(LAYOUT_ITERS):
        temp = LAYOUT_T0 * (1.0 - it / LAYOUT_ITERS) + 1e-5
        vel[:N] = DAMP * vel[:N] + _forces()[:N] * temp
        step = vel[:N]
        sn = np.sqrt((step ** 2).sum(1, keepdims=True)) + 1e-12
        step = step * np.minimum(1.0, LAYOUT_MAX_STEP / sn)
        pos[:N] += step
        np.clip(pos[:N], LAYOUT_MARGIN, 1.0 - LAYOUT_MARGIN, out=pos[:N])

    # ---- 弛豫收敛：退火冻结的是轨迹快照（晚期温度→0，弱弹簧拉不回飞出的珠子）；
    #      这里以恒定步长继续迭代，直到真正的力平衡（最大位移 < RELAX_TOL） ----
    relax_used = 0
    for _ in range(RELAX_ITERS):
        vel[:N] = DAMP * vel[:N] + _forces()[:N] * RELAX_LR
        step = vel[:N]
        sn = np.sqrt((step ** 2).sum(1, keepdims=True)) + 1e-12
        step = step * np.minimum(1.0, LAYOUT_MAX_STEP / sn)
        pos[:N] += step
        np.clip(pos[:N], LAYOUT_MARGIN, 1.0 - LAYOUT_MARGIN, out=pos[:N])
        relax_used += 1
        if float(sn.max()) < RELAX_TOL:
            break

    # ---- 绘图素材 ----
    rec_segs = np.stack([np.stack([pos[r_src, 0], pos[r_src, 1]], 1),
                         np.stack([pos[r_dst, 0], pos[r_dst, 1]], 1)], axis=1)
    i_segs = np.stack([np.stack([in_xy[i_ch, 0], in_xy[i_ch, 1]], 1),
                       pos[i_col]], axis=1)
    o_segs = np.stack([np.stack([out_xy[o_act, 0], out_xy[o_act, 1]], 1),
                       pos[o_col]], axis=1)
    return dict(col_xy=(pos[:N, 0], pos[:N, 1]),
                edges=rec_segs, edge_rgba=rec_rgba,
                edge_lws=LW_REC_BASE + LW_REC_GAIN * r_wn,
                edge_wn=r_wn, rec_src=r_src, rec_sign=r_sign,
                relax_iters=relax_used,
                in_xy=in_xy, out_xy=out_xy,
                in_segs=i_segs, out_segs=o_segs,
                in_rgba=in_rgba, out_rgba=out_rgba,
                in_edge_alpha=i_wn, out_edge_alpha=o_wn,
                out_col=o_col, out_wn=o_wn,
                in_lws=LW_IO_BASE + LW_IO_GAIN * i_wn,
                out_lws=LW_IO_BASE + LW_IO_GAIN * o_wn,
                in_ch=i_ch, in_col=i_col)


# ==========================================
# 2.5 热力图列重排：对称化 → 谱嵌入 → k-means 簇排序 → 簇内 τ 排序 → 相邻交换微调
#     返回 perm：perm[新行号] = 原始柱号（强连接柱在热力图行序上彼此靠近）
# ==========================================
def column_permutation(st, n_clusters=HEAT_CLUSTERS, seed=HEAT_SEED,
                       passes=HEAT_LOCAL_PASSES):
    N, K = int(st['N']), int(st['K'])
    ri = st['rec_idx'].long().numpy()
    rw = st['rec_w'].float().abs().numpy()
    Wd = np.zeros((N, N))
    Wd[np.arange(N)[:, None], ri] = st['rec_w'].float().numpy()   # 带符号权重
    A = np.abs(Wd)
    A = (A + A.T) * 0.5                                   # |W| 对称化（排列目标用）

    # ---- 谱聚类亲和：行归一化符号输入轮廓的余弦相似度 ----
    # （|W| 对称图密度 K/N≈37.5% 近似完全图，无社区结构；
    #   行轮廓相似度才反映"哪些柱输入模式相似"，分块才有意义）
    nrm = np.linalg.norm(Wd, axis=1, keepdims=True) + 1e-12
    Wn = Wd / nrm
    Aff = np.clip(Wn @ Wn.T, 0.0, None)
    np.fill_diagonal(Aff, 0.0)
    d = Aff.sum(1) + 1e-12
    Dm12 = 1.0 / np.sqrt(d)
    L = -Aff * np.outer(Dm12, Dm12)                          # 归一化拉普拉斯
    L[np.arange(N), np.arange(N)] += 1.0
    w, V = np.linalg.eigh(L)
    m = max(2, min(n_clusters, N - 2))
    emb = V[:, 1:1 + m]                                    # 谱嵌入（最小非平凡向量）

    # k-means（确定性：最远点初始化）
    rng = np.random.default_rng(seed)
    centers = [int(np.argmax(d))]                          # 从度最大柱起步
    dist = ((emb - emb[centers[0]]) ** 2).sum(1)
    for _ in range(m - 1):
        centers.append(int(np.argmax(dist)))
        dist = np.minimum(dist, ((emb - emb[centers[-1]]) ** 2).sum(1))
    cent = emb[centers]
    for _ in range(200):
        labels = np.argmin(((emb[:, None, :] - cent[None]) ** 2).sum(-1), 1)
        newc = np.array([emb[labels == c].mean(0) if (labels == c).any()
                         else cent[c] for c in range(m)])
        if np.allclose(newc, cent, atol=1e-10):
            break
        cent = newc
    labels = np.argmin(((emb[:, None, :] - cent[None]) ** 2).sum(-1), 1)

    # 簇排序：按簇中心的第一谱坐标（Fiedler）全局定序
    order_c = sorted(range(m), key=lambda c: emb[labels == c, 0].mean())
    tau = np.asarray(st['tau_e_init'].float()).ravel()
    perm = []
    bounds = []
    for c in order_c:
        members = sorted((i for i in range(N) if labels[i] == c),
                         key=lambda i: float(tau[i]))      # 簇内 τ 排序
        bounds.append((len(perm), len(perm) + len(members)))
        perm.extend(members)
    perm = np.array(perm, dtype=np.int64)

    # 局部搜索微调：相邻交换，最小化 Σ w·|行距|（线性排列目标）
    pos = np.empty(N, dtype=np.int64)
    pos[perm] = np.arange(N)
    ii, jj = np.nonzero(np.triu(A, 1))
    ww = A[ii, jj]
    inc = [[] for _ in range(N)]
    for i, j, w in zip(ii.tolist(), jj.tolist(), ww.tolist()):
        inc[i].append((j, w)); inc[j].append((i, w))
    for _ in range(passes):
        improved = False
        for r in range(N - 1):
            a, b = int(perm[r]), int(perm[r + 1])
            delta = 0.0
            for nb, w in inc[a]:
                pn = pos[nb]
                if pn == r + 1:
                    continue
                delta += w * (abs(r + 1 - pn) - abs(r - pn))
            for nb, w in inc[b]:
                pn = pos[nb]
                if pn == r:
                    continue
                delta += w * (abs(r - pn) - abs(r + 1 - pn))
            if delta < -1e-9:
                perm[r], perm[r + 1] = b, a
                pos[a], pos[b] = r + 1, r
                improved = True
        if not improved:
            break
    h = max(1, N // (2 * m))
    def _band_mass(order):
        pn = np.empty(N, dtype=np.int64)
        pn[order] = np.arange(N)
        de = np.abs(pn[ii] - pn[jj])
        return float((ww * (de <= h)).sum() / ww.sum())
    print(f'[热力图重排] ±{h} 行带内权重占比: 原序 {_band_mass(np.arange(N)):.3f}'
          f' → 新序 {_band_mass(perm):.3f}（{m} 簇，局部搜索后）', flush=True)
    return perm, bounds


# ==========================================
# 3. 渲染器
# ==========================================
class Renderer:
    def __init__(self, layout, frames, window, title, frame_rate,
                 perm=None, wmat=None, cluster_bounds=None):
        self.layout = layout
        self.frames = frames
        self.window = window
        self.K = frame_rate
        self.E_hist = np.stack([f['e'] for f in frames])       # [T,256]，列=思考帧
        self.R_hist = np.stack([f['rec'] for f in frames])     # [T,256] 循环驱动电流（带符号）
        self.wmat = wmat
        self.cluster_bounds = cluster_bounds or []
        self.rec_max = float(np.percentile(np.abs(self.R_hist), 97.0)) or 1.0
        if perm is not None:                                   # 列重排：强连接柱行相邻
            self.E_hist = np.ascontiguousarray(self.E_hist[:, perm])
            self.R_hist = np.ascontiguousarray(self.R_hist[:, perm])
        self.OBS_hist = np.stack([f['obs'] for f in frames])   # [T,32]
        self.PROB_hist = np.stack([f['probs'] for f in frames])  # [T,3]

        self.fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
        gs = GridSpec(2, 2, figure=self.fig,
                      left=0.02, right=0.98, top=0.90, bottom=0.03,
                      width_ratios=[1.0, 1.35], height_ratios=[1.0, 0.62],
                      hspace=0.16, wspace=0.10)
        self.ax_game = self.fig.add_subplot(gs[0, 0])
        self.ax_topo = self.fig.add_subplot(gs[0, 1])
        # 底部三段：E 激活热力图 | τ_eff 实时热力图 | 排序后权重矩阵
        gs_bot = gs[1, :].subgridspec(1, 3, width_ratios=[1.0, 1.0, 0.5],
                                      wspace=0.06)
        self.ax_heat = self.fig.add_subplot(gs_bot[0, 0])
        self.ax_tau = self.fig.add_subplot(gs_bot[0, 1])
        self.ax_mat = self.fig.add_subplot(gs_bot[0, 2])
        for ax in (self.ax_game, self.ax_topo, self.ax_heat,
                   self.ax_tau, self.ax_mat):
            ax.set_facecolor(BG)
            for sp in ax.spines.values():
                sp.set_color(GRID)

        self._setup_game_axes()
        self.fig.canvas.draw()                       # 先量棋盘格实际像素
        bb = self.ax_game.get_window_extent()
        self.cell_px = min(bb.width, bb.height) / BOARD_SPAN
        self.pt2px = self.fig.dpi / 72.0
        # 像素对齐：滚窗 = 热力图面板整宽像素列数 → 1 思考帧 = 1 设备像素
        self._heat_n, self._heat_a, self._heat_w = self._pixel_align(self.ax_heat)
        if self.window != self._heat_n:
            print(f'[像素对齐] 热力图滚窗 {self.window} → {self._heat_n}'
                  f'（面板整宽像素列，1帧=1px）', flush=True)
        self.window = self._heat_n
        self._setup_game()
        self._setup_topo()
        self._setup_heat()
        self._setup_rec()
        self._setup_matrix()
        self._setup_text(title)

    # ---------- 棋盘 ----------
    def _setup_game_axes(self):
        ax = self.ax_game
        ax.set_xlim(-0.6, G - 0.4)
        ax.set_ylim(G - 0.4, -0.6)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])

    def _px(self, frac):
        """格子比例 → matplotlib 尺寸（pt² 面积 / pt 线宽）"""
        return frac * self.cell_px / self.pt2px

    def _setup_game(self):
        ax = self.ax_game
        for g in range(G + 1):
            ax.axhline(g - 0.5, color=GRID, lw=0.6)
            ax.axvline(g - 0.5, color=GRID, lw=0.6)
        ax.set_title('GAME — fixed map 10×10', color=FG, fontsize=13,
                     family='monospace', pad=10, loc='left')
        self.scat_food = ax.scatter([], [], s=self._px(2 * FOOD_R) ** 2, c=FOOD,
                                    marker='o', zorder=5)
        self.scat_body = ax.scatter([], [], s=self._px(SNAKE_FILL) ** 2,
                                    marker='s', zorder=4,
                                    edgecolors=(0, 0, 0, BODY_EDGE_A), linewidths=0.8)
        self.body_line, = ax.plot([], [], color='#E6E6E6', alpha=POLY_ALPHA,
                                  lw=self._px(POLY_LW_FRAC),
                                  solid_joinstyle='round', solid_capstyle='round',
                                  zorder=5)
        self.scat_eye = ax.scatter([], [], s=self._px(2 * EYE_R) ** 2,
                                   facecolors=BG, edgecolors='#CCCCCC',
                                   linewidths=0.9, zorder=6)
        self.txt_score = ax.text(0.02, 0.02, '', transform=ax.transAxes,
                                 color=FG, fontsize=15, family='monospace',
                                 zorder=10,
                                 bbox=dict(facecolor=BG, alpha=0.65,
                                           edgecolor='none', pad=2.0))

    # ---------- 拓扑 ----------
    def _setup_topo(self):
        ax = self.ax_topo
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        N = len(self.layout['col_xy'][0])
        ax.set_title(f'NETWORK — {N} columns · elastic mesh '
                     f'(rec L0=ø · in L0={L0_IN:g} · out L0={L0_OUT:g} · '
                     f'{K_IN_MULT:.0f}×/{K_OUT_MULT:.0f}×)',
                     color=FG, fontsize=13, family='monospace', pad=10, loc='left')
        L = self.layout
        self.rec_base = L['edge_rgba'][:, :3].copy()          # 静默暗色（E/I）
        rec_hi = np.where(L['rec_sign'][:, None] > 0,
                          _hex2rgb(COLOR_REC_E_HI), _hex2rgb(COLOR_REC_I_HI))
        self.rec_hi = rec_hi                                   # 激活亮色（E/I 各自）
        self.out_base_edge = L['out_rgba'][:, :3].copy()
        self.out_hi = np.full((L['out_rgba'].shape[0], 3), _hex2rgb(COLOR_OUT_HI))
        self.lc_rec = ax.add_collection(LineCollection(L['edges'],
                                                       colors=L['edge_rgba'],
                                                       linewidths=L['edge_lws']))
        self.lc_in = ax.add_collection(LineCollection(L['in_segs'],
                                                      colors=L['in_rgba'],
                                                      linewidths=L['in_lws']))
        self.lc_out = ax.add_collection(LineCollection(L['out_segs'],
                                                       colors=L['out_rgba'],
                                                       linewidths=L['out_lws']))
        cx, cy = L['col_xy']
        self.scat_col = ax.scatter(cx, cy, s=26, c=DIM, zorder=4,
                                   edgecolors='#3A3A3A', linewidths=0.5)
        # 输入=组别彩标方块，输出=琥珀方块；亮度实时调制（静默 30% → 满亮）
        self.in_base = np.array([_hex2rgb(OBS32_GROUP_COLORS[ch]) for ch in range(32)])
        self.out_base = _hex2rgb(ACCENT_OUT)
        self.in_edge_rgb = np.array([_hex2rgb(OBS32_GROUP_COLORS[ch])
                                     for ch in L['in_ch']])
        self.scat_in = ax.scatter(L['in_xy'][:, 0], L['in_xy'][:, 1], s=30,
                                  c=self.in_base, marker='s', zorder=5)
        self.scat_out = ax.scatter(L['out_xy'][:, 0], L['out_xy'][:, 1], s=42,
                                   c=self.out_base[None, :], marker='s', zorder=5)
        # 输入 32 通道语义标签（组色 + 编号），右对齐排在输入列左侧
        for ch, (x, y) in enumerate(L['in_xy']):
            ax.text(x - 0.012, y, f'{ch} {OBS32_PROJ_LABELS[ch]}',
                    color=OBS32_GROUP_COLORS[ch], fontsize=LABEL_FS,
                    family='Microsoft YaHei', va='center', ha='right')
        for a, (x, y) in enumerate(L['out_xy']):
            ax.text(x + 0.012, y, ACTION_ZH[a], color=ACCENT_OUT, fontsize=8.5,
                    family='Microsoft YaHei', va='center')
        # 同心双环：白圈常驻（跟随当前/最新动作），蓝圈仅 act 帧叠加且更大
        self.ring_w = ax.scatter([], [], s=100, facecolors='none',
                                 edgecolors=RING, linewidths=1.3, zorder=6)
        self.ring_b = ax.scatter([], [], s=BLUE_RING_S, facecolors='none',
                                 edgecolors=BLUE_RING, linewidths=2.0, zorder=6)

    # ---------- 热力图 ----------
    @staticmethod
    def _pixel_align(ax):
        """列-像素 1:1 对齐参数：返回 (整宽像素列数, xlim 左端, xlim 跨度)。

        xlim 跨度取轴实测宽（非整数）→ 恰 1 数据列 = 1 设备像素；xlim 左端把
        图像左缘推到整数像素边界（右侧 <1px 露底色，深底主题下不可见）。
        滚动 1 列 = 平移整 1px，最近邻不再跳列/变宽，消除隔帧闪烁与残余跳动。"""
        bb = ax.get_window_extent()
        n = int(bb.width)
        a = bb.x0 - np.ceil(bb.x0)
        return n, a, bb.width

    def _setup_heat(self):
        ax = self.ax_heat
        n, a, w = self._heat_n, self._heat_a, self._heat_w
        self.data = np.full((256, n), np.nan, dtype=np.float32)
        self.im = ax.imshow(self.data, aspect='auto', cmap=CMAP_ACT,
                            vmin=0.0, vmax=1.0, origin='lower',
                            extent=(0, n, 0, 256),
                            interpolation='nearest')
        ax.set_ylim(0, 256)
        ax.set_xlim(a, a + w)
        ax.set_xticks([]); ax.set_yticks([1, 64, 128, 192, 256])
        ax.set_yticklabels(['1', '64', '128', '192', '256'])
        ax.set_ylabel('column', color=FG, fontsize=11, family='monospace')
        ax.set_title('COLUMN ACTIVATIONS E — newest at left',
                     color=FG, fontsize=12.5, family='monospace', pad=8, loc='left')
        for sp in ax.spines.values():
            sp.set_visible(False)
        self.cursor, = ax.plot([0, 0], [0, 256],
                               color=FG, lw=1.2, alpha=0.9)

    # ---------- rec 循环驱动电流热力图（带符号：蓝=净抑制驱动 / 橙=净兴奋驱动） ----------
    def _setup_rec(self):
        ax = self.ax_tau
        n, a, w = self._heat_n, self._heat_a, self._heat_w
        self.rdata = np.full((256, n), np.nan, dtype=np.float32)
        self.im_rec = ax.imshow(self.rdata, aspect='auto', cmap=CMAP_REC,
                                vmin=-self.rec_max, vmax=self.rec_max,
                                origin='lower', extent=(0, n, 0, 256),
                                interpolation='nearest')
        ax.set_ylim(0, 256)
        ax.set_xlim(a, a + w)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title('RECURRENT DRIVE rec — newest at left', color=FG,
                     fontsize=10.5, family='monospace', pad=6, loc='left')
        for sp in ax.spines.values():
            sp.set_visible(False)
        self.rcursor, = ax.plot([0, 0], [0, 256],
                                color=FG, lw=1.0, alpha=0.8)

    # ---------- 排序后连接矩阵（E 红 / I 蓝，簇边界细线） ----------
    def _setup_matrix(self):
        ax = self.ax_mat
        ax.set_xticks([]); ax.set_yticks([])
        if self.wmat is None:
            ax.set_title('SORTED CONNECTIVITY — n/a', color=FG, fontsize=11,
                         family='monospace', pad=6, loc='left')
            return
        wmax = float(np.percentile(np.abs(self.wmat), 97.0)) or 1.0
        ax.imshow(self.wmat, cmap=CMAP_W, vmin=-wmax, vmax=wmax,
                  origin='lower', interpolation='nearest')
        for lo, hi in self.cluster_bounds:
            if 0 < hi < self.wmat.shape[0]:
                ax.axhline(hi - 0.5, color=FG, lw=0.45, alpha=0.28)
                ax.axvline(hi - 0.5, color=FG, lw=0.45, alpha=0.28)
        ax.set_title('SORTED |W| — E red / I blue',
                     color=FG, fontsize=10.5, family='monospace', pad=6, loc='left')

    def _setup_text(self, title):
        # 标题可能含中文（模型名/说明），用雅黑字体栈避免豆腐块
        self.fig.text(0.02, 0.955, title,
                      color=FG, fontsize=14, family='Microsoft YaHei')

    # ---------- 每帧更新 ----------
    def draw(self, fi, fi_game=None):
        # fi = 思考帧索引（脑面板/热力图/动作）；fi_game = 棋盘帧索引（可独立 pacing）
        fb = self.frames[min(fi, len(self.frames) - 1)]
        fr = self.frames[min(fi_game if fi_game is not None else fi,
                             len(self.frames) - 1)]
        # --- 棋盘：渐变满格方块 + 折线 + 前偏眼睛 ---
        body = fr['body']
        n = len(body)
        t = np.arange(n) / max(1, n - 1)
        cols = ((1 - t)[:, None] * np.array(SNAKE_C_HEAD)
                + t[:, None] * np.array(SNAKE_C_TAIL))
        self.scat_body.set_offsets(np.stack([body[:, 1], body[:, 0]], 1))
        self.scat_body.set_facecolors(np.hstack([cols, np.ones((n, 1))]))
        if n > 1:
            self.body_line.set_data(body[:, 1], body[:, 0])
        else:
            self.body_line.set_data([], [])
        self.scat_food.set_offsets([fr['food'][1], fr['food'][0]])
        fwd = np.array([DIRS[fr['dir']][1], DIRS[fr['dir']][0]], float)  # (x,y)
        perp = np.array([-fwd[1], fwd[0]])
        c = np.array([fr['head'][1], fr['head'][0]], float)
        eyes = np.stack([c + fwd * EYE_FWD + perp * EYE_SIDE,
                         c + fwd * EYE_FWD - perp * EYE_SIDE])
        self.scat_eye.set_offsets(eyes)
        self.txt_score.set_text(f'score {fr["score"]:>2}/98 · step {fr["step"]:>5}')
        # --- 拓扑：柱/输入/输出实时亮度（每思考帧）+ 边亮度随源端激活 + 动作蓝圈 ---
        e = fb['e']
        L = self.layout
        self.scat_col.set_facecolors(CMAP_ACT(np.clip(e, 0, 1)))
        v = np.clip(fb['obs'], 0, 1)          # obs 已含 INPUT_DECAY^k 衰减
        v = np.where(np.arange(32) < 8, v, v / 8.0)
        v = np.clip(v, 0, 1)
        self.scat_in.set_facecolors(
            np.clip(self.in_base * (0.30 + 0.70 * v[:, None]), 0, 1))
        p = fb['probs']
        pd = np.clip(p / max(float(p.max()), 1e-9), 0, 1)
        self.scat_out.set_facecolors(
            np.clip(self.out_base[None, :] * (0.30 + 0.70 * pd[:, None]), 0, 1))
        act_c = np.clip(e, 0, 1)              # 源端激活（信号从 src 流向 dst）
        # rec 边：颜色/亮度都随源柱激活——静默=暗基色，激活=亮色+高 alpha
        a_src = act_c[L['rec_src']][:, None]
        w_n = L['edge_wn'][:, None]
        rec_rgb = self.rec_base + (self.rec_hi - self.rec_base) * a_src
        ra = np.clip(EDGE_DIM_BASE + EDGE_DIM_GAIN * w_n
                     + EDGE_ACT_GAIN * w_n * a_src, 0, 1)
        self.lc_rec.set_color(list(map(tuple, np.hstack([rec_rgb, ra]))))
        # 输入边：组色 × 通道值发光（亮度含 |w| 基础）
        if IN_EDGE_FLOW:
            base = 0.10 + 0.55 * L['in_edge_alpha']           # 亮度 ∝ |w|
            rgba = np.hstack([self.in_edge_rgb,
                              (base * (0.25 + 0.75 * v[L['in_ch']]))[:, None]])
            self.lc_in.set_color(list(map(tuple, rgba)))
        # 输出边：颜色/亮度随源柱激活（静默暗琥珀 → 激活亮琥珀）
        o_src = act_c[L['out_col']][:, None]
        o_w = L['out_wn'][:, None]
        out_rgb = self.out_base_edge + (self.out_hi - self.out_base_edge) * o_src
        oa = np.clip(EDGE_DIM_BASE + EDGE_DIM_GAIN * o_w
                     + EDGE_ACT_GAIN * o_w * o_src, 0, 1)
        self.lc_out.set_color(list(map(tuple, np.hstack([out_rgb, oa]))))
        # 同心双环：白圈常驻（跟随当前/最新动作），蓝圈仅 act 帧叠加（更大→同心圆）
        node_w = fb['act'] if fb['act'] >= 0 else fb['act_running']
        if node_w >= 0:
            self.ring_w.set_offsets([L['out_xy'][node_w]])
        else:
            self.ring_w.set_offsets([[-1, -1]])
        if fb['act'] >= 0:
            self.ring_b.set_offsets([L['out_xy'][fb['act']]])
        else:
            self.ring_b.set_offsets([[-1, -1]])
        # --- 热力图（滚窗：第 0 列固定为最新，整条从左向右流走）---
        hi = min(fi + 1, self.E_hist.shape[0])
        lo = max(0, hi - self.window)
        win = self.E_hist[lo:hi][::-1]               # 反转：列 0 = 最新活动
        self.data[:, :win.shape[0]] = win.T
        self.data[:, win.shape[0]:] = np.nan
        self.im.set_data(self.data)
        # --- rec 循环驱动面板（同滚窗同向：列 0 = 最新）---
        hi_r = min(fi + 1, self.R_hist.shape[0])
        lo_r = max(0, hi_r - self.window)
        rwin = self.R_hist[lo_r:hi_r][::-1]
        self.rdata[:, :rwin.shape[0]] = rwin.T
        self.rdata[:, rwin.shape[0]:] = np.nan
        self.im_rec.set_data(self.rdata)

    def rgb(self):
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())
        return buf[:, :, :3].copy()


# ==========================================
# 4. 编码
# ==========================================
def pick_writer(out_path, fps, W, H):
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    if not vw.isOpened():
        raise RuntimeError('cv2.VideoWriter 打开失败')
    # 入参 RGB，此处唯一一次反转为 BGR 交给 cv2
    return lambda rgb: vw.write(rgb[:, :, ::-1]), vw.release


# ==========================================
# 主流程
# ==========================================
def main():
    ap = argparse.ArgumentParser(description='win 模型通关过程视频化 v3')
    ap.add_argument('--model', default='artifacts/test16c_cheat7b/16c_cheat7b_win_model.pth')
    ap.add_argument('--out', default='win_run_1080p_v3.mp4')
    ap.add_argument('--min-food', type=int, default=98,
                    help='门禁：回放最终食物数低于该值拒绝渲染（高效版传 97）')
    ap.add_argument('--fps', type=int, default=FPS)
    ap.add_argument('--frames-per-brain', type=int, default=None,
                    help='每个思考帧（神经步）占的视频帧数（默认取 FRAMES_PER_BRAIN）')
    ap.add_argument('--frames-per-game', type=int, default=None,
                    help='每个游戏步占的视频帧数（默认取 FRAMES_PER_GAME）')
    ap.add_argument('--skip', type=int, default=1, help='每 N 视频帧渲一帧（紧凑版用）')
    ap.add_argument('--window', type=int, default=HEAT_WINDOW,
                    help='热力图滚窗（思考帧数；实际钳到面板整宽像素列做 1帧=1px 对齐）')
    ap.add_argument('--max-steps', type=int, default=8000)
    ap.add_argument('--hold', type=int, default=HOLD, help='末帧定格帧数')
    ap.add_argument('--preview', action='store_true', help='只渲首/中/末样张')
    ap.add_argument('--sample-step', type=int, default=None,
                    help='preview 时额外出一张该游戏步的示例帧')
    ap.add_argument('--e0', choices=('zero', 'random'), default=None,
                    help='初始神经元强度：默认出片=zero（真实轨迹）、preview=random（点亮看图）')
    args = ap.parse_args()

    torch.manual_seed(0)
    data = torch.load(args.model, map_location='cpu', weights_only=False)
    cfg = sim.Config()
    for k, v in data.get('config', {}).items():
        if not k.startswith('__'):
            try:
                setattr(cfg, k, v)
            except Exception:
                pass
    st = data['brain']
    model_tag = os.path.basename(args.model)
    print(f"模型: {model_tag} food={data.get('food')} steps={data.get('steps')} "
          f"N={st['N']} K={st['K']}", flush=True)

    fpb = args.frames_per_brain or FRAMES_PER_BRAIN   # 每神经步视频帧数
    fpg = args.frames_per_game or FRAMES_PER_GAME     # 每游戏步视频帧数
    k_rate = int(cfg.FRAME_RATE)                      # 每游戏步的思维帧数
    print('回放采集...', flush=True)
    e0 = args.e0 or (E0_INIT_PREVIEW if args.preview else E0_INIT)
    frames, won, n_steps = rollout(cfg, st, args.max_steps, e0_init=e0)
    final_food = frames[-1]['score']
    print(f"回放完成: {n_steps} 步 × {cfg.FRAME_RATE} 思考帧 | {len(frames)} 帧 | "
          f"won={won} | final score={final_food}/{cfg.TARGET_FOOD} | e0={e0}",
          flush=True)
    if final_food < args.min_food:
        raise SystemExit(f'未达门禁（final={final_food} < min-food={args.min_food}），拒绝渲染')

    layout = precompute_layout(st)
    print('弹性布局收敛完成', flush=True)
    title = (f'16c-cheat-7b — WIN RUN · {model_tag} · '
             f'fixed map seed {cfg.MAP_SEED} · {final_food}/{cfg.TARGET_FOOD}')
    perm, bounds = (column_permutation(st) if HEAT_CLUSTER
                    else (None, []))
    # 排序后连接矩阵（行/列同 perm；带符号：正=兴奋红，负=抑制蓝）
    nsz = int(st['N'])
    W = np.zeros((nsz, nsz))
    W[np.arange(nsz)[:, None],
      st['rec_idx'].long().numpy()] = st['rec_w'].float().numpy()
    if perm is not None:
        W = W[np.ix_(perm, perm)]
    renderer = Renderer(layout, frames, args.window, title, int(cfg.FRAME_RATE),
                        perm=perm, wmat=W, cluster_bounds=bounds)

    if args.preview:
        os.makedirs('preview', exist_ok=True)
        for tag, fi in (('first', 0), ('mid', len(frames) // 2),
                        ('late', int(len(frames) * 0.93)), ('last', len(frames) - 1)):
            renderer.draw(fi)
            pth = f'preview/win_frame_{tag}.png'
            renderer.fig.savefig(pth, facecolor=BG)
            print('样张:', pth, flush=True)
        if args.sample_step:
            cand = next((i for i, f in enumerate(frames)
                         if f['step'] >= args.sample_step and f['act'] >= 0), None)
            if cand is None:
                print(f'--sample-step {args.sample_step}: 未到达该步', flush=True)
            else:
                renderer.draw(cand, cand)
                pth = f'preview/sample_step{args.sample_step}.png'
                renderer.fig.savefig(pth, facecolor=BG)
                print('示例帧:', pth, flush=True)
        return

    W, H = 1920, 1080
    write, release = pick_writer(args.out, args.fps, W, H)
    content = (len(frames) - 1) * fpb + 1
    total = content + args.hold
    print(f'渲染 {total} 帧 @ {args.fps}fps -> {args.out} '
          f'(每神经步{fpb}帧 / 每游戏步{fpg}帧 · '
          f'时长 {total / args.fps / 60:.1f} 分钟)', flush=True)
    t0 = time.time()
    last_key = None
    last_rgb = None
    n_draw = 0
    for t in range(0, content, args.skip):
        fi = min(t // fpb, len(frames) - 1)      # 思考帧（脑面板/热力图）
        gi = min((t // fpg) * k_rate, len(frames) - 1)   # 游戏步：每 fpg 帧推进一整步
        if (fi, gi) != last_key:                 # 内容未变的视频帧复用上一画面
            renderer.draw(fi, gi)
            last_rgb = renderer.rgb()
            last_key = (fi, gi)
            n_draw += 1
        write(last_rgb)
        if n_draw % 500 == 0 and n_draw > 0:
            el = time.time() - t0
            print(f'  视频帧 {t + 1}/{total} (实际绘制 {n_draw}, {el:.0f}s, '
                  f'ETA {el / n_draw * (total // args.skip - n_draw) / 60:.0f}min)',
                  flush=True)
    for _ in range(args.hold):
        write(last_rgb)
    release()
    print(f'完成: {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)', flush=True)
    # H.264 转码（系统 ffmpeg 可用时）；成功即删 mp4v 原稿省空间
    import shutil
    if shutil.which('ffmpeg'):
        h264 = args.out.replace('.mp4', '_h264.mp4')
        os.system(f'ffmpeg -y -i "{args.out}" -c:v libx264 -pix_fmt yuv420p '
                  f'-crf 20 "{h264}" -loglevel error')
        if os.path.exists(h264) and os.path.getsize(h264) > 1e6:
            os.remove(args.out)
            print(f'H.264 版本: {h264}（mp4v 原稿已删）', flush=True)
        else:
            print('H.264 转码失败，保留 mp4v 原稿', flush=True)
    else:
        print('（未找到 ffmpeg，保留 mp4v 原稿）', flush=True)


if __name__ == '__main__':
    main()
