# ==========================================
# make_intro_video.py —— 前置介绍动画（60s · 1080p60 · 与结果视频同一视觉语言）
#
# 四幕（无片头片尾）：
#   1. 0-12s    贪吃蛇环境与 32 路输入语义（棋盘居中，头4/尾4/食8/身8/障8 依次点亮
#               + 蛇真实右转走一步，观测帧随头旋转平移）
#   2. 12-24s   最简网络原理：4×4 皮质柱 3入2出（严格按原网络动力学，无自环）
#               状态值/权重值/兴奋抑制性质 + 输入切换平滑迁移 → 不同输出胜出
#   3. 24-36s   完整 256/32/3 网络：16×16 方阵初始 + 随机权重 → 弹簧迭代塑性
#               （逐迭代连续采样，无跳变）
#   4. 36-60s   EA：种群 4096 → 两阶段筛选(12局→2048；6局→1024→18局) → 精英保留
#               → 交叉-变异 → 新一代；适应度公式 + 双训练过程曲线（最佳/精英/平均三线）
#
# 弹簧力学与结果视频拓扑面板同源（常数 import 自 render_win_video），初始位置 16×16 正方形。
# 曲线数据 test16b_simp_history.json / 16c_cheat7b_history.json。
# 用法：
#   python make_intro_video.py --stills          # 关键帧 PNG（preview/intro_still_*.png）
#   python make_intro_video.py                   # 全片 intro_1080p.mp4 → ffmpeg 转 intro_1080p_h264.mp4
# ==========================================
import argparse
import json
import os
import shutil
import time

import numpy as np

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe

import render_win_video as rwv

# ---- 风格（全部沿用结果视频） ----
BG, FG, GRID, DIM = rwv.BG, rwv.FG, rwv.GRID, rwv.DIM
GRP_HEAD, GRP_TAIL = rwv.GRP_HEAD, rwv.GRP_TAIL
GRP_FOOD, GRP_BODY, GRP_WALL = rwv.GRP_FOOD, rwv.GRP_BODY, rwv.GRP_WALL
C_E_DIM, C_I_DIM = rwv.COLOR_REC_E, rwv.COLOR_REC_I
C_E_HI, C_I_HI, C_OUT_HI = rwv.COLOR_REC_E_HI, rwv.COLOR_REC_I_HI, rwv.COLOR_OUT_HI
ACCENT_IN, ACCENT_OUT = rwv.ACCENT_IN, rwv.ACCENT_OUT
FOOD_C = rwv.FOOD
CMAP_ACT = rwv.CMAP_ACT
YH = 'Microsoft YaHei'
MONO = ['monospace', 'Microsoft YaHei']

def _hex2rgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])

# ---- 时间轴（秒；总长 60，缩减只改这里） ----
FPS = 60
TOTAL = 60.0
T_TURN0 = 8.8                      # 转向走步起点
T_MOVE = (8.8, 9.3)                # 蛇移动动画窗（加速：0.5s）
T2A = (12.0, 24.0)                 # 最简网络原理
T2B = (24.0, 36.0)                 # 完整网络
T_SPRING = (26.5, 31.8)            # 弹簧塑性
T3 = (36.0, 60.0)                  # EA
T_CURVES = (53.2, 60.0)            # 曲线

# ---- 缓动 ----
def clamp01(x):
    return np.clip(x, 0.0, 1.0)

def seg(t, a, b):
    return clamp01((t - a) / (b - a))

def smooth(p):
    p = clamp01(p)
    return p * p * (3.0 - 2.0 * p)

def smoother(p):
    p = clamp01(p)
    return p * p * p * (p * (6.0 * p - 15.0) + 10.0)

def lerp(a, b, p):
    return a + (b - a) * p

def win(t, a, b, fin=0.15, fout=0.15):
    if t < a or t > b:
        return 0.0
    r = 1.0
    if fin > 0:
        r = min(r, (t - a) / fin)
    if fout > 0:
        r = min(r, (b - t) / fout)
    return r

# ---- 像素排版 ----
def FS(px):
    return px * 72.0 / 100.0

def LW(px):
    return px * 72.0 / 100.0

_SCRATCH = None
_MW = {}

def measure_px(s, fs_px):
    key = (s, round(fs_px, 2))
    if key in _MW:
        return _MW[key]
    global _SCRATCH
    if _SCRATCH is None:
        _SCRATCH = plt.figure(figsize=(19.2, 10.8), dpi=100)
        _SCRATCH.canvas.draw()
    t = _SCRATCH.text(0.5, 0.5, s, fontsize=FS(fs_px), family=YH)
    _SCRATCH.canvas.draw()
    w = float(t.get_window_extent().width)
    t.remove()
    _MW[key] = w
    return w

def draw_segments(ax, x, y, segs, fs_px, anchor='left'):
    widths = [measure_px(s, fs_px) for s, _ in segs]
    cx = x - sum(widths) / 2.0 if anchor == 'center' else x
    for (s, c), w in zip(segs, widths):
        ax.text(cx, y, s, color=c, fontsize=FS(fs_px), family=YH,
                ha='left', va='baseline')
        cx += w
    return sum(widths)

# ==========================================
# 场景 1：棋盘与输入语义
# ==========================================
G = rwv.G
CELL = 72.0
BX0, BY1 = 600.0, 900.0
SNAKE = [(5, 5), (5, 4), (4, 4), (4, 3), (3, 3), (3, 2)]       # 转向前
SNAKE_NEW = [(6, 5)] + SNAKE[:-1]                               # 右转走一步后
FOOD = (8, 7)

def bpx(cell):
    r, c = cell
    return BX0 + (c + 0.5) * CELL, BY1 - (r + 0.5) * CELL

def dir_index(vec):
    return rwv.DIRS.index(tuple(vec))

def d8_table(dir_idx):
    d = rwv.DIRS[dir_idx]
    left = rwv.DIRS[(dir_idx + 3) % 4]
    right = rwv.DIRS[(dir_idx + 1) % 4]
    add = lambda a, b: (a[0] + b[0], a[1] + b[1])
    neg = lambda a: (-a[0], -a[1])
    return [d, add(d, left), left, (left[0] - d[0], left[1] - d[1]),
            neg(d), (right[0] - d[0], right[1] - d[1]), right, add(d, right)]

def ray_hit_body(head, body_set, vec):
    for k in range(1, G + 1):
        c = (head[0] + vec[0] * k, head[1] + vec[1] * k)
        if c in body_set:
            return k, c
    return None

def ray_hit_wall(head, body_set, vec):
    last = None
    for k in range(1, G + 2):
        c = (head[0] + vec[0] * k, head[1] + vec[1] * k)
        if not (0 <= c[0] < G and 0 <= c[1] < G) or (c in body_set):
            return k, last
        last = c
    return G, last

def rot_px(p, center, deg):
    a = np.deg2rad(deg)
    x, y = p[0] - center[0], p[1] - center[1]
    return (center[0] + x * np.cos(a) - y * np.sin(a),
            center[1] + x * np.sin(a) + y * np.cos(a))

RAY_Z = 13.0        # 射线/源点/标签恒在蛇身之上（蛇身方块最高 10、眼睛 11）
RAY_EDGE = '#ADADAD'  # 射线淡灰描边（不刺眼）

def _dimc(color, bright):
    """颜色亮度调制：暗态 = 不透明的深色（混向黑），而非透明。"""
    return tuple(np.clip(np.array(_hex2rgb(color)) * bright, 0, 1))

def draw_ray(ax, x0, y0, ux, uy, ln, color, alpha, lw,
             head_len=28.0, head_w=18.0, bright=1.0):
    """统一形状：大头三角直线（直线 + 大三角箭头，带细白描边），画在蛇身之上。
    激活与否只改 bright（颜色亮度，不透明）；alpha 仅用于出现/消失过渡。"""
    ln = max(ln, head_len + 4.0)
    col = _dimc(color, bright)
    x1, y1 = x0 + ux * ln, y0 + uy * ln
    bx, by = x1 - ux * head_len, y1 - uy * head_len
    px_, py_ = -uy, ux
    tri = ([x1, bx + px_ * head_w, bx - px_ * head_w],
           [y1, by + py_ * head_w, by - py_ * head_w])
    # 细白描边：加粗白色底线 + 放大白三角垫底，本色画在其上
    gx, gy = (x1 + 2.0 * bx) / 3.0, (y1 + 2.0 * by) / 3.0
    sc = 1.12
    ax.plot([x0, x1], [y0, y1], color=RAY_EDGE, lw=lw + 2.6, alpha=alpha,
            solid_capstyle='round', zorder=RAY_Z)
    ax.fill([gx + (vx - gx) * sc for vx in tri[0]],
            [gy + (vy - gy) * sc for vy in tri[1]],
            facecolor=RAY_EDGE, edgecolor='none', alpha=alpha, zorder=RAY_Z)
    ax.plot([x0, bx], [y0, by], color=col, lw=lw, alpha=alpha,
            solid_capstyle='round', zorder=RAY_Z)
    ax.fill(tri[0], tri[1], facecolor=col, edgecolor='none', alpha=alpha,
            zorder=RAY_Z)

def draw_origin(ax, x0, y0, color, alpha, bright=1.0):
    """射线源头：圆标恒画在蛇身之上。"""
    if alpha <= 0:
        return
    ax.add_patch(plt.Circle((x0, y0), 9.0, facecolor=_dimc(color, bright),
                            edgecolor=BG, lw=LW(2.2), alpha=alpha,
                            zorder=RAY_Z + 1))

GROUPS = [
    ('head', 1.5, 2.5, 4, GRP_HEAD, '头·4', '朝向 one-hot'),
    ('tail', 2.5, 3.4, 8, GRP_TAIL, '尾·4', '尾方向 one-hot'),
    ('food', 3.4, 5.2, 16, GRP_FOOD, '食·8', '食物投影感受野 · 不被遮挡'),
    ('body', 5.2, 7.0, 24, GRP_BODY, '身·8', '自体射线 · 值 = 8/距离'),
    ('wall', 7.0, 8.8, 32, GRP_WALL, '障·8', '障碍射线（墙∪身） · 28号通道 = √蛇长'),
]

def draw_board(ax, alpha, body, dir_deg=0.0):
    """body: [(row,col)...] 可为浮点插值位置。"""
    if alpha <= 0:
        return
    for g in range(G + 1):
        ax.plot([BX0, BX0 + G * CELL], [BY1 - g * CELL] * 2,
                color=GRID, lw=LW(0.9), alpha=alpha, zorder=1)
        ax.plot([BX0 + g * CELL] * 2, [BY1 - G * CELL, BY1],
                color=GRID, lw=LW(0.9), alpha=alpha, zorder=1)
    ax.plot([BX0, BX0 + G * CELL, BX0 + G * CELL, BX0, BX0],
            [BY1, BY1, BY1 - G * CELL, BY1 - G * CELL, BY1],
            color=FG, lw=LW(1.6), alpha=0.55 * alpha, zorder=2)
    for i, cell in enumerate(body):
        x, y = bpx(cell)
        f = i / max(len(body) - 1, 1)
        col = tuple(np.array(rwv.SNAKE_C_HEAD) * (1 - f)
                    + np.array(rwv.SNAKE_C_TAIL) * f)
        ax.fill([x - .44 * CELL, x + .44 * CELL, x + .44 * CELL, x - .44 * CELL],
                [y - .44 * CELL, y - .44 * CELL, y + .44 * CELL, y + .44 * CELL],
                facecolor=col, edgecolor=(0, 0, 0, rwv.BODY_EDGE_A * alpha),
                lw=LW(1.1), zorder=4 + (len(body) - i))   # 头最高，逐节降低
    hx, hy = bpx(body[0])
    th = np.deg2rad(dir_deg)
    fx, fy = np.cos(th), np.sin(th)
    px_, py_ = -fy, fx
    for sgn in (-1, 1):
        cx = hx + fx * .19 * CELL + sgn * px_ * .17 * CELL
        cy = hy + fy * .19 * CELL + sgn * py_ * .17 * CELL
        ax.fill([cx - .09 * CELL, cx + .09 * CELL, cx + .09 * CELL, cx - .09 * CELL],
                [cy - .09 * CELL, cy - .09 * CELL, cy + .09 * CELL, cy + .09 * CELL],
                facecolor='#CCCCCC', edgecolor='none', alpha=alpha,
                zorder=4 + len(body) + 1)
    fx_, fy_ = bpx(FOOD)
    ax.add_patch(plt.Circle((fx_, fy_), rwv.FOOD_R * CELL, facecolor=FOOD_C,
                            edgecolor='none', alpha=alpha, zorder=5))

def draw_dir_arrows(ax, cell, dir_active, color, alpha, deg=0.0, bright=1.0):
    """one-hot 指示：只画激活方向的一支箭头（不透明）。"""
    if alpha <= 0:
        return
    cx, cy = bpx(cell)
    a = np.deg2rad(deg)
    d = rwv.DIRS[dir_active]
    sx, sy = d[1], -d[0]
    ux = sx * np.cos(a) - sy * np.sin(a)
    uy = sx * np.sin(a) + sy * np.cos(a)
    n = np.hypot(ux, uy)
    draw_ray(ax, cx, cy, ux / n, uy / n, 1.62 * CELL, color, alpha,
             LW(5.0), bright=bright)
    draw_origin(ax, cx, cy, color, alpha, bright)

def draw_food_field(ax, head, dir_idx, alpha, deg=0.0, labels=True, bright=1.0):
    """连续数值：长度 = 投影值（亮度辅助强化）；8 条射线全部显示、不透明。"""
    if alpha <= 0:
        return
    v = np.array(FOOD, float) - np.array(head, float)
    vv = max(float(v @ v), 1.0)
    cx, cy = bpx(head)
    a = np.deg2rad(deg)
    for vec in d8_table(dir_idx):
        u = np.array(vec, float)
        u /= np.linalg.norm(u)
        val = 8.0 * max(0.0, float(v @ u)) / vv
        sx, sy = vec[1], -vec[0]
        n = np.hypot(sx, sy)
        ux = (sx / n) * np.cos(a) - (sy / n) * np.sin(a)
        uy = (sx / n) * np.sin(a) + (sy / n) * np.cos(a)
        ln = 0.45 * CELL + min(val / 8.0, 1.0) * 3.0 * CELL
        draw_ray(ax, cx, cy, ux, uy, ln, GRP_FOOD, 0.7 * alpha, LW(5.0),
                 bright=(0.35 + 0.65 * val / 8.0) * bright)
        if labels and val > 0.05:
            ax.text(cx + ux * (ln + 22), cy + uy * (ln + 22), f'{val:.2f}',
                    color=_dimc(GRP_FOOD, bright), fontsize=FS(15), family=MONO,
                    ha='center', va='center', alpha=0.9 * alpha, zorder=RAY_Z)
    draw_origin(ax, cx, cy, GRP_FOOD, alpha, bright)

def draw_sector_rays(ax, head, body_set, dir_idx, alpha, deg, color,
                     hit_body, labels, bright=1.0):
    """8 条射线全部完整显示、不透明：有命中 → 全亮直线到命中格（长度=距离，
    值 8/距离）；无命中 → 暗色（非透明）直线到棋盘边缘。源头恒在蛇身之上。"""
    if alpha <= 0:
        return
    cx, cy = bpx(head)
    a = np.deg2rad(deg)
    for vec in d8_table(dir_idx):
        sx, sy = vec[1], -vec[0]
        n = np.hypot(sx, sy)
        ux0, uy0 = sx / n, sy / n
        ux = ux0 * np.cos(a) - uy0 * np.sin(a)
        uy = ux0 * np.sin(a) + uy0 * np.cos(a)
        hit = (ray_hit_body if hit_body else ray_hit_wall)(head, body_set, vec)
        strong = hit is not None and hit[1] is not None
        if strong:
            end = hit[1]
        elif hit is not None:
            # 紧邻格即被挡（墙∪身 k=1 时无前格）：停在阻挡格上
            nc = (head[0] + vec[0] * hit[0], head[1] + vec[1] * hit[0])
            if 0 <= nc[0] < G and 0 <= nc[1] < G:
                end, strong = nc, True
            else:
                end = nc
        else:
            kmax = 0
            while kmax < G:
                nc = (head[0] + vec[0] * (kmax + 1),
                      head[1] + vec[1] * (kmax + 1))
                if not (0 <= nc[0] < G and 0 <= nc[1] < G):
                    break
                kmax += 1
            if kmax > 0:
                end = (head[0] + vec[0] * kmax, head[1] + vec[1] * kmax)
            else:
                nc = (head[0] + vec[0], head[1] + vec[1])
                if not (0 <= nc[0] < G and 0 <= nc[1] < G):
                    continue
                end = nc
        tx, ty = bpx(end)
        if deg != 0.0:
            tx, ty = rot_px((tx, ty), (cx, cy), deg)
        draw_ray(ax, cx, cy, ux, uy, np.hypot(tx - cx, ty - cy), color,
                 0.7 * alpha, LW(5.0),
                 bright=bright if strong else 0.35 * bright)
        if labels and strong:
            ax.text(tx + ux * 40, ty + uy * 40, f'{8.0 / hit[0]:.2f}',
                    color=_dimc(color, bright), fontsize=FS(15), family=MONO,
                    ha='center', va='center', alpha=0.9 * alpha, zorder=RAY_Z)
    draw_origin(ax, cx, cy, color, alpha, bright)

def draw_group(ax, key, fade, body, dir_idx=0, deg=0.0, labels=True, bright=1.0):
    head, body_set = body[0], set(body[1:])
    if key == 'head':
        draw_dir_arrows(ax, head, dir_idx, GRP_HEAD, fade, deg, bright)
    elif key == 'tail':
        ext = (body[-2][0] - body[-1][0], body[-2][1] - body[-1][1])
        draw_dir_arrows(ax, body[-1], dir_index(ext), GRP_TAIL, fade, 0.0, bright)
    elif key == 'food':
        draw_food_field(ax, head, dir_idx, fade, deg=deg, labels=labels,
                        bright=bright)
    elif key == 'body':
        draw_sector_rays(ax, head, body_set, dir_idx, fade, deg, GRP_BODY,
                         True, labels, bright)
    elif key == 'wall':
        draw_sector_rays(ax, head, body_set, dir_idx, fade, deg, GRP_WALL,
                         False, labels, bright)

def obs32_values(body, dir_idx):
    """与画面同一套几何算出的 32 通道实时值（head/tail/food/body/wall 各一组）。"""
    head, body_set = body[0], set(body[1:])
    hv = [0.0] * 4
    hv[dir_idx] = 1.0
    ext = (body[-2][0] - body[-1][0], body[-2][1] - body[-1][1])
    tv = [0.0] * 4
    tv[dir_index(ext)] = 1.0
    v = np.array(FOOD, float) - np.array(head, float)
    vv = max(float(v @ v), 1.0)
    fv, bv, wv = [], [], []
    for vec in d8_table(dir_idx):
        u = np.array(vec, float)
        u /= np.linalg.norm(u)
        fv.append(8.0 * max(0.0, float(v @ u)) / vv)
        bh = ray_hit_body(head, body_set, vec)
        bv.append(8.0 / bh[0] if bh else 0.0)
        wk = ray_hit_wall(head, body_set, vec)
        wv.append(8.0 / wk[0])
    return dict(head=hv, tail=tv, food=fv, body=bv, wall=wv)

PANEL_ROWS = (('蛇头方向', 'head'), ('蛇尾方向', 'tail'), ('食物投影', 'food'),
              ('自体射线', 'body'), ('障碍射线', 'wall'))
PANEL_GCOL = dict(enumerate((GRP_HEAD, GRP_TAIL, GRP_FOOD, GRP_BODY, GRP_WALL)))
PANEL_X, PANEL_Y0, PANEL_DY = 1360.0, 836.0, 80.0   # ax 的 y 轴向上：首行在最上

def draw_obs_panel(ax, t, cur):
    """右侧逐条同步值面板：行随分组出现；浮点一律两位小数，等宽工整，不高亮。"""
    ha = smooth(seg(t, GROUPS[0][1], GROUPS[0][1] + 0.4))
    if ha <= 0:
        return
    ax.text(PANEL_X, 944, 'OBS 32 · 实时值', color=FG, fontsize=FS(21),
            family=YH, ha='left', va='center', alpha=ha, zorder=10)
    ax.plot([PANEL_X, 1880], [892, 892], color=GRID, lw=LW(1.2),
            alpha=ha, zorder=9)
    for i, (name, key) in enumerate(PANEL_ROWS):
        a = smooth(seg(t, GROUPS[i][1], GROUPS[i][1] + 0.35))
        if a <= 0:
            continue
        y = PANEL_Y0 - i * PANEL_DY
        ax.text(PANEL_X, y, name + '：', color=PANEL_GCOL[i], fontsize=FS(19),
                family=YH, ha='left', va='center', alpha=a, zorder=10)
        if key in ('head', 'tail'):          # one-hot 行整数，连续行一律两位小数
            s = ' '.join(f'{v:.0f}' for v in cur[key])
        else:
            s = ' '.join(f'{v:.2f}' for v in cur[key])
        ax.text(PANEL_X + 108, y, '[' + s + ']', color=FG,
                fontsize=FS(16), family=MONO,
                ha='left', va='center', alpha=0.88 * a, zorder=10)

def scene1(ax, t):
    moving = seg(t, *T_MOVE)                     # 走步插值 0→1
    cells = [tuple(np.array(o) * (1 - smoother(moving))
                   + np.array(n) * smoother(moving))
             for o, n in zip(SNAKE, SNAKE_NEW)]
    deg = -90.0 * smoother(moving) if t >= T_MOVE[0] else 0.0
    draw_board(ax, smooth(seg(t, 0.0, 0.6)), cells, dir_deg=deg)
    if t > 0.5:
        draw_segments(ax, 960, 118,
                      [('10×10 棋盘', FG), ('  ·  ', DIM),
                       ('动作：直行 / 左转 / 右转', FG), ('  ·  ', DIM),
                       ('1 环境步 = 5 思考帧', FG)], 22, anchor='center')
    if t < T_TURN0:
        for key, t0, t1, cnt, gcol, gname, gnote in GROUPS:
            if t < t0:
                continue
            if key in ('head', 'tail'):                   # 方向指示常亮
                fade, bright = 1.0, 1.0
            elif t <= t1:                                 # 登场：透明度过渡
                fade, bright = smooth((t - t0) / 0.15), 1.0
            else:                                         # 退场：降亮度不降透明度
                fade = 1.0
                bright = 1.0 - 0.75 * smooth(clamp01((t - t1) / 0.5))
            draw_group(ax, key, fade, SNAKE, dir_idx=0, bright=bright)
            ca = win(t, t0, t1, 0.12, 0.18)
            if ca > 0:
                ax.text(960, 70, f'{gname} ｜ {gnote}', color=gcol,
                        fontsize=FS(24), family=YH, ha='center', va='center',
                        alpha=ca, zorder=10)
    else:
        # 右转走一步：旧观测帧原位淡出（透明度，短暂过渡）；
        # 新射线等走步动画完全结束后才淡入；尾组不随头转
        fade_old = 1.0 - smooth(clamp01(moving / 0.5))
        na = smooth(clamp01((t - T_MOVE[1]) / 0.35))     # 动画完成后新射线才出现
        for key, t0, t1, cnt, gcol, gname, gnote in GROUPS:
            if key == 'tail':
                draw_group(ax, key, fade_old, SNAKE, bright=0.35)
                draw_group(ax, key, na, SNAKE_NEW, bright=1.0)
                continue
            if key == 'head':
                draw_dir_arrows(ax, cells[0], 0, GRP_HEAD, fade_old,
                                deg=0.0, bright=0.35)
                draw_dir_arrows(ax, cells[0], 1, GRP_HEAD, na,
                                deg=0.0, bright=1.0)
                continue
            draw_group(ax, key, fade_old, SNAKE, dir_idx=0, deg=0.0,
                       labels=False, bright=0.35)
            draw_group(ax, key, na, SNAKE_NEW, dir_idx=1,
                       labels=(na >= 0.9))
        if t >= T_MOVE[1] + 0.4:
            ax.text(960, 70, '右转：蛇整体走一步，观测帧随新头位/新朝向刷新',
                    color=FG, fontsize=FS(24), family=YH, ha='center', va='center',
                    alpha=win(t, T_MOVE[1] + 0.4, 12.0, 0.2, 0.0), zorder=10)
    # ---- 右侧 32 通道实时值面板（走步结束后换新观测值） ----
    if t >= T_MOVE[1]:
        draw_obs_panel(ax, t, obs32_values(SNAKE_NEW, 1))
    else:
        draw_obs_panel(ax, t, obs32_values(SNAKE, 0))
    cnt = 0
    for key, t0, t1, c, *_ in GROUPS:
        if t >= t0:
            cnt = c
    ax.text(1868, 1030, f'OBS {cnt}/32', color=FG, fontsize=FS(26), family=MONO,
            ha='right', va='center', zorder=10)

# ==========================================
# 场景 2A：最简网络原理（4×4 皮质柱 · 3入2出 · 严格按原网络动力学）
#   E ← σ( 输入 + Σ w_rec·E[src] + τ·E − w_ei·I )   每柱兴奋态（rec 无自环）
#   I ← σ( w_ie·E )                                  柱内抑制
#   输出 = W_out·E → argmax
# 演示：输入 [1,0,0] → 收敛 → 输出1 胜出；切换 [0,1,0] → 状态平滑迁移 → 输出2 胜出
# ==========================================
DEMO_TAU, DEMO_WEI, DEMO_WIE = 0.55, 0.8, 1.2
DEMO_REC = {
    0: [(13, -0.70), (12, -0.54)],
    1: [(7, +0.20), (10, -0.29)],
    2: [(10, +0.42), (9, -0.80)],
    3: [(8, -0.80), (2, +0.63)],
    4: [(7, -0.80), (11, +0.80)],
    5: [(1, -0.19), (13, +0.16)],
    6: [(12, +0.26), (0, +0.80)],
    7: [(8, -0.80), (10, -0.58)],
    8: [(14, +0.40), (0, +0.42)],
    9: [(1, +0.18), (7, -0.43)],
    10: [(4, +0.15), (5, -0.25)],
    11: [(14, -0.64), (4, -0.29)],
    12: [(7, +0.49), (13, +0.36)],
    13: [(14, +0.75), (9, -0.32)],
    14: [(8, +0.20), (5, -0.59)],
    15: [(14, -0.80), (10, +0.32)],
}
DEMO_WIN = np.zeros((16, 3))          # 每列的输入扇入权重（随机初始化+筛选）
DEMO_WIN[2, 0] = 0.54
DEMO_WIN[3, 0] = 0.76
DEMO_WIN[4, 0] = 0.91
DEMO_WIN[10, 0] = 0.36
DEMO_WIN[1, 1] = 0.34
DEMO_WIN[3, 1] = 0.52
DEMO_WIN[8, 1] = 0.94
DEMO_WIN[15, 1] = 0.63
DEMO_WIN[6, 2] = 0.42
DEMO_WIN[9, 2] = 0.52
DEMO_WIN[12, 2] = 0.61
DEMO_WIN[15, 2] = 0.57
DEMO_WOUT = np.zeros((2, 16))
DEMO_WOUT[0, 4] = 1.23
DEMO_WOUT[0, 5] = -0.59
DEMO_WOUT[0, 10] = 0.53
DEMO_WOUT[1, 0] = -1.15
DEMO_WOUT[1, 7] = 1.13
DEMO_WOUT[1, 8] = 1.16
DEMO_POS = [(850.0 + 150 * c, 774.0 - 128 * r) for r in range(4) for c in range(4)]
DEMO_A_IN = np.array([1.0, 0.0, 0.0])
DEMO_B_IN = np.array([0.0, 1.0, 0.0])
DEMO_IN_Y = (450.0, 575.0, 700.0)     # 输入钉等距
DEMO_NIT = 8                          # 每相位稳态预演迭代数
DEMO_R = 36.0
DEMO_TAU_R = 0.35                     # 状态松弛时间常数（秒，加快）
DEMO_PHASE_T = (13.0, 15.4, 17.8, 20.2)   # 各相位切换起点（0.25s 渐变 + 指数松弛）
DEMO_PHASE_X = (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]),
                np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 1.0]))
DEMO_CAPTIONS = ('输入 [0,0,1]：弱驱动 → 输出 2 微弱领先',
                 '切换 [0,1,0]：输出 2 明确胜出',
                 '切换 [1,0,0]：偏好翻转到输出 1',
                 '同时 [1,0,1]：两路驱动积分 → 输出 1 保持胜出')

def demo_frames():
    """严格按 forward_batch 口径链式预演各相位稳态（每相位 8 次迭代续接）。"""
    E = np.zeros(16)
    I = np.zeros(16)
    states = []
    for x in DEMO_PHASE_X:
        for _ in range(DEMO_NIT):
            ext = DEMO_WIN @ x
            rec = np.array([sum(w * E[s] for s, w in DEMO_REC[d])
                            for d in range(16)])
            En = 1.0 / (1.0 + np.exp(-(ext + rec + DEMO_TAU * E
                                       - DEMO_WEI * I)))
            In = 1.0 / (1.0 + np.exp(-DEMO_WIE * En))
            E, I = En, In
        states.append((E.copy(), I.copy()))
    return states
_DEMO_FRAMES = demo_frames()

def demo_sample(t):
    """t → (E, I, logits, x)。相位序列：输入 0.25s 渐变切换，内部状态以指数
    松弛（τ=DEMO_TAU_R）趋向该相位稳态；logits 由当前 E 实时算出 → 输出随之渐变。"""
    z = np.zeros(16)
    i = -1
    for k_, ts in enumerate(DEMO_PHASE_T):
        if t >= ts:
            i = k_
    if i < 0:
        return z, z, np.zeros(2), np.zeros(3)
    x_cur, st_cur = DEMO_PHASE_X[i], _DEMO_FRAMES[i]
    if i == 0:
        x_prev, st_prev = np.zeros(3), (z, z)
    else:
        x_prev, st_prev = DEMO_PHASE_X[i - 1], _DEMO_FRAMES[i - 1]
    sw = smooth(seg(t, DEMO_PHASE_T[i], DEMO_PHASE_T[i] + 0.25))
    x = x_prev * (1 - sw) + x_cur * sw
    k = np.exp(-max(t - (DEMO_PHASE_T[i] + 0.25), 0.0) / DEMO_TAU_R)
    E = st_prev[0] + (st_cur[0] - st_prev[0]) * (1.0 - k)
    I = st_prev[1] + (st_cur[1] - st_prev[1]) * (1.0 - k)
    lg = DEMO_WOUT @ E
    return E, I, lg, x

def draw_edge(ax, p0, p1, color, bend, lw, alpha, label=None, bright=1.0,
              avoid=None, prog=1.0):
    """曲线连接 + 小三角箭头 + 权重标签（橙=兴奋 蓝=抑制，颜色亮度=源端活动）。
    标签位置自动避开 avoid 柱圆；prog<1 时按生长进度画边（出现过程渐变）。"""
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    L = max(np.hypot(dx, dy), 1.0)
    nx, ny = dy / L, -dx / L
    tt = np.linspace(0.0, max(prog, 0.02), 26)
    bx_ = x0 + dx * tt + nx * (4 * tt * (1 - tt) * bend)
    by_ = y0 + dy * tt + ny * (4 * tt * (1 - tt) * bend)
    col = _dimc(color, bright)
    ax.plot(bx_, by_, color=col, lw=lw, alpha=alpha * smoother(seg(prog, 0.0, 0.3)),
            solid_capstyle='round', zorder=3)
    if prog >= 0.999:
        ux, uy = bx_[-1] - bx_[-2], by_[-1] - by_[-2]
        n = np.hypot(ux, uy)
        ux, uy = ux / n, uy / n
        hx, hy = bx_[-1], by_[-1]
        bxx, byy = hx - ux * 11, hy - uy * 11
        px_, py_ = -uy, ux
        ax.fill([hx, bxx + px_ * 6.5, bxx - px_ * 6.5],
                [hy, byy + py_ * 6.5, byy - py_ * 6.5],
                facecolor=col, edgecolor='none', alpha=alpha, zorder=3)
    if label is not None:
        lx, ly = bx_[13] + nx * 16, by_[13] + ny * 16
        if avoid is not None:
            for off in (16.0, -16.0, 32.0, -32.0, 50.0, -50.0):
                cxl, cyl = bx_[13] + nx * off, by_[13] + ny * off
                if all((cxl - ax_) ** 2 + (cyl - ay_) ** 2 > (DEMO_R + 10) ** 2
                       for ax_, ay_ in avoid):
                    lx, ly = cxl, cyl
                    break
        ax.text(lx, ly, label, color=col,
                fontsize=FS(12.5), family=MONO, ha='center', va='center',
                alpha=0.9 * alpha * smoother(seg(prog, 0.85, 1.0)), zorder=6)

def scene2a(ax, t):
    """最简网络原理：4×4 皮质柱 3入2出，状态值/权重值/性质 + 输入切换渐变。"""
    net_a = smooth(seg(t, 12.0, 12.7))
    if net_a <= 0:
        return
    E, I, lg, x = demo_sample(t)
    # ---- 公式与图例 ----
    draw_segments(ax, 960, 300,
                  [('E ← σ( ', FG), ('输入', ACCENT_IN), (' + ', FG),
                   ('W_rec·E', C_E_HI), (' + ', FG), ('τ·E', FG),
                   (' − ', FG), ('w_ei·I', C_I_HI), (' )', FG)], 32,
                  anchor='center')
    draw_segments(ax, 960, 248,
                  [('I ← σ( w_ie·E )', C_I_HI),
                   ('        动作 = argmax( ', FG), ('W_out·E', C_OUT_HI),
                   (' )', FG)], 27, anchor='center')
    ax.text(960, 200, '节点 = 皮质柱（E/I 同规模标注：上 = 兴奋态 E · 下蓝 = 柱内抑制 I） · '
                      '边上数字 = 权重值 · 橙 = 兴奋(+) / 蓝 = 抑制(−) · 循环无自环',
            color=DIM, fontsize=FS(17), family=YH, ha='center', va='center',
            alpha=net_a, zorder=6)
    # ---- 输入钉（值实时显示，切换时渐变） ----
    for j in range(3):
        px_, py_ = 660.0, DEMO_IN_Y[j]
        ax.fill([px_ - 13, px_ + 13, px_ + 13, px_ - 13],
                [py_ - 13, py_ - 13, py_ + 13, py_ + 13],
                facecolor=_dimc(ACCENT_IN, 0.35 + 0.65 * x[j]),
                edgecolor='none', alpha=net_a, zorder=5)
        ax.text(px_ - 24, py_, f'输入{j + 1}', color=FG, fontsize=FS(16),
                family=YH, ha='right', va='center', alpha=net_a, zorder=6)
        ax.text(px_, py_ - 26, f'{x[j]:.2f}', color=ACCENT_IN,
                fontsize=FS(14), family=MONO, ha='center', va='center',
                alpha=net_a, zorder=6)
    # ---- 边出现级联：12.0s 起每条边 0.42s 生长完成，首尾相接 ----
    _edge_n = [0]

    def _eprog():
        _i = _edge_n[0]
        _edge_n[0] += 1
        return smoother(seg(t, 12.0 + _i * 0.013, 12.0 + _i * 0.013 + 0.42))

    # ---- 输入边（宽度/标签=权重，亮度随输入值） ----
    for j in range(3):
        for c in range(16):
            w = DEMO_WIN[c, j]
            if w == 0:
                continue
            tx, ty = DEMO_POS[c]
            ux_, uy_ = tx - 660.0, ty - DEMO_IN_Y[j]
            L = np.hypot(ux_, uy_)
            ex, ey = tx - ux_ / L * (DEMO_R + 5), ty - uy_ / L * (DEMO_R + 5)
            draw_edge(ax, (692.0, DEMO_IN_Y[j]), (ex, ey),
                      ACCENT_IN, 14.0 if c % 2 else -14.0,
                      1.2 + 1.6 * w, net_a, label=f'+{w:.2f}',
                      bright=0.35 + 0.65 * x[j], avoid=DEMO_POS, prog=_eprog())
    # ---- 循环边（无自环；亮度随源端 E） ----
    for d, srcs in DEMO_REC.items():
        for s, w in srcs:
            p0, p1 = DEMO_POS[s], DEMO_POS[d]
            ux_, uy_ = p1[0] - p0[0], p1[1] - p0[1]
            L = np.hypot(ux_, uy_)
            ex, ey = p1[0] - ux_ / L * (DEMO_R + 4), p1[1] - uy_ / L * (DEMO_R + 4)
            sx, sy = p0[0] + ux_ / L * (DEMO_R + 4), p0[1] + uy_ / L * (DEMO_R + 4)
            if abs(w) > 0.55:                      # 0↔2 长程互抑制：弧线跨顶
                bend = 70.0 if d == 0 else -70.0
            else:
                bend = 12.0 if (s + d) % 2 else -12.0
            draw_edge(ax, (sx, sy), (ex, ey),
                      C_E_HI if w > 0 else C_I_HI, bend, 1.2 + 1.8 * abs(w),
                      net_a, label=f'+{w:.2f}' if w > 0 else f'−{abs(w):.2f}',
                      bright=0.35 + 0.65 * E[s], avoid=DEMO_POS, prog=_eprog())
    # ---- 皮质柱节点（E 值 + 柱内抑制 I 内盘） ----
    for c in range(16):
        cx, cy = DEMO_POS[c]
        ev = float(np.clip(E[c], 0.0, 1.0))
        ax.add_patch(plt.Circle((cx, cy), DEMO_R,
                                facecolor=CMAP_ACT(max(ev, 0.06)),
                                edgecolor=FG, lw=LW(1.0), alpha=net_a,
                                zorder=5))
        iv = float(np.clip(I[c], 0.0, 1.0))
        ax.text(cx, cy + 8, f'E {ev:.2f}', color=FG,
                fontsize=FS(14), family=MONO, ha='center', va='center',
                alpha=net_a, zorder=7,
                path_effects=[pe.withStroke(linewidth=2.4, foreground=BG)])
        ax.text(cx, cy - 16, f'I {iv:.2f}', color=C_I_HI,
                fontsize=FS(14), family=MONO, ha='center', va='center',
                alpha=0.95 * net_a, zorder=7,
                path_effects=[pe.withStroke(linewidth=2.4, foreground=BG)])
    # ---- 输出钉 + logit 柱 + argmax 环 ----
    for o in range(2):
        px_, py_ = 1620.0, (470.0, 610.0)[o]
        is_win = (lg[o] >= lg[1 - o] - 0.02)  # 均势带（浮点噪声级）
        ax.fill([px_ - 17, px_ + 17, px_ + 17, px_ - 17],
                [py_ - 17, py_ - 17, py_ + 17, py_ + 17],
                facecolor=C_OUT_HI, edgecolor='none', alpha=net_a, zorder=5)
        if is_win and t >= DEMO_PHASE_T[0]:
            ax.add_patch(plt.Circle((px_, py_), 27, facecolor='none',
                                    edgecolor=rwv.RING, lw=LW(2.4),
                                    alpha=net_a, zorder=6))
        ax.text(px_, py_ - 34, f'输出{o + 1}', color=ACCENT_OUT,
                fontsize=FS(17), family=YH, ha='center', va='center',
                alpha=net_a, zorder=6)
        bh = 55.0 * max(lg[o], 0.0)
        ax.fill([px_ + 40, px_ + 62, px_ + 62, px_ + 40],
                [py_ - bh / 2, py_ - bh / 2, py_ + bh / 2, py_ + bh / 2],
                facecolor=_dimc(C_OUT_HI, 1.0 if is_win else 0.4),
                edgecolor='none', alpha=net_a, zorder=5)
        ax.text(px_ + 51, py_ + bh / 2 + 18, f'{lg[o]:.2f}', color=C_OUT_HI,
                fontsize=FS(14), family=MONO, ha='center', va='center',
                alpha=net_a, zorder=6)
    # ---- 输出边 ----
    for o in range(2):
        for c, w in ((0, 1.3), (1, .8), (4, .35)) if o == 0 else ((2, 1.3), (3, .8), (6, .35)):
            sx, sy = DEMO_POS[c]
            ux_, uy_ = 1620.0 - 17 - sx, (470.0, 610.0)[o] - sy
            L = np.hypot(ux_, uy_)
            ex, ey = sx + ux_ / L * (DEMO_R + 4), sy + uy_ / L * (DEMO_R + 4)
            draw_edge(ax, (ex, ey), (1601.0, (470.0, 610.0)[o]),
                      C_OUT_HI, 10.0 if c % 2 else -10.0, 1.2 + 1.8 * w,
                      net_a, label=f'+{w:.2f}',
                      bright=0.35 + 0.65 * E[c], avoid=DEMO_POS, prog=_eprog())
    # ---- 相位说明 / 迭代计数 / 参数脚注 ----
    for _pi in range(len(DEMO_PHASE_T)):
        _t0 = DEMO_PHASE_T[_pi]
        _t1 = DEMO_PHASE_T[_pi + 1] if _pi + 1 < len(DEMO_PHASE_T) else 24.0
        _ca = win(t, _t0, _t1, 0.2, 0.2)
        if _ca > 0:
            ax.text(960, 70, DEMO_CAPTIONS[_pi], color=FG, fontsize=FS(22),
                    family=YH, ha='center', va='center', alpha=_ca, zorder=10)
    if t >= 23.0:
        ax.text(960, 40, 'τ_eff∈[0.001, 2] · w_ei, w_ie ≥ 0 · 循环项 = 固定扇入 gather-乘-归约 · 全部参数可进化',
                color=DIM, fontsize=FS(18), family=YH, ha='center', va='center',
                alpha=win(t, 23.0, 24.0, 0.3, 0.0), zorder=10)

# ==========================================
# 场景 2B：完整 256/32/3 网络 + 弹簧塑性
# ==========================================
N_NET, K_NET, N_IN, N_OUT = 256, 96, 32, 3
NET_REGION = (140.0, 1780.0, 120.0, 950.0)

def random_genome(seed=7):
    rng = np.random.default_rng(seed)
    ri = rng.integers(0, N_NET, (N_NET, K_NET))
    off = (ri == np.arange(N_NET)[:, None])
    ri[off] = (ri[off] + 1) % N_NET
    return dict(rec_idx=ri,
                rec_w=rng.normal(0, 0.05, (N_NET, K_NET)),
                W_in=rng.normal(0, 0.1, (N_NET, N_IN)),
                M_in=(rng.random((N_NET, N_IN)) < 0.15),
                W_out=rng.normal(0, 0.1, (N_OUT, N_NET)),
                M_out=(rng.random((N_OUT, N_NET)) < 0.15))

def build_graph(st):
    N, K = N_NET, K_NET
    in_xy = np.stack([np.full(N_IN, rwv.IN_X), np.linspace(0.90, 0.10, N_IN)], 1)
    out_y = np.array([0.5, 0.5 + rwv.OUT_SPAN / 2, 0.5 - rwv.OUT_SPAN / 2])
    out_xy = np.stack([np.full(N_OUT, rwv.OUT_X), out_y], 1)
    kk = min(K, rwv.N_REC_COMPUTE + 2)
    idxs = np.argsort(-np.abs(st['rec_w']), 1)[:, :kk]
    vals = np.take_along_axis(np.abs(st['rec_w']), idxs, 1)
    raw = np.take_along_axis(st['rec_w'], idxs, 1)
    comp = [[] for _ in range(N)]
    for d in range(N):
        seen = set()
        for j in range(kk):
            if len(comp[d]) >= rwv.N_REC_COMPUTE or vals[d, j] <= 0:
                break
            s = int(st['rec_idx'][d, idxs[d, j]])
            if s == d or s in seen:
                continue
            seen.add(s)
            comp[d].append((s, float(vals[d, j]), float(np.sign(raw[d, j]))))
    c_pairs = [(s, d) for d in range(N) for (s, w, g) in comp[d]]
    c_w = [w for d in range(N) for (s, w, g) in comp[d]]
    s_pairs = [(s, d) for d in range(N) for (s, w, g) in comp[d][:rwv.N_REC_SHOW]]
    s_w = [w for d in range(N) for (s, w, g) in comp[d][:rwv.N_REC_SHOW]]
    s_sign = [g for d in range(N) for (s, w, g) in comp[d][:rwv.N_REC_SHOW]]
    sp_src = np.array([p[0] for p in c_pairs], np.int64)
    sp_dst = np.array([p[1] for p in c_pairs], np.int64)
    c_wn = np.array(c_w); c_wn /= max(c_wn.max(), 1e-9)
    r_src = np.array([p[0] for p in s_pairs], np.int64)
    r_dst = np.array([p[1] for p in s_pairs], np.int64)
    r_wn = np.array(s_w); r_wn /= max(r_wn.max(), 1e-9)
    r_sign = np.array(s_sign)
    min_w = np.abs(st['W_in']) * st['M_in']
    order = np.argsort(-min_w, 0)[:rwv.N_IN_FAN, :]
    i_cols = order.T
    i_w = np.take_along_axis(min_w.T, order.T, 1)
    i_w = i_w / np.clip(i_w.max(1, keepdims=True), 1e-9, None)
    i_ch = np.repeat(np.arange(N_IN), rwv.N_IN_FAN)
    i_col = i_cols.flatten(); i_wn = i_w.flatten()
    out_w = np.abs(st['W_out']) * st['M_out']
    o_cols = np.argsort(-out_w, 1)[:, :rwv.N_OUT_FAN]
    o_w = np.take_along_axis(out_w, o_cols, 1)
    o_w = o_w / np.clip(o_w.max(1, keepdims=True), 1e-9, None)
    o_act = np.repeat(np.arange(N_OUT), rwv.N_OUT_FAN)
    o_col = o_cols.flatten(); o_wn = o_w.flatten()
    e_l0 = np.concatenate([np.full(len(sp_src), rwv.L0_REC),
                           np.full(len(i_ch), rwv.L0_IN),
                           np.full(len(o_act), rwv.L0_OUT)])
    e_k = np.concatenate([rwv.K_REC * (0.35 + 0.65 * c_wn),
                          rwv.K_REC * rwv.K_IN_MULT * (0.35 + 0.65 * i_wn),
                          rwv.K_REC * rwv.K_OUT_MULT * (0.35 + 0.65 * o_wn)])
    return dict(in_xy=in_xy, out_xy=out_xy,
                sp_src=sp_src, sp_dst=sp_dst, e_l0=e_l0, e_k=e_k,
                r_src=r_src, r_dst=r_dst, r_wn=r_wn, r_sign=r_sign,
                i_ch=i_ch, i_col=i_col, i_wn=i_wn,
                o_act=o_act, o_col=o_col, o_wn=o_wn)

def spring_frames(graph):
    """逐迭代连续采样（退火相每迭代 1 帧、弛豫相每 10 迭代 1 帧），
    返回 [T,256,2]，播放时线性插值 → 连续无跳变。"""
    N = N_NET
    total = N + N_IN + N_OUT
    in_xy, out_xy = graph['in_xy'], graph['out_xy']
    pos = np.zeros((total, 2))
    pos[N:] = np.concatenate([in_xy, out_xy], 0)
    g1 = np.linspace(0.36, 0.74, 16)
    g2 = np.linspace(0.18, 0.82, 16)
    gx, gy = np.meshgrid(g1, g2)
    pos[:N, 0] = gx.ravel(); pos[:N, 1] = gy.ravel()     # 16×16 正方形初始
    e_src = np.concatenate([graph['sp_src'], N + graph['i_ch'],
                            N + N_IN + graph['o_act']])
    e_dst = np.concatenate([graph['sp_dst'], graph['i_col'], graph['o_col']])
    e_l0, e_k = graph['e_l0'], graph['e_k']
    rep = rwv.REP_MULT * rwv.K_REC * rwv.NODE_D ** 4
    center = np.array([0.55, 0.50])

    def forces(e_k_loc):
        diff = pos[e_dst] - pos[e_src]
        dist = np.sqrt((diff ** 2).sum(1)) + 1e-9
        f = (e_k_loc * (dist - e_l0) / dist)[:, None] * diff
        F = np.zeros_like(pos)
        F[:, 0] += (np.bincount(e_src, weights=f[:, 0], minlength=total)
                    - np.bincount(e_dst, weights=f[:, 0], minlength=total))
        F[:, 1] += (np.bincount(e_src, weights=f[:, 1], minlength=total)
                    - np.bincount(e_dst, weights=f[:, 1], minlength=total))
        delta = pos[:N][:, None, :] - pos[:N][None, :, :]
        d2 = (delta ** 2).sum(-1) + 1e-9
        F[:N] += ((rep / d2 ** 1.5)[..., None] * delta).sum(1)
        if rwv.SPREAD > 0:
            wprof = np.exp(-0.5 * ((pos[:N, 0] - rwv.SPREAD_MID) / rwv.SPREAD_SIGMA) ** 2)
            yc = pos[:N, 1].mean()
            half = max(float(pos[:N, 1].max() - yc),
                       float(yc - pos[:N, 1].min()), 1e-3)
            u = (pos[:N, 1] - yc) / half
            F[:N, 1] += rwv.SPREAD * wprof * np.clip(
                4 * np.abs(u) * (1 - np.abs(u)), 0, 1) * np.sign(u)
        F[:N] += rwv.LAYOUT_CENTER_PULL * (center - pos[:N])
        return F

    # 加速日程表：刚度先大（快速拉入）退火到目标刚度；阻尼先小（带弹簧
    # 惯性/过冲）后大（超阻尼加速收敛）。视觉上 1/3 处即基本成形。
    K_START_MULT = 24.0   # 初始刚度倍率（平方退火 → ×1）
    DAMP_START, DAMP_END = 0.28, 1.25   # 阻尼倍率：先小（活泼回弹）→ 后大（收敛）
    frames = [pos[:N].copy()]
    vel = np.zeros_like(pos)
    for it in range(rwv.LAYOUT_ITERS):
        p = it / max(rwv.LAYOUT_ITERS - 1, 1)
        k_mult = 1.0 + (K_START_MULT - 1.0) * (1.0 - p) ** 2
        damp = rwv.DAMP * (DAMP_START + (DAMP_END - DAMP_START) * p)
        temp = rwv.LAYOUT_T0 * (1.0 - p) + 1e-5
        vel[:N] = damp * vel[:N] + forces(e_k * k_mult)[:N] * temp
        step = vel[:N]
        sn = np.sqrt((step ** 2).sum(1, keepdims=True)) + 1e-12
        step = step * np.minimum(1.0, rwv.LAYOUT_MAX_STEP / sn)
        pos[:N] += step
        np.clip(pos[:N], rwv.LAYOUT_MARGIN, 1.0 - rwv.LAYOUT_MARGIN, out=pos[:N])
        frames.append(pos[:N].copy())
    print(f'弹簧采样 {len(frames)} 帧（退火 {rwv.LAYOUT_ITERS} 迭代）', flush=True)
    return np.stack(frames)

def draw_network(ax, pos_u, graph, alpha, dim=1.0):
    if alpha <= 0:
        return
    x0, x1, y0, y1 = NET_REGION
    P = np.stack([x0 + pos_u[:, 0] * (x1 - x0),
                  y0 + pos_u[:, 1] * (y1 - y0)], 1)
    pins = np.concatenate([graph['in_xy'], graph['out_xy']], 0)
    Pp = np.stack([x0 + pins[:, 0] * (x1 - x0),
                   y0 + pins[:, 1] * (y1 - y0)], 1)
    i_px, o_px = Pp[:N_IN], Pp[N_IN:]
    segs = np.stack([P[graph['r_src']], P[graph['r_dst']]], 1)
    cols = np.zeros((len(graph['r_src']), 4))
    cols[graph['r_sign'] > 0, :3] = _hex2rgb(C_E_DIM)
    cols[graph['r_sign'] <= 0, :3] = _hex2rgb(C_I_DIM)
    cols[:, 3] = np.clip((rwv.EDGE_DIM_BASE + rwv.EDGE_DIM_GAIN * graph['r_wn'])
                         * alpha * dim, 0, 1)
    ax.add_collection(LineCollection(segs, colors=cols,
                                     linewidths=LW(0.5 + 1.5 * graph['r_wn']),
                                     zorder=2))
    segs_i = np.stack([i_px[graph['i_ch']], P[graph['i_col']]], 1)
    cols_i = np.zeros((len(graph['i_ch']), 4))
    cols_i[:, :3] = _hex2rgb(ACCENT_IN)
    cols_i[:, 3] = np.clip((0.12 + 0.78 * graph['i_wn']) * alpha * dim, 0, 1)
    ax.add_collection(LineCollection(segs_i, colors=cols_i,
                                     linewidths=LW(0.5 + 1.3 * graph['i_wn']),
                                     zorder=2))
    segs_o = np.stack([o_px[graph['o_act']], P[graph['o_col']]], 1)
    cols_o = np.zeros((len(graph['o_act']), 4))
    cols_o[:, :3] = _hex2rgb(ACCENT_OUT)
    cols_o[:, 3] = np.clip((rwv.EDGE_DIM_BASE + rwv.EDGE_DIM_GAIN * graph['o_wn'])
                           * alpha * dim, 0, 1)
    ax.add_collection(LineCollection(segs_o, colors=cols_o,
                                     linewidths=LW(0.5 + 1.3 * graph['o_wn']),
                                     zorder=2))
    ax.scatter(P[:, 0], P[:, 1], s=34, facecolor=FG, edgecolor='none',
               alpha=0.85 * alpha * dim, zorder=4)
    gcols = ([GRP_HEAD] * 4 + [GRP_TAIL] * 4 + [GRP_FOOD] * 8
             + [GRP_BODY] * 8 + [GRP_WALL] * 8)
    for j in range(N_IN):
        x, y = i_px[j]
        ax.fill([x - 7, x + 7, x + 7, x - 7], [y - 7, y - 7, y + 7, y + 7],
                facecolor=gcols[j], edgecolor='none', alpha=0.9 * alpha * dim,
                zorder=5)
    acts = ('直行', '左转', '右转')
    for j in range(N_OUT):
        x, y = o_px[j]
        ax.fill([x - 10, x + 10, x + 10, x - 10], [y - 10, y - 10, y + 10, y + 10],
                facecolor=C_OUT_HI, edgecolor='none', alpha=0.9 * alpha * dim,
                zorder=5)
        ax.text(x + 22, y, acts[j], color=ACCENT_OUT, fontsize=FS(18),
                family=YH, ha='left', va='center', alpha=alpha * dim, zorder=5)
    ax.text(i_px[0][0] - 24, (i_px[0][1] + i_px[-1][1]) / 2, '输入 32',
            color=FG, fontsize=FS(19), family=YH, ha='center', va='center',
            rotation=90, alpha=alpha * dim, zorder=5)

def scene2b(ax, t, frames, graph):
    a_in = smooth(seg(t, T2B[0], T2B[0] + 0.5))
    # 事实渲染：每视频帧直接取真实弹簧物理的仿真状态（速度/阻尼/退火原样，
    # 不做跨帧插值）；2500 步仿真按 6 物理步/视频帧 摊进 7s 窗口
    sub = max(1, int(np.ceil((len(frames) - 1)
                             / ((T_SPRING[1] - T_SPRING[0]) * FPS))))
    if t < T_SPRING[0]:
        pos, it = frames[0], 0
        note = '随机初始化 W · N=256 柱 · 16×16 方阵初始位置'
    elif t < T_SPRING[1]:
        it = int(min((t - T_SPRING[0]) * FPS * sub, len(frames) - 1))
        pos, note = frames[it], '弹簧迭代塑性 —— 权重塑造网络形状（逐帧物理渲染）'
    else:
        pos, it = frames[-1], len(frames) - 1
        note = None
    draw_network(ax, pos, graph, a_in)
    if note is not None:
        ax.text(960, 62, note, color=FG, fontsize=FS(24), family=YH,
                ha='center', va='center',
                alpha=win(t, T2B[0] + 0.3, T_SPRING[1], 0.3, 0.2), zorder=10)
    if T_SPRING[0] <= t <= T_SPRING[1]:
        ax.text(1868, 1030, f'弹簧迭代 {it}/{len(frames) - 1}', color=FG,
                fontsize=FS(24), family=MONO, ha='right', va='center',
                alpha=0.9, zorder=10)
    if t >= T_SPRING[1] - 0.1:
        ax.text(1868, 1030, 'N=256 · K=96 · τ_eff∈[0.001,2]', color=FG,
                fontsize=FS(24), family=MONO, ha='right', va='center',
                alpha=win(t, T_SPRING[1] - 0.1, T2B[1], 0.3, 0.3), zorder=10)

# ==========================================
# 场景 3：EA（两阶段筛选 + 精英 + 交叉变异 + 曲线）
# ==========================================
POP, ELITE, GRID_N, DOT = 4096, 1024, 64, 11.0
POPX0, POPY1 = 340.0, 900.0
CURVE_A = CURVE_B = None

def draw_curve_panel(ax, x0, x1, y0, y1, data3, title, tcol, formula,
                     alpha, grow, win_line=None, milestone=False,
                     ymax_common=None):
    """data3 = (best, elite, avg) 同长度数组；y 向上（值大在上）。
    ymax_common：两面板统一量程；给出时在左侧标注 0/98 刻度。"""
    for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                 ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=GRID, lw=LW(1.4),
                alpha=alpha, zorder=3)
    ax.text((x0 + x1) / 2, y1 + 44, title, color=tcol, fontsize=FS(24),
            family=YH, ha='center', va='center', alpha=alpha, zorder=6)
    best, elite, avg = data3
    dmax = float(np.nanmax(best))
    pad = dmax * 0.08 + 2
    ymax = max(dmax + pad, (win_line or 0) + pad)
    if ymax_common is not None:
        ymax = ymax_common
    xs = np.arange(len(best))
    px = x0 + 46 + (xs / max(len(best) - 1, 1)) * (x1 - x0 - 66)
    def to_y(d):
        return y0 + 34 + (np.clip(d, 0, None) / ymax) * (y1 - y0 - 74)   # 值大→靠上
    nshow = max(2, int(grow * len(best)))
    ax.plot(px[:nshow], to_y(avg)[:nshow], color=DIM, lw=LW(1.6),
            alpha=0.8 * alpha, zorder=4)
    ax.plot(px[:nshow], to_y(elite)[:nshow], color=tcol, lw=LW(1.8),
            alpha=0.5 * alpha, zorder=4)
    ax.plot(px[:nshow], to_y(best)[:nshow], color=tcol, lw=LW(2.6),
            alpha=0.95 * alpha, zorder=5, solid_capstyle='round')
    # 图例随面板出现即显示（右下角），不等待曲线生长完成
    for i, (d, cc, llw, aa, name) in enumerate(
            ((avg, DIM, 1.6, 0.8, '平均'), (elite, tcol, 1.8, 0.5, '精英'),
             (best, tcol, 2.6, 0.95, '最佳'))):
        yy = y0 + 92 - i * 30               # 右下角：曲线后期爬升在右上，右下恒空
        ax.plot([x1 - 150, x1 - 118], [yy, yy], color=cc, lw=LW(llw),
                alpha=aa * alpha, zorder=6)
        ax.text(x1 - 110, yy, name, color=cc, fontsize=FS(14), family=YH,
                ha='left', va='center', alpha=alpha, zorder=6)
    if win_line is not None and grow >= 1.0:
        wy = to_y(win_line)
        ax.plot([x0 + 46, x1 - 20], [wy, wy], color=C_OUT_HI, lw=LW(1.2),
                ls=(0, (5, 4)), alpha=0.8 * alpha, zorder=4)
        ax.text(x0 + 52, wy + 18, f'通关 {win_line:.0f}', color=C_OUT_HI,
                fontsize=FS(16), family=YH, ha='left', va='center', alpha=alpha)
    if milestone and grow >= 1.0:
        gi = int(np.nanargmax(best))
        ax.add_patch(plt.Circle((px[gi], to_y(best)[gi]), 6, facecolor=tcol,
                                edgecolor=BG, lw=LW(1.4), alpha=alpha, zorder=6))
        ax.text(px[gi] + 12, to_y(best)[gi] + 24, f'gen{gi}',
                color=tcol, fontsize=FS(16), family=MONO, ha='left',
                va='center', alpha=alpha)
        ax.plot([x0 + 46, x1 - 20], [to_y(best)[gi]] * 2, color=tcol,
                lw=LW(1.0), ls=(0, (4, 4)), alpha=0.55 * alpha, zorder=4)
        ax.text(x0 + 52, to_y(best[gi]) + 16, f'最大 {best[gi]:.0f}',
                color=tcol, fontsize=FS(13), family=MONO, ha='left',
                va='center', alpha=alpha)
    ax.text(x0 + 46, y0 + 12, 'gen 0', color=DIM, fontsize=FS(14), family=MONO,
            ha='left', va='center', alpha=alpha)
    if ymax_common is not None and win_line is not None and grow >= 1.0:
        for vv, lab in ((0.0, '0'), (98.0, '98')):
            ax.text(x0 + 8, to_y(vv), lab, color=DIM, fontsize=FS(13),
                    family=MONO, ha='left', va='center', alpha=alpha,
                    zorder=6)
    ax.text(x1 - 20, y0 + 12, f'gen {len(best) - 1}', color=DIM, fontsize=FS(14),
            family=MONO, ha='right', va='center', alpha=alpha)
    ax.text(x0 + 14, y1 - 8, 'food', color=DIM, fontsize=FS(14), family=YH,
            ha='left', va='top', alpha=alpha)
    if formula is not None:
        draw_segments(ax, (x0 + x1) / 2, y0 - 46, formula, 21, anchor='center')

# ---- 种群随机颜色 / 重拓展预定表（固定种子） ----
_POP_RNG = np.random.default_rng(23)
POP_COL = plt.cm.hsv(_POP_RNG.random(POP))[:, :3]     # 4096 个随机色相
MUT_MASK = _POP_RNG.random(POP) < 0.08                # 变异闪烁子集
SPAWN_RANK = _POP_RNG.permutation(POP)                # 重拓展出现次序
SPLICE_PA = _POP_RNG.integers(0, 1024, POP)           # 拼半父本 A（精英）
SPLICE_PB = _POP_RNG.integers(0, 1024, POP)           # 拼半父本 B（精英）
T_APP = 45.0 + 1.6 * (SPAWN_RANK / (POP - 1024.0))    # 各淘汰位重生时刻


def scene3(ax, t):
    a_in = smooth(seg(t, T3[0], T3[0] + 0.6))         # 种群方阵整体淡入
    idx = np.arange(POP)
    px_ = POPX0 + (idx % GRID_N + 0.5) * DOT
    py_ = POPY1 - (idx // GRID_N + 0.5) * DOT

    # 两阶段筛选：先 6 局（4096→2048），再 18 局（2048→精英 1024）
    kill1 = smooth(seg(t, 39.0, 39.8))                # 6 局淘汰
    kill2 = smooth(seg(t, 41.4, 42.2))                # 18 局淘汰
    grow = smoother(seg(t, 45.0, 46.6))               # 残影随重拓展消退
    mut = win(t, 46.8, 48.2, 0.2, 0.3)                # 变异闪烁
    out = 1.0 - smooth(seg(t, 50.4, 52.8))   # 内容播完(49.6)停0.8s后整场一起淡出

    elite = idx < 1024
    dead1 = idx >= 2048                               # 6 局淘汰
    dead2 = (idx >= 1024) & ~dead1                    # 18 局淘汰
    dead = dead1 | dead2

    # ---- 存活个体：随机颜色 + 亮度状态 ----
    bright = np.full(POP, 0.72)
    bright[elite & (t >= 42.2)] = 1.0
    if mut > 0:
        bright[MUT_MASK] = bright[MUT_MASK] + mut * 0.6
    cols = np.zeros((POP, 4))
    cols[:, :3] = POP_COL
    alive_a = np.where(dead2, 1.0 - kill2, np.where(dead1, 1.0 - kill1, 1.0))
    cols[:, 3] = np.clip(a_in * alive_a * np.clip(bright, 0, 1.3), 0, 1) * out
    ax.scatter(px_, py_, s=17, facecolors=cols, edgecolors='none', zorder=4)

    # ---- 淘汰位残影（暗灰点，保持方阵轮廓，重拓展时消退） ----
    ghost = np.zeros((POP, 4))
    ghost[:, :3] = _hex2rgb(DIM)
    ghost[:, 3] = 0.07 * a_in * dead * (1.0 - grow) * out
    ax.scatter(px_, py_, s=10, facecolors=ghost, edgecolors='none', zorder=3)

    # ---- 交叉拼半：新个体 = 精英 A 左半 + 精英 B 右半 ----
    a_new = np.clip(np.where(dead, smoother(seg(t, T_APP, T_APP + 0.35))
                             * a_in, 0.0), 0.0, 1.0) * out
    sel = a_new > 0
    if sel.any():
        ii = idx[sel]
        w = DOT * 0.42
        ca = np.zeros((int(sel.sum()), 4))
        cb = np.zeros((int(sel.sum()), 4))
        ca[:, :3] = POP_COL[SPLICE_PA[ii]]
        cb[:, :3] = POP_COL[SPLICE_PB[ii]]
        ca[:, 3] = a_new[sel]
        cb[:, 3] = a_new[sel]
        ax.scatter(px_[ii] - w / 2, py_[ii], s=14, marker='s',
                   facecolors=ca, edgecolors='none', zorder=5)
        ax.scatter(px_[ii] + w / 2, py_[ii], s=14, marker='s',
                   facecolors=cb, edgecolors='none', zorder=5)

    # ---- 阶段字幕 ----
    caps = (
        (37.5, 39.0, '筛选 1 · 每基因 6 局'),
        (39.0, 40.6, '6 局 → 幸存 2048'),
        (40.6, 42.4, '筛选 2 · 幸存者 18 局'),
        (42.4, 45.0, '18 局 → 精英 1024'),
        (45.0, 50.4, '交叉（两半拼合）+ 变异 → 重新拓展 4096'),
    )
    for t0, t1, txt in caps:
        ca_ = win(t, t0, t1, 0.2, 0.2) * out
        if ca_ > 0:
            ax.text(960, 1032, txt, color=FG, fontsize=FS(24), family=YH,
                    ha='center', va='center', alpha=ca_, zorder=10)

    # ---- 右侧：总数 / 筛选盘数 / 重拓展 ----
    panels = (
        (36.0, 38.0, [('种群总数', FG, 24), ('4096', FG, 40),
                      ('fitness = food + 0.3·food/步数', GRP_FOOD, 17)]),
        (38.0, 40.6, [('筛选 1 · 每基因 6 局', FG, 22),
                      ('→ 幸存 2048', FG, 22)]),
        (40.6, 45.0, [('筛选 2 · 幸存者 18 局', FG, 22),
                      ('→ 精英 1024', FG, 22)]),
    )
    for t0, t1, lines in panels:
        ca_ = win(t, t0, t1, 0.25, 0.25) * out
        if ca_ <= 0:
            continue
        for li, (txt, col, fs) in enumerate(lines):
            ax.text(1170, 700 - li * 62, txt, color=col, fontsize=FS(fs),
                    family=YH, ha='left', va='center', alpha=ca_, zorder=10)

    # ---- 右侧：交叉与变异解释（45.0 起，与左侧重拓展同步） ----
    cx_a, cy_a = 1330.0, 400.0            # 精英A
    cx_b, cy_b = 1610.0, 400.0            # 精英B
    cx_c, cy_c = 1470.0, 210.0            # 子代（两亲本中间下方）
    s_n = 0.62

    def net_dots(cx, cy):
        return [(cx + (-63 + cc * 42) * s_n, cy + (-63 + rr * 42) * s_n)
                for rr in range(4) for cc in range(4)]

    dots_a = net_dots(cx_a, cy_a)
    dots_b = net_dots(cx_b, cy_b)
    dots_c = net_dots(cx_c, cy_c)
    ins_a = [(cx_a - 61, cy_a - 17), (cx_a - 61, cy_a + 17)]
    ins_b = [(cx_b - 61, cy_b - 17), (cx_b - 61, cy_b + 17)]
    ins_c = [(cx_c - 61, cy_c - 17), (cx_c - 61, cy_c + 17)]
    out_a = (cx_a + 61, cy_a)
    out_b = (cx_b + 61, cy_b)
    out_c = (cx_c + 61, cy_c)

    p_a = smoother(seg(t, 45.4, 46.2))    # 精英A 遗传部分迁移
    p_b = smoother(seg(t, 46.2, 47.0))    # 精英B 遗传部分迁移
    dots_in = smoother(seg(t, 45.4, 45.9))
    mut_p = seg(t, 46.8, 48.2)
    ea_x = smooth(seg(t, 45.0, 45.4)) * out

    def E4(p, q):
        return (p[0], p[1], q[0], q[1])

    def E4lerp(e4, q4, p_):
        return (e4[0] + (q4[0] - e4[0]) * p_, e4[1] + (q4[1] - e4[1]) * p_,
                e4[2] + (q4[2] - e4[2]) * p_, e4[3] + (q4[3] - e4[3]) * p_)

    def draw_e4(e4, c, lw=1.2, a_=0.7):
        ax.plot([e4[0], e4[2]], [e4[1], e4[3]], color=c, lw=lw, alpha=a_,
                solid_capstyle='round', zorder=10)

    # 完整网络基底：2入→16柱全连 + 16柱两两互连 + 16柱→1出（无孤立点）
    def full_edges(dots, ins, outp):
        es = []
        for ip in ins:
            for d in dots:
                es.append(E4(ip, d))
        for i in range(16):
            for j in range(i + 1, 16):
                es.append(E4(dots[i], dots[j]))
        for d in dots:
            es.append(E4(d, outp))
        return es

    base_a = full_edges(dots_a, ins_a, out_a)
    base_b = full_edges(dots_b, ins_b, out_b)
    base_c = full_edges(dots_c, ins_c, out_c)

    def draw_net(dots, ins, outp, base, a_):
        if a_ <= 0:
            return
        xs, ys = [], []
        for e4 in base:
            xs += [e4[0], e4[2], np.nan]
            ys += [e4[1], e4[3], np.nan]
        ax.plot(xs, ys, color=FG, lw=LW(0.9), alpha=0.32 * a_,
                solid_capstyle='round', zorder=9)
        for x, y in dots:
            ax.add_patch(plt.Circle((x, y), 4.5 * s_n, facecolor=FG,
                                    edgecolor='none', alpha=0.9 * a_,
                                    zorder=11))
        for ip in ins:
            ax.add_patch(plt.Rectangle((ip[0] - 8, ip[1] - 8), 16, 16,
                                       facecolor=ACCENT_IN, edgecolor='none',
                                       alpha=0.9 * a_, zorder=11))
        ax.add_patch(plt.Rectangle((outp[0] - 8, outp[1] - 8), 16, 16,
                                   facecolor=C_OUT_HI, edgecolor='none',
                                   alpha=0.9 * a_, zorder=11))

    if ea_x > 0:
        # 双亲完整网络（遗传部分高亮，随迁移动画移入子代）
        draw_net(dots_a, ins_a, out_a, base_a, 0.6 * ea_x)
        draw_net(dots_b, ins_b, out_b, base_b, 0.6 * ea_x)
        # 遗传部分高亮并迁移到子代
        inh_a = [(E4(ins_a[0], dots_a[0]), E4(ins_c[0], dots_c[0]), GRP_HEAD),
                 (E4(ins_a[1], dots_a[1]), E4(ins_c[1], dots_c[1]), GRP_HEAD),
                 (E4(dots_a[0], dots_a[1]), E4(dots_c[0], dots_c[1]), C_E_HI),
                 (E4(dots_a[4], dots_a[5]), E4(dots_c[4], dots_c[5]), C_E_HI),
                 (E4(dots_a[1], out_a), E4(dots_c[1], out_c), ACCENT_OUT)]
        inh_b = [(E4(ins_b[0], dots_b[2]), E4(ins_c[0], dots_c[2]), '#6fc9c0'),
                 (E4(ins_b[1], dots_b[3]), E4(ins_c[1], dots_c[3]), '#6fc9c0'),
                 (E4(dots_b[2], dots_b[3]), E4(dots_c[2], dots_c[3]), C_I_HI),
                 (E4(dots_b[6], dots_b[7]), E4(dots_c[6], dots_c[7]), C_I_HI),
                 (E4(dots_b[3], out_b), E4(dots_c[3], out_c), ACCENT_OUT)]
        for e4, q4, c in inh_a:
            draw_e4(E4lerp(e4, q4, p_a), c, 1.7, 0.95 * ea_x)
        for e4, q4, c in inh_b:
            draw_e4(E4lerp(e4, q4, p_b), c, 1.7, 0.95 * ea_x)
        # 子代（完整网络基底，接收遗传高亮部分）
        if dots_in > 0:
            xs_c, ys_c = [], []
            for e4 in base_c:
                xs_c += [e4[0], e4[2], np.nan]
                ys_c += [e4[1], e4[3], np.nan]
            ax.plot(xs_c, ys_c, color=FG, lw=LW(0.9),
                    alpha=0.32 * dots_in * ea_x, solid_capstyle='round',
                    zorder=9)
            for x, y in dots_c:
                ax.add_patch(plt.Circle((x, y), 4.5 * s_n, facecolor=FG,
                                        edgecolor='none',
                                        alpha=0.9 * dots_in * ea_x, zorder=11))
            for ip in ins_c:
                ax.add_patch(plt.Rectangle((ip[0] - 8, ip[1] - 8), 16, 16,
                                           facecolor=ACCENT_IN,
                                           edgecolor='none',
                                           alpha=0.9 * dots_in * ea_x,
                                           zorder=11))
            ax.add_patch(plt.Rectangle((out_c[0] - 8, out_c[1] - 8), 16, 16,
                                       facecolor=C_OUT_HI, edgecolor='none',
                                       alpha=0.9 * dots_in * ea_x, zorder=11))
        # ---- 变异（独立内容）：子代一条循环边随机变色；文字标在网络外 ----
        if mut_p > 0:
            fl = 0.5 + 0.5 * np.sin(mut_p * 12.0 * np.pi)
            c_mut = tuple(np.array(_hex2rgb(C_I_HI)) * (1.0 - 0.7 * fl)
                          + np.array(_hex2rgb('#FFD23F')) * (0.7 * fl))
            draw_e4(E4(dots_c[2], dots_c[3]), c_mut, 2.4, 0.95 * ea_x)
            y_e = dots_c[2][1]
            ax.plot([dots_c[3][0] + 10, dots_c[3][0] + 40], [y_e, y_e],
                    color='#FFD23F', lw=LW(1.0), alpha=0.7 * ea_x,
                    solid_capstyle='round', zorder=11)
            ax.text(dots_c[3][0] + 46, y_e, '变异',
                    color='#FFD23F', fontsize=FS(14), family=YH,
                    ha='left', va='center', alpha=0.9 * ea_x, zorder=11)

    ax.text(1330, 312, '精英A', color=DIM, fontsize=FS(15), family=YH,
            ha='center', va='center', alpha=smooth(seg(t, 45.0, 45.4)) * out,
            zorder=10)
    ax.text(1610, 312, '精英B', color=DIM, fontsize=FS(15), family=YH,
            ha='center', va='center', alpha=smooth(seg(t, 45.0, 45.4)) * out,
            zorder=10)
    ax.text(1470, 152, '子代', color=DIM, fontsize=FS(15), family=YH,
            ha='center', va='center',
            alpha=smooth(seg(t, 45.8, 46.2)) * out, zorder=10)

    # ---- 色块图例（解释左阵随机颜色 / 拼色 / 变异） ----
    d2 = ((POP_COL[:1024][:, None, :] - POP_COL[:1024][None, :, :]) ** 2).sum(-1)
    ia, ib = np.unravel_index(np.argmax(d2), d2.shape)
    cA, cB = POP_COL[ia], POP_COL[ib]
    mut_on = smooth(seg(t, 46.8, 47.3))   # 变异内容淡入后保持到整场淡出
    c_y = np.array(_hex2rgb('#FFD23F'))
    c_mut = tuple(np.array(cB) * (1.0 - 0.8 * mut_on) + c_y * (0.8 * mut_on))
    pa_g = smooth(seg(t, 45.0, 45.4)) * out   # 图例随场景出现，整场结束时一起消失
    ax.text(1170, 655, '交叉与变异', color=FG, fontsize=FS(22),
            family=YH, ha='left', va='center', alpha=pa_g, zorder=10)
    # 内容链：[精英A|精英B] --交叉--> [子代] --变异--> [子代变异后]
    # 三个方块均匀排布，'交叉'/'变异' 标在箭头上，整体随 pa_g 同步淡入淡出
    by0, bh = 585.0, 30.0
    cxs = (1250.0, 1462.0, 1674.0)
    ax.add_patch(plt.Rectangle((cxs[0] - 30, by0), 30, bh, facecolor=cA,
                               alpha=pa_g, zorder=10))
    ax.add_patch(plt.Rectangle((cxs[0], by0), 30, bh, facecolor=cB,
                               alpha=pa_g, zorder=10))
    ax.add_patch(plt.Rectangle((cxs[1] - 15, by0), 15, bh, facecolor=cA,
                               alpha=pa_g, zorder=10))
    ax.add_patch(plt.Rectangle((cxs[1], by0), 15, bh, facecolor=cB,
                               alpha=pa_g, zorder=10))
    ax.add_patch(plt.Rectangle((cxs[2] - 15, by0), 15, bh, facecolor=cB,
                               alpha=mut_on * pa_g, zorder=10))
    ax.add_patch(plt.Rectangle((cxs[2], by0), 15, bh, facecolor=c_mut,
                               alpha=mut_on * pa_g, zorder=10))

    def leg_arrow(x_from, x_to, lab, a_):
        if a_ <= 0:
            return
        ym = by0 + bh / 2
        ax.plot([x_from, x_to - 8], [ym, ym], color=FG, lw=LW(1.4),
                alpha=a_, zorder=10, solid_capstyle='round')
        ax.add_patch(plt.Polygon(((x_to, ym), (x_to - 10, ym + 5),
                                  (x_to - 10, ym - 5)), facecolor=FG,
                                 edgecolor='none', alpha=a_, zorder=10))
        ax.text((x_from + x_to) / 2, ym + 22, lab, color=FG,
                fontsize=FS(15), family=YH, ha='center', va='center',
                alpha=a_, zorder=10)

    leg_arrow(cxs[0] + 38, cxs[1] - 26, '交叉', pa_g)
    leg_arrow(cxs[1] + 26, cxs[2] - 26, '变异', mut_on * pa_g)
    for cx, tx, a_ in ((cxs[0], '精英A 精英B', pa_g),
                       (cxs[1], '子代', pa_g),
                       (cxs[2], '子代变异后', mut_on * pa_g)):
        ax.text(cx, by0 - 18, tx, color=DIM, fontsize=FS(12), family=YH,
                ha='center', va='center', alpha=a_, zorder=10)

    if t >= T_CURVES[0]:
        ca = min(1.0, (t - T_CURVES[0]) / 0.4)
        grow_ = smoother(seg(t, 54.2, 58.5))
        ax.text(960, 1032, '训练过程', color=FG, fontsize=FS(26), family=YH,
                ha='center', va='center', alpha=ca, zorder=10)
        ymax_c = 98.0 * 1.08 + 2.0       # 统一量程：98 通关线上留白
        draw_curve_panel(ax, 250, 920, 300, 800, CURVE_A,
                         '从 0 → 最佳基模', GRP_HEAD,
                         [('适应度 = ', FG), ('food', GRP_FOOD),
                          ('+ 0.3·food/步数', FG)],
                         ca, grow_, win_line=98.0, ymax_common=ymax_c,
                         milestone=True)
        draw_curve_panel(ax, 1000, 1670, 300, 800, CURVE_B,
                         '最佳基模（固定种子）→ 通关', C_OUT_HI,
                         [('适应度 = ', FG), ('food', GRP_FOOD),
                          ('+ 0.3·food/步数', FG)],
                         ca, grow_, win_line=98.0, ymax_common=ymax_c)


# 主渲染
# ==========================================
class Intro:
    def __init__(self):
        self.fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.frames = None
        self.graph = None

    def prep(self):
        st = random_genome()
        self.graph = build_graph(st)
        print('弹簧逐迭代采样中...', flush=True)
        t0 = time.time()
        self.frames = spring_frames(self.graph)
        print(f'共 {len(self.frames)} 帧（{time.time() - t0:.1f}s）', flush=True)

    def frame(self, t):
        ax = self.ax
        ax.clear()
        ax.set_xlim(0, 1920)
        ax.set_ylim(0, 1080)
        ax.axis('off')
        ax.set_facecolor(BG)
        if t < T2A[0]:
            scene1(ax, t)
        elif t < T2B[0]:
            scene2a(ax, t)
        elif t < T3[0]:
            scene2b(ax, t, self.frames, self.graph)
        else:
            scene3(ax, t)
        for _b in (T2A[0], T2B[0], T3[0]):    # 场景切换：旧内容 0.4s 淡出入底色
            _f = win(t, _b - 0.4, _b + 0.4, 0.4, 0.4)
            if _f > 0:
                ax.add_patch(plt.Rectangle((0, 0), 1920, 1080, facecolor=BG,
                                           edgecolor='none', alpha=_f,
                                           zorder=50))

    def still(self, t, path):
        self.frame(t)
        self.fig.savefig(path, facecolor=BG)
        print('still', path, flush=True)

    def render(self, out, fps=FPS, t_end=TOTAL):
        write, release = rwv.pick_writer(out, fps, 1920, 1080)
        n = int(round(t_end * fps))
        t0 = time.time()
        for f in range(n):
            self.frame((f + 0.5) / fps)
            self.fig.canvas.draw()
            buf = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy()
            write(buf)
            if f % 200 == 0:
                el = time.time() - t0
                print(f'帧 {f + 1}/{n} ({el:.0f}s, '
                      f'ETA {el / (f + 1) * (n - f - 1) / 60:.1f}min)', flush=True)
        release()


def load_curve(path, key):
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    return np.asarray(d[key], dtype=float)

def main():
    global CURVE_A, CURVE_B
    ap = argparse.ArgumentParser(description='前置介绍动画 60s')
    ap.add_argument('--stills', action='store_true', help='只出关键帧 PNG')
    ap.add_argument('--out', default='intro_1080p.mp4')
    ap.add_argument('--fps', type=int, default=FPS)
    args = ap.parse_args()

    CURVE_A = (load_curve('test16b_simp_history.json', 'best_food'),
               load_curve('test16b_simp_history.json', 'elite_food'),
               load_curve('test16b_simp_history.json', 'avg_food'))
    CURVE_B = (load_curve('16c_cheat7b_history.json', 'best_food'),
               load_curve('16c_cheat7b_history.json', 'elite_food'),
               load_curve('16c_cheat7b_history.json', 'avg_food'))
    print(f'曲线A best {CURVE_A[0][0]:.1f}→{CURVE_A[0][-1]:.1f} | '
          f'曲线B best {CURVE_B[0][0]:.1f}→{CURVE_B[0][-1]:.1f}', flush=True)
    intro = Intro()
    intro.prep()
    os.makedirs('preview', exist_ok=True)
    if args.stills:
        for t in (0.8, 2.0, 4.3, 6.1, 7.9, 9.3, 10.6, 13.0, 15.0, 17.7, 20.1,
                  22.8, 25.2, 28.5, 31.0, 34.5, 37.5, 40.0, 42.5, 44.5, 46.5,
                  49.0, 52.0, 56.5, 59.0):
            intro.still(t, f'preview/intro_still_{t:04.1f}.png')
        return
    intro.render(args.out, args.fps)
    print(f'完成: {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)', flush=True)
    if shutil.which('ffmpeg'):
        h264 = args.out.replace('.mp4', '_h264.mp4')
        os.system(f'ffmpeg -y -i "{args.out}" -c:v libx264 -pix_fmt yuv420p '
                  f'-crf 20 "{h264}" -loglevel error')
        if os.path.exists(h264) and os.path.getsize(h264) > 1e6:
            os.remove(args.out)
            print(f'H.264 版本: {h264}', flush=True)


if __name__ == '__main__':
    main()
