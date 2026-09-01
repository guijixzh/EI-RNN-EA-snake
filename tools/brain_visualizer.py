# -*- coding: utf-8 -*-
"""
test5a 最优模型 —— 贪吃蛇游玩 + 实时神经元活动可视化服务器

技术栈：
  - 后端：Python 标准库 http.server + SSE(Server-Sent Events) 推送，零新增依赖
  - 前端：static/index.html（原生 HTML/CSS/JS，无 CDN）

运行：
  python brain_visualizer.py [--port 8000] [--model 5a|fast]

接口：
  GET /                前端页面
  GET /api/init?model=  初始化/切换模型，返回拓扑布局与模型元信息
  GET /api/control?action=pause|resume|new&speed=2.0   控制游戏线程
  GET /stream          SSE 事件流（init 事件 + frame 帧）
"""

import argparse
import glob
import importlib.util
import json
import math
import os
import random
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
import networkx as nx

try:
    import community as community_louvain
except ImportError:
    community_louvain = None

# ---- 复用 einbrain 的游戏环境 / 脑区模型 / 加载 / 决策逻辑 ----
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import einbrain
from einbrain import Config, SnakeEnv, deliberate_action

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
DEFAULT_MODEL = os.path.join(MODELS_DIR, "test5a_best_model.pth")
MODEL_FAST_PATH = os.path.join(MODELS_DIR, "test5d_best_model_33.pth")
# 默认加载：优先最新版本的引擎模型（12 > 7g > 7d > 7c > 7b > 7a > 5a）
DEFAULT_MODEL_KEY = next(
    (key for key, fname in [("12", "test12_econ_best_model.pth"),
                            ("7g", "test7g_econ_best_model.pth"),
                            ("7d", "test7d_econ_best_model.pth"),
                            ("7c", "test7c_latest_gen_best.pth"),
                            ("7b", "test7b_best_model.pth"),
                            ("7a", "test7a_v5a_best_model.pth")]
     if os.path.exists(os.path.join(REPO_ROOT, fname))), "5a")

TOP_REC_EDGES = 140      # 拓扑图中展示的递归边 top-K 条数
REC3D_TOP_PER_NEURON = 10   # 3D 弹簧图：每个神经元保留自己最大权重的前 10 条输入边


# ============================================================
# 1b. test7 系进化引擎支持（GeneStack 张量格式）
#     统一接口：module 提供 Config / GeneStack / forward_batch /
#     update_fatigue / BatchedSnakeEnv / load_best_state
#     新增版本（如 test7c）只需在 ENGINES 里加一行
# ============================================================
class Engine:
    def __init__(self, name, module_file, detect):
        self.name = name            # 引擎标识（写入 meta.engine）
        self.module_file = module_file
        self.detect = detect        # detect(cfg_dict) -> bool，基于 pth 内 config 专有字段判断归属
        self._mod = None

    def module(self):
        """按需导入并缓存对应的训练脚本（仅用其类与函数，不执行 main）"""
        if self._mod is None:
            path = os.path.join(REPO_ROOT, self.module_file)
            modname = os.path.splitext(self.module_file)[0] + "_viz"
            spec = importlib.util.spec_from_file_location(modname, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[modname] = mod
            spec.loader.exec_module(mod)
            self._mod = mod
        return self._mod


# 顺序即 detect 优先级：专有字段多的（新版本）放前面
# 注意：test12 的 Config 也含 OBS_MANHATTAN（与 7g 同），必须排在 7g 之前，
#       用其专有字段（OBS_ENC_VERSION / ISLAND_PENALTY）区分。
ENGINES = [
    Engine("12", "test12.py",
           lambda c: "OBS_ENC_VERSION" in c or "ISLAND_PENALTY" in c),
    Engine("7g", "test7g.py", lambda c: "OBS_MANHATTAN" in c),
    Engine("7d", "test7d.py",
           lambda c: "TURN_COST" in c or "TURN_PENALTY" in c or
                     "ONE_SIDED_TURN_DEATH" in c),
    Engine("7c", "test7c.py",
           lambda c: "FATIGUE_TURN_DECAY" in c or "FATIGUE_TURN_GAIN" in c),
    Engine("7b", "test7b.py",
           lambda c: "STARVE_SLOPE" in c or "FOOD_EFF_WEIGHT" in c),
    Engine("7a", "test7a.py",
           lambda c: "OBS_MODE" in c or "EVAL_BATCH" in c),
]

# 内置快捷 key -> 默认模型文件名（不存在时回落到扫描到的第一个）
ENGINE_DEFAULT_FILE = {
    "12": "test12_econ_best_model.pth",
    "7a": "test7a_v5a_best_model.pth",
    "7b": "test7b_best_model.pth",
    "7c": "test7c_latest_gen_best.pth",
    "7d": "test7d_econ_best_model.pth",
    "7g": "test7g_econ_best_model.pth",
}


def detect_engine(path):
    """读取 pth 的 config dict，按 ENGINES 顺序匹配归属引擎；失败返回 None"""
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    cfgd = data.get("config", {}) or {}
    for eng in ENGINES:
        if eng.detect(cfgd):
            return eng
    return None


class BrainGeneAdapter:
    """把 test7 系 GeneStack 单个体包装成 compute_layout 所需的 brain 接口"""
    is_gene_engine = True

    def __init__(self, pop, t7cfg, engine):
        self.pop = pop
        self.t7cfg = t7cfg
        self.engine = engine
        self.N = int(pop.N)
        self.obs_dim = int(pop.O)
        self.action_dim = int(pop.A)
        with torch.no_grad():
            self.M_in = pop.M_in[0].float().cpu()
            self.M_rec = pop.M_rec[0].float().cpu()
            self.M_out = pop.M_out[0].float().cpu()
            self.W_in = pop.W_in[0].float().cpu()
            self.W_rec = pop.W_rec[0].float().cpu()
            self.W_out = pop.W_out[0].float().cpu()
            self.b_out = pop.b_out[0].float().cpu()
            self.tau_e_init = pop.tau_e[0].float().cpu()


def load_gene_model(path, engine):
    """加载 test7 系最优模型 → (BrainGeneAdapter, cfg, food, steps)"""
    mod = engine.module()
    data = torch.load(path, map_location="cpu", weights_only=False)
    cfg = mod.Config()
    for k, v in (data.get("config", {}) or {}).items():
        if not k.startswith("__"):
            setattr(cfg, k, v)
    cfg.DEVICE = "cpu"      # 可视化单个体，CPU 足够；fp32 与训练评估差异可忽略
    cfg.USE_FP16 = False
    res = mod.load_best_state(path, cfg)
    if res is None:
        raise RuntimeError("test7 系模型加载失败: " + path)
    st, food, steps = res
    pop = mod.GeneStack(cfg, B=1, device=torch.device("cpu"))
    pop.random_init()
    pop.set_individual_from_state(0, st)
    pop.refresh_eff()
    brain = BrainGeneAdapter(pop, cfg, engine)
    # test12 部署形态 = 评估形态：弱连接屏蔽（W_rec 最弱 WEAK_MASK_FRAC 置零），
    # 与 apply_weak_mask 同口径（逐行 kthvalue 阈值），保证可视化动力学与训练评估一致
    frac = float(getattr(cfg, "WEAK_MASK_FRAC", 0.0))
    if frac > 0:
        with torch.no_grad():
            W, M = brain.W_rec, brain.M_rec
            mag = W.abs() * M
            for r in range(W.shape[0]):
                m_r = mag[r][M[r] > 0]
                if m_r.numel() == 0:
                    continue
                kk = int(math.ceil(m_r.numel() * frac))
                if kk <= 0:
                    continue
                thr = torch.kthvalue(m_r, kk).values
                W[r] = torch.where((M[r] > 0) & (mag[r] <= thr),
                                   torch.zeros_like(W[r]), W[r])
    return brain, cfg, float(food), float(steps)


# ============================================================
# 1. 模型加载
# ============================================================
def make_cfg_from_dict(cfg_dict):
    """从 pth 内的 config dict 重建动态 cfg 对象；
    缺失字段用 test5a.Config 类属性补齐。"""
    tmp = type("_RtCfg", (), {})()
    for k, v in cfg_dict.items():
        if not k.startswith("__"):
            setattr(tmp, k, v)
    for k, v in vars(Config).items():
        if not k.startswith("__") and not hasattr(tmp, k):
            setattr(tmp, k, v)
    return tmp


def available_models():
    """模型下拉框枚举：内置 5a/fast/12 + 自动扫描根目录所有 test7/test12 系最优模型"""
    models = [("5a", "5a 最优 (256)"), ("fast", "5d 最优 (256)")]
    if os.path.exists(os.path.join(REPO_ROOT, ENGINE_DEFAULT_FILE["12"])):
        models.append(("12", "12 最优 (256)"))
    scanned = sorted(glob.glob(os.path.join(REPO_ROOT, "test7*_best*.pth"))) + \
              sorted(glob.glob(os.path.join(REPO_ROOT, "test12*_best*.pth")))
    for p in scanned:
        name = os.path.basename(p)
        if name == ENGINE_DEFAULT_FILE.get("12"):
            continue   # 已用内置友好标签 "12" 列出，避免重复
        models.append((name, name))
    return models


def resolve_model_path(key):
    """内置 key（5a/fast/7a/7b）→ 默认模型；其他 → 根目录文件名或绝对路径"""
    if key == "fast":
        return MODEL_FAST_PATH
    if key in ENGINE_DEFAULT_FILE:
        fname = ENGINE_DEFAULT_FILE[key]
        path = os.path.join(REPO_ROOT, fname)
        if os.path.exists(path):
            return path
        # 默认文件不存在 → 回落到扫描到的该引擎首个模型
        cands = sorted(glob.glob(os.path.join(
            REPO_ROOT, os.path.splitext(next(e.module_file for e in ENGINES
                                             if e.name == key))[0] + "*_best*.pth")))
        if cands:
            return cands[0]
        return path
    if key not in ("", "5a"):
        if os.path.isabs(key) and os.path.exists(key):
            return key
        cand = os.path.join(REPO_ROOT, key)
        if os.path.exists(cand):
            return cand
    return DEFAULT_MODEL


def load_model(path):
    """加载 pth 最优模型，返回 (brain, cfg, food, steps)；自动识别 test7 系引擎格式"""
    engine = detect_engine(path)
    if engine is not None:
        return load_gene_model(path, engine)
    data = torch.load(path, map_location="cpu", weights_only=False)
    cfg = make_cfg_from_dict(data.get("config", {}))
    result = einbrain.io.load_best_model_brain(path, cfg)
    if result is None:
        raise RuntimeError("模型加载失败（非 test7 系且非 einbrain 格式）: " + path)
    brain, food, steps = result
    return brain, cfg, float(food), float(steps)


# ============================================================
# 2. 拓扑布局计算（移植自 test5_fast.plot_topology_layered 的布局部分）
#    —— 按「社区 → tau_e」分组 2D 展开 + 碰撞分离，一次性算好供前端复用
# ============================================================
def compute_layout(brain, cfg):
    N = brain.N
    tau = brain.tau_e_init.detach().numpy()
    W_rec_np = brain.W_rec.detach().numpy()
    M_rec_np = brain.M_rec.numpy()
    W_in_np = brain.W_in.detach().numpy()
    M_in_np = brain.M_in.numpy()
    W_out_np = brain.W_out.detach().numpy()
    M_out_np = brain.M_out.numpy()

    # ---- 社区划分（以递归子图为图结构）----
    G = nx.DiGraph()
    for i in range(N):
        G.add_node("Col_" + str(i))
    for i in range(N):
        for j in range(N):
            if M_rec_np[i, j] > 0:
                G.add_edge("Col_" + str(j), "Col_" + str(i))
    if community_louvain is not None:
        partition = community_louvain.best_partition(G.to_undirected())
    else:
        communities = nx.algorithms.community.greedy_modularity_communities(G.to_undirected())
        partition = {}
        for ci, com in enumerate(communities):
            for node in com:
                partition[node] = ci
    comm_of_col = np.array([partition.get("Col_" + str(i), 0) for i in range(N)], dtype=int)

    # ---- 社区 → tau_e 排序 ----
    cols_by_comm = {}
    for i in range(N):
        cols_by_comm.setdefault(int(comm_of_col[i]), []).append(i)
    comms = sorted(cols_by_comm.keys(), key=lambda c: -len(cols_by_comm[c]))
    order = []
    boundaries = []
    comm_meta = []
    for c in comms:
        cols = sorted(cols_by_comm[c], key=lambda i: tau[i])
        boundaries.append(len(order))
        order.extend(cols)
        comm_meta.append({"id": int(c), "size": len(cols)})
    boundaries.append(N)
    order = np.array(order, dtype=int)

    # ---- 社区锚点网格 ----
    n_comms = len(comms)
    ncols = max(1, int(math.ceil(math.sqrt(n_comms))))
    nrows = max(1, int(math.ceil(n_comms / ncols)))
    anchor_step_x, anchor_step_y = 0.90, 0.90
    anchor_x0 = 1.30
    total_height = nrows * anchor_step_y
    anchor = {}
    for idx_blk, cid in enumerate(comms):
        r = idx_blk // ncols
        c = idx_blk % ncols
        anchor[int(cid)] = (anchor_x0 + c * anchor_step_x,
                            total_height - (r + 0.5) * anchor_step_y)

    # ---- 块内 spring_layout + 缩放 + 碰撞分离 ----
    pos_col = {}
    block_radius = 0.34 * anchor_step_x
    MIN_NODE_DIST = 0.12
    for idx_blk, cid in enumerate(comms):
        start, end = boundaries[idx_blk], boundaries[idx_blk + 1]
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
                        sub.add_edge(j, i, weight=abs(float(W_rec_np[i, j])))
            try:
                sub_pos = nx.spring_layout(sub, k=0.6, iterations=150, seed=42 + int(cid))
            except Exception:
                sub_pos = {col: (random.uniform(-1, 1), random.uniform(-1, 1)) for col in block}
        xs = [p[0] for p in sub_pos.values()]
        ys = [p[1] for p in sub_pos.values()]
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        span = max(float(np.max(xs) - np.min(xs)), float(np.max(ys) - np.min(ys)), 1e-6)
        scale = (2.0 * block_radius) / span
        ax0, ay0 = anchor[int(cid)]
        for col in block:
            px, py = sub_pos[col]
            pos_col[col] = (ax0 + (px - cx) * scale, ay0 + (py - cy) * scale)

    for idx_blk, cid in enumerate(comms):
        start, end = boundaries[idx_blk], boundaries[idx_blk + 1]
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
                    dx, dy = xa - xb, ya - yb
                    dist = math.hypot(dx, dy)
                    if dist < MIN_NODE_DIST and dist > 1e-9:
                        push = (MIN_NODE_DIST - dist) / 2.0
                        ux, uy = dx / dist, dy / dist
                        pos_col[a] = (xa + ux * push, ya + uy * push)
                        pos_col[b] = (xb - ux * push, yb - uy * push)
                        moved = True
            if not moved:
                break

    # ---- 输入 / 输出层坐标 ----
    OUT_X = anchor_x0 + ncols * anchor_step_x + 0.5
    y_in = [(j + 1.0) / (brain.obs_dim + 1.0) * total_height for j in range(brain.obs_dim)]
    y_out = [(i + 1.0) / (brain.action_dim + 1.0) * total_height for i in range(brain.action_dim)]

    # ---- 边强度归一化 ----
    def _norm_edges(abs_w):
        w = np.abs(abs_w)
        wmin, wmax = float(w.min()), float(w.max())
        if wmax > wmin:
            return (w - wmin) / (wmax - wmin)
        return np.zeros_like(w)

    W_in_abs = _norm_edges(W_in_np)
    W_out_abs = _norm_edges(W_out_np)

    in_edges = []
    for i in range(N):
        for j in range(brain.obs_dim):
            if M_in_np[i, j] > 0:
                in_edges.append([i, j, float(W_in_abs[i, j])])

    out_edges = []
    for i in range(brain.action_dim):
        for j in range(N):
            if M_out_np[i, j] > 0:
                out_edges.append([i, j, float(W_out_abs[i, j])])

    rec_edges_all = [(j, i, float(W_rec_np[i, j]))
                     for i in range(N) for j in range(N) if M_rec_np[i, j] > 0]
    rec_edges_all.sort(key=lambda e: -abs(e[2]))
    rec_edges_all = rec_edges_all[:TOP_REC_EDGES]
    rec_edges = []
    if rec_edges_all:
        abs_ws = [abs(e[2]) for e in rec_edges_all]
        w_min, w_max = min(abs_ws), max(abs_ws)
        for src, tgt, w in rec_edges_all:
            alpha = (abs(w) - w_min) / (w_max - w_min + 1e-8)
            rec_edges.append([src, tgt, w, float(alpha)])

    # 3D 弹簧图边表：每个神经元只保留自己权重最大的前 10 条输入边
    # （W_rec[i, j] = j→i 的权重，按行取 top-10 → 每柱恰好 10 条、无孤立柱）。
    # alpha 用秩（CDF）：权重→弹性换算的归一档位，重尾分布下不受幅值压挤。
    edges_3d = []
    for i in range(N):
        row = [(abs(float(W_rec_np[i, j])), j) for j in range(N) if M_rec_np[i, j] > 0]
        if not row:
            continue
        row.sort(reverse=True)
        for w_abs, j in row[:REC3D_TOP_PER_NEURON]:
            edges_3d.append((j, i, float(W_rec_np[i, j])))
    edges_3d.sort(key=lambda e: -abs(e[2]))
    rec_edges_3d = []
    n_top = len(edges_3d)
    for i, (src, tgt, w) in enumerate(edges_3d):
        alpha = 1.0 - i / (n_top - 1) if n_top > 1 else 1.0
        rec_edges_3d.append([src, tgt, round(w, 4), round(alpha, 3)])

    return {
        "col_x": [float(pos_col[i][0]) for i in range(N)],
        "col_y": [float(pos_col[i][1]) for i in range(N)],
        "in_y": [float(v) for v in y_in],
        "out_y": [float(v) for v in y_out],
        "col_order": order.tolist(),
        "communities": [
            {"id": m["id"], "size": m["size"], "color": idx % 10,
             "x": float(anchor[m["id"]][0]), "y": float(anchor[m["id"]][1])}
            for idx, m in enumerate(comm_meta)
        ],
        "in_edges": in_edges,
        "out_edges": out_edges,
        "rec_edges": rec_edges,
        "rec_edges_3d": rec_edges_3d,
        "W_in_signed": [[i, j, round(float(W_in_np[i, j]), 4)]
                        for i in range(N) for j in range(brain.obs_dim)
                        if M_in_np[i, j] > 0],
        "W_out_signed": [[i, j, round(float(W_out_np[i, j]), 4)]
                         for i in range(brain.action_dim) for j in range(N)
                         if M_out_np[i, j] > 0],
        "tau": [float(v) for v in tau],
        "in_degree": [int(v) for v in brain.M_rec.sum(dim=0).numpy()],
        "range": {"x0": -1.1, "x1": OUT_X + 0.9, "y0": -0.8, "y1": total_height + 0.8},
    }


# ============================================================
# 3. 全局状态
# ============================================================
class GlobalState:
    def __init__(self, model_path, model_key=None):
        self.lock = threading.RLock()
        self.cond = threading.Condition(self.lock)
        self.model_path = os.path.abspath(model_path)
        self.model_key = model_key or os.path.basename(model_path)
        self.brain = None
        self.cfg = None
        self.model_info = {}
        self.layout = None
        self.init_payload = b""
        self.version = 0
        self.frame_id = 0
        self.latest_frame = None
        self.paused = False
        self.request_new = False
        self.speed = 1.0
        self.episode = 0
        self.load_state()

    def load_state(self):
        brain, cfg, food, steps = load_model(self.model_path)
        with self.lock:
            self.brain = brain
            self.cfg = cfg
            self.layout = compute_layout(brain, cfg)
            self.model_info = {
                "model": os.path.basename(self.model_path),
                "model_key": self.model_key,
                "engine": getattr(self.brain, "engine", None) and self.brain.engine.name or "einbrain",
                "is7a": bool(getattr(self.brain, "is_gene_engine", False)),
                "N": int(brain.N),
                "OBS": int(cfg.OBS_DIM),
                "ACTION": int(cfg.ACTION_DIM),
                "food": food,
                "steps": steps,
                "GRID": int(cfg.GRID_SIZE),
                "FRAME_RATE": int(getattr(cfg, "FRAME_RATE", 5)),
                "INPUT_DECAY": float(getattr(cfg, "INPUT_DECAY", 0.9)),
                "tau_min": float(getattr(cfg, "TAU_E_MIN", 0.001)),
                "tau_max": float(getattr(cfg, "TAU_E_MAX", 2.0)),
            }
            payload = {
                "type": "init",
                "layout": self.layout,
                "meta": self.model_info,
                "models": available_models(),
                "episode": self.episode,
            }
            self.init_payload = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.version += 1
            self.request_new = True

    def load_state_raw(self, path, key=None):
        """无条件重载模型（切换模型时）"""
        self.model_path = os.path.abspath(path)
        self.model_key = key or os.path.basename(path)
        self.load_state()
        return True


# ============================================================
# 4. 游戏线程
# ============================================================
def _round_list(values, nd=4):
    """浮点列表压缩精度，显著减小 SSE 帧体积"""
    return [round(float(v), nd) for v in values]


def build_frame(state, brain, cfg, obs, action, avg_logits, E, I, env, ate, done, steps, score):
    with torch.no_grad():
        E_np = _round_list(E.detach().numpy())
        I_np = _round_list(I.detach().numpy())
        exc = _round_list(brain.hormone_excit.detach().numpy())
        inh = _round_list(brain.hormone_inhib.detach().numpy())
        eff_tau = (brain.tau_e_init +
                   cfg.SHORT_TERM_GAIN * brain.short_term_state +
                   cfg.EXCIT_HORMONE_GAIN * brain.hormone_excit -
                   cfg.INHIB_HORMONE_GAIN * brain.hormone_inhib)
        tau_eff = _round_list(torch.clamp(eff_tau, cfg.TAU_E_MIN, cfg.TAU_E_MAX).detach().numpy())
        fatigue = _round_list(brain.consecutive_counts.detach().numpy())
        logits = _round_list(avg_logits.detach().numpy())
    return {
        "type": "frame",
        "obs": obs.tolist() if isinstance(obs, np.ndarray) else list(obs),
        "action": int(action),
        "logits": logits,
        "fatigue": fatigue,
        "E": E_np,
        "I": I_np,
        "hormone_ex": exc,
        "hormone_in": inh,
        "tau": tau_eff,
        "body": [[int(seg[0]), int(seg[1])] for seg in env.body],
        "food": [int(env.food[0]), int(env.food[1])],
        "dir": [int(env.dir[0]), int(env.dir[1])],
        "score": int(score),
        "steps": int(steps),
        "done": bool(done),
        "episode": int(state.episode),
    }


def _run_episode(state, brain, cfg, episode):
    """运行一局完整游戏；返回 True=正常结束 / False=被中断（新局/模型切换）"""
    env = SnakeEnv(grid_size=cfg.GRID_SIZE)
    brain.reset_runtime()
    obs = env.reset()
    E = torch.zeros(brain.N)
    I = torch.zeros(brain.N)
    done = False
    steps = 0
    score = 0

    while not done:
        with state.cond:
            if state.version != episode[0] or state.request_new:
                state.request_new = False
                return False
            if state.paused:
                state.cond.wait_for(
                    lambda: not state.paused or state.request_new or state.version != episode[0],
                    timeout=0.2)
                if state.request_new or state.version != episode[0]:
                    state.request_new = False
                    return False
                # 暂停未解除则继续等待，不推进游戏
                if state.paused:
                    continue
            speed = state.speed

        action, avg_logits, E, I = deliberate_action(brain, obs, E, I)
        brain.update_fatigue(action)

        next_obs, ate, done, _truncated = env.step(action)
        if ate:
            score += 1
        steps += 1

        frame = build_frame(state, brain, cfg, obs, action, avg_logits,
                            E, I, env, ate, done, steps, score)
        with state.cond:
            state.latest_frame = frame
            state.frame_id += 1
            state.cond.notify_all()

        obs = next_obs

        # 步间延迟 = 0.12s / 速度倍率（约等于原版 plt.pause(0.1)）
        time.sleep(0.12 / speed)

    # 局结束：短暂停留，让前端显示完成态
    time.sleep(0.35)
    return True


def _run_episode_gene(state, brain, cfg, episode):
    """test7 系对局：用对应引擎自己的 BatchedSnakeEnv + E-I 动力学（CPU 单个体），
    保证与训练评估的动力学完全一致；返回 True=正常结束 / False=被中断"""
    mod = brain.engine.module()
    pop = brain.pop
    t7cfg = brain.t7cfg
    dev = pop.device
    N = brain.N
    env = mod.BatchedSnakeEnv(t7cfg, 1, dev)
    E = torch.zeros(1, N, device=dev)
    I = torch.zeros(1, N, device=dev)
    st = torch.zeros(1, N, device=dev)
    # 疲劳/压力状态形态按引擎接口自适应：
    #   7a/7b: cts [B,A] 逐动作连续计数（update_fatigue(cts, action)）
    #   7c:    press [B] 转向压力标量（forward_batch 第 6 参数名为 press）
    # 疲劳衰减与引擎配置一致（默认 0.7 会弱化 7g 的 0.9 疲劳，显示行为失真）
    _fat_decay = float(getattr(t7cfg, "FATIGUE_TURN_DECAY", 0.7))
    import inspect as _inspect
    _fat_param = list(_inspect.signature(mod.forward_batch).parameters)[5]
    press_mode = (_fat_param == "press")
    cts = (torch.zeros(1, device=dev) if press_mode
           else torch.zeros(1, brain.action_dim, device=dev))
    done = False
    steps = 0
    score = 0
    zeros_hormone = [0.0] * N

    while not done:
        with state.cond:
            if state.version != episode[0] or state.request_new:
                state.request_new = False
                return False
            if state.paused:
                state.cond.wait_for(
                    lambda: not state.paused or state.request_new or state.version != episode[0],
                    timeout=0.2)
                if state.request_new or state.version != episode[0]:
                    state.request_new = False
                    return False
                if state.paused:
                    continue
            speed = state.speed

        with torch.no_grad():
            obs = env.obs()
            # K 倍帧率思考（与 deliberate_batch 相同流程，额外保留平均 logits 供展示）
            K = t7cfg.FRAME_RATE
            logits_sum = None
            for k in range(K):
                o = obs * (t7cfg.INPUT_DECAY ** k)
                logits, E, I, st = mod.forward_batch(pop, o, E, I, st, cts, t7cfg)
                logits_sum = logits if logits_sum is None else logits_sum + logits
            avg_logits = logits_sum / K
            action_t = torch.argmax(logits_sum, dim=1)
            action = int(action_t[0])
            if press_mode:
                cts = mod.update_fatigue(cts, action_t, decay=_fat_decay)
            else:
                cts = mod.update_fatigue(cts, action_t)
            env.step(action_t)
            ate = bool(env.ate[0])
            done = not bool(env.alive[0])
            tau_eff = (pop.tau_e + t7cfg.SHORT_TERM_GAIN * st).clamp(
                t7cfg.TAU_E_MIN, t7cfg.TAU_E_MAX)
            # press 模式：转向压力只作用于左右转 logits，前端按 [Fwd,Left,Right] 显示
            fatigue_view = ([0.0, float(cts[0]), float(cts[0])] if press_mode
                            else _round_list(cts[0]))
            frame = {
                "type": "frame",
                "obs": obs[0].tolist(),
                "action": action,
                "logits": _round_list(avg_logits[0]),
                "fatigue": fatigue_view,
                "E": _round_list(E[0]),
                "I": _round_list(I[0]),
                "hormone_ex": zeros_hormone,   # test7a 无激素支路，填零保持帧格式兼容
                "hormone_in": zeros_hormone,
                "tau": _round_list(tau_eff[0]),
                "body": [[int(seg[0]), int(seg[1])] for seg in env.body[0, :env.body_len[0]]],
                "food": [int(env.food[0, 0]), int(env.food[0, 1])],
                "dir": [int(env.DIRS[env.dir_idx[0]][0]), int(env.DIRS[env.dir_idx[0]][1])],
                "score": int(score),
                "steps": int(steps),
                "done": bool(done),
                "episode": int(state.episode),
            }
        if ate:
            score += 1
            frame["score"] = int(score)
        steps += 1

        with state.cond:
            state.latest_frame = frame
            state.frame_id += 1
            state.cond.notify_all()

        time.sleep(0.12 / speed)

    time.sleep(0.35)
    return True


def game_loop(state):
    """后台常驻对局（看门狗）：单局异常不会杀死线程，自动尝试下一局；
    响应暂停 / 新局 / 模型切换"""
    my_ver = -1
    my_brain = None
    my_cfg = None

    while True:
        with state.cond:
            if my_ver != state.version:
                my_ver = state.version
                my_brain = state.brain
                my_cfg = state.cfg
            state.episode += 1
            episode_marker = [state.version]
            episode = state.episode

        try:
            if getattr(my_brain, "is_gene_engine", False):
                ok = _run_episode_gene(state, my_brain, my_cfg, episode_marker)
            else:
                ok = _run_episode(state, my_brain, my_cfg, episode_marker)
        except Exception:
            traceback.print_exc()
            print("[GameLoop] 单局异常，1 秒后重开新局")
            time.sleep(1.0)
            continue

        if not ok:
            # 收到新局 / 模型切换请求：外层循环立即重新同步
            time.sleep(0.05)


# ============================================================
# 5. HTTP 服务器
# ============================================================
class QuietThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)):
            return   # 客户端断开（Windows 常见 WinError 10053），静音
        super().handle_error(request, client_address)


class Handler(BaseHTTPRequestHandler):
    server_version = "BrainVisualizer/1.0"

    # ---- 工具 ----
    def send_json(self, obj, code=200):
        body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def serve_static(self, name):
        path = os.path.join(STATIC_DIR, name)
        if not os.path.exists(path):
            self.send_json({"error": "not found"}, 404)
            return
        with open(path, "rb") as f:
            data = f.read()
        ctype = "text/html; charset=utf-8" if name.endswith(".html") else \
                "application/javascript; charset=utf-8" if name.endswith(".js") else \
                "text/css; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        # 禁止缓存，避免浏览器沿用旧版 HTML/JS 导致连接异常
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(data)

    # ---- 路由 ----
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html"):
                self.serve_static("index.html")
            elif path == "/visualizer.js":
                self.serve_static("visualizer.js")
            elif path == "/api/init":
                self.handle_init(qs)
            elif path == "/api/control":
                self.handle_control(qs)
            elif path == "/stream":
                self.handle_stream()
            else:
                self.send_json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def handle_init(self, qs):
        model_key = (qs.get("model", [""])[0] or "").strip()
        with state.cond:
            # 带 ?model= 才切换模型；裸 /api/init 只返回当前状态（避免意外重置回默认模型）
            if model_key:
                path = resolve_model_path(model_key)
                if os.path.abspath(path) != state.model_path:
                    state.load_state_raw(path, key=model_key)
            payload = state.init_payload
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def handle_control(self, qs):
        action = (qs.get("action", [""])[0] or "").strip()
        with state.cond:
            if action == "pause":
                state.paused = True
            elif action == "resume":
                state.paused = False
                state.cond.notify_all()
            elif action == "new":
                state.request_new = True
                state.cond.notify_all()
            if "speed" in qs:
                try:
                    state.speed = max(0.2, min(8.0, float(qs["speed"][0])))
                except ValueError:
                    pass
        self.send_json({"ok": True, "paused": state.paused, "speed": state.speed})

    def handle_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        with state.cond:
            payload_init = state.init_payload
            last_sent = state.frame_id
            last_ver = state.version

        try:
            self.wfile.write(b"data: " + payload_init + b"\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

        try:
            while True:
                with state.cond:
                    # 等待新帧或模型切换；timeout 保证能感知连接断开
                    state.cond.wait_for(
                        lambda: state.frame_id != last_sent or state.version != last_ver,
                        timeout=1.0)
                    if state.frame_id != last_sent:
                        frame = state.latest_frame
                        last_sent = state.frame_id
                    elif state.version != last_ver:
                        payload = state.init_payload
                        last_ver = state.version
                        try:
                            self.wfile.write(b"data: " + payload + b"\n\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError, OSError):
                            return
                        continue
                    else:
                        continue
                payload = json.dumps(frame, separators=(",", ":")).encode("utf-8")
                self.wfile.write(b"data: " + payload + b"\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def log_message(self, fmt, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="贪吃蛇脑活动可视化（test5a / test7a）")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=DEFAULT_MODEL_KEY,
                        help="7a=test7a_v5最优 / 5a / fast，或根目录下任意 test7a*_best*.pth 文件名 / 绝对路径")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    path = resolve_model_path(args.model)
    print("[Init] 加载最优模型: " + os.path.basename(path))
    t0 = time.perf_counter()
    state.__init__(path, model_key=args.model)  # 重新初始化全局状态
    print("[Init] 模型与拓扑布局计算完成: {:.1f}s | N={} OBS={} Food={} Steps={}".format(
        time.perf_counter() - t0, state.model_info["N"], state.model_info["OBS"],
        state.model_info["food"], state.model_info["steps"]))

    threading.Thread(target=game_loop, args=(state,), daemon=True).start()

    # 端口自动避让
    httpd = None
    port = args.port
    for _ in range(20):
        try:
            httpd = QuietThreadingHTTPServer(("127.0.0.1", port), Handler)
            httpd.daemon_threads = True
            break
        except OSError:
            port += 1
    if httpd is None:
        print("[Error] 无法绑定端口")
        sys.exit(1)

    url = "http://127.0.0.1:" + str(port)
    print("")
    print("可视化面板: " + url)
    print("  · 模型切换: " + url + "/?model=5a 或 /?model=fast（页面加载时生效）")
    print("  · Ctrl+C 退出")
    print("")

    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("")
        print("已退出")


# 全局状态（模块级单例）
state = GlobalState(DEFAULT_MODEL, model_key="5a")

if __name__ == "__main__":
    main()