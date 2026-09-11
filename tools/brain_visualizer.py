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


def find_model_file(fname):
    """模型文件查找：根目录 → artifacts/<世代>/ → experiments/test7_series/（产物归档后统一兼容）"""
    p = os.path.join(REPO_ROOT, fname)
    if os.path.exists(p):
        return p
    hits = sorted(glob.glob(os.path.join(REPO_ROOT, "artifacts", "*", fname)))
    if hits:
        return hits[0]
    hit = os.path.join(REPO_ROOT, "experiments", "test7_series", fname)
    return hit if os.path.exists(hit) else p


def scan_model_files(pattern):
    """按通配符扫描根目录 + artifacts/*/ + experiments/test7_series/ 下的模型"""
    hits = glob.glob(os.path.join(REPO_ROOT, pattern))
    hits += glob.glob(os.path.join(REPO_ROOT, "artifacts", "*", pattern))
    hits += glob.glob(os.path.join(REPO_ROOT, "experiments", "test7_series", pattern))
    return sorted(set(hits))


# 默认加载：优先最新版本的引擎模型（16b > 12 > 7g > 7d > 7c > 7b > 7a > 5a）
DEFAULT_MODEL_KEY = next(
    (key for key, fname in [("16b", "test16b_simp_best_model.pth"),
                            ("12", "test12_econ_best_model.pth"),
                            ("7g", "test7g_econ_best_model.pth"),
                            ("7d", "test7d_econ_best_model.pth"),
                            ("7c", "test7c_latest_gen_best.pth"),
                            ("7b", "test7b_best_model.pth"),
                            ("7a", "test7a_v5a_best_model.pth")]
     if os.path.exists(find_model_file(fname))), "5a")

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
#       test16 系列（sparse1）也含 OBS_ENC_VERSION，必须排在 test12 之前，
#       用其专有字段 BRAIN_VERSION=='sparse1' 区分。
ENGINES = [
    Engine("16b", "experiments/test16_series/test16b.py",
           lambda c: c.get("BRAIN_VERSION") == "sparse1"),
    Engine("12", "experiments/test12/test12.py",
           lambda c: "OBS_ENC_VERSION" in c or "ISLAND_PENALTY" in c),
    Engine("7g", "experiments/test7_series/test7g.py", lambda c: "OBS_MANHATTAN" in c),
    Engine("7d", "experiments/test7_series/test7d.py",
           lambda c: "TURN_COST" in c or "TURN_PENALTY" in c or
                     "ONE_SIDED_TURN_DEATH" in c),
    Engine("7c", "experiments/test7_series/test7c.py",
           lambda c: "FATIGUE_TURN_DECAY" in c or "FATIGUE_TURN_GAIN" in c),
    Engine("7b", "experiments/test7_series/test7b.py",
           lambda c: "STARVE_SLOPE" in c or "FOOD_EFF_WEIGHT" in c),
    Engine("7a", "experiments/test7_series/test7a.py",
           lambda c: "OBS_MODE" in c or "EVAL_BATCH" in c),
]

# 内置快捷 key -> 默认模型文件名（不存在时回落到扫描到的第一个）
ENGINE_DEFAULT_FILE = {
    "16b": "test16b_simp_best_model.pth",
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
    """把 test7/16 系 GeneStack 单个体包装成 compute_layout 所需的 brain 接口。

    16 系列（sparse1）无稠密 M_rec/W_rec，只有 rec_idx/rec_w 槽位：
    scatter 稠密化（重复源权重叠加，与 gather-乘-归约前向数学等价），
    布局/边提取/弱连接屏蔽与稠密血统同口径。"""
    is_gene_engine = True

    def __init__(self, pop, t7cfg, engine):
        self.pop = pop
        self.t7cfg = t7cfg
        self.engine = engine
        self.N = int(pop.N)
        self.obs_dim = int(pop.O)
        self.action_dim = int(pop.A)
        self.sparse_rec = getattr(pop, "rec_idx", None) is not None
        self.rec_fanin = int(getattr(pop, "K", 0)) if self.sparse_rec else None
        with torch.no_grad():
            self.M_in = pop.M_in[0].float().cpu()
            self.M_out = pop.M_out[0].float().cpu()
            self.W_in = pop.W_in[0].float().cpu()
            self.W_out = pop.W_out[0].float().cpu()
            self.b_out = pop.b_out[0].float().cpu()
            self.tau_e_init = pop.tau_e[0].float().cpu()
            if self.sparse_rec:
                idx = pop.rec_idx[0].long().cpu().unsqueeze(0)   # [1,N,K]
                w = pop.rec_w[0].float().cpu().unsqueeze(0)
                N = self.N
                M_rec = torch.zeros(1, N, N)
                M_rec.scatter_(2, idx, 1.0)
                W_rec = torch.zeros(1, N, N)
                W_rec.scatter_add_(2, idx, w)
                self.M_rec = M_rec[0]
                self.W_rec = W_rec[0]
            else:
                self.M_rec = pop.M_rec[0].float().cpu()
                self.W_rec = pop.W_rec[0].float().cpu()


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
    """模型下拉框枚举：内置 5a/fast/16b/12 + 自动扫描根目录/artifacts/*/test7/12/16 系最优模型"""
    models = [("5a", "5a 最优 (256)"), ("fast", "5d 最优 (256)")]
    for key, label in [("16b", "16b 最优 (1024 稀疏)"), ("12", "12 最优 (256)")]:
        if os.path.exists(find_model_file(ENGINE_DEFAULT_FILE[key])):
            models.append((key, label))
    scanned = scan_model_files("test16*_best*.pth") + \
              scan_model_files("test7*_best*.pth") + \
              scan_model_files("test12*_best*.pth") + \
              scan_model_files("16c_cheat7b_*model.pth")
    for p in scanned:
        name = os.path.basename(p)
        if name in (ENGINE_DEFAULT_FILE.get("16b"), ENGINE_DEFAULT_FILE.get("12")):
            continue   # 已用内置友好标签列出，避免重复
        models.append((name, name))
    return models


def resolve_model_path(key):
    """内置 key（5a/fast/7a/7b）→ 默认模型；其他 → 根目录/artifacts 文件名或绝对路径"""
    if key == "fast":
        return MODEL_FAST_PATH
    if key in ENGINE_DEFAULT_FILE:
        path = find_model_file(ENGINE_DEFAULT_FILE[key])
        if os.path.exists(path):
            return path
        # 默认文件不存在 → 回落到扫描到的该引擎首个模型
        prefix = os.path.splitext(next(e.module_file for e in ENGINES
                                       if e.name == key))[0]
        cands = scan_model_files(prefix + "*_best*.pth") + \
                scan_model_files(prefix + "*model*.pth")
        if cands:
            return cands[0]
        return path
    if key not in ("", "5a"):
        if os.path.isabs(key) and os.path.exists(key):
            return key
        cand = find_model_file(key)
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

    # ---- 社区划分（以递归子图为图结构；nonzero 向量化，N=1024 可用）----
    rt, ct = np.nonzero(M_rec_np)
    G = nx.DiGraph()
    for i in range(N):
        G.add_node("Col_" + str(i))
    G.add_edges_from(("Col_" + str(j), "Col_" + str(i)) for i, j in zip(rt.tolist(), ct.tolist()))
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
    # 块内 spring（同块 rec 边 nonzero 向量化）
    for idx_blk, cid in enumerate(comms):
        start, end = boundaries[idx_blk], boundaries[idx_blk + 1]
        block = [int(col) for col in order[start:end]]
        if len(block) == 1:
            sub_pos = {block[0]: (0.0, 0.0)}
        else:
            bset = np.zeros(N, dtype=bool)
            bset[block] = True
            mask_blk = M_rec_np[np.ix_(bset, bset)]
            sub_rt, sub_ct = np.nonzero(mask_blk)
            sub = nx.DiGraph()
            for col in block:
                sub.add_node(col)
            cols_arr = np.array(block)
            sub.add_weighted_edges_from(
                ((int(cols_arr[j]), int(cols_arr[i]),
                  abs(float(W_rec_np[cols_arr[i], cols_arr[j]])))
                 for i, j in zip(sub_rt.tolist(), sub_ct.tolist())))
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

    # rec 边提取向量化（N=1024 时 N² Python 循环不可用）：取 |W| top-K
    rec_w_all = W_rec_np[rt, ct]
    order_by_w = np.argsort(-np.abs(rec_w_all))[:TOP_REC_EDGES]
    rec_edges = []
    if len(order_by_w):
        abs_ws = np.abs(rec_w_all[order_by_w])
        w_min, w_max = float(abs_ws.min()), float(abs_ws.max())
        for kk in order_by_w.tolist():
            src, tgt, w = int(ct[kk]), int(rt[kk]), float(rec_w_all[kk])
            alpha = (abs(w) - w_min) / (w_max - w_min + 1e-8)
            rec_edges.append([src, tgt, w, float(alpha)])

    # 3D 弹簧图边表：每个神经元只保留自己权重最大的前 10 条输入边
    # （W_rec[i, j] = j→i 的权重，按行取 top-10 → 每柱恰好 10 条、无孤立柱）。
    # alpha 用秩（CDF）：权重→弹性换算的归一档位，重尾分布下不受幅值压挤。
    # torch.topk 逐行向量化；行内实边不足 K 时补位值=0（跳过）。
    wabs_rows = torch.from_numpy(np.abs(W_rec_np) * M_rec_np)
    k3 = min(REC3D_TOP_PER_NEURON, N)
    _vals, _cols = torch.topk(wabs_rows, k=k3, dim=1)
    edges_3d = []
    vals_np = _vals.numpy()
    cols_np = _cols.numpy()
    for i in range(N):
        for k in range(k3):
            if vals_np[i, k] <= 0:
                break
            j = int(cols_np[i, k])
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
        self.auto_next = True       # 局终自动开下一局（False=等手动 🔄）
        self.stepping = None        # None=连续 | 'game'=按游戏步单步 | 'net'=按网络更新单步
        self.step_pending = 0       # 已授予未消费的单步许可数
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
                "sparse": bool(getattr(brain, "sparse_rec", False)),
                "sparse_fanin": getattr(brain, "rec_fanin", None),
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


def _grant_step(state, episode, unit):
    """单步模式闸口：当前模式==unit 时阻塞等待一次步进许可。

    - 返回 False=对局被打断（新局/切模型）；
    - 模式已切换（如点了另一单位/暂停清除）→ 直接放行，许可留给新模式的闸口，
      防止旧闸口吞掉新模式的许可或死等。
    许可来自 /api/control?action=step（每次点击 +1）；暂停不阻塞单步
    （调试语义：暂停后仍可单步，点步进即解除暂停）。"""
    while True:
        with state.cond:
            if state.version != episode[0] or state.request_new:
                state.request_new = False
                return False
            if state.stepping != unit:
                return True
            if state.step_pending > 0:
                state.step_pending -= 1
                return True
            state.cond.wait_for(
                lambda: (state.version != episode[0] or state.request_new
                         or state.step_pending > 0 or state.stepping != unit),
                timeout=0.5)


def _push_frame(state, frame):
    with state.cond:
        state.latest_frame = frame
        state.frame_id += 1
        state.cond.notify_all()


def build_frame(state, brain, cfg, obs, action, avg_logits, E, I, env, ate, done, steps, score,
                body=None, food=None, dirv=None):
    """构造 SSE 帧。body/food/dirv 给定时用之——必须是与 obs 同一时刻（env.step 之前）
    的快照；否则 obs 与盘面错位一步，蛇一转弯输入通道就和画面对不上。"""
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
    if body is None:
        body = env.body
    if food is None:
        food = env.food
    if dirv is None:
        dirv = env.dir
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
        "body": [[int(seg[0]), int(seg[1])] for seg in body],
        "food": [int(food[0]), int(food[1])],
        "dir": [int(dirv[0]), int(dirv[1])],
        "score": int(score),
        "steps": int(steps),
        "done": bool(done),
        "episode": int(state.episode),
        "paused": bool(state.paused),
        "stepping": state.stepping,
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

        # 单步模式（einbrain 标量路径只有游戏步粒度；net 点击按游戏步执行）
        if state.stepping is not None and not _grant_step(state, episode, "game"):
            return False
        action, avg_logits, E, I = deliberate_action(brain, obs, E, I)
        brain.update_fatigue(action)

        # 与 obs 同时刻的盘面快照（env.step 会原地改 body/food/dir）
        pre_body, pre_food, pre_dir = list(env.body), env.food, env.dir
        next_obs, ate, done, _truncated = env.step(action)
        if ate:
            score += 1
        steps += 1

        frame = build_frame(state, brain, cfg, obs, action, avg_logits,
                            E, I, env, ate, done, steps, score,
                            body=pre_body, food=pre_food, dirv=pre_dir)
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

    pre_body = pre_food = pre_dirv = None   # 与 obs 同时刻的盘面快照（step 前取）

    def build_frame(action, avg_logits, E, I, st, ate, done):
        """当前网络/环境状态 → SSE 帧（游戏步与网络步共用）。
        盘面用 pre_* 快照（与 obs 同一时刻）：env.step 会原地改 body/food/dir，
        直接读会让盘面比观测超前一步，蛇一转弯输入通道就与画面对不上。"""
        with torch.no_grad():
            tau_eff = (pop.tau_e + t7cfg.SHORT_TERM_GAIN * st).clamp(
                t7cfg.TAU_E_MIN, t7cfg.TAU_E_MAX)
            fatigue_view = ([0.0, float(cts[0]), float(cts[0])] if press_mode
                            else _round_list(cts[0]))
            return {
                "type": "frame",
                "obs": obs[0].tolist(),
                "action": int(action),
                "logits": _round_list(avg_logits),
                "fatigue": fatigue_view,
                "E": _round_list(E[0]),
                "I": _round_list(I[0]),
                "hormone_ex": zeros_hormone,   # test7 系无激素支路，填零保持帧格式兼容
                "hormone_in": zeros_hormone,
                "tau": _round_list(tau_eff[0]),
                "body": [[int(seg[0]), int(seg[1])] for seg in pre_body],
                "food": [int(pre_food[0]), int(pre_food[1])],
                "dir": [int(pre_dirv[0]), int(pre_dirv[1])],
                "score": int(score),
                "steps": int(steps),
                "done": bool(done),
                "episode": int(state.episode),
                "paused": bool(state.paused),
                "stepping": state.stepping,
            }

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
            # 与 obs 同时刻的盘面快照（env.step 原地修改前取值）
            pre_body = env.body[0, :env.body_len[0]].tolist()
            pre_food = env.food[0].tolist()
            pre_dirv = env.DIRS[env.dir_idx[0]].tolist()
            # K 倍帧率思考（与 deliberate_batch 相同流程，额外保留平均 logits 供展示）。
            # 网络步进模式：每次内部更新前等一次步进许可，并把中间态（部分 logits
            # 的倾向动作、已更新 E/I/τ）即时推帧——棋盘不动，网络状态逐次演化。
            K = t7cfg.FRAME_RATE
            logits_sum = None
            for k in range(K):
                if state.stepping == "net" and not _grant_step(state, episode, "net"):
                    return False
                o = obs * (t7cfg.INPUT_DECAY ** k)
                logits, E, I, st = mod.forward_batch(pop, o, E, I, st, cts, t7cfg)
                logits_sum = logits if logits_sum is None else logits_sum + logits
                if state.stepping == "net":
                    partial = logits_sum / (k + 1)
                    _push_frame(state, build_frame(
                        torch.argmax(logits_sum, dim=1)[0], partial[0],
                        E, I, st, False, False))
            avg_logits = logits_sum / K
            action_t = torch.argmax(logits_sum, dim=1)
            action = int(action_t[0])
            if press_mode:
                cts = mod.update_fatigue(cts, action_t, decay=_fat_decay)
            else:
                cts = mod.update_fatigue(cts, action_t)
            # 游戏步进模式：K 次网络更新完成后、环境推进前等一次许可
            if state.stepping == "game" and not _grant_step(state, episode, "game"):
                return False
            env.step(action_t)
            ate = bool(env.ate[0])
            done = not bool(env.alive[0])
            # 先更新计数再构帧：单步模式下步数/得分需即时反映本步（否则滞后一步）
            if ate:
                score += 1
            steps += 1
            frame = build_frame(action, avg_logits[0], E, I, st, ate, done)

        _push_frame(state, frame)

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
            continue

        # 局自然结束：自动下一局关闭时挂起等待（🔄 新局或切换模型唤醒）
        with state.cond:
            while (not state.auto_next and my_ver == state.version
                   and not state.request_new):
                state.cond.wait_for(
                    lambda: (state.auto_next or state.request_new
                             or state.version != my_ver),
                    timeout=0.5)
            state.request_new = False


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
                state.stepping = None      # 暂停即退出单步模式
                state.step_pending = 0
            elif action == "resume":
                state.paused = False
                state.stepping = None      # 继续即回到连续运行
                state.step_pending = 0
                state.cond.notify_all()
            elif action == "new":
                state.request_new = True
                state.cond.notify_all()
            elif action == "autonext":
                # 局终自动开下一局开关（on=1/0；缺省取反）
                if "on" in qs:
                    state.auto_next = qs["on"][0] in ("1", "true", "True")
                else:
                    state.auto_next = not state.auto_next
                state.cond.notify_all()
            elif action == "step":
                # 单步推进一次：unit=game（一个游戏步=K 次网络更新）| net（一次网络更新）
                unit = (qs.get("unit", ["game"])[0] or "game").strip()
                unit = unit if unit in ("game", "net") else "game"
                if state.stepping != unit:
                    state.step_pending = 0   # 换单位丢弃旧模式的未消费许可
                state.stepping = unit
                state.step_pending += 1
                state.paused = False       # 单步隐含解除暂停（调试语义）
                state.cond.notify_all()
            elif action == "stepoff":
                state.stepping = None
                state.step_pending = 0
                state.cond.notify_all()
            if "speed" in qs:
                try:
                    state.speed = max(0.2, min(8.0, float(qs["speed"][0])))
                except ValueError:
                    pass
        self.send_json({"ok": True, "paused": state.paused, "speed": state.speed,
                        "auto_next": state.auto_next, "stepping": state.stepping})

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