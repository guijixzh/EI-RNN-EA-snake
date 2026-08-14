"""GPU 全并行进化（test7 语义，无激素 EI-RNN）。

整个种群堆叠为一批 GPU 张量（GeneStack [POP, N, N] 等），交叉/变异全向量化，
一整代种群并行跑在同一批向量化环境里。删除激素支路（test5d 默认激素权重恒 0，
删除后动力学完全一致）。语义与 test5d 对齐：K 帧思考、三元组筛选、
交替冻结、单侧转弯判死、饥饿截断、动作疲劳、短期 tau 调制。
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch

from .evolve import _selection_key, _freeze_active_groups, _dynamic_mutation_rates
from .io import model_path

# ==================== 基础工具 ====================


def _resolve_device(cfg):
    if cfg.DEVICE != 'auto':
        return torch.device(cfg.DEVICE)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _auto_eval_batch(cfg, device):
    if cfg.EVAL_BATCH > 0:
        return min(cfg.EVAL_BATCH, cfg.POP_SIZE)
    if device.type != 'cuda':
        return cfg.POP_SIZE
    n = cfg.NUM_COLUMNS
    try:
        total = torch.cuda.get_device_properties(device).total_memory
    except Exception:
        return cfg.POP_SIZE
    per_ind = n * n * 16.0
    per_ind += n * cfg.OBS_DIM * 6.0
    batch = int(total * cfg.EVAL_MEM_FRAC / per_ind)
    return max(32, min(batch, cfg.POP_SIZE))


# ==================== 种群基因组张量栈 ====================


class GeneStack:
    """整个种群的基因型/表现型堆叠张量。

    形状约定（B=个体数, N=柱数, O=观测维, A=动作维）：
      M_in [B,N,O]  M_rec [B,N,N]  M_out [B,A,N]
      W_in [B,N,O]  W_rec [B,N,N]  W_out [B,A,N]
      b_out [B,A]   tau_e [B,N]    w_ei / w_ie [B,N]
    """

    GENES = ['M_in', 'M_rec', 'M_out',
             'W_in', 'W_rec', 'W_out', 'b_out',
             'tau_e', 'w_ei', 'w_ie']
    G1_WEIGHTS = ['W_in', 'W_rec', 'W_out', 'b_out']
    G1_MASKS = ['M_in', 'M_rec', 'M_out']
    G2_TENSORS = ['tau_e', 'w_ei', 'w_ie']
    EFF = ['W_in_eff', 'W_rec_eff', 'W_out_eff']

    def __init__(self, cfg, B=None, device=None):
        self.cfg = cfg
        self.N = cfg.NUM_COLUMNS
        self.O = cfg.OBS_DIM
        self.A = cfg.ACTION_DIM
        self.P = B if B is not None else cfg.POP_SIZE
        self.device = device if device is not None else _resolve_device(cfg)
        self.dtype = torch.float32
        for g in self.GENES:
            setattr(self, g, None)
        for e in self.EFF:
            setattr(self, e, None)

    def random_init(self):
        cfg = self.cfg
        N, O, A, B = self.N, self.O, self.A, self.P
        dev = self.device
        with torch.no_grad():
            self.M_in = (torch.rand(B, N, O, device=dev) < cfg.INIT_DENSITY).float()
            self.M_rec = (torch.rand(B, N, N, device=dev) < cfg.INIT_DENSITY).float()
            eye = torch.eye(N, dtype=torch.bool, device=dev)
            self.M_rec[:, eye] = 0.0
            self.M_out = (torch.rand(B, A, N, device=dev) < cfg.INIT_DENSITY).float()

            self.W_in = torch.randn(B, N, O, device=dev) * 0.1
            self.W_rec = torch.randn(B, N, N, device=dev) * 0.05
            self.W_out = torch.randn(B, A, N, device=dev) * 0.1
            self.b_out = torch.zeros(B, A, device=dev)

            tau = (torch.rand(B, N, device=dev) * 2 - 1) * cfg.TAU_E_NOISE
            self.tau_e = (cfg.BASE_TAU_E + tau).clamp(cfg.TAU_E_MIN, cfg.TAU_E_MAX)
            self.w_ei = torch.full((B, N), cfg.W_EI, device=dev)
            self.w_ie = torch.full((B, N), cfg.W_IE, device=dev)
        self.dtype = torch.float32

    def empty(self, B=None):
        return GeneStack(self.cfg, B=(B if B is not None else self.P), device=self.device)

    # ---------- 精度 ----------
    def fp32(self):
        for g in self.GENES:
            t = getattr(self, g)
            if t is not None and t.dtype != torch.float32:
                setattr(self, g, t.float())
        self.dtype = torch.float32

    def fp16(self):
        if not getattr(self.cfg, 'USE_FP16', True):
            self.fp32()
            return
        for g in self.GENES:
            t = getattr(self, g)
            if t is not None and t.dtype != torch.float16:
                setattr(self, g, t.half())
        self.dtype = torch.float16

    # ---------- 掩码权重缓存 ----------
    def refresh_eff(self):
        self.W_in_eff = self.W_in * self.M_in
        self.W_rec_eff = self.W_rec * self.M_rec
        self.W_out_eff = self.W_out * self.M_out

    # ---------- 子集 ----------
    def _sub_len(self, idx):
        if isinstance(idx, slice):
            return len(range(*idx.indices(self.P)))
        if isinstance(idx, torch.Tensor):
            return int(idx.numel()) if idx.dtype != torch.bool else int(idx.sum().item())
        if isinstance(idx, (list, np.ndarray)):
            return len(idx)
        return 1

    def __getitem__(self, idx):
        sub = self.empty(B=self._sub_len(idx))
        for g in self.GENES:
            setattr(sub, g, getattr(self, g)[idx])
        for e in self.EFF:
            t = getattr(self, e)
            setattr(sub, e, t[idx] if t is not None else None)
        sub.dtype = self.dtype
        return sub

    def clone_rows(self, idx):
        sub = self[idx]
        for g in self.GENES:
            setattr(sub, g, getattr(sub, g).clone())
        for e in self.EFF:
            t = getattr(sub, e)
            if t is not None:
                setattr(sub, e, t.clone())
        return sub

    # ---------- 单个体 解包/打包（字段与 test5d save_brain_state 兼容）----------
    def individual_state(self, i, use_half=False):
        dt = torch.float16 if use_half else torch.float32
        cpu = torch.device('cpu')
        with torch.no_grad():
            st = {
                'N': int(self.N),
                'M_in': self.M_in[i].to(cpu, dtype=torch.uint8),
                'M_rec': self.M_rec[i].to(cpu, dtype=torch.uint8),
                'M_out': self.M_out[i].to(cpu, dtype=torch.uint8),
                'W_in': self.W_in[i].to(cpu, dtype=dt),
                'W_rec': self.W_rec[i].to(cpu, dtype=dt),
                'W_out': self.W_out[i].to(cpu, dtype=dt),
                'b_out': self.b_out[i].to(cpu, dtype=dt),
                'tau_e_init': self.tau_e[i].to(cpu, dtype=dt),
                'w_ei': self.w_ei[i].to(cpu, dtype=dt),
                'w_ie': self.w_ie[i].to(cpu, dtype=dt),
                'W_hormone1': torch.zeros(self.cfg.HORMONE_NET_HIDDEN, self.N * 3, dtype=dt),
                'b_hormone1': torch.zeros(self.cfg.HORMONE_NET_HIDDEN, dtype=dt),
                'W_excit': torch.zeros(self.N, self.cfg.HORMONE_NET_HIDDEN, dtype=dt),
                'b_excit': torch.zeros(self.N, dtype=dt),
                'W_inhib': torch.zeros(self.N, self.cfg.HORMONE_NET_HIDDEN, dtype=dt),
                'b_inhib': torch.zeros(self.N, dtype=dt),
            }
        return st

    def set_individual_from_state(self, i, st):
        dev = self.device
        with torch.no_grad():
            self.M_in[i] = st['M_in'].float().to(dev)
            self.M_rec[i] = st['M_rec'].float().to(dev)
            self.M_out[i] = st['M_out'].float().to(dev)
            self.W_in[i] = st['W_in'].float().to(dev)
            self.W_rec[i] = st['W_rec'].float().to(dev)
            self.W_out[i] = st['W_out'].float().to(dev)
            self.b_out[i] = st['b_out'].float().to(dev)
            self.tau_e[i] = st['tau_e_init'].float().to(dev)
            self.w_ei[i] = st['w_ei'].float().to(dev)
            self.w_ie[i] = st['w_ie'].float().to(dev)

    # ---------- 整栈打包 / 解包（断点用）----------
    def pack(self):
        return {g: getattr(self, g).to('cpu').clone() for g in self.GENES}

    def unpack(self, d):
        dev = self.device
        for g in self.GENES:
            t = d[g].to(dev)
            if getattr(self.cfg, 'USE_FP16', True) and g.startswith('W'):
                t = t.half()
            setattr(self, g, t)
        self.dtype = torch.float16 if getattr(self.cfg, 'USE_FP16', True) else torch.float32


# ==================== GPU 批量贪吃蛇环境 ====================


def _make_dirs(dev):
    return torch.tensor([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=torch.long, device=dev)


class BatchedSnakeEnv:
    """B 个独立游戏并行（全部状态为 GPU 张量）。死亡个体冻结。"""

    def __init__(self, cfg, B, device):
        self.cfg = cfg
        self.B = B
        self.device = device
        self.G = cfg.GRID_SIZE
        self.MAXLEN = self.G * self.G
        self.DIRS = _make_dirs(device)
        self.reset()

    def reset(self):
        B, G, dev = self.B, self.G, self.device
        center = G // 2
        self.head = torch.full((B, 2), center, dtype=torch.long, device=dev)
        self.dir_idx = torch.randint(0, 4, (B,), device=dev)
        self.body = torch.zeros(B, self.MAXLEN, 2, dtype=torch.long, device=dev)
        self.body[:, 0] = self.head
        self.body[:, 1] = self.head - self.DIRS[self.dir_idx]
        self.body_len = torch.full((B,), 2, dtype=torch.long, device=dev)
        self.food = self._place_food_init()
        self.alive = torch.ones(B, dtype=torch.bool, device=dev)
        self.steps = torch.zeros(B, dtype=torch.long, device=dev)
        self.steps_wo_food = torch.zeros(B, dtype=torch.long, device=dev)
        self.ate = torch.zeros(B, dtype=torch.bool, device=dev)

    def _place_food_init(self):
        dev = self.device
        B, G = self.B, self.G
        head = self.head
        neck = self.body[:, 1]
        cand = torch.randint(0, G, (B, 2), device=dev)
        bad = (cand == head).all(dim=1) | (cand == neck).all(dim=1)
        for _ in range(31):
            if not bad.any():
                break
            re = torch.randint(0, G, (B, 2), device=dev)
            cand = torch.where(bad.unsqueeze(1), re, cand)
            bad = (cand == head).all(dim=1) | (cand == neck).all(dim=1)
        if bad.any():
            occ = torch.zeros(B, G * G, dtype=torch.bool, device=dev)
            occ[torch.arange(B, device=dev), head[:, 0] * G + head[:, 1]] = True
            occ[torch.arange(B, device=dev), neck[:, 0] * G + neck[:, 1]] = True
            free = (~occ).float()
            idx = torch.argmax(free, dim=1)
            fb = torch.stack((idx // G, idx % G), dim=1)
            cand = torch.where(bad.unsqueeze(1), fb, cand)
        return cand

    def _place_food_after_eat(self, eat_mask):
        B, G, dev = self.B, self.G, self.device
        if not eat_mask.any():
            return
        occ_b = self._occupancy_flat() > 0.5
        cand = torch.randint(0, G, (B, 2), device=dev)
        bad = occ_b.gather(1, (cand[:, 0] * G + cand[:, 1]).unsqueeze(1)).squeeze(1)
        for _ in range(31):
            if not bad.any():
                break
            re = torch.randint(0, G, (B, 2), device=dev)
            cand = torch.where(bad.unsqueeze(1), re, cand)
            bad = occ_b.gather(1, (cand[:, 0] * G + cand[:, 1]).unsqueeze(1)).squeeze(1)
        if bad.any():
            free = (~occ_b).float()
            idx = torch.argmax(free, dim=1)
            fb = torch.stack((idx // G, idx % G), dim=1)
            cand = torch.where(bad.unsqueeze(1), fb, cand)
        self.food = torch.where(eat_mask.unsqueeze(1), cand, self.food)

    def _occupancy_flat(self, tail_invalid=False):
        """[B, G*G] 占用图；tail_invalid=True 排除尾节。"""
        B, G, dev = self.B, self.G, self.device
        flat = self.body[:, :, 0] * G + self.body[:, :, 1]
        valid = torch.arange(self.MAXLEN, device=dev)[None, :] < self.body_len[:, None]
        if tail_invalid:
            valid &= torch.arange(self.MAXLEN, device=dev)[None, :] < (self.body_len - 1)[:, None]
        occ = torch.zeros(B, G * G, dtype=torch.float32, device=dev)
        occ.scatter_add_(1, flat.clamp(max=G * G - 1), valid.float())
        return occ

    def obs(self):
        B, G, dev = self.B, self.G, self.device
        head, food = self.head, self.food
        d = self.DIRS[self.dir_idx]
        dx = food[:, 0] - head[:, 0]
        dy = food[:, 1] - head[:, 1]
        left = self.DIRS[(self.dir_idx + 3) % 4]
        right = self.DIRS[(self.dir_idx + 1) % 4]

        obs = torch.zeros(B, 24, dtype=torch.float32, device=dev)
        obs[:, 0] = (dx * d[:, 0] + dy * d[:, 1] > 0).float()
        obs[:, 1] = (dx * left[:, 0] + dy * left[:, 1] > 0).float()
        obs[:, 2] = (dx * right[:, 0] + dy * right[:, 1] > 0).float()
        obs[:, 3] = torch.clamp(torch.hypot(dx.float(), dy.float()) / (G * math.sqrt(2)), 0, 1)

        will_eat = ((head + d) == food).all(dim=1)
        occ = self._occupancy_flat(tail_invalid=True)
        occ_eat = self._occupancy_flat(tail_invalid=False)
        occ_use = torch.where(will_eat[:, None], occ_eat, occ)

        ray_dirs = [left, (left + d), d, (d + right), right]
        for i, rd in enumerate(ray_dirs):
            fp, fs = self._cast_ray(rd, occ_use)
            obs[:, 4 + i * 2] = fp
            obs[:, 4 + i * 2 + 1] = fs

        segs = self.body
        wx = segs[:, :, 0] - head[:, None, 0]
        wy = segs[:, :, 1] - head[:, None, 1]
        rot_x = wx * d[:, None, 0] + wy * d[:, None, 1]
        rot_y = -wx * d[:, None, 1] + wy * d[:, None, 0]
        ang = torch.atan2(rot_y, rot_x) * 180.0 / math.pi
        ang = torch.where(ang < 0, ang + 360.0, ang)
        bucket = ((ang + 22.5) // 45).long() % 8
        close = 1.0 - torch.clamp(torch.hypot(wx.float(), wy.float()) / (G * math.sqrt(2)), 0, 1)
        valid = (torch.arange(self.MAXLEN, device=dev)[None, :] < self.body_len[:, None])
        valid &= (torch.arange(self.MAXLEN, device=dev)[None, :] > 0)
        close = torch.where(valid, close, torch.zeros_like(close))
        for k in range(8):
            m = (bucket == k) & valid
            obs[:, 14 + k] = (close * m).max(dim=1).values

        tail = self.body[torch.arange(B, device=dev), (self.body_len - 1).clamp(min=0)]
        twx = tail[:, 0] - head[:, 0]
        twy = tail[:, 1] - head[:, 1]
        obs[:, 22] = (twx * d[:, 0] + twy * d[:, 1]).float() / G
        obs[:, 23] = (-twx * d[:, 1] + twy * d[:, 0]).float() / G

        return obs

    def _cast_ray(self, rd, occ):
        """沿射线扫描，返回 (free_path, food_signal)。"""
        B, G, dev = self.B, self.G, self.device
        head = self.head
        food = self.food
        first_blocked = torch.full((B,), G + 1, dtype=torch.float32, device=dev)
        food_dist = torch.zeros(B, dtype=torch.float32, device=dev)
        prev_ok = torch.ones(B, dtype=torch.bool, device=dev)
        ar = torch.arange(B, device=dev)
        for k in range(1, G + 1):
            pos = head + rd * k
            r, c = pos[:, 0], pos[:, 1]
            inb = (r >= 0) & (r < G) & (c >= 0) & (c < G)
            on_body = occ[ar, r.clamp(0, G - 1) * G + c.clamp(0, G - 1)] > 0.5
            blocked = (~inb) | on_body
            first_blocked = torch.where(prev_ok & blocked,
                                        torch.full_like(first_blocked, float(k)),
                                        first_blocked)
            food_dist = torch.where(prev_ok & (pos == food).all(dim=1),
                                    torch.full_like(food_dist, float(k)),
                                    food_dist)
            prev_ok = prev_ok & (~blocked)
        free_path = torch.where(first_blocked > G,
                                torch.full_like(first_blocked, float(G)),
                                first_blocked) / G
        fd = torch.where(food_dist > 0, 1.0 - food_dist / G, 0.0)
        return free_path.clamp(0, 1), fd

    def sees_food(self, obs):
        return obs[:, 5:14:2].max(dim=1).values > 0.0

    def step(self, actions):
        B, dev = self.B, self.device
        nd_idx = torch.where(actions == 1, (self.dir_idx + 3) % 4, self.dir_idx)
        nd_idx = torch.where(actions == 2, (self.dir_idx + 1) % 4, nd_idx)
        self.dir_idx = nd_idx
        nd = self.DIRS[nd_idx]

        alive_f = self.alive
        self.steps = torch.where(alive_f, self.steps + 1, self.steps)
        self.steps_wo_food = torch.where(alive_f, self.steps_wo_food + 1, self.steps_wo_food)

        next_head = self.head + nd
        out_b = ((next_head[:, 0] < 0) | (next_head[:, 0] >= self.G) |
                 (next_head[:, 1] < 0) | (next_head[:, 1] >= self.G))

        will_eat = (next_head == self.food).all(dim=1)
        occ = self._occupancy_flat(tail_invalid=True)
        occ_eat = self._occupancy_flat(tail_invalid=False)
        occ_use = torch.where(will_eat[:, None], occ_eat, occ)
        ar = torch.arange(B, device=dev)
        hit = occ_use[ar, next_head[:, 0].clamp(0, self.G - 1) * self.G
                      + next_head[:, 1].clamp(0, self.G - 1)] > 0.5
        crash = alive_f & (out_b | hit)

        move = alive_f & (~crash)
        shifted = torch.zeros_like(self.body)
        shifted[:, 0] = next_head
        shifted[:, 1:] = self.body[:, :-1]
        self.body = torch.where(move[:, None, None], shifted, self.body)
        self.head = torch.where(move[:, None], next_head, self.head)

        ate = move & will_eat
        self.body_len = torch.where(move, (self.body_len + ate.long()).clamp(max=self.MAXLEN),
                                    self.body_len)
        self.steps_wo_food = torch.where(ate, torch.zeros_like(self.steps_wo_food),
                                         self.steps_wo_food)
        self.ate = ate
        self._place_food_after_eat(ate)

        starve = self.steps_wo_food > (2 * self.body_len.float() + 20)
        self.alive = alive_f & (~crash) & (~starve)

    def all_done(self):
        return not bool(self.alive.any().item())


# ==================== 批量前向（无激素 E-I 动力学）====================


def forward_batch(pop, obs, E, I, st, cts, cfg):
    """单次 E-I 迭代（B 个个体并行，无激素支路）。"""
    ext = torch.bmm(pop.W_in_eff, obs.unsqueeze(-1)).squeeze(-1)
    rec = torch.bmm(pop.W_rec_eff, E.unsqueeze(-1)).squeeze(-1)
    total = ext + rec

    st = cfg.SHORT_TERM_DECAY * st + (1 - cfg.SHORT_TERM_DECAY) * E
    tau = (pop.tau_e + cfg.SHORT_TERM_GAIN * st).clamp(cfg.TAU_E_MIN, cfg.TAU_E_MAX)
    w_ei = pop.w_ei.clamp(cfg.W_EI_MIN, cfg.W_EI_MAX)
    w_ie = pop.w_ie.clamp(cfg.W_IE_MIN, cfg.W_IE_MAX)

    E_new = torch.sigmoid(total + tau * E - w_ei * I)
    I_new = torch.sigmoid(w_ie * E_new)

    logits = torch.bmm(pop.W_out_eff, E_new.unsqueeze(-1)).squeeze(-1) + pop.b_out
    fatigue = torch.relu(cts - cfg.FATIGUE_THRESHOLD) * cfg.FATIGUE_GAIN
    fatigue = fatigue.clamp(max=cfg.FATIGUE_MAX)
    logits = logits - fatigue
    return logits, E_new, I_new, st


def update_fatigue(cts, action):
    cur = cts.gather(1, action.unsqueeze(1)) + 1.0
    ncts = torch.zeros_like(cts)
    ncts.scatter_(1, action.unsqueeze(1), cur)
    return ncts


def deliberate_batch(pop, obs, E, I, st, cts, cfg):
    """K 倍帧率思考：内部迭代 K 次，logits 平均后 argmax。"""
    K = cfg.FRAME_RATE
    logits_sum = None
    for k in range(K):
        o = obs * (cfg.INPUT_DECAY ** k)
        logits, E, I, st = forward_batch(pop, o, E, I, st, cts, cfg)
        logits_sum = logits if logits_sum is None else logits_sum + logits
    action = torch.argmax(logits_sum, dim=1)
    return action, E, I, st


# ==================== 种群评估（GPU 全并行）====================


def _eval_chunk(pop, cfg):
    """对单个子种群（B=pop.P）并行评估 cfg.EVAL_EPISODES 局。"""
    B = pop.P
    dev = pop.device
    N = pop.N
    A = pop.A
    env = BatchedSnakeEnv(cfg, B, dev)
    half = torch.float16 if getattr(cfg, 'USE_FP16', True) else torch.float32

    tot_food = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_seen = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_unseen = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_act1 = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_act2 = torch.zeros(B, dtype=torch.float32, device=dev)

    for _ in range(cfg.EVAL_EPISODES):
        env.reset()
        E = torch.zeros(B, N, dtype=half, device=dev)
        I = torch.zeros(B, N, dtype=half, device=dev)
        st = torch.zeros(B, N, dtype=half, device=dev)
        cts = torch.zeros(B, A, dtype=half, device=dev)

        for _ in range(cfg.MAX_STEPS):
            al = env.alive
            obs = env.obs().to(half)
            sees = env.sees_food(obs)
            tot_seen += (al & sees).float()
            tot_unseen += (al & (~sees)).float()

            act, E, I, st = deliberate_batch(pop, obs, E, I, st, cts, cfg)
            cts = update_fatigue(cts, act)
            tot_act1 += (al & (act == 1)).float()
            tot_act2 += (al & (act == 2)).float()

            env.step(act)
            if env.all_done():
                break

    E_ = float(cfg.EVAL_EPISODES)
    metrics = torch.stack((tot_food, tot_seen, tot_unseen), dim=1) / E_

    turn_lim = max(cfg.EVAL_EPISODES, 1)
    c1 = tot_act1.cpu().numpy()
    c2 = tot_act2.cpu().numpy()
    m = metrics.cpu().numpy()
    death = ((c1 > turn_lim) | (c2 > turn_lim)) & ((c1 == 0) | (c2 == 0))
    m[death] = (0.0, 99999.0, 0.0)
    return torch.from_numpy(m).float()


def evaluate_population_gpu(pop, cfg):
    """全种群评估：按显存分块并行，返回 [P,3] = (food, seen, unseen)。"""
    if getattr(cfg, 'USE_FP16', True):
        pop.fp16()
    batch = _auto_eval_batch(cfg, pop.device)
    if batch >= pop.P:
        pop.refresh_eff()
        return _eval_chunk(pop, cfg)

    metrics = torch.zeros(pop.P, 3)
    for lo in range(0, pop.P, batch):
        sub = pop[slice(lo, min(lo + batch, pop.P))]
        sub.refresh_eff()
        m = _eval_chunk(sub, cfg)
        metrics[lo:lo + sub.P] = m.cpu()
    return metrics


# ==================== 进化（GPU 向量化交叉/变异）====================


def evolve_topology_gpu(pop, metrics, cfg, gen=0):
    """进化下一代（语义与 evolve_topology 一致，向量化到 GPU）。"""
    P = pop.P
    N = pop.N
    dev = pop.device
    active = _freeze_active_groups(gen, cfg)
    has_g1 = 'G1' in active
    has_g2 = 'G2' in active
    mut = _dynamic_mutation_rates(cfg, gen)

    threshold = float(getattr(cfg, 'LONG_SNAKE_SCORE_THRESHOLD', 3.0))
    mn = metrics.cpu().numpy()
    order = sorted(range(P), key=lambda i: _selection_key((mn[i][0], mn[i][1], mn[i][2]), threshold),
                   reverse=True)
    elite_idx = order[:cfg.ELITE_SIZE]
    elites = pop[elite_idx]

    new_pop = pop.empty()
    children = pop.empty(B=P - cfg.ELITE_SIZE)
    for g in GeneStack.GENES:
        setattr(children, g, torch.empty(P - cfg.ELITE_SIZE, *getattr(elites, g).shape[1:],
                                         dtype=getattr(elites, g).dtype, device=dev))
    for g in GeneStack.GENES:
        setattr(new_pop, g, torch.cat([getattr(elites, g).clone(), getattr(children, g)], dim=0))

    E = cfg.ELITE_SIZE
    B2 = P - E
    p1_idx = torch.randint(0, E, (B2,), device=dev)
    p2_idx = torch.randint(0, E, (B2,), device=dev)
    p2_idx = torch.where(p2_idx == p1_idx, (p1_idx + 1) % E, p2_idx)

    p1 = elites[p1_idx]
    p2 = elites[p2_idx]

    with torch.no_grad():
        if has_g1:
            col_mask = torch.rand(B2, N, device=dev) > 0.5
            row_mask = col_mask.unsqueeze(1)
            col_mask_2d = col_mask.unsqueeze(2)
            same_p1 = row_mask & col_mask_2d
            same_p2 = (~row_mask) & (~col_mask_2d)
            coin = torch.rand(B2, N, N, device=dev) > 0.5

            children.W_in = torch.where(col_mask.unsqueeze(2), p1.W_in, p2.W_in)
            children.M_in = torch.where(col_mask.unsqueeze(2), p1.M_in, p2.M_in)
            children.W_rec = torch.where(same_p1, p1.W_rec,
                                torch.where(same_p2, p2.W_rec,
                                    torch.where(coin, p1.W_rec, p2.W_rec)))
            children.M_rec = torch.where(same_p1, p1.M_rec,
                                torch.where(same_p2, p2.M_rec,
                                    torch.where(coin, p1.M_rec, p2.M_rec)))
            children.W_out = torch.where(col_mask.unsqueeze(1), p1.W_out, p2.W_out)
            children.M_out = torch.where(col_mask.unsqueeze(1), p1.M_out, p2.M_out)
            bmask = torch.rand(B2, cfg.ACTION_DIM, device=dev) > 0.5
            children.b_out = torch.where(bmask, p1.b_out, p2.b_out)

        if has_g2:
            col2 = torch.rand(B2, N, device=dev) > 0.5
            children.tau_e = torch.where(col2, p1.tau_e, p2.tau_e)
            children.w_ei = torch.where(col2, p1.w_ei, p2.w_ei)
            children.w_ie = torch.where(col2, p1.w_ie, p2.w_ie)

        if has_g1:
            pick = torch.randint(0, 3, (B2,), device=dev)
            topo_gate = torch.rand(B2, device=dev) < mut['topo_mut_prob']
            for ai, attr in enumerate(GeneStack.G1_MASKS):
                sel = (pick == ai) & topo_gate
                if bool(sel.any().item()):
                    t = getattr(children, attr)
                    flip = torch.rand_like(t) < mut['mask_mut_rate']
                    setattr(children, attr,
                            torch.where(sel[:, None, None] & flip, 1.0 - t, t))
            for attr in GeneStack.G1_WEIGHTS:
                t = getattr(children, attr)
                noise = torch.randn_like(t) * mut['weight_mut_std']
                m = (torch.rand_like(t) < mut['weight_mut_frac']).to(t.dtype)
                setattr(children, attr, t + noise * m)

        if has_g2:
            children.tau_e = torch.clamp(children.tau_e + torch.randn_like(children.tau_e) * mut['tau_e_mut_std'],
                                         cfg.TAU_E_MIN, cfg.TAU_E_MAX)
            children.w_ei = torch.clamp(children.w_ei + torch.randn_like(children.w_ei) * mut['w_ei_mut_std'],
                                        cfg.W_EI_MIN, cfg.W_EI_MAX)
            children.w_ie = torch.clamp(children.w_ie + torch.randn_like(children.w_ie) * mut['w_ie_mut_std'],
                                        cfg.W_IE_MIN, cfg.W_IE_MAX)

    for g in GeneStack.GENES:
        setattr(new_pop, g, torch.cat([getattr(elites, g).clone(), getattr(children, g)], dim=0))
    new_pop.dtype = pop.dtype
    return new_pop


# ==================== 保存 / 加载 ====================


def load_best_state(path, cfg):
    """读取最优模型文件，返回个体状态 dict（None=不可用）。"""
    path = model_path(path)
    if not os.path.exists(path):
        return None
    try:
        data = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"  警告: 模型 {path} 读取失败 ({e})，已忽略种子")
        return None
    saved_cfg = data.get('config', {})
    if saved_cfg and (saved_cfg.get('NUM_COLUMNS') != cfg.NUM_COLUMNS or
                      saved_cfg.get('OBS_DIM') != cfg.OBS_DIM or
                      saved_cfg.get('ACTION_DIM') != cfg.ACTION_DIM):
        print(f"  警告: 模型 {path} 与当前配置不匹配 (N/OBS/ACTION_DIM)，已忽略种子")
        return None
    st = data.get('brain')
    if st is None:
        return None
    return st, float(data.get('food', -1.0)), float(data.get('steps', 0.0))


def save_best_model(path, st, cfg, food, seen, unseen):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save({
        'brain': st,
        'food': float(food),
        'steps': float(seen + unseen),
        'config': io_config_dict(cfg),
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }, path)


def io_config_dict(cfg):
    from .io import config_dict
    return config_dict(cfg)


def save_checkpoint7(path, cfg, next_gen, pop, history,
                     cum_eval_time, cum_evolve_time,
                     best_state, best_food, best_seen, best_unseen):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {
        'next_gen': next_gen,
        'pop': pop.pack(),
        'history': history,
        'cum_eval_time': float(cum_eval_time),
        'cum_evolve_time': float(cum_evolve_time),
        'best_state': best_state,
        'best_food': float(best_food),
        'best_seen': float(best_seen),
        'best_unseen': float(best_unseen),
        'config': io_config_dict(cfg),
        'random_state': random.getstate(),
        'torch_rng_state': torch.get_rng_state(),
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    tmp = path + '.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, path)
    print(f"  [Checkpoint] 断点已保存 -> {path} (next_gen={next_gen})")


def load_checkpoint7(path, cfg):
    if not os.path.exists(path):
        return None
    try:
        data = torch.load(path, map_location='cpu', weights_only=False)
    except Exception:
        return None
    saved_cfg = data.get('config', {})
    if saved_cfg and (saved_cfg.get('NUM_COLUMNS') != cfg.NUM_COLUMNS or
                      saved_cfg.get('OBS_DIM') != cfg.OBS_DIM or
                      saved_cfg.get('ACTION_DIM') != cfg.ACTION_DIM):
        print(f"  警告: 断点 {path} 与当前配置不匹配 (N/OBS/ACTION_DIM)，已忽略")
        return None
    return data


# ==================== 主循环 ====================


def run_training_gpu(cfg, visualize=False):
    """GPU 全并行进化主循环。返回 (best_state, history)。"""
    device = _resolve_device(cfg)
    print(f"[GPU] device = {device}  "
          f"({'GPU: ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU 回退'})")
    if device.type == 'cuda':
        mem = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"[GPU] 显存 {mem:.1f} GB, 自动评估批大小 = {_auto_eval_batch(cfg, device)}")
    t_program = time.perf_counter()

    start_gen = 0
    pop = GeneStack(cfg, device=device)
    history = {'gen': [], 'best_food': [], 'avg_food': [], 'best_seen': [], 'best_unseen': []}
    cum_eval_time = 0.0
    cum_evolve_time = 0.0
    best_state = None
    best_food = -1.0
    best_seen = 0.0
    best_unseen = 0.0

    if cfg.AUTO_RESUME:
        ck = load_checkpoint7(cfg.CHECKPOINT_PATH, cfg)
        if ck is not None:
            start_gen = int(ck['next_gen'])
            pop.unpack(ck['pop'])
            history = ck['history']
            cum_eval_time = float(ck.get('cum_eval_time', 0.0))
            cum_evolve_time = float(ck.get('cum_evolve_time', 0.0))
            best_state = ck.get('best_state')
            best_food = float(ck.get('best_food', -1.0))
            best_seen = float(ck.get('best_seen', 0.0))
            best_unseen = float(ck.get('best_unseen', 0.0))
            if best_state is not None:
                random.setstate(ck['random_state'])
                torch.set_rng_state(ck['torch_rng_state'])
            print(f"=== 检测到断点 [{cfg.CHECKPOINT_PATH}] ===")
            print(f"  已完成 {start_gen} 代 -> 从第 {start_gen} 代接续 | "
                  f"历史最优: Food={best_food:.1f} | 已耗时 {cum_eval_time + cum_evolve_time:.1f}s")

    if pop.M_in is None:
        print("初始化种群（GPU 随机初始化）...")
        pop.random_init()

        if cfg.SEED_FROM_BEST:
            for sp in (cfg.SEED_MODEL_PATH, cfg.SEED_MODEL_PATH2):
                seed = load_best_state(sp, cfg)
                if seed is not None:
                    st, s_food, s_steps = seed
                    pop.set_individual_from_state(0, st)
                    best_state = pop.individual_state(0, use_half=False)
                    best_food = s_food
                    print(f"  [Seed] 已注入 {sp} 作为种群种子 (Food={s_food:.1f}, Steps={s_steps:.1f})")
                    break
            else:
                print("  [Seed] 未发现可用的最优模型种子，全新随机初始化")

        save_checkpoint7(cfg.CHECKPOINT_PATH, cfg, 0, pop, history,
                         cum_eval_time, cum_evolve_time,
                         best_state, best_food, best_seen, best_unseen)

    try:
        for gen in range(start_gen, cfg.GENERATIONS):
            t_eval = time.perf_counter()
            metrics = evaluate_population_gpu(pop, cfg)
            eval_time = time.perf_counter() - t_eval
            cum_eval_time += eval_time

            mn = metrics.numpy()
            threshold = float(getattr(cfg, 'LONG_SNAKE_SCORE_THRESHOLD', 3.0))
            best_idx = max(range(cfg.POP_SIZE),
                           key=lambda i: _selection_key((mn[i][0], mn[i][1], mn[i][2]), threshold))
            b_food, b_seen, b_unseen = (float(mn[best_idx][0]), float(mn[best_idx][1]),
                                        float(mn[best_idx][2]))
            avg_food = float(np.mean(mn[:, 0]))

            history['gen'].append(gen)
            history['best_food'].append(b_food)
            history['avg_food'].append(avg_food)
            history['best_seen'].append(b_seen)
            history['best_unseen'].append(b_unseen)

            if (b_food > best_food or
                    (b_food == best_food and best_food >= 0 and
                     ((b_food > threshold and b_unseen > best_unseen) or
                      (b_food <= threshold and b_seen < best_seen)))):
                best_food = b_food
                best_seen = b_seen
                best_unseen = b_unseen
                best_state = pop.individual_state(best_idx, use_half=False)

            if gen < cfg.GENERATIONS - 1:
                t_ev = time.perf_counter()
                pop = evolve_topology_gpu(pop, metrics, cfg, gen=gen)
                evolve_time = time.perf_counter() - t_ev
                cum_evolve_time += evolve_time
            else:
                evolve_time = 0.0

            if (gen + 1) % cfg.PRINT_HISTORY_EVERY == 0:
                print(f"Gen {gen + 1}/{cfg.GENERATIONS} | "
                      f"BestFood: {b_food:.2f} | BestSeen: {b_seen:.1f} | "
                      f"BestUnseen: {b_unseen:.1f} | AvgFood: {avg_food:.2f} | "
                      f"eval {eval_time:.1f}s / evolve {evolve_time:.1f}s")

            if (gen + 1) % cfg.CHECKPOINT_INTERVAL == 0:
                save_checkpoint7(cfg.CHECKPOINT_PATH, cfg, gen + 1, pop, history,
                                 cum_eval_time, cum_evolve_time,
                                 best_state, best_food, best_seen, best_unseen)

    except KeyboardInterrupt:
        nxt = gen + 1 if 'gen' in dir() else start_gen
        print("\n训练被中断 (Ctrl+C)，正在保存断点...")
        save_checkpoint7(cfg.CHECKPOINT_PATH, cfg, nxt, pop, history,
                         cum_eval_time, cum_evolve_time,
                         best_state, best_food, best_seen, best_unseen)
        sys.exit(0)

    t_delta = time.perf_counter() - t_program
    print(f"\nTotal runtime: {t_delta:.1f}s "
          f"(eval {cum_eval_time:.1f}s / evolve {cum_evolve_time:.1f}s)")

    if best_state is None:
        best_state = pop.individual_state(0, use_half=False)
    save_best_model(cfg.BEST_MODEL_PATH, best_state, cfg, best_food, best_seen, best_unseen)
    print(f"\n最优模型已保存: {cfg.BEST_MODEL_PATH} "
          f"(Food={best_food:.2f}, Seen={best_seen:.1f}, Unseen={best_unseen:.1f})")

    if os.path.exists(cfg.CHECKPOINT_PATH):
        os.remove(cfg.CHECKPOINT_PATH)
        print(f"训练已完成，已删除临时断点: {cfg.CHECKPOINT_PATH}")

    if visualize:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            ax1.plot(history['gen'], history['best_food'], label='Best Food', color='red', marker='o', markersize=3)
            ax1.plot(history['gen'], history['avg_food'], label='Avg Food', color='blue', alpha=0.6)
            ax1.set_title("Evolution Progress — Food Count")
            ax1.set_xlabel("Generation")
            ax1.set_ylabel("Food Eaten")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax2.plot(history['gen'], history['best_seen'], label='Best Seen', color='green', marker='s', markersize=3)
            ax2.plot(history['gen'], history['best_unseen'], label='Best Unseen', color='purple', marker='^', markersize=3)
            ax2.set_title("Best Seen/Unseen Steps")
            ax2.set_xlabel("Generation")
            ax2.set_ylabel("Steps")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig('gpu_history.png', dpi=100)
            plt.close(fig)
            print("历史曲线已保存: gpu_history.png")
        except Exception as e:
            print(f"(matplotlib 曲线跳过: {e})")

    return best_state, history


def play_best(cfg, max_steps=300):
    """加载最优模型并在 GPU 上播放一局（打印 ASCII 棋盘）。"""
    res = load_best_state(cfg.BEST_MODEL_PATH, cfg)
    if res is None:
        print("无最优模型可播放")
        return
    st, food, steps = res
    dev = _resolve_device(cfg)
    pop = GeneStack(cfg, B=1, device=dev)
    pop.random_init()
    pop.set_individual_from_state(0, st)
    pop.refresh_eff()
    if getattr(cfg, 'USE_FP16', True):
        pop.fp16()
        pop.refresh_eff()
    env = BatchedSnakeEnv(cfg, 1, dev)
    E = torch.zeros(1, pop.N, dtype=pop.dtype, device=dev)
    I = torch.zeros(1, pop.N, dtype=pop.dtype, device=dev)
    stt = torch.zeros(1, pop.N, dtype=pop.dtype, device=dev)
    cts = torch.zeros(1, pop.A, dtype=pop.dtype, device=dev)
    G = cfg.GRID_SIZE

    def render():
        g = [['.' for _ in range(G)] for _ in range(G)]
        h = env.head[0].tolist()
        b = env.body[0, :env.body_len[0]].tolist()
        f = env.food[0].tolist()
        g[f[0]][f[1]] = '*'
        for i, (r, c) in enumerate(b):
            ch = 'H' if i == 0 else '#'
            if 0 <= r < G and 0 <= c < G:
                g[r][c] = ch
        print('  ' + '\n  '.join(''.join(row) for row in g))

    ep_food = 0
    s = 0
    for s in range(max_steps):
        obs = env.obs().to(pop.dtype)
        act, E, I, stt = deliberate_batch(pop, obs, E, I, stt, cts, cfg)
        cts = update_fatigue(cts, act)
        env.step(act)
        if env.ate[0]:
            ep_food += 1
        if s % 10 == 0:
            print(f"\n--- Step {s} (score {ep_food}) ---")
            render()
        if env.all_done():
            print(f"\n--- 死亡 @ step {s} ---")
            render()
            break
    print(f"\nPlay done: Food={ep_food}, Steps={s + 1}")
