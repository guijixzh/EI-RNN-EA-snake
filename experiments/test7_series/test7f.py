# ==========================================
# test7f.py —— 疲劳脚手架退火版（基于 test7e）
#
# 疲劳扫描结论（test7e_fatigue_scan.py）：
#  - 疲劳是转向密度的唯一有效压制者（适应度计价 20 代压不动密度）；
#  - test7d 系大脑未内化转弯纪律：摘掉疲劳密度→0.996、吃子→4.5，
#    策略+疲劳复合体才是被选择单位，疲劳成了常驻拐杖；
#  - 密度 0.82→0.63 区间吃子几乎免费（61→58），<0.5 才崩盘。
#
# 相对 test7e 的改动：
#  1. 疲劳增益按代退火：gain_t = max(FLOOR, GAIN·ANNEAL_FACTOR^(gen//PERIOD))，
#     默认每 20 代 ×0.7、下限 0.05——逐步撤除脚手架，逼大脑内化转弯纪律；
#     适应度（整局转弯计价）继续提供下探密度的选择压力。
#  2. 路径 test7f_*，种子默认 test7e_econ_best_model.pth。
# 其余（滞回开关默认关 / 饿死斜率 3 / econ 成本）与 test7e 一致。
# ==========================================

import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch


# ==========================================
# 0. 全局配置类
# ==========================================
class Config:
    # --- 进化参数（与 test5d 一致）---
    POP_SIZE = 2048
    GENERATIONS = 100
    ELITE_SIZE = 256
    MUT_RATE = 0.05
    TOPOLOGY_MUT_PROB = 0.05
    WEIGHT_MUT_FRAC = 0.2
    WEIGHT_MUT_STD = 0.1
    TAU_E_MUT_STD = 0.05
    EVO_COS_MODE = 'anneal'
    EVO_COS_PERIOD = 100
    EVO_DYN_DECAY_TAU = 33

    # --- 无激素 EI-RNN（G3 激素组彻底移除，以下仅保留文件格式兼容字段）---
    TRAIN_HORMONE_NET = False
    HORMONE_NET_HIDDEN = 32

    # --- 环境参数 ---
    GRID_SIZE = 10
    EVAL_EPISODES = 10
    MAX_STEPS = 100000

    # --- 脑结构参数（OBS_MODE 决定 OBS_DIM：'24'=test7 射线观测 | '32proj'=32维投影观测）---
    NUM_COLUMNS = 256
    OBS_MODE = '32proj'
    OBS_DIM = 24 if OBS_MODE == '24' else 32
    ACTION_DIM = 3
    INIT_DENSITY = 0.15

    # --- 适应度模式：'econ'=乘法经济学 | 'additive'=加法惩罚 | 'tuple'=元组字典序 ---
    FIT_MODE = 'econ'

    # --- econ 臂参数 ---
    TURN_COST = 1.0     # 整局每单次转向折算的步数成本
    STEP_REF = 10.0     # 每颗食物的参考步数（10x10 盘均食距量级）
    CROWD_W = 0.0       # 拥挤加权：转向成本 × (1 + CROWD_W·food/50)，0=关

    # --- 滞回解码（诊断显示转向边际均值 6.9，默认关闭）---
    TURN_MARGIN = 0.0

    # --- additive 臂参数 ---
    FOOD_EFF_WEIGHT = 0.3
    TURN_PENALTY = 3.0  # 转向占比线性扣分权重

    # --- 单侧转弯判死（默认关闭：早期随机个体普遍摇头，判罚干扰初期筛选）---
    ONE_SIDED_TURN_DEATH = False

    # --- 饿死斜率：steps_wo_food > STARVE_SLOPE*len + 20 ---
    STARVE_SLOPE = 3.0

    # --- 食物扇区信号幅度（仅 32proj 模式）---
    # 诊断实验证明：投影值 0.05-0.25 过弱，随机 W_in 无法产生"看见食物→转向"反射，
    # 随机种群 BestFood 仅 0.8；×4 追平 24 维(2.6)，×8 反超(3.6)。
    OBS_FOOD_SCALE = 8.0

    # 自身/障碍扇区幅度：块可读性化验+三臂对照验证（30代 BestFood 8.8→20.8），
    # 放大后进化可读这些块，EliteFood 4.1→10.9，自撞死亡下降。
    OBS_SELF_SCALE = 8.0
    OBS_OBSTACLE_SCALE = 8.0

    # --- E-I 动力学参数 ---
    BASE_TAU_E = 0.7
    TAU_E_NOISE = 0.1
    TAU_E_MIN = 0.001
    TAU_E_MAX = 2.0
    W_EI = 2.0
    W_IE = 2.0
    W_EI_MUT_STD = 0.1
    W_IE_MUT_STD = 0.1
    W_EI_MIN = 0.0
    W_EI_MAX = 6.0
    W_IE_MIN = 0.0
    W_IE_MAX = 6.0

    # --- 短期 tau 调制 ---
    SHORT_TERM_GAIN = -0.2
    SHORT_TERM_DECAY = 0.3

    # --- 转向疲劳（脚手架退火）---
    FATIGUE_TURN_GAIN = 0.2          # 初始增益（第 0 个周期）
    FATIGUE_TURN_DECAY = 0.9
    ANNEAL_PERIOD = 20               # 每 N 代降一档
    ANNEAL_FACTOR = 0.7              # 降档乘子
    FATIGUE_FLOOR = 0.05             # 增益下限（保留微弱先验，不彻底归零）

    # --- K 倍帧率思考 ---
    FRAME_RATE = 5
    INPUT_DECAY = 0.9

    # --- 进化筛选策略 ---
    LONG_SNAKE_SCORE_THRESHOLD = 3.0
    CYCLE_PATTERN = [('G2', 'G1', 'G3')]

    # --- GPU 并行参数（test7 新增）---
    DEVICE = 'auto'          # 'auto' | 'cuda' | 'cpu'
    USE_FP16 = True          # 评估用半精度（进化噪声远大于 fp16 精度）
    EVAL_BATCH = 0           # 单批并行个体数；0 = 按显存自动估算
    EVAL_MEM_FRAC = 0.55     # 自动估算允许占用的显存比例

    # --- 输出 ---
    PRINT_HISTORY_EVERY = 1

    # --- 断点 / 最优模型 / 种子 ---
    CHECKPOINT_PATH = 'test7f_econ_checkpoint.pth'
    BEST_MODEL_PATH = 'test7f_econ_best_model.pth'
    LATEST_GEN_BEST_MODEL_PATH = 'test7f_econ_latest_gen_best.pth'
    AUTO_RESUME = True
    CHECKPOINT_INTERVAL = 10
    SEED_FROM_BEST = False
    SEED_MODEL_PATH = 'test7e_econ_best_model.pth'
    SEED_MODEL_PATH2 = 'test7e_econ_latest_gen_best.pth'


# ==========================================
# 0b. 基础工具
# ==========================================
def _resolve_device(cfg):
    if cfg.DEVICE != 'auto':
        return torch.device(cfg.DEVICE)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _cfg_dict(cfg):
    merged = {}
    merged.update(vars(cfg.__class__))
    merged.update(vars(cfg))
    return {k: v for k, v in merged.items() if not k.startswith('__')}


def _freeze_active_groups(gen, cfg):
    pattern = getattr(cfg, 'CYCLE_PATTERN', [('G2', 'G1')])
    active = set(pattern[gen % len(pattern)])
    if not bool(getattr(cfg, 'TRAIN_HORMONE_NET', False)):
        active.discard('G3')
    return frozenset(active)


def _fitness_econ(m, cfg):
    """乘法经济学适应度（整局转弯计价版）：
    turns_total = 整局转向数（含死亡前挣扎，m[8]+m[9]）
    cost = steps_last + TURN_COST·(1+CROWD_W·food/50)·turns_total
    fitness = food * min(STEP_REF*food/max(cost,1), 1)
    steps 只计到最后一食（不奖励快死）；挣扎摆动与长蛇期摆动同样计价。
    """
    if m[1] >= 99999:
        return -1e9
    food, steps_last = m[0], m[3]
    turns_total = m[8] + m[9]
    crowd = 1.0 + float(getattr(cfg, 'CROWD_W', 0.0)) * food / 50.0
    cost = steps_last + float(getattr(cfg, 'TURN_COST', 1.0)) * crowd * turns_total
    economy = float(getattr(cfg, 'STEP_REF', 10.0)) * food / max(cost, 1.0)
    return food * min(economy, 1.0)


def _fitness_additive(m, cfg):
    """加法惩罚适应度：food + w*food/steps_last - TURN_PENALTY*turn_ratio
    turn_ratio = 每局转向数/每局步数（摆动率）。
    """
    if m[1] >= 99999:
        return -1e9
    food, steps_last = m[0], m[3]
    eff = food / max(steps_last, 1.0)
    turn_ratio = (m[8] + m[9]) / max(m[1] + m[2], 1.0)
    return (food + float(getattr(cfg, 'FOOD_EFF_WEIGHT', 0.3)) * eff
            - float(getattr(cfg, 'TURN_PENALTY', 3.0)) * turn_ratio)


def _fitness_tuple(m, cfg):
    """test7 原版元组字典序排序键。"""
    if m[1] >= 99999:
        return (-1e9, 0, 0)
    threshold = float(getattr(cfg, 'LONG_SNAKE_SCORE_THRESHOLD', 3.0))
    food, seen, unseen = m[0], m[1], m[2]
    if food > threshold:
        return (food, unseen, -seen)
    return (food, -seen, unseen)


def _make_key_fn(cfg):
    """返回行向量排序键函数 key(m)，m 为 _eval_chunk 输出的单行指标。"""
    mode = getattr(cfg, 'FIT_MODE', 'econ')
    if mode == 'tuple':
        return lambda m: _fitness_tuple(m, cfg)
    if mode == 'additive':
        return lambda m: _fitness_additive(m, cfg)
    return lambda m: _fitness_econ(m, cfg)


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
    per_ind = n * n * 16.0          # W_rec fp32 + eff fp16 + M fp16 + 前向临时区
    per_ind += n * cfg.OBS_DIM * 6.0
    batch = int(total * cfg.EVAL_MEM_FRAC / per_ind)
    return max(32, min(batch, cfg.POP_SIZE))


# ==========================================
# 1. 种群基因组张量栈
# ==========================================
class GeneStack:
    """整个种群的基因型/表现型堆叠张量。

    形状约定（B = 个体数, N = 柱数, O = 观测维, A = 动作维）：
      M_in  [B,N,O]   M_rec [B,N,N]   M_out [B,A,N]
      W_in  [B,N,O]   W_rec [B,N,N]   W_out [B,A,N]
      b_out [B,A]     tau_e [B,N]     w_ei / w_ie [B,N]
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

    # ---------- 分配 ----------
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

    # ---------- 掩码权重缓存（评估期间复用，等价原版 refresh_cached）----------
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
                # 无激素：补零字段保持与 test5d 文件格式兼容
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


# 方向表：0=(0,1) 1=(1,0) 2=(0,-1) 3=(-1,0)；左转=idx+3 mod4，右转=idx+1 mod4
def _make_dirs(dev):
    return torch.tensor([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=torch.long, device=dev)


# ==========================================
# 2. GPU 批量贪吃蛇环境
# ==========================================
class BatchedSnakeEnv:
    """B 个独立游戏并行（全部状态为 GPU 张量）。死亡个体冻结。"""

    def __init__(self, cfg, B, device):
        self.cfg = cfg
        self.B = B
        self.device = device
        self.G = cfg.GRID_SIZE
        self.MAXLEN = self.G * self.G
        self.DIRS = _make_dirs(device)
        self.mode24 = (getattr(cfg, 'OBS_MODE', '32proj') == '24')
        self.reset()

    # ---------- 重置 ----------
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
        self.died = torch.zeros(B, dtype=torch.long, device=dev)   # 0存活 1撞墙 2撞己 3饿死

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
        occ = self._occupancy_flat()
        occ_b = occ > 0.5
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
        new_food = torch.where(eat_mask.unsqueeze(1), cand, self.food)
        self.food = new_food

    def _occupancy_flat(self, tail_invalid=False):
        """[B, G*G] 占用图（bool 计 1）；tail_invalid=True 排除尾节。"""
        B, G, dev = self.B, self.G, self.device
        flat = self.body[:, :, 0] * G + self.body[:, :, 1]
        valid = torch.arange(self.MAXLEN, device=dev)[None, :] < self.body_len[:, None]
        if tail_invalid:
            valid &= torch.arange(self.MAXLEN, device=dev)[None, :] < (self.body_len - 1)[:, None]
        occ = torch.zeros(B, G * G, dtype=torch.float32, device=dev)
        occ.scatter_add_(1, flat.clamp(max=G * G - 1), valid.float())
        return occ

    # ---------- 观测（按 OBS_MODE 分发）----------
    def obs(self):
        if self.mode24:
            return self._obs24()
        return self._obs32()

    def _obs24(self):
        """24 维观测：逐通道照抄 test7.py obs()（食物投影bit/距离/5射线/8身体桶/尾投影）。"""
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

        # --- 自体感知 14:22 ---
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
            m_ = (bucket == k) & valid
            obs[:, 14 + k] = (close * m_).max(dim=1).values

        # --- 尾巴 22:24 ---
        ar = torch.arange(B, device=dev)
        tail = self.body[ar, (self.body_len - 1).clamp(min=0)]
        twx = tail[:, 0] - head[:, 0]
        twy = tail[:, 1] - head[:, 1]
        obs[:, 22] = (twx * d[:, 0] + twy * d[:, 1]).float() / G
        obs[:, 23] = (-twx * d[:, 1] + twy * d[:, 0]).float() / G

        return obs

    def _obs32(self):
        """32 维投影观测（食物扇区投影式连续感知）。"""
        B, G, dev = self.B, self.G, self.device
        head = self.head
        food = self.food
        d = self.DIRS[self.dir_idx]                       # [B,2]

        obs = torch.zeros(B, 32, dtype=torch.float32, device=dev)

        # --- [0:4] 蛇首方向 one-hot（绝对 4 基本方向）---
        # DIRS = [(0,1), (1,0), (0,-1), (-1,0)] -> idx 0,1,2,3
        obs[:, 0] = ((d[:, 0] == 0) & (d[:, 1] == 1)).float()  # 右
        obs[:, 1] = ((d[:, 0] == 1) & (d[:, 1] == 0)).float()  # 下
        obs[:, 2] = ((d[:, 0] == 0) & (d[:, 1] == -1)).float() # 左
        obs[:, 3] = ((d[:, 0] == -1) & (d[:, 1] == 0)).float() # 上

        # --- [4:8] 蛇尾方向 one-hot ---
        tail = self.body[torch.arange(B, device=dev), (self.body_len - 1).clamp(min=0)]
        prev = self.body[torch.arange(B, device=dev), (self.body_len - 2).clamp(min=0)]
        tail_dr = prev[:, 0] - tail[:, 0]
        tail_dc = prev[:, 1] - tail[:, 1]
        obs[:, 4] = ((tail_dr == 0) & (tail_dc == 1)).float()   # 右
        obs[:, 5] = ((tail_dr == 1) & (tail_dc == 0)).float()   # 下
        obs[:, 6] = ((tail_dr == 0) & (tail_dc == -1)).float()  # 左
        obs[:, 7] = ((tail_dr == -1) & (tail_dc == 0)).float()  # 上

        # --- 计算 8 个相对方向 ---
        # 相对方向：[前, 左前, 左, 左后, 后, 右后, 右, 右前]
        left = self.DIRS[(self.dir_idx + 3) % 4]
        right = self.DIRS[(self.dir_idx + 1) % 4]
        d8 = [
            d,                           # 0 前
            d + left,                    # 1 左前
            left,                        # 2 左
            left - d,                    # 3 左后
            -d,                          # 4 后
            right - d,                   # 5 右后（d-left 恒等于 d+right=右前，从未扫过右后）
            right,                       # 6 右
            d + right,                   # 7 右前
        ]

        # --- [8:16] 食物 8 扇区投影式连续感知（忽略遮蔽，永远可见）---
        # channel_i = OBS_FOOD_SCALE * max(0, v·û_i) / |v|² = K·max(0, cosθ)/dist
        # 幅度缩放 K：随机权重下投影信号需足够强才能触发转向反射（见 Config 注释）
        ar = torch.arange(B, device=dev)
        vr = (food[:, 0] - head[:, 0]).float()
        vc = (food[:, 1] - head[:, 1]).float()
        d2 = (vr * vr + vc * vc).clamp(min=1.0)           # |v|²，食物不在头上故 >= 1
        k_scale = float(getattr(self.cfg, 'OBS_FOOD_SCALE', 1.0))
        for i in range(8):
            dx = d8[i][:, 0].float()
            dy = d8[i][:, 1].float()
            norm = torch.sqrt(dx * dx + dy * dy)           # 1 或 √2（逐个体朝向）
            dot = (vr * dx + vc * dy) / norm
            obs[:, 8 + i] = torch.clamp(dot, min=0.0) / d2 * k_scale

        # --- [16:24] 自身 8 扇区距离倒数 ---
        # 身体掩码向量化（与 _occupancy_flat 同技巧；scatter_add 累加语义保证重复/填充索引幂等，
        # 已用蛇长2-20随机自回避蛇身验证与逐个体循环逐位一致）
        flat_body = self.body[:, :, 0] * G + self.body[:, :, 1]
        seg_idx = torch.arange(self.MAXLEN, device=dev)
        seg_valid = (seg_idx[None, :] >= 1) & (seg_idx[None, :] < self.body_len[:, None])
        bf = torch.zeros(B, G * G, dtype=torch.long, device=dev)
        bf.scatter_add_(1, flat_body.clamp(max=G * G - 1), seg_valid.long())
        body_set_mask = bf.view(B, G, G) > 0

        for i in range(8):
            ddr, ddc = d8[i][:, 0], d8[i][:, 1]
            dist = torch.full((B,), float(G + 1), device=dev)
            prev_ok = torch.ones(B, dtype=torch.bool, device=dev)
            for k in range(1, G + 1):
                r = head[:, 0] + ddr * k
                c = head[:, 1] + ddc * k
                inb = (r >= 0) & (r < G) & (c >= 0) & (c < G)
                r_clamp = r.clamp(0, G - 1)
                c_clamp = c.clamp(0, G - 1)
                hit_body = body_set_mask[ar, r_clamp, c_clamp] & inb
                dist = torch.where(prev_ok & hit_body,
                                   torch.full_like(dist, float(k)), dist)
                prev_ok = prev_ok & (~hit_body) & inb
            obs[:, 16 + i] = torch.where(dist <= G, 1.0 / dist, torch.zeros_like(dist))
        k_self = float(getattr(self.cfg, 'OBS_SELF_SCALE', 1.0))
        if k_self != 1.0:
            obs[:, 16:24] = obs[:, 16:24] * k_self

        # --- [24:32] 障碍 8 扇区距离倒数 ---
        for i in range(8):
            ddr, ddc = d8[i][:, 0], d8[i][:, 1]
            dist = torch.full((B,), float(G), device=dev)
            prev_ok = torch.ones(B, dtype=torch.bool, device=dev)
            for k in range(1, G + 1):
                r = head[:, 0] + ddr * k
                c = head[:, 1] + ddc * k
                inb = (r >= 0) & (r < G) & (c >= 0) & (c < G)
                r_clamp = r.clamp(0, G - 1)
                c_clamp = c.clamp(0, G - 1)
                hit_wall = ~inb
                hit_body = body_set_mask[ar, r_clamp, c_clamp] & inb
                blocked = hit_wall | hit_body
                dist = torch.where(prev_ok & blocked,
                                   torch.full_like(dist, float(k)), dist)
                prev_ok = prev_ok & (~blocked)
            obs[:, 24 + i] = 1.0 / dist

        # 身后约定：障碍数 = sqrt(蛇身长度/格子度)
        obs[:, 28] = torch.sqrt(self.body_len.float().clamp(min=1)/self.G)
        k_obs = float(getattr(self.cfg, 'OBS_OBSTACLE_SCALE', 1.0))
        if k_obs != 1.0:
            obs[:, 24:32] = obs[:, 24:32] * k_obs

        return obs

    def _cast_ray(self, rd, occ):
        """沿射线扫描，返回 (free_path 归一化长度, food_signal)。"""
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
        if self.mode24:
            # 24 维：5 条射线的 food_signal 通道（test7 原版口径）
            return obs[:, 5:14:2].max(dim=1).values > 0.0
        # 32 维投影观测：[8:16] 食物扇区投影（恒可见）
        return obs[:, 8:16].max(dim=1).values > 0.0

    # ---------- 步进 ----------
    def step(self, actions):
        B, dev = self.B, self.device
        # 转向：0=直行 1=左转 2=右转
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
        self._place_food_after_eat(ate)

        starve = self.steps_wo_food > (float(getattr(self.cfg, 'STARVE_SLOPE', 3.0))
                                       * self.body_len.float() + 20)
        self.alive = alive_f & (~crash) & (~starve)
        # 死因标记（首次死亡时记录）：1撞墙 2撞己 3饿死
        crash_wall = alive_f & out_b
        crash_self = alive_f & (~out_b) & hit
        starve_now = alive_f & (~crash) & starve
        died_now = torch.where(crash_wall, 1,
                               torch.where(crash_self, 2,
                                           torch.where(starve_now, 3,
                                                       torch.zeros_like(self.steps))))
        self.died = torch.where((self.died == 0) & (died_now > 0), died_now, self.died)
        self.ate = ate

    def all_done(self):
        return not bool(self.alive.any().item())


# ==========================================
# 3. 批量前向（无激素 E-I 动力学）
# ==========================================
def forward_batch(pop, obs, E, I, st, press, cfg):
    """单次 E-I 迭代（B 个个体并行，无激素支路）。

    obs [B,O] E/I/st [B,N] press [B] → logits [B,A], E_new, I_new, st
    press 为转向压力累积器（每环境步更新，K 帧思考内恒定），只罚转向 logits。
    """
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
    logits[:, 1:] = logits[:, 1:] - cfg.FATIGUE_TURN_GAIN * press.unsqueeze(1).to(logits.dtype)
    return logits, E_new, I_new, st


def update_fatigue(press, action, decay=None):
    """转向压力更新：press = DECAY*press + (act!=0)。任意转弯（左/右）都累积压力。"""
    if decay is None:
        decay = 0.7
    return press * decay + (action != 0).to(press.dtype)


def deliberate_batch(pop, obs, E, I, st, press, cfg):
    """K 倍帧率思考：内部迭代 K 次，logits 平均后 argmax（与 test5d 一致）。
    滞回解码：转向动作 logits 统一减 TURN_MARGIN（默认 0=关；直行不罚）。"""
    K = cfg.FRAME_RATE
    logits_sum = None
    for k in range(K):
        o = obs * (cfg.INPUT_DECAY ** k)
        logits, E, I, st = forward_batch(pop, o, E, I, st, press, cfg)
        logits_sum = logits if logits_sum is None else logits_sum + logits
    margin = float(getattr(cfg, 'TURN_MARGIN', 0.0))
    if margin > 0:
        bias = logits_sum.new_tensor([0.0, 1.0, 1.0]).unsqueeze(0)
        logits_sum = logits_sum - margin * bias
    action = torch.argmax(logits_sum, dim=1)
    return action, E, I, st


# ==========================================
# 4. 种群评估（GPU 全并行）
# ==========================================
def _eval_chunk(pop, cfg):
    """对单个子种群（B = pop.P）并行评估 cfg.EVAL_EPISODES 局。"""
    B = pop.P
    dev = pop.device
    N = pop.N
    A = pop.A
    env = BatchedSnakeEnv(cfg, B, dev)
    half = torch.float16 if getattr(cfg, 'USE_FP16', True) else torch.float32

    tot_food = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_seen = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_unseen = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_last = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_prox = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_alive = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_wall = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_self = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_starve = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_act1 = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_act2 = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_turn_last = torch.zeros(B, dtype=torch.float32, device=dev)

    for _ in range(cfg.EVAL_EPISODES):
        env.reset()
        E = torch.zeros(B, N, dtype=half, device=dev)
        I = torch.zeros(B, N, dtype=half, device=dev)
        st = torch.zeros(B, N, dtype=half, device=dev)
        press = torch.zeros(B, dtype=torch.float32, device=dev)
        last = torch.zeros(B, dtype=torch.float32, device=dev)
        turn_cnt = torch.zeros(B, dtype=torch.float32, device=dev)
        turn_last = torch.zeros(B, dtype=torch.float32, device=dev)

        for t in range(cfg.MAX_STEPS):
            al = env.alive
            obs = env.obs().to(half)
            sees = env.sees_food(obs)
            tot_seen += (al & sees).float()
            tot_unseen += (al & (~sees)).float()
            # 接近度：存活期间的 1/欧氏距离（食物永远可见时的方向性梯度）
            fr = (env.food[:, 0] - env.head[:, 0]).float()
            fc = (env.food[:, 1] - env.head[:, 1]).float()
            dist = torch.sqrt(fr * fr + fc * fc).clamp(min=1.0)
            tot_prox += al.float() / dist
            tot_alive += al.float()

            act, E, I, st = deliberate_batch(pop, obs, E, I, st, press, cfg)
            press = update_fatigue(press, act, decay=float(cfg.FATIGUE_TURN_DECAY))
            tot_act1 += (al & (act == 1)).float()
            tot_act2 += (al & (act == 2)).float()
            turn_cnt += (al & (act != 0)).float()

            env.step(act)
            ate_now = al & env.ate
            tot_food += ate_now.float()
            last = torch.where(ate_now, torch.full_like(last, float(t + 1)), last)
            turn_last = torch.where(ate_now, turn_cnt, turn_last)
            if env.all_done():
                break
        tot_last += last
        tot_turn_last += turn_last
        tot_wall += (env.died == 1).float()
        tot_self += (env.died == 2).float()
        tot_starve += (env.died == 3).float()

    E_ = float(cfg.EVAL_EPISODES)
    prox = tot_prox / torch.clamp(tot_alive, min=1e-6)
    # [8:10] 每局平均转向次数（act1/act2）；[10] 吃最后一颗食物为止的转向数
    metrics = torch.stack((tot_food, tot_seen, tot_unseen, tot_last, prox,
                           tot_wall, tot_self, tot_starve,
                           tot_act1 / E_, tot_act2 / E_, tot_turn_last / E_), dim=1)
    metrics[:, :4] /= E_

    # 单侧转弯判死（默认关闭：早期随机个体普遍摇头，判罚干扰初期筛选）
    m = metrics.cpu().numpy()
    if getattr(cfg, 'ONE_SIDED_TURN_DEATH', False):
        turn_lim = max(cfg.EVAL_EPISODES, 1)
        c1 = tot_act1.cpu().numpy()
        c2 = tot_act2.cpu().numpy()
        death = ((c1 > turn_lim) | (c2 > turn_lim)) & ((c1 == 0) | (c2 == 0))
        m[death] = (0.0, 99999.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return torch.from_numpy(m).float()


def evaluate_population_gpu(pop, cfg):
    """全种群评估：按显存分块并行，返回 [P,11] = (food, seen, unseen, steps_last, prox,
    wall_die, self_die, starve_die, act1_avg, act2_avg, turns_last)。"""
    if getattr(cfg, 'USE_FP16', True):
        pop.fp16()
    batch = _auto_eval_batch(cfg, pop.device)
    if batch >= pop.P:
        pop.refresh_eff()
        return _eval_chunk(pop, cfg)

    metrics = torch.zeros(pop.P, 11)
    for lo in range(0, pop.P, batch):
        sub = pop[slice(lo, min(lo + batch, pop.P))]
        sub.refresh_eff()
        m = _eval_chunk(sub, cfg)
        metrics[lo:lo + sub.P] = m.cpu()
    return metrics


# ==========================================
# 5. 进化（GPU 向量化交叉/变异）
# ==========================================
def evolve_topology_gpu(pop, metrics, cfg, gen=0):
    """进化下一代（语义与 test5d.evolve_topology 完全一致，向量化到 GPU）。

    返回新的 GeneStack（P 行）：ELITE 精英 + (P-ELITE) 后代。
    """
    P = pop.P
    N = pop.N
    dev = pop.device
    active = _freeze_active_groups(gen, cfg)
    has_g1 = 'G1' in active
    has_g2 = 'G2' in active

    # --- 变异强度固定为常量（不再随代数衰减，避免后期筛选停滞）---
    cos_factor = 1.0
    dyn_factor = 1.0
    topo_mut_prob = cfg.TOPOLOGY_MUT_PROB * cos_factor
    mask_mut_rate = cfg.MUT_RATE * cos_factor
    weight_mut_frac = cfg.WEIGHT_MUT_FRAC * cos_factor
    weight_mut_std = cfg.WEIGHT_MUT_STD * cos_factor
    tau_mut_std = cfg.TAU_E_MUT_STD * dyn_factor
    w_ei_std = cfg.W_EI_MUT_STD * dyn_factor
    w_ie_std = cfg.W_IE_MUT_STD * dyn_factor

    # --- 精英排序（按 FIT_MODE 分发适应度）---
    mn = metrics.cpu().numpy()
    key_fn = _make_key_fn(cfg)
    order = sorted(range(P), key=lambda i: key_fn(mn[i]),
                   reverse=True)
    elite_idx = order[:cfg.ELITE_SIZE]
    elites = pop[elite_idx]

    # 精英直接进入下一代
    new_pop = pop.empty()
    children = pop.empty(B=P - cfg.ELITE_SIZE)
    for g in GeneStack.GENES:
        setattr(children, g, torch.empty(P - cfg.ELITE_SIZE, *getattr(elites, g).shape[1:],
                                         dtype=getattr(elites, g).dtype, device=dev))
    for g in GeneStack.GENES:
        setattr(new_pop, g, torch.cat([getattr(elites, g).clone(), getattr(children, g)], dim=0))

    # --- 后代：父代采样 + 交叉 + 变异 ---
    E = cfg.ELITE_SIZE
    B2 = P - E
    p1_idx = torch.randint(0, E, (B2,), device=dev)
    p2_idx = torch.randint(0, E, (B2,), device=dev)
    p2_idx = torch.where(p2_idx == p1_idx, (p1_idx + 1) % E, p2_idx)

    p1 = elites[p1_idx]
    p2 = elites[p2_idx]

    with torch.no_grad():
        # ---- G1 交叉：结构组（掩码 + 权重 + 输出偏置）----
        if has_g1:
            col_mask = torch.rand(B2, N, device=dev) > 0.5          # [B2,N]
            row_mask = col_mask.unsqueeze(1)                        # [B2,N,1]
            col_mask_2d = col_mask.unsqueeze(2)                     # [B2,1,N]
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

        # ---- G2 交叉：动力学组 ----
        if has_g2:
            col2 = torch.rand(B2, N, device=dev) > 0.5
            children.tau_e = torch.where(col2, p1.tau_e, p2.tau_e)
            children.w_ei = torch.where(col2, p1.w_ei, p2.w_ei)
            children.w_ie = torch.where(col2, p1.w_ie, p2.w_ie)

        # ---- 变异（逐组独立，冻结组跳过）----
        if has_g1:
            pick = torch.randint(0, 3, (B2,), device=dev)
            topo_gate = torch.rand(B2, device=dev) < topo_mut_prob
            for ai, attr in enumerate(GeneStack.G1_MASKS):
                sel = (pick == ai) & topo_gate
                if bool(sel.any().item()):
                    t = getattr(children, attr)
                    flip = torch.rand_like(t) < mask_mut_rate
                    setattr(children, attr,
                            torch.where(sel[:, None, None] & flip, 1.0 - t, t))
            for attr in GeneStack.G1_WEIGHTS:
                t = getattr(children, attr)
                noise = torch.randn_like(t) * weight_mut_std
                m = (torch.rand_like(t) < weight_mut_frac).to(t.dtype)
                setattr(children, attr, t + noise * m)

        if has_g2:
            children.tau_e = torch.clamp(children.tau_e + torch.randn_like(children.tau_e) * tau_mut_std,
                                         cfg.TAU_E_MIN, cfg.TAU_E_MAX)
            children.w_ei = torch.clamp(children.w_ei + torch.randn_like(children.w_ei) * w_ei_std,
                                        cfg.W_EI_MIN, cfg.W_EI_MAX)
            children.w_ie = torch.clamp(children.w_ie + torch.randn_like(children.w_ie) * w_ie_std,
                                        cfg.W_IE_MIN, cfg.W_IE_MAX)

    for g in GeneStack.GENES:
        setattr(new_pop, g, torch.cat([getattr(elites, g).clone(), getattr(children, g)], dim=0))
    new_pop.dtype = pop.dtype
    return new_pop


# ==========================================
# 6. 保存 / 加载（断点 + 最优模型，兼容 test5d 种子）
# ==========================================
def load_best_state(path, cfg):
    """读取 test5d 格式的最优模型文件，返回个体状态 dict（None=不可用）。"""
    if not os.path.exists(path):
        return None
    try:
        data = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"  警告: 模型 {path} 读取失败 ({e})，已忽略种子")
        return None
    saved_cfg = data.get('config', {})
    if saved_cfg:
        if (saved_cfg.get('NUM_COLUMNS') != cfg.NUM_COLUMNS or
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
        'config': _cfg_dict(cfg),
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }, path)


def save_checkpoint7(path, cfg, next_gen, pop, history,
                     cum_eval_time, cum_evolve_time,
                     best_state, best_food, best_seen, best_unseen, best_last, best_prox):
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
        'best_last': float(best_last),
        'best_prox': float(best_prox),
        'config': _cfg_dict(cfg),
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


# ==========================================
# 7. 主循环
# ==========================================
def run_training(cfg):
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
    best_last = 0.0
    best_prox = 0.0
    # 跨代最优的指标行（seen=99999 → 适应度 -1e9，保证首代必然刷新）
    best_row = np.zeros(11)
    best_row[1] = 99999.0
    latest_gen_best_state = None
    latest_gen_best_food = -1.0
    latest_gen_best_seen = 0.0
    latest_gen_best_unseen = 0.0

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
            best_last = float(ck.get('best_last', 0.0))
            best_prox = float(ck.get('best_prox', 0.0))
            if best_state is not None:
                random.setstate(ck['random_state'])
                torch.set_rng_state(ck['torch_rng_state'])
            print(f"=== 检测到断点 [{cfg.CHECKPOINT_PATH}] ===")
            print(f"  已完成 {start_gen} 代 -> 从第 {start_gen} 代接续 | "
                  f"历史最优: Food={best_food:.1f} | 已耗时 {cum_eval_time + cum_evolve_time:.1f}s")

    if pop.M_in is None:
        t0 = time.perf_counter()
        print("初始化种群（GPU 随机初始化）...")
        pop.random_init()

        # ---- 种子继承：把已有最优模型注入种群首位 ----
        if cfg.SEED_FROM_BEST:
            for sp in (cfg.SEED_MODEL_PATH, cfg.SEED_MODEL_PATH2):
                seed = load_best_state(sp, cfg)
                if seed is not None:
                    st, s_food, s_steps = seed
                    pop.set_individual_from_state(0, st)
                    # 不写入 best_*：best_last=0 会使适应度键虚高（eff=food/max(0,1)），
                    # 后代真实成绩永远追不上，跨代最优被冻结为种子。
                    # 种子留在种群首位，第 0 代评估中自然参与 best_idx 竞争。
                    print(f"  [Seed] 已注入 {sp} 作为种群种子 "
                          f"(Food={s_food:.1f}, Steps={s_steps:.1f})")
                    break
            else:
                print("  [Seed] 未发现可用的最优模型种子，全新随机初始化")
        print(f"  初始化完成 ({time.perf_counter() - t0:.1f}s)")

        save_checkpoint7(cfg.CHECKPOINT_PATH, cfg, 0, pop, history,
                         cum_eval_time, cum_evolve_time,
                         best_state, best_food, best_seen, best_unseen, best_last, best_prox)

    # 退火基准增益（捕获 CLI 覆盖后的初始值，主循环按代计算有效增益）
    cfg._FATIGUE_GAIN0 = float(cfg.FATIGUE_TURN_GAIN)

    try:
        for gen in range(start_gen, cfg.GENERATIONS):
            t_eval = time.perf_counter()
            # 脚手架退火：gain_t = max(FLOOR, GAIN·FACTOR^(gen//PERIOD))，逐周期撤除拐杖
            cfg.FATIGUE_TURN_GAIN = max(
                float(cfg.FATIGUE_FLOOR),
                float(cfg._FATIGUE_GAIN0) * (cfg.ANNEAL_FACTOR ** (gen // cfg.ANNEAL_PERIOD)))
            metrics = evaluate_population_gpu(pop, cfg)
            eval_time = time.perf_counter() - t_eval
            cum_eval_time += eval_time

            mn = metrics.numpy()
            key_fn = _make_key_fn(cfg)
            best_idx = max(range(cfg.POP_SIZE), key=lambda i: key_fn(mn[i]))
            b_food, b_seen, b_unseen, b_last, b_prox = (
                float(mn[best_idx][0]), float(mn[best_idx][1]),
                float(mn[best_idx][2]), float(mn[best_idx][3]), float(mn[best_idx][4]))
            b_turn = (float(mn[best_idx][8]) + float(mn[best_idx][9])) / max(b_seen + b_unseen, 1.0)
            avg_food = float(np.mean(mn[:, 0]))

            # 精英均值（观察选择压力是否在累积）
            elite_order = sorted(range(cfg.POP_SIZE),
                                 key=lambda i: key_fn(mn[i]),
                                 reverse=True)[:cfg.ELITE_SIZE]
            elite_avg_food = float(np.mean([mn[i][0] for i in elite_order]))
            elite_avg_prox = float(np.mean([mn[i][4] for i in elite_order]))

            history['gen'].append(gen)
            history['best_food'].append(b_food)
            history['avg_food'].append(avg_food)
            history['best_seen'].append(b_seen)
            history['best_unseen'].append(b_unseen)

            # 跨代跟踪历史最优（按当前 FIT_MODE 的适应度比较）
            if key_fn(mn[best_idx]) > key_fn(best_row):
                best_food = b_food
                best_seen = b_seen
                best_unseen = b_unseen
                best_last = b_last
                best_prox = b_prox
                best_row = mn[best_idx].copy()
                best_state = pop.individual_state(best_idx, use_half=False)

            # 记录当前代最优（进化会覆盖种群，须在 evolve 前快照）
            latest_gen_best_state = pop.individual_state(best_idx, use_half=False)
            latest_gen_best_food = b_food
            latest_gen_best_seen = b_seen
            latest_gen_best_unseen = b_unseen

            if gen < cfg.GENERATIONS - 1:
                t_ev = time.perf_counter()
                pop = evolve_topology_gpu(pop, metrics, cfg, gen=gen)
                evolve_time = time.perf_counter() - t_ev
                cum_evolve_time += evolve_time
            else:
                evolve_time = 0.0

            if (gen + 1) % cfg.PRINT_HISTORY_EVERY == 0:
                b_eff = b_food / max(b_last, 1.0)
                b_turn = (float(mn[best_idx][8]) + float(mn[best_idx][9])) / max(b_seen + b_unseen, 1.0)
                b_fit = key_fn(mn[best_idx])
                b_fit = b_fit if np.isscalar(b_fit) else b_fit[0]
                avg_wall = float(np.mean(mn[:, 5]))
                avg_self = float(np.mean(mn[:, 6]))
                avg_starve = float(np.mean(mn[:, 7]))
                print(f"Gen {gen + 1}/{cfg.GENERATIONS} | Fat: {cfg.FATIGUE_TURN_GAIN:.3f} | "
                      f"BestFood: {b_food:.2f} | BestSeen: {b_seen:.1f} | "
                      f"BestUnseen: {b_unseen:.1f} | BestEff: {b_eff:.3f} | "
                      f"BestTurn: {b_turn:.3f} | BestFit: {b_fit:.3f} | "
                      f"BestProx: {b_prox:.3f} | AvgFood: {avg_food:.2f} | "
                      f"EliteFood: {elite_avg_food:.2f} | EliteProx: {elite_avg_prox:.3f} | "
                      f"Die(W/S/St): {avg_wall:.2f}/{avg_self:.2f}/{avg_starve:.2f} | "
                      f"eval {eval_time:.1f}s / evolve {evolve_time:.1f}s")

            if (gen + 1) % cfg.CHECKPOINT_INTERVAL == 0:
                save_checkpoint7(cfg.CHECKPOINT_PATH, cfg, gen + 1, pop, history,
                                 cum_eval_time, cum_evolve_time,
                                 best_state, best_food, best_seen, best_unseen, best_last, best_prox)

    except KeyboardInterrupt:
        # 中断代未完成（评估/进化被切断），续训从该代重跑而非跳到下一代
        nxt = gen if 'gen' in dir() else start_gen
        print("\n训练被中断 (Ctrl+C)，正在保存断点...")
        save_checkpoint7(cfg.CHECKPOINT_PATH, cfg, nxt, pop, history,
                         cum_eval_time, cum_evolve_time,
                         best_state, best_food, best_seen, best_unseen, best_last, best_prox)
        print(f"断点已保存: {cfg.CHECKPOINT_PATH} (下次从第 {nxt} 代接续)")
        sys.exit(0)

    t_delta = time.perf_counter() - t_program
    print(f"\nTotal runtime: {t_delta:.1f}s "
          f"(eval {cum_eval_time:.1f}s / evolve {cum_evolve_time:.1f}s)")

    # ---- 训练完成：保存最优模型 ----
    if best_state is None:
        best_idx = 0
        best_state = pop.individual_state(best_idx, use_half=False)
    save_best_model(cfg.BEST_MODEL_PATH, best_state, cfg, best_food, best_seen, best_unseen)
    print(f"\n最优模型已保存: {cfg.BEST_MODEL_PATH} "
          f"(Food={best_food:.2f}, Seen={best_seen:.1f}, Unseen={best_unseen:.1f})")

    if latest_gen_best_state is not None:
        save_best_model(cfg.LATEST_GEN_BEST_MODEL_PATH, latest_gen_best_state, cfg,
                        latest_gen_best_food, latest_gen_best_seen, latest_gen_best_unseen)
        print(f"最新一代最优模型已保存: {cfg.LATEST_GEN_BEST_MODEL_PATH} "
              f"(Food={latest_gen_best_food:.2f}, Seen={latest_gen_best_seen:.1f}, "
              f"Unseen={latest_gen_best_unseen:.1f})")
    

    if os.path.exists(cfg.CHECKPOINT_PATH):
        os.remove(cfg.CHECKPOINT_PATH)
        print(f"训练已完成，已删除临时断点: {cfg.CHECKPOINT_PATH}")

    # ---- 历史曲线（无头环境只存图，不 show）----
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
        hist_path = cfg.CHECKPOINT_PATH.replace('_checkpoint', '_history').replace('.pth', '.png')
        fig.savefig(hist_path, dpi=100)
        plt.close(fig)
        print(f"历史曲线已保存: {hist_path}")
    except Exception as e:
        print(f"(matplotlib 曲线跳过: {e})")

    print(f"\n--- Best Brain Summary ---")
    st = best_state
    print(f"Input connections active:   {st['M_in'].sum().item()}/{pop.N * pop.O}")
    print(f"Internal connections active: {st['M_rec'].sum().item()}/{pop.N * pop.N}")
    print(f"Output connections active:   {st['M_out'].sum().item()}/{pop.A * pop.N}")
    print(f"tau_e range: [{st['tau_e_init'].min().item():.3f}, {st['tau_e_init'].max().item():.3f}]")
    print(f"Wei range:   [{st['w_ei'].min().item():.3f}, {st['w_ei'].max().item():.3f}]")
    print(f"Wie range:   [{st['w_ie'].min().item():.3f}, {st['w_ie'].max().item():.3f}]")


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
    press = torch.zeros(1, dtype=torch.float32, device=dev)
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
    for s in range(max_steps):
        obs = env.obs().to(pop.dtype)
        act, E, I, stt = deliberate_batch(pop, obs, E, I, stt, press, cfg)
        press = update_fatigue(press, act, decay=float(cfg.FATIGUE_TURN_DECAY))
        before = ep_food
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


# ==========================================
# 8. 入口
# ==========================================
def make_smoke_config():
    cfg = Config()
    cfg.POP_SIZE = 16
    cfg.NUM_COLUMNS = 24
    cfg.OBS_DIM = 32
    cfg.ACTION_DIM = 3
    cfg.GENERATIONS = 3
    cfg.ELITE_SIZE = 6
    cfg.EVAL_EPISODES = 1
    cfg.MAX_STEPS = 60
    cfg.FRAME_RATE = 2
    cfg.CHECKPOINT_INTERVAL = 2
    cfg.CHECKPOINT_PATH = 'test7f_smoke_checkpoint.pth'
    cfg.BEST_MODEL_PATH = 'test7f_smoke_best.pth'
    cfg.LATEST_GEN_BEST_MODEL_PATH = 'test7f_smoke_latest_gen_best.pth'
    cfg.SEED_FROM_BEST = False
    cfg.EVAL_BATCH = 16
    cfg.PRINT_HISTORY_EVERY = 1
    return cfg


def main():
    ap = argparse.ArgumentParser(description='test7e — 整局转弯密度计价（基于 test7d 诊断）')
    ap.add_argument('--smoke', action='store_true', help='小规模快速自检')
    ap.add_argument('--gens', type=int, default=None)
    ap.add_argument('--pop', type=int, default=None)
    ap.add_argument('--columns', type=int, default=None)
    ap.add_argument('--episodes', type=int, default=None)
    ap.add_argument('--max-steps', type=int, default=None)
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--fit-mode', type=str, default=None, choices=['econ', 'additive', 'tuple'],
                    help='适应度模式（默认 econ）；决定文件前缀 test7f_econ_*/test7f_add_*')
    ap.add_argument('--turn-cost', type=float, default=None,
                    help='econ 臂：一次转弯折算的额外步数成本（默认 1.0）')
    ap.add_argument('--step-ref', type=float, default=None,
                    help='econ 臂：每颗食物参考步数（默认 10.0）')
    ap.add_argument('--turn-penalty', type=float, default=None,
                    help='additive 臂：转向占比扣分权重（默认 3.0）')
    ap.add_argument('--one-sided-death', action='store_true',
                    help='重新开启单侧转弯判死（默认关闭）')
    ap.add_argument('--starve-slope', type=float, default=None,
                    help='饿死斜率（默认 3.0；test7a 旧值为 2.0）')
    ap.add_argument('--eff-weight', type=float, default=None,
                    help='additive 臂：效率项权重（默认 0.3）')
    ap.add_argument('--turn-margin', type=float, default=None,
                    help='滞回解码：转向 logits 门槛（默认 0=关，诊断显示非抖动）')
    ap.add_argument('--crowd-weight', type=float, default=None,
                    help='拥挤加权 CROWD_W：转向成本 ×(1+W·food/50)（默认 0=关）')
    ap.add_argument('--anneal-period', type=int, default=None,
                    help='疲劳退火周期：每 N 代降一档（默认 20）')
    ap.add_argument('--anneal-factor', type=float, default=None,
                    help='疲劳退火乘子（默认 0.7）')
    ap.add_argument('--fatigue-floor', type=float, default=None,
                    help='疲劳增益下限（默认 0.05）')
    ap.add_argument('--turn-gain', type=float, default=None,
                    help='转向疲劳增益（默认 0.2）')
    ap.add_argument('--turn-decay', type=float, default=None,
                    help='转向压力衰减（默认 0.9）')
    ap.add_argument('--seed-model', type=str, default=None,
                    help='指定种子模型路径（覆盖 SEED_MODEL_PATH 并启用种子注入）')
    ap.add_argument('--play', action='store_true', help='加载最优模型播放一局')
    args = ap.parse_args()

    if args.smoke:
        cfg = make_smoke_config()
    else:
        cfg = Config()
    if args.gens:
        cfg.GENERATIONS = args.gens
    if args.pop:
        cfg.POP_SIZE = args.pop
    if args.columns:
        cfg.NUM_COLUMNS = args.columns
        cfg.INIT_DENSITY = min(0.15, 40.0 / cfg.NUM_COLUMNS)
    if args.episodes:
        cfg.EVAL_EPISODES = args.episodes
    if args.max_steps:
        cfg.MAX_STEPS = args.max_steps
    if args.device:
        cfg.DEVICE = args.device
    if args.fit_mode:
        cfg.FIT_MODE = args.fit_mode
        # 按臂切换文件前缀，双臂互不干扰（smoke 模式保持独立路径，不触碰正式臂文件）
        if not args.smoke:
            arm = 'econ' if args.fit_mode == 'econ' else ('add' if args.fit_mode == 'additive' else 'tup')
            cfg.CHECKPOINT_PATH = f'test7f_{arm}_checkpoint.pth'
            cfg.BEST_MODEL_PATH = f'test7f_{arm}_best_model.pth'
            cfg.LATEST_GEN_BEST_MODEL_PATH = f'test7f_{arm}_latest_gen_best.pth'
    if args.turn_cost is not None:
        cfg.TURN_COST = args.turn_cost
    if args.turn_margin is not None:
        cfg.TURN_MARGIN = args.turn_margin
    if args.crowd_weight is not None:
        cfg.CROWD_W = args.crowd_weight
    if args.step_ref is not None:
        cfg.STEP_REF = args.step_ref
    if args.turn_penalty is not None:
        cfg.TURN_PENALTY = args.turn_penalty
    if args.one_sided_death:
        cfg.ONE_SIDED_TURN_DEATH = True
    if args.starve_slope is not None:
        cfg.STARVE_SLOPE = args.starve_slope
    if args.eff_weight is not None:
        cfg.FOOD_EFF_WEIGHT = args.eff_weight
    if args.turn_gain is not None:
        cfg.FATIGUE_TURN_GAIN = args.turn_gain
    if args.anneal_period is not None:
        cfg.ANNEAL_PERIOD = args.anneal_period
    if args.anneal_factor is not None:
        cfg.ANNEAL_FACTOR = args.anneal_factor
    if args.fatigue_floor is not None:
        cfg.FATIGUE_FLOOR = args.fatigue_floor
    if args.turn_decay is not None:
        cfg.FATIGUE_TURN_DECAY = args.turn_decay
    if args.seed_model:
        cfg.SEED_FROM_BEST = True
        cfg.SEED_MODEL_PATH = args.seed_model

    if args.play:
        play_best(cfg)
        return

    run_training(cfg)


if __name__ == '__main__':
    main()
