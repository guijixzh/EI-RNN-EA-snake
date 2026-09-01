# ==========================================
# test10_lunar.py —— LunarLander-v3 EI-RNN 训练（基于 test7b 模板）
#
# 纠正 test5a_lunar 的核心错误（Bootstrap Threshold）：
#   test5a 把 8 维原始物理量直接进网络（无任何定标），低幅通道（腿接触 bit、
#   小角度）振幅 << σ_win=0.1 的自举阈值，随机种群下永远无法翻转 argmax，
#   进化卡死。判据：输入信号振幅 × 权重初始化标准差 > 网络输出阈值。
#   实测脚本 experiments/measure_lunar_threshold.py 在归一化单位下测得
#   各通道阈值 a*≈1.6~3.2，据此定标 CHANNEL_SCALES = [6.4,3.2,6.4,6.4,3.2,6.4,3.2,3.2]。
#
# 沿用 test7b 已验证要素：
#   - GeneStack 整栈张量进化（精英 + 列掩码交叉 + G1/G2 轮换冻结变异）
#   - E-I 双层动力学（BASE_TAU_E=0.7, w_ei=w_ie=2.0, N=256, INIT_DENSITY=0.15）
#   - K 倍帧率思考 deliberate_batch（K=5, INPUT_DECAY=0.9）
#   - FP16 分块全并行评估 + 显存自适应
#
# 环境：gymnasium LunarLander-v3（观测 8 维 / 动作 Discrete(4)），CPU 同步批量步进。
#
# ---- 第 2 轮改造（噪声与盆地诊断，experiments/diagnose_test10_noise.py 实测）----
# 症状：76 代后 avg=-143±5 平台，best 在 [-17,132] 震荡。诊断结论（按重要性）：
#   0. 致命 bug（round-2 smoke 复查时发现）：BatchedLunarEnv.step 从未刷新 obs_np，
#      网络每帧只看到 t=0 初始观测 —— 智能体全程失明，只能输出恒定动作序列，
#      "集体自由落体"（58/60 坠毁、主引擎占空比 0.00、BestActs 恒 [1/0/0/0]）
#      首要是这个机械后果。已修复：活跃个体逐步刷新观测。
#   1. 评估噪声：2 局估计量噪声 std=30（P95-P5=97），与近 30 代 best-avg 差距 168
#      同量级；k=2 时精英排序 Kendall-τ=0.24 / top-1 命中 0.18（≈随机）；k=16 可用。
#      （CRN 同种子 256 槽散度=0：环境确定性成立，公共随机数有效。）
#   2. 适应度地形：shaping 奖励快速接近发射台，点火刹车立刻亏燃料+亏 shaping，
#      软着陆 +200 在"刹车但仍旧坠毁"山谷另一侧 —— 观测修复后仍需梯度整形。
# 三项修复（均有 CLI 开关可 A/B）：
#   F1 公共随机数 CRN：每代每局一个种子，全体个体、所有评估分块共用（--no-crn 关闭）
#   F2 两阶段评估：全种群 K1 局粗评 → 前 STAGE2_CANDIDATES 名用 K2 局新种子精评
#      （--no-stage2 / --stage2-candidates / --stage2-episodes）
#   F3 适应度整形：
#      - 软着陆部分加分 SOFT_*（--no-soft-bonus 关闭）：坠毁局按终局 obs 的腿接触/
#        |vy|/|vx|/角度给 ≤SOFT_BONUS_CAP 分，制造"刹车→坠得更软→加分"的单调梯度，
#        穿过自由落体盆地；成功局不加（env 已给 +100~140，保持排序语义）
#      - 悬停惩罚 HOVER_PENALTY（--hover-penalty）：500 步截断按坠毁同量级结算，
#        防刹车演化后落入"悬停保平安"新陷阱
# 适应度口径 = env reward + 软着陆加分 - 悬停惩罚；成功率仍按原始 env reward>=200 统计。
# 目标：原始 env 平均 reward >= 200。
#
# ---- 第 3 轮改造（test7h 特性移植 + Stage2 逐半精评）----
#   1. 类正态变异强度（test7h #4）：每子代抽 s ~ LogNormal(0, 0.4) clip [0.25,4]，
#      缩放其全部变异算子（掩码/拓扑触发率 ≤0.5、权重比例 ≤1.0、各 std 线性×s）；
#      精英不变异。动机：固定小变异在收敛种群上只能随机游走（test7g 40 分平台教训）。
#   2. 断点保留 + 逐代 history JSON（test7h #6）：完成后写入 next_gen=GENERATIONS
#      终态断点（续训需提高 --gens），history.json 每代落盘。
#   3. Stage2 逐半精评（successive halving）：128×4 -> 48×8 -> 16×16 阶梯，
#      CRN 配对新种子逐轮加局，省 60% 精评开销且头部精度不降。
#
# 运行（需 gymnasium + box2d-py + torch，推荐 conda env_torch）:
#   python test10_lunar.py --smoke      # 管线自检
#   python test10_lunar.py              # 全量训练
#   python test10_lunar.py --play       # 最优模型演示
# ==========================================

import argparse
import json
import math  # noqa: F401  （保持与 test7b 工具函数签名兼容）
import os
import random
import sys
import time

import numpy as np
import torch

import gymnasium as gym

_ONEHOT_ACT = {a: np.eye(a, dtype=np.float64) for a in (2, 3, 4, 5, 6)}


# ==========================================
# 0. 全局配置类
# ==========================================
class Config:
    # --- 进化参数---
    POP_SIZE = 2048
    GENERATIONS = 100
    ELITE_SIZE = 64
    MUT_RATE = 0.05
    TOPOLOGY_MUT_PROB = 0.05
    WEIGHT_MUT_FRAC = 0.2
    WEIGHT_MUT_STD = 0.1
    TAU_E_MUT_STD = 0.05
    # --- 类正态变异强度（每子代因子 s，缩放其全部变异算子；移植自 test7h）---
    # 中位数 1=原强度；σ=0.4 时 P(s>2)≈2.4%（每代 ~43 个大变异），大步长跳出收敛平台
    MUT_SCALE_DIST = 'lognormal'   # 'lognormal' | 'normal'
    MUT_SCALE_SIGMA = 0.4          # lognormal: σ_ln；normal: s~N(1,σ) clip
    MUT_SCALE_MIN = 0.25
    MUT_SCALE_MAX = 4.0
    CYCLE_PATTERN = [('G2', 'G1', 'G3')]

    # --- 无激素 EI-RNN（文件格式兼容字段）---
    TRAIN_HORMONE_NET = False
    HORMONE_NET_HIDDEN = 32

    # --- 环境参数 ---
    ENV_ID = 'LunarLander-v3'
    OBS_DIM = 8
    ACTION_DIM = 4
    EVAL_EPISODES = 3          # K1 粗评局数（E2: 2局噪声 std=30 → 3局 25）
    MAX_STEPS = 500            # 提前截断：未着陆者按悬停惩罚结算
    SUCCESS_REWARD = 200.0     # 及格线（按原始 env reward，不含加分/惩罚）

    # --- F1 公共随机数：每代每局一个种子，全体个体与所有评估分块共用 ---
    USE_CRN = True
    CRN_SEED = 20250829          # 锚点：同代两跑逐位一致（可复现），换代自动换卷

    # --- F2 两阶段评估：粗评全体 -> 逐半精评头部候选（successive halving）---
    # 阶梯 [(保留候选数, 累计局数)]：128×4 -> 48×8 -> 16×16。
    # 末轮 keep=16 配 ELITE_SIZE=64（16 名 16 局 + 48 名 8 局恰好覆盖精英池）；
    # 总开销 832 局 vs 扁平 128×16=2048 局，省 60%，头部精度不降（E3: k=16 可用）
    STAGE2_ENABLED = True
    STAGE2_LADDER = [(128, 4), (48, 8), (16, 16)]

    # --- F3 适应度整形 ---
    # 悬停惩罚：500 步/gym 截断未终局 → reward - HOVER_PENALTY（与坠毁 -100 同量级）
    HOVER_PENALTY = 100.0
    # 软着陆部分加分（仅坠毁局，用撞击前一帧观测，成功局不加保持排序语义）：
    #   30*腿数 + 40*(1-|vy|) + 20*(1-|vx|) + 20*(1-|angle|/(π/2))，封顶 120
    #   自由落体坠毁撞击前 vy≈3-5 → 加分≈0-40；刹车后软坠毁可拿 60-80 → 单调梯度。
    #   注意不能用终局帧观测：Box2D 接触冲量会把速度瞬间归零产生虚假高分。
    SOFT_LAND_BONUS = True
    SOFT_W_LEGS = 30.0
    SOFT_W_VY = 40.0
    SOFT_W_VX = 20.0
    SOFT_W_ANG = 20.0
    SOFT_BONUS_CAP = 120.0

    # --- 适应度口径版本（改动 F3 权重/惩罚时 +1；不匹配的旧断点保留种群但重置统计）---
    FITNESS_VERSION = 2

    # --- 观测定标（对 test5a 的核心纠正）---
    # 逐维物理范围（归一到 ±1）：x位置、y高度、vx、vy、角度、角速度、左腿、右腿
    OBS_RANGES = [1.5, 1.5, 5.0, 5.0, math.pi, 10.0, 1.0, 1.0]
    # 每通道放大系数 —— 来自 experiments/measure_lunar_threshold.py 实测阈值 a*，
    # 取 max(2*a*, 1)：保证 归一化振幅 × σ_win(0.1) > 决策边界 M。
    CHANNEL_SCALES = [6.4, 3.2, 6.4, 6.4, 3.2, 6.4, 3.2, 3.2]

    # --- 脑结构参数 ---
    NUM_COLUMNS = 256
    INIT_DENSITY = 0.15

    # --- 适应度 ---
    SINGLE_ACTION_FRAC = 0.9   # 单一动作占比超过此值且…
    COLLAPSE_REWARD = -190.0   # …平均 reward 低于此值 → 软淘汰扣分
    COLLAPSE_PENALTY = 100.0

    # --- E-I 动力学参数（与 test7b 一致）---
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

    # --- 动作疲劳 ---
    FATIGUE_GAIN = 1e-5
    FATIGUE_THRESHOLD = 4
    FATIGUE_MAX = 5.0

    # --- K 倍帧率思考 ---
    FRAME_RATE = 5
    INPUT_DECAY = 0.9

    # --- GPU 并行参数 ---
    DEVICE = 'auto'
    USE_FP16 = True
    EVAL_BATCH = 0
    EVAL_MEM_FRAC = 0.55

    # --- 输出 ---
    PRINT_HISTORY_EVERY = 1

    # --- 断点 / 最优模型 ---
    CHECKPOINT_PATH = 'test10_lunar_checkpoint.pth'
    BEST_MODEL_PATH = 'test10_lunar_best_model.pth'
    LATEST_GEN_BEST_MODEL_PATH = 'test10_lunar_latest_gen_best.pth'
    HISTORY_JSON_PATH = 'test10_lunar_history.json'
    AUTO_RESUME = True
    CHECKPOINT_INTERVAL = 5


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


def _fitness(reward, success, act_fracs, cfg):
    """适应度 = 平均 reward - 坍缩软淘汰惩罚。success 仅作监控不进入排序键。"""
    f = float(reward)
    if max(act_fracs) > cfg.SINGLE_ACTION_FRAC and reward < cfg.COLLAPSE_REWARD:
        f -= cfg.COLLAPSE_PENALTY
    return f


def _auto_eval_batch(cfg, device):
    if cfg.EVAL_BATCH > 0:
        return min(cfg.EVAL_BATCH, cfg.POP_SIZE)
    if device.type != 'cuda':
        return min(cfg.POP_SIZE, 256)
    n = cfg.NUM_COLUMNS
    try:
        total = torch.cuda.get_device_properties(device).total_memory
    except Exception:
        return cfg.POP_SIZE
    per_ind = n * n * 16.0
    per_ind += n * cfg.OBS_DIM * 6.0
    batch = int(total * cfg.EVAL_MEM_FRAC / per_ind)
    return max(32, min(batch, cfg.POP_SIZE))


# ==========================================
# 1. 种群基因组张量栈（与 test7b 完全一致）
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

    def refresh_eff(self):
        self.W_in_eff = self.W_in * self.M_in
        self.W_rec_eff = self.W_rec * self.M_rec
        self.W_out_eff = self.W_out * self.M_out

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


# ==========================================
# 2. 批量 LunarLander 环境（CPU 同步步进）
# ==========================================
class BatchedLunarEnv:
    """B 个独立 LunarLander-v3 并行。终止个体冻结在其终态（跳过 step 不累计奖励）。

    obs_tensor() 返回逐维归一化 + CHANNEL_SCALES 定标后的张量（Bootstrap Threshold 层）。
    结算口径（ep_reward）：term 且成功 = 原始 reward；term 且坠毁 = 原始 + 软着陆加分；
    截断/500 步悬停 = 原始 - HOVER_PENALTY。success 一律按原始 env reward 判定。
    """

    def __init__(self, cfg, B):
        self.cfg = cfg
        self.B = B
        self.envs = [gym.make(cfg.ENV_ID) for _ in range(B)]
        self.ranges = np.asarray(cfg.OBS_RANGES, dtype=np.float32)[None, :]
        self.scales = np.asarray(cfg.CHANNEL_SCALES, dtype=np.float32)[None, :]
        self.active = np.ones(B, dtype=bool)      # 本局尚未终止的环境
        self.ep_ret = np.zeros(B, dtype=np.float64)
        self.ep_reward = np.zeros(B, dtype=np.float64)   # 结算后定格（含加分/惩罚）
        self.ep_bonus = np.zeros(B, dtype=np.float64)    # 软着陆加分（监控用）
        self.success = np.zeros(B, dtype=bool)

    def reset(self, seed=None):
        """seed=None → 每 env 独立随机种子（旧口径）；否则全体共用（CRN）。"""
        if seed is None:
            seeds = [random.randrange(2 ** 31) for _ in self.envs]
        else:
            seeds = [seed] * self.B
        self.obs_np = np.stack(
            [e.reset(seed=s)[0] for e, s in zip(self.envs, seeds)]).astype(np.float32)
        self.active[:] = True
        self.ep_ret[:] = 0.0
        self.ep_reward[:] = 0.0
        self.ep_bonus[:] = 0.0
        self.success[:] = False

    def close(self):
        for e in self.envs:
            e.close()

    def obs_tensor(self, device, dtype):
        o = (self.obs_np / self.ranges) * self.scales
        return torch.from_numpy(o.astype(np.float32)).to(device, dtype=dtype)

    def _soft_bonus(self, o):
        """坠毁局部分加分：终局物理量越接近软着陆给越多（封顶 SOFT_BONUS_CAP）。"""
        cfg = self.cfg
        if not getattr(cfg, 'SOFT_LAND_BONUS', False):
            return 0.0
        legs = float(o[6] + o[7])
        vy_t = max(0.0, 1.0 - abs(float(o[3])))
        vx_t = max(0.0, 1.0 - abs(float(o[2])))
        ang_t = max(0.0, 1.0 - abs(float(o[4])) / (math.pi / 2))
        b = (cfg.SOFT_W_LEGS * legs + cfg.SOFT_W_VY * vy_t +
             cfg.SOFT_W_VX * vx_t + cfg.SOFT_W_ANG * ang_t)
        return min(b, cfg.SOFT_BONUS_CAP)

    def _settle(self, i, term, final_obs):
        raw = self.ep_ret[i]
        self.success[i] = raw >= self.cfg.SUCCESS_REWARD
        if term:
            if self.success[i]:
                self.ep_bonus[i] = 0.0
            else:
                self.ep_bonus[i] = self._soft_bonus(final_obs)
            self.ep_reward[i] = raw + self.ep_bonus[i]
        else:
            self.ep_bonus[i] = 0.0
            self.ep_reward[i] = raw - self.cfg.HOVER_PENALTY
        self.active[i] = False

    def step(self, actions_np):
        """actions_np [B,int]；已终止个体忽略动作保持冻结。

        关键：活跃个体每步刷新 obs_np（round-1 版本遗漏了这一行，网络永远只看到
        t=0 的初始观测——智能体全程失明，只能输出恒定动作，是 -140 平台的首要根因）。
        """
        for i in range(self.B):
            if not self.active[i]:
                continue
            a = int(actions_np[i])
            prev = self.obs_np[i]          # 网络本帧看到的 obs = 撞击前状态
            o, r, term, trunc, _ = self.envs[i].step(a)
            self.ep_ret[i] += r
            if term or trunc:
                # 软着陆加分用撞击前观测：终局帧的接触冲量会瞬间归零速度、
                # 翻转腿接触位（smoke 实测 do-nothing 坠毁也能拿 111+/120 的伪影）
                self._settle(i, term=term, final_obs=prev)
            else:
                self.obs_np[i] = o

    def finish_episode(self):
        """截断时刻（500 步上限）仍未终止的个体：按悬停惩罚结算。"""
        for i in np.nonzero(self.active)[0]:
            self._settle(i, term=False, final_obs=None)


# ==========================================
# 3. 批量前向（无激素 E-I 动力学，与 test7b 一致）
# ==========================================
def forward_batch(pop, obs, E, I, st, cts, cfg):
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
    K = cfg.FRAME_RATE
    logits_sum = None
    for k in range(K):
        o = obs * (cfg.INPUT_DECAY ** k)
        logits, E, I, st = forward_batch(pop, o, E, I, st, cts, cfg)
        logits_sum = logits if logits_sum is None else logits_sum + logits
    action = torch.argmax(logits_sum, dim=1)
    return action, E, I, st


# ==========================================
# 4. 种群评估
# ==========================================
def _eval_chunk(pop, cfg, seeds=None, n_eps=None):
    """对一个子种群（B = pop.P）并行评估若干局。

    seeds: 每局一个种子（CRN），None 则每 env 独立随机。n_eps 覆盖 cfg.EVAL_EPISODES。
    返回 [P,8] = (avg_reward含加分/惩罚, success_rate, steps_mean,
                  act0..3_frac, soft_bonus_mean)
    """
    B = pop.P
    dev = pop.device
    n_ep = int(n_eps) if n_eps is not None else int(cfg.EVAL_EPISODES)
    env = BatchedLunarEnv(cfg, B)
    half = torch.float16 if getattr(cfg, 'USE_FP16', True) else torch.float32

    tot_reward = np.zeros(B, dtype=np.float64)
    tot_success = np.zeros(B, dtype=np.float64)
    tot_bonus = np.zeros(B, dtype=np.float64)
    tot_act = np.zeros((B, cfg.ACTION_DIM), dtype=np.float64)

    for ep_i in range(n_ep):
        ep_seed = seeds[ep_i] if seeds is not None else None
        env.reset(seed=ep_seed)
        E = torch.zeros(B, pop.N, dtype=half, device=dev)
        I = torch.zeros(B, pop.N, dtype=half, device=dev)
        stt = torch.zeros(B, pop.N, dtype=half, device=dev)
        cts = torch.zeros(B, pop.A, dtype=half, device=dev)
        active_prev = env.active.copy()

        for _t in range(cfg.MAX_STEPS):
            if not env.active.any():
                break
            obs = env.obs_tensor(dev, half)
            act, E, I, stt = deliberate_batch(pop, obs, E, I, stt, cts, cfg)
            cts = update_fatigue(cts, act)
            act_np = act.cpu().numpy()
            onehot = _ONEHOT_ACT[cfg.ACTION_DIM]
            tot_act[active_prev] += onehot[act_np[active_prev]]
            env.step(act_np)
            active_prev = env.active.copy()

        env.finish_episode()
        tot_reward += env.ep_reward
        tot_success += env.success.astype(np.float64)
        tot_bonus += env.ep_bonus

    n_ep_f = float(n_ep)
    avg_reward = tot_reward / n_ep_f
    success_rate = tot_success / n_ep_f
    steps_taken = tot_act.sum(axis=1) / n_ep_f               # 活跃状态下的决策步数均值
    act_frac = tot_act / np.clip(tot_act.sum(axis=1, keepdims=True), 1, None)

    metrics = np.stack((avg_reward, success_rate, steps_taken,
                        act_frac[:, 0], act_frac[:, 1], act_frac[:, 2], act_frac[:, 3],
                        tot_bonus / n_ep_f), axis=1).astype(np.float32)
    env.close()
    return metrics


def _draw_crn_seeds(cfg, n, gen, salt=0):
    """CRN：一局一个种子，按 (CRN_SEED, gen, salt) 锚定（不消耗全局 RNG）。"""
    rng = random.Random((int(getattr(cfg, 'CRN_SEED', 0)) * 1000003 + int(gen)) * 2 + int(salt))
    return [rng.randrange(2 ** 31) for _ in range(n)]


def evaluate_population_gpu(pop, cfg, gen=0):
    if getattr(cfg, 'USE_FP16', True):
        pop.fp16()
    seeds = _draw_crn_seeds(cfg, cfg.EVAL_EPISODES, gen) if cfg.USE_CRN else None
    batch = _auto_eval_batch(cfg, pop.device)
    if batch >= pop.P:
        pop.refresh_eff()
        metrics = torch.from_numpy(_eval_chunk(pop, cfg, seeds))
    else:
        metrics = torch.zeros(pop.P, 8)
        mn = metrics.numpy()
        for lo in range(0, pop.P, batch):
            sub = pop[slice(lo, min(lo + batch, pop.P))]
            sub.refresh_eff()
            mn[lo:lo + sub.P] = _eval_chunk(sub, cfg, seeds)
        metrics = torch.from_numpy(mn)

    # ---- F2 第二阶段逐半精评：[(keep, cum_eps)] 阶梯，每轮 CRN 新种子加局 ----
    # 未晋级候选保留其已达精度的累计估计；晋级者估计 = (旧均值×prev + 新均值×n_new)/cum
    ladder = getattr(cfg, 'STAGE2_LADDER', None)
    if getattr(cfg, 'STAGE2_ENABLED', False) and ladder and ladder[0][1] > cfg.EVAL_EPISODES:
        mn = metrics.numpy()
        fits = np.array([_fitness(mn[i][0], mn[i][1], mn[i][3:7], cfg)
                         for i in range(pop.P)])
        cand = np.argsort(-fits)[:min(ladder[0][0], pop.P)]
        prev_eps = 0
        for r, (keep, cum_eps) in enumerate(ladder):
            n_new = cum_eps - prev_eps
            if n_new <= 0 or len(cand) == 0:
                break
            sub = pop[cand.tolist()]
            sub.refresh_eff()
            seeds = _draw_crn_seeds(cfg, n_new, gen, salt=1 + r) if cfg.USE_CRN else None
            fresh = _eval_chunk(sub, cfg, seeds, n_eps=n_new)
            if prev_eps == 0:
                mn[cand] = fresh
            else:
                mn[cand] = (mn[cand] * prev_eps + fresh * n_new) / cum_eps
            prev_eps = cum_eps
            if r + 1 < len(ladder):
                nxt_keep = min(ladder[r + 1][0], len(cand))
                rows_fits = np.array([_fitness(mn[i][0], mn[i][1], mn[i][3:7], cfg)
                                      for i in cand])
                cand = cand[np.argsort(-rows_fits)[:nxt_keep]]
        metrics = torch.from_numpy(mn)
    return metrics


# ==========================================
# 5. 进化（GPU 向量化交叉/变异，与 test7b 一致）
# ==========================================
def sample_mut_scale(cfg, B2, dev):
    """每子代变异强度因子 s：lognormal（右偏，大变异小概率）或 normal clip。"""
    sigma = float(getattr(cfg, 'MUT_SCALE_SIGMA', 0.4))
    if getattr(cfg, 'MUT_SCALE_DIST', 'lognormal') == 'normal':
        s = 1.0 + torch.randn(B2, device=dev) * sigma
    else:
        s = torch.exp(torch.randn(B2, device=dev) * sigma)   # LogNormal(0, σ)
    return s.clamp(float(getattr(cfg, 'MUT_SCALE_MIN', 0.25)),
                   float(getattr(cfg, 'MUT_SCALE_MAX', 4.0)))


def evolve_topology_gpu(pop, metrics, cfg, gen=0):
    """进化下一代：ELITE 精英 + (P-ELITE) 后代。

    变异强度：每子代抽因子 s（类正态分布）缩放其全部变异算子（test7h #4）；
    精英不变异。
    """
    P = pop.P
    N = pop.N
    dev = pop.device
    active = _freeze_active_groups(gen, cfg)
    has_g1 = 'G1' in active
    has_g2 = 'G2' in active

    # --- 精英排序（按适应度）---
    mn = metrics.cpu().numpy()
    fit = np.array([_fitness(mn[i][0], mn[i][1], mn[i][3:7], cfg) for i in range(P)])
    order = np.argsort(-fit)
    elite_idx = order[:cfg.ELITE_SIZE]
    elites = pop[elite_idx.tolist()]

    new_pop = pop.empty()
    children = pop.empty(B=P - cfg.ELITE_SIZE)
    for g in GeneStack.GENES:
        setattr(children, g, torch.empty(P - cfg.ELITE_SIZE, *getattr(elites, g).shape[1:],
                                         dtype=getattr(elites, g).dtype, device=dev))

    E = cfg.ELITE_SIZE
    B2 = P - E
    p1_idx = torch.randint(0, E, (B2,), device=dev)
    p2_idx = torch.randint(0, E, (B2,), device=dev)
    p2_idx = torch.where(p2_idx == p1_idx, (p1_idx + 1) % E, p2_idx)

    p1 = elites[p1_idx]
    p2 = elites[p2_idx]

    # --- 每子代变异强度因子（lognormal 中位数 1 = 原强度；右偏大变异）---
    s = sample_mut_scale(cfg, B2, dev)
    topo_mut_prob_i = (cfg.TOPOLOGY_MUT_PROB * s).clamp(max=0.5)          # [B2]
    mask_mut_rate_i = (cfg.MUT_RATE * s).clamp(max=0.5)                   # [B2]
    weight_frac1 = (cfg.WEIGHT_MUT_FRAC * s).clamp(max=1.0)               # [B2]
    weight_std1 = (cfg.WEIGHT_MUT_STD * s)                                # [B2]
    tau_std2 = (cfg.TAU_E_MUT_STD * s).view(B2, 1)
    wei_std2 = (cfg.W_EI_MUT_STD * s).view(B2, 1)
    wie_std2 = (cfg.W_IE_MUT_STD * s).view(B2, 1)

    with torch.no_grad():
        # ---- G1 交叉：结构组 ----
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

        # ---- G2 交叉：动力学组 ----
        if has_g2:
            col2 = torch.rand(B2, N, device=dev) > 0.5
            children.tau_e = torch.where(col2, p1.tau_e, p2.tau_e)
            children.w_ei = torch.where(col2, p1.w_ei, p2.w_ei)
            children.w_ie = torch.where(col2, p1.w_ie, p2.w_ie)

        # ---- 变异（逐组独立，冻结组跳过；强度逐子代 s 缩放，test7h #4）----
        if has_g1:
            pick = torch.randint(0, 3, (B2,), device=dev)
            topo_gate = torch.rand(B2, device=dev) < topo_mut_prob_i
            for ai, attr in enumerate(GeneStack.G1_MASKS):
                sel = (pick == ai) & topo_gate
                if bool(sel.any().item()):
                    t = getattr(children, attr)
                    rate3 = mask_mut_rate_i.view(B2, 1, 1)
                    flip = torch.rand_like(t) < rate3
                    setattr(children, attr,
                            torch.where(sel[:, None, None] & flip, 1.0 - t, t))
            for attr in GeneStack.G1_WEIGHTS:
                t = getattr(children, attr)
                # 按基因张量维度自适应广播 [B2,1,1]/[B2,1]
                v = lambda x: x.view(B2, *([1] * (t.dim() - 1))).to(t.dtype)
                noise = torch.randn_like(t) * v(weight_std1)
                m = (torch.rand_like(t) < v(weight_frac1)).to(t.dtype)
                setattr(children, attr, t + noise * m)

        if has_g2:
            children.tau_e = torch.clamp(children.tau_e + torch.randn_like(children.tau_e)
                                         * tau_std2.to(children.tau_e.dtype),
                                         cfg.TAU_E_MIN, cfg.TAU_E_MAX)
            children.w_ei = torch.clamp(children.w_ei + torch.randn_like(children.w_ei)
                                        * wei_std2.to(children.w_ei.dtype),
                                        cfg.W_EI_MIN, cfg.W_EI_MAX)
            children.w_ie = torch.clamp(children.w_ie + torch.randn_like(children.w_ie)
                                        * wie_std2.to(children.w_ie.dtype),
                                        cfg.W_IE_MIN, cfg.W_IE_MAX)

    for g in GeneStack.GENES:
        setattr(new_pop, g, torch.cat([getattr(elites, g).clone(), getattr(children, g)], dim=0))
    new_pop.dtype = pop.dtype
    return new_pop


# ==========================================
# 6. 保存 / 加载
# ==========================================
def load_best_state(path, cfg):
    if not os.path.exists(path):
        return None
    try:
        data = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"  警告: 模型 {path} 读取失败 ({e})")
        return None
    saved_cfg = data.get('config', {})
    if saved_cfg:
        if (saved_cfg.get('NUM_COLUMNS') != cfg.NUM_COLUMNS or
                saved_cfg.get('OBS_DIM') != cfg.OBS_DIM or
                saved_cfg.get('ACTION_DIM') != cfg.ACTION_DIM or
                saved_cfg.get('CHANNEL_SCALES') != list(cfg.CHANNEL_SCALES)):
            print(f"  警告: 模型 {path} 与当前配置不匹配，已忽略")
            return None
    st = data.get('brain')
    if st is None:
        return None
    return st, float(data.get('reward', -1e9)), float(data.get('success_rate', 0.0))


def save_best_model(path, st, cfg, reward, success_rate):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save({
        'brain': st,
        'reward': float(reward),
        'success_rate': float(success_rate),
        'config': _cfg_dict(cfg),
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }, path)


def save_checkpoint10(path, cfg, next_gen, pop, history,
                      cum_eval_time, cum_evolve_time,
                      best_state, best_reward, best_success):
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
        'best_reward': float(best_reward),
        'best_success': float(best_success),
        'config': _cfg_dict(cfg),
        'random_state': random.getstate(),
        'torch_rng_state': torch.get_rng_state(),
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    tmp = path + '.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, path)
    print(f"  [Checkpoint] 断点已保存 -> {path} (next_gen={next_gen})")


def load_checkpoint10(path, cfg):
    if not os.path.exists(path):
        return None
    try:
        data = torch.load(path, map_location='cpu', weights_only=False)
    except Exception:
        return None
    saved_cfg = data.get('config', {})
    if saved_cfg and (saved_cfg.get('NUM_COLUMNS') != cfg.NUM_COLUMNS or
                      saved_cfg.get('OBS_DIM') != cfg.OBS_DIM or
                      saved_cfg.get('ACTION_DIM') != cfg.ACTION_DIM or
                      saved_cfg.get('CHANNEL_SCALES') != list(cfg.CHANNEL_SCALES)):
        print(f"  警告: 断点 {path} 与当前配置不匹配，已忽略")
        return None
    if saved_cfg and saved_cfg.get('FITNESS_VERSION', 1) != cfg.FITNESS_VERSION:
        # 结构兼容但适应度口径升级：保留种群，重置历史/最优追踪（数值不可比）
        data['reset_stats'] = True
    return data


def save_history_json(path, history):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False)


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
    history = {'gen': [], 'best_reward': [], 'avg_reward': [], 'best_success': [], 'avg_success': []}
    cum_eval_time = 0.0
    cum_evolve_time = 0.0
    best_state = None
    best_reward = -1e18
    best_success = 0.0
    latest_gen_best_state = None
    latest_gen_best_reward = -1e18
    latest_gen_best_success = 0.0

    if cfg.AUTO_RESUME:
        ck = load_checkpoint10(cfg.CHECKPOINT_PATH, cfg)
        if ck is not None:
            reset_stats = bool(ck.pop('reset_stats', False))
            pop.unpack(ck['pop'])
            if reset_stats:
                # 种群保留（技能真实存在），但适应度口径已升级：历史/最优从零计
                print(f"=== 检测到断点 [{cfg.CHECKPOINT_PATH}]（适应度口径 v{cfg.FITNESS_VERSION} 升级）===")
                print("  保留种群基因，重置历史曲线与最优追踪，从第 0 代重新计数")
            else:
                start_gen = int(ck['next_gen'])
                history = ck['history']
                cum_eval_time = float(ck.get('cum_eval_time', 0.0))
                cum_evolve_time = float(ck.get('cum_evolve_time', 0.0))
                best_state = ck.get('best_state')
                best_reward = float(ck.get('best_reward', -1e18))
                best_success = float(ck.get('best_success', 0.0))
                if best_state is not None:
                    random.setstate(ck['random_state'])
                    torch.set_rng_state(ck['torch_rng_state'])
                print(f"=== 检测到断点 [{cfg.CHECKPOINT_PATH}] ===")
                print(f"  已完成 {start_gen} 代 -> 从第 {start_gen} 代接续 | "
                      f"历史最优: Reward={best_reward:.1f}")

    if pop.M_in is None:
        t0 = time.perf_counter()
        print("初始化种群（GPU 随机初始化）...")
        pop.random_init()
        print(f"  初始化完成 ({time.perf_counter() - t0:.1f}s)")
        save_checkpoint10(cfg.CHECKPOINT_PATH, cfg, 0, pop, history,
                          cum_eval_time, cum_evolve_time,
                          best_state, best_reward, best_success)

    try:
        for gen in range(start_gen, cfg.GENERATIONS):
            t_eval = time.perf_counter()
            metrics = evaluate_population_gpu(pop, cfg, gen=gen)
            eval_time = time.perf_counter() - t_eval
            cum_eval_time += eval_time

            mn = metrics.numpy()
            fits = np.array([_fitness(mn[i][0], mn[i][1], mn[i][3:7], cfg) for i in range(cfg.POP_SIZE)])
            best_idx = int(np.argmax(fits))
            b_reward, b_success = float(mn[best_idx][0]), float(mn[best_idx][1])
            avg_reward = float(np.mean(mn[:, 0]))
            avg_success = float(np.mean(mn[:, 1]))

            elite_idx = np.argsort(-fits)[:cfg.ELITE_SIZE]
            elite_avg_reward = float(np.mean(mn[elite_idx, 0]))

            history['gen'].append(gen)
            history['best_reward'].append(b_reward)
            history['avg_reward'].append(avg_reward)
            history['best_success'].append(b_success)
            history['avg_success'].append(avg_success)
            save_history_json(cfg.HISTORY_JSON_PATH, history)   # 逐代落盘（test7h #6）

            if b_reward > best_reward:
                best_reward = b_reward
                best_success = b_success
                best_state = pop.individual_state(best_idx, use_half=False)

            latest_gen_best_state = pop.individual_state(best_idx, use_half=False)
            latest_gen_best_reward = b_reward
            latest_gen_best_success = b_success

            if gen < cfg.GENERATIONS - 1:
                t_ev = time.perf_counter()
                pop = evolve_topology_gpu(pop, metrics, cfg, gen=gen)
                evolve_time = time.perf_counter() - t_ev
                cum_evolve_time += evolve_time
            else:
                evolve_time = 0.0

            if (gen + 1) % cfg.PRINT_HISTORY_EVERY == 0:
                b_acts = "/".join(f"{mn[best_idx][3 + k]:.2f}" for k in range(cfg.ACTION_DIM))
                print(f"Gen {gen + 1}/{cfg.GENERATIONS} | "
                      f"BestReward: {b_reward:.1f} | BestSuccess: {b_success:.2f} | "
                      f"AvgReward: {avg_reward:.1f} | AvgSuccess: {avg_success:.2f} | "
                      f"EliteReward: {elite_avg_reward:.1f} | "
                      f"BestSteps: {mn[best_idx][2]:.0f} | BestBonus: {mn[best_idx][7]:.1f} | "
                      f"BestActs[{b_acts}] | "
                      f"eval {eval_time:.1f}s / evolve {evolve_time:.1f}s")

            if (gen + 1) % cfg.CHECKPOINT_INTERVAL == 0:
                save_checkpoint10(cfg.CHECKPOINT_PATH, cfg, gen + 1, pop, history,
                                  cum_eval_time, cum_evolve_time,
                                  best_state, best_reward, best_success)

    except KeyboardInterrupt:
        nxt = gen if 'gen' in dir() else start_gen
        print("\n训练被中断 (Ctrl+C)，正在保存断点...")
        save_checkpoint10(cfg.CHECKPOINT_PATH, cfg, nxt, pop, history,
                          cum_eval_time, cum_evolve_time,
                          best_state, best_reward, best_success)
        save_history_json(cfg.HISTORY_JSON_PATH, history)
        print(f"断点已保存: {cfg.CHECKPOINT_PATH} (下次从第 {nxt} 代接续)")
        sys.exit(0)

    t_delta = time.perf_counter() - t_program
    print(f"\nTotal runtime: {t_delta:.1f}s "
          f"(eval {cum_eval_time:.1f}s / evolve {cum_evolve_time:.1f}s)")

    if best_state is None:
        best_state = pop.individual_state(0, use_half=False)
    save_best_model(cfg.BEST_MODEL_PATH, best_state, cfg, best_reward, best_success)
    print(f"\n最优模型已保存: {cfg.BEST_MODEL_PATH} "
          f"(Reward={best_reward:.1f}, Success={best_success:.2f})")

    if latest_gen_best_state is not None:
        save_best_model(cfg.LATEST_GEN_BEST_MODEL_PATH, latest_gen_best_state, cfg,
                        latest_gen_best_reward, latest_gen_best_success)
        print(f"最新一代最优模型已保存: {cfg.LATEST_GEN_BEST_MODEL_PATH} "
              f"(Reward={latest_gen_best_reward:.1f})")

    # 断点保留（test7h #6：完成后删除断点导致跨 run 只能种子注入=准重启）
    save_checkpoint10(cfg.CHECKPOINT_PATH, cfg, cfg.GENERATIONS, pop, history,
                      cum_eval_time, cum_evolve_time,
                      best_state, best_reward, best_success)
    save_history_json(cfg.HISTORY_JSON_PATH, history)
    print(f"训练完成，断点已保留: {cfg.CHECKPOINT_PATH} "
          f"(next_gen={cfg.GENERATIONS}，续训需提高 --gens)")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(history['gen'], history['best_reward'], label='Best Reward',
                 color='red', marker='o', markersize=3)
        ax1.plot(history['gen'], history['avg_reward'], label='Avg Reward',
                 color='blue', alpha=0.6)
        ax1.axhline(cfg.SUCCESS_REWARD, color='green', ls='--', alpha=0.5, label='Pass (200)')
        ax1.set_title("Evolution Progress — Episode Reward")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Reward")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.plot(history['gen'], history['best_success'], label='Best Success Rate',
                 color='green', marker='s', markersize=3)
        ax2.plot(history['gen'], history['avg_success'], label='Avg Success Rate',
                 color='purple', alpha=0.6)
        ax2.set_title("Success Rate (reward >= 200)")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Rate")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        hist_path = cfg.CHECKPOINT_PATH.replace('_checkpoint', '_history').replace('.pth', '.png')
        fig.savefig(hist_path, dpi=100)
        plt.close(fig)
        print(f"历史曲线已保存: {hist_path}")
    except Exception as e:
        print(f"(matplotlib 曲线跳过: {e})")


def play_best(cfg, seed=None):
    """加载最优模型渲染一局完整着陆（gym human 窗口）。"""
    res = load_best_state(cfg.BEST_MODEL_PATH, cfg)
    if res is None:
        print("无最优模型可播放")
        return
    st, reward, success = res
    dev = _resolve_device(cfg)
    pop = GeneStack(cfg, B=1, device=dev)
    pop.random_init()
    pop.set_individual_from_state(0, st)
    pop.refresh_eff()
    if getattr(cfg, 'USE_FP16', True):
        pop.fp16()
        pop.refresh_eff()

    env = gym.make(cfg.ENV_ID, render_mode='human')
    obs_np, _ = env.reset(seed=seed)
    ranges = np.asarray(cfg.OBS_RANGES, dtype=np.float32)[None, :]
    scales = np.asarray(cfg.CHANNEL_SCALES, dtype=np.float32)[None, :]

    E = torch.zeros(1, pop.N, dtype=pop.dtype, device=dev)
    I = torch.zeros(1, pop.N, dtype=pop.dtype, device=dev)
    stt = torch.zeros(1, pop.N, dtype=pop.dtype, device=dev)
    cts = torch.zeros(1, pop.A, dtype=pop.dtype, device=dev)

    ep_ret = 0.0
    names = ['无动作', '左引擎', '主引擎', '右引擎']
    for s in range(cfg.MAX_STEPS):
        o = ((obs_np[None, :].astype(np.float32) / ranges) * scales).astype(np.float32)
        obs = torch.from_numpy(o).to(dev, dtype=pop.dtype)
        act, E, I, stt = deliberate_batch(pop, obs, E, I, stt, cts, cfg)
        cts = update_fatigue(cts, act)
        a = int(act.item())
        obs_np, r, term, trunc, _ = env.step(a)
        ep_ret += r
        if s % 50 == 0:
            print(f"step {s}: act={names[a]}, reward={r:+.2f}, 总计={ep_ret:.1f}")
        if term or trunc:
            break
    env.close()
    ok = "成功着陆" if ep_ret >= cfg.SUCCESS_REWARD else "未及格"
    print(f"\nPlay done: Reward={ep_ret:.1f} ({ok}) @ step {s + 1}")
    return ep_ret


# ==========================================
# 8. 入口
# ==========================================
def make_smoke_config():
    cfg = Config()
    cfg.POP_SIZE = 32
    cfg.NUM_COLUMNS = 32
    cfg.GENERATIONS = 3
    cfg.ELITE_SIZE = 8
    cfg.EVAL_EPISODES = 1
    cfg.MAX_STEPS = 120
    cfg.FRAME_RATE = 2
    cfg.STAGE2_LADDER = [(8, 2), (4, 3)]   # 覆盖逐半精评多轮路径
    cfg.CHECKPOINT_INTERVAL = 1
    cfg.CHECKPOINT_PATH = 'test10_lunar_smoke_checkpoint.pth'
    cfg.BEST_MODEL_PATH = 'test10_lunar_smoke_best.pth'
    cfg.LATEST_GEN_BEST_MODEL_PATH = 'test10_lunar_smoke_latest_best.pth'
    cfg.HISTORY_JSON_PATH = 'test10_lunar_smoke_history.json'
    cfg.AUTO_RESUME = False
    cfg.EVAL_BATCH = 16
    cfg.PRINT_HISTORY_EVERY = 1
    return cfg


def main():
    ap = argparse.ArgumentParser(
        description='test10 — LunarLander-v3 EI-RNN（定标 + CRN + 两阶段评估 + 软着陆梯度）')
    ap.add_argument('--smoke', action='store_true', help='小规模快速自检')
    ap.add_argument('--gens', type=int, default=None)
    ap.add_argument('--pop', type=int, default=None)
    ap.add_argument('--columns', type=int, default=None)
    ap.add_argument('--episodes', type=int, default=None,
                    help='K1 粗评局数（默认 3）')
    ap.add_argument('--max-steps', type=int, default=None)
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--no-crn', action='store_true', help='关闭公共随机数（A/B 对照）')
    ap.add_argument('--no-stage2', action='store_true', help='关闭两阶段精评（A/B 对照）')
    ap.add_argument('--stage2-candidates', type=int, default=None,
                    help='逐半精评首轮候选数（默认 128；后续轮按 48/128、16/128 比例缩放）')
    ap.add_argument('--stage2-episodes', type=int, default=None,
                    help='逐半精评末轮累计局数（默认 16；阶梯局数取 E/4、E/2、E）')
    ap.add_argument('--mut-sigma', type=float, default=None,
                    help='类正态变异强度 σ（默认 0.4，中位数 1=原强度）')
    ap.add_argument('--mut-dist', type=str, default=None, choices=['lognormal', 'normal'],
                    help='变异强度分布（默认 lognormal）')
    ap.add_argument('--hover-penalty', type=float, default=None,
                    help='500 步截断悬停惩罚（默认 100；0=关闭）')
    ap.add_argument('--no-soft-bonus', action='store_true',
                    help='关闭软着陆部分加分（A/B 对照）')
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
        cfg.ELITE_SIZE = max(8, cfg.POP_SIZE // 8)
    if args.columns:
        cfg.NUM_COLUMNS = args.columns
    if args.episodes:
        cfg.EVAL_EPISODES = args.episodes
    if args.max_steps:
        cfg.MAX_STEPS = args.max_steps
    if args.device:
        cfg.DEVICE = args.device
    if args.no_crn:
        cfg.USE_CRN = False
    if args.no_stage2:
        cfg.STAGE2_ENABLED = False
    if args.stage2_candidates or args.stage2_episodes:
        c0 = args.stage2_candidates or cfg.STAGE2_LADDER[0][0]
        e0 = args.stage2_episodes or cfg.STAGE2_LADDER[-1][1]
        cfg.STAGE2_LADDER = [(max(1, c0), max(cfg.EVAL_EPISODES + 1, e0 // 4)),
                              (max(1, c0 * 48 // 128), max(cfg.EVAL_EPISODES + 1, e0 // 2)),
                              (max(1, c0 * 16 // 128), e0)]
    if args.mut_sigma is not None:
        cfg.MUT_SCALE_SIGMA = args.mut_sigma
    if args.mut_dist:
        cfg.MUT_SCALE_DIST = args.mut_dist
    if args.hover_penalty is not None:
        cfg.HOVER_PENALTY = args.hover_penalty
    if args.no_soft_bonus:
        cfg.SOFT_LAND_BONUS = False

    if args.play:
        play_best(cfg)
        return

    run_training(cfg)


if __name__ == '__main__':
    main()
