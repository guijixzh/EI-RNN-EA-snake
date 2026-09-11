# ==========================================
# test16c_cheat7b.py —— 固定种子 × 固定单地图 × 7b best 全种群克隆：通关特训
#
# 目标（"cheat"模式）：把 7b 最强模型（test7b_latest_gen_best.pth，saved food=67.0，
#   千局基准 61.35）无损迁移到 16c 稀疏固定扇入基因组，在【同一张固定地图】
#   （固定食物序列 + 固定初始朝向，CRN 确定性生成）上用 16c 优化模式
#   （fast-eval + 精英保留 GA + 类正态变异强度 + simple 适应度）持续精修，
#   直到通关（10×10 盘满 = 98 食）自动停机并保存 win 模型。
#
# 与 test16c 的差异：
#   1) N=256 / OBS_DIM=32（7b 血统，obs='32proj' 8扇区欧氏食物投影语义）；
#      REC_FANIN=96 —— 实测 7b latest_gen_best 每行非零 max=65 → K=96 完全无损。
#   2) 稠密→稀疏转换器 load_best_state_7b_sparse：M_rec/W_rec 每行按 |W| top-K
#      构造 rec_idx/rec_w（预期 0 行超限 0 丢边），全种群克隆注入。
#   3) 前向动力学逐字 7b 口径：cts 按动作计数疲劳（FATIGUE_GAIN=1e-5/THRESH=4/
#      MAX=5）、SHORT_TERM_GAIN=-0.2、STARVE_SLOPE=3.0、FRAME_RATE=5/INPUT_DECAY=0.9。
#   4) 固定单地图：make_bank(MAP_SEED, gen=MAP_GEN, stage=1, ep=0) 预生成一次全程
#      复用；确定性环境 + 确定性网络 → 每个体每代评 1 局即无噪声（K=1），
#      两阶段淘汰/halving/自适应K2/LCB 全部不需要（精简移除）。
#   5) 通关判定：env.step 增加 won = ate & (body_len >= G*G) 终局（16c 无满盘
#      保护，盘满会退化落子）；STOP_ON_WIN 达标即停并另存 *_win_model.pth。
#   6) 池约束（传感池/运动池）默认关闭（SENSORY_FRAC=MOTOR_FRAC=0，用户决策：
#      7b 解剖 100% 保留）；机制保留，--sensory-frac/--motor-frac 可随时开启。
#   7) 随机种子固定：--seed 默认 20260908（CPU+CUDA），地图身份由 --map-seed
#      （默认 20260908）决定 —— 相同命令 = 逐位可复现。
#
# 16c 保留的优化模式：fast-eval（向量化观测+遥测降频+无每步同步）、精英保留
#   GA（ELITE=1/4 亲本 + 双亲交叉 + 按代轮换 G1/G2 冻结）、LogNormal 变异强度、
#   rec 槽位重连/M_in/M_out 翻转拓扑变异、checkpoint 原子落盘 + AUTO_RESUME、
#   CRN 公共随机数（本脚本固定 gen 即固定地图）、history/PNG/AutoDL 通知。
#
# 用法：
#   python test16c_cheat7b.py --selfcheck        # 全部自检（含 7b 兼容性三项）
#   python test16c_cheat7b.py --smoke            # 冒烟（pop16/3代，不依赖种子文件）
#   python test16c_cheat7b.py --pop 256 --gens 3 # 短跑验收（首代 best 应 ≈67 食）
#   python test16c_cheat7b.py --gens 500         # 正式通关长跑（默认全种群克隆）
#   python test16c_cheat7b.py --play             # 固定地图回放最优模型
# ==========================================

import argparse
import json
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
    # --- 进化参数（16c 口径）---
    POP_SIZE = 4096
    GENERATIONS = 500
    ELITE_SIZE = 1024           # 亲本 = 1/4（16c 口径）
    MUT_RATE = 0.05
    TOPOLOGY_MUT_PROB = 0.05
    WEIGHT_MUT_FRAC = 0.2
    WEIGHT_MUT_STD = 0.1
    TAU_E_MUT_STD = 0.05
    EVO_COS_MODE = 'anneal'
    EVO_COS_PERIOD = 100
    EVO_DYN_DECAY_TAU = 33

    # --- 类正态变异强度（每子代因子 s，缩放其全部变异算子）---
    MUT_SCALE_DIST = 'lognormal'
    MUT_SCALE_SIGMA = 0.5
    MUT_SCALE_MIN = 0.25
    MUT_SCALE_MAX = 4.0

    # --- 无激素 EI-RNN（兼容字段）---
    TRAIN_HORMONE_NET = False
    HORMONE_NET_HIDDEN = 32

    # --- 环境参数 ---
    GRID_SIZE = 10
    MAX_STEPS = 100000
    EP_STEPS_CAP = 3000         # 个体级步数硬上限（0=关）：活满即冻结。磨蹭个体
                                # （吃得少但永不饿死）会把整代评估拖到 MAX_STEPS，
                                # 98食赢家只需 ~900-1800 步，3000 仍有 1.7-3 倍余量
    EVAL_EPISODES = 1           # 固定地图确定性评估：每个体 1 局（无噪声）

    # --- 脑结构参数（7b 兼容规模 + 16 稀疏固定扇入）---
    NUM_COLUMNS = 256           # 7b 血统柱数
    REC_FANIN = 96              # 每柱循环输入槽位 K（实测 7b best 行非零 max=65 → 无损）
    OBS_MODE = '32proj'
    OBS_DIM = 32                # 7b 的 32 维投影观测（8 扇区欧氏食物投影）
    ACTION_DIM = 3
    INIT_DENSITY = 0.15         # 仅用于 M_in/M_out（rec 由 REC_FANIN 决定）

    # --- test16c 池约束（本 run 默认关闭：7b 解剖 100% 保留）---
    SENSORY_FRAC = 0.0          # 0=关（回退无约束）；--sensory-frac 可开启
    MOTOR_FRAC = 0.0
    POOL_SEED = 20260907

    # --- 适应度（simple = 7b 口径；固定地图无噪声，LCB 关闭）---
    FIT_MODE = 'simple'
    SIMPLE_EFF_W = 0.3
    FITNESS_VERSION = 20        # v20 = v19 + 稳健化评估口径（robust min）；
                                # 导入 v19 断点时 best 追踪自动重置（预期行为）
    ONE_SIDED_TURN_DEATH = True

    # --- 观测编码版本守卫（checkpoint 校验）---
    OBS_ENC_VERSION = '32proj7b'   # 7b 血统 32 维（勿与 16 系 40 维混淆）
    BRAIN_VERSION = 'sparse1'      # 稀疏固定扇入基因组（与稠密互斥）

    # --- 饿死斜率 / 观测幅度（7b 口径）---
    STARVE_SLOPE = 3.0
    OBS_FOOD_SCALE = 8.0
    OBS_SELF_SCALE = 8.0
    OBS_OBSTACLE_SCALE = 8.0

    # --- E-I 动力学参数（7b 口径）---
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

    # --- 短期 tau 调制（7b 口径）---
    SHORT_TERM_GAIN = -0.2
    SHORT_TERM_DECAY = 0.3

    # --- 动作疲劳（7b 口径：按动作计数 cts，逐字迁移）---
    FATIGUE_GAIN = 1e-5
    FATIGUE_THRESHOLD = 4
    FATIGUE_MAX = 5.0

    # --- K 倍帧率思考 ---
    FRAME_RATE = 5
    INPUT_DECAY = 0.9

    # --- 进化筛选策略 ---
    LONG_SNAKE_SCORE_THRESHOLD = 3.0
    CYCLE_PATTERN = [('G2', 'G1')]     # 逐代轮换冻结组（16c 口径）

    # --- 固定单地图（cheat 核心）---
    USE_CRN = True
    MAP_SEED = 20260908        # 地图身份种子（--map-seed；换图=换种子）
    MAP_GEN = 0                # 地图在 CRN 派生式里的 gen 槽位（固定 → 地图固定）
    CRN_DRAW = 4096            # 每局落子候选流长度（98 食绰绰有余）
    TARGET_FOOD = 98           # 通关 = 盘满 = G*G - 初始体长 2
    STOP_ON_WIN = True         # 达标即停（--no-stop-on-win 关闭）
    WIN_FRAC = None            # 通关占比停机线：None=任一个体通关即停；如 0.95=95% 个体通关才停

    # --- GPU 并行参数 ---
    DEVICE = 'auto'
    USE_FP16 = True
    EVAL_BATCH = 0
    EVAL_MEM_FRAC = 0.4         # 显存上界（0.85 会顶满 8.5GB 触发 Windows 共享内存
                                # 溢出，特定代分配序列下每步成本暴涨 9 倍；0.4 保峰值 <40%）
    FAST_EVAL = True

    # --- 稳健化评估（关键：训练后网络为强混沌系统，单局分数对数值实现敏感
    #     ——同一基因在 4096 批评 85、B=1 批评 22。副本含权重微扰，fitness 取
    #     最差副本 = 逼 GA 选"权重邻域稳健"的解；这样练出的通关模型不依赖
    #     特定 batch/kernel 的数值环境，单独回放同样成立）---
    ROBUST_EVAL = 2            # 评估副本数（1=关；2=原权重+1份微扰副本取 min）
    ROBUST_NOISE_STD = 1e-3    # 微扰副本的权重噪声 std（W_in/W_out/rec_w/b_out/tau_e）
    EVAL_SOLO = False          # B=1 逐行真值口径与批量口径在 CRN 消耗解耦修复后
                               # 已等价（2600/512/128/32/1 行分数一致）；False=批量
                               # （快 ~8 倍）。True 保留为逐行验证模式
    EVAL_WORKERS = 8           # SOLO 口径的并行 worker 子进程数（1=主进程内串行）

    # --- 输出 ---
    PRINT_HISTORY_EVERY = 1

    # --- 断点 / 最优模型 / 种子 ---
    CHECKPOINT_PATH = '16c_cheat7b_checkpoint.pth'
    BEST_MODEL_PATH = '16c_cheat7b_best_model.pth'
    LATEST_GEN_BEST_MODEL_PATH = '16c_cheat7b_latest_gen_best.pth'
    WIN_MODEL_PATH = '16c_cheat7b_win_model.pth'
    HISTORY_JSON_PATH = '16c_cheat7b_history.json'
    AUTO_RESUME = True
    CHECKPOINT_INTERVAL = 10
    SEED_FROM_BEST = True
    SEED_MODEL_PATH = 'artifacts/test7b/test7b_latest_gen_best.pth'   # 7b 最强（saved 67.0 / 千局 61.35）
    SEED_POP = True            # 全种群克隆注入（cheat 精修模式）
    SEED = 20260908            # 全局随机种子（固定）


# ==========================================
# 0b. 基础工具
# ==========================================
def _pool_masks(cfg, N, device):
    """传感池/运动池成员掩码 [N] bool。SENSORY_FRAC/MOTOR_FRAC 任一为 0 时返回
    全 True（=无约束）。本 run 默认全关；保留机制便于 --sensory-frac 开启。"""
    gen = torch.Generator().manual_seed(int(getattr(cfg, 'POOL_SEED', 20260907)))
    g_in = float(getattr(cfg, 'SENSORY_FRAC', 0.0))
    g_out = float(getattr(cfg, 'MOTOR_FRAC', 0.0))
    sin_ = (torch.rand(N, generator=gen) < g_in) if g_in > 0 else torch.ones(N, dtype=torch.bool)
    mot = (torch.rand(N, generator=gen) < g_out) if g_out > 0 else torch.ones(N, dtype=torch.bool)
    return sin_.to(device), mot.to(device)


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


def _fitness_simple(m, cfg):
    """simple 口径（7b/test16c 同式）：fitness = food + k·food/steps_last。
    单侧转弯判死 → -1e9。固定地图下 food 单调承载通关（98=满盘）。"""
    if m[1] >= 99999:
        return -1e9
    if getattr(cfg, 'ONE_SIDED_TURN_DEATH', False) and len(m) > 9:
        a1, a2 = float(m[8]), float(m[9])
        if (a1 > 1.0 and a2 == 0.0) or (a2 > 1.0 and a1 == 0.0):
            return -1e9
    food, steps_last = m[0], m[3]
    if food <= 0:
        return 0.0
    return food + float(getattr(cfg, 'SIMPLE_EFF_W', 0.3)) * food / max(steps_last, 1.0)


def _make_key_fn(cfg, K=None):
    """排序键。固定地图确定性评估无噪声 → 恒用 base（LCB 不适用，K 仅签名兼容）。"""
    return lambda m: _fitness_simple(m, cfg)


def _auto_eval_batch(cfg, device):
    """单次扫描允许的最大副本数（个体×局复制后的批维大小；局维=1 即个体数）。"""
    if cfg.EVAL_BATCH > 0:
        return max(cfg.EVAL_BATCH, 32)
    if device.type != 'cuda':
        return 1 << 20
    n = cfg.NUM_COLUMNS
    k = int(getattr(cfg, 'REC_FANIN', 16))
    try:
        total = torch.cuda.get_device_properties(device).total_memory
    except Exception:
        return 1 << 20
    per_ind = n * k * 24.0
    per_ind += n * cfg.OBS_DIM * 8.0
    per_ind += n * 48.0
    batch = int(total * cfg.EVAL_MEM_FRAC / per_ind)
    return max(32, batch)


def _crn_seed(cfg, gen, stage, ep):
    """每代/阶段/局独立且跨进程确定的种子（python hash 有进程盐，不可用）。"""
    return (int(cfg.MAP_SEED) * 1000003 + int(gen) * 1009
            + int(stage) * 101 + int(ep)) % (2 ** 63 - 1)


def make_bank(cfg, gen, stage, ep, device):
    """生成一局的公共落子流 + 公共初始朝向（全体个体共用）。固定 gen/stage/ep
    即固定地图——本脚本全程只用 (MAP_GEN, 1, 0) 这一张。"""
    g = torch.Generator()
    g.manual_seed(_crn_seed(cfg, gen, stage, ep))
    stream = torch.randint(0, cfg.GRID_SIZE, (cfg.CRN_DRAW, 2), generator=g).to(device)
    dir0 = int(torch.randint(0, 4, (1,), generator=g).item())
    return {'stream': stream, 'dir0': dir0}


# ==========================================
# 1. 种群基因组张量栈（16 稀疏固定扇入，与 test16c 同构）
# ==========================================
class GeneStack:
    """整个种群的基因型/表现型堆叠张量。

    形状约定（B = 个体数, N = 柱数, O = 观测维, A = 动作维, K = REC_FANIN）：
      M_in [B,N,O]   M_out [B,A,N]
      W_in [B,N,O]   W_out [B,A,N]
      rec_idx [B,N,K] int64 —— 循环连接源 id（自连禁止；重复源=权重叠加）
      rec_w   [B,N,K] float
      b_out [B,A]     tau_e [B,N]     w_ei / w_ie [B,N]
    """

    GENES = ['M_in', 'M_out', 'W_in', 'W_out', 'b_out',
             'rec_idx', 'rec_w',
             'tau_e', 'w_ei', 'w_ie']
    G1_WEIGHTS = ['W_in', 'rec_w', 'W_out', 'b_out']
    G1_MASKS = ['M_in', 'M_out', 'rec_idx']   # 拓扑变异三分支；rec_idx 分支=槽位重连
    G2_TENSORS = ['tau_e', 'w_ei', 'w_ie']
    EFF = ['W_in_eff', 'W_out_eff']

    def __init__(self, cfg, B=None, device=None):
        self.cfg = cfg
        self.N = cfg.NUM_COLUMNS
        self.O = cfg.OBS_DIM
        self.A = cfg.ACTION_DIM
        self.K = int(getattr(cfg, 'REC_FANIN', 16))
        self.P = B if B is not None else cfg.POP_SIZE
        self.device = device if device is not None else _resolve_device(cfg)
        self.dtype = torch.float32
        for g in self.GENES:
            setattr(self, g, None)
        for e in self.EFF:
            setattr(self, e, None)

    def random_init(self):
        cfg = self.cfg
        N, O, A, B, K = self.N, self.O, self.A, self.P, self.K
        dev = self.device
        with torch.no_grad():
            self.M_in = (torch.rand(B, N, O, device=dev) < cfg.INIT_DENSITY).float()
            self.M_out = (torch.rand(B, A, N, device=dev) < cfg.INIT_DENSITY).float()
            smask, mmask = _pool_masks(cfg, N, dev)      # 池约束（默认全 True=关）
            self.M_in = self.M_in * smask.view(1, N, 1).float()
            self.M_out = self.M_out * mmask.view(1, 1, N).float()

            self.W_in = torch.randn(B, N, O, device=dev) * 0.1
            self.W_out = torch.randn(B, A, N, device=dev) * 0.1
            self.b_out = torch.zeros(B, A, device=dev)

            # --- 稀疏循环连接：每行 K 个源，拒绝采样禁自连（重复源允许=权重叠加）---
            ar = torch.arange(N, device=dev).view(1, N, 1)
            idx = torch.randint(0, N, (B, N, K), device=dev)
            for _ in range(8):
                bad = idx == ar
                if not bool(bad.any()):
                    break
                idx = torch.where(bad, torch.randint(0, N, (B, N, K), device=dev), idx)
            idx = torch.where(idx == ar, (ar + 1) % N, idx)
            self.rec_idx = idx
            self.rec_w = torch.randn(B, N, K, device=dev) * 0.05

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
            if t is not None and t.is_floating_point() and t.dtype != torch.float32:
                setattr(self, g, t.float())
        self.dtype = torch.float32

    def fp16(self):
        if not getattr(self.cfg, 'USE_FP16', True):
            self.fp32()
            return
        for g in self.GENES:
            t = getattr(self, g)
            if t is not None and t.is_floating_point() and t.dtype != torch.float16:
                setattr(self, g, t.half())
        self.dtype = torch.float16

    def refresh_eff(self):
        self.W_in_eff = self.W_in * self.M_in
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

    def clone_rows(self, idx):
        sub = self[idx]
        for g in self.GENES:
            setattr(sub, g, getattr(sub, g).clone())
        for e in self.EFF:
            t = getattr(sub, e)
            if t is not None:
                setattr(sub, e, t.clone())
        return sub

    def individual_state(self, i, use_half=False):
        dt = torch.float16 if use_half else torch.float32
        cpu = torch.device('cpu')
        with torch.no_grad():
            st = {
                'N': int(self.N),
                'K': int(self.K),
                'M_in': self.M_in[i].to(cpu, dtype=torch.uint8),
                'M_out': self.M_out[i].to(cpu, dtype=torch.uint8),
                'W_in': self.W_in[i].to(cpu, dtype=dt),
                'W_out': self.W_out[i].to(cpu, dtype=dt),
                'b_out': self.b_out[i].to(cpu, dtype=dt),
                'rec_idx': self.rec_idx[i].to(cpu, dtype=torch.int16),
                'rec_w': self.rec_w[i].to(cpu, dtype=dt),
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
            self.M_out[i] = st['M_out'].float().to(dev)
            self.W_in[i] = st['W_in'].float().to(dev)
            self.W_out[i] = st['W_out'].float().to(dev)
            self.b_out[i] = st['b_out'].float().to(dev)
            self.rec_idx[i] = st['rec_idx'].long().to(dev)
            self.rec_w[i] = st['rec_w'].float().to(dev)
            self.tau_e[i] = st['tau_e_init'].float().to(dev)
            self.w_ei[i] = st['w_ei'].float().to(dev)
            self.w_ie[i] = st['w_ie'].float().to(dev)

    def pack(self):
        return {g: getattr(self, g).to('cpu').clone() for g in self.GENES}

    def unpack(self, d):
        dev = self.device
        for g in self.GENES:
            t = d[g].to(dev)
            if t.is_floating_point() and getattr(self.cfg, 'USE_FP16', True):
                t = t.half()
            setattr(self, g, t)
        self.dtype = torch.float16 if getattr(self.cfg, 'USE_FP16', True) else torch.float32


# 方向表：0=(0,1) 1=(1,0) 2=(0,-1) 3=(-1,0)；左转=idx+3 mod4，右转=idx+1 mod4
def _make_dirs(dev):
    return torch.tensor([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=torch.long, device=dev)


_DIRS_CPU = ((0, 1), (1, 0), (0, -1), (-1, 0))   # 扇区偏移表构建用（CPU 常量）


# ==========================================
# 2. GPU 批量贪吃蛇环境（CRN 固定地图版 + 通关判定）
# ==========================================
class BatchedSnakeEnv:
    """B 个独立游戏并行（全部状态为 GPU 张量）。死亡个体冻结。

    与 test16c 的差异：观测为 7b 血统 32 维（_obs32 / _obs32_fast）；step 增加
    won 通关终局（ate & body_len>=G*G → alive=False，won 标记保持）——盘满后
    不再落子/移动，规避 16c 满盘 fallback 落子退化的未定义行为。
    bank={'stream':[DRAW,2],'dir0':int} 单库（本脚本固定单地图，E=1）。
    """

    def __init__(self, cfg, B, device):
        self.cfg = cfg
        self.B = B
        self.device = device
        self.G = cfg.GRID_SIZE
        self.MAXLEN = self.G * self.G
        self.DIRS = _make_dirs(device)
        self.fast_obs = bool(getattr(cfg, 'FAST_EVAL', False))
        self._sec_offs = None                     # [4,8,G,2] 扇区射线偏移表（惰性）
        self.crn = None
        self.reset()

    # ---------- 重置 ----------
    def reset(self, bank=None):
        B, G, dev = self.B, self.G, self.device
        center = G // 2
        self.crn = bank['stream'] if isinstance(bank, dict) else bank
        self.draw_cnt = torch.zeros(B, dtype=torch.long, device=dev)
        self.head = torch.full((B, 2), center, dtype=torch.long, device=dev)
        if isinstance(bank, dict):
            dir0 = bank['dir0']
            if isinstance(dir0, torch.Tensor) and dir0.numel() > 1:
                n = B // dir0.numel()
                self.e_id = torch.arange(B, device=dev) // n
                self.dir_idx = dir0[self.e_id]
            else:
                self.e_id = torch.zeros(B, dtype=torch.long, device=dev)
                d0 = int(dir0.item() if isinstance(dir0, torch.Tensor) else dir0)
                self.dir_idx = torch.full((B,), d0, dtype=torch.long, device=dev)
        else:
            self.e_id = torch.zeros(B, dtype=torch.long, device=dev)
            self.dir_idx = torch.randint(0, 4, (B,), device=dev)
        self.body = torch.zeros(B, self.MAXLEN, 2, dtype=torch.long, device=dev)
        self.body[:, 0] = self.head
        self.body[:, 1] = self.head - self.DIRS[self.dir_idx]
        self.body_len = torch.full((B,), 2, dtype=torch.long, device=dev)
        self.food = self._place_food_init()
        self.alive = torch.ones(B, dtype=torch.bool, device=dev)
        self.won = torch.zeros(B, dtype=torch.bool, device=dev)
        self.steps = torch.zeros(B, dtype=torch.long, device=dev)
        self.steps_wo_food = torch.zeros(B, dtype=torch.long, device=dev)
        self.ate = torch.zeros(B, dtype=torch.bool, device=dev)
        self.died = torch.zeros(B, dtype=torch.long, device=dev)   # 0存活 1撞墙 2撞己 3饿死

    def _next_cand(self, mask=None):
        """取下一批落子候选 [B,2]：CRN 从公共流按个体指针，非 CRN 随机。
        mask 给定时只对 mask 行推进游标并返回有效候选（其余行返回占位值，
        由调用方 where 丢弃）——保证每行的候选消耗只由该行自身的重试需求
        决定，与 batch 内其他行完全解耦。此前全批同步消耗使每行的食物序列
        依赖"批内最慢行的重试轮数"，是同基因跨 batch 分数不同的根源。"""
        if self.crn is not None:
            stream = self.crn
            take = self.draw_cnt % stream.shape[-2]
            if mask is None:
                self.draw_cnt += 1
                if stream.dim() == 3:
                    return stream[self.e_id, take]
                return stream[take]
            self.draw_cnt = torch.where(mask, self.draw_cnt + 1, self.draw_cnt)
            if stream.dim() == 3:
                return stream[self.e_id, take]
            return stream[take]
        return torch.randint(0, self.G, (self.B, 2), device=self.device)

    def _place_food_init(self):
        dev = self.device
        B, G = self.B, self.G
        head = self.head
        neck = self.body[:, 1]
        cand = self._next_cand()
        bad = (cand == head).all(dim=1) | (cand == neck).all(dim=1)
        for _ in range(31):
            if not bad.any():
                break
            re = self._next_cand(bad)
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
        # 首轮候选也按 eat_mask 消耗：未吃食的行不推进游标（否则邻居的吃食
        # 事件会推移本行的食物序列——batch 彩票的第二个耦合源）
        cand = self._next_cand(eat_mask)
        bad = eat_mask & occ_b.gather(1, (cand[:, 0] * G + cand[:, 1]).unsqueeze(1)).squeeze(1)
        for _ in range(31):
            if not bad.any():
                break
            re = self._next_cand(bad)
            cand = torch.where(bad.unsqueeze(1), re, cand)
            bad = bad & occ_b.gather(1, (cand[:, 0] * G + cand[:, 1]).unsqueeze(1)).squeeze(1)
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

    # ---------- 观测（7b 血统 32 维：8 扇区欧氏食物投影 + 1/k 身体/障碍扇区）----------
    def obs(self):
        if self.fast_obs:
            return self._obs32_fast()
        return self._obs32()

    def _sector_offset_table(self):
        """[4,8,G,2] 射线偏移表（惰性构建一次）：d8 顺序同 7b _obs32
        （前/左前/左/左后/后/右后/右/右前），按 dir_idx 预旋转。"""
        if self._sec_offs is not None:
            return self._sec_offs
        G, dev = self.G, self.device
        table = torch.zeros(4, 8, G, 2, dtype=torch.long, device=dev)
        for di in range(4):
            d = tuple(_DIRS_CPU[di])
            left = tuple(_DIRS_CPU[(di + 3) % 4])
            right = tuple(_DIRS_CPU[(di + 1) % 4])
            d8 = (d,                                  # 0 前
                  (d[0] + left[0], d[1] + left[1]),   # 1 左前
                  left,                               # 2 左
                  (left[0] - d[0], left[1] - d[1]),   # 3 左后
                  (-d[0], -d[1]),                     # 4 后
                  (right[0] - d[0], right[1] - d[1]), # 5 右后
                  right,                              # 6 右
                  (d[0] + right[0], d[1] + right[1])) # 7 右前
            for i, (dr, dc) in enumerate(d8):
                table[di, i, :, 0] = dr * torch.arange(1, G + 1)
                table[di, i, :, 1] = dc * torch.arange(1, G + 1)
        self._sec_offs = table
        return table

    def _d8_vectors(self):
        """[B,8,2] 相对方向向量（7b _obs32 同表：前/左前/左/左后/后/右后/右/右前）。"""
        d = self.DIRS[self.dir_idx]
        left = self.DIRS[(self.dir_idx + 3) % 4]
        right = self.DIRS[(self.dir_idx + 1) % 4]
        return torch.stack((d, d + left, left, left - d, -d, right - d, right, d + right),
                           dim=1)

    def _obs32(self):
        """32 维投影观测标量版（与 test7b._obs32 逐通道同式；慢路径/play 用）。"""
        B, G, dev = self.B, self.G, self.device
        head = self.head
        food = self.food
        d = self.DIRS[self.dir_idx]                       # [B,2]

        obs = torch.zeros(B, self.cfg.OBS_DIM, dtype=torch.float32, device=dev)

        # --- [0:4] 蛇首方向 one-hot ---
        obs[:, 0] = ((d[:, 0] == 0) & (d[:, 1] == 1)).float()
        obs[:, 1] = ((d[:, 0] == 1) & (d[:, 1] == 0)).float()
        obs[:, 2] = ((d[:, 0] == 0) & (d[:, 1] == -1)).float()
        obs[:, 3] = ((d[:, 0] == -1) & (d[:, 1] == 0)).float()

        # --- [4:8] 蛇尾方向 one-hot ---
        tail = self.body[torch.arange(B, device=dev), (self.body_len - 1).clamp(min=0)]
        prev = self.body[torch.arange(B, device=dev), (self.body_len - 2).clamp(min=0)]
        tail_dr = prev[:, 0] - tail[:, 0]
        tail_dc = prev[:, 1] - tail[:, 1]
        obs[:, 4] = ((tail_dr == 0) & (tail_dc == 1)).float()
        obs[:, 5] = ((tail_dr == 1) & (tail_dc == 0)).float()
        obs[:, 6] = ((tail_dr == 0) & (tail_dc == -1)).float()
        obs[:, 7] = ((tail_dr == -1) & (tail_dc == 0)).float()

        # --- [8:16] 食物 8 扇区投影式连续感知（忽略遮蔽，永远可见）---
        # channel_i = K · max(0, v·û_i)/|û| / |v|²（欧氏，û 归一化；7b 同式）
        d8 = self._d8_vectors()
        ar = torch.arange(B, device=dev)
        vr = (food[:, 0] - head[:, 0]).float()
        vc = (food[:, 1] - head[:, 1]).float()
        d2 = (vr * vr + vc * vc).clamp(min=1.0)           # |v|²，食物不在头上故 >= 1
        k_scale = float(getattr(self.cfg, 'OBS_FOOD_SCALE', 1.0))
        dx = d8[:, :, 0].float()
        dy = d8[:, :, 1].float()
        norm = torch.sqrt(dx * dx + dy * dy)              # 1 或 √2（逐个体朝向）
        dot = (vr[:, None] * dx + vc[:, None] * dy) / norm
        obs[:, 8:16] = torch.clamp(dot, min=0.0) / d2[:, None] * k_scale

        # --- [16:24] 自身 8 扇区距离倒数 ---
        flat_body = self.body[:, :, 0] * G + self.body[:, :, 1]
        seg_idx = torch.arange(self.MAXLEN, device=dev)
        seg_valid = (seg_idx[None, :] >= 1) & (seg_idx[None, :] < self.body_len[:, None])
        bf = torch.zeros(B, G * G, dtype=torch.long, device=dev)
        bf.scatter_add_(1, flat_body.clamp(max=G * G - 1), seg_valid.long())
        body_set_mask = bf.view(B, G, G) > 0

        for i in range(8):
            ddr, ddc = d8[:, i, 0], d8[:, i, 1]
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
            ddr, ddc = d8[:, i, 0], d8[:, i, 1]
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

        # 身后约定：障碍数 = sqrt(蛇身长度/格子度)（写 28 后整块缩放，7b 同序）
        obs[:, 28] = torch.sqrt(self.body_len.float().clamp(min=1) / self.G)
        k_obs = float(getattr(self.cfg, 'OBS_OBSTACLE_SCALE', 1.0))
        if k_obs != 1.0:
            obs[:, 24:32] = obs[:, 24:32] * k_obs

        return obs

    def _obs32_fast(self):
        """_obs32 的向量化等价实现（fast-eval）：[8:16] 食物 8 扇区投影一次算出；
        [16:32] 16 扇区射线由逐 k 循环改为偏移表一次 gather + cumsum 首命中
        （首命中/出界/默认值语义逐位一致，自检 8 对拍覆盖）。"""
        B, G, dev = self.B, self.G, self.device
        head = self.head
        food = self.food
        d = self.DIRS[self.dir_idx]                       # [B,2]

        obs = torch.zeros(B, self.cfg.OBS_DIM, dtype=torch.float32, device=dev)

        # --- [0:4] 蛇首方向 one-hot ---
        obs[:, 0] = ((d[:, 0] == 0) & (d[:, 1] == 1)).float()
        obs[:, 1] = ((d[:, 0] == 1) & (d[:, 1] == 0)).float()
        obs[:, 2] = ((d[:, 0] == 0) & (d[:, 1] == -1)).float()
        obs[:, 3] = ((d[:, 0] == -1) & (d[:, 1] == 0)).float()

        # --- [4:8] 蛇尾方向 one-hot ---
        tail = self.body[torch.arange(B, device=dev), (self.body_len - 1).clamp(min=0)]
        prev = self.body[torch.arange(B, device=dev), (self.body_len - 2).clamp(min=0)]
        tail_dr = prev[:, 0] - tail[:, 0]
        tail_dc = prev[:, 1] - tail[:, 1]
        obs[:, 4] = ((tail_dr == 0) & (tail_dc == 1)).float()
        obs[:, 5] = ((tail_dr == 1) & (tail_dc == 0)).float()
        obs[:, 6] = ((tail_dr == 0) & (tail_dc == -1)).float()
        obs[:, 7] = ((tail_dr == -1) & (tail_dc == 0)).float()

        # --- [8:16] 食物 8 扇区欧氏投影（7b 同式，全向量化）---
        d8 = self._d8_vectors()
        ar = torch.arange(B, device=dev)
        vr = (food[:, 0] - head[:, 0]).float()
        vc = (food[:, 1] - head[:, 1]).float()
        d2 = (vr * vr + vc * vc).clamp(min=1.0)
        k_scale = float(getattr(self.cfg, 'OBS_FOOD_SCALE', 1.0))
        dx = d8[:, :, 0].float()
        dy = d8[:, :, 1].float()
        norm = torch.sqrt(dx * dx + dy * dy)
        dot = (vr[:, None] * dx + vc[:, None] * dy) / norm
        obs[:, 8:16] = torch.clamp(dot, min=0.0) / d2[:, None] * k_scale

        # --- [16:24]/[24:32] 16 扇区射线：向量化首命中 ---
        flat_body = self.body[:, :, 0] * G + self.body[:, :, 1]
        seg_idx = torch.arange(self.MAXLEN, device=dev)
        seg_valid = (seg_idx[None, :] >= 1) & (seg_idx[None, :] < self.body_len[:, None])
        bf = torch.zeros(B, G * G, dtype=torch.long, device=dev)
        bf.scatter_add_(1, flat_body.clamp(max=G * G - 1), seg_valid.long())
        body_set_mask = bf.view(B, G, G) > 0

        offs = self._sector_offset_table()[self.dir_idx]          # [B,8,G,2]
        r = head[:, 0].view(B, 1, 1) + offs[..., 0]               # [B,8,G]
        c = head[:, 1].view(B, 1, 1) + offs[..., 1]
        inb = (r >= 0) & (r < G) & (c >= 0) & (c < G)
        rc = r.clamp(0, G - 1)
        cc = c.clamp(0, G - 1)
        bodyhit = body_set_mask[ar.view(B, 1, 1), rc, cc]         # [B,8,G]

        # 自身扇区：首个 inb&身体 命中距离（射线出界单调 → 首命中语义一致）
        hit = inb & bodyhit
        has_hit = hit.any(dim=-1)
        cum_hit = torch.cumsum(hit.to(torch.int32), dim=-1)
        kidx = (cum_hit == 1).to(torch.int32).argmax(dim=-1)      # 首命中 k-1
        dist_self = torch.where(has_hit, kidx + 1,
                                torch.full_like(kidx, G + 1)).float()
        obs[:, 16:24] = torch.where(dist_self <= G, 1.0 / dist_self,
                                    torch.zeros_like(dist_self))
        k_self = float(getattr(self.cfg, 'OBS_SELF_SCALE', 1.0))
        if k_self != 1.0:
            obs[:, 16:24] = obs[:, 16:24] * k_self

        # 障碍扇区：首个 出界|身体 阻挡距离（默认 G → 1/G，与 _obs32 一致）
        blocked = (~inb) | bodyhit
        has_blk = blocked.any(dim=-1)
        cum_blk = torch.cumsum(blocked.to(torch.int32), dim=-1)
        kblk = (cum_blk == 1).to(torch.int32).argmax(dim=-1)
        dist_o = torch.where(has_blk, kblk + 1, torch.full_like(kblk, G)).float()
        obs[:, 24:32] = 1.0 / dist_o
        # 身后约定：障碍数 = sqrt(蛇身长度/格子度)（原实现覆盖列 28，同序保留）
        obs[:, 28] = torch.sqrt(self.body_len.float().clamp(min=1) / self.G)
        k_obs = float(getattr(self.cfg, 'OBS_OBSTACLE_SCALE', 1.0))
        if k_obs != 1.0:
            obs[:, 24:32] = obs[:, 24:32] * k_obs

        return obs

    def sees_food(self, obs):
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
        # --- 通关判定：吃到盘满（body_len = G*G）→ won 终局（won 不计死因）---
        won_now = ate & (self.body_len >= self.MAXLEN)
        self.won = self.won | won_now
        self.alive = alive_f & (~crash) & (~starve) & (~won_now)
        # --- 个体级步数上限：活满即冻结（不计死因，防磨蹭个体拖垮整代评估）---
        cap = int(getattr(self.cfg, 'EP_STEPS_CAP', 0))
        if cap > 0:
            self.alive = self.alive & ~(alive_f & (self.steps >= cap))
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
# 3. 批量前向（无激素 E-I 动力学；疲劳为 7b cts 口径，逐字迁移）
# ==========================================
def forward_batch(pop, obs, E, I, st, cts, cfg):
    """单次 E-I 迭代（B 个个体并行，无激素支路）。test16：循环项为固定扇入
    gather-乘-归约；疲劳 = relu(cts - THRESH)·GAIN clamp MAX（7b 按动作计数）。"""
    ext = torch.bmm(pop.W_in_eff, obs.unsqueeze(-1)).squeeze(-1)
    K = pop.rec_idx.shape[-1]
    Eg = torch.gather(E.unsqueeze(-1).expand(E.shape[0], E.shape[1], K),
                      1, pop.rec_idx)                     # [B,N,K] = E[b, idx[b,i,k]]
    rec = (Eg * pop.rec_w).sum(-1)                        # [B,N]
    total = ext + rec

    st = cfg.SHORT_TERM_DECAY * st + (1 - cfg.SHORT_TERM_DECAY) * E
    tau = (pop.tau_e + cfg.SHORT_TERM_GAIN * st).clamp(cfg.TAU_E_MIN, cfg.TAU_E_MAX)
    w_ei = pop.w_ei.clamp(cfg.W_EI_MIN, cfg.W_EI_MAX)
    w_ie = pop.w_ie.clamp(cfg.W_IE_MIN, cfg.W_IE_MAX)

    E_new = torch.sigmoid(total + tau * E - w_ei * I)
    I_new = torch.sigmoid(w_ie * E_new)

    logits = torch.bmm(pop.W_out_eff, E_new.unsqueeze(-1)).squeeze(-1) + pop.b_out
    fatigue = torch.relu(cts - cfg.FATIGUE_THRESHOLD) * cfg.FATIGUE_GAIN
    logits = logits - fatigue.clamp(max=cfg.FATIGUE_MAX)
    return logits, E_new, I_new, st


def update_fatigue(cts, action):
    """7b 口径：当前动作计数 +1（one-hot scatter）。"""
    cur = cts.gather(1, action.unsqueeze(1)) + 1.0
    ncts = torch.zeros_like(cts)
    ncts.scatter_(1, action.unsqueeze(1), cur)
    return ncts


def deliberate_batch(pop, obs, E, I, st, cts, cfg):
    """K 倍帧率思考：内部迭代 K 次，logits 平均后 argmax（与 7b/16c 一致）。"""
    K = cfg.FRAME_RATE
    logits_sum = None
    for k in range(K):
        o = obs * (cfg.INPUT_DECAY ** k)
        logits, E, I, st = forward_batch(pop, o, E, I, st, cts, cfg)
        logits_sum = logits if logits_sum is None else logits_sum + logits
    action = torch.argmax(logits_sum, dim=1)
    return action, E, I, st


# ==========================================
# 4. 种群评估（固定单地图；确定性 → 每个体 1 局）
# ==========================================
def reach_ratio(env, tail_invalid=True):
    """从蛇头可达的自由格占比 [B]（仅观测/诊断用，不进适应度）。"""
    B, G, dev = env.B, env.G, env.device
    occ = env._occupancy_flat(tail_invalid=tail_invalid).view(B, 1, G, G)
    free = occ < 0.5
    reach = torch.zeros(B, 1, G, G, device=dev)
    reach[torch.arange(B, device=dev), 0, env.head[:, 0], env.head[:, 1]] = 1.0
    mp = torch.nn.functional.max_pool2d
    for _ in range(G * G):
        cross = torch.maximum(mp(reach, (3, 1), stride=1, padding=(1, 0)),
                              mp(reach, (1, 3), stride=1, padding=(0, 1)))
        reach = cross * free
    free_cnt = free.sum(dim=(1, 2, 3)).float()
    reach_cnt = (reach > 0).float().sum(dim=(1, 2, 3))
    return torch.where(free_cnt > 0, reach_cnt / free_cnt.clamp(min=1.0),
                       torch.ones_like(reach_cnt))


def _eval_sweep_chunk_fast(pop_rep, cfg, bank):
    """fast-eval 扫描（16b 口径：遥测降频 + 无每步同步 + all_done 周期检查）。
    返回 [P,12] 单局指标：0=food 1=seen 2=unseen 3=steps_last 4=prox
    5=撞墙 6=撞己 7=饿死 8=act1 9=act2 10=turn_last 11=avg_reach。
    固定地图确定性评估：同一 bank 全体共用，1 局/体。"""
    B = pop_rep.P
    dev = pop_rep.device
    N = pop_rep.N
    env = BatchedSnakeEnv(cfg, B, dev)
    env.reset(bank=bank)
    half = torch.float16 if getattr(cfg, 'USE_FP16', True) else torch.float32

    tot_food = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_seen = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_unseen = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_prox = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_act1 = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_act2 = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_reach = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_reach_n = torch.zeros(B, dtype=torch.float32, device=dev)
    ate_pending = torch.zeros(B, dtype=torch.bool, device=dev)

    E = torch.zeros(B, N, dtype=half, device=dev)
    I = torch.zeros(B, N, dtype=half, device=dev)
    st = torch.zeros(B, N, dtype=half, device=dev)
    cts = torch.zeros(B, pop_rep.A, dtype=half, device=dev)
    last = torch.zeros(B, dtype=torch.float32, device=dev)
    turn_cnt = torch.zeros(B, dtype=torch.float32, device=dev)
    turn_last = torch.zeros(B, dtype=torch.float32, device=dev)

    REACH_EVERY = max(1, int(getattr(cfg, 'REACH_EVERY', 16)))
    DONE_EVERY = max(1, int(getattr(cfg, 'ALLDONE_EVERY', 8)))

    for t in range(cfg.MAX_STEPS):
        al = env.alive
        obs = env.obs().to(half)
        sees = env.sees_food(obs)
        tot_seen += (al & sees).float()
        tot_unseen += (al & (~sees)).float()
        fr = (env.food[:, 0] - env.head[:, 0]).float()
        fc = (env.food[:, 1] - env.head[:, 1]).float()
        dist = torch.sqrt(fr * fr + fc * fc).clamp(min=1.0)
        tot_prox += al.float() / dist

        act, E, I, st = deliberate_batch(pop_rep, obs, E, I, st, cts, cfg)
        cts = update_fatigue(cts, act)
        tot_act1 += (al & (act == 1)).float()
        tot_act2 += (al & (act == 2)).float()
        turn_cnt += (al & (act != 0)).float()

        env.step(act)
        ate_now = al & env.ate
        tot_food += ate_now.float()
        last = torch.where(ate_now, torch.full_like(last, float(t + 1)), last)
        turn_last = torch.where(ate_now, turn_cnt, turn_last)
        ate_pending |= ate_now
        if t % REACH_EVERY == REACH_EVERY - 1:
            if bool(ate_pending.any()):
                rr = reach_ratio(env)
                tot_reach += rr * ate_pending.float()
                tot_reach_n += ate_pending.float()
                ate_pending = torch.zeros_like(ate_pending)
        if t % DONE_EVERY == DONE_EVERY - 1 and env.all_done():
            break

    # 尾窗 flush（break / MAX_STEPS 出口统一处理）
    if bool(ate_pending.any()):
        rr = reach_ratio(env)
        tot_reach += rr * ate_pending.float()
        tot_reach_n += ate_pending.float()

    avg_reach = tot_reach / tot_reach_n.clamp(min=1.0)
    metrics = torch.stack((tot_food, tot_seen, tot_unseen, last,
                           tot_prox / torch.clamp(tot_seen + tot_unseen, min=1e-6),
                           (env.died == 1).float(),
                           (env.died == 2).float(),
                           (env.died == 3).float(),
                           tot_act1, tot_act2, turn_last, avg_reach), dim=1)
    if B >= 64:
        print(f"  [sweep] rows={B} steps_run={t + 1} "
              f"max_indiv_steps={int(env.steps.max().item())} "
              f"alive_end={int(env.alive.sum().item())}")
    return metrics


def _eval_solo_one(st, cfg, bank, k_rep, noise_std, seed):
    """单个体的 B=1 真值评估（worker 侧）：k_rep 份副本（原权重+微扰）取最差。"""
    dev = torch.device('cuda')
    torch.manual_seed(seed)
    p1 = GeneStack(cfg, B=1, device=dev)
    p1.random_init()
    p1.set_individual_from_state(0, st)
    rows = []
    for r in range(k_rep):
        if r > 0:
            with torch.no_grad():
                for g in ('W_in', 'W_out', 'rec_w'):
                    tt = getattr(p1, g)
                    setattr(p1, g, tt + torch.randn_like(tt) * noise_std)
                b = p1.b_out
                p1.b_out = b + torch.randn_like(b) * noise_std
                tau = p1.tau_e
                p1.tau_e = (tau + torch.randn_like(tau) * noise_std
                            ).clamp(cfg.TAU_E_MIN, cfg.TAU_E_MAX)
        p1.refresh_eff()
        if getattr(cfg, 'USE_FP16', True):
            p1.fp16(); p1.refresh_eff()
        env = BatchedSnakeEnv(cfg, 1, dev)
        env.reset(bank=bank)
        E = torch.zeros(1, p1.N, dtype=p1.dtype, device=dev)
        I = torch.zeros_like(E); stt = torch.zeros_like(E)
        cts = torch.zeros(1, p1.A, dtype=p1.dtype, device=dev)
        food = 0
        last_eat = 0
        for s in range(int(cfg.MAX_STEPS)):
            if not bool(env.alive.any()):
                break
            obs = env.obs().to(p1.dtype)
            act, E, I, stt = deliberate_batch(p1, obs, E, I, stt, cts, cfg)
            cts = update_fatigue(cts, act)
            env.step(act)
            if bool(env.ate.any()):
                food += int(env.ate.sum())
                last_eat = s + 1
        rows.append((env, food, last_eat))
    # 取最差副本（food 最小者）组 metrics 行（列序同 sweep：0=food 1=seen
    # 2=unseen 3=steps_last 5/6/7=死因）
    e, food, last_eat = min(rows, key=lambda x: x[1])
    seen = float(e.steps.max())
    wall = float((e.died == 1).sum()); selfd = float((e.died == 2).sum())
    starve = float((e.died == 3).sum())
    row = torch.tensor([float(food), seen, 0.0, float(last_eat), 0.0, wall,
                        selfd, starve, 0.0, 0.0, 0.0, 0.0])
    return row


def eval_worker_main(seg_path, out_path):
    """--eval-worker 子进程入口：载入分段（个体states+cfg+bank），B=1 真值评估，
    结果 [n,12] 存 out_path。每个 worker 独立 CUDA 上下文，多 worker 并行互补。"""
    payload = torch.load(seg_path, map_location='cpu', weights_only=False)
    cfg = Config()
    for k, v in payload['cfg'].items():
        if not k.startswith('__'):
            try:
                setattr(cfg, k, v)
            except Exception:
                pass
    cfg.DEVICE = 'cuda'
    dev = torch.device('cuda')
    bank = {'stream': payload['bank_stream'].to(dev),
            'dir0': int(payload['dir0'])}
    k_rep = int(payload['k_rep'])
    noise_std = float(payload['noise_std'])
    out = torch.zeros(len(payload['states']), 12)
    for i, st in enumerate(payload['states']):
        out[i] = _eval_solo_one(st, cfg, bank, k_rep, noise_std,
                                int(payload['seed']) + i * 31)
    torch.save({'metrics': out.cpu()}, out_path)
    print(f"[worker] 段完成: {len(payload['states'])} 个体 -> {out_path}", flush=True)


def _eval_solo_parallel(pop, cfg, bank, n_workers):
    """EVAL_SOLO 多进程并行：128 个体分 n_workers 段，每段一个子进程独立
    CUDA 上下文做 B=1 真值评估；段文件通信，主进程合并 [P,12]。
    注意：微扰噪声由各 worker 独立随机——跨 checkpoint 的逐位重演一致性
    不保留（GA 语义不受影响，min 口径仍是合法的稳健性测量）。"""
    import subprocess
    K_rep = max(1, int(getattr(cfg, 'ROBUST_EVAL', 1)))
    noise_std = float(getattr(cfg, 'ROBUST_NOISE_STD', 1e-3))
    P = pop.P
    ncol = 12
    n_workers = max(1, min(int(n_workers), P))
    seg = (P + n_workers - 1) // n_workers
    cfg_dict = _cfg_dict(cfg)
    bank_cpu = {'stream': bank['stream'].cpu(), 'dir0': int(bank['dir0'])}
    tmpdir = os.path.dirname(os.path.abspath(cfg.CHECKPOINT_PATH)) or '.'
    stamp = f"{int(time.time())}_{os.getpid()}"
    seg_paths, out_paths, procs = [], [], []
    try:
        for w in range(n_workers):
            lo, hi = w * seg, min((w + 1) * seg, P)
            if lo >= hi:
                break
            states = [pop.individual_state(i, use_half=False) for i in range(lo, hi)]
            sp = os.path.join(tmpdir, f'_solo_seg_{stamp}_{w}.pth')
            op = os.path.join(tmpdir, f'_solo_out_{stamp}_{w}.pth')
            torch.save({'states': states, 'cfg': cfg_dict,
                        'bank_stream': bank_cpu['stream'], 'dir0': bank_cpu['dir0'],
                        'k_rep': K_rep, 'noise_std': noise_std,
                        'seed': random.randrange(1 << 30)}, sp)
            seg_paths.append(sp); out_paths.append(op)
            procs.append(subprocess.Popen(
                [sys.executable, '-X', 'utf8', '-u', os.path.abspath(__file__),
                 '--eval-worker', sp, '--eval-worker-out', op],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        codes = [p.wait() for p in procs]
        if any(c != 0 for c in codes):
            raise RuntimeError(f'solo worker 失败: exit codes {codes}')
        out = torch.zeros(P, ncol)
        for w, op in enumerate(out_paths):
            part = torch.load(op, map_location='cpu', weights_only=False)['metrics']
            lo = w * seg
            out[lo:lo + part.shape[0]] = part
        return out
    finally:
        for pth in seg_paths + out_paths:
            try:
                os.remove(pth)
            except OSError:
                pass


def _eval_pop_banks(pop, cfg, bank):
    """评估全种群返回 [P,12]。OOM 时分块自动减半重试。

    EVAL_SOLO=True（B=1 真值口径）：逐个体单独成行评估。口径实验实测：任何
    多行 batch（连 B=2 都）会因 kernel 路径差异引发数值微差并被混沌放大，
    分数系统性偏离单行真值（高估 ~21 分且含环境彩票）；只有 B=1 与回放口径
    一致、fp16/fp32 在 B=1 下一致（秩相关 0.99）。慢（逐个体串行），但唯一自洽。
    ROBUST_EVAL=K>1：每个体 K 份副本（第0份原权重、其余 σ=ROBUST_NOISE_STD
    微扰），取最差副本——权重邻域稳健性筛选。"""
    dev = pop.device
    ncol = 12
    if bool(getattr(cfg, 'EVAL_SOLO', False)):
        n_workers = int(getattr(cfg, 'EVAL_WORKERS', 1))
        if n_workers > 1:
            return _eval_solo_parallel(pop, cfg, bank, n_workers)
        K_rep = max(1, int(getattr(cfg, 'ROBUST_EVAL', 1)))
        noise_std = float(getattr(cfg, 'ROBUST_NOISE_STD', 1e-3))
        out = torch.zeros(pop.P, ncol)
        t_solo = time.perf_counter()
        for i in range(pop.P):
            cands = []
            for r in range(K_rep):
                one = pop[i:i+1]
                if r > 0:
                    with torch.no_grad():
                        for g in ('W_in', 'W_out', 'rec_w'):
                            tt = getattr(one, g)
                            setattr(one, g, tt + torch.randn_like(tt) * noise_std)
                        b = one.b_out
                        one.b_out = b + torch.randn_like(b) * noise_std
                        tau = one.tau_e
                        one.tau_e = (tau + torch.randn_like(tau) * noise_std
                                     ).clamp(cfg.TAU_E_MIN, cfg.TAU_E_MAX)
                one.refresh_eff()
                cands.append(_eval_sweep_chunk_fast(one, cfg, bank).cpu())
            stack = torch.stack(cands, dim=0)
            out[i] = stack[int(stack[:, 0, 0].argmin()), 0]
            if (i + 1) % 16 == 0:
                print(f"  [solo] {i + 1}/{pop.P} "
                      f"({time.perf_counter() - t_solo:.0f}s)", flush=True)
        return out
    K_rep = max(1, int(getattr(cfg, 'ROBUST_EVAL', 1)))
    max_B = _auto_eval_batch(cfg, dev) // K_rep
    out = torch.zeros(pop.P, ncol)
    lo = 0
    while lo < pop.P:
        hi = min(lo + max_B, pop.P)
        try:
            sub = pop[lo:hi]                              # 切片=拷贝，基因栈安全
            n = hi - lo
            if K_rep > 1:
                noise_std = float(getattr(cfg, 'ROBUST_NOISE_STD', 1e-3))
                idx = torch.arange(n, device=dev).repeat_interleave(K_rep)
                big = pop[lo + idx]                       # [n*K, ...] 行主序 p0r0,p0r1,...
                with torch.no_grad():
                    for r in range(1, K_rep):
                        sl = slice(r, n * K_rep, K_rep)   # 第 r 份副本
                        for g in ('W_in', 'W_out', 'rec_w'):
                            t = getattr(big, g)[sl]
                            getattr(big, g)[sl] = t + torch.randn_like(t) * noise_std
                        b = big.b_out[sl]
                        big.b_out[sl] = b + torch.randn_like(b) * noise_std
                        tau = big.tau_e[sl]
                        big.tau_e[sl] = (tau + torch.randn_like(tau) * noise_std
                                         ).clamp(cfg.TAU_E_MIN, cfg.TAU_E_MAX)
                big.refresh_eff()
                m = _eval_sweep_chunk_fast(big, cfg, bank).cpu()
                m = m.view(n, K_rep, ncol)
                worst = m[:, :, 0].argmin(dim=1)          # 最差副本（按 food）
                ar = torch.arange(n)
                out[lo:hi] = m[ar, worst]
            else:
                sub.refresh_eff()
                m = _eval_sweep_chunk_fast(sub, cfg, bank).cpu()
                out[lo:hi] = m
            lo = hi
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            max_B = max(32, max_B // 2)
            print(f"  [fast-eval] OOM：分块减半 -> 每块 {max_B} 个体")
    return out


def evaluate_population_fixedmap(pop, cfg, bank):
    """固定单地图评估：全体个体同一条确定性食物序列各 1 局 → 无评估噪声，
    排序直接按 simple 适应度（无 LCB）。返回 (metrics[P,12], order[P])。"""
    if getattr(cfg, 'USE_FP16', True):
        pop.fp16()
    metrics = _eval_pop_banks(pop, cfg, bank)
    mn = metrics.numpy()
    key_fn = _make_key_fn(cfg)
    order = sorted(range(pop.P), key=lambda i: key_fn(mn[i]), reverse=True)
    return metrics, order


# ==========================================
# 5. 进化（类正态变异强度 + GPU 向量化交叉/变异；16c 原样）
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


def evolve_topology_gpu(pop, metrics, cfg, gen=0, order=None):
    """进化下一代：ELITE 精英 + (P-ELITE) 后代（交叉同 16c；rec 按行整行选父，
    保持每行 K 槽扇入结构）。"""
    P = pop.P
    N = pop.N
    dev = pop.device
    active = _freeze_active_groups(gen, cfg)
    has_g1 = 'G1' in active
    has_g2 = 'G2' in active

    # --- 精英排序 ---
    if order is None:
        mn = metrics.cpu().numpy()
        key_fn = _make_key_fn(cfg)
        order = sorted(range(P), key=lambda i: key_fn(mn[i]), reverse=True)
    elite_idx = order[:cfg.ELITE_SIZE]
    elites = pop[elite_idx]

    new_pop = pop.empty()
    children = pop.empty(B=P - cfg.ELITE_SIZE)

    # --- 后代：父代采样 + 交叉 ---
    E = cfg.ELITE_SIZE
    B2 = P - E
    p1_idx = torch.randint(0, E, (B2,), device=dev)
    p2_idx = torch.randint(0, E, (B2,), device=dev)
    p2_idx = torch.where(p2_idx == p1_idx, (p1_idx + 1) % E, p2_idx)

    p1 = elites[p1_idx]
    p2 = elites[p2_idx]

    # --- 默认全部基因单亲（p1）遗传，活跃组再交叉/变异覆盖（16c 修复版语义）---
    with torch.no_grad():
        for g in GeneStack.GENES:
            setattr(children, g, getattr(p1, g).clone())

    # --- 每子代变异强度因子（lognormal 中位数 1；右偏大变异）---
    s = sample_mut_scale(cfg, B2, dev)
    topo_mut_prob_i = (cfg.TOPOLOGY_MUT_PROB * s).clamp(max=0.5)          # [B2]
    mask_mut_rate_i = (cfg.MUT_RATE * s).clamp(max=0.5)                   # [B2]
    weight_frac1 = (cfg.WEIGHT_MUT_FRAC * s).clamp(max=1.0)               # [B2]
    weight_std1 = (cfg.WEIGHT_MUT_STD * s)                                # [B2]
    tau_std2 = (cfg.TAU_E_MUT_STD * s).view(B2, 1)
    wei_std2 = (cfg.W_EI_MUT_STD * s).view(B2, 1)
    wie_std2 = (cfg.W_IE_MUT_STD * s).view(B2, 1)

    with torch.no_grad():
        # ---- G1 交叉：结构组（掩码 + 权重 + 输出偏置；rec 按行选父）----
        if has_g1:
            col_mask = torch.rand(B2, N, device=dev) > 0.5
            rec_row = torch.rand(B2, N, device=dev) > 0.5
            children.W_in = torch.where(col_mask.unsqueeze(2), p1.W_in, p2.W_in)
            children.M_in = torch.where(col_mask.unsqueeze(2), p1.M_in, p2.M_in)
            children.rec_idx = torch.where(rec_row.unsqueeze(-1), p1.rec_idx, p2.rec_idx)
            children.rec_w = torch.where(rec_row.unsqueeze(-1), p1.rec_w, p2.rec_w)
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

        # ---- 变异（逐组独立，冻结组跳过；强度逐子代 s 缩放）----
        if has_g1:
            pick = torch.randint(0, 3, (B2,), device=dev)
            topo_gate = torch.rand(B2, device=dev) < topo_mut_prob_i
            for ai, attr in enumerate(GeneStack.G1_MASKS):
                sel = (pick == ai) & topo_gate
                if not bool(sel.any().item()):
                    continue
                if attr == 'rec_idx':
                    # 槽位重连：被选子代按 MUT_RATE*s 概率把槽位重连到随机新目标
                    # j≠自身（重复源允许=权重叠加，满足"≤K 连接"约束）
                    t = children.rec_idx
                    rate3 = mask_mut_rate_i.view(B2, 1, 1)
                    flip = torch.rand_like(t, dtype=torch.float32) < rate3.to(torch.float32)
                    ar = torch.arange(N, device=dev).view(1, N, 1)
                    newj = torch.randint(0, N, t.shape, device=dev)
                    newj = torch.where(newj == ar, (ar + 1) % N, newj)
                    setattr(children, attr,
                            torch.where(sel[:, None, None] & flip, newj, t))
                else:
                    t = getattr(children, attr)
                    rate3 = mask_mut_rate_i.view(B2, 1, 1)
                    flip = torch.rand_like(t) < rate3
                    allow = None
                    if attr == 'M_in':
                        allow = _pool_masks(cfg, N, dev)[0].view(1, N, 1)
                    elif attr == 'M_out':
                        allow = _pool_masks(cfg, N, dev)[1].view(1, 1, N)
                    if allow is not None:      # 池边界守卫（池关闭时全 True=无约束）
                        flip = flip & allow
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
# 6. 保存 / 加载（断点 + 最优模型 + 7b 转换器）
# ==========================================
def load_best_state_7b_sparse(path, cfg, verbose=True):
    """7b 稠密基因组 → 16 稀疏固定扇入转换器（无损条件：每行掩码内非零 ≤ K）。

    - M_rec/W_rec [N,N] → 每行取掩码内 |W| 最强的 K 个源作 rec_idx/rec_w；
      行非零数 ≤ K 时全部保留（多余槽位权重 0=断开，可被后续变异激活）。
    - M_in/W_in/M_out/W_out/b_out/tau_e/w_ei/w_ie 形状原生匹配（7b 血统 N/O 同构）。
    返回 (state dict, food, steps) 或 None。"""
    if not path or not os.path.exists(path):
        if verbose:
            print(f"  [Seed7b] 模型不存在: {path}")
        return None
    try:
        data = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        if verbose:
            print(f"  [Seed7b] 模型 {path} 读取失败 ({e})")
        return None
    saved = data.get('config', {})
    if saved and (saved.get('NUM_COLUMNS') != cfg.NUM_COLUMNS or
                  saved.get('OBS_DIM') != cfg.OBS_DIM or
                  saved.get('ACTION_DIM') != cfg.ACTION_DIM):
        if verbose:
            print(f"  [Seed7b] 维度不匹配（种子须为 7b 血统 N={cfg.NUM_COLUMNS}/"
                  f"OBS={cfg.OBS_DIM}）: N={saved.get('NUM_COLUMNS')} "
                  f"OBS={saved.get('OBS_DIM')} ACT={saved.get('ACTION_DIM')}")
        return None
    st = data.get('brain')
    if st is None or 'M_rec' not in st or 'W_rec' not in st or 'rec_idx' in st:
        if verbose:
            print("  [Seed7b] 不是 7b 稠密基因组（缺 M_rec/W_rec 或已是稀疏格式）")
        return None

    N, K = cfg.NUM_COLUMNS, int(cfg.REC_FANIN)
    M_rec = (st['M_rec'] > 0)
    W_rec = st['W_rec'].float()
    # 关键：7b 的 W_rec 几乎全稠密（稀疏性只在 M_rec 掩码），先乘掩码取"有效
    # 权重"再 top-|W| 采样——否则 topk 拉入的掩码外槽位会 gather 出幽灵权重，
    # 引入 7b 前向中不存在的循环连接（实测约 45% 槽位被污染，行为严重退化）。
    W_eff = W_rec * M_rec.float()
    nz_row = M_rec.sum(dim=1).float()
    _, rec_idx = torch.topk(W_eff.abs(), k=K, dim=1)     # 掩码内 |W| 最强的 K 源
    rec_w = W_eff.gather(1, rec_idx)                     # 掩码外槽位恒 0（幽灵免疫）
    # 无自连不变式：topk 拉入的 0 权重对角槽位重定向到下一行（权重仍 0）
    ar = torch.arange(N).view(N, 1)
    self_slot = rec_idx == ar
    rec_idx = torch.where(self_slot, (ar + 1) % N, rec_idx)
    rec_w = torch.where(self_slot, torch.zeros_like(rec_w), rec_w)

    n_trunc = int((nz_row > K).sum())
    n_lost = int((nz_row - K).clamp(min=0).sum())
    if verbose:
        print(f"  [Seed7b] 稠密→稀疏转换: N={N} K={K} | 掩码密度 {M_rec.float().mean():.3f}"
              f" | 行非零 mean={nz_row.mean():.1f} max={int(nz_row.max())}"
              f" | 超限行 {n_trunc} 丢边 {n_lost}"
              f"{'（无损）' if n_trunc == 0 else '（近似：top-|W| 截断）'}")

    out = {
        'N': N, 'K': K,
        'M_in': st['M_in'].clone(), 'M_out': st['M_out'].clone(),
        'W_in': st['W_in'].float().clone(), 'W_out': st['W_out'].float().clone(),
        'b_out': st['b_out'].float().clone(),
        'rec_idx': rec_idx, 'rec_w': rec_w,
        'tau_e_init': st['tau_e_init'].float().clone(),
        'w_ei': st['w_ei'].float().clone(), 'w_ie': st['w_ie'].float().clone(),
    }
    return out, float(data.get('food', -1.0)), float(data.get('steps', 0.0))


def load_best_state(path, cfg):
    """读本 run 稀疏格式的最优模型（play/断点兼容；16c 版校验逻辑）。"""
    if not path or not os.path.exists(path):
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
                saved_cfg.get('ACTION_DIM') != cfg.ACTION_DIM):
            print(f"  警告: 模型 {path} 与当前配置不匹配 (N/OBS/ACTION_DIM)")
            return None
        if saved_cfg.get('OBS_ENC_VERSION') != getattr(cfg, 'OBS_ENC_VERSION', ''):
            print(f"  警告: 模型 {path} 观测编码不符")
            return None
        if saved_cfg.get('BRAIN_VERSION') != getattr(cfg, 'BRAIN_VERSION', 'sparse1'):
            print(f"  警告: 模型 {path} 基因组版本不符")
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


def save_history_json(path, history):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False)


def save_checkpoint7(path, cfg, next_gen, pop, history,
                     cum_eval_time, cum_evolve_time,
                     best_state, best_food, best_seen, best_unseen, best_last, best_prox,
                     best_row=None):
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
        'best_row': None if best_row is None else np.asarray(best_row),
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
    if saved_cfg and saved_cfg.get('OBS_ENC_VERSION') != getattr(cfg, 'OBS_ENC_VERSION', ''):
        print(f"  警告: 断点 {path} 观测编码不符，已忽略")
        return None
    if saved_cfg and saved_cfg.get('BRAIN_VERSION') != getattr(cfg, 'BRAIN_VERSION', 'sparse1'):
        print(f"  警告: 断点 {path} 基因组版本不符，已忽略")
        return None
    return data


def load_notify_token():
    """AutoDL 开发者 Token：优先读脚本同目录/工作目录的 notify_token.txt，
    其次环境变量 AUTODL_TOKEN。找不到返回 None（静默跳过通知）。"""
    def _valid(t):
        return bool(t) and t.isascii() and not any(c.isspace() for c in t)
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, 'notify_token.txt'),
              'notify_token.txt'):
        try:
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f:
                    tok = f.read().strip()
                if _valid(tok):
                    return tok
        except Exception:
            pass
    env = os.environ.get('AUTODL_TOKEN', '').strip()
    return env if _valid(env) else None


def send_autodl_notify(cfg, title, content):
    """AutoDL 微信通知。仅完成/异常时调用；任何失败只打印警告。"""
    tok = load_notify_token()
    if not tok:
        print("[通知] 未找到 notify_token.txt / AUTODL_TOKEN，跳过微信通知")
        return
    try:
        import urllib.request
        req = urllib.request.Request(
            'https://www.autodl.com/api/v1/wechat/message/send',
            data=json.dumps({'title': title[:20], 'name': '16c-cheat-7b 通关特训',
                             'content': content[:200]}).encode('utf-8'),
            headers={'Content-Type': 'application/json',
                     'Authorization': tok},
            method='POST')
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode('utf-8', 'replace')
        print(f"[通知] AutoDL 微信通知已发送: {body[:120]}")
    except Exception as e:
        print(f"[通知] 微信通知发送失败（不影响训练）: {e}")


def plot_history_png(cfg, history):
    """训练过程图：左栏食物曲线（含通关线）；右栏当代通关个体数 + best 适应度。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei',
                                              'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    gens = history['gen']
    ax1.plot(gens, history['best_food'], label='Best Food',
             color='red', marker='o', markersize=2)
    ax1.plot(gens, history['avg_food'], label='Avg Food', color='blue', alpha=0.6)
    if any(v is not None for v in history.get('elite_food', [])):
        ax1.plot(gens, history['elite_food'], label='Elite Food', color='green', alpha=0.8)
    ax1.axhline(cfg.TARGET_FOOD, color='gold', lw=1.5, ls='--',
                label=f'WIN = {cfg.TARGET_FOOD}（盘满）')
    ax1.set_title("Fixed-Map Cheat Run — Food Count")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Food Eaten")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    won = history.get('won_count', [])
    fit = history.get('best_fit', [])
    if len(gens) and len(won):
        ax2.plot(gens[:len(won)], won, color='orange', marker='.', label='通关个体数/代')
    if len(gens) and len(fit):
        ax2.plot(gens[:len(fit)], fit, color='black', lw=1.2, ls='--', label='best 适应度')
    ax2.set_title('Win Count / Best Fitness')
    ax2.set_xlabel('Generation')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    hist_path = cfg.CHECKPOINT_PATH.replace('_checkpoint', '_history').replace('.pth', '.png')
    fig.savefig(hist_path, dpi=100)
    plt.close(fig)
    print(f"历史曲线已保存: {hist_path}")


# ==========================================
# 7. 主循环（固定单地图 + 通关即停）
# ==========================================
def run_training(cfg):
    device = _resolve_device(cfg)
    print(f"[GPU] device = {device}  "
          f"({'GPU: ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU 回退'})")
    if device.type == 'cuda':
        mem = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"[GPU] 显存 {mem:.1f} GB, 自动评估批大小 = {_auto_eval_batch(cfg, device)}")
    print(f"[环境] grid={cfg.GRID_SIZE} max_steps={cfg.MAX_STEPS} "
          f"个体步数上限={getattr(cfg, 'EP_STEPS_CAP', 0)} | "
          f"饿死钟 steps_wo_food > {cfg.STARVE_SLOPE}·len+20 | "
          f"疲劳 cts gain={cfg.FATIGUE_GAIN} thresh={cfg.FATIGUE_THRESHOLD} "
          f"max={cfg.FATIGUE_MAX}（7b 口径）| 单侧转弯判死 {cfg.ONE_SIDED_TURN_DEATH}")
    print(f"[种群] pop={cfg.POP_SIZE} 精英={cfg.ELITE_SIZE} 列={cfg.NUM_COLUMNS} "
          f"扇入K={cfg.REC_FANIN} | 轮换 {cfg.CYCLE_PATTERN}")
    g_in, g_out = float(getattr(cfg, 'SENSORY_FRAC', 0)), float(getattr(cfg, 'MOTOR_FRAC', 0))
    print(f"[池约束] 传感池 {g_in:.0%} / 运动池 {g_out:.0%}"
          f"{'（关闭：7b 解剖无损保留）' if g_in == 0 and g_out == 0 else ''}")
    print(f"[稀疏] rec 槽位/体 = {cfg.NUM_COLUMNS}×{cfg.REC_FANIN}"
          f"（对照稠密 {cfg.NUM_COLUMNS ** 2}，利用率 {cfg.REC_FANIN / cfg.NUM_COLUMNS:.2%}）")
    win_desc = (f"通关线 food≥{cfg.TARGET_FOOD} 且通关个体≥{cfg.WIN_FRAC:.0%}"
                if getattr(cfg, 'WIN_FRAC', None)
                else f"通关线 food≥{cfg.TARGET_FOOD}（任一个体）")
    print(f"[cheat] 固定单地图: MAP_SEED={cfg.MAP_SEED} gen={cfg.MAP_GEN} "
          f"(CRN 派生 {_crn_seed(cfg, cfg.MAP_GEN, 1, 0)}) | {win_desc} | "
          f"达标即停={cfg.STOP_ON_WIN}")
    print(f"[适应度 v{cfg.FITNESS_VERSION}] food + {cfg.SIMPLE_EFF_W}·food/steps_last"
          f"（simple，固定地图确定性 → 无 LCB）")
    print(f"[观测] {cfg.OBS_ENC_VERSION}（7b 血统 32 维 8扇区欧氏投影）| fast_eval={cfg.FAST_EVAL}")
    if int(getattr(cfg, 'ROBUST_EVAL', 1)) > 1:
        print(f"[稳健] 评估副本 {cfg.ROBUST_EVAL}（原权重 + {cfg.ROBUST_EVAL - 1} 份 "
              f"σ={cfg.ROBUST_NOISE_STD} 微扰）→ 逐体取最差副本（过滤混沌彩票解）")
    print(f"[输出] ckpt={cfg.CHECKPOINT_PATH} | best={cfg.BEST_MODEL_PATH} | "
          f"win={cfg.WIN_MODEL_PATH} | history={cfg.HISTORY_JSON_PATH}")
    t_program = time.perf_counter()

    start_gen = 0
    pop = GeneStack(cfg, device=device)
    history = {'gen': [], 'best_food': [], 'avg_food': [], 'best_seen': [],
               'best_unseen': [], 'elite_food': [], 'best_fit': [], 'best_turneff': [],
               'best_reach': [], 'won_count': []}
    cum_eval_time = 0.0
    cum_evolve_time = 0.0
    best_state = None
    best_food = -1.0
    best_seen = 0.0
    best_unseen = 0.0
    best_last = 0.0
    best_prox = 0.0
    best_row = np.zeros(12)
    best_row[1] = 99999.0
    latest_gen_best_state = None
    latest_gen_best_food = -1.0
    latest_gen_best_seen = 0.0
    latest_gen_best_unseen = 0.0
    win_announced = False

    if cfg.AUTO_RESUME:
        ck = load_checkpoint7(cfg.CHECKPOINT_PATH, cfg)
        if ck is not None:
            actual_P = int(ck['pop']['M_in'].shape[0])
            if actual_P != cfg.POP_SIZE:
                sys.exit(f'[错误] 断点 {cfg.CHECKPOINT_PATH} 种群 {actual_P} != '
                         f'POP_SIZE {cfg.POP_SIZE}（--pop 须与断点一致，'
                         f'或删除断点/换 --name 重启）')
            start_gen = int(ck['next_gen'])
            pop.unpack(ck['pop'])
            history = ck.get('history', history)
            for k in history.keys():
                history.setdefault(k, [])
            for k in history.keys():
                pad = len(history.get('gen', [])) - len(history[k])
                if pad > 0:
                    history[k].extend([None] * pad)
            cum_eval_time = float(ck.get('cum_eval_time', 0.0))
            cum_evolve_time = float(ck.get('cum_evolve_time', 0.0))
            best_state = ck.get('best_state')
            best_food = float(ck.get('best_food', -1.0))
            best_seen = float(ck.get('best_seen', 0.0))
            best_unseen = float(ck.get('best_unseen', 0.0))
            best_last = float(ck.get('best_last', 0.0))
            best_prox = float(ck.get('best_prox', 0.0))
            best_row = np.zeros(12)
            best_row[1] = 99999.0
            if 'best_row' in ck and ck['best_row'] is not None:
                best_row = np.asarray(ck['best_row'], dtype=np.float64)
            if best_state is not None:
                random.setstate(ck['random_state'])
                torch.set_rng_state(ck['torch_rng_state'])
            # 适应度版本变更（如 v19 彩票口径断点 -> v20 稳健口径）→ 旧 best 行
            # 跨口径不可比，best 追踪重置（种群与历史保留，从本代重新记录）
            ck_ver = int(ck.get('config', {}).get('FITNESS_VERSION', 1))
            if ck_ver != int(getattr(cfg, 'FITNESS_VERSION', 1)):
                best_row = np.zeros(12)
                best_row[1] = 99999.0
                best_state = None
                best_food = -1.0
                best_seen = best_unseen = best_last = best_prox = 0.0
                print(f"  [适应度版本变更 v{ck_ver} -> v{cfg.FITNESS_VERSION}] "
                      f"best 追踪已重置（从本代重新记录）")
            print(f"=== 检测到断点 [{cfg.CHECKPOINT_PATH}] ===")
            print(f"  已完成 {start_gen} 代 -> 从第 {start_gen} 代接续 | "
                  f"历史最优: Food={best_food:.1f} | 已耗时 {cum_eval_time + cum_evolve_time:.1f}s")

    if pop.M_in is None:
        t0 = time.perf_counter()
        print("初始化种群（GPU 随机初始化）...")
        pop.random_init()
        if cfg.SEED_FROM_BEST and cfg.SEED_MODEL_PATH:
            seed = load_best_state_7b_sparse(cfg.SEED_MODEL_PATH, cfg)
            if seed is not None:
                st, s_food, s_steps = seed
                if bool(getattr(cfg, 'SEED_POP', False)):
                    # cheat 模式：全种群克隆注入（围绕 7b best 的变异精修）
                    for i in range(cfg.POP_SIZE):
                        pop.set_individual_from_state(i, st)
                    print(f"  [Seed] 全种群 {cfg.POP_SIZE} 个体已克隆注入 "
                          f"{cfg.SEED_MODEL_PATH} (Food={s_food:.1f})")
                else:
                    pop.set_individual_from_state(0, st)
                    print(f"  [Seed] 已注入 {cfg.SEED_MODEL_PATH} 作为种群种子 "
                          f"(Food={s_food:.1f}, Steps={s_steps:.1f})")
            else:
                print("  [Seed] 种子模型不可用，全新随机初始化")
        else:
            print("  全新随机初始化（SEED_FROM_BEST=False）")
        print(f"  初始化完成 ({time.perf_counter() - t0:.1f}s)")
        save_checkpoint7(cfg.CHECKPOINT_PATH, cfg, 0, pop, history,
                         cum_eval_time, cum_evolve_time,
                         best_state, best_food, best_seen, best_unseen, best_last, best_prox,
                         best_row=best_row)

    # --- 固定单地图（唯一 bank；跨进程确定性 → 断点续训同图）---
    bank = make_bank(cfg, cfg.MAP_GEN, 1, 0, device)
    print(f"[地图] 已生成固定 bank：初始朝向 dir0={bank['dir0']}，"
          f"落子流 {tuple(bank['stream'].shape)}（全体个体共用）")

    try:
        for gen in range(start_gen, cfg.GENERATIONS):
            t_eval = time.perf_counter()
            metrics, order = evaluate_population_fixedmap(pop, cfg, bank)
            eval_time = time.perf_counter() - t_eval
            cum_eval_time += eval_time

            mn = metrics.numpy()
            key_cur = _make_key_fn(cfg)
            best_idx = order[0]
            b_food, b_seen, b_unseen, b_last, b_prox = (
                float(mn[best_idx][0]), float(mn[best_idx][1]),
                float(mn[best_idx][2]), float(mn[best_idx][3]), float(mn[best_idx][4]))
            b_turn = (float(mn[best_idx][8]) + float(mn[best_idx][9])) / max(b_seen + b_unseen, 1.0)
            b_te = (min(b_last / max(float(mn[best_idx][10]), 1.0), 4.0)
                    if (mn[best_idx][10] > 0 and b_food > 0) else
                    (4.0 if b_food > 0 else 0.0))
            b_fit = _fitness_simple(mn[best_idx], cfg)
            avg_food = float(np.mean(mn[:, 0]))
            elite_food = float(np.mean([mn[i][0] for i in order[:cfg.ELITE_SIZE]]))
            won_cnt = int((mn[:, 0] >= cfg.TARGET_FOOD - 1e-9).sum())

            history['gen'].append(gen)
            history['best_food'].append(b_food)
            history['avg_food'].append(avg_food)
            history['best_seen'].append(b_seen)
            history['best_unseen'].append(b_unseen)
            history['elite_food'].append(elite_food)
            history['best_fit'].append(float(b_fit))
            history['best_turneff'].append(float(b_te))
            history['best_reach'].append(float(mn[best_idx][11]))
            history['won_count'].append(won_cnt)

            # best 追踪（确定性评估，直接 base 适应度比较）
            if key_cur(mn[best_idx]) > key_cur(best_row):
                best_food = b_food
                best_seen = b_seen
                best_unseen = b_unseen
                best_last = b_last
                best_prox = b_prox
                best_row = mn[best_idx].copy()
                best_state = pop.individual_state(best_idx, use_half=False)

            latest_gen_best_state = pop.individual_state(best_idx, use_half=False)
            latest_gen_best_food = b_food
            latest_gen_best_seen = b_seen
            latest_gen_best_unseen = b_unseen

            avg_wall = float(np.mean(mn[:, 5]))
            avg_self = float(np.mean(mn[:, 6]))
            avg_starve = float(np.mean(mn[:, 7]))
            if (gen + 1) % cfg.PRINT_HISTORY_EVERY == 0:
                print(f"Gen {gen + 1}/{cfg.GENERATIONS} | "
                      f"BestFood: {b_food:.2f} | BestFit: {b_fit:.3f} | "
                      f"AvgFood: {avg_food:.2f} | EliteFood: {elite_food:.2f} | "
                      f"Won: {won_cnt}({won_cnt / cfg.POP_SIZE:.0%}) | "
                      f"距通关 {max(cfg.TARGET_FOOD - best_food, 0):.1f} | "
                      f"Die(W/S/St): {avg_wall:.2f}/{avg_self:.2f}/{avg_starve:.2f} | "
                      f"eval {eval_time:.1f}s")

            # --- 通关判定：达标即停（含断点续训时历史已达标）---
            hit_win = (won_cnt > 0 or b_food >= cfg.TARGET_FOOD
                       or best_food >= cfg.TARGET_FOOD)
            if cfg.STOP_ON_WIN and hit_win:
                need = 1
                if getattr(cfg, 'WIN_FRAC', None):
                    need = math.ceil(cfg.WIN_FRAC * cfg.POP_SIZE)
                if won_cnt >= need:
                    win_state = best_state if best_state is not None else latest_gen_best_state
                    save_best_model(cfg.WIN_MODEL_PATH, win_state, cfg,
                                    max(best_food, b_food), best_seen, best_unseen)
                    print(f"\n{'=' * 60}\n*** 通关达标！*** 单一地图（MAP_SEED={cfg.MAP_SEED}）"
                          f"@ Gen {gen + 1} | 通关个体 {won_cnt}/{cfg.POP_SIZE}"
                          f"（停机线 {need}）| best_food={max(best_food, b_food):.0f} | "
                          f"累计 {time.perf_counter() - t_program:.1f}s\n{'=' * 60}")
                    print(f"Win 模型已保存: {cfg.WIN_MODEL_PATH}")
                    break
                # 已有通关个体但种群占比未达停机线：win 模型先行落盘（随 best 刷新覆盖）
                if best_food >= cfg.TARGET_FOOD:
                    save_best_model(cfg.WIN_MODEL_PATH, best_state, cfg,
                                    best_food, best_seen, best_unseen)
                    if not win_announced:
                        print(f"  [WIN] 首个通关个体出现（food={best_food:.0f}），win 模型已先行"
                              f"保存；继续向通关占比 {cfg.WIN_FRAC:.0%}（{need} 个体）推进...")
                        win_announced = True

            if gen < cfg.GENERATIONS - 1:
                t_ev = time.perf_counter()
                pop = evolve_topology_gpu(pop, metrics, cfg, gen=gen, order=order)
                evolve_time = time.perf_counter() - t_ev
                cum_evolve_time += evolve_time

            if (gen + 1) % cfg.CHECKPOINT_INTERVAL == 0:
                save_checkpoint7(cfg.CHECKPOINT_PATH, cfg, gen + 1, pop, history,
                                 cum_eval_time, cum_evolve_time,
                                 best_state, best_food, best_seen, best_unseen,
                                 best_last, best_prox, best_row=best_row)
                save_history_json(cfg.HISTORY_JSON_PATH, history)

    except KeyboardInterrupt:
        nxt = gen if 'gen' in dir() else start_gen
        print("\n训练被中断 (Ctrl+C)，正在保存断点...")
        save_checkpoint7(cfg.CHECKPOINT_PATH, cfg, nxt, pop, history,
                         cum_eval_time, cum_evolve_time,
                         best_state, best_food, best_seen, best_unseen,
                         best_last, best_prox, best_row=best_row)
        save_history_json(cfg.HISTORY_JSON_PATH, history)
        print(f"断点已保存: {cfg.CHECKPOINT_PATH} (下次从第 {nxt} 代接续)")
        sys.exit(0)

    t_delta = time.perf_counter() - t_program
    print(f"\nTotal runtime: {t_delta:.1f}s "
          f"(eval {cum_eval_time:.1f}s / evolve {cum_evolve_time:.1f}s)")

    if best_state is None:
        best_idx = 0
        best_state = pop.individual_state(best_idx, use_half=False)
    save_best_model(cfg.BEST_MODEL_PATH, best_state, cfg, best_food, best_seen, best_unseen)
    print(f"\n最优模型已保存: {cfg.BEST_MODEL_PATH} "
          f"(Food={best_food:.2f}, Seen={best_seen:.1f})")

    if latest_gen_best_state is not None:
        save_best_model(cfg.LATEST_GEN_BEST_MODEL_PATH, latest_gen_best_state, cfg,
                        latest_gen_best_food, latest_gen_best_seen, latest_gen_best_unseen)
        print(f"最新一代最优模型已保存: {cfg.LATEST_GEN_BEST_MODEL_PATH} "
              f"(Food={latest_gen_best_food:.2f})")

    # 断点保留（续训需提高 --gens 或 --no-stop-on-win）
    save_checkpoint7(cfg.CHECKPOINT_PATH, cfg, max(cfg.GENERATIONS, start_gen),
                     pop, history, cum_eval_time, cum_evolve_time,
                     best_state, best_food, best_seen, best_unseen,
                     best_last, best_prox, best_row=best_row)
    save_history_json(cfg.HISTORY_JSON_PATH, history)

    try:
        plot_history_png(cfg, history)
    except Exception as e:
        print(f"(matplotlib 曲线跳过: {e})")

    n_gens = len(history.get('gen', []))
    send_autodl_notify(
        cfg, '16c-cheat-7b 训练完成',
        f"gens={n_gens} best_food={best_food:.2f} "
        f"{'WIN!' if best_food >= cfg.TARGET_FOOD else '未通关'} "
        f"用时{t_delta / 3600:.2f}h。产物: {cfg.BEST_MODEL_PATH}")

    print(f"\n--- Best Brain Summary ---")
    st = best_state
    print(f"Input connections active:   {st['M_in'].sum().item()}/{pop.N * pop.O}")
    nz = int((st['rec_w'].float().abs() > 0).sum().item())
    print(f"Internal connections:       {nz}/{pop.K * pop.N} slots non-zero "
          f"(fan-in K={pop.K}, 稠密对照 {pop.N * pop.N})")
    print(f"Output connections active:   {st['M_out'].sum().item()}/{pop.A * pop.N}")
    print(f"tau_e range: [{st['tau_e_init'].min().item():.3f}, {st['tau_e_init'].max().item():.3f}]")


def play_best(cfg, max_steps=None):
    """加载最优模型在固定地图上播放一局（打印 ASCII 棋盘；cheat 语义：地图即训练图）。"""
    res = load_best_state(cfg.BEST_MODEL_PATH, cfg)
    if res is None:
        res = load_best_state(cfg.WIN_MODEL_PATH, cfg)
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
    bank = make_bank(cfg, cfg.MAP_GEN, 1, 0, dev)
    env = BatchedSnakeEnv(cfg, 1, dev)
    env.reset(bank=bank)
    E = torch.zeros(1, pop.N, dtype=pop.dtype, device=dev)
    I = torch.zeros(1, pop.N, dtype=pop.dtype, device=dev)
    stt = torch.zeros(1, pop.N, dtype=pop.dtype, device=dev)
    cts = torch.zeros(1, pop.A, dtype=pop.dtype, device=dev)
    G = cfg.GRID_SIZE
    max_steps = max_steps or 20000

    def render():
        g = [['.' for _ in range(G)] for _ in range(G)]
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
        act, E, I, stt = deliberate_batch(pop, obs, E, I, stt, cts, cfg)
        cts = update_fatigue(cts, act)
        env.step(act)
        if env.ate[0]:
            ep_food += 1
        if s % 200 == 0:
            print(f"\n--- Step {s} (score {ep_food}) ---")
            render()
        if env.all_done():
            print(f"\n--- 终局 @ step {s} | won={bool(env.won[0])} ---")
            render()
            break
    print(f"\nPlay done: Food={ep_food}, Steps={s + 1}, Won={bool(env.won[0])}")


# ==========================================
# 8. 自检
# ==========================================
def _make_test_snake(G, length, rng):
    """生成一条随机自回避蛇身（从头到尾的格序列）；失败重试。rng=np.random.Generator。"""
    for _attempt in range(200):
        head = (int(rng.integers(0, G)), int(rng.integers(0, G)))
        cells = [head]
        ok = True
        while len(cells) < length:
            r, c = cells[-1]
            nbrs = [(r + dr, c + dc) for dr, dc in _DIRS_CPU]
            nbrs = [p for p in nbrs if 0 <= p[0] < G and 0 <= p[1] < G and p not in cells]
            if not nbrs:
                ok = False
                break
            cells.append(nbrs[int(rng.integers(0, len(nbrs)))])
        if ok:
            return cells
    return None


def selfcheck(cfg):
    dev = _resolve_device(cfg)
    ok_all = True

    print("=== 自检 0：7b 食物 8 扇区欧氏投影公式（[8:16]）===")
    sc0 = Config()
    sc0.DEVICE = cfg.DEVICE
    torch.manual_seed(3)
    env0 = BatchedSnakeEnv(sc0, 4, dev)
    env0.reset()
    env0.head = torch.tensor([[5, 5]] * 4, device=dev)
    env0.dir_idx = torch.zeros(4, dtype=torch.long, device=dev)   # 全体朝 E=(0,1)
    # 依次：正前 (0,3) | 右前 (2,1) | 右前对角 (4,4) | 正左 (0,-4)（vr=行差, vc=列差）
    env0.food = torch.tensor([[5, 8], [7, 6], [9, 9], [1, 5]], device=dev)
    env0.fast_obs = False
    obs0 = env0._obs32().float()
    K = float(sc0.OBS_FOOD_SCALE)
    ok0 = True

    def _chk(label, got, want, tol=1e-4):
        nonlocal ok0, ok_all
        good = abs(got - want) < tol
        ok0 = ok0 and good
        ok_all = ok_all and good
        print(f"  {label}: {got:.4f} (期望 {want:.4f}) {'OK' if good else 'FAIL'}")

    _chk("正前 前向信号[8]", float(obs0[0, 8]), K * 3.0 / 9.0)          # dot=3,d2=9
    _chk("右前 前向信号[8]", float(obs0[1, 8]), K * 1.0 / 5.0)          # v=(2,1) 前=(0,1)
    _chk("右前 右向信号[14]", float(obs0[1, 14]), K * 2.0 / 5.0)        # 右=(1,0)
    _chk("右前 右前信号[15]", float(obs0[1, 15]), K * 3.0 / math.sqrt(2) / 5.0)
    _chk("右前 左前信号[9]", float(obs0[1, 9]), 0.0)                    # 负投影截 0
    _chk("右前对角 右向[14]", float(obs0[2, 14]), K * 4.0 / 32.0)       # v=(4,4)
    _chk("右前对角 右前[15]", float(obs0[2, 15]), K * 8.0 / math.sqrt(2) / 32.0)
    _chk("正左 左向信号[10]", float(obs0[3, 10]), K * 4.0 / 16.0)       # v=(0,-4) 左=(-1,0)
    _chk("正左 前无信号[8]", float(obs0[3, 8]), 0.0)
    # 转身不变性：朝 W 时正前(E)食物应转到"后"通道
    env0.dir_idx = torch.full((4,), 2, dtype=torch.long, device=dev)   # 朝 W=(0,-1)
    obs1 = env0._obs32().float()
    _chk("朝W 正前食物→后信号[12]", float(obs1[0, 12]), K * 3.0 / 9.0)
    _chk("朝W 前无信号[8]", float(obs1[0, 8]), 0.0)
    print(f"  投影公式 {'OK' if ok0 else 'FAIL'}")

    print("=== 自检 1：simple 适应度公式与单侧转弯判死 ===")
    ms = np.zeros(12)
    ms[0], ms[3] = 10.0, 100.0
    f = _fitness_simple(ms, cfg)
    want = 10.0 + 0.3 * 0.1
    good = abs(f - want) < 1e-9
    ok_all = ok_all and good
    print(f"  food=10,SL=100: {f:.4f} (期望 {want:.4f}) {'OK' if good else 'FAIL'}")
    ms[0] = 0.0
    good = _fitness_simple(ms, cfg) == 0.0
    ok_all = ok_all and good
    print(f"  food=0 → 0: {'OK' if good else 'FAIL'}")
    ok1c = True
    for a1, a2, dead in ((50.0, 0.0, True), (0.0, 50.0, True),
                         (50.0, 3.0, False), (0.5, 0.0, False), (13.0, 13.0, False)):
        m2 = np.zeros(12)
        m2[0], m2[3], m2[8], m2[9] = 10.0, 100.0, a1, a2
        is_dead = (_fitness_simple(m2, cfg) == -1e9)
        good = (is_dead == dead)
        ok1c = ok1c and good
    ok_all = ok_all and ok1c
    print(f"  单侧转弯判死 5 例: {'OK' if ok1c else 'FAIL'}")

    print("=== 自检 2：变异强度分布 ===")
    s = sample_mut_scale(cfg, 200000, dev).cpu()
    print(f"  中位数 {s.median():.3f} | P5 {s.quantile(0.05):.3f} / P95 {s.quantile(0.95):.3f}"
          f" | clip[{cfg.MUT_SCALE_MIN},{cfg.MUT_SCALE_MAX}]")

    print("=== 自检 3：固定地图 CRN 确定性（同 bank 两跑逐位一致 + 换图变化）===")
    sc = Config()
    sc.POP_SIZE, sc.NUM_COLUMNS, sc.REC_FANIN = 64, 32, 8
    sc.OBS_DIM, sc.ACTION_DIM = 32, 3
    sc.ELITE_SIZE = 16
    sc.MAX_STEPS, sc.EVAL_BATCH, sc.USE_FP16 = 400, 64, cfg.USE_FP16
    sc.DEVICE = cfg.DEVICE
    torch.manual_seed(7)
    pop = GeneStack(sc, device=_resolve_device(sc))
    pop.random_init()
    pop.fp16()
    bank_a = make_bank(sc, sc.MAP_GEN, 1, 0, pop.device)
    m_a, order_a = evaluate_population_fixedmap(pop, sc, bank_a)
    m_b, order_b = evaluate_population_fixedmap(pop, sc, bank_a)
    same = torch.equal(m_a, m_b) and order_a == order_b
    ok_all = ok_all and same
    print(f"  同图两跑 metrics 逐位一致: {torch.equal(m_a, m_b)} | order 一致: "
          f"{order_a == order_b} {'OK' if same else 'FAIL'}")
    sc.MAP_SEED = sc.MAP_SEED + 1
    bank_c = make_bank(sc, sc.MAP_GEN, 1, 0, pop.device)
    m_c, _ = evaluate_population_fixedmap(pop, sc, bank_c)
    diff = float((m_a[:, 0] - m_c[:, 0]).abs().max())
    good = diff > 0
    ok_all = ok_all and good
    print(f"  换 MAP_SEED 后指标改变（应>0）: max|Δfood|={diff:.3f} {'OK' if good else 'FAIL'}")

    print("=== 自检 4：稀疏 rec 前向 vs 稠密参考逐位等价 ===")
    ok4 = True
    sc4 = Config()
    sc4.NUM_COLUMNS, sc4.REC_FANIN = 32, 16
    sc4.OBS_DIM, sc4.ACTION_DIM = 32, 3
    sc4.DEVICE = cfg.DEVICE
    sc4.USE_FP16 = False
    torch.manual_seed(23)
    p4 = GeneStack(sc4, B=3, device=_resolve_device(sc4))
    p4.random_init()
    p4.refresh_eff()
    Wd = torch.zeros(3, 32, 32, device=p4.device, dtype=p4.rec_w.dtype)
    Wd.scatter_add_(2, p4.rec_idx, p4.rec_w)      # 稀疏→稠密参考（重复源=叠加）
    E0 = torch.randn(3, 32, device=p4.device)
    I0 = torch.rand(3, 32, device=p4.device)
    st0 = torch.rand(3, 32, device=p4.device)
    obs0 = torch.rand(3, 32, device=p4.device)
    cts0 = torch.rand(3, 3, device=p4.device) * 6
    lg_s, E_s, I_s, st_s = forward_batch(p4, obs0, E0, I0, st0, cts0, sc4)
    # 稠密参考前向（7b 语义逐行复刻，rec 用 bmm）
    ext = torch.bmm(p4.W_in_eff, obs0.unsqueeze(-1)).squeeze(-1)
    rec_d = torch.bmm(Wd, E0.unsqueeze(-1)).squeeze(-1)
    stt = sc4.SHORT_TERM_DECAY * st0 + (1 - sc4.SHORT_TERM_DECAY) * E0
    tau = (p4.tau_e + sc4.SHORT_TERM_GAIN * stt).clamp(sc4.TAU_E_MIN, sc4.TAU_E_MAX)
    E_r = torch.sigmoid(ext + rec_d + tau * E0 - p4.w_ei.clamp(0, 6) * I0)
    I_r = torch.sigmoid(p4.w_ie.clamp(0, 6) * E_r)
    lg_r = torch.bmm(p4.W_out_eff, E_r.unsqueeze(-1)).squeeze(-1) + p4.b_out
    fat = torch.relu(cts0 - sc4.FATIGUE_THRESHOLD) * sc4.FATIGUE_GAIN
    lg_r = lg_r - fat.clamp(max=sc4.FATIGUE_MAX)
    d_E = float((E_r - E_s).abs().max())
    d_lg = float((lg_r - lg_s).abs().max())
    ok4 = (d_E < 1e-5) and (d_lg < 1e-5)
    ok_all = ok_all and ok4
    print(f"  E_new 差 {d_E:.2e} | logits 差 {d_lg:.2e} {'OK' if ok4 else 'FAIL'}")

    print("=== 自检 5：重连变异后无自连 & 个体状态往返一致 ===")
    ok5 = True
    sc5 = Config()
    sc5.NUM_COLUMNS, sc5.REC_FANIN = 48, 16
    sc5.OBS_DIM, sc5.ACTION_DIM = 32, 3
    sc5.POP_SIZE, sc5.ELITE_SIZE = 16, 4
    sc5.CYCLE_PATTERN = [('G2', 'G1')]
    sc5.TOPOLOGY_MUT_PROB, sc5.MUT_RATE = 1.0, 1.0
    sc5.DEVICE = cfg.DEVICE
    sc5.USE_FP16 = False
    p5 = GeneStack(sc5, device=_resolve_device(sc5))
    p5.random_init()
    fake_metrics = torch.zeros(16, 12)
    p5n = evolve_topology_gpu(p5, fake_metrics, sc5, gen=1)   # gen=1 → G1 活跃
    ar5 = torch.arange(48, device=p5n.device).view(1, 48, 1)
    n_self = int((p5n.rec_idx == ar5).sum().item())
    oob = int(((p5n.rec_idx < 0) | (p5n.rec_idx >= 48)).sum().item())
    good5 = (n_self == 0 and oob == 0)
    ok5 = ok5 and good5
    ok_all = ok_all and good5
    print(f"  全强度重连后: 自连 {n_self} / 越界 {oob}（期望 0/0）"
          f"{'OK' if good5 else 'FAIL'}")
    st_a = p5n.individual_state(0, use_half=False)
    p5b = GeneStack(sc5, B=1, device=_resolve_device(sc5))
    p5b.random_init()
    p5b.set_individual_from_state(0, st_a)
    same = (torch.equal(p5b.rec_idx[0], st_a['rec_idx'].long().to(p5b.device))
            and torch.equal(p5b.rec_w[0], st_a['rec_w'].float().to(p5b.device)))
    ok5 = ok5 and same
    ok_all = ok_all and same
    print(f"  individual_state 往返一致: {same} {'OK' if same else 'FAIL'}")

    print("=== 自检 6：7b 转换器无损性（真实形态：W_rec 全稠密、稀疏性只在 M_rec）===")
    ok6 = True
    N6, K6 = 256, 96
    g6 = torch.Generator().manual_seed(123)
    M6 = (torch.rand(N6, N6, generator=g6) < 0.18)
    M6.fill_diagonal_(False)
    W6 = torch.randn(N6, N6, generator=g6) * 0.1        # 掩码外非零（7b 真实形态）
    nz6 = M6.sum(1)
    cfg6 = Config()
    cfg6.NUM_COLUMNS, cfg6.REC_FANIN = N6, K6
    fake = {
        'brain': {
            'M_rec': M6.to(torch.uint8), 'W_rec': W6,
            'M_in': (torch.rand(N6, 32, generator=g6) < 0.15).to(torch.uint8),
            'M_out': (torch.rand(3, N6, generator=g6) < 0.15).to(torch.uint8),
            'W_in': torch.randn(N6, 32, generator=g6) * 0.1,
            'W_out': torch.randn(3, N6, generator=g6) * 0.1,
            'b_out': torch.zeros(3),
            'tau_e_init': torch.full((N6,), 0.7),
            'w_ei': torch.full((N6,), 2.0), 'w_ie': torch.full((N6,), 2.0),
        },
        'config': {'NUM_COLUMNS': N6, 'OBS_DIM': 32, 'ACTION_DIM': 3},
        'food': 1.0,
    }
    tmp6 = '_tmp_7b_convert_selfcheck.pth'
    torch.save(fake, tmp6)
    try:
        res6 = load_best_state_7b_sparse(tmp6, cfg6, verbose=False)
    finally:
        os.remove(tmp6)
    ok6 = res6 is not None
    if ok6:
        st6 = res6[0]
        Wd6 = torch.zeros(N6, N6)
        Wd6.scatter_add_(1, st6['rec_idx'], st6['rec_w'].float())
        want6 = W6 * M6.float()
        lossless = int((nz6 > K6).sum()) == 0
        exact = torch.equal(Wd6, want6)
        # 幽灵免疫：掩码外重组恒 0
        ghost = float(Wd6[~M6].abs().max())
        ok6 = lossless and exact and ghost == 0.0
        ok_all = ok_all and ok6
        print(f"  无损行 {int((nz6 <= K6).sum())}/{N6} | 重组==W_rec·M_rec: {exact} | "
              f"掩码外幽灵权重 max={ghost:.2e}（期望 0）{'OK' if ok6 else 'FAIL'}")
    else:
        ok_all = False
        print("  转换器调用失败 FAIL")

    # 真实 7b best 模型转换统计（文件存在时）
    if os.path.exists(cfg.SEED_MODEL_PATH):
        res7 = load_best_state_7b_sparse(cfg.SEED_MODEL_PATH, cfg, verbose=False)
        if res7 is not None:
            st7, food7, _ = res7
            Mr7 = (torch.load(cfg.SEED_MODEL_PATH, map_location='cpu',
                              weights_only=False)['brain']['M_rec'] > 0)
            nzr = Mr7.sum(1)
            lossless = int((nzr > cfg.REC_FANIN).sum()) == 0
            ok_all = ok_all and lossless
            print(f"  真实种子 {cfg.SEED_MODEL_PATH} (food={food7}): "
                  f"行非零 max={int(nzr.max())} ≤ K={cfg.REC_FANIN} → 无损="
                  f"{lossless} {'OK' if lossless else 'FAIL'}")
    else:
        print(f"  （跳过真实种子验证：{cfg.SEED_MODEL_PATH} 不存在）")

    print("=== 自检 7：观测对拍（本 env _obs32/_obs32_fast vs test7b._obs32 逐位）===")
    ok7 = True
    try:
        _repo7 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.insert(0, os.path.join(_repo7, 'experiments', 'test7_series'))
        import test7b as t7b
        tcfg = t7b.Config()
        tcfg.DEVICE = 'cpu'
        rng = np.random.default_rng(20260908)
        B7 = 6
        env_me = BatchedSnakeEnv(sc0, B7, torch.device('cpu'))
        env_7b = t7b.BatchedSnakeEnv(tcfg, B7, torch.device('cpu'))
        # 构造合法自回避蛇（不同长度/朝向），两 env 同状态
        bodies, lens, heads, dirs_, foods = [], [], [], [], []
        for i in range(B7):
            length = int(rng.integers(3, 30))
            cells = _make_test_snake(10, length, rng)
            d_i = int(rng.integers(0, 4))
            dvec = _DIRS_CPU[d_i]
            head = cells[0]
            neck = (head[0] - dvec[0], head[1] - dvec[1])
            if neck in cells[1:]:
                cells = [head, neck] + [p for p in cells[1:] if p != neck][:length - 2]
            bodies.append(cells[:length]); lens.append(len(cells[:length]))
            heads.append(head); dirs_.append(d_i)
            free = [(r, c) for r in range(10) for c in range(10)
                    if (r, c) not in cells[:length]]
            foods.append(free[int(rng.integers(len(free)))])
        maxlen = env_me.MAXLEN
        body_t = torch.zeros(B7, maxlen, 2, dtype=torch.long)
        for i, cells in enumerate(bodies):
            for j, (r, c) in enumerate(cells):
                body_t[i, j] = torch.tensor([r, c])
        for env_x in (env_me, env_7b):
            env_x.body = body_t.clone()
            env_x.body_len = torch.tensor(lens, dtype=torch.long)
            env_x.head = torch.tensor(heads, dtype=torch.long)
            env_x.dir_idx = torch.tensor(dirs_, dtype=torch.long)
            env_x.food = torch.tensor(foods, dtype=torch.long)
        obs_7b = env_7b.obs().float()
        env_me.fast_obs = False
        o_scalar = env_me._obs32().float()
        env_me.fast_obs = True
        o_fast = env_me._obs32_fast().float()
        d_sc = float((o_scalar - obs_7b).abs().max())
        d_fa = float((o_fast - obs_7b).abs().max())
        ok7 = (d_sc == 0.0) and (d_fa == 0.0)
        # 四朝向遍历（同蛇换朝向）
        ok_dir = True
        for di in range(4):
            for env_x in (env_me, env_7b):
                env_x.dir_idx = torch.full((B7,), di, dtype=torch.long)
            env_me.fast_obs = False
            a_ = env_me._obs32().float()
            env_me.fast_obs = True
            b_ = env_me._obs32_fast().float()
            t_ = env_7b.obs().float()
            if float((a_ - t_).abs().max()) != 0.0 or float((b_ - t_).abs().max()) != 0.0:
                ok_dir = False
        ok7 = ok7 and ok_dir
        ok_all = ok_all and ok7
        print(f"  随机蛇 6 条: 标量 max|Δ|={d_sc:.2e} | fast max|Δ|={d_fa:.2e} | "
              f"四朝向遍历: {'OK' if ok_dir else 'FAIL'} {'OK' if ok7 else 'FAIL'}")
    except ImportError as e:
        print(f"  （SKIP：无法 import test7b —— {e}）")

    print("=== 自检 8：7b best 转换后本引擎固定地图行为对拍（端到端）===")
    if os.path.exists(cfg.SEED_MODEL_PATH):
        res8 = load_best_state_7b_sparse(cfg.SEED_MODEL_PATH, cfg, verbose=False)
        if res8 is not None:
            st8, food8, _ = res8
            sc8 = Config()
            sc8.DEVICE = cfg.DEVICE
            sc8.POP_SIZE = 8
            sc8.MAX_STEPS = 20000          # 对拍上限（best 正常几千步内终局）
            sc8.SEED_FROM_BEST = False
            pop8 = GeneStack(sc8, B=8, device=_resolve_device(sc8))
            pop8.random_init()
            for i in range(8):
                pop8.set_individual_from_state(i, st8)
            bank8 = make_bank(sc8, sc8.MAP_GEN, 1, 0, pop8.device)
            m8, _ = evaluate_population_fixedmap(pop8, sc8, bank8)
            fmean = float(m8[:, 0].mean())
            # 参照：7b 千局基准 mean 61.35（不同图单局有方差）。若观测/前向/动力学
            # 任一不等价，随机种子网络水平（<5 食）——≥20 即行为保真强证据。
            good8 = fmean >= 20.0
            ok_all = ok_all and good8
            print(f"  克隆 8 局均值 food={fmean:.1f}（7b 千局基准 61.35；种子 saved "
                  f"{food8}）{'OK' if good8 else 'FAIL'}")
    else:
        print(f"  （SKIP：{cfg.SEED_MODEL_PATH} 不存在）")

    print("=== 自检 9：won 通关终局（构造盘满场景）===")
    ok9 = True
    sc9 = Config()
    sc9.DEVICE = cfg.DEVICE
    env9 = BatchedSnakeEnv(sc9, 2, dev)
    env9.reset()
    # 局0：蛇长 99、头 (0,0) 朝 E，食物 (0,1)（刻意留空）→ 吃下即 body_len=100 → won；
    # 局1：2 节蛇吃一颗 → len=3，不应 won。
    G9 = env9.G
    free99 = [(r, c) for r in range(G9) for c in range(G9)
              if (r, c) not in ((0, 0), (0, 1))][:98]
    body99 = [(0, 0)] + free99                      # 99 格，(0,1) 为唯一空邻格
    body_t9 = torch.zeros(2, env9.MAXLEN, 2, dtype=torch.long, device=dev)
    for j, (r, c) in enumerate(body99):
        body_t9[0, j] = torch.tensor([r, c])
    body_t9[1, 0] = torch.tensor([5, 5], device=dev)
    body_t9[1, 1] = torch.tensor([5, 4], device=dev)
    env9.body = body_t9
    env9.body_len = torch.tensor([99, 2], device=dev)
    env9.head = torch.tensor([[0, 0], [5, 5]], device=dev)
    env9.dir_idx = torch.tensor([0, 0], device=dev)            # 双双朝 E
    env9.food = torch.tensor([[0, 1], [5, 6]], device=dev)     # 各自头正前
    env9.alive = torch.ones(2, dtype=torch.bool, device=dev)
    env9.won = torch.zeros(2, dtype=torch.bool, device=dev)
    env9.step(torch.tensor([0, 0], device=dev))                # 双双直行吃食
    won0 = bool(env9.won[0])
    len0 = int(env9.body_len[0])
    dead0 = not bool(env9.alive[0])
    good9 = won0 and len0 == 100 and dead0
    ok9 = ok9 and good9
    print(f"  99+1食: won={won0} len={len0} alive={not dead0}（期望 True/100/False）"
          f"{'OK' if good9 else 'FAIL'}")
    good9b = (not bool(env9.won[1])) and int(env9.body_len[1]) == 3
    ok9 = ok9 and good9b
    ok_all = ok_all and ok9
    print(f"  普通吃食(len3): won={bool(env9.won[1])}（期望 False）"
          f"{'OK' if good9b else 'FAIL'}")
    print(f"=== 自检完成 {'（全部 OK）' if ok_all else '（存在 FAIL）'} ===")
    return ok_all


# ==========================================
# 9. 入口
# ==========================================
def make_smoke_config():
    cfg = Config()
    cfg.POP_SIZE = 16
    cfg.NUM_COLUMNS = 24
    cfg.REC_FANIN = 8
    cfg.OBS_DIM = 32
    cfg.ACTION_DIM = 3
    cfg.GENERATIONS = 3
    cfg.ELITE_SIZE = 6
    cfg.MAX_STEPS = 60
    cfg.FRAME_RATE = 2
    cfg.CHECKPOINT_INTERVAL = 2
    cfg.CHECKPOINT_PATH = '16c_cheat7b_smoke_checkpoint.pth'
    cfg.BEST_MODEL_PATH = '16c_cheat7b_smoke_best.pth'
    cfg.LATEST_GEN_BEST_MODEL_PATH = '16c_cheat7b_smoke_latest_gen_best.pth'
    cfg.WIN_MODEL_PATH = '16c_cheat7b_smoke_win_model.pth'
    cfg.HISTORY_JSON_PATH = '16c_cheat7b_smoke_history.json'
    cfg.SEED_FROM_BEST = False           # smoke 不依赖外部种子文件
    cfg.STOP_ON_WIN = False
    cfg.EVAL_BATCH = 16
    cfg.PRINT_HISTORY_EVERY = 1
    return cfg


def main():
    ap = argparse.ArgumentParser(
        description='test16c_cheat7b — 固定种子×固定单地图×7b best 全种群克隆通关特训')
    ap.add_argument('--smoke', action='store_true', help='小规模快速自检')
    ap.add_argument('--selfcheck', action='store_true',
                    help='只跑自检（适应度/CRN 确定性/稀疏等价/7b 转换与观测对拍），不训练')
    ap.add_argument('--fanin', type=int, default=None,
                    help='每柱循环输入槽位数 K（默认 96，须 ≤ N-1）')
    ap.add_argument('--elite', type=int, default=None, help='亲本数（默认 POP/4）')
    ap.add_argument('--gens', type=int, default=None)
    ap.add_argument('--pop', type=int, default=None)
    ap.add_argument('--columns', type=int, default=None, help='柱数 N（默认 256=7b 血统）')
    ap.add_argument('--max-steps', type=int, default=None)
    ap.add_argument('--ep-steps-cap', type=int, default=None,
                    help='个体级步数硬上限（默认 3000；0=关。防磨蹭个体拖垮整代评估）')
    ap.add_argument('--eval-workers', type=int, default=None,
                    help='EVAL_SOLO 的并行 worker 子进程数（默认 8；1=主进程内串行）')
    ap.add_argument('--eval-worker', type=str, default=None, help=argparse.SUPPRESS)
    ap.add_argument('--eval-worker-out', type=str, default=None, help=argparse.SUPPRESS)
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--fit-mode', type=str, default=None, choices=['simple', 'econ'],
                    help='simple=food+k·eff（默认）| econ=v7 乘法式')
    ap.add_argument('--simple-eff-w', type=float, default=None,
                    help='simple 模式效率系数 k（默认 0.3）')
    ap.add_argument('--no-one-sided-death', action='store_true',
                    help='关闭单侧转弯判死（默认开启）')
    ap.add_argument('--starve-slope', type=float, default=None,
                    help='饿死斜率（默认 3.0=7b 口径）')
    ap.add_argument('--eval-batch', type=int, default=None)
    ap.add_argument('--no-fast-eval', dest='fast_eval', action='store_false',
                    help='关闭 fast-eval（标量观测慢路径）')
    ap.add_argument('--seed-model', type=str, default=None,
                    help='7b 稠密基因组模型路径（默认 artifacts/test7b/test7b_latest_gen_best.pth）')
    ap.add_argument('--no-seed-pop', dest='seed_pop', action='store_false',
                    help='关闭全种群克隆（改单种子注入随机种群）')
    ap.add_argument('--map-seed', type=int, default=None,
                    help='固定地图种子（默认 20260908；换图=换种子）')
    ap.add_argument('--target-food', type=int, default=None,
                    help='通关目标食物数（默认 98=盘满）')
    ap.add_argument('--no-stop-on-win', dest='stop_on_win', action='store_false',
                    help='通关后不停止（继续训练到 --gens）')
    ap.add_argument('--win-frac', type=float, default=None,
                    help='停机线：通关个体占比（如 0.95=95%% 个体通关才停；默认任一通关即停）')
    ap.add_argument('--robust-eval', type=int, default=None,
                    help='稳健化评估副本数（默认 1=关；建议 2=原权重+微扰副本取最差，过滤混沌彩票解）')
    ap.add_argument('--robust-noise', type=float, default=None,
                    help='微扰副本的权重噪声 std（默认 1e-3）')
    ap.add_argument('--crn-seed', type=int, default=None, help='（兼容名）= --map-seed')
    ap.add_argument('--sensory-frac', type=float, default=None,
                    help='传感池占比（默认 0=关；16c 池约束开关）')
    ap.add_argument('--motor-frac', type=float, default=None,
                    help='运动池占比（默认 0=关）')
    ap.add_argument('--mut-sigma', type=float, default=None,
                    help='变异强度分布弥散（lognormal σ_ln，默认 0.5）')
    ap.add_argument('--mut-dist', type=str, default=None, choices=['lognormal', 'normal'])
    ap.add_argument('--w-mut-std', type=float, default=None,
                    help='权重变异 std（默认 0.1；盆地尺度精修建议 0.002≈稳健微扰同级）')
    ap.add_argument('--w-mut-frac', type=float, default=None,
                    help='权重变异触发比例（默认 0.2；精修建议 0.1）')
    ap.add_argument('--dyn-mut-std', type=float, default=None,
                    help='动力学组变异 std（tau_e/w_ei/w_ie，默认 0.05/0.1/0.1；精修建议 0.002）')
    ap.add_argument('--mask-mut-rate', type=float, default=None,
                    help='M_in/M_out 翻转率（默认 0.05；精修建议 0.01）')
    ap.add_argument('--topo-mut-prob', type=float, default=None,
                    help='拓扑变异概率（rec_idx 重连分支，默认 0.05；精修建议 0.005）')
    ap.add_argument('--seed', type=int, default=20260908,
                    help='固定 CPU+CUDA 随机种子（默认 20260908）')
    ap.add_argument('--resume-pop', type=str, default=None,
                    help='从本血统全种群 checkpoint 导入并续训（复制为断点后 AUTO_RESUME）')
    ap.add_argument('--name', type=str, default='16c_cheat7b',
                    help='本 run 文件名前缀（默认 16c_cheat7b）')
    ap.add_argument('--play', action='store_true',
                    help='固定地图回放最优模型（步数上限配 --play-steps）')
    ap.add_argument('--play-steps', type=int, default=None)
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
    if args.fanin:
        cfg.REC_FANIN = args.fanin
    if args.elite:
        cfg.ELITE_SIZE = args.elite
    if args.max_steps:
        cfg.MAX_STEPS = args.max_steps
    if args.ep_steps_cap is not None:
        cfg.EP_STEPS_CAP = args.ep_steps_cap
    if args.eval_workers is not None:
        cfg.EVAL_WORKERS = args.eval_workers
    if args.eval_worker:
        eval_worker_main(args.eval_worker, args.eval_worker_out)
        return
    if args.device:
        cfg.DEVICE = args.device
    if args.fit_mode:
        cfg.FIT_MODE = args.fit_mode
    if args.simple_eff_w is not None:
        cfg.SIMPLE_EFF_W = args.simple_eff_w
    if args.no_one_sided_death:
        cfg.ONE_SIDED_TURN_DEATH = False
    if args.starve_slope is not None:
        cfg.STARVE_SLOPE = args.starve_slope
    if args.eval_batch is not None:
        cfg.EVAL_BATCH = args.eval_batch
    if not args.fast_eval:
        cfg.FAST_EVAL = False
        cfg.EVAL_MEM_FRAC = 0.55
    if args.seed_model:
        cfg.SEED_FROM_BEST = True
        cfg.SEED_MODEL_PATH = args.seed_model
    cfg.SEED_POP = bool(args.seed_pop)
    if args.map_seed is not None:
        cfg.MAP_SEED = args.map_seed
    if args.crn_seed is not None:
        cfg.MAP_SEED = args.crn_seed
    if args.target_food is not None:
        cfg.TARGET_FOOD = args.target_food
    if not args.stop_on_win:
        cfg.STOP_ON_WIN = False
    if args.win_frac is not None:
        cfg.WIN_FRAC = args.win_frac
    if args.robust_eval is not None:
        cfg.ROBUST_EVAL = max(1, args.robust_eval)
    if args.robust_noise is not None:
        cfg.ROBUST_NOISE_STD = args.robust_noise
    if args.sensory_frac is not None:
        cfg.SENSORY_FRAC = args.sensory_frac
    if args.motor_frac is not None:
        cfg.MOTOR_FRAC = args.motor_frac
    if args.mut_sigma is not None:
        cfg.MUT_SCALE_SIGMA = args.mut_sigma
    if args.mut_dist:
        cfg.MUT_SCALE_DIST = args.mut_dist
    if args.w_mut_std is not None:
        cfg.WEIGHT_MUT_STD = args.w_mut_std
    if args.w_mut_frac is not None:
        cfg.WEIGHT_MUT_FRAC = args.w_mut_frac
    if args.dyn_mut_std is not None:
        cfg.TAU_E_MUT_STD = args.dyn_mut_std
        cfg.W_EI_MUT_STD = args.dyn_mut_std
        cfg.W_IE_MUT_STD = args.dyn_mut_std
    if args.mask_mut_rate is not None:
        cfg.MUT_RATE = args.mask_mut_rate
    if args.topo_mut_prob is not None:
        cfg.TOPOLOGY_MUT_PROB = args.topo_mut_prob
    if args.resume_pop:
        payload = torch.load(args.resume_pop, map_location='cpu', weights_only=False)
        saved = payload.get('config', {})
        if saved and (saved.get('NUM_COLUMNS') != cfg.NUM_COLUMNS or
                      saved.get('OBS_DIM') != cfg.OBS_DIM or
                      saved.get('ACTION_DIM') != cfg.ACTION_DIM):
            sys.exit(f'[错误] resume-pop {args.resume_pop} 与当前维度不匹配')
        if saved and saved.get('BRAIN_VERSION') != getattr(cfg, 'BRAIN_VERSION', 'sparse1'):
            sys.exit(f'[错误] resume-pop {args.resume_pop} 基因组版本不符')
        if saved and saved.get('OBS_ENC_VERSION') != getattr(cfg, 'OBS_ENC_VERSION', ''):
            sys.exit(f'[错误] resume-pop {args.resume_pop} 观测编码不符')
        torch.save(payload, cfg.CHECKPOINT_PATH)
        print(f"[resume-pop] 已导入 {args.resume_pop} "
              f"(next_gen={payload.get('next_gen')}) -> {cfg.CHECKPOINT_PATH}")

    # --- 文件名前缀（smoke 保留自己的前缀）---
    if not args.smoke:
        cfg.CHECKPOINT_PATH = f'{args.name}_checkpoint.pth'
        cfg.BEST_MODEL_PATH = f'{args.name}_best_model.pth'
        cfg.LATEST_GEN_BEST_MODEL_PATH = f'{args.name}_latest_gen_best.pth'
        cfg.WIN_MODEL_PATH = f'{args.name}_win_model.pth'
        cfg.HISTORY_JSON_PATH = f'{args.name}_history.json'

    # --- 校验 ---
    if cfg.REC_FANIN >= cfg.NUM_COLUMNS:
        sys.exit(f'[配置错误] REC_FANIN({cfg.REC_FANIN}) 须 < NUM_COLUMNS({cfg.NUM_COLUMNS})')
    if cfg.ELITE_SIZE >= cfg.POP_SIZE:
        sys.exit(f'[配置错误] ELITE_SIZE({cfg.ELITE_SIZE}) 须 < POP_SIZE({cfg.POP_SIZE})')
    if cfg.TARGET_FOOD > cfg.GRID_SIZE * cfg.GRID_SIZE - 2:
        sys.exit(f'[配置错误] TARGET_FOOD({cfg.TARGET_FOOD}) 超过盘满上限 '
                 f'{cfg.GRID_SIZE * cfg.GRID_SIZE - 2}')

    # --- 固定随机种子（cheat 可复现性核心）---
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"[seed] CPU+CUDA 随机种子已固定 = {args.seed}")

    if args.selfcheck:
        selfcheck(cfg)
        return
    if args.play:
        play_best(cfg, max_steps=args.play_steps)
        return

    try:
        run_training(cfg)
    except Exception as e:
        send_autodl_notify(cfg, '16c-cheat-7b 训练异常退出',
                           f'{type(e).__name__}: {str(e)[:150]}')
        raise


if __name__ == '__main__':
    main()
