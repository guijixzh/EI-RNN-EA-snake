# ==========================================
# test16a.py —— 放宽选择 1/2 + LCB 选择键 + 自适应 K2（抗噪声卡上限；test16 子版本）
#
# 动机（理论估计，详见对话记录）：亲本集合从 1/8（256）放宽到 1/2（1024）：
#   - 边界裁决移出高分噪声稠密区：1/8 时 rank-256 边界（分数~12）±0.4 食内
#     约 25 个真精英的亲本身份每代由噪声掷硬币；1/2 把边界移到中位区
#     （分数~3-4，绝对噪声减半、误杀代价趋零）。
#   - 漂移 ΔF 0.195%/代 → 0.049%/代（×4）；方差耗竭放缓，σ_f 与 ρ 长期更高。
#   - 代价（已知并接受）：i 1.65→0.80（−52%）、每代新变异体 1792→1024（−43%），
#     短期 BestFood 斜率可能放缓。
#   - 冠军精修地板 1.7σ_ε(K) 不随 p 变（benchmark 标定 cv≈11.7%），必须配
#     自适应 K2：δ=1.5 食分辨率、K2_MAX=32 → 地板从 3.1%·score 压到 ~2.0%·score。
#
# 三项改动（基因组/环境/CRN 与 test16 逐字一致，BRAIN_VERSION 仍为 'sparse1'，
# 可用 --resume-pop test16_checkpoint.pth 直接导入 test16 种群）：
#   1. 选择放宽：ELITE_SIZE=1024=STAGE2_KEEP（幸存者即亲本）。评估预算在基础
#      档与 test16 完全相同（2048×4 + 1024×10）——只改"谁进集合"，零额外成本。
#   2. LCB 选择键：sel_key = fitness − λ·SEL_CV·max(food,1)/√K（λ=1, cv=0.12）。
#      同 K 集合内排序不变（等比平移）；实际作用 = 跨代 best 追踪防侥幸
#      （自适应 K2 后各代 K 不同，低 K 代的侥幸高分不再锁定 best 模型）。
#      FITNESS_VERSION 7→8：导入 test16 断点时自动重置 best 追踪（预期行为）。
#   3. 自适应 K2：K2 = clip(⌈(SEL_CV·S/δ)²⌉ − K1, 10, K2_MAX)，S = 历史
#      BestFood 运行最大值，δ=RES_TARGET=1.5。S≤40 时维持基础档 10，
#      S=60 → 18，S=100 → 32（封顶）。history 新增 'k2' 键。
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
    # --- 进化参数（与 test5d/7 系一致）---
    POP_SIZE = 4096 #默认2048
    GENERATIONS = 320
    ELITE_SIZE = 1024           # test16a：亲本=1/2（放宽选择，抗噪声卡上限；test16 为 256）
    MUT_RATE = 0.05
    TOPOLOGY_MUT_PROB = 0.05
    WEIGHT_MUT_FRAC = 0.2
    WEIGHT_MUT_STD = 0.1
    TAU_E_MUT_STD = 0.05
    EVO_COS_MODE = 'anneal'
    EVO_COS_PERIOD = 100
    EVO_DYN_DECAY_TAU = 33

    # --- 类正态变异强度（每子代因子 s，缩放其全部变异算子）---
    MUT_SCALE_DIST = 'lognormal'   # 'lognormal' | 'normal'
    MUT_SCALE_SIGMA = 0.5          # lognormal: σ_ln；normal: s~N(1,σ) clip
    MUT_SCALE_MIN = 0.25
    MUT_SCALE_MAX = 4.0

    # --- 无激素 EI-RNN（兼容字段）---
    TRAIN_HORMONE_NET = False
    HORMONE_NET_HIDDEN = 32

    # --- 环境参数 ---
    GRID_SIZE = 10
    EVAL_EPISODES = 24          # 阶段2 局数 K2（累计 K1+K2 定精英）
    MAX_STEPS = 100000

    # --- 脑结构参数（test16：稀疏固定扇入）---
    NUM_COLUMNS = 1024
    REC_FANIN = 16              # 每突触后神经元输入槽数 K（≤ N-1）
    OBS_MODE = '32proj'
    OBS_DIM = 24 if OBS_MODE == '24' else 40    # test15 血统：32 旧通道 + 8 新通道
    ACTION_DIM = 3
    INIT_DENSITY = 0.15         # 仅用于 M_in/M_out（rec 由 REC_FANIN 决定）

    # --- 适应度模式与版本 ---
    FIT_MODE = 'econ'           # 'econ'=v7 乘法式 | 'simple'=食物+k·效率最简回退
                                # | 'tuple'=元组字典序（旧）
    SIMPLE_EFF_W = 0.3          # simple 模式的效率系数 k（test7b 口径）
    FITNESS_VERSION = 8         # v8 = v7 适应度 + LCB 选择惩罚（仅选择口径；
                                # 报告/history 口径仍是 v7 base）。导入 v7 断点
                                # 时 best 追踪自动重置（预期行为）。

    # --- test16a：LCB 选择键 + 自适应 K2（抗噪声卡上限）---
    SEL_LCB_LAMBDA = 1.0        # 选择键 = fitness − λ·SEL_CV·max(food,1)/√K；0=关
    SEL_CV = 0.12               # 单局分数变异系数（benchmark 标定 11.6%~13.2%）
    RES_TARGET = 1.5            # 目标分辨率 δ（食）：σ_ε(K1+K2) ≤ δ 触发 K2 加密
    K2_MAX = 128                # 自适应 K2 上限（云端档；base>cap 时 cap 自动抬到 base）
    K2_ADAPTIVE = True          # False = 固定 K2=EVAL_EPISODES

    # --- 适应度 v7 因子（乘法式）：food×(1+W_EFF·eff+W_STRAIGHT·straight+W_EDGE·edge_share)×conn ---
    # 括号内各项∈[0,1]、因子非负 → 括号≥1，fitness≥food×conn。
    # 因子随食物等比放大：高食段习惯折损即大额扣分，可扭转后期坏行为。
    W_EFF = 0.3                 # 效率因子（eff=food/steps_last）
                                # 校准：实测 eff≈0.05~0.25（16~4 步/食），因子 4
                                # 使典型效率差 0.05→0.2 ≈ 0.6 分摆幅，与
                                # straight/edge 满幅 0.5 同量级（0.3 时项不可见）
    W_STRAIGHT = 0.01           # 少转弯因子（straight=1−转弯/步）
    W_EDGE = 0.0                # 边角因子（edge_share=圈层总分/(16×蛇长)）

    # --- 观测编码（checkpoint 校验用；改编码/参照系须换新 run）---
    OBS_ENC_VERSION = '40tailflood1'  # 32ego1 + 钟压/尾四方位/三向7步洪水稀缺
    OBS_FOOD_FRAME = 'ego'      # 'ego'=前/右/后/左 | 'abs'=绝对系（实测选择无梯度，勿用于从零训练）

    # --- test15 新通道（[32:40]，详见文件头）---
    OBS_NEW_SCALE = 8.0         # 新通道幅度（仓库标准 K=8）
    OBS_NEW_ENABLED = True      # 总闸：False=新通道置零（A0 对照，与 test12 同数学）
    FLOOD_TAIL_BLOCK = True     # 洪水把尾格视为占用（保守报警）；False=尾可走
    FLOOD_DEPTH = 7             # 洪水 BFS 深度
    BRAIN_VERSION = 'sparse1'   # 基因组版本守卫（稀疏固定扇入基因组，与稠密互斥）

    # --- 单侧转弯判死：只朝一个方向转的蛇评估期判死（淘汰单向绕圈形态）---
    ONE_SIDED_TURN_DEATH = True

    # --- 饿死斜率：steps_wo_food > STARVE_SLOPE*len + 20 ---
    STARVE_SLOPE = 5.0

    # --- 观测：曼哈顿度量固定 ---
    OBS_MANHATTAN = True
    OBS_FOOD_SCALE = 8.0
    OBS_SELF_SCALE = 8.0
    OBS_OBSTACLE_SCALE = 8.0

    # --- 单侧转弯判死（test12 默认开启）：只朝一个方向转的蛇判死 ---
    # 关闭理由（test7d）不再成立：随机初期"摇头"个体两向都转、不受影响；
    # 而单向绕圈蛇在食物 ~28 后自困回环必死（瓶颈实测），须在评估期淘汰。
    ONE_SIDED_TURN_DEATH = True

    # --- 饿死斜率：steps_wo_food > STARVE_SLOPE*len + 20 ---
    STARVE_SLOPE = 5.0

    # --- 观测：曼哈顿度量固定（欧氏对角偏置已验证为错误启发）---
    OBS_MANHATTAN = True
    OBS_FOOD_SCALE = 8.0
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

    # --- 转向疲劳（彻底关闭：0.0；留开关便于回溯）---
    FATIGUE_TURN_GAIN = 0.0
    FATIGUE_TURN_DECAY = 0.9

    # --- K 倍帧率思考 ---
    FRAME_RATE = 5
    INPUT_DECAY = 0.9

    # --- 进化筛选策略 ---
    LONG_SNAKE_SCORE_THRESHOLD = 3.0
    # test16 修复：旧 ('G2','G1','G3') 
    # 单亲遗传（见 evolve_topology_gpu）彻底杜绝。
    CYCLE_PATTERN = [('G2', 'G1')]

    # --- CRN 公共随机数（筛选种子序列固定）---
    USE_CRN = True
    CRN_SEED = 20260827
    CRN_DRAW = 4096             # 每局预生成落子候选流长度

    # --- 两阶段淘汰 ---
    STAGE1_EPS = 12              # 阶段1 局数 K1（全种群，CRN 同库）
    STAGE2_KEEP = 2048          # 阶段1 后幸存数（须 ≥ ELITE_SIZE）
    STAGE2_EPS = 24             # 阶段2 局数 K2 的权威口径（main 会同步进
                                # EVAL_EPISODES；旧代码此参数是死的，改它无效——已修）

    # --- GPU 并行参数 ---
    DEVICE = 'auto'
    USE_FP16 = True
    EVAL_BATCH = 0
    EVAL_MEM_FRAC = 0.55

    # --- 输出 ---
    PRINT_HISTORY_EVERY = 1

    # --- 断点 / 最优模型 / 种子 ---
    CHECKPOINT_PATH = 'test16a_checkpoint.pth'
    BEST_MODEL_PATH = 'test16a_best_model.pth'
    LATEST_GEN_BEST_MODEL_PATH = 'test16a_latest_gen_best.pth'
    HISTORY_JSON_PATH = 'test16a_history.json'
    AUTO_RESUME = True
    CHECKPOINT_INTERVAL = 10
    SEED_FROM_BEST = False
    SEED_MODEL_PATH = ''
    SEED_MODEL_PATH2 = ''
    SEED_POP = False             # 全种群注入种子（血统迁移，--migrate-from）
    SEED_MIGRATE = False         # 迁移模式：旧 32 维单脑 → 列扩展变换后注入

    # ==========================================
    # 旧参数归档（当前不参与适应度/默认关闭；保留开关便于回溯）
    # ==========================================
    ISLAND_THRESHOLD = 0.1      # 孤岛诊断阈值（metrics 列13/14 仅诊断，曾进适应度已废）
    WEAK_MASK_FRAC = 0.0        # 评估期弱连接屏蔽（0=关；7h 期 0.2 实测 +1.8 分）
    FATIGUE_TURN_GAIN = 0.0     # 转向疲劳（彻底关闭，留开关）
    TURN_EFF_W = 3.0            # te 遥测/te-配额参数（v5 起休眠）
    TURN_EFF_CAP = 4.0
    TURN_EFF_MODE = 'ratio'     # 'ratio'=SL/TL | 'tpf'=每食物转弯数
    TE_ELITE = 0                # te-配额精英（0=关）
    IMITATION_W = 0.0           # 模仿引导（教师全局输入，行为不可比；0=关）
    HABIT_W = 0.0               # v6 加法习惯项（已由 v7 乘法式取代）


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
    """适应度 v7（乘法式，test12）：
    fitness = food × (1 + W_EFF·eff + W_STRAIGHT·straight + W_EDGE·edge_share) × conn
    - eff=food/steps_last；straight=1−转弯/步（列17）；edge_share=圈层总分
      /(16×蛇长)（列18）；conn=连通块数倒数（列16，单连通=1，块越多折扣越大，
      严格含尾占用）。
    - 乘法意图：因子随食物等比放大。加法 tie-breaker（v6 预算 <1 食）实测
      撬不动后期坏行为瓶颈；乘法在 50+ 食段 20% 的习惯折损即 10+ 分。
    - 括号内各项∈[0,1]、因子非负 → 括号≥1，fitness ≥ food×conn ≥ 0。
    - 单侧转弯判死保留（评估期淘汰形态，属判死规则非适应度项）。
    """
    if m[1] >= 99999:
        return -1e9
    # 单侧转弯判死（ONE_SIDED_TURN_DEATH，规则同 test7d/e）：整段评估只朝
    # 一个方向转（另一方向 0 次、单侧平均 >1 次/局）→ 判死。
    if getattr(cfg, 'ONE_SIDED_TURN_DEATH', False) and len(m) > 9:
        a1, a2 = float(m[8]), float(m[9])
        if (a1 > 1.0 and a2 == 0.0) or (a2 > 1.0 and a1 == 0.0):
            return -1e9
    food, steps_last = m[0], m[3]
    if food <= 0:
        return 0.0
    eff = food / max(steps_last, 1.0)
    straight = float(m[17]) if len(m) > 17 else 1.0
    edge = float(m[18]) if len(m) > 18 else 1.0
    conn = float(m[16]) if len(m) > 16 else 1.0
    bracket = (1.0
               + float(getattr(cfg, 'W_EFF', 0.3)) * eff
               + float(getattr(cfg, 'W_STRAIGHT', 0.5)) * straight
               + float(getattr(cfg, 'W_EDGE', 0.5)) * edge)
    return food * bracket * conn


def _fitness_tuple(m, cfg):
    """test7 原版元组字典序排序键。"""
    if m[1] >= 99999:
        return (-1e9, 0, 0)
    threshold = float(getattr(cfg, 'LONG_SNAKE_SCORE_THRESHOLD', 3.0))
    food, seen, unseen = m[0], m[1], m[2]
    if food > threshold:
        return (food, unseen, -seen)
    return (food, -seen, unseen)


def _fitness_simple(m, cfg):
    """最简回退口径（test7b）：fitness = food + k·food/max(steps_last,1)。
    判死守卫与 econ 相同（marker/单侧转弯）。"""
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


def _base_fitness(m, cfg):
    """按 FIT_MODE 分派的 base 适应度（报告与 LCB 的底座）。"""
    if getattr(cfg, 'FIT_MODE', 'econ') == 'simple':
        return _fitness_simple(m, cfg)
    return _fitness_econ(m, cfg)


def _sel_key(m, cfg, K):
    """选择键（FITNESS_VERSION=8/9）：base 适应度 − λ·SEL_CV·max(food,1)/√K 的
    LCB 惩罚。同 K 集合内排序不变（等比平移）；作用在跨 K 比较——自适应 K2 后不同
    代的 best 追踪（低 K 代的侥幸高分被正确折价）。tuple 模式不适用 LCB。"""
    base = _base_fitness(m, cfg)
    lam = float(getattr(cfg, 'SEL_LCB_LAMBDA', 0.0))
    cv = float(getattr(cfg, 'SEL_CV', 0.12))
    if lam <= 0 or cv <= 0 or not K or K <= 0:
        return base
    food = float(m[0])
    if food <= 0 or base <= -1e8:
        return base
    return base - lam * cv * food / math.sqrt(K)


def _make_key_fn(cfg, K=None):
    """返回行向量排序键函数 key(m)。K 给定时带 LCB 惩罚（v8/v9）。"""
    mode = getattr(cfg, 'FIT_MODE', 'econ')
    if mode == 'tuple':
        return lambda m: _fitness_tuple(m, cfg)
    if K:
        return lambda m: _sel_key(m, cfg, K)
    return lambda m: _base_fitness(m, cfg)


def _auto_eval_batch(cfg, device):
    """单次扫描允许的最大副本数（个体×局复制后的批维大小）。

    注意：不限 POP_SIZE——局维折叠后批维=个体数×并行局数，按显存估算即可。
    """
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
    # 稀疏 rec：idx int64 8B + w 2B + gather 临时 ~8B；稠密输入 W_in/W_in_eff 4B×O
    per_ind = n * k * 24.0
    per_ind += n * cfg.OBS_DIM * 8.0
    per_ind += n * 48.0                    # 状态/动力学 O(N) 项
    batch = int(total * cfg.EVAL_MEM_FRAC / per_ind)
    return max(32, batch)


def _crn_seed(cfg, gen, stage, ep):
    """每代/阶段/局独立且跨进程确定的种子（python hash 有进程盐，不可用）。"""
    return (int(cfg.CRN_SEED) * 1000003 + int(gen) * 1009
            + int(stage) * 101 + int(ep)) % (2 ** 63 - 1)


def make_bank(cfg, gen, stage, ep, device):
    """生成一局的公共落子流 + 公共初始朝向（全体个体共用）。"""
    g = torch.Generator()
    g.manual_seed(_crn_seed(cfg, gen, stage, ep))
    stream = torch.randint(0, cfg.GRID_SIZE, (cfg.CRN_DRAW, 2), generator=g).to(device)
    dir0 = int(torch.randint(0, 4, (1,), generator=g).item())
    return {'stream': stream, 'dir0': dir0}


def make_banks(cfg, gen, stage, episodes, device):
    return [make_bank(cfg, gen, stage, e, device) for e in range(episodes)]


# ---------- 向量化模仿教师（固定回路 + 单调不变量，与 ref_solver.CycleSolver 同规则）----------
class VectorCycleTeacher:
    """批量教师动作 [B]（0直1左2右）。规则：
    - 安全过滤：界内 & 非颈 & (空|尾(不吃)|食物) & 不变量（吃:fd(h,n)<fd(h,t)，不吃:≤）；
    - 食物在前向段：安全邻格中 fd(n,food) 最小者；
    - 食物不在段：回路后继（若安全），否则任一安全（fd 最小）；
    - 全不安全：维持直行（罕见，mismatch 计入但该步通常即死）。
    全部 [B,4] 张量操作，每步一次调用。"""

    def __init__(self, grid_size, device):
        g = grid_size
        order = []
        for r in range(g):
            cols = range(1, g) if r % 2 == 0 else range(g - 1, 0, -1)
            order += [(r, c) for c in cols]
        order += [(r, 0) for r in range(g - 1, -1, -1)]
        self.N = g * g
        idx = torch.zeros(g, g, dtype=torch.long)
        for i, (r, c) in enumerate(order):
            idx[r, c] = i
        self.idx = idx.to(device)                     # [G,G]
        self.device = device

    @torch.no_grad()
    def act(self, head, food, body, body_len, dir_idx, necks):
        """head/food/body_len/dir_idx [B]；body [B,maxlen,2]；necks [B,2]。"""
        B = head.shape[0]
        G = self.idx.shape[0]
        dev = self.device
        ar = torch.arange(B, device=dev)
        h_i = self.idx[head[:, 0], head[:, 1]]
        t_i = self.idx[body[ar, (body_len - 1).clamp(min=0), 0],
                       body[ar, (body_len - 1).clamp(min=0), 1]]
        f_i = self.idx[food[:, 0], food[:, 1]]
        fd_hf = (f_i - h_i) % self.N
        fd_ht = (t_i - h_i) % self.N

        # 4 邻格 [B,4]
        dirs = torch.tensor([[0, 1], [1, 0], [0, -1], [-1, 0]], device=dev)
        nbr = head.unsqueeze(1) + dirs.unsqueeze(0)               # [B,4,2]
        inb = ((nbr[:, :, 0] >= 0) & (nbr[:, :, 0] < G) &
               (nbr[:, :, 1] >= 0) & (nbr[:, :, 1] < G))
        nbr_c = nbr.clamp(0, G - 1)
        n_i = self.idx[nbr_c[:, :, 0], nbr_c[:, :, 1]]            # [B,4]
        # 占用（尾格视为空）
        flat = body[ar, :, 0] * G + body[ar, :, 1]                # [B,maxlen]
        valid = torch.arange(body.shape[1], device=dev)[None, :] < body_len[:, None]
        tail_flat = body[ar, (body_len - 1).clamp(min=0), 0] * G + \
            body[ar, (body_len - 1).clamp(min=0), 1]
        occ = torch.zeros(B, G * G, dtype=torch.bool, device=dev)
        occ.scatter_(1, flat.clamp(max=G * G - 1), valid)
        occ.scatter_(1, tail_flat.unsqueeze(1), False)
        nbr_flat = nbr_c[:, :, 0] * G + nbr_c[:, :, 1]
        n_occ = occ.gather(1, nbr_flat)                            # [B,4]
        n_food = (nbr[:, :, 0] == food[:, None, 0]) & (nbr[:, :, 1] == food[:, None, 1])
        is_neck = (nbr[:, :, 0] == necks[:, None, 0]) & (nbr[:, :, 1] == necks[:, None, 1])
        # 不变量
        fd_hn = (n_i - h_i.unsqueeze(1)) % self.N                  # [B,4]
        inv = torch.where(n_food, fd_hn < fd_ht.unsqueeze(1),
                          fd_hn <= fd_ht.unsqueeze(1))
        safe = inb & ((~n_occ) | n_food) & inv & (~is_neck)

        # 动作选择
        big = self.N * 10
        in_seg = fd_hf < fd_ht
        # 段内：最小 fd(n,food)；段外：回路后继，否则 fd 最小
        fd_nf = (f_i.unsqueeze(1) - n_i) % self.N                  # [B,4]
        cost_in = torch.where(safe, fd_nf, big)
        succ_i = (h_i + 1) % self.N
        is_succ = (n_i == succ_i.unsqueeze(1)) & safe
        cost_out = torch.where(safe, torch.where(is_succ, -1, fd_hn), big)
        cost = torch.where(in_seg.unsqueeze(1), cost_in, cost_out)
        best = cost.argmin(dim=1)                                  # [B]
        any_safe = safe.any(dim=1)
        # 相对动作：dirs best 与 dir_idx 的差（各 where 均引用原始 raw，防链式污染）
        want = best                                               # DIRS 索引
        raw = (want - dir_idx) % 4
        act = torch.zeros_like(raw)
        act = torch.where(raw == 0, torch.zeros_like(act), act)    # 直行
        act = torch.where(raw == 3, torch.ones_like(act), act)     # 左转
        act = torch.where(raw == 1, torch.full_like(act, 2), act)  # 右转
        # raw==2（反向）不应出现（颈已从邻格排除）
        # no-safe 兜底（接管初期身体可能不满足单调不变量）：向量化洪泛，
        # 选可达空间最大的邻格（语义同标量版 _fallback）
        bad = ~any_safe
        if bool(bad.any()):
            Gg = self.idx.shape[0]
            occm = (occ > 0).view(B, 1, Gg, Gg)
            occm[ar, 0, head[:, 0], head[:, 1]] = False           # 头让位
            free = (~occm).to(torch.float32)
            free[free > 0] = 0                                    # 先全零再播种
            free = torch.zeros_like(free)
            free[ar, 0, head[:, 0], head[:, 1]] = 1.0             # 洪泛种子=头
            wall = occm.to(torch.float32)
            mp = torch.nn.functional.max_pool2d
            for _ in range(Gg * Gg):
                cross = torch.maximum(mp(free, (3, 1), stride=1, padding=(1, 0)),
                                      mp(free, (1, 3), stride=1, padding=(0, 1)))
                free = cross * (1.0 - wall)
            best_sz = None
            fb_act = torch.zeros_like(act)
            for di in range(4):
                nr = nbr[:, di, 0].clamp(0, Gg - 1)
                nc = nbr[:, di, 1].clamp(0, Gg - 1)
                sz = free[ar, 0, nr, nc] + inb[:, di].to(torch.float32) * 1e-3
                sz = torch.where(bad, sz, torch.full_like(sz, -1.0))
                if di == 0:
                    best_sz = sz
                    fb_act = torch.full_like(fb_act, di)
                else:
                    take = sz > best_sz
                    fb_act = torch.where(take, torch.full_like(fb_act, di), fb_act)
                    best_sz = torch.where(take, sz, best_sz)
            act = torch.where(bad, fb_act, act)
        return act


# ==========================================
# 1. 种群基因组张量栈
# ==========================================
class GeneStack:
    """整个种群的基因型/表现型堆叠张量（test16：循环权重固定扇入稀疏化）。

    形状约定（B = 个体数, N = 柱数, O = 观测维, A = 动作维, K = REC_FANIN）：
      M_in [B,N,O]   M_out [B,A,N]
      W_in [B,N,O]   W_out [B,A,N]
      rec_idx [B,N,K] int64 —— 循环连接源 id（自连禁止；重复源=权重叠加）
      rec_w   [B,N,K] float —— 槽位权重（稠密 W_rec/M_rec/W_rec_eff 的替代）
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
            idx = torch.where(idx == ar, (ar + 1) % N, idx)   # 残余兜底：指向下一行
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


def _empty_reach_table(G, depth=7):
    """空盘逐格 BFS 深度≤depth 可达格数（含种子）[G*G]，CPU 预计算一次。

    用作三向洪水稀缺度的位置归一化分母：满自由空间⇒稀缺度 0，与头位置无关。"""
    tab = torch.zeros(G * G)
    for r0 in range(G):
        for c0 in range(G):
            seen = {(r0, c0)}
            frontier = [(r0, c0)]
            for _ in range(depth):
                nxt = []
                for (x, y) in frontier:
                    for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < G and 0 <= ny < G and (nx, ny) not in seen:
                            seen.add((nx, ny))
                            nxt.append((nx, ny))
                frontier = nxt
                if not frontier:
                    break
            tab[r0 * G + c0] = float(len(seen))
    return tab


# ==========================================
# 2. GPU 批量贪吃蛇环境（CRN 版）
# ==========================================
class BatchedSnakeEnv:
    """B 个独立游戏并行（全部状态为 GPU 张量）。死亡个体冻结。

    CRN 多库模式（bank={'stream':[E,DRAW,2], 'dir0':LongTensor[E]}）：
      批维 B = 个体数 × E 局（行主序 p0e0,p0e1,...,p1e0,...），
      e_id = 行号 // (B//E) 决定该行打第几个库的局；
      - 初始朝向按库统一为 dir0[e_id]；
      - 每次落子从自己库的公共流按个体消耗指针取候选，落在占用格则取
        流中下一项（拒绝次数随个体体构差异，属正常 CRN 残差）。
      给定库组与个体基因，各局完全确定 → 同库个体间比较无食物运气差异。
      单库 bank={'stream':[DRAW,2],'dir0':int} 兼容（E=1）。
    """

    def __init__(self, cfg, B, device):
        self.cfg = cfg
        self.B = B
        self.device = device
        self.G = cfg.GRID_SIZE
        self.MAXLEN = self.G * self.G
        self.DIRS = _make_dirs(device)
        self.mode24 = (getattr(cfg, 'OBS_MODE', '32proj') == '24')
        self.empty_reach = _empty_reach_table(
            self.G, int(getattr(cfg, 'FLOOD_DEPTH', 7))).to(device)
        self.crn = None
        self.reset()

    # ---------- 重置 ----------
    def reset(self, bank=None):
        B, G, dev = self.B, self.G, self.device
        center = G // 2
        self.crn = bank
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
        self.steps = torch.zeros(B, dtype=torch.long, device=dev)
        self.steps_wo_food = torch.zeros(B, dtype=torch.long, device=dev)
        self.ate = torch.zeros(B, dtype=torch.bool, device=dev)
        self.died = torch.zeros(B, dtype=torch.long, device=dev)   # 0存活 1撞墙 2撞己 3饿死

    def _next_cand(self):
        """取下一批落子候选 [B,2]：CRN 从各自库的公共流按个体指针，非 CRN 随机。"""
        if self.crn is not None:
            stream = self.crn['stream']
            idx = self.draw_cnt % stream.shape[-2]
            self.draw_cnt += 1
            if stream.dim() == 3:                     # 多库 [E,DRAW,2]
                return stream[self.e_id, idx]
            return stream[idx]                        # 单库 [DRAW,2]
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
            re = self._next_cand()
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
        cand = self._next_cand()
        bad = occ_b.gather(1, (cand[:, 0] * G + cand[:, 1]).unsqueeze(1)).squeeze(1)
        for _ in range(31):
            if not bad.any():
                break
            re = self._next_cand()
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

    # ---------- 观测（与 test7g 一致：曼哈顿食物扇区 + 1/k 身体/障碍扇区）----------
    def obs(self):
        if self.mode24:
            return self._obs24()
        return self._obs40()

    def _obs24(self):
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
            m_ = (bucket == k) & valid
            obs[:, 14 + k] = (close * m_).max(dim=1).values

        ar = torch.arange(B, device=dev)
        tail = self.body[ar, (self.body_len - 1).clamp(min=0)]
        twx = tail[:, 0] - head[:, 0]
        twy = tail[:, 1] - head[:, 1]
        obs[:, 22] = (twx * d[:, 0] + twy * d[:, 1]).float() / G
        obs[:, 23] = (-twx * d[:, 1] + twy * d[:, 0]).float() / G

        return obs

    def _obs40(self):
        """40 维观测 = 旧 32（[0:32]，test12 '32ego1' 逐通道不变）+ 新 8（[32:40]）。"""
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

        # --- 相对方向：[前, 左前, 左, 左后, 后, 右后, 右, 右前] ---
        left = self.DIRS[(self.dir_idx + 3) % 4]
        right = self.DIRS[(self.dir_idx + 1) % 4]
        d8 = [
            d,                           # 0 前
            d + left,                    # 1 左前
            left,                        # 2 左
            left - d,                    # 3 左后
            -d,                          # 4 后
            right - d,                   # 5 右后
            right,                       # 6 右
            d + right,                   # 7 右前
        ]

        # --- [8:16] 食物方位+距离编码（test12 v3，OBS_ENC_VERSION='32ego1'）---
        # [8:12] 4 方位信号 ×K：sig=clamp(û·基方向,0,1)×K。对准 ≈K（与 7h 投影
        #   对准扇区恒 ≈K 同输入量级）；斜 45° → 相邻两方向各 ≈0.707K；
        # [12:16] 距离倒数：sig×K/曼哈顿距离 → 与方位信号联合可线性恢复食物
        #   相对向量（方位通道 + 距离通道，均精确）。
        # 方向基（随 OBS_FOOD_FRAME 切换）：
        #   'ego'（默认）: 前/右/后/左，随头转——与 7h 扇区同自体系；实证绝对系
        #     （'abs'）要求网络先学会 绝对方位⊗头朝向 绑定，随机初网络零初始
        #     相关、选择无梯度（12 代 best food 钉死 ~1.0，惩罚开关无关）。
        vr = (food[:, 0] - head[:, 0]).float()
        vc = (food[:, 1] - head[:, 1]).float()
        dist = (vr.abs() + vc.abs()).clamp(min=1.0)
        k_scale = float(getattr(self.cfg, 'OBS_FOOD_SCALE', 1.0))
        if str(getattr(self.cfg, 'OBS_FOOD_FRAME', 'ego')) == 'abs':
            dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))   # 绝对 E/S/W/N
        else:
            r_ = self.DIRS[(self.dir_idx + 1) % 4]
            l_ = self.DIRS[(self.dir_idx + 3) % 4]
            dirs = ((d[:, 0], d[:, 1]), (r_[:, 0], r_[:, 1]),
                    (-d[:, 0], -d[:, 1]), (l_[:, 0], l_[:, 1]))
        for i, (ax, ay) in enumerate(dirs):
            axf = ax.float() if torch.is_tensor(ax) else float(ax)
            ayf = ay.float() if torch.is_tensor(ay) else float(ay)
            sig = torch.clamp(vr * axf + vc * ayf, min=0.0) / dist
            obs[:, 8 + i] = sig * k_scale
            obs[:, 12 + i] = sig * k_scale / dist
        for i, (ax, ay) in enumerate(dirs):
            axf = ax.float() if torch.is_tensor(ax) else float(ax)
            ayf = ay.float() if torch.is_tensor(ay) else float(ay)
            sig = torch.clamp(vr * axf + vc * ayf, min=0.0) / dist
            obs[:, 8 + i] = sig * k_scale
            obs[:, 12 + i] = sig * k_scale / dist

        # --- [16:24] 自身 8 扇区距离倒数 ---
        ar = torch.arange(B, device=dev)
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

        # --- [24:32] 障碍 8 扇区距离倒数（邻近度语义：对角不折算）---
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
        obs[:, 28] = torch.sqrt(self.body_len.float().clamp(min=1) / self.G)
        k_obs = float(getattr(self.cfg, 'OBS_OBSTACLE_SCALE', 1.0))
        if k_obs != 1.0:
            obs[:, 24:32] = obs[:, 24:32] * k_obs

        # --- [32:40] test15 新通道（OBS_NEW_ENABLED=False 时保持 0 = test12 同数学）---
        if bool(getattr(self.cfg, 'OBS_NEW_ENABLED', True)):
            k_new = float(getattr(self.cfg, 'OBS_NEW_SCALE', 8.0))
            # [32] 饥饿钟压力（追/规换挡紧急度）
            clock = (self.steps_wo_food.float()
                     / (float(getattr(self.cfg, 'STARVE_SLOPE', 5.0))
                        * self.body_len.float() + 20.0))
            obs[:, 32] = clock.clamp(0.0, 1.0) * k_new
            # [33:37] 尾相对方位（前/右/后/左，与食物块同族同序）
            ar = torch.arange(B, device=dev)
            tail_f = self.body[ar, (self.body_len - 1).clamp(min=0)].float()
            vt_r = tail_f[:, 0] - head[:, 0].float()
            vt_c = tail_f[:, 1] - head[:, 1].float()
            d1t = (vt_r.abs() + vt_c.abs()).clamp(min=1.0)
            r_ = self.DIRS[(self.dir_idx + 1) % 4]
            l_ = self.DIRS[(self.dir_idx + 3) % 4]
            tdirs = ((d[:, 0], d[:, 1]), (r_[:, 0], r_[:, 1]),
                     (-d[:, 0], -d[:, 1]), (l_[:, 0], l_[:, 1]))   # 前/右/后/左
            for i, (ax, ay) in enumerate(tdirs):
                axf = ax.float() if torch.is_tensor(ax) else float(ax)
                ayf = ay.float() if torch.is_tensor(ay) else float(ay)
                sig = torch.clamp(vt_r * axf + vt_c * ayf, min=0.0) / d1t
                obs[:, 33 + i] = sig * k_new
            # [37:40] 前/左/右 7 步有限洪水稀缺度
            obs[:, 37:40] = self._flood_scarcity() * k_new

        return obs

    def _flood_scarcity(self):
        """[B,3] 前/左/右 方向 FLOOD_DEPTH 步有限洪水稀缺度（0=空间全达，1=堵死/出界）。

        种子=头沿该方向邻格（占用或出界⇒该向完全稀缺）；蛇身严格占用
        （FLOOD_TAIL_BLOCK=False 时尾格可走）；max_pool 交叉扩散 depth 轮；
        分母=空盘同位置可达数查表（位置归一化，与头在盘何处无关）。"""
        B, G, dev = self.B, self.G, self.device
        cfg = self.cfg
        depth = int(getattr(cfg, 'FLOOD_DEPTH', 7))
        d = self.DIRS[self.dir_idx]
        left = self.DIRS[(self.dir_idx + 3) % 4]
        right = self.DIRS[(self.dir_idx + 1) % 4]
        seeds = torch.stack((d, left, right), dim=1)                # [B,3,2]
        occ = self._occupancy_flat(
            tail_invalid=not bool(getattr(cfg, 'FLOOD_TAIL_BLOCK', True)))
        occ = (occ > 0.5).view(B, 1, G, G)
        free = (~occ).to(torch.float32)
        nb = self.head.unsqueeze(1) + seeds                         # [B,3,2]
        inb = ((nb[..., 0] >= 0) & (nb[..., 0] < G) &
               (nb[..., 1] >= 0) & (nb[..., 1] < G))
        sr = nb[..., 0].clamp(0, G - 1)
        sc = nb[..., 1].clamp(0, G - 1)
        ar = torch.arange(B, device=dev)
        occ_b3 = occ[:, 0]                                          # [B,G,G]
        valid = inb & (~occ_b3[ar.view(B, 1), sr, sc])              # [B,3] 种子须为空格
        seed = torch.zeros(B, 3, G, G, device=dev)
        seed[ar.view(B, 1), torch.arange(3, device=dev).view(1, 3), sr, sc] = valid.float()
        reach = seed
        mp = torch.nn.functional.max_pool2d
        for _ in range(depth):
            cross = torch.maximum(mp(reach, (3, 1), stride=1, padding=(1, 0)),
                                  mp(reach, (1, 3), stride=1, padding=(0, 1)))
            reach = cross * free
        cnt = (reach > 0).float().sum(dim=(2, 3))                   # [B,3]
        cap = self.empty_reach[(sr * G + sc).clamp(0, G * G - 1)]   # [B,3]
        scarcity = (1.0 - cnt / cap.clamp(min=1.0)).clamp(0.0, 1.0)
        return torch.where(valid, scarcity, torch.ones_like(scarcity))

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
            return obs[:, 5:14:2].max(dim=1).values > 0.0
        return obs[:, 8:16].max(dim=1).values > 0.0

    # ---------- 步进 ----------
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
        self._place_food_after_eat(ate)

        starve = self.steps_wo_food > (float(getattr(self.cfg, 'STARVE_SLOPE', 3.0))
                                       * self.body_len.float() + 20)
        self.alive = alive_f & (~crash) & (~starve)
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
    """单次 E-I 迭代（B 个个体并行，无激素支路）。test16：循环项为固定扇入
    gather-乘-归约（rec 三稠密张量的替代）；其余与 test15 一致；
    FATIGUE_TURN_GAIN=0 时 press 惩罚项为 0（疲劳彻底关闭）。"""
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
    gain = float(getattr(cfg, 'FATIGUE_TURN_GAIN', 0.0))
    if gain != 0.0:
        logits[:, 1:] = logits[:, 1:] - gain * press.unsqueeze(1).to(logits.dtype)
    return logits, E_new, I_new, st


def update_fatigue(press, action, decay=None):
    if decay is None:
        decay = 0.7
    return press * decay + (action != 0).to(press.dtype)


def deliberate_batch(pop, obs, E, I, st, press, cfg):
    """K 倍帧率思考：内部迭代 K 次，logits 平均后 argmax。"""
    K = cfg.FRAME_RATE
    logits_sum = None
    for k in range(K):
        o = obs * (cfg.INPUT_DECAY ** k)
        logits, E, I, st = forward_batch(pop, o, E, I, st, press, cfg)
        logits_sum = logits if logits_sum is None else logits_sum + logits
    action = torch.argmax(logits_sum, dim=1)
    return action, E, I, st


# ==========================================
# 4. 种群评估（CRN + 两阶段淘汰）
# ==========================================
def reach_ratio(env, tail_invalid=True):
    """从蛇头可达的自由格占比 [B]（仅观测/诊断用，不进适应度）。
    tail_invalid=True（默认）把尾格视为可通行（历史口径）；
    tail_invalid=False 为严格口径：蛇身含尾格全部算占用（习惯因素2 单连通用）。"""
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


def habit_edge_score(env):
    """习惯因素1（占据边角）单步总分 [B]：格子评分制，按所在圈层计分——
    最外圈每格 16 分，向内每圈减半（10×10 五圈 = 16/8/4/2/1），
    对蛇身占据格求和。总分随蛇长增长；配额用其归一化版（见列 18）。"""
    B, G, dev = env.B, env.G, env.device
    valid = torch.arange(env.MAXLEN, device=dev)[None, :] < env.body_len[:, None]
    layer = torch.minimum(torch.minimum(env.body[:, :, 0], env.body[:, :, 1]),
                          torch.minimum(G - 1 - env.body[:, :, 0],
                                        G - 1 - env.body[:, :, 1]))
    score = 16.0 / (2.0 ** layer.float())                    # 16/8/4/2/1
    return ((score * valid.float()).sum(dim=1))


def habit_edge_share(env):
    """习惯因素1 的蛇长归一化版 [B]∈(0,1]：总分/(16×蛇长)。
    1=蛇身全在最外圈；配额用此值，消除'总分随蛇长单调增长'的混杂
    （否则配额退化为按长度选择，与食物主项重复）。"""
    s = habit_edge_score(env)
    return s / (16.0 * env.body_len.float().clamp(min=1))


def habit_conn_score(env):
    """习惯因素2（单连通）单步得分 [B]：蛇身以外自由空间（严格含尾占用）的
    连通块数倒数——1 块=1，2 块=1/2，3 块=1/3，块越多越差。
    实现：自由格标唯一 id，4 邻 min 标签传播 G² 轮后数不同标签数。"""
    B, G, dev = env.B, env.G, env.device
    occ = env._occupancy_flat().view(B, 1, G, G)
    free = (occ < 0.5).float()
    ids = torch.arange(1, G * G + 1, device=dev).view(1, 1, G, G).float()
    INF = 1e9
    label = torch.where(free > 0.5, ids, torch.full_like(ids, INF))
    mp = torch.nn.functional.max_pool2d

    def min_pool(x, kh, kw, ph, pw):
        return -mp(-x, (kh, kw), stride=1, padding=(ph, pw))

    prev = None
    for it in range(G * G):
        nb = torch.minimum(min_pool(label, 3, 1, 1, 0), min_pool(label, 1, 3, 0, 1))
        label = torch.where(free > 0.5, torch.minimum(nb, label),
                            torch.full_like(label, INF))
        if it % 8 == 7:
            if prev is not None and torch.equal(prev, label):
                break
            prev = label.clone()
    lab = label.view(B, -1)
    fr = free.view(B, -1)
    cnt = torch.zeros(B, G * G + 1, device=dev)
    hit = (fr > 0.5) & (lab < INF)
    cnt.scatter_add_(1, lab.long().clamp(min=0, max=G * G), hit.float())
    ncomp = (cnt > 0).sum(dim=1).float()
    return 1.0 / ncomp.clamp(min=1.0)


def _eval_sweep_chunk(pop_rep, cfg, bank):
    """单次扫描：pop_rep 为个体×E 复制（行主序 p0e0..p0e{E-1},p1e0..），每行打
    bank 中 e_id 对应库的局。返回 [pop_rep.P, 12] 单局指标（无跨局平均）。"""
    B = pop_rep.P
    dev = pop_rep.device
    N = pop_rep.N
    env = BatchedSnakeEnv(cfg, B, dev)
    use_crn = bank is not None
    env.reset(bank=bank if use_crn else None)
    half = torch.float16 if getattr(cfg, 'USE_FP16', True) else torch.float32

    tot_food = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_seen = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_unseen = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_last = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_prox = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_wall = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_self = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_starve = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_act1 = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_act2 = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_turn_last = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_reach = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_reach_n = torch.zeros(B, dtype=torch.float32, device=dev)
    min_reach = torch.full((B,), 1e9, dtype=torch.float32, device=dev)
    # 行为遥测（test12 v5.1 三因素）累计器
    tot_esum = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_eshare = torch.zeros(B, dtype=torch.float32, device=dev)
    tot_conn = torch.zeros(B, dtype=torch.float32, device=dev)
    conn_n = torch.zeros(B, dtype=torch.float32, device=dev)
    beh_steps = torch.zeros(B, dtype=torch.float32, device=dev)

    E = torch.zeros(B, N, dtype=half, device=dev)
    I = torch.zeros(B, N, dtype=half, device=dev)
    st = torch.zeros(B, N, dtype=half, device=dev)
    press = torch.zeros(B, dtype=torch.float32, device=dev)
    last = torch.zeros(B, dtype=torch.float32, device=dev)
    turn_cnt = torch.zeros(B, dtype=torch.float32, device=dev)
    turn_last = torch.zeros(B, dtype=torch.float32, device=dev)
    use_im = float(getattr(cfg, 'IMITATION_W', 0.0)) > 0
    teacher = VectorCycleTeacher(cfg.GRID_SIZE, dev)   # mismatch 恒追踪（代价可忽略）
    mis_cnt = torch.zeros(B, dtype=torch.float32, device=dev)
    mis_steps = torch.zeros(B, dtype=torch.float32, device=dev)

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

        act, E, I, st = deliberate_batch(pop_rep, obs, E, I, st, press, cfg)
        press = update_fatigue(press, act, decay=float(cfg.FATIGUE_TURN_DECAY))
        if teacher is not None:
            ar = torch.arange(B, device=dev)
            necks = env.body[ar, 1]
            t_act = teacher.act(env.head, env.food, env.body, env.body_len,
                                env.dir_idx, necks)
            mis_cnt += (al & (act != t_act)).float()
            mis_steps += al.float()
        tot_act1 += (al & (act == 1)).float()
        tot_act2 += (al & (act == 2)).float()
        turn_cnt += (al & (act != 0)).float()

        # 行为遥测（步进前状态，test12 v5.1 三因素）
        if bool(al.any()):
            # 因素1 边角评分（16/8/4/2/1 圈层制）：总分 + 长度归一化占比
            es = habit_edge_score(env)
            tot_esum += al.float() * es
            tot_eshare += al.float() * es / (16.0 * env.body_len.float().clamp(min=1))
            # 因素2 conn：每 4 存活步采样一次连通块数倒数（严格含尾）
            if t % 4 == 0:
                tot_conn += al.float() * habit_conn_score(env)
                conn_n += al.float()
            beh_steps += al.float()

        env.step(act)
        ate_now = al & env.ate
        tot_food += ate_now.float()
        last = torch.where(ate_now, torch.full_like(last, float(t + 1)), last)
        turn_last = torch.where(ate_now, turn_cnt, turn_last)
        if ate_now.any():
            rr = reach_ratio(env)
            tot_reach += rr * ate_now.float()
            tot_reach_n += ate_now.float()
            # 孤岛惩罚统计（test12）：过程中可达空间最小占比（分母=自由格数）
            min_reach = torch.where(ate_now & (rr < min_reach), rr, min_reach)
        if env.all_done():
            break

    prox = tot_prox / torch.clamp(tot_seen + tot_unseen, min=1e-6)
    avg_reach = tot_reach / tot_reach_n.clamp(min=1.0)
    mismatch = mis_cnt / mis_steps.clamp(min=1.0)
    island = ((min_reach < float(getattr(cfg, 'ISLAND_THRESHOLD', 0.3)))
              & (tot_reach_n > 0)).float()
    edge_sum = tot_esum / beh_steps.clamp(min=1.0)
    edge_share = tot_eshare / beh_steps.clamp(min=1.0)
    conn_score = tot_conn / conn_n.clamp(min=1.0)
    straight = 1.0 - (tot_act1 + tot_act2) / beh_steps.clamp(min=1.0)
    metrics = torch.stack((tot_food, tot_seen, tot_unseen, last, prox,
                           tot_wall + (env.died == 1).float(),
                           tot_self + (env.died == 2).float(),
                           tot_starve + (env.died == 3).float(),
                           tot_act1, tot_act2, turn_last, avg_reach,
                           mismatch, min_reach.clamp(max=1.0), island,
                           edge_sum, conn_score, straight, edge_share), dim=1)
    return metrics


def apply_weak_mask(pop, cfg):
    """评估期弱连接屏蔽（原地作用于 pop 的 rec_w——pop[idx] 是拷贝，基因栈安全）：
    每个体全部槽位权重中 |rec_w| 最小的 WEAK_MASK_FRAC 比例置零（槽位仍在，
    权重为 0 等效断开）。逐体 kthvalue 阈值实现；阈值并列可能多置零数个。"""
    frac = float(getattr(cfg, 'WEAK_MASK_FRAC', 0.0))
    if frac <= 0 or pop.rec_w is None:
        return
    with torch.no_grad():
        W = pop.rec_w.float()                       # [B,N,K]
        B = W.shape[0]
        for r in range(B):
            flat = W[r].abs().flatten()
            kk = int(math.ceil(flat.numel() * frac))
            if kk <= 0:
                continue
            thr = torch.kthvalue(flat, kk).values
            zero = (W[r].abs() <= thr)
            W[r] = torch.where(zero, torch.zeros_like(W[r]), W[r])
        pop.rec_w = W.to(pop.dtype)


def _eval_pop_banks(pop, cfg, banks):
    """E=len(banks) 局并行评估：个体×E 复制进同一批扫描（局维折叠，扫描次数
    不随局数增长——GPU 利用率低时墙上时间 ∝ 扫描次数而非局数）。
    所有分块共用同一组 banks（CRN 关键）；评估副本先做弱连接屏蔽。
    返回 [P,19] = E 局均值（12=mismatch，13=min_reach，14=island（诊断）；
    行为三因素：15=edge_sum（圈层总分）、16=conn_score（连通块倒数）、
    17=straight、18=edge_share（总分/(16×蛇长)，配额用））。"""
    E = len(banks)
    dev = pop.device
    use_crn = banks[0] is not None
    if use_crn:
        bank = {'stream': torch.stack([b['stream'] for b in banks]),
                'dir0': torch.tensor([b['dir0'] for b in banks],
                                     dtype=torch.long, device=dev)}
    else:
        bank = None
    max_B = _auto_eval_batch(cfg, dev)
    per_P = max(1, max_B // E)
    ncol = 19
    out = torch.zeros(pop.P, ncol)
    for lo in range(0, pop.P, per_P):
        hi = min(lo + per_P, pop.P)
        n = hi - lo
        idx = torch.arange(lo, hi, device=dev).repeat(E)   # 块布局：e = 行 // n
        sub = pop[idx]                                      # 高级索引=拷贝，可安全屏蔽
        apply_weak_mask(sub, cfg)
        sub.refresh_eff()
        m = _eval_sweep_chunk(sub, cfg, bank).cpu()
        out[lo:hi] = m.view(E, n, ncol).mean(dim=0)
    return out


def evaluate_population_gpu(pop, cfg, gen=0, k2=None):
    """两阶段淘汰评估（CRN）：
    阶段1：全种群 × K1 局（同库精确可比）→ 保前 STAGE2_KEEP；
    阶段2：幸存者 × K2 局（新库），幸存者指标 = (K1·m1 + K2·m2)/(K1+K2)。
    k2=None 时用 cfg.EVAL_EPISODES；test16a 自适应 K2 由 run_training 逐代传入。
    返回 (metrics[P,19], order[P])：order = 幸存者按累计适应度降序，
    其后为落选者按阶段1适应度降序——精英只能出自幸存者。
    排序键带 LCB（v8）：幸存者按 K=K1+K2、落选者按 K=K1 折价。
    """
    if getattr(cfg, 'USE_FP16', True):
        pop.fp16()
    use_crn = bool(getattr(cfg, 'USE_CRN', True))
    dev = pop.device
    P = pop.P
    K1 = int(getattr(cfg, 'STAGE1_EPS', 3))
    K2 = int(k2 if k2 is not None else cfg.EVAL_EPISODES)
    key1 = _make_key_fn(cfg, K=K1)          # 阶段1 口径
    key2 = _make_key_fn(cfg, K=K1 + K2)     # 累计口径（幸存者）
    two_stage = (getattr(cfg, 'STAGE2_KEEP', 0) >= cfg.ELITE_SIZE
                 and K1 > 0 and K2 > K1)

    m1 = _eval_pop_banks(pop, cfg,
                         make_banks(cfg, gen, 1, K1, dev) if use_crn else [None] * K1)
    mn1 = m1.numpy()
    # sorted(reverse=True) 稳定排序，且兼容 tuple 字典序适应度（np.argsort 不行）
    order1 = sorted(range(P), key=lambda i: key1(mn1[i]), reverse=True)
    if not two_stage:
        return m1, order1

    keep = min(int(getattr(cfg, 'STAGE2_KEEP', 410)), P)
    surv_idx = order1[:keep]
    surv_set = set(surv_idx)
    sub = pop[surv_idx]
    m2 = _eval_pop_banks(sub, cfg,
                         make_banks(cfg, gen, 2, K2, dev) if use_crn else [None] * K2)

    metrics = m1.clone()
    metrics[surv_idx] = (K1 * m1[surv_idx] + K2 * m2) / (K1 + K2)
    mn = metrics.numpy()
    # 幸存者按累计适应度降序在前；落选者按阶段1适应度降序垫后（无精英资格）
    surv_rank = sorted(surv_idx, key=lambda i: key2(mn[i]), reverse=True)
    out_rank = [i for i in order1 if i not in surv_set]
    return metrics, surv_rank + out_rank


# ==========================================
# 5. 进化（类正态变异强度 + GPU 向量化交叉/变异）
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
    """进化下一代：ELITE 精英 + (P-ELITE) 后代（交叉同 test7g）。

    变异强度：每子代抽因子 s（类正态分布）缩放其全部变异算子。
    order：evaluate_population_gpu 给出的排序（幸存者优先），缺省时按适应度自排。
    """
    P = pop.P
    N = pop.N
    dev = pop.device
    active = _freeze_active_groups(gen, cfg)
    has_g1 = 'G1' in active
    has_g2 = 'G2' in active

    # --- 精英排序 ---
    if order is None:
        mn = metrics.cpu().numpy()
        key_fn = _make_key_fn(cfg, K=cfg.STAGE1_EPS + int(cfg.EVAL_EPISODES))
        order = sorted(range(P), key=lambda i: key_fn(mn[i]), reverse=True)
    elite_idx = order[:cfg.ELITE_SIZE]

    # --- te-配额精英：按 te=SL/TL（截断 CAP，要求 food>0）取配额外最优插入 ---
    te_quota = int(getattr(cfg, 'TE_ELITE', 0))
    if te_quota > 0:
        mn = metrics.cpu().numpy()
        cap = float(getattr(cfg, 'TURN_EFF_CAP', 4.0))
        te = np.where((mn[:, 0] > 0) & (mn[:, 10] > 0),
                      np.minimum(mn[:, 3] / np.maximum(mn[:, 10], 1.0), cap), 0.0)
        elite_set = set(elite_idx)
        te_order = sorted(range(P), key=lambda i: te[i], reverse=True)
        extra = [i for i in te_order if i not in elite_set][:te_quota]
        if extra:
            elite_idx = elite_idx[:cfg.ELITE_SIZE - len(extra)] + extra
    # 行为习惯不走配额：v7 起三因素以乘法因子并入适应度
    # （food×(1+…)×conn），选择完全由统一适应度排序驱动。
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

    # --- 默认全部基因单亲（p1）遗传 ---
    # test16 修复 test12 血统隐患：旧实现 children 预分配 torch.empty，冻结组
    # 不被交叉覆盖 → 未初始化内存混入种群（靠 CUDA 缓存分配器返回陈旧值侥幸
    # 可用）。本实现先整栈继承 p1，活跃组再交叉/变异覆盖——冻结组=纯遗传。
    with torch.no_grad():
        for g in GeneStack.GENES:
            setattr(children, g, getattr(p1, g).clone())

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
        # ---- G1 交叉：结构组（掩码 + 权重 + 输出偏置）----
        # rec 语义变化：稠密"行×列双掩码" → 按行（按突触后神经元）整行选父，
        # 保持每行 K 槽扇入结构。
        if has_g1:
            col_mask = torch.rand(B2, N, device=dev) > 0.5
            rec_row = torch.rand(B2, N, device=dev) > 0.5          # rec 按行选父

            children.W_in = torch.where(col_mask.unsqueeze(2), p1.W_in, p2.W_in)
            children.M_in = torch.where(col_mask.unsqueeze(2), p1.M_in, p2.M_in)
            rm = rec_row.unsqueeze(-1)
            children.rec_idx = torch.where(rm, p1.rec_idx, p2.rec_idx)
            children.rec_w = torch.where(rm, p1.rec_w, p2.rec_w)
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
# 6. 保存 / 加载（断点 + 最优模型）
# ==========================================
def load_best_state(path, cfg):
    if not path or not os.path.exists(path):
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
        if saved_cfg.get('OBS_ENC_VERSION', '32proj') != getattr(cfg, 'OBS_ENC_VERSION', '32proj'):
            print(f"  警告: 模型 {path} 观测编码不符 "
                  f"({saved_cfg.get('OBS_ENC_VERSION')} != {cfg.OBS_ENC_VERSION})，已忽略种子")
            return None
        if saved_cfg.get('BRAIN_VERSION') != getattr(cfg, 'BRAIN_VERSION', 'sparse1'):
            print(f"  警告: 模型 {path} 基因组版本不符 "
                  f"({saved_cfg.get('BRAIN_VERSION', '旧格式无版本')} != "
                  f"{getattr(cfg, 'BRAIN_VERSION', '未知')})，已忽略种子")
            return None
    st = data.get('brain')
    if st is None:
        return None
    return st, float(data.get('food', -1.0)), float(data.get('steps', 0.0))


def load_migratable_state(path, cfg):
    """稠密单脑 → 稀疏基因组迁移（test16 未实现，显式拒绝）。

    稠密 W_rec/M_rec → 每行取 |W| 前 K 强可构造 (rec_idx, rec_w)，但 N=1024
    与稠密 256 的柱数不匹配、拓扑语义不同，故不做隐式变换，直接拒绝。"""
    print(f"  警告: test16（稀疏固定扇入 BRAIN_VERSION={getattr(cfg, 'BRAIN_VERSION', '?')}）"
          f"不支持稠密血统迁移（{path}），请从零训练")
    return None


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


def load_notify_token():
    """AutoDL 开发者 Token：优先读脚本同目录/工作目录的 notify_token.txt，
    其次环境变量 AUTODL_TOKEN。找不到（或内容明显不是 Token）返回 None（静默跳过通知）。"""
    def _valid(t):
        # 真 Token 为 ASCII 串；含空白/中文说明还是占位说明文本，视同未配置
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
    """AutoDL 微信通知（api.autodl.com/docs/msg）。仅完成时调用一次；
    任何失败只打印警告，绝不影响训练结果。"""
    tok = load_notify_token()
    if not tok:
        print("[通知] 未找到 notify_token.txt / AUTODL_TOKEN，跳过微信通知")
        return
    try:
        import urllib.request
        req = urllib.request.Request(
            'https://www.autodl.com/api/v1/wechat/message/send',
            data=json.dumps({'title': title[:20], 'name': 'test16a 抗噪声选择进化',
                             'content': content[:200]}).encode('utf-8'),
            headers={'Content-Type': 'application/json',
                     'Authorization': tok},
            method='POST')
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode('utf-8', 'replace')
        print(f"[通知] AutoDL 微信通知已发送: {body[:120]}")
    except Exception as e:
        print(f"[通知] 微信通知发送失败（不影响训练）: {e}")


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
    if saved_cfg and saved_cfg.get('OBS_ENC_VERSION', '32proj') != getattr(cfg, 'OBS_ENC_VERSION', '32proj'):
        print(f"  警告: 断点 {path} 观测编码不符 "
              f"({saved_cfg.get('OBS_ENC_VERSION')} != {cfg.OBS_ENC_VERSION})，已忽略")
        return None
    # 基因组版本守卫：缺 BRAIN_VERSION 键（test15 及更早稠密断点）或版本不符 → 拒绝续训
    if saved_cfg and saved_cfg.get('BRAIN_VERSION') != getattr(cfg, 'BRAIN_VERSION', 'sparse1'):
        print(f"  警告: 断点 {path} 基因组版本不符 "
              f"({saved_cfg.get('BRAIN_VERSION', '旧格式无版本')} != "
              f"{getattr(cfg, 'BRAIN_VERSION', '未知')})，已忽略")
        return None
    return data


# ==========================================
# 7. 主循环
# ==========================================
def plot_history_png(cfg, history):
    """训练过程图：左栏食物曲线；右栏 best 个体适应度按来源堆叠
    （food 红底 / eff 绿 / habit 橙 + 总分黑虚线）。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei',
                                              'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    gens = history['gen']
    ax1.plot(gens, history['best_food'], label='Best Food',
             color='red', marker='o', markersize=3)
    ax1.plot(gens, history['avg_food'], label='Avg Food',
             color='blue', alpha=0.6)
    if any(v is not None for v in history.get('elite_food', [])):
        ax1.plot(gens, history['elite_food'], label='Elite Food',
                 color='green', alpha=0.8)
    ax1.set_title("Evolution Progress — Food Count")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Food Eaten")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右栏：适应度来源堆叠（v7 乘法归因：food×conn / eff / 少转弯 / 边角），黑虚线=总分
    def _arr(key):
        v = history.get(key, [])
        return np.array([0.0 if x is None else float(x) for x in v])
    p_food = _arr('fit_parts_food')
    p_eff = _arr('fit_parts_eff')
    p_st = _arr('fit_parts_straight')
    p_ed = _arr('fit_parts_edge')
    if len(gens) and (len(p_food) or len(history.get('best_fit', []))):
        n = min(len(gens), len(history.get('best_fit', [])) or len(p_food))
        g = gens[:n]
        pf = np.nan_to_num(p_food[:n])
        pe = np.nan_to_num(p_eff[:n])
        ax2.stackplot(g, pf, pe, p_st, p_ed,
                      colors=('#d9534f', '#5cb85c', '#428bca', '#f0ad4e'),
                      alpha=0.75,
                      labels=('食物数', 'eff 效率项',
                              '少转弯项', '边角项'))
        # 各层顶界线（层色细线）：薄层（如 eff 项）靠界线可辨
        stack_top = pf + pe + p_st + p_ed
        c1 = pf
        c2 = pf + pe
        c3 = c2 + p_st
        for cc, col in ((c1, '#d9534f'), (c2, '#5cb85c'),
                        (c3, '#428bca'), (stack_top, '#f0ad4e')):
            ax2.plot(g, cc, color=col, lw=1.0, alpha=0.9)
        pn = np.nan_to_num(_arr('fit_parts_noconn')[:n])
        # 黑虚线 = 四项之和（当期 best 个体适应度），恒在堆叠顶，避免与
        # 历史全局 best_fit（不同步）错位
        ax2.plot(g, stack_top, color='black', lw=1.4, ls='--',
                 label='适应度总分(当期best)')
        # 单连通折扣：无 conn 缩放的假想适应度（虚线）与损失区（斜线填充）
        if np.any(pn > stack_top + 1e-9):
            ax2.plot(g, pn, color='gray', lw=1.2, ls='--',
                     label='无单连通缩放')
        if len(g):
            # 结论标记：图内右下角，按堆叠顺序（自下而上）竖向排列
            _txt = ('食物数 {:.1f}\neff 效率项 {:.2f}\n'
                    '少转弯项 {:.1f}\n边角项 {:.1f}'.format(
                        pf[-1], pe[-1], p_st[-1], p_ed[-1]))
            ax2.text(0.98, 0.02, _txt, transform=ax2.transAxes,
                     ha='right', va='bottom', fontsize=8, color='black',
                     bbox=dict(facecolor='white', alpha=0.75,
                               edgecolor='gray', lw=0.6))

            ax2.fill_between(g, stack_top, pn,
                             where=pn > stack_top + 1e-9,
                             facecolor='none', hatch='///',
                             edgecolor='gray', lw=0.8, ls='--',
                             label='单连通折扣')
        ax2.set_title('Best 个体适应度来源分解（堆叠）')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Fitness')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    hist_path = cfg.CHECKPOINT_PATH.replace('_checkpoint', '_history').replace('.pth', '.png')
    fig.savefig(hist_path, dpi=100)
    plt.close(fig)
    print(f"历史曲线已保存: {hist_path}")


def run_training(cfg):
    device = _resolve_device(cfg)
    print(f"[GPU] device = {device}  "
          f"({'GPU: ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU 回退'})")
    if device.type == 'cuda':
        mem = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"[GPU] 显存 {mem:.1f} GB, 自动评估批大小 = {_auto_eval_batch(cfg, device)}")
    print(f"[环境] grid={cfg.GRID_SIZE} max_steps={cfg.MAX_STEPS} | "
          f"饿死钟 steps_wo_food > {cfg.STARVE_SLOPE}·len+20 | "
          f"疲劳 turn_gain={cfg.FATIGUE_TURN_GAIN} decay={cfg.FATIGUE_TURN_DECAY} | "
          f"单侧转弯判死 {cfg.ONE_SIDED_TURN_DEATH}")
    print(f"[种群] pop={cfg.POP_SIZE} 精英={cfg.ELITE_SIZE} 列={cfg.NUM_COLUMNS} "
          f"扇入K={getattr(cfg, 'REC_FANIN', 16)} | M_in/M_out密度={cfg.INIT_DENSITY} | "
          f"轮换 {cfg.CYCLE_PATTERN}")
    print(f"[稀疏] rec 槽位/体 = {cfg.NUM_COLUMNS}×{getattr(cfg, 'REC_FANIN', 16)}"
          f"（对照稠密 {cfg.NUM_COLUMNS}²={cfg.NUM_COLUMNS ** 2}，"
          f"利用率 {getattr(cfg, 'REC_FANIN', 16) / cfg.NUM_COLUMNS:.2%}）")
    print(f"[筛选] CRN={cfg.USE_CRN}(seed={cfg.CRN_SEED}) | 两阶段 K1={cfg.STAGE1_EPS} → 保 "
          f"{cfg.STAGE2_KEEP} → K2={cfg.EVAL_EPISODES} | "
          f"变异 s~{cfg.MUT_SCALE_DIST}(σ={cfg.MUT_SCALE_SIGMA}) "
          f"clip[{cfg.MUT_SCALE_MIN},{cfg.MUT_SCALE_MAX}]")
    if getattr(cfg, 'FIT_MODE', 'econ') == 'simple':
        fit_desc = (f"food + {cfg.SIMPLE_EFF_W}·food/steps_last（simple 最简回退）")
    else:
        fit_desc = (f"food×(1 + {cfg.W_EFF}·eff + {cfg.W_STRAIGHT}·straight "
                    f"+ {cfg.W_EDGE}·edge_share)×conn")
    print(f"[适应度 v{cfg.FITNESS_VERSION}] {fit_desc} | "
          f"选择键 −{cfg.SEL_LCB_LAMBDA}·{cfg.SEL_CV}·food/√K（LCB） | "
          f"弱连接屏蔽 rec_w×{1 - cfg.WEAK_MASK_FRAC:.0%} | "
          f"单侧判死 {cfg.ONE_SIDED_TURN_DEATH}")
    print(f"[test16a] 亲本 {cfg.ELITE_SIZE}/{cfg.POP_SIZE}（1/2 放宽）| "
          f"自适应K2: δ={cfg.RES_TARGET} max={cfg.K2_MAX} "
          f"(K2=clip(⌈({cfg.SEL_CV}·S/{cfg.RES_TARGET})²⌉−{cfg.STAGE1_EPS},"
          f"{cfg.EVAL_EPISODES},{cfg.K2_MAX}))")
    print(f"[观测] {cfg.OBS_ENC_VERSION} frame={getattr(cfg, 'OBS_FOOD_FRAME', 'ego')} "
          f"K={cfg.OBS_FOOD_SCALE} self/obs×{cfg.OBS_SELF_SCALE}/{cfg.OBS_OBSTACLE_SCALE} | "
          f"孤岛诊断阈值 min_reach<{cfg.ISLAND_THRESHOLD}")
    print(f"[新通道] enabled={cfg.OBS_NEW_ENABLED} scale={cfg.OBS_NEW_SCALE} "
          f"flood_depth={cfg.FLOOD_DEPTH} tail_block={cfg.FLOOD_TAIL_BLOCK}")
    print(f"[输出] ckpt={cfg.CHECKPOINT_PATH} | best={cfg.BEST_MODEL_PATH} | "
          f"history={cfg.HISTORY_JSON_PATH}")
    t_program = time.perf_counter()

    start_gen = 0
    pop = GeneStack(cfg, device=device)
    history = {'gen': [], 'best_food': [], 'avg_food': [], 'best_seen': [],
               'best_unseen': [], 'elite_food': [], 'best_fit': [], 'best_turneff': [],
               'best_epref': [], 'best_conn': [], 'best_straight': [],
               'fit_parts_food': [], 'fit_parts_eff': [],
               'fit_parts_straight': [], 'fit_parts_edge': [],
               'fit_parts_noconn': [], 'k2': []}
    cum_eval_time = 0.0
    cum_evolve_time = 0.0
    best_state = None
    best_food = -1.0
    best_seen = 0.0
    best_unseen = 0.0
    best_last = 0.0
    best_prox = 0.0
    best_row = np.zeros(13)
    best_row[1] = 99999.0
    best_K = int(cfg.STAGE1_EPS) + int(cfg.EVAL_EPISODES)   # best_row 的测量 K（v8 LCB 用）
    latest_gen_best_state = None
    latest_gen_best_food = -1.0
    latest_gen_best_seen = 0.0
    latest_gen_best_unseen = 0.0

    if cfg.AUTO_RESUME:
        ck = load_checkpoint7(cfg.CHECKPOINT_PATH, cfg)
        if ck is not None:
            start_gen = int(ck['next_gen'])
            pop.unpack(ck['pop'])
            history = ck.get('history', history)
            for k in ('elite_food', 'best_fit', 'best_turneff',
                      'best_epref', 'best_conn', 'best_straight',
                      'fit_parts_food', 'fit_parts_eff',
                      'fit_parts_straight', 'fit_parts_edge',
                      'fit_parts_noconn', 'k2'):
                history.setdefault(k, [])
            # 跨版本续训（如 7g 断点）时补齐新键长度，避免曲线错位
            for k in ('elite_food', 'best_fit', 'best_turneff',
                      'best_epref', 'best_conn', 'best_straight',
                      'fit_parts_food', 'fit_parts_eff',
                      'fit_parts_straight', 'fit_parts_edge',
                      'fit_parts_noconn', 'k2'):
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
            best_row = np.zeros(13)
            best_row[1] = 99999.0
            best_K = int(cfg.STAGE1_EPS) + int(cfg.EVAL_EPISODES)
            if 'best_row' in ck:
                best_row = np.asarray(ck['best_row'], dtype=np.float64)
            if best_state is not None:
                random.setstate(ck['random_state'])
                torch.set_rng_state(ck['torch_rng_state'])
            # 适应度版本变化 → 旧 best 行跨公式不可比，重置追踪（history 保留）
            ck_ver = int(ck.get('config', {}).get('FITNESS_VERSION', 1))
            if ck_ver != int(getattr(cfg, 'FITNESS_VERSION', 1)):
                best_row = np.zeros(13)
                best_row[1] = 99999.0
                best_K = int(cfg.STAGE1_EPS) + int(cfg.EVAL_EPISODES)
                best_state = None
                best_food = -1.0
                print(f"  [适应度版本变更 v{ck_ver} -> v{cfg.FITNESS_VERSION}] "
                      f"best 追踪已重置（种群与历史保留，从本代重新记录）")
            print(f"=== 检测到断点 [{cfg.CHECKPOINT_PATH}] ===")
            print(f"  已完成 {start_gen} 代 -> 从第 {start_gen} 代接续 | "
                  f"历史最优: Food={best_food:.1f} | 已耗时 {cum_eval_time + cum_evolve_time:.1f}s")

    if pop.M_in is None:
        t0 = time.perf_counter()
        print("初始化种群（GPU 随机初始化）...")
        pop.random_init()
        if cfg.SEED_FROM_BEST and cfg.SEED_MODEL_PATH:
            if bool(getattr(cfg, 'SEED_MIGRATE', False)):
                seed = load_migratable_state(cfg.SEED_MODEL_PATH, cfg)
            else:
                seed = load_best_state(cfg.SEED_MODEL_PATH, cfg)
            if seed is not None:
                st, s_food, s_steps = seed
                if bool(getattr(cfg, 'SEED_POP', False)):
                    # 血统迁移：全种群注入该底盘
                    for i in range(cfg.POP_SIZE):
                        pop.set_individual_from_state(i, st)
                    print(f"  [Seed] 全种群已注入 {cfg.SEED_MODEL_PATH} "
                          f"(Food={s_food:.1f})，新观测列按 INIT_DENSITY/σ=0.1 激活")
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

    try:
        for gen in range(start_gen, cfg.GENERATIONS):
            t_eval = time.perf_counter()
            # --- 自适应 K2（test16a）：按历史 BestFood 运行最大值保分辨率 δ ---
            #    σ_ε ≈ SEL_CV·S/√(K1+K2) ≤ δ  →  K2 = (SEL_CV·S/δ)² − K1
            K1 = int(cfg.STAGE1_EPS)
            S = max(history['best_food']) if history['best_food'] else 0.0
            K2_BASE = int(cfg.EVAL_EPISODES)            # 已由 main 与 STAGE2_EPS 同步
            k2_t = K2_BASE
            if getattr(cfg, 'K2_ADAPTIVE', True) and S > 0:
                need = math.ceil((cfg.SEL_CV * S / max(cfg.RES_TARGET, 1e-6)) ** 2)
                k2_cap = max(int(cfg.K2_MAX), K2_BASE)  # base 永远生效（修钳位反转）
                k2_t = int(min(max(need - K1, K2_BASE), k2_cap))
            metrics, order = evaluate_population_gpu(pop, cfg, gen=gen, k2=k2_t)
            eval_time = time.perf_counter() - t_eval
            cum_eval_time += eval_time

            mn = metrics.numpy()
            K_eff = K1 + k2_t
            key_cur = _make_key_fn(cfg, K=K_eff)      # 选择键（v8：含 LCB 折价）
            best_idx = order[0]
            b_food, b_seen, b_unseen, b_last, b_prox = (
                float(mn[best_idx][0]), float(mn[best_idx][1]),
                float(mn[best_idx][2]), float(mn[best_idx][3]), float(mn[best_idx][4]))
            b_turn = (float(mn[best_idx][8]) + float(mn[best_idx][9])) / max(b_seen + b_unseen, 1.0)
            b_te = (min(b_last / max(float(mn[best_idx][10]), 1.0), cfg.TURN_EFF_CAP)
                    if (mn[best_idx][10] > 0 and b_food > 0) else
                    (cfg.TURN_EFF_CAP if b_food > 0 else 0.0))
            b_fit = _base_fitness(mn[best_idx], cfg)   # history 报告口径 = base
            b_fit = b_fit if np.isscalar(b_fit) else b_fit[0]
            avg_food = float(np.mean(mn[:, 0]))
            elite_food = float(np.mean([mn[i][0] for i in order[:cfg.ELITE_SIZE]]))

            history['gen'].append(gen)
            history['best_food'].append(b_food)
            history['avg_food'].append(avg_food)
            history['best_seen'].append(b_seen)
            history['best_unseen'].append(b_unseen)
            history['elite_food'].append(elite_food)
            history['best_fit'].append(float(b_fit))
            history['best_turneff'].append(float(b_te))
            history['k2'].append(int(k2_t))

            # best 个体行为遥测（卡55监控：贴边率/头周自由度应随选择缓升）
            history['best_epref'].append(float(mn[best_idx][18]))   # edge_share
            history['best_conn'].append(float(mn[best_idx][16]))    # conn_score
            history['best_straight'].append(float(mn[best_idx][17]))

            # best 个体适应度分解（过程图右栏堆叠：food/eff/habit×3 因素）
            b_row = mn[best_idx]
            # 适应度分解（v7 加法归因，与公式同构）：四份之和恰=总分；
            # noconn 基准 = conn=1 时的假想适应度（差值即单连通折扣）。
            # 教训：顺序累乘分解 (1+a)(1+b)(1+c) 会多出交叉项，使 stack_top
            # 抬高到 noconn 之上 → 绘图条件永假、虚线不显示（实测踩坑）。
            if getattr(cfg, 'FIT_MODE', 'econ') == 'simple':
                # simple 口径：适应度=food+k·eff，无 econ 归因——全部记入 food 份
                history['fit_parts_food'].append(float(b_fit))
                history['fit_parts_eff'].append(0.0)
                history['fit_parts_straight'].append(0.0)
                history['fit_parts_edge'].append(0.0)
                history['fit_parts_noconn'].append(float(b_fit))
            else:
                _c = float(b_row[16]) if len(b_row) > 16 else 1.0
                _eff = float(b_row[0]) / max(float(b_row[3]), 1.0)
                _base = float(b_row[0]) * _c
                history['fit_parts_food'].append(_base)
                history['fit_parts_eff'].append(
                    _base * float(cfg.W_EFF) * _eff)
                history['fit_parts_straight'].append(
                    _base * float(cfg.W_STRAIGHT) * float(b_row[17]))
                history['fit_parts_edge'].append(
                    _base * float(cfg.W_EDGE) * float(b_row[18]))
                history['fit_parts_noconn'].append(
                    float(b_row[0]) * (1.0 + float(cfg.W_EFF) * _eff
                                       + float(cfg.W_STRAIGHT) * float(b_row[17])
                                       + float(cfg.W_EDGE) * float(b_row[18])))

            # best 追踪（v8）：LCB 键跨 K 可比——best_row 用其测量时的 K 折价，
            # 自适应 K2 加密后高 K 代不会被低 K 代的侥幸高分压住
            if key_cur(mn[best_idx]) > _make_key_fn(cfg, K=best_K)(best_row):
                best_food = b_food
                best_seen = b_seen
                best_unseen = b_unseen
                best_last = b_last
                best_prox = b_prox
                best_row = mn[best_idx].copy()
                best_K = K_eff
                best_state = pop.individual_state(best_idx, use_half=False)

            latest_gen_best_state = pop.individual_state(best_idx, use_half=False)
            latest_gen_best_food = b_food
            latest_gen_best_seen = b_seen
            latest_gen_best_unseen = b_unseen

            if gen < cfg.GENERATIONS - 1:
                t_ev = time.perf_counter()
                pop = evolve_topology_gpu(pop, metrics, cfg, gen=gen, order=order)
                evolve_time = time.perf_counter() - t_ev
                cum_evolve_time += evolve_time
            else:
                evolve_time = 0.0

            if (gen + 1) % cfg.PRINT_HISTORY_EVERY == 0:
                avg_wall = float(np.mean(mn[:, 5]))
                avg_self = float(np.mean(mn[:, 6]))
                avg_starve = float(np.mean(mn[:, 7]))
                print(f"Gen {gen + 1}/{cfg.GENERATIONS} | "
                      f"BestFood: {b_food:.2f} | BestFit: {b_fit:.3f} | "
                      f"BestTurnEff: {b_te:.2f} | BestTurn: {b_turn:.3f} | "
                      f"BestReach: {float(mn[best_idx][11]):.3f} | "
                      f"BestEShare: {float(mn[best_idx][18]):.2f} | "
                      f"BestConn: {float(mn[best_idx][16]):.2f} | "
                      f"BestStraight: {float(mn[best_idx][17]):.2f} | "
                      f"AvgFood: {avg_food:.2f} | EliteFood: {elite_food:.2f} | "
                      f"Die(W/S/St): {avg_wall:.2f}/{avg_self:.2f}/{avg_starve:.2f} | "
                      f"K2={k2_t} σ_ε≈{cfg.SEL_CV * max(b_food, 1.0) / math.sqrt(K_eff):.2f} | "
                      f"eval {eval_time:.1f}s / evolve {evolve_time:.1f}s")

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
          f"(Food={best_food:.2f}, Seen={best_seen:.1f}, Unseen={best_unseen:.1f})")

    if latest_gen_best_state is not None:
        save_best_model(cfg.LATEST_GEN_BEST_MODEL_PATH, latest_gen_best_state, cfg,
                        latest_gen_best_food, latest_gen_best_seen, latest_gen_best_unseen)
        print(f"最新一代最优模型已保存: {cfg.LATEST_GEN_BEST_MODEL_PATH} "
              f"(Food={latest_gen_best_food:.2f})")

    # 断点保留（7g 教训：完成后删除断点导致跨 run 只能种子注入=准重启）
    save_checkpoint7(cfg.CHECKPOINT_PATH, cfg, cfg.GENERATIONS, pop, history,
                     cum_eval_time, cum_evolve_time,
                     best_state, best_food, best_seen, best_unseen,
                     best_last, best_prox, best_row=best_row)
    save_history_json(cfg.HISTORY_JSON_PATH, history)
    print(f"训练完成，断点已保留: {cfg.CHECKPOINT_PATH} "
          f"(next_gen={cfg.GENERATIONS}，续训需提高 --gens)")

    # ---- 历史曲线 ----
    try:
        plot_history_png(cfg, history)
    except Exception as e:
        print(f"(matplotlib 曲线跳过: {e})")

    # ---- 训练数据与图表落盘后：AutoDL 微信通知（仅此一次）----
    n_gens = len(history.get('gen', []))
    last_fit = history.get('best_fit', [float('nan')])[-1] if history.get('best_fit') else float('nan')
    send_autodl_notify(
        cfg, 'test16a 训练完成',
        f"gens={n_gens} best_food={best_food:.2f} best_fit={last_fit:.2f} "
        f"seen={best_seen:.1f} 用时{(time.perf_counter() - t_program) / 3600:.2f}h。"
        f"产物: {cfg.BEST_MODEL_PATH} / history.json+png / checkpoint")

    print(f"\n--- Best Brain Summary ---")
    st = best_state
    print(f"Input connections active:   {st['M_in'].sum().item()}/{pop.N * pop.O}")
    nz = int((st['rec_w'].float().abs() > 0).sum().item())
    print(f"Internal connections:       {nz}/{pop.K * pop.N} slots non-zero "
          f"(fan-in K={pop.K}, 稠密对照 {pop.N * pop.N})")
    print(f"Output connections active:   {st['M_out'].sum().item()}/{pop.A * pop.N}")
    print(f"tau_e range: [{st['tau_e_init'].min().item():.3f}, {st['tau_e_init'].max().item():.3f}]")


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
# 8. 自检（CRN 确定性 / 变异分布 / 适应度公式）
# ==========================================
def selfcheck(cfg):
    print("=== 自检 0：test12 食物方位+距离编码（相对系 ego）===")
    dev = _resolve_device(cfg)
    sc0 = Config()
    sc0.DEVICE = cfg.DEVICE
    torch.manual_seed(3)
    env0 = BatchedSnakeEnv(sc0, 4, dev)
    env0.reset()
    env0.head = torch.tensor([[5, 5]] * 4, device=dev)
    env0.dir_idx = torch.zeros(4, dtype=torch.long, device=dev)   # 全体朝 E：前/右/后/左 = E/S/W/N
    # 依次：正前 d=3 | 右前方 (vr,vc)=(2,1) d=3 | 右前对角 (4,4) d=8 | 正左 (0,-4)
    env0.food = torch.tensor([[5, 8], [7, 6], [9, 9], [1, 5]], device=dev)
    obs0 = env0._obs40().float()
    K = float(sc0.OBS_FOOD_SCALE)
    ok0 = True
    def _chk(label, got, want, tol=1e-5):
        nonlocal ok0
        good = abs(got - want) < tol
        ok0 = ok0 and good
        print(f"  {label}: {got:.4f} (期望 {want:.4f}) {'OK' if good else 'FAIL'}")
    _chk("正前 方向信号[8]", float(obs0[0, 8]), K)
    _chk("正前 距离倒数[12]", float(obs0[0, 12]), K / 3.0)
    _chk("右前方 信号右[9]", float(obs0[1, 9]), 2.0 * K / 3.0)
    _chk("右前方 信号前[8]", float(obs0[1, 8]), K / 3.0)
    _chk("右前方 倒数右[13]", float(obs0[1, 13]), (2.0 / 3.0) * K / 3.0)
    _chk("右前对角 双信号[8]", float(obs0[2, 8]), 0.5 * K)
    _chk("右前对角 双信号[9]", float(obs0[2, 9]), 0.5 * K)
    _chk("右前对角 倒数[12]", float(obs0[2, 12]), 0.5 * K / 8.0)
    _chk("正左 信号左[11]", float(obs0[3, 11]), K)
    _chk("正左 前无信号[8]", float(obs0[3, 8]), 0.0)
    # 转身不变性：朝 W 时正前(E)食物应从"前"转到"后"通道
    env0.dir_idx = torch.full((4,), 2, dtype=torch.long, device=dev)   # 朝 W
    obs1 = env0._obs40().float()
    _chk("朝W 正前食物→信号后[10]", float(obs1[0, 10]), K)
    _chk("朝W 正前食物→前无信号[8]", float(obs1[0, 8]), 0.0)

    print("=== 自检 0b（E0）：习惯因素1 edge_pref（占据边角）===")
    ok0b = True
    scb = Config(); scb.DEVICE = cfg.DEVICE
    envb = BatchedSnakeEnv(scb, 3, dev)
    envb.reset()
    body = torch.zeros(3, envb.MAXLEN, 2, dtype=torch.long, device=dev)
    for i in range(12):                                    # 全在外环：顶边+右列拐角
        body[0, i] = (torch.tensor([0, i]) if i < 10 else torch.tensor([i - 9, 9]))
    for i, (r, c) in enumerate(((3, 3), (3, 4), (4, 4), (4, 3))):   # 全在内部 2×2
        body[1, i] = torch.tensor([r, c])
    for i in range(8):                                     # 混合：4 环 + 4 内
        body[2, i] = (torch.tensor([0, i]) if i < 4 else torch.tensor([3, i - 2]))
    envb.body = body
    envb.body_len = torch.tensor([12, 4, 8], device=dev)
    envb.head = body[torch.arange(3, device=dev), 0]
    envb.food = torch.tensor([[9, 0], [0, 9], [5, 5]], device=dev)
    es = habit_edge_score(envb).cpu().numpy()
    # 圈层分 16/8/4/2/1：局0 全外环 12×16=192；局1 内部 2×2 → (3,3)L3=2,
    # (3,4)L3=2, (4,4)L4=1, (4,3)L3=2 → 7；局2 混合 4 外环 64 + 内部
    # (3,2)L2=4,(3,3)L3=2,(3,4)L3=2,(3,5)L3=2 → 10 → 74
    want = (192.0, 7.0, 74.0)
    for i in range(3):
        good = abs(es[i] - want[i]) < 1e-5
        ok0b = ok0b and good
        print(f"  局{i} edge_score={es[i]:.1f} (期望 {want[i]:.1f}) "
              f"{'OK' if good else 'FAIL'}")
    sh = habit_edge_share(envb).cpu().numpy()
    want_sh = (192 / (16 * 12), 7 / (16 * 4), 74 / (16 * 8))
    for i in range(3):
        good = abs(sh[i] - want_sh[i]) < 1e-5
        ok0b = ok0b and good
        print(f"  局{i} edge_share={sh[i]:.4f} (期望 {want_sh[i]:.4f}) "
              f"{'OK' if good else 'FAIL'}")
    good = es[0] > es[2] > es[1] and sh[0] > sh[2] > sh[1]
    ok0b = ok0b and good
    print(f"  单调性 全环>混合>全内 {'OK' if good else 'FAIL'}")
    print(f"  E0 因素1 {'OK' if ok0b else 'FAIL'}")

    print("=== 自检 0b2（E0）：习惯因素2 conn_score（连通块倒数，严格含尾）===")
    ok0b2 = True
    envc = BatchedSnakeEnv(scb, 3, dev)
    envc.reset()
    body = torch.zeros(3, envc.MAXLEN, 2, dtype=torch.long, device=dev)
    for i in range(10):                                    # 局0: col4 整列墙 + 尾(9,3)
        body[0, i] = torch.tensor([i, 4])
    body[0, 10] = torch.tensor([9, 3])                     # 尾在左腔：真分割
    for i in range(3):                                     # 局1: 开阔盘面短蛇
        body[1, i] = torch.tensor([0, i])
    for i, (r, c) in enumerate(((0, 1), (1, 1), (1, 0))):  # 局2: 3 格 L 形围死角 (0,0)
        body[2, i] = torch.tensor([r, c])
    envc.body = body
    envc.body_len = torch.tensor([11, 3, 3], device=dev)
    envc.head = torch.tensor([[0, 3], [0, 0], [0, 1]], device=dev)  # 头=body[0] 位置
    envc.food = torch.tensor([[0, 8], [9, 9], [9, 0]], device=dev)
    cs = habit_conn_score(envc).cpu().numpy()
    # 局0 两腔 → 1/2；局1 开阔 → 1；局2 角落 3 格围出 (0,0) → 1/2
    want = (0.5, 1.0, 0.5)
    for i in range(3):
        good = abs(cs[i] - want[i]) < 1e-5
        ok0b2 = ok0b2 and good
        print(f"  局{i} conn_score={cs[i]:.3f} (期望 {want[i]:.1f}) "
              f"{'OK' if good else 'FAIL'}")
    print(f"  E0 因素2 {'OK' if ok0b2 else 'FAIL'}")

    print("=== 自检 0b3（E0）：习惯因素3 straight（少转弯）===")
    ok0b3 = True
    for a1, a2, steps, want_s in ((0.0, 0.0, 200.0, 1.0), (50.0, 0.0, 200.0, 0.75),
                                  (200.0, 0.0, 200.0, 0.0)):
        got = 1.0 - (a1 + a2) / max(steps, 1.0)
        good = abs(got - want_s) < 1e-9
        ok0b3 = ok0b3 and good
        print(f"  转弯{a1 + a2:.0f}/{steps:.0f}步 straight={got:.2f} "
              f"(期望 {want_s:.2f}) {'OK' if good else 'FAIL'}")
    print(f"  E0 因素3 {'OK' if ok0b3 else 'FAIL'}")

    print("=== 自检 0c：单侧转弯判死 ===")
    ok0c = True
    for a1, a2, dead in ((50.0, 0.0, True), (0.0, 50.0, True),
                         (50.0, 3.0, False), (0.5, 0.0, False),
                         (13.0, 13.0, False)):
        ms = np.zeros(18)
        ms[0], ms[3], ms[10], ms[8], ms[9] = 10.0, 100.0, 0.0, a1, a2
        f = _fitness_econ(ms, cfg)
        is_dead = (f == -1e9)
        good = (is_dead == dead)
        ok0c = ok0c and good
        print(f"  左转{a1:.1f}/右转{a2:.1f} → fit={f:>10.2f} "
              f"{'判死' if is_dead else '存活'} (期望{'判死' if dead else '存活'}) "
              f"{'OK' if good else 'FAIL'}")
    print(f"  单侧转弯判死 {'OK' if ok0c else 'FAIL'}")

    print("=== 自检 1：适应度公式（v7 乘法式）===")
    m = np.zeros(19)
    m[0], m[3] = 10.0, 100.0                       # food=10, eff=0.1
    m[17], m[18], m[16] = 0.5, 0.25, 0.5           # straight/edge_share/conn
    f1 = _fitness_econ(m, cfg)
    want1 = 10.0 * (1 + cfg.W_EFF * 0.1 + cfg.W_STRAIGHT * 0.5
                    + cfg.W_EDGE * 0.25) * 0.5
    print(f"  组合值: {f1:.4f} (期望 {want1:.4f}) "
          f"{'OK' if abs(f1 - want1) < 1e-9 else 'FAIL'}")
    m[17], m[18], m[16] = 0.0, 0.0, 1.0            # 全坏习惯+单连通 → 下限
    f2 = _fitness_econ(m, cfg)
    f2_lo = 10.0 * (1 + cfg.W_EFF * 0.1)
    print(f"  下限: {f2:.4f} (期望 {f2_lo:.4f}=food×(1+eff项)) "
          f"{'OK' if abs(f2 - f2_lo) < 1e-9 else 'FAIL'}")
    m[17], m[18], m[16] = 1.0, 1.0, 1.0            # 上限
    f3 = _fitness_econ(m, cfg)
    f3_hi = 10.0 * (1 + cfg.W_EFF * 0.1 + cfg.W_STRAIGHT + cfg.W_EDGE)
    print(f"  上限: {f3:.4f} (期望 {f3_hi:.4f}) "
          f"{'OK' if abs(f3 - f3_hi) < 1e-9 else 'FAIL'}")
    m[0] = 0.0
    f4 = _fitness_econ(m, cfg)
    print(f"  food=0: {f4:.4f} (期望 0) {'OK' if f4 == 0.0 else 'FAIL'}")
    print("=== 自检 1d：simple 最简回退口径 ===")
    sc1d = Config()
    sc1d.FIT_MODE = 'simple'
    sc1d.SIMPLE_EFF_W = 0.3
    ms = np.zeros(19)
    ms[0], ms[3] = 10.0, 100.0
    fs = _fitness_simple(ms, sc1d)
    want_s = 10.0 + 0.3 * 0.1
    good_s = abs(fs - want_s) < 1e-9 and abs(_base_fitness(ms, sc1d) - want_s) < 1e-9
    print(f"  food=10, SL=100: {fs:.4f} (期望 {want_s:.4f}) "
          f"{'OK' if good_s else 'FAIL'} | base 分派一致: {good_s}")

    print("=== 自检 1b（v7 性质）：乘法杠杆随食物放大 ===")
    rng = np.random.default_rng(11)
    ok1b = True
    for food in (10.0, 100.0):
        mA = np.zeros(19); mA[0] = food; mA[3] = food * 10
        mA[17], mA[18], mA[16] = 0.0, 0.0, 1.0     # 坏习惯
        mB = np.zeros(19); mB[0] = food; mB[3] = food * 10
        mB[17], mB[18], mB[16] = 1.0, 1.0, 1.0     # 好习惯
        d = _fitness_econ(mB, cfg) - _fitness_econ(mA, cfg)
        print(f"  food={food:.0f}: 习惯全好-全坏 = {d:+.2f} 分")
    # 同样的习惯差异，100 食时的杠杆应是 10 食时的 10 倍
    m = np.zeros(19)
    m[3] = 1e9
    lever = []
    for food in (10.0, 100.0):
        mA = np.zeros(19); mA[0] = food; mA[3] = food * 10
        mA[17], mA[18], mA[16] = 0.0, 0.0, 1.0
        mB = np.zeros(19); mB[0] = food; mB[3] = food * 10
        mB[17], mB[18], mB[16] = 1.0, 1.0, 1.0
        lever.append(_fitness_econ(mB, cfg) - _fitness_econ(mA, cfg))
    ratio_ok = abs(lever[1] / max(lever[0], 1e-9) - 10.0) < 0.1
    ok1b = ratio_ok
    print(f"  杠杆比(100食/10食) = {lever[1] / max(lever[0], 1e-9):.2f} (期望 10) "
          f"{'OK' if ratio_ok else 'FAIL'}")
    # 下界：fitness ≥ food×conn 恒成立（随机 2000 组）
    ok_lb = True
    for _ in range(2000):
        m2 = np.zeros(19)
        m2[0] = rng.random() * 60 + 1
        m2[3] = m2[0] * rng.random() * 20 + 1
        m2[17], m2[18] = rng.random(), rng.random()
        m2[16] = rng.random() * 0.9 + 0.1
        f = _fitness_econ(m2, cfg)
        if f < m2[0] * m2[16] - 1e-9:
            ok_lb = False
            break
    print(f"  下界 fitness ≥ food×conn（2000 组随机）: {ok_lb} "
          f"{'OK' if ok_lb else 'FAIL'}")
    ok1b = ok1b and ok_lb

    print("=== 自检 1c（v7 性质）：history 分解与公式一致性 ===")
    rng2 = np.random.default_rng(5)
    ok1c = True
    for _ in range(1000):
        row = np.zeros(19)
        row[0] = rng2.random() * 50 + 1
        row[3] = row[0] * rng2.random() * 20 + 1
        row[17], row[18] = rng2.random(), rng2.random()
        row[16] = rng2.random() * 0.9 + 0.1
        c = row[16]
        base = row[0] * c
        parts = (base + base * cfg.W_EFF * (row[0] / max(row[3], 1))
                 + base * cfg.W_STRAIGHT * row[17]
                 + base * cfg.W_EDGE * row[18])
        noconn = row[0] * (1 + cfg.W_EFF * (row[0] / max(row[3], 1))
                           + cfg.W_STRAIGHT * row[17] + cfg.W_EDGE * row[18])
        if (abs(parts - _fitness_econ(row, cfg)) > 1e-6
                or noconn < parts - 1e-9):
            ok1c = False
            break
    print(f"  分解和=公式 & noconn≥总分（1000 组随机）: {ok1c} "
          f"{'OK' if ok1c else 'FAIL'}")

    print("=== 自检 2：变异强度分布 ===")
    dev = _resolve_device(cfg)
    s = sample_mut_scale(cfg, 200000, dev).cpu()
    print(f"  中位数 {s.median():.3f} | 分位 5% {s.quantile(0.05):.3f} / "
          f"95% {s.quantile(0.95):.3f} | P(s>2)={float((s > 2).float().mean()):.4f} "
          f"| P(s<0.5)={float((s < 0.5).float().mean()):.4f} "
          f"| max {s.max():.3f}")

    print("=== 自检 3：弱连接屏蔽生效校验（稀疏 rec_w）===")
    torch.manual_seed(11)
    p3 = GeneStack(cfg, B=16, device=dev)
    p3.random_init()
    frac = float(cfg.WEAK_MASK_FRAC)
    w_before = p3.rec_w.clone()
    p3c = p3[torch.arange(16, device=dev)]           # 拷贝（模拟评估路径）
    before = int((p3c.rec_w.float().abs() > 0).sum().item())
    apply_weak_mask(p3c, cfg)
    after = int((p3c.rec_w.float().abs() > 0).sum().item())
    genes_intact = torch.equal(p3.rec_w, w_before)
    print(f"  非零 rec_w 槽位 {before} → {after}（屏蔽 {1 - after / max(before, 1):.1%}，"
          f"目标 {frac:.0%}）| 原基因栈未动: {genes_intact} "
          f"{'OK' if genes_intact and abs((1 - after / max(before, 1)) - frac) < 0.02 else 'FAIL'}")

    print("=== 自检 4：CRN 确定性（同代同库两跑逐位一致，小种群，含屏蔽）===")
    sc = Config()                     # 独立小配置：自检不该跑全尺寸种群
    sc.POP_SIZE, sc.NUM_COLUMNS = 64, 32
    sc.ELITE_SIZE, sc.STAGE1_EPS, sc.EVAL_EPISODES, sc.STAGE2_KEEP = 16, 2, 3, 32
    sc.MAX_STEPS, sc.EVAL_BATCH, sc.USE_FP16 = 400, 64, cfg.USE_FP16
    sc.DEVICE = cfg.DEVICE
    sc.WEAK_MASK_FRAC = cfg.WEAK_MASK_FRAC
    torch.manual_seed(7)
    pop = GeneStack(sc, device=_resolve_device(sc))
    pop.random_init()
    pop.fp16()
    m_a, order_a = evaluate_population_gpu(pop, sc, gen=0)
    m_b, order_b = evaluate_population_gpu(pop, sc, gen=0)
    same = torch.equal(m_a, m_b) and order_a == order_b
    print(f"  metrics 逐位一致: {torch.equal(m_a, m_b)} | order 一致: {order_a == order_b} "
          f"{'OK' if same else 'FAIL'}")
    m_c, _ = evaluate_population_gpu(pop, sc, gen=1)
    diff = float((m_a[:, 0] - m_c[:, 0]).abs().max())
    print(f"  换代换库后指标改变（应>0）: max|Δfood|={diff:.3f} "
          f"{'OK' if diff > 0 else 'FAIL'}")

    print("=== 自检 5：稀疏 rec 前向 vs 稠密参考逐位等价 ===")
    ok5 = True
    sc5 = Config()
    sc5.NUM_COLUMNS, sc5.REC_FANIN = 32, 16
    sc5.OBS_DIM, sc5.ACTION_DIM = 40, 3
    sc5.DEVICE = cfg.DEVICE
    sc5.USE_FP16 = False
    torch.manual_seed(23)
    p5 = GeneStack(sc5, B=3, device=_resolve_device(sc5))
    p5.random_init()
    p5.refresh_eff()
    # 稀疏 → 稠密参考（scatter_add：重复源=权重叠加，与 gather-求和语义一致）
    Wd = torch.zeros(3, 32, 32, device=p5.device, dtype=p5.rec_w.dtype)
    Wd.scatter_add_(2, p5.rec_idx, p5.rec_w)
    E0 = torch.randn(3, 32, device=p5.device)
    I0 = torch.rand(3, 32, device=p5.device)
    st0 = torch.rand(3, 32, device=p5.device)
    obs0 = torch.rand(3, 40, device=p5.device)
    press0 = torch.zeros(3, device=p5.device)
    # gather 前向（本实现）
    lg_s, E_s, I_s, st_s = forward_batch(p5, obs0, E0, I0, st0, press0, sc5)
    # 稠密参考前向（逐行复刻公式，rec 用 bmm）
    ext = torch.bmm(p5.W_in_eff, obs0.unsqueeze(-1)).squeeze(-1)
    rec_d = torch.bmm(Wd, E0.unsqueeze(-1)).squeeze(-1)
    stt = sc5.SHORT_TERM_DECAY * st0 + (1 - sc5.SHORT_TERM_DECAY) * E0
    tau = (p5.tau_e + sc5.SHORT_TERM_GAIN * stt).clamp(sc5.TAU_E_MIN, sc5.TAU_E_MAX)
    E_r = torch.sigmoid(ext + rec_d + tau * E0 - p5.w_ei.clamp(0, 6) * I0)
    I_r = torch.sigmoid(p5.w_ie.clamp(0, 6) * E_r)
    lg_r = torch.bmm(p5.W_out_eff, E_r.unsqueeze(-1)).squeeze(-1) + p5.b_out
    d_rec = float((rec_d - (torch.gather(E0.unsqueeze(-1).expand(3, 32, p5.K), 1, p5.rec_idx)
                            * p5.rec_w).sum(-1)).abs().max())
    d_E = float((E_r - E_s).abs().max())
    d_lg = float((lg_r - lg_s).abs().max())
    ok5 = (d_rec < 1e-5) and (d_E < 1e-5) and (d_lg < 1e-5)
    print(f"  rec 项差 {d_rec:.2e} | E_new 差 {d_E:.2e} | logits 差 {d_lg:.2e} "
          f"{'OK' if ok5 else 'FAIL'}")

    print("=== 自检 6：重连变异后无自连 & 个体状态往返一致 ===")
    ok6 = True
    sc6 = Config()
    sc6.NUM_COLUMNS, sc6.REC_FANIN = 48, 16
    sc6.OBS_DIM, sc6.ACTION_DIM = 40, 3
    sc6.POP_SIZE, sc6.ELITE_SIZE = 16, 4
    sc6.STAGE2_KEEP = 8
    sc6.CYCLE_PATTERN = [('G2', 'G1')]
    sc6.TOPOLOGY_MUT_PROB, sc6.MUT_RATE = 1.0, 1.0     # 强制全体重连
    sc6.DEVICE = cfg.DEVICE
    sc6.USE_FP16 = False
    p6 = GeneStack(sc6, device=_resolve_device(sc6))
    p6.random_init()
    fake_metrics = torch.zeros(16, 19)
    p6n = evolve_topology_gpu(p6, fake_metrics, sc6, gen=1)   # gen=1 → G1 活跃（重连生效）
    ar6 = torch.arange(48, device=p6n.device).view(1, 48, 1)
    n_self = int((p6n.rec_idx == ar6).sum().item())
    oob = int(((p6n.rec_idx < 0) | (p6n.rec_idx >= 48)).sum().item())
    good6 = (n_self == 0 and oob == 0)
    ok6 = ok6 and good6
    print(f"  全强度重连后: 自连 {n_self} 个 / 越界 {oob} 个（期望 0/0）"
          f"{'OK' if good6 else 'FAIL'}")
    st_a = p6n.individual_state(0, use_half=False)
    p6b = GeneStack(sc6, B=1, device=_resolve_device(sc6))
    p6b.random_init()
    p6b.set_individual_from_state(0, st_a)
    dev6 = p6b.device
    same = (torch.equal(p6b.rec_idx[0], st_a['rec_idx'].long().to(dev6))
            and torch.equal(p6b.rec_w[0], st_a['rec_w'].float().to(dev6)))
    print(f"  individual_state 往返一致: {same} {'OK' if same else 'FAIL'}")
    ok6 = ok6 and same

    print("=== 自检 7：LCB 选择键（v8）跨 K 行为 ===")
    ok7 = True
    sc7 = Config()
    sc7.FIT_MODE = 'econ'
    sc7.SEL_LCB_LAMBDA = 1.0
    sc7.SEL_CV = 0.12
    # (a) 同 K 集合内：LCB 等比不平（同 food 同 K → 排序与 base 一致）
    mA, mB = np.zeros(19), np.zeros(19)
    mA[0], mA[3] = 20.0, 200.0
    mB[0], mB[3] = 22.0, 240.0
    kA14, kB14 = _sel_key(mA, sc7, 14), _sel_key(mB, sc7, 14)
    ord_same = (kB14 > kA14) == (_fitness_econ(mB, sc7) > _fitness_econ(mA, sc7))
    ok7 = ok7 and ord_same
    print(f"  同K排序保持: {'OK' if ord_same else 'FAIL'}")
    # (b) 跨 K：同 food 下低 K（噪声大）被更强折价 → 侥幸低K分不再占优
    kA10 = _sel_key(mA, sc7, 10)
    lcb14 = 1.0 * 0.12 * 20.0 / math.sqrt(14)
    lcb10 = 1.0 * 0.12 * 20.0 / math.sqrt(10)
    monot = abs((kA14 - kA10) - (lcb10 - lcb14)) < 1e-9 and lcb10 > lcb14
    ok7 = ok7 and monot
    print(f"  低K折价更大: K=10 折 {lcb10:.3f} > K=14 折 {lcb14:.3f} "
          f"{'OK' if monot else 'FAIL'}")
    # (c) λ=0 / food=0 → 退回 base
    sc7.SEL_LCB_LAMBDA = 0.0
    off = _sel_key(mA, sc7, 10) == _fitness_econ(mA, sc7)
    sc7.SEL_LCB_LAMBDA = 1.0
    mA0 = np.zeros(19)
    z0 = _sel_key(mA0, sc7, 10) == 0.0 and _fitness_econ(mA0, sc7) == 0.0
    ok7 = ok7 and off and z0
    print(f"  λ=0 退回 base: {off} | food=0 恒 0: {z0} "
          f"{'OK' if off and z0 else 'FAIL'}")
    # (d) 配置自洽：ELITE ≤ STAGE2_KEEP（幸存者即亲本）
    cfg_ok = cfg.ELITE_SIZE <= cfg.STAGE2_KEEP
    ok7 = ok7 and cfg_ok
    print(f"  ELITE({cfg.ELITE_SIZE}) ≤ STAGE2_KEEP({cfg.STAGE2_KEEP}): {cfg_ok} "
          f"{'OK' if cfg_ok else 'FAIL'}")
    print("=== 自检完成 ===")


# ==========================================
# 9. 入口
# ==========================================
def make_smoke_config():
    cfg = Config()
    cfg.POP_SIZE = 16
    cfg.NUM_COLUMNS = 24
    cfg.REC_FANIN = 8
    cfg.OBS_DIM = 40
    cfg.ACTION_DIM = 3
    cfg.GENERATIONS = 3
    cfg.ELITE_SIZE = 6
    cfg.STAGE1_EPS = 1
    cfg.EVAL_EPISODES = 2
    cfg.STAGE2_EPS = 2
    cfg.STAGE2_KEEP = 8
    cfg.MAX_STEPS = 60
    cfg.FRAME_RATE = 2
    cfg.CHECKPOINT_INTERVAL = 2
    cfg.CHECKPOINT_PATH = 'test16a_smoke_checkpoint.pth'
    cfg.BEST_MODEL_PATH = 'test16a_smoke_best.pth'
    cfg.LATEST_GEN_BEST_MODEL_PATH = 'test16a_smoke_latest_gen_best.pth'
    cfg.HISTORY_JSON_PATH = 'test16a_smoke_history.json'
    cfg.SEED_FROM_BEST = False
    cfg.EVAL_BATCH = 16
    cfg.PRINT_HISTORY_EVERY = 1
    return cfg


def main():
    ap = argparse.ArgumentParser(
        description='test16a — 亲本放宽 1/2 + LCB 选择键 + 自适应 K2（兼容 test16 种群）')
    ap.add_argument('--smoke', action='store_true', help='小规模快速自检')
    ap.add_argument('--selfcheck', action='store_true',
                    help='只跑自检（适应度/变异分布/CRN 确定性/稀疏等价性），不训练')
    ap.add_argument('--fanin', type=int, default=None,
                    help='每个神经元循环输入槽位数 K（默认 16，须 ≤ N-1）')
    ap.add_argument('--elite', type=int, default=None,
                    help='亲本数（test16a 默认 POP/2=1024，放宽选择）')
    ap.add_argument('--sel-lcb', type=float, default=None,
                    help='LCB 选择惩罚 λ（默认 1.0；0=退回纯 v7 适应度）')
    ap.add_argument('--sel-cv', type=float, default=None,
                    help='单局分数变异系数（默认 0.12，benchmark 标定）')
    ap.add_argument('--res-target', type=float, default=None,
                    help='自适应 K2 的目标分辨率 δ（食，默认 1.5）')
    ap.add_argument('--k2-max', type=int, default=None,
                    help='自适应 K2 上限（默认 32）')
    ap.add_argument('--no-k2-adapt', action='store_true',
                    help='关闭自适应 K2（固定 K2=EVAL_EPISODES）')
    ap.add_argument('--gens', type=int, default=None)
    ap.add_argument('--pop', type=int, default=None)
    ap.add_argument('--columns', type=int, default=None)
    ap.add_argument('--episodes', type=int, default=None, help='阶段2 局数 K2')
    ap.add_argument('--max-steps', type=int, default=None)
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--fit-mode', type=str, default=None,
                    choices=['econ', 'simple', 'tuple'],
                    help='econ=v7 乘法式（默认）| simple=食物+k·效率最简回退 | tuple=旧')
    ap.add_argument('--eff-weight', type=float, default=None,
                    help='econ 模式效率因子 W_EFF')
    ap.add_argument('--simple-eff-w', type=float, default=None,
                    help='simple 模式效率系数 k（默认 0.3，test7b 口径）')
    ap.add_argument('--turn-eff-w', type=float, default=None,
                    help='转弯效率权重（默认 3.0，解法器基准校准）')
    ap.add_argument('--turn-eff-cap', type=float, default=None,
                    help='转弯效率饱和上限（默认 4，密度 0.25 饱和）')
    ap.add_argument('--turn-eff-mode', type=str, default=None,
                    choices=['ratio', 'tpf'],
                    help='转弯效率口径：ratio=SL/TL（默认）| tpf=每食物转弯数（备选）')
    ap.add_argument('--weak-mask-frac', type=float, default=None,
                    help='评估期 rec_w 弱连接屏蔽比例（默认 0=关）')
    ap.add_argument('--te-elite', type=int, default=None,
                    help='te-配额精英数（默认 0=关；行为学 B5 对策）')
    ap.add_argument('--w-straight', type=float, default=None,
                    help='v7 少转弯因子 W_STRAIGHT（默认 0.5）')
    ap.add_argument('--w-edge', type=float, default=None,
                    help='v7 边角因子 W_EDGE（默认 0.5）')
    ap.add_argument('--imitation-w', type=float, default=None,
                    help='模仿引导权重（默认 0=关；建议 2.0）')
    ap.add_argument('--island-threshold', type=float, default=None,
                    help='孤岛判定阈值：min_reach<阈值×自由格 触发（默认 0.3）')
    ap.add_argument('--no-one-sided-death', action='store_true',
                    help='关闭单侧转弯判死（默认开启：淘汰单向绕圈形态）')
    ap.add_argument('--stage1-eps', type=int, default=None,
                    help='阶段1 局数 K1（默认 3）')
    ap.add_argument('--stage2-eps', type=int, default=None,
                    help='阶段2 局数 K2（默认 10）')
    ap.add_argument('--stage2-keep', type=int, default=None,
                    help='阶段1 后幸存数（默认 410，须 ≥ 精英数）')
    ap.add_argument('--no-crn', action='store_true', help='关闭公共随机数（回退随机库）')
    ap.add_argument('--crn-seed', type=int, default=None)
    ap.add_argument('--mut-sigma', type=float, default=None,
                    help='变异强度分布弥散（lognormal σ_ln，默认 0.4）')
    ap.add_argument('--mut-dist', type=str, default=None, choices=['lognormal', 'normal'])
    ap.add_argument('--turn-gain', type=float, default=None,
                    help='转向疲劳增益（默认 0=彻底关闭）')
    ap.add_argument('--turn-decay', type=float, default=None)
    ap.add_argument('--starve-slope', type=float, default=None)
    ap.add_argument('--eval-batch', type=int, default=None)
    ap.add_argument('--seed-model', type=str, default=None,
                    help='指定种子模型路径（单个体注入随机种群，覆盖全种群续训）')
    ap.add_argument('--resume-pop', type=str, default=None,
                    help='从任意全种群 checkpoint 导入并续训（复制为本 run 的断点后 AUTO_RESUME）')
    ap.add_argument('--name', type=str, default=None,
                    help='本 run 文件名前缀（checkpoint/best/latest/history 独立，默认 econ）')
    ap.add_argument('--migrate-from', dest='migrate_from', type=str, default=None,
                    help='test12 旧32维单脑迁移：M_in/W_in 列扩展后全种群注入（新列 INIT_DENSITY/σ=0.1）')
    ap.add_argument('--no-new-obs', dest='no_new_obs', action='store_true',
                    help='新 8 通道置零（A0 对照，与 test12 严格同数学）')
    ap.add_argument('--play', action='store_true')
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
    if args.sel_lcb is not None:
        cfg.SEL_LCB_LAMBDA = args.sel_lcb
    if args.sel_cv is not None:
        cfg.SEL_CV = args.sel_cv
    if args.res_target is not None:
        cfg.RES_TARGET = args.res_target
    if args.k2_max is not None:
        cfg.K2_MAX = args.k2_max
    if args.no_k2_adapt:
        cfg.K2_ADAPTIVE = False
    if cfg.REC_FANIN >= cfg.NUM_COLUMNS:
        sys.exit(f'[配置错误] REC_FANIN({cfg.REC_FANIN}) 须 < NUM_COLUMNS({cfg.NUM_COLUMNS})')
    if args.episodes:
        cfg.EVAL_EPISODES = args.episodes
    if args.max_steps:
        cfg.MAX_STEPS = args.max_steps
    if args.device:
        cfg.DEVICE = args.device
    if args.fit_mode:
        cfg.FIT_MODE = args.fit_mode
        if args.fit_mode == 'simple':
            cfg.FITNESS_VERSION = 9    # 口径不同，best 追踪独立（9 = food+k·eff + LCB）
        if not args.smoke:
            arm = {'econ': 'econ', 'simple': 'simp', 'tuple': 'tup'}[args.fit_mode]
            cfg.CHECKPOINT_PATH = f'test16a_{arm}_checkpoint.pth'
            cfg.BEST_MODEL_PATH = f'test16a_{arm}_best_model.pth'
            cfg.LATEST_GEN_BEST_MODEL_PATH = f'test16a_{arm}_latest_gen_best.pth'
            cfg.HISTORY_JSON_PATH = f'test16a_{arm}_history.json'
    if args.eff_weight is not None:
        cfg.W_EFF = args.eff_weight
    if args.simple_eff_w is not None:
        cfg.SIMPLE_EFF_W = args.simple_eff_w
    if args.w_straight is not None:
        cfg.W_STRAIGHT = args.w_straight
    if args.w_edge is not None:
        cfg.W_EDGE = args.w_edge
    if args.turn_eff_w is not None:
        cfg.TURN_EFF_W = args.turn_eff_w
    if args.turn_eff_cap is not None:
        cfg.TURN_EFF_CAP = args.turn_eff_cap
    if args.turn_eff_mode:
        cfg.TURN_EFF_MODE = args.turn_eff_mode
    if args.weak_mask_frac is not None:
        cfg.WEAK_MASK_FRAC = args.weak_mask_frac
    if args.te_elite is not None:
        cfg.TE_ELITE = args.te_elite
    if args.imitation_w is not None:
        cfg.IMITATION_W = args.imitation_w
    if args.island_threshold is not None:
        cfg.ISLAND_THRESHOLD = args.island_threshold
    if args.no_one_sided_death:
        cfg.ONE_SIDED_TURN_DEATH = False
    if args.name:
        cfg.CHECKPOINT_PATH = f'test16a_{args.name}_checkpoint.pth'
        cfg.BEST_MODEL_PATH = f'test16a_{args.name}_best_model.pth'
        cfg.LATEST_GEN_BEST_MODEL_PATH = f'test16a_{args.name}_latest_gen_best.pth'
        cfg.HISTORY_JSON_PATH = f'test16a_{args.name}_history.json'
    if args.resume_pop:
        payload = torch.load(args.resume_pop, map_location='cpu', weights_only=False)
        saved = payload.get('config', {})
        if saved and (saved.get('NUM_COLUMNS') != cfg.NUM_COLUMNS or
                      saved.get('OBS_DIM') != cfg.OBS_DIM or
                      saved.get('ACTION_DIM') != cfg.ACTION_DIM):
            sys.exit(f'[错误] resume-pop {args.resume_pop} 与当前维度不匹配')
        if saved and saved.get('BRAIN_VERSION') != getattr(cfg, 'BRAIN_VERSION', 'sparse1'):
            sys.exit(f'[错误] resume-pop {args.resume_pop} 基因组版本不符 '
                     f'({saved.get("BRAIN_VERSION", "旧格式无版本")} != '
                     f'{cfg.BRAIN_VERSION})——test16 稀疏固定扇入基因组与 '
                     f'test15 及更早稠密断点互斥，无法导入')
        if saved and saved.get('OBS_ENC_VERSION', '32proj') != getattr(cfg, 'OBS_ENC_VERSION', '32proj'):
            sys.exit(f'[错误] resume-pop {args.resume_pop} 观测编码不符 '
                     f'({saved.get("OBS_ENC_VERSION")} != {cfg.OBS_ENC_VERSION})')
        parent = os.path.dirname(os.path.abspath(cfg.CHECKPOINT_PATH))
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(payload, cfg.CHECKPOINT_PATH)
        print(f"[resume-pop] 已导入 {args.resume_pop} "
              f"(next_gen={payload.get('next_gen')}) -> {cfg.CHECKPOINT_PATH}")
    if args.stage1_eps is not None:
        cfg.STAGE1_EPS = args.stage1_eps
    if args.stage2_eps is not None:
        cfg.STAGE2_EPS = args.stage2_eps
        cfg.EVAL_EPISODES = args.stage2_eps
    if args.stage2_keep is not None:
        cfg.STAGE2_KEEP = args.stage2_keep
    if args.no_crn:
        cfg.USE_CRN = False
    if args.crn_seed is not None:
        cfg.CRN_SEED = args.crn_seed
    if args.mut_sigma is not None:
        cfg.MUT_SCALE_SIGMA = args.mut_sigma
    if args.mut_dist:
        cfg.MUT_SCALE_DIST = args.mut_dist
    if args.turn_gain is not None:
        cfg.FATIGUE_TURN_GAIN = args.turn_gain
    if args.turn_decay is not None:
        cfg.FATIGUE_TURN_DECAY = args.turn_decay
    if args.starve_slope is not None:
        cfg.STARVE_SLOPE = args.starve_slope
    if args.eval_batch is not None:
        cfg.EVAL_BATCH = args.eval_batch
    if args.seed_model:
        cfg.SEED_FROM_BEST = True
        cfg.SEED_MODEL_PATH = args.seed_model
    if args.migrate_from:
        sys.exit('[错误] test16（稀疏固定扇入）不支持 --migrate-from 稠密血统迁移，'
                 '请从零训练')
    if args.no_new_obs:
        cfg.OBS_NEW_ENABLED = False

    # K2 基准同步：STAGE2_EPS 为权威口径（修复旧版"改 Config 的 K2 无效"问题）
    if cfg.STAGE2_EPS != cfg.EVAL_EPISODES:
        print(f"[K2 基准同步] EVAL_EPISODES {cfg.EVAL_EPISODES} -> STAGE2_EPS "
              f"{cfg.STAGE2_EPS}（STAGE2_EPS 为权威口径）")
        cfg.EVAL_EPISODES = cfg.STAGE2_EPS
    if cfg.ELITE_SIZE >= cfg.POP_SIZE:
        sys.exit(f'[配置错误] ELITE_SIZE({cfg.ELITE_SIZE}) 须 < POP_SIZE({cfg.POP_SIZE})'
                 f'——相等时子代数为 0，种群永不演化')
    if cfg.STAGE2_KEEP > cfg.POP_SIZE:
        print(f"[提示] STAGE2_KEEP({cfg.STAGE2_KEEP}) > POP_SIZE({cfg.POP_SIZE})，"
              f"阶段2 按全体幸存处理")

    if cfg.STAGE2_KEEP < cfg.ELITE_SIZE:
        sys.exit(f'[配置错误] STAGE2_KEEP({cfg.STAGE2_KEEP}) 须 ≥ ELITE_SIZE({cfg.ELITE_SIZE})')

    if args.selfcheck:
        selfcheck(cfg)
        return
    if args.play:
        play_best(cfg)
        return

    try:
        run_training(cfg)
    except Exception as e:
        # 云端租卡场景：异常中断即推送告警，避免实例空转计费（AutoDL 官方建议场景）
        send_autodl_notify(cfg, 'test16a 训练异常退出',
                           f'{type(e).__name__}: {str(e)[:150]}')
        raise


if __name__ == '__main__':
    main()
