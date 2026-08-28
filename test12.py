# ==========================================
# test12.py —— 基于 test7h 的改版：绝对方位食物编码 + 孤岛惩罚
#
# 相对 test7h 的两处改动：
#  1. 食物感知重写（obs[8:16]，保持 32 维，OBS_ENC_VERSION='32ego1'）：
#     - [8:12] 4 相对方位信号 ×K（前/右/后/左，随头转）：sig=clamp(û·dir,0,1)×K。
#       食物恰在该方向直线上 → 信号 ≈K；斜 45° → 相邻两方向各 ≈0.707K；
#       与 7h 扇区同为自体系、同输入量级（对准 ≈K）；
#     - [12:16] 4 相对方位距离倒数：sig×K/曼哈顿距离。与方位信号联合可精确
#       恢复食物相对向量（方位+距离双通道）。替换原 8 扇区曼哈顿投影。
#     编码演进（对照实验定位，experiments/diag_test12_vs_7h.py）：
#       v1 绝对系+原始 sig → 卡 1.0；v2 绝对系+sig×K → 仍卡 1.0（量级无关）；
#       v3(本版) 相对系——绝对系要求网络先学会 绝对方位⊗头朝向 绑定，
#       随机初网络零初始相关、选择无梯度。'abs' 帧保留为 Config 开关。
#  2. 孤岛惩罚（FITNESS_VERSION 3）：过程中（每次吃食采样点）蛇头可达
#     空间 < ISLAND_THRESHOLD×(总空间−蛇长) 记该局触发，metrics 列 [13]
#     min_reach、[14] 局触发率；适应度按触发率线性折减
#     fit ×= 1−(1−ISLAND_PENALTY)·rate（=局级 ×pen 跨局平均的一阶形式；
#     v1 的'任一局触发即整体 ×0.1'实测与食物数正相关、反向压选择，已废）。
#     动机：7h 系行为学结论"主死因=空间挤压自撞"，排除分割空间与进入
#     孤岛的倾向。复用 reach_ratio 洪泛填充（分母即自由格数）。
#  3. checkpoint 新增 OBS_ENC_VERSION 校验：7h 旧 checkpoint 观测编码
#     不同但维度相同，resume-pop/seed-model 时按编码版本拒绝，须用新 run。
#
# ==========================================
# 以下为 test7h 原始说明（CRN 精确筛选 + 两阶段淘汰 + 类正态变异，基于 test7g）
#
# v2（解法器基准校准，experiments/solver_reference.py）：
#  - 适应度转弯效率升为一等力量：W 0.5→3.0、CAP 8→4（ratio 口径 SL/TL）。
#    依据：同库同食数配对下，有序解法器比 7h/7b 模型高 +1.6 分（> 单食边际
#    1.25 → 渐进转型可攀爬）；旧参数仅 +0.15 分形同虚设。CAP=4 ⇔ 密度 0.25
#    饱和 → 长直段折叠拿满 3 分不被压制（7d 乘法计价压折叠的教训不复发）。
#  - 评估期弱连接屏蔽 W_rec×20%（训练=部署同口径，实测 +1.8 分）。
#  - FITNESS_VERSION：断点跨版本续训自动重置 best 追踪。
#  - 重要实验事实：纯哈密顿回路跟随 40/40 全部饿死（均 0.4 食）——饿死钟
#    （3·len+20）在蛇长<27 时短于回路平均遇食距离 ~50 步，环境规则本身
#    禁止纯有序策略，早期强制抄近路；有序参考 = 回路+安全捷径（密度 0.30，
#    te 3.2，每食物 9 步）。
#
# test7g 40 平台诊断结论（experiments/diagnose_test7g_plateau.py，40 局实测）：
#  - 适应度排序无过错：τ(现适应度, food)=0.99、反转 0 例；7b 脑在本适应度下
#    可得 ~60 分而种群困在 ~40 → 瓶颈在选择噪声与血统，不在公式；
#  - 选择噪声淹没信号：个体内 10 局 SEM=2.18 vs 精英间 σ=3.34，上代第 1 名
#    重评 28.7 / 第 5 名 39.6——排序大半凭食物运气；2048×10 局极值统计虚增
#    best ≈+7 分（"40.9 平台"真实水平 ≈33）；
#  - avg 自 gen23 冻结 25 代：固定小变异在收敛种群上只能随机游走。
#
# 相对 test7g 的改动：
#  1. 适应度重写（econ）：fitness = food + 0.3·food/steps_last
#       + TURN_EFF_W·min(steps_last/max(turns_last,1), CAP)/CAP
#     - reach 项删除（它塑形出"守干净空间饿死"：55% 局饿死于口袋食物旁，
#       序号>31 段 26-33% 食物对头部不可达仍不吃）；avg_reach 仅保留观测。
#     - 转弯效率 = 最后一食为止步数/同窗口转弯数（整局口径下末食后直行
#       游荡会白拿无穷高效率）；CAP=8 饱和，项幅 ≤0.5 分（tie-breaker：
#       实测 7b 穿行密度 0.88 与平台脑游荡 0.80 几乎相同，转弯项不可能区分
#       二者，任何 ≥12 分的权重都会把 23 分直线脑排到 50 分 7b 之上）。
#  2. CRN 公共随机数（种子序列固定）：每代每局预生成食物流 bank（初始朝向/
#     初始食物/逐次落子候选），全种群全分块共用；代间/阶段间换新种子。
#     政策为确定性 argmax、环境给定 bank 后完全确定 → 同代个体分数无任何
#     食物运气差异，排序翻转与 best 极值虚增同时消失。
#  3. 两阶段淘汰：阶段1 全种群 ×STAGE1_EPS 局（CRN 同库精确可比）→ 保前
#     STAGE2_KEEP 名；阶段2 幸存者 ×STAGE2_EPS 局新库，按累计 (K1+K2) 局
#     定精英。默认 2048×3 + 410×10 = -50% 评估量，精英判据 10→13 局。
#  4. 类正态变异强度：每子代抽 s ~ LogNormal(0, MUT_SIGMA) clip [0.25,4]
#     （中位数 1=现行强度；P(s>2)≈2.4% → 每代 ~43 个大变异），缩放该子代
#     全部变异算子（掩码翻转率≤0.5 / 拓扑触发率≤0.5 / 权重扰动比例≤1.0
#     及各 std）；精英不变异。
#  5. 疲劳彻底关闭（FATIGUE_TURN_GAIN=0.0，留开关）；单侧转弯判死保持关闭。
#  6. 训练完成不再删除断点（7g 教训：跨 run 只能种子注入=准重启）；
#     逐代 history 落盘 JSON。
# 观测：曼哈顿度量固定（欧氏对角偏置 √2 已验证为错误启发，不再提供回退）。
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

    # --- 类正态变异强度（每子代因子 s，缩放其全部变异算子）---
    MUT_SCALE_DIST = 'lognormal'   # 'lognormal' | 'normal'
    MUT_SCALE_SIGMA = 0.4          # lognormal: σ_ln；normal: s~N(1,σ) clip
    MUT_SCALE_MIN = 0.25
    MUT_SCALE_MAX = 4.0

    # --- 无激素 EI-RNN（兼容字段）---
    TRAIN_HORMONE_NET = False
    HORMONE_NET_HIDDEN = 32

    # --- 环境参数 ---
    GRID_SIZE = 10
    EVAL_EPISODES = 10          # 阶段2 局数 K2（累计 K1+K2 定精英）
    MAX_STEPS = 100000

    # --- 脑结构参数 ---
    NUM_COLUMNS = 256
    OBS_MODE = '32proj'
    OBS_DIM = 24 if OBS_MODE == '24' else 32
    ACTION_DIM = 3
    INIT_DENSITY = 0.15

    # --- 适应度模式：'econ'=三项和式（本版主模式）| 'tuple'=元组字典序 ---
    FIT_MODE = 'econ'

    # --- 适应度版本（公式变更时 +1；断点版本不一致则重置 best 追踪）---
    FITNESS_VERSION = 3

    # --- 孤岛惩罚（test12）：按各局触发率线性折减 fit×(1−(1−pen)·rate) ---
    # reach_ratio 分母=自由格数（总空间−蛇长），即"蛇头可达空间 < 30% 自由格"。
    # 采样点=每次吃食（复用 avg_reach 统计点，零额外洪泛成本）。rate=1（全程
    # 触发）→ ×pen；rate=0 → 不罚。pen=1.0 关闭。
    ISLAND_THRESHOLD = 0.3
    ISLAND_PENALTY = 0.1

    # --- 观测编码版本（'32ego1'=test12 相对方位+距离通道；'32proj'=test7h 扇区投影）---
    # checkpoint 校验用：7h 与 12 维度同为 32，仅凭 OBS_DIM 无法区分。
    # v1('32abs') 绝对系 sig 驱动不足；v2('32abs2') 绝对系 ×K 仍卡 1.0；
    # 本版改相对系（帧对照实证：绝对系要求网络先学会 绝对方位⊗头朝向 绑定，
    # 随机初网络零初始相关、选择无梯度）。改 OBS_FOOD_FRAME 须换新 run。
    OBS_ENC_VERSION = '32ego1'

    # --- 食物方位参照系：'ego'=前/右/后/左（默认，随头转；与 7h 扇区同系）---
    #   | 'abs'=绝对 E/S/W/N（原始需求，保留开关；实测从零训练选择无梯度）
    OBS_FOOD_FRAME = 'ego'

    # --- econ：fitness = food + W_e·eff + W_t·min(SL/TL, CAP)/CAP ---
    FOOD_EFF_WEIGHT = 0.3   # 吃子效率权重（7b 验证量级，防固定回路退化）
    # 转弯效率（解法器基准实测校准，results/test7h_solver_reference.json）：
    #   有序解法器 te=SL/TL≈3.2 / 密度 0.30；模型 te≈1.13 / 密度 0.88。
    #   W=3, CAP=4 → 同食数下解法器比模型高 ~1.6 分 > 单食边际 1.25 分
    #   → 渐进转型的变异体（少 1 食但路径有序）仍胜出，选择可攀爬；
    #   旧参数 W=0.5/CAP=8 仅 +0.15 分（形同虚设）。
    #   CAP=4 ⇔ 密度 0.25（长直段折返）即饱和 → 折叠不受压制；
    #   后期"多转弯换一食"的损失 ≤0.3 分，不可能阻断吃食。
    TURN_EFF_W = 3.0        # 转弯效率权重（项幅 ≤3 分）
    TURN_EFF_CAP = 4.0      # 转弯效率饱和上限（SL/TL ≥ 4 后不再加分）
    TURN_EFF_MODE = 'ratio'  # 'ratio'=SL/TL | 'tpf'=每食物转弯数（备选口径）

    # --- 评估期弱连接屏蔽（H2，训练=部署同口径；基因不动）---
    # 实测屏蔽 W_rec 最弱 20% 在同库配对下 +1.8 分（10% 反而 −1.4，
    # 30% +1.1，40% −0.8）：弱内连接是 ~2 分的内噪损耗。进化评估即用
    # 屏蔽后的成绩 → 训练产物=部署形态。
    WEAK_MASK_FRAC = 0.20

    # --- te-配额精英（行为学研究 B5 对策：精英 te≥2 占比 0% = 选择无料可选）---
    # 每代从全种群按 te=SL/TL（截断 CAP、要求 food>0）取 TE_ELITE 名插入精英
    # 尾部（顶替适应度排名最末的精英），保证稀有有序变异体进入繁殖池。
    TE_ELITE = 0                # 0=关（默认）；建议 24-32

    # --- 模仿引导（混沌盆地突破：稠密行为梯度，教师=固定回路单调序解法器）---
    # fitness += IMITATION_W·(1−mismatch)，mismatch=存活步中与教师动作不一致
    # 的比例（指标列[12]）。教师逻辑向量化，评估零额外扫描成本。
    # 行为学依据：B3X 证明策略与身体构型共适应、中途换风格必死 → 模仿须从
    # 出生塑形；te-配额 28 代证明精英池内重组无法转型 → 需要本稠密梯度。
    IMITATION_W = 0.0           # 0=关（默认）；建议验证 2.0

    # --- 单侧转弯判死（保持关闭：早期随机个体普遍摇头，判罚干扰初期筛选）---
    ONE_SIDED_TURN_DEATH = False

    # --- 饿死斜率：steps_wo_food > STARVE_SLOPE*len + 20 ---
    STARVE_SLOPE = 3.0

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
    CYCLE_PATTERN = [('G2', 'G1', 'G3')]

    # --- CRN 公共随机数（筛选种子序列固定）---
    USE_CRN = True
    CRN_SEED = 20260827
    CRN_DRAW = 4096             # 每局预生成落子候选流长度

    # --- 两阶段淘汰 ---
    STAGE1_EPS = 3              # 阶段1 局数 K1（全种群，CRN 同库）
    STAGE2_KEEP = 410           # 阶段1 后幸存数（须 ≥ ELITE_SIZE）
    STAGE2_EPS = 10             # 阶段2 局数 K2（新库；累计 K1+K2 定精英）

    # --- GPU 并行参数 ---
    DEVICE = 'auto'
    USE_FP16 = True
    EVAL_BATCH = 0
    EVAL_MEM_FRAC = 0.55

    # --- 输出 ---
    PRINT_HISTORY_EVERY = 1

    # --- 断点 / 最优模型 / 种子 ---
    CHECKPOINT_PATH = 'test7h_econ_checkpoint.pth'
    BEST_MODEL_PATH = 'test7h_econ_best_model.pth'
    LATEST_GEN_BEST_MODEL_PATH = 'test7h_econ_latest_gen_best.pth'
    HISTORY_JSON_PATH = 'test7h_econ_history.json'
    AUTO_RESUME = True
    CHECKPOINT_INTERVAL = 10
    SEED_FROM_BEST = False
    SEED_MODEL_PATH = ''
    SEED_MODEL_PATH2 = ''


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
    """三项和式适应度（food 主导 + 微量吃子效率 + 转弯效率一等力量）：
    fitness = food + EFF_W·(food/steps_last)
              + TURN_EFF_W · min(steps_last/turns_last, CAP)/CAP     [ratio 模式]
              + TURN_EFF_W · max(0, 1 − (turns_last/food)/CAP_TPF)   [tpf 备选]
    - 转弯效率窗口截断到"最后一食为止"（SL/TL 同窗口，游荡不改变比值）；
    - W=3/CAP=4 由解法器基准校准（见 Config 注释）：同食数下有序路径
      比模型高 ~1.6 分 > 单食边际 1.25 → 渐进可攀爬；CAP=4 即密度 0.25
      饱和 → 长直段折叠拿满，不被压制；
    - turns_last=0 且 food>0 → 效率取 CAP（全程直行=完美）。
    m 列：0 food, 1 seen, 2 unseen, 3 steps_last, 10 turns_last
    test12 追加列：13 min_reach, 14 island_flag（任一采样点 min_reach<
    ISLAND_THRESHOLD×自由格）→ 适应度 ×ISLAND_PENALTY。
    """
    if m[1] >= 99999:
        return -1e9
    food, steps_last, turns_last = m[0], m[3], m[10]
    if food <= 0:
        return 0.0
    eff = food / max(steps_last, 1.0)
    cap = float(getattr(cfg, 'TURN_EFF_CAP', 4.0))
    w = float(getattr(cfg, 'TURN_EFF_W', 3.0))
    if getattr(cfg, 'TURN_EFF_MODE', 'ratio') == 'tpf':
        te_pts = max(0.0, 1.0 - (turns_last / food) / 10.0)
    else:
        te = cap if turns_last <= 0 else min(steps_last / max(turns_last, 1.0), cap)
        te_pts = te / cap
    fit = food + float(getattr(cfg, 'FOOD_EFF_WEIGHT', 0.3)) * eff + w * te_pts
    iw = float(getattr(cfg, 'IMITATION_W', 0.0))
    if iw > 0 and len(m) > 12:
        fit += iw * (1.0 - float(m[12]))       # 模仿项：mismatch 率惩罚
    # 孤岛惩罚（test12，按局触发率线性折减）：fit ×= (1−(1−pen)·rate)，
    # rate∈[0,1] 为各局 island 触发比例。等价于"局级 ×pen 后跨局平均"的一阶
    # 形式，消除 v1 的'13 局任一触发即整体 ×0.1'一票否决——实测该实现惩罚
    # 与食物数正相关（教师解法器 47.5% 局触发、且触发局 food≈98），选择被
    # 反向压向'只吃 1 食'。
    if len(m) > 14:
        pen = float(getattr(cfg, 'ISLAND_PENALTY', 0.1))
        fit *= 1.0 - (1.0 - pen) * float(m[14])
    return fit


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
    return lambda m: _fitness_econ(m, cfg)


def _auto_eval_batch(cfg, device):
    """单次扫描允许的最大副本数（个体×局复制后的批维大小）。

    注意：不限 POP_SIZE——局维折叠后批维=个体数×并行局数，按显存估算即可。
    """
    if cfg.EVAL_BATCH > 0:
        return max(cfg.EVAL_BATCH, 32)
    if device.type != 'cuda':
        return 1 << 20
    n = cfg.NUM_COLUMNS
    try:
        total = torch.cuda.get_device_properties(device).total_memory
    except Exception:
        return 1 << 20
    per_ind = n * n * 16.0
    per_ind += n * cfg.OBS_DIM * 6.0
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
    """整个种群的基因型/表现型堆叠张量（与 test7g 完全一致）。

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
        return self._obs32()

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

    def _obs32(self):
        """32 维投影观测（食物扇区曼哈顿度量；身体/障碍 1/k 邻近度）。"""
        B, G, dev = self.B, self.G, self.device
        head = self.head
        food = self.food
        d = self.DIRS[self.dir_idx]                       # [B,2]

        obs = torch.zeros(B, 32, dtype=torch.float32, device=dev)

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
    """单次 E-I 迭代（B 个个体并行，无激素支路）。与 test7g 一致；
    FATIGUE_TURN_GAIN=0 时 press 惩罚项为 0（疲劳彻底关闭）。"""
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
def reach_ratio(env):
    """从蛇头可达的自由格占比 [B]（仅观测用，不进适应度）。"""
    B, G, dev = env.B, env.G, env.device
    occ = env._occupancy_flat(tail_invalid=True).view(B, 1, G, G)
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
    metrics = torch.stack((tot_food, tot_seen, tot_unseen, last, prox,
                           tot_wall + (env.died == 1).float(),
                           tot_self + (env.died == 2).float(),
                           tot_starve + (env.died == 3).float(),
                           tot_act1, tot_act2, turn_last, avg_reach,
                           mismatch, min_reach.clamp(max=1.0), island), dim=1)
    return metrics


def apply_weak_mask(pop, cfg):
    """评估期弱连接屏蔽（原地作用于 pop 的 W_rec——pop[idx] 是拷贝，基因栈安全）：
    每个体活跃连接中 |W_rec| 最小的 WEAK_MASK_FRAC 比例置零。
    逐行 kthvalue 阈值实现（避免 CUDA sort/gather 与 inf 哨兵的兼容性问题；
    阈值上的并列幅值可能多置零数个，确定性无碍）。"""
    frac = float(getattr(cfg, 'WEAK_MASK_FRAC', 0.0))
    if frac <= 0 or pop.W_rec is None:
        return
    with torch.no_grad():
        W = pop.W_rec.float()
        M = pop.M_rec
        mag = W.abs() * M
        keep = M > 0
        for r in range(W.shape[0]):
            m_r = mag[r][keep[r]]
            if m_r.numel() == 0:
                continue
            kk = int(math.ceil(m_r.numel() * frac))
            if kk <= 0:
                continue
            thr = torch.kthvalue(m_r, kk).values
            zero = keep[r] & (mag[r] <= thr)
            W[r] = torch.where(zero, torch.zeros_like(W[r]), W[r])
        pop.W_rec = W.to(pop.dtype)


def _eval_pop_banks(pop, cfg, banks):
    """E=len(banks) 局并行评估：个体×E 复制进同一批扫描（局维折叠，扫描次数
    不随局数增长——GPU 利用率低时墙上时间 ∝ 扫描次数而非局数）。
    所有分块共用同一组 banks（CRN 关键）；评估副本先做弱连接屏蔽。
    返回 [P,15] = E 局均值（列12=mismatch 率，13=min_reach，14=island_flag）。"""
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
    ncol = 15
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


def evaluate_population_gpu(pop, cfg, gen=0):
    """两阶段淘汰评估（CRN）：
    阶段1：全种群 × K1 局（同库精确可比）→ 保前 STAGE2_KEEP；
    阶段2：幸存者 × K2 局（新库），幸存者指标 = (K1·m1 + K2·m2)/(K1+K2)。
    返回 (metrics[P,15], order[P])：order = 幸存者按累计适应度降序，
    其后为落选者按阶段1适应度降序——精英只能出自幸存者。
    """
    if getattr(cfg, 'USE_FP16', True):
        pop.fp16()
    key_fn = _make_key_fn(cfg)
    P = pop.P
    use_crn = bool(getattr(cfg, 'USE_CRN', True))
    dev = pop.device
    K1 = int(getattr(cfg, 'STAGE1_EPS', 3))
    K2 = int(cfg.EVAL_EPISODES)
    two_stage = (getattr(cfg, 'STAGE2_KEEP', 0) >= cfg.ELITE_SIZE
                 and K1 > 0 and K2 > K1)

    m1 = _eval_pop_banks(pop, cfg,
                         make_banks(cfg, gen, 1, K1, dev) if use_crn else [None] * K1)
    mn1 = m1.numpy()
    # sorted(reverse=True) 稳定排序，且兼容 tuple 字典序适应度（np.argsort 不行）
    order1 = sorted(range(P), key=lambda i: key_fn(mn1[i]), reverse=True)
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
    surv_rank = sorted(surv_idx, key=lambda i: key_fn(mn[i]), reverse=True)
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
        key_fn = _make_key_fn(cfg)
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
    elites = pop[elite_idx]

    new_pop = pop.empty()
    children = pop.empty(B=P - cfg.ELITE_SIZE)
    for g in GeneStack.GENES:
        setattr(children, g, torch.empty(P - cfg.ELITE_SIZE, *getattr(elites, g).shape[1:],
                                         dtype=getattr(elites, g).dtype, device=dev))

    # --- 后代：父代采样 + 交叉 ---
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
        # ---- G1 交叉：结构组（掩码 + 权重 + 输出偏置）----
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

        # ---- 变异（逐组独立，冻结组跳过；强度逐子代 s 缩放）----
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
            data=json.dumps({'title': title[:20], 'name': 'test12 贪吃蛇进化',
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
    print(f"[筛选] CRN={cfg.USE_CRN} | 两阶段 K1={cfg.STAGE1_EPS} → 保 "
          f"{cfg.STAGE2_KEEP} → K2={cfg.EVAL_EPISODES} | "
          f"变异 s~{cfg.MUT_SCALE_DIST}(σ={cfg.MUT_SCALE_SIGMA}) "
          f"clip[{cfg.MUT_SCALE_MIN},{cfg.MUT_SCALE_MAX}]")
    print(f"[适应度 v{cfg.FITNESS_VERSION}] food + {cfg.FOOD_EFF_WEIGHT}·eff + "
          f"{cfg.TURN_EFF_W}·min(SL/TL,{cfg.TURN_EFF_CAP:.0f})/{cfg.TURN_EFF_CAP:.0f}"
          f"（{cfg.TURN_EFF_MODE} 口径）| 弱连接屏蔽 "
          f"W_rec×{1 - cfg.WEAK_MASK_FRAC:.0%} | te配额精英 {cfg.TE_ELITE}")
    print(f"[test12] obs={cfg.OBS_ENC_VERSION} | 孤岛惩罚：min_reach<"
          f"{cfg.ISLAND_THRESHOLD} → ×{cfg.ISLAND_PENALTY}")
    t_program = time.perf_counter()

    start_gen = 0
    pop = GeneStack(cfg, device=device)
    history = {'gen': [], 'best_food': [], 'avg_food': [], 'best_seen': [],
               'best_unseen': [], 'elite_food': [], 'best_fit': [], 'best_turneff': []}
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
            for k in ('elite_food', 'best_fit', 'best_turneff'):
                history.setdefault(k, [])
            # 跨版本续训（如 7g 断点）时补齐新键长度，避免曲线错位
            for k in ('elite_food', 'best_fit', 'best_turneff'):
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
            seed = load_best_state(cfg.SEED_MODEL_PATH, cfg)
            if seed is not None:
                st, s_food, s_steps = seed
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
            metrics, order = evaluate_population_gpu(pop, cfg, gen=gen)
            eval_time = time.perf_counter() - t_eval
            cum_eval_time += eval_time

            mn = metrics.numpy()
            key_fn = _make_key_fn(cfg)
            best_idx = order[0]
            b_food, b_seen, b_unseen, b_last, b_prox = (
                float(mn[best_idx][0]), float(mn[best_idx][1]),
                float(mn[best_idx][2]), float(mn[best_idx][3]), float(mn[best_idx][4]))
            b_turn = (float(mn[best_idx][8]) + float(mn[best_idx][9])) / max(b_seen + b_unseen, 1.0)
            b_te = (min(b_last / max(float(mn[best_idx][10]), 1.0), cfg.TURN_EFF_CAP)
                    if (mn[best_idx][10] > 0 and b_food > 0) else
                    (cfg.TURN_EFF_CAP if b_food > 0 else 0.0))
            b_fit = key_fn(mn[best_idx])
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

            if key_fn(mn[best_idx]) > key_fn(best_row):
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
                      f"AvgFood: {avg_food:.2f} | EliteFood: {elite_food:.2f} | "
                      f"Die(W/S/St): {avg_wall:.2f}/{avg_self:.2f}/{avg_starve:.2f} | "
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
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(history['gen'], history['best_food'], label='Best Food',
                 color='red', marker='o', markersize=3)
        ax1.plot(history['gen'], history['avg_food'], label='Avg Food',
                 color='blue', alpha=0.6)
        if history.get('elite_food'):
            ax1.plot(history['gen'], history['elite_food'], label='Elite Food',
                     color='green', alpha=0.8)
        ax1.set_title("Evolution Progress — Food Count")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Food Eaten")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.plot(history['gen'], history['best_seen'], label='Best Seen',
                 color='green', marker='s', markersize=3)
        ax2.plot(history['gen'], history['best_unseen'], label='Best Unseen',
                 color='purple', marker='^', markersize=3)
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

    # ---- 训练数据与图表落盘后：AutoDL 微信通知（仅此一次）----
    n_gens = len(history.get('gen', []))
    last_fit = history.get('best_fit', [float('nan')])[-1] if history.get('best_fit') else float('nan')
    send_autodl_notify(
        cfg, 'test12 训练完成',
        f"gens={n_gens} best_food={best_food:.2f} best_fit={last_fit:.2f} "
        f"seen={best_seen:.1f} 用时{(time.perf_counter() - t_program) / 3600:.2f}h。"
        f"产物: {cfg.BEST_MODEL_PATH} / history.json+png / checkpoint")

    print(f"\n--- Best Brain Summary ---")
    st = best_state
    print(f"Input connections active:   {st['M_in'].sum().item()}/{pop.N * pop.O}")
    print(f"Internal connections active: {st['M_rec'].sum().item()}/{pop.N * pop.N}")
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
    obs0 = env0._obs32().float()
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
    obs1 = env0._obs32().float()
    _chk("朝W 正前食物→信号后[10]", float(obs1[0, 10]), K)
    _chk("朝W 正前食物→前无信号[8]", float(obs1[0, 8]), 0.0)

    print("=== 自检 0b：孤岛惩罚（按局触发率线性折减）===")
    mi = np.zeros(15)
    mi[0], mi[3], mi[10] = 10.0, 100.0, 0.0
    f_no = _fitness_econ(mi, cfg)
    pen = float(cfg.ISLAND_PENALTY)
    ok0b = True
    for rate in (1.0, 0.5, 0.0):
        mi[14] = rate
        f = _fitness_econ(mi, cfg)
        want = f_no * (1.0 - (1.0 - pen) * rate)
        good = abs(f - want) < 1e-9
        ok0b = ok0b and good
        print(f"  触发率{rate:.1f}: {f:.4f} (期望 {want:.4f}) {'OK' if good else 'FAIL'}")
    print(f"  孤岛惩罚分级 {'OK' if ok0b else 'FAIL'}")

    print("=== 自检 1：适应度公式（v3：ratio W=3 CAP=4）===")
    m = np.zeros(12)
    m[0], m[3], m[10] = 10.0, 100.0, 10.0   # food=10, SL=100, TL=10 → te=10→cap
    f1 = _fitness_econ(m, cfg)
    expect1 = 10 + 0.3 * 0.1 + 3.0 * 1.0
    m[10] = 0.0                              # 全程直行 → te=cap
    f2 = _fitness_econ(m, cfg)
    m2 = np.zeros(12)                        # 高密度：SL=100, TL=50 → te=2 → 1.5 分
    m2[0], m2[3], m2[10] = 10.0, 100.0, 50.0
    f3 = _fitness_econ(m2, cfg)
    expect3 = 10 + 0.3 * 0.1 + 3.0 * 0.5
    m[0] = 0.0                               # 零食 → 0
    f4 = _fitness_econ(m, cfg)
    print(f"  有序(te≥cap): {f1:.4f} (期望 {expect1:.4f}) "
          f"{'OK' if abs(f1 - expect1) < 1e-9 else 'FAIL'}")
    print(f"  全直行: {f2:.4f} (应= {expect1:.4f}) "
          f"{'OK' if abs(f2 - expect1) < 1e-9 else 'FAIL'}")
    print(f"  高密度(te=2): {f3:.4f} (期望 {expect3:.4f}) "
          f"{'OK' if abs(f3 - expect3) < 1e-9 else 'FAIL'}")
    print(f"  food=0: {f4:.4f} (期望 0) {'OK' if f4 == 0.0 else 'FAIL'}")

    print("=== 自检 2：变异强度分布 ===")
    dev = _resolve_device(cfg)
    s = sample_mut_scale(cfg, 200000, dev).cpu()
    print(f"  中位数 {s.median():.3f} | 分位 5% {s.quantile(0.05):.3f} / "
          f"95% {s.quantile(0.95):.3f} | P(s>2)={float((s > 2).float().mean()):.4f} "
          f"| P(s<0.5)={float((s < 0.5).float().mean()):.4f} "
          f"| max {s.max():.3f}")

    print("=== 自检 3：弱连接屏蔽生效校验 ===")
    torch.manual_seed(11)
    p3 = GeneStack(cfg, B=16, device=dev)
    p3.random_init()
    frac = float(cfg.WEAK_MASK_FRAC)
    w_before = p3.W_rec.clone()
    p3c = p3[torch.arange(16, device=dev)]           # 拷贝（模拟评估路径）
    before = (p3c.W_rec * p3c.M_rec != 0).sum().item()
    apply_weak_mask(p3c, cfg)
    after = (p3c.W_rec * p3c.M_rec != 0).sum().item()
    genes_intact = torch.equal(p3.W_rec, w_before)
    print(f"  活跃 W_rec 连接 {before} → {after}（屏蔽 {1 - after / before:.1%}，"
          f"目标 {frac:.0%}）| 原基因栈未动: {genes_intact} "
          f"{'OK' if genes_intact and abs((1 - after / before) - frac) < 0.02 else 'FAIL'}")

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
    print("=== 自检完成 ===")


# ==========================================
# 9. 入口
# ==========================================
def make_smoke_config():
    cfg = Config()
    cfg.POP_SIZE = 16
    cfg.NUM_COLUMNS = 24
    cfg.OBS_DIM = 32
    cfg.ACTION_DIM = 3
    cfg.GENERATIONS = 3
    cfg.ELITE_SIZE = 6
    cfg.STAGE1_EPS = 1
    cfg.EVAL_EPISODES = 2
    cfg.STAGE2_KEEP = 8
    cfg.MAX_STEPS = 60
    cfg.FRAME_RATE = 2
    cfg.CHECKPOINT_INTERVAL = 2
    cfg.CHECKPOINT_PATH = 'test12_smoke_checkpoint.pth'
    cfg.BEST_MODEL_PATH = 'test12_smoke_best.pth'
    cfg.LATEST_GEN_BEST_MODEL_PATH = 'test12_smoke_latest_gen_best.pth'
    cfg.HISTORY_JSON_PATH = 'test12_smoke_history.json'
    cfg.SEED_FROM_BEST = False
    cfg.EVAL_BATCH = 16
    cfg.PRINT_HISTORY_EVERY = 1
    return cfg


def main():
    ap = argparse.ArgumentParser(description='test7h — CRN 精确筛选 + 两阶段淘汰 + 类正态变异')
    ap.add_argument('--smoke', action='store_true', help='小规模快速自检')
    ap.add_argument('--selfcheck', action='store_true',
                    help='只跑自检（适应度/变异分布/CRN 确定性），不训练')
    ap.add_argument('--gens', type=int, default=None)
    ap.add_argument('--pop', type=int, default=None)
    ap.add_argument('--columns', type=int, default=None)
    ap.add_argument('--episodes', type=int, default=None, help='阶段2 局数 K2')
    ap.add_argument('--max-steps', type=int, default=None)
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--fit-mode', type=str, default=None, choices=['econ', 'tuple'])
    ap.add_argument('--eff-weight', type=float, default=None,
                    help='吃子效率权重（默认 0.3）')
    ap.add_argument('--turn-eff-w', type=float, default=None,
                    help='转弯效率权重（默认 3.0，解法器基准校准）')
    ap.add_argument('--turn-eff-cap', type=float, default=None,
                    help='转弯效率饱和上限（默认 4，密度 0.25 饱和）')
    ap.add_argument('--turn-eff-mode', type=str, default=None,
                    choices=['ratio', 'tpf'],
                    help='转弯效率口径：ratio=SL/TL（默认）| tpf=每食物转弯数（备选）')
    ap.add_argument('--weak-mask-frac', type=float, default=None,
                    help='评估期 W_rec 弱连接屏蔽比例（默认 0.20，0=关）')
    ap.add_argument('--te-elite', type=int, default=None,
                    help='te-配额精英数（默认 0=关；行为学 B5 对策）')
    ap.add_argument('--imitation-w', type=float, default=None,
                    help='模仿引导权重（默认 0=关；建议 2.0）')
    ap.add_argument('--island-threshold', type=float, default=None,
                    help='孤岛判定阈值：min_reach<阈值×自由格 触发（默认 0.3）')
    ap.add_argument('--island-penalty', type=float, default=None,
                    help='孤岛适应度罚因子（默认 0.1；1.0=关闭）')
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
    if args.episodes:
        cfg.EVAL_EPISODES = args.episodes
    if args.max_steps:
        cfg.MAX_STEPS = args.max_steps
    if args.device:
        cfg.DEVICE = args.device
    if args.fit_mode:
        cfg.FIT_MODE = args.fit_mode
        if not args.smoke:
            arm = 'econ' if args.fit_mode == 'econ' else 'tup'
            cfg.CHECKPOINT_PATH = f'test12_{arm}_checkpoint.pth'
            cfg.BEST_MODEL_PATH = f'test12_{arm}_best_model.pth'
            cfg.LATEST_GEN_BEST_MODEL_PATH = f'test12_{arm}_latest_gen_best.pth'
            cfg.HISTORY_JSON_PATH = f'test12_{arm}_history.json'
    if args.eff_weight is not None:
        cfg.FOOD_EFF_WEIGHT = args.eff_weight
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
    if args.island_penalty is not None:
        cfg.ISLAND_PENALTY = args.island_penalty
    if args.name:
        cfg.CHECKPOINT_PATH = f'test12_{args.name}_checkpoint.pth'
        cfg.BEST_MODEL_PATH = f'test12_{args.name}_best_model.pth'
        cfg.LATEST_GEN_BEST_MODEL_PATH = f'test12_{args.name}_latest_gen_best.pth'
        cfg.HISTORY_JSON_PATH = f'test12_{args.name}_history.json'
    if args.resume_pop:
        payload = torch.load(args.resume_pop, map_location='cpu', weights_only=False)
        saved = payload.get('config', {})
        if saved and (saved.get('NUM_COLUMNS') != cfg.NUM_COLUMNS or
                      saved.get('OBS_DIM') != cfg.OBS_DIM or
                      saved.get('ACTION_DIM') != cfg.ACTION_DIM):
            sys.exit(f'[错误] resume-pop {args.resume_pop} 与当前维度不匹配')
        if saved and saved.get('OBS_ENC_VERSION', '32proj') != getattr(cfg, 'OBS_ENC_VERSION', '32proj'):
            sys.exit(f'[错误] resume-pop {args.resume_pop} 观测编码不符 '
                     f'({saved.get("OBS_ENC_VERSION")} != {cfg.OBS_ENC_VERSION})，'
                     f'7h 断点不能导入 test12（编码不同），请从零训练或用 --seed-model')
        parent = os.path.dirname(os.path.abspath(cfg.CHECKPOINT_PATH))
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(payload, cfg.CHECKPOINT_PATH)
        print(f"[resume-pop] 已导入 {args.resume_pop} "
              f"(next_gen={payload.get('next_gen')}) -> {cfg.CHECKPOINT_PATH}")
    if args.stage1_eps is not None:
        cfg.STAGE1_EPS = args.stage1_eps
    if args.stage2_eps is not None:
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
        send_autodl_notify(cfg, 'test12 训练异常退出',
                           f'{type(e).__name__}: {str(e)[:150]}')
        raise


if __name__ == '__main__':
    main()
