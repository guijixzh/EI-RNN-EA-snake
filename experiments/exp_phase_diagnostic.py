# ==========================================
# exp_phase_diagnostic.py —— Phase-0 相位归因实验（预注册）
#
# 问题：test7b/test12 后期训练上不去，是否因为"后期游戏段（长蛇）模式缺失"
#       （前后期质变假设），即网络即使置身后期状态也无法表现后期行为？
#
# 方法：让现有冠军脑从指定蛇长直接"出生"续评（两种出生构型），
#       与同一批脑自然局中"到达同一长度之后"的增食分布对照。
#   - coil 出生：蛇身蛇形紧凑摆放（最有利的有序后期构型）；
#   - teacher 出生：教师解法器从 len2 打到 len L 的真实身体状态快照
#     （教师从同一状态 in-clock 可打 93 分，即"93 分可达"的参照构型）。
#   两被试各用各的原生口径：test12(32ego1, SLOPE=5, 无疲劳) /
#   test7b(32proj, SLOPE=3, FATIGUE=1e-5)，前向/环境语义直接 import 原模块。
#   出生时脑状态清零（冷脑启动，预注册口径；自然局对照不含此因素，
#   由 teacher/coil 双构型与 L=2 对照档交叉检验 harness 偏差）。
#
# ------- 预注册判定门 P0（实验前写死，run 前不可改）-------
# 对每个被试 S、每个档 L∈{30,40,50}（L=20 辅助，L=2 仅 harness 对照）：
#   ratio(S,L,mode) = median(出生档增食) / median(自然局到达 L 后增食)
#   squeeze(S,L,mode) = 出生局死因中撞己(含空间挤压)占比
# P0-CONFIRM（确认后期质变主因，转入激素重启）：
#   存在 S,L 使 ratio < 0.5 在 coil 与 teacher 两种出生下同时成立，
#   且该 (S,L) 处两种出生的 squeeze 均为众数死因。
# P0-STRONG：上述 ratio < 0.25。
# P0-STOP（证伪"后期模式缺失"，归因改写为过渡期/共适应问题）：
#   所有 (S,L≥20,mode) 的 ratio ≥ 0.5。
# 其余情形：PARTIAL，需人工复核分长度段曲线后再定。
#
# ------- 修订 v1.1（冒烟轮发现，全量轮之前写入）-------
# coil 出生构型增设 OOD 对照：若 L=2 出生档即崩塌（med < 0.5×自然局全局中位），
# 判定 coil 构型对该被试混杂"出生构型失适应"（如 7b 脑对角落出发完全失适应），
# 该被试的门判定仅采用 teacher 构型。L=2 对照档本来就是 harness 偏差检查，
# 此修订只是把检查结论接入门逻辑，不改变 ratio 阈值本身。
# 教师追溯对"未到 L 即饿死的行"就地复活重开（slope=3 口径下 len2 钟仅 26 步，
# 教师偶有失手；复活上限每人 8 次，失败行按饿死计并输出 trace_failed 计数）。
# ==========================================

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test12 as t12   # noqa: E402
import test7b as t7b   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIERS = [2, 20, 30, 40, 50]
GATE_TIERS = [30, 40, 50]
# 自然局/出生局共用的行为分段（按步进前蛇长）
BANDS = [(2, 12), (12, 22), (22, 32), (32, 42), (42, 52),
         (52, 62), (62, 72), (72, 82), (82, 92), (92, 101)]
DEATH_NAMES = {0: 'alive/censored', 1: 'wall', 2: 'self', 3: 'starve'}
STEP_CAP = 6000


# ---------- 配置与被试装载 ----------

def build_cfg_from_ckpt(module, ckpt_path):
    """用 checkpoint 内保存的 config 覆盖模块默认 Config（保证原生语义）。"""
    data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    saved = data.get('config', {})
    cfg = module.Config()
    for k, v in saved.items():
        if k.startswith('__'):
            continue
        if isinstance(v, (int, float, str, bool, list, tuple)) or v is None:
            setattr(cfg, k, v)
    cfg.DEVICE = 'auto'
    return cfg, float(data.get('food', -1.0))


def broadcast_brain(module, cfg, ckpt_path, B, device):
    """单脑 checkpoint → B 行 GeneStack（全体同一基因）。"""
    loaded = module.load_best_state(ckpt_path, cfg)
    if loaded is None:
        raise RuntimeError(f'checkpoint 不可读: {ckpt_path}')
    st, food, steps = loaded
    pop = module.GeneStack(cfg, B=B, device=device)
    pop.random_init()
    for i in range(B):
        pop.set_individual_from_state(i, st)
    if device.type == 'cpu':
        cfg.USE_FP16 = False            # CPU 路径禁半精度（fp16 bmm 不支持）
    pop.fp16()
    pop.refresh_eff()
    return pop, food


# ---------- CRN 库（t12 用）；7b 无 CRN，用全局种子 ----------

def make_stacked_bank(cfg, n_banks, tag, device):
    banks = [t12.make_bank(cfg, 700000 + tag * 17 + 3, 0, e, device)
             for e in range(n_banks)]
    return {'stream': torch.stack([b['stream'] for b in banks]),
            'dir0': torch.tensor([b['dir0'] for b in banks],
                                 dtype=torch.long, device=device)}


# ---------- 出生构型 ----------

def serpentine_path(G, L):
    """蛇形路径 p[0..L-1]（尾→头）：第 0 行左→右，第 1 行右→左……"""
    path = []
    for r in range(G):
        cols = range(G) if r % 2 == 0 else range(G - 1, -1, -1)
        for c in cols:
            path.append((r, c))
            if len(path) >= L:
                return torch.tensor(path, dtype=torch.long)
    return torch.tensor(path, dtype=torch.long)


def spawn_coil(env, L, module):
    """蛇形紧凑构型：body[0]=头=p[L-1]，dir=头两节走向；重摆食物。"""
    B, G, dev = env.B, env.G, env.device
    p = serpentine_path(G, L).to(dev)             # [L,2] 尾→头
    body = torch.zeros(B, env.MAXLEN, 2, dtype=torch.long, device=dev)
    body[:, :L] = p.flip(0).unsqueeze(0)          # 头在前
    env.body = body
    env.body_len = torch.full((B,), L, dtype=torch.long, device=dev)
    env.head = p[L - 1].unsqueeze(0).expand(B, 2).contiguous()
    dvec = (p[L - 1] - p[L - 2]).tolist()
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    env.dir_idx = torch.full((B,), dirs.index(tuple(dvec)),
                             dtype=torch.long, device=dev)
    env.steps.zero_()
    env.steps_wo_food.zero_()
    env.alive.fill_(True)
    env.died.zero_()
    env.ate.fill_(False)
    _replace_food_avoiding_body(env, module)


def _replace_food_avoiding_body(env, module):
    B, G, dev = env.B, env.G, env.device
    occ = env._occupancy_flat() > 0.5

    def draw():
        if hasattr(env, '_next_cand'):          # t12: CRN 流取候选
            return env._next_cand()
        return torch.randint(0, G, (B, 2), device=dev)   # 7b: 全局 RNG

    cand = draw()
    bad = occ.gather(1, (cand[:, 0] * G + cand[:, 1]).unsqueeze(1)).squeeze(1)
    for _ in range(64):
        if not bad.any():
            break
        re = draw()
        cand = torch.where(bad.unsqueeze(1), re, cand)
        bad = occ.gather(1, (cand[:, 0] * G + cand[:, 1]).unsqueeze(1)).squeeze(1)
    if bad.any():
        free = ~occ
        idx = torch.argmax(free.float(), dim=1)
        fb = torch.stack((idx // G, idx % G), dim=1)
        cand = torch.where(bad.unsqueeze(1), fb, cand)
    env.food = cand


def _reinit_rows(env, mask):
    """就地重开指定行（同 reset 语义：中心出生 len2 新方向新食物）。"""
    B, G, dev = env.B, env.G, env.device
    idx = torch.nonzero(mask).squeeze(1)
    env.head[idx] = torch.tensor([G // 2, G // 2], device=dev)
    env.dir_idx[idx] = torch.randint(0, 4, (len(idx),), device=dev)
    env.body[idx] = 0
    env.body[idx, 0] = env.head[idx]
    env.body[idx, 1] = env.head[idx] - env.DIRS[env.dir_idx[idx]]
    env.body_len[idx] = 2
    env.alive[idx] = True
    env.died[idx] = 0
    env.steps[idx] = 0
    env.steps_wo_food[idx] = 0
    env.ate[idx] = False
    # 重摆食物：全批取候选（消耗流指针无碍，快照行以快照指针为准），仅更新指定行
    if hasattr(env, '_next_cand'):
        cand = env._next_cand()
    else:
        cand = torch.randint(0, G, (B, 2), device=dev)
    occ = env._occupancy_flat() > 0.5
    bad = occ.gather(1, (cand[:, 0] * G + cand[:, 1]).unsqueeze(1)).squeeze(1)
    for _ in range(8):
        if not bool(bad[mask].any()):
            break
        if hasattr(env, '_next_cand'):
            re = env._next_cand()
        else:
            re = torch.randint(0, G, (B, 2), device=dev)
        cand = torch.where(bad.unsqueeze(1), re, cand)
        bad = occ.gather(1, (cand[:, 0] * G + cand[:, 1]).unsqueeze(1)).squeeze(1)
    env.food[idx] = cand[idx]


def teacher_trace_spawn(env, cfg, L, device, max_trace=3000, max_revive=8):
    """教师从 len2 打到 len L，逐行快照真实身体状态。

    slope 紧的口径（如 7b slope=3）下个别行会在 len2 饿死、永远到不了 L——
    就地复活重开（上限 max_revive 次）；复活耗尽仍失败的行为保持死亡，
    在结果中计为 trace_failed。"""
    teacher = t12.VectorCycleTeacher(env.G, device)
    B = env.B
    have = torch.zeros(B, dtype=torch.bool, device=device)
    snap = {
        'body': torch.zeros(B, env.MAXLEN, 2, dtype=torch.long, device=device),
        'head': torch.zeros(B, 2, dtype=torch.long, device=device),
        'len': torch.zeros(B, dtype=torch.long, device=device),
        'dir': torch.zeros(B, dtype=torch.long, device=device),
        'food': torch.zeros(B, 2, dtype=torch.long, device=device),
        'draw': (torch.zeros(B, dtype=torch.long, device=device)
                 if hasattr(env, 'draw_cnt') else None),
    }
    n_revive = 0
    for t in range(max_trace):
        if bool(have.all()):
            break
        if not bool(env.alive.any()):
            # 无存活行且仍有未达标行：全部立即复活一次（不耗 revive 配额判断放下面）
            dead = ~have
            if n_revive + int(dead.sum()) > max_revive * B:
                break
            _reinit_rows(env, dead)
            n_revive += int(dead.sum())
        ar = torch.arange(B, device=device)
        necks = env.body[ar, 1]
        t_act = teacher.act(env.head, env.food, env.body, env.body_len,
                            env.dir_idx, necks)
        env.step(t_act)
        hit = (~have) & env.alive & (env.body_len >= L)
        if hit.any():
            snap['body'][hit] = env.body[hit]
            snap['head'][hit] = env.head[hit]
            snap['len'][hit] = env.body_len[hit]
            snap['dir'][hit] = env.dir_idx[hit]
            snap['food'][hit] = env.food[hit]
            if snap['draw'] is not None:
                snap['draw'][hit] = env.draw_cnt[hit]
            have |= hit
        # 未达标即死的行：复活重开
        dead = (~have) & (~env.alive)
        if bool(dead.any()):
            if n_revive + int(dead.sum()) <= max_revive * B:
                _reinit_rows(env, dead)
                n_revive += int(dead.sum())
    n_failed = int((~have).sum())
    # 快照写回：冷脑续局（steps/饿死钟清零，食物沿用教师刚重摆的）
    m = have
    env.body[m] = snap['body'][m]
    env.head[m] = snap['head'][m]
    env.body_len[m] = snap['len'][m]
    env.dir_idx[m] = snap['dir'][m]
    env.food[m] = snap['food'][m]
    if snap['draw'] is not None:
        env.draw_cnt[m] = snap['draw'][m]
    env.steps.zero_()
    env.steps_wo_food.zero_()
    env.alive = have.clone()
    env.died[~have] = 3          # 追溯失败行按饿死计
    env.died[have] = 0
    env.ate.fill_(False)
    return n_failed


# ---------- 单次批量评估 ----------

def run_sweep(module, pop, cfg, device, mode, tier, n_games, seed_tag,
              step_cap=STEP_CAP):
    """mode ∈ {'natural','coil','teacher'}；返回逐局与聚合指标。"""
    B = n_games
    is_t12 = (module is not t7b)     # t12/t15 同族走 CRN 库路径；t7b 无 CRN
    env = module.BatchedSnakeEnv(cfg, B, device)
    if is_t12:
        bank = make_stacked_bank(cfg, B, seed_tag, device)
        env.reset(bank=bank)
    else:
        torch.manual_seed(1234567 + seed_tag * 101)
        env.reset()

    if mode == 'coil':
        spawn_coil(env, tier, module)
        n_failed = 0
    elif mode == 'teacher':
        n_failed = teacher_trace_spawn(env, cfg, tier, device)
    else:
        n_failed = 0

    N = pop.N
    half = torch.float16 if getattr(cfg, 'USE_FP16', True) else torch.float32
    E = torch.zeros(B, N, dtype=half, device=device)
    I = torch.zeros(B, N, dtype=half, device=device)
    st = torch.zeros(B, N, dtype=half, device=device)
    if module is t7b:                       # 7b: 连续动作计数 [B,A]
        fatigue = torch.zeros(B, cfg.ACTION_DIM, dtype=torch.float32,
                              device=device)
    else:                                   # t12: 转弯疲劳 press [B]
        fatigue = torch.zeros(B, dtype=torch.float32, device=device)

    hist = torch.zeros(B, env.MAXLEN, dtype=torch.float32, device=device)
    band_turns = torch.zeros(B, len(BANDS), dtype=torch.float32, device=device)
    band_steps = torch.zeros(B, len(BANDS), dtype=torch.float32, device=device)
    censored = False

    for t in range(min(int(getattr(cfg, 'MAX_STEPS', 100000)), step_cap)):
        al = env.alive
        if not bool(al.any()):
            break
        obs = env.obs().to(half)
        len_pre = env.body_len
        if module is t7b:
            act, E, I, st = module.deliberate_batch(pop, obs, E, I, st,
                                                    fatigue, cfg)
            fatigue = module.update_fatigue(fatigue, act)
        else:
            act, E, I, st = module.deliberate_batch(pop, obs, E, I, st,
                                                    fatigue, cfg)
            fatigue = t12.update_fatigue(fatigue, act,
                                         decay=float(cfg.FATIGUE_TURN_DECAY))
        env.step(act)
        ate = al & env.ate
        hist.scatter_add_(1, len_pre.clamp(max=env.MAXLEN - 1).unsqueeze(1),
                          ate.float().unsqueeze(1))
        turn = (act != 0)
        for bi, (lo, hi) in enumerate(BANDS):
            m = al & (len_pre >= lo) & (len_pre < hi)
            band_steps[:, bi] += m.float()
            band_turns[:, bi] += (m & turn).float()
    censored = bool(env.alive.any())

    conn_end = t12.habit_conn_score(env)
    reach_end = t12.reach_ratio(env)
    food_total = hist.sum(dim=1)
    died = env.died.clamp(min=0, max=3)
    rows = {
        'food': food_total.cpu(), 'died': died.cpu(),
        'steps': env.steps.cpu(), 'death_len': env.body_len.cpu(),
        'conn_end': conn_end.cpu(), 'reach_end': reach_end.cpu(),
        'hist': hist.cpu(), 'band_turns': band_turns.cpu(),
        'band_steps': band_steps.cpu(), 'censored': censored,
        'trace_failed': n_failed,
    }
    return rows


def postL_food(hist, L):
    """到达长度 L 之后的增食（吃时蛇长 ≥ L 的进食次数）。hist: [B, MAXLEN]"""
    return hist[:, L:].sum(dim=1)


def reached_L(hist, L):
    """是否到达过长度 L（吃时蛇长 ≥ L-1 至少一次）。"""
    return hist[:, max(L - 1, 0):].sum(dim=1) > 0


# ---------- 聚合与门判定 ----------

def _q(x, ps):
    x = np.asarray(x, dtype=np.float64)
    return {f'p{int(p*100)}': float(np.quantile(x, p)) for p in ps}


def summarize_spawned(rows):
    died_np = rows['died'].numpy()
    n = len(died_np)
    causes = {DEATH_NAMES[c]: float((died_np == c).mean()) for c in (1, 2, 3)}
    causes['censored'] = 1.0 if rows['censored'] else 0.0
    return {
        'n': n,
        'food_median': float(np.median(rows['food'].numpy())),
        'food_mean': float(rows['food'].mean()),
        **_q(rows['food'].numpy(), (0.25, 0.75)),
        'steps_median': float(np.median(rows['steps'].numpy())),
        'death_causes': causes,
        'conn_at_death_median': float(np.median(rows['conn_end'].numpy())),
        'reach_at_death_median': float(np.median(rows['reach_end'].numpy())),
        'trace_failed': int(rows.get('trace_failed', 0)),
    }


def summarize_natural(rows, tiers):
    hist = rows['hist']
    out = {'n': len(rows['food']), 'food_median': float(np.median(rows['food'].numpy())),
           'food_mean': float(rows['food'].mean()),
           'death_causes': {DEATH_NAMES[c]: float((rows['died'].numpy() == c).mean())
                            for c in (1, 2, 3)},
           'tiers': {}}
    for L in tiers:
        m = reached_L(hist, L)
        n_reach = int(m.sum())
        if n_reach == 0:
            out['tiers'][str(L)] = {'n_reach': 0}
            continue
        pf = postL_food(hist, L)[m].numpy()
        band_ids = [bi for bi, (lo, hi) in enumerate(BANDS) if lo >= L]
        turns = rows['band_turns'][:, band_ids].sum(dim=1)[m].numpy()
        steps = rows['band_steps'][:, band_ids].sum(dim=1)[m].numpy()
        td = turns / np.clip(steps, 1, None)
        # 自然局在 ≥L 段内的死因（死时长度≥L）
        dl = rows['death_len'].numpy()[m]
        dc = rows['died'].numpy()[m]
        late_death = (dl >= L) & (dc > 0)
        out['tiers'][str(L)] = {
            'n_reach': n_reach,
            'reach_rate': float(m.float().mean()),
            'postL_food_median': float(np.median(pf)),
            'postL_food_mean': float(pf.mean()),
            **_q(pf, (0.25, 0.75)),
            'turn_density_postL': float(td.mean()),
            'death_causes_postL': {DEATH_NAMES[c]:
                                   float(((dc == c) & late_death).sum() / max(int(late_death.sum()), 1))
                                   for c in (1, 2, 3)},
        }
    return out


def gate_p0(results, tiers):
    """预注册门 P0（v1.1 修订：coil 构型 OOD 对照降权）。

    修订依据（冒烟轮发现，全量轮之前写入）：7b 脑对 coil 出生在 L=2 对照档
    即崩塌（角落出发对中心出生训练的脑是分布外构型），此时 coil 各档数据
    混杂"出生构型失适应"而非"后期模式缺失"，门判定对该被试仅采用 teacher
    构型；t12 的 L=2 对照正常则双构型照常。"""
    verdict_lines = []
    confirm = False
    strong = False
    stop = True
    for sname, sub in results.items():
        ctrl = sub['spawned']['coil'].get('2')
        nat_med = float(sub['natural'].get('food_median', 0.0))
        coil_ok = (ctrl is not None and nat_med > 0
                   and ctrl['food_median'] >= 0.5 * nat_med)
        if not coil_ok:
            cm = ctrl['food_median'] if ctrl else float('nan')
            verdict_lines.append(
                f'{sname}: coil L=2 对照失效 (med={cm:.1f} vs natural '
                f'{nat_med:.1f}) → coil 构型判 OOD 混杂，门仅用 teacher')
        for mode in ('coil', 'teacher'):
            if mode == 'coil' and not coil_ok:
                continue
            for L in GATE_TIERS:
                sp = sub['spawned'][mode].get(str(L))
                nat = sub['natural']['tiers'].get(str(L))
                if sp is None or nat is None or nat.get('n_reach', 0) < 15:
                    continue
                ratio = sp['food_median'] / max(nat['postL_food_median'], 1e-6)
                squeeze = sp['death_causes'].get('self', 0.0)
                stop = False
                verdict_lines.append(
                    f'{sname} {mode} L={L}: ratio={ratio:.2f} '
                    f'(spawned {sp["food_median"]:.1f} vs natural {nat["postL_food_median"]:.1f}), '
                    f'self-death={squeeze:.0%}')
                if ratio < 0.5:
                    cond = squeeze >= max(sp['death_causes'].get('wall', 0),
                                          sp['death_causes'].get('starve', 0),
                                          sp['death_causes'].get('censored', 0))
                    if cond:
                        confirm = True
                    if ratio < 0.25:
                        strong = True
    if stop:
        return 'P0-STOP', verdict_lines
    if strong and confirm:
        return 'P0-CONFIRM(STRONG)', verdict_lines
    if confirm:
        return 'P0-CONFIRM', verdict_lines
    return 'PARTIAL', verdict_lines


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='冒烟口径：小样本+少档')
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    tiers = [2, 30, 50] if args.quick else TIERS
    n_spawn = 32 if args.quick else 128
    n_nat = 64 if args.quick else 320
    step_cap = 2500 if args.quick else STEP_CAP

    device = torch.device('cuda' if args.device != 'cpu' and
                          torch.cuda.is_available() else 'cpu')
    print(f'device={device}  tiers={tiers}  n_spawn={n_spawn}  n_nat={n_nat}')

    subjects = {
        'test12': (t12, os.path.join(ROOT, 'test12_econ_latest_gen_best.pth')),
        'test7b': (t7b, os.path.join(ROOT, 'test7b_latest_gen_best.pth')),
    }

    results = {}
    for s_idx, (sname, (module, ckpt)) in enumerate(subjects.items()):
        if not os.path.exists(ckpt):
            print(f'跳过 {sname}: 缺 {ckpt}')
            continue
        t0 = time.time()
        cfg, ckpt_food = build_cfg_from_ckpt(module, ckpt)
        sub = {'checkpoint': os.path.basename(ckpt), 'checkpoint_food': ckpt_food,
               'obs_enc': getattr(cfg, 'OBS_ENC_VERSION',
                                  getattr(cfg, 'OBS_MODE', '?')),
               'starve_slope': getattr(cfg, 'STARVE_SLOPE', None),
               'spawned': {}, 'natural': None}
        print(f'\n=== {sname} (ckpt_food={ckpt_food:.1f}, '
              f'enc={sub["obs_enc"]}, slope={sub["starve_slope"]}) ===')
        for m_idx, mode in enumerate(('coil', 'teacher')):
            sub['spawned'][mode] = {}
            for t_idx, L in enumerate(tiers):
                tag = s_idx * 100000 + m_idx * 1000 + t_idx * 10
                pop, _ = broadcast_brain(module, cfg, ckpt, n_spawn, device)
                rows = run_sweep(module, pop, cfg, device, mode, L, n_spawn,
                                 seed_tag=tag, step_cap=step_cap)
                sub['spawned'][mode][str(L)] = summarize_spawned(rows)
                s = sub['spawned'][mode][str(L)]
                print(f'  [{mode} L={L:>2}] food med={s["food_median"]:5.1f} '
                      f'mean={s["food_mean"]:5.1f} steps={s["steps_median"]:5.0f} '
                      f'death={ {k: round(v, 2) for k, v in s["death_causes"].items()} } '
                      f'[{time.time()-t0:.0f}s]')
        tag = s_idx * 100000 + 900
        pop, _ = broadcast_brain(module, cfg, ckpt, n_nat, device)
        rows = run_sweep(module, pop, cfg, device, 'natural', 2, n_nat,
                         seed_tag=tag, step_cap=step_cap)
        sub['natural'] = summarize_natural(rows, tiers)
        print(f'  [natural] food med={sub["natural"]["food_median"]:.1f}')
        for L in tiers:
            tr = sub['natural']['tiers'].get(str(L), {})
            if tr.get('n_reach', 0):
                print(f'    natural L>={L}: reach={tr["reach_rate"]:.0%} '
                      f'postL med={tr["postL_food_median"]:.1f} '
                      f'turn_dens={tr["turn_density_postL"]:.2f}')
        results[sname] = sub

    verdict, lines = gate_p0(results, tiers)
    print('\n===== 预注册门 P0 =====')
    for ln in lines:
        print(' ' + ln)
    print(f'>>> 判定: {verdict}')

    out = {'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
           'config': {'tiers': tiers, 'n_spawn': n_spawn, 'n_nat': n_nat,
                      'step_cap': step_cap, 'device': str(device)},
           'gate': {'verdict': verdict, 'detail': lines},
           'results': results}
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    out_path = os.path.join(ROOT, 'results', 'phase_diagnostic.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f'已写出 {out_path}')


if __name__ == '__main__':
    main()
