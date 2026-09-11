# ==========================================
# experiments/solver_reference.py —— 曼哈顿回路解法器基准：适应度指向性验证
#
# 问题：候选适应度是否真的把"理想策略"排在模型之上？拍脑袋定权重不可靠，
# 本实验用确定性解法器给出同食数下的有序路径解法作为基准：
#   - S1 纯哈密顿回路跟随（10×10，列 1-9 逐行蛇形 + 列 0 回廊）：
#     零风险完赛，有序形态下界参考（贴墙长直段、L 形、低密度折叠）；
#   - S2 回路+安全捷径（BFS 最短路朝食物；安全检查=移动后洪泛可达空间≥蛇长+2；
#     蛇长≥55 或无路时退回跟回路）：有序形态实用参考。
# 协议（配对）：7h best（曼哈顿口径）与 7b best（原生欧氏口径）在同一组
# 40 个 CRN 库上各打 40 局；解法器同库打完整局，事后在每个体对应的
# 模型食物数 N 处截断 → 同食数下比较各候选适应度 F(解法器@N) vs F(模型@N)。
# 候选：现行 W=0.5 比值式 | 比值式 W=1.5/3.0 | TPF 式 W=3（负对照，
# 检验"后期复杂机动反向表征"）| 无效率项。
# 判定（预注册）：≥90% 配对局解法器胜出且平均差距 ≥1.5 分 ⇔ 公式成立；
# 同时报告"最后 5 食"区段密度与各公式在该区段的扣分幅度。
# 产出：results/test7h_solver_reference.json / .png
# ==========================================
import json
import math
import os
import sys
from collections import deque

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments', 'test7_series'))  # 归档后模块路径
import test7h as t7h  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = 10
EPS = 40
DEV = torch.device('cpu')


# ---------- 哈密顿回路（列 1-9 逐行蛇形 + 列 0 回廊）----------
def build_cycle(g=G):
    order = []
    for r in range(g):
        cols = range(1, g) if r % 2 == 0 else range(g - 1, 0, -1)
        order += [(r, c) for c in cols]
    order += [(r, 0) for r in range(g - 1, -1, -1)]
    assert len(order) == g * g, len(order)
    pos = {cell: i for i, cell in enumerate(order)}
    succ = [order[(i + 1) % (g * g)] for i in range(g * g)]
    return order, pos, succ


CYCLE_ORDER, CYCLE_POS, CYCLE_SUCC = build_cycle()
DIRVEC = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}


def action_toward(cur_dir, cur_cell, nxt_cell):
    """从 cur_cell 到相邻格 nxt_cell 的动作（0直行 1左转 2右转）。"""
    dr, dc = nxt_cell[0] - cur_cell[0], nxt_cell[1] - cur_cell[1]
    want = {(0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3}[(dr, dc)]
    d = (want - cur_dir) % 4
    return 0 if d == 0 else (2 if d == 1 else 1)


def bfs_path(head, food, blocked):
    """最短路 head→food（4 连通，blocked 为占用格集合）。返回首步格或 None。"""
    if head == tuple(food):
        return None
    q = deque([(head[0], head[1])])
    prev = {head: None}
    while q:
        r, c = q.popleft()
        if (r, c) == tuple(food):
            # 回溯到首步
            cur = (r, c)
            while prev[cur] != head:
                cur = prev[cur]
            return cur
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < G and 0 <= nc < G and (nr, nc) not in prev and (nr, nc) not in blocked:
                prev[(nr, nc)] = (r, c)
                q.append((nr, nc))
    return None


def flood_size(start, occupied):
    """从 start 出发 4 连通洪泛可达的自由格数（start 须自由）。"""
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < G and 0 <= nc < G and (nr, nc) not in seen and (nr, nc) not in occupied:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen)


def flood_region(start, occupied):
    """从 start 出发 4 连通洪泛，返回区域集合（含 start；start 不得在 occupied）。"""
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < G and 0 <= nc < G and (nr, nc) not in seen and (nr, nc) not in occupied:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def safe_step(head, cur_dir, cand, body_set, tail, L, eating):
    """走向 cand 的安全性（经典双重条件）：
    - 不吃食：尾格腾出且必须在新头可达域内（尾可达=可继续卸身）+ 自由空间 ≥ 蛇长；
    - 吃食：尾不动，要求自由空间 ≥ 蛇长+2 的余量。"""
    if eating:
        occ = body_set
        region = flood_region(cand, occ)
        return len(region) - 1 >= L + 2
    occ = body_set - {tail}
    region = flood_region(cand, occ)
    return (tail in region) and (len(region) - 1 >= L)


def rollout_solver(cfg, banks, mode, max_steps=12000):
    """解法器批量局：B=len(banks)，个体 i 打库 i。记录完整轨迹（截断事后做）。

    回路跟随占用回退：初始身体未必是回路连续弧（朝向来自库），回路后继格
    被占时改走任一洪泛安全的自由邻格，≤ 蛇长 步后身体自然落回回路弧。
    """
    B = len(banks)
    env = t7h.BatchedSnakeEnv(cfg, B, DEV)
    bank = {'stream': torch.stack([b['stream'] for b in banks]).to(DEV),
            'dir0': torch.tensor([b['dir0'] for b in banks], dtype=torch.long)}
    env.reset(bank=bank)
    logs = [dict(acts=[], events=[], died=0) for _ in range(B)]

    for t in range(max_steps):
        if not bool(env.alive.any()):
            break
        heads = env.head.cpu().numpy()
        lens = env.body_len.cpu().numpy()
        foods = env.food.cpu().numpy()
        dirs = env.dir_idx.cpu().numpy()
        swof = env.steps_wo_food.cpu().numpy()
        bodies = env.body.cpu().numpy()
        acts = np.zeros(B, dtype=np.int64)
        for i in range(B):
            if not bool(env.alive[i]):
                continue
            head = (int(heads[i][0]), int(heads[i][1]))
            food = (int(foods[i][0]), int(foods[i][1]))
            cur_dir = int(dirs[i])
            L = int(lens[i])
            body = [(int(r), int(c)) for r, c in bodies[i, :max(L, 1)]]
            body_set = set(body)
            tail = body[-1]
            neck = body[1] if L >= 2 else None
            blocked = body_set - {tail}     # 尾格下一步腾出（除非吃食在其上——食物不在身上）

            act = None
            budget = int(3 * L + 20 - swof[i])          # 距饿死剩余步数
            cycle_dist = (CYCLE_POS[food] - CYCLE_POS[head]) % (G * G)
            need_shortcut = budget < cycle_dist + 2     # 跟回路会饿死 → 必须抄近路
            if mode == 's2' and (L < 55 or need_shortcut):
                path_step = bfs_path(head, food, blocked)
                if path_step is not None and path_step != neck:
                    ok = safe_step(head, cur_dir, path_step, body_set, tail, L,
                                   path_step == food)
                    if ok:
                        act = action_toward(cur_dir, head, path_step)
                    elif need_shortcut:
                        # 饿死钟不足：风险分级——可达空间 ≥ 半长仍可冒险抄
                        occ = (body_set - {tail}) if path_step != food else body_set
                        if len(flood_region(path_step, occ)) - 1 >= L // 2:
                            act = action_toward(cur_dir, head, path_step)
            if act is None:
                succ = CYCLE_SUCC[CYCLE_POS[head]]
                if succ not in blocked:
                    act = action_toward(cur_dir, head, succ)
                else:
                    # 回路后继被占：任选洪泛安全的自由邻格；全不安全时选
                    # 洪泛区域最大的邻格（比直行赴死多一线生机）
                    best, best_sz = None, -1
                    for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                        nr, nc = head[0] + dr, head[1] + dc
                        if 0 <= nr < G and 0 <= nc < G and (nr, nc) not in blocked \
                                and (nr, nc) != neck:
                            cand = (nr, nc)
                            if safe_step(head, cur_dir, cand, body_set, tail, L,
                                         cand == food):
                                act = action_toward(cur_dir, head, cand)
                                break
                            occ = (body_set - {tail}) if cand != food else body_set
                            sz = len(flood_region(cand, occ))
                            if sz > best_sz:
                                best, best_sz = cand, sz
                    if act is None and best is not None:
                        act = action_toward(cur_dir, head, best)
            if act is None:
                act = 0  # 彻底被困，直行赴死（记录死因）
            acts[i] = act

        at = torch.from_numpy(acts).to(DEV)
        alive_before = env.alive.clone()
        env.step(at)
        ate = alive_before & env.ate
        ate_np = ate.cpu().numpy()
        al_np = alive_before.cpu().numpy()
        for i in range(B):
            logs[i]['acts'].append(int(acts[i]))
            logs[i]['alive_step'] = logs[i].get('alive_step', 0) + int(al_np[i])
            if ate_np[i]:
                logs[i]['events'].append(t + 1)
        logs_died = env.died.cpu().numpy()
        for i in range(B):
            logs[i]['died'] = int(logs_died[i])
    return logs


def rollout_model(cfg, st, banks, max_steps=20000):
    """模型克隆局：B=len(banks) 个克隆各打各的库。记录完整轨迹。"""
    B = len(banks)
    cfg = _cpu_cfg(cfg)
    pop = t7h.GeneStack(cfg, B=B, device=DEV)
    pop.random_init()
    for i in range(B):
        pop.set_individual_from_state(i, st)
    pop.refresh_eff()
    env = t7h.BatchedSnakeEnv(cfg, B, DEV)
    bank = {'stream': torch.stack([b['stream'] for b in banks]).to(DEV),
            'dir0': torch.tensor([b['dir0'] for b in banks], dtype=torch.long)}
    env.reset(bank=bank)
    N = pop.N
    E = torch.zeros(B, N, dtype=torch.float32, device=DEV)
    I = torch.zeros(B, N, dtype=torch.float32, device=DEV)
    stt = torch.zeros(B, N, dtype=torch.float32, device=DEV)
    press = torch.zeros(B, dtype=torch.float32, device=DEV)
    logs = [dict(acts=[], events=[], died=0) for _ in range(B)]
    for t in range(max_steps):
        if not bool(env.alive.any()):
            break
        obs = env.obs()
        act, E, I, stt = t7h.deliberate_batch(pop, obs, E, I, stt, press, cfg)
        press = t7h.update_fatigue(press, act, decay=float(cfg.FATIGUE_TURN_DECAY))
        alive_before = env.alive.clone()
        env.step(act)
        ate = alive_before & env.ate
        acts_np = act.cpu().numpy()
        ate_np = ate.cpu().numpy()
        al_np = alive_before.cpu().numpy()
        for i in range(B):
            logs[i]['acts'].append(int(acts_np[i]))
            logs[i]['alive_step'] = logs[i].get('alive_step', 0) + int(al_np[i])
            if ate_np[i]:
                logs[i]['events'].append(t + 1)
        died_np = env.died.cpu().numpy()
        for i in range(B):
            logs[i]['died'] = int(died_np[i])
    return logs


def _cpu_cfg(cfg):
    cfg.DEVICE = 'cpu'
    cfg.USE_FP16 = False
    return cfg


def window_metrics(log, N):
    """截断到第 N 食的窗口指标。N=0 → None。"""
    if N <= 0 or len(log['events']) < N:
        return None
    SL = log['events'][N - 1]
    acts = log['acts'][:SL]
    TL = sum(1 for a in acts if a != 0)
    # 最后 5 食区段
    k0 = max(0, N - 5)
    t0 = log['events'][k0] if k0 > 0 else 0
    seg_acts = log['acts'][t0:SL]
    seg_turns = sum(1 for a in seg_acts if a != 0)
    seg_steps = SL - t0
    return dict(N=N, SL=SL, TL=TL, density=TL / max(SL, 1),
                tpf=TL / N, spf=SL / N,
                last5_density=seg_turns / max(seg_steps, 1),
                last5_tpf=seg_turns / min(5, N - k0) if N - k0 > 0 else 0.0)


def fitness_variants(m, eff_w=0.3, cap=8.0):
    """候选适应度：输入 window_metrics 输出的 dict。"""
    N, SL, TL = m['N'], m['SL'], m['TL']
    eff = N / max(SL, 1)
    te = cap if TL <= 0 else min(SL / max(TL, 1), cap)
    te_pts = te / cap
    tpf_pen = max(0.0, 1.0 - (TL / N) / 10.0)
    return {
        'cur_W0.5': N + eff_w * eff + 0.5 * te_pts,
        'ratio_W1.5': N + eff_w * eff + 1.5 * te_pts,
        'ratio_W3.0': N + eff_w * eff + 3.0 * te_pts,
        'tpf_W3.0': N + eff_w * eff + 3.0 * tpf_pen,
        'noeff_W3.0': N + 3.0 * te_pts,
    }


def load_model(path):
    data = torch.load(os.path.join(ROOT, path), map_location='cpu', weights_only=False)
    cfg = t7h.Config()
    cfg.OBS_MANHATTAN = False          # pre-7g 键缺失时回退欧氏
    cfg.FATIGUE_TURN_GAIN = 0.0
    for k, v in data.get('config', {}).items():
        if hasattr(cfg, k) and isinstance(v, (int, float, bool, str)):
            setattr(cfg, k, v)
    cfg = _cpu_cfg(cfg)
    cfg.MAX_STEPS = 20000
    cfg.EVAL_EPISODES = EPS
    return data['brain'], cfg, data


def paired_compare(solver_logs, model_logs, protocol='common'):
    """配对比较。protocol='common'：每库 N = min(模型食数, 解法器食数)——
    同食数下比较路径质量（全库可用）；'model'：N = 模型食数（严格，
    仅统计解法器达到该食数的库）。"""
    out = {}
    per_variant = {k: dict(wins=0, n=0, margins=[]) for k in fitness_variants(
        dict(N=1, SL=10, TL=1))}
    profiles = dict(solver=[], model=[])
    for i in range(len(model_logs)):
        model_N = len(model_logs[i]['events'])
        solver_N = len(solver_logs[i]['events'])
        N = min(model_N, solver_N) if protocol == 'common' else model_N
        if N <= 0:
            continue
        sm = window_metrics(solver_logs[i], N)
        mm = window_metrics(model_logs[i], N)
        if sm is None or mm is None:
            continue
        fs, fm = fitness_variants(sm), fitness_variants(mm)
        for k in per_variant:
            per_variant[k]['n'] += 1
            d = fs[k] - fm[k]
            per_variant[k]['margins'].append(d)
            if d > 0:
                per_variant[k]['wins'] += 1
        profiles['solver'].append(sm)
        profiles['model'].append(mm)
    for k, v in per_variant.items():
        if v['n'] == 0 or not v['margins']:
            out[k] = dict(win_rate=float('nan'), margin_mean=float('nan'),
                          margin_min=float('nan'), n=0)
            continue
        mg = np.array(v['margins'])
        out[k] = dict(win_rate=v['wins'] / max(v['n'], 1),
                      margin_mean=float(mg.mean()), margin_min=float(mg.min()),
                      n=v['n'])
    prof = {}
    for side in ('solver', 'model'):
        rows = profiles[side]
        prof[side] = dict(
            density=float(np.mean([r['density'] for r in rows])),
            tpf=float(np.mean([r['tpf'] for r in rows])),
            spf=float(np.mean([r['spf'] for r in rows])),
            last5_density=float(np.mean([r['last5_density'] for r in rows])),
            last5_tpf=float(np.mean([r['last5_tpf'] for r in rows])),
            food=float(np.mean([r['N'] for r in rows])))
    return out, prof


def main():
    base = t7h.Config()
    base = _cpu_cfg(base)
    banks = t7h.make_banks(base, 999, 7, EPS, DEV)
    result = {}

    print(f'[解法器] S1 纯回路 / S2 回路+安全捷径，各 {EPS} 库 ...')
    s1 = rollout_solver(base, banks, 's1')
    s2 = rollout_solver(base, banks, 's2')
    for name, logs in (('S1', s1), ('S2', s2)):
        full = [len(l['events']) for l in logs]
        dd = [l['died'] for l in logs]
        n_wall, n_self, n_starve = (dd.count(1), dd.count(2), dd.count(3))
        print(f'  {name}: 完整局食物 mean {np.mean(full):.1f} / min {min(full)} / '
              f'max {max(full)} | 死因 墙{n_wall}/撞己{n_self}/饿死{n_starve}')
        result[f'{name}_full'] = dict(food_mean=float(np.mean(full)),
                                      food_min=int(min(full)), food_max=int(max(full)),
                                      died_wall=n_wall, died_self=n_self,
                                      died_starve=n_starve)

    for tag, path in (('7h_best', 'test7h_econ_best_model.pth'),
                      ('7b_best', 'artifacts/test7b/test7b_best_model.pth')):
        st, cfg, meta = load_model(path)
        print(f'[模型] {tag} <- {path} | manhattan={cfg.OBS_MANHATTAN} | '
              f'训练时 food={meta.get("food")}')
        mlogs = rollout_model(cfg, st, banks)
        foods = [len(l['events']) for l in mlogs]
        print(f'  food mean {np.mean(foods):.1f} ± {np.std(foods):.1f} '
              f'(min {min(foods)} / max {max(foods)})')
        for sname, slogs in (('S1', s1), ('S2', s2)):
            for proto in ('common', 'model'):
                cmp_, prof = paired_compare(slogs, mlogs, protocol=proto)
                result[f'{tag}_vs_{sname}_{proto}'] = dict(
                    fitness=cmp_, profiles=prof,
                    model_food_mean=float(np.mean(foods)))
                if proto == 'common' or cmp_['cur_W0.5']['n'] > 0:
                    print(f'  --- {sname} [{proto}] 配对（n={cmp_["cur_W0.5"]["n"]}）---')
                    print(f'  {"公式":<12}{"胜率":>8}{"平均差距":>10}{"最小差距":>10}')
                    for k in ('cur_W0.5', 'ratio_W1.5', 'ratio_W3.0', 'tpf_W3.0',
                              'noeff_W3.0'):
                        v = cmp_[k]
                        if v['n'] == 0:
                            continue
                        print(f'  {k:<12}{v["win_rate"]:>8.1%}{v["margin_mean"]:>10.2f}'
                              f'{v["margin_min"]:>10.2f}')
                    if cmp_['cur_W0.5']['n'] > 0:
                        for side in ('solver', 'model'):
                            p = prof[side]
                            print(f'    [{side}] 密度 {p["density"]:.2f} | TPF {p["tpf"]:.1f} | '
                                  f'步/食 {p["spf"]:.0f} | 末5食密度 {p["last5_density"]:.2f} | '
                                  f'末5食TPF {p["last5_tpf"]:.1f}')

    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    with open(os.path.join(ROOT, 'results', 'test7h_solver_reference.json'), 'w',
              encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print('[落盘] results/test7h_solver_reference.json')


if __name__ == '__main__':
    main()
