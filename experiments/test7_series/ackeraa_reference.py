# ==========================================
# experiments/ackeraa_reference.py —— Ackeraa nn_97 参考蛇标定（免饿死钟，带棋盘可视化）
#
# 协议（用户定稿）：
#  - 参考蛇 = Ackeraa/snake 项目 nn_97（前馈 [32,12,8,4]，8 射线观测 + 头/尾 one-hot，
#    绝对方向输出，其原生环境无饿死钟）——在我们的 BatchedSnakeEnv 中忠实复刻，
#    STARVE_SLOPE=1e9 免饿死钟（撞墙/撞己照常），同 40 个 CRN 库；
#  - 7h best 同库 40 局（原生规则含饿死钟）；
#  - 参考蛇在每个体对应的模型食数 N 处截断 → 同食数配对比较候选适应度；
#  - 可视化：两玩家在食物数 15/25/35/N 的棋盘+路径对比 PNG；
#  - 若参考蛇无钟也大面积早死/到不了 N → 停下问用户，不自行改设计。
# 产出：results/ackeraa_reference.json / ackeraa_vs_7h_boards.png / ackeraa_vs_7h_paths.png
# ==========================================
import importlib.util
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'experiments', 'test7_series'))  # 归档后模块路径
import test7h as t7h  # noqa: E402
from solver_reference import window_metrics  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
G = 10
EPS = 40
DEV = torch.device('cpu')

# Ackeraa 射线方向（其 (dx,dy)=(col,row) → 本 env (dr,dc)=(dy,dx)）
ACK_RAYS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
# Ackeraa DIRECTIONS [(0,-1),(0,1),(-1,0),(1,0)] in (dx,dy) → 本 env (dr,dc)
ACK_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]      # 0上 1下 2左 3右
OUR_DIR_IDX = {(0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3}   # test7h DIRS 表


def load_ackeraa_net(path):
    spec = importlib.util.spec_from_file_location('nn', os.path.join(ROOT, 'third_party/ackeraa/nn.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    sys.modules['nn'] = m
    net = torch.load(path, map_location='cpu', weights_only=False)
    net.eval()
    return net


def ackeraa_states(heads, foods, bodies, lens, B):
    """按 Ackeraa get_state 逐个体构造 32 维观测（含 1/dis 墙距、射线食物/身体布尔、
    头/尾方向 one-hot）。"""
    states = []
    for i in range(B):
        head = (int(heads[i][0]), int(heads[i][1]))
        food = (int(foods[i][0]), int(foods[i][1]))
        L = int(lens[i])
        body = [(int(r), int(c)) for r, c in bodies[i, :max(L, 1)]]
        body_set = set(body)
        neck = body[1] if L >= 2 else head
        tail, tail_prev = body[-1], (body[-2] if L >= 2 else body[-1])

        st = []
        for dr, dc in ACK_RAYS:
            r, c = head[0] + dr, head[1] + dc
            dis = 1.0
            see_food = see_self = 0.0
            while 0 <= r < G and 0 <= c < G:
                if (r, c) == food:
                    see_food = 1.0
                elif (r, c) in body_set:
                    see_self = 1.0
                dis += 1.0
                r += dr
                c += dc
            st += [1.0 / dis, see_food, see_self]
        hd = (head[0] - neck[0], head[1] - neck[1])
        td = (tail_prev[0] - tail[0], tail_prev[1] - tail[1])
        for vec in (hd, td):
            onehot = [0.0] * 4
            onehot[ACK_DIRS.index(vec)] = 1.0
            st += onehot
        states.append(st)
    return states


def rollout_reference(cfg, banks, net, max_steps=20000):
    """参考蛇（免饿死钟）批量局：B=len(banks)，个体 i 打库 i。记录完整轨迹。"""
    B = len(banks)
    env = t7h.BatchedSnakeEnv(cfg, B, DEV)
    bank = {'stream': torch.stack([b['stream'] for b in banks]).to(DEV),
            'dir0': torch.tensor([b['dir0'] for b in banks], dtype=torch.long)}
    env.reset(bank=bank)
    logs = [dict(acts=[], events=[], path=[], snaps=[], died=0) for _ in range(B)]

    for t in range(max_steps):
        if not bool(env.alive.any()):
            break
        heads = env.head.cpu().numpy()
        lens = env.body_len.cpu().numpy()
        foods = env.food.cpu().numpy()
        bodies = env.body.cpu().numpy()
        dirs = env.dir_idx.cpu().numpy()
        states = torch.tensor(ackeraa_states(heads, foods, bodies, lens, B),
                              dtype=torch.float32)
        with torch.no_grad():
            out = net(states)
        abs_dir = out.argmax(dim=1).numpy()

        acts = np.zeros(B, dtype=np.int64)
        for i in range(B):
            if not bool(env.alive[i]):
                continue
            want = ACK_DIRS[int(abs_dir[i])]
            d = (OUR_DIR_IDX[want] - int(dirs[i])) % 4
            acts[i] = 0 if d == 0 else (2 if d == 1 else 1)

        at = torch.from_numpy(acts).to(DEV)
        alive_before = env.alive.clone()
        env.step(at)
        ate = alive_before & env.ate
        ate_np = ate.cpu().numpy()
        al_np = alive_before.cpu().numpy()
        hs = env.head.cpu().numpy()
        fs = env.food.cpu().numpy()
        bs = env.body.cpu().numpy()
        ls2 = env.body_len.cpu().numpy()
        for i in range(B):
            if not al_np[i]:
                continue
            logs[i]['acts'].append(int(acts[i]))
            logs[i]['path'].append((int(hs[i][0]), int(hs[i][1])))
            if ate_np[i]:
                logs[i]['events'].append(t + 1)
                logs[i]['snaps'].append((
                    t + 1,
                    [(int(r), int(c)) for r, c in bs[i, :int(ls2[i])]],
                    (int(fs[i][0]), int(fs[i][1]))))
        died_np = env.died.cpu().numpy()
        for i in range(B):
            logs[i]['died'] = int(died_np[i])
    return logs


def rollout_model_traj(cfg, st, banks, max_steps=20000):
    """7h best 克隆局（原生规则），同库，记录完整轨迹。"""
    B = len(banks)
    cfg.DEVICE = 'cpu'
    cfg.USE_FP16 = False
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
    logs = [dict(acts=[], events=[], path=[], snaps=[], died=0) for _ in range(B)]
    for t in range(max_steps):
        if not bool(env.alive.any()):
            break
        obs = env.obs()
        act, E, I, stt = t7h.deliberate_batch(pop, obs, E, I, stt, press, cfg)
        press = t7h.update_fatigue(press, act, decay=float(cfg.FATIGUE_TURN_DECAY))
        alive_before = env.alive.clone()
        env.step(act)
        ate = alive_before & env.ate
        act_np = act.cpu().numpy()
        ate_np = ate.cpu().numpy()
        al_np = alive_before.cpu().numpy()
        hs = env.head.cpu().numpy()
        fs = env.food.cpu().numpy()
        bs = env.body.cpu().numpy()
        ls2 = env.body_len.cpu().numpy()
        for i in range(B):
            if not al_np[i]:
                continue
            logs[i]['acts'].append(int(act_np[i]))
            logs[i]['path'].append((int(hs[i][0]), int(hs[i][1])))
            if ate_np[i]:
                logs[i]['events'].append(t + 1)
                logs[i]['snaps'].append((
                    t + 1,
                    [(int(r), int(c)) for r, c in bs[i, :int(ls2[i])]],
                    (int(fs[i][0]), int(fs[i][1]))))
        died_np = env.died.cpu().numpy()
        for i in range(B):
            logs[i]['died'] = int(died_np[i])
    return logs


def fitness_candidates(m):
    N, SL, TL = m['N'], m['SL'], m['TL']
    eff = N / max(SL, 1)
    out = {'food_only': N + 0.3 * eff}
    for w in (1.5, 2.0, 3.0):
        out[f'ratio_W{w}_C4'] = N + 0.3 * eff + w * min(SL / max(TL, 1), 4.0) / 4.0
    out['ratio_W3_C8'] = N + 0.3 * eff + 3.0 * min(SL / max(TL, 1), 8.0) / 8.0
    out['tpf_W3'] = N + 0.3 * eff + 3.0 * max(0.0, 1.0 - (TL / N) / 10.0)
    return out


def draw_board(ax, snap, path_window, title):
    """棋盘：蛇身按龄渐变（旧深新浅），头部白，食物红，近5食轨迹蓝线。"""
    t, cells, food = snap
    grid = np.zeros((G, G, 3))
    n = len(cells)
    for j, (r, c) in enumerate(cells):
        age = j / max(n - 1, 1)          # 0=尾(旧) → 1=头(新)
        grid[r, c] = [0.15 + 0.35 * age, 0.25 + 0.35 * age, 0.7 + 0.25 * age]
    hr, hc = cells[0]
    grid[hr, hc] = [1.0, 1.0, 1.0]
    ax.imshow(grid, origin='upper')
    if path_window:
        pr = [p[0] for p in path_window]
        pc = [p[1] for p in path_window]
        ax.plot(pc, pr, '-', color='orange', linewidth=1.4, alpha=0.9)
        ax.plot(pc[-1], pr[-1], 'o', color='orange', markersize=3)
    ax.plot(food[1], food[0], 's', color='red', markersize=9)
    ax.set_xticks(range(G))
    ax.set_yticks(range(G))
    ax.grid(True, color='gray', linewidth=0.3, alpha=0.4)
    ax.set_title(title, fontsize=9)


def main():
    base = t7h.Config()
    base.DEVICE = 'cpu'
    base.USE_FP16 = False
    base.STARVE_SLOPE = 1e9            # 免饿死钟（仅参考蛇）
    base.MAX_STEPS = 20000
    banks = t7h.make_banks(base, 777, 1, EPS, DEV)

    print(f'[参考蛇] Ackeraa nn_97（免饿死钟），{EPS} 库 ...')
    net = load_ackeraa_net(os.path.join(ROOT, 'third_party/ackeraa/nn_97.pth'))
    rlogs = rollout_reference(base, banks, net)
    rfood = [len(l['events']) for l in rlogs]
    rdied = [l['died'] for l in rlogs]
    print(f'  完整局食物 mean {np.mean(rfood):.1f} ± {np.std(rfood):.1f} '
          f'(min {min(rfood)} / max {max(rfood)}) | '
          f'死因 墙{rdied.count(1)}/撞己{rdied.count(2)}/饿死{rdied.count(3)}')

    print('[模型] 7h best（原生规则，同库）...')
    mdata = torch.load(os.path.join(ROOT, 'test7h_econ_best_model.pth'),
                       map_location='cpu', weights_only=False)
    mcfg = t7h.Config()
    mcfg.OBS_MANHATTAN = False
    mcfg.FATIGUE_TURN_GAIN = 0.0
    for k, v in mdata.get('config', {}).items():
        if hasattr(mcfg, k) and isinstance(v, (int, float, bool, str)):
            setattr(mcfg, k, v)
    mcfg.MAX_STEPS = 20000
    mlogs = rollout_model_traj(mcfg, mdata['brain'], banks)
    mfood = [len(l['events']) for l in mlogs]
    mdied = [l['died'] for l in mlogs]
    print(f'  food mean {np.mean(mfood):.1f} ± {np.std(mfood):.1f} '
          f'(min {min(mfood)} / max {max(mfood)}) | '
          f'死因 墙{mdied.count(1)}/撞己{mdied.count(2)}/饿死{mdied.count(3)}')

    # ---- 配对截断（模型食数 N）----
    result = dict(ref_food=rfood, model_food=mfood)
    keys = list(fitness_candidates(dict(N=1, SL=10, TL=1)).keys())
    acc = {k: dict(wins=0, n=0, margins=[]) for k in keys}
    prof_r, prof_m = [], []
    reach = 0
    for i in range(EPS):
        N = len(mlogs[i]['events'])
        if N <= 0 or len(rlogs[i]['events']) < N:
            continue
        reach += 1
        rm = window_metrics(rlogs[i], N)
        mm = window_metrics(mlogs[i], N)
        fr, fm = fitness_candidates(rm), fitness_candidates(mm)
        for k in keys:
            d = fr[k] - fm[k]
            acc[k]['n'] += 1
            acc[k]['margins'].append(d)
            if d > 0:
                acc[k]['wins'] += 1
        prof_r.append(rm)
        prof_m.append(mm)
    print(f'\n[配对] 参考蛇到达模型食数的库：{reach}/{EPS}')
    if reach < EPS * 0.6:
        print('  [警告] 参考蛇大面积到不了模型食数 —— 按协议停下问用户')
    print(f'  {"公式":<14}{"胜率":>8}{"平均差距":>10}{"最小差距":>10}')
    for k in keys:
        v = acc[k]
        if v['n'] == 0:
            print(f'  {k:<14}  n=0')
            continue
        mg = np.array(v['margins'])
        result[f'fit_{k}'] = dict(win_rate=v['wins'] / v['n'],
                                  margin_mean=float(mg.mean()),
                                  margin_min=float(mg.min()), n=v['n'])
        print(f'  {k:<14}{v["wins"] / v["n"]:>8.1%}{mg.mean():>10.2f}{mg.min():>10.2f}')
    for side, prof in (('ref', prof_r), ('model', prof_m)):
        if prof:
            result[f'prof_{side}'] = dict(
                density=float(np.mean([p['density'] for p in prof])),
                tpf=float(np.mean([p['tpf'] for p in prof])),
                spf=float(np.mean([p['spf'] for p in prof])),
                last5_density=float(np.mean([p['last5_density'] for p in prof])),
                last5_tpf=float(np.mean([p['last5_tpf'] for p in prof])))
            p = result[f'prof_{side}']
            print(f'  [{side}] 密度 {p["density"]:.2f} | TPF {p["tpf"]:.1f} | '
                  f'步/食 {p["spf"]:.0f} | 末5食密度 {p["last5_density"]:.2f}')

    # ---- 可视化：选中位库，食物数 15/25/35/N 四局面 ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        order = np.argsort([len(l['events']) for l in mlogs])
        bi = int(order[len(order) // 2])          # 中位模型局
        N = len(mlogs[bi]['events'])
        stops = [k for k in (15, 25, 35, N) if k <= N]
        if len(stops) > 1 and stops[-1] == stops[-2]:
            stops = stops[:-1]
        fig, axes = plt.subplots(len(stops), 2, figsize=(8, 4 * len(stops)))
        if len(stops) == 1:
            axes = axes.reshape(1, 2)
        for row, k in enumerate(stops):
            for col, (logs, tag) in enumerate(((rlogs, 'Ackeraa nn_97'), (mlogs, '7h best'))):
                lg = logs[bi]
                if len(lg['snaps']) < k:
                    axes[row][col].axis('off')
                    continue
                t0 = lg['events'][max(0, k - 6)] if k >= 6 else 0
                t1 = lg['events'][k - 1]
                window = lg['path'][t0:t1]
                draw_board(axes[row][col], lg['snaps'][k - 1], window,
                           f'{tag} @food={k} (bank {bi})')
        plt.tight_layout()
        out_png = os.path.join(ROOT, 'results', 'ackeraa_vs_7h_boards.png')
        fig.savefig(out_png, dpi=110)
        plt.close(fig)
        print(f'[落盘] {out_png}')

        # 全轨迹对比
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for col, (logs, tag) in enumerate(((rlogs, 'Ackeraa nn_97 full path'), (mlogs, '7h best full path'))):
            lg = logs[bi]
            pr = [p[0] for p in lg['path']]
            pc = [p[1] for p in lg['path']]
            axes[col].plot(pc, pr, '-', linewidth=0.7, alpha=0.6)
            axes[col].set_xlim(-0.5, G - 0.5)
            axes[col].set_ylim(G - 0.5, -0.5)
            axes[col].set_title(f'{tag} (food={len(lg["events"])})', fontsize=9)
            axes[col].grid(True, alpha=0.3)
        plt.tight_layout()
        out_png2 = os.path.join(ROOT, 'results', 'ackeraa_vs_7h_paths.png')
        fig.savefig(out_png2, dpi=110)
        plt.close(fig)
        print(f'[落盘] {out_png2}')
    except Exception as e:
        print(f'(绘图跳过: {e})')

    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    with open(os.path.join(ROOT, 'results', 'ackeraa_reference.json'), 'w',
              encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print('[落盘] results/ackeraa_reference.json')


if __name__ == '__main__':
    main()
