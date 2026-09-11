# ==========================================
# test7d_diag.py —— 频繁转向根因完整诊断（零训练）
#
# 对指定模型（默认 test7d_econ_best）并发跑 ~200 局，产出四类证据：
#  A. 转弯结构：每 20 步窗口转向密度时间线、直行游程长度分布、
#     死亡前 60 步密度 vs 全局密度（挣扎段是否恶化）
#  B. argmax 抖动：每步 top-2 logits 边际分布，按"决策=直行/转向"分组
#     （转向决策边际显著更小 → 抖动假设成立）
#  C. 权重范数：多个历代最优模型 W_out/W_in/b_out 的 L2 范数对比
#     （逐代膨胀 → 疲劳军备竞赛假设成立）
#  D. 空间碎片化：吃食/死亡时刻自由格最大连通域占比，与前一窗口转弯
#     密度的相关（密集转弯→空洞/腾挪空间缩小的因果证据）
# ==========================================
import os, sys, time, importlib.util, math
from collections import defaultdict, deque

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location('t7d', os.path.join(ROOT, 'experiments', 'test7_series', 'test7d.py'))
t7d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t7d)

import numpy as np
import torch

GAMES = 200
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'test7d_econ_best_model.pth'


# ---------- C. 权重范数 ----------
def weight_norms():
    print('\n===== C. 历代模型权重范数（军备竞赛检验）=====')
    cfg = t7d.Config()
    models = ['test7a_v5a_best_model.pth', 'test7b_latest_gen_best.pth',
              'test7c_best_model.pth', 'test7d_econ_best_model.pth']
    print(f"{'model':<38} {'W_out':>8} {'W_in':>8} {'b_out':>8} {'W_rec':>8}")
    for p in models:
        if not os.path.exists(p):
            print(f'{p:<38} (缺失)')
            continue
        st = torch.load(p, map_location='cpu', weights_only=False)['brain']
        f = lambda k: float(st[k].float().norm())
        print(f'{p:<38} {f("W_out"):>8.2f} {f("W_in"):>8.2f} {f("b_out"):>8.2f} {f("W_rec"):>8.2f}')


# ---------- 自由格最大连通域占比（CPU flood fill）----------
def largest_free_ratio(grid):
    G = grid.shape[0]
    free = ~grid.astype(bool)
    seen = np.zeros_like(free)
    best = 0
    total_free = int(free.sum())
    if total_free == 0:
        return 0.0
    for r in range(G):
        for c in range(G):
            if free[r, c] and not seen[r, c]:
                size, q = 0, [(r, c)]
                seen[r, c] = True
                while q:
                    y, x = q.pop()
                    size += 1
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < G and 0 <= nx < G and free[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((ny, nx))
                best = max(best, size)
    return best / total_free


def run_behavior(model_path, games=GAMES):
    cfg = t7d.Config()
    cfg.EVAL_EPISODES = 1
    cfg.SEED_FROM_BEST = False
    dev = t7d._resolve_device(cfg)
    res = t7d.load_best_state(model_path, cfg)
    if res is None:
        print(f'错误: 无法加载 {model_path}')
        sys.exit(1)
    st, _, _ = res
    print(f'model = {model_path}  games = {games}  device = {dev}')

    B = games
    pop = t7d.GeneStack(cfg, B=B, device=dev)
    pop.random_init()
    for i in range(B):
        pop.set_individual_from_state(i, st)
    pop.refresh_eff()
    half = torch.float16
    pop.fp16(); pop.refresh_eff()

    env = t7d.BatchedSnakeEnv(cfg, B, dev)
    G = cfg.GRID_SIZE
    N, A = pop.N, pop.A
    K = cfg.FRAME_RATE

    E = torch.zeros(B, N, dtype=half, device=dev)
    I = torch.zeros(B, N, dtype=half, device=dev)
    stt = torch.zeros(B, N, dtype=half, device=dev)
    press = torch.zeros(B, dtype=torch.float32, device=dev)

    ar = torch.arange(B, device=dev)
    # 累积器
    margin_turn, margin_straight = [], []          # top-2 logits 边际（分组）
    straight_runs = []                              # 直行游程长度
    win_turns = np.zeros(60)                        # 每 20 步窗口转向数（前 1200 步）
    win_steps = np.zeros(60)
    density_global = np.zeros(B)                    # 全局密度分子/分母
    steps_global = np.zeros(B)
    pre_death_density = np.zeros(B)                 # 死亡前 60 步
    pre_death_steps = np.zeros(B)
    frag_events = []                                # (前窗密度, 最大连通域占比, 蛇长)
    act_buf = torch.zeros(B, dtype=torch.long, device=dev) - 1  # 上一步动作（游程统计）
    run_len = torch.zeros(B, dtype=torch.long, device=dev)
    recent = [deque(maxlen=20) for _ in range(0)]   # 不用，改用环形张量
    ring = torch.zeros(B, 20, dtype=torch.long, device=dev)     # 最近 20 步动作
    ring_ptr = 0
    t0 = time.perf_counter()

    for t in range(cfg.MAX_STEPS):
        al = env.alive
        if not al.any():
            break
        obs = env.obs().to(half)
        # K 帧思考（复刻 deliberate_batch，但记录 logits_sum）
        logits_sum = None
        for k in range(K):
            o = obs * (cfg.INPUT_DECAY ** k)
            logits, E, I, stt = t7d.forward_batch(pop, o, E, I, stt, press, cfg)
            logits_sum = logits if logits_sum is None else logits_sum + logits
        top2 = logits_sum.float().topk(2, dim=1)
        margin = (top2.values[:, 0] - top2.values[:, 1])
        act = top2.indices[:, 0]

        # margin 分组记录（仅存活个体）
        is_turn = (act != 0)
        margin_turn.extend(margin[al & is_turn].cpu().tolist())
        margin_straight.extend(margin[al & (~is_turn)].cpu().tolist())

        # 直行游程（先读后写：转向步会把 run_len 清零，须在更新前记录已积累的游程）
        ended_run = al & is_turn & (act_buf == 0)
        straight_runs.extend(run_len[ended_run].cpu().tolist())
        cont = (act == act_buf) & (act == 0) & al
        run_len = torch.where(cont, run_len + 1, torch.where(al & (act == 0),
                                torch.ones_like(run_len), torch.zeros_like(run_len)))
        act_buf = torch.where(al, act, act_buf)

        # 环形窗口 / 全局密度
        ring[:, ring_ptr % 20] = act * al.long()
        ring_ptr += 1
        density_global += (al & is_turn).float().cpu().numpy()
        steps_global += al.float().cpu().numpy()
        if t // 20 < 60:
            win_turns[t // 20] += (al & is_turn).float().sum().item()
            win_steps[t // 20] += al.float().sum().item()

        press = t7d.update_fatigue(press, act, decay=float(cfg.FATIGUE_TURN_DECAY))
        prev_alive = al.clone()
        env.step(act)
        ate_now = al & env.ate
        died_now = prev_alive & (~env.alive)

        # 碎片化采样：每局最多 3 次吃食 + 死亡
        if ate_now.any() or died_now.any():
            sel = ate_now | died_now
            idxs = sel.nonzero(as_tuple=True)[0].cpu().tolist()
            for i in idxs[:64]:  # 每步最多采样 64 个个体，控制 flood fill 成本
                key = (int(i), int(env.body_len[i]))
                if died_now[i] or (ate_now[i] and env.body_len[i] % 15 == 0):
                    body = env.body[i, :env.body_len[i]].cpu().numpy()
                    grid = np.zeros((G, G), dtype=np.int32)
                    grid[body[:, 0], body[:, 1]] = 1
                    dens = float((ring[i] != 0).float().mean().item())
                    frag_events.append((dens, largest_free_ratio(grid), int(env.body_len[i])))

        if died_now.any():
            pre_death_density += ((ring != 0).float().sum(dim=1) * died_now.float()).cpu().numpy()
            pre_death_steps += (20.0 * died_now.float()).cpu().numpy()

    elapsed = time.perf_counter() - t0
    print(f'仿真完成 ({elapsed:.0f}s)')

    # ===== 输出 =====
    print('\n===== A. 转弯结构 =====')
    dens_timeline = np.where(win_steps > 0, win_turns / np.maximum(win_steps, 1), 0)
    print('每 20 步窗口转向密度（前 600 步）:')
    for w in range(0, 30):
        print(f'  step {w*20:>4}-{w*20+19:<4}: {dens_timeline[w]:.3f}')
    gd = density_global.sum() / max(steps_global.sum(), 1)
    pd_ = pre_death_density.sum() / max(pre_death_steps.sum(), 1)
    print(f'全局转向密度: {gd:.3f}   死亡前 20 步密度: {pd_:.3f}')
    sr = np.array([r for r in straight_runs if r > 0])
    if len(sr):
        print(f'直行游程长度: mean={sr.mean():.2f} P25={np.percentile(sr,25):.0f} '
              f'P50={np.percentile(sr,50):.0f} P75={np.percentile(sr,75):.0f} max={sr.max()}')
        print(f'  游程=1 占比: {(sr==1).mean()*100:.1f}%  游程≤2 占比: {(sr<=2).mean()*100:.1f}%')

    print('\n===== B. argmax 边际（抖动检验）=====')
    mt = np.array(margin_turn); ms = np.array(margin_straight)
    if len(mt) and len(ms):
        print(f'转向决策 top-2 边际: mean={mt.mean():.3f} P25={np.percentile(mt,25):.3f} P50={np.percentile(mt,50):.3f}')
        print(f'直行决策 top-2 边际: mean={ms.mean():.3f} P25={np.percentile(ms,25):.3f} P50={np.percentile(ms,50):.3f}')
        print(f'转向决策中边际<0.3 占比: {(mt<0.3).mean()*100:.1f}%（高 → 抖动假设成立）')

    print('\n===== D. 碎片化 × 窗口转弯密度 =====')
    if len(frag_events) > 10:
        d = np.array([e[0] for e in frag_events])
        f = np.array([e[1] for e in frag_events])
        L = np.array([e[2] for e in frag_events])
        print(f'样本数 {len(d)}  蛇长 mean={L.mean():.0f}')
        print(f'自由格最大连通域占比: mean={f.mean():.3f} P25={np.percentile(f,25):.3f}')
        for lo, hi in [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]:
            m = (d >= lo) & (d < hi)
            if m.sum() > 5:
                print(f'  前窗转弯密度 [{lo:.1f},{hi:.1f}): n={m.sum():>4} '
                      f'连通域占比 mean={f[m].mean():.3f}')
        print(f'相关系数(密度 vs 连通域占比): {np.corrcoef(d, f)[0,1]:.3f}')


if __name__ == '__main__':
    weight_norms()
    run_behavior(MODEL)
