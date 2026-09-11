# ==========================================
# test7f_diag_bearing.py —— 对角线吸引子假设探针（零训练）
#
# 假设：食物 8 扇区投影把"期望航向"编码为连续量，对角方位的期望航向
#       在网格上只能以 L,R 交替实现 → 连续转弯走斜线 = 低自由能吸引子。
# 验证：
#  A. 方位→动作映射：空盘、食物固定距离、扫描 16 个方位，看对角方位
#     是否输出交替转向 / 正交方位输出直行（网络内在的对角偏好）；
#  B. 对角扇区屏蔽消融：真实对局中把食物对角通道 obs[9,11,13,15] 置零，
#     若摆动立即坍缩（密度大降）→ 假设成立。
# ==========================================
import os, sys, time, importlib.util, math

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location('t7f', os.path.join(ROOT, 'experiments', 'test7_series', 'test7f.py'))
t7f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t7f)

import numpy as np
import torch

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'test7f_econ_best_model.pth'


def load_pop(cfg, model_path, dev):
    res = t7f.load_best_state(model_path, cfg)
    if res is None:
        print(f'错误: 无法加载 {model_path}')
        sys.exit(1)
    st, _, _ = res
    pop = t7f.GeneStack(cfg, B=1, device=dev)
    pop.random_init()
    pop.set_individual_from_state(0, st)
    pop.refresh_eff()
    pop.fp16(); pop.refresh_eff()
    return pop


# ---------- A. 方位→动作映射 ----------
def bearing_map(model_path):
    cfg = t7f.Config()
    dev = t7f._resolve_device(cfg)
    pop = load_pop(cfg, model_path, dev)
    half = torch.float16
    N = pop.N
    E = torch.zeros(1, N, dtype=half, device=dev)
    I = torch.zeros(1, N, dtype=half, device=dev)
    stt = torch.zeros(1, N, dtype=half, device=dev)
    press = torch.zeros(1, dtype=torch.float32, device=dev)

    env = t7f.BatchedSnakeEnv(cfg, 1, dev)   # 只为借用 obs() 构造器
    G = cfg.GRID_SIZE
    env.reset()
    # 手工设置：头在中心、朝右、蛇长 2（无自身遮蔽）、食物按方位摆放
    env.head[0] = torch.tensor([G // 2, G // 2], device=dev)
    env.dir_idx[0] = 0                        # 右
    env.body[0, 0] = env.head[0]
    env.body[0, 1] = env.head[0] - env.DIRS[0]
    env.body_len[0] = 2

    print(f'\n===== A. 方位→动作映射（{os.path.basename(model_path)}）=====')
    print('（前=直行 L=左转 R=右转；食物距离 4 格；空盘无墙影响）')
    print(f"{'方位角':>8} {'位置':>10} -> 动作")
    # 朝右 (0,1)。方位角 0=正前（右），逆时针为正（上方向）
    for ang_deg in range(-180, 180, 22):
        rad = math.radians(ang_deg)
        # 屏幕坐标：row 向下。前方=(0,1)。方位角正 = 顺时针（向下）偏转
        dr = int(round(4 * math.sin(rad)))
        dc = int(round(4 * math.cos(rad)))
        if abs(dr) + abs(dc) == 0:
            continue
        env.food[0] = torch.tensor([G // 2 + dr, G // 2 + dc], device=dev)
        obs = env.obs().to(half)
        act, E, I, stt = t7f.deliberate_batch(pop, obs, E, I, stt, press, cfg)
        act = int(act.item())
        sym = ['前', 'L', 'R'][act]
        diag = '对角' if abs(dr) > 0 and abs(dc) > 0 else '正交'
        print(f"{ang_deg:>7}° ({dr:+d},{dc:+d}) -> {sym}   [{diag}]")


# ---------- B. 对角扇区屏蔽消融 ----------
def ablation(model_path, games=150):
    cfg = t7f.Config()
    cfg.EVAL_EPISODES = 1
    dev = t7f._resolve_device(cfg)
    B = games
    res = t7f.load_best_state(model_path, cfg)
    st, _, _ = res
    pop = t7f.GeneStack(cfg, B=B, device=dev)
    pop.random_init()
    for i in range(B):
        pop.set_individual_from_state(i, st)
    pop.refresh_eff(); pop.fp16(); pop.refresh_eff()
    env = t7f.BatchedSnakeEnv(cfg, B, dev)
    N, half = pop.N, torch.float16

    print(f'\n===== B. 对角食物扇区屏蔽消融（{os.path.basename(model_path)}，{games} 局）=====')
    print(f"{'屏蔽':>6} | {'food':>7} {'密度':>6} {'游程≤2':>6} | 撞墙%  自撞%  饿死%")
    for mask_diag in [False, True]:
        env.reset()
        E = torch.zeros(B, N, dtype=half, device=dev)
        I = torch.zeros(B, N, dtype=half, device=dev)
        stt = torch.zeros(B, N, dtype=half, device=dev)
        press = torch.zeros(B, dtype=torch.float32, device=dev)
        act_buf = torch.zeros(B, dtype=torch.long, device=dev) - 1
        run_len = torch.zeros(B, dtype=torch.long, device=dev)
        tot_turn = np.zeros(B); tot_step = np.zeros(B)
        short_runs = total_runs = 0
        foods = np.zeros(B)
        for t in range(cfg.MAX_STEPS):
            al = env.alive
            if not al.any():
                break
            obs = env.obs().to(half)
            if mask_diag:
                obs[:, 9] = 0; obs[:, 11] = 0; obs[:, 13] = 0; obs[:, 15] = 0
            act, E, I, stt = t7f.deliberate_batch(pop, obs, E, I, stt, press, cfg)
            press = t7f.update_fatigue(press, act, decay=float(cfg.FATIGUE_TURN_DECAY))
            ended = al & (act != 0) & (act_buf == 0)
            rl = run_len[ended].cpu().numpy()
            short_runs += int((rl <= 2).sum()); total_runs += int(len(rl))
            cont = (act == act_buf) & (act == 0) & al
            run_len = torch.where(cont, run_len + 1,
                                  torch.where(al & (act == 0),
                                              torch.ones_like(run_len),
                                              torch.zeros_like(run_len)))
            act_buf = torch.where(al, act, act_buf)
            tot_turn += (al & (act != 0)).float().cpu().numpy()
            tot_step += al.float().cpu().numpy()
            env.step(act)
            foods += (al & env.ate).float().cpu().numpy()
        dens = tot_turn.sum() / max(tot_step.sum(), 1)
        wall = (env.died == 1).float().mean().item() * 100
        self_ = (env.died == 2).float().mean().item() * 100
        starve = (env.died == 3).float().mean().item() * 100
        print(f"{'是' if mask_diag else '否':>6} | {foods.mean():>7.2f} {dens:>6.3f} "
              f"{short_runs / max(total_runs,1)*100:>5.1f}% | "
              f"{wall:>5.1f} {self_:>5.1f} {starve:>5.1f}")


if __name__ == '__main__':
    m = MODEL if os.path.exists(MODEL) else 'test7e_econ_best_model.pth'
    bearing_map(m)
    ablation(m)
