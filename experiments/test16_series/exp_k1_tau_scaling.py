# ==========================================
# exp_k1_tau_scaling.py —— “前向坍缩”检验 v2：K=1 + τ 下扫（快速衰减方向）
#
# 机制：τ·E 是自保持项。K=5 时每个决策对旧状态松弛 5 次（局部增益
# g≈0.3~0.6，g^5≈0.002~0.08，旧状态基本洗净 → 决策≈当前输入的不动点）；
# K=1 只松弛 1 次（残留 g≈0.5）→ 决策被旧状态污染（表现为饿死暴增）。
# 因此 K=1 的等价补偿方向是【减小 τ 让状态快速衰减】（朝纯前向），
# 而非上轮反方向的加大 τ（×5/几何和均更差，已证）。
#
# 协议：16 张地图（ep 0..15）批量 B=16；K=1，τ_e × factor 下扫
# 1.00 → 0.01。参照：K=5 τ 原版基线 food 总 971 / mean 60.69 / wins 1。
# ==========================================
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import test16c_cheat7b as sim

MODEL = 'artifacts/test16c_cheat7b/16c_cheat7b_win_model.pth'
N_EP = 16
STEP_CAP = 3000
BASELINE = 'K=5 τ原版 基线: food 总 971 mean 60.69 wins 1/16 死亡(己11/饿4)'


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    data = torch.load(MODEL, map_location='cpu', weights_only=False)
    cfg = sim.Config()
    for k, v in data.get('config', {}).items():
        if not k.startswith('__'):
            try:
                setattr(cfg, k, v)
            except Exception:
                pass
    cfg.DEVICE = str(dev)
    st = data['brain']
    print(f"模型: {MODEL} food={data.get('food')}/{cfg.TARGET_FOOD} "
          f"N={st['N']} K={st['K']} FRAME_RATE={cfg.FRAME_RATE} "
          f"INPUT_DECAY={cfg.INPUT_DECAY} "
          f"tau_e[{st['tau_e_init'].min():.2f},{st['tau_e_init'].max():.2f}]",
          flush=True)

    B = N_EP
    pop = sim.GeneStack(cfg, B=B, device=dev)
    pop.random_init()
    for i in range(B):
        pop.set_individual_from_state(i, st)
    pop.refresh_eff()
    if getattr(cfg, 'USE_FP16', True):
        pop.fp16()
        pop.refresh_eff()
    base_tau = pop.tau_e.clone()

    banks = [sim.make_bank(cfg, cfg.MAP_GEN, 1, ep, dev) for ep in range(B)]
    bank = {'stream': torch.stack([b['stream'] for b in banks], 0),
            'dir0': torch.tensor([b['dir0'] for b in banks], device=dev)}
    env = sim.BatchedSnakeEnv(cfg, B=B, device=dev)

    def apply_tau(factor):
        tau = (base_tau * factor).clamp(float(cfg.TAU_E_MIN),
                                        float(cfg.TAU_E_MAX))
        pop.tau_e = tau.to(pop.dtype)

    def run():
        env.reset(bank=bank)
        E = torch.zeros(B, pop.N, dtype=pop.dtype, device=dev)
        I = torch.zeros_like(E)
        stt = torch.zeros_like(E)
        cts = torch.zeros(B, pop.A, dtype=pop.dtype, device=dev)
        for _ in range(STEP_CAP):
            if not bool(env.alive.any()):
                break
            obs = env.obs().to(pop.dtype)
            lg, E, I, stt = sim.forward_batch(pop, obs, E, I, stt, cts, cfg)
            act = torch.argmax(lg, dim=1)
            cts = sim.update_fatigue(cts, act)
            env.step(act)
        food = (env.body_len - 2).float().cpu()
        steps = env.steps.cpu()
        wins = int(env.won.sum())
        died = env.died.cpu()
        return food, steps, wins, died

    sweep = (1.00, 0.80, 0.65, 0.50, 0.40, 0.30, 0.20, 0.10, 0.03)
    print(f"\n参照 {BASELINE}", flush=True)
    print(f"协议: K=1（每环境步 1 思考帧，无输入衰减） {B} 张地图 × 至多 "
          f"{STEP_CAP} 步 | τ_e × factor 下扫", flush=True)
    print(f"{'factor':>7} | {'food总':>6} {'mean':>6} {'max':>5} {'wins':>4} "
          f"| 死亡(墙/己/饿)", flush=True)
    for f in sweep:
        apply_tau(f)
        food, steps, wins, died = run()
        print(f"{f:>7.2f} | {food.sum():6.0f} {food.mean():6.2f} "
              f"{food.max():5.0f} {wins:4d} "
              f"| {[(died == i).sum().item() for i in (1, 2, 3)]}", flush=True)

    print("\n判读：若存在 interior 最优（τ 因子 0~1 之间显著优于两端）→ K=1 "
          "可用但需重调 τ；若单调（越小越好，极端处逼近基线）→ 7b 本来就"
          "只剩前向（循环项在 K=1 下毫无价值）；若全程远低于基线 → 5 帧迭代"
          "本身必要，K=1 不可救。", flush=True)


if __name__ == '__main__':
    main()
