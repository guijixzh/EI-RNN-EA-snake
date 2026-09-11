# ==========================================
# exp_7b_memory_probe_random.py —— 7b 记忆探针·全随机变体
#
# 与 exp_7b_memory_probe.py 的差别：中间 14 个输入槽位不再是“14 个状态的
# 无重复排列”，而是**独立均匀随机抽取（有放回，15 选 14 次）**——状态可重复
# 出现、也可缺席，路径多样性远大于排列版。
# 指标同前：动作一致率 / logits L2(末,首) / 参照系（其它槽位相对首步）/ 跨序列
# 末步两两 L2（路径特异性）。
# ==========================================
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import test16c_cheat7b as sim

MODEL = 'artifacts/test16c_cheat7b/16c_cheat7b_win_model.pth'
N_STATE = 15
N_PERM = 64
S_IDX = 0
SEQ_LEN = 16


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
    pop = sim.GeneStack(cfg, B=1, device=dev)
    pop.random_init()
    pop.set_individual_from_state(0, st)
    pop.refresh_eff()
    if getattr(cfg, 'USE_FP16', True):
        pop.fp16()
        pop.refresh_eff()

    states = []
    for ep in range(N_STATE):
        bank = sim.make_bank(cfg, cfg.MAP_GEN, 1, ep, dev)
        env = sim.BatchedSnakeEnv(cfg, 1, dev)
        env.reset(bank=bank)
        states.append(env.obs()[0].float().cpu())

    rng = np.random.default_rng(0)
    seqs = []
    for _ in range(N_PERM):
        mid = rng.integers(0, N_STATE, size=SEQ_LEN - 2)   # 有放回全随机
        seqs.append([S_IDX] + list(mid) + [S_IDX])

    K = int(cfg.FRAME_RATE)
    ends, begins = [], []
    agree = 0
    d_end = []
    d_ref = []
    for q in seqs:
        E = torch.zeros(1, pop.N, dtype=pop.dtype, device=dev)
        I = torch.zeros_like(E)
        stt = torch.zeros_like(E)
        cts = torch.zeros(1, pop.A, dtype=pop.dtype, device=dev)
        outs = []
        for si in q:
            obs = states[si].to(dev, pop.dtype)[None, :]
            lg_sum = None
            for k in range(K):
                o = obs * (cfg.INPUT_DECAY ** k)
                lg, E, I, stt = sim.forward_batch(pop, o, E, I, stt, cts, cfg)
                lg_sum = lg if lg_sum is None else lg_sum + lg
            act = torch.argmax(lg_sum, dim=1)
            cts = sim.update_fatigue(cts, act)
            outs.append(lg_sum[0].float().cpu())
        agree += int(outs[0].argmax() == outs[-1].argmax())
        d_end.append(float(torch.dist(outs[-1], outs[0])))
        d_ref.append(float(np.mean([float(torch.dist(outs[i], outs[0]))
                                    for i in range(1, SEQ_LEN - 1)])))
        ends.append(outs[-1].numpy())
        begins.append(outs[0].numpy())

    E_m = np.stack(ends)
    B_m = np.stack(begins)
    de = [float(np.linalg.norm(E_m[i] - E_m[j]))
          for i in range(N_PERM) for j in range(i + 1, N_PERM)]
    d_end = np.array(d_end)
    d_ref = np.array(d_ref)
    print(f"=== 全随机变体（中间 14 槽位有放回独立抽取，64 序列 × 16 游戏步） ===",
          flush=True)
    print(f"动作一致率: {agree}/{N_PERM}", flush=True)
    print(f"logits L2 末-首: mean={d_end.mean():.3f} max={d_end.max():.3f}", flush=True)
    print(f"参照系 其它槽位相对首步 L2: mean={d_ref.mean():.3f}", flush=True)
    print(f"比值 d(末,首)/参照: mean={(d_end / np.clip(d_ref, 1e-9, None)).mean():.1%} "
          f"max={(d_end / np.clip(d_ref, 1e-9, None)).max():.1%}", flush=True)
    print(f"跨序列 末步输出两两 L2: mean={np.mean(de):.3f} max={np.max(de):.3f}"
          f"  （对比 排列版 0.400）", flush=True)
    print(f"末步 logits 均值: {np.round(E_m.mean(0), 2)}  首步均值: {np.round(B_m.mean(0), 2)}",
          flush=True)


if __name__ == '__main__':
    main()
