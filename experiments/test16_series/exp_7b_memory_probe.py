# ==========================================
# exp_7b_memory_probe.py —— 7b 网络记忆探针
#
# 问题：7b 网络是否“非记忆”？同一输入，历史不同，输出是否不同？
#
# 方法（用户设计）：
#   15 个真实输入状态 = 15 张不同地图 reset 后的 32 维观测；
#   64 条长度 16 的输入序列：首尾均为状态 S，中间 14 个状态自由随机排列
#   （= 16 个游戏步；每步 5 思考帧 deliberation + INPUT_DECAY=0.9 + 疲劳计数）。
#   对比每条序列“第 1 步输出 vs 第 16 步输出”：
#     - 动作（argmax）一致率
#     - logits L2 距离 d(end,begin)，参照系 = 同序列内其它(不同)输入与
#       begin 输出的平均 L2 距离（输入本身能引起的输出变化幅度）
#     - 循环状态 ||E_end − E_begin||
#   双口径：cts（疲劳动作计数）跟随 / 冻结。冻结 = 纯循环回路的记忆，
#   跟随 = 游戏真实语义（cts 是网络输入的一部分，属外置计数器）。
# ==========================================
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import test16c_cheat7b as sim

MODEL = 'artifacts/test7b/test7b_best_model.pth'      # 7b 实验基模（稠密基因组 → 7b_sparse 转换器）
N_STATE = 15
N_PERM = 64
S_IDX = 0                      # 首尾固定状态（取第 0 个输入）
SEQ_LEN = 160                      # 160 个游戏步（首尾 S，中间 158 步自由采样）


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
    loaded = sim.load_best_state_7b_sparse(MODEL, cfg, verbose=True)
    assert loaded is not None, '7b 稠密基因组转换失败'
    st, food, steps = loaded
    tgt = data.get('config', {}).get('TARGET_FOOD')
    print(f"通关证据: food={food}/{tgt} steps={steps}", flush=True)
    print(f"模型: {MODEL}（7b 实验基模，稠密→稀疏 K={st['K']}） "
          f"tau_e[{st['tau_e_init'].min():.2f},{st['tau_e_init'].max():.2f}] "
          f"w_ei={st['w_ei'].mean():.2f} w_ie={st['w_ie'].mean():.2f}", flush=True)

    pop = sim.GeneStack(cfg, B=1, device=dev)
    pop.random_init()
    pop.set_individual_from_state(0, st)
    pop.refresh_eff()
    if getattr(cfg, 'USE_FP16', True):
        pop.fp16()
        pop.refresh_eff()

    # ---- 15 个真实输入状态（15 张不同地图的初始观测） ----
    states = []
    for ep in range(N_STATE):
        bank = sim.make_bank(cfg, cfg.MAP_GEN, 1, ep, dev)
        env = sim.BatchedSnakeEnv(cfg, 1, dev)
        env.reset(bank=bank)
        states.append(env.obs()[0].float().cpu())
    D = np.array([[float(torch.dist(a, b)) for b in states] for a in states])
    print(f"15 个输入状态两两 L2：min={D[D > 0].min():.2f} "
          f"mean={D[D > 0].mean():.2f}（确认互异）", flush=True)

    # ---- 64 条序列：[S, 14 个自由排列, S] ----
    rng = np.random.default_rng(0)
    others = [i for i in range(N_STATE) if i != S_IDX]
    seqs = [[S_IDX] + list(rng.choice(others, SEQ_LEN - 2)) + [S_IDX]
            for _ in range(N_PERM)]

    K = int(cfg.FRAME_RATE)

    HOLD_EXTRA = 4                # 回归后继续按 S 保持的步数（观察热机偏移消散）

    def run_seq(seq, freeze_cts):
        E = torch.zeros(1, pop.N, dtype=pop.dtype, device=dev)
        I = torch.zeros_like(E)
        stt = torch.zeros_like(E)
        cts = torch.zeros(1, pop.A, dtype=pop.dtype, device=dev)
        outs, extra = [], []
        for si in seq + [seq[-1]] * HOLD_EXTRA:
            obs = states[si].to(dev, pop.dtype)[None, :]
            lg_sum = None
            for k in range(K):
                o = obs * (cfg.INPUT_DECAY ** k)
                lg, E, I, stt = sim.forward_batch(pop, o, E, I, stt, cts, cfg)
                lg_sum = lg if lg_sum is None else lg_sum + lg
            act = torch.argmax(lg_sum, dim=1)
            if not freeze_cts:
                cts = sim.update_fatigue(cts, act)
            rec = (lg_sum[0].float().cpu(), int(act[0]), E.clone())
            if len(outs) < SEQ_LEN:
                outs.append(rec)
            else:
                extra.append(rec)
        return outs, extra

    for freeze, tag in ((True, 'cts 冻结（纯循环回路记忆）'),
                        (False, 'cts 跟随（游戏真实语义，外置疲劳计数）')):
        recs, extras = zip(*[run_seq(q, freeze) for q in seqs])
        agree = sum(r[0][1] == r[-1][1] for r in recs)
        d_end = np.array([float(torch.dist(r[-1][0], r[0][0])) for r in recs])
        d_ref = np.array([np.mean([float(torch.dist(r[i][0], r[0][0]))
                                   for i in range(1, SEQ_LEN - 1)])
                          for r in recs])
        dE = np.array([float(torch.dist(r[-1][2][0], r[0][2][0])) for r in recs])
        ratio = d_end / np.clip(d_ref, 1e-9, None)
        print(f"\n=== {tag} ===", flush=True)
        print(f"动作一致率（首步 vs 末步）: {agree}/{N_PERM}", flush=True)
        print(f"logits L2 末-首: mean={d_end.mean():.3f} max={d_end.max():.3f}", flush=True)
        print(f"参照系 不同输入相对首步的 L2: mean={d_ref.mean():.3f}", flush=True)
        print(f"比值 d(end,begin)/参照: mean={ratio.mean():.1%} "
              f"max={ratio.max():.1%}", flush=True)
        print(f"循环状态 ||E_end−E_begin||: mean={dE.mean():.2f} "
              f"(E 向量典型范数 {np.mean([torch.norm(r[0][2][0]).item() for r in recs]):.2f})",
              flush=True)
        mism = [(qi, r[0][1], r[-1][1]) for qi, r in enumerate(recs)
                if r[0][1] != r[-1][1]]
        if mism:
            print("不一致样本 (序列号, 首步动作, 末步动作):", mism[:8], flush=True)
        # 热机偏移消散：回归后保持 S 的 1..4 步，与首步输出的 L2
        for h in range(HOLD_EXTRA):
            dh = np.array([float(torch.dist(x[h][0], r[0][0]))
                           for x, r in zip(extras, recs)])
            acts = sum(x[h][1] != r[0][1] for x, r in zip(extras, recs))
            print(f"  回归后保持 S 第 {h + 1} 步: ||lg−lg_begin|| mean={dh.mean():.3f} "
                  f"max={dh.max():.3f} 动作偏离 {acts}/{N_PERM}", flush=True)

    print("\n判读：若动作一致率高且 d(end,begin) 远小于参照系（比值 << 1），"
          "说明经历 14 个中间输入后回到同一输入，网络输出与无历史时几乎相同"
          " → 在“游戏步粒度的输出”意义上是非记忆（反应式）的。", flush=True)


if __name__ == '__main__':
    main()
