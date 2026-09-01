# 教师解法器 vs best 模型：同库同批全套关键数值 + 适应度逐项分解。
# 目的：检验 test12 适应度设计（food + eff + te + 孤岛）各 Term 的量级与
# 饱和/区分度是否合理（以 40/40 通关的教师解法器为"物理可达上界"参照）。
# 用法: python experiments/diag_teacher12_fitness.py [--n 40] [--model test12_econ_best_model.pth]
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

import test12 as t12

DEATH = {0: '存活', 1: '撞墙', 2: '撞己', 3: '饿死'}


def fit_terms(m, cfg):
    """适应度逐项分解：返回 (总, food, eff项, te项, 孤岛系数)。"""
    food, steps_last, turns_last = m['food'], m['steps_last'], m['turns_last']
    if food <= 0:
        return 0.0, 0.0, 0.0, 0.0, 1.0
    cap = float(cfg.TURN_EFF_CAP)
    w = float(cfg.TURN_EFF_W)
    te = cap if turns_last <= 0 else min(steps_last / max(turns_last, 1.0), cap)
    eff_t = float(cfg.FOOD_EFF_WEIGHT) * food / max(steps_last, 1.0)
    te_t = w * te / cap
    fac = 1.0 - (1.0 - float(cfg.ISLAND_PENALTY)) * m['flag']
    return (food + eff_t + te_t) * fac, food, eff_t, te_t, fac


def summarize(name, rows, cfg):
    f = lambda k: np.array([r[k] for r in rows], dtype=float)
    food, steps, turns = f('food'), f('steps'), f('turns')
    sl, tl = f('steps_last'), f('turns_last')
    te = np.where(tl > 0, np.minimum(sl / np.maximum(tl, 1), cfg.TURN_EFF_CAP), cfg.TURN_EFF_CAP)
    fits, fs, es, ts, facs = zip(*(fit_terms(r, cfg) for r in rows))
    fits, fs, es, ts, facs = map(np.array, (fits, fs, es, ts, facs))
    minr, avgr = f('minr'), f('avg_r')
    print(f"\n===== {name}（N={len(rows)}）=====")
    print(f"终局: food 中位 {np.median(food):.0f} 均值 {food.mean():.1f} | "
          f"步数均值 {steps.mean():.0f} | 死因 {dict(zip(*np.unique([DEATH[r['died']] for r in rows], return_counts=True)))}")
    print(f"效率: 步/食 {np.where(food>0, sl/np.maximum(food,1), 0)[food>0].mean():.1f} | "
          f"转/食 {np.where(food>0, tl/np.maximum(food,1), 0)[food>0].mean():.2f} | "
          f"te=SL/TL 中位 {np.median(te[food>0]):.2f} 均值 {te[food>0].mean():.2f} | "
          f"te≥CAP({cfg.TURN_EFF_CAP:.0f}) 占比 {(te[food>0] >= cfg.TURN_EFF_CAP).mean():.0%}")
    print(f"可达: 吃食点均 reach {avgr[food>0].mean():.2f} | 局 min_reach 中位 "
          f"{np.median(minr[food>0]):.2f} | 孤岛触发率(局级) {f('flag').mean():.0%}")
    print(f"适应度分解: 总 {fits.mean():.2f} = food {fs.mean():.2f} "
          f"+ eff项 {es.mean():.3f} + te项 {ts.mean():.2f} | 孤岛系数均值 {facs.mean():.3f}")
    return dict(food=food.mean(), fit=fits.mean(), eff=es.mean(), te=ts.mean(),
                fac=facs.mean(), te_med=np.median(te[food > 0]) if (food > 0).any() else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--model', default='test12_econ_best_model.pth')
    args = ap.parse_args()

    cfg = t12.Config()
    cfg.MAX_STEPS = 100000
    dev = t12._resolve_device(cfg)
    B = args.n
    banks = t12.make_banks(cfg, 0, 1, B, dev)
    bank = {'stream': torch.stack([b['stream'] for b in banks]),
            'dir0': torch.tensor([b['dir0'] for b in banks],
                                 dtype=torch.long, device=dev)}

    teacher = t12.VectorCycleTeacher(cfg.GRID_SIZE, dev)
    rows_t = run_tracked(teacher=teacher, cfg=cfg, bank=bank, B=B, dev=dev)
    s_t = summarize('教师解法器（回+捷径, 物理上界参照）', rows_t, cfg)

    # --- best 模型 ---
    res = t12.load_best_state(args.model, cfg)
    if res is None:
        print('[跳过] best 模型不兼容，仅教师结果')
        return
    st, _, _ = res
    pop = t12.GeneStack(cfg, B=B, device=dev)
    pop.random_init()
    for i in range(B):
        pop.set_individual_from_state(i, st)
    pop.refresh_eff()
    if getattr(cfg, 'USE_FP16', True):
        pop.fp16()
        pop.refresh_eff()
    rows_m = run_tracked(model=pop, cfg=cfg, bank=bank, B=B, dev=dev)
    s_m = summarize(f'best 模型（{args.model}）', rows_m, cfg)

    # --- 适应度设计合理性 ---
    print("\n===== 适应度设计合理性分析 =====")
    d_food = s_t['food'] - s_m['food']
    d_fit = s_t['fit'] - s_m['fit']
    print(f"教师-模型 适应度差 {d_fit:.1f} 分，其中 food 差贡献 {d_food:.1f} 分 "
          f"({d_food / max(d_fit, 1e-9):.0%})、te 项差 {s_t['te'] - s_m['te']:+.2f} 分、"
          f"eff 项差 {s_t['eff'] - s_m['eff']:+.3f} 分")
    print(f"单食边际 = 1 + {cfg.FOOD_EFF_WEIGHT}·eff ≈ 1.0~1.3 分；te 项满幅 "
          f"{cfg.TURN_EFF_W} 分 = {cfg.TURN_EFF_W / 1.0:.0f} 食当量（tie-breaker 量级"
          f"{'合理' if cfg.TURN_EFF_W < 10 else '过大'}）")
    print(f"CAP={cfg.TURN_EFF_CAP:.0f} 对教师 te 中位 {s_t['te_med']:.2f} "
          f"{'未饱和（参考可区分）' if s_t['te_med'] < cfg.TURN_EFF_CAP else '饱和（te 项对参考失效）'}")
    print(f"孤岛项: 教师触发率见上 → 系数 ×{s_t['fac']:.2f}；"
          f"{'对参考损伤可控（仍远高于模型）' if s_t['fit'] > s_m['fit'] else '警告：惩罚把参考压到模型之下'}")


def run_tracked(teacher=None, model=None, cfg=None, bank=None, B=0, dev=None):
    """带 turn/step 跟踪的策略运行（teacher 与 model 二选一）。"""
    env = t12.BatchedSnakeEnv(cfg, B, dev)
    env.reset(bank=bank)
    ar = torch.arange(B, device=dev)
    E = torch.zeros(B, cfg.NUM_COLUMNS, dtype=torch.float16 if getattr(cfg, 'USE_FP16', True) else torch.float32, device=dev)
    I = torch.zeros_like(E)
    stt = torch.zeros_like(E)
    press = torch.zeros(B, dtype=torch.float32, device=dev)
    turn_cnt = torch.zeros(B, dtype=torch.float32, device=dev)
    rows = [dict(food=0, steps_last=0, turns_last=0, minr=1e9, r_sum=0.0, r_n=0)
            for _ in range(B)]
    for t in range(cfg.MAX_STEPS):
        al = env.alive
        obs = env.obs().to(E.dtype)
        necks = env.body[ar, 1]   # 每步重算（教师安全过滤依赖当前颈部）
        if teacher is not None:
            act = teacher.act(env.head, env.food, env.body, env.body_len,
                              env.dir_idx, necks)
        else:
            act, E, I, stt = t12.deliberate_batch(model, obs, E, I, stt, press, cfg)
        turn_cnt += (al & (act != 0)).float()
        env.step(act)
        ate = al & env.ate
        if ate.any():
            rr = t12.reach_ratio(env)
            for i in range(B):
                if bool(ate[i]):
                    r = rows[i]
                    r['food'] += 1
                    r['steps_last'] = t + 1
                    r['turns_last'] = int(turn_cnt[i])
                    v = float(rr[i])
                    r['r_sum'] += v
                    r['r_n'] += 1
                    r['minr'] = min(r['minr'], v)
        if env.all_done():
            break
    for i, r in enumerate(rows):
        r['steps'] = int(env.steps[i])
        r['turns'] = int(turn_cnt[i])
        r['len'] = int(env.body_len[i])
        r['died'] = int(env.died[i])
        r['minr'] = min(r['minr'], 1.0)
        r['avg_r'] = r['r_sum'] / max(r['r_n'], 1)
        r['flag'] = 1 if (r['minr'] < cfg.ISLAND_THRESHOLD and r['r_n'] > 0) else 0
    return rows


if __name__ == '__main__':
    main()
