# ==========================================
# exp_obs40_judge.py —— test15 Phase B 判定（预注册门 G1/G2/G3）
#
# G1-food：A1(test15 迁移+新通道 15 代) 冠军 vs ab1_c0 对照冠军，
#          ≥3 组共享 CRN 库配对复评，差 ≥+1.5
#          （test14 教训：单库方差 ±5，单库消融方向都不可信）
# G2-phase（真终点）：Phase-0 teacher-spawn 复测——A1 冠军从 len 30/40/50
#          教师身体直接出生的增食 ratio，对照 test12 冠军基线
#          （phase_diagnostic.json：L40=0.25-0.50 / L50=0.33-0.67），
#          判据：L40 或 L50 ratio ≥0.7 且自然局食物不降（≥ 基线 44.7×0.9）
# G3-health：自然局 len≥40 段死因中自撞占比下降（基线：自然局崩塌段
#          91-98% 为出生档口径；此处用自然局 postL40 段死因对比）
# ==========================================

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import test12 as t12   # noqa: E402
import test15 as t15   # noqa: E402
import exp_phase_diagnostic as epd   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
A1_CKPT = os.path.join(ROOT, 'artifacts/test15/test15_b1_a1_latest_gen_best.pth')
C0_CKPT = os.path.join(ROOT, 'artifacts/test14/test14_ab1_c0_latest_gen_best.pth')
HIST_A1 = os.path.join(ROOT, 'artifacts/test15/test15_b1_a1_history.json')
BASE = json.load(open(os.path.join(ROOT, 'results', 'phase_diagnostic.json'),
                      encoding='utf-8'))


def eval_food(module, cfg, st, B, banks, device):
    pop = module.GeneStack(cfg, B=B, device=device)
    pop.random_init()
    for i in range(B):
        pop.set_individual_from_state(i, st)
    if device.type == 'cpu':
        cfg.USE_FP16 = False
    pop.fp16()
    pop.refresh_eff()
    m = module._eval_pop_banks(pop, cfg, banks)
    return m[:, 0].numpy()


def load15(path, new_enabled=True):
    cfg, _ = epd.build_cfg_from_ckpt(t15, path)
    cfg.OBS_NEW_ENABLED = new_enabled
    data = torch.load(path, map_location='cpu', weights_only=False)
    st = data['brain']
    return cfg, st, float(data.get('food', -1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--banks', type=int, default=3)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}')

    for p in (A1_CKPT, C0_CKPT, HIST_A1):
        if not os.path.exists(p):
            sys.exit(f'[错误] 缺产物: {p}')

    # ---- G1-food：多库配对复评 ----
    print('\n===== G1-food 多库配对复评（12局×64副本×3库）=====')
    cfg15, st15, _ = load15(A1_CKPT, True)
    # ab1_c0 是 test14 零激素臂 = test12 同构，直接用 test12 原生评估
    cfgc0, _ = epd.build_cfg_from_ckpt(t12, C0_CKPT)
    datac0 = torch.load(C0_CKPT, map_location='cpu', weights_only=False)
    stc0 = datac0['brain']
    diffs = []
    rows = []
    for k in range(args.banks):
        tag = 910 + k
        b15 = t15.make_banks(cfg15, tag, 0, 12, device)
        b12 = t12.make_banks(cfgc0, tag, 0, 12, device)
        a1 = eval_food(t15, cfg15, st15, 64, b15, device)
        c0 = eval_food(t12, cfgc0, stc0, 64, b12, device)
        diffs.append(float(a1.mean() - c0.mean()))
        rows.append({'banks_tag': tag, 'a1': float(a1.mean()),
                     'c0': float(c0.mean()), 'diff': diffs[-1]})
        print(f'  库{tag}: A1={a1.mean():.2f}  C0={c0.mean():.2f}  Δ={diffs[-1]:+.2f}')
    g1_diff = float(np.mean(diffs))
    g1 = g1_diff >= 1.5
    print(f'[G1] Δ均值={g1_diff:+.3f}（判据 ≥+1.5）→ {"PASS" if g1 else "FAIL"}')

    # ---- G2-phase：Phase-0 teacher-spawn 复测（A1 冠军）----
    print('\n===== G2-phase teacher出生复测（64局/档）=====')
    cfg15n, _ = epd.build_cfg_from_ckpt(t15, A1_CKPT)
    cfg15n.OBS_NEW_ENABLED = True
    sub = {'spawned': {}, 'natural': None}
    for L in (30, 40, 50):
        pop, _ = epd.broadcast_brain(t15, cfg15n, A1_CKPT, 64, device)
        r = epd.run_sweep(t15, pop, cfg15n, device, 'teacher', L, 64,
                          seed_tag=3310 + L, step_cap=2500)
        sub['spawned'].setdefault('teacher', {})[str(L)] = epd.summarize_spawned(r)
        s = sub['spawned']['teacher'][str(L)]
        base = BASE['results']['test12']['spawned']['teacher'].get(str(L), {})
        base_med = base.get('food_median', float('nan'))
        print(f'  [teacher L={L}] food med={s["food_median"]:5.1f} '
              f'(test12基线 {base_med:.1f}) death={s["death_causes"]}')
    # 自然局（食物不降 + G3 死因）
    pop, _ = epd.broadcast_brain(t15, cfg15n, A1_CKPT, 128, device)
    rn = epd.run_sweep(t15, pop, cfg15n, device, 'natural', 2, 128,
                       seed_tag=3399, step_cap=2500)
    nat = epd.summarize_natural(rn, [30, 40, 50])
    sub['natural'] = nat
    print(f'  [natural] food med={nat["food_median"]:.1f}')
    for L in (40, 50):
        tr = nat['tiers'].get(str(L), {})
        if tr.get('n_reach'):
            print(f'    natural L>={L}: postL med={tr["postL_food_median"]:.1f} '
                  f'death={ {k: round(v, 2) for k, v in tr["death_causes_postL"].items()} }')

    # 判定
    l40 = sub['spawned']['teacher']['40']['food_median']
    l50 = sub['spawned']['teacher']['50']['food_median']
    nat12 = BASE['results']['test12']['natural']['tiers']
    base_l40 = nat12['40']['postL_food_median']
    base_l50 = nat12['50']['postL_food_median']
    r40 = l40 / max(base_l40, 1e-6)
    r50 = l50 / max(base_l50, 1e-6)
    food_ok = nat['food_median'] >= 44.7 * 0.9
    g2 = (r40 >= 0.7 or r50 >= 0.7) and food_ok
    print(f'\n[G2] ratio L40={r40:.2f} L50={r50:.2f}（判据 ≥0.7 至少一档），'
          f'自然局中位 {nat["food_median"]:.1f}（判据 ≥40.2）→ '
          f'{"PASS" if g2 else "FAIL"}')
    late40 = nat['tiers']['40'].get('death_causes_postL', {})
    g3 = late40.get('self', 1.0) < 0.85
    print(f'[G3] 自然局 len≥40 段自撞死因占比={late40.get("self", 1.0):.2f}'
          f'（判据 <0.85）→ {"PASS" if g3 else "FAIL"}')

    verdict = ('G-PASS → test15 长跑 150+ 代' if (g1 and g2) else
               'G1 单过（食物升但相位未修）→ 谨慎长跑并加强相位分析' if g1 else
               'G-NULL → 信息侧该边际证伪归档，转容量解释')
    print(f'\n>>> 总判定: {verdict}')

    out = {'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
           'g1': {'diff_mean': g1_diff, 'rows': rows, 'pass': bool(g1)},
           'g2': {'ratio_l40': r40, 'ratio_l50': r50, 'nat_food': nat['food_median'],
                  'pass': bool(g2)},
           'g3': {'late_self_share': late40.get('self'), 'pass': bool(g3)},
           'natural': nat, 'spawned_teacher': sub['spawned']['teacher'],
           'verdict': verdict}
    with open(os.path.join(ROOT, 'results', 'obs40_ab_b1.json'), 'w',
              encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f'已写出 {os.path.join(ROOT, "results", "obs40_ab_b1.json")}')


if __name__ == '__main__':
    main()
