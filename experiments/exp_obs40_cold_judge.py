# ==========================================
# exp_obs40_cold_judge.py —— test15 冷启动塑形判定（预注册三读数）
#
# 主读数  ：cold1 gen30 elite_food vs test12 历史 gen30（同 CRN 食物流）
#           ≥+1.5 正向塑形 / ±1.5 中性 / ≤-1.5 负向
# 载重读数：cold1 gen30 冠军消融（新通道置零 vs 开通，≥3 库配对）
#           ——温启动时为负，冷启动若转正 = 信息已塑形进回路
# 形态读数：best_straight / best_conn / best_epref 曲线 vs test12 历史
# ==========================================

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test15 as t15   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLD_HIST = os.path.join(ROOT, 'test15_cold1_history.json')
REF_HIST = os.path.join(ROOT, 'test12_econ_history.json')
CKPT = os.path.join(ROOT, 'test15_cold1_latest_gen_best.pth')


def load_hist(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def ablation(ckpt, device, banks_tags=(920, 921, 922)):
    data = torch.load(ckpt, map_location='cpu', weights_only=False)
    cfg, _ = build15(data)
    st = data['brain']
    out = {}
    for mode, enabled in (('on', True), ('off', False)):
        cfg_m = cfg.__class__()
        for k, v in data.get('config', {}).items():
            if k.startswith('__'):
                continue
            if isinstance(v, (int, float, str, bool, list, tuple)) or v is None:
                setattr(cfg_m, k, v)
        cfg_m.OBS_DIM = 40
        cfg_m.OBS_ENC_VERSION = '40tailflood1'
        cfg_m.BRAIN_VERSION = 'base1'
        cfg_m.OBS_NEW_ENABLED = enabled
        cfg_m.DEVICE = 'auto'
        pop = t15.GeneStack(cfg_m, B=64, device=device)
        pop.random_init()
        for i in range(64):
            pop.set_individual_from_state(i, st)
        pop.fp16()
        pop.refresh_eff()
        foods = []
        for tag in banks_tags:
            banks = t15.make_banks(cfg_m, tag, 0, 12, device)
            m = t15._eval_pop_banks(pop, cfg_m, banks)
            foods.append(float(m[:, 0].mean()))
        out[mode] = foods
    return out


def build15(data):
    cfg = t15.Config()
    for k, v in data.get('config', {}).items():
        if k.startswith('__'):
            continue
        if isinstance(v, (int, float, str, bool, list, tuple)) or v is None:
            setattr(cfg, k, v)
    cfg.OBS_DIM = 40
    cfg.OBS_ENC_VERSION = '40tailflood1'
    cfg.BRAIN_VERSION = 'base1'
    return cfg, data


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cold = load_hist(COLD_HIST)
    ref = load_hist(REF_HIST)
    n = min(len(cold['gen']), 30, len(ref['gen']))

    print('===== 逐代曲线（cold1-40维 vs test12历史-32维，同CRN食物流）=====')
    print('gen   elite(cold/ref)        best(cold/ref)        straight(c/r)  conn(c/r)')
    for i in range(n):
        print(f"{cold['gen'][i]:>3}   {cold['elite_food'][i]:6.2f}/{ref['elite_food'][i]:<6.2f}"
              f"   {cold['best_food'][i]:6.2f}/{ref['best_food'][i]:<6.2f}"
              f"   {cold['best_straight'][i]:.2f}/{ref['best_straight'][i]:.2f}"
              f"      {cold['best_conn'][i]:.2f}/{ref['best_conn'][i]:.2f}")

    # 主读数
    c_e = float(np.mean(cold['elite_food'][-5:]))
    r_e = float(np.mean(ref['elite_food'][n - 5:n]))
    c_b = float(np.mean(cold['best_food'][-5:]))
    r_b = float(np.mean(ref['best_food'][n - 5:n]))
    print(f'\n[主读数] last5 elite: cold={c_e:.2f} vs ref={r_e:.2f} '
          f'(Δ={c_e - r_e:+.2f}) | last5 best: cold={c_b:.2f} vs ref={r_b:.2f} '
          f'(Δ={c_b - r_b:+.2f})')
    d_elite = c_e - r_e
    verdict = ('信息正向塑形涌现路径' if d_elite >= 1.5 else
               '中性（30 代内信息未改变涌现轨迹）' if d_elite > -1.5 else
               '负向（信息通道拖累早期涌现）')
    print(f'[主读数判定] Δelite={d_elite:+.2f} → {verdict}')

    # 载重读数
    if os.path.exists(CKPT):
        ab = ablation(CKPT, device)
        on_m = float(np.mean(ab['on']))
        off_m = float(np.mean(ab['off']))
        drop = on_m - off_m
        print(f'\n[载重读数] gen30 冠军 3库配对消融: '
              f'开={on_m:.2f} 关={off_m:.2f} 载重={drop:+.2f} '
              f'（>0 = 新信息已塑形进回路且被依赖）')
        print(f'  逐库: on={["%.2f" % x for x in ab["on"]]} '
              f'off={["%.2f" % x for x in ab["off"]]}')
    else:
        print(f'\n[载重读数] 缺 {CKPT}，跳过')

    # 形态读数
    print('\n[形态读数] last5 均值对比（cold/ref）：')
    for key in ('best_straight', 'best_conn', 'best_epref'):
        c = float(np.mean(cold[key][-5:]))
        r = float(np.mean(ref[key][n - 5:n]))
        print(f'  {key}: {c:.3f} vs {r:.3f} (Δ={c - r:+.3f})')

    out = {'verdict': verdict, 'd_elite': d_elite, 'd_best': c_b - r_b}
    with open(os.path.join(ROOT, 'results', 'obs40_cold1.json'), 'w',
              encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('已写出 results/obs40_cold1.json')


if __name__ == '__main__':
    main()
