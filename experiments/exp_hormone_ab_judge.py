# ==========================================
# exp_hormone_ab_judge.py —— 期相激素 mini A/B 判定（预注册门 G1 + 消融载重）
#
# 读取三臂 history（test14_{tag}_{arm}_history.json）+ 最优模型，输出：
#   1. 各臂 last5（elite_food 最后 5 代均值）与逐代曲线摘要
#   2. 门 G1：max(last5[h1], last5[h2]) − last5[c0] ≥ +1.5 → PASS
#   3. H1 基线复现检查：h1 gen1 elite_food − c0 gen1 elite_food ≤ 1.0
#   4. 激素遥测：horm_e/i 场强与 horm_food_corr 逐代演化（死通道检测）
#   5. 消融载重（过门臂才做）：过门臂 latest_gen_best 在 CRN 同库下
#      开激素 vs 关激素（--no-hormone 同数学）评估，载重=关激素掉分 ≥1
# 判定输出：G1-PASS(载重成立→转长跑) / G1-DEAD(建议 σ 加倍重试一次)
#           / G1-NULL(负结论归档，转 C 通道观测增维)
# ==========================================

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test14 as t14   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B_EVAL = 128   # 消融评估副本行数
E_EVAL = 12    # CRN 局数


def load_hist(tag, arm):
    p = os.path.join(ROOT, f'test14_{tag}_{arm}_history.json')
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def last5(hist, key='elite_food'):
    v = hist.get(key, [])[-5:]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else float('nan')


def ablation_eval(ckpt_path, enable_hormone, device):
    """CRN 同库下评估单脑 checkpoint（test14 格式），返回 [E] 逐局食物。"""
    data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    saved = data.get('config', {})
    cfg = t14.Config()
    for k, v in saved.items():
        if k.startswith('__'):
            continue
        if isinstance(v, (int, float, str, bool, list, tuple)) or v is None:
            setattr(cfg, k, v)
    cfg.HORMONE_ENABLE = enable_hormone
    cfg.DEVICE = 'auto'
    st = data['brain']
    pop = t14.GeneStack(cfg, B=B_EVAL, device=device)
    pop.random_init()
    for i in range(B_EVAL):
        pop.set_individual_from_state(i, st)
    if device.type == 'cpu':
        cfg.USE_FP16 = False
    pop.fp16()
    pop.refresh_eff()
    banks = t14.make_banks(cfg, 777, 0, E_EVAL, device)
    m = t14._eval_pop_banks(pop, cfg, banks)
    return m[:, 0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', type=str, default='ab1')
    ap.add_argument('--skip-ablation', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hists = {arm: load_hist(args.tag, arm) for arm in ('c0', 'h1', 'h2')}
    if any(h is None for h in hists.values()):
        missing = [a for a, h in hists.items() if h is None]
        sys.exit(f'[错误] 缺 history: {missing}')

    print('===== 逐代 elite_food =====')
    n = max(len(h['gen']) for h in hists.values())
    print('gen    ' + ''.join(f'{a:>8}' for a in ('c0', 'h1', 'h2')))
    for i in range(n):
        row = f'{hists["c0"]["gen"][i]:<6}'
        for arm in ('c0', 'h1', 'h2'):
            v = hists[arm]['elite_food'][i] if i < len(hists[arm]['elite_food']) else None
            row += f'{v:>8.2f}' if v is not None else ' ' * 8
        print(row)

    print('\n===== 激素遥测（h1 / h2）=====')
    for arm in ('h1', 'h2'):
        h = hists[arm]
        he, hi, hc = h['horm_e'], h['horm_i'], h['horm_food_corr']
        print(f'{arm}: horm_e gen1={he[0]:.5f} gen中={np.mean(he):.5f} gen末={he[-1]:.5f} | '
              f'horm_i gen末={hi[-1]:.5f} | horm_food_corr 末={hc[-1]:+.2f} 均值={np.mean(hc):+.2f}')

    l5 = {arm: last5(h) for arm, h in hists.items()}
    print(f'\n===== last5 elite_food =====')
    for arm in ('c0', 'h1', 'h2'):
        print(f'  {arm}: {l5[arm]:.3f}')

    best_gain = max(l5['h1'], l5['h2']) - l5['c0']
    best_arm = 'h1' if l5['h1'] >= l5['h2'] else 'h2'
    print(f'\n最大增益 {best_arm}: {best_gain:+.3f}（门 G1 阈值 +1.5）')

    # H1 基线复现检查
    h1_base = hists['h1']['elite_food'][0] - hists['c0']['elite_food'][0]
    print(f'H1 基线复现: gen1 差 {h1_base:+.3f}（判据 |·|≤1.0）')

    verdict_lines = []
    gate_pass = best_gain >= 1.5
    base_ok = abs(h1_base) <= 1.0

    # 激素场演化检测（死通道）
    he0, he1 = hists[best_arm]['horm_e'][0], hists[best_arm]['horm_e'][-1]
    field_evolved = abs(he1 - he0) > 0.1 * max(he0, 1e-9)
    field_alive = max(he1, hists[best_arm]['horm_i'][-1]) > 1e-6

    load_ok = None
    if gate_pass and not args.skip_ablation:
        ckpt = os.path.join(ROOT, f'test14_{args.tag}_{best_arm}_latest_gen_best.pth')
        if os.path.exists(ckpt):
            on = ablation_eval(ckpt, True, device)
            off = ablation_eval(ckpt, False, device)
            drop = float(np.mean(on) - np.mean(off))
            load_ok = drop >= 1.0
            print(f'\n消融载重（{best_arm} latest_gen_best, {E_EVAL}局×{B_EVAL}副本）: '
                  f'开激素={np.mean(on):.2f} 关激素={np.mean(off):.2f} 掉分={drop:.2f} '
                  f'（判据 ≥1.0）→ {"载重成立" if load_ok else "非载重（激素非必需）"}')
        else:
            print(f'\n[警告] 找不到 {ckpt}，跳过消融载重')

    print('\n===== 门 G1 判定 =====')
    if not base_ok:
        verdict_lines.append('H1 基线复现失败（初始化破坏性过强）→ 实验无效，降 σ 重跑')
    if gate_pass:
        if load_ok:
            verdict_lines.append(f'G1-PASS：{best_arm} 过门且激素载重成立 → 转长跑（150+ 代）')
        elif load_ok is None:
            verdict_lines.append(f'G1-PASS（未做消融）→ 补消融载重验证后决定转长跑')
        else:
            verdict_lines.append('G1 过门但激素非载重（收益来自协同进化噪声）→ 谨慎长跑并复检')
    elif not field_alive or not field_evolved:
        verdict_lines.append('G1-DEAD：激素场未演化/恒零 → 死通道复发，σ 加倍重试一次（预定）')
    else:
        verdict_lines.append('G1-NULL：激素场在动但无可选择收益 → 负结论归档，转 C 通道观测增维')

    for ln in verdict_lines:
        print('  ' + ln)

    out = {'tag': args.tag, 'last5': l5, 'best_gain': best_gain,
           'best_arm': best_arm, 'h1_base_check': h1_base,
           'load_ok': load_ok, 'verdict': verdict_lines}
    outp = os.path.join(ROOT, 'results', f'hormone_ab_{args.tag}.json')
    with open(outp, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f'已写出 {outp}')


if __name__ == '__main__':
    main()
