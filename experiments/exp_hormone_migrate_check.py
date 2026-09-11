# ==========================================
# exp_hormone_migrate_check.py —— test12→test14 血统迁移等价性验证（预注册）
#
# 目标：证明 test14 的激素支路满足两条硬性质——
#   P1 零激素严格等价：同一冠军底盘，test14(HORMONE_ENABLE=False) 的评估
#      与 test12 原生评估逐局一致（CRN 同库）⇒ C0 臂 = test12 同构。
#   P2 激素开通为小扰动（v1.1 修订）：迁移后激素基因 σ=0.02 随机初始化下，
#      开通激素的评估相对 P1 的 Δfood 均值 |·| ≤ 2.0（=1×个体评估噪声 SEM，
#      7g 诊断实测 2.18——EI 动力学是混沌吸引子，任意非零扰动都会重排逐局
#      结局，轨迹级一致不可能；判据应为"扰动量级 ≈ 评估噪声"而非"轨迹不变"），
#      且激素场遥测非零（场确实在动，通道非死）。
#   P3 断点版本守卫：test14 拒绝续训 test12 旧断点（缺 BRAIN_VERSION 键）。
# 判定：P1、P2、P3 全过 → 迁移工具可信，mini A/B 三臂放行。
# ==========================================

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test12 as t12   # noqa: E402
import test14 as t14   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, 'test12_econ_latest_gen_best.pth')
B = 64          # 评估副本行数（同一底盘）
E = 8           # CRN 局数
GEN_TAG = 4242  # CRN 库种子标签（远离训练序列）


def build_cfg(module):
    data = torch.load(CKPT, map_location='cpu', weights_only=False)
    saved = data.get('config', {})
    cfg = module.Config()
    for k, v in saved.items():
        if k.startswith('__'):
            continue
        if isinstance(v, (int, float, str, bool, list, tuple)) or v is None:
            setattr(cfg, k, v)
    cfg.DEVICE = 'auto'
    return cfg, float(data.get('food', -1.0))


def broadcast(module, cfg, st, B, device):
    pop = module.GeneStack(cfg, B=B, device=device)
    pop.random_init()
    for i in range(B):
        pop.set_individual_from_state(i, st)
    if device.type == 'cpu':
        cfg.USE_FP16 = False
    pop.fp16()
    pop.refresh_eff()
    return pop


def eval_food(pop, cfg, device):
    banks = t14.make_banks(cfg, GEN_TAG, 0, E, device) \
        if module_of(cfg) is t14 else t12.make_banks(cfg, GEN_TAG, 0, E, device)
    m = (t14 if module_of(cfg) is t14 else t12)._eval_pop_banks(pop, cfg, banks)
    return m


_MOD = {}


def module_of(cfg):
    return _MOD[id(cfg)]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}')
    ok = True

    # --- P1: 零激素等价 ---
    cfg12, food12 = build_cfg(t12)
    cfg14, _ = build_cfg(t14)
    cfg14_bak = cfg14.HORMONE_ENABLE
    cfg14.HORMONE_ENABLE = False
    _MOD[id(cfg12)] = t12
    _MOD[id(cfg14)] = t14
    data = torch.load(CKPT, map_location='cpu', weights_only=False)
    st = data['brain']

    pop12 = broadcast(t12, cfg12, st, B, device)
    pop14 = broadcast(t14, cfg14, st, B, device)

    m12 = eval_food(pop12, cfg12, device)
    m14 = eval_food(pop14, cfg14, device)
    d12 = (m12[:, 0] - m14[:, 0]).abs()
    p1_max = float(d12.max())
    print(f'\n[P1] test12 food={float(m12[:, 0].mean()):.4f}  '
          f'test14(no-horm) food={float(m14[:, 0].mean()):.4f}  '
          f'max|Δ局|={p1_max:.6f}')
    if p1_max > 1e-3:
        print('[P1] FAIL: 零激素评估与 test12 不一致')
        ok = False
    else:
        print('[P1] PASS')

    # --- P2: 开通激素 = 小扰动 + 场非零 ---
    cfg14.HORMONE_ENABLE = True
    m14h = eval_food(pop14, cfg14, device)
    delta = m14h[:, 0] - m14[:, 0]
    med_delta = float(np.median(delta.numpy()))
    he = float(m14h[:, 19].mean())
    hi = float(m14h[:, 20].mean())
    print(f'\n[P2] 开通激素 food={float(m14h[:, 0].mean()):.4f}  '
          f'Δfood median={med_delta:+.4f} (P25={float(np.quantile(delta.numpy(), .25)):+.4f} '
          f'P75={float(np.quantile(delta.numpy(), .75)):+.4f})  '
          f'horm_e={he:.5f} horm_i={hi:.5f}')
    p2_small = abs(float(delta.numpy().mean())) <= 2.0   # ≤1×评估噪声 SEM(2.18, 7g诊断)
    p2_alive = (he > 1e-6) or (hi > 1e-6)
    print(f'     Δfood mean={float(delta.numpy().mean()):+.4f}（判据 |均值|≤2.0=1×SEM）')
    if not p2_small:
        print('[P2] FAIL: 初始扰动过大（σ=0.02 应近基线）')
        ok = False
    elif not p2_alive:
        print('[P2] FAIL: 激素场恒零（死通道）')
        ok = False
    else:
        print('[P2] PASS（小扰动 + 场非零）')
    cfg14.HORMONE_ENABLE = cfg14_bak

    # --- P3: 断点版本守卫（真实续训场景：test14 cfg + test12 旧断点）---
    cfg14c = build_cfg(t14)[0]
    cfg14c.CHECKPOINT_PATH = '/tmp/nonexist_t14.pth'
    r = t14.load_checkpoint7(CKPT, cfg14c)   # test12 旧断点无 BRAIN_VERSION → 应拒绝
    p3 = r is None
    print(f'\n[P3] test14.load_checkpoint7(test12断点) → {"拒绝" if p3 else "接受"}')
    if not p3:
        print('[P3] FAIL: 版本守卫未生效')
        ok = False
    else:
        print('[P3] PASS')

    print('\n=====\n迁移等价性总判定: ' + ('PASS ✓' if ok else 'FAIL ✗'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
