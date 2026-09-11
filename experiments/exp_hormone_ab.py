# ==========================================
# exp_hormone_ab.py —— 期相激素 v1 mini A/B 编排器（预注册）
#
# 三臂（全部从 test12 冠军底盘全种群迁移出发，CRN 同种子逐代配对）：
#   C0: --arm C0   零激素对照（与 test12 严格同数学）
#   H1: --arm H1   冻结底盘，只进化激素基因（激素可救性探针）
#   H2: --arm H2   全基因协同进化
#   各 15 代，POP=2048（原生口径），NAME=ab1_{c0,h1,h2}
#
# ------- 预注册判定门 G1（run 前写死）-------
# 主判据：last5 = 各臂 elite_food 最后 5 代均值
#   G1-PASS：max(last5[H1], last5[H2]) − last5[C0] ≥ +1.5
#            → 对过门臂做消融载重验证（见 exp_hormone_ab_judge.py），
#              载重成立（关激素掉分 ≥1）→ 转长跑（150+ 代）
#   G1-DEAD：H1/H2 激素场遥测 gen15 ≈ gen1（场未演化，|Δ|<10%）且 last5 无差异
#            → 死通道复发或激素无可选择效应，按预定做一次 σ 加倍重试
#   G1-NULL：其余情形 → 负结论归档（底盘冻结下激素不可救 / 协同进化无效），
#            转向观测增维（C 通道）路线
# H1 基线复现检查：H1 gen1 elite_food 与 C0 gen1 差 ≤1.0（同为迁移底盘，
#   H1 仅多 σ=0.02 激素扰动；超差 ⇒ 初始化破坏性过强，实验无效需降 σ 重跑）
# ==========================================

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
MIGRATE_FROM = os.path.join(ROOT, 'artifacts/test12/test12_econ_latest_gen_best.pth')

ARMS = {
    'c0': ['--arm', 'C0'],
    'h1': ['--arm', 'H1'],
    'h2': ['--arm', 'H2'],
}


def run_arm(arm, name, gens, pop):
    cmd = [PY, os.path.join(ROOT, 'test14.py'),
           *ARMS[arm],
           '--migrate-from', MIGRATE_FROM,
           '--gens', str(gens),
           '--pop', str(pop),
           '--name', name]
    print(f'\n>>> [{time.strftime("%H:%M:%S")}] {" ".join(cmd)}')
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    print(f'<<< [{time.strftime("%H:%M:%S")}] {name} 退出码 {r.returncode} '
          f'耗时 {(time.time()-t0)/60:.1f}min')
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gens', type=int, default=15)
    ap.add_argument('--pop', type=int, default=2048)
    ap.add_argument('--arms', type=str, default='c0,h1,h2',
                    help='逗号分隔的臂（默认全部三臂顺序跑）')
    ap.add_argument('--tag', type=str, default='ab1')
    args = ap.parse_args()

    if not os.path.exists(MIGRATE_FROM):
        sys.exit(f'[错误] 迁移底盘不存在: {MIGRATE_FROM}')

    fails = {}
    for arm in args.arms.split(','):
        arm = arm.strip()
        name = f'{args.tag}_{arm}'
        rc = run_arm(arm, name, args.gens, args.pop)
        fails[arm] = rc

    print('\n===== mini A/B 三臂完成 =====')
    for arm, rc in fails.items():
        print(f'  {arm}: 退出码 {rc}')
    print('判定请运行: python experiments/exp_hormone_ab_judge.py '
          f'--tag {args.tag}')


if __name__ == '__main__':
    main()
