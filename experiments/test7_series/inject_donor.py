# ==========================================
# experiments/inject_donor.py —— 供体脑注入全种群 checkpoint（不训练）
#
# 背景：诊断证明 7g 40 平台是"血统 repertoire 损失 × 种子注入重奠基 × 选择噪声"的
# 搜索困局：7b 脑在 7g 适应度下可得 ~60 分而种群困在 ~40。本工具把供体脑
# （默认 7b 67 分模型）以 k 份副本写入现有全种群 checkpoint 的尾部子代行，
# 让下一代评估自然竞争进精英——为血统重新引入"穿行/折叠"技能基因。
#
# 用法：
#   python experiments/inject_donor.py                       # 默认注入 8 份
#   python experiments/inject_donor.py --eval                # 注入前先在曼哈顿口径下
#                                                            # 克隆评估供体实力（GPU 分钟级）
#   python experiments/inject_donor.py --copies 16 --out xxx.pth
# 之后：把产物复制为 test7h_econ_checkpoint.pth 即可 AUTO_RESUME 全种群真续训。
# ==========================================
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'experiments', 'test7_series'))  # 归档后模块路径
import test7g as t7  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 供体 brain state 键 → GeneStack.GENES 键
_KEYMAP = {'M_in': 'M_in', 'M_rec': 'M_rec', 'M_out': 'M_out',
           'W_in': 'W_in', 'W_rec': 'W_rec', 'W_out': 'W_out',
           'b_out': 'b_out', 'tau_e_init': 'tau_e', 'w_ei': 'w_ei', 'w_ie': 'w_ie'}


def load_cfg_eval(saved_cfg):
    """按目标 run（checkpoint）的 config 构造评估口径：供体在它将进入的环境里打分。

    pre-7g 键缺失时回退（OBS_MANHATTAN=False, FATIGUE_TURN_GAIN=0.0），再由
    checkpoint config 覆盖——即评估口径=注入目标 run 的口径。
    """
    cfg = t7.Config()
    cfg.OBS_MANHATTAN = False
    cfg.FATIGUE_TURN_GAIN = 0.0
    for k, v in saved_cfg.items():
        if hasattr(cfg, k) and isinstance(v, (int, float, bool, str)):
            setattr(cfg, k, v)
    return cfg


def eval_donor(cfg, st, episodes):
    """克隆 episodes 份并行各打 1 局（=训练 _eval_chunk 口径），返回统计。"""
    dev = t7._resolve_device(cfg)
    pop = t7.GeneStack(cfg, B=episodes, device=dev)
    pop.random_init()
    for i in range(episodes):
        pop.set_individual_from_state(i, st)
    if cfg.USE_FP16:
        pop.fp16()
    pop.refresh_eff()
    m = t7._eval_chunk(pop, cfg)
    food = m[:, 0].numpy()
    turn = ((m[:, 8] + m[:, 9]) / (m[:, 1] + m[:, 2]).clamp(min=1)).numpy()
    return dict(food_mean=float(food.mean()), food_std=float(food.std()),
                food_min=float(food.min()), food_max=float(food.max()),
                turn_density=float(turn.mean()),
                manhattan=bool(cfg.OBS_MANHATTAN),
                fatigue=float(cfg.FATIGUE_TURN_GAIN))


def main():
    ap = argparse.ArgumentParser(description='供体脑注入全种群 checkpoint')
    ap.add_argument('--checkpoint', default='test7g_econ_checkpoint copy.pth',
                    help='源全种群 checkpoint（默认 7g 续训断点副本）')
    ap.add_argument('--donor', default='artifacts/test7b/test7b_best_model.pth', help='供体模型文件')
    ap.add_argument('--copies', type=int, default=8, help='注入副本数（写入尾部子代行）')
    ap.add_argument('--out', default='test7h_seed_checkpoint.pth', help='输出 checkpoint 路径')
    ap.add_argument('--eval', action='store_true', help='注入前在目标 run 口径下评估供体')
    ap.add_argument('--eval-episodes', type=int, default=40)
    args = ap.parse_args()

    ck_path = os.path.join(ROOT, args.checkpoint)
    ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    P = ck['pop']['W_in'].shape[0]
    cfg_run = ck.get('config', {})
    print(f"[源] {args.checkpoint} | pop={P} | next_gen={ck.get('next_gen')} | "
          f"best_food={ck.get('best_food')}")

    donor_path = os.path.join(ROOT, args.donor)
    ddata = torch.load(donor_path, map_location='cpu', weights_only=False)
    st = ddata.get('brain')
    if st is None:
        sys.exit(f'[错误] {args.donor} 无 brain 字段')
    dcfg = ddata.get('config', {})
    # 维度校验：供体必须与目标种群同构
    for name, dim in [('NUM_COLUMNS', cfg_run.get('NUM_COLUMNS')),
                      ('OBS_DIM', cfg_run.get('OBS_DIM')),
                      ('ACTION_DIM', cfg_run.get('ACTION_DIM'))]:
        if dcfg.get(name) != dim:
            sys.exit(f'[错误] 供体 {name}={dcfg.get(name)} 与种群 {dim} 不匹配')
    print(f"[供体] {args.donor} | 训练时 food={ddata.get('food')} | N={dcfg.get('NUM_COLUMNS')}")

    cfg_eval = load_cfg_eval(cfg_run)
    cfg_eval.EVAL_EPISODES = 1
    cfg_eval.MAX_STEPS = min(cfg_eval.MAX_STEPS, 20000)

    if args.eval:
        print(f"[评估] 供体在目标口径下（manhattan={cfg_eval.OBS_MANHATTAN}, "
              f"fatigue={cfg_eval.FATIGUE_TURN_GAIN}）克隆 {args.eval_episodes} 局...")
        stats = eval_donor(cfg_eval, st, args.eval_episodes)
        print(f"  food {stats['food_mean']:.1f} ± {stats['food_std']:.1f} "
              f"(min {stats['food_min']:.0f} / max {stats['food_max']:.0f}) | "
              f"整局转向密度 {stats['turn_density']:.2f}")
        with open(os.path.join(ROOT, 'results', 'donor_eval.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=1)

    # ---- 注入：尾部 k 行写供体（张量 half 对齐种群） ----
    k = min(args.copies, P)
    new_pop = dict(ck['pop'])
    for skey, gkey in _KEYMAP.items():
        t_donor = st[skey].to(torch.float32)
        # uint8 掩码 → half 0/1；权重直接 half
        t_donor = t_donor.to(ck['pop'][gkey].dtype)
        rows = new_pop[gkey]
        rows[P - k:] = t_donor.unsqueeze(0)
        new_pop[gkey] = rows
    ck['pop'] = new_pop
    ck['donor_injected'] = dict(src=args.donor, copies=k, rows=[P - k, P - 1],
                                donor_food=float(ddata.get('food', -1)))

    out_path = os.path.join(ROOT, args.out)
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(ck, out_path)
    print(f"[注入] 行 {P - k}..{P - 1}（尾部子代行）× {k} 份 ← 供体基因")
    print(f"[写出] {out_path}（源文件未动）")

    # ---- 结构自检：重载并比对 ----
    ck2 = torch.load(out_path, map_location='cpu', weights_only=False)
    ok = True
    for skey, gkey in _KEYMAP.items():
        a = ck2['pop'][gkey][P - 1].to(torch.float32)
        b = st[skey].to(torch.float32)
        if a.shape != b.shape or not torch.equal(a, b):
            ok = False
            print(f'  [自检失败] {gkey}')
    print(f"[自检] 供体行一致: {'通过' if ok else '失败'} | next_gen={ck2.get('next_gen')} "
          f"| best_food={ck2.get('best_food')}")
    print(f"\n[后续] 复制为 test7h 断点后启动全种群真续训：")
    print(f'  cp "{out_path}" test7h_econ_checkpoint.pth')


if __name__ == '__main__':
    main()
