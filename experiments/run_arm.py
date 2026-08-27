# -*- coding: utf-8 -*-
"""
run_arm.py — 检查实验运行器：在 test7a 之上按指定 OBS/FIT/缩放组合跑一个实验臂。

用法示例（你的 GPU 全规模三臂）:
  python experiments/run_arm.py --tag r0_control --obs 24    --fit tuple  --gens 30
  python experiments/run_arm.py --tag r1_obs24  --obs 24    --fit scalar --gens 30
  python experiments/run_arm.py --tag r2_fit    --obs 32proj --fit tuple  --gens 30 --scale 8
  python experiments/run_arm.py --tag rfix      --obs 32proj --fit scalar --gens 30 --scale 8

断点文件/最优模型/历史曲线自动按 tag 命名，互不覆盖。
"""
import argparse
import importlib.util
import os

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--tag', required=True)
parser.add_argument('--obs', default='32proj', choices=['24', '32proj'])
parser.add_argument('--fit', default='scalar', choices=['scalar', 'tuple'])
parser.add_argument('--scale', type=float, default=None, help='OBS_FOOD_SCALE 覆盖')
parser.add_argument('--self-scale', type=float, default=None, help='OBS_SELF_SCALE 覆盖')
parser.add_argument('--obstacle-scale', type=float, default=None, help='OBS_OBSTACLE_SCALE 覆盖')
parser.add_argument('--gens', type=int, default=30)
parser.add_argument('--pop', type=int, default=2048)
parser.add_argument('--cpu', action='store_true', help='CPU 运行（自动关 fp16）')
parser.add_argument('--seed', type=int, default=42, help='固定随机种子（臂间可比）')
args = parser.parse_args()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location('t7a', os.path.join(ROOT, 'test7a.py'))
t7a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t7a)

cfg = t7a.Config()
cfg.OBS_MODE = args.obs
cfg.OBS_DIM = 24 if args.obs == '24' else 32
cfg.FIT_MODE = args.fit
if args.scale is not None:
    cfg.OBS_FOOD_SCALE = args.scale
if args.self_scale is not None:
    cfg.OBS_SELF_SCALE = args.self_scale
if args.obstacle_scale is not None:
    cfg.OBS_OBSTACLE_SCALE = args.obstacle_scale
cfg.GENERATIONS = args.gens
cfg.POP_SIZE = args.pop
cfg.ELITE_SIZE = max(4, cfg.POP_SIZE // 8)
cfg.CHECKPOINT_PATH = f'test7a_arm_{args.tag}_checkpoint.pth'
cfg.BEST_MODEL_PATH = f'test7a_arm_{args.tag}_best.pth'
cfg.LATEST_GEN_BEST_MODEL_PATH = f'test7a_arm_{args.tag}_latest_best.pth'
cfg.DEVICE = 'cpu' if args.cpu else 'auto'
if args.cpu:
    cfg.USE_FP16 = False

torch.manual_seed(args.seed)
print(f"[arm:{args.tag}] obs={cfg.OBS_MODE} fit={cfg.FIT_MODE} "
      f"scale={cfg.OBS_FOOD_SCALE}/self={cfg.OBS_SELF_SCALE}/obst={cfg.OBS_OBSTACLE_SCALE} "
      f"gens={cfg.GENERATIONS} pop={cfg.POP_SIZE} device={cfg.DEVICE} fp16={cfg.USE_FP16}")
t7a.run_training(cfg)
