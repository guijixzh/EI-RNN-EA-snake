"""ProjSnakeEnv（einbrain/env.py）与 test7b BatchedSnakeEnv 的等价性对拍。

共用一个 torch CPU Generator 驱动：初始朝向/食物、每步动作、吃食后的食物重放，
两套环境消费同一决策流，逐步比对 obs（[B,32]）、蛇身/长度/步数/死因。
容差内一致 → PPO 微调时 test7b 脑收到的输入分布与进化期一致。

用法: python experiments/verify_proj_env.py [B] [steps]
"""
import os
import sys
import importlib.util

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

spec = importlib.util.spec_from_file_location('t7b', os.path.join(ROOT, 'test7b.py'))
t7b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t7b)

from einbrain.env import ProjSnakeEnv


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    cfg = t7b.Config()
    cfg.MAX_STEPS = 10 ** 9          # 饿死/撞死自然终局，排除 MAX_STEPS 干扰
    dev = t7b._resolve_device(cfg)
    G = cfg.GRID_SIZE
    print(f"device={dev}  B={B}  steps<={STEPS}")

    gen = torch.Generator().manual_seed(20260830)

    # --- 食物重放公共流（吃食事件按序消费）---
    food_stream = torch.randint(0, G, (20000, 2), generator=gen)
    food_ptr = 0

    # --- 批量环境（test7b 原版）---
    benv = t7b.BatchedSnakeEnv(cfg, B, dev)
    dirs0 = torch.randint(0, 4, (B,), generator=gen)
    foods0 = torch.randint(0, G, (B, 2), generator=gen)
    benv.reset()
    ar = torch.arange(B, device=dev)
    benv.dir_idx[:] = dirs0.to(dev)
    benv.body[:, 1] = benv.head - t7b._make_dirs(dev)[dirs0.to(dev)]
    benv.food[:] = foods0.to(dev)

    # --- CPU 复刻环境 ---
    penvs = [ProjSnakeEnv(grid_size=G, max_steps=10 ** 9, cfg=cfg) for _ in range(B)]
    for i in range(B):
        penvs[i].reset(dir_idx=int(dirs0[i]), food=foods0[i].tolist())

    max_obs_diff = 0.0
    mismatches = []
    n_steps_alive = 0
    n_eats = 0
    n_starve = 0
    n_episode = 0

    t = 0
    while t < STEPS:
        acts = torch.randint(0, 3, (B,), generator=gen)
        acts_dev = acts.to(dev)
        alive_before = benv.alive.clone()

        # --- 比对死亡前状态集合（两端必须同步存活）---
        p_alive = [i for i in range(B) if penvs[i].died == 0]
        b_alive = alive_before.nonzero().flatten().tolist()
        if set(p_alive) != set(b_alive):
            mismatches.append(f"t={t} 存活集不同步: batch={sorted(b_alive)[:8]}... proj={sorted(p_alive)[:8]}...")
            break

        if not b_alive:
            # 全灭 → 两端同步重开新局（新朝向/新食物），继续扩大覆盖
            n_episode += 1
            dirs0 = torch.randint(0, 4, (B,), generator=gen)
            foods0 = torch.randint(0, G, (B, 2), generator=gen)
            benv.reset()
            benv.dir_idx[:] = dirs0.to(dev)
            benv.body[:, 1] = benv.head - t7b._make_dirs(dev)[dirs0.to(dev)]
            benv.food[:] = foods0.to(dev)
            for i in range(B):
                penvs[i].reset(dir_idx=int(dirs0[i]), food=foods0[i].tolist())
            ob_all = benv.obs()
            for i in range(B):
                d = float(np.abs(ob_all[i].cpu().numpy() - penvs[i]._get_obs()).max())
                max_obs_diff = max(max_obs_diff, d)
                if d > 1e-3:
                    mismatches.append(f"t={t} 重置后 env{i} obs_diff={d:.2e}")
            t += 1
            continue

        # 直线偏置动作（0.6/0.2/0.2），让蛇身变长、覆盖多扇区观测与饿死路径
        acts = torch.where(torch.rand(B, generator=gen) < 0.6, torch.zeros(B, dtype=torch.long), acts)
        acts_dev = acts.to(dev)

        # --- 步进：批量环境整体推一步 ---
        benv.step(acts_dev)
        ate_rows = (benv.ate & alive_before).nonzero().flatten().tolist()

        # --- 吃食者重放公共食物流（覆盖批量环境内部随机放置）---
        stream_cell = {}
        for i in ate_rows:
            cell = food_stream[food_ptr].tolist()
            food_ptr += 1
            benv.food[i] = torch.tensor(cell, device=dev)
            stream_cell[i] = tuple(cell)
            n_eats += 1

        # --- 单环境步进 + 逐步比对 ---
        ob_all = benv.obs()
        for i in b_alive:
            o2, ate, done, trunc = penvs[i].step(int(acts[i]))
            if ate:
                penvs[i].food = stream_cell[i]   # 覆盖其内部随机放置（两端同步吃）
                o2 = penvs[i]._get_obs()         # step 返回的 obs 是覆盖前算的，重算
            n_steps_alive += 1
            ob = ob_all[i].cpu().numpy()
            d = float(np.abs(ob - o2).max())
            if d > max_obs_diff:
                max_obs_diff = d
            bad = []
            if d > 1e-3:
                bad.append(f"obs_diff={d:.2e}")
            if bool(benv.alive[i]) != (penvs[i].died == 0):
                bad.append(f"alive: batch={bool(benv.alive[i])} proj={penvs[i].died == 0}")
            if int(benv.body_len[i]) != penvs[i].body_len:
                bad.append(f"len: batch={int(benv.body_len[i])} proj={penvs[i].body_len}")
            if int(benv.steps[i]) != penvs[i].steps:
                bad.append(f"steps: {int(benv.steps[i])} vs {penvs[i].steps}")
            if int(benv.steps_wo_food[i]) != penvs[i].steps_without_food:
                bad.append(f"swf: {int(benv.steps_wo_food[i])} vs {penvs[i].steps_without_food}")
            if int(benv.died[i]) != penvs[i].died:
                bad.append(f"died: {int(benv.died[i])} vs {penvs[i].died}")
            bh = benv.body[i, 0].cpu().tolist()
            if tuple(bh) != penvs[i].head:
                bad.append(f"head: {tuple(bh)} vs {penvs[i].head}")
            if bad:
                mismatches.append(f"t={t} env{i} act={int(acts[i])}: " + "; ".join(bad))
                if len(mismatches) >= 10:
                    break
        if len(mismatches) >= 10:
            break

        n_starve += sum(1 for i in b_alive if penvs[i].died == 3)
        t += 1

    print(f"\n--- 对拍结果 ---")
    print(f"重开局数={n_episode}  存活步数={n_steps_alive}  吃食事件={n_eats}  "
          f"饿死={n_starve}  obs最大差={max_obs_diff:.3e}")
    if mismatches:
        print(f"不一致 {len(mismatches)} 处:")
        for m in mismatches:
            print("  " + m)
        sys.exit(1)
    print("PASS: ProjSnakeEnv 与 test7b BatchedSnakeEnv 等价")


if __name__ == '__main__':
    main()
