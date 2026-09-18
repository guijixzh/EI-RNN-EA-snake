#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SiNNtry V5 最终三算符显式控制器 —— 单文件可运行版

不加载任何神经网络、不加载任何训练权重、不需要前面的蒸馏模型。
运行时只使用最终显式规则：

1) 即时合法性 L(a)
2) 未来头尾拓扑连通性 R(a)
3) 食物可达性 G(a)
4) 食物 Manhattan 距离 D_F(a)
5) 尾部 Manhattan 距离 D_T(a)

最终决策：
  A_safe = { a | L(a)=1 and R(a)=1 }

  若存在 a∈A_safe 且 G(a)=1：
      A = argmin D_F(a)
  否则：
      A = argmax D_T(a)

完全平局时优先直行，其次左转、右转。

默认：10×10 棋盘，STARVE_SLOPE=3，最多 8000 步，100 局。
依赖：python >= 3.9, numpy, torch

示例：
  python SiNNtry_V5_final_formula_run.py
  python SiNNtry_V5_final_formula_run.py --episodes 200 --seed 20270001
  python SiNNtry_V5_final_formula_run.py --episodes 20 --print-each
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


# ============================================================
# 0. 固定环境参数
# ============================================================
G = 10
MAXLEN = G * G
ALL = (1 << (G * G)) - 1
DIRS_NP = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=np.int64)
DIRS_T = torch.tensor(DIRS_NP, dtype=torch.long)

NOT_RIGHT = 0
NOT_LEFT = 0
for r in range(G):
    for c in range(G):
        bit = 1 << (r * G + c)
        if c < G - 1:
            NOT_RIGHT |= bit
        if c > 0:
            NOT_LEFT |= bit


# ============================================================
# 1. 与原评估口径一致的最小批量贪吃蛇环境
# ============================================================
class SnakeBatch:
    """只保留最终公式闭环所需的环境状态与更新逻辑。"""

    def __init__(self, episodes: int, starve_slope: float = 3.0):
        self.B = int(episodes)
        self.starve_slope = float(starve_slope)
        self.reset()

    def reset(self):
        B = self.B
        center = G // 2

        self.draw_cnt = torch.zeros(B, dtype=torch.long)
        self.head = torch.full((B, 2), center, dtype=torch.long)
        self.dir_idx = torch.randint(0, 4, (B,), dtype=torch.long)

        self.body = torch.zeros(B, MAXLEN, 2, dtype=torch.long)
        self.body[:, 0] = self.head
        self.body[:, 1] = self.head - DIRS_T[self.dir_idx]
        self.body_len = torch.full((B,), 2, dtype=torch.long)

        self.food = self._place_food_init()
        self.alive = torch.ones(B, dtype=torch.bool)
        self.steps = torch.zeros(B, dtype=torch.long)
        self.steps_wo_food = torch.zeros(B, dtype=torch.long)
        self.died = torch.zeros(B, dtype=torch.long)  # 0存活 1撞墙 2撞己 3饿死

    def _next_cand(self):
        return torch.randint(0, G, (self.B, 2), dtype=torch.long)

    def _place_food_init(self):
        B = self.B
        head = self.head
        neck = self.body[:, 1]
        cand = self._next_cand()
        bad = (cand == head).all(dim=1) | (cand == neck).all(dim=1)
        for _ in range(31):
            if not bad.any():
                break
            re = self._next_cand()
            cand = torch.where(bad.unsqueeze(1), re, cand)
            bad = (cand == head).all(dim=1) | (cand == neck).all(dim=1)

        if bad.any():
            occ = torch.zeros(B, G * G, dtype=torch.bool)
            ar = torch.arange(B)
            occ[ar, head[:, 0] * G + head[:, 1]] = True
            occ[ar, neck[:, 0] * G + neck[:, 1]] = True
            idx = torch.argmax((~occ).float(), dim=1)
            fb = torch.stack((idx // G, idx % G), dim=1)
            cand = torch.where(bad.unsqueeze(1), fb, cand)
        return cand

    def _occupancy_flat(self, tail_invalid=False):
        B = self.B
        flat = self.body[:, :, 0] * G + self.body[:, :, 1]
        seg = torch.arange(MAXLEN)[None, :]
        valid = seg < self.body_len[:, None]
        if tail_invalid:
            valid &= seg < (self.body_len - 1)[:, None]
        occ = torch.zeros(B, G * G, dtype=torch.float32)
        occ.scatter_add_(1, flat.clamp(max=G * G - 1), valid.float())
        return occ

    def _place_food_after_eat(self, eat_mask):
        if not eat_mask.any():
            return
        B = self.B
        occ_b = self._occupancy_flat() > 0.5
        cand = self._next_cand()
        idx = cand[:, 0] * G + cand[:, 1]
        bad = occ_b.gather(1, idx.unsqueeze(1)).squeeze(1)

        for _ in range(31):
            if not bad.any():
                break
            re = self._next_cand()
            cand = torch.where(bad.unsqueeze(1), re, cand)
            idx = cand[:, 0] * G + cand[:, 1]
            bad = occ_b.gather(1, idx.unsqueeze(1)).squeeze(1)

        if bad.any():
            idx = torch.argmax((~occ_b).float(), dim=1)
            fb = torch.stack((idx // G, idx % G), dim=1)
            cand = torch.where(bad.unsqueeze(1), fb, cand)

        self.food = torch.where(eat_mask.unsqueeze(1), cand, self.food)

    def step(self, actions: torch.Tensor):
        B = self.B
        alive_f = self.alive

        # 0=直行, 1=左转, 2=右转
        nd_idx = torch.where(actions == 1, (self.dir_idx + 3) % 4, self.dir_idx)
        nd_idx = torch.where(actions == 2, (self.dir_idx + 1) % 4, nd_idx)
        self.dir_idx = nd_idx
        nd = DIRS_T[nd_idx]

        self.steps = torch.where(alive_f, self.steps + 1, self.steps)
        self.steps_wo_food = torch.where(alive_f, self.steps_wo_food + 1, self.steps_wo_food)

        next_head = self.head + nd
        out_b = (
            (next_head[:, 0] < 0) | (next_head[:, 0] >= G) |
            (next_head[:, 1] < 0) | (next_head[:, 1] >= G)
        )

        will_eat = (next_head == self.food).all(dim=1)
        occ = self._occupancy_flat(tail_invalid=True)
        occ_eat = self._occupancy_flat(tail_invalid=False)
        occ_use = torch.where(will_eat[:, None], occ_eat, occ)

        ar = torch.arange(B)
        flat_next = next_head[:, 0].clamp(0, G - 1) * G + next_head[:, 1].clamp(0, G - 1)
        hit = occ_use[ar, flat_next] > 0.5
        crash = alive_f & (out_b | hit)

        move = alive_f & (~crash)
        shifted = torch.zeros_like(self.body)
        shifted[:, 0] = next_head
        shifted[:, 1:] = self.body[:, :-1]
        self.body = torch.where(move[:, None, None], shifted, self.body)
        self.head = torch.where(move[:, None], next_head, self.head)

        ate = move & will_eat
        self.body_len = torch.where(
            move,
            (self.body_len + ate.long()).clamp(max=MAXLEN),
            self.body_len,
        )
        self.steps_wo_food = torch.where(ate, torch.zeros_like(self.steps_wo_food), self.steps_wo_food)
        self._place_food_after_eat(ate)

        starve = self.steps_wo_food > (self.starve_slope * self.body_len.float() + 20.0)
        self.alive = alive_f & (~crash) & (~starve)

        crash_wall = alive_f & out_b
        crash_self = alive_f & (~out_b) & hit
        starve_now = alive_f & (~crash) & starve
        died_now = torch.where(
            crash_wall,
            torch.ones_like(self.steps),
            torch.where(
                crash_self,
                torch.full_like(self.steps, 2),
                torch.where(starve_now, torch.full_like(self.steps, 3), torch.zeros_like(self.steps)),
            ),
        )
        self.died = torch.where((self.died == 0) & (died_now > 0), died_now, self.died)

    def all_done(self):
        return not bool(self.alive.any().item())


# ============================================================
# 2. 最终公式所需的拓扑算符
# ============================================================
def flood_component(start_idx: int, food_idx: int, tail_idx: int, blocked: int):
    """
    在 10×10 四邻接自由图上，从未来蛇头做洪泛。

    返回：
      reach      : 可达集合的 bitset
      food_reach : 食物是否与未来蛇头同连通分量
      tail_reach : 未来蛇尾是否与未来蛇头同连通分量
    """
    free = ALL & ~blocked
    reach = 1 << start_idx

    for _ in range(G * G):
        nb = (
            (reach << G)
            | (reach >> G)
            | ((reach & NOT_RIGHT) << 1)
            | ((reach & NOT_LEFT) >> 1)
        ) & free & ALL
        nr = reach | nb
        if nr == reach:
            break
        reach = nr

    food_reach = ((reach >> food_idx) & 1) == 1
    tail_reach = ((reach >> tail_idx) & 1) == 1
    return reach, food_reach, tail_reach


def choose_actions(env: SnakeBatch):
    """最终 V5 显式控制方程。"""
    B = env.B
    actions = np.zeros(B, dtype=np.int64)  # 死亡个体/兜底保持直行

    alive = env.alive.numpy()
    lens = env.body_len.numpy().astype(int)
    dirs = env.dir_idx.numpy().astype(int)
    bodies = env.body.numpy().astype(int)
    foods = env.food.numpy().astype(int)

    for b in np.flatnonzero(alive):
        length = lens[b]
        body = bodies[b, :length]
        head = body[0]
        food = foods[b]
        food_idx = int(food[0]) * G + int(food[1])

        # pref[j] = 前 j 节身体占据 bitset
        pref = [0] * (length + 1)
        mask = 0
        for j, (r, c) in enumerate(body):
            mask |= 1 << (int(r) * G + int(c))
            pref[j + 1] = mask

        candidates = []

        # act 0=直行，1=左，2=右；turn 与原环境完全一致
        for act, turn in enumerate((0, -1, 1)):
            nd = (dirs[b] + turn) % 4
            nh = head + DIRS_NP[nd]

            # L(a)：即时合法性
            if np.any(nh < 0) or np.any(nh >= G):
                continue

            ni = int(nh[0]) * G + int(nh[1])
            eat = bool(np.array_equal(nh, food))
            head_bit = 1 << (int(head[0]) * G + int(head[1]))

            # 不吃时尾巴会移动，所以允许进入当前尾格；吃时尾巴不动。
            collision = (pref[length] if eat else pref[length - 1]) & ~head_bit
            if (collision >> ni) & 1:
                continue

            # 构造动作后的拓扑阻塞集合与未来尾巴。
            blocked = pref[length - 1] if eat else pref[max(length - 2, 0)]
            future_tail = body[length - 1] if eat else body[max(length - 2, 0)]
            ti = int(future_tail[0]) * G + int(future_tail[1])

            _, food_reach, tail_reach = flood_component(ni, food_idx, ti, blocked)

            # 两个最终保留的整数几何距离。
            food_manhattan = int(abs(int(nh[0]) - int(food[0])) + abs(int(nh[1]) - int(food[1])))
            tail_manhattan = int(
                abs(int(nh[0]) - int(future_tail[0])) + abs(int(nh[1]) - int(future_tail[1]))
            )

            candidates.append(
                {
                    "act": act,
                    "R": bool(tail_reach),   # 未来头尾拓扑连通
                    "G": bool(food_reach),   # 食物是否处于同一自由连通域
                    "DF": food_manhattan,
                    "DT": tail_manhattan,
                }
            )

        if not candidates:
            actions[b] = 0
            continue

        # A_safe = {a | L(a)=1, R(a)=1}
        safe = [q for q in candidates if q["R"]]
        if not safe:
            # 理论上的兜底：若所有合法动作都破坏头尾连通，则至少选即时合法动作。
            safe = candidates

        # 若至少一个安全动作仍能到达食物：最小化 D_F(a)
        food_pool = [q for q in safe if q["G"]]
        if food_pool:
            best_value = min(q["DF"] for q in food_pool)
            best = [q for q in food_pool if q["DF"] == best_value]
        else:
            # 食物暂时不可达：最大化 D_T(a)，追尾释放空间。
            best_value = max(q["DT"] for q in safe)
            best = [q for q in safe if q["DT"] == best_value]

        # 完全平局时固定优先：直行 > 左 > 右。
        best.sort(key=lambda q: (0 if q["act"] == 0 else 1 if q["act"] == 1 else 2))
        actions[b] = best[0]["act"]

    return torch.tensor(actions, dtype=torch.long)


# ============================================================
# 3. 闭环实跑与统计
# ============================================================
def run(episodes: int, seed: int, max_steps: int, starve_slope: float):
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))

    env = SnakeBatch(episodes, starve_slope=starve_slope)
    start = time.time()

    for _ in range(max_steps):
        if env.all_done():
            break
        env.step(choose_actions(env))

    food = (env.body_len - 2).numpy().astype(int)
    death = env.died.numpy().astype(int)
    steps = env.steps.numpy().astype(int)

    result = {
        "controller": "SiNNtry V5 final explicit 3-operator controller",
        "episodes": int(episodes),
        "seed": int(seed),
        "grid_size": G,
        "max_steps": int(max_steps),
        "starve_slope": float(starve_slope),
        "food_mean": float(food.mean()),
        "food_median": float(np.median(food)),
        "food_std": float(food.std(ddof=1)) if episodes > 1 else 0.0,
        "food_min": int(food.min()),
        "food_max": int(food.max()),
        "ge62": int((food >= 62).sum()),
        "ge80": int((food >= 80).sum()),
        "ge90": int((food >= 90).sum()),
        "perfect98": int((food >= 98).sum()),
        "death_wall": int((death == 1).sum()),
        "death_self": int((death == 2).sum()),
        "death_starve": int((death == 3).sum()),
        "alive_at_cutoff": int((death == 0).sum()),
        "steps_mean": float(steps.mean()),
        "runtime_sec": float(time.time() - start),
        "food": food.tolist(),
        "death": death.tolist(),
        "steps": steps.tolist(),
    }
    return result


def print_summary(r, print_each=False):
    print("\n" + "=" * 68)
    print("SiNNtry V5 最终显式公式闭环结果")
    print("=" * 68)
    print(f"局数                 : {r['episodes']}")
    print(f"随机种子             : {r['seed']}")
    print(f"平均吃食             : {r['food_mean']:.2f}")
    print(f"中位数               : {r['food_median']:.2f}")
    print(f"标准差               : {r['food_std']:.2f}")
    print(f"最低 / 最高          : {r['food_min']} / {r['food_max']}")
    print(f">= 62 食             : {r['ge62']}/{r['episodes']}")
    print(f">= 80 食             : {r['ge80']}/{r['episodes']}")
    print(f">= 90 食             : {r['ge90']}/{r['episodes']}")
    print(f"98 食通关            : {r['perfect98']}/{r['episodes']}")
    print(f"撞墙 / 自撞 / 饿死   : {r['death_wall']} / {r['death_self']} / {r['death_starve']}")
    print(f"截断时仍存活         : {r['alive_at_cutoff']}")
    print(f"平均步数             : {r['steps_mean']:.1f}")
    print(f"运行时间             : {r['runtime_sec']:.2f} s")

    if print_each:
        print("\n逐局结果：")
        for i, (f, d, s) in enumerate(zip(r["food"], r["death"], r["steps"]), 1):
            reason = {0: "存活/截断", 1: "撞墙", 2: "自撞", 3: "饿死"}.get(d, str(d))
            print(f"  #{i:03d}: food={f:2d}, steps={s:4d}, end={reason}")


def main():
    p = argparse.ArgumentParser(description="SiNNtry V5 最终三算符显式控制器：单文件闭环实跑")
    p.add_argument("--episodes", type=int, default=100, help="并行实跑局数，默认 100")
    p.add_argument("--seed", type=int, default=20260912, help="随机种子")
    p.add_argument("--max-steps", type=int, default=8000, help="每局最大步数，默认 8000")
    p.add_argument("--starve-slope", type=float, default=3.0, help="饿死斜率，默认 3")
    p.add_argument("--print-each", action="store_true", help="额外打印每一局结果")
    p.add_argument("--out", default="SiNNtry_V5_run_result.json", help="结果 JSON 输出路径")
    args = p.parse_args()

    if args.episodes < 1:
        raise SystemExit("--episodes 必须 >= 1")

    r = run(args.episodes, args.seed, args.max_steps, args.starve_slope)
    print_summary(r, args.print_each)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存：{out.resolve()}")


if __name__ == "__main__":
    main()
