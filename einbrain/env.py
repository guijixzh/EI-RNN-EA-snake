"""轻量级贪吃蛇环境（射线视野 + 无奖励设计）。

统一采用 test6 的超集语义：
    - step 返回 (next_obs, ate, done, truncated)
      done     = 撞墙 / 撞到身体（真正死亡）
      truncated = 饿死（steps_without_food 超时）或超过 MAX_STEPS
    - 自体感知 8 方向桶按『自身长度』归一化（短蛇/长蛇各节相对位置可辨）
    - 空转检测（head_history 环形缓冲）与饥饿窗口由 cfg 控制

观测布局 [0:24]：
    [0:3]    食物方向 bit（前 / 左前 / 右前）
    [3]      食物距离（欧氏距离，归一化到 [0,1]）
    [4:14]   5 条射线 × 2（自由路径比, 食物信号）
    [14:22]  自体感知：蛇头朝向参考系下 8 个方向桶（近端度）
    [22:24]  蛇尾在蛇头局部坐标系下的相对坐标
"""
from __future__ import annotations

import collections
import math
import random

import numpy as np


def _obs_sees_food(obs):
    """判断当前观测是否『看到食物』：任一射线食物信号 > 0。

    批量版本：传入 [B, 24] 数组时返回 [B] 布尔向量。
    """
    o = np.asarray(obs)
    if o.ndim == 1:
        return bool(np.any(o[5:14:2] > 0.0))
    return np.any(o[:, 5:14:2] > 0.0, axis=1)


def ray_obs_sees_food(obs):
    """判断『RaySnakeEnv』观测是否看到食物（8 扇区食物块 [8:16]）。

    批量版本：传入 [B, 32] 数组时返回 [B] 布尔向量。
    """
    o = np.asarray(obs)
    if o.ndim == 1:
        return bool(np.any(o[8:16] > 0.0))
    return np.any(o[:, 8:16] > 0.0, axis=1)


class SnakeEnv:
    def __init__(self, grid_size=10, max_steps=500, cfg=None):
        self.grid_size = grid_size
        self.obs_dim = 24
        self.max_steps = max_steps
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.head = (self.grid_size // 2, self.grid_size // 2)
        self.dir = random.choice(((0, 1), (1, 0), (0, -1), (-1, 0)))
        self.body = [self.head, (self.head[0] - self.dir[0], self.head[1] - self.dir[1])]
        self._place_food()
        self.food_count = 0
        self.steps = 0
        self.steps_without_food = 0
        if self.cfg is not None:
            window = max(2, int(getattr(self.cfg, 'LOITER_WINDOW', 12)))
        else:
            window = 12
        self.head_history = collections.deque(maxlen=window)
        self.loiter_now = False
        return self._get_obs()

    def _place_food(self):
        while True:
            self.food = (random.randint(0, self.grid_size - 1),
                         random.randint(0, self.grid_size - 1))
            if self.food not in self.body:
                break

    def _cast_ray(self, direction, tail_included=True):
        """沿 direction 发射射线，返回 (自由路径长度比, 食物信号)。

        tail_included=True 时检测完整身体（吃到食物、尾巴不移动的场景）；
        False 时排除即将移走的尾巴（与 step 的碰撞检测保持一致）。
        """
        max_dist = self.grid_size
        obstacle_dist = max_dist
        food_dist = -1

        body_set = self.body if tail_included else self.body[:-1]

        for step in range(1, max_dist + 1):
            r = self.head[0] + direction[0] * step
            c = self.head[1] + direction[1] * step

            if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
                obstacle_dist = step
                break
            if (r, c) in body_set:
                obstacle_dist = step
                break
            if (r, c) == self.food and food_dist < 0:
                food_dist = step

        free_path_length = obstacle_dist / max_dist
        if food_dist > 0:
            food_signal = 1.0 - (food_dist / max_dist)
        else:
            food_signal = 0.0
        return free_path_length, food_signal

    def _get_obs(self, will_eat=False):
        obs = np.zeros(self.obs_dim, dtype=np.float32)

        left_dir = (-self.dir[1], self.dir[0])
        right_dir = (self.dir[1], -self.dir[0])
        dx = self.food[0] - self.head[0]
        dy = self.food[1] - self.head[1]
        if (dx * self.dir[0] + dy * self.dir[1]) > 0:       obs[0] = 1.0
        if (dx * left_dir[0] + dy * left_dir[1]) > 0:       obs[1] = 1.0
        if (dx * right_dir[0] + dy * right_dir[1]) > 0:     obs[2] = 1.0
        obs[3] = min(math.hypot(dx, dy) / (self.grid_size * math.sqrt(2)), 1.0)

        ray_dirs = [
            left_dir,
            (self.dir[0] + left_dir[0], self.dir[1] + left_dir[1]),
            self.dir,
            (self.dir[0] + right_dir[0], self.dir[1] + right_dir[1]),
            right_dir,
        ]
        tail_included = will_eat
        for i, rd in enumerate(ray_dirs):
            free_path, food_sig = self._cast_ray(rd, tail_included=tail_included)
            obs[4 + i * 2]     = free_path
            obs[4 + i * 2 + 1] = food_sig

        # --- 自体感知：8 方向桶（蛇头朝向参考系），按自身长度归一化 ---
        dxh, dyh = self.dir
        self_len = max(1.0, float(len(self.body) - 1))
        for seg in self.body[1:]:
            wx = seg[0] - self.head[0]
            wy = seg[1] - self.head[1]
            rot_x = wx * dxh + wy * dyh
            rot_y = -wx * dyh + wy * dxh
            ang = math.degrees(math.atan2(rot_y, rot_x))
            if ang < 0:
                ang += 360.0
            bucket = int((ang + 22.5) // 45) % 8
            closeness = 1.0 - min(math.hypot(wx, wy) / self_len, 1.0)
            if closeness > obs[14 + bucket]:
                obs[14 + bucket] = closeness

        # --- 蛇尾局部坐标（追尾策略的关键线索）---
        tail = self.body[-1]
        wx = tail[0] - self.head[0]
        wy = tail[1] - self.head[1]
        obs[22] = (wx * dxh + wy * dyh) / self.grid_size
        obs[23] = (-wx * dyh + wy * dxh) / self.grid_size

        return obs

    def _check_loiter(self, next_head):
        """蛇头在 LOITER_WINDOW 步窗口内重复 ≥ LOITER_REPEAT 次 → 空转。"""
        if self.cfg is None:
            return False
        rep = int(getattr(self.cfg, 'LOITER_REPEAT', 3))
        self.head_history.append(next_head)
        return self.head_history.count(next_head) >= rep

    def step(self, action):
        """执行动作。返回 (next_obs, ate, done, truncated)。"""
        if action == 1:
            self.dir = (-self.dir[1], self.dir[0])
        elif action == 2:
            self.dir = (self.dir[1], -self.dir[0])

        next_head = (self.head[0] + self.dir[0], self.head[1] + self.dir[1])
        self.steps += 1
        self.steps_without_food += 1

        will_eat = (next_head == self.food)
        body_to_check = self.body if will_eat else self.body[:-1]
        if (next_head[0] < 0 or next_head[0] >= self.grid_size or
            next_head[1] < 0 or next_head[1] >= self.grid_size or
            next_head in body_to_check):
            return self._get_obs(will_eat), False, True, False

        self.body.insert(0, next_head)
        self.head = next_head

        ate_food = False
        if self.head == self.food:
            self.food_count += 1
            ate_food = True
            self.steps_without_food = 0
            self._place_food()
        else:
            self.body.pop()

        self.loiter_now = self._check_loiter(next_head)

        starve_bias = int(getattr(self.cfg, 'STARVE_BIAS', 12)) if self.cfg else 12
        if self.steps_without_food > 2 * len(self.body) + starve_bias:
            return self._get_obs(), ate_food, False, True
        if self.steps >= self.max_steps:
            return self._get_obs(), ate_food, False, True

        return self._get_obs(), ate_food, False, False


# ==================== RaySnakeEnv：32 维头朝向相对 8 方向观测 ====================

# 绝对 4 基本方向 one-hot 编码（与 BatchedRaySnakeEnv 共享）
RAY_DIRS_ABS = [(0, 1), (1, 0), (0, -1), (-1, 0)]          # 0:右 1:下 2:左 3:上
RAY_DIR_IDX = {d: i for i, d in enumerate(RAY_DIRS_ABS)}


def _ray_8_dirs(head_dir):
    """由蛇首朝向 (dr, dc) 派生 8 个头部相对方向向量。

    顺序（相对蛇首逆时针）：[前, 左前, 左, 左后, 后, 右后, 右, 右前]。
    """
    dr, dc = head_dir
    left = (-dc, dr)
    right = (dc, -dr)
    return [
        (dr, dc),              # 0 前
        (dr - dc, dc + dr),    # 1 左前
        (-dc, dr),             # 2 左
        (-dr - dc, -dc + dr),  # 3 左后
        (-dr, -dc),            # 4 后
        (-dr + dc, -dc - dr),  # 5 右后
        (dc, -dr),             # 6 右
        (dr + dc, dc - dr),    # 7 右前
    ]


def _ray_sector(d8, vr, vc):
    """相对向量 (vr, vc) 落入的扇区索引（0..7）。

    按『相对蛇首朝向的旋转角』量化到 45° 扇区，边界点由 round 确定性判定
    （避免 dot 最近方向在 45° 边界上的平局歧义，如颈节恒在正后方）。
    """
    if vr == 0 and vc == 0:
        return 0
    hdr, hdc = d8[0]                                   # 蛇首朝向（前）
    rel = math.atan2(vc, vr) - math.atan2(hdc, hdr)
    rel_deg = math.degrees(rel) % 360.0
    return int(round(rel_deg / 45.0)) % 8


class RaySnakeEnv(SnakeEnv):
    """32 维头朝向相对 8 方向射线观测贪吃蛇。

    观测布局 [0:32]：
        [0:4]    蛇首方向 one-hot（绝对 4 基本方向）
        [4:8]    蛇尾方向 one-hot（蛇末节移动朝向，绝对 4 基本方向）
        [8:16]   食物 8 扇区距离倒数（沿头部相对 8 方向射线扫描食物距离）
        [16:24]  自身 8 扇区距离倒数（沿头部相对 8 方向射线扫描最近身体节距离）
        [24:32]  8 方向障碍距离倒数；其中 [28]（后）约定为 sqrt(蛇身长度/格子度)。

    食物/自身扇区改为连续距离倒数（1/dist），保留方向信息的同时提供距离梯度。
    """

    def __init__(self, grid_size=10, max_steps=500, cfg=None):
        super().__init__(grid_size, max_steps, cfg)
        self.obs_dim = 32

    def sees_food(self, obs):
        return ray_obs_sees_food(obs)

    def _get_obs(self, will_eat=False):
        obs = np.zeros(32, dtype=np.float32)
        head = self.head
        d = self.dir

        # --- [0:4] 蛇首方向 one-hot ---
        obs[RAY_DIR_IDX[d]] = 1.0

        # --- [4:8] 蛇尾方向 one-hot（末节移动朝向：从蛇尾指向其前一节）---
        tail = self.body[-1]
        prev = self.body[-2]
        td = (prev[0] - tail[0], prev[1] - tail[1])
        obs[4 + RAY_DIR_IDX[td]] = 1.0

        d8 = _ray_8_dirs(d)

        # --- [8:16] 食物 8 扇区距离倒数 ---
        for i, (dx, dy) in enumerate(d8):
            dist = self.grid_size + 1
            for step in range(1, self.grid_size + 1):
                r = head[0] + dx * step
                c = head[1] + dy * step
                if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
                    break
                if (r, c) == self.food:
                    dist = step
                    break
            obs[8 + i] = 1.0 / dist if dist <= self.grid_size else 0.0

        # --- [16:24] 自身 8 扇区距离倒数 ---
        body_set = set(self.body[1:])
        for i, (dx, dy) in enumerate(d8):
            dist = self.grid_size + 1
            for step in range(1, self.grid_size + 1):
                r = head[0] + dx * step
                c = head[1] + dy * step
                if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
                    break
                if (r, c) in body_set:
                    dist = step
                    break
            obs[16 + i] = 1.0 / dist if dist <= self.grid_size else 0.0

        # --- [24:32] 障碍距离倒数（8 方向射线）---
        body_set = set(self.body[1:])
        for i, (dx, dy) in enumerate(d8):
            dist = self.grid_size
            for step in range(1, self.grid_size + 1):
                r = head[0] + dx * step
                c = head[1] + dy * step
                if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
                    dist = step
                    break
                if (r, c) in body_set:
                    dist = step
                    break
            obs[24 + i] = 1.0 / dist

        # 身后约定：障碍数 = sqrt(蛇身长度/阶数)
        obs[24 + 4] = np.sqrt(len(self.body)/self.grid_size)

        return obs
