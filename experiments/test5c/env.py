import numpy as np
import math
import random


# ==========================================
# 贪吃蛇环境（test5c 版）
# ==========================================
# 与 test5a 的 SnakeEnv 行为完全一致（10x10, 3 动作, 无奖励设计），
# 提供 _get_grid_state() -> 10x10x5 多通道网格，供 CNN 前端使用
# （复制自 test5b/env.py；保持行为一致，使 CNN 与教师数据分布对齐）。
#
# 动作：0=直行, 1=左转, 2=右转
# ==========================================
class SnakeEnv:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.obs_dim = 24  # 保留（仅兼容信息；test5c 实际输入为 CNN 特征）
        self.reset()

    def reset(self):
        self.head = (self.grid_size // 2, self.grid_size // 2)
        self.dir = random.choice(((0, 1), (1, 0), (0, -1), (-1, 0)))
        self.body = [self.head, (self.head[0] - self.dir[0], self.head[1] - self.dir[1])]
        self._place_food()
        self.food_count = 0
        self.steps = 0
        self.steps_without_food = 0
        return self._get_obs()

    def _place_food(self):
        """在非蛇身空格中随机放置食物（空格枚举，O(100) 恒定开销）。"""
        gs = self.grid_size
        body_set = set(self.body)
        free = [(r, c) for r in range(gs) for c in range(gs)
                if (r, c) not in body_set]
        if not free:
            return False
        self.food = random.choice(free)
        return True

    # ---------- 原始 24 维观测（test5a 兼容，保留） ----------
    def _cast_ray(self, direction, tail_included=True):
        """沿 direction 发射射线，返回 (自由路径长度比, 食物信号)。"""
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
        """24 维观测，布局与 test5a 完全一致。"""
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

        # 自体感知：8 方向桶
        dxh, dyh = self.dir
        max_self_dist = self.grid_size * math.sqrt(2)
        for seg in self.body[1:]:
            wx = seg[0] - self.head[0]
            wy = seg[1] - self.head[1]
            rot_x = wx * dxh + wy * dyh
            rot_y = -wx * dyh + wy * dxh
            ang = math.degrees(math.atan2(rot_y, rot_x))
            if ang < 0:
                ang += 360.0
            bucket = int((ang + 22.5) // 45) % 8
            closeness = 1.0 - min(math.hypot(wx, wy) / max_self_dist, 1.0)
            if closeness > obs[14 + bucket]:
                obs[14 + bucket] = closeness

        # 蛇尾局部坐标
        tail = self.body[-1]
        wx = tail[0] - self.head[0]
        wy = tail[1] - self.head[1]
        obs[22] = (wx * dxh + wy * dyh) / self.grid_size
        obs[23] = (-wx * dyh + wy * dxh) / self.grid_size

        return obs

    # ---------- 新增：10x10x5 多通道网格状态（CNN 输入） ----------
    def _get_grid_state(self):
        """构造 10x10x5 网格张量（channel_last, float32）。

        通道布局：
          ch0: 食物位置 one-hot
          ch1: 身体占据（含头；full body）
          ch2: 头部 one-hot
          ch3: 蛇尾 one-hot
          ch4: 方向箭头场（正前方=1.0, 左前/右前=0.5）
        """
        gs = self.grid_size
        grid = np.zeros((gs, gs, 5), dtype=np.float32)

        # ch0: 食物
        grid[self.food[0], self.food[1], 0] = 1.0

        # ch1: 身体（全部）
        for seg in self.body:
            grid[seg[0], seg[1], 1] = 1.0

        # ch2: 头部
        grid[self.head[0], self.head[1], 2] = 1.0

        # ch3: 尾巴
        tail = self.body[-1]
        grid[tail[0], tail[1], 3] = 1.0

        # ch4: 方向箭头（头部前方 / 左前 / 右前）
        dx, dy = self.dir
        left = (-dy, dx)
        right = (dy, -dx)
        for d, val in ((self.dir, 1.0), (left, 0.5), (right, 0.5)):
            nr = self.head[0] + d[0]
            nc = self.head[1] + d[1]
            if 0 <= nr < gs and 0 <= nc < gs:
                grid[nr, nc, 4] = max(grid[nr, nc, 4], val)

        return grid

    def step(self, action):
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
            return self._get_obs(will_eat), False, True

        self.body.insert(0, next_head)
        self.head = next_head

        ate_food = False
        if self.head == self.food:
            self.food_count += 1
            ate_food = True
            self.steps_without_food = 0
            placed = self._place_food()
            if not placed:
                # 棋盘已满（蛇身占满 100 格）：游戏胜利结束
                return self._get_obs(), True, True
        else:
            self.body.pop()

        if self.steps_without_food > 2*len(self.body) + 20:
            return self._get_obs(), False, True

        return self._get_obs(), ate_food, False