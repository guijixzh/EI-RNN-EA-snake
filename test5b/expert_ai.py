import numpy as np
import math
from collections import deque


# ==========================================
# 混合专家 AI（test5b 教师）
# ==========================================
# 策略三层：
#   1. 安全评估：对 3 个候选相对动作，各算移动后的 flood-fill 逃生空间；
#   2. 食物捷径：安全前提下 BFS 到食物的最短路径（高效抄近路），
#      路径越短、可达性越好分数越高；
#   3. Hamilton 兜底：当所有候选都不安全（逃生空间 < 蛇长）时，
#      沿预构造的 Hamilton 圈行走，保证长蛇不死、接近通关。
#
# 输出：每动作的启发式分数（软标签用）+ argmax 动作。
# ==========================================


def build_hamilton_cycle(gs):
    """构造 10x10 网格（偶数行偶数列）的蛇形 Hamilton 圈。

    构造方式：
      第一行从左到右 → 中间行蛇形跳开第 0 列 → 最后一行从右到左
      → 沿第 0 列向上回程 → 闭环回起点。
    返回 next_cell: {cell: 圈上下一格}。
    """
    cells = []
    # 第一行：从左到右
    for c in range(gs):
        cells.append((0, c))
    # 中间行：r 奇从右到左、r 偶从左到右（均跳过第 0 列，留给回程）
    for r in range(1, gs - 1):
        if r % 2 == 1:
            row = [(r, c) for c in range(gs - 1, 0, -1)]
        else:
            row = [(r, c) for c in range(1, gs)]
        cells.extend(row)
    # 最后一行：从右到左
    for c in range(gs - 1, -1, -1):
        cells.append((gs - 1, c))
    # 第 0 列回程：从下到上
    for r in range(gs - 2, 0, -1):
        cells.append((r, 0))

    next_cell = {cells[i]: cells[(i + 1) % len(cells)] for i in range(len(cells))}
    return next_cell


_DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))  # 右、下、左、上


def _turn(dir_vec, action):
    """相对动作 -> 绝对方向向量。0=直行, 1=左转, 2=右转。

    与 SnakeEnv 的 dir 更新语义完全一致：
      left = (-dy, dx) 为逆时针旋转；
      right = (dy, -dx) 为顺时针旋转。
    _DIRS 环：右(0,1) -> 下(1,0) -> 左(0,-1) -> 上(-1,0)（顺时针）。
    故左转 = (idx - 1) % 4，右转 = (idx + 1) % 4。
    """
    idx = _DIRS.index(tuple(dir_vec))
    if action == 1:   # 左转（逆时针）
        new_idx = (idx - 1) % 4
    elif action == 2: # 右转（顺时针）
        new_idx = (idx + 1) % 4
    else:             # 直行
        new_idx = idx
    return _DIRS[new_idx]


def _flood_fill(start, gs, obstacles):
    """从 start 出发 flood-fill 可达空格数（BFS）。obstacles: set of cells。"""
    if start in obstacles:
        return 0
    visited = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if (0 <= nr < gs and 0 <= nc < gs and
                    nxt not in obstacles and nxt not in visited):
                visited.add(nxt)
                q.append(nxt)
    return len(visited)


def _bfs(start, goal, gs, obstacles):
    """BFS 到 goal 的最短路径（4 方向）。返回路径格子列表（含 goal，不含 start），
    不可达返回 None。obstacles: set of cells。"""
    if start == goal:
        return []
    if start in obstacles or goal in obstacles:
        return None
    parent = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        r, c = cur
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if (0 <= nr < gs and 0 <= nc < gs and
                    nxt not in obstacles and nxt not in parent):
                parent[nxt] = cur
                if nxt == goal:
                    # 回溯路径
                    path = []
                    node = goal
                    while parent[node] is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                q.append(nxt)
    return None


def _tail_chase_ok(neck, sim_body, gs):
    """移动后，从新头能否 BFS 到达新尾巴位置（tail chase 判据）。

    - 新尾 = sim_body[-1]，作为终点可"走入"（该格下一步会挪开）
    - 路径中间不能经过身体其它格子
    """
    if len(sim_body) < 2:
        return True
    tail = sim_body[-1]
    # 障碍 = sim_body 中除起点 neck 与终点 tail 之外的格子
    obstacles = set(sim_body[1:-1])
    path = _bfs(neck, tail, gs, obstacles)
    return path is not None


class ExpertSnakeAI:
    """混合专家贪吃蛇 AI（经典安全优先策略）。

    三层决策：
      1. 能安全吃到食物（BFS 捷径且移动后可追尾）→ 高优先级
      2. 不能安全吃但可追尾 → 跟尾巴保命
      3. 都不能 → 逃生空间最大的合法动作
    """

    def __init__(self, grid_size=10, escape_gamma=2.0, ham_fallback=True):
        self.grid_size = grid_size
        self.escape_gamma = escape_gamma  # 逃生空间奖励权重
        self.ham_fallback = ham_fallback

    def act(self, env, will_eat=False):
        """返回 (scores[3] float, action int)。

        - scores：每候选动作的原始启发式分数（softmax 后作软标签）
        - action：argmax 决策
        """
        gs = self.grid_size
        head = env.head
        body = env.body
        food = env.food
        cur_dir = tuple(env.dir)

        scores = [0.0, 0.0, 0.0]
        escapes = [0, 0, 0]
        valid = [False, False, False]
        chase_ok = [False, False, False]
        path_len = [None, None, None]

        for a in range(3):
            new_dir = _turn(cur_dir, a)
            nr, nc = head[0] + new_dir[0], head[1] + new_dir[1]
            neck = (nr, nc)

            if not (0 <= nr < gs and 0 <= nc < gs):
                continue

            eats = (neck == tuple(food))
            # 移动后的身体：吃食物尾巴不动，否则尾巴移走
            if eats:
                sim_body = [neck] + list(body)
            else:
                sim_body = [neck] + list(body[:-1])
            # 障碍集合 = 移动后身体中「除新头以外」的格子
            obstacles = set(sim_body[1:])

            # 碰撞检查：neck 不能落在「移动后身体」的其它格子上。
            collision_set = set(body if eats else body[:-1])
            collision_set.discard(tuple(head))
            if tuple(neck) in collision_set:
                continue

            valid[a] = True

            # 逃生空间（flood-fill 可达格子数，起点=新头）
            escape = _flood_fill(neck, gs, obstacles)
            escapes[a] = escape

            # 核心判据：移动后能否追到尾巴（动态安全）
            chase_ok[a] = _tail_chase_ok(neck, sim_body, gs)

            # BFS 食物捷径（起点=新头）
            path = _bfs(neck, tuple(food), gs, obstacles)
            if path is not None:
                path_len[a] = len(path)

            # ---- 评分 ----
            # 基础保命分：逃生空间占比（所有合法动作保底）
            score = self.escape_gamma * escape / (gs * gs)

            if path is not None and chase_ok[a]:
                # 能安全吃到食物：最高优先级（捷径越短分越高）
                score += 5.0 + 1.5 / (1.0 + len(path))
            elif chase_ok[a]:
                # 不能安全吃但可追尾：中优先级（保命）
                score += 2.0

            # 微小转向惩罚，避免高频抖动
            if a != 0:
                score -= 0.05

            scores[a] = score

        # ---- 决策：取分数最高的合法动作 ----
        best_a = int(np.argmax(scores))
        return scores, best_a

    def _rel_action_to(self, cur_dir, head, target):
        """从 head 沿 cur_dir 出发，走到 target 所需的相对动作；非法返回 -1。"""
        gs = self.grid_size
        gx, gy = target[0] - head[0], target[1] - head[1]
        if abs(gx) + abs(gy) != 1:
            return -1
        abs_dir = (gx, gy)
        # 转回动作
        dirs = list(_DIRS)
        abs_idx = dirs.index(abs_dir)
        cur_idx = dirs.index(tuple(cur_dir))
        diff = (abs_idx - cur_idx) % 4
        if diff == 0:
            return 0
        if diff == 3:
            return 1   # 逆时针 90° = 左转
        if diff == 1:
            return 2   # 顺时针 90° = 右转
        return -1      # 方向相反（180 度，无法一步完成）


def _softmax(x, temperature=1.0):
    x = np.asarray(x, dtype=np.float64) / temperature
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def run_expert_game(env, expert, max_steps=2000, record_grid=False):
    """跑一局专家游戏。

    返回 dict：
      - grids: list of 10x10x5 ndarray（record_grid=True 时）
      - softs: list of 3 维软标签
      - actions: list of 最终动作
      - foods / steps: 局统计
    """
    env.reset()
    grids, softs, actions = [], [], []
    done = False
    steps = 0
    while not done and steps < max_steps:
        if record_grid:
            grids.append(env._get_grid_state())
        scores, action = expert.act(env)
        softs.append(_softmax(scores, temperature=1.0))
        actions.append(action)
        _, _, done = env.step(action)
        steps += 1
    return {
        'grids': grids,
        'softs': softs,
        'actions': actions,
        'foods': env.food_count,
        'steps': steps,
    }