# ==========================================
# experiments/ref_solver.py —— 理想参考解法器（chynl/snake GraphAgent 移植）+ CRN mini-env
#
# 参考蛇算法（chuyangliu/chynl 系）：
#   1. BFS 最短路去食物 + 前向检查：模拟吃完后仍能 BFS 到尾才走；
#   2. 不安全 → longer_path（侧绕一格加长绕行等尾腾位，3 步换 1 步）；
#   3. 仍无路 → 远离食物苟活；
#   4. 蛇长 >= 16：DFS+Warnsdorf 启发现算"头→尾、覆盖全部可达格"的哈密顿
#      路径并锁定索引（严格沿索引递减跟随；身体非路径格不在索引内，后继格
#      只可能是自由格或已腾出的尾 → 安全；索引耗尽（到达尾）时重建）。
#
# mini-env：纯 Python 逐个体复刻 test7h.BatchedSnakeEnv 规则（CRN bank 同构：
# 同 dir0、逐个体食物流消耗一致），`python experiments/ref_solver.py` 自带
# 与 GPU env 的逐位校验。
# ==========================================
import copy
import random
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'experiments', 'test7_series'))  # 归档后模块路径
import test7h as t7h  # noqa: E402

G = 10
DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]      # test7h DIRS（dr,dc）
DIR_IDX = {d: i for i, d in enumerate(DIRS)}


# ---------- CRN mini-env（逐个体，规则与 BatchedSnakeEnv 逐位一致）----------
class MiniEnv:
    def __init__(self, bank, starve_slope=3.0, max_steps=20000):
        self.stream = bank['stream'].tolist() if isinstance(bank['stream'], torch.Tensor) \
            else bank['stream']
        self.dir0 = int(bank['dir0'])
        self.starve_slope = starve_slope
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.head = (G // 2, G // 2)
        self.dir = self.dir0
        self.body = [self.head, (self.head[0] - DIRS[self.dir][0],
                                 self.head[1] - DIRS[self.dir][1])]
        self.draw = 0
        self.steps_wo_food = 0
        self.alive = True
        self.won = False
        self.died = 0
        self.acts = []
        self.path = [self.head]
        self.events = []
        self.snaps = []
        self.death_ctx = None
        self.food = self._draw_food(exclude=[self.head, self.body[1]])

    def _draw_food(self, exclude):
        while True:
            cand = tuple(self.stream[self.draw % len(self.stream)])
            self.draw += 1
            if cand not in exclude:
                return cand
            if len(exclude) >= G * G:        # 盘满（通关）
                self.won = True
                self.alive = False
                return self.food

    def step(self, action):
        if not self.alive:
            return
        if action == 1:
            self.dir = (self.dir + 3) % 4
        elif action == 2:
            self.dir = (self.dir + 1) % 4
        nh = (self.head[0] + DIRS[self.dir][0], self.head[1] + DIRS[self.dir][1])
        will_eat = (nh == self.food)
        occ = set(self.body) if will_eat else set(self.body[:-1])
        out = not (0 <= nh[0] < G and 0 <= nh[1] < G)
        hit = nh in occ
        self.steps_wo_food += 1
        if out or hit:
            self.alive = False
            self.died = 1 if out else 2
            self.death_ctx = dict(step=len(self.acts) + 1, L=len(self.body),
                                  head=nh, food=self.food,
                                  acts_tail=self.acts[-30:],
                                  body=list(self.body),
                                  free_reach=self._flood(nh, occ))
            self.acts.append(action)
            return
        self.body.insert(0, nh)
        self.head = nh
        if will_eat:
            self.events.append(len(self.acts) + 1)
            self.food = self._draw_food(exclude=set(self.body))
            self.snaps.append((list(self.body), self.food))
            self.steps_wo_food = 0
        else:
            self.body.pop()
        if self.steps_wo_food > self.starve_slope * len(self.body) + 20:
            self.alive = False
            self.died = 3
            occ = set(self.body)
            self.death_ctx = dict(step=len(self.acts) + 1, L=len(self.body),
                                  head=self.head, food=self.food,
                                  acts_tail=self.acts[-30:],
                                  body=list(self.body),
                                  free_reach=self._flood(self.head, occ),
                                  food_reachable=self.food in self._flood_cells(
                                      self.head, occ))
        self.acts.append(action)
        self.path.append(nh)

    def _flood(self, start, occupied):
        return len(self._flood_cells(start, occupied))

    def _flood_cells(self, start, occupied):
        seen = {start}
        q = [start]
        while q:
            r, c = q.pop()
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < G and 0 <= nc < G and (nr, nc) not in seen \
                        and (nr, nc) not in occupied:
                    seen.add((nr, nc))
                    q.append((nr, nc))
        return seen


def absdir_to_action(want_idx, cur_dir):
    """绝对方向索引 → env 相对动作。"""
    d = (want_idx - cur_dir) % 4
    return 0 if d == 0 else (2 if d == 1 else 1)


# ---------- 固定回路 + 单调序捷径参考蛇（每步局部可证安全）----------
# 不变量：从尾到头沿回路索引严格递增（模 100 单回绕）——即头→尾的前向段内
# 无身体。由此：(a) 回路后继格必在前向段内 → 永不撞、永不困 → 必通关；
# (b) 任何保持不变量的移动都安全：新头索引必须落在 (头, 尾) 开区间内
# （吃食时尾不动，同式；不吃时尾格腾出可入）。
class CycleSolver:
    def __init__(self):
        # 回路：行 0..9 蛇形（列 1..9），列 0 回廊（同 solver_reference）
        order = []
        for r in range(G):
            cols = range(1, G) if r % 2 == 0 else range(G - 1, 0, -1)
            order += [(r, c) for c in cols]
        order += [(r, 0) for r in range(G - 1, -1, -1)]
        self.idx = {cell: i for i, cell in enumerate(order)}
        self.N = G * G

    def fd(self, a, b):
        """回路前向距离 a→b。"""
        return (self.idx[b] - self.idx[a]) % self.N

    def next_action(self, env):
        body = env.body
        head, food, tail, neck = body[0], env.food, body[-1], body[1]
        body_set = set(body)
        L = len(body)

        # 候选：非颈邻格 && (空 || 尾(不吃时腾出) || 食物) && 保持不变量
        safe = []
        for di, (dr, dc) in enumerate(DIRS):
            n = (head[0] + dr, head[1] + dc)
            if n == neck:
                continue
            eating = (n == food)
            if n in body_set and n != tail and not eating:
                continue
            if not (0 <= n[0] < G and 0 <= n[1] < G):
                continue
            # 吃食（尾不动）：食物必须严格在前向开区间；不吃（尾腾出）：允许==尾
            if eating:
                if self.fd(head, n) >= self.fd(head, tail):
                    continue
            else:
                if self.fd(head, n) > self.fd(head, tail):
                    continue
            safe.append((di, n))
        if not safe:
            # 接管的中途状态可能不满足回路单调不变量 → 几何洪泛安全兜底
            return self._fallback(env, safe, body_set, L, food)

        # ---- 食物不在前向段：跟回路（后继恒安全）----
        if self.fd(head, food) >= self.fd(head, tail):
            succ_i = (self.idx[head] + 1) % self.N
            for di, n in safe:
                if self.idx[n] == succ_i:
                    return absdir_to_action(di, env.dir)
            return self._fallback(env, safe, body_set, L, food)

        # ---- 食物在段内：选使回路距离单调递减的安全邻格（必存在=回路后继，
        #      fd(n,food) = fd(head,food)-1）→ 无振荡、保证吃到 ----
        best, best_fd = None, None
        for di, n in safe:
            f = self.fd(n, food)
            if best_fd is None or f < best_fd:
                best, best_fd = di, f
        return absdir_to_action(best, env.dir)

    def _fallback(self, env, safe, body_set, L, food):
        """safe 空（初始身体可能不单调）时的几何洪泛安全兜底。"""
        head, tail = env.body[0], env.body[-1]
        best, best_key = 0, None
        for di, (dr, dc) in enumerate(DIRS):
            n = (head[0] + dr, head[1] + dc)
            if n == env.body[1] if len(env.body) >= 2 else False:
                continue
            if not (0 <= n[0] < G and 0 <= n[1] < G):
                continue
            occ = body_set - ({tail} if n != food else set())
            if n in occ:
                continue
            reach = len(self._flood(n, occ))
            if reach < L:
                continue
            key = (abs(n[0] - food[0]) + abs(n[1] - food[1]))
            if best_key is None or key < best_key:
                best, best_key = di, key
        return absdir_to_action(best, env.dir)

    def _flood(self, start, occupied):
        seen = {start}
        q = [start]
        while q:
            r, c = q.pop()
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < G and 0 <= nc < G and (nr, nc) not in seen \
                        and (nr, nc) not in occupied:
                    seen.add((nr, nc))
                    q.append((nr, nc))
        return seen


# ---------- chynl GraphAgent 移植（保留备查，B 实验默认用 CycleSolver）----------
class RefSolver:
    HAMILTON_THRESHOLD = 16
    HAMILTON_SEARCH_LIMIT = 10_000

    def __init__(self, seed=0):
        self.rand = random.Random(seed)
        self.reset()

    def reset(self):
        self.hamilton_index = None

    # --- 基础谓词：界内 && (空 || 食物 || 尾) ---
    def reachable(self, cell, body_set, food, tail):
        if not (0 <= cell[0] < G and 0 <= cell[1] < G):
            return False
        if cell == tail or cell == food:
            return True
        return cell not in body_set

    def neighbors(self, cell, body_set, food, tail, visited):
        out = []
        for dr, dc in DIRS:
            nbr = (cell[0] + dr, cell[1] + dc)
            if self.reachable(nbr, body_set, food, tail) and nbr not in visited:
                out.append(nbr)
        self.rand.shuffle(out)
        return out

    def bfs_cells(self, src, dst, body_set, food, tail):
        """src→dst 最短细胞序列（不含 src）；不可达 None；src==dst → []。"""
        if src == dst:
            return []
        visited = {src}
        q = [(src, [])]
        while q:
            cur, path = q.pop(0)
            if cur == dst:
                return path
            for nbr in self.neighbors(cur, body_set, food, tail, visited):
                q.append((nbr, path + [nbr]))
                visited.add(nbr)
        return None

    def next_action(self, env):
        body = env.body
        head, food, tail, neck = body[0], env.food, body[-1], body[1]
        body_set = set(body)
        L = len(body)

        # 候选邻格（排除颈；len==2 时颈在吃食时也不腾出，同样排除）
        cand = []
        for di, (dr, dc) in enumerate(DIRS):
            nbr = (head[0] + dr, head[1] + dc)
            if nbr == neck:
                continue
            if self.reachable(nbr, body_set, food, tail):
                cand.append((di, nbr))
        if not cand:
            return 0

        def act_to_cell(cell):
            dr, dc = cell[0] - head[0], cell[1] - head[1]
            return absdir_to_action(DIR_IDX[(dr, dc)], env.dir)

        # ---- 1. 哈密顿模式：严格沿索引递减 ----
        if self.hamilton_index is not None:
            hi = self.hamilton_index
            hv = hi[head[0]][head[1]]
            succ_val = hv - 1 if hv > 1 else None
            succ = None
            if succ_val is not None:
                for di, cell in cand:
                    if hi[cell[0]][cell[1]] == succ_val:
                        succ = cell
                        break
            if succ is None:
                self.reset()          # 索引耗尽（到达尾端），重建
            else:
                return act_to_cell(succ)

        # ---- 2. 蛇够长：现算哈密顿路径（头→尾覆盖全部可达格）----
        if L >= self.HAMILTON_THRESHOLD:
            path = self._hamilton_path(head, tail, body_set, food)
            if path:
                idx = [[0] * G for _ in range(G)]
                val = G * G
                for cell in [head] + path:
                    idx[cell[0]][cell[1]] = val
                    val -= 1
                idx[tail[0]][tail[1]] = 1
                self.hamilton_index = idx
                succ_val = idx[head[0]][head[1]] - 1
                for di, cell in cand:
                    if idx[cell[0]][cell[1]] == succ_val:
                        return act_to_cell(cell)
                # 路径建好但后继不在邻格（理论不可能）——落入下方启发式

        # ---- 3. 短路去食物 + 前向检查（模拟吃完后仍能到尾）----
        path = self.bfs_cells(head, food, body_set, food, tail)
        if path:
            nxt = path[0]
            sim = self._simulate_eat(body, food, path)
            if sim is not None:
                sb, sfood = sim
                back = self.bfs_cells(sfood, sb[-1], set(sb), sfood, sb[-1])
                if back is not None:
                    return act_to_cell(nxt)

        # ---- 4. 绕尾（较长路径等尾腾位）----
        first_dir = self._longer_path_first(tail, body, food, tail)
        if first_dir is not None:
            return absdir_to_action(first_dir, env.dir)

        # ---- 5. 远离食物苟活 ----
        _, far = max(cand, key=lambda t: abs(t[1][0] - food[0]) + abs(t[1][1] - food[1]))
        return act_to_cell(far)

    def _simulate_eat(self, body, food, cells):
        """沿 cells 走到食物（吃下，尾不缩），返回 (new_body, new_food)。"""
        b = list(body)
        for cell in cells:
            b.insert(0, cell)
            if cell == food:
                return b, food
            if len(b) > 1 and cell != food:
                b.pop()
        return None

    def _longer_path_first(self, dst, body, food, tail):
        """chynl longer_path 的首步绝对方向索引：沿最短路每步尝试侧绕一格。"""
        body_set = set(body)
        shortest = self.bfs_cells(body[0], dst, body_set, food, tail)
        if not shortest:
            return None
        cur = body[0]
        second_tail = body[-2] if len(body) >= 2 else None   # 一步后成为新尾
        for step_cell in shortest:
            step_dir = (step_cell[0] - cur[0], step_cell[1] - cur[1])
            sd = DIR_IDX[step_dir]
            nxt = step_cell
            perp = [1, 3] if sd in (0, 2) else [0, 2]
            for pd in perp:
                ed = DIRS[pd]
                c_ext = (cur[0] + ed[0], cur[1] + ed[1])
                n_ext = (nxt[0] + ed[0], nxt[1] + ed[1])
                c_ok = self.reachable(c_ext, body_set, food, tail) \
                    and c_ext != food
                n_ok = self.reachable(n_ext, body_set, food, tail) \
                    or n_ext == second_tail
                if c_ok and n_ok:
                    return pd          # 先侧绕（此步动作）
            cur = nxt
        return DIR_IDX[(shortest[0][0] - body[0][0], shortest[0][1] - body[0][1])]

    def _hamilton_path(self, head, tail, body_set, food):
        # 可达格数（不含头；含尾——终点）
        seen = {head}
        q = [head]
        while q:
            cur = q.pop()
            for nbr in self.neighbors(cur, body_set, food, tail, seen):
                seen.add(nbr)
                q.append(nbr)
        target = len(seen) - 1          # 步数（头除外）
        visited = {head}
        path = []
        self.hb_count = 0
        if self._hb(head, tail, body_set, food, visited, path, target):
            return path
        return None

    def _hb(self, cur, dst, body_set, food, visited, path, target):
        if len(path) == target:
            return cur == dst
        if self.hb_count >= self.HAMILTON_SEARCH_LIMIT:
            return False
        self.hb_count += 1
        nbrs = self.neighbors(cur, body_set, food, dst, visited)
        nbrs.sort(key=lambda c: len(self.neighbors(c, body_set, food, dst, visited)))
        for nbr in nbrs:
            if nbr == dst and len(path) < target - 1:
                continue
            visited.add(nbr)
            path.append(nbr)
            if self._hb(nbr, dst, body_set, food, visited, path, target):
                return True
            path.pop()
            visited.remove(nbr)
        return False


# ---------- 批量参考局 ----------
def rollout_reference(banks, starve_slope=1e9, max_steps=8000, seed0=0,
                      verbose=False, solver_cls=CycleSolver):
    import time
    logs = []
    for i, bank in enumerate(banks):
        t0 = time.time()
        env = MiniEnv(bank, starve_slope=starve_slope, max_steps=max_steps)
        init_food = env.food
        solver = solver_cls() if solver_cls is CycleSolver \
            else solver_cls(seed=seed0 + i)
        t = 0
        while env.alive and t < max_steps:
            a = solver.next_action(env)
            env.step(a)
            t += 1
        logs.append(dict(acts=env.acts, events=env.events, path=env.path,
                         snaps=env.snaps, died=env.died, death_ctx=env.death_ctx,
                         won=env.won, init_food=init_food))
        if verbose:
            print(f'    bank{i}: food={len(env.events)} died={env.died} '
                  f'steps={t} ({time.time() - t0:.1f}s)', flush=True)
    return logs


# ---------- 校验：mini-env 与 GPU env 同库同个体逐位对齐 ----------
def verify_mini_env(cfg, pop, banks, n=8):
    from test7h import BatchedSnakeEnv, deliberate_batch, update_fatigue
    dev = torch.device('cpu')
    cfg = _cpu_cfg(cfg)
    sub = pop[list(range(n))]
    sub.refresh_eff()
    env = BatchedSnakeEnv(cfg, n, dev)
    bank = {'stream': torch.stack([b['stream'] for b in banks[:n]]),
            'dir0': torch.tensor([b['dir0'] for b in banks[:n]], dtype=torch.long)}
    env.reset(bank=bank)
    N = pop.N
    E = torch.zeros(n, N); I = torch.zeros(n, N); st = torch.zeros(n, N)
    press = torch.zeros(n)
    events_gpu = [[] for _ in range(n)]
    died_gpu = [0] * n
    for t in range(cfg.MAX_STEPS):
        if not bool(env.alive.any()):
            break
        obs = env.obs()
        act, E, I, st = deliberate_batch(sub, obs, E, I, st, press, cfg)
        press = update_fatigue(press, act, decay=float(cfg.FATIGUE_TURN_DECAY))
        al = env.alive.clone()
        env.step(act)
        ate = (al & env.ate).numpy()
        for i in range(n):
            if ate[i]:
                events_gpu[i].append(t + 1)
        d = env.died.numpy()
        for i in range(n):
            if d[i] and died_gpu[i] == 0:
                died_gpu[i] = int(d[i])
    ok = True
    for i in range(n):
        menv = MiniEnv(banks[i], starve_slope=float(cfg.STARVE_SLOPE),
                       max_steps=cfg.MAX_STEPS)
        sub1 = pop[[i]]
        sub1.refresh_eff()
        E1 = torch.zeros(1, N); I1 = torch.zeros(1, N); st1 = torch.zeros(1, N)
        p1 = torch.zeros(1)
        e1 = BatchedSnakeEnv(cfg, 1, dev)
        e1.reset(bank={'stream': banks[i]['stream'], 'dir0': banks[i]['dir0']})
        while bool(e1.alive.any()) and menv.alive:
            obs = e1.obs()
            act, E1, I1, st1 = deliberate_batch(sub1, obs, E1, I1, st1, p1, cfg)
            p1 = update_fatigue(p1, act, decay=float(cfg.FATIGUE_TURN_DECAY))
            a = int(act[0])
            e1.step(torch.tensor([a]))
            menv.step(a)
        same = menv.events == events_gpu[i] and menv.died == died_gpu[i]
        if not same:
            ok = False
            print(f'  [不匹配] 个体{i}: mini {len(menv.events)}食/died{menv.died} '
                  f'vs gpu {len(events_gpu[i])}食/died{died_gpu[i]}')
    print(f'[校验] mini-env vs GPU env：{"逐位一致" if ok else "存在偏差"}（n={n}）')
    return ok


def _cpu_cfg(cfg):
    c = copy.copy(cfg)
    c.DEVICE = 'cpu'
    c.USE_FP16 = False
    return c


if __name__ == '__main__':
    cfg = t7h.Config()
    banks = t7h.make_banks(cfg, 555, 1, 5, torch.device('cpu'))
    torch.manual_seed(3)
    pop = t7h.GeneStack(cfg, B=8, device=torch.device('cpu'))
    pop.random_init()
    verify_mini_env(cfg, pop, banks, n=8)
    logs = rollout_reference(banks)
    for i, lg in enumerate(logs):
        print(f'  ref bank{i}: food={len(lg["events"])} died={lg["died"]}')
