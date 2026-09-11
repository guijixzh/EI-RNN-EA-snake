"""obs40 新通道语义探针（可视化问题归因，2026-09-06）：

1. [33:37] 尾相对方位：4 朝向 × 8 自我系方位（前/右前/右/.../左前）受控摆放，
   打印激活通道 vs 几何预期 —— 验证是否为正确的自我系（ego）分解。
2. [37:40] 洪水稀缺：空旷基线 / 前向堵死 / 墙角不对称三场景，
   同时给出原始可达数 cnt 与空盘参考 cap（复刻 _flood_scarcity 内部量），
   检查"满空间仍有输出"与"左右空间差大但数值差小"两个观察。

用法：python experiments/probe_obs40_semantics.py
"""
import importlib.util, os, sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location('t16b', os.path.join(ROOT, 'experiments', 'test16_series', 'test16b.py'))
t16b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t16b)

cfg = t16b.Config()
cfg.DEVICE = 'cpu'
cfg.USE_FP16 = False
dev = torch.device('cpu')
env = t16b.BatchedSnakeEnv(cfg, 1, dev)
DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]          # 0右 1下 2左 3上（dr,dc）
DIR_NAMES = ['右(东)', '下(南)', '左(西)', '上(北)']
FRONT, RIGHT, BACK, LEFT = '前', '右', '后', '左'


def ego_vec(dir_idx, bearing):
    """朝向 dir_idx 下的自我系方位单位向量（dr,dc）：前=d, 右=d+1, 后=-d, 左=d+3。"""
    d = DIRS[dir_idx]
    if bearing == FRONT:   return d
    if bearing == RIGHT:   return DIRS[(dir_idx + 1) % 4]
    if bearing == BACK:    return (-d[0], -d[1])
    if bearing == LEFT:    return DIRS[(dir_idx + 3) % 4]
    raise ValueError(bearing)


def set_state(head, dir_idx, tail, extra_body=()):
    """手动构造 env 状态：head=(r,c)，tail=(r,c)，extra_body 额外身段（靠近头一侧在前）。"""
    env.reset(bank=None)
    env.head[0, 0], env.head[0, 1] = head
    env.dir_idx[0] = dir_idx
    body = [head, tail] + list(extra_body)
    for i, (r, c) in enumerate(body):
        env.body[0, i, 0], env.body[0, i, 1] = r, c
    env.body_len[0] = len(body)
    env.food[0, 0], env.food[0, 1] = 0, 0     # 与尾块无关


# ============ 1. 尾相对方位 [33:37] ============
print('=== [33:37] 尾相对方位：朝向 × 8 自我系方位（距离 2）===')
print('通道顺序 = [前, 右, 后, 左]；预期 = 尾在自我系 b 方位时，b 的两个分量通道按投影比例激活\n')
BEARINGS = [FRONT, '右前', RIGHT, '右后', BACK, '左后', LEFT, '左前']
mismatches = 0
for dir_idx in range(4):
    print(f'--- 朝向 {DIR_NAMES[dir_idx]} (d={DIRS[dir_idx]}) ---')
    for b in BEARINGS:
        if len(b) == 1:
            v = ego_vec(dir_idx, b)
            tail = (5 + 2 * v[0], 5 + 2 * v[1])
        else:
            v1, v2 = ego_vec(dir_idx, b[0]), ego_vec(dir_idx, b[1])
            tail = (5 + v1[0] + v2[0], 5 + v1[1] + v2[1])
        set_state((5, 5), dir_idx, tail)
        obs = env._obs40_fast()[0]
        act = obs[33:37].tolist()
        # 预期：主方位通道=1.0（对角=两通道各 0.5）
        ok = True
        if len(b) == 1:
            expect = {FRONT: (1, 0, 0, 0), RIGHT: (0, 1, 0, 0),
                      BACK: (0, 0, 1, 0), LEFT: (0, 0, 0, 1)}[b]
            ok = all(abs(a - e) < 1e-5 for a, e in zip([x / 8 for x in act], expect))
        else:
            base = b[0]      # '右前'→右, '左前'→左
            exp_pair = sorted([base, b[1]])
            got = sorted([[FRONT, RIGHT, BACK, LEFT][i] for i, x in enumerate(act) if x > 0.1])
            ok = got == exp_pair and abs(sum(x / 8 for x in act) - 1.0) < 1e-4
        if not ok:
            mismatches += 1
        print(f'  尾在{b}(自我系): 前={act[0]/8:.2f} 右={act[1]/8:.2f} '
              f'后={act[2]/8:.2f} 左={act[3]/8:.2f}  {"✓" if ok else "✗ 预期不符"}')
print(f'\n尾方位 32 例中不符 {mismatches} 例' + ('——语义=正确的自我系投影' if mismatches == 0 else ''))

# ============ 2. 洪水稀缺 [37:40] ============
def flood_internals(env):
    """复刻 _flood_scarcity 内部：返回 [(cnt, cap, scarcity)]×3 前/左/右。"""
    import torch.nn.functional as F
    B, G = 1, env.G
    depth = int(getattr(cfg, 'FLOOD_DEPTH', 7))
    d = env.DIRS[env.dir_idx]
    left = env.DIRS[(env.dir_idx + 3) % 4]
    right = env.DIRS[(env.dir_idx + 1) % 4]
    seeds = torch.stack((d, left, right), dim=1)
    occ = env._occupancy_flat(tail_invalid=not bool(getattr(cfg, 'FLOOD_TAIL_BLOCK', True)))
    occ = (occ > 0.5).view(B, 1, G, G)
    free = (~occ).to(torch.float32)
    nb = env.head.unsqueeze(1) + seeds
    inb = ((nb[..., 0] >= 0) & (nb[..., 0] < G) & (nb[..., 1] >= 0) & (nb[..., 1] < G))
    sr, sc = nb[..., 0].clamp(0, G - 1), nb[..., 1].clamp(0, G - 1)
    ar = torch.arange(B)
    valid = inb & (~occ[:, 0][ar.view(B, 1), sr, sc])
    seed = torch.zeros(B, 3, G, G)
    seed[ar.view(B, 1), torch.arange(3).view(1, 3), sr, sc] = valid.float()
    reach = seed
    for _ in range(depth):
        cross = torch.maximum(F.max_pool2d(reach, (3, 1), stride=1, padding=(1, 0)),
                              F.max_pool2d(reach, (1, 3), stride=1, padding=(0, 1)))
        reach = cross * free
    cnt = (reach > 0).float().sum(dim=(2, 3))
    cap = env.empty_reach[(sr * G + sc).clamp(0, G * G - 1)]
    scar = (1.0 - cnt / cap.clamp(min=1.0)).clamp(0.0, 1.0)
    scar = torch.where(valid, scar, torch.ones_like(scar))
    return [(int(cnt[0, i]), int(cap[0, i]), float(scar[0, i])) for i in range(3)]


def scene(tag, head, dir_idx, tail, extra=()):
    set_state(head, dir_idx, tail, extra_body=extra)
    obs = env._obs40_fast()[0]
    ins = flood_internals(env)
    print(f'{tag}')
    print(f'  洪前: 稀缺={obs[37]/8:.3f} (cnt={ins[0][0]}/cap={ins[0][1]})   '
          f'洪左: 稀缺={obs[38]/8:.3f} (cnt={ins[1][0]}/cap={ins[1][1]})   '
          f'洪右: 稀缺={obs[39]/8:.3f} (cnt={ins[2][0]}/cap={ins[2][1]})\n')


print('\n=== [37:40] 洪水稀缺（depth=7；cnt=实际可达数, cap=同位置空盘参考）===')
scene('A 空旷基线：中央短蛇朝东（棋盘无其他阻挡）', (5, 5), 0, (5, 3))
scene('B 前向堵死：前方一格被身段占据', (5, 5), 0, (5, 4), extra=[(5, 6), (4, 6)])
scene('C 墙角不对称：头贴上边墙朝东（左=北贴墙, 右=南开阔）', (0, 3), 0, (0, 1))
scene('D 中盘不对称：右侧被长身段半围（左开阔）', (5, 5), 0, (5, 3),
      extra=[(6, 5), (6, 6), (6, 7), (7, 7), (7, 6), (7, 5)])
