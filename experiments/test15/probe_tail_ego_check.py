"""尾相对方位 [33:37] 语义核查（独立物理定义版 + 真实对局逐帧核对）。

与上一版探针的区别：期望值不再从代码 DIRS 推导（避免循环论证），而是用
叉积物理定义"蛇的左手边"：heading d=(dr,dc) 时，physical_left(d)=(-dc,dr)
（屏幕坐标系 row 向下；用真实世界校验过：朝北上=屏幕左，朝南左=屏幕右…）。

两部分：
  A. 合成用例：4 朝向 × 8 方位受控摆放，打印 激活 vs 物理期望 全表。
  B. 真实对局：从可视化服务器 SSE 抓帧，逐帧用帧内的 head/dir/tail 几何
     独立计算期望通道，与帧内 obs[33:37] 比对，打印明细样例。
"""
import importlib.util, json, os, sys, time, urllib.request
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

# DIRS: 0右(东,0,1) 1下(南,1,0) 2左(西,0,-1) 3上(北,-1,0)；row 向下增长
DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
DIR_CN = ['朝东(屏幕右)', '朝南(屏幕下)', '朝西(屏幕左)', '朝北(屏幕上)']
CH = ['前', '右', '后', '左']


def physical_left(d):
    """叉积物理定义：蛇的左手边。已用真实世界校验：朝北→左=西(屏幕左)；
    朝南→左=东(屏幕右)；朝东→左=北(屏幕上)；朝西→左=南(屏幕下)。"""
    return (-d[1], d[0])


def expected_tail_channels(delta_r, delta_c, d):
    """给定尾-头位移与朝向，返回物理期望 (前,右,后,左) 各通道占比。
    代码式: sig = max(v·a, 0) / max(|Δr|+|Δc|, 1)，显示值 = sig×8。"""
    dist = max(abs(delta_r) + abs(delta_c), 1)
    f = max(delta_r * d[0] + delta_c * d[1], 0) / dist
    b = max(-(delta_r * d[0] + delta_c * d[1]), 0) / dist
    l = max(delta_r * physical_left(d)[0] + delta_c * physical_left(d)[1], 0) / dist
    r = max(delta_r * (d[1], -d[0])[0] + delta_c * (d[1], -d[0])[1], 0) / dist
    return (f * 8, r * 8, b * 8, l * 8)


def screen_bearing(delta_r, delta_c):
    """尾相对头的屏幕方位（北=上，即 -r 方向）。"""
    import math
    names = ['北', '东北', '东', '东南', '南', '西南', '西', '西北']
    # 屏幕罗盘角：北=0，东=90（row 向下 → 北=−r，东=+c）
    a = math.degrees(math.atan2(delta_c, -delta_r)) % 360
    return names[int((a + 22.5) // 45) % 8]


def set_state(head, dir_idx, tail):
    env.reset(bank=None)
    env.head[0, 0], env.head[0, 1] = head
    env.dir_idx[0] = dir_idx
    for i, (r, c) in enumerate([head, tail]):
        env.body[0, i, 0], env.body[0, i, 1] = r, c
    env.body_len[0] = 2
    env.food[0, 0], env.food[0, 1] = 0, 0


# ============ A. 合成用例全表（期望=物理叉积定义，非代码推导） ============
print('=' * 96)
print('A. 合成用例：4 朝向 × 8 方位，尾距头 2 格。期望列=物理叉积定义独立计算')
print('   通道顺序 = [前, 右, 后, 左]，表值为 ×8 后的显示值')
print('=' * 96)
BEARINGS = {'前': 0, '右前': 1, '右': 2, '右后': 3, '后': 4, '左后': 5, '左': 6, '左前': 7}
EGO8 = ['前', '右前', '右', '右后', '后', '左后', '左', '左前']
n_bad = 0
for dir_idx in range(4):
    d = DIRS[dir_idx]
    lft = physical_left(d)
    print(f'\n--- 蛇{DIR_CN[dir_idx]}  d={d}  蛇的左手边={lft} ---')
    print(f"{'尾在(自我系)':<10}{'尾-头位移(Δr,Δc)':<16}{'屏幕方位':<8}"
          f"{'期望 前/右/后/左':<24}{'实际 前/右/后/左':<24}判定")
    for name in EGO8:
        k = BEARINGS[name]
        ang = k * 3.14159265 / 4
        # 自我系方位向量：前=d，右手系（往右转为正）……直接用物理基构造
        rgt = (d[1], -d[0])            # 蛇的右手边 = -left
        v = (d[0] * (1 if k in (0, 1, 7) else 0), 0)  # 占位，下面精确算
        # 8 方位单位位移 = d 的 cos 分量 + right 的 sin 分量（k=0 前, 2 右, 4 后, 6 左）
        import math
        cd, sd = math.cos(ang), math.sin(ang)
        delta_r = d[0] * cd + rgt[0] * sd
        delta_c = d[1] * cd + rgt[1] * sd
        # 取整到 ±2 格（对角=1+1）
        dr = int(round(delta_r * 2)) if abs(delta_r) > 1e-9 else 0
        dc = int(round(delta_c * 2)) if abs(delta_c) > 1e-9 else 0
        set_state((5, 5), dir_idx, (5 + dr, 5 + dc))
        obs = env._obs40_fast()[0]
        act = [obs[33 + i].item() for i in range(4)]
        exp = list(expected_tail_channels(dr, dc, d))
        ok = all(abs(a - e) < 1e-4 for a, e in zip(act, exp))
        n_bad += (not ok)
        print(f"{name:<12}({dr:+d},{dc:+d}){'':<8}{screen_bearing(dr, dc):<6}"
              f"{'/'.join(f'{x:5.2f}' for x in exp):<26}"
              f"{'/'.join(f'{x:5.2f}' for x in act):<26}{'✓' if ok else '✗'}")
print(f'\nA 部结论：32 例中物理期望不符 {n_bad} 例')

# ============ B. 真实对局逐帧核对 ============
print('\n' + '=' * 96)
print('B. 真实对局核对：从可视化服务器抓 10 秒 SSE 帧，逐帧独立验证（含截图同类场景）')
print('=' * 96)
url = 'http://127.0.0.1:8765/stream'
frames = []
try:
    req = urllib.request.urlopen(url, timeout=30)
    t0 = time.time()
    buf = b''
    while time.time() - t0 < 12:
        chunk = req.read(65536)
        if not chunk:
            break
        buf += chunk
        while b'\n\n' in buf:
            line, buf = buf.split(b'\n\n', 1)
            if line.startswith(b'data: '):
                try:
                    d = json.loads(line[6:])
                    if d.get('type') == 'frame':
                        frames.append(d)
                except Exception:
                    pass
except Exception as e:
    print('（服务器未运行，跳过 B 部分：', e, '）')

if frames:
    n_ok = n_tot = 0
    bad = []
    shown = 0
    print(f'抓到 {len(frames)} 帧（约 12 秒），逐帧核对：')
    print(f"{'步':<5}{'蛇头(朝向)':<16}{'尾位置':<10}{'尾-头位移':<12}{'屏幕方位':<8}"
          f"{'期望 前/右/后/左':<24}{'实际 前/右/后/左':<24}判定")
    for fr in frames:
        body = fr['body']
        if len(body) < 2 or len(fr['obs']) < 40:
            continue
        head, tail = body[0], body[-1]
        dr_, dc_ = tail[0] - head[0], tail[1] - head[1]
        d = tuple(fr['dir'])
        exp = list(expected_tail_channels(dr_, dc_, d))
        act = [fr['obs'][33 + i] for i in range(4)]
        ok = all(abs(a - e) < 2e-3 for a, e in zip(act, exp))
        n_tot += 1
        n_ok += ok
        if not ok:
            bad.append((fr['steps'], d, (dr_, dc_), exp, act))
        # 打印样例：前 8 帧 + 朝南且有右侧分量的帧（用户截图场景）多打 2 帧
        interesting = (d == (1, 0) and abs(act[1]) > 0.5)
        if shown < 8 or (interesting and shown < 12):
            dn = {(0, 1): '东', (1, 0): '南', (0, -1): '西', (-1, 0): '北'}.get(d, str(d))
            print(f"{fr['steps']:<6}{f'({head[0]},{head[1]}){dn}':<16}"
                  f"{f'({tail[0]},{tail[1]})':<10}{f'({dr_:+d},{dc_:+d})':<12}"
                  f"{screen_bearing(dr_, dc_):<8}"
                  f"{'/'.join(f'{x:5.2f}' for x in exp):<26}"
                  f"{'/'.join(f'{x:5.2f}' for x in act):<26}{'✓' if ok else '✗'}")
            shown += 1
    print(f'\nB 部结论：{n_tot} 帧中与物理期望不符 {n_tot - n_ok} 帧')
    if bad:
        for b_ in bad[:5]:
            print('  不符帧:', b_)
    print('\n自我系左/右对照表（关键：蛇朝下时，蛇的右手边=屏幕左侧）：')
    print('  蛇朝北(屏幕上)：自我左=屏幕左  自我右=屏幕右   ← 与屏幕直觉一致')
    print('  蛇朝南(屏幕下)：自我左=屏幕右  自我右=屏幕左   ← 镜像！你截图正是此朝向')
    print('  蛇朝东(屏幕右)：自我左=屏幕上  自我右=屏幕下')
    print('  蛇朝西(屏幕左)：自我左=屏幕下  自我右=屏幕上')
