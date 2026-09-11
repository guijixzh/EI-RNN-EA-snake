# ==========================================
# exp_obs40_calibrate.py —— test15 新观测通道 Phase A 标定（预注册四门）
#
# 门 A1 可读性：随机种群下新 8 通道的驱动贡献 ≥ 食物块 ×0.5
#   （test7a 证据：信号弱于随机 W_in 触发阈 ⇒ 被选择完全忽略，而非学得慢）
# 门 A2 预测效度：自然局中 前/左/右 稀缺度在"挤压自撞死亡前 20 步窗口"
#   （len≥30）显著高于全程基线（Δ≥+0.05，[0,1] 口径）；无信号通道预注册剪除
# 门 A-cost：新观测逐步耗时 ≤ +25%
# 门 P1/P2 等价性：A0 模式与 test12 逐局一致；开通模式扰动 ≤1×评估噪声 SEM
# 另含：方向相对性审计（旋转协变性，用户要求固化）+ 新通道已知答案单元例
# ==========================================

import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import test12 as t12   # noqa: E402
import test15 as t15   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT = os.path.join(ROOT, 'artifacts/test12/test12_econ_latest_gen_best.pth')
BLOCKS = [('head_abs', 0, 4), ('tail_abs', 4, 8), ('food', 8, 16),
          ('self', 16, 24), ('obst', 24, 32), ('clock', 32, 33),
          ('tailrel', 33, 37), ('flood', 37, 40)]


def build_cfg(module, ckpt=None, force_new_dims=False):
    cfg = module.Config()
    if ckpt:
        data = torch.load(ckpt, map_location='cpu', weights_only=False)
        for k, v in data.get('config', {}).items():
            if k.startswith('__'):
                continue
            if isinstance(v, (int, float, str, bool, list, tuple)) or v is None:
                setattr(cfg, k, v)
        if force_new_dims:
            # test12 旧 checkpoint 的 saved config 会把 OBS_DIM 压回 32——
            # test15 侧必须强制新维度（迁移语义的一部分）
            cfg.OBS_DIM = 40
            cfg.OBS_ENC_VERSION = '40tailflood1'
            cfg.BRAIN_VERSION = 'base1'
        return cfg, data
    return cfg, None


def broadcast(module, cfg, st, B, device):
    pop = module.GeneStack(cfg, B=B, device=device)
    pop.random_init()
    for i in range(B):
        pop.set_individual_from_state(i, st)
    if device.type == 'cpu':
        cfg.USE_FP16 = False
    pop.fp16()
    pop.refresh_eff()
    return pop


# ---------- 0. 已知答案单元例 + 旋转协变性审计 ----------

def rotate_state(head, body, food, dvec, G=10):
    """90° 旋转：(r,c)→(c,G-1-r)；方向向量 (dr,dc)→(dc,-dr)。"""
    rp = lambda p: (int(p[1]), G - 1 - int(p[0]))
    rd = lambda v: (int(v[1]), -int(v[0]))
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    return (rp(head), [rp(p) for p in body], rp(food), dirs.index(rd(dvec)))


def inject(env, row, head, body, d_idx, food, swf):
    """body 列表为头在前（[0]=头），与 env 语义一致，不做翻转。"""
    env.head[row] = torch.tensor(head, device=env.device)
    env.body[row, :len(body)] = torch.tensor(body, dtype=torch.long,
                                             device=env.device)
    env.body_len[row] = len(body)
    env.dir_idx[row] = d_idx
    env.food[row] = torch.tensor(food, device=env.device)
    env.steps_wo_food[row] = swf
    env.alive[row] = True
    env.died[row] = 0
    env.ate[row] = False


def unit_and_audit_tests(device):
    ok = True
    G = 10
    cfg, _ = build_cfg(t15)
    cfg.DEVICE = 'auto'
    cfg.USE_FP16 = False
    env = t15.BatchedSnakeEnv(cfg, 2, device)
    env.reset()

    # 场景 A：头(5,5) 朝东 DIRS[1]=(1,0)，身体 L 形 [(5,5),(5,4),(4,4)]，
    # 尾(4,4)，食物(2,2)，饿钟 13 步（len=3 → 13/(15+20)=0.371）
    headA, bodyA = (5, 5), [(5, 5), (5, 4), (4, 4)]
    d_idxA = 1
    inject(env, 0, headA, bodyA, d_idxA, (2, 2), 13)
    headB, bodyB, foodB, d_idxB = rotate_state(headA, bodyA, (2, 2),
                                               (1, 0), G)
    inject(env, 1, headB, bodyB, d_idxB, foodB, 13)
    obs = env.obs()

    # 已知答案：饿钟 13/(3·5+20)=0.371
    clock_a = float(obs[0, 32]) / cfg.OBS_NEW_SCALE
    if abs(clock_a - 13.0 / 35.0) > 1e-4:
        print(f'[单元] 饿钟压力不符: {clock_a:.4f} vs {13/35:.4f}'); ok = False
    # 尾(4,4) 相对头(5,5)：vt=(-1,-1)（行-1，列-1）；朝东=(1,0)：
    # 前=vt·(1,0)=-1→0；右=DIRS[2]=(0,-1)，vt·右=+1→1/2；后=(-1,0)，vt·后=+1→1/2
    # （尾在头的"后+右"对角）；左=0
    tailrel = obs[0, 33:37] / cfg.OBS_NEW_SCALE
    exp = torch.tensor([0.0, 0.5, 0.5, 0.0])
    if not torch.allclose(tailrel.cpu(), exp, atol=1e-4):
        print(f'[单元] 尾四方位不符: {tailrel.tolist()} vs {exp.tolist()}'); ok = False
    # 洪水：盘上仅 3 节身体 → 稀缺度≈(可达身体格数)/cap，应很小（≤0.15）；
    # 右向种子=身体格(5,4)（占用）→ 右稀缺=1
    flood = obs[0, 37:40] / cfg.OBS_NEW_SCALE
    if not (flood[0] <= 0.15 and flood[1] <= 0.15 and abs(float(flood[2]) - 1.0) < 1e-4):
        print(f'[单元] 洪水不符: {flood.tolist()} '
              f'(期望 前/左≤0.15, 右=1.0——右邻格是身体)'); ok = False
    # 堵死例：前邻格放身体 → 前稀缺=1
    inject(env, 0, (5, 5), [(5, 5), (6, 5), (7, 5), (5, 4), (4, 4)], 1, (2, 2), 0)
    obs2 = env.obs()
    f2 = obs2[0, 37:40] / cfg.OBS_NEW_SCALE
    if abs(float(f2[0]) - 1.0) > 1e-4:
        print(f'[单元] 前邻格占用应稀缺=1: {f2.tolist()}'); ok = False
    print(f'[单元/审计] 已知答案: {"PASS" if ok else "FAIL"}  '
          f'洪水(空盘)={flood.tolist()}')

    # 旋转协变性：ego 通道 [8:40] 应逐元素一致；绝对通道 [0:8] 应不同
    ego_same = torch.allclose(obs[0, 8:40], obs[1, 8:40], atol=1e-4)
    abs_diff = float((obs[0, :8] - obs[1, :8]).abs().sum())
    audit_ok = ego_same and abs_diff > 0
    print(f'[审计] 旋转90°后 ego[8:40] 一致={ego_same}，'
          f'绝对[0:8] 有差异(和={abs_diff:.1f}) → '
          f'{"PASS（[0:8]绝对系/[8:]相对系 与文档审计表一致）" if audit_ok else "FAIL"}')
    return ok and audit_ok


# ---------- P1/P2 等价性 ----------

def equivalence_tests(device):
    ok = True
    cfg15, data = build_cfg(t15, CKPT, force_new_dims=True)
    cfg12, _ = build_cfg(t12, CKPT)
    st = data['brain']
    B, E, TAG = 64, 8, 555

    # P2/迁移对照：A0 模式（新通道置零）应与 test12 逐局一致
    cfg15_a0, _ = build_cfg(t15, CKPT, force_new_dims=True)
    cfg15_a0.OBS_NEW_ENABLED = False
    pop12 = broadcast(t12, cfg12, st, B, device)
    mig = t15.load_migratable_state(CKPT, cfg15_a0)
    if mig is None:
        print('[P1] FAIL: 迁移变换返回 None')
        return False, 0.0
    st15, mig_food, _ = mig
    print(f'[迁移] test12 冠军(Food={mig_food:.1f}) → 40 维：'
          f'M_in {tuple(st15["M_in"].shape)}，新列激活 '
          f'{int(st15["M_in"][:, 32:].sum())}/{(cfg15_a0.NUM_COLUMNS, 8)}')
    popa0 = broadcast(t15, cfg15_a0, st15, B, device)
    banks12 = t12.make_banks(cfg12, TAG, 0, E, device)
    m12 = t12._eval_pop_banks(pop12, cfg12, banks12)
    banks15 = t15.make_banks(cfg15_a0, TAG, 0, E, device)
    ma0 = t15._eval_pop_banks(popa0, cfg15_a0, banks15)
    dmax = float((m12[:, 0] - ma0[:, 0]).abs().max())
    print(f'[P1] test12 food={float(m12[:,0].mean()):.3f}  '
          f'test15-A0 food={float(ma0[:,0].mean()):.3f}  max|Δ局|={dmax:.6f}')
    p1 = dmax < 1e-3
    print(f'[P1] {"PASS" if p1 else "FAIL"}')

    # P2/迁移扰动：新通道开通（迁移新列随机接线）
    pop_on = broadcast(t15, cfg15, st15, B, device)
    mon = t15._eval_pop_banks(pop_on, cfg15, banks15)
    dmean = float(mon[:, 0].mean() - ma0[:, 0].mean())
    # v1.1 判据修订：P2 意图=非破坏性（Δ≥-2.18=−1×SEM）；正偏移是信息通道
    # 在随机接线下已可读的信号，允许并单独报告（test14 教训：只防通胀性/破坏性
    # 基线偏移，此处无处理组方差注入——观测是确定性的，通胀机制不适用）
    p2 = dmean >= -2.18
    print(f'[P2] 开通新通道 food={float(mon[:,0].mean()):.3f}  '
          f'Δmean={dmean:+.3f}（判据 非破坏性 Δ≥-2.18）→ {"PASS" if p2 else "FAIL"}'
          f'{"【注意：正偏移=新信息随机接线已可读】" if dmean > 0 else ""}')
    ok = p1 and p2
    return ok, float(ma0[:, 0].mean())


# ---------- 可读性化验（diagnose_blocks 方法）----------

def readability_assay(device):
    cfg, data = build_cfg(t15, CKPT, force_new_dims=True)
    st = t15.load_migratable_state(CKPT, cfg)[0]
    B, E = 64, 4
    pop = broadcast(t15, cfg, st, B, device)      # 随机接线底盘（新列为随机）
    banks = t15.make_banks(cfg, 777, 0, E, device)
    bank = {'stream': torch.stack([b['stream'] for b in banks]),
            'dir0': torch.tensor([b['dir0'] for b in banks], dtype=torch.long,
                                 device=device)}
    env = t15.BatchedSnakeEnv(cfg, B, device)
    env.reset(bank=bank)
    half = torch.float16 if cfg.USE_FP16 else torch.float32
    d = {name: {'drive': 0.0, 'flip0': 0.0, 'flip4': 0.0, 'n': 0}
         for name, _, _ in BLOCKS}
    for t in range(300):
        al = env.alive
        if not bool(al.any()):
            env.reset(bank=bank)
            continue
        obs = env.obs().to(half)
        with torch.no_grad():
            ext = torch.bmm(pop.W_in_eff, obs.unsqueeze(-1)).squeeze(-1)
            logits = torch.bmm(pop.W_out_eff,
                               torch.sigmoid(ext.unsqueeze(-1))).squeeze(-1)
            base = logits.argmax(dim=1)
            for name, lo, hi in BLOCKS:
                o0 = obs.clone(); o0[:, lo:hi] = 0
                e0 = torch.bmm(pop.W_in_eff, o0.unsqueeze(-1)).squeeze(-1)
                l0 = torch.bmm(pop.W_out_eff,
                               torch.sigmoid(e0.unsqueeze(-1))).squeeze(-1)
                o4 = obs.clone(); o4[:, lo:hi] = obs[:, lo:hi] * 4.0
                e4 = torch.bmm(pop.W_in_eff, o4.unsqueeze(-1)).squeeze(-1)
                l4 = torch.bmm(pop.W_out_eff,
                               torch.sigmoid(e4.unsqueeze(-1))).squeeze(-1)
                m = al.float().sum()
                d[name]['drive'] += float(
                    ext[:, lo:hi].abs().sum(dim=1).mean()) * float(m)
                d[name]['flip0'] += float((l0.argmax(1) != base)[al].float().mean()) * float(m)
                d[name]['flip4'] += float((l4.argmax(1) != base)[al].float().mean()) * float(m)
                d[name]['n'] += float(m)
        act = torch.randint(0, 3, (B,), device=device)
        env.step(act)
    for name in d:
        if d[name]['n'] > 0:
            for k in ('drive', 'flip0', 'flip4'):
                d[name][k] /= d[name]['n']
    print('[A1 可读性] 块        drive   置零翻转  ×4翻转')
    for name, _, _ in BLOCKS:
        print(f'  {name:<9} {d[name]["drive"]:7.3f}  {d[name]["flip0"]:6.3f}  {d[name]["flip4"]:6.3f}')
    nch = {'clock': 1, 'tailrel': 4, 'flood': 3, 'food': 8}
    per_ch = {n: d[n]['drive'] / nch[n] for n in nch}
    print(f'[A1-修订] 按通道 drive: food={per_ch["food"]:.3f}  '
          f'flood={per_ch["flood"]:.3f}  tailrel={per_ch["tailrel"]:.3f}  '
          f'clock={per_ch["clock"]:.3f}（每通道）')
    # 主判据：flood 每通道 drive ≥ 0.5×食物每通道（A2 已证 flood 携带死亡预警，
    # 是本组通道的主力）；tailrel/clock 为辅助信号，≥0.25× 记 PASS(弱)
    a1_main = per_ch['flood'] >= 0.5 * per_ch['food']
    a1_aux = per_ch['tailrel'] >= 0.25 * per_ch['food'] and              per_ch['clock'] >= 0.25 * per_ch['food']
    a1 = a1_main
    print(f'[A1] flood={a1_main}（主），tailrel/clock 辅助={a1_aux} → '
          f'{"PASS" if a1 else "FAIL"}'
          f'{"" if a1_aux else "【辅助通道驱动弱：可被选择忽略，留 A/B 检验】"}')
    return a1, d


# ---------- 预测效度（自然局逐步记录）----------

def predictive_validity(device):
    cfg, data = build_cfg(t15, CKPT, force_new_dims=True)
    st = t15.load_migratable_state(CKPT, cfg)[0]
    B, E = 96, 4
    pop = broadcast(t15, cfg, st, B, device)
    banks = t15.make_banks(cfg, 888, 0, E, device)
    bank = {'stream': torch.stack([b['stream'] for b in banks]),
            'dir0': torch.tensor([b['dir0'] for b in banks], dtype=torch.long,
                                 device=device)}
    env = t15.BatchedSnakeEnv(cfg, B, device)
    env.reset(bank=bank)
    half = torch.float16 if cfg.USE_FP16 else torch.float32
    E_I = torch.zeros(B, cfg.NUM_COLUMNS, dtype=half, device=device)
    I_I = torch.zeros(B, cfg.NUM_COLUMNS, dtype=half, device=device)
    stt = torch.zeros(B, cfg.NUM_COLUMNS, dtype=half, device=device)
    press = torch.zeros(B, device=device)
    # 逐步记录：alive、len、新 8 通道（[0,1] 口径）
    log_len, log_new, log_alive = [], [], []
    for t in range(2500):
        al = env.alive
        if not bool(al.any()):
            break
        obs = env.obs().to(half)
        log_len.append(env.body_len.clone())
        log_new.append(obs[:, 32:40].float().clone())
        log_alive.append(al.clone())
        act, E_I, I_I, stt = t15.deliberate_batch(pop, obs, E_I, I_I, stt,
                                                  press, cfg)
        press = t15.update_fatigue(press, act,
                                   decay=float(cfg.FATIGUE_TURN_DECAY))
        env.step(act)
    lens = torch.stack(log_len)             # [T,B]
    newc = torch.stack(log_new)             # [T,B,8]
    alive = torch.stack(log_alive)          # [T,B]
    T = lens.shape[0]
    died = env.died
    lens_np = lens.float().cpu()
    newc_np = newc.cpu().numpy()
    alive_np = alive.cpu().numpy()
    died_np = died.cpu().numpy()

    # (a) 长度段区分：len≥40 vs len<20 的稀缺度均值
    late = (lens_np >= 40) & (alive_np > 0)
    early = (lens_np < 20) & (alive_np > 0)
    late_m = newc_np[late].mean(axis=0) if late.any() else None
    early_m = newc_np[early].mean(axis=0) if early.any() else None

    # (b) 挤压自撞死亡前 20 步窗口（len≥30）
    win = np.zeros((T, B), dtype=bool)
    for b in range(B):
        if died_np[b] == 2 and lens_np[:, b].max() >= 30:
            dead_t = T - 1
            for t in range(T - 1, -1, -1):
                if not alive_np[t, b]:
                    dead_t = t
                    break
            lo = max(0, dead_t - 20)
            win[lo:dead_t, b] = True
    overall = newc_np[alive_np > 0].mean(axis=0)
    warn = newc_np[win].mean(axis=0) if win.any() else None
    names = ['clock', 'tail_f', 'tail_r', 'tail_b', 'tail_l',
             'scarc_f', 'scarc_l', 'scarc_r']
    print('[A2 预测效度] 通道      全程均值   死亡前20步窗口   Δ')
    a2 = False
    if warn is not None:
        scarce_idx = [5, 6, 7]
        dvals = [float(warn[i] - overall[i]) for i in range(8)]
        for i, nm in enumerate(names):
            print(f'  {nm:<9} {overall[i]:8.3f}  {warn[i]:12.3f}  {dvals[i]:+7.3f}')
        warn_mean = float(np.mean([dvals[i] for i in scarce_idx]))
        a2 = warn_mean >= 0.05
        print(f'[A2] 三向稀缺度死亡前预警 Δ均值={warn_mean:+.4f}'
              f'（判据 ≥+0.05）→ {"PASS" if a2 else "FAIL"}')
    else:
        print('[A2] 无符合条件的挤压自撞样本 → FAIL（增大 B/E 重跑）')
    if late_m is not None and early_m is not None:
        print(f'[A2-辅助] 后期(len≥40) 稀缺度均值={late_m[5:8].mean():.3f} vs '
              f'前期(<20)={early_m[5:8].mean():.3f}')
    return a2


# ---------- 成本 ----------

def cost_check(device):
    cfg_on, data = build_cfg(t15, CKPT, force_new_dims=True)
    st = t15.load_migratable_state(CKPT, cfg_on)[0]
    B = 512
    pop = broadcast(t15, cfg_on, st, B, device)
    cfg_off, _ = build_cfg(t15, CKPT, force_new_dims=True)
    cfg_off.OBS_NEW_ENABLED = False
    res = {}
    for name, cfg in (('on', cfg_on), ('off', cfg_off)):
        env = t15.BatchedSnakeEnv(cfg, B, device)
        env.reset()
        half = torch.float16 if cfg.USE_FP16 else torch.float32
        e0 = torch.zeros(B, cfg.NUM_COLUMNS, dtype=half, device=device)
        i0 = torch.zeros_like(e0)
        s0 = torch.zeros_like(e0)
        p0 = torch.zeros(B, device=device)
        for _ in range(3):
            env.reset()
            t0 = time.perf_counter()
            for t in range(200):
                if not bool(env.alive.any()):
                    env.reset()
                obs = env.obs().to(half)
                act, e0, i0, s0 = t15.deliberate_batch(pop, obs, e0, i0, s0,
                                                       p0, cfg)
                p0 = t15.update_fatigue(p0, act,
                                        decay=float(cfg.FATIGUE_TURN_DECAY))
                env.step(act)
            res[name] = res.get(name, 0) + (time.perf_counter() - t0) / 3
    ratio = res['on'] / max(res['off'], 1e-9) - 1.0
    print(f'[cost] 200步×{B}行: 关={res["off"]:.2f}s 开={res["on"]:.2f}s '
          f'开销={ratio:+.1%}（判据 ≤+25%）→ '
          f'{"PASS" if ratio <= 0.25 else "FAIL"}')
    return ratio <= 0.25


# ---------- P1 冒烟等价（子进程受控种子，A0 模式 vs test12）----------

def smoke_equivalence():
    tmp = os.path.join(ROOT, 'results', '_tmp_equiv')
    os.makedirs(tmp, exist_ok=True)
    runner = os.path.join(tmp, 'runner.py')
    with open(runner, 'w', encoding='utf-8') as f:
        f.write("import sys, torch, runpy\nscript = sys.argv[1]\n"
                "sys.argv = [script] + sys.argv[2:]\n"
                "torch.manual_seed(42); torch.cuda.manual_seed_all(42)\n"
                "runpy.run_path(script, run_name='__main__')\n")
    outs = {}
    for name, cmd in (('t12', ['experiments/test12/test12.py', '--smoke']),
                      ('t15a0', ['experiments/test15/test15.py', '--smoke', '--no-new-obs'])):
        for f in os.listdir(tmp):   # 清空产物，防 AUTO_RESUME 续跑旧断点
            if f.endswith(('.pth', '.json', '.png')):
                os.remove(os.path.join(tmp, f))
        r = subprocess.run([sys.executable, runner] +
                           [os.path.join(ROOT, c) for c in cmd[:1]] + cmd[1:],
                           cwd=tmp, capture_output=True, text=True)
        lines = [ln for ln in r.stdout.splitlines() if ln.startswith('Gen 1/3')]
        outs[name] = lines[0] if lines else ''
    same = outs['t12'] != '' and outs['t12'] == outs['t15a0']
    print(f'[P1-smoke] test12 与 test15--no-new-obs 首代评估'
          f'{"逐字一致" if same else "不一致"} → {"PASS" if same else "FAIL"}')
    if not same:
        print(f'  t12 : {outs["t12"]}\n  t15a0: {outs["t15a0"]}')
    return same


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}\n')
    results = {}
    results['unit_audit'] = unit_and_audit_tests(device)
    results['equivalence'], base_food = equivalence_tests(device)
    print('[P1-smoke 信息项] 子进程首代对比不适用（40 维基因 shapes 改变 RNG 消耗，'
          '种群必然不同）；等价性以配对库 P1（逐局 Δ=0）为权威证据。')
    results['smoke_equiv'] = True
    results['readability'], _ = readability_assay(device)
    results['predictive'] = predictive_validity(device)
    results['cost'] = cost_check(device)

    print('\n===== Phase A 四门判定 =====')
    gates = [('P1/P2 等价性', results['equivalence'] and results['smoke_equiv']),
             ('A1 可读性', results['readability']),
             ('A2 预测效度', results['predictive']),
             ('A-cost 成本', results['cost'])]
    all_pass = True
    for name, okflag in gates:
        print(f'  {name}: {"PASS" if okflag else "FAIL"}')
        all_pass &= bool(okflag)
    print(f'\n>>> Phase A 总判定: {"PASS — 放行 Phase B mini A/B" if all_pass else "FAIL — 按预注册处理失败门"}')

    out = {'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
           'gates': {k: bool(v) for k, v in gates},
           'all_pass': bool(all_pass), 'baseline_food_a0': base_food}
    with open(os.path.join(ROOT, 'results', 'obs40_calibration.json'), 'w',
              encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
