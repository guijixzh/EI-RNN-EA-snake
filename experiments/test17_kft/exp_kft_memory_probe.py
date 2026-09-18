# -*- coding: utf-8 -*-
"""test17 K帧首帧输入/尾帧读出（--kframe-input first --kframe-read tail）记忆探针。

对象：qk_ft（首帧输入+尾帧读出，pop256×100 代从零训练）best 模型；
      参照 qk_base（同种子默认口径 best 模型，反应式基线）与随机初始化网络（动力学下限）。

方法（沿用 exp_7b_memory_probe*.py 序列探针方法论 + 本模式特有的步内保持探针）：
  探针1 步内信号保持（首帧模式特有）：单步内 K=5 帧手动前向——第 0 帧喂真实观测、
        第 1~4 帧零输入（press 同置零）——逐帧记录 logits。三个模型在同一首帧调度下
        对比：若 E/I 循环状态把首帧信息带到了尾帧，则尾帧 logits 对不同观测仍应有
        区分度（跨 15 个真实初始状态的方差 V(k)）。
        指标：每帧跨状态方差 V(k)、保留率 V(尾帧)/V(首帧)、首帧/尾帧 argmax 一致率、
        跨状态 logits 两两 L2（首帧 vs 尾帧）。
  探针2 跨步历史依赖：每模型按其自身原生口径跑（ft=first+tail / base=decay+sum）。
        15 真实状态、64 条 [S, 中间独立均匀采样, S] 序列（长 64 游戏步），所有序列
        同起点同终点输入、初始 E/I/st 全零 → 末步输出的任何跨序列方差都纯来自历史
        路径。指标：动作一致率、logits d(end,begin)、参照系比值（换输入引起的输出
        变化幅度）、跨序列末步两两 L2（路径特异性）、E 状态相对漂移、热机偏移消散
        （回到 S 后再保持 4 步看偏离衰减）。

用法（仓库根目录）：
  python experiments/test17_kft/exp_kft_memory_probe.py
  python experiments/test17_kft/exp_kft_memory_probe.py \
      --ft snake_std_qk_ft_best_model.pth --base snake_std_qk_base_best_model.pth \
      --out results/test17_kft/memory_probe.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import snake_std as sim                                  # noqa: E402


def set_sim(name):
    """切换被测动力学模块（默认 snake_std；test17b_leak = 实验B泄漏积分器 fork）。"""
    global sim
    if name == 'snake_std':
        return
    import importlib.util as _ilu
    _p = os.path.join(ROOT, 'experiments', 'test17_kft', name + '.py')
    _spec = _ilu.spec_from_file_location(name, _p)
    _m = _ilu.module_from_spec(_spec)
    sys.modules[name] = _m
    _spec.loader.exec_module(_m)
    sim = _m

N_STATE = 15                                             # 真实初始状态数
S_IDX = 0                                                # 序列首尾钉住的状态
N_SEQ = 64                                               # 序列条数（批维）
SEQ_LEN = 64                                             # 每条序列游戏步数
HOLD_EXTRA = 4                                           # 回到 S 后保持步数
PROBE_SEED = 20260912                                    # 随机网初始化/序列采样种子


def load_model(path, device):
    """载入 best 模型：回填 config → GeneStack → 注入 brain → 精度/eff 对齐。
    兼容两种基因组：16 稀疏（rec_idx/rec_w）直读；7b 稠密（W_rec/M_rec）经
    标准实现转换器自动转稀疏（无损条件：每行掩码内非零 ≤ K）。"""
    data = torch.load(path, map_location='cpu', weights_only=False)
    cfg = sim.Config()
    for k, v in (data.get('config') or {}).items():
        if not str(k).startswith('__'):
            try:
                setattr(cfg, k, v)
            except Exception:
                pass
    if device.type == 'cpu':
        cfg.USE_FP16 = False                             # CPU 无 fp16 bmm，退 fp32
    if 'rec_idx' not in data['brain'] and 'W_rec' in data['brain']:
        res = sim.load_best_state_7b_dense_as_sparse(path, cfg)
        if res is None:
            raise RuntimeError(f'7b 稠密模型转换失败: {path}')
        data['brain'], data['food'], _ = res
    pop = sim.GeneStack(cfg, B=1, device=device)
    pop.random_init()                                    # 分配基因张量（同 play_best）
    pop.set_individual_from_state(0, data['brain'])
    pop.fp16()
    pop.refresh_eff()                                    # eff 张量随精度重建（同 play_best）
    return cfg, pop, data


def random_net(cfg_ft, device, self_conn=None, train_wii=None, wii_init=0.0,
               euler_leak=None):
    """随机初始化网络（与被测模型同架构），作动力学下限参照。
    self_conn/train_wii/euler_leak 缺省时跟随 cfg_ft 的开关。"""
    cfg = sim.Config()
    for k in ('NUM_COLUMNS', 'REC_FANIN', 'OBS_DIM', 'ACTION_DIM', 'OBS_MODE'):
        setattr(cfg, k, getattr(cfg_ft, k))
    cfg.USE_FP16 = cfg_ft.USE_FP16
    cfg.ALLOW_SELF_CONN = (getattr(cfg_ft, 'ALLOW_SELF_CONN', False)
                           if self_conn is None else self_conn)
    cfg.TRAIN_WII = (getattr(cfg_ft, 'TRAIN_WII', False)
                     if train_wii is None else train_wii)
    cfg.TRAIN_WEE = getattr(cfg_ft, 'TRAIN_WEE', False)
    if euler_leak is None:
        euler_leak = getattr(cfg_ft, 'EULER_LEAK', False)
    cfg.EULER_LEAK = euler_leak
    if euler_leak:
        cfg.TAU_E_MAX = min(float(getattr(cfg_ft, 'TAU_E_MAX', 2.0)), 1.0)
    torch.manual_seed(PROBE_SEED)
    pop = sim.GeneStack(cfg, B=1, device=device)
    pop.random_init()
    if train_wii and wii_init != 0.0:
        with torch.no_grad():
            pop.w_ii += wii_init
    pop.fp16()
    pop.refresh_eff()
    return cfg, pop


def build_real_states(cfg, device):
    """15 个真实初始观测（CRN 多库 reset，与训练分布同源）。"""
    banks = sim.make_banks(cfg, 777 + PROBE_SEED % 1000, 9, N_STATE, device)
    bank = {'stream': torch.stack([b['stream'] for b in banks]),
            'dir0': torch.tensor([b['dir0'] for b in banks], device=device)}
    env = sim.BatchedSnakeEnv(cfg, N_STATE, device)
    env.reset(bank=bank)
    return env.obs().float()                             # [S, O]


def step_frames(pop, cfg, obs, E, I, st, press, mode_in, mode_read):
    """一个游戏步：按模式跑 K 帧并按读出口径汇总，返回 (logits, 逐帧logits, E,I,st)。"""
    per_frame = []
    for k in range(cfg.FRAME_RATE):
        if mode_in == 'first':
            o = obs if k == 0 else torch.zeros_like(obs)
            p = press if k == 0 else torch.zeros_like(press)
        else:
            o = obs * (cfg.INPUT_DECAY ** k)
            p = press
        lg, E, I, st, _, _ = sim.forward_batch(pop, o, E, I, st, p, cfg)
        per_frame.append(lg.float())
    lg = per_frame[-1] if mode_read == 'tail' else torch.stack(per_frame).sum(0)
    return lg, per_frame, E, I, st


def probe_within_step(pop, cfg, obs_states, mode_in='first'):
    """探针1：首帧调度下逐帧记录 logits，量化信号在零输入帧内的存活。"""
    S, O = obs_states.shape
    dev = pop.device
    dt = pop.dtype
    obs = obs_states.to(device=dev, dtype=dt)
    E = torch.zeros(S, cfg.NUM_COLUMNS, device=dev, dtype=dt)
    I = torch.zeros_like(E)
    st = torch.zeros_like(E)
    press = torch.zeros(S, device=dev, dtype=dt)
    frames = []
    for k in range(cfg.FRAME_RATE):
        if mode_in == 'first':
            o = obs if k == 0 else torch.zeros_like(obs)
            p = press if k == 0 else torch.zeros_like(press)
        else:
            o = obs * (cfg.INPUT_DECAY ** k)
            p = press
        lg, E, I, st, _, _ = sim.forward_batch(pop, o, E, I, st, p, cfg)
        frames.append(lg.float())
    V = [f.var(dim=0).mean().item() for f in frames]     # 每帧跨状态方差均值
    pd0 = torch.cdist(frames[0], frames[0]).mean().item()
    pdt = torch.cdist(frames[-1], frames[-1]).mean().item()
    agree = (frames[-1].argmax(1) == frames[0].argmax(1)).float().mean().item()
    return {
        'V_per_frame': V,
        'retention_var': V[-1] / max(V[0], 1e-12),
        'pairwiseL2_frame0': pd0,
        'pairwiseL2_tail': pdt,
        'argmax_frame0_tail_agree': agree,
    }


def probe_history(pop, cfg, obs_states, mode_in, mode_read):
    """探针2：[S,中间随机,S] 序列——同起点同终点输入下，末步输出是否随历史路径分化。"""
    L, B = SEQ_LEN, N_SEQ
    S = obs_states.shape[0]
    dev = pop.device
    dt = pop.dtype
    rng = np.random.default_rng(PROBE_SEED)
    idx = np.zeros((L, B), dtype=np.int64)
    idx[0] = S_IDX
    idx[-1] = S_IDX
    idx[1:-1] = rng.integers(0, S, size=(L - 2, B))
    mid_mask = torch.from_numpy((idx[1:-1] != S_IDX))    # 参照系只计真正换输入的步
    obs_d = obs_states.to(device=dev, dtype=dt)

    E = torch.zeros(B, cfg.NUM_COLUMNS, device=dev, dtype=dt)
    I = torch.zeros_like(E)
    st = torch.zeros_like(E)
    press = torch.zeros(B, device=dev, dtype=dt)
    logits_rec = torch.zeros(L, B, cfg.ACTION_DIM)
    e_begin = e_end = None
    hold_off = []
    for t in range(L + HOLD_EXTRA):
        if t < L:
            si = torch.from_numpy(idx[t]).to(dev)
            obs_t = obs_d[si]
        else:                                            # 热机消散：持续喂 S
            obs_t = obs_d[S_IDX].expand(B, -1)
        lg, _, E, I, st = step_frames(pop, cfg, obs_t, E, I, st, press,
                                      mode_in, mode_read)
        if t == 0:
            logits_rec[0] = lg.float().cpu()
            e_begin = E.float().clone()
        elif t == L - 1:
            logits_rec[-1] = lg.float().cpu()
            e_end = E.float().clone()
        elif t < L - 1:
            logits_rec[t] = lg.float().cpu()
        else:
            hold_off.append(((lg.float().cpu() - logits_rec[0]).norm(dim=1)).mean().item())

    begin, end, mid = logits_rec[0], logits_rec[-1], logits_rec[1:-1]
    d_eb = (end - begin).norm(dim=1).mean().item()
    ref_d = ((mid - begin).norm(dim=2) * mid_mask.float()).sum().item() \
        / max(mid_mask.float().sum().item(), 1.0)
    pd_end = torch.cdist(end, end)
    nz = pd_end > 0
    pd_end = pd_end[nz].mean().item() if bool(nz.any()) else 0.0
    pd_mid = torch.cdist(mid[mid_mask], mid[mid_mask]).mean().item() \
        if bool(mid_mask.any()) else -1.0
    return {
        'seq_len': L, 'n_seq': B, 'n_state': S,
        'action_match_end_begin': (end.argmax(1) == begin.argmax(1)).float().mean().item(),
        'd_end_begin': d_eb,
        'ref_change_input': ref_d,
        'ratio_d_over_ref': d_eb / max(ref_d, 1e-12),
        'path_specificity_pairwiseL2_end': pd_end,
        'pairwiseL2_mid': pd_mid,
        'E_drift_rel': ((e_end - e_begin).norm(dim=1)
                        / e_end.norm(dim=1).clamp(min=1e-6)).mean().item(),
        'hold_offset_decay': hold_off,
    }


def rep(pop, n):
    """单个体复制到 n 份（高级索引=拷贝；同 _eval_pop_banks 用法）。"""
    p = pop[[0] * n]
    p.refresh_eff()
    return p


def main():
    ap = argparse.ArgumentParser(description='test17 K帧首帧输入/尾帧读出 记忆探针')
    ap.add_argument('--ft', default='artifacts/test17/snake_std_qk_ft_best_model.pth')
    ap.add_argument('--base', default='artifacts/test17/snake_std_qk_base_best_model.pth')
    ap.add_argument('--out', default='results/test17_kft/memory_probe.json')
    ap.add_argument('--prescan', action='store_true',
                    help='预检模式：随机网扫 w_ii×自连 的步内保留率后退出（实验A开训前）')
    ap.add_argument('--sim', default='snake_std',
                    help='动力学模块：snake_std（默认）| test17b_leak（实验B fork）')
    ap.add_argument('--euler-leak', dest='euler_leak', action='store_true',
                    help='预检时强制随机网开启 EULER_LEAK（τ 上限 1.0）')
    args = ap.parse_args()

    set_sim(args.sim)
    print(f"[sim] 动力学模块 = {args.sim}"
          + ("（实验B 泄漏积分器）" if args.sim != 'snake_std' else ""))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = {}

    cfg_ft, pop_ft, meta_ft = load_model(args.ft, device)
    print(f"[ft] {args.ft} (food={meta_ft.get('food')}, "
          f"N={cfg_ft.NUM_COLUMNS}, K={cfg_ft.REC_FANIN}, "
          f"enc={getattr(cfg_ft, 'OBS_ENC_VERSION', '?')})")

    obs_states = build_real_states(cfg_ft, device)
    print(f"[states] {obs_states.shape[0]} 个真实初始观测 (obs_dim={obs_states.shape[1]})")

    if args.prescan:
        print("\n=== 预检：随机网步内保留率 V(尾)/V(首)（first 调度，K=5）===")
        rows = {}
        if args.euler_leak:
            cfg_r, pop_r = random_net(cfg_ft, device, train_wii=False,
                                      euler_leak=True)
            r = probe_within_step(rep(pop_r, N_STATE), cfg_r, obs_states,
                                  mode_in='first')
            rows['euler_leak'] = r
            print(f"  euler_leak(τ≈0.7): V={['%.2e' % v for v in r['V_per_frame']]} "
                  f"保留率={r['retention_var']:.3f} 首/尾argmax一致={r['argmax_frame0_tail_agree']:.2f}")
        else:
            for sc in (False, True):
                for w in (0.0, 1.0, 3.0, 5.0):
                    cfg_r, pop_r = random_net(cfg_ft, device, self_conn=sc,
                                              train_wii=(w > 0), wii_init=w)
                    r = probe_within_step(rep(pop_r, N_STATE), cfg_r, obs_states,
                                          mode_in='first')
                    rows[f'sc={int(sc)},w_ii={w:g}'] = r
                    print(f"  自连={int(sc)} w_ii={w:g}: V={['%.2e' % v for v in r['V_per_frame']]} "
                          f"保留率={r['retention_var']:.2e} 首/尾argmax一致={r['argmax_frame0_tail_agree']:.2f}")
        parent = os.path.dirname(os.path.abspath(args.out))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({'prescan': rows}, f, ensure_ascii=False, indent=2)
        print(f"预检结果已写入 {args.out}")
        return

    cfg_b, pop_b, meta_b = load_model(args.base, device)
    print(f"[base] {args.base} (food={meta_b.get('food')})")
    cfg_r, pop_r = random_net(cfg_ft, device)
    print(f"[random] 同架构随机初始化 (seed={PROBE_SEED})")

    models = {os.path.splitext(os.path.basename(args.ft))[0]: (cfg_ft, pop_ft, 'first', 'tail'),
              os.path.splitext(os.path.basename(args.base))[0]: (cfg_b, pop_b, 'decay', 'sum'),
              'random': (cfg_r, pop_r, 'first', 'tail')}
    for name, (cfg, pop, mi, mr) in models.items():
        native = f'{mi}+{mr}'
        r1 = probe_within_step(rep(pop, N_STATE), cfg, obs_states, mode_in='first')
        r2 = probe_history(rep(pop, N_SEQ), cfg, obs_states, mi, mr)
        out[name] = {'native_mode': native, 'within_step': r1, 'history': r2}
        print(f"\n=== {name}（原生口径 {native}）===")
        print(f"  探针1 步内保持: V(k)={['%.3e' % v for v in r1['V_per_frame']]}")
        print(f"    保留率 V(尾)/V(首)={r1['retention_var']:.4f} | "
              f"首/尾 argmax 一致率={r1['argmax_frame0_tail_agree']:.3f} | "
              f"跨状态L2 首={r1['pairwiseL2_frame0']:.3f} 尾={r1['pairwiseL2_tail']:.3f}")
        print(f"  探针2 历史依赖: 动作一致率={r2['action_match_end_begin']:.3f} | "
              f"d(end,begin)={r2['d_end_begin']:.4f} | 参照系(换输入)={r2['ref_change_input']:.4f} "
              f"| 比值={r2['ratio_d_over_ref']:.4f}")
        print(f"    路径特异性(末步两两L2)={r2['path_specificity_pairwiseL2_end']:.4f} | "
              f"E相对漂移={r2['E_drift_rel']:.4f} | 热机消散={['%.4f' % h for h in r2['hold_offset_decay']]}")

    parent = os.path.dirname(os.path.abspath(args.out))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {args.out}")


if __name__ == '__main__':
    main()
