# ==========================================
# expand_fanin.py —— 16 系稀疏基因组槽位扩容工具（REC_FANIN K_old → K_new）
#
# 动机：test16b 历史最优（simp gen272, food 65.0）为 K=16 固定扇入，结构分析
# （test16b 日志 §8）表明循环拓扑仍是随机图且进化未重排——16 槽位是容量瓶颈。
# 本工具把已训练模型/全种群断点的 rec_idx/rec_w 从 [.., N, K_old] 扩到
# [.., N, K_new]：前 K_old 槽原样保留，新增槽位补随机源 + 默认零权
# （"槽位权重为 0 的连接 M=1 但 W=0，W*M=0 等效断开"，einbrain/io.py 同款语义）
# → zero 模式扩容后行为与原模型严格等价（浮点和归约次序差以内），新槽位
# 交给进化接管（rec_w 权重噪声 + 槽位重连变异）。config['REC_FANIN'] 如实
# 更新，后续 --seed-model / --resume-pop / AUTO_RESUME 全链路一致。
#
# 输入自动识别：
#   best-model（含 'brain' 键）→ 扩容 brain + config['REC_FANIN']
#   全种群断点（含 'pop' 键）  → 扩容 pop.rec_idx/rec_w + best_state + config
#     （云端 320 代 simp 断点扩容后 --resume-pop 即全种群无缝续训，路径A）
#
# 自检（--check，默认开；zero 模式为硬门）：
#   P1 解析：稠密化（scatter/scatter_add）后 W_rec 逐位相等、M_rec 为超集、
#            新边权重恰为 0
#   P2 行为：test16b.deliberate_batch 同观测/同 press 驱动 200 步对拍，
#            max|ΔE| 与 argmax 翻转统计（zero 硬门：max|ΔE|≤1e-4 且无实质翻转）
#   P3 守卫负例：原 K 模型载入 K_new run 必须被 load_best_state 拒绝
#
# 用法：
#   python experiments/expand_fanin.py --src test16b_simp_best_model.pth --fanin 32
#   python experiments/expand_fanin.py --src test16b_simp_checkpoint.pth --fanin 32 \
#       --out test16b_simp_e32_checkpoint.pth        # 断点模式（云端路径A）
#   续训：
#   python test16b.py --fit-mode simple --fanin 32 --seed-model test16b_simp_best_e32.pth
#   python test16b.py --fit-mode simple --fanin 32 --resume-pop <扩容断点>
# ==========================================

import argparse
import importlib.util
import os
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_test16():
    """动态加载仓库根 test16b.py（评估/前向语义权威实现，同 einbrain/vis.py 惯例）。"""
    p = os.path.join(ROOT, 'test16b.py')
    spec = importlib.util.spec_from_file_location('_t16_expand', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def expand_slots(rec_idx, rec_w, k_new, new_source='fresh', init='zero',
                 seed=20260906, max_bytes=64_000_000):
    """槽位扩容：[..., N, K_old] → [..., N, k_new]（dtype 保持不变）。

    new_source='fresh'  新槽优先采该行未连过的源（直接扩大唯一边集，对症
                        "16 连接狭窄"；槽间亦去重，10 轮拒绝采样）
    new_source='random' 与 test16b.random_init 完全同款（允许重复源=权重叠加）
    init='zero'         新槽权重 0（严格等价，进化经权重噪声/重连接管）
    init='noise'        新槽权重 randn*0.05（同 random_init 尺度，立即有弱信号）
    自连一律禁止（random_init 同款兜底 (ar+1)%N）。
    """
    k_old = rec_idx.shape[-1]
    if k_new <= k_old:
        raise ValueError(f'k_new({k_new}) 须 > k_old({k_old})')
    lead = rec_idx.shape[:-2]
    N = rec_idx.shape[-2]
    L = 1
    for d in lead:
        L *= d
    n_new = k_new - k_old
    # 分块上限：中间比较张量 [c,N,n_new,(k_old+n_new)] 控制在 ~64MB
    per_row = n_new * (k_old + n_new)
    chunk = max(1, min(256, max_bytes // (N * per_row * 1)))

    g = torch.Generator().manual_seed(int(seed))
    idx_flat = rec_idx.reshape(L, N, k_old).long()
    out_i = torch.empty(L, N, k_new, dtype=torch.long)
    out_w = torch.empty(L, N, k_new, dtype=torch.float32)
    out_i[:, :, :k_old] = idx_flat
    out_w[:, :, :k_old] = rec_w.reshape(L, N, k_old).float()
    ar = torch.arange(N).view(1, N, 1)

    for s in range(0, L, chunk):
        e = min(s + chunk, L)
        new = torch.randint(0, N, (e - s, N, n_new), generator=g)
        if new_source == 'fresh':
            ex = idx_flat[s:e]
            for _ in range(10):
                bad = new == ar
                bad |= (new.unsqueeze(-1) == ex.unsqueeze(-2)).any(-1)
                d = (new.unsqueeze(-2) == new.unsqueeze(-1))
                eye = torch.eye(n_new, dtype=torch.bool)
                bad |= (d & ~eye).any(-1)
                if not bool(bad.any()):
                    break
                new = torch.where(bad, torch.randint(0, N, new.shape, generator=g), new)
        new = torch.where(new == ar, (ar + 1) % N, new)   # 自连兜底（同 random_init）
        if init == 'zero':
            wnew = torch.zeros(e - s, N, n_new)
        else:
            wnew = torch.randn(e - s, N, n_new, generator=g) * 0.05
        out_i[s:e, :, k_old:] = new
        out_w[s:e, :, k_old:] = wnew

    out_i = out_i.reshape(*lead, N, k_new).to(rec_idx.dtype)
    out_w = out_w.reshape(*lead, N, k_new).to(rec_w.dtype)
    return out_i, out_w


def densify(idx, w, N):
    """[N,K] 槽位 → 稠密 (M_rec, W_rec)（einbrain/io._densify_sparse_rec 同款）。"""
    M = torch.zeros(N, N)
    W = torch.zeros(N, N)
    i = idx.long()
    M.scatter_(1, i, 1.0)
    W.scatter_add_(1, i, w.float())
    return M, W


def slot_stats(idx, w):
    """[N,K] 单脑槽位统计。"""
    N, K = idx.shape
    uniq = torch.tensor([len(set(idx[i].tolist())) for i in range(N)])
    zero_slots = int((w == 0).sum())
    return {
        'K': int(K),
        'unique_edges': int(len(set(zip(torch.arange(N).repeat_interleave(K).tolist(),
                                        idx.reshape(-1).tolist())))),
        'fanin_unique_min': int(uniq.min()), 'fanin_unique_mean': float(uniq.float().mean()),
        'fanin_unique_max': int(uniq.max()),
        'zero_weight_slots': zero_slots,
        'weight_mass': float(w.abs().sum()),
    }


def _single_brain(data):
    """best-model 或全种群断点 → 单脑 state（断点取 best_state）。"""
    if 'brain' in data:
        return data['brain']
    return data.get('best_state')


def check_zero_equivalence(src_path, dst_path, steps=200, strict=True):
    """P1 解析 + P2 行为对拍（单脑级）。返回 (ok, report)。

    strict=True（zero 模式）硬门：W_rec 逐位相等、新边零权、max|ΔE|≤1e-4 且
    argmax 翻转仅允许发生在 <1e-3 的近平局。
    strict=False（noise 模式）弱校验：旧边权重不被扰动；行为差异仅报告不判门。"""
    rep = {}
    a = torch.load(src_path, map_location='cpu', weights_only=False)
    b = torch.load(dst_path, map_location='cpu', weights_only=False)
    st_a, st_b = _single_brain(a), _single_brain(b)
    N = int(st_a['N'])
    Ma, Wa = densify(st_a['rec_idx'], st_a['rec_w'], N)
    Mb, Wb = densify(st_b['rec_idx'], st_b['rec_w'], N)
    new_mask = (Mb > 0) & (Ma <= 0)
    rep['P1_new_edges'] = int(new_mask.sum())
    if strict:
        rep['P1_W_rec_bitwise_equal'] = bool(torch.equal(Wa, Wb))
        rep['P1_M_rec_superset'] = bool(((Mb >= Ma).all()))
        rep['P1_new_edges_all_zero_weight'] = bool((Wb[new_mask] == 0).all())
        p1_ok = rep['P1_W_rec_bitwise_equal'] and rep['P1_M_rec_superset'] \
            and rep['P1_new_edges_all_zero_weight']
    else:
        diff = (Wa != Wb)
        rep['P1_W_diff_only_at_new_edges'] = bool((diff & ~new_mask).sum() == 0)
        p1_ok = rep['P1_W_diff_only_at_new_edges']

    t16 = _load_test16()
    traces = {}
    for tag, path, st in (('src', src_path, st_a), ('dst', dst_path, st_b)):
        data = a if tag == 'src' else b
        cfg = t16.Config()
        for key in ('NUM_COLUMNS', 'OBS_DIM', 'ACTION_DIM', 'REC_FANIN', 'FRAME_RATE',
                    'SHORT_TERM_DECAY', 'SHORT_TERM_GAIN', 'INPUT_DECAY',
                    'TAU_E_MIN', 'TAU_E_MAX', 'W_EI_MIN', 'W_EI_MAX',
                    'W_IE_MIN', 'W_IE_MAX', 'FATIGUE_TURN_GAIN', 'FATIGUE_TURN_DECAY'):
            if key in data['config']:
                setattr(cfg, key, data['config'][key])
        # model 模式顺带走正式 load_best_state 守卫（K 守卫正例）；checkpoint
        # 模式的 src 无 brain 键，直接用 P1 已解析的内存单脑
        if 'brain' in data:
            res = t16.load_best_state(path, cfg)
            if res is None:
                return False, {**rep, 'P2_error': f'{path} load_best_state 守卫未过'}
        if st is None:
            return False, {**rep, 'P2_error': f'{path} 无单脑可对拍'}
        pop = t16.GeneStack(cfg, B=1, device=torch.device('cpu'))
        pop.random_init()
        pop.set_individual_from_state(0, st)
        pop.fp32()
        pop.refresh_eff()
        g = torch.Generator().manual_seed(123)
        obs_seq = torch.randn(steps, 1, cfg.OBS_DIM, generator=g)
        E = torch.zeros(1, pop.N)
        I = torch.zeros(1, pop.N)
        stt = torch.zeros(1, pop.N)
        press = torch.zeros(1)
        Es, Ls, As, Ps = [], [], [], []
        for t in range(steps):
            # deliberate_batch 内联展开（保留逐帧 logits 供近平局分析）
            logits_sum = None
            for k in range(cfg.FRAME_RATE):
                o = obs_seq[t] * (cfg.INPUT_DECAY ** k)
                logits, E, I, stt = t16.forward_batch(pop, o, E, I, stt, press, cfg)
                logits_sum = logits if logits_sum is None else logits_sum + logits
            act = torch.argmax(logits_sum, dim=1)
            Ps.append(press.clone())            # press 序列取自 src，dst 喂同一序列
            press = t16.update_fatigue(press, act, decay=float(cfg.FATIGUE_TURN_DECAY))
            Es.append(E.clone())
            Ls.append(logits.clone())
            As.append(act.clone())
        traces[tag] = (torch.stack(Es), torch.stack(Ls), torch.stack(As), torch.stack(Ps))

    Ea, La, Aa, _ = traces['src']
    Eb, Lb, Ab, _ = traces['dst']
    dE = float((Ea - Eb).abs().max())
    dL = float((La - Lb).abs().max())
    mism = int((Aa != Ab).sum())
    margins = []
    if mism:
        srt = La.topk(2, dim=2).values                     # [steps, 1, 2]
        marg = (srt[:, :, 0] - srt[:, :, 1]).clamp(min=0)  # [steps, 1]
        flip_steps = (Aa != Ab).nonzero()
        margins = [float(marg[t, 0]) for t, _ in flip_steps]
    rep['P2_steps'] = steps
    rep['P2_max_dE'] = dE
    rep['P2_max_dlogit'] = dL
    rep['P2_argmax_flips'] = mism
    rep['P2_flip_margins'] = margins[:8]
    if strict:
        # zero 硬门：动力学差 ≤1e-4；argmax 翻转仅允许发生在 <1e-3 的近平局上
        p2_ok = dE <= 1e-4 and all(m < 1e-3 for m in margins)
    else:
        p2_ok = True                              # noise：行为差异仅报告，不判门
    return p1_ok and p2_ok, rep


def check_guard_negative(src_path, k_new):
    """P3：原 K 模型载入 K_new run 必须被 load_best_state 显式拒绝。"""
    t16 = _load_test16()
    cfg = t16.Config()
    data = torch.load(src_path, map_location='cpu', weights_only=False)
    for key in ('NUM_COLUMNS', 'OBS_DIM', 'ACTION_DIM'):
        if key in data['config']:
            setattr(cfg, key, data['config'][key])
    cfg.REC_FANIN = int(k_new)
    return t16.load_best_state(src_path, cfg) is None


def main():
    ap = argparse.ArgumentParser(
        description='16 系稀疏基因组槽位扩容（REC_FANIN K_old → K_new；zero 模式行为严格等价）')
    ap.add_argument('--src', required=True, help='输入 best-model 或全种群断点 .pth')
    ap.add_argument('--out', default=None,
                    help='输出路径（默认 <stem>_e<K_new>.pth；断点输入加 _checkpoint 后缀）')
    ap.add_argument('--fanin', type=int, default=32, help='目标槽位数 K_new（默认 32）')
    ap.add_argument('--init', choices=['zero', 'noise'], default='zero',
                    help='新槽权重初始化：zero=严格等价（默认）| noise=randn*0.05')
    ap.add_argument('--new-source', choices=['fresh', 'random'], default='fresh',
                    help='新槽源采样：fresh=避开该行已连源（默认）| random=random_init 同款')
    ap.add_argument('--seed', type=int, default=20260906, help='扩容采样种子（可复现）')
    ap.add_argument('--steps', type=int, default=200, help='行为对拍步数')
    ap.add_argument('--no-check', dest='check', action='store_false',
                    help='跳过自检（不推荐）')
    args = ap.parse_args()

    if not os.path.exists(args.src):
        raise SystemExit(f'[错误] 输入不存在: {args.src}')
    data = torch.load(args.src, map_location='cpu', weights_only=False)
    if 'brain' in data:
        mode = 'model'
    elif 'pop' in data:
        mode = 'checkpoint'
    else:
        raise SystemExit('[错误] 无法识别输入：既无 brain（best-model）也无 pop（全种群断点）键')

    if mode == 'model':
        brains = [('brain', data['brain'])]
    else:
        brains = [('pop', data['pop'])]
        if data.get('best_state'):
            brains.append(('best_state', data['best_state']))

    for tag, st in brains:
        k_old = int(st['rec_idx'].shape[-1])
        print(f'[{tag}] K={k_old} rec_idx{tuple(st["rec_idx"].shape)} '
              f'rec_w{tuple(st["rec_w"].shape)} dtype={st["rec_idx"].dtype}')
        ni, nw = expand_slots(st['rec_idx'], st['rec_w'], args.fanin,
                              new_source=args.new_source, init=args.init, seed=args.seed)
        st['rec_idx'] = ni
        st['rec_w'] = nw
        if 'K' in st:
            st['K'] = int(args.fanin)
        if tag == 'brain':
            print('  扩容后:', slot_stats(ni, nw))
        else:
            print(f'  扩容后: rec_idx{tuple(ni.shape)} rec_w{tuple(nw.shape)} '
                  f'dtype={ni.dtype}/{nw.dtype}')

    data['config'] = {**data['config'], 'REC_FANIN': int(args.fanin)}
    data['expand'] = {
        'src': os.path.basename(args.src), 'k_new': int(args.fanin),
        'init': args.init, 'new_source': args.new_source, 'seed': int(args.seed),
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    if args.out:
        out = args.out
    else:
        stem, ext = os.path.splitext(os.path.basename(args.src))
        suffix = '' if stem.endswith('_checkpoint') else '_best'
        out = os.path.join(os.path.dirname(os.path.abspath(args.src)),
                           f'{stem}_e{args.fanin}{suffix}{ext}')
    os.makedirs(os.path.dirname(os.path.abspath(out)) or '.', exist_ok=True)
    torch.save(data, out)
    print(f'[输出] {out} (config.REC_FANIN -> {args.fanin})')

    ok = True
    if args.check:
        if mode == 'model':
            probe = out
        elif data.get('best_state'):
            # 断点模式：对 best_state 单脑做同等自检（pop 扩容与同一函数，结构性同构）
            probe = out + '.best_probe.pth'
            torch.save({'brain': data['best_state'], 'food': data.get('best_food', -1),
                        'steps': 0.0, 'config': data['config'], 'saved_at': ''}, probe)
        else:
            print('[自检] 断点无 best_state，仅做结构扩容（pop 与 brain 共用同一变换函数）')
            return
        print(f'[自检] P1 解析 + P2 行为对拍 + P3 K 守卫负例（init={args.init}，'
              f'{"硬门" if args.init == "zero" else "弱校验+报告"}）...')
        ok, rep = check_zero_equivalence(args.src, probe, steps=args.steps,
                                         strict=(args.init == 'zero'))
        for k, v in rep.items():
            print(f'  {k}: {v}')
        p3 = check_guard_negative(args.src, args.fanin)
        print(f'  P3_guard_rejects_K_mismatch: {p3}')
        ok = ok and p3
        if mode != 'model':
            os.remove(probe)
        if not ok:
            os.remove(out)
            raise SystemExit('[自检失败] 已删除输出，请勿使用（zero 模式必须严格等价）')
        print(f'[自检通过] {out} 可作为 --seed-model / --resume-pop 种子使用')
    else:
        print('[警告] 已跳过自检——zero 模式等价性未验证')


if __name__ == '__main__':
    main()
