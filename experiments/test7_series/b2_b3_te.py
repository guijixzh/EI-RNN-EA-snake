# ==========================================
# experiments/b2_b3_te.py —— B2 详解 + B3 扩展(k=40/50/60) + te 可重复性
#
# B2D：逐食段分解——绕路比 SL/md（几何绕路）与 SL/cd（回路距离占比）分离，
#      加直线游程统计（最长直行/游程≤2 占比）——区分"有序长直绕行"与
#      "混沌密集微转"。
# B3X：反事实缝合扩展 k=40/50（7h 与 7b 双方）+ k=60（仅 7b）。达成阶段
#      免钟（纯决策探针），接管阶段免钟，报告再吃食数统计与成功率。
# TEREP：checkpoint 精英 top128 × 5 库逐个体 te——跨库可重复性（半库相关）
#        与 te-food 相关性（配额选择的前提检验）。
# 产出：results/b2_b3_te.json + b2_detail.png
# ==========================================
import copy
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'experiments', 'test7_series'))  # 归档后模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test7h as t7h  # noqa: E402
from ref_solver import MiniEnv, CycleSolver, rollout_reference  # noqa: E402
from behavior_study import (ModelBrain, rollout_model_mini, rollout_ref_mini)  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
G = 10
EPS = 40
DEV = torch.device('cpu')


def cpu_cfg(base=None):
    cfg = copy.copy(base if base is not None else t7h.Config())
    cfg.DEVICE = 'cpu'
    cfg.USE_FP16 = False
    cfg.MAX_STEPS = 20000
    return cfg


def load_brain(path):
    data = torch.load(os.path.join(ROOT, path), map_location='cpu', weights_only=False)
    cfg = t7h.Config()
    cfg.OBS_MANHATTAN = False          # pre-7g 回退
    cfg.FATIGUE_TURN_GAIN = 0.0
    for k, v in data.get('config', {}).items():
        if hasattr(cfg, k) and isinstance(v, (int, float, bool, str)):
            setattr(cfg, k, v)
    return data['brain'], cpu_cfg(cfg)


def seg_detail(acts, events, path, init_food, snaps, cyc):
    """逐食段：SL/转弯/md(曼哈顿)/cd(回路前向距离)/最长直行/游程≤2占比。"""
    rows = []
    start = 0
    for i, et in enumerate(events):
        dst = snaps[i - 1][1] if i >= 1 else init_food
        src = path[start] if start < len(path) else dst
        sl = et - start
        a_seg = acts[start:et]
        turns = sum(1 for a in a_seg if a != 0)
        # 直行游程
        runs, run = [], 0
        for a in a_seg:
            if a == 0:
                run += 1
            else:
                if run:
                    runs.append(run)
                run = 0
        if run:
            runs.append(run)
        md = abs(src[0] - dst[0]) + abs(src[1] - dst[1])
        cd = cyc.fd(src, dst)
        rows.append(dict(SL=sl, turns=turns, md=md, cd=cd,
                         longest=max(runs) if runs else 0,
                         le2=float(np.mean([r <= 2 for r in runs])) if runs else 0.0))
        start = et
    return rows


def bin_agg(rows, key, w=5, max_bin=8):
    bins = {}
    for i, r in enumerate(rows):
        k = min(i // w, max_bin)
        bins.setdefault(k, []).append(r[key])
    return {k: float(np.mean(v)) for k, v in sorted(bins.items())}


def main():
    torch.manual_seed(0)
    base = t7h.Config()
    banks = t7h.make_banks(base, 777, 1, EPS, DEV)
    cyc = CycleSolver()
    result = {}

    # ---- 局面：7h / 7b / 参考 ----
    st7h, cfg7h = load_brain('test7h_econ_best_model.pth')
    st7b, cfg7b = load_brain('artifacts/test7b/test7b_best_model.pth')

    print('[参考] CycleSolver（免钟）...')
    rlogs = rollout_reference(banks, starve_slope=1e9, max_steps=30000)

    print('[模型局] 7h（免钟，供 B2/B3）...')
    mlogs7h_free = [rollout_model_mini(cfg7h, st7h, banks[i], starve_slope=1e9)
                    for i in range(EPS)]
    print('  7h 免钟 food mean %.1f' % np.mean([len(l['events']) for l in mlogs7h_free]))
    print('[模型局] 7b（免钟）...')
    mlogs7b_free = [rollout_model_mini(cfg7b, st7b, banks[i], starve_slope=1e9)
                    for i in range(EPS)]
    print('  7b 免钟 food mean %.1f' % np.mean([len(l['events']) for l in mlogs7b_free]))

    # ================= B2D =================
    print('\n[B2D] 逐食段分解（段序 5 分箱）')
    def profile(logs):
        rows_all = []
        for lg in logs:
            rows_all += seg_detail(lg['acts'], lg['events'], lg['path'],
                                   lg.get('init_food'), lg['snaps'], cyc)
        return dict(
            SL=bin_agg(rows_all, 'SL'),
            density=bin_agg(rows_all, 'turns'),
            detour_md={k: v for k, v in zip(
                bin_agg(rows_all, 'SL').keys(),
                [bin_agg(rows_all, 'SL')[k] / max(bin_agg(rows_all, 'md')[k], 1)
                 for k in bin_agg(rows_all, 'SL').keys()])},
            detour_cd={k: v for k, v in zip(
                bin_agg(rows_all, 'SL').keys(),
                [bin_agg(rows_all, 'SL')[k] / max(bin_agg(rows_all, 'cd')[k], 1)
                 for k in bin_agg(rows_all, 'SL').keys()])},
            md=bin_agg(rows_all, 'md'), cd=bin_agg(rows_all, 'cd'),
            longest=bin_agg(rows_all, 'longest'),
            le2=bin_agg(rows_all, 'le2'))
    p7h, p7b, pref = profile(mlogs7h_free), profile(mlogs7b_free), profile(rlogs)
    result['B2D'] = dict(model7h=p7h, model7b=p7b, ref=pref)
    hdr = f'{"段":<7}{"|7h转弯/段":>10}{"7b转弯":>8}{"参转弯":>8}{"|7h绕/md":>9}{"7b绕":>7}{"参绕":>7}' \
          f'{"|7hSL/cd":>9}{"参SL/cd":>9}{"|7h最长直":>9}{"参最长直":>9}'
    print(hdr)
    ks = sorted(set(p7h['density']) | set(pref['density']))
    for k in ks:
        print(f'{k * 5:>3}-{k * 5 + 4:<3}'
              f'{p7h["density"].get(k, float("nan")):>8.2f}'
              f'{p7b["density"].get(k, float("nan")):>8.2f}'
              f'{pref["density"].get(k, float("nan")):>8.2f}'
              f'{p7h["detour_md"].get(k, float("nan")):>9.2f}'
              f'{p7b["detour_md"].get(k, float("nan")):>7.2f}'
              f'{pref["detour_md"].get(k, float("nan")):>7.2f}'
              f'{p7h["detour_cd"].get(k, float("nan")):>9.2f}'
              f'{pref["detour_cd"].get(k, float("nan")):>9.2f}'
              f'{p7h["longest"].get(k, float("nan")):>9.1f}'
              f'{pref["longest"].get(k, float("nan")):>9.1f}')

    # ================= B3X =================
    print('\n[B3X] 反事实缝合扩展（达成与接管全程免钟，纯决策探针）')
    b3x = {}

    def handover(tag, brain, cfg, logs_free, ks):
        for k in ks:
            gains_m2r, gains_r2m, reach = [], [], 0
            for i in range(EPS):
                lg = logs_free[i]
                if len(lg['events']) < k:
                    continue
                reach += 1
                mpart = rollout_model_mini(cfg, brain, banks[i], starve_slope=1e9,
                                           take_food=k)
                r2m = rollout_ref_mini(banks[i], starve_slope=1e9,
                                       start_state=mpart['start_state'])
                gains_m2r.append(len(r2m['events']))
                rpart = rollout_ref_mini(banks[i], starve_slope=1e9, take_food=k)
                m2r = rollout_model_mini(cfg, brain, banks[i], starve_slope=1e9,
                                         start_state=rpart['start_state'])
                gains_r2m.append(len(m2r['events']))
            if reach:
                g1, g2 = np.array(gains_m2r), np.array(gains_r2m)
                b3x[f'{tag}_k{k}'] = dict(
                    reach=reach,
                    model_to_ref_mean=float(g1.mean()), model_to_ref_med=float(np.median(g1)),
                    model_to_ref_min=int(g1.min()),
                    model_to_ref_ge15=float((g1 >= 15).mean()),
                    ref_to_model_mean=float(g2.mean()), ref_to_model_med=float(np.median(g2)),
                    ref_to_model_min=int(g2.min()),
                    ref_to_model_ge15=float((g2 >= 15).mean()))
                print(f'  [{tag}] k={k}（达成 {reach}/40）: 模型起手→参考再吃 '
                      f'{g1.mean():.1f}/中位 {np.median(g1):.0f}/min {g1.min()}'
                      f'（≥15食 {float((g1 >= 15).mean()):.0%}）| '
                      f'参考起手→模型再吃 {g2.mean():.1f}/中位 {np.median(g2):.0f}'
                      f'/min {g2.min()}（≥15食 {float((g2 >= 15).mean()):.0%}）')
    handover('7h', st7h, cfg7h, mlogs7h_free, (40, 50))
    handover('7b', st7b, cfg7b, mlogs7b_free, (40, 50, 60))
    result['B3X'] = b3x

    # ================= TEREP =================
    print('\n[TEREP] 精英 top128 te 跨库可重复性')
    ck = torch.load(os.path.join(ROOT, 'test7h_econ_checkpoint.pth'),
                    map_location='cpu', weights_only=False)
    cfg5 = t7h.Config()
    for k, v in ck.get('config', {}).items():
        if hasattr(cfg5, k) and isinstance(v, (int, float, bool, str)):
            setattr(cfg5, k, v)
    cfg5 = cpu_cfg(cfg5)
    n = 128
    genes = {g: ck['pop'][g][:n] for g in t7h.GeneStack.GENES}
    pop5 = t7h.GeneStack(cfg5, B=n, device=DEV)
    pop5.unpack(genes)
    pop5.fp32()          # checkpoint 基因为 half，CPU fp32 评估需转回
    te_banks, food_banks = [], []
    for bi in range(5):
        sub = pop5.clone_rows(list(range(n)))
        sub.refresh_eff()
        m = t7h._eval_pop_banks(sub, cfg5, [banks[bi]])
        te = np.where(m[:, 10] > 0, m[:, 3] / m[:, 10].clamp(min=1), np.nan)
        te = np.clip(te, 0, 4)
        te[np.asarray(m[:, 0]) <= 0] = np.nan
        te_banks.append(te)
        food_banks.append(m[:, 0].numpy())
    TE = np.stack(te_banks, axis=1)      # [128, 5]
    FD = np.stack(food_banks, axis=1)
    te_mean = np.nanmean(TE, axis=1)
    # 半库重复性：前2库均值 vs 后3库均值 的 Pearson
    half1, half2 = np.nanmean(TE[:, :2], axis=1), np.nanmean(TE[:, 2:], axis=1)
    ok = ~(np.isnan(half1) | np.isnan(half2))
    r = float(np.corrcoef(half1[ok], half2[ok])[0, 1])
    fd_mean = np.nanmean(FD, axis=1)
    okf = ~(np.isnan(te_mean) | np.isnan(fd_mean))
    r_fd = float(np.corrcoef(te_mean[okf], fd_mean[okf])[0, 1])
    # te 值的库间标准差（相对波动）
    te_std_rel = float(np.nanmean(np.nanstd(TE, axis=1) / np.maximum(te_mean, 0.1)))
    result['TEREP'] = dict(half_corr=r, te_food_corr=r_fd,
                           te_std_rel=te_std_rel,
                           te_mean_quantiles=[float(x) for x in np.nanquantile(te_mean, [0.1, 0.25, 0.5, 0.75, 0.9])],
                           n_nan=int(np.isnan(te_mean).sum()))
    print(f'  半库相关 r={r:.3f} | te-food 相关 r={r_fd:.3f} | '
          f'库间相对波动 {te_std_rel:.1%} | te 均值分位 '
          f'{[round(float(x), 2) for x in np.nanquantile(te_mean, [0.1, 0.25, 0.5, 0.75, 0.9])]}')

    # ---- 落盘 + 绘图 ----
    with open(os.path.join(ROOT, 'results', 'b2_b3_te.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print('\n[落盘] results/b2_b3_te.json')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
        ks = sorted(set(p7h['density']) | set(pref['density']))
        x = [int(k) * 5 for k in ks]
        ax = axes[0]
        ax.plot(x, [p7h['density'].get(k, np.nan) for k in ks], 'o-', label='7h')
        ax.plot(x, [p7b['density'].get(k, np.nan) for k in ks], '^-', label='7b')
        ax.plot(x, [pref['density'].get(k, np.nan) for k in ks], 's-', label='ref')
        ax.set_title('B2D turn density per segment')
        ax.set_xlabel('segment bin'); ax.legend(fontsize=8)
        ax = axes[1]
        ax.plot(x, [p7h['longest'].get(k, np.nan) for k in ks], 'o-', label='7h longest straight')
        ax.plot(x, [pref['longest'].get(k, np.nan) for k in ks], 's-', label='ref longest straight')
        ax.set_title('B2D longest straight run per segment')
        ax.set_xlabel('segment bin'); ax.legend(fontsize=8)
        ax = axes[2]
        labels = [f'k={k}' for k in ks3]
        w = 0.35
        xs = np.arange(len(ks3))
        ax.bar(xs - w / 2, [b3x[f'7b_k{k}']['model_to_ref_mean'] for k in ks3],
               width=w, label='7b: model start → ref extra')
        ax.bar(xs + w / 2, [b3x[f'7b_k{k}']['ref_to_model_mean'] for k in ks3],
               width=w, label='7b: ref start → model extra')
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_title('B3X 7b handover (clock-free)')
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(os.path.join(ROOT, 'results', 'b2_b3_te.png'), dpi=110)
        print('[落盘] results/b2_b3_te.png')
    except Exception as e:
        print(f'(绘图跳过: {e})')


if __name__ == '__main__':
    main()
