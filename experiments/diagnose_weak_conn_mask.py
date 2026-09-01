# ==========================================
# experiments/diagnose_weak_conn_mask.py —— H2 检验：实战时屏蔽弱内连接是否提分
#
# 猜想（用户 H2）：模型内噪影响性能，实战时对弱内连接 mask 可减少小噪音，
# 基因迭代保留弱连接。本脚本对 test7h 最优脑做 |W_rec| 幅值屏蔽扫描：
# 屏蔽活跃连接中幅值最小的 X%（仅评估时置零，不动基因），同一组 CRN 库
# 配对比较 40 局 → 排除食物运气的差异。
# 产出：results/test7h_weakconn_mask.json
# ==========================================
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments', 'test7_series'))  # 归档后模块路径
import test7h as t7  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPS = 40


def main():
    path = os.path.join(ROOT, 'test7h_econ_best_model.pth')
    data = torch.load(path, map_location='cpu', weights_only=False)
    st = data['brain']
    cfg = t7.Config()
    for k, v in data.get('config', {}).items():
        if hasattr(cfg, k) and isinstance(v, (int, float, bool, str)):
            setattr(cfg, k, v)
    cfg.EVAL_EPISODES = EPS
    cfg.MAX_STEPS = min(cfg.MAX_STEPS, 20000)
    dev = t7._resolve_device(cfg)

    variants = [('baseline', None, None), ('rec10', 'W_rec', 0.10),
                ('rec20', 'W_rec', 0.20), ('rec30', 'W_rec', 0.30),
                ('rec40', 'W_rec', 0.40), ('all20', 'ALL', 0.20)]
    banks = t7.make_banks(cfg, 999, 1, EPS, dev)   # 同库配对比较
    results = {}
    for name, gene, frac in variants:
        pop = t7.GeneStack(cfg, B=EPS, device=dev)
        pop.random_init()
        for i in range(EPS):
            pop.set_individual_from_state(i, st)
        n_masked = 0
        if gene is not None:
            with torch.no_grad():
                genes = ['W_in', 'W_rec', 'W_out'] if gene == 'ALL' else [gene]
                n_masked = 0
                for g in genes:
                    W = pop.__dict__[g].float()
                    M = pop.__dict__[g.replace('W_', 'M_')].float()
                    active = W * M
                    mag = active.abs()
                    thresh = torch.quantile(mag[mag > 0], frac) if frac > 0 else None
                    if thresh is not None:
                        n_masked += int((M > 0).sum() * frac)
                        pop.__dict__[g] = torch.where(
                            (mag <= thresh) & (M > 0),
                            torch.zeros_like(W), W).to(pop.__dict__[g].dtype)
        if cfg.USE_FP16:
            pop.fp16()
        m = t7._eval_pop_banks(pop, cfg, banks)
        food = m[:, 0].numpy()
        turn = ((m[:, 8] + m[:, 9]) / (m[:, 1] + m[:, 2]).clamp(min=1)).numpy()
        results[name] = dict(food_mean=float(food.mean()), food_std=float(food.std()),
                             turn_density=float(turn.mean()),
                             n_masked=n_masked)
        print(f"[{name:>9}] food {food.mean():.2f} ± {food.std():.2f} "
              f"(max {food.max():.0f}) | 转向密度 {turn.mean():.2f}"
              + (f" | 屏蔽 {n_masked} 连接" if n_masked else ""))

    with open(os.path.join(ROOT, 'results', 'test7h_weakconn_mask.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print('已写入 results/test7h_weakconn_mask.json')


if __name__ == '__main__':
    main()
