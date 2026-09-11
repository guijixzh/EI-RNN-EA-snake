# ==========================================
# eval_7b_random1000.py —— 7b 原版模型：N 盘随机地图并发评测 + 最佳盘视频
#
# 流程：
#   1. 加载 7b 原版稠密检查点 → 稀疏转换（load_best_state_7b_sparse，无损条件见转换器）
#   2. N 盘随机地图（ep=0..N-1，一盘一流）B=N 并发评测，记录每盘 food/steps
#   3. 取 food 最高（并列取步数最少）的一盘
#   4. B=1 重放该盘并逐步采集，校验重放 food 与并发一致（CRN 按行解耦，应完全一致）
#   5. 用 render_win_video 的同一渲染配置出片（60fps / 2帧每神经步 / 10帧每游戏步，
#      只保留 H.264 压缩版）
# ==========================================

import argparse
import os
import sys
import time

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_win_video as rwv
import test16c_cheat7b as sim


def main():
    ap = argparse.ArgumentParser(description='7b 原版随机地图并发评测 + 最佳盘视频')
    ap.add_argument('--model', default='test7b_latest_gen_best.pth')
    ap.add_argument('--n-maps', type=int, default=1000)
    ap.add_argument('--out', default='7b_best_random_h264.mp4')
    ap.add_argument('--max-steps', type=int, default=3000)
    ap.add_argument('--map', type=int, default=None,
                    help='直接重放该地图序号（跳过并发评测；464=上次评测最佳盘 80/98）')
    ap.add_argument('--fps', type=int, default=rwv.FPS)
    ap.add_argument('--frames-per-brain', type=int, default=rwv.FRAMES_PER_BRAIN)
    ap.add_argument('--frames-per-game', type=int, default=rwv.FRAMES_PER_GAME)
    args = ap.parse_args()
    fpb, fpg = args.frames_per_brain, args.frames_per_game

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)

    # ---- 1. 加载 7b 原版（动力学超参以 7b checkpoint 的 config 为准）----
    data = torch.load(args.model, map_location='cpu', weights_only=False)
    cfg = sim.Config()
    for k, v in data.get('config', {}).items():
        if not k.startswith('__'):
            try:
                setattr(cfg, k, v)
            except Exception:
                pass
    out = sim.load_best_state_7b_sparse(args.model, cfg)
    assert out is not None, '7b 稠密→稀疏转换失败'
    st = out[0]
    print(f"7b 原版: batched food={data.get('food')} steps={data.get('steps')} "
          f"N={st['N']} K={st['K']} FRAME_RATE={cfg.FRAME_RATE}", flush=True)

    # ---- 2. N 盘随机地图并发评测（一盘一流，ep=0..N-1）；--map 跳过评测直接重放 ----
    K = int(cfg.FRAME_RATE)
    if args.map is not None:
        best_i = args.map
        print(f'--map {best_i}: 跳过并发评测，直接重放该盘'
              f'（make_bank 按 ep 确定性播种，与当时评测地图逐子一致）', flush=True)
    else:
        print(f'并发评测 {args.n_maps} 盘随机地图 (B={args.n_maps})...', flush=True)
        pop = sim.GeneStack(cfg, B=args.n_maps, device=dev)
        pop.random_init()
        for i in range(args.n_maps):
            pop.set_individual_from_state(i, st)
        pop.refresh_eff()
        if getattr(cfg, 'USE_FP16', True):
            pop.fp16(); pop.refresh_eff()

        banks = [sim.make_bank(cfg, cfg.MAP_GEN, 1, i, dev)
                 for i in range(args.n_maps)]
        bank = {'stream': torch.stack([b['stream'] for b in banks]),
                'dir0': torch.tensor([b['dir0'] for b in banks], device=dev)}
        env = sim.BatchedSnakeEnv(cfg, args.n_maps, dev)
        env.reset(bank=bank)
        E = torch.zeros(args.n_maps, pop.N, dtype=pop.dtype, device=dev)
        I = torch.zeros_like(E); stt = torch.zeros_like(E)
        cts = torch.zeros(args.n_maps, pop.A, dtype=pop.dtype, device=dev)
        food_vec = torch.zeros(args.n_maps, dtype=torch.long, device=dev)
        t0 = time.time()
        for s in range(args.max_steps):
            if not bool(env.alive.any()):
                break
            obs = env.obs().to(pop.dtype)
            act, E, I, stt = sim.deliberate_batch(pop, obs, E, I, stt, cts, cfg)
            cts = sim.update_fatigue(cts, act)
            env.step(act)
            food_vec += env.ate.long()
            if (s + 1) % 200 == 0:
                alive = int(env.alive.sum())
                print(f'  step {s + 1}: 存活 {alive}/{args.n_maps} · '
                      f'best food {int(food_vec.max())}', flush=True)
        steps_vec = env.steps.cpu().clone()
        food_np = food_vec.cpu().numpy()
        steps_np = steps_vec.cpu().numpy()
        print(f'评测完成 {time.time() - t0:.0f}s | food: max={food_np.max()} '
              f'mean={food_np.mean():.1f} median={int(np.median(food_np))} | '
              f'零分盘 {(food_np == 0).sum()}', flush=True)

        # ---- 3. 最佳盘（food 优先，并列取步数少）----
        order = sorted(range(args.n_maps), key=lambda i: (-food_np[i], steps_np[i]))
        print('Top-10:', [(i, int(food_np[i]), int(steps_np[i])) for i in order[:10]],
              flush=True)
        best_i = order[0]
        print(f'最佳盘: #{best_i} food={food_np[best_i]} steps={steps_np[best_i]}',
              flush=True)

    # ---- 4. B=1 重放最佳盘并逐步采集 ----
    bank1 = (sim.make_bank(cfg, cfg.MAP_GEN, 1, best_i, dev) if args.map is not None
             else banks[best_i])
    print(f'重放最佳盘 #{best_i}（B=1，逐步采集）...', flush=True)
    frames, won, n_steps = rwv.rollout(cfg, st, args.max_steps,
                                       e0_init='zero', bank=bank1)
    replay_food = frames[-1]['score']
    print(f"重放: {n_steps} 步 × {K} 思考帧 | {len(frames)} 帧 | won={won} | "
          f"score={replay_food}/{cfg.TARGET_FOOD}", flush=True)
    if args.map is None and replay_food != int(food_np[best_i]):
        print(f'⚠ 重放 food ({replay_food}) 与并发 ({int(food_np[best_i])}) 不一致，'
              f'视频以重放实际轨迹为准', flush=True)

    # ---- 5. 渲染（与通关视频同一可视化配置；只保留 H.264）----
    layout = rwv.precompute_layout(st)
    print('弹性布局收敛完成', flush=True)
    perm, bounds = rwv.column_permutation(st)
    nsz = int(st['N'])
    W = np.zeros((nsz, nsz))
    W[np.arange(nsz)[:, None],
      st['rec_idx'].long().numpy()] = st['rec_w'].float().numpy()
    if perm is not None:
        W = W[np.ix_(perm, perm)]
    model_tag = os.path.basename(args.model)
    title = (f'7b 原版 — BEST OF {args.n_maps} RANDOM MAPS · {model_tag} · '
             f'map #{best_i} · {replay_food}/{cfg.TARGET_FOOD}')
    renderer = rwv.Renderer(layout, frames, rwv.HEAT_WINDOW, title, K,
                            perm=perm, wmat=W, cluster_bounds=bounds)

    master = args.out.replace('_h264.mp4', '.mp4')
    write, release = rwv.pick_writer(master, args.fps, 1920, 1080)
    content = (len(frames) - 1) * fpb + 1
    total = content + rwv.HOLD
    print(f'渲染 {total} 帧 @ {args.fps}fps -> {master} '
          f'(每神经步{fpb}帧 / 每游戏步{fpg}帧 · '
          f'时长 {total / args.fps / 60:.1f} 分钟)', flush=True)
    t0 = time.time()
    last_key = None
    last_rgb = None
    n_draw = 0
    for t in range(content):
        fi = min(t // fpb, len(frames) - 1)
        gi = min((t // fpg) * K, len(frames) - 1)
        if (fi, gi) != last_key:
            renderer.draw(fi, gi)
            last_rgb = renderer.rgb()
            last_key = (fi, gi)
            n_draw += 1
        write(last_rgb)
        if n_draw % 500 == 0 and n_draw > 0:
            el = time.time() - t0
            print(f'  视频帧 {t + 1}/{content} (绘制 {n_draw}, {el:.0f}s, '
                  f'ETA {el / n_draw * (content - t) / 60:.0f}min)', flush=True)
    for _ in range(rwv.HOLD):
        write(last_rgb)
    release()

    import shutil
    if shutil.which('ffmpeg'):
        os.system(f'ffmpeg -y -i "{master}" -c:v libx264 -pix_fmt yuv420p '
                  f'-crf 20 "{args.out}" -loglevel error')
        if os.path.exists(args.out) and os.path.getsize(args.out) > 1e6:
            os.remove(master)
            print(f'完成: {args.out} '
                  f'({os.path.getsize(args.out) / 1e6:.1f} MB, mp4v 原稿已删)',
                  flush=True)
        else:
            print(f'转码失败，保留原稿: {master}', flush=True)
    else:
        print(f'（未找到 ffmpeg）完成: {master}', flush=True)


if __name__ == '__main__':
    main()
