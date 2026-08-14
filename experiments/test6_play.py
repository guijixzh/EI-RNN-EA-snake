# ==========================================
# test6_play.py —— test6 最优模型游玩可视化（独立程序）
#
# 加载 test6_best_model.pth，用 matplotlib 动画自动播放贪吃蛇
# （argmax 贪心策略，与训练评估完全一致），支持多局连播与统计。
#
# 用法：
#   python test6_play.py                    自动播放，默认 5 局
#   python test6_play.py --episodes 20      连播 20 局
#   python test6_play.py --model xxx.pth    指定模型文件
#   python test6_play.py --fast             关动画快速连播（只打印统计）
#   python test6_play.py --speed 0.05       动画帧间隔（秒）
#   python test6_play.py --grid 12          网格大小（默认 10，须与模型一致）
#
# 快捷键（动画模式）：
#   Q / Esc  退出
#   空格      暂停 / 继续
# ==========================================

import argparse
import importlib.util
import os
import sys
import time

# ---- 先解析参数：--fast 需在导入 test6 前设置 Agg 后端 ----
parser = argparse.ArgumentParser(description="test6 最优模型贪吃蛇游玩可视化")
parser.add_argument('--episodes', type=int, default=5, help='连播局数（默认 5）')
parser.add_argument('--model', type=str, default='test6_best_model.pth',
                    help='模型文件路径（默认 test6_best_model.pth）')
parser.add_argument('--fast', action='store_true',
                    help='快速模式：关动画，只打印每局与统计')
parser.add_argument('--speed', type=float, default=0.1,
                    help='动画帧间隔秒数（默认 0.1）')
parser.add_argument('--grid', type=int, default=10,
                    help='网格大小（默认 10，须与模型 OBS_DIM=24 兼容）')
parser.add_argument('--max-steps', type=int, default=1000,
                    help='单局最大步数（默认 500）')
args = parser.parse_args()

if args.fast:
    import matplotlib
    matplotlib.use('Agg')
else:
    import matplotlib
    # 保留默认交互后端（TkAgg 等）

import torch
import numpy as np
import matplotlib.pyplot as plt

# ---- 复用 test6.py 组件（不重复实现）----
_spec = importlib.util.spec_from_file_location("test6", "test6.py")
test6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(test6)


def load_native_model(path):
    """加载最优模型（兼容 test6 保存格式与 test5d 旧格式）。

    返回 (brain) 或 None（文件不存在 / 不兼容）。
    """
    if not os.path.exists(path):
        print(f"错误: 模型文件不存在: {path}")
        return None
    cfg = test6.Config()
    try:
        result = test6.load_best_model_brain(path, cfg)
    except Exception as e:
        print(f"错误: 模型加载失败 ({e})")
        return None
    if result is None:
        print(f"错误: 模型 {path} 与当前配置不兼容 (N/OBS/ACTION_DIM)")
        return None
    brain, food, steps = result
    return brain


def run_episode(brain, cfg, env, render_ax=None, img=None, speed=0.1,
                paused_cb=None, check_exit=None):
    """用 argmax 策略跑一局。

    - render_ax 为 None 时纯计算（--fast）
    - 返回 (foods, steps)
    """
    E, I, short, he, hi, counts = test6.make_zero_states(brain, 1)
    obs = env.reset()
    ep_food = 0
    steps = 0
    done = truncated = False

    with torch.no_grad():
        while not done and not truncated and steps < cfg.MAX_STEPS:
            # ---- 决策（与 evaluate 完全一致：K=FRAME_RATE 思考 + argmax）----
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
            logits, _, E, I, short, he, hi = test6.forward_ppo_k(
                brain, obs_t, E, I, short, he, hi, counts)
            action = int(logits.squeeze(0).argmax().item())
            counts = test6.update_counts(counts, torch.tensor([[action]]))

            # ---- 执行 ----
            obs, ate, done, truncated = env.step(action)
            if ate:
                ep_food += 1
            steps += 1

            # ---- 渲染 ----
            if render_ax is not None:
                if paused_cb is not None and check_exit is not None:
                    # 检查暂停/退出（由外部回调处理）
                    if check_exit():
                        return None  # 信号：用户退出
                    while paused_cb():
                        plt.pause(0.05)
                grid = test6.render_snake_game(env)
                img.set_data(grid)
                render_ax.set_title(
                    f"Step: {env.steps} | Score: {len(env.body) - 2} "
                    f"| Food: {ep_food}", fontsize=11)
                render_ax.figure.canvas.draw_idle()
                plt.pause(speed)

    return ep_food, steps


def main():
    # ---- 加载模型 ----
    brain = load_native_model(args.model)
    if brain is None:
        sys.exit(1)

    cfg = test6.Config()
    cfg.GRID_SIZE = args.grid
    cfg.MAX_STEPS = args.max_steps

    print(f"=== test6 游玩展示 ===")
    print(f"  模型: {args.model}")
    if hasattr(brain, 'N'):
        print(f"  脑结构: {brain.N} 皮质柱 | OBS={brain.obs_dim} | ACT={brain.action_dim}")
    active = brain.M_in.sum().item() + brain.M_rec.sum().item() + brain.M_out.sum().item()
    print(f"  激活连接: {active:.0f}")
    print(f"  连播: {args.episodes} 局 | 动画: {'关(fast)' if args.fast else '开'}")

    # ---- 动画模式初始化画布（含键盘监听：Q/Esc 退出，空格 暂停）----
    render_ax = None
    img = None
    paused = False
    quit_flag = False

    def _paused_cb():
        return paused

    def _check_exit():
        return quit_flag

    if not args.fast:
        plt.ion()
        fig, render_ax = plt.subplots(figsize=(6, 6))
        env_init = test6.SnakeEnv(grid_size=args.grid, max_steps=args.max_steps)
        img = render_ax.imshow(test6.render_snake_game(env_init),
                               cmap='viridis', vmin=0, vmax=1)
        render_ax.set_title("test6 Best Brain Playing Snake | Q:退出 空格:暂停")
        render_ax.axis('off')

        def on_key(event):
            nonlocal paused, quit_flag
            if event.key in ('q', 'Q', 'escape'):
                quit_flag = True
            elif event.key == ' ':
                paused = not paused

        fig.canvas.mpl_connect('key_press_event', on_key)

    # ---- 连播 ----
    env = test6.SnakeEnv(grid_size=args.grid, max_steps=args.max_steps)
    all_foods = []
    all_steps = []
    try:
        for ep in range(args.episodes):
            result = run_episode(brain, cfg, env,
                                 render_ax=render_ax, img=img,
                                 speed=args.speed,
                                 paused_cb=_paused_cb, check_exit=_check_exit)
            if result is None:
                print("用户退出")
                break
            foods, steps = result
            all_foods.append(foods)
            all_steps.append(steps)
            print(f"  第 {ep + 1} 局: 吃 {foods} 个食物 | 存活 {steps} 步 "
                  f"| 最终长度 {len(env.body) - 2}")
    except KeyboardInterrupt:
        print("\n手动中断")

    # ---- 统计 ----
    if all_foods:
        avg_food = float(np.mean(all_foods))
        avg_steps = float(np.mean(all_steps))
        best_idx = int(np.argmax(all_foods))
        print("\n=== 统计 ===")
        print(f"  平均每局食物: {avg_food:.2f}")
        print(f"  平均存活步数: {avg_steps:.1f}")
        print(f"  最佳一局: 第 {best_idx + 1} 局 ({all_foods[best_idx]} 食物)")
    else:
        print("未完成任何一局")

    if not args.fast:
        plt.ioff()
        # 保留最终画面供查看
        try:
            plt.show()
        except Exception:
            pass


if __name__ == "__main__":
    main()