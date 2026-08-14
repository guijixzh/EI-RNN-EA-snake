# ==========================================
# test6_short_run.py —— test6.py 端到端短训练验证
# 用小规模配置跑完整主循环（含种子加载、rollout、GAE、PPO、评估、保存），
# 验证 run_training 全过程无异常。
# ==========================================

import matplotlib
matplotlib.use('Agg')   # 非交互后端：run_training 末尾的可视化不弹窗阻塞
import torch
import numpy as np
import importlib.util
import os
import time

spec = importlib.util.spec_from_file_location("test6", "test6.py")
test6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test6)

Cfg = test6.Config

# 小规模配置：4 环境 × 8 步 = 32 样本，2 轮迭代
def short_cfg():
    cfg = Cfg()
    cfg.N_ENVS = 4
    cfg.ROLLOUT_LEN = 8
    cfg.MINIBATCH_SIZE = 8
    cfg.PPO_EPOCHS = 2
    cfg.TOTAL_ITERATIONS = 2
    cfg.EVAL_INTERVAL = 1          # 每轮都评估（验证评估通路）
    cfg.EVAL_EPISODES = 2
    cfg.CHECKPOINT_INTERVAL = 1    # 每轮都保存断点（验证保存通路）
    # 用临时检查点/模型路径，避免污染真实文件
    cfg.CHECKPOINT_PATH = 'test6_short_checkpoint.pth'
    cfg.BEST_MODEL_PATH = 'test6_short_best.pth'
    cfg.AUTO_RESUME = True         # 验证断点保存/续接
    cfg.SEED_FROM_TEST5D = True    # 验证种子加载
    cfg.TEST5D_MODEL_PATH = 'test5d_best_model.pth'
    return cfg


if __name__ == "__main__":
    cfg = short_cfg()
    print("=== test6 端到端短训练验证（2 轮）===")
    # 清理可能残留的临时文件
    for p in (cfg.CHECKPOINT_PATH, cfg.BEST_MODEL_PATH):
        if os.path.exists(p):
            os.remove(p)

    t0 = time.perf_counter()
    # 直接调用 run_training —— 它内部会加载 test5d 种子并跑完整 PPO 循环
    # 注意：run_training 完成后会删除断点并启动可视化；这里可视化会阻塞，
    # 但 TOTAL_ITERATIONS=2 训练极短，我们接受 matplotlib 弹出。
    # （实际验证中可视化窗口可能等待关闭；若需自动化可用
    #   matplotlib.use('Agg')，这里保持原样以便观察。）
    test6.run_training(cfg)

    print(f"\n端到端短训练完成，耗时 {time.perf_counter() - t0:.1f}s")
    print("验证：")
    print(f"  - 临时最优模型存在: {os.path.exists(cfg.BEST_MODEL_PATH)}")
    print(f"  - 临时断点已删除: {not os.path.exists(cfg.CHECKPOINT_PATH)}")