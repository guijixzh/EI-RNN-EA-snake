"""K 倍帧率思考：AI 与游戏环境的交互接口。"""
from __future__ import annotations

import numpy as np
import torch


def deliberate_action(brain, obs, E, I, K=None, decay=None):
    """游戏环境每前进一步，AI 在固定观测上做 K 次内部更新后给出动作。

    - 输入衰减：第 k 次内部更新使用 obs * (decay^k)，使 E-I 递归动力学
      在外部输入逐次衰减下『思考』。
    - 输出平均：K 步的 logits 求均值，argmax 得实际动作。
    - 内部状态 E / I 跨游戏步连续保持（不重置）。
    - 返回 (action, avg_logits, E, I)。

    注意：必须用 no_grad 而非 inference_mode——后者会把 E/I 与激素 buffer
    变成 inference tensor，离开该模式后 reset_runtime 的原地操作会报错。
    """
    cfg = brain.cfg
    if K is None:
        K = cfg.FRAME_RATE
    if decay is None:
        decay = cfg.INPUT_DECAY

    obs_np = np.asarray(obs, dtype=np.float32)
    scales = [decay ** k for k in range(K)]
    logits_sum = None
    with torch.no_grad():
        for k in range(K):
            obs_t = torch.from_numpy(obs_np * scales[k])
            logits, E, I = brain(obs_t, E, I)
            if logits_sum is None:
                logits_sum = logits
            else:
                logits_sum = logits_sum + logits

    avg_logits = logits_sum / K
    action = torch.argmax(avg_logits).item()
    return action, avg_logits, E, I
