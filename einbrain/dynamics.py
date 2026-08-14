"""E-I 皮质柱共享核心数学。

test5d（CPU 单脑）、test6（PPO 可微）、test7（GPU 批量）三套实现
共用的 E-I 动力学公式，抽离于此以避免三处重复与漂移：

    E_new = sigmoid(total_in + tau_eff * E_prev - w_ei_eff * I_prev)
    I_new = sigmoid(w_ie_eff * E_new)

以及激素单柱释放、扩散衰减、短期调制、动作疲劳等子模块。
所有函数均为逐元素/矩阵运算，天然支持 [N]（单脑）与 [B, N]（批量）两种形状。
"""
from __future__ import annotations

import torch


def clamp_w(w, lo, hi):
    """权重边界保护（运行时仅 clamp 数据，不影响图）。"""
    return torch.clamp(w, lo, hi)


def hormone_commands(excit_logits, inhib_logits, gate_threshold=0.0):
    """单柱释放门控（test5d v2）：每类激素同一帧最多在一个柱释放。

    释放强度 = sigmoid(最大 logit)，最大 logit 严格 > 阈值才释放
    （0 初始化网络的 logit=0 不触发释放，激素输出恒为 0）。
    兴奋与抑制各自独立选择释放柱。返回 (excit_cmd, inhib_cmd)。
    """
    max_e = excit_logits.max()
    gate_e = (max_e > gate_threshold).float()
    excit_cmd = torch.zeros_like(excit_logits)
    argmax_e = int(torch.argmax(excit_logits).item())
    excit_cmd[argmax_e] = gate_e * torch.sigmoid(max_e)

    max_i = inhib_logits.max()
    gate_i = (max_i > gate_threshold).float()
    inhib_cmd = torch.zeros_like(inhib_logits)
    argmax_i = int(torch.argmax(inhib_logits).item())
    inhib_cmd[argmax_i] = gate_i * torch.sigmoid(max_i)
    return excit_cmd, inhib_cmd


def diffuse_decay(cmd, state, decay, diffusion, M_norm):
    """激素沿拓扑扩散 + 长期衰减。"""
    return (1 - decay) * cmd + decay * (
        (1 - diffusion) * state + diffusion * torch.matmul(M_norm, state)
    )


def effective_tau_e(tau_init, short_term, horm_e, horm_i, cfg):
    """有效 tau_e = init + 短期调制 + 激素调制，再 clamp。"""
    tau = tau_init + \
        cfg.SHORT_TERM_GAIN * short_term + \
        cfg.EXCIT_HORMONE_GAIN * horm_e - \
        cfg.INHIB_HORMONE_GAIN * horm_i
    return torch.clamp(tau, cfg.TAU_E_MIN, cfg.TAU_E_MAX)


def ei_update(total_in, E_prev, I_prev, tau_eff, w_ei_eff, w_ie_eff):
    """E-I 离散代数更新。返回 (E_new, I_new)。"""
    E_new = torch.sigmoid(total_in + tau_eff * E_prev - w_ei_eff * I_prev)
    I_new = torch.sigmoid(w_ie_eff * E_new)
    return E_new, I_new


def fatigue_penalty(counts, cfg):
    """纯次数动作疲劳：fatigue = relu(计数-阈值)*增益，封顶。"""
    fatigue = torch.relu(counts - cfg.FATIGUE_THRESHOLD) * cfg.FATIGUE_GAIN
    return torch.clamp(fatigue, max=cfg.FATIGUE_MAX)
