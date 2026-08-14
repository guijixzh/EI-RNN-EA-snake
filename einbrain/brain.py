"""统一 E-I 皮质柱脑区模型。

整合 test5d / test6 的 EIBrainRegion：
    - 拓扑基因型：M_in / M_rec / M_out 稀疏掩码
    - 表现型权重：W_in / W_rec / W_out / b_out，逐柱 tau_e / w_ei / w_ie
    - 激素调控前馈网络（默认 0 初始化，TRAIN_HORMONE_NET=False 时 G3 冻结）
    - 动作疲劳（纯连续次数）、短期 tau 调制
    - PPO value head：V / b_v（test6 新增）

两种前向接口：
    - forward(obs_t, E, I)        推理（test5d 语义，单柱激素释放 + 缓存权重）
    - forward_ppo(obs_t, E, I, ...) 单步批量可微前向（test6 语义，1 步截断 BPTT）
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import dynamics


class EIBrainRegion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.N = cfg.NUM_COLUMNS
        self.obs_dim = cfg.OBS_DIM
        self.action_dim = cfg.ACTION_DIM
        self.train_hormone = bool(getattr(cfg, 'TRAIN_HORMONE_NET', False))

        # --- 基因型：拓扑掩码 ---
        self.M_in = (torch.rand(self.N, self.obs_dim) < cfg.INIT_DENSITY).float()
        self.M_rec = (torch.rand(self.N, self.N) < cfg.INIT_DENSITY).float()
        torch.diagonal(self.M_rec).zero_()
        self.M_out = (torch.rand(self.action_dim, self.N) < cfg.INIT_DENSITY).float()

        # --- 表现型：突触权重 ---
        self.W_in = nn.Parameter(torch.randn(self.N, self.obs_dim) * 0.1)
        self.W_rec = nn.Parameter(torch.randn(self.N, self.N) * 0.05)
        self.W_out = nn.Parameter(torch.randn(self.action_dim, self.N) * 0.1)
        self.b_out = nn.Parameter(torch.zeros(self.action_dim))

        # --- 每柱体 tau_e 初始值 ---
        self.tau_e_init = nn.Parameter(
            cfg.BASE_TAU_E + (torch.rand(self.N) * 2 - 1) * cfg.TAU_E_NOISE
        )

        # --- 每柱体 Wei / Wie（E-I 耦合强度，可进化）---
        self.w_ei = nn.Parameter(torch.full((self.N,), cfg.W_EI))
        self.w_ie = nn.Parameter(torch.full((self.N,), cfg.W_IE))

        # --- 激素调控前馈网络 ---
        hormone_input_dim = self.N * 3
        self.W_hormone1 = nn.Parameter(torch.zeros(cfg.HORMONE_NET_HIDDEN, hormone_input_dim))
        self.b_hormone1 = nn.Parameter(torch.zeros(cfg.HORMONE_NET_HIDDEN))
        self.W_excit = nn.Parameter(torch.zeros(self.N, cfg.HORMONE_NET_HIDDEN))
        self.b_excit = nn.Parameter(torch.zeros(self.N))
        self.W_inhib = nn.Parameter(torch.zeros(self.N, cfg.HORMONE_NET_HIDDEN))
        self.b_inhib = nn.Parameter(torch.zeros(self.N))

        # --- PPO value head（critic）---
        self.V = nn.Parameter(torch.zeros(self.N))
        self.b_v = nn.Parameter(torch.zeros(1))

        # --- 运行时状态（非进化参数）---
        self.register_buffer('hormone_excit', torch.zeros(self.N))
        self.register_buffer('hormone_inhib', torch.zeros(self.N))
        self.register_buffer('short_term_state', torch.zeros(self.N))
        self.register_buffer('consecutive_counts', torch.zeros(self.action_dim))
        self.register_buffer('last_excit_cmd', torch.zeros(self.N))
        self.register_buffer('last_inhib_cmd', torch.zeros(self.N))

        self.baseline = None

        # --- 缓存：掩码权重与归一化扩散矩阵（forward 中复用）---
        self.register_buffer('W_rec_eff', torch.zeros(self.N, self.N))
        self.register_buffer('W_out_eff', torch.zeros(self.action_dim, self.N))
        self.register_buffer('M_norm', torch.zeros(self.N, self.N))
        self.refresh_cached()

    def reset_runtime(self):
        """每局开始前重置激素、短期状态与连续动作计数。"""
        self.hormone_excit.zero_()
        self.hormone_inhib.zero_()
        self.short_term_state.zero_()
        self.consecutive_counts.zero_()

    # ==================== 推理前向（test5d 语义）====================
    def forward(self, obs_t, E_prev, I_prev):
        cfg = self.cfg

        # 1. 外部与循环输入（W_rec 用缓存掩码权重）
        ext_in = torch.matmul(self.W_in * self.M_in, obs_t)
        rec_in = torch.matmul(self.W_rec_eff, E_prev)
        total_in = ext_in + rec_in

        # 2. 激素调控前馈网络（单柱释放门控）
        hormone_input = torch.cat([E_prev, I_prev, total_in], dim=-1)
        h_hidden = torch.relu(torch.matmul(self.W_hormone1, hormone_input) + self.b_hormone1)
        excit_logits = torch.matmul(self.W_excit, h_hidden) + self.b_excit
        inhib_logits = torch.matmul(self.W_inhib, h_hidden) + self.b_inhib
        gate_thr = float(getattr(cfg, 'HORMONE_GATE_THRESHOLD', 0.0))
        excit_cmd, inhib_cmd = dynamics.hormone_commands(excit_logits, inhib_logits, gate_thr)
        self.last_excit_cmd.copy_(excit_cmd)
        self.last_inhib_cmd.copy_(inhib_cmd)

        # 3. 激素沿拓扑扩散 + 长期衰减
        self.hormone_excit = dynamics.diffuse_decay(
            excit_cmd, self.hormone_excit,
            cfg.HORMONE_DECAY, cfg.EXCIT_DIFFUSION, self.M_norm)
        self.hormone_inhib = dynamics.diffuse_decay(
            inhib_cmd, self.hormone_inhib,
            cfg.HORMONE_DECAY, cfg.INHIB_DIFFUSION, self.M_norm)

        # 4. 短期状态
        self.short_term_state = cfg.SHORT_TERM_DECAY * self.short_term_state + \
            (1 - cfg.SHORT_TERM_DECAY) * E_prev

        # 5. 有效 tau_e / Wei / Wie
        effective_tau_e = dynamics.effective_tau_e(
            self.tau_e_init, self.short_term_state,
            self.hormone_excit, self.hormone_inhib, cfg)
        w_ei_eff = dynamics.clamp_w(self.w_ei, cfg.W_EI_MIN, cfg.W_EI_MAX)
        w_ie_eff = dynamics.clamp_w(self.w_ie, cfg.W_IE_MIN, cfg.W_IE_MAX)

        # 6. E-I 离散代数更新
        E_new, I_new = dynamics.ei_update(
            total_in, E_prev, I_prev, effective_tau_e, w_ei_eff, w_ie_eff)

        # 7. 动作输出 + 疲劳抑制
        action_logits = torch.matmul(self.W_out_eff, E_new) + self.b_out
        action_logits = action_logits - dynamics.fatigue_penalty(self.consecutive_counts, cfg)

        return action_logits, E_new, I_new

    # ==================== PPO 可微前向（test6 语义）====================
    def forward_ppo(self, obs_t, E, I, short_term, horm_e, horm_i, counts):
        """单步批量可微前向（1 步 BPTT，状态全部显式传入）。

        参数均为 [B, ...] 批量张量；E/I/short/hormone 必须来自上一步
        （内部会 detach，仅保留本步到 obs_t+参数的梯度路径）。
        返回 (logits[B,A], value[B], E_next, I_next, short_next, he_next, hi_next)。
        """
        cfg = self.cfg
        E = E.detach()
        I = I.detach()
        if short_term is not None:
            short_term = short_term.detach()
        if horm_e is not None:
            horm_e = horm_e.detach()
        if horm_i is not None:
            horm_i = horm_i.detach()
        counts = counts.detach()

        # 1. 外部与循环输入（掩码现算，梯度只流向 W）
        ext_in = torch.matmul(obs_t, (self.W_in * self.M_in).T)
        rec_in = torch.matmul(E, (self.W_rec * self.M_rec).T)
        total_in = ext_in + rec_in

        # 2. 激素（默认冻结跳过；解冻用可微软释放）
        if self.train_hormone and horm_e is not None:
            hormone_input = torch.cat([E, I, total_in], dim=-1)
            h_hidden = torch.relu(torch.matmul(hormone_input, self.W_hormone1.T) + self.b_hormone1)
            excit_logits = torch.matmul(h_hidden, self.W_excit.T) + self.b_excit
            inhib_logits = torch.matmul(h_hidden, self.W_inhib.T) + self.b_inhib
            gate_thr = float(getattr(cfg, 'HORMONE_GATE_THRESHOLD', 0.0))
            excit_w = torch.softmax(excit_logits, dim=-1)
            max_e = excit_logits.max(dim=-1, keepdim=True).values
            gate_e = (max_e > gate_thr).float()
            excit_cmd = excit_w * gate_e * torch.sigmoid(max_e)
            inhib_w = torch.softmax(inhib_logits, dim=-1)
            max_i = inhib_logits.max(dim=-1, keepdim=True).values
            gate_i = (max_i > gate_thr).float()
            inhib_cmd = inhib_w * gate_i * torch.sigmoid(max_i)

            horm_e_next = (1 - cfg.HORMONE_DECAY) * excit_cmd + \
                cfg.HORMONE_DECAY * ((1 - cfg.EXCIT_DIFFUSION) * horm_e +
                                     cfg.EXCIT_DIFFUSION * torch.matmul(horm_e, self.M_norm.T))
            horm_i_next = (1 - cfg.HORMONE_DECAY) * inhib_cmd + \
                cfg.HORMONE_DECAY * ((1 - cfg.INHIB_DIFFUSION) * horm_i +
                                     cfg.INHIB_DIFFUSION * torch.matmul(horm_i, self.M_norm.T))
            horm_e_next = horm_e_next.detach()
            horm_i_next = horm_i_next.detach()
        else:
            horm_e_next = None
            horm_i_next = None
            horm_e = horm_i = None

        # 3. 短期状态（历史 detach，只保留本步梯度）
        if short_term is not None:
            short_next = (cfg.SHORT_TERM_DECAY * short_term +
                          (1 - cfg.SHORT_TERM_DECAY) * E).detach()
        else:
            short_next = None
            short_term = torch.zeros_like(E)

        # 4. 有效 tau_e / Wei / Wie
        horm_e_eff = horm_e if horm_e is not None else torch.zeros_like(E)
        horm_i_eff = horm_i if horm_i is not None else torch.zeros_like(E)
        effective_tau_e = dynamics.effective_tau_e(
            self.tau_e_init, short_term, horm_e_eff, horm_i_eff, cfg)
        w_ei_eff = dynamics.clamp_w(self.w_ei, cfg.W_EI_MIN, cfg.W_EI_MAX)
        w_ie_eff = dynamics.clamp_w(self.w_ie, cfg.W_IE_MIN, cfg.W_IE_MAX)

        # 5. E-I 离散代数更新
        E_new, I_new = dynamics.ei_update(
            total_in, E, I, effective_tau_e, w_ei_eff, w_ie_eff)

        # 6. 动作输出 + 疲劳
        action_logits = torch.matmul(E_new, (self.W_out * self.M_out).T) + self.b_out
        action_logits = action_logits - dynamics.fatigue_penalty(counts, cfg)

        # 7. value head（critic）
        value = torch.matmul(E_new, self.V.unsqueeze(1)).squeeze(1) + self.b_v

        return (action_logits, value, E_new.detach(), I_new.detach(),
                short_next, horm_e_next, horm_i_next)

    # ==================== 基础工具 ====================
    def update_fatigue(self, action):
        """更新连续动作计数（选定动作 +1，其余清零）。"""
        with torch.no_grad():
            cur = float(self.consecutive_counts[action]) + 1.0
            self.consecutive_counts.zero_()
            self.consecutive_counts[action] = cur

    def refresh_cached(self):
        """刷新前向缓存：W_rec*M_rec、W_out*M_out 与归一化扩散矩阵 M_norm。"""
        with torch.no_grad():
            self.W_rec_eff.copy_(self.W_rec.data * self.M_rec)
            self.W_out_eff.copy_(self.W_out.data * self.M_out)
            deg = self.M_rec.sum(dim=1, keepdim=True) + 1e-8
            self.M_norm.copy_(self.M_rec / deg)

    def save_genetic_baseline(self):
        """保存遗传基线。M_* 克隆快照（变异原地翻转）；其余存引用。"""
        self.baseline = {
            'W_in': self.W_in.data,
            'W_rec': self.W_rec.data,
            'W_out': self.W_out.data,
            'b_out': self.b_out.data,
            'M_in': self.M_in.clone(), 'M_rec': self.M_rec.clone(), 'M_out': self.M_out.clone(),
            'tau_e_init': self.tau_e_init.data,
            'w_ei': self.w_ei.data,
            'w_ie': self.w_ie.data,
            'V': self.V.data,
            'b_v': self.b_v.data,
            'W_hormone1': self.W_hormone1.data, 'b_hormone1': self.b_hormone1.data,
            'W_excit': self.W_excit.data, 'b_excit': self.b_excit.data,
            'W_inhib': self.W_inhib.data, 'b_inhib': self.b_inhib.data,
        }

    def clone(self):
        """轻量克隆：只复制遗传基因与基线，远快于 copy.deepcopy。

        运行时状态置零；基线引用共享（其中可能被原地修改的张量均为独立克隆）。
        """
        new = EIBrainRegion.__new__(EIBrainRegion)
        nn.Module.__init__(new)

        new.cfg = self.cfg
        new.N = self.N
        new.obs_dim = self.obs_dim
        new.action_dim = self.action_dim
        new.train_hormone = self.train_hormone

        new.M_in = self.M_in.clone()
        new.M_rec = self.M_rec.clone()
        new.M_out = self.M_out.clone()

        new.W_in = nn.Parameter(self.W_in.data.clone())
        new.W_rec = nn.Parameter(self.W_rec.data.clone())
        new.W_out = nn.Parameter(self.W_out.data.clone())
        new.b_out = nn.Parameter(self.b_out.data.clone())
        new.tau_e_init = nn.Parameter(self.tau_e_init.data.clone())
        new.w_ei = nn.Parameter(self.w_ei.data.clone())
        new.w_ie = nn.Parameter(self.w_ie.data.clone())
        new.V = nn.Parameter(self.V.data.clone())
        new.b_v = nn.Parameter(self.b_v.data.clone())

        new.W_hormone1 = nn.Parameter(self.W_hormone1.data.clone())
        new.b_hormone1 = nn.Parameter(self.b_hormone1.data.clone())
        new.W_excit = nn.Parameter(self.W_excit.data.clone())
        new.b_excit = nn.Parameter(self.b_excit.data.clone())
        new.W_inhib = nn.Parameter(self.W_inhib.data.clone())
        new.b_inhib = nn.Parameter(self.b_inhib.data.clone())

        new.register_buffer('hormone_excit', torch.zeros(self.N))
        new.register_buffer('hormone_inhib', torch.zeros(self.N))
        new.register_buffer('short_term_state', torch.zeros(self.N))
        new.register_buffer('consecutive_counts', torch.zeros(self.action_dim))
        new.register_buffer('last_excit_cmd', torch.zeros(self.N))
        new.register_buffer('last_inhib_cmd', torch.zeros(self.N))
        new.register_buffer('W_rec_eff', torch.zeros(self.N, self.N))
        new.register_buffer('W_out_eff', torch.zeros(self.action_dim, self.N))
        new.register_buffer('M_norm', torch.zeros(self.N, self.N))

        new.baseline = self.baseline

        new.refresh_cached()
        return new

    def restore_genetic_baseline(self):
        if self.baseline:
            self.W_in.data = self.baseline['W_in'].clone()
            self.W_rec.data = self.baseline['W_rec'].clone()
            self.W_out.data = self.baseline['W_out'].clone()
            self.b_out.data = self.baseline['b_out'].clone()
            self.M_in = self.baseline['M_in'].clone()
            self.M_rec = self.baseline['M_rec'].clone()
            self.M_out = self.baseline['M_out'].clone()
            self.tau_e_init.data = self.baseline['tau_e_init'].clone()
            self.w_ei.data = self.baseline['w_ei'].clone()
            self.w_ie.data = self.baseline['w_ie'].clone()
            self.V.data = self.baseline['V'].clone()
            self.b_v.data = self.baseline['b_v'].clone()
            self.W_hormone1.data = self.baseline['W_hormone1'].clone()
            self.b_hormone1.data = self.baseline['b_hormone1'].clone()
            self.W_excit.data = self.baseline['W_excit'].clone()
            self.b_excit.data = self.baseline['b_excit'].clone()
            self.W_inhib.data = self.baseline['W_inhib'].clone()
            self.b_inhib.data = self.baseline['b_inhib'].clone()
            self.refresh_cached()
