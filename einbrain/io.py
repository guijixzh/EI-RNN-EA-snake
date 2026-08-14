"""统一保存 / 加载（多格式兼容）。

格式兼容矩阵：
    - test5d 格式：10 项核心基因（M_*/W_*/b_out/tau_e_init/w_ei/w_ie/激素网络）
    - test6 格式：test5d + value head（V/b_v）
    - test7 格式：test5d 字段 + 激素字段补零（无激素）

load_brain_state 能读取以上全部格式（缺 V/b_v 时默认零初始化）。
模型文件路径统一经 model_path() 解析：优先绝对/相对路径，其次 <仓库>/models/。
"""
from __future__ import annotations

import os
import random
import time

import torch
import torch.nn as nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def model_path(name):
    """解析模型文件路径：优先给定路径，其次 <仓库>/models/<name>，最后 CWD。"""
    if os.path.isabs(name) or os.path.exists(name):
        return name
    cand = os.path.join(_REPO_ROOT, 'models', os.path.basename(name))
    if os.path.exists(cand):
        return cand
    return name


def config_dict(cfg):
    """收集生效配置为可序列化 dict（实例属性优先于类属性）。"""
    merged = {}
    merged.update(vars(cfg.__class__))
    merged.update(vars(cfg))
    return {k: v for k, v in merged.items() if not k.startswith('__')}


def save_brain_state(brain, use_half=True):
    """将单个个体的遗传属性打包为可序列化 dict（含 value head）。

    掩码转 uint8，权重转 fp16（进化噪声量级远大于 fp16 精度）；最优模型
    单独用 fp32 全精度保存。
    """
    dtype = torch.float16 if use_half else torch.float32
    with torch.no_grad():
        return {
            'N': brain.N,
            'M_in': brain.M_in.to(torch.uint8),
            'M_rec': brain.M_rec.to(torch.uint8),
            'M_out': brain.M_out.to(torch.uint8),
            'W_in': brain.W_in.data.to(dtype),
            'W_rec': brain.W_rec.data.to(dtype),
            'W_out': brain.W_out.data.to(dtype),
            'b_out': brain.b_out.data.to(dtype),
            'tau_e_init': brain.tau_e_init.data.to(dtype),
            'w_ei': brain.w_ei.data.to(dtype),
            'w_ie': brain.w_ie.data.to(dtype),
            'W_hormone1': brain.W_hormone1.data.to(dtype),
            'b_hormone1': brain.b_hormone1.data.to(dtype),
            'W_excit': brain.W_excit.data.to(dtype),
            'b_excit': brain.b_excit.data.to(dtype),
            'W_inhib': brain.W_inhib.data.to(dtype),
            'b_inhib': brain.b_inhib.data.to(dtype),
            'V': brain.V.data.to(dtype),
            'b_v': brain.b_v.data.to(dtype),
        }


def load_brain_state(state, cfg):
    """从 save_brain_state 的 dict 重建 EIBrainRegion 个体。

    兼容 test5d/6/7 三种格式：缺 V/b_v 时默认零初始化。
    """
    from .brain import EIBrainRegion

    new = EIBrainRegion.__new__(EIBrainRegion)
    nn.Module.__init__(new)

    new.cfg = cfg
    new.N = int(state['N'])
    new.obs_dim = cfg.OBS_DIM
    new.action_dim = cfg.ACTION_DIM
    new.train_hormone = bool(getattr(cfg, 'TRAIN_HORMONE_NET', False))

    new.M_in = state['M_in'].float()
    new.M_rec = state['M_rec'].float()
    new.M_out = state['M_out'].float()

    new.W_in = nn.Parameter(state['W_in'].float())
    new.W_rec = nn.Parameter(state['W_rec'].float())
    new.W_out = nn.Parameter(state['W_out'].float())
    new.b_out = nn.Parameter(state['b_out'].float())
    new.tau_e_init = nn.Parameter(state['tau_e_init'].float())
    new.w_ei = nn.Parameter(state['w_ei'].float())
    new.w_ie = nn.Parameter(state['w_ie'].float())

    new.W_hormone1 = nn.Parameter(state['W_hormone1'].float())
    new.b_hormone1 = nn.Parameter(state['b_hormone1'].float())
    new.W_excit = nn.Parameter(state['W_excit'].float())
    new.b_excit = nn.Parameter(state['b_excit'].float())
    new.W_inhib = nn.Parameter(state['W_inhib'].float())
    new.b_inhib = nn.Parameter(state['b_inhib'].float())

    new.V = nn.Parameter(state.get('V', torch.zeros(new.N)).float())
    new.b_v = nn.Parameter(state.get('b_v', torch.zeros(1)).float())

    new.register_buffer('hormone_excit', torch.zeros(new.N))
    new.register_buffer('hormone_inhib', torch.zeros(new.N))
    new.register_buffer('short_term_state', torch.zeros(new.N))
    new.register_buffer('consecutive_counts', torch.zeros(new.action_dim))
    new.register_buffer('last_excit_cmd', torch.zeros(new.N))
    new.register_buffer('last_inhib_cmd', torch.zeros(new.N))
    new.register_buffer('W_rec_eff', torch.zeros(new.N, new.N))
    new.register_buffer('W_out_eff', torch.zeros(new.action_dim, new.N))
    new.register_buffer('M_norm', torch.zeros(new.N, new.N))

    new.baseline = None

    new.refresh_cached()
    new.save_genetic_baseline()
    return new


def _size_matches(saved_cfg, cfg):
    """规模不匹配的断点/模型不能复用于当前代码。"""
    if not saved_cfg:
        return True
    return (saved_cfg.get('NUM_COLUMNS') == cfg.NUM_COLUMNS and
            saved_cfg.get('OBS_DIM') == cfg.OBS_DIM and
            saved_cfg.get('ACTION_DIM') == cfg.ACTION_DIM)


def save_best_model(path, brain, cfg, food, steps):
    """保存最优个体（fp32 全精度，体积小，便于直接加载复用）。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save({
        'brain': save_brain_state(brain, use_half=False),
        'food': float(food),
        'steps': float(steps),
        'config': config_dict(cfg),
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }, path)


def load_best_model_brain(path, cfg):
    """从最优模型文件恢复 (brain, food, steps)；不可用时返回 None。"""
    path = model_path(path)
    if not os.path.exists(path):
        return None
    try:
        data = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"警告: 最优模型 {path} 读取失败 ({e})，已忽略种子")
        return None
    if not _size_matches(data.get('config', {}), cfg):
        print(f"警告: 最优模型 {path} 与当前配置不匹配 (N/OBS/ACTION_DIM)，已忽略种子")
        return None
    brain = load_brain_state(data['brain'], cfg)
    return brain, float(data.get('food', -1.0)), float(data.get('steps', 0.0))


def save_checkpoint(path, cfg, next_gen, population, history,
                    cum_eval_time=0.0, cum_evolve_time=0.0,
                    best_brain=None, best_food=-1.0, best_steps=0.0):
    """保存训练断点：先写 .tmp 临时文件，再原子替换正式文件。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {
        'next_gen': next_gen,
        'population': [save_brain_state(ind) for ind in population],
        'history': history,
        'cum_eval_time': float(cum_eval_time),
        'cum_evolve_time': float(cum_evolve_time),
        'best_brain': save_brain_state(best_brain) if best_brain is not None else None,
        'best_food': float(best_food),
        'best_steps': float(best_steps),
        'config': config_dict(cfg),
        'random_state': random.getstate(),
        'torch_rng_state': torch.get_rng_state(),
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    tmp_path = path + '.tmp'
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    print(f"  [Checkpoint] 断点已保存 -> {path} (next_gen={next_gen})")


def load_checkpoint(path, cfg):
    """读取断点；文件不存在或配置不兼容时返回 None。"""
    if not os.path.exists(path):
        return None
    data = torch.load(path, map_location='cpu', weights_only=False)
    if not _size_matches(data.get('config', {}), cfg):
        print(f"警告: 断点 {path} 与当前配置不匹配 (N/OBS/ACTION_DIM)，已忽略")
        return None
    population = [load_brain_state(s, cfg) for s in data['population']]
    best_brain = (load_brain_state(data['best_brain'], cfg)
                  if data.get('best_brain') is not None else None)
    random.setstate(data['random_state'])
    torch.set_rng_state(data['torch_rng_state'])
    return {
        'next_gen': int(data['next_gen']),
        'population': population,
        'history': data['history'],
        'cum_eval_time': float(data.get('cum_eval_time', 0.0)),
        'cum_evolve_time': float(data.get('cum_evolve_time', 0.0)),
        'best_brain': best_brain,
        'best_food': float(data.get('best_food', -1.0)),
        'best_steps': float(data.get('best_steps', 0.0)),
    }
