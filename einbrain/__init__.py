"""einbrain —— 经过考验有效的 EI-RNN（E-I 皮质柱脑区）整合通用包。

整合三套已过考验的训练范式，共用同一脑模型（EIBrainRegion）：
    - evolve.evolution：      CPU 进化训练（test5d 语义，多进程 + 两阶段筛选）
    - ppo.run_training：      PPO 强化学习（test6 语义，K 帧思考 + 截断 BPTT）
    - gpu.run_training_gpu：  GPU 全并行进化（test7 语义，GeneStack 张量化）

模型文件（test5d/6/7 三种格式）统一经 io.load_best_model_brain / io.model_path
读取，可互相作为种子注入。

快速上手：
    from einbrain import Config, run_evolution
    cfg = Config()
    best_brain, history = run_evolution(cfg)
"""
from .config import Config, make_smoke_config
from .env import SnakeEnv, _obs_sees_food
from .brain import EIBrainRegion
from .dynamics import (clamp_w, hormone_commands, diffuse_decay,
                       effective_tau_e, ei_update, fatigue_penalty)
from .deliberation import deliberate_action
from . import io
from .evolve import (evaluate_individual, evolve_topology, evaluate_population,
                     run_evolution, _selection_key, _freeze_active_groups)
from .ppo import (forward_ppo_k, update_counts, make_zero_states,
                  trainable_parameters, compute_gae, ppo_update, evaluate,
                  run_training)
from . import vis

__all__ = [
    'Config', 'make_smoke_config',
    'SnakeEnv', '_obs_sees_food',
    'EIBrainRegion',
    'clamp_w', 'hormone_commands', 'diffuse_decay',
    'effective_tau_e', 'ei_update', 'fatigue_penalty',
    'deliberate_action',
    'io',
    'evaluate_individual', 'evolve_topology', 'evaluate_population',
    'run_evolution', '_selection_key', '_freeze_active_groups',
    'forward_ppo_k', 'update_counts', 'make_zero_states',
    'trainable_parameters', 'compute_gae', 'ppo_update', 'evaluate', 'run_training',
    'vis',
]

__version__ = '0.1.0'
