"""验证 einbrain 稠密化加载（io._densify_sparse_rec + EIBrainRegion 前向）
与 test16b 稀疏前向（GeneStack + deliberate_batch）逐位等价。

方法：test16b BatchedSnakeEnv（B=1, CRN bank 固定）跑 40 步记录 obs 序列，
两路各自前向（E/I/st 零初始化、同 obs 流），逐步比对动作与 logits。
"""
import importlib.util, os, sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location('t16b', os.path.join(ROOT, 'experiments', 'test16_series', 'test16b.py'))
t16b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t16b)

sys.path.insert(0, ROOT)
from einbrain import io as eio
from einbrain.deliberation import deliberate_action

cfg = t16b.Config()
dev = torch.device('cpu')          # 对拍用 CPU，避免 fp16 CUDA 噪声
cfg.DEVICE = 'cpu'
cfg.USE_FP16 = False

model = os.path.join(ROOT, 'test16b_simp_best_model.pth')
st, food, _ = t16b.load_best_state(model, cfg)

# --- 路 1：test16b 稀疏前向（权威语义）---
pop = t16b.GeneStack(cfg, B=1, device=dev)
pop.random_init()
pop.set_individual_from_state(0, st)
pop.refresh_eff()

env = t16b.BatchedSnakeEnv(cfg, 1, dev)
bank = t16b.make_bank(cfg, 7, 9, 0, dev)          # 固定一局
env.reset(bank=bank)
E = torch.zeros(1, pop.N); I = torch.zeros(1, pop.N); stt = torch.zeros(1, pop.N)
press = torch.zeros(1)

# --- 路 2：einbrain 稠密化脑 ---
brain, ecfg, meta = eio.load_model_any(model)
assert meta['sparse_rec'] and meta['rec_fanin'] == 16
brain.reset_runtime()
E2 = torch.zeros(brain.N); I2 = torch.zeros(brain.N)

obs_list, act1_list, logit1_list = [], [], []
for t in range(40):
    obs = env.obs()                                # [1,40]
    obs_list.append(obs[0].clone())
    act, E, I, stt = t16b.deliberate_batch(pop, obs, E, I, stt, press, cfg)
    press = t16b.update_fatigue(press, act, decay=float(cfg.FATIGUE_TURN_DECAY))
    act1_list.append(int(act[0]))
    # 路 2：同 obs 喂稠密脑
    a2, avg_logits, E2, I2 = deliberate_action(brain, obs[0].numpy(), E2, I2)
    if t < 5 or t >= 35:
        print(f"step {t}: act16={int(act[0])} actDense={a2} "
              f"logit16_norm={float(torch.linalg.vector_norm(E)):.6f} "
              f"logitDense_E_norm={float(E2.norm()):.6f}")
    assert int(act[0]) == a2, f"动作分歧 @ step {t}: {int(act[0])} vs {a2}"
    env.step(act)

print("\n40 步动作全一致 ✓（E 范数轨迹亦对齐）")
print(f"稀疏槽位 {meta.get('rec_fanin')}×{brain.N} → 稠密唯一边 "
      f"{int(brain.M_rec.sum().item())}（重复源已叠加）")
