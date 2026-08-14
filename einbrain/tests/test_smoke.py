"""einbrain 包自检脚本。

覆盖：
    1) 三种历史模型格式（test5d / test6 / test7）统一加载 + 游玩
    2) GPU 批量前向 == 单脑前向 严格等价（零激素）
    3) 三种训练管线小规模跑通：run_evolution / run_training / run_training_gpu

运行：
    python einbrain/tests/test_smoke.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402

from einbrain import Config, SnakeEnv, deliberate_action, io, make_smoke_config  # noqa: E402


def _pass(name):
    print(f"[PASS] {name}")


def test_model_loading():
    cfg = Config()
    for name in ['test5d_best_model.pth', 'test6_best_model.pth', 'test7_best_model.pth']:
        res = io.load_best_model_brain(name, cfg)
        assert res is not None, f"{name} 加载失败"
        brain, _, _ = res
        env = SnakeEnv(grid_size=cfg.GRID_SIZE)
        obs = env.reset()
        E = torch.zeros(brain.N)
        I = torch.zeros(brain.N)
        done = False
        steps = 0
        for _ in range(60):
            a, _, E, I = deliberate_action(brain, obs, E, I)
            obs, ate, d, t = env.step(a)
            done = d or t
            steps += 1
            if done:
                break
        assert steps > 0
        _pass(f"加载+游玩 {name}（N={brain.N}，跑 {steps} 步）")


def test_gpu_forward_equivalence():
    from einbrain.gpu import GeneStack, forward_batch, update_fatigue

    cfg = Config()
    cfg.TRAIN_HORMONE_NET = False
    cfg.USE_FP16 = False
    torch.manual_seed(123)
    brain = __import__('einbrain').EIBrainRegion(cfg)
    brain.save_genetic_baseline()
    brain.reset_runtime()

    pop = GeneStack(cfg, B=1, device=torch.device('cpu'))
    pop.random_init()
    pop.set_individual_from_state(0, io.save_brain_state(brain, use_half=False))
    pop.fp32()
    pop.refresh_eff()

    E1 = torch.zeros(1, cfg.NUM_COLUMNS)
    I1 = torch.zeros(1, cfg.NUM_COLUMNS)
    st1 = torch.zeros(1, cfg.NUM_COLUMNS)
    c1 = torch.zeros(1, 3)
    E2 = torch.zeros(cfg.NUM_COLUMNS)
    I2 = torch.zeros(cfg.NUM_COLUMNS)
    max_diff = 0.0
    for _ in range(30):
        obs = torch.rand(1, cfg.OBS_DIM)
        lg1, E2, I2 = brain(obs[0], E2, I2)
        a1 = int(torch.argmax(lg1).item())
        brain.update_fatigue(a1)
        lg2, E1, I1, st1 = forward_batch(pop, obs, E1, I1, st1, c1, cfg)
        a2 = int(torch.argmax(lg2[0]).item())
        assert a1 == a2
        c1 = update_fatigue(c1, torch.tensor([a2]))
        d = max((lg1 - lg2[0]).abs().max().item(),
                (E2 - E1[0]).abs().max().item(),
                (I2 - I1[0]).abs().max().item())
        max_diff = max(max_diff, d)
    assert max_diff < 1e-5, f"前向不等价: {max_diff}"
    _pass(f"GPU 批量前向 == 单脑前向（max diff = {max_diff}）")


def test_evolution_pipeline():
    from einbrain import run_evolution

    cfg = make_smoke_config()
    cfg.POP_SIZE = 8
    cfg.GENERATIONS = 2
    cfg.NUM_COLUMNS = 16
    cfg.MAX_STEPS = 50
    best, hist = run_evolution(cfg)
    assert len(hist['gen']) == 2
    _cleanup(cfg)
    _pass("run_evolution 小规模跑通")


def test_ppo_pipeline():
    from einbrain import run_training

    cfg = make_smoke_config()
    cfg.N_ENVS = 2
    cfg.ROLLOUT_LEN = 6
    cfg.TOTAL_ITERATIONS = 1
    cfg.MINIBATCH_SIZE = 16
    cfg.EVAL_INTERVAL = 1
    best, hist = run_training(cfg)
    assert len(hist['iter']) == 1
    _cleanup(cfg)
    _pass("run_training（PPO）小规模跑通")


def test_gpu_pipeline():
    from einbrain.gpu import run_training_gpu

    cfg = make_smoke_config()
    cfg.POP_SIZE = 8
    cfg.GENERATIONS = 2
    cfg.NUM_COLUMNS = 16
    cfg.MAX_STEPS = 50
    cfg.DEVICE = 'cpu'
    cfg.USE_FP16 = False
    best_state, hist = run_training_gpu(cfg)
    assert len(hist['gen']) == 2
    _cleanup(cfg)
    _pass("run_training_gpu 小规模跑通")


def _cleanup(cfg):
    for p in (cfg.CHECKPOINT_PATH, cfg.BEST_MODEL_PATH):
        if os.path.exists(p):
            os.remove(p)


if __name__ == '__main__':
    test_model_loading()
    test_gpu_forward_equivalence()
    test_evolution_pipeline()
    test_ppo_pipeline()
    test_gpu_pipeline()
    print("\n=== einbrain 自检全部通过 ===")
