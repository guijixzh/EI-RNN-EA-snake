# ==========================================
# test6_smoke_test.py —— test6.py（K=5）运行时冒烟测试
# 验证：
#   1. SnakeEnv 环境 reset/step、obs 维度、(next_obs, ate, done, truncated) 返回
#   2. 八桶自体感知输入"按自身长度归一化"改造生效
#   3. forward_ppo 单步可微 + forward_ppo_k（K=5 思考）可微
#   4. K=5 rollout → 初始状态 buffer → GAE → PPO 单轮更新（同口径重放）
#   5. 种子加载（test5d_best_model.pth）
# ==========================================

import torch
import numpy as np
import importlib.util
import time

# 导入 test6（不触发主循环，因为 __name__ != '__main__'）
spec = importlib.util.spec_from_file_location("test6", "test6.py")
test6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test6)

Cfg = test6.Config


def test_env():
    print("=== Test 1: SnakeEnv ===")
    env = test6.SnakeEnv(grid_size=10, max_steps=500)
    obs = env.reset()
    assert obs.shape == (24,), f"obs shape 应为 (24,)，实际 {obs.shape}"
    assert obs.dtype == np.float32, f"obs dtype 应为 float32，实际 {obs.dtype}"

    # 跑 50 步随机动作，验证返回协议
    for _ in range(50):
        a = np.random.randint(0, 3)
        o2, ate, done, truncated = env.step(a)
        assert o2.shape == (24,)
        assert isinstance(ate, bool)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert not (done and truncated), "done 与 truncated 不应同时为 True"
        if done or truncated:
            env.reset()
            break
    print("  [PASS] env step/reset 协议正常")
    assert obs.min() >= -1.0 and obs.max() <= 1.0, f"obs 值域异常 [{obs.min()}, {obs.max()}]"
    print(f"  obs 值域: [{obs.min():.3f}, {obs.max():.3f}]")


def test_self_bucket_length_norm():
    print("=== Test 2: 八桶按自身长度归一化 ===")
    env = test6.SnakeEnv(grid_size=10, max_steps=500)
    # 短蛇：1 节身体，在蛇头左侧 1 格
    env.head = (5, 5)
    env.dir = (0, 1)
    env.body = [(5, 5), (4, 5)]
    env.food = (9, 9)
    obs_short = env._get_obs()

    # 长蛇：10 节身体，向左侧延伸
    env.head = (5, 5)
    env.dir = (0, 1)
    body = [(5, 5)]
    for i in range(1, 11):
        body.append((5 - i, 5))
    env.body = body
    env.food = (9, 9)
    obs_long = env._get_obs()

    # bucket 方向：蛇头朝下(0,1)，左侧=西(-1,0)
    # ang = atan2(1, 0) = 90° → bucket = int((90+22.5)//45) = 2
    print(f"  短蛇(1节) left-bucket: {obs_short[14 + 2]:.3f} | 长蛇(10节) left-bucket: {obs_long[14 + 2]:.3f}")
    assert obs_short[14 + 2] >= 0.0 and obs_long[14 + 2] >= 0.0
    # 自身长度归一化：长蛇最近节（距离 1 / 长度 10）近端度高于短蛇（距离 1 / 长度 1）
    assert obs_long[14 + 2] > obs_short[14 + 2], "长蛇近端度应高于短蛇（自身长度归一化）"
    print("  [PASS] 八桶归一化按自身长度生效（长蛇近端度 > 短蛇）")


def test_forward_differentiable():
    print("=== Test 3: forward_ppo 与 forward_ppo_k(K=5) 可微 ===")
    cfg = Cfg()
    cfg.N_ENVS = 4
    brain = test6.EIBrainRegion(cfg)

    B = 4
    E, I, short, he, hi, counts = test6.make_zero_states(brain, B)
    obs = np.random.randn(B, 24).astype(np.float32)
    obs_t = torch.from_numpy(obs)

    # ---- 单步 forward_ppo 可微 ----
    logits, val, _, _, _, _, _ = brain.forward_ppo(obs_t, E, I, short, he, hi, counts)
    assert logits.shape == (B, 3)
    assert val.shape == (B,)
    loss1 = (logits ** 2).mean() + (val ** 2).mean()
    loss1.backward()
    assert brain.W_in.grad is not None, "W_in 无梯度"
    assert brain.W_rec.grad is not None, "W_rec 无梯度"
    assert brain.tau_e_init.grad is not None, "tau_e_init 无梯度"
    assert brain.V.grad is not None, "V 无梯度"
    brain.zero_grad()

    # ---- K=5 思考（forward_ppo_k）可微 ----
    brain.zero_grad()
    logits_k, val_k, E_f, I_f, short_f, he_f, hi_f = test6.forward_ppo_k(
        brain, obs_t, E, I, short, he, hi, counts, K=5, decay=0.9)
    assert logits_k.shape == (B, 3)
    assert val_k.shape == (B,)
    loss_k = (logits_k ** 2).mean() + (val_k ** 2).mean()
    loss_k.backward()
    assert brain.W_in.grad is not None, "forward_ppo_k: W_in 无梯度"
    assert brain.W_rec.grad is not None, "forward_ppo_k: W_rec 无梯度"
    assert brain.tau_e_init.grad is not None, "forward_ppo_k: tau_e_init 无梯度"
    assert brain.V.grad is not None, "forward_ppo_k: V 无梯度"
    print(f"  logits[{logits.shape}], K=5 logits[{logits_k.shape}], 两条前向均可微")
    print("  [PASS] forward_ppo + forward_ppo_k(K=5) 可微 + value head 正常")


def test_ppo_k5_update():
    print("=== Test 4: K=5 rollout → GAE → PPO 单轮更新 ===")
    cfg = Cfg()
    cfg.N_ENVS = 4
    cfg.ROLLOUT_LEN = 8
    cfg.MINIBATCH_SIZE = 8
    cfg.PPO_EPOCHS = 2
    # 显式 K=5（与 Config 默认一致，明确验证）
    cfg.FRAME_RATE = 5
    cfg.INPUT_DECAY = 0.9

    brain = test6.EIBrainRegion(cfg)
    params = test6.trainable_parameters(brain, cfg)
    optimizer = torch.optim.Adam(params, lr=cfg.LR)

    B = cfg.N_ENVS
    N = brain.N
    T = cfg.ROLLOUT_LEN
    K = cfg.FRAME_RATE

    envs = [test6.SnakeEnv(grid_size=cfg.GRID_SIZE, max_steps=cfg.MAX_STEPS) for _ in range(B)]
    obs_stack = np.stack([e.reset() for e in envs], axis=0).astype(np.float32)
    E, I, short, he, hi, counts = test6.make_zero_states(brain, B)

    obs_buf = torch.zeros(T, B, 24, dtype=torch.float32)
    act_buf = torch.zeros(T, B, dtype=torch.long)
    logp_buf = torch.zeros(T, B, dtype=torch.float32)
    val_buf = torch.zeros(T, B, dtype=torch.float32)
    rew_buf = torch.zeros(T, B, dtype=torch.float32)
    mask_buf = torch.zeros(T, B, dtype=torch.float32)
    E0_buf = torch.zeros(T, B, N)
    I0_buf = torch.zeros(T, B, N)
    short0_buf = torch.zeros(T, B, N)
    counts0_buf = torch.zeros(T, B, 3)
    he0_buf = None
    hi0_buf = None

    from torch.distributions import Categorical

    # ---- K=5 rollout ----
    for t in range(T):
        obs_t = torch.from_numpy(obs_stack)
        # 记录初始状态（供 ppo_update 重放 K 次思考）
        E0_buf[t] = E
        I0_buf[t] = I
        short0_buf[t] = short
        counts0_buf[t] = counts

        with torch.no_grad():
            logits, val, E, I, short, he, hi = test6.forward_ppo_k(
                brain, obs_t, E, I, short, he, hi, counts, K=K, decay=cfg.INPUT_DECAY)

        dist = Categorical(logits=logits)
        act = dist.sample()
        logp = dist.log_prob(act)

        obs_buf[t] = obs_t
        act_buf[t] = act
        logp_buf[t] = logp.detach()
        val_buf[t] = val.detach()

        seen_mask = test6._obs_sees_food(obs_stack)
        step_rew = np.where(seen_mask, cfg.SEEN_STEP_REWARD, cfg.UNSEEN_STEP_REWARD)
        counts = test6.update_counts(counts, act)

        act_np = act.numpy()
        next_obs = np.zeros_like(obs_stack)
        for i in range(B):
            o2, ate, done, tr = envs[i].step(int(act_np[i]))
            r = float(step_rew[i])
            if ate:
                r += cfg.EAT_REWARD
            if done or tr:
                r += cfg.DEATH_REWARD
            next_obs[i] = o2
            rew_buf[t, i] = r
            mask_buf[t, i] = 0.0 if done else 1.0
            if done or tr:
                next_obs[i] = envs[i].reset()
                E[i].zero_(); I[i].zero_(); short[i].zero_(); counts[i].zero_()
        obs_stack = next_obs

    with torch.no_grad():
        obs_t = torch.from_numpy(obs_stack)
        _, last_val, _, _, _, _, _ = test6.forward_ppo_k(
            brain, obs_t, E, I, short, he, hi, counts, K=K, decay=cfg.INPUT_DECAY)

    # ---- GAE ----
    adv_buf, ret_buf = test6.compute_gae(rew_buf, val_buf, mask_buf, last_val, cfg)
    assert adv_buf.shape == (T, B) and ret_buf.shape == (T, B)

    # ---- PPO 更新（新签名：初始状态 buffer）----
    p_loss, v_loss, ent = test6.ppo_update(
        brain, optimizer, cfg,
        obs_buf, act_buf, logp_buf, adv_buf, ret_buf,
        E0_buf, I0_buf, short0_buf, he0_buf, hi0_buf, counts0_buf)

    assert np.isfinite(p_loss), f"policy loss 非有限值: {p_loss}"
    assert np.isfinite(v_loss), f"value loss 非有限值: {v_loss}"
    assert np.isfinite(ent), f"entropy 非有限值: {ent}"
    print(f"  K={K} rollout→GAE→PPO: policy_loss={p_loss:.4f}, value_loss={v_loss:.4f}, entropy={ent:.4f}")
    print("  [PASS] K=5 全链路正常")


def test_seed_loading():
    print("=== Test 5: 种子加载 + K=5 口径重评 ===")
    cfg = Cfg()
    cfg.FRAME_RATE = 5
    result = test6.load_best_model_brain(cfg.TEST5D_MODEL_PATH, cfg)
    if result is None:
        print(f"  [SKIP] 未找到 {cfg.TEST5D_MODEL_PATH}，跳过")
        return
    brain, food, steps = result
    assert brain.N == cfg.NUM_COLUMNS

    # K=5 前向可跑通
    E, I, short, he, hi, counts = test6.make_zero_states(brain, 2)
    obs = np.random.randn(2, 24).astype(np.float32)
    logits, val, *_ = test6.forward_ppo_k(
        brain, torch.from_numpy(obs), E, I, short, he, hi, counts, K=cfg.FRAME_RATE)
    assert logits.shape == (2, 3) and val.shape == (2,)

    # K=5 重评基线（诊断打印；不阻塞）
    env = test6.SnakeEnv(grid_size=cfg.GRID_SIZE, max_steps=cfg.MAX_STEPS)
    base_food, base_steps = test6.evaluate(brain, env, cfg, episodes=3)
    # 也用 K=1 探针对比（诊断）
    k1_food = test6.evaluate_k1_probe(brain, env, cfg, episodes=3)
    print(f"  文件标注 Food={food:.1f}（K=5+旧编码口径）| "
          f"test6 K=5 重评: {base_food:.1f} | K=1 探针: {k1_food:.1f}")
    print("  [PASS] 种子加载 + K=5 前向 + 重评基线正常")


if __name__ == "__main__":
    t0 = time.perf_counter()
    test_env()
    test_self_bucket_length_norm()
    test_forward_differentiable()
    test_ppo_k5_update()
    test_seed_loading()
    print(f"\n全部冒烟测试通过，耗时 {time.perf_counter() - t0:.1f}s")