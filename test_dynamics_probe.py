# test_dynamics_probe_v4.py
# 终极修正版：直接调用 test7b 原生 deliberate_batch，确保动力学完全一致

import os
import sys
import copy
import argparse
import importlib.util
import torch
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# 导入 test7b 模块
_spec = importlib.util.spec_from_file_location('t7b', os.path.join(ROOT, 'test7b.py'))
t7b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t7b)

def load_pop(model_path):
    """加载模型为 test7b 的 GeneStack 格式"""
    cfg = t7b.Config()
    res = t7b.load_best_state(model_path, cfg)
    if res is None: raise RuntimeError("无法加载模型")
    st, _, _ = res
    dev = torch.device("cpu")
    
    pop = t7b.GeneStack(cfg, B=1, device=dev)
    pop.random_init()
    pop.set_individual_from_state(0, st)
    pop.fp32()  # 探针用 fp32 保证精度
    pop.refresh_eff()
    return pop, cfg

def run_probe_v4(model_path, warmup=800, follow_up=400):
    pop, cfg = load_pop(model_path)
    dev = torch.device("cpu")
    
    # 初始化环境和状态
    env = t7b.BatchedSnakeEnv(cfg, 1, dev)
    env.reset()
    E = torch.zeros(1, pop.N, dtype=torch.float32, device=dev)
    I = torch.zeros(1, pop.N, dtype=torch.float32, device=dev)
    st = torch.zeros(1, pop.N, dtype=torch.float32, device=dev)
    cts = torch.zeros(1, pop.A, dtype=torch.float32, device=dev)
    
    print(f"[探针V4] 正在进行 {warmup} 步热身跑 (使用 test7b 原生动力学 K={cfg.FRAME_RATE})...")
    with torch.no_grad():
        for t in range(warmup):
            obs = env.obs()
            # 【关键修复】直接调用 test7b 的原生思考函数
            act, E, I, st = t7b.deliberate_batch(pop, obs, E, I, st, cts, cfg)
            cts = t7b.update_fatigue(cts, act)
            env.step(act)
            if not env.alive[0]:
                print(f"[探针V4] 热身阶段死亡 (步数:{t}, 长度:{env.body_len[0].item()})，请减小 warmup 参数")
                return
                
    warmup_score = env.body_len[0].item() - 2
    print(f"[探针V4] 热身结束。当前蛇长: {warmup_score+2}。开始注入微扰并追踪后续 {follow_up} 步...")
    print("------------------------------------------------")
    
    # 保存原始权重
    W_rec_orig = pop.W_rec.clone()
    
    # 测试不同级别的微扰
    for sigma in [0.0, 1e-5, 1e-4, 1e-3]:
        # 深拷贝环境和脑状态，确保起点绝对一致
        env_p = copy.deepcopy(env)
        E_p, I_p, st_p, cts_p = E.clone(), I.clone(), st.clone(), cts.clone()
        
        # 注入扰动
        pop.W_rec = W_rec_orig + torch.randn_like(W_rec_orig) * sigma
        pop.refresh_eff()
        
        actions = []
        with torch.no_grad():
            for t in range(follow_up):
                obs = env_p.obs()
                act, E_p, I_p, st_p = t7b.deliberate_batch(pop, obs, E_p, I_p, st_p, cts_p, cfg)
                cts_p = t7b.update_fatigue(cts_p, act)
                env_p.step(act)
                actions.append(act.item())
                if not env_p.alive[0]:
                    break
                    
        final_len = env_p.body_len[0].item() - 2
        print(f"微扰 σ={sigma:.5f} | 最终得分: {final_len:3d} | 存活步数: {len(actions)}")
        
    # 恢复原始权重
    pop.W_rec = W_rec_orig
    pop.refresh_eff()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="test7b_latest_gen_best.pth")
    ap.add_argument("--warmup", type=int, default=800, help="热身步数，让蛇变长")
    ap.add_argument("--follow_up", type=int, default=400, help="注入扰动后的追踪步数")
    args = ap.parse_args()
    run_probe_v4(args.model, args.warmup, args.follow_up)