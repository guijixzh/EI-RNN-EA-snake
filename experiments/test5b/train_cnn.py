"""CNN 预训练：从专家数据集蒸馏 (网格 -> 动作分布)。

损失:
    L = alpha * KL(soft_label || softmax(logits))
      + (1-alpha) * CE(hard_label, logits)

用法:
    python train_cnn.py --data dataset.npz --epochs 5 --alpha 0.7
                        --out cnn_encoder.pth --batch 1024

输出 cnn_encoder.pth:
    { 'model_state': CNNEncoder.state_dict(), 'train_loss': float, ... }
"""
import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cnn import CNNEncoder
from collect_data import load_dataset


def train_cnn(data_path, epochs=5, alpha=0.7, batch=1024, lr=1e-3,
              out_path='cnn_encoder.pth', seed=42, val_frac=0.05):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] device = {device}")

    # ---- 加载数据 ----
    print(f"[Train] 加载 {data_path} ...", flush=True)
    t0 = time.perf_counter()
    d = load_dataset(data_path)
    grids = d['grids']        # (N,10,10,5) float32
    softs = d['softs']        # (N,3) float32（已 softmax）
    actions = d['actions']    # (N,) int64
    N = len(grids)
    print(f"[Train] 样本 {N} | 用时 {time.perf_counter()-t0:.1f}s", flush=True)

    # ---- 校验数据集统计 ----
    print(f"[Train] 数据集动作分布: "
          f"Fwd {np.mean(actions==0)*100:.1f}% / "
          f"Left {np.mean(actions==1)*100:.1f}% / "
          f"Right {np.mean(actions==2)*100:.1f}%", flush=True)

    # ---- 划分 train / val ----
    idx = np.random.permutation(N)
    n_val = max(1, int(N * val_frac))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    print(f"[Train] train {len(train_idx)} / val {len(val_idx)}", flush=True)

    grids_t = torch.from_numpy(grids[train_idx])
    softs_t = torch.from_numpy(softs[train_idx])
    acts_t = torch.from_numpy(actions[train_idx])
    grids_v = torch.from_numpy(grids[val_idx])
    softs_v = torch.from_numpy(softs[val_idx])
    acts_v = torch.from_numpy(actions[val_idx])

    # ---- 模型 ----
    model = CNNEncoder()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] CNNEncoder 参数量 = {n_params:,}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_loss = float('inf')
    history = []

    for ep in range(epochs):
        model.train()
        t_ep = time.perf_counter()
        total_loss = 0.0
        n_batch = 0

        # 洗牌
        perm = torch.randperm(len(train_idx))
        for i in range(0, len(perm), batch):
            idx_b = perm[i:i + batch]
            gb = grids_t[idx_b].to(device)
            sb = softs_t[idx_b].to(device)
            ab = acts_t[idx_b].to(device)

            logits_a, _ = model.forward_train(gb)  # (B, 3) 独立动作头

            # KL 蒸馏损失（软标签）
            log_probs = F.log_softmax(logits_a, dim=-1)
            kl = F.kl_div(log_probs, sb, reduction='batchmean')

            # 交叉熵（硬标签）
            ce = F.cross_entropy(logits_a, ab)

            loss = alpha * kl + (1 - alpha) * ce

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_batch += 1

        sched.step()

        # 验证
        model.eval()
        with torch.no_grad():
            gv = grids_v.to(device)
            sv = softs_v.to(device)
            av = acts_v.to(device)
            val_logits, _ = model.forward_train(gv)
            val_kl = F.kl_div(F.log_softmax(val_logits, dim=-1), sv,
                              reduction='batchmean').item()
            val_ce = F.cross_entropy(val_logits, av).item()
            val_loss = alpha * val_kl + (1 - alpha) * val_ce
            pred = val_logits.argmax(dim=-1)
            acc = (pred == av).float().mean().item()

        avg_loss = total_loss / max(1, n_batch)
        history.append({'epoch': ep, 'train_loss': avg_loss,
                        'val_loss': val_loss, 'val_kl': val_kl,
                        'val_ce': val_ce, 'val_acc': acc})

        print(f"[Train] epoch {ep+1}/{epochs} | "
              f"train_loss {avg_loss:.4f} | "
              f"val_loss {val_loss:.4f} (kl {val_kl:.3f} / ce {val_ce:.3f}) | "
              f"val_acc {acc*100:.1f}% | "
              f"{time.perf_counter()-t_ep:.1f}s", flush=True)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'model_state': model.state_dict(),
                'epoch': ep,
                'best_val_loss': best_loss,
                'history': history,
            }, out_path)
            print(f"[Train] 已保存最优 -> {out_path}", flush=True)

    # ---- 最终报告 ----
    print(f"\n[Train] 完成！耗时 {time.perf_counter()-t0:.1f}s")
    print(f"[Train] 最优 val_loss {best_loss:.4f} | 保存于 {out_path}")
    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='dataset.npz')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='KL 蒸馏损失权重（1-alpha 为交叉熵权重）')
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--out', type=str, default='cnn_encoder.pth')
    args = parser.parse_args()
    train_cnn(args.data, args.epochs, args.alpha, args.batch, args.lr,
              args.out)