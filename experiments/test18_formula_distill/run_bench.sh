#!/bin/bash
# test18 M2 统一评测：5 臂 best + 7b 源网络 + V5 公式参照（starve=3，100 局）
PY=/d/Anaconda/envs/env_torch/python.exe
D=experiments/test18_formula_distill
$PY $D/exp18_benchmark.py \
  --model $D/test18a_fd_scratch_best_model.pth \
  --model $D/test18a_fd_7b_best_model.pth \
  --model $D/test18a_fd_7b_anneal_best_model.pth \
  --model $D/test18a_ctrl_simple_7b_best_model.pth \
  --model $D/test18a_ctrl_simple_scratch_best_model.pth \
  --model test7b_base_model.pth \
  --episodes 100 --seed 20260913 \
  --out results/test18_formula_distill/benchmark_final.json
