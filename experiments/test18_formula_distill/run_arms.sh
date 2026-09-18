#!/bin/bash
# test18 公式反哺 5 臂实验矩阵（顺序跑，RTX 5070 单卡）
PY=/d/Anaconda/envs/env_torch/python.exe
COMMON="--obs 32proj --pop 256 --elite 64 --stage2-keep 128 --stage1-eps 4 --stage2-eps 8 --stage2a-eps 2 --stage2b-eps 6 --gens 100 --seed 42 --starve-slope 3"
BANK="--fit-mode formula --formula-bank formula_bank_32proj_v1.pt"

echo "=== [$(date +%H:%M:%S)] fd_scratch ==="
$PY test18a_formula.py $COMMON $BANK --name fd_scratch > ../../logs/test18_fd_scratch.log 2>&1
echo "=== [$(date +%H:%M:%S)] fd_7b ==="
$PY test18a_formula.py $COMMON $BANK --seed-model ../../test7b_base_model.pth --seed-pop --name fd_7b > ../../logs/test18_fd_7b.log 2>&1
echo "=== [$(date +%H:%M:%S)] fd_7b_anneal ==="
$PY test18a_formula.py $COMMON $BANK --seed-model ../../test7b_base_model.pth --seed-pop --formula-anneal-gen 60 --name fd_7b_anneal > ../../logs/test18_fd_7b_anneal.log 2>&1
echo "=== [$(date +%H:%M:%S)] ctrl_simple_7b ==="
$PY test18a_formula.py $COMMON --fit-mode simple --seed-model ../../test7b_base_model.pth --seed-pop --name ctrl_simple_7b > ../../logs/test18_ctrl_simple_7b.log 2>&1
echo "=== [$(date +%H:%M:%S)] ctrl_simple_scratch ==="
$PY test18a_formula.py $COMMON --fit-mode simple --name ctrl_simple_scratch > ../../logs/test18_ctrl_simple_scratch.log 2>&1
echo "=== [$(date +%H:%M:%S)] ALL DONE ==="
