# 用法: python experiments/run_seeded.py <script.py> <seed> [args...]（仓库根目录运行）
# 给无 --seed 的旧脚本（如 experiments/test16_series/test16a.py）注入 CPU+CUDA 随机种子后运行 main()，
# 用于与 test16b --seed 的 A/B 同种子对照。脚本须以 __main__ 守卫调用 main()。
import importlib.util
import random
import sys
import torch

script, seed, *rest = sys.argv[1:]
spec = importlib.util.spec_from_file_location('_seeded_target', script)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
seed = int(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
sys.argv = [script] + rest
mod.main()
