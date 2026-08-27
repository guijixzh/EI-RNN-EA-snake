"""test5d benchmark script - same parameters as test7 bench for fair comparison."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

import importlib.util
spec = importlib.util.spec_from_file_location('test5d', os.path.join(os.path.dirname(__file__), '..', 'experiments', 'test5d.py'))
test5d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test5d)

cfg = test5d.Config()
cfg.POP_SIZE = 2048
cfg.ELITE_SIZE = 256
cfg.GENERATIONS = 5
cfg.EVAL_EPISODES = 2
cfg.MAX_STEPS = 100
cfg.CHECKPOINT_PATH = 'test5d_bench_checkpoint.pth'
cfg.BEST_MODEL_PATH = 'test5d_bench_best.pth'
cfg.SEED_FROM_BEST = False
cfg.PRINT_HISTORY_EVERY = 1
cfg.AUTO_RESUME = False
cfg.SCREEN_ENABLE = False
cfg.PARALLEL_EVAL = False  # serial for fair comparison with test7

env = test5d.SnakeEnv(grid_size=cfg.GRID_SIZE)

population = [test5d.EIBrainRegion(cfg) for _ in range(cfg.POP_SIZE)]
for ind in population:
    ind.save_genetic_baseline()

eval_pool = None

t_program = time.perf_counter()
cum_eval_time = 0.0
cum_evolve_time = 0.0
best_ever_brain = None
best_ever_food = -1.0
best_ever_steps = 0.0
best_ever_seen = 0.0
best_ever_unseen = 0.0

for gen in range(cfg.GENERATIONS):
    t_eval = time.perf_counter()
    metrics = test5d.evaluate_population(population, cfg, env, eval_pool)
    eval_time = time.perf_counter() - t_eval
    cum_eval_time += eval_time

    threshold = float(getattr(cfg, 'LONG_SNAKE_SCORE_THRESHOLD', 25.0))
    best_idx = max(range(len(metrics)),
                   key=lambda i: test5d._selection_key(metrics[i], threshold))
    best_food = metrics[best_idx][0]
    best_seen = metrics[best_idx][1]
    best_unseen = metrics[best_idx][2]
    avg_food = sum(m[0] for m in metrics) / len(metrics)

    best_brain = population[best_idx]

    prev_seen = float(getattr(best_ever_brain, '_track_seen', best_ever_seen)) if best_ever_brain is not None else best_ever_seen
    prev_unseen = float(getattr(best_ever_brain, '_track_unseen', best_ever_unseen)) if best_ever_brain is not None else best_ever_unseen
    if (best_food > best_ever_food or
            (best_food == best_ever_food and
             ((best_food > threshold and best_unseen > prev_unseen) or
              (best_food <= threshold and best_seen < prev_seen)))):
        best_ever_food = best_food
        best_ever_steps = best_seen + best_unseen
        best_ever_seen = best_seen
        best_ever_unseen = best_unseen
        best_ever_brain = best_brain.clone()
        best_ever_brain._track_seen = best_seen
        best_ever_brain._track_unseen = best_unseen

    if gen < cfg.GENERATIONS - 1:
        t_ev = time.perf_counter()
        population = test5d.evolve_topology(population, metrics, cfg, gen=gen)
        evolve_time = time.perf_counter() - t_ev
        cum_evolve_time += evolve_time
    else:
        evolve_time = 0.0

    print(f"Gen {gen+1}/{cfg.GENERATIONS} | "
          f"BestFood: {best_food:.1f} | BestSeen: {best_seen:.1f} | BestUnseen: {best_unseen:.1f} | "
          f"AvgFood: {avg_food:.1f} | "
          f"eval {eval_time:.1f}s / evolve {evolve_time:.1f}s")

total = time.perf_counter() - t_program
print(f"\nTotal runtime: {total:.1f}s (eval {cum_eval_time:.1f}s / evolve {cum_evolve_time:.1f}s)")
