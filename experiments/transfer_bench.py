"""迁移/泛化能力基准：10×10 训练的 7b / 12 最佳模型 × 三种空间 × 各 1000 盘。

空间：
    12×12、15×15 —— 方形，直接改 cfg.GRID_SIZE，环境语义与训练完全同构；
    12×15   —— 非方形，用矩形环境子类（行界 GR、列界 GC 独立）：
              食物撒点/占用图/步进边界/8 向射线射程全部按矩形精确实现，
              射线循环上界 KMAX=max(GR,GC)，obs28 身后格度用 GEFF=(GR+GC)/2。
              等价性保障：GR=GC 时与原版环境逐步逐位对拍（见 --selfcheck）。

模型：
    7b = test7b_best_model.pth（10×10, N=256 稠密, 32proj）
    12 = test12_econ_best_model.pth（10×10, N=256 稠密, 32ego1）
    （用户所称"12b"按 12 系最佳 test12_econ 理解）

用法：
    python experiments/transfer_bench.py            # 全部 6 组
    python experiments/transfer_bench.py --selfcheck  # 仅矩形等价性对拍
"""
import importlib.util, inspect, json, math, os, sys, time
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENGINES = {
    '7b': ('test7b.py', 'test7b_best_model.pth'),
    '12': ('test12.py', 'test12_econ_best_model.pth'),
}
SPACES = {
    '12x12': (12, 12),
    '15x15': (15, 15),
    '12x15': (12, 15),
}
# 尾盘上限：大棋盘上强策略可能近乎无限存活，按空间封顶（截断盘按截断时计食，
# 各模型同口径，只影响尾部个体）
MAX_STEPS_CAP = {'12x12': 20000, '15x15': 30000, '12x15': 30000}
GAMES = 1000
GAMES_PER_CHUNK = 250
SEED0 = 20260906


def load_module(fname):
    spec = importlib.util.spec_from_file_location(fname[:-3] + '_xb', os.path.join(ROOT, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def patch_method(cls, name, extra_ns):
    """取 cls.name 源码，做矩形几何替换后重新绑定为 Rect 子类方法。

    替换规则（只动边界/量程，不动语义）：
      射线循环上界  G -> self.KMAX
      边界谓词      r < G / c < G -> r < self.GR / c < self.GC
      自身射线阈值  dist <= G -> dist <= self.KMAX
      射线默认距离  float(G + 1) / float(G) -> self.KMAX + 1 / self.KMAX
      身后格度      / self.G -> / self.GEFF
    """
    src = inspect.getsource(getattr(cls, name))
    import textwrap
    new = textwrap.dedent(src)
    pats = [
        ('for k in range(1, G + 1):', 'for k in range(1, self.KMAX + 1):'),
        ('(r >= 0) & (r < G) & (c >= 0) & (c < G)',
         '(r >= 0) & (r < self.GR) & (c >= 0) & (c < self.GC)'),
        ('dist <= G,', 'dist <= self.KMAX,'),
        ('float(G + 1)', 'float(self.KMAX + 1)'),
        ('float(G),', 'float(self.KMAX),'),
        ('clamp(min=1) / self.G)', 'clamp(min=1) / self.GEFF)'),
    ]
    counts = {p: new.count(p[0]) for p in pats}
    for old, repl in pats:
        new = new.replace(old, repl)
    ns = dict(extra_ns)
    exec(compile(new, f'<rect_{name}>', 'exec'), ns)
    return ns[name], counts


def make_rect_env_cls(mod, GR, GC):
    """基于引擎的 BatchedSnakeEnv 生成 (GR×GC) 矩形环境子类。"""
    Base = mod.BatchedSnakeEnv
    import torch as _t

    class RectSnakeEnv(Base):
        def __init__(self, cfg, B, device):
            self.cfg = cfg
            self.B = B
            self.device = device
            self.GR, self.GC = GR, GC
            self.G = GC                 # 大边：模块级遥测辅助函数按 self.G 分配网格
            self.KMAX = max(GR, GC)
            self.GEFF = (GR + GC) / 2.0
            self.MAXLEN = GR * GC
            self.DIRS = mod._make_dirs(device)
            self.mode24 = (getattr(cfg, 'OBS_MODE', '32proj') == '24')
            self.crn = None
            self.reset()

        def reset(self, bank=None):
            assert bank is None, '矩形基准不支持 CRN bank'
            dev = self.device
            self.draw_cnt = torch.zeros(self.B, dtype=torch.long, device=dev)
            self.head = torch.stack((torch.full((self.B,), self.GR // 2, dtype=torch.long, device=dev),
                                     torch.full((self.B,), self.GC // 2, dtype=torch.long, device=dev)), dim=1)
            self.dir_idx = torch.randint(0, 4, (self.B,), device=dev)
            self.body = torch.zeros(self.B, self.MAXLEN, 2, dtype=torch.long, device=dev)
            self.body[:, 0] = self.head
            self.body[:, 1] = self.head - self.DIRS[self.dir_idx]
            self.body_len = torch.full((self.B,), 2, dtype=torch.long, device=dev)
            self.food = self._place_food_init()
            self.alive = torch.ones(self.B, dtype=torch.bool, device=dev)
            self.steps = torch.zeros(self.B, dtype=torch.long, device=dev)
            self.steps_wo_food = torch.zeros(self.B, dtype=torch.long, device=dev)
            self.ate = torch.zeros(self.B, dtype=torch.bool, device=dev)
            self.died = torch.zeros(self.B, dtype=torch.long, device=dev)

        def _next_cand(self):
            if self.GR == self.GC:
                return torch.randint(0, self.GR, (self.B, 2), device=self.device)
            dev = self.device
            return torch.stack((torch.randint(0, self.GR, (self.B,), device=dev),
                                torch.randint(0, self.GC, (self.B,), device=dev)), dim=1)

        def _place_food_init(self):
            dev = self.device
            B, G = self.B, self.GC
            head, neck = self.head, self.body[:, 1]
            cand = self._next_cand()
            bad = (cand == head).all(dim=1) | (cand == neck).all(dim=1)
            for _ in range(31):
                if not bad.any():
                    break
                re = self._next_cand()
                cand = torch.where(bad.unsqueeze(1), re, cand)
                bad = (cand == head).all(dim=1) | (cand == neck).all(dim=1)
            if bad.any():
                occ = torch.zeros(B, G * G, dtype=torch.bool, device=dev)
                occ[torch.arange(B, device=dev), head[:, 0] * G + head[:, 1]] = True
                occ[torch.arange(B, device=dev), neck[:, 0] * G + neck[:, 1]] = True
                free = (~occ).float()
                idx = torch.argmax(free, dim=1)
                fb = torch.stack((idx // G, idx % G), dim=1)
                cand = torch.where(bad.unsqueeze(1), fb, cand)
            return cand

        def _place_food_after_eat(self, eat_mask):
            B, G, dev = self.B, self.GC, self.device
            if not eat_mask.any():
                return
            occ = self._occupancy_flat()
            occ_b = occ > 0.5
            cand = self._next_cand()
            bad = occ_b.gather(1, (cand[:, 0] * G + cand[:, 1]).unsqueeze(1)).squeeze(1)
            for _ in range(31):
                if not bad.any():
                    break
                re = self._next_cand()
                cand = torch.where(bad.unsqueeze(1), re, cand)
                bad = occ_b.gather(1, (cand[:, 0] * G + cand[:, 1]).unsqueeze(1)).squeeze(1)
            if bad.any():
                free = (~occ_b).float()
                idx = torch.argmax(free, dim=1)
                fb = torch.stack((idx // G, idx % G), dim=1)
                cand = torch.where(bad.unsqueeze(1), fb, cand)
            self.food = torch.where(eat_mask.unsqueeze(1), cand, self.food)

        def _occupancy_flat(self, tail_invalid=False):
            B, G, dev = self.B, self.GC, self.device
            flat = self.body[:, :, 0] * G + self.body[:, :, 1]
            valid = torch.arange(self.MAXLEN, device=dev)[None, :] < self.body_len[:, None]
            if tail_invalid:
                valid &= torch.arange(self.MAXLEN, device=dev)[None, :] < (self.body_len - 1)[:, None]
            occ = torch.zeros(B, G * G, dtype=torch.float32, device=dev)
            occ.scatter_add_(1, flat.clamp(max=G * G - 1), valid.float())
            return occ

        def step(self, actions):
            B, dev = self.B, self.device
            nd_idx = torch.where(actions == 1, (self.dir_idx + 3) % 4, self.dir_idx)
            nd_idx = torch.where(actions == 2, (self.dir_idx + 1) % 4, nd_idx)
            self.dir_idx = nd_idx
            nd = self.DIRS[nd_idx]

            alive_f = self.alive
            self.steps = torch.where(alive_f, self.steps + 1, self.steps)
            self.steps_wo_food = torch.where(alive_f, self.steps_wo_food + 1, self.steps_wo_food)

            next_head = self.head + nd
            out_b = ((next_head[:, 0] < 0) | (next_head[:, 0] >= self.GR) |
                     (next_head[:, 1] < 0) | (next_head[:, 1] >= self.GC))

            will_eat = (next_head == self.food).all(dim=1)
            occ = self._occupancy_flat(tail_invalid=True)
            occ_eat = self._occupancy_flat(tail_invalid=False)
            occ_use = torch.where(will_eat[:, None], occ_eat, occ)
            ar = torch.arange(B, device=dev)
            hit = occ_use[ar, next_head[:, 0].clamp(0, self.GR - 1) * self.GC
                          + next_head[:, 1].clamp(0, self.GC - 1)] > 0.5
            crash = alive_f & (out_b | hit)

            move = alive_f & (~crash)
            shifted = torch.zeros_like(self.body)
            shifted[:, 0] = next_head
            shifted[:, 1:] = self.body[:, :-1]
            self.body = torch.where(move[:, None, None], shifted, self.body)
            self.head = torch.where(move[:, None], next_head, self.head)

            ate = move & will_eat
            self.body_len = torch.where(move, (self.body_len + ate.long()).clamp(max=self.MAXLEN),
                                        self.body_len)
            self.steps_wo_food = torch.where(ate, torch.zeros_like(self.steps_wo_food),
                                             self.steps_wo_food)
            self._place_food_after_eat(ate)
            self.ate = ate
            self.alive = alive_f & (~crash)
            self.died = torch.where(crash, self._death_cause(out_b, hit), self.died)
            # 饿死判定（与原版 step 收尾一致）
            starve = (self.alive & (self.steps_wo_food >
                                    float(getattr(self.cfg, 'STARVE_SLOPE', 5.0)) * self.body_len + 20))
            self.alive = self.alive & (~starve)
            self.died = torch.where(starve, torch.full_like(self.died, 3), self.died)

        def _death_cause(self, out_b, hit):
            d = torch.zeros(self.B, dtype=torch.long, device=self.device)
            return torch.where(out_b, torch.full_like(d, 1),
                               torch.where(hit, torch.full_like(d, 2), d))

    # obs32：源码级几何补丁（只改边界/量程/身后格度）
    ns = {'torch': _t, 'math': math}
    fn, counts = patch_method(Base, '_obs32', ns)
    RectSnakeEnv._obs32 = fn
    RectSnakeEnv._patch_counts = counts
    return RectSnakeEnv


def selfcheck():
    """等价性对拍（顺序重放法）：两环境共享全局 RNG，不能交替调用——
    各自用同一种子独立跑完整轨迹并记录，再比较记录。GR=GC=10 时矩形环境
    与原版必须逐步逐位一致（状态+obs+食物重撒）。"""
    mod = load_module(ENGINES['12'][0])
    cfg = mod.Config()
    cfg.DEVICE = 'cpu'
    cfg.USE_FP16 = False
    dev = torch.device('cpu')
    Rect = make_rect_env_cls(mod, 10, 10)
    print('补丁命中计数:', Rect._patch_counts)

    def trajectory(make_env, n_steps, seed, B=8):
        torch.manual_seed(seed)
        env = make_env()
        rec = {'obs': [], 'head': [], 'food': [], 'len': [], 'alive': [], 'died': []}
        acts = torch.randint(0, 3, (n_steps, B))
        for t in range(n_steps):
            rec['obs'].append(env.obs().clone())
            rec['head'].append(env.head.clone())
            rec['food'].append(env.food.clone())
            rec['len'].append(env.body_len.clone())
            rec['alive'].append(env.alive.clone())
            rec['died'].append(env.died.clone())
            env.step(acts[t])
        # 追加强制重撒路径（env 活着与否不影响重撒函数）
        foods = []
        for _ in range(20):
            env._place_food_after_eat(torch.ones(B, dtype=torch.bool))
            foods.append(env.food.clone())
        return rec, foods

    ra, fa = trajectory(lambda: mod.BatchedSnakeEnv(cfg, 8, dev), 300, 7)
    rb, fb = trajectory(lambda: Rect(cfg, 8, dev), 300, 7)
    for k in ('obs', 'head', 'food', 'len', 'alive', 'died'):
        for t, (x, y) in enumerate(zip(ra[k], rb[k])):
            assert torch.equal(x, y), f'{k} 不一致 @ step {t}'
    for t, (x, y) in enumerate(zip(fa, fb)):
        assert torch.equal(x, y), f'重撒食物不一致 @ iter {t}'
    print('selfcheck ✓：GR=GC 时矩形环境与原版 300 步状态+obs+重撒食物逐位一致')


def load_model(mod, model_file):
    data = torch.load(os.path.join(ROOT, model_file), map_location='cpu', weights_only=False)
    cfg = mod.Config()
    for k, v in (data.get('config', {}) or {}).items():
        if not k.startswith('__'):
            setattr(cfg, k, v)
    res = mod.load_best_state(os.path.join(ROOT, model_file), cfg)
    if res is None:
        raise RuntimeError('加载失败: ' + model_file)
    st, food, steps = res
    return st, cfg, food


def run_bench(mod, model_file, tag, GR, GC, device):
    Rect = make_rect_env_cls(mod, GR, GC)
    orig_env = mod.BatchedSnakeEnv
    mod.BatchedSnakeEnv = Rect          # 评估 chunk 内直接引用模块全局类名
    space_tag = f'{GR}x{GC}'
    try:
        st, cfg, saved_food = load_model(mod, model_file)
        cfg.GRID_SIZE = GR if GR == GC else GC   # 方形直接改；矩形取大边——
        # Rect 环境不读此字段（用 GR/GC），但模块内按 cfg.GRID_SIZE 建的
        # 遥测辅助结构（如 12 的 VectorCycleTeacher 回路索引表）必须覆盖
        # 全部 (row<GR, col<GC) 位置，否则 CUDA gather 越界
        cfg.EVAL_EPISODES = 1           # 每行 1 局（7b chunk 默认多局循环，须关）
        cfg.MAX_STEPS = min(int(cfg.MAX_STEPS), MAX_STEPS_CAP[space_tag])
        cfg.DEVICE = str(device)
        dev = device

        auto = mod._auto_eval_batch(cfg, dev)
        batch = min(auto, GAMES_PER_CHUNK, GAMES)
        pop = mod.GeneStack(cfg, B=batch, device=dev)
        pop.random_init()
        for i in range(batch):
            pop.set_individual_from_state(i, st)
        if getattr(cfg, 'USE_FP16', True):
            pop.fp16()
            pop.refresh_eff()

        foods, walls, selfs, starves, stepss, markers = [], [], [], [], [], []
        done = 0
        t0 = time.perf_counter()
        while done < GAMES:
            n = min(batch, GAMES - done)
            try:
                sub = pop[:n]
                if hasattr(mod, '_eval_sweep_chunk'):
                    m = mod._eval_sweep_chunk(sub, cfg, None).cpu().numpy()
                else:
                    m = mod._eval_chunk(sub, cfg).cpu().numpy()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                batch = max(32, batch // 2)
                pop = mod.GeneStack(cfg, B=batch, device=dev)
                pop.random_init()
                for i in range(batch):
                    pop.set_individual_from_state(i, st)
                if getattr(cfg, 'USE_FP16', True):
                    pop.fp16()
                    pop.refresh_eff()
                continue
            marker = m[:, 1] >= 99999      # test7b chunk 的单侧转弯判死标记
            keep = ~marker
            foods.extend(m[:, 0].tolist())
            walls.extend(m[:, 5].tolist())
            selfs.extend(m[:, 6].tolist())
            starves.extend(m[:, 7].tolist())
            stepss.extend((m[keep, 1] + m[keep, 2]).tolist())
            markers.extend(marker.tolist())
            done += n
            print(f'  [{tag}] {done}/{GAMES}（{time.perf_counter() - t0:.0f}s）', flush=True)
    finally:
        mod.BatchedSnakeEnv = orig_env

    import numpy as np
    eng_tag = tag.split('@')[0]
    fa = np.array(foods)
    ma = np.array(markers)
    out = {
        'engine': eng_tag, 'model': model_file, 'space': f'{GR}x{GC}',
        'games': GAMES, 'elapsed_s': round(time.perf_counter() - t0, 1),
        'max_steps_cap': MAX_STEPS_CAP[f'{GR}x{GC}'],
        'mean_food': float(fa.mean()), 'median_food': float(np.median(fa)),
        'std_food': float(fa.std()),
        'pct': {f'P{p}': float(np.percentile(fa, p)) for p in (5, 25, 50, 75, 90)},
        'min_food': float(fa.min()), 'max_food': float(fa.max()),
        'die_wall': float(np.mean(np.array(walls) > 0)),
        'die_self': float(np.mean(np.array(selfs) > 0)),
        'die_starve': float(np.mean(np.array(starves) > 0)),
        'die_oneturn': float(ma.mean()),
        'mean_steps': float(np.mean(stepss)),
    }
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None,
                    help='只跑子集：引擎名（7b/12）或 "引擎@空间"（7b@12x15）')
    ap.add_argument('--force', action='store_true', help='忽略已有结果重跑')
    args = ap.parse_args()
    if '--selfcheck' in sys.argv:
        selfcheck()
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device =', device, flush=True)
    out_json = os.path.join(ROOT, 'results', 'transfer_bench.json')
    results = []
    if os.path.exists(out_json) and not args.force:
        with open(out_json, encoding='utf-8') as f:
            results = json.load(f)
    done_keys = {(r['engine'], r['space']) for r in results}
    for eng_tag, (fname, model_file) in ENGINES.items():
        if args.only and args.only != eng_tag and not args.only.startswith(eng_tag + '@'):
            continue
        mod = load_module(fname)
        for space_tag, (GR, GC) in SPACES.items():
            if args.only and '@' in args.only and args.only != f'{eng_tag}@{space_tag}':
                continue
            if (eng_tag, space_tag) in done_keys:
                print(f'跳过已完成 {eng_tag}@{space_tag}')
                continue
            tag = f'{eng_tag}@{space_tag}'
            print(f'\n=== {tag}（{model_file}）===', flush=True)
            torch.manual_seed(SEED0)
            r = run_bench(mod, model_file, tag, GR, GC, device)
            print(f"  => mean={r['mean_food']:.2f} median={r['median_food']:.1f} "
                  f"std={r['std_food']:.2f} | 死因 墙{r['die_wall']:.0%} "
                  f"己{r['die_self']:.0%} 饿{r['die_starve']:.0%} "
                  f"单侧{r['die_oneturn']:.0%} | 均步 {r['mean_steps']:.0f} | "
                  f"{r['elapsed_s']}s", flush=True)
            results.append(r)
            # 原子替换写（崩溃安全：写临时文件再替换，避免半截 JSON）
            tmp = out_json + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=1)
            os.replace(tmp, out_json)

    print('\n===== 汇总（训练空间 10×10 → 迁移空间 1000 盘）=====')
    print(f"{'模型':<6}{'空间':<8}{'mean':>7}{'median':>8}{'std':>7}{'P5':>5}{'P90':>5}"
          f"{'墙死':>7}{'己死':>7}{'饿死':>7}{'均步':>7}")
    for r in results:
        print(f"{r['engine']:<6}{r['space']:<8}{r['mean_food']:>7.2f}{r['median_food']:>8.1f}"
              f"{r['std_food']:>7.2f}{r['pct']['P5']:>5.0f}{r['pct']['P90']:>5.0f}"
              f"{r['die_wall']:>7.1%}{r['die_self']:>7.1%}{r['die_starve']:>7.1%}"
              f"{r['mean_steps']:>7.0f}")
    print('\n已写入 results/transfer_bench.json')


if __name__ == '__main__':
    main()
