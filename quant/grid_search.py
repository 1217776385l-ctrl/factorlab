"""P1v2 · 网格搜索调参:用户约束(持仓≤5只、周期≤1个月、回测2020至今)下朝收益最优化。
搜索空间:因子子集(2^n) × TopN{3,5} × 频率{W,2W,M} × 加权{等权,ICIR}
评估:2020-01-01 至最新;另切 2020-2023(选参窗)/ 2024-2026(验证窗)看稳健性。
产出:reports/grid_dump.json(全网格)+ 最优配置重跑导出终端。
"""
import os
import json, time, itertools, argparse, sys
import numpy as np
import pandas as pd
import panel, factors, research, engine, export

BASE = os.environ.get("FACTORLAB_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
START = '2020-01-01'

def perf(nav, bench=None):
    nav = nav.sort_index().dropna()
    if len(nav) < 60:
        return None
    ret = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    ann = nav.iloc[-1] ** (1 / max(years, 1e-9)) - 1
    vol = ret.std() * np.sqrt(252)
    mdd = (nav / nav.cummax() - 1).min()
    out = {'annual': float(ann), 'sharpe': float(ann / vol) if vol > 0 else 0.0, 'mdd': float(mdd), 'n': int(len(nav))}
    if bench is not None:
        b = bench.reindex(nav.index).ffill().dropna()
        if len(b) > 60:
            b = b / b.iloc[0]
            b_ann = b.iloc[-1] ** (1 / max(years, 1e-9)) - 1
            out['excess'] = float(ann - b_ann)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    args = ap.parse_args()
    t0 = time.time()
    p = panel.build_panel()
    bench = pd.read_parquet(f"{BASE}/data/index_000300.parquet")
    bench['close'] = pd.to_numeric(bench['close'], errors='coerce')
    bench_s = bench.set_index(pd.to_datetime(bench['date']))['close']

    F = research.direction_fix(factors.compute_factors(p))
    health = research.ichealth(p, F)
    F = research.empirical_direction(F, health)
    Fz = research.zscore(F)
    print(f"factors · {time.time()-t0:.0f}s", flush=True)

    # 候选因子池:体检 t 值达标的(方向已修正)
    pool = [r['factor'] for r in sorted(health, key=lambda r: -abs(r['icir'])) if abs(r['t_stat']) >= 1.7]
    pool = pool[:8]
    icir = {r['factor']: r['icir'] for r in health}
    print(f"pool({len(pool)}):", pool, flush=True)

    # 子集枚举:单因子 + 组合(全子集太多时用贪心+关键组合;8 因子 2^8=256 可承受)
    subsets = []
    for r in range(1, len(pool) + 1):
        subsets.extend(itertools.combinations(pool, r))
    # 限制子集大小 ≤6(控制组合数与过拟合)
    subsets = [s for s in subsets if len(s) <= 6]
    subsets = subsets[args.shard::args.nshards]
    print(f"subsets shard {args.shard}/{args.nshards}: {len(subsets)}", flush=True)

    results = []
    cnt = 0
    OUT = f"{BASE}/reports/grid_dump_{args.shard}.json"
    for sub in subsets:
        sub = list(sub)
        w = [icir[f] for f in sub]
        w = [x / sum(w) for x in w]
        for weight_mode in ('equal', 'icir'):
            weights = None if weight_mode == 'equal' else w
            score = research.combine(Fz, sub, weights=weights)
            for top_n in (3, 5):
                for freq in ('W', '2W', 'M'):
                    r = engine.run_strategy(p, score, top_n=top_n, freq=freq, start=START)
                    full = perf(r['nav'], bench_s)
                    if full is None:
                        continue
                    nav = r['nav']
                    sel = perf(nav[nav.index < pd.Timestamp('2024-01-01')], bench_s)
                    oos = perf(nav[nav.index >= pd.Timestamp('2024-01-01')], bench_s)
                    results.append({'factors': sub, 'weight': weight_mode, 'top_n': top_n, 'freq': freq,
                                    'annual': round(full['annual'], 4), 'excess': round(full.get('excess', float('nan')), 4),
                                    'sharpe': round(full['sharpe'], 3), 'mdd': round(full['mdd'], 4),
                                    'sel_excess': round(sel.get('excess', float('nan')), 4) if sel else None,
                                    'oos_excess': round(oos.get('excess', float('nan')), 4) if oos else None,
                                    'oos_annual': round(oos['annual'], 4) if oos else None})
                    cnt += 1
                    if cnt % 100 == 0:
                        json.dump(results, open(OUT, 'w'), indent=1)
                        print(f"  {cnt} combos · {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(results).sort_values('annual', ascending=False)
    json.dump(results, open(f"{BASE}/reports/grid_dump_{args.shard}.json", 'w'), indent=1)
    print("=== TOP 15 by 年化(2020至今)===", flush=True)
    for _, row in df.head(15).iterrows():
        print(f"  {row['annual']:+.1%} exc={row['excess']:+.1%} sh={row['sharpe']:.2f} mdd={row['mdd']:.0%} "
              f"N={row['top_n']} {row['freq']} {row['weight']} | 24-26验证:{row['oos_annual']:+.1%} "
              f"| {'+'.join(f[1:].split('_',1)[0] for f in row['factors'])}", flush=True)
    print(f"GRID DONE · {len(results)} combos · {time.time()-t0:.0f}s", flush=True)

if __name__ == '__main__':
    main()
