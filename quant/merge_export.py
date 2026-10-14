"""合并网格分片结果 → 选最优配置 → 重建 P1v2 并导出终端 + 生成网格报告素材。"""
import os
import json, glob, time
import numpy as np
import pandas as pd
import panel, factors, research, engine, export

BASE = os.environ.get("FACTORLAB_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
START = '2020-01-01'

def main():
    t0 = time.time()
    files = sorted(glob.glob(f"{BASE}/reports/grid_dump_*.json"))
    rows = []
    for fp in files:
        rows.extend(json.load(open(fp)))
    df = pd.DataFrame(rows).sort_values('annual', ascending=False).reset_index(drop=True)
    print(f"total combos: {len(df)}", flush=True)
    top15 = df.head(15).to_dict('records')
    json.dump({'top15': top15, 'n': len(df)}, open(f"{BASE}/reports/grid_top.json", 'w'), indent=1, ensure_ascii=False)
    for i, r in df.head(12).iterrows():
        print(f"  {r['annual']:+.1%} exc={r['excess']:+.1%} sh={r['sharpe']:.2f} mdd={r['mdd']:.0%} "
              f"N={r['top_n']} {r['freq']} {r['weight']} | 24-26:{r['oos_annual'] if r['oos_annual'] is not None else '—'} "
              f"| {'+'.join(f.split('_',1)[1].replace('turn20','换手').replace('chip','筹码') for f in r['factors'])}", flush=True)

    best = df.iloc[0]
    sub = list(best['factors']); wn = best['weight']; top_n = int(best['top_n']); freq = best['freq']
    print(f"\nBEST: {sub} | {weight_mode_str(wn)} | Top{top_n} | {freq}", flush=True)

    # 重建最优策略
    p = panel.build_panel()
    bench = pd.read_parquet(f"{BASE}/data/index_000300.parquet")
    bench['close'] = pd.to_numeric(bench['close'], errors='coerce')
    bench_s = bench.set_index(pd.to_datetime(bench['date']))['close']
    names = json.load(open(f"{BASE}/data/meta/name_map.json"))

    F = research.direction_fix(factors.compute_factors(p))
    health = research.ichealth(p, F)
    F = research.empirical_direction(F, health)
    Fz = research.zscore(F)
    weights = None
    if wn == 'icir':
        w = [next(r['icir'] for r in health if r['factor'] == f) for f in sub]
        ssum = sum(w)
        weights = [x / ssum for x in w]
    score = research.combine(Fz, sub, weights=weights)
    r1 = engine.run_strategy(p, score, top_n=top_n, freq=freq, start=START, name='P1v2')
    st1 = engine.perf_stats(r1['nav'], bench=bench_s)
    print('P1v2:', {k: round(v, 4) if isinstance(v, float) else v for k, v in st1.items() if not isinstance(v, pd.Series)}, flush=True)
    ic1 = engine.monthly_rank_ic(score, p, 21, START)
    layers = engine.layered(p, score, 5, top_n=40, freq=freq, start=START)
    layer_navs = {k: v['nav'] for k, v in layers.items()}

    last_rb = sorted(r1['picks_hist'])[-1]
    picks = r1['picks_hist'][last_rb]
    holdings = [{'code': c, 'name': names.get(c, ''), 'score': float(score.loc[last_rb, c]), 'weight': 1/len(picks)}
                for c in picks if c in score.columns and last_rb in score.index]
    holdings = sorted(holdings, key=lambda x: -x['score'])

    freq_cn = {'W': '每周', '2W': '双周', 'M': '月度'}[freq]
    report = {
        'meta': {'key': 'p1', 'name': f'P1 精选Top{top_n}·{freq_cn}', 'generated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
                 'universe': '全A剔ST/次新/流动性后20%', 'rebalance': freq_cn, 'top_n': top_n,
                 'period_start': str(r1['nav'].index[0].date()), 'period_end': str(r1['nav'].index[-1].date()),
                 'cost_model': '双边0.2%', 'fill_model': 'T+1开盘·一字板不成交·跌停顺延',
                 'holdings_date': str(last_rb.date())},
        'metrics': {**{k: v for k, v in st1.items() if not isinstance(v, pd.Series)},
                    'avg_turnover': float(r1['turnover'].mean()), 'n_trades': r1['n_trades'],
                    'blocked_buys': r1['blocked_buys']},
        'nav': export.nav_payload(r1['nav'], bench_s),
        'annual': export.annual_payload(r1['nav'], bench_s),
        'ic': export.ic_payload(ic1),
        'layers': export.layers_payload(layer_navs),
        'holdings_latest': holdings,
        'notes': [
            f'网格搜索最优:{len(df)} 个组合(因子子集×TopN{{3,5}}×{{周,双周,月}}×{{等权,ICIR加权}}),以 2020 至今年化收益最大选取。',
            f'因子组合:{" + ".join(sub)}({weight_mode_str(wn)});2024-2026 验证窗年化 {best["oos_annual"]:+.1%}。',
            '网格在全窗口选优存在过拟合风险,验证窗与参数敏感性见报告;5 只持仓单票 20%,集中度风险高。',
            '历史回测含幸存者偏差,绝对收益偏乐观;T+1开盘成交、一字板不追、跌停顺延。',
        ],
    }
    export.export_one(report)
    json.dump({'p1v2_best': {'factors': sub, 'weight': wn, 'top_n': top_n, 'freq': freq,
                             'stats': {k: v for k, v in st1.items() if not isinstance(v, pd.Series)},
                             'top15': top15, 'n_combos': len(df)}},
              open(f"{BASE}/reports/p1v2_dump.json", 'w'), indent=1, ensure_ascii=False, default=str)
    print(f"MERGE+EXPORT DONE · {time.time()-t0:.0f}s", flush=True)

def weight_mode_str(w):
    return 'ICIR加权' if w == 'icir' else '等权'

if __name__ == '__main__':
    main()
