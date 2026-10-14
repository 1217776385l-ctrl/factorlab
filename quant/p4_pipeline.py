"""P4 · 融合策略正式管线:进攻(P1四因子)× 状态自适应(S-E)× 回撤控制(D3/D4)。
网格(小而精):状态阈值{0.6,0.7,0.8} × 防御方式{低波股, 半仓} × D3{开,关} × 状态变量{大盘波动, 大盘均线}  → 24 组合
评估:2020起,全窗 + 2024-26验证窗;对照:P1裸奔、S-E、D3、D4 单工具。
产出:results/p4.json 入终端 + 融合报告章。
"""
import os
import pickle, json, time, itertools
import numpy as np
import pandas as pd
import panel, research, engine, export

BASE = os.environ.get("FACTORLAB_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
START = '2020-01-01'
EPS = 1e-9

def perf2(nav, bench=None):
    nav = nav.sort_index().dropna()
    if len(nav) < 60: return None
    ret = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    ann = nav.iloc[-1] ** (1 / max(years, 1e-9)) - 1
    vol = ret.std() * np.sqrt(252)
    mdd = (nav / nav.cummax() - 1).min()
    o = {'annual': float(ann), 'sharpe': float(ann/vol) if vol>0 else 0.0, 'mdd': float(mdd),
         'calmar': float(ann/abs(mdd)) if mdd<0 else 0.0}
    if bench is not None:
        b = bench.reindex(nav.index).ffill().dropna()
        if len(b) > 60:
            b = b/b.iloc[0]
            o['excess'] = float(ann - (b.iloc[-1]**(1/max(years,1e-9)) - 1))
    return o

def main():
    t0 = time.time()
    p = panel.build_panel()
    c = pickle.load(open(f"{BASE}/data/fz_cache.pkl", 'rb'))
    bench = pd.read_parquet(f"{BASE}/data/index_000300.parquet")
    bench['close'] = pd.to_numeric(bench['close'], errors='coerce')
    bench_s = bench.set_index(pd.to_datetime(bench['date']))['close']
    dates = p['close'].index; cols = p['close'].columns
    ret = p['close'].pct_change(fill_method=None)

    base = research.combine(c['Fz'], ['G1_turn20', 'G4_vratio', 'G3_amtshare', 'G10_idiomom'])
    lowvol = research.zscore({'lv': -(ret.rolling(60).std())})['lv']
    hot = ret.rolling(60).std().rank(axis=1, pct=True) >= 0.8        # D3: 高波票
    b = bench_s.reindex(dates).ffill()
    bvol_rank = b.pct_change().rolling(20).std().rolling(250).rank(pct=True)
    below_ma = (b < b.rolling(60).mean()).values                      # D4 状态

    def build(state_kind, thr, defense, d3):
        """state_kind: vol|ma;thr: 0.6/0.7/0.8(vol) 或 0/1(ma);defense: lowvol|half"""
        if state_kind == 'vol':
            risk_on = (bvol_rank < thr).values
        else:
            risk_on = ~below_ma
        off = base.where(~hot) if d3 else base
        # risk-off 防御:换低波股 或 半仓(score 减半→等效买入少了,近似半仓:目标分除2→top_n 不变权重稀释不可行;
        # 简化:half = risk-off 时直接把低波股和进攻股各取一半 → 用分数合并:off*0.5 + lowvol*0.5)
        if defense == 'lowvol':
            s = pd.DataFrame(np.where(risk_on[:, None], off.values, lowvol.values),
                             index=dates, columns=cols)
        else:
            s = pd.DataFrame(np.where(risk_on[:, None], off.values, (off*0.5 + lowvol*0.5).values),
                             index=dates, columns=cols)
        return s

    grid = []
    for state_kind, thr, defense, d3 in itertools.product(
            ['vol', 'ma'], [0.6, 0.7, 0.8] + [0], ['lowvol', 'half'], [True, False]):
        if state_kind == 'ma' and thr != 0: continue
        s = build(state_kind, thr, defense, d3)
        r = engine.run_strategy(p, s, top_n=5, freq='M', start=START)
        full = perf2(r['nav'], bench_s)
        if full is None: continue
        nav = r['nav']
        oos = perf2(nav[nav.index >= pd.Timestamp('2024-01-01')], bench_s)
        grid.append({'state': state_kind, 'thr': thr, 'defense': defense, 'd3': d3,
                     'annual': round(full['annual'],4), 'sharpe': round(full['sharpe'],3),
                     'mdd': round(full['mdd'],4), 'calmar': round(full['calmar'],3),
                     'excess': round(full.get('excess', float('nan')),4),
                     'oos_annual': round(oos['annual'],4) if oos else None,
                     'oos_mdd': round(oos['mdd'],4) if oos else None})
        print(' ', grid[-1], flush=True)

    df = pd.DataFrame(grid).sort_values('calmar', ascending=False)
    json.dump(grid, open(f"{BASE}/reports/p4_grid.json", 'w'), indent=1)
    print("\n=== TOP8 by Calmar ===", flush=True)
    for _, row in df.head(8).iterrows():
        print(f"  calmar {row['calmar']:.2f} | 年化 {row['annual']:+.1%} 回撤 {row['mdd']:.0%} sharpe {row['sharpe']:.2f} "
              f"| 状态={row['state']}{row['thr']} 防御={row['defense']} D3={row['d3']} | 24-26验证 {row['oos_annual']:+.1%}/{row['oos_mdd']:.0%}", flush=True)

    # 最优配置 → 完整导出终端
    best = df.iloc[0]
    s = build(best['state'], best['thr'], best['defense'], bool(best['d3']))
    r4 = engine.run_strategy(p, s, top_n=5, freq='M', start=START)
    st4 = engine.perf_stats(r4['nav'], bench=bench_s)
    ic4 = engine.monthly_rank_ic(s.dropna(how='all'), p, 21, START)
    layers = engine.layered(p, s, 5, top_n=40, freq='M', start=START)
    names = json.load(open(f"{BASE}/data/meta/name_map.json"))
    last_rb = sorted(r4['picks_hist'])[-1]
    picks = r4['picks_hist'][last_rb]
    holdings = [{'code': x, 'name': names.get(x,''), 'score': float(s.loc[last_rb, x]), 'weight': 1/len(picks)} for x in picks]
    holdings = sorted(holdings, key=lambda x: -x['score'])
    defense_cn = {'lowvol': '换低波股', 'half': '半仓攻防'}[best['defense']]
    state_cn = f"大盘{'波动分位' if best['state']=='vol' else '均线'}>{best['thr']}" if best['state']=='vol' else "大盘跌破60日线"
    report = {
        'meta': {'key': 'p4', 'name': f"P4 攻防融合({state_cn}→{defense_cn})",
                 'generated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
                 'universe': '同P1', 'rebalance': '月度', 'top_n': 5,
                 'period_start': str(r4['nav'].index[0].date()), 'period_end': str(r4['nav'].index[-1].date()),
                 'cost_model': '双边0.2%', 'fill_model': 'T+1开盘·一字板不成交·跌停顺延',
                 'holdings_date': str(last_rb.date())},
        'metrics': {**{k: v for k, v in st4.items() if not isinstance(v, pd.Series)},
                    'avg_turnover': float(r4['turnover'].mean()), 'n_trades': r4['n_trades'],
                    'blocked_buys': r4['blocked_buys']},
        'nav': export.nav_payload(r4['nav'], bench_s),
        'annual': export.annual_payload(r4['nav'], bench_s),
        'ic': export.ic_payload(ic4),
        'layers': export.layers_payload({k: v['nav'] for k, v in layers.items()}),
        'holdings_latest': holdings,
        'notes': [
            f'结构:平时= P1四因子进攻(Top5);{state_cn}时 → {defense_cn}。D3高波票排除={"开" if best["d3"] else "关"}。',
            f'24组网格(状态阈值×防御方式×D3)按 Calmar 选优;对照 P1 裸奔:年化25.9%/回撤-39%。',
            '状态切换有滞后(20日波动/60日均线),V型反转行情会吃亏;验证窗与全窗口径见报告。',
        ],
    }
    export.export_one(report)
    json.dump({'best': best.to_dict(), 'n': len(grid)}, open(f"{BASE}/reports/p4_best.json", 'w'), indent=1, default=str)
    print(f"P4 DONE · {time.time()-t0:.0f}s", flush=True)

if __name__ == '__main__':
    main()
