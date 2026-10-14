"""主流程:数据 → 因子 → 研究 → 策略 → 敏感性/简洁性 → 导出终端 + 报告素材。
运行前需 data/panel.pkl。产出: results/*.json, terminal/data.js, reports/pipeline_dump.json
"""
import os
import json, pickle, time
import numpy as np
import pandas as pd
import panel, factors, engine, strategies, export, research

BASE = os.environ.get("FACTORLAB_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
START = '2016-07-01'
IS_END = pd.Timestamp('2021-12-31')
DUMP = {}

def oos_stats(st_full, st_is):
    """拆样本外段指标:直接对 nav 切片。"""
    pass

def main():
    t0 = time.time()
    print('== 1. panel ==', flush=True)
    p = panel.build_panel()
    bench = pd.read_parquet(f"{BASE}/data/index_000300.parquet")
    bench['close'] = pd.to_numeric(bench['close'], errors='coerce')
    bench_s = bench.set_index(pd.to_datetime(bench['date']))['close']
    names = json.load(open(f"{BASE}/data/meta/name_map.json"))
    print(f"panel {p['close'].shape} · {time.time()-t0:.0f}s", flush=True)

    print('== 2. factors ==', flush=True)
    F = research.direction_fix(factors.compute_factors(p))
    Fz = research.zscore(F)
    print(f"factors done · {time.time()-t0:.0f}s", flush=True)

    print('== 3. factor health (IC) ==', flush=True)
    health = research.ichealth(p, F)
    DUMP['health'] = [{k: v for k, v in r.items() if k != 'ic_series'} for r in health]
    for r in health:
        print(f"  {r['factor']:>14}  IC={r['ic_mean']:+.4f}  ICIR={r['icir']:+.2f}  t={r['t_stat']:+.1f}  IS={r['is_ic']:+.4f} OOS={r['oos_ic']:+.4f} 正年份{r['pos_years']:.0%}", flush=True)

    # 实证方向修正:本批先验「高好」的动量/热度类因子在A股月频实际全部反向。
    # 选因子与合成前按 IC 符号统一为「大=好」;P2 事件条件保留原始先验(作为可证伪假设)。
    F = research.empirical_direction(F, health)
    Fz = research.zscore(F)
    print('  (directions re-aligned by empirical IC signs)', flush=True)

    print('== 4. corr + selection ==', flush=True)
    sample_dates = research.rebalance_sample(p, START)[-24::3]
    C = research.cross_corr(Fz, sample_dates)
    chosen = research.greedy_select(health, Fz)
    print('chosen:', chosen, flush=True)
    DUMP['corr'] = C.round(3).to_dict()
    DUMP['chosen'] = chosen

    print('== 5. P1 多因子等权合成(方向对齐+去重) ==', flush=True)
    s1 = research.combine(Fz, chosen)
    r1 = engine.run_strategy(p, s1, top_n=50, freq='M', start=START, name='P1')
    st1 = engine.perf_stats(r1['nav'], bench=bench_s)
    print('P1:', {k: round(v, 4) if isinstance(v, float) else v for k, v in st1.items()}, flush=True)

    # 分层展示:用 P1 score 分五层
    layers = engine.layered(p, s1, 5, top_n=None or 40, freq='M', start=START)
    layer_navs = {k: v['nav'] for k, v in layers.items()}

    # IC 序列(组合得分)
    ic1 = engine.monthly_rank_ic(s1, p, 21, START)

    # 最新持仓
    last_rb = sorted(r1['picks_hist'])[-1]
    picks = r1['picks_hist'][last_rb]
    holdings = [{'code': c, 'name': names.get(c, ''), 'score': float(s1.loc[last_rb, c]), 'weight': 1 / len(picks)} for c in picks]
    holdings = sorted(holdings, key=lambda x: -x['score'])

    avg_to = float(r1['turnover'].mean())

    report1 = {
        'meta': {'key': 'p1', 'name': 'P1 量价多因子', 'generated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
                 'universe': '全A剔ST/次新/流动性后20%', 'rebalance': '月度', 'top_n': 50,
                 'period_start': str(r1['nav'].index[0].date()), 'period_end': str(r1['nav'].index[-1].date()),
                 'cost_model': '双边0.2%', 'fill_model': 'T+1开盘·一字板不成交·跌停顺延',
                 'holdings_date': str(last_rb.date())},
        'metrics': {**{k: v for k, v in st1.items() if not isinstance(v, pd.Series)},
                    'avg_turnover': avg_to, 'n_trades': r1['n_trades'], 'blocked_buys': r1['blocked_buys']},
        'nav': export.nav_payload(r1['nav'], bench_s),
        'annual': export.annual_payload(r1['nav'], bench_s),
        'ic': export.ic_payload(ic1),
        'layers': export.layers_payload(layer_navs),
        'holdings_latest': holdings,
        'notes': [
            '历史回测含幸存者偏差(退市股不完整),绝对收益偏乐观,相对结论更可信。',
            '样本内 2016-07~2021-12 开发,样本外 2022-01 起;详细拆分见报告。',
            '顺延卖出期间旧仓收益按原权重计入;一字板无法成交的买入已剔除。',
        ],
    }
    export.export_one(report1)
    DUMP['p1_stats'] = {k: v for k, v in st1.items() if not isinstance(v, pd.Series)}

    print('== 6. 参数敏感性 ==', flush=True)
    sens = []
    for tn in (30, 50, 100):
        for fq in ('M', '2W'):
            rr = engine.run_strategy(p, s1, top_n=tn, freq=fq, start=START)
            ss = engine.perf_stats(rr['nav'], bench=bench_s)
            sens.append({'top_n': tn, 'freq': fq, 'annual': round(ss['annual_return'], 4),
                         'excess': round(ss.get('excess_annual', float('nan')), 4),
                         'sharpe': round(ss['sharpe'], 3), 'mdd': round(ss['max_drawdown'], 4)})
            print(' ', sens[-1], flush=True)
    DUMP['sensitivity'] = sens

    print('== 7. 简洁性:逐因子剔除 ==', flush=True)
    used = chosen
    simplicity = []
    base_ex = st1.get('excess_annual')
    for drop in used:
        keep = [f for f in used if f != drop]
        s_sub = strategies.p1_score(Fz, names=keep)
        rr = engine.run_strategy(p, s_sub, top_n=50, freq='M', start=START)
        ss = engine.perf_stats(rr['nav'], bench=bench_s)
        simplicity.append({'dropped': drop, 'kept': keep, 'excess': round(ss.get('excess_annual', float('nan')), 4),
                           'sharpe': round(ss['sharpe'], 3)})
        print(' ', simplicity[-1], flush=True)
    DUMP['simplicity'] = {'used': used, 'leave_one_out': simplicity, 'base_excess': base_ex}

    print('== 8. P2 缩量回踩反转 ==', flush=True)
    s2 = strategies.p2_score(F)   # 用原始方向因子构造条件
    r2 = engine.run_strategy(p, s2, top_n=15, freq='M', start=START, name='P2')
    st2 = engine.perf_stats(r2['nav'], bench=bench_s)
    print('P2:', {k: round(v, 4) if isinstance(v, float) else v for k, v in st2.items()}, flush=True)
    ic2 = engine.monthly_rank_ic(s2.dropna(how='all'), p, 21, START)
    report2 = {
        'meta': {'key': 'p2', 'name': 'P2 缩量回踩反转', 'generated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
                 'universe': '同P1 · 条件筛选', 'rebalance': '月度', 'top_n': 15,
                 'period_start': str(r2['nav'].index[0].date()), 'period_end': str(r2['nav'].index[-1].date()),
                 'cost_model': '双边0.2%', 'fill_model': 'T+1开盘·一字板不成交·跌停顺延',
                 'holdings_date': str(sorted(r2['picks_hist'])[-1].date()) if r2['picks_hist'] else ''},
        'metrics': {**{k: v for k, v in st2.items() if not isinstance(v, pd.Series)},
                    'avg_turnover': float(r2['turnover'].mean()) if len(r2['turnover']) else 0,
                    'n_trades': r2['n_trades'], 'blocked_buys': r2['blocked_buys']},
        'nav': export.nav_payload(r2['nav'], bench_s),
        'annual': export.annual_payload(r2['nav'], bench_s),
        'notes': [
            '事件型策略:特质动量前30% + 回撤35%~70%带 + 量能收缩 + 20日回调,空仓期不计收益。',
            '交易次数有限,胜率参考意义低,重点看盈亏比与回撤。',
        ],
    }
    export.export_one(report2)
    DUMP['p2_stats'] = {k: v for k, v in st2.items() if not isinstance(v, pd.Series)}

    with open(f"{BASE}/reports/pipeline_dump.json", 'w', encoding='utf-8') as f:
        json.dump(DUMP, f, ensure_ascii=False, indent=1, default=str)
    print(f"ALL DONE · {time.time()-t0:.0f}s", flush=True)

if __name__ == '__main__':
    main()
