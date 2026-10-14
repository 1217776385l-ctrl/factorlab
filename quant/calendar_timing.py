"""日历择时层:统计(2015-2019 训练)→ 验证(2020-2026)→ 融入 P4 状态机。
只检验有先验依据的日历假设:①月份季节性 ②Turn-of-Month ③长假前避险(国庆/春节)。
"""
import os
import pickle, json
import numpy as np
import pandas as pd
import panel, research, engine

BASE = os.environ.get("FACTORLAB_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EPS = 1e-9

def month_table(bret, lo, hi):
    """各月平均日收益(%)与上涨概率。"""
    seg = bret[(bret.index.year >= lo) & (bret.index.year <= hi)]
    g = seg.groupby(seg.index.month)
    return pd.DataFrame({'mean_bp': (g.mean() * 1e4).round(1), 'win%': (g.apply(lambda x: (x > 0).mean()) * 100).round(0)})

def tom_table(bret, lo, hi):
    """ToM:每月末4日+次月前3日 vs 其余。"""
    seg = bret[(bret.index.year >= lo) & (bret.index.year <= hi)]
    months = sorted(set(seg.index.to_period('M')))
    tom_days = set()
    for m in months:
        cur = seg[seg.index.to_period('M') == m]
        nxt = seg[seg.index.to_period('M') == m + 1]
        if len(cur) >= 4: tom_days.update(cur.index[-4:])
        if len(nxt) >= 3: tom_days.update(nxt.index[:3])
    in_tom = seg.index.isin(tom_days)
    return {'in_bp': round(float(seg[in_tom].mean() * 1e4), 1), 'out_bp': round(float(seg[~in_tom].mean() * 1e4), 1)}

def preholiday_table(bret, lo, hi):
    """长假前 N 日:国庆(9月底)、春节(1月底2月初,用1月最后3个交易日近似)。"""
    seg = bret[(bret.index.year >= lo) & (bret.index.year <= hi)]
    months = sorted(set(seg.index.to_period('M')))
    pre = set()
    for m in months:
        cur = seg[seg.index.to_period('M') == m]
        nxt = seg[seg.index.to_period('M') == m + 1]
        # 月末最后3日 且 次月存在(近似长假前:9月末/1月末均被覆盖;也含普通月末——需区分)
        if m.month in (9, 1) and len(cur) >= 3:
            pre.update(cur.index[-3:])
    in_pre = seg.index.isin(pre)
    return {'pre_bp': round(float(seg[in_pre].mean() * 1e4), 1),
            'all_bp': round(float(seg.mean() * 1e4), 1), 'n': int(in_pre.sum())}

def weak_month_signal(bret_train, bret_valid):
    """训练期找最弱月(若显著),验证期检验方向。"""
    tr = month_table(bret_train, 2015, 2019)
    weakest = int(tr['mean_bp'].idxmin())
    va = month_table(bret_valid, 2020, 2026)
    v = va.loc[weakest]
    ok = v['mean_bp'] < 0
    return weakest, tr.loc[weakest], v, ok

if __name__ == '__main__':
    p = panel.build_panel()
    bench = pd.read_parquet(f"{BASE}/data/index_000300.parquet")
    bench['close'] = pd.to_numeric(bench['close'], errors='coerce')
    b = bench.set_index(pd.to_datetime(bench['date']))['close']
    bret = b.pct_change().dropna()

    print("=== ① 月份季节性(训练 2015-19 → 验证 20-26)===")
    weak_m, tr_stat, va_stat, ok = weak_month_signal(bret, bret)
    print(f"训练期最弱月:{weak_m} 月({tr_stat['mean_bp']}bp/日, 胜率{tr_stat['win%']}%)")
    print(f"验证期同月:{va_stat['mean_bp']}bp/日 → 方向一致: {ok}")

    print("\n=== ② Turn-of-Month ===")
    print("训练(15-19):", tom_table(bret, 2015, 2019), " 验证(20-26):", tom_table(bret, 2020, 2026))

    print("\n=== ③ 长假前(9月末/1月末,前3日)===")
    print("训练(15-19):", preholiday_table(bret, 2015, 2019), " 验证(20-26):", preholiday_table(bret, 2020, 2026))

    # 应用:验证通过的季节窗口 → P4 状态机加"日历防守"(与波动防守取并集)
    print("\n=== 融入 P4 ===")
    c = pickle.load(open(f"{BASE}/data/fz_cache.pkl", 'rb'))
    base = research.combine(c['Fz'], ['G1_turn20', 'G4_vratio', 'G3_amtshare', 'G10_idiomom'])
    ret = p['close'].pct_change(fill_method=None)
    lowvol = research.zscore({'lv': -(ret.rolling(60).std())})['lv']
    bench_s = b
    bvol_rank = bench_s.reindex(p['close'].index).ffill().pct_change().rolling(20).std().rolling(250).rank(pct=True)
    risk_on_vol = (bvol_rank < 0.6).values
    if ok:
        # 日历防守:验证过的弱月
        cal_defense = (p['close'].index.month == weak_m)
        risk_on = risk_on_vol & (~cal_defense)
        tag = f"+日历防守({weak_m}月)"
    else:
        risk_on = risk_on_vol
        tag = "(弱月未通过验证,不加日历)"
    s = pd.DataFrame(np.where(risk_on[:, None], base.values, (base*0.5 + lowvol*0.5).values),
                     index=p['close'].index, columns=p['close'].columns)
    r4c = engine.run_strategy(p, s, top_n=5, freq='M', start='2020-01-01')
    st = engine.perf_stats(r4c['nav'], bench=bench_s)
    print(f"P4+日历 {tag}: 年化 {st['annual_return']:+.1%} | Sharpe {st['sharpe']:.2f} | 回撤 {st['max_drawdown']:.0%} (对照 P4: 19.8%/-32%)")
    json.dump({'weak_month': weak_m, 'validated': bool(ok)}, open(f"{BASE}/reports/calendar.json", 'w'))
