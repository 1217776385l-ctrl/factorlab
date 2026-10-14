"""A股月频组合回测引擎(分块持仓法,单次遍历)。
规则:T 日信号 → T+1 开盘等权买入;持有至下一信号日 T+1 开盘卖出。
约束:开盘一字涨停/停牌→放弃买入;跌停/停牌→顺延至可交易首日开盘卖出(超15日按收盘强平)。
成本:换手率 × 双边合计费率。已知近似(报告披露):顺延卖出期间旧仓收益按原权重计入。
"""
import os
import numpy as np
import pandas as pd

BASE = os.environ.get("FACTORLAB_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COST = 0.002

def _thresholds(cols):
    return pd.Series([0.195 if c.startswith(('sz.30', 'sh.68')) else 0.097 for c in cols], index=cols)

def rebalance_dates(dates, freq='M', start='2016-07-01'):
    s = pd.Series(dates, index=dates)
    if freq == 'M':
        rb = s.groupby(s.index.to_period('M')).last()
    elif freq == 'W':
        rb = s[s.index.dayofweek == 4]          # 每周五
    else:  # 2W:每两周五
        rb = s[s.index.dayofweek == 4].groupby(pd.Grouper(freq='2W-FRI')).last()
    return [pd.Timestamp(d) for d in rb.dropna().values if pd.Timestamp(d) >= pd.Timestamp(start)]

def eligible_universe(p, date, amt_q=0.2, min_hist=250):
    st = p['isST'].loc[date]; susp = p['suspended'].loc[date]
    amt20 = p['amount'].loc[:date].tail(20).mean()
    alive = p['close'].loc[:date].notna().sum()
    thr = amt20.quantile(amt_q)
    ok = (~st) & (~susp) & (alive >= min_hist) & (amt20 > thr) & amt20.notna()
    return ok[ok].index

def run_strategy(p, score, *, top_n=50, freq='M', start='2016-07-01', cost=COST, name='strategy'):
    """向量化分块回测:numpy 区间累加,单次 <1s。规则同前(T+1开盘/一字板不追/跌停顺延15日)。"""
    import numpy as np
    dates = pd.DatetimeIndex(p['close'].index)
    N, M = len(dates), p['close'].shape[1]
    cols = list(p['close'].columns)
    cidx = {c: j for j, c in enumerate(cols)}
    close = p['close'].to_numpy(dtype=np.float64)
    openp = p['open'].to_numpy(dtype=np.float64)
    susp = p['suspended'].to_numpy(dtype=bool)
    isST = p['isST'].to_numpy(dtype=bool)
    with np.errstate(divide='ignore', invalid='ignore'):
        gap = openp / np.vstack([close[:1], close[:-1]]) - 1.0
        cr = np.zeros_like(close); cr[1:] = close[1:] / close[:-1] - 1.0
        er = close / openp - 1.0                      # 买入首日 开→收
        orr = gap.copy()                              # 卖出日 昨收→开
    cr = np.nan_to_num(cr); er = np.nan_to_num(er); orr = np.nan_to_num(orr); gap = np.nan_to_num(gap)
    th = np.array([0.195 if c.startswith(('sz.30', 'sh.68')) else 0.097 for c in cols])
    dpos = {d: i for i, d in enumerate(dates)}
    start_i = int(np.searchsorted(dates, pd.Timestamp(start)))
    rbs = [dpos[d] for d in rebalance_dates(dates, freq, start) if d in dpos]

    # 信号矩阵(score)对齐到面板网格
    S = np.full((N, M), np.nan)
    sc = score.reindex(index=dates, columns=cols)
    S[:] = sc.to_numpy(dtype=np.float64)
    sig_rows = np.where(~np.isnan(S).all(axis=1))[0]

    # 预计算:20日均额 与 累计有效日( Universe 过滤)
    amt = p['amount'].to_numpy(dtype=np.float64)
    amt20 = np.full((N, M), np.nan)
    if N > 0:
        valid = ~np.isnan(amt)
        csum = np.nancumsum(amt, axis=0); cn = valid.cumsum(axis=0)
        i_idx = np.arange(N); lo = np.maximum(i_idx - 19, 0)
        has_prev = lo > 0
        hp2 = has_prev[:, None]
        num = csum[i_idx] - np.where(hp2, csum[np.maximum(lo - 1, 0)], 0.0)
        den = cn[i_idx] - np.where(hp2, cn[np.maximum(lo - 1, 0)], 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            row = np.where(den > 0, num / np.maximum(den, 1), np.nan)
        amt20[:] = row
    amt20q = np.nanquantile(amt20, 0.2, axis=1)
    alive = (~np.isnan(close)).cumsum(axis=0)

    H_mid = np.zeros((N, M), dtype=np.float64)
    H_ent = np.zeros((N, M), dtype=np.float64)
    H_ext = np.zeros((N, M), dtype=np.float64)
    Wsum = np.zeros(N, dtype=np.float64)
    picks_hist, turnover_log = {}, {}
    prev_picks = set(); blocked_buys = n_trades = 0
    hold = np.zeros(M, dtype=bool)

    for bi, si in enumerate(rbs):
        ib = si + 1
        if ib >= N: break
        ie = min((rbs[bi + 1] if bi + 1 < len(rbs) else N - 1) + 1, N - 1)
        # 最近可用信号(≤40天)
        sr = sig_rows[sig_rows <= si]
        if len(sr) == 0 or si - sr[-1] > 40: continue
        row = S[sr[-1]]
        ok_amt = amt20[si] > amt20q[si]
        ok_alive = alive[si] >= 250
        ok = ok_amt & ok_alive & (~susp[si]) & (~isST[si])
        cand = np.where(ok & ~np.isnan(row))[0]
        if len(cand) == 0: continue
        # 确定性排序:分数降序 + 列序 tie-break(消除并列值的任意挑选)
        order = np.lexsort((cand, -row[cand]))
        picks = cand[order[:top_n]]
        w = 1.0 / len(picks)
        buy_d_date = dates[ib]
        picks_hist[buy_d_date] = [cols[j] for j in picks]
        overlap = len(set(picks.tolist()) & prev_picks) / len(picks)
        turnover_log[buy_d_date] = 1 - overlap
        prev_picks = set(picks.tolist())
        for j in picks:
            # 买入日:首个 非停牌 且 开盘未一字涨停
            bd = -1
            for k in range(ib, ie + 1):
                if susp[k, j]: continue
                if gap[k, j] >= th[j]: continue
                bd = k; break
            if bd < 0:
                blocked_buys += 1; continue
            n_trades += 1
            sd = -1
            for k in range(ie, min(ie + 15, N)):
                if susp[k, j]: continue
                if gap[k, j] <= -th[j]: continue
                sd = k; break
            forced = sd < 0
            if forced:
                sd = min(ie + 15, N - 1)
            H_ent[bd, j] += w
            if sd > bd + 1:
                H_mid[bd + 1:sd, j] += w
            H_ext[sd, j] += w
            Wsum[bd:sd + 1] += w
    gross = (H_ent * er).sum(axis=1) + (H_mid * cr).sum(axis=1) + (H_ext * orr).sum(axis=1)
    # 换手成本计入分子(与组合收益同口径归一)
    for d, to in turnover_log.items():
        gross[dpos[d]] -= to * cost
    port = np.zeros(N)
    act = Wsum > 0
    port[act] = gross[act] / Wsum[act]
    nav = pd.Series(np.cumprod(1 + port[start_i:]), index=dates[start_i:])
    port_s = pd.Series(port, index=dates)
    return {'nav': nav, 'port_ret': port_s, 'picks_hist': picks_hist,
            'turnover': pd.Series(turnover_log), 'blocked_buys': blocked_buys,
            'n_trades': n_trades, 'n_periods': len(picks_hist)}

def layered(p, factor, n_layers=5, **kw):
    out = {}
    rank = factor.rank(axis=1, pct=True)
    for L in range(1, n_layers + 1):
        lo, hi = (L - 1) / n_layers, L / n_layers
        sliced = factor.where((rank > lo) & (rank <= hi))
        out[L] = run_strategy(p, sliced, **kw)
    return out

def monthly_rank_ic(factor, p, fwd_days=21, start='2016-07-01'):
    close = p['close']
    fwd = close.shift(-fwd_days) / close - 1
    ics = {}
    for d in rebalance_dates(pd.DatetimeIndex(p['close'].index), 'M', start):
        if d not in factor.index: continue
        df = pd.concat([factor.loc[d], fwd.loc[d]], axis=1).dropna()
        if len(df) > 100:
            ics[d] = float(df.corr(method='spearman').iloc[0, 1])
    return pd.Series(ics)

def perf_stats(nav, bench=None, fpy=252):
    nav = nav.sort_index().dropna()
    ret = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    ann = (nav.iloc[-1] / nav.iloc[0]) ** (1 / max(years, 1e-9)) - 1
    vol = ret.std() * np.sqrt(fpy)
    dd = (nav / nav.cummax() - 1).min()
    out = {'total_return': float(nav.iloc[-1] / nav.iloc[0] - 1), 'annual_return': float(ann),
           'annual_vol': float(vol), 'sharpe': float(ann / vol) if vol > 0 else 0.0,
           'max_drawdown': float(dd), 'calmar': float(ann / abs(dd)) if dd < 0 else 0.0,
           'start': str(nav.index[0].date()), 'end': str(nav.index[-1].date()), 'years': round(years, 2),
           'win_rate_period': float((ret > 0).mean())}
    if bench is not None:
        b = bench.reindex(nav.index).ffill().dropna()
        b_ret = b.pct_change().reindex(ret.index).fillna(0)
        exc = ret - b_ret
        b_ann = (b.iloc[-1] / b.iloc[0]) ** (1 / max(years, 1e-9)) - 1
        out['bench_annual'] = float(b_ann)
        out['excess_annual'] = float(ann - b_ann)
        out['info_ratio'] = float(exc.mean() / exc.std() * np.sqrt(fpy)) if exc.std() > 0 else 0.0
        out['bench_nav'] = b
    return out
