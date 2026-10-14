"""第二波新因子 A1-A8(全部日线可算,基于 basis)。
轻量回测:单配置 RankIC + Top5 月频(2020起) + 与现有12因子相关性。"""
import os
import pickle, json, time
import numpy as np
import pandas as pd
import panel, research, engine, export

BASE = os.environ.get("FACTORLAB_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EPS = 1e-9
START = '2020-01-01'

def compute_new_factors(p):
    close = p['close']; openp = p['open']; vol = p['volume']; turn = p['turn']; high = p['high']; low = p['low']
    ret = close.pct_change(fill_method=None)
    gap = openp / close.shift(1) - 1            # 隔夜跳空(昨收→今开)
    intr = close / openp - 1                     # 日内(开→收)
    F = {}

    # A1 隔夜跳空反转:近20日隔夜收益和 ×(-1)(大=好:被高开损耗少的)
    F['A1_nightrev'] = -(gap.rolling(20).sum())

    # A2 日内收益占比:20日 Σ日内 / (Σ日内+Σ隔夜)
    ig = intr.rolling(20).sum(); gg = gap.rolling(20).sum()
    F['A2_intradayshare'] = ig / (ig.abs() + gg.abs() + EPS)

    # A3 触板衰竭:120日内「高开≥9.7% 且 日内收跌」次数 ×(-1)
    lim = (gap >= 0.095) & (intr < 0)
    F['A3_limitfade'] = -(lim.rolling(120).sum())

    # A4 波动率凸性:G8 的加速度——(20日下行/上行波动)/(60日同) - 1 ×(-1)
    up = ret.where(ret > 0); dn = ret.where(ret < 0)
    r20 = (dn.rolling(20).std() / (up.rolling(20).std() + EPS))
    r60 = (dn.rolling(60).std() / (up.rolling(60).std() + EPS))
    F['A4_volconvex'] = -(r20 / (r60 + EPS) - 1)

    # A5 量价弹性不对称:下跌日 |收益|~log量 相关 - 上涨日相关 ×(-1)
    lv = np.log(vol + 1)
    def corr_sub(mask):
        r_m = ret.where(mask); v_m = lv.where(mask)
        mc = r_m - r_m.rolling(30).mean(); ml = v_m - v_m.rolling(30).mean()
        num = (mc * ml).rolling(30).mean()
        den = np.sqrt((mc**2).rolling(30).mean() * (ml**2).rolling(30).mean() + EPS)
        return num / den
    c_up = corr_sub(ret > 0); c_dn = corr_sub(ret < 0)
    F['A5_volbeta'] = -(c_dn.fillna(0) - c_up.fillna(0))

    # A6 特质收益偏度(60日)×(-1):彩票票回避
    mkt = ret.mean(axis=1)
    x = ret.sub(mkt, axis=0)
    m2 = (x ** 2).rolling(60).mean(); m3 = (x ** 3).rolling(60).mean()
    F['A6_skew'] = -(m3 / (m2 ** 1.5 + EPS))

    # A7 拥挤度代理:20日换手分位 × 60日涨幅分位 ×(-1)
    tq = turn.rolling(20).mean().rank(axis=1, pct=True)
    rq = close.pct_change(60).rank(axis=1, pct=True)
    F['A7_crowd'] = -(tq * rq)

    # A8 动量断裂:20日特质动量 - 60日特质动量(减速) ×(-1)
    var_m = mkt.rolling(60).var()
    cov_im = x.mul(mkt, axis=0).rolling(60).mean() - x.rolling(60).mean().mul(mkt.rolling(60).mean(), axis=0)
    beta = cov_im.div(var_m + EPS, axis=0)
    idio = ret - beta.mul(mkt, axis=0)
    im20 = idio.rolling(20).sum(); im60 = idio.rolling(60).sum()
    F['A8_mombreak'] = -(im20 / (im60.abs() + EPS))

    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in F.items()}

def quick_test(p, fac, name, c):
    ic = engine.monthly_rank_ic(fac, p, 21, START)
    if len(ic) < 10:
        return {'name': name, 'ic': None}
    # 方向:按 IC 自动对齐(研究期)
    icir = ic.mean() / ic.std() if ic.std() > 0 else 0
    if icir < 0:
        fac = -fac
        ic = engine.monthly_rank_ic(fac, p, 21, START)
        icir = ic.mean() / ic.std() if ic.std() > 0 else 0
    r = engine.run_strategy(p, fac, top_n=5, freq='M', start=START)
    st = engine.perf_stats(r['nav'])
    # 与现有因子相关性(横截面 rank,采样日)
    sample = [d for d in p['close'].index if d >= pd.Timestamp(START)][-12::3]
    cors = {}
    for k, vf in c['Fz'].items():
        vals = []
        for d in sample:
            df = pd.concat([fac.loc[d], vf.loc[d]], axis=1).dropna()
            if len(df) > 200:
                vals.append(abs(df.corr(method='spearman').iloc[0, 1]))
        cors[k] = np.mean(vals) if vals else np.nan
    vals = [v for v in cors.values() if not np.isnan(v)]
    if vals:
        maxcorr = max(vals); argmax = list(cors)[list(cors.values()).index(maxcorr)]
    else:
        maxcorr, argmax = np.nan, '—'
    return {'name': name, 'ic_mean': round(float(ic.mean()), 4), 'icir': round(float(icir), 2),
            't': round(float(ic.mean() / (ic.std() / np.sqrt(len(ic)))), 1),
            'annual': round(float(st['annual_return']), 4), 'sharpe': round(float(st['sharpe']), 2),
            'mdd': round(float(st['max_drawdown']), 3),
            'max_corr': round(float(maxcorr), 2), 'corr_with': argmax}

if __name__ == '__main__':
    t0 = time.time()
    p = panel.build_panel()
    c = pickle.load(open(f"{BASE}/data/fz_cache.pkl", 'rb'))
    F = compute_new_factors(p)
    print(f"8 new factors · {time.time()-t0:.0f}s", flush=True)
    out = []
    for name, fac in F.items():
        res = quick_test(p, fac, name, c)
        out.append(res)
        print(res, flush=True)
    json.dump(out, open(f"{BASE}/reports/new_factors.json", 'w'), indent=1, ensure_ascii=False)
    print(f"NEW FACTORS DONE · {time.time()-t0:.0f}s", flush=True)
