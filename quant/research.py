"""因子研究流程:体检 → 相关性 → 筛选 → 合成 → 参数敏感性 → 简洁性。输出 JSON + 报告素材。"""
import os
import json, pickle, itertools
import numpy as np
import pandas as pd
import panel, factors, engine

BASE = os.environ.get("FACTORLAB_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
START = '2016-07-01'          # 前 120 日滚动窗口预热
IS_END, OOS_START = '2021-12-31', '2022-01-01'
KW = dict(top_n=50, freq='M', start=START)

def direction_fix(F):
    """统一方向:先验「低好」的因子取负 → 全部「大=好」,便于合成。"""
    low_good = ['G1_turn20', 'G7_ampc', 'G9_ret20', 'G6_chip_pnl']  # G6 筹码盈亏:获利少→抛压小(先验)
    return {k: (-v if k in low_good else v) for k, v in F.items()}

def empirical_direction(F, health):
    """实证方向修正:按单因子 IC 符号再翻转(IC<0 → 取负),全部变成「大=好」。
    这是纯技术性统一:选因子/合成前必须与实证方向一致,否则组合方向做反。"""
    sign = {r['factor']: (1 if r['ic_mean'] >= 0 else -1) for r in health}
    return {k: (v * sign.get(k, 1)) for k, v in F.items()}

def zscore(F):
    return {k: v.sub(v.mean(axis=1), axis=0).div(v.std(axis=1) + 1e-12, axis=0).clip(-5, 5) for k, v in F.items()}

def ichealth(p, F, fwd_days=21):
    rows = []
    for k, f in F.items():
        ic = engine.monthly_rank_ic(f, p, fwd_days, START)
        is_ic = ic[ic.index <= pd.Timestamp(IS_END)]
        oos_ic = ic[ic.index >= pd.Timestamp(OOS_START)]
        rows.append({'factor': k, 'ic_mean': ic.mean(), 'icir': ic.mean() / ic.std() if ic.std() > 0 else 0,
                     't_stat': ic.mean() / (ic.std() / np.sqrt(len(ic))) if ic.std() > 0 else 0,
                     'is_ic': is_ic.mean(), 'oos_ic': oos_ic.mean(),
                     'pos_years': float((ic.groupby(ic.index.year).mean() > 0).mean()),
                     'n': len(ic), 'ic_series': ic})
    return rows

def cross_corr(F, sample_dates):
    """因子间横截面 rank 相关(按采样日平均)。"""
    names = list(F)
    cors = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for a, b in itertools.combinations(names, 2):
        vals = []
        for d in sample_dates:
            if d in F[a].index and d in F[b].index:
                df = pd.concat([F[a].loc[d], F[b].loc[d]], axis=1).dropna()
                if len(df) > 200:
                    vals.append(df.corr(method='spearman').iloc[0, 1])
        cors.loc[a, b] = cors.loc[b, a] = np.mean(vals) if vals else np.nan
    return cors

def greedy_select(health, F, max_per_group_corr=0.6, topk=6):
    """按 |ICIR| 贪心选因子:与已选因子的平均横截面相关 < 0.6。"""
    order = sorted(health, key=lambda r: abs(r['icir']), reverse=True)
    chosen = []
    sample = sorted(F[order[0]['factor']].index)[-24::3]
    for r in order:
        k = r['factor']
        if len(chosen) >= topk:
            break
        ok = True
        for c in chosen:
            vals = []
            for d in sample:
                df = pd.concat([F[k].loc[d], F[c].loc[d]], axis=1).dropna()
                if len(df) > 200:
                    vals.append(abs(df.corr(method='spearman').iloc[0, 1]))
            if vals and np.mean(vals) > max_per_group_corr:
                ok = False; break
        if ok:
            chosen.append(k)
    return chosen

def combine(Fz, names, weights=None):
    """等权/加权合成。规则:有效因子数 < 一半 → NaN(不许无信息股票混进排序);
    不做 0 填充(0 填充会造成大量并列分,TopN 边缘在并列带内任意取值)。"""
    arr = np.stack([Fz[n].to_numpy(dtype=np.float64) for n in names])
    w = np.asarray(weights if weights is not None else [1.0] * len(names), dtype=np.float64)
    w = (w / np.nansum(w))[:, None, None]
    mask = ~np.isnan(arr)
    cnt = mask.sum(axis=0)
    total = np.nansum(np.where(mask, arr * w, 0.0), axis=0)
    mean = total / np.maximum(cnt, 1)
    mean[cnt < (len(names) + 1) // 2] = np.nan
    return pd.DataFrame(mean, index=Fz[names[0]].index, columns=Fz[names[0]].columns)

def run_all():
    p = panel.build_panel()
    print('panel ready', flush=True)
    F = direction_fix(factors.compute_factors(p))
    Fz = zscore(F)
    print('factors ready', flush=True)

    health = ichealth(p, F)
    for r in health:
        r.pop('ic_series')
    print('health done', flush=True)

    sample_dates = [d for d in rebalance_sample(p, START)][-20::2]
    C = cross_corr(Fz, sample_dates)
    print('corr done', flush=True)

    chosen = greedy_select(health, Fz)
    print('chosen:', chosen, flush=True)

    fwd = p['close'].shift(-21) / p['close'] - 1
    return p, F, Fz, health, C, chosen

def rebalance_sample(p, start):
    return engine.rebalance_dates(pd.DatetimeIndex(p['close'].index), 'M', start)

if __name__ == '__main__':
    p, F, Fz, health, C, chosen = run_all()
    with open(f'{BASE}/data/research_cache.pkl', 'wb') as f:
        pickle.dump({'health': health, 'corr': C, 'chosen': chosen}, f)
    print('saved')
