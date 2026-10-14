"""策略定义:P1 量价共振(多因子线性合成)、P2 缩量回踩反转(事件型)。"""
import numpy as np
import pandas as pd
import engine

# ---------- P1 量价共振:四视角组内等权、组间等权 ----------
GROUPS = {
    '关注度': ['G2_attention', 'G3_amtshare'],          # 注意力跃升+量能迁移
    '反转过热': ['G1_turn20', 'G9_ret20', 'G6_chip_pnl'],  # 已方向统一(大=好)
    '波动结构': ['G8_upvol', 'G7_ampc'],                # 上涨力道/振幅收缩
    '动量确认': ['G10_idiomom', 'G11_upratio'],
}

def p1_score(Fz, names=None):
    """names 覆盖时用子集(用于简洁性实验)。"""
    parts = []
    for g, factors_g in GROUPS.items():
        use = [f for f in factors_g if names is None or f in names]
        if not use:
            continue
        s = Fz[use[0]].copy()
        for f in use[1:]:
            s = s.add(Fz[f], fill_value=0)
        parts.append(s / len(use))
    total = parts[0].copy()
    for x in parts[1:]:
        total = total.add(x, fill_value=0)
    return total / len(parts)

def p1_factors_used(chosen):
    return [f for g in GROUPS.values() for f in g if f in chosen]

# ---------- P2 缩量回踩反转(事件型) ----------
def p2_score(F, quantile=0.7):
    """条件:特质动量前30% + 回撤位于35%~70%区间 + 量能收缩 + 近期回调。
    score = 特质动量;不满足条件 → NaN(自然空仓)。"""
    mom = F['G10_idiomom']
    thr = mom.quantile(quantile, axis=1)          # 每日横截面分位
    cond_mom = mom.ge(thr, axis=0)
    cond_dd = (F['G12_ddpos'] >= 0.35) & (F['G12_ddpos'] <= 0.70)
    cond_vol = F['G4_vratio'] < 1.0
    cond_pull = F['G9_ret20'] < 0
    mask = cond_mom & cond_dd & cond_vol & cond_pull
    return mom.where(mask)

P2_PARAMS = dict(top_n=15, freq='M')

def sensitivity_grid(p, score, tops=(30, 50, 100), freqs=('M', '2W')):
    out = []
    for tn, fq in itertools_grid(tops, freqs):
        r = engine.run_strategy(p, score, top_n=tn, freq=fq, start='2016-07-01')
        st = engine.perf_stats(r['nav'])
        out.append({'top_n': tn, 'freq': fq, 'annual': st['annual_return'],
                    'sharpe': st['sharpe'], 'mdd': st['max_drawdown']})
    return out

def itertools_grid(a, b):
    return [(x, y) for x in a for y in b]
