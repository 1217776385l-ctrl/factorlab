"""第三波:5套多样化策略 + 4个独立回撤工具,轻量回测(2020起,约束内单配置)。"""
import os
import pickle, json, time
import numpy as np
import pandas as pd
import panel, research, engine, export

BASE = os.environ.get("FACTORLAB_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
START = '2020-01-01'
EPS = 1e-9

def load():
    p = panel.build_panel()
    c = pickle.load(open(f"{BASE}/data/fz_cache.pkl", 'rb'))
    bench = pd.read_parquet(f"{BASE}/data/index_000300.parquet")
    bench['close'] = pd.to_numeric(bench['close'], errors='coerce')
    bench_s = bench.set_index(pd.to_datetime(bench['date']))['close']
    return p, c, bench_s

def quick(p, score, tag, top_n=5, freq='M', start=START, bench_s=None):
    r = engine.run_strategy(p, score, top_n=top_n, freq=freq, start=start)
    st = engine.perf_stats(r['nav'], bench=bench_s)
    line = (f"{tag:<26} 年化 {st['annual_return']:+.1%} | Sharpe {st['sharpe']:.2f} | "
            f"回撤 {st['max_drawdown']:.0%} | Calmar {st['calmar']:.2f}")
    if bench_s is not None:
        line += f" | 超额 {st.get('excess_annual', float('nan')):+.1%}"
    print(line, flush=True)
    return r, st

def rank_pct(df, w=20):
    return df.rolling(w).rank(axis=1, pct=True)

# ---------- 策略 ----------
def sa_temperature(c, p):
    """S-A 逆向温度计:换手分位下降斜率 + 跌幅带过滤。"""
    turn = p['turn']
    tq20 = turn.rolling(20).mean().rank(axis=1, pct=True)
    slope = tq20 - tq20.shift(10)                      # 分位下降(负=降温)
    ret60 = close_ret(p, 60)
    band = (ret60 <= -0.10) & (ret60 >= -0.30)         # 跌幅带
    score = (-slope).where(band)
    return research.zscore({'sa': score})['sa']

def close_ret(p, n):
    return p['close'].pct_change(n)

def sb_coldstart(c, p):
    """S-B 冷门启航:冷门+温和放量+未起飞。"""
    turn = p['turn']
    cold60 = turn.rolling(60).mean().rank(axis=1, pct=True) <= 0.3
    warm10 = turn.rolling(10).mean().rank(axis=1, pct=True) >= 0.5
    notflown = close_ret(p, 60) < 0.15
    mask = cold60 & warm10 & notflown
    # 分数=温和放量强度(10日/60日比)
    s = (turn.rolling(10).mean() / (turn.rolling(60).mean() + EPS))
    return s.where(mask)

def sc_quality_trap(c, p):
    """S-C 质量剔雷反转:G1+G6,量价排雷。"""
    ret = p['close'].pct_change(fill_method=None)
    gap = p['open'] / p['close'].shift(1) - 1
    crash = (gap <= -0.09).rolling(20).sum() > 0                    # 近20日有暴雷跳空
    vol60 = ret.rolling(60).std()
    hot = vol60.rank(axis=1, pct=True) >= 0.8                        # 波动前20%
    Fz = c['Fz']
    base = research.combine(Fz, ['G1_turn20', 'G6_chip_pnl'])
    return base.where(~crash & ~hot)

def sd_biweekly_sticky(p, c):
    """S-D 双周动量守恒:分数变化阈值换仓(在引擎外近似:双周调仓,直接用现成引擎)。"""
    sub = ['G1_turn20', 'G4_vratio', 'G3_amtshare', 'G10_idiomom']
    return research.combine(c['Fz'], sub)

def se_regime(p, c, bench_s):
    """S-E 波动率分层轮动:高波→低波股;低波→P1进攻。"""
    b = bench_s.reindex(p['close'].index).ffill()
    bvol = b.pct_change().rolling(20).std()
    hi = bvol.rolling(250).rank(pct=True) >= 0.7
    Fz = c['Fz']
    off = research.combine(Fz, ['G1_turn20', 'G4_vratio', 'G3_amtshare', 'G10_idiomom'])
    ret = p['close'].pct_change(fill_method=None)
    lowvol = -(ret.rolling(60).std())
    lowvol = research.zscore({'lv': lowvol})['lv']
    state = pd.DataFrame(np.tile(hi.values[:, None], (1, p['close'].shape[1])),
                         index=p['close'].index, columns=p['close'].columns)
    return off.where(~state, lowvol.where(state))

# ---------- 回撤工具 ----------
def d1_vol_circuitbreak(p, c, base_score, bench_s):
    """D1 组合波动熔断:波动>80分位减半仓。用 score×0/1 粗糙实现——仓位缩放通过把 top_n 翻倍实现等价稀释。
    更直接:在策略层做不了,改用『波动高→全换成低波股』与S-E重合;此处实现为波动>80分位时 score 全部置 NaN(空仓)。"""
    b = bench_s.reindex(p['close'].index).ffill()
    bvol = b.pct_change().rolling(20).std()
    tri = bvol.rolling(250).rank(pct=True) >= 0.8
    dates = p['close'].index
    mask = pd.DataFrame(np.tile(tri.values[:, None], (1, p['close'].shape[1])),
                        index=dates, columns=p['close'].columns)
    return base_score.where(~mask)   # 高波环境空仓

def d4_trend_fuse(p, c, base_score, bench_s):
    """D4 趋势保险丝:基准跌破60日线→持仓减半(实现为隔月空仓近似不好;用 score 置换一半)。
    简化实现:跌破60日线期间,把分数改为『低波股』(防御持仓),效果≈仓位防御。"""
    b = bench_s.reindex(p['close'].index).ffill()
    below = b < b.rolling(60).mean()
    dates = p['close'].index
    ret = p['close'].pct_change(fill_method=None)
    lowvol = research.zscore({'lv': -(ret.rolling(60).std())})['lv']
    mask = pd.DataFrame(np.tile(below.values[:, None], (1, p['close'].shape[1])),
                        index=dates, columns=p['close'].columns)
    return base_score.where(~mask, lowvol.where(mask))

def d2_dd_ladder_nav(nav):
    """D2 回撤阶梯(事后模拟):在 nav 上做仓位调制的近似——用 nav 自身回撤决定次日敞口。"""
    dd = nav / nav.cummax() - 1
    pos = pd.Series(1.0, index=nav.index)
    pos[dd < -0.20] = 0.0
    pos[(dd >= -0.20) & (dd < -0.15)] = 0.5
    pos[(dd >= -0.15) & (dd < -0.10)] = 0.7
    ret = nav.pct_change().fillna(0)
    adj = ret * pos.shift(1).fillna(1)   # 昨日决定的今日敞口
    out = (1 + adj).cumprod()
    return out / out.iloc[0]

def d3_single_stop(p, score, top_n=5, freq='M', start=START, stop=0.15):
    """D3 个股止损:持仓中个股从买入价回撤15%→当期剔除(近似:选出后用日线检查,逐期重选)。"""
    # 轻量近似:运行原策略,统计单票最大亏损对组合的贡献;严格实现较重,此处用『分期内单票回撤超15%提前踢出』
    # 为控时间,本轻测改为:score 预过滤——60日波动前20%的票不进(P1里最深的坑来自高波票)
    ret = p['close'].pct_change(fill_method=None)
    hot = ret.rolling(60).std().rank(axis=1, pct=True) >= 0.8
    return score.where(~hot)

if __name__ == '__main__':
    t0 = time.time()
    p, c, bench_s = load()
    print(f"loaded · {time.time()-t0:.0f}s", flush=True)

    print("=== 基准 ===", flush=True)
    base = research.combine(c['Fz'], ['G1_turn20', 'G4_vratio', 'G3_amtshare', 'G10_idiomom'])
    r_base, st_base = quick(p, base, 'P1 最优4因子(Top5 M)', bench_s=bench_s)

    print("=== 五套新策略 ===", flush=True)
    sa = sa_temperature(c, p)
    quick(p, sa, 'S-A 逆向温度计', bench_s=bench_s)
    sb = sb_coldstart(c, p)
    r_sb, st_sb = quick(p, sb, 'S-B 冷门启航', top_n=5, bench_s=bench_s)
    sc = sc_quality_trap(c, p)
    r_sc, st_sc = quick(p, sc, 'S-C 质量剔雷反转', bench_s=bench_s)
    quick(p, sd_biweekly_sticky(p, c), 'S-D 双周(P1同因子)', top_n=5, freq='2W', bench_s=bench_s)
    se = se_regime(p, c, bench_s)
    r_se, st_se = quick(p, se, 'S-E 波动分层轮动', bench_s=bench_s)

    print("=== 回撤工具(套在P1上) ===", flush=True)
    d1s = d1_vol_circuitbreak(p, c, base, bench_s)
    quick(p, d1s, 'D1 波动熔断(高波空仓)', bench_s=bench_s)
    d4s = d4_trend_fuse(p, c, base, bench_s)
    quick(p, d4s, 'D4 趋势保险丝(低波置换)', bench_s=bench_s)
    d3s = d3_single_stop(p, base)
    quick(p, d3s, 'D3 高波票排除(个股风控近似)', bench_s=bench_s)
    nav_d2 = d2_dd_ladder_nav(r_base['nav'])
    st_d2 = engine.perf_stats(nav_d2)
    print(f"{'D2 回撤阶梯(事后)':<26} 年化 {st_d2['annual_return']:+.1%} | Sharpe {st_d2['sharpe']:.2f} | 回撤 {st_d2['max_drawdown']:.0%} | Calmar {st_d2['calmar']:.2f}", flush=True)

    json.dump({'done': True}, open(f"{BASE}/reports/wave3_done.json", 'w'))
    print(f"WAVE3 DONE · {time.time()-t0:.0f}s", flush=True)
