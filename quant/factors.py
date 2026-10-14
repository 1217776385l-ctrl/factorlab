"""量价因子库 G1-G12。输入 panel,输出 {因子名: DataFrame(日期×代码)}。
全部基于后复权价 + 原始量/额/换手率,只用 T 日及之前信息。
注意方向:研究阶段统一为「因子值越大 → 越好」后进组合;单因子先验方向见注释。
"""
import numpy as np
import pandas as pd

EPS = 1e-9

def compute_factors(p):
    close = p['close']           # 后复权
    high = p['high']; low = p['low']
    vol = p['volume']; amt = p['amount']; turn = p['turn']
    ret = close.pct_change(fill_method=None)
    F = {}

    # G1 换手率水平:先验「高=过热→低好」
    F['G1_turn20'] = turn.rolling(20).mean()

    # G2 关注度跃升:5日均换手 / 60日均换手,先验「高好」
    F['G2_attention'] = turn.rolling(5).mean() / (turn.rolling(60).mean() + EPS)

    # G3 量能市场分位迁移:个股成交额份额 5日均 / 60日均 - 1,先验「高好」
    share = amt.div(amt.sum(axis=1), axis=0)
    F['G3_amtshare'] = share.rolling(5).mean() / (share.rolling(60).mean() + EPS) - 1

    # G4 放量倍数:5日均量 / 60日均量,先验「分位置:低位放量好」
    F['G4_vratio'] = vol.rolling(5).mean() / (vol.rolling(60).mean() + EPS)

    # G5 量价相关:20日 corr(收益, log成交额),先验「正好」
    lv = np.log(amt + 1)
    mc = ret - ret.rolling(20).mean()
    ml = lv - lv.rolling(20).mean()
    F['G5_pvcorr'] = (mc * ml).rolling(20).mean() / np.sqrt(
        (mc ** 2).rolling(20).mean() * (ml ** 2).rolling(20).mean() + EPS)

    # G6 筹码盈亏代理:换手率加权 20 日收益(交易者平均盈亏),先验「适度/偏低好」
    cap = float(np.nanpercentile(turn.values, 99.9)) if bool(turn.notna().any().any()) else 100.0
    w = turn.clip(upper=cap).fillna(0)
    F['G6_chip_pnl'] = (w * ret.fillna(0)).rolling(20).sum() / (w.rolling(20).sum() + EPS)

    # G7 振幅收缩:5日均振幅 / 60日均振幅,先验「低=蓄势」
    amp = (high - low) / close
    F['G7_ampc'] = amp.rolling(5).mean() / (amp.rolling(60).mean() + EPS)

    # G8 波动不对称:上涨日波动 / 下跌日波动(20日),先验「高好」
    up = ret.where(ret > 0); dn = ret.where(ret < 0)
    F['G8_upvol'] = up.rolling(20, min_periods=8).std() / (dn.rolling(20, min_periods=8).std() + EPS)

    # G9 短期反转:20日收益,先验「低好」
    F['G9_ret20'] = close.pct_change(20)

    # G10 特质动量:60日 Σ(个股收益 - beta×市场收益),先验「高好」
    mkt = ret.mean(axis=1)
    x = ret.sub(mkt, axis=0)
    var_m = mkt.rolling(60).var()
    cov_im = x.mul(mkt, axis=0).rolling(60).mean() - x.rolling(60).mean().mul(mkt.rolling(60).mean(), axis=0)
    beta = cov_im.div(var_m + EPS, axis=0)
    F['G10_idiomom'] = (ret - beta.mul(mkt, axis=0)).rolling(60).sum()

    # G11 动量质量:60日上涨日占比,先验「高好」
    F['G11_upratio'] = (ret > 0).rolling(60).mean()

    # G12 回撤位置(斐波那契连续化):0=120日最高点,1=最低点。复权价口径(跨除权正确)。
    hh = high.rolling(120).max(); ll = low.rolling(120).min()
    F['G12_ddpos'] = ((hh - close) / (hh - ll + EPS)).clip(0, 1)

    F = {k: v.replace([np.inf, -np.inf], np.nan) for k, v in F.items()}
    return F
