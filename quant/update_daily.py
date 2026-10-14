"""FactorLab DB · 每日增量更新(收盘后跑一次,几分钟)。
1. 拉最新交易日全 A 日线,追加到 data/kline/;
2. 重建 panel.pkl(float32);
3. 增量更新因子缓存(fz_cache.pkl 全量重算,~25s,可接受);
4. 刷新交易日元数据。
用法: python update_daily.py
"""
import os, time
import baostock as bs
import pandas as pd

BASE = os.environ.get("FACTORLAB_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KLINE = f"{BASE}/data/kline"
FIELDS = "date,code,open,high,low,close,volume,amount,turn,tradestatus,isST"

def latest_trading_day():
    for back in range(0, 10):
        day = (pd.Timestamp.now() - pd.Timedelta(days=back)).strftime("%Y-%m-%d")
        rs = bs.query_trade_dates(d1=day, d2=day)
        while rs.error_code == '0' and rs.next():
            if rs.get_row_data()[1] == '1':
                return day
    return None

def update_kline(day):
    """拉取 day 当日全市场日线,仅追加新日期。"""
    added = 0
    # 全市场当日行情:逐股太慢;用 query_history 一天范围批量
    rs = bs.query_all_stock(day=day)
    rows = []
    while rs.error_code == '0' and rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)
    stocks = df[df['code'].str.match(r'(sh\.60[0-9]|sz\.00[0-3]|sz\.30[01])')]
    print(f"universe {day}: {len(stocks)} stocks")
    for code in stocks['code']:
        fp = f"{KLINE}/{code.replace('.', '_')}.parquet"
        r2 = bs.query_history_k_data_plus(code, FIELDS, start_date=day, end_date=day,
                                          frequency='d', adjustflag='1')
        rows2 = []
        while r2.error_code == '0' and r2.next():
            rows2.append(r2.get_row_data())
        if rows2:
            new = pd.DataFrame(rows2, columns=FIELDS.split(','))
            if os.path.exists(fp):
                old = pd.read_parquet(fp)
                if day in set(old['date']):
                    continue
                pd.concat([old, new], ignore_index=True).to_parquet(fp, index=False)
            else:
                new.to_parquet(fp, index=False)
            added += 1
        time.sleep(0.05)
    print(f"kline updated: {added} stocks")

def main():
    t0 = time.time()
    bs.login()
    day = latest_trading_day()
    print("latest trading day:", day, flush=True)
    if day:
        update_kline(day)
    bs.logout()
    # 重建面板
    import panel
    p = panel.build_panel(force=True)
    print(f"panel rebuilt: {p['close'].shape} · {time.time()-t0:.0f}s", flush=True)
    # 因子缓存
    import factors, research
    F = research.direction_fix(factors.compute_factors(p))
    health = research.ichealth(p, F)
    F = research.empirical_direction(F, health)
    Fz = research.zscore(F)
    with open(f"{BASE}/data/fz_cache.pkl", 'wb') as f:
        import pickle
        pickle.dump({'F': F, 'Fz': Fz, 'health_ic': {r['factor']: r['ic_mean'] for r in health}}, f)
    print(f"DB UPDATE COMPLETE · {time.time()-t0:.0f}s", flush=True)

if __name__ == '__main__':
    main()
