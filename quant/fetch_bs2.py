"""baostock 单次调用拉取(后复权价+原始量额换手),2进程分片,被踢自动重连。
用法: python fetch_bs2.py --shard i --nshards 2
"""
import argparse, os, sys, time
import baostock as bs
import pandas as pd

BASE = os.environ.get("FACTORLAB_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KLINE_DIR = f"{BASE}/data/kline"
START, END = "2015-01-01", "2026-09-30"
FIELDS = "date,code,open,high,low,close,volume,amount,turn,tradestatus,isST"

def get_universe():
    for back in range(10):
        day = (pd.Timestamp.now() - pd.Timedelta(days=back)).strftime("%Y-%m-%d")
        rs = bs.query_all_stock(day=day)
        rows = []
        while rs.error_code == '0' and rs.next():
            rows.append(rs.get_row_data())
        if rows:
            df = pd.DataFrame(rows, columns=rs.fields)
            print(f"universe day={day}, total={len(rows)}", flush=True)
            break
    codes = sorted(set(df[df['code'].str.match(r'(sh\.6[089]|sz\.0[03]|sz\.3)')]['code']))
    return codes

def fetch_one(code):
    out = f"{KLINE_DIR}/{code.replace('.', '_')}.parquet"
    if os.path.exists(out):
        return 'skip'
    rs = bs.query_history_k_data_plus(code, FIELDS, start_date=START, end_date=END,
                                      frequency='d', adjustflag='1')
    rows = []
    while rs.error_code == '0' and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return 'empty'
    df = pd.DataFrame(rows, columns=rs.fields)
    df = df.rename(columns={})  # hfq 价格 + 原始量额换手
    df.to_parquet(out, index=False)
    return 'ok'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=2)
    args = ap.parse_args()
    bs.login()
    codes = get_universe()
    mine = codes[args.shard::args.nshards]
    print(f"shard {args.shard}/{args.nshards}: {len(mine)} codes", flush=True)
    stat = {'ok': 0, 'skip': 0, 'empty': 0, 'err': 0}
    empty_streak = 0
    t0 = time.time()
    for i, code in enumerate(mine):
        try:
            r = fetch_one(code)
            stat[r] += 1
            empty_streak = empty_streak + 1 if r == 'empty' else 0
        except Exception as e:
            stat['err'] += 1
            print(f"ERR {code}: {e}", flush=True)
            try:
                bs.logout()
            except Exception:
                pass
            time.sleep(5); bs.login()
        if empty_streak >= 8:
            print("!! kicked (8 empty) → reconnect", flush=True)
            try:
                bs.logout()
            except Exception:
                pass
            time.sleep(20); bs.login()
            empty_streak = 0
        time.sleep(0.15)
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"[{i+1}/{len(mine)}] {stat} elapsed={el:.0f}s eta={el/(i+1)*(len(mine)-i-1):.0f}s", flush=True)
    print(f"DONE shard{args.shard}: {stat}", flush=True)
    bs.logout()

if __name__ == '__main__':
    main()
