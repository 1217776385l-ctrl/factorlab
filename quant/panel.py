"""把逐股 parquet(baostock 后复权价+原始量额)合并为面板,缓存 pickle。"""
import os, pickle
import numpy as np
import pandas as pd

BASE = os.environ.get("FACTORLAB_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get("FACTORLAB_DATA", os.path.join(BASE, "data"))
KLINE_DIR = os.path.join(DATA_DIR, "kline")
CACHE = os.path.join(BASE, "data", "panel.pkl")

def build_panel(force=False):
    if os.path.exists(CACHE) and not force:
        with open(CACHE, 'rb') as f:
            return pickle.load(f)
    frames = []
    for fn in sorted(os.listdir(KLINE_DIR)):
        frames.append(pd.read_parquet(f"{KLINE_DIR}/{fn}"))
    raw = pd.concat(frames, ignore_index=True)
    for c in ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn']:
        raw[c] = pd.to_numeric(raw[c], errors='coerce')
    raw['isST'] = pd.to_numeric(raw['isST'], errors='coerce').fillna(0).astype(int)
    raw['tradestatus'] = pd.to_numeric(raw['tradestatus'], errors='coerce').fillna(0).astype(int)
    raw = raw[(raw['close'] > 0)].copy()
    raw = raw.drop_duplicates(subset=['date', 'code'])
    raw['date'] = pd.to_datetime(raw['date'])
    piv = {}
    for c in ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn']:
        piv[c] = raw.pivot_table(index='date', columns='code', values=c, aggfunc='last')
    st = raw.pivot_table(index='date', columns='code', values='isST', aggfunc='last')
    ts = raw.pivot_table(index='date', columns='code', values='tradestatus', aggfunc='last')
    # 统一命名:价格即后复权口径
    panel = {}
    for k in ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn']:
        panel[k] = piv[k].sort_index().astype(np.float32)
    panel['isST'] = (st == 1).reindex(panel['close'].index).fillna(True)   # 缺信息日按ST保守处理
    panel['suspended'] = ((ts == 0) | piv['close'].isna() | (piv['volume'] <= 0)).reindex(panel['close'].index).fillna(True)
    with open(CACHE, 'wb') as f:
        pickle.dump(panel, f)
    return panel

if __name__ == '__main__':
    p = build_panel(force=True)
    print('dates:', p['close'].index[0].date(), '→', p['close'].index[-1].date(), '| stocks:', p['close'].shape[1])
