"""P3 · 机器学习截面模型(LightGBM):
- 特征:12 因子 z 值 + 市场状态变量 + 因子×状态交互(让模型自己学「什么状态下信哪个因子」)
- 标签:未来 21 日收益的横截面 rank(0~1)
- 训练:expanding-window walk-forward,逐年滚动; embargo=25 日防标签重叠泄漏
- 防过拟合:超参保守固定,不调参;严格时间切分;与 P1 线性同引擎同约束对比
输出:results/p3_ml.json + reports 追加素材
"""
import json, pickle, time, warnings, os
import numpy as np
import pandas as pd
import lightgbm as lgb
import panel, factors, research, strategies, engine, export

warnings.filterwarnings('ignore')
BASE = os.environ.get("FACTORLAB_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
START = '2016-07-01'
FWD = 21

PARAMS = dict(objective='regression', num_leaves=15, max_depth=4, learning_rate=0.05,
              n_estimators=250, min_child_samples=200, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=5.0, n_jobs=4, verbose=-1, random_state=7)

def build_features(p, Fz):
    """因子 z + 状态变量 + 交互。全部只用 T 日信息。"""
    feats = {}
    for k, v in Fz.items():
        feats[f'f_{k}'] = v
    close = p['close']; turn = p['turn']
    ret = close.pct_change(fill_method=None)
    mkt = ret.mean(axis=1)
    # 市场状态(每日一个值,广播到全截面)
    mkt_mom20 = mkt.rolling(20).sum()                              # 大盘动量
    mkt_vol20 = mkt.rolling(20).std() * np.sqrt(252)               # 大盘波动
    mkt_turn = turn.mean(axis=1)                                   # 全市场平均换手
    mkt_turn_pct = mkt_turn.rolling(250).rank(pct=True)            # 换手的历史分位(情绪温度)
    disp = ret.std(axis=1)                                         # 横截面离散度
    state = pd.DataFrame({
        's_mktmom': mkt_mom20, 's_mktvol': mkt_vol20,
        's_turnpct': mkt_turn_pct, 's_disp': disp,
    }, index=close.index)
    # 广播:每日状态乘到所有股票列
    for c in state.columns:
        feats[c] = pd.DataFrame(np.tile(state[c].values[:, None], (1, close.shape[1])),
                                index=close.index, columns=close.columns)
    # 交互:核心因子 × 情绪温度/大盘动量(模型可学的条件结构)
    for fk in ['G2_attention', 'G9_ret20', 'G10_idiomom', 'G4_vratio', 'G1_turn20', 'G6_chip_pnl']:
        if f'f_{fk}' in feats:
            for sc in ['s_turnpct', 's_mktmom']:
                feats[f'x_{fk}_{sc}'] = feats[f'f_{fk}'] * feats[sc]
    return feats

def make_dataset(feats, p):
    """长表:(date, code) → features + label(未来21日收益 rank)。只用 T+25 天后有数据的样本。"""
    close = p['close']
    fwd = close.shift(-FWD) / close - 1
    label = fwd.rank(axis=1, pct=True)          # 横截面 rank 0~1
    X_cols = list(feats)
    dates = close.index
    idx, rows = [], []
    # 每季度采样一天(训练数据量控制)× 全截面 → 8 年 ≈ 32 日 × ~3000 股 ≈ 10 万行
    sample_days = [d for i, d in enumerate(dates) if i % 60 == 0]
    for d in sample_days:
        if d not in label.index: continue
        y = label.loc[d]
        X = [feats[c].loc[d] for c in X_cols]
        df = pd.concat([y] + X, axis=1, keys=['y'] + X_cols).dropna()
        ok = engine.eligible_universe(p, d)
        df = df.loc[df.index.intersection(ok)]
        if len(df) > 80:
            rows.append(df)
            idx.append(d)
    data = pd.concat(rows)
    data['date'] = np.concatenate([[d] * len(r) for d, r in zip(idx, rows)])
    return data, X_cols

def walk_forward_predict(data, X_cols, close_index, min_train=30000):
    """逐年 expanding:用 <Y 的数据(剔除与前 25 日重叠)训练,预测 Y 年全部采样日。"""
    data = data.sort_index()
    data['date'] = pd.to_datetime(data['date'])
    years = sorted(set(data['date'].dt.year))
    preds = []
    importances = {}
    first_train_year = years[0] + 2   # 至少 2 年数据才开始
    for Y in years:
        if Y < first_train_year:
            continue
        train = data[(data['date'].dt.year < Y) &
                     (data['date'] < pd.Timestamp(f'{Y}-01-01') - pd.Timedelta(days=25))]
        test = data[data['date'].dt.year == Y]
        if len(train) < min_train or len(test) == 0:
            continue
        model = lgb.LGBMRegressor(**PARAMS)
        model.fit(train[X_cols], train['y'])
        p_test = model.predict(test[X_cols])
        test = test.assign(pred=p_test)
        preds.append(test[['date', 'y', 'pred']])
        importances[Y] = dict(zip(X_cols, model.feature_importances_.tolist()))
        print(f"  year {Y}: train={len(train)} test={len(test)}", flush=True)
    return pd.concat(preds), importances

def preds_to_score(pred_df, close_columns):
    """预测长表 → 日期×代码 score 矩阵(按日 z-score)。"""
    g = pred_df.groupby('date')['pred']
    pred_df = pred_df.assign(z=(pred_df['pred'] - g.transform('mean')) / (g.transform('std') + 1e-12))
    mat = pred_df.pivot_table(index='date', columns=pred_df.index, values='z')
    mat.index = pd.to_datetime(mat.index)
    return mat

def main():
    t0 = time.time()
    print('== ML P3: build panel & features ==', flush=True)
    p = panel.build_panel()
    bench = pd.read_parquet(f"{BASE}/data/index_000300.parquet")
    bench['close'] = pd.to_numeric(bench['close'], errors='coerce')
    bench_s = bench.set_index(pd.to_datetime(bench['date']))['close']
    F = research.direction_fix(factors.compute_factors(p))
    health = research.ichealth(p, F)
    F = research.empirical_direction(F, health)   # 与 P1 一致:按实证 IC 统一方向
    Fz = research.zscore(F)
    feats = build_features(p, Fz)
    print(f"features: {len(feats)} cols · {time.time()-t0:.0f}s", flush=True)

    print('== build dataset ==', flush=True)
    data, X_cols = make_dataset(feats, p)
    print(f"dataset: {len(data)} rows · {time.time()-t0:.0f}s", flush=True)

    print('== walk-forward ==', flush=True)
    pred, imp = walk_forward_predict(data, X_cols, p['close'].index)
    score = preds_to_score(pred, p['close'].columns)
    print(f"predictions: {score.shape} · {time.time()-t0:.0f}s", flush=True)

    # 回测(与 P1 完全同引擎同约束)
    r3 = engine.run_strategy(p, score, top_n=50, freq='M', start='2019-01-01', name='P3')
    st3 = engine.perf_stats(r3['nav'], bench=bench_s)
    print('P3:', {k: round(v, 4) if isinstance(v, float) else v for k, v in st3.items() if not isinstance(v, pd.Series)}, flush=True)
    ic3 = engine.monthly_rank_ic(score, p, FWD, '2019-01-01')

    names = json.load(open(f"{BASE}/data/meta/name_map.json"))
    last_rb = sorted(r3['picks_hist'])[-1]
    picks = r3['picks_hist'][last_rb]
    holdings = [{'code': c, 'name': names.get(c, ''), 'score': float(score.loc[last_rb, c]), 'weight': 1/len(picks)}
                for c in picks if c in score.columns and last_rb in score.index]
    holdings = sorted(holdings, key=lambda x: -x['score'])

    # 特征重要性聚合(gain 归一,按年平均)
    imp_df = pd.DataFrame(imp).fillna(0)
    imp_mean = (imp_df / imp_df.sum()).mean(axis=1).sort_values(ascending=False)
    top_imp = [{'feature': k, 'gain': round(float(v), 4)} for k, v in imp_mean.head(15).items()]

    report = {
        'meta': {'key': 'p3', 'name': 'P3 机器学习(LightGBM)', 'generated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
                 'universe': '同P1 · walk-forward 逐年扩窗训练', 'rebalance': '月度', 'top_n': 50,
                 'period_start': str(r3['nav'].index[0].date()), 'period_end': str(r3['nav'].index[-1].date()),
                 'cost_model': '双边0.2%', 'fill_model': 'T+1开盘·一字板不成交·跌停顺延',
                 'holdings_date': str(last_rb.date())},
        'metrics': {**{k: v for k, v in st3.items() if not isinstance(v, pd.Series)},
                    'avg_turnover': float(r3['turnover'].mean()), 'n_trades': r3['n_trades'],
                    'blocked_buys': r3['blocked_buys']},
        'nav': export.nav_payload(r3['nav'], bench_s),
        'annual': export.annual_payload(r3['nav'], bench_s),
        'ic': export.ic_payload(ic3),
        'holdings_latest': holdings,
        'notes': [
            'LightGBM 截面模型:12因子 + 市场状态 + 因子×状态交互,滚动训练(embargo 25日),超参保守固定未调参。',
            '预测期从第一个训练年起(至少2年数据),全部为真样本外;与P1线性同引擎对比。',
            '树模型学习因子交互与状态依赖,是「多视角叠加」的机器学习实现;特征重要性见报告。',
        ],
    }
    export.export_one(report)

    dump = {'p3_stats': {k: v for k, v in st3.items() if not isinstance(v, pd.Series)},
            'p3_importance': top_imp,
            'p3_by_year_ic': {str(y): float(ic3[ic3.index.year == y].mean()) for y in sorted(set(ic3.index.year))}}
    fp = f"{BASE}/reports/ml_dump.json"
    existing = {}
    if os.path.exists(fp):
        existing = json.load(open(fp))
    existing.update(dump)
    json.dump(existing, open(fp, 'w'), ensure_ascii=False, indent=1)
    print(f"ML DONE · {time.time()-t0:.0f}s", flush=True)

if __name__ == '__main__':
    main()
