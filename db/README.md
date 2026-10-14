# FactorLab DB · 数据库说明

> 定位:**所有策略共用的数据底座**。策略会迭代、会推倒重来,数据层一次建成、持续更新、不再重拉。

## 目录结构

```
Desktop/因子/
├── data/
│   ├── kline/          # 逐股日线(后复权价+原始量额换手), parquet, 2015-01-05 起
│   ├── panel.pkl       # 面板缓存:date×code 矩阵 × 7字段 + isST/suspended, float32
│   ├── fz_cache.pkl    # 因子缓存:12个量价因子全期矩阵(F/Fz/实证IC)
│   └── meta/name_map.json  # 代码→名称
├── quant/              # 代码
│   ├── update_daily.py # 每日收盘后增量更新(几分钟)
│   └── ...
└── terminal/           # 可视化终端(schema 见 README-schema.md)
```

## 证券池口径(重要,血泪教训)

**只保留 A 股股票**:sh.60xxxx(沪主板)、sh.688xxx(科创板)、sz.00xxxx(深主板)、sz.30xxxx(创业板)。
**排除**:sz.399xxx / sh.000xxx(指数!)、基金、债券。baostock 的 `query_all_stock` 会把指数混进来,历史上曾污染过整条回测(254 只指数顶替股票)。新增代码时必须过这个正则。

## 快速使用(新策略从这里开始)

```python
import panel, factors, research, engine

p = panel.build_panel()                      # ~90s 首次加载缓存
c = pickle.load(open('data/fz_cache.pkl','rb'))
Fz = c['Fz']                                 # 12 因子矩阵,方向已按实证 IC 对齐

score = 你的选股逻辑(Fz, p)                  # 输出 date×code 分数矩阵
r = engine.run_strategy(p, score, top_n=5, freq='M', start='2020-01-01')
print(engine.perf_stats(r['nav']))
```

引擎内置 A 股约束:T+1 开盘成交、一字板不追、跌停顺延(≤15日)、双边 0.2% 成本、剔 ST/次新/低流动性。

## 更新节奏

- **每日**(收盘后):`python3 quant/update_daily.py` — 追加当日数据+重建缓存(建议挂 cron)
- **手动**:`python3 quant/panel.py` 可随时 force 重建面板

## 数据已知局限(诚实清单)

1. 退市股不完整 → 幸存者偏差,回测绝对收益偏乐观,相对比较可信;
2. 2015 年前数据未拉(需要更长历史时再补);
3. baostock 服务器限速:全量重拉 ≈ 2h,故一切以增量更新为主。
