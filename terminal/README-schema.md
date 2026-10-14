# 回测终端 · 新策略接入指南

终端是**通用可视化面板**,不是一次性的:任何策略只要按下面的 schema 导出 JSON,就能在终端里展示。

## 快速接入(三步)

1. 你的策略回测完后,构造一个符合 schema 的 dict(见下);
2. 调 `quant/export.py` 的 `export_one(report)` → 自动写入 `results/{key}.json` 并重建 `terminal/data.js`;
3. 打开 `terminal/index.html`(`python3 -m http.server` 或直接双击)→ 顶部出现新策略标签。

## Schema

```python
report = {
  "meta": {
    "key":            "p1",                      # 唯一英文 id(文件名)
    "name":           "P1 量价共振",              # 终端标签显示名
    "generated":      "2026-09-24 04:00",
    "universe":       "全A剔ST/次新/低流动性",
    "rebalance":      "月度" | "双周",
    "top_n":          50,
    "period_start":   "2016-07-01",
    "period_end":     "2026-09-23",
    "cost_model":     "双边0.2%",
    "fill_model":     "T+1开盘·一字板不成交·跌停顺延",
    "holdings_date":  "2026-09-01",              # 持仓快照对应调仓日
  },
  "metrics": {           # 数值全部为小数(比例)
    "annual_return", "bench_annual", "excess_annual", "sharpe", "max_drawdown",
    "calmar", "info_ratio", "annual_vol", "win_rate_period",
    "avg_turnover", "n_trades", "blocked_buys", "total_return", "years",
  },
  "nav": {               # 净值(起点=1)
    "dates":     ["2016-07-01", ...],
    "strategy":  [1.0, ...],
    "benchmark": [1.0, ...],   # 可选;有则画基准线+相对净值(独立轴)
  },
  "annual": [            # 年度收益,可选
    {"year": 2016, "strategy": 0.12, "benchmark": 0.05, "excess": 0.07}, ...
  ],
  "ic": {                # 月度 RankIC,可选;缺省隐藏 IC 面板
    "dates": [...], "values": [...], "mean": 0.03, "icir": 0.4, "t": 2.5, "n": 100
  },
  "layers": {            # 因子分层净值,可选
    "dates": [...], "names": ["Q1".."Q5"], "series": [[...], ...]   # 起点=1
  },
  "holdings_latest": [   # 最新持仓,可选
    {"code": "sh.600519", "name": "贵州茅台", "score": 2.31, "weight": 0.02}, ...
  ],
  "notes": [ "诚实声明/注意事项, 每条一个列表项" ],
}
```

## 复用已有函数(quant/export.py)

- `nav_payload(nav_series, bench_series)` — Series → dates/strategy/benchmark
- `annual_payload(nav, bench)` — 年度收益+超额
- `ic_payload(ic_series)` — RankIC 序列+统计
- `layers_payload({1: nav, ..., 5: nav})` — 分层净值

## 设计约束

- 颜色:**红涨绿跌**(A 股习惯),策略线金色,基准灰,超额青蓝虚线(独立 y 轴)。
- 双击 `index.html` 即可用(file:// 协议):数据经 `data.js` 注入,不走 fetch,无 CORS 问题。
- `results/*.json` 是存档真相;`data.js` 永远可由 `python3 quant/export.py` 重建。

---
*Made by LYU Minglei*
