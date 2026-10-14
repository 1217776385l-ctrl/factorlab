# FactorLab · A股多因子研究与回测系统

FactorLab 是一套面向 A 股市场的**多因子研究系统**:从数据获取、因子计算、统计检验、参数寻优到组合回测与可视化,形成完整闭环。它不是交易机器人,而是一台「策略实验台」——任何选股想法都能在这套流水线上被严格检验。

**在线演示(回测终端)**:部署 GitHub Pages 后直接打开 `terminal/` 目录即可查看真实回测结果,支持多策略切换与任意起始日期切片。

## 核心特性

- **数据层**:5,000+ 只 A 股日线(后复权价+原始量/额/换手),baostock 免费数据源,增量更新
- **因子层**:12 个量价因子(换手率反转、短期反转、筹码盈亏代理、特质动量、量价相关、波动不对称等),全部通过 Rank IC / 分层回测 / 样本外检验
- **引擎层**:向量化分块回测,内置 A 股现实约束——T+1 开盘成交、一字板不追、跌停顺延卖出、双边 0.2% 成本、剔 ST/次新/低流动性
- **研究层**:参数网格搜索、逐因子剔除的简洁性检验、训练/验证窗分离、机器学习截面模型(LightGBM walk-forward)
- **终端**:零依赖 HTML 面板,ECharts 本地化,新策略按 schema 导出 JSON 即自动展示

## 架构流水线

```
fetch_bs2.py ──▶ data/kline/ ──▶ panel.py ──▶ panel.pkl
 (baostock)       逐股日线        (面板缓存)      │
                                                 ▼
                                          factors.py ──▶ fz_cache.pkl
                                          (G1-G12 因子)      │
                                                             ▼
                        research.py ──▶ strategies.py ──▶ engine.py
                        (IC/相关性/筛选)   (P1/P2/P3/P4)   (分块回测)
                                                             │
                              gen_report.py ◀── export.py ◀──┘
                              (markdown 报告)   (results/*.json)
                                                     │
                                                     ▼
                                            terminal/index.html
                                              (可视化回测终端)
```

## 三步复现

```bash
pip install -r requirements.txt

# 1. 拉数据(全 A 日线,约 1.5-2 小时;baostock 免费,无需 token)
python3 quant/fetch_bs2.py

# 2. 构建面板与因子缓存(单核约 2 分钟)
python3 quant/panel.py && python3 quant/compute_cache.py

# 3. 跑一次单因子回测(引擎自检)
python3 -c "
import pickle, panel, research, engine
p = panel.build_panel()
c = pickle.load(open('data/fz_cache.pkl','rb'))
s = research.combine(c['Fz'], ['G1_turn20','G4_vratio','G3_amtshare','G10_idiomom'])
r = engine.run_strategy(p, s, top_n=5, freq='M', start='2020-01-01')
print(engine.perf_stats(r['nav']))
"
```

数据目录可通过环境变量重定向:`FACTORLAB_DATA=/path/to/data`(kline 只读引用),输出位置由 `FACTORLAB_ROOT`(仓库根)决定。

## 策略结果快照(2020-01 至 2026-09,全 A 剔 ST/次新/低流动性,双边 0.2% 成本)

| 策略 | 结构 | 年化 | 最大回撤 | Sharpe | 说明 |
|---|---|---|---|---|---|
| **P1 精选 Top5·月频** | 换手率反转+量比+份额迁移+特质动量,等权 Top5 | **+25.9%** | -39% | 0.77 | 进攻原型;收益高度依赖 2021 年风格 |
| **P4 攻防融合**(主推) | 大盘高波动时半仓切换低波股 | +19.8% | **-32%** | 0.79 | 2020 熊市年 -15%→+1%,深回撤天数减半 |
| 沪深300(基准) | — | +1.3% | — | — | — |

策略机制详解见 [reports/策略报告.md](reports/策略报告.md)(含每套策略"赚什么钱、为什么"的通俗说明)。

## 已知局限(必读)

1. **幸存者偏差**:数据源不含完整退市股,回测绝对收益偏乐观;结论以因子间相对比较为准;
2. **过拟合风险**:参数网格在全窗口选优,尽管有 2024-2026 验证窗与逐因子剔除检验,实盘预期仍应打折;
3. **风格依赖**:收益与 2024-2026 小盘风格高度相关,风格反转时回撤控制只能缓解不能免疫;
4. **非线性因子失效模式**:强 IC 因子未必能承受极端集中持仓(案例:隔夜反转因子 ICIR 0.56 但 Top5 组合年化 -44%);
5. 本项目仅供研究学习,不构成投资建议。

## 目录结构

```
├── quant/        # 全部代码(数据/面板/因子/引擎/研究/策略/导出/报告/更新)
├── data/         # 数据目录(gitignore,由 fetch_bs2.py 生成)
├── db/           # 数据库说明(含数据源证券池过滤的教训)
├── terminal/     # 可视化回测终端(GitHub Pages 演示)
├── reports/      # 策略报告 + 因子体检报告 + 回测产物 JSON
├── results/      # 终端数据源(schema 见 terminal/README-schema.md)
└── docs/         # 策略设计文档
```

## License

MIT(代码)。图表库 [ECharts](https://echarts.apache.org/) 为 Apache-2.0 许可(已随仓库分发于 `terminal/echarts.min.js`)。
