"""从 results/*.json + reports/pipeline_dump.json 生成 markdown 报告:
reports/因子体检报告.md + reports/策略报告.md
样本内外拆分直接由 nav 序列重算,不依赖回测进程。
"""
import json, glob, os
import numpy as np
import pandas as pd

BASE = os.environ.get("FACTORLAB_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IS_END = pd.Timestamp('2021-12-31')
OOS_START = pd.Timestamp('2022-01-01')

def perf_from_nav(nav: pd.Series, bench: pd.Series = None):
    nav = nav.sort_index()
    ret = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    ann = nav.iloc[-1] ** (1 / max(years, 1e-9)) - 1
    vol = ret.std() * np.sqrt(252)
    mdd = (nav / nav.cummax() - 1).min()
    out = {'annual': ann, 'sharpe': ann / vol if vol > 0 else 0, 'mdd': mdd, 'n': len(nav)}
    if bench is not None:
        b = bench.reindex(nav.index).ffill().dropna()
        if len(b) > 10:
            b = b / b.iloc[0]
            b_ann = b.iloc[-1] ** (1 / max(years, 1e-9)) - 1
            out['bench_ann'] = b_ann
            out['excess'] = ann - b_ann
    return out

def pct(x, d=1):
    return '—' if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:.{d}f}%"

def gen_health_report(dump):
    h = sorted(dump['health'], key=lambda r: -abs(r['icir']))
    lines = ["# 因子体检报告", "", f"> 生成:{pd.Timestamp.now():%Y-%m-%d %H:%M} · 月频 RankIC · 前瞻21个交易日",
             "", "方向已统一:「IC为正」表示因子值大→未来收益高。t>2 视为显著。",
             "", "| 因子 | 含义 | IC均值 | ICIR | t值 | 样本内IC(16-21) | 样本外IC(22-26) | 衰减 | 正年份占比 |",
             "|---|---|---|---|---|---|---|---|---|"]
    desc = {
        'G1_turn20': '换手率(过热)', 'G2_attention': '关注度跃升', 'G3_amtshare': '量能份额迁移',
        'G4_vratio': '放量倍数', 'G5_pvcorr': '量价相关', 'G6_chip_pnl': '筹码盈亏',
        'G7_ampc': '振幅收缩', 'G8_upvol': '上涨力道/下跌力道', 'G9_ret20': '短期反转',
        'G10_idiomom': '特质动量', 'G11_upratio': '动量质量', 'G12_ddpos': '回撤位置',
    }
    for r in h:
        dec = (r['oos_ic'] - r['is_ic']) / (abs(r['is_ic']) + 1e-9)
        lines.append(f"| {r['factor']} | {desc.get(r['factor'],'')} | {r['ic_mean']:+.3f} | {r['icir']:+.2f} | {r['t_stat']:+.1f} | {r['is_ic']:+.3f} | {r['oos_ic']:+.3f} | {dec:+.0%} | {r['pos_years']:.0%} |")
    chosen = dump.get('chosen', [])
    lines += ["", "## 入选组合(相关性筛选后)", "", ", ".join(f"`{c}`" for c in chosen), ""]
    C = dump.get('corr', {})
    if C:
        names = list(C.keys())
        lines += ["## 横截面相关性(日频均值)", "", "| " + " | ".join(names) + " |",
                  "|" + "---|" * len(names)]
        for a in names:
            lines.append("| " + " | ".join(f"{C[a][b]:.2f}" if isinstance(C[a].get(b), (int, float)) else "—" for b in names) + " |")
    return "\n".join(lines) + "\n"

def gen_strategy_report(dump):
    files = sorted(glob.glob(f"{BASE}/results/*.json"))
    name_map = json.load(open(f"{BASE}/data/meta/name_map.json"))
    out = ["# 策略报告", "", f"> 生成:{pd.Timestamp.now():%Y-%m-%d %H:%M} · 基准:沪深300 · 成本:双边0.2% · T+1开盘成交", ""]
    for fp in files:
        r = json.load(open(fp))
        if r['meta']['key'] == 'mock':
            continue
        dates = pd.to_datetime(r['nav']['dates'])
        nav = pd.Series(r['nav']['strategy'], index=dates)
        bench = pd.Series(r['nav'].get('benchmark', []), index=dates) if r['nav'].get('benchmark') else None
        full = perf_from_nav(nav, bench)
        is_p = perf_from_nav(nav[nav.index <= IS_END], bench)
        oos = perf_from_nav(nav[nav.index >= OOS_START], bench)
        m = r['metrics']
        out += [f"## {r['meta']['name']}", "",
                f"- **区间**:{r['meta']['period_start']} → {r['meta']['period_end']}({m.get('years','?')}年)· {r['meta']['rebalance']}调仓 Top{r['meta']['top_n']}",
                f"- **全样本**:年化 {pct(full['annual'])} / Sharpe {full['sharpe']:.2f} / 最大回撤 {pct(full['mdd'])}"
                + (f" / 超额 {pct(full.get('excess'))}" if 'excess' in full else ""),
                f"- **样本内(16-21)**:年化 {pct(is_p['annual'])}" + (f" / 超额 {pct(is_p.get('excess'))}" if 'excess' in is_p else "") + f"({is_p['n']}日)",
                f"- **样本外(22-26)**:年化 {pct(oos['annual'])}" + (f" / 超额 {pct(oos.get('excess'))}" if 'excess' in oos else "") + f"({oos['n']}日)",
                f"- **交易**:成交 {m.get('n_trades','—')} 笔 · 一个月均换手 {pct(m.get('avg_turnover'))} · 一字板放弃买入 {m.get('blocked_buys','—')} 次",
                ""]
        # 年度表
        if r.get('annual'):
            out += ["| 年份 | 策略 | 基准 | 超额 |", "|---|---|---|---|"]
            for a in r['annual']:
                out.append(f"| {a['year']} | {pct(a.get('strategy'))} | {pct(a.get('benchmark'))} | {pct(a.get('excess'))} |")
            out.append("")
        # 持仓
        if r.get('holdings_latest'):
            out += [f"### 最新持仓(调仓日 {r['meta'].get('holdings_date','')},前20)", "",
                    "| # | 代码 | 名称 | 综合分 | 权重 |", "|---|---|---|---|---|"]
            for i, x in enumerate(r['holdings_latest'][:20], 1):
                out.append(f"| {i} | {x['code']} | {x.get('name') or name_map.get(x['code'],'')} | {x['score']:.3f} | {x['weight']:.2%} |")
            out.append("")
        for n in r.get('notes', []):
            out.append(f"> {n}")
        out.append("")
    # 敏感性与简洁性
    if dump.get('sensitivity'):
        out += ["## P1 参数敏感性(TopN × 调仓频率)", "", "| TopN | 频率 | 年化 | 超额 | Sharpe | 最大回撤 |", "|---|---|---|---|---|---|"]
        for s in dump['sensitivity']:
            out.append(f"| {s['top_n']} | {s['freq']} | {pct(s['annual'])} | {pct(s['excess'])} | {s['sharpe']:.2f} | {pct(s['mdd'])} |")
        out.append("")
    sim = dump.get('simplicity', {})
    if sim.get('leave_one_out'):
        out += ["## 简洁性检验:逐因子剔除(样本外是否反而更好)", "",
                f"基准组合(全因子)超额:{pct(sim.get('base_excess'))}", "",
                "| 剔除因子 | 剩余因子 | 超额 | Sharpe | Δ超额 |", "|---|---|---|---|---|"]
        for s in sim['leave_one_out']:
            d = (s['excess'] - sim['base_excess']) if (sim.get('base_excess') is not None and not np.isnan(s['excess'])) else None
            out.append(f"| {s['dropped']} | {len(s['kept'])}个 | {pct(s['excess'])} | {s['sharpe']:.2f} | {pct(d) if d is not None else '—'} |")
        out += ["", "> Δ超额>0 = 剔掉后更好 → 该因子是冗余/拖累,应从组合中移除。", ""]
    # ML 章节
    ml_fp = f"{BASE}/reports/ml_dump.json"
    if os.path.exists(ml_fp):
        ml = json.load(open(ml_fp))
        if ml.get('p3_stats'):
            s = ml['p3_stats']
            out += ["## P3 · LightGBM 机器学习截面模型", "",
                    f"- 全样本:年化 {pct(s.get('annual_return'))} / Sharpe {s.get('sharpe', 0):.2f} / 最大回撤 {pct(s.get('max_drawdown'))}"
                    + (f" / 超额 {pct(s.get('excess_annual'))}" if s.get('excess_annual') is not None else ""),
                    "- 设置:12因子 + 市场状态 + 因子×状态交互 → 预测未来21日收益 rank;walk-forward 逐年扩窗,embargo 25日,超参固定不调参。",
                    "- 与 P1 同引擎同约束对比。", ""]
        if ml.get('p3_importance'):
            out += ["### 特征重要性(gain 占比 Top15)", "", "| 特征 | 重要性 |", "|---|---|"]
            for x in ml['p3_importance']:
                out.append(f"| `{x['feature']}` | {x['gain']:.1%} |")
            out += ["", "> 交互项(`x_`前缀)排名靠前 = 模型确实在利用「市场状态 × 因子」的条件结构。", ""]
        if ml.get('p3_by_year_ic'):
            out += ["### 分年预测 IC(全部为真样本外)", "", "| 年份 | 平均 RankIC |", "|---|---|"]
            for y, v in ml['p3_by_year_ic'].items():
                out.append(f"| {y} | {v:+.3f} |")
            out.append("")
    # 网格搜索章节
    grid_fp = f"{BASE}/reports/grid_top.json"
    if os.path.exists(grid_fp):
        g = json.load(open(grid_fp))
        if g.get('top15'):
            out += [f"## P1v2 网格搜索 Top15({g['n']} 组合:因子子集×TopN{{3,5}}×{{周,双周,月}}×{{等权,ICIR加权}})",
                    "", "回测窗口 2020 至今;「验证窗」= 2024-2026(未参与选参的时间段,检验稳健性)。", "",
                    "| 年化 | 超额 | Sharpe | 回撤 | N | 频率 | 加权 | 验证窗年化 | 因子组合 |", "|---|---|---|---|---|---|---|---|---|"]
            for s in g['top15']:
                fs = '+'.join(f.split('_', 1)[1] for f in s['factors'])
                oos = s.get('oos_annual')
                out.append(f"| {s['annual']:+.1%} | {s['excess']:+.1%} | {s['sharpe']:.2f} | {s['mdd']:.0%} | {s['top_n']} | {s['freq']} | {s['weight']} | {f'{oos:+.1%}' if oos is not None else '—'} | {fs} |")
            out += ["", "> 全窗口选优存在过拟合风险:优先关注「验证窗年化」与全窗一致的组合。", ""]
    return "\n".join(out) + "\n"

if __name__ == '__main__':
    dump = json.load(open(f"{BASE}/reports/pipeline_dump.json"))
    open(f"{BASE}/reports/因子体检报告.md", 'w').write(gen_health_report(dump))
    open(f"{BASE}/reports/策略报告.md", 'w').write(gen_strategy_report(dump))
    print("reports generated")
