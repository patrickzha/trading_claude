"""
优化版绩效报告 v4
"""
import numpy as np
import pandas as pd

from stat_validation import print_statistical_validation


def run_backtest_with_metrics(backtest_results_list, benchmark_returns=None):
    if not backtest_results_list:
        print("错误：回测数据为空！")
        return

    df = pd.DataFrame(backtest_results_list)
    df["weekly_return"] = df["weekly_return"].fillna(0.0)
    df["net_worth"] = df["net_worth"].ffill().fillna(100.0)

    weeks_per_year = 52
    rf_annual = 0.04
    rf_weekly = (1 + rf_annual) ** (1 / weeks_per_year) - 1

    total_weeks = len(df)
    years = total_weeks / weeks_per_year
    final_nw = df["net_worth"].iloc[-1]
    initial_nw = df["net_worth"].iloc[0]
    cagr = (final_nw / initial_nw) ** (1 / years) - 1 if years > 0 else 0.0

    ann_vol = df["weekly_return"].std() * np.sqrt(weeks_per_year)
    excess = df["weekly_return"] - rf_weekly
    sharpe = (excess.mean() / excess.std()) * np.sqrt(weeks_per_year) if excess.std() != 0 else 0.0

    downside = excess[excess < 0]
    sortino = (excess.mean() / downside.std()) * np.sqrt(weeks_per_year) if len(downside) > 0 and downside.std() != 0 else 0.0

    rolling_max = df["net_worth"].cummax()
    dd = (df["net_worth"] - rolling_max) / rolling_max
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    wins = df["weekly_return"][df["weekly_return"] > 0]
    losses = df["weekly_return"][df["weekly_return"] < 0]
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0
    pl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    cash_weeks = sum(df["picks"].apply(lambda x: len(x) == 0))

    print("=" * 70)
    print(" 【优化版 v4 AI 周频策略回测报告】")
    print("=" * 70)
    print(f" 回测区间: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
    print(f" 总交易周数: {total_weeks} 周（空仓 {cash_weeks} 周，占比 {cash_weeks/total_weeks:.1%}）")
    print(f" 周胜率: {(df['weekly_return'] > 0).mean() * 100:.1f}%")
    print(f" 盈亏比: {pl_ratio:.2f}")
    print(f" 累计收益率: {((final_nw / initial_nw) - 1) * 100:.1f}%")
    print(f" 年化复合收益 (CAGR): {cagr * 100:.2f}%")
    print("-" * 60)
    print(f" 年化波动率: {ann_vol * 100:.2f}%")
    print(f" 最大回撤: {max_dd * 100:.2f}%")
    print(f" 夏普比率: {sharpe:.2f}")
    print(f" 索提诺比率: {sortino:.2f}")
    print(f" Calmar 比率: {calmar:.2f}")
    print("=" * 70)

    if benchmark_returns is not None:
        bench = pd.Series(benchmark_returns).reset_index(drop=True)
        if len(bench) == total_weeks:
            bench_nw = 100 * (1 + bench).cumprod()
            bench_cagr = (bench_nw.iloc[-1] / 100) ** (1 / years) - 1 if years > 0 else 0.0
            bench_dd = (bench_nw - bench_nw.cummax()) / bench_nw.cummax()
            print(" 【基准对比 (SPY 买入持有)】")
            print(f" 基准 CAGR: {bench_cagr * 100:.2f}%")
            print(f" 基准最大回撤: {bench_dd.min() * 100:.2f}%")
            print(f" 超额年化收益: {(cagr - bench_cagr) * 100:+.2f}%")
            print("=" * 70)

    print(" 成本假设: 双边 0.30%（手续费+滑点）")
    print(" 空仓收益: 年化 4%（货币基金）")
    print(" 成交规则: 周一 Open 买入，周五 Close 卖出")
    print(" 风控规则: 周内跌幅>2%触发止损（默认值，实际以调用时传入的 stop_loss_pct 为准），行业集中度≤2只")
    print(" 模型目标: 回归未来5日收益率（Huber Loss）")
    print(" 特征维度: 21维（12时序+4截面排名+4微观结构+1Regime）")
    print(" 截面排名: 逐日分组独立计算，消除跨日尺度偏差")
    print(" SHAP分析: 每20周输出特征重要性图")
    print("=" * 70)

    print_statistical_validation(df, rf_weekly, weeks_per_year)