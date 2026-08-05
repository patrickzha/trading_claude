"""
完整验证流程：5年数据 + Granger检验 + 熔断规则贡献分解 + 最终综合回测报告
================================================================
一次性拉取5年数据，避免 Granger/熔断A/B/最终回测三个环节各自重复下载。
新功能（HMM regime、排序目标、conformal、meta-label）全部保持默认关闭，
这份报告代表的是"现有默认策略在5年数据+新增风控（财报避让+账户熔断）
下"的真实水平，不是新功能加持后的乐观版本。
"""
from __future__ import annotations

import time
import pandas as pd

from backtest import (
    fetch_market_data, fetch_earnings_calendar, run_walk_forward_backtest,
    get_weekly_trading_dates,
)
from optimized_metrics_v4 import run_backtest_with_metrics
from stat_validation import compute_period_metrics
from us_stock_screener import SECTOR_MAP


def main():
    t0 = time.time()
    print("#" * 70)
    print("# 第1步：拉取5年历史数据")
    print("#" * 70)
    c_df, o_df, h_df, l_df, v_df, vix, spy_close = fetch_market_data(days=1825)
    common_idx = c_df.index
    print(f"数据拉取耗时: {time.time()-t0:.1f}s\n")

    print("#" * 70)
    print("# 第2步：Granger 因果检验（VIX 是否领先于股票池未来收益）")
    print("#" * 70)
    from statsmodels.tsa.stattools import grangercausalitytests, adfuller
    market_ret = c_df.mean(axis=1).pct_change().dropna()
    vix_chg = vix.pct_change().dropna()
    idx = market_ret.index.intersection(vix_chg.index)
    data = pd.DataFrame({"market_ret": market_ret.loc[idx], "vix_chg": vix_chg.loc[idx]}).dropna()
    try:
        adf_p = adfuller(data["market_ret"])[1]
        print(f"  ADF检验 股票池日收益率: p={adf_p:.4f}")
        results = grangercausalitytests(data[["market_ret", "vix_chg"]], maxlag=10, verbose=False)
        sig_lags = [lag for lag in range(1, 11) if results[lag][0]["ssr_ftest"][1] < 0.05]
        for lag in range(1, 11):
            p = results[lag][0]["ssr_ftest"][1]
            print(f"  滞后{lag:2d}天: p={p:.4f}" + ("  <- 显著" if p < 0.05 else ""))
        print(f"  结论: {'检测到显著领先关系 ' + str(sig_lags) if sig_lags else '5年数据下依然没有检测到显著领先关系，VIX熔断规则缺乏统计支持'}")
    except Exception as e:
        print(f"  [错误] Granger检验失败: {type(e).__name__}: {e}")
    print()

    print("#" * 70)
    print("# 第3步：宏观熔断规则贡献分解（开 vs 关）")
    print("#" * 70)
    rf_weekly = (1 + 0.04) ** (1 / 52) - 1

    def _run(enable_macro_filter):
        label = "开启熔断" if enable_macro_filter else "关闭熔断"
        print(f"\n  --- {label} ---")
        res = run_walk_forward_backtest(c_df, o_df, h_df, l_df, v_df, vix,
                                        enable_shap=False, enable_macro_filter=enable_macro_filter)
        df = pd.DataFrame(res)
        df["weekly_return"] = df["weekly_return"].fillna(0.0)
        df["net_worth"] = df["net_worth"].ffill().fillna(100.0)
        m = compute_period_metrics(df, rf_weekly, 52)
        m["label"] = label
        return m

    with_filter = _run(True)
    without_filter = _run(False)
    print("\n  【熔断规则贡献分解】")
    for m in (with_filter, without_filter):
        print(f"  {m['label']:10s}  CAGR {m['cagr']*100:6.2f}%  Sharpe {m['sharpe']:5.2f}  "
              f"最大回撤 {m['max_dd']*100:6.2f}%  胜率 {m['win_rate']*100:5.1f}%  ({m['n_weeks']}周)")
    print(f"  熔断规则带来的增量: CAGR {(with_filter['cagr']-without_filter['cagr'])*100:+.2f}pp, "
          f"Sharpe {with_filter['sharpe']-without_filter['sharpe']:+.2f}")
    print()

    print("#" * 70)
    print("# 第4步：拉取财报日历（用于最终回测的财报避让）")
    print("#" * 70)
    earnings_calendar = fetch_earnings_calendar(list(SECTOR_MAP.keys()))
    print()

    print("#" * 70)
    print("# 第5步：最终综合回测（默认配置 + 财报避让 + 账户熔断，5年数据）")
    print("#" * 70)
    final_results = run_walk_forward_backtest(
        c_df, o_df, h_df, l_df, v_df, vix,
        enable_shap=True, earnings_calendar=earnings_calendar,
    )

    bench_rets = None
    if spy_close is not None:
        used_dates = {r["date"] for r in final_results}
        bench_rets = []
        for monday, friday in get_weekly_trading_dates(common_idx):
            if monday.strftime("%Y-%m-%d") not in used_dates:
                continue
            try:
                bench_rets.append((spy_close.loc[friday] - spy_close.loc[monday]) / spy_close.loc[monday])
            except Exception:
                bench_rets.append(0.0)

    run_backtest_with_metrics(final_results, benchmark_returns=bench_rets)

    total_elapsed = time.time() - t0
    print(f"\n全流程总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f} 分钟)")


if __name__ == "__main__":
    main()
