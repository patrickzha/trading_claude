"""
生存偏差修复后，用"旧默认参数"（止损5%、RSI阈值75、不加新近度权重，即本session
三个改进被采纳之前的原始默认值）在同一份点位时间正确的554只股票池上跑一遍，
跟"验证结论"第27条里"新默认参数"在同一股票池上的结果对比，隔离出：三个已采纳
的改进（止损2%/RSI阈值100/新近度加权）在修复后的、没有生存偏差加成的股票池上，
是否仍然比旧默认更好——而不是简单假设"验证结论"第6/11/14条的方向能直接搬过来。
"""
from __future__ import annotations

import pickle

from backtest import run_walk_forward_backtest, get_weekly_trading_dates
from optimized_metrics_v4 import run_backtest_with_metrics
from run_survivorship_bias_fix_validation import compute_point_in_time_membership, CACHE_PATH, EARNINGS_CACHE


def main():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    c_df, o_df, h_df, l_df, v_df, vix, spy_close = (
        cache["c"], cache["o"], cache["h"], cache["l"], cache["v"], cache["vix"], cache["spy"]
    )
    sector_map = cache["sector_map"]
    added_dates = cache["added_dates"]
    removal_dates_used = cache["removal_dates"]

    membership = compute_point_in_time_membership(c_df.index, list(c_df.columns), added_dates, removal_dates_used)

    with open(EARNINGS_CACHE, "rb") as f:
        earnings_calendar_full = pickle.load(f)

    import os
    n_workers = max(1, os.cpu_count() or 1)
    results = run_walk_forward_backtest(
        c_df, o_df, h_df, l_df, v_df, vix,
        enable_shap=False, n_workers=n_workers, xgb_n_jobs=1,
        earnings_calendar=earnings_calendar_full,
        sector_map=sector_map,
        point_in_time_membership=membership,
        stop_loss_pct=0.05,
        rsi_overbought_threshold=75.0,
        use_recency_weighting=False,
    )

    bench_rets = None
    if spy_close is not None:
        used_dates = {r["date"] for r in results}
        bench_rets = []
        for monday, friday in get_weekly_trading_dates(c_df.index):
            if monday.strftime("%Y-%m-%d") not in used_dates:
                continue
            try:
                bench_rets.append((spy_close.loc[friday] - spy_close.loc[monday]) / spy_close.loc[monday])
            except Exception:
                bench_rets.append(0.0)

    print("\n[点位时间正确的股票池 —— 旧默认参数(止损5%/RSI75/不加权)对照]")
    run_backtest_with_metrics(results, benchmark_returns=bench_rets)


if __name__ == "__main__":
    main()
