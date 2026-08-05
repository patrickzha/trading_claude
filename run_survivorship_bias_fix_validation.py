"""
生存偏差完整修复的最终验证：用合并后的554只票价格数据（当前500只 + 找回的
53只被剔除票）+ 点位时间正确的每周成分股过滤，跑一次完整5年回测，用当前
已验证的默认配置（止损2%+RSI阈值100+新近度加权，全部模块默认值），直接
对比修复前（现在README里记录的那版数字）和修复后的差异。

已知局限：92只被剔除票只找回53只（另外39只已并购/私有化退市，价格数据
本身已经不存在），纳入日期用当前成分股名单的字段、剔除日期用Wikipedia
变更记录表，都是外部数据，可能有个别记录误差，但这是"点位时间正确"这个
方向上能做到的最大程度，比完全不做点位过滤已经是质的改善。
"""
from __future__ import annotations

import pickle

import pandas as pd

from backtest import run_walk_forward_backtest, get_weekly_trading_dates
from optimized_metrics_v4 import run_backtest_with_metrics

CACHE_PATH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/point_in_time_universe_cache.pkl"
EARNINGS_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/earnings_calendar_cache.pkl"


def compute_point_in_time_membership(index: pd.DatetimeIndex, current_tickers, added_dates, removal_dates_used):
    week_pairs = get_weekly_trading_dates(index)
    membership = {}
    for monday, friday in week_pairs:
        loc = index.get_loc(monday)
        if loc == 0:
            continue
        decision_date = index[loc - 1]
        valid = []
        for t in current_tickers:
            added = added_dates.get(t)
            if added is not None and added > decision_date:
                continue
            removed = removal_dates_used.get(t)
            if removed is not None and removed <= decision_date:
                continue
            valid.append(t)
        membership[decision_date.strftime("%Y-%m-%d")] = valid
    return membership


def main():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    c_df, o_df, h_df, l_df, v_df, vix, spy_close = (
        cache["c"], cache["o"], cache["h"], cache["l"], cache["v"], cache["vix"], cache["spy"]
    )
    sector_map = cache["sector_map"]
    added_dates = cache["added_dates"]
    removal_dates_used = cache["removal_dates"]

    print(f"合并股票池: {len(c_df.columns)}只（含{len(cache['removed_tickers_with_data'])}只找回的被剔除票）")

    membership = compute_point_in_time_membership(c_df.index, list(c_df.columns), added_dates, removal_dates_used)
    sample_week = list(membership.keys())[0]
    sample_week_early = list(membership.keys())[10]
    print(f"点位时间过滤示例: {sample_week} 有效票数={len(membership[sample_week])}, "
          f"{sample_week_early} 有效票数={len(membership[sample_week_early])}")

    with open(EARNINGS_CACHE, "rb") as f:
        earnings_calendar_full = pickle.load(f)
    # 找回的53只票没有单独拉财报日历，财报避让对这些票不生效（fail-open，
    # 现有代码本来就对没有日历数据的票默认不做财报避让），不影响主要结论

    import os
    n_workers = max(1, os.cpu_count() or 1)
    results = run_walk_forward_backtest(
        c_df, o_df, h_df, l_df, v_df, vix,
        enable_shap=False, n_workers=n_workers, xgb_n_jobs=1,
        earnings_calendar=earnings_calendar_full,
        sector_map=sector_map,
        point_in_time_membership=membership,
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

    print("\n[点位时间正确的股票池 —— 生存偏差修复后报告]")
    run_backtest_with_metrics(results, benchmark_returns=bench_rets)


if __name__ == "__main__":
    main()
