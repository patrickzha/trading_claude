"""
验证"验证结论"第45条的价值+动量组合分散化效应，在独立的2014-2020历史
区间上是不是也成立——这段区间跟点位时间正确的2022-2026区间完全不重叠，
包含2018Q4暴跌和2020年新冠暴跌，是判断第45条到底是真实规律还是单一
窗口巧合的关键测试。

跟"验证结论"第17条同样的已知局限：2014-2020这份缓存没有点位时间正确的
成分股过滤，用当前484只成分股名单套历史，方向可信、绝对数字可能偏乐观。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from backtest import run_walk_forward_backtest, get_weekly_trading_dates
from value_investing import backtest_value_strategy

HIST_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/historical_2014_2020_cache.pkl"
CACHE_PATH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/point_in_time_universe_cache.pkl"

WEIGHTS = [0.0, 0.3, 0.5, 0.7, 1.0]


def metrics(rets, periods_per_year=52.0):
    rets = np.array(rets)
    nav = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(nav)
    mdd = float(np.min((nav - peak) / peak))
    n_years = len(rets) / periods_per_year
    cagr = float(nav[-1] ** (1 / n_years) - 1)
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(periods_per_year))
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": mdd}


def main():
    with open(HIST_CACHE, "rb") as f:
        hist = pickle.load(f)
    c_df, o_df, h_df, l_df, v_df, vix = hist["c"], hist["o"], hist["h"], hist["l"], hist["v"], hist["vix"]
    spy_close = hist.get("spy")

    with open(CACHE_PATH, "rb") as f:
        pit_cache = pickle.load(f)
    sector_map = pit_cache["sector_map"]  # 用当前sector_map，跟"验证结论"第17条一致的简化

    import os
    n_workers = max(1, os.cpu_count() or 1)
    print("跑动量策略(2014-2020独立区间)...")
    mom_results = run_walk_forward_backtest(
        c_df, o_df, h_df, l_df, v_df, vix,
        enable_shap=False, n_workers=n_workers, xgb_n_jobs=1,
        enable_earnings_blackout=False,  # 这份缓存没有拉这段历史的财报日历
        sector_map=sector_map,
    )
    mom_df = pd.DataFrame(mom_results)[["date", "weekly_return"]]
    mom_df["date"] = pd.to_datetime(mom_df["date"])
    mom_df = mom_df.set_index("date")
    print(f"动量策略: {len(mom_df)}周")

    print("跑价值策略(2014-2020独立区间)...")
    all_tickers = list(c_df.columns)
    fake_membership = {c_df.index[i].strftime("%Y-%m-%d"): all_tickers for i in range(len(c_df))}
    _, daily_nav = backtest_value_strategy(c_df, fake_membership, spy_close, rebalance_days=126, top_n=30)
    nav_df = pd.DataFrame(daily_nav, columns=["date", "nav"]).set_index("date")

    week_pairs = get_weekly_trading_dates(c_df.index)
    val_rows = []
    for monday, friday in week_pairs:
        try:
            nav_at_monday = nav_df.loc[:monday, "nav"].iloc[-1]
            nav_at_friday = nav_df.loc[:friday, "nav"].iloc[-1]
        except IndexError:
            continue
        val_rows.append({"date": monday, "value_ret": nav_at_friday / nav_at_monday - 1})
    val_weekly_ret = pd.DataFrame(val_rows).set_index("date")["value_ret"]

    merged = mom_df.join(val_weekly_ret, how="inner")
    print(f"对齐后共 {len(merged)} 周")
    corr = merged["weekly_return"].corr(merged["value_ret"])
    print(f"动量策略 vs 价值策略 周收益率相关系数(2014-2020): {corr:.4f}")

    print(f"\n{'value_weight':>12} | {'CAGR':>8} | {'Sharpe':>8} | {'MaxDD':>8}")
    for w in WEIGHTS:
        blended = w * merged["value_ret"] + (1 - w) * merged["weekly_return"]
        m = metrics(blended.values)
        label = f"{w:.1f}(纯价值)" if w == 1.0 else (f"{w:.1f}(纯动量)" if w == 0.0 else f"{w:.1f}")
        print(f"{label:>12} | {m['CAGR']*100:7.2f}% | {m['Sharpe']:8.3f} | {m['MaxDD']*100:7.2f}%")


if __name__ == "__main__":
    main()
