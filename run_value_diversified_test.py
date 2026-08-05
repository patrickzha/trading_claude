"""
价值策略防御性改造：降低集中度(20-30只 -> 50/100只) + 加行业分散约束
(单行业最多N只)，测试能不能在保留部分收益优势的同时把"验证结论"第38条
发现的深度真实回撤(-20%~-43%)压下来。两个独立历史区间都测。
"""
from __future__ import annotations

import pickle

from value_investing import backtest_value_strategy, metrics_from_periods
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

PIT_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/point_in_time_universe_cache.pkl"
HIST_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/historical_2014_2020_cache.pkl"

CONFIGS = {
    "top30_baseline(对照)": {"top_n": 30, "max_per_sector": None},
    "top50_sector5": {"top_n": 50, "max_per_sector": 5},
    "top100_sector8": {"top_n": 100, "max_per_sector": 8},
}


def run_on(price_df, membership, spy_close, sector_map, label):
    print(f"\n{'='*20} {label} {'='*20}")
    for name, kwargs in CONFIGS.items():
        results, daily_nav = backtest_value_strategy(
            price_df, membership, spy_close, rebalance_days=126, sector_map=sector_map, **kwargs)
        m = metrics_from_periods(results, daily_nav=daily_nav, periods_per_year=2.0)
        if m["CAGR"] is None:
            print(f"  {name}: 周期数不足")
            continue
        print(f"  {name}: n_periods={m['n_periods']}, CAGR={m['CAGR']*100:.2f}%, "
              f"Sharpe={m['Sharpe']:.3f}, MaxDD(逐日真实)={m['TrueDailyMaxDD']*100:.2f}%, "
              f"SPY_CAGR={m['SPY_CAGR']*100 if m['SPY_CAGR'] else float('nan'):.2f}%")


def main():
    with open(PIT_CACHE, "rb") as f:
        cache = pickle.load(f)
    price_df = cache["c"]
    spy_close = cache.get("spy")
    sector_map = cache["sector_map"]
    membership = compute_point_in_time_membership(price_df.index, list(price_df.columns),
                                                    cache["added_dates"], cache["removal_dates"])
    run_on(price_df, membership, spy_close, sector_map, "点位时间正确554只(2022-2026)")

    with open(HIST_CACHE, "rb") as f:
        hist = pickle.load(f)
    hist_price_df = hist["c"]
    hist_spy = hist.get("spy")
    all_tickers = list(hist_price_df.columns)
    fake_membership = {hist_price_df.index[i].strftime("%Y-%m-%d"): all_tickers for i in range(len(hist_price_df))}
    run_on(hist_price_df, fake_membership, hist_spy, sector_map, "独立历史区间(2014-2020)")


if __name__ == "__main__":
    main()
