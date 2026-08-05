"""
长周期基本面价值投资策略的首次验证。半年度调仓，跟同期SPY买入持有对比。
"""
from __future__ import annotations

import pickle

from value_investing import backtest_value_strategy, metrics_from_periods
from run_survivorship_bias_fix_validation import compute_point_in_time_membership, CACHE_PATH

CONFIGS = {
    "top30_semiannual": {"rebalance_days": 126, "top_n": 30},
    "top20_semiannual": {"rebalance_days": 126, "top_n": 20},
    "top30_quarterly": {"rebalance_days": 63, "top_n": 30},
}


def main():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    price_df = cache["c"]
    spy_close = cache.get("spy")
    added_dates = cache["added_dates"]
    removal_dates = cache["removal_dates"]

    membership = compute_point_in_time_membership(price_df.index, list(price_df.columns), added_dates, removal_dates)
    print(f"数据: {len(price_df.columns)}只票, {price_df.index[0].date()} ~ {price_df.index[-1].date()}")

    for name, kwargs in CONFIGS.items():
        periods_per_year = 252 / kwargs["rebalance_days"]
        results, daily_nav = backtest_value_strategy(price_df, membership, spy_close, **kwargs)
        m = metrics_from_periods(results, daily_nav=daily_nav, periods_per_year=periods_per_year)
        print(f"\n--- {name} ({kwargs}) ---")
        for r in results:
            print(f"  {r['rebalance_date']}: n_picks={r['n_picks']}, "
                  f"period_ret={r['period_ret']*100:+.2f}%, spy_ret={(r['spy_ret'] or 0)*100:+.2f}%")
        if m["CAGR"] is not None:
            print(f"  汇总: n_periods={m['n_periods']}, CAGR={m['CAGR']*100:.2f}%, "
                  f"Sharpe={m['Sharpe']:.3f}, MaxDD(调仓点间)={m['MaxDD']*100:.2f}%, "
                  f"MaxDD(逐日真实)={m['TrueDailyMaxDD']*100 if m['TrueDailyMaxDD'] is not None else float('nan'):.2f}%, "
                  f"SPY同期CAGR={m['SPY_CAGR']*100 if m['SPY_CAGR'] else float('nan'):.2f}%")
        else:
            print("  周期数不足")


if __name__ == "__main__":
    main()
