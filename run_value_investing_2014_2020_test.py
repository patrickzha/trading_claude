"""
长周期价值投资策略在2014-2020独立历史区间上的复核——这段区间包含2018Q4
暴跌和2020年3月新冠暴跌，能不能扛住真正的熊市，是判断"验证结论"里那个
2022-2026区间(MaxDD=0.00%)到底是真实规律还是纯粹幸运窗口的关键测试。

跟"验证结论"第17条同样的已知局限：这份缓存用的是"今天"的484只成分股
名单去跑2014-2020的历史，没有点位时间正确的成分股过滤(生存偏差)，
结论方向可信、绝对数字可能偏乐观。
"""
from __future__ import annotations

import pickle

import numpy as np

from value_investing import backtest_value_strategy, metrics_from_periods

CACHE_PATH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/historical_2014_2020_cache.pkl"


def main():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    price_df = cache["c"]
    spy_close = cache.get("spy")
    print(f"数据: {len(price_df.columns)}只票, {price_df.index[0].date()} ~ {price_df.index[-1].date()}")

    # 没有point-in-time membership，全部484只票都当有效候选(跟"验证结论"
    # 第17条同样的已知局限)
    all_tickers = list(price_df.columns)
    fake_membership = {price_df.index[i].strftime("%Y-%m-%d"): all_tickers for i in range(len(price_df))}

    for name, kwargs in {
        "top30_semiannual": {"rebalance_days": 126, "top_n": 30},
        "top30_quarterly": {"rebalance_days": 63, "top_n": 30},
    }.items():
        periods_per_year = 252 / kwargs["rebalance_days"]
        results, daily_nav = backtest_value_strategy(price_df, fake_membership, spy_close, **kwargs)
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


if __name__ == "__main__":
    main()
