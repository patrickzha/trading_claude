"""
Cycle2小修小补：卖空余额/融券余量变化信号的独立标准化测试。用FINRA
双周更新的融券余量数据，算"最近一次结算的融券余量变化率"，按
decision_date做时点回溯(只用已经公开发布的最新一期数据，不会有前视
偏差——FINRA的settlement date到实际公开日期有几天滞后，这里简化为
settlement date当天就可获得，是保守估计里"信号更早生效"的方向，如果
测出来是负的，加上真实滞后只会更负，不影响"没有edge"这个结论方向；
如果测出来是正的，需要额外确认滞后期这个简化会不会让结论过于乐观)。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from scipy import stats

from backtest import get_weekly_trading_dates

PIT_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/point_in_time_universe_cache.pkl"
SI_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/short_interest_cache.pkl"


def main():
    with open(PIT_CACHE, "rb") as f:
        cache = pickle.load(f)
    open_df, close_df = cache["o"], cache["c"]

    with open(SI_CACHE, "rb") as f:
        si_cache = pickle.load(f)
    si_data = si_cache["data"]

    # 预处理：每只票按日期排好序，算相邻两期的变化率
    si_series = {}
    for ticker, records in si_data.items():
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df["short_qty"] = pd.to_numeric(df["short_qty"], errors="coerce")
        df["si_change_pct"] = df["short_qty"].pct_change()
        si_series[ticker] = df.dropna(subset=["si_change_pct"]).set_index("date")

    week_pairs = get_weekly_trading_dates(close_df.index)
    rows = []
    for monday, friday in week_pairs:
        loc = close_df.index.get_loc(monday)
        if loc == 0:
            continue
        decision_date = close_df.index[loc - 1]

        for ticker, df in si_series.items():
            if ticker not in open_df.columns or ticker not in close_df.columns:
                continue
            known = df[df.index <= decision_date]
            if len(known) == 0:
                continue
            si_change = known["si_change_pct"].iloc[-1]
            if pd.isna(si_change):
                continue

            try:
                buy = open_df.at[monday, ticker]
                sell = close_df.at[friday, ticker]
                if pd.isna(buy) or pd.isna(sell) or buy <= 0:
                    continue
                fwd_ret = (sell - buy) / buy
            except (KeyError, IndexError):
                continue

            rows.append({"date": monday.strftime("%Y-%m-%d"), "ticker": ticker,
                        "si_change": si_change, "fwd_ret": fwd_ret})

    df = pd.DataFrame(rows)
    print(f"总样本数: {len(df)}")
    if len(df) < 30:
        print("样本太少")
        return

    corr = df["si_change"].corr(df["fwd_ret"])
    print(f"si_change vs fwd_ret 相关系数: {corr:.4f}")

    q = df["si_change"].quantile([0.2, 0.4, 0.6, 0.8]).values
    print("\n按融券余量变化率分5档看前瞻收益率(理论：融券余量大增=更多人看空=可能预示下跌):")
    bins = [-np.inf] + list(q) + [np.inf]
    df["bucket"] = pd.cut(df["si_change"], bins=bins, labels=["Q1(降幅最大)", "Q2", "Q3", "Q4", "Q5(增幅最大)"])
    for label, sub in df.groupby("bucket", observed=True):
        print(f"  {label}: n={len(sub)}, 平均fwd_ret={sub['fwd_ret'].mean()*100:+.3f}%, "
              f"胜率={np.mean(sub['fwd_ret']>0):.1%}")

    q1 = df[df["bucket"] == "Q1(降幅最大)"]["fwd_ret"]
    q5 = df[df["bucket"] == "Q5(增幅最大)"]["fwd_ret"]
    t, p = stats.ttest_ind(q5, q1, equal_var=False)
    print(f"\nQ5(融券大增)vs Q1(融券大减) t检验: t={t:.3f}, p={p:.4f}")


if __name__ == "__main__":
    main()
