"""
Cycle2小修小补：分析师预期修正动量信号的独立标准化测试(不进主pipeline，
参考PEAD那次的做法)。用"最近90天净升评数(升评-降评)"这个时点特征，
跟接下来一周的实际收益率做相关性/分位数分析，先看有没有戏，再决定
要不要往主策略里整合。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from backtest import get_weekly_trading_dates

PIT_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/point_in_time_universe_cache.pkl"
REVISIONS_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/analyst_revisions_cache.pkl"

LOOKBACK_DAYS = 90


def main():
    with open(PIT_CACHE, "rb") as f:
        cache = pickle.load(f)
    open_df, close_df = cache["o"], cache["c"]

    with open(REVISIONS_CACHE, "rb") as f:
        rev_cache = pickle.load(f)
    rev_data = rev_cache["data"]

    week_pairs = get_weekly_trading_dates(close_df.index)
    rows = []
    for monday, friday in week_pairs:
        loc = close_df.index.get_loc(monday)
        if loc == 0:
            continue
        decision_date = close_df.index[loc - 1]
        lookback_start = decision_date - pd.Timedelta(days=LOOKBACK_DAYS)

        for ticker, df in rev_data.items():
            if ticker not in open_df.columns or ticker not in close_df.columns:
                continue
            window = df[(df["GradeDate"] > lookback_start) & (df["GradeDate"] <= decision_date)]
            if len(window) == 0:
                continue
            net_rev = int((window["Action"] == "up").sum()) - int((window["Action"] == "down").sum())
            if net_rev == 0:
                continue  # 只关心有明确净升/降评信号的票-周组合

            try:
                buy = open_df.at[monday, ticker]
                sell = close_df.at[friday, ticker]
                if pd.isna(buy) or pd.isna(sell) or buy <= 0:
                    continue
                fwd_ret = (sell - buy) / buy
            except (KeyError, IndexError):
                continue

            rows.append({"date": monday.strftime("%Y-%m-%d"), "ticker": ticker,
                        "net_rev": net_rev, "fwd_ret": fwd_ret})

    df = pd.DataFrame(rows)
    print(f"总样本数(有明确净升降评信号的票-周组合): {len(df)}")
    if len(df) < 30:
        print("样本太少，无法做有意义的分析")
        return

    corr = df["net_rev"].corr(df["fwd_ret"])
    print(f"net_rev vs fwd_ret 相关系数: {corr:.4f}")

    print("\n按net_rev分组看平均前瞻收益率:")
    for label, mask in [("net_rev>=2(强烈升评)", df["net_rev"] >= 2),
                        ("net_rev==1(轻度升评)", df["net_rev"] == 1),
                        ("net_rev==-1(轻度降评)", df["net_rev"] == -1),
                        ("net_rev<=-2(强烈降评)", df["net_rev"] <= -2)]:
        sub = df[mask]
        if len(sub) > 0:
            print(f"  {label}: n={len(sub)}, 平均fwd_ret={sub['fwd_ret'].mean()*100:+.3f}%, "
                  f"胜率={np.mean(sub['fwd_ret']>0):.1%}")


if __name__ == "__main__":
    main()
