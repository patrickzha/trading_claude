"""
Cycle38小修小补(d)：下行偏度(只用负收益部分计算偏度，更精确的尾部风险
度量)代替普通偏度(第200条)，检验这个更精细的实现方式是否比普通偏度
证据更强。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from scipy.stats import skew

from run_survivorship_bias_fix_validation import compute_point_in_time_membership
from run_skewness_factor_test import SCRATCH, LOOKBACK, REBALANCE_DAYS, TOP_N, _lookup_membership, true_mdd, metrics


def select_low_downside_skew_portfolio(price_df, valid_tickers, rebalance_date, top_n=TOP_N):
    rows = []
    for t in valid_tickers:
        if t not in price_df.columns:
            continue
        px = price_df[t].loc[:rebalance_date].dropna().tail(LOOKBACK)
        if len(px) < LOOKBACK - 5:
            continue
        rets = px.pct_change().dropna()
        neg_rets = rets[rets < 0]
        if len(neg_rets) < 15:
            continue
        sk = float(skew(neg_rets))
        rows.append({"ticker": t, "downside_skew": sk})

    if len(rows) < top_n:
        return [r["ticker"] for r in rows]
    df = pd.DataFrame(rows).sort_values("downside_skew", ascending=False)
    return df["ticker"].head(top_n).tolist()


def backtest_downside_skew(price_df, membership, rebalance_days=REBALANCE_DAYS):
    index = price_df.index
    n = len(index)
    membership_keys = sorted(membership.keys())
    daily_nav = []
    rebalance_idx = LOOKBACK + 5
    running_nav = 1.0

    while rebalance_idx + rebalance_days < n:
        rebalance_date = index[rebalance_idx]
        date_str = rebalance_date.strftime("%Y-%m-%d")
        valid = _lookup_membership(membership, membership_keys, date_str)
        if valid is None:
            rebalance_idx += rebalance_days
            continue

        picks = select_low_downside_skew_portfolio(price_df, valid, rebalance_date)
        picks = [p for p in picks if p in price_df.columns]
        if len(picks) < 5:
            rebalance_idx += rebalance_days
            continue

        period_days = index[rebalance_idx:min(rebalance_idx + rebalance_days, n)]
        entry_prices = {t: price_df[t].loc[:rebalance_date].dropna().iloc[-1] for t in picks}
        for d in period_days:
            day_val, total_w = 0.0, 0.0
            for t, p0 in entry_prices.items():
                if d in price_df.index and t in price_df.columns:
                    p = price_df.at[d, t]
                    if pd.notna(p) and p0:
                        day_val += p / p0
                        total_w += 1
            if total_w > 0:
                daily_nav.append((d, running_nav * (day_val / total_w)))
        if daily_nav:
            running_nav = daily_nav[-1][1]
        rebalance_idx += rebalance_days

    return daily_nav


def main():
    with open("/Users/zhang/Desktop/trading/sp500_membership_added_dates.json") as f:
        import json
        added_dates_raw = json.load(f)
    added_dates = {t: pd.Timestamp(d) for t, d in added_dates_raw.items()}

    periods = [
        ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl", True),
        ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl", False),
        ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl", False),
    ]

    for label, price_cache_path, is_pit in periods:
        with open(price_cache_path, "rb") as f:
            cache = pickle.load(f)
        price_df, spy_close = cache["c"], cache.get("spy")
        if is_pit:
            membership = compute_point_in_time_membership(price_df.index, list(price_df.columns),
                                                            cache["added_dates"], cache["removal_dates"])
        else:
            membership = compute_point_in_time_membership(price_df.index, list(price_df.columns), added_dates, {})

        nav = backtest_downside_skew(price_df, membership)
        print(f"\n=== {label} ===")
        if len(nav) < 30:
            print("  样本不足，跳过")
            continue
        cagr, sharpe, mdd = metrics(nav)
        print(f"  下行偏度组合: CAGR={cagr*100:.2f}%, Sharpe={sharpe:.3f}, MaxDD={mdd*100:.2f}%")


if __name__ == "__main__":
    main()
