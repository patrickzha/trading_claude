"""
Cycle62小修(d)：EV/EBIT因子(第284条)的max_multiple筛选阈值(默认60倍)敏感性
测试，跟原始价值因子的max_pe=40这个已经隐含存在但从未专门测试过敏感性的
筛选参数类似，这次直接给EV/EBIT做一次。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from ev_ebit_investing import select_ev_ebit_portfolio
from run_survivorship_bias_fix_validation import compute_point_in_time_membership, CACHE_PATH

REBALANCE_DAYS = 126
TOP_N = 30


def backtest_simple(price_df, membership, max_multiple):
    index = price_df.index
    n = len(index)
    rebalance_idx = 252
    daily_nav = []
    running_nav = 1.0
    while rebalance_idx + REBALANCE_DAYS < n:
        rebalance_date = index[rebalance_idx]
        date_str = rebalance_date.strftime("%Y-%m-%d")
        valid_tickers = membership.get(date_str)
        if valid_tickers is None:
            keys = sorted(membership.keys())
            valid_tickers = membership[max([k for k in keys if k <= date_str], default=keys[0])]
        picks = [p for p in select_ev_ebit_portfolio(price_df, valid_tickers, rebalance_date,
                                                       top_n=TOP_N, max_multiple=max_multiple)
                 if p in price_df.columns]
        period_days = index[rebalance_idx:min(rebalance_idx + REBALANCE_DAYS, n)]
        entry_prices = {t: price_df[t].loc[:rebalance_date].dropna().iloc[-1] for t in picks
                         if len(price_df[t].loc[:rebalance_date].dropna()) > 0}
        for d in period_days:
            vals = []
            for t, p0 in entry_prices.items():
                if d in price_df.index and t in price_df.columns:
                    p = price_df.at[d, t]
                    if pd.notna(p) and p0:
                        vals.append(p / p0)
            if vals:
                daily_nav.append((d, running_nav * np.mean(vals)))
        if daily_nav:
            running_nav = daily_nav[-1][1]
        rebalance_idx += REBALANCE_DAYS
    nav_series = pd.DataFrame(daily_nav, columns=["date", "nav"]).set_index("date")["nav"]
    rets = nav_series.pct_change().dropna()
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252))
    n_years = len(nav_series) / 252
    cagr = float(nav_series.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else None
    peak = nav_series.cummax()
    mdd = float(((nav_series - peak) / peak).min())
    return sharpe, cagr, mdd


with open(CACHE_PATH, "rb") as f:
    cache = pickle.load(f)
price_df, spy_close = cache["c"], cache.get("spy")
membership = compute_point_in_time_membership(
    price_df.index, list(price_df.columns), cache["added_dates"], cache["removal_dates"])

print(f"{'='*70}\n EV/EBIT max_multiple筛选阈值敏感性(默认60)\n{'='*70}")
for mm in [30, 45, 60, 90]:
    s, c, m = backtest_simple(price_df, membership, mm)
    print(f"  max_multiple={mm}: Sharpe={s:.3f}, CAGR={c*100:.2f}%, MaxDD={m*100:.2f}%")

print("\n全部完成。")
