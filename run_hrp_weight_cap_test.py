"""
Cycle64小修(a)：给`_hrp_weights()`加了可选的`max_weight`参数(见README第290
条留下的开放问题)后，直接测试加30%单票权重上限对三段区间历史回测Sharpe/
MaxDD的实际影响——理论上应该影响很小(第290条已经确认极端集中是罕见事件，
1.1%-3.3%的调仓日)，这次直接验证这个预期，不假设。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from value_investing import select_value_portfolio, _hrp_weights

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"

PERIODS = [
    ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl", True),
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl", False),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl", False),
]

REBALANCE_DAYS = 5
COV_WINDOW = 60
STOP_LOSS_PCT = 0.03


def backtest_hrp_value(price_df, membership, max_weight):
    from run_survivorship_bias_fix_validation import compute_point_in_time_membership  # noqa
    index = price_df.index
    n = len(index)
    rebalance_idx = 252 + COV_WINDOW
    daily_nav = []
    running_nav = 1.0
    while rebalance_idx + REBALANCE_DAYS < n:
        rebalance_date = index[rebalance_idx]
        date_str = rebalance_date.strftime("%Y-%m-%d")
        valid_tickers = membership.get(date_str)
        if valid_tickers is None:
            keys = sorted(membership.keys())
            valid_tickers = membership[max([k for k in keys if k <= date_str], default=keys[0])]
        picks = [p for p in select_value_portfolio(price_df, valid_tickers, rebalance_date, top_n=30)
                 if p in price_df.columns]
        if len(picks) < 5:
            rebalance_idx += REBALANCE_DAYS
            continue
        hist_window = price_df[picks].loc[:rebalance_date].tail(COV_WINDOW + 1)
        rets_window = hist_window.pct_change().dropna()
        valid_cols = rets_window.columns[rets_window.notna().all()]
        if len(valid_cols) >= 5:
            hrp_w = _hrp_weights(rets_window[valid_cols], max_weight=max_weight)
            weights = pd.Series(0.0, index=picks)
            for t in valid_cols:
                weights[t] = hrp_w[t]
            weights = weights / weights.sum() if weights.sum() > 0 else pd.Series(1.0 / len(picks), index=picks)
        else:
            weights = pd.Series(1.0 / len(picks), index=picks)

        entry_prices = {t: price_df[t].loc[:rebalance_date].dropna().iloc[-1] for t in picks
                         if len(price_df[t].loc[:rebalance_date].dropna()) > 0}
        period_days = index[rebalance_idx:min(rebalance_idx + REBALANCE_DAYS, n)]
        stopped_out = {}
        for d in period_days:
            day_val, total_w = 0.0, 0.0
            for t, p0 in entry_prices.items():
                if d in price_df.index and t in price_df.columns:
                    w = weights.get(t, 0)
                    if t in stopped_out:
                        day_val += w * stopped_out[t]
                        total_w += w
                        continue
                    p = price_df.at[d, t]
                    if pd.notna(p) and p0:
                        factor = p / p0
                        if factor <= (1 - STOP_LOSS_PCT):
                            factor = 1 - STOP_LOSS_PCT
                            stopped_out[t] = factor
                        day_val += w * factor
                        total_w += w
            if total_w > 0:
                daily_nav.append((d, running_nav * (day_val / total_w)))
        if daily_nav:
            running_nav = daily_nav[-1][1]
        rebalance_idx += REBALANCE_DAYS

    nav_series = pd.DataFrame(daily_nav, columns=["date", "nav"]).set_index("date")["nav"]
    rets = nav_series.pct_change().dropna()
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252))
    peak = nav_series.cummax()
    mdd = float(((nav_series - peak) / peak).min())
    return sharpe, mdd


from run_survivorship_bias_fix_validation import compute_point_in_time_membership

for label, cache_path, is_pit in PERIODS:
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    price_df = cache["c"]
    if is_pit:
        membership = compute_point_in_time_membership(
            price_df.index, list(price_df.columns), cache["added_dates"], cache["removal_dates"])
    else:
        membership = {price_df.index[i].strftime("%Y-%m-%d"): list(price_df.columns) for i in range(len(price_df))}

    print(f"\n=== {label} ===")
    for mw in [None, 0.3]:
        s, m = backtest_hrp_value(price_df, membership, mw)
        label_mw = "无上限(现有默认)" if mw is None else f"上限{mw*100:.0f}%"
        print(f"  {label_mw}: Sharpe={s:.3f}, MaxDD={m*100:.2f}%")

print("\n全部完成。")
