"""
Cycle27大修大补②：第140条的最小方差组合优化框架在价值腿上三段全部改善，
这次测同一个框架能不能推广到低波动腿——低波动腿本身选的就是已实现波动率
最低的30只票，候选池内部的波动率差异天然比价值腿小(价值腿是按估值选，
波动率是完全独立的维度，候选池内部波动率差异可能很大)，最小方差优化
在这种"本来就已经不那么分散"的候选池上还能不能带来同样的边际改善，
是一个开放问题，如实测。
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from low_vol_investing import select_low_vol_portfolio
from value_investing import _min_variance_weights
from combo_strategy import true_max_drawdown
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
COV_WINDOW = 60


def backtest_weighted_lowvol(price_df, membership, spy_close, rebalance_days=21, top_n=30,
                              cost_bi=0.003, weighting="equal"):
    index = price_df.index
    n = len(index)
    daily_nav = []
    rebalance_idx = 100 + (COV_WINDOW if weighting == "min_variance" else 0)
    running_nav = 1.0
    prev_weights = None

    while rebalance_idx + rebalance_days < n:
        rebalance_date = index[rebalance_idx]
        date_str = rebalance_date.strftime("%Y-%m-%d")
        valid_tickers = membership.get(date_str)
        if valid_tickers is None:
            keys = sorted(membership.keys())
            valid_tickers = membership[max([k for k in keys if k <= date_str], default=keys[0])]

        picks = select_low_vol_portfolio(price_df, valid_tickers, rebalance_date, top_n=top_n)
        picks = [p for p in picks if p in price_df.columns]
        if len(picks) < 5:
            rebalance_idx += rebalance_days
            continue

        if weighting == "min_variance":
            hist_window = price_df[picks].loc[:rebalance_date].tail(COV_WINDOW + 1)
            rets_window = hist_window.pct_change().dropna()
            valid_cols = rets_window.columns[rets_window.notna().all()]
            if len(valid_cols) >= 5:
                mv_weights = _min_variance_weights(rets_window[valid_cols])
                weights = pd.Series(0.0, index=picks)
                for t in valid_cols:
                    weights[t] = mv_weights[t]
                weights = weights / weights.sum() if weights.sum() > 0 else pd.Series(1.0 / len(picks), index=picks)
            else:
                weights = pd.Series(1.0 / len(picks), index=picks)
        else:
            weights = pd.Series(1.0 / len(picks), index=picks)

        turnover = 1.0 if prev_weights is None else sum(
            abs(weights.get(t, 0) - prev_weights.get(t, 0)) for t in set(weights.index) | set(prev_weights.index)) / 2
        running_nav *= (1 - turnover * cost_bi)
        prev_weights = weights

        period_days = index[rebalance_idx:min(rebalance_idx + rebalance_days, n)]
        entry_prices = {}
        for t in picks:
            try:
                entry_prices[t] = price_df[t].loc[:rebalance_date].dropna().iloc[-1]
            except IndexError:
                continue
        for d in period_days:
            day_val, total_w = 0.0, 0.0
            for t, p0 in entry_prices.items():
                if d in price_df.index and t in price_df.columns:
                    p = price_df.at[d, t]
                    if pd.notna(p) and p0:
                        w = weights.get(t, 0)
                        day_val += w * (p / p0)
                        total_w += w
            if total_w > 0:
                daily_nav.append((d, running_nav * (day_val / total_w)))
        if daily_nav:
            running_nav = daily_nav[-1][1]
        rebalance_idx += rebalance_days

    return daily_nav


def metrics(nav_list):
    nav = np.array([n for _, n in nav_list])
    n_years = len(nav) / 252
    cagr = float((nav[-1] / nav[0]) ** (1 / n_years) - 1) if n_years > 0 else None
    rets = np.diff(nav) / nav[:-1]
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252))
    mdd = true_max_drawdown(nav)
    return cagr, sharpe, mdd


periods = [
    ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl", True),
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl", False),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl", False),
]
with open("/Users/zhang/Desktop/trading/sp500_membership_added_dates.json") as f:
    added_dates_raw = json.load(f)
added_dates = {t: pd.Timestamp(d) for t, d in added_dates_raw.items()}

for label, price_cache_path, is_pit in periods:
    with open(price_cache_path, "rb") as f:
        cache = pickle.load(f)
    price_df, spy_close = cache["c"], cache.get("spy")
    if is_pit:
        membership = compute_point_in_time_membership(price_df.index, list(price_df.columns),
                                                        cache["added_dates"], cache["removal_dates"])
    else:
        membership = compute_point_in_time_membership(price_df.index, list(price_df.columns), added_dates, {})

    nav_equal = backtest_weighted_lowvol(price_df, membership, spy_close, weighting="equal")
    nav_mv = backtest_weighted_lowvol(price_df, membership, spy_close, weighting="min_variance")
    e_cagr, e_sharpe, e_mdd = metrics(nav_equal)
    m_cagr, m_sharpe, m_mdd = metrics(nav_mv)
    print(f"\n=== {label} ===")
    print(f"  等权(现有): CAGR={e_cagr*100:.2f}%, Sharpe={e_sharpe:.3f}, MaxDD={e_mdd*100:.2f}%")
    print(f"  最小方差组合优化: CAGR={m_cagr*100:.2f}%, Sharpe={m_sharpe:.3f}, MaxDD={m_mdd*100:.2f}%")
