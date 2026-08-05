"""
Cycle47大修大补②(框架改动，新机制——移动止损)：现有止损(第230-244条)是
固定止损(相对买入价下跌超过阈值触发)，这次测试移动止损(trailing stop，
相对持有期内的最高价下跌超过阈值触发)——这是一个真正不同的机制：固定
止损只保护"买入后没涨就跌"这种情形，移动止损额外保护"先涨后跌回吐利润"
这种情形，理论上应该能在牛市中段回调时提供额外保护，但代价是可能过早
被正常的价格波动"甩出"盈利头寸(丧失后续的进一步上涨)。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from value_investing import select_value_portfolio
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
REBALANCE_DAYS = 21
TOP_N = 30


def _lookup_membership(membership, membership_keys, date_str):
    valid = membership.get(date_str)
    if valid is not None:
        return valid
    candidates = [k for k in membership_keys if k <= date_str]
    if not candidates:
        return None
    return membership[max(candidates)]


def backtest_value_trailing_stop(price_df, membership, rebalance_days=REBALANCE_DAYS, stop_loss_pct=0.05,
                                  trailing=False):
    index = price_df.index
    n = len(index)
    membership_keys = sorted(membership.keys())
    daily_nav = []
    rebalance_idx = 252 + 5
    running_nav = 1.0

    while rebalance_idx + rebalance_days < n:
        rebalance_date = index[rebalance_idx]
        date_str = rebalance_date.strftime("%Y-%m-%d")
        valid = _lookup_membership(membership, membership_keys, date_str)
        if valid is None:
            rebalance_idx += rebalance_days
            continue

        picks = select_value_portfolio(price_df, valid, rebalance_date, top_n=TOP_N)
        picks = [p for p in picks if p in price_df.columns]
        if len(picks) < 5:
            rebalance_idx += rebalance_days
            continue

        period_days = index[rebalance_idx:min(rebalance_idx + rebalance_days, n)]
        entry_prices = {t: price_df[t].loc[:rebalance_date].dropna().iloc[-1] for t in picks}
        peak_factor = {t: 1.0 for t in picks}
        stopped_out: dict[str, float] = {}

        for d in period_days:
            day_val, total_w = 0.0, 0.0
            for t, p0 in entry_prices.items():
                if t in stopped_out:
                    day_val += stopped_out[t]
                    total_w += 1
                    continue
                if d in price_df.index and t in price_df.columns:
                    p = price_df.at[d, t]
                    if pd.notna(p) and p0:
                        factor = p / p0
                        if trailing:
                            peak_factor[t] = max(peak_factor[t], factor)
                            threshold = peak_factor[t] * (1 - stop_loss_pct)
                        else:
                            threshold = 1 - stop_loss_pct
                        if factor <= threshold:
                            factor = threshold
                            stopped_out[t] = factor
                        day_val += factor
                        total_w += 1
            if total_w > 0:
                daily_nav.append((d, running_nav * (day_val / total_w)))
        if daily_nav:
            running_nav = daily_nav[-1][1]
        rebalance_idx += rebalance_days

    return daily_nav


def true_mdd(nav):
    nav = np.array(nav)
    peak = np.maximum.accumulate(nav)
    return float(np.min((nav - peak) / peak))


def metrics(nav_list):
    nav = np.array([n for _, n in nav_list])
    n_years = len(nav) / 252
    cagr = float((nav[-1] / nav[0]) ** (1 / n_years) - 1) if n_years > 0 else None
    rets = np.diff(nav) / nav[:-1]
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252))
    mdd = true_mdd(nav)
    return cagr, sharpe, mdd


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
        price_df = cache["c"]
        if is_pit:
            membership = compute_point_in_time_membership(price_df.index, list(price_df.columns),
                                                            cache["added_dates"], cache["removal_dates"])
        else:
            membership = compute_point_in_time_membership(price_df.index, list(price_df.columns), added_dates, {})

        print(f"\n=== {label} ===")
        nav_fixed = backtest_value_trailing_stop(price_df, membership, stop_loss_pct=0.05, trailing=False)
        f_cagr, f_sharpe, f_mdd = metrics(nav_fixed)
        print(f"  固定止损5%(第236-238条): CAGR={f_cagr*100:.2f}%, Sharpe={f_sharpe:.3f}, MaxDD={f_mdd*100:.2f}%")

        nav_trail = backtest_value_trailing_stop(price_df, membership, stop_loss_pct=0.05, trailing=True)
        t_cagr, t_sharpe, t_mdd = metrics(nav_trail)
        print(f"  移动止损5%: CAGR={t_cagr*100:.2f}%, Sharpe={t_sharpe:.3f}, MaxDD={t_mdd*100:.2f}%")


if __name__ == "__main__":
    main()
