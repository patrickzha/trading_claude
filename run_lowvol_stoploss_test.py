"""
Cycle44大修大补②(框架改动，止损推广测试)：把第230-233条价值腿止损的
正面发现推广到低波动腿，检验这个新风控机制是否像min_variance权重
(第141条推广失败)那样只在价值腿场景下有效，还是像"止损"这种更通用
的下行保护机制一样能跨场景生效——理论上止损不依赖选股逻辑本身，
应该比仓位分配类框架更容易泛化。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from low_vol_investing import select_low_vol_portfolio
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


def backtest_lowvol_with_stoploss(price_df, membership, rebalance_days=REBALANCE_DAYS, stop_loss_pct=None):
    index = price_df.index
    n = len(index)
    membership_keys = sorted(membership.keys())
    daily_nav = []
    rebalance_idx = 100
    running_nav = 1.0

    while rebalance_idx + rebalance_days < n:
        rebalance_date = index[rebalance_idx]
        date_str = rebalance_date.strftime("%Y-%m-%d")
        valid = _lookup_membership(membership, membership_keys, date_str)
        if valid is None:
            rebalance_idx += rebalance_days
            continue

        picks = select_low_vol_portfolio(price_df, valid, rebalance_date, top_n=TOP_N)
        picks = [p for p in picks if p in price_df.columns]
        if len(picks) < 5:
            rebalance_idx += rebalance_days
            continue

        period_days = index[rebalance_idx:min(rebalance_idx + rebalance_days, n)]
        entry_prices = {t: price_df[t].loc[:rebalance_date].dropna().iloc[-1] for t in picks}
        stopped_out = {}

        for d in period_days:
            day_val, total_w = 0.0, 0.0
            for t, p0 in entry_prices.items():
                if d in price_df.index and t in price_df.columns:
                    if t in stopped_out:
                        day_val += stopped_out[t]
                        total_w += 1
                        continue
                    p = price_df.at[d, t]
                    if pd.notna(p) and p0:
                        factor = p / p0
                        if stop_loss_pct is not None and factor <= (1 - stop_loss_pct):
                            factor = 1 - stop_loss_pct
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
        nav_base = backtest_lowvol_with_stoploss(price_df, membership, stop_loss_pct=None)
        b_cagr, b_sharpe, b_mdd = metrics(nav_base)
        print(f"  基线(无止损): CAGR={b_cagr*100:.2f}%, Sharpe={b_sharpe:.3f}, MaxDD={b_mdd*100:.2f}%")

        nav_sl = backtest_lowvol_with_stoploss(price_df, membership, stop_loss_pct=0.10)
        s_cagr, s_sharpe, s_mdd = metrics(nav_sl)
        print(f"  止损10%: CAGR={s_cagr*100:.2f}%, Sharpe={s_sharpe:.3f}, MaxDD={s_mdd*100:.2f}%")


if __name__ == "__main__":
    main()
