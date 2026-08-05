"""
Cycle45大修大补②(框架改动，新机制——止盈规则)：只测过止损(第230-239条)，
从未测过止盈(上涨超过阈值就锁定收益，不再继续持有到调仓日)——这是
管理"利润是否应该提前锁定"这个不同于下行风险管理的问题。价值腿"低估
+优质"的选股逻辑，如果一只票短期内涨幅过大，可能意味着估值优势已经
被市场消化完毕，提前止盈锁定收益、把资金腾出来可能有正贡献；也可能
反过来错过后续的进一步上涨(动量延续)。方向不预判，直接实测。
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


def backtest_value_with_takeprofit(price_df, membership, rebalance_days=REBALANCE_DAYS, take_profit_pct=None):
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
        locked: dict[str, float] = {}

        for d in period_days:
            day_val, total_w = 0.0, 0.0
            for t, p0 in entry_prices.items():
                if d in price_df.index and t in price_df.columns:
                    if t in locked:
                        day_val += locked[t]
                        total_w += 1
                        continue
                    p = price_df.at[d, t]
                    if pd.notna(p) and p0:
                        factor = p / p0
                        if take_profit_pct is not None and factor >= (1 + take_profit_pct):
                            factor = 1 + take_profit_pct
                            locked[t] = factor
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
        nav_base = backtest_value_with_takeprofit(price_df, membership, take_profit_pct=None)
        b_cagr, b_sharpe, b_mdd = metrics(nav_base)
        print(f"  基线(无止盈): CAGR={b_cagr*100:.2f}%, Sharpe={b_sharpe:.3f}, MaxDD={b_mdd*100:.2f}%")

        for tp in [0.15, 0.25]:
            nav_tp = backtest_value_with_takeprofit(price_df, membership, take_profit_pct=tp)
            t_cagr, t_sharpe, t_mdd = metrics(nav_tp)
            print(f"  止盈{tp:.0%}: CAGR={t_cagr*100:.2f}%, Sharpe={t_sharpe:.3f}, MaxDD={t_mdd*100:.2f}%")


if __name__ == "__main__":
    main()
