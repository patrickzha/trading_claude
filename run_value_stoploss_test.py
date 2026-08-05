"""
Cycle44大修大补①(框架改动，新风控机制——止损推广到长周期腿)：现有逐票
止损(2%阈值)只应用在动量腿(周频持有，止损空间小)，价值腿(月度/半年度
调仓，持有期长得多)从未测试过止损保护。这次测试给价值腿加一个止损
规则(个股相对买入价跌破阈值就提前卖出、空出的资金保持现金到下次
调仓)，理论上长周期持有腿单次亏损空间更大，止损保护的潜在价值可能
比动量腿更大。
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


def backtest_value_with_stoploss(price_df, membership, rebalance_days=REBALANCE_DAYS, stop_loss_pct=None):
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
        stopped_out = set()

        for d in period_days:
            day_val, total_w = 0.0, 0.0
            for t, p0 in entry_prices.items():
                if d in price_df.index and t in price_df.columns:
                    p = price_df.at[d, t]
                    if pd.notna(p) and p0:
                        if t in stopped_out:
                            day_val += (1 - stop_loss_pct) if stop_loss_pct else p / p0
                            total_w += 1
                            continue
                        ret = p / p0 - 1
                        if stop_loss_pct is not None and ret <= -stop_loss_pct:
                            stopped_out.add(t)
                            day_val += (1 - stop_loss_pct)
                        else:
                            day_val += p / p0
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
        nav_base = backtest_value_with_stoploss(price_df, membership, stop_loss_pct=None)
        b_cagr, b_sharpe, b_mdd = metrics(nav_base)
        print(f"  基线(无止损): CAGR={b_cagr*100:.2f}%, Sharpe={b_sharpe:.3f}, MaxDD={b_mdd*100:.2f}%")

        for sl in [0.10, 0.15, 0.20]:
            nav_sl = backtest_value_with_stoploss(price_df, membership, stop_loss_pct=sl)
            if len(nav_sl) < 30:
                continue
            s_cagr, s_sharpe, s_mdd = metrics(nav_sl)
            print(f"  止损{sl:.0%}: CAGR={s_cagr*100:.2f}%, Sharpe={s_sharpe:.3f}, MaxDD={s_mdd*100:.2f}%")


if __name__ == "__main__":
    main()
