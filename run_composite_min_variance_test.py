"""
Cycle34小修小补(a)：第170条的截面复合评分组合在2/3区间(2014-2020/2009-2014)
表现不如现有三元组合，这次测试叠加min_variance权重(第140条已验证的价值腿
仓位分配框架，这里换个应用对象)能不能把复合评分组合的表现拉上来——
如果个股选择本身没问题、只是等权分配次优，min_variance应该能带来改善；
如果叠加后依然不如现有配置，说明问题更可能出在选股逻辑(复合评分/动量
信号构造方式)本身，不是仓位分配层面。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from run_cross_sectional_composite_test import compute_composite_scores, _lookup_membership, MOM_WINDOW, SCRATCH
from value_investing import _min_variance_weights
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

COV_WINDOW = 60
REBALANCE_DAYS = 63


def backtest_composite_mv(price_df, membership, rebalance_days=REBALANCE_DAYS):
    index = price_df.index
    n = len(index)
    membership_keys = sorted(membership.keys())
    daily_nav = []
    rebalance_idx = MOM_WINDOW + COV_WINDOW + 5
    running_nav = 1.0

    while rebalance_idx + rebalance_days < n:
        rebalance_date = index[rebalance_idx]
        date_str = rebalance_date.strftime("%Y-%m-%d")
        valid = _lookup_membership(membership, membership_keys, date_str)
        if valid is None:
            rebalance_idx += rebalance_days
            continue

        picks = compute_composite_scores(price_df, valid, rebalance_date)
        picks = [p for p in picks if p in price_df.columns]
        if len(picks) < 5:
            rebalance_idx += rebalance_days
            continue

        hist_window = price_df[picks].loc[:rebalance_date].tail(COV_WINDOW + 1)
        rets_window = hist_window.pct_change().dropna()
        valid_cols = rets_window.columns[rets_window.notna().all()]
        if len(valid_cols) >= 5:
            mv_w = _min_variance_weights(rets_window[valid_cols])
            weights = pd.Series(0.0, index=picks)
            for t in valid_cols:
                weights[t] = mv_w[t]
            weights = weights / weights.sum() if weights.sum() > 0 else pd.Series(1.0 / len(picks), index=picks)
        else:
            weights = pd.Series(1.0 / len(picks), index=picks)

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
        price_df, spy_close = cache["c"], cache.get("spy")
        if is_pit:
            membership = compute_point_in_time_membership(price_df.index, list(price_df.columns),
                                                            cache["added_dates"], cache["removal_dates"])
        else:
            membership = compute_point_in_time_membership(price_df.index, list(price_df.columns), added_dates, {})

        nav = backtest_composite_mv(price_df, membership)
        if len(nav) < 30:
            print(f"\n=== {label} ===\n  样本不足，跳过")
            continue
        cagr, sharpe, mdd = metrics(nav)
        print(f"\n=== {label} ===")
        print(f"  复合评分+min_variance权重: CAGR={cagr*100:.2f}%, Sharpe={sharpe:.3f}, MaxDD={mdd*100:.2f}%")


if __name__ == "__main__":
    main()
