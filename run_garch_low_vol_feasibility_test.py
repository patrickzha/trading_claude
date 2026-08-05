"""
Cycle66大修大补②(续)：GARCH低波动腿可行性初筛——先在2022-2026单段区间测
(不是三段都上，控制耗时)，用跟`low_vol_investing.py`同等配置(21天调仓、
30只、等权、不含止损)做公平对比，如果方向有希望再决定要不要投入三段区间
完整验证。
"""
from __future__ import annotations

import pickle
import time

import numpy as np
import pandas as pd

from garch_low_vol_investing import select_garch_low_vol_portfolio
from low_vol_investing import select_low_vol_portfolio
from run_survivorship_bias_fix_validation import compute_point_in_time_membership, CACHE_PATH

REBALANCE_DAYS = 21
TOP_N = 30


def backtest_simple(price_df, membership, select_fn, **kwargs):
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

        picks = select_fn(price_df, valid_tickers, rebalance_date, top_n=TOP_N, **kwargs)
        picks = [p for p in picks if p in price_df.columns]
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
    return {"Sharpe": sharpe, "CAGR": cagr, "MaxDD": mdd, "n_days": len(nav_series)}


def main():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    price_df, spy_close = cache["c"], cache.get("spy")
    membership = compute_point_in_time_membership(
        price_df.index, list(price_df.columns), cache["added_dates"], cache["removal_dates"])

    print("跑GARCH低波动腿(可能较慢，逐票拟合GARCH(1,1))...")
    t0 = time.time()
    r_garch = backtest_simple(price_df, membership, select_garch_low_vol_portfolio)
    elapsed = (time.time() - t0) / 60
    print(f"GARCH低波动腿: Sharpe={r_garch['Sharpe']:.3f}, CAGR={r_garch['CAGR']*100:.2f}%, "
          f"MaxDD={r_garch['MaxDD']*100:.2f}%, 耗时={elapsed:.1f}分钟")

    print("\n跑现有trailing已实现波动率低波动腿(对照)...")
    r_realized = backtest_simple(price_df, membership, select_low_vol_portfolio)
    print(f"现有实现波动率低波动腿: Sharpe={r_realized['Sharpe']:.3f}, CAGR={r_realized['CAGR']*100:.2f}%, "
          f"MaxDD={r_realized['MaxDD']*100:.2f}%")

    print("\n全部完成。")


if __name__ == "__main__":
    main()
