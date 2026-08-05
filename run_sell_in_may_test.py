"""
Cycle48大修大补①(框架改动，新数据维度——日历效应)："Sell in May and go
away"是股票市场最经典的日历异象之一(5月到10月降低敞口，11月到次年4月
满仓)，机制上众说纷纭(夏季交易清淡、机构调仓周期等)，学术界证据混杂，
不同市场/不同年代结论不一致。这是完全不同于价格/基本面/成交量的新数据
维度(纯日历时间)，直接对三元组合实测，不预判方向。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from combo_strategy import run_combo_strategy
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
SUMMER_EXPOSURE = 0.5  # 5-10月降到50%敞口


def true_mdd(nav):
    nav = np.array(nav)
    peak = np.maximum.accumulate(nav)
    return float(np.min((nav - peak) / peak))


def apply_sell_in_may(combo_nav: pd.Series, summer_exposure: float = SUMMER_EXPOSURE) -> pd.Series:
    rets = combo_nav.pct_change().dropna()
    is_summer = rets.index.month.isin([5, 6, 7, 8, 9, 10])
    adjusted_rets = rets.copy()
    adjusted_rets[is_summer] = rets[is_summer] * summer_exposure
    return (1 + adjusted_rets).cumprod()


def metrics_from_nav(nav: pd.Series):
    rets = nav.pct_change().dropna()
    n_years = len(nav) / 252
    cagr = float(nav.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else None
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252))
    mdd = true_mdd(nav.values)
    return cagr, sharpe, mdd


def main():
    with open("/Users/zhang/Desktop/trading/sp500_membership_added_dates.json") as f:
        import json
        added_dates_raw = json.load(f)
    added_dates = {t: pd.Timestamp(d) for t, d in added_dates_raw.items()}

    periods = [
        ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl",
         f"{SCRATCH}/default_full_results_cache.pkl", True),
        ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl",
         f"{SCRATCH}/momentum_2014_2020_results_cache.pkl", False),
        ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl",
         f"{SCRATCH}/momentum_2009_2014_results_cache.pkl", False),
    ]

    for label, price_cache_path, mom_cache_path, is_pit in periods:
        with open(price_cache_path, "rb") as f:
            cache = pickle.load(f)
        price_df, spy_close = cache["c"], cache.get("spy")
        with open(mom_cache_path, "rb") as f:
            mom_results = pickle.load(f)

        if is_pit:
            membership = compute_point_in_time_membership(price_df.index, list(price_df.columns),
                                                            cache["added_dates"], cache["removal_dates"])
        else:
            membership = compute_point_in_time_membership(price_df.index, list(price_df.columns), added_dates, {})

        report = run_combo_strategy(price_df, membership, spy_close, mom_results)
        combo_nav = report["combo_nav"]

        b_cagr, b_sharpe, b_mdd = report["CAGR"], report["Sharpe"], report["MaxDD"]
        sim_nav = apply_sell_in_may(combo_nav)
        s_cagr, s_sharpe, s_mdd = metrics_from_nav(sim_nav)

        print(f"\n=== {label} ===")
        print(f"  基线(全年满仓): CAGR={b_cagr*100:.2f}%, Sharpe={b_sharpe:.3f}, MaxDD={b_mdd*100:.2f}%")
        print(f"  Sell in May(5-10月降到{SUMMER_EXPOSURE:.0%}敞口): CAGR={s_cagr*100:.2f}%, Sharpe={s_sharpe:.3f}, MaxDD={s_mdd*100:.2f}%")


if __name__ == "__main__":
    main()
