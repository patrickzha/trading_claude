"""
Cycle38小修小补(c)：偏度因子(第200/201/202条)跟价值因子组合测试——第201条
已经确认两者选股几乎不重叠(7%)，理论上具备互补分散化的条件(类似第55条
动量跟价值的低相关性)，这次实测二元组合(价值+偏度各半资金，各自独立
选股)相对价值单腿(第38条)和偏度单腿(第200条)是否有改善。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from value_investing import backtest_value_strategy
from run_skewness_factor_test import backtest_skew
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"


def true_mdd(nav):
    nav = np.array(nav)
    peak = np.maximum.accumulate(nav)
    return float(np.min((nav - peak) / peak))


def metrics_from_nav_series(nav: pd.Series):
    rets = nav.pct_change().dropna()
    n_years = len(nav) / 252
    cagr = float(nav.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else None
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252))
    mdd = true_mdd(nav.values)
    return cagr, sharpe, mdd


def metrics_from_navlist(nav_list):
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

        _, value_nav_list = backtest_value_strategy(price_df, membership, spy_close, rebalance_days=21)
        skew_nav_list = backtest_skew(price_df, membership)

        value_nav = pd.DataFrame(value_nav_list, columns=["date", "nav"]).set_index("date")["nav"]
        skew_nav = pd.DataFrame(skew_nav_list, columns=["date", "nav"]).set_index("date")["nav"]

        combined = pd.DataFrame({"value": value_nav, "skew": skew_nav}).dropna()
        combined_norm = combined / combined.iloc[0]
        combo_nav = combined_norm.mean(axis=1)

        v_cagr, v_sharpe, v_mdd = metrics_from_navlist(value_nav_list)
        s_cagr, s_sharpe, s_mdd = metrics_from_navlist(skew_nav_list)
        c_cagr, c_sharpe, c_mdd = metrics_from_nav_series(combo_nav)

        print(f"\n=== {label} ===")
        print(f"  价值单腿: CAGR={v_cagr*100:.2f}%, Sharpe={v_sharpe:.3f}, MaxDD={v_mdd*100:.2f}%")
        print(f"  偏度单腿: CAGR={s_cagr*100:.2f}%, Sharpe={s_sharpe:.3f}, MaxDD={s_mdd*100:.2f}%")
        print(f"  价值+偏度二元组合: CAGR={c_cagr*100:.2f}%, Sharpe={c_sharpe:.3f}, MaxDD={c_mdd*100:.2f}%")


if __name__ == "__main__":
    main()
