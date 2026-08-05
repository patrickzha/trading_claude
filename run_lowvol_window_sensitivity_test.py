"""
Cycle17小修小补：低波动策略的vol_window(计算已实现波动率用的回看窗口，现在默认60
交易日)换成30/90天，看结果对这个参数选择敏不敏感。如果30/60/90三个窗口结果差不多，
说明60天不是"刚好调出来的脆弱值"；如果差别很大，说明这个参数本身就是过拟合来源，
需要在README里如实标注。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from backtest import get_weekly_trading_dates
from low_vol_investing import backtest_low_vol_strategy
from run_survivorship_bias_fix_validation import compute_point_in_time_membership


def block_bootstrap_sharpe(rets, block=4, n_boot=3000, periods_per_year=52.0, seed=42):
    rng = np.random.default_rng(seed)
    rets = np.array(rets)
    n = len(rets)
    n_blocks = int(np.ceil(n / block))
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n - block + 1, size=n_blocks)
        sample = np.concatenate([rets[i:i + block] for i in idx])[:n]
        boot.append(sample.mean() / (sample.std() + 1e-9) * np.sqrt(periods_per_year))
    boot = np.array(boot)
    obs = rets.mean() / (rets.std() + 1e-9) * np.sqrt(periods_per_year)
    return obs, float(np.mean(boot <= 0))


def to_weekly(daily_nav, price_df):
    nav_df = pd.DataFrame(daily_nav, columns=["date", "nav"]).set_index("date")["nav"]
    week_pairs = get_weekly_trading_dates(price_df.index)
    rows = []
    for monday, friday in week_pairs:
        try:
            a, b = nav_df.loc[:monday].iloc[-1], nav_df.loc[:friday].iloc[-1]
        except IndexError:
            continue
        rows.append({"date": monday, "ret": b / a - 1})
    return pd.DataFrame(rows).set_index("date")["ret"]


SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
periods = [
    ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl", True),
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl", False),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl", False),
]
WINDOWS = [30, 60, 90]

for label, price_cache_path, is_pit in periods:
    with open(price_cache_path, "rb") as f:
        cache = pickle.load(f)
    price_df, spy_close = cache["c"], cache.get("spy")
    if is_pit:
        membership = compute_point_in_time_membership(price_df.index, list(price_df.columns),
                                                        cache["added_dates"], cache["removal_dates"])
    else:
        membership = {price_df.index[i].strftime("%Y-%m-%d"): list(price_df.columns) for i in range(len(price_df))}

    print(f"\n=== {label} ===")
    for w in WINDOWS:
        _, nav = backtest_low_vol_strategy(price_df, membership, spy_close, rebalance_days=21,
                                            top_n=30, vol_window=w)
        ret = to_weekly(nav, price_df).dropna()
        obs, p0 = block_bootstrap_sharpe(ret.values)
        n_years = len(nav) / 252
        cagr = (nav[-1][1] ** (1 / n_years) - 1) if n_years > 0 else float("nan")
        print(f"  vol_window={w}: Sharpe={obs:.3f}, P(<=0)={p0:.1%}, CAGR={cagr*100:.2f}%")
