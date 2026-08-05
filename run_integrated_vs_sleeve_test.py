"""
Cycle17大修大补①验证：整合多因子(single composite) vs 现有的sleeve式
(三个独立组合平均)，三段区间对比。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from backtest import get_weekly_trading_dates
from value_investing import backtest_value_strategy
from low_vol_investing import backtest_low_vol_strategy
from integrated_multi_factor import backtest_integrated_strategy
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

for label, price_cache_path, is_pit in periods:
    with open(price_cache_path, "rb") as f:
        cache = pickle.load(f)
    price_df, spy_close = cache["c"], cache.get("spy")
    if is_pit:
        membership = compute_point_in_time_membership(price_df.index, list(price_df.columns),
                                                        cache["added_dates"], cache["removal_dates"])
    else:
        membership = {price_df.index[i].strftime("%Y-%m-%d"): list(price_df.columns) for i in range(len(price_df))}

    # sleeve方式(现有)
    _, val_nav = backtest_value_strategy(price_df, membership, spy_close, rebalance_days=21, top_n=30)
    _, lv_nav = backtest_low_vol_strategy(price_df, membership, spy_close, rebalance_days=21, top_n=30)
    val_ret = to_weekly(val_nav, price_df)
    lv_ret = to_weekly(lv_nav, price_df)
    sleeve_ret = pd.DataFrame({"val": val_ret, "lv": lv_ret}).dropna().mean(axis=1)

    # integrated方式(新)
    _, int_nav = backtest_integrated_strategy(price_df, membership, spy_close, rebalance_days=21, top_n=30)
    int_ret = to_weekly(int_nav, price_df)

    sleeve_obs, sleeve_p0 = block_bootstrap_sharpe(sleeve_ret.values)
    int_obs, int_p0 = block_bootstrap_sharpe(int_ret.dropna().values)

    print(f"\n=== {label} ===")
    print(f"  sleeve方式(现有): Sharpe={sleeve_obs:.3f}, P(<=0)={sleeve_p0:.1%}")
    print(f"  integrated方式(新): Sharpe={int_obs:.3f}, P(<=0)={int_p0:.1%}")
