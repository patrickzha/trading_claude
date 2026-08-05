"""
Cycle18小修小补：combo_strategy.py现在是三条腿(动量/价值/低波动)硬编码等权
(combined_norm.mean(axis=1))，从来没有测过按风险贡献加权(逆波动率加权，
标准风险平价思路，只用trailing波动率、不看收益率，比"按历史Sharpe加权"更
不容易过拟合)是否更好。这次用已有的三段缓存数据+缓存的动量结果对比等权 vs
逆波动率加权。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from value_investing import backtest_value_strategy
from low_vol_investing import backtest_low_vol_strategy
from combo_strategy import true_max_drawdown
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"

periods = [
    ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl",
     f"{SCRATCH}/default_full_results_cache.pkl", True),
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl",
     f"{SCRATCH}/momentum_2014_2020_results_cache.pkl", False),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl",
     f"{SCRATCH}/momentum_2009_2014_results_cache.pkl", False),
]


def metrics(nav):
    n_years = len(nav) / 252
    cagr = float(nav.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else None
    rets = nav.pct_change().dropna()
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252))
    mdd = true_max_drawdown(nav.values)
    return cagr, sharpe, mdd


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
        membership = {price_df.index[i].strftime("%Y-%m-%d"): list(price_df.columns) for i in range(len(price_df))}

    mom_df = pd.DataFrame(mom_results)[["date", "net_worth"]]
    mom_df["date"] = pd.to_datetime(mom_df["date"])
    mom_series = mom_df.set_index("date")["net_worth"]
    mom_daily = mom_series.reindex(price_df.index, method="ffill").dropna()

    _, val_nav = backtest_value_strategy(price_df, membership, spy_close, rebalance_days=21, top_n=30)
    _, lv_nav = backtest_low_vol_strategy(price_df, membership, spy_close, rebalance_days=21, top_n=30)

    legs = {
        "momentum": mom_daily / mom_daily.iloc[0],
        "value": pd.DataFrame(val_nav, columns=["date", "nav"]).set_index("date")["nav"],
        "low_vol": pd.DataFrame(lv_nav, columns=["date", "nav"]).set_index("date")["nav"],
    }
    combined = pd.DataFrame(legs).dropna()
    combined_norm = combined / combined.iloc[0]

    # 等权(现有)
    equal_nav = combined_norm.mean(axis=1)

    # 逆波动率加权(只用trailing 60日已实现波动率算权重，不看收益率，逐日滚动重算)
    daily_rets = combined_norm.pct_change()
    trailing_vol = daily_rets.rolling(60).std()
    inv_vol = 1.0 / trailing_vol.replace(0, np.nan)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    weights = weights.shift(1)  # 用昨天为止的波动率算今天的权重，不用同一天的信息
    weighted_ret = (daily_rets * weights).sum(axis=1)
    weighted_ret = weighted_ret.iloc[61:]  # 跳过60日warm-up
    invvol_nav = (1 + weighted_ret).cumprod()
    invvol_nav.iloc[0] = 1.0 if len(invvol_nav) else None

    eq_cagr, eq_sharpe, eq_mdd = metrics(equal_nav)
    iv_cagr, iv_sharpe, iv_mdd = metrics(invvol_nav)

    print(f"\n=== {label} ===")
    print(f"  等权(现有): CAGR={eq_cagr*100:.2f}%, Sharpe={eq_sharpe:.3f}, MaxDD={eq_mdd*100:.2f}%")
    print(f"  逆波动率加权(新): CAGR={iv_cagr*100:.2f}%, Sharpe={iv_sharpe:.3f}, MaxDD={iv_mdd*100:.2f}%  (跳过前60日warm-up)")
    print(f"  平均权重: 动量={weights['momentum'].mean():.2f}, 价值={weights['value'].mean():.2f}, 低波动={weights['low_vol'].mean():.2f}")
