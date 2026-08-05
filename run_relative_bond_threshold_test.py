"""
Cycle17小修小补：full_system.py现在用固定绝对值(60日10年期国债收益率变化>0.3
百分点)触发债券减仓；这个绝对值在不同利率环境下含义不一样(2009-2014超低利率
时代的0.3pp波动跟2022年加息周期的0.3pp波动，统计意义完全不同)。这次测试改成
相对(百分位)版本：用trailing 2年(504交易日)的trend滚动分布，当前trend超过该
窗口内80分位数时才触发减仓(窗口只用过去数据，不看未来，没有前视偏差)，跟固定
0.3pp版本对比Sharpe。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import yfinance as yf

from combo_strategy import run_combo_strategy
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
BOND_WEIGHT = 0.4
BOND_WEIGHT_HIKE = 0.1
TREND_WINDOW = 60
PCTL_LOOKBACK = 504
PCTL_THRESHOLD = 0.80


def sharpe_of(full_ret):
    if len(full_ret) < 5:
        return float("nan")
    return float(full_ret.mean() / (full_ret.std() + 1e-9) * np.sqrt(252))


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
        membership = {price_df.index[i].strftime("%Y-%m-%d"): list(price_df.columns) for i in range(len(price_df))}

    equity_report = run_combo_strategy(price_df, membership, spy_close, mom_results)
    equity_nav = equity_report["combo_nav"]
    equity_ret = equity_nav.pct_change()

    start = equity_nav.index[0].strftime("%Y-%m-%d")
    end = equity_nav.index[-1].strftime("%Y-%m-%d")
    data = yf.download(["IEF", "^TNX"], start=start, end=end, progress=False,
                       group_by="ticker", auto_adjust=True)
    bond_close = data["IEF"]["Close"].dropna()
    tnx = data["^TNX"]["Close"].dropna()
    bond_ret = bond_close.reindex(equity_nav.index).ffill().pct_change()
    tnx_trend = tnx.reindex(equity_nav.index).ffill().diff(TREND_WINDOW)

    # 固定绝对值版(现有)
    bw_fixed = pd.Series(BOND_WEIGHT, index=equity_ret.index)
    bw_fixed[tnx_trend.reindex(equity_ret.index) > 0.3] = BOND_WEIGHT_HIKE
    fixed_ret = ((1 - bw_fixed) * equity_ret + bw_fixed * bond_ret).dropna()

    # 相对(百分位)版(新)：只用trailing PCTL_LOOKBACK天的历史trend分布算百分位，不看未来
    trend_full = tnx.diff(TREND_WINDOW).dropna()
    rolling_pctl = trend_full.rolling(PCTL_LOOKBACK).apply(
        lambda x: (x.iloc[:-1] < x.iloc[-1]).mean() if len(x) > 1 else np.nan, raw=False)
    rolling_pctl = rolling_pctl.reindex(equity_ret.index).ffill()
    bw_rel = pd.Series(BOND_WEIGHT, index=equity_ret.index)
    bw_rel[rolling_pctl > PCTL_THRESHOLD] = BOND_WEIGHT_HIKE
    rel_ret = ((1 - bw_rel) * equity_ret + bw_rel * bond_ret).dropna()

    n_valid_pctl = rolling_pctl.notna().sum()
    print(f"\n=== {label} ===")
    print(f"  固定0.3pp版 Sharpe = {sharpe_of(fixed_ret):.3f}  (触发减仓天数占比: {(bw_fixed==BOND_WEIGHT_HIKE).mean():.1%})")
    print(f"  相对(80分位)版 Sharpe = {sharpe_of(rel_ret):.3f}  (触发减仓天数占比: {(bw_rel==BOND_WEIGHT_HIKE).mean():.1%}, 有效百分位天数占比: {n_valid_pctl/len(rolling_pctl):.1%})")
