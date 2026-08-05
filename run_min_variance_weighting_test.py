"""
Cycle27大修大补①(框架改动)：这次会话的所有组合(动量/价值/低波动/质量)
选出候选之后，仓位分配从来都是简单等权——这次测一个真正不同的框架：
用现代投资组合理论(Markowitz)的最小方差组合权重，代替等权。这不是
"换个选股信号"，是"选股逻辑不变、仓位分配的数学框架换成真正的组合优化"，
属于CLAUDE.md新规则要求的框架改动。

方法：价值腿30只候选不变(选股逻辑跟原来完全一样)，每次调仓时用trailing
60日的日收益协方差矩阵算最小方差权重(w = Σ^-1 1 / (1' Σ^-1 1))，加一个
小的ridge项(对角线加0.0001)防止协方差矩阵病态/接近奇异(30个资产、60天
历史，样本协方差矩阵天然容易不稳定)，负权重截断为0后重新归一化(现实中
不能做空，这是标准的长仓最小方差近似处理，不是严格的约束优化)。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from value_investing import select_value_portfolio
from combo_strategy import true_max_drawdown
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
COV_WINDOW = 60
RIDGE = 0.0001


def min_variance_weights(returns_window: pd.DataFrame) -> pd.Series:
    cov = returns_window.cov().values
    cov = cov + np.eye(len(cov)) * RIDGE
    ones = np.ones(len(cov))
    try:
        inv_cov_ones = np.linalg.solve(cov, ones)
    except np.linalg.LinAlgError:
        return pd.Series(1.0 / len(returns_window.columns), index=returns_window.columns)
    w = inv_cov_ones / inv_cov_ones.sum()
    w = np.clip(w, 0, None)
    if w.sum() <= 0:
        return pd.Series(1.0 / len(returns_window.columns), index=returns_window.columns)
    w = w / w.sum()
    return pd.Series(w, index=returns_window.columns)


def backtest_weighted_value(price_df, membership, spy_close, rebalance_days=126, top_n=30,
                             cost_bi=0.003, weighting="equal"):
    index = price_df.index
    n = len(index)
    daily_nav = []
    rebalance_idx = 252 + COV_WINDOW
    running_nav = 1.0
    prev_weights: pd.Series | None = None

    while rebalance_idx + rebalance_days < n:
        rebalance_date = index[rebalance_idx]
        date_str = rebalance_date.strftime("%Y-%m-%d")
        valid_tickers = membership.get(date_str)
        if valid_tickers is None:
            keys = sorted(membership.keys())
            valid_tickers = membership[max([k for k in keys if k <= date_str], default=keys[0])]

        picks = select_value_portfolio(price_df, valid_tickers, rebalance_date, top_n=top_n)
        picks = [p for p in picks if p in price_df.columns]
        if len(picks) < 5:
            rebalance_idx += rebalance_days
            continue

        hist_window = price_df[picks].loc[:rebalance_date].tail(COV_WINDOW + 1)
        rets_window = hist_window.pct_change().dropna()
        valid_cols = rets_window.columns[rets_window.notna().all()]
        rets_window = rets_window[valid_cols]

        if weighting == "equal" or len(valid_cols) < 5:
            weights = pd.Series(1.0 / len(picks), index=picks)
        else:
            mv_weights = min_variance_weights(rets_window)
            weights = pd.Series(0.0, index=picks)
            for t in valid_cols:
                weights[t] = mv_weights[t]
            if weights.sum() <= 0:
                weights = pd.Series(1.0 / len(picks), index=picks)
            else:
                weights = weights / weights.sum()

        turnover = 0.0
        if prev_weights is not None:
            all_tickers = set(weights.index) | set(prev_weights.index)
            turnover = sum(abs(weights.get(t, 0) - prev_weights.get(t, 0)) for t in all_tickers) / 2
        running_nav *= (1 - turnover * cost_bi)
        prev_weights = weights

        period_days = index[rebalance_idx:min(rebalance_idx + rebalance_days, n)]
        entry_prices = {}
        for t in picks:
            try:
                entry_prices[t] = price_df[t].loc[:rebalance_date].dropna().iloc[-1]
            except IndexError:
                continue
        for d in period_days:
            day_val = 0.0
            total_w = 0.0
            for t, p0 in entry_prices.items():
                if d in price_df.index and t in price_df.columns:
                    p = price_df.at[d, t]
                    if pd.notna(p) and p0:
                        day_val += weights.get(t, 0) * (p / p0)
                        total_w += weights.get(t, 0)
            if total_w > 0:
                daily_nav.append((d, running_nav * (day_val / total_w)))
        if daily_nav:
            running_nav = daily_nav[-1][1]

        rebalance_idx += rebalance_days

    return daily_nav


def metrics(nav_list):
    nav = np.array([n for _, n in nav_list])
    n_years = len(nav) / 252
    cagr = float((nav[-1] / nav[0]) ** (1 / n_years) - 1) if n_years > 0 else None
    rets = np.diff(nav) / nav[:-1]
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252))
    mdd = true_max_drawdown(nav)
    return cagr, sharpe, mdd


periods = [
    ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl", True),
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl", False),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl", False),
]

import json
with open("/Users/zhang/Desktop/trading/sp500_membership_added_dates.json") as f:
    added_dates_raw = json.load(f)
added_dates = {t: pd.Timestamp(d) for t, d in added_dates_raw.items()}

for label, price_cache_path, is_pit in periods:
    with open(price_cache_path, "rb") as f:
        cache = pickle.load(f)
    price_df, spy_close = cache["c"], cache.get("spy")

    if is_pit:
        membership = compute_point_in_time_membership(price_df.index, list(price_df.columns),
                                                        cache["added_dates"], cache["removal_dates"])
    else:
        membership = compute_point_in_time_membership(price_df.index, list(price_df.columns), added_dates, {})

    nav_equal = backtest_weighted_value(price_df, membership, spy_close, weighting="equal")
    nav_mv = backtest_weighted_value(price_df, membership, spy_close, weighting="min_variance")

    e_cagr, e_sharpe, e_mdd = metrics(nav_equal)
    m_cagr, m_sharpe, m_mdd = metrics(nav_mv)

    print(f"\n=== {label} ===")
    print(f"  等权(现有): CAGR={e_cagr*100:.2f}%, Sharpe={e_sharpe:.3f}, MaxDD={e_mdd*100:.2f}%")
    print(f"  最小方差组合优化(新框架): CAGR={m_cagr*100:.2f}%, Sharpe={m_sharpe:.3f}, MaxDD={m_mdd*100:.2f}%")
