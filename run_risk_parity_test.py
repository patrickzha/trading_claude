"""
Cycle30大修大补①(框架改动)：风险平价(risk parity/equal risk contribution)
——跟第140条的最小方差组合优化是相关但不同的框架。最小方差追求"整个
组合方差最小"，容易把权重集中堆到少数几个低波动/低相关的资产上(第144条
量化过：单一票拿到过4倍等权仓位)；风险平价追求"每个资产对组合总风险的
贡献相等"，天然更分散，不会为了压低总方差把仓位大幅堆到某几只票上。
这是这次会话第一次测试风险平价这个框架，用scipy.optimize求解等风险
贡献权重(最小化风险贡献的方差，长仓约束+权重和为1)。
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from value_investing import select_value_portfolio
from stats_utils import true_max_drawdown
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
COV_WINDOW = 60


def risk_parity_weights(returns_window: pd.DataFrame, ridge: float = 0.0001) -> pd.Series:
    """求等风险贡献权重：最小化各资产风险贡献(w_i*(Sigma w)_i)的方差，
    约束权重非负、和为1。用等权做初始值，SLSQP求解，求解失败时退化成
    等权(fail-safe，不让一次数值优化失败拖垮整个回测)。"""
    cov = returns_window.cov().values
    cov = cov + np.eye(len(cov)) * ridge
    n = len(cov)

    def risk_contrib_variance(w):
        # 用相对偏差(rc-target)/target而不是绝对偏差——个股日收益方差量级
        # 本身很小(1e-4~1e-3)，绝对平方误差在这个量级下会小到SLSQP的默认
        # 收敛容差(ftol)判断"已经收敛"，即使风险贡献相对彼此还很不均衡，
        # 这是第一版试点踩到的真实bug(第155条)，这里是修复后的版本。
        port_var = w @ cov @ w
        marginal = cov @ w
        rc = w * marginal
        target = port_var / n
        if abs(target) < 1e-12:
            return 0.0
        return np.sum(((rc - target) / target) ** 2)

    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    result = minimize(risk_contrib_variance, w0, method="SLSQP", bounds=bounds,
                      constraints=constraints, options={"maxiter": 200, "ftol": 1e-10})
    if not result.success or np.any(np.isnan(result.x)):
        return pd.Series(w0, index=returns_window.columns)
    w = np.clip(result.x, 0, None)
    if w.sum() <= 0:
        return pd.Series(w0, index=returns_window.columns)
    return pd.Series(w / w.sum(), index=returns_window.columns)


def backtest_weighted_value(price_df, membership, spy_close, rebalance_days=126, top_n=30,
                             cost_bi=0.003, weighting="equal"):
    index = price_df.index
    n = len(index)
    daily_nav = []
    rebalance_idx = 252 + COV_WINDOW
    running_nav = 1.0
    prev_weights = None

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

        if weighting == "risk_parity":
            hist_window = price_df[picks].loc[:rebalance_date].tail(COV_WINDOW + 1)
            rets_window = hist_window.pct_change().dropna()
            valid_cols = rets_window.columns[rets_window.notna().all()]
            if len(valid_cols) >= 5:
                rp_weights = risk_parity_weights(rets_window[valid_cols])
                weights = pd.Series(0.0, index=picks)
                for t in valid_cols:
                    weights[t] = rp_weights[t]
                weights = weights / weights.sum() if weights.sum() > 0 else pd.Series(1.0 / len(picks), index=picks)
            else:
                weights = pd.Series(1.0 / len(picks), index=picks)
        else:
            weights = pd.Series(1.0 / len(picks), index=picks)

        turnover = 1.0 if prev_weights is None else sum(
            abs(weights.get(t, 0) - prev_weights.get(t, 0)) for t in set(weights.index) | set(prev_weights.index)) / 2
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


def metrics(nav_list):
    nav = np.array([n for _, n in nav_list])
    n_years = len(nav) / 252
    cagr = float((nav[-1] / nav[0]) ** (1 / n_years) - 1) if n_years > 0 else None
    rets = np.diff(nav) / nav[:-1]
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252))
    mdd = true_max_drawdown(nav)
    return cagr, sharpe, mdd


if __name__ == "__main__":
    periods = [
        ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl", True),
        ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl", False),
        ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl", False),
    ]
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
        nav_rp = backtest_weighted_value(price_df, membership, spy_close, weighting="risk_parity")
        e_cagr, e_sharpe, e_mdd = metrics(nav_equal)
        r_cagr, r_sharpe, r_mdd = metrics(nav_rp)
        print(f"\n=== {label} ===")
        print(f"  等权(现有): CAGR={e_cagr*100:.2f}%, Sharpe={e_sharpe:.3f}, MaxDD={e_mdd*100:.2f}%")
        print(f"  风险平价(新框架): CAGR={r_cagr*100:.2f}%, Sharpe={r_sharpe:.3f}, MaxDD={r_mdd*100:.2f}%")
