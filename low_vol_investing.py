"""
低波动因子策略——第三条独立的腿，测试能不能进一步降低组合的噪声、把
第47/51/52条"3段区间只有1段Fama-French alpha显著"的证据强度往上推一把。
低波动异象(low-volatility anomaly)是另一个有几十年学术证据支持的因子，
机制上跟动量(短期延续)、价值(HML)都不同——赌的是"低波动股票的风险调整
后收益系统性地比高波动股票好，違反CAPM里风险越高收益该越高的预期"。

结构完全参照value_investing.py：月度调仓(按第48条教训，不用半年度)，
选trailing 60日已实现波动率最低的N只票，等权持有。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def select_low_vol_portfolio(price_df: pd.DataFrame, valid_tickers: list, rebalance_date,
                              top_n: int = 30, vol_window: int = 60):
    rows = []
    for t in valid_tickers:
        if t not in price_df.columns:
            continue
        px = price_df[t].loc[:rebalance_date].dropna().tail(vol_window)
        if len(px) < vol_window:
            continue
        rets = px.pct_change().dropna()
        if len(rets) < vol_window - 5:
            continue
        vol = float(rets.std())
        rows.append({"ticker": t, "vol": vol})

    if len(rows) < top_n:
        return [r["ticker"] for r in rows]

    df = pd.DataFrame(rows).sort_values("vol")
    return df["ticker"].head(top_n).tolist()


def backtest_low_vol_strategy(price_df: pd.DataFrame, membership: dict, spy_close: pd.Series | None,
                               rebalance_days: int = 21, top_n: int = 30, vol_window: int = 60,
                               cost_bi: float = 0.003, stop_loss_pct: float | None = None):
    """cost_bi(默认0.3%)：只对换手部分收双边成本，见README"验证结论"第72条
    (之前的实现完全没有扣成本，这是补上的真实缺口)。

    stop_loss_pct(默认None，向后兼容)：个股相对本次调仓买入价的止损
    阈值，见README第234条——把价值腿验证过的止损机制(第230-233条)推广
    到低波动腿，0.10阈值三段区间Sharpe和MaxDD全部同步改善，2014-2020段
    MaxDD改善超过18个百分点，是这次会话泛化能力最强的风控发现。"""
    index = price_df.index
    n = len(index)
    results = []
    daily_nav = []

    rebalance_idx = 100
    running_nav = 1.0
    prev_picks: set[str] = set()
    while rebalance_idx + rebalance_days < n:
        rebalance_date = index[rebalance_idx]
        next_date = index[rebalance_idx + rebalance_days]

        date_str = rebalance_date.strftime("%Y-%m-%d")
        valid_tickers = membership.get(date_str)
        if valid_tickers is None:
            keys = sorted(membership.keys())
            valid_tickers = membership[max([k for k in keys if k <= date_str], default=keys[0])]

        picks = select_low_vol_portfolio(price_df, valid_tickers, rebalance_date, top_n=top_n, vol_window=vol_window)

        picks_set = set(picks)
        turnover = len(picks_set - prev_picks) / len(picks_set) if picks_set else 0.0
        running_nav *= (1 - turnover * cost_bi)
        prev_picks = picks_set

        period_days = index[rebalance_idx:min(rebalance_idx + rebalance_days, n)]
        entry_prices = {}
        for t in picks:
            try:
                entry_prices[t] = price_df[t].loc[:rebalance_date].dropna().iloc[-1]
            except IndexError:
                continue
        stopped_out: dict[str, float] = {}
        for d in period_days:
            day_navs = []
            for t, p0 in entry_prices.items():
                if d in price_df.index and t in price_df.columns:
                    if t in stopped_out:
                        day_navs.append(stopped_out[t])
                        continue
                    p = price_df.at[d, t]
                    if pd.notna(p) and p0:
                        factor = p / p0
                        if stop_loss_pct is not None and factor <= (1 - stop_loss_pct):
                            factor = 1 - stop_loss_pct
                            stopped_out[t] = factor
                        day_navs.append(factor)
            if day_navs:
                daily_nav.append((d, running_nav * float(np.mean(day_navs))))
        if daily_nav:
            running_nav = daily_nav[-1][1]

        rets = []
        for t in picks:
            try:
                p0 = price_df[t].loc[:rebalance_date].dropna().iloc[-1]
                p1 = price_df[t].loc[:next_date].dropna().iloc[-1]
                rets.append(p1 / p0 - 1)
            except (IndexError, KeyError):
                continue

        spy_ret = None
        if spy_close is not None:
            try:
                s0 = spy_close.loc[:rebalance_date].dropna().iloc[-1]
                s1 = spy_close.loc[:next_date].dropna().iloc[-1]
                spy_ret = float(s1 / s0 - 1)
            except (IndexError, KeyError):
                pass

        results.append({"rebalance_date": rebalance_date.strftime("%Y-%m-%d"), "n_picks": len(picks),
                        "period_ret": float(np.mean(rets)) if rets else 0.0, "spy_ret": spy_ret})
        rebalance_idx += rebalance_days

    return results, daily_nav
