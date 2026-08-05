"""
Cycle7：经典12-1月动量因子(UMD)——第55条发现低波动腿跟价值腿相关性太高
(0.51-0.74)，不是真正独立的第三条腿，这里换一个跟价值因子在学术上本来
就是负相关/低相关的经典信号：过去12个月收益率(剔除最近1个月，避免短期
反转噪声)排名最高的股票，月度调仓等权持有——这是最标准的学术UMD动量
因子实现，用纯价格数据，不依赖XGBoost模型，机制上跟现有的"周频ML动量
策略"、"低PE/PB价值"、"低波动"都不同，是真正意义上第四种独立构造方式。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def select_momentum_portfolio(price_df: pd.DataFrame, valid_tickers: list, rebalance_date,
                               top_n: int = 30, lookback: int = 252, skip: int = 21):
    rows = []
    for t in valid_tickers:
        if t not in price_df.columns:
            continue
        px = price_df[t].loc[:rebalance_date].dropna()
        if len(px) < lookback + skip:
            continue
        p_end = px.iloc[-skip - 1] if skip > 0 else px.iloc[-1]
        p_start = px.iloc[-lookback - skip - 1] if len(px) > lookback + skip else px.iloc[0]
        if p_start <= 0:
            continue
        mom_12_1 = p_end / p_start - 1
        rows.append({"ticker": t, "mom": mom_12_1})

    if len(rows) < top_n:
        return [r["ticker"] for r in rows]

    df = pd.DataFrame(rows).sort_values("mom", ascending=False)
    return df["ticker"].head(top_n).tolist()


def backtest_classic_momentum(price_df: pd.DataFrame, membership: dict, spy_close: pd.Series | None,
                               rebalance_days: int = 21, top_n: int = 30, lookback: int = 252, skip: int = 21,
                               stop_loss_pct: float | None = None):
    """stop_loss_pct(默认None，向后兼容不止损)：个股相对本次调仓买入价的
    止损阈值，实现方式跟`low_vol_investing.backtest_low_vol_strategy`/
    `value_investing.backtest_value_strategy`完全一致(跌破买入价阈值后
    该票本持有期内净值贡献冻结在`1-stop_loss_pct`，不再跟随实际价格，
    直到下次调仓重新按新买入价计算)——见README第230+条，止损已经在价值
    腿、低波动腿上反复验证是稳健的正贡献，这次给动量腿(经典UMD实现，不是
    周频ML动量腿)补上同样的机制，方便第490条"止损在2004-2009真正危机窗口
    是否依然有效"这个检验推广到动量腿。"""
    index = price_df.index
    n = len(index)
    results = []
    daily_nav = []

    rebalance_idx = lookback + skip + 5
    running_nav = 1.0
    while rebalance_idx + rebalance_days < n:
        rebalance_date = index[rebalance_idx]
        next_date = index[rebalance_idx + rebalance_days]

        date_str = rebalance_date.strftime("%Y-%m-%d")
        valid_tickers = membership.get(date_str)
        if valid_tickers is None:
            keys = sorted(membership.keys())
            valid_tickers = membership[max([k for k in keys if k <= date_str], default=keys[0])]

        picks = select_momentum_portfolio(price_df, valid_tickers, rebalance_date,
                                          top_n=top_n, lookback=lookback, skip=skip)

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
