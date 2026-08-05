"""
Cycle17大修大补①：整合多因子选股(single composite score)，代替现在的
"三个独立sleeve等权平均净值"结构。现在的三元组合是三个完全独立的30只票
组合各自跑、净值再平均；这次测试真正意义上不同的组合构建方法——在候选
池里对每只票同时算价值z-score和低波动z-score，加权合成一个综合分，
直接选综合分最高的30只票，只买一个组合，不是三个组合的净值平均。

理论上的区别：sleeve方式即使两个因子选出的票完全不重叠，风险暴露也是
两个独立组合的平均；integrated方式要求单只票同时在两个维度都不差，
选出来的是"价值+低波动"交集特征更强的票，理论上集中度更高、因子暴露
更纯粹，但候选池可能更小(需要两个维度都过筛选)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from us_stock_screener import _compute_fundamental_features


def select_integrated_portfolio(price_df: pd.DataFrame, valid_tickers: list, rebalance_date,
                                 top_n: int = 30, max_pe: float = 40.0, vol_window: int = 60,
                                 value_weight: float = 0.5):
    rows = []
    for t in valid_tickers:
        if t not in price_df.columns:
            continue
        px_series = price_df[t].loc[:rebalance_date].dropna()
        if len(px_series) < vol_window:
            continue
        curr_price = px_series.iloc[-1]

        feat = _compute_fundamental_features(t, rebalance_date, curr_price)
        pe, pb, roe = feat["pe_ratio"], feat["pb_ratio"], feat["roe"]
        if not (pe is not None and not np.isnan(pe) and 0 < pe < max_pe):
            continue
        if not (pb is not None and not np.isnan(pb) and pb > 0):
            continue
        if roe is None or np.isnan(roe):
            continue

        vol_px = px_series.tail(vol_window)
        rets = vol_px.pct_change().dropna()
        if len(rets) < vol_window - 5:
            continue
        vol = float(rets.std())

        rows.append({"ticker": t, "pe": pe, "pb": pb, "roe": roe, "vol": vol})

    if len(rows) < top_n:
        return [r["ticker"] for r in rows]

    df = pd.DataFrame(rows)
    df["rank_pe"] = df["pe"].rank(ascending=True, pct=True)
    df["rank_pb"] = df["pb"].rank(ascending=True, pct=True)
    df["rank_roe"] = df["roe"].rank(ascending=False, pct=True)
    df["value_z"] = (df["rank_pe"] + df["rank_pb"] + df["rank_roe"]) / 3
    df["lowvol_z"] = df["vol"].rank(ascending=True, pct=True)
    df["composite"] = value_weight * df["value_z"] + (1 - value_weight) * df["lowvol_z"]
    df = df.sort_values("composite")
    return df["ticker"].head(top_n).tolist()


def backtest_integrated_strategy(price_df: pd.DataFrame, membership: dict, spy_close: pd.Series | None,
                                  rebalance_days: int = 21, top_n: int = 30, cost_bi: float = 0.003,
                                  value_weight: float = 0.5):
    index = price_df.index
    n = len(index)
    daily_nav = []
    results = []

    rebalance_idx = 252
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

        picks = select_integrated_portfolio(price_df, valid_tickers, rebalance_date,
                                            top_n=top_n, value_weight=value_weight)

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
        for d in period_days:
            day_navs = []
            for t, p0 in entry_prices.items():
                if d in price_df.index and t in price_df.columns:
                    p = price_df.at[d, t]
                    if pd.notna(p) and p0:
                        day_navs.append(p / p0)
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
        results.append({"rebalance_date": date_str, "n_picks": len(picks),
                        "period_ret": float(np.mean(rets)) if rets else 0.0})
        rebalance_idx += rebalance_days

    return results, daily_nav
