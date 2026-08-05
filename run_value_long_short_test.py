"""
Cycle36大修大补①(框架改动，新机制——市场中性多空对冲)：用已验证的价值
因子排名(第38条，PE/PB/ROE复合排名，长周期持有)构造多空对冲组合——
做多价值分最高的一档(便宜+优质)、做空价值分最低的一档(贵+差)，对冲掉
市场beta。这跟第37条的配对交易(distance method，无方向性、随机配对
统计套利)是完全不同的机制：这次是"给一个已知有效的多头信号加一条对冲
空头腿"，测试市场中性化之后，风险调整后收益(尤其是在2008/2022这类
系统性下跌区间)会不会比纯多头版本更好——多空组合理论上牺牲一部分
牛市收益，换取熊市/系统性下跌时的保护，是分散化框架之外，第一次真正
测试"对冲市场beta"这个机制本身的价值。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from us_stock_screener import _compute_fundamental_features
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
REBALANCE_DAYS = 126
TOP_N = 20


def _lookup_membership(membership, membership_keys, date_str):
    valid = membership.get(date_str)
    if valid is not None:
        return valid
    candidates = [k for k in membership_keys if k <= date_str]
    if not candidates:
        return None
    return membership[max(candidates)]


def compute_value_ranks(price_df, valid_tickers, rebalance_date, max_pe=60.0):
    rows = []
    for t in valid_tickers:
        if t not in price_df.columns:
            continue
        px_series = price_df[t].loc[:rebalance_date].dropna()
        if len(px_series) == 0:
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
        rows.append({"ticker": t, "pe": pe, "pb": pb, "roe": roe})

    if len(rows) < TOP_N * 2:
        return [], []

    df = pd.DataFrame(rows)
    df["rank_pe"] = df["pe"].rank(ascending=True)
    df["rank_pb"] = df["pb"].rank(ascending=True)
    df["rank_roe"] = df["roe"].rank(ascending=False)
    df["composite"] = (df["rank_pe"] + df["rank_pb"] + df["rank_roe"]) / 3
    df = df.sort_values("composite")

    long_picks = df["ticker"].head(TOP_N).tolist()
    short_picks = df["ticker"].tail(TOP_N).tolist()
    return long_picks, short_picks


def backtest_long_short(price_df, membership, rebalance_days=REBALANCE_DAYS):
    index = price_df.index
    n = len(index)
    membership_keys = sorted(membership.keys())
    daily_nav_long_only = []
    daily_nav_long_short = []
    rebalance_idx = 252 + 5
    running_nav_lo = 1.0
    running_nav_ls = 1.0

    while rebalance_idx + rebalance_days < n:
        rebalance_date = index[rebalance_idx]
        date_str = rebalance_date.strftime("%Y-%m-%d")
        valid = _lookup_membership(membership, membership_keys, date_str)
        if valid is None:
            rebalance_idx += rebalance_days
            continue

        long_picks, short_picks = compute_value_ranks(price_df, valid, rebalance_date)
        long_picks = [p for p in long_picks if p in price_df.columns]
        short_picks = [p for p in short_picks if p in price_df.columns]
        if len(long_picks) < 5 or len(short_picks) < 5:
            rebalance_idx += rebalance_days
            continue

        period_days = index[rebalance_idx:min(rebalance_idx + rebalance_days, n)]
        long_entry = {t: price_df[t].loc[:rebalance_date].dropna().iloc[-1] for t in long_picks
                      if len(price_df[t].loc[:rebalance_date].dropna()) > 0}
        short_entry = {t: price_df[t].loc[:rebalance_date].dropna().iloc[-1] for t in short_picks
                       if len(price_df[t].loc[:rebalance_date].dropna()) > 0}

        for d in period_days:
            long_val, long_w = 0.0, 0.0
            for t, p0 in long_entry.items():
                if d in price_df.index and t in price_df.columns:
                    p = price_df.at[d, t]
                    if pd.notna(p) and p0:
                        long_val += p / p0
                        long_w += 1
            short_val, short_w = 0.0, 0.0
            for t, p0 in short_entry.items():
                if d in price_df.index and t in price_df.columns:
                    p = price_df.at[d, t]
                    if pd.notna(p) and p0:
                        short_val += p / p0
                        short_w += 1
            if long_w > 0:
                daily_nav_long_only.append((d, running_nav_lo * (long_val / long_w)))
            if long_w > 0 and short_w > 0:
                long_ret_factor = long_val / long_w
                short_ret_factor = short_val / short_w
                ls_factor = 1.0 + (long_ret_factor - 1.0) - (short_ret_factor - 1.0)
                daily_nav_long_short.append((d, running_nav_ls * ls_factor))

        if daily_nav_long_only:
            running_nav_lo = daily_nav_long_only[-1][1]
        if daily_nav_long_short:
            running_nav_ls = daily_nav_long_short[-1][1]
        rebalance_idx += rebalance_days

    return daily_nav_long_only, daily_nav_long_short


def true_mdd(nav):
    nav = np.array(nav)
    peak = np.maximum.accumulate(nav)
    return float(np.min((nav - peak) / peak))


def metrics(nav_list):
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

        nav_lo, nav_ls = backtest_long_short(price_df, membership)
        print(f"\n=== {label} ===")
        if len(nav_lo) >= 30:
            lo_cagr, lo_sharpe, lo_mdd = metrics(nav_lo)
            print(f"  纯多头(只做多价值分最高档): CAGR={lo_cagr*100:.2f}%, Sharpe={lo_sharpe:.3f}, MaxDD={lo_mdd*100:.2f}%")
        if len(nav_ls) >= 30:
            ls_cagr, ls_sharpe, ls_mdd = metrics(nav_ls)
            print(f"  多空对冲(多最高档空最低档): CAGR={ls_cagr*100:.2f}%, Sharpe={ls_sharpe:.3f}, MaxDD={ls_mdd*100:.2f}%")


if __name__ == "__main__":
    main()
