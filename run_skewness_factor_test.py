"""
Cycle38大修大补①(框架改动，新数据维度——收益率分布高阶矩)：已实现收益率
偏度(skewness)作为选股因子。不同于已验证的"波动率大小"(低波动因子，
第55条，衡量收益率分布的二阶矩/离散程度)，这次测试"分布形状"这个三阶
矩统计量：学术上"betting against skewness"(Bali, Cakici, Whitelaw 2011
等)是一个独立于低波动异象的因子——负偏度(左尾更肥，暴跌集中)的股票
理论上该有正的风险溢价补偿投资者承担的尾部风险，做多低偏度/正偏度
(暴跌风险相对小)的股票理论上该系统性跑赢。纯基于已有价格数据，
不需要新的外部数据源。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from scipy.stats import skew

from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
LOOKBACK = 126
REBALANCE_DAYS = 21
TOP_N = 30


def _lookup_membership(membership, membership_keys, date_str):
    valid = membership.get(date_str)
    if valid is not None:
        return valid
    candidates = [k for k in membership_keys if k <= date_str]
    if not candidates:
        return None
    return membership[max(candidates)]


def select_low_skew_portfolio(price_df, valid_tickers, rebalance_date, top_n=TOP_N):
    rows = []
    for t in valid_tickers:
        if t not in price_df.columns:
            continue
        px = price_df[t].loc[:rebalance_date].dropna().tail(LOOKBACK)
        if len(px) < LOOKBACK - 5:
            continue
        rets = px.pct_change().dropna()
        if len(rets) < LOOKBACK - 10:
            continue
        sk = float(skew(rets))
        rows.append({"ticker": t, "skew": sk})

    if len(rows) < top_n:
        return [r["ticker"] for r in rows]
    df = pd.DataFrame(rows).sort_values("skew", ascending=False)
    return df["ticker"].head(top_n).tolist()


def backtest_skew(price_df, membership, rebalance_days=REBALANCE_DAYS):
    index = price_df.index
    n = len(index)
    membership_keys = sorted(membership.keys())
    daily_nav = []
    rebalance_idx = LOOKBACK + 5
    running_nav = 1.0

    while rebalance_idx + rebalance_days < n:
        rebalance_date = index[rebalance_idx]
        date_str = rebalance_date.strftime("%Y-%m-%d")
        valid = _lookup_membership(membership, membership_keys, date_str)
        if valid is None:
            rebalance_idx += rebalance_days
            continue

        picks = select_low_skew_portfolio(price_df, valid, rebalance_date)
        picks = [p for p in picks if p in price_df.columns]
        if len(picks) < 5:
            rebalance_idx += rebalance_days
            continue

        period_days = index[rebalance_idx:min(rebalance_idx + rebalance_days, n)]
        entry_prices = {t: price_df[t].loc[:rebalance_date].dropna().iloc[-1] for t in picks}
        for d in period_days:
            day_val, total_w = 0.0, 0.0
            for t, p0 in entry_prices.items():
                if d in price_df.index and t in price_df.columns:
                    p = price_df.at[d, t]
                    if pd.notna(p) and p0:
                        day_val += p / p0
                        total_w += 1
            if total_w > 0:
                daily_nav.append((d, running_nav * (day_val / total_w)))
        if daily_nav:
            running_nav = daily_nav[-1][1]
        rebalance_idx += rebalance_days

    return daily_nav


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

        nav = backtest_skew(price_df, membership)
        print(f"\n=== {label} ===")
        if len(nav) < 30:
            print("  样本不足，跳过")
            continue
        cagr, sharpe, mdd = metrics(nav)
        spy_aligned = spy_close.reindex([d for d, _ in nav]).ffill().dropna() if spy_close is not None else None
        spy_cagr = None
        if spy_aligned is not None and len(spy_aligned) > 1:
            n_years = len(nav) / 252
            spy_cagr = float((spy_aligned.iloc[-1] / spy_aligned.iloc[0]) ** (1 / n_years) - 1)
        print(f"  低偏度(正偏度优先)组合: CAGR={cagr*100:.2f}%, Sharpe={sharpe:.3f}, MaxDD={mdd*100:.2f}%")
        if spy_cagr is not None:
            print(f"  同期SPY CAGR={spy_cagr*100:.2f}% (超额={(cagr-spy_cagr)*100:+.2f}%)")


if __name__ == "__main__":
    main()
