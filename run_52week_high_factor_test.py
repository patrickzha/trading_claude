"""
Cycle41大修大补①(框架改动，新数据维度——行为锚定效应)：52周高点距离
(52-week high proximity)作为选股因子。George & Hwang (2004)等文献里
一个独立于经典价格动量的行为金融异象：投资者会把当前价格跟"过去52周
最高价"做心理锚定，接近52周高点的股票往往被低估其后续上涨潜力(投资
者不愿意在"接近历史新高"的位置继续加仓，直到基本面驱动的新信息迫使
价格突破前高)，构造方式(价格/52周最高价的比值)本身跟经典动量(过去
N月的收益率)统计上不完全相同，测试能否提供独立的边际信息。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
LOOKBACK_52W = 252
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


def select_near_52w_high_portfolio(price_df, valid_tickers, rebalance_date, top_n=TOP_N):
    rows = []
    for t in valid_tickers:
        if t not in price_df.columns:
            continue
        px = price_df[t].loc[:rebalance_date].dropna().tail(LOOKBACK_52W)
        if len(px) < LOOKBACK_52W - 20:
            continue
        curr_price = px.iloc[-1]
        high_52w = px.max()
        if high_52w <= 0:
            continue
        proximity = curr_price / high_52w  # 越接近1，越接近52周高点
        rows.append({"ticker": t, "proximity": proximity})

    if len(rows) < top_n:
        return [r["ticker"] for r in rows]
    df = pd.DataFrame(rows).sort_values("proximity", ascending=False)
    return df["ticker"].head(top_n).tolist()


def backtest_52w_high(price_df, membership, rebalance_days=REBALANCE_DAYS):
    index = price_df.index
    n = len(index)
    membership_keys = sorted(membership.keys())
    daily_nav = []
    rebalance_idx = LOOKBACK_52W + 5
    running_nav = 1.0

    while rebalance_idx + rebalance_days < n:
        rebalance_date = index[rebalance_idx]
        date_str = rebalance_date.strftime("%Y-%m-%d")
        valid = _lookup_membership(membership, membership_keys, date_str)
        if valid is None:
            rebalance_idx += rebalance_days
            continue

        picks = select_near_52w_high_portfolio(price_df, valid, rebalance_date)
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

        nav = backtest_52w_high(price_df, membership)
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
        print(f"  52周高点邻近组合: CAGR={cagr*100:.2f}%, Sharpe={sharpe:.3f}, MaxDD={mdd*100:.2f}%")
        if spy_cagr is not None:
            print(f"  同期SPY CAGR={spy_cagr*100:.2f}% (超额={(cagr-spy_cagr)*100:+.2f}%)")


if __name__ == "__main__":
    main()
