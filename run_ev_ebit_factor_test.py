"""
Cycle62大修大补②(续)：EV/EBIT因子(见ev_ebit_investing.py)标准化测试——
(1)独立跑一遍半年度调仓回测(跟第38条原始价值因子同等配置：126天调仓、
top_n=30、等权、不含止损/HRP等后续优化，做公平的"原始因子质量"对比)，
(2)检验EV/EBIT选股结果跟现有P/E-P/B-ROE价值因子的重叠度/相关性，用
第134/137条同样的方法论判断是不是真正独立的信号，还是"换个壳子"。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from ev_ebit_investing import select_ev_ebit_portfolio, compute_ev_ebit
from value_investing import select_value_portfolio
from run_survivorship_bias_fix_validation import compute_point_in_time_membership, CACHE_PATH

REBALANCE_DAYS = 126
TOP_N = 30


def backtest_simple(price_df, membership, spy_close, select_fn, **select_kwargs):
    index = price_df.index
    n = len(index)
    rebalance_idx = 252
    daily_nav = []
    running_nav = 1.0
    while rebalance_idx + REBALANCE_DAYS < n:
        rebalance_date = index[rebalance_idx]
        date_str = rebalance_date.strftime("%Y-%m-%d")
        valid_tickers = membership.get(date_str)
        if valid_tickers is None:
            keys = sorted(membership.keys())
            valid_tickers = membership[max([k for k in keys if k <= date_str], default=keys[0])]

        picks = select_fn(price_df, valid_tickers, rebalance_date, top_n=TOP_N, **select_kwargs)
        picks = [p for p in picks if p in price_df.columns]
        period_days = index[rebalance_idx:min(rebalance_idx + REBALANCE_DAYS, n)]
        entry_prices = {t: price_df[t].loc[:rebalance_date].dropna().iloc[-1] for t in picks
                         if len(price_df[t].loc[:rebalance_date].dropna()) > 0}
        for d in period_days:
            vals = []
            for t, p0 in entry_prices.items():
                if d in price_df.index and t in price_df.columns:
                    p = price_df.at[d, t]
                    if pd.notna(p) and p0:
                        vals.append(p / p0)
            if vals:
                daily_nav.append((d, running_nav * np.mean(vals)))
        if daily_nav:
            running_nav = daily_nav[-1][1]
        rebalance_idx += REBALANCE_DAYS

    nav_series = pd.DataFrame(daily_nav, columns=["date", "nav"]).set_index("date")["nav"]
    rets = nav_series.pct_change().dropna()
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252))
    n_years = len(nav_series) / 252
    cagr = float(nav_series.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else None
    peak = nav_series.cummax()
    mdd = float(((nav_series - peak) / peak).min())
    return {"Sharpe": sharpe, "CAGR": cagr, "MaxDD": mdd, "n_days": len(nav_series)}


def main():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    price_df, spy_close = cache["c"], cache.get("spy")
    membership = compute_point_in_time_membership(
        price_df.index, list(price_df.columns), cache["added_dates"], cache["removal_dates"])

    print(f"{'='*70}\n (1) EV/EBIT因子 vs 原始P/E-P/B-ROE价值因子，公平对比(126天调仓/30只/等权)\n{'='*70}")
    r_ev = backtest_simple(price_df, membership, spy_close, select_ev_ebit_portfolio)
    r_val = backtest_simple(price_df, membership, spy_close, select_value_portfolio)
    print(f"  EV/EBIT因子: Sharpe={r_ev['Sharpe']:.3f}, CAGR={r_ev['CAGR']*100:.2f}%, MaxDD={r_ev['MaxDD']*100:.2f}%, n_days={r_ev['n_days']}")
    print(f"  原始价值因子(对照): Sharpe={r_val['Sharpe']:.3f}, CAGR={r_val['CAGR']*100:.2f}%, MaxDD={r_val['MaxDD']*100:.2f}%, n_days={r_val['n_days']}")

    print(f"\n{'='*70}\n (2) EV/EBIT选股结果跟现有价值因子的重叠度/相关性检验(第134/137条同方法论)\n{'='*70}")
    index = price_df.index
    sample_dates = [index[252 + i * 200] for i in range(1, 5) if 252 + i * 200 < len(index)]
    membership_keys = sorted(membership.keys())
    overlaps = []
    for d in sample_dates:
        date_str = d.strftime("%Y-%m-%d")
        valid = membership.get(date_str)
        if valid is None:
            valid = membership[max([k for k in membership_keys if k <= date_str], default=membership_keys[0])]
        ev_picks = set(select_ev_ebit_portfolio(price_df, valid, d, top_n=TOP_N))
        val_picks = set(select_value_portfolio(price_df, valid, d, top_n=TOP_N))
        overlap = len(ev_picks & val_picks)
        overlaps.append(overlap)
        print(f"  {date_str}: EV/EBIT选{len(ev_picks)}只, 价值因子选{len(val_picks)}只, 重叠{overlap}只")

    print(f"\n  平均重叠比例: {np.mean(overlaps)/TOP_N*100:.1f}%")

    # 原始指标层面的相关系数(不是组合净值)，同一决策日横截面
    d = sample_dates[-1]
    date_str = d.strftime("%Y-%m-%d")
    valid = membership.get(date_str) or membership[max([k for k in membership_keys if k <= date_str], default=membership_keys[0])]
    rows = []
    from us_stock_screener import _compute_fundamental_features
    for t in valid:
        if t not in price_df.columns:
            continue
        px = price_df[t].loc[:d].dropna()
        if len(px) == 0:
            continue
        p = px.iloc[-1]
        ev_ebit = compute_ev_ebit(t, d, p)
        feat = _compute_fundamental_features(t, d, p)
        pe = feat["pe_ratio"]
        if ev_ebit is not None and pe is not None and not np.isnan(pe):
            rows.append({"ev_ebit": ev_ebit, "pe": pe})
    if len(rows) > 10:
        df = pd.DataFrame(rows)
        corr = df["ev_ebit"].corr(df["pe"])
        print(f"\n  原始指标层面相关系数(EV/EBIT vs P/E, {date_str}, n={len(df)}): {corr:.4f}")

    print("\n全部完成。")


if __name__ == "__main__":
    main()
