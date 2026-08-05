"""
Cycle36大修大补②(框架改动，新配置粒度——板块轮动)：用11个GICS板块ETF
(XLK科技/XLF金融/XLV医疗/XLE能源/XLI工业/XLY可选消费/XLP必需消费/
XLU公用事业/XLB原材料/XLRE房地产/XLC通信服务)的trailing 3个月动量
做月度轮动，持有动量最强的3个板块等权。这是这次会话第一次在"板块"
这个粒度上做动态配置——此前所有测试都在"个股"(选哪些股票)或"资产
类别"(股票/债券/黄金/商品配比)这两个粒度操作，板块轮动是介于两者
之间的第三个粒度层级，机制依据是行业轮动/板块动量在学术界也有一定
文献支持(不同经济周期阶段不同板块表现系统性不同)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
TOP_N = 3
MOM_WINDOW = 63
REBALANCE_DAYS = 21

PERIODS = {
    "2014-2020": ("2014-06-02", "2020-12-31"),
    "2022-2026": ("2021-08-02", "2026-07-30"),
}


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


def backtest_sector_rotation(price_df, rebalance_days=REBALANCE_DAYS):
    index = price_df.index
    n = len(index)
    daily_nav = []
    rebalance_idx = MOM_WINDOW + 5
    running_nav = 1.0

    while rebalance_idx + rebalance_days < n:
        rebalance_date = index[rebalance_idx]
        mom_scores = {}
        for t in SECTOR_ETFS:
            px = price_df[t].loc[:rebalance_date].dropna()
            if len(px) < MOM_WINDOW + 5:
                continue
            mom_scores[t] = px.iloc[-1] / px.iloc[-MOM_WINDOW] - 1
        if len(mom_scores) < TOP_N:
            rebalance_idx += rebalance_days
            continue

        picks = sorted(mom_scores, key=mom_scores.get, reverse=True)[:TOP_N]
        period_days = index[rebalance_idx:min(rebalance_idx + rebalance_days, n)]
        entry_prices = {t: price_df[t].loc[:rebalance_date].dropna().iloc[-1] for t in picks}
        for d in period_days:
            day_val, total_w = 0.0, 0.0
            for t, p0 in entry_prices.items():
                if d in price_df.index:
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


def main():
    print("拉取11个GICS板块ETF+SPY数据...")
    tickers = SECTOR_ETFS + ["SPY"]
    data = yf.download(tickers, start="2013-01-01", end="2026-08-01",
                        progress=False, group_by="ticker", auto_adjust=True)
    closes = {t: data[t]["Close"].dropna() for t in tickers}
    price_df = pd.DataFrame(closes)
    for t in tickers:
        print(f"  {t} 数据起点: {closes[t].index[0].date()}")

    for label, (start, end) in PERIODS.items():
        sub = price_df.loc[start:end].dropna()
        if len(sub) < 100:
            print(f"\n=== {label} ===\n  数据不足，跳过")
            continue
        print(f"\n=== {label} (n={len(sub)}天) ===")
        nav_rotation = backtest_sector_rotation(sub)
        if len(nav_rotation) < 30:
            print("  样本不足，跳过")
            continue
        r_cagr, r_sharpe, r_mdd = metrics(nav_rotation)

        spy_nav = sub["SPY"] / sub["SPY"].iloc[0]
        n_years = len(spy_nav) / 252
        spy_cagr = float(spy_nav.iloc[-1] ** (1 / n_years) - 1)
        spy_mdd = true_mdd(spy_nav.values)
        spy_rets = spy_nav.pct_change().dropna()
        spy_sharpe = float(spy_rets.mean() / (spy_rets.std() + 1e-9) * np.sqrt(252))

        print(f"  板块轮动(top3): CAGR={r_cagr*100:.2f}%, Sharpe={r_sharpe:.3f}, MaxDD={r_mdd*100:.2f}%")
        print(f"  纯SPY买入持有: CAGR={spy_cagr*100:.2f}%, Sharpe={spy_sharpe:.3f}, MaxDD={spy_mdd*100:.2f}%")


if __name__ == "__main__":
    main()
