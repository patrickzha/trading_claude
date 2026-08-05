"""
国际市场ETF轮动策略测试——拉2004-2026全区间数据(一次性拉够长，覆盖
这次会话测过的全部四段窗口：2004-2009危机段、2009-2014、2014-2020、
2022-2026)，测试:
1. 国际轮动策略本身的表现(CAGR/Sharpe/真实MaxDD)
2. 分区间跟美股标普500(SPY)的相关性——理论上应该比同一市场内部的
   因子分散更低，但2008年这种全球系统性危机可能仍然会让相关性飙升
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import yfinance as yf

from international_rotation import fetch_international_etf_data, backtest_international_rotation, ETF_UNIVERSE

CACHE_PATH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/international_etf_cache.pkl"

PERIODS = {
    "2004-2009(含危机)": ("2004-06-01", "2009-06-02"),
    "2009-2014": ("2009-06-01", "2014-06-02"),
    "2014-2020": ("2014-06-02", "2020-12-31"),
    "2022-2026": ("2021-08-02", "2026-07-30"),
}


def true_mdd(nav):
    nav = np.array(nav)
    peak = np.maximum.accumulate(nav)
    return float(np.min((nav - peak) / peak))


def main():
    print(f"拉取 {len(ETF_UNIVERSE)} 只国际ETF 2004-2026 全区间数据...")
    full_df = fetch_international_etf_data("2004-01-01", "2026-08-01", CACHE_PATH)
    print(f"数据范围: {full_df.index[0].date()} ~ {full_df.index[-1].date()}")

    spy = yf.download("SPY", start="2004-01-01", end="2026-08-01", progress=False, auto_adjust=True)
    if isinstance(spy.columns, pd.MultiIndex):
        spy_close = spy.xs("Close", axis=1, level=0).squeeze(axis=1)
    else:
        spy_close = spy["Close"] if "Close" in spy.columns else spy.iloc[:, 0]

    for label, (start, end) in PERIODS.items():
        sub_df = full_df.loc[start:end].dropna(axis=1, thresh=int(0.9 * len(full_df.loc[start:end])))
        if len(sub_df.columns) < 4:
            print(f"\n=== {label} ===\n  可用ETF数量不足({len(sub_df.columns)}只)，跳过")
            continue
        daily_nav = backtest_international_rotation(sub_df, rebalance_days=21, top_n=4)
        if len(daily_nav) < 30:
            print(f"\n=== {label} ===\n  有效交易日不足，跳过")
            continue
        nav_df = pd.DataFrame(daily_nav, columns=["date", "nav"]).set_index("date")["nav"]

        n_years = len(nav_df) / 252
        cagr = float(nav_df.iloc[-1] ** (1 / n_years) - 1)
        rets = nav_df.pct_change().dropna()
        sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252))
        mdd = true_mdd(nav_df.values)

        spy_sub = spy_close.reindex(nav_df.index).ffill().dropna()
        spy_rets = spy_sub.pct_change().dropna()
        common_idx = rets.index.intersection(spy_rets.index)
        corr = rets.loc[common_idx].corr(spy_rets.loc[common_idx]) if len(common_idx) > 30 else None

        print(f"\n=== {label} ({len(sub_df.columns)}只ETF, n={len(nav_df)}天) ===")
        print(f"  可用ETF: {sub_df.columns.tolist()}")
        print(f"  CAGR={cagr*100:.2f}%, Sharpe={sharpe:.3f}, 真实MaxDD={mdd*100:.2f}%")
        print(f"  跟SPY相关系数: {corr:.4f}" if corr is not None else "  相关系数样本不足")


if __name__ == "__main__":
    main()
