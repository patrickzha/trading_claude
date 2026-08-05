"""
国际市场ETF动量轮动策略——真正意义上的"大修大补"：不是在标普500这个
池子里换个信号，是换一整个资产类别。用发达+新兴市场的国家/地区股票
ETF构建轮动组合，赌的是"不同国家股市的经济周期不完全同步，动量延续
在国家层面一样成立"，跟这次会话所有其他策略(标普500个股层面的动量/
价值/低波动)完全不共享底层股票池，理论上应该有更低的相关性——除非
2008年那种全球系统性危机把所有市场都拖下水(第62条已经证明这种情况下
连同一个市场内部的因子分散都会失效，国际分散大概率也扛不住，但值得
真正测一次，不能只凭直觉判断)。

ETF池：EFA(发达市场除美国)、VGK(欧洲)、EWJ(日本)、EWU(英国)、
EWG(德国)、EWA(澳大利亚)、EWC(加拿大)、EWY(韩国)、EWZ(巴西)、
EWT(台湾)、EWH(香港)、EEM(新兴市场综合)——都是长期存在、流动性好的
iShares/其他发行商ETF，历史数据覆盖面广。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import yfinance as yf

ETF_UNIVERSE = ["EFA", "VGK", "EWJ", "EWU", "EWG", "EWA", "EWC", "EWY", "EWZ", "EWT", "EWH", "EEM"]


def _extract_close_series(df):
    if isinstance(df.columns, pd.MultiIndex):
        close = df.xs("Close", axis=1, level=0)
    else:
        close = df[["Close"]] if "Close" in df.columns else df.iloc[:, [0]]
    return close.squeeze(axis=1)


def fetch_international_etf_data(start: str, end: str, cache_path: str):
    data = yf.download(ETF_UNIVERSE, start=start, end=end, progress=False, group_by="ticker", auto_adjust=True)
    close_dict = {}
    got, no_data = [], []
    for t in ETF_UNIVERSE:
        try:
            df_t = data[t].dropna() if isinstance(data.columns, pd.MultiIndex) else data.dropna()
            if len(df_t) > 250:
                close_dict[t] = df_t["Close"]
                got.append(t)
            else:
                no_data.append(t)
        except Exception:
            no_data.append(t)
    print(f"国际ETF有效数据: {len(got)}/{len(ETF_UNIVERSE)}（{no_data}没有数据，可能是该时期还没上市）")
    c_df = pd.DataFrame(close_dict)
    with open(cache_path, "wb") as f:
        pickle.dump({"c": c_df, "got": got, "no_data": no_data}, f)
    return c_df


def select_top_etfs(price_df: pd.DataFrame, rebalance_date, top_n: int = 4,
                     lookback: int = 252, skip: int = 21):
    rows = []
    for t in price_df.columns:
        px = price_df[t].loc[:rebalance_date].dropna()
        if len(px) < lookback + skip:
            continue
        p_end = px.iloc[-skip - 1] if skip > 0 else px.iloc[-1]
        p_start = px.iloc[-lookback - skip - 1] if len(px) > lookback + skip else px.iloc[0]
        if p_start <= 0:
            continue
        mom = p_end / p_start - 1
        rows.append({"etf": t, "mom": mom})
    if len(rows) < top_n:
        return [r["etf"] for r in rows]
    df = pd.DataFrame(rows).sort_values("mom", ascending=False)
    return df["etf"].head(top_n).tolist()


def backtest_international_rotation(price_df: pd.DataFrame, rebalance_days: int = 21,
                                     top_n: int = 4, lookback: int = 252, skip: int = 21):
    index = price_df.index
    n = len(index)
    daily_nav = []
    rebalance_idx = lookback + skip + 5
    running_nav = 1.0

    while rebalance_idx + rebalance_days <= n:
        rebalance_date = index[rebalance_idx]
        picks = select_top_etfs(price_df, rebalance_date, top_n=top_n, lookback=lookback, skip=skip)

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
        rebalance_idx += rebalance_days

    return daily_nav
