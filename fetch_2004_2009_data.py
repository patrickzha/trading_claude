"""
第61条提到：价值策略这条腿被SEC数据2009年之前完全空白卡死，但动量和
低波动这两条腿只需要价格数据，yfinance的价格历史通常能回溯到90年代或
IPO日——这次拉2004-06~2009-06，包含2008年金融危机，测"动量+低波动"
两条腿(不含价值)在真正的深度熊市里表现如何，是免费数据能做到的、更早
一段区间的部分验证。
"""
from __future__ import annotations

import pickle

import pandas as pd
import yfinance as yf

HIST_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/historical_2014_2020_cache.pkl"
OUTPUT_PATH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/historical_2004_2009_cache.pkl"

START = "2004-06-01"
END = "2009-06-02"


def _extract_close_series(df):
    if isinstance(df.columns, pd.MultiIndex):
        close = df.xs("Close", axis=1, level=0)
    else:
        close = df[["Close"]] if "Close" in df.columns else df.iloc[:, [0]]
    return close.squeeze(axis=1)


def main():
    with open(HIST_CACHE, "rb") as f:
        hist = pickle.load(f)
    tickers = list(hist["c"].columns)
    print(f"拉取 {len(tickers)} 只票 {START}~{END} 的历史价格...")

    data = yf.download(tickers, start=START, end=END, progress=False, group_by="ticker", auto_adjust=True)
    data_spy = yf.download("SPY", start=START, end=END, progress=False, auto_adjust=True)

    close_dict, open_dict, high_dict, low_dict, vol_dict = {}, {}, {}, {}, {}
    got, no_data = [], []
    for t in tickers:
        try:
            df_t = data[t].dropna() if isinstance(data.columns, pd.MultiIndex) else data.dropna()
            if len(df_t) > 500:
                close_dict[t] = df_t["Close"]
                open_dict[t] = df_t["Open"]
                high_dict[t] = df_t["High"]
                low_dict[t] = df_t["Low"]
                vol_dict[t] = df_t["Volume"]
                got.append(t)
            else:
                no_data.append(t)
        except Exception:
            no_data.append(t)

    print(f"有效数据: {len(got)}/{len(tickers)}")

    c_df = pd.DataFrame(close_dict)
    o_df = pd.DataFrame(open_dict)
    h_df = pd.DataFrame(high_dict)
    l_df = pd.DataFrame(low_dict)
    v_df = pd.DataFrame(vol_dict)
    spy_close = _extract_close_series(data_spy)
    assert spy_close.ndim == 1

    cache = {"c": c_df, "o": o_df, "h": h_df, "l": l_df, "v": v_df, "spy": spy_close, "got": got, "no_data": no_data}
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(cache, f)
    print(f"已写入 {OUTPUT_PATH}")
    print(f"日期范围: {c_df.index[0].date()} ~ {c_df.index[-1].date()}, {len(c_df)}个交易日")


if __name__ == "__main__":
    main()
