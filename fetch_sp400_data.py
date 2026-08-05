"""
拉取标普400中盘股股票池5年OHLCV数据，缓存供中小盘股票池验证用。
跟 backtest.py 的 fetch_market_data() 逻辑一致，只是股票池换成 sp400_universe.json。
"""
from __future__ import annotations

import json
import pickle
import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

UNIVERSE_PATH = "sp400_universe.json"
OUTPUT_PATH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/sp400_data_cache.pkl"
DAYS = 1825


def main():
    with open(UNIVERSE_PATH) as f:
        universe = json.load(f)
    tickers = list(universe["tickers"].keys())
    print(f"拉取 {len(tickers)} 只标普400中盘票的5年历史数据...")
    t0 = time.time()

    end = datetime.today()
    start = end - timedelta(days=DAYS)
    all_tickers = tickers + ["^VIX"]
    data = yf.download(all_tickers, start=start, end=end, progress=False,
                       group_by="ticker", auto_adjust=True)

    close_dict, open_dict, high_dict, low_dict, vol_dict = {}, {}, {}, {}, {}
    skipped = []
    for t in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                df_t = data[t].dropna()
            else:
                df_t = data.dropna()
            if len(df_t) > 250:
                close_dict[t] = df_t["Close"]
                open_dict[t] = df_t["Open"]
                high_dict[t] = df_t["High"]
                low_dict[t] = df_t["Low"]
                vol_dict[t] = df_t["Volume"]
            else:
                skipped.append(t)
        except Exception:
            skipped.append(t)
            continue

    c_df = pd.DataFrame(close_dict).dropna(how="all")
    o_df = pd.DataFrame(open_dict).dropna(how="all")
    h_df = pd.DataFrame(high_dict).dropna(how="all")
    l_df = pd.DataFrame(low_dict).dropna(how="all")
    v_df = pd.DataFrame(vol_dict).dropna(how="all")

    try:
        if isinstance(data.columns, pd.MultiIndex):
            vix = data["^VIX"]["Close"]
        else:
            vix = data["Close"] if "^VIX" in data.columns else None
    except Exception:
        vix = None
    if vix is None or vix.empty:
        vix = pd.Series(20, index=c_df.index)

    common_idx = c_df.index.intersection(o_df.index).intersection(v_df.index)
    c_df, o_df, h_df, l_df, v_df = [df.loc[common_idx] for df in [c_df, o_df, h_df, l_df, v_df]]
    vix = vix.reindex(common_idx).ffill()

    print(f"有效股票: {len(c_df.columns)}/{len(tickers)}（跳过 {len(skipped)} 只）")
    print(f"交易日: {len(common_idx)}, {common_idx[0].date()} ~ {common_idx[-1].date()}")

    try:
        spy = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
        spy_close = spy["Close"]
        if isinstance(spy_close, pd.DataFrame):
            spy_close = spy_close.iloc[:, 0]
        spy_close = spy_close.reindex(common_idx).ffill()
    except Exception:
        spy_close = None

    cache = {"c": c_df, "o": o_df, "h": h_df, "l": l_df, "v": v_df, "vix": vix, "spy": spy_close}
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(cache, f)

    print(f"耗时: {(time.time()-t0)/60:.1f}分钟")
    print(f"已写入 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
