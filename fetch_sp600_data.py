"""
拉取标普600小盘股股票池5年OHLCV数据，缓存供小盘股票池验证用。
跟 fetch_sp400_data.py 逻辑一致，只是股票池换成 sp600_universe.json。
"""
from __future__ import annotations

import json
import pickle
import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

UNIVERSE_PATH = "sp600_universe.json"
OUTPUT_PATH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/sp600_data_cache.pkl"
DAYS = 1825


def main():
    with open(UNIVERSE_PATH) as f:
        universe = json.load(f)
    tickers = list(universe["tickers"].keys())
    print(f"拉取 {len(tickers)} 只标普600小盘票的5年历史数据...")
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

    c_df = pd.DataFrame(close_dict).dropna(how="all")
    o_df = pd.DataFrame(open_dict).dropna(how="all")
    h_df = pd.DataFrame(high_dict).dropna(how="all")
    l_df = pd.DataFrame(low_dict).dropna(how="all")
    v_df = pd.DataFrame(vol_dict).dropna(how="all")

    vix = None
    if isinstance(data.columns, pd.MultiIndex) and "^VIX" in data.columns.get_level_values(0):
        vix = data["^VIX"]["Close"].dropna()

    spy_close = None
    try:
        spy_data = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
        spy_close = spy_data["Close"].dropna()
        if isinstance(spy_close, pd.DataFrame):
            spy_close = spy_close.iloc[:, 0]
    except Exception as e:
        print(f"  [警告] SPY拉取失败: {e}")

    elapsed = (time.time() - t0) / 60
    print(f"完成，耗时{elapsed:.1f}分钟。有效票数={len(c_df.columns)}，跳过={len(skipped)}")
    if skipped:
        print(f"  跳过的票(数据不足250天): {skipped[:20]}{'...' if len(skipped) > 20 else ''}")

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump({"c": c_df, "o": o_df, "h": h_df, "l": l_df, "v": v_df, "vix": vix, "spy": spy_close}, f)
    print(f"已写入 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
