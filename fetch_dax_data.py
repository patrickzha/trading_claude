"""拉取DAX40股票池5年OHLCV数据，供动量策略可行性验证用。跟fetch_sp600_data.py
逻辑一致，只是股票池换成dax_universe.json，基准换成德国DAX指数(^GDAXI)。"""
from __future__ import annotations

import json
import pickle
import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

UNIVERSE_PATH = "dax_universe.json"
OUTPUT_PATH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/dax_data_cache.pkl"
DAYS = 1825


def main():
    with open(UNIVERSE_PATH) as f:
        universe = json.load(f)
    tickers = list(universe["tickers"].keys())
    print(f"拉取 {len(tickers)} 只DAX40票的5年历史数据...")
    t0 = time.time()

    end = datetime.today()
    start = end - timedelta(days=DAYS)
    data = yf.download(tickers, start=start, end=end, progress=False,
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

    vix = None  # 用美股VIX作为宏观熔断信号的代理(如实标注：不是德国本地波动率指数)
    try:
        vix_data = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
        vix = vix_data["Close"].dropna()
        if isinstance(vix, pd.DataFrame):
            vix = vix.iloc[:, 0]
    except Exception:
        pass

    dax_close = None
    try:
        dax_data = yf.download("^GDAXI", start=start, end=end, progress=False, auto_adjust=True)
        dax_close = dax_data["Close"].dropna()
        if isinstance(dax_close, pd.DataFrame):
            dax_close = dax_close.iloc[:, 0]
    except Exception as e:
        print(f"  [警告] DAX指数拉取失败: {e}")

    elapsed = (time.time() - t0) / 60
    print(f"完成，耗时{elapsed:.1f}分钟。有效票数={len(c_df.columns)}，跳过={len(skipped)}")
    if skipped:
        print(f"  跳过: {skipped}")

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump({"c": c_df, "o": o_df, "h": h_df, "l": l_df, "v": v_df, "vix": vix, "spy": dax_close}, f)
    print(f"已写入 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
