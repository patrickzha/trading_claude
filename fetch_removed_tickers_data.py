"""
生存偏差完整修复第一步：拉取2021-08-02至今被剔除出标普500、且现在仍不在
名单里的92只票的历史价格+行业分类。这些票完全不在 sp500_data_cache.pkl
里（那份缓存只包含"现在"的500只成分股），是"验证结论"第13条量化过的
生存偏差的另一半——现在缺的这部分。

很多票是被并购退市（比如XLNX被AMD收购、CERN被Oracle收购），yfinance
拿不到数据很正常（公司已经不存在了），这部分票会被跳过、无法找回，
是生存偏差修复能做到的上限，不是bug。价格数据只拉到"仍在交易"的部分。
"""
from __future__ import annotations

import json
import pickle
import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

REMOVED_LIST_CSV = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/removed_tickers.csv"
OUTPUT_PATH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/removed_tickers_data_cache.pkl"

START = "2021-06-01"
END = "2026-08-01"

# GICS 短标签映射，跟 build_sp500_universe.py 保持一致，方便sector_map兼容
SECTOR_KEYWORDS_FALLBACK = "UNKNOWN"


def main():
    removed_df = pd.read_csv(REMOVED_LIST_CSV)
    tickers = removed_df["removed_ticker"].tolist()
    print(f"尝试拉取 {len(tickers)} 只已剔除票的历史数据...")

    all_tickers = tickers
    data = yf.download(all_tickers, start=START, end=END, progress=False,
                       group_by="ticker", auto_adjust=True)

    close_dict, open_dict, high_dict, low_dict, vol_dict = {}, {}, {}, {}, {}
    sectors = {}
    got_data, no_data = [], []
    for t in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                df_t = data[t].dropna()
            else:
                df_t = data.dropna()
            if len(df_t) > 100:
                close_dict[t] = df_t["Close"]
                open_dict[t] = df_t["Open"]
                high_dict[t] = df_t["High"]
                low_dict[t] = df_t["Low"]
                vol_dict[t] = df_t["Volume"]
                got_data.append(t)
            else:
                no_data.append(t)
        except Exception:
            no_data.append(t)
            continue

    print(f"有价格数据: {len(got_data)}/{len(tickers)}")
    print(f"无数据(大概率已并购退市): {no_data}")

    # 拉行业分类——用 yfinance .info，逐个拉（有数据的这批不多，~40-60只，可以接受）
    print("拉取行业分类...")
    t0 = time.time()
    for t in got_data:
        try:
            info = yf.Ticker(t).info
            sector_raw = info.get("sector", "")
            sectors[t] = sector_raw if sector_raw else SECTOR_KEYWORDS_FALLBACK
        except Exception:
            sectors[t] = SECTOR_KEYWORDS_FALLBACK
    print(f"行业分类耗时: {time.time()-t0:.1f}s")

    cache = {
        "close": close_dict, "open": open_dict, "high": high_dict,
        "low": low_dict, "volume": vol_dict, "sectors": sectors,
        "got_data": got_data, "no_data": no_data,
    }
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(cache, f)
    print(f"已写入 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
