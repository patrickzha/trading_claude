"""
Cycle23大修大补①："剔除票回补"这一半——第108条只做了"纳入日期过滤"，
一直标注"剔除票回补"是更难的一半、这次会话之前没做。这次尝试做：用
Wikipedia"Selected changes"完整变更记录表(不再只取53只2021年后剔除的
票，取全部有记录的历史剔除)，找出2009-2021年间被剔除、且不在当前503只
股票池、也不在已经拉过的53只(2021年后剔除)里的公司，尝试拉它们的历史
价格数据，能拉到多少算多少——如实预期这次成功率不会是100%(部分公司
可能已经因为破产/私有化/合并太久远，yfinance没有数据，或者ticker代码
在多次并购重组后已经指向别的公司)。
"""
from __future__ import annotations

import json
import pickle
import time

import pandas as pd
import requests
import yfinance as yf
from io import StringIO

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
OUTPUT_PATH = f"{SCRATCH}/removed_tickers_2009_2020_data_cache.pkl"

START = "2009-01-01"
END = "2021-01-01"


def fetch_full_removal_history() -> dict:
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    resp = requests.get(WIKI_URL, headers=headers, timeout=30)
    tables = pd.read_html(StringIO(resp.text))
    changes = tables[1]
    changes.columns = ["effective_date", "added_ticker", "added_name", "removed_ticker", "removed_name", "reason"]
    changes["effective_date_parsed"] = pd.to_datetime(changes["effective_date"], errors="coerce")
    removal_dates = {}
    for _, row in changes.iterrows():
        t = row["removed_ticker"]
        if pd.isna(t):
            continue
        d = row["effective_date_parsed"]
        if pd.isna(d):
            continue
        if t not in removal_dates or d < removal_dates[t]:
            removal_dates[t] = d
    return removal_dates


def main():
    print("拉取Wikipedia完整变更记录表...")
    removal_dates = fetch_full_removal_history()
    print(f"共找到 {len(removal_dates)} 只票的剔除记录(历史全部，不限年份)")

    with open("/Users/zhang/Desktop/trading/sp500_universe.json") as f:
        current_universe = json.load(f)
    current_tickers = set(current_universe["tickers"].keys())

    with open(f"{SCRATCH}/removed_tickers_data_cache.pkl", "rb") as f:
        already_fetched = set(pickle.load(f)["got_data"])
    print(f"已经拉过价格数据的剔除票(2021年后剔除那批): {len(already_fetched)}只")

    target_tickers = []
    for t, d in removal_dates.items():
        if pd.Timestamp(START) <= d <= pd.Timestamp(END) and t not in current_tickers and t not in already_fetched:
            target_tickers.append(t)
    target_tickers = sorted(set(target_tickers))
    print(f"2009-2021年间被剔除、且需要新拉数据的票: {len(target_tickers)}只")
    print(f"样例: {target_tickers[:20]}")

    close_dict, open_dict, high_dict, low_dict, vol_dict, sector_dict = {}, {}, {}, {}, {}, {}
    got, no_data = [], []
    for i, t in enumerate(target_tickers):
        try:
            hist = yf.download(t, start=START, end=END, progress=False, auto_adjust=True)
            if hist is None or len(hist) < 100:
                no_data.append(t)
                continue
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            close_dict[t] = hist["Close"]
            open_dict[t] = hist["Open"]
            high_dict[t] = hist["High"]
            low_dict[t] = hist["Low"]
            vol_dict[t] = hist["Volume"]
            got.append(t)
        except Exception:
            no_data.append(t)
        if (i + 1) % 25 == 0:
            print(f"  进度 {i+1}/{len(target_tickers)}, 成功{len(got)}只")
        time.sleep(0.05)

    print(f"\n最终结果: {len(got)}/{len(target_tickers)} 只票成功拉到数据")
    print(f"没拉到数据的票(样例): {no_data[:20]}")

    cache = {
        "close": close_dict, "open": open_dict, "high": high_dict, "low": low_dict, "volume": vol_dict,
        "got_data": got, "no_data": no_data, "removal_dates": {t: removal_dates[t] for t in got},
    }
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(cache, f)
    print(f"已写入 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
