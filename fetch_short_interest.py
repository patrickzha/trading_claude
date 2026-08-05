"""
Cycle2小修小补：卖空余额/融券余量变化信号的数据准备。yfinance的
sharesShort只有最新1-2个月快照，没法回测；FINRA(美国金融业监管局)
免费公开发布双周更新的融券余量数据，有完整历史(至少回溯到2020年)，
用POST查询按symbolCode过滤能拿到单只票的完整时间序列。
"""
from __future__ import annotations

import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

CACHE_PATH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/point_in_time_universe_cache.pkl"
OUTPUT_PATH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/short_interest_cache.pkl"
FINRA_URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"


def fetch_one(ticker):
    body = {"limit": 250, "compareFilters": [
        {"compareType": "EQUAL", "fieldName": "symbolCode", "fieldValue": ticker}
    ]}
    try:
        r = requests.post(FINRA_URL, headers={"Content-Type": "application/json", "Accept": "application/json"},
                          data=json.dumps(body), timeout=20)
        if r.status_code != 200:
            return ticker, None
        data = r.json()
        if not data:
            return ticker, None
        records = [{"date": d["settlementDate"], "short_qty": d["currentShortPositionQuantity"],
                   "days_to_cover": d.get("daysToCoverQuantity")} for d in data]
        return ticker, records
    except Exception:
        return ticker, None


def main():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    tickers = list(cache["c"].columns)
    print(f"抓取 {len(tickers)} 只票的FINRA融券余量历史...")

    results = {}
    got, no_data = [], []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_one, t): t for t in tickers}
        done = 0
        for fut in as_completed(futures):
            ticker, records = fut.result()
            if records:
                results[ticker] = records
                got.append(ticker)
            else:
                no_data.append(ticker)
            done += 1
            if done % 50 == 0:
                print(f"  完成 {done}/{len(tickers)}...")

    print(f"耗时 {time.time()-t0:.1f}s, 有数据: {len(got)}/{len(tickers)}")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump({"data": results, "got": got, "no_data": no_data}, f)
    print(f"已写入 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
