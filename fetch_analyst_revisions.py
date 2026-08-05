"""
Cycle2小修小补：分析师预期修正动量信号的数据准备。yfinance的
`Ticker.recommendations`只有最近3个月快照，没法回测；但
`Ticker.upgrades_downgrades`有历史评级变动事件(带真实日期GradeDate)，
可以回溯多年，用来算"最近N天净升评数(升评-降评)"这个时点特征。

先对全部554只票做一次抓取(限流：用ThreadPoolExecutor，参考
fetch_pead_data.py同样的模式)，存到本地缓存，供后续标准化测试用。
"""
from __future__ import annotations

import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf

CACHE_PATH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/point_in_time_universe_cache.pkl"
OUTPUT_PATH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/analyst_revisions_cache.pkl"


def fetch_one(ticker):
    try:
        df = yf.Ticker(ticker).upgrades_downgrades
        if df is None or len(df) == 0:
            return ticker, None
        df = df.reset_index()
        return ticker, df[["GradeDate", "Firm", "ToGrade", "FromGrade", "Action"]]
    except Exception:
        return ticker, None


def main():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    tickers = list(cache["c"].columns)
    print(f"抓取 {len(tickers)} 只票的评级变动历史...")

    results = {}
    got, no_data = [], []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_one, t): t for t in tickers}
        done = 0
        for fut in as_completed(futures):
            ticker, df = fut.result()
            if df is not None and len(df) > 0:
                results[ticker] = df
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
