"""
拉取标普500全量股票的历史财报日期+盈利惊喜幅度(Surprise%)，供PEAD
(财报后动量漂移)子策略研究用。PEAD是经过大量学术验证的独立现象——
财报大幅超预期后，股价会在接下来几周持续温和上涨（市场对新信息的
消化不是瞬间完成的），是跟价格动量完全不同来源的信号，现在的主策略
反而是完全排除财报期的票（财报避让），这条子策略正好补上这块。

跟 backtest.py 的 fetch_earnings_calendar 同样用 ThreadPoolExecutor 并发拉，
yfinance 的 get_earnings_dates(limit=30) 实测能拿到2014年至今的历史，
覆盖整个回测期绰绰有余。
"""
from __future__ import annotations

import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import yfinance as yf

from us_stock_screener import SECTOR_MAP

OUTPUT_PATH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/pead_data_cache.pkl"


def fetch_one(ticker: str):
    try:
        t = yf.Ticker(ticker)
        ed = t.get_earnings_dates(limit=30)
        if ed is None or len(ed) == 0:
            return ticker, None
        ed = ed[ed["Reported EPS"].notna() & ed["Surprise(%)"].notna()]
        if len(ed) == 0:
            return ticker, None
        records = [
            {"date": idx.tz_localize(None).normalize(), "surprise_pct": float(row["Surprise(%)"])}
            for idx, row in ed.iterrows()
        ]
        return ticker, records
    except Exception:
        return ticker, None


def main():
    tickers = list(SECTOR_MAP.keys())
    print(f"拉取 {len(tickers)} 只票的历史财报惊喜数据...")
    t0 = time.time()
    cache = {}
    failed = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(fetch_one, t): t for t in tickers}
        done = 0
        for future in as_completed(futures):
            ticker, records = future.result()
            if records:
                cache[ticker] = records
            else:
                failed.append(ticker)
            done += 1
            if done % 50 == 0:
                print(f"  完成 {done}/{len(tickers)}...")

    print(f"耗时: {time.time()-t0:.1f}s")
    print(f"成功: {len(cache)}/{len(tickers)}, 失败: {len(failed)}")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(cache, f)
    print(f"已写入 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
