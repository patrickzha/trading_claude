"""
生成标普400中盘股股票池 + GICS行业映射，缓存成 sp400_universe.json。
====================================================================
标普500是全球分析师覆盖最密集、定价最有效的股票池，技术面动量信号在这里
天然难找edge；中盘股覆盖度更低，历史上动量因子在中小盘更有效——这是这次
"继续优化"round里判断"最容易看到效果"的一条新方向，用来测试"换股票池"
本身是不是比继续在标普500里抠模型细节更有希望。

跟 build_sp500_universe.py 用同一套 GICS 行业编码，保持跟现有 SECTOR_CODE_MAP
兼容，不用改模型代码。
"""
from __future__ import annotations

import json

import requests
import pandas as pd
from io import StringIO

GICS_SECTORS = [
    "Communication Services", "Consumer Discretionary", "Consumer Staples",
    "Energy", "Financials", "Health Care", "Industrials", "Information Technology",
    "Materials", "Real Estate", "Utilities",
]
SECTOR_TO_CODE = {name: i for i, name in enumerate(GICS_SECTORS)}
SECTOR_TO_SHORT = {
    "Communication Services": "COMM", "Consumer Discretionary": "CONS_D",
    "Consumer Staples": "CONS_S", "Energy": "ENERGY", "Financials": "FIN",
    "Health Care": "HEALTH", "Industrials": "INDU", "Information Technology": "TECH",
    "Materials": "MATER", "Real Estate": "REAL", "Utilities": "UTIL",
}

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
OUTPUT_PATH = "sp400_universe.json"


def fetch_sp400_table() -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    resp = requests.get(WIKI_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    return tables[0]


def build_universe() -> dict:
    df = fetch_sp400_table()
    universe = {}
    for _, row in df.iterrows():
        ticker = str(row["Symbol"]).strip().replace(".", "-")
        sector = str(row["GICS Sector"]).strip()
        if sector not in SECTOR_TO_CODE:
            continue
        universe[ticker] = {
            "sector": SECTOR_TO_SHORT[sector],
            "sector_code": SECTOR_TO_CODE[sector],
        }
    return universe


if __name__ == "__main__":
    universe = build_universe()
    print(f"拉到 {len(universe)} 只标普400中盘成分股")
    sector_counts = {}
    for info in universe.values():
        sector_counts[info["sector"]] = sector_counts.get(info["sector"], 0) + 1
    for sec, cnt in sorted(sector_counts.items(), key=lambda x: -x[1]):
        print(f"  {sec:8s}: {cnt}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump({
            "sector_to_code": SECTOR_TO_CODE,
            "tickers": universe,
        }, f, indent=2, ensure_ascii=False)
    print(f"已写入 {OUTPUT_PATH}")
