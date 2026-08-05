"""
Cycle60大修大补②(框架改动，新数据维度——股票池粒度，真小盘)：生成标普600
小盘股股票池 + GICS行业映射，缓存成 sp600_universe.json。
====================================================================
第20条已经测过标普400中盘股，结果是标普500全面更好，推翻了"中小盘更容易
找到动量优势"这个假设——但标普400严格说是中盘股，不是真正的小盘股，学术
文献里动量/价值因子溢价最强的证据通常来自更小市值的股票池(覆盖度更低、
套利资金更少)。标普600是标普道琼斯官方定义的美股小盘股指数，市值区间
明显低于标普400，是第20条"换股票池"这条线索里唯一还没测过的粒度，用同一套
GICS行业编码保持跟现有SECTOR_CODE_MAP兼容。
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

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
OUTPUT_PATH = "sp600_universe.json"


def fetch_sp600_table() -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    resp = requests.get(WIKI_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    return tables[0]


def build_universe() -> dict:
    df = fetch_sp600_table()
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
    print(f"拉到 {len(universe)} 只标普600小盘成分股")
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
