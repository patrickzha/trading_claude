"""
生成标普500股票池 + GICS行业映射，缓存成 sp500_universe.json。
====================================================================
原来的 SECTOR_MAP 是一份写死的50只高beta票（PLTR/COIN/MARA/RIOT这类），
选择标准不明确，用户要求换成标普500这种更宽、更有代表性的股票池。

数据源：Wikipedia "List of S&P 500 companies" 表格，字段包含 Symbol +
GICS Sector。Yahoo Finance 的股票代码用连字符分隔股票类别（如 BRK-B），
Wikipedia 用句点（BRK.B），这里做一次转换。

结果缓存到本地 JSON，而不是每次跑回测都现抓 Wikipedia——名单变动很慢
（一年几次增减），没必要每次运行都依赖一次外部网络请求；需要刷新时
重新跑一次这个脚本就行。
"""
from __future__ import annotations

import json

import requests
import pandas as pd
from io import StringIO

# GICS 11个官方行业，固定编码顺序（字母序，纯类别标签，编码本身没有大小含义）
GICS_SECTORS = [
    "Communication Services", "Consumer Discretionary", "Consumer Staples",
    "Energy", "Financials", "Health Care", "Industrials", "Information Technology",
    "Materials", "Real Estate", "Utilities",
]
SECTOR_TO_CODE = {name: i for i, name in enumerate(GICS_SECTORS)}
# 短标签，用于打印/可读性（不参与建模，建模用的是数字 code）
SECTOR_TO_SHORT = {
    "Communication Services": "COMM", "Consumer Discretionary": "CONS_D",
    "Consumer Staples": "CONS_S", "Energy": "ENERGY", "Financials": "FIN",
    "Health Care": "HEALTH", "Industrials": "INDU", "Information Technology": "TECH",
    "Materials": "MATER", "Real Estate": "REAL", "Utilities": "UTIL",
}

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
OUTPUT_PATH = "sp500_universe.json"


def fetch_sp500_table() -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    resp = requests.get(WIKI_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    return tables[0]


def build_universe() -> dict:
    df = fetch_sp500_table()
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
    print(f"拉到 {len(universe)} 只标普500成分股")
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
