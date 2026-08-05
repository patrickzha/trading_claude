"""
Cycle64大修大补②(框架改动，新资产类别——非美国发达市场个股)：生成德国DAX40
股票池。这次会话此前测试过的"换股票池"方向全部还在美股范围内(标普400
中盘第20条、标普600小盘第277/281条)，国际市场此前也只测过ETF级别的
国家轮动(第64条，否决)，从未在个股层面测试过完全不同司法管辖区/货币/
交易时区的股票池——德国是欧洲最大经济体，DAX40流动性好、yfinance覆盖
完整(`.DE`后缀)，是这条线索里数据可行性最好的起点。
"""
from __future__ import annotations

import json
from io import StringIO

import pandas as pd
import requests

WIKI_URL = "https://en.wikipedia.org/wiki/DAX"
OUTPUT_PATH = "dax_universe.json"

SECTOR_MAP_LOOSE = {
    "Automobiles": "CONS_D", "Chemicals": "MATER", "Software": "TECH",
    "Insurance": "FIN", "Banks": "FIN", "Industrial Engineering": "INDU",
    "Health Care": "HEALTH", "Utilities": "UTIL", "Real Estate": "REAL",
    "Telecommunications": "COMM", "Retail": "CONS_D", "Personal Care": "CONS_S",
    "Basic Resources": "MATER", "Construction": "INDU", "Technology": "TECH",
    "Financial Services": "FIN", "Consumer Products": "CONS_S",
    "Media": "COMM", "Travel": "CONS_D", "Chemical": "MATER",
}


def main():
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    resp = requests.get(WIKI_URL, headers=headers, timeout=30)
    tables = pd.read_html(StringIO(resp.text))
    df = tables[4]
    universe = {}
    for _, row in df.iterrows():
        ticker = str(row["Ticker"]).strip()
        sector_raw = str(row["Prime Standard Sector"]).strip()
        sector = SECTOR_MAP_LOOSE.get(sector_raw, "INDU")
        if not ticker or ticker == "nan":
            continue
        # Wikipedia的Ticker列本身已经带交易所后缀(比如ADS.DE、AIR.PA)，
        # 第一版脚本直接拼接".DE"产生了"ADS.DE.DE"这种错误代码，导致
        # yfinance全部40只票请求失败——如实记录这个bug，修复为直接用
        # 原始ticker，不再额外拼接后缀。
        universe[ticker] = {"sector": sector}
    print(f"拉到 {len(universe)} 只DAX40成分股")
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"tickers": universe}, f, indent=2, ensure_ascii=False)
    print(f"已写入 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
