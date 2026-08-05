"""
修复生存偏差第一阶段：现在训练/回测用的股票池是"今天"的标普500名单去套整段
历史，"验证结论"第13条量化过这个问题——78只票是回测期间才被纳入的，在真正
纳入之前不该被当成候选；另外92只曾经的成分股已经被剔除、完全不在数据里。

这个脚本先解决第一半（更容易做的一半）：用 Wikipedia 当前成分股表的"纳入
日期"字段，对每个 ticker 记录它是什么时候进入标普500的，生成一个
{ticker: 纳入日期} 的映射，缓存成 json。backtest/us_stock_screener 里
可以用这个映射，在某一周的 decision_date 之前还没被纳入的票，直接不参与
那一周的候选——避免用"未来才会被纳入指数"的票伪造出历史候选。

第二半（更难的一半：把历史上已经被剔除、现在完全不在名单里的公司重新
找回来，包括它们的价格数据和行业分类）需要更大的工程量，这个脚本先不做，
用 use_point_in_time_universe 开关先做第一半、单独验证效果。
"""
from __future__ import annotations

import json

import requests
import pandas as pd
from io import StringIO

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
OUTPUT_PATH = "sp500_membership_added_dates.json"


def main():
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    resp = requests.get(WIKI_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    current = tables[0]

    added_dates = {}
    for _, row in current.iterrows():
        ticker = str(row["Symbol"]).strip().replace(".", "-")
        date_added = row.get("Date added")
        if pd.isna(date_added):
            continue
        try:
            parsed = pd.to_datetime(date_added).strftime("%Y-%m-%d")
            added_dates[ticker] = parsed
        except Exception:
            continue

    print(f"共 {len(added_dates)} 只票有纳入日期记录")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(added_dates, f, indent=2)
    print(f"已写入 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
