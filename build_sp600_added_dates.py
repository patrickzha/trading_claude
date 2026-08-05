"""
Cycle61大修大补①：给标普600小盘股票池做部分点位时间(PIT)修正——第277条
用"当前"成分股名单测的动量腿结果(Sharpe=0.47)存在生存偏差，需要至少
排除"未来才被纳入、但被提前用于更早历史周"这个最直接的前视偏差来源。

如实标注局限(比标普500的完整修复第27条弱)：这次只做"纳入日期"这一半的
修正(排除纳入日期之后才有效的候选)，没有像标普500那样额外抓回"已被
剔除的历史成分股"的价格数据——那需要一个新的、单独的历史成分股价格
抓取管道(参考`fetch_removed_tickers_data.py`)，工程量接近再造一次完整
的标普500 PIT修复项目，这次会话时间预算内不投入。这意味着：这次修正
后的历史股票池，对更早的周而言，仍然会缺失"当时在指数里、但后来被剔除
的票"，是一个方向明确但幅度未知的残余偏差(标普500经验：完整修复比
"只做纳入日期"这一半的修复重要得多，第27条主要的Sharpe暴跌来自剔除票
的缺失，不是纳入票的过早使用)——如实预期这次的部分修正效果会明显小于
标普500那次的完整修复，不能类比第27条的量级。
"""
from __future__ import annotations

import json
from io import StringIO

import pandas as pd
import requests

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
OUTPUT_PATH = "sp600_added_dates.json"


def main():
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    resp = requests.get(WIKI_URL, headers=headers, timeout=30)
    tables = pd.read_html(StringIO(resp.text))
    changes = tables[1]
    changes.columns = ["date", "added_ticker", "added_name", "removed_ticker", "removed_name", "reason"]
    changes["date_parsed"] = pd.to_datetime(changes["date"], errors="coerce")

    added_dates = {}
    for _, row in changes.iterrows():
        t = row["added_ticker"]
        d = row["date_parsed"]
        if pd.isna(t) or pd.isna(d):
            continue
        t = str(t).strip().replace(".", "-")
        # 同一只票可能多次被纳入(剔除后又纳入)，保留最早一次纳入日期，
        # 这样对"这次这个部分修正版本"是保守的(尽量少排除、少产生新偏差)
        if t not in added_dates or d < added_dates[t]:
            added_dates[t] = d

    print(f"从变更记录表拿到 {len(added_dates)} 只票的纳入日期记录")

    with open(OUTPUT_PATH, "w") as f:
        json.dump({t: d.strftime("%Y-%m-%d") for t, d in added_dates.items()}, f, indent=2)
    print(f"已写入 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
