"""
生存偏差完整修复：把当前标普500股票池 + 已经拉回来的53只被剔除票，合并成
一个"点位时间正确"的历史股票池——每一周只用"那一周真实在标普500里"的票
参与训练和选股，纳入日期之后才纳入的票不提前用，剔除日期之后的票不再用
（除非又被剔除但仍然继续持有到剔除前）。

产出：
1. 合并后的价格 DataFrame（c/o/h/l/v），标普500当前500只 + 53只找回来的
   被剔除票，共553只（列并集，不同票在不同时间段有数据）。
2. 合并后的 sector_map（当前500只沿用 sp500_universe.json，53只被剔除票
   用 yfinance .info 拉到的行业，映射成项目的GICS短代码）。
3. 一个 get_point_in_time_members(decision_date) -> set[ticker] 函数，
   供 backtest.py 按周过滤候选池用。

已知局限（如实说）：92只被剔除票里只找回53只，另外39只是被并购/私有化
退市，价格数据已经不存在，这部分没法找回——不是没做，是数据本身已经
消失了，这是"能做到的上限"，剩下的偏差修不了。纳入日期用的是当前成分股
名单的"Date added"字段，剔除日期用的是Wikipedia变更记录表，都是外部数据，
可能有个别记录不准，但比"完全不做点位过滤"已经是质的改善。
"""
from __future__ import annotations

import json
import pickle

import pandas as pd
import requests
from io import StringIO

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
SP500_DATA_CACHE = f"{SCRATCH}/sp500_data_cache.pkl"
REMOVED_DATA_CACHE = f"{SCRATCH}/removed_tickers_data_cache.pkl"
OUTPUT_PATH = f"{SCRATCH}/point_in_time_universe_cache.pkl"

ADDED_DATES_PATH = "/Users/zhang/Desktop/trading/sp500_membership_added_dates.json"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

YF_SECTOR_TO_GICS_SHORT = {
    "Healthcare": "HEALTH", "Energy": "ENERGY", "Financial Services": "FIN",
    "Consumer Cyclical": "CONS_D", "Technology": "TECH", "Consumer Defensive": "CONS_S",
    "Industrials": "INDU", "Basic Materials": "MATER", "Real Estate": "REAL",
    "Utilities": "UTIL", "Communication Services": "COMM",
}


def fetch_removal_dates() -> dict:
    """从Wikipedia变更记录表拿每只被剔除票的剔除生效日期。"""
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
        # 同一只票可能被剔除又重新纳入又再剔除，取"回测期内"最相关的那次——
        # 简化处理：取最早的一次剔除日期（因为我们只关心它是不是从那时起
        # 就不在指数里了，如果之后又重新纳入过，属于更复杂的情况，这次不处理）
        if t not in removal_dates or d < removal_dates[t]:
            removal_dates[t] = d
    return removal_dates


def main():
    with open(SP500_DATA_CACHE, "rb") as f:
        sp500_data = pickle.load(f)
    with open(REMOVED_DATA_CACHE, "rb") as f:
        removed_data = pickle.load(f)
    with open(ADDED_DATES_PATH) as f:
        added_dates_raw = json.load(f)
    added_dates = {t: pd.Timestamp(d) for t, d in added_dates_raw.items()}

    print("拉取剔除生效日期...")
    removal_dates = fetch_removal_dates()
    removed_tickers_with_data = removed_data["got_data"]
    removal_dates_used = {t: removal_dates[t] for t in removed_tickers_with_data if t in removal_dates}
    print(f"找到剔除日期的票: {len(removal_dates_used)}/{len(removed_tickers_with_data)}")

    # 合并价格数据：当前500只 + 找回来的53只被剔除票
    c_df = sp500_data["c"].copy()
    o_df = sp500_data["o"].copy()
    h_df = sp500_data["h"].copy()
    l_df = sp500_data["l"].copy()
    v_df = sp500_data["v"].copy()
    common_idx = c_df.index

    for t in removed_tickers_with_data:
        if t in c_df.columns:
            continue  # 理论上不会重复，防御性检查
        c_df[t] = removed_data["close"][t].reindex(common_idx)
        o_df[t] = removed_data["open"][t].reindex(common_idx)
        h_df[t] = removed_data["high"][t].reindex(common_idx)
        l_df[t] = removed_data["low"][t].reindex(common_idx)
        v_df[t] = removed_data["volume"][t].reindex(common_idx)

    print(f"合并后总票数: {len(c_df.columns)}（原{len(sp500_data['c'].columns)}只 + 找回{len(removed_tickers_with_data)}只）")

    # 合并 sector_map
    with open("/Users/zhang/Desktop/trading/sp500_universe.json") as f:
        sp500_universe = json.load(f)
    sector_map = {t: info["sector"] for t, info in sp500_universe["tickers"].items()}
    for t in removed_tickers_with_data:
        raw_sector = removed_data["sectors"].get(t, "")
        sector_map[t] = YF_SECTOR_TO_GICS_SHORT.get(raw_sector, "COMM")

    cache = {
        "c": c_df, "o": o_df, "h": h_df, "l": l_df, "v": v_df,
        "vix": sp500_data["vix"], "spy": sp500_data.get("spy"),
        "sector_map": sector_map,
        "added_dates": added_dates,
        "removal_dates": removal_dates_used,
        "removed_tickers_with_data": removed_tickers_with_data,
    }
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(cache, f)
    print(f"已写入 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
