"""
从 SEC EDGAR 的 Company Facts API 拉取基本面数据，缓存成按"实际披露日期"
索引的时间序列，供 us_stock_screener.py 的估值特征使用。
================================================================
之前调研过 yfinance 的免费基本面数据，历史深度只有5-6个季度，覆盖不了
5年回测窗口。SEC EDGAR 的 Company Facts API（data.sec.gov/api/xbrl/
companyfacts/）是免费的、历史能倒回到XBRL强制披露开始（大部分公司
2009年前后），而且每条数据自带 `filed`（实际公开披露日期），不用像
yfinance那样自己猜一个保守延迟——这是用点位数据做时间序列最容易踩坑的
地方（前视偏差），SEC这份数据从根上解决了这个问题。

只取4个字段，够算 P/E、P/B、ROE、负债权益比这4个基本面特征：
- EarningsPerShareDiluted（年报EPS，做TTM代理，一年刷新一次）
- NetIncomeLoss（年报净利润，同上）
- StockholdersEquity（季度快照，季度刷新）
- Liabilities（季度快照，做"负债"的代理——XBRL里没有统一干净的"Total Debt"
  标签，不同公司用的标签五花八门，Liabilities（总负债）是覆盖率最高、
  最标准化的替代）
- EntityCommonStockSharesOutstanding（dei命名空间，股本数，季度快照）

去重规则：同一个 end（报告期）在后续财报里会作为"上期对比数"重复出现，
只保留每个 end 第一次被披露（filed 最早）的那条——这代表这个数字"第一次
真正变成公开信息"的时间点，后续财报重复展示的同一个数字不该被当成新信息。
只保留 10-Q / 10-K 正式表单，跳过 /A 修正版，避免修正/重述数据的复杂性。
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

HEADERS = {"User-Agent": "trading-research contact@example.com"}
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sec_fundamentals_cache.json")

# 每个概念给一条"标签优先级列表"——不同公司在 XBRL 里对同一个财务概念
# 用的标准标签不完全统一（比如有没有少数股东权益、EPS是分开报还是合并报），
# 抽样测过10只票已经发现好几个标准名字覆盖不全的情况，所以按优先级依次
# 尝试，取第一个有数据的。canonical_name 是存进缓存时统一用的字段名。
# 不缓存 EPS 标签本身——抽样发现不同公司在"分开报基本/稀释EPS"还是
# "合并报"上不统一，标签覆盖率不稳定。net_income 和 shares_outstanding
# 两个标签覆盖率都很好，自己算 EPS = net_income / shares_outstanding
# 更稳健，也顺带减少一个可能缺失的字段。
FIELD_FALLBACKS = {
    "net_income": [("us-gaap", "NetIncomeLoss"),
                   ("us-gaap", "ProfitLoss")],
    "stockholders_equity": [("us-gaap", "StockholdersEquity"),
                            ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")],
    "liabilities": [("us-gaap", "Liabilities")],
    "shares_outstanding": [("dei", "EntityCommonStockSharesOutstanding"),
                           ("us-gaap", "CommonStockSharesOutstanding"),
                           ("us-gaap", "CommonStockSharesIssued")],
    # 【Cycle74新增，见README第318条】第318条的基本面ML测试只有4个特征，
    # 结论明确指向"特征信息量不够，不是模型架构的问题"，这次扩充3个新
    # 概念，都是利润表/现金流量表的"流量"科目，年度口径(fy_only=True)：
    "revenue": [("us-gaap", "Revenues"),
                ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax")],
    "gross_profit": [("us-gaap", "GrossProfit")],
    "operating_cash_flow": [("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
                            ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")],
}

# 流量类科目(利润表/现金流量表)要用fy_only=True按整年口径去重，避免把单季度
# 数字误当成年度数字(见_dedup_by_end的docstring)；存量类科目(资产负债表)是
# 某个时间点快照，不需要这个参数。
FLOW_FIELDS = {"net_income", "revenue", "gross_profit", "operating_cash_flow"}


def fetch_ticker_to_cik() -> dict:
    resp = requests.get(TICKER_MAP_URL, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    return {v["ticker"]: str(v["cik_str"]).zfill(10) for v in data.values()}


def _period_days(entry: dict) -> int | None:
    if "start" not in entry:
        return None
    from datetime import date
    s = date.fromisoformat(entry["start"])
    e = date.fromisoformat(entry["end"])
    return (e - s).days


def _dedup_by_end(entries: list, fy_only: bool = False) -> list:
    """
    同一个 end 只保留 filed 最早的那条（正式10-Q/10-K，不含修正版）。

    fy_only=True 用在利润表科目（净利润这类"流量"数据）上。关键坑：不能只
    信 XBRL 自带的 fp=='FY' 标签——实测发现苹果这类公司的10-K里会把"整年"
    和"单独第四季度"两条数据都标成 fp='FY'（第四季度数据来自10-K附注里的
    "分季度财务数据"披露），如果只按 fp 过滤，会把季度数字误当成年度数字，
    同一个 end 日期下两个完全不同的数字（年度42亿 vs 单季度8亿）用 end 一
    去重就会随机选中一个。改成直接按实际跨度天数判断（350~380天才算"真的
    是整年"），不依赖标签本身，更可靠。资产负债表科目（净资产、负债这类
    "存量"数据）是某个时间点的快照，没有这种口径混淆问题，不用设这个参数。
    """
    valid = [e for e in entries if e.get("form") in ("10-Q", "10-K")]
    if fy_only:
        valid = [e for e in valid if (_period_days(e) or 0) >= 350]
    best_by_end: dict = {}
    for e in valid:
        end = e["end"]
        if end not in best_by_end or e["filed"] < best_by_end[end]["filed"]:
            best_by_end[end] = e
    return sorted(best_by_end.values(), key=lambda x: x["end"])


def fetch_one_ticker(ticker: str, cik: str) -> tuple:
    try:
        resp = requests.get(FACTS_URL.format(cik=cik), headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            return ticker, None
        data = resp.json()
        facts = data.get("facts", {})

        result = {}
        for canonical_name, candidates in FIELD_FALLBACKS.items():
            fy_only = canonical_name in FLOW_FIELDS
            # 【Cycle74修复，真实bug】原来的逻辑是"优先级列表里第一个有任何
            # 数据的标签就用它，不再往下试"——实测发现AAPL这类公司在2018年
            # ASC 606新收入准则生效前后从`Revenues`标签切换到
            # `RevenueFromContractWithCustomerExcludingAssessedTax`标签，
            # 旧标签只剩2016-2018这11条历史数据，新逻辑会因为旧标签"有数据"
            # 就直接采用只有11条的旧标签、完全无视新标签117条覆盖2017-2026的
            # 数据，revenue覆盖率被这个bug严重低估(AAPL缓存里revenue只有3条
            # 年度记录，其他字段都有18-19条)。改成把候选列表里所有标签的
            # 原始条目合并起来，一起交给_dedup_by_end按end日期去重(同一个
            # end如果两个标签都有，按filed更早的那条为准，这是标签迁移
            # 期间两个标签短暂重叠时的合理默认)，不再"选中第一个就不看别的"。
            all_entries = []
            for namespace, tag in candidates:
                ns_data = facts.get(namespace, {})
                if tag not in ns_data:
                    continue
                units = ns_data[tag]["units"]
                unit_key = next(iter(units.keys()))
                all_entries.extend(units[unit_key])
            if all_entries:
                entries = _dedup_by_end(all_entries, fy_only=fy_only)
                if entries:
                    result[canonical_name] = [
                        {"end": e["end"], "filed": e["filed"], "val": e["val"], "fp": e.get("fp", "")}
                        for e in entries
                    ]
        return ticker, result if result else None
    except Exception:
        return ticker, None


def build_cache(tickers: list[str], n_workers: int = 6) -> dict:
    print("拉取 ticker -> CIK 映射...")
    ticker_to_cik = fetch_ticker_to_cik()

    matched = [(t, ticker_to_cik[t]) for t in tickers if t in ticker_to_cik]
    unmatched = [t for t in tickers if t not in ticker_to_cik]
    if unmatched:
        print(f"  [警告] {len(unmatched)} 只票在SEC映射里找不到CIK，跳过：{unmatched[:10]}{'...' if len(unmatched) > 10 else ''}")
    print(f"匹配到CIK: {len(matched)}/{len(tickers)}")

    cache = {}
    failed = []
    t0 = time.time()
    # SEC 建议不超过 ~10 请求/秒，6个并发线程、每个请求本身有网络延迟，
    # 实测综合下来不会超限
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(fetch_one_ticker, t, cik): t for t, cik in matched}
        done = 0
        for future in as_completed(futures):
            ticker, result = future.result()
            if result:
                cache[ticker] = result
            else:
                failed.append(ticker)
            done += 1
            if done % 50 == 0:
                print(f"  完成 {done}/{len(matched)}...")

    print(f"耗时: {time.time()-t0:.1f}s")
    print(f"成功: {len(cache)}/{len(matched)}, 失败: {len(failed)}")
    if failed:
        print(f"  失败列表（前20个）: {failed[:20]}")
    return cache


if __name__ == "__main__":
    import shutil
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from us_stock_screener import SECTOR_MAP

    backup_path = OUTPUT_PATH.replace(".json", "_PRE_EXPANDED_FIELDS_BACKUP.json")
    if os.path.exists(OUTPUT_PATH) and not os.path.exists(backup_path):
        shutil.copy(OUTPUT_PATH, backup_path)
        print(f"已备份原文件(只有4个字段的旧版)到 {backup_path}")

    cache = build_cache(list(SECTOR_MAP.keys()))
    with open(OUTPUT_PATH, "w") as f:
        json.dump(cache, f)
    print(f"已写入 {OUTPUT_PATH}")
