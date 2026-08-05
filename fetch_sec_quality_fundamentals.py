"""
Cycle26大修大补②：跳出这次会话反复验证过"没有真实预测力"的21维技术特征
框架，尝试一个完全不同的选股范式——质量因子(quality investing)。这次
会话已经测过动量(技术面)、价值(P/E+P/B+ROE)、低波动(已实现波动率)三种
因子，质量因子是学术界公认的第四种、机制上跟价值因子低相关的独立因子
(Novy-Marx 2013等研究：高毛利率/高经营利润率的公司即使估值不便宜，
未来收益依然显著跑赢；应计利润异象：经营现金流跟净利润差距大的公司
[利润质量差]未来收益更差)。

复用已经验证过时间点正确性的SEC EDGAR Company Facts API管道(跟
fetch_sec_fundamentals.py同样的方法论)，这次多抓5个字段：
- Revenues(营业收入)、GrossProfit(毛利润)：算毛利率
- OperatingIncomeLoss(经营利润)：算经营利润率
- NetCashProvidedByUsedInOperatingActivities(经营活动现金流)：跟净利润
  对比算应计利润(净利润-经营现金流，越大说明利润质量越差)
- Assets(总资产)：算资产增长率(学术界的"资产增长异象"：资产扩张过快的
  公司未来收益往往更差，可能是过度投资/管理层过度自信的信号)

如实说明：这些字段的SEC覆盖率预期会比净利润/股东权益这两个最基础的
科目更低(不是所有公司都用完全一致的标签披露)，抓取时会如实统计覆盖率，
覆盖率过低就诚实报告、不勉强凑数据。
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from fetch_sec_fundamentals import fetch_ticker_to_cik, _dedup_by_end

HEADERS = {"User-Agent": "trading-research contact@example.com"}
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sec_quality_fundamentals_cache.json")

FIELD_FALLBACKS = {
    "revenues": [("us-gaap", "Revenues"),
                 ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
                 ("us-gaap", "SalesRevenueNet")],
    "gross_profit": [("us-gaap", "GrossProfit")],
    "operating_income": [("us-gaap", "OperatingIncomeLoss")],
    "operating_cash_flow": [("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
                             ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")],
    "total_assets": [("us-gaap", "Assets")],
}
FLOW_FIELDS = {"revenues", "gross_profit", "operating_income", "operating_cash_flow"}  # 利润表/现金流量表科目，用fy_only去重


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
            # 【Cycle75修复，同README第330条在fetch_sec_fundamentals.py发现
            # 的同一类bug】原逻辑"优先级列表里第一个有数据的标签就用、不看
            # 后面标签"，公司在ASC 606等准则切换XBRL标签时会导致只用到
            # 覆盖率差得多的旧标签(`revenues`这里甚至有3个候选标签，比
            # fetch_sec_fundamentals.py的revenue字段还多一个，风险更高)。
            # 改成合并所有候选标签的原始条目再统一去重。
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
    print(f"匹配到CIK: {len(matched)}/{len(tickers)}")

    cache = {}
    failed = []
    t0 = time.time()
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

    # 如实统计每个字段的覆盖率，不掩盖数据质量问题
    field_coverage = {f: 0 for f in FIELD_FALLBACKS}
    for t, r in cache.items():
        for f in field_coverage:
            if f in r:
                field_coverage[f] += 1
    print("各字段覆盖率:")
    for f, n in field_coverage.items():
        print(f"  {f}: {n}/{len(cache)} ({n/len(cache):.1%})")

    return cache


if __name__ == "__main__":
    import shutil
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from us_stock_screener import SECTOR_MAP

    backup_path = OUTPUT_PATH.replace(".json", "_PRE_TAG_MERGE_FIX_BACKUP.json")
    if os.path.exists(OUTPUT_PATH) and not os.path.exists(backup_path):
        shutil.copy(OUTPUT_PATH, backup_path)
        print(f"已备份原文件(标签合并修复前)到 {backup_path}")

    cache = build_cache(list(SECTOR_MAP.keys()))
    with open(OUTPUT_PATH, "w") as f:
        json.dump(cache, f)
    print(f"已写入 {OUTPUT_PATH}")
