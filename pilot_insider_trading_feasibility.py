"""
可行性试点：SEC Form 4内幕交易数据(高管/董事买卖自家股票)——这是这次
会话要找的"真正跟公司基本面比率无关的独立信息来源"的候选之一。这次不
直接上全量500只票的历史抓取(规模远大于之前任何一次SEC数据抓取，光是
苹果一家公司"recent"窗口里就有587条Form 4记录，500只票乘上去，外加
每条记录都要单独抓一次XML解析，量级可能是几万到十几万次请求，SEC限速
10次/秒，全量抓取可能要几个小时)，先用10只票做一次小规模试点，确认
数据管道走得通、看一眼真实数据长什么样，再决定是否值得投入全量抓取。
"""
from __future__ import annotations

import re
import time
from xml.etree import ElementTree as ET

import requests

HEADERS = {"User-Agent": "trading-research contact@example.com"}
PILOT_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "JPM", "XOM", "PFE", "WMT", "DIS", "BA"]


def get_cik_map():
    resp = requests.get("https://www.sec.gov/files/company_tickers.json", headers=HEADERS, timeout=20)
    data = resp.json()
    return {v["ticker"]: str(v["cik_str"]).zfill(10) for v in data.values()}


def get_form4_filings(cik: str, max_n: int = 30):
    resp = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=HEADERS, timeout=20)
    if resp.status_code != 200:
        return []
    d = resp.json()
    recent = d["filings"]["recent"]
    forms = recent["form"]
    out = []
    for i, f in enumerate(forms):
        if f == "4":
            out.append({
                "date": recent["filingDate"][i],
                "accession": recent["accessionNumber"][i].replace("-", ""),
                "doc": recent["primaryDocument"][i],
            })
        if len(out) >= max_n:
            break
    return out


def parse_form4_xml(cik_nozero: str, accession: str, doc: str):
    # primaryDocument指向XSLT渲染后的HTML展示页(比如xslF345X06/form4.xml，
    # 后缀是.xml但内容其实是HTML，直接解析会静默失败)——试点第一版踩了这个
    #坑，导致0笔交易的假阴性结果。真正的原始XML在accession目录根下，
    # 文件名去掉子目录前缀即可。
    doc_filename = doc.rsplit("/", 1)[-1]
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_nozero}/{accession}/{doc_filename}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return []
        root = ET.fromstring(resp.content)
        out = []
        for txn in root.findall(".//nonDerivativeTransaction"):
            code_el = txn.find(".//transactionCode")
            code = code_el.text if code_el is not None else None
            if code not in ("P", "S"):
                continue
            shares_el = txn.find(".//transactionShares/value")
            date_el = txn.find(".//transactionDate/value")
            shares = float(shares_el.text) if shares_el is not None else None
            date = date_el.text if date_el is not None else None
            out.append({"code": code, "shares": shares, "date": date})
        return out
    except Exception as e:
        return [{"error": str(e)}]


def main():
    print("拉取ticker->CIK映射...")
    cik_map = get_cik_map()

    total_filings_checked = 0
    total_p_s_txns = 0
    t0 = time.time()

    for ticker in PILOT_TICKERS:
        cik = cik_map.get(ticker)
        if not cik:
            print(f"{ticker}: 找不到CIK")
            continue
        filings = get_form4_filings(cik, max_n=15)
        print(f"\n{ticker}: 最近{len(filings)}条Form 4记录(只看recent窗口，不翻历史分页)")
        p_count, s_count = 0, 0
        for filing in filings[:8]:  # 每只票只深入解析前8条，控制试点规模
            cik_nozero = str(int(cik))
            txns = parse_form4_xml(cik_nozero, filing["accession"], filing["doc"])
            total_filings_checked += 1
            for t in txns:
                if "error" in t:
                    continue
                total_p_s_txns += 1
                if t["code"] == "P":
                    p_count += 1
                elif t["code"] == "S":
                    s_count += 1
            time.sleep(0.15)  # 控制请求速率，SEC建议不超过10次/秒
        print(f"  买入(P)交易: {p_count}笔, 卖出(S)交易: {s_count}笔")

    print(f"\n试点完成: 检查了{total_filings_checked}份filing, 提取到{total_p_s_txns}笔P/S交易")
    print(f"耗时: {time.time()-t0:.1f}秒")
    print(f"\n规模外推: 全量500只票 x 假设每只票平均需要拉30-50份filing(试点只拉了每只票recent窗口的前15条中的8条)")
    print(f"= 15000-25000次filing请求，按这次试点的速率({total_filings_checked}份/{time.time()-t0:.1f}秒)外推，全量大概需要{(15000/(total_filings_checked/(time.time()-t0)))/60:.0f}-{(25000/(total_filings_checked/(time.time()-t0)))/60:.0f}分钟")


if __name__ == "__main__":
    main()
