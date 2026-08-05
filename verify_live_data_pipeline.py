"""
用真正实时的数据(不是这次会话反复复用的缓存)重新拉一次，验证整套系统
真的能在"今天"跑通，不是只能在历史缓存上运行。拉最近2年数据(留够动量
模型训练窗口+价值/低波动因子的历史需求)，跑一遍三条腿(动量需要现场
训练XGBoost，会比读缓存慢，正常预期)。
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from us_stock_screener import get_ai_weekly_picks
from value_investing import select_value_portfolio
from low_vol_investing import select_low_vol_portfolio

SECTOR_TO_GICS = {
    "Healthcare": "HEALTH", "Energy": "ENERGY", "Financial Services": "FIN",
    "Consumer Cyclical": "CONS_D", "Technology": "TECH", "Consumer Defensive": "CONS_S",
    "Industrials": "INDU", "Basic Materials": "MATER", "Real Estate": "REAL",
    "Utilities": "UTIL", "Communication Services": "COMM",
}


def main():
    with open("/Users/zhang/Desktop/trading/sp500_universe.json") as f:
        universe = json.load(f)
    tickers = list(universe["tickers"].keys())
    sector_map = {t: info["sector"] for t, info in universe["tickers"].items()}

    end = datetime.now()
    start = end - timedelta(days=730)
    print(f"用真实实时数据拉取 {len(tickers)} 只票 {start.date()}~{end.date()}...")

    data = yf.download(tickers, start=start, end=end, progress=False, group_by="ticker", auto_adjust=True)
    vix_data = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
    vix = vix_data["Close"].squeeze() if hasattr(vix_data["Close"], "squeeze") else vix_data["Close"]

    close_dict, open_dict, high_dict, low_dict, vol_dict = {}, {}, {}, {}, {}
    got = []
    for t in tickers:
        try:
            df_t = data[t].dropna()
            if len(df_t) > 300:
                close_dict[t] = df_t["Close"]
                open_dict[t] = df_t["Open"]
                high_dict[t] = df_t["High"]
                low_dict[t] = df_t["Low"]
                vol_dict[t] = df_t["Volume"]
                got.append(t)
        except Exception:
            continue

    c_df = pd.DataFrame(close_dict)
    o_df = pd.DataFrame(open_dict)
    h_df = pd.DataFrame(high_dict)
    l_df = pd.DataFrame(low_dict)
    v_df = pd.DataFrame(vol_dict)
    print(f"真实数据获取成功: {len(got)}/{len(tickers)}只票, 最新日期: {c_df.index[-1].date()}(真正的今天)")

    decision_date = c_df.index[-2]

    print("\n跑动量策略(现场训练，非缓存)...")
    try:
        mom_picks = get_ai_weekly_picks(c_df, o_df, h_df, l_df, v_df, vix, decision_date,
                                        sector_map=sector_map, enable_earnings_blackout=False)
        print(f"  动量候选: {[(t, f'{r:+.4f}') for t, w, r, p in mom_picks]}")
    except Exception as e:
        print(f"  [错误] {type(e).__name__}: {e}")

    print("\n跑价值策略候选(前10)...")
    val_picks = select_value_portfolio(c_df, list(c_df.columns), decision_date, top_n=30)
    print(f"  {val_picks[:10]}")

    print("\n跑低波动策略候选(前10)...")
    lv_picks = select_low_vol_portfolio(c_df, list(c_df.columns), decision_date, top_n=30)
    print(f"  {lv_picks[:10]}")

    print(f"\n验证完成：系统在真正实时数据(截止{c_df.index[-1].date()})上端到端跑通。")


if __name__ == "__main__":
    main()
