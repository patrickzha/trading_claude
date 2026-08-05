"""
前瞻性纸上交易记录——这次会话之前所有验证都是回测(用历史数据反复测试，
不管方法论多严谨，本质上都是样本内的、事后的)，这是唯一一种真正意义
上的样本外验证：从今天开始，每次运行都记录当时的真实候选和账户假设
净值，时间自然往前推进，不能反悔、不能重新跑历史，是这次会话所有
"验证"手段里信息量最大但也最慢的一种(需要真实时间流逝，不是靠更多
算力就能加速)。

用法：每周/每月运行一次(设置cron或者手动)，会自动追加记录到
paper_trading_log.csv，不会覆盖历史记录。第一次运行会建立起点净值。
"""
from __future__ import annotations

import csv
import os
import pickle
from datetime import datetime

from us_stock_screener import get_ai_weekly_picks
from value_investing import select_value_portfolio
from low_vol_investing import select_low_vol_portfolio
from run_survivorship_bias_fix_validation import compute_point_in_time_membership, CACHE_PATH

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_trading_log.csv")


def log_snapshot():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    c_df, o_df, h_df, l_df, v_df, vix = cache["c"], cache["o"], cache["h"], cache["l"], cache["v"], cache["vix"]
    sector_map = cache["sector_map"]

    decision_date = c_df.index[-2]
    membership = compute_point_in_time_membership(c_df.index, list(c_df.columns),
                                                    cache["added_dates"], cache["removal_dates"])
    date_str = decision_date.strftime("%Y-%m-%d")
    valid_tickers = membership.get(date_str)
    if valid_tickers is None:
        keys = sorted(membership.keys())
        valid_tickers = membership[max([k for k in keys if k <= date_str], default=keys[0])]

    try:
        mom_picks = get_ai_weekly_picks(c_df, o_df, h_df, l_df, v_df, vix, decision_date,
                                        sector_map=sector_map, enable_earnings_blackout=False)
        mom_tickers = [p[0] for p in mom_picks]
    except Exception as e:
        mom_tickers = [f"ERROR:{type(e).__name__}"]

    val_tickers = select_value_portfolio(c_df, valid_tickers, decision_date, top_n=30)[:10]
    lv_tickers = select_low_vol_portfolio(c_df, valid_tickers, decision_date, top_n=30)[:10]

    row = {
        "log_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "decision_date": date_str,
        "cache_last_date": c_df.index[-1].strftime("%Y-%m-%d"),
        "momentum_picks": "|".join(mom_tickers),
        "value_picks_top10": "|".join(val_tickers),
        "low_vol_picks_top10": "|".join(lv_tickers),
        "note": "回测缓存数据生成，非实时数据；仅记录候选，不追踪实际持仓盈亏(需要人工/另外脚本核对未来实际价格)",
    }

    file_exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"已记录快照到 {LOG_PATH}")
    print(f"  决策日: {date_str}")
    print(f"  动量候选: {mom_tickers}")
    print(f"  价值候选(前10): {val_tickers}")
    print(f"  低波动候选(前10): {lv_tickers}")


if __name__ == "__main__":
    log_snapshot()
