"""
真正的实时纸上交易记录脚本——这是唯一能验证"这套系统在实盘里到底灵不灵"的
机制(回测数字不能代替，见README"现状总结")。跟`generate_current_recommendations.py`
的区别：那个用的是截止某天的缓存数据，做展示/离线复现用；这个每次运行都
用yfinance实时拉取截止当天的数据，生成真正的"如果今天要交易"的候选，并且
会反过来核对上一次记录的候选后续实际涨跌了多少。

用法：
    .conda/bin/python3 run_weekly_paper_trading.py

设计成可以被定时任务(cron/RemoteTrigger)每周自动调用，不需要人工干预：
1. 拉实时数据，生成本周三条腿候选(权重/参数用README"现状总结"里当前
   推荐的旗舰配置：leg_weights=50/40/10 + HRP权重 + 价值腿止损2%/低波动
   腿止损5% + value_rebalance_days=5)。
2. 追加写入 paper_trading_log.csv。
3. 回头核对上一条记录(如果有)的候选，用当时的picks + 当时的decision_date
   + 现在的实时价格，算出"如果照着买了，到今天为止涨跌了多少"，写进
   同一个csv的新增列，不需要额外维护第二份文件。

如实提醒：这个脚本本身不下单，只记录信号；要不要真的照着操作是你自己的
决定。至少要攒够3-6个月、10+条记录才能开始谈"这套系统实盘表现如何"，
单次运行本身没有意义。
"""
from __future__ import annotations

import csv
import os
from datetime import datetime

import pandas as pd
import yfinance as yf

from backtest import fetch_market_data
from us_stock_screener import get_ai_weekly_picks, SECTOR_MAP
from value_investing import select_value_portfolio, _hrp_weights
from low_vol_investing import select_low_vol_portfolio

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_trading_log.csv")
FIELDNAMES = [
    "log_timestamp", "decision_date", "cache_last_date",
    "momentum_picks", "value_picks_top10", "low_vol_picks_top10",
    "note",
    # 回填列：核对上一条记录的实际后续表现(第一条记录没有"上一条"，留空)
    "prev_decision_date", "prev_momentum_ret_pct", "prev_value_ret_pct", "prev_low_vol_ret_pct",
]


def _simple_return(price_df: pd.DataFrame, tickers: list[str], start_date, end_date) -> float | None:
    """等权组合从start_date到end_date的简单收益率(百分比)，用于回填核对。
    只用能同时在两个日期都拿到价格的票，跳过缺数据的(如实标注可能有存活
    偏差，但这里只是核对候选质量的快速检查，不是严谨回测)。"""
    rets = []
    for t in tickers:
        if t not in price_df.columns:
            continue
        try:
            p0 = price_df[t].loc[:start_date].dropna().iloc[-1]
            p1 = price_df[t].loc[:end_date].dropna().iloc[-1]
            rets.append(p1 / p0 - 1)
        except (IndexError, KeyError):
            continue
    if not rets:
        return None
    return float(sum(rets) / len(rets) * 100)


def main():
    print("拉取实时市场数据(yfinance，标普500全量)...")
    c_df, o_df, h_df, l_df, v_df, vix, spy_close = fetch_market_data()

    decision_date = c_df.index[-1]
    print(f"决策日(最新交易日): {decision_date.date()}")

    valid_tickers = list(c_df.columns)

    # ---- 1. 动量腿：本周ML候选 ----
    print("\n生成动量腿候选...")
    try:
        mom_picks_raw = get_ai_weekly_picks(
            c_df, o_df, h_df, l_df, v_df, vix, decision_date,
            sector_map=SECTOR_MAP, enable_earnings_blackout=False,
        )
        momentum_picks = [t for t, *_ in mom_picks_raw] if mom_picks_raw else []
    except Exception as e:
        print(f"  [警告] 动量腿生成失败: {type(e).__name__}: {e}")
        momentum_picks = []
    print(f"  动量腿候选: {momentum_picks}")

    # ---- 2. 价值腿：HRP权重前10 ----
    print("生成价值腿候选(HRP权重)...")
    val_picks = select_value_portfolio(c_df, valid_tickers, decision_date, top_n=30)
    hist_window = c_df[val_picks].loc[:decision_date].tail(61)
    rets_window = hist_window.pct_change().dropna()
    valid_cols = rets_window.columns[rets_window.notna().all()]
    if len(valid_cols) >= 5:
        hrp_w = _hrp_weights(rets_window[valid_cols], max_weight=0.3)
        weights = pd.Series(0.0, index=val_picks)
        for t in valid_cols:
            weights[t] = hrp_w[t]
        weights = weights / weights.sum() if weights.sum() > 0 else pd.Series(1.0 / len(val_picks), index=val_picks)
        value_picks_top10 = weights.sort_values(ascending=False).head(10).index.tolist()
    else:
        value_picks_top10 = val_picks[:10]
    print(f"  价值腿候选(前10): {value_picks_top10}")

    # ---- 3. 低波动腿 ----
    print("生成低波动腿候选...")
    lv_picks = select_low_vol_portfolio(c_df, valid_tickers, decision_date, top_n=30)
    low_vol_picks_top10 = lv_picks[:10]
    print(f"  低波动腿候选(前10): {low_vol_picks_top10}")

    # ---- 回填：核对上一条记录的实际表现 ----
    prev_decision_date = prev_mom_ret = prev_val_ret = prev_lv_ret = ""
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            prev_decision_date = last.get("decision_date", "")
            if prev_decision_date:
                try:
                    prev_dt = pd.Timestamp(prev_decision_date)
                    prev_mom_list = [t for t in last.get("momentum_picks", "").split("|") if t]
                    prev_val_list = [t for t in last.get("value_picks_top10", "").split("|") if t]
                    prev_lv_list = [t for t in last.get("low_vol_picks_top10", "").split("|") if t]
                    r_mom = _simple_return(c_df, prev_mom_list, prev_dt, decision_date)
                    r_val = _simple_return(c_df, prev_val_list, prev_dt, decision_date)
                    r_lv = _simple_return(c_df, prev_lv_list, prev_dt, decision_date)
                    prev_mom_ret = f"{r_mom:.2f}" if r_mom is not None else ""
                    prev_val_ret = f"{r_val:.2f}" if r_val is not None else ""
                    prev_lv_ret = f"{r_lv:.2f}" if r_lv is not None else ""
                    print(f"\n核对上一条记录({prev_decision_date} -> {decision_date.date()})的实际表现:")
                    print(f"  动量腿: {prev_mom_ret}%  价值腿: {prev_val_ret}%  低波动腿: {prev_lv_ret}%")
                except Exception as e:
                    print(f"  [警告] 回填上一条记录失败: {type(e).__name__}: {e}")

    # ---- 写入新记录 ----
    file_exists = os.path.exists(LOG_PATH)
    if file_exists:
        with open(LOG_PATH, newline="", encoding="utf-8") as f:
            existing_header = next(csv.reader(f), [])
        if existing_header != FIELDNAMES:
            raise RuntimeError(
                f"paper_trading_log.csv表头({existing_header})跟当前FIELDNAMES({FIELDNAMES})不一致，"
                "不能直接追加(会导致列错位)——先手动迁移旧数据到新表头，再重新运行。"
            )
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "log_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "decision_date": decision_date.strftime("%Y-%m-%d"),
            "cache_last_date": decision_date.strftime("%Y-%m-%d"),
            "momentum_picks": "|".join(momentum_picks),
            "value_picks_top10": "|".join(value_picks_top10),
            "low_vol_picks_top10": "|".join(low_vol_picks_top10),
            "note": "实时数据生成(旗舰配置leg_weights=50/40/10+HRP+止损2%/5%)，不代表投资建议",
            "prev_decision_date": prev_decision_date,
            "prev_momentum_ret_pct": prev_mom_ret,
            "prev_value_ret_pct": prev_val_ret,
            "prev_low_vol_ret_pct": prev_lv_ret,
        })
    print(f"\n已写入 {LOG_PATH}")


if __name__ == "__main__":
    main()
