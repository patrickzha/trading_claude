"""
Cycle18大修大补②：统一的每周实盘入口 + 这个项目一直缺的一环——闭环真实
表现追踪。

现状(这次整理前)：`generate_weekly_signal.py`只覆盖动量腿；
`paper_trading_log.py`覆盖三条腿的候选但只记录当次候选，从来没有回头去看
"上次记录的候选后来到底涨跌了多少"——两个脚本都只往前生成信号，没有
闭环。`paper_trading_log.py`自己的文档字符串写着"这是唯一真正意义上的
样本外验证"，但代码从来没有真正兑现这句话，因为它压根不检查历史准确性。

这个脚本做三件事：
1. 用真实实时数据(不是历史缓存)生成本周三条腿的候选 + 动态债券对冲建议
   (full_system.py的逻辑，用live ^TNX/IEF数据)。
2. 闭环验证：读取上一次运行记录的候选(decision_date + 三条腿ticker列表)，
   用这次拉到的真实价格反推上次那批候选到下一个交易周实际涨跌了多少，
   算出动量/价值/低波动/组合(等权)各自的已实现收益，累加成一条真正的、
   逐周推进、不能回测重来的样本外净值曲线(realized_combo_nav)。
3. 把这周的新候选 + 上周的已实现表现都写进weekly_full_report_log.csv，
   同一个文件里同时看到"预测"和"验证"两部分，不会脱节。

用法：python weekly_full_report.py（建议每周一收盘后或周二早上跑，
配合cron）。第一次运行没有历史记录可对照，只生成本周候选，不产生已实现
收益这一行；从第二次运行开始才会出现"上周已实现表现"这部分。
"""
from __future__ import annotations

import csv
import json
import os
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from us_stock_screener import get_ai_weekly_picks
from value_investing import select_value_portfolio
from low_vol_investing import select_low_vol_portfolio
from backtest import fetch_earnings_calendar

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weekly_full_report_log.csv")
EARNINGS_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".earnings_calendar_cache.pkl")
EARNINGS_CACHE_DAYS = 7
RATE_TREND_THRESHOLD = 0.3
TREND_WINDOW = 60
BOND_WEIGHT_DEFAULT = 0.4
BOND_WEIGHT_HIKE = 0.1

SECTOR_TO_GICS = {
    "Healthcare": "HEALTH", "Energy": "ENERGY", "Financial Services": "FIN",
    "Consumer Cyclical": "CONS_D", "Technology": "TECH", "Consumer Defensive": "CONS_S",
    "Industrials": "INDU", "Basic Materials": "MATER", "Real Estate": "REAL",
    "Utilities": "UTIL", "Communication Services": "COMM",
}


def _load_or_fetch_earnings_calendar(tickers):
    if os.path.exists(EARNINGS_CACHE_PATH):
        age_days = (datetime.now().timestamp() - os.path.getmtime(EARNINGS_CACHE_PATH)) / 86400
        if age_days < EARNINGS_CACHE_DAYS:
            with open(EARNINGS_CACHE_PATH, "rb") as f:
                return pickle.load(f)
    calendar = fetch_earnings_calendar(tickers, n_workers=16)
    with open(EARNINGS_CACHE_PATH, "wb") as f:
        pickle.dump(calendar, f)
    return calendar


def fetch_live_universe():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sp500_universe.json")) as f:
        universe = json.load(f)
    tickers = list(universe["tickers"].keys())
    sector_map = {t: SECTOR_TO_GICS.get(info["sector"], "OTHER") for t, info in universe["tickers"].items()}

    end = datetime.now()
    start = end - timedelta(days=730)
    data = yf.download(tickers, start=start, end=end, progress=False, group_by="ticker", auto_adjust=True)
    vix_data = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
    vix = vix_data["Close"].squeeze() if hasattr(vix_data["Close"], "squeeze") else vix_data["Close"]

    close_dict, open_dict, high_dict, low_dict, vol_dict = {}, {}, {}, {}, {}
    for t in tickers:
        try:
            df_t = data[t].dropna()
            if len(df_t) > 300:
                close_dict[t] = df_t["Close"]
                open_dict[t] = df_t["Open"]
                high_dict[t] = df_t["High"]
                low_dict[t] = df_t["Low"]
                vol_dict[t] = df_t["Volume"]
        except Exception:
            continue

    c_df = pd.DataFrame(close_dict)
    o_df = pd.DataFrame(open_dict)
    h_df = pd.DataFrame(high_dict)
    l_df = pd.DataFrame(low_dict)
    v_df = pd.DataFrame(vol_dict)
    return c_df, o_df, h_df, l_df, v_df, vix, sector_map


def fetch_bond_recommendation(equity_index):
    data = yf.download(["IEF", "^TNX"], start=(datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d"),
                       end=datetime.now().strftime("%Y-%m-%d"), progress=False, group_by="ticker", auto_adjust=True)
    tnx = data["^TNX"]["Close"].dropna()
    trend = float(tnx.diff(TREND_WINDOW).iloc[-1])
    bond_weight = BOND_WEIGHT_HIKE if trend > RATE_TREND_THRESHOLD else BOND_WEIGHT_DEFAULT
    return bond_weight, trend


def realized_return_for_picks(tickers, c_df, decision_date_str, as_of_date):
    """用真实价格算一组票从decision_date到as_of_date(或更早的最近可用日)的等权已实现收益。"""
    if not tickers or decision_date_str not in c_df.index.strftime("%Y-%m-%d").tolist():
        return None, 0
    entry_idx = c_df.index.get_loc(pd.Timestamp(decision_date_str))
    exit_idx = min(entry_idx + 5, len(c_df.index) - 1)  # 持仓一周(约5个交易日)
    rets = []
    for t in tickers:
        if t not in c_df.columns:
            continue
        try:
            p0 = c_df[t].iloc[entry_idx]
            p1 = c_df[t].iloc[exit_idx]
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                rets.append(p1 / p0 - 1)
        except Exception:
            continue
    if not rets:
        return None, 0
    return float(np.mean(rets)), len(rets)


def main():
    print("=" * 70)
    print(" 每周完整系统报告(三条腿 + 动态债券对冲 + 上周真实表现闭环验证)")
    print("=" * 70)
    print(f" 运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("拉取真实实时数据(2年窗口)...")
    c_df, o_df, h_df, l_df, v_df, vix, sector_map = fetch_live_universe()
    decision_date = c_df.index[-2]
    decision_date_str = decision_date.strftime("%Y-%m-%d")
    print(f"决策日: {decision_date_str}（数据最新日期: {c_df.index[-1].date()}）\n")

    # ---------- 闭环验证：先看上一次记录的候选后来表现如何 ----------
    prior_row = None
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            prior_row = rows[-1]

    if prior_row is not None and prior_row["decision_date"] != decision_date_str:
        print("=" * 70)
        print(f" 上周闭环验证(decision_date={prior_row['decision_date']}的候选，真实后续表现)")
        print("=" * 70)
        leg_realized = {}
        for leg in ["momentum", "value", "low_vol"]:
            tickers = [t for t in prior_row.get(f"{leg}_picks", "").split("|") if t and not t.startswith("ERROR")]
            ret, n = realized_return_for_picks(tickers, c_df, prior_row["decision_date"], decision_date_str)
            leg_realized[leg] = ret
            if ret is not None:
                print(f"  {leg}腿: {n}只票等权已实现收益 = {ret*100:+.2f}%")
            else:
                print(f"  {leg}腿: 无法计算(票不在当前数据里或数据不足)")
        valid_rets = [r for r in leg_realized.values() if r is not None]
        combo_ret = float(np.mean(valid_rets)) if valid_rets else None
        if combo_ret is not None:
            print(f"  三腿等权组合已实现收益 = {combo_ret*100:+.2f}%")
        prior_cum = float(prior_row.get("cumulative_realized_nav", 1.0) or 1.0)
        new_cum = prior_cum * (1 + combo_ret) if combo_ret is not None else prior_cum
        print(f"  累计已实现净值(从第一次有闭环数据开始): {new_cum:.4f}")
    else:
        leg_realized = {"momentum": None, "value": None, "low_vol": None}
        combo_ret = None
        new_cum = 1.0
        print("首次运行或本周已经跑过，没有新的闭环数据可验证。\n")

    # ---------- 本周新候选 ----------
    print("\n拉取/复用财报日历...")
    earnings_calendar = _load_or_fetch_earnings_calendar(list(c_df.columns))

    print("\n" + "=" * 70)
    print(" 本周三条腿候选")
    print("=" * 70)
    try:
        mom_picks_raw = get_ai_weekly_picks(
            c_df, o_df, h_df, l_df, v_df, vix, decision_date,
            sector_map=sector_map, earnings_calendar=earnings_calendar,
            xgb_n_jobs=max(1, (os.cpu_count() or 1) - 1),
        )
        mom_tickers = [p[0] for p in mom_picks_raw]
        print(f" 动量腿: {mom_tickers if mom_tickers else '空仓(宏观熔断或无合格候选)'}")
    except Exception as e:
        mom_tickers = [f"ERROR:{type(e).__name__}"]
        print(f" 动量腿: [错误] {e}")

    val_tickers = select_value_portfolio(c_df, list(c_df.columns), decision_date, top_n=30)
    print(f" 价值腿(前10/{len(val_tickers)}只): {val_tickers[:10]}")

    lv_tickers = select_low_vol_portfolio(c_df, list(c_df.columns), decision_date, top_n=30)
    print(f" 低波动腿(前10/{len(lv_tickers)}只): {lv_tickers[:10]}")

    print("\n" + "=" * 70)
    print(" 动态债券对冲建议(full_system.py逻辑，live ^TNX/IEF数据)")
    print("=" * 70)
    try:
        bond_weight, trend = fetch_bond_recommendation(c_df.index)
        print(f" 60日10年期国债收益率趋势变化: {trend:+.3f}百分点 (阈值{RATE_TREND_THRESHOLD}pp)")
        print(f" 建议债券(IEF)敞口: {bond_weight:.0%}   股票敞口: {1-bond_weight:.0%}")
    except Exception as e:
        bond_weight = None
        print(f" [错误] {type(e).__name__}: {e}")

    print("\n" + "=" * 70)
    print(" 如实提醒：三条腿等权配置(README第55/56/57条)是这次会话测试过的所有")
    print(" 方向里证据最扎实的，但不是已证实的alpha来源；动态债券对冲(第65-69条)")
    print(" 是跨资产类别分散化里证据最强的部分，但在加息周期等特定环境下会失效。")
    print(" 本周实际表现要等下次运行才能在上面的'闭环验证'部分看到。")

    # ---------- 写日志 ----------
    row = {
        "log_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "decision_date": decision_date_str,
        "momentum_picks": "|".join(mom_tickers),
        "value_picks": "|".join(val_tickers),
        "low_vol_picks": "|".join(lv_tickers),
        "bond_weight_suggestion": f"{bond_weight:.2f}" if bond_weight is not None else "",
        "realized_momentum_return": f"{leg_realized['momentum']:.6f}" if leg_realized.get("momentum") is not None else "",
        "realized_value_return": f"{leg_realized['value']:.6f}" if leg_realized.get("value") is not None else "",
        "realized_low_vol_return": f"{leg_realized['low_vol']:.6f}" if leg_realized.get("low_vol") is not None else "",
        "realized_combo_return": f"{combo_ret:.6f}" if combo_ret is not None else "",
        "cumulative_realized_nav": f"{new_cum:.6f}",
    }
    file_exists = os.path.exists(LOG_PATH)
    if prior_row is None or prior_row["decision_date"] != decision_date_str:
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"\n已追加写入 {LOG_PATH}")
    else:
        print(f"\n本周已经记录过(decision_date={decision_date_str})，不重复写入。")


if __name__ == "__main__":
    main()
