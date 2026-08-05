"""
分批建仓可行性调研——项目只有日线数据，没有分钟级数据，"一天内分批买"这个
最直观的分批建仓做法测不了。唯一能测的版本是"一周内跨天分批"：不是周一
开盘一次性买满，改成周一/周二/周三三天各买1/3（均摊成本价），周五收盘照常
全部卖出、止损规则不变。

这套策略选出来的票本质是动量延续逻辑——选的就是"已经在涨、预期还会继续
涨"的票，跨天分批建仓意味着后面几天用更高的价格买（如果继续涨）或者更低
的价格买（如果回调），是在给动量策略的核心假设增加不确定性，不是明显的
好主意，但值得实测而不是纯靠直觉判断。用现有的OHLCV数据可以模拟，不需要
新数据源。

跟主策略完全独立的诊断脚本，不改 us_stock_screener.py / backtest.py，
复用 get_ai_weekly_picks 选出的候选名单，只是把结算方式从"周一开盘一次性
买入"换成"周一/二/三三天分批买入"两个版本对比。
"""
from __future__ import annotations

import pickle
import os

import numpy as np
import pandas as pd

from backtest import get_weekly_trading_dates, STOP_LOSS_PCT, COST_BI
from us_stock_screener import get_ai_weekly_picks

DATA_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/sp500_data_cache.pkl"
EARNINGS_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/earnings_calendar_cache.pkl"


def simulate_staggered(ticker, monday, friday, close_df, open_df, high_df, low_df,
                       stop_loss_pct=STOP_LOSS_PCT, n_tranches=3):
    """
    跟 simulate_weekly_return 逻辑一致（止损检查从买入次日开始、逐日检查），
    区别是买入价换成"前 n_tranches 个交易日开盘价的均价"，止损价也跟着这个
    均价算，止损检查从第 n_tranches 天之后开始（分批还没买完之前不设止损，
    避免"还没建好仓就被打止损"这种不合理情况）。
    """
    try:
        week_days = close_df.loc[monday:friday, ticker].index
        if len(week_days) < n_tranches + 1:
            return 0.0
        tranche_days = week_days[:n_tranches]
        tranche_prices = open_df.loc[tranche_days, ticker]
        if tranche_prices.isna().any() or (tranche_prices <= 0).any():
            return 0.0
        buy_price = float(tranche_prices.mean())
        stop_price = buy_price * (1 - stop_loss_pct)

        remaining_days = week_days[n_tranches:]
        for day in remaining_days:
            day_low = low_df.loc[day, ticker] if day in low_df.index else np.nan
            if pd.notna(day_low) and day_low <= stop_price:
                return (stop_price - buy_price) / buy_price - COST_BI * 0.5

        sell_price = close_df.loc[friday, ticker]
        if pd.isna(sell_price) or sell_price <= 0:
            return 0.0
        return (sell_price - buy_price) / buy_price - COST_BI
    except Exception:
        return 0.0


def main():
    with open(DATA_CACHE, "rb") as f:
        full = pickle.load(f)
    c_df, o_df, h_df, l_df, v_df, vix = full["c"], full["o"], full["h"], full["l"], full["v"], full["vix"]
    with open(EARNINGS_CACHE, "rb") as f:
        earnings_calendar = pickle.load(f)

    # 只测最近~50周（诊断性质，不需要跑满5年），复用主模型选股逻辑，
    # 分别用"周一一次性买入" vs "周一二三分批买入"结算同一批候选，对比差异。
    week_pairs = get_weekly_trading_dates(c_df.index)[-25:-5]
    n_jobs = max(1, (os.cpu_count() or 1) - 1)

    onetime_rets, staggered_rets = [], []
    for i, (monday, friday) in enumerate(week_pairs):
        loc = c_df.index.get_loc(monday)
        if loc == 0:
            continue
        decision_date = c_df.index[loc - 1]
        picks = get_ai_weekly_picks(
            c_df, o_df, h_df, l_df, v_df, vix, decision_date,
            shap_enabled=False, earnings_calendar=earnings_calendar, xgb_n_jobs=n_jobs,
        )
        if not picks:
            continue
        week_onetime, week_staggered = [], []
        for ticker, weight, pred_ret, pos_size in picks:
            buy_price = o_df.loc[monday, ticker] if monday in o_df.index else None
            if buy_price is None or pd.isna(buy_price) or buy_price <= 0:
                continue
            stop_price = buy_price * (1 - STOP_LOSS_PCT)
            week_days = c_df.loc[monday:friday, ticker].index
            onetime_ret = 0.0
            for day in week_days[1:]:
                day_low = l_df.loc[day, ticker] if day in l_df.index else np.nan
                if pd.notna(day_low) and day_low <= stop_price:
                    onetime_ret = (stop_price - buy_price) / buy_price - COST_BI * 0.5
                    break
            else:
                sell_price = c_df.loc[friday, ticker]
                onetime_ret = (sell_price - buy_price) / buy_price - COST_BI if pd.notna(sell_price) and sell_price > 0 else 0.0
            week_onetime.append(onetime_ret * weight * pos_size)

            staggered_ret = simulate_staggered(ticker, monday, friday, c_df, o_df, h_df, l_df)
            week_staggered.append(staggered_ret * weight * pos_size)

        if week_onetime:
            onetime_rets.append(sum(week_onetime))
            staggered_rets.append(sum(week_staggered))
        print(f"  第{i+1}周 {monday.date()}: 一次性={sum(week_onetime):.4f}  分批={sum(week_staggered):.4f}")

    onetime_rets, staggered_rets = np.array(onetime_rets), np.array(staggered_rets)
    print(f"\n一次性买入: 累计={np.prod(1+onetime_rets)-1:.2%}  Sharpe(粗)={onetime_rets.mean()/(onetime_rets.std()+1e-9)*np.sqrt(52):.2f}")
    print(f"分批买入(周一二三): 累计={np.prod(1+staggered_rets)-1:.2%}  Sharpe(粗)={staggered_rets.mean()/(staggered_rets.std()+1e-9)*np.sqrt(52):.2f}")


if __name__ == "__main__":
    main()
