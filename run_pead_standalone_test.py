"""
PEAD(财报后动量漂移)独立信号测试——不整合进主策略，先看这个信号单独有没有
edge。逻辑：每周一，找出最近若干个交易日内公布过财报、且盈利超预期(Surprise%
超过阈值)的票，按超预期幅度排序，选前3只，跟主策略一样的方式模拟持仓（周一
开盘买、周五收盘卖，2%止损），对比SPY基准。

第一版最宽松参数（阈值5%、不带熔断）结果很差（CAGR 5.82% vs SPY 12.71%，
最大回撤-35.59%，胜率38.8%）——但没有熔断保护本身就不公平比较（主策略的
正收益主要靠熔断规则），这里再测一版"更严格阈值+带上跟主策略同一套宏观熔断"
的版本，排除"纯粹因为没熔断保护"这个混淆因素，看PEAD信号本身到底有没有
底子。如果这个信号本身就没有edge，就没有必要再往下做"整合成卫星仓位"这一步。

不需要改 us_stock_screener.py / backtest.py 任何代码，只复用
simulate_weekly_return / get_weekly_trading_dates / check_macro_warning
三个纯函数。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from backtest import simulate_weekly_return, get_weekly_trading_dates, STOP_LOSS_PCT, CASH_YIELD
from us_stock_screener import check_macro_warning

DATA_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/sp500_data_cache.pkl"
PEAD_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/pead_data_cache.pkl"

MAX_POS = 3


def run_pead_backtest(surprise_threshold, lookback_days, use_macro_filter,
                      c_df, o_df, h_df, l_df, vix, spy_close, ticker_events):
    week_pairs = get_weekly_trading_dates(c_df.index)
    cash_yield_weekly = (1 + CASH_YIELD) ** (1 / 52) - 1

    results = []
    total_weeks_with_candidates = 0
    for monday, friday in week_pairs:
        loc = c_df.index.get_loc(monday)
        if loc == 0:
            continue
        decision_date = c_df.index[loc - 1]

        if use_macro_filter:
            is_danger, _ = check_macro_warning(c_df, vix, decision_date)
            if is_danger:
                results.append({"date": monday.strftime("%Y-%m-%d"), "weekly_return": cash_yield_weekly})
                continue

        candidates = []
        for ticker, df in ticker_events.items():
            recent = df[(df["date"] <= decision_date) &
                       (df["date"] >= decision_date - pd.Timedelta(days=lookback_days * 1.5))]
            if len(recent) == 0:
                continue
            latest = recent.iloc[-1]
            if latest["surprise_pct"] >= surprise_threshold:
                candidates.append((ticker, latest["surprise_pct"]))

        if not candidates:
            weekly_return = cash_yield_weekly
        else:
            total_weeks_with_candidates += 1
            candidates.sort(key=lambda x: -x[1])
            picks = candidates[:MAX_POS]
            rets = []
            for ticker, _surprise in picks:
                _, net_ret = simulate_weekly_return(
                    ticker, monday, friday, c_df, o_df, h_df, l_df,
                    stop_loss_pct=STOP_LOSS_PCT,
                )
                rets.append(net_ret)
            weekly_return = float(np.mean(rets)) if rets else cash_yield_weekly

        results.append({"date": monday.strftime("%Y-%m-%d"), "weekly_return": weekly_return})

    rets = np.array([r["weekly_return"] for r in results])
    nav = np.cumprod(1 + rets)
    n_weeks = len(rets)
    cagr = nav[-1] ** (52 / n_weeks) - 1
    sharpe = rets.mean() / (rets.std() + 1e-9) * np.sqrt(52)
    peak = np.maximum.accumulate(nav)
    mdd = np.min((nav - peak) / peak)
    win_rate = (rets > 0).mean()

    print(f"  阈值={surprise_threshold}% 窗口={lookback_days}天 熔断={use_macro_filter}: "
          f"候选周占比{total_weeks_with_candidates/n_weeks:.1%}  "
          f"CAGR={cagr:.2%}  Sharpe={sharpe:.2f}  最大回撤={mdd:.2%}  周胜率={win_rate:.1%}")


def main():
    with open(DATA_CACHE, "rb") as f:
        full = pickle.load(f)
    c_df, o_df, h_df, l_df, vix, spy_close = full["c"], full["o"], full["h"], full["l"], full["vix"], full.get("spy")
    with open(PEAD_CACHE, "rb") as f:
        pead_cache = pickle.load(f)

    ticker_events = {}
    for ticker, records in pead_cache.items():
        if ticker not in c_df.columns:
            continue
        ticker_events[ticker] = pd.DataFrame(records).sort_values("date")

    if spy_close is not None:
        spy_ret = (spy_close.iloc[-1] / spy_close.iloc[0]) ** (252 / len(spy_close)) - 1
        print(f"SPY基准: CAGR≈{spy_ret:.2%}\n")

    configs = [
        (5.0, 10, False),
        (10.0, 10, False),
        (5.0, 10, True),
        (10.0, 10, True),
        (15.0, 10, True),
    ]
    for threshold, lookback, use_macro in configs:
        run_pead_backtest(threshold, lookback, use_macro, c_df, o_df, h_df, l_df, vix, spy_close, ticker_events)


if __name__ == "__main__":
    main()
