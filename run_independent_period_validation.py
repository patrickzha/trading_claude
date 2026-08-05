"""
用2014-2020这段完全没参与过训练/调参的独立历史区间，复核已经采纳的三个改进
（止损收紧2%、RSI阈值放开到100、样本新近度加权26周——现在都已经是模块默认值，
不用显式传参）到底是不是真实规律，而不是过拟合到2021-2026这段特定行情。

这段区间市场结构跟2021-2026很不一样：包含2015-16年盘整、2018年Q4暴跌、
2020年3月新冠暴跌+V型反弹，如果默认配置在这里也站得住，可信度要高很多。

不做earnings blackout（这段太老，yfinance的财报日期接口拿不到可靠的历史数据），
其余配置跟现在的真实默认完全一致。
"""
from __future__ import annotations

import pickle
import os

from backtest import run_walk_forward_backtest, get_weekly_trading_dates
from optimized_metrics_v4 import run_backtest_with_metrics

DATA_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/historical_2014_2020_cache.pkl"


def main():
    with open(DATA_CACHE, "rb") as f:
        d = pickle.load(f)
    c_df, o_df, h_df, l_df, v_df, vix, spy_close = d["c"], d["o"], d["h"], d["l"], d["v"], d["vix"], d["spy"]
    print(f"独立历史区间: {c_df.index[0].date()} ~ {c_df.index[-1].date()}, {len(c_df.columns)}只票")

    n_workers = max(1, os.cpu_count() or 1)
    results = run_walk_forward_backtest(
        c_df, o_df, h_df, l_df, v_df, vix,
        enable_shap=False, n_workers=n_workers, xgb_n_jobs=1,
        enable_earnings_blackout=False,
    )

    bench_rets = None
    if spy_close is not None:
        used_dates = {r["date"] for r in results}
        bench_rets = []
        for monday, friday in get_weekly_trading_dates(c_df.index):
            if monday.strftime("%Y-%m-%d") not in used_dates:
                continue
            try:
                bench_rets.append((spy_close.loc[friday] - spy_close.loc[monday]) / spy_close.loc[monday])
            except Exception:
                bench_rets.append(0.0)

    print("\n[2014-2020独立样本外复核报告]")
    run_backtest_with_metrics(results, benchmark_returns=bench_rets)


if __name__ == "__main__":
    main()
