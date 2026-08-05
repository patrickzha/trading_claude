"""
Cycle18小修小补：财报避让规则(enable_earnings_blackout，默认True)已经实装
好几个session了，但从来没有量化过"这条规则本身到底改善了多少"——只在
README第930行附近提到过数据源失败率高的问题，没有一次干净的开/关对比。
这次用session标准的三窗口初筛方法论测。

注意：跑这个脚本期间不要改us_stock_screener.py/backtest.py。
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np

from backtest import run_walk_forward_backtest

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
DATA_CACHE = os.path.join(SCRATCH, "sp500_data_cache.pkl")
EARNINGS_CACHE = os.path.join(SCRATCH, "earnings_calendar_cache.pkl")

N_WORKERS = max(1, os.cpu_count() or 1)


def rough_metrics(results):
    rets = np.array([r["weekly_return"] for r in results]) if results else np.array([])
    if len(rets) < 2:
        return {"cum": 0.0, "mdd": 0.0, "sharpe": 0.0, "n_weeks": len(rets)}
    nav = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(nav)
    mdd = float(np.min((nav - peak) / peak))
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(52))
    return {"cum": float(nav[-1] - 1), "mdd": mdd, "sharpe": sharpe, "n_weeks": len(rets)}


def slice_window(data, end_idx, window_weeks=25, lookback_days=150):
    start_idx = max(0, end_idx - window_weeks * 5 - lookback_days)
    idx_slice = data["c"].index[start_idx:end_idx]
    return {
        "c": data["c"].loc[idx_slice], "o": data["o"].loc[idx_slice],
        "h": data["h"].loc[idx_slice], "l": data["l"].loc[idx_slice],
        "v": data["v"].loc[idx_slice], "vix": data["vix"].loc[idx_slice],
    }


def run_one(win, earnings_calendar, enable_blackout, n_workers):
    return run_walk_forward_backtest(
        win["c"], win["o"], win["h"], win["l"], win["v"], win["vix"],
        enable_shap=False, n_workers=n_workers, xgb_n_jobs=1,
        earnings_calendar=earnings_calendar,
        enable_earnings_blackout=enable_blackout,
    )


def main():
    t0 = time.time()
    print("=" * 70)
    print(" Cycle18小修小补：财报避让规则(enable_earnings_blackout)开/关效果量化")
    print("=" * 70)

    with open(DATA_CACHE, "rb") as f:
        full = pickle.load(f)
    data = {"c": full["c"], "o": full["o"], "h": full["h"], "l": full["l"],
            "v": full["v"], "vix": full["vix"]}
    n_total = len(data["c"].index)
    print(f"数据总长度: {n_total}天, {data['c'].index[0].date()} ~ {data['c'].index[-1].date()}")

    with open(EARNINGS_CACHE, "rb") as f:
        earnings_calendar = pickle.load(f)

    window_ends = {"近25周": n_total, "中段窗口(约1年前)": n_total - 300, "早段窗口(约2年前)": n_total - 600}

    deltas = []
    for win_name, end_idx in window_ends.items():
        win = slice_window(data, end_idx)
        print(f"\n{'='*20} 窗口: {win_name} {'='*20}")

        on_res = run_one(win, earnings_calendar, True, N_WORKERS)
        on_m = rough_metrics(on_res)
        print(f"  开启财报避让(默认): {on_m}")

        off_res = run_one(win, earnings_calendar, False, N_WORKERS)
        off_m = rough_metrics(off_res)
        print(f"  关闭财报避让: {off_m}")

        d = on_m["sharpe"] - off_m["sharpe"]
        deltas.append(d)
        print(f"  Sharpe差值(开启-关闭) = {d:+.3f}")

    print(f"\n{'='*20} 汇总 {'='*20}")
    print(f"三窗口Sharpe差值(开启-关闭): {[round(d,3) for d in deltas]}")
    same_direction = all(d > 0 for d in deltas) or all(d < 0 for d in deltas)
    if all(d > 0 for d in deltas):
        print("结论: 三窗口一致——开启财报避让更好")
    elif all(d < 0 for d in deltas):
        print("结论: 三窗口一致——开启财报避让更差")
    else:
        print("结论: 方向不一致，视为噪声")
    print(f"\n总耗时: {(time.time()-t0)/60:.1f}分钟")


if __name__ == "__main__":
    main()
