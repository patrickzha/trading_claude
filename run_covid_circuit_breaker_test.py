"""
Cycle19大修大补①：第87条测账户级回撤熔断参数敏感性时，用的是2021-2026这段
整体偏强的窗口，熔断几乎从未触发，测不出真正的差异——这次换成`historical_
2014_2020_cache.pkl`里已经缓存好的真实数据，专门切出2020年新冠崩盘+V型
反弹这段(2019-08~2020-12，前段留够ROLLING_WINDOW=300个交易日的训练回看
窗口)，这是这次会话里唯一一段"真正快、真正深、又在几个月内反弹回来"的
危机窗口，理论上应该是账户级4周滚动回撤熔断最可能被触发、也最可能看出
参数选择重要性的场景。

注意：跑这个脚本期间不要改us_stock_screener.py/backtest.py。
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np

from backtest import run_walk_forward_backtest

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
HIST_CACHE = os.path.join(SCRATCH, "historical_2014_2020_cache.pkl")
EARNINGS_CACHE = os.path.join(SCRATCH, "earnings_calendar_cache.pkl")

N_WORKERS = max(1, os.cpu_count() or 1)
START = "2019-08-01"
END = "2020-12-31"

CANDIDATES = {
    "threshold_-5%(更敏感)": {"account_dd_threshold": -0.05},
    "threshold_-12%(更迟钝)": {"account_dd_threshold": -0.12},
    "scale_30%(更激进)": {"account_dd_scale": 0.3},
    "scale_70%(更温和)": {"account_dd_scale": 0.7},
    "无熔断(对照组)": {"enable_account_circuit_breaker": False},
}


def rough_metrics(results):
    rets = np.array([r["weekly_return"] for r in results]) if results else np.array([])
    if len(rets) < 2:
        return {"cum": 0.0, "mdd": 0.0, "sharpe": 0.0, "n_weeks": len(rets)}
    nav = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(nav)
    mdd = float(np.min((nav - peak) / peak))
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(52))
    return {"cum": float(nav[-1] - 1), "mdd": mdd, "sharpe": sharpe, "n_weeks": len(rets)}


def main():
    t0 = time.time()
    print("=" * 70)
    print(" Cycle19大修大补①：用2020年新冠崩盘窗口重测账户级回撤熔断参数敏感性")
    print("=" * 70)

    with open(HIST_CACHE, "rb") as f:
        full = pickle.load(f)
    win = {k: full[k].loc[START:END] for k in ["c", "o", "h", "l", "v", "vix"]}
    print(f"窗口: {win['c'].index[0].date()} ~ {win['c'].index[-1].date()}, {len(win['c'])}个交易日")

    with open(EARNINGS_CACHE, "rb") as f:
        earnings_calendar = pickle.load(f)

    print("\n--- baseline(默认-8%/五折) ---")
    base_res = run_walk_forward_backtest(
        win["c"], win["o"], win["h"], win["l"], win["v"], win["vix"],
        enable_shap=False, n_workers=N_WORKERS, xgb_n_jobs=1, earnings_calendar=earnings_calendar,
    )
    base_m = rough_metrics(base_res)
    print(f"  baseline: {base_m}")

    results = {"baseline": base_m}
    for cand_name, kwargs in CANDIDATES.items():
        print(f"\n--- {cand_name} ({kwargs}) ---")
        res = run_walk_forward_backtest(
            win["c"], win["o"], win["h"], win["l"], win["v"], win["vix"],
            enable_shap=False, n_workers=N_WORKERS, xgb_n_jobs=1, earnings_calendar=earnings_calendar,
            **kwargs,
        )
        m = rough_metrics(res)
        print(f"  {cand_name}: {m}")
        results[cand_name] = m

    print(f"\n{'='*20} 汇总 {'='*20}")
    for name, m in results.items():
        delta = m["sharpe"] - base_m["sharpe"] if name != "baseline" else 0.0
        print(f"  {name}: Sharpe={m['sharpe']:.3f} (差值{delta:+.3f}), MDD={m['mdd']*100:.2f}%, n_weeks={m['n_weeks']}")

    print(f"\n总耗时: {(time.time()-t0)/60:.1f}分钟")


if __name__ == "__main__":
    main()
