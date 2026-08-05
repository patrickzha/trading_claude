"""
Cycle19小修小补(3+4)：验证第94条的机制推断——"宏观VIX熔断已经抢先生效，
导致账户级回撤熔断在默认配置下几乎从不触发"——不能只靠"关掉账户熔断结果
不变"这一个间接证据，直接测：关掉宏观熔断(enable_macro_filter=False)后，
账户级熔断是否真的会开始触发？同时测"两层熔断全部关掉、只剩逐票止损"这个
最坏情况下2020年新冠崩盘的真实回撤有多深，把#94的分层保护结构量化完整。
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

CONFIGS = {
    "宏观熔断开+账户熔断开(默认,第94条baseline)": {},
    "宏观熔断关+账户熔断开": {"enable_macro_filter": False},
    "宏观熔断关+账户熔断关(只剩逐票止损)": {"enable_macro_filter": False, "enable_account_circuit_breaker": False},
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
    print(" Cycle19小修小补：验证宏观熔断是否预占账户熔断的触发空间(COVID窗口)")
    print("=" * 70)

    with open(HIST_CACHE, "rb") as f:
        full = pickle.load(f)
    win = {k: full[k].loc[START:END] for k in ["c", "o", "h", "l", "v", "vix"]}
    print(f"窗口: {win['c'].index[0].date()} ~ {win['c'].index[-1].date()}")

    with open(EARNINGS_CACHE, "rb") as f:
        earnings_calendar = pickle.load(f)

    for cname, kwargs in CONFIGS.items():
        print(f"\n--- {cname} ({kwargs}) ---")
        res = run_walk_forward_backtest(
            win["c"], win["o"], win["h"], win["l"], win["v"], win["vix"],
            enable_shap=False, n_workers=N_WORKERS, xgb_n_jobs=1, earnings_calendar=earnings_calendar,
            **kwargs,
        )
        m = rough_metrics(res)
        print(f"  {cname}: {m}")

    print(f"\n总耗时: {(time.time()-t0)/60:.1f}分钟")


if __name__ == "__main__":
    main()
