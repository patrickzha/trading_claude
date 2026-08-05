"""
Cycle35大修大补②(框架改动，新仓位sizing哲学)：Kelly准则动态仓位sizing——
用trailing窗口估计的均值/方差算出Kelly最优仓位比例f*=mu/sigma^2(clip到
[0,1]，不加杠杆，只在"满仓"和"现金"之间动态调整)，代替现有固定仓位或
`use_vol_targeting`(第90/91/97条，只降不加的启发式波动率目标)。Kelly
准则优化的是长期对数收益增长率，是有严格数学推导的仓位sizing框架，跟
vol_targeting的启发式"波动率越高、敞口越低"不是同一套理论基础，值得
独立验证。

应用在动量腿(已有236周现成数据，不用重新训练)。每21个交易日(约1个月)
重新估计一次trailing 126个交易日(约6个月)的日收益率均值/方差。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
LOOKBACK_WEEKS = 26
REBALANCE_EVERY_WEEKS = 4
MAX_KELLY_FRACTION = 1.0


def true_mdd(nav):
    nav = np.array(nav)
    peak = np.maximum.accumulate(nav)
    return float(np.min((nav - peak) / peak))


def metrics(weekly_rets):
    rets = np.array(weekly_rets)
    nav = np.cumprod(1 + rets)
    n_years = len(rets) / 52
    cagr = float(nav[-1] ** (1 / n_years) - 1) if n_years > 0 else None
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(52))
    mdd = true_mdd(nav)
    return cagr, sharpe, mdd


def kelly_sizing(weekly_rets: list[float]) -> list[float]:
    rets = np.array(weekly_rets)
    n = len(rets)
    sized_rets = []
    current_f = 1.0
    for i in range(n):
        if i > 0 and i % REBALANCE_EVERY_WEEKS == 0 and i >= LOOKBACK_WEEKS:
            window = rets[i - LOOKBACK_WEEKS:i]
            mu, sigma2 = window.mean(), window.var()
            if sigma2 > 1e-10:
                f = mu / sigma2
                current_f = float(np.clip(f, 0.0, MAX_KELLY_FRACTION))
            else:
                current_f = 1.0
        sized_rets.append(rets[i] * current_f)
    return sized_rets


def main():
    with open(f"{SCRATCH}/default_full_results_cache.pkl", "rb") as f:
        mom_results_2022 = pickle.load(f)
    with open(f"{SCRATCH}/momentum_2014_2020_results_cache.pkl", "rb") as f:
        mom_results_2014 = pickle.load(f)
    with open(f"{SCRATCH}/momentum_2009_2014_results_cache.pkl", "rb") as f:
        mom_results_2009 = pickle.load(f)

    periods = [
        ("2022-2026", mom_results_2022),
        ("2014-2020", mom_results_2014),
        ("2009-2014", mom_results_2009),
    ]

    for label, mom_results in periods:
        weekly_rets = [r["weekly_return"] for r in mom_results]
        b_cagr, b_sharpe, b_mdd = metrics(weekly_rets)

        sized_rets = kelly_sizing(weekly_rets)
        k_cagr, k_sharpe, k_mdd = metrics(sized_rets)

        print(f"\n=== {label} ===")
        print(f"  基线(满仓，无Kelly sizing): CAGR={b_cagr*100:.2f}%, Sharpe={b_sharpe:.3f}, MaxDD={b_mdd*100:.2f}%")
        print(f"  +Kelly准则动态sizing: CAGR={k_cagr*100:.2f}%, Sharpe={k_sharpe:.3f}, MaxDD={k_mdd*100:.2f}%")


if __name__ == "__main__":
    main()
