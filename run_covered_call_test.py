"""
Cycle67大修大补②：备兑看涨期权(covered call)——每月卖出虚值看涨期权收取
权利金，代价是放弃超过行权价之后的上涨空间。跟第296/297条的保护性看跌/
看跌价差正好是相反的机制方向：那两条是"付钱买保护"，这次是"收钱、放弃部分
上涨"，理论上应该在震荡/温和上涨市场里提升Sharpe(多一份稳定的权利金收入)，
在强趋势上涨市场里拖累收益(错过大涨)，直接测试这个预期是否成立。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from scipy.stats import norm

from full_system import run_full_system
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"

PERIODS = [
    ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl",
     f"{SCRATCH}/default_full_results_cache.pkl", True),
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl",
     f"{SCRATCH}/momentum_2014_2020_results_cache.pkl", False),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl",
     f"{SCRATCH}/momentum_2009_2014_results_cache.pkl", False),
]

BASE = dict(value_weighting="hrp", value_rebalance_days=5,
            value_stop_loss_pct=0.03, low_vol_stop_loss_pct=0.05)

R_FREE = 0.04
MONTH_DAYS = 21


def bs_call_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def load_period(price_cache_path, mom_cache_path, is_pit):
    with open(price_cache_path, "rb") as f:
        cache = pickle.load(f)
    price_df, spy_close = cache["c"], cache.get("spy")
    vix = cache.get("vix")
    with open(mom_cache_path, "rb") as f:
        mom_results = pickle.load(f)
    if is_pit:
        membership = compute_point_in_time_membership(
            price_df.index, list(price_df.columns), cache["added_dates"], cache["removal_dates"])
    else:
        membership = {price_df.index[i].strftime("%Y-%m-%d"): list(price_df.columns) for i in range(len(price_df))}
    return price_df, spy_close, mom_results, membership, vix


def apply_covered_call(equity_nav: pd.Series, vix: pd.Series, otm_pct: float, notional_frac: float):
    idx = equity_nav.index
    n = len(idx)
    vix_aligned = vix.reindex(idx).ffill() / 100.0
    rets = []
    i = 0
    while i + MONTH_DAYS < n:
        S0 = equity_nav.iloc[i]
        ST = equity_nav.iloc[i + MONTH_DAYS]
        sigma = vix_aligned.iloc[i]
        if pd.isna(sigma) or sigma <= 0:
            sigma = 0.20
        K = S0 * (1 + otm_pct)
        T = MONTH_DAYS / 252.0
        premium = bs_call_price(S0, K, T, R_FREE, sigma)
        payoff = max(ST - K, 0.0)
        raw_ret = ST / S0 - 1
        contribution = notional_frac * (premium / S0 - payoff / S0)
        rets.append((idx[i + MONTH_DAYS], raw_ret + contribution))
        i += MONTH_DAYS

    dates, r = zip(*rets)
    r = np.array(r)
    nav = np.cumprod(1 + r)
    sharpe = float(np.mean(r) / (np.std(r) + 1e-9) * np.sqrt(252 / MONTH_DAYS))
    peak = np.maximum.accumulate(nav)
    mdd = float(np.min((nav - peak) / peak))
    n_years = len(nav) * MONTH_DAYS / 252
    cagr = float(nav[-1] ** (1 / n_years) - 1) if n_years > 0 else None
    return sharpe, mdd, cagr


def main():
    for label, price_cache_path, mom_cache_path, is_pit in PERIODS:
        price_df, spy_close, mom_results, membership, vix = load_period(price_cache_path, mom_cache_path, is_pit)
        print(f"\n{'='*70}\n {label}\n{'='*70}")

        r_base = run_full_system(price_df, membership, spy_close, mom_results, **BASE)
        equity_nav = r_base["full_nav"]
        n_years = len(equity_nav) / 252
        base_cagr = float(equity_nav.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else None
        print(f"  无备兑(基线): Sharpe={r_base['Sharpe']:.3f}, MaxDD={r_base['MaxDD']*100:.2f}%, CAGR={base_cagr*100:.2f}%")

        if vix is None:
            print("  [警告] 这段区间没有VIX数据，跳过")
            continue

        for otm_pct, notional_frac in [(0.05, 1.0), (0.10, 1.0), (0.05, 0.5)]:
            sharpe, mdd, cagr = apply_covered_call(equity_nav, vix, otm_pct, notional_frac)
            print(f"  备兑看涨(OTM={otm_pct*100:.0f}%, 覆盖比例={notional_frac*100:.0f}%): "
                  f"Sharpe={sharpe:.3f}, MaxDD={mdd*100:.2f}%, CAGR={cagr*100:.2f}%")

    print("\n全部完成。")


if __name__ == "__main__":
    main()
