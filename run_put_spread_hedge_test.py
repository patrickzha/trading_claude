"""
Cycle66大修大补①：看跌期权价差(put spread)对冲——直接回应第296条发现的
核心问题("保险费本身持续侵蚀收益，尤其在没有出险的加息/温和下跌行情里")。
结构：买入OTM5%的看跌期权(行权价=95%现价)，同时卖出OTM15%的看跌期权
(行权价=85%现价)收权利金，用卖出的权利金部分抵消买入成本——代价是保护
上限被锁定在85%这个点位，跌破85%之后就不再有额外保护(跟纯买入看跌期权
比，损失了"无限保护"，换来更低的净成本)。
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


def bs_put_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


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


def apply_put_spread(equity_nav: pd.Series, vix: pd.Series, long_otm: float, short_otm: float):
    idx = equity_nav.index
    n = len(idx)
    vix_aligned = vix.reindex(idx).ffill() / 100.0
    hedged_rets = []
    i = 0
    while i + MONTH_DAYS < n:
        S0 = equity_nav.iloc[i]
        ST = equity_nav.iloc[i + MONTH_DAYS]
        sigma = vix_aligned.iloc[i]
        if pd.isna(sigma) or sigma <= 0:
            sigma = 0.20
        K_long = S0 * (1 - long_otm)
        K_short = S0 * (1 - short_otm)
        T = MONTH_DAYS / 252.0
        premium_long = bs_put_price(S0, K_long, T, R_FREE, sigma)
        premium_short = bs_put_price(S0, K_short, T, R_FREE, sigma)
        net_premium = premium_long - premium_short
        payoff_long = max(K_long - ST, 0.0)
        payoff_short = max(K_short - ST, 0.0)
        net_payoff = payoff_long - payoff_short
        raw_ret = ST / S0 - 1
        hedge_contribution = -net_premium / S0 + net_payoff / S0
        hedged_ret = raw_ret + hedge_contribution
        hedged_rets.append((idx[i + MONTH_DAYS], hedged_ret))
        i += MONTH_DAYS

    dates, rets = zip(*hedged_rets)
    rets = np.array(rets)
    nav = np.cumprod(1 + rets)
    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-9) * np.sqrt(252 / MONTH_DAYS))
    peak = np.maximum.accumulate(nav)
    mdd = float(np.min((nav - peak) / peak))
    return sharpe, mdd


def main():
    for label, price_cache_path, mom_cache_path, is_pit in PERIODS:
        price_df, spy_close, mom_results, membership, vix = load_period(price_cache_path, mom_cache_path, is_pit)
        print(f"\n{'='*70}\n {label}\n{'='*70}")

        r_base = run_full_system(price_df, membership, spy_close, mom_results, **BASE)
        equity_nav = r_base["full_nav"]
        print(f"  无对冲(基线): Sharpe={r_base['Sharpe']:.3f}, MaxDD={r_base['MaxDD']*100:.2f}%")

        if vix is None:
            print("  [警告] 这段区间没有VIX数据，跳过")
            continue

        for long_otm, short_otm in [(0.03, 0.08), (0.05, 0.10), (0.05, 0.08)]:
            sharpe, mdd = apply_put_spread(equity_nav, vix, long_otm, short_otm)
            print(f"  看跌价差(买{long_otm*100:.0f}%OTM/卖{short_otm*100:.0f}%OTM): "
                  f"Sharpe={sharpe:.3f}, MaxDD={mdd*100:.2f}%")

    print("\n全部完成。")


if __name__ == "__main__":
    main()
