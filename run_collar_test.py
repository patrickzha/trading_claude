"""
Cycle70大修大补①：领口策略(collar) = 备兑看涨(第302条已修正为正确实现，
只作用股票敞口) + 保护性看跌(第296条同样只应该作用股票敞口，之前没有
这个问题因为那次本来就是直接测equity_nav，没有股债混合的错误)——用卖
看涨收的权利金部分资助买看跌的成本，是这次会话期权框架的自然收尾：
分别验证过看跌保护(负面)、看跌价差(仍负面但更优)、备兑看涨(regime-
dependent)之后，测试两者结合能不能同时利用"备兑看涨在多数环境下的
稳定权利金收入"和"保护性看跌在下跌时的确定性保护"，尤其想看能不能
弥补第302条发现的"备兑看涨在2022-2026单独用会变差"这个短板。
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


def bs_price(S, K, T, r, sigma, is_call):
    if T <= 0 or sigma <= 0:
        return max((S - K) if is_call else (K - S), 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def apply_collar(equity_ret: pd.Series, vix_series: pd.Series, call_otm: float, put_otm: float):
    equity_ret = equity_ret.copy()
    idx = equity_ret.index
    n = len(idx)
    vix_aligned = vix_series.reindex(idx).ffill() / 100.0
    equity_nav_local = (1 + equity_ret).cumprod()
    i = 0
    while i + MONTH_DAYS < n:
        S0 = equity_nav_local.iloc[i]
        ST = equity_nav_local.iloc[i + MONTH_DAYS]
        sigma = vix_aligned.iloc[i]
        if pd.isna(sigma) or sigma <= 0:
            sigma = 0.20
        K_call = S0 * (1 + call_otm)
        K_put = S0 * (1 - put_otm)
        T = MONTH_DAYS / 252.0
        premium_call = bs_price(S0, K_call, T, R_FREE, sigma, is_call=True)
        premium_put = bs_price(S0, K_put, T, R_FREE, sigma, is_call=False)
        payoff_call = max(ST - K_call, 0.0)
        payoff_put = max(K_put - ST, 0.0)
        net_premium = premium_call - premium_put  # 卖call收权利金 - 买put付权利金
        net_payoff = payoff_put - payoff_call  # 买put的赔付 - 卖call的赔付(要付给对手方)
        entry_date, exit_date = idx[i], idx[i + MONTH_DAYS]
        equity_ret.loc[entry_date] = equity_ret.loc[entry_date] + net_premium / S0
        equity_ret.loc[exit_date] = equity_ret.loc[exit_date] + net_payoff / S0
        i += MONTH_DAYS
    return equity_ret


import full_system as fs

_orig_call_overlay = fs._apply_covered_call_overlay


def main():
    for label, price_cache_path, mom_cache_path, is_pit in PERIODS:
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

        print(f"\n{'='*70}\n {label}\n{'='*70}")
        if vix is None:
            print("  [警告] 无VIX数据，跳过")
            continue

        r_base = run_full_system(price_df, membership, spy_close, mom_results, **BASE)
        print(f"  无对冲(基线): Sharpe={r_base['Sharpe']:.3f}, MaxDD={r_base['MaxDD']*100:.2f}%, CAGR={r_base['CAGR']*100:.2f}%")

        for call_otm, put_otm in [(0.05, 0.05), (0.05, 0.10), (0.10, 0.05)]:
            fs._apply_covered_call_overlay = lambda ret, vix_s, otm, _co=call_otm, _po=put_otm: apply_collar(ret, vix_s, _co, _po)
            r_collar = run_full_system(price_df, membership, spy_close, mom_results,
                                        use_covered_call=True, vix_series=vix, **BASE)
            fs._apply_covered_call_overlay = _orig_call_overlay
            print(f"  领口(卖call{call_otm*100:.0f}%OTM/买put{put_otm*100:.0f}%OTM): "
                  f"Sharpe={r_collar['Sharpe']:.3f}, MaxDD={r_collar['MaxDD']*100:.2f}%, CAGR={r_collar['CAGR']*100:.2f}%")

    print("\n全部完成。")


if __name__ == "__main__":
    main()
