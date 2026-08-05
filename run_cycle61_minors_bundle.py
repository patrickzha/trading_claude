"""
Cycle61小修(a)(b)合并脚本。

小修(a)：承接第280条建立的方法论警示("bootstrap显著≠不是历史路径拟合")，
直接检验第276条已采纳的3%止损，在三段区间各自的前半/后半两个独立子区间
是否都分别跑赢5%对照，而不是只在聚合层面显著——如果只有一半子区间贡献了
全部改善，是"历史路径拟合"风险的直接证据；如果两半都独立成立，进一步
巩固3%止损不是拟合了某一段特定行情的判断。

小修(b)：3%止损配置(第276条)跟gold_split(第272/274条)组合是否依然稳定——
这两项分别在各自默认对方为0的基础上验证过，没测过同时叠加的情况。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

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


def load_period(price_cache_path, mom_cache_path, is_pit):
    with open(price_cache_path, "rb") as f:
        cache = pickle.load(f)
    price_df, spy_close = cache["c"], cache.get("spy")
    with open(mom_cache_path, "rb") as f:
        mom_results = pickle.load(f)
    if is_pit:
        membership = compute_point_in_time_membership(
            price_df.index, list(price_df.columns), cache["added_dates"], cache["removal_dates"])
    else:
        membership = {price_df.index[i].strftime("%Y-%m-%d"): list(price_df.columns) for i in range(len(price_df))}
    return price_df, spy_close, mom_results, membership


def sharpe_mdd(nav: pd.Series):
    rets = nav.pct_change().dropna()
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252))
    peak = nav.cummax()
    mdd = float(((nav - peak) / peak).min())
    return sharpe, mdd


print(f"\n{'='*70}\n 小修(a): 3%止损前后半段独立稳健性复核\n{'='*70}")
for label, price_cache_path, mom_cache_path, is_pit in PERIODS:
    price_df, spy_close, mom_results, membership = load_period(price_cache_path, mom_cache_path, is_pit)
    print(f"\n--- {label} ---")

    r_base = run_full_system(price_df, membership, spy_close, mom_results,
                              value_weighting="hrp", value_rebalance_days=5,
                              value_stop_loss_pct=0.05, low_vol_stop_loss_pct=0.05)
    r_new = run_full_system(price_df, membership, spy_close, mom_results,
                             value_weighting="hrp", value_rebalance_days=5,
                             value_stop_loss_pct=0.03, low_vol_stop_loss_pct=0.05)

    nav_base, nav_new = r_base["full_nav"], r_new["full_nav"]
    common_idx = nav_base.index.intersection(nav_new.index)
    nav_base, nav_new = nav_base.loc[common_idx], nav_new.loc[common_idx]
    mid = len(common_idx) // 2

    for half_name, sl in [("前半", slice(0, mid)), ("后半", slice(mid, None))]:
        nb = nav_base.iloc[sl]
        nn = nav_new.iloc[sl]
        sb, mb = sharpe_mdd(nb / nb.iloc[0])
        sn, mn = sharpe_mdd(nn / nn.iloc[0])
        better = "3%更好" if sn > sb else "5%更好(反例)"
        print(f"  {half_name}({len(nb)}天): 5%止损 Sharpe={sb:.3f}/MaxDD={mb*100:.2f}%, "
              f"3%止损 Sharpe={sn:.3f}/MaxDD={mn*100:.2f}% -> {better}")

print(f"\n{'='*70}\n 小修(b): 3%止损 + gold_split组合兼容性\n{'='*70}")
for label, price_cache_path, mom_cache_path, is_pit in PERIODS:
    price_df, spy_close, mom_results, membership = load_period(price_cache_path, mom_cache_path, is_pit)
    print(f"\n--- {label} ---")
    for gs in [0.0, 0.3]:
        for sl in [0.05, 0.03]:
            r = run_full_system(price_df, membership, spy_close, mom_results,
                                 value_weighting="hrp", value_rebalance_days=5,
                                 value_stop_loss_pct=sl, low_vol_stop_loss_pct=0.05,
                                 gold_split=gs)
            print(f"  gold_split={gs}, value_stop={sl}: Sharpe={r['Sharpe']:.3f}, MaxDD={r['MaxDD']*100:.2f}%")

print(f"\n{'='*70}\n 小修(c): 2014-2020段value腿收益率偏度持续偏高的成因排查(第276/278条留下的疑问)\n{'='*70}")
from value_investing import select_value_portfolio, _hrp_weights

label, price_cache_path, mom_cache_path, is_pit = PERIODS[1]
price_df, spy_close, mom_results, membership = load_period(price_cache_path, mom_cache_path, is_pit)
index = price_df.index
n = len(index)
rebalance_idx = 252 + 60
period_rets = []
period_dates = []
while rebalance_idx + 5 < n:
    rebalance_date = index[rebalance_idx]
    date_str = rebalance_date.strftime("%Y-%m-%d")
    valid_tickers = membership.get(date_str)
    if valid_tickers is None:
        keys = sorted(membership.keys())
        valid_tickers = membership[max([k for k in keys if k <= date_str], default=keys[0])]
    picks = [p for p in select_value_portfolio(price_df, valid_tickers, rebalance_date, top_n=30) if p in price_df.columns]
    if len(picks) >= 5:
        hist_window = price_df[picks].loc[:rebalance_date].tail(61)
        rets_window = hist_window.pct_change().dropna()
        valid_cols = rets_window.columns[rets_window.notna().all()]
        if len(valid_cols) >= 5:
            opt_w = _hrp_weights(rets_window[valid_cols])
            weights = pd.Series(0.0, index=picks)
            for t in valid_cols:
                weights[t] = opt_w[t]
            weights = weights / weights.sum() if weights.sum() > 0 else pd.Series(1.0 / len(picks), index=picks)
        else:
            weights = pd.Series(1.0 / len(picks), index=picks)
        p0 = price_df[picks].loc[:rebalance_date].iloc[-1]
        p1 = price_df[picks].iloc[rebalance_idx + 5]
        ret = float((weights * (p1 / p0 - 1)).sum())
        period_rets.append(ret)
        period_dates.append(date_str)
    rebalance_idx += 5

rets_arr = np.array(period_rets)
top5_idx = np.argsort(rets_arr)[-5:][::-1]
print(f"  2014-2020段value腿(HRP,5天调仓)共{len(rets_arr)}期，偏度={pd.Series(rets_arr).skew():.3f}")
print(f"  最大的5期收益: " + ", ".join(f"{period_dates[i]}={rets_arr[i]*100:.2f}%" for i in top5_idx))
print(f"  全部期数均值={rets_arr.mean()*100:.3f}%, 中位数={np.median(rets_arr)*100:.3f}%, "
      f"最大值单期贡献占全部正收益期总和的比例={rets_arr[top5_idx[0]]/rets_arr[rets_arr>0].sum()*100:.1f}%")

print("\n全部完成。")
