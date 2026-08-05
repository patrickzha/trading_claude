"""
Cycle17小修小补：动量策略每周选出的3只票，跟同一时点的价值/低波动候选池(各30只)
重叠度有多高。如果动量选的票经常也在价值/低波动候选池里，说明三条腿没有真正意义上
的分散化，combo_strategy.py里"三元组合降低波动"的效果可能被高估；如果基本不重叠，
说明三条腿确实在买不同的票，分散化逻辑站得住脚。
"""
from __future__ import annotations

import pickle

from value_investing import select_value_portfolio
from low_vol_investing import select_low_vol_portfolio
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
periods = [
    ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl",
     f"{SCRATCH}/default_full_results_cache.pkl", True),
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl",
     f"{SCRATCH}/momentum_2014_2020_results_cache.pkl", False),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl",
     f"{SCRATCH}/momentum_2009_2014_results_cache.pkl", False),
]

for label, price_cache_path, mom_cache_path, is_pit in periods:
    with open(price_cache_path, "rb") as f:
        cache = pickle.load(f)
    price_df, spy_close = cache["c"], cache.get("spy")
    with open(mom_cache_path, "rb") as f:
        mom_results = pickle.load(f)

    if is_pit:
        membership = compute_point_in_time_membership(price_df.index, list(price_df.columns),
                                                        cache["added_dates"], cache["removal_dates"])
    else:
        membership = {price_df.index[i].strftime("%Y-%m-%d"): list(price_df.columns) for i in range(len(price_df))}

    total_mom_picks = 0
    overlap_with_value = 0
    overlap_with_lowvol = 0
    overlap_with_either = 0
    n_weeks_checked = 0

    for row in mom_results:
        picks = row.get("picks") or []
        if not picks:
            continue
        date_str = row["date"]
        valid_tickers = membership.get(date_str)
        if valid_tickers is None:
            keys = sorted(membership.keys())
            older = [k for k in keys if k <= date_str]
            if not older:
                continue
            valid_tickers = membership[max(older)]

        try:
            rebalance_date = price_df.index[price_df.index.searchsorted(date_str)]
        except Exception:
            continue

        val_picks = set(select_value_portfolio(price_df, valid_tickers, rebalance_date, top_n=30))
        lv_picks = set(select_low_vol_portfolio(price_df, valid_tickers, rebalance_date, top_n=30))

        n_weeks_checked += 1
        for t in picks:
            total_mom_picks += 1
            in_val = t in val_picks
            in_lv = t in lv_picks
            overlap_with_value += int(in_val)
            overlap_with_lowvol += int(in_lv)
            overlap_with_either += int(in_val or in_lv)

    if total_mom_picks == 0:
        print(f"\n=== {label} === 无有效动量选股周，跳过")
        continue

    print(f"\n=== {label} ({n_weeks_checked}周, 共{total_mom_picks}次动量选股) ===")
    print(f"  动量票同时在价值候选池(30只)里的比例: {overlap_with_value/total_mom_picks:.1%}")
    print(f"  动量票同时在低波动候选池(30只)里的比例: {overlap_with_lowvol/total_mom_picks:.1%}")
    print(f"  动量票同时在价值或低波动任一候选池里的比例: {overlap_with_either/total_mom_picks:.1%}")
    print(f"  (随机基线参考：候选池30只/成分股~500只 ≈ 6%单腿命中率，两腿并集≈12%)")
