"""
Cycle59后续：第276条(拟写)记录的"value止损3%+low_vol止损5%非对称组合"在三段
区间全部同步改善(Sharpe/MaxDD双双变好)，看起来是难得的干净正面结果——但
这次会话反复验证过(第239/245/257-261条)，"更紧的止损阈值Sharpe更好"这类
结果经常是"触发比例过高→下行被普遍钉死、上行不受限"这种人为凸性伪影，
不是真实风控改善。这个诊断直接照搬第239/257-261条建立的方法论，在正式
写入结论前先检查：value腿止损从5%收紧到3%，触发比例有没有明显跳升、
持有期收益率分布有没有出现异常偏度。

跟第239条(1%止损，触发比例67%)、第267/268条(3%对称止损，触发比例43%，
判定为"证据不足以支持"的边界情形)不同的地方：这次是价值腿单独收紧到
3%(low_vol腿保持5%不变)，触发比例应该比第267/268条的"两条腿都3%"更低，
但仍然需要实测验证，不能假设。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from scipy.stats import skew

from value_investing import select_value_portfolio, _hrp_weights
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"

PERIODS = [
    ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl", True),
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl", False),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl", False),
]

REBALANCE_DAYS = 5
COV_WINDOW = 60


def instrumented_backtest(price_df, membership, stop_loss_pct):
    """精简版backtest_value_strategy，只为了拿到触发比例和逐期收益分布，
    不算成本/基准对比(诊断用，不是正式回测)。"""
    index = price_df.index
    n = len(index)
    rebalance_idx = 252 + COV_WINDOW
    period_rets = []
    n_triggered = 0
    n_total_positions = 0

    while rebalance_idx + REBALANCE_DAYS < n:
        rebalance_date = index[rebalance_idx]
        next_date = index[rebalance_idx + REBALANCE_DAYS]
        date_str = rebalance_date.strftime("%Y-%m-%d")
        valid_tickers = membership.get(date_str)
        if valid_tickers is None:
            keys = sorted(membership.keys())
            valid_tickers = membership[max([k for k in keys if k <= date_str], default=keys[0])]

        picks = select_value_portfolio(price_df, valid_tickers, rebalance_date, top_n=30)
        picks = [p for p in picks if p in price_df.columns]
        if len(picks) < 5:
            rebalance_idx += REBALANCE_DAYS
            continue

        hist_window = price_df[picks].loc[:rebalance_date].tail(COV_WINDOW + 1)
        rets_window = hist_window.pct_change().dropna()
        valid_cols = rets_window.columns[rets_window.notna().all()]
        if len(valid_cols) >= 5:
            opt_weights = _hrp_weights(rets_window[valid_cols])
            weights = pd.Series(0.0, index=picks)
            for t in valid_cols:
                weights[t] = opt_weights[t]
            weights = weights / weights.sum() if weights.sum() > 0 else pd.Series(1.0 / len(picks), index=picks)
        else:
            weights = pd.Series(1.0 / len(picks), index=picks)

        entry_prices = {}
        for t in picks:
            try:
                entry_prices[t] = price_df[t].loc[:rebalance_date].dropna().iloc[-1]
            except IndexError:
                continue

        period_days = index[rebalance_idx:min(rebalance_idx + REBALANCE_DAYS, n)]
        stopped_out: dict[str, float] = {}
        day_vals = []
        for d in period_days:
            day_val = 0.0
            total_w = 0.0
            for t, p0 in entry_prices.items():
                if d in price_df.index and t in price_df.columns:
                    w = weights.get(t, 0)
                    if t in stopped_out:
                        day_val += w * stopped_out[t]
                        total_w += w
                        continue
                    p = price_df.at[d, t]
                    if pd.notna(p) and p0:
                        factor = p / p0
                        if factor <= (1 - stop_loss_pct):
                            factor = 1 - stop_loss_pct
                            stopped_out[t] = factor
                        day_val += w * factor
                        total_w += w
            if total_w > 0:
                day_vals.append(day_val / total_w)

        n_total_positions += len(entry_prices)
        n_triggered += len(stopped_out)
        if day_vals:
            period_rets.append(day_vals[-1] - 1.0)

        rebalance_idx += REBALANCE_DAYS

    return np.array(period_rets), n_triggered, n_total_positions


for label, cache_path, is_pit in PERIODS:
    print(f"\n{'='*70}\n {label}\n{'='*70}")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    price_df = cache["c"]
    if is_pit:
        membership = compute_point_in_time_membership(
            price_df.index, list(price_df.columns), cache["added_dates"], cache["removal_dates"])
    else:
        membership = {price_df.index[i].strftime("%Y-%m-%d"): list(price_df.columns) for i in range(len(price_df))}

    for pct in [0.05, 0.03]:
        rets, n_trig, n_pos = instrumented_backtest(price_df, membership, pct)
        trig_rate = n_trig / n_pos * 100 if n_pos else 0.0
        print(f"  止损{pct*100:.0f}%: 持仓-期数={n_pos}, 触发次数={n_trig}, 触发比例={trig_rate:.1f}%, "
              f"周期数={len(rets)}, 偏度={skew(rets):.3f}, 单期最大盈利={rets.max()*100:.2f}%, "
              f"单期最大亏损={rets.min()*100:.2f}%")

print("\n全部完成。")
