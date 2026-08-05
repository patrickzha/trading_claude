"""
Cycle46大修大补①(框架改动，止损机制精细化——止损后资金再分配 vs 冻结)：
现有止损实现(第230-239条)触发后把这笔仓位的净值贡献冻结在(1-止损比例)，
直到下次调仓，这笔资金实际上没有产生任何后续收益。这次测试"止损后
把这部分权重让给组合里还持有的其他票"(等价于止损后资金立即重新配置
到剩余持仓，而不是晾在冻结状态)是否比现在的处理方式更好。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from value_investing import select_value_portfolio
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
REBALANCE_DAYS = 21
TOP_N = 30
STOP_LOSS_PCT = 0.05


def _lookup_membership(membership, membership_keys, date_str):
    valid = membership.get(date_str)
    if valid is not None:
        return valid
    candidates = [k for k in membership_keys if k <= date_str]
    if not candidates:
        return None
    return membership[max(candidates)]


def backtest_value_stoploss_reinvest(price_df, membership, rebalance_days=REBALANCE_DAYS,
                                      stop_loss_pct=STOP_LOSS_PCT, reinvest=False):
    """reinvest=False：现有实现(第230-239条)，止损后净值贡献冻结在
    (1-stop_loss_pct)，直到下次调仓。reinvest=True：止损触发当天记账
    冻结价值，从下一天开始把这只票整个踢出日度平均的分母，等价于把
    它腾出的资金份额平均分给还存活的其他持仓(不是额外买入，是"不再
    被这只已经躺平的票拖累分母")。"""
    index = price_df.index
    n = len(index)
    membership_keys = sorted(membership.keys())
    daily_nav = []
    rebalance_idx = 252 + 5
    running_nav = 1.0

    while rebalance_idx + rebalance_days < n:
        rebalance_date = index[rebalance_idx]
        date_str = rebalance_date.strftime("%Y-%m-%d")
        valid = _lookup_membership(membership, membership_keys, date_str)
        if valid is None:
            rebalance_idx += rebalance_days
            continue

        picks = select_value_portfolio(price_df, valid, rebalance_date, top_n=TOP_N)
        picks = [p for p in picks if p in price_df.columns]
        if len(picks) < 5:
            rebalance_idx += rebalance_days
            continue

        period_days = index[rebalance_idx:min(rebalance_idx + rebalance_days, n)]
        entry_prices = {t: price_df[t].loc[:rebalance_date].dropna().iloc[-1] for t in picks}
        stopped_out: dict[str, float] = {}
        excluded: set = set()  # reinvest模式下，止损后从下一天起彻底移出分母的票

        for d in period_days:
            day_val, total_w = 0.0, 0.0
            newly_stopped = []
            for t, p0 in entry_prices.items():
                if reinvest and t in excluded:
                    continue
                if t in stopped_out:
                    day_val += stopped_out[t]
                    total_w += 1
                    continue
                if d in price_df.index and t in price_df.columns:
                    p = price_df.at[d, t]
                    if pd.notna(p) and p0:
                        factor = p / p0
                        if factor <= (1 - stop_loss_pct):
                            factor = 1 - stop_loss_pct
                            newly_stopped.append(t)
                        day_val += factor
                        total_w += 1
            if total_w > 0:
                daily_nav.append((d, running_nav * (day_val / total_w)))
            elif daily_nav:
                daily_nav.append((d, daily_nav[-1][1]))

            for t in newly_stopped:
                stopped_out[t] = 1 - stop_loss_pct
                if reinvest:
                    excluded.add(t)

        if daily_nav:
            running_nav = daily_nav[-1][1]
        rebalance_idx += rebalance_days

    return daily_nav


def true_mdd(nav):
    nav = np.array(nav)
    peak = np.maximum.accumulate(nav)
    return float(np.min((nav - peak) / peak))


def metrics(nav_list):
    nav = np.array([n for _, n in nav_list])
    n_years = len(nav) / 252
    cagr = float((nav[-1] / nav[0]) ** (1 / n_years) - 1) if n_years > 0 else None
    rets = np.diff(nav) / nav[:-1]
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252))
    mdd = true_mdd(nav)
    return cagr, sharpe, mdd


def main():
    with open("/Users/zhang/Desktop/trading/sp500_membership_added_dates.json") as f:
        import json
        added_dates_raw = json.load(f)
    added_dates = {t: pd.Timestamp(d) for t, d in added_dates_raw.items()}

    periods = [
        ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl", True),
        ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl", False),
        ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl", False),
    ]

    for label, price_cache_path, is_pit in periods:
        with open(price_cache_path, "rb") as f:
            cache = pickle.load(f)
        price_df = cache["c"]
        if is_pit:
            membership = compute_point_in_time_membership(price_df.index, list(price_df.columns),
                                                            cache["added_dates"], cache["removal_dates"])
        else:
            membership = compute_point_in_time_membership(price_df.index, list(price_df.columns), added_dates, {})

        print(f"\n=== {label} ===")
        nav_freeze = backtest_value_stoploss_reinvest(price_df, membership, reinvest=False)
        f_cagr, f_sharpe, f_mdd = metrics(nav_freeze)
        print(f"  冻结(第230-239条现有实现): CAGR={f_cagr*100:.2f}%, Sharpe={f_sharpe:.3f}, MaxDD={f_mdd*100:.2f}%")

        nav_reinvest = backtest_value_stoploss_reinvest(price_df, membership, reinvest=True)
        r_cagr, r_sharpe, r_mdd = metrics(nav_reinvest)
        print(f"  再分配(止损后移出分母): CAGR={r_cagr*100:.2f}%, Sharpe={r_sharpe:.3f}, MaxDD={r_mdd*100:.2f}%")


if __name__ == "__main__":
    main()
