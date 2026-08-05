"""
Cycle12大修大补最终整合：把第55条的三元股票组合(动量+价值+低波动)当作
"股票敞口"，第66条的动态债券配置逻辑当作"债券敞口"，按60/40比例(动态
债券部分按利率趋势调整)组合成一个完整的、跨资产类别的系统，在三段有
完整数据的区间(2022-2026/2014-2020/2009-2014，2004-2009因为价值策略
缺基本面数据只能用第66条已有的纯股债版本，不重复)上测试完整表现。

这是这次会话所有分散化研究的最终整合产物——回答"如果真的要拿一个配置
去实盘，全部证据加起来支持的最佳组合长什么样"这个问题。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import yfinance as yf

from combo_strategy import run_combo_strategy
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
BOND_WEIGHT = 0.4


def true_mdd(nav_series):
    nav = np.array(nav_series)
    peak = np.maximum.accumulate(nav)
    return float(np.min((nav - peak) / peak))


def get_dynamic_bond_return(index, tlt, spy_placeholder_ret, tnx):
    tnx_aligned = tnx.reindex(index).ffill()
    tnx_trend = tnx_aligned.diff(60)
    bond_weight_internal = pd.Series(1.0, index=index)  # 债券腿自己始终是债券，不需要内部再分股债
    tlt_ret = tlt.reindex(index).ffill().pct_change()
    return tlt_ret


def main():
    print("拉取 TLT / ^TNX 数据...")
    data = yf.download(["TLT", "^TNX"], start="2009-01-01", end="2026-08-01",
                       progress=False, group_by="ticker", auto_adjust=True)
    tlt = data["TLT"]["Close"].dropna()
    tnx = data["^TNX"]["Close"].dropna()

    periods = [
        ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl", f"{SCRATCH}/pit_full_run_with_trades.pkl", True),
        ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl", f"{SCRATCH}/momentum_2014_2020_results_cache.pkl", False),
        ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl", f"{SCRATCH}/momentum_2009_2014_results_cache.pkl", False),
    ]

    for label, price_cache_path, mom_cache_path, is_pit in periods:
        with open(price_cache_path, "rb") as f:
            cache = pickle.load(f)
        price_df, spy_close = cache["c"], cache.get("spy")
        if is_pit:
            membership = compute_point_in_time_membership(price_df.index, list(price_df.columns),
                                                            cache["added_dates"], cache["removal_dates"])
        else:
            membership = {price_df.index[i].strftime("%Y-%m-%d"): list(price_df.columns) for i in range(len(price_df))}
        with open(mom_cache_path, "rb") as f:
            mom_results = pickle.load(f)

        equity_report = run_combo_strategy(price_df, membership, spy_close, mom_results)
        equity_nav = equity_report["combo_nav"]

        tnx_trend = tnx.reindex(equity_nav.index).ffill().diff(60)
        bond_weight_series = pd.Series(BOND_WEIGHT, index=equity_nav.index)
        bond_weight_series[tnx_trend > 0.3] = 0.1

        tlt_ret = tlt.reindex(equity_nav.index).ffill().pct_change()
        equity_ret = equity_nav.pct_change()

        full_ret = (1 - bond_weight_series) * equity_ret + bond_weight_series * tlt_ret
        full_ret = full_ret.dropna()
        full_nav = (1 + full_ret).cumprod()

        n_years = len(full_nav) / 252
        cagr = float(full_nav.iloc[-1] ** (1 / n_years) - 1)
        sharpe = float(full_ret.mean() / (full_ret.std() + 1e-9) * np.sqrt(252))
        mdd = true_mdd(full_nav.values)

        print(f"\n=== {label} ===")
        print(f"  纯三元股票组合: CAGR={equity_report['CAGR']*100:.2f}%, MaxDD={equity_report['MaxDD']*100:.2f}%")
        print(f"  加动态债券后完整系统: CAGR={cagr*100:.2f}%, Sharpe={sharpe:.3f}, MaxDD={mdd*100:.2f}%")


if __name__ == "__main__":
    main()
