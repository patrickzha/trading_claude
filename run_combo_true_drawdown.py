"""
Cycle8：给第55条的三元组合(ML动量+价值+低波动，目前最佳配置)算真实的
逐日最大回撤——第38条已经证明"用调仓点之间的收益率算回撤"会严重低估
真实风险，虽然这里用的是周频动量收益率(比半年度细得多)，但价值/低波动
两条腿本身是逐日重建净值的，动量策略只有周频颗粒度，组合起来的"真实"
日内回撤没有专门算过，这次补上，作为"如果现在要实盘"这个配置的完整
风险画像最后一块拼图。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from backtest import get_weekly_trading_dates
from value_investing import backtest_value_strategy
from low_vol_investing import backtest_low_vol_strategy
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"


def true_max_drawdown(nav_series):
    nav = np.array(nav_series)
    peak = np.maximum.accumulate(nav)
    return float(np.min((nav - peak) / peak))


def main():
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

        # 动量策略：只有周频净值(net_worth)，重采样成跟value/low_vol对齐的
        # 日度索引(周内每天用当周净值，近似——比周频本身细但不如真正逐日
        # 交易细，如实标注这个近似)
        mom_df = pd.DataFrame(mom_results)[["date", "net_worth"]]
        mom_df["date"] = pd.to_datetime(mom_df["date"])
        mom_series = mom_df.set_index("date")["net_worth"]
        mom_daily = mom_series.reindex(price_df.index, method="ffill").dropna()
        mom_daily = mom_daily / mom_daily.iloc[0]

        _, val_nav = backtest_value_strategy(price_df, membership, spy_close, rebalance_days=21, top_n=30)
        val_series = pd.DataFrame(val_nav, columns=["date", "nav"]).set_index("date")["nav"]

        _, lv_nav = backtest_low_vol_strategy(price_df, membership, spy_close, rebalance_days=21, top_n=30)
        lv_series = pd.DataFrame(lv_nav, columns=["date", "nav"]).set_index("date")["nav"]

        combined = pd.DataFrame({"mom": mom_daily, "val": val_series, "lv": lv_series}).dropna()
        combined_norm = combined / combined.iloc[0]
        combo_nav = combined_norm.mean(axis=1)

        mdd_combo = true_max_drawdown(combo_nav.values)
        mdd_mom = true_max_drawdown(combined_norm["mom"].values)
        mdd_val = true_max_drawdown(combined_norm["val"].values)
        mdd_lv = true_max_drawdown(combined_norm["lv"].values)

        n_years = len(combo_nav) / 252
        cagr = float(combo_nav.iloc[-1] ** (1 / n_years) - 1)
        daily_rets = combo_nav.pct_change().dropna()
        sharpe = float(daily_rets.mean() / (daily_rets.std() + 1e-9) * np.sqrt(252))

        print(f"\n=== {label} ===")
        print(f"  三元组合: CAGR={cagr*100:.2f}%, 日频Sharpe={sharpe:.3f}, 真实逐日MaxDD={mdd_combo*100:.2f}%")
        print(f"  对照单腿MaxDD: 动量(周频近似)={mdd_mom*100:.2f}%, 价值={mdd_val*100:.2f}%, 低波动={mdd_lv*100:.2f}%")


if __name__ == "__main__":
    main()
