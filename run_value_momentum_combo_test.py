"""
Cycle2小修小补：价值+动量组合因子测试。两个策略机制完全不同（动量赌
短期延续、价值赌HML因子长期均值回归)，理论上相关性低、组合应该有
分散化收益。用跟"验证结论"第35条核心卫星测试同样的组合数学，把动量
策略的周收益率序列跟价值策略的周收益率序列(daily_nav重采样成周频)
按几个比例混合，看Sharpe/回撤有没有改善——这不是重新训练模型，纯粹是
两条已经算好的收益率序列的线性组合。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from backtest import get_weekly_trading_dates
from value_investing import backtest_value_strategy
from run_survivorship_bias_fix_validation import compute_point_in_time_membership, CACHE_PATH

MOMENTUM_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/pit_full_run_with_trades.pkl"

WEIGHTS = [0.0, 0.3, 0.5, 0.7, 1.0]  # 0=纯动量, 1=纯价值


def metrics(rets, periods_per_year=52.0):
    rets = np.array(rets)
    nav = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(nav)
    mdd = float(np.min((nav - peak) / peak))
    n_years = len(rets) / periods_per_year
    cagr = float(nav[-1] ** (1 / n_years) - 1)
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(periods_per_year))
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": mdd}


def main():
    with open(MOMENTUM_CACHE, "rb") as f:
        mom_results = pickle.load(f)
    mom_df = pd.DataFrame(mom_results)[["date", "weekly_return"]]
    mom_df["date"] = pd.to_datetime(mom_df["date"])
    mom_df = mom_df.set_index("date")

    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    price_df, spy_close = cache["c"], cache.get("spy")
    membership = compute_point_in_time_membership(price_df.index, list(price_df.columns),
                                                    cache["added_dates"], cache["removal_dates"])
    _, daily_nav = backtest_value_strategy(price_df, membership, spy_close, rebalance_days=126, top_n=30)
    nav_df = pd.DataFrame(daily_nav, columns=["date", "nav"]).set_index("date")

    # 直接用跟动量策略完全一样的周一到周五定义(get_weekly_trading_dates)
    # 算价值策略的"周收益率"，两边严格按同一套周边界对齐，不用重采样
    # (重采样成W-FRI的周边界跟动量策略实际用的交易日周边界不一定完全
    # 重合，之前直接join导致0行对齐失败就是这个原因)
    week_pairs = get_weekly_trading_dates(price_df.index)
    val_rows = []
    for monday, friday in week_pairs:
        try:
            nav_at_monday = nav_df.loc[:monday, "nav"].iloc[-1]
            nav_at_friday = nav_df.loc[:friday, "nav"].iloc[-1]
        except IndexError:
            continue
        val_rows.append({"date": monday, "value_ret": nav_at_friday / nav_at_monday - 1})
    val_weekly_ret = pd.DataFrame(val_rows).set_index("date")["value_ret"]

    merged = mom_df.join(val_weekly_ret, how="inner")
    print(f"对齐后共 {len(merged)} 周")

    corr = merged["weekly_return"].corr(merged["value_ret"])
    print(f"动量策略 vs 价值策略 周收益率相关系数: {corr:.4f}")

    print(f"\n{'value_weight':>12} | {'CAGR':>8} | {'Sharpe':>8} | {'MaxDD':>8}")
    for w in WEIGHTS:
        blended = w * merged["value_ret"] + (1 - w) * merged["weekly_return"]
        m = metrics(blended.values)
        label = f"{w:.1f}(纯价值)" if w == 1.0 else (f"{w:.1f}(纯动量)" if w == 0.0 else f"{w:.1f}")
        print(f"{label:>12} | {m['CAGR']*100:7.2f}% | {m['Sharpe']:8.3f} | {m['MaxDD']*100:7.2f}%")


if __name__ == "__main__":
    main()
