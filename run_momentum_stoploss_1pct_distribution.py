"""
Cycle52大修大补①(补全第257/258条的验证缺口)：只重跑1%止损这一档(不是
三档都重跑，节省约48分钟)，这次正确捕获`win_rate`以及完整的周收益率
分布(不只是汇总统计量)，核实1%止损是否存在第239/245条那类"触发比例
过高导致虚假凸性"的异常分布特征——用胜率、单周最大盈利/亏损、收益率
偏度这几个指标交叉检验，不能只看Sharpe/MaxDD两个数字。
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from backtest import fetch_market_data, run_walk_forward_backtest
from stat_validation import compute_period_metrics

if __name__ == "__main__":
    print("拉取5年市场数据...")
    c_df, o_df, h_df, l_df, v_df, vix, _ = fetch_market_data()

    print(f"\n{'='*70}\n  跑一遍：止损1%，这次捕获完整分布指标\n{'='*70}")
    t0 = time.time()
    rf_weekly = (1 + 0.04) ** (1 / 52) - 1
    results = run_walk_forward_backtest(
        c_df, o_df, h_df, l_df, v_df, vix,
        enable_shap=False,
        stop_loss_pct=0.01,
        n_workers=8,
    )
    df = pd.DataFrame(results)
    df["weekly_return"] = df["weekly_return"].fillna(0.0)
    df["net_worth"] = df["net_worth"].ffill().fillna(100.0)
    m = compute_period_metrics(df, rf_weekly, 52)
    elapsed = (time.time() - t0) / 60

    rets = df["weekly_return"].to_numpy(dtype=float)
    from scipy.stats import skew
    print(f"\n{'='*70}\n 止损1%完整分布诊断\n{'='*70}")
    print(f"  CAGR={m['cagr']*100:.2f}%, Sharpe={m['sharpe']:.3f}, MaxDD={m['max_dd']*100:.2f}%, "
          f"胜率={m['win_rate']*100:.1f}%, 耗时={elapsed:.1f}分钟")
    print(f"  周收益率偏度: {skew(rets):.3f}")
    print(f"  单周最大盈利: {rets.max()*100:.2f}%, 单周最大亏损: {rets.min()*100:.2f}%")
    print(f"  单周收益<=-1%(接近止损线)的周数: {(rets <= -0.009).sum()}/{len(rets)} "
          f"({(rets <= -0.009).sum()/len(rets)*100:.1f}%)")
    print(f"  单周收益在[-1.5%, -0.5%]区间(疑似被止损线'钉住')的周数: "
          f"{((rets >= -0.015) & (rets <= -0.005)).sum()}/{len(rets)}")
