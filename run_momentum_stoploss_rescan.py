"""
Cycle51大修大补①(框架改动，方法论复用——动量腿止损阈值重新扫描)：
动量腿的2%止损(`STOP_LOSS_PCT`)是会话早期就定下的参数，从未用这次
会话建立的标准方法论(第236-239条：多档阈值扫描+核实触发比例，避免
过紧阈值的伪影)系统性重新检验过。这次用完整walk-forward(5年数据、
8核并行)重新测1%/2%/3%三档，看2%是否依然是最优选择。

如实说明：每档大约需要~20-30分钟(完整XGBoost walk-forward训练)，
三档合计可能耗时超过1小时，是这次会话时间成本最高的一次单项测试，
后台运行不阻塞其他工作。
"""
from __future__ import annotations

import time

import pandas as pd

from backtest import fetch_market_data, run_walk_forward_backtest
from stat_validation import compute_period_metrics


def _run(c_df, o_df, h_df, l_df, v_df, vix, label, stop_loss_pct):
    print(f"\n{'='*70}\n  跑一遍：{label} (stop_loss_pct={stop_loss_pct})\n{'='*70}")
    t0 = time.time()
    rf_weekly = (1 + 0.04) ** (1 / 52) - 1
    results = run_walk_forward_backtest(
        c_df, o_df, h_df, l_df, v_df, vix,
        enable_shap=False,
        stop_loss_pct=stop_loss_pct,
        n_workers=8,
    )
    df = pd.DataFrame(results)
    df["weekly_return"] = df["weekly_return"].fillna(0.0)
    df["net_worth"] = df["net_worth"].ffill().fillna(100.0)
    m = compute_period_metrics(df, rf_weekly, 52)
    m["label"] = label
    m["elapsed_min"] = (time.time() - t0) / 60
    return m


if __name__ == "__main__":
    print("拉取5年市场数据...")
    c_df, o_df, h_df, l_df, v_df, vix, _ = fetch_market_data()

    results = []
    for pct in [0.01, 0.02, 0.03]:
        m = _run(c_df, o_df, h_df, l_df, v_df, vix, f"止损{pct:.0%}", pct)
        results.append(m)
        print(f"  完成: CAGR={m['cagr']*100:.2f}%, Sharpe={m['sharpe']:.3f}, MaxDD={m['max_dd']*100:.2f}%, "
              f"耗时{m['elapsed_min']:.1f}分钟")

    print(f"\n{'='*70}\n 汇总\n{'='*70}")
    for m in results:
        print(f"  {m['label']}: CAGR={m['cagr']*100:.2f}%, Sharpe={m['sharpe']:.3f}, MaxDD={m['max_dd']*100:.2f}%")
