"""
交易成本敏感性分析（诊断用，不是待采纳的候选）+ rolling_window_days 全量公平重测。
================================================================
现在双边0.3%的成本假设(COST_BI)是拍脑袋定的，没有随持仓规模/流动性变化的
滑点模型（见README"已知限制"）。这里不是要找"最优成本参数"——成本是市场
决定的，不是策略能调的旋钮——而是想知道：如果这个假设本身有误差（比如
实际滑点比0.3%更高），回测数字会崩到什么程度。如果结果对这个假设极度
敏感，说明选股alpha本身就薄，经不起假设误差；如果不敏感，说明策略的边际
足够厚，值得继续往下投入。

第一版用 monkeypatch 改模块级 COST_BI 常量，结果四档成本给出完全相同的
数字——排查发现 ProcessPoolExecutor 用 spawn 方式起 worker 进程，worker
会重新从磁盘 import 模块，不会继承父进程运行时改过的模块属性，所以训练
标签和结算用的其实一直是文件里的原始值 0.003，没有真的测到。已经把
COST_BI 改成跟 stop_loss_pct 一样的显式参数（cost_bi），从
run_walk_forward_backtest 一路传到 worker 的 config dict 里，才能在
ProcessPoolExecutor 下正确生效，重新验证过 0.001 和 0.02 两档给出不同
结果之后才跑这一版。

顺路把 rolling_window_days 也用全量数据公平重测（三窗口初筛里 600天因为
窗口本身只有约275天数据，测不出效果），并且这次跑完整的统计验证
（run_backtest_with_metrics，含样本内外+bootstrap），而不只是粗算Sharpe，
因为初测结果（CAGR 37%->40%）看起来是个真实信号，值得跟止损/RSI/新近度
加权同一个严格程度确认一遍。
"""
from __future__ import annotations

import pickle
import time

import numpy as np

from backtest import run_walk_forward_backtest, get_weekly_trading_dates
from optimized_metrics_v4 import run_backtest_with_metrics

DATA_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/sp500_data_cache.pkl"
EARNINGS_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/earnings_calendar_cache.pkl"

COST_LEVELS = [0.001, 0.003, 0.005, 0.008]  # 0.3%是当前默认


def rough_metrics(results):
    rets = np.array([r["weekly_return"] for r in results]) if results else np.array([])
    if len(rets) < 2:
        return {"cum": 0.0, "sharpe": 0.0, "n_weeks": len(rets)}
    nav = np.cumprod(1 + rets)
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(52))
    return {"cum": float(nav[-1] - 1), "sharpe": sharpe, "n_weeks": len(rets)}


def main():
    with open(DATA_CACHE, "rb") as f:
        full = pickle.load(f)
    c_df, o_df, h_df, l_df, v_df, vix, spy_close = (
        full["c"], full["o"], full["h"], full["l"], full["v"], full["vix"], full.get("spy")
    )
    with open(EARNINGS_CACHE, "rb") as f:
        earnings_calendar = pickle.load(f)

    n_workers = 8

    print("=" * 70)
    print(" 交易成本敏感性分析（cost_bi 已修复为显式参数，正确穿透worker进程）")
    print("=" * 70)
    for cost in COST_LEVELS:
        t0 = time.time()
        results = run_walk_forward_backtest(
            c_df, o_df, h_df, l_df, v_df, vix,
            enable_shap=False, n_workers=n_workers, xgb_n_jobs=1,
            earnings_calendar=earnings_calendar,
            cost_bi=cost,
        )
        m = rough_metrics(results)
        print(f"cost_bi={cost:.3f}: CAGR≈{(1+m['cum'])**(52/max(m['n_weeks'],1))-1:.2%}  "
              f"累计={m['cum']:.4f}  Sharpe(粗)={m['sharpe']:.2f}  "
              f"耗时{(time.time()-t0)/60:.1f}分钟")

    print("\n" + "=" * 70)
    print(" rolling_window_days 全量数据公平重测（含完整统计验证）")
    print("=" * 70)
    bench_rets = None
    for rw in [300, 600]:
        t0 = time.time()
        results = run_walk_forward_backtest(
            c_df, o_df, h_df, l_df, v_df, vix,
            enable_shap=False, n_workers=n_workers, xgb_n_jobs=1,
            earnings_calendar=earnings_calendar,
            rolling_window_days=rw,
        )
        print(f"rolling_window_days={rw} 耗时{(time.time()-t0)/60:.1f}分钟")
        if spy_close is not None and bench_rets is None:
            used_dates = {r["date"] for r in results}
            bench_rets = []
            for monday, friday in get_weekly_trading_dates(c_df.index):
                if monday.strftime("%Y-%m-%d") not in used_dates:
                    continue
                try:
                    bench_rets.append((spy_close.loc[friday] - spy_close.loc[monday]) / spy_close.loc[monday])
                except Exception:
                    bench_rets.append(0.0)
        print(f"\n[rolling_window_days={rw} 全量报告]")
        run_backtest_with_metrics(results, benchmark_returns=bench_rets)


if __name__ == "__main__":
    main()
