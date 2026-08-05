"""
标普500 + 收紧止损（5% -> 2%），验证盈亏比是否真的改善。
================================================================
诊断发现：默认5%止损下，盈亏比只有0.84（平均亏损比平均盈利还大）。
在两段不同的历史窗口（50只票缓存数据）上测试不同止损阈值，都发现同一个
方向：止损越紧，盈亏比和累计收益越好——胜率不一定跟着变好（说明确实
存在"止损太紧提前止损、之后又涨回去"的假摔成本），但整体上收紧止损带来
的"少亏"效果超过了假摔的代价。

这里用 2% 止损（比默认5%紧不少，但不是测过的最极端值0.5%，避免选到
过拟合到某个窄样本的极端点）在完整5年标普500数据上做最终确认，同样用
样本内/样本外+显著性检验的方法论，不能只看整体数字就下结论。
"""
from __future__ import annotations

import os
import pickle
import time

from backtest import run_walk_forward_backtest, get_weekly_trading_dates
from optimized_metrics_v4 import run_backtest_with_metrics

DATA_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/sp500_data_cache.pkl"
EARNINGS_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/earnings_calendar_cache.pkl"


def main():
    t0 = time.time()
    with open(DATA_CACHE, "rb") as f:
        d = pickle.load(f)
    c_df, o_df, h_df, l_df, v_df, vix, spy_close = d["c"], d["o"], d["h"], d["l"], d["v"], d["vix"], d["spy"]
    common_idx = c_df.index
    print(f"复用缓存数据，股票数: {len(c_df.columns)}, 交易日: {len(common_idx)}")

    n_workers = max(1, os.cpu_count() or 1)
    print(f"并行worker数: {n_workers}")

    with open(EARNINGS_CACHE, "rb") as f:
        earnings_calendar = pickle.load(f)
    print(f"复用缓存的财报日历: {len(earnings_calendar)}只票")

    results = run_walk_forward_backtest(
        c_df, o_df, h_df, l_df, v_df, vix,
        enable_shap=False, n_workers=n_workers, xgb_n_jobs=1,
        earnings_calendar=earnings_calendar,
        stop_loss_pct=0.02,
    )
    print(f"回测耗时: {(time.time()-t0)/60:.1f}分钟")

    bench_rets = None
    if spy_close is not None:
        used_dates = {r["date"] for r in results}
        bench_rets = []
        for monday, friday in get_weekly_trading_dates(common_idx):
            if monday.strftime("%Y-%m-%d") not in used_dates:
                continue
            try:
                bench_rets.append((spy_close.loc[friday] - spy_close.loc[monday]) / spy_close.loc[monday])
            except Exception:
                bench_rets.append(0.0)

    run_backtest_with_metrics(results, benchmark_returns=bench_rets)
    print(f"\n全流程总耗时: {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
