"""
Cycle18小修小补：账户级回撤熔断的两个参数(account_dd_threshold默认-8%，
account_dd_scale默认打五折)从来没有做过敏感性测试，一直是plan阶段随手定的
数字。这次用session已经验证过的标准方法论(三窗口初筛，方向一致才晋级全量
验证)测试更敏感(-5%触发)/更迟钝(-12%触发)、更激进(打三折)/更温和(打七折)
四个变体。

注意：跑这个脚本期间不要改us_stock_screener.py/backtest.py——
ProcessPoolExecutor worker用spawn方式重新import，主进程代码被改会导致
worker和主进程配置对不上。
"""
from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np

from backtest import run_walk_forward_backtest, get_weekly_trading_dates
from optimized_metrics_v4 import run_backtest_with_metrics

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
DATA_CACHE = os.path.join(SCRATCH, "sp500_data_cache.pkl")
EARNINGS_CACHE = os.path.join(SCRATCH, "earnings_calendar_cache.pkl")
RESULT_JSON = os.path.join(SCRATCH, "account_dd_sensitivity_result.json")

N_WORKERS = max(1, os.cpu_count() or 1)

SCREEN_CANDIDATES = {
    "threshold_-5%(更敏感)": {"account_dd_threshold": -0.05},
    "threshold_-12%(更迟钝)": {"account_dd_threshold": -0.12},
    "scale_30%(更激进)": {"account_dd_scale": 0.3},
    "scale_70%(更温和)": {"account_dd_scale": 0.7},
}


def rough_metrics(results):
    rets = np.array([r["weekly_return"] for r in results]) if results else np.array([])
    if len(rets) < 2:
        return {"cum": 0.0, "mdd": 0.0, "sharpe": 0.0, "n_weeks": len(rets)}
    nav = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(nav)
    mdd = float(np.min((nav - peak) / peak))
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(52))
    return {"cum": float(nav[-1] - 1), "mdd": mdd, "sharpe": sharpe, "n_weeks": len(rets)}


def slice_window(data, end_idx, window_weeks=25, lookback_days=150):
    start_idx = max(0, end_idx - window_weeks * 5 - lookback_days)
    idx_slice = data["c"].index[start_idx:end_idx]
    return {
        "c": data["c"].loc[idx_slice], "o": data["o"].loc[idx_slice],
        "h": data["h"].loc[idx_slice], "l": data["l"].loc[idx_slice],
        "v": data["v"].loc[idx_slice], "vix": data["vix"].loc[idx_slice],
    }


def run_one(win, earnings_calendar, extra_kwargs, n_workers):
    return run_walk_forward_backtest(
        win["c"], win["o"], win["h"], win["l"], win["v"], win["vix"],
        enable_shap=False, n_workers=n_workers, xgb_n_jobs=1,
        earnings_calendar=earnings_calendar,
        **extra_kwargs,
    )


def main():
    t0 = time.time()
    print("=" * 70)
    print(" Cycle18小修小补：账户级回撤熔断参数敏感性(threshold/scale)")
    print("=" * 70)

    with open(DATA_CACHE, "rb") as f:
        full = pickle.load(f)
    data = {"c": full["c"], "o": full["o"], "h": full["h"], "l": full["l"],
            "v": full["v"], "vix": full["vix"]}
    spy_close = full.get("spy")
    n_total = len(data["c"].index)
    print(f"数据总长度: {n_total}天, {data['c'].index[0].date()} ~ {data['c'].index[-1].date()}")
    print(f"并行worker数: {N_WORKERS}")

    with open(EARNINGS_CACHE, "rb") as f:
        earnings_calendar = pickle.load(f)

    window_ends = {"近25周": n_total, "中段窗口(约1年前)": n_total - 300, "早段窗口(约2年前)": n_total - 600}

    screen_results = {}
    for win_name, end_idx in window_ends.items():
        win = slice_window(data, end_idx)
        print(f"\n{'='*20} 初筛窗口: {win_name} {'='*20}")

        base_res = run_one(win, earnings_calendar, {}, N_WORKERS)
        base_metrics = rough_metrics(base_res)
        print(f"  baseline(默认-8%/五折): {base_metrics}")

        for cand_name, kwargs in SCREEN_CANDIDATES.items():
            try:
                res = run_one(win, earnings_calendar, kwargs, N_WORKERS)
                m = rough_metrics(res)
            except Exception as e:
                print(f"  [错误] {cand_name} 在 {win_name} 跑失败: {type(e).__name__}: {e}")
                m = {"cum": None, "mdd": None, "sharpe": None, "n_weeks": 0, "error": str(e)}
            print(f"  {cand_name}: {m}")
            screen_results.setdefault(cand_name, {})[win_name] = {"baseline": base_metrics, "treatment": m}

        with open(RESULT_JSON, "w") as f:
            json.dump({"screen": screen_results, "promoted": [], "full": {}}, f, indent=2, default=str)

    promoted = []
    print(f"\n{'='*20} 初筛汇总与晋级判定 {'='*20}")
    for cand_name, per_win in screen_results.items():
        deltas = []
        broken = False
        for win_name, pair in per_win.items():
            t = pair["treatment"]
            b = pair["baseline"]
            if t.get("sharpe") is None or t.get("n_weeks", 0) == 0:
                broken = True
                break
            deltas.append(t["sharpe"] - b["sharpe"])
        if broken:
            print(f"  {cand_name}: 跑失败或候选被滤空，不晋级")
            continue
        same_direction = all(d > 0 for d in deltas) or all(d < 0 for d in deltas)
        verdict = "晋级全量验证" if all(d > 0 for d in deltas) else ("不晋级(一致更差)" if same_direction else "不晋级(方向不一致/噪声)")
        print(f"  {cand_name}: 三窗口Sharpe差值={[round(d,3) for d in deltas]} -> {verdict}")
        if all(d > 0 for d in deltas):
            promoted.append(cand_name)

    full_candidates = {name: SCREEN_CANDIDATES[name] for name in promoted}
    print(f"\n进入全量验证的候选: {list(full_candidates.keys())}")

    full_results = {}
    if full_candidates:
        print(f"\n{'='*20} 全量验证阶段 {'='*20}")
        base_full = run_one(data, earnings_calendar, {}, N_WORKERS)
        bench_rets = None
        if spy_close is not None:
            used_dates = {r["date"] for r in base_full}
            bench_rets = []
            for monday, friday in get_weekly_trading_dates(data["c"].index):
                if monday.strftime("%Y-%m-%d") not in used_dates:
                    continue
                try:
                    bench_rets.append((spy_close.loc[friday] - spy_close.loc[monday]) / spy_close.loc[monday])
                except Exception:
                    bench_rets.append(0.0)
        print("\n[baseline 全量报告]")
        run_backtest_with_metrics(base_full, benchmark_returns=bench_rets)
        full_results["baseline"] = rough_metrics(base_full)

        for cand_name, kwargs in full_candidates.items():
            print(f"\n[{cand_name} 全量报告]")
            res = run_one(data, earnings_calendar, kwargs, N_WORKERS)
            run_backtest_with_metrics(res, benchmark_returns=bench_rets)
            full_results[cand_name] = rough_metrics(res)
    else:
        print("\n没有候选晋级全量验证，四个变体在三窗口初筛阶段都不是一致改善。")

    with open(RESULT_JSON, "w") as f:
        json.dump({"screen": screen_results, "promoted": promoted, "full": full_results}, f, indent=2, default=str)

    print(f"\n总耗时: {(time.time()-t0)/60:.1f}分钟")


if __name__ == "__main__":
    main()
