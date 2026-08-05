"""
批量验证第六轮：超参数网格搜索——之前只测过"整体加强正则化"这一个bundle
（更浅的树+更强的L1/L2一起变），结果比默认差，但没有把每个维度单独拆开测过。
这次把 max_depth/min_child_weight/reg_lambda/reg_alpha/gamma 五个维度各自
独立变化（不是一起变），同时测"更保守"和"更激进"两个方向，看有没有单独
某一个维度的调整是有效的、只是被之前的bundle测试掩盖了。

baseline 现在已经是上一轮验证过的新默认配置（RSI阈值=100，即不再用RSI过热
拒绝候选——全量验证过，CAGR 19.29%->26.80%、Sharpe 0.76->1.00，全指标同步
改善，置信区间不再跨0），不用再显式传参，函数默认值已经改了。

方法论跟之前完全一样：三窗口方向一致（Sharpe同向变化）才晋级全量验证，
不一致/都更差的判定为噪声或负向，不投入~35分钟的全量算力。

注意：这个脚本运行期间不要再改 us_stock_screener.py / backtest.py——
ProcessPoolExecutor 的 worker 进程用 spawn 方式重新从磁盘 import 模块，
如果主进程还在跑、磁盘上的代码被改了，worker 会用新代码 + 主进程传来的
旧 config dict，字段对不上直接 KeyError（踩过这个坑，见上一轮跑崩的教训）。
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
MACRO_CACHE = os.path.join(SCRATCH, "macro_df_cache.pkl")
SH_CACHE = os.path.join(SCRATCH, "sh_inverse_etf_cache.pkl")
RESULT_JSON = os.path.join(SCRATCH, "batch_candidates_hp_search_result.json")

N_WORKERS = max(1, os.cpu_count() or 1)

with open(MACRO_CACHE, "rb") as _f:
    _MACRO_DF = pickle.load(_f)
with open(SH_CACHE, "rb") as _f:
    _SH_CLOSE = pickle.load(_f)

SCREEN_CANDIDATES = {
    "depth_3": {"xgb_max_depth": 3},
    "depth_5": {"xgb_max_depth": 5},
    "depth_6": {"xgb_max_depth": 6},
    "min_child_weight_3": {"xgb_min_child_weight": 3.0},
    "min_child_weight_0.5": {"xgb_min_child_weight": 0.5},
    "reg_lambda_3": {"xgb_reg_lambda": 3.0},
    "reg_alpha_0.5": {"xgb_reg_alpha": 0.5},
    "gamma_0.5": {"xgb_gamma": 0.5},
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
    print(" 批量候选验证第二轮：阶段A 8个新方向的3窗口初筛 + 自动全量验证")
    print(" baseline = RSI阈值100新默认配置")
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

    # ---------- 阶段1：三窗口初筛 ----------
    screen_results = {}
    for win_name, end_idx in window_ends.items():
        win = slice_window(data, end_idx)
        print(f"\n{'='*20} 初筛窗口: {win_name} {'='*20}")

        print(f"\n--- baseline ---")
        base_res = run_one(win, earnings_calendar, {}, N_WORKERS)
        base_metrics = rough_metrics(base_res)
        print(f"  baseline: {base_metrics}")

        for cand_name, kwargs in SCREEN_CANDIDATES.items():
            print(f"\n--- {cand_name} ({kwargs}) ---")
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

    # ---------- 阶段2：判定 ----------
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
        print(f"  {cand_name}: 三窗口Sharpe差值={[round(d,2) for d in deltas]} -> {verdict}")
        if all(d > 0 for d in deltas):
            promoted.append(cand_name)

    full_candidates = {name: SCREEN_CANDIDATES[name] for name in promoted}
    print(f"\n进入全量验证的候选: {list(full_candidates.keys())}")

    # ---------- 阶段3：全量5年验证 ----------
    full_results = {}
    print(f"\n{'='*20} 全量5年验证阶段 {'='*20}")

    print(f"\n--- 全量 baseline ---")
    tb = time.time()
    base_full = run_one(data, earnings_calendar, {}, N_WORKERS)
    print(f"  baseline全量耗时: {(time.time()-tb)/60:.1f}分钟")
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

    if not full_candidates:
        print("\n没有候选通过初筛，跳过全量验证阶段。")

    for cand_name, kwargs in full_candidates.items():
        print(f"\n--- 全量 {cand_name} ({kwargs}) ---")
        tc = time.time()
        try:
            res = run_one(data, earnings_calendar, kwargs, N_WORKERS)
            print(f"  {cand_name}全量耗时: {(time.time()-tc)/60:.1f}分钟")
            print(f"\n[{cand_name} 全量报告]")
            run_backtest_with_metrics(res, benchmark_returns=bench_rets)
            full_results[cand_name] = rough_metrics(res)
        except Exception as e:
            print(f"  [错误] {cand_name} 全量验证失败: {type(e).__name__}: {e}")
            full_results[cand_name] = {"error": str(e)}

        with open(RESULT_JSON, "w") as f:
            json.dump({"screen": screen_results, "promoted": promoted, "full": full_results}, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f" 全部完成，总耗时: {(time.time()-t0)/60:.1f} 分钟")
    print(f"{'='*70}")
    with open(RESULT_JSON, "w") as f:
        json.dump({"screen": screen_results, "promoted": promoted, "full": full_results}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
