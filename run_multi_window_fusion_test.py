"""
时间维度模型融合(300天/600天窗口模型融合)的三窗口初筛。用新增的
use_multi_window_fusion / fusion_window_days 参数：每周除了默认的300天
滚动窗口模型，额外训练一个600天窗口的模型，两边预测收益率各半加权融合
后再排序选股，直觉是短窗口适应regime变化快、长窗口样本多更稳，融合起来
可能互补。

生存偏差修复之后（见README"验证结论"第27条），这次改用点位时间正确的
554只股票池做初筛，不再用旧的、把"今天"名单套历史的有偏股票池——避免
在一个已知有严重前视偏差的数据上做决策。

600天窗口需要更长的历史预热（tail(660天)），数据总长度只有约5年
(1254个交易日)，三个窗口里最早的"早段窗口"预热会不够完整，如实记录、
不强行拉长（拉长会挤到2021年之前，缓存里没有这段数据）。
"""
from __future__ import annotations

import json
import os
import pickle
import time

from backtest import run_walk_forward_backtest, get_weekly_trading_dates
from run_batch_candidates import rough_metrics
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
CACHE_PATH = os.path.join(SCRATCH, "point_in_time_universe_cache.pkl")
EARNINGS_CACHE = os.path.join(SCRATCH, "earnings_calendar_cache.pkl")
RESULT_JSON = os.path.join(SCRATCH, "multi_window_fusion_screen_result.json")

N_WORKERS = max(1, os.cpu_count() or 1)


def slice_window(cache, end_idx, window_weeks=25, lookback_days=785):
    start_idx = max(0, end_idx - window_weeks * 5 - lookback_days)
    idx_slice = cache["c"].index[start_idx:end_idx]
    return {
        "c": cache["c"].loc[idx_slice], "o": cache["o"].loc[idx_slice],
        "h": cache["h"].loc[idx_slice], "l": cache["l"].loc[idx_slice],
        "v": cache["v"].loc[idx_slice], "vix": cache["vix"].loc[idx_slice],
    }


def run_one(win, earnings_calendar, sector_map, added_dates, removal_dates, extra_kwargs, n_workers):
    membership = compute_point_in_time_membership(win["c"].index, list(win["c"].columns), added_dates, removal_dates)
    return run_walk_forward_backtest(
        win["c"], win["o"], win["h"], win["l"], win["v"], win["vix"],
        enable_shap=False, n_workers=n_workers, xgb_n_jobs=1,
        earnings_calendar=earnings_calendar,
        sector_map=sector_map,
        point_in_time_membership=membership,
        **extra_kwargs,
    )


def main():
    t0 = time.time()
    print("=" * 70)
    print(" 时间维度模型融合(300/600天窗口)三窗口初筛 —— 点位时间正确股票池")
    print("=" * 70)

    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    with open(EARNINGS_CACHE, "rb") as f:
        earnings_calendar = pickle.load(f)
    sector_map = cache["sector_map"]
    added_dates = cache["added_dates"]
    removal_dates = cache["removal_dates"]

    n_total = len(cache["c"].index)
    print(f"数据总长度: {n_total}天, {cache['c'].index[0].date()} ~ {cache['c'].index[-1].date()}")
    print(f"并行worker数: {N_WORKERS}")

    window_ends = {"近25周": n_total, "中段窗口(约1年前)": n_total - 300, "早段窗口(约2年前)": n_total - 600}

    screen_results = {}
    for win_name, end_idx in window_ends.items():
        win = slice_window(cache, end_idx)
        print(f"\n{'='*20} 初筛窗口: {win_name} ({len(win['c'])}天) {'='*20}")

        print("\n--- baseline (无融合) ---")
        base_res = run_one(win, earnings_calendar, sector_map, added_dates, removal_dates, {}, N_WORKERS)
        base_metrics = rough_metrics(base_res)
        print(f"  baseline: {base_metrics}")

        print("\n--- multi_window_fusion (300+600天融合) ---")
        try:
            res = run_one(win, earnings_calendar, sector_map, added_dates, removal_dates,
                          {"use_multi_window_fusion": True, "fusion_window_days": 600}, N_WORKERS)
            m = rough_metrics(res)
        except Exception as e:
            print(f"  [错误] 融合在 {win_name} 跑失败: {type(e).__name__}: {e}")
            m = {"cum": None, "mdd": None, "sharpe": None, "n_weeks": 0, "error": str(e)}
        print(f"  fusion: {m}")
        screen_results[win_name] = {"baseline": base_metrics, "fusion": m}

        with open(RESULT_JSON, "w") as f:
            json.dump(screen_results, f, indent=2, default=str)

    print(f"\n{'='*20} 初筛汇总 {'='*20}")
    deltas = []
    broken = False
    for win_name, pair in screen_results.items():
        t, b = pair["fusion"], pair["baseline"]
        if t.get("sharpe") is None or t.get("n_weeks", 0) == 0:
            broken = True
            print(f"  {win_name}: 跑失败，跳过判定")
            continue
        d = t["sharpe"] - b["sharpe"]
        deltas.append(d)
        print(f"  {win_name}: baseline_sharpe={b['sharpe']:.2f} fusion_sharpe={t['sharpe']:.2f} delta={d:.2f}")

    if broken:
        print("\n结论: 有窗口跑失败，不判定，需要先排查错误。")
    elif all(d > 0 for d in deltas):
        print(f"\n结论: 三窗口方向一致为正 {[round(d,2) for d in deltas]}，晋级全量5年验证。")
    elif all(d < 0 for d in deltas):
        print(f"\n结论: 三窗口方向一致为负 {[round(d,2) for d in deltas]}，不采纳。")
    else:
        print(f"\n结论: 三窗口方向不一致 {[round(d,2) for d in deltas]}，判定为噪声，不采纳。")

    print(f"\n总耗时: {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
