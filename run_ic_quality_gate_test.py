"""
入场质量筛选补充：用"验证结论"第29条发现的诊断结果——模型验证折IC经常
预警("IC 过低，模型可能无效")但从来没有真正被用来做决策，只是打印一条
警告——新增 use_ic_quality_gate / ic_quality_threshold 参数，本周验证折IC
低于阈值时直接空仓，不是"模型不知道能不能信自己"，而是"模型自己测出来
这周排序能力不可信，就不硬选"。

在点位时间正确的554只股票池上做三窗口初筛，方法论跟其他候选一致。
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np

from backtest import run_walk_forward_backtest
from run_batch_candidates import rough_metrics
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
CACHE_PATH = os.path.join(SCRATCH, "point_in_time_universe_cache.pkl")
EARNINGS_CACHE = os.path.join(SCRATCH, "earnings_calendar_cache.pkl")

N_WORKERS = max(1, os.cpu_count() or 1)

SCREEN_CANDIDATES = {
    "ic_gate_0.02": {"use_ic_quality_gate": True, "ic_quality_threshold": 0.02},
    "ic_gate_0.05": {"use_ic_quality_gate": True, "ic_quality_threshold": 0.05},
    "ic_gate_0.0": {"use_ic_quality_gate": True, "ic_quality_threshold": 0.0},
}


def slice_window(cache, end_idx, window_weeks=25, lookback_days=150):
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
    print(" 入场质量筛选(IC gate)三窗口初筛 —— 点位时间正确股票池")
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

        print("\n--- baseline (无IC筛选) ---")
        base_res = run_one(win, earnings_calendar, sector_map, added_dates, removal_dates, {}, N_WORKERS)
        base_metrics = rough_metrics(base_res)
        print(f"  baseline: {base_metrics}")

        for cand_name, kwargs in SCREEN_CANDIDATES.items():
            print(f"\n--- {cand_name} ({kwargs}) ---")
            try:
                res = run_one(win, earnings_calendar, sector_map, added_dates, removal_dates, kwargs, N_WORKERS)
                m = rough_metrics(res)
            except Exception as e:
                print(f"  [错误] {cand_name} 在 {win_name} 跑失败: {type(e).__name__}: {e}")
                m = {"cum": None, "mdd": None, "sharpe": None, "n_weeks": 0, "error": str(e)}
            print(f"  {cand_name}: {m}")
            screen_results.setdefault(cand_name, {})[win_name] = {"baseline": base_metrics, "treatment": m}

    print(f"\n{'='*20} 初筛汇总与晋级判定 {'='*20}")
    promoted = []
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
            print(f"  {cand_name}: 跑失败，不晋级")
            continue
        same_direction = all(d > 0 for d in deltas) or all(d < 0 for d in deltas)
        verdict = "晋级全量验证" if all(d > 0 for d in deltas) else ("不晋级(一致更差)" if same_direction else "不晋级(方向不一致/噪声)")
        print(f"  {cand_name}: 三窗口Sharpe差值={[round(d,2) for d in deltas]} -> {verdict}")
        if all(d > 0 for d in deltas):
            promoted.append(cand_name)

    print(f"\n晋级候选: {promoted}")
    print(f"总耗时: {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
