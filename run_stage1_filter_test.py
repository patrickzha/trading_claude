"""
两阶段架构(粗筛500->健康候选 + 精选模型选3只)的三窗口初筛。第一阶段用
最简单的规则粗筛：跌破近期均线(默认50日)的候选直接排除，不用模型、成本
几乎为0，让第二阶段(现有XGBoost模型)只在"还在趋势里"的候选池里挑，
而不是在全市场(含大量明显走弱的票)里矮子里拔将军。

用use_stage1_filter开关，只影响预测阶段的候选池(训练样本不变，训练仍然
用全历史，跟"点位时间正确"的过滤方式是同一个模式：只收窄推理时看到的
候选范围)。测两个均线窗口：50日(标准)、20日(更激进、更贴近动量本身的
时间尺度)。
"""
from __future__ import annotations

import os
import pickle
import time

from backtest import run_walk_forward_backtest
from run_batch_candidates import rough_metrics
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
CACHE_PATH = os.path.join(SCRATCH, "point_in_time_universe_cache.pkl")
EARNINGS_CACHE = os.path.join(SCRATCH, "earnings_calendar_cache.pkl")

N_WORKERS = max(1, os.cpu_count() or 1)

SCREEN_CANDIDATES = {
    "stage1_ma50": {"use_stage1_filter": True, "stage1_ma_window": 50},
    "stage1_ma20": {"use_stage1_filter": True, "stage1_ma_window": 20},
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
    print(" 两阶段架构(粗筛均线健康度)三窗口初筛 —— 点位时间正确股票池")
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

        print("\n--- baseline (无粗筛) ---")
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
