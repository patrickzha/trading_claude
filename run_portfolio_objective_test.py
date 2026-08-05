"""
训练目标改成直接优化组合结果(listwise/portfolio-level目标)的三窗口初筛。

测两个候选：
1. rank_pairwise：model_objective="ranking"(XGBRanker, rank:pairwise)，
   这个之前"验证结论"第12条测过判定不通过，但那次测试是在新近度加权
   (use_recency_weighting)成为默认值之前——这次实现时发现一个真实bug：
   XGBRanker配合qid分组时，sample_weight要求长度等于分组数、不是样本数，
   之前"三窗口初筛全部不通过"的排序目标测试很可能从来没有正确加上过
   新近度权重(如果那次测试之前就有权重传入的话，应该是每次训练直接崩溃
   被外层except吞掉，等效于空转)。已经在_fit_ensemble里修复(按组收缩
   成per-group权重，数值上跟原来的per-sample权重语义完全等价)，这次是
   在修复+生存偏差修复后的554只股票池上重新测一次，不是简单重复旧结论。
2. rank_ndcg：model_objective="ranking" + xgb_rank_objective="rank:ndcg"，
   真正意义上的"组合级"目标——rank:ndcg要求label是非负整数相关度分级，
   不接受连续收益率，实现时新增了_to_ndcg_relevance()把每组(每周)内的
   收益率按组内排名分成5档(0=最差,4=最好)作为训练标签，直接把"横截面
   相对排序"编码进标签本身，而且NDCG的折损增益公式天然更看重排在最前面
   的几个候选是否排对了(跟实盘只交易前3名高度吻合)，是pairwise(不区分
   位置、只看两两谁大谁小)之外真正不同的机制。

baseline是当前默认配置(model_objective="regression")，都在点位时间正确
的554只股票池上测。
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
    "rank_pairwise": {"model_objective": "ranking", "xgb_rank_objective": "rank:pairwise"},
    "rank_ndcg": {"model_objective": "ranking", "xgb_rank_objective": "rank:ndcg"},
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
    print(" 训练目标改成组合级(listwise)三窗口初筛 —— 点位时间正确股票池")
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

        print("\n--- baseline (regression, 当前默认) ---")
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
