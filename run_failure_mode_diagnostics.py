"""
更细粒度的失败模式分析（诊断，不是候选测试）。用当前默认配置（止损2%、
RSI阈值100、新近度加权26周）在点位时间正确的554只股票池上跑一次完整5年
回测，这次 backtest.py 的 results 里带上了每笔交易的明细（ticker/sector/
pred_ret/net_ret/exit_reason），用来回答几个问题：

1. 模型预测的收益率高低，是否真的跟实际结果正相关？（如果不是，说明模型
   排序能力本身就有问题，不是止损/仓位这些下游规则的问题）
2. 止损触发的交易占了多大比例的亏损？跟"扛到周五"的交易比，哪种亏得更多？
3. 亏损是否集中在某几个行业？
4. pred_ret 最高的那批候选（模型最有信心的），实际表现是不是真的最好？

结果原始数据也会 pickle 下来，方便后续"入场质量筛选补充"直接复用，不用
重新跑一次5年回测。
"""
from __future__ import annotations

import pickle
import time

import numpy as np

from backtest import run_walk_forward_backtest
from run_survivorship_bias_fix_validation import compute_point_in_time_membership, CACHE_PATH, EARNINGS_CACHE

RESULTS_PICKLE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/pit_full_run_with_trades.pkl"


def analyze(results):
    all_trades = []
    for r in results:
        for t in r["trades"]:
            t = dict(t)
            t["date"] = r["date"]
            all_trades.append(t)

    print(f"\n总交易笔数: {len(all_trades)}")

    # ---- 1. pred_ret 排序能力：按 pred_ret 分5档，看每档实际 net_ret 均值 ----
    pred_rets = np.array([t["pred_ret"] for t in all_trades], dtype=float)
    net_rets = np.array([t["net_ret"] for t in all_trades], dtype=float)
    order = np.argsort(pred_rets)
    n = len(all_trades)
    print("\n【1. 按预测收益率分5档，看实际净收益率是否单调】")
    for q in range(5):
        idx = order[int(q * n / 5):int((q + 1) * n / 5)]
        if len(idx) == 0:
            continue
        print(f"  Q{q+1}(pred_ret均值{pred_rets[idx].mean():+.4f}): "
              f"n={len(idx)}, 实际net_ret均值={net_rets[idx].mean():+.4f}, "
              f"胜率={np.mean(net_rets[idx] > 0):.1%}")
    corr = np.corrcoef(pred_rets, net_rets)[0, 1]
    print(f"  pred_ret vs net_ret 全样本相关系数: {corr:.4f}")

    # ---- 2. 止损 vs 非止损交易的贡献 ----
    print("\n【2. 止损触发 vs 未触发止损，两类交易各自表现】")
    for reason in ["stop_loss", "none"]:
        sub = [t["net_ret"] for t in all_trades if t["exit_reason"] == reason]
        if sub:
            print(f"  {reason}: n={len(sub)}({len(sub)/n:.1%}), "
                  f"net_ret均值={np.mean(sub):+.4f}, 合计贡献={np.sum(sub):+.2f}")

    # ---- 3. 按行业拆分 ----
    print("\n【3. 按行业拆分平均净收益率与胜率】")
    sectors = sorted(set(t["sector"] for t in all_trades))
    sector_stats = []
    for s in sectors:
        sub = [t["net_ret"] for t in all_trades if t["sector"] == s]
        if len(sub) >= 5:
            sector_stats.append((s, len(sub), np.mean(sub), np.mean(np.array(sub) > 0)))
    sector_stats.sort(key=lambda x: x[2])
    for s, cnt, avg, wr in sector_stats:
        print(f"  {s:10s}: n={cnt:4d}, net_ret均值={avg:+.4f}, 胜率={wr:.1%}")

    # ---- 4. 最有信心 vs 最没信心的候选（当周pred_ret排名） ----
    print("\n【4. 当周3个候选里，pred_ret排名第1/2/3的分别表现如何】")
    rank_buckets = {0: [], 1: [], 2: []}
    for r in results:
        trades_sorted = sorted(r["trades"], key=lambda t: t["pred_ret"], reverse=True)
        for rank, t in enumerate(trades_sorted[:3]):
            rank_buckets[rank].append(t["net_ret"])
    for rank in range(3):
        sub = rank_buckets[rank]
        if sub:
            print(f"  当周第{rank+1}高预测: n={len(sub)}, net_ret均值={np.mean(sub):+.4f}, 胜率={np.mean(np.array(sub)>0):.1%}")


def main():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    c_df, o_df, h_df, l_df, v_df, vix = cache["c"], cache["o"], cache["h"], cache["l"], cache["v"], cache["vix"]
    sector_map, added_dates, removal_dates = cache["sector_map"], cache["added_dates"], cache["removal_dates"]

    membership = compute_point_in_time_membership(c_df.index, list(c_df.columns), added_dates, removal_dates)

    with open(EARNINGS_CACHE, "rb") as f:
        earnings_calendar = pickle.load(f)

    import os
    n_workers = max(1, os.cpu_count() or 1)
    t0 = time.time()
    results = run_walk_forward_backtest(
        c_df, o_df, h_df, l_df, v_df, vix,
        enable_shap=False, n_workers=n_workers, xgb_n_jobs=1,
        earnings_calendar=earnings_calendar,
        sector_map=sector_map,
        point_in_time_membership=membership,
    )
    print(f"\n回测完成: {len(results)}周, 耗时{(time.time()-t0)/60:.1f}分钟")

    with open(RESULTS_PICKLE, "wb") as f:
        pickle.dump(results, f)

    analyze(results)


if __name__ == "__main__":
    main()
