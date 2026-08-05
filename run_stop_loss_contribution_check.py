"""
Cycle19小修小补(第4条)：第95条推断"逐票2%止损本身已经把组合层面回撤控制
住了，不是哪一层熔断抢跑"，这次用已经缓存好的三段区间trade级别数据
(mom_results里每条trade记录都有exit_reason字段)直接量化止损的触发频率
和它对单笔交易结果的影响，把第95条的推断坐实成具体数字，而不是停留在
"止损触发了100次"这种单一数字上。
"""
from __future__ import annotations

import pickle

import numpy as np

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"

periods = [
    ("2022-2026", f"{SCRATCH}/default_full_results_cache.pkl"),
    ("2014-2020(含2018Q4暴跌+2020新冠暴跌)", f"{SCRATCH}/momentum_2014_2020_results_cache.pkl"),
    ("2009-2014", f"{SCRATCH}/momentum_2009_2014_results_cache.pkl"),
]

for label, path in periods:
    with open(path, "rb") as f:
        mom_results = pickle.load(f)

    all_trades = [t for r in mom_results for t in r.get("trades", [])]
    if not all_trades:
        print(f"\n=== {label} ===\n  无trade级别数据(缓存里没有trades字段)，跳过")
        continue

    n_total = len(all_trades)
    stop_loss_trades = [t for t in all_trades if t.get("exit_reason") == "stop_loss"]
    n_sl = len(stop_loss_trades)
    other_trades = [t for t in all_trades if t.get("exit_reason") != "stop_loss"]

    sl_rets = [t["trade_ret"] for t in stop_loss_trades if "trade_ret" in t]
    other_rets = [t["trade_ret"] for t in other_trades if "trade_ret" in t]
    all_rets = [t["trade_ret"] for t in all_trades if "trade_ret" in t]

    worst_without_sl_cap = min(other_rets) if other_rets else None
    worst_with_sl = min(sl_rets) if sl_rets else None

    print(f"\n=== {label} ===")
    print(f"  总交易数: {n_total}, 止损触发: {n_sl}({n_sl/n_total:.1%})")
    print(f"  止损触发的交易平均单笔亏损: {np.mean(sl_rets)*100:.2f}%" if sl_rets else "  无止损交易")
    print(f"  非止损交易平均单笔收益: {np.mean(other_rets)*100:.2f}%" if other_rets else "  无非止损交易")
    print(f"  非止损交易里最差单笔: {worst_without_sl_cap*100:.2f}%（如果没有止损，单笔亏损理论上可以比这更深）" if worst_without_sl_cap else "")
    print(f"  止损交易里最差单笔: {worst_with_sl*100:.2f}%（止损把单笔最坏结果限制在这个量级附近）" if worst_with_sl else "")
    print(f"  全部交易收益标准差: {np.std(all_rets)*100:.2f}%")
