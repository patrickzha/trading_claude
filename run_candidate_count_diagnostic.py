"""
Cycle20小修小补：动量腿每周入选候选数(0-3只，宏观熔断/入场质量筛选/合格
候选不足时会少于3只甚至空仓)有没有系统性地跟随行情好坏变化——如果候选
数量本身就是一个隐性的"模型对当前环境有没有信心"的信号，理论上可以拿来
做二次过滤或者仓位调整参考(不是这次就要采纳成新功能，是先看这个诊断
信号本身站不站得住脚)。用已缓存的三段trade级别数据，比较"候选数=3"跟
"候选数<3"两组周的后续实际表现(net_worth/weekly_return)有没有系统性差异。
"""
from __future__ import annotations

import pickle

import numpy as np

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"

periods = [
    ("2022-2026", f"{SCRATCH}/default_full_results_cache.pkl"),
    ("2014-2020", f"{SCRATCH}/momentum_2014_2020_results_cache.pkl"),
    ("2009-2014", f"{SCRATCH}/momentum_2009_2014_results_cache.pkl"),
]

for label, path in periods:
    with open(path, "rb") as f:
        mom_results = pickle.load(f)

    counts = [len(r.get("picks", [])) for r in mom_results]
    rets = [r["weekly_return"] for r in mom_results]

    full_count = [r["weekly_return"] for r, c in zip(mom_results, counts) if c == 3]
    partial_count = [r["weekly_return"] for r, c in zip(mom_results, counts) if 0 < c < 3]
    zero_count = [r["weekly_return"] for r, c in zip(mom_results, counts) if c == 0]

    print(f"\n=== {label} ===")
    print(f"  候选数分布: {dict(zip(*np.unique(counts, return_counts=True)))}")
    print(f"  候选数=3的周(n={len(full_count)}): 平均周收益={np.mean(full_count)*100:.3f}%, 标准差={np.std(full_count)*100:.3f}%" if full_count else "  无候选数=3的周")
    print(f"  候选数1-2的周(n={len(partial_count)}): 平均周收益={np.mean(partial_count)*100:.3f}%, 标准差={np.std(partial_count)*100:.3f}%" if partial_count else "  无候选数1-2的周")
    print(f"  候选数=0(空仓)的周(n={len(zero_count)}): (空仓周本身收益恒为0，不比较)")
    print(f"  全部周平均收益(基线对照): {np.mean(rets)*100:.3f}%")
