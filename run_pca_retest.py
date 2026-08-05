"""
PCA特征去相关(use_pca_features)修复NaN处理bug后的重新验证——之前那版三窗口
初筛因为PCA遇到NaN直接崩溃、大量周退化成空仓，产出的Sharpe差值是几百万这种
明显异常的数字，判定结果本身不可信。修复（PCA前先用训练折均值填补缺失值）
后在同一套三窗口方法论上重新测一遍。

用 ProcessPoolExecutor（spawn方式）必须把实际执行逻辑放进
if __name__ == "__main__": 保护起来，否则每个子进程重新 import 这个模块时
会把顶层代码再跑一遍，递归再起子进程，直接死循环——上一版忘了加这个保护，
脚本卡死不报错，已经手动kill掉重写。
"""
from __future__ import annotations

import pickle

from run_batch_candidates import slice_window, run_one, rough_metrics, DATA_CACHE, EARNINGS_CACHE


def main():
    with open(DATA_CACHE, "rb") as f:
        full = pickle.load(f)
    data = {k: full[k] for k in ["c", "o", "h", "l", "v", "vix"]}
    with open(EARNINGS_CACHE, "rb") as f:
        earnings_calendar = pickle.load(f)

    n_total = len(data["c"].index)
    window_ends = {"近25周": n_total, "中段窗口(约1年前)": n_total - 300, "早段窗口(约2年前)": n_total - 600}

    deltas = []
    for win_name, end_idx in window_ends.items():
        win = slice_window(data, end_idx)
        print(f"\n{'='*20} {win_name} {'='*20}")
        base_res = run_one(win, earnings_calendar, {}, 8)
        base_m = rough_metrics(base_res)
        print(f"  baseline: {base_m}")
        pca_res = run_one(win, earnings_calendar, {"use_pca_features": True}, 8)
        pca_m = rough_metrics(pca_res)
        print(f"  pca_features: {pca_m}")
        delta = pca_m["sharpe"] - base_m["sharpe"]
        deltas.append(delta)

    print(f"\n三窗口Sharpe差值: {[round(d,2) for d in deltas]}")
    if all(d > 0 for d in deltas):
        print("-> 方向一致更好，晋级全量验证")
    elif all(d < 0 for d in deltas):
        print("-> 方向一致更差，不晋级")
    else:
        print("-> 方向不一致，不晋级")


if __name__ == "__main__":
    main()
