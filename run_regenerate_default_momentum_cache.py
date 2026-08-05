"""
Cycle71大修大补①(重大修正)：`default_full_results_cache.pkl`(这次会话
2022-2026区间被引用次数最多的动量腿缓存，几乎所有combo_strategy/
full_system测试都拿它当mom_results用)从来没有做过生存偏差修正——
生成它的脚本(`run_diagnostics_e.py`)用的是`sp500_data_cache.pkl`(当前
不存在了，未经点位时间扩展的普通503只票缓存)，调用`run_walk_forward_
backtest`时也没有传`point_in_time_membership`。2014-2020/2009-2014两段
区间早在Cycle22(第108-110条)就做过同等修正(有`_PRE_PIT_BACKUP`备份
文件为证)，但2022-2026这段被漏掉了，一直沿用到现在。

直接验证：`default_full_results_cache.pkl`自己算出来的Sharpe=1.47，
而这次(Cycle70，第306条)用正确点位时间修正跑出来的同一区间基线是
Sharpe=-0.02——量级差距和第27条当年发现的"37.15%→3.20%"是同一类问题。

这次重新生成一份正确版本，覆盖原文件(保留一份备份)，供后续所有引用
`default_full_results_cache.pkl`的地方使用。
"""
from __future__ import annotations

import os
import pickle
import shutil

from backtest import run_walk_forward_backtest
from run_survivorship_bias_fix_validation import compute_point_in_time_membership, CACHE_PATH

RESULTS_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/default_full_results_cache.pkl"
BACKUP_PATH = RESULTS_CACHE.replace(".pkl", "_PRE_PIT_BACKUP.pkl")


def main():
    if os.path.exists(RESULTS_CACHE) and not os.path.exists(BACKUP_PATH):
        shutil.copy(RESULTS_CACHE, BACKUP_PATH)
        print(f"已备份原文件到 {BACKUP_PATH}")

    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    c_df, o_df, h_df, l_df, v_df, vix = cache["c"], cache["o"], cache["h"], cache["l"], cache["v"], cache["vix"]
    sector_map = cache["sector_map"]

    membership = compute_point_in_time_membership(
        c_df.index, list(c_df.columns), cache["added_dates"], cache["removal_dates"])

    n_workers = max(1, os.cpu_count() or 1)
    print("跑一次当前默认配置的全量5年回测(已正确应用point_in_time_membership + 完整sector_map)...")
    results = run_walk_forward_backtest(
        c_df, o_df, h_df, l_df, v_df, vix,
        enable_shap=False, n_workers=n_workers, xgb_n_jobs=1,
        enable_earnings_blackout=False,
        point_in_time_membership=membership,
        sector_map=sector_map,
    )
    with open(RESULTS_CACHE, "wb") as f:
        pickle.dump(results, f)
    print(f"已用正确的PIT修正版本覆盖 {RESULTS_CACHE}")

    import numpy as np
    import pandas as pd
    df = pd.DataFrame(results)
    rets = df["weekly_return"].fillna(0.0)
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(52))
    print(f"新缓存自身的Sharpe(粗算): {sharpe:.3f}(应该接近-0.02附近，不是1.47)")


if __name__ == "__main__":
    main()
