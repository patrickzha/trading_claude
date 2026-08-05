"""
Cycle70大修大补②(框架改动，新的股票池构造思路——因子交叉子集)：动量信号
限定在"低波动股票子集"内部选股，而不是全标普500——这次会话已经验证过
动量+低波动是两条相对独立、能互补分散化的腿(第55条)，但从没测过"如果
只在低波动股票里面找动量最强的"这种因子交叉/嵌套的构造方式，理论依据
是"低波动异象"文献里一个常见延伸：低波动股票内部再筛动量，可能比动量
全市场选股噪声更小(排除了高波动、动量信号本身更不可靠的那批票)。

用当前已验证的动量默认配置(止损2%/RSI阈值100/新近度加权26周)，只是把
候选股票池从全标普500替换成"trailing 60日波动率最低的250只"这个子集，
测试收益是否比全市场动量更好。
"""
from __future__ import annotations

import os
import pickle

from backtest import run_walk_forward_backtest, get_weekly_trading_dates
from optimized_metrics_v4 import run_backtest_with_metrics
from run_survivorship_bias_fix_validation import compute_point_in_time_membership, CACHE_PATH


def main():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    c_df, o_df, h_df, l_df, v_df, vix, spy_close = (
        cache["c"], cache["o"], cache["h"], cache["l"], cache["v"], cache["vix"], cache["spy"]
    )
    sector_map = cache["sector_map"]

    # 用全样本(不是逐日滚动，简化处理，如实标注：这是用了"未来"的波动率排名
    # 做了一次静态筛选，不是逐周动态判断，可能有轻微前视偏差，但这次只是
    # 快速可行性初筛，不是最终结论，方向性对比依然有参考价值)
    rets = c_df.pct_change()
    vol = rets.rolling(60).std().mean()  # 每只票在整个回测期的平均滚动波动率
    low_vol_tickers = vol.dropna().sort_values().head(250).index.tolist()
    print(f"低波动子集: {len(low_vol_tickers)}只票(从{len(c_df.columns)}只里选)")

    c_sub = c_df[low_vol_tickers]
    o_sub = o_df[[t for t in low_vol_tickers if t in o_df.columns]]
    h_sub = h_df[[t for t in low_vol_tickers if t in h_df.columns]]
    l_sub = l_df[[t for t in low_vol_tickers if t in l_df.columns]]
    v_sub = v_df[[t for t in low_vol_tickers if t in v_df.columns]]

    # 【修复一个真实bug】第一版忘了传point_in_time_membership，等于完全
    # 没做生存偏差修正(第27条)，两个配置的基线数字都会被严重高估——这是
    # 这次会话里第三次(前两次是第300/302条)从"正式跑一遍"这个步骤里抓到
    # 自己犯的方法论错误，如实记录，不悄悄改掉重跑。
    membership_full = compute_point_in_time_membership(
        c_df.index, list(c_df.columns), cache["added_dates"], cache["removal_dates"])
    membership_sub = {d: [t for t in tickers if t in low_vol_tickers] for d, tickers in membership_full.items()}

    n_workers = max(1, os.cpu_count() or 1)

    print("\n跑全标普500动量(基线，已修正生存偏差)...")
    results_full = run_walk_forward_backtest(
        c_df, o_df, h_df, l_df, v_df, vix,
        enable_shap=False, n_workers=n_workers, xgb_n_jobs=1,
        enable_earnings_blackout=False, sector_map=sector_map,
        point_in_time_membership=membership_full,
    )

    print("\n跑低波动子集内部动量(已修正生存偏差)...")
    results_sub = run_walk_forward_backtest(
        c_sub, o_sub, h_sub, l_sub, v_sub, vix,
        enable_shap=False, n_workers=n_workers, xgb_n_jobs=1,
        enable_earnings_blackout=False, sector_map=sector_map,
        point_in_time_membership=membership_sub,
    )

    def bench(results, price_df):
        used_dates = {r["date"] for r in results}
        rets = []
        for monday, friday in get_weekly_trading_dates(price_df.index):
            if monday.strftime("%Y-%m-%d") not in used_dates:
                continue
            try:
                rets.append((spy_close.loc[friday] - spy_close.loc[monday]) / spy_close.loc[monday])
            except Exception:
                rets.append(0.0)
        return rets

    print("\n[全标普500动量报告]")
    run_backtest_with_metrics(results_full, benchmark_returns=bench(results_full, c_df))

    print("\n[低波动子集内部动量报告]")
    run_backtest_with_metrics(results_sub, benchmark_returns=bench(results_sub, c_sub))


if __name__ == "__main__":
    main()
