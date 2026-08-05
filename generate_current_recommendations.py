"""
Cycle9：实用价值收尾——生成"如果现在要交易，三条腿分别选什么"的真实
输出。用最新可用的价格缓存(point_in_time_universe_cache.pkl，截止到
2026-07-30，如果要真正实盘应该重新拉当天数据，这里为了演示完整可用性
复用已有缓存，日期差距在几天内对结果没有实质影响)，跑：
1. ML周频动量策略：当前决策日的3只候选
2. 长周期价值策略：本月的30只候选(按第276条推荐配置展示HRP权重+3%止损)
3. 低波动策略：本月的30只候选(按第285条推荐配置维持等权+5%止损)

如实提醒：这是"如果现在要用这套系统"的真实输出格式演示，不是投资建议，
所有已知局限(生存偏差残留、选股模型排序能力弱、组合alpha只有2/3区间
显著)都适用于这里生成的具体候选名单。

【Cycle64更新，第290条】此前这里只展示等权的top10候选名单，跟README
"现状总结"里实际推荐的配置(`value_weighting="hrp"`+`value_stop_loss_pct=
0.03`+`value_rebalance_days=5`，第276/278/280条完整验证过)不一致——这次
补上HRP权重展示，让这个"最终用户可见"的脚本真正反映这次会话验证过的
最佳配置，不是一个过时的简化演示。
"""
from __future__ import annotations

import pickle

import pandas as pd

from us_stock_screener import get_ai_weekly_picks
from value_investing import select_value_portfolio, _hrp_weights
from low_vol_investing import select_low_vol_portfolio
from run_survivorship_bias_fix_validation import compute_point_in_time_membership, CACHE_PATH


def main():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    c_df, o_df, h_df, l_df, v_df, vix = cache["c"], cache["o"], cache["h"], cache["l"], cache["v"], cache["vix"]
    sector_map = cache["sector_map"]

    decision_date = c_df.index[-2]  # 留一天给"今天"当预测目标日
    print(f"决策日: {decision_date.date()}（数据缓存最新日期: {c_df.index[-1].date()}）")
    print("如实提醒：这是缓存数据的演示输出，不是今天实时数据，仅用于展示系统")
    print("完整可用性；如要真正参考，需要先重新拉取截止今天的最新价格数据。\n")

    membership = compute_point_in_time_membership(c_df.index, list(c_df.columns),
                                                    cache["added_dates"], cache["removal_dates"])
    date_str = decision_date.strftime("%Y-%m-%d")
    valid_tickers = membership.get(date_str)
    if valid_tickers is None:
        keys = sorted(membership.keys())
        valid_tickers = membership[max([k for k in keys if k <= date_str], default=keys[0])]

    print("=" * 60)
    print(" 1. ML周频动量策略候选(本周)")
    print("=" * 60)
    try:
        picks = get_ai_weekly_picks(
            c_df, o_df, h_df, l_df, v_df, vix, decision_date,
            sector_map=sector_map, enable_earnings_blackout=False,
        )
        if picks:
            for ticker, weight, pred_ret, pos_size in picks:
                print(f"  {ticker}: 权重={weight:.2%}, 预测收益率={pred_ret:+.4f}, 仓位系数={pos_size:.2f}")
        else:
            print("  本周宏观熔断触发或无合格候选，建议空仓。")
    except Exception as e:
        print(f"  [错误] {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print(" 2. 长周期价值策略候选(本月，HRP权重(单票上限30%，第293条)，第276条推荐配置：value_stop_loss_pct=0.03)")
    print("=" * 60)
    val_picks = select_value_portfolio(c_df, valid_tickers, decision_date, top_n=30)
    hist_window = c_df[val_picks].loc[:decision_date].tail(61)
    rets_window = hist_window.pct_change().dropna()
    valid_cols = rets_window.columns[rets_window.notna().all()]
    if len(valid_cols) >= 5:
        # max_weight=0.3(第290/293条)：极少数调仓日会出现单票权重顶格70%+的
        # 情况(已实现波动率极端离群导致)，历史回测层面几乎零成本，实盘层面
        # 能避免"一次调仓意外全仓压一只票"的尾部风险，这个展示脚本默认开启。
        hrp_w = _hrp_weights(rets_window[valid_cols], max_weight=0.3)
        weights = pd.Series(0.0, index=val_picks)
        for t in valid_cols:
            weights[t] = hrp_w[t]
        weights = weights / weights.sum() if weights.sum() > 0 else pd.Series(1.0 / len(val_picks), index=val_picks)
        for t in weights.sort_values(ascending=False).head(10).index:
            print(f"  {t}: HRP权重={weights[t]:.2%}")
    else:
        print("  [警告] 协方差窗口数据不足，无法算HRP权重，回退显示前10只(等权)")
        for t in val_picks[:10]:
            print(f"  {t}")

    print("\n" + "=" * 60)
    print(" 3. 低波动策略候选(本月，按60日已实现波动率排名前10)")
    print("=" * 60)
    lv_picks = select_low_vol_portfolio(c_df, valid_tickers, decision_date, top_n=30)
    for t in lv_picks[:10]:
        print(f"  {t}")

    print("\n" + "=" * 60)
    print(" 如实提醒：三元组合(动量/价值/低波动)不是把三份名单合并成一份统一名单")
    print(" 【第383条更新，2026-08】权重从等权精调成`{value:0.50, low_vol:0.40,")
    print(" momentum:0.10}`(第366/377条)，价值腿止损从`value_stop_loss_pct=")
    print(" 0.03`进一步收紧到`0.02`(第383条，价值腿权重提高后止损重要性联动")
    print(" 提高，完整走过点估计/触发比例诊断/显著性检验三步流程，三段区间")
    print(" 全部P=0.000)——动量腿标准显著性检验3段区间全部不通过(第307条)是")
    print(" 权重降低的原始理由，完整系统层面3段区间Sharpe/CAGR全部同步改善且")
    print(" 多次block bootstrap检验通过，是这次会话证据最扎实的配置更新之一，")
    print(" 不再是等权——价值腿内部用HRP权重(第159/276条)、`value_rebalance_")
    print(" days=5`，低波动腿内部等权、`low_vol_stop_loss_pct=0.05`(第285条")
    print(" 确认不宜进一步收紧)，完整调用方式见`full_system.py`的`run_full_")
    print(" system(leg_weights=..., value_stop_loss_pct=0.02)`。")
    print(" 这不是已证实的alpha来源，只是目前测试过的所有方向里证据最扎实的配置。")


if __name__ == "__main__":
    main()
