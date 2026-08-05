"""
每周实盘信号生成器
==================
这个项目从一开始的目的就是"周一买、周五卖"，但之前所有代码都是回测/诊断
工具，没有一个能在周一早上直接跑一下、告诉你这周该买哪几只、买多少仓位的
入口。这个脚本就是补这一块。

用法：
    python generate_weekly_signal.py

每次运行会：
1. 拉取到今天为止的最新数据
2. 用当前验证过的默认配置（宏观熔断两条腿 + 财报避让 + 账户级回撤熔断）
   生成这一周的候选
3. 把结果打印出来，同时追加写入 weekly_signal_log.csv——这份日志是长期
   验证"这套系统在实盘里到底灵不灵"的唯一办法：回测数字再好看，都不能
   代替真实、未来才知道结果的记录。几个月后可以拿这份日志里的信号去对照
   实际走势，看跟回测预期是否一致。

注意：这个脚本只生成信号，不接交易接口、不自动下单——要不要真的按这个
信号操作，是你自己的决定，脚本只是把"模型这周怎么想"如实记录下来。
"""
from __future__ import annotations

import csv
import os
from datetime import datetime

from backtest import fetch_market_data, fetch_earnings_calendar
from us_stock_screener import get_ai_weekly_picks, SECTOR_MAP

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weekly_signal_log.csv")
EARNINGS_CACHE_DAYS = 7  # 财报日历缓存有效期，避免每周都重新拉500只票


def _load_or_fetch_earnings_calendar():
    import pickle
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".earnings_calendar_cache.pkl")
    if os.path.exists(cache_path):
        age_days = (datetime.now().timestamp() - os.path.getmtime(cache_path)) / 86400
        if age_days < EARNINGS_CACHE_DAYS:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
    calendar = fetch_earnings_calendar(list(SECTOR_MAP.keys()), n_workers=16)
    with open(cache_path, "wb") as f:
        pickle.dump(calendar, f)
    return calendar


def main():
    print("=" * 70)
    print(" 每周实盘信号生成")
    print("=" * 70)
    print(f" 运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" 股票池: 标普500全量（{len(SECTOR_MAP)}只）")
    print()

    print("拉取最新市场数据...")
    close_df, open_df, high_df, low_df, volume_df, vix, _ = fetch_market_data(days=800)
    target_date = close_df.index[-1]
    print(f"最新交易日: {target_date.date()}")

    print("拉取/复用财报日历...")
    earnings_calendar = _load_or_fetch_earnings_calendar()

    print("\n生成本周信号...")
    # 用默认单模型配置（不加正则化、不做集成）——这两个都在5年标普500
    # 回测里实测过，结果比默认单模型更差（正则化：超额收益+4.16%->+1.76%；
    # 集成size=5：超额收益变成-5.80%，反而跑输大盘），不是"没试过更花哨的
    # 方法"，是试过了，确认默认配置目前最好，没有再加码的必要。
    picks = get_ai_weekly_picks(
        close_df, open_df, high_df, low_df, volume_df, vix, target_date,
        shap_enabled=False,
        earnings_calendar=earnings_calendar,
        xgb_n_jobs=max(1, (os.cpu_count() or 1) - 1),
    )

    print("\n" + "=" * 70)
    print(f" 【{target_date.date()} 收盘后生成 → 下周一开盘买入】")
    print("=" * 70)
    if not picks:
        print(" 本周无信号（可能触发了宏观熔断，或候选全部被过滤）——建议空仓，现金按年化4%计息。")
    else:
        # "权重"是风险平价算出来的相对分配（几只票加起来=100%），"仓位系数"
        # 是基于预测置信度的额外折扣（越有把握的票系数越接近1，没把握的票
        # 打折）。真正该投入的资金比例 = 权重 x 仓位系数，两者分开看容易让人
        # 不知道到底该投多少，这里直接算出"实际建议仓位%"，并把折扣出来的
        # 剩余部分显式标成"建议保留现金"，不要自己再去脑补这个乘法。
        print(f" {'股票':6s} {'实际建议仓位':>10s} {'预测收益率':>10s}   (权重x仓位系数)")
        total_effective = 0.0
        for ticker, weight, pred_ret, pos_size in picks:
            effective_pct = weight * pos_size
            total_effective += effective_pct
            print(f" {ticker:6s} {effective_pct*100:9.1f}% {pred_ret*100:9.2f}%   "
                  f"({weight*100:.1f}% x {pos_size:.2f})")
        cash_pct = max(0.0, 1.0 - total_effective)
        print("-" * 70)
        print(f" 合计投入仓位: {total_effective*100:.1f}%    建议保留现金: {cash_pct*100:.1f}%")
        print(" （“建议保留现金”部分是仓位系数打折出来的，不是必须持有的比例——")
        print(" 系数越低说明模型对这只票信心越弱，打折是为了别对低把握的票下重注）")
        print()
        print(" 【止损纪律，非常重要】每只票从买入价跌 2% 就止损卖出，2%比5%盈亏比")
        print(" 更好这个方向性结论仍然成立，但如实提醒：生存偏差修复后(README验证")
        print(" 结论第27条)，这条动量策略单独使用时全样本Sharpe只有约-0.05(95%置信")
        print(" 区间跨0)，选股本身没有统计上站得住脚的正向edge，收益主要靠止损/熔断")
        print(" 规则做下行保护撑着——止损纪律能让'不亏很多'更可靠，不能让这条策略")
        print(" 单独变成一个可信的正收益来源，仓位分配请参考三元组合(动量+价值+")
        print(" 低波动)+止损配置的完整体系，不要只依赖这一条腿的信号。")
    print("=" * 70)

    # 追加写入日志，供未来回顾对照实际表现
    file_exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["generated_at", "decision_date", "ticker", "weight", "pred_ret",
                             "pos_size", "effective_allocation_pct"])
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not picks:
            writer.writerow([now, target_date.strftime("%Y-%m-%d"), "", "", "", "", ""])
        else:
            for ticker, weight, pred_ret, pos_size in picks:
                writer.writerow([now, target_date.strftime("%Y-%m-%d"), ticker,
                                 f"{weight:.6f}", f"{pred_ret:.6f}", f"{pos_size:.4f}",
                                 f"{weight*pos_size*100:.2f}"])
    print(f"\n已追加写入日志: {LOG_PATH}")
    print("提醒：这只是模型信号，不是投资建议；\"实际建议仓位%\"是基于风险平价和模型置信度算出来的")
    print("相对比例，不代表你应该投入的绝对资金金额，下单前请自己判断风险承受能力和实际可用资金。")


if __name__ == "__main__":
    main()
