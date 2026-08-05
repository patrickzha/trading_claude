"""
三元组合策略(ML周频动量 + 长周期价值 + 低波动因子)的统一实现——把这次
session零散的分析脚本(run_three_way_combo_test.py / run_combo_true_drawdown.py
等)整理成一个干净、可复用的模块，是目前测试过的所有方向里统计证据最强
(见README"验证结论"第55/56/57条)、真实风险画像也自洽的配置。

如实说明这不是"发现了新edge"的宣告，是"目前最站得住脚、经过三段独立
历史区间交叉验证的候选"——bootstrap检验3段里2-3段显著，Fama-French
alpha检验3段里2段显著，逐日真实回撤在2/3区间比任何单腿都浅。

用法：
    from combo_strategy import run_combo_strategy
    report = run_combo_strategy(price_df, sector_map, membership, spy_close,
                                 mom_results=<已跑好的动量策略周频结果>)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import get_weekly_trading_dates
from value_investing import backtest_value_strategy
from low_vol_investing import backtest_low_vol_strategy


def true_max_drawdown(nav_series) -> float:
    nav = np.array(nav_series)
    peak = np.maximum.accumulate(nav)
    return float(np.min((nav - peak) / peak))


def run_combo_strategy(price_df: pd.DataFrame, membership: dict, spy_close: pd.Series | None,
                        mom_results: list[dict] | None, value_top_n: int = 30, low_vol_top_n: int = 30,
                        rebalance_days: int = 21, include_momentum: bool = True,
                        value_rebalance_days: int | None = None,
                        value_weighting: str = "equal",
                        value_stop_loss_pct: float | None = None,
                        low_vol_stop_loss_pct: float | None = None,
                        cost_bi: float = 0.003,
                        value_hrp_max_weight: float | None = None,
                        value_factor_set: str = "pe_pb_roe",
                        leg_weights: dict | None = None):
    """跑组合(等权)，返回逐日净值序列 + 汇总指标。mom_results是
    run_walk_forward_backtest()已经跑好的结果(list of dict，含date/
    weekly_return/net_worth)——动量这条腿不在这里重新训练模型，直接
    复用已有回测结果，避免每次调用都要跑一次~30分钟的XGBoost训练。

    include_momentum=False时跳过动量腿，只用价值+低波动二元组合(见README
    "验证结论"第75条：动量腿需要~30分钟重新训练、边际贡献在3段区间里
    只有1段明显为正、1段基本无贡献、1段轻微负贡献，追求更快更便宜的
    系统时，二元组合是一个诚实的轻量化选项，不需要mom_results参数)。

    value_rebalance_days(默认None，等于rebalance_days，向后兼容)：价值腿
    单独覆盖调仓频率。见README第102条：价值腿改成周频(5天)调仓在三段区间
    Sharpe全部提升，2014-2020这段最大回撤从-43.18%压缩到-26.74%(此前
    "验证结论"标注过这段价值策略尾部风险比SPY自己的新冠暴跌还深，周频
    调仓明显缓解了这个问题)；低波动腿同样测过但方向不一致，所以只给
    价值腿开放这个覆盖参数，低波动腿仍然用主`rebalance_days`。

    value_weighting(默认"equal"，向后兼容)：价值腿仓位分配方式，"min_variance"
    时用trailing 60日协方差矩阵算最小方差权重代替等权，见README第140条——
    价值腿选股逻辑不变，三段区间Sharpe和最大回撤全部同步改善，是这次会话
    第一个真正框架级(仓位分配数学方法，不是新信号)的正面发现；"risk_parity"
    时用等风险贡献权重(第155条)，2/3区间Sharpe改善，比min_variance稍弱但
    更稳健；"hrp"时用层次风险平价(第159条)，三段区间Sharpe和MaxDD全部
    同步改善、回撤改善幅度三者里最大，目前证据最扎实。同样的框架在低
    波动腿上测过(第141/156条)只有risk_parity部分成立，所以只给价值腿
    开放这个参数。

    【第425/426条重要提醒】三条腿之间的协方差结构不是跨时间恒定的——
    2009-2014段(次贷危机后复苏初期，市场整体个股相关性结构性偏高)三腿
    两两相关系数明显高于2014-2020/2022-2026段，这是当时市场整体相关性
    regime的体现(第426条已用标普500个股两两相关系数独立验证)，不是这
    三条腿特有的现象。任何依赖历史协方差稳定性的权重方案(min_variance/
    risk_parity/hrp，包括这个参数本身)都应该预期存在样本外漂移风险，
    不是一次算出来就能长期不变的静态解。

    value_stop_loss_pct(默认None，向后兼容)：价值腿个股止损阈值，见
    README第230-232条——10%阈值三段区间Sharpe和MaxDD全部同步改善，
    且部分区间(2022-2026)在收益层面测出真正的统计显著性(不只是风险
    管理效应)，是这次会话组合改进类发现里证据最扎实的少数案例之一。

    low_vol_stop_loss_pct(默认None，向后兼容)：低波动腿个股止损阈值，
    见README第234条——把价值腿止损推广到低波动腿，三段区间同样全部
    同步改善(2014-2020段MaxDD改善超过18个百分点)，止损是这次会话唯一
    同时在两条腿都验证过"三段全部改善"的风控机制。

    leg_weights(默认None，向后兼容等权`{momentum:1/3, value:1/3,
    low_vol:1/3}`)：三条腿的自定义权重字典，见README第365条——降低
    动量腿权重(比如`{"value":0.45,"low_vol":0.45,"momentum":0.10}`)
    在三段区间腿层面测出Sharpe一致改善、2/3区间block bootstrap显著，
    但2014-2020段MaxDD明显恶化(-6.93%→-8.45%)，机制上可能是动量腿
    虽然自身收益edge接近0(第307条)，但对组合仍有一定分散化/风险管理
    价值，减权会削弱这部分。完整系统层面(叠加止损/债券对冲)的验证
    结果见后续条目，采纳前请务必参考完整系统层面的数字，不要只看
    这里腿层面的Sharpe改善就直接采纳。

    当前推荐值`{"value":0.50,"low_vol":0.40,"momentum":0.10}`(第377条)。
    【第399条重要提醒】不要用CVaR最小化(第390条)、直接Sharpe最大化
    (第395-398条)或任何正则化版本的优化目标函数去"改进"这个权重——
    这三种优化范式在这次会话缓存的3段历史区间上全部会收敛到重仓/
    单压价值腿(极端情况是100%)，看起来"三段区间全部支持"，但这只是
    在利用价值腿这17年历史上持续领先的同一个模式，不是独立证据，本质
    是Markowitz类优化对样本估计误差的经典过拟合陷阱(第398条)。这条
    leg_weights优化线索(第365→372/373/377/379/383/390/395-399条)已经
    充分探索完毕，不需要再尝试新的优化目标函数去寻找"更优"权重。"""
    val_top_n_local, lv_top_n_local = value_top_n, low_vol_top_n
    val_rebalance = value_rebalance_days if value_rebalance_days is not None else rebalance_days
    legs = {}

    if include_momentum:
        if mom_results is None:
            raise ValueError("include_momentum=True 时必须提供 mom_results")
        mom_df = pd.DataFrame(mom_results)[["date", "net_worth"]]
        mom_df["date"] = pd.to_datetime(mom_df["date"])
        mom_series = mom_df.set_index("date")["net_worth"]
        mom_daily = mom_series.reindex(price_df.index, method="ffill").dropna()
        legs["momentum"] = mom_daily / mom_daily.iloc[0]

    _, val_nav = backtest_value_strategy(price_df, membership, spy_close,
                                         rebalance_days=val_rebalance, top_n=val_top_n_local,
                                         weighting=value_weighting, stop_loss_pct=value_stop_loss_pct,
                                         cost_bi=cost_bi, hrp_max_weight=value_hrp_max_weight,
                                         factor_set=value_factor_set)
    legs["value"] = pd.DataFrame(val_nav, columns=["date", "nav"]).set_index("date")["nav"]

    _, lv_nav = backtest_low_vol_strategy(price_df, membership, spy_close,
                                          rebalance_days=rebalance_days, top_n=lv_top_n_local,
                                          stop_loss_pct=low_vol_stop_loss_pct, cost_bi=cost_bi)
    legs["low_vol"] = pd.DataFrame(lv_nav, columns=["date", "nav"]).set_index("date")["nav"]

    combined = pd.DataFrame(legs).dropna()
    combined_norm = combined / combined.iloc[0]
    if leg_weights is None:
        combo_nav = combined_norm.mean(axis=1)
    else:
        w = pd.Series({leg: leg_weights.get(leg, 0.0) for leg in combined_norm.columns})
        if abs(w.sum() - 1.0) > 1e-6:
            raise ValueError(f"leg_weights之和必须为1.0，实际是{w.sum():.4f}(见README第373条：权重和不为1会静默产生错误的组合净值，不是明显的报错，容易被忽略)")
        combo_nav = (combined_norm * w).sum(axis=1)

    n_years = len(combo_nav) / 252
    cagr = float(combo_nav.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else None
    daily_rets = combo_nav.pct_change().dropna()
    sharpe = float(daily_rets.mean() / (daily_rets.std() + 1e-9) * np.sqrt(252))
    mdd = true_max_drawdown(combo_nav.values)

    spy_cagr = None
    if spy_close is not None:
        spy_aligned = spy_close.reindex(combo_nav.index).ffill().dropna()
        if len(spy_aligned) > 1:
            spy_cagr = float((spy_aligned.iloc[-1] / spy_aligned.iloc[0]) ** (1 / n_years) - 1)

    leg_mdd = {leg: true_max_drawdown(combined_norm[leg].values) for leg in combined_norm.columns}

    report = {
        "combo_nav": combo_nav,
        "leg_nav": combined_norm,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDD": mdd,
        "SPY_CAGR": spy_cagr,
        "leg_MaxDD": leg_mdd,
        "n_days": len(combo_nav),
    }
    return report


def print_report(report: dict, label: str = ""):
    print(f"\n{'='*60}\n 三元组合策略报告 {label}\n{'='*60}")
    print(f" 交易日数: {report['n_days']}")
    print(f" CAGR: {report['CAGR']*100:.2f}%" if report['CAGR'] is not None else " CAGR: N/A")
    print(f" Sharpe(日频年化): {report['Sharpe']:.3f}")
    print(f" 真实最大回撤: {report['MaxDD']*100:.2f}%")
    if report["SPY_CAGR"] is not None:
        print(f" 同期SPY CAGR: {report['SPY_CAGR']*100:.2f}% (超额: {(report['CAGR']-report['SPY_CAGR'])*100:+.2f}%)")
    print(" 各单腿最大回撤(对照):")
    for leg, mdd in report["leg_MaxDD"].items():
        print(f"   {leg}: {mdd*100:.2f}%")
    print("=" * 60)
    print(" 如实提醒：这是三段独立历史区间交叉验证过的最佳配置(README验证结论")
    print(" 第55/56/57条)，不是「确认存在新alpha」——bootstrap检验多数区间显著，")
    print(" 但Fama-French因子回归只有2/3区间的alpha显著，请勿过度解读为已证实的edge。")
