"""
最终整合系统——这次会话所有研究成果的正式交付模块：
股票敞口(三元组合：ML周频动量 + 长周期价值 + 低波动因子，见README"验证
结论"第55/56/57条) + 动态债券对冲层(按10年期国债收益率趋势调整IEF敞口，
见第65/66/67/68/69条)。

如实声明：这不是"发现了新alpha"的系统，是把这次会话验证过的几个真实
效应（适度的股票内部分散化 + 股债负相关性，经过2008年信用危机和
2000-2002科技股泡沫破灭两次不同类型危机的压力测试）组合成一个逻辑
自洽的资产配置框架。默认用IEF而不是TLT做债券敞口(第69条：中久期在
样本内测试中全面优于长久期)。

用法：
    from full_system import run_full_system
    report = run_full_system(price_df, membership, spy_close, mom_results,
                              bond_ticker="IEF")
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from combo_strategy import run_combo_strategy, true_max_drawdown

DEFAULT_BOND_WEIGHT = 0.4
DEFAULT_BOND_WEIGHT_HIKE = 0.1
DEFAULT_RATE_TREND_THRESHOLD = 0.3  # 60日10年期国债收益率变化(百分点)
DEFAULT_TREND_WINDOW = 60
DEFAULT_GOLD_SPLIT = 0.0  # 非股票敞口里分给黄金的比例，0=纯债券(向后兼容默认行为)
DEFAULT_DOLLAR_SPLIT = 0.0  # 非股票敞口里分给美元指数(UUP)的比例，0=不使用(向后兼容默认行为)
DEFAULT_HYG_SPLIT = 0.0  # 非股票敞口里分给高收益公司债(HYG)的比例，0=不使用(向后兼容默认行为)
DEFAULT_COVERED_CALL_OTM = 0.05  # 备兑看涨期权虚值幅度，见README第299/301条
COVERED_CALL_R_FREE = 0.04
COVERED_CALL_MONTH_DAYS = 21


def _bs_call_price(S, K, T, r, sigma):
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _apply_covered_call_overlay(equity_ret: pd.Series, vix_series: pd.Series, otm_pct: float) -> pd.Series:
    """每`COVERED_CALL_MONTH_DAYS`个交易日卖一次虚值看涨期权，权利金计入
    期初那一天的收益、payoff计入期末那一天的收益，中间的日收益不受影响
    (期权买入即持有到期，不做逐日盯市)。见README第299/301条：三段区间
    Sharpe/MaxDD/CAGR全部同步改善，2/3区间block bootstrap显著。"""
    equity_ret = equity_ret.copy()
    idx = equity_ret.index
    n = len(idx)
    vix_aligned = vix_series.reindex(idx).ffill() / 100.0
    equity_nav_local = (1 + equity_ret).cumprod()
    i = 0
    while i + COVERED_CALL_MONTH_DAYS < n:
        S0 = equity_nav_local.iloc[i]
        ST = equity_nav_local.iloc[i + COVERED_CALL_MONTH_DAYS]
        sigma = vix_aligned.iloc[i]
        if pd.isna(sigma) or sigma <= 0:
            sigma = 0.20
        K = S0 * (1 + otm_pct)
        T = COVERED_CALL_MONTH_DAYS / 252.0
        premium = _bs_call_price(S0, K, T, COVERED_CALL_R_FREE, sigma)
        payoff = max(ST - K, 0.0)
        entry_date = idx[i]
        exit_date = idx[i + COVERED_CALL_MONTH_DAYS]
        equity_ret.loc[entry_date] = equity_ret.loc[entry_date] + premium / S0
        equity_ret.loc[exit_date] = equity_ret.loc[exit_date] - payoff / S0
        i += COVERED_CALL_MONTH_DAYS
    return equity_ret
DEFAULT_USE_VOL_TARGETING = False  # 股票腿波动率目标化开关，默认关闭(向后兼容)
DEFAULT_VOL_TARGET = 0.15
DEFAULT_VOL_WINDOW = 20
VOL_TARGETING_EXPOSURE_MIN = 0.3
VOL_TARGETING_EXPOSURE_MAX = 1.0  # 上限固定100%，不加杠杆——见README第90/91条：
# 敞口上限设150%时测出来的"改善"其实是杠杆假象，只有把上限锁在100%(只降
# 不加)才是干净的、真正测出危机保护效果的版本，这个上限不开放给调用方修改。


def fetch_bond_and_rate_data(start, end, bond_ticker: str = "IEF", gold_ticker: str | None = None,
                              dollar_ticker: str | None = None, hyg_ticker: str | None = None):
    tickers = ([bond_ticker, "^TNX"] + ([gold_ticker] if gold_ticker else [])
               + ([dollar_ticker] if dollar_ticker else []) + ([hyg_ticker] if hyg_ticker else []))
    data = yf.download(tickers, start=start, end=end, progress=False,
                       group_by="ticker", auto_adjust=True)
    bond_close = data[bond_ticker]["Close"].dropna()
    tnx = data["^TNX"]["Close"].dropna()
    gold_close = data[gold_ticker]["Close"].dropna() if gold_ticker else None
    dollar_close = data[dollar_ticker]["Close"].dropna() if dollar_ticker else None
    hyg_close = data[hyg_ticker]["Close"].dropna() if hyg_ticker else None
    return bond_close, tnx, gold_close, dollar_close, hyg_close


def run_full_system(price_df: pd.DataFrame, membership: dict, spy_close: pd.Series,
                     mom_results: list[dict], bond_ticker: str = "IEF",
                     bond_weight: float = DEFAULT_BOND_WEIGHT,
                     bond_weight_hike: float = DEFAULT_BOND_WEIGHT_HIKE,
                     rate_trend_threshold: float = DEFAULT_RATE_TREND_THRESHOLD,
                     gold_ticker: str | None = "GLD",
                     gold_split: float = DEFAULT_GOLD_SPLIT,
                     dollar_ticker: str | None = "UUP",
                     dollar_split: float = DEFAULT_DOLLAR_SPLIT,
                     hyg_ticker: str | None = "HYG",
                     hyg_split: float = DEFAULT_HYG_SPLIT,
                     use_vol_targeting: bool = DEFAULT_USE_VOL_TARGETING,
                     vol_target: float = DEFAULT_VOL_TARGET,
                     vol_window: int = DEFAULT_VOL_WINDOW,
                     value_rebalance_days: int | None = None,
                     value_weighting: str = "equal",
                     value_stop_loss_pct: float | None = None,
                     low_vol_stop_loss_pct: float | None = None,
                     value_hrp_max_weight: float | None = None,
                     use_covered_call: bool = False,
                     covered_call_otm_pct: float = DEFAULT_COVERED_CALL_OTM,
                     vix_series: pd.Series | None = None,
                     value_factor_set: str = "pe_pb_roe",
                     leg_weights: dict | None = None,
                     cost_bi: float = 0.003):
    """跑完整系统(三元股票组合 + 动态债券对冲层 + 可选的黄金分散化层 +
    可选的股票腿波动率目标化)，返回逐日净值 + 汇总指标。

    value_rebalance_days(默认None，向后兼容)：透传给`run_combo_strategy`，
    见README第102条——价值腿改周频(5天)调仓，三段区间Sharpe/MaxDD全部
    改善，是目前证据最充分的具体改进，建议实际使用时显式传`=5`。

    gold_split(默认0，向后兼容纯债券行为)：非股票敞口(bond_weight/
    bond_weight_hike算出来的那部分)里分给黄金的比例，比如bond_weight=0.4、
    gold_split=0.5时，正常状态下股票60% + 债券20% + 黄金20%。依据见
    README"验证结论"第84条：黄金跟SPY相关系数四段历史(含2008年真正危机)
    都在-0.03~0.15之间，从未像股票内部分散化那样在危机期间飙升，且在
    2022-2026加息周期(债券分散化失效的那段)反而是黄金表现最好的区间，
    机制上黄金/债券对抗的是不同类型的宏观冲击(通胀 vs 衰退)，是互补
    而非重叠的分散化来源。

    use_vol_targeting(默认False，向后兼容)：对股票腿按trailing已实现
    波动率动态降低敞口(只降不加，上限锁死100%)，依据见README第90/91/97条
    (股票腿单独测)和第120条(完整系统层面，PIT修正后数据，三段区间Sharpe
    全部同向改善或持平)——敞口允许加杠杆时测出来的"改善"是杠杆假象，锁定
    100%上限后才是干净的结果，默认关闭，作为可选风控层提供。

    dollar_split(默认0，向后兼容)：非股票敞口里分给美元指数(UUP)的比例，
    见README"验证结论"第206条——美元指数三段区间(2009-2014/2014-2020/
    2022-2026)Sharpe和MaxDD全部同步改善，是继黄金(第84/85条)、债券
    (第65/66条)之后第三个干净通过"三段全部更好"标准的分散化候选，
    UUP-GLD相关系数常年在-0.35~-0.44(明显负相关)、UUP-IEF相关系数
    -0.38~0.11不等，是真正独立的第三条分散化来源，理论上可以跟gold_split
    同时启用。gold_split/dollar_split两者之和不应超过1(超过时bond会
    分到负权重，函数不做防御性检查，调用方需自行保证)。

    value_stop_loss_pct(默认None，向后兼容)：透传给`run_combo_strategy`
    的价值腿个股止损阈值，见README第230-232条——0.10三段区间Sharpe/MaxDD
    全部同步改善，且部分区间收益层面统计显著，建议实际使用时显式传`=0.10`。

    value_factor_set(默认"pe_pb_roe"，向后兼容)：透传给`select_value_
    portfolio`，"pe_ps"时改用P/E+P/S二因子排名代替原有P/E-P/B-ROE三因子，
    见README第326/327条——三段区间Sharpe全面改善、2/3区间通过跟SPY的
    显著性检验，是这次会话基本面方向证据最强的结果，但同时标注了多重
    检验风险，采纳前建议先在完整系统层面重新验证一次（不能直接假设价值腿
    单独测出的效果，跟三元组合+止损+权重方案+动态债券对冲这一整套现有
    机制叠加后效果不变，第294条已经示范过"腿层面的真实改进在完整系统层面
    被多层稀释到统计上看不出来"这种情况确实会发生）。第331条完整系统层面
    验证过`pe_ps`确实略优于`pe_pb_roe`(Sharpe 1.849 vs 1.827)，但第349条
    子窗口分割检验发现这个微弱优势(+1.2%)对区间切分方式敏感——2022-2026
    拆成两个子窗口分别看，两次都是`pe_pb_roe`更高，只有拼接成完整一段时
    `pe_ps`才更高(Sharpe比率不能线性拼接，不是bug，但说明这个优势不够稳健)。
    **不建议在没有更多证据之前把`value_factor_set`默认值改成`"pe_ps"`**，
    当前默认`"pe_pb_roe"`保持不变是经过深思熟虑的选择，不是尚未来得及更新。

    leg_weights(默认None，向后兼容等权)：透传给`run_combo_strategy`，
    当前推荐值`{"value":0.50,"low_vol":0.40,"momentum":0.10}`(第377条)。
    【第399条重要提醒】不要用CVaR最小化(第390条)、直接Sharpe最大化
    (第395-398条)或正则化版本去"改进"这个权重——这些优化范式在这次
    会话缓存的3段历史区间上全部会收敛到重仓/单压价值腿，是Markowitz类
    优化对样本估计误差的经典过拟合陷阱，不是真发现，这条优化线索已经
    充分探索完毕(第365→399条)。

    cost_bi(默认0.003，向后兼容)：透传给`run_combo_strategy`的单边换手
    成本假设，之前这里没有暴露这个参数，`run_full_system`层面没法做
    成本敏感性测试(第389条Cycle85发现)，这次补上。"""
    equity_report = run_combo_strategy(price_df, membership, spy_close, mom_results,
                                       value_rebalance_days=value_rebalance_days,
                                       value_weighting=value_weighting,
                                       value_stop_loss_pct=value_stop_loss_pct,
                                       low_vol_stop_loss_pct=low_vol_stop_loss_pct,
                                       value_hrp_max_weight=value_hrp_max_weight,
                                       value_factor_set=value_factor_set,
                                       leg_weights=leg_weights,
                                       cost_bi=cost_bi)
    equity_nav = equity_report["combo_nav"]

    start = equity_nav.index[0].strftime("%Y-%m-%d")
    end = equity_nav.index[-1].strftime("%Y-%m-%d")
    bond_close, tnx, gold_close, dollar_close, hyg_close = fetch_bond_and_rate_data(
        start, end, bond_ticker, gold_ticker if gold_split > 0 else None,
        dollar_ticker if dollar_split > 0 else None, hyg_ticker if hyg_split > 0 else None)

    tnx_trend = tnx.reindex(equity_nav.index).ffill().diff(DEFAULT_TREND_WINDOW)
    non_equity_w = pd.Series(bond_weight, index=equity_nav.index)
    non_equity_w[tnx_trend > rate_trend_threshold] = bond_weight_hike

    bond_ret = bond_close.reindex(equity_nav.index).ffill().pct_change()
    equity_ret = equity_nav.pct_change()

    if use_vol_targeting:
        trailing_vol = equity_ret.rolling(vol_window).std() * np.sqrt(252)
        exposure = (vol_target / trailing_vol).clip(VOL_TARGETING_EXPOSURE_MIN, VOL_TARGETING_EXPOSURE_MAX)
        exposure = exposure.shift(1)  # 用T-1为止的波动率决定T日敞口，不用当天信息
        equity_ret = (exposure * equity_ret).dropna()  # 降下来的敞口视为持有零息现金，不建模计息

    if use_covered_call:
        if vix_series is None:
            raise ValueError("use_covered_call=True 时必须传入 vix_series")
        equity_ret = _apply_covered_call_overlay(equity_ret, vix_series, covered_call_otm_pct)

    bond_frac = 1.0 - gold_split - dollar_split - hyg_split
    non_equity_ret = bond_frac * bond_ret
    if gold_split > 0 and gold_close is not None:
        gold_ret = gold_close.reindex(equity_nav.index).ffill().pct_change()
        non_equity_ret = non_equity_ret + gold_split * gold_ret
    if dollar_split > 0 and dollar_close is not None:
        dollar_ret = dollar_close.reindex(equity_nav.index).ffill().pct_change()
        non_equity_ret = non_equity_ret + dollar_split * dollar_ret
    if hyg_split > 0 and hyg_close is not None:
        hyg_ret = hyg_close.reindex(equity_nav.index).ffill().pct_change()
        non_equity_ret = non_equity_ret + hyg_split * hyg_ret
    full_ret = ((1 - non_equity_w) * equity_ret + non_equity_w * non_equity_ret).dropna()
    full_nav = (1 + full_ret).cumprod()

    n_years = len(full_nav) / 252
    cagr = float(full_nav.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else None
    sharpe = float(full_ret.mean() / (full_ret.std() + 1e-9) * np.sqrt(252))
    mdd = true_max_drawdown(full_nav.values)

    return {
        "full_nav": full_nav,
        "equity_only_report": equity_report,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDD": mdd,
        "bond_ticker": bond_ticker,
        "gold_split": gold_split,
        "use_vol_targeting": use_vol_targeting,
        "n_days": len(full_nav),
    }


def print_full_report(report: dict, label: str = ""):
    print(f"\n{'='*60}\n 完整系统报告(三元股票组合 + 动态{report['bond_ticker']}对冲) {label}\n{'='*60}")
    print(f" 交易日数: {report['n_days']}")
    print(f" CAGR: {report['CAGR']*100:.2f}%" if report["CAGR"] is not None else " CAGR: N/A")
    print(f" Sharpe(日频年化): {report['Sharpe']:.3f}")
    print(f" 真实最大回撤: {report['MaxDD']*100:.2f}%")
    eq = report["equity_only_report"]
    print(f" (对照：纯股票三元组合 CAGR={eq['CAGR']*100:.2f}%, MaxDD={eq['MaxDD']*100:.2f}%)")
    print("=" * 60)
    print(" 如实提醒：这不是「发现了新alpha」的系统，是把已验证的股票内部适度")
    print(" 分散化(README第55-57条) + 股债负相关性(第65-69条，经过两次不同类型")
    print(" 历史危机压力测试) 组合成的资产配置框架，不保证未来表现。")
