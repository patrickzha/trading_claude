"""
优化版回测驱动 v4
=================
1. 周一 Open 买入，周五 Close 卖出
2. 双边扣 0.3% 成本
3. 周内止损模拟（单周跌幅 >2% 按止损价结算——5年回测验证过比5%更好，见 us_stock_screener.py 的 STOP_LOSS_PCT）
4. 空仓周现金收益年化 4%
5. 支持 SHAP 输出
"""
from __future__ import annotations

import os
# 必须在 import numpy/pandas/xgboost 之前设置：BLAS/OpenMP 库的线程数在库
# 第一次被用到时就固定下来，不受 xgb_n_jobs=1 这个 Python 层参数的约束——
# 8个worker进程各自内部又开满8核的BLAS线程池，会在直方图构建等计算上产生
# 浮点求和顺序竞争，导致同一份数据+同样的random_state，并行跑两次结果都
# 不一样（诊断过：n_workers=1时两次跑bit-for-bit相同，n_workers=8时两次
# 跑同一个27周切片的累计收益从0.178漂到0.225，选出的候选都会变）。这是
# 本session所有并行(n_workers>1)全量验证结果都可能携带的一个未知量级的
# 执行噪声来源，不是市场规律或代码bug造成的差异。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from us_stock_screener import (
    get_ai_weekly_picks, SECTOR_MAP, COST_BI, CASH_YIELD, STOP_LOSS_PCT, MAX_POS, MAX_SECTOR,
    ROLLING_WINDOW, RSI_OVERBOUGHT_THRESHOLD, USE_RECENCY_WEIGHTING, RECENCY_HALFLIFE_WEEKS,
)

# ProcessPoolExecutor 的 worker 进程用来存放只读的大 DataFrame + 配置 +
# 该 worker 自己的 market_cache（通过 initializer 设置一次，跨这个 worker
# 处理的所有周复用，不用每个任务都重新 pickle 一遍 DataFrame）。
_WORKER_CTX: dict = {}


def _init_worker(close_df, open_df, high_df, low_df, volume_df, vix_series, config):
    _WORKER_CTX["close_df"] = close_df
    _WORKER_CTX["open_df"] = open_df
    _WORKER_CTX["high_df"] = high_df
    _WORKER_CTX["low_df"] = low_df
    _WORKER_CTX["volume_df"] = volume_df
    _WORKER_CTX["vix_series"] = vix_series
    _WORKER_CTX["config"] = config
    _WORKER_CTX["market_cache"] = {}


def fetch_inverse_etf_data(start, end, index: pd.DatetimeIndex) -> pd.Series | None:
    """
    拉反向S&P500 ETF（SH，做空大盘1倍）的收盘价，供 use_inverse_hedge 开关用——
    现在触发宏观熔断就完全空仓，测试一下"留一小部分仓位做反向对冲"代替纯空仓
    会不会更好。reindex+ffill 到传入的交易日索引上。
    """
    import yfinance as yf
    try:
        df = yf.download("SH", start=start, end=end, progress=False, auto_adjust=True)
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close.reindex(index).ffill()
    except Exception as e:
        print(f"  [警告] 反向ETF(SH)数据拉取失败：{type(e).__name__}: {e}")
        return None


def _inverse_etf_weekly_return(inverse_etf_close, monday, friday, cost_bi) -> float:
    try:
        buy = inverse_etf_close.loc[monday]
        sell = inverse_etf_close.loc[friday]
        if pd.isna(buy) or pd.isna(sell) or buy <= 0:
            return 0.0
        return (sell - buy) / buy - cost_bi
    except Exception:
        return 0.0


def _atr_based_stop_pct(ticker, decision_date, close_df, high_df, low_df,
                        multiplier: float = 1.5, min_pct: float = 0.01, max_pct: float = 0.05) -> float:
    """
    用 ATR(14) 算每只票自己的止损幅度，代替所有票一刀切的固定百分比——波动大
    的票给更宽的止损空间，波动小的给更窄。不改 get_ai_weekly_picks 的返回接口
    （picks 元组形状不变），直接在这里用已有的价格数据现算，独立、不影响其他
    调用方（generate_weekly_signal.py 等）。clip 到 [min_pct, max_pct] 避免
    极端波动票算出离谱的止损幅度。
    """
    try:
        c = close_df[ticker].loc[:decision_date].tail(20)
        h = high_df[ticker].loc[:decision_date].tail(15)
        l = low_df[ticker].loc[:decision_date].tail(15)
        if len(c) < 15 or len(h) < 15 or len(l) < 15:
            return None
        prev_c = c.iloc[-15:-1].values
        tr1 = h.iloc[-14:].values - l.iloc[-14:].values
        tr2 = np.abs(h.iloc[-14:].values - prev_c)
        tr3 = np.abs(l.iloc[-14:].values - prev_c)
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr = np.mean(tr) if len(tr) > 0 else 0.0
        curr_price = c.iloc[-1]
        if curr_price <= 0 or atr <= 0:
            return None
        stop_pct = (atr / curr_price) * multiplier
        return float(np.clip(stop_pct, min_pct, max_pct))
    except Exception:
        return None


def _compute_week_task(i, monday, friday, decision_date, do_shap, shap_path):
    """单周的选股 + 逐票结算，只依赖截止 decision_date 的历史数据，不依赖
    其他周的回测结果，可以放心跨周并行。账户级熔断（依赖前几周的实际收益）
    不在这里算，等所有周都跑完、按时间顺序收集齐了之后再串行算一遍。"""
    try:
        ctx = _WORKER_CTX
        cfg = ctx["config"]

        # 生存偏差修复：point_in_time_membership 提供时，只用"这一周真实在
        # 股票池里"的票参与训练和选股——不提前用还没被纳入的票，不用已经
        # 被剔除的票。只影响传进 get_ai_weekly_picks 的候选范围，不影响下面
        # simulate_weekly_return 用的价格数据（那个只按ticker名字查价格，
        # 跟"是不是当期成分股"无关）。
        pit_close_df, pit_open_df, pit_high_df, pit_low_df, pit_volume_df = (
            ctx["close_df"], ctx["open_df"], ctx["high_df"], ctx["low_df"], ctx["volume_df"]
        )
        if cfg.get("point_in_time_membership") is not None:
            valid_tickers = cfg["point_in_time_membership"].get(decision_date.strftime("%Y-%m-%d"))
            if valid_tickers is not None:
                valid_tickers = [t for t in valid_tickers if t in pit_close_df.columns]
                pit_close_df = pit_close_df[valid_tickers]
                pit_open_df = pit_open_df[valid_tickers]
                pit_high_df = pit_high_df[valid_tickers]
                pit_low_df = pit_low_df[valid_tickers]
                pit_volume_df = pit_volume_df[valid_tickers]

        picks = get_ai_weekly_picks(
            pit_close_df, pit_open_df, pit_high_df, pit_low_df,
            pit_volume_df, ctx["vix_series"], decision_date,
            shap_enabled=do_shap, shap_output_path=shap_path,
            market_cache=ctx["market_cache"],
            enable_macro_filter=cfg["enable_macro_filter"], regime_mode=cfg["regime_mode"],
            model_objective=cfg["model_objective"],
            use_conformal_filter=cfg["use_conformal_filter"], conformal_coverage=cfg["conformal_coverage"],
            use_meta_label=cfg["use_meta_label"], meta_label_threshold=cfg["meta_label_threshold"],
            earnings_calendar=cfg["earnings_calendar"], enable_earnings_blackout=cfg["enable_earnings_blackout"],
            enable_vix_leg=cfg["enable_vix_leg"], enable_momentum_leg=cfg["enable_momentum_leg"],
            xgb_n_jobs=cfg["xgb_n_jobs"],
            xgb_max_depth=cfg["xgb_max_depth"], xgb_min_child_weight=cfg["xgb_min_child_weight"],
            xgb_reg_alpha=cfg["xgb_reg_alpha"], xgb_reg_lambda=cfg["xgb_reg_lambda"],
            xgb_gamma=cfg["xgb_gamma"], xgb_ensemble_size=cfg["xgb_ensemble_size"],
            max_pos=cfg["max_pos"], max_sector=cfg["max_sector"],
            pos_size_floor=cfg["pos_size_floor"], pos_size_ceiling=cfg["pos_size_ceiling"],
            use_fundamental_features=cfg["use_fundamental_features"],
            rsi_overbought_threshold=cfg["rsi_overbought_threshold"],
            use_multi_horizon_momentum=cfg["use_multi_horizon_momentum"],
            use_sector_relative_momentum=cfg["use_sector_relative_momentum"],
            use_volume_zscore=cfg["use_volume_zscore"],
            use_recency_weighting=cfg["use_recency_weighting"],
            recency_halflife_weeks=cfg["recency_halflife_weeks"],
            rolling_window_days=cfg["rolling_window_days"],
            use_dynamic_position_count=cfg["use_dynamic_position_count"],
            use_vol_adjusted_target=cfg["use_vol_adjusted_target"],
            use_triple_barrier_labels=cfg["use_triple_barrier_labels"],
            label_stop_loss_pct=cfg["label_stop_loss_pct"],
            macro_df=cfg["macro_df"], use_macro_features=cfg["use_macro_features"],
            cost_bi=cfg["cost_bi"],
            use_correlation_dedup=cfg["use_correlation_dedup"], correlation_threshold=cfg["correlation_threshold"],
            use_calendar_features=cfg["use_calendar_features"],
            use_pca_features=cfg["use_pca_features"], pca_variance_threshold=cfg["pca_variance_threshold"],
            sector_map=cfg["sector_map"],
            use_full_cross_sectional_standardization=cfg["use_full_cross_sectional_standardization"],
            use_prediction_calibration=cfg["use_prediction_calibration"],
            model_family=cfg["model_family"],
            use_stacking=cfg["use_stacking"],
            use_shap_pruning=cfg["use_shap_pruning"],
            shap_pruned_features=cfg["shap_pruned_features"],
            use_multi_window_fusion=cfg["use_multi_window_fusion"],
            fusion_window_days=cfg["fusion_window_days"],
            use_ic_quality_gate=cfg["use_ic_quality_gate"],
            ic_quality_threshold=cfg["ic_quality_threshold"],
            xgb_rank_objective=cfg["xgb_rank_objective"],
            use_stage1_filter=cfg["use_stage1_filter"],
            stage1_ma_window=cfg["stage1_ma_window"],
            classification_top_frac=cfg["classification_top_frac"],
        )
        pick_results = []
        for ticker, weight, pred_ret, pos_size in picks:
            effective_stop_pct = cfg["stop_loss_pct"]
            if cfg["use_atr_stop"]:
                atr_stop = _atr_based_stop_pct(
                    ticker, decision_date, ctx["close_df"], ctx["high_df"], ctx["low_df"],
                    multiplier=cfg["atr_stop_multiplier"], min_pct=cfg["atr_stop_min"], max_pct=cfg["atr_stop_max"],
                )
                if atr_stop is not None:
                    effective_stop_pct = atr_stop
            exit_reason, net_ret = simulate_weekly_return(
                ticker, monday, friday, ctx["close_df"], ctx["open_df"], ctx["high_df"], ctx["low_df"],
                stop_loss_pct=effective_stop_pct, take_profit_pct=cfg["take_profit_pct"],
                cost_bi=cfg["cost_bi"],
                trailing_stop_pct=cfg["trailing_stop_pct"], partial_take_profit=cfg["partial_take_profit"],
                staggered_entry_days=cfg["staggered_entry_days"],
            )
            sector = (cfg["sector_map"] or SECTOR_MAP).get(ticker, "TECH")
            pick_results.append((ticker, weight, pred_ret, pos_size, net_ret, exit_reason, sector))
        return i, monday, pick_results
    except Exception as e:
        print(f"  [警告] 第{i}周({monday.date()})计算失败，按空仓处理：{type(e).__name__}: {e}")
        return i, monday, []


def get_weekly_trading_dates(index: pd.DatetimeIndex):
    weeks = index.to_period("W-FRI")
    pairs = []
    for period in sorted(set(weeks)):
        days = index[weeks == period]
        if len(days) >= 2:
            pairs.append((days[0], days[-1]))
    return pairs


def simulate_weekly_return(ticker, monday, friday, close_df, open_df, high_df, low_df,
                           stop_loss_pct: float = STOP_LOSS_PCT,
                           take_profit_pct: float | None = None,
                           cost_bi: float = COST_BI,
                           trailing_stop_pct: float | None = None,
                           partial_take_profit: bool = False,
                           staggered_entry_days: int | None = None):
    """
    模拟一次周内交易结算，返回 (exit_reason, net_ret)，exit_reason 是
    "stop_loss" / "take_profit" / "partial_take_profit" / "none"（持有到周五收盘）。

    take_profit_pct=None（默认）时完全是原来的行为，只检查止损，不检查
    止盈，跟之前返回布尔值时的止损判定逻辑逐日等价（老版本用 .min() 一次
    性判断"整周最低点是否跌破止损价"，这里改成逐日检查、遇到就立刻返回，
    数学上是同一个判定，只是同时支持了要按时间顺序跟止盈比较谁先触发）。

    同一天内止损价和止盈价都被触发时，保守假设止损先发生——日线数据不知道
    盘中真实的先后顺序，不能假设对自己有利的那个先触发。

    trailing_stop_pct 不为 None 时，止损价不再固定在买入价，而是跟着"买入
    到当前为止的最高价"走——用前一天（不含当天）为止的最高价算当天的移动
    止损线，再检查当天最低价是否跌破，避免用当天的高点反过来判定当天触发
    （日线数据不知道盘中高点/低点谁先发生，这样处理更保守）。

    partial_take_profit=True 时，止盈触发不再全平，只平一半仓位（按止盈价
    结算，成本减半），剩下一半简化处理为直接持有到周五收盘（不再检查后续
    止损/止盈），两腿按50/50加权。

    staggered_entry_days(默认None，向后兼容单日买入)：不为None时，买入价
    换成"前N个交易日开盘价的均价"（比如=3表示周一/周二/周三各买1/3），
    止损/止盈检查从第N天之后才开始（分批还没买完之前不设止损，避免"还没
    建好仓就被打止损"）。见README第26条留下的悬念——只在诊断窗口测过，
    这次用来做正式的三窗口验证。
    """
    try:
        if staggered_entry_days is not None and staggered_entry_days > 1:
            week_days = close_df.loc[monday:friday, ticker].index
            if len(week_days) < staggered_entry_days + 1:
                return "none", 0.0
            tranche_days = week_days[:staggered_entry_days]
            tranche_prices = open_df.loc[tranche_days, ticker]
            if tranche_prices.isna().any() or (tranche_prices <= 0).any():
                return "none", 0.0
            buy_price = float(tranche_prices.mean())
            entry_offset = staggered_entry_days
        else:
            buy_price = open_df.loc[monday, ticker]
            if pd.isna(buy_price) or buy_price <= 0:
                return "none", 0.0
            entry_offset = 1

        stop_price = buy_price * (1 - stop_loss_pct)
        take_profit_price = buy_price * (1 + take_profit_pct) if take_profit_pct else None
        running_peak = buy_price

        if ticker in low_df.columns:
            week_days_after_open = low_df.loc[monday:friday, ticker].iloc[entry_offset:].index
            has_high_col = ticker in high_df.columns
            for day in week_days_after_open:
                day_low = low_df.loc[day, ticker] if day in low_df.index else np.nan
                day_high = high_df.loc[day, ticker] if (has_high_col and day in high_df.index) else np.nan

                effective_stop = stop_price
                if trailing_stop_pct is not None:
                    effective_stop = running_peak * (1 - trailing_stop_pct)

                if pd.notna(day_low) and day_low <= effective_stop:
                    stop_ret = (effective_stop - buy_price) / buy_price - cost_bi * 0.5
                    return "stop_loss", stop_ret

                if take_profit_price is not None and pd.notna(day_high) and day_high >= take_profit_price:
                    if partial_take_profit:
                        tp_leg = (take_profit_price - buy_price) / buy_price - cost_bi * 0.5
                        hold_sell = close_df.loc[friday, ticker]
                        if pd.isna(hold_sell) or hold_sell <= 0:
                            hold_leg = tp_leg
                        else:
                            hold_leg = (hold_sell - buy_price) / buy_price - cost_bi * 0.5
                        return "partial_take_profit", 0.5 * tp_leg + 0.5 * hold_leg
                    tp_ret = (take_profit_price - buy_price) / buy_price - cost_bi * 0.5
                    return "take_profit", tp_ret

                if trailing_stop_pct is not None and pd.notna(day_high):
                    running_peak = max(running_peak, day_high)

        sell_price = close_df.loc[friday, ticker]
        if pd.isna(sell_price) or sell_price <= 0:
            return "none", 0.0

        gross_ret = (sell_price - buy_price) / buy_price
        net_ret = gross_ret - cost_bi
        return "none", net_ret
    except Exception as e:
        print(f"  [警告] {ticker} {monday.date()}周收益结算失败，按0处理：{type(e).__name__}: {e}")
        return "none", 0.0


def run_walk_forward_backtest(close_df, open_df, high_df, low_df, volume_df, vix_series,
                              min_lookback: int = 120, enable_shap: bool = False,
                              enable_macro_filter: bool = True,
                              regime_mode: str = "threshold",
                              model_objective: str = "regression",
                              use_conformal_filter: bool = False,
                              conformal_coverage: float = 0.90,
                              use_meta_label: bool = False,
                              meta_label_threshold: float = 0.55,
                              earnings_calendar: dict | None = None,
                              enable_earnings_blackout: bool = True,
                              enable_account_circuit_breaker: bool = True,
                              account_dd_lookback: int = 4,
                              account_dd_threshold: float = -0.08,
                              account_dd_scale: float = 0.5,
                              enable_vix_leg: bool = True,
                              enable_momentum_leg: bool = True,
                              n_workers: int = 1,
                              xgb_n_jobs: int = 1,
                              xgb_max_depth: int = 4,
                              xgb_min_child_weight: float = 1.0,
                              xgb_reg_alpha: float = 0.0,
                              xgb_reg_lambda: float = 1.0,
                              xgb_gamma: float = 0.0,
                              xgb_ensemble_size: int = 1,
                              stop_loss_pct: float = STOP_LOSS_PCT,
                              max_pos: int = MAX_POS,
                              max_sector: int = MAX_SECTOR,
                              take_profit_pct: float | None = None,
                              trailing_stop_pct: float | None = None,
                              partial_take_profit: bool = False,
                              staggered_entry_days: int | None = None,
                              use_atr_stop: bool = False,
                              atr_stop_multiplier: float = 1.5,
                              atr_stop_min: float = 0.01,
                              atr_stop_max: float = 0.05,
                              use_correlation_dedup: bool = False,
                              correlation_threshold: float = 0.7,
                              use_calendar_features: bool = False,
                              use_pca_features: bool = False,
                              pca_variance_threshold: float = 0.95,
                              sector_map: dict | None = None,
                              use_full_cross_sectional_standardization: bool = False,
                              use_prediction_calibration: bool = False,
                              model_family: str = "xgboost",
                              use_stacking: bool = False,
                              use_shap_pruning: bool = False,
                              shap_pruned_features: tuple = ("mom_rank", "atr_rank", "volume_divergence",
                                                             "intraday_trend", "mfi"),
                              use_inverse_hedge: bool = False,
                              hedge_fraction: float = 0.3,
                              inverse_etf_close: pd.Series | None = None,
                              point_in_time_membership: dict | None = None,
                              use_cooldown: bool = False,
                              cooldown_weeks: int = 2,
                              use_avoid_consecutive: bool = False,
                              use_kelly_sizing: bool = False,
                              kelly_lookback_trades: int = 30,
                              kelly_scale_min: float = 0.3,
                              kelly_scale_max: float = 1.5,
                              use_streak_scaling: bool = False,
                              streak_threshold: int = 3,
                              streak_scale_down: float = 0.5,
                              streak_scale_up: float = 1.2,
                              pos_size_floor: float = 0.3,
                              pos_size_ceiling: float = 1.0,
                              use_fundamental_features: bool = False,
                              rsi_overbought_threshold: float = RSI_OVERBOUGHT_THRESHOLD,
                              use_multi_horizon_momentum: bool = False,
                              use_sector_relative_momentum: bool = False,
                              use_volume_zscore: bool = False,
                              use_recency_weighting: bool = USE_RECENCY_WEIGHTING,
                              recency_halflife_weeks: float = RECENCY_HALFLIFE_WEEKS,
                              rolling_window_days: int = ROLLING_WINDOW,
                              use_dynamic_position_count: bool = False,
                              use_vol_adjusted_target: bool = False,
                              use_triple_barrier_labels: bool = False,
                              label_stop_loss_pct: float = STOP_LOSS_PCT,
                              macro_df: pd.DataFrame | None = None,
                              use_macro_features: bool = False,
                              cost_bi: float = COST_BI,
                              use_multi_window_fusion: bool = False,
                              fusion_window_days: int = 600,
                              use_ic_quality_gate: bool = False,
                              ic_quality_threshold: float = 0.02,
                              xgb_rank_objective: str = "rank:pairwise",
                              use_stage1_filter: bool = False,
                              stage1_ma_window: int = 50,
                              classification_top_frac: float = 0.1):
    week_pairs = get_weekly_trading_dates(close_df.index)
    print(f"回测区间: {close_df.index[0].date()} ~ {close_df.index[-1].date()}")
    print(f"总周数: {len(week_pairs)}\n")

    # 先筛出真正会跑的周（历史够长），并按原来的顺序确定性地算好哪些周要出
    # SHAP——shap_counter 的进度只取决于"第几个满足 i>50 的合格周"，跟实际
    # 回测结果无关，可以在并行之前就完全确定，保证并行/串行两种跑法选中的
    # SHAP 周完全一样。
    valid_weeks = []
    shap_counter = 0
    for i, (monday, friday) in enumerate(week_pairs):
        hist_before = close_df.loc[:monday]
        if len(hist_before) < min_lookback:
            continue
        loc = close_df.index.get_loc(monday)
        if loc == 0:
            continue
        decision_date = close_df.index[loc - 1]
        do_shap = enable_shap and (shap_counter % 20 == 0) and i > 50
        if do_shap:
            shap_counter += 1
        shap_path = f"shap_week_{i}.png" if do_shap else None
        valid_weeks.append((i, monday, friday, decision_date, do_shap, shap_path))

    config = dict(
        enable_macro_filter=enable_macro_filter, regime_mode=regime_mode,
        model_objective=model_objective, use_conformal_filter=use_conformal_filter,
        conformal_coverage=conformal_coverage, use_meta_label=use_meta_label,
        meta_label_threshold=meta_label_threshold, earnings_calendar=earnings_calendar,
        enable_earnings_blackout=enable_earnings_blackout,
        enable_vix_leg=enable_vix_leg, enable_momentum_leg=enable_momentum_leg,
        xgb_n_jobs=xgb_n_jobs,
        xgb_max_depth=xgb_max_depth, xgb_min_child_weight=xgb_min_child_weight,
        xgb_reg_alpha=xgb_reg_alpha, xgb_reg_lambda=xgb_reg_lambda, xgb_gamma=xgb_gamma,
        xgb_ensemble_size=xgb_ensemble_size,
        stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
        max_pos=max_pos, max_sector=max_sector,
        pos_size_floor=pos_size_floor, pos_size_ceiling=pos_size_ceiling,
        use_fundamental_features=use_fundamental_features,
        rsi_overbought_threshold=rsi_overbought_threshold,
        use_multi_horizon_momentum=use_multi_horizon_momentum,
        use_sector_relative_momentum=use_sector_relative_momentum,
        use_volume_zscore=use_volume_zscore,
        use_recency_weighting=use_recency_weighting,
        recency_halflife_weeks=recency_halflife_weeks,
        rolling_window_days=rolling_window_days,
        use_dynamic_position_count=use_dynamic_position_count,
        use_vol_adjusted_target=use_vol_adjusted_target,
        use_triple_barrier_labels=use_triple_barrier_labels,
        label_stop_loss_pct=label_stop_loss_pct,
        macro_df=macro_df, use_macro_features=use_macro_features,
        cost_bi=cost_bi,
        trailing_stop_pct=trailing_stop_pct, partial_take_profit=partial_take_profit,
        staggered_entry_days=staggered_entry_days,
        use_atr_stop=use_atr_stop, atr_stop_multiplier=atr_stop_multiplier,
        atr_stop_min=atr_stop_min, atr_stop_max=atr_stop_max,
        use_correlation_dedup=use_correlation_dedup, correlation_threshold=correlation_threshold,
        use_calendar_features=use_calendar_features,
        use_pca_features=use_pca_features, pca_variance_threshold=pca_variance_threshold,
        sector_map=sector_map,
        use_full_cross_sectional_standardization=use_full_cross_sectional_standardization,
        use_prediction_calibration=use_prediction_calibration,
        model_family=model_family,
        use_stacking=use_stacking,
        use_shap_pruning=use_shap_pruning,
        shap_pruned_features=shap_pruned_features,
        point_in_time_membership=point_in_time_membership,
        use_multi_window_fusion=use_multi_window_fusion,
        fusion_window_days=fusion_window_days,
        use_ic_quality_gate=use_ic_quality_gate,
        ic_quality_threshold=ic_quality_threshold,
        xgb_rank_objective=xgb_rank_objective,
        use_stage1_filter=use_stage1_filter,
        stage1_ma_window=stage1_ma_window,
        classification_top_frac=classification_top_frac,
    )

    # ---- 并行阶段：每周的选股+结算只依赖截止 decision_date 的历史数据，
    # 不依赖其他周的回测结果，可以放心跨周并行跑。 ----
    week_results: dict = {}
    if n_workers <= 1:
        _init_worker(close_df, open_df, high_df, low_df, volume_df, vix_series, config)
        for done, (i, monday, friday, decision_date, do_shap, shap_path) in enumerate(valid_weeks):
            _, m, pick_results = _compute_week_task(i, monday, friday, decision_date, do_shap, shap_path)
            week_results[i] = (m, pick_results)
            if (done + 1) % 10 == 0:
                print(f"  完成 {done+1}/{len(valid_weeks)} 周...")
    else:
        print(f"  用 {n_workers} 个进程并行跑 {len(valid_weeks)} 周...")
        with ProcessPoolExecutor(
            max_workers=n_workers, initializer=_init_worker,
            initargs=(close_df, open_df, high_df, low_df, volume_df, vix_series, config),
        ) as executor:
            futures = [
                executor.submit(_compute_week_task, i, monday, friday, decision_date, do_shap, shap_path)
                for i, monday, friday, decision_date, do_shap, shap_path in valid_weeks
            ]
            done = 0
            for future in as_completed(futures):
                i, m, pick_results = future.result()
                week_results[i] = (m, pick_results)
                done += 1
                if done % 10 == 0:
                    print(f"  完成 {done}/{len(valid_weeks)} 周...")

    # ---- 串行汇总阶段：账户级熔断、止损冷静期、避免连续选中同票、凯利仓位、
    # 连胜连负动态仓位，这几个都依赖"前几周实际发生了什么"，只能按时间顺序算，
    # 不能像选股+结算那样跨周并行。 ----
    results = []
    cash_yield_weekly = (1 + CASH_YIELD) ** (1 / 52) - 1
    total_stop_losses = 0
    total_take_profits = 0
    recent_realized_returns: list[float] = []
    account_derisked_weeks = 0
    stopped_out_week: dict[str, int] = {}  # ticker -> 最近一次止损发生的周序号
    last_week_tickers: set[str] = set()
    cooldown_skipped = 0
    consecutive_skipped = 0
    recent_trade_returns: list[float] = []  # 最近若干笔单票交易的净收益率（不是组合周收益），供凯利仓位用
    win_streak = 0
    loss_streak = 0
    kelly_scaled_weeks = 0
    streak_scaled_weeks = 0

    for week_idx, (i, monday, friday, decision_date, do_shap, shap_path) in enumerate(valid_weeks):
        _, pick_results = week_results[i]

        account_scale = 1.0
        if enable_account_circuit_breaker and len(recent_realized_returns) >= account_dd_lookback:
            trailing_window = recent_realized_returns[-account_dd_lookback:]
            trailing_cum = np.prod([1 + r for r in trailing_window]) - 1
            if trailing_cum < account_dd_threshold:
                account_scale = account_dd_scale
                account_derisked_weeks += 1

        # 连胜连负动态仓位：比账户熔断更细粒度，跟踪最近的连续盈亏周数而不是
        # 等回撤累积到阈值才反应。连负N周打折仓位，连胜N周适度加仓。
        streak_scale = 1.0
        if use_streak_scaling:
            if loss_streak >= streak_threshold:
                streak_scale = streak_scale_down
                streak_scaled_weeks += 1
            elif win_streak >= streak_threshold:
                streak_scale = streak_scale_up

        # 凯利仓位：用最近 kelly_lookback_trades 笔单票交易的胜率和盈亏比算凯利
        # 最优仓位比例，代替固定的风险平价+置信度打折公式的整体缩放。
        kelly_scale = 1.0
        if use_kelly_sizing and len(recent_trade_returns) >= 10:
            window = recent_trade_returns[-kelly_lookback_trades:]
            wins = [r for r in window if r > 0]
            losses = [r for r in window if r <= 0]
            if wins and losses:
                p = len(wins) / len(window)
                avg_win = np.mean(wins)
                avg_loss = abs(np.mean(losses))
                if avg_loss > 1e-8:
                    b = avg_win / avg_loss
                    kelly_f = p - (1 - p) / b
                    kelly_scale = float(np.clip(kelly_f, kelly_scale_min, kelly_scale_max))
                    kelly_scaled_weeks += 1

        combined_scale = account_scale * streak_scale * kelly_scale

        if not pick_results:
            if use_inverse_hedge and inverse_etf_close is not None:
                hedge_ret = _inverse_etf_weekly_return(inverse_etf_close, monday, friday, cost_bi)
                weekly_return = hedge_fraction * hedge_ret + (1 - hedge_fraction) * cash_yield_weekly
            else:
                weekly_return = cash_yield_weekly
            pick_names = []
            week_trades = []
        else:
            rets = []
            pick_names = []
            this_week_tickers = set()
            week_trades = []
            for ticker, weight, pred_ret, pos_size, net_ret, exit_reason, sector in pick_results:
                if exit_reason == "stop_loss":
                    total_stop_losses += 1
                    stopped_out_week[ticker] = week_idx
                elif exit_reason == "take_profit":
                    total_take_profits += 1

                # 止损冷静期：这只票最近 cooldown_weeks 周内止损过，这周不交易它，
                # 只是把这部分仓位空出来（不替换成下一个候选），效果上等于部分空仓。
                if use_cooldown and ticker in stopped_out_week:
                    weeks_since_stop = week_idx - stopped_out_week[ticker]
                    if 0 < weeks_since_stop <= cooldown_weeks:
                        cooldown_skipped += 1
                        continue

                # 避免连续选中同一只票：跟上周的持仓重叠就跳过，防止模型对某只票
                # 系统性偏好（可能是过拟合到这只票本身，不是真的动量信号）。
                if use_avoid_consecutive and ticker in last_week_tickers:
                    consecutive_skipped += 1
                    continue

                trade_ret = net_ret * weight * pos_size * combined_scale
                rets.append(trade_ret)
                pick_names.append(ticker)
                this_week_tickers.add(ticker)
                recent_trade_returns.append(net_ret)
                week_trades.append({
                    "ticker": ticker, "sector": sector, "pred_ret": pred_ret,
                    "net_ret": net_ret, "exit_reason": exit_reason, "trade_ret": trade_ret,
                })
            weekly_return = sum(rets) if rets else cash_yield_weekly
            last_week_tickers = this_week_tickers

        if use_streak_scaling:
            if weekly_return > 0:
                win_streak += 1
                loss_streak = 0
            elif weekly_return < 0:
                loss_streak += 1
                win_streak = 0

        recent_realized_returns.append(weekly_return)

        prev_nw = results[-1]["net_worth"] if results else 100.0
        net_worth = prev_nw * (1 + weekly_return)

        results.append({
            "date": monday.strftime("%Y-%m-%d"),
            "weekly_return": weekly_return,
            "net_worth": net_worth,
            "picks": pick_names,
            "trades": week_trades,
        })

    print(f"\n  周内止损触发次数: {total_stop_losses}")
    if take_profit_pct:
        print(f"  周内止盈触发次数: {total_take_profits}")
    print(f"  账户级熔断降仓周数: {account_derisked_weeks}"
          f"（最近{account_dd_lookback}周滚动收益跌破{account_dd_threshold:.0%}时仓位打{account_dd_scale:.0%}）")
    if use_cooldown:
        print(f"  止损冷静期跳过次数: {cooldown_skipped}")
    if use_avoid_consecutive:
        print(f"  避免连续选中跳过次数: {consecutive_skipped}")
    if use_kelly_sizing:
        print(f"  凯利仓位生效周数: {kelly_scaled_weeks}")
    if use_streak_scaling:
        print(f"  连负降仓周数: {streak_scaled_weeks}")
    return results


def fetch_market_data(days: int = 1825):
    """
    拉取股票池 + VIX + SPY 基准的历史 OHLCV，做成回测需要的对齐 DataFrame。
    抽成独立函数，方便 backtest.py 的 __main__ 和其他分析脚本
    （比如熔断规则贡献分解）复用，不用重复写一遍下载/对齐逻辑。
    """
    import yfinance as yf

    end = datetime.today()
    start = end - timedelta(days=days)
    tickers = list(SECTOR_MAP.keys())

    print("拉取历史数据（含 Open/High/Low/Close/Volume）...")
    all_tickers = tickers + ["^VIX"]
    # auto_adjust=True：显式要求复权价，避免拆股/分红被误判成暴涨暴跌
    data = yf.download(all_tickers, start=start, end=end, progress=False,
                       group_by="ticker", auto_adjust=True)

    close_dict, open_dict, high_dict, low_dict, vol_dict = {}, {}, {}, {}, {}
    for t in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                df_t = data[t].dropna()
            else:
                df_t = data.dropna()
            if len(df_t) > 60:
                close_dict[t] = df_t["Close"]
                open_dict[t] = df_t["Open"]
                high_dict[t] = df_t["High"]
                low_dict[t] = df_t["Low"]
                vol_dict[t] = df_t["Volume"]
        except Exception as e:
            print(f"  [警告] {t} 数据处理失败，跳过：{type(e).__name__}: {e}")
            continue

    c_df = pd.DataFrame(close_dict).dropna(how="all")
    o_df = pd.DataFrame(open_dict).dropna(how="all")
    h_df = pd.DataFrame(high_dict).dropna(how="all")
    l_df = pd.DataFrame(low_dict).dropna(how="all")
    v_df = pd.DataFrame(vol_dict).dropna(how="all")

    # VIX
    try:
        if isinstance(data.columns, pd.MultiIndex):
            vix = data["^VIX"]["Close"]
        else:
            vix = data["Close"] if "^VIX" in data.columns else None
    except Exception:
        vix = None

    if vix is None or vix.empty:
        print("警告：未获取到 VIX，宏观预警失效。")
        vix = pd.Series(20, index=c_df.index)

    common_idx = c_df.index.intersection(o_df.index).intersection(v_df.index)
    c_df, o_df, h_df, l_df, v_df = [df.loc[common_idx] for df in [c_df, o_df, h_df, l_df, v_df]]
    vix = vix.reindex(common_idx).ffill()

    print(f"有效股票: {len(c_df.columns)}, 交易日: {len(common_idx)}")

    # 基准 SPY
    try:
        spy = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
        spy_close = spy["Close"]
        if isinstance(spy_close, pd.DataFrame):
            spy_close = spy_close.iloc[:, 0]
        spy_close = spy_close.reindex(common_idx).ffill()
    except Exception:
        spy_close = None

    return c_df, o_df, h_df, l_df, v_df, vix, spy_close


def fetch_macro_data(start, end, index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    拉取美元指数(DXY)、10年期国债收益率(TNX)、13周国债收益率(IRX，做收益率
    曲线短端)、高收益债ETF(HYG)、投资级债ETF(LQD，HYG/LQD比价做信用利差代理)
    的原始收盘价，供 us_stock_screener.py 的 use_macro_features 开关用——衍生
    特征（5日变化率）在 _market_context() 里算，这里只负责拉数据、对齐索引。
    reindex+ffill 到传入的交易日索引上，跟股票池数据对齐。
    """
    import yfinance as yf

    tickers = {"dxy": "DX-Y.NYB", "tnx": "^TNX", "irx": "^IRX", "hyg": "HYG", "lqd": "LQD"}
    cols = {}
    for name, t in tickers.items():
        try:
            df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            cols[name] = close.reindex(index).ffill()
        except Exception as e:
            print(f"  [警告] 宏观数据 {name}({t}) 拉取失败，跳过：{type(e).__name__}: {e}")
    if len(cols) < 5:
        print("  [警告] 宏观特征数据不完整，use_macro_features 开启时会退化成全0。")
        return None
    return pd.DataFrame(cols)


def _fetch_one_earnings_date(ticker: str):
    import yfinance as yf
    try:
        df = yf.Ticker(ticker).get_earnings_dates(limit=60)
        if df is not None and not df.empty:
            dates = pd.to_datetime(df.index)
            if dates.tz is not None:
                dates = dates.tz_localize(None)
            return ticker, sorted(dates)
    except Exception:
        pass
    return ticker, None


def fetch_earnings_calendar(tickers: list[str], n_workers: int = 16) -> dict[str, list[pd.Timestamp]]:
    """
    一次性拉取股票池里每只票的历史财报日期，缓存成 {ticker: [财报日期,...]}，
    整个回测期间复用，不用每周重新请求。这是网络 I/O 瓶颈（不是 CPU 瓶颈），
    用线程池并发请求——GIL 不阻塞网络等待，用线程而不是进程池更合适，
    也不用付进程间传数据的开销。

    fail-open：单只票拉取失败/没数据，直接跳过它，不缓存空列表以外的结果，
    调用方看到某只票没有财报日期数据时，默认不对它做财报避让，
    避免数据源抖动误伤整个候选池。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed as thread_as_completed

    calendar: dict[str, list[pd.Timestamp]] = {}
    failed = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_fetch_one_earnings_date, t): t for t in tickers}
        for future in thread_as_completed(futures):
            ticker, dates = future.result()
            if dates is not None:
                calendar[ticker] = dates
            else:
                failed.append(ticker)
    if failed:
        print(f"  [警告] {len(failed)} 只票财报日期拉取失败，这些票不做财报避让。")
    print(f"  财报日历: {len(calendar)}/{len(tickers)} 只票拿到数据")
    return calendar


if __name__ == "__main__":
    import os

    c_df, o_df, h_df, l_df, v_df, vix, spy_close = fetch_market_data()
    common_idx = c_df.index

    # 运行回测（enable_shap=True 会每 20 周输出一张 SHAP 图）；
    # 按 CPU 核数多进程并行跑，每个 worker 里 XGBoost 单线程，避免跟外层的
    # 进程级并行抢核心。
    n_workers = max(1, os.cpu_count() or 1)
    results = run_walk_forward_backtest(
        c_df, o_df, h_df, l_df, v_df, vix, enable_shap=True,
        n_workers=n_workers, xgb_n_jobs=1,
    )

    # 基准周收益（只取 results 里实际用到的那些周，按日期对齐，
    # 不能直接从头截断——run_walk_forward_backtest 会跳过开头历史不足的周）
    bench_rets = None
    if spy_close is not None:
        used_dates = {r["date"] for r in results}
        bench_rets = []
        for monday, friday in get_weekly_trading_dates(common_idx):
            if monday.strftime("%Y-%m-%d") not in used_dates:
                continue
            try:
                bench_rets.append((spy_close.loc[friday] - spy_close.loc[monday]) / spy_close.loc[monday])
            except Exception:
                bench_rets.append(0.0)

    from optimized_metrics_v4 import run_backtest_with_metrics
    run_backtest_with_metrics(results, benchmark_returns=bench_rets)