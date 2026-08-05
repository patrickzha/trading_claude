"""
优化版 AI 选股引擎 v4-fixed
============================
修复：预测阶段截面排名 KeyError
"""
from __future__ import annotations

import bisect
import json
import os
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBRanker, XGBClassifier
from hmmlearn.hmm import GaussianHMM
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ==================== 参数 ====================
MIN_PRICE = 3.0
MIN_VOLUME = 5e5
ROLLING_WINDOW = 300
FORECAST_HORIZON = 5
COST_BI = 0.003
CASH_YIELD = 0.04
MAX_POS = 3
MAX_SECTOR = 2
STOP_LOSS_PCT = 0.02  # 5年标普500回测验证过：5%止损盈亏比0.84，收紧到2%后
                       # 盈亏比1.30、Sharpe 0.42->0.76、样本外Sharpe由负转正
RSI_OVERBOUGHT_THRESHOLD = 100.0  # 5年标普500回测验证过：RSI>75就拒绝候选(旧默认)
                       # CAGR/Sharpe/盈亏比/超额收益/样本外Sharpe全面变差，
                       # 且95%置信区间跨0；改成100(等于不再用RSI过热拒绝候选)后
                       # 全部指标同步改善，CAGR 19.29%->26.80%，Sharpe 0.76->1.00，
                       # 置信区间不再跨0。跟止盈那次的教训是同一个机制：策略选出
                       # 来的都是正在上涨的动量股，"RSI过热就拒绝"砍掉的恰恰是
                       # 动量延续最强的那批候选。
USE_RECENCY_WEIGHTING = True  # 5年标普500回测验证过：训练样本按新近度指数
                       # 衰减加权（半衰期26周），全部指标同步改善：
                       # CAGR 26.80%->37.15%，Sharpe 1.00->1.30，样本外Sharpe
                       # 0.43->1.03（接近翻倍），置信区间[0.22,1.72]->[0.55,2.05]。
                       # 样本外Sharpe大幅改善，直接对上了这个项目一直存在的
                       # "样本外相对样本内明显衰减"问题（过拟合到旧行情的信号）。
RECENCY_HALFLIFE_WEEKS = 26.0

# 原来的50只高beta票（PLTR/COIN/MARA/RIOT这类）清单没有明确的选股依据，
# 只留着当 sp500_universe.json 缺失时的兜底（fail-open，不让脚本因为缺一个
# 数据文件就整个跑不起来）。正式股票池改成标普500，见 build_sp500_universe.py。
_LEGACY_SECTOR_MAP = {
    "PLTR":"TECH","F":"CONS","VALE":"ENERGY","AAL":"CONS","DAL":"CONS",
    "CCL":"CONS","NCLH":"CONS","RCL":"CONS","UBER":"TECH","LYFT":"TECH",
    "SOFI":"FIN","HOOD":"FIN","RIVN":"CONS","LCID":"CONS","PINS":"TECH",
    "SNAP":"TECH","DKNG":"TECH","COIN":"FIN","MARA":"TECH","RIOT":"TECH",
    "MU":"TECH","INTC":"TECH","CSCO":"TECH","PYPL":"FIN","SHOP":"TECH",
    "ROKU":"TECH","PTON":"CONS","ENPH":"ENERGY","SEDG":"ENERGY",
    "FSLR":"ENERGY","PLUG":"ENERGY","NIO":"CONS","XPEV":"CONS","LI":"CONS",
    "BABA":"TECH","JD":"TECH","PDD":"TECH","BIDU":"TECH","NTES":"TECH",
    "KEY":"FIN","ZION":"FIN","HBAN":"FIN","RF":"FIN","CFG":"FIN",
    "FITB":"FIN","C":"FIN","BAC":"FIN","WFC":"FIN","T":"TECH","VZ":"TECH",
}


def _load_sector_map() -> dict:
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sp500_universe.json")
    try:
        with open(json_path) as f:
            data = json.load(f)
        return {ticker: info["sector"] for ticker, info in data["tickers"].items()}
    except Exception as e:
        print(f"  [警告] 未能加载 sp500_universe.json（{type(e).__name__}: {e}），"
              f"退回旧的50只票清单。运行 build_sp500_universe.py 生成标普500股票池。")
        return _LEGACY_SECTOR_MAP


SECTOR_MAP = _load_sector_map()

# GICS 11个行业的数字编码（纯类别标签，跟 build_sp500_universe.py 的
# SECTOR_TO_SHORT 保持一致），"CONS" 是旧50票清单用的简化标签，兜底时对齐到 CONS_D。
SECTOR_CODE_MAP = {
    "COMM": 0, "CONS_D": 1, "CONS_S": 2, "ENERGY": 3, "FIN": 4,
    "HEALTH": 5, "INDU": 6, "TECH": 7, "MATER": 8, "REAL": 9, "UTIL": 10,
    "CONS": 1,
}

FEATURE_NAMES = [
    "mom_1m", "rsi", "v_ratio", "price_to_ma20",
    "atr_ratio", "macd_hist", "adx", "mfi",
    "amplitude_20", "relative_strength", "vix_change_5d", "sector",
    "mom_rank", "rsi_rank", "v_ratio_rank", "atr_rank",
    "intraday_trend", "close_position", "gap", "volume_divergence",
    "regime",
]
# use_fundamental_features=True 时追加在特征向量最后面（见 _extract_feature_vector）
FUNDAMENTAL_FEATURE_NAMES = ["pe_ratio", "pb_ratio", "roe", "leverage"]
# 同样的"默认关闭、条件追加"模式，用于几个还没验证过的新特征方向
MULTI_HORIZON_MOMENTUM_NAMES = ["mom_3m", "mom_6m"]
SECTOR_RELATIVE_MOM_NAMES = ["sector_relative_mom"]
VOLUME_ZSCORE_NAMES = ["volume_zscore"]


def wilder_rsi(prices: np.ndarray, window: int = 14) -> float:
    if len(prices) < window + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:window])
    avg_loss = np.mean(losses[:window])
    if avg_loss == 0:
        return 100.0
    for i in range(window, len(deltas)):
        avg_gain = (avg_gain * (window - 1) + gains[i]) / window
        avg_loss = (avg_loss * (window - 1) + losses[i]) / window
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def detect_regime(vix_curr: float, market_ma20: float, market_ma50: float) -> int:
    if vix_curr > 25 and market_ma20 < market_ma50:
        return 2
    elif vix_curr < 15 and market_ma20 > market_ma50:
        return 0
    return 1


def _detect_regime_hmm(all_closes: pd.DataFrame, vix_series: pd.Series,
                       decision_date, n_states: int = 3) -> int:
    """
    用 GaussianHMM 在（大盘自身日收益率、大盘10日波动率、VIX日变化率）三个维度上
    做无监督状态识别，代替 detect_regime 里硬编码的 VIX 阈值判断。

    Granger 因果检验显示 VIX 变化对这个股票池的未来收益没有显著的领先关系
    （diagnostics_granger_vix.py），所以这里把"大盘自身的动量和波动率"作为
    主要输入，VIX 变化只作为辅助维度加进去，不单独依赖它、也不用它的绝对水平
    设硬阈值。只用截止 decision_date 的历史数据拟合，避免用到未来信息。

    失败（数据不够、拟合报错）时回退到 1（中性），不让 HMM 的问题拖垮主流程。
    """
    try:
        hist = all_closes.loc[:decision_date].tail(400)
        mkt_close = hist.mean(axis=1)
        mkt_ret = mkt_close.pct_change()
        mkt_vol = mkt_ret.rolling(10).std()
        vix_hist = vix_series.loc[:decision_date].reindex(mkt_ret.index).ffill()
        vix_chg = vix_hist.pct_change()

        feat = pd.DataFrame({"mkt_ret": mkt_ret, "mkt_vol": mkt_vol, "vix_chg": vix_chg}).dropna()
        if len(feat) < 100:
            return 1

        X = ((feat - feat.mean()) / (feat.std() + 1e-8)).values
        model = GaussianHMM(n_components=n_states, covariance_type="diag",
                            n_iter=100, random_state=42)
        model.fit(X)
        state_seq = model.predict(X)
        current_state = state_seq[-1]

        # HMM 出来的状态编号是任意的，按"该状态下大盘平均日收益"从高到低
        # 重新映射成 0(健康)/1(中性)/2(危险)，跟原 detect_regime 的含义对齐，
        # 否则同一个物理状态每次刷新可能被贴上不同的数字标签，等于给模型喂噪声。
        state_mean_ret = feat.assign(state=state_seq).groupby("state")["mkt_ret"].mean()
        ordered_states = state_mean_ret.sort_values(ascending=False).index.tolist()
        remap = {state: rank for rank, state in enumerate(ordered_states)}
        return remap[current_state]
    except Exception as e:
        print(f"  [警告] HMM regime 拟合失败，回退为中性(1)：{type(e).__name__}: {e}")
        return 1


def _market_context(all_closes: pd.DataFrame, vix_series: pd.Series,
                    decision_date, cache: dict | None = None,
                    regime_mode: str = "threshold",
                    macro_df: pd.DataFrame | None = None,
                    use_macro_features: bool = False) -> dict:
    """全市场级别统计量（market_ret_20 / vix变化 / regime，可选宏观特征），只依赖
    decision_date，与 ticker 无关，可按 (decision_date, regime_mode, use_macro_features)
    安全缓存（逻辑与之前逐 ticker 内联时完全一致）。regime_mode="threshold" 用原有
    硬阈值规则，"hmm" 用 GaussianHMM。"""
    cache_key = (decision_date, regime_mode, use_macro_features)
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    market_rets = []
    for col in all_closes.columns:
        hist = all_closes[col].loc[:decision_date].tail(20)
        if len(hist) >= 20:
            market_rets.append(hist.iloc[-1] / hist.iloc[0] - 1)
    market_ret_20 = np.mean(market_rets) if market_rets else 0.0

    vix_hist = vix_series.loc[:decision_date].tail(5)
    vix_curr = vix_hist.iloc[-1]
    vix_5ago = vix_hist.iloc[0]
    vix_chg = (vix_curr - vix_5ago) / vix_5ago

    if regime_mode == "hmm":
        regime = _detect_regime_hmm(all_closes, vix_series, decision_date)
    else:
        mkt_ser = all_closes.mean(axis=1).loc[:decision_date].tail(50)
        regime = detect_regime(vix_curr, mkt_ser.tail(20).mean(), mkt_ser.tail(50).mean())

    ctx = {"market_ret_20": market_ret_20, "vix_chg": vix_chg, "regime": regime}

    # 宏观特征扩充：除了 VIX，加信用利差（HYG/LQD比价5日变化，比价走低=信用
    # 利差走阔=risk-off）、美元指数5日变化、国债收益率曲线斜率(10年期-13周
    # 期)5日变化——补上 Granger 检验发现 VIX 单独站不住脚的缺口，看看这几个
    # 更宏观的信号有没有领先关系。macro_df 是原始价格/收益率级数据（列：
    # dxy/tnx/irx/hyg/lqd），只在这里做5日变化的衍生计算。
    if use_macro_features and macro_df is not None:
        try:
            sub = macro_df.loc[:decision_date].tail(5)
            if len(sub) >= 5 and sub.notna().all().all():
                dxy_chg_5d = float((sub["dxy"].iloc[-1] - sub["dxy"].iloc[0]) / sub["dxy"].iloc[0])
                spread = sub["hyg"] / sub["lqd"]
                credit_spread_chg_5d = float((spread.iloc[-1] - spread.iloc[0]) / spread.iloc[0])
                slope = sub["tnx"] - sub["irx"]
                yield_slope_chg_5d = float(slope.iloc[-1] - slope.iloc[0])
            else:
                dxy_chg_5d = credit_spread_chg_5d = yield_slope_chg_5d = 0.0
        except Exception:
            dxy_chg_5d = credit_spread_chg_5d = yield_slope_chg_5d = 0.0
        ctx["dxy_chg_5d"] = dxy_chg_5d
        ctx["credit_spread_chg_5d"] = credit_spread_chg_5d
        ctx["yield_slope_chg_5d"] = yield_slope_chg_5d

    if cache is not None:
        cache[cache_key] = ctx
    return ctx


_SEC_FUNDAMENTALS_CACHE: dict | None = None


def _load_sec_fundamentals() -> dict:
    """
    加载 fetch_sec_fundamentals.py 生成的 sec_fundamentals_cache.json，
    每只票每个字段按披露日期(filed)排好序，供 _lookup_latest_known() 用
    二分查找做"时间点回溯"——只用 decision_date 当天已经真实公开的数据，
    不会有前视偏差。缺文件/加载失败时返回空字典（fail-open），基本面
    特征全部退化成缺失值，不影响技术面特征正常工作。
    """
    global _SEC_FUNDAMENTALS_CACHE
    if _SEC_FUNDAMENTALS_CACHE is not None:
        return _SEC_FUNDAMENTALS_CACHE
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sec_fundamentals_cache.json")
    try:
        with open(json_path) as f:
            raw = json.load(f)
        processed = {}
        for ticker, fields in raw.items():
            processed[ticker] = {}
            for metric, entries in fields.items():
                sorted_entries = sorted(entries, key=lambda e: e["filed"])
                processed[ticker][metric] = {
                    "filed_dates": [e["filed"] for e in sorted_entries],
                    "values": [e["val"] for e in sorted_entries],
                }
        _SEC_FUNDAMENTALS_CACHE = processed
    except Exception as e:
        print(f"  [警告] 未能加载 sec_fundamentals_cache.json（{type(e).__name__}: {e}），"
              f"基本面特征将全部缺失。运行 fetch_sec_fundamentals.py 生成缓存。")
        _SEC_FUNDAMENTALS_CACHE = {}
    return _SEC_FUNDAMENTALS_CACHE


def _lookup_latest_known(field_data: dict, decision_date_str: str) -> float | None:
    """二分查找 filed <= decision_date_str 里最新的一条，没有就返回 None。"""
    filed_dates = field_data["filed_dates"]
    idx = bisect.bisect_right(filed_dates, decision_date_str) - 1
    if idx < 0:
        return None
    return field_data["values"][idx]


def _compute_fundamental_features(ticker: str, decision_date, curr_price: float) -> dict:
    """
    P/E、P/B、ROE、负债权益比，全部用 SEC 披露日期做时间点回溯。缺数据
    (这只票没抓到、缺某个字段、分母是0或负数算不出比率) 时对应特征给
    np.nan——XGBoost 原生支持缺失值，训练时会自己学怎么处理，不用像
    old-school方法那样编一个"看似合理"的默认值去掩盖缺失，那样反而可能
    把缺失当成有信息量的信号喂给模型。

    ps_ratio/gross_margin/ocf_to_ni(见README第324条附近，Cycle74)：
    在原有4个比率之外新增3个，用`fetch_sec_fundamentals.py`新抓的
    revenue/gross_profit/operating_cash_flow字段算出来——直接回应第318条
    "只有4个基础特征，特征信息量不够"这个明确指出的瓶颈，不是重新发明
    新指标，是把已经在抓的原始数据用得更充分。gross_profit覆盖率明显
    低于其他字段(约45%，很多公司在XBRL里不单独披露这条科目，是数据
    本身的限制，不是抓取逻辑的问题)，缺失时正常给nan。
    """
    cache = _load_sec_fundamentals()
    nan = float("nan")
    result = {"pe_ratio": nan, "pb_ratio": nan, "roe": nan, "leverage": nan,
              "ps_ratio": nan, "gross_margin": nan, "ocf_to_ni": nan,
              "revenue_growth_yoy": nan}

    ticker_data = cache.get(ticker)
    if not ticker_data:
        return result

    decision_str = decision_date.strftime("%Y-%m-%d")
    prior_year_str = (decision_date - pd.Timedelta(days=365)).strftime("%Y-%m-%d")

    net_income = (_lookup_latest_known(ticker_data["net_income"], decision_str)
                 if "net_income" in ticker_data else None)
    equity = (_lookup_latest_known(ticker_data["stockholders_equity"], decision_str)
             if "stockholders_equity" in ticker_data else None)
    liabilities = (_lookup_latest_known(ticker_data["liabilities"], decision_str)
                  if "liabilities" in ticker_data else None)
    shares = (_lookup_latest_known(ticker_data["shares_outstanding"], decision_str)
             if "shares_outstanding" in ticker_data else None)
    revenue = (_lookup_latest_known(ticker_data["revenue"], decision_str)
              if "revenue" in ticker_data else None)
    gross_profit = (_lookup_latest_known(ticker_data["gross_profit"], decision_str)
                   if "gross_profit" in ticker_data else None)
    operating_cash_flow = (_lookup_latest_known(ticker_data["operating_cash_flow"], decision_str)
                           if "operating_cash_flow" in ticker_data else None)

    if net_income is not None and shares and shares > 0:
        eps = net_income / shares
        if eps > 0:
            result["pe_ratio"] = curr_price / eps

    if equity is not None and shares and shares > 0 and equity > 0:
        book_value_per_share = equity / shares
        if book_value_per_share > 0:
            result["pb_ratio"] = curr_price / book_value_per_share

    if net_income is not None and equity and equity > 0:
        result["roe"] = net_income / equity

    if liabilities is not None and equity and equity > 0:
        result["leverage"] = liabilities / equity

    if revenue is not None and revenue > 0 and shares and shares > 0:
        sales_per_share = revenue / shares
        if sales_per_share > 0:
            result["ps_ratio"] = curr_price / sales_per_share

    if gross_profit is not None and revenue is not None and revenue > 0:
        result["gross_margin"] = gross_profit / revenue

    if operating_cash_flow is not None and net_income is not None and net_income != 0:
        result["ocf_to_ni"] = operating_cash_flow / net_income

    if "revenue" in ticker_data:
        revenue_prior = _lookup_latest_known(ticker_data["revenue"], prior_year_str)
        if revenue is not None and revenue_prior is not None and revenue_prior > 0:
            result["revenue_growth_yoy"] = revenue / revenue_prior - 1

    return result


def build_raw_features(ticker, c_series, o_series, h_series, l_series, v_series,
                       vix_series, sector, all_closes, decision_date,
                       market_cache: dict | None = None,
                       regime_mode: str = "threshold",
                       use_fundamental_features: bool = False,
                       rsi_overbought_threshold: float = RSI_OVERBOUGHT_THRESHOLD,
                       use_multi_horizon_momentum: bool = False,
                       use_volume_zscore: bool = False,
                       macro_df: pd.DataFrame | None = None,
                       use_macro_features: bool = False,
                       use_calendar_features: bool = False) -> dict | None:
    if len(c_series) < 60:
        return None
    c = c_series.values
    o = o_series.values
    h = h_series.values
    l = l_series.values
    v = v_series.values
    curr_p = c[-1]

    if curr_p < MIN_PRICE or np.mean(v[-20:]) < MIN_VOLUME:
        return None

    ma20 = np.mean(c[-20:])
    ma5 = np.mean(c[-5:])
    if curr_p < ma20 or ma5 < ma20:
        return None

    mom_1m = (c[-1] - c[-20]) / c[-20]
    if mom_1m <= 0.01:
        return None

    rsi = wilder_rsi(c, 14)
    if rsi > rsi_overbought_threshold:
        return None

    v_ratio = np.mean(v[-5:]) / np.mean(v[-30:]) if np.mean(v[-30:]) > 0 else 1.0
    price_to_ma20 = curr_p / ma20

    # 多周期动量：学术上3-6个月动量比1个月更稳健，1个月动量单独用噪声偏大。
    # 历史不够长时给 NaN，交给 XGBoost 原生缺失值处理（跟基本面特征同样的模式）。
    mom_3m = (c[-1] - c[-63]) / c[-63] if len(c) >= 64 else np.nan
    mom_6m = (c[-1] - c[-126]) / c[-126] if len(c) >= 127 else np.nan

    # 成交量异常：现在的 v_ratio 只看5日/30日均量比值，这里换个角度，看当前
    # 5日均量相对过去60日均量分布的偏离程度（标准化后的z分数），捕捉"放量"
    # 这种更极端的资金进场信号，跟 v_ratio 不是同一个维度。
    vol_hist = v[-60:] if len(v) >= 60 else v
    vol_mean, vol_std = np.mean(vol_hist), np.std(vol_hist)
    volume_zscore = (np.mean(v[-5:]) - vol_mean) / (vol_std + 1e-8)

    # 已实现波动率（20日日收益率标准差）——不进特征向量，只在
    # use_vol_adjusted_target=True 时用来把训练标签从"原始收益率"换成
    # "收益率/波动率"，让模型学"风险调整后谁更值得买"而不是单纯"谁涨得多"。
    # 纯用过去数据算，不涉及未来信息。
    daily_rets = np.diff(c[-21:]) / c[-21:-1] if len(c) >= 21 else np.diff(c) / c[:-1]
    realized_vol_20d = float(np.std(daily_rets)) if len(daily_rets) > 0 else 0.0

    # 月末/季末效应：机构月末/季末再平衡可能对动量股有额外买卖盘压力。
    # decision_date 到当月最后一天的自然日天数 <=5 算"月末周"，季末月份
    # （3/6/9/12月）里的月末周再单独标一个季末标记。
    month_end_date = decision_date + pd.offsets.MonthEnd(0)
    days_to_month_end = (month_end_date - decision_date).days
    is_month_end_week = 1.0 if days_to_month_end <= 5 else 0.0
    is_quarter_end_week = 1.0 if (is_month_end_week and decision_date.month in (3, 6, 9, 12)) else 0.0

    prev_c = c[-15:-1]
    tr1 = h[-14:] - l[-14:]
    tr2 = np.abs(h[-14:] - prev_c)
    tr3 = np.abs(l[-14:] - prev_c)
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = np.mean(tr) if len(tr) > 0 else 0.0
    atr_ratio = atr / curr_p if curr_p > 0 else 0.0

    s = pd.Series(c)
    ema12 = s.ewm(span=12, adjust=False).mean().iloc[-1]
    ema26 = s.ewm(span=26, adjust=False).mean().iloc[-1]
    macd_hist = ema12 - ema26

    diffs = np.diff(c)
    atr14 = np.mean(np.abs(diffs)[-14:]) if len(diffs) >= 14 else np.mean(np.abs(diffs))
    plus_dm = np.where(diffs > 0, diffs, 0)
    minus_dm = np.where(diffs < 0, -diffs, 0)
    di_plus = 100 * np.mean(plus_dm[-14:]) / (atr14 + 1e-8)
    di_minus = 100 * np.mean(minus_dm[-14:]) / (atr14 + 1e-8)
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-8)
    adx = dx

    tp = (h + l + c) / 3
    mfr = tp * v
    pos_flow = np.sum(mfr[-14:][diffs[-14:] > 0]) if len(diffs) >= 14 else 0
    neg_flow = np.sum(mfr[-14:][diffs[-14:] < 0]) if len(diffs) >= 14 else 0
    mfi = 100 - (100 / (1 + pos_flow / (neg_flow + 1e-8)))

    amp20 = (np.max(c[-20:]) - np.min(c[-20:])) / np.mean(c[-20:])

    mkt_ctx = _market_context(all_closes, vix_series, decision_date, market_cache, regime_mode,
                              macro_df=macro_df, use_macro_features=use_macro_features)
    market_ret_20 = mkt_ctx["market_ret_20"]
    vix_chg = mkt_ctx["vix_chg"]
    regime = mkt_ctx["regime"]
    rel_strength = (c[-1] / c[-20]) - 1 - market_ret_20

    intraday_trend = (c[-1] - o[-1]) / (h[-1] - l[-1] + 1e-8)
    close_position = (c[-1] - l[-1]) / (h[-1] - l[-1] + 1e-8)
    gap = (o[-1] - c[-2]) / c[-2] if len(c) >= 2 else 0.0
    price_new_high = c[-1] >= np.max(c[-20:-1]) if len(c) >= 21 else False
    vol_new_high = v[-1] >= np.max(v[-20:-1]) if len(v) >= 21 else False
    volume_divergence = 1.0 if price_new_high and not vol_new_high else 0.0

    sector_code = SECTOR_CODE_MAP.get(sector, 0)

    feat = {
        "mom_1m": mom_1m, "rsi": rsi, "v_ratio": v_ratio, "price_to_ma20": price_to_ma20,
        "atr_ratio": atr_ratio, "macd_hist": macd_hist, "adx": adx, "mfi": mfi,
        "amplitude_20": amp20, "relative_strength": rel_strength,
        "vix_change_5d": vix_chg, "sector": sector_code,
        "intraday_trend": intraday_trend, "close_position": close_position,
        "gap": gap, "volume_divergence": volume_divergence,
        "regime": regime, "atr": atr, "realized_vol_20d": realized_vol_20d,
    }
    if use_fundamental_features:
        feat.update(_compute_fundamental_features(ticker, decision_date, curr_p))
    if use_multi_horizon_momentum:
        feat["mom_3m"] = mom_3m
        feat["mom_6m"] = mom_6m
    if use_volume_zscore:
        feat["volume_zscore"] = volume_zscore
    if use_macro_features and "dxy_chg_5d" in mkt_ctx:
        feat["dxy_chg_5d"] = mkt_ctx["dxy_chg_5d"]
        feat["credit_spread_chg_5d"] = mkt_ctx["credit_spread_chg_5d"]
        feat["yield_slope_chg_5d"] = mkt_ctx["yield_slope_chg_5d"]
    if use_calendar_features:
        feat["is_month_end_week"] = is_month_end_week
        feat["is_quarter_end_week"] = is_quarter_end_week
    return feat


def _triple_barrier_return(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series,
                           j: int, forecast_horizon: int, stop_loss_pct: float,
                           cost_bi: float = COST_BI) -> float:
    """
    训练标签跟实盘规则对齐：现在的简单标签是"周一到周五的收盘价收益率"，但
    实盘持仓周内有2%止损规则，模型没学到"这笔单会不会提前止损出局"这件事。
    这里用未来 forecast_horizon 天的日内最低价判断是否会先触发止损，逻辑跟
    backtest.py 的 simulate_weekly_return() 完全一致（止损从买入次日开始检查、
    触发按止损价结算只收一半成本，没触发则按到期收盘价结算收全额成本）。
    """
    buy_price = o.iloc[j + 1] if j + 1 < len(o) else c.iloc[j + 1]
    stop_price = buy_price * (1 - stop_loss_pct)
    end_idx = min(j + forecast_horizon, len(c) - 1)
    for k in range(j + 2, end_idx + 1):
        if k < len(l) and pd.notna(l.iloc[k]) and l.iloc[k] <= stop_price:
            return (stop_price - buy_price) / buy_price - cost_bi * 0.5
    exit_price = c.iloc[end_idx]
    return (exit_price - buy_price) / buy_price - cost_bi


EXTRA_RANK_FEATURES = ["price_to_ma20", "macd_hist", "adx", "mfi", "amplitude_20",
                       "relative_strength", "intraday_trend", "close_position", "gap"]


def _add_cross_sectional_ranks(raw_dict: dict, extra_features: list | None = None) -> dict:
    """
    计算截面排名，返回新字典（深拷贝），不修改输入。原来只对 mom_1m/rsi/v_ratio/
    atr_ratio 4个特征算排名，其余连续特征（price_to_ma20/macd_hist等）留着原始
    值，牛熊市之间scale会漂移。extra_features（配合 use_full_cross_sectional_
    standardization 开关）给 EXTRA_RANK_FEATURES 里的特征也补上排名版本，不删掉
    原始值，只是追加。
    """
    if not raw_dict:
        return {}

    result = {t: dict(f) for t, f in raw_dict.items()}
    rank_features = ["mom_1m", "rsi", "v_ratio", "atr_ratio"] + (extra_features or [])

    if len(result) < 2:
        for t in result:
            for feat_name in rank_features:
                result[t][f"{feat_name}_rank"] = 0.5
        return result

    for feat_name in rank_features:
        vals = [(t, result[t][feat_name]) for t in result]
        vals_sorted = sorted(vals, key=lambda x: x[1])
        n = len(vals_sorted)
        for rank, (t, _) in enumerate(vals_sorted):
            result[t][f"{feat_name}_rank"] = rank / max(n - 1, 1)

    return result


def _add_sector_relative_momentum(raw_dict: dict) -> dict:
    """
    现有 relative_strength 是相对大盘算的，这里换成相对同行业当天横截面均值——
    去掉行业本身的beta（比如某天科技股整体都涨），只留"在同行业里谁更强"这个
    更干净的信号。同一个函数结构（不修改输入、返回深拷贝）跟 _add_cross_sectional_ranks
    保持一致。
    """
    if not raw_dict:
        return {}
    result = {t: dict(f) for t, f in raw_dict.items()}
    by_sector: dict = {}
    for t, f in result.items():
        by_sector.setdefault(f["sector"], []).append(f["mom_1m"])
    sector_avg = {sec: float(np.mean(moms)) for sec, moms in by_sector.items()}
    for t, f in result.items():
        result[t]["sector_relative_mom"] = f["mom_1m"] - sector_avg[f["sector"]]
    return result


def _check_vix_surge(vix_series: pd.Series, target_date: pd.Timestamp,
                     threshold: float = 0.30) -> tuple[bool, str]:
    """VIX 5日涨幅超阈值。Granger 检验（diagnostics_granger_vix.py）显示这条腿
    对该股票池的未来收益没有显著领先关系，统计上站不住脚，默认仍保留是为了
    能在 check_macro_warning 里独立开关做 A/B，不是因为已证明它有效。"""
    vix_sub = vix_series.loc[:target_date].tail(30)
    if len(vix_sub) < 5:
        return False, ""
    vix_curr = vix_sub.iloc[-1]
    vix_5ago = vix_sub.iloc[-5]
    chg = (vix_curr - vix_5ago) / vix_5ago
    if chg > threshold:
        return True, f"🚨 VIX熔断 {target_date.date()}: {chg:.1%}"
    return False, ""


def _check_momentum_breakdown(closes_df: pd.DataFrame, target_date: pd.Timestamp) -> tuple[bool, str]:
    """大盘动量/回撤破位。alpha 分解（analyze_macro_filter_contribution.py）显示
    熔断规则整体贡献了历史上几乎全部的正收益——这条腿大概率是主要来源
    （VIX 那条已被 Granger 检验证伪），具体贡献大小需要单独跑分解才能确认。"""
    sub = closes_df.loc[:target_date].tail(30)
    if len(sub) < 20:
        return False, ""
    norm = sub.div(sub.iloc[0]) * 100
    mkt = norm.mean(axis=1)
    recent_ret = (mkt.iloc[-1] - mkt.iloc[-10]) / mkt.iloc[-10]
    dd = (mkt.iloc[-1] - mkt.cummax().iloc[-1]) / mkt.cummax().iloc[-1]
    ma5, ma20 = mkt.tail(5).mean(), mkt.tail(20).mean()
    weekly = mkt.pct_change().dropna().tail(20)
    dyn = np.percentile(weekly, 10) * 10 if len(weekly) >= 10 else -0.04
    if recent_ret < dyn or (ma5 < ma20 and dd < -0.03):
        return True, f"🚨 动量熔断 {target_date.date()}: 回撤{recent_ret:.1%}"
    return False, ""


def check_macro_warning(closes_df: pd.DataFrame, vix_series: pd.Series,
                        target_date: pd.Timestamp,
                        enable_vix_leg: bool = True,
                        enable_momentum_leg: bool = True) -> tuple[bool, str]:
    """宏观熔断：VIX突增 或 大盘动量破位，任一触发即空仓。两条腿可独立开关，
    用来做 alpha 分解（究竟是哪条腿在真正起作用）。"""
    try:
        if enable_vix_leg:
            triggered, msg = _check_vix_surge(vix_series, target_date)
            if triggered:
                return True, msg
        if enable_momentum_leg:
            triggered, msg = _check_momentum_breakdown(closes_df, target_date)
            if triggered:
                return True, msg
    except Exception as e:
        print(f"  [警告] {target_date.date()} 宏观预警计算失败，本周按无风险处理："
              f"{type(e).__name__}: {e}")
    return False, ""


def _has_upcoming_earnings(ticker: str, target_date, close_index: pd.DatetimeIndex,
                           earnings_calendar: dict | None, horizon: int = 5) -> bool:
    """
    检查 (target_date, target_date之后horizon个交易日] 这个持仓窗口内该票有没有财报。
    fail-open：earnings_calendar 里没有这只票的数据，直接当作"没有财报"处理，
    不因为数据源抖动误伤整个候选池。
    """
    if not earnings_calendar:
        return False
    dates = earnings_calendar.get(ticker)
    if not dates:
        return False
    try:
        loc = close_index.get_loc(target_date)
    except KeyError:
        return False
    future_days = close_index[loc + 1: loc + 1 + horizon]
    if len(future_days) == 0:
        return False
    window_start, window_end = future_days[0], future_days[-1]
    return any(window_start <= d <= window_end for d in dates)


def _extract_feature_vector(feat: dict, excluded_features: frozenset | None = None) -> list:
    """从特征字典中提取固定顺序的特征向量，.get() 兜底防止 KeyError。
    "pe_ratio" 只在 use_fundamental_features=True 时才会出现在 feat 里，
    这个标志在一整次 get_ai_weekly_picks() 调用里对所有票统一生效，
    不会出现同一批训练矩阵里有的行21维、有的行25维这种参差不齐的情况。

    excluded_features（配合 use_shap_pruning 开关）按名字排除掉基础21个
    特征里贡献最小的几个——SHAP在三个不同decision_date上一致显示
    mom_rank/atr_rank贡献几乎为0，本身就是"排名"版本的mom_1m/atr_ratio，
    可能是重复信息在拖累树模型分裂效率。"""
    base_named = [
        ("mom_1m", feat["mom_1m"]), ("rsi", feat["rsi"]), ("v_ratio", feat["v_ratio"]),
        ("price_to_ma20", feat["price_to_ma20"]), ("atr_ratio", feat["atr_ratio"]),
        ("macd_hist", feat["macd_hist"]), ("adx", feat["adx"]), ("mfi", feat["mfi"]),
        ("amplitude_20", feat["amplitude_20"]), ("relative_strength", feat["relative_strength"]),
        ("vix_change_5d", feat["vix_change_5d"]), ("sector", feat["sector"]),
        ("mom_rank", feat.get("mom_rank", 0.5)), ("rsi_rank", feat.get("rsi_rank", 0.5)),
        ("v_ratio_rank", feat.get("v_ratio_rank", 0.5)), ("atr_rank", feat.get("atr_rank", 0.5)),
        ("intraday_trend", feat["intraday_trend"]), ("close_position", feat["close_position"]),
        ("gap", feat["gap"]), ("volume_divergence", feat["volume_divergence"]),
        ("regime", feat["regime"]),
    ]
    if excluded_features:
        base_named = [(n, v) for n, v in base_named if n not in excluded_features]
    vec = [v for _, v in base_named]
    if "pe_ratio" in feat:
        vec.extend([feat["pe_ratio"], feat["pb_ratio"], feat["roe"], feat["leverage"]])
    if "mom_3m" in feat:
        vec.extend([feat["mom_3m"], feat["mom_6m"]])
    if "sector_relative_mom" in feat:
        vec.append(feat["sector_relative_mom"])
    if "volume_zscore" in feat:
        vec.append(feat["volume_zscore"])
    if "dxy_chg_5d" in feat:
        vec.extend([feat["dxy_chg_5d"], feat["credit_spread_chg_5d"], feat["yield_slope_chg_5d"]])
    if "is_month_end_week" in feat:
        vec.extend([feat["is_month_end_week"], feat["is_quarter_end_week"]])
    if f"{EXTRA_RANK_FEATURES[0]}_rank" in feat:
        vec.extend([feat[f"{name}_rank"] for name in EXTRA_RANK_FEATURES])
    return vec


def get_ai_weekly_picks(close_df, open_df, high_df, low_df, volume_df,
                        vix_series, target_date, shap_enabled: bool = False,
                        shap_output_path: str = "shap_summary.png",
                        market_cache: dict | None = None,
                        enable_macro_filter: bool = True,
                        regime_mode: str = "threshold",
                        model_objective: str = "regression",
                        use_conformal_filter: bool = False,
                        conformal_coverage: float = 0.90,
                        use_meta_label: bool = False,
                        meta_label_threshold: float = 0.55,
                        earnings_calendar: dict | None = None,
                        enable_earnings_blackout: bool = True,
                        earnings_horizon: int = 5,
                        enable_vix_leg: bool = True,
                        enable_momentum_leg: bool = True,
                        xgb_n_jobs: int = 1,
                        xgb_max_depth: int = 4,
                        xgb_min_child_weight: float = 1.0,
                        xgb_reg_alpha: float = 0.0,
                        xgb_reg_lambda: float = 1.0,
                        xgb_gamma: float = 0.0,
                        xgb_ensemble_size: int = 1,
                        max_pos: int = MAX_POS,
                        max_sector: int = MAX_SECTOR,
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
                        use_multi_window_fusion: bool = False,
                        fusion_window_days: int = 600,
                        use_ic_quality_gate: bool = False,
                        ic_quality_threshold: float = 0.02,
                        xgb_rank_objective: str = "rank:pairwise",
                        use_stage1_filter: bool = False,
                        stage1_ma_window: int = 50,
                        classification_top_frac: float = 0.1,
                        _return_raw_candidates: bool = False):
    # 时间维度模型融合用：递归调用自身跑另一个窗口长度时，直接把这次调用
    # 收到的全部参数原样透传过去（只改 rolling_window_days 等几个字段），
    # 不用把几十个参数再抄一遍——签名以后再加参数也不用记得同步这里。
    _all_args = dict(locals())
    if sector_map is None:
        sector_map = SECTOR_MAP
    if enable_macro_filter:
        is_danger, warning = check_macro_warning(
            close_df, vix_series, target_date,
            enable_vix_leg=enable_vix_leg, enable_momentum_leg=enable_momentum_leg,
        )
        if is_danger:
            print(warning)
            return []

    if market_cache is None:
        market_cache = {}

    # 注意：训练样本本来就是滚动窗口（默认约300+60个交易日≈1.4年），不是拿
    # 全部历史扩张训练——rolling_window_days 只是把这个窗口长度做成可调参数，
    # 用来测试"窗口更短更快适应regime变化"还是"窗口更长样本更多更稳"哪个好，
    # 不是从扩张窗口换成滚动窗口（本来就是滚动窗口）。
    history = close_df.loc[:target_date].tail(rolling_window_days + 60)
    o_history = open_df.loc[:target_date].tail(rolling_window_days + 60)
    h_history = high_df.loc[:target_date].tail(rolling_window_days + 60)
    l_history = low_df.loc[:target_date].tail(rolling_window_days + 60)
    v_history = volume_df.loc[:target_date].tail(rolling_window_days + 60)

    if len(history) < 100:
        return []

    # ---- 收集训练样本 ----
    samples_by_date = {}

    for ticker in history.columns:
        c = history[ticker].dropna()
        o = o_history[ticker].dropna() if ticker in o_history.columns else pd.Series()
        h = h_history[ticker].dropna() if ticker in h_history.columns else pd.Series()
        l = l_history[ticker].dropna() if ticker in l_history.columns else pd.Series()
        v = v_history[ticker].dropna() if ticker in v_history.columns else pd.Series()

        if len(c) < 60 or len(o) < 60 or len(h) < 60 or len(l) < 60 or len(v) < 60:
            continue

        sector = sector_map.get(ticker, "TECH")
        dates = c.index

        for j in range(40, len(c) - FORECAST_HORIZON):
            decision_d = dates[j]
            feat = build_raw_features(
                ticker, c.iloc[:j+1], o.iloc[:j+1], h.iloc[:j+1],
                l.iloc[:j+1], v.iloc[:j+1],
                vix_series, sector, close_df.loc[:decision_d], decision_d,
                market_cache=market_cache, regime_mode=regime_mode,
                use_fundamental_features=use_fundamental_features,
                rsi_overbought_threshold=rsi_overbought_threshold,
                use_multi_horizon_momentum=use_multi_horizon_momentum,
                use_volume_zscore=use_volume_zscore,
                macro_df=macro_df, use_macro_features=use_macro_features,
                use_calendar_features=use_calendar_features,
            )
            if feat is None:
                continue

            if use_triple_barrier_labels:
                future_ret = _triple_barrier_return(o, h, l, c, j, FORECAST_HORIZON, label_stop_loss_pct,
                                                    cost_bi=cost_bi)
            else:
                future_open = o.iloc[j + 1] if j + 1 < len(o) else c.iloc[j + 1]
                future_close = c.iloc[j + FORECAST_HORIZON]
                future_ret = (future_close - future_open) / future_open - cost_bi
            if use_vol_adjusted_target:
                # 训练标签换成"波动率调整后收益"，让模型学风险调整后谁更值得买，
                # 不是单纯谁涨得多。预测阶段会用同样的 realized_vol_20d 换算回
                # 原始收益率尺度，pred_ret 的含义（仓位计算、日志展示）不受影响。
                future_ret = future_ret / (feat["realized_vol_20d"] + 1e-4)

            if decision_d not in samples_by_date:
                samples_by_date[decision_d] = []
            samples_by_date[decision_d].append((ticker, feat, future_ret))

    total_samples = sum(len(v) for v in samples_by_date.values())
    if total_samples < 80:
        print(f"  [提示] 训练样本仅 {total_samples} 条，跳过。")
        return []

    # === 逐日分组截面排名 ===
    ranked_samples = []
    for d, group in samples_by_date.items():
        raw_dict = {t: f for t, f, _ in group}
        ranked_dict = _add_cross_sectional_ranks(
            raw_dict, extra_features=EXTRA_RANK_FEATURES if use_full_cross_sectional_standardization else None)
        if use_sector_relative_momentum:
            ranked_dict = _add_sector_relative_momentum(ranked_dict)
        for t, f, r in group:
            ranked_samples.append((d, t, ranked_dict[t], r))

    # 构建训练矩阵
    train_X, train_y, train_dates = [], [], []
    excluded_features = frozenset(shap_pruned_features) if use_shap_pruning else None
    for decision_d, ticker, feat, future_ret in ranked_samples:
        train_X.append(_extract_feature_vector(feat, excluded_features=excluded_features))
        train_y.append(future_ret)
        train_dates.append(decision_d)

    X_arr = np.array(train_X)
    y_arr = np.array(train_y)
    dates_arr = pd.to_datetime(pd.Series(train_dates)).values

    # 时序验证
    sort_idx = np.argsort(dates_arr)
    X_s, y_s = X_arr[sort_idx], y_arr[sort_idx]
    # 排序目标（rank:pairwise）需要按 query group（这里是 decision_date）分组，
    # 且同组行必须连续——上面已经按日期整体排序，这里转成整数组号即可。
    qid_s = pd.factorize(dates_arr[sort_idx])[0]
    split = int(len(X_s) * 0.8)
    X_tr, X_val = X_s[:split], X_s[split:]
    y_tr, y_val = y_s[:split], y_s[split:]
    qid_tr, qid_val = qid_s[:split], qid_s[split:]

    # PCA特征去相关：21个原始特征里有些本身高度相关（比如mom_1m和mom_rank），
    # 冗余信息可能在干扰树模型分裂。用训练折自己拟合PCA（不看验证折，避免
    # 信息泄漏），转成互相正交的主成分，保留能解释pca_variance_threshold
    # 比例方差的成分数。全量训练（下面 X_s）会单独再拟合一次PCA（用全部
    # 训练数据），预测阶段要用这个版本做变换保持一致，存在 pca_full 里。
    # sklearn的PCA不像XGBoost原生支持NaN——项目里默认允许特征里有NaN（交给
    # XGBoost自己处理缺失值），PCA前必须先用训练折自己的均值填补，否则遇到
    # 任何一行有NaN（比如mom_3m/mom_6m历史不够、基本面数据缺失）直接报错。
    pca_full = None
    imputer_full = None
    # 线性模型族（model_family="linear"）跟PCA一样不原生支持NaN，需要同一套
    # 均值填补。两个开关任一打开都需要填补，用同一个 imputer_diag/imputer_full。
    needs_imputation = use_pca_features or model_family == "linear" or use_stacking
    if needs_imputation:
        from sklearn.impute import SimpleImputer
        imputer_diag = SimpleImputer(strategy="mean")
        X_tr = imputer_diag.fit_transform(X_tr)
        if len(X_val) > 0:
            X_val = imputer_diag.transform(X_val)
    if use_pca_features:
        from sklearn.decomposition import PCA
        pca_diag = PCA(n_components=pca_variance_threshold, svd_solver="full")
        X_tr = pca_diag.fit_transform(X_tr)
        if len(X_val) > 0:
            X_val = pca_diag.transform(X_val)

    # 样本新近度加权：默认关闭时权重全为None，行为不变。开启时按离最新决策日的
    # 天数做指数衰减（半衰期默认52周），同一个 decision_date 下所有票权重相同，
    # 不影响 XGBRanker 组内的相对排序，只影响不同周之间谁在训练里更"说了算"。
    weight_s = weight_tr = weight_val = None
    if use_recency_weighting:
        dates_sorted = dates_arr[sort_idx]
        latest_date = dates_sorted.max()
        days_from_latest = (latest_date - dates_sorted).astype("timedelta64[D]").astype(float)
        weight_s = 0.5 ** (days_from_latest / (recency_halflife_weeks * 7.0))
        weight_tr, weight_val = weight_s[:split], weight_s[split:]

    def _make_model(seed: int = 42):
        # model_family="linear"：换成ElasticNet代替XGBoost树模型，测试"IC经常
        # 接近0"是不是因为树模型在这种低信噪比数据上容易过拟合到噪声分裂点——
        # 线性模型结构简单得多，如果反而更稳，说明问题出在模型复杂度而不是特征。
        # 只支持回归（ElasticNet没有排序目标的等价物），忽略 model_objective。
        if model_family == "linear":
            from sklearn.linear_model import ElasticNet
            return ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=seed, max_iter=5000)
        # 正则化参数默认值 = XGBoost 自己的默认值，不传就是原来的行为。
        common_kwargs = dict(
            n_estimators=100, max_depth=xgb_max_depth, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=xgb_min_child_weight,
            reg_alpha=xgb_reg_alpha, reg_lambda=xgb_reg_lambda, gamma=xgb_gamma,
            random_state=seed, verbosity=0, n_jobs=xgb_n_jobs,
        )
        if model_objective == "ranking":
            return XGBRanker(objective=xgb_rank_objective, **common_kwargs)
        if model_objective == "classification":
            return XGBClassifier(objective="binary:logistic", **common_kwargs)
        return XGBRegressor(objective='reg:pseudohubererror', **common_kwargs)

    def _to_topk_label(y, qid, top_frac: float = 0.1):
        # 回归换分类：预测"这只票能不能进入当周横截面涨幅前top_frac"，而不是
        # 预测具体收益率数值——同一组(qid=同一个decision_date)内收益率最高
        # 的 top_frac 比例标1，其余标0，直接把问题从"这只票未来涨多少"换成
        # "这只票是不是这周的赢家"，跟回归/pairwise排序都是不同的优化目标。
        y = np.asarray(y, dtype=float)
        label = np.zeros(len(y), dtype=np.int32)
        for g in np.unique(qid):
            mask = qid == g
            n = int(mask.sum())
            if n <= 1:
                continue
            k = max(1, int(np.ceil(n * top_frac)))
            threshold = np.sort(y[mask])[-k]
            label[mask] = (y[mask] >= threshold).astype(np.int32)
        return label

    def _to_ndcg_relevance(y, qid, n_grades: int = 5):
        # rank:ndcg 要求 label 是非负整数"相关度分级"，不接受连续收益率——
        # 按组内（同一个qid=同一个decision_date）收益率排名分成 n_grades 档
        # （0=组内最差一档，n_grades-1=组内最好一档），直接把"组合级/横截面
        # 相对排序"这个目标编码进训练标签本身，而不是像 rank:pairwise 那样
        # 直接拿连续收益率去做两两比较。
        y = np.asarray(y, dtype=float)
        relevance = np.zeros(len(y), dtype=np.int32)
        for g in np.unique(qid):
            mask = qid == g
            n = int(mask.sum())
            if n <= 1:
                continue
            order = np.argsort(np.argsort(y[mask]))  # 组内从小到大的名次(0..n-1)
            grade = np.floor(order / n * n_grades).astype(np.int32)
            relevance[mask] = np.clip(grade, 0, n_grades - 1)
        return relevance

    def _fit_ensemble(X, y, qid=None, sample_weight=None):
        # 用不同随机种子训练多个模型取平均，降低单个模型自身的随机方差——
        # 跟正则化是不同性质的手段（正则化是让单个模型更保守/更有偏，
        # 集成是平均掉每个模型自己的噪声），ensemble_size=1时就是原来的单模型。
        is_ranking = model_objective == "ranking" and model_family != "linear"
        if is_ranking and xgb_rank_objective == "rank:ndcg" and qid is not None:
            y = _to_ndcg_relevance(y, qid)
        if model_objective == "classification" and model_family != "linear" and qid is not None:
            y = _to_topk_label(y, qid, top_frac=classification_top_frac)
        fit_weight = sample_weight
        if is_ranking and sample_weight is not None:
            # XGBRanker 在有 qid 分组时，sample_weight 要求长度等于分组数、
            # 不是样本数（每组一个权重）——新近度权重本来就只跟 decision_date
            # 有关，同一组(同一个qid)内所有样本权重天然相等，取每组第一个
            # 样本的权重收缩成 per-group 数组，数值上跟原来的语义完全等价，
            # 只是满足 XGBoost 的接口要求。qid 要求是连续分组（XGBRanker本身
            # 的硬性要求），用分组边界（值变化处）取第一个样本即可，不依赖
            # qid 的数值大小顺序。
            group_start = np.concatenate(([True], qid[1:] != qid[:-1]))
            fit_weight = sample_weight[group_start]
        models = []
        for k in range(max(1, xgb_ensemble_size)):
            m = _make_model(seed=42 + k)
            fit_kwargs = {} if fit_weight is None else {"sample_weight": fit_weight}
            if is_ranking:
                m.fit(X, y, qid=qid, **fit_kwargs)
            else:
                m.fit(X, y, **fit_kwargs)
            models.append(m)
        return models

    def _ensemble_predict(models, X):
        if model_objective == "classification" and model_family != "linear":
            preds_stack = np.column_stack([m.predict_proba(X)[:, 1] for m in models])
        else:
            preds_stack = np.column_stack([m.predict(X) for m in models])
        return preds_stack.mean(axis=1)

    conformal_q = None
    meta_model = None
    calibrator = None
    if len(X_val) >= 10:
        try:
            val_models = _fit_ensemble(X_tr, y_tr, qid_tr, sample_weight=weight_tr)
            preds = _ensemble_predict(val_models, X_val)
            ic = np.corrcoef(preds, y_val)[0, 1]
            mse = np.mean((preds - y_val) ** 2)
            print(f"  [验证] 训练 {len(X_tr)} / 验证 {len(X_val)} — IC: {ic:.3f}, MSE: {mse:.4f}")
            if ic < 0.02:
                print("  [警告] IC 过低，模型可能无效。")
            if use_ic_quality_gate and ic < ic_quality_threshold:
                print(f"  [入场质量筛选] 本周验证折IC({ic:.3f})低于阈值({ic_quality_threshold})，判定模型本周排序能力不可信，空仓。")
                return []

            # split-conformal 校准：用验证折的残差分布估计给定覆盖率下的误差幅度，
            # 预测阶段用它给每个候选算区间，而不是只信一个点估计。
            # 排序目标输出的是相对打分不是收益率，残差没有可比的物理含义，跳过。
            if use_conformal_filter and model_objective not in ("ranking", "classification"):
                residuals = np.abs(y_val - preds)
                conformal_q = float(np.quantile(residuals, conformal_coverage))

            # 预测值概率校准：用验证折的"模型预测值 -> 实际收益率"这组映射拟合
            # isotonic回归（保序回归，只要求单调，不假设线性关系），修正模型点
            # 估计的系统性偏差（比如预测0.05但实际平均只有0.03这种整体偏移）。
            # 排序目标/分类目标同样跳过，前者没有可比的物理含义，后者的预测值
            # 已经是[0,1]概率、不是收益率，套同一套校准逻辑没有意义。
            if use_prediction_calibration and model_objective not in ("ranking", "classification") and len(np.unique(preds)) >= 5:
                from sklearn.isotonic import IsotonicRegression
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(preds, y_val)

            # meta-labeling：二级分类器，学"如果照这个信号交易，扣完成本后
            # 是否真的会赚钱"。用 val_model（只在训练折上拟合过）在验证折上的
            # 预测 + 原始特征做输入，标签 = 验证折实际收益(已扣成本)是否为正。
            # 注意：训练用的是 val_model 的打分分布，推理时用的是全量模型的
            # 打分分布，两者不完全一致，是这个方案本身的局限——样本量也只有
            # 验证折这一小块，噪声比主模型大，不要把它的判断看得比主模型更可信。
            if use_meta_label:
                meta_X = np.column_stack([X_val, preds])
                meta_y = (y_val > 0).astype(int)
                if len(np.unique(meta_y)) == 2:
                    meta_model = XGBClassifier(
                        n_estimators=60, max_depth=3, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=42, verbosity=0, n_jobs=xgb_n_jobs,
                    )
                    meta_model.fit(meta_X, meta_y)
                else:
                    print("  [提示] meta-label 验证折里只有一种结果（全赚或全亏），跳过二级模型。")
        except Exception as e:
            print(f"  [提示] 验证失败: {e}")

    if use_conformal_filter and model_objective == "ranking":
        print("  [提示] conformal 过滤只对回归目标有意义，排序目标下已跳过。")

    # 全量训练
    if needs_imputation:
        imputer_full = SimpleImputer(strategy="mean")
        X_s = imputer_full.fit_transform(X_s)
    if use_pca_features:
        pca_full = PCA(n_components=pca_variance_threshold, svd_solver="full")
        X_s = pca_full.fit_transform(X_s)
    try:
        models = _fit_ensemble(X_s, y_s, qid_s, sample_weight=weight_s)
    except Exception as e:
        print(f"  [错误] 训练失败: {e}")
        return []

    # 跨模型stacking：XGBoost（树模型）+ ElasticNet（线性模型）各自独立训练，
    # 预测时按简单平均融合两者打分——两个模型族的误差更可能不相关，融合才
    # 有意义（区别于之前测过的"同模型换随机种子"ensemble，那个已经验证过
    # 没用）。跟 model_family="linear" 不同：那个是替换掉XGBoost，这个是
    # XGBoost继续用，额外多训一个线性模型混进来。
    stacking_linear_model = None
    if use_stacking and model_objective not in ("ranking", "classification"):
        try:
            from sklearn.linear_model import ElasticNet
            stacking_linear_model = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=5000)
            fit_kwargs = {} if weight_s is None else {"sample_weight": weight_s}
            stacking_linear_model.fit(X_s, y_s, **fit_kwargs)
        except Exception as e:
            print(f"  [提示] stacking线性子模型训练失败，跳过: {e}")
            stacking_linear_model = None

    # === SHAP 分析 ===
    # TreeExplainer 只能解释单个模型，集成时用第一个成员代表整体做特征重要性
    # 展示——这只是诊断用途（默认关闭、只是偶尔画一张图），不影响实际预测。
    if shap_enabled and len(X_s) > 100 and not use_pca_features and model_family != "linear" and not use_shap_pruning:
        try:
            feature_names = FEATURE_NAMES + FUNDAMENTAL_FEATURE_NAMES if use_fundamental_features else FEATURE_NAMES
            explainer = shap.TreeExplainer(models[0])
            shap_sample_size = min(2000, len(X_val))
            shap_idx = np.random.choice(len(X_val), shap_sample_size, replace=False)
            shap_values = explainer.shap_values(X_val[shap_idx])

            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_val[shap_idx], feature_names=feature_names,
                              show=False, plot_size=(10, 8))
            plt.tight_layout()
            plt.savefig(shap_output_path, dpi=150)
            plt.close()
            print(f"  [SHAP] 特征重要性图已保存: {shap_output_path}")

            mean_shap = np.mean(np.abs(shap_values), axis=0)
            ranked_features = sorted(zip(feature_names, mean_shap), key=lambda x: x[1], reverse=True)
            print(f"  [SHAP] 特征重要性完整排名（{len(ranked_features)}个特征）:")
            for name, importance in ranked_features:
                print(f"         {name:20s}: {importance:.4f}")
        except Exception as e:
            print(f"  [SHAP] 分析失败: {e}")

    # ---- 预测当日候选 ----
    # mom_6m 需要至少126个交易日的历史，原来固定 tail(60) 会导致开启
    # use_multi_horizon_momentum 时预测阶段永远算不出来（训练阶段用的是
    # 滚动窗口内的全部历史，不受这个60天限制，两边不能不一致）。
    pred_lookback_days = 130 if use_multi_horizon_momentum else 60
    all_raw = {}
    for ticker in close_df.columns:
        try:
            hist_c = close_df[ticker].loc[:target_date].dropna().tail(pred_lookback_days)
            hist_o = open_df[ticker].loc[:target_date].dropna().tail(pred_lookback_days)
            hist_h = high_df[ticker].loc[:target_date].dropna().tail(pred_lookback_days)
            hist_l = low_df[ticker].loc[:target_date].dropna().tail(pred_lookback_days)
            hist_v = volume_df[ticker].loc[:target_date].dropna().tail(pred_lookback_days)
            if len(hist_c) < 60:
                continue
            if use_stage1_filter:
                # 两阶段架构第一阶段：粗筛掉明显不健康的候选（跌破近期均线，
                # 说明趋势已经转弱），只用简单规则、不用模型，成本几乎为0，
                # 让第二阶段的模型只在"看起来还在趋势里"的候选池里做精选，
                # 而不是在全市场（含大量明显走弱的票）里矮子里拔将军。
                if len(hist_c) < stage1_ma_window or hist_c.iloc[-1] <= hist_c.tail(stage1_ma_window).mean():
                    continue
            sector = sector_map.get(ticker, "TECH")
            feat = build_raw_features(
                ticker, hist_c, hist_o, hist_h, hist_l, hist_v,
                vix_series, sector, close_df, target_date,
                market_cache=market_cache, regime_mode=regime_mode,
                use_fundamental_features=use_fundamental_features,
                rsi_overbought_threshold=rsi_overbought_threshold,
                use_multi_horizon_momentum=use_multi_horizon_momentum,
                use_volume_zscore=use_volume_zscore,
                macro_df=macro_df, use_macro_features=use_macro_features,
                use_calendar_features=use_calendar_features,
            )
            if feat is None:
                continue
            all_raw[ticker] = feat
        except Exception as e:
            print(f"  [警告] {ticker} 候选特征计算失败，跳过：{type(e).__name__}: {e}")
            continue

    if not all_raw:
        return []

    # 预测日截面排名（深拷贝，不修改原始字典）
    all_raw_ranked = _add_cross_sectional_ranks(
        all_raw, extra_features=EXTRA_RANK_FEATURES if use_full_cross_sectional_standardization else None)
    if use_sector_relative_momentum:
        all_raw_ranked = _add_sector_relative_momentum(all_raw_ranked)

    candidates = []
    for ticker, feat in all_raw_ranked.items():
        features = np.array([_extract_feature_vector(feat, excluded_features=excluded_features)])
        if imputer_full is not None:
            features = imputer_full.transform(features)
        if use_pca_features and pca_full is not None:
            features = pca_full.transform(features)
        pred_ret = _ensemble_predict(models, features)[0]
        if use_stacking and stacking_linear_model is not None:
            linear_pred = stacking_linear_model.predict(features)[0]
            pred_ret = 0.5 * pred_ret + 0.5 * linear_pred
        if use_prediction_calibration and calibrator is not None:
            pred_ret = float(calibrator.predict([pred_ret])[0])
        if use_vol_adjusted_target:
            # 模型预测的是波动率调整后收益，换算回原始收益率尺度，
            # 下游仓位计算、日志展示用的 pred_ret 语义保持不变。
            pred_ret = pred_ret * (feat["realized_vol_20d"] + 1e-4)
        candidate = {
            "ticker": ticker,
            "pred_ret": pred_ret,
            "atr": feat["atr"],
            "sector": feat["sector"],
        }
        if meta_model is not None:
            meta_features = np.concatenate([features[0], [pred_ret]]).reshape(1, -1)
            candidate["meta_prob"] = float(meta_model.predict_proba(meta_features)[0, 1])
        candidates.append(candidate)

    candidates.sort(key=lambda x: x["pred_ret"], reverse=True)

    if _return_raw_candidates:
        return candidates

    if use_multi_window_fusion:
        fusion_kwargs = dict(_all_args)
        fusion_kwargs["rolling_window_days"] = fusion_window_days
        fusion_kwargs["use_multi_window_fusion"] = False
        fusion_kwargs["_return_raw_candidates"] = True
        # 主调用已经过了熔断检查（否则这里根本跑不到），不用在长窗口子调用
        # 里再查一遍——同一个 target_date 结果必然一样，白算一次。
        fusion_kwargs["enable_macro_filter"] = False
        fusion_candidates = get_ai_weekly_picks(**fusion_kwargs)
        fusion_pred = {c["ticker"]: c["pred_ret"] for c in fusion_candidates}
        for c in candidates:
            if c["ticker"] in fusion_pred:
                c["pred_ret"] = 0.5 * c["pred_ret"] + 0.5 * fusion_pred[c["ticker"]]
        candidates.sort(key=lambda x: x["pred_ret"], reverse=True)

    # conformal 置信过滤：只留下"就算按验证折的误差幅度往坏了算，预测收益率
    # 依然是正的"这类候选，而不是只信模型给的点估计。
    if conformal_q is not None:
        before_n = len(candidates)
        candidates = [c for c in candidates if c["pred_ret"] - conformal_q > 0]
        print(f"  [conformal] 覆盖率{conformal_coverage:.0%}下误差幅度±{conformal_q:.4f}，"
              f"候选从{before_n}只过滤到{len(candidates)}只。")

    # meta-label 过滤：二级模型觉得"扣完成本后不太可能真的赚钱"的候选，剔除掉。
    if meta_model is not None:
        before_n = len(candidates)
        candidates = [c for c in candidates if c["meta_prob"] > meta_label_threshold]
        print(f"  [meta-label] 阈值{meta_label_threshold:.2f}下，"
              f"候选从{before_n}只过滤到{len(candidates)}只。")

    # 财报避让：持仓周（target_date后1~horizon个交易日）内有财报的候选剔除，
    # 避免动量策略持仓周里撞上财报变成赌财报大小年。
    if enable_earnings_blackout and earnings_calendar:
        before_n = len(candidates)
        candidates = [
            c for c in candidates
            if not _has_upcoming_earnings(c["ticker"], target_date, close_df.index,
                                          earnings_calendar, earnings_horizon)
        ]
        if before_n != len(candidates):
            print(f"  [财报避让] 剔除 {before_n - len(candidates)} 只持仓周内有财报的候选。")

    # 动态持仓数量：候选质量分布决定选几只，而不是每周都固定填满 max_pos。
    # 只做"往少了选"（低于候选池中位数的直接剔除），不会超过 max_pos 上限，
    # 跟 pos_size（调仓位轻重）是两回事，这个调的是选几只。
    if use_dynamic_position_count and candidates:
        median_pred = float(np.median([c["pred_ret"] for c in candidates]))
        before_n = len(candidates)
        candidates = [c for c in candidates if c["pred_ret"] >= median_pred]
        print(f"  [动态持仓] 预测收益低于候选池中位数的候选剔除，{before_n}只筛到{len(candidates)}只。")

    # 行业集中度过滤 + 候选相关性去重：即使不同行业，也可能选出两只实际走势
    # 高度相关的票（比如同一条产业链上下游），持仓分散度名义上够但实际上不够。
    # 用最近60个交易日的日收益率相关系数，跳过跟已选中候选相关性超过阈值的候选。
    corr_matrix = None
    if use_correlation_dedup and candidates:
        try:
            tickers_pool = [c["ticker"] for c in candidates]
            hist = close_df[tickers_pool].loc[:target_date].tail(61)
            rets = hist.pct_change().dropna()
            if len(rets) >= 20:
                corr_matrix = rets.corr()
        except Exception as e:
            print(f"  [警告] 候选相关性计算失败，跳过去重：{type(e).__name__}: {e}")

    selected = []
    sector_counts = {}
    for c in candidates:
        sec = c["sector"]
        if sector_counts.get(sec, 0) >= max_sector:
            continue
        if corr_matrix is not None and selected:
            t = c["ticker"]
            if t in corr_matrix.columns:
                too_correlated = any(
                    s["ticker"] in corr_matrix.columns and corr_matrix.loc[t, s["ticker"]] > correlation_threshold
                    for s in selected
                )
                if too_correlated:
                    continue
        selected.append(c)
        sector_counts[sec] = sector_counts.get(sec, 0) + 1
        if len(selected) >= max_pos:
            break

    if not selected:
        return []

    # 风险平价权重
    inv_atrs = [1.0 / (s["atr"] + 1e-8) for s in selected]
    total_inv = sum(inv_atrs)
    weights = [ia / total_inv for ia in inv_atrs]

    # 动态仓位
    pred_rets = [c["pred_ret"] for c in selected]
    max_pred = max(pred_rets) if pred_rets else 0

    # 排序目标输出的是相对打分，不是预期收益率，没有"0=盈亏平衡"这个语义，
    # 不能直接套回归模式那套 max(0, pred_ret) 的公式——这里改成在选中的几只
    # 票里做 min-max 归一化，取相对强弱作为仓位驱动。
    score_min = min(pred_rets) if pred_rets else 0
    score_range = (max_pred - score_min) if pred_rets else 0

    pos_size_range = pos_size_ceiling - pos_size_floor
    result = []
    for s, w in zip(selected, weights):
        if model_objective in ("ranking", "classification"):
            confidence = (s["pred_ret"] - score_min) / score_range if score_range > 1e-8 else 1.0
            pos_size = pos_size_floor + pos_size_range * confidence
        elif max_pred > 0:
            pos_size = pos_size_floor + pos_size_range * max(0, s["pred_ret"]) / max_pred
        else:
            pos_size = pos_size_floor
        pos_size = min(pos_size, pos_size_ceiling)
        result.append((s["ticker"], w, s["pred_ret"], pos_size))

    return result