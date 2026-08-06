"""
增强型指数化 (Enhanced Indexing) 引擎——框架级大修大补(第504条)。

跟现有"挑几十只票、自定义权重、脱离基准"的因子押注模式完全不同的一条
新范式：标普500市值权重打底 + 现有因子分数(价值/低波动/动量)做小幅超配
减配 + 把跟踪误差控制在100-200bp。文献来源：Cremers & Petajisto (2009,
RFS)、Petajisto (2013, FAJ)——"多元化选股者"(enhanced indexing)扣费后
年化跑赢基准约1.26%，是文献里有实证支持的象限；而当前系统的持仓模式接近
文献里表现最差的"因子押注者"象限。

如实说明这个引擎要验证的核心问题(用户已明确预期)：能把beta从0.313拉到
接近1、上涨周胜率会上升，但这是拿82.5%的下跌周防御优势去换的，不是净
增益——综合胜率大概率升到50%出头，不会显著超过50%。本模块只负责把
"市值权重+因子tilt+跟踪误差约束"这套机制忠实实现出来，结论由
run_enhanced_indexing_test.py 如实验证、如实记录，不预设立场。

数据前提(已验证)：市值权重 = 价格 × SEC披露的 shares_outstanding(用
filed日期做点位时间回溯，无前视偏差)。只有 2022-2026 段(~87-88%的票
有点位时间正确的shares)能构建市值权重基准；2009-2014 段 SEC 数据覆盖
不到、2014-2020 段价格缓存已丢失，本引擎如实标注为数据受限，不做降级
代测(等权基准不是增强型指数化)。

【shares数据质量护栏】实测发现 sec_fundamentals_cache.json 里有单条
shares 记录比同票正常值大~1000倍(如PKG 2023-04-28记录 89932185000 vs
真实约8990万)的坏数据，会直接毁掉市值权重。_robust_shares 用"同票最近
4年披露值的median"做异常值检测：偏离median 3倍以上时回退用median本身
(而不是整只剔除——剔除大票会系统性扭曲基准)。这个护栏只影响本引擎；
现有 value 腿用 shares 算 P/E/P/B，坏值会让该票被screen掉，是可生存的
错误，不在本任务范围内修(如实标注，见第504条)。

机制说明：
- 市值权重基准：每周调仓(周一收盘建仓、持有到周五收盘、下周一再平衡)，
  权重=点位时间正确的市值权重。基准与组合用同一套机制，跟踪误差在同一
  口径下度量，跟SPY可比(SPY是外部参照，不是基准)。
- 因子分数：价值分(PE/PB/ROE综合排名，复用 us_stock_screener 的点位
  时间正确基本面)、低波动分(trailing 60日已实现波动率)、动量分(trailing
  126日价格动量，经典UMD；不用XGBoost pred_ret——该信号已证近零信息且
  需~30分钟重训)。三个分数各做横截面z-score，composite = 三z的nanmean。
- 跟踪误差优化：默认用"TE标定tilt"(一维二分λ命中目标TE)——确定性、
  每周精确命中目标、对500只票数值稳定；SLSQP约束优化(method="slsqp")
  保留为可选，但实测在500变量、日收益方差量级(1e-4)下数值不稳定(部分
  周会失败退回tilt)，不是默认。两种都是文献里标准的增强指数构造，结果
  如实标注用的是哪种。
"""
from __future__ import annotations

import bisect

import numpy as np
import pandas as pd

from backtest import get_weekly_trading_dates
from us_stock_screener import _compute_fundamental_features, _load_sec_fundamentals
from value_investing import _shrunk_cov

# 数据质量护栏(沿用 run_market_cap_stratification.py)：shares 明显错误值跳过、
# 披露日期超过2年视为过期不可信。
_MIN_SHARES = 1000
_MAX_SHARES_STALE_DAYS = 730
_SHARES_MEDIAN_WINDOW_DAYS = 1460  # median参考窗口：最近4年披露值
_SHARES_OUTLIER_FACTOR = 3.0       # 偏离median超过3倍视为异常值


def _robust_shares(ticker: str, decision_date: pd.Timestamp) -> tuple[float | None, pd.Timestamp | None]:
    """点位时间正确的 shares_outstanding，带同票median异常值护栏。
    SEC缓存里 filed <= decision_date 的最新一条；如果该值偏离同票最近
    4年披露值的median超过3倍(实测有~1000倍膨胀的坏数据，如PKG)，回退
    用median本身。返回 (shares, filed_date)。"""
    cache = _load_sec_fundamentals()
    ticker_data = cache.get(ticker)
    if not ticker_data or "shares_outstanding" not in ticker_data:
        return None, None
    field = ticker_data["shares_outstanding"]
    ds = decision_date.strftime("%Y-%m-%d")
    idx = bisect.bisect_right(field["filed_dates"], ds) - 1
    if idx < 0:
        return None, None
    val = field["values"][idx]
    filed = field["filed_dates"][idx]
    if val is None:
        return None, None
    cutoff = (decision_date - pd.Timedelta(days=_SHARES_MEDIAN_WINDOW_DAYS)).strftime("%Y-%m-%d")
    pool = [v for f, v in zip(field["filed_dates"][:idx + 1], field["values"][:idx + 1])
            if v is not None and v > 0 and f >= cutoff]
    if not pool:
        pool = [v for v in field["values"][:idx + 1] if v is not None and v > 0]
    if not pool:
        return None, None
    med = float(np.median(pool))
    if med <= 0:
        return None, None
    val_f = float(val)
    if val_f > _SHARES_OUTLIER_FACTOR * med or val_f < med / _SHARES_OUTLIER_FACTOR:
        val_f = med
    return val_f, pd.Timestamp(filed)


def compute_cap_weights(price_df: pd.DataFrame, valid_tickers: list, decision_date: pd.Timestamp) -> pd.Series:
    """市值权重 = 价格 × 点位时间正确的shares，归一化。返回覆盖全集索引的
    Series(无shares的票记0)。沿用市场分层脚本的数据质量护栏：shares<=1000
    明显错误值跳过、披露日期距今超过730天视为过期不可信、单值偏离median
    3倍以上回退median(见_robust_shares)。"""
    mcs: dict[str, float] = {}
    for t in valid_tickers:
        if t not in price_df.columns:
            continue
        px_series = price_df[t].loc[:decision_date].dropna()
        if len(px_series) == 0:
            continue
        price = float(px_series.iloc[-1])
        shares, filed = _robust_shares(t, decision_date)
        if shares is None or shares <= _MIN_SHARES or price <= 0:
            continue
        if filed is not None and (decision_date - filed).days > _MAX_SHARES_STALE_DAYS:
            continue
        mcs[t] = price * shares
    weights = pd.Series(0.0, index=valid_tickers)
    if mcs:
        total = sum(mcs.values())
        for t, mc in mcs.items():
            weights[t] = mc / total
    return weights


def compute_factor_scores(price_df: pd.DataFrame, valid_tickers: list, decision_date: pd.Timestamp,
                          vol_window: int = 60, mom_window: int = 126) -> pd.DataFrame:
    """三个现有因子分数 + 综合分。返回 DataFrame(index=ticker)：
    value(综合排名分，越小越便宜，缺基本面为NaN)、low_vol(已实现波动率)、
    mom(trailing价格动量)、value_z/low_vol_z/mom_z(横截面z，越大越好)、
    composite(三z的nanmean，越大越该超配)。"""
    px = price_df.loc[:decision_date]
    rows: dict[str, dict] = {}
    value_pool = []
    for t in valid_tickers:
        if t not in price_df.columns:
            continue
        s = px[t].dropna()
        if len(s) < 3:
            continue
        row: dict = {}
        curr_price = float(s.iloc[-1])
        # --- 价值分：复用点位时间正确的基本面，沿用 select_value_portfolio 的
        # screen(0<pe<40, pb>0, roe非nan)和综合分(pe升+pb升+roe降的排名均)。
        feat = _compute_fundamental_features(t, decision_date, curr_price)
        pe, pb, roe = feat["pe_ratio"], feat["pb_ratio"], feat["roe"]
        if (pe is not None and not np.isnan(pe) and 0 < pe < 40 and
                pb is not None and not np.isnan(pb) and pb > 0 and
                roe is not None and not np.isnan(roe)):
            row["pe"] = pe
            row["pb"] = pb
            row["roe"] = roe
            value_pool.append(t)
        # --- 低波动分：trailing已实现波动率
        if len(s) >= vol_window + 1:
            row["low_vol"] = float(s.tail(vol_window + 1).pct_change().std())
        # --- 动量分：trailing价格动量(经典UMD，126日)
        if len(s) >= mom_window + 1:
            row["mom"] = float(s.iloc[-1] / s.iloc[-1 - mom_window] - 1)
        rows[t] = row

    df = pd.DataFrame.from_dict(rows, orient="index")
    if df.empty:
        return pd.DataFrame(columns=["value", "low_vol", "mom", "value_z", "low_vol_z", "mom_z", "composite"],
                            index=valid_tickers)
    # 价值综合排名(越小越便宜)
    if value_pool:
        v = df.loc[value_pool]
        df.loc[value_pool, "value"] = (
            v["pe"].rank(ascending=True) + v["pb"].rank(ascending=True) + v["roe"].rank(ascending=False)
        ) / 3
    # 横截面z，符号对齐为"越大越好"
    for col, better_when in (("value", -1), ("low_vol", -1), ("mom", 1)):
        if col not in df.columns:
            continue
        sub = df[col].dropna()
        if len(sub) >= 3:
            mu, sd = sub.mean(), sub.std()
            if sd > 0:
                df.loc[sub.index, f"{col}_z"] = better_when * (sub - mu) / sd
    df["composite"] = df.reindex(columns=["value_z", "low_vol_z", "mom_z"]).mean(axis=1, skipna=True)
    return df


def _weekly_daily_cov(price_df: pd.DataFrame, tickers: list, decision_date: pd.Timestamp,
                      window: int = 60) -> np.ndarray | None:
    """trailing window日收益的Ledoit-Wolf收缩协方差(日频，TE按252天年化)。
    覆盖不全的列剔除；少于5列返回None。"""
    cols = [t for t in tickers if t in price_df.columns]
    if len(cols) < 5:
        return None
    hist = price_df[cols].loc[:decision_date].tail(window + 1)
    rets = hist.pct_change().dropna()
    valid_cols = [c for c in rets.columns if rets[c].notna().all()]
    if len(valid_cols) < 5:
        return None
    try:
        return _shrunk_cov(rets[valid_cols])
    except Exception:
        return None


def _te_annual(w: np.ndarray, cap: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(np.clip((w - cap) @ cov @ (w - cap), 0.0, None))) * np.sqrt(252.0)


def _calibrated_tilt_weights(cap_w: np.ndarray, z: np.ndarray, cov: np.ndarray,
                             te_target_annual: float) -> np.ndarray:
    """TE标定tilt：w(λ) = clip(cap·(1+λz),0)再归一化，二分λ使年化跟踪误差
    命中目标。TE随λ单调，二分收敛快、确定性好、每周都精确命中目标。"""
    lam_lo, lam_hi = 0.0, 10.0
    te_const = te_target_annual / np.sqrt(252.0)

    def te_of(lam: float) -> float:
        w = np.clip(cap_w * (1.0 + lam * z), 0.0, None)
        s = w.sum()
        if s <= 0:
            return 0.0
        return float(np.sqrt(np.clip((w / s - cap_w) @ cov @ (w / s - cap_w), 0.0, None)))

    if te_of(lam_hi) < te_const:
        lam = lam_hi
    else:
        for _ in range(60):
            mid = 0.5 * (lam_lo + lam_hi)
            if te_of(mid) < te_const:
                lam_lo = mid
            else:
                lam_hi = mid
        lam = lam_lo
    w = np.clip(cap_w * (1.0 + lam * z), 0.0, None)
    s = w.sum()
    return w / s if s > 0 else cap_w


def _slsqp_tilt_weights(cap_w: np.ndarray, z: np.ndarray, cov: np.ndarray,
                        te_target_annual: float, active_cap: float | None = None) -> np.ndarray | None:
    """SLSQP约束优化：min −Σw·z，约束 Σw=1、w≥0、(w−cap)'Σ(w−cap) ≤
    te²/252(协方差日频、TE年化)，可选每票主动权重上限 |w−cap| ≤ active_cap。
    目标/约束梯度解析给出；协方差先按对角线均值缩放，避免约束残差在日收益
    方差量级(1e-4~1e-3)下被SLSQP默认容差误判为"已收敛"(第155条同类坑)。
    失败/实现TE超目标20%时返回None(调用方退回tilt)。"""
    n = len(cap_w)
    if n < 5:
        return None
    scale = float(np.mean(np.diag(cov))) or 1e-4
    cov_n = cov / scale
    rhs = (te_target_annual ** 2 / 252.0) / scale
    cap_n = cap_w.astype(float)
    z_n = z.astype(float)

    def obj(w):
        return -float(z_n @ w)

    def obj_jac(w):
        return -z_n

    def te_ineq(w):
        diff = w - cap_n
        return rhs - float(diff @ cov_n @ diff)

    def te_ineq_jac(w):
        return -2.0 * cov_n @ (w - cap_n)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                    "jac": lambda w: np.ones(n)},
                   {"type": "ineq", "fun": te_ineq, "jac": te_ineq_jac}]
    if active_cap is not None:
        for i in range(n):
            constraints.append({"type": "ineq",
                                "fun": lambda w, i=i: active_cap - abs(w[i] - cap_n[i]),
                                "jac": lambda w, i=i: _active_cap_jac(n, i, w, cap_n)})

    from scipy.optimize import minimize
    bounds = [(0.0, 1.0)] * n
    result = minimize(obj, cap_n, method="SLSQP", jac=obj_jac, bounds=bounds,
                      constraints=constraints, options={"maxiter": 1000, "ftol": 1e-12})
    if not result.success or np.any(np.isnan(result.x)):
        return None
    w = np.clip(result.x, 0.0, None)
    s = w.sum()
    if s <= 0:
        return None
    w = w / s
    if _te_annual(w, cap_n, cov) <= te_target_annual * 1.2:
        return w
    return None


def _active_cap_jac(n: int, i: int, w: np.ndarray, cap_n: np.ndarray) -> np.ndarray:
    j = np.zeros(n)
    j[i] = -np.sign(w[i] - cap_n[i])
    return j


def _tilt_weights(cap_w: np.ndarray, z: np.ndarray, cov: np.ndarray,
                  te_target_annual: float, method: str,
                  active_cap: float | None = None) -> tuple[np.ndarray, str]:
    """统一入口：按method选优化器，失败退回TE标定tilt。返回 (weights, method_used)。"""
    if method == "slsqp":
        w = _slsqp_tilt_weights(cap_w, z, cov, te_target_annual, active_cap=active_cap)
        if w is not None:
            return w, "slsqp"
        return _calibrated_tilt_weights(cap_w, z, cov, te_target_annual), "tilt"
    return _calibrated_tilt_weights(cap_w, z, cov, te_target_annual), "tilt"


def _week_task(args: dict) -> dict:
    """ProcessPoolExecutor worker任务：算某周的组合权重。多个te_target共享
    cap_w/scores/cov，只把优化器按te分别跑。"""
    price_df = _W_PRICE_DF
    membership = _W_MEMBERSHIP
    monday, friday, decision_date, te_targets, vol_window, mom_window, active_cap, method = (
        args["monday"], args["friday"], args["decision_date"],
        args["te_targets"], args["vol_window"], args["mom_window"],
        args["active_cap"], args["method"])

    date_str = decision_date.strftime("%Y-%m-%d")
    valid = membership.get(date_str)
    if valid is None:
        keys = sorted(membership.keys())
        valid = membership[max([k for k in keys if k <= date_str], default=keys[0])]
    valid = [t for t in valid if t in price_df.columns]

    cap_w = compute_cap_weights(price_df, valid, decision_date)
    covered = list(cap_w[cap_w > 0].index)
    if len(covered) < 10:
        return {"monday": monday, "friday": friday, "cap_w": cap_w.to_dict(),
                "weights": {te: cap_w.to_dict() for te in te_targets}, "n_covered": len(covered)}

    scores = compute_factor_scores(price_df, covered, decision_date, vol_window=vol_window,
                                   mom_window=mom_window)
    tiltable = scores["composite"].dropna()
    if len(tiltable) < 10:
        return {"monday": monday, "friday": friday, "cap_w": cap_w.to_dict(),
                "weights": {te: cap_w.to_dict() for te in te_targets}, "n_covered": len(covered)}

    cov = _weekly_daily_cov(price_df, list(tiltable.index), decision_date)
    if cov is None:
        return {"monday": monday, "friday": friday, "cap_w": cap_w.to_dict(),
                "weights": {te: cap_w.to_dict() for te in te_targets}, "n_covered": len(covered)}

    cap_sub = cap_w.loc[tiltable.index].values
    z_sub = tiltable.values.astype(float)
    out_w = {}
    methods_used = {}
    for te in te_targets:
        w_sub, method_used = _tilt_weights(cap_sub, z_sub, cov, te, method=method,
                                           active_cap=active_cap)
        methods_used[te] = method_used
        # 全universe权重：tiltable用优化权重，非tiltable保持cap权重不动
        full = cap_w.copy()
        full.loc[tiltable.index] = w_sub
        s = full.sum()
        if s > 0:
            full = full / s
        out_w[te] = full.to_dict()

    return {"monday": monday, "friday": friday, "cap_w": cap_w.to_dict(),
            "weights": out_w, "n_covered": len(covered), "methods_used": methods_used}


# worker全局变量(仿 backtest.py 的 _init_worker 模式，避免每次任务都pickle大DataFrame)
_W_PRICE_DF: pd.DataFrame | None = None
_W_MEMBERSHIP: dict | None = None


def _init_worker(price_df: pd.DataFrame, membership: dict):
    global _W_PRICE_DF, _W_MEMBERSHIP
    _W_PRICE_DF = price_df
    _W_MEMBERSHIP = membership


def run_enhanced_indexing_multi(price_df: pd.DataFrame, membership: dict, spy_close: pd.Series | None,
                                te_targets: list[float] | None = None, cost_bi: float = 0.003,
                                vol_window: int = 60, mom_window: int = 126,
                                active_cap: float | None = None, method: str = "tilt",
                                n_workers: int = 8) -> dict[float, dict]:
    """增强型指数化回测：周频调仓(周一收盘建仓、持有到周五收盘、下周一再
    平衡，与系统节奏一致且连续持仓，跟SPY可比)。市值权重打底 + 因子tilt，
    跟踪误差约束在 te_targets(年化，每个target一份报告)。多个te_target共享
    每周的cap_w/scores/cov，只重跑优化器。

    te_targets: 年化跟踪误差目标列表(0.015 = 150bp)。
    cost_bi: 调仓时对换手部分收的双边成本(沿用value_investing口径)。
    method: "tilt"(默认，TE标定tilt，稳健)或"slsqp"(约束优化，数值不稳定时
    自动退回tilt)。
    """
    index = price_df.index
    week_pairs = get_weekly_trading_dates(index)
    te_targets = te_targets or [0.015]

    tasks = []
    for monday, friday in week_pairs:
        loc = index.get_loc(monday)
        if loc == 0:
            continue
        decision_date = index[loc - 1]
        tasks.append({"monday": monday, "friday": friday, "decision_date": decision_date,
                      "te_targets": te_targets, "vol_window": vol_window, "mom_window": mom_window,
                      "active_cap": active_cap, "method": method})

    if n_workers and n_workers > 1 and len(tasks) > 1:
        try:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                                     initargs=(price_df, membership)) as ex:
                week_results = list(ex.map(_week_task, tasks))
        except Exception as e:
            # 某些受限环境(沙箱/容器)下ProcessPoolExecutor起不来，退回串行
            print(f"  [警告] ProcessPoolExecutor不可用({type(e).__name__}: {e})，退回串行")
            _init_worker(price_df, membership)
            week_results = [_week_task(t) for t in tasks]
    else:
        _init_worker(price_df, membership)
        week_results = [_week_task(t) for t in tasks]

    return _assemble_reports(price_df, spy_close, week_results, te_targets, cost_bi)


def run_enhanced_indexing(price_df: pd.DataFrame, membership: dict, spy_close: pd.Series | None,
                          te_target: float = 0.015, cost_bi: float = 0.003,
                          vol_window: int = 60, mom_window: int = 126,
                          active_cap: float | None = None, method: str = "tilt",
                          n_workers: int = 8) -> dict:
    """单target便捷入口，返回一份报告dict(见 run_enhanced_indexing_multi)。"""
    reports = run_enhanced_indexing_multi(price_df, membership, spy_close,
                                          te_targets=[te_target], cost_bi=cost_bi,
                                          vol_window=vol_window, mom_window=mom_window,
                                          active_cap=active_cap, method=method,
                                          n_workers=n_workers)
    return reports[te_target]


def _assemble_reports(price_df: pd.DataFrame, spy_close: pd.Series | None,
                      week_results: list[dict], te_targets: list[float],
                      cost_bi: float) -> dict[float, dict]:
    """把每周权重拼成各te_target的日净值路径 + 周收益 + 指标。组合与基准
    同一套机制：周一收盘建仓(周一=调仓日只扣换手成本，无价格收益)，周二~
    周五 close/close，权重周内固定。周收益从日nav在每周最后交易日采样。"""
    index = price_df.index
    fridays = [r["friday"] for r in week_results]
    reports = {}
    for te in te_targets:
        day_ret_p: dict[pd.Timestamp, float] = {}
        day_ret_b: dict[pd.Timestamp, float] = {}
        active_shares = []
        turnover_weeks = []
        methods_used = set()
        prev_w: pd.Series | None = None
        prev_wb: pd.Series | None = None
        prev_friday = None

        for r in week_results:
            monday, friday = r["monday"], r["friday"]
            w = pd.Series(r["weights"][te])
            wb = pd.Series(r["cap_w"])
            methods_used.add(r.get("methods_used", {}).get(te, "n/a"))

            # 换手成本(口径同value_investing：首次建仓全换手，之后只对换手部分收)
            if prev_w is not None and len(w) > 0:
                turn = sum(abs(w.get(t, 0) - prev_w.get(t, 0)) for t in set(w.index) | set(prev_w.index)) / 2
            else:
                turn = 1.0 if len(w) > 0 else 0.0
            turn_b = (0.0 if prev_wb is None else
                      sum(abs(wb.get(t, 0) - prev_wb.get(t, 0)) for t in set(wb.index) | set(prev_wb.index)) / 2) if len(wb) > 0 else 0.0
            active_shares.append(float((w - wb).abs().sum() / 2))
            turnover_weeks.append(turn)

            idx = index[(index >= monday) & (index <= friday)]
            # 周一：用上周权重记录"上周五收盘→本周一收盘"的价格收益(标准周度
            # 再平衡口径，保证日净值覆盖全部5个交易日、跟SPY逐日可比)；首周
            # 无上周权重，周一只有建仓成本。再平衡换手成本也计提在周一。
            if prev_w is not None:
                day_ret_p[monday] = day_ret_p.get(monday, 0.0) + _daily_weighted_ret(price_df, prev_w, prev_friday, monday)
            if prev_wb is not None:
                day_ret_b[monday] = day_ret_b.get(monday, 0.0) + _daily_weighted_ret(price_df, prev_wb, prev_friday, monday)
            day_ret_p[monday] = day_ret_p.get(monday, 0.0) - turn * cost_bi
            day_ret_b[monday] = day_ret_b.get(monday, 0.0) - turn_b * cost_bi
            # 周二~周五：本周新权重 close/close
            for d0, d1 in zip(idx, idx[1:]):
                day_ret_p[d1] = day_ret_p.get(d1, 0.0) + _daily_weighted_ret(price_df, w, d0, d1)
                day_ret_b[d1] = day_ret_b.get(d1, 0.0) + _daily_weighted_ret(price_df, wb, d0, d1)
            prev_w, prev_wb = w, wb
            prev_friday = friday

        dr = pd.Series(day_ret_p).sort_index()
        db = pd.Series(day_ret_b).sort_index()
        port_nav = (1.0 + dr.fillna(0)).cumprod()
        bench_nav = (1.0 + db.fillna(0)).cumprod()

        # 周收益：从日nav在每周最后交易日(周五)采样，与SPY周收益可比
        p_w = port_nav.reindex(fridays).dropna()
        b_w = bench_nav.reindex(fridays).dropna()
        wr = p_w.pct_change().dropna()
        br = b_w.pct_change().dropna()

        n_years = len(port_nav) / 252
        cagr = float(port_nav.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else None
        daily_rets = port_nav.pct_change().dropna()
        sharpe = float(daily_rets.mean() / (daily_rets.std() + 1e-9) * np.sqrt(252))
        mdd = _true_max_drawdown(port_nav.values)

        excess = wr - br
        te_annual = float(excess.std() * np.sqrt(52)) if len(excess) > 2 else None
        ir = float(excess.mean() / (excess.std() + 1e-9) * np.sqrt(52)) if len(excess) > 2 else None

        # SPY周收益用同一口径：周五采样、周五→周五(跟wr/br基期一致)
        spy_wr = pd.Series(dtype=float)
        if spy_close is not None:
            spy_wr = spy_close.reindex(fridays).ffill().dropna().pct_change().dropna()
        beta_spy = None
        if len(spy_wr) > 2 and len(wr) > 2:
            aligned = pd.concat([wr, spy_wr], axis=1, keys=["port", "spy"]).dropna()
            if len(aligned) > 5:
                beta_spy = float(np.cov(aligned["port"], aligned["spy"])[0, 1] / np.var(aligned["spy"]))
        up_win = _win_rate(wr, spy_wr, up_only=True)
        down_win = _win_rate(wr, spy_wr, up_only=False)
        overall_win = _win_rate(wr, spy_wr, up_only=None)

        spy_cagr = None
        if spy_close is not None:
            spy_aligned = spy_close.reindex(port_nav.index).ffill().dropna()
            if len(spy_aligned) > 1:
                spy_cagr = float((spy_aligned.iloc[-1] / spy_aligned.iloc[0]) ** (1 / n_years) - 1)

        reports[te] = {
            "nav": port_nav,
            "benchmark_nav": bench_nav,
            "weekly_port": wr,
            "weekly_bench": br,
            "weekly_spy": spy_wr,
            "CAGR": cagr,
            "Sharpe": sharpe,
            "MaxDD": mdd,
            "SPY_CAGR": spy_cagr,
            "te_annual": te_annual,
            "ir": ir,
            "beta_spy": beta_spy,
            "up_win": up_win,
            "down_win": down_win,
            "overall_win": overall_win,
            "active_share": float(np.mean(active_shares)) if active_shares else None,
            "avg_turnover": float(np.mean(turnover_weeks)) if turnover_weeks else None,
            "n_weeks": len(week_results),
            "n_days": len(port_nav),
            "methods_used": methods_used,
            "te_target": te,
        }
    return reports


def _daily_weighted_ret(price_df: pd.DataFrame, w: pd.Series, d0, d1) -> float:
    """单日组合收益 = Σ w_i × (close_{d1}/close_{d0} − 1)，权重归一。"""
    tot = 0.0
    wsum = 0.0
    for t, wi in w.items():
        if wi <= 0 or t not in price_df.columns:
            continue
        try:
            p0 = float(price_df.at[d0, t])
            p1 = float(price_df.at[d1, t])
        except (KeyError, TypeError):
            continue
        if p0 and not np.isnan(p0) and not np.isnan(p1):
            tot += wi * (p1 / p0 - 1)
            wsum += wi
    return tot / wsum if wsum > 0 else 0.0


def _true_max_drawdown(nav) -> float:
    nav = np.asarray(nav, dtype=float)
    peak = np.maximum.accumulate(nav)
    return float(np.min((nav - peak) / peak))


def _win_rate(port_weekly: pd.Series, spy_weekly: pd.Series, up_only: bool | None) -> float | None:
    """port周收益跑赢SPY的比例。up_only=True只看SPY上涨周，False只看下跌周，
    None看全部。没有匹配周时返回None。"""
    aligned = pd.concat([port_weekly, spy_weekly], axis=1, keys=["port", "spy"]).dropna()
    if len(aligned) == 0:
        return None
    if up_only is True:
        sub = aligned[aligned["spy"] > 0]
    elif up_only is False:
        sub = aligned[aligned["spy"] < 0]
    else:
        sub = aligned
    if len(sub) == 0:
        return None
    return float((sub["port"] > sub["spy"]).mean())
