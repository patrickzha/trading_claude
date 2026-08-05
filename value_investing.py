"""
长周期基本面价值投资策略——用SEC EDGAR基本面数据(P/E、P/B、ROE、负债权益比)
构建价值因子组合，半年度调仓、长期持有，跟现有周频动量策略是完全不同的
逻辑：赌"便宜+优质的公司长期会被重新定价"，不是"短期动量延续"。

复用 us_stock_screener.py 里已经验证过时间点正确性的
_compute_fundamental_features()（SEC披露日期做时间点回溯，不会有前视
偏差），用点位时间正确的554只股票池(跟生存偏差修复用同一份数据，
只有约5年历史，半年度调仓下只有约10个独立持有期——样本量天然偏少，
这是长周期策略的固有代价，如实标注，不是bug)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from us_stock_screener import _compute_fundamental_features


def select_value_portfolio(price_df: pd.DataFrame, valid_tickers: list, rebalance_date,
                            top_n: int = 30, max_pe: float = 40.0,
                            sector_map: dict | None = None, max_per_sector: int | None = None,
                            factor_weights: tuple = (1.0, 1.0, 1.0),
                            sector_neutral: bool = False,
                            factor_set: str = "pe_pb_roe"):
    """在valid_tickers范围内算价值因子组合分，返回选中的ticker列表。
    max_per_sector不为None时，按综合分从高到低依次挑选，同一行业选够
    max_per_sector只后跳过——防止价值因子天然向金融/能源/工业这类"便宜股"
    集中的行业过度集中，用来测试"降低集中度能不能改善真实回撤"。

    sector_neutral(默认False，向后兼容)：True时P/E、P/B、ROE的排名改成
    "行业内部排名"而不是"全市场排名"(需要同时传sector_map)——见README
    第68轮左右的探索，去掉价值因子对金融/能源/工业这类传统低估值行业的
    结构性偏好，只保留"在自己所在行业里相对便宜/优质"这个信息，理论上
    更接近学术定义的价值因子(消除行业beta)。跟max_per_sector(限制选股
    结果里每个行业最多几只)是两回事：max_per_sector在排名之后做筛选，
    sector_neutral是在排名这一步本身就消除行业间的可比性差异。

    factor_set(默认"pe_pb_roe"，向后兼容原有行为)："pe_ps"时改用P/E+P/S
    两个因子的排名平均(不用P/B、ROE)——见README第326/327条：这个更简单
    的二因子组合在三段完整历史区间上比原有P/E-P/B-ROE三因子、以及更复杂
    的XGBoost非线性模型(第318/324条)都更强，2/3区间通过跟SPY的显著性
    检验。第327条同时也标注了这是连续调整出来的第三个变体，有多重检验
    风险，采纳时请留意这一点，不代表这个选项已经达到跟"pe_pb_roe"同等
    的验证成熟度。"""
    rows = []
    for t in valid_tickers:
        if t not in price_df.columns:
            continue
        px_series = price_df[t].loc[:rebalance_date].dropna()
        if len(px_series) == 0:
            continue
        curr_price = px_series.iloc[-1]
        feat = _compute_fundamental_features(t, rebalance_date, curr_price)
        pe, pb, roe, ps = feat["pe_ratio"], feat["pb_ratio"], feat["roe"], feat["ps_ratio"]
        # 经典价值screen：只要真正有正收益、正净资产的公司(PE/PB算得出来
        # 且为正)，排除PE高得离谱的（可能是分母接近0的噪声）
        if not (pe is not None and not np.isnan(pe) and 0 < pe < max_pe):
            continue
        if factor_set == "pe_ps":
            if not (ps is not None and not np.isnan(ps) and ps > 0):
                continue
            rows.append({"ticker": t, "pe": pe, "ps": ps})
            continue
        if not (pb is not None and not np.isnan(pb) and pb > 0):
            continue
        if roe is None or np.isnan(roe):
            continue
        rows.append({"ticker": t, "pe": pe, "pb": pb, "roe": roe})

    if len(rows) < top_n:
        return [r["ticker"] for r in rows]

    df = pd.DataFrame(rows)

    if factor_set == "pe_ps":
        # 见README第326/327条：简单二因子(P/E+P/S)排名平均，不做行业中性/
        # 权重自定义这类第326条之外没有验证过的组合，保持跟已验证配置一致。
        df["rank_pe"] = df["pe"].rank(ascending=True)
        df["rank_ps"] = df["ps"].rank(ascending=True)
        df["composite"] = (df["rank_pe"] + df["rank_ps"]) / 2
        df = df.sort_values("composite")
        if max_per_sector is not None and sector_map is not None:
            selected = []
            sector_counts: dict[str, int] = {}
            for t in df["ticker"]:
                s = sector_map.get(t, "UNKNOWN")
                if sector_counts.get(s, 0) >= max_per_sector:
                    continue
                selected.append(t)
                sector_counts[s] = sector_counts.get(s, 0) + 1
                if len(selected) >= top_n:
                    break
            return selected
        return df["ticker"].head(top_n).tolist()

    # 排名越低越好(PE/PB)，roe排名越高越好——三个排名加权平均当综合价值分。
    # factor_weights=(w_pe, w_pb, w_roe)，默认(1,1,1)是原来的均权。
    if sector_neutral and sector_map is not None:
        df["sector"] = df["ticker"].map(lambda t: sector_map.get(t, "UNKNOWN"))
        df["rank_pe"] = df.groupby("sector")["pe"].rank(ascending=True)
        df["rank_pb"] = df.groupby("sector")["pb"].rank(ascending=True)
        df["rank_roe"] = df.groupby("sector")["roe"].rank(ascending=False)
        # 行业内排名的量纲(1~行业内票数)天生比全市场排名小，直接除以行业
        # 内票数归一化到[0,1]区间，避免大行业的排名数值系统性压过小行业。
        sector_counts_all = df.groupby("sector")["ticker"].transform("count")
        df["rank_pe"] = df["rank_pe"] / sector_counts_all
        df["rank_pb"] = df["rank_pb"] / sector_counts_all
        df["rank_roe"] = df["rank_roe"] / sector_counts_all
    else:
        df["rank_pe"] = df["pe"].rank(ascending=True)
        df["rank_pb"] = df["pb"].rank(ascending=True)
        df["rank_roe"] = df["roe"].rank(ascending=False)
    w_pe, w_pb, w_roe = factor_weights
    df["composite"] = (w_pe * df["rank_pe"] + w_pb * df["rank_pb"] + w_roe * df["rank_roe"]) / sum(factor_weights)
    df = df.sort_values("composite")

    if max_per_sector is not None and sector_map is not None:
        selected = []
        sector_counts: dict[str, int] = {}
        for t in df["ticker"]:
            s = sector_map.get(t, "UNKNOWN")
            if sector_counts.get(s, 0) >= max_per_sector:
                continue
            selected.append(t)
            sector_counts[s] = sector_counts.get(s, 0) + 1
            if len(selected) >= top_n:
                break
        return selected

    return df["ticker"].head(top_n).tolist()


def _shrunk_cov(returns_window: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf收缩协方差估计——把噪声很大的样本协方差矩阵向一个
    结构化目标(sklearn默认用单位矩阵乘常数)收缩，收缩强度由数据自动
    决定，不是人工设一个ridge数值。见README第314条：动机是`_min_variance_
    weights`现有的ridge=0.0001是个固定的、跟数据规模无关的经验值，
    样本股票数接近历史天数时(这条腿top_n=30，cov_window默认60天，
    比例并不算宽松)样本协方差矩阵本身估计误差很大，Ledoit-Wolf是
    金融计量文献里针对这个问题的标准解法，不是重新发明一个新方法。"""
    from sklearn.covariance import LedoitWolf
    return LedoitWolf().fit(returns_window.values).covariance_


def _min_variance_weights(returns_window: pd.DataFrame, ridge: float = 0.0001,
                           cov_estimator: str = "sample") -> pd.Series:
    """最小方差组合权重：w = Σ^-1·1 / (1'·Σ^-1·1)。

    cov_estimator(默认"sample"，向后兼容)："sample"用原来的做法(样本
    协方差+固定ridge防止矩阵病态)；"ledoit_wolf"改用Ledoit-Wolf收缩
    估计(见README第314条)，不再需要额外加ridge，收缩本身就保证矩阵
    非奇异。负权重截断为0再归一化(长仓近似，不是严格的约束二次规划，
    但足以验证框架方向)。见README第140条。"""
    if cov_estimator == "ledoit_wolf":
        cov = _shrunk_cov(returns_window)
    else:
        cov = returns_window.cov().values
        cov = cov + np.eye(len(cov)) * ridge
    ones = np.ones(len(cov))
    try:
        inv_cov_ones = np.linalg.solve(cov, ones)
    except np.linalg.LinAlgError:
        return pd.Series(1.0 / len(returns_window.columns), index=returns_window.columns)
    w = inv_cov_ones / inv_cov_ones.sum()
    w = np.clip(w, 0, None)
    if w.sum() <= 0:
        return pd.Series(1.0 / len(returns_window.columns), index=returns_window.columns)
    return pd.Series(w / w.sum(), index=returns_window.columns)


def _risk_parity_weights(returns_window: pd.DataFrame, ridge: float = 0.0001) -> pd.Series:
    """等风险贡献(risk parity)权重：最小化各资产风险贡献相对目标的相对
    偏差平方和(用相对偏差，不是绝对偏差——第155条踩过的真实坑：绝对
    偏差在个股日收益方差量级(1e-4~1e-3)下小到让SLSQP的默认容差误判
    "已收敛"，权重实际上一步都没挪动)。求解失败/权重和非正时退化成
    等权(fail-safe)。见README第155/156条。"""
    from scipy.optimize import minimize
    cov = returns_window.cov().values
    cov = cov + np.eye(len(cov)) * ridge
    n = len(cov)

    def risk_contrib_variance(w):
        port_var = w @ cov @ w
        marginal = cov @ w
        rc = w * marginal
        target = port_var / n
        if abs(target) < 1e-12:
            return 0.0
        return np.sum(((rc - target) / target) ** 2)

    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    result = minimize(risk_contrib_variance, w0, method="SLSQP", bounds=bounds,
                      constraints=constraints, options={"maxiter": 200, "ftol": 1e-10})
    if not result.success or np.any(np.isnan(result.x)):
        return pd.Series(w0, index=returns_window.columns)
    w = np.clip(result.x, 0, None)
    if w.sum() <= 0:
        return pd.Series(w0, index=returns_window.columns)
    return pd.Series(w / w.sum(), index=returns_window.columns)


def _hrp_get_quasi_diag(link):
    link = link.astype(int)
    sort_idx = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_idx.max() >= num_items:
        sort_idx.index = range(0, sort_idx.shape[0] * 2, 2)
        df0 = sort_idx[sort_idx >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_idx[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_idx = pd.concat([sort_idx, df1])
        sort_idx = sort_idx.sort_index()
        sort_idx.index = range(sort_idx.shape[0])
    return sort_idx.tolist()


def _hrp_cluster_var(cov, items):
    sub_cov = cov.loc[items, items]
    ivp = 1.0 / np.diag(sub_cov)
    ivp /= ivp.sum()
    return float(ivp @ sub_cov @ ivp)


def _hrp_recursive_bisection(cov, sort_ix_tickers):
    """标准HRP递归二分，只有长度>1的簇才继续切分——第159条踩过的真实
    死循环bug：单元素簇对半切会重新产出自己，忘了这个终止条件会卡死。"""
    weights = pd.Series(1.0, index=sort_ix_tickers)
    clusters = [sort_ix_tickers]
    while any(len(c) > 1 for c in clusters):
        new_clusters = []
        for c in clusters:
            if len(c) <= 1:
                new_clusters.append(c)
                continue
            mid = len(c) // 2
            c0, c1 = c[:mid], c[mid:]
            var0 = _hrp_cluster_var(cov, c0)
            var1 = _hrp_cluster_var(cov, c1)
            alpha = 1 - var0 / (var0 + var1)
            weights[c0] *= alpha
            weights[c1] *= 1 - alpha
            new_clusters.append(c0)
            new_clusters.append(c1)
        clusters = new_clusters
    return weights


def _cap_and_renormalize(w: pd.Series, max_weight: float) -> pd.Series:
    """把权重逐步压到max_weight上限以下，超出部分按比例分给未封顶的
    持仓，迭代到没有仓位超过上限为止(标准的cap-and-redistribute做法)。"""
    w = w.copy()
    for _ in range(len(w)):
        over = w[w > max_weight]
        if over.empty:
            break
        excess = (over - max_weight).sum()
        w[over.index] = max_weight
        under_mask = w < max_weight
        if not under_mask.any():
            break
        w[under_mask] += excess * (w[under_mask] / w[under_mask].sum())
    return w / w.sum()


def _hrp_weights(returns_window: pd.DataFrame, max_weight: float | None = None) -> pd.Series:
    """层次风险平价(HRP, López de Prado 2016)：相关性->距离矩阵->层次
    聚类->quasi-diagonalization->递归二分配权重，不需要协方差矩阵求逆
    也不需要数值优化，理论上对协方差估计误差比min_variance/risk_parity
    更稳健。见README第159条。

    max_weight(默认None，向后兼容不设上限)：单票权重上限，见README第290
    条——递归二分配权重对已实现波动率极端离群的候选没有任何保护，罕见
    情况下(1.1%-3.3%的调仓日)会把50%以上仓位压在一只票上。设置这个参数
    后，超过上限的部分会按比例重新分配给未封顶的持仓(cap-and-redistribute)。
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    try:
        corr = returns_window.corr()
        cov = returns_window.cov()
        dist = np.sqrt(0.5 * (1 - corr))
        condensed = squareform(dist.values, checks=False)
        link = linkage(condensed, method="single")
        sort_ix = _hrp_get_quasi_diag(link)
        sort_ix_tickers = [returns_window.columns[i] for i in sort_ix]
        w = _hrp_recursive_bisection(cov, sort_ix_tickers)
        w = w.reindex(returns_window.columns)
        if w.isna().any() or w.sum() <= 0:
            return pd.Series(1.0 / len(returns_window.columns), index=returns_window.columns)
        w = w / w.sum()
        if max_weight is not None:
            w = _cap_and_renormalize(w, max_weight)
        return w
    except Exception:
        return pd.Series(1.0 / len(returns_window.columns), index=returns_window.columns)


def backtest_value_strategy(price_df: pd.DataFrame, membership: dict, spy_close: pd.Series | None,
                             rebalance_days: int = 126, top_n: int = 30, max_pe: float = 40.0,
                             sector_map: dict | None = None, max_per_sector: int | None = None,
                             factor_weights: tuple = (1.0, 1.0, 1.0), cost_bi: float = 0.003,
                             weighting: str = "equal", cov_window: int = 60,
                             stop_loss_pct: float | None = None,
                             hrp_max_weight: float | None = None,
                             factor_set: str = "pe_pb_roe"):
    """半年度(默认126个交易日)调仓的价值组合回测。membership是
    compute_point_in_time_membership()算出来的 {date_str: [ticker,...]}。
    返回逐期结果列表——同时按每日价格重建组合净值路径(daily_nav)，不是只
    看调仓点之间的首尾收益率，否则如果一次暴跌+反弹刚好都发生在同一个
    调仓区间内，最大回撤会被"抹平"成看不见，严重低估真实持有风险。

    cost_bi(默认0.3%，跟动量策略同一个假设)：每次调仓时，只对换手的那部分
    仓位收双边成本——跟上一期持仓有重叠的票不算换手(继续持有不产生交易
    成本)，只有新买入/卖出的部分才收费。之前的实现完全没有扣成本，这是
    这次会话回过头补上的一个真实缺口(见README"验证结论"第72条)。

    weighting(默认"equal"，向后兼容)："min_variance"时用trailing
    cov_window日协方差矩阵算最小方差权重代替等权，见README第140条——
    选股逻辑完全不变，只改仓位分配的数学框架，三段独立历史区间Sharpe
    和最大回撤全部同步改善(代价是CAGR小幅让步)。"risk_parity"时用等
    风险贡献权重(见README第155条)，2/3区间Sharpe改善，比min_variance
    稍弱但更稳健(第156条：在候选池风险同质化的场景下不容易被协方差
    估计噪声带偏)。"hrp"时用层次风险平价(见README第159条)，三段区间
    Sharpe和MaxDD全部同步改善，回撤改善幅度是三种方案里最大的，且理论
    上对协方差估计误差最稳健(不需要矩阵求逆也不需要数值优化)，目前是
    三种权重方案里证据最扎实的一个。

    stop_loss_pct(默认None，向后兼容不止损)：个股相对本次调仓买入价的
    止损阈值，比如0.10表示跌破买入价10%就"止损"(该票在本持有期内后续
    净值贡献固定在-10%这个水平，不再跟随实际价格波动，直到下次调仓才
    重新按新买入价计算)。见README第230条：价值腿此前从未做过止损保护
    (只有动量腿有)，10%阈值三段独立历史区间Sharpe和MaxDD全部同步改善，
    2014-2020段MaxDD改善超过16个百分点，是这次会话继仓位分配框架之后
    第一个"新风控机制"级别的正面发现。"""
    index = price_df.index
    n = len(index)
    results = []
    daily_nav = []  # [(date, nav), ...]，跨整个回测期连续拼接

    rebalance_idx = 252 + (cov_window if weighting in ("min_variance", "min_variance_lw", "risk_parity", "hrp") else 0)
    running_nav = 1.0
    prev_weights: pd.Series | None = None
    while rebalance_idx + rebalance_days < n:
        rebalance_date = index[rebalance_idx]
        next_date = index[rebalance_idx + rebalance_days]

        date_str = rebalance_date.strftime("%Y-%m-%d")
        valid_tickers = membership.get(date_str)
        if valid_tickers is None:
            # 找最近的一个key(membership按周构建，不一定精确命中这个日期)
            keys = sorted(membership.keys())
            valid_tickers = membership[max([k for k in keys if k <= date_str], default=keys[0])]

        picks = select_value_portfolio(price_df, valid_tickers, rebalance_date, top_n=top_n, max_pe=max_pe,
                                        sector_map=sector_map, max_per_sector=max_per_sector,
                                        factor_weights=factor_weights, factor_set=factor_set)
        picks = [p for p in picks if p in price_df.columns]

        if weighting in ("min_variance", "min_variance_lw", "risk_parity", "hrp") and len(picks) >= 5:
            hist_window = price_df[picks].loc[:rebalance_date].tail(cov_window + 1)
            rets_window = hist_window.pct_change().dropna()
            valid_cols = rets_window.columns[rets_window.notna().all()]
            if len(valid_cols) >= 5:
                if weighting == "hrp":
                    opt_weights = _hrp_weights(rets_window[valid_cols], max_weight=hrp_max_weight)
                elif weighting == "min_variance_lw":
                    opt_weights = _min_variance_weights(rets_window[valid_cols], cov_estimator="ledoit_wolf")
                else:
                    weight_fns = {"min_variance": _min_variance_weights, "risk_parity": _risk_parity_weights}
                    opt_weights = weight_fns[weighting](rets_window[valid_cols])
                weights = pd.Series(0.0, index=picks)
                for t in valid_cols:
                    weights[t] = opt_weights[t]
                weights = weights / weights.sum() if weights.sum() > 0 else pd.Series(1.0 / len(picks), index=picks)
            else:
                weights = pd.Series(1.0 / len(picks), index=picks)
        else:
            weights = pd.Series(1.0 / len(picks), index=picks) if picks else pd.Series(dtype=float)

        if prev_weights is not None and len(weights) > 0:
            all_tickers = set(weights.index) | set(prev_weights.index)
            turnover = sum(abs(weights.get(t, 0) - prev_weights.get(t, 0)) for t in all_tickers) / 2
        elif len(weights) > 0:
            turnover = 1.0  # 首次建仓，全部算换手
        else:
            turnover = 0.0
        running_nav *= (1 - turnover * cost_bi)
        prev_weights = weights

        # 逐日重建净值：每只票用调仓日价格做基准，按weights加权平均每日
        # "净值倍数"(weighting="equal"时weights全部相等，退化成原来的
        # 简单平均，行为完全不变)。
        period_days = index[rebalance_idx:min(rebalance_idx + rebalance_days, n)]
        entry_prices = {}
        for t in picks:
            try:
                entry_prices[t] = price_df[t].loc[:rebalance_date].dropna().iloc[-1]
            except IndexError:
                continue
        stopped_out: dict[str, float] = {}  # ticker -> 冻结在止损阈值的净值倍数(1-stop_loss_pct)
        for d in period_days:
            day_val = 0.0
            total_w = 0.0
            for t, p0 in entry_prices.items():
                if d in price_df.index and t in price_df.columns:
                    w = weights.get(t, 0)
                    if t in stopped_out:
                        day_val += w * stopped_out[t]
                        total_w += w
                        continue
                    p = price_df.at[d, t]
                    if pd.notna(p) and p0:
                        factor = p / p0
                        if stop_loss_pct is not None and factor <= (1 - stop_loss_pct):
                            factor = 1 - stop_loss_pct
                            stopped_out[t] = factor
                        day_val += w * factor
                        total_w += w
            if total_w > 0:
                daily_nav.append((d, running_nav * (day_val / total_w)))
        if daily_nav:
            running_nav = daily_nav[-1][1]

        rets = []
        for t in picks:
            try:
                p0 = price_df[t].loc[:rebalance_date].dropna().iloc[-1]
                p1 = price_df[t].loc[:next_date].dropna().iloc[-1]
                rets.append(p1 / p0 - 1)
            except (IndexError, KeyError):
                continue

        spy_ret = None
        if spy_close is not None:
            try:
                s0 = spy_close.loc[:rebalance_date].dropna().iloc[-1]
                s1 = spy_close.loc[:next_date].dropna().iloc[-1]
                spy_ret = float(s1 / s0 - 1)
            except (IndexError, KeyError):
                pass

        results.append({
            "rebalance_date": rebalance_date.strftime("%Y-%m-%d"),
            "n_picks": len(picks),
            "period_ret": float(np.mean(rets)) if rets else 0.0,
            "spy_ret": spy_ret,
        })
        rebalance_idx += rebalance_days

    return results, daily_nav


def true_daily_max_drawdown(daily_nav):
    """用逐日重建的净值路径算真实的最大回撤(peak-to-trough)，不是只看
    调仓点之间的首尾收益率——后者会把"同一个持有期内先跌后涨"的回撤
    完全抹平，严重低估真实风险。"""
    if len(daily_nav) < 2:
        return None
    nav = np.array([v for _, v in daily_nav])
    peak = np.maximum.accumulate(nav)
    return float(np.min((nav - peak) / peak))


def metrics_from_periods(results, daily_nav=None, periods_per_year: float = 2.0):
    rets = np.array([r["period_ret"] for r in results])
    if len(rets) < 2:
        return {"n_periods": len(rets), "CAGR": None, "Sharpe": None, "MaxDD": None, "TrueDailyMaxDD": None}
    nav = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(nav)
    mdd = float(np.min((nav - peak) / peak))
    n_years = len(rets) / periods_per_year
    cagr = float(nav[-1] ** (1 / n_years) - 1) if n_years > 0 else None
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(periods_per_year))
    spy_rets = np.array([r["spy_ret"] for r in results if r["spy_ret"] is not None])
    spy_cagr = None
    if len(spy_rets) == len(rets):
        spy_nav = np.cumprod(1 + spy_rets)
        spy_cagr = float(spy_nav[-1] ** (1 / n_years) - 1) if n_years > 0 else None
    true_mdd = true_daily_max_drawdown(daily_nav) if daily_nav else None
    return {"n_periods": len(rets), "CAGR": cagr, "Sharpe": sharpe, "MaxDD": mdd,
            "TrueDailyMaxDD": true_mdd, "SPY_CAGR": spy_cagr}
