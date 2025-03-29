from graph.state import AgentState, show_agent_reasoning
from tools.api import (
    get_financial_metrics,
    get_market_cap,
    search_line_items,
    get_insider_trades,
    get_company_news,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from utils.progress import progress
from utils.llm import call_llm
import statistics


class PhilFisherSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str

'''
菲利普·费雪代理
 - 传奇成长投资者，精通市场调查分析
'''
def phil_fisher_agent(state: AgentState):
    """
    Analyzes stocks using Phil Fisher's investing principles:
      - Seek companies with long-term above-average growth potential
      - Emphasize quality of management and R&D
      - Look for strong margins, consistent growth, and manageable leverage
      - Combine fundamental 'scuttlebutt' style checks with basic sentiment and insider data
      - Willing to pay up for quality, but still mindful of valuation
      - Generally focuses on long-term compounding

    Returns a bullish/bearish/neutral signal with confidence and reasoning.
    """
    '''
    使用菲利普·费雪的投资原则分析股票：
      - 寻找具有长期高于平均水平增长潜力的公司
      - 强调管理层质量和研发投入
      - 关注强劲的利润率、持续的增长和可控的杠杆
      - 将基本面的"市场调查"式检查与基本的情绪和内部人交易数据相结合
      - 愿意为优质公司支付溢价，但仍需注意估值
      - 通常专注于长期复利增长

    返回看涨/看跌/中性信号，并附带置信度和理由。
    '''
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    analysis_data = {}
    fisher_analysis = {}

    for ticker in tickers:
        progress.update_status("phil_fisher_agent", ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=5)

        progress.update_status("phil_fisher_agent", ticker, "Gathering financial line items")
        # Include relevant line items for Phil Fisher's approach:
        #   - Growth & Quality: revenue, net_income, earnings_per_share, R&D expense
        #   - Margins & Stability: operating_income, operating_margin, gross_margin
        #   - Management Efficiency & Leverage: total_debt, shareholders_equity, free_cash_flow
        #   - Valuation: net_income, free_cash_flow (for P/E, P/FCF), ebit, ebitda
        # 寻找与菲利普·费雪的投资原则相关的财务指标：
        #   - 增长 & 质量：收入、净利润、每股收益、研发费用
        #   - 利润率 & 稳定性：营业收入、营业收入利润率、毛利率
        #   - 管理层效率 & 杠杆：总负债、股东权益、自由现金流
        #   - 估值：净利润、自由现金流（P/E、P/FCF）、EBIT、EBITDA
        financial_line_items = search_line_items(
            ticker,
            [
                "revenue",
                "net_income",
                "earnings_per_share",
                "free_cash_flow",
                "research_and_development",
                "operating_income",
                "operating_margin",
                "gross_margin",
                "total_debt",
                "shareholders_equity",
                "cash_and_equivalents",
                "ebit",
                "ebitda",
            ],
            end_date,
            period="annual",
            limit=5,
        )

        progress.update_status("phil_fisher_agent", ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date)

        progress.update_status("phil_fisher_agent", ticker, "Fetching insider trades")
        insider_trades = get_insider_trades(ticker, end_date, start_date=None, limit=50)

        progress.update_status("phil_fisher_agent", ticker, "Fetching company news")
        company_news = get_company_news(ticker, end_date, start_date=None, limit=50)

        progress.update_status("phil_fisher_agent", ticker, "Analyzing growth & quality")
        growth_quality = analyze_fisher_growth_quality(financial_line_items)

        progress.update_status("phil_fisher_agent", ticker, "Analyzing margins & stability")
        margins_stability = analyze_margins_stability(financial_line_items)

        progress.update_status("phil_fisher_agent", ticker, "Analyzing management efficiency & leverage")
        mgmt_efficiency = analyze_management_efficiency_leverage(financial_line_items)

        progress.update_status("phil_fisher_agent", ticker, "Analyzing valuation (Fisher style)")
        fisher_valuation = analyze_fisher_valuation(financial_line_items, market_cap)

        progress.update_status("phil_fisher_agent", ticker, "Analyzing insider activity")
        insider_activity = analyze_insider_activity(insider_trades)

        progress.update_status("phil_fisher_agent", ticker, "Analyzing sentiment")
        sentiment_analysis = analyze_sentiment(company_news)

        # Combine partial scores with weights typical for Fisher:
        #   30% Growth & Quality
        #   25% Margins & Stability
        #   20% Management Efficiency
        #   15% Valuation
        #   5% Insider Activity
        #   5% Sentiment
        # 结合部分分数与菲利普·费雪的投资原则的权重：
        #   30% 增长 & 质量
        #   25% 利润率 & 稳定性
        #   20% 管理层效率
        #   15% 估值
        #   5% 内部人活动
        #   5% 情绪
        total_score = (
            growth_quality["score"] * 0.30
            + margins_stability["score"] * 0.25
            + mgmt_efficiency["score"] * 0.20
            + fisher_valuation["score"] * 0.15
            + insider_activity["score"] * 0.05
            + sentiment_analysis["score"] * 0.05
        )

        max_possible_score = 10

        # Simple bullish/neutral/bearish signal
        # 简单的看涨/看跌/中性信号
        if total_score >= 7.5:
            signal = "bullish"
        elif total_score <= 4.5:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_possible_score,
            "growth_quality": growth_quality,
            "margins_stability": margins_stability,
            "management_efficiency": mgmt_efficiency,
            "valuation_analysis": fisher_valuation,
            "insider_activity": insider_activity,
            "sentiment_analysis": sentiment_analysis,
        }

        progress.update_status("phil_fisher_agent", ticker, "Generating Phil Fisher-style analysis")
        fisher_output = generate_fisher_output(
            ticker=ticker,
            analysis_data=analysis_data,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )

        fisher_analysis[ticker] = {
            "signal": fisher_output.signal,
            "confidence": fisher_output.confidence,
            "reasoning": fisher_output.reasoning,
        }

        progress.update_status("phil_fisher_agent", ticker, "Done")

    # Wrap results in a single message
    # 包装结果在单个消息中
    message = HumanMessage(content=json.dumps(fisher_analysis), name="phil_fisher_agent")

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(fisher_analysis, "Phil Fisher Agent")

    state["data"]["analyst_signals"]["phil_fisher_agent"] = fisher_analysis
    return {"messages": [message], "data": state["data"]}


def analyze_fisher_growth_quality(financial_line_items: list) -> dict:
    """
    Evaluate growth & quality:
      - Consistent Revenue Growth
      - Consistent EPS Growth
      - R&D as a % of Revenue (if relevant, indicative of future-oriented spending)
    """
    '''
    评估增长 & 质量：
      - 一致的营业收入增长
      - 一致的每股收益增长
      - R&D 作为收入的 %（如果适用，表明未来的投资）
    '''
    if not financial_line_items or len(financial_line_items) < 2:
        return {
            "score": 0,
            "details": "Insufficient financial data for growth/quality analysis",
        }

    details = []
    raw_score = 0  # up to 9 raw points => scale to 0–10 # 最多9个原始分数 => 缩放到0-10

    # 1. Revenue Growth (YoY)
    # 1. 营业收入增长（同比）
    revenues = [fi.revenue for fi in financial_line_items if fi.revenue is not None]
    if len(revenues) >= 2:
        # We'll look at the earliest vs. latest to gauge multi-year growth if possible
        # 如果有多个年份的数据，我们将看最早的与最新的
        latest_rev = revenues[0]
        oldest_rev = revenues[-1]
        if oldest_rev > 0:
            rev_growth = (latest_rev - oldest_rev) / abs(oldest_rev)
            if rev_growth > 0.80:
                raw_score += 3
                details.append(f"Very strong multi-period revenue growth: {rev_growth:.1%}")
            elif rev_growth > 0.40:
                raw_score += 2
                details.append(f"Moderate multi-period revenue growth: {rev_growth:.1%}")
            elif rev_growth > 0.10:
                raw_score += 1
                details.append(f"Slight multi-period revenue growth: {rev_growth:.1%}")
            else:
                details.append(f"Minimal or negative multi-period revenue growth: {rev_growth:.1%}")
        else:
            details.append("Oldest revenue is zero/negative; cannot compute growth.")
    else:
        details.append("Not enough revenue data points for growth calculation.")

    # 2. EPS Growth (YoY)
    # 2. 每股收益增长（同比）
    eps_values = [fi.earnings_per_share for fi in financial_line_items if fi.earnings_per_share is not None]
    if len(eps_values) >= 2:
        latest_eps = eps_values[0]
        oldest_eps = eps_values[-1]
        if abs(oldest_eps) > 1e-9:
            eps_growth = (latest_eps - oldest_eps) / abs(oldest_eps)
            if eps_growth > 0.80:
                raw_score += 3
                details.append(f"Very strong multi-period EPS growth: {eps_growth:.1%}")
            elif eps_growth > 0.40:
                raw_score += 2
                details.append(f"Moderate multi-period EPS growth: {eps_growth:.1%}")
            elif eps_growth > 0.10:
                raw_score += 1
                details.append(f"Slight multi-period EPS growth: {eps_growth:.1%}")
            else:
                details.append(f"Minimal or negative multi-period EPS growth: {eps_growth:.1%}")
        else:
            details.append("Oldest EPS near zero; skipping EPS growth calculation.")
    else:
        details.append("Not enough EPS data points for growth calculation.")

    # 3. R&D as % of Revenue (if we have R&D data)
    # 3. R&D 作为收入的 %（如果有 R&D 数据）
    rnd_values = [fi.research_and_development for fi in financial_line_items if fi.research_and_development is not None]
    if rnd_values and revenues and len(rnd_values) == len(revenues):
        # We'll just look at the most recent for a simple measure
        # 我们只看最新的
        recent_rev = revenues[0] if revenues[0] else 1e-9
        rnd_ratio = recent_rnd / recent_rev
        # Generally, Fisher admired companies that invest aggressively in R&D,
        # but it must be appropriate. We'll assume "3%-15%" is healthy, just as an example.
        # 通常，费雪赞赏公司积极投资于 R&D，但这必须适当。我们将假设“3%-15%”是健康的，只是作为一个例子。
        if 0.03 <= rnd_ratio <= 0.15:
            raw_score += 3
            details.append(f"R&D ratio {rnd_ratio:.1%} indicates significant investment in future growth")
        elif rnd_ratio > 0.15:
            raw_score += 2
            details.append(f"R&D ratio {rnd_ratio:.1%} is very high (could be good if well-managed)")
        elif rnd_ratio > 0.0:
            raw_score += 1
            details.append(f"R&D ratio {rnd_ratio:.1%} is somewhat low but still positive")
        else:
            details.append("No meaningful R&D expense ratio")
    else:
        details.append("Insufficient R&D data to evaluate")

    # scale raw_score (max 9) to 0–10
    # 将 raw_score（最多9）缩放到 0-10
    final_score = min(10, (raw_score / 9) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_margins_stability(financial_line_items: list) -> dict:
    """
    Looks at margin consistency (gross/operating margin) and general stability over time.
    """
    '''
    查看利润率的一致性（毛利率/营业收入）和时间上的稳定性。
    '''
    if not financial_line_items or len(financial_line_items) < 2:
        return {
            "score": 0,
            "details": "Insufficient data for margin stability analysis",
        }

    details = []
    raw_score = 0  # up to 6 => scale to 0-10 # 最多6个原始分数 => 缩放到0-10

    # 1. Operating Margin Consistency
    # 1. 营业收入利润率一致性
    op_margins = [fi.operating_margin for fi in financial_line_items if fi.operating_margin is not None]
    if len(op_margins) >= 2:
        # Check if margins are stable or improving (comparing oldest to newest)
        # 检查利润率是否稳定或提高（比较最早与最新）
        oldest_op_margin = op_margins[-1]
        newest_op_margin = op_margins[0]
        if newest_op_margin >= oldest_op_margin > 0:
            raw_score += 2
            details.append(f"Operating margin stable or improving ({oldest_op_margin:.1%} -> {newest_op_margin:.1%})")
        elif newest_op_margin > 0:
            raw_score += 1
            details.append(f"Operating margin positive but slightly declined")
        else:
            details.append(f"Operating margin may be negative or uncertain")
    else:
        details.append("Not enough operating margin data points")

    # 2. Gross Margin Level
    # 2. 毛利率水平
    gm_values = [fi.gross_margin for fi in financial_line_items if fi.gross_margin is not None]
    if gm_values:
        # We'll just take the most recent
        # 我们只看最新的
        recent_gm = gm_values[0]
        if recent_gm > 0.5:
            raw_score += 2
            details.append(f"Strong gross margin: {recent_gm:.1%}")
        elif recent_gm > 0.3:
            raw_score += 1
            details.append(f"Moderate gross margin: {recent_gm:.1%}")
        else:
            details.append(f"Low gross margin: {recent_gm:.1%}")
    else:
        details.append("No gross margin data available")

    # 3. Multi-year Margin Stability
    #   e.g. if we have at least 3 data points, see if standard deviation is low.
    # 3. 多年的利润率稳定性
    #   例如，如果我们至少有3个数据点，看看标准差是否低。
    if len(op_margins) >= 3:
        stdev = statistics.pstdev(op_margins)
        if stdev < 0.02:
            raw_score += 2
            details.append("Operating margin extremely stable over multiple years")
        elif stdev < 0.05:
            raw_score += 1
            details.append("Operating margin reasonably stable")
        else:
            details.append("Operating margin volatility is high")
    else:
        details.append("Not enough margin data points for volatility check")

    # scale raw_score (max 6) to 0-10
    # 将 raw_score（最多6）缩放到 0-10
    final_score = min(10, (raw_score / 6) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_management_efficiency_leverage(financial_line_items: list) -> dict:
    """
    Evaluate management efficiency & leverage:
      - Return on Equity (ROE)
      - Debt-to-Equity ratio
      - Possibly check if free cash flow is consistently positive
    """
    '''
    评估管理层效率 & 杠杆：
      - 股东权益报酬率（ROE）
      - 股东权益与负债比率
      - 可能检查自由现金流是否一直是积极的
    '''
    if not financial_line_items:
        return {
            "score": 0,
            "details": "No financial data for management efficiency analysis",
        }

    details = []
    raw_score = 0  # up to 6 => scale to 0–10 # 最多6个原始分数 => 缩放到0-10

    # 1. Return on Equity (ROE)
    # 1. 股东权益报酬率（ROE）
    ni_values = [fi.net_income for fi in financial_line_items if fi.net_income is not None]
    eq_values = [fi.shareholders_equity for fi in financial_line_items if fi.shareholders_equity is not None]
    if ni_values and eq_values and len(ni_values) == len(eq_values):
        recent_ni = ni_values[0]
        recent_eq = eq_values[0] if eq_values[0] else 1e-9
        if recent_ni > 0:
            roe = recent_ni / recent_eq
            if roe > 0.2:
                raw_score += 3
                details.append(f"High ROE: {roe:.1%}")
            elif roe > 0.1:
                raw_score += 2
                details.append(f"Moderate ROE: {roe:.1%}")
            elif roe > 0:
                raw_score += 1
                details.append(f"Positive but low ROE: {roe:.1%}")
            else:
                details.append(f"ROE is near zero or negative: {roe:.1%}")
        else:
            details.append("Recent net income is zero or negative, hurting ROE")
    else:
        details.append("Insufficient data for ROE calculation")

    # 2. Debt-to-Equity
    # 2. 股东权益与负债比率
    debt_values = [fi.total_debt for fi in financial_line_items if fi.total_debt is not None]
    if debt_values and eq_values and len(debt_values) == len(eq_values):
        recent_debt = debt_values[0]
        recent_equity = eq_values[0] if eq_values[0] else 1e-9
        dte = recent_debt / recent_equity
        if dte < 0.3:
            raw_score += 2
            details.append(f"Low debt-to-equity: {dte:.2f}")
        elif dte < 1.0:
            raw_score += 1
            details.append(f"Manageable debt-to-equity: {dte:.2f}")
        else:
            details.append(f"High debt-to-equity: {dte:.2f}")
    else:
        details.append("Insufficient data for debt/equity analysis")

    # 3. FCF Consistency
    # 3. FCF 一致性
    fcf_values = [fi.free_cash_flow for fi in financial_line_items if fi.free_cash_flow is not None]
    if fcf_values and len(fcf_values) >= 2:
        # Check if FCF is positive in recent years
        # 检查 FCF 是否在最近的几年中是积极的
        positive_fcf_count = sum(1 for x in fcf_values if x and x > 0)
        # We'll be simplistic: if most are positive, reward
        # 我们会简化：如果大多数都是积极的，奖励
        ratio = positive_fcf_count / len(fcf_values)
        if ratio > 0.8:
            raw_score += 1
            details.append(f"Majority of periods have positive FCF ({positive_fcf_count}/{len(fcf_values)})")
        else:
            details.append(f"Free cash flow is inconsistent or often negative")
    else:
        details.append("Insufficient or no FCF data to check consistency")

    final_score = min(10, (raw_score / 6) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_fisher_valuation(financial_line_items: list, market_cap: float | None) -> dict:
    """
    Phil Fisher is willing to pay for quality and growth, but still checks:
      - P/E
      - P/FCF
      - (Optionally) Enterprise Value metrics, but simpler approach is typical
    We will grant up to 2 points for each of two metrics => max 4 raw => scale to 0–10.
    """
    '''
    Phil Fisher 愿意为质量和增长买单，但仍然检查：
      - P/E
      - P/FCF
      - （可选）企业价值指标，但通常更简单的方法是典型的
    我们将为两个指标中的每一个最多授予2个点 => 最多4个原始 => 缩放到0-10。
    '''
    if not financial_line_items or market_cap is None:
        return {"score": 0, "details": "Insufficient data to perform valuation"}

    details = []
    raw_score = 0

    # Gather needed data
    # 收集所需的数据
    net_incomes = [fi.net_income for fi in financial_line_items if fi.net_income is not None]
    fcf_values = [fi.free_cash_flow for fi in financial_line_items if fi.free_cash_flow is not None]

    # 1) P/E
    recent_net_income = net_incomes[0] if net_incomes else None
    if recent_net_income and recent_net_income > 0:
        pe = market_cap / recent_net_income
        pe_points = 0
        if pe < 20:
            pe_points = 2
            details.append(f"Reasonably attractive P/E: {pe:.2f}")
        elif pe < 30:
            pe_points = 1
            details.append(f"Somewhat high but possibly justifiable P/E: {pe:.2f}")
        else:
            details.append(f"Very high P/E: {pe:.2f}")
        raw_score += pe_points
    else:
        details.append("No positive net income for P/E calculation")

    # 2) P/FCF
    recent_fcf = fcf_values[0] if fcf_values else None
    if recent_fcf and recent_fcf > 0:
        pfcf = market_cap / recent_fcf
        pfcf_points = 0
        if pfcf < 20:
            pfcf_points = 2
            details.append(f"Reasonable P/FCF: {pfcf:.2f}")
        elif pfcf < 30:
            pfcf_points = 1
            details.append(f"Somewhat high P/FCF: {pfcf:.2f}")
        else:
            details.append(f"Excessively high P/FCF: {pfcf:.2f}")
        raw_score += pfcf_points
    else:
        details.append("No positive free cash flow for P/FCF calculation")

    # scale raw_score (max 4) to 0–10
    # 将 raw_score（最多4）缩放到 0-10
    final_score = min(10, (raw_score / 4) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_insider_activity(insider_trades: list) -> dict:
    """
    Simple insider-trade analysis:
      - If there's heavy insider buying, we nudge the score up.
      - If there's mostly selling, we reduce it.
      - Otherwise, neutral.
    """
    # Default is neutral (5/10).
    '''
    简单的内部交易分析：
      - 如果有大量的内部购买，我们会鼓励分数。
      - 如果大部分都是卖出，我们会减少它。
      - 否则，中性。
    '''
    # 默认是中性（5/10）。
    score = 5
    details = []

    if not insider_trades:
        details.append("No insider trades data; defaulting to neutral")
        return {"score": score, "details": "; ".join(details)}

    buys, sells = 0, 0
    for trade in insider_trades:
        if trade.transaction_shares is not None:
            if trade.transaction_shares > 0:
                buys += 1
            elif trade.transaction_shares < 0:
                sells += 1

    total = buys + sells
    if total == 0:
        details.append("No buy/sell transactions found; neutral")
        return {"score": score, "details": "; ".join(details)}

    buy_ratio = buys / total
    if buy_ratio > 0.7:
        score = 8
        details.append(f"Heavy insider buying: {buys} buys vs. {sells} sells")
    elif buy_ratio > 0.4:
        score = 6
        details.append(f"Moderate insider buying: {buys} buys vs. {sells} sells")
    else:
        score = 4
        details.append(f"Mostly insider selling: {buys} buys vs. {sells} sells")

    return {"score": score, "details": "; ".join(details)}


def analyze_sentiment(news_items: list) -> dict:
    """
    Basic news sentiment: negative keyword check vs. overall volume.
    """
    '''
    基本的新闻情绪：负面关键词检查与总体数量。
    '''
    if not news_items:
        return {"score": 5, "details": "No news data; defaulting to neutral sentiment"}

    negative_keywords = ["lawsuit", "fraud", "negative", "downturn", "decline", "investigation", "recall"]
    negative_count = 0
    for news in news_items:
        title_lower = (news.title or "").lower()
        if any(word in title_lower for word in negative_keywords):
            negative_count += 1

    details = []
    if negative_count > len(news_items) * 0.3:
        score = 3
        details.append(f"High proportion of negative headlines: {negative_count}/{len(news_items)}")
    elif negative_count > 0:
        score = 6
        details.append(f"Some negative headlines: {negative_count}/{len(news_items)}")
    else:
        score = 8
        details.append("Mostly positive/neutral headlines")

    return {"score": score, "details": "; ".join(details)}
'''
你是一个菲利普·费雪AI代理，使用他的原则做出投资决策：

1. 强调长期增长潜力和管理层质量
2. 关注投资研发以开发未来产品/服务的公司
3. 寻找强劲的盈利能力和稳定的利润率
4. 愿意为优秀公司支付更高价格，但仍需注意估值
5. 依靠深入研究（市场调查）和全面的基本面检查

在提供理由时，要做到详尽和具体：
1. 详细讨论公司的增长前景，包括具体的指标和趋势
2. 评估管理层质量及其资本配置决策
3. 强调可能推动未来增长的研发投资和产品管线
4. 用精确数字评估利润率和盈利指标的一致性
5. 解释能够维持3-5年以上增长的竞争优势
6. 使用菲利普·费雪的方法论、以增长为重点、面向长期的语气

看涨示例："这家公司展现出我们寻求的持续增长特征，五年来收入年增长率达18%。管理层通过将15%的收入分配给研发展现出卓越的远见，这已产生了三条有前景的新产品线。22-24%的稳定营业利润率表明定价能力和运营效率应该会继续..."

看跌示例："尽管在一个增长的行业中运营，管理层未能将研发投资（仅占收入的5%）转化为有意义的新产品。利润率在10-15%之间波动，显示出运营执行力不稳定。公司面临来自三个拥有更优分销网络的大型竞争对手的日益激烈的竞争。考虑到这些关于长期增长可持续性的担忧..."

你必须输出一个JSON对象，包含：
- "signal": "bullish"或"bearish"或"neutral"
- "confidence": 0到100之间的浮点数
- "reasoning": 详细的解释
'''
def generate_fisher_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> PhilFisherSignal:
    """
    Generates a JSON signal in the style of Phil Fisher.
    """
    '''
    生成一个 JSON 信号，其样式与 Phil Fisher 相同。
    '''
    template = ChatPromptTemplate.from_messages(
        [
            (
              "system",
              """You are a Phil Fisher AI agent, making investment decisions using his principles:
  
              1. Emphasize long-term growth potential and quality of management.
              2. Focus on companies investing in R&D for future products/services.
              3. Look for strong profitability and consistent margins.
              4. Willing to pay more for exceptional companies but still mindful of valuation.
              5. Rely on thorough research (scuttlebutt) and thorough fundamental checks.
              
              When providing your reasoning, be thorough and specific by:
              1. Discussing the company's growth prospects in detail with specific metrics and trends
              2. Evaluating management quality and their capital allocation decisions
              3. Highlighting R&D investments and product pipeline that could drive future growth
              4. Assessing consistency of margins and profitability metrics with precise numbers
              5. Explaining competitive advantages that could sustain growth over 3-5+ years
              6. Using Phil Fisher's methodical, growth-focused, and long-term oriented voice
              
              For example, if bullish: "This company exhibits the sustained growth characteristics we seek, with revenue increasing at 18% annually over five years. Management has demonstrated exceptional foresight by allocating 15% of revenue to R&D, which has produced three promising new product lines. The consistent operating margins of 22-24% indicate pricing power and operational efficiency that should continue to..."
              
              For example, if bearish: "Despite operating in a growing industry, management has failed to translate R&D investments (only 5% of revenue) into meaningful new products. Margins have fluctuated between 10-15%, showing inconsistent operational execution. The company faces increasing competition from three larger competitors with superior distribution networks. Given these concerns about long-term growth sustainability..."
              
              You must output a JSON object with:
                - "signal": "bullish" or "bearish" or "neutral"
                - "confidence": a float between 0 and 100
                - "reasoning": a detailed explanation
              """,
            ),
            (
              "human",
              """Based on the following analysis, create a Phil Fisher-style investment signal.

              Analysis Data for {ticker}:
              {analysis_data}

              Return the trading signal in this JSON format:
              {{
                "signal": "bullish/bearish/neutral",
                "confidence": float (0-100),
                "reasoning": "string"
              }}
              """,
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    def create_default_signal():
        return PhilFisherSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis, defaulting to neutral"
        )

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=PhilFisherSignal,
        agent_name="phil_fisher_agent",
        default_factory=create_default_signal,
    )
