from graph.state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from tools.api import get_financial_metrics, get_market_cap, search_line_items
from utils.llm import call_llm
from utils.progress import progress


class WarrenBuffettSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str

'''
沃伦·巴菲特代理
 - 奥马哈先知，寻找合理价格的优质公司
'''
def warren_buffett_agent(state: AgentState):
    """Analyzes stocks using Buffett's principles and LLM reasoning."""
    '''
    使用巴菲特的投资原则和LLM推理来分析股票。
    '''
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    # Collect all analysis for LLM reasoning
    # 将所有分析汇总到LLM推理中
    analysis_data = {}
    buffett_analysis = {}

    for ticker in tickers:
        progress.update_status("warren_buffett_agent", ticker, "Fetching financial metrics")
        # Fetch required data
        # 获取所需的数据
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=5)

        progress.update_status("warren_buffett_agent", ticker, "Gathering financial line items")
        financial_line_items = search_line_items(
            ticker,
            [
                "capital_expenditure",
                "depreciation_and_amortization",
                "net_income",
                "outstanding_shares",
                "total_assets",
                "total_liabilities",
                "dividends_and_other_cash_distributions",
                "issuance_or_purchase_of_equity_shares",
            ],
            end_date,
        )

        progress.update_status("warren_buffett_agent", ticker, "Getting market cap")
        # Get current market cap
        # 获取当前市值
        market_cap = get_market_cap(ticker, end_date)

        progress.update_status("warren_buffett_agent", ticker, "Analyzing fundamentals")
        # Analyze fundamentals
        # 分析基本面
        fundamental_analysis = analyze_fundamentals(metrics)

        progress.update_status("warren_buffett_agent", ticker, "Analyzing consistency")
        consistency_analysis = analyze_consistency(financial_line_items)

        progress.update_status("warren_buffett_agent", ticker, "Analyzing moat")
        moat_analysis = analyze_moat(metrics)

        progress.update_status("warren_buffett_agent", ticker, "Analyzing management quality")
        mgmt_analysis = analyze_management_quality(financial_line_items)

        progress.update_status("warren_buffett_agent", ticker, "Calculating intrinsic value")
        intrinsic_value_analysis = calculate_intrinsic_value(financial_line_items)

        # Calculate total score
        # 计算总分
        total_score = fundamental_analysis["score"] + consistency_analysis["score"] + moat_analysis["score"] + mgmt_analysis["score"]
        max_possible_score = 10 + moat_analysis["max_score"] + mgmt_analysis["max_score"]
        # fundamental_analysis + consistency combined were up to 10 points total
        # moat can add up to 3, mgmt can add up to 2, for example
        # 基本面分析和一致性分析总分最高为10分
        # 护城河最高可加3分，管理层最高可加2分，举例来说

        # Add margin of safety analysis if we have both intrinsic value and current price
        # 如果同时有内在价值和当前价格，则添加安全边际分析
        margin_of_safety = None
        intrinsic_value = intrinsic_value_analysis["intrinsic_value"]
        if intrinsic_value and market_cap:
            margin_of_safety = (intrinsic_value - market_cap) / market_cap

        # Generate trading signal using a stricter margin-of-safety requirement
        # if fundamentals+moat+management are strong but margin_of_safety < 0.3, it's neutral
        # if fundamentals are very weak or margin_of_safety is severely negative -> bearish
        # else bullish
        # 使用更严格的安全边际要求生成交易信号
        # 如果基本面+护城河+管理层很强但安全边际 < 0.3，则为中性
        # 如果基本面很弱或安全边际严重为负 -> 看跌
        # 否则看涨
        if (total_score >= 0.7 * max_possible_score) and margin_of_safety and (margin_of_safety >= 0.3):
            signal = "bullish"
        elif total_score <= 0.3 * max_possible_score or (margin_of_safety is not None and margin_of_safety < -0.3):
            # negative margin of safety beyond -30% could be overpriced -> bearish
            # 安全边际超过-30%可能是高估 -> 看跌
            signal = "bearish"
        else:
            signal = "neutral"

        # Combine all analysis results
        # 汇总所有分析结果
        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_possible_score,
            "fundamental_analysis": fundamental_analysis,
            "consistency_analysis": consistency_analysis,
            "moat_analysis": moat_analysis,
            "management_analysis": mgmt_analysis,
            "intrinsic_value_analysis": intrinsic_value_analysis,
            "market_cap": market_cap,
            "margin_of_safety": margin_of_safety,
        }

        progress.update_status("warren_buffett_agent", ticker, "Generating Warren Buffett analysis")
        buffett_output = generate_buffett_output(
            ticker=ticker,
            analysis_data=analysis_data,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )

        # Store analysis in consistent format with other agents
        # 将分析结果存储在一致的格式中
        buffett_analysis[ticker] = {
            "signal": buffett_output.signal,
            "confidence": buffett_output.confidence, # Normalize between 0 to 100 # 将置信度归一化到0到100之间
            "reasoning": buffett_output.reasoning,
        }

        progress.update_status("warren_buffett_agent", ticker, "Done")

    # Create the message
    # 创建消息
    message = HumanMessage(content=json.dumps(buffett_analysis), name="warren_buffett_agent")

    # Show reasoning if requested
    # 显示推理
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(buffett_analysis, "Warren Buffett Agent")

    # Add the signal to the analyst_signals list
    # 将信号添加到analyst_signals列表
    state["data"]["analyst_signals"]["warren_buffett_agent"] = buffett_analysis

    return {"messages": [message], "data": state["data"]}


def analyze_fundamentals(metrics: list) -> dict[str, any]:
    """Analyze company fundamentals based on Buffett's criteria."""
    '''
    基于巴菲特的标准分析公司基本面。
    '''
    if not metrics:
        return {"score": 0, "details": "Insufficient fundamental data"}

    latest_metrics = metrics[0]

    score = 0
    reasoning = []

    # Check ROE (Return on Equity)
    # 检查 ROE（归属于母公司股东的净利润率）
    if latest_metrics.return_on_equity and latest_metrics.return_on_equity > 0.15:  # 15% ROE threshold
        score += 2
        reasoning.append(f"Strong ROE of {latest_metrics.return_on_equity:.1%}")
    elif latest_metrics.return_on_equity:
        reasoning.append(f"Weak ROE of {latest_metrics.return_on_equity:.1%}")
    else:
        reasoning.append("ROE data not available")

    # Check Debt to Equity
    # 检查债务与权益的比率
    if latest_metrics.debt_to_equity and latest_metrics.debt_to_equity < 0.5:
        score += 2
        reasoning.append("Conservative debt levels")
    elif latest_metrics.debt_to_equity:
        reasoning.append(f"High debt to equity ratio of {latest_metrics.debt_to_equity:.1f}")
    else:
        reasoning.append("Debt to equity data not available")

    # Check Operating Margin
    # 检查运营利润率
    if latest_metrics.operating_margin and latest_metrics.operating_margin > 0.15:
        score += 2
        reasoning.append("Strong operating margins")
    elif latest_metrics.operating_margin:
        reasoning.append(f"Weak operating margin of {latest_metrics.operating_margin:.1%}")
    else:
        reasoning.append("Operating margin data not available")

    # Check Current Ratio
    # 检查流动比率
    if latest_metrics.current_ratio and latest_metrics.current_ratio > 1.5:
        score += 1
        reasoning.append("Good liquidity position")
    elif latest_metrics.current_ratio:
        reasoning.append(f"Weak liquidity with current ratio of {latest_metrics.current_ratio:.1f}")
    else:
        reasoning.append("Current ratio data not available")

    return {"score": score, "details": "; ".join(reasoning), "metrics": latest_metrics.model_dump()}


def analyze_consistency(financial_line_items: list) -> dict[str, any]:
    """Analyze earnings consistency and growth."""
    '''
    分析收益的一致性和增长。
    '''
    if len(financial_line_items) < 4:  # Need at least 4 periods for trend analysis # 需要至少4个时期进行趋势分析
        return {"score": 0, "details": "Insufficient historical data"}

    score = 0
    reasoning = []

    # Check earnings growth trend
    # 检查收益增长趋势
    earnings_values = [item.net_income for item in financial_line_items if item.net_income]
    if len(earnings_values) >= 4:
        # Simple check: is each period's earnings bigger than the next?
        # 简单检查：每个时期的收益是否都比下一个时期的收益大？
        earnings_growth = all(earnings_values[i] > earnings_values[i + 1] for i in range(len(earnings_values) - 1))

        if earnings_growth:
            score += 3
            reasoning.append("Consistent earnings growth over past periods")
        else:
            reasoning.append("Inconsistent earnings growth pattern")

        # Calculate total growth rate from oldest to latest
        # 计算从最早到最新的总增长率
        if len(earnings_values) >= 2 and earnings_values[-1] != 0:
            growth_rate = (earnings_values[0] - earnings_values[-1]) / abs(earnings_values[-1])
            reasoning.append(f"Total earnings growth of {growth_rate:.1%} over past {len(earnings_values)} periods")
    else:
        reasoning.append("Insufficient earnings data for trend analysis")

    return {
        "score": score,
        "details": "; ".join(reasoning),
    }


def analyze_moat(metrics: list) -> dict[str, any]:
    """
    Evaluate whether the company likely has a durable competitive advantage (moat).
    For simplicity, we look at stability of ROE/operating margins over multiple periods
    or high margin over the last few years. Higher stability => higher moat score.
    """
    '''
    评估公司是否可能具有持久的竞争优势（护城河）。
    为简单起见，我们查看ROE/运营利润率在多个时期的稳定性，
    或者最近几年的高利润率。更高的稳定性 => 更高的护城河得分。
    '''
    if not metrics or len(metrics) < 3:
        return {"score": 0, "max_score": 3, "details": "Insufficient data for moat analysis"}

    reasoning = []
    moat_score = 0
    historical_roes = []
    historical_margins = []

    for m in metrics:
        if m.return_on_equity is not None:
            historical_roes.append(m.return_on_equity)
        if m.operating_margin is not None:
            historical_margins.append(m.operating_margin)

    # Check for stable or improving ROE
    # 检查ROE的稳定性或改进
    if len(historical_roes) >= 3:
        stable_roe = all(r > 0.15 for r in historical_roes)
        if stable_roe:
            moat_score += 1
            reasoning.append("Stable ROE above 15% across periods (suggests moat)")
        else:
            reasoning.append("ROE not consistently above 15%")

    # Check for stable or improving operating margin
    # 检查运营利润率的稳定性或改进
    if len(historical_margins) >= 3:
        stable_margin = all(m > 0.15 for m in historical_margins)
        if stable_margin:
            moat_score += 1
            reasoning.append("Stable operating margins above 15% (moat indicator)")
        else:
            reasoning.append("Operating margin not consistently above 15%")

    # If both are stable/improving, add an extra point
    # 如果两者都稳定/改进，则再加一个点
    if moat_score == 2:
        moat_score += 1
        reasoning.append("Both ROE and margin stability indicate a solid moat")

    return {
        "score": moat_score,
        "max_score": 3,
        "details": "; ".join(reasoning),
    }


def analyze_management_quality(financial_line_items: list) -> dict[str, any]:
    """
    Checks for share dilution or consistent buybacks, and some dividend track record.
    A simplified approach:
      - if there's net share repurchase or stable share count, it suggests management
        might be shareholder-friendly.
      - if there's a big new issuance, it might be a negative sign (dilution).
    """
    '''
    检查股份稀释或持续回购，以及股息支付记录。
    简化的方法：
      - 如果有净股份回购或股份数量稳定，表明管理层
        可能对股东友好。
      - 如果有大规模新股发行，可能是负面信号（稀释）。
    '''
    if not financial_line_items:
        return {"score": 0, "max_score": 2, "details": "Insufficient data for management analysis"}

    reasoning = []
    mgmt_score = 0

    latest = financial_line_items[0]
    if hasattr(latest, "issuance_or_purchase_of_equity_shares") and latest.issuance_or_purchase_of_equity_shares and latest.issuance_or_purchase_of_equity_shares < 0:
        # Negative means the company spent money on buybacks
        # 负数意味着公司花了钱回购股票
        mgmt_score += 1
        reasoning.append("Company has been repurchasing shares (shareholder-friendly)")

    if hasattr(latest, "issuance_or_purchase_of_equity_shares") and latest.issuance_or_purchase_of_equity_shares and latest.issuance_or_purchase_of_equity_shares > 0:
        # Positive issuance means new shares => possible dilution
        # 正发行意味着新股票 => 可能的稀释
        reasoning.append("Recent common stock issuance (potential dilution)")
    else:
        reasoning.append("No significant new stock issuance detected")

    # Check for any dividends
    # 检查是否有股息
    if hasattr(latest, "dividends_and_other_cash_distributions") and latest.dividends_and_other_cash_distributions and latest.dividends_and_other_cash_distributions < 0:
        mgmt_score += 1
        reasoning.append("Company has a track record of paying dividends")
    else:
        reasoning.append("No or minimal dividends paid")

    return {
        "score": mgmt_score,
        "max_score": 2,
        "details": "; ".join(reasoning),
    }


def calculate_owner_earnings(financial_line_items: list) -> dict[str, any]:
    """Calculate owner earnings (Buffett's preferred measure of true earnings power).
    Owner Earnings = Net Income + Depreciation - Maintenance CapEx"""
    '''
    计算所有者收益（巴菲特衡量真实盈利能力的首选指标）。
    所有者收益 = 净利润 + 折旧 - 维护性资本支出
    '''
    if not financial_line_items or len(financial_line_items) < 1:
        return {"owner_earnings": None, "details": ["Insufficient data for owner earnings calculation"]}

    latest = financial_line_items[0]

    net_income = latest.net_income
    depreciation = latest.depreciation_and_amortization
    capex = latest.capital_expenditure

    if not all([net_income, depreciation, capex]):
        return {"owner_earnings": None, "details": ["Missing components for owner earnings calculation"]}

    # Estimate maintenance capex (typically 70-80% of total capex)
    # 估计维护性资本支出（通常为总资本支出的70-80%）
    maintenance_capex = capex * 0.75
    owner_earnings = net_income + depreciation - maintenance_capex

    return {
        "owner_earnings": owner_earnings,
        "components": {"net_income": net_income, "depreciation": depreciation, "maintenance_capex": maintenance_capex},
        "details": ["Owner earnings calculated successfully"],
    }


def calculate_intrinsic_value(financial_line_items: list) -> dict[str, any]:
    """Calculate intrinsic value using DCF with owner earnings."""
    '''
    使用所有者收益的DCF方法计算内在价值。
    '''
    if not financial_line_items:
        return {"intrinsic_value": None, "details": ["Insufficient data for valuation"]}

    # Calculate owner earnings
    # 计算所有者收益
    earnings_data = calculate_owner_earnings(financial_line_items)
    if not earnings_data["owner_earnings"]:
        return {"intrinsic_value": None, "details": earnings_data["details"]}

    owner_earnings = earnings_data["owner_earnings"]

    # Get current market data
    # 获取当前市场数据
    latest_financial_line_items = financial_line_items[0]
    shares_outstanding = latest_financial_line_items.outstanding_shares

    if not shares_outstanding:
        return {"intrinsic_value": None, "details": ["Missing shares outstanding data"]}

    # Buffett's DCF assumptions (conservative approach)
    # 巴菲特的DCF假设（保守方法）
    growth_rate = 0.05  # Conservative 5% growth # 保守的5%增长
    discount_rate = 0.09  # Typical ~9% discount rate # 典型的~9%折扣率
    terminal_multiple = 12
    projection_years = 10

    # Sum of discounted future owner earnings
    # 未来所有者收益的总和
    future_value = 0
    for year in range(1, projection_years + 1):
        future_earnings = owner_earnings * (1 + growth_rate) ** year
        present_value = future_earnings / (1 + discount_rate) ** year
        future_value += present_value

    # Terminal value
    # 终值
    terminal_value = (owner_earnings * (1 + growth_rate) ** projection_years * terminal_multiple) / ((1 + discount_rate) ** projection_years)

    intrinsic_value = future_value + terminal_value

    return {
        "intrinsic_value": intrinsic_value,
        "owner_earnings": owner_earnings,
        "assumptions": {
            "growth_rate": growth_rate,
            "discount_rate": discount_rate,
            "terminal_multiple": terminal_multiple,
            "projection_years": projection_years,
        },
        "details": ["Intrinsic value calculated using DCF model with owner earnings"],
    }

'''
你是一个沃伦·巴菲特AI代理。基于巴菲特的原则做出投资信号决策：
- 能力圈：只投资于你理解的企业
- 安全边际（> 30%）：以显著低于内在价值的价格买入
- 经济护城河：寻找持久的竞争优势
- 优质管理层：寻找保守的、以股东为导向的团队
- 财务实力：偏好低负债、高股本回报率
- 长期视角：投资企业，而不仅仅是股票
- 只有在基本面恶化或估值远超内在价值时才卖出

在提供推理时，要做到详尽和具体：
1. 解释最影响你决策的关键因素（正面和负面的都要）
2. 突出说明公司如何符合或违反巴菲特的具体原则
3. 在相关处提供量化证据（如具体的利润率、ROE值、债务水平）
4. 以巴菲特风格对投资机会做出总结评估
5. 使用巴菲特的语气和对话风格来解释

例如，如果看涨："我特别欣赏[具体优势]，这让我想起我们早期投资See's Candies时看到的[类似特征]..."
例如，如果看跌："资本回报率的下降让我想起我们最终退出的伯克希尔纺织业务，因为..."

严格遵循这些指导原则。
'''
def generate_buffett_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> WarrenBuffettSignal:
    """Get investment decision from LLM with Buffett's principles"""
    '''
    使用巴菲特的投资原则从LLM获取投资决策
    '''
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Warren Buffett AI agent. Decide on investment signals based on Warren Buffett's principles:
                - Circle of Competence: Only invest in businesses you understand
                - Margin of Safety (> 30%): Buy at a significant discount to intrinsic value
                - Economic Moat: Look for durable competitive advantages
                - Quality Management: Seek conservative, shareholder-oriented teams
                - Financial Strength: Favor low debt, strong returns on equity
                - Long-term Horizon: Invest in businesses, not just stocks
                - Sell only if fundamentals deteriorate or valuation far exceeds intrinsic value

                When providing your reasoning, be thorough and specific by:
                1. Explaining the key factors that influenced your decision the most (both positive and negative)
                2. Highlighting how the company aligns with or violates specific Buffett principles
                3. Providing quantitative evidence where relevant (e.g., specific margins, ROE values, debt levels)
                4. Concluding with a Buffett-style assessment of the investment opportunity
                5. Using Warren Buffett's voice and conversational style in your explanation

                For example, if bullish: "I'm particularly impressed with [specific strength], reminiscent of our early investment in See's Candies where we saw [similar attribute]..."
                For example, if bearish: "The declining returns on capital remind me of the textile operations at Berkshire that we eventually exited because..."

                Follow these guidelines strictly.
                """,
            ),
            (
                "human",
                """Based on the following data, create the investment signal as Warren Buffett would:

                Analysis Data for {ticker}:
                {analysis_data}

                Return the trading signal in the following JSON format exactly:
                {{
                  "signal": "bullish" | "bearish" | "neutral",
                  "confidence": float between 0 and 100,
                  "reasoning": "string"
                }}
                """,
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    # Default fallback signal in case parsing fails
    def create_default_warren_buffett_signal():
        return WarrenBuffettSignal(signal="neutral", confidence=0.0, reasoning="Error in analysis, defaulting to neutral")

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=WarrenBuffettSignal,
        agent_name="warren_buffett_agent",
        default_factory=create_default_warren_buffett_signal,
    )
