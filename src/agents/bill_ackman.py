from langchain_openai import ChatOpenAI
from graph.state import AgentState, show_agent_reasoning
from tools.api import get_financial_metrics, get_market_cap, search_line_items
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from utils.progress import progress
from utils.llm import call_llm

class BillAckmanSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str

'''
比尔·阿克曼代理
 - 激进投资者，采取大胆立场并推动变革
'''
def bill_ackman_agent(state: AgentState):
    """
    Analyzes stocks using Bill Ackman's investing principles and LLM reasoning.
    Fetches multiple periods of data so we can analyze long-term trends.
    """
    '''
    使用比尔·阿克曼的投资原则和 LLM 推理来分析股票。
    获取多个时期的数据以分析长期趋势。
    '''
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    
    analysis_data = {}
    ackman_analysis = {}
    
    for ticker in tickers:
        progress.update_status("bill_ackman_agent", ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=5)
        
        progress.update_status("bill_ackman_agent", ticker, "Gathering financial line items")
        # Request multiple periods of data (annual or TTM) for a more robust long-term view.
        # 请求多个时期的数据（年度或TTM）以获得更稳健的长期视角
        financial_line_items = search_line_items(
            ticker,
            [
                "revenue",
                "operating_margin",
                "debt_to_equity",
                "free_cash_flow",
                "total_assets",
                "total_liabilities",
                "dividends_and_other_cash_distributions",
                "outstanding_shares"
            ],
            end_date,
            period="annual",  # or "ttm" if you prefer trailing 12 months
            limit=5           # fetch up to 5 annual periods (or more if needed)
        )
        
        progress.update_status("bill_ackman_agent", ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date)
        
        progress.update_status("bill_ackman_agent", ticker, "Analyzing business quality")
        quality_analysis = analyze_business_quality(metrics, financial_line_items)
        
        progress.update_status("bill_ackman_agent", ticker, "Analyzing balance sheet and capital structure")
        balance_sheet_analysis = analyze_financial_discipline(metrics, financial_line_items)
        
        progress.update_status("bill_ackman_agent", ticker, "Calculating intrinsic value & margin of safety")
        valuation_analysis = analyze_valuation(financial_line_items, market_cap)
        
        # Combine partial scores or signals
        # 合并各项得分和信号
        total_score = quality_analysis["score"] + balance_sheet_analysis["score"] + valuation_analysis["score"]
        max_possible_score = 15  # Adjust weighting as desired
        
        # Generate a simple buy/hold/sell (bullish/neutral/bearish) signal
        # 生成简单的买入/持有/卖出（牛市/中性/熊市）信号
        if total_score >= 0.7 * max_possible_score:
            signal = "bullish"
        elif total_score <= 0.3 * max_possible_score:
            signal = "bearish"
        else:
            signal = "neutral"
        
        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_possible_score,
            "quality_analysis": quality_analysis,
            "balance_sheet_analysis": balance_sheet_analysis,
            "valuation_analysis": valuation_analysis
        }
        
        progress.update_status("bill_ackman_agent", ticker, "Generating Bill Ackman analysis")
        ackman_output = generate_ackman_output(
            ticker=ticker, 
            analysis_data=analysis_data,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )
        
        ackman_analysis[ticker] = {
            "signal": ackman_output.signal,
            "confidence": ackman_output.confidence,
            "reasoning": ackman_output.reasoning
        }
        
        progress.update_status("bill_ackman_agent", ticker, "Done")
    
    # Wrap results in a single message for the chain
    # 将结果包装在单个消息中
    message = HumanMessage(
        content=json.dumps(ackman_analysis),
        name="bill_ackman_agent"
    )
    
    # Show reasoning if requested
    # 显示推理
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(ackman_analysis, "Bill Ackman Agent")
    
    # Add signals to the overall state
    # 将信号添加到整体状态
    state["data"]["analyst_signals"]["bill_ackman_agent"] = ackman_analysis

    return {
        "messages": [message],
        "data": state["data"]
    }


def analyze_business_quality(metrics: list, financial_line_items: list) -> dict:
    """
    Analyze whether the company has a high-quality business with stable or growing cash flows,
    durable competitive advantages, and potential for long-term growth.
    """
    '''
    分析公司是否具有高质量的业务，包括：
    - 稳定或增长的现金流
    - 持久的竞争优势
    - 长期增长潜力
    '''
    score = 0
    details = []
    
    if not metrics or not financial_line_items:
        return {
            "score": 0,
            "details": "Insufficient data to analyze business quality"
        }
    
    # 1. Multi-period revenue growth analysis
    # 多期收入增长分析
    revenues = [item.revenue for item in financial_line_items if item.revenue is not None]
    if len(revenues) >= 2:
        # Check if overall revenue grew from first to last
        # 检查整体收入是否从第一年增长到最后一年
        initial, final = revenues[0], revenues[-1]
        if initial and final and final > initial:
            # Simple growth rate
            # 简单增长率
            growth_rate = (final - initial) / abs(initial)
            if growth_rate > 0.5:  # e.g., 50% growth over the available time
                score += 2
                details.append(f"Revenue grew by {(growth_rate*100):.1f}% over the full period.")
            else:
                score += 1
                details.append(f"Revenue growth is positive but under 50% cumulatively ({(growth_rate*100):.1f}%).")
        else:
            details.append("Revenue did not grow significantly or data insufficient.")
    else:
        details.append("Not enough revenue data for multi-period trend.")
    
    # 2. Operating margin and free cash flow consistency
    # We'll check if operating_margin or free_cash_flow are consistently positive/improving
    # 运营利润率和自由现金流一致性
    # 我们将检查运营利润率或自由现金流是否在大多数时期都是积极的/增长的
    fcf_vals = [item.free_cash_flow for item in financial_line_items if item.free_cash_flow is not None]
    op_margin_vals = [item.operating_margin for item in financial_line_items if item.operating_margin is not None]
    
    if op_margin_vals:
        # Check if the majority of operating margins are > 15%
        # 我们将检查运营利润率是否在大多数时期都超过15%
        above_15 = sum(1 for m in op_margin_vals if m > 0.15)
        if above_15 >= (len(op_margin_vals) // 2 + 1):
            score += 2
            details.append("Operating margins have often exceeded 15%.")
        else:
            details.append("Operating margin not consistently above 15%.")
    else:
        details.append("No operating margin data across periods.")
    
    if fcf_vals:
        # Check if free cash flow is positive in most periods
        # 我们将检查自由现金流是否在大多数时期都是积极的
        positive_fcf_count = sum(1 for f in fcf_vals if f > 0)
        if positive_fcf_count >= (len(fcf_vals) // 2 + 1):
            score += 1
            details.append("Majority of periods show positive free cash flow.")
        else:
            details.append("Free cash flow not consistently positive.")
    else:
        details.append("No free cash flow data across periods.")
    
    # 3. Return on Equity (ROE) check from the latest metrics
    # (If you want multi-period ROE, you'd need that in financial_line_items as well.)
    # 股本回报率（ROE）检查，使用最新的指标
    # （如果你想要多期ROE，你需要在financial_line_items中也包含这个指标）
    latest_metrics = metrics[0]
    if latest_metrics.return_on_equity and latest_metrics.return_on_equity > 0.15:
        score += 2
        details.append(f"High ROE of {latest_metrics.return_on_equity:.1%}, indicating potential moat.")
    elif latest_metrics.return_on_equity:
        details.append(f"ROE of {latest_metrics.return_on_equity:.1%} is not indicative of a strong moat.")
    else:
        details.append("ROE data not available in metrics.")
    
    return {
        "score": score,
        "details": "; ".join(details)
    }


def analyze_financial_discipline(metrics: list, financial_line_items: list) -> dict:
    """
    Evaluate the company's balance sheet over multiple periods:
    - Debt ratio trends
    - Capital returns to shareholders over time (dividends, buybacks)
    """
    '''
    评估公司在多个时期的资产负债表：
    - 债务比率趋势
    - 随时间推移对股东的资本回报（股息、回购）
    '''
    score = 0
    details = []
    
    if not metrics or not financial_line_items:
        return {
            "score": 0,
            "details": "Insufficient data to analyze financial discipline"
        }
    
    # 1. Multi-period debt ratio or debt_to_equity
    # Check if the company's leverage is stable or improving
    # 多期债务比率或股本回报率
    # 检查公司的杠杆是否稳定或提高
    debt_to_equity_vals = [item.debt_to_equity for item in financial_line_items if item.debt_to_equity is not None]
    
    # If we have multi-year data, see if D/E ratio has gone down or stayed <1 across most periods
    # 如果我们有多年的数据，检查负债权益比是否下降或在大多数时期保持在1以下
    if debt_to_equity_vals:
        below_one_count = sum(1 for d in debt_to_equity_vals if d < 1.0)
        if below_one_count >= (len(debt_to_equity_vals) // 2 + 1):
            score += 2
            details.append("Debt-to-equity < 1.0 for the majority of periods.")
        else:
            details.append("Debt-to-equity >= 1.0 in many periods.")
    else:
        # Fallback to total_liabilities/total_assets if D/E not available
        # 如果D/E不可用，回退到负债/资产比率
        liab_to_assets = []
        for item in financial_line_items:
            if item.total_liabilities and item.total_assets and item.total_assets > 0:
                liab_to_assets.append(item.total_liabilities / item.total_assets)
        
        if liab_to_assets:
            below_50pct_count = sum(1 for ratio in liab_to_assets if ratio < 0.5)
            if below_50pct_count >= (len(liab_to_assets) // 2 + 1):
                score += 2
                details.append("Liabilities-to-assets < 50% for majority of periods.")
            else:
                details.append("Liabilities-to-assets >= 50% in many periods.")
        else:
            details.append("No consistent leverage ratio data available.")
    
    # 2. Capital allocation approach (dividends + share counts)
    # If the company paid dividends or reduced share count over time, it may reflect discipline
    # 资本分配方法（股息+股本）
    # 如果公司支付股息或减少股本，它可能反映纪律性
    dividends_list = [item.dividends_and_other_cash_distributions for item in financial_line_items if item.dividends_and_other_cash_distributions is not None]
    if dividends_list:
        # Check if dividends were paid (i.e., negative outflows to shareholders) in most periods
        # 检查股东是否支付了股息（即，向股东支付了负的流出）在大多数时期
        paying_dividends_count = sum(1 for d in dividends_list if d < 0)
        if paying_dividends_count >= (len(dividends_list) // 2 + 1):
            score += 1
            details.append("Company has a history of returning capital to shareholders (dividends).")
        else:
            details.append("Dividends not consistently paid or no data.")
    else:
        details.append("No dividend data found across periods.")
    
    # Check for decreasing share count (simple approach):
    # We can compare first vs last if we have at least two data points
    # 检查股本是否在多年来减少（简单的方法）
    # 我们可以比较第一年和最后一年，如果我们至少有两个数据点
    shares = [item.outstanding_shares for item in financial_line_items if item.outstanding_shares is not None]
    if len(shares) >= 2:
        if shares[-1] < shares[0]:
            score += 1
            details.append("Outstanding shares have decreased over time (possible buybacks).")
        else:
            details.append("Outstanding shares have not decreased over the available periods.")
    else:
        details.append("No multi-period share count data to assess buybacks.")
    
    return {
        "score": score,
        "details": "; ".join(details)
    }


def analyze_valuation(financial_line_items: list, market_cap: float) -> dict:
    """
    Ackman invests in companies trading at a discount to intrinsic value.
    We can do a simplified DCF or an FCF-based approach.
    This function currently uses the latest free cash flow only, 
    but you could expand it to use an average or multi-year FCF approach.
    """
    '''
    阿克曼投资于股价低于内在价值的公司。
    我们可以做一个简化的DCF或基于FCF的方法。
    此函数目前仅使用最新的自由现金流，但您可以扩展它以使用平均或多年的FCF方法。
    '''
    score = 0
    details = []
    if not financial_line_items or market_cap is None:
        return {
            "score": 0,
            "details": "Insufficient data to perform valuation"
        }
    
    # Example: use the most recent item for FCF
    # 示例：使用最新的项目进行FCF
    latest = financial_line_items[-1]  # the last one is presumably the most recent
    fcf = latest.free_cash_flow if latest.free_cash_flow else 0
    
    # For demonstration, let's do a naive approach:
    # 对于演示，让我们做一个简单的方法：
    growth_rate = 0.06
    discount_rate = 0.10
    terminal_multiple = 15
    projection_years = 5
    
    if fcf <= 0:
        return {
            "score": 0,
            "details": f"No positive FCF for valuation; FCF = {fcf}",
            "intrinsic_value": None
        }
    
    present_value = 0
    for year in range(1, projection_years + 1):
        future_fcf = fcf * (1 + growth_rate) ** year
        pv = future_fcf / ((1 + discount_rate) ** year)
        present_value += pv
    
    # Terminal Value
    # 终值
    terminal_value = (fcf * (1 + growth_rate) ** projection_years * terminal_multiple) \
                     / ((1 + discount_rate) ** projection_years)
    intrinsic_value = present_value + terminal_value
    
    # Compare with market cap => margin of safety
    # 与市值进行比较 => 保证金安全
    margin_of_safety = (intrinsic_value - market_cap) / market_cap
    
    score = 0
    if margin_of_safety > 0.3:
        score += 3
    elif margin_of_safety > 0.1:
        score += 1
    
    details = [
        f"Calculated intrinsic value: ~{intrinsic_value:,.2f}",
        f"Market cap: ~{market_cap:,.2f}",
        f"Margin of safety: {margin_of_safety:.2%}"
    ]
    
    return {
        "score": score,
        "details": "; ".join(details),
        "intrinsic_value": intrinsic_value,
        "margin_of_safety": margin_of_safety
    }

'''
你是一个Bill Ackman AI代理，使用他的原则做出投资决策：

1. 寻找具有持久竞争优势（护城河）的高质量企业。
2. 优先考虑稳定的自由现金流和增长潜力。
3. 倡导强有力的财务纪律（合理的杠杆率，高效的资本配置）。
4. 重视估值：关注内在价值和安全边际。
5. 长期持有集中的投资组合，具有高度信念。
6. 如果管理层或运营改善能释放价值，可能采取激进投资者方式。


规则：
- 评估品牌实力、市场地位或其他护城河。
- 检查自由现金流生成能力，稳定或增长的收益。
- 分析资产负债表健康状况（合理的债务，良好的ROE）。
- 以低于内在价值的折扣价买入；折扣越高 => 信念越强。
- 如果管理层表现不佳或存在战略改进的机会，则进行干预。
- 提供基于数据的理性建议（看涨、看跌或中性）。

在提供推理时，要做到全面和具体：
1. 详细解释业务质量和竞争优势
2. 突出影响决策的具体财务指标（自由现金流、利润率、杠杆率）
3. 讨论运营改善或管理层变更的潜力
4. 提供有数据支持的清晰估值评估
5. 识别可能释放价值的具体催化剂
6. 使用Bill Ackman自信、分析性且有时具有对抗性的风格

看涨示例："该企业产生了15%利润率的优异自由现金流，并拥有竞争对手难以复制的主导市场地位。目前仅以12倍自由现金流交易，相对于内在价值有40%的折扣，而且管理层最近的资本配置决策表明..."
看跌示例："尽管市场地位尚可，但自由现金流利润率在三年内从12%下降到8%。管理层继续通过追求低ROIC的收购做出糟糕的资本配置决策。考虑到运营挑战，目前18倍自由现金流的估值没有提供安全边际..."
'''
def generate_ackman_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> BillAckmanSignal:
    """
    Generates investment decisions in the style of Bill Ackman.
    """
    '''
    生成Bill Ackman的投资决策
    '''
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Bill Ackman AI agent, making investment decisions using his principles:

            1. Seek high-quality businesses with durable competitive advantages (moats).
            2. Prioritize consistent free cash flow and growth potential.
            3. Advocate for strong financial discipline (reasonable leverage, efficient capital allocation).
            4. Valuation matters: target intrinsic value and margin of safety.
            5. Invest with high conviction in a concentrated portfolio for the long term.
            6. Potential activist approach if management or operational improvements can unlock value.
            

            Rules:
            - Evaluate brand strength, market position, or other moats.
            - Check free cash flow generation, stable or growing earnings.
            - Analyze balance sheet health (reasonable debt, good ROE).
            - Buy at a discount to intrinsic value; higher discount => stronger conviction.
            - Engage if management is suboptimal or if there's a path for strategic improvements.
            - Provide a rational, data-driven recommendation (bullish, bearish, or neutral).
            
            When providing your reasoning, be thorough and specific by:
            1. Explaining the quality of the business and its competitive advantages in detail
            2. Highlighting the specific financial metrics that most influenced your decision (FCF, margins, leverage)
            3. Discussing any potential for operational improvements or management changes
            4. Providing a clear valuation assessment with numerical evidence
            5. Identifying specific catalysts that could unlock value
            6. Using Bill Ackman's confident, analytical, and sometimes confrontational style
            
            For example, if bullish: "This business generates exceptional free cash flow with a 15% margin and has a dominant market position that competitors can't easily replicate. Trading at only 12x FCF, there's a 40% discount to intrinsic value, and management's recent capital allocation decisions suggest..."
            For example, if bearish: "Despite decent market position, FCF margins have deteriorated from 12% to 8% over three years. Management continues to make poor capital allocation decisions by pursuing low-ROIC acquisitions. Current valuation at 18x FCF provides no margin of safety given the operational challenges..."
            """
        ),
        (
            "human",
            """Based on the following analysis, create an Ackman-style investment signal.
        
            Analysis Data for {ticker}:
            {analysis_data}

            Return the trading signal in this JSON format:
            {{
              "signal": "bullish/bearish/neutral",
              "confidence": float (0-100),
              "reasoning": "string"
            }}
            """
        )
    ])

    prompt = template.invoke({
        "analysis_data": json.dumps(analysis_data, indent=2),
        "ticker": ticker
    })

    def create_default_bill_ackman_signal():
        return BillAckmanSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis, defaulting to neutral"
        )

    return call_llm(
        prompt=prompt, 
        model_name=model_name, 
        model_provider=model_provider, 
        pydantic_model=BillAckmanSignal, 
        agent_name="bill_ackman_agent", 
        default_factory=create_default_bill_ackman_signal,
    )
