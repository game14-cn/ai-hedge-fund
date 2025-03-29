from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
import json

from tools.api import get_financial_metrics, get_market_cap, search_line_items


##### Valuation Agent #####
'''
估值代理
 - 计算股票内在价值并生成交易信号
'''
def valuation_agent(state: AgentState):
    """Performs detailed valuation analysis using multiple methodologies for multiple tickers."""
    '''
    使用多种方法对多个股票进行详细的估值分析。
    '''
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    # Initialize valuation analysis for each ticker
    # 初始化每个品种的估值分析
    valuation_analysis = {}

    for ticker in tickers:
        progress.update_status("valuation_agent", ticker, "Fetching financial data")

        # Fetch the financial metrics
        # 获取财务指标
        financial_metrics = get_financial_metrics(
            ticker=ticker,
            end_date=end_date,
            period="ttm",
        )

        # Add safety check for financial metrics
        # 添加财务指标的安全检查
        if not financial_metrics:
            progress.update_status("valuation_agent", ticker, "Failed: No financial metrics found")
            continue
        
        metrics = financial_metrics[0]

        progress.update_status("valuation_agent", ticker, "Gathering line items")
        # Fetch the specific line_items that we need for valuation purposes
        # 获取我们需要的特定财务指标
        financial_line_items = search_line_items(
            ticker=ticker,
            line_items=[
                "free_cash_flow",
                "net_income",
                "depreciation_and_amortization",
                "capital_expenditure",
                "working_capital",
            ],
            end_date=end_date,
            period="ttm",
            limit=2,
        )

        # Add safety check for financial line items
        # 添加财务指标的安全检查
        if len(financial_line_items) < 2:
            progress.update_status("valuation_agent", ticker, "Failed: Insufficient financial line items")
            continue

        # Pull the current and previous financial line items
        # 提取当前和上一个财务指标
        current_financial_line_item = financial_line_items[0]
        previous_financial_line_item = financial_line_items[1]

        progress.update_status("valuation_agent", ticker, "Calculating owner earnings")
        # Calculate working capital change
        # 计算工作资本变动
        working_capital_change = current_financial_line_item.working_capital - previous_financial_line_item.working_capital

        # Owner Earnings Valuation (Buffett Method)
        # 所有者权益估值（巴菲特方法）
        owner_earnings_value = calculate_owner_earnings_value(
            net_income=current_financial_line_item.net_income,
            depreciation=current_financial_line_item.depreciation_and_amortization,
            capex=current_financial_line_item.capital_expenditure,
            working_capital_change=working_capital_change,
            growth_rate=metrics.earnings_growth,
            required_return=0.15,
            margin_of_safety=0.25,
        )

        progress.update_status("valuation_agent", ticker, "Calculating DCF value")
        # DCF Valuation
        # 折现现金流估值
        dcf_value = calculate_intrinsic_value(
            free_cash_flow=current_financial_line_item.free_cash_flow,
            growth_rate=metrics.earnings_growth,
            discount_rate=0.10,
            terminal_growth_rate=0.03,
            num_years=5,
        )

        progress.update_status("valuation_agent", ticker, "Comparing to market value")
        # Get the market cap
        # 获取市值
        market_cap = get_market_cap(ticker=ticker, end_date=end_date)

        # Calculate combined valuation gap (average of both methods)
        # 计算两种估值方法的估值差距（平均值）
        dcf_gap = (dcf_value - market_cap) / market_cap
        owner_earnings_gap = (owner_earnings_value - market_cap) / market_cap
        valuation_gap = (dcf_gap + owner_earnings_gap) / 2

        if valuation_gap > 0.15:  # More than 15% undervalued # 超过15%的低估
            signal = "bullish"
        elif valuation_gap < -0.15:  # More than 15% overvalued # 超过15%的高估
            signal = "bearish"
        else:
            signal = "neutral"

        # Create the reasoning
        # 创建推理
        reasoning = {}
        reasoning["dcf_analysis"] = {
            "signal": ("bullish" if dcf_gap > 0.15 else "bearish" if dcf_gap < -0.15 else "neutral"),
            "details": f"Intrinsic Value: ${dcf_value:,.2f}, Market Cap: ${market_cap:,.2f}, Gap: {dcf_gap:.1%}",
        }

        reasoning["owner_earnings_analysis"] = {
            "signal": ("bullish" if owner_earnings_gap > 0.15 else "bearish" if owner_earnings_gap < -0.15 else "neutral"),
            "details": f"Owner Earnings Value: ${owner_earnings_value:,.2f}, Market Cap: ${market_cap:,.2f}, Gap: {owner_earnings_gap:.1%}",
        }

        confidence = round(abs(valuation_gap), 2) * 100
        valuation_analysis[ticker] = {
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        progress.update_status("valuation_agent", ticker, "Done")

    message = HumanMessage(
        content=json.dumps(valuation_analysis),
        name="valuation_agent",
    )

    # Print the reasoning if the flag is set
    # 打印推理
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(valuation_analysis, "Valuation Analysis Agent")

    # Add the signal to the analyst_signals list
    # 将信号添加到analyst_signals列表
    state["data"]["analyst_signals"]["valuation_agent"] = valuation_analysis

    return {
        "messages": [message],
        "data": data,
    }


def calculate_owner_earnings_value(
    net_income: float,
    depreciation: float,
    capex: float,
    working_capital_change: float,
    growth_rate: float = 0.05,
    required_return: float = 0.15,
    margin_of_safety: float = 0.25,
    num_years: int = 5,
) -> float:
    """
    Calculates the intrinsic value using Buffett's Owner Earnings method.

    Owner Earnings = Net Income
                    + Depreciation/Amortization
                    - Capital Expenditures
                    - Working Capital Changes

    Args:
        net_income: Annual net income
        depreciation: Annual depreciation and amortization
        capex: Annual capital expenditures
        working_capital_change: Annual change in working capital
        growth_rate: Expected growth rate
        required_return: Required rate of return (Buffett typically uses 15%)
        margin_of_safety: Margin of safety to apply to final value
        num_years: Number of years to project

    Returns:
        float: Intrinsic value with margin of safety
    """
    '''
    使用巴菲特的所有者收益法计算内在价值。

    所有者收益 = 净收入
                + 折旧/摊销
                - 资本支出
                - 营运资金变动

    参数:
        net_income: 年度净收入
        depreciation: 年度折旧和摊销
        capex: 年度资本支出
        working_capital_change: 年度营运资金变动
        growth_rate: 预期增长率
        required_return: 要求回报率（巴菲特通常使用15%）
        margin_of_safety: 应用于最终价值的安全边际
        num_years: 预测年数

    返回:
        float: 包含安全边际的内在价值
    '''
    if not all([isinstance(x, (int, float)) for x in [net_income, depreciation, capex, working_capital_change]]):
        return 0

    # Calculate initial owner earnings
    # 计算初始所有者收益
    owner_earnings = net_income + depreciation - capex - working_capital_change

    if owner_earnings <= 0:
        return 0

    # Project future owner earnings
    # 预测未来所有者收益
    future_values = []
    for year in range(1, num_years + 1):
        future_value = owner_earnings * (1 + growth_rate) ** year
        discounted_value = future_value / (1 + required_return) ** year
        future_values.append(discounted_value)

    # Calculate terminal value (using perpetuity growth formula)
    # 计算终值（使用 perpetuity growth 公式）
    terminal_growth = min(growth_rate, 0.03)  # Cap terminal growth at 3% # 将终值增长率限制在3%
    terminal_value = (future_values[-1] * (1 + terminal_growth)) / (required_return - terminal_growth)
    terminal_value_discounted = terminal_value / (1 + required_return) ** num_years

    # Sum all values and apply margin of safety
    # 计算所有值并应用安全边际
    intrinsic_value = sum(future_values) + terminal_value_discounted
    value_with_safety_margin = intrinsic_value * (1 - margin_of_safety)

    return value_with_safety_margin


def calculate_intrinsic_value(
    free_cash_flow: float,
    growth_rate: float = 0.05,
    discount_rate: float = 0.10,
    terminal_growth_rate: float = 0.02,
    num_years: int = 5,
) -> float:
    """
    Computes the discounted cash flow (DCF) for a given company based on the current free cash flow.
    Use this function to calculate the intrinsic value of a stock.
    """
    '''
    基于当前自由现金流计算公司的折现现金流(DCF)。
    使用此函数计算股票的内在价值。
    '''
    # Estimate the future cash flows based on the growth rate
    # 根据增长率估计未来现金流
    cash_flows = [free_cash_flow * (1 + growth_rate) ** i for i in range(num_years)]

    # Calculate the present value of projected cash flows
    # 计算未来现金流的现值
    present_values = []
    for i in range(num_years):
        present_value = cash_flows[i] / (1 + discount_rate) ** (i + 1)
        present_values.append(present_value)

    # Calculate the terminal value
    # 计算终值
    terminal_value = cash_flows[-1] * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
    terminal_present_value = terminal_value / (1 + discount_rate) ** num_years

    # Sum up the present values and terminal value
    # 计算现值和终值的总和
    dcf_value = sum(present_values) + terminal_present_value

    return dcf_value


def calculate_working_capital_change(
    current_working_capital: float,
    previous_working_capital: float,
) -> float:
    """
    Calculate the absolute change in working capital between two periods.
    A positive change means more capital is tied up in working capital (cash outflow).
    A negative change means less capital is tied up (cash inflow).

    Args:
        current_working_capital: Current period's working capital
        previous_working_capital: Previous period's working capital

    Returns:
        float: Change in working capital (current - previous)
    """
    '''
    计算两个期间之间的营运资金绝对变化。
    正变化意味着更多资金被占用在营运资金中（现金流出）。
    负变化意味着较少资金被占用（现金流入）。

    参数:
        current_working_capital: 当期营运资金
        previous_working_capital: 上期营运资金

    返回:
        float: 营运资金变化（当期 - 上期）
    '''
    return current_working_capital - previous_working_capital
