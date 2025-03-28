from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
from tools.api import get_prices, prices_to_df
import json

##### 风险管理代理 #####
def risk_management_agent(state: AgentState):
    # 基于多个交易品种的实际风险因素控制仓位大小。
    """Controls position sizing based on real-world risk factors for multiple tickers."""
    portfolio = state["data"]["portfolio"]
    data = state["data"]
    tickers = data["tickers"]

    # Initialize risk analysis for each ticker
    # 初始化每个品种的风险分析
    risk_analysis = {}
    # Store prices here to avoid redundant API calls
    # 存储价格以避免重复的API调用
    current_prices = {}  

    for ticker in tickers:
        progress.update_status("risk_management_agent", ticker, "Analyzing price data")

        prices = get_prices(
            ticker=ticker,
            start_date=data["start_date"],
            end_date=data["end_date"],
        )

        if not prices:
            progress.update_status("risk_management_agent", ticker, "Failed: No price data found")
            continue

        prices_df = prices_to_df(prices)

        progress.update_status("risk_management_agent", ticker, "Calculating position limits")

        # Calculate portfolio value
        # 计算投资组合价值
        current_price = prices_df["close"].iloc[-1]
        current_prices[ticker] = current_price  # Store the current price

        # Calculate current position value for this ticker
        # 计算当前仓位价值
        current_position_value = portfolio.get("cost_basis", {}).get(ticker, 0)

        # Calculate total portfolio value using stored prices
        # 使用存储的价格计算总投资组合价值
        total_portfolio_value = portfolio.get("cash", 0) + sum(portfolio.get("cost_basis", {}).get(t, 0) for t in portfolio.get("cost_basis", {}))

        # Base limit is 20% of portfolio for any single position
        # 基础限制是投资组合的20%，任何单个位置
        position_limit = total_portfolio_value * 0.20

        # For existing positions, subtract current position value from limit
        # 对于现有仓位，从限制中减去当前仓位价值
        remaining_position_limit = position_limit - current_position_value

        # Ensure we don't exceed available cash
        # 确保我们不会超过可用现金
        max_position_size = min(remaining_position_limit, portfolio.get("cash", 0))

        risk_analysis[ticker] = {
            "remaining_position_limit": float(max_position_size),
            "current_price": float(current_price),
            "reasoning": {
                "portfolio_value": float(total_portfolio_value),
                "current_position": float(current_position_value),
                "position_limit": float(position_limit),
                "remaining_limit": float(remaining_position_limit),
                "available_cash": float(portfolio.get("cash", 0)),
            },
        }

        progress.update_status("risk_management_agent", ticker, "Done")

    message = HumanMessage(
        content=json.dumps(risk_analysis),
        name="risk_management_agent",
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(risk_analysis, "Risk Management Agent")

    # Add the signal to the analyst_signals list
    # 将信号添加到analyst_signals列表
    state["data"]["analyst_signals"]["risk_management_agent"] = risk_analysis

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }
