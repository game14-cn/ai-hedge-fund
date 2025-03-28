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
import math


class BenGrahamSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str

'''
本杰明·格雷厄姆代理
 - 价值投资之父，只买具有安全边际的隐藏瑰宝
'''
def ben_graham_agent(state: AgentState):
    """
    使用本杰明·格雷厄姆的经典价值投资原则分析股票：
    1. 多年的盈利稳定性。
    2. 稳健的财务实力（低负债，充足的流动性）。
    3. 相对于内在价值的折价（例如格雷厄姆数或净净值）。
    4. 充足的安全边际。
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    analysis_data = {}
    graham_analysis = {}

    for ticker in tickers:
        progress.update_status("ben_graham_agent", ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=10)

        progress.update_status("ben_graham_agent", ticker, "Gathering financial line items")
        financial_line_items = search_line_items(ticker, ["earnings_per_share", "revenue", "net_income", "book_value_per_share", "total_assets", "total_liabilities", "current_assets", "current_liabilities", "dividends_and_other_cash_distributions", "outstanding_shares"], end_date, period="annual", limit=10)

        progress.update_status("ben_graham_agent", ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date)

        # Perform sub-analyses
        # 执行子分析
        progress.update_status("ben_graham_agent", ticker, "Analyzing earnings stability")
        earnings_analysis = analyze_earnings_stability(metrics, financial_line_items)

        progress.update_status("ben_graham_agent", ticker, "Analyzing financial strength")
        strength_analysis = analyze_financial_strength(metrics, financial_line_items)

        progress.update_status("ben_graham_agent", ticker, "Analyzing Graham valuation")
        valuation_analysis = analyze_valuation_graham(metrics, financial_line_items, market_cap)

        # Aggregate scoring
        # 聚合评分
        total_score = earnings_analysis["score"] + strength_analysis["score"] + valuation_analysis["score"]
        max_possible_score = 15  # total possible from the three analysis functions

        # Map total_score to signal
        # 将总分映射到信号
        if total_score >= 0.7 * max_possible_score:
            signal = "bullish"
        elif total_score <= 0.3 * max_possible_score:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {"signal": signal, "score": total_score, "max_score": max_possible_score, "earnings_analysis": earnings_analysis, "strength_analysis": strength_analysis, "valuation_analysis": valuation_analysis}

        progress.update_status("ben_graham_agent", ticker, "Generating Ben Graham analysis")
        graham_output = generate_graham_output(
            ticker=ticker,
            analysis_data=analysis_data,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )

        graham_analysis[ticker] = {"signal": graham_output.signal, "confidence": graham_output.confidence, "reasoning": graham_output.reasoning}

        progress.update_status("ben_graham_agent", ticker, "Done")

    # Wrap results in a single message for the chain
    # 将结果包装在单个消息中
    message = HumanMessage(content=json.dumps(graham_analysis), name="ben_graham_agent")

    # Optionally display reasoning
    # 显示推理过程
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(graham_analysis, "Ben Graham Agent")

    # Store signals in the overall state
    # 将信号存储在整体状态中
    state["data"]["analyst_signals"]["ben_graham_agent"] = graham_analysis

    return {"messages": [message], "data": state["data"]}

"""
格雷厄姆要求至少有几年持续的正盈利（理想情况下5年以上）。
我们将检查：
1. 每股收益为正的年数。
2. 从第一期到最后一期的每股收益增长情况。
"""
def analyze_earnings_stability(metrics: list, financial_line_items: list) -> dict:
    """
    Graham wants at least several years of consistently positive earnings (ideally 5+).
    We'll check:
    1. Number of years with positive EPS.
    2. Growth in EPS from first to last period.
    """
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": score, "details": "Insufficient data for earnings stability analysis"}

    eps_vals = []
    for item in financial_line_items:
        if item.earnings_per_share is not None:
            eps_vals.append(item.earnings_per_share)

    if len(eps_vals) < 2:
        details.append("Not enough multi-year EPS data.")
        return {"score": score, "details": "; ".join(details)}

    # 1. Consistently positive EPS
    #   计算每股收益为正的年数
    positive_eps_years = sum(1 for e in eps_vals if e > 0)
    total_eps_years = len(eps_vals)
    if positive_eps_years == total_eps_years:
        score += 3
        details.append("EPS was positive in all available periods.")
    elif positive_eps_years >= (total_eps_years * 0.8):
        score += 2
        details.append("EPS was positive in most periods.")
    else:
        details.append("EPS was negative in multiple periods.")

    # 2. EPS growth from earliest to latest
    #   计算从第一年到最后一年每股收益的增长情况
    if eps_vals[-1] > eps_vals[0]:
        score += 1
        details.append("EPS grew from earliest to latest period.")
    else:
        details.append("EPS did not grow from earliest to latest period.")

    return {"score": score, "details": "; ".join(details)}

"""
格雷厄姆检查流动性（流动比率 >= 2）、可控的债务水平，
以及股息记录（最好有一定的派息历史）。
"""
def analyze_financial_strength(metrics: list, financial_line_items: list) -> dict:
    """
    Graham checks liquidity (current ratio >= 2), manageable debt,
    and dividend record (preferably some history of dividends).
    """
    score = 0
    details = []

    if not financial_line_items:
        return {"score": score, "details": "No data for financial strength analysis"}

    latest_item = financial_line_items[-1]
    total_assets = latest_item.total_assets or 0
    total_liabilities = latest_item.total_liabilities or 0
    current_assets = latest_item.current_assets or 0
    current_liabilities = latest_item.current_liabilities or 0

    # 1. Current ratio
    #   流动比率 = 流动资产 / 流动负债
    if current_liabilities > 0:
        current_ratio = current_assets / current_liabilities
        if current_ratio >= 2.0:
            score += 2
            details.append(f"Current ratio = {current_ratio:.2f} (>=2.0: solid).")
        elif current_ratio >= 1.5:
            score += 1
            details.append(f"Current ratio = {current_ratio:.2f} (moderately strong).")
        else:
            details.append(f"Current ratio = {current_ratio:.2f} (<1.5: weaker liquidity).")
    else:
        details.append("Cannot compute current ratio (missing or zero current_liabilities).")

    # 2. Debt vs. Assets
    #   资产负债率 = 总负债 / 总资产
    if total_assets > 0:
        debt_ratio = total_liabilities / total_assets
        if debt_ratio < 0.5:
            score += 2
            details.append(f"Debt ratio = {debt_ratio:.2f}, under 0.50 (conservative).")
        elif debt_ratio < 0.8:
            score += 1
            details.append(f"Debt ratio = {debt_ratio:.2f}, somewhat high but could be acceptable.")
        else:
            details.append(f"Debt ratio = {debt_ratio:.2f}, quite high by Graham standards.")
    else:
        details.append("Cannot compute debt ratio (missing total_assets).")

    # 3. Dividend track record
    #   检查公司是否在这些期间内派息
    div_periods = [item.dividends_and_other_cash_distributions for item in financial_line_items if item.dividends_and_other_cash_distributions is not None]
    if div_periods:
        # In many data feeds, dividend outflow is shown as a negative number
        # (money going out to shareholders). We'll consider any negative as 'paid a dividend'.
        # 在许多数据源中，股息支出显示为负数
        # （支付给股东的现金）。我们将任何负数视为"已支付股息"。
        div_paid_years = sum(1 for d in div_periods if d < 0)
        if div_paid_years > 0:
            # e.g. if at least half the periods had dividends
            # （至少一半的期间内派息）
            if div_paid_years >= (len(div_periods) // 2 + 1):
                score += 1
                details.append("Company paid dividends in the majority of the reported years.")
            else:
                details.append("Company has some dividend payments, but not most years.")
        else:
            details.append("Company did not pay dividends in these periods.")
    else:
        details.append("No dividend data available to assess payout consistency.")

    return {"score": score, "details": "; ".join(details)}

"""
格雷厄姆估值的核心方法：
1. 净净值检查：（流动资产 - 总负债）与市值比较
2. 格雷厄姆数：sqrt(22.5 * 每股收益 * 每股账面价值)
3. 将每股价格与格雷厄姆数比较 => 安全边际
"""
def analyze_valuation_graham(metrics: list, financial_line_items: list, market_cap: float) -> dict:
    """
    Core Graham approach to valuation:
    1. Net-Net Check: (Current Assets - Total Liabilities) vs. Market Cap
    2. Graham Number: sqrt(22.5 * EPS * Book Value per Share)
    3. Compare per-share price to Graham Number => margin of safety
    """
    if not financial_line_items or not market_cap or market_cap <= 0:
        return {"score": 0, "details": "Insufficient data to perform valuation"}

    latest = financial_line_items[-1]
    current_assets = latest.current_assets or 0
    total_liabilities = latest.total_liabilities or 0
    book_value_ps = latest.book_value_per_share or 0
    eps = latest.earnings_per_share or 0
    shares_outstanding = latest.outstanding_shares or 0

    details = []
    score = 0

    # 1. 净净值检查
    #   净流动资产价值 = 流动资产 - 总负债
    #   如果净流动资产价值 > 市值 => 历史上这是一个强烈的买入信号
    net_current_asset_value = current_assets - total_liabilities
    if net_current_asset_value > 0 and shares_outstanding > 0:
        net_current_asset_value_per_share = net_current_asset_value / shares_outstanding
        price_per_share = market_cap / shares_outstanding if shares_outstanding else 0

        details.append(f"Net Current Asset Value = {net_current_asset_value:,.2f}")
        details.append(f"NCAV Per Share = {net_current_asset_value_per_share:,.2f}")
        details.append(f"Price Per Share = {price_per_share:,.2f}")

        if net_current_asset_value > market_cap:
            score += 4  # Very strong Graham signal
            details.append("Net-Net: NCAV > Market Cap (classic Graham deep value).")
        else:
            # For partial net-net discount
            # 部分净净值折扣
            if net_current_asset_value_per_share >= (price_per_share * 0.67):
                score += 2
                details.append("NCAV Per Share >= 2/3 of Price Per Share (moderate net-net discount).")
    else:
        details.append("NCAV not exceeding market cap or insufficient data for net-net approach.")

    # 2. 格雷厄姆数
    #   格雷厄姆数 = sqrt(22.5 * 每股收益 * 每股账面价值)
    #   将结果与当前每股价格进行比较
    #   如果格雷厄姆数远大于价格，表明股票被低估
    graham_number = None
    if eps > 0 and book_value_ps > 0:
        graham_number = math.sqrt(22.5 * eps * book_value_ps)
        details.append(f"Graham Number = {graham_number:.2f}")
    else:
        details.append("Unable to compute Graham Number (EPS or Book Value missing/<=0).")

    # 3. 相对于格雷厄姆数的安全边际
    if graham_number and shares_outstanding > 0:
        current_price = market_cap / shares_outstanding
        if current_price > 0:
            margin_of_safety = (graham_number - current_price) / current_price
            details.append(f"Margin of Safety (Graham Number) = {margin_of_safety:.2%}")
            if margin_of_safety > 0.5:
                score += 3
                details.append("Price is well below Graham Number (>=50% margin).")
            elif margin_of_safety > 0.2:
                score += 1
                details.append("Some margin of safety relative to Graham Number.")
            else:
                details.append("Price close to or above Graham Number, low margin of safety.")
        else:
            details.append("Current price is zero or invalid; can't compute margin of safety.")
    # else: already appended details for missing graham_number

    return {"score": score, "details": "; ".join(details)}

'''
你是一个本杰明·格雷厄姆 AI 代理，使用他的原则做出投资决策：
1. 坚持通过低于内在价值的买入价格来确保安全边际（例如，使用格雷厄姆数、净净值）。
2. 强调公司的财务实力（低杠杆、充足的流动资产）。
3. 偏好多年稳定的收益。
4. 考虑股息记录以增加额外的安全性。
5. 避免投机或高增长假设；专注于已被证实的指标。

在提供你的理由时，要做到详尽且具体：
1. 解释最影响你决策的关键估值指标（格雷厄姆数、净流动资产价值、市盈率等）
2. 突出具体的财务实力指标（流动比率、债务水平等）
3. 引用收益随时间的稳定性或不稳定性
4. 提供精确的数字作为量化证据
5. 将当前指标与格雷厄姆的具体阈值进行比较（例如："流动比率 2.5 超过了格雷厄姆要求的最低 2.0"）
6. 使用本杰明·格雷厄姆保守、分析性的语气和风格来解释

看涨示例："该股票相对于净流动资产价值有 35% 的折价，提供了充足的安全边际。2.5 的流动比率和 0.3 的资产负债率表明财务状况良好..."
看跌示例："尽管收益稳定，但当前 50 美元的价格超过了我们计算的 35 美元格雷厄姆数，没有提供安全边际。此外，仅为 1.2 的流动比率低于格雷厄姆偏好的 2.0 阈值..."
            
返回一个理性的建议：看涨、看跌或中性，并附带置信度（0-100）和详细的理由。
'''
def generate_graham_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> BenGrahamSignal:
    """
    按照本杰明·格雷厄姆的风格生成投资决策：
    - 重视价值、安全边际、净净值、稳健的资产负债表和稳定的收益。
    - 返回 JSON 结构的结果：{ signal（信号）, confidence（置信度）, reasoning（理由） }。
    """

    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Benjamin Graham AI agent, making investment decisions using his principles:
            1. Insist on a margin of safety by buying below intrinsic value (e.g., using Graham Number, net-net).
            2. Emphasize the company's financial strength (low leverage, ample current assets).
            3. Prefer stable earnings over multiple years.
            4. Consider dividend record for extra safety.
            5. Avoid speculative or high-growth assumptions; focus on proven metrics.
            
            When providing your reasoning, be thorough and specific by:
            1. Explaining the key valuation metrics that influenced your decision the most (Graham Number, NCAV, P/E, etc.)
            2. Highlighting the specific financial strength indicators (current ratio, debt levels, etc.)
            3. Referencing the stability or instability of earnings over time
            4. Providing quantitative evidence with precise numbers
            5. Comparing current metrics to Graham's specific thresholds (e.g., "Current ratio of 2.5 exceeds Graham's minimum of 2.0")
            6. Using Benjamin Graham's conservative, analytical voice and style in your explanation
            
            For example, if bullish: "The stock trades at a 35% discount to net current asset value, providing an ample margin of safety. The current ratio of 2.5 and debt-to-equity of 0.3 indicate strong financial position..."
            For example, if bearish: "Despite consistent earnings, the current price of $50 exceeds our calculated Graham Number of $35, offering no margin of safety. Additionally, the current ratio of only 1.2 falls below Graham's preferred 2.0 threshold..."
                        
            Return a rational recommendation: bullish, bearish, or neutral, with a confidence level (0-100) and thorough reasoning.
            """
        ),
        """
        基于以下分析，创建一个格雷厄姆风格的投资信号：

        {ticker} 的分析数据：
        {analysis_data}

        返回的 JSON 必须严格按照以下格式：
        {
            "signal": "bullish"（看涨）或 "bearish"（看跌）或 "neutral"（中性）,
            "confidence": float（0-100 的浮点数）,
            "reasoning": "string"（分析理由）
        }
        """
        (
            "human",
            """Based on the following analysis, create a Graham-style investment signal:

            Analysis Data for {ticker}:
            {analysis_data}

            Return JSON exactly in this format:
            {{
              "signal": "bullish" or "bearish" or "neutral",
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

    def create_default_ben_graham_signal():
        return BenGrahamSignal(signal="neutral", confidence=0.0, reasoning="Error in generating analysis; defaulting to neutral.")

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=BenGrahamSignal,
        agent_name="ben_graham_agent",
        default_factory=create_default_ben_graham_signal,
    )
