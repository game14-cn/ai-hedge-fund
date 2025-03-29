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

class CathieWoodSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str

'''
凯西·伍德代理
 - 成长投资女王，相信创新和颠覆的力量
'''
def cathie_wood_agent(state: AgentState):
    """
    Analyzes stocks using Cathie Wood's investing principles and LLM reasoning.
    1. Prioritizes companies with breakthrough technologies or business models
    2. Focuses on industries with rapid adoption curves and massive TAM (Total Addressable Market).
    3. Invests mostly in AI, robotics, genomic sequencing, fintech, and blockchain.
    4. Willing to endure short-term volatility for long-term gains.
    """
    '''
    使用凯西·伍德的投资原则和LLM推理分析股票。
    1. 优先考虑具有突破性技术或商业模式的公司
    2. 关注具有快速采用曲线和巨大总可寻址市场（TAM）的行业。
    3. 主要投资于人工智能、机器人、基因组测序、金融科技和区块链。
    4. 愿意承受短期波动以获取长期收益。
    '''
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    analysis_data = {}
    cw_analysis = {}

    for ticker in tickers:
        progress.update_status("cathie_wood_agent", ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=5)

        progress.update_status("cathie_wood_agent", ticker, "Gathering financial line items")
        # Request multiple periods of data (annual or TTM) for a more robust view.
        # 请求多个时期的数据（年度或TTM）以获得更稳健的视图
        financial_line_items = search_line_items(
            ticker,
            [
                "revenue",
                "gross_margin",
                "operating_margin",
                "debt_to_equity",
                "free_cash_flow",
                "total_assets",
                "total_liabilities",
                "dividends_and_other_cash_distributions",
                "outstanding_shares",
                "research_and_development",
                "capital_expenditure",
                "operating_expense",

            ],
            end_date,
            period="annual",
            limit=5
        )

        progress.update_status("cathie_wood_agent", ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date)

        progress.update_status("cathie_wood_agent", ticker, "Analyzing disruptive potential")
        disruptive_analysis = analyze_disruptive_potential(metrics, financial_line_items)

        progress.update_status("cathie_wood_agent", ticker, "Analyzing innovation-driven growth")
        innovation_analysis = analyze_innovation_growth(metrics, financial_line_items)

        progress.update_status("cathie_wood_agent", ticker, "Calculating valuation & high-growth scenario")
        valuation_analysis = analyze_cathie_wood_valuation(financial_line_items, market_cap)

        # Combine partial scores or signals
        # 结合部分分数或信号
        total_score = disruptive_analysis["score"] + innovation_analysis["score"] + valuation_analysis["score"]
        max_possible_score = 15  # Adjust weighting as desired

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
            "disruptive_analysis": disruptive_analysis,
            "innovation_analysis": innovation_analysis,
            "valuation_analysis": valuation_analysis
        }

        progress.update_status("cathie_wood_agent", ticker, "Generating Cathie Wood analysis")
        cw_output = generate_cathie_wood_output(
            ticker=ticker,
            analysis_data=analysis_data,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )

        cw_analysis[ticker] = {
            "signal": cw_output.signal,
            "confidence": cw_output.confidence,
            "reasoning": cw_output.reasoning
        }

        progress.update_status("cathie_wood_agent", ticker, "Done")

    message = HumanMessage(
        content=json.dumps(cw_analysis),
        name="cathie_wood_agent"
    )

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(cw_analysis, "Cathie Wood Agent")

    state["data"]["analyst_signals"]["cathie_wood_agent"] = cw_analysis

    return {
        "messages": [message],
        "data": state["data"]
    }


def analyze_disruptive_potential(metrics: list, financial_line_items: list) -> dict:
    """
    Analyze whether the company has disruptive products, technology, or business model.
    Evaluates multiple dimensions of disruptive potential:
    1. Revenue Growth Acceleration - indicates market adoption
    2. R&D Intensity - shows innovation investment
    3. Gross Margin Trends - suggests pricing power and scalability
    4. Operating Leverage - demonstrates business model efficiency
    5. Market Share Dynamics - indicates competitive position
    """
    '''
    分析公司是否具有颠覆性产品、技术或商业模式。
    评估颠覆性潜力的多个维度：
    1. 收入增长加速度 - 表明市场采用程度
    2. 研发强度 - 显示创新投资
    3. 毛利率趋势 - 表明定价能力和可扩展性
    4. 运营杠杆 - 展示商业模式效率
    5. 市场份额动态 - 表明竞争地位
    '''
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {
            "score": 0,
            "details": "Insufficient data to analyze disruptive potential"
        }

    # 1. Revenue Growth Analysis - Check for accelerating growth
    # 1. 收入增长分析 - 检查增长是否在加速
    revenues = [item.revenue for item in financial_line_items if item.revenue]
    if len(revenues) >= 3:  # Need at least 3 periods to check acceleration # 至少需要3个时期才能检查加速
        growth_rates = []
        for i in range(len(revenues)-1):
            if revenues[i] and revenues[i+1]:
                growth_rate = (revenues[i+1] - revenues[i]) / abs(revenues[i]) if revenues[i] != 0 else 0
                growth_rates.append(growth_rate)

        # Check if growth is accelerating
        # 检查是否正在加速增长
        if len(growth_rates) >= 2 and growth_rates[-1] > growth_rates[0]:
            score += 2
            details.append(f"Revenue growth is accelerating: {(growth_rates[-1]*100):.1f}% vs {(growth_rates[0]*100):.1f}%")

        # Check absolute growth rate
        # 检查绝对增长率
        latest_growth = growth_rates[-1] if growth_rates else 0
        if latest_growth > 1.0:
            score += 3
            details.append(f"Exceptional revenue growth: {(latest_growth*100):.1f}%")
        elif latest_growth > 0.5:
            score += 2
            details.append(f"Strong revenue growth: {(latest_growth*100):.1f}%")
        elif latest_growth > 0.2:
            score += 1
            details.append(f"Moderate revenue growth: {(latest_growth*100):.1f}%")
    else:
        details.append("Insufficient revenue data for growth analysis")

    # 2. Gross Margin Analysis - Check for expanding margins
    # 2. 毛利率分析 - 检查是否在扩大
    gross_margins = [item.gross_margin for item in financial_line_items if hasattr(item, 'gross_margin') and item.gross_margin is not None]
    if len(gross_margins) >= 2:
        margin_trend = gross_margins[-1] - gross_margins[0]
        if margin_trend > 0.05:  # 5% improvement
            score += 2
            details.append(f"Expanding gross margins: +{(margin_trend*100):.1f}%")
        elif margin_trend > 0:
            score += 1
            details.append(f"Slightly improving gross margins: +{(margin_trend*100):.1f}%")

        # Check absolute margin level
        # 检查绝对利润率
        if gross_margins[-1] > 0.50:  # High margin business    # 高利润率的企业
            score += 2
            details.append(f"High gross margin: {(gross_margins[-1]*100):.1f}%")
    else:
        details.append("Insufficient gross margin data")

    # 3. Operating Leverage Analysis
    # 3. 运营杠杆分析
    revenues = [item.revenue for item in financial_line_items if item.revenue]
    operating_expenses = [
        item.operating_expense
        for item in financial_line_items
        if hasattr(item, "operating_expense") and item.operating_expense
    ]

    if len(revenues) >= 2 and len(operating_expenses) >= 2:
        rev_growth = (revenues[-1] - revenues[0]) / abs(revenues[0])
        opex_growth = (operating_expenses[-1] - operating_expenses[0]) / abs(operating_expenses[0])

        if rev_growth > opex_growth:
            score += 2
            details.append("Positive operating leverage: Revenue growing faster than expenses")
    else:
        details.append("Insufficient data for operating leverage analysis")

    # 4. R&D Investment Analysis
    # 4. R&D投资分析
    rd_expenses = [item.research_and_development for item in financial_line_items if hasattr(item, 'research_and_development') and item.research_and_development is not None]
    if rd_expenses and revenues:
        rd_intensity = rd_expenses[-1] / revenues[-1]
        if rd_intensity > 0.15:  # High R&D intensity
            score += 3
            details.append(f"High R&D investment: {(rd_intensity*100):.1f}% of revenue")
        elif rd_intensity > 0.08:
            score += 2
            details.append(f"Moderate R&D investment: {(rd_intensity*100):.1f}% of revenue")
        elif rd_intensity > 0.05:
            score += 1
            details.append(f"Some R&D investment: {(rd_intensity*100):.1f}% of revenue")
    else:
        details.append("No R&D data available")

    # Normalize score to be out of 5
    # 将分数标准化为5分
    max_possible_score = 12  # Sum of all possible points    # 所有可能点的总和
    normalized_score = (score / max_possible_score) * 5

    return {
        "score": normalized_score,
        "details": "; ".join(details),
        "raw_score": score,
        "max_score": max_possible_score
    }


def analyze_innovation_growth(metrics: list, financial_line_items: list) -> dict:
    """
    Evaluate the company's commitment to innovation and potential for exponential growth.
    Analyzes multiple dimensions:
    1. R&D Investment Trends - measures commitment to innovation
    2. Free Cash Flow Generation - indicates ability to fund innovation
    3. Operating Efficiency - shows scalability of innovation
    4. Capital Allocation - reveals innovation-focused management
    5. Growth Reinvestment - demonstrates commitment to future growth
    """
    '''
    评估公司对创新的承诺和指数级增长的潜力。
    分析多个维度：
    1. 研发投资趋势 - 衡量对创新的承诺
    2. 自由现金流生成 - 表明资助创新的能力
    3. 运营效率 - 显示创新的可扩展性
    4. 资本分配 - 揭示以创新为重点的管理
    5. 增长再投资 - 展示对未来增长的承诺
    '''
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {
            "score": 0,
            "details": "Insufficient data to analyze innovation-driven growth"
        }

    # 1. R&D Investment Trends
    # 1. R&D投资趋势
    rd_expenses = [
        item.research_and_development
        for item in financial_line_items
        if hasattr(item, "research_and_development") and item.research_and_development
    ]
    revenues = [item.revenue for item in financial_line_items if item.revenue]

    if rd_expenses and revenues and len(rd_expenses) >= 2:
        # Check R&D growth rate
        # 检查R&D增长率
        rd_growth = (rd_expenses[-1] - rd_expenses[0]) / abs(rd_expenses[0]) if rd_expenses[0] != 0 else 0
        if rd_growth > 0.5:  # 50% growth in R&D    # 50%的R&D增长率
            score += 3
            details.append(f"Exceptional R&D investment growth: +{(rd_growth*100):.1f}%")
            score += 3
            details.append(f"Strong R&D investment growth: +{(rd_growth*100):.1f}%")
        elif rd_growth > 0.2:
            score += 2
            details.append(f"Moderate R&D investment growth: +{(rd_growth*100):.1f}%")

        # Check R&D intensity trend
        # 检查R&D强度趋势
        rd_intensity_start = rd_expenses[0] / revenues[0]
        rd_intensity_end = rd_expenses[-1] / revenues[-1]
        if rd_intensity_end > rd_intensity_start:
            score += 2
            details.append(f"Increasing R&D intensity: {(rd_intensity_end*100):.1f}% vs {(rd_intensity_start*100):.1f}%")
    else:
        details.append("Insufficient R&D data for trend analysis")

    # 2. Free Cash Flow Analysis
    # 2. 自由现金流分析
    fcf_vals = [item.free_cash_flow for item in financial_line_items if item.free_cash_flow]
    if fcf_vals and len(fcf_vals) >= 2:
        # Check FCF growth and consistency
        # 检查FCF增长和一致性
        fcf_growth = (fcf_vals[-1] - fcf_vals[0]) / abs(fcf_vals[0])
        positive_fcf_count = sum(1 for f in fcf_vals if f > 0)

        if fcf_growth > 0.3 and positive_fcf_count == len(fcf_vals):
            score += 3
            details.append("Strong and consistent FCF growth, excellent innovation funding capacity")
        elif positive_fcf_count >= len(fcf_vals) * 0.75:
            score += 2
            details.append("Consistent positive FCF, good innovation funding capacity")
        elif positive_fcf_count > len(fcf_vals) * 0.5:
            score += 1
            details.append("Moderately consistent FCF, adequate innovation funding capacity")
    else:
        details.append("Insufficient FCF data for analysis")

    # 3. Operating Efficiency Analysis
    # 3. 运营效率分析
    op_margin_vals = [item.operating_margin for item in financial_line_items if item.operating_margin]
    if op_margin_vals and len(op_margin_vals) >= 2:
        # Check margin improvement
        # 检查利润率的提高
        margin_trend = op_margin_vals[-1] - op_margin_vals[0]

        if op_margin_vals[-1] > 0.15 and margin_trend > 0:
            score += 3
            details.append(f"Strong and improving operating margin: {(op_margin_vals[-1]*100):.1f}%")
        elif op_margin_vals[-1] > 0.10:
            score += 2
            details.append(f"Healthy operating margin: {(op_margin_vals[-1]*100):.1f}%")
        elif margin_trend > 0:
            score += 1
            details.append("Improving operating efficiency")
    else:
        details.append("Insufficient operating margin data")

    # 4. Capital Allocation Analysis
    # 4. 资本分配分析
    capex = [item.capital_expenditure for item in financial_line_items if hasattr(item, 'capital_expenditure') and item.capital_expenditure]
    if capex and revenues and len(capex) >= 2:
        capex_intensity = abs(capex[-1]) / revenues[-1]
        capex_growth = (abs(capex[-1]) - abs(capex[0])) / abs(capex[0]) if capex[0] != 0 else 0

        if capex_intensity > 0.10 and capex_growth > 0.2:
            score += 2
            details.append("Strong investment in growth infrastructure")
        elif capex_intensity > 0.05:
            score += 1
            details.append("Moderate investment in growth infrastructure")
    else:
        details.append("Insufficient CAPEX data")

    # 5. Growth Reinvestment Analysis
    # 5. 增长再投资分析
    dividends = [item.dividends_and_other_cash_distributions for item in financial_line_items if hasattr(item, 'dividends_and_other_cash_distributions') and item.dividends_and_other_cash_distributions]
    if dividends and fcf_vals:
        # Check if company prioritizes reinvestment over dividends
        # 检查公司是否优先考虑再投资而不是分红
        latest_payout_ratio = dividends[-1] / fcf_vals[-1] if fcf_vals[-1] != 0 else 1
        if latest_payout_ratio < 0.2:  # Low dividend payout ratio suggests reinvestment focus # 低分红支付比率表明再投资的重点
            score += 2
            details.append("Strong focus on reinvestment over dividends")
        elif latest_payout_ratio < 0.4:
            score += 1
            details.append("Moderate focus on reinvestment over dividends")
    else:
        details.append("Insufficient dividend data")

    # Normalize score to be out of 5
    # 将分数标准化为5分
    max_possible_score = 15  # Sum of all possible points    # 所有可能点的总和
    normalized_score = (score / max_possible_score) * 5

    return {
        "score": normalized_score,
        "details": "; ".join(details),
        "raw_score": score,
        "max_score": max_possible_score
    }


def analyze_cathie_wood_valuation(financial_line_items: list, market_cap: float) -> dict:
    """
    Cathie Wood often focuses on long-term exponential growth potential. We can do
    a simplified approach looking for a large total addressable market (TAM) and the
    company's ability to capture a sizable portion.
    """
    '''
    凯西·伍德经常关注长期指数级增长潜力。我们可以采用
    一种简化的方法来寻找巨大的总可寻址市场（TAM）和
    公司获取可观市场份额的能力。
    '''
    if not financial_line_items or market_cap is None:
        return {
            "score": 0,
            "details": "Insufficient data for valuation"
        }

    latest = financial_line_items[-1]
    fcf = latest.free_cash_flow if latest.free_cash_flow else 0

    if fcf <= 0:
        return {
            "score": 0,
            "details": f"No positive FCF for valuation; FCF = {fcf}",
            "intrinsic_value": None
        }

    # Instead of a standard DCF, let's assume a higher growth rate for an innovative company.
    # 与其使用标准的DCF，我们假设一个创新公司的更高增长率。
    # Example values:
    growth_rate = 0.20  # 20% annual growth # 20%年增长率
    discount_rate = 0.15  # 15% discount rate # 15%折扣率
    terminal_multiple = 25  # Terminal value multiple # 终值倍数
    projection_years = 5  # Years to project # 要预测的年数
    # Present Value of Future Cash Flows
    # 未来现金流量的现值
    present_value = 0
    discount_rate = 0.15
    terminal_multiple = 25
    projection_years = 5

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

    margin_of_safety = (intrinsic_value - market_cap) / market_cap

    score = 0
    if margin_of_safety > 0.5:
        score += 3
    elif margin_of_safety > 0.2:
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
您是一个凯西·伍德 AI 代理，使用她的原则做出投资决策：

1. 寻找利用颠覆性创新的公司。
2. 强调指数级增长潜力和巨大的总可寻址市场（TAM）。
3. 专注于技术、医疗保健或其他面向未来的行业。
4. 考虑多年时间跨度的潜在突破。
5. 愿意承受较高波动性以追求高回报。
6. 评估管理层的愿景和投资研发的能力。

规则：
- 识别颠覆性或突破性技术。
- 评估多年收入增长的强大潜力。
- 检查公司是否能在大市场中有效扩张。
- 使用以增长为导向的估值方法。
- 提供数据驱动的建议（看涨、看跌或中性）。

在提供理由时，请通过以下方式进行全面而具体的分析：
1. 识别公司正在利用的具体颠覆性技术/创新
2. 突出显示表明指数级潜力的增长指标（收入加速、扩大的 TAM）
3. 讨论 5 年以上时间跨度的长期愿景和变革潜力
4. 解释公司如何可能颠覆传统行业或创造新市场
5. 关注可能推动未来增长的研发投资和创新管线
6. 使用凯西·伍德乐观、面向未来和具有坚定信念的语气

看涨示例："该公司的 AI 驱动平台正在改变 5000 亿美元的医疗保健分析市场，平台采用率同比从 40% 加速到 65%。他们占收入 22% 的研发投资正在创造技术护城河，使他们能够在这个不断扩大的市场中占据重要份额。目前的估值还未反映我们预期的指数级增长轨迹，因为..."
看跌示例："虽然在基因组领域运营，但该公司缺乏真正的颠覆性技术，仅仅是在逐步改进现有技术。研发支出仅占收入的 8%，表明对突破性创新的投资不足。收入增长从同比 45% 放缓至 20%，缺乏我们在变革性公司中寻找的指数级采用曲线的证据..."
'''
def generate_cathie_wood_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> CathieWoodSignal:
    """
    Generates investment decisions in the style of Cathie Wood.
    """
    '''
    以凯西·伍德的风格生成投资决策。
    '''
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Cathie Wood AI agent, making investment decisions using her principles:

            1. Seek companies leveraging disruptive innovation.
            2. Emphasize exponential growth potential, large TAM.
            3. Focus on technology, healthcare, or other future-facing sectors.
            4. Consider multi-year time horizons for potential breakthroughs.
            5. Accept higher volatility in pursuit of high returns.
            6. Evaluate management's vision and ability to invest in R&D.

            Rules:
            - Identify disruptive or breakthrough technology.
            - Evaluate strong potential for multi-year revenue growth.
            - Check if the company can scale effectively in a large market.
            - Use a growth-biased valuation approach.
            - Provide a data-driven recommendation (bullish, bearish, or neutral).
            
            When providing your reasoning, be thorough and specific by:
            1. Identifying the specific disruptive technologies/innovations the company is leveraging
            2. Highlighting growth metrics that indicate exponential potential (revenue acceleration, expanding TAM)
            3. Discussing the long-term vision and transformative potential over 5+ year horizons
            4. Explaining how the company might disrupt traditional industries or create new markets
            5. Addressing R&D investment and innovation pipeline that could drive future growth
            6. Using Cathie Wood's optimistic, future-focused, and conviction-driven voice
            
            For example, if bullish: "The company's AI-driven platform is transforming the $500B healthcare analytics market, with evidence of platform adoption accelerating from 40% to 65% YoY. Their R&D investments of 22% of revenue are creating a technological moat that positions them to capture a significant share of this expanding market. The current valuation doesn't reflect the exponential growth trajectory we expect as..."
            For example, if bearish: "While operating in the genomics space, the company lacks truly disruptive technology and is merely incrementally improving existing techniques. R&D spending at only 8% of revenue signals insufficient investment in breakthrough innovation. With revenue growth slowing from 45% to 20% YoY, there's limited evidence of the exponential adoption curve we look for in transformative companies..."
            """
        ),
        (
            "human",
            """Based on the following analysis, create a Cathie Wood-style investment signal.

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

    def create_default_cathie_wood_signal():
        return CathieWoodSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis, defaulting to neutral"
        )

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=CathieWoodSignal,
        agent_name="cathie_wood_agent",
        default_factory=create_default_cathie_wood_signal,
    )

# source: https://ark-invest.com