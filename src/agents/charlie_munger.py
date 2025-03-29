from graph.state import AgentState, show_agent_reasoning
from tools.api import get_financial_metrics, get_market_cap, search_line_items, get_insider_trades, get_company_news
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from utils.progress import progress
from utils.llm import call_llm

class CharlieMungerSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str

'''
查理·芒格代理
 - 沃伦·巴菲特的合伙人，只在合理价格买入优质企业
'''
def charlie_munger_agent(state: AgentState):
    """
    Analyzes stocks using Charlie Munger's investing principles and mental models.
    Focuses on moat strength, management quality, predictability, and valuation.
    """
    '''
    使用查理·芒格的投资原则和思维模型来分析股票。
    重点关注护城河强度、管理层质量、可预测性和估值。
    '''
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    
    analysis_data = {}
    munger_analysis = {}
    
    for ticker in tickers:
        progress.update_status("charlie_munger_agent", ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=10)  # Munger looks at longer periods
        
        progress.update_status("charlie_munger_agent", ticker, "Gathering financial line items")
        financial_line_items = search_line_items(
            ticker,
            [
                "revenue",
                "net_income",
                "operating_income",
                "return_on_invested_capital",
                "gross_margin",
                "operating_margin",
                "free_cash_flow",
                "capital_expenditure",
                "cash_and_equivalents",
                "total_debt",
                "shareholders_equity",
                "outstanding_shares",
                "research_and_development",
                "goodwill_and_intangible_assets",
            ],
            end_date,
            period="annual",
            limit=10  # Munger examines long-term trends    # 芒格会研究长期趋势
        )
        
        progress.update_status("charlie_munger_agent", ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date)
        
        progress.update_status("charlie_munger_agent", ticker, "Fetching insider trades")
        # Munger values management with skin in the game
        # 芒格重视管理层与公司利益一致
        insider_trades = get_insider_trades(
            ticker,
            end_date,
            # Look back 2 years for insider trading patterns
            # 回溯2年查看内部交易模式
            start_date=None,
            limit=100
        )
        
        progress.update_status("charlie_munger_agent", ticker, "Fetching company news")
        # Munger avoids businesses with frequent negative press
        # 芒格避免频繁负面新闻的公司
        company_news = get_company_news(
            ticker,
            end_date,
            # Look back 1 year for news
            # 回溯1年的新闻
            start_date=None,
            limit=100
        )
        
        progress.update_status("charlie_munger_agent", ticker, "Analyzing moat strength")
        moat_analysis = analyze_moat_strength(metrics, financial_line_items)
        
        progress.update_status("charlie_munger_agent", ticker, "Analyzing management quality")
        management_analysis = analyze_management_quality(financial_line_items, insider_trades)
        
        progress.update_status("charlie_munger_agent", ticker, "Analyzing business predictability")
        predictability_analysis = analyze_predictability(financial_line_items)
        
        progress.update_status("charlie_munger_agent", ticker, "Calculating Munger-style valuation")
        valuation_analysis = calculate_munger_valuation(financial_line_items, market_cap)
        
        # Combine partial scores with Munger's weighting preferences
        # Munger weights quality and predictability higher than current valuation
        # Munger的权重偏好质量和预测能力高于当前估值
        # 权重偏好质量和预测能力高于当前估值
        total_score = (
            moat_analysis["score"] * 0.35 +
            management_analysis["score"] * 0.25 +
            predictability_analysis["score"] * 0.25 +
            valuation_analysis["score"] * 0.15
        )
        
        max_possible_score = 10  # Scale to 0-10 # 0-10 得分
        
        # Generate a simple buy/hold/sell signal # 生成一个简单的买入/持有/卖出信号
        if total_score >= 7.5:  # Munger has very high standards # 芒格的标准非常高
            signal = "buy"
        if total_score >= 7.5:  # Munger has very high standards # 芒格的标准非常高
            signal = "hold"
            signal = "bullish"
        elif total_score <= 4.5:
            signal = "bearish"
        else:
            signal = "neutral"
        
        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_possible_score,
            "moat_analysis": moat_analysis,
            "management_analysis": management_analysis,
            "predictability_analysis": predictability_analysis,
            "valuation_analysis": valuation_analysis,
            # Include some qualitative assessment from news # 包含一些定性的新闻分析
            "news_sentiment": analyze_news_sentiment(company_news) if company_news else "No news data available"
        }
        
        progress.update_status("charlie_munger_agent", ticker, "Generating Charlie Munger analysis")
        munger_output = generate_munger_output(
            ticker=ticker, 
            analysis_data=analysis_data,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )
        
        munger_analysis[ticker] = {
            "signal": munger_output.signal,
            "confidence": munger_output.confidence,
            "reasoning": munger_output.reasoning
        }
        
        progress.update_status("charlie_munger_agent", ticker, "Done")
    
    # Wrap results in a single message for the chain
    # 包装结果在一个单独的消息中
    message = HumanMessage(
        content=json.dumps(munger_analysis),
        name="charlie_munger_agent"
    )
    
    # Show reasoning if requested
    # 显示推理
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(munger_analysis, "Charlie Munger Agent")
    
    # Add signals to the overall state
    # 添加信号到整体状态
    state["data"]["analyst_signals"]["charlie_munger_agent"] = munger_analysis

    return {
        "messages": [message],
        "data": state["data"]
    }


def analyze_moat_strength(metrics: list, financial_line_items: list) -> dict:
    """
    Analyze the business's competitive advantage using Munger's approach:
    - Consistent high returns on capital (ROIC)
    - Pricing power (stable/improving gross margins)
    - Low capital requirements
    - Network effects and intangible assets (R&D investments, goodwill)
    """
    '''
    使用芒格的方法分析企业的竞争优势：
    - 持续的高资本回报率(ROIC)
    - 定价能力(稳定/提升的毛利率)
    - 低资本需求
    - 网络效应和无形资产(研发投入、商誉)
    '''
    score = 0
    details = []
    
    if not metrics or not financial_line_items:
        return {
            "score": 0,
            "details": "Insufficient data to analyze moat strength"
        }
    
    # 1. Return on Invested Capital (ROIC) analysis - Munger's favorite metric
    # 1. 投资资本回报率(ROIC)分析 - 芒格最喜欢的指标
    roic_values = [item.return_on_invested_capital for item in financial_line_items 
                   if hasattr(item, 'return_on_invested_capital') and item.return_on_invested_capital is not None]
    
    if roic_values:
        # Check if ROIC consistently above 15% (Munger's threshold)
        # 检查ROIC在15%以上的持续时间
        high_roic_count = sum(1 for r in roic_values if r > 0.15)
        if high_roic_count >= len(roic_values) * 0.8:  # 80% of periods show high ROIC # 80%的周期显示高ROIC
            score += 3
            details.append(f"Excellent ROIC: >15% in {high_roic_count}/{len(roic_values)} periods")
        elif high_roic_count >= len(roic_values) * 0.5:  # 50% of periods # 50%的周期
            score += 2
            details.append(f"Good ROIC: >15% in {high_roic_count}/{len(roic_values)} periods")
        elif high_roic_count > 0:
            score += 1
            details.append(f"Mixed ROIC: >15% in only {high_roic_count}/{len(roic_values)} periods")
        else:
            details.append("Poor ROIC: Never exceeds 15% threshold")
    else:
        details.append("No ROIC data available")
    
    # 2. Pricing power - check gross margin stability and trends
    # 2.定价能力 - 检查毛利率的稳定性和趋势
    gross_margins = [item.gross_margin for item in financial_line_items 
                    if hasattr(item, 'gross_margin') and item.gross_margin is not None]
    
    if gross_margins and len(gross_margins) >= 3:
        # Munger likes stable or improving gross margins
        # 芒格喜欢稳定或提升的毛利率
        margin_trend = sum(1 for i in range(1, len(gross_margins)) if gross_margins[i] >= gross_margins[i-1])
        if margin_trend >= len(gross_margins) * 0.7:  # Improving in 70% of periods # 在70%的周期中
            score += 2
            details.append("Strong pricing power: Gross margins consistently improving")
        elif sum(gross_margins) / len(gross_margins) > 0.3:  # Average margin > 30%
            score += 1
            details.append(f"Good pricing power: Average gross margin {sum(gross_margins)/len(gross_margins):.1%}")
        else:
            details.append("Limited pricing power: Low or declining gross margins")
    else:
        details.append("Insufficient gross margin data")
    
    # 3. Capital intensity - Munger prefers low capex businesses
    # 3.资本强度 - 芒格更喜欢低资本需求的企业
    if len(financial_line_items) >= 3:
        capex_to_revenue = []
        for item in financial_line_items:
            if (hasattr(item, 'capital_expenditure') and item.capital_expenditure is not None and 
                hasattr(item, 'revenue') and item.revenue is not None and item.revenue > 0):
                # Note: capital_expenditure is typically negative in financial statements
                # 资本支出通常在财务报表中是负数
                capex_ratio = abs(item.capital_expenditure) / item.revenue
                capex_to_revenue.append(capex_ratio)
        
        if capex_to_revenue:
            avg_capex_ratio = sum(capex_to_revenue) / len(capex_to_revenue)
            if avg_capex_ratio < 0.05:  # Less than 5% of revenue # 收入的5%
                score += 2
                details.append(f"Low capital requirements: Avg capex {avg_capex_ratio:.1%} of revenue")
            elif avg_capex_ratio < 0.10:  # Less than 10% of revenue # 收入的10%
                score += 1
                details.append(f"Moderate capital requirements: Avg capex {avg_capex_ratio:.1%} of revenue")
            else:
                details.append(f"High capital requirements: Avg capex {avg_capex_ratio:.1%} of revenue")
        else:
            details.append("No capital expenditure data available")
    else:
        details.append("Insufficient data for capital intensity analysis")
    
    # 4. Intangible assets - Munger values R&D and intellectual property
    # 4.无形资产 - 芒格重视研发和知识产权
    r_and_d = [item.research_and_development for item in financial_line_items
              if hasattr(item, 'research_and_development') and item.research_and_development is not None]
    
    goodwill_and_intangible_assets = [item.goodwill_and_intangible_assets for item in financial_line_items
               if hasattr(item, 'goodwill_and_intangible_assets') and item.goodwill_and_intangible_assets is not None]

    if r_and_d and len(r_and_d) > 0:
        if sum(r_and_d) > 0:  # If company is investing in R&D # 如果公司在研发
            score += 1
            details.append("Invests in R&D, building intellectual property")
    
    if (goodwill_and_intangible_assets and len(goodwill_and_intangible_assets) > 0):
        score += 1
        details.append("Significant goodwill/intangible assets, suggesting brand value or IP")
    
    # Scale score to 0-10 range
    # 缩放得分到0-10范围
    final_score = min(10, score * 10 / 9)  # Max possible raw score is 9 # 最大可能的原始得分是9
    
    return {
        "score": final_score,
        "details": "; ".join(details)
    }


def analyze_management_quality(financial_line_items: list, insider_trades: list) -> dict:
    """
    Evaluate management quality using Munger's criteria:
    - Capital allocation wisdom
    - Insider ownership and transactions
    - Cash management efficiency
    - Candor and transparency
    - Long-term focus
    """
    '''
    使用芒格的标准评估管理层质量：
    - 资本配置的智慧
    - 内部人持股和交易
    - 现金管理效率
    - 坦诚和透明度
    - 长期关注度
    '''
    score = 0
    details = []
    
    if not financial_line_items:
        return {
            "score": 0,
            "details": "Insufficient data to analyze management quality"
        }
    
    # 1. Capital allocation - Check FCF to net income ratio
    # Munger values companies that convert earnings to cash
    # 1. 资本配置 - 检查FCF与净利润的比率
    # 芒格重视将收益转化为现金的公司
    fcf_values = [item.free_cash_flow for item in financial_line_items 
                 if hasattr(item, 'free_cash_flow') and item.free_cash_flow is not None]
    
    net_income_values = [item.net_income for item in financial_line_items 
                        if hasattr(item, 'net_income') and item.net_income is not None]
    
    if fcf_values and net_income_values and len(fcf_values) == len(net_income_values):
        # Calculate FCF to Net Income ratio for each period
        # 计算每个期间的FCF与净利润的比率
        fcf_to_ni_ratios = []
        for i in range(len(fcf_values)):
            if net_income_values[i] and net_income_values[i] > 0:
                fcf_to_ni_ratios.append(fcf_values[i] / net_income_values[i])
        
        if fcf_to_ni_ratios:
            avg_ratio = sum(fcf_to_ni_ratios) / len(fcf_to_ni_ratios)
            if avg_ratio > 1.1:  # FCF > net income suggests good accounting # FCF大于净利润，表明良好的会计
                score += 3
                details.append(f"Excellent debt management: FCF/NI ratio of {avg_ratio:.2f}")
                score += 3
                details.append(f"Excellent cash conversion: FCF/NI ratio of {avg_ratio:.2f}")
            elif avg_ratio > 0.9:  # FCF roughly equals net income # FCF大致等于净利润
                score += 2
                details.append(f"Good cash conversion: FCF/NI ratio of {avg_ratio:.2f}")
            elif avg_ratio > 0.7:  # FCF somewhat lower than net income # FCF略低于净利润
                score += 1
                details.append(f"Moderate cash conversion: FCF/NI ratio of {avg_ratio:.2f}")
            else:
                details.append(f"Poor cash conversion: FCF/NI ratio of only {avg_ratio:.2f}")
        else:
            details.append("Could not calculate FCF to Net Income ratios")
    else:
        details.append("Missing FCF or Net Income data")
    
    # 2. Debt management - Munger is cautious about debt
    # 2. 债务管理 - 芒格对债务非常谨慎
    debt_values = [item.total_debt for item in financial_line_items 
                  if hasattr(item, 'total_debt') and item.total_debt is not None]
    
    equity_values = [item.shareholders_equity for item in financial_line_items 
                    if hasattr(item, 'shareholders_equity') and item.shareholders_equity is not None]
    
    if debt_values and equity_values and len(debt_values) == len(equity_values):
        # Calculate D/E ratio for most recent period
        # 计算最近的D/E比率
        recent_de_ratio = debt_values[0] / equity_values[0] if equity_values[0] > 0 else float('inf')
        
        if recent_de_ratio < 0.3:  # Very low debt # 非常低的债务
            score += 3
            details.append(f"Conservative debt management: D/E ratio of {recent_de_ratio:.2f}")
        elif recent_de_ratio < 0.7:  # Moderate debt # 适度的债务
            score += 2
            details.append(f"Prudent debt management: D/E ratio of {recent_de_ratio:.2f}")
        elif recent_de_ratio < 1.5:  # Higher but still reasonable debt # 较高但仍然合理的债务
            score += 1
            details.append(f"Moderate debt management: D/E ratio of {recent_de_ratio:.2f}")
            score += 1
            details.append(f"Moderate debt level: D/E ratio of {recent_de_ratio:.2f}")
        else:
            details.append(f"High debt level: D/E ratio of {recent_de_ratio:.2f}")
    else:
        details.append("Missing debt or equity data")
    
    # 3. Cash management efficiency - Munger values appropriate cash levels
    # 3. 现金管理效率 - 芒格重视适当的现金水平
    cash_values = [item.cash_and_equivalents for item in financial_line_items
                  if hasattr(item, 'cash_and_equivalents') and item.cash_and_equivalents is not None]
    revenue_values = [item.revenue for item in financial_line_items
                     if hasattr(item, 'revenue') and item.revenue is not None]
    
    if cash_values and revenue_values and len(cash_values) > 0 and len(revenue_values) > 0:
        # Calculate cash to revenue ratio (Munger likes 10-20% for most businesses)
        # 计算现金与收入比率(芒格喜欢大多数企业的10-20%现金储备)
        cash_to_revenue = cash_values[0] / revenue_values[0] if revenue_values[0] > 0 else 0
        
        if 0.1 <= cash_to_revenue <= 0.25:
            # Goldilocks zone - not too much, not too little
            #  Goldilocks区域 - 不多不少
            score += 2
            details.append(f"Prudent cash management: Cash/Revenue ratio of {cash_to_revenue:.2f}")
        elif 0.05 <= cash_to_revenue < 0.1 or 0.25 < cash_to_revenue <= 0.4:
            # Reasonable but not ideal
            # 合理但不理想
            score += 1
            details.append(f"Acceptable cash position: Cash/Revenue ratio of {cash_to_revenue:.2f}")
        elif cash_to_revenue > 0.4:
            # Too much cash - potentially inefficient capital allocation
            # 现金过多 - 潜在的低效资本配置
            details.append(f"Excess cash reserves: Cash/Revenue ratio of {cash_to_revenue:.2f}")
        else:
            # Too little cash - potentially risky
            # 现金过少 - 潜在的风险
            details.append(f"Low cash reserves: Cash/Revenue ratio of {cash_to_revenue:.2f}")
    else:
        details.append("Insufficient cash or revenue data")
    
    # 4. Insider activity - Munger values skin in the game
    # 4. 内部人员活动 - 芒格重视内部人员
    if insider_trades and len(insider_trades) > 0:
        # Count buys vs. sells
        # 计算买入和卖出
        buys = sum(1 for trade in insider_trades if hasattr(trade, 'transaction_type') and 
                   trade.transaction_type and trade.transaction_type.lower() in ['buy', 'purchase'])
        sells = sum(1 for trade in insider_trades if hasattr(trade, 'transaction_type') and 
                    trade.transaction_type and trade.transaction_type.lower() in ['sell', 'sale'])
        
        # Calculate the buy ratio
        # 计算买入比率
        total_trades = buys + sells
        if total_trades > 0:
            buy_ratio = buys / total_trades
            if buy_ratio > 0.7:  # Strong insider buying # 强烈的内部人员购买
                score += 2
                details.append(f"Strong insider buying: {buys}/{total_trades} transactions are purchases")
            elif buy_ratio > 0.4:  # Balanced insider activity # 平衡的内部人员活动
                score += 1
                details.append(f"Balanced insider trading: {buys}/{total_trades} transactions are purchases")
            elif buy_ratio < 0.1 and sells > 5:  # Heavy selling # 大量的卖出
                score -= 1  # Penalty for excessive selling # 卖出过多的惩罚
                details.append(f"Concerning insider selling: {sells}/{total_trades} transactions are sales")
            else:
                details.append(f"Mixed insider activity: {buys}/{total_trades} transactions are purchases")
        else:
            details.append("No recorded insider transactions")
    else:
        details.append("No insider trading data available")
    
    # 5. Consistency in share count - Munger prefers stable/decreasing shares
    # 5. 股份数量的一致性 - 芒格更喜欢稳定或减少的股份
    share_counts = [item.outstanding_shares for item in financial_line_items
                   if hasattr(item, 'outstanding_shares') and item.outstanding_shares is not None]
    
    if share_counts and len(share_counts) >= 3:
        if share_counts[0] < share_counts[-1] * 0.95:  # 5%+ reduction in shares # 5%+的股份减少
            score += 2
            details.append("Shareholder-friendly: Reducing share count over time")
        elif share_counts[0] < share_counts[-1] * 1.05:  # Stable share count # 稳定的股份数量
            score += 1
            details.append("Stable share count: Limited dilution")
        elif share_counts[0] > share_counts[-1] * 1.2:  # >20% dilution # >20%的稀释
            score -= 1  # Penalty for excessive dilution # 过度稀释的惩罚
            details.append("Concerning dilution: Share count increased significantly")
        else:
            details.append("Moderate share count increase over time")
    else:
        details.append("Insufficient share count data")
    
    # Scale score to 0-10 range
    # Maximum possible raw score would be 12 (3+3+2+2+2)
    # 缩放得分到0-10范围
    # 最大可能的原始得分是12(3+3+2+2+2)
    final_score = max(0, min(10, score * 10 / 12))
    
    return {
        "score": final_score,
        "details": "; ".join(details)
    }


def analyze_predictability(financial_line_items: list) -> dict:
    """
    Assess the predictability of the business - Munger strongly prefers businesses
    whose future operations and cashflows are relatively easy to predict.
    """
    '''
    评估企业的可预测性 - 芒格强烈偏好那些
    未来运营和现金流相对容易预测的企业。
    '''
    score = 0
    details = []
    
    if not financial_line_items or len(financial_line_items) < 5:
        return {
            "score": 0,
            "details": "Insufficient data to analyze business predictability (need 5+ years)"
        }
    
    # 1. Revenue stability and growth
    # 1. 收入稳定性和增长
    revenues = [item.revenue for item in financial_line_items 
               if hasattr(item, 'revenue') and item.revenue is not None]
    
    if revenues and len(revenues) >= 5:
        # Calculate year-over-year growth rates
        # 计算年增长率
        growth_rates = [(revenues[i] / revenues[i+1] - 1) for i in range(len(revenues)-1)]
        
        avg_growth = sum(growth_rates) / len(growth_rates)
        growth_volatility = sum(abs(r - avg_growth) for r in growth_rates) / len(growth_rates)
        
        if avg_growth > 0.05 and growth_volatility < 0.1:
            # Steady, consistent growth (Munger loves this)
            # 稳定，一致的增长(芒格非常喜欢)
            score += 3
            details.append(f"Highly predictable revenue: {avg_growth:.1%} avg growth with low volatility")
        elif avg_growth > 0 and growth_volatility < 0.2:
            # Positive but somewhat volatile growth
            # 正增长，但波动较大
            score += 2
            details.append(f"Moderately predictable revenue: {avg_growth:.1%} avg growth with some volatility")
        elif avg_growth > 0:
            # Growing but unpredictable
            # 增长但不可预测
            score += 1
            details.append(f"Growing but less predictable revenue: {avg_growth:.1%} avg growth with high volatility")
        else:
            details.append(f"Declining or highly unpredictable revenue: {avg_growth:.1%} avg growth")
    else:
        details.append("Insufficient revenue history for predictability analysis")
    
    # 2. Operating income stability
    # 2. 经营收入稳定性
    op_income = [item.operating_income for item in financial_line_items 
                if hasattr(item, 'operating_income') and item.operating_income is not None]
    
    if op_income and len(op_income) >= 5:
        # Count positive operating income periods
        # 计算正的经营收入期间
        positive_periods = sum(1 for income in op_income if income > 0)
        
        if positive_periods == len(op_income):
            # Consistently profitable operations
            # 稳定的经营收入
            score += 3
            details.append("Highly predictable operations: Operating income positive in all periods")
        elif positive_periods >= len(op_income) * 0.8:
            # Mostly profitable operations
            # 大部分的经营收入
            score += 2
            details.append(f"Predictable operations: Operating income positive in {positive_periods}/{len(op_income)} periods")
        elif positive_periods >= len(op_income) * 0.6:
            # Somewhat profitable operations
            # 一些盈利的操作
            score += 1
            details.append(f"Somewhat predictable operations: Operating income positive in {positive_periods}/{len(op_income)} periods")
        else:
            details.append(f"Unpredictable operations: Operating income positive in only {positive_periods}/{len(op_income)} periods")
    else:
        details.append("Insufficient operating income history")
    
    # 3. Margin consistency - Munger values stable margins
    # 3. 利润率的一致性 - 芒格重视稳定的利润率
    op_margins = [item.operating_margin for item in financial_line_items 
                 if hasattr(item, 'operating_margin') and item.operating_margin is not None]
    
    if op_margins and len(op_margins) >= 5:
        # Calculate margin volatility
        # 计算利润率的波动
        avg_margin = sum(op_margins) / len(op_margins)
        margin_volatility = sum(abs(m - avg_margin) for m in op_margins) / len(op_margins)
        
        if margin_volatility < 0.03:  # Very stable margins # 非常稳定的利润率
            score += 2
            details.append(f"Highly predictable margins: {avg_margin:.1%} avg with minimal volatility")
        elif margin_volatility < 0.07:  # Moderately stable margins # 适度稳定的利润率
            score += 1
            details.append(f"Moderately predictable margins: {avg_margin:.1%} avg with some volatility")
        else:
            details.append(f"Unpredictable margins: {avg_margin:.1%} avg with high volatility ({margin_volatility:.1%})")
    else:
        details.append("Insufficient margin history")
    
    # 4. Cash generation reliability
    # 4. 现金生成的可靠性
    fcf_values = [item.free_cash_flow for item in financial_line_items 
                 if hasattr(item, 'free_cash_flow') and item.free_cash_flow is not None]
    
    if fcf_values and len(fcf_values) >= 5:
        # Count positive FCF periods
        # 计算正的FCF期间
        positive_fcf_periods = sum(1 for fcf in fcf_values if fcf > 0)
        
        if positive_fcf_periods == len(fcf_values):
            # Consistently positive FCF
            # 稳定的正的FCF
            score += 2
            details.append("Highly predictable cash generation: Positive FCF in all periods")
        elif positive_fcf_periods >= len(fcf_values) * 0.8:
            # Mostly positive FCF
            # 大部分的正的FCF
            score += 1
            details.append(f"Predictable cash generation: Positive FCF in {positive_fcf_periods}/{len(fcf_values)} periods")
        else:
            details.append(f"Unpredictable cash generation: Positive FCF in only {positive_fcf_periods}/{len(fcf_values)} periods")
    else:
        details.append("Insufficient free cash flow history")
    
    # Scale score to 0-10 range
    # Maximum possible raw score would be 10 (3+3+2+2)
    # 缩放得分到0-10范围
    # 最大可能的原始得分是10(3+3+2+2)
    final_score = min(10, score * 10 / 10)
    
    return {
        "score": final_score,
        "details": "; ".join(details)
    }


def calculate_munger_valuation(financial_line_items: list, market_cap: float) -> dict:
    """
    Calculate intrinsic value using Munger's approach:
    - Focus on owner earnings (approximated by FCF)
    - Simple multiple on normalized earnings
    - Prefer paying a fair price for a wonderful business
    """
    '''
    使用芒格的方法计算内在价值：
    - 关注所有者收益(以自由现金流FCF为近似)
    - 对标准化收益使用简单倍数
    - 倾向于为优质企业支付合理价格
    '''
    score = 0
    details = []
    
    if not financial_line_items or market_cap is None:
        return {
            "score": 0,
            "details": "Insufficient data to perform valuation"
        }
    
    # Get FCF values (Munger's preferred "owner earnings" metric)
    # 获取FCF值(芒格的首选“所有者收益”指标)
    fcf_values = [item.free_cash_flow for item in financial_line_items 
                 if hasattr(item, 'free_cash_flow') and item.free_cash_flow is not None]
    
    if not fcf_values or len(fcf_values) < 3:
        return {
            "score": 0,
            "details": "Insufficient free cash flow data for valuation"
        }
    
    # 1. Normalize earnings by taking average of last 3-5 years
    # (Munger prefers to normalize earnings to avoid over/under-valuation based on cyclical factors)
    # 1. 对最近的3-5年的收益进行标准化
    # (芒格倾向于对收益进行标准化，以避免基于循环因素的过度/低估)
    normalized_fcf = sum(fcf_values[:min(5, len(fcf_values))]) / min(5, len(fcf_values))
    
    if normalized_fcf <= 0:
        return {
            "score": 0,
            "details": f"Negative or zero normalized FCF ({normalized_fcf}), cannot value",
            "intrinsic_value": None
        }
    
    # 2. Calculate FCF yield (inverse of P/FCF multiple)
    # 2. 计算FCF收益率(逆P/FCF倍数)
    fcf_yield = normalized_fcf / market_cap
    
    # 3. Apply Munger's FCF multiple based on business quality
    # Munger would pay higher multiples for wonderful businesses
    # Let's use a sliding scale where higher FCF yields are more attractive
    # 3. 根据企业质量应用芒格的FCF倍数
    # 芒格会为优质企业支付更高的倍数
    # 让我们使用滑动刻度，其中更高的FCF收益率更有吸引力
    if fcf_yield > 0.15:  # >15% FCF yield (P/FCF < 6.67x) # >15%的FCF收益率(P/FCF<6.67x)
        score += 5
    if fcf_yield > 0.08:  # >8% FCF yield (P/FCF < 12.5x) # >8%的FCF收益率(P/FCF<12.5x)
        score += 4
        details.append(f"Excellent value: {fcf_yield:.1%} FCF yield")
    elif fcf_yield > 0.05:  # >5% FCF yield (P/FCF < 20x) # >5%的FCF收益率(P/FCF<20x)
        score += 3
        details.append(f"Good value: {fcf_yield:.1%} FCF yield")
    elif fcf_yield > 0.03:  # >3% FCF yield (P/FCF < 33x) # >3%的FCF收益率(P/FCF<33x)
        score += 1
        details.append(f"Fair value: {fcf_yield:.1%} FCF yield")
    else:
        details.append(f"Expensive: Only {fcf_yield:.1%} FCF yield")
    
    # 4. Calculate simple intrinsic value range
    # Munger tends to use straightforward valuations, avoiding complex DCF models
    # 4. 计算简单的内在价值范围
    # 芒格倾向于使用简单的估值，避免复杂的DCF模型
    conservative_value = normalized_fcf * 10  # 10x FCF = 10% yield # 10xFCF=10%收益率
    reasonable_value = normalized_fcf * 15    # 15x FCF ≈ 6.7% yield # 15xFCF≈6.7%收益率
    optimistic_value = normalized_fcf * 20    # 20x FCF = 5% yield # 20xFCF=5%收益率
    
    # 5. Calculate margins of safety
    # 这里的注释有误，应该是“计算安全边际”
    current_to_reasonable = (reasonable_value - market_cap) / market_cap
    
    if current_to_reasonable > 0.3:  # >30% upside # >30%的盈利
        score += 5
        details.append(f"Very large margin of safety: {current_to_reasonable:.1%} upside to reasonable value")
        score += 3
        details.append(f"Large margin of safety: {current_to_reasonable:.1%} upside to reasonable value")
    elif current_to_reasonable > 0.1:  # >10% upside # >10%的盈利
        score += 3
        score += 2
        details.append(f"Moderate margin of safety: {current_to_reasonable:.1%} upside to reasonable value")
    elif current_to_reasonable > -0.1:  # Within 10% of reasonable value # 10%以内的合理价值
        score += 2
        score += 1
        details.append(f"Fair price: Within 10% of reasonable value ({current_to_reasonable:.1%})")
    else:
        details.append(f"Expensive: {-current_to_reasonable:.1%} premium to reasonable value")
    
    # 6. Check earnings trajectory for additional context
    # Munger likes growing owner earnings
    # 6. 检查收益轨迹以获得更多上下文
    # 芒格喜欢增长的所有者收益
    if len(fcf_values) >= 3:
        recent_avg = sum(fcf_values[:3]) / 3
        older_avg = sum(fcf_values[-3:]) / 3 if len(fcf_values) >= 6 else fcf_values[-1]
        
        if recent_avg > older_avg * 1.2:  # >20% growth in FCF # >20%的FCF增长
            score += 3
            details.append("Growing FCF trend adds to intrinsic value")
        elif recent_avg > older_avg:
            score += 2
            details.append("Stable to growing FCF supports valuation")
        else:
            details.append("Declining FCF trend is concerning")
    
    # Scale score to 0-10 range
    # Maximum possible raw score would be 10 (4+3+3)
    # 缩放得分到0-10范围
    # 最大可能的原始得分是10(4+3+3)
    final_score = min(10, score * 10 / 10) 
    
    return {
        "score": final_score,
        "details": "; ".join(details),
        "intrinsic_value_range": {
            "conservative": conservative_value,
            "reasonable": reasonable_value,
            "optimistic": optimistic_value
        },
        "fcf_yield": fcf_yield,
        "normalized_fcf": normalized_fcf
    }


def analyze_news_sentiment(news_items: list) -> str:
    """
    Simple qualitative analysis of recent news.
    Munger pays attention to significant news but doesn't overreact to short-term stories.
    """
    '''
    对近期新闻进行简单的定性分析。
    芒格关注重要新闻，但不会对短期新闻过度反应。
    '''
    if not news_items or len(news_items) == 0:
        return "No news data available"
    
    # Just return a simple count for now - in a real implementation, this would use NLP
    # 现在只是简单地返回一个计数 - 在实际实现中，这将使用NLP
    return f"Qualitative review of {len(news_items)} recent news items would be needed"

'''
你是一个查理·芒格AI代理，使用他的原则做出投资决策：

1. 关注企业的质量和可预测性
2. 运用多学科的思维模型来分析投资
3. 寻找强大、持久的竞争优势(护城河)
4. 强调长期思维和耐心
5. 重视管理层的诚信和能力
6. 优先考虑投资资本回报率高的企业
7. 为优质企业支付合理的价格
8. 永远不要高估，始终要求有安全边际
9. 避开复杂和你不理解的企业
10. "反向思考，永远反向思考" - 专注于避免愚蠢而不是追求聪明

规则：
- 赞赏运营和现金流可预测、稳定的企业
- 重视具有高ROIC和定价能力的企业
- 偏好经济学原理简单易懂的企业
- 欣赏与公司利益一致且对股东友好的管理层资本配置
- 关注长期经济效益而不是短期指标
- 对业务快速变化或过度稀释股份的企业保持怀疑
- 避免过度杠杆或金融工程
- 提供基于数据的理性建议(看涨、看跌或中性)

在提供理由时，要做到详尽和具体：
1. 解释最影响你决策的关键因素(正面和负面)
2. 运用至少2-3个具体的思维模型或学科来解释你的思考
3. 在相关处提供量化证据(例如，具体的ROIC值、利润率趋势)
4. 在分析中指出你会"避免"什么(反向思考问题)
5. 使用查理·芒格直接、简洁的对话风格来解释

看涨示例："22%的高ROIC展示了公司的护城河。当应用基本的微观经济学时，我们可以看到竞争对手会难以..."
看跌示例："我看到这家企业在资本配置上犯了一个典型错误。正如我经常说的[相关的芒格主义]，这家公司似乎在..."
            '''
def generate_munger_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> CharlieMungerSignal:
    """
    Generates investment decisions in the style of Charlie Munger.
    """
    '''
    以查理·芒格的风格生成投资决策。
    '''
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Charlie Munger AI agent, making investment decisions using his principles:

            1. Focus on the quality and predictability of the business.
            2. Rely on mental models from multiple disciplines to analyze investments.
            3. Look for strong, durable competitive advantages (moats).
            4. Emphasize long-term thinking and patience.
            5. Value management integrity and competence.
            6. Prioritize businesses with high returns on invested capital.
            7. Pay a fair price for wonderful businesses.
            8. Never overpay, always demand a margin of safety.
            9. Avoid complexity and businesses you don't understand.
            10. "Invert, always invert" - focus on avoiding stupidity rather than seeking brilliance.
            
            Rules:
            - Praise businesses with predictable, consistent operations and cash flows.
            - Value businesses with high ROIC and pricing power.
            - Prefer simple businesses with understandable economics.
            - Admire management with skin in the game and shareholder-friendly capital allocation.
            - Focus on long-term economics rather than short-term metrics.
            - Be skeptical of businesses with rapidly changing dynamics or excessive share dilution.
            - Avoid excessive leverage or financial engineering.
            - Provide a rational, data-driven recommendation (bullish, bearish, or neutral).
            
            When providing your reasoning, be thorough and specific by:
            1. Explaining the key factors that influenced your decision the most (both positive and negative)
            2. Applying at least 2-3 specific mental models or disciplines to explain your thinking
            3. Providing quantitative evidence where relevant (e.g., specific ROIC values, margin trends)
            4. Citing what you would "avoid" in your analysis (invert the problem)
            5. Using Charlie Munger's direct, pithy conversational style in your explanation
            
            For example, if bullish: "The high ROIC of 22% demonstrates the company's moat. When applying basic microeconomics, we can see that competitors would struggle to..."
            For example, if bearish: "I see this business making a classic mistake in capital allocation. As I've often said about [relevant Mungerism], this company appears to be..."
            """
        ),
        (
            "human",
            """Based on the following analysis, create a Munger-style investment signal.

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

    def create_default_charlie_munger_signal():
        return CharlieMungerSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis, defaulting to neutral"
        )

    return call_llm(
        prompt=prompt, 
        model_name=model_name, 
        model_provider=model_provider, 
        pydantic_model=CharlieMungerSignal, 
        agent_name="charlie_munger_agent", 
        default_factory=create_default_charlie_munger_signal,
    )