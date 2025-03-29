import sys

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import questionary

import matplotlib.pyplot as plt
import pandas as pd
from colorama import Fore, Style, init
import numpy as np
import itertools

from llm.models import LLM_ORDER, get_model_info
from utils.analysts import ANALYST_ORDER
from main import run_hedge_fund
from tools.api import (
    get_company_news,
    get_price_data,
    get_prices,
    get_financial_metrics,
    get_insider_trades,
)
from utils.display import print_backtest_results, format_backtest_row
from typing_extensions import Callable

init(autoreset=True)


class Backtester:
    def __init__(
        self,
        agent: Callable,
        tickers: list[str],
        start_date: str,
        end_date: str,
        initial_capital: float,
        model_name: str = "gpt-4o",
        model_provider: str = "OpenAI",
        selected_analysts: list[str] = [],
        initial_margin_requirement: float = 0.0,
    ):
        """
        :param agent: The trading agent (Callable).
        :param tickers: List of tickers to backtest.
        :param start_date: Start date string (YYYY-MM-DD).
        :param end_date: End date string (YYYY-MM-DD).
        :param initial_capital: Starting portfolio cash.
        :param model_name: Which LLM model name to use (gpt-4, etc).
        :param model_provider: Which LLM provider (OpenAI, etc).
        :param selected_analysts: List of analyst names or IDs to incorporate.
        :param initial_margin_requirement: The margin ratio (e.g. 0.5 = 50%).
        """
        """
        :param agent: 交易代理（可调用对象）。
        :param tickers: 用于回测的股票代码列表。
        :param start_date: 开始日期字符串（YYYY-MM-DD格式）。
        :param end_date: 结束日期字符串（YYYY-MM-DD格式）。
        :param initial_capital: 初始投资资金。
        :param model_name: 要使用的 LLM 模型名称（如 gpt-4 等）。
        :param model_provider: LLM 提供商（如 OpenAI 等）。
        :param selected_analysts: 要包含的分析师名称或 ID 列表。
        :param initial_margin_requirement: 保证金比率（如 0.5 表示 50%）。
        """
        self.agent = agent
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.model_name = model_name
        self.model_provider = model_provider
        self.selected_analysts = selected_analysts

        # Store the margin ratio (e.g. 0.5 means 50% margin required).
        # 存储保证金比率（如 0.5 表示 50% 的保证金）。
        self.margin_ratio = initial_margin_requirement

        # Initialize portfolio with support for long/short positions
        # 初始化投资组合，支持多头/空头仓位
        self.portfolio_values = []
        self.portfolio = {
            "cash": initial_capital,
            "margin_used": 0.0,  # total margin usage across all short positions # 所有空头仓位的总保证金使用量
            "positions": {
                ticker: {
                    "long": 0,               # Number of shares held long # 多头仓位的股票数量
                    "short": 0,              # Number of shares held short # 空头仓位的股票数量
                    "long_cost_basis": 0.0,  # Average cost basis per share (long) # 多头仓位的平均成本
                    "short_cost_basis": 0.0, # Average cost basis per share (short) # 空头仓位的平均成本
                    "short_margin_used": 0.0 # Dollars of margin used for this ticker's short # 此股票的空头保证金使用量
                } for ticker in tickers
            },
            "realized_gains": {
                ticker: {
                    "long": 0.0,   # Realized gains from long positions # 多头仓位的已实现盈亏
                    "short": 0.0,  # Realized gains from short positions # 空头仓位的已实现盈亏
                } for ticker in tickers
            }
        }

    def execute_trade(self, ticker: str, action: str, quantity: float, current_price: float):
        """
        Execute trades with support for both long and short positions.
        `quantity` is the number of shares the agent wants to buy/sell/short/cover.
        We will only trade integer shares to keep it simple.
        """
        """
        执行交易，支持多头和空头仓位。
        `quantity` 是代理想要买入/卖出/做空/平仓的股票数量。
        我们只交易整数股以保持简单。
        """
        if quantity <= 0:
            return 0

        quantity = int(quantity)  # force integer shares # 强制整股交易
        position = self.portfolio["positions"][ticker]

        if action == "buy":
            cost = quantity * current_price
            if cost <= self.portfolio["cash"]:
                # Weighted average cost basis for the new total
                # 计算新的加权平均成本
                old_shares = position["long"]
                old_cost_basis = position["long_cost_basis"]
                new_shares = quantity
                total_shares = old_shares + new_shares

                if total_shares > 0:
                    total_old_cost = old_cost_basis * old_shares
                    total_new_cost = cost
                    position["long_cost_basis"] = (total_old_cost + total_new_cost) / total_shares

                position["long"] += quantity
                self.portfolio["cash"] -= cost
                return quantity
            else:
                # Calculate maximum affordable quantity
                # 计算可购买的最大数量
                max_quantity = int(self.portfolio["cash"] / current_price)
                if max_quantity > 0:
                    cost = max_quantity * current_price
                    old_shares = position["long"]
                    old_cost_basis = position["long_cost_basis"]
                    total_shares = old_shares + max_quantity

                    if total_shares > 0:
                        total_old_cost = old_cost_basis * old_shares
                        total_new_cost = cost
                        position["long_cost_basis"] = (total_old_cost + total_new_cost) / total_shares

                    position["long"] += max_quantity
                    self.portfolio["cash"] -= cost
                    return max_quantity
                return 0

        elif action == "sell":
            # You can only sell as many as you own
            # 只能卖出你拥有的数量
            quantity = min(quantity, position["long"])
            if quantity > 0:
                # Realized gain/loss using average cost basis
                # 计算使用平均成本的已实现盈亏
                avg_cost_per_share = position["long_cost_basis"] if position["long"] > 0 else 0
                realized_gain = (current_price - avg_cost_per_share) * quantity
                self.portfolio["realized_gains"][ticker]["long"] += realized_gain

                position["long"] -= quantity
                self.portfolio["cash"] += quantity * current_price

                if position["long"] == 0:
                    position["long_cost_basis"] = 0.0

                return quantity

        elif action == "short":
            """
            Typical short sale flow:
              1) Receive proceeds = current_price * quantity
              2) Post margin_required = proceeds * margin_ratio
              3) Net effect on cash = +proceeds - margin_required
            """
            """
            典型的卖空流程：
              1) 收到卖空所得 = 当前价格 * 数量
              2) 缴纳保证金 = 卖空所得 * 保证金比率
              3) 现金净影响 = +卖空所得 - 保证金
            """
            proceeds = current_price * quantity
            margin_required = proceeds * self.margin_ratio
            if margin_required <= self.portfolio["cash"]:
                # Weighted average short cost basis
                # 计算加权平均卖空成本
                old_short_shares = position["short"]
                old_cost_basis = position["short_cost_basis"]
                new_shares = quantity
                total_shares = old_short_shares + new_shares

                if total_shares > 0:
                    total_old_cost = old_cost_basis * old_short_shares
                    total_new_cost = current_price * new_shares
                    position["short_cost_basis"] = (total_old_cost + total_new_cost) / total_shares

                position["short"] += quantity

                # Update margin usage
                # 更新保证金使用量
                position["short_margin_used"] += margin_required
                self.portfolio["margin_used"] += margin_required

                # Increase cash by proceeds, then subtract the required margin
                # 增加现金，然后减去所需的保证金
                self.portfolio["cash"] += proceeds
                self.portfolio["cash"] -= margin_required
                return quantity
            else:
                # Calculate maximum shortable quantity
                # 计算可卖空的最大数量
                if self.margin_ratio > 0:
                    max_quantity = int(self.portfolio["cash"] / (current_price * self.margin_ratio))
                else:
                    max_quantity = 0

                if max_quantity > 0:
                    proceeds = current_price * max_quantity
                    margin_required = proceeds * self.margin_ratio

                    old_short_shares = position["short"]
                    old_cost_basis = position["short_cost_basis"]
                    total_shares = old_short_shares + max_quantity

                    if total_shares > 0:
                        total_old_cost = old_cost_basis * old_short_shares
                        total_new_cost = current_price * max_quantity
                        position["short_cost_basis"] = (total_old_cost + total_new_cost) / total_shares

                    position["short"] += max_quantity
                    position["short_margin_used"] += margin_required
                    self.portfolio["margin_used"] += margin_required

                    self.portfolio["cash"] += proceeds
                    self.portfolio["cash"] -= margin_required
                    return max_quantity
                return 0

        elif action == "cover":
            """
            When covering shares:
              1) Pay cover cost = current_price * quantity
              2) Release a proportional share of the margin
              3) Net effect on cash = -cover_cost + released_margin
            """
            """
            平仓股票时：
              1) 支付平仓成本 = 当前价格 * 数量
              2) 释放相应比例的保证金
              3) 现金净影响 = -平仓成本 + 释放的保证金
            """
            quantity = min(quantity, position["short"])
            if quantity > 0:
                cover_cost = quantity * current_price
                avg_short_price = position["short_cost_basis"] if position["short"] > 0 else 0
                realized_gain = (avg_short_price - current_price) * quantity

                if position["short"] > 0:
                    portion = quantity / position["short"]
                else:
                    portion = 1.0

                margin_to_release = portion * position["short_margin_used"]

                position["short"] -= quantity
                position["short_margin_used"] -= margin_to_release
                self.portfolio["margin_used"] -= margin_to_release

                # Pay the cost to cover, but get back the released margin
                # 支付平仓成本，但获得释放的保证金
                self.portfolio["cash"] += margin_to_release
                self.portfolio["cash"] -= cover_cost

                self.portfolio["realized_gains"][ticker]["short"] += realized_gain

                if position["short"] == 0:
                    position["short_cost_basis"] = 0.0
                    position["short_margin_used"] = 0.0

                return quantity

        return 0

    def calculate_portfolio_value(self, current_prices):
        """
        Calculate total portfolio value, including:
          - cash
          - market value of long positions
          - unrealized gains/losses for short positions
        """
        """
        计算投资组合总价值，包括：
          - 现金
          - 多头仓位的市场价值
          - 空头仓位的未实现盈亏
        """
        total_value = self.portfolio["cash"]

        for ticker in self.tickers:
            position = self.portfolio["positions"][ticker]
            price = current_prices[ticker]

            # Long position value
            # 多头仓位价值
            long_value = position["long"] * price
            total_value += long_value

            # Short position unrealized PnL = short_shares * (short_cost_basis - current_price)
            # 空头仓位未实现盈亏 = short_shares * (short_cost_basis - current_price)
            if position["short"] > 0:
                total_value += position["short"] * (position["short_cost_basis"] - price)

        return total_value

    def prefetch_data(self):
        """Pre-fetch all data needed for the backtest period."""
        # 请在整个回测期间预取所需的所有数据
        print("\nPre-fetching data for the entire backtest period...")

        # Convert end_date string to datetime, fetch up to 1 year before
        # 将 end_date 字符串转换为 datetime，提前 1 年
        end_date_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
        start_date_dt = end_date_dt - relativedelta(years=1)
        start_date_str = start_date_dt.strftime("%Y-%m-%d")

        for ticker in self.tickers:
            # Fetch price data for the entire period, plus 1 year
            # 为整个期间获取价格数据，再提前 1 年
            get_prices(ticker, start_date_str, self.end_date)

            # Fetch financial metrics
            # 获取财务指标
            get_financial_metrics(ticker, self.end_date, limit=10)

            # Fetch insider trades
            # 获取 Insider 交易
            get_insider_trades(ticker, self.end_date, start_date=self.start_date, limit=1000)

            # Fetch company news
            # 获取公司新闻
            get_company_news(ticker, self.end_date, start_date=self.start_date, limit=1000)

        print("Data pre-fetch complete.")

    def parse_agent_response(self, agent_output):
        """Parse JSON output from the agent (fallback to 'hold' if invalid)."""
        # 解析代理的 JSON 输出（如果无效则回退到 'hold'）
        import json

        try:
            decision = json.loads(agent_output)
            return decision
        except Exception:
            print(f"Error parsing action: {agent_output}")
            return {"action": "hold", "quantity": 0}

    def run_backtest(self):
        # Pre-fetch all data at the start
        # 在开始时预取所有数据
        self.prefetch_data()

        dates = pd.date_range(self.start_date, self.end_date, freq="B")
        table_rows = []
        performance_metrics = {
            'sharpe_ratio': None,
            'sortino_ratio': None,
            'max_drawdown': None,
            'long_short_ratio': None,
            'gross_exposure': None,
            'net_exposure': None
        }

        print("\nStarting backtest...")

        # Initialize portfolio values list with initial capital
        # 使用初始资本初始化投资组合价值列表
        if len(dates) > 0:
            self.portfolio_values = [{"Date": dates[0], "Portfolio Value": self.initial_capital}]
        else:
            self.portfolio_values = []

        for current_date in dates:
            lookback_start = (current_date - timedelta(days=30)).strftime("%Y-%m-%d")
            current_date_str = current_date.strftime("%Y-%m-%d")
            previous_date_str = (current_date - timedelta(days=1)).strftime("%Y-%m-%d")

            # Skip if there's no prior day to look back (i.e., first date in the range)
            # 如果没有前一天来回顾（即范围中的第一天），则跳过
            if lookback_start == current_date_str:
                continue

            # Get current prices for all tickers
            # 获取所有股票的当前价格
            try:
                current_prices = {
                    ticker: get_price_data(ticker, previous_date_str, current_date_str).iloc[-1]["close"]
                    for ticker in self.tickers
                }
            except Exception:
                # If data is missing or there's an API error, skip this day
                # 如果数据缺失或出现 API 错误，请跳过这一天
                print(f"Error fetching prices between {previous_date_str} and {current_date_str}")
                continue

            # ---------------------------------------------------------------
            # 1) Execute the agent's trades # 
            # 1) 执行代理的交易
            # ---------------------------------------------------------------
            output = self.agent(
                tickers=self.tickers,
                start_date=lookback_start,
                end_date=current_date_str,
                portfolio=self.portfolio,
                model_name=self.model_name,
                model_provider=self.model_provider,
                selected_analysts=self.selected_analysts,
            )
            decisions = output["decisions"]
            analyst_signals = output["analyst_signals"]

            # Execute trades for each ticker
            # 为每个股票执行交易
            executed_trades = {}
            for ticker in self.tickers:
                decision = decisions.get(ticker, {"action": "hold", "quantity": 0})
                action, quantity = decision.get("action", "hold"), decision.get("quantity", 0)

                executed_quantity = self.execute_trade(ticker, action, quantity, current_prices[ticker])
                executed_trades[ticker] = executed_quantity

            # ---------------------------------------------------------------
            # 2) Now that trades have executed trades, recalculate the final
            #    portfolio value for this day.
            # 2) 现在交易已经执行完成，重新计算今天的最终投资组合价值。
            # ---------------------------------------------------------------
            total_value = self.calculate_portfolio_value(current_prices)

            # Also compute long/short exposures for final post‐trade state
            # 还计算最终交易后的多头/空头暴露
            long_exposure = sum(
                self.portfolio["positions"][t]["long"] * current_prices[t]
                for t in self.tickers
            )
            short_exposure = sum(
                self.portfolio["positions"][t]["short"] * current_prices[t]
                for t in self.tickers
            )

            # Calculate gross and net exposures
            # 计算总暴露和净暴露
            gross_exposure = long_exposure + short_exposure
            net_exposure = long_exposure - short_exposure
            long_short_ratio = (
                long_exposure / short_exposure if short_exposure > 1e-9 else float('inf')
            )

            # Track each day's portfolio value in self.portfolio_values
            # 在 self.portfolio_values 中记录每一天的投资组合价值
            self.portfolio_values.append({
                "Date": current_date,
                "Portfolio Value": total_value,
                "Long Exposure": long_exposure,
                "Short Exposure": short_exposure,
                "Gross Exposure": gross_exposure,
                "Net Exposure": net_exposure,
                "Long/Short Ratio": long_short_ratio
            })

            # ---------------------------------------------------------------
            # 3) Build the table rows to display
            # 3) 构建要显示的表格行
            # ---------------------------------------------------------------
            date_rows = []

            # For each ticker, record signals/trades
            # 对于每个股票，记录信号/交易
            for ticker in self.tickers:
                ticker_signals = {}
                for agent_name, signals in analyst_signals.items():
                    if ticker in signals:
                        ticker_signals[agent_name] = signals[ticker]

                bullish_count = len([s for s in ticker_signals.values() if s.get("signal", "").lower() == "bullish"])
                bearish_count = len([s for s in ticker_signals.values() if s.get("signal", "").lower() == "bearish"])
                neutral_count = len([s for s in ticker_signals.values() if s.get("signal", "").lower() == "neutral"])

                # Calculate net position value
                # 计算净仓位价值
                pos = self.portfolio["positions"][ticker]
                long_val = pos["long"] * current_prices[ticker]
                short_val = pos["short"] * current_prices[ticker]
                net_position_value = long_val - short_val

                # Get the action and quantity from the decisions
                # 从决策中获取动作和数量
                action = decisions.get(ticker, {}).get("action", "hold")
                quantity = executed_trades.get(ticker, 0)
                
                # Append the agent action to the table rows
                # 将代理动作添加到表格行
                date_rows.append(
                    format_backtest_row(
                        date=current_date_str,
                        ticker=ticker,
                        action=action,
                        quantity=quantity,
                        price=current_prices[ticker],
                        shares_owned=pos["long"] - pos["short"],  # net shares # 净仓位
                        position_value=net_position_value,
                        bullish_count=bullish_count,
                        bearish_count=bearish_count,
                        neutral_count=neutral_count,
                    )
                )
            # ---------------------------------------------------------------
            # 4) Calculate performance summary metrics
            # 4) 计算绩效摘要指标
            # ---------------------------------------------------------------
            total_realized_gains = sum(
                self.portfolio["realized_gains"][t]["long"] +
                self.portfolio["realized_gains"][t]["short"]
                for t in self.tickers
            )

            # Calculate cumulative return vs. initial capital
            # 计算累计回报与初始资本
            portfolio_return = ((total_value + total_realized_gains) / self.initial_capital - 1) * 100

            # Add summary row for this day
            # 为本日添加摘要行
            date_rows.append(
                format_backtest_row(
                    date=current_date_str,
                    ticker="",
                    action="",
                    quantity=0,
                    price=0,
                    shares_owned=0,
                    position_value=0,
                    bullish_count=0,
                    bearish_count=0,
                    neutral_count=0,
                    is_summary=True,
                    total_value=total_value,
                    return_pct=portfolio_return,
                    cash_balance=self.portfolio["cash"],
                    total_position_value=total_value - self.portfolio["cash"],
                    sharpe_ratio=performance_metrics["sharpe_ratio"],
                    sortino_ratio=performance_metrics["sortino_ratio"],
                    max_drawdown=performance_metrics["max_drawdown"],
                ),
            )

            table_rows.extend(date_rows)
            print_backtest_results(table_rows)

            # Update performance metrics if we have enough data
            # 如果我们有足够的数据，则更新绩效指标
            if len(self.portfolio_values) > 3:
                self._update_performance_metrics(performance_metrics)

        return performance_metrics

    def _update_performance_metrics(self, performance_metrics):
        """Helper method to update performance metrics using daily returns."""
        # 辅助方法，使用每日回报更新绩效指标
        values_df = pd.DataFrame(self.portfolio_values).set_index("Date")
        values_df["Daily Return"] = values_df["Portfolio Value"].pct_change()
        clean_returns = values_df["Daily Return"].dropna()

        if len(clean_returns) < 2:
            return  # not enough data points # 不够数据点

        # Assumes 252 trading days/year
        # 假设每年 252 个交易日
        daily_risk_free_rate = 0.0434 / 252
        excess_returns = clean_returns - daily_risk_free_rate
        mean_excess_return = excess_returns.mean()
        std_excess_return = excess_returns.std()

        # Sharpe ratio
        # 夏普比率
        if std_excess_return > 1e-12:
            performance_metrics["sharpe_ratio"] = np.sqrt(252) * (mean_excess_return / std_excess_return)
        else:
            performance_metrics["sharpe_ratio"] = 0.0

        # Sortino ratio
        # 索提诺比率
        negative_returns = excess_returns[excess_returns < 0]
        if len(negative_returns) > 0:
            downside_std = negative_returns.std()
            if downside_std > 1e-12:
                performance_metrics["sortino_ratio"] = np.sqrt(252) * (mean_excess_return / downside_std)
            else:
                performance_metrics["sortino_ratio"] = float('inf') if mean_excess_return > 0 else 0
        else:
            performance_metrics["sortino_ratio"] = float('inf') if mean_excess_return > 0 else 0

        # Maximum drawdown
        # 最大回撤
        rolling_max = values_df["Portfolio Value"].cummax()
        drawdown = (values_df["Portfolio Value"] - rolling_max) / rolling_max
        performance_metrics["max_drawdown"] = drawdown.min() * 100

    def analyze_performance(self):
        """Creates a performance DataFrame, prints summary stats, and plots equity curve."""
        # 创建绩效 DataFrame，打印摘要统计信息，并绘制权益曲线
        if not self.portfolio_values:
            print("No portfolio data found. Please run the backtest first.")
            return pd.DataFrame()

        performance_df = pd.DataFrame(self.portfolio_values).set_index("Date")
        if performance_df.empty:
            print("No valid performance data to analyze.")
            return performance_df

        final_portfolio_value = performance_df["Portfolio Value"].iloc[-1]
        total_realized_gains = sum(
            self.portfolio["realized_gains"][ticker]["long"] for ticker in self.tickers
        )
        total_return = ((final_portfolio_value - self.initial_capital) / self.initial_capital) * 100

        print(f"\n{Fore.WHITE}{Style.BRIGHT}PORTFOLIO PERFORMANCE SUMMARY:{Style.RESET_ALL}")
        print(f"Total Return: {Fore.GREEN if total_return >= 0 else Fore.RED}{total_return:.2f}%{Style.RESET_ALL}")
        print(f"Total Realized Gains/Losses: {Fore.GREEN if total_realized_gains >= 0 else Fore.RED}${total_realized_gains:,.2f}{Style.RESET_ALL}")

        # Plot the portfolio value over time
        # 绘制投资组合价值随时间的变化
        plt.figure(figsize=(12, 6))
        plt.plot(performance_df.index, performance_df["Portfolio Value"], color="blue")
        plt.title("Portfolio Value Over Time")
        plt.ylabel("Portfolio Value ($)")
        plt.xlabel("Date")
        plt.grid(True)
        plt.show()

        # Compute daily returns
        # 计算每日回报
        performance_df["Daily Return"] = performance_df["Portfolio Value"].pct_change().fillna(0)
        daily_rf = 0.0434 / 252  # daily risk-free rate
        mean_daily_return = performance_df["Daily Return"].mean()
        std_daily_return = performance_df["Daily Return"].std()

        # Annualized Sharpe Ratio
        # 年度夏普比率
        if std_daily_return != 0:
            annualized_sharpe = np.sqrt(252) * ((mean_daily_return - daily_rf) / std_daily_return)
        else:
            annualized_sharpe = 0
        print(f"\nSharpe Ratio: {Fore.YELLOW}{annualized_sharpe:.2f}{Style.RESET_ALL}")

        # Max Drawdown
        # 最大回撤
        rolling_max = performance_df["Portfolio Value"].cummax()
        drawdown = (performance_df["Portfolio Value"] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        max_drawdown_date = drawdown.idxmin()
        if pd.notnull(max_drawdown_date):
            print(f"Maximum Drawdown: {Fore.RED}{max_drawdown * 100:.2f}%{Style.RESET_ALL} (on {max_drawdown_date.strftime('%Y-%m-%d')})")
        else:
            print(f"Maximum Drawdown: {Fore.RED}0.00%{Style.RESET_ALL}")

        # Win Rate
        # 胜率
        winning_days = len(performance_df[performance_df["Daily Return"] > 0])
        total_days = max(len(performance_df) - 1, 1)
        win_rate = (winning_days / total_days) * 100
        print(f"Win Rate: {Fore.GREEN}{win_rate:.2f}%{Style.RESET_ALL}")

        # Average Win/Loss Ratio
        # 平均赢/输比
        positive_returns = performance_df[performance_df["Daily Return"] > 0]["Daily Return"]
        negative_returns = performance_df[performance_df["Daily Return"] < 0]["Daily Return"]
        avg_win = positive_returns.mean() if not positive_returns.empty else 0
        avg_loss = abs(negative_returns.mean()) if not negative_returns.empty else 0
        if avg_loss != 0:
            win_loss_ratio = avg_win / avg_loss
        else:
            win_loss_ratio = float('inf') if avg_win > 0 else 0
        print(f"Win/Loss Ratio: {Fore.GREEN}{win_loss_ratio:.2f}{Style.RESET_ALL}")

        # Maximum Consecutive Wins / Losses
        # 最大连续赢/输
        returns_binary = (performance_df["Daily Return"] > 0).astype(int)
        if len(returns_binary) > 0:
            max_consecutive_wins = max((len(list(g)) for k, g in itertools.groupby(returns_binary) if k == 1), default=0)
            max_consecutive_losses = max((len(list(g)) for k, g in itertools.groupby(returns_binary) if k == 0), default=0)
        else:
            max_consecutive_wins = 0
            max_consecutive_losses = 0

        print(f"Max Consecutive Wins: {Fore.GREEN}{max_consecutive_wins}{Style.RESET_ALL}")
        print(f"Max Consecutive Losses: {Fore.RED}{max_consecutive_losses}{Style.RESET_ALL}")

        return performance_df


### 4. Run the Backtest #####
# 4. 运行回测
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run backtesting simulation")
    parser.add_argument(
        "--tickers",
        type=str,
        required=False,
        help="Comma-separated list of stock ticker symbols (e.g., AAPL,MSFT,GOOGL)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.now() - relativedelta(months=1)).strftime("%Y-%m-%d"),
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100000,
        help="Initial capital amount (default: 100000)",
    )
    parser.add_argument(
        "--margin-requirement",
        type=float,
        default=0.0,
        help="Margin ratio for short positions, e.g. 0.5 for 50% (default: 0.0)",
    )

    args = parser.parse_args()

    # Parse tickers from comma-separated string
    # 从逗号分隔的字符串解析 tickers
    tickers = [ticker.strip() for ticker in args.tickers.split(",")] if args.tickers else []

    # Choose analysts
    # 选择分析师
    selected_analysts = None
    choices = questionary.checkbox(
        "Use the Space bar to select/unselect analysts.",
        # 使用空格键选择/取消选择分析师。
        choices=[questionary.Choice(display, value=value) for display, value in ANALYST_ORDER],
        instruction="\n\nPress 'a' to toggle all.\n\nPress Enter when done to run the hedge fund.",
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:green"),
                ("selected", "fg:green noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
    ).ask()

    if not choices:
        print("\n\nInterrupt received. Exiting...")
        sys.exit(0)
    else:
        selected_analysts = choices
        print(
            f"\nSelected analysts: "
            f"{', '.join(Fore.GREEN + choice.title().replace('_', ' ') + Style.RESET_ALL for choice in choices)}"
        )

    # Select LLM model
    # 选择 LLM 模型
    model_choice = questionary.select(
        "Select your LLM model:",
        choices=[questionary.Choice(display, value=value) for display, value, _ in LLM_ORDER],
        style=questionary.Style([
            ("selected", "fg:green bold"),
            ("pointer", "fg:green bold"),
            ("highlighted", "fg:green"),
            ("answer", "fg:green bold"),
        ])
    ).ask()

    if not model_choice:
        print("\n\nInterrupt received. Exiting...")
        sys.exit(0)
    else:
        model_info = get_model_info(model_choice)
        if model_info:
            model_provider = model_info.provider.value
            print(f"\nSelected {Fore.CYAN}{model_provider}{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")
        else:
            model_provider = "Unknown"
            print(f"\nSelected model: {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")

    # Create and run the backtester
    # 创建并运行回测器
    backtester = Backtester(
        agent=run_hedge_fund,
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        model_name=model_choice,
        model_provider=model_provider,
        selected_analysts=selected_analysts,
        initial_margin_requirement=args.margin_requirement,
    )

    performance_metrics = backtester.run_backtest()
    performance_df = backtester.analyze_performance()
