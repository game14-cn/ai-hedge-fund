# AI 对冲基金

这是一个 AI 驱动的对冲基金概念验证项目。该项目旨在探索使用人工智能进行交易决策。本项目仅供**教育**目的使用，不适用于实际交易或投资。

该系统由多个协同工作的代理组成：

1. 本杰明·格雷厄姆代理 - 价值投资之父，只买具有安全边际的隐藏瑰宝
2. 比尔·阿克曼代理 - 激进投资者，采取大胆立场并推动变革
3. 凯西·伍德代理 - 成长投资女王，相信创新和颠覆的力量
4. 查理·芒格代理 - 沃伦·巴菲特的合伙人，只在合理价格买入优质企业
5. 菲利普·费雪代理 - 传奇成长投资者，精通市场调查分析
6. 斯坦利·德鲁肯米勒代理 - 宏观投资传奇，寻找具有成长潜力的不对称机会
7. 沃伦·巴菲特代理 - 奥马哈先知，寻找合理价格的优质公司
8. 估值代理 - 计算股票内在价值并生成交易信号
9. 情绪代理 - 分析市场情绪并生成交易信号
10. 基本面代理 - 分析基本面数据并生成交易信号
11. 技术面代理 - 分析技术指标并生成交易信号
12. 风险管理器 - 计算风险指标并设置仓位限制
13. 投资组合管理器 - 做出最终交易决策并生成订单
    
<img width="1042" alt="Screenshot 2025-03-22 at 6 19 07 PM" src="https://github.com/user-attachments/assets/cbae3dcf-b571-490d-b0ad-3f0f035ac0d4" />

**注意**：系统仅模拟交易决策，不进行实际交易。

[![Twitter Follow](https://img.shields.io/twitter/follow/virattt?style=social)](https://twitter.com/virattt)

## 免责声明

本项目仅供**教育和研究目的**使用。

- 不适用于实际交易或投资
- 不提供任何保证或担保
- 过往业绩不代表未来表现
- 创建者不承担任何财务损失责任
- 投资决策请咨询财务顾问

使用本软件即表示您同意仅将其用于学习目的。

## 目录
- [设置](#设置)
- [使用方法](#使用方法)
  - [运行对冲基金](#运行对冲基金)
  - [运行回测器](#运行回测器)
- [项目结构](#项目结构)
- [贡献](#贡献)
- [功能请求](#功能请求)
- [许可证](#许可证)

## 设置

克隆仓库：
```bash
git clone https://github.com/virattt/ai-hedge-fund.git
cd ai-hedge-fund
```

1. 安装 Poetry（如果尚未安装）：
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. 安装依赖：
```bash
poetry install
```

3. 设置环境变量：
```bash
# 创建 .env 文件存放 API 密钥
cp .env.example .env
```

4. 设置 API 密钥：
```bash
# 用于运行 OpenAI 托管的 LLM（gpt-4o, gpt-4o-mini 等）
# 从 https://platform.openai.com/ 获取 OpenAI API 密钥
OPENAI_API_KEY=your-openai-api-key

# 用于运行 Groq 托管的 LLM（deepseek, llama3 等）
# 从 https://groq.com/ 获取 Groq API 密钥
GROQ_API_KEY=your-groq-api-key

# 用于获取为对冲基金提供动力的金融数据
# 从 https://financialdatasets.ai/ 获取 Financial Datasets API 密钥
FINANCIAL_DATASETS_API_KEY=your-financial-datasets-api-key
```

**重要提示**：您必须设置 `OPENAI_API_KEY`、`GROQ_API_KEY`、`ANTHROPIC_API_KEY` 或 `DEEPSEEK_API_KEY` 才能使对冲基金正常工作。如果您想使用所有提供商的 LLM，则需要设置所有 API 密钥。

AAPL、GOOGL、MSFT、NVDA 和 TSLA 的金融数据是免费的，不需要 API 密钥。

对于任何其他股票代码，您需要在 .env 文件中设置 `FINANCIAL_DATASETS_API_KEY`。

## 使用方法

### 运行对冲基金
```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA
```

**示例输出：**
<img width="992" alt="Screenshot 2025-01-06 at 5 50 17 PM" src="https://github.com/user-attachments/assets/e8ca04bf-9989-4a7d-a8b4-34e04666663b" />

您还可以指定 `--show-reasoning` 标志将每个代理的推理过程打印到控制台。

```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --show-reasoning
```

您可以选择指定开始和结束日期，以针对特定时间段做出决策。

```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01 
```

### 运行回测器

```bash
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA
```

**示例输出：**
<img width="941" alt="Screenshot 2025-01-06 at 5 47 52 PM" src="https://github.com/user-attachments/assets/00e794ea-8628-44e6-9a84-8f8a31ad3b47" />

您可以选择指定开始和结束日期，以对特定时间段进行回测。

```bash
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01
```

## 项目结构
```
ai-hedge-fund/
├── src/
│   ├── agents/                   # 代理定义和工作流
│   │   ├── bill_ackman.py        # 比尔·阿克曼代理
│   │   ├── fundamentals.py       # 基本面分析代理
│   │   ├── portfolio_manager.py  # 投资组合管理代理
│   │   ├── risk_manager.py       # 风险管理代理
│   │   ├── sentiment.py          # 情绪分析代理
│   │   ├── technicals.py         # 技术分析代理
│   │   ├── valuation.py          # 估值分析代理
│   │   ├── warren_buffett.py     # 沃伦·巴菲特代理
│   ├── tools/                    # 代理工具
│   │   ├── api.py                # API 工具
│   ├── backtester.py             # 回测工具
│   ├── main.py                   # 主入口点
├── pyproject.toml
├── ...
```

## 贡献

1. Fork 仓库
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建拉取请求

**重要提示**：请保持拉取请求小而集中。这将使其更容易审查和合并。

## 功能请求

如果您有功能请求，请开启一个 [issue](https://github.com/virattt/ai-hedge-fund/issues) 并确保标记为 `enhancement`。

## 许可证

本项目采用 MIT 许可证 - 有关详细信息，请参阅 LICENSE 文件。