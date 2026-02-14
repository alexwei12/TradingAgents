# China A-Share Usage Guide

TradingAgents supports China A-shares (Shanghai SSE, Shenzhen SZSE, Beijing BSE) through AKShare integration.

## Quick Start

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
ta = TradingAgentsGraph(debug=True, config=config)

# Analyze a China A-share
state, decision = ta.propagate("600519.SH", "2024-12-15")
print(decision)
```

## Ticker Format

| Exchange | Suffix | Example |
|----------|--------|---------|
| Shanghai SSE | `.SH` | `600519.SH` (Kweichow Moutai) |
| Shenzhen SZSE | `.SZ` | `000001.SZ` (Ping An Bank) |
| Beijing BSE | `.BJ` | `835305.BJ` (Example stock) |

## How It Works

1. **Automatic Detection**: Tickers ending with `.SH`, `.SZ`, or `.BJ` are automatically routed to AKShare
2. **No API Key Required**: AKShare scrapes public data - no key needed
3. **Chinese Data**: Financial reports and news are in Chinese - LLMs handle them natively

## Supported Data

| Data Type | Source | Notes |
|-----------|--------|-------|
| OHLCV Prices | `stock_zh_a_hist()` | 15+ years historical |
| Technical Indicators | stockstats + AKShare | SMA, EMA, MACD, RSI, Bollinger, etc. |
| Fundamentals | Eastmoney | All available fields |
| Financial Statements | Eastmoney reports | Annual/quarterly filtering |
| News | `stock_news_em()` | Eastmoney news |
| Macro News | CCTV finance | Multi-day collection |
| Insider Transactions | Tonghuashun | Shareholder changes |

## Examples

### Shanghai Stocks
```python
# Kweichow Moutai
state, decision = ta.propagate("600519.SH", "2024-12-15")

# ICBC
state, decision = ta.propagate("601398.SH", "2024-12-15")
```

### Shenzhen Stocks
```python
# Ping An Bank
state, decision = ta.propagate("000001.SZ", "2024-12-15")

# BYD
state, decision = ta.propagate("002594.SZ", "2024-12-15")
```

### Beijing Stocks
```python
# Beijing Stock Exchange
state, decision = ta.propagate("835305.BJ", "2024-12-15")
```

## Limitations

- Network access required (scrapes Eastmoney/Sina)
- Chinese holidays affect trading calendar
- Daily granularity only
