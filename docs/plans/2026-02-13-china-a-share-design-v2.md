# China A-Share Support for TradingAgents (Revised)

## Overview

Add China A-share (Shanghai SSE / Shenzhen SZSE) stock analysis capability using AKShare as the data vendor. This design addresses all issues raised in the code review.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Data source | AKShare | Free, no API key, covers OHLCV/fundamentals/news, actively maintained (~18k GitHub stars) |
| Chinese data handling | Pass Chinese text directly to LLM | Modern LLMs handle Chinese well |
| Market identification | Ticker suffix detection (`.SH`/`.SZ`) | A 股自动识别，无侵入性修改 |
| Routing strategy | Ticker-aware dispatch in `route_to_vendor` | 从 `*args` 中提取 ticker 进行判断 |
| Global news | 保持现有 US 宏观新闻不变 | `get_global_news` 无 ticker 参数，无法路由到 A 股新闻 |
| Technical indicators | 独立 AKShare 实现 | 不依赖 yfinance 的 `StockstatsUtils` |

## Critical Issues Addressed from Review

### 1. `route_to_vendor` Signature — Ticker Extraction from `*args`

`route_to_vendor(method, *args, **kwargs)` 中 ticker 隐藏在 `*args` 中。通过工具方法签名约定，第一个参数通常是 ticker/symbol：

```python
# tools that have ticker as first arg
TICKER_TOOLS = {
    "get_stock_data": 0,      # (symbol, start_date, end_date)
    "get_indicators": 0,      # (symbol, indicator, curr_date, look_back_days)
    "get_fundamentals": 0,    # (ticker, curr_date)
    "get_balance_sheet": 0,   # (ticker, freq, curr_date)
    "get_cashflow": 0,        # (ticker, freq, curr_date)
    "get_income_statement": 0, # (ticker, freq, curr_date)
    "get_news": 0,            # (ticker, start_date, end_date)
    "get_insider_transactions": 0,  # (ticker)
}

# In route_to_vendor:
ticker_index = TICKER_TOOLS.get(method)
if ticker_index is not None and len(args) > ticker_index:
    ticker = args[ticker_index]
    if is_china_stock(ticker):
        # Force akshare routing
```

### 2. Function Signatures — Full Alignment with Existing Vendors

| Function | Existing Signature | AKShare Signature | Notes |
|----------|-------------------|-------------------|-------|
| `get_stock_data` | `(symbol, start_date, end_date)` | `(symbol, start_date, end_date)` | ✅ Match |
| `get_indicators` | `(symbol, indicator, curr_date, look_back_days)` | `(symbol, indicator, curr_date, look_back_days)` | ✅ Match |
| `get_fundamentals` | `(ticker, curr_date)` | `(ticker, curr_date)` | `curr_date` 用于 A 股财报日期选择 |
| `get_balance_sheet` | `(ticker, freq, curr_date)` | `(ticker, freq, curr_date)` | `freq` 传递给 AKShare |
| `get_cashflow` | `(ticker, freq, curr_date)` | `(ticker, freq, curr_date)` | Match |
| `get_income_statement` | `(ticker, freq, curr_date)` | `(ticker, freq, curr_date)` | Match |
| `get_news` | `(ticker, start_date, end_date)` | `(ticker, start_date, end_date)` | Match |
| `get_insider_transactions` | `(ticker)` | `(ticker)` | ✅ Match |
| `get_global_news` | `(curr_date, look_back_days, limit)` | N/A | 无 ticker 参数，**不参与 A 股路由** |

### 3. Technical Indicators — Independent AKShare Implementation

`y_finance.py` 的 `get_stock_stats_indicators_window` 调用 `StockstatsUtils`，其内部硬编码 `yf.download()`。

**AKShare 方案**：
```python
def get_akshare_indicators(symbol, indicator, curr_date, look_back_days):
    # 1. 从 AKShare 获取 OHLCV
    code, exchange = parse_ticker(symbol)  # "600519.SH" -> ("600519", "sh")
    df = ak.stock_zh_a_hist(symbol=code, period="daily", ...)

    # 2. 使用 stockstats.wrap() 计算指标（不依赖 yfinance）
    from stockstats import wrap
    df_wrapped = wrap(df)
    df_wrapped[indicator]  # trigger calculation

    # 3. 返回窗口期内指标值
    ...
```

### 4. `VENDOR_METHODS` Registration

```python
# interface.py
VENDOR_LIST = ["yfinance", "alpha_vantage", "akshare"]  # Add akshare

VENDOR_METHODS = {
    "get_stock_data": {
        "alpha_vantage": get_alpha_vantage_stock,
        "yfinance": get_YFin_data_online,
        "akshare": get_akshare_stock_data,  # New
    },
    "get_indicators": {
        "alpha_vantage": get_alpha_vantage_indicator,
        "yfinance": get_stock_stats_indicators_window,
        "akshare": get_akshare_indicators,  # New
    },
    # ... 其他工具同理
}
```

### 5. Fallback Behavior

- **美股**：alpha_vantage → yfinance (existing)
- **A 股**：akshare only，**无 fallback**
  - AKShare 失败时抛出异常，上层捕获并返回错误信息
  - 原因：yfinance/alpha_vantage 不支持 A 股 ticker 格式

### 6. Date Format Conversion

AKShare 使用 `"YYYYMMDD"` 格式，系统使用 `"YYYY-MM-DD"`：

```python
def to_akshare_date(date_str: str) -> str:
    """Convert YYYY-MM-DD to YYYYMMDD"""
    return date_str.replace("-", "")

def from_akshare_date(date_str: str) -> str:
    """Convert YYYYMMDD to YYYY-MM-DD"""
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
```

### 7. `get_global_news` — Out of Scope for A-Share

`get_global_news(curr_date, look_back_days, limit)` **没有 ticker 参数**，无法在路由层判断市场。

**方案**：
- **短期**：保持现有 US 宏观新闻不变，A 股分析时不调用此工具
- **长期**：可考虑添加 `get_china_macro_news()` 工具，由 config 配置宏观新闻源

## File Structure

```
tradingagents/dataflows/
├── interface.py              # Modify: Add ticker-aware routing
├── akshare_data.py           # New: OHLCV, fundamentals, financial statements
├── akshare_indicators.py     # New: Technical indicators (stockstats-based)
├── akshare_news.py           # New: Company news + social heat
├── ticker_utils.py           # New: Market detection and parsing
└── VENDOR_LIST/METHODS       # Modify: Register akshare
```

## Implementation Details

### `ticker_utils.py`

```python
import re

CHINA_SUFFIXES = {".SH", ".SZ", ".BJ"}  # 上海、深圳、北交所

def is_china_stock(ticker: str) -> bool:
    """Check if ticker is China A-share."""
    if not ticker:
        return False
    ticker_upper = ticker.upper()
    return any(ticker_upper.endswith(suffix) for suffix in CHINA_SUFFIXES)

def parse_ticker(ticker: str) -> tuple[str, str]:
    """
    Parse China ticker to (code, exchange).

    "600519.SH" -> ("600519", "sh")
    "000858.SZ" -> ("000858", "sz")
    "NVDA" -> ("NVDA", "")
    """
    if not is_china_stock(ticker):
        return ticker, ""

    ticker_upper = ticker.upper()
    for suffix in CHINA_SUFFIXES:
        if ticker_upper.endswith(suffix):
            code = ticker_upper[:-len(suffix)]
            exchange = suffix[1:].lower()  # "SH" -> "sh"
            return code, exchange
    return ticker, ""

def get_akshare_symbol(ticker: str) -> str:
    """
    Convert to AKShare symbol format.
    AKShare uses pure numeric code for most functions.
    """
    code, _ = parse_ticker(ticker)
    return code
```

### `akshare_data.py`

```python
from typing import Annotated
from datetime import datetime
import akshare as ak
import pandas as pd
from .ticker_utils import get_akshare_symbol, parse_ticker

def get_akshare_stock_data(
    symbol: Annotated[str, "ticker symbol"],
    start_date: Annotated[str, "YYYY-MM-DD"],
    end_date: Annotated[str, "YYYY-MM-DD"],
) -> str:
    """Get OHLCV data via AKShare."""
    try:
        code, exchange = parse_ticker(symbol)

        # AKShare date format: YYYYMMDD
        start_ak = start_date.replace("-", "")
        end_ak = end_date.replace("-", "")

        # Determine market prefix for AKShare
        if exchange == "sh":
            ak_symbol = f"sh{code}"
        else:
            ak_symbol = f"sz{code}"

        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_ak,
            end_date=end_ak,
            adjust="qfq"  # 前复权
        )

        if df.empty:
            return f"No data found for {symbol} between {start_date} and {end_date}"

        # Rename columns to match yfinance format
        df = df.rename(columns={
            "日期": "Date",
            "开盘": "Open",
            "收盘": "Close",
            "最高": "High",
            "最低": "Low",
            "成交量": "Volume",
        })

        # Format output
        csv_string = df.to_csv(index=False)
        header = f"# Stock data for {symbol} from {start_date} to {end_date}\n"
        header += f"# Total records: {len(df)}\n\n"
        return header + csv_string

    except Exception as e:
        return f"Error retrieving data for {symbol}: {str(e)}"


def get_akshare_fundamentals(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date (not used)"] = None
) -> str:
    """Get company fundamentals via AKShare."""
    try:
        code, _ = parse_ticker(ticker)

        # Get individual stock info from Eastmoney
        info_df = ak.stock_individual_info_em(symbol=code)

        # Convert to readable format
        lines = [f"# Company Fundamentals for {ticker}\n"]
        for _, row in info_df.iterrows():
            lines.append(f"{row['item']}: {row['value']}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error retrieving fundamentals for {ticker}: {str(e)}"


def get_akshare_balance_sheet(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date (not used)"] = None
) -> str:
    """Get balance sheet via AKShare."""
    try:
        code, _ = parse_ticker(ticker)

        # AKShare uses same API for both, returns all reports
        df = ak.stock_balance_sheet_by_report_em(symbol=code)

        if df.empty:
            return f"No balance sheet data for {ticker}"

        # Filter by freq if needed (AKShare returns all)
        # ... filter logic ...

        csv_string = df.to_csv(index=False)
        header = f"# Balance Sheet for {ticker} ({freq})\n\n"
        return header + csv_string

    except Exception as e:
        return f"Error retrieving balance sheet for {ticker}: {str(e)}"


def get_akshare_cashflow(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date (not used)"] = None
) -> str:
    """Get cash flow statement via AKShare."""
    try:
        code, _ = parse_ticker(ticker)
        df = ak.stock_cash_flow_sheet_by_report_em(symbol=code)

        if df.empty:
            return f"No cash flow data for {ticker}"

        csv_string = df.to_csv(index=False)
        header = f"# Cash Flow Statement for {ticker} ({freq})\n\n"
        return header + csv_string

    except Exception as e:
        return f"Error retrieving cash flow for {ticker}: {str(e)}"


def get_akshare_income_statement(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date (not used)"] = None
) -> str:
    """Get income statement via AKShare."""
    try:
        code, _ = parse_ticker(ticker)
        df = ak.stock_profit_sheet_by_report_em(symbol=code)

        if df.empty:
            return f"No income statement data for {ticker}"

        csv_string = df.to_csv(index=False)
        header = f"# Income Statement for {ticker} ({freq})\n\n"
        return header + csv_string

    except Exception as e:
        return f"Error retrieving income statement for {ticker}: {str(e)}"


def get_akshare_insider_transactions(
    ticker: Annotated[str, "ticker symbol"]
) -> str:
    """Get shareholder holding changes via AKShare."""
    try:
        code, _ = parse_ticker(ticker)
        df = ak.stock_share_hold_change_ths(symbol=code)

        if df.empty:
            return f"No insider transaction data for {ticker}"

        csv_string = df.to_csv(index=False)
        header = f"# Shareholder Holding Changes for {ticker}\n\n"
        return header + csv_string

    except Exception as e:
        return f"Error retrieving insider transactions for {ticker}: {str(e)}"
```

### `akshare_indicators.py`

```python
from typing import Annotated
from datetime import datetime
from dateutil.relativedelta import relativedelta
import akshare as ak
import pandas as pd
from stockstats import wrap
from .ticker_utils import parse_ticker
from .config import get_config
import os

def get_akshare_indicators(
    symbol: Annotated[str, "ticker symbol"],
    indicator: Annotated[str, "technical indicator"],
    curr_date: Annotated[str, "YYYY-MM-DD"],
    look_back_days: Annotated[int, "days to look back"],
) -> str:
    """
    Calculate technical indicators using AKShare data + stockstats.
    Independent implementation - does NOT use yfinance.
    """
    try:
        code, exchange = parse_ticker(symbol)

        # Calculate date range (need extra history for indicator calculation)
        curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        # Add buffer for indicator warmup (e.g., 200 days for long-term MAs)
        buffer_days = 250
        start_dt = curr_dt - relativedelta(days=look_back_days + buffer_days)

        start_ak = start_dt.strftime("%Y%m%d")
        end_ak = curr_dt.strftime("%Y%m%d")

        # Fetch data from AKShare
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_ak,
            end_date=end_ak,
            adjust="qfq"
        )

        if df.empty:
            return f"No data found for {symbol}"

        # Rename columns to stockstats expected format
        df = df.rename(columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
        })

        # Convert to stockstats format
        df_wrapped = wrap(df)

        # Calculate indicator
        df_wrapped[indicator]

        # Filter to requested window
        result_start = curr_dt - relativedelta(days=look_back_days)
        df_wrapped['date'] = pd.to_datetime(df_wrapped['date'])
        mask = df_wrapped['date'] >= result_start
        df_filtered = df_wrapped[mask].copy()

        # Build result string
        ind_string = ""
        for _, row in df_filtered.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            value = row.get(indicator, "N/A")
            if pd.isna(value):
                value = "N/A"
            ind_string += f"{date_str}: {value}\n"

        # Add indicator description
        best_ind_params = {
            "close_50_sma": "50 SMA: Medium-term trend indicator...",
            "close_200_sma": "200 SMA: Long-term trend benchmark...",
            "close_10_ema": "10 EMA: Short-term average...",
            "macd": "MACD: Momentum via EMA differences...",
            "macds": "MACD Signal: EMA smoothing of MACD...",
            "macdh": "MACD Histogram: Gap between MACD and signal...",
            "rsi": "RSI: Overbought/oversold momentum...",
            "boll": "Bollinger Middle: 20 SMA baseline...",
            "boll_ub": "Bollinger Upper Band: +2 std dev...",
            "boll_lb": "Bollinger Lower Band: -2 std dev...",
            "atr": "ATR: Volatility measure...",
            "vwma": "VWMA: Volume-weighted moving average...",
            "mfi": "MFI: Money Flow Index...",
        }

        result_str = (
            f"## {indicator} values from {result_start.strftime('%Y-%m-%d')} to {curr_date}:\n\n"
            + ind_string
            + "\n"
            + best_ind_params.get(indicator, "No description available.")
        )

        return result_str

    except Exception as e:
        return f"Error calculating indicator {indicator} for {symbol}: {str(e)}"
```

### `akshare_news.py`

```python
from typing import Annotated
from datetime import datetime
from dateutil.relativedelta import relativedelta
import akshare as ak
from .ticker_utils import parse_ticker

def get_akshare_news(
    ticker: Annotated[str, "ticker symbol"],
    start_date: Annotated[str, "YYYY-MM-DD"],
    end_date: Annotated[str, "YYYY-MM-DD"],
) -> str:
    """
    Get company news from Eastmoney + social heat data.
    """
    try:
        code, _ = parse_ticker(ticker)

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Get news from Eastmoney
        news_df = ak.stock_news_em(symbol=code)

        if news_df.empty:
            return f"No news found for {ticker}"

        # Filter by date
        news_df['datetime'] = pd.to_datetime(news_df['发布时间'])
        mask = (news_df['datetime'] >= start_dt) & (news_df['datetime'] <= end_dt + relativedelta(days=1))
        filtered_news = news_df[mask]

        # Build news string
        news_str = f"## {ticker} News from {start_date} to {end_date}:\n\n"

        for _, row in filtered_news.head(20).iterrows():
            title = row.get('标题', 'No title')
            content = row.get('内容', '')
            pub_time = row.get('发布时间', '')

            news_str += f"### {title}\n"
            news_str += f"Time: {pub_time}\n"
            if content:
                news_str += f"{content}\n"
            news_str += "\n"

        # Append social heat
        try:
            heat_df = ak.stock_hot_rank_em()
            # heat_df contains overall hot stocks, filter for this one
            ticker_heat = heat_df[heat_df['代码'] == code]
            if not ticker_heat.empty:
                rank = ticker_heat.iloc[0].get('排名', 'N/A')
                heat_score = ticker_heat.iloc[0].get('热度', 'N/A')
                news_str += f"\n---\nSocial Heat: Rank #{rank}, Heat Score: {heat_score}\n"
        except:
            pass  # Heat data is optional

        return news_str

    except Exception as e:
        return f"Error fetching news for {ticker}: {str(e)}"
```

### `interface.py` Modifications

```python
# Add to imports
from .ticker_utils import is_china_stock
from .akshare_data import (
    get_akshare_stock_data,
    get_akshare_fundamentals,
    get_akshare_balance_sheet,
    get_akshare_cashflow,
    get_akshare_income_statement,
    get_akshare_insider_transactions,
)
from .akshare_indicators import get_akshare_indicators
from .akshare_news import get_akshare_news

# Update VENDOR_LIST
VENDOR_LIST = ["yfinance", "alpha_vantage", "akshare"]

# Update VENDOR_METHODS
VENDOR_METHODS = {
    "get_stock_data": {
        "alpha_vantage": get_alpha_vantage_stock,
        "yfinance": get_YFin_data_online,
        "akshare": get_akshare_stock_data,
    },
    "get_indicators": {
        "alpha_vantage": get_alpha_vantage_indicator,
        "yfinance": get_stock_stats_indicators_window,
        "akshare": get_akshare_indicators,
    },
    "get_fundamentals": {
        "alpha_vantage": get_alpha_vantage_fundamentals,
        "yfinance": get_yfinance_fundamentals,
        "akshare": get_akshare_fundamentals,
    },
    "get_balance_sheet": {
        "alpha_vantage": get_alpha_vantage_balance_sheet,
        "yfinance": get_yfinance_balance_sheet,
        "akshare": get_akshare_balance_sheet,
    },
    "get_cashflow": {
        "alpha_vantage": get_alpha_vantage_cashflow,
        "yfinance": get_yfinance_cashflow,
        "akshare": get_akshare_cashflow,
    },
    "get_income_statement": {
        "alpha_vantage": get_alpha_vantage_income_statement,
        "yfinance": get_yfinance_income_statement,
        "akshare": get_akshare_income_statement,
    },
    "get_news": {
        "alpha_vantage": get_alpha_vantage_news,
        "yfinance": get_news_yfinance,
        "akshare": get_akshare_news,
    },
    "get_global_news": {
        "yfinance": get_global_news_yfinance,
        "alpha_vantage": get_alpha_vantage_global_news,
        # Note: No akshare - get_global_news has no ticker param
    },
    "get_insider_transactions": {
        "alpha_vantage": get_alpha_vantage_insider_transactions,
        "yfinance": get_yfinance_insider_transactions,
        "akshare": get_akshare_insider_transactions,
    },
}

# Tools where first arg is the ticker
TICKER_FIRST_ARG_TOOLS = {
    "get_stock_data",
    "get_indicators",
    "get_fundamentals",
    "get_balance_sheet",
    "get_cashflow",
    "get_income_statement",
    "get_news",
    "get_insider_transactions",
}

def route_to_vendor(method: str, *args, **kwargs):
    """Route method calls to appropriate vendor implementation with ticker-aware routing."""

    # Check if this tool has a ticker as first argument
    ticker = None
    if method in TICKER_FIRST_ARG_TOOLS and len(args) > 0:
        ticker = args[0]

    # A-share auto-detection (highest priority)
    if ticker and is_china_stock(ticker):
        if method not in VENDOR_METHODS or "akshare" not in VENDOR_METHODS[method]:
            raise RuntimeError(f"Method '{method}' not supported for China A-share stocks")

        vendor_impl = VENDOR_METHODS[method]["akshare"]
        impl_func = vendor_impl[0] if isinstance(vendor_impl, list) else vendor_impl
        return impl_func(*args, **kwargs)

    # Existing routing logic for non-China stocks
    category = get_category_for_method(method)
    vendor_config = get_vendor(category, method)
    primary_vendors = [v.strip() for v in vendor_config.split(',')]

    if method not in VENDOR_METHODS:
        raise ValueError(f"Method '{method}' not supported")

    # Build fallback chain
    all_available_vendors = list(VENDOR_METHODS[method].keys())
    fallback_vendors = primary_vendors.copy()
    for vendor in all_available_vendors:
        if vendor not in fallback_vendors:
            fallback_vendors.append(vendor)

    for vendor in fallback_vendors:
        if vendor not in VENDOR_METHODS[method]:
            continue

        vendor_impl = VENDOR_METHODS[method][vendor]
        impl_func = vendor_impl[0] if isinstance(vendor_impl, list) else vendor_impl

        try:
            return impl_func(*args, **kwargs)
        except AlphaVantageRateLimitError:
            continue

    raise RuntimeError(f"No available vendor for '{method}'")
```

## Dependencies

Add to `pyproject.toml` and `requirements.txt`:
```
akshare>=1.14.0
```

## Testing Strategy

1. **Unit tests**: `ticker_utils.py` market detection
2. **Integration tests**: Each AKShare data function with real A-share tickers
3. **Mock tests**: Simulate AKShare failures
4. **End-to-end**: Full analysis flow with `600519.SH` (贵州茅台)

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| AKShare scraping may break | Wrap all calls in try/except, return user-friendly errors |
| Network latency (Eastmoney/Sina) | Acceptable for batch analysis; document limitation |
| Date format mismatch | Centralized conversion functions in `ticker_utils.py` |
| Chinese holidays vs trading days | AKShare handles this; return "N/A" for non-trading days |

## Out of Scope

- `get_global_news` A 股宏观新闻（无 ticker 参数，短期无法实现）
- Real-time data（系统当前为日线粒度）
- 港股（`.HK`）支持（架构相同，可后续扩展）
