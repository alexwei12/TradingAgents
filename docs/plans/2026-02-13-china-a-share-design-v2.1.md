# China A-Share Support for TradingAgents (v2.1)

## Overview

Add China A-share (Shanghai SSE / Shenzhen SZSE) stock analysis capability to TradingAgents, using AKShare as the data vendor. The implementation extends the existing vendor routing framework with minimal invasive changes.

## Key Changes from v1

| Issue | v1 Solution | v2.1 Solution |
|-------|-------------|---------------|
| `route_to_vendor` lacks ticker parameter | Assumed ticker in `*args` | **Thread-local context** storing current ticker |
| Function signature mismatch | Not addressed | **Aligned signatures** matching existing vendors |
| `StockstatsUtils` hardcodes yfinance | Not addressed | **Independent AKShare implementation** using stockstats with AKShare OHLCV |
| `VENDOR_METHODS` registration | Not addressed | **Full registration** in both VENDOR_METHODS and VENDOR_LIST |
| `get_global_news` routing | Not addressed | **Context-aware routing** via thread-local ticker |
| Date format conversion | Not addressed | **Explicit conversion** in akshare_data.py |

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Data source | AKShare | Free, no API key, covers OHLCV/fundamentals/news/social sentiment |
| Chinese data handling | Feed Chinese text directly to LLM | Modern LLMs handle Chinese well |
| Market identification | Ticker suffix auto-detection (`.SH`/`.SZ`) | Zero config, backward compatible |
| Ticker passing mechanism | Thread-local context (`TradingContext`) | Solves `get_global_news` ticker-less problem |
| Indicators implementation | Independent from `StockstatsUtils` | Avoids yfinance hardcoding |

## Architecture

### Thread-Local Context Pattern

Introduce a thread-local context to store the current ticker being analyzed:

```python
# tradingagents/dataflows/trading_context.py
import contextvars

_current_ticker: contextvars.ContextVar[str] = contextvars.ContextVar('current_ticker', default='')

def set_current_ticker(ticker: str):
    _current_ticker.set(ticker)

def get_current_ticker() -> str:
    return _current_ticker.get()
```

This context is set at the start of `TradingAgentsGraph.propagate()` and cleared after completion. All vendor routing can now query this context.

### Data Flow

```
propagate("600519.SH", "2026-01-15")
    │
    ▼
TradingContext.set_current_ticker("600519.SH")  # NEW
    │
    ▼
ticker_utils.detect_market("600519.SH") → "CN_SH"
    │
    ▼
route_to_vendor() → checks TradingContext → routes to "akshare"
    │
    ▼
akshare_data.py / akshare_news.py
    │
    ▼
Analyst agents receive data (Chinese text) → LLM analyzes
```

### Routing Priority

1. **Check TradingContext** for current ticker → if A-share → force akshare
2. **Check explicit config** (tool_vendors / data_vendors) for non-CN tickers
3. **Fallback chain** follows existing vendor ordering

## New Files

### `tradingagents/dataflows/trading_context.py`

Thread-local context for passing ticker to routing layer:

```python
from contextvars import ContextVar
from typing import Optional

_current_ticker: ContextVar[Optional[str]] = ContextVar('current_ticker', default=None)

class TradingContext:
    """Thread-local context for storing current trading session state."""
    
    @staticmethod
    def set_ticker(ticker: str) -> None:
        _current_ticker.set(ticker)
    
    @staticmethod
    def get_ticker() -> Optional[str]:
        return _current_ticker.get()
    
    @staticmethod
    def clear() -> None:
        _current_ticker.set(None)
    
    @staticmethod
    def is_china_stock() -> bool:
        """Check if current ticker is a China A-share."""
        ticker = _current_ticker.get()
        if not ticker:
            return False
        return ticker.endswith('.SH') or ticker.endswith('.SZ')
```

### `tradingagents/dataflows/ticker_utils.py`

Market detection and parsing utilities:

```python
def detect_market(ticker: str) -> str:
    """Detect market from ticker suffix.
    
    Returns: "CN_SH", "CN_SZ", "US", etc.
    """

def parse_ticker(ticker: str) -> tuple[str, str]:
    """Split ticker into code and exchange.
    
    "600519.SH" → ("600519", "sh")
    "NVDA" → ("NVDA", "")
    """

def is_china_stock(ticker: str) -> bool:
    """Shortcut: returns True for .SH/.SZ tickers."""
```

### `tradingagents/dataflows/akshare_data.py`

Core stock data and fundamentals. **All functions match existing vendor signatures:**

| Function | AKShare API | Signature (aligned) |
|----------|------------|-------------------|
| `get_stock_data` | `ak.stock_zh_a_hist()` | `(symbol, start_date, end_date)` |
| `get_indicators` | OHLCV + `stockstats.wrap()` | `(symbol, indicator, curr_date)` |
| `get_fundamentals` | `ak.stock_individual_info_em()` | `(ticker, curr_date)` |
| `get_balance_sheet` | `ak.stock_balance_sheet_by_report_em()` | `(ticker, freq, curr_date)` |
| `get_cashflow` | `ak.stock_cash_flow_sheet_by_report_em()` | `(ticker, freq, curr_date)` |
| `get_income_statement` | `ak.stock_profit_sheet_by_report_em()` | `(ticker, freq, curr_date)` |

**Date format conversion:**
```python
def _to_akshare_date(date_str: str) -> str:
    """Convert YYYY-MM-DD to YYYYMMDD for AKShare."""
    return date_str.replace("-", "")

def _from_akshare_date(date_str: str) -> str:
    """Convert YYYYMMDD to YYYY-MM-DD."""
    if len(date_str) == 8:
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return date_str
```

**Independent indicators implementation (not using StockstatsUtils):**
```python
def get_akshare_indicators(symbol: str, indicator: str, curr_date: str) -> str:
    """Calculate technical indicators using AKShare OHLCV data.
    
    NOTE: This is an INDEPENDENT implementation, not using StockstatsUtils
    which hardcodes yfinance. Uses stockstats with AKShare data source.
    """
    # 1. Get OHLCV via akshare
    akshare_date = _to_akshare_date(curr_date)
    end_date = pd.Timestamp.today().strftime("%Y%m%d")
    start_date = (pd.Timestamp.today() - pd.DateOffset(years=15)).strftime("%Y%m%d")
    
    df = ak.stock_zh_a_hist(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        adjust="qfq"
    )
    
    # 2. Use stockstats with this data
    df = wrap(df)
    df[indicator]  # trigger calculation
    
    # 3. Return matching row
    curr_date_str = pd.to_datetime(curr_date).strftime("%Y-%m-%d")
    matching = df[df['日期'].str.startswith(curr_date_str)]
    # ... return indicator value
```

### `tradingagents/dataflows/akshare_news.py`

News, macro, insider, and social sentiment data:

| Function | AKShare API | Signature |
|----------|------------|-----------|
| `get_news` | `ak.stock_news_em()` | `(ticker, start_date, end_date)` |
| `get_global_news` | `ak.news_cctv()` / `ak.stock_zh_a_alerts_cls()` | `(curr_date, look_back_days, limit)` |
| `get_insider_transactions` | `ak.stock_share_hold_change_ths()` | `(ticker)` |

**`get_global_news` routing (checks TradingContext):**
```python
def get_global_news(curr_date: str, look_back_days: int = 7, limit: int = 5) -> str:
    """Get global news. Routes to Chinese macro for A-shares."""
    from .trading_context import TradingContext
    
    if TradingContext.is_china_stock():
        # Return Chinese macro news for A-share tickers
        return _get_china_macro_news(curr_date, look_back_days, limit)
    else:
        # Existing behavior: Federal Reserve / global macro
        return _get_global_macro_news(curr_date, look_back_days, limit)
```

## Modified Files

### `tradingagents/dataflows/interface.py`

**1. Import TradingContext:**
```python
from .trading_context import TradingContext
```

**2. Modify `route_to_vendor` to check TradingContext:**
```python
def route_to_vendor(method: str, *args, **kwargs):
    category = get_category_for_method(method)
    
    # Check TradingContext for China stock routing
    if TradingContext.is_china_stock():
        vendor = "akshare"  # Force akshare for A-shares
    else:
        vendor_config = get_vendor(category, method)
        vendor = vendor_config.strip()
    
    # ... rest of routing logic
```

**3. Register AKShare in `VENDOR_LIST`:**
```python
VENDOR_LIST = [
    "yfinance",
    "alpha_vantage",
    "akshare",  # NEW
]
```

**4. Register AKShare in `VENDOR_METHODS`:**
```python
from .akshare_data import (
    get_stock_data as get_akshare_stock_data,
    get_indicators as get_akshare_indicators,
    get_fundamentals as get_akshare_fundamentals,
    get_balance_sheet as get_akshare_balance_sheet,
    get_cashflow as get_akshare_cashflow,
    get_income_statement as get_akshare_income_statement,
)
from .akshare_news import (
    get_news as get_akshare_news,
    get_global_news as get_akshare_global_news,
    get_insider_transactions as get_akshare_insider_transactions,
)

VENDOR_METHODS = {
    # ... existing entries ...
    
    # AKShare additions
    "get_stock_data": {
        "akshare": get_akshare_stock_data,
    },
    "get_indicators": {
        "akshare": get_akshare_indicators,
    },
    "get_fundamentals": {
        "akshare": get_akshare_fundamentals,
    },
    "get_balance_sheet": {
        "akshare": get_akshare_balance_sheet,
    },
    "get_cashflow": {
        "akshare": get_akshare_cashflow,
    },
    "get_income_statement": {
        "akshare": get_akshare_income_statement,
    },
    "get_news": {
        "akshare": get_akshare_news,
    },
    "get_global_news": {
        "akshare": get_akshare_global_news,
    },
    "get_insider_transactions": {
        "akshare": get_akshare_insider_transactions,
    },
}
```

### `tradingagents/graph/trading_graph.py`

**Set TradingContext at propagate start:**
```python
def propagate(self, company: str, date: str, ...):
    from tradingagents.dataflows.trading_context import TradingContext
    
    try:
        TradingContext.set_ticker(company)
        # ... existing propagate logic
    finally:
        TradingContext.clear()  # Always clean up
```

### `pyproject.toml` + `requirements.txt`

Add dependency:
```
akshare>=1.14.0
```

### DuckDB Cache Extension (Optional Enhancement)

Leverage existing DuckDB cache plan for AKShare data:

```python
# In akshare_data.py, add caching decorator
from functools import lru_cache
import hashlib

def _cache_key(prefix: str, *args) -> str:
    args_str = "_".join(str(a) for a in args)
    return f"{prefix}_{hashlib.md5(args_str.encode()).hexdigest()}"
```

## Error Handling & Fallback

### Fallback Strategy

| Scenario | Behavior |
|----------|----------|
| AKShare fails (network/parsing) | Raise `AKShareError`, do NOT fallback to yfinance (unsupported) |
| AlphaVantage rate limit | Fallback to yfinance (existing behavior) |
| YFinance fails | Raise error (existing behavior) |

**New error class:**
```python
class AKShareError(Exception):
    """Raised when AKShare data fetch fails."""
    pass
```

### Graceful Degradation

- If AKShare returns empty data → return friendly message: `"No data available for {ticker} on {date}. May be a non-trading day."`
- If network timeout → return: `"Data service temporarily unavailable. Please try again later."`

## Prerequisites & Risks

| Item | Status | Mitigation |
|------|--------|------------|
| AKShare library | Not installed | `pip install akshare` — no API key needed |
| A-share trading calendar | Chinese holidays differ | If date falls on non-trading day, AKShare returns empty data |
| Network access | AKShare scrapes Eastmoney/Sina | Document limitation; works well in domestic network |
| Financial report format | CAS field names in Chinese | Pass Chinese text directly to LLM |
| Date format | AKShare uses YYYYMMDD | Convert in `akshare_data.py` |

## Testing Plan

| Test Type | Coverage |
|-----------|----------|
| Unit tests | `ticker_utils.py` - market detection |
| Unit tests | `trading_context.py` - context set/get/clear |
| Integration tests | `akshare_data.py` - real data fetch for 600519.SH |
| Integration tests | End-to-end `propagate("600519.SH", "2026-01-15")` |
| Mock tests | AKShare unavailable - graceful error handling |
| Mock tests | `get_global_news` with China ticker → Chinese macro |

## Out of Scope

- Hong Kong stocks (`.HK`) — can be added later
- Real-time/intraday data — current system is daily granularity
- Config-based A-share vendor selection — A-share always uses AKShare
- A-share → yfinance fallback — yfinance doesn't support A-shares
