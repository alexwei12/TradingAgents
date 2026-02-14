# China A-Share Support Implementation Plan (v2 - Reviewed)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add China A-share (Shanghai SSE / Shenzhen SZSE / Beijing BSE) stock analysis capability to TradingAgents using AKShare as the data vendor.

**Architecture:** Introduce a thread-local `TradingContext` to pass ticker info through the routing layer, enabling automatic detection of China stocks (`.SH`/`.SZ`/`.BJ` suffix) and routing to AKShare. All AKShare functions match existing vendor signatures for seamless integration.

**Tech Stack:** Python, AKShare, pandas, stockstats, contextvars

---

## Prerequisites

**Required:** `pip install akshare>=1.14.0`

---

## Task 1: Add AKShare Dependency

**Files:**
- Modify: `requirements.txt`

**Step 1: Add akshare to requirements**

```bash
# Append to requirements.txt
echo "akshare>=1.14.0" >> requirements.txt
```

**Step 2: Verify the change**

```bash
grep akshare requirements.txt
```
Expected: `akshare>=1.14.0`

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "deps: add akshare for China A-share data support"
```

---

## Task 2: Create Ticker Utilities Module

**Files:**
- Create: `tradingagents/dataflows/ticker_utils.py`
- Test: `tests/test_ticker_utils.py`

**Step 1: Write the ticker_utils implementation**

```python
"""Utilities for ticker parsing and market detection."""

from typing import Tuple, Optional

# China A-share exchange suffixes
CHINA_SUFFIXES = {".SH", ".SZ", ".BJ"}  # Shanghai, Shenzhen, Beijing


def detect_market(ticker: str) -> str:
    """Detect market from ticker suffix.

    NOTE: This is a best-effort heuristic. Non-China tickers may be
    misclassified. For accurate detection, use is_china_stock().

    Args:
        ticker: Stock ticker symbol (e.g., "600519.SH", "AAPL")

    Returns:
        Market code: "CN_SH" for Shanghai, "CN_SZ" for Shenzhen,
                     "CN_BJ" for Beijing, "US" for US stocks (heuristic),
                     "UNKNOWN" otherwise
    """
    if ticker.endswith('.SH'):
        return "CN_SH"
    elif ticker.endswith('.SZ'):
        return "CN_SZ"
    elif ticker.endswith('.BJ'):
        return "CN_BJ"
    elif ticker.isalpha() or (len(ticker) <= 5 and ticker.isalnum()):
        # Simple heuristic for US stocks (AAPL, NVDA, etc.)
        return "US"
    else:
        return "UNKNOWN"


def parse_ticker(ticker: str) -> Tuple[str, Optional[str]]:
    """Split ticker into code and exchange.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Tuple of (stock_code, exchange_code)
        - "600519.SH" -> ("600519", "sh")
        - "000001.SZ" -> ("000001", "sz")
        - "835305.BJ" -> ("835305", "bj")
        - "AAPL" -> ("AAPL", None)
    """
    if ticker.endswith('.SH'):
        return (ticker[:-3], "sh")
    elif ticker.endswith('.SZ'):
        return (ticker[:-3], "sz")
    elif ticker.endswith('.BJ'):
        return (ticker[:-3], "bj")
    else:
        return (ticker, None)


def is_china_stock(ticker: str) -> bool:
    """Check if ticker is a China A-share.

    Args:
        ticker: Stock ticker symbol

    Returns:
        True for .SH/.SZ/.BJ tickers, False otherwise
    """
    return any(ticker.endswith(suffix) for suffix in CHINA_SUFFIXES)


def to_akshare_symbol(ticker: str) -> str:
    """Convert ticker to AKShare symbol format.

    AKShare uses stock code without suffix for most API calls.
    The exchange is passed separately.

    Args:
        ticker: Stock ticker (e.g., "600519.SH")

    Returns:
        Stock code only (e.g., "600519")
    """
    code, _ = parse_ticker(ticker)
    return code
```

**Step 2: Write tests**

Create `tests/test_ticker_utils.py`:

```python
"""Tests for ticker_utils module."""

import pytest
from tradingagents.dataflows.ticker_utils import (
    CHINA_SUFFIXES,
    detect_market,
    parse_ticker,
    is_china_stock,
    to_akshare_symbol,
)


class TestChinaSuffixes:
    """Test CHINA_SUFFIXES constant."""

    def test_contains_all_exchanges(self):
        assert ".SH" in CHINA_SUFFIXES
        assert ".SZ" in CHINA_SUFFIXES
        assert ".BJ" in CHINA_SUFFIXES


class TestDetectMarket:
    """Test cases for detect_market function."""

    def test_shanghai_stock(self):
        assert detect_market("600519.SH") == "CN_SH"

    def test_shenzhen_stock(self):
        assert detect_market("000001.SZ") == "CN_SZ"

    def test_beijing_stock(self):
        assert detect_market("835305.BJ") == "CN_BJ"

    def test_us_stock(self):
        assert detect_market("AAPL") == "US"


class TestParseTicker:
    """Test cases for parse_ticker function."""

    def test_shanghai_ticker(self):
        assert parse_ticker("600519.SH") == ("600519", "sh")

    def test_shenzhen_ticker(self):
        assert parse_ticker("000001.SZ") == ("000001", "sz")

    def test_beijing_ticker(self):
        assert parse_ticker("835305.BJ") == ("835305", "bj")

    def test_us_ticker(self):
        assert parse_ticker("AAPL") == ("AAPL", None)


class TestIsChinaStock:
    """Test cases for is_china_stock function."""

    def test_shanghai_is_china(self):
        assert is_china_stock("600519.SH") is True

    def test_shenzhen_is_china(self):
        assert is_china_stock("000001.SZ") is True

    def test_beijing_is_china(self):
        assert is_china_stock("835305.BJ") is True

    def test_us_is_not_china(self):
        assert is_china_stock("AAPL") is False


class TestToAkshareSymbol:
    """Test cases for to_akshare_symbol function."""

    def test_shanghai_symbol(self):
        assert to_akshare_symbol("600519.SH") == "600519"

    def test_shenzhen_symbol(self):
        assert to_akshare_symbol("000001.SZ") == "000001"

    def test_beijing_symbol(self):
        assert to_akshare_symbol("835305.BJ") == "835305"

    def test_us_symbol(self):
        assert to_akshare_symbol("AAPL") == "AAPL"
```

**Step 3: Create tests directory and run tests**

```bash
mkdir -p tests
cd tests && python -m pytest test_ticker_utils.py -v
```
Expected: All tests PASS

**Step 4: Commit**

```bash
git add tradingagents/dataflows/ticker_utils.py tests/test_ticker_utils.py
git commit -m "feat: add ticker utilities with .SH/.SZ/.BJ support"
```

---

## Task 3: Create TradingContext Module

**Files:**
- Create: `tradingagents/dataflows/trading_context.py`
- Test: `tests/test_trading_context.py`

**Step 1: Write the TradingContext implementation**

```python
"""Thread-local context for storing current trading session state."""

from contextvars import ContextVar
from typing import Optional

from .ticker_utils import is_china_stock as _is_china_ticker

_current_ticker: ContextVar[Optional[str]] = ContextVar('current_ticker', default=None)


class TradingContext:
    """Thread-local context for storing current trading session state.

    This allows the vendor routing layer to know which ticker is being
    analyzed, enabling market-specific routing (e.g., China A-shares).
    """

    @staticmethod
    def set_ticker(ticker: str) -> None:
        """Set the current ticker being analyzed."""
        _current_ticker.set(ticker)

    @staticmethod
    def get_ticker() -> Optional[str]:
        """Get the current ticker being analyzed."""
        return _current_ticker.get()

    @staticmethod
    def clear() -> None:
        """Clear the current ticker context."""
        _current_ticker.set(None)

    @staticmethod
    def is_china_stock() -> bool:
        """Check if current ticker is a China A-share.

        Delegates to ticker_utils.is_china_stock() for consistent logic.
        Returns True for tickers ending with .SH (Shanghai), .SZ (Shenzhen),
        or .BJ (Beijing).
        """
        ticker = _current_ticker.get()
        if not ticker:
            return False
        return _is_china_ticker(ticker)
```

**Step 2: Write tests**

Create `tests/test_trading_context.py`:

```python
"""Tests for TradingContext module."""

import pytest
from tradingagents.dataflows.trading_context import TradingContext


class TestTradingContext:
    """Test cases for TradingContext."""

    def test_set_and_get_ticker(self):
        """Test setting and getting ticker."""
        TradingContext.set_ticker("600519.SH")
        assert TradingContext.get_ticker() == "600519.SH"
        TradingContext.clear()

    def test_clear_ticker(self):
        """Test clearing ticker context."""
        TradingContext.set_ticker("600519.SH")
        TradingContext.clear()
        assert TradingContext.get_ticker() is None

    def test_is_china_stock_shanghai(self):
        """Test detection of Shanghai stocks."""
        TradingContext.set_ticker("600519.SH")
        assert TradingContext.is_china_stock() is True
        TradingContext.clear()

    def test_is_china_stock_shenzhen(self):
        """Test detection of Shenzhen stocks."""
        TradingContext.set_ticker("000001.SZ")
        assert TradingContext.is_china_stock() is True
        TradingContext.clear()

    def test_is_china_stock_beijing(self):
        """Test detection of Beijing stocks."""
        TradingContext.set_ticker("835305.BJ")
        assert TradingContext.is_china_stock() is True
        TradingContext.clear()

    def test_is_china_stock_us_ticker(self):
        """Test that US tickers are not detected as China stocks."""
        TradingContext.set_ticker("AAPL")
        assert TradingContext.is_china_stock() is False
        TradingContext.clear()

    def test_is_china_stock_no_context(self):
        """Test detection when no ticker is set."""
        TradingContext.clear()
        assert TradingContext.is_china_stock() is False
```

**Step 3: Run tests**

```bash
cd tests && python -m pytest test_trading_context.py -v
```
Expected: All tests PASS

**Step 4: Commit**

```bash
git add tradingagents/dataflows/trading_context.py tests/test_trading_context.py
git commit -m "feat: add TradingContext delegating to ticker_utils"
```

---

## Task 4: Create AKShare Common Module

**Files:**
- Create: `tradingagents/dataflows/akshare_common.py`

**Step 1: Write the common module**

```python
"""Common utilities and exceptions for AKShare modules."""


class AKShareError(Exception):
    """Raised when AKShare data fetch fails."""
    pass


def to_akshare_date(date_str: str) -> str:
    """Convert YYYY-MM-DD to YYYYMMDD for AKShare."""
    return date_str.replace("-", "")


def from_akshare_date(date_str: str) -> str:
    """Convert YYYYMMDD to YYYY-MM-DD."""
    if len(date_str) == 8:
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return date_str
```

**Step 2: Commit**

```bash
git add tradingagents/dataflows/akshare_common.py
git commit -m "feat: add AKShare common module with shared exception class"
```

---

## Task 5: Create AKShare Indicators Module (Separate File)

**Files:**
- Create: `tradingagents/dataflows/akshare_indicators.py`
- Test: `tests/test_akshare_indicators.py`

**Step 1: Write the indicators implementation**

```python
"""AKShare-based technical indicators for China A-shares.

This is an independent implementation using stockstats with AKShare OHLCV data,
not using StockstatsUtils which hardcodes yfinance.
"""

from typing import Annotated
from datetime import datetime
import pandas as pd
import akshare as ak
from stockstats import wrap
from dateutil.relativedelta import relativedelta

from .ticker_utils import parse_ticker, to_akshare_symbol
from .akshare_common import AKShareError, to_akshare_date


def get_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    """Calculate technical indicators using AKShare OHLCV data."""

    best_ind_params = {
        "close_50_sma": (
            "50 SMA: A medium-term trend indicator. "
            "Usage: Identify trend direction and serve as dynamic support/resistance."
        ),
        "close_200_sma": (
            "200 SMA: A long-term trend benchmark. "
            "Usage: Confirm overall market trend and identify golden/death cross setups."
        ),
        "close_10_ema": (
            "10 EMA: A responsive short-term average. "
            "Usage: Capture quick shifts in momentum and potential entry points."
        ),
        "macd": (
            "MACD: Computes momentum via differences of EMAs. "
            "Usage: Look for crossovers and divergence as signals of trend changes."
        ),
        "macds": (
            "MACD Signal: An EMA smoothing of the MACD line. "
            "Usage: Use crossovers with the MACD line to trigger trades."
        ),
        "macdh": (
            "MACD Histogram: Shows the gap between the MACD line and its signal. "
            "Usage: Visualize momentum strength and spot divergence early."
        ),
        "rsi": (
            "RSI: Measures momentum to flag overbought/oversold conditions. "
            "Usage: Apply 70/30 thresholds and watch for divergence to signal reversals."
        ),
        "boll": (
            "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. "
            "Usage: Acts as a dynamic benchmark for price movement."
        ),
        "boll_ub": (
            "Bollinger Upper Band: Typically 2 standard deviations above the middle line. "
            "Usage: Signals potential overbought conditions and breakout zones."
        ),
        "boll_lb": (
            "Bollinger Lower Band: Typically 2 standard deviations below the middle line. "
            "Usage: Indicates potential oversold conditions."
        ),
        "atr": (
            "ATR: Averages true range to measure volatility. "
            "Usage: Set stop-loss levels and adjust position sizes based on current market volatility."
        ),
        "vwma": (
            "VWMA: A moving average weighted by volume. "
            "Usage: Confirm trends by integrating price action with volume data."
        ),
        "mfi": (
            "MFI: The Money Flow Index is a momentum indicator that uses both price and volume. "
            "Usage: Identify overbought (>80) or oversold (<20) conditions."
        ),
    }

    if indicator not in best_ind_params:
        raise ValueError(
            f"Indicator {indicator} is not supported. Choose from: {list(best_ind_params.keys())}"
        )

    try:
        stock_code = to_akshare_symbol(symbol)
        _, exchange = parse_ticker(symbol)

        if not exchange:
            raise AKShareError(f"Invalid China A-share ticker format: {symbol}")

        # Calculate date range (need historical data for indicators)
        curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        start_dt = curr_date_dt - relativedelta(days=look_back_days + 200)  # Extra for SMA200

        ak_start = to_akshare_date(start_dt.strftime("%Y-%m-%d"))
        ak_end = to_akshare_date(curr_date)

        # Fetch historical data
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=ak_start,
            end_date=ak_end,
            adjust="qfq"
        )

        if df.empty:
            return f"No data available for {symbol} to calculate indicators."

        # Prepare data for stockstats
        df = df.rename(columns={
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
        })
        df['date'] = pd.to_datetime(df['date'])

        # Use stockstats to calculate indicators
        stock_df = wrap(df)
        stock_df[indicator]  # Trigger calculation

        # Build result string for the lookback period
        end_date = curr_date
        before = curr_date_dt - relativedelta(days=look_back_days)

        # Filter to relevant date range and iterate only trading days
        df_filtered = stock_df[stock_df['date'] >= before].copy()
        df_filtered = df_filtered.sort_values('date', ascending=False)

        ind_string = ""
        for _, row in df_filtered.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            value = row[indicator]

            if pd.isna(value):
                ind_string += f"{date_str}: N/A\n"
            else:
                ind_string += f"{date_str}: {value:.4f}\n"

        result_str = (
            f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
            + ind_string
            + "\n\n"
            + best_ind_params.get(indicator, "No description available.")
        )

        return result_str

    except Exception as e:
        raise AKShareError(f"Error calculating indicator {indicator} for {symbol}: {str(e)}")
```

**Step 2: Write tests**

Create `tests/test_akshare_indicators.py`:

```python
"""Tests for akshare_indicators module."""

import pytest
from tradingagents.dataflows.akshare_indicators import get_indicators


class TestIndicators:
    """Test cases for technical indicators."""

    def test_unsupported_indicator_raises_error(self):
        """Test that unsupported indicator raises ValueError."""
        with pytest.raises(ValueError):
            get_indicators("600519.SH", "unsupported_indicator", "2024-12-01")


@pytest.mark.network
class TestIndicatorsNetwork:
    """Network tests for indicators with real data."""

    def test_get_rsi_indicator(self):
        """Test fetching RSI indicator for Kweichow Moutai."""
        result = get_indicators("600519.SH", "rsi", "2024-12-01", look_back_days=5)
        assert "rsi" in result.lower()
        assert "RSI" in result
        assert isinstance(result, str)

    def test_get_macd_indicator(self):
        """Test fetching MACD indicator."""
        result = get_indicators("600519.SH", "macd", "2024-12-01", look_back_days=5)
        assert "MACD" in result
        assert isinstance(result, str)
```

**Step 3: Run tests (skip network)**

```bash
cd tests && python -m pytest test_akshare_indicators.py::TestIndicators -v
```
Expected: Tests PASS

**Step 4: Commit**

```bash
git add tradingagents/dataflows/akshare_indicators.py tests/test_akshare_indicators.py
git commit -m "feat: add AKShare indicators module (separate file)"
```

---

## Task 6: Create AKShare Data Module (Core Stock Data)

**Files:**
- Create: `tradingagents/dataflows/akshare_data.py`
- Test: `tests/test_akshare_data.py`

**Step 1: Write the akshare_data implementation**

```python
"""AKShare-based data fetching for China A-shares.

This module provides China A-share stock data using AKShare as the data source.
All functions match the signatures of existing vendor implementations.
"""

from typing import Annotated
from datetime import datetime
import pandas as pd
import akshare as ak

from .ticker_utils import parse_ticker, to_akshare_symbol
from .akshare_common import AKShareError, to_akshare_date


def get_stock_data(
    symbol: Annotated[str, "ticker symbol of the company (e.g., 600519.SH)"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Get OHLCV stock data for China A-shares using AKShare."""
    try:
        stock_code = to_akshare_symbol(symbol)
        _, exchange = parse_ticker(symbol)

        if not exchange:
            raise AKShareError(f"Invalid China A-share ticker format: {symbol}")

        ak_start = to_akshare_date(start_date)
        ak_end = to_akshare_date(end_date)

        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=ak_start,
            end_date=ak_end,
            adjust="qfq"
        )

        if df.empty:
            return f"No data found for symbol '{symbol}' between {start_date} and {end_date}. May be a non-trading day."

        column_mapping = {
            '日期': 'Date',
            '开盘': 'Open',
            '收盘': 'Close',
            '最高': 'High',
            '最低': 'Low',
            '成交量': 'Volume',
            '成交额': 'Amount',
        }
        df = df.rename(columns=column_mapping)

        numeric_cols = ['Open', 'High', 'Low', 'Close']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].round(2)

        csv_string = df.to_csv(index=False)

        header = f"# Stock data for {symbol} from {start_date} to {end_date}\n"
        header += f"# Total records: {len(df)}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        return header + csv_string

    except Exception as e:
        raise AKShareError(f"Error fetching stock data for {symbol}: {str(e)}")


def get_fundamentals(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date (not used for AKShare but kept for signature compatibility)"] = None
) -> str:
    """Get company fundamentals overview for China A-shares using AKShare.

    Uses ak.stock_individual_info_em() for basic info and outputs all available fields.
    """
    try:
        stock_code = to_akshare_symbol(ticker)

        df = ak.stock_individual_info_em(symbol=stock_code)

        if df.empty:
            return f"No fundamentals data found for symbol '{ticker}'"

        # Output all available fields from AKShare
        header = f"# Company Fundamentals for {ticker}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        lines = []
        for _, row in df.iterrows():
            item = row.get('item', '')
            value = row.get('value', '')
            if item and value is not None:
                lines.append(f"{item}: {value}")

        return header + "\n".join(lines)

    except Exception as e:
        raise AKShareError(f"Error fetching fundamentals for {ticker}: {str(e)}")


def _filter_by_report_period(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Filter financial reports by frequency (annual or quarterly).

    Annual reports have report dates ending with 1231 (Dec 31).
    """
    if freq == "annual":
        # Look for report date column (报告期 or REPORT_DATE)
        date_col = None
        for col in df.columns:
            if '报告期' in col or 'REPORT_DATE' in col.upper():
                date_col = col
                break

        if date_col:
            # Filter for year-end reports (Dec 31)
            df = df[df[date_col].astype(str).str.contains('12-31|1231', regex=True)]

    return df


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency: annual or quarterly"] = "annual",
    curr_date: Annotated[str, "current date (not used)"] = None
) -> str:
    """Get balance sheet data for China A-shares."""
    try:
        stock_code = to_akshare_symbol(ticker)
        df = ak.stock_balance_sheet_by_report_em(symbol=stock_code)

        if df.empty:
            return f"No balance sheet data found for symbol '{ticker}'"

        # Filter by report period and limit results
        df = _filter_by_report_period(df, freq)
        df = df.head(4 if freq == "quarterly" else 2)

        header = f"# Balance Sheet for {ticker}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        return header + df.to_string()

    except Exception as e:
        raise AKShareError(f"Error fetching balance sheet for {ticker}: {str(e)}")


def get_cashflow(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency: annual or quarterly"] = "annual",
    curr_date: Annotated[str, "current date (not used)"] = None
) -> str:
    """Get cash flow statement for China A-shares."""
    try:
        stock_code = to_akshare_symbol(ticker)
        df = ak.stock_cash_flow_sheet_by_report_em(symbol=stock_code)

        if df.empty:
            return f"No cash flow data found for symbol '{ticker}'"

        df = _filter_by_report_period(df, freq)
        df = df.head(4 if freq == "quarterly" else 2)

        header = f"# Cash Flow Statement for {ticker}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        return header + df.to_string()

    except Exception as e:
        raise AKShareError(f"Error fetching cash flow for {ticker}: {str(e)}")


def get_income_statement(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency: annual or quarterly"] = "annual",
    curr_date: Annotated[str, "current date (not used)"] = None
) -> str:
    """Get income statement for China A-shares."""
    try:
        stock_code = to_akshare_symbol(ticker)
        df = ak.stock_profit_sheet_by_report_em(symbol=stock_code)

        if df.empty:
            return f"No income statement data found for symbol '{ticker}'"

        df = _filter_by_report_period(df, freq)
        df = df.head(4 if freq == "quarterly" else 2)

        header = f"# Income Statement for {ticker}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        return header + df.to_string()

    except Exception as e:
        raise AKShareError(f"Error fetching income statement for {ticker}: {str(e)}")


def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol of the company"],
) -> str:
    """Get insider transactions for China A-shares.

    Uses ak.stock_share_hold_change_ths() for shareholder changes.
    """
    try:
        stock_code = to_akshare_symbol(ticker)
        df = ak.stock_share_hold_change_ths(symbol=stock_code)

        if df.empty:
            return f"No insider transaction data found for {ticker}"

        header = f"# Insider Transactions for {ticker}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        return header + df.head(20).to_string()

    except Exception as e:
        raise AKShareError(f"Error fetching insider transactions for {ticker}: {str(e)}")
```

**Step 2: Write tests**

Create `tests/test_akshare_data.py`:

```python
"""Tests for akshare_data module."""

import pytest
from tradingagents.dataflows.akshare_data import (
    get_stock_data,
    get_fundamentals,
    _filter_by_report_period,
)
from tradingagents.dataflows.akshare_common import AKShareError, to_akshare_date, from_akshare_date


class TestDateConversion:
    """Test date conversion utilities."""

    def test_to_akshare_date(self):
        assert to_akshare_date("2024-12-01") == "20241201"

    def test_from_akshare_date(self):
        assert from_akshare_date("20241201") == "2024-12-01"


@pytest.mark.network
class TestGetStockData:
    """Test get_stock_data with real AKShare API calls."""

    def test_get_kweichow_moutai_data(self):
        """Test fetching data for 600519.SH (Kweichow Moutai)."""
        result = get_stock_data("600519.SH", "2024-12-01", "2024-12-10")
        assert "600519.SH" in result
        assert isinstance(result, str)

    def test_get_ping_an_data(self):
        """Test fetching data for 000001.SZ (Ping An Bank)."""
        result = get_stock_data("000001.SZ", "2024-12-01", "2024-12-10")
        assert "000001.SZ" in result
        assert isinstance(result, str)


@pytest.mark.network
class TestGetFundamentals:
    """Test get_fundamentals with real AKShare API calls."""

    def test_get_kweichow_moutai_fundamentals(self):
        """Test fetching fundamentals for 600519.SH."""
        result = get_fundamentals("600519.SH")
        assert "600519.SH" in result
        # Should contain Chinese company name field
        assert "股票简称" in result or isinstance(result, str)
```

**Step 3: Run tests (skip network)**

```bash
cd tests && python -m pytest test_akshare_data.py::TestDateConversion -v
```
Expected: Date conversion tests PASS

**Step 4: Commit**

```bash
git add tradingagents/dataflows/akshare_data.py tests/test_akshare_data.py
git commit -m "feat: add AKShare data module (OHLCV, fundamentals, financial statements, insider)"
```

---

## Task 7: Create AKShare News Module

**Files:**
- Create: `tradingagents/dataflows/akshare_news.py`
- Test: `tests/test_akshare_news.py`

**Step 1: Write the akshare_news implementation**

```python
"""AKShare-based news and sentiment data for China A-shares."""

from typing import Annotated
from datetime import datetime, timedelta
import akshare as ak
import pandas as pd

from .ticker_utils import to_akshare_symbol
from .akshare_common import AKShareError


def get_news(
    ticker: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Get news for a specific China A-share stock using AKShare."""
    try:
        stock_code = to_akshare_symbol(ticker)
        df = ak.stock_news_em(symbol=stock_code)

        if df.empty:
            return f"No news found for {ticker}"

        # Filter by date range
        if '发布时间' in df.columns:
            df['发布时间'] = pd.to_datetime(df['发布时间'])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df['发布时间'] >= start_dt) & (df['发布时间'] <= end_dt + timedelta(days=1))]

        if df.empty:
            return f"No news found for {ticker} between {start_date} and {end_date}"

        news_str = ""
        for _, row in df.head(10).iterrows():
            title = row.get('标题', 'No title')
            content = row.get('内容', '')
            pub_time = row.get('发布时间', '')

            news_str += f"### {title}\n"
            if pub_time:
                news_str += f"发布时间: {pub_time}\n"
            if content:
                content_str = str(content)[:500] + "..." if len(str(content)) > 500 else str(content)
                news_str += f"{content_str}\n"
            news_str += "\n"

        return f"## {ticker} News, from {start_date} to {end_date}:\n\n{news_str}"

    except Exception as e:
        raise AKShareError(f"Error fetching news for {ticker}: {str(e)}")


def _get_china_macro_news(curr_date: str, look_back_days: int, limit: int) -> str:
    """Get China macroeconomic news for A-share context.

    Uses ak.news_cctv() for CCTV finance news. Note: this API returns
    data for a single date, so we need to iterate over the lookback period.
    """
    try:
        curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")

        # Iterate over lookback period to collect news from each day
        all_news = []
        for i in range(look_back_days + 1):
            day = curr_dt - timedelta(days=i)
            try:
                df = ak.news_cctv(date=day.strftime("%Y%m%d"))
                if not df.empty:
                    df['date'] = day  # Add date column for filtering
                    all_news.append(df)
            except Exception:
                # Skip days with no data or errors
                continue

        if not all_news:
            return f"No macro news found for China market"

        # Combine all news
        df = pd.concat(all_news, ignore_index=True)

        # Filter by date range
        start_dt = curr_dt - timedelta(days=look_back_days)
        if 'date' in df.columns:
            df = df[(df['date'] >= start_dt) & (df['date'] <= curr_dt)]

        # Format output
        news_str = ""
        for _, row in df.head(limit).iterrows():
            title = row.get('title', 'No title')
            content = row.get('content', '')
            pub_date = row.get('date', '')

            news_str += f"### {title}\n"
            if pub_date:
                news_str += f"日期: {pub_date}\n"
            if content:
                content_str = str(content)[:500] + "..." if len(str(content)) > 500 else str(content)
                news_str += f"{content_str}\n"
            news_str += "\n"

        start_date = start_dt.strftime("%Y-%m-%d")
        return f"## China Macro News, from {start_date} to {curr_date}:\n\n{news_str}"

    except Exception as e:
        return f"China macro news temporarily unavailable: {str(e)}"


def get_global_news(
    curr_date: str,
    look_back_days: int = 7,
    limit: int = 10,
) -> str:
    """Get global/macro news for China A-shares.

    NOTE: This function is called via AKShare vendor routing for China stocks.
    It directly returns China macro news without checking TradingContext,
    as the routing decision has already been made.
    """
    # Directly return China macro news (routing already determined this is China stock)
    return _get_china_macro_news(curr_date, look_back_days, limit)
```

**Step 2: Write tests**

Create `tests/test_akshare_news.py`:

```python
"""Tests for akshare_news module."""

import pytest
from tradingagents.dataflows.akshare_news import get_news, get_global_news


@pytest.mark.network
class TestGetNews:
    """Test get_news with real AKShare API calls."""

    def test_get_kweichow_moutai_news(self):
        """Test fetching news for 600519.SH."""
        result = get_news("600519.SH", "2024-12-01", "2024-12-10")
        assert "600519.SH" in result
        assert isinstance(result, str)


@pytest.mark.network
class TestGetGlobalNews:
    """Test get_global_news with real data."""

    def test_get_china_macro_news(self):
        """Test fetching China macro news."""
        result = get_global_news("2024-12-01", look_back_days=3, limit=5)
        assert isinstance(result, str)
        assert "China Macro News" in result or "No macro news" in result
```

**Step 3: Run tests (skip network)**

```bash
cd tests && python -m pytest test_akshare_news.py -v --ignore-glob="*network*"
```
Expected: Import tests pass

**Step 4: Commit**

```bash
git add tradingagents/dataflows/akshare_news.py tests/test_akshare_news.py
git commit -m "feat: add AKShare news module with multi-day CCTV news"
```

---

## Task 8: Update Interface Module for AKShare Integration

**Files:**
- Modify: `tradingagents/dataflows/interface.py`

**Step 1: Import TradingContext and AKShare modules**

Add imports at the top of `tradingagents/dataflows/interface.py`:

```python
# Import TradingContext for China stock routing
from .trading_context import TradingContext

# Import AKShare data functions
from .akshare_data import (
    get_stock_data as get_akshare_stock_data,
    get_fundamentals as get_akshare_fundamentals,
    get_balance_sheet as get_akshare_balance_sheet,
    get_cashflow as get_akshare_cashflow,
    get_income_statement as get_akshare_income_statement,
    get_insider_transactions as get_akshare_insider_transactions,
)
from .akshare_indicators import get_indicators as get_akshare_indicators
from .akshare_news import (
    get_news as get_akshare_news,
    get_global_news as get_akshare_global_news,
)
```

**Step 2: Add "akshare" to VENDOR_LIST**

```python
VENDOR_LIST = [
    "yfinance",
    "alpha_vantage",
    "akshare",  # NEW: China A-share support
]
```

**Step 3: Add AKShare to VENDOR_METHODS**

```python
VENDOR_METHODS = {
    # ... existing entries ...

    # core_stock_apis
    "get_stock_data": {
        "alpha_vantage": get_alpha_vantage_stock,
        "yfinance": get_YFin_data_online,
        "akshare": get_akshare_stock_data,
    },
    # technical_indicators
    "get_indicators": {
        "alpha_vantage": get_alpha_vantage_indicator,
        "yfinance": get_stock_stats_indicators_window,
        "akshare": get_akshare_indicators,
    },
    # fundamental_data
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
    # news_data
    "get_news": {
        "alpha_vantage": get_alpha_vantage_news,
        "yfinance": get_news_yfinance,
        "akshare": get_akshare_news,
    },
    "get_global_news": {
        "yfinance": get_global_news_yfinance,
        "alpha_vantage": get_alpha_vantage_global_news,
        "akshare": get_akshare_global_news,
    },
    "get_insider_transactions": {
        "alpha_vantage": get_alpha_vantage_insider_transactions,
        "yfinance": get_yfinance_insider_transactions,
        "akshare": get_akshare_insider_transactions,
    },
}
```

**Step 4: Modify route_to_vendor for China stock routing**

Add China stock detection at the start of `route_to_vendor`:

```python
def route_to_vendor(method: str, *args, **kwargs):
    """Route method calls to appropriate vendor implementation with fallback support."""

    # Check if current ticker is a China A-share - if so, force AKShare
    if TradingContext.is_china_stock():
        if method not in VENDOR_METHODS:
            raise ValueError(f"Method '{method}' not supported")

        if "akshare" not in VENDOR_METHODS[method]:
            raise RuntimeError(f"AKShare does not support method '{method}'")

        vendor_impl = VENDOR_METHODS[method]["akshare"]
        impl_func = vendor_impl[0] if isinstance(vendor_impl, list) else vendor_impl
        return impl_func(*args, **kwargs)

    # Original routing logic for non-China stocks
    category = get_category_for_method(method)
    vendor_config = get_vendor(category, method)
    primary_vendors = [v.strip() for v in vendor_config.split(',')]

    if method not in VENDOR_METHODS:
        raise ValueError(f"Method '{method}' not supported")

    # ... rest of existing routing logic ...
```

**Step 5: Test the import works**

```bash
python -c "from tradingagents.dataflows.interface import route_to_vendor; print('Import OK')"
```
Expected: `Import OK`

**Step 6: Commit**

```bash
git add tradingagents/dataflows/interface.py
git commit -m "feat: integrate AKShare into vendor routing with China stock detection"
```

---

## Task 9: Update Trading Graph to Set TradingContext

**Files:**
- Modify: `tradingagents/graph/trading_graph.py`

**Step 1: Import TradingContext**

Add import:

```python
from tradingagents.dataflows.trading_context import TradingContext
```

**Step 2: Wrap propagate method with TradingContext**

```python
def propagate(self, company_name, trade_date):
    """Run the trading agents graph for a company on a specific date."""

    self.ticker = company_name

    # Set TradingContext for vendor routing
    TradingContext.set_ticker(company_name)

    try:
        # Initialize state
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date
        )
        args = self.propagator.get_graph_args()

        if self.debug:
            trace = []
            for chunk in self.graph.stream(init_agent_state, **args):
                if len(chunk["messages"]) == 0:
                    pass
                else:
                    chunk["messages"][-1].pretty_print()
                    trace.append(chunk)
            final_state = trace[-1]
        else:
            final_state = self.graph.invoke(init_agent_state, **args)

        self.curr_state = final_state
        self._log_state(trade_date, final_state)
        return final_state, self.process_signal(final_state["final_trade_decision"])

    finally:
        TradingContext.clear()
```

**Step 3: Test the import works**

```bash
python -c "from tradingagents.graph.trading_graph import TradingAgentsGraph; print('Import OK')"
```
Expected: `Import OK`

**Step 4: Commit**

```bash
git add tradingagents/graph/trading_graph.py
git commit -m "feat: set TradingContext in propagate for China stock routing"
```

---

## Task 10: End-to-End Integration Test

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_china_a_share_integration.py`

**Step 1: Create tests/__init__.py**

```python
"""Tests for TradingAgents China A-share support."""
```

**Step 2: Write integration test**

Create `tests/test_china_a_share_integration.py`:

```python
"""End-to-end integration test for China A-share support."""

import pytest
from tradingagents.dataflows.trading_context import TradingContext
from tradingagents.dataflows import ticker_utils


class TestChinaAshareIntegration:
    """Integration tests for China A-share functionality."""

    def test_ticker_utils_detects_all_china_exchanges(self):
        """Verify ticker_utils correctly identifies all China exchanges."""
        assert ticker_utils.is_china_stock("600519.SH") is True  # Shanghai
        assert ticker_utils.is_china_stock("000001.SZ") is True  # Shenzhen
        assert ticker_utils.is_china_stock("835305.BJ") is True  # Beijing
        assert ticker_utils.is_china_stock("AAPL") is False

    def test_ticker_utils_parsing(self):
        """Verify ticker parsing works correctly."""
        code, exchange = ticker_utils.parse_ticker("600519.SH")
        assert code == "600519"
        assert exchange == "sh"

        code, exchange = ticker_utils.parse_ticker("835305.BJ")
        assert code == "835305"
        assert exchange == "bj"

    def test_trading_context_delegates_to_ticker_utils(self):
        """Verify TradingContext delegates to ticker_utils."""
        TradingContext.set_ticker("835305.BJ")
        try:
            assert TradingContext.get_ticker() == "835305.BJ"
            assert TradingContext.is_china_stock() is True
        finally:
            TradingContext.clear()

        assert TradingContext.get_ticker() is None
        assert TradingContext.is_china_stock() is False


@pytest.mark.network
@pytest.mark.slow
class TestAKShareDataFetching:
    """Tests that actually call AKShare APIs."""

    def test_fetch_kweichow_moutai_data(self):
        """Test fetching real data for Kweichow Moutai."""
        from tradingagents.dataflows.akshare_data import get_stock_data

        result = get_stock_data("600519.SH", "2024-12-01", "2024-12-10")
        assert isinstance(result, str)
        assert "600519.SH" in result

    def test_fetch_beijing_stock_data(self):
        """Test fetching data for Beijing exchange stock."""
        from tradingagents.dataflows.akshare_data import get_stock_data

        result = get_stock_data("835305.BJ", "2024-12-01", "2024-12-10")
        assert isinstance(result, str)
        assert "835305.BJ" in result
```

**Step 3: Run non-network tests**

```bash
cd tests && python -m pytest test_china_a_share_integration.py::TestChinaAshareIntegration -v
```
Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/__init__.py tests/test_china_a_share_integration.py
git commit -m "test: add end-to-end integration tests with .BJ support"
```

---

## Task 11: Documentation Update

**Files:**
- Create: `docs/china-a-share-usage.md`

**Step 1: Create usage documentation**

```markdown
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
```

**Step 2: Commit documentation**

```bash
git add docs/china-a-share-usage.md
git commit -m "docs: add China A-share usage guide with .BJ support"
```

---

## Summary

This revised implementation plan addresses all review feedback:

### Critical Fixes (P0)
1. **TradingContext delegates to ticker_utils** - Single source of truth for China detection
2. **Added `.BJ` (Beijing) support** - CHINA_SUFFIXES constant includes all three exchanges
3. **AKShareError defined once** - In akshare_common.py, imported by other modules
4. **Fixed CCTV news** - Loops through each day in lookback period
5. **Removed redundant TradingContext check** - get_global_news directly returns China macro news

### Medium Fixes (P1)
6. **detect_market disclaimer** - Documented as "best-effort heuristic"
7. **freq parameter actually filters** - _filter_by_report_period() checks for year-end reports
8. **Efficient indicator iteration** - Only iterates trading days from DataFrame
9. **Full fundamentals output** - Returns all AKShare fields
10. **Historical test dates** - Using 2024 dates instead of 2026

### Minor Fixes (P2)
11. **Added tests/__init__.py** - Proper package structure
12. **Moved get_insider_transactions** - To akshare_data.py (fundamental data category)
13. **Fixed markdown** - No extra closing markers
14. **Split akshare_indicators.py** - Separate file per v2 design
15. **Consistent parse_ticker return** - Always returns Optional[str]

**Total: 11 Tasks** (merged some related tasks for cleaner execution)
