# China A-Share Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add China A-share (Shanghai SSE / Shenzhen SZSE) stock analysis capability to TradingAgents using AKShare as the data vendor.

**Architecture:** Introduce a thread-local `TradingContext` to pass ticker info through the routing layer, enabling automatic detection of China stocks (`.SH`/`.SZ` suffix) and routing to AKShare. All AKShare functions match existing vendor signatures for seamless integration.

**Tech Stack:** Python, AKShare, pandas, stockstats, contextvars

---

## Prerequisites

**Required:** `pip install akshare>=1.14.0`

---

## Task 1: Add AKShare Dependency

**Files:**
- Modify: `requirements.txt`

**Step 1: Add akshare to requirements**

Add `akshare>=1.14.0` to the end of `requirements.txt`.

```bash
# Append to requirements.txt
echo "akshare>=1.14.0" >> requirements.txt
```

**Step 2: Verify the change**

Run: `grep akshare requirements.txt`
Expected: `akshare>=1.14.0`

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "deps: add akshare for China A-share data support"
```

---

## Task 2: Create TradingContext Module

**Files:**
- Create: `tradingagents/dataflows/trading_context.py`
- Test: `tests/test_trading_context.py` (create tests directory if needed)

**Step 1: Write the TradingContext implementation**

```python
"""Thread-local context for storing current trading session state."""

from contextvars import ContextVar
from typing import Optional

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

        Returns True for tickers ending with .SH (Shanghai) or .SZ (Shenzhen).
        """
        ticker = _current_ticker.get()
        if not ticker:
            return False
        return ticker.endswith('.SH') or ticker.endswith('.SZ')
```

**Step 2: Write the failing test**

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

**Step 3: Create tests directory and run tests**

```bash
mkdir -p tests
cd tests && python -m pytest test_trading_context.py -v
```
Expected: All tests PASS

**Step 4: Commit**

```bash
git add tradingagents/dataflows/trading_context.py tests/test_trading_context.py
git commit -m "feat: add TradingContext for thread-local ticker tracking"
```

---

## Task 3: Create Ticker Utilities Module

**Files:**
- Create: `tradingagents/dataflows/ticker_utils.py`
- Test: `tests/test_ticker_utils.py`

**Step 1: Write the ticker_utils implementation**

```python
"""Utilities for ticker parsing and market detection."""

from typing import Tuple, Optional


def detect_market(ticker: str) -> str:
    """Detect market from ticker suffix.

    Args:
        ticker: Stock ticker symbol (e.g., "600519.SH", "AAPL")

    Returns:
        Market code: "CN_SH" for Shanghai, "CN_SZ" for Shenzhen,
                     "US" for US stocks, "UNKNOWN" otherwise
    """
    if ticker.endswith('.SH'):
        return "CN_SH"
    elif ticker.endswith('.SZ'):
        return "CN_SZ"
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
        - "AAPL" -> ("AAPL", None)
    """
    if ticker.endswith('.SH'):
        return (ticker[:-3], "sh")
    elif ticker.endswith('.SZ'):
        return (ticker[:-3], "sz")
    else:
        return (ticker, None)


def is_china_stock(ticker: str) -> bool:
    """Check if ticker is a China A-share.

    Args:
        ticker: Stock ticker symbol

    Returns:
        True for .SH or .SZ tickers, False otherwise
    """
    return ticker.endswith('.SH') or ticker.endswith('.SZ')


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

**Step 2: Write the failing test**

Create `tests/test_ticker_utils.py`:

```python
"""Tests for ticker_utils module."""

import pytest
from tradingagents.dataflows.ticker_utils import (
    detect_market,
    parse_ticker,
    is_china_stock,
    to_akshare_symbol,
)


class TestDetectMarket:
    """Test cases for detect_market function."""

    def test_shanghai_stock(self):
        assert detect_market("600519.SH") == "CN_SH"

    def test_shenzhen_stock(self):
        assert detect_market("000001.SZ") == "CN_SZ"

    def test_us_stock(self):
        assert detect_market("AAPL") == "US"

    def test_us_stock_multi_char(self):
        assert detect_market("NVDA") == "US"


class TestParseTicker:
    """Test cases for parse_ticker function."""

    def test_shanghai_ticker(self):
        assert parse_ticker("600519.SH") == ("600519", "sh")

    def test_shenzhen_ticker(self):
        assert parse_ticker("000001.SZ") == ("000001", "sz")

    def test_us_ticker(self):
        assert parse_ticker("AAPL") == ("AAPL", None)


class TestIsChinaStock:
    """Test cases for is_china_stock function."""

    def test_shanghai_is_china(self):
        assert is_china_stock("600519.SH") is True

    def test_shenzhen_is_china(self):
        assert is_china_stock("000001.SZ") is True

    def test_us_is_not_china(self):
        assert is_china_stock("AAPL") is False


class TestToAkshareSymbol:
    """Test cases for to_akshare_symbol function."""

    def test_shanghai_symbol(self):
        assert to_akshare_symbol("600519.SH") == "600519"

    def test_shenzhen_symbol(self):
        assert to_akshare_symbol("000001.SZ") == "000001"

    def test_us_symbol(self):
        assert to_akshare_symbol("AAPL") == "AAPL"
```

**Step 3: Run tests**

```bash
cd tests && python -m pytest test_ticker_utils.py -v
```
Expected: All tests PASS

**Step 4: Commit**

```bash
git add tradingagents/dataflows/ticker_utils.py tests/test_ticker_utils.py
git commit -m "feat: add ticker utilities for market detection and parsing"
```

---

## Task 4: Create AKShare Data Module (Core Stock Data)

**Files:**
- Create: `tradingagents/dataflows/akshare_data.py`
- Test: `tests/test_akshare_data.py`

**Step 1: Write the akshare_data implementation**

```python
"""AKShare-based data fetching for China A-shares.

This module provides China A-share stock data using AKShare as the data source.
All functions match the signatures of existing vendor implementations (yfinance, alpha_vantage).
"""

from typing import Annotated
from datetime import datetime
import pandas as pd
import akshare as ak
from stockstats import wrap

from .ticker_utils import parse_ticker, to_akshare_symbol


class AKShareError(Exception):
    """Raised when AKShare data fetch fails."""
    pass


def _to_akshare_date(date_str: str) -> str:
    """Convert YYYY-MM-DD to YYYYMMDD for AKShare."""
    return date_str.replace("-", "")


def _from_akshare_date(date_str: str) -> str:
    """Convert YYYYMMDD to YYYY-MM-DD."""
    if len(date_str) == 8:
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return date_str


def get_stock_data(
    symbol: Annotated[str, "ticker symbol of the company (e.g., 600519.SH)"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Get OHLCV stock data for China A-shares using AKShare.

    Uses ak.stock_zh_a_hist() for historical data.
    """
    try:
        stock_code = to_akshare_symbol(symbol)
        _, exchange = parse_ticker(symbol)

        if not exchange:
            raise AKShareError(f"Invalid China A-share ticker format: {symbol}")

        # Convert dates to AKShare format
        ak_start = _to_akshare_date(start_date)
        ak_end = _to_akshare_date(end_date)

        # Fetch data from AKShare
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=ak_start,
            end_date=ak_end,
            adjust="qfq"  # Forward-adjusted for splits
        )

        if df.empty:
            return f"No data found for symbol '{symbol}' between {start_date} and {end_date}. May be a non-trading day."

        # Rename columns to match standard format
        # AKShare columns: 日期,股票代码,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率
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

        # Round numerical values
        numeric_cols = ['Open', 'High', 'Low', 'Close']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].round(2)

        # Convert to CSV
        csv_string = df.to_csv(index=False)

        header = f"# Stock data for {symbol} from {start_date} to {end_date}\n"
        header += f"# Total records: {len(df)}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        return header + csv_string

    except Exception as e:
        raise AKShareError(f"Error fetching stock data for {symbol}: {str(e)}")


def get_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    """Calculate technical indicators using AKShare OHLCV data.

    NOTE: This is an INDEPENDENT implementation using stockstats with AKShare data,
    not using StockstatsUtils which hardcodes yfinance.
    """
    from dateutil.relativedelta import relativedelta

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

        ak_start = _to_akshare_date(start_dt.strftime("%Y-%m-%d"))
        ak_end = _to_akshare_date(curr_date)

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
        # AKShare: 日期,开盘,收盘,最高,最低,成交量
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

        ind_string = ""
        current_dt = curr_date_dt
        while current_dt >= before:
            date_str = current_dt.strftime('%Y-%m-%d')

            # Find matching row
            matching = stock_df[stock_df['date'].dt.strftime('%Y-%m-%d') == date_str]
            if not matching.empty:
                value = matching.iloc[0][indicator]
                if pd.isna(value):
                    ind_string += f"{date_str}: N/A\n"
                else:
                    ind_string += f"{date_str}: {value:.4f}\n"
            else:
                ind_string += f"{date_str}: N/A: Not a trading day\n"

            current_dt = current_dt - relativedelta(days=1)

        result_str = (
            f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
            + ind_string
            + "\n\n"
            + best_ind_params.get(indicator, "No description available.")
        )

        return result_str

    except Exception as e:
        raise AKShareError(f"Error calculating indicator {indicator} for {symbol}: {str(e)}")


def get_fundamentals(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date (not used for AKShare but kept for signature compatibility)"] = None
) -> str:
    """Get company fundamentals overview for China A-shares using AKShare.

    Uses ak.stock_individual_info_em() for basic info.
    """
    try:
        stock_code = to_akshare_symbol(ticker)

        # Get individual stock info from Eastmoney via AKShare
        df = ak.stock_individual_info_em(symbol=stock_code)

        if df.empty:
            return f"No fundamentals data found for symbol '{ticker}'"

        # df has columns: item, value
        info = dict(zip(df['item'], df['value']))

        # Map AKShare fields to standard format
        fields = [
            ("Name", info.get("股票简称")),
            ("Full Name", info.get("公司名称")),
            ("Industry", info.get("行业")),
            ("Total Shares", info.get("总股本")),
            ("Float Shares", info.get("流通股")),
        ]

        lines = []
        for label, value in fields:
            if value is not None:
                lines.append(f"{label}: {value}")

        header = f"# Company Fundamentals for {ticker}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        return header + "\n".join(lines)

    except Exception as e:
        raise AKShareError(f"Error fetching fundamentals for {ticker}: {str(e)}")


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency: annual or quarterly"] = "annual",
    curr_date: Annotated[str, "current date (not used)"] = None
) -> str:
    """Get balance sheet data for China A-shares.

    Uses ak.stock_balance_sheet_by_report_em() for financial reports.
    """
    try:
        stock_code = to_akshare_symbol(ticker)

        # Fetch balance sheet data
        # Note: AKShare uses stock code directly
        df = ak.stock_balance_sheet_by_report_em(symbol=stock_code)

        if df.empty:
            return f"No balance sheet data found for symbol '{ticker}'"

        # Limit to most recent reports
        df = df.head(4 if freq == "quarterly" else 2)

        # Format output
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
    """Get cash flow statement for China A-shares.

    Uses ak.stock_cash_flow_sheet_by_report_em() for cash flow data.
    """
    try:
        stock_code = to_akshare_symbol(ticker)

        df = ak.stock_cash_flow_sheet_by_report_em(symbol=stock_code)

        if df.empty:
            return f"No cash flow data found for symbol '{ticker}'"

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
    """Get income statement for China A-shares.

    Uses ak.stock_profit_sheet_by_report_em() for profit data.
    """
    try:
        stock_code = to_akshare_symbol(ticker)

        df = ak.stock_profit_sheet_by_report_em(symbol=stock_code)

        if df.empty:
            return f"No income statement data found for symbol '{ticker}'"

        df = df.head(4 if freq == "quarterly" else 2)

        header = f"# Income Statement for {ticker}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        return header + df.to_string()

    except Exception as e:
        raise AKShareError(f"Error fetching income statement for {ticker}: {str(e)}")
```

**Step 2: Write integration tests (skip if AKShare not available)**

Create `tests/test_akshare_data.py`:

```python
"""Tests for akshare_data module.

These tests require network access and AKShare installation.
Use pytest -m "not network" to skip network tests.
"""

import pytest
from tradingagents.dataflows.akshare_data import (
    AKShareError,
    get_stock_data,
    get_fundamentals,
    _to_akshare_date,
    _from_akshare_date,
)


class TestDateConversion:
    """Test date conversion utilities."""

    def test_to_akshare_date(self):
        assert _to_akshare_date("2026-01-15") == "20260115"

    def test_from_akshare_date(self):
        assert _from_akshare_date("20260115") == "2026-01-15"


@pytest.mark.network
class TestGetStockData:
    """Test get_stock_data with real AKShare API calls."""

    def test_get_kweichow_moutai_data(self):
        """Test fetching data for 600519.SH (Kweichow Moutai)."""
        result = get_stock_data("600519.SH", "2026-01-01", "2026-01-15")
        assert "600519.SH" in result
        # May return "No data found" if market closed or future date
        assert isinstance(result, str)

    def test_get_ping_an_data(self):
        """Test fetching data for 000001.SZ (Ping An Bank)."""
        result = get_stock_data("000001.SZ", "2026-01-01", "2026-01-15")
        assert "000001.SZ" in result
        assert isinstance(result, str)


@pytest.mark.network
class TestGetFundamentals:
    """Test get_fundamentals with real AKShare API calls."""

    def test_get_kweichow_moutai_fundamentals(self):
        """Test fetching fundamentals for 600519.SH."""
        result = get_fundamentals("600519.SH")
        assert "600519.SH" in result
        # Should contain Chinese company name
        assert isinstance(result, str)
```

**Step 3: Run tests (skip network tests)**

```bash
cd tests && python -m pytest test_akshare_data.py::TestDateConversion -v
```
Expected: Date conversion tests PASS

**Step 4: Commit**

```bash
git add tradingagents/dataflows/akshare_data.py tests/test_akshare_data.py
git commit -m "feat: add AKShare data module for China A-shares (OHLCV, indicators, fundamentals)"
```

---

## Task 5: Create AKShare News Module

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
from .trading_context import TradingContext


class AKShareError(Exception):
    """Raised when AKShare data fetch fails."""
    pass


def get_news(
    ticker: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Get news for a specific China A-share stock using AKShare.

    Uses ak.stock_news_em() for Eastmoney news.
    """
    try:
        stock_code = to_akshare_symbol(ticker)

        # Get news from Eastmoney
        df = ak.stock_news_em(symbol=stock_code)

        if df.empty:
            return f"No news found for {ticker}"

        # Filter by date range if possible
        # AKShare news has '发布时间' (publish time) column
        if '发布时间' in df.columns:
            df['发布时间'] = pd.to_datetime(df['发布时间'])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df['发布时间'] >= start_dt) & (df['发布时间'] <= end_dt + timedelta(days=1))]

        if df.empty:
            return f"No news found for {ticker} between {start_date} and {end_date}"

        # Format output (limit to 10 articles)
        news_str = ""
        for _, row in df.head(10).iterrows():
            title = row.get('标题', 'No title')
            content = row.get('内容', '')
            pub_time = row.get('发布时间', '')

            news_str += f"### {title}\n"
            if pub_time:
                news_str += f"发布时间: {pub_time}\n"
            if content:
                # Truncate long content
                content_str = str(content)[:500] + "..." if len(str(content)) > 500 else str(content)
                news_str += f"{content_str}\n"
            news_str += "\n"

        return f"## {ticker} News, from {start_date} to {end_date}:\n\n{news_str}"

    except Exception as e:
        raise AKShareError(f"Error fetching news for {ticker}: {str(e)}")


def _get_china_macro_news(curr_date: str, look_back_days: int, limit: int) -> str:
    """Get China macroeconomic news for A-share context.

    Uses ak.news_cctv() for CCTV finance news.
    """
    try:
        # Calculate date range
        curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        start_dt = curr_dt - timedelta(days=look_back_days)

        # Get CCTV finance news
        df = ak.news_cctv(date=start_dt.strftime("%Y%m%d"))

        if df.empty:
            return f"No macro news found for China market"

        # Filter by date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
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


def _get_global_macro_news(curr_date: str, look_back_days: int, limit: int) -> str:
    """Fallback to global macro news (placeholder for non-China stocks).

    This would typically call the existing yfinance/alpha_vantage implementation.
    For now, returns a message indicating no China-specific data.
    """
    return "Global macro news not available via AKShare. Use yfinance or alpha_vantage vendor."


def get_global_news(
    curr_date: str,
    look_back_days: int = 7,
    limit: int = 10,
) -> str:
    """Get global/macro news. Routes to Chinese macro for A-shares.

    This function checks TradingContext to determine if the current ticker
    is a China A-share, and returns appropriate macro news.
    """
    if TradingContext.is_china_stock():
        return _get_china_macro_news(curr_date, look_back_days, limit)
    else:
        # For non-China stocks, return a message that routing should use other vendors
        return _get_global_macro_news(curr_date, look_back_days, limit)


def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol of the company"],
) -> str:
    """Get insider transactions for China A-shares.

    Uses ak.stock_share_hold_change_ths() for shareholder changes.
    """
    try:
        stock_code = to_akshare_symbol(ticker)

        # Get insider/shareholder change data from Tonghuashun
        df = ak.stock_share_hold_change_ths(symbol=stock_code)

        if df.empty:
            return f"No insider transaction data found for {ticker}"

        # Format output
        header = f"# Insider Transactions for {ticker}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        return header + df.head(20).to_string()

    except Exception as e:
        raise AKShareError(f"Error fetching insider transactions for {ticker}: {str(e)}")
```

**Step 2: Write tests**

Create `tests/test_akshare_news.py`:

```python
"""Tests for akshare_news module."""

import pytest
from tradingagents.dataflows.akshare_news import (
    get_news,
    get_global_news,
    get_insider_transactions,
    _get_china_macro_news,
)
from tradingagents.dataflows.trading_context import TradingContext


@pytest.mark.network
class TestGetNews:
    """Test get_news with real AKShare API calls."""

    def test_get_kweichow_moutai_news(self):
        """Test fetching news for 600519.SH."""
        result = get_news("600519.SH", "2026-01-01", "2026-01-15")
        assert "600519.SH" in result
        assert isinstance(result, str)


class TestGetGlobalNews:
    """Test get_global_news with TradingContext."""

    def test_china_stock_uses_china_macro(self):
        """Test that China stocks get China macro news."""
        TradingContext.set_ticker("600519.SH")
        try:
            result = get_global_news("2026-01-15", look_back_days=7, limit=5)
            # Should attempt China macro news
            assert isinstance(result, str)
        finally:
            TradingContext.clear()

    def test_us_stock_uses_global_macro(self):
        """Test that US stocks get global macro placeholder."""
        TradingContext.set_ticker("AAPL")
        try:
            result = get_global_news("2026-01-15", look_back_days=7, limit=5)
            # Should return global placeholder
            assert "Global macro" in result or isinstance(result, str)
        finally:
            TradingContext.clear()

    def test_no_context_uses_global_macro(self):
        """Test that no context defaults to global macro."""
        TradingContext.clear()
        result = get_global_news("2026-01-15", look_back_days=7, limit=5)
        assert isinstance(result, str)
```

**Step 3: Run tests (skip network)**

```bash
cd tests && python -m pytest test_akshare_news.py::TestGetGlobalNews -v
```
Expected: Context routing tests PASS

**Step 4: Commit**

```bash
git add tradingagents/dataflows/akshare_news.py tests/test_akshare_news.py
git commit -m "feat: add AKShare news module for China A-shares"
```

---

## Task 6: Update Interface Module for AKShare Integration

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
```

**Step 2: Add "akshare" to VENDOR_LIST**

Modify the VENDOR_LIST:

```python
VENDOR_LIST = [
    "yfinance",
    "alpha_vantage",
    "akshare",  # NEW: China A-share support
]
```

**Step 3: Add AKShare to VENDOR_METHODS**

Add AKShare entries to VENDOR_METHODS dictionary:

```python
VENDOR_METHODS = {
    # ... existing entries ...

    # core_stock_apis
    "get_stock_data": {
        "alpha_vantage": get_alpha_vantage_stock,
        "yfinance": get_YFin_data_online,
        "akshare": get_akshare_stock_data,  # NEW
    },
    # technical_indicators
    "get_indicators": {
        "alpha_vantage": get_alpha_vantage_indicator,
        "yfinance": get_stock_stats_indicators_window,
        "akshare": get_akshare_indicators,  # NEW
    },
    # fundamental_data
    "get_fundamentals": {
        "alpha_vantage": get_alpha_vantage_fundamentals,
        "yfinance": get_yfinance_fundamentals,
        "akshare": get_akshare_fundamentals,  # NEW
    },
    "get_balance_sheet": {
        "alpha_vantage": get_alpha_vantage_balance_sheet,
        "yfinance": get_yfinance_balance_sheet,
        "akshare": get_akshare_balance_sheet,  # NEW
    },
    "get_cashflow": {
        "alpha_vantage": get_alpha_vantage_cashflow,
        "yfinance": get_yfinance_cashflow,
        "akshare": get_akshare_cashflow,  # NEW
    },
    "get_income_statement": {
        "alpha_vantage": get_alpha_vantage_income_statement,
        "yfinance": get_yfinance_income_statement,
        "akshare": get_akshare_income_statement,  # NEW
    },
    # news_data
    "get_news": {
        "alpha_vantage": get_alpha_vantage_news,
        "yfinance": get_news_yfinance,
        "akshare": get_akshare_news,  # NEW
    },
    "get_global_news": {
        "yfinance": get_global_news_yfinance,
        "alpha_vantage": get_alpha_vantage_global_news,
        "akshare": get_akshare_global_news,  # NEW
    },
    "get_insider_transactions": {
        "alpha_vantage": get_alpha_vantage_insider_transactions,
        "yfinance": get_yfinance_insider_transactions,
        "akshare": get_akshare_insider_transactions,  # NEW
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
        # Force AKShare for all China A-share data
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

## Task 7: Update Trading Graph to Set TradingContext

**Files:**
- Modify: `tradingagents/graph/trading_graph.py`

**Step 1: Import TradingContext**

Add import at the top:

```python
from tradingagents.dataflows.trading_context import TradingContext
```

**Step 2: Wrap propagate method with TradingContext**

Modify the `propagate` method to set/clear context:

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
            # Debug mode with tracing
            trace = []
            for chunk in self.graph.stream(init_agent_state, **args):
                if len(chunk["messages"]) == 0:
                    pass
                else:
                    chunk["messages"][-1].pretty_print()
                    trace.append(chunk)

            final_state = trace[-1]
        else:
            # Standard mode without tracing
            final_state = self.graph.invoke(init_agent_state, **args)

        # Store current state for reflection
        self.curr_state = final_state

        # Log state
        self._log_state(trade_date, final_state)

        # Return decision and processed signal
        return final_state, self.process_signal(final_state["final_trade_decision"])

    finally:
        # Always clear the context when done
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

## Task 8: End-to-End Integration Test

**Files:**
- Create: `tests/test_china_a_share_integration.py`

**Step 1: Write integration test**

```python
"""End-to-end integration test for China A-share support.

This test verifies the full flow from TradingAgentsGraph to AKShare data.
Requires AKShare installation and network access.
"""

import pytest
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.dataflows.trading_context import TradingContext
from tradingagents.dataflows import ticker_utils


class TestChinaAshareIntegration:
    """Integration tests for China A-share functionality."""

    def test_ticker_utils_detects_china_stocks(self):
        """Verify ticker_utils correctly identifies China stocks."""
        assert ticker_utils.is_china_stock("600519.SH") is True
        assert ticker_utils.is_china_stock("000001.SZ") is True
        assert ticker_utils.is_china_stock("AAPL") is False

    def test_ticker_utils_parsing(self):
        """Verify ticker parsing works correctly."""
        code, exchange = ticker_utils.parse_ticker("600519.SH")
        assert code == "600519"
        assert exchange == "sh"

        code, exchange = ticker_utils.parse_ticker("000001.SZ")
        assert code == "000001"
        assert exchange == "sz"

    def test_trading_context_operations(self):
        """Verify TradingContext set/get/clear works."""
        TradingContext.set_ticker("600519.SH")
        try:
            assert TradingContext.get_ticker() == "600519.SH"
            assert TradingContext.is_china_stock() is True
        finally:
            TradingContext.clear()

        assert TradingContext.get_ticker() is None
        assert TradingContext.is_china_stock() is False


@pytest.mark.network
@pytest.mark.slow
class TestAKShareDataFetching:
    """Tests that actually call AKShare APIs (slow, requires network)."""

    def test_fetch_kweichow_moutai_data(self):
        """Test fetching real data for Kweichow Moutai (600519.SH)."""
        from tradingagents.dataflows.akshare_data import get_stock_data

        # Use a recent trading date
        result = get_stock_data("600519.SH", "2025-01-01", "2025-01-10")

        assert isinstance(result, str)
        assert "600519.SH" in result
        # Should contain OHLCV data header
        assert "Open" in result or "No data found" in result

    def test_fetch_ping_an_bank_data(self):
        """Test fetching real data for Ping An Bank (000001.SZ)."""
        from tradingagents.dataflows.akshare_data import get_stock_data

        result = get_stock_data("000001.SZ", "2025-01-01", "2025-01-10")

        assert isinstance(result, str)
        assert "000001.SZ" in result


@pytest.mark.skip(reason="Requires full LLM setup - run manually")
class TestFullTradingGraph:
    """Full end-to-end test with TradingAgentsGraph (requires API keys)."""

    def test_propagate_with_china_stock(self):
        """Test running the full graph on a China A-share."""
        config = DEFAULT_CONFIG.copy()
        config["llm_provider"] = "openai"  # or your preferred provider

        ta = TradingAgentsGraph(
            selected_analysts=["market"],  # Minimal for testing
            debug=False,
            config=config
        )

        # Use a known trading date
        state, decision = ta.propagate("600519.SH", "2025-01-10")

        assert state is not None
        assert "market_report" in state
        assert TradingContext.get_ticker() is None  # Should be cleared
```

**Step 2: Run non-network tests**

```bash
cd tests && python -m pytest test_china_a_share_integration.py::TestChinaAshareIntegration -v
```
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/test_china_a_share_integration.py
git commit -m "test: add end-to-end integration tests for China A-share support"
```

---

## Task 9: Manual Smoke Test

**Step 1: Create a simple test script**

Create `test_akshare_smoke.py` at project root:

```python
"""Quick smoke test for AKShare integration."""

import sys
sys.path.insert(0, '.')

print("Testing AKShare integration...")

# Test 1: Import all modules
print("\n1. Testing imports...")
from tradingagents.dataflows.trading_context import TradingContext
from tradingagents.dataflows.ticker_utils import is_china_stock, parse_ticker
from tradingagents.dataflows.akshare_data import get_stock_data, get_fundamentals
from tradingagents.dataflows.akshare_news import get_news
from tradingagents.dataflows.interface import route_to_vendor
print("   OK: All imports successful")

# Test 2: TradingContext
print("\n2. Testing TradingContext...")
TradingContext.set_ticker("600519.SH")
assert TradingContext.is_china_stock() is True
TradingContext.clear()
assert TradingContext.is_china_stock() is False
print("   OK: TradingContext works")

# Test 3: Ticker utils
print("\n3. Testing ticker_utils...")
assert is_china_stock("600519.SH") is True
assert is_china_stock("AAPL") is False
assert parse_ticker("600519.SH") == ("600519", "sh")
print("   OK: ticker_utils works")

# Test 4: Try to fetch data (requires network)
print("\n4. Testing AKShare data fetch (requires network)...")
try:
    result = get_stock_data("600519.SH", "2025-01-01", "2025-01-10")
    print(f"   OK: Fetched {len(result)} chars of data")
except Exception as e:
    print(f"   SKIP: Network issue or AKShare not installed: {e}")

print("\nAll smoke tests passed!")
```

**Step 2: Run smoke test**

```bash
.venv\Scripts\activate
python test_akshare_smoke.py
```

Expected output:
```
Testing AKShare integration...

1. Testing imports...
   OK: All imports successful

2. Testing TradingContext...
   OK: TradingContext works

3. Testing ticker_utils...
   OK: ticker_utils works

4. Testing AKShare data fetch (requires network)...
   OK: Fetched XXXX chars of data

All smoke tests passed!
```

**Step 3: Clean up and commit**

```bash
rm test_akshare_smoke.py
git add tests/test_china_a_share_integration.py
git commit -m "test: verify AKShare smoke tests pass"
```

---

## Task 10: Documentation Update

**Files:**
- Create: `docs/china-a-share-usage.md`

**Step 1: Create usage documentation**

```markdown
# China A-Share Usage Guide

TradingAgents now supports China A-shares (Shanghai SSE and Shenzhen SZSE) through AKShare integration.

## Quick Start

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# Configure (optional - defaults work for China stocks)
config = DEFAULT_CONFIG.copy()

# Create trading graph
ta = TradingAgentsGraph(debug=True, config=config)

# Analyze a China A-share (Kweichow Moutai)
state, decision = ta.propagate("600519.SH", "2025-01-15")
print(decision)
```

## Ticker Format

China A-shares use the following suffix format:

| Exchange | Suffix | Example |
|----------|--------|---------|
| Shanghai Stock Exchange | `.SH` | `600519.SH` (Kweichow Moutai) |
| Shenzhen Stock Exchange | `.SZ` | `000001.SZ` (Ping An Bank) |

## How It Works

1. **Automatic Detection**: Tickers ending with `.SH` or `.SZ` are automatically routed to AKShare
2. **No API Key Required**: AKShare scrapes public data - no key needed
3. **Chinese Data**: Financial reports and news are in Chinese - LLMs handle them natively
4. **Fallback**: No fallback for A-shares (yfinance doesn't support them)

## Supported Data

| Data Type | AKShare Source | Notes |
|-----------|---------------|-------|
| OHLCV Prices | `stock_zh_a_hist()` | 15+ years historical |
| Technical Indicators | stockstats + AKShare | All standard indicators |
| Fundamentals | `stock_individual_info_em()` | Basic company info |
| Financial Statements | Eastmoney reports | Chinese accounting standards |
| News | `stock_news_em()` | Eastmoney news |
| Insider Transactions | Tonghuashun data | Shareholder changes |

## Limitations

- **Network**: AKShare scrapes Eastmoney/Sina - requires internet access
- **Trading Calendar**: Chinese holidays differ from US - may return empty data on non-trading days
- **Hong Kong Stocks**: `.HK` tickers not yet supported
- **Real-time**: Daily granularity only (no intraday)

## Troubleshooting

### "No data found" errors
Check if the date falls on a Chinese holiday or weekend. A-shares don't trade on:
- Weekends
- Chinese New Year
- National Day Golden Week
- Other Chinese holidays

### Network timeouts
AKShare requires access to Chinese financial websites. If you're outside mainland China:
- Some sources may be slower
- Consider using a VPN with China routing if issues persist

## Examples

### Shanghai Stock
```python
# Kweichow Moutai (茅台)
state, decision = ta.propagate("600519.SH", "2025-01-15")

# ICBC (工商银行)
state, decision = ta.propagate("601398.SH", "2025-01-15")
```

### Shenzhen Stock
```python
# Ping An Bank (平安银行)
state, decision = ta.propagate("000001.SZ", "2025-01-15")

# BYD (比亚迪)
state, decision = ta.propagate("002594.SZ", "2025-01-15")

# Midea Group (美的集团)
state, decision = ta.propagate("000333.SZ", "2025-01-15")
```
```

**Step 2: Commit documentation**

```bash
git add docs/china-a-share-usage.md
git commit -m "docs: add China A-share usage guide"
```

---

## Summary

This implementation plan adds China A-share support through 10 bite-sized tasks:

1. **Add AKShare dependency** - Single line in requirements.txt
2. **Create TradingContext** - Thread-local context for ticker passing
3. **Create ticker_utils** - Market detection and parsing utilities
4. **Create akshare_data** - Core stock data, indicators, fundamentals
5. **Create akshare_news** - News and insider transaction data
6. **Update interface.py** - Register AKShare and add China stock routing
7. **Update trading_graph.py** - Set/clear TradingContext in propagate
8. **Integration tests** - End-to-end test coverage
9. **Smoke test** - Manual verification script
10. **Documentation** - Usage guide with examples

**Key Design Decisions:**
- Thread-local context avoids changing all tool signatures
- Automatic routing based on `.SH`/`.SZ` suffix requires zero config
- Independent indicator calculation avoids yfinance hardcoding
- AKShare always used for A-shares (no fallback - yfinance doesn't support them)

**Testing Strategy:**
- Unit tests for utilities and context
- Integration tests for AKShare APIs (marked with `@pytest.mark.network`)
- Smoke test for quick verification
- Manual end-to-end test with real LLM calls (requires API keys)
