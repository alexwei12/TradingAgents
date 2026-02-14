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
