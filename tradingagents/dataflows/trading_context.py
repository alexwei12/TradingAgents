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
