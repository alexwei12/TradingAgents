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
