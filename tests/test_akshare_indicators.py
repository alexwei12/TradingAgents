"""Tests for akshare_indicators module."""

import pytest
from tradingagents.dataflows.akshare_indicators import get_indicators


class TestIndicators:
    """Test cases for technical indicators."""

    def test_unsupported_indicator_raises_error(self):
        """Test that unsupported indicator raises ValueError."""
        with pytest.raises(ValueError):
            get_indicators("688981.SH", "unsupported_indicator", "2024-12-01")


@pytest.mark.network
class TestIndicatorsNetwork:
    """Network tests for indicators with real data."""

    def test_get_rsi_indicator(self):
        """Test fetching RSI indicator for 中芯国际 (688981.SH)."""
        result = get_indicators("688981.SH", "rsi", "2024-12-01", look_back_days=5)
        assert "rsi" in result.lower()
        assert "RSI" in result
        assert isinstance(result, str)

    def test_get_macd_indicator(self):
        """Test fetching MACD indicator."""
        result = get_indicators("688981.SH", "macd", "2024-12-01", look_back_days=5)
        assert "MACD" in result
        assert isinstance(result, str)
