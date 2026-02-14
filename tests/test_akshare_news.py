"""Tests for akshare_news module."""

import pytest
from tradingagents.dataflows.akshare_news import get_news, get_global_news


@pytest.mark.network
class TestGetNews:
    """Test get_news with real AKShare API calls."""

    def test_get_kweichow_moutai_news(self):
        """Test fetching news for 688981.SH."""
        result = get_news("688981.SH", "2024-12-01", "2024-12-10")
        assert "688981.SH" in result
        assert isinstance(result, str)


@pytest.mark.network
class TestGetGlobalNews:
    """Test get_global_news with real data."""

    def test_get_china_macro_news(self):
        """Test fetching China macro news."""
        result = get_global_news("2024-12-01", look_back_days=3, limit=5)
        assert isinstance(result, str)
        assert "China Macro News" in result or "No macro news" in result
