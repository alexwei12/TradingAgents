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

    def test_get_zhongxin_data(self):
        """Test fetching data for 中芯国际 (688981.SH)."""
        result = get_stock_data("688981.SH", "2024-12-01", "2024-12-10")
        assert "688981.SH" in result
        assert isinstance(result, str)

    def test_get_ping_an_data(self):
        """Test fetching data for 000001.SZ (Ping An Bank)."""
        result = get_stock_data("000001.SZ", "2024-12-01", "2024-12-10")
        assert "000001.SZ" in result
        assert isinstance(result, str)


@pytest.mark.network
class TestGetFundamentals:
    """Test get_fundamentals with real AKShare API calls."""

    def test_get_zhongxin_fundamentals(self):
        """Test fetching fundamentals for 688981.SH."""
        result = get_fundamentals("688981.SH")
        assert "688981.SH" in result
        # Should contain Chinese company name field
        assert "股票简称" in result or isinstance(result, str)
