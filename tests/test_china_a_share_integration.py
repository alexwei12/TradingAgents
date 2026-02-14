"""End-to-end integration test for China A-share support."""

import pytest
from tradingagents.dataflows.trading_context import TradingContext
from tradingagents.dataflows import ticker_utils


class TestChinaAshareIntegration:
    """Integration tests for China A-share functionality."""

    def test_ticker_utils_detects_all_china_exchanges(self):
        """Verify ticker_utils correctly identifies all China exchanges."""
        assert ticker_utils.is_china_stock("688981.SH") is True  # Shanghai
        assert ticker_utils.is_china_stock("000001.SZ") is True  # Shenzhen
        assert ticker_utils.is_china_stock("835305.BJ") is True  # Beijing
        assert ticker_utils.is_china_stock("AAPL") is False

    def test_ticker_utils_parsing(self):
        """Verify ticker parsing works correctly."""
        code, exchange = ticker_utils.parse_ticker("688981.SH")
        assert code == "688981"
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

    def test_fetch_zhongxin_international_data(self):
        """Test fetching real data for 中芯国际 (688981.SH)."""
        from tradingagents.dataflows.akshare_data import get_stock_data

        result = get_stock_data("688981.SH", "2024-12-01", "2024-12-10")
        assert isinstance(result, str)
        assert "688981.SH" in result

    def test_fetch_beijing_stock_data(self):
        """Test fetching data for Beijing exchange stock."""
        from tradingagents.dataflows.akshare_data import get_stock_data

        result = get_stock_data("835305.BJ", "2024-12-01", "2024-12-10")
        assert isinstance(result, str)
        assert "835305.BJ" in result
