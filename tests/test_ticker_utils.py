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
