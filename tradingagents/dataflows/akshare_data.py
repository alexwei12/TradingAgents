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
