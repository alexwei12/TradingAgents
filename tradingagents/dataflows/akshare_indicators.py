"""AKShare-based technical indicators for China A-shares.

This is an independent implementation using stockstats with AKShare OHLCV data,
not using StockstatsUtils which hardcodes yfinance.
"""

from typing import Annotated
from datetime import datetime
import pandas as pd
import akshare as ak
from stockstats import wrap
from dateutil.relativedelta import relativedelta

from .ticker_utils import parse_ticker, to_akshare_symbol
from .akshare_common import AKShareError, to_akshare_date


def get_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    """Calculate technical indicators using AKShare OHLCV data."""

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

        ak_start = to_akshare_date(start_dt.strftime("%Y-%m-%d"))
        ak_end = to_akshare_date(curr_date)

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

        # Rename Chinese column names to English to avoid stockstats issues
        df_en = df.rename(columns={
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
        })
        df_en['date'] = pd.to_datetime(df_en['date'])

        # Use stockstats to calculate indicators
        stock_df = wrap(df_en)

        # Filter to lookback period
        end_date = curr_date
        before = curr_date_dt - relativedelta(days=look_back_days)

        # Extract the lookback period data using index-based slicing
        # Reset index to make date a column
        stock_df_reset = stock_df.reset_index()

        # Find the start index for the lookback period
        start_idx = 0
        for idx, dt in enumerate(stock_df_reset['date']):
            if dt >= before:
                start_idx = idx
                break

        # Extract date and value columns using iloc
        dates = stock_df_reset.iloc[start_idx:start_idx+look_back_days+1, 0]
        values = stock_df_reset.iloc[start_idx:start_idx+look_back_days+1, -1]

        # Build indicator string by iterating through dates
        ind_string = ""
        for dt, value in zip(dates, values):
            if pd.isna(value):
                ind_string += f"{dt.strftime('%Y-%m-%d')}: N/A\n"
            else:
                ind_string += f"{dt.strftime('%Y-%m-%d')}: {value:.4f}\n"

        result_str = (
            f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
            + ind_string
            + "\n\n"
            + best_ind_params.get(indicator, "No description available.")
        )

        return result_str

    except Exception as e:
        raise AKShareError(f"Error calculating indicator {indicator} for {symbol}: {str(e)}")