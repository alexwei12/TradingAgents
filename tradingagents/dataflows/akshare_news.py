"""AKShare-based news and sentiment data for China A-shares."""

from typing import Annotated
from datetime import datetime, timedelta
import akshare as ak
import pandas as pd

from .ticker_utils import to_akshare_symbol
from .akshare_common import AKShareError


def get_news(
    ticker: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Get news for a specific China A-share stock using AKShare."""
    try:
        stock_code = to_akshare_symbol(ticker)
        df = ak.stock_news_em(symbol=stock_code)

        if df.empty:
            return f"No news found for {ticker}"

        # Filter by date range
        if '发布时间' in df.columns:
            df['发布时间'] = pd.to_datetime(df['发布时间'])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df['发布时间'] >= start_dt) & (df['发布时间'] <= end_dt + timedelta(days=1))]

        if df.empty:
            return f"No news found for {ticker} between {start_date} and {end_date}"

        news_str = ""
        for _, row in df.head(10).iterrows():
            title = row.get('标题', 'No title')
            content = row.get('内容', '')
            pub_time = row.get('发布时间', '')

            news_str += f"### {title}\n"
            if pub_time:
                news_str += f"发布时间: {pub_time}\n"
            if content:
                content_str = str(content)[:500] + "..." if len(str(content)) > 500 else str(content)
                news_str += f"{content_str}\n"
            news_str += "\n"

        return f"## {ticker} News, from {start_date} to {end_date}:\n\n{news_str}"

    except Exception as e:
        raise AKShareError(f"Error fetching news for {ticker}: {str(e)}")


def _get_china_macro_news(curr_date: str, look_back_days: int, limit: int) -> str:
    """Get China macroeconomic news for A-share context.

    Uses ak.news_cctv() for CCTV finance news. Note: this API returns
    data for a single date, so we need to iterate over the lookback period.
    """
    try:
        curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")

        # Iterate over lookback period to collect news from each day
        all_news = []
        for i in range(look_back_days + 1):
            day = curr_dt - timedelta(days=i)
            try:
                df = ak.news_cctv(date=day.strftime("%Y%m%d"))
                if not df.empty:
                    df['date'] = day  # Add date column for filtering
                    all_news.append(df)
            except Exception:
                # Skip days with no data or errors
                continue

        if not all_news:
            return f"No macro news found for China market"

        # Combine all news
        df = pd.concat(all_news, ignore_index=True)

        # Filter by date range
        start_dt = curr_dt - timedelta(days=look_back_days)
        if 'date' in df.columns:
            df = df[(df['date'] >= start_dt) & (df['date'] <= curr_dt)]

        # Format output
        news_str = ""
        for _, row in df.head(limit).iterrows():
            title = row.get('title', 'No title')
            content = row.get('content', '')
            pub_date = row.get('date', '')

            news_str += f"### {title}\n"
            if pub_date:
                news_str += f"日期: {pub_date}\n"
            if content:
                content_str = str(content)[:500] + "..." if len(str(content)) > 500 else str(content)
                news_str += f"{content_str}\n"
            news_str += "\n"

        start_date = start_dt.strftime("%Y-%m-%d")
        return f"## China Macro News, from {start_date} to {curr_date}:\n\n{news_str}"

    except Exception as e:
        return f"China macro news temporarily unavailable: {str(e)}"


def get_global_news(
    curr_date: str,
    look_back_days: int = 7,
    limit: int = 10,
) -> str:
    """Get global/macro news for China A-shares.

    NOTE: This function is called via AKShare vendor routing for China stocks.
    It directly returns China macro news without checking TradingContext,
    as the routing decision has already been made.
    """
    # Directly return China macro news (routing already determined this is China stock)
    return _get_china_macro_news(curr_date, look_back_days, limit)
