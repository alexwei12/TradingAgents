"""Common utilities and exceptions for AKShare modules."""


class AKShareError(Exception):
    """Raised when AKShare data fetch fails."""
    pass


def to_akshare_date(date_str: str) -> str:
    """Convert YYYY-MM-DD to YYYYMMDD for AKShare."""
    return date_str.replace("-", "")


def from_akshare_date(date_str: str) -> str:
    """Convert YYYYMMDD to YYYY-MM-DD."""
    if len(date_str) == 8:
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return date_str
