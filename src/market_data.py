"""Historical prices and simple announcement event-window returns."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from .models import EventReturn


def yahoo_ticker(stock_code: str) -> str:
    from .hkex_client import normalize_stock_code

    # HKEX displays five-digit codes (00001), while the internal validator
    # uses six digits to normalize input (000001). Yahoo uses four digits:
    # HKEX 00005 is Yahoo 0005.HK and HKEX 00001 is Yahoo 0001.HK.
    return f"{normalize_stock_code(stock_code)[-4:]}.HK"


def _normalize_price_columns(prices: Any) -> Any:
    """Convert yfinance's one-ticker MultiIndex columns to plain strings."""

    if prices is None or not hasattr(prices, "columns"):
        return prices
    if getattr(prices.columns, "nlevels", 1) > 1:
        prices = prices.copy()
        prices.columns = [column[0] if isinstance(column, tuple) else column for column in prices.columns]
    return prices


def build_candlestick_chart(prices: Any) -> Any:
    """Build a Plotly K-line/candlestick chart from Yahoo price data."""

    if prices is None or len(prices) == 0:
        return None
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError("Install plotly to display the candlestick chart.") from exc

    frame = _normalize_price_columns(prices.copy())
    frame = frame.reset_index()
    date_column = frame.columns[0]
    figure = go.Figure(
        data=[
            go.Candlestick(
                x=frame[date_column],
                open=frame["Open"],
                high=frame["High"],
                low=frame["Low"],
                close=frame["Close"],
                name="Price",
            )
        ]
    )
    figure.update_layout(
        title="K-line price history",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=550,
    )
    return figure


def calculate_event_return(prices: Any, event_date: date, days_after: int = 5) -> EventReturn:
    """Calculate a close-to-close return around an event from a price table."""

    start = event_date
    end = event_date + timedelta(days=days_after)
    if prices is None or len(prices) == 0:
        return EventReturn(event_date, start, end, None, None, None, "No price data available")

    frame = _normalize_price_columns(prices.copy())
    frame.index = [value.date() if hasattr(value, "date") else value for value in frame.index]
    available = frame.loc[(frame.index >= start) & (frame.index <= end)]
    if available.empty or "Close" not in available:
        return EventReturn(event_date, start, end, None, None, None, "No trading days in event window")
    first = float(available["Close"].iloc[0])
    last = float(available["Close"].iloc[-1])
    if first == 0:
        return EventReturn(event_date, start, end, first, last, None, "Invalid zero starting price")
    return EventReturn(event_date, start, end, first, last, (last / first - 1) * 100, "OK")


def fetch_prices(stock_code: str, start: date, end: date) -> Any:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install yfinance to retrieve historical prices.") from exc
    try:
        prices = yf.download(
            yahoo_ticker(stock_code),
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        return _normalize_price_columns(prices)
    except Exception as exc:
        raise RuntimeError(f"Yahoo Finance could not provide prices: {exc}") from exc
