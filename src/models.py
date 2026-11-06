"""Shared data structures used by the stock-analysis application."""

from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass(frozen=True)
class Announcement:
    title: str
    published_at: date
    url: str
    document_type: str = "html"
    text: Optional[str] = None


@dataclass(frozen=True)
class SentimentResult:
    label: str
    confidence: float
    chunks: int


@dataclass(frozen=True)
class EventReturn:
    announcement_date: date
    window_start: date
    window_end: date
    start_price: Optional[float]
    end_price: Optional[float]
    return_percent: Optional[float]
    status: str
