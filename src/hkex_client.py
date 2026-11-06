"""HKEX announcement retrieval and parsing.

The HKEX website is an external dependency and may change its form fields or
anti-bot behaviour. Keeping this code isolated makes that change manageable.
"""

from __future__ import annotations

from datetime import date, datetime
import json
import re
from typing import Any, Iterable
from urllib.parse import urljoin

from .models import Announcement

HKEX_SEARCH_URL = "https://www1.hkexnews.hk/search/titlesearch.xhtml?lang=en"
HKEX_BASE_URL = "https://www1.hkexnews.hk"
HKEX_STOCK_LOOKUP_URL = f"{HKEX_BASE_URL}/search/partial.do"
HKEX_RESULTS_URL = f"{HKEX_BASE_URL}/search/titleSearchServlet.do"
_CODE_RE = re.compile(r"^\d{1,6}$")


def normalize_stock_code(value: str) -> str:
    """Return a six-digit HKEX code or raise ``ValueError``."""

    code = str(value).strip().replace(".HK", "").replace(".hk", "")
    if not _CODE_RE.fullmatch(code):
        raise ValueError("Enter an HKEX stock code containing 1 to 6 digits.")
    return code.zfill(6)


def _parse_date(value: str) -> date | None:
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%Y %H:%M", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    match = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", value)
    if match:
        return date(*(int(part) for part in match.groups()))
    return None


def parse_announcement_rows(rows: Iterable[dict[str, Any]], base_url: str = HKEX_SEARCH_URL) -> list[Announcement]:
    """Convert normalized HKEX row dictionaries into announcements.

    This parser is deliberately independent of ``requests`` and HTML details,
    which makes it straightforward to test against saved HKEX responses.
    """

    announcements: list[Announcement] = []
    for row in rows:
        title = str(row.get("title", "")).strip()
        published = _parse_date(str(row.get("date", "")))
        raw_url = str(row.get("url", "")).strip()
        url = urljoin(base_url, raw_url)
        if title and published and raw_url:
            announcements.append(
                Announcement(
                    title=title,
                    published_at=published,
                    url=url,
                    document_type=str(row.get("type") or ("pdf" if raw_url.lower().split("?", 1)[0].endswith(".pdf") else "html")).lower(),
                )
            )
    return announcements


def parse_search_results(payload: dict[str, Any], limit: int | None = None) -> list[Announcement]:
    """Parse the JSON object returned by ``titleSearchServlet.do``."""

    raw_result = payload.get("result") or "[]"
    try:
        result = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
    except json.JSONDecodeError as exc:
        raise RuntimeError("HKEX returned an invalid announcement response.") from exc

    rows = [
        {
            "title": item.get("TITLE") or item.get("SHORT_TEXT", ""),
            "date": item.get("DATE_TIME", ""),
            "url": item.get("FILE_LINK") or item.get("DOD_WEB_PATH", ""),
            "type": item.get("FILE_TYPE", "pdf"),
        }
        for item in (result or [])
    ]
    return parse_announcement_rows(rows if limit is None else rows[:limit], base_url=HKEX_BASE_URL)


def _parse_jsonp(response_text: str) -> dict[str, Any]:
    """Parse HKEX's callback-wrapped stock lookup response."""

    start = response_text.find("{")
    end = response_text.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError("HKEX returned an invalid stock lookup response.")
    try:
        return json.loads(response_text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise RuntimeError("HKEX returned an invalid stock lookup response.") from exc


def fetch_announcements(
    stock_code: str,
    limit: int | None = None,
    timeout: int = 15,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[Announcement]:
    """Fetch recent HKEX announcements for a stock code.

    HKEX can change its search form, so failures are surfaced as actionable
    ``RuntimeError`` messages instead of returning fabricated results.
    """

    code = normalize_stock_code(stock_code)
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Install the application requirements before using live HKEX retrieval.") from exc

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; COMP5513-StockAnalysis/1.0; educational demo)",
        "Referer": HKEX_SEARCH_URL,
    }
    session = requests.Session()
    try:
        session.get(HKEX_SEARCH_URL, headers=headers, timeout=timeout).raise_for_status()
        lookup = session.get(
            HKEX_STOCK_LOOKUP_URL,
            params={"lang": "EN", "type": "A", "name": code, "market": "SEHK", "callback": "callback"},
            headers=headers,
            timeout=timeout,
        )
        lookup.raise_for_status()
        stock_info = _parse_jsonp(lookup.text).get("stockInfo", [])
        stock = next((item for item in stock_info if str(item.get("code", "")).zfill(6) == code), None)
        if stock is None:
            raise RuntimeError(f"HKEX could not find stock code {code}.")

        today = date.today()
        start_date = start_date or today.replace(day=1)
        end_date = end_date or today
        if start_date > end_date:
            raise ValueError("The start date must be on or before the end date.")
        row_range = min(max(limit, 1), 100) if limit is not None else 100
        params = {
            "sortDir": "0",
            "sortByOptions": "DateTime",
            "category": "0",
            "market": "SEHK",
            "stockId": str(stock["stockId"]),
            "documentType": "",
            "fromDate": start_date.strftime("%Y%m%d"),
            "toDate": end_date.strftime("%Y%m%d"),
            "title": "",
            "searchType": "0",
            "t1code": "",
            "t2Gcode": "",
            "t2code": "",
            "rowRange": str(row_range),
            "lang": "E",
        }
        announcements: list[Announcement] = []
        while True:
            response = session.get(HKEX_RESULTS_URL, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            page = parse_search_results(payload)
            announcements = page
            if limit is not None or not payload.get("hasNextRow") or not page:
                break
            params["rowRange"] = str(int(params["rowRange"]) + 100)
    except requests.RequestException as exc:
        raise RuntimeError(f"HKEX could not be reached: {exc}") from exc

    if limit is not None:
        announcements = announcements[:limit]
    if not announcements:
        raise RuntimeError(f"HKEX found no announcements for {code} between {start_date} and {end_date}.")
    return announcements[:limit]
