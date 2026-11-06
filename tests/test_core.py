from datetime import date

import pandas as pd
import pytest

from src.document_parser import extract_html_text, normalize_text
from src.hkex_client import normalize_stock_code, parse_announcement_rows, parse_search_results
from src.market_data import calculate_event_return, yahoo_ticker


def test_normalize_stock_code():
    assert normalize_stock_code("5.HK") == "000005"
    with pytest.raises(ValueError):
        normalize_stock_code("ABC")


def test_parse_announcement_rows_skips_invalid_rows():
    rows = parse_announcement_rows([
        {"title": "Annual Results", "date": "2026-01-03", "url": "/doc.pdf", "type": "pdf"},
        {"title": "Missing date", "url": "/bad"},
    ])
    assert len(rows) == 1
    assert rows[0].url.endswith("/doc.pdf")
    assert rows[0].document_type == "pdf"


def test_parse_hkex_search_results():
    announcements = parse_search_results(
        {
            "result": '[{"TITLE":"Annual Results","DATE_TIME":"06/11/2026 17:15",'
            '"FILE_TYPE":"PDF","FILE_LINK":"/listedco/results.pdf"}]'
        }
    )
    assert announcements[0].title == "Annual Results"
    assert announcements[0].published_at == date(2026, 11, 6)
    assert announcements[0].url.endswith("/listedco/results.pdf")


def test_extract_html_text_removes_scripts():
    assert extract_html_text("<script>bad()</script><main> Profit   increased </main>") == "Profit increased"
    assert normalize_text("a   b") == "a b"


def test_event_return_uses_trading_days():
    prices = pd.DataFrame({"Close": [100.0, 110.0]}, index=pd.to_datetime(["2026-01-02", "2026-01-07"]))
    result = calculate_event_return(prices, date(2026, 1, 2), days_after=5)
    assert result.return_percent == pytest.approx(10.0)
    assert result.status == "OK"


def test_market_ticker():
    assert yahoo_ticker("5") == "0005.HK"
