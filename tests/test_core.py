from datetime import date

import pandas as pd
import pytest

from app import _summarize_model_rows
from src.document_parser import extract_html_text, normalize_text
from src.hkex_client import normalize_stock_code, parse_announcement_rows, parse_search_results
from src.market_data import calculate_event_return, include_next_trading_days, yahoo_ticker
from src.sentiment import normalize_sentiment_label, recommendation_for_label


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


def test_price_history_includes_only_next_three_trading_days():
    prices = pd.DataFrame(
        {"Close": [100, 101, 102, 103, 104, 105]},
        index=pd.to_datetime(
            ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]
        ),
    )
    result = include_next_trading_days(prices, date(2026, 1, 6), days=3)
    assert list(result.index) == list(prices.index)

    result = include_next_trading_days(prices, date(2026, 1, 5), days=3)
    assert list(result.index) == list(prices.index[:5])


def test_sentiment_recommendation_mapping():
    assert recommendation_for_label("positive") == "Buy"
    assert recommendation_for_label("negative") == "Sell"
    assert recommendation_for_label("neutral") == "Hold"


def test_model_labels_are_normalized():
    assert normalize_sentiment_label("Sigma/financial-sentiment-analysis", "LABEL_0") == "negative"
    assert normalize_sentiment_label("Sigma/financial-sentiment-analysis", "LABEL_1") == "neutral"
    assert normalize_sentiment_label("Sigma/financial-sentiment-analysis", "LABEL_2") == "positive"
    assert normalize_sentiment_label("ProsusAI/finbert", "LABEL_0") == "positive"
    assert normalize_sentiment_label("NOSIBLE/financial-sentiment-v1.1-base", "LABEL_0") == "negative"


def test_model_summary_uses_majority_prediction_and_average_confidence():
    rows = [
        ["2026-01-01", "A", "ProsusAI/finbert", "positive", "Buy", "80.0%", "summary"],
        ["2026-01-02", "B", "ProsusAI/finbert", "negative", "Sell", "60.0%", "summary"],
        ["2026-01-03", "C", "ProsusAI/finbert", "positive", "Buy", "90.0%", "summary"],
    ]
    summary = _summarize_model_rows(rows)
    prosus = summary[0]
    assert prosus[1:4] == [2, 0, 1]
    assert prosus[4:6] == ["76.7%", "Buy"]
    assert prosus[6] == 3
