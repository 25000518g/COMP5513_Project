# COMP5513 HKEX Stock Analysis

An educational Python/Gradio application for exploring Hong Kong-listed stock announcements, local financial sentiment, and historical price movement.

## Features

- Enter a one-to-six digit HKEX stock code, such as `00005`.
- Choose a start date and end date to limit HKEX announcements and price history.
- Retrieve announcements and notices from HKEX.
- Save downloaded documents under `downloads/<stock-code>/` before analysis.
- Extract text from HTML or text-based PDF announcements.
- Classify announcement sentiment locally with FinBERT (`ProsusAI/finbert`).
- Fetch historical prices through Yahoo Finance.
- Display the selected price range as a K-line/candlestick chart.
- Download all matching announcements and notices silently into `downloads/<stock-code>/`.
- Generate a local AI executive summary for each downloaded announcement.
- Calculate simple returns over the announcement date and following five calendar days.
- Display sentiment, confidence, executive summary, and event return together in one results table.

## Setup

Python 3.10 or newer is recommended.

```text
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

The first sentiment analysis downloads FinBERT and the first executive summary downloads the summarization model; these first requests may take longer than later requests. The app also needs network access to HKEX and Yahoo Finance.

Each analysis creates a folder such as `downloads/00001/` and stores the retrieved PDF files there. The sentiment step then reads those saved local files. Downloaded files are excluded from Git by `.gitignore`.

Use dates in `YYYY-MM-DD` format. The selected end date is included in the HKEX request and price-history range.

## Project structure

- `app.py` — Gradio user interface and workflow orchestration
- `src/hkex_client.py` — HKEX retrieval and stock-code validation
- `src/document_parser.py` — HTML/PDF extraction
- `src/sentiment.py` — cached local sentiment model
- `src/summarizer.py` — cached local AI executive-summary model
- `src/market_data.py` — price retrieval and event returns
- `src/models.py` — shared data classes
- `tests/` — deterministic unit tests

## Testing

Run the unit tests:

```text
pytest -q
```

## Limitations and ethics

HKEX may change its website structure or restrict automated requests. Some announcements are scanned PDFs and require OCR, which this demo intentionally does not implement. Yahoo Finance may have missing prices or incomplete ticker coverage; HKEX five-digit codes are converted to Yahoo's four-digit format where required. Sentiment is a model prediction, not a verified fact, and an event-window return does not demonstrate that an announcement caused a price change.

This project is for coursework and education only. It is not financial advice, does not recommend buying or selling securities, and should not be used as the sole basis for an investment decision.
