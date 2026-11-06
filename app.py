"""Gradio entry point for the HKEX stock-analysis demo."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import logging
from pathlib import Path
from time import perf_counter

from src.document_parser import download_and_extract
from src.hkex_client import fetch_announcements, normalize_stock_code
from src.market_data import build_candlestick_chart, calculate_event_return, fetch_prices
from src.sentiment import analyze_sentiment
from src.summarizer import generate_executive_summary

DISCLAIMER = "This educational analysis is not financial advice. Sentiment and event returns do not prove causation."
DOWNLOAD_ROOT = Path("downloads")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _parse_gui_date(value: object, field_name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError as exc:
        raise ValueError(f"Enter {field_name} as YYYY-MM-DD.") from exc


def analyze_stock(stock_code: str, start_date: object, end_date: object):
    """Run the workflow and stream status, results, and errors to Gradio."""
    started = perf_counter()
    rows = []
    chart = None

    def update(message: str):
        logger.info(message)
        return message, chart, rows

    yield update("⏳ Starting analysis and validating the HKEX stock code...")
    try:
        code = normalize_stock_code(stock_code)
        selected_start = _parse_gui_date(start_date, "the start date")
        selected_end = _parse_gui_date(end_date, "the end date")
        if selected_start > selected_end:
            raise ValueError("The start date must be on or before the end date.")
        logger.info("Starting analysis for HKEX %s", code)
        yield update(f"🔎 Looking up HKEX announcements for `{code}` from {selected_start} to {selected_end}...")
        announcements = fetch_announcements(
            code,
            start_date=selected_start,
            end_date=selected_end,
        )
    except ValueError as exc:
        logger.warning("Invalid stock code: %s", exc)
        yield update(f"❌ Error while retrieving announcements: {exc}")
        return
    except RuntimeError as exc:
        logger.exception("Announcement lookup failed")
        yield update(f"❌ Error while retrieving announcements: {exc}")
        return

    yield update(f"✅ Found {len(announcements)} announcement(s). Preparing analysis...")
    stock_folder = DOWNLOAD_ROOT / code
    try:
        prices = fetch_prices(code, selected_start, selected_end + timedelta(days=1))
        if prices is not None and len(prices) > 0:
            chart = build_candlestick_chart(prices)
            yield update(f"📊 Loaded {len(prices)} trading-day price records for the selected range.")
        else:
            yield update("⚠️ No Yahoo Finance prices were found for the selected range. Continuing with announcements.")
    except RuntimeError as exc:
        logger.exception("Price-history retrieval failed")
        prices = None
        yield update(f"⚠️ Price history unavailable: {exc}. Continuing with announcements.")

    for index, announcement in enumerate(announcements, start=1):
        extension = ".pdf" if announcement.document_type.lower() == "pdf" else ".html"
        saved_path = stock_folder / f"{announcement.published_at.isoformat()}_{index:03d}{extension}"
        yield update(f"🔄 [{index}/{len(announcements)}] Processing announcement notice...")
        sentiment_label = "Unavailable"
        sentiment_confidence = None
        executive_summary = "Unavailable"
        try:
            text = download_and_extract(announcement.url, save_path=saved_path)
            logger.info("Downloaded and extracted %s to %s", announcement.url, saved_path)
            sentiment = analyze_sentiment(text)
            sentiment_label = sentiment.label
            sentiment_confidence = f"{sentiment.confidence:.1%}"
            executive_summary = generate_executive_summary(text)
        except (RuntimeError, ValueError) as exc:
            logger.exception("Document or sentiment analysis failed for %s", announcement.title)
            executive_summary = f"Unavailable: {exc}"

        yield update(f"📈 [{index}/{len(announcements)}] Calculating event-window return...")
        try:
            event = calculate_event_return(prices, announcement.published_at)
            return_percent = event.return_percent
            market_status = event.status
        except (RuntimeError, ValueError) as exc:
            logger.exception("Market-data analysis failed for %s", announcement.title)
            return_percent = None
            market_status = str(exc)
        rows.append(
            [
                announcement.published_at.isoformat(),
                announcement.title,
                sentiment_label,
                sentiment_confidence,
                executive_summary,
                return_percent,
                market_status,
            ]
        )
        yield update(f"✅ [{index}/{len(announcements)}] Finished analysis.")

    elapsed = perf_counter() - started
    status = f"✅ Completed {len(announcements)} announcement(s) for HKEX {code} from {selected_start} to {selected_end} in {elapsed:.1f}s. {DISCLAIMER}"
    yield update(status)


def build_demo():
    try:
        import gradio as gr  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Install the requirements before launching the app.") from exc

    with gr.Blocks(title="HKEX Stock Analysis") as demo:
        gr.Markdown("# HKEX Stock Analysis\nEnter a six-digit HKEX stock code to inspect announcements and market response.")
        with gr.Row():
            stock_code = gr.Textbox(label="HKEX stock code", value="00005", placeholder="For example: 00005")
            start_date = gr.Textbox(label="Start date (YYYY-MM-DD)", value=date.today().replace(day=1).isoformat())
            end_date = gr.Textbox(label="End date (YYYY-MM-DD)", value=date.today().isoformat())
            run = gr.Button("Analyze", variant="primary")
        status = gr.Markdown(value="Ready. Enter an HKEX stock code and click Analyze.")
        chart = gr.Plot(label="K-line candlestick price history")
        events = gr.Dataframe(
            headers=["Announcement date", "Title", "Sentiment", "Confidence", "AI Executive Summary", "Return %", "Market status"],
            label="Event-window returns and announcement analysis",
            datatype=["str", "str", "str", "str", "str", "number", "str"],
            wrap=True,
        )
        run.click(
            analyze_stock,
            inputs=[stock_code, start_date, end_date],
            outputs=[status, chart, events],
            trigger_mode="multiple",
            concurrency_limit=None,
        )
        gr.Markdown(f"*{DISCLAIMER}*")
    return demo


if __name__ == "__main__":
    build_demo().launch()
