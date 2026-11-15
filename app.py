"""Gradio entry point for the HKEX stock-analysis demo."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import logging
from pathlib import Path
from time import perf_counter

from src.document_parser import download_and_extract
from src.hkex_client import fetch_announcements, normalize_stock_code
from src.market_data import build_candlestick_chart, fetch_prices, include_next_trading_days
from src.sentiment import MODEL_NAMES, analyze_sentiment, recommendation_for_label
from src.summarizer import generate_executive_summary

DISCLAIMER = "This educational analysis is not financial advice. Sentiment and event returns do not prove causation."
DOWNLOAD_ROOT = Path("downloads")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _summarize_model_rows(rows: list[list[object]]) -> list[list[object]]:
    """Aggregate announcement predictions into one conservative row per model."""
    summary: dict[str, dict[str, object]] = {
        model_name: {"Buy": 0, "Hold": 0, "Sell": 0, "confidence": [], "analyzed": 0}
        for model_name in MODEL_NAMES
    }
    for row in rows:
        if len(row) < 6:
            continue
        model_name, label, prediction = str(row[2]), str(row[3]).lower(), str(row[4])
        if model_name not in summary or prediction not in {"Buy", "Hold", "Sell"}:
            continue
        model_summary = summary[model_name]
        model_summary[prediction] = int(model_summary[prediction]) + 1
        model_summary["analyzed"] = int(model_summary["analyzed"]) + 1
        try:
            confidence = float(str(row[5]).rstrip("%")) / 100
        except ValueError:
            continue
        model_summary["confidence"].append(confidence)

    result = []
    for model_name in MODEL_NAMES:
        model_summary = summary[model_name]
        counts = {prediction: int(model_summary[prediction]) for prediction in ("Buy", "Hold", "Sell")}
        highest = max(counts.values())
        winners = [prediction for prediction, count in counts.items() if count == highest and count > 0]
        overall = winners[0] if len(winners) == 1 else "Hold"
        confidences = model_summary["confidence"]
        average_confidence = sum(confidences) / len(confidences) if confidences else None
        result.append(
            [
                model_name,
                counts["Buy"],
                counts["Hold"],
                counts["Sell"],
                f"{average_confidence:.1%}" if average_confidence is not None else "Unavailable",
                overall if winners else "Unavailable",
                int(model_summary["analyzed"]),
            ]
        )
    return result


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
    summary_rows = []
    chart = None

    def update(message: str):
        logger.info(message)
        return message, chart, rows, summary_rows

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
        prices = fetch_prices(code, selected_start, selected_end + timedelta(days=21))
        prices = include_next_trading_days(prices, selected_end, days=3)
        if prices is not None and len(prices) > 0:
            chart = build_candlestick_chart(prices)
            yield update(
                f"📊 Loaded {len(prices)} trading-day price records, including the next 3 sessions after the selected end date."
            )
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
        executive_summary = "Unavailable"
        model_results = []
        try:
            text = download_and_extract(announcement.url, save_path=saved_path)
            logger.info("Downloaded and extracted %s to %s", announcement.url, saved_path)
            for model_name in MODEL_NAMES:
                try:
                    sentiment = analyze_sentiment(text, model_name=model_name)
                    model_results.append(
                        [
                            announcement.published_at.isoformat(),
                            announcement.title,
                            model_name,
                            sentiment.label,
                            recommendation_for_label(sentiment.label),
                            f"{sentiment.confidence:.1%}",
                        ]
                    )
                except (RuntimeError, ValueError) as exc:
                    logger.exception("Sentiment analysis failed with %s", model_name)
                    model_results.append(
                        [
                            announcement.published_at.isoformat(),
                            announcement.title,
                            model_name,
                            "Unavailable",
                            "Unavailable",
                            f"Unavailable: {exc}",
                        ]
                    )
            try:
                executive_summary = generate_executive_summary(text)
            except (RuntimeError, ValueError) as exc:
                logger.exception("Executive summary generation failed for %s", announcement.title)
                executive_summary = f"Unavailable: {exc}"
        except (RuntimeError, ValueError) as exc:
            logger.exception("Document or sentiment analysis failed for %s", announcement.title)
            executive_summary = f"Unavailable: {exc}"
            model_results = [
                [
                    announcement.published_at.isoformat(),
                    announcement.title,
                    model_name,
                    "Unavailable",
                    "Unavailable",
                    f"Unavailable: {exc}",
                ]
                for model_name in MODEL_NAMES
            ]

        yield update(f"📝 [{index}/{len(announcements)}] Updating announcement analysis results...")
        for result_row in model_results:
            rows.append(result_row + [executive_summary])
        summary_rows[:] = _summarize_model_rows(rows)
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
            headers=[
                "Announcement date",
                "Title",
                "FinBERT model",
                "Sentiment",
                "Prediction",
                "Confidence",
                "AI Executive Summary",
            ],
            label="Announcement analysis results",
            datatype=["str", "str", "str", "str", "str", "str", "str"],
            wrap=True,
        )
        model_summary = gr.Dataframe(
            headers=[
                "FinBERT model",
                "Buy predictions",
                "Hold predictions",
                "Sell predictions",
                "Average confidence",
                "Overall prediction",
                "Announcements analyzed",
            ],
            label="Prediction summary by model",
            datatype=["str", "number", "number", "number", "str", "str", "number"],
            wrap=True,
        )
        run.click(
            analyze_stock,
            inputs=[stock_code, start_date, end_date],
            outputs=[status, chart, events, model_summary],
            trigger_mode="multiple",
            concurrency_limit=None,
        )
        gr.Markdown(f"*{DISCLAIMER}*")
    return demo


if __name__ == "__main__":
    build_demo().launch()
