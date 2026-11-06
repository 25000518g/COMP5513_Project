"""Local AI executive summaries for announcement text."""

from __future__ import annotations

from functools import lru_cache
import re

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"


def _chunks(text: str, size: int = 850) -> list[str]:
    words = text.split()
    return [" ".join(words[index : index + size]) for index in range(0, len(words), size)] or [""]


@lru_cache(maxsize=1)
def _pipeline():
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError("Install transformers and torch for AI executive summaries.") from exc
    return pipeline("summarization", model=MODEL_NAME, tokenizer=MODEL_NAME)


def _fallback_summary(text: str, sentences: int = 3) -> str:
    """Return a readable extractive summary if the model is unavailable."""

    candidates = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if len(part.strip()) > 40]
    keywords = ("result", "profit", "revenue", "dividend", "acquisition", "董事", "financial", "loss", "公告")
    ranked = sorted(
        enumerate(candidates),
        key=lambda item: (sum(word in item[1].lower() for word in keywords), -item[0]),
        reverse=True,
    )
    selected = sorted(ranked[:sentences], key=lambda item: item[0])
    return " ".join(sentence for _, sentence in selected) or text[:500]


def generate_executive_summary(text: str, bullets: int = 3) -> str:
    """Generate a concise executive summary, preferring the local AI model."""

    if not text or not text.strip():
        raise ValueError("Cannot summarize an empty announcement.")
    try:
        results = _pipeline()(
            _chunks(text),
            max_length=110,
            min_length=25,
            do_sample=False,
            truncation=True,
        )
        summaries = [str(result["summary_text"]).strip() for result in results if result.get("summary_text")]
        if summaries:
            return "\n".join(f"• {summary}" for summary in summaries[:bullets])
    except Exception:
        # A summary should not prevent sentiment and price analysis from being shown.
        pass
    return "\n".join(f"• {sentence}" for sentence in _fallback_summary(text, bullets).split(". "))
