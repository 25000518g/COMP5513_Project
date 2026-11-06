"""Memory-safe local executive summaries for announcement text.

The summary is extractive: it ranks sentences using financial-event terms and
returns the most informative sentences without loading a second large model.
This keeps the app usable in a small CPU-only container alongside FinBERT.
"""

from __future__ import annotations

import re


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
    """Generate a concise local executive summary without another model load."""

    if not text or not text.strip():
        raise ValueError("Cannot summarize an empty announcement.")
    return "\n".join(f"• {sentence}" for sentence in _fallback_summary(text[:20_000], bullets).split(". "))
