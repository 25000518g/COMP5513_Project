"""Local financial sentiment analysis with lazy model loading."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

from .models import SentimentResult

MODEL_NAME = "ProsusAI/finbert"


def _chunks(text: str, size: int = 1_500) -> list[str]:
    words = text.split()
    return [" ".join(words[index : index + size]) for index in range(0, len(words), size)] or [""]


@lru_cache(maxsize=1)
def _pipeline():
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError("Install transformers and torch for local sentiment analysis.") from exc
    return pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME)


def analyze_sentiment(text: str) -> SentimentResult:
    if not text or not text.strip():
        raise ValueError("Cannot analyze an empty announcement.")
    results = _pipeline()(_chunks(text), truncation=True)
    scores: dict[str, float] = defaultdict(float)
    for result in results:
        scores[str(result["label"]).lower()] += float(result["score"])
    label, total = max(scores.items(), key=lambda item: item[1])
    return SentimentResult(label=label, confidence=total / len(results), chunks=len(results))
