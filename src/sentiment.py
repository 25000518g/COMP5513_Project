"""Local financial sentiment analysis with lazy model loading."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import os

from .models import SentimentResult

MODEL_NAME = "ProsusAI/finbert"
MODEL_NAMES = (
    MODEL_NAME,
    "NOSIBLE/financial-sentiment-v1.1-base",
    "Sigma/financial-sentiment-analysis",
)
MODEL_LABELS = {
    MODEL_NAME: {
        "label_0": "positive",
        "label_1": "negative",
        "label_2": "neutral",
    },
    "NOSIBLE/financial-sentiment-v1.1-base": {
        "label_0": "negative",
        "label_1": "positive",
        "label_2": "neutral",
    },
    "Sigma/financial-sentiment-analysis": {
        "label_0": "negative",
        "label_1": "neutral",
        "label_2": "positive",
    },
}


def _chunks(text: str, size: int = 1_500) -> list[str]:
    words = text.split()
    return [" ".join(words[index : index + size]) for index in range(0, len(words), size)] or [""]


@lru_cache(maxsize=len(MODEL_NAMES))
def _pipeline(model_name: str = MODEL_NAME):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError("Install transformers and torch for local sentiment analysis.") from exc
    return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)


def recommendation_for_label(label: str) -> str:
    """Translate a FinBERT sentiment label into a cautious trading signal."""
    normalized = label.strip().lower()
    if normalized == "positive":
        return "Buy"
    if normalized == "negative":
        return "Sell"
    return "Hold"


def normalize_sentiment_label(model_name: str, label: str) -> str:
    """Convert model-specific output labels to positive, negative, or neutral."""
    normalized = label.strip().lower()
    return MODEL_LABELS.get(model_name, {}).get(normalized, normalized)


def analyze_sentiment(text: str, model_name: str = MODEL_NAME) -> SentimentResult:
    if not text or not text.strip():
        raise ValueError("Cannot analyze an empty announcement.")
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unsupported sentiment model: {model_name}")
    results = _pipeline(model_name)(_chunks(text[:20_000]), truncation=True)
    scores: dict[str, float] = defaultdict(float)
    for result in results:
        label = normalize_sentiment_label(model_name, str(result["label"]))
        scores[label] += float(result["score"])
    label, total = max(scores.items(), key=lambda item: item[1])
    return SentimentResult(label=label, confidence=total / len(results), chunks=len(results))
