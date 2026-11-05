from __future__ import annotations
from typing import Optional
from transformers import pipeline

_SENTIMENT_PIPE = None

def _get_pipeline():
    global _SENTIMENT_PIPE
    if _SENTIMENT_PIPE is None:
        _SENTIMENT_PIPE = pipeline(
            task="sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
    return _SENTIMENT_PIPE

def sentiment_score(text: Optional[str]) -> float:
    if not text or not text.strip():
        return 0.0
    pipe = _get_pipeline()
    res = pipe(text[:512])[0]
    label = res["label"].upper()
    prob = float(res["score"])
    if label == "POSITIVE":
        return prob
    elif label == "NEGATIVE":
        return -prob
    return 0.0
