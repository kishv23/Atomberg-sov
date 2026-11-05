"""Document-level sentiment in [-1, 1].

Backends:
  transformers  DistilBERT SST-2 via Hugging Face (best quality, downloads a model)
  lexicon       fast offline scorer tuned for product reviews (no dependencies)
  auto          transformers when installed, otherwise lexicon
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Callable, List, Optional

log = logging.getLogger(__name__)

MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

POSITIVE = {
    "best", "great", "good", "excellent", "amazing", "love", "silent", "quiet", "smooth", "efficient",
    "saves", "saving", "premium", "stylish", "recommend", "worth", "reliable", "powerful",
    "awesome", "perfect", "superb", "fantastic", "happy", "satisfied", "affordable", "value",
}
NEGATIVE = {
    "bad", "worst", "poor", "noisy", "loud", "broke", "broken", "defective", "issue", "issues",
    "problem", "problems", "waste", "expensive", "slow", "fail", "failed", "failure", "disappointed",
    "complaint", "return", "refund", "faulty", "cheap", "hate", "useless", "wobble", "hum",
}
NEGATORS = {"not", "no", "never", "dont", "don't", "isnt", "isn't", "wasnt", "wasn't", "hardly", "without"}
INTENSIFIERS = {"very": 1.5, "really": 1.5, "extremely": 2.0, "super": 1.5, "so": 1.3, "too": 1.3}

_TOKEN = re.compile(r"[a-z']+")


def lexicon_score(text: Optional[str]) -> float:
    if not text or not text.strip():
        return 0.0
    tokens = _TOKEN.findall(text.lower())
    score, hits = 0.0, 0
    for i, tok in enumerate(tokens):
        polarity = 1.0 if tok in POSITIVE else -1.0 if tok in NEGATIVE else 0.0
        if not polarity:
            continue
        window = tokens[max(0, i - 3) : i]
        if any(w in NEGATORS for w in window):
            polarity = -polarity * 0.8
        for w in window:
            polarity *= INTENSIFIERS.get(w, 1.0)
        score += polarity
        hits += 1
    if hits == 0:
        return 0.0
    # Squash to [-1, 1]; more agreeing evidence moves further from zero.
    return max(-1.0, min(1.0, score / (hits + 1)))


@lru_cache(maxsize=1)
def _transformers_pipeline():
    from transformers import pipeline  # imported lazily: heavy and optional

    return pipeline(task="sentiment-analysis", model=MODEL, truncation=True)


def transformers_score_batch(texts: List[str]) -> List[float]:
    pipe = _transformers_pipeline()
    clean = [t[:512] if t and t.strip() else "" for t in texts]
    idx = [i for i, t in enumerate(clean) if t]
    out = [0.0] * len(texts)
    for i, res in zip(idx, pipe([clean[i] for i in idx], batch_size=32)):
        prob = float(res["score"])
        out[i] = prob if res["label"].upper() == "POSITIVE" else -prob
    return out


def resolve_backend(name: str = "auto") -> str:
    if name in ("lexicon", "transformers"):
        return name
    try:
        import transformers  # noqa: F401

        return "transformers"
    except ImportError:
        log.info("transformers not installed; using lexicon sentiment")
        return "lexicon"


def score_batch(texts: List[str], backend: str = "auto") -> List[float]:
    backend = resolve_backend(backend)
    if backend == "transformers":
        return transformers_score_batch(texts)
    return [lexicon_score(t) for t in texts]


def sentiment_score(text: Optional[str], backend: str = "auto") -> float:
    """Backwards-compatible single-text helper."""
    return score_batch([text or ""], backend)[0]


def label(score: float, threshold: float = 0.25) -> str:
    if score > threshold:
        return "positive"
    if score < -threshold:
        return "negative"
    return "neutral"


Scorer = Callable[[List[str]], List[float]]
