"""Aspect-based sentiment analysis (ABSA).

Finds which product aspects people talk about (noise, energy, remote/app,
design, price, airflow, build, service) and the sentiment attached to each,
per brand mentioned in the same sentence.

Backends:
  pyabsa   PyABSA ATEPC multilingual checkpoint (extracts aspects and polarity)
  lexicon  offline aspect keyword map + clause-level lexicon sentiment
  auto     pyabsa when installed, otherwise lexicon
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List

from sov.brands import BrandMatcher
from sov.sentiment import label, lexicon_score

log = logging.getLogger(__name__)

ASPECTS: Dict[str, List[str]] = {
    "noise": ["noise", "noisy", "silent", "quiet", "sound", "hum", "humming", "loud"],
    "energy": ["energy", "power", "electricity", "bill", "saving", "saves", "bldc", "watt", "watts", "efficient", "efficiency", "star rating"],
    "smart_features": ["remote", "app", "wifi", "wi-fi", "alexa", "google home", "voice", "smart", "timer", "iot"],
    "design": ["design", "look", "looks", "style", "stylish", "colour", "color", "finish", "aesthetic", "light", "led"],
    "price": ["price", "cost", "expensive", "cheap", "affordable", "value", "worth", "budget", "₹", "rs"],
    "airflow": ["airflow", "air flow", "speed", "rpm", "breeze", "cooling", "air delivery", "cmm"],
    "build_quality": ["build", "quality", "durable", "durability", "wobble", "motor", "blade", "blades", "sturdy"],
    "service": ["service", "warranty", "installation", "support", "customer care", "technician", "delivery"],
}

_SENTENCE = re.compile(r"(?<=[.!?\n])\s+|\s+[|•]\s+")
_ASPECT_PATTERNS = {
    a: re.compile(r"(?<![a-z])(" + "|".join(re.escape(t) for t in terms) + r")(?![a-z])", re.IGNORECASE)
    for a, terms in ASPECTS.items()
}


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE.split(text or "") if s and s.strip()]


def lexicon_aspects(text: str) -> List[dict]:
    """Returns [{aspect, term, sentiment, polarity, sentence}] for each aspect hit."""
    found = []
    for sentence in split_sentences(text):
        polarity = lexicon_score(sentence)
        for aspect, pattern in _ASPECT_PATTERNS.items():
            m = pattern.search(sentence)
            if m:
                found.append(
                    {
                        "aspect": aspect,
                        "term": m.group(1).lower(),
                        "sentiment": polarity,
                        "polarity": label(polarity, 0.15),
                        "sentence": sentence,
                    }
                )
    return found


@lru_cache(maxsize=1)
def _pyabsa_extractor():
    from pyabsa import AspectTermExtraction as ATEPC  # optional, heavy

    return ATEPC.AspectExtractor("multilingual", auto_device=True, cal_perplexity=False)


def _canonical_aspect(term: str) -> str:
    for aspect, pattern in _ASPECT_PATTERNS.items():
        if pattern.search(term):
            return aspect
    return "other"


def pyabsa_aspects(text: str) -> List[dict]:
    extractor = _pyabsa_extractor()
    found = []
    for sentence in split_sentences(text):
        res = extractor.predict(sentence, print_result=False, save_result=False, ignore_error=True)
        for term, pol, conf in zip(res.get("aspect", []), res.get("sentiment", []), res.get("confidence", [1.0] * 99)):
            score = {"Positive": 1.0, "Negative": -1.0}.get(pol, 0.0) * float(conf)
            found.append(
                {"aspect": _canonical_aspect(term), "term": term.lower(), "sentiment": score, "polarity": pol.lower(), "sentence": sentence}
            )
    return found


def resolve_backend(name: str = "auto") -> str:
    if name in ("pyabsa", "lexicon"):
        return name
    try:
        import pyabsa  # noqa: F401

        return "pyabsa"
    except ImportError:
        log.info("pyabsa not installed; using lexicon ABSA")
        return "lexicon"


def extract(text: str, backend: str = "auto") -> List[dict]:
    return pyabsa_aspects(text) if resolve_backend(backend) == "pyabsa" else lexicon_aspects(text)


def brand_aspect_table(texts: List[str], matcher: BrandMatcher, backend: str = "auto") -> List[dict]:
    """Aggregates aspect sentiment per brand.

    An aspect is attributed to a brand when both appear in the same sentence,
    which keeps "Atomberg is silent but Havells is noisy" from blending.
    """
    backend = resolve_backend(backend)  # resolve once, not per text
    agg = defaultdict(lambda: {"mentions": 0, "positive": 0, "negative": 0, "neutral": 0, "score_sum": 0.0})
    for text in texts:
        for hit in extract(text, backend):
            counts = matcher.count(hit["sentence"])
            for brand, n in counts.items():
                if n == 0:
                    continue
                row = agg[(brand, hit["aspect"])]
                row["mentions"] += 1
                row[hit["polarity"] if hit["polarity"] in ("positive", "negative") else "neutral"] += 1
                row["score_sum"] += hit["sentiment"]
    out = []
    for (brand, aspect), r in sorted(agg.items()):
        out.append(
            {
                "brand": brand,
                "aspect": aspect,
                "mentions": r["mentions"],
                "positive": r["positive"],
                "negative": r["negative"],
                "neutral": r["neutral"],
                "avg_sentiment": r["score_sum"] / r["mentions"],
                "net_sentiment": (r["positive"] - r["negative"]) / r["mentions"],
            }
        )
    return out
