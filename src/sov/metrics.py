"""Share-of-Voice metrics.

Mention Share (RMS)  share of raw brand mentions
wSoV                 mentions weighted by engagement and sentiment
SoPV                 share of positive voice, i.e. wSoV restricted to positive items
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping

# Approximate organic click-through rate by search position. Google results
# carry no view counts, so rank is the best available proxy for reach.
_CTR_BY_RANK = [0.28, 0.15, 0.11, 0.08, 0.07, 0.05, 0.04, 0.03, 0.03, 0.02]
_SERP_IMPRESSIONS = 100_000  # nominal monthly impressions per query, used to scale CTR to a view-like number


def safe_num(x) -> float:
    """Converts counts that may be None, '' or NaN into a non-negative float."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(v) or v < 0 else v


def ctr_for_rank(rank) -> float:
    r = int(safe_num(rank)) or 1
    if r <= len(_CTR_BY_RANK):
        return _CTR_BY_RANK[r - 1]
    return max(0.002, 0.02 * 10 / r)


def engagement_score(platform: str, views=None, likes=None, comments=None, rank=None) -> float:
    """Log-scaled engagement. Search results without stats fall back to rank-estimated reach."""
    v, l, c = safe_num(views), safe_num(likes), safe_num(comments)
    if platform == "google" and v == 0 and l == 0 and c == 0:
        v = ctr_for_rank(rank) * _SERP_IMPRESSIONS
    return 0.6 * math.log1p(v) + 0.3 * math.log1p(l) + 0.1 * math.log1p(c)


def sentiment_weight(sentiment: float) -> float:
    """Maps sentiment in [-1, 1] to a weight in [0, 1]; neutral counts half."""
    try:
        s = float(sentiment)
    except (TypeError, ValueError):
        s = 0.0
    if math.isnan(s):
        s = 0.0
    s = max(-1.0, min(1.0, s))
    return (1 + s) / 2


def shares(totals: Mapping[str, float]) -> Dict[str, float]:
    total = sum(totals.values())
    if total <= 0:
        return {k: 0.0 for k in totals}
    return {k: v / total for k, v in totals.items()}


def aggregate(items: Iterable[dict], brands: list[str]) -> dict:
    """Aggregates per-item mentions into RMS, wSoV and SoPV totals and shares.

    Each item needs: mentions (brand -> count), engagement, sentiment, sentiment_label.
    An item's weight is split across brands in proportion to their mentions.
    """
    rms = {b: 0.0 for b in brands}
    wsov = {b: 0.0 for b in brands}
    sopv = {b: 0.0 for b in brands}
    sentiment_counts = {b: {"positive": 0, "neutral": 0, "negative": 0} for b in brands}

    for it in items:
        m = it["mentions"]
        total = sum(m.values())
        if total == 0:
            continue
        weight = it["engagement"] * sentiment_weight(it["sentiment"])
        for b, c in m.items():
            if c == 0:
                continue
            rms[b] += c
            share = (c / total) * weight
            wsov[b] += share
            if it["sentiment_label"] == "positive":
                sopv[b] += share
            sentiment_counts[b][it["sentiment_label"]] += 1

    return {
        "rms": {"totals": rms, "share": shares(rms)},
        "wsov": {"totals": wsov, "share": shares(wsov)},
        "sopv": {"totals": sopv, "share": shares(sopv)},
        "sentiment_breakdown": sentiment_counts,
    }


def benchmark(summary: dict, primary: str) -> list[dict]:
    """Ranks every brand against the primary brand on each metric."""
    rows = []
    for metric in ("rms", "wsov", "sopv"):
        share = summary[metric]["share"]
        ordered = sorted(share, key=share.get, reverse=True)
        mine = share.get(primary, 0.0)
        leader = ordered[0] if ordered else None
        rows.append(
            {
                "metric": metric,
                "primary_share": mine,
                "primary_rank": ordered.index(primary) + 1 if primary in ordered else None,
                "leader": leader,
                "leader_share": share.get(leader, 0.0),
                "gap_to_leader": (share.get(leader, 0.0) - mine) if leader != primary else 0.0,
            }
        )
    return rows
