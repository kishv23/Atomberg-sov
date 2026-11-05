"""Query expansion so a single seed can reach the crawl budget.

Google CSE returns at most 100 results per query and YouTube search tails off
after a few hundred, so a 2,000-result budget needs several query variants.
"""
from __future__ import annotations

from typing import Iterable, List

# Modifiers that read naturally after the seed ("smart fan review").
SUFFIX_MODIFIERS = {"review", "reviews", "india", "noise", "wifi", "vs", "comparison", "price"}


def expand_query(seed: str, modifiers: Iterable[str], brands: Iterable[str] = (), limit: int = 30) -> List[str]:
    """Seed first, then modifier variants, then brand-qualified variants. No duplicates."""
    seed = seed.strip()
    variants = [seed]
    for m in modifiers:
        m = m.strip()
        if not m:
            continue
        suffix = m.lower() in SUFFIX_MODIFIERS or m.lower().startswith("under ")
        variants.append(f"{seed} {m}" if suffix else f"{m} {seed}")
    for b in brands:
        variants.append(f"{b} {seed}")
    out, seen = [], set()
    for v in variants:
        key = " ".join(v.lower().split())
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out[:limit]


def plan_crawl(variants: List[str], budget: int, youtube_per_variant: int, google_per_variant: int) -> List[dict]:
    """Allocates the result budget across variants and platforms, seed first."""
    plan, remaining = [], budget
    for v in variants:
        if remaining <= 0:
            break
        yt = min(youtube_per_variant, remaining)
        remaining -= yt
        g = min(google_per_variant, max(remaining, 0), 100)
        remaining -= g
        plan.append({"query": v, "youtube": yt, "google": g})
    return plan
