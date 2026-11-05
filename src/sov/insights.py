"""Turns a SoV summary into ranked, plain-language findings for the dashboard."""
from __future__ import annotations

from typing import List


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def generate_insights(summary: dict, min_aspect_mentions: int = 5) -> List[dict]:
    primary = summary["primary"]
    out: List[dict] = []

    for row in summary.get("benchmark", []):
        name = {"rms": "mention share", "wsov": "engagement-weighted SoV", "sopv": "share of positive voice"}[row["metric"]]
        if row["primary_rank"] == 1:
            out.append({"kind": "strength", "metric": row["metric"],
                        "text": f"{primary} leads {name} at {_pct(row['primary_share'])}."})
        else:
            out.append({"kind": "gap", "metric": row["metric"],
                        "text": f"{primary} ranks #{row['primary_rank']} on {name} ({_pct(row['primary_share'])}), "
                                f"{_pct(row['gap_to_leader'])} behind {row['leader']}."})

    # Visibility vs quality: high mentions but weaker positive share signals a perception problem.
    rms = summary["rms"]["share"].get(primary, 0.0)
    sopv = summary["sopv"]["share"].get(primary, 0.0)
    if rms > 0 and sopv < rms * 0.85:
        out.append({"kind": "risk", "metric": "sopv",
                    "text": f"{primary}'s positive share ({_pct(sopv)}) trails its mention share ({_pct(rms)}): visibility is not converting into positive sentiment."})

    # Rising competitors: a competitor whose weighted share beats its raw share is punching above its weight.
    for b in summary["brands"]:
        if b == primary:
            continue
        r, w = summary["rms"]["share"].get(b, 0.0), summary["wsov"]["share"].get(b, 0.0)
        if r >= 0.02 and w > r * 1.25:
            out.append({"kind": "watch", "metric": "wsov", "brand": b,
                        "text": f"{b} gets {_pct(w)} of weighted voice from {_pct(r)} of mentions: its content earns outsized engagement."})

    # Aspect strengths and weaknesses for the primary brand.
    aspects = [a for a in summary.get("aspects", []) if a["brand"] == primary and a["mentions"] >= min_aspect_mentions]
    for a in sorted(aspects, key=lambda a: a["net_sentiment"]):
        if a["net_sentiment"] <= -0.2:
            out.append({"kind": "risk", "metric": "absa", "aspect": a["aspect"],
                        "text": f"Negative talk about {primary}'s {a['aspect'].replace('_', ' ')} ({a['negative']} of {a['mentions']} mentions)."})
        elif a["net_sentiment"] >= 0.3:
            out.append({"kind": "strength", "metric": "absa", "aspect": a["aspect"],
                        "text": f"{a['aspect'].replace('_', ' ').capitalize()} is a strength for {primary} ({a['positive']} of {a['mentions']} mentions positive)."})

    order = {"risk": 0, "gap": 1, "watch": 2, "strength": 3}
    return sorted(out, key=lambda i: order[i["kind"]])
