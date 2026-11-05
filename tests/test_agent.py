import importlib.util

import pytest

from sov.agent import build_graph, run_agent
from sov.insights import generate_insights

CFG = {
    "project_name": "t",
    "crawl": {"max_results_per_query": 30, "youtube_per_variant": 10, "google_per_variant": 5},
    "platforms": {"youtube": {"api_key_env": "NOPE_YT"}, "google_cse": {"api_key_env": "NOPE_G", "engine_id_env": "NOPE_CX"}},
    "keywords": {"seeds": ["smart fan"], "modifiers": ["best", "review", "BLDC", "wifi"]},
    "brands": {"primary": {"name": "Atomberg"}, "competitors": [{"name": "Havells"}]},
    "sentiment": {"backend": "lexicon"},
    "absa": {"backend": "lexicon"},
}


def fake_collector(platform, texts):
    def collect(query, n):
        return [
            {"platform": platform, "query": query, "rank": i + 1, "url": f"{platform}:{query}:{i}",
             "title": texts[i % len(texts)], "snippet": "", "raw_text": texts[i % len(texts)], "publisher": "p",
             "views": 1000 if platform == "youtube" else None}
            for i in range(n)
        ]
    return collect


TEXTS = ["Atomberg fan is silent and efficient", "Havells fan is noisy", "Atomberg remote is great"]


def test_agent_crawls_until_budget_then_analyzes():
    collectors = {"youtube": fake_collector("youtube", TEXTS), "google": fake_collector("google", TEXTS)}
    state = run_agent(CFG, collectors=collectors, use_langgraph=False)
    assert len(state["rows"]) == 30
    assert state["summary"]["total_items"] == 30
    assert state["summary"]["rms"]["share"]["Atomberg"] > state["summary"]["rms"]["share"]["Havells"]
    assert state["insights"]
    assert any("planned" in line for line in state["log"])


def test_agent_stops_when_variants_return_nothing_new():
    calls = []

    def same_results(query, n):
        calls.append(query)
        return [{"platform": "google", "query": query, "rank": 1, "url": "same", "raw_text": "Atomberg"}]

    cfg = {**CFG, "crawl": {**CFG["crawl"], "max_results_per_query": 1000}}  # budget never binds
    state = run_agent(cfg, collectors={"google": same_results}, use_langgraph=False)
    assert len(state["plan"]) > 4
    assert len(state["rows"]) == 1
    assert len(calls) == 4  # first call adds a row, then three stalls in a row stop the crawl


def test_one_failing_platform_does_not_stop_the_crawl():
    def broken(query, n):
        raise RuntimeError("quota exceeded")

    state = run_agent(CFG, collectors={"youtube": broken, "google": fake_collector("google", TEXTS)}, use_langgraph=False)
    assert state["rows"] and all(r["platform"] == "google" for r in state["rows"])


def test_run_agent_requires_collectors():
    with pytest.raises(RuntimeError, match="No collectors"):
        run_agent(CFG, collectors={})


@pytest.mark.skipif(importlib.util.find_spec("langgraph") is None, reason="langgraph not installed")
def test_langgraph_and_fallback_runner_agree():
    collectors = {"youtube": fake_collector("youtube", TEXTS)}
    a = run_agent(CFG, collectors=collectors, use_langgraph=True)
    b = run_agent(CFG, collectors=collectors, use_langgraph=False)
    assert a["summary"]["rms"] == b["summary"]["rms"]


def test_build_graph_without_langgraph_returns_runner():
    assert hasattr(build_graph({}, use_langgraph=False), "invoke")


def test_insights_flag_gaps_risks_and_rising_competitors():
    summary = {
        "primary": "A",
        "brands": ["A", "B"],
        "rms": {"share": {"A": 0.7, "B": 0.3}},
        "wsov": {"share": {"A": 0.5, "B": 0.5}},
        "sopv": {"share": {"A": 0.4, "B": 0.6}},
        "benchmark": [
            {"metric": "rms", "primary_rank": 1, "primary_share": 0.7, "leader": "A", "gap_to_leader": 0},
            {"metric": "sopv", "primary_rank": 2, "primary_share": 0.4, "leader": "B", "gap_to_leader": 0.2},
        ],
        "aspects": [{"brand": "A", "aspect": "noise", "mentions": 5, "positive": 0, "negative": 4, "net_sentiment": -0.8}],
    }
    kinds = [i["kind"] for i in generate_insights(summary)]
    assert kinds[0] == "risk"
    assert "gap" in kinds and "watch" in kinds and "strength" in kinds
