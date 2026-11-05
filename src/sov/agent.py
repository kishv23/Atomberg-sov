"""Autonomous Share-of-Voice agent built as a LangGraph state machine.

    plan -> crawl -> (budget met or no variants left?) -> analyze -> insights -> END
               ^-------------------- no ---------------------|

The crawl node processes one query variant per step and the router decides
whether to keep crawling, so the agent stops as soon as it reaches the
2,000-result budget or runs out of useful variants (a variant returning
nothing new counts as exhausted).

LangGraph is used when installed. Otherwise an equivalent minimal runner
executes the same nodes and edges, so the agent also works in lightweight
environments and in tests.
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Dict, List, Optional, TypedDict

import pandas as pd

from sov.analyze import prepare, score, summarise
from sov import absa
from sov.brands import brands_from_config
from sov.expand import expand_query, plan_crawl
from sov.insights import generate_insights

log = logging.getLogger(__name__)


class SoVState(TypedDict, total=False):
    config: dict
    seed: str
    plan: List[dict]
    cursor: int
    rows: List[dict]
    seen_urls: List[str]
    stalled: int
    summary: dict
    insights: List[dict]
    log: List[str]


Collector = Callable[[str, int], List[dict]]


def default_collectors(cfg: dict) -> Dict[str, Collector]:
    """Live collectors for platforms whose API keys are present."""
    out: Dict[str, Collector] = {}
    yt_key = os.getenv(cfg["platforms"]["youtube"]["api_key_env"])
    if yt_key:
        from sov.collectors.youtube import fetch_youtube

        out["youtube"] = lambda q, n: fetch_youtube(yt_key, q, n)
    g = cfg["platforms"]["google_cse"]
    g_key, g_cx = os.getenv(g["api_key_env"]), os.getenv(g["engine_id_env"])
    if g_key and g_cx:
        from sov.collectors.google import fetch_google

        out["google"] = lambda q, n: fetch_google(g_key, g_cx, q, n)
    return out


def build_nodes(collectors: Dict[str, Collector], max_stalls: int = 3):
    def plan(state: SoVState) -> SoVState:
        cfg = state["config"]
        crawl = cfg.get("crawl", {})
        brands = [b.name for b in brands_from_config(cfg)]
        variants = expand_query(state["seed"], cfg["keywords"].get("modifiers", []), brands)
        p = plan_crawl(
            variants,
            crawl.get("max_results_per_query", 2000),
            crawl.get("youtube_per_variant", 200),
            crawl.get("google_per_variant", 100),
        )
        return {"plan": p, "cursor": 0, "rows": state.get("rows", []), "seen_urls": [], "stalled": 0,
                "log": [f"planned {len(p)} query variants for '{state['seed']}'"]}

    def crawl(state: SoVState) -> SoVState:
        step = state["plan"][state["cursor"]]
        seen = set(state["seen_urls"])
        new_rows = []
        for platform, collect in collectors.items():
            n = step.get(platform, 0)
            if not n:
                continue
            try:
                fetched = collect(step["query"], n)
            except Exception as exc:  # one platform failing must not stop the run
                log.warning("%s failed for '%s': %s", platform, step["query"], exc)
                fetched = []
            for r in fetched:
                if r.get("url") and r["url"] not in seen:
                    seen.add(r["url"])
                    new_rows.append({**r, "seed": state["seed"]})
        stalled = state["stalled"] + 1 if not new_rows else 0
        return {
            "rows": state["rows"] + new_rows,
            "seen_urls": list(seen),
            "cursor": state["cursor"] + 1,
            "stalled": stalled,
            "log": state["log"] + [f"crawled '{step['query']}': +{len(new_rows)} new (total {len(state['rows']) + len(new_rows)})"],
        }

    def should_continue(state: SoVState) -> str:
        budget = state["config"].get("crawl", {}).get("max_results_per_query", 2000)
        if len(state["rows"]) >= budget:
            return "analyze"
        if state["cursor"] >= len(state["plan"]) or state["stalled"] >= max_stalls:
            return "analyze"
        return "crawl"

    def analyze(state: SoVState) -> SoVState:
        if not state["rows"]:
            return {"summary": {}, "log": state["log"] + ["no rows collected; nothing to analyze"]}
        df, matcher = score(prepare(pd.DataFrame(state["rows"])), state["config"])
        summary = summarise(df, state["config"], matcher)
        summary["aspects"] = absa.brand_aspect_table(df["scan_text"].tolist(), matcher, state["config"].get("absa", {}).get("backend", "auto"))
        return {"summary": summary, "log": state["log"] + [f"analyzed {summary['total_items']} unique items"]}

    def insights(state: SoVState) -> SoVState:
        found = generate_insights(state["summary"]) if state.get("summary") else []
        return {"insights": found, "log": state["log"] + [f"generated {len(found)} insights"]}

    return {"plan": plan, "crawl": crawl, "analyze": analyze, "insights": insights}, should_continue


class _SequentialGraph:
    """Minimal stand-in for a compiled LangGraph with the same topology."""

    def __init__(self, nodes, router, max_steps: int = 500):
        self.nodes, self.router, self.max_steps = nodes, router, max_steps

    def invoke(self, state: SoVState) -> SoVState:
        state = {**state, **self.nodes["plan"](state)}
        for _ in range(self.max_steps):
            if not state["plan"] or self.router(state) == "analyze":
                break
            state = {**state, **self.nodes["crawl"](state)}
        state = {**state, **self.nodes["analyze"](state)}
        return {**state, **self.nodes["insights"](state)}


def build_graph(collectors: Dict[str, Collector], use_langgraph: Optional[bool] = None):
    nodes, router = build_nodes(collectors)
    if use_langgraph is None:
        try:
            import langgraph  # noqa: F401

            use_langgraph = True
        except ImportError:
            use_langgraph = False
    if not use_langgraph:
        return _SequentialGraph(nodes, router)

    from langgraph.graph import END, StateGraph

    g = StateGraph(SoVState)
    for name, fn in nodes.items():
        g.add_node(name, fn)
    g.set_entry_point("plan")
    g.add_conditional_edges("plan", lambda s: router(s) if s["plan"] else "analyze", {"crawl": "crawl", "analyze": "analyze"})
    g.add_conditional_edges("crawl", router, {"crawl": "crawl", "analyze": "analyze"})
    g.add_edge("analyze", "insights")
    g.add_edge("insights", END)
    return g.compile()


def run_agent(cfg: dict, seed: Optional[str] = None, collectors: Optional[Dict[str, Collector]] = None, use_langgraph: Optional[bool] = None) -> SoVState:
    collectors = collectors if collectors is not None else default_collectors(cfg)
    if not collectors:
        raise RuntimeError("No collectors configured. Set YOUTUBE_API_KEY and/or GOOGLE_CSE_API_KEY + GOOGLE_CSE_ENGINE_ID.")
    graph = build_graph(collectors, use_langgraph)
    # Deep crawls loop through many crawl steps; raise LangGraph's default recursion limit.
    kwargs = {"config": {"recursion_limit": 200}} if not isinstance(graph, _SequentialGraph) else {}
    return graph.invoke({"config": cfg, "seed": seed or cfg["keywords"]["seeds"][0]}, **kwargs)
