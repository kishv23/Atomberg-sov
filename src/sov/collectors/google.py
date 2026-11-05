"""Google Programmable Search (Custom Search JSON API) collector."""
from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Callable, List, Optional

import requests
import yaml

from sov.io import ensure_dirs, write_csv

log = logging.getLogger(__name__)

ENDPOINT = "https://www.googleapis.com/customsearch/v1"
MAX_RESULTS = 100  # the API refuses start > 91
PAGE_SIZE = 10


def _get_with_retry(session, params: dict, retries: int = 3, sleep: Callable[[float], None] = time.sleep) -> dict:
    for attempt in range(retries + 1):
        resp = session.get(ENDPOINT, params=params, timeout=20)
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt < retries:
                sleep(2**attempt)
                continue
        if resp.status_code >= 400:
            raise RuntimeError(f"Google CSE {resp.status_code}: {resp.text[:200]}")
        return resp.json()
    raise RuntimeError("Google CSE retries exhausted")


def fetch_google(
    api_key: str,
    engine_id: str,
    query: str,
    max_results: int = MAX_RESULTS,
    session: Optional[requests.Session] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> List[dict]:
    """Fetches up to 100 results for a query in the unified row schema."""
    session = session or requests.Session()
    max_results = min(max_results, MAX_RESULTS)
    rows: List[dict] = []
    start = 1
    while len(rows) < max_results and start <= MAX_RESULTS - PAGE_SIZE + 1:
        params = {"key": api_key, "cx": engine_id, "q": query, "start": start, "num": PAGE_SIZE}
        log.info("Google '%s' results %d-%d", query, start, start + PAGE_SIZE - 1)
        items = _get_with_retry(session, params, sleep=sleep).get("items", [])
        if not items:
            break
        for item in items:
            if len(rows) >= max_results:
                break
            rows.append(
                {
                    "platform": "google",
                    "query": query,
                    "rank": len(rows) + 1,
                    "url": item.get("link"),
                    "title": item.get("title"),
                    "snippet": item.get("snippet"),
                    "publisher": item.get("displayLink"),
                    "views": None,
                    "likes": None,
                    "comments": None,
                    "published_at": None,
                    "raw_text": f"{item.get('title', '')} {item.get('snippet', '')}",
                }
            )
        start += PAGE_SIZE
    return rows


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--query", default=None)
    parser.add_argument("--out", default="data/google.csv")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    gcfg = cfg["platforms"]["google_cse"]
    api_key, engine_id = os.getenv(gcfg["api_key_env"]), os.getenv(gcfg["engine_id_env"])
    if not api_key or not engine_id:
        raise SystemExit(f"Set {gcfg['api_key_env']} and {gcfg['engine_id_env']}")

    ensure_dirs(os.path.dirname(args.out) or ".")
    rows = fetch_google(api_key, engine_id, args.query or cfg["keywords"]["seeds"][0])
    write_csv(rows, args.out)
    log.info("Saved %d Google rows to %s", len(rows), args.out)


if __name__ == "__main__":
    main()
