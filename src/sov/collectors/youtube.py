"""YouTube Data API v3 collector."""
from __future__ import annotations

import argparse
import logging
import os
from typing import List, Optional

import yaml

from sov.io import ensure_dirs, write_csv

log = logging.getLogger(__name__)

PAGE_SIZE = 50  # API maximum for search.list and videos.list


def build_client(api_key: str):
    from googleapiclient.discovery import build  # optional dependency

    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)


def _int_or_none(stats: dict, key: str) -> Optional[int]:
    return int(stats[key]) if key in stats else None


def fetch_youtube(api_key: Optional[str], query: str, max_results: int = 200, client=None, region: str = "IN") -> List[dict]:
    """Fetches up to max_results videos with view, like and comment counts."""
    youtube = client or build_client(api_key)
    rows: List[dict] = []
    seen = set()
    page_token = None

    while len(rows) < max_results:
        search = (
            youtube.search()
            .list(q=query, part="id,snippet", type="video", maxResults=min(PAGE_SIZE, max_results - len(rows)), pageToken=page_token, regionCode=region)
            .execute()
        )
        items = [i for i in search.get("items", []) if i["id"].get("videoId") not in seen]
        if not items:
            break
        ids = [i["id"]["videoId"] for i in items]
        details = youtube.videos().list(part="snippet,statistics", id=",".join(ids)).execute()
        stats_by_id = {v["id"]: v.get("statistics", {}) for v in details.get("items", [])}

        for item in items:
            vid = item["id"]["videoId"]
            seen.add(vid)
            snip = item["snippet"]
            stats = stats_by_id.get(vid, {})
            rows.append(
                {
                    "platform": "youtube",
                    "query": query,
                    "rank": len(rows) + 1,
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "title": snip.get("title"),
                    "snippet": snip.get("description"),
                    "publisher": snip.get("channelTitle"),
                    "views": _int_or_none(stats, "viewCount"),
                    "likes": _int_or_none(stats, "likeCount"),
                    "comments": _int_or_none(stats, "commentCount"),
                    "published_at": snip.get("publishedAt"),
                    "raw_text": f"{snip.get('title', '')} {snip.get('description', '')}",
                }
            )
            if len(rows) >= max_results:
                break

        page_token = search.get("nextPageToken")
        if not page_token:
            break
    return rows


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--query", default=None)
    parser.add_argument("--max", type=int, default=200)
    parser.add_argument("--out", default="data/youtube.csv")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    env = cfg["platforms"]["youtube"]["api_key_env"]
    api_key = os.getenv(env)
    if not api_key:
        raise SystemExit(f"Set {env}")

    ensure_dirs(os.path.dirname(args.out) or ".")
    rows = fetch_youtube(api_key, args.query or cfg["keywords"]["seeds"][0], args.max)
    write_csv(rows, args.out)
    log.info("Saved %d YouTube rows to %s", len(rows), args.out)


if __name__ == "__main__":
    main()
