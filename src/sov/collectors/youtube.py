import os
import argparse
import logging
import yaml
from googleapiclient.discovery import build
from sov.io import write_csv, ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def fetch_youtube(api_key: str, query: str, max_results: int = 200):
    """
    Fetch up to max_results YouTube videos for a query.
    - Uses pagination (pageToken) to fetch beyond 50 results.
    - Enriches rows with views, likes, comments, published date.
    """
    youtube = build("youtube", "v3", developerKey=api_key)

    rows = []
    page_token = None
    rank, fetched = 1, 0

    while fetched < max_results:
        batch_size = min(50, max_results - fetched)
        search_request = youtube.search().list(
            q=query,
            part="id,snippet",
            maxResults=batch_size,
            type="video",
            pageToken=page_token
        )
        search_response = search_request.execute()

        video_ids = [item["id"]["videoId"] for item in search_response.get("items", [])]
        if not video_ids:
            break

        # Fetch statistics for the batch
        video_request = youtube.videos().list(
            part="snippet,statistics",
            id=",".join(video_ids)
        )
        video_response = video_request.execute()
        video_map = {item["id"]: item for item in video_response.get("items", [])}

        for item in search_response.get("items", []):
            vid = item["id"]["videoId"]
            snippet = item["snippet"]
            stats = video_map.get(vid, {}).get("statistics", {})

            rows.append({
                "platform": "youtube",
                "query": query,
                "rank": rank,
                "url": f"https://www.youtube.com/watch?v={vid}",
                "title": snippet.get("title"),
                "snippet": snippet.get("description"),
                "publisher": snippet.get("channelTitle"),
                "views": int(stats.get("viewCount", 0)) if "viewCount" in stats else None,
                "likes": int(stats.get("likeCount", 0)) if "likeCount" in stats else None,
                "comments": int(stats.get("commentCount", 0)) if "commentCount" in stats else None,
                "published_at": snippet.get("publishedAt"),
                "raw_text": f"{snippet.get('title','')} {snippet.get('description','')}"
            })
            rank += 1
            fetched += 1

        page_token = search_response.get("nextPageToken")
        if not page_token:
            break

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="data/youtube.csv")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    api_key = os.getenv(cfg["platforms"]["youtube"]["api_key_env"])
    if not api_key:
        logging.error("Missing YOUTUBE_API_KEY env var.")
        return

    ensure_dirs(os.path.dirname(args.out))
    query = cfg["keywords"]["seeds"][0]
    max_results = min(cfg["default_top_n"], 200)  # cap at 200
    logging.info(f"Fetching up to {max_results} YouTube results for query: '{query}'")
    rows = fetch_youtube(api_key, query, max_results)
    write_csv(rows, args.out)
    logging.info(f"Saved {len(rows)} YouTube rows → {args.out}")


if __name__ == "__main__":
    main()
