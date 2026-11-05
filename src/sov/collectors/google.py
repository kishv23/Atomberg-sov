import os
import argparse
import logging
import yaml
import requests
from sov.io import write_csv, ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def fetch_google(api_key: str, engine_id: str, query: str, max_results: int = 100):
    """
    Fetch up to 100 Google Custom Search results for a query.
    - Paginates using 'start' parameter (1, 11, 21…).
    - Each API call returns up to 10 results.
    - Returns rows in a unified schema.
    """
    rows = []
    start, rank = 1, 1

    while rank <= max_results and start <= 91:  # Google CSE cap ~100 results
        url = (
            "https://www.googleapis.com/customsearch/v1"
            f"?key={api_key}&cx={engine_id}&q={query}&start={start}"
        )
        logging.info(f"Fetching Google results {start}–{start+9} for query '{query}'")
        resp = requests.get(url).json()

        items = resp.get("items", [])
        if not items:
            logging.warning(f"No more items returned at start={start}. Stopping pagination.")
            break

        for item in items:
            rows.append({
                "platform": "google",
                "query": query,
                "rank": rank,
                "url": item.get("link"),
                "title": item.get("title"),
                "snippet": item.get("snippet"),
                "publisher": item.get("displayLink"),
                "views": None,       # Not available from Google CSE
                "likes": None,
                "comments": None,
                "published_at": None,
                "raw_text": f"{item.get('title','')} {item.get('snippet','')}"
            })
            rank += 1
            if rank > max_results:
                break

        start += 10  # move to next page

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="data/google.csv")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    api_key = os.getenv(cfg["platforms"]["google_cse"]["api_key_env"])
    engine_id = os.getenv(cfg["platforms"]["google_cse"]["engine_id_env"])
    if not api_key or not engine_id:
        logging.error("Missing GOOGLE_CSE_API_KEY or GOOGLE_CSE_ENGINE_ID env vars.")
        return

    ensure_dirs(os.path.dirname(args.out))
    query = cfg["keywords"]["seeds"][0]  # default = "smart fan"
    max_results = min(cfg["default_top_n"], 100)  # Google CSE hard limit
    logging.info(f"Fetching up to {max_results} Google results for query: '{query}'")

    rows = fetch_google(api_key, engine_id, query, max_results)
    write_csv(rows, args.out)
    logging.info(f"Saved {len(rows)} Google rows → {args.out}")


if __name__ == "__main__":
    main()
