"""Scores crawled rows and writes the Share-of-Voice outputs."""
from __future__ import annotations

import argparse
import glob
import logging
import os
from typing import List

import pandas as pd
import yaml

from sov import absa, sentiment
from sov.brands import BrandMatcher, brands_from_config
from sov.io import ensure_dirs, read_csvs, write_csv, write_json
from sov.metrics import aggregate, benchmark, engagement_score

log = logging.getLogger(__name__)

TEXT_COLUMNS = ["title", "snippet", "raw_text", "publisher", "views", "likes", "comments", "published_at", "rank", "platform", "query"]


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "brands" not in cfg:
        raise ValueError(f"{path} has no 'brands' section")
    return cfg


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    for col in TEXT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    # raw_text is title + snippet in both collectors, so scan it once instead of tripling the text.
    fallback = df["title"].fillna("") + " " + df["snippet"].fillna("")
    raw = df["raw_text"].fillna("").astype(str)
    df["scan_text"] = raw.where(raw.str.strip() != "", fallback)
    return df


def score(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, BrandMatcher]:
    matcher = BrandMatcher(brands_from_config(cfg))
    scfg = cfg.get("sentiment", {})
    threshold = scfg.get("threshold", 0.25)

    df["mentions"] = df["scan_text"].map(matcher.count)
    df["dominant_brand"] = df["mentions"].map(lambda m: max(m, key=m.get) if sum(m.values()) else "")
    df["sentiment"] = sentiment.score_batch(df["scan_text"].tolist(), scfg.get("backend", "auto"))
    df["sentiment_label"] = df["sentiment"].map(lambda s: sentiment.label(s, threshold))
    df["engagement"] = [
        engagement_score(p, v, l, c, r) for p, v, l, c, r in zip(df["platform"], df["views"], df["likes"], df["comments"], df["rank"])
    ]
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    return df, matcher


def summarise(df: pd.DataFrame, cfg: dict, matcher: BrandMatcher) -> dict:
    items = df[["mentions", "engagement", "sentiment", "sentiment_label"]].to_dict(orient="records")
    summary = {
        "project": cfg.get("project_name", "SoV"),
        "queries": sorted(df["query"].dropna().unique().tolist()),
        "total_items": int(len(df)),
        "items_with_brand": int((df["dominant_brand"] != "").sum()),
        "platforms": df["platform"].value_counts().to_dict(),
        "brands": matcher.names,
        "primary": matcher.primary[0] if matcher.primary else matcher.names[0],
        **aggregate(items, matcher.names),
    }
    summary["benchmark"] = benchmark(summary, summary["primary"])
    summary["top_publishers"] = (
        df.groupby("publisher")["url"].count().sort_values(ascending=False).head(10).rename("count").reset_index().to_dict(orient="records")
    )
    return summary


def analyze(inputs: List[str], cfg: dict, out_dir: str) -> dict | None:
    df = read_csvs(inputs)
    if df.empty:
        log.error("No data rows found. Run the collectors first.")
        return None
    df, matcher = score(prepare(df), cfg)
    summary = summarise(df, cfg, matcher)
    aspects = absa.brand_aspect_table(df["scan_text"].tolist(), matcher, cfg.get("absa", {}).get("backend", "auto"))
    summary["aspects"] = aspects

    ensure_dirs(out_dir)
    scored = df.copy()
    scored["mentions"] = scored["mentions"].map(lambda m: {k: v for k, v in m.items() if v})
    scored.to_csv(os.path.join(out_dir, "scored.csv"), index=False, encoding="utf-8")
    write_json(summary, os.path.join(out_dir, "summary.json"))
    write_csv(aspects, os.path.join(out_dir, "aspects.csv"), columns=None)
    brand_rows = [
        {
            "brand": b,
            "mentions": summary["rms"]["totals"][b],
            "mention_share": summary["rms"]["share"][b],
            "wsov_share": summary["wsov"]["share"][b],
            "sopv_share": summary["sopv"]["share"][b],
            **summary["sentiment_breakdown"][b],
        }
        for b in matcher.names
    ]
    write_csv(brand_rows, os.path.join(out_dir, "brand_summary.csv"), columns=None)
    log.info("Scored %d items -> %s", len(df), out_dir)
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--inputs", nargs="*", default=None)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    inputs = args.inputs or glob.glob(os.path.join(cfg["output"]["data_dir"], "*.csv"))
    analyze(inputs, cfg, args.out_dir or cfg["output"]["out_dir"])


if __name__ == "__main__":
    main()
