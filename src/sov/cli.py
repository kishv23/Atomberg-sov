"""Command line entry point.

    sov run      autonomous agent: crawl YouTube + Google, analyze, write insights
    sov analyze  re-score CSVs already in data/ (no API calls)
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import re

import pandas as pd

from sov.analyze import analyze, load_config
from sov.agent import run_agent
from sov.insights import generate_insights
from sov.io import ensure_dirs, write_csv, write_json


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def cmd_run(args) -> int:
    cfg = load_config(args.config)
    if args.budget:
        cfg.setdefault("crawl", {})["max_results_per_query"] = args.budget
    seeds = [args.query] if args.query else cfg["keywords"]["seeds"]
    data_dir, out_dir = cfg["output"]["data_dir"], cfg["output"]["out_dir"]
    ensure_dirs(data_dir, out_dir)
    for seed in seeds:
        state = run_agent(cfg, seed)
        for line in state["log"]:
            logging.info(line)
        raw = os.path.join(data_dir, f"crawl_{_slug(seed)}.csv")
        write_csv(state["rows"], raw)
        logging.info("saved %d raw rows to %s", len(state["rows"]), raw)
    # Re-score everything collected so far so outputs cover every seed.
    summary = analyze(glob.glob(os.path.join(data_dir, "*.csv")), cfg, out_dir)
    if summary:
        insights = generate_insights(summary)
        write_json(insights, os.path.join(out_dir, "insights.json"))
        for i in insights:
            print(f"[{i['kind']}] {i['text']}")
    return 0


def cmd_analyze(args) -> int:
    cfg = load_config(args.config)
    inputs = args.inputs or glob.glob(os.path.join(cfg["output"]["data_dir"], "*.csv"))
    out_dir = args.out_dir or cfg["output"]["out_dir"]
    summary = analyze(inputs, cfg, out_dir)
    if not summary:
        return 1
    insights = generate_insights(summary)
    write_json(insights, os.path.join(out_dir, "insights.json"))
    table = pd.DataFrame(
        {
            "mention_share": summary["rms"]["share"],
            "wsov": summary["wsov"]["share"],
            "sopv": summary["sopv"]["share"],
        }
    ).sort_values("wsov", ascending=False)
    print(table.map(lambda v: f"{v:.1%}").to_string())
    print()
    for i in insights:
        print(f"[{i['kind']}] {i['text']}")
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(prog="sov")
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="crawl, analyze and report")
    run.add_argument("--query", help="seed query (defaults to keywords.seeds in config)")
    run.add_argument("--budget", type=int, help="max results per seed (default from config)")
    run.set_defaults(func=cmd_run)

    an = sub.add_parser("analyze", help="score CSVs in data/ without crawling")
    an.add_argument("--inputs", nargs="*")
    an.add_argument("--out_dir")
    an.set_defaults(func=cmd_analyze)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
