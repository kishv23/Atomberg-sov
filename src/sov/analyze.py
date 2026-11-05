from __future__ import annotations
import os, math, json, argparse, logging, glob, yaml
from typing import Dict, List
import pandas as pd
from datetime import datetime

from sov.io import ensure_dirs, read_csvs, write_csv, write_json
from sov.brands import compile_brand_regexes, count_mentions
from sov.sentiment import sentiment_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ----------------- Helpers -----------------
def safe_num(x) -> float:
    try:
        return float(x) if x not in [None, ""] else 0.0
    except Exception:
        return 0.0

def engagement_score(views, likes, comments) -> float:
    return (0.6 * math.log1p(safe_num(views)) +
            0.3 * math.log1p(safe_num(likes)) +
            0.1 * math.log1p(safe_num(comments)))

def wsov_weight(eng: float, sent: float) -> float:
    return eng * (1 + sent) / 2  # sentiment [-1,1] → weight [0,1]

def sentiment_label(score: float, threshold: float = 0.25) -> str:
    if score > threshold: return "positive"
    if score < -threshold: return "negative"
    return "neutral"

def compute_shares(totals: Dict[str, float]) -> Dict[str, float]:
    s = sum(totals.values()) or 1.0
    return {k: v/s for k, v in totals.items()}

# ----------------- Analyzer -----------------
def analyze(inputs: List[str], cfg: dict, out_dir: str):
    brands_all = list(dict.fromkeys(cfg["brands"]["primary"] + cfg["brands"]["competitors"]))
    brand_patterns = compile_brand_regexes(cfg["brands"]["primary"], cfg["brands"]["competitors"])

    df = read_csvs(inputs).drop_duplicates(subset=["url"]).reset_index(drop=True)
    if df.empty:
        logging.error("No data rows found. Run collectors first.")
        return

    # Fill missing columns
    for col in ["title","snippet","raw_text","publisher","views","likes","comments","published_at"]:
        if col not in df.columns: df[col] = None
    df["scan_text"] = (df["title"].fillna("") + " " + df["snippet"].fillna("") + " " + df["raw_text"].fillna(""))

    # Mentions
    mentions, dom = [], []
    for txt in df["scan_text"]:
        m = count_mentions(txt, brand_patterns)
        mentions.append(json.dumps(m))
        dom.append(max(m, key=m.get) if sum(m.values())>0 else "")
    df["brand_mentions_json"], df["dominant_brand"] = mentions, dom

    # Sentiment & Engagement
    logging.info("Running sentiment (may download model on first run)…")
    df["sentiment"] = df["scan_text"].map(sentiment_score)
    df["sentiment_label"] = df["sentiment"].map(sentiment_label)
    df["engagement"] = [engagement_score(v,l,c) for v,l,c in zip(df["views"],df["likes"],df["comments"])]
    df["wsov_item"] = [wsov_weight(e,s) for e,s in zip(df["engagement"],df["sentiment"])]

    # Parse date
    def parse_date(x):
        try: return datetime.fromisoformat(str(x).replace("Z",""))
        except: return None
    df["published_at"] = df["published_at"].map(parse_date)

    # Aggregates
    rms, wsov, sopv = {b:0 for b in brands_all}, {b:0 for b in brands_all}, {b:0 for b in brands_all}
    sentiment_counts = {b:{"positive":0,"neutral":0,"negative":0} for b in brands_all}

    for _, row in df.iterrows():
        m = json.loads(row["brand_mentions_json"])
        total = sum(m.values())
        w, sent_lab = row["wsov_item"], row["sentiment_label"]

        for b in brands_all: rms[b] += int(m.get(b,0))
        if total>0 and w>0:
            for b,c in m.items():
                if c>0:
                    share=(c/total)*w
                    wsov[b]+=share
                    if sent_lab=="positive": sopv[b]+=share
        for b in brands_all:
            if m.get(b,0)>0: sentiment_counts[b][sent_lab]+=1

    summary = {
        "project": cfg.get("project_name","SoV"),
        "query": cfg["keywords"]["seeds"][0],
        "total_items": len(df),
        "brands": brands_all,
        "rms": {"totals": rms, "share": compute_shares(rms)},
        "wsov": {"totals": wsov, "share": compute_shares(wsov)},
        "sopv": {"totals": sopv, "share": compute_shares(sopv)},
        "sentiment_breakdown": sentiment_counts,
        "top_publishers": (df.groupby("publisher")["url"].count()
                             .sort_values(ascending=False).head(10)
                             .reset_index().rename(columns={"url":"count"})
                             .to_dict(orient="records"))
    }

    ensure_dirs(out_dir)
    df.to_csv(os.path.join(out_dir,"scored.csv"),index=False,encoding="utf-8")
    write_json(summary, os.path.join(out_dir,"summary.json"))

    brand_rows=[]
    for b in brands_all:
        brand_rows.append({
            "brand":b,
            "mentions":rms[b],
            "wsov":wsov[b],
            "sopv":sopv[b],
            **sentiment_counts[b]
        })
    write_csv(brand_rows, os.path.join(out_dir,"brand_summary.csv"))
    logging.info(f"Saved {len(df)} rows → scored.csv, summary.json, brand_summary.csv")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="config.yaml")
    parser.add_argument("--inputs",nargs="*",default=None)
    parser.add_argument("--out_dir",default="out")
    args=parser.parse_args()

    with open(args.config,"r",encoding="utf-8") as f: cfg=yaml.safe_load(f)
    inputs=args.inputs or glob.glob(os.path.join(cfg["output"]["data_dir"],"*.csv"))
    analyze(inputs,cfg,args.out_dir)

if __name__=="__main__":
    main()
