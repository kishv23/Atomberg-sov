from __future__ import annotations
import os
import json
from typing import List, Dict, Any
import pandas as pd

UNIFIED_COLUMNS = [
    "platform", "query", "rank", "url", "title", "snippet", "publisher",
    "views", "likes", "comments", "published_at", "raw_text"
]

def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)

_DEFAULT = object()


def write_csv(rows: List[Dict[str, Any]], path: str, columns=_DEFAULT) -> None:
    """Writes rows to CSV. By default enforces the unified crawl schema;
    pass columns=None to keep the rows' own columns."""
    df = pd.DataFrame(rows)
    if columns is _DEFAULT:
        columns = UNIFIED_COLUMNS
    if columns is not None:
        for col in columns:
            if col not in df.columns:
                df[col] = None
        df = df[columns]
    df.to_csv(path, index=False, encoding="utf-8")

def read_csvs(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if not frames:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)
    return pd.concat(frames, ignore_index=True)

def write_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
