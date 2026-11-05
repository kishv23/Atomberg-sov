import json

import pandas as pd
import pytest

from sov.analyze import analyze, load_config

ROWS = [
    {"platform": "youtube", "query": "smart fan", "rank": 1, "url": "u1", "title": "Atomberg Renesa review",
     "snippet": "Super silent and efficient BLDC fan", "publisher": "TechGuru", "views": 50000, "likes": 900, "comments": 40,
     "published_at": "2025-03-01T00:00:00Z", "raw_text": "Atomberg Renesa review. Super silent and efficient BLDC fan"},
    {"platform": "google", "query": "smart fan", "rank": 2, "url": "u2", "title": "Havells fan problems",
     "snippet": "Noisy motor and poor service", "publisher": "forum.example", "views": None, "likes": None, "comments": None,
     "published_at": None, "raw_text": "Havells fan problems. Noisy motor and poor service"},
    {"platform": "google", "query": "smart fan", "rank": 3, "url": "u2", "title": "duplicate url", "snippet": "",
     "publisher": "x", "raw_text": "duplicate"},
]


@pytest.fixture
def cfg():
    return {
        "project_name": "test",
        "brands": {"primary": {"name": "Atomberg", "aliases": ["Renesa"]}, "competitors": [{"name": "Havells"}]},
        "sentiment": {"backend": "lexicon"},
        "absa": {"backend": "lexicon"},
    }


def test_analyze_end_to_end(tmp_path, cfg):
    src = tmp_path / "in.csv"
    pd.DataFrame(ROWS).to_csv(src, index=False)
    summary = analyze([str(src)], cfg, str(tmp_path / "out"))

    assert summary["total_items"] == 2  # duplicate URL dropped
    assert summary["primary"] == "Atomberg"
    assert summary["rms"]["totals"] == {"Atomberg": 1, "Havells": 1}
    # Google row has no stats but must still carry weight.
    assert summary["wsov"]["totals"]["Havells"] > 0
    assert summary["sopv"]["share"]["Atomberg"] == pytest.approx(1.0)

    aspects = {(a["brand"], a["aspect"]): a for a in summary["aspects"]}
    assert aspects[("Atomberg", "noise")]["positive"] == 1
    assert aspects[("Havells", "service")]["negative"] == 1

    out = tmp_path / "out"
    for name in ("scored.csv", "summary.json", "brand_summary.csv", "aspects.csv"):
        assert (out / name).exists(), name
    assert json.loads((out / "summary.json").read_text())["benchmark"][0]["primary_rank"] == 1


def test_analyze_returns_none_without_data(tmp_path, cfg):
    assert analyze([str(tmp_path / "missing.csv")], cfg, str(tmp_path / "out")) is None


def test_repo_config_is_valid():
    cfg = load_config("config.yaml")
    assert len(cfg["brands"]["competitors"]) == 10
