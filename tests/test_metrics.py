import math

import pytest

from sov.metrics import aggregate, benchmark, ctr_for_rank, engagement_score, safe_num, sentiment_weight, shares


def test_safe_num_handles_nan_none_and_strings():
    assert safe_num(float("nan")) == 0.0
    assert safe_num(None) == 0.0
    assert safe_num("") == 0.0
    assert safe_num("12") == 12.0


def test_google_rows_without_stats_get_rank_based_engagement():
    top = engagement_score("google", rank=1)
    deep = engagement_score("google", rank=80)
    assert top > deep > 0
    assert not math.isnan(engagement_score("google", views=float("nan"), rank=3))


def test_youtube_engagement_uses_stats():
    assert engagement_score("youtube", views=1_000_000, likes=10_000, comments=500) > engagement_score("youtube", views=100)


def test_ctr_decreases_with_rank():
    assert ctr_for_rank(1) > ctr_for_rank(10) > ctr_for_rank(50)


@pytest.mark.parametrize("s,w", [(-1, 0.0), (0, 0.5), (1, 1.0), (5, 1.0), (float("nan"), 0.5), (None, 0.5)])
def test_sentiment_weight(s, w):
    assert sentiment_weight(s) == w


def test_shares_sum_to_one_and_handle_zero():
    assert sum(shares({"a": 1, "b": 3}).values()) == pytest.approx(1.0)
    assert shares({"a": 0, "b": 0}) == {"a": 0.0, "b": 0.0}


def _item(mentions, engagement=10.0, sentiment=0.8, label="positive"):
    return {"mentions": mentions, "engagement": engagement, "sentiment": sentiment, "sentiment_label": label}


def test_aggregate_splits_weight_by_mention_share():
    summary = aggregate([_item({"A": 3, "B": 1})], ["A", "B"])
    assert summary["rms"]["totals"] == {"A": 3, "B": 1}
    assert summary["wsov"]["share"]["A"] == pytest.approx(0.75)
    assert summary["sopv"]["share"]["B"] == pytest.approx(0.25)


def test_negative_items_count_for_wsov_but_not_sopv():
    summary = aggregate([_item({"A": 1}, sentiment=-0.5, label="negative"), _item({"B": 1})], ["A", "B"])
    assert summary["sopv"]["totals"]["A"] == 0
    assert summary["wsov"]["totals"]["A"] > 0
    assert summary["sentiment_breakdown"]["A"]["negative"] == 1


def test_benchmark_reports_rank_and_gap():
    summary = aggregate([_item({"A": 1}), _item({"B": 3})], ["A", "B"])
    rms = next(r for r in benchmark(summary, "A") if r["metric"] == "rms")
    assert rms["primary_rank"] == 2
    assert rms["leader"] == "B"
    assert rms["gap_to_leader"] == pytest.approx(0.5)
