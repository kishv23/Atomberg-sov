from sov.absa import brand_aspect_table, lexicon_aspects, resolve_backend as absa_backend, split_sentences
from sov.brands import Brand, BrandMatcher
from sov.sentiment import label, lexicon_score, score_batch


def test_lexicon_polarity():
    assert lexicon_score("Excellent fan, very quiet and efficient") > 0.25
    assert lexicon_score("Worst purchase, noisy and broke in a week") < -0.25
    assert lexicon_score("") == 0.0


def test_negation_flips_polarity():
    assert lexicon_score("not good") < 0
    assert lexicon_score("not noisy at all") > 0


def test_labels():
    assert label(0.6) == "positive"
    assert label(-0.6) == "negative"
    assert label(0.1) == "neutral"


def test_score_batch_lexicon_backend():
    assert score_batch(["great", "bad", ""], backend="lexicon") == [lexicon_score("great"), lexicon_score("bad"), 0.0]


def test_split_sentences():
    assert split_sentences("Great fan. Remote broke! Ok") == ["Great fan.", "Remote broke!", "Ok"]


def test_lexicon_aspects_finds_aspect_and_polarity():
    hits = {h["aspect"]: h["polarity"] for h in lexicon_aspects("Super silent motor. The remote app is useless.")}
    assert hits["noise"] == "positive"
    assert hits["smart_features"] == "negative"


def test_aspects_are_attributed_per_sentence():
    matcher = BrandMatcher([Brand("Atomberg", primary=True), Brand("Havells")])
    rows = brand_aspect_table(["Atomberg is silent and efficient. Havells is noisy."], matcher, backend="lexicon")
    by_key = {(r["brand"], r["aspect"]): r for r in rows}
    assert by_key[("Atomberg", "noise")]["positive"] == 1
    assert by_key[("Havells", "noise")]["negative"] == 1
    assert ("Havells", "energy") not in by_key


def test_auto_backend_falls_back_when_pyabsa_missing():
    try:
        import pyabsa  # noqa: F401
    except ImportError:
        assert absa_backend("auto") == "lexicon"


def test_query_terms_and_aspect_nouns_are_neutral():
    # Every crawled item contains the query "smart fan"; it must not bias sentiment.
    assert lexicon_score("smart fan") == 0.0
    assert lexicon_score("low noise fan") == 0.0
