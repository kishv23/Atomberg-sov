import pytest

from sov.brands import Brand, BrandMatcher, brands_from_config


@pytest.fixture
def matcher():
    return BrandMatcher(
        [
            Brand("Atomberg", ("Renesa", "Efficio", "Atomberg Renesa"), primary=True),
            Brand("V-Guard"),
            Brand("Havells"),
        ]
    )


def test_product_mention_counts_once_for_parent(matcher):
    assert matcher.count("Atomberg Renesa review")["Atomberg"] == 1


def test_bare_product_name_rolls_up(matcher):
    assert matcher.count("Renesa vs Efficio")["Atomberg"] == 2


def test_separator_variants(matcher):
    counts = matcher.count("V Guard, VGuard and v-guard fans")
    assert counts["V-Guard"] == 3


def test_word_boundaries(matcher):
    assert matcher.count("Havellsx fan")["Havells"] == 0


def test_handles_missing_text(matcher):
    assert sum(matcher.count(None).values()) == 0


def test_duplicate_alias_is_rejected():
    with pytest.raises(ValueError, match="claimed by both"):
        BrandMatcher([Brand("A", ("X",)), Brand("B", ("x",))])


def test_legacy_flat_config_folds_products_into_first_brand():
    cfg = {"brands": {"primary": ["Atomberg", "Atomberg Renesa"], "competitors": ["Orient"]}}
    brands = brands_from_config(cfg)
    assert brands[0].name == "Atomberg" and brands[0].primary
    assert "Atomberg Renesa" in brands[0].aliases
    assert [b.name for b in brands] == ["Atomberg", "Orient"]


def test_structured_config():
    cfg = {"brands": {"primary": {"name": "Atomberg", "aliases": ["Renesa"]}, "competitors": [{"name": "Usha"}]}}
    assert [b.name for b in brands_from_config(cfg)] == ["Atomberg", "Usha"]


def test_brand_followed_by_product_is_one_mention_without_explicit_alias():
    m = BrandMatcher([Brand("Atomberg", ("Renesa",), primary=True)])
    assert m.count("Atomberg Renesa and Atomberg-Renesa")["Atomberg"] == 2
