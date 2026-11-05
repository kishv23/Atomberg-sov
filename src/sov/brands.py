import re
from typing import Dict, List, Pattern

SPECIAL_HYPHEN = r"[\s\-]?"

def _normalize_brand(brand: str) -> str:
    return brand.strip()

def _brand_to_pattern(brand: str) -> str:
    tokens = re.split(r"\s+", brand.strip())
    joined = SPECIAL_HYPHEN.join(map(re.escape, tokens))
    return rf"(?<![A-Za-z0-9]){joined}(?![A-Za-z0-9])"

def compile_brand_regexes(primary: List[str], competitors: List[str]) -> Dict[str, Pattern]:
    all_brands = list(dict.fromkeys([_normalize_brand(b) for b in primary + competitors]))
    return {b: re.compile(_brand_to_pattern(b), re.IGNORECASE) for b in all_brands}

def count_mentions(text: str, brand_patterns: Dict[str, Pattern]) -> Dict[str, int]:
    text = text or ""
    return {brand: len(pat.findall(text)) for brand, pat in brand_patterns.items()}
