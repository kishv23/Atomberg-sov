"""Brand mention detection.

Each brand has a canonical name plus aliases (product lines, spellings).
All aliases are compiled into one alternation ordered longest-first, so
"Atomberg Renesa" counts once for Atomberg instead of once for the product
and again for the parent brand.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Pattern

SEPARATOR = r"[\s\-]?"


@dataclass(frozen=True)
class Brand:
    name: str
    aliases: tuple = field(default_factory=tuple)
    primary: bool = False

    @property
    def terms(self) -> List[str]:
        return [self.name, *self.aliases]


def _term_to_pattern(term: str) -> str:
    tokens = [t for t in re.split(r"[\s\-]+", term.strip()) if t]
    return SEPARATOR.join(map(re.escape, tokens))


class BrandMatcher:
    def __init__(self, brands: Iterable[Brand]):
        self.brands: List[Brand] = list(brands)
        if not self.brands:
            raise ValueError("at least one brand is required")
        self._owner: Dict[str, str] = {}
        terms: List[str] = []
        for b in self.brands:
            for t in b.terms:
                key = t.lower()
                if key in self._owner and self._owner[key] != b.name:
                    raise ValueError(f"alias '{t}' is claimed by both {self._owner[key]} and {b.name}")
                self._owner[key] = b.name
                terms.append(t)
        terms.sort(key=len, reverse=True)
        alternation = "|".join(f"(?:{_term_to_pattern(t)})" for t in terms)
        self._pattern: Pattern = re.compile(rf"(?<![A-Za-z0-9])(?:{alternation})(?![A-Za-z0-9])", re.IGNORECASE)

    @property
    def names(self) -> List[str]:
        return [b.name for b in self.brands]

    @property
    def primary(self) -> List[str]:
        return [b.name for b in self.brands if b.primary]

    def _resolve(self, matched: str) -> str:
        normalised = re.sub(r"[\s\-]+", " ", matched.lower()).strip()
        for key, owner in self._owner.items():
            if re.sub(r"[\s\-]+", " ", key) == normalised:
                return owner
        # Separator variants such as "VGuard" for "V-Guard".
        compact = normalised.replace(" ", "")
        for key, owner in self._owner.items():
            if re.sub(r"[\s\-]+", "", key) == compact:
                return owner
        raise KeyError(matched)

    def count(self, text: str | None) -> Dict[str, int]:
        counts = {name: 0 for name in self.names}
        for m in self._pattern.finditer(text or ""):
            counts[self._resolve(m.group(0))] += 1
        return counts


def brands_from_config(cfg: dict) -> List[Brand]:
    """Accepts both the structured format and the legacy flat lists."""
    section = cfg["brands"]
    out: List[Brand] = []

    def parse(entry, primary: bool) -> Brand:
        if isinstance(entry, str):
            return Brand(entry, primary=primary)
        return Brand(entry["name"], tuple(entry.get("aliases", [])), primary=primary)

    primary = section.get("primary", [])
    if isinstance(primary, dict):
        primary = [primary]
    # Legacy: ["Atomberg", "Atomberg Renesa", ...] -> one brand with aliases.
    if primary and all(isinstance(p, str) for p in primary):
        head, *rest = primary
        out.append(Brand(head, tuple(rest), primary=True))
    else:
        out.extend(parse(p, True) for p in primary)
    out.extend(parse(c, False) for c in section.get("competitors", []))
    return out
