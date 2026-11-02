"""Romanization + reverse-source-scan helpers for cross-lingual entity lookup.

v2a §5.5: alias expansion happens BEFORE find_or_create_entity queries the
DB, not as post-hoc repair. Bidirectional:

- Non-ASCII name → unidecode romanization candidate.
- ASCII name → scan source_text for Cyrillic words that round-trip to the
  same normalized form.

Stopwords + 3-char minimum + NFKC casefold normalization keep false
positives out of the alias graph.
"""

from __future__ import annotations

import re
import unicodedata

from unidecode import unidecode

# Cyrillic words (entity-name candidates).
#
# review-r2 fix 4 (2026-05-01): the first character class accepts both
# upper- AND lower-case Cyrillic — chat traffic frequently writes names
# uncapitalized ("видел миша вчера"), and requiring capitals lost half the
# reverse-source-scan coverage. Defensive-posture note: with this regex
# permissive, CYRILLIC_STOPWORDS is now the ONLY defense against pulling
# function words ("или" / "она" / "был" / ...) into the alias graph.
# That set was expanded from 16 → 41 in r1 fix 3 specifically so this
# shift is safe; future false-positive bugs ("X is recovering common
# word Y") are fixed by extending the stopword set, not by tightening
# the regex.
_CYRILLIC_WORD_RE = re.compile("[А-ЯЁЇІЄҐа-яёїієґ][А-ЯЁЇІЄҐа-яёїієґ̀-ͯ]+")

# Cyrillic function words that should never be entity-name candidates,
# whether for reverse-source-scan romanization OR for lowercase-token
# DB lookups in priming. Matched case-insensitively; stored lowercase.
CYRILLIC_STOPWORDS: frozenset[str] = frozenset(
    {
        # Pronouns
        "я",
        "ты",
        "он",
        "она",
        "оно",
        "мы",
        "вы",
        "они",
        "их",
        "его",
        "её",
        "ему",
        "ей",
        "им",
        "ими",
        "мне",
        "нам",
        "вам",
        "наш",
        "ваш",
        "свой",
        "тот",
        "та",
        "те",
        "это",
        "эта",
        "эти",
        "тут",
        "там",
        "здесь",
        # Conjunctions / particles / prepositions
        "и",
        "а",
        "но",
        "или",
        "что",
        "как",
        "так",
        "же",
        "ли",
        "бы",
        "уже",
        "ещё",
        "тоже",
        # Frequent verbs / interrogatives / common nouns that romanize
        # to short English-looking strings
        "да",
        "нет",
        "мир",
        "был",
        "была",
        "было",
        "были",
        "есть",
    }
)


def _normalize(s: str) -> str:
    """NFKC + casefold + strip — idempotent canonical form for alias matching."""
    return unicodedata.normalize("NFKC", s).casefold().strip()


def romanize(name: str) -> str | None:
    """Return romanized form if `name` contains non-ASCII; None otherwise.

    Whitespace is stripped from the output.
    """
    if name.isascii():
        return None
    r = unidecode(name).strip()
    if not r or r == name:
        return None
    return r


def candidate_aliases(name: str, source_text: str | None) -> list[str]:
    """Generate alias candidates BEFORE the entity DB lookup.

    Non-ASCII name → its romanized form (when distinct).
    ASCII name → Cyrillic tokens in source_text that romanize to the same
    normalized form. Stopwords and tokens shorter than 3 chars are dropped.

    Output is order-preserving deduped; never includes `name` itself.
    """
    aliases: list[str] = []

    romanized = romanize(name)
    if romanized:
        aliases.append(romanized)

    if source_text and name.isascii() and len(name) >= 3:
        normalized_target = _normalize(name)
        for token in _CYRILLIC_WORD_RE.findall(source_text):
            if len(token) < 3:
                continue
            # _normalize (NFKC + casefold + strip) so decomposed Cyrillic
            # forms ("е́" / "й") match the precomposed entries in
            # CYRILLIC_STOPWORDS. Without NFKC, raw .casefold() lets through
            # decomposed variants and weakens the only false-positive guard.
            if _normalize(token) in CYRILLIC_STOPWORDS:
                continue
            if _normalize(unidecode(token)) == normalized_target:
                aliases.append(token)

    seen: set[str] = set()
    out: list[str] = []
    for a in aliases:
        if a == name or a in seen:
            continue
        seen.add(a)
        out.append(a)
    return out
