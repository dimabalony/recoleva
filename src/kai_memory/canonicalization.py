"""Cross-script (Cyrillic↔Latin) entity-name canonicalization.

Sub-design C deferred-cluster (design §5, gated by the PRECISION-gated
dedup-precision metric, design §3 Instrument 2).

The resolver (`cognitive_memory.find_or_create_entity`) historically merged
Latin-script name variants but left cross-script Cyrillic forms as separate
canonical entities, because `unidecode` is a 1:1 transliteration whose output
diverges from the Latin spelling users actually type (soft-sign apostrophes,
iotated-vowel scheme, final й/и vs y). `translit_key` folds that systematic
scheme variance into a single comparable key so `Миша`/`Misha`, `Катя`/`Katya`,
`Тверь`/`Tver` collapse — while keeping genuinely distinct names apart.

**PRECISION-FIRST / UNDER-MERGE.** A false merge collapsing two distinct people
is the catastrophic error. This module therefore handles ONLY clean
transliteration. It deliberately does NOT touch:

- diminutive↔formal nicknames (`Миша`↔`Михаил`) — a name-equivalence problem,
- spelling/ending variants (`Петрау`↔`Petrov`, `-ау`↔`-ов`) — fuzzy,

both of which carry false-merge risk. Those stay out of scope; residual messy
forks are resolved by human-reviewed data cleanup, never auto-merged here.
"""

from __future__ import annotations

import re
import unicodedata

from unidecode import unidecode

# Iotated-vowel scheme folding: user-typed `ya/yu/yo/ye` vs unidecode's
# `ia/iu/io/ie` for я/ю/ё/е. Fold the `y`-form onto the `i`-form so both spell
# the same key. Applied longest-first is unnecessary (all 2-char, disjoint).
_IOTATION_FOLD: tuple[tuple[str, str], ...] = (
    ("yo", "io"),
    ("yu", "iu"),
    ("ya", "ia"),
    ("ye", "ie"),
)

# Strip everything that isn't a lowercase latin letter, digit, or single space.
# Critically this drops the apostrophe `unidecode` emits for the soft/hard sign
# (ь/ъ), e.g. "Tver'" → "tver".
_NON_KEY_CHARS = re.compile(r"[^a-z0-9 ]+")


def translit_key(name: str) -> str:
    """Return a cross-script canonical key for an entity name.

    Equal keys ⇒ the same clean transliteration. Distinct names yield distinct
    keys. An empty/garbage input yields ``""`` (which never merges — see
    `same_canonical_entity`).

    Steps: unidecode → NFKC + casefold → drop non-key chars (soft-sign
    apostrophes, punctuation) → fold iotated-vowel scheme variance → fold a
    token-final ``y`` to ``i`` (й/и romanize to ``i``; users often type ``y``).
    """
    ascii_form = unidecode(name)
    folded = unicodedata.normalize("NFKC", ascii_form).casefold()
    folded = _NON_KEY_CHARS.sub("", folded)
    for y_form, i_form in _IOTATION_FOLD:
        folded = folded.replace(y_form, i_form)
    tokens = [(t[:-1] + "i") if t.endswith("y") else t for t in folded.split()]
    return " ".join(tokens).strip()


def same_canonical_entity(
    name_a: str,
    name_b: str,
    type_a: str | None = None,
    type_b: str | None = None,
) -> bool:
    """Whether two surfaces denote the same canonical entity.

    True iff their `translit_key`s are equal AND non-empty AND their entity
    types are compatible. Type-keying makes `agent`-Kai never merge into
    `person`-Kai. When either type is unknown (``None``) the types are treated
    as compatible — the key-equality still gates the decision.
    """
    if type_a is not None and type_b is not None and type_a != type_b:
        return False
    key_a = translit_key(name_a)
    if not key_a:
        return False
    return key_a == translit_key(name_b)
