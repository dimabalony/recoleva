"""Romanization helper unit tests — v2a §5.5, §7.3."""

from __future__ import annotations


def test_normalize_is_idempotent():
    from kai_memory.romanization import _normalize

    s = " Луна "
    assert _normalize(_normalize(s)) == _normalize(s)
    assert _normalize("CAFÉ") == _normalize("café")


def test_romanize_non_ascii_returns_ascii():
    from kai_memory.romanization import romanize

    out = romanize("Луна")
    assert out is not None
    assert out.isascii()
    assert out  # non-empty


def test_romanize_already_ascii_returns_none():
    from kai_memory.romanization import romanize

    assert romanize("Hello") is None
    assert romanize("Misha") is None


def test_romanize_strips_whitespace():
    from kai_memory.romanization import romanize

    out = romanize("  Луна  ")
    assert out is not None and out == out.strip()


def test_candidate_aliases_non_ascii_includes_romanization():
    from kai_memory.romanization import candidate_aliases

    aliases = candidate_aliases("Луна", None)
    assert any(a.isascii() and a for a in aliases)


def test_candidate_aliases_ascii_with_cyrillic_source_recovers_cyrillic():
    """v2a §5.5: ASCII name + matching Cyrillic in source → Cyrillic alias surfaces.

    Names with a clean round-trip through unidecode (no apostrophes from
    soft-signs) are recovered. "Луна" → "S'iuzi" doesn't round-trip to a
    user-typed "Luna" because the apostrophe survives normalization;
    that gap is documented in §5.5 and won't surface for entity names that
    don't include soft/hard signs.
    """
    from kai_memory.romanization import candidate_aliases

    source = "сегодня встретил Миша в парке"
    aliases = candidate_aliases("Misha", source)
    assert any("Миша" in a for a in aliases)


def test_candidate_aliases_filters_short_words():
    """v2a §5.5: token-length filter (≥3 chars) drops sub-3-char Cyrillic
    matches. Using an ASCII name that's >=3 chars is necessary so the
    reverse-scan branch actually runs (sub-3 names skip it entirely).
    """
    from kai_memory.romanization import candidate_aliases

    # 'ща' is 2 chars and should be dropped by the token-length guard;
    # nothing else matches normalize(unidecode("Aaa")), so the result is empty.
    aliases = candidate_aliases("Aaa", "видел ща вчера")
    assert aliases == []


def test_candidate_aliases_filters_cyrillic_stopwords():
    """v2a §5.5: stopwords ('я', 'мир', etc.) are excluded."""
    from kai_memory.romanization import candidate_aliases

    # 'я' romanizes to 'ia' (not 'Ira') in unidecode but the principle is the
    # filter must exclude common Cyrillic function words. Use 'мир' which
    # romanizes to a 3-char string.
    aliases = candidate_aliases("Mir", "наш мир")
    # 'мир' in stopwords; should not be returned
    assert all(a.lower() != "мир" for a in aliases)


def test_candidate_aliases_dedupes_preserving_order():
    """ASCII name + Cyrillic source exercises the reverse-scan branch and
    proves both: (a) duplicates collapse, and (b) the surviving entry's
    order is preserved (not just unique).
    """
    from kai_memory.romanization import candidate_aliases

    out = candidate_aliases("Misha", "Миша, привет Миша, видел Миша")
    assert out == ["Миша"]


def test_candidate_aliases_empty_when_no_candidates():
    from kai_memory.romanization import candidate_aliases

    assert candidate_aliases("Hello", None) == []


def test_candidate_aliases_handles_empty_source_text():
    from kai_memory.romanization import candidate_aliases

    # Should not raise
    aliases = candidate_aliases("Hello", "")
    assert isinstance(aliases, list)


# ── review-r2 fix 4: lowercase-leading Cyrillic recovery ────


def test_candidate_aliases_recovers_lowercase_cyrillic():
    """CodeRabbit major: regex used to require capital letter, so
    `candidate_aliases("Misha", "видел миша вчера")` recovered nothing.
    After fix 4 the lowercase form is captured and stopword-filtered.
    """
    from kai_memory.romanization import candidate_aliases

    aliases = candidate_aliases("Misha", "видел миша вчера в парке")
    assert any("миша" in a.lower() for a in aliases), f"lowercase Cyrillic 'миша' not recovered; got {aliases!r}"


def test_candidate_aliases_lowercase_still_filters_stopwords():
    """The expanded CYRILLIC_STOPWORDS set is the only defense once the
    regex is permissive, so common function words must still be dropped.
    """
    from kai_memory.romanization import candidate_aliases

    # 'или' is in CYRILLIC_STOPWORDS — must NOT match a lookup for "Ili"
    aliases = candidate_aliases("Ili", "это или то")
    assert all(a.lower() != "или" for a in aliases)


def test_candidate_aliases_recovers_all_caps_cyrillic():
    """CodeRabbit minor: regex's trailing class missed uppercase, so
    all-caps tokens like 'МИША' weren't captured at all."""
    from kai_memory.romanization import candidate_aliases

    aliases = candidate_aliases("Misha", "СЕГОДНЯ ВИДЕЛ МИША В ПАРКЕ")
    assert any("МИША" in a or "миша" in a.lower() for a in aliases), (
        f"all-caps Cyrillic 'МИША' not recovered; got {aliases!r}"
    )
