"""Unit tests for the dimension-normalizer soft whitelist.

The normalizer collapses known synonyms onto canonical labels while
preserving Sonnet's freedom to coin new dimension names as the system
grows. These tests lock both behaviours: alias-map hits collapse, unknown
inputs pass through.
"""

from __future__ import annotations

from kai_memory.artifacts._dimension_normalizer import (
    ALIAS_MAP,
    CANONICAL_DIMENSIONS,
    canonical_list_for_prompt,
    normalize,
)


def test_canonical_passes_through_unchanged() -> None:
    assert normalize("context_discipline") == "context_discipline"
    assert normalize("honesty_under_pressure") == "honesty_under_pressure"


def test_alias_collapses_to_canonical() -> None:
    assert normalize("context_isolation_discipline") == "context_discipline"
    assert normalize("context_bleed_control") == "context_discipline"
    assert normalize("honesty_under_social_pressure") == "honesty_under_pressure"
    assert normalize("structured_synthesis") == "proactive_synthesis"


def test_unknown_dimension_passes_through() -> None:
    """Soft whitelist: new dimensions Sonnet coins must survive intact."""
    novel = "some_brand_new_axis_not_seen_before"
    assert normalize(novel) == novel


def test_uppercase_input_lowercases() -> None:
    assert normalize("Context_Discipline") == "context_discipline"
    assert normalize("HONESTY_UNDER_PRESSURE") == "honesty_under_pressure"


def test_whitespace_stripped() -> None:
    assert normalize("  context_discipline  ") == "context_discipline"


def test_empty_input_returns_empty() -> None:
    assert normalize("") == ""
    assert normalize("   ") == ""


def test_suffix_stripping_maps_to_canonical() -> None:
    """Decorative tails like ``_failure_mode`` collapse to the bare canonical."""
    # "compression_failure" is itself in ALIAS_MAP; check that the suffix
    # path handles a synthetic case where only the base matches.
    assert normalize("context_discipline_failure_mode") == "context_discipline"
    assert normalize("context_discipline_in_practice") == "context_discipline"


def test_suffix_strip_only_applies_when_stripped_form_is_known() -> None:
    """Suffix-strip is a conservative fallback: it only fires if stripping
    yields a canonical or aliased name. Bare-unknown coinages survive
    intact so Sonnet retains the power to define new dimensions whose
    name happens to end with a known suffix.
    """
    # "totally_new_axis_failure_mode" -> stripped to "totally_new_axis",
    # not canonical, not aliased → return original unchanged.
    assert normalize("totally_new_axis_failure_mode") == "totally_new_axis_failure_mode"


def test_alias_map_targets_are_all_canonical() -> None:
    """Every alias must point at a canonical name — otherwise normalize()
    could return a non-canonical label and break downstream DISTINCT ON.
    """
    bad = {dst for dst in ALIAS_MAP.values() if dst not in CANONICAL_DIMENSIONS}
    assert not bad, f"alias targets not in CANONICAL_DIMENSIONS: {bad}"


def test_canonical_list_for_prompt_is_sorted_csv() -> None:
    rendered = canonical_list_for_prompt()
    parts = [p.strip() for p in rendered.split(",")]
    assert parts == sorted(parts)
    assert set(parts) == set(CANONICAL_DIMENSIONS)


def test_no_alias_collides_with_canonical() -> None:
    """An alias source MUST NOT also be a canonical name — otherwise the
    canonical short-circuit would beat the alias map and the entry would
    be dead code.
    """
    collisions = set(ALIAS_MAP) & CANONICAL_DIMENSIONS
    assert not collisions, f"aliases shadowed by canonical names: {collisions}"
