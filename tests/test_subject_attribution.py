"""Unit tests for INC 3 A3 subject attribution.

`resolve_subject_id` maps a fact's raw subject name to a canonical entity id by
reusing #201 cross-script canonicalization (`same_canonical_entity`) over the
trace's candidate entities — so `Миша` attributes to the canonical `Misha` entity
and a subject that matches no candidate stays unattributed (id=None). Shared by
the live L2 extraction path and the backfill tool.
"""

from __future__ import annotations

import uuid

from kai_memory.subject_attribution import resolve_subject_id


def _cand(name, etype="person"):
    return (uuid.uuid4(), name, etype)


def test_exact_name_match_returns_id():
    c = _cand("Misha Petrov")
    assert resolve_subject_id("Misha Petrov", [c]) == c[0]


def test_cross_script_subject_resolves_to_canonical_latin_id():
    misha = _cand("Misha")  # canonical Latin entity
    other = _cand("Katya")
    # Cyrillic subject must resolve to the Latin canonical via translit_key.
    assert resolve_subject_id("Миша", [other, misha]) == misha[0]


def test_no_matching_candidate_returns_none():
    assert resolve_subject_id("Maksim", [_cand("Misha"), _cand("Katya")]) is None


def test_empty_or_garbage_subject_returns_none():
    assert resolve_subject_id("", [_cand("Misha")]) is None
    assert resolve_subject_id("   ", [_cand("Misha")]) is None
    assert resolve_subject_id("Misha", []) is None


def test_picks_the_canonically_matching_candidate():
    katya = _cand("Katya")
    misha = _cand("Misha Petrov")
    # 'Миша Петров' (Cyrillic) → translit 'misha petrov' → matches misha.
    assert resolve_subject_id("Миша Петров", [katya, misha]) == misha[0]


# ── candidate expansion (name + aliases) + in-place fact annotation ─────────


class _Ent:
    def __init__(self, name, etype="person", aliases=None):
        self.id = uuid.uuid4()
        self.name = name
        self.entity_type = etype
        self.aliases = aliases or []


def test_candidates_expand_name_and_aliases():
    from kai_memory.subject_attribution import candidates_from_entities

    e = _Ent("Misha", aliases=["Миша", "Дзмитры"])
    cands = candidates_from_entities([e])
    names = {c[1] for c in cands}
    assert names == {"Misha", "Миша", "Дзмитры"}
    assert all(c[0] == e.id for c in cands)


def test_attribute_fact_subjects_stamps_subject_id_via_alias():
    from kai_memory.subject_attribution import attribute_fact_subjects

    misha = _Ent("Misha Petrov", aliases=["Миша"])
    katya = _Ent("Katya")
    facts = [
        {"subject": "Миша", "field": "role", "value": "founder", "polarity": "is"},
        {"subject": "Katya", "field": "city", "value": "Corfu", "polarity": "is"},
        {"subject": "Ghost", "field": "x", "value": "y", "polarity": "is"},  # unresolved
        {"field": "no-subject", "value": "z", "polarity": "is"},  # no subject key
    ]
    attribute_fact_subjects(facts, [misha, katya])
    assert facts[0]["subject_id"] == str(misha.id)  # via alias
    assert facts[1]["subject_id"] == str(katya.id)
    assert facts[2]["subject_id"] is None  # unresolved
    assert "subject_id" not in facts[3]  # untouched (no subject)
    # raw subject preserved
    assert facts[0]["subject"] == "Миша"


def test_ambiguous_canonical_match_returns_none():
    """Two DIFFERENT entities sharing the subject's canonical key → unattributed
    (precision-first; never order-dependently smear)."""
    a = (uuid.uuid4(), "Misha", "person")
    b = (uuid.uuid4(), "Миша", "person")  # same translit key 'misha', different id
    assert resolve_subject_id("Misha", [a, b]) is None
