"""Phase B grounding v2 — encode weak predicates + curator calibration.

Defect 1: the grounding rule binary-suppressed discussed/researched topics,
discarding the true weak signal (weeks of research became transient noise).
Defect 2: the curation prompt had no calibration constraint, upgrading
research+course facts into identity/skill claims ("advanced motorcyclist").

Fix: person `interests` slot (weak predicate encoded in the VALUE), grounding
rule extended to route the signal there instead of dropping it, ONE curator
calibration sentence. Capture is liberal by design — the extractor sees one
conversation and cannot judge recurrence; decay/reinforcement + the MULTI cap
+ salience ranking decide persistence (the junk gate).
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from kai_memory.extraction.memory_extraction import _validate_and_default_fact_slots
from kai_memory.durable_schema import (
    DURABLE_ATTRIBUTE_SCHEMA,
    SLOT_DURABILITY_RUBRIC,
    Cardinality,
    render_schema_for_prompt,
    schema_keys,
)
from kai_memory.slot_profile import aggregate_slot_profile
from kai_memory.slot_summarizer import _CURATION_PROMPT

# ── the interests slot ────────────────────────────────────────────────────────


def test_person_gains_interests_slot():
    assert "interests" in schema_keys("person")
    slot = next(s for s in DURABLE_ATTRIBUTE_SCHEMA["person"] if s.key == "interests")
    assert slot.cardinality is Cardinality.MULTI
    assert slot.importance == 5
    desc = " ".join(slot.description.split())
    assert "weak" in desc.lower()
    assert "never as a skill, ownership, or identity claim" in desc
    # capture-liberally by design: NO recurrence gating (the extractor sees one
    # conversation; decay/reinforcement decide persistence)
    assert "repeatedly" not in desc and "recurring" not in desc
    assert "user" not in desc.lower()  # universal phrasing


def test_interests_surfaces_in_prompt_catalog():
    catalog = render_schema_for_prompt()
    assert "- interests:" in catalog


def test_validator_accepts_interests_for_person():
    facts = [
        {
            "subject_id": "p1",
            "slot": "interests",
            "durability": "durable",
            "field": "interest",
            "value": "researching motorcycles",
        }
    ]
    _validate_and_default_fact_slots(facts, [SimpleNamespace(id="p1", entity_type="person")])
    assert facts[0]["slot"] == "interests"


# ── grounding rule: encode, don't suppress ───────────────────────────────────


def test_grounding_rule_encodes_weak_predicates():
    rubric = " ".join(SLOT_DURABILITY_RUBRIC.split())
    # fabrication guard intact
    assert "record only attributes the subject actually HAS or IS" in rubric
    assert "Do NOT record things the subject merely discussed" in rubric
    # the weak signal is ROUTED, not dropped
    assert "Record active research or interest AS the weak predicate it is" in rubric
    assert "never as an ownership, skill, or identity fact" in rubric
    # capture-liberally: no recurrence gating in the rule
    assert "recurring" not in rubric and "repeatedly explores" not in rubric


# ── curator calibration ──────────────────────────────────────────────────────


def test_curation_prompt_has_one_calibration_sentence():
    assert (
        "State only what the facts support: never upgrade weak predicates "
        "(interest, research, considering) into identity, skill, or ownership claims."
    ) in _CURATION_PROMPT
    # ONE principled sentence, not a rule-list
    assert _CURATION_PROMPT.count("never upgrade weak predicates") == 1


# ── motorcyclist regression + junk gate (aggregation level) ──────────────────


def _interest_rows(entity_id, values):
    now = datetime.now(UTC)
    return [
        {
            "trace_id": f"t-{i}",
            "created_at": now,
            "salience": 0.5,
            "significance": 3,
            "facts": [
                {
                    "subject_id": entity_id,
                    "slot": "interests",
                    "field": "interest",
                    "value": v,
                    "polarity": "is",
                    "durability": "durable",
                }
            ],
        }
        for i, v in enumerate(values)
    ]


def test_weak_predicate_facts_land_on_card_as_interests():
    # The motorcyclist class: research facts must SURFACE (as interest), and at
    # this layer nothing can inflate them into a skill/identity field — the slot
    # IS the epistemic label; the curator's inflation path is prompt-pinned above.
    rows = _interest_rows("p1", ["researching motorcycles", "interested in track days"])
    profile = aggregate_slot_profile(rows, "p1", "person")
    card = profile.canonical_fields["interests"]
    assert "researching motorcycles" in card and "interested in track days" in card


def test_interests_junk_gate_caps_accumulation():
    # Capture-liberally is safe because the MULTI cap bounds the card: 12
    # distinct interests in → at most multi_cap (8) retained, salience-ranked.
    rows = _interest_rows("p1", [f"interest {i}" for i in range(12)])
    profile = aggregate_slot_profile(rows, "p1", "person")
    kept = profile.canonical_fields["interests"].split(", ")
    assert len(kept) <= 8
