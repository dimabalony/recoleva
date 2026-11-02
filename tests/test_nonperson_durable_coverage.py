"""Non-person durable-fact extraction coverage (task: close the Sonnet-fallback gap).

Diagnosis (2026-07-02): non-person entities ship Sonnet-fallback cards because the
extraction funnel starves them — the prompt focus rule named only people, every
durability exemplar was person-flavored, and project/agent were fieldless +
hard-routed to off_schema. Downstream (attribution, validators, aggregator,
producer) was verified type-agnostic. These tests pin the minimal fix AND the
guard rails that must NOT move (person catalog, "uncertain → transient" junk
guard, grounding rule).
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from kai_memory.extraction.memory_extraction import EXTRACTION_PROMPT, _validate_and_default_fact_slots
from kai_memory.artifacts.producers.entity_schema import _MIN_ATTRIBUTED_FIELDS
from kai_memory.durable_schema import (
    DURABLE_ANY_SUBJECT_SLOTS,
    DURABLE_ATTRIBUTE_SCHEMA,
    DURABLE_SINGLE_SLOTS,
    DURABLE_SLOT_SET,
    FIRST_PARTY_DURABLE_SLOTS,
    SLOT_DURABILITY_RUBRIC,
    Cardinality,
    render_schema_for_prompt,
    schema_keys,
)
from kai_memory.slot_profile import aggregate_slot_profile

# ── new generic slot groups ───────────────────────────────────────────────────


def test_project_has_generic_durable_slots():
    keys = schema_keys("project")
    assert {"purpose", "status", "roles"} <= keys
    roles = next(s for s in DURABLE_ATTRIBUTE_SCHEMA["project"] if s.key == "roles")
    assert roles.cardinality is Cardinality.MULTI  # several people can hold roles


def test_agent_has_generic_durable_slots():
    keys = schema_keys("agent")
    assert {"purpose", "capabilities"} <= keys
    caps = next(s for s in DURABLE_ATTRIBUTE_SCHEMA["agent"] if s.key == "capabilities")
    assert caps.cardinality is Cardinality.MULTI


def test_place_gains_significance():
    assert "significance" in schema_keys("place")
    assert {"region", "place_type"} <= schema_keys("place")  # existing slots untouched


def test_remaining_types_stay_fieldless():
    for etype in ("topic", "object", "event", "group"):
        assert DURABLE_ATTRIBUTE_SCHEMA[etype] == []


def test_schema_prompt_catalog_includes_project_and_agent():
    catalog = render_schema_for_prompt()
    assert "project:" in catalog and "agent:" in catalog


# ── rubric: coverage added, guard rails pinned ───────────────────────────────


def test_rubric_no_catalog_list_drops_project_and_agent():
    # project/agent now have catalogs; the hard off_schema routing must not name them.
    assert "(topic, object, event, group)" in SLOT_DURABILITY_RUBRIC
    assert "(topic, project, object, event, group, agent)" not in SLOT_DURABILITY_RUBRIC


def test_rubric_gains_generic_nonperson_durable_exemplars():
    rubric = " ".join(SLOT_DURABILITY_RUBRIC.split())
    assert "project" in rubric and "standing capabilit" in rubric
    # universal voice: relationships anchor on PEOPLE in the agent's world, not a
    # single principal ("the user") — the person belongs in the fact, not the field
    assert "people the agent knows" in rubric
    assert "a person's role" in rubric
    assert "people in the agent's life" in rubric
    assert "for the people it serves" in rubric
    assert "the user" not in rubric
    # generic — never personal names/examples
    for personal in ("Misha", "Kai", "Corfu", "Examplecorp", "Katya"):
        assert personal not in rubric


def test_rubric_standing_capability_carveout_is_principled():
    # An agent's STANDING capability is durable; run-level outcomes and internal
    # tool/feature inventories stay operational (no credential/inventory junk hole).
    assert "internal" in SLOT_DURABILITY_RUBRIC and "tool/feature" in SLOT_DURABILITY_RUBRIC


def test_rubric_uncertain_transient_junk_guard_untouched():
    assert 'When uncertain → "transient"' in SLOT_DURABILITY_RUBRIC


def test_rubric_grounding_rule_fabrication_guard_intact():
    # The junk-profiles NO-GO guard (2026-06-20): subjects only get what they HAVE or ARE.
    # Grounding v2 (2026-07-03) extended the rule to ENCODE weak predicates instead of
    # dropping them — the fabrication half is pinned here; the encode half is pinned in
    # test_grounding_v2.py. Whitespace-normalized (the rubric line-wraps).
    normalized = " ".join(SLOT_DURABILITY_RUBRIC.split())
    assert "record only attributes the subject actually HAS or IS" in normalized
    assert "never as an ownership, skill, or identity fact" in normalized


# ── person catalog snapshot guard (must NOT move in this change) ─────────────

_PERSON_SNAPSHOT = [
    ("full_name", Cardinality.SINGLE, 10),  # A3 (2026-07-20) — display-only identity slot (C-1)
    ("current_location", Cardinality.SINGLE, 9),
    ("key_relationships", Cardinality.MULTI, 9),
    ("chronic_health_conditions", Cardinality.MULTI, 8),
    ("allergies", Cardinality.MULTI, 8),
    ("date_of_birth", Cardinality.SINGLE, 7),
    ("occupation", Cardinality.SINGLE, 7),
    ("employer", Cardinality.SINGLE, 6),
    ("languages", Cardinality.MULTI, 6),
    ("citizenship", Cardinality.MULTI, 6),
    ("hometown", Cardinality.SINGLE, 5),
    ("stable_preferences", Cardinality.MULTI, 5),
    ("interests", Cardinality.MULTI, 5),  # grounding v2 (2026-07-03) — deliberate addition
    ("physical_attributes", Cardinality.MULTI, 4),
]


def test_new_slot_descriptions_never_anchor_a_single_principal():
    # No "owner"/"the user" hard-anchored in FIELD semantics — the person belongs
    # in the fact's subject/value. (org.relationship_to_user predates this change,
    # has prod data, and is explicitly out of scope — follow-up debt.)
    for etype in ("project", "agent", "place"):
        for slot in DURABLE_ATTRIBUTE_SCHEMA[etype]:
            assert "user" not in slot.description.lower(), (etype, slot.key)
            assert "user" not in slot.key, (etype, slot.key)


def test_person_catalog_unchanged_snapshot():
    actual = [(s.key, s.cardinality, s.importance) for s in DURABLE_ATTRIBUTE_SCHEMA["person"]]
    assert actual == _PERSON_SNAPSHOT


# ── durable classification sets (D-1 / P-1 / Part C, 2026-07-20) ────────────────


def test_full_name_slot_present_single_display_only():
    """A3: full_name is a SINGLE person slot ranked at the top (importance 10)."""
    fn = next(s for s in DURABLE_ATTRIBUTE_SCHEMA["person"] if s.key == "full_name")
    assert fn.cardinality == Cardinality.SINGLE
    assert fn.importance == 10
    assert "full_name" in schema_keys("person")


def test_durable_slot_set_is_any_subject_plus_first_party():
    """DURABLE_SLOT_SET is exactly the union; the two subsets are disjoint."""
    assert DURABLE_SLOT_SET == DURABLE_ANY_SUBJECT_SLOTS | FIRST_PARTY_DURABLE_SLOTS
    assert not (DURABLE_ANY_SUBJECT_SLOTS & FIRST_PARTY_DURABLE_SLOTS)
    # every durable slot is a real person schema key (precision guard: closed catalog)
    assert DURABLE_SLOT_SET <= schema_keys("person")


def test_first_party_gated_slots_are_family_health():
    """P-1 gates exactly the sensitive MULTI family/health slots — nothing else."""
    assert FIRST_PARTY_DURABLE_SLOTS == {"key_relationships", "chronic_health_conditions", "allergies"}


def test_soft_person_slots_are_not_durable():
    """stable_preferences / interests / physical_attributes stay transient (flooding risk)."""
    assert {"stable_preferences", "interests", "physical_attributes"}.isdisjoint(DURABLE_SLOT_SET)


def test_durable_single_slots_are_single_cardinality_subset():
    """Part C acts only on SINGLE durable slots; derivation matches the schema cardinality."""
    person_single = {s.key for s in DURABLE_ATTRIBUTE_SCHEMA["person"] if s.cardinality == Cardinality.SINGLE}
    assert DURABLE_SINGLE_SLOTS == DURABLE_SLOT_SET & person_single
    assert DURABLE_SINGLE_SLOTS == {
        "full_name",
        "current_location",
        "date_of_birth",
        "occupation",
        "employer",
        "hometown",
    }
    # the MULTI durable slots (incl. family/health) are never single-supersede targets
    assert FIRST_PARTY_DURABLE_SLOTS.isdisjoint(DURABLE_SINGLE_SLOTS)


def test_organization_catalog_snapshot():
    # relationship_to_user → relationship (user gate, 2026-07-02): the one legacy
    # single-principal field NAME, migrated data-safely by alembic 092. Other org
    # slots unchanged.
    actual = [(s.key, s.cardinality) for s in DURABLE_ATTRIBUTE_SCHEMA["organization"]]
    assert actual == [
        ("relationship", Cardinality.SINGLE),
        ("location", Cardinality.SINGLE),
        ("industry", Cardinality.SINGLE),
        ("services_or_products", Cardinality.MULTI),
    ]


def test_no_slot_key_anchors_a_single_principal():
    for etype, slots in DURABLE_ATTRIBUTE_SCHEMA.items():
        for slot in slots:
            assert "user" not in slot.key, (etype, slot.key)


def test_migrated_and_new_relationship_facts_share_one_slot():
    # The point of migrating (not aliasing): old rewritten facts and new facts
    # aggregate, dedup, and tombstone under ONE field name.
    now = datetime.now(UTC)
    rows = [
        {
            "trace_id": f"t-{i}",
            "created_at": now,
            "salience": 0.6,
            "significance": 3,
            "facts": [f],
        }
        for i, f in enumerate(
            [
                {
                    "subject_id": "org-1",
                    "slot": "relationship",
                    "field": "relationship",
                    "value": "employer",
                    "polarity": "is",
                    "durability": "durable",
                },  # migrated legacy fact
                {
                    "subject_id": "org-1",
                    "slot": "relationship",
                    "field": "relationship",
                    "value": "employer",
                    "polarity": "is_not",
                    "durability": "durable",
                },  # tombstone still matches
            ]
        )
    ]
    profile = aggregate_slot_profile(rows, "org-1", "organization")
    assert "relationship" not in profile.canonical_fields  # is_not erased the value — continuity works


def test_unmigrated_legacy_slot_no_longer_types():
    # Shows WHY 092 must rewrite trace facts: a leftover legacy slot name would
    # stop matching the typed catalog and degrade to off-schema keying.
    now = datetime.now(UTC)
    rows = [
        {
            "trace_id": "t-0",
            "created_at": now,
            "salience": 0.6,
            "significance": 3,
            "facts": [
                {
                    "subject_id": "org-1",
                    "slot": "relationship_to_user",
                    "field": "relationship_to_user",
                    "value": "employer",
                    "polarity": "is",
                    "durability": "durable",
                }
            ],
        }
    ]
    profile = aggregate_slot_profile(rows, "org-1", "organization")
    assert "relationship" not in profile.canonical_fields


# ── validators accept the new slots (and still degrade cross-typed ones) ─────


def _entity(eid: str, etype: str):
    return SimpleNamespace(id=eid, entity_type=etype)


def test_validator_accepts_new_project_slot():
    facts = [{"subject_id": "p1", "slot": "roles", "durability": "durable", "field": "role", "value": "alex — owner"}]
    _validate_and_default_fact_slots(facts, [_entity("p1", "project")])
    assert facts[0]["slot"] == "roles"


def test_validator_accepts_new_agent_slot():
    facts = [{"subject_id": "a1", "slot": "capabilities", "durability": "durable", "field": "cap", "value": "x"}]
    _validate_and_default_fact_slots(facts, [_entity("a1", "agent")])
    assert facts[0]["slot"] == "capabilities"


def test_validator_still_degrades_cross_typed_slot():
    # "purpose" is not a topic slot — degrades to off_schema, never dropped.
    facts = [{"subject_id": "t1", "slot": "purpose", "durability": "durable", "field": "purpose", "value": "x"}]
    _validate_and_default_fact_slots(facts, [_entity("t1", "topic")])
    assert facts[0]["slot"] == "off_schema"


# ── extraction prompt focus rule ─────────────────────────────────────────────


def test_prompt_focus_covers_tracked_entities_not_just_people():
    normalized = " ".join(EXTRACTION_PROMPT.split())  # the rule line wraps
    assert "facts about people and other tracked entities" in normalized
    assert "(places, organizations, projects, agents)" in normalized


def test_prompt_carries_no_personal_examples_in_new_text():
    focus_line = next(ln for ln in EXTRACTION_PROMPT.splitlines() if "tracked entities" in ln)
    for personal in ("Misha", "Kai", "Corfu", "Examplecorp"):
        assert personal not in focus_line


# ── Stage 3: the thin gate actually flips for a non-person entity ────────────


def _rows_for(entity_id: str, slot_facts: list[tuple[str, str, str]]):
    now = datetime.now(UTC)
    return [
        {
            "trace_id": f"t-{i}",
            "created_at": now,
            "salience": 0.6,
            "significance": 3,
            "facts": [
                {
                    "subject_id": entity_id,
                    "slot": slot,
                    "field": field,
                    "value": value,
                    "polarity": "is",
                    "durability": "durable",
                }
            ],
        }
        for i, (slot, field, value) in enumerate(slot_facts)
    ]


def test_project_with_three_attributed_fields_passes_thin_gate():
    """≥3 populated slots ⇒ the deterministic card ships (no Sonnet fallback)."""
    rows = _rows_for(
        "proj-1",
        [
            ("purpose", "purpose", "agent-first flight search API"),
            ("status", "status", "active"),
            ("roles", "role", "alex — owner"),
        ],
    )
    profile = aggregate_slot_profile(rows, "proj-1", "project")
    assert len(profile.canonical_fields) >= _MIN_ATTRIBUTED_FIELDS


def test_agent_capabilities_accumulate_as_multi():
    rows = _rows_for(
        "agent-1",
        [
            ("capabilities", "capability", "schedules calendar events"),
            ("capabilities", "capability", "sends messages"),
            ("purpose", "purpose", "personal assistant"),
        ],
    )
    profile = aggregate_slot_profile(rows, "agent-1", "agent")
    caps = profile.canonical_fields["capabilities"]  # MULTI accumulates distinct values
    assert "schedules calendar events" in caps and "sends messages" in caps
