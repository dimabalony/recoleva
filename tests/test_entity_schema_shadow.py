"""INC 3 §3.4 — EntitySchema slot aggregator (producer wiring).

Exercises ``_slot_profile``'s rows→slot-aggregate→relationship-fold→would_fallback
path with a mocked session (no DB), plus the flag-gated shadow log's error safety.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kai_memory.artifacts.producers.entity_schema import EntitySchemaProducer

pytestmark = pytest.mark.asyncio


def _entity(entity_type: str = "person"):
    e = MagicMock()
    e.id = uuid.uuid4()
    e.entity_type = entity_type
    return e


async def test_slot_aggregator_attributes_and_flags_fallback():
    entity = _entity()
    eid = str(entity.id)
    row = SimpleNamespace(
        trace_id="t1",
        created_at=datetime(2026, 6, 1, tzinfo=UTC),
        salience=0.5,
        significance=3,
        facts=[
            {"slot": "occupation", "value": "founder", "polarity": "is", "durability": "durable", "subject_id": eid},
            # off-schema (operational) → never on the card
            {"slot": "off_schema", "value": "1", "polarity": "is", "durability": "operational", "subject_id": eid},
            # attributed to a DIFFERENT entity → dropped
            {
                "slot": "current_location",
                "value": "Corfu",
                "polarity": "is",
                "durability": "durable",
                "subject_id": str(uuid.uuid4()),
            },
        ],
    )
    session = AsyncMock()
    session.execute = AsyncMock(return_value=[row])

    profile, would_fallback = await EntitySchemaProducer()._slot_profile(session, entity, [])

    assert profile.canonical_fields == {"occupation": "founder"}  # off-schema + other-entity dropped
    assert profile.canonical_slot_values["occupation"][0]["value"] == "founder"
    assert would_fallback is True  # 1 populated slot < 3 (min)


async def test_slot_relationship_fold_fills_absent_slot():
    entity = _entity()
    target_id = uuid.uuid4()
    # no fact-bearing traces → empty rows; the works_at relationship provides employer.
    session = AsyncMock()
    name_row = SimpleNamespace(id=target_id, name="Examplecorp")
    session.execute = AsyncMock(side_effect=[[], [name_row]])  # facts SQL, then names

    rel = SimpleNamespace(predicate="works_at", target_entity_id=target_id, confidence=0.9)
    profile, would_fallback = await EntitySchemaProducer()._slot_profile(session, entity, [rel])

    assert profile.canonical_fields == {"employer": "Examplecorp"}  # §3.4 fold
    assert would_fallback is True  # 0 fact-derived slots


async def test_deterministic_card_swallows_errors_and_returns_none():
    """_deterministic_card swallows errors (never breaks the producer) and returns
    None so the caller falls back to Sonnet."""
    entity = _entity()
    session = AsyncMock()
    session.execute = AsyncMock(side_effect=RuntimeError("boom"))
    # must not raise despite the failing query, and must signal fallback
    card = await EntitySchemaProducer()._deterministic_card(session, entity, {"role": "x"}, [], log_shadow=True)
    assert card is None


async def test_shadow_path_invokes_profile_curation(monkeypatch):
    """The shadow path runs the Haiku whole-entity curator on the aggregated profile
    (after aggregate_slot_profile) before logging the diff."""
    entity = _entity()
    eid = str(entity.id)
    row = SimpleNamespace(
        trace_id="t1",
        created_at=datetime(2026, 6, 1, tzinfo=UTC),
        salience=0.5,
        significance=3,
        facts=[
            {"slot": "languages", "value": "English", "polarity": "is", "durability": "durable", "subject_id": eid},
        ],
    )
    session = AsyncMock()
    session.execute = AsyncMock(return_value=[row])

    seen = {}

    async def fake_curate(profile, *, entity_name=None, provider=None):
        seen["called"] = True
        seen["entity_name"] = entity_name
        return profile

    monkeypatch.setattr("kai_memory.artifacts.producers.entity_schema.curate_profile", fake_curate)
    await EntitySchemaProducer()._deterministic_card(session, entity, {"role": "x"}, [], log_shadow=True)
    assert seen.get("called") is True  # the curator was invoked on the aggregated profile
    assert seen.get("entity_name") is entity.name  # with this entity


async def test_shadow_diff_carries_offschema_counts(monkeypatch, caplog):
    """§12 P2-C — the shadow_diff log carries off_schema_durable_count + retained_count."""
    import logging

    entity = _entity()
    eid = str(entity.id)
    row = SimpleNamespace(
        trace_id="t1",
        created_at=datetime(2026, 6, 1, tzinfo=UTC),
        salience=0.5,
        significance=3,
        facts=[
            {
                "slot": "off_schema",
                "value": "SampleGym",
                "field": "gym",
                "polarity": "is",
                "durability": "durable",
                "subject_id": eid,
            },
        ],
    )
    session = AsyncMock()
    session.execute = AsyncMock(return_value=[row])

    async def fake_curate(profile, *, entity_name=None, provider=None):
        # Mimic the real curate_profile: a FRESH SlotProfile (off-schema counts reset to
        # their 0 default), same substrate — so this test catches a regression where the
        # counts get captured AFTER curation instead of before.
        from kai_memory.slot_profile import SlotProfile

        return SlotProfile(
            canonical_fields=dict(profile.canonical_fields),
            canonical_slot_values=profile.canonical_slot_values,
        )

    monkeypatch.setattr("kai_memory.artifacts.producers.entity_schema.curate_profile", fake_curate)
    with caplog.at_level(logging.INFO):
        await EntitySchemaProducer()._deterministic_card(session, entity, {}, [], log_shadow=True)
    # the single off_schema durable fact (gym: SampleGym) → counts 1; they must be the
    # PRE-curation values, not the rebuilt profile's reset-to-0 defaults.
    assert '"off_schema_durable_count": 1' in caplog.text
    assert '"off_schema_retained_count": 1' in caplog.text


async def test_deterministic_card_returns_curated_for_rich_profile(monkeypatch):
    # >=3 attributed slots → not would_fallback → returns the curated card for the live cutover.
    entity = _entity()
    eid = str(entity.id)
    row = SimpleNamespace(
        trace_id="t1",
        created_at=datetime(2026, 6, 1, tzinfo=UTC),
        salience=0.5,
        significance=3,
        facts=[
            {"slot": "occupation", "value": "founder", "polarity": "is", "durability": "durable", "subject_id": eid},
            {"slot": "current_location", "value": "Corfu", "polarity": "is", "durability": "durable", "subject_id": eid},
            {"slot": "citizenship", "value": "RU", "polarity": "is", "durability": "durable", "subject_id": eid},
        ],
    )
    session = AsyncMock()
    session.execute = AsyncMock(return_value=[row])

    async def fake_curate(profile, *, entity_name=None, provider=None):
        from kai_memory.slot_profile import SlotProfile

        return SlotProfile(
            canonical_fields={"summary": "Founder in Corfu, Russian"},
            canonical_slot_values=profile.canonical_slot_values,
        )

    monkeypatch.setattr("kai_memory.artifacts.producers.entity_schema.curate_profile", fake_curate)
    card = await EntitySchemaProducer()._deterministic_card(session, entity, {"role": "x"}, [], log_shadow=False)
    assert card == {"summary": "Founder in Corfu, Russian"}


async def test_deterministic_card_returns_none_for_thin_profile(monkeypatch):
    # <3 attributed slots → would_fallback → None so the caller ships Sonnet (no thin card).
    entity = _entity()
    eid = str(entity.id)
    row = SimpleNamespace(
        trace_id="t1",
        created_at=datetime(2026, 6, 1, tzinfo=UTC),
        salience=0.5,
        significance=3,
        facts=[
            {"slot": "occupation", "value": "founder", "polarity": "is", "durability": "durable", "subject_id": eid},
        ],
    )
    session = AsyncMock()
    session.execute = AsyncMock(return_value=[row])

    async def fake_curate(profile, *, entity_name=None, provider=None):
        return profile  # passthrough — would_fallback is already True on a single slot

    monkeypatch.setattr("kai_memory.artifacts.producers.entity_schema.curate_profile", fake_curate)
    card = await EntitySchemaProducer()._deterministic_card(session, entity, {"role": "x"}, [], log_shadow=False)
    assert card is None
