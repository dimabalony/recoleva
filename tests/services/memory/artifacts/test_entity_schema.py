"""Unit tests for EntitySchemaProducer (skip paths + payload composition).

Discovery via SQL CTE and end-to-end pipeline behavior live in
``tests/integration/test_artifact_pipeline.py``. Here we exercise the
parts that don't need a real DB: the merged_into_id skip, alias handling,
and the confidence heuristic.
"""

from __future__ import annotations

import hashlib
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from kai_memory.identity import AGENT_IDENTITY_DATA_KEY

from kai_memory.storage.models.entity_relationship import EntityRelationship
from kai_memory.store import cognitive_config
from kai_memory.artifacts.producer import ProducerSubject
from kai_memory.artifacts.producers import entity_schema as entity_schema_module
from kai_memory.artifacts.producers.entity_schema import (
    EntitySchemaProducer,
)


class _FakeEntity:
    def __init__(self, *, name: str, entity_type: str, aliases=None, data=None, merged_into_id=None) -> None:
        self.id = uuid.uuid4()
        self.name = name
        self.entity_type = entity_type
        self.aliases = aliases if aliases is not None else []
        self.data = data if data is not None else {}
        self.merged_into_id = merged_into_id


def _empty_relationships_result():
    result = MagicMock()
    result.scalars.return_value.all.return_value = []
    return result


def _trace_contents_result(contents: list[str]):
    result = MagicMock()
    result.scalars.return_value.all.return_value = contents
    return result


def _confidence_result(trace_count: int):
    result = MagicMock()
    result.first = MagicMock(return_value=(trace_count,))
    return result


async def test_produce_returns_none_for_merged_entity() -> None:
    producer = EntitySchemaProducer()
    merged = _FakeEntity(name="Old", entity_type="person", merged_into_id=uuid.uuid4())
    session = MagicMock()
    session.get = AsyncMock(return_value=merged)
    subject = ProducerSubject(
        key=str(merged.id),
        payload={"entity_id": merged.id, "latest_trace_id": uuid.uuid4()},
    )
    result = await producer.produce(session, subject, uuid.uuid4())
    assert result is None


async def test_produce_returns_none_when_entity_missing() -> None:
    producer = EntitySchemaProducer()
    session = MagicMock()
    session.get = AsyncMock(return_value=None)
    subject = ProducerSubject(
        key="unknown",
        payload={"entity_id": uuid.uuid4(), "latest_trace_id": None},
    )
    result = await producer.produce(session, subject, uuid.uuid4())
    assert result is None


async def test_produce_composes_payload_from_entity_state() -> None:
    """No relationships, two traces -> confidence 0.75, canonical_fields from data."""
    producer = EntitySchemaProducer()
    entity = _FakeEntity(
        name="Луна",
        entity_type="animal",
        aliases=["Luna"],
        data={"breed": "Pomeranian", "color": "white"},
    )
    latest_trace = uuid.uuid4()
    session = MagicMock()
    session.get = AsyncMock(return_value=entity)

    # Cache hit in entity.data skips trace extraction; then relationships + confidence.
    session.execute = AsyncMock(side_effect=[_empty_relationships_result(), _confidence_result(2)])

    subject = ProducerSubject(
        key=str(entity.id),
        payload={"entity_id": entity.id, "latest_trace_id": latest_trace},
    )
    result = await producer.produce(session, subject, uuid.uuid4())
    assert result is not None
    assert result.artifact_type == "EntitySchema"
    assert result.subject_entity_id == entity.id
    payload = result.payload
    assert payload["canonical_name"] == "Луна"
    assert payload["entity_type"] == "animal"
    assert payload["aliases"] == ["Luna"]
    assert payload["canonical_fields"] == {"breed": "Pomeranian", "color": "white"}
    assert payload["last_evidence_trace_id"] == str(latest_trace)
    assert payload["confidence"] == pytest.approx(0.75)


async def test_produce_confidence_full_with_three_or_more_traces() -> None:
    producer = EntitySchemaProducer()
    entity = _FakeEntity(name="X", entity_type="person", data={"role": "known"})
    session = MagicMock()
    session.get = AsyncMock(return_value=entity)
    session.execute = AsyncMock(side_effect=[_empty_relationships_result(), _confidence_result(3)])

    subject = ProducerSubject(
        key=str(entity.id),
        payload={"entity_id": entity.id, "latest_trace_id": uuid.uuid4()},
    )
    result = await producer.produce(session, subject, uuid.uuid4())
    assert result is not None
    assert result.payload["confidence"] == pytest.approx(1.0)


async def test_produce_includes_canonical_symmetric_relationship_when_entity_is_to_side() -> None:
    producer = EntitySchemaProducer()
    entity = _FakeEntity(name="Misha", entity_type="person", data={"role": "known"})
    partner_id = uuid.uuid4()
    rel = EntityRelationship(
        id=uuid.uuid4(),
        from_entity_id=partner_id,
        to_entity_id=entity.id,
        relationship_type="partner_of",
        confidence=0.91,
    )
    session = MagicMock()
    session.get = AsyncMock(return_value=entity)

    rel_result = MagicMock()
    rel_result.scalars.return_value.all.return_value = [rel]
    session.execute = AsyncMock(side_effect=[rel_result, _confidence_result(3)])

    result = await producer.produce(
        session,
        ProducerSubject(key=str(entity.id), payload={"entity_id": entity.id, "latest_trace_id": uuid.uuid4()}),
        uuid.uuid4(),
    )

    assert result is not None
    assert result.payload["relationships"] == [
        {"predicate": "partner_of", "target_entity_id": str(partner_id), "confidence": 0.91}
    ]
    relationship_query = session.execute.await_args_list[0].args[0]
    assert "from_entity_id" in str(relationship_query)
    assert "to_entity_id" in str(relationship_query)


async def test_produce_content_hash_excludes_last_evidence_trace_id() -> None:
    """Two productions with identical canonical content but DIFFERENT
    ``last_evidence_trace_id`` must produce the same content_hash so
    ``BaseProducer.supersede()`` can short-circuit to action='noop'.

    Regression: prod observed 7 EntitySchema versions in 6 days on a
    single entity, payload byte-identical except for the trace pointer
    (``last_evidence_trace_id`` mutates every time a new trace touches
    the entity). Without excluding it from the hash, every L3 run
    re-supersedes. This test pins the fix.
    """
    producer = EntitySchemaProducer()
    entity = _FakeEntity(
        name="Луна",
        entity_type="animal",
        aliases=["Luna"],
        data={"breed": "Pomeranian"},
    )
    session = MagicMock()
    session.get = AsyncMock(return_value=entity)

    # Run 1 — trace pointer A.
    session.execute = AsyncMock(side_effect=[_empty_relationships_result(), _confidence_result(3)])
    trace_a = uuid.uuid4()
    result_a = await producer.produce(
        session,
        ProducerSubject(key="x", payload={"entity_id": entity.id, "latest_trace_id": trace_a}),
        uuid.uuid4(),
    )

    # Run 2 — trace pointer B (different), everything else identical.
    session.execute = AsyncMock(side_effect=[_empty_relationships_result(), _confidence_result(3)])
    trace_b = uuid.uuid4()
    result_b = await producer.produce(
        session,
        ProducerSubject(key="x", payload={"entity_id": entity.id, "latest_trace_id": trace_b}),
        uuid.uuid4(),
    )

    assert result_a is not None and result_b is not None
    # Pointers MUST differ in the payload (provenance is preserved on disk).
    assert result_a.payload["last_evidence_trace_id"] != result_b.payload["last_evidence_trace_id"]
    # But the content_hash MUST match — that's the whole point of the fix.
    assert result_a.content_hash is not None
    assert result_a.content_hash == result_b.content_hash, (
        "EntitySchema content_hash must be stable across runs when only "
        "last_evidence_trace_id changes; otherwise BaseProducer.supersede "
        "churns daily supersessions on byte-identical content."
    )


async def test_produce_content_hash_changes_when_canonical_content_changes() -> None:
    """Sanity: the hash MUST change when an actual content field
    (canonical_fields, aliases, etc.) changes, otherwise we'd noop on
    real updates.
    """
    producer = EntitySchemaProducer()
    session = MagicMock()

    # Same trace pointer, but the data differs.
    trace_id = uuid.uuid4()
    entity_a = _FakeEntity(name="Луна", entity_type="animal", data={"breed": "Pomeranian"})
    entity_b = _FakeEntity(name="Луна", entity_type="animal", data={"breed": "Spitz"})

    session.get = AsyncMock(return_value=entity_a)
    session.execute = AsyncMock(side_effect=[_empty_relationships_result(), _confidence_result(3)])
    result_a = await producer.produce(
        session,
        ProducerSubject(key="x", payload={"entity_id": entity_a.id, "latest_trace_id": trace_id}),
        uuid.uuid4(),
    )

    session.get = AsyncMock(return_value=entity_b)
    session.execute = AsyncMock(side_effect=[_empty_relationships_result(), _confidence_result(3)])
    result_b = await producer.produce(
        session,
        ProducerSubject(key="x", payload={"entity_id": entity_b.id, "latest_trace_id": trace_id}),
        uuid.uuid4(),
    )

    assert result_a is not None and result_b is not None
    assert result_a.content_hash != result_b.content_hash


async def test_produce_with_no_linked_traces_keeps_empty_fields_without_sonnet(monkeypatch: pytest.MonkeyPatch) -> None:
    producer = EntitySchemaProducer()
    entity = _FakeEntity(name="Sparse", entity_type="person")
    session = MagicMock()
    session.get = AsyncMock(return_value=entity)
    session.execute = AsyncMock(
        side_effect=[
            _trace_contents_result([]),
            _empty_relationships_result(),
            _confidence_result(1),
        ]
    )
    call_llm = AsyncMock(return_value='{"role":"friend"}')
    monkeypatch.setattr(entity_schema_module, "call_llm_text", call_llm)

    result = await producer.produce(
        session,
        ProducerSubject(key=str(entity.id), payload={"entity_id": entity.id, "latest_trace_id": uuid.uuid4()}),
        uuid.uuid4(),
    )

    assert result is not None
    assert result.payload["canonical_fields"] == {}
    assert entity.data == {}
    call_llm.assert_not_awaited()


async def test_produce_extracts_canonical_fields_from_one_linked_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    producer = EntitySchemaProducer()
    entity = _FakeEntity(name="Луна", entity_type="animal")
    session = MagicMock()
    session.get = AsyncMock(return_value=entity)
    session.execute = AsyncMock(
        side_effect=[
            _trace_contents_result(["Луна is Olga's white Pomeranian in Corfu."]),
            _empty_relationships_result(),
            _confidence_result(1),
        ]
    )
    provider = object()
    monkeypatch.setattr(entity_schema_module, "get_provider_for_model", lambda settings, model: provider)
    monkeypatch.setattr(entity_schema_module, "start_generation_span", lambda name, model: None)
    call_llm = AsyncMock(return_value='{"breed":"Pomeranian","color":"white","owner":"Olga","location":"Corfu"}')
    monkeypatch.setattr(entity_schema_module, "call_llm_text", call_llm)

    result = await producer.produce(
        session,
        ProducerSubject(key=str(entity.id), payload={"entity_id": entity.id, "latest_trace_id": uuid.uuid4()}),
        uuid.uuid4(),
    )

    assert result is not None
    assert result.payload["canonical_fields"] == {
        "breed": "Pomeranian",
        "color": "white",
        "owner": "Olga",
        "location": "Corfu",
    }
    # §8: the cache entry is source/version-tagged; stripping the _-meta keys
    # recovers the canonical_fields the payload ships.
    assert entity.data["_source"] == "sonnet"
    assert entity.data["_v"] == 1
    assert {k: v for k, v in entity.data.items() if not k.startswith("_")} == result.payload["canonical_fields"]
    call_llm.assert_awaited_once()
    _, prompt = call_llm.await_args.args[:2]
    assert "Entity: Луна" in prompt
    assert "Entity type: animal" in prompt
    assert "Maximum 10 keys" in prompt
    assert "STABLE CANONICAL facts" in prompt
    assert "Луна is Olga's white Pomeranian in Corfu." in prompt
    assert call_llm.await_args.kwargs["thinking_mode"] == "disabled"
    assert call_llm.await_args.kwargs["task_name"] == "memory.entity_schema.extract_from_traces"
    assert call_llm.await_args.kwargs["max_tokens"] == 1024


def test_parse_canonical_fields_response_strips_markdown_fences() -> None:
    entity_id = uuid.uuid4()
    raw = '```json\n{"breed":"Pomeranian","color":"white"}\n```'

    result = entity_schema_module._parse_canonical_fields_response(raw, entity_id)

    assert result == {"breed": "Pomeranian", "color": "white"}


async def test_produce_caps_linked_trace_query_at_config_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    cognitive_config.set_for_testing("entity_schema_trace_window_limit", 20)
    cognitive_config.set_for_testing("entity_schema_trace_window_days", 180)
    try:
        producer = EntitySchemaProducer()
        entity = _FakeEntity(name="Busy", entity_type="person")
        session = MagicMock()
        session.get = AsyncMock(return_value=entity)
        session.execute = AsyncMock(
            side_effect=[
                _trace_contents_result([f"trace {i}" for i in range(25)]),
                _empty_relationships_result(),
                _confidence_result(3),
            ]
        )
        monkeypatch.setattr(entity_schema_module, "get_provider_for_model", lambda settings, model: object())
        monkeypatch.setattr(entity_schema_module, "start_generation_span", lambda name, model: None)
        call_llm = AsyncMock(return_value='{"role":"known"}')
        monkeypatch.setattr(entity_schema_module, "call_llm_text", call_llm)

        result = await producer.produce(
            session,
            ProducerSubject(key=str(entity.id), payload={"entity_id": entity.id, "latest_trace_id": uuid.uuid4()}),
            uuid.uuid4(),
        )

        assert result is not None
        trace_query_params = session.execute.await_args_list[0].args[1]
        assert trace_query_params["limit"] == 20
        assert trace_query_params["window_days"] == 180
        _, prompt = call_llm.await_args.args[:2]
        assert "- trace 19" in prompt
        assert "- trace 20" not in prompt
    finally:
        cognitive_config.reset_cache()


async def test_produce_uses_populated_entity_data_cache_without_sonnet(monkeypatch: pytest.MonkeyPatch) -> None:
    producer = EntitySchemaProducer()
    entity = _FakeEntity(name="Olga", entity_type="person", data={"role": "Misha's mother"})
    session = MagicMock()
    session.get = AsyncMock(return_value=entity)
    session.execute = AsyncMock(side_effect=[_empty_relationships_result(), _confidence_result(3)])
    call_llm = AsyncMock(return_value='{"role":"wrong"}')
    monkeypatch.setattr(entity_schema_module, "call_llm_text", call_llm)

    result = await producer.produce(
        session,
        ProducerSubject(key=str(entity.id), payload={"entity_id": entity.id, "latest_trace_id": uuid.uuid4()}),
        uuid.uuid4(),
    )

    assert result is not None
    assert result.payload["canonical_fields"] == {"role": "Misha's mother"}
    call_llm.assert_not_awaited()
    assert session.execute.await_count == 2


async def test_produce_rejects_malformed_json_extraction(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    producer = EntitySchemaProducer()
    entity = _FakeEntity(name="Elena", entity_type="person")
    session = MagicMock()
    session.get = AsyncMock(return_value=entity)
    session.execute = AsyncMock(
        side_effect=[
            _trace_contents_result(["Elena is Misha's partner."]),
            _empty_relationships_result(),
            _confidence_result(1),
        ]
    )
    monkeypatch.setattr(entity_schema_module, "get_provider_for_model", lambda settings, model: object())
    monkeypatch.setattr(entity_schema_module, "start_generation_span", lambda name, model: None)
    monkeypatch.setattr(entity_schema_module, "call_llm_text", AsyncMock(return_value="{not json"))

    with caplog.at_level("WARNING"):
        result = await producer.produce(
            session,
            ProducerSubject(key=str(entity.id), payload={"entity_id": entity.id, "latest_trace_id": uuid.uuid4()}),
            uuid.uuid4(),
        )

    assert result is not None
    assert result.payload["canonical_fields"] == {}
    assert entity.data == {}
    assert str(entity.id) in caplog.text


async def test_produce_warns_when_extraction_returns_empty_json(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    producer = EntitySchemaProducer()
    entity = _FakeEntity(name="Луна", entity_type="animal")
    session = MagicMock()
    session.get = AsyncMock(return_value=entity)
    session.execute = AsyncMock(
        side_effect=[
            _trace_contents_result(["Луна is Olga's white Pomeranian in Corfu."]),
            _empty_relationships_result(),
            _confidence_result(1),
        ]
    )
    monkeypatch.setattr(entity_schema_module, "get_provider_for_model", lambda settings, model: object())
    monkeypatch.setattr(entity_schema_module, "start_generation_span", lambda name, model: None)
    monkeypatch.setattr(entity_schema_module, "call_llm_text", AsyncMock(return_value="{}"))

    with caplog.at_level("WARNING"):
        result = await producer.produce(
            session,
            ProducerSubject(key=str(entity.id), payload={"entity_id": entity.id, "latest_trace_id": uuid.uuid4()}),
            uuid.uuid4(),
        )

    assert result is not None
    assert result.payload["canonical_fields"] == {}
    assert entity.data == {}
    assert str(entity.id) in caplog.text
    assert "Луна" in caplog.text
    assert "extraction returned empty" in caplog.text
    assert "sonnet_response_len=2" in caplog.text
    assert f"sonnet_response_sha256={hashlib.sha256(b'{}').hexdigest()}" in caplog.text
    assert "Sonnet returned" not in caplog.text


async def test_produce_rejects_non_string_extraction_values(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    producer = EntitySchemaProducer()
    entity = _FakeEntity(name="Katya", entity_type="person")
    session = MagicMock()
    session.get = AsyncMock(return_value=entity)
    session.execute = AsyncMock(
        side_effect=[
            _trace_contents_result(["Katya is in Perm."]),
            _empty_relationships_result(),
            _confidence_result(1),
        ]
    )
    monkeypatch.setattr(entity_schema_module, "get_provider_for_model", lambda settings, model: object())
    monkeypatch.setattr(entity_schema_module, "start_generation_span", lambda name, model: None)
    monkeypatch.setattr(entity_schema_module, "call_llm_text", AsyncMock(return_value='{"hometown":"Perm","count":2}'))

    with caplog.at_level("WARNING"):
        result = await producer.produce(
            session,
            ProducerSubject(key=str(entity.id), payload={"entity_id": entity.id, "latest_trace_id": uuid.uuid4()}),
            uuid.uuid4(),
        )

    assert result is not None
    assert result.payload["canonical_fields"] == {}
    assert entity.data == {}
    assert str(entity.id) in caplog.text


async def test_produce_handles_sonnet_api_errors_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    producer = EntitySchemaProducer()
    entity = _FakeEntity(name="Нора", entity_type="animal")
    session = MagicMock()
    session.get = AsyncMock(return_value=entity)
    session.execute = AsyncMock(
        side_effect=[
            _trace_contents_result(["Нора is a Pomeranian companion to Луна."]),
            _empty_relationships_result(),
            _confidence_result(1),
        ]
    )
    monkeypatch.setattr(entity_schema_module, "get_provider_for_model", lambda settings, model: object())
    monkeypatch.setattr(entity_schema_module, "start_generation_span", lambda name, model: None)
    call_llm = AsyncMock(side_effect=RuntimeError("sonnet unavailable"))
    monkeypatch.setattr(entity_schema_module, "call_llm_text", call_llm)

    with caplog.at_level("WARNING"):
        result = await producer.produce(
            session,
            ProducerSubject(key=str(entity.id), payload={"entity_id": entity.id, "latest_trace_id": uuid.uuid4()}),
            uuid.uuid4(),
        )

    assert result is not None
    assert result.payload["canonical_fields"] == {}
    assert entity.data == {}
    assert str(entity.id) in caplog.text


# ── INC 3 live cutover (entityschema_live — all entity types, flag-gated) ────
# The earlier person-only `entityschema_live_for_person` flag was retired; the
# all-types `entityschema_live` flag is the single live cutover gate now.


def _patch_flags(monkeypatch, *, shadow: bool = False, live_all: bool = False) -> None:
    flags = {
        "entityschema_deterministic_enabled": shadow,
        "entityschema_live": live_all,
    }
    monkeypatch.setattr(cognitive_config, "get_bool", lambda key: flags.get(key, False))


async def _run_cutover(monkeypatch, *, entity_type, new_card, shadow=False, live_all=False):
    """Drive produce() with the Sonnet card cached in entity.data and a mocked
    _deterministic_card returning ``new_card`` — isolating the cutover DECISION.
    Returns (payload_canonical_fields, deterministic_mock, sonnet_fields)."""
    sonnet = {"role": "founder", "location": "Corfu"}
    producer = EntitySchemaProducer()
    entity = _FakeEntity(name="Misha", entity_type=entity_type, data=dict(sonnet))  # data → Sonnet cache hit
    session = MagicMock()
    session.get = AsyncMock(return_value=entity)
    session.execute = AsyncMock(side_effect=[_empty_relationships_result(), _confidence_result(3)])
    _patch_flags(monkeypatch, shadow=shadow, live_all=live_all)
    det = AsyncMock(return_value=new_card)
    monkeypatch.setattr(EntitySchemaProducer, "_deterministic_card", det)
    result = await producer.produce(
        session,
        ProducerSubject(key=str(entity.id), payload={"entity_id": entity.id, "latest_trace_id": uuid.uuid4()}),
        uuid.uuid4(),
    )
    assert result is not None
    return result.payload["canonical_fields"], det, sonnet


async def test_cutover_off_ships_sonnet(monkeypatch):
    # entityschema_live OFF (default) → ship Sonnet, never compute the new path.
    fields, det, sonnet = await _run_cutover(monkeypatch, entity_type="person", new_card={"summary": "new"})
    assert fields == sonnet
    assert det.await_count == 0


async def test_full_cutover_off_non_person_ships_sonnet(monkeypatch):
    # entityschema_live OFF → a non-person ships Sonnet (unchanged), new path not computed.
    fields, det, sonnet = await _run_cutover(monkeypatch, entity_type="topic", new_card={"summary": "new"})
    assert fields == sonnet
    assert det.await_count == 0


async def test_full_cutover_on_non_person_rich_ships_new(monkeypatch):
    # entityschema_live ON → a non-person with a rich (non-None) new card ships NEW.
    fields, det, _ = await _run_cutover(
        monkeypatch, entity_type="topic", new_card={"summary": "curated topic card"}, live_all=True
    )
    assert fields == {"summary": "curated topic card"}
    assert det.await_count == 1


async def test_full_cutover_on_person_ships_new(monkeypatch):
    # entityschema_live ON covers persons too (the all-types flag is the only live gate).
    fields, det, _ = await _run_cutover(
        monkeypatch, entity_type="person", new_card={"summary": "curated person card"}, live_all=True
    )
    assert fields == {"summary": "curated person card"}
    assert det.await_count == 1


async def test_full_cutover_on_thin_new_card_falls_back_to_sonnet(monkeypatch):
    # entityschema_live ON but the new path is thin/empty (None) → Sonnet fallback,
    # never an empty card. The self-protecting "NEW where good, Sonnet where thin".
    fields, det, sonnet = await _run_cutover(monkeypatch, entity_type="topic", new_card=None, live_all=True)
    assert fields == sonnet
    assert det.await_count == 1  # computed, but None → Sonnet


async def test_shadow_on_logs_but_does_not_flip_when_live_off(monkeypatch):
    # Shadow ON, live OFF: the deterministic card is computed+logged for measurement, but
    # the LIVE payload stays Sonnet (no flip).
    fields, det, sonnet = await _run_cutover(
        monkeypatch, entity_type="person", new_card={"summary": "new"}, shadow=True
    )
    assert fields == sonnet
    assert det.await_count == 1
    assert det.await_args.kwargs["log_shadow"] is True  # shadow_diff logging preserved


async def test_cutover_shadow_on_and_live_on_ships_new_and_logs(monkeypatch):
    # The real prod cutover state: shadow ON (measuring) + live ON → ship the NEW card
    # AND keep logging the shadow_diff (log_shadow=True).
    fields, det, _ = await _run_cutover(
        monkeypatch, entity_type="person", new_card={"summary": "curated"}, shadow=True, live_all=True
    )
    assert fields == {"summary": "curated"}
    assert det.await_count == 1
    assert det.await_args.kwargs["log_shadow"] is True


async def _run_cutover_tracking_sonnet(monkeypatch, *, new_card, shadow=False, live_all=True):
    """Like _run_cutover but mocks _canonical_fields too (cold-cache entity) so we can
    assert whether the Sonnet extraction was actually called."""
    producer = EntitySchemaProducer()
    entity = _FakeEntity(name="Misha", entity_type="topic", data={})  # cold cache → would need Sonnet
    session = MagicMock()
    session.get = AsyncMock(return_value=entity)
    session.execute = AsyncMock(side_effect=[_empty_relationships_result(), _confidence_result(3)])
    _patch_flags(monkeypatch, shadow=shadow, live_all=live_all)
    sonnet = AsyncMock(return_value={"role": "sonnet"})
    monkeypatch.setattr(EntitySchemaProducer, "_canonical_fields", sonnet)
    det = AsyncMock(return_value=new_card)
    monkeypatch.setattr(EntitySchemaProducer, "_deterministic_card", det)
    result = await producer.produce(
        session,
        ProducerSubject(key=str(entity.id), payload={"entity_id": entity.id, "latest_trace_id": uuid.uuid4()}),
        uuid.uuid4(),
    )
    assert result is not None
    return result.payload["canonical_fields"], sonnet


async def test_full_cutover_rich_new_card_skips_sonnet_extraction(monkeypatch):
    # The perf fix: live ON + shadow OFF + a rich new card → Sonnet (_canonical_fields)
    # is NEVER called — it's a true fallback, not an unconditional extra cost at rollout.
    fields, sonnet = await _run_cutover_tracking_sonnet(monkeypatch, new_card={"summary": "curated"})
    assert fields == {"summary": "curated"}
    assert sonnet.await_count == 0  # Sonnet skipped


async def test_full_cutover_thin_new_card_calls_sonnet_fallback(monkeypatch):
    # live ON + shadow OFF + thin/None new card → Sonnet IS computed as the fallback.
    fields, sonnet = await _run_cutover_tracking_sonnet(monkeypatch, new_card=None)
    assert fields == {"role": "sonnet"}
    assert sonnet.await_count == 1


async def test_shadow_on_computes_sonnet_for_diff_even_when_card_ships(monkeypatch):
    # shadow ON + live ON + rich card → Sonnet computed once (needed for the agreement
    # diff) and the NEW card still ships.
    fields, sonnet = await _run_cutover_tracking_sonnet(monkeypatch, new_card={"summary": "curated"}, shadow=True)
    assert fields == {"summary": "curated"}
    assert sonnet.await_count == 1  # shadow diff needs Sonnet


async def test_agent_binding_does_not_skip_extraction_or_enter_canonical_fields(monkeypatch) -> None:
    producer = EntitySchemaProducer()
    entity = _FakeEntity(
        name="Atlas",
        entity_type="agent",
        data={AGENT_IDENTITY_DATA_KEY: "host-assistant", "stale": {"nested": "discard"}},
    )
    session = MagicMock()
    session.get = AsyncMock(return_value=entity)
    session.execute = AsyncMock(
        side_effect=[
            _trace_contents_result(["Atlas is a thoughtful assistant."]),
            _empty_relationships_result(),
            _confidence_result(1),
        ]
    )
    monkeypatch.setattr(entity_schema_module, "get_provider_for_model", lambda *args: object())
    monkeypatch.setattr(entity_schema_module, "start_generation_span", lambda *args: None)
    call_llm = AsyncMock(return_value='{"role":"assistant","_agent_identity_id":"other-host","_source":"spoofed"}')
    monkeypatch.setattr(entity_schema_module, "call_llm_text", call_llm)

    produced = await producer.produce(
        session,
        ProducerSubject(key=str(entity.id), payload={"entity_id": entity.id, "latest_trace_id": uuid.uuid4()}),
        uuid.uuid4(),
    )

    assert produced is not None
    assert produced.payload["canonical_fields"] == {"role": "assistant"}
    assert entity.data == {
        "role": "assistant",
        AGENT_IDENTITY_DATA_KEY: "host-assistant",
        "_source": "sonnet",
        "_v": 1,
    }
    assert await producer._canonical_fields(session, entity) == {"role": "assistant"}
    call_llm.assert_awaited_once()
