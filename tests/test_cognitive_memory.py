"""Unit tests for CognitiveMemoryService (mocked DB)."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.dialects import postgresql

from kai_memory.storage.models.memory_entity import MemoryEntity
from kai_memory.store import cognitive_config
from kai_memory.store.cognitive_memory import DURABLE_INITIAL_SALIENCE, CognitiveMemoryService, _has_durable_fact
from kai_memory.store.embeddings import EMBEDDING_DIMENSIONS


# ── Durable-fact detection (write-path fact-loss fix) ─────────────────────────


def test_has_durable_fact_detects_per_fact_tag():
    assert _has_durable_fact({"facts": [{"field": "x", "durability": "durable"}]}) is True
    assert _has_durable_fact({"facts": [{"field": "x", "durability": "transient"}]}) is False


def test_has_durable_fact_detects_trace_level_marker():
    assert _has_durable_fact({"durability": "durable"}) is True


def test_has_durable_fact_false_on_empty_or_none():
    assert _has_durable_fact(None) is False
    assert _has_durable_fact({}) is False
    assert _has_durable_fact({"facts": []}) is False


def test_durable_initial_salience_floor_is_above_prune_line():
    # The floor (0.35) must sit above the 0.01 prune threshold; lifts a low-significance
    # durable fact (sig-1 → 0.2) above it without touching significance.
    assert DURABLE_INITIAL_SALIENCE > 0.01
    assert max(1 / 5.0, DURABLE_INITIAL_SALIENCE) == DURABLE_INITIAL_SALIENCE  # sig-1 gets lifted


# ── Store tests ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_store_memory_trace():
    session = AsyncMock()
    session.flush = AsyncMock()

    with patch("kai_memory.store.cognitive_memory.generate_embedding", return_value=[0.1] * 1536):
        svc = CognitiveMemoryService(session)
        trace = await svc.store_trace(
            content="Misha works at Examplecorp",
            memory_type="fact",
            significance=3,
            source_type="conversation",
            source_chat_id=12345,
        )
    assert trace is not None
    assert trace.content == "Misha works at Examplecorp"
    assert trace.memory_type == "fact"
    assert trace.significance == 3
    assert trace.salience == pytest.approx(0.6)
    assert trace.initial_salience == pytest.approx(0.6)
    session.add.assert_called_once()
    assert session.flush.await_count == 2  # once for trace, once for embedding


@pytest.mark.asyncio
async def test_store_trace_sets_initial_salience_from_significance():
    session = AsyncMock()
    session.flush = AsyncMock()

    with patch("kai_memory.store.cognitive_memory.generate_embedding", return_value=[0.1] * 1536):
        svc = CognitiveMemoryService(session)
        trace = await svc.store_trace(
            content="Critical allergy info",
            memory_type="fact",
            significance=5,
            source_type="conversation",
        )
    assert trace.salience == pytest.approx(1.0)
    assert trace.initial_salience == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_store_trace_with_emotion():
    session = AsyncMock()
    session.flush = AsyncMock()

    with patch("kai_memory.store.cognitive_memory.generate_embedding", return_value=[0.1] * 1536):
        svc = CognitiveMemoryService(session)
        trace = await svc.store_trace(
            content="Intense deadline conversation",
            memory_type="episode",
            significance=4,
            source_type="conversation",
            emotion_valence=-0.3,
            emotion_arousal=0.7,
        )
    assert trace.emotion_valence == pytest.approx(-0.3)
    assert trace.emotion_arousal == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_store_trace_embedding_failure_still_saves():
    session = AsyncMock()
    session.flush = AsyncMock()

    with patch("kai_memory.store.cognitive_memory.generate_embedding", side_effect=RuntimeError("API down")):
        svc = CognitiveMemoryService(session)
        trace = await svc.store_trace(
            content="Some memory",
            memory_type="fact",
            significance=3,
            source_type="conversation",
        )
    assert trace is not None
    assert trace.embedding is None
    session.flush.assert_awaited_once()  # only the first flush (trace save)


@pytest.mark.asyncio
async def test_store_trace_refuses_zero_vector_embedding():
    """Storage refuses a zero-norm embedding (defense-in-depth): the trace is
    saved with NULL embedding (backfillable) rather than a poison vector that
    NaN-sorts to the top of every search."""
    session = AsyncMock()
    session.flush = AsyncMock()

    svc = CognitiveMemoryService(session)
    trace = await svc.store_trace(
        content="Some memory",
        memory_type="fact",
        significance=3,
        source_type="conversation",
        embedding=[0.0] * EMBEDDING_DIMENSIONS,
    )
    assert trace is not None
    assert trace.embedding is None
    # Only the initial trace save flushes; the zero embedding is never written.
    session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_store_trace_all_memory_types():
    """All valid memory_type values should be storable."""
    session = AsyncMock()
    session.flush = AsyncMock()

    with patch("kai_memory.store.cognitive_memory.generate_embedding", return_value=[0.1] * 1536):
        svc = CognitiveMemoryService(session)
        for mtype in ["fact", "episode", "impression", "preference", "plan", "intention"]:
            trace = await svc.store_trace(
                content=f"Test {mtype}",
                memory_type=mtype,
                significance=3,
                source_type="conversation",
            )
            assert trace.memory_type == mtype


# ── Search tests ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_traces_by_embedding():
    session = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = []
    session.execute = AsyncMock(return_value=mock_result)

    with patch("kai_memory.store.cognitive_memory.generate_embedding", return_value=[0.1] * 1536):
        svc = CognitiveMemoryService(session)
        results = await svc.search_traces("Misha work", limit=5)
    assert isinstance(results, list)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_search_traces_embedding_failure_returns_empty():
    session = AsyncMock()

    with patch("kai_memory.store.cognitive_memory.generate_embedding", side_effect=RuntimeError("fail")):
        svc = CognitiveMemoryService(session)
        results = await svc.search_traces("anything")
    assert results == []


@pytest.mark.asyncio
async def test_search_traces_raw_embedding_failure_returns_empty():
    """Retrieval path: when query embedding raises (key down), the semantic
    component degrades to empty results — no crash, no write — so the
    non-semantic priming paths (entity-linked, graph, intentions) still run."""
    from kai_memory.store.embeddings import EmbeddingError

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=AssertionError("must not query DB when embedding fails"))

    with patch(
        "kai_memory.store.cognitive_memory.generate_embedding",
        side_effect=EmbeddingError("key down"),
    ):
        svc = CognitiveMemoryService(session)
        results = await svc.search_traces_raw("anything")
    assert results == []
    session.execute.assert_not_called()


@pytest.mark.asyncio
async def test_search_traces_with_memory_type_filter():
    session = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = []
    session.execute = AsyncMock(return_value=mock_result)

    with patch("kai_memory.store.cognitive_memory.generate_embedding", return_value=[0.1] * 1536):
        svc = CognitiveMemoryService(session)
        results = await svc.search_traces("test", memory_types=["fact", "episode"])
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_search_memories_excludes_merged_entities():
    session = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = []
    session.execute = AsyncMock(return_value=mock_result)

    with patch("kai_memory.store.cognitive_memory.generate_embedding", return_value=[0.1] * 1536):
        svc = CognitiveMemoryService(session)
        await svc.search_traces("test")

    # Execute order is pin-plan (0), search SELECT (1), restore-defaults (2).
    stmt = session.execute.call_args_list[1].args[0]
    sql = str(stmt.compile(dialect=postgresql.dialect()))
    assert "memory_entities.merged_into_id IS NOT NULL" in sql
    assert "NOT (EXISTS" in sql


@pytest.mark.asyncio
async def test_search_reranks_by_salience_and_arousal():
    """Higher salience and arousal should boost score."""
    session = AsyncMock()

    trace_high = MagicMock()
    trace_high.salience = 0.9
    trace_high.emotion_arousal = 0.8

    trace_low = MagicMock()
    trace_low.salience = 0.2
    trace_low.emotion_arousal = 0.1

    mock_result = MagicMock()
    mock_result.all.return_value = [
        (trace_low, 0.85),  # higher similarity but low salience
        (trace_high, 0.80),  # lower similarity but high salience
    ]
    session.execute = AsyncMock(return_value=mock_result)

    with patch("kai_memory.store.cognitive_memory.generate_embedding", return_value=[0.1] * 1536):
        svc = CognitiveMemoryService(session)
        results = await svc.search_traces("test", limit=10)

    # trace_high should rank first due to salience+arousal boost
    assert results[0][0] is trace_high
    assert results[1][0] is trace_low


# ── Access tracking tests ─────────────────────────────────


@pytest.mark.asyncio
async def test_record_access_uses_atomic_increment():
    """access_count must use atomic SQL increment, not read-modify-write."""
    session = AsyncMock()
    session.execute = AsyncMock()

    svc = CognitiveMemoryService(session)
    await svc.record_access(uuid.uuid4())

    call_args = session.execute.call_args
    stmt = call_args[0][0]
    compiled = stmt.compile(compile_kwargs={"literal_binds": True})
    sql_str = str(compiled)
    assert "access_count + " in sql_str or "access_count +" in sql_str


@pytest.mark.asyncio
async def test_record_access_appends_timestamp():
    """access_timestamps must be appended to, not replaced."""
    session = AsyncMock()
    session.execute = AsyncMock()

    svc = CognitiveMemoryService(session)
    await svc.record_access(uuid.uuid4())

    call_args = session.execute.call_args
    stmt = call_args[0][0]
    compiled = stmt.compile(compile_kwargs={"literal_binds": True})
    sql_str = str(compiled)
    # Should use JSONB concatenation (||) or jsonb_build_array
    assert "jsonb_build_array" in sql_str or "||" in sql_str


# ── Entity operation tests ────────────────────────────────


@pytest.mark.asyncio
async def test_find_or_create_entity_creates_new(monkeypatch):
    """Combined lookup query returns nothing → INSERT ON CONFLICT path."""
    # Focus on the core lookup+insert sequence; the log-only Tier-0 shadow probe
    # (task #8) adds a query on this miss path and has its own dedicated tests.
    monkeypatch.setitem(cognitive_config._CACHE, "tier0_shadow_probe_enabled", 0.0)
    session = AsyncMock()

    lookup_result = MagicMock()
    lookup_result.scalar_one_or_none.return_value = None
    new_entity = MagicMock()
    new_entity.id = uuid.uuid4()
    new_entity.name = "Maria"
    insert_result = MagicMock()
    insert_result.scalar_one.return_value = new_entity

    session.execute = AsyncMock(side_effect=[lookup_result, insert_result])

    svc = CognitiveMemoryService(session)
    entity = await svc.find_or_create_entity("Maria", "person")
    assert entity.name == "Maria"
    assert session.execute.await_count == 2  # combined lookup + insert


@pytest.mark.asyncio
async def test_find_or_create_entity_finds_existing():
    """Combined lookup query returns entity → bump_mention path."""
    session = AsyncMock()

    existing = MagicMock()
    existing.id = uuid.uuid4()
    existing.name = "Maria"

    lookup_result = MagicMock()
    lookup_result.scalar_one_or_none.return_value = existing

    session.execute = AsyncMock(side_effect=[lookup_result, MagicMock()])

    svc = CognitiveMemoryService(session)
    entity = await svc.find_or_create_entity("Maria", "person")
    assert entity.name == "Maria"
    assert session.execute.await_count == 2  # combined lookup + bump_mention


@pytest.mark.asyncio
async def test_find_or_create_entity_finds_by_alias():
    """Combined lookup matches via alias JSONB containment."""
    session = AsyncMock()

    existing = MagicMock()
    existing.id = uuid.uuid4()
    existing.name = "Maria Petrova"

    lookup_result = MagicMock()
    lookup_result.scalar_one_or_none.return_value = existing

    session.execute = AsyncMock(side_effect=[lookup_result, MagicMock()])

    svc = CognitiveMemoryService(session)
    entity = await svc.find_or_create_entity("Maria", "person")
    assert entity.name == "Maria Petrova"


@pytest.mark.asyncio
async def test_find_or_create_entity_atomic_mention_bump():
    """mention_count increment must be atomic SQL, not read-modify-write."""
    session = AsyncMock()

    existing = MagicMock()
    existing.id = uuid.uuid4()

    lookup_result = MagicMock()
    lookup_result.scalar_one_or_none.return_value = existing

    session.execute = AsyncMock(side_effect=[lookup_result, MagicMock()])

    svc = CognitiveMemoryService(session)
    await svc.find_or_create_entity("Misha", "person")

    # Second call is the bump_mention UPDATE
    bump_call = session.execute.call_args_list[1]
    stmt = bump_call[0][0]
    compiled = stmt.compile(compile_kwargs={"literal_binds": True})
    sql_str = str(compiled)
    assert "mention_count + " in sql_str or "mention_count +" in sql_str


# ── Link tests ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_link_trace_to_entity_new_link():
    """New link: INSERT ON CONFLICT returns the new row."""
    session = AsyncMock()
    new_link = MagicMock()
    new_link.role = "about"

    insert_result = MagicMock()
    insert_result.scalar_one_or_none.return_value = new_link
    session.execute = AsyncMock(return_value=insert_result)

    svc = CognitiveMemoryService(session)
    link = await svc.link_trace_to_entity(uuid.uuid4(), uuid.uuid4(), "about")
    assert link.role == "about"
    session.execute.assert_awaited_once()  # only the INSERT


@pytest.mark.asyncio
async def test_link_trace_to_entity_duplicate_is_idempotent():
    """Duplicate link: INSERT ON CONFLICT DO NOTHING returns None, then SELECT fetches existing."""
    session = AsyncMock()
    existing_link = MagicMock()
    existing_link.role = "about"

    insert_result = MagicMock()
    insert_result.scalar_one_or_none.return_value = None  # conflict — no row returned
    fetch_result = MagicMock()
    fetch_result.scalar_one.return_value = existing_link

    session.execute = AsyncMock(side_effect=[insert_result, fetch_result])

    svc = CognitiveMemoryService(session)
    link = await svc.link_trace_to_entity(uuid.uuid4(), uuid.uuid4(), "about")
    assert link.role == "about"
    assert session.execute.await_count == 2  # INSERT + SELECT fallback


@pytest.mark.asyncio
async def test_link_trace_to_entity_custom_role():
    session = AsyncMock()
    new_link = MagicMock()
    new_link.role = "source"

    insert_result = MagicMock()
    insert_result.scalar_one_or_none.return_value = new_link
    session.execute = AsyncMock(return_value=insert_result)

    svc = CognitiveMemoryService(session)
    link = await svc.link_trace_to_entity(uuid.uuid4(), uuid.uuid4(), "source")
    assert link.role == "source"


@pytest.mark.asyncio
async def test_link_trace_to_entity_uses_on_conflict():
    """SQL statement must use INSERT ... ON CONFLICT DO NOTHING."""
    session = AsyncMock()
    new_link = MagicMock()
    insert_result = MagicMock()
    insert_result.scalar_one_or_none.return_value = new_link
    session.execute = AsyncMock(return_value=insert_result)

    svc = CognitiveMemoryService(session)
    await svc.link_trace_to_entity(uuid.uuid4(), uuid.uuid4(), "about")

    call_args = session.execute.call_args
    stmt = call_args[0][0]
    compiled = stmt.compile(compile_kwargs={"literal_binds": True})
    sql_str = str(compiled)
    assert "ON CONFLICT" in sql_str
    assert "DO NOTHING" in sql_str


@pytest.mark.asyncio
async def test_link_trace_walks_to_survivor():
    session = AsyncMock()
    trace_id = uuid.uuid4()
    absorbed = MemoryEntity(
        id=uuid.uuid4(),
        name="Natasha",
        entity_type="person",
        merged_into_id=uuid.uuid4(),
    )
    survivor = MemoryEntity(
        id=absorbed.merged_into_id,
        name="Elena",
        entity_type="person",
    )
    new_link = MagicMock()
    insert_result = MagicMock()
    insert_result.scalar_one_or_none.return_value = new_link
    session.execute = AsyncMock(return_value=insert_result)
    session.get = AsyncMock(side_effect=[absorbed, survivor])

    svc = CognitiveMemoryService(session)
    link = await svc.link_trace_to_entity(trace_id, absorbed.id, "about")

    assert link is new_link
    stmt = session.execute.call_args[0][0]
    assert stmt.compile().params["entity_id"] == survivor.id


@pytest.mark.asyncio
async def test_link_trace_resolves_chain_of_six():
    session = AsyncMock()
    trace_id = uuid.uuid4()
    from itertools import pairwise

    entities = [MemoryEntity(id=uuid.uuid4(), name=f"E{i}", entity_type="person") for i in range(7)]
    for current, next_entity in pairwise(entities):
        current.merged_into_id = next_entity.id
    new_link = MagicMock()
    insert_result = MagicMock()
    insert_result.scalar_one_or_none.return_value = new_link
    session.execute = AsyncMock(return_value=insert_result)
    session.get = AsyncMock(side_effect=entities)

    svc = CognitiveMemoryService(session)
    await svc.link_trace_to_entity(trace_id, entities[0].id, "about")

    stmt = session.execute.call_args[0][0]
    assert stmt.compile().params["entity_id"] == entities[-1].id


# ── Invalidation tests ────────────────────────────────────


@pytest.mark.asyncio
async def test_invalidate_trace():
    session = AsyncMock()
    session.execute = AsyncMock()

    svc = CognitiveMemoryService(session)
    trace_id = uuid.uuid4()
    new_trace_id = uuid.uuid4()
    await svc.invalidate_trace(trace_id, invalidated_by=new_trace_id, confidence=0.9)

    call_args = session.execute.call_args
    stmt = call_args[0][0]
    compiled = stmt.compile(compile_kwargs={"literal_binds": True})
    sql_str = str(compiled)
    assert "invalidated_at" in sql_str
    assert "invalidated_by" in sql_str


@pytest.mark.asyncio
async def test_invalidate_trace_only_if_not_already_invalidated():
    """Should only invalidate traces where invalidated_at IS NULL."""
    session = AsyncMock()
    session.execute = AsyncMock()

    svc = CognitiveMemoryService(session)
    await svc.invalidate_trace(uuid.uuid4())

    call_args = session.execute.call_args
    stmt = call_args[0][0]
    compiled = stmt.compile(compile_kwargs={"literal_binds": True})
    sql_str = str(compiled)
    assert "invalidated_at IS NULL" in sql_str


@pytest.mark.asyncio
async def test_invalidate_trace_default_confidence():
    session = AsyncMock()
    session.execute = AsyncMock()

    svc = CognitiveMemoryService(session)
    await svc.invalidate_trace(uuid.uuid4())

    # Default confidence = 1.0
    session.execute.assert_awaited_once()


# ── Read helper tests ─────────────────────────────────────


@pytest.mark.asyncio
async def test_get_trace():
    session = AsyncMock()
    mock_trace = MagicMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_trace
    session.execute = AsyncMock(return_value=mock_result)

    svc = CognitiveMemoryService(session)
    result = await svc.get_trace(uuid.uuid4())
    assert result is mock_trace


@pytest.mark.asyncio
async def test_get_trace_not_found():
    session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=mock_result)

    svc = CognitiveMemoryService(session)
    result = await svc.get_trace(uuid.uuid4())
    assert result is None


@pytest.mark.asyncio
async def test_get_entity():
    session = AsyncMock()
    mock_entity = MagicMock()
    mock_entity.merged_into_id = None
    session.get = AsyncMock(return_value=mock_entity)

    svc = CognitiveMemoryService(session)
    entity_id = uuid.uuid4()
    result = await svc.get_entity(entity_id)
    assert result is mock_entity
    session.get.assert_awaited_once_with(MemoryEntity, entity_id)


@pytest.mark.asyncio
async def test_get_entity_follows_merge_chain():
    first = MemoryEntity(id=uuid.uuid4(), name="A", entity_type="person")
    second = MemoryEntity(id=uuid.uuid4(), name="B", entity_type="person")
    survivor = MemoryEntity(id=uuid.uuid4(), name="C", entity_type="person")
    first.merged_into_id = second.id
    second.merged_into_id = survivor.id
    session = AsyncMock()
    session.get = AsyncMock(side_effect=[first, second, survivor])

    svc = CognitiveMemoryService(session)
    result = await svc.get_entity(first.id)

    assert result is survivor


@pytest.mark.asyncio
async def test_get_entity_returns_none_on_cycle(caplog):
    first = MemoryEntity(id=uuid.uuid4(), name="A", entity_type="person")
    second = MemoryEntity(id=uuid.uuid4(), name="B", entity_type="person")
    first.merged_into_id = second.id
    second.merged_into_id = first.id
    session = AsyncMock()
    session.get = AsyncMock(side_effect=[first, second])

    svc = CognitiveMemoryService(session)
    result = await svc.get_entity(first.id)

    assert result is None
    assert "Merge cycle detected" in caplog.text
