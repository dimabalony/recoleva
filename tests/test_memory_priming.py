"""Unit tests for Level 1 memory priming service."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai_memory.retrieval import memory_priming as _mp
from kai_memory.retrieval.memory_priming import (
    MemoryPrimingService,
    _compute_access_score,
    _compute_recency_score,
    _derive_embedding_query,
    _EVENT_CONTENT_SEPARATOR,
    _MAX_QUERY_CHARS,
    _format_priming_context,
    extract_entity_names,
)


@pytest.fixture(autouse=True)
def _reset_weights_cache():
    """Reset module-level retrieval weights cache between tests."""
    _mp._WEIGHTS_CACHE = None
    yield
    _mp._WEIGHTS_CACHE = None


# ── Entity name extraction tests ──────────────────────────


def test_extract_entity_names_capitalized():
    assert "Maria" in extract_entity_names("Let's ask Maria about dinner")


def test_extract_entity_names_cyrillic():
    names = extract_entity_names("спроси Марию o встрече")
    assert "Марию" in names


def test_extract_entity_names_multi_word():
    names = extract_entity_names("Talk to Maria Petrova tomorrow")
    assert "Maria Petrova" in names


def test_extract_entity_names_empty():
    assert extract_entity_names("hello there") == []


def test_extract_entity_names_deduplicates():
    names = extract_entity_names("Maria said Maria likes hiking")
    assert names.count("Maria") == 1


def test_extract_entity_names_multiple():
    names = extract_entity_names("Misha met Maria at Examplecorp")
    assert "Misha" in names
    assert "Maria" in names
    assert "Examplecorp" in names


# ── Scoring tests ─────────────────────────────────────────


def test_recency_score_today():
    now = datetime.now(UTC)
    assert _compute_recency_score(now, now) == pytest.approx(1.0, abs=0.01)


def test_recency_score_7_days():
    now = datetime.now(UTC)
    week_ago = now - timedelta(days=7)
    score = _compute_recency_score(week_ago, now)
    assert score == pytest.approx(0.5, abs=0.05)


def test_recency_score_30_days():
    now = datetime.now(UTC)
    month_ago = now - timedelta(days=30)
    score = _compute_recency_score(month_ago, now)
    assert score < 0.15


def test_access_score_zero():
    assert _compute_access_score(0) == 0


def test_access_score_increases_with_count():
    s1 = _compute_access_score(1)
    s5 = _compute_access_score(5)
    s20 = _compute_access_score(20)
    assert 0 < s1 < s5 < s20


def test_access_score_logarithmic():
    """Doubling access count should not double the score (logarithmic)."""
    s10 = _compute_access_score(10)
    s20 = _compute_access_score(20)
    assert s20 < s10 * 2


# ── Format tests ──────────────────────────────────────────


def test_format_empty():
    assert _format_priming_context([]) == ""


def test_format_with_type_prefix():
    now = datetime.now(UTC)
    trace = MagicMock()
    trace.memory_type = "fact"
    trace.content = "Misha works at Examplecorp"
    trace.created_at = now  # recent — no time label
    result = _format_priming_context([(trace, 0.8)], now=now)
    assert "[fact]" in result
    assert "Misha works at Examplecorp" in result


def test_format_intention_prefix():
    now = datetime.now(UTC)
    trace = MagicMock()
    trace.memory_type = "intention"
    trace.content = "Ask about his new job"
    trace.created_at = now
    result = _format_priming_context([(trace, 0.9)], now=now)
    assert "[intention — act on this]" in result


def test_format_header():
    now = datetime.now(UTC)
    trace = MagicMock()
    trace.memory_type = "fact"
    trace.content = "test"
    trace.created_at = now
    result = _format_priming_context([(trace, 0.8)], now=now)
    assert result.startswith("## Active Associations")


# ── Priming service tests ─────────────────────────────────


def _make_trace(
    content="test memory",
    memory_type="fact",
    salience=0.6,
    access_count=0,
    emotion_arousal=None,
    created_at=None,
    trace_id=None,
):
    t = MagicMock()
    t.id = trace_id or uuid.uuid4()
    t.content = content
    t.memory_type = memory_type
    t.salience = salience
    t.access_count = access_count
    t.emotion_arousal = emotion_arousal
    t.created_at = created_at or datetime.now(UTC)
    return t


@pytest.mark.asyncio
async def test_prime_returns_empty_on_empty_message():
    svc = MemoryPrimingService(AsyncMock())
    assert (await svc.prime(""))[0] == ""
    assert (await svc.prime("   "))[0] == ""


_WEIGHTS = {
    "semantic": 0.35,
    "graph": 0.25,
    "recency": 0.15,
    "access": 0.15,
    "arousal": 0.10,
}


@pytest.mark.asyncio
async def test_prime_rolls_back_session_on_semantic_timeout(monkeypatch):
    """When semantic search times out, the session must be rolled back.

    Reason: asyncio.wait_for cancels the inner coroutine, but asyncpg's
    cancel-in-flight leaves the SQLAlchemy session with an unrolled-back
    transaction. Without an explicit rollback, Paths 2-4 and downstream
    prompt_builder loaders all reuse the same session and cascade-fail
    with PendingRollbackError (8s2b). Burned us 3x in the week of
    2026-05-07 → 2026-05-13.
    """
    import asyncio as _asyncio

    monkeypatch.setattr(_mp, "_SEMANTIC_TIMEOUT_S", 0.05)
    session = AsyncMock()
    svc = MemoryPrimingService(session)

    async def slow_search(*_args, **_kwargs):
        await _asyncio.sleep(0.5)  # > 0.05s patched timeout
        return []

    with (
        patch.object(svc.memory_svc, "get_retrieval_weights", return_value=_WEIGHTS),
        patch.object(svc.memory_svc, "search_traces_raw", side_effect=slow_search),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[]),
    ):
        await svc.prime("hello there")

    session.rollback.assert_awaited()


@pytest.mark.asyncio
async def test_prime_rolls_back_session_on_semantic_error():
    """Same rollback contract as the timeout case, for the Exception branch.

    A query that fails for non-timeout reasons (asyncpg disconnect, syntax
    error, etc.) also leaves the session in a needs-rollback state.
    """
    session = AsyncMock()
    svc = MemoryPrimingService(session)

    async def failing_search(*_args, **_kwargs):
        raise RuntimeError("boom")

    with (
        patch.object(svc.memory_svc, "get_retrieval_weights", return_value=_WEIGHTS),
        patch.object(svc.memory_svc, "search_traces_raw", side_effect=failing_search),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[]),
    ):
        await svc.prime("hello there")

    session.rollback.assert_awaited()


@pytest.mark.asyncio
async def test_recover_session_falls_back_to_invalidate_when_rollback_raises():
    """If rollback itself raises (connection gone after cancel), the
    recovery helper must fall back to invalidate() so the session isn't
    left in a half-broken state for the next path.
    """
    session = AsyncMock()
    session.rollback.side_effect = RuntimeError("connection lost")
    svc = MemoryPrimingService(session)

    await svc._recover_session()

    session.rollback.assert_awaited()
    session.invalidate.assert_awaited()


@pytest.mark.asyncio
async def test_recover_session_swallows_invalidate_failure():
    """Even if both rollback and invalidate fail, _recover_session must
    not propagate — priming-path callers rely on it as best-effort
    cleanup, not as a guaranteed recovery primitive.
    """
    session = AsyncMock()
    session.rollback.side_effect = RuntimeError("rollback boom")
    session.invalidate.side_effect = RuntimeError("invalidate boom")
    svc = MemoryPrimingService(session)

    # Should not raise.
    await svc._recover_session()

    session.rollback.assert_awaited()
    session.invalidate.assert_awaited()


@pytest.mark.asyncio
async def test_prime_continues_paths_after_semantic_timeout(monkeypatch):
    """End-to-end: after a semantic timeout, Path 2 (entity-linked) still
    runs. This is the user-visible behaviour we want: the chat turn still
    builds a prompt and replies, just without the semantic candidates.
    """
    import asyncio as _asyncio

    monkeypatch.setattr(_mp, "_SEMANTIC_TIMEOUT_S", 0.05)
    session = AsyncMock()
    svc = MemoryPrimingService(session)

    async def slow_search(*_args, **_kwargs):
        await _asyncio.sleep(0.5)
        return []

    entity = MagicMock()
    entity.id = uuid.uuid4()
    linked_trace = _make_trace()

    with (
        patch.object(svc.memory_svc, "get_retrieval_weights", return_value=_WEIGHTS),
        patch.object(svc.memory_svc, "search_traces_raw", side_effect=slow_search),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[entity]) as find_mock,
        patch.object(svc.memory_svc, "get_entity_linked_traces", return_value=[linked_trace]) as linked_mock,
        patch.object(svc.memory_svc, "traverse_entity_graph", return_value=[]),
        patch.object(svc.memory_svc, "get_intention_traces", return_value=[]),
    ):
        await svc.prime("Hello Maria")

    # Both recovery and the post-timeout Path 2 call happened.
    session.rollback.assert_awaited()
    find_mock.assert_awaited()
    linked_mock.assert_awaited()


@pytest.mark.asyncio
async def test_prime_returns_empty_on_no_results():
    session = AsyncMock()
    svc = MemoryPrimingService(session)

    with (
        patch.object(
            svc.memory_svc,
            "get_retrieval_weights",
            return_value={
                "semantic": 0.35,
                "graph": 0.25,
                "recency": 0.15,
                "access": 0.15,
                "arousal": 0.10,
            },
        ),
        patch.object(svc.memory_svc, "search_traces_raw", return_value=[]),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[]),
    ):
        result, _ = await svc.prime("hello there")
    assert result == ""


@pytest.mark.asyncio
async def test_prime_semantic_search_called():
    session = AsyncMock()
    svc = MemoryPrimingService(session)
    trace = _make_trace()

    with (
        patch.object(
            svc.memory_svc,
            "get_retrieval_weights",
            return_value={
                "semantic": 0.35,
                "graph": 0.25,
                "recency": 0.15,
                "access": 0.15,
                "arousal": 0.10,
            },
        ),
        patch.object(svc.memory_svc, "search_traces_raw", return_value=[(trace, 0.8)]) as search_mock,
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[]),
        patch.object(svc.memory_svc, "record_access_batch"),
    ):
        result, _ = await svc.prime("anything")
    search_mock.assert_awaited_once()
    assert "test memory" in result


@pytest.mark.asyncio
async def test_prime_entity_linked_memories():
    session = AsyncMock()
    svc = MemoryPrimingService(session)

    entity = MagicMock()
    entity.id = uuid.uuid4()
    linked_trace = _make_trace(content="Maria likes hiking")

    with (
        patch.object(
            svc.memory_svc,
            "get_retrieval_weights",
            return_value={
                "semantic": 0.35,
                "graph": 0.25,
                "recency": 0.15,
                "access": 0.15,
                "arousal": 0.10,
            },
        ),
        patch.object(svc.memory_svc, "search_traces_raw", return_value=[]),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[entity]),
        patch.object(svc.memory_svc, "get_entity_linked_traces", return_value=[linked_trace]),
        patch.object(svc.memory_svc, "traverse_entity_graph", return_value=[]),
        patch.object(svc.memory_svc, "get_intention_traces", return_value=[]),
        patch.object(svc.memory_svc, "record_access_batch"),
    ):
        result, _ = await svc.prime("Let's talk about Maria")
    assert "Maria likes hiking" in result


@pytest.mark.asyncio
async def test_prime_graph_traversal_surfaces_connected():
    """Maria mentioned → partner_of Misha → Misha's memories surface."""
    session = AsyncMock()
    svc = MemoryPrimingService(session)

    maria = MagicMock()
    maria.id = uuid.uuid4()
    misha_id = uuid.uuid4()
    misha_trace = _make_trace(content="Misha works at Examplecorp")

    with (
        patch.object(
            svc.memory_svc,
            "get_retrieval_weights",
            return_value={
                "semantic": 0.35,
                "graph": 0.25,
                "recency": 0.15,
                "access": 0.15,
                "arousal": 0.10,
            },
        ),
        patch.object(svc.memory_svc, "search_traces_raw", return_value=[]),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[maria]),
        patch.object(svc.memory_svc, "get_entity_linked_traces", return_value=[]),
        patch.object(
            svc.memory_svc,
            "get_entity_linked_traces_with_entity",
            return_value=[
                (misha_trace, misha_id),
            ],
        ),
        patch.object(svc.memory_svc, "traverse_entity_graph", return_value=[(misha_id, 0.9)]),
        patch.object(svc.memory_svc, "get_intention_traces", return_value=[]),
        patch.object(svc.memory_svc, "record_access_batch"),
    ):
        result, _ = await svc.prime("does Maria live somewhere")
    assert "Misha works at Examplecorp" in result


@pytest.mark.asyncio
async def test_prime_intention_surfacing():
    session = AsyncMock()
    svc = MemoryPrimingService(session)

    entity = MagicMock()
    entity.id = uuid.uuid4()
    intention = _make_trace(
        content="Ask John about his new job",
        memory_type="intention",
    )

    with (
        patch.object(
            svc.memory_svc,
            "get_retrieval_weights",
            return_value={
                "semantic": 0.35,
                "graph": 0.25,
                "recency": 0.15,
                "access": 0.15,
                "arousal": 0.10,
            },
        ),
        patch.object(svc.memory_svc, "search_traces_raw", return_value=[]),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[entity]),
        patch.object(svc.memory_svc, "get_entity_linked_traces", return_value=[]),
        patch.object(svc.memory_svc, "traverse_entity_graph", return_value=[]),
        patch.object(svc.memory_svc, "get_intention_traces", return_value=[intention]),
        patch.object(svc.memory_svc, "record_access_batch"),
    ):
        result, _ = await svc.prime("John called me yesterday")
    assert "intention" in result
    assert "Ask John about his new job" in result


@pytest.mark.asyncio
async def test_prime_deduplicates_across_paths():
    """Same trace from semantic + entity should appear once."""
    session = AsyncMock()
    svc = MemoryPrimingService(session)

    shared_id = uuid.uuid4()
    trace = _make_trace(content="shared memory", trace_id=shared_id)
    entity = MagicMock()
    entity.id = uuid.uuid4()

    with (
        patch.object(
            svc.memory_svc,
            "get_retrieval_weights",
            return_value={
                "semantic": 0.35,
                "graph": 0.25,
                "recency": 0.15,
                "access": 0.15,
                "arousal": 0.10,
            },
        ),
        patch.object(svc.memory_svc, "search_traces_raw", return_value=[(trace, 0.8)]),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[entity]),
        patch.object(svc.memory_svc, "get_entity_linked_traces", return_value=[trace]),
        patch.object(svc.memory_svc, "traverse_entity_graph", return_value=[]),
        patch.object(svc.memory_svc, "get_intention_traces", return_value=[]),
        patch.object(svc.memory_svc, "record_access_batch"),
    ):
        result, _ = await svc.prime("Talk to Maria")
    # Should appear only once despite being found by two paths
    assert result.count("shared memory") == 1


@pytest.mark.asyncio
async def test_prime_uses_retrieval_config_weights():
    """Must read weights from DB, not use hardcoded values."""
    session = AsyncMock()
    svc = MemoryPrimingService(session)
    trace = _make_trace()

    with (
        patch.object(svc.memory_svc, "get_retrieval_weights") as weights_mock,
        patch.object(svc.memory_svc, "search_traces_raw", return_value=[(trace, 0.8)]),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[]),
        patch.object(svc.memory_svc, "record_access_batch"),
    ):
        weights_mock.return_value = {
            "semantic": 0.5,
            "graph": 0.2,
            "recency": 0.1,
            "access": 0.1,
            "arousal": 0.1,
        }
        await svc.prime("test message long enough")
    weights_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_prime_recency_boosts_recent():
    """Today's memory should score higher than last month's."""
    session = AsyncMock()
    svc = MemoryPrimingService(session)

    now = datetime.now(UTC)
    recent = _make_trace(content="recent", created_at=now, salience=0.5, access_count=0)
    old = _make_trace(content="old", created_at=now - timedelta(days=30), salience=0.5, access_count=0)

    # Both have same semantic score — recency should differentiate
    with (
        patch.object(
            svc.memory_svc,
            "get_retrieval_weights",
            return_value={
                "semantic": 0.0,
                "graph": 0.0,
                "recency": 1.0,
                "access": 0.0,
                "arousal": 0.0,
            },
        ),
        patch.object(svc.memory_svc, "search_traces_raw", return_value=[(recent, 0.5), (old, 0.5)]),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[]),
        patch.object(svc.memory_svc, "record_access_batch"),
    ):
        result, _ = await svc.prime("anything")
    # Recent should be listed first
    assert result.index("recent") < result.index("old")


@pytest.mark.asyncio
async def test_prime_arousal_boosts_emotional():
    session = AsyncMock()
    svc = MemoryPrimingService(session)

    calm = _make_trace(content="calm", emotion_arousal=0.1)
    intense = _make_trace(content="intense", emotion_arousal=0.9)

    with (
        patch.object(
            svc.memory_svc,
            "get_retrieval_weights",
            return_value={
                "semantic": 0.0,
                "graph": 0.0,
                "recency": 0.0,
                "access": 0.0,
                "arousal": 1.0,
            },
        ),
        patch.object(svc.memory_svc, "search_traces_raw", return_value=[(calm, 0.5), (intense, 0.5)]),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[]),
        patch.object(svc.memory_svc, "record_access_batch"),
    ):
        result, _ = await svc.prime("anything")
    assert result.index("intense") < result.index("calm")


@pytest.mark.asyncio
async def test_prime_respects_limit():
    session = AsyncMock()
    svc = MemoryPrimingService(session)

    traces = [(_make_trace(content=f"memory {i}"), 0.8 - i * 0.01) for i in range(20)]

    with (
        patch.object(
            svc.memory_svc,
            "get_retrieval_weights",
            return_value={
                "semantic": 0.35,
                "graph": 0.25,
                "recency": 0.15,
                "access": 0.15,
                "arousal": 0.10,
            },
        ),
        patch.object(svc.memory_svc, "search_traces_raw", return_value=traces),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[]),
        patch.object(svc.memory_svc, "record_access_batch"),
    ):
        result, _ = await svc.prime("anything", limit=5)
    # Count actual memory lines (excluding header)
    lines = [line for line in result.split("\n") if line.startswith("- [")]
    assert len(lines) == 5


@pytest.mark.asyncio
async def test_prime_records_access():
    session = AsyncMock()
    svc = MemoryPrimingService(session)
    trace = _make_trace()

    with (
        patch.object(
            svc.memory_svc,
            "get_retrieval_weights",
            return_value={
                "semantic": 0.35,
                "graph": 0.25,
                "recency": 0.15,
                "access": 0.15,
                "arousal": 0.10,
            },
        ),
        patch.object(svc.memory_svc, "search_traces_raw", return_value=[(trace, 0.8)]),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[]),
        patch.object(svc.memory_svc, "record_access_batch") as access_mock,
    ):
        await svc.prime("anything")
    access_mock.assert_awaited_once_with([trace.id])


@pytest.mark.asyncio
async def test_prime_access_failure_non_fatal():
    """record_access failing should not crash priming."""
    session = AsyncMock()
    svc = MemoryPrimingService(session)
    trace = _make_trace()

    with (
        patch.object(
            svc.memory_svc,
            "get_retrieval_weights",
            return_value={
                "semantic": 0.35,
                "graph": 0.25,
                "recency": 0.15,
                "access": 0.15,
                "arousal": 0.10,
            },
        ),
        patch.object(svc.memory_svc, "search_traces_raw", return_value=[(trace, 0.8)]),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[]),
        patch.object(svc.memory_svc, "record_access_batch", side_effect=RuntimeError("DB")),
    ):
        result, _ = await svc.prime("anything")
    assert "test memory" in result  # still returns results despite access failure


# ── Error handling tests ──────────────────────────────────


@pytest.mark.asyncio
async def test_prime_semantic_search_failure_falls_back():
    """Semantic search failing should not prevent entity-based results."""
    session = AsyncMock()
    svc = MemoryPrimingService(session)

    entity = MagicMock()
    entity.id = uuid.uuid4()
    trace = _make_trace(content="entity-linked memory")

    with (
        patch.object(
            svc.memory_svc,
            "get_retrieval_weights",
            return_value={
                "semantic": 0.35,
                "graph": 0.25,
                "recency": 0.15,
                "access": 0.15,
                "arousal": 0.10,
            },
        ),
        patch.object(svc.memory_svc, "search_traces_raw", side_effect=RuntimeError("embed fail")),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[entity]),
        patch.object(svc.memory_svc, "get_entity_linked_traces", return_value=[trace]),
        patch.object(svc.memory_svc, "traverse_entity_graph", return_value=[]),
        patch.object(svc.memory_svc, "get_intention_traces", return_value=[]),
        patch.object(svc.memory_svc, "record_access_batch"),
    ):
        result, _ = await svc.prime("Maria is here")
    assert "entity-linked memory" in result


@pytest.mark.asyncio
async def test_prime_graph_traversal_failure_falls_back():
    session = AsyncMock()
    svc = MemoryPrimingService(session)

    entity = MagicMock()
    entity.id = uuid.uuid4()
    trace = _make_trace(content="direct link")

    with (
        patch.object(
            svc.memory_svc,
            "get_retrieval_weights",
            return_value={
                "semantic": 0.35,
                "graph": 0.25,
                "recency": 0.15,
                "access": 0.15,
                "arousal": 0.10,
            },
        ),
        patch.object(svc.memory_svc, "search_traces_raw", return_value=[]),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[entity]),
        patch.object(svc.memory_svc, "get_entity_linked_traces", return_value=[trace]),
        patch.object(svc.memory_svc, "traverse_entity_graph", side_effect=RuntimeError("CTE fail")),
        patch.object(svc.memory_svc, "get_intention_traces", return_value=[]),
        patch.object(svc.memory_svc, "record_access_batch"),
    ):
        result, _ = await svc.prime("Maria called")
    assert "direct link" in result


@pytest.mark.asyncio
async def test_prime_total_failure_returns_empty():
    session = AsyncMock()
    svc = MemoryPrimingService(session)

    with (
        patch.object(svc.memory_svc, "get_retrieval_weights", side_effect=RuntimeError("DB down")),
        patch.object(svc.memory_svc, "search_traces_raw", side_effect=RuntimeError("fail")),
        patch.object(svc.memory_svc, "find_entities_by_names", side_effect=RuntimeError("fail")),
    ):
        result, _ = await svc.prime("anything")
    assert result == ""


# ── Graph traversal unit tests ────────────────────────────


@pytest.mark.asyncio
async def test_graph_traversal_cycle_detection():
    """A↔B cycle must not infinite-loop — CTE uses visited array."""
    session = AsyncMock()
    svc = MemoryPrimingService(session)

    # The SQL uses visited array for cycle detection.
    # We verify the SQL params include cycle-safe structure by checking
    # that the method completes without hanging.
    mock_result = MagicMock()
    mock_result.__iter__ = MagicMock(return_value=iter([]))
    svc.memory_svc.session.execute = AsyncMock(return_value=mock_result)

    result = await svc.memory_svc.traverse_entity_graph([uuid.uuid4()], max_depth=3)
    assert isinstance(result, list)
    # Verify the SQL contains visited array cycle detection
    call_args = svc.memory_svc.session.execute.call_args
    sql_text = str(call_args[0][0])
    assert "visited" in sql_text.lower()
    assert "ANY" in sql_text


@pytest.mark.asyncio
async def test_graph_traversal_depth_limit():
    """CTE must use :max_depth parameter to limit traversal."""
    session = AsyncMock()
    svc = MemoryPrimingService(session)

    mock_result = MagicMock()
    mock_result.__iter__ = MagicMock(return_value=iter([]))
    svc.memory_svc.session.execute = AsyncMock(return_value=mock_result)

    await svc.memory_svc.traverse_entity_graph([uuid.uuid4()], max_depth=2)

    call_args = svc.memory_svc.session.execute.call_args
    params = call_args[0][1]
    assert params["max_depth"] == 2


@pytest.mark.asyncio
async def test_graph_traversal_strength_weighted():
    """CTE energy formula must include er.confidence for strength weighting."""
    session = AsyncMock()
    svc = MemoryPrimingService(session)

    mock_result = MagicMock()
    mock_result.__iter__ = MagicMock(return_value=iter([]))
    svc.memory_svc.session.execute = AsyncMock(return_value=mock_result)

    await svc.memory_svc.traverse_entity_graph([uuid.uuid4()])

    call_args = svc.memory_svc.session.execute.call_args
    sql_text = str(call_args[0][0])
    assert "confidence" in sql_text
    assert "mention_count" in sql_text  # mention-aware decay factor


@pytest.mark.asyncio
async def test_graph_traversal_ignores_expired():
    """CTE must filter on valid_until IS NULL to skip expired relationships."""
    session = AsyncMock()
    svc = MemoryPrimingService(session)

    mock_result = MagicMock()
    mock_result.__iter__ = MagicMock(return_value=iter([]))
    svc.memory_svc.session.execute = AsyncMock(return_value=mock_result)

    await svc.memory_svc.traverse_entity_graph([uuid.uuid4()])

    call_args = svc.memory_svc.session.execute.call_args
    sql_text = str(call_args[0][0])
    assert "valid_until IS NULL" in sql_text


@pytest.mark.asyncio
async def test_graph_traversal_min_confidence():
    """CTE must filter on min_confidence parameter."""
    session = AsyncMock()
    svc = MemoryPrimingService(session)

    mock_result = MagicMock()
    mock_result.__iter__ = MagicMock(return_value=iter([]))
    svc.memory_svc.session.execute = AsyncMock(return_value=mock_result)

    await svc.memory_svc.traverse_entity_graph([uuid.uuid4()], min_confidence=0.5)

    call_args = svc.memory_svc.session.execute.call_args
    params = call_args[0][1]
    assert params["min_confidence"] == 0.5
    sql_text = str(call_args[0][0])
    assert "min_confidence" in sql_text


@pytest.mark.asyncio
async def test_graph_traversal_mention_count_affects_energy():
    """CTE must include mention_count factor in energy propagation.

    Verifies the SQL joins memory_entities as me_target and uses
    mention_count in the energy formula. An entity with mention_count=10+
    propagates more energy than one with mention_count=0 (same confidence).
    """
    session = AsyncMock()
    svc = MemoryPrimingService(session)

    mock_result = MagicMock()
    mock_result.__iter__ = MagicMock(return_value=iter([]))
    svc.memory_svc.session.execute = AsyncMock(return_value=mock_result)

    await svc.memory_svc.traverse_entity_graph([uuid.uuid4()])

    call_args = svc.memory_svc.session.execute.call_args
    sql_text = str(call_args[0][0])
    # Mention-count aware propagation: joins target entity, uses mention_count
    assert "me_target" in sql_text
    assert "mention_count" in sql_text
    assert "LN" in sql_text or "ln" in sql_text.lower()
    # Factor range [0.4, 0.6]
    assert "0.4" in sql_text
    assert "0.2" in sql_text


# ── Access frequency scoring test ─────────────────────────


@pytest.mark.asyncio
async def test_prime_access_frequency_boosts_popular():
    """Trace with higher access_count should rank higher when other factors equal."""
    session = AsyncMock()
    svc = MemoryPrimingService(session)

    now = datetime.now(UTC)
    popular = _make_trace(content="popular", access_count=20, created_at=now)
    unpopular = _make_trace(content="unpopular", access_count=0, created_at=now)

    with (
        patch.object(
            svc.memory_svc,
            "get_retrieval_weights",
            return_value={
                "semantic": 0.0,
                "graph": 0.0,
                "recency": 0.0,
                "access": 1.0,
                "arousal": 0.0,
            },
        ),
        patch.object(
            svc.memory_svc,
            "search_traces_raw",
            return_value=[
                (popular, 0.5),
                (unpopular, 0.5),
            ],
        ),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[]),
        patch.object(svc.memory_svc, "record_access_batch"),
    ):
        result, _ = await svc.prime("anything")
    assert result.index("popular") < result.index("unpopular")


# ── Stop word filter tests ────────────────────────────────


def test_extract_entity_names_filters_stop_words():
    """Common sentence starters should be filtered out."""
    names = extract_entity_names("Where does Misha work?")
    assert "Misha" in names
    assert "Where" not in names


def test_extract_entity_names_filters_russian_stop_words():
    names = extract_entity_names("Как Марии дела?")  # "Как" is a stop word
    assert "Как" not in names
    assert "Марии" in names


# ── Task 6b: lowercase entity alias fallback ──────────────


@pytest.mark.asyncio
async def test_lowercase_fallback_when_no_capitalized_entities():
    """v2a §9 Task 6b: lowercase message tokens scan entity names + aliases."""
    session = AsyncMock()
    svc = MemoryPrimingService(session)

    entity = MagicMock()
    entity.id = uuid.uuid4()
    linked_trace = _make_trace(content="dog detail")

    with (
        patch.object(
            svc.memory_svc,
            "get_retrieval_weights",
            return_value={
                "semantic": 0.35,
                "graph": 0.25,
                "recency": 0.15,
                "access": 0.15,
                "arousal": 0.10,
            },
        ),
        patch.object(svc.memory_svc, "search_traces_raw", return_value=[]),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[]) as cap_mock,
        patch.object(
            svc.memory_svc,
            "find_entities_by_lowercase_tokens",
            return_value=[entity],
        ) as lower_mock,
        patch.object(svc.memory_svc, "get_entity_linked_traces", return_value=[linked_trace]),
        patch.object(svc.memory_svc, "traverse_entity_graph", return_value=[]),
        patch.object(svc.memory_svc, "get_intention_traces", return_value=[]),
        patch.object(svc.memory_svc, "record_access_batch"),
    ):
        # Lowercase-only message — capitalized regex returns nothing.
        result, _ = await svc.prime("got the dog vaccinated yesterday")

    # Capitalized lookup not called (entity_names was empty).
    cap_mock.assert_not_called()
    lower_mock.assert_awaited_once()
    # The fallback's tokens were lowercased and length-filtered.
    args, _kwargs = lower_mock.call_args
    tokens = args[0]
    assert all(t == t.lower() for t in tokens)
    assert all(len(t) >= 3 for t in tokens)
    assert "dog" in tokens or "vaccinated" in tokens or "yesterday" in tokens
    assert "dog detail" in result


@pytest.mark.asyncio
async def test_lowercase_fallback_skipped_when_capitalized_present():
    """When the regex finds capitalized names, the lowercase fallback does NOT run."""
    session = AsyncMock()
    svc = MemoryPrimingService(session)

    entity = MagicMock()
    entity.id = uuid.uuid4()
    linked_trace = _make_trace()

    with (
        patch.object(
            svc.memory_svc,
            "get_retrieval_weights",
            return_value={
                "semantic": 0.35,
                "graph": 0.25,
                "recency": 0.15,
                "access": 0.15,
                "arousal": 0.10,
            },
        ),
        patch.object(svc.memory_svc, "search_traces_raw", return_value=[]),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[entity]),
        patch.object(svc.memory_svc, "find_entities_by_lowercase_tokens", return_value=[]) as lower_mock,
        patch.object(svc.memory_svc, "get_entity_linked_traces", return_value=[linked_trace]),
        patch.object(svc.memory_svc, "traverse_entity_graph", return_value=[]),
        patch.object(svc.memory_svc, "get_intention_traces", return_value=[]),
        patch.object(svc.memory_svc, "record_access_batch"),
    ):
        await svc.prime("Maria came over")

    lower_mock.assert_not_called()


# ── Task 7: Temporal grounding ────────────────────────────


def test_format_trace_entry_recent_no_label():
    """v2a §6.2: recent traces (within min_days) get no relative label."""
    from kai_memory.retrieval.memory_priming import _format_trace_entry

    now = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)
    trace = _make_trace(content="dog vaccinated")
    trace.created_at = now - timedelta(hours=3)

    out = _format_trace_entry(trace, now)
    assert "remembered" not in out
    assert "[fact]" in out
    assert "dog vaccinated" in out


def test_format_trace_entry_yesterday_label():
    from kai_memory.retrieval.memory_priming import _format_trace_entry

    now = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)
    trace = _make_trace()
    trace.created_at = now - timedelta(hours=30)

    out = _format_trace_entry(trace, now)
    assert "remembered yesterday" in out


def test_format_trace_entry_days_label():
    from kai_memory.retrieval.memory_priming import _format_trace_entry

    now = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)
    trace = _make_trace()
    trace.created_at = now - timedelta(days=3)

    out = _format_trace_entry(trace, now)
    assert "remembered 3d ago" in out


def test_format_trace_entry_weeks_label():
    from kai_memory.retrieval.memory_priming import _format_trace_entry

    now = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)
    trace = _make_trace()
    trace.created_at = now - timedelta(days=14)

    out = _format_trace_entry(trace, now)
    assert "remembered 2w ago" in out


def test_format_trace_entry_months_label():
    from kai_memory.retrieval.memory_priming import _format_trace_entry

    now = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)
    trace = _make_trace()
    trace.created_at = now - timedelta(days=180)

    out = _format_trace_entry(trace, now)
    assert "remembered 6mo ago" in out


def test_format_trace_entry_years_label():
    from kai_memory.retrieval.memory_priming import _format_trace_entry

    now = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)
    trace = _make_trace()
    trace.created_at = now - timedelta(days=730)

    out = _format_trace_entry(trace, now)
    assert "remembered 2y ago" in out


def test_format_trace_entry_future_no_label():
    """Future timestamps (clock skew) → no label, no crash."""
    from kai_memory.retrieval.memory_priming import _format_trace_entry

    now = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)
    trace = _make_trace()
    trace.created_at = now + timedelta(hours=2)

    out = _format_trace_entry(trace, now)
    assert "remembered" not in out


def test_format_trace_entry_naive_timestamp_treated_as_utc():
    """Naive timestamps treated as UTC, not crash."""
    from kai_memory.retrieval.memory_priming import _format_trace_entry

    now = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)
    trace = _make_trace()
    trace.created_at = (now - timedelta(days=3)).replace(tzinfo=None)

    out = _format_trace_entry(trace, now)
    assert "remembered 3d ago" in out


def test_format_trace_entry_intention_keeps_action_marker():
    from kai_memory.retrieval.memory_priming import _format_trace_entry

    now = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)
    trace = _make_trace()
    trace.memory_type = "intention"
    trace.created_at = now - timedelta(days=3)

    out = _format_trace_entry(trace, now)
    assert "act on this" in out
    assert "remembered 3d ago" in out


# ── Task 7b: Structured priming log emission ─────────────


@pytest.mark.asyncio
async def test_prime_emits_structured_retrieve_complete_log(caplog):
    """v2a §8.3: prime() emits memory.retrieve.complete JSON per telemetry contract."""
    import logging

    from memory_test_support import ChatContext

    session = AsyncMock()
    svc = MemoryPrimingService(session)

    trace = _make_trace(content="dog visited vet")

    chat_ctx = ChatContext(chat_id=-1009000000001, topic_id=1506, venue="telegram")

    with (
        caplog.at_level(logging.INFO, logger="app.services.chat.memory_priming"),
        patch.object(
            svc.memory_svc,
            "get_retrieval_weights",
            return_value={
                "semantic": 0.35,
                "graph": 0.25,
                "recency": 0.15,
                "access": 0.15,
                "arousal": 0.10,
            },
        ),
        patch.object(svc.memory_svc, "search_traces_raw", return_value=[(trace, 0.84)]),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[]),
        patch.object(svc.memory_svc, "record_access_batch"),
    ):
        await svc.prime("Maria came over", chat_ctx=chat_ctx)

    # Hotfix 2026-05-01: payload is now JSON-encoded into the message body
    # because Python's default formatter drops `extra={}` keys.
    import json as _json

    matching = []
    for r in caplog.records:
        msg = r.getMessage()
        prefix = "memory.retrieve.complete "
        if msg.startswith(prefix):
            matching.append(_json.loads(msg[len(prefix) :]))
    assert matching, "expected memory.retrieve.complete log record"
    rec = matching[0]

    # Required top-level fields per backend-docs/ai-workforce/observability/memory-v2-telemetry.md
    assert rec["chat_id"] == -1009000000001
    assert rec["topic_id"] == 1506
    assert rec["venue"] == "telegram"
    assert rec["intent"] is None  # M1 has no query-type routing
    assert isinstance(rec["candidates_total"], int)
    assert isinstance(rec["returned"], int)
    assert isinstance(rec["duration_ms"], int)
    assert rec["score_floor_applied"] is False
    assert isinstance(rec["primed"], list)
    assert rec["primed"], "primed list should be non-empty when traces returned"

    entry = rec["primed"][0]
    assert "trace_id" in entry
    assert isinstance(entry["trace_id"], str)
    assert entry["memory_type"] == "fact"
    assert entry["rank"] == 1
    assert isinstance(entry["score"], float)

    # score_breakdown — M1 active components + M3 placeholders
    sb = entry["score_breakdown"]
    for key in (
        "semantic",
        "graph",
        "recency",
        "access",
        "arousal",
        "salience_observed",
        "memory_type_bias",
        "anaphoric_recency_multiplier",
        "diversity_penalty",
        "inhibition_penalty",
    ):
        assert key in sb, f"missing score_breakdown key: {key}"

    # M3 placeholders are 0 / 1.0 in M1
    assert sb["memory_type_bias"] == 0.0
    assert sb["diversity_penalty"] == 0.0
    assert sb["inhibition_penalty"] == 0.0
    assert sb["anaphoric_recency_multiplier"] == 1.0

    # path_attribution mirrors which paths surfaced this trace
    assert "path_attribution" in entry
    assert isinstance(entry["path_attribution"], list)
    assert "semantic" in entry["path_attribution"]


@pytest.mark.asyncio
async def test_prime_log_omits_chat_context_when_unavailable(caplog):
    """No chat_ctx → chat_id/topic_id/venue are None."""
    import logging

    session = AsyncMock()
    svc = MemoryPrimingService(session)

    trace = _make_trace()

    with (
        caplog.at_level(logging.INFO, logger="app.services.chat.memory_priming"),
        patch.object(
            svc.memory_svc,
            "get_retrieval_weights",
            return_value={
                "semantic": 0.35,
                "graph": 0.25,
                "recency": 0.15,
                "access": 0.15,
                "arousal": 0.10,
            },
        ),
        patch.object(svc.memory_svc, "search_traces_raw", return_value=[(trace, 0.5)]),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[]),
        patch.object(svc.memory_svc, "record_access_batch"),
    ):
        await svc.prime("anything", chat_ctx=None)

    import json as _json

    matching = []
    for r in caplog.records:
        msg = r.getMessage()
        prefix = "memory.retrieve.complete "
        if msg.startswith(prefix):
            matching.append(_json.loads(msg[len(prefix) :]))
    assert matching
    rec = matching[0]
    assert rec["chat_id"] is None
    assert rec["topic_id"] is None
    assert rec["venue"] is None


@pytest.mark.asyncio
async def test_prime_log_records_path_attribution_correctly(caplog):
    """v2a §8.3: path_attribution lists every retrieval path that surfaced a trace."""
    import logging

    session = AsyncMock()
    svc = MemoryPrimingService(session)

    trace = _make_trace(content="dog detail")
    entity = MagicMock()
    entity.id = uuid.uuid4()

    with (
        caplog.at_level(logging.INFO, logger="app.services.chat.memory_priming"),
        patch.object(
            svc.memory_svc,
            "get_retrieval_weights",
            return_value={
                "semantic": 0.35,
                "graph": 0.25,
                "recency": 0.15,
                "access": 0.15,
                "arousal": 0.10,
            },
        ),
        patch.object(svc.memory_svc, "search_traces_raw", return_value=[(trace, 0.7)]),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[entity]),
        patch.object(svc.memory_svc, "get_entity_linked_traces", return_value=[trace]),
        patch.object(svc.memory_svc, "traverse_entity_graph", return_value=[]),
        patch.object(svc.memory_svc, "get_intention_traces", return_value=[]),
        patch.object(svc.memory_svc, "record_access_batch"),
    ):
        await svc.prime("Maria visit")

    import json as _json

    matching = []
    for r in caplog.records:
        msg = r.getMessage()
        prefix = "memory.retrieve.complete "
        if msg.startswith(prefix):
            matching.append(_json.loads(msg[len(prefix) :]))
    assert matching
    rec = matching[0]
    paths = set(rec["primed"][0]["path_attribution"])
    assert "semantic" in paths
    assert "entity_link" in paths


# ── review-r1 fix 3: reuse _CYRILLIC_STOPWORDS in lowercase tokenizer ────


def test_lowercase_tokens_filters_cyrillic_function_words():
    """Both reviews flag: function words like 'или', 'наш', 'она', 'они',
    'его', 'тот' are NOT in `_STOP_WORDS` but ARE in `_CYRILLIC_STOPWORDS`
    in romanization.py. Without reusing them, every Russian message
    triggers spurious DB lookups.
    """
    from kai_memory.retrieval.memory_priming import _extract_lowercase_tokens

    msg = "она и он встретили его в парке или там"
    tokens = _extract_lowercase_tokens(msg)
    # All Cyrillic stopwords must be filtered out
    for stop in ("она", "или", "его", "там", "она", "наш", "они"):
        assert stop not in tokens, f"stopword {stop!r} leaked through"


def test_lowercase_tokens_keeps_legit_content_words():
    """Sanity: stopword filtering doesn't drop genuine content."""
    from kai_memory.retrieval.memory_priming import _extract_lowercase_tokens

    msg = "луна побежала за нора"
    tokens = _extract_lowercase_tokens(msg)
    assert "луна" in tokens
    assert "нора" in tokens
    assert "побежала" in tokens


# ── review-r1 fix 8: NaN/Infinity guard in _build_primed_payload ────


def test_finite_helper_coerces_non_finite_to_zero():
    """Pure helper test — keeps the contract local."""
    import math

    from kai_memory.retrieval.memory_priming import _finite

    assert _finite(0.84) == 0.84
    assert _finite(0) == 0.0
    assert _finite(None) == 0.0
    assert _finite(float("nan")) == 0.0
    assert _finite(float("inf")) == 0.0
    assert _finite(float("-inf")) == 0.0
    # Sanity: regular finite values pass through, including 0.0.
    assert math.isfinite(_finite(1e-300))


def test_build_primed_payload_strips_non_finite_values():
    """End-to-end: a candidate with NaN/Inf components emits 0.0 instead."""
    paths = {"abc": {"semantic"}}
    trace = MagicMock()
    trace.id = "abc"
    trace.memory_type = "fact"
    trace.salience = float("nan")
    components_with_nan = {
        "semantic": float("inf"),
        "graph": 0.0,
        "recency": 0.0,
        "access": float("nan"),
        "arousal": 0.0,
    }
    payload = MemoryPrimingService._build_primed_payload([(trace, float("nan"), components_with_nan)], paths)
    entry = payload[0]
    assert entry["score"] == 0.0
    sb = entry["score_breakdown"]
    assert sb["semantic"] == 0.0
    assert sb["access"] == 0.0
    assert sb["salience_observed"] == 0.0
    # Sane components still pass through unchanged.
    assert sb["graph"] == 0.0
    assert sb["recency"] == 0.0
    assert sb["arousal"] == 0.0


# ── hotfix 2026-05-01: structured payload survives Python's default formatter ──


@pytest.mark.asyncio
async def test_retrieve_complete_payload_survives_default_formatter(caplog):
    """Regression guard: the M1 telemetry contract requires the structured
    payload to reach CloudWatch. Python's default formatter
    (`%(levelname)s:%(name)s:%(message)s`, used by basicConfig) drops
    `extra={}` keys silently, so the payload must be inline-serialized
    into the message body. Render the LogRecord through that exact
    format string and assert the JSON survives.
    """
    import json as _json
    import logging

    from memory_test_support import ChatContext

    session = AsyncMock()
    svc = MemoryPrimingService(session)
    trace = _make_trace(content="x")
    chat_ctx = ChatContext(chat_id=42, topic_id=None, venue="telegram")

    with (
        caplog.at_level(logging.INFO, logger="app.services.chat.memory_priming"),
        patch.object(
            svc.memory_svc,
            "get_retrieval_weights",
            return_value={"semantic": 0.35, "graph": 0.25, "recency": 0.15, "access": 0.15, "arousal": 0.10},
        ),
        patch.object(svc.memory_svc, "search_traces_raw", return_value=[(trace, 0.5)]),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[]),
        patch.object(svc.memory_svc, "record_access_batch"),
    ):
        await svc.prime("hello there", chat_ctx=chat_ctx)

    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    rendered = next(
        (formatter.format(r) for r in caplog.records if r.getMessage().startswith("memory.retrieve.complete ")),
        None,
    )
    assert rendered is not None, "no memory.retrieve.complete log record produced"

    # The rendered message must contain a JSON object with chat_id + primed.
    body = rendered.split("memory.retrieve.complete ", 1)[1]
    payload = _json.loads(body)
    assert payload["chat_id"] == 42
    assert payload["venue"] == "telegram"
    assert "primed" in payload


# ── _derive_embedding_query tests ─────────────────────────


def test_derive_embedding_query_event_message_uses_header_only():
    """Event messages embed the YAML-ish header, NOT the full content.

    Background: PR #136's diagnostic caught a 45kB Fireflies meeting
    transcript being shipped to the embedder. The fix is upstream: for
    ``[EVENT]`` messages, embed only the header (title, summary,
    participants) — that's where the semantic anchor lives anyway.
    """
    header = "[EVENT] meeting\ntitle: Standup 2026-05-15\nparticipants: [Alice, Bob]\nsummary: discussed roadmap"
    transcript = "very long meeting transcript " * 2000
    full_message = header + _EVENT_CONTENT_SEPARATOR + transcript

    result = _derive_embedding_query(full_message)
    assert result == header
    assert transcript not in result


def test_derive_embedding_query_non_event_truncated_to_max_chars():
    """Non-event messages get char-bounded as a soft cap; the hard
    byte-bound still runs inside ``embeddings.py``.
    """
    long_message = "x" * (_MAX_QUERY_CHARS + 1000)
    result = _derive_embedding_query(long_message)
    assert len(result) == _MAX_QUERY_CHARS


def test_derive_embedding_query_short_message_passes_through():
    result = _derive_embedding_query("what's on for tomorrow?")
    assert result == "what's on for tomorrow?"


def test_derive_embedding_query_event_without_separator_still_truncated():
    """A message starting with ``[EVENT]`` but missing the separator falls
    through to the default truncation path (defensive — should never
    happen in practice but the code must not crash).
    """
    message = "[EVENT] something without separator " + "y" * 5000
    result = _derive_embedding_query(message)
    assert len(result) == _MAX_QUERY_CHARS


def test_derive_embedding_query_event_header_bounded_by_max_chars():
    """A pathologically long ``[EVENT]`` header is itself bounded so a
    bug in the header builder can't accidentally re-create the original
    issue.
    """
    header = "[EVENT] " + "h" * (_MAX_QUERY_CHARS * 2)
    message = header + _EVENT_CONTENT_SEPARATOR + "transcript"
    result = _derive_embedding_query(message)
    assert len(result) == _MAX_QUERY_CHARS


# ── A2: salience as the 6th priming component (default-inert) ──────────────

_LEGACY_5 = ("semantic", "graph", "recency", "access", "arousal")


def test_default_retrieval_weights_salience_is_zero():
    from kai_memory.store.cognitive_memory_constants import DEFAULT_RETRIEVAL_WEIGHTS

    assert DEFAULT_RETRIEVAL_WEIGHTS["salience"] == 0.0
    # adding the 0.0 salience weight doesn't change the total (still ~1.0 over the 5)
    assert sum(DEFAULT_RETRIEVAL_WEIGHTS.values()) == sum(
        v for k, v in DEFAULT_RETRIEVAL_WEIGHTS.items() if k != "salience"
    )


def test_weighted_component_score_salience_zero_is_bit_identical():
    from kai_memory.retrieval.memory_priming import weighted_component_score

    comps = {"semantic": 0.8, "graph": 0.4, "recency": 0.6, "access": 0.3, "arousal": 0.2, "salience": 0.9}
    weights = {"semantic": 0.35, "graph": 0.25, "recency": 0.15, "access": 0.15, "arousal": 0.10, "salience": 0.0}
    legacy = sum(comps[k] * weights[k] for k in _LEGACY_5)
    # EXACT equality — salience * 0.0 == 0 adds nothing to the sum
    assert weighted_component_score(comps, weights) == legacy
    # also tolerant of a weights dict that PREDATES salience (cached 5-key dict)
    legacy_weights = {k: weights[k] for k in _LEGACY_5}
    assert weighted_component_score(comps, legacy_weights) == legacy


def test_weighted_component_score_salience_positive_adds_term():
    from kai_memory.retrieval.memory_priming import weighted_component_score

    comps = {"semantic": 0.8, "graph": 0.0, "recency": 0.0, "access": 0.0, "arousal": 0.0, "salience": 1.0}
    weights = {"semantic": 0.5, "graph": 0.0, "recency": 0.0, "access": 0.0, "arousal": 0.0, "salience": 0.5}
    assert weighted_component_score(comps, weights) == pytest.approx(0.8 * 0.5 + 1.0 * 0.5)


def test_compute_salience_score_clamps_and_coerces():
    from kai_memory.retrieval.memory_priming import _compute_salience_score

    assert _compute_salience_score(_make_trace(salience=0.7)) == pytest.approx(0.7)
    assert _compute_salience_score(_make_trace(salience=1.9)) == 1.0  # clamped to [0,1]
    assert _compute_salience_score(_make_trace(salience=-0.3)) == 0.0
    assert _compute_salience_score(_make_trace(salience=None)) == 0.0  # None → 0.0
    assert _compute_salience_score(_make_trace(salience=float("nan"))) == 0.0  # NaN scrubbed


def test_score_candidates_salience_zero_matches_legacy_5_component():
    # The regression guard: with salience weight 0.0 the score is EXACTLY the
    # pre-A2 5-component weighted sum for a fixed candidate.
    svc = MemoryPrimingService(AsyncMock())
    trace = _make_trace(
        salience=0.9, access_count=3, emotion_arousal=0.4, created_at=datetime.now(UTC) - timedelta(days=5)
    )
    weights = {"semantic": 0.35, "graph": 0.25, "recency": 0.15, "access": 0.15, "arousal": 0.10, "salience": 0.0}
    candidates = {str(trace.id): (trace, {"semantic": 0.8, "graph": 0.3})}
    [(_t, score, full)] = svc._score_candidates(candidates, weights, limit=10)
    expected = sum(full[k] * weights[k] for k in _LEGACY_5)  # salience NOT included
    assert score == expected  # EXACT — default-inert
    assert full["salience"] == pytest.approx(0.9)  # the component is still computed/logged


def test_score_candidates_higher_salience_outscores_when_weighted():
    svc = MemoryPrimingService(AsyncMock())
    now = datetime.now(UTC)
    hi = _make_trace(content="hi", salience=0.9, created_at=now, access_count=0, trace_id=uuid.uuid4())
    lo = _make_trace(content="lo", salience=0.1, created_at=now, access_count=0, trace_id=uuid.uuid4())
    comps = {"semantic": 0.5, "graph": 0.0}  # identical non-salience components
    weights = {"semantic": 0.5, "graph": 0.0, "recency": 0.0, "access": 0.0, "arousal": 0.0, "salience": 0.5}
    ranked = svc._score_candidates({str(hi.id): (hi, dict(comps)), str(lo.id): (lo, dict(comps))}, weights, limit=10)
    scores = {str(t.id): s for t, s, _ in ranked}
    assert scores[str(hi.id)] > scores[str(lo.id)]  # higher salience wins
    assert str(ranked[0][0].id) == str(hi.id)  # and ranks first


# ── weights_override — experiment/benchmark callers only ──────────────────────
#
# Phase 0 safety constraint (2026-07-02 design): the in-app weight-experiment job
# threads candidate weights through an EXPLICIT constructor override. The live
# path (no override) must stay bit-identical: cached DB weights, same call shape.
# Never monkeypatch class attributes in-app — that leaks into concurrent chats.


def _stubbed_priming_service(override=None):
    with patch("kai_memory.retrieval.memory_priming.CognitiveMemoryService"):
        if override is None:
            svc = MemoryPrimingService(AsyncMock())
        else:
            svc = MemoryPrimingService(AsyncMock(), weights_override=override)
    svc._collect_candidates = AsyncMock(return_value=([MagicMock()], {}))
    captured = {}

    def _capture_score(candidates, weights, limit):
        captured["weights"] = weights
        return []

    svc._score_candidates = _capture_score
    return svc, captured


@pytest.mark.asyncio
async def test_live_path_without_override_uses_cached_db_weights():
    svc, captured = _stubbed_priming_service()
    sentinel = {"semantic": 0.99}
    svc._get_cached_weights = AsyncMock(return_value=sentinel)

    await svc.prime_with_traces("hello", record_access=False)

    svc._get_cached_weights.assert_awaited_once()  # live path reads the DB cache
    assert captured["weights"] is sentinel


@pytest.mark.asyncio
async def test_weights_override_bypasses_db_and_scores_with_override():
    override = {"semantic": 0.40, "graph": 0.20, "recency": 0.15, "access": 0.15, "arousal": 0.10, "salience": 0.0}
    svc, captured = _stubbed_priming_service(override=override)
    svc._get_cached_weights = AsyncMock(side_effect=AssertionError("override must not read DB weights"))

    await svc.prime_with_traces("hello", record_access=False)

    assert captured["weights"] == override


@pytest.mark.asyncio
async def test_weights_override_is_instance_scoped_no_leak():
    override = {"semantic": 1.0}
    _override_svc, _ = _stubbed_priming_service(override=override)  # held alive to prove no leak

    live_svc, captured = _stubbed_priming_service()  # constructed AFTER the override instance
    sentinel = {"semantic": 0.42}
    live_svc._get_cached_weights = AsyncMock(return_value=sentinel)

    await live_svc.prime_with_traces("hello", record_access=False)

    assert captured["weights"] is sentinel  # concurrent live service unaffected


# ── #273: intentions score on their OWN component, never forged semantic/graph ─
#
# Path 4 used to stamp `semantic = 1.0` and `graph = 1.0` onto every intention.
# Those were scores it never earned — arriving via path P is not evidence of
# scoring X on signal P — and the 0.60 of unearned composite crowded genuinely
# answering traces out of the top-8 (48/70 → 56/70 golds@8 when stripped).


async def _gather_intention_only(svc, intention, *, semantic=None):
    """Drive `_gather` with ONLY Path 4 (+ optional Path 1) returning anything."""
    entity = MagicMock()
    entity.id = uuid.uuid4()
    with (
        patch.object(svc.memory_svc, "search_traces_raw", return_value=semantic or []),
        patch.object(svc.memory_svc, "find_entities_by_names", return_value=[entity]),
        patch.object(svc.memory_svc, "get_entity_linked_traces", return_value=[]),
        patch.object(svc.memory_svc, "traverse_entity_graph", return_value=[]),
        patch.object(svc.memory_svc, "get_intention_traces", return_value=[intention]),
    ):
        return await svc._collect_candidates("Ask John about it", limit=10)


@pytest.mark.asyncio
async def test_gather_intention_sets_intention_component_not_forged_scores():
    svc = MemoryPrimingService(AsyncMock())
    intention = _make_trace(content="Ask John about his new job", memory_type="intention")

    all_traces, paths = await _gather_intention_only(svc, intention)

    comps = all_traces[str(intention.id)][1]
    assert comps["intention"] == 1.0  # scores on its own component
    # THE regression guard: the forged 1.0s are gone. Absent (⇒ 0.0 at scoring
    # time), not present-and-zero — Path 4 must not author these keys at all.
    assert "semantic" not in comps
    assert "graph" not in comps
    assert "intention" in paths[str(intention.id)]


@pytest.mark.asyncio
async def test_gather_intention_preserves_earned_semantic_score():
    """An intention that ALSO legitimately matched Path 1 keeps its real score."""
    svc = MemoryPrimingService(AsyncMock())
    intention = _make_trace(content="Ask John about his new job", memory_type="intention")

    all_traces, _paths = await _gather_intention_only(svc, intention, semantic=[(intention, 0.42)])

    comps = all_traces[str(intention.id)][1]
    assert comps["semantic"] == pytest.approx(0.42)  # earned — NOT overwritten with 1.0
    assert comps["intention"] == 1.0


def test_default_intention_weight_is_sized_against_a_real_composite():
    from kai_memory.store.cognitive_memory_constants import DEFAULT_RETRIEVAL_WEIGHTS

    w = DEFAULT_RETRIEVAL_WEIGHTS["intention"]
    assert w > 0.0  # intentions must remain servable — do NOT zero them out
    # THE sizing constraint, and the easy mistake: `semantic` is itself a
    # weighted signal, so a PERFECT semantic match earns only w_semantic (0.35),
    # not 1.0. Sizing the intention weight against the 0.60 forgery it replaces
    # (e.g. "0.30 is safely under 0.60") still outranks a trace matching at
    # semantic 0.85 — the same crowding-out bug at a lower number. The honest
    # ceiling to stay under is the semantic weight itself.
    assert w < DEFAULT_RETRIEVAL_WEIGHTS["semantic"]


def test_weighted_component_score_respects_intention_weight():
    from kai_memory.retrieval.memory_priming import weighted_component_score

    comps = {"semantic": 0.0, "graph": 0.0, "recency": 0.0, "access": 0.0, "arousal": 0.0, "intention": 1.0}
    weights = {"semantic": 0.35, "graph": 0.25, "recency": 0.15, "access": 0.15, "arousal": 0.10, "intention": 0.15}
    assert weighted_component_score(comps, weights) == pytest.approx(0.15)
    # weight is honoured, not hardcoded: halve it, halve the contribution
    assert weighted_component_score(comps, {**weights, "intention": 0.075}) == pytest.approx(0.075)
    # and a weights dict PREDATING the intention component (cached 6-key) is tolerated
    assert weighted_component_score(comps, {k: v for k, v in weights.items() if k != "intention"}) == 0.0


def test_intention_no_longer_outranks_a_genuinely_answering_trace():
    """The #273 bug, end to end, at the SHIPPED default weights.

    A strongly-answering fact (semantic 0.85) vs an equally-fresh intention with
    no earned semantic/graph. Under the forgery the intention scored
    1.0*0.35 + 1.0*0.25 = 0.60 and buried the fact. Now it must rank behind.
    Uses DEFAULT_RETRIEVAL_WEIGHTS deliberately: this pins the weight we ship,
    so retuning `intention` upward past the crowding-out point fails here.
    """
    from kai_memory.store.cognitive_memory_constants import DEFAULT_RETRIEVAL_WEIGHTS

    svc = MemoryPrimingService(AsyncMock())
    now = datetime.now(UTC)
    fact = _make_trace(content="John works at Acme", created_at=now, trace_id=uuid.uuid4())
    intention = _make_trace(content="Ask John about it", memory_type="intention", created_at=now, trace_id=uuid.uuid4())

    ranked = svc._score_candidates(
        {
            str(fact.id): (fact, {"semantic": 0.85}),
            str(intention.id): (intention, {"intention": 1.0}),
        },
        dict(DEFAULT_RETRIEVAL_WEIGHTS),
        limit=10,
    )
    assert str(ranked[0][0].id) == str(fact.id)  # the answering trace wins


def test_intention_still_surfaces_over_a_weak_match():
    """Guard the OTHER direction: intentions are prospective memories that must
    still be servable — the fix must not amount to zeroing Path 4 out. A fresh
    relevant intention still beats a weak-but-sticky trace (the ~59% band at
    top_score ~0.34) at the shipped defaults."""
    from kai_memory.store.cognitive_memory_constants import DEFAULT_RETRIEVAL_WEIGHTS

    svc = MemoryPrimingService(AsyncMock())
    now = datetime.now(UTC)
    weak = _make_trace(content="tangential", created_at=now, trace_id=uuid.uuid4())
    intention = _make_trace(content="Ask John about it", memory_type="intention", created_at=now, trace_id=uuid.uuid4())

    ranked = svc._score_candidates(
        {
            str(weak.id): (weak, {"semantic": 0.34}),
            str(intention.id): (intention, {"intention": 1.0}),
        },
        dict(DEFAULT_RETRIEVAL_WEIGHTS),
        limit=10,
    )
    assert str(ranked[0][0].id) == str(intention.id)


def test_primed_payload_logs_intention_component():
    svc = MemoryPrimingService(AsyncMock())
    trace = _make_trace(memory_type="intention")
    payload = svc._build_primed_payload([(trace, 0.3, {"intention": 1.0})], {str(trace.id): {"intention"}})
    assert payload[0]["score_breakdown"]["intention"] == 1.0
    # the forged components are reported at their honest value
    assert payload[0]["score_breakdown"]["semantic"] == 0.0
    assert payload[0]["score_breakdown"]["graph"] == 0.0
