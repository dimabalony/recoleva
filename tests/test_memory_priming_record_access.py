"""record_access gating in MemoryPrimingService (Sub-design C INC 2 Part B).

The read-only gold-eval path must retrieve memory WITHOUT bumping access counts;
the default keeps the access write so real request-path callers are unchanged.
Kept in its own module so this Part B change doesn't churn the large
``test_memory_priming.py`` suite.
"""

from __future__ import annotations

import uuid as _uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from kai_memory.retrieval import memory_priming as _mp
from kai_memory.retrieval.memory_priming import MemoryPrimingService

_WEIGHTS = {"semantic": 0.35, "graph": 0.25, "recency": 0.15, "access": 0.15, "arousal": 0.10}


@pytest.mark.asyncio
async def test_prime_record_access_false_skips_access_write(monkeypatch):
    """record_access=False (the read-only gold-eval path) must NOT bump access
    counts; the default keeps the access write so real callers are unchanged."""
    svc = MemoryPrimingService(AsyncMock())
    trace = MagicMock()
    trace.id = _uuid.uuid4()
    # _collect_candidates returns (dict[id -> (trace, components)], paths) — match that shape
    monkeypatch.setattr(svc, "_collect_candidates", AsyncMock(return_value=({str(trace.id): (trace, {})}, {})))
    monkeypatch.setattr(svc, "_get_cached_weights", AsyncMock(return_value=_WEIGHTS))
    monkeypatch.setattr(svc, "_score_candidates", lambda candidates, weights, limit: [(trace, 1.0, {})])
    monkeypatch.setattr(svc, "_build_primed_payload", lambda top, paths: [])
    monkeypatch.setattr(svc, "_emit_retrieve_complete_log", lambda *a, **k: None)
    monkeypatch.setattr(_mp, "_format_priming_context", lambda pairs: "mem")
    svc.memory_svc.record_access_batch = AsyncMock()

    await svc.prime("who is X", record_access=False)
    svc.memory_svc.record_access_batch.assert_not_awaited()

    await svc.prime("who is X")  # default True
    svc.memory_svc.record_access_batch.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_weights_read_recovers_the_session_and_is_not_cached(monkeypatch):
    """A deferred-access prime (record_access=False) runs no access write whose failure
    path would clean the session afterwards, and rule serving reuses the session. The
    weights read must recover the session itself, and must not pin the fallback."""
    recover = AsyncMock()
    monkeypatch.setattr(_mp, "recover_session", recover)
    monkeypatch.setattr(_mp, "_WEIGHTS_CACHE", None)
    session = AsyncMock()
    svc = MemoryPrimingService(session)
    svc.memory_svc.get_retrieval_weights = AsyncMock(side_effect=ConnectionError("db down"))

    weights = await svc._get_cached_weights()

    assert weights == dict(_mp.DEFAULT_RETRIEVAL_WEIGHTS)
    recover.assert_awaited_once_with(session)
    assert _mp._WEIGHTS_CACHE is None  # the next prime retries the DB

    svc.memory_svc.get_retrieval_weights = AsyncMock(return_value=_WEIGHTS)
    assert await svc._get_cached_weights() == _WEIGHTS


class _LoadedRow:
    """A loaded ORM row. A rollback expires it, and reading it afterwards fails the way
    a lazy refresh outside the greenlet bridge does (MissingGreenlet)."""

    def __init__(self, **fields):
        object.__setattr__(self, "_fields", fields)
        object.__setattr__(self, "_expired", False)

    def __getattr__(self, name):
        if self._expired:
            raise RuntimeError(f"expired instance refreshed outside the greenlet bridge ({name})")
        try:
            return self._fields[name]
        except KeyError:
            raise AttributeError(name) from None


@pytest.mark.asyncio
async def test_failed_weights_read_leaves_loaded_candidates_scoreable(monkeypatch):
    """The weights read runs before the candidates load, so the recovery rollback it
    triggers on failure cannot expire rows that scoring and rendering then read."""
    loaded: list[_LoadedRow] = []

    async def recover(_session):
        for row in loaded:
            object.__setattr__(row, "_expired", True)

    async def collect(*_args, **_kwargs):
        row = _LoadedRow(
            id=_uuid.uuid4(),
            content="Misha works at Examplecorp",
            memory_type="fact",
            salience=0.6,
            access_count=2,
            emotion_arousal=None,
            created_at=datetime.now(UTC),
            last_accessed_at=None,
            durable=False,
            is_durable=False,
            metadata_={},
            typed_metadata=None,
        )
        loaded.append(row)
        return {str(row._fields["id"]): (row, {"semantic": 0.9})}, {}

    monkeypatch.setattr(_mp, "recover_session", recover)
    monkeypatch.setattr(_mp, "_WEIGHTS_CACHE", None)
    svc = MemoryPrimingService(AsyncMock())
    svc.memory_svc.get_retrieval_weights = AsyncMock(side_effect=ConnectionError("db down"))
    monkeypatch.setattr(svc, "_collect_candidates", collect)
    monkeypatch.setattr(svc, "_emit_retrieve_complete_log", lambda *a, **k: None)

    formatted, ids = await svc.prime("who is Misha", record_access=False)

    assert ids == [loaded[0]._fields["id"]]
    assert "Misha works at Examplecorp" in formatted
