"""Recognition boundary (DB-839): `recognize` and its staged commit/discard result.

The real `_prime_memories`, `track_retrieved_traces` and
`CognitiveMemoryService.record_access_batch` run. The fakes sit at the boundaries
only: L1 retrieval (`MemoryPrimingService`, which needs embeddings and the DB),
rule serving (a DB read), and the database session behind `kai_memory.ports.session_scope`.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

import pytest

from kai_memory.consolidation import consolidation_trigger
from kai_memory.procedural import procedural_serving
from kai_memory.retrieval import memory_priming
from memory_test_support import ChatContext
from kai_memory import recognition
from kai_memory.recognition import PrimedTurn, StagedRecognition, drain_pending_access, recognize


class _Retrieval:
    """L1 retrieval boundary: returns fixed traces and records each `prime` call."""

    calls: list[dict]
    trace_ids: list[uuid.UUID]

    def __init__(self, _session, **_kwargs):
        pass

    async def prime(self, last_message, **kwargs):
        type(self).calls.append({"message": last_message, **kwargs})
        return "## Active Associations\n- [fact] recalled", list(type(self).trace_ids)


class _Session:
    """DB boundary: records the statements a write path executes."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.statements: list = []
        self.scopes_opened = 0
        self.gate: asyncio.Event | None = None  # set: execute blocks until it fires

    async def execute(self, statement):
        if self.gate is not None:
            await self.gate.wait()
        if self.fail:
            raise ConnectionError("database unavailable")
        self.statements.append(statement)


@pytest.fixture
def trace_ids():
    return [uuid.uuid4(), uuid.uuid4()]


@pytest.fixture
def retrieval(monkeypatch, trace_ids):
    retrieval = type("Retrieval", (_Retrieval,), {"calls": [], "trace_ids": trace_ids})
    monkeypatch.setattr(memory_priming, "MemoryPrimingService", retrieval)

    async def _no_rules(_session, _ctx, **_kwargs):
        return None, []

    monkeypatch.setattr(procedural_serving, "get_learned_behaviors", _no_rules)
    return retrieval


@pytest.fixture
def tracked(monkeypatch):
    """The real reconsolidation registry, fresh per test, with a call counter."""
    registry: dict = {}
    monkeypatch.setattr(consolidation_trigger, "_retrieved_during_priming", registry)
    calls: list[list[uuid.UUID]] = []
    real = consolidation_trigger.track_retrieved_traces

    def _track(chat_ctx, ids):
        calls.append(list(ids))
        real(chat_ctx, ids)

    monkeypatch.setattr(consolidation_trigger, "track_retrieved_traces", _track)
    return registry, calls


@pytest.fixture
def db(monkeypatch):
    session = _Session()

    @asynccontextmanager
    async def _scope():
        session.scopes_opened += 1
        yield session

    import kai_memory.ports as memory_ports

    monkeypatch.setattr(memory_ports, "session_scope", _scope)
    return session


@pytest.fixture
def ctx():
    return ChatContext(chat_id=42, topic_id=7)


def _access_writes(session: _Session) -> list[list[uuid.UUID]]:
    """The trace-id list of every access UPDATE the session executed."""
    writes = []
    for statement in session.statements:
        assert statement.table.name == "memory_traces"
        params = statement.compile().params
        assert params["access_count_1"] == 1
        writes.extend(v for v in params.values() if isinstance(v, list))
    return writes


def _key(ctx: ChatContext) -> tuple:
    return (ctx.venue, ctx.chat_id, ctx.topic_id)




async def test_commit_applies_once(retrieval, tracked, db, ctx, trace_ids):
    registry, calls = tracked
    staged = await recognize(object(), "who is X", chat_ctx=ctx)

    assert retrieval.calls[0]["record_access"] is True  # immediate write stays in priming
    assert staged.primed.trace_ids == trace_ids
    assert calls == []  # nothing tracked before the host settles it

    assert staged.commit() is True
    assert staged.access_task is None  # access was not deferred, so commit starts no write
    assert await drain_pending_access(1.0) is True
    assert staged.outcome == "committed"
    assert registry[_key(ctx)] == trace_ids
    assert calls == [trace_ids]
    assert db.scopes_opened == 0


async def test_double_commit_is_a_noop(retrieval, tracked, db, ctx, trace_ids):
    _registry, calls = tracked
    staged = await recognize(object(), "who is X", chat_ctx=ctx, defer_access=True)

    assert staged.commit() is True
    first = staged.access_task
    assert staged.commit() is False
    assert staged.access_task is first
    assert await drain_pending_access(1.0) is True
    assert calls == [trace_ids]
    assert _access_writes(db) == [trace_ids]


async def test_commit_after_discard_is_a_noop(retrieval, tracked, db, ctx):
    """A late ACK after the steer was rejected must not apply anything."""
    registry, calls = tracked
    staged = await recognize(object(), "who is X", chat_ctx=ctx, defer_access=True)

    assert staged.discard() is True
    assert staged.commit() is False
    assert staged.access_task is None
    assert await drain_pending_access(1.0) is True
    assert staged.outcome == "discarded"
    assert calls == []
    assert registry == {}
    assert db.scopes_opened == 0


async def test_discard_after_commit_is_a_noop(retrieval, tracked, db, ctx, trace_ids):
    registry, _calls = tracked
    staged = await recognize(object(), "who is X", chat_ctx=ctx, defer_access=True)

    assert staged.commit() is True
    assert staged.discard() is False
    assert staged.discard() is False
    assert await drain_pending_access(1.0) is True
    assert staged.outcome == "committed"
    assert registry[_key(ctx)] == trace_ids
    assert _access_writes(db) == [trace_ids]


async def test_deferred_access_writes_exactly_once_on_commit(retrieval, tracked, db, ctx, trace_ids):
    staged = await recognize(object(), "who is X", chat_ctx=ctx, defer_access=True)

    assert retrieval.calls[0]["record_access"] is False  # priming skipped its write
    assert db.statements == []  # nothing written before commit

    assert staged.commit() is True
    task = staged.access_task
    assert task in recognition._access_tasks  # registered, so it is owned and drainable
    assert await drain_pending_access(1.0) is True
    assert task.done() and task not in recognition._access_tasks
    assert db.scopes_opened == 1  # its own session, not the closed priming one
    assert _access_writes(db) == [trace_ids]


async def test_discard_writes_nothing(retrieval, tracked, db, ctx):
    registry, calls = tracked
    staged = await recognize(object(), "who is X", chat_ctx=ctx, defer_access=True)

    assert staged.discard() is True
    assert staged.discard() is False
    assert recognition._access_tasks == set()
    assert await drain_pending_access(1.0) is True
    assert db.scopes_opened == 0
    assert calls == []
    assert registry == {}


async def test_deferred_commit_without_traces_opens_no_session(retrieval, tracked, db, ctx):
    retrieval.trace_ids = []
    staged = await recognize(object(), "who is X", chat_ctx=ctx, defer_access=True)

    assert staged.commit() is True
    assert staged.access_task is None
    assert await drain_pending_access(1.0) is True
    assert db.scopes_opened == 0


async def test_drain_waits_for_pending_writes(retrieval, tracked, db, ctx, trace_ids):
    db.gate = asyncio.Event()
    staged = await recognize(object(), "who is X", chat_ctx=ctx, defer_access=True)
    assert staged.commit() is True
    task = staged.access_task

    assert await drain_pending_access(0.05) is False  # the write is still blocked on the DB
    assert task in recognition._access_tasks
    db.gate.set()
    assert await drain_pending_access(1.0) is True
    assert recognition._access_tasks == set()
    assert _access_writes(db) == [trace_ids]


async def test_drain_with_nothing_pending_returns_at_once():
    assert await drain_pending_access(0) is True


async def test_tracking_failure_is_fail_open(retrieval, monkeypatch, db, ctx, trace_ids):
    def _boom(*_args):
        raise RuntimeError("registry broken")

    monkeypatch.setattr(consolidation_trigger, "track_retrieved_traces", _boom)
    staged = await recognize(object(), "who is X", chat_ctx=ctx, defer_access=True)

    assert staged.commit() is True
    assert await drain_pending_access(1.0) is True
    assert _access_writes(db) == [trace_ids]  # one failed write does not cancel the other


async def test_access_failure_is_logged_and_leaves_the_registry(retrieval, tracked, db, ctx, trace_ids, caplog):
    registry, _calls = tracked
    db.fail = True
    staged = await recognize(object(), "who is X", chat_ctx=ctx, defer_access=True)

    with caplog.at_level(logging.WARNING, logger=recognition.__name__):
        assert staged.commit() is True
        task = staged.access_task
        assert await drain_pending_access(1.0) is True

    assert task.exception() is None  # fail-open: the task itself never raises
    assert task not in recognition._access_tasks
    assert registry[_key(ctx)] == trace_ids
    assert db.scopes_opened == 1
    assert "Deferred access recording failed" in caplog.text


def test_commit_without_a_running_loop_is_fail_open(tracked, ctx, trace_ids, caplog, monkeypatch):
    registry, _calls = tracked
    staged = StagedRecognition(PrimedTurn(None, trace_ids, [], None), ctx, access_deferred=True)
    # Without a loop no write coroutine may even be created: one created and never
    # awaited would leak (a log record's traceback can keep it alive past any gc).
    created: list = []

    def _write(ids):
        created.append(ids)

        async def _noop():
            return None

        return _noop()

    monkeypatch.setattr(recognition, "_record_deferred_access", _write)

    with caplog.at_level(logging.WARNING, logger=recognition.__name__):
        assert staged.commit() is True

    assert created == []

    assert staged.access_task is None
    assert registry[_key(ctx)] == trace_ids
    assert recognition._access_tasks == set()
    assert "Deferred access scheduling failed" in caplog.text


async def test_commit_without_chat_context_skips_tracking(retrieval, tracked, db, trace_ids):
    _registry, calls = tracked
    staged = await recognize(object(), "who is X", defer_access=True)

    assert staged.commit() is True
    assert await drain_pending_access(1.0) is True
    assert calls == []
    assert _access_writes(db) == [trace_ids]


async def test_recognize_calls_the_hook_through_the_module_global(monkeypatch, ctx):
    primed = PrimedTurn("context", [uuid.uuid4()], ["rule-1"], "rules")
    seen: dict = {}

    async def _hook(*args, **kwargs):
        seen.update(kwargs, args=args)
        return primed

    monkeypatch.setattr(recognition, "_prime_memories", _hook)
    staged = await recognize(object(), "hi", ctx, None, phase="steered", input_id="in-1", defer_access=True)

    assert staged.primed is primed
    assert staged.chat_ctx is ctx
    assert seen["record_access"] is False
    assert seen["phase"] == "steered"
    assert seen["input_id"] == "in-1"
