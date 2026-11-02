"""Unit tests for ArtifactPipeline orchestration (M2 Wave 1).

These tests stay hermetic — no Postgres, no real DB. They exercise:
- Producer-failure isolation (savepoint commit vs rollback paths)
- ChangesMadeEntry aggregation timing per design §3.2 (failed-savepoint
  entries discarded, committed entries surface in the returned list)
- Savepoint setup failure path (`begin_nested()` raises → producer skipped)
- Telemetry span attributes (required keys per the locked contract)

DB-rooted invariants — two-stub real savepoint isolation, BaseProducer
content-hash dedup + conflict — live in
``tests/integration/test_artifact_pipeline.py`` against a pgvector test
container.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from kai_memory.artifact_pipeline import ArtifactPipeline
from kai_memory.artifacts.producer import BaseProducer, ProducerSubject
from kai_memory.artifacts.types import (
    ChangesMadeEntry,
    ProducedArtifact,
)

pytestmark = pytest.mark.asyncio


# ── Test fixtures ─────────────────────────────────────────


def _make_session() -> MagicMock:
    """A session whose ``begin_nested()`` returns a tracked async-CM-ish savepoint."""
    session = MagicMock()
    session.begin_nested = AsyncMock(side_effect=_make_savepoint)
    session.execute = AsyncMock()
    return session


def _make_savepoint() -> MagicMock:
    sp = MagicMock()
    sp.is_active = True
    sp.commit = AsyncMock()
    sp.rollback = AsyncMock()
    return sp


def _entry(action: str, name: str = "stub") -> ChangesMadeEntry:
    return ChangesMadeEntry(
        action=action,  # type: ignore[arg-type]
        artifact_id=uuid.uuid4() if action != "noop" else None,
        prev_artifact_id=None,
        artifact_type="EntitySchema",
        subject_entity_id=None,
        human_summary=f"{name}: {action}",
        producer_name=name,
        producer_version=1,
    )


class _StubProducer:
    """Test double — captures pipeline calls without touching a DB.

    Matches the ArtifactProducer Protocol shape (name, version,
    artifact_type + the three async methods). The pipeline's
    ``_iterate_subjects`` calls ``discover_subjects`` then per-subject
    ``produce`` and ``supersede``. We arrange ``produce`` to raise
    mid-iteration when ``raise_on_subject`` is set.
    """

    name: str = "stub"
    version: int = 1
    artifact_type = "EntitySchema"

    def __init__(
        self,
        *,
        name: str = "stub",
        subjects: int = 1,
        raise_on_subject: int | None = None,
        action: str = "create",
    ) -> None:
        self.name = name
        self._subjects = subjects
        self._raise_at = raise_on_subject
        self._action = action
        self.produce_calls = 0
        self.supersede_calls = 0

    async def discover_subjects(self, session, run_id):
        return [ProducerSubject(key=f"{self.name}-{i}", payload={}) for i in range(self._subjects)]

    async def produce(self, session, subject, run_id):
        idx = self.produce_calls
        self.produce_calls += 1
        if self._raise_at is not None and idx == self._raise_at:
            raise RuntimeError(f"{self.name}: simulated produce failure on {idx}")
        return ProducedArtifact(
            artifact_type="EntitySchema",
            schema_version=1,
            subject_entity_id=None,
            dimension=None,
            payload={"name": subject.key, "entity_type": "person"},
            human_summary=f"{self.name} produced {subject.key}",
        )

    async def supersede(self, session, prev, new, run_id):
        self.supersede_calls += 1
        return _entry(self._action, name=self.name)


# ── Tests ─────────────────────────────────────────────────


async def test_run_all_aggregates_committed_entries() -> None:
    session = _make_session()
    run_id = uuid.uuid4()
    pipeline = ArtifactPipeline(session, run_id)

    p1 = _StubProducer(name="p1", subjects=2, action="create")
    p2 = _StubProducer(name="p2", subjects=1, action="supersede")
    pipeline.register(p1)
    pipeline.register(p2)

    entries = await pipeline.run_all()

    assert len(entries) == 3
    assert {e.producer_name for e in entries} == {"p1", "p2"}
    assert pipeline._producers_run == 2
    assert pipeline._producers_failed == 0


async def test_failed_savepoint_discards_entries() -> None:
    """Producer raises mid-produce → its savepoint rolls back, entries dropped.

    Mirrors §3.2 timing: per-producer in-memory buffer is NOT merged into
    the run-aggregated list when the savepoint rolls back.
    """
    session = _make_session()
    run_id = uuid.uuid4()
    pipeline = ArtifactPipeline(session, run_id)

    crashing = _StubProducer(name="crash", subjects=2, raise_on_subject=1)
    healthy = _StubProducer(name="ok", subjects=1, action="create")
    pipeline.register(crashing)
    pipeline.register(healthy)

    entries = await pipeline.run_all()

    # Healthy producer's entry committed; crashed producer's entries dropped.
    assert [e.producer_name for e in entries] == ["ok"]
    assert pipeline._producers_run == 1
    assert pipeline._producers_failed == 1


async def test_savepoint_setup_failure_skips_producer() -> None:
    """begin_nested() raises → producer skipped, NOT run without isolation."""
    session = _make_session()

    # First begin_nested raises (savepoint setup failure for producer 1).
    # Second succeeds (producer 2 runs normally).
    raising = AsyncMock(side_effect=[RuntimeError("savepoint setup boom")])
    succeeding = AsyncMock(side_effect=_make_savepoint)

    call_count = {"n": 0}

    async def begin_nested_fn():
        call_count["n"] += 1
        if call_count["n"] == 1:
            return await raising()
        return await succeeding()

    session.begin_nested = begin_nested_fn

    run_id = uuid.uuid4()
    pipeline = ArtifactPipeline(session, run_id)

    skipped = _StubProducer(name="skipped", subjects=2)
    healthy = _StubProducer(name="ok", subjects=1, action="create")
    pipeline.register(skipped)
    pipeline.register(healthy)

    entries = await pipeline.run_all()

    # Skipped producer's produce/supersede must NOT have been invoked —
    # that's the "not run without isolation" guarantee.
    assert skipped.produce_calls == 0
    assert skipped.supersede_calls == 0
    assert [e.producer_name for e in entries] == ["ok"]
    assert pipeline._producers_failed == 1
    assert pipeline._producers_run == 1


async def test_pipeline_log_includes_locked_contract_attrs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """memory/consolidate emits the contract-required attributes."""
    session = _make_session()
    run_id = uuid.uuid4()
    pipeline = ArtifactPipeline(
        session,
        run_id,
        phase="synthesize",
        pass_number=2,
        chat_id=42,
        topic_id=7,
        venue="scheduler",
    )
    pipeline.register(_StubProducer(name="p", subjects=1))

    with caplog.at_level(logging.INFO, logger="app.services.memory.artifact_pipeline"):
        await pipeline.run_all()

    messages = [r.message for r in caplog.records if r.message.startswith("memory/consolidate")]
    assert messages, "memory/consolidate log was not emitted"
    payload = _parse_inline_json(messages[-1])
    # Locked-contract required attributes (memory-v2-telemetry.md §1).
    for key in (
        "run_id",
        "phase",
        "pass_number",
        "producers_run",
        "producers_failed",
        "artifacts_produced",
        "chat_id",
        "topic_id",
        "venue",
        "duration_ms",
    ):
        assert key in payload, f"missing {key} in memory/consolidate payload"
    assert payload["phase"] == "synthesize"
    assert payload["pass_number"] == 2
    assert payload["producers_run"] == 1
    assert payload["artifacts_produced"] == {"EntitySchema": 1}


async def test_pipeline_log_emits_when_orchestrator_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """memory/consolidate still emits when orchestration exits via exception."""
    session = _make_session()
    run_id = uuid.uuid4()
    pipeline = ArtifactPipeline(session, run_id)
    pipeline.register(_StubProducer(name="p", subjects=1))

    async def fail_after_partial_progress(producer) -> str:
        pipeline._producers_failed += 1
        raise RuntimeError("orchestrator-level failure")

    pipeline._run_one = fail_after_partial_progress  # type: ignore[method-assign]

    with caplog.at_level(logging.INFO, logger="app.services.memory.artifact_pipeline"):
        with pytest.raises(RuntimeError, match="orchestrator-level failure"):
            await pipeline.run_all()

    messages = [r.message for r in caplog.records if r.message.startswith("memory/consolidate")]
    assert messages, "memory/consolidate log was not emitted"
    payload = _parse_inline_json(messages[-1])
    assert payload["event_type"] == "memory/consolidate"
    assert payload["run_id"] == str(run_id)
    assert payload["producers_failed"] == 1
    assert payload["producers_run"] == 0


async def test_producer_log_includes_savepoint_outcome(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Per-producer log carries savepoint_outcome + producer_version."""
    session = _make_session()
    run_id = uuid.uuid4()
    pipeline = ArtifactPipeline(session, run_id)

    crashing = _StubProducer(name="crash", subjects=1, raise_on_subject=0)
    healthy = _StubProducer(name="ok", subjects=1)
    pipeline.register(crashing)
    pipeline.register(healthy)

    with caplog.at_level(logging.INFO, logger="app.services.memory.artifact_pipeline"):
        await pipeline.run_all()

    producer_logs = {
        record_payload["event_type"]: record_payload
        for record_payload in (
            _parse_inline_json(r.message) for r in caplog.records if r.message.startswith("memory/artifact-producer.")
        )
    }
    assert producer_logs["memory/artifact-producer.crash"]["savepoint_outcome"] == "rolled_back"
    assert producer_logs["memory/artifact-producer.ok"]["savepoint_outcome"] == "committed"
    assert producer_logs["memory/artifact-producer.ok"]["artifacts_produced"] == 1
    assert producer_logs["memory/artifact-producer.crash"]["artifacts_produced"] == 0
    for payload in producer_logs.values():
        for key in (
            "run_id",
            "duration_ms",
            "producer_version",
            "subjects_discovered",
            "venue",
            "chat_id",
            "topic_id",
        ):
            assert key in payload


async def test_systemic_savepoint_failure_aborts_pipeline(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """>50% savepoint setup failures → pipeline aborts after the tripwire fires.

    With 4 producers and a strict ``>0.5`` threshold:
      - failure 1 (1/4 = 0.25)  → continue
      - failure 2 (2/4 = 0.50)  → continue (not yet > 0.5)
      - failure 3 (3/4 = 0.75)  → BREAK; pipeline aborts before producer 4

    Asserting ``begin_nested.call_count == 3`` locks in that early-abort
    contract; relying on the systemic-failure log alone wouldn't catch a
    regression where the loop kept iterating after the threshold trip.
    """
    session = _make_session()

    # All begin_nested calls raise.
    session.begin_nested = AsyncMock(side_effect=RuntimeError("session wedged"))

    run_id = uuid.uuid4()
    pipeline = ArtifactPipeline(session, run_id)

    p1 = _StubProducer(name="p1")
    p2 = _StubProducer(name="p2")
    p3 = _StubProducer(name="p3")
    p4 = _StubProducer(name="p4")
    for p in (p1, p2, p3, p4):
        pipeline.register(p)

    with caplog.at_level(logging.ERROR, logger="app.services.memory.artifact_pipeline"):
        await pipeline.run_all()

    systemic = [r for r in caplog.records if r.message.startswith("pipeline.savepoint_systemic_failure")]
    assert systemic, "expected pipeline.savepoint_systemic_failure log"
    # Early-abort: the 4th producer's savepoint setup must NOT have been attempted.
    assert session.begin_nested.call_count == 3
    assert p4.produce_calls == 0


# ── BaseProducer noop / conflict tests (no DB needed) ─────


class _BaseStub(BaseProducer):
    """Concrete BaseProducer subclass with discover/produce stubs."""

    name = "base-stub"
    version = 1
    artifact_type = "EntitySchema"

    async def discover_subjects(self, session, run_id):
        return []

    async def produce(self, session, subject, run_id):
        return None


def _stub_artifact(content_hash: str) -> Any:
    artifact = MagicMock()
    artifact.id = uuid.uuid4()
    artifact.content_hash = content_hash
    return artifact


async def test_base_producer_supersede_noop_on_hash_match() -> None:
    """When prev.content_hash == computed hash, return action='noop' and write nothing."""
    producer = _BaseStub()
    payload = {"name": "Луна", "entity_type": "animal"}
    from kai_memory.artifacts.producer import compute_content_hash

    digest = compute_content_hash(payload)
    prev = _stub_artifact(digest)
    new = ProducedArtifact(
        artifact_type="EntitySchema",
        schema_version=1,
        subject_entity_id=None,
        dimension=None,
        payload=payload,
        human_summary="rebuild",
    )
    session = MagicMock()
    session.execute = AsyncMock()

    entry = await producer.supersede(session, prev, new, uuid.uuid4())

    assert entry.action == "noop"
    assert entry.artifact_id == prev.id
    assert session.execute.await_count == 0


async def test_base_producer_supersede_conflict_when_returning_empty() -> None:
    """2-step variant: guard UPDATE returns no rows → 'conflict', no INSERT.

    With UPDATE-then-INSERT ordering and the DEFERRABLE FK on
    superseded_by_id, the guard UPDATE atomically claims prev (status +
    back-pointer set in one statement) and the INSERT follows. When the
    UPDATE returns empty (another producer raced and won), we skip the
    INSERT entirely — no orphan row, artifact_id=None on the entry.
    """
    producer = _BaseStub()
    new_payload = {"name": "Луна", "entity_type": "animal", "rev": 2}
    new = ProducedArtifact(
        artifact_type="EntitySchema",
        schema_version=1,
        subject_entity_id=None,
        dimension=None,
        payload=new_payload,
        human_summary="rebuild",
    )
    prev = _stub_artifact("old-digest-not-matching-new")

    session = MagicMock()
    # The guard UPDATE returns empty rows (concurrent supersession).
    update_result = MagicMock()
    update_result.scalars.return_value.all.return_value = []
    session.execute = AsyncMock(return_value=update_result)

    entry = await producer.supersede(session, prev, new, uuid.uuid4())

    assert entry.action == "conflict"
    assert entry.prev_artifact_id == prev.id
    # No INSERT happened — no orphaned row to track.
    assert entry.artifact_id is None
    # Only the guard UPDATE was issued.
    assert session.execute.await_count == 1


async def test_base_producer_supersede_2_step_writes_update_then_insert() -> None:
    """Successful supersede issues exactly 2 statements: combined-UPDATE / INSERT."""
    producer = _BaseStub()
    new_payload = {"name": "Луна", "entity_type": "animal", "rev": 2}
    new = ProducedArtifact(
        artifact_type="EntitySchema",
        schema_version=1,
        subject_entity_id=None,
        dimension=None,
        payload=new_payload,
        human_summary="rebuild",
    )
    prev = _stub_artifact("old-digest-not-matching-new")

    session = MagicMock()
    # Guard UPDATE returns prev.id (atomic claim succeeded).
    guard_result = MagicMock()
    guard_result.scalars.return_value.all.return_value = [prev.id]
    insert_result = MagicMock()
    session.execute = AsyncMock(side_effect=[guard_result, insert_result])

    entry = await producer.supersede(session, prev, new, uuid.uuid4())

    assert entry.action == "supersede"
    assert entry.prev_artifact_id == prev.id
    assert entry.artifact_id is not None
    # Combined-UPDATE then INSERT — no separate back-pointer fix-up.
    assert session.execute.await_count == 2


async def test_base_producer_supersede_create_when_no_prev() -> None:
    """prev=None path issues only the INSERT (no UPDATE) and returns 'create'."""
    producer = _BaseStub()
    payload = {"name": "Луна", "entity_type": "animal"}
    new = ProducedArtifact(
        artifact_type="EntitySchema",
        schema_version=1,
        subject_entity_id=None,
        dimension=None,
        payload=payload,
        human_summary="fresh",
    )
    session = MagicMock()
    session.execute = AsyncMock()

    entry = await producer.supersede(session, None, new, uuid.uuid4())

    assert entry.action == "create"
    assert entry.prev_artifact_id is None
    assert entry.artifact_id is not None
    assert session.execute.await_count == 1  # INSERT only, no UPDATE


# ── pipeline.producer_failed log shape (Bug 3) ────────────


async def test_producer_failed_log_carries_traceback_and_structured_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The producer_failed log must include exc_info AND a structured JSON payload.

    Pre-fix: the log was a printf-style line without structured fields. CloudWatch
    queries on `event_type=pipeline.producer_failed` couldn't surface the
    exception type / message inline; root-causing required docker exec into
    prod (TraitProducer rollback in PR #119 was the motivating incident).

    Post-fix: payload mirrors `_emit_producer_log` with error_type and
    error_message, and exc_info=True preserves the Python traceback in
    record.exc_text.
    """
    session = _make_session()
    run_id = uuid.uuid4()
    pipeline = ArtifactPipeline(session, run_id)

    crashing = _StubProducer(name="boom", subjects=1, raise_on_subject=0)
    pipeline.register(crashing)

    with caplog.at_level(logging.ERROR, logger="app.services.memory.artifact_pipeline"):
        await pipeline.run_all()

    failed_records = [r for r in caplog.records if r.message.startswith("pipeline.producer_failed")]
    assert failed_records, "expected pipeline.producer_failed log"
    record = failed_records[0]

    # exc_info=True populated → record.exc_info is a 3-tuple AND
    # record.exc_text is rendered with the exception class name.
    assert record.exc_info is not None
    assert record.exc_info[0] is RuntimeError
    assert record.exc_text is not None
    assert "RuntimeError" in record.exc_text

    # Structured payload mirrors the _emit_producer_log convention.
    payload = _parse_inline_json(record.message)
    assert payload["event_type"] == "pipeline.producer_failed"
    assert payload["producer"] == "boom"
    assert payload["producer_version"] == 1
    assert payload["error_type"] == "RuntimeError"
    assert "simulated produce failure" in payload["error_message"]
    assert payload["run_id"] == str(run_id)


# ── Helpers ───────────────────────────────────────────────


def _parse_inline_json(message: str) -> dict[str, Any]:
    """Extract the JSON body from ``logger.info("event_name <json>", ...)`` lines."""
    _, _, body = message.partition(" ")
    return json.loads(body)
