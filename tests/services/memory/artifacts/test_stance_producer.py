"""Unit tests for StanceProducer (Sub-design D Wave 2 §3.4).

Locks the parser, append-in-place semantics, and the supersede-on-drift
threshold. Full L3 integration lives in
``tests/integration/test_persona_pipeline.py``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from kai_memory.store import cognitive_config
from kai_memory.artifacts.producer import ProducerSubject
from kai_memory.artifacts.producers.stance_producer import (
    StanceProducer,
    _parse_stance_response,
    _split_events,
)
from kai_memory.artifacts.types import Artifact, ProducedArtifact


# ── parser ─────────────────────────────────────────────────


def test_parse_stance_response_filters_below_floor() -> None:
    """confidence < persona_stance_confidence_floor (default 0.4) → dropped."""
    raw = """[
      {"topic": "hype_skepticism", "position": "You distrust hype.",
       "reasoning": "Track record of overpromising.", "confidence": 0.7,
       "events": []},
      {"topic": "uncommitted", "position": "p", "reasoning": "r", "confidence": 0.2}
    ]"""
    parsed = _parse_stance_response(raw)
    assert len(parsed) == 1
    assert parsed[0]["topic"] == "hype_skepticism"


def test_parse_stance_response_drops_invalid() -> None:
    raw = """[
      {"topic": "", "position": "p", "reasoning": "r", "confidence": 0.5},
      {"topic": "x", "reasoning": "r", "confidence": 0.5},
      {"topic": "y", "position": "p", "reasoning": "", "confidence": 0.5}
    ]"""
    assert _parse_stance_response(raw) == []


def test_parse_stance_response_normalizes_lengths() -> None:
    raw = (
        '[{"topic": "' + ("t" * 80) + '",'
        '"position": "' + ("p" * 350) + '",'
        '"reasoning": "' + ("r" * 700) + '",'
        '"confidence": 0.5, "events": []}]'
    )
    parsed = _parse_stance_response(raw)
    assert len(parsed) == 1
    assert len(parsed[0]["topic"]) == 64
    assert len(parsed[0]["position"]) == 300
    assert len(parsed[0]["reasoning"]) == 600


# ── split_events ───────────────────────────────────────────


def test_split_events_filters_invalid_kinds_and_traces() -> None:
    valid = uuid.uuid4()
    hallucinated = uuid.uuid4()
    events = [
        {"kind": "agreed", "trace_id": str(valid), "note": "ok"},
        {"kind": "outcome_invalidated", "trace_id": str(valid), "note": "bad"},
        {"kind": "garbage", "trace_id": str(valid)},
        # hallucinated trace_id → trace_id=None on the resulting event
        {"kind": "outcome_validated", "trace_id": str(hallucinated), "note": "h"},
    ]
    reinforcements, challenges = _split_events(events, {valid})
    assert len(reinforcements) == 2  # agreed + outcome_validated (with trace_id=None)
    assert len(challenges) == 1
    assert reinforcements[0].kind == "agreed"
    assert reinforcements[1].trace_id is None  # hallucinated was nulled


# ── produce() ──────────────────────────────────────────────


async def test_produce_emits_stance_keyed_on_kai_self() -> None:
    producer = StanceProducer()
    agent_id = uuid.uuid4()
    valid = uuid.uuid4()
    subject = ProducerSubject(
        key=f"stance::{agent_id}::hype_skepticism",
        payload={
            "agent_self_id": agent_id,
            "topic": "hype_skepticism",
            "position": "You distrust hype.",
            "reasoning": "Pattern of overpromising in tech announcements.",
            "confidence": 0.7,
            "events": [{"kind": "agreed", "trace_id": str(valid), "note": "n"}],
            "valid_trace_ids": {valid},
        },
    )
    result = await producer.produce(MagicMock(), subject, uuid.uuid4())
    assert result is not None
    assert result.subject_entity_id == agent_id
    assert result.dimension == "hype_skepticism"
    assert result.payload["topic"] == "hype_skepticism"
    assert len(result.payload["reinforcement_signals"]) == 1
    assert result.payload["challenge_signals"] == []


# ── supersede append-in-place ──────────────────────────────


def _make_prev_stance(
    *,
    position: str = "p",
    confidence: float = 0.6,
    last_validated_at: str | None = None,
) -> Artifact:
    return Artifact(
        id=uuid.uuid4(),
        artifact_type="Stance",
        schema_version=1,
        producer_name="StanceProducer",
        producer_version=1,
        subject_entity_id=uuid.uuid4(),
        payload={
            "topic": "hype_skepticism",
            "position": position,
            "reasoning": "r",
            "confidence": confidence,
            "reinforcement_signals": [],
            "challenge_signals": [],
            "last_validated_at": last_validated_at or datetime.now(UTC).isoformat(),
        },
        content_hash="h",
        status="active",
        supersedes_id=None,
        created_at=datetime.now(UTC),
        expires_at=None,
    )


def _make_new_produced(
    *,
    position: str = "p",
    confidence: float = 0.6,
    reinforcement_count: int = 1,
    challenge_count: int = 0,
) -> ProducedArtifact:
    return ProducedArtifact(
        artifact_type="Stance",
        schema_version=1,
        subject_entity_id=uuid.uuid4(),
        dimension="hype_skepticism",
        payload={
            "topic": "hype_skepticism",
            "position": position,
            "reasoning": "r",
            "confidence": confidence,
            "reinforcement_signals": [
                {"kind": "agreed", "trace_id": None, "occurred_at": datetime.now(UTC).isoformat(), "note": ""}
                for _ in range(reinforcement_count)
            ],
            "challenge_signals": [
                {"kind": "disagreed", "trace_id": None, "occurred_at": datetime.now(UTC).isoformat(), "note": ""}
                for _ in range(challenge_count)
            ],
            "last_validated_at": datetime.now(UTC).isoformat(),
        },
        content_hash="newhash",
        human_summary="s",
    )


async def test_supersede_appends_in_place_on_small_drift() -> None:
    """Same position + small confidence delta → append, no INSERT."""
    producer = StanceProducer()
    prev = _make_prev_stance(position="You distrust hype.", confidence=0.6)
    new = _make_new_produced(position="You distrust hype.", confidence=0.65, reinforcement_count=2)

    session = MagicMock()
    session.execute = AsyncMock()
    entry = await producer.supersede(session, prev, new, uuid.uuid4())
    # Action recorded as supersede (in-place mutation, but rollback can revert payload).
    assert entry.action == "supersede"
    assert entry.artifact_id == prev.id
    # Single execute (the in-place UPDATE). No INSERT.
    assert session.execute.await_count == 1


async def test_supersede_returns_noop_when_no_new_evidence_and_touches_last_validated_at() -> None:
    """Same position + same confidence + no events → noop, but DOES touch
    ``last_validated_at`` so the staleness gate in ``_expire_stale_stances``
    resets ("I still hold this view, no new evidence" is a validation signal).
    """
    producer = StanceProducer()
    original_lva = "2026-05-01T00:00:00+00:00"
    prev = _make_prev_stance(position="x", confidence=0.6, last_validated_at=original_lva)
    new = _make_new_produced(position="x", confidence=0.6, reinforcement_count=0, challenge_count=0)

    session = MagicMock()
    session.execute = AsyncMock()
    entry = await producer.supersede(session, prev, new, uuid.uuid4())
    assert entry.action == "noop"
    # ONE call — the last_validated_at touch UPDATE. No other SQL.
    assert session.execute.await_count == 1
    # Inspect the UPDATE payload to verify last_validated_at moved forward.
    call_args = session.execute.await_args
    update_stmt = call_args.args[0]
    # The Update statement carries the new payload in its parameters.
    compiled = update_stmt.compile()
    new_payload = compiled.params.get("payload")
    assert new_payload is not None
    assert new_payload["last_validated_at"] != original_lva
    # Other fields preserved (topic/position/confidence unchanged on noop)
    assert new_payload["position"] == prev.payload["position"]
    assert new_payload["confidence"] == prev.payload["confidence"]


async def test_supersede_falls_through_to_base_on_position_change() -> None:
    """Position changed → real BaseProducer 5-step supersede."""
    producer = StanceProducer()
    prev = _make_prev_stance(position="You distrust hype.", confidence=0.6)
    new = _make_new_produced(
        position="You strongly distrust hype, even from credible sources.",
        confidence=0.6,
    )

    # Wire the SQL stubs the base supersede expects.
    class _StubResult:
        def scalars(self):
            class _S:
                def all(self_inner):
                    return [prev.id]

            return _S()

    session = MagicMock()
    session.execute = AsyncMock(return_value=_StubResult())
    entry = await producer.supersede(session, prev, new, uuid.uuid4())
    assert entry.action == "supersede"
    assert entry.prev_artifact_id == prev.id
    # New row INSERT means a fresh artifact_id (different from prev.id).
    assert entry.artifact_id is not None
    assert entry.artifact_id != prev.id
    # Base supersede: UPDATE prev + INSERT new = 2 SQL calls.
    assert session.execute.await_count == 2


async def test_supersede_falls_through_on_large_confidence_drift() -> None:
    """Confidence delta >= 0.25 → real supersede even with same position."""
    producer = StanceProducer()
    prev = _make_prev_stance(position="x", confidence=0.6)
    new = _make_new_produced(position="x", confidence=0.95)  # delta 0.35

    class _StubResult:
        def scalars(self):
            class _S:
                def all(self_inner):
                    return [prev.id]

            return _S()

    session = MagicMock()
    session.execute = AsyncMock(return_value=_StubResult())
    entry = await producer.supersede(session, prev, new, uuid.uuid4())
    assert entry.action == "supersede"
    assert entry.artifact_id != prev.id
    assert session.execute.await_count == 2


async def test_supersede_creates_when_no_prev() -> None:
    """prev=None → BaseProducer create branch."""
    producer = StanceProducer()
    new = _make_new_produced(position="x", confidence=0.6)
    session = MagicMock()
    session.execute = AsyncMock()
    entry = await producer.supersede(session, None, new, uuid.uuid4())
    assert entry.action == "create"
    # Single INSERT.
    assert session.execute.await_count == 1


# ── anti-pattern guards (design §7.2 lint) ─────────────────


def test_stance_position_capped_at_300_chars() -> None:
    from pydantic import ValidationError

    from kai_memory.artifact_schemas import StancePayload

    with pytest.raises(ValidationError):
        StancePayload(
            topic="t",
            position="x" * 301,
            reasoning="r",
            confidence=0.5,
            last_validated_at=datetime.now(UTC),
        )


def test_stance_reasoning_capped_at_600_chars() -> None:
    from pydantic import ValidationError

    from kai_memory.artifact_schemas import StancePayload

    with pytest.raises(ValidationError):
        StancePayload(
            topic="t",
            position="p",
            reasoning="r" * 601,
            confidence=0.5,
            last_validated_at=datetime.now(UTC),
        )


# ── max_tokens budget (P0 truncation fix) ─────────────────


async def test_call_sonnet_uses_8192_max_tokens() -> None:
    """StanceProducer must request max_tokens=8192.

    Bumped together with IdentityRecord and TraitProducer — all three
    used 2048 and showed 0-output on prod from silent JSON truncation.
    """
    from kai_memory.artifacts.producers import stance_producer as sp_module

    captured: dict[str, object] = {}

    class _FakeProvider:
        pass

    async def fake_call_llm_text(provider, prompt, *, max_tokens, span, **kwargs):
        captured["max_tokens"] = max_tokens
        captured["thinking_mode"] = kwargs.get("thinking_mode")
        captured["task_name"] = kwargs.get("task_name")
        return "[]"

    def fake_get_provider(settings, model):
        return _FakeProvider()

    def fake_span(name, model):
        return None

    producer = StanceProducer()
    trace = MagicMock()
    trace.id = uuid.uuid4()
    trace.created_at = datetime.now(UTC)
    trace.content = "agent committed to a stance"

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(sp_module, "call_llm_text", fake_call_llm_text)
        monkey.setattr(sp_module, "get_provider_for_model", fake_get_provider)
        monkey.setattr(sp_module, "start_generation_span", fake_span)
        await producer._call_sonnet_stance_extraction([trace])
    finally:
        monkey.undo()

    assert captured["max_tokens"] == 8192
    assert captured["thinking_mode"] == "disabled"
    assert captured["task_name"] == "memory.stance.extract"


# ── _expire_stale_stances — two-gate eviction ──────────────────────────────


async def _run_expire_with_rows(rows: list[dict]) -> tuple[StanceProducer, MagicMock, int]:
    """Drive _expire_stale_stances against a fake session whose UPDATE returns ``rows``.

    Returns (producer, session_mock, return_count). Each row dict must have
    ``dimension``, ``conf``, and ``last_validated_at`` keys to match the
    RETURNING clause shape.
    """
    producer = StanceProducer()

    class _Row:
        def __init__(self, d):
            self.dimension = d["dimension"]
            self.conf = d["conf"]
            self.last_validated_at = d["last_validated_at"]

    result = MagicMock()
    result.all = MagicMock(return_value=[_Row(r) for r in rows])
    session = MagicMock()
    session.execute = AsyncMock(return_value=result)
    n = await producer._expire_stale_stances(session, uuid.uuid4())
    return producer, session, n


async def test_expire_stale_stances_low_confidence_recent_fires_gate_1() -> None:
    """Confidence < persona_stance_confidence_floor (0.4) → row expired even if recent."""
    cognitive_config.set_for_testing("persona_stance_confidence_floor", 0.4)
    cognitive_config.set_for_testing("persona_stance_retention_days", 7.0)
    try:
        _, session, n = await _run_expire_with_rows(
            [{"dimension": "weak_stance", "conf": 0.30, "last_validated_at": datetime.now(UTC).isoformat()}]
        )
    finally:
        cognitive_config.reset_cache()
    assert n == 1
    # SQL was executed exactly once — the UPDATE.
    assert session.execute.await_count == 1


async def test_expire_stale_stances_high_confidence_stale_fires_gate_2() -> None:
    """Recent confidence is high but last_validated_at is past retention → expired."""
    cognitive_config.set_for_testing("persona_stance_confidence_floor", 0.4)
    cognitive_config.set_for_testing("persona_stance_retention_days", 7.0)
    try:
        stale_iso = (datetime.now(UTC) - timedelta(days=30)).isoformat()
        _, _, n = await _run_expire_with_rows(
            [{"dimension": "old_stance", "conf": 0.95, "last_validated_at": stale_iso}]
        )
    finally:
        cognitive_config.reset_cache()
    assert n == 1


async def test_expire_stale_stances_low_confidence_and_stale_fires_once() -> None:
    """Both gates fire simultaneously — single row expired, not double-counted."""
    cognitive_config.set_for_testing("persona_stance_confidence_floor", 0.4)
    cognitive_config.set_for_testing("persona_stance_retention_days", 7.0)
    try:
        stale_iso = (datetime.now(UTC) - timedelta(days=30)).isoformat()
        _, _, n = await _run_expire_with_rows(
            [{"dimension": "both_gates", "conf": 0.30, "last_validated_at": stale_iso}]
        )
    finally:
        cognitive_config.reset_cache()
    assert n == 1


async def test_expire_stale_stances_high_confidence_recent_does_not_expire() -> None:
    """Healthy stance — neither gate fires. UPDATE returns empty result."""
    cognitive_config.set_for_testing("persona_stance_confidence_floor", 0.4)
    cognitive_config.set_for_testing("persona_stance_retention_days", 7.0)
    try:
        _, _, n = await _run_expire_with_rows([])
    finally:
        cognitive_config.reset_cache()
    assert n == 0


async def test_expire_stale_stances_sql_includes_both_gates() -> None:
    """Lock the SQL shape so a future change can't quietly drop a gate.

    Compiles the UPDATE statement and asserts both ``confidence < :floor``
    and ``last_validated_at < :cutoff`` appear in the same WHERE clause.
    """
    producer = StanceProducer()
    cognitive_config.set_for_testing("persona_stance_confidence_floor", 0.4)
    cognitive_config.set_for_testing("persona_stance_retention_days", 7.0)

    captured_sql: dict[str, str] = {}

    async def capture_execute(stmt, params=None):
        # Render parameters literally so we see the full predicate.
        captured_sql["sql"] = str(stmt).lower()
        result = MagicMock()
        result.all = MagicMock(return_value=[])
        return result

    session = MagicMock()
    session.execute = capture_execute
    try:
        await producer._expire_stale_stances(session, uuid.uuid4())
    finally:
        cognitive_config.reset_cache()
    sql = captured_sql["sql"]
    assert "confidence" in sql
    assert "last_validated_at" in sql
    # Two-gate OR — both predicates in the same WHERE clause.
    assert ":floor" in sql
    assert ":cutoff" in sql
    # status flips to superseded
    assert "status = 'superseded'" in sql
    # Codex review feedback: also lock the NULL fallback so a future schema
    # change can't quietly remove the COALESCE that protects against
    # pre-feature stances missing the field.
    assert "coalesce" in sql
    assert "1970-01-01" in sql
    # OR semantics between gates (not AND — that would be the wrong logic)
    assert " or " in sql


# ── Capacity-drop logging ──────────────────────────────────────────────────


async def test_capacity_drop_logged_when_cap_blocks_new_candidates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When cap is full and new candidates arrive, log a single INFO line
    naming the dropped topics. Pre-fix this was a silent ``continue``."""
    import logging as logging_module

    producer = StanceProducer()
    agent_id = uuid.uuid4()
    valid_traces = {uuid.uuid4(), uuid.uuid4()}

    # 6 active topics already exist; cap is 6.
    active_topics = {f"existing_{i}" for i in range(6)}
    # 3 NEW candidates (different topic names) — all should be dropped.
    candidates = [
        {
            "topic": f"new_topic_{i}",
            "position": "p",
            "reasoning": "r",
            "confidence": 0.85,
            "events": [],
        }
        for i in range(3)
    ]

    # Stub the helpers so we only exercise the capacity loop.
    async def _expire(*a, **kw):
        return 0

    async def _kai_self(*a, **kw):
        return agent_id

    async def _topics(*a, **kw):
        return active_topics

    async def _traces(*a, **kw):
        # Two faux traces, enough to satisfy the `< 2` check.
        t1 = MagicMock(id=next(iter(valid_traces)))
        t2 = MagicMock(id=list(valid_traces)[1])
        return [t1, t2]

    async def _sonnet(*a, **kw):
        return candidates

    producer._expire_stale_stances = _expire  # type: ignore[method-assign]
    producer._get_kai_self_entity_id = _kai_self  # type: ignore[method-assign]
    producer._fetch_active_stance_topics = _topics  # type: ignore[method-assign]
    producer._fetch_recent_traces = _traces  # type: ignore[method-assign]
    producer._call_sonnet_stance_extraction = _sonnet  # type: ignore[method-assign]

    cognitive_config.set_for_testing("persona_max_active_stances", 6.0)
    try:
        with caplog.at_level(logging_module.INFO, logger="app.services.memory.artifacts.producers.stance_producer"):
            subjects = await producer.discover_subjects(MagicMock(), uuid.uuid4())
    finally:
        cognitive_config.reset_cache()

    # No new subjects accepted (cap=6, active=6, capacity_remaining=0).
    assert subjects == []
    # Single INFO line names the dropped topics.
    drop_logs = [r for r in caplog.records if "dropped at cap" in r.getMessage()]
    assert len(drop_logs) == 1
    msg = drop_logs[0].getMessage()
    assert "3 candidate" in msg
    assert "cap=6" in msg
    # First topic name appears in the log (first-5 truncation)
    assert "new_topic_0" in msg


# ── Noop-touch preservation ────────────────────────────────────────────────


async def test_noop_touch_preserves_existing_signal_arrays() -> None:
    """The noop-path UPDATE writes only ``last_validated_at`` — existing
    ``reinforcement_signals`` and ``challenge_signals`` arrays must pass
    through verbatim. Pre-fix the noop was zero-SQL; the new touch must
    not accidentally drop accumulated history.
    """
    producer = StanceProducer()
    original_lva = "2026-05-01T00:00:00+00:00"
    prev = _make_prev_stance(position="x", confidence=0.6, last_validated_at=original_lva)
    # Inject existing signal history into prev to ensure it survives.
    prev.payload["reinforcement_signals"] = [
        {"kind": "agreed", "trace_id": "11111111-1111-1111-1111-111111111111", "note": "kept"},
    ]
    prev.payload["challenge_signals"] = [
        {"kind": "disagreed", "trace_id": "22222222-2222-2222-2222-222222222222", "note": "also kept"},
    ]
    new = _make_new_produced(position="x", confidence=0.6, reinforcement_count=0, challenge_count=0)

    session = MagicMock()
    session.execute = AsyncMock()
    await producer.supersede(session, prev, new, uuid.uuid4())

    update_stmt = session.execute.await_args.args[0]
    new_payload = update_stmt.compile().params.get("payload")
    assert new_payload is not None
    # last_validated_at moved forward
    assert new_payload["last_validated_at"] != original_lva
    # Arrays preserved verbatim (not replaced with empty, not partially merged)
    assert new_payload["reinforcement_signals"] == prev.payload["reinforcement_signals"]
    assert new_payload["challenge_signals"] == prev.payload["challenge_signals"]


# ── Challenge-majority — explicit DEFERRED contract ────────────────────────


async def test_supersede_challenge_majority_is_currently_deferred() -> None:
    """Explicit pin: the design contract (§3.4: confidence "drops with
    invalidations") is NOT yet implemented. A candidate with a challenge-
    majority and tiny confidence drift takes the append-in-place path
    today, not real supersession. This test pins that behavior so the
    next developer can grep for it.

    When the deferred feature ships, this test should flip to assert
    real supersession fires AND/OR confidence is decremented.
    """
    producer = StanceProducer()
    prev = _make_prev_stance(position="held position", confidence=0.90)
    # Same position, near-same confidence — but 3 challenge events vs 0 reinforcement.
    # Per docstring + §3.4 this SHOULD trigger supersession or confidence drop.
    # It does not — this test pins the current behavior so the gap is visible.
    new = _make_new_produced(
        position="held position",
        confidence=0.89,  # within _REFINEMENT_DELTA → append-in-place path
        reinforcement_count=0,
        challenge_count=3,
    )

    session = MagicMock()
    session.execute = AsyncMock()
    entry = await producer.supersede(session, prev, new, uuid.uuid4())

    # Today: append-in-place. Confidence weighted-avg (0.7*0.90 + 0.3*0.89 = 0.897)
    # → no decrement, no supersession, no expiry. Filed as follow-up; gate 2
    # (staleness) catches this indirectly within retention_days.
    assert entry.action == "supersede"  # in-place mutation, recorded as supersede
    update_stmt = session.execute.await_args.args[0]
    payload = update_stmt.compile().params.get("payload")
    # Confidence rose slightly because new=0.89 is weighted in at 0.3, not dropped.
    # If/when the deferred feature lands, this assertion must be updated to
    # show the decremented value or the test should fail and be rewritten
    # to assert real supersession.
    assert payload["confidence"] >= 0.85, (
        "DEFERRED: §3.4 'confidence drops with invalidations' not implemented; "
        "if this assertion fails because confidence dropped, the deferred feature "
        "shipped — update the test accordingly."
    )
