"""Unit tests for IdentityRecordProducer drift parsing + skip paths.

Subject discovery (DB query for self_reflection traces) is exercised by
``tests/integration/test_artifact_pipeline.py``. Here we lock the parsing
contract and the §4.1 invariant that ``subject_entity_id`` is non-NULL.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from kai_memory.artifacts.producer import ProducerSubject
from kai_memory.artifacts.producers.identity_record import (
    _AGENT_SELF_ENTITY_TYPE,
    IdentityRecordProducer,
    _coerce_magnitude,
    _parse_drift_response,
)


def test_parse_drift_response_strips_code_fence() -> None:
    raw = """```json
[
  {"dimension": "directness", "direction": "increasing", "magnitude": 0.6,
   "driving_trace_ids": ["00000000-0000-0000-0000-000000000001"]}
]
```"""
    parsed = _parse_drift_response(raw)
    assert len(parsed) == 1
    assert parsed[0]["dimension"] == "directness"


def test_parse_drift_response_returns_empty_on_empty_array() -> None:
    assert _parse_drift_response("[]") == []
    assert _parse_drift_response(None) == []
    assert _parse_drift_response("garbage") == []


def test_parse_drift_response_drops_non_dict_items() -> None:
    raw = '[{"dimension": "x", "direction": "stable", "magnitude": 0.4, "driving_trace_ids": []}, "junk", 7]'
    parsed = _parse_drift_response(raw)
    assert len(parsed) == 1
    assert parsed[0]["dimension"] == "x"


def test_filter_eligible_normalizes_synonym_to_canonical() -> None:
    """`_filter_eligible` snaps known synonyms onto the canonical name so
    the partial unique index ``idx_artifacts_one_active`` sees a single
    ``(subject, dimension)`` pair instead of N variants.
    """
    producer = IdentityRecordProducer()
    dims = [
        {
            "dimension": "context_isolation_discipline",
            "direction": "increasing",
            "magnitude": 0.6,
            "driving_trace_ids": [],
        },
        {
            "dimension": "honesty_under_social_pressure",
            "direction": "stable",
            "magnitude": 0.55,
            "driving_trace_ids": [],
        },
    ]
    eligible = producer._filter_eligible(dims)
    names = {d["dimension"] for d in eligible}
    assert names == {"context_discipline", "honesty_under_pressure"}


def test_filter_eligible_passes_through_unknown_dimension() -> None:
    """Soft whitelist: Sonnet can still coin a new dimension."""
    producer = IdentityRecordProducer()
    dims = [
        {
            "dimension": "novel_axis_that_isnt_mapped",
            "direction": "increasing",
            "magnitude": 0.55,
            "driving_trace_ids": [],
        },
    ]
    eligible = producer._filter_eligible(dims)
    assert len(eligible) == 1
    assert eligible[0]["dimension"] == "novel_axis_that_isnt_mapped"


def test_filter_eligible_resolves_collision_keeps_highest_magnitude() -> None:
    """Two raw synonyms collapse to one canonical entry — strongest wins.

    Without per-canonical dedup, `produce()` would run twice for the same
    `(subject, dimension)` pair; the partial unique index would still
    keep the DB honest via supersede, but a wasted Sonnet call per
    collision is avoidable churn.
    """
    producer = IdentityRecordProducer()
    dims = [
        {
            "dimension": "context_isolation_discipline",
            "direction": "increasing",
            "magnitude": 0.6,
            "driving_trace_ids": ["a"],
        },
        {
            "dimension": "context_bleed_control",
            "direction": "stable",
            "magnitude": 0.85,
            "driving_trace_ids": ["b"],
        },
    ]
    eligible = producer._filter_eligible(dims)
    assert len(eligible) == 1
    entry = eligible[0]
    assert entry["dimension"] == "context_discipline"
    assert entry["magnitude"] == 0.85
    # Highest-magnitude variant's metadata is the one retained.
    assert entry["direction"] == "stable"
    assert entry["driving_trace_ids"] == ["b"]


# ── produce() per-dimension + invariant tests ─────────────


def test_coerce_magnitude_handles_strings_and_nulls() -> None:
    assert _coerce_magnitude(0.5) == 0.5
    assert _coerce_magnitude("0.4") == pytest.approx(0.4)
    assert _coerce_magnitude(None) == 0.0
    assert _coerce_magnitude("not-a-number") == 0.0
    # Clamped above 1.0
    assert _coerce_magnitude(1.7) == 1.0
    # Clamped below 0.0
    assert _coerce_magnitude(-0.5) == 0.0


async def test_produce_emits_record_for_eligible_dimension() -> None:
    """produce() takes a per-dimension subject and emits the matching record."""
    producer = IdentityRecordProducer()
    agent_self_id = uuid.uuid4()
    trace_a = uuid.uuid4()
    trace_b = uuid.uuid4()
    epoch = datetime.now(UTC) - timedelta(days=7)

    drift = {
        "dimension": "humor_style",
        "direction": "stable",
        "magnitude": 0.85,
        "driving_trace_ids": [str(trace_a), str(trace_b)],
    }
    subject = ProducerSubject(
        key=f"agent-self:{agent_self_id}::humor_style",
        payload={
            "agent_self_id": agent_self_id,
            "drift": drift,
            "valid_trace_ids": {trace_a, trace_b},
            "fallback_trace_ids": [trace_a, trace_b],
            "epoch_start_seed": epoch,
        },
    )
    result = await producer.produce(MagicMock(), subject, uuid.uuid4())
    assert result is not None
    assert result.dimension == "humor_style"
    # §4.1 invariant — subject_entity_id MUST be non-NULL on every produced
    # IdentityRecord (Postgres treats NULLs as distinct in the unique index).
    assert result.subject_entity_id == agent_self_id
    assert result.payload["subject_entity_id"] == str(agent_self_id)
    assert result.payload["magnitude"] == pytest.approx(0.85)
    # epoch_start carries the trace-derived seed verbatim — stable across runs.
    # Pydantic's mode="json" serialises UTC offsets as ``Z`` instead of ``+00:00``;
    # parse both ends for the comparison.
    assert datetime.fromisoformat(result.payload["epoch_start"].replace("Z", "+00:00")) == epoch
    assert set(result.payload["driving_observations"]) == {str(trace_a), str(trace_b)}


async def test_discover_subjects_emits_one_subject_per_eligible_dimension() -> None:
    """Sonnet returns 3 dimensions; 2 are above min_magnitude → 2 subjects."""
    producer = IdentityRecordProducer()
    agent_self_id = uuid.uuid4()
    trace_a = uuid.uuid4()

    async def _self(session):
        return agent_self_id

    producer._get_agent_self_entity_id = _self  # type: ignore[method-assign]

    async def _fake_drift(traces, window_days):
        return [
            {"dimension": "directness", "direction": "increasing", "magnitude": 0.55, "driving_trace_ids": []},
            {"dimension": "warmth", "direction": "stable", "magnitude": 0.10, "driving_trace_ids": []},
            {"dimension": "humor_style", "direction": "stable", "magnitude": 0.85, "driving_trace_ids": []},
        ]

    producer._call_sonnet_drift_analysis_for_traces = _fake_drift  # type: ignore[method-assign]

    session = MagicMock()
    fake_traces = [_FakeTrace(trace_a) for _ in range(3)]
    result = MagicMock()
    result.scalars.return_value.all.return_value = fake_traces
    session.execute = AsyncMock(return_value=result)

    subjects = await producer.discover_subjects(session, uuid.uuid4())
    dims = [s.payload["drift"]["dimension"] for s in subjects]
    # 'warmth' is filtered (magnitude < 0.3); the other two surface.
    assert sorted(dims) == ["directness", "humor_style"]
    # Strongest sorted first per filter ordering.
    assert dims[0] == "humor_style"
    assert all("epoch_start_seed" in s.payload for s in subjects)


async def test_discover_subjects_returns_empty_when_no_dimensions_eligible() -> None:
    producer = IdentityRecordProducer()

    async def _self(session):
        return uuid.uuid4()

    async def _fake_drift(traces, window_days):
        return [{"dimension": "x", "direction": "stable", "magnitude": 0.05, "driving_trace_ids": []}]

    producer._get_agent_self_entity_id = _self  # type: ignore[method-assign]
    producer._call_sonnet_drift_analysis_for_traces = _fake_drift  # type: ignore[method-assign]

    session = MagicMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = [_FakeTrace(uuid.uuid4()) for _ in range(3)]
    session.execute = AsyncMock(return_value=result)
    subjects = await producer.discover_subjects(session, uuid.uuid4())
    assert subjects == []


async def test_discover_subjects_skips_when_no_agent_self_entity() -> None:
    producer = IdentityRecordProducer()

    # Patch lookup to return None.
    async def _no_self(session):
        return None

    producer._get_agent_self_entity_id = _no_self  # type: ignore[method-assign]
    session = MagicMock()
    session.execute = AsyncMock()
    subjects = await producer.discover_subjects(session, uuid.uuid4())
    assert subjects == []


async def test_discover_subjects_skips_below_three_traces() -> None:
    producer = IdentityRecordProducer()
    agent_id = uuid.uuid4()

    async def _self(session):
        return agent_id

    producer._get_agent_self_entity_id = _self  # type: ignore[method-assign]

    session = MagicMock()
    # Two traces only.
    fake_traces = [_FakeTrace(uuid.uuid4()) for _ in range(2)]
    result = MagicMock()
    result.scalars.return_value.all.return_value = fake_traces
    session.execute = AsyncMock(return_value=result)
    subjects = await producer.discover_subjects(session, uuid.uuid4())
    assert subjects == []


# ── agent-self lookup determinism (Bug 2) ─────────────────


def test_agent_self_entity_type_is_pinned_to_agent() -> None:
    """Lookup MUST pin to entity_type='agent' (not the legacy IN-list).

    Prod has multiple entities named "Kai" with mixed entity_types
    (topic / agent / person / project / object). The canonical Kai entity
    was inserted by migration 067 with entity_type='agent'; an IN-list +
    bare LIMIT 1 with no ORDER BY picks one arbitrarily across runs and
    breaks IdentityRecord's per-(subject, dimension) supersession.
    """
    assert _AGENT_SELF_ENTITY_TYPE == "agent"


async def test_get_agent_self_entity_id_query_filters_by_agent_type() -> None:
    """Compile the Select and assert the WHERE pins entity_type to a single value.

    Lock-in test: catches a regression where a future change reverts to
    the IN-list lookup (which is what made the lookup non-deterministic
    on prod with 5 differently-typed Kai entities).
    """
    producer = IdentityRecordProducer()

    captured: dict[str, object] = {}

    async def fake_execute(stmt):
        captured["stmt"] = stmt
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        result.scalars.return_value.all.return_value = []
        return result

    session = MagicMock()
    session.execute = fake_execute  # type: ignore[assignment]

    await producer._get_agent_self_entity_id(session)

    stmt = captured["stmt"]
    compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    # Pin: equality, not IN-list. The legacy buggy form used `IN ('person', 'agent')`.
    assert "memory_entities.entity_type = 'agent'" in compiled
    assert "memory_entities.entity_type IN" not in compiled
    # Determinism cushion: ORDER BY (created_at, id) — secondary key on id
    # guarantees stable selection even if two agent-typed rows ever share
    # a created_at (CR feedback: timestamp ties are theoretically possible).
    assert "ORDER BY memory_entities.created_at, memory_entities.id" in compiled


# ── Helpers ───────────────────────────────────────────────


class _FakeTrace:
    def __init__(self, tid: uuid.UUID) -> None:
        self.id = tid
        self.content = "..."
        self.created_at = datetime.now(UTC)
        self.memory_type = "self_reflection"


# ── max_tokens budget (P0 truncation fix) ─────────────────


async def test_call_sonnet_uses_8192_max_tokens() -> None:
    """IdentityRecordProducer must request max_tokens=8192.

    Pre-2026-05-11 the budget was 2048 — chronic truncation on prod
    (~9/24h) cut the drift JSON mid-array; parser saw incomplete output
    and returned []; producer committed 0 artifacts. The "drift vs
    snapshot" 0-output mystery is the same bug class as #122 Bug 1.
    """
    from kai_memory.artifacts.producers import identity_record as ir_module

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

    producer = IdentityRecordProducer()
    trace = MagicMock()
    trace.id = uuid.uuid4()
    trace.created_at = datetime.now(UTC)
    trace.content = "agent reflected"

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(ir_module, "call_llm_text", fake_call_llm_text)
        monkey.setattr(ir_module, "get_provider_for_model", fake_get_provider)
        monkey.setattr(ir_module, "start_generation_span", fake_span)
        await producer._call_sonnet_drift_analysis_for_traces([trace], window_days=7.0)
    finally:
        monkey.undo()

    assert captured["max_tokens"] == 8192
    assert captured["thinking_mode"] == "disabled"
    assert captured["task_name"] == "memory.identity.drift"
