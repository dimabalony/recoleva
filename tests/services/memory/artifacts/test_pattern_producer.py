"""Unit tests for PatternProducer (Sub-design D Wave 2 §3.5).

Per the D Wave 2 brief, integration coverage of full L3 pipeline + DB
state lives in ``tests/integration/test_persona_pipeline.py`` (driven by
the testcontainer). Here we lock the per-producer contract:

- JSON parser tolerance + caps
- ``produce()`` returns the expected ``ProducedArtifact`` shape with the
  stable content_hash on ``(dimension, observation)``
- Subject discovery skip paths (too few traces, Sonnet returns [], cap)
- Promotion-to-trait threshold logic via the helper

We deliberately do NOT touch ``memory_artifacts`` triggers / SQL — that
needs the testcontainer and the integration test owns it.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from kai_memory.store import cognitive_config
from kai_memory.artifacts.producer import ProducerSubject, compute_content_hash
from kai_memory.artifacts.producers.pattern_producer import (
    PatternProducer,
    _parse_pattern_response,
)


# ── parser ─────────────────────────────────────────────────


def test_parse_pattern_response_strips_code_fence() -> None:
    raw = """```json
[
  {"dimension": "directness", "observation": "agent says no without softening", "driving_trace_ids": []}
]
```"""
    parsed = _parse_pattern_response(raw)
    assert len(parsed) == 1
    assert parsed[0]["dimension"] == "directness"


def test_parse_pattern_response_drops_invalid_items() -> None:
    raw = (
        '[{"dimension": "x", "observation": "y", "driving_trace_ids": []},'
        ' {"dimension": "", "observation": "z"},'  # empty dimension
        ' {"observation": "no dimension"},'
        ' "junk", 7,'
        ' {"dimension": "valid", "observation": "    "}]'  # whitespace observation
    )
    parsed = _parse_pattern_response(raw)
    assert len(parsed) == 1
    assert parsed[0]["dimension"] == "x"


def test_parse_pattern_response_truncates_long_strings() -> None:
    """Defensive: payload validation caps observation at 250 / dimension at 64."""
    long_obs = "a" * 400
    long_dim = "d" * 80
    raw = f'[{{"dimension": "{long_dim}", "observation": "{long_obs}", "driving_trace_ids": []}}]'
    parsed = _parse_pattern_response(raw)
    assert len(parsed) == 1
    assert len(parsed[0]["observation"]) == 250
    assert len(parsed[0]["dimension"]) == 64


def test_parse_pattern_response_handles_empty_and_garbage() -> None:
    assert _parse_pattern_response(None) == []
    assert _parse_pattern_response("") == []
    assert _parse_pattern_response("[]") == []
    assert _parse_pattern_response("not json at all") == []


def test_parse_pattern_response_normalizes_synonym_to_canonical() -> None:
    """Known synonym collapses to canonical via the dimension normalizer.

    Prod 2026-05-15 had 8 variants of "context discipline"; the normalizer
    is the choke point that prevents that recurring at write time.
    """
    raw = (
        '[{"dimension": "context_isolation_discipline",'
        ' "observation": "agent isolates contexts well", "driving_trace_ids": []}]'
    )
    parsed = _parse_pattern_response(raw)
    assert len(parsed) == 1
    assert parsed[0]["dimension"] == "context_discipline"


def test_parse_pattern_response_passes_through_unknown_dimension() -> None:
    """Soft whitelist: a name not in the alias map survives unchanged so
    Sonnet retains the ability to coin new dimensions.
    """
    raw = '[{"dimension": "completely_novel_axis", "observation": "something new", "driving_trace_ids": []}]'
    parsed = _parse_pattern_response(raw)
    assert len(parsed) == 1
    assert parsed[0]["dimension"] == "completely_novel_axis"


def test_parse_pattern_response_dedupes_same_run_after_normalization() -> None:
    """Two raw entries that share an observation and whose dimensions
    normalize to the same canonical collapse into one — driving_trace_ids
    are unioned (order preserved, no duplicates).

    Prevents wasted reinforce() calls when Sonnet emits the same behaviour
    under two synonym dimension names in a single response.
    """
    raw = (
        "["
        '{"dimension": "context_isolation_discipline",'
        ' "observation": "agent isolates contexts well",'
        ' "driving_trace_ids": ["t1", "t2"]},'
        '{"dimension": "context_bleed_control",'
        ' "observation": "agent isolates contexts well",'
        ' "driving_trace_ids": ["t2", "t3"]}'
        "]"
    )
    parsed = _parse_pattern_response(raw)
    assert len(parsed) == 1
    entry = parsed[0]
    assert entry["dimension"] == "context_discipline"
    assert entry["observation"] == "agent isolates contexts well"
    assert entry["driving_trace_ids"] == ["t1", "t2", "t3"]


# ── produce() ──────────────────────────────────────────────


async def test_produce_emits_pattern_with_stable_hash() -> None:
    """The stable content_hash is computed on (dimension, observation) only.

    Re-running discovery with a different observed_at MUST produce the
    same hash so supersede() can short-circuit to reinforcement.
    """
    producer = PatternProducer()
    trace_a = uuid.uuid4()
    valid_ids = {trace_a}
    subject_a = ProducerSubject(
        key="pattern::directness::test",
        payload={
            "dimension": "directness",
            "observation": "agent commits to a recommendation without hedging",
            "driving_trace_ids": [str(trace_a)],
            "valid_trace_ids": valid_ids,
        },
    )
    result_a = await producer.produce(MagicMock(), subject_a, uuid.uuid4())
    # Same dimension+observation → same hash. Different driving traces
    # don't change the hash because they're not part of the stable subset.
    subject_b = ProducerSubject(
        key="pattern::directness::test",
        payload={
            "dimension": "directness",
            "observation": "agent commits to a recommendation without hedging",
            "driving_trace_ids": [],
            "valid_trace_ids": set(),
        },
    )
    result_b = await producer.produce(MagicMock(), subject_b, uuid.uuid4())
    assert result_a is not None and result_b is not None
    assert result_a.content_hash == result_b.content_hash
    # Hash is the documented compute over the stable subset.
    assert result_a.content_hash == compute_content_hash(
        {
            "dimension": "directness",
            "observation": "agent commits to a recommendation without hedging",
        }
    )


async def test_produce_filters_hallucinated_trace_ids() -> None:
    """Trace ids not in ``valid_trace_ids`` are dropped from driving_traces."""
    producer = PatternProducer()
    trace_a = uuid.uuid4()
    hallucinated = uuid.uuid4()
    subject = ProducerSubject(
        key="k",
        payload={
            "dimension": "directness",
            "observation": "x",
            "driving_trace_ids": [str(trace_a), str(hallucinated)],
            "valid_trace_ids": {trace_a},
        },
    )
    result = await producer.produce(MagicMock(), subject, uuid.uuid4())
    assert result is not None
    assert result.payload["driving_traces"] == [str(trace_a)]


# ── discover_subjects ──────────────────────────────────────


class _FakeTrace:
    def __init__(self, tid: uuid.UUID) -> None:
        self.id = tid
        self.content = "agent reflected"
        self.created_at = datetime.now(UTC)
        self.memory_type = "self_reflection"


async def test_discover_subjects_skips_below_two_traces() -> None:
    producer = PatternProducer()
    session = MagicMock()
    # _expire_stale gets called first; mock execute to handle both paths.
    session.execute = AsyncMock()
    expire_result = MagicMock()
    expire_result.rowcount = 0
    fetch_result = MagicMock()
    fetch_result.scalars.return_value.all.return_value = [_FakeTrace(uuid.uuid4())]
    session.execute.side_effect = [expire_result, fetch_result]

    subjects = await producer.discover_subjects(session, uuid.uuid4())
    assert subjects == []


async def test_discover_subjects_caps_to_max_active_patterns() -> None:
    """Sonnet returns 12 candidates; cap is 8 → 8 subjects accepted."""
    producer = PatternProducer()
    session = MagicMock()
    expire_result = MagicMock()
    expire_result.rowcount = 0
    traces = [_FakeTrace(uuid.uuid4()) for _ in range(5)]
    fetch_result = MagicMock()
    fetch_result.scalars.return_value.all.return_value = traces
    session.execute = AsyncMock(side_effect=[expire_result, fetch_result])

    async def fake_extract(traces_arg, window_days):
        return [
            {
                "dimension": f"dim_{i}",
                "observation": f"observation {i}",
                "driving_trace_ids": [],
            }
            for i in range(12)
        ]

    producer._call_sonnet_pattern_extraction = fake_extract  # type: ignore[method-assign]

    cognitive_config.set_for_testing("persona_max_active_patterns", 8.0)
    try:
        subjects = await producer.discover_subjects(session, uuid.uuid4())
    finally:
        cognitive_config.reset_cache()
    assert len(subjects) == 8


async def test_discover_subjects_returns_empty_on_no_candidates() -> None:
    producer = PatternProducer()
    session = MagicMock()
    expire_result = MagicMock()
    expire_result.rowcount = 0
    traces = [_FakeTrace(uuid.uuid4()) for _ in range(3)]
    fetch_result = MagicMock()
    fetch_result.scalars.return_value.all.return_value = traces
    session.execute = AsyncMock(side_effect=[expire_result, fetch_result])

    async def empty_extract(traces_arg, window_days):
        return []

    producer._call_sonnet_pattern_extraction = empty_extract  # type: ignore[method-assign]
    subjects = await producer.discover_subjects(session, uuid.uuid4())
    assert subjects == []


# ── promotion threshold ────────────────────────────────────


async def test_maybe_promote_returns_zero_below_threshold() -> None:
    """Reinforcement count below threshold → no Trait inserted."""
    producer = PatternProducer()
    session = MagicMock()
    session.execute = AsyncMock()  # Should never execute SQL.

    cognitive_config.set_for_testing("persona_pattern_promotion_threshold", 3.0)
    try:
        promoted = await producer._maybe_promote_to_trait(
            session,
            pattern_id=uuid.uuid4(),
            dimension="directness",
            observation="x",
            reinforcement_count=2,  # below 3
            run_id=uuid.uuid4(),
        )
    finally:
        cognitive_config.reset_cache()
    assert promoted == 0
    session.execute.assert_not_called()


async def test_maybe_promote_skips_when_no_kai_self_entity() -> None:
    """Without a kai-self entity row, promotion silently skips."""
    producer = PatternProducer()

    async def _no_self(session):
        return None

    producer._get_kai_self_entity_id = _no_self  # type: ignore[method-assign]
    session = MagicMock()
    session.execute = AsyncMock()
    promoted = await producer._maybe_promote_to_trait(
        session,
        pattern_id=uuid.uuid4(),
        dimension="x",
        observation="y",
        reinforcement_count=5,
        run_id=uuid.uuid4(),
    )
    assert promoted == 0


async def test_maybe_promote_skips_when_trait_already_exists() -> None:
    """If a PersonalityTrait for the dimension is already active, skip."""
    producer = PatternProducer()
    agent_id = uuid.uuid4()

    async def _self(session):
        return agent_id

    producer._get_kai_self_entity_id = _self  # type: ignore[method-assign]
    session = MagicMock()

    # First execute: existence check returns one row → trait already exists
    existing_row = MagicMock()
    existing_row.first = MagicMock(return_value=(1,))
    session.execute = AsyncMock(return_value=existing_row)

    promoted = await producer._maybe_promote_to_trait(
        session,
        pattern_id=uuid.uuid4(),
        dimension="directness",
        observation="x",
        reinforcement_count=10,
        run_id=uuid.uuid4(),
    )
    assert promoted == 0
    # Single execute (the existence-check). No INSERT issued.
    assert session.execute.await_count == 1


# ── anti-pattern guards (design §7.2 lint) ─────────────────


def test_pattern_observation_capped_at_250_chars() -> None:
    """Schema-level lint: PatternPayload.observation is bounded."""
    from pydantic import ValidationError

    from kai_memory.artifact_schemas import PatternPayload

    # max_length=250 is the assertion; build a payload above limit.
    with pytest.raises(ValidationError):
        PatternPayload(
            dimension="x",
            observation="y" * 251,
            driving_traces=[],
            observed_at=datetime.now(UTC),
            reinforcement_count=1,
        )


# ── max_tokens budget ──────────────────────────────────────


async def test_call_sonnet_uses_8192_max_tokens_and_disables_thinking() -> None:
    """PatternProducer must request max_tokens=8192 AND thinking_mode='disabled'.

    The 2048 ceiling truncated EVERY production run pre-fix and silently
    committed 0 artifacts (parser saw incomplete JSON, returned []).
    Adaptive thinking ate the 8192 budget on ~33% of runs after that fix.
    Locking BOTH the budget AND thinking_mode here so neither regresses.
    """
    from kai_memory.artifacts.producers import pattern_producer as pp_module

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

    producer = PatternProducer()
    trace_a = uuid.uuid4()
    trace = MagicMock()
    trace.id = trace_a
    trace.created_at = datetime.now(UTC)
    trace.content = "agent reflected"

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(pp_module, "call_llm_text", fake_call_llm_text)
        monkey.setattr(pp_module, "get_provider_for_model", fake_get_provider)
        monkey.setattr(pp_module, "start_generation_span", fake_span)
        await producer._call_sonnet_pattern_extraction([trace], window_days=7.0)
    finally:
        monkey.undo()

    assert captured["max_tokens"] == 8192
    assert captured["thinking_mode"] == "disabled"
    assert captured["task_name"] == "memory.pattern.extract"
