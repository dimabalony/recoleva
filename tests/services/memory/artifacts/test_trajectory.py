"""Unit tests for TrajectoryProducer Sonnet parsing + fallback path."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock


from kai_memory.artifacts.producers.trajectory import (
    TrajectoryProducer,
    _fallback_summary,
)


class _FakeTrace:
    def __init__(self, tid: uuid.UUID, content: str, ts: datetime) -> None:
        self.id = tid
        self.content = content
        self.created_at = ts
        self.memory_type = "fact"


def test_fallback_summary_truncates_long_content() -> None:
    long_content = "A" * 500
    trace = _FakeTrace(uuid.uuid4(), long_content, datetime.now(UTC))
    summary = _fallback_summary(trace)
    assert summary.startswith("[fact] ")
    assert len(summary) <= 130  # prefix + 120-char body


async def test_summarise_events_uses_sonnet_summaries_when_present() -> None:
    producer = TrajectoryProducer()
    trace_id = uuid.uuid4()
    trace = _FakeTrace(trace_id, "Pomeranian Луна was at the vet today", datetime.now(UTC))

    async def _fake_call(prompt):
        return {str(trace_id): "Visited the vet"}

    producer._call_sonnet_summaries = _fake_call  # type: ignore[method-assign]
    events = await producer._summarise_events(uuid.uuid4(), [trace])
    assert len(events) == 1
    assert events[0].trace_id == trace_id
    assert events[0].summary == "Visited the vet"


async def test_summarise_events_falls_back_when_sonnet_silent() -> None:
    producer = TrajectoryProducer()
    trace = _FakeTrace(uuid.uuid4(), "Луна moved house", datetime.now(UTC))

    async def _empty(prompt):
        return {}

    producer._call_sonnet_summaries = _empty  # type: ignore[method-assign]
    events = await producer._summarise_events(uuid.uuid4(), [trace])
    assert len(events) == 1
    # The fallback should kick in — fallback_summary returns a "[memory_type] head"
    # string, never an empty one.
    assert events[0].summary.startswith("[fact]")


async def test_summarise_events_makes_naive_timestamps_aware() -> None:
    producer = TrajectoryProducer()
    naive_ts = datetime.now()  # noqa: DTZ005 — explicitly testing naive→aware coercion
    trace = _FakeTrace(uuid.uuid4(), "x", naive_ts)

    async def _empty(prompt):
        return {}

    producer._call_sonnet_summaries = _empty  # type: ignore[method-assign]
    events = await producer._summarise_events(uuid.uuid4(), [trace])
    assert events[0].timestamp.tzinfo is not None


async def test_call_sonnet_summaries_returns_empty_on_provider_failure() -> None:
    producer = TrajectoryProducer()
    # Patch get_provider_for_model to raise — simulates upstream outage.
    import kai_memory.artifacts.producers.trajectory as tmod

    real = tmod.get_provider_for_model
    tmod.get_provider_for_model = MagicMock(side_effect=RuntimeError("boom"))
    try:
        result = await producer._call_sonnet_summaries("prompt")
    finally:
        tmod.get_provider_for_model = real
    assert result == {}


async def test_call_sonnet_summaries_uses_module_parser_seam(monkeypatch) -> None:
    import kai_memory.artifacts.producers.trajectory as tmod

    trace_id = str(uuid.uuid4())
    parser = MagicMock(return_value=[{"trace_id": trace_id, "summary": "Changed plans"}])
    monkeypatch.setattr(tmod, "get_settings", MagicMock())
    monkeypatch.setattr(tmod, "get_provider_for_model", MagicMock())
    monkeypatch.setattr(tmod, "start_generation_span", MagicMock(return_value=None))
    monkeypatch.setattr(tmod, "call_llm_text", AsyncMock(return_value="synthetic response"))
    monkeypatch.setattr(tmod, "parse_json_array_response", parser)

    result = await TrajectoryProducer()._call_sonnet_summaries("prompt")

    assert result == {trace_id: "Changed plans"}
    parser.assert_called_once_with("synthetic response")


async def test_call_sonnet_summaries_uses_8192_max_tokens() -> None:
    """Regression guard: max_tokens must stay >= 8192.

    PR-#122 raised PatternProducer from 2048 → 8192 after silent truncation;
    this PR did the same for TrajectoryProducer. If a future change drops the
    cap back toward 4096, batches with many traces will produce
    fallback-only events for the JSON tail. Asserting the kwarg keeps the
    regression observable.
    """
    import kai_memory.artifacts.producers.trajectory as tmod

    producer = TrajectoryProducer()

    captured: dict[str, object] = {}

    async def _fake_call(_provider, _prompt, **kwargs):
        captured.update(kwargs)
        return "[]"

    real_call = tmod.call_llm_text
    real_provider = tmod.get_provider_for_model
    real_span = tmod.start_generation_span
    tmod.call_llm_text = _fake_call  # type: ignore[assignment]
    tmod.get_provider_for_model = MagicMock(return_value=MagicMock())
    tmod.start_generation_span = MagicMock(return_value=None)
    try:
        await producer._call_sonnet_summaries("prompt")
    finally:
        tmod.call_llm_text = real_call  # type: ignore[assignment]
        tmod.get_provider_for_model = real_provider
        tmod.start_generation_span = real_span

    max_tokens = captured.get("max_tokens")
    assert isinstance(max_tokens, int) and max_tokens >= 8192, (
        f"Trajectory summariser must request >= 8192 output tokens; got {max_tokens}"
    )
    assert captured.get("thinking_mode") == "disabled"
    assert captured.get("task_name") == "memory.trajectory.summary"
