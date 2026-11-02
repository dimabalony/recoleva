"""Host observability failures preserve completed SDK operations."""

from memory_host_policy import synthetic_policy

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kai_memory import sdk


class FailingTelemetry:
    def span(self, name, model):
        raise RuntimeError("collector unavailable")

    def event(self, name, **fields):
        raise RuntimeError("collector unavailable")

    def counter(self, name, value=1, **tags):
        raise RuntimeError("collector unavailable")


def client():
    return sdk.MemoryClient(
        sdk.HostAdapters(
            overrides=synthetic_policy(),
            inference=object(),
            embedder=object(),
            history=object(),
            session_factory=lambda: None,
            telemetry=FailingTelemetry(),
        )
    )


@pytest.mark.asyncio
async def test_extract_returns_completed_result_despite_event_failure(monkeypatch):
    from kai_memory.consolidation import consolidation_trigger

    value = client()
    value._load_config_once = AsyncMock()
    extract = AsyncMock(return_value={"stored": 1})
    monkeypatch.setattr(consolidation_trigger, "extract_now", extract)
    try:
        assert await value.extract(sdk.Chat()) == {"stored": 1, "completed": True}
        extract.assert_awaited_once()
    finally:
        await value.close()


@pytest.mark.asyncio
async def test_consolidate_preserves_success_despite_event_failure(monkeypatch):
    from kai_memory.consolidation import consolidation

    value = client()
    completed = {"success": True}
    work = AsyncMock(return_value=completed)
    monkeypatch.setattr(consolidation, "consolidate", work)
    try:
        assert await value.consolidate() is completed
        work.assert_awaited_once()
    finally:
        await value.close()


@pytest.mark.asyncio
async def test_recall_returns_staged_result_despite_counter_failure(monkeypatch):
    from kai_memory import recognition, host_calls

    value = client()
    value._load_config_once = AsyncMock()
    result = SimpleNamespace(primed=SimpleNamespace(trace_ids=["synthetic"]))
    work = AsyncMock(return_value=result)
    monkeypatch.setattr(recognition, "recognize", work)

    @asynccontextmanager
    async def committed_session():
        yield object()

    monkeypatch.setattr(host_calls, "session_scope", committed_session)
    try:
        assert await value.recall("synthetic", sdk.Chat()) is result
        work.assert_awaited_once()
    finally:
        await value.close()


@pytest.mark.asyncio
async def test_span_start_failure_does_not_prevent_inference(monkeypatch):
    from kai_memory.host_calls import start_generation_span, call_llm_text, get_provider_for_model

    inference = SimpleNamespace(complete=AsyncMock(return_value="synthetic answer"))
    value = sdk.MemoryClient(
        sdk.HostAdapters(
            overrides=synthetic_policy(),
            inference=inference,
            embedder=object(),
            history=object(),
            session_factory=lambda: None,
            telemetry=FailingTelemetry(),
        )
    )

    async def work():
        span = start_generation_span("synthetic", "fake")
        assert span is None
        return await call_llm_text(get_provider_for_model(None, "fake"), "synthetic prompt", span=span)

    try:
        assert await value.run_owned(lambda: asyncio.create_task(work())) == "synthetic answer"
        inference.complete.assert_awaited_once()
    finally:
        await value.close()
