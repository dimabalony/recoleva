"""The SDK inference binding owns each supplied generation span."""

from memory_host_policy import synthetic_policy

import asyncio

import pytest

from kai_memory.host_calls import call_llm_text, get_provider_for_model, start_generation_span
from kai_memory.sdk import HostAdapters, MemoryClient


class RecordingSpan:
    def __init__(self, *, fail_update=False, fail_end=False):
        self.updates = []
        self.end_calls = 0
        self.fail_update = fail_update
        self.fail_end = fail_end

    def update(self, **fields):
        self.updates.append(fields)
        if self.fail_update:
            raise RuntimeError("synthetic telemetry update failure")

    def end(self):
        self.end_calls += 1
        if self.fail_end:
            raise RuntimeError("synthetic telemetry end failure")


class RecordingTelemetry:
    def __init__(self, span):
        self.value = span

    def span(self, name, model):
        return self.value

    def event(self, *args, **kwargs):
        pass

    def counter(self, *args, **kwargs):
        pass


class FakeInference:
    def __init__(self, *, result="synthetic answer", error=None, entered=None):
        self.result, self.error, self.entered = result, error, entered

    async def complete(self, prompt, *, model, max_tokens, thinking_mode="adaptive"):
        if self.entered is not None:
            self.entered.set()
            await asyncio.Event().wait()
        if self.error is not None:
            raise self.error
        return self.result


def make_client(inference, span, **overrides):
    return MemoryClient(
        HostAdapters(
            inference=inference,
            embedder=object(),
            history=object(),
            session_factory=lambda: None,
            telemetry=RecordingTelemetry(span),
            overrides=synthetic_policy(**overrides),
        )
    )


def start_call(client):
    async def invoke():
        span = start_generation_span("memory/test", "fake")
        return await call_llm_text(get_provider_for_model(None, "fake"), "synthetic prompt", span=span)

    task = client.run_owned(lambda: asyncio.create_task(invoke()))
    client.adopt(task)
    return task


@pytest.mark.parametrize("result", ["synthetic answer " * 80, None])
async def test_success_ends_span_once_with_bounded_redacted_output(result):
    span = RecordingSpan()
    redacted_inputs = []

    def redact(value):
        redacted_inputs.append(value)
        return "redacted"

    client = make_client(FakeInference(result=result), span, redact_urls_for_trace=redact)
    try:
        assert await start_call(client) == result
        assert span.updates == [{"output": "redacted" if result else None}]
        assert redacted_inputs == ([result[:500]] if result else [])
        assert span.end_calls == 1
    finally:
        await client.close()


@pytest.mark.parametrize("fail_update,fail_end", [(False, False), (True, False), (False, True), (True, True)])
async def test_telemetry_failure_preserves_success(fail_update, fail_end):
    span = RecordingSpan(fail_update=fail_update, fail_end=fail_end)
    client = make_client(FakeInference(), span)
    try:
        assert await start_call(client) == "synthetic answer"
        assert span.end_calls == 1
    finally:
        await client.close()


@pytest.mark.parametrize("fail_telemetry", [False, True])
async def test_inference_failure_is_preserved_and_span_ends(fail_telemetry):
    error = ValueError("synthetic inference failure")
    span = RecordingSpan(fail_update=fail_telemetry, fail_end=fail_telemetry)
    client = make_client(FakeInference(error=error), span)
    try:
        with pytest.raises(ValueError) as caught:
            await start_call(client)
        assert caught.value is error
        assert span.updates == [{"output": "error"}]
        assert span.end_calls == 1
    finally:
        await client.close()


@pytest.mark.parametrize("fail_telemetry", [False, True])
async def test_cancellation_is_preserved_and_span_ends(fail_telemetry):
    entered = asyncio.Event()
    span = RecordingSpan(fail_update=fail_telemetry, fail_end=fail_telemetry)
    client = make_client(FakeInference(entered=entered), span)
    task = start_call(client)
    try:
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert span.updates == [{"output": "cancelled"}]
        assert span.end_calls == 1
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await client.close()


async def test_redaction_failure_does_not_change_success_or_skip_end():
    span = RecordingSpan()

    def redact(value):
        raise RuntimeError("synthetic redaction failure")

    client = make_client(FakeInference(), span, redact_urls_for_trace=redact)
    try:
        assert await start_call(client) == "synthetic answer"
        assert span.updates == []
        assert span.end_calls == 1
    finally:
        await client.close()
