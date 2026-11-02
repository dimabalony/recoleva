"""Capture contention metrics never replace the transaction's typed busy result."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from kai_memory import ports
from kai_memory.procedural import rule_capture


@pytest.mark.parametrize("acquired", [True, False])
@pytest.mark.parametrize("telemetry_failure", [False, True])
async def test_capture_guard_counts_only_busy_and_preserves_result(acquired, telemetry_failure, caplog):
    private_text = "synthetic-private-telemetry-error"
    counter = Mock(side_effect=RuntimeError(private_text) if telemetry_failure else None)
    session = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(scalar_one=lambda: acquired)))
    with (
        ports.scoped({"telemetry": SimpleNamespace(counter=counter)}),
        caplog.at_level("DEBUG", logger=rule_capture.logger.name),
    ):
        if acquired:
            await rule_capture.lock_procedural_capture(session)
            counter.assert_not_called()
        else:
            with pytest.raises(rule_capture.ProceduralCaptureBusy):
                await rule_capture.lock_procedural_capture(session)
            counter.assert_called_once_with("memory.procedural.capture_busy", 1)
    session.execute.assert_awaited_once()
    assert "pg_try_advisory_xact_lock" in str(session.execute.await_args.args[0])
    assert private_text not in caplog.text


async def test_unbound_telemetry_cannot_replace_capture_busy():
    session = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(scalar_one=lambda: False)))
    previous_resolver = ports.set_default_resolver(None)
    previous_bindings = ports.bind(telemetry=ports.UNBOUND)
    try:
        with ports.scoped({}), pytest.raises(rule_capture.ProceduralCaptureBusy):
            await rule_capture.lock_procedural_capture(session)
    finally:
        ports.bind(**previous_bindings)
        ports.set_default_resolver(previous_resolver)
    session.execute.assert_awaited_once()


async def test_capture_metric_does_not_swallow_cancellation():
    session = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(scalar_one=lambda: False)))
    counter = Mock(side_effect=asyncio.CancelledError)
    with ports.scoped({"telemetry": SimpleNamespace(counter=counter)}), pytest.raises(asyncio.CancelledError):
        await rule_capture.lock_procedural_capture(session)
    counter.assert_called_once_with("memory.procedural.capture_busy", 1)
