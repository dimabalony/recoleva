"""Unit tests for Level 2 consolidation trigger (5-min silence timer)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory_test_support import ChatContext
from kai_memory.consolidation.consolidation_trigger import (
    _pending_timers,
    cancel_all_timers,
    get_pending_timer_count,
    schedule_consolidation,
)


@pytest.fixture(autouse=True)
def _cleanup_timers():
    """Clean up timers after each test."""
    yield
    cancel_all_timers()
    _pending_timers.clear()


# ── Timer scheduling tests ��───────────────────────────────


@pytest.mark.asyncio
async def test_schedule_creates_timer():
    with patch("kai_memory.consolidation.consolidation_trigger.CONSOLIDATION_DELAY_SECONDS", 3600):
        schedule_consolidation(ChatContext(chat_id=123, topic_id=None))
    assert ("telegram", 123, None) in _pending_timers
    assert not _pending_timers[("telegram", 123, None)].done()
    assert get_pending_timer_count() == 1


@pytest.mark.asyncio
async def test_schedule_resets_on_new_message():
    """Scheduling again should cancel the old timer and create a new one."""
    with patch("kai_memory.consolidation.consolidation_trigger.CONSOLIDATION_DELAY_SECONDS", 3600):
        schedule_consolidation(ChatContext(chat_id=123, topic_id=None))
        old_task = _pending_timers[("telegram", 123, None)]

        schedule_consolidation(ChatContext(chat_id=123, topic_id=None))
        new_task = _pending_timers[("telegram", 123, None)]

    # old_task.cancel() was called — let the loop process it
    await asyncio.sleep(0)
    assert old_task.cancelled() or old_task.done()
    assert old_task is not new_task
    assert not new_task.done()
    assert get_pending_timer_count() == 1


@pytest.mark.asyncio
async def test_concurrent_chats_independent():
    """Different (venue, chat_id, topic_id) tuples should have independent timers."""
    with patch("kai_memory.consolidation.consolidation_trigger.CONSOLIDATION_DELAY_SECONDS", 3600):
        schedule_consolidation(ChatContext(chat_id=123, topic_id=None))
        schedule_consolidation(ChatContext(chat_id=456, topic_id=None))
        schedule_consolidation(ChatContext(chat_id=123, topic_id=789))

    assert get_pending_timer_count() == 3
    assert ("telegram", 123, None) in _pending_timers
    assert ("telegram", 456, None) in _pending_timers
    assert ("telegram", 123, 789) in _pending_timers


@pytest.mark.asyncio
async def test_venues_independent_for_same_chat_topic():
    """Same (chat_id, topic_id) across different venues must NOT collide."""
    with patch("kai_memory.consolidation.consolidation_trigger.CONSOLIDATION_DELAY_SECONDS", 3600):
        schedule_consolidation(ChatContext(chat_id=123, topic_id=2, venue="telegram"))
        schedule_consolidation(ChatContext.for_event())
        schedule_consolidation(ChatContext.for_heartbeat())
        # Also schedule another event-venue that should collide with for_event() —
        # both use (event, None, None), so the second cancels the first.
        schedule_consolidation(ChatContext(venue="event", chat_id=None, topic_id=None))

    assert ("telegram", 123, 2) in _pending_timers
    assert ("event", None, None) in _pending_timers
    assert ("heartbeat", None, None) in _pending_timers
    # Three distinct keys, even though the headless ones share (None, None)
    assert get_pending_timer_count() == 3


@pytest.mark.asyncio
async def test_cancel_all_timers():
    with patch("kai_memory.consolidation.consolidation_trigger.CONSOLIDATION_DELAY_SECONDS", 3600):
        schedule_consolidation(ChatContext(chat_id=123, topic_id=None))
        schedule_consolidation(ChatContext(chat_id=456, topic_id=None))

    cancel_all_timers()
    assert get_pending_timer_count() == 0


# ── Timer firing tests ────────────────────────────────────


@pytest.mark.asyncio
async def test_timer_fires_extraction():
    """After delay, extraction should run."""
    ctx = ChatContext(chat_id=123, topic_id=None, is_group=False)
    with (
        patch("kai_memory.consolidation.consolidation_trigger.CONSOLIDATION_DELAY_SECONDS", 0.01),
        patch("kai_memory.consolidation.consolidation_trigger._run_extraction") as mock_run,
    ):
        mock_run.return_value = None
        schedule_consolidation(ctx)

        # Wait for timer to fire
        await asyncio.sleep(0.05)

    mock_run.assert_awaited_once_with(ctx)
    assert ("telegram", 123, None) not in _pending_timers


@pytest.mark.asyncio
async def test_timer_cancelled_does_not_fire():
    """Cancelling a timer should prevent extraction."""
    with (
        patch("kai_memory.consolidation.consolidation_trigger.CONSOLIDATION_DELAY_SECONDS", 0.05),
        patch("kai_memory.consolidation.consolidation_trigger._run_extraction") as mock_run,
    ):
        schedule_consolidation(ChatContext(chat_id=123, topic_id=None))
        _pending_timers[("telegram", 123, None)].cancel()

        await asyncio.sleep(0.1)

    mock_run.assert_not_awaited()


# ── Extraction execution tests ────────────────────────────


def _savepoint() -> MagicMock:
    """Stand-in for ``session.begin_nested()``: an async context manager that does
    not swallow the extraction's exception."""
    savepoint = MagicMock()
    savepoint.__aenter__ = AsyncMock(return_value=None)
    savepoint.__aexit__ = AsyncMock(return_value=False)
    return savepoint


@pytest.mark.asyncio
async def test_run_extraction_creates_consolidation_run():
    """_run_extraction should create a consolidation_run record."""
    from kai_memory.consolidation.consolidation_trigger import _run_extraction

    mock_session = AsyncMock()

    # Mock the consolidation_runs check — no running consolidation
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.flush = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.begin_nested = MagicMock(return_value=_savepoint())

    ctx = ChatContext(chat_id=123, topic_id=None, is_group=False)

    with (
        patch("kai_memory.consolidation.consolidation_trigger.session_scope") as mock_scope,
        patch("kai_memory.extraction.memory_extraction.ExtractionService") as mock_svc_cls,
    ):
        mock_scope.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_scope.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_svc = AsyncMock()
        mock_svc.extract_after_cursor = AsyncMock(
            return_value=({"traces_created": 2, "entities_found": 3, "contradictions": 0}, None)
        )
        mock_svc_cls.return_value = mock_svc

        await _run_extraction(ctx)

    mock_session.add.assert_called_once()  # consolidation_run record
    mock_svc.extract_after_cursor.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_extraction_skips_if_running():
    """Should skip if another L2 consolidation is already running."""
    from kai_memory.consolidation.consolidation_trigger import _run_extraction

    mock_session = AsyncMock()

    # Mock: there IS a recent running consolidation (started just now)
    from datetime import UTC, datetime

    running_run = MagicMock()
    running_run.started_at = datetime.now(UTC)  # not stale
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = running_run
    mock_session.execute = AsyncMock(return_value=mock_result)

    ctx = ChatContext(chat_id=123, topic_id=None, is_group=False)

    with (
        patch("kai_memory.consolidation.consolidation_trigger.session_scope") as mock_scope,
    ):
        mock_scope.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_scope.return_value.__aexit__ = AsyncMock(return_value=False)

        await _run_extraction(ctx)

    # session.add should NOT be called (early return — run is recent, not stale)
    mock_session.add.assert_not_called()


@pytest.mark.asyncio
async def test_run_extraction_failure_sets_status_failed():
    """Extraction failure should mark the run as failed, not crash."""
    from kai_memory.consolidation.consolidation_trigger import _run_extraction

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.flush = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.begin_nested = MagicMock(return_value=_savepoint())

    ctx = ChatContext(chat_id=123, topic_id=None, is_group=False)

    with (
        patch("kai_memory.consolidation.consolidation_trigger.session_scope") as mock_scope,
        patch("kai_memory.extraction.memory_extraction.ExtractionService") as mock_svc_cls,
    ):
        mock_scope.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_scope.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_svc = AsyncMock()
        mock_svc.extract_after_cursor = AsyncMock(side_effect=RuntimeError("DB exploded"))
        mock_svc_cls.return_value = mock_svc

        # Should NOT raise
        await _run_extraction(ctx)

    # The consolidation_run record added via session.add should have failed status
    added_run = mock_session.add.call_args[0][0]
    assert added_run.status == "failed"
    assert "DB exploded" in added_run.error_message


@pytest.mark.parametrize("reschedule", [False, True])
async def test_busy_extraction_defers_without_spending_failure_budget_or_priority(monkeypatch, reschedule):
    from contextlib import asynccontextmanager
    from datetime import datetime, UTC
    from uuid import uuid4
    from kai_memory.consolidation import consolidation_trigger as trigger
    from kai_memory.extraction.memory_extraction import ExtractionService
    from kai_memory.extraction_cursor import ExtractionCursor
    from kai_memory.procedural.rule_capture import ProceduralCaptureBusy
    from kai_memory.store.cognitive_memory_constants import VALID_RUN_STATUSES

    ctx = ChatContext(chat_id=991191, venue="telegram")
    key = (ctx.venue, ctx.chat_id, ctx.topic_id)
    cursor = ExtractionCursor(datetime.now(UTC), uuid4())
    priority = uuid4()
    trigger.track_retrieved_traces(ctx, [priority])
    session = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=result)
    session.flush = AsyncMock()
    session.begin_nested.return_value = _savepoint()

    @asynccontextmanager
    async def scope():
        yield session

    monkeypatch.setattr(trigger, "session_scope", scope)
    monkeypatch.setattr(trigger, "restore_cursor", AsyncMock(return_value=cursor))
    monkeypatch.setattr(trigger, "failure_streak", AsyncMock(return_value=(99, 99, "existing-failed-page")))
    schedule = MagicMock()
    monkeypatch.setattr(trigger, "schedule_consolidation", schedule)
    extraction = AsyncMock(side_effect=ProceduralCaptureBusy)
    monkeypatch.setattr(ExtractionService, "extract_after_cursor", extraction)
    try:
        assert await trigger._run_extraction(ctx, reschedule=reschedule) is None
        assert extraction.await_args.kwargs["skip_page_at"] == "existing-failed-page"
        run = session.add.call_args.args[0]
        assert run.status == "deferred" and run.status in VALID_RUN_STATUSES and run.completed_at is not None
        assert run.run_metadata == {
            "chat": {"venue": ctx.venue, "chat_id": ctx.chat_id, "topic_id": ctx.topic_id},
            "start_cursor": cursor.to_metadata(),
            "deferred_reason": "capture_busy",
        }
        assert key not in trigger._last_extraction
        assert trigger.pop_priority_candidates(ctx) == [priority]
        if reschedule:
            schedule.assert_called_once_with(ctx)
        else:
            schedule.assert_not_called()
    finally:
        trigger._last_extraction.pop(key, None)
        trigger._retrieved_during_priming.pop(key, None)
