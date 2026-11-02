"""Level 2 consolidation trigger — 5-minute silence timer for memory extraction."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import select, update

from kai_memory.host_calls import session_scope, utc_now
from kai_memory.storage.models.consolidation_run import ConsolidationRun
from kai_memory.extraction_cursor import ExtractionCursor, chat_filter, failure_streak, restore_cursor
from kai_memory.ownership import adopt_if_owned, current_owner, owned_state

if TYPE_CHECKING:
    from kai_memory.ports import ChatContext

# Pinned to the pre-DB-842 module path: CloudWatch queries, feature-watch criteria and
# log filters match the logger name, which must not follow the module's move.
_LEGACY_LOGGER_NAME = "app.services.chat.consolidation_trigger"
logger = logging.getLogger(_LEGACY_LOGGER_NAME)

CONSOLIDATION_DELAY_SECONDS = 300
STALE_RUN_MINUTES = 15
LAST_EXTRACTION_TTL_DAYS = 7
# A run that fails is retried on a timer this many times at the same cursor. After
# this many item-write failures on one page, the next run skips that page (and says
# so) rather than wedging the chat's extraction on one bad item.
MAX_FAILED_RUNS_AT_CURSOR = 3

# In-memory state — acceptable to lose on restart. Extraction retries naturally
# on next conversation pause. No DB persistence needed for 5-min timers; the
# extraction cursor itself is persisted on the completed run row (DB-794).

# Global registry of pending consolidation timers: (venue, chat_id, topic_id) → task
_pending_timers: dict[tuple[str, int | None, int | None], asyncio.Task] = {}

# Cursor of the last completed extraction per (venue, chat_id, topic_id), a cache
# over the run rows — pruned periodically, restored from the DB on a miss
_last_extraction: dict[tuple[str, int | None, int | None], ExtractionCursor] = {}

# Trace IDs retrieved during priming — L2 treats them as priority contradiction candidates
_retrieved_during_priming: dict[tuple[str, int | None, int | None], list[uuid.UUID]] = {}


def track_retrieved_traces(
    chat_ctx: ChatContext,
    trace_ids: list[uuid.UUID],
) -> None:
    """Record trace IDs surfaced during priming for possible reconsolidation."""
    if not trace_ids:
        return
    registry = _priming_registry()
    key = (chat_ctx.venue, chat_ctx.chat_id, chat_ctx.topic_id)
    existing = registry.setdefault(key, [])
    for tid in trace_ids:
        if tid not in existing:
            existing.append(tid)
    if len(existing) > 50:
        registry[key] = existing[-50:]


def _priming_registry() -> dict[tuple[str, int | None, int | None], list[uuid.UUID]]:
    """The current memory client's reconsolidation candidates, or the process registry (Kai)."""
    return owned_state("retrieved_during_priming", dict, _retrieved_during_priming)


def _timers() -> dict[tuple[str, int | None, int | None], asyncio.Task]:  # type: ignore[type-arg]
    """The current memory client's pending timers, or the process registry (Kai)."""
    return owned_state("pending_timers", dict, _pending_timers)


def _cursors() -> dict[tuple[str, int | None, int | None], ExtractionCursor]:
    """The current memory client's cursor cache, or the process cache (Kai)."""
    return owned_state("last_extraction", dict, _last_extraction)


def pop_priority_candidates(chat_ctx: ChatContext) -> list[uuid.UUID] | None:
    """Take the chat's reconsolidation candidates for the extraction about to run."""
    return _priming_registry().pop((chat_ctx.venue, chat_ctx.chat_id, chat_ctx.topic_id), None)


def schedule_consolidation(
    chat_ctx: ChatContext,
) -> None:
    """Schedule a consolidation after CONSOLIDATION_DELAY_SECONDS of silence.

    Calling again cancels the pending timer and restarts it.
    """
    key = (chat_ctx.venue, chat_ctx.chat_id, chat_ctx.topic_id)
    owner = current_owner()
    if owner is not None and not owner.accepts_work():
        logger.debug("Consolidation timer not scheduled: its memory client is closing")
        return
    timers = _timers()

    existing = timers.get(key)
    if existing and not existing.done():
        existing.cancel()
        logger.debug(
            "Consolidation timer reset for venue=%s, chat_id=%s, topic_id=%s",
            chat_ctx.venue,
            chat_ctx.chat_id,
            chat_ctx.topic_id,
        )

    task = asyncio.create_task(
        _consolidation_timer(chat_ctx),
        name=f"consolidation-{chat_ctx.venue}-{chat_ctx.chat_id}-{chat_ctx.topic_id}",
    )
    task.add_done_callback(_log_task_exception)
    adopt_if_owned(task)
    timers[key] = task


def _log_task_exception(t: asyncio.Task) -> None:  # type: ignore[type-arg]
    """Surface unhandled exceptions from fire-and-forget extraction tasks."""
    if not t.cancelled() and t.exception():
        logger.error("Extraction task failed: %s", t.exception())


async def _consolidation_timer(
    chat_ctx: ChatContext,
) -> None:
    """Wait for silence period, then run extraction."""
    key = (chat_ctx.venue, chat_ctx.chat_id, chat_ctx.topic_id)
    current_task = asyncio.current_task()
    try:
        await asyncio.sleep(CONSOLIDATION_DELAY_SECONDS)
    except asyncio.CancelledError:
        return
    finally:
        # Only remove the mapping if we're STILL the registered task —
        # schedule_consolidation() may have replaced us while we were cancelled.
        timers = _timers()
        if timers.get(key) is current_task:
            timers.pop(key, None)

    await _run_extraction(chat_ctx)


def _prune_last_extraction() -> None:
    """Drop _last_extraction entries whose last message is older than LAST_EXTRACTION_TTL_DAYS."""
    cutoff = utc_now() - timedelta(days=LAST_EXTRACTION_TTL_DAYS)
    cursors, registry = _cursors(), _priming_registry()
    stale = [k for k, v in cursors.items() if v.created_at < cutoff]
    for k in stale:
        cursors.pop(k, None)
    # Also prune retrieved traces for expired chats
    for k in stale:
        registry.pop(k, None)


async def extract_now(chat_ctx: ChatContext) -> dict[str, Any] | None:
    """One L2 run for the chat now, leaving no timer behind (``MemoryClient.extract``).

    The cursor is restored from, and committed with, the chat's consolidation runs, as
    on Kai's timer path. Returns the run's stats, or None when another run was in
    progress, capture contention deferred it without advancing its cursor, or it
    failed (recorded failed and logged).
    """
    return await _run_extraction(chat_ctx, reschedule=False)


async def _run_extraction(
    chat_ctx: ChatContext,
    *,
    reschedule: bool = True,
) -> dict[str, Any] | None:
    """Execute Level 2 extraction if no other run is in progress for this chat.

    Returns completed stats, or None when skipped, deferred by capture contention,
    or failed. Deferrals preserve the cursor and failure budget. ``reschedule``
    schedules the retry or backlog timer; Kai's timer path keeps it on.
    """
    _prune_last_extraction()
    chat_id = chat_ctx.chat_id
    topic_id = chat_ctx.topic_id
    venue = chat_ctx.venue
    key = (venue, chat_id, topic_id)
    # run_metadata is structured as {"chat": {...}, "cursor": {...}, "stats": {...}}
    # so we can distinguish queryable chat identity from writable stats. Venue is
    # part of chat identity so each venue's runs are isolated. Historical rows
    # written before A1's backfill are stamped with venue='telegram' so the
    # contains() filter still matches them.
    run_filter = chat_filter(chat_ctx)
    completed_stats: dict[str, Any] | None = None
    retry_after_failure = False
    deferred = False

    async with session_scope() as session:
        running = await session.execute(
            select(ConsolidationRun).where(
                ConsolidationRun.level == 2,
                ConsolidationRun.status == "running",
                ConsolidationRun.run_metadata.contains(run_filter),
            )
        )
        stale_row = running.scalar_one_or_none()
        if stale_row:
            age_minutes = (utc_now() - stale_row.started_at).total_seconds() / 60
            if age_minutes < STALE_RUN_MINUTES:
                logger.info(
                    "Skipping L2 extraction — run in progress for venue=%s, chat_id=%s",
                    venue,
                    chat_id,
                )
                return None
            logger.warning("Marking stale L2 run %s as failed (age=%.0f min)", stale_row.id, age_minutes)
            await session.execute(
                update(ConsolidationRun)
                .where(ConsolidationRun.id == stale_row.id)
                .values(status="failed", completed_at=utc_now(), error_message="stale: orphaned run")
            )

        # Restore order: this process's cache, then the latest completed run's
        # cursor (or, for a legacy row without one, its completed_at, once).
        cursor = _cursors().get(key)
        if cursor is None:
            cursor = await restore_cursor(session, chat_ctx)
        start_cursor = cursor.to_metadata() if cursor is not None else None
        failed_streak, store_streak, failed_page = await failure_streak(
            session, chat_ctx, cursor, MAX_FAILED_RUNS_AT_CURSOR + 1
        )

        run = ConsolidationRun(level=2, status="running", run_metadata=run_filter)
        session.add(run)
        await session.flush()

        from kai_memory.extraction.memory_extraction import ExtractionService, ExtractionStoreFailure
        from kai_memory.procedural.rule_capture import ProceduralCaptureBusy

        try:
            priority_candidates = pop_priority_candidates(chat_ctx)

            svc = ExtractionService(session)
            # The savepoint makes the run all-or-nothing: a failure rolls back this
            # run's writes, so re-reading the same window next time cannot duplicate
            # them, and the failed status below still commits.
            async with session.begin_nested():
                stats, cursor = await svc.extract_after_cursor(
                    chat_ctx,
                    cursor,
                    priority_contradiction_candidates=priority_candidates,
                    skip_page_at=failed_page if store_streak >= MAX_FAILED_RUNS_AT_CURSOR else None,
                )
                # Extraction logs and continues past some failures. A swallowed DB
                # error leaves the transaction aborted, and a RELEASE that fails
                # there cannot be rolled back to the savepoint. Probing here raises
                # inside the block instead, so the run fails without a cursor.
                await session.execute(select(1))

            run.status = "completed"
            run.completed_at = utc_now()
            run.traces_processed = stats.get("traces_created", 0)
            # Preserve the chat identity so the contains() query still matches
            # completed runs. The cursor commits with this run's extraction writes;
            # "cursor": None is a real cursor (nothing consumed yet), not a legacy row.
            run.run_metadata = {
                **run_filter,
                "cursor": cursor.to_metadata() if cursor is not None else None,
                "stats": stats,
            }

            completed_stats = stats

            logger.info(
                "L2 consolidation complete: venue=%s, chat_id=%s, topic_id=%s, traces=%d, entities=%d, "
                "messages_read=%d, pages_read=%d, cursor_advanced=%s, backlog_remaining=%s",
                venue,
                chat_id,
                topic_id,
                stats.get("traces_created", 0),
                stats.get("entities_found", 0),
                stats.get("messages_read", 0),
                stats.get("pages_read", 0),
                stats.get("cursor_advanced", False),
                stats.get("backlog_remaining", False),
            )

        except ProceduralCaptureBusy:
            # Contention is not a failed page: retain the cursor and retry budget.
            run.status = "deferred"
            run.completed_at = utc_now()
            run.run_metadata = {**run_filter, "start_cursor": start_cursor, "deferred_reason": "capture_busy"}
            track_retrieved_traces(chat_ctx, priority_candidates or [])
            deferred = True
            logger.info("L2 consolidation deferred reason=capture_busy")
        except Exception as e:
            logger.error("L2 consolidation failed: %s", e, exc_info=True)
            run.status = "failed"
            run.completed_at = utc_now()
            run.error_message = str(e)[:500]
            # The start cursor bounds retries of this window; the failing page's
            # cursor counts repeated item-write failures on that page.
            run.run_metadata = {**run_filter, "start_cursor": start_cursor, "store_failed": False}
            if isinstance(e, ExtractionStoreFailure):
                page_cursor = e.page_cursor.to_metadata() if e.page_cursor is not None else None
                run.run_metadata |= {"store_failed": True, "failed_page_cursor": page_cursor}
            retry_after_failure = failed_streak < MAX_FAILED_RUNS_AT_CURSOR

    # Only after the commit: a cached cursor ahead of the database would skip
    # messages whose extraction writes never landed.
    if completed_stats is None:
        if (deferred or retry_after_failure) and reschedule:
            # Contention uses the ordinary delay without spending failure retries;
            # actual failures remain bounded by MAX_FAILED_RUNS_AT_CURSOR.
            schedule_consolidation(chat_ctx)
        return None
    if cursor is not None:
        _cursors()[key] = cursor
    if completed_stats.get("backlog_remaining") and reschedule:
        # The per-run page cap left messages above the cursor: drain them on the
        # next timer instead of waiting for the next conversation.
        schedule_consolidation(chat_ctx)
    return completed_stats


def get_pending_timer_count() -> int:
    """Return number of active consolidation timers (for monitoring)."""
    return sum(1 for t in _pending_timers.values() if not t.done())


def cancel_all_timers() -> None:
    """Cancel all pending timers (for shutdown)."""
    for task in _pending_timers.values():
        if not task.done():
            task.cancel()
    _pending_timers.clear()
