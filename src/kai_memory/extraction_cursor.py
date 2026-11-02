"""L2 extraction cursor: the last conversation message extraction consumed (DB-794).

L2 used to decide "what has not been read yet" from a clock: the next lower bound
was the run's completion time, and the reader took ``created_at > since`` once,
``LIMIT 200``. Both lost messages silently — rows 201+ of a backlog, and any row
created between the read and the completion stamp, fell below the next bound.

The bound is now a cursor ``(created_at, id)`` of the last message read and
processed, whatever extraction kept from it. Reads are keyset pages in
``(created_at, id)`` order, so equal timestamps tie-break on ``id`` and no row is
skipped or read twice. The cursor is persisted on the completed L2 run's
``run_metadata["cursor"]``, next to the ``chat`` identity, and commits together
with the extraction writes (contract: backend-docs
``ai-workforce/plans/design/2026-10-09-memory-boundary-map-and-evaluation-contract.md``
§5.1).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kai_memory.storage.models.consolidation_run import ConsolidationRun
from kai_memory import ports

if TYPE_CHECKING:
    from kai_memory.ports import ChatContext

# One page is one extraction batch (one Haiku call). The run reads at most
# MAX_PAGES_PER_RUN pages; the rest stays above the cursor for the next run.
PAGE_SIZE = 200
MAX_PAGES_PER_RUN = 5


class HistoryMessage(Protocol):
    """What memory reads from a stored conversation message (the history port's rows)."""

    id: uuid.UUID
    created_at: datetime


@dataclass(frozen=True)
class ExtractionCursor:
    """Position after the last consumed message.

    ``id is None`` is the legacy bound: a run row written before cursors existed
    only has its ``completed_at``, read as "strictly after this time" exactly as
    the old reader did. The first run that consumes a message replaces it with a
    real ``(created_at, id)`` cursor.
    """

    created_at: datetime
    id: uuid.UUID | None

    @classmethod
    def after(cls, message: HistoryMessage) -> ExtractionCursor:
        return cls(created_at=message.created_at, id=message.id)

    def to_metadata(self) -> dict[str, Any]:
        # UTC, so a cached cursor and one restored from the database serialize alike.
        return {
            "created_at": self.created_at.astimezone(UTC).isoformat(),
            "id": str(self.id) if self.id is not None else None,
        }

    @classmethod
    def from_metadata(cls, raw: dict[str, Any] | None) -> ExtractionCursor | None:
        if not raw:
            return None
        raw_id = raw.get("id")
        return cls(
            created_at=datetime.fromisoformat(raw["created_at"]),
            id=uuid.UUID(raw_id) if raw_id else None,
        )


def chat_filter(chat_ctx: ChatContext) -> dict[str, Any]:
    """The run row's queryable chat identity, matched with JSONB ``contains()``."""
    return {"chat": {"venue": chat_ctx.venue, "chat_id": chat_ctx.chat_id, "topic_id": chat_ctx.topic_id}}


async def read_messages_page(
    session: AsyncSession,
    chat_ctx: ChatContext,
    after: ExtractionCursor | None,
    limit: int,
) -> list[Any]:
    """Messages of this chat after ``after``, oldest first, at most ``limit``.

    The conversation-history port (``ports.read_conversation_page``, DB-843) owns the
    read: keyset pages in ``(created_at, id)`` order, so equal timestamps tie-break on
    ``id`` and no row is skipped or read twice; ``after.id is None`` is the legacy
    bound "strictly after this time"; headless venues (``chat_id=None``) match their
    own rows.

    Known residual: ``created_at`` is stamped at flush, but a row is visible only
    at commit. A writer whose transaction is still open (the legacy event path
    holds one across the whole turn) can commit a row below a cursor that has
    already passed it. Closing that needs the writer and this reader to share a
    per-chat lock, a host-side change; the old completion-clock bound lost a
    superset of these rows.
    """
    rows: list[Any] = await ports.read_conversation_page(session, chat_ctx, after, limit)
    return rows


async def restore_cursor(session: AsyncSession, chat_ctx: ChatContext) -> ExtractionCursor | None:
    """The cursor of the latest completed L2 run for this chat.

    A row with a ``cursor`` key is authoritative, even when it is ``None`` (the run
    found no messages yet). A legacy row without the key falls back to its
    ``completed_at``, once: the run that follows writes the key. No completed run
    at all means extraction starts from the first message.
    """
    result = await session.execute(
        select(ConsolidationRun)
        .where(
            ConsolidationRun.level == 2,
            ConsolidationRun.status == "completed",
            ConsolidationRun.run_metadata.contains(chat_filter(chat_ctx)),
        )
        .order_by(ConsolidationRun.completed_at.desc().nulls_last())
        .limit(1)
    )
    last_run = result.scalar_one_or_none()
    if last_run is None:
        return None
    metadata = last_run.run_metadata or {}
    if "cursor" in metadata:
        return ExtractionCursor.from_metadata(metadata["cursor"])
    if last_run.completed_at is None:
        return None
    return ExtractionCursor(created_at=last_run.completed_at, id=None)


def page_key(cursor: ExtractionCursor | None) -> str:
    """Identity of the page that starts at ``cursor`` (``None``: the first message)."""
    return json.dumps(cursor.to_metadata() if cursor is not None else None, sort_keys=True)


async def failure_streak(
    session: AsyncSession, chat_ctx: ChatContext, cursor: ExtractionCursor | None, limit: int
) -> tuple[int, int, str | None]:
    """Recent consecutive failed L2 runs for this chat, newest first.

    Returns ``(failed, store_failed, failed_page)``. ``failed`` counts failures that
    started at ``cursor`` (it bounds timer retries). ``store_failed`` counts leading
    item-write failures on one page, ``failed_page`` (its ``page_key``). Reads at most
    ``limit`` runs; a completed run ends both streaks.
    """
    result = await session.execute(
        select(ConsolidationRun)
        .where(
            ConsolidationRun.level == 2,
            ConsolidationRun.status.in_(("completed", "failed")),
            ConsolidationRun.run_metadata.contains(chat_filter(chat_ctx)),
        )
        .order_by(ConsolidationRun.started_at.desc())
        .limit(limit)
    )
    runs = [(run.status, run.run_metadata or {}) for run in result.scalars()]
    start = cursor.to_metadata() if cursor is not None else None
    failed = 0
    for status, metadata in runs:
        if status != "failed" or "start_cursor" not in metadata or metadata["start_cursor"] != start:
            break
        failed += 1
    store_failed, failed_page = 0, None
    for status, metadata in runs:
        if status != "failed" or not metadata.get("store_failed") or "failed_page_cursor" not in metadata:
            break
        key = page_key(ExtractionCursor.from_metadata(metadata["failed_page_cursor"]))
        if failed_page is not None and key != failed_page:
            break
        failed_page = key
        store_failed += 1
    return failed, store_failed, failed_page
